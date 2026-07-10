import asyncio
import hashlib
import logging
import re
import statistics
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 잡별 라인 메타(발음/번역) 임시 저장소 — BackgroundTasks가 같은 프로세스에서 돌므로
# 인메모리로 충분하다 (프로세스가 죽으면 잡 실행 자체가 사라지므로 내구성 손해도 없음).
_PENDING_LINE_META: dict[str, list[dict[str, Any]]] = {}
# 강제 재생성 잡 — 동일 (audio_hash, lyrics_hash) 재사용을 건너뛰고 정렬을 다시 돌린다
_PENDING_FORCE: set[str] = set()


def stash_line_meta(job_id: str, line_meta: list[dict[str, Any]]) -> None:
    _PENDING_LINE_META[job_id] = line_meta


# 잡별 가사 출처 표기 (예: 보카로 가사 위키) — 완성된 싱크에 함께 저장된다
_PENDING_ATTRIBUTION: dict[str, dict[str, Any]] = {}


def stash_attribution(job_id: str, attribution: dict[str, Any]) -> None:
    _PENDING_ATTRIBUTION[job_id] = attribution


def stash_force(job_id: str) -> None:
    _PENDING_FORCE.add(job_id)


def _normalize_line(s: str) -> str:
    return " ".join(s.split())


def merge_line_meta(timestamps: list[dict[str, Any]], line_meta: list[dict[str, Any]]) -> int:
    """세그먼트에 발음/번역을 라인 텍스트 매칭으로 병합. 병합된 세그먼트 수를 반환."""
    by_text: dict[str, dict[str, Any]] = {}
    for m in line_meta:
        t = _normalize_line(m.get("text", "") or "")
        if t and t not in by_text:
            by_text[t] = m

    merged = 0
    for seg in timestamps:
        m = by_text.get(_normalize_line(seg.get("text", "") or ""))
        if not m:
            continue
        if m.get("pronunciation"):
            seg["pronunciation"] = m["pronunciation"]
            _attach_pron_segments(seg)
        if m.get("translation"):
            seg["translation"] = m["translation"]
        merged += 1
    return merged


def _attach_pron_segments(seg: dict[str, Any]) -> None:
    """발음 음절별 타이밍 산출 — 정렬된 글자 타이밍 + 모라 분해 + DP 매칭.

    전사 모델을 다시 돌리지 않는다 (기존 CTC 글자 타이밍을 모라 수로 내부 분할).
    품질 미달/실패 시 필드를 남기지 않아 클라이언트가 그라데이션으로 폴백한다.
    """
    pron = seg.get("pronunciation")
    words = seg.get("words")
    if not pron or not words:
        seg.pop("pron_segments", None)
        return
    try:
        from everyric2.text.reading import pron_segments_for_line

        char_spans = [
            (w.get("word", ""), float(w.get("start", 0.0)), float(w.get("end", 0.0)))
            for w in words
        ]
        segments = pron_segments_for_line(char_spans, seg.get("text", "") or "", pron)
        if segments:
            seg["pron_segments"] = segments
        else:
            seg.pop("pron_segments", None)
    except Exception:
        logger.exception("pron_segments computation failed; falling back to gradient fill")
        seg.pop("pron_segments", None)


def compute_audio_hash(file_path: Path) -> str:
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()


async def process_job(job_id: str) -> None:
    from everyric2.server.db.connection import get_session
    from everyric2.server.db.repository import JobRepository, SyncRepository, hash_lyrics

    async with get_session() as session:
        job_repo = JobRepository(session)
        job = await job_repo.get_by_id(job_id)

        if not job:
            logger.error(f"Job not found: {job_id}")
            return

        await job_repo.update_status(job_id, "processing", progress=10)

    try:
        lyrics_hash_value = hash_lyrics(job.lyrics)

        download_result = await asyncio.get_event_loop().run_in_executor(
            None, _download_and_hash, job.video_id
        )
        audio_hash = download_result["audio_hash"]
        audio_path = download_result["audio_path"]
        await _set_progress(job_id, 35)  # 다운로드 완료

        forced = job_id in _PENDING_FORCE
        _PENDING_FORCE.discard(job_id)

        async with get_session() as session:
            sync_repo = SyncRepository(session)
            existing = await sync_repo.get_by_audio_and_lyrics_hash(audio_hash, lyrics_hash_value)
            if existing and not forced:
                meta = _PENDING_LINE_META.pop(job_id, None)
                attr = _PENDING_ATTRIBUTION.pop(job_id, None)
                updated = dict(existing.timestamps)
                changed = False
                if meta:
                    segs = [dict(s) for s in updated.get("segments", [])]
                    if merge_line_meta(segs, meta):
                        updated["segments"] = segs
                        changed = True
                if attr is not None:
                    updated["attribution"] = attr
                    changed = True
                if changed:
                    # JSON 컬럼은 재할당해야 변경이 감지된다
                    existing.timestamps = updated
                job_repo = JobRepository(session)
                await job_repo.update_status(
                    job_id, "completed", progress=100, result_id=existing.id
                )
                logger.info(f"Job {job_id} reused existing sync (audio_hash match)")
                Path(audio_path).unlink(missing_ok=True)
                return

        # 정렬(분리+CTC+멜로디)은 수십 초 걸리는 단일 블록 — 진행률이 멈춰 보이지 않게
        # 45%에서 시작해 85%까지 천천히 차오르는 티커를 함께 돌린다
        await _set_progress(job_id, 45)
        ticker = asyncio.create_task(_tick_progress(job_id, start=45, cap=85))
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, _run_alignment, audio_path, job.lyrics, job.language
            )
        finally:
            ticker.cancel()
        await _set_progress(job_id, 90)  # 정렬 완료, 저장 중

        meta = _PENDING_LINE_META.pop(job_id, None)
        if meta:
            merged = merge_line_meta(result["timestamps"], meta)
            logger.info(f"Line meta merged on {merged} segments")

        async with get_session() as session:
            job_repo = JobRepository(session)
            sync_repo = SyncRepository(session)

            sync_result = await sync_repo.create(
                video_id=job.video_id,
                lyrics_hash=lyrics_hash_value,
                timestamps=result["timestamps"],
                language=result.get("language"),
                engine="ctc",
                quality_score=result.get("quality_score"),
                audio_hash=audio_hash,
                extra=_build_extra(result, _PENDING_ATTRIBUTION.pop(job_id, None)),
            )

            await job_repo.update_status(
                job_id, "completed", progress=100, result_id=sync_result.id
            )
            logger.info(f"Job completed: {job_id}")

    except Exception as e:
        logger.exception(f"Job failed: {job_id}")
        _PENDING_LINE_META.pop(job_id, None)
        _PENDING_ATTRIBUTION.pop(job_id, None)
        _PENDING_FORCE.discard(job_id)
        async with get_session() as session:
            job_repo = JobRepository(session)
            await job_repo.update_status(job_id, "failed", error=str(e))


async def _set_progress(job_id: str, progress: int) -> None:
    from everyric2.server.db.connection import get_session
    from everyric2.server.db.repository import JobRepository

    async with get_session() as session:
        await JobRepository(session).update_status(job_id, "processing", progress=progress)


async def _tick_progress(job_id: str, start: int, cap: int, interval: float = 4.0) -> None:
    """긴 단계 동안 진행률을 cap까지 천천히 올린다 — 취소되면 그대로 멈춘다."""
    progress = start
    try:
        while progress < cap:
            await asyncio.sleep(interval)
            progress = min(cap, progress + 4)
            await _set_progress(job_id, progress)
    except asyncio.CancelledError:
        pass


def _download_and_hash(video_id: str) -> dict:
    from everyric2.audio.downloader import YouTubeDownloader

    downloader = YouTubeDownloader()
    youtube_url = f"https://www.youtube.com/watch?v={video_id}"
    dl_result = downloader.download(youtube_url)
    audio_hash = compute_audio_hash(dl_result.audio_path)

    return {
        "audio_path": str(dl_result.audio_path),
        "audio_hash": audio_hash,
    }


def _build_extra(result: dict[str, Any], attribution: dict[str, Any] | None) -> dict[str, Any] | None:
    """싱크 JSON의 segments 밖 부가정보(디버그 메타, 출처 표기, 템포) 조립."""
    extra: dict[str, Any] = {}
    if result.get("debug"):
        extra["debug"] = result["debug"]
    if result.get("tempo"):
        extra["tempo"] = result["tempo"]
    if attribution is not None:
        extra["attribution"] = attribution
    return extra or None


def _estimate_tempo(audio) -> dict[str, Any] | None:
    """librosa로 BPM·첫 비트 시각 추정 — 가라오케 레인의 박자/마디 격자용.

    보컬로이드 곡은 대부분 고정 BPM이라 (bpm, beat_offset)만으로 전 곡 격자를
    재구성할 수 있다. 실패는 치명적이지 않으므로 None으로 조용히 폴백.
    """
    try:
        import librosa
        import numpy as np

        y = np.asarray(audio.waveform, dtype=np.float32)
        sr = int(audio.sample_rate)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, trim=False)
        bpm = float(np.atleast_1d(tempo)[0])
        if not (30.0 <= bpm <= 300.0) or len(beats) < 8:
            return None
        beat_times = librosa.frames_to_time(beats, sr=sr)
        # 첫 비트 위상 — 비트 간격의 중앙값으로 격자를 안정화
        interval = float(np.median(np.diff(beat_times)))
        if interval > 0:
            bpm = 60.0 / interval
        return {"bpm": round(bpm, 2), "beat_offset": round(float(beat_times[0]), 3)}
    except Exception:
        logger.exception("Tempo estimation failed; lane falls back to seconds grid")
        return None


def _separate_vocals(audio):
    """demucs 보컬 분리 (실패/미설치 시 None) — VAD 클램프와 멜로디 f0가 공유한다."""
    try:
        import torch

        from everyric2.audio.separator import VocalSeparator

        separator = VocalSeparator()
        if not separator.is_available():
            logger.info("demucs not installed; skipping VAD clamp / using mix for melody")
            return None
        return separator.separate(audio, use_gpu=torch.cuda.is_available()).vocals
    except Exception:
        logger.exception("Vocal separation failed; skipping VAD clamp")
        return None


def _repeat_key(text: str) -> str:
    """반복행 판정용 키 — 공백/기호를 지우고 대소문자를 무시한 텍스트."""
    return re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE).casefold()


def _clamp_repeated_outliers(results, clamped: set[int]) -> None:
    """같은 가사가 3번 이상 반복될 때, 형제 라인 duration 중앙값 대비 병적으로 긴
    라인을 중앙값 길이로 잘라낸다(시작 유지, end = start + 중앙값).

    CTC가 같은 훅을 반복해 부를 때 글자를 특정 렌디션에 몰아 흩뿌려, 같은 텍스트의
    다른 반복 라인은 ~2초인데 한 라인만 7초 outlier로 늘어나는 케이스를 잡는다.
    형제가 2개 이하이거나 중앙값 자체가 비정상(<0.5s)이면 건드리지 않는다.
    """
    groups: dict[str, list[int]] = {}
    for i, r in enumerate(results):
        key = _repeat_key(r.text)
        if key:
            groups.setdefault(key, []).append(i)
    for idxs in groups.values():
        if len(idxs) < 3:
            continue
        median = statistics.median(results[i].end_time - results[i].start_time for i in idxs)
        if median < 0.5:
            continue
        limit = max(median * 2.5, 4.0)
        for i in idxs:
            if i in clamped:
                continue  # 기존 규칙이 이미 처리한 라인은 그대로 둔다
            r = results[i]
            if r.end_time - r.start_time > limit:
                r.end_time = r.start_time + median
                clamped.add(i)


def _pull_post_interlude_starts(results, vad_result, clamped: set[int]) -> None:
    """긴 간주(직전 라인 end와 8초 이상 벌어짐) 뒤 첫 라인의 시작이 실제 보컬 시작보다
    늦게 잡히면, 라인이 속한 가창 블록의 시작으로 라인 start를 당긴다.

    앵커는 "간주 이후 첫 리전"이 아니라 **라인과 겹치는 첫 리전에서 뒤로(≤2s 간격)
    이어지는 리전 체인의 시작**이다 — 간주 초입의 고립된 잔향/애드립 리전(체인 밖)에
    끌려가 3배 가드에 걸리는 오탐을 막는다 (熱異常 실측: 40초 간주 초입 0.6초 잔향).
    end는 유지하고, 당긴 결과 duration이 원래의 3배를 넘으면 오탐으로 보고 건너뛴다.
    """
    for i in range(1, len(results)):
        r = results[i]
        prev_end = results[i - 1].end_time
        if r.start_time - prev_end < 8.0:
            continue
        # 간주~라인 구간의 발성 리전 (시간순)
        regions = sorted(
            (reg for reg in vad_result.regions if reg.end > prev_end and reg.start < r.end_time),
            key=lambda reg: reg.start,
        )
        j = next((k for k, reg in enumerate(regions) if reg.end > r.start_time), None)
        if j is None:
            continue
        # 라인과 겹치는 첫 리전에서 뒤로 이어지는 가창 블록의 시작까지 역추적
        while j > 0 and regions[j].start - regions[j - 1].end <= 2.0:
            j -= 1
        anchor = regions[j].start
        if anchor > r.start_time - 1.5:
            continue
        new_start = anchor - 0.15
        orig_dur = r.end_time - r.start_time
        if r.end_time - new_start > 3.0 * orig_dur:
            continue  # 3배 초과로 늘어나면 오탐 — 적용하지 않는다
        r.start_time = new_start
        clamped.add(i)


def _clamp_stretched_lines(results, vad_result):
    """가사에 없는 반복 가창(라인 내부 퍼짐)으로 병적으로 길어진 라인을 잘라낸다.

    CTC는 같은 가사가 여러 번 불리면 글자들을 여러 렌디션에 걸쳐 흩뿌릴 수 있다
    (라인 사이 star로는 못 잡는 케이스). 지속 8초 초과 + 발성 커버리지 50% 미만인
    라인만 첫 발성 구간 끝으로 클램프한다 — 정상 라인은 건드리지 않는다.
    여기에 더해 반복행 outlier 클램프와 간주 후 시작 앵커 당기기를 함께 적용한다.
    반환: (results, 클램프된 라인 인덱스 집합)
    """
    clamped: set[int] = set()
    for i, r in enumerate(results):
        dur = r.end_time - r.start_time
        if dur <= 8.0:
            continue
        regions = [
            reg for reg in vad_result.regions if reg.end > r.start_time and reg.start < r.end_time
        ]
        if not regions:
            continue
        vocal = sum(min(reg.end, r.end_time) - max(reg.start, r.start_time) for reg in regions)
        if vocal / dur >= 0.5:
            continue
        new_end = min(r.end_time, max(regions[0].end + 0.3, r.start_time + 1.5))
        if new_end < r.end_time:
            r.end_time = new_end
            clamped.add(i)
    # 반복행 형제 대비 outlier로 늘어난 라인 + 간주 뒤 늦게 시작한 라인도 보정
    _clamp_repeated_outliers(results, clamped)
    _pull_post_interlude_starts(results, vad_result, clamped)
    if clamped:
        logger.info(f"Clamped {len(clamped)} pathologically stretched lines")
    return results, clamped


def _run_alignment(audio_path: str, lyrics: str, language: str | None) -> dict:
    from everyric2.alignment.factory import EngineFactory
    from everyric2.audio.loader import AudioLoader
    from everyric2.config.settings import get_settings
    from everyric2.inference.prompt import LyricLine

    settings = get_settings()
    loader = AudioLoader()
    audio_path_obj = Path(audio_path)

    try:
        audio = loader.load(audio_path_obj)
        lyric_lines = LyricLine.from_text(lyrics)

        engine = EngineFactory.get_engine("ctc", settings.alignment)
        if not engine.is_available():
            raise RuntimeError("CTC engine not available")

        results = engine.align(audio, lyric_lines, language=language or "auto")

        # 보컬 스템 1회 분리 → VAD로 라인 경계 보정(가사에 없는 추임새/간주로 늘어진
        # 라인을 실제 발성 구간으로 되돌림) + 아래 멜로디 f0 추출에 재사용
        vocals = _separate_vocals(audio) if settings.melody.separate_vocals else None
        vad_regions: list[tuple[float, float]] | None = None
        clamped_lines: set[int] = set()
        if vocals is not None:
            try:
                from everyric2.alignment.timing_postprocess import TimingPostProcessor
                from everyric2.audio.vad import VocalActivityDetector

                vad_result = VocalActivityDetector().detect(vocals)
                # extend_to_vocal은 끄는다: 가사에 없는 반복 가창/애드립도 "보컬 활동"이라
                # 라인을 그쪽으로 늘려버린다 (star 토큰이 흡수해 둔 구간을 도로 끌어안는 역효과)
                pp = TimingPostProcessor(settings.segmentation, extend_to_vocal=False).process(
                    results, vad_result, "line"
                )
                results, clamped_lines = _clamp_stretched_lines(pp.results, vad_result)
                vad_regions = [(round(reg.start, 2), round(reg.end, 2)) for reg in vad_result.regions]
                logger.info(f"Timing post-process: {pp.stats}")
            except Exception:
                logger.exception("VAD timing post-process failed; using raw alignment")

        timestamps = []
        for i, r in enumerate(results):
            seg = {
                "text": r.text,
                "start": r.start_time,
                "end": r.end_time,
            }
            if r.confidence is not None:
                seg["confidence"] = r.confidence
            if r.word_segments:
                seg["words"] = [
                    {"word": w.word, "start": w.start, "end": w.end, "confidence": w.confidence}
                    for w in r.word_segments
                ]
            if vad_regions is not None:
                # 라인 구간 중 실제 발성 비율 + 클램프 여부 — 확장 디버그 스트립용
                dur = max(0.001, r.end_time - r.start_time)
                vocal = sum(
                    max(0.0, min(e, r.end_time) - max(s, r.start_time)) for s, e in vad_regions
                )
                seg["debug"] = {
                    "active_ratio": round(vocal / dur, 2),
                    "clamped": i in clamped_lines,
                }
            timestamps.append(seg)

        # 가라오케용 음정(MIDI 노트) 주석 — 실패해도 싱크 생성 자체는 계속한다
        if settings.melody.enabled:
            try:
                from everyric2.melody.extractor import MelodyExtractor

                extractor = MelodyExtractor(settings.melody)
                if extractor.is_available():
                    # vocal_regions는 넘기지 않는다 — extractor가 라인 스팬 합집합으로
                    # 자체 마스킹한다 (VAD 마스크는 조용한 벌스 노트를 소실시킴)
                    annotated = extractor.annotate_timestamps(audio, timestamps, vocals=vocals)
                    logger.info(f"Melody notes annotated on {annotated} spans")
                else:
                    logger.warning("Melody enabled but torchfcpe is not installed; skipping")
            except Exception:
                logger.exception("Melody extraction failed; continuing without notes")

        avg_confidence = None
        confidences = [t.get("confidence") for t in timestamps if t.get("confidence") is not None]
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)

        detected_lang = language
        if hasattr(engine, "_current_lang"):
            detected_lang = engine._current_lang

        # 곡 단위 디버그 메타 — star가 흡수한 구간(가사 밖 가창)과 VAD 발성 구간
        star_spans = [list(s) for s in getattr(engine, "_last_star_spans", [])]
        debug_meta = {
            "star_spans": star_spans,
            "vad_regions": [list(v) for v in vad_regions] if vad_regions is not None else None,
        }

        return {
            "timestamps": timestamps,
            "language": detected_lang,
            "quality_score": avg_confidence,
            "debug": debug_meta,
            # 가라오케 레인의 마디 단위 고정 창·비트 격자용 — 실패해도 None으로 계속
            "tempo": _estimate_tempo(audio),
        }
    finally:
        audio_path_obj.unlink(missing_ok=True)
