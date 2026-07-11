import asyncio
import hashlib
import logging
import math
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
    이미 pron_segments가 있으면(독음 정렬 경로 산출값 등) DP 근사로 덮어쓰지 않는다 —
    캐시 재사용 시 라인 메타 재병합이 정확한 정렬 스팬을 훼손하는 것을 막는다.
    """
    if seg.get("pron_segments"):
        return
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
        # 라인 메타(발음)를 정렬 단계로 넘겨 독음(ko) 정렬 경로 판단에 쓴다 (아직 pop 안 함)
        pending_meta = _PENDING_LINE_META.get(job_id)
        ticker = asyncio.create_task(_tick_progress(job_id, start=45, cap=85))
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, _run_alignment, audio_path, job.lyrics, job.language, pending_meta
            )
        finally:
            ticker.cancel()
        await _set_progress(job_id, 90)  # 정렬 완료, 저장 중

        meta = _PENDING_LINE_META.pop(job_id, None)
        # 독음 정렬 경로는 발음/번역/pron_segments를 이미 세그먼트에 붙였으므로 재병합 생략
        if meta and result.get("alignment_text") != "pronunciation":
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


def _extend_phrase_final_tails(results, vad_result, clamped: set[int]) -> None:
    """소절 끝(뒤에 0.3초 이상 갭) 라인의 끝을 실제 발성 끝까지 연장한다.

    CTC는 마지막 음절을 온셋에서 끊어 늘임음(held note)의 감쇠를 따라가지 않는다 —
    SRT/VAD 이중 실측으로 phrase-final 라인의 86~100%가 median 0.4~0.66초 일찍
    끝남이 확인됨. 라인 끝이 속한 VAD 리전의 끝까지(단 다음 라인 시작 -0.05초,
    캡 이내) 라인과 마지막 글자의 end를 함께 연장한다.
    캡은 적응형: 리전 꼬리가 3초 이내이고 다음 라인이 8초 안에 이어지면 진짜
    늘임음으로 보고 +2.5초까지, 그 밖(꼬리>3초 병합 리전 의심, 또는 간주 직전
    라인)은 +1.5초로 보수적으로 자른다. 간주 직전은 리전 끝이 잔향·악기 유입으로
    실제 발성보다 늦게 잡히는 데다 재실행 간 ±1초 가까이 흔들려서(커버 실측
    cue#37: 리전 꼬리 1.5→2.3s 변동으로 과연장 악화) 꼬리 길이만으로는 진짜
    늘임음과 구분할 수 없다 — 잔존 3건(사비 중간, 다음 줄 갭 ~3초)과 과연장
    1건(간주 앞, 갭 22초)을 가르는 신호는 다음 라인까지의 갭이었다.
    소절 중간(butted) 라인과 이미 클램프로 잘라낸 라인은 건드리지 않는다.
    """
    for i, r in enumerate(results):
        if i in clamped:
            continue
        next_start = results[i + 1].start_time if i + 1 < len(results) else float("inf")
        if next_start - r.end_time <= 0.3:
            continue  # butted — 다음 음절이 바로 이어지는 라인은 그대로
        region = next(
            (reg for reg in vad_result.regions if reg.start <= r.end_time < reg.end), None
        )
        if region is None:
            continue  # 라인 끝이 발성 리전 밖 — 따라갈 꼬리가 없다
        real_tail = region.end - r.end_time <= 3.0 and next_start - r.end_time < 8.0
        cap = 2.5 if real_tail else 1.5
        new_end = min(region.end, next_start - 0.05, r.end_time + cap)
        if new_end <= r.end_time + 0.05:
            continue
        r.end_time = new_end
        if r.word_segments:
            r.word_segments[-1].end = new_end


def _diff_fixes(
    fixes: dict[int, list[str]],
    label: str,
    before: list[tuple[float, float]],
    results,
    tol: float = 0.01,
) -> None:
    """스테이지 전후 타이밍 diff로 어떤 규칙이 어떤 라인을 고쳤는지 라벨링 (디버그용)."""
    for i, r in enumerate(results):
        if abs(r.start_time - before[i][0]) > tol or abs(r.end_time - before[i][1]) > tol:
            labels = fixes.setdefault(i, [])
            if label not in labels:
                labels.append(label)


def _clamp_stretched_lines(results, vad_result, fixes: dict[int, list[str]] | None = None):
    """가사에 없는 반복 가창(라인 내부 퍼짐)으로 병적으로 길어진 라인을 잘라낸다.

    CTC는 같은 가사가 여러 번 불리면 글자들을 여러 렌디션에 걸쳐 흩뿌릴 수 있다
    (라인 사이 star로는 못 잡는 케이스). 지속 8초 초과 + 발성 커버리지 50% 미만인
    라인만 첫 발성 구간 끝으로 클램프한다 — 정상 라인은 건드리지 않는다.
    여기에 더해 반복행 outlier 클램프·간주 후 시작 앵커 당기기·소절 끝 늘임음
    연장을 함께 적용한다. fixes를 넘기면 규칙별 적용 라인을 라벨링한다(디버그).
    반환: (results, 클램프된 라인 인덱스 집합)
    """
    clamped: set[int] = set()
    before = [(r.start_time, r.end_time) for r in results] if fixes is not None else None
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
    if fixes is not None and before is not None:
        _diff_fixes(fixes, "stretch", before, results)
        before = [(r.start_time, r.end_time) for r in results]
    # 반복행 형제 대비 outlier로 늘어난 라인 + 간주 뒤 늦게 시작한 라인도 보정
    _clamp_repeated_outliers(results, clamped)
    if fixes is not None and before is not None:
        _diff_fixes(fixes, "repeat", before, results)
        before = [(r.start_time, r.end_time) for r in results]
    _pull_post_interlude_starts(results, vad_result, clamped)
    if fixes is not None and before is not None:
        _diff_fixes(fixes, "pull", before, results)
        before = [(r.start_time, r.end_time) for r in results]
    # 소절 끝 늘임음은 실제 발성 끝까지 연장 (클램프된 라인 제외)
    _extend_phrase_final_tails(results, vad_result, clamped)
    if fixes is not None and before is not None:
        _diff_fixes(fixes, "tail", before, results)
    if clamped:
        logger.info(f"Clamped {len(clamped)} pathologically stretched lines")
    return results, clamped


def _snap_silence_undershoot(results, vad_result, clamped: set[int]) -> None:
    """무음(간주)에 좌초한 라인을 다음 발성 온셋으로 스냅한다 (독음 정렬 전용).

    독음(ko) 정렬의 유일한 실패 모드: 간주 전후 전이 라인이 실제 가창(간주 이후)보다
    앞선 무음 구간에 배치된다 (熱異常 L94: 실제 165.5s인데 간주 무음 146s로 언더슛).
    라인 창의 발성 커버리지가 매우 낮고(<0.25 — 순수 무음 또는 간주 초입 잔향 blip만
    스침) 다음 발성 리전까지 >=1.5s 벌어져 있으면 그 온셋 직전(-0.15s)으로 당긴다.
    라인 전체가 무음에 갇혔으면(끝이 다음 온셋 이전) 길이를 보존해 통째로 이동한다.

    ``_pull_post_interlude_starts``(늦게 잡힌 시작을 앞으로 당김)와 방향이 겹칠 수 있어
    **_clamp_stretched_lines 이전에** 돌려, 좌초 라인이 먼저 제자리(다음 온셋)를 잡게 한다
    — 그래야 뒤따르는 정상 라인이 간주 후 첫 라인으로 오인돼 도로 당겨지지 않는다.
    커버리지가 조금이라도 있으면(온셋 리드·늘임음 꼬리 포함) 보수적으로 건드리지 않는다.
    """
    regions = sorted(vad_result.regions, key=lambda reg: reg.start)
    if not regions:
        return
    for i, r in enumerate(results):
        if i in clamped:
            continue
        s, e = r.start_time, r.end_time
        dur = max(1e-6, e - s)
        cover = sum(max(0.0, min(reg.end, e) - max(reg.start, s)) for reg in regions) / dur
        if cover >= 0.25:
            continue  # 라인이 발성과 유의미하게 겹침 → 정상 배치, 건드리지 않음
        nxt = next((reg for reg in regions if reg.start >= s), None)
        if nxt is None:
            continue  # 뒤에 발성 없음 (곡 끝 무음) → 스냅할 온셋 없음
        if nxt.start - s < 1.5:
            continue  # 온셋 직전의 짧은 리드타임은 정상
        new_start = nxt.start - 0.15
        next_line_start = results[i + 1].start_time if i + 1 < len(results) else float("inf")
        if new_start >= next_line_start:
            continue  # 다음 라인을 침범 → 오탐, 적용하지 않음
        if e <= nxt.start:
            # 라인 전체가 무음에 갇힘 → 길이 보존하고 통째로 온셋으로 이동(다음 라인 앞까지)
            r.start_time = new_start
            r.end_time = min(new_start + dur, next_line_start)
            if r.word_segments:
                _shift_word_segments(r.word_segments, r.start_time, r.end_time)
        else:
            r.start_time = new_start  # 시작만 무음, 라인이 이미 리전에 걸침 → start만 스냅
        clamped.add(i)


def _shift_word_segments(word_segments, new_start: float, new_end: float) -> None:
    """word_segments를 [new_start, new_end] 구간으로 선형 리스케일(제자리)."""
    if not word_segments:
        return
    old_start = word_segments[0].start
    old_end = word_segments[-1].end
    span = old_end - old_start
    target = new_end - new_start
    if span <= 0:
        n = len(word_segments)
        step = target / n if n else 0.0
        for k, w in enumerate(word_segments):
            w.start = new_start + step * k
            w.end = new_start + step * (k + 1)
        return
    for w in word_segments:
        w.start = new_start + (w.start - old_start) / span * target
        w.end = new_start + (w.end - old_start) / span * target


def _geomean(values: list[float]) -> float | None:
    xs = [v for v in values if v is not None and v > 0]
    if not xs:
        return None
    return math.exp(sum(math.log(v) for v in xs) / len(xs))


def _star_swallowed_vocal(star_spans, vad_regions) -> float:
    """단일 star span이 실제 VAD 발성 구간을 삼킨 최대 겹침 길이(초).

    독음(ko) 정렬이 초고속/간주 구간에서 실패하면, 와일드카드 star(log 1.0)가 실제
    후반 가창을 통째로 흡수하고 그 라인들을 앞으로 압축한다 (VWVtIg5cdDU 실측: star
    한 개가 후반 가창 21s를 삼킴). 다만 이 값만으로는 '실가사 압축(VWV)'과 '가사 없는
    브릿지 정상 흡수(熱異常도 20.7s 삼키지만 배치 정상)'를 못 가른다. 그래서 이건
    비용 게이트(값이 크면 ja와 대조)로만 쓰고, 최종 판정은 간주 이후 발성 구간의
    라인 점유를 ko/ja 비교하는 호출부(_post_interlude_window)가 한다.
    """
    def ov(s, r):
        return max(0.0, min(s[1], r[1]) - max(s[0], r[0]))

    regions = [(reg.start, reg.end) for reg in vad_regions]
    return max((sum(ov(s, r) for r in regions) for s in star_spans), default=0.0)


def _post_interlude_window(vad_regions, min_gap_sec: float) -> tuple[float, float] | None:
    """최대 간주(무음 갭) 이후의 발성 창 [gap_end, last_vocal_end].

    간주는 오디오가 고정하는 구조라 star(정렬마다 위치 변동)보다 안정적인 앵커다.
    연속 VAD 리전 사이 최대 갭이 min_gap_sec 이상이면 그 갭 끝~마지막 발성 끝을
    '간주 이후 창'으로 돌려준다. 큰 간주가 없으면 None(폴백 판단 안 함).
    """
    regions = sorted((reg.start, reg.end) for reg in vad_regions)
    if len(regions) < 2:
        return None
    best_gap, gap_end = 0.0, None
    for (_, e0), (s1, _) in zip(regions, regions[1:]):
        if s1 - e0 > best_gap:
            best_gap, gap_end = s1 - e0, s1
    if gap_end is None or best_gap < min_gap_sec:
        return None
    return (gap_end, regions[-1][1])


def _lines_span_overlap(results, span: tuple[float, float]) -> float:
    """results 라인들이 [span] 구간과 겹친 총 길이(초) — 그 구간의 '라인 점유량'."""
    lo, hi = span
    return sum(max(0.0, min(r.end_time, hi) - max(r.start_time, lo)) for r in results)


def _pron_by_text(line_meta: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    """line_meta를 정규화 텍스트 → 메타 dict로 색인 (merge_line_meta와 동일 규칙)."""
    by_text: dict[str, dict[str, Any]] = {}
    for m in line_meta or []:
        t = _normalize_line(m.get("text", "") or "")
        if t and t not in by_text:
            by_text[t] = m
    return by_text


def _pron_coverage(lyric_lines, by_text: dict[str, dict[str, Any]]) -> float:
    """발음 표기가 붙은 라인 비율 (0~1)."""
    if not lyric_lines:
        return 0.0
    have = 0
    for ln in lyric_lines:
        m = by_text.get(_normalize_line(ln.text))
        if m and (m.get("pronunciation") or "").strip():
            have += 1
    return have / len(lyric_lines)


def _align_with_pronunciation(engine, audio, lyric_lines, by_text: dict[str, dict[str, Any]]):
    """독음(ko) 텍스트로 CTC 정렬 후 원문 라인에 역매핑.

    반환: (results, pron_data)
      results: 원문 텍스트 SyncResult 목록 (타이밍/word_segments는 독음 정렬 역매핑값).
      pron_data: line_idx → {"pronunciation", "translation", "pron_segments"}.
    """
    from everyric2.inference.prompt import LyricLine, SyncResult, WordSegment
    from everyric2.text.reading import map_pron_alignment_to_line

    pron_for_line = [
        (by_text.get(_normalize_line(ln.text)) or {}).get("pronunciation") or ""
        for ln in lyric_lines
    ]
    pron_lines = [
        LyricLine(text=pron, line_number=ln.line_number)
        for pron, ln in zip(pron_for_line, lyric_lines)
    ]
    ko_results = engine.align(audio, pron_lines, language="ko")

    results = []
    pron_data: dict[int, dict[str, Any]] = {}
    for i, (ln, kr) in enumerate(zip(lyric_lines, ko_results)):
        pron = pron_for_line[i]
        ko_words = kr.word_segments or []
        spans = [(w.word, w.start, w.end) for w in ko_words]

        words = pron_segments = None
        if pron and spans:
            words, pron_segments = map_pron_alignment_to_line(ln.text, pron, spans)

        word_segments = (
            [WordSegment(word=w["word"], start=w["start"], end=w["end"]) for w in words]
            if words
            else None
        )
        line_conf = _geomean([w.confidence for w in ko_words])
        if word_segments and line_conf is not None:
            for ws in word_segments:
                ws.confidence = round(line_conf, 6)

        results.append(
            SyncResult(
                line_number=ln.line_number,
                text=ln.text,
                start_time=kr.start_time,
                end_time=kr.end_time,
                confidence=round(line_conf, 6) if line_conf is not None else None,
                word_segments=word_segments,
            )
        )
        meta = by_text.get(_normalize_line(ln.text)) or {}
        pron_data[i] = {
            "pronunciation": pron or None,
            "translation": meta.get("translation"),
            "pron_segments": pron_segments,
        }
    return results, pron_data


def _run_alignment(
    audio_path: str,
    lyrics: str,
    language: str | None,
    line_meta: list[dict[str, Any]] | None = None,
) -> dict:
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

        # 독음(ko) 정렬 경로: 커버리지가 충분하면 한국어 발음 텍스트+kor adapter로 정렬하고
        # 원문 라인에 역매핑한다. 미달/실패 시 원문 정렬로 폴백 (회귀 0).
        by_text = _pron_by_text(line_meta)
        coverage = _pron_coverage(lyric_lines, by_text)
        pron_data: dict[int, dict[str, Any]] | None = None
        alignment_text = "original"
        if settings.alignment.use_pronunciation and coverage >= 0.9:
            try:
                results, pron_data = _align_with_pronunciation(engine, audio, lyric_lines, by_text)
                alignment_text = "pronunciation"
                logger.info(f"Pronunciation alignment used (coverage={coverage:.2f})")
            except Exception:
                logger.exception("Pronunciation alignment failed; falling back to original text")
                results = engine.align(audio, lyric_lines, language=language or "auto")
                pron_data = None
        else:
            if settings.alignment.use_pronunciation:
                logger.info(f"Pronunciation coverage {coverage:.2f} < 0.9; using original text")
            results = engine.align(audio, lyric_lines, language=language or "auto")

        # 독음 정렬의 star span (아래 VAD 확보 후 '발성 삼킴' 게이트에 쓴다) — 재정렬 전에 포착
        pron_star_spans = (
            list(getattr(engine, "_last_star_spans", []))
            if alignment_text == "pronunciation"
            else []
        )

        # 보컬 스템 1회 분리 → VAD로 라인 경계 보정(가사에 없는 추임새/간주로 늘어진
        # 라인을 실제 발성 구간으로 되돌림) + 아래 멜로디 f0 추출에 재사용
        vocals = _separate_vocals(audio) if settings.melody.separate_vocals else None
        vad_regions: list[tuple[float, float]] | None = None
        clamped_lines: set[int] = set()
        # 보정 전 원본(raw CTC) 타이밍 + 규칙별 보정 라벨 — 확장 디버그 오버레이용
        raw_spans = [(r.start_time, r.end_time) for r in results]
        fixes: dict[int, list[str]] = {}
        if vocals is not None:
            try:
                from everyric2.alignment.timing_postprocess import TimingPostProcessor
                from everyric2.audio.vad import VocalActivityDetector

                vad_result = VocalActivityDetector().detect(vocals)
                # 독음 정렬이 실제 발성을 star 와일드카드로 삼켰는지 검사한다. star 하나가
                # 후반 가창을 통째로 흡수하면 그 라인들이 앞으로 압축·오배치된다
                # (VWVtIg5cdDU(初音ミクの消失) 실측: star 한 개가 후반 가창 21s를 삼켜
                # 후반 라인이 ~40s 앞으로 압축, 불가능한 음절 속도). 단 삼킴 크기만으론
                # 熱異常(브릿지에서 20.7s 삼키지만 배치 정상)과 못 가른다 — 그래서 이건
                # 비용 게이트로만 쓰고, 판정은 '간주 이후 발성 창을 어느 정렬이 채우는가'로
                # 한다. ko가 그 창을 비우고(라인을 앞으로 압축) ja가 크게 채우면 ja 폴백,
                # 둘이 비슷하면(熱異常: ja도 채움, 배치 차이는 국소) ko 유지.
                if alignment_text == "pronunciation" and settings.alignment.star_vocal_fallback_sec > 0:
                    swallowed = _star_swallowed_vocal(pron_star_spans, vad_result.regions)
                    post_win = _post_interlude_window(
                        vad_result.regions, settings.alignment.interlude_min_gap_sec
                    )
                    if swallowed >= settings.alignment.star_vocal_fallback_sec and post_win:
                        ja_candidate = engine.align(audio, lyric_lines, language=language or "auto")
                        ko_fill = _lines_span_overlap(results, post_win)
                        ja_fill = _lines_span_overlap(ja_candidate, post_win)
                        if ja_fill - ko_fill >= settings.alignment.post_interlude_fill_margin_sec:
                            logger.warning(
                                f"Pronunciation alignment vacated the post-interlude window "
                                f"[{post_win[0]:.1f}-{post_win[1]:.1f}]s (ko fills {ko_fill:.1f}s vs "
                                f"ja {ja_fill:.1f}s, +{ja_fill - ko_fill:.1f}s; star swallowed "
                                f"{swallowed:.1f}s); falling back to original-text alignment"
                            )
                            results = ja_candidate
                            alignment_text = "original"
                            pron_data = None
                            raw_spans = [(r.start_time, r.end_time) for r in results]
                            fixes = {}
                        else:
                            logger.info(
                                f"Star swallowed {swallowed:.1f}s but both alignments fill the "
                                f"post-interlude window similarly (ko {ko_fill:.1f}s, ja {ja_fill:.1f}s, "
                                f"+{ja_fill - ko_fill:.1f}s < "
                                f"{settings.alignment.post_interlude_fill_margin_sec}s); keeping "
                                f"pronunciation alignment"
                            )
                # extend_to_vocal은 끄는다: 가사에 없는 반복 가창/애드립도 "보컬 활동"이라
                # 라인을 그쪽으로 늘려버린다 (star 토큰이 흡수해 둔 구간을 도로 끌어안는 역효과)
                pp = TimingPostProcessor(settings.segmentation, extend_to_vocal=False).process(
                    results, vad_result, "line"
                )
                # 0.2s 넘게 움직인 라인만 pp 라벨 — 미세 조정까지 고스트로 그리면 소음
                _diff_fixes(fixes, "pp", raw_spans, pp.results, tol=0.2)
                # 독음 정렬의 무음 언더슛(전이 라인이 간주에 좌초) 교정 — ko 경로에만 적용.
                # _clamp_stretched_lines(내부 _pull이 간주 후 첫 라인을 당김) **이전에** 돌려
                # 좌초 라인이 먼저 다음 온셋을 잡게 한다 (뒤 라인 오인 당김 방지).
                snapped: set[int] = set()
                if alignment_text == "pronunciation":
                    snap_before = [(r.start_time, r.end_time) for r in pp.results]
                    _snap_silence_undershoot(pp.results, vad_result, snapped)
                    _diff_fixes(fixes, "snap", snap_before, pp.results)
                results, clamped_lines = _clamp_stretched_lines(pp.results, vad_result, fixes=fixes)
                clamped_lines |= snapped
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
            # 독음 정렬 경로: 발음 음절 스팬을 멜로디 앵커·발음 표시용으로 직접 부착한다
            # (기존 DP 근사 pron_segments 대신 — 실제 정렬 타이밍이라 더 정확).
            if pron_data is not None:
                pd = pron_data.get(i) or {}
                if pd.get("pronunciation"):
                    seg["pronunciation"] = pd["pronunciation"]
                if pd.get("translation"):
                    seg["translation"] = pd["translation"]
                if pd.get("pron_segments"):
                    seg["pron_segments"] = pd["pron_segments"]
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
                # 보정된 라인은 보정 전 원본 타이밍 + 적용 규칙 라벨을 함께 내려준다
                fx = fixes.get(i)
                if fx:
                    seg["debug"]["orig"] = [round(raw_spans[i][0], 2), round(raw_spans[i][1], 2)]
                    seg["debug"]["fixes"] = fx
            timestamps.append(seg)

        # 가라오케용 음정(MIDI 노트) 주석 — 실패해도 싱크 생성 자체는 계속한다
        f0_curve = None
        if settings.melody.enabled:
            try:
                from everyric2.melody.extractor import MelodyExtractor

                extractor = MelodyExtractor(settings.melody)
                if extractor.is_available():
                    # vocal_regions는 넘기지 않는다 — extractor가 라인 스팬 합집합으로
                    # 자체 마스킹한다 (VAD 마스크는 조용한 벌스 노트를 소실시킴)
                    annotated = extractor.annotate_timestamps(audio, timestamps, vocals=vocals)
                    # 디버그 오버레이용 RAW f0 곡선 (다운샘플, 옥타브 폴딩 전)
                    f0_curve = extractor.last_f0_curve
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

        # 곡 단위 디버그 메타 — star가 흡수한 구간(가사 밖 가창)과 VAD 발성 구간,
        # 그리고 어떤 텍스트로 정렬했는지(원문 vs 독음) 클라 디버그 표시용.
        # 독음을 유지한 경우 debug star는 ko 정렬의 것이어야 한다 — star 가드가 교차검증용
        # ja 정렬을 돌리면 engine._last_star_spans가 ja star로 덮이므로 미리 포착해 둔
        # pron_star_spans(ko star)를 쓴다. 원문 정렬(폴백 포함)은 _last_star_spans가 맞다.
        final_star_source = pron_star_spans if alignment_text == "pronunciation" else getattr(
            engine, "_last_star_spans", []
        )
        star_spans = [list(s) for s in final_star_source]
        debug_meta = {
            "star_spans": star_spans,
            "vad_regions": [list(v) for v in vad_regions] if vad_regions is not None else None,
            "alignment_text": alignment_text,
            # 음정 모델(RMVPE/FCPE) RAW f0 곡선 — 레인 디버그 오버레이용
            "f0_curve": f0_curve,
        }

        return {
            "timestamps": timestamps,
            "language": detected_lang,
            "quality_score": avg_confidence,
            "debug": debug_meta,
            "alignment_text": alignment_text,
            # 가라오케 레인의 마디 단위 고정 창·비트 격자용 — 실패해도 None으로 계속
            "tempo": _estimate_tempo(audio),
        }
    finally:
        audio_path_obj.unlink(missing_ok=True)
