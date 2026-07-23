import asyncio
import hashlib
import logging
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

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


# 사용자 취소 요청 잡 — 취소 API가 넣고, 워커가 단계 경계에서 확인해 중단한다.
# 이미 도는 CTC/demucs 스레드 자체는 중단하지 못하므로 '경계 취소'다
# (대기열 슬롯 진입·다운로드 직후·정렬 시작 전·저장 전). 확인 시 집합에서 제거된다.
_CANCEL_REQUESTED: set[str] = set()


def request_cancel(job_id: str) -> None:
    _CANCEL_REQUESTED.add(job_id)


async def _consume_cancel(job_id: str) -> bool:
    """취소 요청이 있으면 잡을 실패(취소) 상태로 마감하고 True."""
    if job_id not in _CANCEL_REQUESTED:
        return False
    _CANCEL_REQUESTED.discard(job_id)
    from everyric2.server.db.connection import get_session
    from everyric2.server.db.repository import JobRepository

    async with get_session() as session:
        await JobRepository(session).update_status(job_id, "failed", error="요청으로 취소했어요")
    logger.info(f"Job {job_id} cancelled at a stage boundary")
    return True


# 진행 단계 → 전역 진행률 창 (lo, hi). 단계 내부의 실제 진행 콜백은 없으므로
# 창 안에서 시간 기반으로 차오르고, job API가 창 기준 단계별 퍼센트를 계산한다.
STAGE_WINDOWS: dict[str, tuple[int, int]] = {
    "다운로드": (10, 34),
    "캐시 확인": (34, 36),
    "보컬 분리": (36, 50),
    "전사 정렬": (50, 72),
    "타이밍 보정": (72, 80),
    "멜로디 분석": (80, 88),
    "저장": (90, 100),
}

# 동시 처리 슬롯 — 정렬(demucs+CTC+멜로디)은 GPU/RAM을 크게 쓰므로 상한 없이 병렬로
# 돌리면 OOM이 난다. 초과분은 status=queued로 대기하고 확장이 "대기열"로 표시한다.
_JOB_SEMAPHORE: asyncio.Semaphore | None = None


def _job_slot() -> asyncio.Semaphore:
    global _JOB_SEMAPHORE
    if _JOB_SEMAPHORE is None:
        from everyric2.config.settings import get_settings

        _JOB_SEMAPHORE = asyncio.Semaphore(max(1, get_settings().server.max_concurrent_jobs))
    return _JOB_SEMAPHORE


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


def _audio_duration_sec(file_path: str) -> float | None:
    """다운로드된 오디오 길이(초) — 헤더만 읽어 즉시 반환. 실패 시 None(상한 검사 생략)."""
    try:
        import soundfile as sf

        info = sf.info(file_path)
        return float(info.frames) / float(info.samplerate or 1)
    except Exception:
        return None


async def process_job(job_id: str) -> None:
    from everyric2.server.db.connection import get_session
    from everyric2.server.db.repository import JobRepository

    async with get_session() as session:
        job_repo = JobRepository(session)
        job = await job_repo.get_by_id(job_id)

        if not job:
            logger.error(f"Job not found: {job_id}")
            return

        # 슬롯이 다 차 있으면 대기열 — job API가 queue_position을 계산해 내려준다
        await job_repo.update_status(job_id, "queued", progress=0)

    slot = _job_slot()
    if slot.locked():
        logger.info(f"Job {job_id} waiting for a processing slot")
    async with slot:
        # 대기열에 있는 동안 취소된 잡은 슬롯을 잡자마자 놓아준다
        if await _consume_cancel(job_id):
            return
        await _process_job_inner(job_id, job)


async def _complete_from_cache_db(
    job_id: str, job, audio_hash: str, lyrics_hash_value: str
) -> bool:
    """(audio_hash, lyrics_hash)가 일치하는 기존 싱크가 있으면 정렬 없이 잡을 완료한다.

    교차 영상 재사용: 조회(GET /api/sync·job API)는 전부 video_id 컬럼 기반이라, 다른
    영상의 행을 재사용만 하면 이 영상은 completed인데 가사가 영영 안 뜨고 초기화(DELETE)도
    지울 행이 없어 복구 불가였다 (동일 오디오 재업로드/공식 오디오 실측). 이 영상 몫의
    행을 복사 생성하고, 대기 중인 발음/번역 메타·출처도 원본이 아닌 이 행에만 반영한다.
    재사용(완료)했으면 True. 오디오 파일 정리는 호출부 몫이다 (원격 워커는 서버에 파일이
    없고 로컬에서 지운다 — _try_complete_from_cache가 인프로세스용 래퍼)."""
    from everyric2.server.db.connection import get_session
    from everyric2.server.db.repository import JobRepository, SyncRepository

    async with get_session() as session:
        sync_repo = SyncRepository(session)
        existing = await sync_repo.get_by_audio_and_lyrics_hash(audio_hash, lyrics_hash_value)
        if not existing:
            return False
        meta = _PENDING_LINE_META.pop(job_id, None)
        attr = _PENDING_ATTRIBUTION.pop(job_id, None)
        target = existing
        if existing.video_id != job.video_id:
            src = dict(existing.timestamps)
            segments = [dict(s) for s in src.pop("segments", [])]
            target = await sync_repo.create(
                video_id=job.video_id,
                lyrics_hash=lyrics_hash_value,
                timestamps=segments,
                language=existing.language,
                engine=existing.engine,
                quality_score=existing.quality_score,
                audio_hash=audio_hash,
                extra=src,
            )
            logger.info(
                f"Job {job_id}: copied sync from video {existing.video_id} "
                f"(same audio+lyrics) into {job.video_id}"
            )
        updated = dict(target.timestamps)
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
            target.timestamps = updated
        await JobRepository(session).update_status(
            job_id, "completed", progress=100, result_id=target.id
        )
        logger.info(f"Job {job_id} reused existing sync (audio_hash match)")
        return True


async def _try_complete_from_cache(
    job_id: str, job, audio_hash: str, lyrics_hash_value: str, audio_path: str
) -> bool:
    """캐시 완결 시 다운로드한 오디오까지 정리하는 인프로세스 래퍼.

    원격 워커 경로는 서버에 오디오 파일이 없으므로 이 래퍼 대신 _complete_from_cache_db를
    직접 부르고, 로컬 오디오는 워커 쪽 hooks.cache_check가 지운다."""
    completed = await _complete_from_cache_db(job_id, job, audio_hash, lyrics_hash_value)
    if completed:
        Path(audio_path).unlink(missing_ok=True)
    return completed


class PipelineError(Exception):
    """사용자에게 보이는 파이프라인 실패 (예: 영상 과길이). str(e)가 실패 문구가 된다."""


def over_length_message(duration_sec: float, max_audio_sec: int) -> str:
    """과길이 영상 거부 문구 — run_pipeline(다운로드 후)과 미디어 캐시 프리플라이트가 공유한다."""
    return (
        f"영상이 너무 길어요 ({duration_sec / 60:.0f}분). 싱크 생성은 "
        f"{max_audio_sec // 60}분 이하의 노래 영상에서만 지원해요."
    )


@dataclass
class JobInput:
    """run_pipeline 입력 — 인프로세스는 스태시를 peek해, 원격은 claim 응답으로 채운다.

    오디오 확보 우선순위(_acquire_audio): audio_path(인프로세스가 미디어 캐시에서 추출해 둔
    로컬 파일) > audio_url(원격 워커가 서버 캐시 파일을 HTTP로 받음) > yt-dlp 다운로드.
    앞의 두 경로가 실패하면 yt-dlp로 폴백하고, audio_hash는 어느 경로든 받은 파일로 동일 계산.
    """

    job_id: str
    video_id: str
    lyrics: str
    language: str | None = None
    line_meta: list[dict[str, Any]] | None = None
    attribution: dict[str, Any] | None = None
    force: bool = False
    max_audio_sec: int = 0
    # 미디어 캐시 연동 — 인프로세스는 로컬 파일 경로, 원격 워커는 인증 헤더 딸린 HTTP URL
    audio_path: str | None = None
    audio_url: str | None = None
    audio_url_headers: dict[str, str] | None = None


@dataclass
class PipelineResult:
    """run_pipeline 성공 결과 — 인프로세스 저장 경로/원격 result 제출이 그대로 저장한다."""

    timestamps: list[dict[str, Any]]
    language: str | None
    quality_score: float | None
    audio_hash: str
    extra: dict[str, Any] | None


class PipelineHooks(Protocol):
    """파이프라인 코어가 앞뒤(진행률·취소·캐시)를 위임하는 콜백 묶음.

    - report: 순수 진행률 보고(취소 소진 없음). 다운로드 틱·단계 모니터가 쓴다.
    - progress: 진행률 보고 + 취소 확인. False면 취소 요청됨 → 코어가 중단(None 반환).
    - cache_check: (audio_hash, lyrics) 캐시 완결 판정. True면 잡 완료·오디오 정리까지
      끝났으므로 코어가 정렬을 건너뛰고 중단한다.
    """

    async def report(self, progress: int, stage: str) -> None: ...

    async def progress(self, progress: int, stage: str) -> bool: ...

    async def cache_check(self, audio_hash: str, audio_path: str) -> bool: ...


class InProcessHooks:
    """서버 프로세스가 직접 처리할 때의 hooks — 기존 _set_progress/_consume_cancel/
    _try_complete_from_cache를 감싸 리팩터 전과 같은 관찰 동작을 낸다."""

    def __init__(self, job_id: str, job) -> None:
        self.job_id = job_id
        self.job = job

    async def report(self, progress: int, stage: str) -> None:
        await _set_progress(self.job_id, progress, stage)

    async def progress(self, progress: int, stage: str) -> bool:
        # _set_progress는 취소 대기 중이면 쓰기를 건너뛴다(가드). 이어 취소를 소진해
        # 리팩터 전의 "경계에서 취소 확인" 동작을 그대로 재현한다.
        await _set_progress(self.job_id, progress, stage)
        return not await _consume_cancel(self.job_id)

    async def cache_check(self, audio_hash: str, audio_path: str) -> bool:
        from everyric2.server.db.repository import hash_lyrics

        return await _try_complete_from_cache(
            self.job_id, self.job, audio_hash, hash_lyrics(self.job.lyrics), audio_path
        )


async def run_pipeline(job: JobInput, hooks: PipelineHooks) -> PipelineResult | None:
    """생성 파이프라인 코어 — 인프로세스 워커와 원격 워커가 공유한다.

    hooks.progress로 단계·진행률을 보고하고(False면 취소 → None 반환으로 중단),
    hooks.cache_check로 (audio_hash, lyrics) 캐시 완결을 판정한다(True면 정렬 생략,
    None 반환). 성공하면 PipelineResult를, 취소/캐시 완결이면 None을 돌려준다. 영상
    과길이 등 사용자 노출 실패는 PipelineError로 올린다. 관찰 가능한 동작(단계 문구·
    진행률 값·취소 경계·캐시 동작·실패 문구·틱/모니터 UX)은 리팩터 전과 동일하다."""
    if not await hooks.progress(10, "다운로드"):
        return None

    dl_ticker = asyncio.create_task(
        _tick_progress(hooks.report, start=10, cap=33, interval=2.0, stage="다운로드")
    )
    try:
        download_result = await asyncio.get_event_loop().run_in_executor(
            None, _acquire_audio, job
        )
    finally:
        dl_ticker.cancel()
    audio_hash = download_result["audio_hash"]
    audio_path = download_result["audio_path"]

    # 노래가 아닌 초장시간 영상(팟캐스트/라이브 다시보기)이 GPU 슬롯을 몇 시간씩 점유하는
    # 것을 막는다 — 상한 초과는 정렬 전에 친절하게 실패. 취소 경계(아래 progress)보다 먼저
    # 검사해 리팩터 전 "다운로드 직후 → 과길이 검사 → 캐시 확인" 순서를 보존한다.
    if job.max_audio_sec > 0:
        duration = _audio_duration_sec(audio_path)
        if duration and duration > job.max_audio_sec:
            Path(audio_path).unlink(missing_ok=True)
            raise PipelineError(over_length_message(duration, job.max_audio_sec))

    # 다운로드 완료 → 캐시 확인 (취소 경계 겸)
    if not await hooks.progress(35, "캐시 확인"):
        Path(audio_path).unlink(missing_ok=True)
        return None

    if not job.force and await hooks.cache_check(audio_hash, audio_path):
        # 캐시로 완결 — 오디오는 hooks.cache_check가 이미 정리했다
        return None

    # 캐시 미스 → 정렬 진입 (취소 경계 겸)
    if not await hooks.progress(36, "보컬 분리"):
        Path(audio_path).unlink(missing_ok=True)
        return None

    # 정렬(CTC+분리+보정+멜로디)은 수십 초 걸리는 단일 블록 — 정렬 스레드가 단계명을
    # stage_holder에 쓰고, 모니터가 단계 창 안에서 진행률을 차오르게 하며 보고한다
    stage_holder: dict[str, str] = {"stage": "보컬 분리"}
    monitor = asyncio.create_task(_stage_monitor(hooks.report, stage_holder, start=36))
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            _run_alignment,
            audio_path,
            job.lyrics,
            job.language,
            job.line_meta,
            lambda name: stage_holder.__setitem__("stage", name),
        )
    finally:
        monitor.cancel()

    # 정렬 완료, 저장 단계 (취소 경계 겸) — 오디오는 _run_alignment의 finally가 정리했다
    if not await hooks.progress(90, "저장"):
        return None

    # 독음 정렬 경로는 발음/번역/pron_segments를 이미 세그먼트에 붙였으므로 재병합 생략
    if job.line_meta and result.get("alignment_text") != "pronunciation":
        merged = merge_line_meta(result["timestamps"], job.line_meta)
        logger.info(f"Line meta merged on {merged} segments")

    return PipelineResult(
        timestamps=result["timestamps"],
        language=result.get("language"),
        quality_score=result.get("quality_score"),
        audio_hash=audio_hash,
        extra=_build_extra(result, job.attribution),
    )


async def _process_job_inner(job_id: str, job) -> None:
    from everyric2.config.settings import get_settings as _get_settings
    from everyric2.server.db.connection import get_session
    from everyric2.server.db.repository import JobRepository, SyncRepository, hash_lyrics

    # 스태시(발음/번역 메타·출처·강제)를 peek해 코어 입력을 만든다. 정상 완료/실패 시
    # 아래에서 pop한다 (캐시 완결 경로는 _complete_from_cache_db가 이미 pop). force는
    # 코어 입력으로 캡처했으니 여기서 discard한다.
    force = job_id in _PENDING_FORCE
    _PENDING_FORCE.discard(job_id)
    max_audio_sec = _get_settings().server.max_job_audio_sec

    # 슬롯 획득 직후 = 잡이 이 프로세스로 넘어오는 순간 → 미디어 캐시 조회(있으면 추출 사용).
    # 과길이 프리플라이트는 다운로드 없이 즉시 실패시킨다.
    from everyric2.server.media_cache import prepare_cached_audio

    cache_path, fail_reason = await prepare_cached_audio(job.video_id, job_id, max_audio_sec)
    if fail_reason:
        _PENDING_LINE_META.pop(job_id, None)
        _PENDING_ATTRIBUTION.pop(job_id, None)
        async with get_session() as session:
            await JobRepository(session).update_status(job_id, "failed", error=fail_reason)
        logger.info(f"Job {job_id} rejected (media cache preflight): {fail_reason}")
        return

    job_input = JobInput(
        job_id=job_id,
        video_id=job.video_id,
        lyrics=job.lyrics,
        language=job.language,
        line_meta=_PENDING_LINE_META.get(job_id),
        attribution=_PENDING_ATTRIBUTION.get(job_id),
        force=force,
        max_audio_sec=max_audio_sec,
        audio_path=cache_path,
    )
    try:
        result = await run_pipeline(job_input, InProcessHooks(job_id, job))
        if result is None:
            # 취소 또는 캐시 완결 — 잡 상태·오디오 정리는 각 경로가 이미 끝냈다
            _PENDING_LINE_META.pop(job_id, None)
            _PENDING_ATTRIBUTION.pop(job_id, None)
            return

        async with get_session() as session:
            job_repo = JobRepository(session)
            sync_repo = SyncRepository(session)

            sync_result = await sync_repo.create(
                video_id=job.video_id,
                lyrics_hash=hash_lyrics(job.lyrics),
                timestamps=result.timestamps,
                language=result.language,
                engine="ctc",
                quality_score=result.quality_score,
                audio_hash=result.audio_hash,
                extra=result.extra,
            )

            await job_repo.update_status(
                job_id, "completed", progress=100, result_id=sync_result.id
            )
            logger.info(f"Job completed: {job_id}")
        _PENDING_LINE_META.pop(job_id, None)
        _PENDING_ATTRIBUTION.pop(job_id, None)

    except PipelineError as e:
        # 사용자 노출 실패 (과길이 등) — 친절한 한국어 문구를 그대로 보존
        _PENDING_LINE_META.pop(job_id, None)
        _PENDING_ATTRIBUTION.pop(job_id, None)
        async with get_session() as session:
            await JobRepository(session).update_status(job_id, "failed", error=str(e))
        logger.info(f"Job {job_id} rejected: {e}")

    except Exception as e:
        logger.exception(f"Job failed: {job_id}")
        _PENDING_LINE_META.pop(job_id, None)
        _PENDING_ATTRIBUTION.pop(job_id, None)
        _PENDING_FORCE.discard(job_id)
        async with get_session() as session:
            job_repo = JobRepository(session)
            await job_repo.update_status(job_id, "failed", error=str(e))


async def _set_progress(job_id: str, progress: int, stage: str | None = None) -> None:
    from everyric2.server.db.connection import get_session
    from everyric2.server.db.repository import JobRepository

    # 취소 대기 중이면 진행률 갱신을 멈춘다 — 취소 API가 이미 failed로 마감했는데
    # 모니터가 processing으로 되돌려 쓰면 클라이언트가 failed↔processing 왕복을 본다
    if job_id in _CANCEL_REQUESTED:
        return
    async with get_session() as session:
        await JobRepository(session).update_status(
            job_id, "processing", progress=progress, stage=stage
        )


async def _tick_progress(
    report, start: int, cap: int, interval: float = 4.0, stage: str | None = None
) -> None:
    """긴 단계 동안 진행률을 cap까지 천천히 올린다 — 취소되면 그대로 멈춘다.

    report는 순수 진행률 보고 콜백(hooks.report) — 취소를 소진하지 않는다. 틱이 취소를
    소진해 버리면 경계의 progress가 취소를 못 보고 잡이 그대로 진행되므로 반드시 report다."""
    progress = start
    try:
        while progress < cap:
            await asyncio.sleep(interval)
            progress = min(cap, progress + 4)
            await report(progress, stage)
    except asyncio.CancelledError:
        pass


async def _stage_monitor(report, stage_holder: dict[str, str], start: int, interval: float = 2.0) -> None:
    """정렬 블록 동안 stage_holder의 현재 단계를 읽어 단계명+진행률을 report로 보고한다.

    단계가 바뀌면 그 단계 창의 시작으로 점프하고, 같은 단계가 유지되는 동안은
    틱마다 창 폭의 1/6씩 상한까지 차오른다 (내부 진행 콜백이 없는 근사치)."""
    progress = float(start)
    last_stage: str | None = None
    try:
        while True:
            await asyncio.sleep(interval)
            stage = stage_holder.get("stage")
            if not stage:
                continue
            lo, hi = STAGE_WINDOWS.get(stage, (36, 88))
            if stage != last_stage:
                last_stage = stage
                progress = max(progress, float(lo))
            else:
                progress = min(float(hi), progress + (hi - lo) / 6.0)
            await report(int(progress), stage)
    except asyncio.CancelledError:
        pass


def _acquire_audio(job: "JobInput") -> dict:
    """오디오 확보 — audio_path(로컬 캐시 추출) > audio_url(서버 캐시 HTTP) > yt-dlp.

    앞선 캐시 경로가 실패하면 조용히 yt-dlp로 폴백한다(INFO 로그 1줄). audio_hash는 어느
    경로든 확보한 파일로 동일하게 계산한다 — 캐시/다운로드가 같은 원본이면 해시도 같아
    교차 영상 캐시 재사용이 그대로 동작한다."""
    # 인프로세스: 서버가 미디어 캐시에서 추출해 넘긴 로컬 파일 직사용
    if job.audio_path:
        p = Path(job.audio_path)
        if p.exists():
            try:
                return {"audio_path": str(p), "audio_hash": compute_audio_hash(p)}
            except Exception:
                logger.info("캐시 오디오 파일을 읽지 못해 yt-dlp로 폴백해요 (video %s)", job.video_id)
        else:
            logger.info("캐시 오디오 파일이 없어 yt-dlp로 폴백해요 (video %s)", job.video_id)
    # 원격 워커: 서버 캐시 파일을 인증 헤더로 HTTP 다운로드
    if job.audio_url:
        try:
            path = _http_download_audio(
                job.audio_url, job.audio_url_headers, job.video_id, job.job_id
            )
            return {"audio_path": str(path), "audio_hash": compute_audio_hash(path)}
        except Exception:
            logger.info("서버 캐시 오디오 받기 실패 — yt-dlp로 폴백해요 (video %s)", job.video_id)
    return _download_and_hash(job.video_id, job.job_id)


def _http_download_audio(
    url: str, headers: dict[str, str] | None, video_id: str, job_id: str
) -> Path:
    """서버 캐시 오디오를 HTTP로 받아 임시 파일에 저장 (원격 워커 전용). 의존성 없이 requests."""
    import requests

    from everyric2.config.settings import get_settings

    temp_dir = get_settings().audio.temp_dir
    temp_dir.mkdir(parents=True, exist_ok=True)
    dest = temp_dir / f"{video_id}-{job_id[:8]}-cache.m4a"
    with requests.get(url, headers=headers or {}, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    return dest


def _download_and_hash(video_id: str, job_id: str) -> dict:
    from everyric2.audio.downloader import YouTubeDownloader

    downloader = YouTubeDownloader()
    youtube_url = f"https://www.youtube.com/watch?v={video_id}"
    # 잡별 고유 파일명 — 기본 %(title)s 템플릿은 같은 영상의 동시 잡이 한 파일을 두고
    # 경합해 Windows에서 WinError 32(파일 사용 중)로 다운로드가 깨진다
    dl_result = downloader.download(youtube_url, filename=f"{video_id}-{job_id[:8]}")
    audio_hash = compute_audio_hash(dl_result.audio_path)

    return {
        "audio_path": str(dl_result.audio_path),
        "audio_hash": audio_hash,
    }


def _build_extra(result: dict[str, Any], attribution: dict[str, Any] | None) -> dict[str, Any] | None:
    """싱크 JSON의 segments 밖 부가정보(디버그 메타, 출처 표기, 템포, 키) 조립."""
    extra: dict[str, Any] = {}
    if result.get("debug"):
        extra["debug"] = result["debug"]
    if result.get("tempo"):
        extra["tempo"] = result["tempo"]
    if result.get("key"):
        extra["key"] = result["key"]
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
    """demucs 보컬 분리 (실패/미설치 시 None) — VAD 클램프와 멜로디 f0가 공유한다.

    분리기는 웜 캐시 싱글턴(get_shared_separator)에서 가져와 잡마다 재생성하지 않는다 (WS2-A)."""
    try:
        import torch

        from everyric2.audio.separator import get_shared_separator

        separator = get_shared_separator()
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


def _splice_alignments(ko_results, ja_results, post_win: tuple[float, float]) -> int | None:
    """ko 정렬이 간주 이후 블록을 앞으로 압축했을 때 전곡 ja 폴백 대신 스플라이스한다.

    간주 전 라인은 ko(독음 음절 타이밍) 유지, ja가 간주 이후에 배치한 첫 라인(k)부터는
    ja 타이밍으로 교체한다 (ko_results를 제자리 수정). 가사 순서는 고정이고 CTC 정렬은
    라인 순서 단조라, ja 기준 '간주 이후 첫 라인' 인덱스가 곧 텍스트상 후반 블록의 시작이다.
    경계에서 ko 마지막 유지 라인이 ja 첫 교체 라인을 침범하면 끝을 클램프한다.

    반환: 교체 시작 인덱스 k. 스플라이스가 성립하지 않으면 None (ko_results 무변경):
      - ja가 간주 이후에 아무 라인도 안 둠 (가드 오발 — 호출부가 전곡 폴백)
      - k == 0 (전곡 교체 = 전곡 폴백과 동일하므로 기존 경로에 맡김)
      - ko 유지 구간(k 이전)에 경계를 넘는 라인이 있음 (압축이 간주를 걸침 — 부분 보존 불가)
    """
    gap_end = post_win[0]
    k = next(
        (i for i, r in enumerate(ja_results) if r.start_time >= gap_end - 0.5),
        None,
    )
    if not k:  # None 또는 0
        return None
    bound = ja_results[k].start_time
    if any(r.start_time >= bound for r in ko_results[:k]):
        return None
    for i in range(k):
        r = ko_results[i]
        if r.end_time > bound:
            r.end_time = max(r.start_time + 0.01, bound)
            if r.word_segments:
                _shift_word_segments(r.word_segments, r.start_time, r.end_time)
    ko_results[k:] = ja_results[k:]
    return k


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
        # 음절별 confidence까지 함께 넘겨 글자별 conf 역매핑을 살린다 (라인 균일 부여 회귀 수정)
        spans = [(w.word, w.start, w.end, w.confidence) for w in ko_words]

        words = pron_segments = None
        if pron and spans:
            words, pron_segments = map_pron_alignment_to_line(ln.text, pron, spans)

        word_segments = (
            [WordSegment(word=w["word"], start=w["start"], end=w["end"]) for w in words]
            if words
            else None
        )
        line_conf = _geomean([w.confidence for w in ko_words])
        if word_segments:
            # 글자별 conf(reading이 음절 conf 기하평균으로 산출) 우선, 매핑 불가 글자는 라인 기하평균 폴백
            for ws, w in zip(word_segments, words):
                c = w.get("confidence")
                if c is None:
                    c = line_conf
                ws.confidence = round(c, 6) if c is not None else None

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
    on_stage: Any | None = None,
) -> dict:
    from everyric2.audio.loader import AudioLoader
    from everyric2.config.settings import get_settings
    from everyric2.inference.prompt import LyricLine

    def report(stage: str) -> None:
        # 진행 단계 보고 — 실패해도 정렬 자체는 계속한다
        if on_stage is not None:
            try:
                on_stage(stage)
            except Exception:
                pass

    settings = get_settings()
    loader = AudioLoader()
    audio_path_obj = Path(audio_path)

    # WS2-B 병렬 f0 실행기 — 정렬 도중 예외가 나도 outer finally가 반드시 정리하도록 밖에 둔다
    f0_executor = None

    try:
        audio = loader.load(audio_path_obj)
        lyric_lines = LyricLine.from_text(lyrics)

        # CTC 엔진은 웜 캐시 싱글턴 — 같은 언어의 두 번째 잡부터 모델 재로드 0회 (WS2-A).
        # torch를 최상위 import하는 모듈이라 반드시 여기서 지연 import한다 (API 전용 모드
        # 프로세스에 torch가 딸려 들어오지 않게 — main.py 지연 임포트 계약).
        from everyric2.alignment.ctc_engine import get_shared_ctc_engine

        engine = get_shared_ctc_engine(settings.alignment)
        if not engine.is_available():
            raise RuntimeError("CTC engine not available")

        # 보컬 스템 1회 분리 — 원 설계(CLI --separate)대로 정렬 입력으로 쓰고, 아래 VAD
        # 라인 경계 보정과 멜로디 f0 추출에 재사용한다. 반주가 빠진 스템은 CTC emission이
        # 훨씬 깨끗해 고밀도 믹스/이펙트 구간에서 정렬 품질이 오른다. 미설치/실패 시 믹스 폴백.
        report("보컬 분리")
        need_vocals = settings.melody.separate_vocals or settings.alignment.align_on_vocals
        vocals = _separate_vocals(audio) if need_vocals else None
        align_audio = (
            vocals if (vocals is not None and settings.alignment.align_on_vocals) else audio
        )

        # WS2-B: 멜로디 f0 전곡 추론을 CTC 정렬과 병렬로 시작한다 — f0 추론은 정렬 결과에
        # 무의존이라(전곡 신호만 처리) GPU 유휴를 줄인다. 노트 부착(annotate)은 정렬·타이밍
        # 보정이 끝난 뒤 이 f0를 주입해 수행한다. 진행 stage 표시 순서(보컬 분리→전사 정렬→
        # 타이밍 보정→멜로디 분석)는 그대로 — f0는 백그라운드라 보고 stage를 바꾸지 않는다.
        # 멜로디 실패는 비치명(노트 없이 계속)이므로 여기서 예외를 삼키지 않고 result()에서 처리.
        f0_future = None
        melody_extractor = None
        if settings.melody.enabled:
            from everyric2.melody.extractor import get_shared_extractor

            melody_extractor = get_shared_extractor(settings.melody)
            if melody_extractor.is_available():
                import concurrent.futures

                f0_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                f0_future = f0_executor.submit(melody_extractor.precompute_f0, audio, vocals)
            else:
                logger.warning("Melody enabled but torchfcpe is not installed; skipping")
                melody_extractor = None

        report("전사 정렬")
        # 독음(ko) 정렬 경로: 커버리지가 충분하면 한국어 발음 텍스트+kor adapter로 정렬하고
        # 원문 라인에 역매핑한다. 미달/실패 시 원문 정렬로 폴백 (회귀 0).
        by_text = _pron_by_text(line_meta)
        coverage = _pron_coverage(lyric_lines, by_text)
        pron_data: dict[int, dict[str, Any]] | None = None
        alignment_text = "original"
        if settings.alignment.use_pronunciation and coverage >= 0.9:
            try:
                results, pron_data = _align_with_pronunciation(
                    engine, align_audio, lyric_lines, by_text
                )
                alignment_text = "pronunciation"
                logger.info(f"Pronunciation alignment used (coverage={coverage:.2f})")
            except Exception:
                logger.exception("Pronunciation alignment failed; falling back to original text")
                results = engine.align(align_audio, lyric_lines, language=language or "auto")
                pron_data = None
        else:
            if settings.alignment.use_pronunciation:
                logger.info(f"Pronunciation coverage {coverage:.2f} < 0.9; using original text")
            results = engine.align(align_audio, lyric_lines, language=language or "auto")

        # 독음 정렬의 star span (아래 VAD 확보 후 '발성 삼킴' 게이트에 쓴다) — 재정렬 전에 포착
        pron_star_spans = (
            list(getattr(engine, "_last_star_spans", []))
            if alignment_text == "pronunciation"
            else []
        )

        # VAD로 라인 경계 보정 — 가사에 없는 추임새/간주로 늘어진 라인을 실제 발성 구간으로
        report("타이밍 보정")
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
                        ja_candidate = engine.align(
                            align_audio, lyric_lines, language=language or "auto"
                        )
                        ko_fill = _lines_span_overlap(results, post_win)
                        ja_fill = _lines_span_overlap(ja_candidate, post_win)
                        if ja_fill - ko_fill >= settings.alignment.post_interlude_fill_margin_sec:
                            splice_k = (
                                _splice_alignments(results, ja_candidate, post_win)
                                if settings.alignment.star_guard_splice
                                else None
                            )
                            if splice_k is not None:
                                # 간주 전 라인은 ko(음절 타이밍) 보존 + 간주 후는 ja로 교체.
                                # 교체된 라인의 ko pron_segments는 압축된 타이밍이라 무효 —
                                # 스팬만 버리면 발음·번역 텍스트는 남고, 캐시 재병합이 ja
                                # 타이밍 기반 DP 근사로 노트 스팬을 복원한다.
                                logger.warning(
                                    f"Pronunciation alignment vacated the post-interlude window "
                                    f"[{post_win[0]:.1f}-{post_win[1]:.1f}]s (ko fills {ko_fill:.1f}s "
                                    f"vs ja {ja_fill:.1f}s; star swallowed {swallowed:.1f}s); "
                                    f"splicing ko[:{splice_k}] + ja[{splice_k}:]"
                                )
                                alignment_text = "spliced"
                                if pron_data:
                                    for idx in range(splice_k, len(results)):
                                        pd = pron_data.get(idx)
                                        if pd:
                                            pd["pron_segments"] = None
                            else:
                                logger.warning(
                                    f"Pronunciation alignment vacated the post-interlude window "
                                    f"[{post_win[0]:.1f}-{post_win[1]:.1f}]s (ko fills {ko_fill:.1f}s "
                                    f"vs ja {ja_fill:.1f}s, +{ja_fill - ko_fill:.1f}s; star swallowed "
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
            # 원문 정렬 경로는 엔진이 라인 confidence를 안 채운다(word에만 존재) —
            # ko 경로와 동일하게 글자 conf의 기하평균으로 보충해 quality_score와
            # 레인/패널의 곡 단위 conf 통계가 모든 곡에서 동작하게 한다
            line_conf = r.confidence
            if line_conf is None and r.word_segments:
                line_conf = _geomean([w.confidence for w in r.word_segments])
            if line_conf is not None:
                seg["confidence"] = round(line_conf, 6)
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

        # 가라오케용 음정(MIDI 노트) 주석 — 실패해도 싱크 생성 자체는 계속한다.
        # f0 전곡 추론은 위에서 정렬과 병렬로 이미 돌고 있다(f0_future) — 여기서 그 결과를
        # 받아 정렬 결과에 노트만 부착한다. 미가용/미설정은 위에서 이미 걸러 melody_extractor가
        # None이다(경고도 이미 1줄 기록). 멜로디 실패는 비치명 — 노트 없이 계속.
        report("멜로디 분석")
        f0_curve = None
        song_key = None
        if melody_extractor is not None:
            try:
                precomputed_f0 = f0_future.result() if f0_future is not None else None
                # vocal_regions는 넘기지 않는다 — extractor가 라인 스팬 합집합으로
                # 자체 마스킹한다 (VAD 마스크는 조용한 벌스 노트를 소실시킴)
                annotated = melody_extractor.annotate_timestamps(
                    audio, timestamps, vocals=vocals, precomputed_f0=precomputed_f0
                )
                # 디버그 오버레이용 RAW f0 곡선 (다운샘플, 옥타브 폴딩 전)
                f0_curve = melody_extractor.last_f0_curve
                # 곡 키 (K-S 추정) — 레인 표시용, 스냅 보정은 extractor 내부에서 완료
                song_key = melody_extractor.last_key
                logger.info(f"Melody notes annotated on {annotated} spans")
            except Exception:
                logger.exception("Melody extraction failed; continuing without notes")
            finally:
                if f0_executor is not None:
                    f0_executor.shutdown(wait=True)

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
        # pron_star_spans(ko star)를 쓴다. 원문 정렬(폴백 포함)은 _last_star_spans가 맞고,
        # 스플라이스는 후반(교체 구간)을 지배하는 ja star(_last_star_spans)를 그대로 쓴다.
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
            # 곡 키 (멜로디 분석 부산물) — 레인 좌상단 표시용
            "key": song_key,
        }
    finally:
        # 성공 경로는 멜로디 블록의 finally가 이미 shutdown(wait=True)했다 — 정렬 예외로
        # 여기로 빠졌을 때만 남은 f0 스레드를 정리한다(멱등, 실행 중 future는 기다리지 않음)
        if f0_executor is not None:
            f0_executor.shutdown(wait=False)
        audio_path_obj.unlink(missing_ok=True)
