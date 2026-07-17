"""원격 GPU 워커 풀 API — 워커가 아웃바운드 폴링으로 잡을 클레임·처리·제출한다.

서버는 API+DB+큐만 맡고, 생성 파이프라인(다운로드/분리/정렬/멜로디)은 원격 워커(집 PC
GPU 등)가 돌린다. 인증은 X-Worker-Key(EVERYRIC_SERVER_WORKER_KEY) 한 개를 공유하는
개인 풀 모델 — 워커는 worker_id로 구분한다. 리스(어느 워커가 어떤 잡을 무는지 + 만료)는
서버 인메모리 레지스트리로 관리해 Job 테이블 스키마를 건드리지 않는다 (서버 단일 프로세스
전제. 재시작 시 유실은 좀비 잡 정리(db/connection.py)가 이미 커버하는 동작이라 일관적).
"""

import asyncio
import time
from typing import Any

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from everyric2 import __version__
from everyric2.config.settings import get_settings
from everyric2.server.db.connection import get_session
from everyric2.server.db.repository import JobRepository, SyncRepository, hash_lyrics

router = APIRouter(prefix="/api/worker", tags=["worker"])

# 리스 레지스트리 {job_id: (worker_id, expires_at_epoch)} — 하트비트(progress)가 갱신하고,
# 만료분은 claim 처리 시 lazy 스윕으로 queued 복원한다.
_LEASES: dict[str, tuple[str, float]] = {}

# claim의 select→마킹을 직렬화 — 여러 워커가 동시에 폴링하면 같은 잡을 두 번 물 수 있다.
# 단일 프로세스 서버라 프로세스 내 락으로 충분하다 (sync.py의 _CREATE_LOCK과 같은 이유).
_CLAIM_LOCK = asyncio.Lock()


def _require_worker_key(x_worker_key: str | None) -> None:
    """워커 키 인증 — 키 미설정이면 기능 비활성(403), 불일치도 403."""
    worker_key = get_settings().server.worker_key
    if not worker_key or x_worker_key != worker_key:
        raise HTTPException(status_code=403, detail="워커 키가 필요해요")


def _lease_seconds() -> int:
    return max(1, get_settings().server.worker_lease_sec)


def _require_lease(job_id: str, worker_id: str | None) -> None:
    """이 잡의 리스를 이 워커가 쥐고 있는지 검증 — 아니면 409 (타 워커·만료 거부)."""
    lease = _LEASES.get(job_id)
    if lease is None or lease[0] != (worker_id or ""):
        raise HTTPException(status_code=409, detail="리스가 없거나 다른 워커 소유예요")


def _pop_stashes(job_id: str) -> None:
    """잡별 인메모리 스태시(발음/번역 메타·출처·강제) 정리 — 완료/실패/취소 시 (멱등)."""
    from everyric2.server.worker import (
        _PENDING_ATTRIBUTION,
        _PENDING_FORCE,
        _PENDING_LINE_META,
    )

    _PENDING_LINE_META.pop(job_id, None)
    _PENDING_ATTRIBUTION.pop(job_id, None)
    _PENDING_FORCE.discard(job_id)


async def _sweep_expired_leases() -> None:
    """만료 리스의 잡을 queued로 되돌리고 레지스트리에서 제거 (claim 시 lazy 호출).

    아직 processing인 잡만 복원한다 — 그 사이 완료/실패(취소 포함)했으면 그대로 둔다.
    스태시(line_meta 등)는 peek 방식이라 재클레임 시 다시 전달된다."""
    now = time.time()
    expired = [jid for jid, (_, exp) in _LEASES.items() if exp < now]
    if not expired:
        return
    async with get_session() as session:
        repo = JobRepository(session)
        for jid in expired:
            job = await repo.get_by_id(jid)
            if job and job.status == "processing":
                await repo.update_status(jid, "queued", progress=0)
    for jid in expired:
        _LEASES.pop(jid, None)


# ── 요청/응답 모델 ────────────────────────────────────────────────


class ClaimRequest(BaseModel):
    worker_id: str
    version: str


class WorkerJob(BaseModel):
    job_id: str
    video_id: str
    lyrics: str
    language: str | None = None
    line_meta: list[dict[str, Any]] | None = None
    attribution: dict[str, Any] | None = None
    force: bool = False
    max_audio_sec: int = 0


class ClaimResponse(BaseModel):
    job: WorkerJob | None = None
    lease_seconds: int = 0


class ProgressRequest(BaseModel):
    progress: int
    stage: str


class ProgressResponse(BaseModel):
    cancel_requested: bool = False


class CacheCheckRequest(BaseModel):
    audio_hash: str


class CacheCheckResponse(BaseModel):
    completed: bool = False


class ResultRequest(BaseModel):
    timestamps: list[dict[str, Any]]
    language: str | None = None
    quality_score: float | None = None
    audio_hash: str | None = None
    extra: dict[str, Any] | None = None


class FailRequest(BaseModel):
    error: str


class AcceptResponse(BaseModel):
    accepted: bool


# ── 엔드포인트 ────────────────────────────────────────────────────


@router.post("/claim", response_model=ClaimResponse)
async def claim_job(request: ClaimRequest, x_worker_key: str | None = Header(default=None)):
    """가장 오래된 queued 잡을 하나 물어 준다 (없으면 job=null).

    버전이 서버와 다르면 409 (스키마 어긋남 방지 — 워커 업데이트 유도). 만료 리스는
    먼저 스윕해 큐로 되돌린 뒤 선택한다. line_meta/attribution/force는 인메모리 스태시를
    peek(제거하지 않음)해서 싣는다 — 리스 만료로 재클레임되면 다시 전달돼야 하기 때문."""
    _require_worker_key(x_worker_key)
    if request.version != __version__:
        raise HTTPException(
            status_code=409,
            detail=(
                f"워커 버전({request.version})이 서버({__version__})와 달라요. "
                "워커를 업데이트해 주세요."
            ),
        )

    from everyric2.server.worker import _PENDING_ATTRIBUTION, _PENDING_FORCE, _PENDING_LINE_META

    async with _CLAIM_LOCK:
        await _sweep_expired_leases()
        lease_sec = _lease_seconds()
        async with get_session() as session:
            repo = JobRepository(session)
            job = await repo.get_oldest_queued()
            if not job:
                return ClaimResponse(job=None)
            await repo.update_status(job.id, "processing", progress=0, stage="워커 할당")
            payload = WorkerJob(
                job_id=job.id,
                video_id=job.video_id,
                lyrics=job.lyrics,
                language=job.language,
                line_meta=_PENDING_LINE_META.get(job.id),
                attribution=_PENDING_ATTRIBUTION.get(job.id),
                force=job.id in _PENDING_FORCE,
                max_audio_sec=get_settings().server.max_job_audio_sec,
            )
            job_id = job.id
        _LEASES[job_id] = (request.worker_id, time.time() + lease_sec)
    return ClaimResponse(job=payload, lease_seconds=lease_sec)


@router.post("/jobs/{job_id}/progress", response_model=ProgressResponse)
async def report_progress(
    job_id: str,
    request: ProgressRequest,
    x_worker_key: str | None = Header(default=None),
    x_worker_id: str | None = Header(default=None),
):
    """진행률·단계 보고(하트비트 겸) — 응답의 cancel_requested가 true면 워커는 경계에서
    포기하고 아무것도 제출하지 않는다.

    _set_progress를 경유해 취소 가드(failed↔processing 왕복 방지)를 재사용한다. 취소면
    스태시를 정리하고, 리스는 남겨 만료 스윕에 맡긴다 — 잡은 cancel API가 이미 failed로
    마킹했으므로 스윕이 queued로 되돌리지 않는다. (틱/모니터의 잦은 보고가 리스를 지워
    경계 progress가 리스를 잃는 것을 막으려면 여기서 리스를 건드리지 않아야 한다.)"""
    _require_worker_key(x_worker_key)
    _require_lease(job_id, x_worker_id)

    from everyric2.server.worker import _CANCEL_REQUESTED, _set_progress

    await _set_progress(job_id, request.progress, request.stage)
    if job_id in _CANCEL_REQUESTED:
        _pop_stashes(job_id)
        return ProgressResponse(cancel_requested=True)
    lease = _LEASES.get(job_id)
    if lease:
        _LEASES[job_id] = (lease[0], time.time() + _lease_seconds())
    return ProgressResponse(cancel_requested=False)


@router.post("/jobs/{job_id}/cache-check", response_model=CacheCheckResponse)
async def cache_check(
    job_id: str,
    request: CacheCheckRequest,
    x_worker_key: str | None = Header(default=None),
    x_worker_id: str | None = Header(default=None),
):
    """(audio_hash, lyrics) 캐시 완결 판정 — 인프로세스와 같은 _complete_from_cache_db 로직
    재사용(S1 교차 영상 캐시가 원격에서도 유지된다). completed=true면 워커는 정렬을 건너뛰고
    로컬 오디오를 지운다."""
    _require_worker_key(x_worker_key)
    _require_lease(job_id, x_worker_id)

    from everyric2.server.worker import _complete_from_cache_db

    async with get_session() as session:
        job = await JobRepository(session).get_by_id(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="잡을 찾을 수 없어요")
    completed = await _complete_from_cache_db(
        job_id, job, request.audio_hash, hash_lyrics(job.lyrics)
    )
    if completed:
        _LEASES.pop(job_id, None)
        _pop_stashes(job_id)
    return CacheCheckResponse(completed=completed)


@router.post("/jobs/{job_id}/result", response_model=AcceptResponse)
async def submit_result(
    job_id: str,
    request: ResultRequest,
    x_worker_key: str | None = Header(default=None),
    x_worker_id: str | None = Header(default=None),
):
    """정렬 결과 제출 → SyncResult 생성 + 잡 completed (인프로세스 저장 경로와 동일 데이터).

    status가 processing이고 리스 소유자일 때만 수락한다 — 취소된 잡(failed)·좀비 정리된
    잡의 뒤늦은 결과를 거부한다."""
    _require_worker_key(x_worker_key)
    _require_lease(job_id, x_worker_id)

    async with get_session() as session:
        job_repo = JobRepository(session)
        job = await job_repo.get_by_id(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="잡을 찾을 수 없어요")
        if job.status != "processing":
            _LEASES.pop(job_id, None)
            raise HTTPException(status_code=409, detail=f"잡이 이미 {job.status} 상태예요")
        sync_result = await SyncRepository(session).create(
            video_id=job.video_id,
            lyrics_hash=hash_lyrics(job.lyrics),
            timestamps=request.timestamps,
            language=request.language,
            engine="ctc",
            quality_score=request.quality_score,
            audio_hash=request.audio_hash,
            extra=request.extra,
        )
        await job_repo.update_status(
            job_id, "completed", progress=100, result_id=sync_result.id
        )
    _LEASES.pop(job_id, None)
    _pop_stashes(job_id)
    return AcceptResponse(accepted=True)


@router.post("/jobs/{job_id}/fail", response_model=AcceptResponse)
async def submit_fail(
    job_id: str,
    request: FailRequest,
    x_worker_key: str | None = Header(default=None),
    x_worker_id: str | None = Header(default=None),
):
    """워커 쪽 다운로드 실패·파이프라인 예외 보고 → 잡 failed. 메시지는 사용자에게 보이므로
    워커가 친절한 한국어 문구를 실어 보낸다.

    이미 취소(failed)된 잡의 "취소했어요" 문구를 덮어쓰지 않도록 processing일 때만 반영한다."""
    _require_worker_key(x_worker_key)
    _require_lease(job_id, x_worker_id)

    async with get_session() as session:
        job_repo = JobRepository(session)
        job = await job_repo.get_by_id(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="잡을 찾을 수 없어요")
        if job.status == "processing":
            await job_repo.update_status(job_id, "failed", error=request.error)
    _LEASES.pop(job_id, None)
    _pop_stashes(job_id)
    return AcceptResponse(accepted=True)
