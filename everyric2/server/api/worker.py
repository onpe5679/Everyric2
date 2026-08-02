"""원격 GPU 워커 풀 API — 워커가 아웃바운드 폴링으로 잡을 클레임·처리·제출한다.

서버는 API+DB+큐만 맡고, 생성 파이프라인(다운로드/분리/정렬/멜로디)은 원격 워커(집 PC
GPU 등)가 돌린다. 인증은 X-Worker-Key(EVERYRIC_SERVER_WORKER_KEY) 한 개를 공유하는
개인 풀 모델 — 워커는 worker_id로 구분한다. 리스(어느 워커가 어떤 잡을 무는지 + 만료)는
서버 인메모리 레지스트리로 관리해 Job 테이블 스키마를 건드리지 않는다 (서버 단일 프로세스
전제. 재시작 시 유실은 좀비 잡 정리(db/connection.py)가 이미 커버하는 동작이라 일관적).

잡은 두 종류다: sync(가사 싱크 생성, 기본) 우선, 없으면 link_validate(반주 상관으로 커버가
원곡과 같은 반주를 쓰는지 판정 → SyncLink 자동 생성). 리스 레지스트리는 공유하되 링크 잡은
키를 ``link:{id}``로 네임스페이스 분리한다.
"""

import asyncio
import contextlib
import logging
import os
import time
from typing import Any

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from everyric2 import __version__
from everyric2.config.settings import get_settings
from everyric2.server.db.connection import get_session
from everyric2.server.db.repository import (
    JobRepository,
    LinkJobRepository,
    SyncLinkRepository,
    SyncRepository,
    hash_lyrics,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/worker", tags=["worker"])

# 리스 레지스트리 {job_id: (worker_id, expires_at_epoch)} — 하트비트(progress)가 갱신하고,
# 만료분은 claim 처리 시 lazy 스윕으로 queued 복원한다. 링크 잡은 키가 "link:{id}"다.
_LEASES: dict[str, tuple[str, float]] = {}

# 서버가 미디어 캐시에서 추출한 워커 전달용 오디오 {job_id: local_path}. 워커 인증 뒤
# (GET /jobs/{job_id}/audio)에서만 서빙하고, 잡 터미널 지점에서 삭제한다 (저작권 규약).
_WORKER_AUDIO: dict[str, str] = {}

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


def _link_lease_seconds() -> int:
    """링크 잡 리스(초) — 처리가 짧아 진행 하트비트를 생략하므로 넉넉히 300s로 잡는다."""
    return 300


def _require_lease(lease_key: str, worker_id: str | None) -> None:
    """이 잡의 리스를 이 워커가 쥐고 있는지 검증 — 아니면 409 (타 워커·만료 거부).

    lease_key는 sync 잡이면 job_id, 링크 잡이면 "link:{id}"."""
    lease = _LEASES.get(lease_key)
    if lease is None or lease[0] != (worker_id or ""):
        raise HTTPException(status_code=409, detail="리스가 없거나 다른 워커 소유예요")


def _pop_stashes(job_id: str) -> None:
    """잡별 인메모리 스태시(발음/번역 메타·출처·강제·대기 예고) 정리 — 완료/실패/취소 시 (멱등)."""
    from everyric2.server.worker import (
        _PENDING_ATTRIBUTION,
        _PENDING_FORCE,
        _PENDING_LINE_META,
        _PENDING_LINE_META_LANG,
        _PENDING_META_WAIT,
        _PENDING_TITLE,
    )

    _PENDING_LINE_META.pop(job_id, None)
    _PENDING_LINE_META_LANG.pop(job_id, None)
    _PENDING_ATTRIBUTION.pop(job_id, None)
    _PENDING_TITLE.pop(job_id, None)
    _PENDING_FORCE.discard(job_id)
    # 대기 예고는 claim이 peek로 실어 보내므로(재클레임 대비) 터미널 지점에서만 비운다
    _PENDING_META_WAIT.discard(job_id)


def _peek_line_meta(job_id: str) -> list[dict[str, Any]] | None:
    """대기 중인 line_meta 스태시 조회 (제거하지 않음) — claim·하트비트의 단일 읽기 지점.

    **None과 빈 리스트는 다른 뜻이다**: None은 "아직 안 왔음"(워커는 상한까지 계속 기다린다),
    빈 리스트는 "붙일 것 없음 확정"(확장이 번역 실패를 알린 것 → 즉시 원문 정렬)이다.
    이 구분은 코어 스태시의 규약(_PENDING_LINE_META 주석)이고, 워커 API는 그대로 중계만 한다."""
    from everyric2.server.worker import _PENDING_LINE_META

    return _PENDING_LINE_META.get(job_id)


def _cleanup_worker_audio(job_id: str) -> None:
    """서버가 추출해 둔 워커 전달용 오디오 삭제 (잡 터미널·리스 만료 시, 멱등)."""
    from pathlib import Path

    path = _WORKER_AUDIO.pop(job_id, None)
    if path:
        Path(path).unlink(missing_ok=True)


async def _prepare_worker_audio(
    job_id: str, video_id: str, max_audio_sec: int
) -> tuple[str | None, str | None]:
    """sync 잡 확정 직후 미디어 캐시 프리플라이트 → (audio_url | None, fail_reason | None).

    히트면 추출 파일을 _WORKER_AUDIO에 등록하고 워커가 받을 audio 엔드포인트 경로를 준다
    (claim 락 밖에서 호출 — ffmpeg 추출이 다른 워커의 claim을 막지 않게)."""
    from everyric2.server.media_cache import prepare_cached_audio

    cache_path, fail_reason = await prepare_cached_audio(video_id, job_id, max_audio_sec)
    if fail_reason:
        return None, fail_reason
    if cache_path:
        _WORKER_AUDIO[job_id] = cache_path
        return f"/api/worker/jobs/{job_id}/audio", None
    return None, None


async def _sweep_expired_leases() -> None:
    """만료 리스의 잡을 queued로 되돌리고 레지스트리에서 제거 (claim + 주기 태스크가 호출).

    아직 processing인 잡만 복원한다 — 그 사이 완료/실패(취소 포함)했으면 그대로 둔다.
    스태시(line_meta 등)는 peek 방식이라 재클레임 시 다시 전달된다. 링크 잡("link:{id}")도
    같은 규약으로 되돌린다. 추출해 둔 워커 오디오도 함께 정리한다(재클레임 시 재추출)."""
    now = time.time()
    expired = [key for key, (_, exp) in _LEASES.items() if exp < now]
    if not expired:
        return
    async with get_session() as session:
        job_repo = JobRepository(session)
        link_repo = LinkJobRepository(session)
        for key in expired:
            if key.startswith("link:"):
                link_id = key[len("link:") :]
                link_job = await link_repo.get_by_id(link_id)
                if link_job and link_job.status == "processing":
                    await link_repo.update_status(link_id, "queued")
            else:
                job = await job_repo.get_by_id(key)
                if job and job.status == "processing":
                    await job_repo.update_status(key, "queued", progress=0)
    for key in expired:
        _LEASES.pop(key, None)
        if not key.startswith("link:"):
            _cleanup_worker_audio(key)


# ── 만료 리스 주기 스윕 ───────────────────────────────────────────
#
# 스윕이 claim에만 얹혀 있으면 **워커가 하나인 배포에서 그 워커가 죽는 순간 스윕이 영영
# 발화하지 않는다** — 이후 claim이 없으니 만료 리스를 훑는 주체가 없고, 잡은 processing에
# 영구 정착한다. 2차 피해가 더 크다: get_active_by_video가 pending/queued/processing을
# 활성으로 보므로 이후 같은 (video, lyrics) 요청이 죽은 잡에 합류해 새 잡을 만들지 않는다
# → 서버 재기동까지 그 가사로는 재생성이 불가능하다(그 영상이 봉인된다).
#
# 그래서 스윕을 claim에서 떼어 앱 lifespan의 주기 태스크로도 돌린다. claim 안의 lazy 스윕은
# 그대로 남긴다 — 다중 워커에서는 그쪽이 회수 지연을 0에 가깝게 줄인다.
#
# 조회 경로(get_active_by_video)에서 "리스 만료 잡은 활성으로 세지 않는" 대안은 채택하지
# 않았다: 리스는 원격 워커 잡에만 존재하고, **인프로세스 워커(local_worker=true)와
# _queue_after_line_meta의 "번역 대기" 구간은 리스 없이 정상적으로 processing**이다.
# 리스 부재를 사망으로 읽으면 살아 있는 그 잡들을 죽은 것으로 오판해 같은 영상에 중복 잡을
# 만들고, 두 잡이 같은 임시 오디오를 잡아 WinError 32를 부른다(합류 로직이 막으려던 바로 그
# 사고). 리스 만료는 리스를 아는 쪽에서만 판정한다.
#
# 간격: 회수 지연 상한이 이 간격이므로 리스 만료(worker_lease_sec, 기본 120s)보다 충분히
# 짧게 잡는다. 리스가 하나도 없으면 _sweep_expired_leases가 DB를 건드리지 않고 즉시
# 반환하므로 워커 풀을 쓰지 않는 배포에서는 비용이 사실상 0이다.
LEASE_SWEEP_INTERVAL_SEC = 20.0

_SWEEPER_TASK: asyncio.Task | None = None


async def _lease_sweep_loop() -> None:
    """LEASE_SWEEP_INTERVAL_SEC마다 만료 리스를 스윕한다 (취소로만 끝난다).

    한 번의 실패로 루프가 죽으면 위의 영구 processing 결함이 그대로 돌아오므로, 스윕
    예외는 로그로 남기고 다음 주기에 재시도한다 (취소는 그대로 전파해야 종료가 된다).
    첫 스윕 전에 한 번 기다리는 것은 의도다 — 기동 직후의 좀비는 init_db의 재기동 정리가
    이미 처리했고, 리스 레지스트리는 그 시점에 비어 있다.

    _CLAIM_LOCK을 잡지 않는다: 스윕은 **이미 만료된**(exp < now) 리스만 건드리고 claim은
    락 안에서 항상 방금 만든 리스를 넣으므로 둘이 같은 항목을 다툴 수 없다. 잡을 processing
    으로 마킹한 뒤 리스를 넣기 전의 짧은 창에도 그 잡은 레지스트리에 없어 스윕 대상이
    아니다. 반대로 락을 잡으면 DB가 느린 순간의 스윕이 워커 claim 전체를 막는다.
    """
    while True:
        await asyncio.sleep(LEASE_SWEEP_INTERVAL_SEC)
        try:
            await _sweep_expired_leases()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Lease sweep failed; retrying at the next interval")


def start_lease_sweeper() -> None:
    """주기 스윕 시작 — **앱 lifespan에서만 호출한다**. 멱등.

    임포트 시점에 태스크를 만들지 않는 것이 중요하다: 이 레포의 서버 테스트는 앱을 띄우지
    않고 라우트 코루틴을 직접 await하므로(httpx 미사용 규약), 임포트만으로 태스크가 뜨면
    실행 중인 루프가 없어 실패하거나 테스트 루프에 매달린 태스크가 남는다.
    """
    global _SWEEPER_TASK
    if _SWEEPER_TASK is None or _SWEEPER_TASK.done():
        _SWEEPER_TASK = asyncio.create_task(_lease_sweep_loop())


async def stop_lease_sweeper() -> None:
    """주기 스윕 취소 + 종료 확인 — lifespan 종료에서 반드시 호출한다 (태스크 누수 금지)."""
    global _SWEEPER_TASK
    task = _SWEEPER_TASK
    _SWEEPER_TASK = None
    if task is None:
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def reclaim_expired_leases() -> None:
    """만료 리스 회수 — 생성 경로(api/sync)가 활성 잡을 보기 **전에** 부르는 공개 진입점.

    주기 태스크와 **같은 함수**를 호출한다. 사망 판정 규칙을 두 곳에 각각 두면 언젠가
    갈라지고, 갈라진 쪽이 살아 있는 잡을 죽었다고 부르는 순간 중복 잡이 생긴다.

    이것이 ①의 두 번째 방어선이며 **이벤트 루프에 의존하지 않는다**: 주기 태스크가 아직
    돌지 않았거나(간격 이내) 어떤 이유로든 죽었더라도, 죽은 워커의 잡이 그 (영상, 가사)의
    재생성을 막고 있으면 생성 요청 자체가 그것을 회수한다.

    "리스가 만료된 잡을 활성 목록에서 **제외**"하는 방식을 쓰지 않은 이유: 제외만 하면 죽은
    잡은 여전히 processing으로 남은 채 새 잡이 하나 더 생기고, 나중에 스윕이 죽은 잡을
    queued로 되돌리는 순간 같은 (영상, 가사) 잡이 둘 돌게 된다. 회수를 먼저 하면 그 잡이
    queued가 되어 요청은 **살아 있는 그 잡에 합류**한다 — 봉인은 풀리고 중복은 없다.
    리스가 하나도 만료돼 있지 않으면 DB를 건드리지 않고 즉시 반환한다.
    """
    await _sweep_expired_leases()


# ── 요청/응답 모델 ────────────────────────────────────────────────


class ClaimRequest(BaseModel):
    worker_id: str
    version: str
    # 이 워커가 하트비트 응답으로 늦은 line_meta를 받아 정렬 직전에 기다릴 수 있는지.
    # **버전 문자열로는 이 능력을 알 수 없다** — __version__은 릴리스마다 올라가지 않아
    # (계속 "0.1.0") 구버전 코드를 돌리는 워커도 버전 게이트를 그대로 통과한다. 그래서
    # 능력은 능력으로 물어본다: 안 보내는 워커에게는 대기를 지시하지 않고(기존 동작 =
    # 원문 정렬) 서버가 경고를 남겨, 조용한 품질 저하가 로그에 드러나게 한다.
    supports_line_meta_heartbeat: bool = False


class WorkerJob(BaseModel):
    job_id: str
    video_id: str
    lyrics: str
    language: str | None = None
    line_meta: list[dict[str, Any]] | None = None
    attribution: dict[str, Any] | None = None
    force: bool = False
    max_audio_sec: int = 0
    # 서버 미디어 캐시 히트 시 워커가 yt-dlp 대신 받아 갈 오디오 경로 (없으면 yt-dlp).
    audio_url: str | None = None
    # "line_meta(번역·독음)가 이 잡에는 나중에 온다" — 워커는 다운로드·보컬 분리를 먼저 돌리고
    # 정렬 진입 직전에 상한을 둔 대기를 한 번 넣는다(하트비트 응답으로 받는다). line_meta가
    # 이미 실려 있으면 코어가 대기를 만들지 않으므로 함께 와도 무해하다.
    await_line_meta: bool = False


class WorkerLinkJob(BaseModel):
    link_job_id: str
    video_id: str
    source_video_id: str
    max_audio_sec: int = 0


class ClaimResponse(BaseModel):
    # kind="sync"면 job이, "link_validate"면 link_job이 채워진다 (기본 sync — 구버전 워커는
    # 버전 게이트로 이미 차단되므로 호환 부담 없음).
    kind: str = "sync"
    job: WorkerJob | None = None
    link_job: WorkerLinkJob | None = None
    lease_seconds: int = 0


class ProgressRequest(BaseModel):
    progress: int
    stage: str
    # 늦게 도착하는 line_meta(+출처)를 이 응답에 실어 달라는 요청 — 원격 워커의 번역 병렬 경로.
    # 워커는 받기 전까지만 켠다: 2초마다 오는 하트비트에 번역 전문을 매번 되돌려 보내지 않기
    # 위해서다. 이 필드를 모르는 구버전 워커는 보내지 않으므로 기본값에서 기존 동작 그대로다.
    want_line_meta: bool = False


class ProgressResponse(BaseModel):
    cancel_requested: bool = False
    # want_line_meta를 켠 하트비트에만 채운다. **None과 빈 리스트의 뜻이 다르다** —
    # None은 "아직 안 왔음"(워커는 상한까지 더 기다린다), 빈 리스트는 "붙일 것 없음 확정"
    # (워커는 즉시 원문 정렬로 간다). 서버는 스태시 값을 그대로 중계하고 판단은 코어가 한다.
    line_meta: list[dict[str, Any]] | None = None
    # line_meta와 함께 붙는 가사 출처 표기 — 같은 attach 호출로 오므로 같은 채널로 보낸다.
    # 원격 워커는 extra["attribution"]을 자기 쪽에서 조립하므로, 이걸 안 보내면 클레임 뒤에
    # 도착한 출처가 결과에서 조용히 사라진다 (제목·아티스트는 서버가 저장 시점에 붙여서 무관).
    attribution: dict[str, Any] | None = None


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


class LinkResultRequest(BaseModel):
    match: bool
    offset_sec: float
    confidence: float


class FailRequest(BaseModel):
    error: str
    # sync 잡(/jobs/{id}/fail) 전용 — jobs.failure_kind (MoRef 감사 #3). 원격 워커(cli.py)가
    # classify_job_failure로 계산해 실어 보낸다. "cancelled"는 이 경로로 오지 않는다(취소는
    # cancel API가 서버 쪽에서 이미 확정한다 — report_progress의 cancel_requested 참고).
    # 구버전 워커는 이 필드를 안 보내므로 기본값 None(미분류)으로 남는다.
    failure_kind: str | None = None
    # link-jobs(/link-jobs/{id}/fail) 전용 — True면 오류가 아니라 무다운로드 원칙에 따른
    # 정책적 종결(cache_miss_no_download 등, MoRef 감사 #4). sync 잡 쪽은 이 필드를 읽지
    # 않는다(항상 기본값 False로 무해하게 무시됨).
    declined: bool = False


class AcceptResponse(BaseModel):
    accepted: bool


# ── 엔드포인트 ────────────────────────────────────────────────────


@router.post("/claim", response_model=ClaimResponse)
async def claim_job(request: ClaimRequest, x_worker_key: str | None = Header(default=None)):
    """가장 오래된 queued 잡을 하나 물어 준다 (없으면 job=null).

    sync 잡 우선, 없으면 link 잡을 FIFO로 준다. 버전이 서버와 다르면 409. 만료 리스는 먼저
    스윕해 큐로 되돌린 뒤 선택한다. sync 잡은 확정 직후(claim 락 밖) 미디어 캐시를 조회해
    히트면 audio_url을, 과길이면 즉시 실패시키고 job=null을 돌려준다. line_meta 등 스태시는
    peek(제거하지 않음)해 재클레임 시 다시 전달되게 한다.

    line_meta가 아직 없는 예고 잡(_PENDING_META_WAIT)은 await_line_meta=true로 실어 보낸다 —
    워커는 다운로드·보컬 분리를 먼저 돌리고 정렬 직전 대기에서 하트비트로 받는다."""
    _require_worker_key(x_worker_key)
    if request.version != __version__:
        raise HTTPException(
            status_code=409,
            detail=(
                f"워커 버전({request.version})이 서버({__version__})와 달라요. "
                "워커를 업데이트해 주세요."
            ),
        )

    from everyric2.server.worker import (
        _PENDING_ATTRIBUTION,
        _PENDING_FORCE,
        _PENDING_META_WAIT,
    )

    max_audio_sec = get_settings().server.max_job_audio_sec
    sync_job_id: str | None = None
    sync_video_id: str | None = None
    sync_payload: WorkerJob | None = None
    link_payload: WorkerLinkJob | None = None

    async with _CLAIM_LOCK:
        await _sweep_expired_leases()
        lease_sec = _lease_seconds()
        async with get_session() as session:
            repo = JobRepository(session)
            job = await repo.get_oldest_queued()
            if job:
                await repo.update_status(job.id, "processing", progress=0, stage="워커 할당")
                sync_payload = WorkerJob(
                    job_id=job.id,
                    video_id=job.video_id,
                    lyrics=job.lyrics,
                    language=job.language,
                    line_meta=_peek_line_meta(job.id),
                    attribution=_PENDING_ATTRIBUTION.get(job.id),
                    force=job.id in _PENDING_FORCE,
                    max_audio_sec=max_audio_sec,
                    await_line_meta=(
                        job.id in _PENDING_META_WAIT and request.supports_line_meta_heartbeat
                    ),
                )
                if job.id in _PENDING_META_WAIT and not request.supports_line_meta_heartbeat:
                    # 이 잡은 늦은 번역·독음을 기다려야 하는데 이 워커는 받을 수 없다 →
                    # 원문 정렬로 떨어져 독음 정렬 품질을 잃는다. 조용히 넘기지 않고 남긴다.
                    logger.warning(
                        "Job %s awaits line_meta but worker %s cannot receive it over the "
                        "heartbeat; it will align on the original text. Update the worker.",
                        job.id,
                        request.worker_id,
                    )
                sync_job_id = job.id
                sync_video_id = job.video_id
        if sync_payload is not None:
            _LEASES[sync_job_id] = (request.worker_id, time.time() + lease_sec)
        else:
            # sync 잡이 없으면 link 잡을 FIFO로 클레임한다
            async with get_session() as session:
                link_repo = LinkJobRepository(session)
                link_job = await link_repo.get_oldest_queued()
                if link_job:
                    await link_repo.update_status(link_job.id, "processing")
                    link_payload = WorkerLinkJob(
                        link_job_id=link_job.id,
                        video_id=link_job.video_id,
                        source_video_id=link_job.source_video_id,
                        max_audio_sec=max_audio_sec,
                    )
                    link_lease_key = f"link:{link_job.id}"
            if link_payload is not None:
                _LEASES[link_lease_key] = (request.worker_id, time.time() + _link_lease_seconds())

    # 락 밖: sync 잡이면 미디어 캐시 프리플라이트(조회+ffmpeg 추출은 전역 Semaphore(1))
    if sync_payload is not None:
        audio_url, fail_reason = await _prepare_worker_audio(
            sync_job_id, sync_video_id, max_audio_sec
        )
        if fail_reason:
            # 과길이 — 다운로드 없이 즉시 실패시키고 이 잡은 건너뛴다 (워커가 다시 폴링)
            async with get_session() as session:
                await JobRepository(session).update_status(sync_job_id, "failed", error=fail_reason)
            _LEASES.pop(sync_job_id, None)
            _cleanup_worker_audio(sync_job_id)
            _pop_stashes(sync_job_id)
            return ClaimResponse(kind="sync", job=None)
        sync_payload.audio_url = audio_url
        return ClaimResponse(kind="sync", job=sync_payload, lease_seconds=lease_sec)

    if link_payload is not None:
        return ClaimResponse(
            kind="link_validate", link_job=link_payload, lease_seconds=_link_lease_seconds()
        )
    return ClaimResponse(kind="sync", job=None)


@router.get("/jobs/{job_id}/audio")
async def get_job_audio(
    job_id: str,
    x_worker_key: str | None = Header(default=None),
    x_worker_id: str | None = Header(default=None),
):
    """서버가 미디어 캐시에서 추출한 워커 전달용 오디오 — 리스 소유 워커만(409). FileResponse.

    외부 재서빙 없음: 워커 인증(X-Worker-Key) + 리스 소유 뒤에만 존재한다 (저작권 규약)."""
    _require_worker_key(x_worker_key)
    _require_lease(job_id, x_worker_id)
    path = _WORKER_AUDIO.get(job_id)
    if not path or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="캐시 오디오가 없어요")
    return FileResponse(path, media_type="audio/mp4", filename=f"{job_id}.m4a")


@router.post("/jobs/{job_id}/progress", response_model=ProgressResponse)
async def report_progress(
    job_id: str,
    request: ProgressRequest,
    x_worker_key: str | None = Header(default=None),
    x_worker_id: str | None = Header(default=None),
):
    """진행률·단계 보고(하트비트 겸) — 응답의 cancel_requested가 true면 워커는 경계에서
    포기하고 아무것도 제출하지 않는다.

    **이 응답은 서버→워커 역방향 채널을 겸한다.** 원격 워커는 claim 시점의 스태시 스냅샷만
    받으므로, 클레임 뒤에 도착한 line_meta(번역·독음)와 출처를 알 방법이 이것뿐이다 —
    want_line_meta를 켠 하트비트에 스태시 값을 그대로 실어 보내면, 워커 쪽 코어가 정렬 진입
    직전 대기에서 그 값을 집어 독음 정렬로 진행한다 (아직 안 옴=None / 없음 확정=빈 리스트).
    대기 구간에도 _stage_monitor가 2초마다 이 엔드포인트를 치므로 리스가 계속 갱신된다 —
    갱신이 끊기면 만료 스윕이 잡을 회수해 다른 워커에게 넘기므로, 대기 중 하트비트는 이
    경로의 필수 전제다.

    _set_progress를 경유해 취소 가드(failed↔processing 왕복 방지)를 재사용한다. 취소면
    스태시를 정리하고(더 실어 보낼 것도 없다), 리스는 남겨 만료 스윕에 맡긴다 — 잡은 cancel
    API가 이미 failed로 마킹했으므로 스윕이 queued로 되돌리지 않는다. (틱/모니터의 잦은 보고가
    리스를 지워 경계 progress가 리스를 잃는 것을 막으려면 여기서 리스를 건드리지 않아야 한다.)"""
    _require_worker_key(x_worker_key)
    _require_lease(job_id, x_worker_id)

    from everyric2.server.worker import _CANCEL_REQUESTED, _PENDING_ATTRIBUTION, _set_progress

    await _set_progress(job_id, request.progress, request.stage)
    if job_id in _CANCEL_REQUESTED:
        _pop_stashes(job_id)
        return ProgressResponse(cancel_requested=True)
    lease = _LEASES.get(job_id)
    if lease:
        _LEASES[job_id] = (lease[0], time.time() + _lease_seconds())
    if not request.want_line_meta:
        return ProgressResponse(cancel_requested=False)
    return ProgressResponse(
        cancel_requested=False,
        line_meta=_peek_line_meta(job_id),
        attribution=_PENDING_ATTRIBUTION.get(job_id),
    )


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
        _cleanup_worker_audio(job_id)
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
        from everyric2.server.worker import (
            layer_origin,
            peek_attribution,
            peek_title,
            record_translation_layer,
            resolve_layer_lang,
            translation_layer_lines,
        )

        # 인프로세스 저장 경로(_process_job_inner)와 같은 번역 언어 분리 — 프로덕션은
        # 이 원격 워커 경로로 생성되므로 여기 없으면 새 싱크의 레이어가 영영 안 남고,
        # 비ko 번역이 legacy 슬롯에 실려 한국어 사용자가 남의 언어를 받는다.
        # 판정 기준은 요청자 언어가 아니라 세그에 실린 번역의 언어다(resolve_layer_lang).
        # 스태시는 아래 _pop_stashes 전까지 살아 있으므로 여기서 읽을 수 있다.
        meta_lang = resolve_layer_lang(job, job_id)
        job_attr = peek_attribution(job_id)
        await record_translation_layer(
            session,
            job.video_id,
            [s.get("text") or "" for s in request.timestamps],
            translation_layer_lines(request.timestamps),
            meta_lang,
            origin=layer_origin(job_attr),
            attribution=job_attr,
        )
        if meta_lang != "ko":
            for seg in request.timestamps:
                seg.pop("translation", None)

        title, artist = peek_title(job_id)
        sync_result = await SyncRepository(session).create(
            video_id=job.video_id,
            lyrics_hash=hash_lyrics(job.lyrics),
            timestamps=request.timestamps,
            language=request.language,
            engine="ctc",
            quality_score=request.quality_score,
            audio_hash=request.audio_hash,
            extra=request.extra,
            title=title,
            artist=artist,
        )
        await job_repo.update_status(
            job_id, "completed", progress=100, result_id=sync_result.id
        )
    _LEASES.pop(job_id, None)
    _cleanup_worker_audio(job_id)
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
            await job_repo.update_status(
                job_id, "failed", error=request.error, failure_kind=request.failure_kind
            )
    _LEASES.pop(job_id, None)
    _cleanup_worker_audio(job_id)
    _pop_stashes(job_id)
    return AcceptResponse(accepted=True)


# ── 링크 검증 잡 결과 ─────────────────────────────────────────────


@router.post("/link-jobs/{link_job_id}/result", response_model=AcceptResponse)
async def submit_link_result(
    link_job_id: str,
    request: LinkResultRequest,
    x_worker_key: str | None = Header(default=None),
    x_worker_id: str | None = Header(default=None),
):
    """반주 상관 판정 결과 제출. status·리스 소유 검증은 sync 잡 규약 준용(리스 키 "link:{id}").

    match=true면 SyncLink를 자동 생성한다. offset 부호 규약: 워커의 correlate_offset이 돌려준
    offset_sec은 t_cover - t_source다 — GET /api/sync/{video_id}가 소스 타임스탬프를
    ``t / rate + offset``으로 사상(sync.py _shift_time)하므로, video_id=커버, source=원곡으로
    그대로 upsert하면 커버 재생 시점에 정확히 맞는다 (test가 이 부호를 못 박는다)."""
    _require_worker_key(x_worker_key)
    lease_key = f"link:{link_job_id}"
    _require_lease(lease_key, x_worker_id)

    async with get_session() as session:
        repo = LinkJobRepository(session)
        link_job = await repo.get_by_id(link_job_id)
        if not link_job:
            raise HTTPException(status_code=404, detail="링크 잡을 찾을 수 없어요")
        if link_job.status != "processing":
            _LEASES.pop(lease_key, None)
            raise HTTPException(status_code=409, detail=f"링크 잡이 이미 {link_job.status} 상태예요")
        await repo.mark_done(
            link_job_id, request.match, request.offset_sec, request.confidence
        )
        if request.match:
            # 반주 상관 게이트를 통과한 자동 링크만 verified=True — 수동 링크와 구분된다
            await SyncLinkRepository(session).upsert(
                link_job.video_id,
                link_job.source_video_id,
                request.offset_sec,
                rate=1.0,
                verified=True,
            )
    _LEASES.pop(lease_key, None)
    return AcceptResponse(accepted=True)


@router.post("/link-jobs/{link_job_id}/fail", response_model=AcceptResponse)
async def submit_link_fail(
    link_job_id: str,
    request: FailRequest,
    x_worker_key: str | None = Header(default=None),
    x_worker_id: str | None = Header(default=None),
):
    """링크 잡 실패/거절 보고 → status=failed(오류) 또는 declined(request.declined=True,
    무다운로드 정책 종결 — MoRef 감사 #4). processing일 때만 반영(뒤늦은/중복 보고 무시)."""
    _require_worker_key(x_worker_key)
    lease_key = f"link:{link_job_id}"
    _require_lease(lease_key, x_worker_id)

    async with get_session() as session:
        repo = LinkJobRepository(session)
        link_job = await repo.get_by_id(link_job_id)
        if not link_job:
            raise HTTPException(status_code=404, detail="링크 잡을 찾을 수 없어요")
        if link_job.status == "processing":
            if request.declined:
                await repo.mark_declined(link_job_id, request.error)
            else:
                await repo.mark_failed(link_job_id, request.error)
    _LEASES.pop(lease_key, None)
    return AcceptResponse(accepted=True)
