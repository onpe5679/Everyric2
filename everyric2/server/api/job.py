import re
import statistics
import time
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from everyric2.server.db.connection import get_session
from everyric2.server.db.repository import JobMetricRepository, JobRepository, SyncRepository

router = APIRouter(prefix="/api/job", tags=["job"])

# job_id는 서버가 발급한 UUID — 형식 밖 문자열이 쿼리 파라미터로 흐르지 않게 한다
_JOB_ID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")

# ETA 산출의 최소 결과치 — median - elapsed가 0 이하(이미 중앙값을 넘겨 처리 중)여도
# "곧 끝나요"를 5초 밑으로는 내리지 않는다(팀리드 지시: floored at 5). 0을 주면 확장
# 진행 UI가 "0초 남음"과 "끝났는데 응답이 안 옴"을 구분 못 한다.
_ETA_FLOOR_SEC = 5

# depth별 median duration 캐시 — GET /api/job/{id}는 폴링(대개 1~2s 간격)되므로 매 요청마다
# JobMetric 20건을 다시 긁으면 낭비다. 30초 TTL이면 새 JobMetric 반영이 최대 30초 늦어질
# 뿐이고(ETA는 애초에 근사치), 그 대가로 폴링 요청의 절대다수가 DB를 안 친다.
_ETA_CACHE_TTL_SEC = 30.0
_ETA_MEDIAN_CACHE: dict[str, tuple[float, float]] = {}


def _validate_job_id(job_id: str) -> None:
    if not _JOB_ID_RE.match(job_id):
        raise HTTPException(status_code=422, detail="invalid job_id")


async def _median_duration(session, depth: str | None) -> float | None:
    """최근 20건 duration_sec의 중앙값 — depth를 주면 그 깊이만, None이면 전체(all-depth
    폴백/대기열 ETA). 데이터가 없으면 None(호출부가 eta 필드를 비운다)."""
    cache_key = depth or "*all*"
    now = time.monotonic()
    cached = _ETA_MEDIAN_CACHE.get(cache_key)
    if cached is not None and now - cached[0] < _ETA_CACHE_TTL_SEC:
        return cached[1]
    durations = await JobMetricRepository(session).recent_durations(depth, limit=20)
    if not durations:
        return None
    value = statistics.median(durations)
    _ETA_MEDIAN_CACHE[cache_key] = (now, value)
    return value


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: int = 0
    timestamps: list[dict[str, Any]] | None = None
    error: str | None = None
    # 현재 진행 단계명 + 단계 내 진행률(%) — 확장 진행 칩 표시용
    stage: str | None = None
    stage_progress: int | None = None
    # 대기열 순번 (1 = 다음 차례) — 처리 슬롯이 다 차서 status=queued일 때만
    queue_position: int | None = None
    # 진행 중인 분석 깊이(fast/medium/heavy, 확장의 "heavy로 전환" 배지 재료) — 인프로세스
    # 워커가 처리 중일 때만 채워진다(worker.peek_job_depth, additive). 원격 워커 처리 중인
    # 잡·라우팅이 아직 안 끝난 순간·레거시 스택은 None.
    depth: str | None = None
    # 처리 중 잡의 예상 잔여 시간(초) — 같은 depth의 최근 20건 median에서 경과 시간을 뺀
    # 값(데이터 없으면 all-depth 폴백, 그마저 없으면 None). 최소 5초로 바닥(_ETA_FLOOR_SEC).
    eta_sec: int | None = None
    # 대기열(queued) 잡의 예상 대기 시간(초) — queue_position × all-depth median. median
    # 데이터가 아직 없으면(서버 갓 기동 등) None.
    queue_eta_sec: int | None = None


@router.post("/{job_id}/cancel")
async def cancel_job(job_id: str):
    """진행/대기 중 잡 취소 — 워커가 단계 경계에서 확인해 중단한다.

    이미 도는 CTC/demucs 스레드는 즉시 멈추지 못하므로 다음 경계(다운로드 후·정렬 전·
    저장 전)에서 끊긴다. 대기열(queued) 잡은 슬롯을 잡는 즉시 놓아준다. 끝난 잡은 그대로."""
    _validate_job_id(job_id)
    from everyric2.server.worker import request_cancel

    async with get_session() as session:
        job_repo = JobRepository(session)
        job = await job_repo.get_by_id(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.status in ("pending", "queued", "processing"):
            request_cancel(job_id)
            await job_repo.update_status(
                job_id, "failed", error="요청으로 취소했어요", failure_kind="cancelled"
            )
            return {"job_id": job_id, "cancelled": True}
        return {"job_id": job_id, "cancelled": False, "status": job.status}


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    from everyric2.server.worker import peek_job_depth, peek_processing_elapsed, STAGE_WINDOWS

    _validate_job_id(job_id)
    async with get_session() as session:
        job_repo = JobRepository(session)
        job = await job_repo.get_by_id(job_id)

        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        # 단계별 퍼센트 = 전역 진행률을 단계 창(lo,hi) 기준으로 환산
        stage = job.stage if job.status == "processing" else None
        stage_progress = None
        if stage and stage in STAGE_WINDOWS:
            lo, hi = STAGE_WINDOWS[stage]
            if hi > lo:
                stage_progress = max(0, min(100, round((job.progress - lo) * 100 / (hi - lo))))

        queue_position = None
        if job.status == "queued":
            queue_position = await job_repo.count_queued_before(job.created_at, exclude_id=job.id) + 1

        depth: str | None = None
        eta_sec: int | None = None
        queue_eta_sec: int | None = None
        if job.status == "processing":
            # 원격 워커가 처리 중인 잡은 peek_job_depth가 항상 None을 준다(라이브 깊이는
            # 이 서버 프로세스에서 정렬이 실제로 돌 때만 안다) — eta는 그럴 때 all-depth
            # 폴백으로 그대로 낸다(스코프: worker.peek_job_depth 독스트링).
            depth = peek_job_depth(job.id)
            median = await _median_duration(session, depth) if depth else None
            if median is None:
                median = await _median_duration(session, None)
            if median is not None:
                elapsed = peek_processing_elapsed(job.id) or 0.0
                eta_sec = max(_ETA_FLOOR_SEC, round(median - elapsed))
        elif job.status == "queued" and queue_position:
            median_all = await _median_duration(session, None)
            if median_all is not None:
                queue_eta_sec = round(queue_position * median_all)

        response = JobStatusResponse(
            job_id=job.id,
            status=job.status,
            progress=job.progress,
            error=job.error,
            stage=stage,
            stage_progress=stage_progress,
            queue_position=queue_position,
            depth=depth,
            eta_sec=eta_sec,
            queue_eta_sec=queue_eta_sec,
        )

        if job.status == "completed" and job.result_id:
            sync_repo = SyncRepository(session)
            # **이 잡이 만든 싱크**를 돌려준다. 최신 싱크를 돌려주면 그 사이 같은 영상에 다른
            # 가사로 만들어진 싱크가 이 잡의 결과로 나가 계약이 틀린다 (result_id를 저장해
            # 두고도 무시하던 동작).
            result = await sync_repo.get_by_id(job.result_id)
            if result is None:
                # 그 행이 사라졌을 때만 최신으로 폴백한다 — DELETE /api/sync/{video_id}
                # (싱크 초기화)는 sync_results만 지우고 completed 잡 행은 남기므로,
                # result_id가 죽은 id를 가리키는 상태가 실제로 생긴다. 되살릴 수 없는 id
                # 하나 때문에 응답을 비우는 것보다 같은 영상의 최신 싱크를 주는 편이 낫다
                # (기존 동작과 동일한 폴백이라 회귀도 없다).
                results = await sync_repo.get_by_video(job.video_id)
                result = results[0] if results else None
            if result is not None:
                response.timestamps = result.timestamps.get("segments", [])

        return response
