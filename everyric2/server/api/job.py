import re
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from everyric2.server.db.connection import get_session
from everyric2.server.db.repository import JobRepository, SyncRepository

router = APIRouter(prefix="/api/job", tags=["job"])

# job_id는 서버가 발급한 UUID — 형식 밖 문자열이 쿼리 파라미터로 흐르지 않게 한다
_JOB_ID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")


def _validate_job_id(job_id: str) -> None:
    if not _JOB_ID_RE.match(job_id):
        raise HTTPException(status_code=422, detail="invalid job_id")


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
            await job_repo.update_status(job_id, "failed", error="요청으로 취소했어요")
            return {"job_id": job_id, "cancelled": True}
        return {"job_id": job_id, "cancelled": False, "status": job.status}


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    from everyric2.server.worker import STAGE_WINDOWS

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

        response = JobStatusResponse(
            job_id=job.id,
            status=job.status,
            progress=job.progress,
            error=job.error,
            stage=stage,
            stage_progress=stage_progress,
            queue_position=queue_position,
        )

        if job.status == "completed" and job.result_id:
            sync_repo = SyncRepository(session)
            results = await sync_repo.get_by_video(job.video_id)
            if results:
                response.timestamps = results[0].timestamps.get("segments", [])

        return response
