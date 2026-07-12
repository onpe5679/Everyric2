from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from everyric2.server.db.connection import get_session
from everyric2.server.db.repository import JobRepository, SyncRepository

router = APIRouter(prefix="/api/job", tags=["job"])


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: int = 0
    timestamps: list[dict[str, Any]] | None = None
    error: str | None = None
    # 현재 진행 단계명 + 단계 내 진행률(%) — 확장 진행 칩 표시용
    stage: str | None = None
    stage_progress: int | None = None


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    from everyric2.server.worker import STAGE_WINDOWS

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

        response = JobStatusResponse(
            job_id=job.id,
            status=job.status,
            progress=job.progress,
            error=job.error,
            stage=stage,
            stage_progress=stage_progress,
        )

        if job.status == "completed" and job.result_id:
            sync_repo = SyncRepository(session)
            results = await sync_repo.get_by_video(job.video_id)
            if results:
                response.timestamps = results[0].timestamps.get("segments", [])

        return response
