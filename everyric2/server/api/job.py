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


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    async with get_session() as session:
        job_repo = JobRepository(session)
        job = await job_repo.get_by_id(job_id)

        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        response = JobStatusResponse(
            job_id=job.id,
            status=job.status,
            progress=job.progress,
            error=job.error,
        )

        if job.status == "completed" and job.result_id:
            sync_repo = SyncRepository(session)
            results = await sync_repo.get_by_video(job.video_id)
            if results:
                response.timestamps = results[0].timestamps.get("segments", [])

        return response
