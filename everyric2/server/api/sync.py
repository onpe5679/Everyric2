from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from everyric2.server.db.connection import get_session
from everyric2.server.db.repository import JobRepository, SyncRepository, hash_lyrics

router = APIRouter(prefix="/api/sync", tags=["sync"])


class SyncLookupResponse(BaseModel):
    found: bool
    sync_id: str | None = None
    timestamps: list[dict[str, Any]] | None = None
    lyrics_source: str | None = None
    quality_score: float | None = None
    audio_hash: str | None = None
    language: str | None = None
    created_at: str | None = None


class GenerateRequest(BaseModel):
    video_id: str
    lyrics: str
    lyrics_source: str = "user_input"
    language: str | None = None


class GenerateResponse(BaseModel):
    job_id: str
    status: str
    estimated_time: int = 15


class SearchByAudioRequest(BaseModel):
    audio_hash: str


class CopySyncRequest(BaseModel):
    source_video_id: str
    target_video_id: str
    lyrics: str | None = None


class RegenerateRequest(BaseModel):
    video_id: str
    lyrics: str
    language: str | None = None
    force: bool = False


@router.get("/{video_id}", response_model=SyncLookupResponse)
async def get_sync(video_id: str, lyrics_hash: str | None = None):
    async with get_session() as session:
        repo = SyncRepository(session)

        if lyrics_hash:
            result = await repo.get_by_video_and_hash(video_id, lyrics_hash)
            if result:
                return SyncLookupResponse(
                    found=True,
                    sync_id=result.id,
                    timestamps=result.timestamps.get("segments", []),
                    lyrics_source=result.engine,
                    quality_score=result.quality_score,
                    audio_hash=result.audio_hash,
                    language=result.language,
                    created_at=result.created_at.isoformat() if result.created_at else None,
                )
        else:
            results = await repo.get_by_video(video_id)
            if results:
                result = results[0]
                return SyncLookupResponse(
                    found=True,
                    sync_id=result.id,
                    timestamps=result.timestamps.get("segments", []),
                    lyrics_source=result.engine,
                    quality_score=result.quality_score,
                    audio_hash=result.audio_hash,
                    language=result.language,
                    created_at=result.created_at.isoformat() if result.created_at else None,
                )

        return SyncLookupResponse(found=False)


@router.post("/generate", response_model=GenerateResponse)
async def generate_sync(request: GenerateRequest, background_tasks: BackgroundTasks):
    lyrics_hash_value = hash_lyrics(request.lyrics)

    async with get_session() as session:
        sync_repo = SyncRepository(session)
        existing = await sync_repo.get_by_video_and_hash(request.video_id, lyrics_hash_value)
        if existing:
            return GenerateResponse(
                job_id=existing.id,
                status="completed",
                estimated_time=0,
            )

        job_repo = JobRepository(session)
        job = await job_repo.create(
            video_id=request.video_id,
            lyrics=request.lyrics,
            language=request.language,
        )
        job_id = job.id

    from everyric2.server.worker import process_job

    background_tasks.add_task(process_job, job_id)

    return GenerateResponse(
        job_id=job_id,
        status="processing",
        estimated_time=15,
    )


@router.post("/search-by-audio", response_model=SyncLookupResponse)
async def search_by_audio_hash(request: SearchByAudioRequest):
    async with get_session() as session:
        repo = SyncRepository(session)
        result = await repo.get_by_audio_hash(request.audio_hash)
        if result:
            return SyncLookupResponse(
                found=True,
                sync_id=result.id,
                timestamps=result.timestamps.get("segments", []),
                lyrics_source=result.engine,
                quality_score=result.quality_score,
                audio_hash=result.audio_hash,
                language=result.language,
                created_at=result.created_at.isoformat() if result.created_at else None,
            )
        return SyncLookupResponse(found=False)


@router.get("/list/{video_id}")
async def list_syncs_for_video(video_id: str):
    async with get_session() as session:
        repo = SyncRepository(session)
        results = await repo.get_by_video(video_id)
        return {
            "video_id": video_id,
            "syncs": [
                {
                    "sync_id": r.id,
                    "lyrics_hash": r.lyrics_hash,
                    "audio_hash": r.audio_hash,
                    "quality_score": r.quality_score,
                    "language": r.language,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in results
            ],
        }


class SearchSyncRequest(BaseModel):
    title: str | None = None
    artist: str | None = None
    limit: int = 10


@router.post("/search")
async def search_available_syncs(request: SearchSyncRequest):
    async with get_session() as session:
        repo = SyncRepository(session)
        results = await repo.get_all_unique_videos(limit=request.limit * 3)
        return {
            "syncs": [
                {
                    "video_id": r.video_id,
                    "audio_hash": r.audio_hash,
                    "quality_score": r.quality_score,
                    "language": r.language,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "lyrics_preview": _get_lyrics_preview(r.timestamps),
                }
                for r in results
            ]
        }


def _get_lyrics_preview(timestamps: dict) -> str:
    segments = timestamps.get("segments", [])
    if not segments:
        return ""
    texts = [s.get("text", "") for s in segments[:3]]
    return " / ".join(texts)[:100]


@router.post("/regenerate", response_model=GenerateResponse)
async def regenerate_sync(request: RegenerateRequest, background_tasks: BackgroundTasks):
    lyrics_hash_value = hash_lyrics(request.lyrics)

    async with get_session() as session:
        if not request.force:
            sync_repo = SyncRepository(session)
            existing = await sync_repo.get_by_video_and_hash(request.video_id, lyrics_hash_value)
            if existing:
                return GenerateResponse(
                    job_id=existing.id,
                    status="completed",
                    estimated_time=0,
                )

        job_repo = JobRepository(session)
        job = await job_repo.create(
            video_id=request.video_id,
            lyrics=request.lyrics,
            language=request.language,
        )
        job_id = job.id

    from everyric2.server.worker import process_job

    background_tasks.add_task(process_job, job_id)

    return GenerateResponse(
        job_id=job_id,
        status="processing",
        estimated_time=15,
    )
