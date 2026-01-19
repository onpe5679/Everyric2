import hashlib
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from everyric2.server.db.models import Job, SyncResult


def hash_lyrics(lyrics: str) -> str:
    return hashlib.sha256(lyrics.strip().encode()).hexdigest()[:16]


class SyncRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_video_and_hash(self, video_id: str, lyrics_hash: str) -> SyncResult | None:
        result = await self.session.execute(
            select(SyncResult).where(
                SyncResult.video_id == video_id,
                SyncResult.lyrics_hash == lyrics_hash,
            )
        )
        return result.scalar_one_or_none()

    async def get_by_video(self, video_id: str) -> list[SyncResult]:
        result = await self.session.execute(
            select(SyncResult)
            .where(SyncResult.video_id == video_id)
            .order_by(SyncResult.created_at.desc())
        )
        return list(result.scalars().all())

    async def create(
        self,
        video_id: str,
        lyrics_hash: str,
        timestamps: list[dict[str, Any]],
        language: str | None = None,
        engine: str = "ctc",
        quality_score: float | None = None,
    ) -> SyncResult:
        sync_result = SyncResult(
            video_id=video_id,
            lyrics_hash=lyrics_hash,
            timestamps={"segments": timestamps},
            language=language,
            engine=engine,
            quality_score=quality_score,
        )
        self.session.add(sync_result)
        await self.session.flush()
        return sync_result


class JobRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_id(self, job_id: str) -> Job | None:
        result = await self.session.execute(select(Job).where(Job.id == job_id))
        return result.scalar_one_or_none()

    async def get_pending(self, limit: int = 10) -> list[Job]:
        result = await self.session.execute(
            select(Job).where(Job.status == "pending").order_by(Job.created_at).limit(limit)
        )
        return list(result.scalars().all())

    async def create(
        self,
        video_id: str,
        lyrics: str,
        language: str | None = None,
    ) -> Job:
        lyrics_hash = hash_lyrics(lyrics)
        job = Job(
            video_id=video_id,
            lyrics=lyrics,
            lyrics_hash=lyrics_hash,
            language=language,
        )
        self.session.add(job)
        await self.session.flush()
        return job

    async def update_status(
        self,
        job_id: str,
        status: str,
        progress: int | None = None,
        result_id: str | None = None,
        error: str | None = None,
    ) -> None:
        values: dict[str, Any] = {"status": status}
        if progress is not None:
            values["progress"] = progress
        if result_id is not None:
            values["result_id"] = result_id
        if error is not None:
            values["error"] = error

        await self.session.execute(update(Job).where(Job.id == job_id).values(**values))
