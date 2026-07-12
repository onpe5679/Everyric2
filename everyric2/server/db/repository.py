import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from everyric2.server.db.models import ActionLog, Job, SyncLink, SyncResult, VideoOffset


def hash_lyrics(lyrics: str) -> str:
    return hashlib.sha256(lyrics.strip().encode()).hexdigest()[:16]


class SyncRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_video_and_hash(self, video_id: str, lyrics_hash: str) -> SyncResult | None:
        # force 재생성은 같은 (video_id, lyrics_hash) 행을 여러 개 만들 수 있다 — 최신 우선
        result = await self.session.execute(
            select(SyncResult)
            .where(
                SyncResult.video_id == video_id,
                SyncResult.lyrics_hash == lyrics_hash,
            )
            .order_by(SyncResult.created_at.desc())
        )
        return result.scalars().first()

    async def get_by_audio_hash(self, audio_hash: str) -> SyncResult | None:
        result = await self.session.execute(
            select(SyncResult)
            .where(SyncResult.audio_hash == audio_hash)
            .order_by(SyncResult.created_at.desc())
        )
        return result.scalar_one_or_none()

    async def get_by_audio_and_lyrics_hash(
        self, audio_hash: str, lyrics_hash: str
    ) -> SyncResult | None:
        # force 재생성으로 동일 해시 행이 복수 존재할 수 있다 — 최신 우선
        result = await self.session.execute(
            select(SyncResult)
            .where(
                SyncResult.audio_hash == audio_hash,
                SyncResult.lyrics_hash == lyrics_hash,
            )
            .order_by(SyncResult.created_at.desc())
        )
        return result.scalars().first()

    async def get_by_video(self, video_id: str) -> list[SyncResult]:
        result = await self.session.execute(
            select(SyncResult)
            .where(SyncResult.video_id == video_id)
            .order_by(SyncResult.created_at.desc())
        )
        return list(result.scalars().all())

    async def get_all_unique_videos(self, limit: int = 50) -> list[SyncResult]:
        """Get one sync result per unique video_id, ordered by most recent."""
        from sqlalchemy import func

        # Subquery to get max created_at for each video_id
        subquery = (
            select(SyncResult.video_id, func.max(SyncResult.created_at).label("max_created"))
            .group_by(SyncResult.video_id)
            .subquery()
        )

        # Join to get full SyncResult rows
        result = await self.session.execute(
            select(SyncResult)
            .join(
                subquery,
                (SyncResult.video_id == subquery.c.video_id)
                & (SyncResult.created_at == subquery.c.max_created),
            )
            .order_by(SyncResult.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def delete_by_video(self, video_id: str) -> int:
        """이 영상의 모든 싱크 삭제(초기화) — 잘못 붙여넣은 가사 등에서 완전히 새로 시작.
        삭제된 행 수를 반환."""
        result = await self.session.execute(
            delete(SyncResult).where(SyncResult.video_id == video_id)
        )
        return result.rowcount or 0

    async def create(
        self,
        video_id: str,
        lyrics_hash: str,
        timestamps: list[dict[str, Any]],
        language: str | None = None,
        engine: str = "ctc",
        quality_score: float | None = None,
        audio_hash: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> SyncResult:
        sync_result = SyncResult(
            video_id=video_id,
            lyrics_hash=lyrics_hash,
            audio_hash=audio_hash,
            # extra: segments 밖의 곡 단위 부가정보 (예: {"debug": {...}})
            timestamps={"segments": timestamps, **(extra or {})},
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

    async def get_active_by_video(self, video_id: str, lyrics_hash: str) -> Job | None:
        """같은 영상·같은 가사로 이미 진행 중(pending/processing)인 잡 — 중복 생성 차단용.

        같은 잡이 2개 돌면 같은 임시 오디오 파일을 두 프로세스가 잡아 Windows에서
        WinError 32(파일 잠금)로 다운로드가 깨진다 — 생성 요청은 진행 중 잡에 합류시킨다.
        """
        result = await self.session.execute(
            select(Job)
            .where(
                Job.video_id == video_id,
                Job.lyrics_hash == lyrics_hash,
                Job.status.in_(["pending", "processing"]),
            )
            .order_by(Job.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

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
        stage: str | None = None,
    ) -> None:
        values: dict[str, Any] = {"status": status}
        if progress is not None:
            values["progress"] = progress
        if result_id is not None:
            values["result_id"] = result_id
        if error is not None:
            values["error"] = error
        if stage is not None:
            values["stage"] = stage

        await self.session.execute(update(Job).where(Job.id == job_id).values(**values))


class VideoOffsetRepository:
    """영상별 사용자 싱크 오프셋 upsert/조회."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get(self, video_id: str) -> float | None:
        result = await self.session.execute(
            select(VideoOffset).where(VideoOffset.video_id == video_id)
        )
        row = result.scalar_one_or_none()
        return row.offset_sec if row else None

    async def upsert(self, video_id: str, offset_sec: float) -> None:
        result = await self.session.execute(
            select(VideoOffset).where(VideoOffset.video_id == video_id)
        )
        row = result.scalar_one_or_none()
        if row:
            row.offset_sec = offset_sec
        else:
            self.session.add(VideoOffset(video_id=video_id, offset_sec=offset_sec))
        await self.session.flush()


class ActionLogRepository:
    """파괴적 행위 로그 — 영상·행위별 최근 24시간 횟수로 일일 한도를 검사한다."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def log(self, action: str, video_id: str) -> None:
        self.session.add(ActionLog(action=action, video_id=video_id))
        await self.session.flush()

    async def count_recent(self, action: str, video_id: str, hours: int = 24) -> int:
        since = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=hours)
        result = await self.session.execute(
            select(func.count())
            .select_from(ActionLog)
            .where(
                ActionLog.action == action,
                ActionLog.video_id == video_id,
                ActionLog.created_at >= since,
            )
        )
        return int(result.scalar_one())


class SyncLinkRepository:
    """싱크 링크 CRUD (video_id 고유 → PK 기반 upsert)."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get(self, video_id: str) -> SyncLink | None:
        result = await self.session.execute(
            select(SyncLink).where(SyncLink.video_id == video_id)
        )
        return result.scalar_one_or_none()

    async def delete_involving(self, video_id: str) -> int:
        """이 영상이 소유자이거나 소스인 링크 전부 삭제 — 싱크 초기화 시 정합성 유지
        (소스 싱크가 사라진 링크를 남겨두면 빌려 쓰던 영상의 조회가 깨진다)."""
        result = await self.session.execute(
            delete(SyncLink).where(
                or_(SyncLink.video_id == video_id, SyncLink.source_video_id == video_id)
            )
        )
        return result.rowcount or 0

    async def upsert(self, video_id: str, source_video_id: str, offset_sec: float) -> SyncLink:
        existing = await self.get(video_id)
        if existing:
            existing.source_video_id = source_video_id
            existing.offset_sec = offset_sec
            await self.session.flush()
            return existing
        link = SyncLink(
            video_id=video_id, source_video_id=source_video_id, offset_sec=offset_sec
        )
        self.session.add(link)
        await self.session.flush()
        return link

    async def delete(self, video_id: str) -> bool:
        existing = await self.get(video_id)
        if not existing:
            return False
        await self.session.delete(existing)
        await self.session.flush()
        return True
