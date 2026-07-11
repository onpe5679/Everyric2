from datetime import datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import JSON, DateTime, Float, String, Text, func
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(AsyncAttrs, DeclarativeBase):
    pass


class SyncResult(Base):
    __tablename__ = "sync_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    video_id: Mapped[str] = mapped_column(String(32), index=True)
    lyrics_hash: Mapped[str] = mapped_column(String(64), index=True)
    audio_hash: Mapped[str | None] = mapped_column(String(32), index=True)
    timestamps: Mapped[dict[str, Any]] = mapped_column(JSON)
    language: Mapped[str | None] = mapped_column(String(8))
    engine: Mapped[str] = mapped_column(String(16), default="ctc")
    quality_score: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    video_id: Mapped[str] = mapped_column(String(32), index=True)
    lyrics: Mapped[str] = mapped_column(Text)
    lyrics_hash: Mapped[str] = mapped_column(String(64))
    language: Mapped[str | None] = mapped_column(String(8))
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    result_id: Mapped[str | None] = mapped_column(String(36))
    error: Mapped[str | None] = mapped_column(Text)
    progress: Mapped[int] = mapped_column(default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class SyncLink(Base):
    """다른 영상의 싱크를 오프셋과 함께 재사용하는 링크 (inst/커버 영상용).

    video_id는 링크의 소유자(PK=고유). source_video_id의 실제 싱크를 offset_sec만큼
    시프트해 조회 시 대신 내려준다. 자기 싱크가 있으면 링크보다 우선한다.
    """

    __tablename__ = "sync_links"

    video_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    source_video_id: Mapped[str] = mapped_column(String(32), index=True)
    offset_sec: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
