"""정렬 품질 피드백(sync_feedback) 수집 테스트 — 확장 별점·오류 제보 UI의 서버 계약.

기존 서버 테스트 규약(test_sync_versions.py와 동일): 격리된 in-memory SQLite로
connection.async_session을 몽키패치하고 라우트 코루틴을 직접 await한다(httpx 불사용).

여기서 못박는 계약:
  ① 싱크가 없어도 받는다(sync_id=None) — "싱크가 안 만들어져요" 류 제보도 유효하다.
  ② 싱크가 있으면 제출 시점의 최신 싱크 sync_id·engine_version을 함께 새긴다 —
     세대별 품질 집계의 재료(재생성되면 같은 영상도 다른 세대).
  ③ rating은 1~5 밖이면 요청 스키마에서 거부된다. category도 정해진 어휘만.
"""

import asyncio
import contextlib

import pytest
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from everyric2.server.api.sync import FeedbackRequest, submit_feedback
from everyric2.server.db import connection as db_conn
from everyric2.server.db.models import Base, SyncFeedback
from everyric2.server.db.repository import SyncRepository

VIDEO = "FEEDFEEDFB1"


@contextlib.asynccontextmanager
async def _env():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    orig = db_conn.async_session
    db_conn.async_session = sm
    try:
        yield sm
    finally:
        db_conn.async_session = orig
        await engine.dispose()


def test_feedback_without_sync_is_accepted_with_null_sync_id():
    async def body():
        async with _env() as sm:
            out = await submit_feedback(FeedbackRequest(video_id=VIDEO, rating=4))
            assert out == {"ok": True}
            async with sm() as s:
                rows = (await s.execute(select(SyncFeedback))).scalars().all()
                assert len(rows) == 1
                assert rows[0].video_id == VIDEO
                assert rows[0].rating == 4
                assert rows[0].sync_id is None
                assert rows[0].engine_version is None

    asyncio.run(body())


def test_feedback_stamps_latest_sync_generation():
    async def body():
        async with _env() as sm:
            async with sm() as s:
                sync = await SyncRepository(s).create(
                    video_id=VIDEO,
                    lyrics_hash="h1",
                    timestamps=[{"text": "가", "start": 0.0, "end": 1.0}],
                    language="ja",
                )
                await s.commit()
            await submit_feedback(
                FeedbackRequest(
                    video_id=VIDEO, rating=2, category="timing", comment="후렴이 밀려요"
                )
            )
            async with sm() as s:
                row = (await s.execute(select(SyncFeedback))).scalars().one()
                assert row.sync_id == sync.id
                assert row.engine_version == sync.engine_version  # 세대 스탬프가 같이 남는다
                assert row.category == "timing"
                assert row.comment == "후렴이 밀려요"

    asyncio.run(body())


def test_rating_and_category_are_schema_validated():
    with pytest.raises(ValidationError):
        FeedbackRequest(video_id=VIDEO, rating=0)
    with pytest.raises(ValidationError):
        FeedbackRequest(video_id=VIDEO, rating=6)
    with pytest.raises(ValidationError):
        FeedbackRequest(video_id=VIDEO, rating=3, category="nonsense")


# ── depth 스탬프 (2026-08-04, 디버그 패널 깊이별 비교 재료) ─────────────


def test_depth_is_optional_and_defaults_to_none():
    async def body():
        async with _env():
            out = await submit_feedback(FeedbackRequest(video_id=VIDEO, rating=5))
            assert out == {"ok": True}
            async with db_conn.async_session() as s:
                row = (await s.execute(select(SyncFeedback))).scalars().one()
                assert row.depth is None

    asyncio.run(body())


def test_depth_is_stored_when_the_client_sends_it():
    async def body():
        async with _env() as sm:
            await submit_feedback(
                FeedbackRequest(video_id=VIDEO, rating=1, depth="heavy")
            )
            async with sm() as s:
                row = (await s.execute(select(SyncFeedback))).scalars().one()
                assert row.depth == "heavy"

    asyncio.run(body())


def test_depth_is_schema_validated():
    with pytest.raises(ValidationError):
        FeedbackRequest(video_id=VIDEO, rating=3, depth="turbo")
