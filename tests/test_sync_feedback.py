"""정렬 품질 피드백(sync_feedback) 수집 테스트 — 확장 별점·오류 제보 UI의 서버 계약.

기존 서버 테스트 규약(test_sync_versions.py와 동일): 격리된 in-memory SQLite로
connection.async_session을 몽키패치하고 라우트 코루틴을 직접 await한다(httpx 불사용).

여기서 못박는 계약:
  ① 싱크가 없어도 받는다(sync_id=None) — "싱크가 안 만들어져요" 류 제보도 유효하다.
  ② 신클라이언트가 표시 중인 sync_id를 보내면 재생성이 끼어도 그 세대에 귀속한다.
     필드가 없는 구클라이언트만 제출 시점 최신 싱크로 폴백한다.
  ③ rating은 1~5 밖이면 요청 스키마에서 거부된다. category도 정해진 어휘만.
"""

import asyncio
import contextlib
from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from everyric2.server.api.sync import FeedbackRequest, submit_feedback
from everyric2.server.db import connection as db_conn
from everyric2.server.db.models import Base, SyncFeedback
from everyric2.server.db.repository import SyncLinkRepository, SyncRepository

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


def test_feedback_targets_the_displayed_generation_not_the_new_latest():
    async def body():
        async with _env() as sm:
            async with sm() as s:
                shown = await SyncRepository(s).create(
                    video_id=VIDEO,
                    lyrics_hash="shown",
                    timestamps=[{"text": "가", "start": 0.0, "end": 1.0}],
                    engine_version="shown-engine",
                    extra={"debug": {"routing": {"route": "fast"}}},
                )
                shown.created_at = datetime(2026, 8, 29, 12, 0, 0)
                latest = await SyncRepository(s).create(
                    video_id=VIDEO,
                    lyrics_hash="latest",
                    timestamps=[{"text": "나", "start": 0.0, "end": 1.0}],
                    engine_version="latest-engine",
                    extra={"debug": {"routing": {"route": "heavy"}}},
                )
                latest.created_at = shown.created_at + timedelta(seconds=1)
                await s.commit()
            await submit_feedback(
                FeedbackRequest(
                    video_id=VIDEO,
                    sync_id=shown.id,
                    rating=1,
                    depth="heavy",  # 서버가 표시 세대의 실제 fast로 교정해야 한다
                )
            )
            async with sm() as s:
                row = (await s.execute(select(SyncFeedback))).scalars().one()
                assert row.sync_id == shown.id
                assert row.sync_id != latest.id
                assert row.engine_version == "shown-engine"
                assert row.depth == "fast"

    asyncio.run(body())


def test_explicit_null_sync_id_does_not_fall_back_to_latest():
    async def body():
        async with _env() as sm:
            async with sm() as s:
                await SyncRepository(s).create(
                    video_id=VIDEO,
                    lyrics_hash="latest",
                    timestamps=[{"text": "가", "start": 0.0, "end": 1.0}],
                )
                await s.commit()
            await submit_feedback(FeedbackRequest(video_id=VIDEO, sync_id=None, rating=2))
            async with sm() as s:
                row = (await s.execute(select(SyncFeedback))).scalars().one()
                assert row.sync_id is None
                assert row.engine_version is None

    asyncio.run(body())


def test_unknown_sync_id_does_not_fall_back_to_latest():
    async def body():
        async with _env() as sm:
            async with sm() as s:
                await SyncRepository(s).create(
                    video_id=VIDEO,
                    lyrics_hash="latest",
                    timestamps=[{"text": "가", "start": 0.0, "end": 1.0}],
                )
                await s.commit()
            await submit_feedback(
                FeedbackRequest(
                    video_id=VIDEO,
                    sync_id="00000000-0000-0000-0000-000000000000",
                    rating=2,
                )
            )
            async with sm() as s:
                row = (await s.execute(select(SyncFeedback))).scalars().one()
                assert row.sync_id is None
                assert row.engine_version is None

    asyncio.run(body())


def test_sync_id_from_an_unrelated_video_is_not_attributed():
    other_video = "OTHERVIDEO1"

    async def body():
        async with _env() as sm:
            async with sm() as s:
                unrelated = await SyncRepository(s).create(
                    video_id=other_video,
                    lyrics_hash="other",
                    timestamps=[{"text": "가", "start": 0.0, "end": 1.0}],
                )
                await s.commit()
            await submit_feedback(
                FeedbackRequest(video_id=VIDEO, sync_id=unrelated.id, rating=2)
            )
            async with sm() as s:
                row = (await s.execute(select(SyncFeedback))).scalars().one()
                assert row.sync_id is None
                assert row.engine_version is None

    asyncio.run(body())


def test_linked_video_feedback_can_target_the_displayed_source_generation():
    source_video = "SOURCEVID01"

    async def body():
        async with _env() as sm:
            async with sm() as s:
                source = await SyncRepository(s).create(
                    video_id=source_video,
                    lyrics_hash="source",
                    timestamps=[{"text": "가", "start": 0.0, "end": 1.0}],
                    engine_version="source-engine",
                    extra={"debug": {"routing": {"route": "medium"}}},
                )
                await SyncLinkRepository(s).upsert(VIDEO, source_video, offset_sec=0.5)
                await s.commit()
            await submit_feedback(
                FeedbackRequest(video_id=VIDEO, sync_id=source.id, rating=3, depth="fast")
            )
            async with sm() as s:
                row = (await s.execute(select(SyncFeedback))).scalars().one()
                assert row.sync_id == source.id
                assert row.engine_version == "source-engine"
                assert row.depth == "medium"

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


def test_legacy_request_depth_is_not_overwritten_by_the_new_latest_row():
    async def body():
        async with _env() as sm:
            async with sm() as s:
                await SyncRepository(s).create(
                    video_id=VIDEO,
                    lyrics_hash="latest",
                    timestamps=[{"text": "가", "start": 0.0, "end": 1.0}],
                    extra={"debug": {"routing": {"route": "heavy"}}},
                )
                await s.commit()
            # 구형 확장: depth는 알지만 sync_id 필드는 아직 없다.
            await submit_feedback(FeedbackRequest(video_id=VIDEO, rating=1, depth="fast"))
            async with sm() as s:
                row = (await s.execute(select(SyncFeedback))).scalars().one()
                assert row.depth == "fast"

    asyncio.run(body())


def test_depth_is_schema_validated():
    with pytest.raises(ValidationError):
        FeedbackRequest(video_id=VIDEO, rating=3, depth="turbo")
