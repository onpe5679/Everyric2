"""POST /api/sync/exists — 배치 싱크 존재 조회 테스트 (결함 #5, additive).

기존 서버 테스트 규약(test_stats_views.py와 동일): 격리된 in-memory SQLite로
connection.async_session을 몽키패치하고 라우트 코루틴을 직접 await한다(httpx 불사용).

여기서 못박는 계약:
  ① 자기 싱크(sync_results)가 있는 영상은 True.
  ② 자기 싱크는 없지만 링크(sync_links, 빌려 온 싱크)만 있는 영상도 True — GET
     /api/sync/{video_id}가 링크 폴백을 내주므로 확장 배지도 존재로 봐야 한다.
  ③ 둘 다 없는 영상은 False.
  ④ 응답은 요청 video_id 전체를 덮는 dict다.
  ⑤ video_id 형식이 잘못되면 422.
  ⑥ GET이 아니라 POST로 등록된다 — {video_id} 캐치올에 먹히지 않는다.
"""

import asyncio
import contextlib

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from everyric2.server.api.sync import SyncExistsRequest, router, sync_exists
from everyric2.server.db import connection as db_conn
from everyric2.server.db.models import Base
from everyric2.server.db.repository import SyncLinkRepository, SyncRepository

OWN_VIDEO = "OWNSYNCVID1"
LINKED_VIDEO = "LINKEDVID01"
SOURCE_VIDEO = "SOURCEVID01"
MISSING_VIDEO = "NOSYNCVID01"
LYRICS_HASH = "h1"
SEGMENTS = [{"text": "가사", "start": 0.0, "end": 1.0}]


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


def test_own_sync_and_linked_sync_are_both_true_missing_is_false():
    async def body():
        async with _env() as sm:
            async with sm() as s:
                await SyncRepository(s).create(
                    video_id=OWN_VIDEO, lyrics_hash=LYRICS_HASH, timestamps=SEGMENTS
                )
                await SyncRepository(s).create(
                    video_id=SOURCE_VIDEO, lyrics_hash=LYRICS_HASH, timestamps=SEGMENTS
                )
                await SyncLinkRepository(s).upsert(LINKED_VIDEO, SOURCE_VIDEO, offset_sec=1.5)
                await s.commit()

            resp = await sync_exists(
                SyncExistsRequest(video_ids=[OWN_VIDEO, LINKED_VIDEO, MISSING_VIDEO])
            )
            assert resp.exists == {OWN_VIDEO: True, LINKED_VIDEO: True, MISSING_VIDEO: False}

    asyncio.run(body())


def test_link_without_a_resolvable_source_is_still_true():
    """링크 존재 자체가 True 기준이다 — 소스 싱크의 실존 여부까지 다시 확인하지 않는다
    (get_sync가 자기 싱크 없는 영상에 링크 폴백을 내주는 것과 같은 낙관적 계약)."""

    async def body():
        async with _env() as sm:
            async with sm() as s:
                await SyncLinkRepository(s).upsert(LINKED_VIDEO, "GHOSTSOURCE1", offset_sec=0.0)
                await s.commit()

            resp = await sync_exists(SyncExistsRequest(video_ids=[LINKED_VIDEO]))
            assert resp.exists == {LINKED_VIDEO: True}

    asyncio.run(body())


def test_response_covers_the_entire_request_even_when_all_missing():
    other_missing = "ANOTHERMISS"  # 11자 — 유효한 video_id 형식
    assert len(other_missing) == 11

    async def body():
        async with _env():
            resp = await sync_exists(
                SyncExistsRequest(video_ids=[MISSING_VIDEO, other_missing])
            )
            assert resp.exists == {MISSING_VIDEO: False, other_missing: False}

    asyncio.run(body())


def test_empty_request_returns_empty_dict():
    async def body():
        async with _env():
            resp = await sync_exists(SyncExistsRequest(video_ids=[]))
            assert resp.exists == {}

    asyncio.run(body())


def test_malformed_video_id_is_rejected_with_422():
    async def body():
        async with _env():
            try:
                await sync_exists(SyncExistsRequest(video_ids=["not-a-valid-id!"]))
                assert False, "should have raised 422"
            except HTTPException as e:
                assert e.status_code == 422

    asyncio.run(body())


def test_registered_as_post_not_get():
    """{video_id} 캐치올 GET에 먹히지 않으려면 POST여야 한다 — 라우트 등록을 직접 확인."""
    matches = [r for r in router.routes if getattr(r, "path", None) == "/api/sync/exists"]
    assert matches, "‎/api/sync/exists 라우트가 등록돼 있지 않다"
    assert "POST" in matches[0].methods
    assert "GET" not in matches[0].methods
