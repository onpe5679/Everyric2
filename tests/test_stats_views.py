"""조회수 카운터(SyncView) 테스트 — GET /api/sync/{video_id} 성공 시 증가 + POST
/api/stats/views 배치 조회.

기존 서버 테스트 규약(test_sync_versions.py와 동일): 격리된 in-memory SQLite로
connection.async_session을 몽키패치하고 라우트 코루틴을 직접 await한다(httpx 불사용).

여기서 못박는 계약:
  ① 자기 싱크 조회(lyrics_hash 유무 무관)가 성공하면 조회수가 1 늘어난다.
  ② 링크로 빌려온 싱크 조회는 **보는 영상**(video_id)의 조회수를 늘린다(원곡이 아니다).
  ③ 싱크가 없는 조회(found=false)는 조회수를 건드리지 않는다.
  ④ POST /api/stats/views는 요청한 video_id 전부를 돌려주고, 기록이 없는 id는 0.
  ⑤ 조회수 증가 실패가 조회 자체(get_sync의 응답)를 막지 않는다.
"""

import asyncio
import contextlib

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from everyric2.server.api.stats import ViewsRequest, get_view_counts
from everyric2.server.api.sync import get_sync
from everyric2.server.db import connection as db_conn
from everyric2.server.db.models import Base
from everyric2.server.db.repository import SyncLinkRepository, SyncRepository, SyncViewRepository

VIDEO = "VIEWVIDEO01"
COVER = "COVERVIDEO1"
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


async def _views(sm, video_id: str) -> int:
    async with sm() as s:
        found = await SyncViewRepository(s).get_many([video_id])
        return found.get(video_id, 0)


def test_found_sync_lookup_increments_views():
    async def body():
        async with _env() as sm:
            async with sm() as s:
                await SyncRepository(s).create(
                    video_id=VIDEO, lyrics_hash=LYRICS_HASH, timestamps=SEGMENTS
                )
                await s.commit()

            assert await _views(sm, VIDEO) == 0
            resp = await get_sync(VIDEO)
            assert resp.found is True
            assert await _views(sm, VIDEO) == 1
            await get_sync(VIDEO)
            assert await _views(sm, VIDEO) == 2

    asyncio.run(body())


def test_not_found_lookup_does_not_increment():
    async def body():
        async with _env() as sm:
            resp = await get_sync("NOSYNCVID01")
            assert resp.found is False
            assert await _views(sm, "NOSYNCVID01") == 0

    asyncio.run(body())


def test_linked_lookup_increments_the_viewed_video_not_the_source():
    async def body():
        async with _env() as sm:
            async with sm() as s:
                await SyncRepository(s).create(
                    video_id=VIDEO, lyrics_hash=LYRICS_HASH, timestamps=SEGMENTS
                )
                await SyncLinkRepository(s).upsert(COVER, VIDEO, offset_sec=1.5)
                await s.commit()

            resp = await get_sync(COVER)
            assert resp.found is True
            assert resp.linked is not None
            assert await _views(sm, COVER) == 1
            assert await _views(sm, VIDEO) == 0  # 원곡은 그대로

    asyncio.run(body())


def test_batch_views_endpoint_fills_missing_ids_with_zero():
    async def body():
        async with _env() as sm:
            async with sm() as s:
                await SyncViewRepository(s).increment(VIDEO)
                await SyncViewRepository(s).increment(VIDEO)
                await SyncViewRepository(s).increment(VIDEO)
                await s.commit()

            resp = await get_view_counts(
                ViewsRequest(video_ids=[VIDEO, "NEVERSEENV1"])
            )
            assert resp.views == {VIDEO: 3, "NEVERSEENV1": 0}

    asyncio.run(body())


def test_batch_views_rejects_malformed_video_id():
    async def body():
        async with _env():
            try:
                await get_view_counts(ViewsRequest(video_ids=["not-valid"]))
                assert False, "should have raised 422"
            except Exception as e:
                assert getattr(e, "status_code", None) == 422

    asyncio.run(body())


def test_batch_views_empty_list_returns_empty_dict():
    async def body():
        async with _env():
            resp = await get_view_counts(ViewsRequest(video_ids=[]))
            assert resp.views == {}

    asyncio.run(body())
