"""쿼터 조회(GET /api/limits/{video_id}) 테스트.

기존 서버 테스트 규약(test_sync_versions.py와 동일): 격리된 in-memory SQLite로
connection.async_session을 몽키패치하고 라우트 코루틴을 직접 await한다(httpx 불사용).

여기서 못박는 계약:
  ① admin_api_key 미설정이면 enforced=False, used=0/remaining=limit(무제한을 값으로 표현).
  ② 설정돼 있으면 실제 소비 지점(ActionLogRepository.count_recent)과 같은 집계를 낸다.
  ③ 조회 자체는 action_logs에 아무것도 남기지 않는다(조회가 한도를 깎지 않는다).
  ④ destructive는 reset/regenerate 중 더 많이 쓴 쪽(worst case)을 보여준다.
"""

import asyncio
import contextlib

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from everyric2.config.settings import get_settings
from everyric2.server.api.limits import get_limits
from everyric2.server.db import connection as db_conn
from everyric2.server.db.models import Base
from everyric2.server.db.repository import ActionLogRepository

VIDEO = "LIMITVIDEO1"


@contextlib.asynccontextmanager
async def _env(admin_api_key: str = "", daily_destructive_limit: int = 2):
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

    server = get_settings().server
    saved = (server.admin_api_key, server.daily_destructive_limit)
    object.__setattr__(server, "admin_api_key", admin_api_key)
    object.__setattr__(server, "daily_destructive_limit", daily_destructive_limit)
    try:
        yield sm
    finally:
        db_conn.async_session = orig
        object.__setattr__(server, "admin_api_key", saved[0])
        object.__setattr__(server, "daily_destructive_limit", saved[1])
        await engine.dispose()


def test_unenforced_when_admin_key_unset():
    async def body():
        async with _env(admin_api_key=""):
            resp = await get_limits(VIDEO)
            assert resp.enforced is False
            assert resp.generate.used == 0
            assert resp.destructive.used == 0
            assert resp.generate.remaining == resp.generate.limit
            assert resp.destructive.remaining == resp.destructive.limit

    asyncio.run(body())


def test_invalid_video_id_is_422():
    async def body():
        async with _env(admin_api_key="secret"):
            try:
                await get_limits("not-a-valid-id")
                assert False, "should have raised 422"
            except Exception as e:
                assert getattr(e, "status_code", None) == 422

    asyncio.run(body())


def test_reflects_actual_action_log_usage():
    async def body():
        async with _env(admin_api_key="secret", daily_destructive_limit=2):
            async with db_conn.async_session() as s:
                repo = ActionLogRepository(s)
                await repo.log("generate", VIDEO)
                await repo.log("generate", VIDEO)
                await repo.log("generate", VIDEO)
                await repo.log("reset", VIDEO)
                await s.commit()

            resp = await get_limits(VIDEO)
            assert resp.enforced is True
            assert resp.generate.used == 3
            assert resp.generate.remaining == resp.generate.limit - 3
            assert resp.destructive.used == 1  # reset 1건
            assert resp.destructive.remaining == 1  # limit(2) - 1

    asyncio.run(body())


def test_destructive_shows_the_worse_of_reset_and_regenerate():
    async def body():
        async with _env(admin_api_key="secret", daily_destructive_limit=2):
            async with db_conn.async_session() as s:
                repo = ActionLogRepository(s)
                await repo.log("reset", VIDEO)
                await repo.log("regenerate", VIDEO)
                await repo.log("regenerate", VIDEO)
                await s.commit()

            resp = await get_limits(VIDEO)
            # reset=1, regenerate=2 -> worst case인 2를 보여준다(각자 독립 집계라 실제로는
            # 둘 다 각자의 한도를 따로 검사하지만, 배지 하나로는 보수적인 값을 낸다)
            assert resp.destructive.used == 2
            assert resp.destructive.remaining == 0

    asyncio.run(body())


def test_lookup_itself_does_not_consume_the_quota():
    """조회를 여러 번 반복해도 action_logs는 늘지 않는다 — 조회가 한도를 깎아먹지 않는다."""

    async def body():
        async with _env(admin_api_key="secret"):
            await get_limits(VIDEO)
            await get_limits(VIDEO)
            await get_limits(VIDEO)

            resp = await get_limits(VIDEO)
            assert resp.generate.used == 0
            assert resp.destructive.used == 0

    asyncio.run(body())


def test_remaining_floors_at_zero_when_over_limit():
    async def body():
        async with _env(admin_api_key="secret", daily_destructive_limit=1):
            async with db_conn.async_session() as s:
                repo = ActionLogRepository(s)
                await repo.log("reset", VIDEO)
                await repo.log("reset", VIDEO)
                await repo.log("reset", VIDEO)
                await s.commit()

            resp = await get_limits(VIDEO)
            assert resp.destructive.used == 3
            assert resp.destructive.remaining == 0  # 음수가 아니라 0으로 floor

    asyncio.run(body())
