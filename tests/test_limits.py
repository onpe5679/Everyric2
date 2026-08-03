"""쿼터 조회(GET /api/limits/{video_id}) 테스트.

기존 서버 테스트 규약(test_sync_versions.py와 동일): 격리된 in-memory SQLite로
connection.async_session을 몽키패치하고 라우트 코루틴을 직접 await한다(httpx 불사용).

여기서 못박는 계약:
  ① admin_api_key 미설정이면 enforced=False, used=0/remaining=limit(무제한을 값으로 표현).
  ② 설정돼 있으면 실제 소비 지점(ActionLogRepository.count_recent)과 같은 집계를 낸다.
  ③ 조회 자체는 action_logs에 아무것도 남기지 않는다(조회가 한도를 깎지 않는다).
  ④ destructive는 reset/regenerate 중 더 많이 쓴 쪽(worst case)을 보여준다.
  ⑤ next_reset_at(2026-08-04 보강) — 실제 기전은 자정 리셋이 아니라 행위별 롤링 24시간
     창이라, "가장 오래된 기록이 창을 벗어나는 시각"이 유일하게 사실과 일치하는 다음
     리셋 시각이다. destructive는 동률(reset count == regenerate count)이면 **둘 다**
     창을 벗어나야 worst-case used가 실제로 줄어들므로 더 늦은 쪽을 낸다 — 먼저 벗어나는
     쪽(MIN)을 냈다면 클라이언트가 "이 시각에 리셋"이라고 안내해도 실제로는 안 줄어드는
     거짓 안내가 된다.
"""

import asyncio
import contextlib
from datetime import datetime, timedelta, timezone

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from everyric2.config.settings import get_settings
from everyric2.server.api.limits import get_limits
from everyric2.server.db import connection as db_conn
from everyric2.server.db.models import ActionLog, Base
from everyric2.server.db.repository import ActionLogRepository

VIDEO = "LIMITVIDEO1"


def _utc_naive(**delta_kwargs) -> datetime:
    """count_recent/oldest_recent와 같은 시계(naive UTC)로 오프셋 시각을 만든다."""
    return datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(**delta_kwargs)


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


# ── next_reset_at (2026-08-04 보강) ──────────────────────────────────────


def test_next_reset_at_is_none_when_unused():
    """쓴 게 없으면 줄어들 것도 없다 — None(빈 UI 상태, "리셋 시각 없음")."""

    async def body():
        async with _env(admin_api_key="secret"):
            resp = await get_limits(VIDEO)
            assert resp.generate.next_reset_at is None
            assert resp.destructive.next_reset_at is None

    asyncio.run(body())


def test_next_reset_at_is_none_when_unenforced():
    """무제한 배포(enforced=False)도 소비가 없으니 None."""

    async def body():
        async with _env(admin_api_key=""):
            resp = await get_limits(VIDEO)
            assert resp.generate.next_reset_at is None
            assert resp.destructive.next_reset_at is None

    asyncio.run(body())


def test_next_reset_at_matches_the_oldest_log_plus_window():
    """generate처럼 단일 행위면 가장 오래된 기록 + window_hours가 정확한 다음 리셋 시각."""

    async def body():
        async with _env(admin_api_key="secret"):
            oldest = _utc_naive(hours=-2)  # 2시간 전 — 창(24h) 안, 가장 오래됨
            newer = _utc_naive(hours=-1)  # 1시간 전 — 더 최근
            async with db_conn.async_session() as s:
                s.add(ActionLog(action="generate", video_id=VIDEO, created_at=oldest))
                s.add(ActionLog(action="generate", video_id=VIDEO, created_at=newer))
                await s.commit()

            resp = await get_limits(VIDEO)
            assert resp.generate.used == 2
            expected = oldest + timedelta(hours=resp.window_hours)
            actual = datetime.fromisoformat(resp.generate.next_reset_at)
            assert abs((actual - expected).total_seconds()) < 1

    asyncio.run(body())


def test_destructive_next_reset_at_uses_the_later_expiry_on_a_tie():
    """reset·regenerate가 동률(worst-case used를 같이 결정)이면, **둘 다** 창을 벗어나야
    used가 실제로 줄어든다 — 먼저 벗어나는 쪽(더 이른 시각)을 냈다면 그 시각에 아직
    used가 안 줄어드는 거짓 안내가 된다. 더 늦게 벗어나는 쪽을 내야 한다."""

    async def body():
        async with _env(admin_api_key="secret", daily_destructive_limit=5):
            reset_oldest = _utc_naive(hours=-3)  # 3시간 전 -> 만료 now+21h (더 이르다)
            regenerate_oldest = _utc_naive(hours=-1)  # 1시간 전 -> 만료 now+23h (더 늦다)
            async with db_conn.async_session() as s:
                s.add(ActionLog(action="reset", video_id=VIDEO, created_at=reset_oldest))
                s.add(
                    ActionLog(action="regenerate", video_id=VIDEO, created_at=regenerate_oldest)
                )
                await s.commit()

            resp = await get_limits(VIDEO)
            assert resp.destructive.used == 1  # reset=1, regenerate=1 -> 동률
            expected = regenerate_oldest + timedelta(hours=resp.window_hours)  # 더 늦은 쪽
            actual = datetime.fromisoformat(resp.destructive.next_reset_at)
            assert abs((actual - expected).total_seconds()) < 1
            # 더 이른(틀린) 후보였다면 이 값과 같았을 것 — 실수로 MIN을 썼는지 못박는다
            wrong = reset_oldest + timedelta(hours=resp.window_hours)
            assert abs((actual - wrong).total_seconds()) > 3600

    asyncio.run(body())


def test_destructive_next_reset_at_follows_the_winning_action_when_not_tied():
    """동률이 아니면(worst-case를 결정하는 쪽이 하나뿐) 그 행위의 오래된 기록만 본다."""

    async def body():
        async with _env(admin_api_key="secret", daily_destructive_limit=5):
            regenerate_oldest = _utc_naive(hours=-2)
            async with db_conn.async_session() as s:
                s.add(
                    ActionLog(action="regenerate", video_id=VIDEO, created_at=regenerate_oldest)
                )
                s.add(ActionLog(action="regenerate", video_id=VIDEO, created_at=_utc_naive(hours=-1)))
                # reset은 아예 없음 -> regenerate(2)가 단독으로 worst-case를 결정
                await s.commit()

            resp = await get_limits(VIDEO)
            assert resp.destructive.used == 2
            expected = regenerate_oldest + timedelta(hours=resp.window_hours)
            actual = datetime.fromisoformat(resp.destructive.next_reset_at)
            assert abs((actual - expected).total_seconds()) < 1

    asyncio.run(body())
