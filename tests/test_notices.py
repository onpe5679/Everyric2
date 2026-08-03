"""공지 시스템(notices) 테스트 — GET /api/notices + 어드민 생성/비활성화.

기존 서버 테스트 규약(test_sync_versions.py와 동일): 격리된 in-memory SQLite로
connection.async_session을 몽키패치하고 라우트 코루틴을 직접 await한다(httpx 불사용).

여기서 못박는 계약:
  ① 목록은 active=True + (ends_at이 없거나 미래)인 공지만, 최신순.
  ② 비활성화(deactivate)나 만료(ends_at 과거)된 공지는 목록에서 빠진다.
  ③ 생성/비활성화는 admin_api_key 미설정이면 503, 키 불일치면 403.
"""

import asyncio
import contextlib
from datetime import datetime, timedelta, timezone

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from everyric2.config.settings import get_settings
from everyric2.server.api.notices import (
    CreateNoticeRequest,
    create_notice,
    deactivate_notice,
    list_notices,
)
from everyric2.server.db import connection as db_conn
from everyric2.server.db.models import Base

def _now() -> datetime:
    """NoticeRepository.list_active가 대조하는 것과 같은 기준 시각(naive UTC) — 테스트가
    미래에 실행돼도 만료 판정이 실제 벽시계와 어긋나지 않게 고정 상수 대신 매번 구한다."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


@contextlib.asynccontextmanager
async def _env(admin_key: str = ""):
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
    # test_cookies_admin.py와 같은 관례 — pydantic 설정 인스턴스를 직접 되돌린다(환경변수
    # 경유는 프로세스 전역이라 병렬 테스트와 간섭할 수 있다).
    server = get_settings().server
    orig_key = server.admin_api_key
    object.__setattr__(server, "admin_api_key", admin_key)
    try:
        yield sm
    finally:
        db_conn.async_session = orig
        object.__setattr__(server, "admin_api_key", orig_key)
        await engine.dispose()


def test_list_is_empty_with_no_notices():
    async def body():
        async with _env():
            resp = await list_notices()
            assert resp.notices == []

    asyncio.run(body())


def test_create_requires_admin_key_when_configured():
    async def body():
        async with _env(admin_key="secret"):
            try:
                await create_notice(
                    CreateNoticeRequest(title="t", body="b", level="info"), x_api_key=None
                )
                assert False, "should have raised 403"
            except Exception as e:
                assert getattr(e, "status_code", None) == 403

    asyncio.run(body())


def test_create_is_503_when_admin_key_unset():
    async def body():
        async with _env(admin_key=""):
            try:
                await create_notice(
                    CreateNoticeRequest(title="t", body="b", level="info"), x_api_key="anything"
                )
                assert False, "should have raised 503"
            except Exception as e:
                assert getattr(e, "status_code", None) == 503

    asyncio.run(body())


def test_create_then_list_round_trip_with_correct_key():
    async def body():
        async with _env(admin_key="secret"):
            created = await create_notice(
                CreateNoticeRequest(title="점검 안내", body="오늘 자정 점검", level="warning"),
                x_api_key="secret",
            )
            assert created.ok is True

            resp = await list_notices()
            assert len(resp.notices) == 1
            item = resp.notices[0]
            assert item.title == "점검 안내"
            assert item.level == "warning"
            assert item.id == created.id

    asyncio.run(body())


def test_deactivated_notice_disappears_from_the_list():
    async def body():
        async with _env(admin_key="secret"):
            created = await create_notice(
                CreateNoticeRequest(title="t", body="b", level="info"), x_api_key="secret"
            )
            assert (await list_notices()).notices

            deactivated = await deactivate_notice(created.id, x_api_key="secret")
            assert deactivated.id == created.id

            assert (await list_notices()).notices == []

    asyncio.run(body())


def test_deactivate_unknown_id_is_404():
    async def body():
        async with _env(admin_key="secret"):
            try:
                await deactivate_notice(999999, x_api_key="secret")
                assert False, "should have raised 404"
            except Exception as e:
                assert getattr(e, "status_code", None) == 404

    asyncio.run(body())


def test_expired_notice_is_excluded_but_future_ends_at_is_kept():
    async def body():
        async with _env(admin_key="secret"):
            now = _now()
            await create_notice(
                CreateNoticeRequest(
                    title="expired", body="b", level="info", ends_at=now - timedelta(days=1)
                ),
                x_api_key="secret",
            )
            await create_notice(
                CreateNoticeRequest(
                    title="still valid", body="b", level="info", ends_at=now + timedelta(days=1)
                ),
                x_api_key="secret",
            )
            await create_notice(
                CreateNoticeRequest(title="no expiry", body="b", level="info"),
                x_api_key="secret",
            )

            titles = {n.title for n in (await list_notices()).notices}
            assert titles == {"still valid", "no expiry"}

    asyncio.run(body())
