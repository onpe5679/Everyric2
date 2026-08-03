"""공지 시스템(notices) 테스트 — GET /api/notices + 어드민 생성/비활성화.

기존 서버 테스트 규약(test_sync_versions.py와 동일): 격리된 in-memory SQLite로
connection.async_session을 몽키패치하고 라우트 코루틴을 직접 await한다(httpx 불사용).

여기서 못박는 계약:
  ① 목록은 active=True + (ends_at이 없거나 미래)인 공지만, 최신순.
  ② 비활성화(deactivate)나 만료(ends_at 과거)된 공지는 목록에서 빠진다.
  ③ 생성/비활성화는 admin_api_key 미설정이면 503, 키 불일치면 403.
  ④ 다국어화(2026-08-04, additive) — translations를 실어 생성하면 그대로 왕복하고,
     안 실으면 None(구버전 확장·마이그레이션 전 행과 같은 모양) — title/body(한국어
     기본)는 이 필드의 유무와 무관하게 항상 채워진다(폴백 언어 보장).
"""

import asyncio
import contextlib
from datetime import datetime, timedelta, timezone

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from everyric2.config.settings import get_settings
from everyric2.server.api.notices import (
    CreateNoticeRequest,
    NoticeTranslation,
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


def test_notice_without_translations_round_trips_as_none():
    """기존(구버전 확장·마이그레이션 전) 행과 같은 모양 — translations 생략 시 None,
    title/body는 그대로 나간다(폴백 언어 보장)."""
    async def body():
        async with _env(admin_key="secret"):
            await create_notice(
                CreateNoticeRequest(title="점검 안내", body="오늘 자정 점검", level="info"),
                x_api_key="secret",
            )
            item = (await list_notices()).notices[0]
            assert item.translations is None
            assert item.title == "점검 안내"
            assert item.body == "오늘 자정 점검"

    asyncio.run(body())


def test_notice_with_translations_round_trips_per_language():
    """언어별 title/body가 생성 그대로 조회에 실려 나온다 — 한국어 기본 필드도 그대로 유지."""
    async def body():
        async with _env(admin_key="secret"):
            await create_notice(
                CreateNoticeRequest(
                    title="점검 안내",
                    body="오늘 자정 점검",
                    level="info",
                    translations={
                        "en": NoticeTranslation(title="Maintenance notice", body="Maintenance tonight at midnight"),
                        "ja": NoticeTranslation(title="メンテナンスのお知らせ", body="今夜0時にメンテナンス"),
                    },
                ),
                x_api_key="secret",
            )
            item = (await list_notices()).notices[0]
            # 한국어 기본 필드는 다국어화 이전과 완전히 동일한 뜻으로 유지된다
            assert item.title == "점검 안내"
            assert item.body == "오늘 자정 점검"
            assert item.translations is not None
            assert item.translations["en"].title == "Maintenance notice"
            assert item.translations["en"].body == "Maintenance tonight at midnight"
            assert item.translations["ja"].title == "メンテナンスのお知らせ"
            assert item.translations["ja"].body == "今夜0時にメンテナンス"
            # ko는 translations 안에 있을 필요가 없다(위 기본 필드가 이미 한국어다) —
            # 넣지 않았다는 사실 자체도 확인해 둔다(부분 언어만 있어도 되는 계약)
            assert "ko" not in item.translations

    asyncio.run(body())


def test_notice_translations_persist_as_plain_dict_in_db():
    """저장 경로에서 NoticeTranslation(pydantic) 인스턴스가 아니라 순수 dict로 JSON 컬럼에
    들어가는지 — DB에서 다시 읽어도(세션 재조회) 깨지지 않아야 한다(직렬화 계약)."""
    async def body():
        async with _env(admin_key="secret"):
            from everyric2.server.db.repository import NoticeRepository

            created = await create_notice(
                CreateNoticeRequest(
                    title="t", body="b", level="info",
                    translations={"en": NoticeTranslation(title="et", body="eb")},
                ),
                x_api_key="secret",
            )
            async with db_conn.async_session() as s:
                notice = await NoticeRepository(s).get_by_id(created.id)
                assert notice is not None
                assert notice.translations == {"en": {"title": "et", "body": "eb"}}
                assert isinstance(notice.translations["en"], dict)  # NoticeTranslation 인스턴스가 아니다

    asyncio.run(body())
