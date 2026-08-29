"""쿼터 조회(GET /api/limits/{video_id}) 테스트.

기존 서버 테스트 규약(test_sync_versions.py와 동일): 격리된 in-memory SQLite로
connection.async_session을 몽키패치하고 라우트 코루틴을 직접 await한다(httpx 불사용).

여기서 못박는 계약:
  ① admin_api_key 미설정이면 enforced=False, used=0/remaining=limit(무제한을 값으로 표현).
  ② 설정돼 있으면 실제 소비 지점(ActionLogRepository.count_recent)과 같은 집계를 낸다.
  ③ 조회 자체는 action_logs에 아무것도 남기지 않는다(조회가 한도를 깎지 않는다).
  ④ **집계 축은 이용자다**(2026-08-10, docs/user-quota-spec.md §1) — 경로의 video_id는
     형식 검증에만 쓰인다. 같은 영상이라도 이용자가 다르면 사용량이 섞이지 않고, 같은
     이용자면 영상이 달라도 이어 쓴다. 식별자가 없으면 400(fail-closed)이고, actor가
     비어 있는 옛 기록(축 전환 이전)은 어떤 이용자의 사용량으로도 세지 않는다.
  ⑤ **destructive는 초기화+강제 재생성의 합계**다(같은 문서 §2, 2026-08-10). 예전에는 두
     행위가 각자 상한을 소비하고 표시만 max()로 뭉쳐 "한 예산처럼 보이는데 두 예산"이었다.
     회복 시각도 합계 기준 — 어느 한 건만 창을 벗어나도 합계가 줄기 때문에 **가장 이른**
     기록이 답이다(동률이면 더 늦은 쪽이라는 옛 규칙은 합산에서는 성립하지 않는다).
  ⑥ next_reset_at — 실제 기전은 자정 리셋이 아니라 롤링 24시간 창이라, "가장 오래된 기록이
     창을 벗어나는 시각"이 유일하게 사실과 일치하는 다음 리셋 시각이다.
  ⑦ link(action="link" — 사람이 직접 잇는 POST /api/sync/link)·upgrade(action="upgrade")는
     generate/destructive와 겹치지 않는 자기만의 카운터다. link이 세던 것은 2026-08-10까지
     자동 후보 탐색("link_candidates")이었다 — 사용자가 요청한 적도 실패해도 볼 수도 없는
     배경 요청이라 배지 라벨과 어긋나 있었다(같은 문서 §7).
  ⑧ 어드민(면제 이용자)의 응답은 **enforced=False + 실제 used**다(같은 문서 §5). 강제
     형태로 소진값을 주면 확장이 면제 이용자를 차단 상태로 그린다.
  ⑨ remaining은 절대 음수가 되지 않는다(0으로 floor) — 확장이 그 값을 그대로 그린다.
"""

import asyncio
import contextlib
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from everyric2.config.settings import get_settings
from everyric2.server.api.limits import get_limits
from everyric2.server.api.sync import DAILY_LINK_LIMIT
from everyric2.server.db import connection as db_conn
from everyric2.server.db.models import ActionLog, Base
from everyric2.server.db.repository import ActionLogRepository

VIDEO = "LIMITVIDEO1"
OTHER_VIDEO = "LIMITVIDEO2"
# 게이트웨이가 x-lyric-user로 넘기는 값 — 발급 키 이용자는 불변 키 ID, 익명은 솔트 해시.
ACTOR = "user-aaa"
OTHER_ACTOR = "user-bbb"


def _utc_naive(**delta_kwargs) -> datetime:
    """count_recent/oldest_recent와 같은 시계(naive UTC)로 오프셋 시각을 만든다."""
    return datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(**delta_kwargs)


@contextlib.asynccontextmanager
async def _env(
    admin_api_key: str = "", daily_destructive_limit: int = 4, daily_upgrade_limit: int = 10
):
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
    saved = (server.admin_api_key, server.daily_destructive_limit, server.daily_upgrade_limit)
    object.__setattr__(server, "admin_api_key", admin_api_key)
    object.__setattr__(server, "daily_destructive_limit", daily_destructive_limit)
    object.__setattr__(server, "daily_upgrade_limit", daily_upgrade_limit)
    try:
        yield sm
    finally:
        db_conn.async_session = orig
        object.__setattr__(server, "admin_api_key", saved[0])
        object.__setattr__(server, "daily_destructive_limit", saved[1])
        object.__setattr__(server, "daily_upgrade_limit", saved[2])
        await engine.dispose()


def test_unenforced_when_admin_key_unset():
    async def body():
        async with _env(admin_api_key=""):
            # 한도 미강제 배포는 이용자 식별자 없이도 답한다(기존 로컬 동작 유지)
            resp = await get_limits(VIDEO)
            assert resp.enforced is False
            assert resp.generate.used == 0
            assert resp.destructive.used == 0
            assert resp.generate.remaining == resp.generate.limit
            assert resp.destructive.remaining == resp.destructive.limit
            # link·upgrade도 같은 "무제한을 값으로" 계약을 따른다 — 각자 자기 한도값으로.
            assert resp.link.used == 0
            assert resp.link.remaining == resp.link.limit == DAILY_LINK_LIMIT
            assert resp.upgrade.used == 0
            assert resp.upgrade.remaining == resp.upgrade.limit

    asyncio.run(body())


def test_invalid_video_id_is_422():
    async def body():
        async with _env(admin_api_key="secret"):
            try:
                await get_limits("not-a-valid-id", x_lyric_user=ACTOR)
                assert False, "should have raised 422"
            except Exception as e:
                assert getattr(e, "status_code", None) == 422

    asyncio.run(body())


def test_reflects_actual_action_log_usage():
    async def body():
        async with _env(admin_api_key="secret", daily_destructive_limit=4):
            async with db_conn.async_session() as s:
                repo = ActionLogRepository(s)
                await repo.log("generate", VIDEO, ACTOR)
                await repo.log("generate", VIDEO, ACTOR)
                await repo.log("generate", VIDEO, ACTOR)
                await repo.log("reset", VIDEO, ACTOR)
                await repo.log("link", VIDEO, ACTOR)
                await repo.log("upgrade", VIDEO, ACTOR)
                await s.commit()

            resp = await get_limits(VIDEO, x_lyric_user=ACTOR)
            assert resp.enforced is True
            assert resp.generate.used == 3
            assert resp.generate.remaining == resp.generate.limit - 3
            assert resp.destructive.used == 1  # reset 1건
            assert resp.destructive.remaining == 3  # limit(4) - 1
            assert resp.link.used == 1
            assert resp.link.remaining == resp.link.limit - 1
            # upgrade는 자기만의 로그(1건)만 반영한다 — generate(3건)와 섞이지 않는다
            assert resp.upgrade.used == 1
            assert resp.upgrade.remaining == resp.upgrade.limit - 1

    asyncio.run(body())


# ── 이용자 축 (2026-08-10, docs/user-quota-spec.md §1) ─────────────────────


def test_usage_is_per_actor_not_per_video():
    """같은 영상이라도 이용자가 다르면 사용량이 섞이지 않는다 — 예전에는 A가 쓴 만큼
    B가 그 영상을 처음 열자마자 0회로 시작했다(공개 서비스에서 성립하지 않는 구조)."""

    async def body():
        async with _env(admin_api_key="secret"):
            async with db_conn.async_session() as s:
                repo = ActionLogRepository(s)
                for _ in range(3):
                    await repo.log("generate", VIDEO, ACTOR)
                await s.commit()

            mine = await get_limits(VIDEO, x_lyric_user=ACTOR)
            theirs = await get_limits(VIDEO, x_lyric_user=OTHER_ACTOR)
            assert mine.generate.used == 3
            assert theirs.generate.used == 0
            assert theirs.generate.remaining == theirs.generate.limit

    asyncio.run(body())


def test_usage_follows_the_actor_across_videos():
    """A가 다른 영상으로 옮겨도 같은 개인 예산을 이어 쓴다 — 영상만 바꾸면 예산이 새로
    생기던 구멍(영상 축 집계)이 닫혔다."""

    async def body():
        async with _env(admin_api_key="secret"):
            async with db_conn.async_session() as s:
                repo = ActionLogRepository(s)
                await repo.log("generate", VIDEO, ACTOR)
                await repo.log("generate", OTHER_VIDEO, ACTOR)
                await s.commit()

            here = await get_limits(VIDEO, x_lyric_user=ACTOR)
            there = await get_limits(OTHER_VIDEO, x_lyric_user=ACTOR)
            assert here.generate.used == 2
            assert there.generate.used == 2  # 경로의 video_id는 집계에 쓰이지 않는다

    asyncio.run(body())


def test_missing_actor_is_rejected_when_enforced():
    """식별자가 없거나 빈 값이면 400 — 영상 단위/공용 이용자로 폴백하면 원래 문제로
    되돌아가고, 통과시키면 한도가 없는 우회로가 된다(fail-closed)."""

    async def body():
        async with _env(admin_api_key="secret"):
            for bad in (None, "", "   "):
                with pytest.raises(HTTPException) as exc:
                    await get_limits(VIDEO, x_lyric_user=bad)
                assert exc.value.status_code == 400

    asyncio.run(body())


def test_legacy_rows_without_actor_are_not_counted_for_anyone():
    """축 전환 이전 기록(actor=NULL)은 누구의 사용량도 아니다 — 사후 복원할 근거가 없어
    비운 채로 남기고 집계에서 제외한다(공용 이용자로 채우면 문제를 재현한다)."""

    async def body():
        async with _env(admin_api_key="secret"):
            async with db_conn.async_session() as s:
                for _ in range(5):
                    s.add(ActionLog(action="generate", video_id=VIDEO))  # actor 없음
                    s.add(ActionLog(action="reset", video_id=VIDEO))
                await s.commit()

            for actor in (ACTOR, OTHER_ACTOR):
                resp = await get_limits(VIDEO, x_lyric_user=actor)
                assert resp.generate.used == 0
                assert resp.destructive.used == 0
                assert resp.generate.next_reset_at is None

    asyncio.run(body())


# ── link·upgrade 버킷 (2026-08-04 보강) ──────────────────────────────────


def test_link_bucket_is_independent_of_generate_and_destructive():
    """수동 잇기(action="link")는 generate/destructive와 겹치지 않는 자기만의 카운터다."""

    async def body():
        async with _env(admin_api_key="secret"):
            async with db_conn.async_session() as s:
                repo = ActionLogRepository(s)
                await repo.log("link", VIDEO, ACTOR)
                await repo.log("link", VIDEO, ACTOR)
                await s.commit()

            resp = await get_limits(VIDEO, x_lyric_user=ACTOR)
            assert resp.link.used == 2
            assert resp.link.limit == DAILY_LINK_LIMIT
            assert resp.link.remaining == DAILY_LINK_LIMIT - 2
            # 다른 버킷은 전혀 영향받지 않는다
            assert resp.generate.used == 0
            assert resp.destructive.used == 0

    asyncio.run(body())


def test_link_bucket_next_reset_at_matches_oldest_log_plus_window():
    async def body():
        async with _env(admin_api_key="secret"):
            oldest = _utc_naive(hours=-5)
            async with db_conn.async_session() as s:
                s.add(
                    ActionLog(
                        action="link",
                        video_id=VIDEO,
                        actor=ACTOR,
                        created_at=oldest,
                    )
                )
                await s.commit()

            resp = await get_limits(VIDEO, x_lyric_user=ACTOR)
            assert resp.link.used == 1
            expected = oldest + timedelta(hours=resp.window_hours)
            actual = datetime.fromisoformat(resp.link.next_reset_at)
            assert abs((actual - expected).total_seconds()) < 1

    asyncio.run(body())


def test_upgrade_bucket_is_independent_of_generate_and_destructive():
    """action="upgrade"는 generate/destructive와 겹치지 않는 자기만의 카운터다
    (2026-08-04 2차 정정 — 처음의 generate 복사 계약은 폐기됨)."""

    async def body():
        async with _env(admin_api_key="secret"):
            async with db_conn.async_session() as s:
                repo = ActionLogRepository(s)
                await repo.log("upgrade", VIDEO, ACTOR)
                await repo.log("upgrade", VIDEO, ACTOR)
                await s.commit()

            resp = await get_limits(VIDEO, x_lyric_user=ACTOR)
            assert resp.upgrade.used == 2
            assert resp.upgrade.remaining == resp.upgrade.limit - 2
            # 다른 버킷은 전혀 영향받지 않는다
            assert resp.generate.used == 0
            assert resp.destructive.used == 0
            assert resp.link.used == 0

    asyncio.run(body())


def test_upgrade_bucket_limit_reflects_daily_upgrade_limit_setting():
    """운영자가 daily_upgrade_limit(기본 10)을 바꾸면 그 값이 그대로 노출된다 —
    generate(DAILY_GENERATE_LIMIT, 코드 상수)와 달리 upgrade는 설정으로 조절 가능."""

    async def body():
        async with _env(admin_api_key="secret", daily_upgrade_limit=7):
            resp = await get_limits(VIDEO, x_lyric_user=ACTOR)
            assert resp.upgrade.limit == 7
            assert resp.upgrade.remaining == 7

    asyncio.run(body())


def test_upgrade_bucket_next_reset_at_matches_oldest_log_plus_window():
    async def body():
        async with _env(admin_api_key="secret"):
            oldest = _utc_naive(hours=-4)
            async with db_conn.async_session() as s:
                s.add(
                    ActionLog(
                        action="upgrade", video_id=VIDEO, actor=ACTOR, created_at=oldest
                    )
                )
                await s.commit()

            resp = await get_limits(VIDEO, x_lyric_user=ACTOR)
            assert resp.upgrade.used == 1
            expected = oldest + timedelta(hours=resp.window_hours)
            actual = datetime.fromisoformat(resp.upgrade.next_reset_at)
            assert abs((actual - expected).total_seconds()) < 1

    asyncio.run(body())


# ── destructive 합산 예산 (2026-08-10, docs/user-quota-spec.md §2) ──────────


def test_destructive_sums_reset_and_regenerate():
    """초기화와 강제 재생성은 **하나의 예산**이다 — 표시도 합계다.

    예전 계약(폐기): 둘 중 더 많이 쓴 쪽(max)을 보여줬고, 실제 집행도 각자 독립이라
    상한의 2배까지 쓸 수 있었다. 아래 reset 1 + regenerate 2는 예전이라면 used=2였다."""

    async def body():
        async with _env(admin_api_key="secret", daily_destructive_limit=4):
            async with db_conn.async_session() as s:
                repo = ActionLogRepository(s)
                await repo.log("reset", VIDEO, ACTOR)
                await repo.log("regenerate", VIDEO, ACTOR)
                await repo.log("regenerate", VIDEO, ACTOR)
                await s.commit()

            resp = await get_limits(VIDEO, x_lyric_user=ACTOR)
            assert resp.destructive.used == 3  # max(1, 2)=2가 아니라 1+2=3
            assert resp.destructive.remaining == 1

    asyncio.run(body())


def test_lookup_itself_does_not_consume_the_quota():
    """조회를 여러 번 반복해도 action_logs는 늘지 않는다 — 조회가 한도를 깎아먹지 않는다."""

    async def body():
        async with _env(admin_api_key="secret"):
            await get_limits(VIDEO, x_lyric_user=ACTOR)
            await get_limits(VIDEO, x_lyric_user=ACTOR)
            await get_limits(VIDEO, x_lyric_user=ACTOR)

            resp = await get_limits(VIDEO, x_lyric_user=ACTOR)
            assert resp.generate.used == 0
            assert resp.destructive.used == 0

    asyncio.run(body())


def test_remaining_floors_at_zero_when_over_limit():
    """used > limit이어도 remaining은 0이다 — 음수를 보내면 확장이 그대로 그리고 소진
    표시도 안 붙는다(응답 계약)."""

    async def body():
        async with _env(admin_api_key="secret", daily_destructive_limit=1):
            async with db_conn.async_session() as s:
                repo = ActionLogRepository(s)
                await repo.log("reset", VIDEO, ACTOR)
                await repo.log("reset", VIDEO, ACTOR)
                await repo.log("regenerate", VIDEO, ACTOR)
                await s.commit()

            resp = await get_limits(VIDEO, x_lyric_user=ACTOR)
            assert resp.destructive.used == 3
            assert resp.destructive.remaining == 0  # 음수가 아니라 0으로 floor

    asyncio.run(body())


def test_required_fields_are_always_present():
    """필수 항목(enforced·generate·destructive·window_hours)은 항상 포함된다 — 확장에
    누락 대비 방어 코드가 없다."""

    async def body():
        async with _env(admin_api_key="secret"):
            for resp in (
                await get_limits(VIDEO, x_lyric_user=ACTOR),
                await get_limits(VIDEO, x_api_key="secret", x_lyric_user=ACTOR),
            ):
                payload = resp.model_dump()
                for key in ("enforced", "generate", "destructive", "window_hours"):
                    assert key in payload and payload[key] is not None
                for bucket in ("generate", "link", "upgrade", "destructive"):
                    assert payload[bucket]["limit"] > 0  # limit=0을 무제한 뜻으로 쓰지 않는다
                    assert payload[bucket]["remaining"] >= 0

    asyncio.run(body())


# ── 어드민 관측 (2026-08-10, docs/user-quota-spec.md §5) ───────────────────


def test_admin_response_is_unenforced_with_real_usage():
    """면제 이용자의 응답은 "강제 아님 + 실제 사용량"이다. 강제 형태로 소진값을 보내면
    확장이 면제 이용자를 차단 상태로 그린다(실측 2026-08-10: 당일 5건을 쓴 소유자에게
    20/20 만땅이 보이던 반대 방향의 거짓말도 같은 뿌리)."""

    async def body():
        async with _env(admin_api_key="secret", daily_destructive_limit=1):
            async with db_conn.async_session() as s:
                repo = ActionLogRepository(s)
                await repo.log("generate", VIDEO, ACTOR)
                await repo.log("reset", VIDEO, ACTOR)
                await repo.log("regenerate", VIDEO, ACTOR)
                await s.commit()

            admin = await get_limits(VIDEO, x_api_key="secret", x_lyric_user=ACTOR)
            assert admin.enforced is False  # 확장은 "무제한 (사용 N회)"로 그린다
            assert admin.generate.used == 1  # 만땅이 아니라 실제 사용량
            assert admin.destructive.used == 2
            assert admin.destructive.limit > 0  # limit=0("0/0 소진")으로 무제한을 표현하지 않는다

            # 같은 데이터를 비어드민으로 조회하면 강제 형태로 나온다
            user = await get_limits(VIDEO, x_lyric_user=ACTOR)
            assert user.enforced is True
            assert user.destructive.used == 2
            assert user.destructive.remaining == 0

    asyncio.run(body())


# ── next_reset_at (2026-08-04 보강) ──────────────────────────────────────


def test_next_reset_at_is_none_when_unused():
    """쓴 게 없으면 줄어들 것도 없다 — None(빈 UI 상태, "리셋 시각 없음")."""

    async def body():
        async with _env(admin_api_key="secret"):
            resp = await get_limits(VIDEO, x_lyric_user=ACTOR)
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
                s.add(
                    ActionLog(
                        action="generate", video_id=VIDEO, actor=ACTOR, created_at=oldest
                    )
                )
                s.add(
                    ActionLog(
                        action="generate", video_id=VIDEO, actor=ACTOR, created_at=newer
                    )
                )
                await s.commit()

            resp = await get_limits(VIDEO, x_lyric_user=ACTOR)
            assert resp.generate.used == 2
            expected = oldest + timedelta(hours=resp.window_hours)
            actual = datetime.fromisoformat(resp.generate.next_reset_at)
            assert abs((actual - expected).total_seconds()) < 1

    asyncio.run(body())


def test_destructive_next_reset_at_uses_the_earliest_record_of_the_combined_budget():
    """합산 예산의 회복 시각은 **가장 이른 기록**이 창을 벗어나는 시점이다 — 합계는 어느
    한 건만 빠져도 값이 줄기 때문이다.

    이 단언은 예전 계약(독립 집계 + max 표시 시절의 "동률이면 더 늦은 쪽")을 뒤집는다:
    그 규칙은 두 행위가 각자의 카운터를 max로 겨루던 시절에만 옳았다. 지금 더 늦은 쪽을
    내면 실제로는 그 전에 이미 한 칸이 회복되는데 안 된다고 안내하는 거짓말이 된다."""

    async def body():
        async with _env(admin_api_key="secret", daily_destructive_limit=4):
            reset_oldest = _utc_naive(hours=-3)  # 3시간 전 -> 만료 now+21h (더 이르다)
            regenerate_oldest = _utc_naive(hours=-1)  # 1시간 전 -> 만료 now+23h (더 늦다)
            async with db_conn.async_session() as s:
                s.add(
                    ActionLog(
                        action="reset", video_id=VIDEO, actor=ACTOR, created_at=reset_oldest
                    )
                )
                s.add(
                    ActionLog(
                        action="regenerate",
                        video_id=VIDEO,
                        actor=ACTOR,
                        created_at=regenerate_oldest,
                    )
                )
                await s.commit()

            resp = await get_limits(VIDEO, x_lyric_user=ACTOR)
            assert resp.destructive.used == 2  # 합계 — max(1,1)=1이 아니다
            expected = reset_oldest + timedelta(hours=resp.window_hours)  # 더 이른 쪽
            actual = datetime.fromisoformat(resp.destructive.next_reset_at)
            assert abs((actual - expected).total_seconds()) < 1
            # 더 늦은(예전 규칙) 후보였다면 이 값과 같았을 것 — 회귀를 못박는다
            wrong = regenerate_oldest + timedelta(hours=resp.window_hours)
            assert abs((actual - wrong).total_seconds()) > 3600

    asyncio.run(body())


def test_destructive_next_reset_at_ignores_records_outside_the_window():
    """창(24h) 밖 기록은 세지도, 회복 시각을 정하지도 않는다."""

    async def body():
        async with _env(admin_api_key="secret", daily_destructive_limit=4):
            expired = _utc_naive(hours=-30)  # 창 밖
            inside = _utc_naive(hours=-2)
            async with db_conn.async_session() as s:
                s.add(
                    ActionLog(
                        action="reset", video_id=VIDEO, actor=ACTOR, created_at=expired
                    )
                )
                s.add(
                    ActionLog(
                        action="regenerate", video_id=VIDEO, actor=ACTOR, created_at=inside
                    )
                )
                await s.commit()

            resp = await get_limits(VIDEO, x_lyric_user=ACTOR)
            assert resp.destructive.used == 1
            expected = inside + timedelta(hours=resp.window_hours)
            actual = datetime.fromisoformat(resp.destructive.next_reset_at)
            assert abs((actual - expected).total_seconds()) < 1

    asyncio.run(body())
