"""이용자 단위 일일 한도 — 소비 경로(생성·재생성·초기화·수동 잇기) 회귀 테스트.

SSOT: docs/user-quota-spec.md (운영자 결정 2026-08-10). 조회 응답 계약은 test_limits.py가,
앞단 헤더의 해석 자체는 test_quota_headers.py가, 여기서는 **실제로 예산을 깎는 경로**를
못박는다.

기존 서버 테스트 규약 그대로: 격리된 in-memory SQLite로 connection.async_session을
몽키패치하고 라우트 코루틴을 직접 await한다(httpx 불사용).

여기서 못박는 계약:
  ① 이용자 A/B는 같은 영상에서도 각자의 예산을 끝까지 쓴다(§1) — 예전에는 (행위, 영상)
     집계라 A가 다 쓰면 그 영상을 처음 여는 B가 0회로 시작했다.
  ② 같은 이용자는 영상을 옮겨도 같은 예산을 이어 쓴다(§1) — 영상만 바꾸면 예산이 새로
     생기던 구멍이 닫혔다.
  ③ 식별자 없는 요청은 거절되고 기록도 남지 않는다(§1, fail-closed).
  ④ 초기화 + 강제 재생성은 **하나의 합산 예산**(기본 4회/24h)이다(§2).
  ⑤ 재생성 요청은 세 갈래로 분류된다(§3) — 깊이 지정/구세대 싱크는 업그레이드, 나머지
     force는 파괴적.
  ⑥ 진행 중인 잡에 합류하는 요청은 어느 예산도 소비하지 않는다(§3) — 한도 검사가 합류
     판정 **뒤에** 있어야 성립한다.
  ⑦ 지울 것이 없는 초기화는 예산을 소비하지 않는다(§4, 멱등성).
  ⑧ 어드민은 거절만 면제받고 사용량은 기록된다(§5).
  ⑨ **한도값은 앞단이 준다**(§6) — 그 값으로 집행하고 그 값을 표시한다. 못 받으면 서버
     기본값으로 내려앉되 요청은 통과하고 경고가 남는다(조용한 티어 무력화 금지).
  ⑩ **link 버킷은 수동 잇기다**(§7) — POST /api/sync/link를 이용자 단위로 센다.
  ⑪ **운영 작업 식별자**(§8)는 사람 예산에 섞이지 않고, 면제받되 기록은 남는다.
  ⑫ 마이그레이션은 멱등하다 — 구 스키마 파일 DB에 init_db()를 두 번 돌려도 안전하다.
"""

import asyncio
import contextlib
import logging

import pytest
from fastapi import BackgroundTasks, HTTPException
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from everyric2.config.settings import get_settings
from everyric2.server import worker as worker_core
from everyric2.server.api import sync as sync_api
from everyric2.server.api.limits import get_limits
from everyric2.server.api.quota_headers import (
    OPS_ACTOR_BULK_INGEST,
    reset_limit_warning_state,
)
from everyric2.server.api.sync import (
    GenerateRequest,
    RegenerateRequest,
    SyncLinkRequest,
    create_sync_link,
    generate_sync,
    regenerate_sync,
    reset_video_syncs,
)
from everyric2.server.db import connection as db_conn
from everyric2.server.db.models import ActionLog, Base
from everyric2.server.db.repository import ActionLogRepository, SyncRepository

VIDEO = "QUOTAVIDEO1"
OTHER_VIDEO = "QUOTAVIDEO2"
ACTOR_A = "user-quota-a"
ACTOR_B = "user-quota-b"
ADMIN_KEY = "admin-secret"
LYRICS = "첫 줄\n두 번째 줄"
SEGMENTS = [{"text": "첫 줄", "start": 1.0, "end": 2.0}]


@pytest.fixture(autouse=True)
def _fresh_warn_state():
    """앞단 한도값 부재 경고는 첫 1회 + 샘플링이라 프로세스 수명 동안 누적된다 —
    비우지 않으면 뒤 테스트에서 "경고가 안 나온다"는 거짓 실패가 난다."""
    reset_limit_warning_state()
    yield
    reset_limit_warning_state()


@contextlib.asynccontextmanager
async def _env(admin_api_key: str = ADMIN_KEY, **server_overrides):
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
    # local_worker=False: 잡을 queued로만 마킹한다 — 기본값(True)이면 실제 다운로드·정렬
    # 파이프라인을 백그라운드에 얹는다(유닛 테스트에서 돌릴 수 없다).
    overrides = {"admin_api_key": admin_api_key, "local_worker": False, **server_overrides}
    saved = {k: getattr(server, k) for k in overrides}
    for k, v in overrides.items():
        object.__setattr__(server, k, v)

    def _clear():
        worker_core._PENDING_FORCE.clear()
        worker_core._PENDING_LINE_META.clear()
        worker_core._PENDING_ATTRIBUTION.clear()
        worker_core._PENDING_TITLE.clear()
        worker_core._PENDING_META_WAIT.clear()

    _clear()
    try:
        yield sm
    finally:
        db_conn.async_session = orig
        for k, v in saved.items():
            object.__setattr__(server, k, v)
        _clear()
        await engine.dispose()


async def _seed_sync(video_id=VIDEO, *, legacy=False, lyrics_hash="seed-hash") -> None:
    """이 영상에 싱크 하나를 심는다. legacy=True면 engine_version을 비워 **스탬프 이전
    세대**(구세대 싱크)로 만든다 — 재분류 §3의 두 번째 갈래가 그 조건을 본다."""
    async with db_conn.async_session() as s:
        await SyncRepository(s).create(
            video_id=video_id,
            lyrics_hash=lyrics_hash,
            timestamps=SEGMENTS,
            engine="ctc",
            engine_version=None if legacy else "cur-stack-1",
        )
        await s.commit()


async def _count(action, actor) -> int:
    async with db_conn.async_session() as s:
        return await ActionLogRepository(s).count_recent(action, actor)


async def _total_logs() -> int:
    async with db_conn.async_session() as s:
        return int((await s.execute(select(func.count()).select_from(ActionLog))).scalar_one())


# ── ① 이용자별 독립 예산 ────────────────────────────────────────────


def test_two_actors_each_get_their_own_budget_on_the_same_video():
    async def body():
        async with _env():
            prev = sync_api.DAILY_GENERATE_LIMIT
            sync_api.DAILY_GENERATE_LIMIT = 2
            try:
                # A가 상한(2)까지 쓴다 — 가사를 바꿔 캐시 히트·합류를 비켜 매번 새 잡을 만든다
                for i in range(2):
                    await generate_sync(
                        GenerateRequest(video_id=VIDEO, lyrics=f"{LYRICS}\nA{i}"),
                        BackgroundTasks(),
                        x_lyric_user=ACTOR_A,
                    )
                with pytest.raises(HTTPException) as exc:
                    await generate_sync(
                        GenerateRequest(video_id=VIDEO, lyrics=f"{LYRICS}\nA초과"),
                        BackgroundTasks(),
                        x_lyric_user=ACTOR_A,
                    )
                assert exc.value.status_code == 429

                # B는 **같은 영상**인데도 자기 몫을 그대로 쓴다
                for i in range(2):
                    resp = await generate_sync(
                        GenerateRequest(video_id=VIDEO, lyrics=f"{LYRICS}\nB{i}"),
                        BackgroundTasks(),
                        x_lyric_user=ACTOR_B,
                    )
                    assert resp.status == "processing"
                with pytest.raises(HTTPException) as exc:
                    await generate_sync(
                        GenerateRequest(video_id=VIDEO, lyrics=f"{LYRICS}\nB초과"),
                        BackgroundTasks(),
                        x_lyric_user=ACTOR_B,
                    )
                assert exc.value.status_code == 429

                assert await _count("generate", ACTOR_A) == 2
                assert await _count("generate", ACTOR_B) == 2
            finally:
                sync_api.DAILY_GENERATE_LIMIT = prev

    asyncio.run(body())


def test_actor_budget_follows_the_user_across_videos():
    async def body():
        async with _env():
            prev = sync_api.DAILY_GENERATE_LIMIT
            sync_api.DAILY_GENERATE_LIMIT = 2
            try:
                await generate_sync(
                    GenerateRequest(video_id=VIDEO, lyrics=LYRICS),
                    BackgroundTasks(),
                    x_lyric_user=ACTOR_A,
                )
                await generate_sync(
                    GenerateRequest(video_id=OTHER_VIDEO, lyrics=LYRICS),
                    BackgroundTasks(),
                    x_lyric_user=ACTOR_A,
                )
                # 세 번째는 어느 영상이든 막힌다 — 영상을 바꿔도 예산은 하나다
                with pytest.raises(HTTPException) as exc:
                    await generate_sync(
                        GenerateRequest(video_id="QUOTAVIDEO3", lyrics=LYRICS),
                        BackgroundTasks(),
                        x_lyric_user=ACTOR_A,
                    )
                assert exc.value.status_code == 429
                assert await _count("generate", ACTOR_A) == 2
            finally:
                sync_api.DAILY_GENERATE_LIMIT = prev

    asyncio.run(body())


# ── ③ 식별자 없는 요청 ──────────────────────────────────────────────


@pytest.mark.parametrize("actor", [None, "", "   "])
def test_missing_actor_is_rejected_and_leaves_no_log(actor):
    async def body():
        async with _env() as sm:
            with pytest.raises(HTTPException) as exc:
                await generate_sync(
                    GenerateRequest(video_id=VIDEO, lyrics=LYRICS),
                    BackgroundTasks(),
                    x_lyric_user=actor,
                )
            assert exc.value.status_code == 400
            assert "이용자" in exc.value.detail
            assert await _total_logs() == 0  # 거절된 요청은 기록도 남기지 않는다
            async with sm() as s:
                jobs = (await s.execute(text("SELECT COUNT(*) FROM jobs"))).scalar_one()
            assert jobs == 0  # 잡도 만들지 않는다

    asyncio.run(body())


def test_actor_is_not_required_when_limits_are_off():
    """한도 미강제 배포(로컬 단일 사용자)는 예전 그대로 식별자 없이 통과한다."""

    async def body():
        async with _env(admin_api_key=""):
            for i in range(3):
                resp = await generate_sync(
                    GenerateRequest(video_id=VIDEO, lyrics=f"{LYRICS}\n{i}"),
                    BackgroundTasks(),
                )
                assert resp.status == "processing"
            assert await _total_logs() == 0  # 한도가 없으면 기록도 남기지 않는다

    asyncio.run(body())


# ── ④ 파괴적 행위 합산 예산 ─────────────────────────────────────────


def test_reset_and_force_regenerate_share_one_budget_of_four():
    """초기화 2 + 강제 재생성 2 뒤 다섯 번째가 429. 예전에는 각자 상한을 따로 소비해
    실제로는 상한의 2배까지 가능했다(표시는 max로 뭉쳐 한 예산처럼 보였다)."""

    async def body():
        async with _env(daily_destructive_limit=4):
            # 초기화 2회 — 지운 것이 있어야 예산을 소비한다(§4)
            for i in range(2):
                await _seed_sync(lyrics_hash=f"h{i}")
                res = await reset_video_syncs(VIDEO, x_lyric_user=ACTOR_A)
                assert res["removed_syncs"] == 1

            # 강제 재생성 2회 — 싱크가 없어 구세대 판정에도 걸리지 않는다(파괴적으로 분류)
            for i in range(2):
                await regenerate_sync(
                    RegenerateRequest(video_id=VIDEO, lyrics=f"{LYRICS}\nF{i}", force=True),
                    BackgroundTasks(),
                    x_lyric_user=ACTOR_A,
                )

            assert await _count("reset", ACTOR_A) == 2
            assert await _count("regenerate", ACTOR_A) == 2
            limits = await get_limits(VIDEO, x_lyric_user=ACTOR_A)
            assert limits.destructive.used == 4  # 합계
            assert limits.destructive.remaining == 0

            # 다섯 번째는 초기화든 강제 재생성이든 막힌다
            await _seed_sync(lyrics_hash="h-last")
            with pytest.raises(HTTPException) as exc:
                await reset_video_syncs(VIDEO, x_lyric_user=ACTOR_A)
            assert exc.value.status_code == 429
            with pytest.raises(HTTPException) as exc:
                await regenerate_sync(
                    RegenerateRequest(video_id=VIDEO, lyrics=f"{LYRICS}\nF마지막", force=True),
                    BackgroundTasks(),
                    x_lyric_user=ACTOR_A,
                )
            assert exc.value.status_code == 429
            # 거절된 초기화는 아무것도 지우지 않았다
            async with db_conn.async_session() as s:
                assert len(await SyncRepository(s).get_by_video(VIDEO)) == 1

    asyncio.run(body())


# ── ⑤ 재생성 재분류 (세 갈래) ────────────────────────────────────────


def test_force_with_min_depth_counts_as_upgrade():
    """확장의 API 래퍼가 force를 무조건 붙인다 — 깊이 지정이 있으면 업그레이드다."""

    async def body():
        async with _env():
            await _seed_sync()
            await regenerate_sync(
                RegenerateRequest(
                    video_id=VIDEO, lyrics=LYRICS, force=True, min_depth="heavy"
                ),
                BackgroundTasks(),
                x_lyric_user=ACTOR_A,
            )
            assert await _count("upgrade", ACTOR_A) == 1
            assert await _count("regenerate", ACTOR_A) == 0
            assert await _count("generate", ACTOR_A) == 0

    asyncio.run(body())


def test_force_without_depth_on_a_legacy_sync_counts_as_upgrade():
    """구세대 싱크(engine_version 없음)용 업그레이드 버튼은 깊이 인자 없이 호출한다 —
    그 형태를 업그레이드로 구제한다(판정이 그 버튼보다 넓다는 것은 인정된 결정)."""

    async def body():
        async with _env():
            await _seed_sync(legacy=True)
            await regenerate_sync(
                RegenerateRequest(video_id=VIDEO, lyrics=LYRICS, force=True),
                BackgroundTasks(),
                x_lyric_user=ACTOR_A,
            )
            assert await _count("upgrade", ACTOR_A) == 1
            assert await _count("regenerate", ACTOR_A) == 0

    asyncio.run(body())


def test_force_without_depth_on_a_current_sync_counts_as_destructive():
    """현행 스택으로 만든 싱크에 대한 force는 그대로 파괴적 재생성이다."""

    async def body():
        async with _env():
            await _seed_sync(legacy=False)
            await regenerate_sync(
                RegenerateRequest(video_id=VIDEO, lyrics=LYRICS, force=True),
                BackgroundTasks(),
                x_lyric_user=ACTOR_A,
            )
            assert await _count("regenerate", ACTOR_A) == 1
            assert await _count("upgrade", ACTOR_A) == 0

    asyncio.run(body())


def test_plain_regenerate_still_counts_as_generate():
    """force도 깊이도 없는 재생성은 GPU 소비가 /generate와 같으므로 같은 예산을 쓴다."""

    async def body():
        async with _env():
            await regenerate_sync(
                RegenerateRequest(video_id=VIDEO, lyrics=LYRICS),
                BackgroundTasks(),
                x_lyric_user=ACTOR_A,
            )
            assert await _count("generate", ACTOR_A) == 1
            assert await _count("upgrade", ACTOR_A) == 0
            assert await _count("regenerate", ACTOR_A) == 0

    asyncio.run(body())


# ── ⑥ 진행 중인 잡 합류 ─────────────────────────────────────────────


def test_joining_an_active_job_consumes_no_budget():
    """연타로 이미 도는 잡에 합류하는 재시도는 GPU를 쓰지 않는다 — 예산도 쓰면 안 된다.
    한도 검사가 합류 판정 **앞**에 있던 시절에는 강제 재생성 연타가 예산을 태웠다."""

    async def body():
        async with _env(daily_destructive_limit=1):
            await _seed_sync(legacy=False)
            first = await regenerate_sync(
                RegenerateRequest(video_id=VIDEO, lyrics=LYRICS, force=True),
                BackgroundTasks(),
                x_lyric_user=ACTOR_A,
            )
            assert await _count("regenerate", ACTOR_A) == 1

            # 같은 가사로 다시 — 상한이 1인데도 429가 아니라 그 잡에 합류한다
            for _ in range(3):
                joined = await regenerate_sync(
                    RegenerateRequest(video_id=VIDEO, lyrics=LYRICS, force=True),
                    BackgroundTasks(),
                    x_lyric_user=ACTOR_A,
                )
                assert joined.job_id == first.job_id
            assert await _count("regenerate", ACTOR_A) == 1  # 그대로

    asyncio.run(body())


# ── ⑦ 초기화 멱등성 ────────────────────────────────────────────────


def test_reset_with_nothing_to_delete_consumes_no_budget():
    """응답 유실로 재시도해도 지울 것이 없으면 아무 일도 안 일어난다 — 예산도 그대로."""

    async def body():
        async with _env(daily_destructive_limit=1):
            for _ in range(3):
                res = await reset_video_syncs(VIDEO, x_lyric_user=ACTOR_A)
                assert res["removed_syncs"] == 0 and res["removed_links"] == 0
            assert await _count("reset", ACTOR_A) == 0

            # 예산이 그대로라 진짜 초기화는 여전히 가능하다
            await _seed_sync()
            res = await reset_video_syncs(VIDEO, x_lyric_user=ACTOR_A)
            assert res["removed_syncs"] == 1
            assert await _count("reset", ACTOR_A) == 1

    asyncio.run(body())


# ── ⑧ 어드민 관측 ──────────────────────────────────────────────────


def test_admin_bypasses_the_limit_but_usage_is_recorded():
    """어드민은 거절만 면제받는다 — 예전에는 검사 자체를 건너뛰어 기록도 안 남아 소유자
    화면이 항상 만땅으로 보였다(실측 2026-08-10)."""

    async def body():
        async with _env(daily_destructive_limit=1):
            for i in range(3):
                await _seed_sync(lyrics_hash=f"h{i}")
                res = await reset_video_syncs(
                    VIDEO, x_api_key=ADMIN_KEY, x_lyric_user=ACTOR_A
                )
                assert res["removed_syncs"] == 1  # 상한(1)을 넘겨도 통과한다
            assert await _count("reset", ACTOR_A) == 3  # 기록은 남는다

            limits = await get_limits(VIDEO, x_api_key=ADMIN_KEY, x_lyric_user=ACTOR_A)
            assert limits.enforced is False  # 확장은 "무제한 (사용 3회)"로 그린다
            assert limits.destructive.used == 3

    asyncio.run(body())


# ── ⑨ 앞단이 주는 한도값 (§6) ───────────────────────────────────────


def test_gateway_limit_is_enforced_instead_of_the_server_default():
    """앞단이 준 값이 **집행**에 쓰인다 — 서버 기본값(20)보다 작은 값을 주면 거기서 막힌다.

    예전에는 두 값 중 작은 쪽이 항상 먼저 걸려, 앞단이 티어에 준 값이 서버 상수보다 크면
    영원히 도달할 수 없었다(trusted 60 · 여기 전원 20 → 실효 20)."""

    async def body():
        async with _env():
            for i in range(2):
                await generate_sync(
                    GenerateRequest(video_id=VIDEO, lyrics=f"{LYRICS}\n{i}"),
                    BackgroundTasks(),
                    x_lyric_user=ACTOR_A,
                    x_lyric_limits="generate=2",
                )
            with pytest.raises(HTTPException) as exc:
                await generate_sync(
                    GenerateRequest(video_id=VIDEO, lyrics=f"{LYRICS}\n초과"),
                    BackgroundTasks(),
                    x_lyric_user=ACTOR_A,
                    x_lyric_limits="generate=2",
                )
            assert exc.value.status_code == 429
            assert "2회/24시간" in exc.value.detail  # 사유에도 앞단 값이 실린다

    asyncio.run(body())


def test_gateway_limit_above_the_server_default_actually_lets_you_through():
    """티어 차등의 요점 — 앞단이 서버 상수보다 **큰** 값을 주면 그만큼 통과해야 한다."""

    async def body():
        async with _env():
            prev = sync_api.DAILY_GENERATE_LIMIT
            sync_api.DAILY_GENERATE_LIMIT = 2  # 서버 기본값을 낮게 잡아 두고
            try:
                for i in range(5):  # 앞단이 5를 주면 5번 통과한다
                    resp = await generate_sync(
                        GenerateRequest(video_id=VIDEO, lyrics=f"{LYRICS}\n{i}"),
                        BackgroundTasks(),
                        x_lyric_user=ACTOR_A,
                        x_lyric_limits="generate=5",
                    )
                    assert resp.status == "processing"
                assert await _count("generate", ACTOR_A) == 5
            finally:
                sync_api.DAILY_GENERATE_LIMIT = prev

    asyncio.run(body())


def test_gateway_limit_is_reflected_in_the_limits_response():
    """표시와 집행이 같은 값을 써야 한다 — 다르면 "20/20인데 429"가 난다."""

    async def body():
        async with _env():
            limits = await get_limits(
                VIDEO,
                x_lyric_user=ACTOR_A,
                x_lyric_limits="generate=60,link=25,upgrade=30,destructive=8",
            )
            assert (limits.generate.limit, limits.generate.remaining) == (60, 60)
            assert limits.link.limit == 25
            assert limits.upgrade.limit == 30
            assert limits.destructive.limit == 8
            assert limits.enforced is True

    asyncio.run(body())


def test_unlimited_from_the_gateway_is_never_sent_as_a_limit_value():
    """무제한은 enforced=false로 표현한다 — 상한 0을 그 뜻으로 쓰면 확장이 0/0 소진으로
    그린다(응답 계약)."""

    async def body():
        async with _env():
            await generate_sync(
                GenerateRequest(video_id=VIDEO, lyrics=LYRICS),
                BackgroundTasks(),
                x_lyric_user=ACTOR_A,
                x_lyric_limits="generate=unlimited,link=0,upgrade=-1,destructive=none",
            )
            limits = await get_limits(
                VIDEO,
                x_lyric_user=ACTOR_A,
                x_lyric_limits="generate=unlimited,link=0,upgrade=-1,destructive=none",
            )
            assert limits.enforced is False  # 전 버킷 무제한
            for bucket in (limits.generate, limits.link, limits.upgrade, limits.destructive):
                assert bucket.limit > 0  # 0을 무제한의 뜻으로 싣지 않는다
                assert bucket.remaining >= 0
            assert limits.generate.used == 1  # 무제한이어도 사용량은 기록된다

    asyncio.run(body())


def test_partially_unlimited_stays_enforced_and_never_looks_exhausted():
    """한 버킷만 무제한이면 enforced는 True를 유지한다 — False로 내면 한도가 살아 있는
    버킷까지 확장이 무제한으로 그려 429가 예고 없이 튀어나온다."""

    async def body():
        async with _env():
            header = "generate=unlimited,link=25,upgrade=30,destructive=8"
            for i in range(3):
                await generate_sync(
                    GenerateRequest(video_id=VIDEO, lyrics=f"{LYRICS}\n{i}"),
                    BackgroundTasks(),
                    x_lyric_user=ACTOR_A,
                    x_lyric_limits=header,
                )
            limits = await get_limits(VIDEO, x_lyric_user=ACTOR_A, x_lyric_limits=header)
            assert limits.enforced is True
            assert limits.generate.used == 3
            assert limits.generate.remaining > 0  # 무제한 버킷은 소진으로 보이지 않는다
            assert limits.destructive.limit == 8

    asyncio.run(body())


def test_missing_gateway_limit_falls_back_and_warns_but_never_rejects(caplog):
    """🔴 못 받아도 요청은 통과한다(배포 순서 유연성) — 다만 티어 무력화 상태이므로
    **경고가 남아야** 한다. 조용히 넘어가면 "티어를 줬는데 왜 안 먹지"를 아무도 못 찾는다."""

    async def body():
        async with _env():
            prev = sync_api.DAILY_GENERATE_LIMIT
            sync_api.DAILY_GENERATE_LIMIT = 3
            try:
                with caplog.at_level(logging.WARNING):
                    for i in range(3):
                        resp = await generate_sync(
                            GenerateRequest(video_id=VIDEO, lyrics=f"{LYRICS}\n{i}"),
                            BackgroundTasks(),
                            x_lyric_user=ACTOR_A,
                        )
                        assert resp.status == "processing"  # 거절하지 않는다
                    # 서버 기본값이 그대로 집행된다
                    with pytest.raises(HTTPException) as exc:
                        await generate_sync(
                            GenerateRequest(video_id=VIDEO, lyrics=f"{LYRICS}\n초과"),
                            BackgroundTasks(),
                            x_lyric_user=ACTOR_A,
                        )
                    assert exc.value.status_code == 429
                messages = [r.getMessage() for r in caplog.records]
                assert any("x-lyric-limits" in m for m in messages), messages
                assert any("기본값" in m for m in messages), messages
            finally:
                sync_api.DAILY_GENERATE_LIMIT = prev

    asyncio.run(body())


def test_unparseable_gateway_limit_warns_and_falls_back(caplog):
    """앞단이 이상한 값을 보내도 요청을 죽이지 않는다 — 사유만 남기고 기본값을 쓴다."""

    async def body():
        async with _env():
            with caplog.at_level(logging.WARNING):
                resp = await generate_sync(
                    GenerateRequest(video_id=VIDEO, lyrics=LYRICS),
                    BackgroundTasks(),
                    x_lyric_user=ACTOR_A,
                    x_lyric_limits="generate=많이",
                )
            assert resp.status == "processing"
            assert caplog.records, "해석 실패가 조용히 넘어갔다"

    asyncio.run(body())


# ── ⑩ 수동 잇기가 link 버킷이다 (§7) ────────────────────────────────


async def _seed_link_source(video_id: str) -> None:
    """수동 잇기의 소스가 되려면 그 영상에 실제 싱크가 있어야 한다(create_sync_link 계약)."""
    await _seed_sync(video_id, lyrics_hash=f"src-{video_id}")


def test_manual_link_is_counted_per_actor_and_shows_in_the_link_bucket():
    """확장 배지 "커버 잇기"가 세는 것이 이제 이 행위다 — 예전에는 이 경로에 한도 검사도
    기록도 전혀 없었고, 배지는 사용자가 요청한 적 없는 배경 요청을 세고 있었다(§7)."""

    async def body():
        async with _env():
            await _seed_link_source("LINKSOURCE1")
            for i in range(2):
                link = await create_sync_link(
                    SyncLinkRequest(
                        video_id=f"LINKCOVER{i}0", source_video_id="LINKSOURCE1", offset_sec=1.0
                    ),
                    x_lyric_user=ACTOR_A,
                )
                assert link.verified is False

            assert await _count("link", ACTOR_A) == 2
            assert await _count("link", ACTOR_B) == 0  # 이용자 단위다

            limits = await get_limits(VIDEO, x_lyric_user=ACTOR_A)
            assert limits.link.used == 2
            assert limits.link.remaining == limits.link.limit - 2
            # 다른 예산은 건드리지 않는다
            assert limits.generate.used == 0 and limits.destructive.used == 0

    asyncio.run(body())


def test_manual_link_follows_the_gateway_limit():
    async def body():
        async with _env():
            await _seed_link_source("LINKSOURCE1")
            await create_sync_link(
                SyncLinkRequest(video_id="LINKCOVER10", source_video_id="LINKSOURCE1"),
                x_lyric_user=ACTOR_A,
                x_lyric_limits="link=1",
            )
            with pytest.raises(HTTPException) as exc:
                await create_sync_link(
                    SyncLinkRequest(video_id="LINKCOVER20", source_video_id="LINKSOURCE1"),
                    x_lyric_user=ACTOR_A,
                    x_lyric_limits="link=1",
                )
            assert exc.value.status_code == 429
            assert "커버 잇기" in exc.value.detail

    asyncio.run(body())


def test_manual_link_rejected_before_anything_is_created_consumes_no_budget():
    """자기 링크·소스 없음으로 거절되는 요청은 아무것도 만들지 않으므로 예산을 안 먹는다
    (생성 경로의 "GPU를 태우는 분기 직전에만 센다"와 같은 규율)."""

    async def body():
        async with _env():
            with pytest.raises(HTTPException) as exc:  # 소스에 싱크가 없다
                await create_sync_link(
                    SyncLinkRequest(video_id="LINKCOVER10", source_video_id="LINKSOURCE1"),
                    x_lyric_user=ACTOR_A,
                )
            assert exc.value.status_code == 400
            with pytest.raises(HTTPException):  # 자기 자신 링크
                await create_sync_link(
                    SyncLinkRequest(video_id="LINKCOVER10", source_video_id="LINKCOVER10"),
                    x_lyric_user=ACTOR_A,
                )
            assert await _count("link", ACTOR_A) == 0

    asyncio.run(body())


def test_manual_link_requires_an_actor_when_limits_are_enforced():
    """소비 경로이므로 §1의 fail-closed가 그대로 적용된다."""

    async def body():
        async with _env():
            await _seed_link_source("LINKSOURCE1")
            with pytest.raises(HTTPException) as exc:
                await create_sync_link(
                    SyncLinkRequest(video_id="LINKCOVER10", source_video_id="LINKSOURCE1"),
                    x_lyric_user=None,
                )
            assert exc.value.status_code == 400
            assert await _total_logs() == 0

    asyncio.run(body())


# ── ⑪ 운영 작업 식별자 (§8) ─────────────────────────────────────────


def test_ops_actor_does_not_share_a_humans_budget():
    """운영 스크립트가 사람 식별자를 빌려 쓰면 통계가 오염된다 — 전용 이름을 쓴다."""

    async def body():
        async with _env():
            prev = sync_api.DAILY_GENERATE_LIMIT
            sync_api.DAILY_GENERATE_LIMIT = 1
            try:
                await generate_sync(
                    GenerateRequest(video_id=VIDEO, lyrics=f"{LYRICS}\n사람"),
                    BackgroundTasks(),
                    x_lyric_user=ACTOR_A,
                )
                # 운영 작업이 여러 번 돌아도 사람 예산은 그대로 1이다
                for i in range(5):
                    await generate_sync(
                        GenerateRequest(video_id=VIDEO, lyrics=f"{LYRICS}\n운영{i}"),
                        BackgroundTasks(),
                        x_lyric_user=OPS_ACTOR_BULK_INGEST,
                    )
                assert await _count("generate", ACTOR_A) == 1
                assert await _count("generate", OPS_ACTOR_BULK_INGEST) == 5

                # 사람은 여전히 자기 상한에서 막힌다(운영 작업이 예산을 먹지 않았다)
                with pytest.raises(HTTPException) as exc:
                    await generate_sync(
                        GenerateRequest(video_id=VIDEO, lyrics=f"{LYRICS}\n사람2"),
                        BackgroundTasks(),
                        x_lyric_user=ACTOR_A,
                    )
                assert exc.value.status_code == 429
            finally:
                sync_api.DAILY_GENERATE_LIMIT = prev

    asyncio.run(body())


def test_ops_actor_is_exempt_from_rejection_but_always_recorded():
    """🔴 "어드민 키면 식별자 없이 통과"로 풀지 않는다(§8) — 면제하더라도 기록은 남는다.
    어드민 키 없이도 운영 식별자만으로 면제된다(대량 인제스트는 원래 그런 작업이다)."""

    async def body():
        async with _env(daily_destructive_limit=1):
            for i in range(3):
                await _seed_sync(lyrics_hash=f"h{i}")
                res = await reset_video_syncs(VIDEO, x_lyric_user=OPS_ACTOR_BULK_INGEST)
                assert res["removed_syncs"] == 1  # 상한(1)을 넘겨도 통과한다
            assert await _count("reset", OPS_ACTOR_BULK_INGEST) == 3  # 기록은 남는다

            limits = await get_limits(VIDEO, x_lyric_user=OPS_ACTOR_BULK_INGEST)
            assert limits.enforced is False  # 면제 이용자의 응답 형태(§5와 같다)
            assert limits.destructive.used == 3

    asyncio.run(body())


def test_ops_actor_still_needs_an_identifier():
    """운영 작업도 식별자 자체는 있어야 한다 — 없으면 기록을 남길 대상이 없다."""

    async def body():
        async with _env():
            with pytest.raises(HTTPException) as exc:
                await generate_sync(
                    GenerateRequest(video_id=VIDEO, lyrics=LYRICS),
                    BackgroundTasks(),
                    x_api_key=ADMIN_KEY,  # 어드민 키가 있어도 통과하지 않는다
                )
            assert exc.value.status_code == 400
            assert await _total_logs() == 0

    asyncio.run(body())


# ── 헤더 바인딩 ────────────────────────────────────────────────────


def test_quota_routes_bind_the_gateway_headers():
    """게이트웨이가 붙이는 헤더 이름이 실제로 라우트 파라미터에 묶여 있는지 못박는다.

    다른 테스트는 라우트 코루틴을 직접 부르므로(이 리포의 규약) FastAPI의 헤더 이름 변환
    (x_lyric_user → "x-lyric-user")을 검증하지 못한다 — 파라미터 이름을 오타내면 서버는
    조용히 모든 요청을 "식별자 없음"으로 보고 400을 내고, 한도 헤더 쪽은 더 조용하게
    "앞단 값 없음"으로 내려앉는다. app.routes 조사로 확인한다(test_sync_link.py의 라우트
    섀도잉 검증과 같은 방식 — HTTP 클라이언트 불사용).

    **상수와 파라미터 이름의 짝을 여기서 못박는다**: 이름은 quota_headers.py의 상수가
    SSOT지만, 실제 바인딩은 라우트 파라미터 이름이 결정한다. 상수만 바꾸고 파라미터를
    안 바꾸면 아무 오류 없이 헤더를 못 받게 된다(게이트웨이 확정 시 실제로 일어날 일).

    link-candidates는 **의도적으로 빠져 있다**: 그 경로는 2026-08-10부터 어떤 예산도
    소비하지 않으므로 식별자를 요구하지 않는다(docs/user-quota-spec.md §7)."""
    from fastapi.routing import APIRoute

    from everyric2.server.api.quota_headers import ACTOR_HEADER, LIMIT_HEADER
    from everyric2.server.main import app

    targets = {
        ("GET", "/api/limits/{video_id}"),
        ("POST", "/api/sync/generate"),
        ("POST", "/api/sync/regenerate"),
        ("POST", "/api/sync/generate-from-caption"),
        ("POST", "/api/sync/link"),
        ("DELETE", "/api/sync/{video_id}"),
    }
    seen = set()
    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        for method in route.methods:
            if (method, route.path) not in targets:
                continue
            aliases = {p.alias.lower() for p in route.dependant.header_params}
            assert ACTOR_HEADER in aliases, f"{method} {route.path}: {aliases}"
            assert LIMIT_HEADER in aliases, f"{method} {route.path}: {aliases}"
            seen.add((method, route.path))
    assert seen == targets  # 경로가 사라지거나 이름이 바뀌면 여기서 걸린다


def test_link_candidates_route_binds_no_quota_headers():
    """자동 후보 탐색은 예산을 소비하지 않으므로 쿼터 헤더를 받지 않는다(§7).

    받아 두고 안 쓰면 다음 사람이 "여기도 세고 있나 보다"로 읽고, 실제로 세기 시작하면
    라벨과 내용이 다시 어긋난다 — 그 어긋남이 §7이 고친 결함이다."""
    from fastapi.routing import APIRoute

    from everyric2.server.api.quota_headers import ACTOR_HEADER, LIMIT_HEADER
    from everyric2.server.main import app

    for route in app.routes:
        if isinstance(route, APIRoute) and route.path == "/api/sync/{video_id}/link-candidates":
            aliases = {p.alias.lower() for p in route.dependant.header_params}
            assert ACTOR_HEADER not in aliases and LIMIT_HEADER not in aliases, aliases
            return
    raise AssertionError("link-candidates 라우트를 못 찾았다")


# ── ⑨ 마이그레이션 멱등성 ───────────────────────────────────────────


def test_init_db_is_idempotent_on_a_legacy_action_logs_schema(tmp_path):
    """actor 열이 없던 시절의 파일 DB에 init_db()를 두 번 돌려도 오류가 없고, 기존 행은
    actor=NULL로 남는다(사후 복원할 근거가 없으므로 비운 채로 둔다)."""

    db_file = tmp_path / "legacy.db"
    url = f"sqlite+aiosqlite:///{db_file}"

    async def body():
        engine = create_async_engine(url)
        # 구 스키마 재현 — actor 열도, 복합 인덱스도 없다
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "CREATE TABLE action_logs ("
                    " id VARCHAR(36) NOT NULL,"
                    " action VARCHAR(16) NOT NULL,"
                    " video_id VARCHAR(32) NOT NULL,"
                    " created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,"
                    " PRIMARY KEY (id))"
                )
            )
            await conn.execute(
                text(
                    "INSERT INTO action_logs (id, action, video_id) "
                    "VALUES ('old-1', 'generate', 'OLDVIDEO001')"
                )
            )

        orig_engine, orig_url = db_conn.engine, db_conn.DATABASE_URL
        db_conn.engine, db_conn.DATABASE_URL = engine, url
        try:
            await db_conn.init_db()
            await db_conn.init_db()  # 두 번째 실행도 안전해야 한다
            async with engine.begin() as conn:
                cols = {row[1] for row in await conn.execute(text("PRAGMA table_info(action_logs)"))}
                assert "actor" in cols
                indexes = {
                    row[1] for row in await conn.execute(text("PRAGMA index_list(action_logs)"))
                }
                assert "ix_action_logs_actor_action_created" in indexes
                actor = (
                    await conn.execute(text("SELECT actor FROM action_logs WHERE id='old-1'"))
                ).scalar_one()
                assert actor is None
        finally:
            db_conn.engine, db_conn.DATABASE_URL = orig_engine, orig_url
            await engine.dispose()

    asyncio.run(body())
