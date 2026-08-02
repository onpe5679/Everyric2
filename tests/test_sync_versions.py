"""싱크 버저닝(sync_result_versions) 테스트 — 재처리 A/B 고스트 비교·(향후) 롤백의 재료.

기존 서버 테스트 규약을 그대로 따른다: 격리된 in-memory SQLite로 connection.async_session을
몽키패치하고 라우트 코루틴을 직접 await(asyncio.run). httpx/TestClient는 쓰지 않는다
(test_sync_link.py·test_api_lifecycle.py와 동일한 하네스).

여기서 못박는 계약:
  ① 최초 생성(같은 video_id의 기존 행이 없음)은 스냅샷을 만들지 않는다.
  ② 두 번째 이후 저장(SyncRepository.create)은 그 직전 최신 행을 스냅샷하고, video_id당
     스냅샷은 항상 최신 1건만 남는다(두 번 교체하면 첫 세대는 사라지고 직전 세대만 남음).
  ③ timestamps가 완전히 같은 재저장은 스냅샷을 만들지 않는다(무의미한 diff 방지).
  ④ GET /api/sync/{video_id}/previous 응답 모양 — found/timestamps/language/
     quality_score/created_at/replaced_at, 없으면 404가 아니라 found=false.
"""

import asyncio
import contextlib
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from everyric2.server.api.sync import get_previous_sync_version
from everyric2.server.db import connection as db_conn
from everyric2.server.db.models import Base, SyncResultVersion
from everyric2.server.db.repository import SyncRepository, SyncResultVersionRepository

VIDEO = "VERVERVERV1"

# SyncResult.created_at은 SQLite server_default=func.now()라 초 단위로 저장된다
# (JobRepository.count_queued_before 주석·test_api_lifecycle.py의
# test_job_status_returns_the_sync_this_job_actually_made가 이미 같은 제약을 명시).
# 실제 재처리는 GPU 정렬을 거쳐 수십 초~수 분이 걸려 같은 초 충돌이 사실상 없지만, 이
# 테스트처럼 연속 호출이 같은 초 안에 몰리면 "직전 최신 1건" 판정이 생성 순서를 못 잡는다
# — 세대 순서가 검증의 핵심이므로 다른 GET 조회 테스트들과 달리 여기서는 명시적으로
# created_at을 벌려 순서를 못박는다.
_BASE_TIME = datetime(2026, 8, 1, 12, 0, 0)


async def _force_creation_order(sm, rows: list) -> None:
    """rows(생성한 순서 그대로)의 created_at을 오름차순으로 명시 재기록해 "직전 최신"
    쿼리가 초 단위 동률 없이 그 순서를 그대로 읽게 만든다."""
    async with sm() as s:
        for i, row in enumerate(rows):
            fresh = await SyncRepository(s).get_by_id(row.id)
            fresh.created_at = _BASE_TIME + timedelta(minutes=i)
        await s.commit()

SEGMENTS_V1 = [{"text": "첫 세대 가사", "start": 1.0, "end": 2.0}]
SEGMENTS_V2 = [{"text": "두 번째 세대 가사", "start": 1.0, "end": 2.5}]
SEGMENTS_V3 = [{"text": "세 번째 세대 가사", "start": 1.0, "end": 3.0}]


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


async def _create(sm, *, video_id=VIDEO, lyrics_hash, segments, **kwargs):
    """production 경로와 같이 저장마다 새 세션을 열어(get_session 관례) 실제 DB 왕복을
    거친 값으로 스냅샷 비교가 이뤄지는지까지 검증한다(같은 세션의 identity map에 남은
    파이썬 객체 동일성에 기대지 않는다)."""
    async with sm() as s:
        row = await SyncRepository(s).create(
            video_id=video_id, lyrics_hash=lyrics_hash, timestamps=segments, engine="ctc", **kwargs
        )
        await s.commit()
        return row


async def _version_count(sm) -> int:
    async with sm() as s:
        return len((await s.execute(select(SyncResultVersion))).scalars().all())


async def _get_version(sm, video_id=VIDEO):
    async with sm() as s:
        return await SyncResultVersionRepository(s).get(video_id)


# ── ① 최초 생성은 스냅샷을 만들지 않는다 ──────────────────────────


def test_first_create_makes_no_snapshot():
    async def body():
        async with _env() as sm:
            await _create(sm, lyrics_hash="h1", segments=SEGMENTS_V1)

            assert await _get_version(sm) is None
            assert await _version_count(sm) == 0

    asyncio.run(body())


# ── ② 교체마다 스냅샷은 항상 "직전 최신" 1건만 유지 ────────────────


def test_second_create_snapshots_the_first_generation():
    async def body():
        async with _env() as sm:
            await _create(sm, lyrics_hash="h1", segments=SEGMENTS_V1, quality_score=0.5)
            await _create(sm, lyrics_hash="h2", segments=SEGMENTS_V2, quality_score=0.9)

            snap = await _get_version(sm)
            assert snap is not None
            assert snap.timestamps["segments"] == SEGMENTS_V1
            assert snap.quality_score == 0.5
            assert await _version_count(sm) == 1

    asyncio.run(body())


def test_third_create_replaces_the_snapshot_with_only_the_latest_previous():
    """두 번 교체하면 1세대(v1)는 완전히 사라지고 직전(v2)만 남아야 한다 — 여러 세대를
    쌓지 않는다는 "1세대 보관" 설계 요구의 핵심 계약."""

    async def body():
        async with _env() as sm:
            v1 = await _create(sm, lyrics_hash="h1", segments=SEGMENTS_V1)
            v2 = await _create(sm, lyrics_hash="h2", segments=SEGMENTS_V2)
            # v1·v2가 같은 초에 만들어지면(빠른 연속 호출) "직전 최신" 쿼리가 created_at
            # 동률을 못 갈라 v3 스냅샷 대상이 흔들린다 — 순서를 명시적으로 못박는다
            # (실제 재처리는 GPU 정렬로 수십 초 이상 걸려 이 동률이 사실상 없다).
            await _force_creation_order(sm, [v1, v2])
            await _create(sm, lyrics_hash="h3", segments=SEGMENTS_V3)

            snap = await _get_version(sm)
            assert snap.timestamps["segments"] == SEGMENTS_V2  # v1이 아니라 v2(직전)
            assert await _version_count(sm) == 1  # 여러 세대가 쌓이지 않는다

    asyncio.run(body())


def test_snapshotting_is_per_video_id():
    """다른 video_id의 교체는 서로의 스냅샷을 건드리지 않는다."""

    async def body():
        async with _env() as sm:
            other = "OTHERVIDEO1"
            await _create(sm, video_id=VIDEO, lyrics_hash="h1", segments=SEGMENTS_V1)
            await _create(sm, video_id=VIDEO, lyrics_hash="h2", segments=SEGMENTS_V2)
            await _create(sm, video_id=other, lyrics_hash="h1", segments=SEGMENTS_V1)

            assert await _get_version(sm, VIDEO) is not None
            assert await _get_version(sm, other) is None  # other는 아직 최초 생성뿐
            assert await _version_count(sm) == 1

    asyncio.run(body())


# ── ③ 내용이 완전히 같은 재저장은 스냅샷하지 않는다 ─────────────────


def test_identical_timestamps_resave_does_not_snapshot():
    """캐시 재사용의 교차 영상 복사 등으로 같은 내용을 그대로 다시 저장하는 경로가 있다
    (worker._complete_from_cache_db) — 아무것도 안 바뀐 "교체"는 고스트 비교에 무의미한
    diff만 만드므로 스냅샷을 남기지 않는다."""

    async def body():
        async with _env() as sm:
            await _create(sm, lyrics_hash="h1", segments=SEGMENTS_V1)
            # 같은 세그먼트 내용으로 재저장 (lyrics_hash만 다를 수 있음 — 스냅샷 판단은
            # timestamps 내용 기준이다)
            await _create(sm, lyrics_hash="h1-again", segments=list(SEGMENTS_V1))

            assert await _get_version(sm) is None
            assert await _version_count(sm) == 0

    asyncio.run(body())


def test_identical_then_a_real_change_still_snapshots_the_last_distinct_generation():
    async def body():
        async with _env() as sm:
            await _create(sm, lyrics_hash="h1", segments=SEGMENTS_V1)
            await _create(sm, lyrics_hash="h1-again", segments=list(SEGMENTS_V1))  # 무변화, 스킵
            await _create(sm, lyrics_hash="h2", segments=SEGMENTS_V2)  # 실제 변화

            snap = await _get_version(sm)
            assert snap.timestamps["segments"] == SEGMENTS_V1
            assert await _version_count(sm) == 1

    asyncio.run(body())


# ── ④ GET /api/sync/{video_id}/previous 응답 모양 ──────────────────


def test_previous_endpoint_returns_found_false_without_a_snapshot():
    async def body():
        async with _env():
            resp = await get_previous_sync_version(VIDEO)
            assert resp.found is False
            assert resp.timestamps is None
            assert resp.created_at is None
            assert resp.replaced_at is None

    asyncio.run(body())


def test_previous_endpoint_returns_found_false_after_only_the_first_create():
    """최초 생성만 있고 아직 재처리(=두 번째 저장)가 없었으면 조회는 found=false다."""

    async def body():
        async with _env() as sm:
            await _create(sm, lyrics_hash="h1", segments=SEGMENTS_V1)

            resp = await get_previous_sync_version(VIDEO)
            assert resp.found is False

    asyncio.run(body())


def test_previous_endpoint_shape_after_a_replacement():
    async def body():
        async with _env() as sm:
            await _create(sm, lyrics_hash="h1", segments=SEGMENTS_V1, quality_score=0.42)
            await _create(sm, lyrics_hash="h2", segments=SEGMENTS_V2, quality_score=0.77)

            resp = await get_previous_sync_version(VIDEO)
            assert resp.found is True
            assert resp.timestamps == SEGMENTS_V1  # 직전(v1) 세대의 세그먼트
            assert resp.quality_score == 0.42
            assert resp.created_at is not None  # 원래 생성 시각(ISO 문자열)
            assert resp.replaced_at is not None  # 교체(스냅샷)된 시각(ISO 문자열)
            # 두 시각은 서로 다른 의미의 필드다(원래 생성 vs 교체) — 형식만 검증
            datetime_fields = (resp.created_at, resp.replaced_at)
            assert all(isinstance(v, str) and "T" in v for v in datetime_fields)

    asyncio.run(body())
