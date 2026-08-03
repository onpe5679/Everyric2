"""신구/깊이별 비교용 이력 API 테스트 — GET /api/sync/{video_id}/versions,
GET /api/sync/{video_id}/versions/{result_id}.

이 둘은 sync_result_versions(직전 1건 스냅샷, test_sync_versions.py)과 다른 재료를 쓴다 —
sync_results 자체가 재생성마다 INSERT-only로 쌓이는 이력이라(SyncResult.__doc__), 여기서는
그 행들을 그대로 최대 10건 나열/조회한다. 디버그 패널이 fast/medium/heavy 세대를 나란히
놓고 고스트 비교하는 재료다.

기존 서버 테스트 규약(test_sync_versions.py와 동일): 격리된 in-memory SQLite로
connection.async_session을 몽키패치하고 라우트 코루틴을 직접 await한다(httpx 불사용).
"""

import asyncio
import contextlib
from datetime import datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from everyric2.server.api.sync import get_sync_version_detail, list_sync_versions
from everyric2.server.db import connection as db_conn
from everyric2.server.db.models import Base
from everyric2.server.db.repository import SyncRepository

VIDEO = "VERLISTVID1"
OTHER = "OTHERVIDEO9"

# SyncResult.created_at은 SQLite server_default=func.now()라 초 단위로 저장된다 — 연속
# 생성 호출이 같은 초 안에 몰리면 "최신순" 판정이 생성 순서를 못 잡는다(test_sync_versions.py
# 의 같은 제약·같은 해법). 순서 검증이 핵심인 테스트에서만 명시적으로 시각을 벌린다.
_BASE_TIME = datetime(2026, 8, 4, 12, 0, 0)


async def _force_creation_order(sm, rows: list) -> None:
    async with sm() as s:
        for i, row in enumerate(rows):
            fresh = await SyncRepository(s).get_by_id(row.id)
            fresh.created_at = _BASE_TIME + timedelta(minutes=i)
        await s.commit()


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


async def _create(sm, *, video_id=VIDEO, lyrics_hash, segments, extra=None, **kwargs):
    async with sm() as s:
        row = await SyncRepository(s).create(
            video_id=video_id, lyrics_hash=lyrics_hash, timestamps=segments, extra=extra, **kwargs
        )
        await s.commit()
        return row


def test_versions_list_is_empty_for_unknown_video():
    async def body():
        async with _env():
            resp = await list_sync_versions(VIDEO)
            assert resp.versions == []

    asyncio.run(body())


def test_versions_list_newest_first_with_depth_extracted_from_debug_routing():
    async def body():
        async with _env() as sm:
            v1 = await _create(
                sm, lyrics_hash="h1", segments=[{"text": "a"}],
                extra={"debug": {"routing": {"route": "fast"}}},
            )
            v2 = await _create(
                sm, lyrics_hash="h2", segments=[{"text": "b"}],
                extra={"debug": {"routing": {"route": "heavy"}}},
                engine_version="stack-2",
            )
            await _force_creation_order(sm, [v1, v2])

            resp = await list_sync_versions(VIDEO)
            assert len(resp.versions) == 2
            # 최신순 — 방금 만든 heavy가 먼저
            assert resp.versions[0].depth == "heavy"
            assert resp.versions[0].engine_version == "stack-2"
            assert resp.versions[1].depth == "fast"

    asyncio.run(body())


def test_versions_list_depth_is_none_for_legacy_rows_without_routing():
    async def body():
        async with _env() as sm:
            await _create(sm, lyrics_hash="h1", segments=[{"text": "a"}])  # extra 없음(구세대)

            resp = await list_sync_versions(VIDEO)
            assert resp.versions[0].depth is None

    asyncio.run(body())


def test_versions_list_is_capped_at_ten_and_scoped_per_video():
    async def body():
        async with _env() as sm:
            for i in range(12):
                await _create(sm, lyrics_hash=f"h{i}", segments=[{"text": str(i)}])
            await _create(sm, video_id=OTHER, lyrics_hash="oh1", segments=[{"text": "other"}])

            resp = await list_sync_versions(VIDEO)
            assert len(resp.versions) == 10  # 최대 10건

            other_resp = await list_sync_versions(OTHER)
            assert len(other_resp.versions) == 1  # 서로 섞이지 않는다

    asyncio.run(body())


def test_version_detail_returns_full_timestamps_for_owned_result():
    async def body():
        async with _env() as sm:
            row = await _create(
                sm, lyrics_hash="h1",
                segments=[{"text": "첫 줄", "start": 0.0, "end": 1.0}],
                extra={"debug": {"routing": {"route": "medium"}}},
                quality_score=0.42,
            )

            detail = await get_sync_version_detail(VIDEO, row.id)
            assert detail.id == row.id
            assert detail.timestamps == [{"text": "첫 줄", "start": 0.0, "end": 1.0}]
            assert detail.quality_score == 0.42
            assert detail.depth == "medium"

    asyncio.run(body())


def test_version_detail_404s_when_video_id_does_not_own_the_result():
    """다른 영상의 result_id로 조회하면 404 — 남의 싱크 본문을 엿보는 것을 막는다."""

    async def body():
        async with _env() as sm:
            row = await _create(sm, lyrics_hash="h1", segments=[{"text": "a"}])

            try:
                await get_sync_version_detail(OTHER, row.id)
                assert False, "should have raised 404"
            except Exception as e:
                assert getattr(e, "status_code", None) == 404

    asyncio.run(body())


def test_version_detail_404s_for_unknown_result_id():
    async def body():
        async with _env():
            try:
                await get_sync_version_detail(VIDEO, "no-such-id")
                assert False, "should have raised 404"
            except Exception as e:
                assert getattr(e, "status_code", None) == 404

    asyncio.run(body())
