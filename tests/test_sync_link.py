"""싱크 링크 API 테스트 (inst/커버가 다른 영상 전사를 오프셋과 함께 재사용).

실 DB를 건드리지 않도록 격리된 in-memory SQLite로 connection.async_session을 몽키패치하고
라우트 핸들러(코루틴)를 직접 호출한다(httpx 미설치 환경 회피). pytest-asyncio asyncio_mode
미설정이라 각 테스트는 asyncio.run으로 감싼다. 라우트 섀도잉은 app.routes 순서로 검증.
"""
import asyncio
import contextlib

import pytest
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from everyric2.server.api.sync import (
    SyncLinkRequest,
    create_sync_link,
    delete_sync_link,
    get_sync,
    list_available_syncs,
)
from everyric2.server.db import connection as db_conn
from everyric2.server.db.models import Base
from everyric2.server.db.repository import SyncRepository
from everyric2.server.main import app

SOURCE_SEGMENTS = [
    {
        "text": "테스트 라인",
        "start": 1.0,
        "end": 2.0,
        "words": [{"word": "테", "start": 1.0, "end": 1.5, "confidence": 0.5}],
        "notes": [{"midi": 60, "start": 1.0, "end": 1.4}],
        "pron_segments": [{"text": "테", "start": 1.0, "end": 1.5, "confidence": 0.5}],
        "debug": {"active_ratio": 0.9, "clamped": False, "orig": [1.0, 2.0]},
    }
]
SOURCE_EXTRA = {
    "debug": {
        "vad_regions": [[1.0, 2.0]],
        "star_spans": [[0.0, 0.5]],
        "f0_curve": {"t0": 0.5, "dt": 0.1, "midi": [60]},
        "alignment_text": "original",
    },
    "tempo": {"bpm": 120.0, "beat_offset": 0.3},
    "attribution": {"name": "보카로 위키"},
}


@contextlib.asynccontextmanager
async def _env(seed_source=True, seed_dst_own=False):
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    test_sessionmaker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    orig = db_conn.async_session
    db_conn.async_session = test_sessionmaker  # get_session이 호출 시점에 참조하는 모듈 전역
    try:
        async with test_sessionmaker() as s:
            if seed_source:
                await SyncRepository(s).create(
                    video_id="SRC", lyrics_hash="h1", timestamps=SOURCE_SEGMENTS,
                    engine="ctc", audio_hash="a1", extra=SOURCE_EXTRA,
                )
            if seed_dst_own:
                await SyncRepository(s).create(
                    video_id="DST", lyrics_hash="h2",
                    timestamps=[{"text": "자기 싱크", "start": 5.0, "end": 6.0}],
                    engine="ctc", audio_hash="a2",
                )
            await s.commit()
        yield
    finally:
        db_conn.async_session = orig
        await engine.dispose()


def test_link_resolves_with_positive_offset():
    async def body():
        async with _env():
            link = await create_sync_link(
                SyncLinkRequest(video_id="DST", source_video_id="SRC", offset_sec=10.0)
            )
            assert link.source_video_id == "SRC"

            resp = await get_sync("DST")
            assert resp.found is True
            assert resp.linked == {"source_video_id": "SRC", "offset_sec": 10.0}
            seg = resp.timestamps[0]
            assert seg["start"] == 11.0 and seg["end"] == 12.0
            assert seg["words"][0]["start"] == 11.0
            assert seg["words"][0]["confidence"] == 0.5  # conf는 시프트 안 됨
            assert seg["notes"][0]["start"] == 11.0
            assert seg["pron_segments"][0]["start"] == 11.0
            assert seg["debug"]["orig"] == [11.0, 12.0]
            assert resp.debug["vad_regions"] == [[11.0, 12.0]]
            assert resp.debug["star_spans"] == [[10.0, 10.5]]
            assert resp.debug["f0_curve"]["t0"] == 10.5
            assert resp.debug["alignment_text"] == "original"  # 비시간 필드 보존
            assert resp.tempo["beat_offset"] == 10.3
            assert resp.tempo["bpm"] == 120.0  # bpm은 시프트 안 됨
            assert resp.attribution["name"] == "보카로 위키"

    asyncio.run(body())


def test_link_resolves_with_negative_offset():
    async def body():
        async with _env():
            await create_sync_link(
                SyncLinkRequest(video_id="DST", source_video_id="SRC", offset_sec=-0.5)
            )
            resp = await get_sync("DST")
            assert resp.timestamps[0]["start"] == 0.5
            assert resp.timestamps[0]["end"] == 1.5
            assert resp.debug["f0_curve"]["t0"] == 0.0

    asyncio.run(body())


def test_own_sync_takes_priority_over_link():
    async def body():
        async with _env(seed_dst_own=True):
            # DST는 자기 싱크가 있는데 링크도 걸어 둔다 → 조회는 자기 싱크 우선
            await create_sync_link(
                SyncLinkRequest(video_id="DST", source_video_id="SRC", offset_sec=10.0)
            )
            resp = await get_sync("DST")
            assert resp.found is True
            assert resp.linked is None  # 링크 아님
            assert resp.timestamps[0]["text"] == "자기 싱크"
            assert resp.timestamps[0]["start"] == 5.0  # 시프트 없음

    asyncio.run(body())


def test_self_link_rejected():
    async def body():
        async with _env():
            with pytest.raises(HTTPException) as exc:
                await create_sync_link(
                    SyncLinkRequest(video_id="SRC", source_video_id="SRC", offset_sec=0.0)
                )
            assert exc.value.status_code == 400

    asyncio.run(body())


def test_link_to_source_without_sync_rejected():
    async def body():
        async with _env():
            with pytest.raises(HTTPException) as exc:
                await create_sync_link(
                    SyncLinkRequest(video_id="DST", source_video_id="NOPE", offset_sec=0.0)
                )
            assert exc.value.status_code == 400

    asyncio.run(body())


def test_unlink_removes_link():
    async def body():
        async with _env():
            await create_sync_link(
                SyncLinkRequest(video_id="DST", source_video_id="SRC", offset_sec=3.0)
            )
            assert (await get_sync("DST")).found is True

            removed = await delete_sync_link("DST")
            assert removed["removed"] is True
            # 링크 해제 후 자기 싱크도 없으니 미발견
            assert (await get_sync("DST")).found is False
            # 두 번째 해제는 removed False
            assert (await delete_sync_link("DST"))["removed"] is False

    asyncio.run(body())


def test_relink_is_upsert():
    async def body():
        async with _env():
            await create_sync_link(
                SyncLinkRequest(video_id="DST", source_video_id="SRC", offset_sec=5.0)
            )
            await create_sync_link(
                SyncLinkRequest(video_id="DST", source_video_id="SRC", offset_sec=7.0)
            )
            resp = await get_sync("DST")
            assert resp.linked["offset_sec"] == 7.0
            assert resp.timestamps[0]["start"] == 8.0  # 1.0 + 7.0

    asyncio.run(body())


def test_list_returns_candidates():
    async def body():
        async with _env():
            syncs = await list_available_syncs(limit=10)  # bare 배열 (확장 클라이언트 계약)
            assert isinstance(syncs, list)
            assert len(syncs) == 1
            item = syncs[0]
            assert item["video_id"] == "SRC"
            assert item["first_line"] == "테스트 라인"
            assert item["line_count"] == 1
            assert item["attribution_name"] == "보카로 위키"
            assert item["alignment_text"] == "original"

    asyncio.run(body())


def test_list_route_declared_before_video_id_route():
    # GET /api/sync/list 가 GET /api/sync/{video_id} 보다 먼저 선언돼야 섀도잉되지 않는다
    paths = [r.path for r in app.routes if getattr(r, "path", "").startswith("/api/sync")]
    assert "/api/sync/list" in paths and "/api/sync/{video_id}" in paths
    assert paths.index("/api/sync/list") < paths.index("/api/sync/{video_id}")
