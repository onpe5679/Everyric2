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
    reset_video_syncs,
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
                    video_id="SRCSRCSRC01", lyrics_hash="h1", timestamps=SOURCE_SEGMENTS,
                    engine="ctc", audio_hash="a1", extra=SOURCE_EXTRA,
                )
            if seed_dst_own:
                await SyncRepository(s).create(
                    video_id="DSTDSTDST01", lyrics_hash="h2",
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
                SyncLinkRequest(video_id="DSTDSTDST01", source_video_id="SRCSRCSRC01", offset_sec=10.0)
            )
            assert link.source_video_id == "SRCSRCSRC01"

            resp = await get_sync("DSTDSTDST01")
            assert resp.found is True
            assert resp.linked == {"source_video_id": "SRCSRCSRC01", "offset_sec": 10.0, "rate": 1.0}
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
                SyncLinkRequest(video_id="DSTDSTDST01", source_video_id="SRCSRCSRC01", offset_sec=-0.5)
            )
            resp = await get_sync("DSTDSTDST01")
            assert resp.timestamps[0]["start"] == 0.5
            assert resp.timestamps[0]["end"] == 1.5
            assert resp.debug["f0_curve"]["t0"] == 0.0

    asyncio.run(body())


def test_own_sync_takes_priority_over_link():
    async def body():
        async with _env(seed_dst_own=True):
            # DST는 자기 싱크가 있는데 링크도 걸어 둔다 → 조회는 자기 싱크 우선
            await create_sync_link(
                SyncLinkRequest(video_id="DSTDSTDST01", source_video_id="SRCSRCSRC01", offset_sec=10.0)
            )
            resp = await get_sync("DSTDSTDST01")
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
                    SyncLinkRequest(video_id="SRCSRCSRC01", source_video_id="SRCSRCSRC01", offset_sec=0.0)
                )
            assert exc.value.status_code == 400

    asyncio.run(body())


def test_link_to_source_without_sync_rejected():
    async def body():
        async with _env():
            with pytest.raises(HTTPException) as exc:
                await create_sync_link(
                    SyncLinkRequest(video_id="DSTDSTDST01", source_video_id="NOPENOPE001", offset_sec=0.0)
                )
            assert exc.value.status_code == 400

    asyncio.run(body())


def test_unlink_removes_link():
    async def body():
        async with _env():
            await create_sync_link(
                SyncLinkRequest(video_id="DSTDSTDST01", source_video_id="SRCSRCSRC01", offset_sec=3.0)
            )
            assert (await get_sync("DSTDSTDST01")).found is True

            removed = await delete_sync_link("DSTDSTDST01")
            assert removed["removed"] is True
            # 링크 해제 후 자기 싱크도 없으니 미발견
            assert (await get_sync("DSTDSTDST01")).found is False
            # 두 번째 해제는 removed False
            assert (await delete_sync_link("DSTDSTDST01"))["removed"] is False

    asyncio.run(body())


def test_relink_is_upsert():
    async def body():
        async with _env():
            await create_sync_link(
                SyncLinkRequest(video_id="DSTDSTDST01", source_video_id="SRCSRCSRC01", offset_sec=5.0)
            )
            await create_sync_link(
                SyncLinkRequest(video_id="DSTDSTDST01", source_video_id="SRCSRCSRC01", offset_sec=7.0)
            )
            resp = await get_sync("DSTDSTDST01")
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
            assert item["video_id"] == "SRCSRCSRC01"
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


def test_reset_deletes_syncs_and_involving_links():
    async def body():
        async with _env(seed_dst_own=True):
            # DST가 SRC를 빌려 쓰는 링크가 있는 상태에서 SRC를 초기화하면
            # SRC의 싱크와 SRC가 소스인 링크가 함께 사라져야 한다
            await create_sync_link(
                SyncLinkRequest(video_id="DSTDSTDST01", source_video_id="SRCSRCSRC01", offset_sec=1.0)
            )
            res = await reset_video_syncs("SRCSRCSRC01")
            assert res["removed_syncs"] == 1
            assert res["removed_links"] == 1

            resp = await get_sync("SRCSRCSRC01")
            assert resp.found is False
            # DST의 자기 싱크는 영향받지 않고, 죽은 링크로 빌려 오지도 않는다
            resp2 = await get_sync("DSTDSTDST01")
            assert resp2.found is True
            assert resp2.linked is None

    asyncio.run(body())


def test_reset_on_video_without_sync_is_noop():
    async def body():
        async with _env(seed_source=False):
            res = await reset_video_syncs("NOPENOPE001")
            assert res["removed_syncs"] == 0
            assert res["removed_links"] == 0

    asyncio.run(body())


def test_reset_route_does_not_shadow_link_delete():
    # DELETE /api/sync/link/{video_id} 가 DELETE /api/sync/{video_id} 보다 먼저 선언돼야
    # 링크 해제가 싱크 초기화로 오인되지 않는다
    delete_paths = [
        r.path
        for r in app.routes
        if getattr(r, "path", "").startswith("/api/sync") and "DELETE" in getattr(r, "methods", set())
    ]
    assert "/api/sync/link/{video_id}" in delete_paths and "/api/sync/{video_id}" in delete_paths
    assert delete_paths.index("/api/sync/link/{video_id}") < delete_paths.index("/api/sync/{video_id}")


def test_generate_joins_active_job():
    # 버튼 연타 방어: 같은 영상·같은 가사 생성 요청은 진행 중 잡에 합류해야 한다 —
    # 중복 잡 2개가 동시에 같은 임시 오디오 파일을 다운로드하면 WinError 32로 깨진다
    from fastapi import BackgroundTasks

    from everyric2.server.api.sync import GenerateRequest, generate_sync

    async def body():
        async with _env(seed_source=False):
            req = GenerateRequest(video_id="VIDVIDVID01", lyrics="가사 한 줄\n두 줄")
            r1 = await generate_sync(req, BackgroundTasks())
            assert r1.status == "processing"

            r2 = await generate_sync(req, BackgroundTasks())
            assert r2.job_id == r1.job_id  # 새 잡을 만들지 않고 합류
            assert r2.status == "processing"

            # 다른 가사는 별도 잡 (붙여넣기 내용을 고친 재시도는 막지 않는다)
            r3 = await generate_sync(
                GenerateRequest(video_id="VIDVIDVID01", lyrics="완전히 다른 가사"), BackgroundTasks()
            )
            assert r3.job_id != r1.job_id

    asyncio.run(body())


def test_regenerate_joins_active_job():
    from fastapi import BackgroundTasks

    from everyric2.server.api.sync import (
        GenerateRequest,
        RegenerateRequest,
        generate_sync,
        regenerate_sync,
    )

    async def body():
        async with _env(seed_source=False):
            r1 = await generate_sync(
                GenerateRequest(video_id="VIDVIDVID01", lyrics="가사"), BackgroundTasks()
            )
            r2 = await regenerate_sync(
                RegenerateRequest(video_id="VIDVIDVID01", lyrics="가사", force=True), BackgroundTasks()
            )
            assert r2.job_id == r1.job_id  # 진행 중이면 재생성도 합류

    asyncio.run(body())


def test_user_offset_roundtrip():
    from everyric2.server.api.sync import UserOffsetRequest, save_user_offset

    async def body():
        async with _env():
            await save_user_offset("SRCSRCSRC01", UserOffsetRequest(offset_sec=1.5))
            resp = await get_sync("SRCSRCSRC01")
            assert resp.found is True
            assert resp.user_offset == 1.5
            # 싱크 없는 영상도 오프셋은 저장·조회된다 (found=false여도 내려감)
            await save_user_offset("NOSYNCNOS01", UserOffsetRequest(offset_sec=-0.3))
            resp2 = await get_sync("NOSYNCNOS01")
            assert resp2.found is False
            assert resp2.user_offset == -0.3

    asyncio.run(body())


def test_destructive_daily_limit_with_admin_bypass():
    from everyric2.config.settings import get_settings

    async def body():
        async with _env():
            server = get_settings().server
            orig_key, orig_limit = server.admin_api_key, server.daily_destructive_limit
            object.__setattr__(server, "admin_api_key", "admin-secret")
            object.__setattr__(server, "daily_destructive_limit", 1)
            try:
                # 비어드민: 1회 허용, 2회째 429
                await reset_video_syncs("SRCSRCSRC01", x_api_key=None)
                with pytest.raises(HTTPException) as exc:
                    await reset_video_syncs("SRCSRCSRC01", x_api_key="wrong")
                assert exc.value.status_code == 429
                # 어드민 키는 한도 없이 통과
                await reset_video_syncs("SRCSRCSRC01", x_api_key="admin-secret")
            finally:
                object.__setattr__(server, "admin_api_key", orig_key)
                object.__setattr__(server, "daily_destructive_limit", orig_limit)

    asyncio.run(body())
