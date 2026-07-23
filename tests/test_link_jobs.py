"""링크 검증 잡(WS1-E) + 워커 오디오 전달 인가(WS1-D) 테스트.

기존 워커 풀 테스트 규약을 그대로 따른다: 격리된 in-memory SQLite로 connection.async_session을
몽키패치하고 라우트 코루틴을 직접 await(asyncio.run). 리스/스태시/워커오디오 전역은 테스트마다
비운다. httpx/TestClient는 쓰지 않는다.
"""

import asyncio
import contextlib
import os
import tempfile
import time

import pytest
from fastapi import HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from everyric2 import __version__
from everyric2.config.settings import get_settings
from everyric2.server import worker as worker_core
from everyric2.server.api import worker as worker_api
from everyric2.server.api.link_jobs import (
    LinkJobRequest,
    create_link_job,
    get_link_job,
)
from everyric2.server.api.sync import get_sync
from everyric2.server.api.worker import (
    ClaimRequest,
    FailRequest,
    LinkResultRequest,
    claim_job,
    get_job_audio,
    submit_link_fail,
    submit_link_result,
)
from everyric2.server.db import connection as db_conn
from everyric2.server.db.models import Base
from everyric2.server.db.repository import JobRepository, LinkJobRepository, SyncRepository

WKEY = "test-worker-key"
WID = "worker-A"
COVER = "COVERvideo1"  # 11자 video_id
SOURCE = "SOURCEvid01"


@contextlib.asynccontextmanager
async def _env(worker_key: str = WKEY):
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
    prev_key = server.worker_key
    object.__setattr__(server, "worker_key", worker_key)
    worker_api._LEASES.clear()
    worker_api._WORKER_AUDIO.clear()
    worker_core._CANCEL_REQUESTED.clear()
    try:
        yield sm
    finally:
        db_conn.async_session = orig
        object.__setattr__(server, "worker_key", prev_key)
        worker_api._LEASES.clear()
        worker_api._WORKER_AUDIO.clear()
        worker_core._CANCEL_REQUESTED.clear()
        await engine.dispose()


async def _seed_source_sync(sm, video_id=SOURCE):
    async with sm() as s:
        await SyncRepository(s).create(
            video_id=video_id,
            lyrics_hash="h1",
            timestamps=[{"text": "원곡 라인", "start": 1.0, "end": 2.0}],
            engine="ctc",
            audio_hash="a1",
        )
        await s.commit()


async def _seed_queued_sync_job(sm, video_id="VIDVIDVID01", lyrics="가사"):
    async with sm() as s:
        job = await JobRepository(s).create(video_id=video_id, lyrics=lyrics)
        await JobRepository(s).update_status(job.id, "queued", progress=0)
        await s.commit()
        return job.id


# ── link-jobs API: 생성/중복 병합/조회 ────────────────────────────


def test_create_link_job_returns_id():
    async def body():
        async with _env() as sm:
            resp = await create_link_job(LinkJobRequest(video_id=COVER, source_video_id=SOURCE))
            assert resp.id
            async with sm() as s:
                lj = await LinkJobRepository(s).get_by_id(resp.id)
                assert lj.status == "queued"
                assert lj.video_id == COVER and lj.source_video_id == SOURCE

    asyncio.run(body())


def test_create_link_job_dedups_active_pair():
    async def body():
        async with _env():
            r1 = await create_link_job(LinkJobRequest(video_id=COVER, source_video_id=SOURCE))
            r2 = await create_link_job(LinkJobRequest(video_id=COVER, source_video_id=SOURCE))
            assert r2.id == r1.id  # 진행 중 동일 쌍은 새 잡을 만들지 않고 병합
            # 다른 쌍은 별개
            r3 = await create_link_job(LinkJobRequest(video_id=COVER, source_video_id="OTHERvid001"))
            assert r3.id != r1.id

    asyncio.run(body())


def test_create_link_job_rejects_self():
    async def body():
        async with _env():
            with pytest.raises(HTTPException) as e:
                await create_link_job(LinkJobRequest(video_id=COVER, source_video_id=COVER))
            assert e.value.status_code == 400

    asyncio.run(body())


def test_get_link_job_status_roundtrip():
    async def body():
        async with _env() as sm:
            r = await create_link_job(LinkJobRequest(video_id=COVER, source_video_id=SOURCE))
            status = await get_link_job(r.id)
            assert status.status == "queued"
            assert status.match is None and status.offset_sec is None
            # 존재하지 않는 잡은 404
            with pytest.raises(HTTPException) as e:
                await get_link_job("00000000-0000-0000-0000-000000000000")
            assert e.value.status_code == 404
            # 형식 밖 id는 422
            with pytest.raises(HTTPException) as e2:
                await get_link_job("not-a-uuid")
            assert e2.value.status_code == 422
            _ = sm

    asyncio.run(body())


# ── claim: sync 우선, 없으면 link_validate ────────────────────────


def test_claim_returns_link_job_when_no_sync():
    async def body():
        async with _env() as sm:
            r = await create_link_job(LinkJobRequest(video_id=COVER, source_video_id=SOURCE))
            resp = await claim_job(
                ClaimRequest(worker_id=WID, version=__version__), x_worker_key=WKEY
            )
            assert resp.kind == "link_validate"
            assert resp.job is None
            assert resp.link_job is not None
            assert resp.link_job.link_job_id == r.id
            assert resp.link_job.video_id == COVER
            assert resp.link_job.source_video_id == SOURCE
            # 링크 잡은 processing + "link:{id}" 리스
            async with sm() as s:
                lj = await LinkJobRepository(s).get_by_id(r.id)
                assert lj.status == "processing"
            assert worker_api._LEASES[f"link:{r.id}"][0] == WID

    asyncio.run(body())


def test_claim_prefers_sync_over_link():
    async def body():
        async with _env() as sm:
            sync_id = await _seed_queued_sync_job(sm)
            await create_link_job(LinkJobRequest(video_id=COVER, source_video_id=SOURCE))
            resp = await claim_job(
                ClaimRequest(worker_id=WID, version=__version__), x_worker_key=WKEY
            )
            assert resp.kind == "sync"
            assert resp.job is not None
            assert resp.job.job_id == sync_id
            assert resp.link_job is None

    asyncio.run(body())


# ── link result → SyncLink 생성 + 부호 규약 ───────────────────────


def test_link_result_match_creates_synclink_with_correct_sign():
    async def body():
        async with _env() as sm:
            await _seed_source_sync(sm)
            r = await create_link_job(LinkJobRequest(video_id=COVER, source_video_id=SOURCE))
            await claim_job(ClaimRequest(worker_id=WID, version=__version__), x_worker_key=WKEY)

            resp = await submit_link_result(
                r.id,
                LinkResultRequest(match=True, offset_sec=10.0, confidence=0.8),
                x_worker_key=WKEY,
                x_worker_id=WID,
            )
            assert resp.accepted is True
            # 링크 잡 done + 결과 기록
            status = await get_link_job(r.id)
            assert status.status == "done" and status.match is True
            assert status.offset_sec == 10.0 and status.confidence == 0.8
            # 리스 해제
            assert f"link:{r.id}" not in worker_api._LEASES

            # 부호 검증: 커버(COVER)는 자기 싱크가 없으니 SyncLink로 원곡(SOURCE) 싱크를
            # offset +10 시프트해 빌려온다 → 원곡 start 1.0 → 커버에서 11.0
            got = await get_sync(COVER)
            assert got.found is True
            assert got.linked == {"source_video_id": SOURCE, "offset_sec": 10.0, "rate": 1.0}
            assert got.timestamps[0]["start"] == 11.0
            assert got.timestamps[0]["end"] == 12.0

    asyncio.run(body())


def test_link_result_no_match_creates_no_synclink():
    async def body():
        async with _env() as sm:
            await _seed_source_sync(sm)
            r = await create_link_job(LinkJobRequest(video_id=COVER, source_video_id=SOURCE))
            await claim_job(ClaimRequest(worker_id=WID, version=__version__), x_worker_key=WKEY)
            await submit_link_result(
                r.id,
                LinkResultRequest(match=False, offset_sec=3.0, confidence=0.1),
                x_worker_key=WKEY,
                x_worker_id=WID,
            )
            status = await get_link_job(r.id)
            assert status.status == "done" and status.match is False
            # 커버는 링크가 없으니 미발견
            got = await get_sync(COVER)
            assert got.found is False
            _ = sm

    asyncio.run(body())


def test_link_result_rejects_foreign_worker():
    async def body():
        async with _env():
            r = await create_link_job(LinkJobRequest(video_id=COVER, source_video_id=SOURCE))
            await claim_job(ClaimRequest(worker_id=WID, version=__version__), x_worker_key=WKEY)
            with pytest.raises(HTTPException) as e:
                await submit_link_result(
                    r.id,
                    LinkResultRequest(match=True, offset_sec=0.0, confidence=0.9),
                    x_worker_key=WKEY,
                    x_worker_id="worker-B",
                )
            assert e.value.status_code == 409

    asyncio.run(body())


def test_link_fail_marks_failed():
    async def body():
        async with _env():
            r = await create_link_job(LinkJobRequest(video_id=COVER, source_video_id=SOURCE))
            await claim_job(ClaimRequest(worker_id=WID, version=__version__), x_worker_key=WKEY)
            resp = await submit_link_fail(
                r.id, FailRequest(error="원곡 오디오를 못 받았어요"),
                x_worker_key=WKEY, x_worker_id=WID,
            )
            assert resp.accepted is True
            status = await get_link_job(r.id)
            assert status.status == "failed"
            assert status.error == "원곡 오디오를 못 받았어요"

    asyncio.run(body())


# ── WS1-D: 워커 오디오 전달 인가 ──────────────────────────────────


def test_job_audio_requires_worker_key():
    async def body():
        async with _env():
            with pytest.raises(HTTPException) as e:
                await get_job_audio("job-1", x_worker_key="wrong", x_worker_id=WID)
            assert e.value.status_code == 403

    asyncio.run(body())


def test_job_audio_rejects_non_lease_owner_409():
    async def body():
        async with _env():
            # 리스는 WID 소유인데 다른 워커가 오디오를 요청 → 409
            worker_api._LEASES["job-1"] = (WID, time.time() + 100)
            with pytest.raises(HTTPException) as e:
                await get_job_audio("job-1", x_worker_key=WKEY, x_worker_id="worker-B")
            assert e.value.status_code == 409

    asyncio.run(body())


def test_job_audio_404_when_no_prepared_file():
    async def body():
        async with _env():
            worker_api._LEASES["job-1"] = (WID, time.time() + 100)
            with pytest.raises(HTTPException) as e:
                await get_job_audio("job-1", x_worker_key=WKEY, x_worker_id=WID)
            assert e.value.status_code == 404

    asyncio.run(body())


def test_job_audio_served_to_lease_owner():
    async def body():
        async with _env():
            fd, path = tempfile.mkstemp(suffix=".m4a")
            os.write(fd, b"\x00\x00\x00\x00fake-audio")
            os.close(fd)
            try:
                worker_api._LEASES["job-1"] = (WID, time.time() + 100)
                worker_api._WORKER_AUDIO["job-1"] = path
                resp = await get_job_audio("job-1", x_worker_key=WKEY, x_worker_id=WID)
                assert isinstance(resp, FileResponse)
                assert str(resp.path) == path
            finally:
                os.unlink(path)

    asyncio.run(body())
