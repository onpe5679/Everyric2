"""원격 GPU 워커 풀 API 테스트 (claim/progress/cache-check/result/fail + 로컬 워커 토글).

실 DB를 건드리지 않도록 격리된 in-memory SQLite로 connection.async_session을 몽키패치하고
라우트 핸들러(코루틴)를 직접 호출한다(httpx 미설치 환경 회피). 각 테스트는 asyncio.run으로
감싼다. 리스 레지스트리·취소 집합·스태시는 모듈 전역이라 테스트마다 비운다.
"""
import asyncio
import contextlib

import pytest
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from everyric2 import __version__
from everyric2.config.settings import get_settings
from everyric2.server import worker as worker_core
from everyric2.server.api import worker as worker_api
from everyric2.server.api.worker import (
    CacheCheckRequest,
    ClaimRequest,
    FailRequest,
    ProgressRequest,
    ResultRequest,
    cache_check,
    claim_job,
    report_progress,
    submit_fail,
    submit_result,
)
from everyric2.server.db import connection as db_conn
from everyric2.server.db.models import Base
from everyric2.server.db.repository import JobRepository, SyncRepository, hash_lyrics

WKEY = "test-worker-key"
WID = "worker-A"


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
    prev_key, prev_local = server.worker_key, server.local_worker
    object.__setattr__(server, "worker_key", worker_key)
    # 인메모리 전역 초기화 — 테스트 간 오염 방지
    worker_api._LEASES.clear()
    worker_core._CANCEL_REQUESTED.clear()
    worker_core._PENDING_LINE_META.clear()
    worker_core._PENDING_ATTRIBUTION.clear()
    worker_core._PENDING_FORCE.clear()
    try:
        yield sm
    finally:
        db_conn.async_session = orig
        object.__setattr__(server, "worker_key", prev_key)
        object.__setattr__(server, "local_worker", prev_local)
        worker_api._LEASES.clear()
        worker_core._CANCEL_REQUESTED.clear()
        worker_core._PENDING_LINE_META.clear()
        worker_core._PENDING_ATTRIBUTION.clear()
        worker_core._PENDING_FORCE.clear()
        await engine.dispose()


async def _seed_queued_job(sm, video_id="VIDVIDVID01", lyrics="가사 한 줄\n두 줄"):
    async with sm() as s:
        job = await JobRepository(s).create(video_id=video_id, lyrics=lyrics)
        await JobRepository(s).update_status(job.id, "queued", progress=0)
        await s.commit()
        return job.id


# ── claim ─────────────────────────────────────────────────────────


def test_claim_requires_worker_key():
    async def body():
        async with _env():
            # 키 없음
            with pytest.raises(HTTPException) as e:
                await claim_job(ClaimRequest(worker_id=WID, version=__version__), x_worker_key=None)
            assert e.value.status_code == 403
            # 키 불일치
            with pytest.raises(HTTPException) as e2:
                await claim_job(
                    ClaimRequest(worker_id=WID, version=__version__), x_worker_key="wrong"
                )
            assert e2.value.status_code == 403

    asyncio.run(body())


def test_claim_version_mismatch_409():
    # 버전 검사는 큐 조회 전에 돈다 — 시드 없이도 409
    async def body():
        async with _env():
            with pytest.raises(HTTPException) as e:
                await claim_job(ClaimRequest(worker_id=WID, version="0.0.0-nope"), x_worker_key=WKEY)
            assert e.value.status_code == 409

    asyncio.run(body())


def test_claim_returns_job_with_payload():
    async def body():
        async with _env() as sm:
            job_id = await _seed_queued_job(sm, lyrics="라라라\n루루루")
            worker_core.stash_line_meta(job_id, [{"text": "라라라", "pronunciation": "lalala"}])

            resp = await claim_job(ClaimRequest(worker_id=WID, version=__version__), x_worker_key=WKEY)
            assert resp.job is not None
            assert resp.job.job_id == job_id
            assert resp.job.lyrics == "라라라\n루루루"
            assert resp.job.line_meta == [{"text": "라라라", "pronunciation": "lalala"}]
            assert resp.lease_seconds == get_settings().server.worker_lease_sec
            # 잡은 processing으로 마킹되고 리스가 등록된다
            async with sm() as s:
                j = await JobRepository(s).get_by_id(job_id)
                assert j.status == "processing"
            assert worker_api._LEASES[job_id][0] == WID
            # 스태시는 peek — 클레임 후에도 남아 재클레임 시 다시 전달된다
            assert job_id in worker_core._PENDING_LINE_META

    asyncio.run(body())


def test_claim_empty_queue_returns_null():
    async def body():
        async with _env():
            resp = await claim_job(ClaimRequest(worker_id=WID, version=__version__), x_worker_key=WKEY)
            assert resp.job is None

    asyncio.run(body())


def test_expired_lease_swept_and_reclaimable():
    async def body():
        async with _env() as sm:
            job_id = await _seed_queued_job(sm)
            r1 = await claim_job(ClaimRequest(worker_id=WID, version=__version__), x_worker_key=WKEY)
            assert r1.job.job_id == job_id
            # 리스를 과거로 만료시킨다 (워커 하트비트 끊김)
            worker_api._LEASES[job_id] = (WID, 0.0)
            # 다른 워커가 claim → 만료 스윕이 잡을 queued로 되돌리고 재클레임된다
            r2 = await claim_job(
                ClaimRequest(worker_id="worker-B", version=__version__), x_worker_key=WKEY
            )
            assert r2.job is not None
            assert r2.job.job_id == job_id
            assert worker_api._LEASES[job_id][0] == "worker-B"

    asyncio.run(body())


# ── progress ──────────────────────────────────────────────────────


def test_progress_renews_lease():
    async def body():
        async with _env() as sm:
            job_id = await _seed_queued_job(sm)
            await claim_job(ClaimRequest(worker_id=WID, version=__version__), x_worker_key=WKEY)
            worker_api._LEASES[job_id] = (WID, 1.0)  # 곧 만료할 값으로 낮춘다
            resp = await report_progress(
                job_id, ProgressRequest(progress=42, stage="보컬 분리"),
                x_worker_key=WKEY, x_worker_id=WID,
            )
            assert resp.cancel_requested is False
            assert worker_api._LEASES[job_id][1] > 1.0  # 갱신됨
            async with sm() as s:
                j = await JobRepository(s).get_by_id(job_id)
                assert j.progress == 42 and j.stage == "보컬 분리"

    asyncio.run(body())


def test_progress_reports_cancel_and_clears_stash():
    async def body():
        async with _env() as sm:
            job_id = await _seed_queued_job(sm)
            await claim_job(ClaimRequest(worker_id=WID, version=__version__), x_worker_key=WKEY)
            worker_core.stash_line_meta(job_id, [{"text": "x"}])
            # 취소 API를 흉내: 취소 집합 등록 + 잡 failed 마킹
            worker_core.request_cancel(job_id)
            async with sm() as s:
                await JobRepository(s).update_status(job_id, "failed", error="요청으로 취소했어요")
                await s.commit()
            resp = await report_progress(
                job_id, ProgressRequest(progress=50, stage="전사 정렬"),
                x_worker_key=WKEY, x_worker_id=WID,
            )
            assert resp.cancel_requested is True
            assert job_id not in worker_core._PENDING_LINE_META  # 스태시 정리
            # 취소 가드로 failed가 processing으로 되돌려지지 않는다
            async with sm() as s:
                j = await JobRepository(s).get_by_id(job_id)
                assert j.status == "failed"

    asyncio.run(body())


def test_progress_rejects_foreign_worker():
    async def body():
        async with _env() as sm:
            job_id = await _seed_queued_job(sm)
            await claim_job(ClaimRequest(worker_id=WID, version=__version__), x_worker_key=WKEY)
            with pytest.raises(HTTPException) as e:
                await report_progress(
                    job_id, ProgressRequest(progress=50, stage="전사 정렬"),
                    x_worker_key=WKEY, x_worker_id="worker-B",
                )
            assert e.value.status_code == 409

    asyncio.run(body())


# ── cache-check ───────────────────────────────────────────────────


def test_cache_check_completes_from_existing_sync():
    async def body():
        async with _env() as sm:
            lyrics = "라인1\n라인2"
            async with sm() as s:
                await SyncRepository(s).create(
                    video_id="AAAAAAAAAA1",
                    lyrics_hash=hash_lyrics(lyrics),
                    timestamps=[{"text": "라인1", "start": 1.0, "end": 2.0}],
                    audio_hash="hashX",
                )
                await s.commit()
            job_id = await _seed_queued_job(sm, video_id="BBBBBBBBBB1", lyrics=lyrics)
            await claim_job(ClaimRequest(worker_id=WID, version=__version__), x_worker_key=WKEY)

            resp = await cache_check(
                job_id, CacheCheckRequest(audio_hash="hashX"),
                x_worker_key=WKEY, x_worker_id=WID,
            )
            assert resp.completed is True
            # 잡 완료 + 리스 해제
            async with sm() as s:
                j = await JobRepository(s).get_by_id(job_id)
                assert j.status == "completed"
            assert job_id not in worker_api._LEASES

    asyncio.run(body())


def test_cache_check_miss_returns_false():
    async def body():
        async with _env() as sm:
            job_id = await _seed_queued_job(sm)
            await claim_job(ClaimRequest(worker_id=WID, version=__version__), x_worker_key=WKEY)
            resp = await cache_check(
                job_id, CacheCheckRequest(audio_hash="no-such-hash"),
                x_worker_key=WKEY, x_worker_id=WID,
            )
            assert resp.completed is False
            assert job_id in worker_api._LEASES  # 미스면 리스 유지

    asyncio.run(body())


# ── result ────────────────────────────────────────────────────────


def test_result_accepted_creates_sync():
    async def body():
        async with _env() as sm:
            job_id = await _seed_queued_job(sm, video_id="CCCCCCCCCC1")
            await claim_job(ClaimRequest(worker_id=WID, version=__version__), x_worker_key=WKEY)
            resp = await submit_result(
                job_id,
                ResultRequest(
                    timestamps=[{"text": "라인", "start": 0.0, "end": 1.0}],
                    language="ja",
                    quality_score=0.87,
                    audio_hash="hashR",
                    extra={"tempo": {"bpm": 128.0}},
                ),
                x_worker_key=WKEY, x_worker_id=WID,
            )
            assert resp.accepted is True
            async with sm() as s:
                j = await JobRepository(s).get_by_id(job_id)
                assert j.status == "completed" and j.result_id
                rows = await SyncRepository(s).get_by_video("CCCCCCCCCC1")
                assert len(rows) == 1
                assert rows[0].audio_hash == "hashR"
                assert rows[0].timestamps["tempo"] == {"bpm": 128.0}
            assert job_id not in worker_api._LEASES  # 리스 해제

    asyncio.run(body())


def test_result_rejected_for_cancelled_job():
    async def body():
        async with _env() as sm:
            job_id = await _seed_queued_job(sm)
            await claim_job(ClaimRequest(worker_id=WID, version=__version__), x_worker_key=WKEY)
            # 취소된(=failed) 잡의 뒤늦은 결과는 거부
            async with sm() as s:
                await JobRepository(s).update_status(job_id, "failed", error="요청으로 취소했어요")
                await s.commit()
            with pytest.raises(HTTPException) as e:
                await submit_result(
                    job_id,
                    ResultRequest(timestamps=[{"text": "x", "start": 0.0, "end": 1.0}]),
                    x_worker_key=WKEY, x_worker_id=WID,
                )
            assert e.value.status_code == 409

    asyncio.run(body())


def test_result_rejects_foreign_worker():
    async def body():
        async with _env() as sm:
            job_id = await _seed_queued_job(sm)
            await claim_job(ClaimRequest(worker_id=WID, version=__version__), x_worker_key=WKEY)
            with pytest.raises(HTTPException) as e:
                await submit_result(
                    job_id,
                    ResultRequest(timestamps=[{"text": "x", "start": 0.0, "end": 1.0}]),
                    x_worker_key=WKEY, x_worker_id="worker-B",
                )
            assert e.value.status_code == 409

    asyncio.run(body())


# ── fail ──────────────────────────────────────────────────────────


def test_fail_marks_job_failed():
    async def body():
        async with _env() as sm:
            job_id = await _seed_queued_job(sm)
            await claim_job(ClaimRequest(worker_id=WID, version=__version__), x_worker_key=WKEY)
            resp = await submit_fail(
                job_id, FailRequest(error="영상을 못 받아왔어요 (403)"),
                x_worker_key=WKEY, x_worker_id=WID,
            )
            assert resp.accepted is True
            async with sm() as s:
                j = await JobRepository(s).get_by_id(job_id)
                assert j.status == "failed"
                assert j.error == "영상을 못 받아왔어요 (403)"
            assert job_id not in worker_api._LEASES

    asyncio.run(body())


# ── LOCAL_WORKER 토글 ─────────────────────────────────────────────


def test_local_worker_false_queues_without_add_task():
    from fastapi import BackgroundTasks

    from everyric2.server.api.sync import GenerateRequest, generate_sync

    async def body():
        async with _env() as sm:
            object.__setattr__(get_settings().server, "local_worker", False)
            bg = BackgroundTasks()
            resp = await generate_sync(
                GenerateRequest(video_id="VIDVIDVID01", lyrics="가사 A\n가사 B"), bg
            )
            # add_task 없이 큐잉만 — BackgroundTasks에 태스크가 등록되지 않는다
            assert len(bg.tasks) == 0
            async with sm() as s:
                j = await JobRepository(s).get_by_id(resp.job_id)
                assert j.status == "queued"

    asyncio.run(body())


def test_local_worker_true_registers_add_task():
    from fastapi import BackgroundTasks

    from everyric2.server.api.sync import GenerateRequest, generate_sync

    async def body():
        async with _env():
            object.__setattr__(get_settings().server, "local_worker", True)
            bg = BackgroundTasks()
            await generate_sync(
                GenerateRequest(video_id="VIDVIDVID02", lyrics="가사 C\n가사 D"), bg
            )
            # 기존 동작: 인프로세스 처리 태스크가 등록된다
            assert len(bg.tasks) == 1

    asyncio.run(body())
