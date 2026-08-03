"""진행 중 깊이 배지 + ETA(GET /api/job/{id} additive 필드) 테스트.

기존 서버 테스트 규약(test_sync_versions.py와 동일): 격리된 in-memory SQLite로
connection.async_session을 몽키패치하고 라우트 코루틴을 직접 await한다(httpx 불사용).

여기서 못박는 계약:
  ① depth는 processing 잡만, worker._JOB_DEPTH(인프로세스, DB 아님)에서 읽는다.
  ② eta_sec = 같은 depth의 최근 20건 median - 경과시간(바닥 5초). depth 데이터가 없으면
     all-depth 폴백, 그마저 없으면 None.
  ③ queue_eta_sec = queue_position × all-depth median. median 데이터가 없으면 None.
  ④ worker.stash_processing_start/peek_processing_elapsed/pop_processing_duration —
     jobs.updated_at이 아니라 인프로세스 monotonic 스탬프로 duration을 잰다(1회성 pop).
  ⑤ JobMetricRepository.recent_durations는 최근 limit건만, depth 필터가 있으면 그 깊이만.
"""

import asyncio
import contextlib
import time

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from everyric2.server import worker as worker_core
from everyric2.server.api import job as job_api
from everyric2.server.api.job import get_job_status
from everyric2.server.db import connection as db_conn
from everyric2.server.db.models import Base
from everyric2.server.db.repository import JobMetricRepository, JobRepository

VIDEO = "ETAVIDEO001"


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
    # ETA median 캐시(30s TTL)는 프로세스 전역이라 테스트 사이에 새지 않게 매번 비운다
    job_api._ETA_MEDIAN_CACHE.clear()
    worker_core._JOB_DEPTH.clear()
    worker_core._JOB_PROCESSING_START.clear()
    try:
        yield sm
    finally:
        db_conn.async_session = orig
        job_api._ETA_MEDIAN_CACHE.clear()
        worker_core._JOB_DEPTH.clear()
        worker_core._JOB_PROCESSING_START.clear()
        await engine.dispose()


async def _make_job(sm, status: str = "processing", stage: str | None = None) -> str:
    async with sm() as s:
        job = await JobRepository(s).create(video_id=VIDEO, lyrics="가사")
        await JobRepository(s).update_status(job.id, status, stage=stage)
        await s.commit()
        return job.id


async def _record_metrics(sm, depth: str | None, durations: list[float]) -> None:
    async with sm() as s:
        repo = JobMetricRepository(s)
        for d in durations:
            await repo.record(job_id="seed", video_id=VIDEO, depth=depth, duration_sec=d)
        await s.commit()


# ── ① depth는 processing + 인프로세스 _JOB_DEPTH에서만 ──────────────


def test_depth_is_none_when_not_processing():
    async def body():
        async with _env() as sm:
            job_id = await _make_job(sm, status="queued")
            resp = await get_job_status(job_id)
            assert resp.depth is None

    asyncio.run(body())


def test_depth_reflects_in_process_registry_while_processing():
    async def body():
        async with _env() as sm:
            job_id = await _make_job(sm, status="processing", stage="전사 정렬")
            worker_core._JOB_DEPTH[job_id] = "heavy"

            resp = await get_job_status(job_id)
            assert resp.depth == "heavy"

    asyncio.run(body())


# ── ② eta_sec ────────────────────────────────────────────────────


def test_eta_sec_is_none_without_any_metric_data():
    async def body():
        async with _env() as sm:
            job_id = await _make_job(sm, status="processing")
            resp = await get_job_status(job_id)
            assert resp.eta_sec is None

    asyncio.run(body())


def test_eta_sec_uses_same_depth_median_minus_elapsed():
    async def body():
        async with _env() as sm:
            job_id = await _make_job(sm, status="processing")
            worker_core._JOB_DEPTH[job_id] = "medium"
            await _record_metrics(sm, "medium", [100.0, 100.0, 100.0])
            # 처리 시작을 10초 전으로 스탬프 — median(100) - elapsed(~10) ≈ 90
            worker_core._JOB_PROCESSING_START[job_id] = time.monotonic() - 10.0

            resp = await get_job_status(job_id)
            assert resp.eta_sec is not None
            assert 85 <= resp.eta_sec <= 95

    asyncio.run(body())


def test_eta_sec_falls_back_to_all_depth_median_when_depth_unknown():
    async def body():
        async with _env() as sm:
            job_id = await _make_job(sm, status="processing")
            # depth 자체가 아직 안 정해짐(fast 라우팅 진입 전 등) — _JOB_DEPTH에 없음
            await _record_metrics(sm, "heavy", [40.0])
            await _record_metrics(sm, "fast", [20.0])
            worker_core._JOB_PROCESSING_START[job_id] = time.monotonic()

            resp = await get_job_status(job_id)
            assert resp.eta_sec is not None  # all-depth median(30)에서 나온 값

    asyncio.run(body())


def test_eta_sec_is_floored_at_five():
    async def body():
        async with _env() as sm:
            job_id = await _make_job(sm, status="processing")
            worker_core._JOB_DEPTH[job_id] = "fast"
            await _record_metrics(sm, "fast", [10.0])
            # 이미 median을 훨씬 넘겨 처리 중 — median - elapsed가 크게 음수
            worker_core._JOB_PROCESSING_START[job_id] = time.monotonic() - 500.0

            resp = await get_job_status(job_id)
            assert resp.eta_sec == 5

    asyncio.run(body())


# ── ③ queue_eta_sec ─────────────────────────────────────────────


def test_queue_eta_sec_is_position_times_all_depth_median():
    async def body():
        async with _env() as sm:
            await _record_metrics(sm, "fast", [10.0])
            await _record_metrics(sm, "heavy", [30.0])  # all-depth median = 20.0

            # queue_position을 2로 만들기 위해 먼저 대기 중인 잡을 하나 더 심는다
            other_id = await _make_job(sm, status="queued")
            job_id = await _make_job(sm, status="queued")

            resp = await get_job_status(job_id)
            assert resp.queue_position == 2
            assert resp.queue_eta_sec == 40  # 2 * 20.0

    asyncio.run(body())


def test_queue_eta_sec_is_none_without_metric_data():
    async def body():
        async with _env() as sm:
            job_id = await _make_job(sm, status="queued")
            resp = await get_job_status(job_id)
            assert resp.queue_eta_sec is None

    asyncio.run(body())


# ── ④ worker 처리 시작 스탬프 ────────────────────────────────────


def test_processing_start_stamp_is_in_process_not_db_derived():
    job_id = "stamp-test-job"
    worker_core.stash_processing_start(job_id)
    elapsed = worker_core.peek_processing_elapsed(job_id)
    assert elapsed is not None
    assert elapsed >= 0.0

    # peek는 비파괴 — 여러 번 불러도 스탬프가 살아 있다
    assert worker_core.peek_processing_elapsed(job_id) is not None

    duration = worker_core.pop_processing_duration(job_id)
    assert duration is not None
    # pop은 1회성 — 이후 조회는 스탬프가 없다(None)
    assert worker_core.peek_processing_elapsed(job_id) is None
    assert worker_core.pop_processing_duration(job_id) is None


def test_peek_elapsed_without_a_stamp_is_none():
    assert worker_core.peek_processing_elapsed("never-stashed-job") is None


# ── ⑤ JobMetricRepository ───────────────────────────────────────


def test_recent_durations_filters_by_depth_and_respects_limit():
    async def body():
        async with _env() as sm:
            await _record_metrics(sm, "fast", [1.0, 2.0])
            await _record_metrics(sm, "heavy", [9.0])

            async with sm() as s:
                repo = JobMetricRepository(s)
                fast_only = await repo.recent_durations("fast", limit=20)
                assert sorted(fast_only) == [1.0, 2.0]

                all_depths = await repo.recent_durations(None, limit=20)
                assert sorted(all_depths) == [1.0, 2.0, 9.0]

                capped = await repo.recent_durations(None, limit=2)
                assert len(capped) == 2

    asyncio.run(body())
