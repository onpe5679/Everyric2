"""잡 수명주기·한도 회귀 테스트 (전수조사에서 확인된 서버 API 결함 ①~⑦).

기존 서버 테스트 규약을 그대로 따른다: 격리된 in-memory SQLite로 connection.async_session을
몽키패치하고 라우트 코루틴을 직접 await(asyncio.run). httpx/TestClient는 쓰지 않는다.

여기서 못박는 계약:
  ① 만료 리스 스윕은 claim이 없어도 돈다 (워커 하나가 죽어도 잡이 봉인되지 않는다).
  ② 재기동 좀비 정리는 link_jobs도 대상이다 (processing → queued).
  ③ 빈/기호뿐인 가사는 400 — 0줄 싱크가 캐시로 영구화되는 경로를 입구에서 막는다.
  ④ GPU를 태우는 경로(generate·link-candidates·link-jobs)에 상한·쿨다운이 있다.
  ⑤ 취소는 부활하지 않는다 (조건부 상태 쓰기).
  ⑥ 잡 종결과 스태시 쓰기가 겹쳐도 스태시가 새지 않고 응답이 사실과 일치한다.
  ⑦ GET /api/job/{id}는 그 잡이 만든 싱크(result_id)를 돌려준다.
"""

import asyncio
import contextlib
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import BackgroundTasks, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from everyric2 import __version__
from everyric2.config.settings import get_settings
from everyric2.server import worker as worker_core
from everyric2.server.api import sync as sync_api
from everyric2.server.api import worker as worker_api
from everyric2.server.api.job import get_job_status
from everyric2.server.api.link_jobs import LinkJobRequest, create_link_job
from everyric2.server.api.sync import (
    GenerateRequest,
    LineMeta,
    RegenerateRequest,
    _attach_line_meta_to_job,
    _queue_after_line_meta,
    find_link_candidates,
    generate_sync,
    regenerate_sync,
)
from everyric2.server.api.worker import ClaimRequest, claim_job
from everyric2.server.db import connection as db_conn
from everyric2.server.db import orphan_reaper
from everyric2.server.db.models import Base, Job, LinkJob
from everyric2.server.db.repository import (
    JobRepository,
    LinkJobRepository,
    SyncRepository,
    hash_lyrics,
)

WKEY = "test-worker-key"
WID = "worker-A"
VIDEO = "VIDVIDVID01"
COVER = "COVERvideo1"
SOURCE = "SOURCEvid01"
SOURCE_TITLE = "熱異常 / いよわ feat.初音ミク"
COVER_TITLE = "熱異常 歌ってみた【足立レイ】"
# 하한(MIN_LYRICS_LINES)을 넉넉히 넘는 정상 가사
LYRICS = "첫 줄\n두 번째 줄"


@contextlib.asynccontextmanager
async def _env(worker_key: str = WKEY, **server_overrides):
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
    overrides = {"worker_key": worker_key, **server_overrides}
    saved = {k: getattr(server, k) for k in overrides}
    for k, v in overrides.items():
        object.__setattr__(server, k, v)

    def _clear_globals():
        worker_api._LEASES.clear()
        worker_api._WORKER_AUDIO.clear()
        worker_core._CANCEL_REQUESTED.clear()
        worker_core._PENDING_LINE_META.clear()
        worker_core._PENDING_ATTRIBUTION.clear()
        worker_core._PENDING_TITLE.clear()
        worker_core._PENDING_FORCE.clear()
        worker_core._PENDING_META_WAIT.clear()

    _clear_globals()
    try:
        yield sm
    finally:
        db_conn.async_session = orig
        for k, v in saved.items():
            object.__setattr__(server, k, v)
        _clear_globals()
        await engine.dispose()


async def _seed_queued_job(sm, video_id=VIDEO, lyrics=LYRICS) -> str:
    async with sm() as s:
        job = await JobRepository(s).create(video_id=video_id, lyrics=lyrics)
        await JobRepository(s).update_status(job.id, "queued", progress=0)
        await s.commit()
        return job.id


async def _job(sm, job_id) -> Job:
    async with sm() as s:
        return await JobRepository(s).get_by_id(job_id)


async def _count_jobs(sm) -> int:
    async with sm() as s:
        return len((await s.execute(select(Job))).scalars().all())


# ── ① 만료 리스 스윕이 claim 없이도 돈다 ──────────────────────────


@pytest.fixture(autouse=True)
def _cached_pair_everywhere(monkeypatch):
    """기본값: 미디어 캐시 완비로 가정 — link-candidates의 캐시 쌍 게이트(기본 켜짐)가
    이 모듈이 검증하는 예산·한도 기전과 무관하게 제출을 막지 않도록 한다.
    게이트 자체는 test_link_candidates.py가 검증한다."""
    monkeypatch.setattr(sync_api.media_cache, "lookup_cached", lambda _vid: True)
    sync_api._RECENT_LYRICS_HASH.clear()
    yield
    sync_api._RECENT_LYRICS_HASH.clear()


def test_lease_sweeper_requeues_a_dead_workers_job_without_any_claim():
    """워커 하나가 잡을 물고 죽은 뒤 **아무도 claim하지 않아도** 잡이 회수돼야 한다.

    스윕이 claim 안에서만 돌던 동안에는 이 상황에서 잡이 processing에 영구 정착했고,
    get_active_by_video가 그 죽은 잡을 활성으로 봐서 같은 (영상, 가사) 요청이 새 잡을
    만들지 못했다 — 서버 재기동까지 그 가사로 재생성 불가.
    """

    async def body():
        async with _env() as sm:
            job_id = await _seed_queued_job(sm)
            claimed = await claim_job(
                ClaimRequest(worker_id=WID, version=__version__), x_worker_key=WKEY
            )
            assert claimed.job.job_id == job_id
            assert (await _job(sm, job_id)).status == "processing"

            # 워커 사망: 리스를 과거로 만료시키고 이후 claim은 한 번도 하지 않는다
            worker_api._LEASES[job_id] = (WID, 0.0)

            # 스윕 완료를 이벤트로 기다린다 — DB를 폴링하며 기다리면 안 된다:
            # 테스트 DB는 :memory: + StaticPool이라 **연결이 하나**여서, 폴링 루프가 그
            # 연결을 계속 물고 있으면 스윕의 UPDATE가 순서를 못 잡아 예산을 통째로 태운다
            # (실측: 같은 테스트가 1/3 확률로 실패, 스윕 예외 로그는 없음 = 실행이 밀린 것).
            # 프로덕션은 파일 DB(연결 풀 분리 + WAL + busy_timeout)에 간격이 20s라 무관한
            # 테스트 하네스 인공물이지만, 흔들리는 테스트는 그 자체로 결함이라 없앤다.
            swept = asyncio.Event()
            real_sweep = worker_api._sweep_expired_leases

            async def sweep_and_signal():
                await real_sweep()
                swept.set()

            prev_interval = worker_api.LEASE_SWEEP_INTERVAL_SEC
            worker_api._sweep_expired_leases = sweep_and_signal
            worker_api.LEASE_SWEEP_INTERVAL_SEC = 0.01
            worker_api.start_lease_sweeper()
            try:
                # claim을 한 번도 하지 않았는데 주기 태스크만으로 스윕이 발화해야 한다
                await asyncio.wait_for(swept.wait(), timeout=10.0)
            finally:
                await worker_api.stop_lease_sweeper()
                worker_api.LEASE_SWEEP_INTERVAL_SEC = prev_interval
                worker_api._sweep_expired_leases = real_sweep

            assert (await _job(sm, job_id)).status == "queued"
            assert job_id not in worker_api._LEASES

    asyncio.run(body())


def test_generate_reclaims_a_dead_workers_job_instead_of_joining_it():
    """①의 **두 번째 방어선** — 이벤트 루프·주기 태스크에 의존하지 않는다.

    워커가 잡을 물고 죽으면 그 잡은 processing에 남아 get_active_by_video에 활성으로 잡히고,
    이후 같은 (영상, 가사) 생성 요청이 **죽은 잡에 합류**해 새 잡을 만들지 않는다 →
    서버 재기동까지 그 가사로 재생성 불가(그 영상이 봉인된다). 주기 스윕이 아직 돌지
    않았어도(간격 이내) 생성 요청 경로가 스스로 회수해야 한다.

    회수 후 그 잡은 queued이므로 요청은 **살아 있는 그 잡에 합류**한다 — 봉인은 풀리고
    중복 잡도 생기지 않는다(중복 잡 2개가 같은 임시 오디오를 잡으면 WinError 32).
    """

    async def body():
        async with _env(local_worker=False) as sm:
            job_id = await _seed_queued_job(sm)
            await claim_job(ClaimRequest(worker_id=WID, version=__version__), x_worker_key=WKEY)
            assert (await _job(sm, job_id)).status == "processing"

            # 워커 사망 — 리스만 만료시킨다. 주기 태스크는 띄우지 않는다(이 경로만 검증)
            worker_api._LEASES[job_id] = (WID, 0.0)
            assert worker_api._SWEEPER_TASK is None

            resp = await generate_sync(
                GenerateRequest(video_id=VIDEO, lyrics=LYRICS), BackgroundTasks()
            )

            assert resp.job_id == job_id  # 회수된 그 잡에 합류
            assert (await _job(sm, job_id)).status == "queued"  # 봉인 해제
            assert await _count_jobs(sm) == 1  # 중복 잡 없음
            assert job_id not in worker_api._LEASES

    asyncio.run(body())


def test_regenerate_also_reclaims_a_dead_workers_job():
    async def body():
        async with _env(local_worker=False) as sm:
            job_id = await _seed_queued_job(sm)
            await claim_job(ClaimRequest(worker_id=WID, version=__version__), x_worker_key=WKEY)
            worker_api._LEASES[job_id] = (WID, 0.0)

            resp = await regenerate_sync(
                RegenerateRequest(video_id=VIDEO, lyrics=LYRICS), BackgroundTasks()
            )

            assert resp.job_id == job_id
            assert (await _job(sm, job_id)).status == "queued"
            assert await _count_jobs(sm) == 1

    asyncio.run(body())


def test_reclaim_does_not_touch_a_live_workers_job():
    """리스가 살아 있는 잡은 그대로 processing이고, 생성 요청은 정상적으로 합류한다 —
    리스 **부재**나 **유효한 리스**를 사망으로 오판하면 살아 있는 잡을 두 번 돌린다."""

    async def body():
        async with _env(local_worker=False) as sm:
            job_id = await _seed_queued_job(sm)
            await claim_job(ClaimRequest(worker_id=WID, version=__version__), x_worker_key=WKEY)

            resp = await generate_sync(
                GenerateRequest(video_id=VIDEO, lyrics=LYRICS), BackgroundTasks()
            )

            assert resp.job_id == job_id
            assert (await _job(sm, job_id)).status == "processing"  # 건드리지 않았다
            assert worker_api._LEASES[job_id][0] == WID
            assert await _count_jobs(sm) == 1

    asyncio.run(body())


def test_reclaim_leaves_in_process_and_meta_wait_jobs_alone():
    """**리스 부재 ≠ 사망.** 인프로세스 워커(local_worker=true)와 "번역 대기" 구간은 리스
    없이 정상적으로 processing이다. 리스 부재를 사망으로 읽었다면 이 잡들이 회수돼 같은
    영상에 중복 잡이 생기고 두 잡이 같은 임시 오디오를 잡아 WinError 32가 난다."""

    async def body():
        async with _env() as sm:
            async with sm() as s:
                job = await JobRepository(s).create(video_id=VIDEO, lyrics=LYRICS)
                # 인프로세스 처리 중 / 번역 대기 중 — 어느 쪽도 리스가 없다
                await JobRepository(s).update_status(
                    job.id, "processing", progress=48, stage="번역 대기"
                )
                await s.commit()
            assert job.id not in worker_api._LEASES

            await worker_api.reclaim_expired_leases()

            got = await _job(sm, job.id)
            assert got.status == "processing"  # 살아 있는 잡을 죽이지 않았다
            assert got.stage == "번역 대기"

    asyncio.run(body())


def test_lease_sweeper_also_requeues_link_jobs():
    async def body():
        async with _env() as sm:
            created = await create_link_job(
                LinkJobRequest(video_id=COVER, source_video_id=SOURCE)
            )
            await claim_job(ClaimRequest(worker_id=WID, version=__version__), x_worker_key=WKEY)
            worker_api._LEASES[f"link:{created.id}"] = (WID, 0.0)

            await worker_api._sweep_expired_leases()

            async with sm() as s:
                assert (await LinkJobRepository(s).get_by_id(created.id)).status == "queued"
            assert f"link:{created.id}" not in worker_api._LEASES

    asyncio.run(body())


def test_lease_sweeper_leaves_terminal_jobs_alone():
    """취소·완료된 잡은 리스가 만료돼도 queued로 되살리지 않는다 (무한 진동 방지)."""

    async def body():
        async with _env() as sm:
            job_id = await _seed_queued_job(sm)
            await claim_job(ClaimRequest(worker_id=WID, version=__version__), x_worker_key=WKEY)
            async with sm() as s:
                await JobRepository(s).update_status(job_id, "failed", error="요청으로 취소했어요")
                await s.commit()
            worker_api._LEASES[job_id] = (WID, 0.0)

            await worker_api._sweep_expired_leases()

            assert (await _job(sm, job_id)).status == "failed"

    asyncio.run(body())


def test_importing_the_app_does_not_start_the_sweeper():
    """임포트만으로 태스크가 뜨면 앱을 띄우지 않는 이 레포의 테스트가 루프에 태스크를 남긴다."""
    import everyric2.server.main  # noqa: F401

    assert worker_api._SWEEPER_TASK is None


def test_lifespan_starts_and_stops_the_sweeper(monkeypatch):
    """lifespan이 스윕을 띄우고 **종료 시 반드시 취소**한다 (태스크 누수 금지)."""
    from everyric2.server import main as server_main

    async def _noop():
        return None

    # 실 DB(파일 sqlite)를 건드리지 않도록 init_db/close_db를 무력화한다
    monkeypatch.setattr(server_main, "init_db", _noop)
    monkeypatch.setattr(server_main, "close_db", _noop)
    monkeypatch.setattr(server_main, "_gpu_available", lambda: True)

    async def body():
        async with server_main.lifespan(server_main.app):
            task = worker_api._SWEEPER_TASK
            assert task is not None and not task.done()
        assert worker_api._SWEEPER_TASK is None
        assert task.cancelled() or task.done()

    asyncio.run(body())


# ── ② 재기동 좀비 정리가 link_jobs도 대상 ─────────────────────────


def test_init_db_requeues_zombie_link_jobs_and_fails_zombie_sync_jobs(monkeypatch):
    """재기동 정리가 jobs만 훑으면 link_jobs는 processing으로 영구 잔류한다 —
    _LEASES는 재기동으로 유실돼 스윕도 되돌릴 수 없고, get_active_pair가 그 잡을 활성으로
    봐서 같은 쌍은 영구 pending이다.

    링크 잡은 failed가 아니라 **queued**로 되돌린다: 행 안의 (video_id, source_video_id)만으로
    완결돼 유실될 인메모리 상태가 없고, failed로 두면 get_recent_attempt가 그것을 쿨다운으로
    세어 그 쌍의 자동 재제출을 link_retry_cooldown_days 동안 막는다.
    """
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(db_conn, "engine", engine)
    monkeypatch.setattr(db_conn, "async_session", sm)

    async def body():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        async with sm() as s:
            zombie_link = await LinkJobRepository(s).create(COVER, SOURCE)
            await LinkJobRepository(s).update_status(zombie_link.id, "processing")
            queued_link = await LinkJobRepository(s).create(COVER, "OTHERvid001")
            zombie_job = await JobRepository(s).create(video_id=VIDEO, lyrics=LYRICS)
            await JobRepository(s).update_status(zombie_job.id, "processing", progress=40)
            await s.commit()

        await db_conn.init_db()

        async with sm() as s:
            repo = LinkJobRepository(s)
            assert (await repo.get_by_id(zombie_link.id)).status == "queued"
            # 아직 아무도 물지 않은 queued 링크 잡은 손대지 않는다
            assert (await repo.get_by_id(queued_link.id)).status == "queued"
            job = await JobRepository(s).get_by_id(zombie_job.id)
            assert job.status == "failed"
        await engine.dispose()

    asyncio.run(body())


# ── ③ 빈/기호뿐인 가사는 400 ──────────────────────────────────────


@pytest.mark.parametrize("lyrics", ["", "   ", "\n\n\n", "  \n \t \n", "...\n---\n***"])
def test_generate_rejects_lyrics_with_no_usable_line(lyrics):
    """빈 가사는 0줄 싱크를 만들고 그 lyrics_hash가 **캐시 히트로 영구히 0줄**을 돌려준다
    (GET /api/sync/{id}가 found:true, timestamps:[]). 잡을 만들기 전에 400으로 끊는다."""

    async def body():
        async with _env() as sm:
            with pytest.raises(HTTPException) as exc:
                await generate_sync(
                    GenerateRequest(video_id=VIDEO, lyrics=lyrics), BackgroundTasks()
                )
            assert exc.value.status_code == 400
            assert "가사" in exc.value.detail  # 사용자가 무엇을 하면 되는지 알 수 있어야 한다
            assert await _count_jobs(sm) == 0  # 잡을 만들지 않는다

    asyncio.run(body())


@pytest.mark.parametrize("lyrics", ["", "   \n  "])
def test_regenerate_rejects_lyrics_with_no_usable_line(lyrics):
    async def body():
        async with _env() as sm:
            with pytest.raises(HTTPException) as exc:
                await regenerate_sync(
                    RegenerateRequest(video_id=VIDEO, lyrics=lyrics), BackgroundTasks()
                )
            assert exc.value.status_code == 400
            assert await _count_jobs(sm) == 0

    asyncio.run(body())


def test_generate_accepts_a_single_real_line():
    """하한이 정상 입력을 막지 않는다 — 짧은 후크 한 줄도 사용자가 의도한 가사다."""

    async def body():
        async with _env(local_worker=False) as sm:
            resp = await generate_sync(
                GenerateRequest(video_id=VIDEO, lyrics="한 줄뿐인 가사"), BackgroundTasks()
            )
            assert resp.status == "processing"
            assert await _count_jobs(sm) == 1

    asyncio.run(body())


# ── ④ GPU를 태우는 경로의 한도 ────────────────────────────────────


def test_generate_daily_limit_counts_only_new_jobs_and_exempts_admin():
    """/generate에는 한도가 전혀 없었다 — 가사를 한 글자만 바꾸면 매번 새 lyrics_hash가
    되어 캐시·합류를 비켜 새 GPU 잡이 생긴다. 상한은 파괴적 행위보다 훨씬 느슨하지만
    무한 반복은 잘라낸다. **캐시 히트·진행 중 잡 합류는 예산을 먹지 않는다.**"""

    async def body():
        async with _env(admin_api_key="admin-secret", local_worker=False) as sm:
            prev = sync_api.DAILY_GENERATE_LIMIT
            sync_api.DAILY_GENERATE_LIMIT = 1
            try:
                first = await generate_sync(
                    GenerateRequest(video_id=VIDEO, lyrics=LYRICS), BackgroundTasks()
                )
                # 같은 가사 재요청 = 진행 중 잡 합류 → GPU를 쓰지 않으므로 한도와 무관
                joined = await generate_sync(
                    GenerateRequest(video_id=VIDEO, lyrics=LYRICS), BackgroundTasks()
                )
                assert joined.job_id == first.job_id
                assert await _count_jobs(sm) == 1

                # 가사를 바꾼 새 잡은 상한에 걸린다
                with pytest.raises(HTTPException) as exc:
                    await generate_sync(
                        GenerateRequest(video_id=VIDEO, lyrics=LYRICS + "\n세 번째 줄"),
                        BackgroundTasks(),
                    )
                assert exc.value.status_code == 429
                assert await _count_jobs(sm) == 1

                # 어드민 키는 면제
                await generate_sync(
                    GenerateRequest(video_id=VIDEO, lyrics=LYRICS + "\n네 번째 줄"),
                    BackgroundTasks(),
                    x_api_key="admin-secret",
                )
                assert await _count_jobs(sm) == 2
            finally:
                sync_api.DAILY_GENERATE_LIMIT = prev

    asyncio.run(body())


def test_generate_limit_is_off_without_an_admin_key():
    """로컬 단일 사용자(admin_api_key 미설정)는 기존처럼 제한이 없다."""

    async def body():
        async with _env(admin_api_key="", local_worker=False) as sm:
            prev = sync_api.DAILY_GENERATE_LIMIT
            sync_api.DAILY_GENERATE_LIMIT = 1
            try:
                for i in range(3):
                    await generate_sync(
                        GenerateRequest(video_id=VIDEO, lyrics=f"{LYRICS}\n{i}번"),
                        BackgroundTasks(),
                    )
                assert await _count_jobs(sm) == 3
            finally:
                sync_api.DAILY_GENERATE_LIMIT = prev

    asyncio.run(body())


async def _seed_titled_sync(sm, video_id, title):
    async with sm() as s:
        await SyncRepository(s).create(
            video_id=video_id,
            lyrics_hash="h1",
            timestamps=[{"text": "라인", "start": 1.0, "end": 2.0}],
            engine="ctc",
            title=title,
        )
        await s.commit()


async def _count_link_jobs(sm) -> int:
    async with sm() as s:
        return len((await s.execute(select(LinkJob))).scalars().all())


def test_link_candidates_daily_limit_caps_gpu_submissions():
    """GET 하나가 GPU 잡(영상 2개 다운로드 + demucs ×2 + 상관)을 제출한다. 억제가
    (영상, 후보) 쌍 쿨다운뿐이면 쿨다운을 비켜 가는 반복 제출이 가능하다 — 쌍과 무관한
    영상 단위 상한이 한 겹 더 필요하다."""

    async def body():
        # link_retry_cooldown_days=0 → 쌍 쿨다운 비활성(그 억제를 비켜 간 상황을 재현)
        async with _env(
            admin_api_key="admin-secret", link_retry_cooldown_days=0
        ) as sm:
            await _seed_titled_sync(sm, SOURCE, SOURCE_TITLE)
            prev = sync_api.DAILY_LINK_CANDIDATE_LIMIT
            sync_api.DAILY_LINK_CANDIDATE_LIMIT = 1
            try:
                first = await find_link_candidates(COVER, title=COVER_TITLE)
                assert first.status == "submitted"
                # 그 잡이 끝난 것으로 만든다 (get_active_pair의 pending 억제를 비켜 간다)
                async with sm() as s:
                    await LinkJobRepository(s).mark_done(first.job_id, False, 0.0, 0.1)
                    await s.commit()

                with pytest.raises(HTTPException) as exc:
                    await find_link_candidates(COVER, title=COVER_TITLE)
                assert exc.value.status_code == 429
                assert await _count_link_jobs(sm) == 1

                # 어드민 키는 면제된다
                admin = await find_link_candidates(
                    COVER, title=COVER_TITLE, x_api_key="admin-secret"
                )
                assert admin.status == "submitted"
                assert await _count_link_jobs(sm) == 2
            finally:
                sync_api.DAILY_LINK_CANDIDATE_LIMIT = prev

    asyncio.run(body())


def test_link_candidates_no_op_paths_do_not_consume_budget():
    """has_sync·none 등 GPU를 쓰지 않는 응답은 예산을 먹지 않는다."""

    async def body():
        async with _env(admin_api_key="admin-secret") as sm:
            prev = sync_api.DAILY_LINK_CANDIDATE_LIMIT
            sync_api.DAILY_LINK_CANDIDATE_LIMIT = 1
            try:
                # 코퍼스에 후보가 없다 → none (제출 없음)
                for _ in range(3):
                    resp = await find_link_candidates(COVER, title=COVER_TITLE)
                    assert resp.status == "none"
                # 예산이 남아 있으므로 후보가 생기면 여전히 제출된다
                await _seed_titled_sync(sm, SOURCE, SOURCE_TITLE)
                assert (await find_link_candidates(COVER, title=COVER_TITLE)).status == "submitted"
            finally:
                sync_api.DAILY_LINK_CANDIDATE_LIMIT = prev

    asyncio.run(body())


def test_post_link_jobs_respects_the_pair_cooldown():
    """POST /api/link-jobs의 억제는 진행 중 중복 확인뿐이었다 — 끝난 쌍을 반복 제출하면
    매번 GPU를 새로 태울 수 있다. 자동 제출 경로와 같은 쿨다운 기준을 쓴다."""

    async def body():
        async with _env(link_retry_cooldown_days=14) as sm:
            first = await create_link_job(LinkJobRequest(video_id=COVER, source_video_id=SOURCE))
            async with sm() as s:
                await LinkJobRepository(s).mark_done(first.id, True, 1.0, 0.9)
                await s.commit()

            again = await create_link_job(LinkJobRequest(video_id=COVER, source_video_id=SOURCE))
            assert again.id == first.id  # 새 잡을 만들지 않고 이력을 돌려준다
            assert await _count_link_jobs(sm) == 1

            # 쿨다운이 지난 이력은 막지 않는다
            async with sm() as s:
                row = await LinkJobRepository(s).get_by_id(first.id)
                row.created_at = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(
                    days=30
                )
                await s.commit()
            fresh = await create_link_job(LinkJobRequest(video_id=COVER, source_video_id=SOURCE))
            assert fresh.id != first.id
            assert await _count_link_jobs(sm) == 2

    asyncio.run(body())


# ── ⑤ 취소가 부활하지 않는다 ──────────────────────────────────────


def test_queue_after_line_meta_does_not_revive_a_cancelled_job(monkeypatch):
    """취소 확인과 상태 쓰기 사이에 취소가 들어오면, 무조건 쓰기는 failed를 queued로 되살린다
    → 워커가 물어 processing이 되고, 취소된 잡은 fail을 제출하지 않아 processing에 남고,
    만료 스윕이 다시 queued로 돌려 무한 진동한다. 조건부 쓰기로 그 창을 없앤다."""

    async def body():
        async with _env(local_worker=False) as sm:
            job_id = await _seed_queued_job(sm)
            # line_meta가 이미 도착한 상태 → await_line_meta_arrival이 즉시 True
            worker_core.stash_line_meta(job_id, [{"text": "첫 줄"}])

            async def fake_consume(jid: str) -> bool:
                # 취소 API가 정확히 이 순간(확인 직후, 상태 쓰기 직전)에 도착한 상황
                async with sm() as s:
                    await JobRepository(s).update_status(
                        jid, "failed", error="요청으로 취소했어요"
                    )
                    await s.commit()
                return False  # 확인 시점에는 아직 취소를 보지 못했다

            monkeypatch.setattr(worker_core, "_consume_cancel", fake_consume)

            await _queue_after_line_meta(job_id)

            job = await _job(sm, job_id)
            assert job.status == "failed"
            assert job.error == "요청으로 취소했어요"

    asyncio.run(body())


def test_queue_after_line_meta_still_queues_a_live_job():
    """정상 경로는 그대로 — 조건부 쓰기가 대기 중 잡의 큐 진입을 막지 않는다."""

    async def body():
        async with _env(local_worker=False) as sm:
            job_id = await _seed_queued_job(sm)
            worker_core.stash_line_meta(job_id, [{"text": "첫 줄"}])

            await _queue_after_line_meta(job_id)

            assert (await _job(sm, job_id)).status == "queued"

    asyncio.run(body())


# ── ⑥ 잡 종결 vs 스태시 쓰기 경합 ────────────────────────────────


@contextlib.asynccontextmanager
async def _flip_job_on_recheck(sm, job_id: str, status: str, **fields):
    """`_attach_line_meta_to_job`의 **두 번째** get_session 직전에 잡을 종결시킨다.

    첫 읽기(진행 중 확인)와 스태시 쓰기 사이에 잡이 끝나는 실제 경합을 결정적으로 재현하는
    주입점이다 — 그 창에서 예전 코드는 스태시를 영구 잔류시키고 applied="stashed"로
    사실과 다르게 답했다.
    """
    real = sync_api.get_session
    calls = {"n": 0}

    @contextlib.asynccontextmanager
    async def patched():
        calls["n"] += 1
        if calls["n"] == 2:
            async with sm() as s:
                await JobRepository(s).update_status(job_id, status, **fields)
                await s.commit()
        async with real() as s:
            yield s

    sync_api.get_session = patched
    try:
        yield
    finally:
        sync_api.get_session = real


def test_attach_line_meta_merges_and_reclaims_stash_when_the_job_completes_mid_write():
    async def body():
        async with _env() as sm:
            async with sm() as s:
                job = await JobRepository(s).create(video_id=VIDEO, lyrics=LYRICS)
                await JobRepository(s).update_status(job.id, "processing", progress=50)
                sync_row = await SyncRepository(s).create(
                    video_id=VIDEO,
                    lyrics_hash=hash_lyrics(LYRICS),
                    timestamps=[{"text": "첫 줄", "start": 1.0, "end": 2.0}],
                    engine="ctc",
                )
                await s.commit()

            async with _flip_job_on_recheck(
                sm, job.id, "completed", progress=100, result_id=sync_row.id
            ):
                applied = await _attach_line_meta_to_job(
                    job.id, [LineMeta(text="첫 줄", translation="first line")]
                )

            # 응답이 실제로 일어난 일과 일치한다 (stashed가 아니라 merged)
            assert applied is not None
            assert applied.applied == "merged"
            assert applied.merged_segments == 1
            # 스태시가 회수됐다 (프로세스 수명 동안 잔류하던 누수)
            assert job.id not in worker_core._PENDING_LINE_META
            # 메타가 실제로 싱크에 남았다
            async with sm() as s:
                row = await SyncRepository(s).get_by_id(sync_row.id)
                assert row.timestamps["segments"][0]["translation"] == "first line"

    asyncio.run(body())


def test_attach_line_meta_drops_and_reclaims_stash_when_the_job_fails_mid_write():
    async def body():
        async with _env() as sm:
            async with sm() as s:
                job = await JobRepository(s).create(video_id=VIDEO, lyrics=LYRICS)
                await JobRepository(s).update_status(job.id, "processing", progress=50)
                await s.commit()

            async with _flip_job_on_recheck(
                sm, job.id, "failed", error="요청으로 취소했어요"
            ):
                applied = await _attach_line_meta_to_job(
                    job.id, [LineMeta(text="첫 줄", translation="first line")]
                )

            assert applied is not None
            assert applied.applied == "dropped"
            assert job.id not in worker_core._PENDING_LINE_META

    asyncio.run(body())


def test_attach_line_meta_still_stashes_for_a_live_job():
    """정상 경로 — 진행 중 잡은 그대로 스태시되고 applied="stashed"다."""

    async def body():
        async with _env() as sm:
            async with sm() as s:
                job = await JobRepository(s).create(video_id=VIDEO, lyrics=LYRICS)
                await JobRepository(s).update_status(job.id, "processing", progress=50)
                await s.commit()

            applied = await _attach_line_meta_to_job(
                job.id, [LineMeta(text="첫 줄", translation="first line")]
            )
            assert applied is not None and applied.applied == "stashed"
            assert worker_core._PENDING_LINE_META[job.id][0]["translation"] == "first line"

    asyncio.run(body())


# ── ⑦ GET /api/job/{id}가 result_id를 존중한다 ────────────────────


def test_job_status_returns_the_sync_this_job_actually_made():
    """result_id를 저장해 두고도 그 영상의 '최신' 싱크를 돌려주면, 그 사이 다른 가사로 만든
    싱크가 이 잡의 결과로 나가 계약이 틀린다."""

    async def body():
        async with _env() as sm:
            async with sm() as s:
                repo = SyncRepository(s)
                mine = await repo.create(
                    video_id=VIDEO,
                    lyrics_hash=hash_lyrics(LYRICS),
                    timestamps=[{"text": "이 잡의 결과", "start": 1.0, "end": 2.0}],
                    engine="ctc",
                )
                newer = await repo.create(
                    video_id=VIDEO,
                    lyrics_hash="other-hash",
                    timestamps=[{"text": "나중에 만든 다른 싱크", "start": 5.0, "end": 6.0}],
                    engine="ctc",
                )
                # created_at은 초 단위라 동일 초 동결이 생긴다 — 순서를 명시적으로 못박는다
                base = datetime(2026, 7, 25, 12, 0, 0)
                mine.created_at = base
                newer.created_at = base + timedelta(minutes=5)
                job = await JobRepository(s).create(video_id=VIDEO, lyrics=LYRICS)
                await JobRepository(s).update_status(
                    job.id, "completed", progress=100, result_id=mine.id
                )
                await s.commit()

            resp = await get_job_status(job.id)
            assert resp.status == "completed"
            assert resp.timestamps[0]["text"] == "이 잡의 결과"

    asyncio.run(body())


def test_job_status_falls_back_to_latest_when_the_result_row_is_gone():
    """DELETE /api/sync/{video_id}(초기화)는 싱크만 지우고 completed 잡 행은 남긴다 —
    죽은 result_id 하나로 응답을 비우지 않고 같은 영상의 최신 싱크로 폴백한다."""

    async def body():
        async with _env() as sm:
            async with sm() as s:
                remaining = await SyncRepository(s).create(
                    video_id=VIDEO,
                    lyrics_hash="new-hash",
                    timestamps=[{"text": "남아 있는 싱크", "start": 3.0, "end": 4.0}],
                    engine="ctc",
                )
                job = await JobRepository(s).create(video_id=VIDEO, lyrics=LYRICS)
                await JobRepository(s).update_status(
                    job.id,
                    "completed",
                    progress=100,
                    result_id="00000000-0000-0000-0000-000000000000",
                )
                await s.commit()

            resp = await get_job_status(job.id)
            assert resp.timestamps[0]["text"] == "남아 있는 싱크"
            assert remaining.id  # 폴백이 짚은 대상이 존재한다

    asyncio.run(body())


# ── ⑧ 고아 잡 TTL 리퍼 (외부 감사 #7) ─────────────────────────────
#
# 리스 스위퍼(①)는 원격 워커가 리스를 쥔 잡만 커버한다. 인프로세스 워커와 "번역 대기"
# 구간(test_reclaim_leaves_in_process_and_meta_wait_jobs_alone이 못박은 그 케이스)은 리스가
# 없어 그 스위퍼의 대상이 아니다 — 실측(2026-08)으로 그 구간에서 하트비트가 끊긴 잡 하나가
# 6.3시간 동안 processing에 정체했다. 이 리퍼는 updated_at(마지막 진행 갱신) 기준의 TTL로
# 그 구멍을 별도로 메운다.


async def _backdate_processing_job(sm, job_id: str, updated_at: datetime, **fields) -> None:
    """잡을 processing으로 두고 updated_at을 직접 지정해 커밋한다.

    updated_at은 onupdate=func.now()이지만, ORM이 그 속성에 명시적으로 대입된 값을 SET
    절에 실으므로 onupdate가 덮어쓰지 않는다 — 이 파일의 다른 테스트(예:
    test_post_link_jobs_respects_the_pair_cooldown)가 created_at에 쓰는 것과 같은 패턴."""
    async with sm() as s:
        job = await JobRepository(s).get_by_id(job_id)
        job.status = "processing"
        job.updated_at = updated_at
        for k, v in fields.items():
            setattr(job, k, v)
        await s.commit()


def test_orphan_sweep_reaps_a_job_whose_heartbeat_went_stale():
    """실측 사고 재현: stage='번역 대기', progress=48로 하트비트가 끊긴 잡은 TTL을 넘기면
    failed로 회수되고, 에러 메시지는 사용자에게 보일 한국어 한 문장이어야 한다."""

    async def body():
        async with _env(orphan_job_ttl_min=50) as sm:
            job_id = await _seed_queued_job(sm)
            stale = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(
                hours=6, minutes=20
            )
            await _backdate_processing_job(sm, job_id, stale, progress=48, stage="번역 대기")

            reaped = await orphan_reaper.sweep_orphan_jobs()

            assert reaped == 1
            job = await _job(sm, job_id)
            assert job.status == "failed"
            assert job.error == orphan_reaper.ORPHAN_RECOVERY_MESSAGE

    asyncio.run(body())


def test_orphan_sweep_preserves_a_job_with_a_recent_heartbeat():
    """방금 진행률을 보고한(updated_at이 최근인) 잡은 TTL 안이므로 건드리지 않는다 —
    판정 기준이 created_at(시작 시각)이었다면 오래 걸리는 정상 잡까지 죽였을 것이다."""

    async def body():
        async with _env(orphan_job_ttl_min=50) as sm:
            job_id = await _seed_queued_job(sm)
            recent = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=2)
            await _backdate_processing_job(sm, job_id, recent, progress=60, stage="타이밍 보정")

            reaped = await orphan_reaper.sweep_orphan_jobs()

            assert reaped == 0
            job = await _job(sm, job_id)
            assert job.status == "processing"
            assert job.progress == 60

    asyncio.run(body())


def test_orphan_sweep_ignores_non_processing_jobs():
    """queued 등 processing이 아닌 잡은 updated_at이 아무리 오래돼도 건드리지 않는다."""

    async def body():
        async with _env(orphan_job_ttl_min=50) as sm:
            job_id = await _seed_queued_job(sm)  # status == "queued"
            stale = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=10)
            async with sm() as s:
                job = await JobRepository(s).get_by_id(job_id)
                job.updated_at = stale
                await s.commit()

            reaped = await orphan_reaper.sweep_orphan_jobs()

            assert reaped == 0
            assert (await _job(sm, job_id)).status == "queued"

    asyncio.run(body())


def test_orphan_sweep_disabled_when_ttl_is_non_positive():
    """TTL을 0 이하로 두면 리퍼가 비활성 — 며칠이 지나도 회수하지 않는다(설정의 "0 disables"
    관례)."""

    async def body():
        async with _env(orphan_job_ttl_min=0) as sm:
            job_id = await _seed_queued_job(sm)
            stale = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=3)
            await _backdate_processing_job(sm, job_id, stale, progress=48, stage="번역 대기")

            reaped = await orphan_reaper.sweep_orphan_jobs()

            assert reaped == 0
            assert (await _job(sm, job_id)).status == "processing"

    asyncio.run(body())


def test_importing_the_app_does_not_start_the_orphan_sweeper():
    """리스 스위퍼(test_importing_the_app_does_not_start_the_sweeper)와 같은 이유 —
    임포트만으로 태스크가 뜨면 앱을 띄우지 않는 이 레포의 테스트에 태스크가 남는다."""
    import everyric2.server.main  # noqa: F401

    assert orphan_reaper._SWEEPER_TASK is None


def test_lifespan_starts_and_stops_the_orphan_sweeper(monkeypatch):
    """lifespan이 고아 잡 스위퍼도 리스 스위퍼와 함께 띄우고, 종료 시 반드시 취소한다."""
    from everyric2.server import main as server_main

    async def _noop():
        return None

    monkeypatch.setattr(server_main, "init_db", _noop)
    monkeypatch.setattr(server_main, "close_db", _noop)
    monkeypatch.setattr(server_main, "_gpu_available", lambda: True)

    async def body():
        async with server_main.lifespan(server_main.app):
            lease_task = worker_api._SWEEPER_TASK
            orphan_task = orphan_reaper._SWEEPER_TASK
            assert lease_task is not None and not lease_task.done()
            assert orphan_task is not None and not orphan_task.done()
        assert worker_api._SWEEPER_TASK is None
        assert orphan_reaper._SWEEPER_TASK is None
        assert orphan_task.cancelled() or orphan_task.done()

    asyncio.run(body())
