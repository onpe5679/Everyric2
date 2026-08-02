"""원격 GPU 워커에서의 line_meta 지연 도착 — 하트비트 역방향 채널 테스트.

원격 워커는 claim 시점의 스태시 스냅샷만 받으므로, 정렬 도중에 도착한 번역·독음을 받을 길이
진행률 하트비트 응답뿐이다. 여기서 못 박는 계약:
  ① 하트비트 응답에 line_meta가 실려 워커 쪽 코어 리졸버가 그것으로 정렬한다.
  ② "아직 안 왔음(None)"과 "붙일 것 없음 확정(빈 리스트)"이 구분돼 전달된다.
  ③ 상한을 넘으면 원문 정렬로 폴백한다 (무한 대기 없음).
  ④ 대기 중에도 취소가 먹는다 (결과를 제출하지 않는다).
  ⑤ 대기 중 하트비트가 리스를 갱신해 만료 스윕이 잡을 회수하지 않는다.
  ⑥ 새 필드를 모르는 구버전 워커/서버가 붙어도 깨지지 않는다 (기존 원문 정렬 동작).

**서버 프로세스와 워커 프로세스의 전역을 일부러 분리한다.** 실제 배포에서는 둘이 다른
프로세스라 스태시(_PENDING_LINE_META)가 공유되지 않는데, 한 프로세스에서 도는 테스트는
그냥 두면 워커 쪽 코어가 서버 스태시를 직접 보고 통과해 버려(위양성) 하트비트가 채널로
동작하는지 전혀 검증하지 못한다. 그래서 서버 쪽 읽기 지점(worker_api._peek_line_meta)만
테스트 로컬 dict로 갈아끼우고, 코어 전역은 워커 쪽 시야로 남긴다 — 코어가 line_meta를
보게 되는 유일한 경로가 RemoteHooks._absorb다.

GPU·네트워크는 쓰지 않는다: 오디오 확보와 정렬 본체만 목이고, HTTP는 실제 라우트 핸들러로
브리지한다(요청/응답 모델을 그대로 지난다). 대기 상한·폴링 간격·모니터 간격은 주입해
테스트가 실제로 오래 기다리지 않게 한다.
"""

import asyncio
import contextlib
import time

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from everyric2 import __version__, cli, gpu_mem
from everyric2.config.settings import get_settings
from everyric2.server import worker as worker_core
from everyric2.server.api import worker as worker_api
from everyric2.server.api.worker import (
    CacheCheckRequest,
    ClaimRequest,
    ProgressRequest,
    ResultRequest,
    cache_check,
    claim_job,
    report_progress,
    submit_result,
)
from everyric2.server.db import connection as db_conn
from everyric2.server.db.models import Base
from everyric2.server.db.repository import JobRepository, SyncRepository, hash_lyrics

VIDEO = "REMOTEVID01"
LYRICS = "一行目\n二行目"
WKEY = "test-worker-key"
WID = "worker-remote"
META = [{"text": "一行目", "pronunciation": "이치교메", "translation": "첫 줄"}]


# ── 환경 ──────────────────────────────────────────────────────────


@contextlib.asynccontextmanager
async def _env(**server_overrides):
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
    overrides = {"worker_key": WKEY, "local_worker": False, "media_cache_url": ""}
    overrides.update(server_overrides)
    saved = {k: getattr(server, k) for k in overrides}
    for k, v in overrides.items():
        object.__setattr__(server, k, v)
    _clear_globals()
    try:
        yield sm
    finally:
        db_conn.async_session = orig
        for k, v in saved.items():
            object.__setattr__(server, k, v)
        _clear_globals()
        await engine.dispose()


def _clear_globals() -> None:
    worker_api._LEASES.clear()
    worker_api._WORKER_AUDIO.clear()
    worker_core._PENDING_LINE_META.clear()
    worker_core._PENDING_ATTRIBUTION.clear()
    worker_core._PENDING_TITLE.clear()
    worker_core._PENDING_FORCE.clear()
    worker_core._PENDING_META_WAIT.clear()
    worker_core._CANCEL_REQUESTED.clear()


async def _seed_queued_job(sm, lyrics=LYRICS) -> str:
    async with sm() as s:
        job = await JobRepository(s).create(video_id=VIDEO, lyrics=lyrics)
        await JobRepository(s).update_status(job.id, "queued", progress=0)
        await s.commit()
        return job.id


async def _claim(worker_id: str = WID, supports: bool = True):
    """claim 호출 — supports=False면 하트비트 채널을 모르는 구버전 워커를 흉내낸다."""
    return await claim_job(
        ClaimRequest(
            worker_id=worker_id,
            version=__version__,
            supports_line_meta_heartbeat=supports,
        ),
        x_worker_key=WKEY,
    )


async def _heartbeat(job_id: str, want: bool = True, progress: int = 48, stage: str | None = None):
    return await report_progress(
        job_id,
        ProgressRequest(
            progress=progress,
            stage=stage or worker_core.LINE_META_WAIT_STAGE,
            want_line_meta=want,
        ),
        x_worker_key=WKEY,
        x_worker_id=WID,
    )


def _server_view(monkeypatch, stash: dict) -> None:
    """서버 쪽 line_meta 읽기 지점만 테스트 로컬 dict로 갈아끼운다 (프로세스 분리 흉내).

    코어 전역(_PENDING_LINE_META)은 워커 쪽 시야로 남으므로, 코어가 line_meta를 보게 되는
    유일한 경로는 하트비트 응답을 흡수하는 RemoteHooks._absorb다."""
    monkeypatch.setattr(worker_api, "_peek_line_meta", lambda job_id: stash.get(job_id))


# ── 파이프라인 목 (GPU 없음) ───────────────────────────────────────


def _mock_audio(monkeypatch, tmp_path):
    audio_file = tmp_path / "audio.m4a"
    audio_file.write_bytes(b"fake-audio")
    monkeypatch.setattr(
        worker_core,
        "_acquire_audio",
        lambda job: {"audio_path": str(audio_file), "audio_hash": "deadbeef"},
    )
    monkeypatch.setattr(worker_core, "_audio_duration_sec", lambda p: 200.0)
    monkeypatch.setattr(gpu_mem, "reclaim_after_job", lambda: None)
    return audio_file


def _mock_alignment(monkeypatch, tmp_path, captured: dict):
    """_run_alignment 대역 — 리졸버를 부르고 무엇을 받았는지 기록한다 (실제 계약과 동일)."""
    audio_file = _mock_audio(monkeypatch, tmp_path)

    def fake(
        audio_path, lyrics, language, line_meta=None, on_stage=None, resolver=None,
        video_id=None, min_depth=None,
    ):
        try:
            if on_stage is not None:
                on_stage("보컬 분리")
            if resolver is not None:
                # 실제 _run_alignment와 같은 순서 — 대기 단계를 먼저 보고하고 나서 기다린다
                # (이 보고가 대기 중 하트비트의 단계명이 되고, 진행률 창을 대기 창에 묶는다)
                if on_stage is not None:
                    on_stage(worker_core.LINE_META_WAIT_STAGE)
                line_meta = resolver()
            captured["line_meta"] = line_meta
            captured["calls"] = captured.get("calls", 0) + 1
            if on_stage is not None:
                on_stage("전사 정렬")
            return {
                "timestamps": [
                    {"text": "一行目", "start": 0.0, "end": 1.0},
                    {"text": "二行目", "start": 1.0, "end": 2.0},
                ],
                "language": language or "ja",
                "quality_score": 0.9,
                "debug": {"alignment_text": "original"},
                "alignment_text": "original",
                "tempo": None,
                "key": None,
            }
        finally:
            audio_file.unlink(missing_ok=True)

    monkeypatch.setattr(worker_core, "_run_alignment", fake)
    return audio_file


def _fast_waits(monkeypatch, bound: float, monitor_interval: float = 0.02) -> None:
    """대기 상한·폴링 간격·하트비트 간격을 주입 — 테스트가 실제로 오래 기다리지 않게."""
    monkeypatch.setattr(worker_core, "LINE_META_WAIT_SEC", bound)
    monkeypatch.setattr(worker_core, "LINE_META_POLL_SEC", 0.01)
    real_monitor = worker_core._stage_monitor
    monkeypatch.setattr(
        worker_core,
        "_stage_monitor",
        lambda report, stage_holder, start, interval=None: real_monitor(
            report, stage_holder, start, monitor_interval
        ),
    )


def _bridge(loop, job_id: str, calls: list):
    """RemoteHooks._post 대역 — 실제 라우트 핸들러로 브리지한다 (요청/응답 모델 그대로).

    워커의 HTTP는 스레드에서 도는 동기 호출이므로, 라우트 코루틴은 메인 루프에 실어 보낸다
    (실제 배포의 '별 프로세스 + HTTP'와 같은 비동기 경계를 재현한다)."""

    def post(path: str, body: dict) -> dict:
        calls.append((path, dict(body)))
        if path.endswith("/progress"):
            coro = report_progress(
                job_id,
                ProgressRequest(**body),
                x_worker_key=WKEY,
                x_worker_id=WID,
            )
        elif path.endswith("/cache-check"):
            coro = cache_check(
                job_id,
                CacheCheckRequest(**body),
                x_worker_key=WKEY,
                x_worker_id=WID,
            )
        else:  # pragma: no cover - 테스트가 쓰지 않는 경로
            raise AssertionError(f"예상치 못한 워커 API 호출: {path}")
        return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=10).model_dump()

    return post


# ── ① 하트비트가 line_meta를 실어 보낸다 (서버 쪽 계약) ────────────


def test_claim_flags_await_line_meta_for_pending_jobs():
    """line_meta 예고 잡은 claim 응답에 await_line_meta=true로 나간다 (값은 아직 없다)."""

    async def body():
        async with _env() as sm:
            job_id = await _seed_queued_job(sm)
            worker_core.stash_line_meta_wait(job_id)
            resp = await _claim()
            assert resp.job is not None
            assert resp.job.await_line_meta is True
            assert resp.job.line_meta is None

    asyncio.run(body())


def test_claim_does_not_flag_ordinary_jobs():
    """예고가 없는 평범한 잡은 false — 켜지면 매 잡이 상한만큼 헛되게 기다린다."""

    async def body():
        async with _env() as sm:
            await _seed_queued_job(sm)
            resp = await _claim()
            assert resp.job.await_line_meta is False

    asyncio.run(body())


def test_heartbeat_carries_line_meta_and_attribution_when_requested():
    """want_line_meta를 켠 하트비트에 스태시 값이 실려 나간다 (출처도 같은 채널로)."""

    async def body():
        async with _env() as sm:
            job_id = await _seed_queued_job(sm)
            worker_core.stash_line_meta_wait(job_id)
            await _claim()
            # 클레임 뒤에 확장이 번역을 붙인 상황
            worker_core.stash_line_meta(job_id, META)
            worker_core.stash_attribution(job_id, {"source": "보카로 가사 위키"})

            resp = await _heartbeat(job_id, want=True)
            assert resp.cancel_requested is False
            assert resp.line_meta == META
            assert resp.attribution == {"source": "보카로 가사 위키"}
            # 중계일 뿐 소비가 아니다 — 재클레임/재시도가 다시 받아야 한다
            assert worker_core._PENDING_LINE_META[job_id] == META

    asyncio.run(body())


# ── ② "아직 없음"과 "확정 없음"의 구분 ────────────────────────────


def test_heartbeat_distinguishes_not_arrived_from_confirmed_empty():
    """None(아직)과 빈 리스트(붙일 것 없음 확정)는 다른 값으로 전달된다.

    이 구분이 무너지면 워커가 영원히 오지 않을 값을 상한까지 기다리거나(빈 리스트를 None으로
    보냈을 때) 아직 오는 중인 번역을 버린다(None을 빈 리스트로 보냈을 때)."""

    async def body():
        async with _env() as sm:
            job_id = await _seed_queued_job(sm)
            worker_core.stash_line_meta_wait(job_id)
            await _claim()

            # 아직 안 옴
            assert (await _heartbeat(job_id)).line_meta is None
            # 확장이 번역 실패를 빈 배열로 알린 상태
            worker_core.stash_line_meta(job_id, [])
            resp = await _heartbeat(job_id)
            assert resp.line_meta == []
            assert resp.line_meta is not None

    asyncio.run(body())


def test_claim_with_confirmed_empty_meta_does_not_arm_a_wait():
    """빈 리스트가 클레임에 실려 오면 기다릴 것이 없다 — 대기를 걸지 않는다.

    falsy 검사로 짜면 이 잡이 아무도 보내지 않을 값을 상한까지 기다린다."""

    async def body():
        async with _env(max_job_audio_sec=0) as sm:
            job_id = await _seed_queued_job(sm)
            worker_core.stash_line_meta_wait(job_id)
            worker_core.stash_line_meta(job_id, [])
            resp = await _claim()
            assert resp.job.line_meta == []
            assert resp.job.await_line_meta is True

            captured = await _run_loop_once(resp)
            # 대기를 걸지 않았고(리졸버 없음) 원문으로 정렬했다
            assert captured["job_input"].await_line_meta is False
            assert captured["hooks_want"] is False

    asyncio.run(body())


# ── ⑤ 대기 중 리스 갱신 (스윕이 잡을 회수하지 않는다) ──────────────


def test_wait_heartbeat_renews_lease_and_survives_the_sweep():
    """대기 단계 하트비트가 리스를 갱신해 만료 스윕이 잡을 회수하지 않는다.

    갱신이 끊기면 스윕이 잡을 queued로 되돌려 다른 워커에게 넘긴다(같은 잡을 두 번 돌리고,
    처음 워커의 결과는 뒤늦은 제출로 거부된다). 음성 대조군으로 "하트비트가 없으면 실제로
    회수된다"까지 확인해 이 테스트가 헛돌지 않음을 보인다."""

    async def body():
        async with _env() as sm:
            job_id = await _seed_queued_job(sm)
            worker_core.stash_line_meta_wait(job_id)
            await _claim()

            # 리스를 이미 만료한 값으로 낮춘다 (긴 대기로 갱신이 밀린 상황)
            worker_api._LEASES[job_id] = (WID, time.time() - 1.0)
            await _heartbeat(job_id, want=True)
            assert worker_api._LEASES[job_id][1] > time.time()

            await worker_api._sweep_expired_leases()
            assert worker_api._LEASES[job_id][0] == WID
            async with sm() as s:
                assert (await JobRepository(s).get_by_id(job_id)).status == "processing"

            # 음성 대조군: 하트비트를 멈추면 같은 스윕이 잡을 회수한다
            worker_api._LEASES[job_id] = (WID, time.time() - 1.0)
            await worker_api._sweep_expired_leases()
            assert job_id not in worker_api._LEASES
            async with sm() as s:
                assert (await JobRepository(s).get_by_id(job_id)).status == "queued"

    asyncio.run(body())


def test_wait_stage_is_the_heartbeat_the_monitor_sends():
    """대기 중에도 하트비트가 계속 나가야 한다 — 코어 모니터가 그 역할을 한다.

    대기는 정렬 스레드에서 블로킹으로 일어나므로, 리스를 살려 두는 주체는 이벤트 루프에서
    도는 _stage_monitor뿐이다. 대기 단계에서도 멈추지 않고 report를 계속 부르는지 본다."""

    async def body():
        reported: list[tuple[int, str]] = []

        async def report(progress, stage):
            reported.append((progress, stage))

        holder = {"stage": worker_core.LINE_META_WAIT_STAGE}
        monitor = asyncio.create_task(
            worker_core._stage_monitor(report, holder, start=36, interval=0.01)
        )
        await asyncio.sleep(0.2)
        monitor.cancel()

        assert len(reported) >= 3, "대기 중 하트비트가 멈췄다 — 리스가 회수된다"
        assert {stage for _, stage in reported} == {worker_core.LINE_META_WAIT_STAGE}

    asyncio.run(body())


def test_cancel_response_does_not_leak_line_meta():
    """취소 응답은 취소만 알린다 — 죽은 잡에 실어 보낼 것은 없다 (스태시도 정리된다)."""

    async def body():
        async with _env() as sm:
            job_id = await _seed_queued_job(sm)
            worker_core.stash_line_meta_wait(job_id)
            await _claim()
            worker_core.stash_line_meta(job_id, META)
            worker_core.request_cancel(job_id)
            async with sm() as s:
                await JobRepository(s).update_status(job_id, "failed", error="요청으로 취소했어요")
                await s.commit()

            resp = await _heartbeat(job_id, want=True)
            assert resp.cancel_requested is True
            assert resp.line_meta is None
            assert job_id not in worker_core._PENDING_LINE_META
            assert job_id not in worker_core._PENDING_META_WAIT

    asyncio.run(body())


def test_terminal_submit_clears_the_await_flag():
    """잡이 끝나면 대기 예고 스태시도 비운다 — 남으면 정리 지점 없이 샌다."""

    async def body():
        async with _env() as sm:
            job_id = await _seed_queued_job(sm)
            worker_core.stash_line_meta_wait(job_id)
            await _claim()
            await submit_result(
                job_id,
                ResultRequest(timestamps=[{"text": "一行目", "start": 0.0, "end": 1.0}]),
                x_worker_key=WKEY,
                x_worker_id=WID,
            )
            assert job_id not in worker_core._PENDING_META_WAIT

    asyncio.run(body())


# ── ⑥ 구버전 워커 호환 (새 필드를 모른다) ──────────────────────────


def test_old_worker_is_not_told_to_wait_and_is_warned_about(caplog):
    """능력을 광고하지 않은 워커에게는 대기를 지시하지 않는다 (기존 동작 = 원문 정렬).

    버전 문자열로는 이 능력을 알 수 없다(__version__이 릴리스마다 오르지 않아 구버전 코드도
    claim 게이트를 통과한다). 그래서 대기 지시는 능력 광고에만 붙이고, 이 잡이 독음 정렬
    품질을 잃는다는 사실은 경고로 남겨 조용한 저하가 로그에 드러나게 한다."""

    async def body():
        async with _env() as sm:
            job_id = await _seed_queued_job(sm)
            worker_core.stash_line_meta_wait(job_id)
            with caplog.at_level("WARNING", logger="everyric2.server.api.worker"):
                resp = await _claim(supports=False)
            # 기다릴 수 없는 워커에게 기다리라고 하지 않는다 (지시했다면 상한만 태운다)
            assert resp.job.await_line_meta is False
            assert "align on the original text" in caplog.text
            assert job_id in caplog.text

    asyncio.run(body())


def test_old_worker_request_without_the_flag_gets_no_line_meta():
    """want_line_meta를 모르는 구버전 워커에게는 실어 보내지 않는다 (기존 동작 그대로).

    구버전 워커는 응답 본문에서 cancel_requested만 읽으므로 필드가 늘어도 깨지지 않고,
    line_meta를 못 받아 예전처럼 원문 정렬로 완주한다 — 조용한 품질 저하는 남으므로
    이 경로의 안전망은 버전 게이트(claim 409)다."""

    async def body():
        async with _env() as sm:
            job_id = await _seed_queued_job(sm)
            worker_core.stash_line_meta_wait(job_id)
            await _claim()
            worker_core.stash_line_meta(job_id, META)

            # 구버전 워커의 요청 본문 — 새 필드가 아예 없다 (모델 기본값으로 받아들여진다)
            legacy = ProgressRequest(progress=48, stage="보컬 분리")
            assert legacy.want_line_meta is False
            resp = await report_progress(
                job_id, legacy, x_worker_key=WKEY, x_worker_id=WID
            )
            assert resp.cancel_requested is False
            assert resp.line_meta is None
            # 리스 갱신 같은 기존 동작은 그대로다
            assert worker_api._LEASES[job_id][0] == WID

    asyncio.run(body())


def test_worker_against_old_server_falls_back_to_original(monkeypatch, tmp_path):
    """line_meta 필드를 안 보내는 구버전 서버에 붙어도 상한 안에 원문 정렬로 완주한다."""

    async def body():
        async with _env() as sm:
            job_id = await _seed_queued_job(sm)
            _fast_waits(monkeypatch, bound=0.2)
            captured: dict = {}
            _mock_alignment(monkeypatch, tmp_path, captured)

            hooks = cli.RemoteHooks("http://x", WKEY, WID, job_id, await_line_meta=True)
            # 구버전 서버 응답: 새 필드가 없다
            monkeypatch.setattr(hooks, "_post", lambda path, b: {"cancel_requested": False})
            job_input = worker_core.JobInput(
                job_id=job_id, video_id=VIDEO, lyrics=LYRICS, await_line_meta=True
            )
            started = time.monotonic()
            result = await worker_core.run_pipeline(job_input, hooks)
            elapsed = time.monotonic() - started

            assert elapsed < 5.0  # 무한 대기가 아니다
            assert captured["line_meta"] is None  # 원문 정렬
            assert result is not None  # 그래도 결과는 나온다

    asyncio.run(body())


# ── 워커 쪽 흡수 (_absorb) 단위 계약 ──────────────────────────────


def test_absorb_stashes_line_meta_into_the_core_view():
    """하트비트 응답을 워커 프로세스의 코어 전역에 넣는다 — 코어 대기가 그것을 본다."""

    async def body():
        async with _env():
            hooks = cli.RemoteHooks("http://x", WKEY, WID, "job-1", await_line_meta=True)
            assert hooks._progress_body(48, "번역 대기")["want_line_meta"] is True

            hooks._absorb(
                {"cancel_requested": False, "line_meta": META, "attribution": {"source": "위키"}}
            )
            assert worker_core._PENDING_LINE_META["job-1"] == META
            assert worker_core._PENDING_ATTRIBUTION["job-1"] == {"source": "위키"}
            # 받은 뒤로는 요청을 내린다 (매 하트비트에 번역 전문을 되돌려 받지 않는다)
            assert "want_line_meta" not in hooks._progress_body(50, "전사 정렬")

    asyncio.run(body())


def test_absorb_keeps_waiting_while_line_meta_is_absent():
    """None은 흡수하지 않는다 — 스태시 키가 생기면 코어가 "확정"으로 오해한다."""

    async def body():
        async with _env():
            hooks = cli.RemoteHooks("http://x", WKEY, WID, "job-2", await_line_meta=True)
            hooks._absorb({"cancel_requested": False, "line_meta": None})
            assert "job-2" not in worker_core._PENDING_LINE_META
            assert hooks._progress_body(48, "번역 대기")["want_line_meta"] is True

    asyncio.run(body())


def test_absorb_takes_confirmed_empty_as_arrival():
    """빈 리스트는 도착이다 — 스태시에 그대로 넣어 코어가 즉시 원문 정렬로 가게 한다."""

    async def body():
        async with _env():
            hooks = cli.RemoteHooks("http://x", WKEY, WID, "job-3", await_line_meta=True)
            hooks._absorb({"cancel_requested": False, "line_meta": []})
            assert worker_core._PENDING_LINE_META["job-3"] == []
            assert worker_core._wait_for_line_meta("job-3", 30.0) is None

    asyncio.run(body())


def test_absorb_ignores_line_meta_for_ordinary_jobs():
    """예고가 없는 잡은 요청도 흡수도 하지 않는다 (플래그가 채널의 유일한 스위치)."""

    async def body():
        async with _env():
            hooks = cli.RemoteHooks("http://x", WKEY, WID, "job-4")
            assert "want_line_meta" not in hooks._progress_body(10, "다운로드")
            hooks._absorb({"cancel_requested": False, "line_meta": META})
            assert "job-4" not in worker_core._PENDING_LINE_META

    asyncio.run(body())


def test_absorb_marks_cancel_for_the_core_wait():
    """대기 중 취소는 하트비트로만 온다 — 코어 취소 집합에 넣어 대기를 끊게 한다."""

    async def body():
        async with _env():
            hooks = cli.RemoteHooks("http://x", WKEY, WID, "job-5", await_line_meta=True)
            hooks._absorb({"cancel_requested": True})
            assert "job-5" in worker_core._CANCEL_REQUESTED
            try:
                worker_core._wait_for_line_meta("job-5", 30.0)
            except worker_core.JobCancelled:
                pass
            else:
                raise AssertionError("대기가 취소로 끊기지 않았다")

    asyncio.run(body())


def test_job_boundary_clears_worker_process_state():
    """잡 경계에서 워커 프로세스의 잔여물을 지운다 (상주 워커에 쌓이지 않게)."""

    async def body():
        async with _env():
            worker_core.stash_line_meta("job-6", META)
            worker_core.stash_attribution("job-6", {"source": "위키"})
            worker_core.request_cancel("job-6")
            cli._clear_job_state("job-6")
            assert "job-6" not in worker_core._PENDING_LINE_META
            assert "job-6" not in worker_core._PENDING_ATTRIBUTION
            assert "job-6" not in worker_core._CANCEL_REQUESTED

    asyncio.run(body())


# ── 워커 루프 배선 (claim → JobInput/hooks) ────────────────────────


async def _run_loop_once(claim_resp) -> dict:
    """_worker_loop를 한 잡만 돌린다 — claim은 준 응답으로 대역, run_pipeline은 기록만.

    claim 응답의 await_line_meta가 JobInput과 hooks로 어떻게 배선되는지 본다."""
    from _pytest.monkeypatch import MonkeyPatch

    captured: dict = {}

    async def fake_run_pipeline(job_input, hooks):
        captured["job_input"] = job_input
        captured["hooks_want"] = hooks._want_line_meta
        return None  # 취소/캐시 완결과 같은 경로 — 제출 없음

    mp = MonkeyPatch()
    try:
        mp.setattr(cli, "_worker_claim", lambda base, key, wid: claim_resp.model_dump())
        mp.setattr(worker_core, "run_pipeline", fake_run_pipeline)
        mp.setattr(gpu_mem, "reclaim_after_job", lambda: None)
        await cli._worker_loop("http://x", WKEY, WID, poll=0.01, once=True)
    finally:
        mp.undo()
    return captured


def test_loop_arms_the_wait_for_pending_jobs():
    """예고 잡은 JobInput.await_line_meta=True + hooks 요청 플래그로 배선된다."""

    async def body():
        async with _env(max_job_audio_sec=0) as sm:
            job_id = await _seed_queued_job(sm)
            worker_core.stash_line_meta_wait(job_id)
            resp = await _claim()
            captured = await _run_loop_once(resp)
            assert captured["job_input"].job_id == job_id
            assert captured["job_input"].await_line_meta is True
            assert captured["hooks_want"] is True

    asyncio.run(body())


def test_loop_leaves_ordinary_jobs_untouched():
    """예고가 없으면 예전 그대로 — 대기도 요청도 없다 (회귀 0)."""

    async def body():
        async with _env(max_job_audio_sec=0) as sm:
            await _seed_queued_job(sm)
            resp = await _claim()
            captured = await _run_loop_once(resp)
            assert captured["job_input"].await_line_meta is False
            assert captured["hooks_want"] is False

    asyncio.run(body())


# ── ①③④ 실제 라우트와 맞물린 종단 경로 ───────────────────────────


async def _e2e(monkeypatch, tmp_path, *, bound: float, server_stash: dict, job_id: str):
    """claim → run_pipeline(하트비트 브리지) → 결과 제출까지 실제 코드로 돈다.

    서버 쪽 line_meta 시야만 server_stash로 분리했으므로, 코어가 번역을 보게 되는 유일한
    경로는 하트비트 응답이다."""
    _server_view(monkeypatch, server_stash)
    _fast_waits(monkeypatch, bound=bound)
    captured: dict = {}
    _mock_alignment(monkeypatch, tmp_path, captured)

    claim_resp = await _claim()
    job = claim_resp.job
    assert job is not None and job.job_id == job_id
    awaits = job.await_line_meta and job.line_meta is None

    calls: list = []
    hooks = cli.RemoteHooks("http://x", WKEY, WID, job_id, await_line_meta=awaits)
    monkeypatch.setattr(hooks, "_post", _bridge(asyncio.get_running_loop(), job_id, calls))
    job_input = worker_core.JobInput(
        job_id=job_id,
        video_id=job.video_id,
        lyrics=job.lyrics,
        line_meta=job.line_meta,
        attribution=job.attribution,
        await_line_meta=awaits,
    )
    captured["calls_log"] = calls
    captured["result"] = await worker_core.run_pipeline(job_input, hooks)
    return captured


def test_late_line_meta_reaches_alignment_over_the_heartbeat(monkeypatch, tmp_path):
    """① 클레임 뒤에 서버로 들어온 번역이 하트비트를 타고 정렬 입력이 된다.

    이것이 원격 워커 병렬 경로의 핵심이다 — 이 경로가 없으면 매 잡이 조용히 원문 정렬로
    떨어져 독음 정렬 품질을 잃는다."""

    async def body():
        async with _env() as sm:
            job_id = await _seed_queued_job(sm)
            worker_core.stash_line_meta_wait(job_id)
            server_stash: dict = {}

            async def attach_later():
                # 워커가 다운로드·보컬 분리를 도는 사이에 번역이 끝난 상황
                await asyncio.sleep(0.12)
                server_stash[job_id] = META

            captured, _ = await asyncio.gather(
                _e2e(monkeypatch, tmp_path, bound=5.0, server_stash=server_stash, job_id=job_id),
                attach_later(),
            )

            # 정렬이 늦게 온 번역을 실제로 받았다 (코어 전역에는 흡수 전까지 없었다)
            assert captured["line_meta"] == META
            # 결과에도 병합된다
            result = captured["result"]
            assert result is not None
            assert result.timestamps[0]["translation"] == "첫 줄"
            # 요청 플래그는 받을 때까지만 켜져 있었다
            wants = [b.get("want_line_meta", False) for p, b in captured["calls_log"]]
            assert wants[0] is True and wants[-1] is False
            # 대기 구간에도 하트비트가 계속 나갔다 — 이것이 서버 리스를 살려 둔다
            beats = [b for p, b in captured["calls_log"] if p.endswith("/progress")]
            waits = [b for b in beats if b["stage"] == worker_core.LINE_META_WAIT_STAGE]
            assert len(waits) >= 2, "대기 중 하트비트가 멈췄다 — 리스가 회수된다"

    asyncio.run(body())


def test_lease_survives_a_wait_longer_than_the_lease(monkeypatch, tmp_path):
    """⑤ 리스보다 긴 대기에도 만료 스윕이 잡을 회수하지 않는다 (하트비트가 계속 갱신한다).

    회수되면 잡이 queued로 되돌아가 다른 워커가 같은 곡을 다시 돌리고, 이 워커의 하트비트·
    결과 제출은 409로 거부된다 — 그래서 대기 중 하트비트는 이 경로의 필수 전제다.
    리스를 1초로 두고 그보다 긴 1.2초 뒤에 번역을 붙이면서, 그 사이 만료 스윕을 계속 돌린다."""

    async def body():
        async with _env(worker_lease_sec=1) as sm:
            job_id = await _seed_queued_job(sm)
            worker_core.stash_line_meta_wait(job_id)
            server_stash: dict = {}
            sweeping = True

            async def sweeper():
                # 서버가 claim마다 돌리는 그 만료 스윕 — 갱신이 끊기면 여기서 회수된다
                while sweeping:
                    await worker_api._sweep_expired_leases()
                    await asyncio.sleep(0.05)

            async def attach_after_lease_would_expire():
                await asyncio.sleep(1.2)  # 리스(1s)보다 긴 대기
                server_stash[job_id] = META

            sweep_task = asyncio.create_task(sweeper())
            try:
                captured, _ = await asyncio.gather(
                    _e2e(
                        monkeypatch, tmp_path, bound=6.0, server_stash=server_stash, job_id=job_id
                    ),
                    attach_after_lease_would_expire(),
                )
            finally:
                sweeping = False
                await sweep_task

            # 리스를 잃지 않았으므로 번역이 그대로 도착해 정렬 입력이 됐다
            assert captured["line_meta"] == META
            assert worker_api._LEASES[job_id][0] == WID
            async with sm() as s:
                assert (await JobRepository(s).get_by_id(job_id)).status == "processing"

    asyncio.run(body())


def test_bound_exceeded_falls_back_to_original_alignment(monkeypatch, tmp_path):
    """③ 아무도 붙여 주지 않으면 상한 뒤 원문 정렬로 완주한다 (잡이 걸리지 않는다)."""

    async def body():
        async with _env() as sm:
            job_id = await _seed_queued_job(sm)
            worker_core.stash_line_meta_wait(job_id)
            started = time.monotonic()
            captured = await _e2e(
                monkeypatch, tmp_path, bound=0.3, server_stash={}, job_id=job_id
            )
            elapsed = time.monotonic() - started

            assert 0.3 <= elapsed < 5.0  # 상한을 지켰다
            assert captured["line_meta"] is None
            assert captured["result"] is not None

    asyncio.run(body())


def test_confirmed_empty_over_heartbeat_releases_the_wait(monkeypatch, tmp_path):
    """② 대기 중에 번역 실패를 빈 배열로 알리면 상한을 소모하지 않고 즉시 원문 정렬로 간다.

    "아직 안 왔음"과 구분되지 않으면 이 잡은 아무도 보내지 않을 값을 상한까지 기다린다."""

    async def body():
        async with _env() as sm:
            job_id = await _seed_queued_job(sm)
            worker_core.stash_line_meta_wait(job_id)
            server_stash: dict = {}

            async def attach_empty():
                await asyncio.sleep(0.05)
                server_stash[job_id] = []

            started = time.monotonic()
            captured, _ = await asyncio.gather(
                _e2e(monkeypatch, tmp_path, bound=30.0, server_stash=server_stash, job_id=job_id),
                attach_empty(),
            )
            elapsed = time.monotonic() - started

            assert elapsed < 5.0  # 상한(30s)을 전혀 쓰지 않았다
            assert captured["line_meta"] is None
            assert captured["result"] is not None

    asyncio.run(body())


def test_cancel_during_the_wait_stops_before_alignment(monkeypatch, tmp_path):
    """④ 대기 중 취소하면 정렬을 태우지 않고 끝난다 (결과 제출 없음).

    취소 집합은 한 프로세스 테스트에서 서버·워커가 공유하므로, 취소가 **하트비트를 타고
    건너온다**는 것 자체는 test_absorb_marks_cancel_for_the_core_wait(워커 쪽 시야만 두고
    _absorb로만 채워지는 경로)와 test_cancel_response_does_not_leak_line_meta(응답이 실제로
    cancel_requested를 실어 보낸다)가 못 박는다. 여기서 보는 것은 종단 마감 동작이다."""

    async def body():
        async with _env() as sm:
            job_id = await _seed_queued_job(sm)
            worker_core.stash_line_meta_wait(job_id)
            server_stash: dict = {}

            async def cancel_later():
                await asyncio.sleep(0.05)
                # 취소 API가 하는 일: 취소 집합 등록 + 잡 failed 마킹
                worker_core.request_cancel(job_id)
                async with sm() as s:
                    await JobRepository(s).update_status(
                        job_id, "failed", error="요청으로 취소했어요"
                    )
                    await s.commit()

            started = time.monotonic()
            captured, _ = await asyncio.gather(
                _e2e(monkeypatch, tmp_path, bound=30.0, server_stash=server_stash, job_id=job_id),
                cancel_later(),
            )
            elapsed = time.monotonic() - started

            assert elapsed < 5.0  # 상한(30s)을 기다리지 않고 끊겼다
            assert captured.get("calls") is None  # 정렬은 아예 돌지 않았다
            assert captured["result"] is None  # 제출할 결과도 없다
            async with sm() as s:
                job = await JobRepository(s).get_by_id(job_id)
                assert job.status == "failed"
                assert job.error == "요청으로 취소했어요"
                assert (
                    await SyncRepository(s).get_by_video_and_hash(VIDEO, hash_lyrics(LYRICS))
                ) is None

    asyncio.run(body())
