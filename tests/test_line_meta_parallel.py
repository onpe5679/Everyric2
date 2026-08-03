"""번역·독음(line_meta)을 다운로드·보컬 분리와 겹치는 병렬 경로 테스트.

여기서 못 박는 계약:
  ① line_meta 없이 잡을 만들 수 있고, 나중에 붙인 line_meta가 **정렬에 반영**된다.
  ② 상한 안에 안 오면 원문 정렬로 폴백한다 — 대기는 언제나 유한하다(무한 대기 없음).
  ③ line_meta를 본문에 실어 보내는 기존 경로는 그대로 동작한다 (대기 없음).
  ④ 대기 중에도 취소가 먹는다.
  ⑤ 원격 워커 배포(local_worker=False)는 큐 진입을 늦춰 조용한 원문 정렬 폴백을 막는다.

GPU는 절대 쓰지 않는다: 오디오 확보(_acquire_audio)와 정렬 본체(_run_alignment)만 목으로
갈아끼우고 그 사이의 코어(run_pipeline·리졸버·스태시·취소 경계)는 실제 코드다. DB는 기존
서버 테스트 규약대로 격리된 in-memory SQLite로 connection.async_session을 몽키패치한다.
"""

import asyncio
import contextlib
import time

import pytest
from fastapi import BackgroundTasks, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from everyric2 import gpu_mem
from everyric2.config.settings import get_settings
from everyric2.server import worker as worker_core
from everyric2.server.api.sync import (
    GenerateRequest,
    LineMeta,
    LineMetaAttachRequest,
    _queue_after_line_meta,
    attach_line_meta,
    generate_sync,
)
from everyric2.server.db import connection as db_conn
from everyric2.server.db.models import Base
from everyric2.server.db.repository import JobRepository, SyncRepository, hash_lyrics

VIDEO = "PARPARPAR01"
LYRICS = "一行目\n二行目"


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
    saved = {k: getattr(server, k) for k in server_overrides}
    for k, v in server_overrides.items():
        object.__setattr__(server, k, v)
    _clear_stashes()
    try:
        yield sm
    finally:
        db_conn.async_session = orig
        for k, v in saved.items():
            object.__setattr__(server, k, v)
        _clear_stashes()
        await engine.dispose()


def _clear_stashes() -> None:
    worker_core._PENDING_LINE_META.clear()
    worker_core._PENDING_ATTRIBUTION.clear()
    worker_core._PENDING_TITLE.clear()
    worker_core._PENDING_FORCE.clear()
    worker_core._PENDING_META_WAIT.clear()
    worker_core._CANCEL_REQUESTED.clear()


async def _create_job(sm, lyrics=LYRICS, video_id=VIDEO) -> str:
    async with sm() as s:
        job = await JobRepository(s).create(video_id=video_id, lyrics=lyrics)
        await s.commit()
        return job.id


def _fake_alignment(captured: dict, audio_file):
    """_run_alignment 대역 — 리졸버를 부르고 그 결과를 기록한 뒤 최소 결과를 돌려준다.

    실제 함수와 같은 계약을 지킨다: 단계를 보고하고, finally에서 오디오 파일을 지운다."""

    def fake(
        audio_path, lyrics, language, line_meta=None, on_stage=None, resolver=None,
        video_id=None, min_depth=None, on_depth=None,
    ):
        try:
            if on_stage is not None:
                on_stage("보컬 분리")
            if resolver is not None:
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

    return fake


def _mock_audio(monkeypatch, tmp_path):
    """다운로드와 GPU 정리 훅만 목으로 — torch를 끌어오지 않고 코어를 그대로 돌린다."""
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


@contextlib.contextmanager
def _mock_pipeline(monkeypatch, tmp_path, captured: dict):
    """다운로드·정렬·GPU 정리만 목으로 — 그 사이 코어는 실제 코드가 돈다."""
    audio_file = _mock_audio(monkeypatch, tmp_path)
    monkeypatch.setattr(worker_core, "_run_alignment", _fake_alignment(captured, audio_file))
    yield audio_file


# ── ① line_meta 없이 잡 생성 → 나중에 붙이면 정렬에 반영 ───────────


def test_generate_without_line_meta_creates_job_and_reports_wait():
    """가사 텍스트만으로 잡이 생기고, 서버가 기다려 줄 상한을 응답에 알린다."""

    async def body():
        async with _env(local_worker=True) as sm:
            resp = await generate_sync(
                GenerateRequest(
                    video_id=VIDEO, lyrics=LYRICS, line_meta_pending=True, title="熱異常"
                ),
                BackgroundTasks(),
            )
            assert resp.status == "processing"
            assert resp.line_meta_wait_sec == worker_core.LINE_META_WAIT_SEC
            # 잡은 만들어졌고 line_meta는 아직 없다 — 다운로드가 곧 시작될 수 있는 상태
            async with sm() as s:
                job = await JobRepository(s).get_by_id(resp.job_id)
                assert job is not None and job.lyrics == LYRICS
            assert resp.job_id not in worker_core._PENDING_LINE_META
            # 인프로세스 워커는 "정렬 진입 직전에 기다려라" 예고를 스태시로 받는다
            assert resp.job_id in worker_core._PENDING_META_WAIT
            # 제목처럼 번역과 무관한 메타는 그대로 함께 실려 온다
            assert worker_core.peek_title(resp.job_id) == ("熱異常", None)

    asyncio.run(body())


def test_late_line_meta_reaches_alignment(monkeypatch, tmp_path):
    """정렬 진입 직전에 도착한 line_meta가 정렬 입력으로 들어가고 결과에 병합된다."""

    async def body():
        async with _env(local_worker=True, media_cache_url="") as sm:
            job_id = await _create_job(sm)
            worker_core.stash_line_meta_wait(job_id)
            captured: dict = {}
            with _mock_pipeline(monkeypatch, tmp_path, captured):

                async def attach_later():
                    # 다운로드·보컬 분리가 진행되는 사이에 번역이 끝나는 상황
                    await asyncio.sleep(0.1)
                    return await attach_line_meta(
                        job_id,
                        LineMetaAttachRequest(
                            line_meta=[
                                LineMeta(
                                    text="一行目", pronunciation="이치교메", translation="첫 줄"
                                )
                            ]
                        ),
                    )

                _, attach_resp = await asyncio.gather(
                    worker_core.process_job(job_id), attach_later()
                )

            assert attach_resp.applied == "stashed"
            # 정렬이 실제로 늦게 온 line_meta를 받았다
            assert captured["line_meta"] == [
                {"text": "一行目", "pronunciation": "이치교메", "translation": "첫 줄"}
            ]
            # 잡은 완료되고 저장된 싱크에 번역이 병합돼 있다
            async with sm() as s:
                job = await JobRepository(s).get_by_id(job_id)
                assert job.status == "completed"
                sync = await SyncRepository(s).get_by_video_and_hash(VIDEO, hash_lyrics(LYRICS))
                segs = sync.timestamps["segments"]
                assert segs[0]["translation"] == "첫 줄"
                assert segs[0]["pronunciation"] == "이치교메"
            # 스태시는 완료와 함께 정리된다
            assert job_id not in worker_core._PENDING_LINE_META
            assert job_id not in worker_core._PENDING_META_WAIT

    asyncio.run(body())


def test_wait_stage_sits_between_separation_and_alignment():
    """대기 단계에 진행률 창이 등록돼 있어야 한다 — 없으면 _stage_monitor의 기본 창(36,88)이
    걸려 대기 중에 진행률이 88까지 치솟는다(사용자에게는 다 된 것처럼 보이는 가짜 진행)."""
    lo, hi = worker_core.STAGE_WINDOWS[worker_core.LINE_META_WAIT_STAGE]
    align_lo, _ = worker_core.STAGE_WINDOWS["전사 정렬"]
    sep_lo, sep_hi = worker_core.STAGE_WINDOWS["보컬 분리"]
    # 보컬 분리 창 안쪽 끝~전사 정렬 시작 사이 — 어느 방향으로도 진행률이 되돌아가지 않는다
    assert sep_lo < lo < hi <= align_lo <= sep_hi


def test_stage_monitor_holds_progress_inside_the_wait_window():
    """대기가 길어져도 진행률은 대기 창 상한을 넘지 않는다."""

    async def body():
        reported: list[tuple[int, str]] = []

        async def report(progress, stage):
            reported.append((progress, stage))

        holder = {"stage": worker_core.LINE_META_WAIT_STAGE}
        monitor = asyncio.create_task(
            worker_core._stage_monitor(report, holder, start=36, interval=0.01)
        )
        await asyncio.sleep(0.3)
        monitor.cancel()

        assert reported, "모니터가 대기 단계를 한 번도 보고하지 않았다"
        assert {stage for _, stage in reported} == {worker_core.LINE_META_WAIT_STAGE}
        hi = worker_core.STAGE_WINDOWS[worker_core.LINE_META_WAIT_STAGE][1]
        assert max(progress for progress, _ in reported) <= hi

    asyncio.run(body())


def test_empty_attach_releases_wait_immediately(monkeypatch, tmp_path):
    """번역 실패를 빈 배열로 알리면 상한을 기다리지 않고 즉시 원문 정렬로 진행한다."""

    async def body():
        async with _env(local_worker=True, media_cache_url="") as sm:
            job_id = await _create_job(sm)
            worker_core.stash_line_meta_wait(job_id)
            monkeypatch.setattr(worker_core, "LINE_META_WAIT_SEC", 30.0)
            captured: dict = {}
            with _mock_pipeline(monkeypatch, tmp_path, captured):

                async def attach_empty():
                    await asyncio.sleep(0.05)
                    return await attach_line_meta(job_id, LineMetaAttachRequest(line_meta=[]))

                started = time.monotonic()
                await asyncio.gather(worker_core.process_job(job_id), attach_empty())
                elapsed = time.monotonic() - started

            # 상한(30s)을 전혀 소모하지 않았다
            assert elapsed < 5.0
            assert captured["line_meta"] is None
            async with sm() as s:
                job = await JobRepository(s).get_by_id(job_id)
                assert job.status == "completed"

    asyncio.run(body())


# ── ② 상한 초과 → 원문 정렬 폴백 (무한 대기 없음) ──────────────────


def test_wait_is_bounded_and_falls_back_to_original(monkeypatch, tmp_path):
    """아무도 붙여 주지 않아도 상한 안에 원문 정렬로 완주한다 — 잡이 걸리지 않는다."""

    async def body():
        async with _env(local_worker=True, media_cache_url="") as sm:
            job_id = await _create_job(sm)
            worker_core.stash_line_meta_wait(job_id)
            monkeypatch.setattr(worker_core, "LINE_META_WAIT_SEC", 0.3)
            captured: dict = {}
            with _mock_pipeline(monkeypatch, tmp_path, captured):
                started = time.monotonic()
                await worker_core.process_job(job_id)
                elapsed = time.monotonic() - started

            # 상한을 지켰다 (넉넉한 여유를 두고도 무한 대기가 아님을 증명)
            assert elapsed < 5.0
            # 대기 상한을 실제로 소모했고, 정렬은 line_meta 없이(원문) 돌았다
            assert elapsed >= 0.3
            assert captured["line_meta"] is None
            assert captured["calls"] == 1
            async with sm() as s:
                job = await JobRepository(s).get_by_id(job_id)
                assert job.status == "completed"
                sync = await SyncRepository(s).get_by_video_and_hash(VIDEO, hash_lyrics(LYRICS))
                assert "translation" not in sync.timestamps["segments"][0]

    asyncio.run(body())


def test_line_meta_arriving_after_the_bound_still_merges(monkeypatch, tmp_path):
    """상한을 넘겨 원문으로 정렬한 뒤 도착한 line_meta도 표시용으로는 결과에 병합된다."""

    async def body():
        async with _env(local_worker=True, media_cache_url="") as sm:
            job_id = await _create_job(sm)
            worker_core.stash_line_meta_wait(job_id)
            monkeypatch.setattr(worker_core, "LINE_META_WAIT_SEC", 0.1)
            captured: dict = {}
            audio_file = _mock_audio(monkeypatch, tmp_path)
            base = _fake_alignment(captured, audio_file)

            def align_then_late_arrival(*args, **kwargs):
                out = base(*args, **kwargs)
                # 상한을 넘겨 원문으로 정렬한 뒤에야 번역이 도착한 상황을 재현
                worker_core.stash_line_meta(
                    job_id, [{"text": "二行目", "translation": "둘째 줄"}]
                )
                return out

            monkeypatch.setattr(worker_core, "_run_alignment", align_then_late_arrival)
            await worker_core.process_job(job_id)

            # 정렬 자체는 line_meta 없이 원문으로 돌았다
            assert captured["line_meta"] is None
            async with sm() as s:
                sync = await SyncRepository(s).get_by_video_and_hash(VIDEO, hash_lyrics(LYRICS))
                assert sync.timestamps["segments"][1]["translation"] == "둘째 줄"

    asyncio.run(body())


# ── ③ 기존 경로(본문 동봉) 회귀 없음 ──────────────────────────────


def test_body_line_meta_path_is_unchanged(monkeypatch, tmp_path):
    """line_meta를 본문에 실어 보내는 구버전 호출자는 대기 없이 그대로 동작한다."""

    async def body():
        async with _env(local_worker=True, media_cache_url="") as sm:
            resp = await generate_sync(
                GenerateRequest(
                    video_id=VIDEO,
                    lyrics=LYRICS,
                    line_meta=[LineMeta(text="一行目", translation="첫 줄")],
                ),
                BackgroundTasks(),
            )
            # 기다릴 것이 없으므로 상한은 0이고 대기 예고 스태시도 없다
            assert resp.line_meta_wait_sec == 0.0
            assert resp.job_id not in worker_core._PENDING_META_WAIT
            assert worker_core._PENDING_LINE_META[resp.job_id] == [
                {"text": "一行目", "pronunciation": None, "translation": "첫 줄"}
            ]

            captured: dict = {}
            with _mock_pipeline(monkeypatch, tmp_path, captured):
                started = time.monotonic()
                await worker_core.process_job(resp.job_id)
                elapsed = time.monotonic() - started

            # 리졸버가 아예 만들어지지 않아 폴링 한 번도 없다
            assert elapsed < 2.0
            assert captured["line_meta"] == [
                {"text": "一行目", "pronunciation": None, "translation": "첫 줄"}
            ]

    asyncio.run(body())


def test_pending_flag_ignored_when_line_meta_is_in_the_body():
    """플래그와 값이 함께 오면 값이 이긴다 — 이미 가진 것을 기다리지 않는다."""

    async def body():
        async with _env(local_worker=True):
            resp = await generate_sync(
                GenerateRequest(
                    video_id=VIDEO,
                    lyrics=LYRICS,
                    line_meta_pending=True,
                    line_meta=[LineMeta(text="一行目", translation="첫 줄")],
                ),
                BackgroundTasks(),
            )
            assert resp.line_meta_wait_sec == 0.0
            assert resp.job_id not in worker_core._PENDING_META_WAIT

    asyncio.run(body())


# ── ④ 대기 중 취소 ────────────────────────────────────────────────


def test_cancel_during_wait_stops_the_job(monkeypatch, tmp_path):
    """대기 중 취소하면 정렬을 태우지 않고 즉시 끝난다 (상한을 기다리지 않는다)."""

    async def body():
        async with _env(local_worker=True, media_cache_url="") as sm:
            job_id = await _create_job(sm)
            worker_core.stash_line_meta_wait(job_id)
            monkeypatch.setattr(worker_core, "LINE_META_WAIT_SEC", 30.0)
            captured: dict = {}
            with _mock_pipeline(monkeypatch, tmp_path, captured):

                async def cancel_later():
                    # 취소 API가 하는 일: 취소 집합 등록 + 잡 failed 마킹
                    await asyncio.sleep(0.15)
                    worker_core.request_cancel(job_id)
                    async with sm() as s:
                        await JobRepository(s).update_status(
                            job_id, "failed", error="요청으로 취소했어요"
                        )
                        await s.commit()

                started = time.monotonic()
                await asyncio.gather(worker_core.process_job(job_id), cancel_later())
                elapsed = time.monotonic() - started

            # 상한(30s)을 기다리지 않고 끊겼다
            assert elapsed < 5.0
            # 정렬은 아예 실행되지 않았다 (리졸버 단계에서 중단)
            assert captured.get("calls") is None
            async with sm() as s:
                job = await JobRepository(s).get_by_id(job_id)
                assert job.status == "failed"
                assert job.error == "요청으로 취소했어요"
                assert (
                    await SyncRepository(s).get_by_video_and_hash(VIDEO, hash_lyrics(LYRICS))
                ) is None
            # 취소는 소진되고 스태시도 남지 않는다
            assert job_id not in worker_core._CANCEL_REQUESTED
            assert job_id not in worker_core._PENDING_META_WAIT

    asyncio.run(body())


# ── ⑤ 원격 워커 배포: 큐 진입을 늦춘다 ────────────────────────────


def test_remote_deployment_delays_queueing_until_line_meta():
    """GPU 없는 API 전용 서버는 line_meta가 올 때까지 큐에 올리지 않는다."""

    async def body():
        async with _env(local_worker=False) as sm:
            job_id = await _create_job(sm)
            waiter = asyncio.create_task(_queue_after_line_meta(job_id))
            # 대기 중에는 queued가 아니라 processing + "번역 대기" — 워커가 물어 가지 않는다
            for _ in range(40):
                await asyncio.sleep(0.02)
                async with sm() as s:
                    job = await JobRepository(s).get_by_id(job_id)
                if job.stage == worker_core.LINE_META_WAIT_STAGE:
                    break
            assert job.status == "processing"
            assert job.stage == worker_core.LINE_META_WAIT_STAGE
            async with sm() as s:
                assert await JobRepository(s).get_oldest_queued() is None

            worker_core.stash_line_meta(job_id, [{"text": "一行目", "translation": "첫 줄"}])
            await waiter
            async with sm() as s:
                job = await JobRepository(s).get_by_id(job_id)
                assert job.status == "queued"
                # 이제 원격 워커가 클레임할 수 있고, 스태시가 클레임 페이로드로 실려 간다
                assert (await JobRepository(s).get_oldest_queued()).id == job_id

    asyncio.run(body())


def test_remote_deployment_wait_is_bounded(monkeypatch):
    """아무도 붙여 주지 않아도 상한 뒤에 큐로 올라간다 (원문 정렬로 완주)."""

    async def body():
        async with _env(local_worker=False) as sm:
            job_id = await _create_job(sm)
            monkeypatch.setattr(worker_core, "LINE_META_WAIT_SEC", 0.3)
            started = time.monotonic()
            await _queue_after_line_meta(job_id)
            elapsed = time.monotonic() - started
            assert 0.3 <= elapsed < 5.0
            async with sm() as s:
                job = await JobRepository(s).get_by_id(job_id)
                assert job.status == "queued"

    asyncio.run(body())


def test_remote_deployment_cancel_during_wait_does_not_queue(monkeypatch):
    """대기 중 취소된 잡은 큐에 올리지 않는다."""

    async def body():
        async with _env(local_worker=False) as sm:
            job_id = await _create_job(sm)
            monkeypatch.setattr(worker_core, "LINE_META_WAIT_SEC", 30.0)
            waiter = asyncio.create_task(_queue_after_line_meta(job_id))
            await asyncio.sleep(0.1)
            worker_core.request_cancel(job_id)
            async with sm() as s:
                await JobRepository(s).update_status(job_id, "failed", error="요청으로 취소했어요")
                await s.commit()
            started = time.monotonic()
            await waiter
            assert time.monotonic() - started < 5.0
            async with sm() as s:
                job = await JobRepository(s).get_by_id(job_id)
                assert job.status == "failed"
                assert await JobRepository(s).get_oldest_queued() is None

    asyncio.run(body())


def test_dispatch_without_pending_flag_queues_immediately():
    """플래그가 없으면 원격 배포의 큐 진입은 예전 그대로 즉시다."""

    async def body():
        async with _env(local_worker=False) as sm:
            resp = await generate_sync(
                GenerateRequest(video_id=VIDEO, lyrics=LYRICS), BackgroundTasks()
            )
            async with sm() as s:
                job = await JobRepository(s).get_by_id(resp.job_id)
                assert job.status == "queued"

    asyncio.run(body())


# ── 붙이기 엔드포인트 자체의 계약 ─────────────────────────────────


def test_attach_merges_into_an_already_completed_sync():
    """캐시 히트로 잡이 먼저 끝났으면 완성된 싱크에 직접 병합한다 (번역이 버려지지 않는다)."""

    async def body():
        async with _env(local_worker=True) as sm:
            job_id = await _create_job(sm)
            async with sm() as s:
                sync = await SyncRepository(s).create(
                    video_id=VIDEO,
                    lyrics_hash=hash_lyrics(LYRICS),
                    timestamps=[
                        {"text": "一行目", "start": 0.0, "end": 1.0},
                        {"text": "二行目", "start": 1.0, "end": 2.0},
                    ],
                    engine="ctc",
                )
                await JobRepository(s).update_status(
                    job_id, "completed", progress=100, result_id=sync.id
                )
                await s.commit()

            resp = await attach_line_meta(
                job_id,
                LineMetaAttachRequest(
                    line_meta=[
                        LineMeta(text="一行目", translation="첫 줄"),
                        LineMeta(text="二行目", translation="둘째 줄"),
                    ],
                    title="熱異常",
                ),
            )
            assert (resp.applied, resp.merged_segments) == ("merged", 2)
            async with sm() as s:
                saved = await SyncRepository(s).get_by_video_and_hash(VIDEO, hash_lyrics(LYRICS))
                segs = saved.timestamps["segments"]
                assert [seg["translation"] for seg in segs] == ["첫 줄", "둘째 줄"]
                assert saved.title == "熱異常"
            # 끝난 잡에는 스태시를 남기지 않는다 (정리 지점이 없어 새므로)
            assert job_id not in worker_core._PENDING_LINE_META

    asyncio.run(body())


def test_attach_to_cancelled_job_is_dropped():
    async def body():
        async with _env(local_worker=True) as sm:
            job_id = await _create_job(sm)
            async with sm() as s:
                await JobRepository(s).update_status(
                    job_id, "failed", error="요청으로 취소했어요"
                )
                await s.commit()
            resp = await attach_line_meta(
                job_id, LineMetaAttachRequest(line_meta=[LineMeta(text="一行目")])
            )
            assert resp.applied == "dropped"
            assert job_id not in worker_core._PENDING_LINE_META

    asyncio.run(body())


def test_attach_unknown_job_is_404():
    async def body():
        async with _env(local_worker=True):
            with pytest.raises(HTTPException) as e:
                await attach_line_meta(
                    "00000000-0000-0000-0000-000000000000", LineMetaAttachRequest()
                )
            assert e.value.status_code == 404

    asyncio.run(body())


def test_attach_rejects_malformed_job_id():
    async def body():
        async with _env(local_worker=True):
            with pytest.raises(HTTPException) as e:
                await attach_line_meta("not-a-job-id", LineMetaAttachRequest())
            assert e.value.status_code == 422

    asyncio.run(body())
