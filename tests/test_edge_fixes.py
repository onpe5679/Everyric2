"""엣지케이스 검수 수리 회귀 테스트.

- S1: 교차 영상 (audio_hash, lyrics_hash) 재사용이 새 영상 몫의 싱크 행을 복사 생성
  (예전엔 completed인데 조회가 영원히 found=false, 초기화도 무효였다)
- S2: 오디오 길이 판독(_audio_duration_sec)
- 취소: cancel API + 워커 경계 소진(_consume_cancel)
- M1: 링크 배속(rate) 시간축 사상
- L1/L2/M2: video_id·tone·입력 길이 422
"""
import asyncio
import contextlib

import pytest
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from everyric2.server.db import connection as db_conn
from everyric2.server.db.models import Base

VID_A = "AAAAAAAAAA1"
VID_B = "BBBBBBBBBB1"


@contextlib.asynccontextmanager
async def _db():
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


class TestCrossVideoCacheCopy:
    def test_copy_creates_row_for_new_video(self, tmp_path):
        from everyric2.server.api.sync import get_sync
        from everyric2.server.db.repository import JobRepository, SyncRepository, hash_lyrics
        from everyric2.server.worker import _try_complete_from_cache

        lyrics = "라인1\n라인2"
        segs = [{"text": "라인1", "start": 1.0, "end": 2.0}]
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"x")

        async def body():
            async with _db() as sm:
                async with sm() as s:
                    await SyncRepository(s).create(
                        video_id=VID_A,
                        lyrics_hash=hash_lyrics(lyrics),
                        timestamps=segs,
                        audio_hash="hashA",
                        extra={"tempo": {"bpm": 100.0}},
                    )
                    job = await JobRepository(s).create(video_id=VID_B, lyrics=lyrics)
                    await s.commit()

                ok = await _try_complete_from_cache(
                    job.id, job, "hashA", hash_lyrics(lyrics), str(audio)
                )
                assert ok is True

                # 핵심: B 영상 조회가 이제 found=True (예전엔 영원히 false·복구 불가)
                resp_b = await get_sync(VID_B)
                assert resp_b.found is True
                assert resp_b.timestamps[0]["text"] == "라인1"
                assert resp_b.tempo == {"bpm": 100.0}
                # 원본(A)도 그대로 살아 있다
                resp_a = await get_sync(VID_A)
                assert resp_a.found is True
                # 잡은 B의 새 행을 result로 가리킨다
                async with sm() as s:
                    j = await JobRepository(s).get_by_id(job.id)
                    assert j.status == "completed"
                    assert j.result_id == resp_b.sync_id

        asyncio.run(body())

    def test_same_video_reuse_unchanged(self, tmp_path):
        from everyric2.server.db.repository import JobRepository, SyncRepository, hash_lyrics
        from everyric2.server.worker import _try_complete_from_cache

        lyrics = "라인1"
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"x")

        async def body():
            async with _db() as sm:
                async with sm() as s:
                    created = await SyncRepository(s).create(
                        video_id=VID_A,
                        lyrics_hash=hash_lyrics(lyrics),
                        timestamps=[{"text": "라인1", "start": 0.0, "end": 1.0}],
                        audio_hash="hashA",
                    )
                    sync_id = created.id
                    job = await JobRepository(s).create(video_id=VID_A, lyrics=lyrics)
                    await s.commit()
                ok = await _try_complete_from_cache(
                    job.id, job, "hashA", hash_lyrics(lyrics), str(audio)
                )
                assert ok is True
                async with sm() as s:
                    j = await JobRepository(s).get_by_id(job.id)
                    assert j.result_id == sync_id  # 같은 영상이면 기존 행 그대로 재사용
                    rows = await SyncRepository(s).get_by_video(VID_A)
                    assert len(rows) == 1  # 불필요한 복사 없음

        asyncio.run(body())

    def test_no_match_returns_false(self, tmp_path):
        from everyric2.server.db.repository import JobRepository
        from everyric2.server.worker import _try_complete_from_cache

        audio = tmp_path / "a.wav"
        audio.write_bytes(b"x")

        async def body():
            async with _db() as sm:
                async with sm() as s:
                    job = await JobRepository(s).create(video_id=VID_B, lyrics="가사")
                    await s.commit()
                assert await _try_complete_from_cache(job.id, job, "hx", "hy", str(audio)) is False
                assert audio.exists()  # 미스면 오디오는 정렬 단계가 쓰도록 남겨 둔다

        asyncio.run(body())


class TestCancel:
    def test_cancel_active_job_and_worker_boundary(self):
        from everyric2.server import worker
        from everyric2.server.api.job import cancel_job, get_job_status
        from everyric2.server.db.repository import JobRepository

        async def body():
            async with _db() as sm:
                async with sm() as s:
                    job = await JobRepository(s).create(video_id=VID_A, lyrics="가사")
                    await s.commit()
                res = await cancel_job(job.id)
                assert res["cancelled"] is True
                # 워커가 다음 단계 경계에서 소진하고 집합을 비운다
                assert await worker._consume_cancel(job.id) is True
                assert job.id not in worker._CANCEL_REQUESTED
                assert await worker._consume_cancel(job.id) is False  # 멱등
                status = await get_job_status(job.id)
                assert status.status == "failed"

        asyncio.run(body())

    def test_cancel_finished_job_is_noop(self):
        from everyric2.server.api.job import cancel_job
        from everyric2.server.db.repository import JobRepository

        async def body():
            async with _db() as sm:
                async with sm() as s:
                    job = await JobRepository(s).create(video_id=VID_A, lyrics="가사")
                    await s.commit()
                async with sm() as s:
                    await JobRepository(s).update_status(job.id, "completed", progress=100)
                    await s.commit()
                res = await cancel_job(job.id)
                assert res["cancelled"] is False
                assert res["status"] == "completed"

        asyncio.run(body())

    def test_invalid_job_id_422(self):
        from everyric2.server.api.job import cancel_job

        async def body():
            with pytest.raises(HTTPException) as e:
                await cancel_job("../etc/passwd")
            assert e.value.status_code == 422

        asyncio.run(body())


class TestRateShift:
    def test_rate_maps_time_axis(self):
        from everyric2.server.api.sync import _shift_sync_timestamps

        data = {
            "segments": [{"text": "x", "start": 10.0, "end": 20.0}],
            "tempo": {"bpm": 120.0, "beat_offset": 1.0},
            "debug": {"f0_curve": {"t0": 10.0, "dt": 0.1}},
        }
        out = _shift_sync_timestamps(data, offset=2.0, rate=2.0)
        assert out["segments"][0]["start"] == 7.0  # 10/2 + 2
        assert out["segments"][0]["end"] == 12.0
        assert out["tempo"]["bpm"] == 240.0  # 배속만큼 빨라진 곡
        assert out["tempo"]["beat_offset"] == 2.5
        assert out["debug"]["f0_curve"]["dt"] == 0.05

    def test_rate_one_is_pure_shift(self):
        from everyric2.server.api.sync import _shift_sync_timestamps

        data = {"segments": [{"text": "x", "start": 10.0, "end": 20.0}], "tempo": {"bpm": 120.0}}
        out = _shift_sync_timestamps(data, offset=2.0)
        assert out["segments"][0]["start"] == 12.0
        assert out["tempo"]["bpm"] == 120.0  # rate=1이면 BPM 불변


class TestDurationProbe:
    def test_reads_wav_duration(self, tmp_path):
        import numpy as np
        import soundfile as sf

        from everyric2.server.worker import _audio_duration_sec

        p = tmp_path / "t.wav"
        sf.write(p, np.zeros(16000, dtype="float32"), 16000)
        assert _audio_duration_sec(str(p)) == pytest.approx(1.0, abs=0.01)

    def test_bad_file_returns_none(self, tmp_path):
        from everyric2.server.worker import _audio_duration_sec

        p = tmp_path / "bad.wav"
        p.write_bytes(b"not a wav at all")
        assert _audio_duration_sec(str(p)) is None


class TestInputGuards:
    def test_translate_too_long_422(self):
        from everyric2.server.api.translate import TranslateRequest, translate_lyrics

        with pytest.raises(HTTPException) as e:
            translate_lyrics(TranslateRequest(text="가" * 15001))
        assert e.value.status_code == 422

    def test_translate_too_many_lines_422(self):
        from everyric2.server.api.translate import TranslateRequest, translate_lyrics

        with pytest.raises(HTTPException) as e:
            translate_lyrics(TranslateRequest(text="\n".join(["가"] * 401)))
        assert e.value.status_code == 422

    def test_translate_bad_tone_422(self):
        from everyric2.server.api.translate import TranslateRequest, translate_lyrics

        with pytest.raises(HTTPException) as e:
            translate_lyrics(TranslateRequest(text="こんにちは", tone="not_a_real_tone"))
        assert e.value.status_code == 422

    def test_get_sync_invalid_video_id_422(self):
        from everyric2.server.api.sync import get_sync

        async def body():
            for bad in ("x" * 5000, "한글아이디123", "short", "AAAAAAAAAA1 "):
                with pytest.raises(HTTPException) as e:
                    await get_sync(bad)
                assert e.value.status_code == 422

        asyncio.run(body())
