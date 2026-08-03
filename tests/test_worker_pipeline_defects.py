"""워커 파이프라인 결함 회귀 테스트.

전수조사에서 코드 근거와 함께 확인된 결함들을 못박는다. 각 테스트는 "고치기 전이라면
반드시 실패하는가"를 기준으로 썼다 — 그러지 못하는 항목은 계약 고정으로만 남겼다.

① 전부 OOV인 곡이 "성공 + 무경고"로 저장되던 것. 정렬 글자가 0개인 줄은 ctc_engine이
   보간으로 채우고 전 줄이 실패하면 전체 구간 균등 분배가 되는데, 그 줄들은
   word_segments=None이라 conf가 없어 quality_score=None이 됐고 확장의 저신뢰 경고는
   `qualityScore != null && qualityScore < 0.001`(content.ts:649,1197)을 요구하므로 아예
   발화하지 않았다. 서버가 사실을 실어 보내야 한다.
② 나중에 도착한 빈 line_meta가 앞서 붙은 진짜 메타를 지우던 것.
③ 모든 다운로드 실패가 `Download failed: {e}` 하나로 접혀 영문 원문이 사용자에게 노출되던 것.
④ audio_hash 계약 고정 — 경로 의존은 값싼 내용 해시로 고칠 수 없어 유지한다
   (근거는 worker._acquire_audio 독스트링의 실측).
⑤ 교차 영상 캐시 복사 로그가 정작 복사가 없는 분기에 붙어 있던 것.
⑥ m4a 경로에서 과길이 검사가 통째로 생략되던 것 (libsndfile이 m4a를 못 읽는다).
"""

import contextlib
import logging
import shutil
import subprocess
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from everyric2.server.db import connection as db_conn
from everyric2.server.db.models import Base

VID_A = "AAAAAAAAAA2"
VID_B = "BBBBBBBBBB2"

# 확장이 저신뢰 경고를 띄우는 임계 (everyric2-chrome/src/content.ts:649, 1197).
# 서버가 내려보낸 quality_score가 이 조건에 걸려야 사용자가 경고를 본다.
EXT_LOW_QUALITY_THRESHOLD = 0.001


def _warns_in_extension(quality_score) -> bool:
    """확장의 경고 조건을 그대로 재현: `qualityScore != null && qualityScore < 0.001`."""
    return quality_score is not None and quality_score < EXT_LOW_QUALITY_THRESHOLD


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


# ── ① 정렬 실패가 quality_score에 실린다 ────────────────────────────


class TestAlignmentCoverageReachesQualityScore:
    def test_no_aligned_line_reports_zero_not_none(self):
        """정렬된 줄이 0개면 quality_score가 None이 아니라 확정 저신뢰여야 한다.

        None이면 확장 경고가 조용히 죽는다 — 이 프로젝트에서 가장 해로운 실패(안 된 것을
        됐다고 말하는 것)의 마지막 고리다.
        """
        from everyric2.server.worker import _quality_with_coverage

        score, meta = _quality_with_coverage(None, aligned_lines=0, total_lines=40)
        assert score is not None
        assert _warns_in_extension(score)
        assert meta["failed"] is True
        assert (meta["aligned_lines"], meta["total_lines"]) == (0, 40)
        assert meta["ratio"] == 0.0

    def test_minority_coverage_reports_low_and_keeps_the_measured_value(self):
        """2/40줄만 정렬돼도 그 2줄 평균이 곡 점수로 올라가던 것 — 이제 경고에 걸린다."""
        from everyric2.server.worker import _quality_with_coverage

        score, meta = _quality_with_coverage(0.05, aligned_lines=2, total_lines=40)
        assert _warns_in_extension(score)
        # 측정값을 버리지 않는다 — 근거가 debug에 남아야 되짚을 수 있다
        assert meta["measured_conf"] == 0.05
        assert meta["ratio"] == 0.05

    def test_healthy_song_value_is_untouched(self):
        """정상 곡(전 줄 정렬)의 저장값은 한 치도 바뀌지 않는다 — 오폭 방지."""
        from everyric2.server.worker import _quality_with_coverage

        score, meta = _quality_with_coverage(0.0492, aligned_lines=30, total_lines=30)
        assert score == 0.0492
        assert not _warns_in_extension(score)
        assert "failed" not in meta
        assert meta["ratio"] == 1.0

    def test_majority_coverage_is_kept(self):
        """과반이 정렬됐으면 측정값 유지 — 판정은 '과반이 보간 산물인가'만 본다."""
        from everyric2.server.worker import ALIGNED_LINE_RATIO_MIN, _quality_with_coverage

        assert ALIGNED_LINE_RATIO_MIN == 0.5
        score, meta = _quality_with_coverage(0.03, aligned_lines=20, total_lines=40)
        assert score == 0.03
        assert "failed" not in meta

    def test_all_oov_song_end_to_end_through_run_alignment(self, monkeypatch, tmp_path):
        """전부 OOV인 곡을 _run_alignment로 실제로 통과시켜 배선을 못박는다.

        고치기 전에는 이 곡의 quality_score가 None이라 확장이 경고를 못 띄웠다.
        """
        result = _run_alignment_with_unaligned_engine(monkeypatch, tmp_path, 12)

        assert len(result["timestamps"]) == 12
        # 정렬된 글자가 없으니 세그먼트에 conf도 words도 없다 (= 균등 보간 산물)
        assert all("confidence" not in s for s in result["timestamps"])
        assert all("words" not in s for s in result["timestamps"])
        # 고치기 전 값(=측정 conf)이 None이었음을 함께 못박는다 — 그것이 확장 경고가
        # 조용히 죽던 원인이다 (실측: 이 곡의 avg_confidence는 None이었다)
        coverage = result["debug"]["align_coverage"]
        assert coverage["measured_conf"] is None
        assert not _warns_in_extension(coverage["measured_conf"])
        # 그런데도 결과는 저신뢰로 확정돼 확장 경고에 걸린다
        assert _warns_in_extension(result["quality_score"])
        assert coverage["aligned_lines"] == 0
        assert coverage["total_lines"] == 12
        assert coverage["failed"] is True

    def test_fully_aligned_song_end_to_end_reports_full_coverage(self, monkeypatch, tmp_path):
        """정상 곡은 커버리지 1.0에 측정 conf 그대로 — 회귀 대칭 확인."""
        result = _run_alignment_with_unaligned_engine(
            monkeypatch, tmp_path, 12, conf=0.05, aligned=True
        )
        coverage = result["debug"]["align_coverage"]
        assert coverage["ratio"] == 1.0
        assert "failed" not in coverage
        assert result["quality_score"] == pytest.approx(0.05, abs=1e-6)
        assert not _warns_in_extension(result["quality_score"])


class _UnalignedEngine:
    """CTC 엔진 대역 — 전부 OOV인 곡(정렬 글자 0개)을 재현한다.

    ctc_engine은 정렬된 글자가 0개인 줄을 [None, None, None]으로 남기고 이웃 사이로
    보간하며, 전 줄이 실패하면 전체 구간에 균등 분배한다. 그 결과 라인 타이밍은 채워지지만
    ``confidence``와 ``word_segments``는 없다 — 그 상태를 그대로 만든다.
    """

    def __init__(self, adapter: str = "jpn", conf: float | None = None, aligned: bool = False):
        self._current_adapter = adapter
        self._current_lang = "ja"
        self._last_star_spans: list = []
        self._conf = conf
        self._aligned = aligned

    def is_available(self) -> bool:
        return True

    def align(self, audio, lyrics, language=None, progress_callback=None):
        from everyric2.inference.prompt import SyncResult, WordSegment

        out = []
        total = max(1, len(lyrics))
        span = 12.0 / total  # 전 구간 균등 분배(보간 결과와 같은 모양)
        for k, ln in enumerate(lyrics):
            words = None
            if self._aligned:
                chars = [c for c in ln.text if not c.isspace()] or ["x"]
                step = span / len(chars)
                words = [
                    WordSegment(
                        word=c,
                        start=k * span + j * step,
                        end=k * span + (j + 1) * step,
                        confidence=self._conf,
                    )
                    for j, c in enumerate(chars)
                ]
            out.append(
                SyncResult(
                    text=ln.text,
                    start_time=k * span,
                    end_time=(k + 1) * span,
                    confidence=self._conf if self._aligned else None,
                    line_number=ln.line_number,
                    word_segments=words,
                )
            )
        return out


def _run_alignment_with_unaligned_engine(
    monkeypatch, tmp_path, lines: int, conf: float | None = None, aligned: bool = False
):
    """_run_alignment을 실제로 돌린다 — 오디오 로드/보컬 분리/멜로디/템포만 스텁."""
    import numpy as np

    from everyric2.alignment import ctc_engine as ctc_mod
    from everyric2.audio import loader as loader_mod
    from everyric2.config.settings import get_settings
    from everyric2.server import worker as worker_mod

    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"fake")
    fake_audio = loader_mod.AudioData(
        waveform=np.zeros(16000, dtype="float32"), sample_rate=16000, duration=1.0
    )

    class _FakeLoader:
        def load(self, path):
            return fake_audio

    engine = _UnalignedEngine(conf=conf, aligned=aligned)
    monkeypatch.setattr(loader_mod, "AudioLoader", _FakeLoader)
    monkeypatch.setattr(ctc_mod, "get_shared_ctc_engine", lambda _s: engine)
    monkeypatch.setattr(worker_mod, "_separate_vocals", lambda _a: None)
    monkeypatch.setattr(worker_mod, "_estimate_tempo", lambda _a: None)

    settings = get_settings()
    saved_melody = settings.melody.enabled
    # 이 헬퍼는 레거시 CTC 엔진을 직접 대역(_UnalignedEngine)한다(위 get_shared_ctc_engine
    # 대역) — 새 스택(기본값 owsm/omniasr)은 이 경로가 없으므로 레거시로 강제 고정한다.
    saved_engine = settings.alignment.engine
    object.__setattr__(settings.melody, "enabled", False)
    object.__setattr__(settings.alignment, "engine", "ctc")
    try:
        lyrics = "\n".join(f"揺らめく光の中で{i}" for i in range(lines))
        return worker_mod._run_alignment(str(audio_file), lyrics, "ja")
    finally:
        object.__setattr__(settings.melody, "enabled", saved_melody)
        object.__setattr__(settings.alignment, "engine", saved_engine)


# ── ② 나중 빈 line_meta가 앞의 진짜 메타를 지우지 않는다 ─────────────


class TestStashLineMetaIsMonotonic:
    @pytest.fixture(autouse=True)
    def _clean_stash(self):
        from everyric2.server import worker as worker_mod

        worker_mod._PENDING_LINE_META.pop("job-1", None)
        yield
        worker_mod._PENDING_LINE_META.pop("job-1", None)

    def test_late_empty_resend_does_not_wipe_attached_meta(self):
        """재현: pending으로 생성 → 35줄 attach → 클라이언트 재시도가 []를 재전송."""
        from everyric2.server import worker as worker_mod

        meta = [{"text": f"行{i}", "pronunciation": f"gyo{i}"} for i in range(35)]
        worker_mod.stash_line_meta("job-1", meta)
        worker_mod.stash_line_meta("job-1", [])

        assert worker_mod._PENDING_LINE_META["job-1"] == meta

    def test_empty_then_real_meta_still_replaces(self):
        """반대 방향은 막지 않는다 — 빈 확정 신호 뒤에 진짜 메타가 오면 채택한다."""
        from everyric2.server import worker as worker_mod

        worker_mod.stash_line_meta("job-1", [])
        meta = [{"text": "行", "translation": "행"}]
        worker_mod.stash_line_meta("job-1", meta)
        assert worker_mod._PENDING_LINE_META["job-1"] == meta

    def test_real_meta_can_be_replaced_by_other_real_meta(self):
        from everyric2.server import worker as worker_mod

        worker_mod.stash_line_meta("job-1", [{"text": "a", "translation": "1"}])
        second = [{"text": "b", "translation": "2"}]
        worker_mod.stash_line_meta("job-1", second)
        assert worker_mod._PENDING_LINE_META["job-1"] == second

    def test_empty_arrival_contract_is_preserved(self):
        """빈 리스트는 여전히 "붙일 것 없음 확정" 신호다 — 대기가 즉시 끝나야 한다.

        이 계약을 깨면(빈 배열을 거부하면) 번역 실패 잡이 상한까지 헛되게 기다린다.
        """
        from everyric2.server import worker as worker_mod

        worker_mod.stash_line_meta("job-1", [])
        assert "job-1" in worker_mod._PENDING_LINE_META
        # 상한을 크게 줘도 즉시 None(원문 정렬 폴백)으로 돌아온다
        assert worker_mod._wait_for_line_meta("job-1", 30.0) is None


# ── ③ 다운로드 실패 분류 ────────────────────────────────────────────


class TestDownloadErrorClassification:
    @pytest.mark.parametrize(
        "text,code,terminal",
        [
            # 서버 구성 문제 (yt-dlp _video.py:2985) — 사용자가 할 수 있는 게 없다
            (
                "ERROR: [youtube] abc: No supported JavaScript runtime could be found. "
                "Only deno is enabled by default;",
                "js_runtime",
                True,
            ),
            # 나이 제한 (yt-dlp AGE_GATE_REASONS, _video.py:2894)
            ("ERROR: [youtube] abc: Sign in to confirm your age", "age_restricted", True),
            ("ERROR: [youtube] abc: This video is age-restricted", "age_restricted", True),
            # 봇 확인은 IP 평판 게이트라 login_required와 갈라져 있고 재시도 대상이다
            # (문구에 로그인 힌트가 함께 붙어 나오므로 순서로 갈린다)
            (
                "ERROR: [youtube] abc: Sign in to confirm you're not a bot. Use --cookies",
                "bot_check",
                False,
            ),
            (
                "ERROR: [youtube] abc: This video is only available for registered users",
                "login_required",
                True,
            ),
            (
                "ERROR: [youtube] abc: Login details are needed to download this content.",
                "login_required",
                True,
            ),
            # 429도 403과 같은 계열 (이 IP의 요청 속도를 막은 것)
            ("ERROR: [youtube] abc: HTTP Error 429: Too Many Requests", "throttled", False),
            # 지역 차단 (common.py:1257, _video.py:4043)
            (
                "ERROR: [youtube] abc: The uploader has not made this video "
                "available in your country",
                "geo_blocked",
                True,
            ),
            (
                "ERROR: This video is not available from your location due to geo restriction",
                "geo_blocked",
                True,
            ),
            # "unavailable"이 함께 들어 있어도 지역 차단이 먼저 판정돼야 한다 — 순서가 판정이다
            (
                "ERROR: [youtube] abc: Video unavailable. The uploader has blocked it "
                "in your country",
                "geo_blocked",
                True,
            ),
            # 영구 실패 (예전 판정과 동일한 부분일치 유지)
            ("ERROR: [youtube] abc: Private video. Sign in if you've been granted access",
             "unavailable", True),
            ("ERROR: [youtube] abc: Video unavailable", "unavailable", True),
            # 일시 실패 — 재시도 가치가 있다
            ("ERROR: unable to download video data: HTTP Error 403: Forbidden",
             "throttled", False),
            ("ERROR: [youtube] abc: Unable to download webpage: The read operation timed out",
             "network", False),
        ],
    )
    def test_each_failure_gets_its_own_korean_message(self, text, code, terminal):
        from everyric2.audio.downloader import classify_download_error

        failure = classify_download_error(text)
        assert failure is not None, text
        assert failure.code == code
        assert failure.terminal is terminal
        # 사용자에게 보일 문구는 한국어여야 한다 (영문 원문 노출이 이 결함의 증상이었다)
        assert any("가" <= ch <= "힣" for ch in failure.message)

    def test_actionable_failures_say_what_to_do(self):
        """(a) 사용자가 할 수 있는 일이 있는 실패는 그 조치를 말한다."""
        from everyric2.audio.downloader import classify_download_error

        for text in (
            "Sign in to confirm your age",
            "Sign in to confirm you're not a bot",
        ):
            failure = classify_download_error(text)
            assert "쿠키" in failure.message

    def test_server_side_failure_says_the_user_can_do_nothing(self):
        """(d) 서버 구성 문제는 사용자가 헛되게 재시도하지 않게 그렇다고 말한다."""
        from everyric2.audio.downloader import classify_download_error

        failure = classify_download_error("No supported JavaScript runtime could be found")
        assert "서버" in failure.message
        assert "조치는 없어요" in failure.message

    def test_unclassified_failure_falls_back_to_the_raw_text(self):
        """못 가리는 실패는 분류하지 않는다 — 엉뚱한 한국어 안내가 원문보다 해롭다."""
        from everyric2.audio.downloader import _classified_error, classify_download_error

        raw = "ERROR: [youtube] abc: some brand new failure mode nobody has seen"
        assert classify_download_error(raw) is None
        err = _classified_error(RuntimeError(raw), "https://y/watch?v=abc", "Download failed")
        assert err.code == "unknown"
        assert raw in str(err)

    def test_geo_restricted_exception_is_detected_by_type(self):
        """문자열보다 강한 근거 — yt-dlp의 GeoRestrictedError는 형으로 잡는다."""
        from yt_dlp.utils import GeoRestrictedError

        from everyric2.audio.downloader import classify_download_error

        cause = GeoRestrictedError("blocked", countries=["JP"])
        # 문구에 지역 관련 단어가 하나도 없어도 형으로 판정된다
        failure = classify_download_error("blocked", cause)
        assert failure.code == "geo_blocked"

    def test_unavailable_still_raises_video_unavailable_error(self):
        """상위에서 형으로 구분해 온 경로를 깨지 않는다."""
        from everyric2.audio.downloader import VideoUnavailableError, _classified_error

        err = _classified_error(
            RuntimeError("Video unavailable"), "https://y/watch?v=abc", "Download failed"
        )
        assert isinstance(err, VideoUnavailableError)
        assert err.terminal is True

    def test_classified_message_is_what_the_user_sees(self):
        """잡 실패 문구는 `error=str(e)`로 저장된다 — str(e)가 곧 사용자 문구다."""
        from everyric2.audio.downloader import _classified_error

        err = _classified_error(
            RuntimeError("HTTP Error 403: Forbidden"), "https://y/watch?v=abc", "Download failed"
        )
        assert "403" in str(err)
        assert "다시 시도" in str(err)
        assert "Forbidden" not in str(err)  # 영문 원문은 로그용(cause_text)으로만 남는다
        assert err.cause_text == "HTTP Error 403: Forbidden"


class TestJobFailureClassification:
    """jobs.failure_kind 분류 (MoRef 감사 #3) — status="failed" 하나로 사용자 취소·다운로드의
    외부 요인·진짜 시스템 오류가 뭉뚱그려지던 결함의 수정. classify_job_failure는 취소를
    다루지 않는다(그 경로는 cancel API·_consume_cancel이 별도로 "cancelled"를 못 박는다)."""

    def test_downloader_classified_failure_is_external(self):
        """downloader.py가 이미 분류한 실패(로그인요구 등)는 우리 시스템 바깥 요인이다."""
        from everyric2.audio.downloader import _classified_error
        from everyric2.server.worker import classify_job_failure

        err = _classified_error(
            RuntimeError("Sign in to confirm you're not a bot"),
            "https://y/watch?v=abc",
            "Download failed",
        )
        assert classify_job_failure(err) == "external"

    def test_video_unavailable_is_external(self):
        """VideoUnavailableError도 DownloadError 계열이라 external이다."""
        from everyric2.audio.downloader import _classified_error
        from everyric2.server.worker import classify_job_failure

        err = _classified_error(
            RuntimeError("Video unavailable"), "https://y/watch?v=abc", "Download failed"
        )
        assert classify_job_failure(err) == "external"

    def test_dependency_error_is_system_not_external(self):
        """ffmpeg 미설치는 DownloadError 계열이 아니지만, 다운로드 예외 계층에 있다고 해서
        외부 요인은 아니다 — 우리 서버 구성 문제이므로 system."""
        from everyric2.audio.downloader import DependencyError
        from everyric2.server.worker import classify_job_failure

        assert classify_job_failure(DependencyError("ffmpeg 미설치")) == "system"

    def test_js_runtime_code_is_system_despite_download_error_type(self):
        """js_runtime도 downloader.py 자신이 (d) "서버 구성 문제"로 분류한 케이스다 — 형은
        DownloadError지만 login_required 등과 달리 우리 쪽 결함이라 system으로 남긴다."""
        from everyric2.audio.downloader import _classified_error
        from everyric2.server.worker import classify_job_failure

        err = _classified_error(
            RuntimeError("No supported JavaScript runtime could be found"),
            "https://y/watch?v=abc",
            "Download failed",
        )
        assert err.code == "js_runtime"
        assert classify_job_failure(err) == "system"

    def test_unclassified_download_error_is_none_not_forced(self):
        """downloader가 패턴을 못 찾아 code="unknown"으로 영문 원문만 노출한 실패는 외부
        요인인지 우리 쪽 결함인지 판단할 근거가 없다 — 억지로 external/system에 넣지 않는다."""
        from everyric2.audio.downloader import _classified_error
        from everyric2.server.worker import classify_job_failure

        err = _classified_error(
            RuntimeError("some brand new failure mode nobody has seen"),
            "https://y/watch?v=abc",
            "Download failed",
        )
        assert err.code == "unknown"
        assert classify_job_failure(err) is None

    def test_non_download_exception_is_system(self):
        """CTC/demucs 크래시 등 downloader와 무관한 예외는 전부 진짜 시스템 오류다."""
        from everyric2.server.worker import classify_job_failure

        assert classify_job_failure(RuntimeError("CUDA out of memory")) == "system"


# ── 다운로드 egress 순회 ────────────────────────────────────────────
#
# 운영자가 공인 IP를 격리해 워커 env에 출구가 하나 박혀 있고, 예전 구현은 폴백이 없어 그 출구가
# 막히는 순간 다운로드가 통째로 멈췄다. 여기서 못박는 비대칭: **403은 다음 출구로, 삭제는 즉시
# 중단.** 이게 없으면 삭제된 영상 하나를 출구 여러 개로 두들겨 차단을 피하려다 차단을 부른다.
#
# 계층 분할: "재시도할까"는 앱, "어느 출구로"는 호스트. 앱은 egress 값을 불투명 문자열로만
# 다뤄야 하고(IP든 프록시 URL이든), 적용 지점은 _apply_egress 하나다 — 아래 TestEgressBoundary가
# 그 경계를 못박는다(오늘 source_address → 나중 proxy 전환이 그 함수 하나로 끝나는가).


def _downloader(monkeypatch, tmp_path, single: str | None = None):
    """네트워크·ffmpeg 없이 다운로더를 만든다 (의존성 검사만 무력화)."""
    from everyric2.audio.downloader import YouTubeDownloader
    from everyric2.config.settings import AudioSettings

    monkeypatch.setattr(YouTubeDownloader, "_check_dependencies", lambda self: None)
    return YouTubeDownloader(AudioSettings(temp_dir=tmp_path, source_address=single))


def _fail(text: str):
    """분류된 실패를 만드는 헬퍼 — 실제 예외 경로와 같은 분류를 통과시킨다."""
    from everyric2.audio.downloader import _classified_error

    return _classified_error(RuntimeError(text), "https://y/watch?v=abc", "Download failed")


class TestEgressTargetList:
    def test_comma_list_is_parsed_and_deduped_in_order(self):
        from everyric2.audio.downloader import parse_egress_targets

        assert parse_egress_targets("1.1.1.1, 2.2.2.2 ,,1.1.1.1") == ["1.1.1.1", "2.2.2.2"]
        assert parse_egress_targets("") == []
        assert parse_egress_targets(None) == []

    def test_values_are_opaque_strings(self):
        """IP 형태를 가정한 검증을 넣으면 프록시 URL 전환에서 깨진다 — 그대로 통과해야 한다."""
        from everyric2.audio.downloader import parse_egress_targets

        assert parse_egress_targets("http://127.0.0.1:3128, http://127.0.0.1:3129") == [
            "http://127.0.0.1:3128",
            "http://127.0.0.1:3129",
        ]
        assert parse_egress_targets("socks5://127.0.0.1:1080") == ["socks5://127.0.0.1:1080"]
        assert parse_egress_targets("not-an-ip-at-all") == ["not-an-ip-at-all"]

    def test_single_env_var_still_works(self, monkeypatch, tmp_path):
        """기존 배포가 쓰는 EVERYRIC_AUDIO_SOURCE_ADDRESS를 없애면 배포가 깨진다."""
        dl = _downloader(monkeypatch, tmp_path, single="59.8.243.210")
        assert dl._egress_targets() == ["59.8.243.210"]

    def test_list_env_var_takes_priority_and_keeps_the_single_value(self, monkeypatch, tmp_path):
        from everyric2.audio.downloader import EGRESS_ENV

        monkeypatch.setenv(EGRESS_ENV, "10.0.0.1,10.0.0.2")
        dl = _downloader(monkeypatch, tmp_path, single="59.8.243.210")
        # 목록이 앞, 기존 단일 값은 잃지 않게 뒤에 붙는다
        assert dl._egress_targets() == ["10.0.0.1", "10.0.0.2", "59.8.243.210"]

    def test_legacy_list_env_var_is_accepted(self, monkeypatch, tmp_path):
        """이 세션에 운영자에게 먼저 알린 이름이라 env에 이미 들어갔을 수 있다."""
        from everyric2.audio.downloader import LEGACY_ADDRESSES_ENV

        monkeypatch.setenv(LEGACY_ADDRESSES_ENV, "10.0.0.1,10.0.0.2")
        dl = _downloader(monkeypatch, tmp_path)
        assert dl._egress_targets() == ["10.0.0.1", "10.0.0.2"]

    def test_canonical_name_wins_over_legacy(self, monkeypatch, tmp_path):
        from everyric2.audio.downloader import EGRESS_ENV, LEGACY_ADDRESSES_ENV

        monkeypatch.setenv(EGRESS_ENV, "10.0.0.9")
        monkeypatch.setenv(LEGACY_ADDRESSES_ENV, "10.0.0.1,10.0.0.2")
        dl = _downloader(monkeypatch, tmp_path)
        assert dl._egress_targets() == ["10.0.0.9"]

    def test_overlapping_single_value_is_not_tried_twice(self, monkeypatch, tmp_path):
        from everyric2.audio.downloader import EGRESS_ENV

        monkeypatch.setenv(EGRESS_ENV, "10.0.0.1,59.8.243.210")
        dl = _downloader(monkeypatch, tmp_path, single="59.8.243.210")
        assert dl._egress_targets() == ["10.0.0.1", "59.8.243.210"]

    def test_nothing_configured_means_no_egress_applied(self, monkeypatch, tmp_path):
        dl = _downloader(monkeypatch, tmp_path)
        assert dl._egress_targets() == []
        opts: dict = {}
        dl._add_network_options(opts)
        assert opts == {}

    def test_single_arg_call_still_applies_the_first_target(self, monkeypatch, tmp_path):
        """youtube_captions.ydl_opts가 인자 없이 부른다 — 그 경로 동작이 바뀌면 안 된다."""
        from everyric2.audio.downloader import EGRESS_ENV

        monkeypatch.setenv(EGRESS_ENV, "10.0.0.1,10.0.0.2")
        dl = _downloader(monkeypatch, tmp_path)
        opts: dict = {}
        dl._add_network_options(opts)
        assert opts["source_address"] == "10.0.0.1"


class TestEgressBoundary:
    """egress 적용 지점이 **_apply_egress 하나**여야 한다 (운영자 설계 요청).

    오늘은 source_address, 나중엔 프록시 URL. 그 전환이 이 함수 하나로 끝나는지를 여기서
    못박는다 — 목록 순회·재시도 판정·로깅은 값을 해석하지 않으므로 손댈 곳이 없어야 한다.
    """

    def test_today_it_sets_source_address(self):
        from everyric2.audio.downloader import _apply_egress

        opts: dict = {}
        _apply_egress(opts, "10.0.0.1")
        assert opts == {"source_address": "10.0.0.1"}

    def test_it_does_not_validate_the_value(self):
        """프록시 URL이 들어와도 그대로 통과한다 (형태 검증 금지)."""
        from everyric2.audio.downloader import _apply_egress

        opts: dict = {}
        _apply_egress(opts, "http://127.0.0.1:3128")
        assert "http://127.0.0.1:3128" in opts.values()

    def test_swapping_the_function_switches_the_whole_pipeline(self, monkeypatch, tmp_path):
        """프록시로 갈아탄 미래를 시뮬레이션 — 이 함수만 바꿔도 순회가 그대로 돈다.

        _apply_egress를 proxy 버전으로 교체하고, 403 → 다음 출구 → 성공 경로를 실제로 돌려
        ydl_opts에 proxy가 실렸는지 본다. 다른 곳을 고쳐야 한다면 이 테스트가 깨진다.
        """
        from everyric2.audio import downloader as dl_mod

        def proxy_egress(ydl_opts, egress):
            ydl_opts["proxy"] = egress

        monkeypatch.setattr(dl_mod, "_apply_egress", proxy_egress)
        monkeypatch.setenv(dl_mod.EGRESS_ENV, "http://127.0.0.1:3128,http://127.0.0.1:3129")
        dl = _downloader(monkeypatch, tmp_path)

        seen: list[dict] = []

        def fake_once(url, output_dir, filename, egress):
            opts: dict = {}
            dl._add_network_options(opts, egress)
            seen.append(opts)
            if egress == "http://127.0.0.1:3128":
                raise _fail("HTTP Error 403: Forbidden")
            return dl_mod.DownloadResult(
                audio_path=tmp_path / "a.wav", title="t", duration=1.0, url=url
            )

        monkeypatch.setattr(dl, "_download_once", fake_once)
        dl.download("https://www.youtube.com/watch?v=aaaaaaaaaaa")

        # 두 출구를 순서대로 시도했고, 값은 전부 proxy로 실렸다 (source_address는 하나도 없다)
        assert seen == [
            {"proxy": "http://127.0.0.1:3128"},
            {"proxy": "http://127.0.0.1:3129"},
        ]


class TestEgressRotation:
    @pytest.mark.parametrize(
        "text,retryable",
        [
            ("HTTP Error 403: Forbidden", True),
            ("HTTP Error 429: Too Many Requests", True),
            ("Unable to download webpage: The read operation timed out", True),
            ("Sign in to confirm you're not a bot", True),
            ("Video unavailable", False),
            ("Private video", False),
            ("This video is age-restricted", False),
            ("The uploader has not made this video available in your country", False),
            ("No supported JavaScript runtime could be found", False),
            ("some brand new failure nobody has classified", False),
        ],
    )
    def test_retry_decision_table(self, text, retryable):
        """출구를 바꿔 볼 값어치가 있는 실패만 재시도한다 — 이 표가 그 비대칭이다."""
        from everyric2.audio.downloader import egress_retryable

        assert egress_retryable(_fail(text)) is retryable

    def _rotating(self, monkeypatch, tmp_path, targets: str, failures: dict):
        """출구별로 실패를 지정하고 시도 순서를 기록하는 다운로더."""
        from everyric2.audio.downloader import EGRESS_ENV, DownloadResult

        monkeypatch.setenv(EGRESS_ENV, targets)
        dl = _downloader(monkeypatch, tmp_path)
        tried: list[str | None] = []

        def fake_once(url, output_dir, filename, egress):
            tried.append(egress)
            if egress in failures:
                raise _fail(failures[egress])
            return DownloadResult(
                audio_path=tmp_path / "a.wav", title="t", duration=1.0, url=url
            )

        monkeypatch.setattr(dl, "_download_once", fake_once)
        return dl, tried

    def test_403_moves_to_the_next_target(self, monkeypatch, tmp_path):
        dl, tried = self._rotating(
            monkeypatch, tmp_path, "10.0.0.1,10.0.0.2", {"10.0.0.1": "HTTP Error 403: Forbidden"}
        )
        result = dl.download("https://www.youtube.com/watch?v=aaaaaaaaaaa")
        assert tried == ["10.0.0.1", "10.0.0.2"]  # 순서대로, 두 번째에서 성공
        assert result.title == "t"

    def test_deleted_video_stops_immediately(self, monkeypatch, tmp_path):
        """삭제된 영상 하나를 출구 여러 개로 두들기지 않는다 — 접촉만 배수로 늘어난다."""
        from everyric2.audio.downloader import VideoUnavailableError

        dl, tried = self._rotating(
            monkeypatch, tmp_path, "10.0.0.1,10.0.0.2,10.0.0.3", {"10.0.0.1": "Video unavailable"}
        )
        with pytest.raises(VideoUnavailableError):
            dl.download("https://www.youtube.com/watch?v=aaaaaaaaaaa")
        assert tried == ["10.0.0.1"]  # 나머지 출구는 건드리지 않았다

    def test_age_restricted_stops_immediately(self, monkeypatch, tmp_path):
        """쿠키 문제는 출구를 바꿔도 로그인이 생기지 않는다."""
        from everyric2.audio.downloader import DownloadError

        dl, tried = self._rotating(
            monkeypatch,
            tmp_path,
            "10.0.0.1,10.0.0.2",
            {"10.0.0.1": "This video is age-restricted"},
        )
        with pytest.raises(DownloadError):
            dl.download("https://www.youtube.com/watch?v=aaaaaaaaaaa")
        assert tried == ["10.0.0.1"]

    def test_one_pass_only_and_all_failed_is_said_in_the_message(self, monkeypatch, tmp_path):
        """무한 루프 금지 — 목록을 한 바퀴만. 전부 실패는 문구로 알린다(운영자 요청 ④)."""
        from everyric2.audio.downloader import DownloadError

        dl, tried = self._rotating(
            monkeypatch,
            tmp_path,
            "10.0.0.1,10.0.0.2,10.0.0.3",
            {a: "HTTP Error 403: Forbidden" for a in ("10.0.0.1", "10.0.0.2", "10.0.0.3")},
        )
        with pytest.raises(DownloadError) as e:
            dl.download("https://www.youtube.com/watch?v=aaaaaaaaaaa")
        assert tried == ["10.0.0.1", "10.0.0.2", "10.0.0.3"]  # 정확히 한 바퀴
        assert "3개 전부 실패" in str(e.value)
        assert e.value.code == "throttled"  # 분류는 보존된다

    def test_single_target_does_not_claim_a_count(self, monkeypatch, tmp_path):
        """폴백이 없는 배포에 "N개 전부 실패"를 붙이면 사용자에게 운영 용어가 새어 나간다."""
        dl, tried = self._rotating(
            monkeypatch, tmp_path, "10.0.0.1", {"10.0.0.1": "HTTP Error 403: Forbidden"}
        )
        with pytest.raises(Exception) as e:
            dl.download("https://www.youtube.com/watch?v=aaaaaaaaaaa")
        assert tried == ["10.0.0.1"]
        assert "전부 실패" not in str(e.value)
        assert "403" in str(e.value)

    def test_no_target_configured_tries_exactly_once(self, monkeypatch, tmp_path):
        dl, tried = self._rotating(monkeypatch, tmp_path, "", {})
        dl.download("https://www.youtube.com/watch?v=aaaaaaaaaaa")
        assert tried == [None]  # 출구 지정 없이 기본 경로 1회

    def test_successful_target_is_logged(self, monkeypatch, tmp_path, caplog):
        """관측이 없으면 폴백이 실제로 도는지 알 수 없다 (운영자 요청 ③)."""
        dl, _ = self._rotating(
            monkeypatch, tmp_path, "10.0.0.1,10.0.0.2", {"10.0.0.1": "HTTP Error 403: Forbidden"}
        )
        with caplog.at_level(logging.INFO, logger="everyric2.audio.downloader"):
            dl.download("https://www.youtube.com/watch?v=aaaaaaaaaaa")
        # 실패한 출구와 실패 종류, 성공한 출구가 모두 남는다
        assert "10.0.0.1" in caplog.text
        assert "code=throttled" in caplog.text
        assert "성공 (egress=10.0.0.2" in caplog.text

    def test_video_info_rotates_too(self, monkeypatch, tmp_path):
        """링크 검증 경로(get_video_info)도 같은 단일 장애점에 걸려 있었다."""
        from everyric2.audio.downloader import EGRESS_ENV, VideoInfo

        monkeypatch.setenv(EGRESS_ENV, "10.0.0.1,10.0.0.2")
        dl = _downloader(monkeypatch, tmp_path)
        tried: list[str | None] = []

        def fake_once(url, egress):
            tried.append(egress)
            if egress == "10.0.0.1":
                raise _fail("HTTP Error 403: Forbidden")
            return VideoInfo(title="t", duration=1.0, url=url)

        monkeypatch.setattr(dl, "_extract_info_once", fake_once)
        assert dl.get_video_info("https://www.youtube.com/watch?v=aaaaaaaaaaa").title == "t"
        assert tried == ["10.0.0.1", "10.0.0.2"]


# ── opus 우선 다운로드: 산출물 발견 + 디코드 구제 ──────────────────────
#
# 전곡을 wav로 트랜스코드해 곡당 수십 MB가 캐시에 쌓이던 것을 opus 우선 스트림카피로
# 바꿨다(운영자 요청: 전곡 wav 보존 용량 과다). 소스 코덱에 따라 산출물 확장자가
# opus/m4a/webm으로 갈리므로 wav 하나만 찾던 예전 글롭은 못 찾는다 — 이 자리를
# _locate_downloaded_file(requested_downloads 우선 → 예상 경로 → 확장자 글롭)과
# _ensure_decodable(디코드 프로브 실패 시에만 로컬 wav 구제)로 대체했다.


def _fake_yt_dlp_module(monkeypatch, info: dict):
    """`import yt_dlp`가 이 가짜 모듈을 받도록 sys.modules에 심는다.

    _download_once는 함수 안에서 지역 import를 쓰므로(``import yt_dlp``), 실제 네트워크
    없이 그 경로를 통째로 돌리려면 모듈 자체를 바꿔치기해야 한다.
    """
    import sys
    import types

    fake = types.ModuleType("yt_dlp")

    class _FakeYDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, url, download):
            return info

    fake.YoutubeDL = _FakeYDL
    utils_ns = types.SimpleNamespace()
    utils_ns.DownloadError = type("FakeYtDlpDownloadError", (Exception,), {})
    utils_ns.sanitize_filename = lambda s: s
    fake.utils = utils_ns
    monkeypatch.setitem(sys.modules, "yt_dlp", fake)
    return fake


class TestDownloadedFileDiscovery:
    def test_requested_downloads_filepath_is_used_when_present(self, monkeypatch, tmp_path):
        """최신 yt-dlp가 후처리 후 최종 경로를 실어 주면 그 값을 그대로 믿는다."""
        from everyric2.audio import downloader as dl_mod

        dl = _downloader(monkeypatch, tmp_path)
        monkeypatch.setattr(dl_mod, "_probe_decodable", lambda p: True)

        produced = tmp_path / "제목과-다른-실제파일.opus"
        produced.write_bytes(b"opus-bytes")
        _fake_yt_dlp_module(
            monkeypatch,
            {
                "title": "My Song",
                "duration": 12.0,
                "requested_downloads": [{"filepath": str(produced)}],
            },
        )

        result = dl._download_once(
            "https://www.youtube.com/watch?v=aaaaaaaaaaa", tmp_path, None, None
        )
        assert result.audio_path == produced
        assert result.title == "My Song"

    def test_glob_discovery_finds_opus_when_no_requested_downloads(self, monkeypatch, tmp_path):
        """구버전 yt-dlp(또는 필드 누락)는 확장자 제한 글롭으로 내려간다 — wav 전용이 아니다."""
        from everyric2.audio import downloader as dl_mod

        dl = _downloader(monkeypatch, tmp_path)
        monkeypatch.setattr(dl_mod, "_probe_decodable", lambda p: True)

        # 예상 경로("My Song.opus")는 없고, 제목 접두사로 시작하는 실제 산출물만 있다 —
        # 글롭이 찾아야 한다.
        actual = tmp_path / "My Song [id123].opus"
        actual.write_bytes(b"opus-bytes")
        _fake_yt_dlp_module(monkeypatch, {"title": "My Song", "duration": 5.0})

        result = dl._download_once(
            "https://www.youtube.com/watch?v=aaaaaaaaaaa", tmp_path, None, None
        )
        assert result.audio_path == actual

    def test_explicit_filename_glob_never_grabs_another_jobs_file(self, monkeypatch, tmp_path):
        """기존 불변식: filename이 있으면 글롭도 그 접두사로 좁힌다."""
        from everyric2.audio import downloader as dl_mod

        dl = _downloader(monkeypatch, tmp_path)
        monkeypatch.setattr(dl_mod, "_probe_decodable", lambda p: True)

        mine = tmp_path / "job-abc [x].m4a"
        mine.write_bytes(b"m4a-bytes")
        other = tmp_path / "job-xyz [y].m4a"
        other.write_bytes(b"m4a-bytes-other")
        _fake_yt_dlp_module(monkeypatch, {"title": "irrelevant", "duration": 3.0})

        result = dl._download_once(
            "https://www.youtube.com/watch?v=aaaaaaaaaaa", tmp_path, "job-abc", None
        )
        assert result.audio_path == mine


class TestDecodabilityFallback:
    """스트림카피 산출물이 아주 드물게 안 열리는 경우를 로컬 트랜스코드로만 구제한다
    (유튜브 재접촉 없음)."""

    def test_decodable_file_is_kept_as_is(self, monkeypatch, tmp_path):
        from everyric2.audio import downloader as dl_mod

        dl = _downloader(monkeypatch, tmp_path)
        monkeypatch.setattr(dl_mod, "_probe_decodable", lambda p: True)

        src = tmp_path / "a.opus"
        src.write_bytes(b"opus-bytes")
        assert dl._ensure_decodable(src) == src
        assert src.exists()  # 손대지 않는다

    def test_probe_failure_transcodes_locally_and_removes_original(self, monkeypatch, tmp_path):
        from everyric2.audio import downloader as dl_mod

        dl = _downloader(monkeypatch, tmp_path)
        monkeypatch.setattr(dl_mod, "_probe_decodable", lambda p: False)

        src = tmp_path / "broken.opus"
        src.write_bytes(b"not-really-opus")

        calls = []

        def fake_run(cmd, check, capture_output):
            calls.append(cmd)
            # 실제 ffmpeg 대신 wav 출력만 흉내낸다
            out_path = Path(cmd[-1])
            out_path.write_bytes(b"RIFF....WAVEfmt ")
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(dl_mod.subprocess, "run", fake_run)

        result = dl._ensure_decodable(src)
        assert result == src.with_suffix(".wav")
        assert result.exists()
        assert not src.exists()  # 디코드 안 되는 원본은 지운다
        assert calls and calls[0][0] == "ffmpeg"
        assert "-i" in calls[0] and str(src) in calls[0]
        # 유튜브를 다시 접촉하지 않는다 — yt_dlp를 아예 안 불렀다는 것이 이 테스트의 전제

    def test_transcode_failure_still_cleans_up_original_and_raises(self, monkeypatch, tmp_path):
        from everyric2.audio import downloader as dl_mod
        from everyric2.audio.downloader import DownloadError

        dl = _downloader(monkeypatch, tmp_path)
        monkeypatch.setattr(dl_mod, "_probe_decodable", lambda p: False)

        src = tmp_path / "broken.opus"
        src.write_bytes(b"not-really-opus")

        def fake_run(cmd, check, capture_output):
            raise subprocess.CalledProcessError(1, cmd)

        monkeypatch.setattr(dl_mod.subprocess, "run", fake_run)

        with pytest.raises(DownloadError):
            dl._ensure_decodable(src)
        assert not src.exists()  # 실패해도 디코드 안 되는 원본을 남기지 않는다


# ── ④ audio_hash 계약 고정 ──────────────────────────────────────────


class TestAudioHashContract:
    """해시는 바꾸지 않았다 — 경로 독립을 값싸게 만들 방법이 없다(worker._acquire_audio 실측).

    대신 캐시 키가 의존하는 계약을 못박는다: 같은 바이트면 같은 해시, 다르면 다른 해시,
    그리고 **32자 hex**여야 한다 (SyncResult.audio_hash가 String(32)다 — 길이가 늘면
    저장이 조용히 잘린다).
    """

    def test_same_bytes_same_hash_and_fits_the_column(self, tmp_path):
        from everyric2.server.worker import compute_audio_hash

        a = tmp_path / "a.wav"
        b = tmp_path / "b.m4a"
        a.write_bytes(b"same-audio-bytes" * 1000)
        b.write_bytes(b"same-audio-bytes" * 1000)

        ha, hb = compute_audio_hash(a), compute_audio_hash(b)
        assert ha == hb
        assert len(ha) == 32 and all(c in "0123456789abcdef" for c in ha)

    def test_different_bytes_different_hash(self, tmp_path):
        from everyric2.server.worker import compute_audio_hash

        a = tmp_path / "a.wav"
        b = tmp_path / "b.wav"
        a.write_bytes(b"audio-one")
        b.write_bytes(b"audio-two")
        assert compute_audio_hash(a) != compute_audio_hash(b)


# ── ⑤ 캐시 복사 로그가 실제 복사 분기에 있다 ────────────────────────


class TestCacheCopyLogging:
    def _run(self, caplog, tmp_path, same_video: bool):
        import asyncio

        from everyric2.server.db.repository import JobRepository, SyncRepository, hash_lyrics
        from everyric2.server.worker import _try_complete_from_cache

        lyrics = "라인1\n라인2"
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"x")

        async def body():
            async with _db() as sm:
                async with sm() as s:
                    await SyncRepository(s).create(
                        video_id=VID_A,
                        lyrics_hash=hash_lyrics(lyrics),
                        timestamps=[{"text": "라인1", "start": 1.0, "end": 2.0}],
                        audio_hash="hashA",
                    )
                    job = await JobRepository(s).create(
                        video_id=VID_A if same_video else VID_B, lyrics=lyrics
                    )
                    await s.commit()
                with caplog.at_level(logging.INFO, logger="everyric2.server.worker"):
                    ok = await _try_complete_from_cache(
                        job.id, job, "hashA", hash_lyrics(lyrics), str(audio)
                    )
                assert ok is True

        asyncio.run(body())
        return caplog.text

    def test_cross_video_copy_is_logged(self, caplog, tmp_path):
        """복구 불가 사고를 낸 경로 — 발생을 로그로 추적할 수 있어야 한다."""
        text = self._run(caplog, tmp_path, same_video=False)
        assert "copied sync" in text
        assert VID_A in text and VID_B in text

    def test_same_video_reuse_does_not_claim_a_copy(self, caplog, tmp_path):
        """복사가 없는 분기가 "copied"라고 말하면 로그가 거짓이 된다."""
        text = self._run(caplog, tmp_path, same_video=True)
        assert "copied sync" not in text
        assert "reusing this video's own sync" in text


# ── ⑥ m4a 경로에서도 과길이 검사가 산다 ─────────────────────────────


_HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg/ffprobe 필요")
class TestDurationProbeCoversM4a:
    @staticmethod
    def _make_m4a(tmp_path, seconds: int = 3):
        dest = tmp_path / "cached.m4a"
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
                "-c:a", "aac", "-b:a", "64k", str(dest),
            ],
            check=True,
            capture_output=True,
        )
        return dest

    def test_libsndfile_cannot_read_m4a(self, tmp_path):
        """이 결함의 전제를 실행으로 못박는다 — 추론이 아니라 실측이다.

        libsndfile 1.2.2는 m4a/AAC를 지원하지 않는다(available_formats에 없음). 미디어 캐시
        경로는 `-acodec copy`로 m4a를 넘기므로, soundfile만 쓰던 예전 구현은 그 경로에서
        항상 None을 돌려줬고 호출부의 `if duration and ...`이 상한 검사를 건너뛰었다.
        """
        import soundfile as sf

        path = self._make_m4a(tmp_path)
        assert "M4A" not in sf.available_formats()
        with pytest.raises(Exception):
            sf.info(str(path))

    def test_duration_is_read_from_m4a(self, tmp_path):
        from everyric2.server.worker import _audio_duration_sec

        path = self._make_m4a(tmp_path, seconds=3)
        assert _audio_duration_sec(str(path)) == pytest.approx(3.0, abs=0.3)

    def test_over_length_m4a_is_rejected_by_the_pipeline(self, tmp_path, monkeypatch):
        """캐시 lookup이 duration을 안 줘도 파이프라인이 상한을 지킨다.

        고치기 전에는 길이가 None이 되어 `if duration and ...`이 통째로 건너뛰어졌고,
        장시간 영상이 GPU 슬롯을 점유했다.
        """
        import asyncio

        from everyric2.server import worker as worker_mod

        path = self._make_m4a(tmp_path, seconds=3)

        class _Hooks:
            async def report(self, progress, stage):
                pass

            async def progress(self, progress, stage):
                return True

            async def cache_check(self, audio_hash, audio_path):
                return False

        monkeypatch.setattr(
            worker_mod,
            "_acquire_audio",
            lambda job: {"audio_path": str(path), "audio_hash": "deadbeef"},
        )
        job = worker_mod.JobInput(
            job_id="job-len",
            video_id=VID_A,
            lyrics="라인1",
            max_audio_sec=2,  # 3초 파일 > 2초 상한
        )
        with pytest.raises(worker_mod.PipelineError) as e:
            asyncio.run(worker_mod.run_pipeline(job, _Hooks()))
        assert "너무 길어요" in str(e.value)
        assert not path.exists()  # 거부하면서 오디오도 정리한다
