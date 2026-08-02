"""``OwsmEngine`` 서버 이식 검증 — 프리픽스 프레임 가드가 가장 중요하다.

``everyric2/alignment/owsm_engine.py``와 ``everyric2/alignment/_owsm_worker.py`` 모듈
docstring이 기록한 핵심 실패 모드: OWSM 인코더는 ``[<lang>, <asr>]`` 2토큰 프리픽스에
조건화돼 있어 그 상태가 인코더 출력 **앞에 붙는다.** 이식 과정에서 프리픽스 프레임 검증이
유실되면 **오류 없이** 모든 타임스탬프가 조용히 ~160ms 밀린다 — 이 스위트가 그 회귀를
못박는다. GPU도 ESPnet도 없이 검증 가능하도록 가드(``_owsm_worker._verify_prefix_surplus``)
를 순수 함수로 뽑아 뒀다. 실제 추론(``_owsm_worker.run``)은 격리 venv와 GPU가 있어야
돌기 때문에 이 스위트에서는 절대 실행하지 않는다(무거운 연산 금지 규약) — 대신 서브프로세스
호출·에러 래핑·워커 응답 파싱은 목(mock)으로 검증한다.
"""

from __future__ import annotations

import math

import pytest

from everyric2.alignment import _owsm_worker, owsm_engine
from everyric2.alignment.base import AlignmentError, EngineNotAvailableError
from everyric2.audio.loader import AudioData
from everyric2.config.settings import AlignmentSettings
from everyric2.inference.prompt import LyricLine


def _audio() -> AudioData:
    return AudioData(waveform=None, sample_rate=16000, duration=1.0)  # type: ignore[arg-type]


# ── 프리픽스 프레임 가드 — 가장 중요한 테스트 ──────────────────────────────────────


def test_verify_prefix_surplus_passes_when_surplus_matches_prefix_len():
    # 정상 조건: 잉여 프레임(2) == 프리픽스 길이(2, [<lang>, <asr>]) — 예외 없이 통과한다.
    _owsm_worker._verify_prefix_surplus(total_frames=102, audio_frames=100, prefix_len=2)


def test_verify_prefix_surplus_raises_when_prefix_frames_go_unaccounted():
    # 프리픽스 프레임이 오디오로 오인되는 바로 그 실패 모드: 잉여가 1인데 프리픽스는 2다.
    # 이 가드가 없으면 이 상황이 조용히 통과하고 모든 타임스탬프가 ~160ms 밀린다.
    with pytest.raises(RuntimeError, match="unexpected OWSM encoder length"):
        _owsm_worker._verify_prefix_surplus(total_frames=101, audio_frames=100, prefix_len=2)


def test_verify_prefix_surplus_raises_when_surplus_is_zero():
    # 프리픽스가 아예 안 붙은 것처럼 보이는 극단값도 실패해야 한다 — 조용한 통과 금지.
    with pytest.raises(RuntimeError, match="unexpected OWSM encoder length"):
        _owsm_worker._verify_prefix_surplus(total_frames=100, audio_frames=100, prefix_len=2)


def test_verify_prefix_surplus_raises_when_surplus_overshoots():
    with pytest.raises(RuntimeError, match="unexpected OWSM encoder length"):
        _owsm_worker._verify_prefix_surplus(total_frames=105, audio_frames=100, prefix_len=2)


def test_expected_audio_frames_matches_conv2d8_arithmetic():
    # 30초 버퍼(16kHz) 기준 — hop 160(10ms) STFT 다음 conv2d8 3단(커널3·스트라이드2).
    n_samples = 30 * 16_000
    frames = _owsm_worker._expected_audio_frames(n_samples)

    expected = n_samples // 160 + 1
    for _ in range(3):
        expected = (expected - 3) // 2 + 1

    assert frames == expected
    assert frames > 0


def test_expected_audio_frames_monotonic_in_sample_count():
    # 오디오가 길수록 프레임 수도 늘어나야 한다 — 산술이 뒤집히면 즉시 드러난다.
    small = _owsm_worker._expected_audio_frames(16_000)
    large = _owsm_worker._expected_audio_frames(16_000 * 60)
    assert large > small


# ── 워커 응답 -> 서버 계약(SyncResult) 변환 ─────────────────────────────────────────


def test_build_sync_results_line_count_mismatch_raises():
    lyrics = [LyricLine(text="가", line_number=1), LyricLine(text="나", line_number=2)]
    with pytest.raises(AlignmentError):
        owsm_engine._build_sync_results(lyrics, ["가", "나"], {"lines": [{}]})


def test_build_sync_results_maps_segs_to_word_segments():
    lyrics = [LyricLine(text="안녕", line_number=1)]
    result = {
        "audio_sec": 2.0,
        "lines": [
            {
                "segs": [
                    {"t": "안", "start": 0.1, "end": 0.5, "confidence": 0.9},
                    {"t": "녕", "start": 0.5, "end": 0.9, "confidence": 0.8},
                ]
            }
        ],
    }
    results, words = owsm_engine._build_sync_results(lyrics, ["안녕"], result)

    assert len(results) == 1
    sync = results[0]
    assert sync.text == "안녕"
    assert sync.start_time == pytest.approx(0.1)
    assert sync.end_time == pytest.approx(0.9)
    assert sync.word_segments is not None
    assert [w.word for w in sync.word_segments] == ["안", "녕"]
    assert [w.confidence for w in sync.word_segments] == [0.9, 0.8]
    assert len(words) == 2
    # 라인 단위 confidence — 이게 없으면 새 스택 라우팅(worker._line_log_conf_median)이
    # 항상 None을 봐서 문턱 판정 자체가 성립하지 않는다(2026-08-03 실곡 검증 결함).
    assert sync.confidence is not None
    assert sync.confidence == pytest.approx(owsm_engine._line_confidence([0.9, 0.8]))


def test_build_sync_results_interpolates_unaligned_lines():
    # 다중 글자 SentencePiece 토큰이 그 줄에서 하나도 매칭 안 되면(전부 OOV) 워커가 빈
    # segs를 낸다 — 그 줄은 앞뒤 정렬 줄 사이 간격으로 보간되고 순서가 깨지면 안 된다.
    lyrics = [
        LyricLine(text="가", line_number=1),
        LyricLine(text="?", line_number=2),
        LyricLine(text="다", line_number=3),
    ]
    result = {
        "audio_sec": 3.0,
        "lines": [
            {"segs": [{"t": "가", "start": 0.0, "end": 1.0, "confidence": 0.9}]},
            {"segs": []},
            {"segs": [{"t": "다", "start": 2.0, "end": 3.0, "confidence": 0.9}]},
        ],
    }
    results, _ = owsm_engine._build_sync_results(lyrics, ["가", "?", "다"], result)

    assert results[0].end_time <= results[1].start_time
    assert results[1].start_time <= results[1].end_time <= results[2].start_time
    assert results[1].confidence is None  # 빈 segs — 실측 글자가 없다
    assert results[0].confidence is not None
    assert results[2].confidence is not None


# ── 라인 신뢰도(_line_confidence) — 2026-08-03 실곡 검증에서 잡힌 결함: 이 함수가
# 없던 시절 SyncResult.confidence가 항상 None이라 새 스택 라우팅 신호(line_log_conf_
# median)가 절대 값을 못 봤다. 벤치 hf_ctc.py::_confidence(원시 로그점수 평균 후 exp)와
# 수학적으로 같은 값을 내야 한다 — 문턱값이 그 공식에 대해 실측 보정됐기 때문이다.
def test_line_confidence_matches_bench_log_mean_formula():
    raw_log_scores = [-0.1, -0.2, -0.05]
    word_confs = [round(math.exp(min(0.0, s)), 6) for s in raw_log_scores]
    expected = math.exp(sum(raw_log_scores) / len(raw_log_scores))
    assert owsm_engine._line_confidence(word_confs) == pytest.approx(expected, abs=1e-6)


def test_line_confidence_none_when_no_measured_words():
    assert owsm_engine._line_confidence([]) is None
    assert owsm_engine._line_confidence([None, None]) is None


# ── OwsmEngine 계약: is_available / align / emission_for / get_engine_type ────────────


def test_get_engine_type():
    assert owsm_engine.OwsmEngine.get_engine_type() == "owsm"


def test_transcribe_not_implemented():
    engine = owsm_engine.OwsmEngine()
    with pytest.raises(NotImplementedError):
        engine.transcribe(_audio())


def test_emission_for_not_supported():
    # OwsmEngine은 emission_for를 override하지 않는다 — 서브프로세스 격리라 emission
    # 텐서가 프로세스 경계를 못 넘는다(모듈 docstring 참고). 기본 구현(None)이 그대로다.
    engine = owsm_engine.OwsmEngine()
    assert engine.emission_for(_audio()) is None


def test_is_available_false_when_worker_python_missing(tmp_path, monkeypatch):
    missing = tmp_path / "nope" / "python.exe"
    monkeypatch.setattr(owsm_engine, "_default_owsm_python", lambda: missing)
    assert owsm_engine.OwsmEngine().is_available() is False


def test_is_available_false_when_snapshot_missing(tmp_path, monkeypatch):
    python_path = tmp_path / "python.exe"
    python_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(owsm_engine, "_default_owsm_python", lambda: python_path)
    monkeypatch.setattr(owsm_engine, "_find_snapshot", lambda: None)
    assert owsm_engine.OwsmEngine().is_available() is False


def test_is_available_true_when_both_present(tmp_path, monkeypatch):
    python_path = tmp_path / "python.exe"
    python_path.write_text("", encoding="utf-8")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    monkeypatch.setattr(owsm_engine, "_default_owsm_python", lambda: python_path)
    monkeypatch.setattr(owsm_engine, "_find_snapshot", lambda: snapshot)
    assert owsm_engine.OwsmEngine().is_available() is True


def test_align_rejects_empty_lyrics():
    engine = owsm_engine.OwsmEngine()
    with pytest.raises(AlignmentError):
        engine.align(_audio(), [])


def test_align_raises_engine_not_available_without_worker_python(tmp_path, monkeypatch):
    # 조용한 폴백 금지 — 워커 인터프리터가 없으면 명시적으로 죽어야 한다.
    missing = tmp_path / "nope" / "python.exe"
    monkeypatch.setattr(owsm_engine, "_default_owsm_python", lambda: missing)
    engine = owsm_engine.OwsmEngine()
    with pytest.raises(EngineNotAvailableError):
        engine.align(_audio(), [LyricLine(text="안녕", line_number=1)])


def test_align_raises_engine_not_available_without_snapshot(tmp_path, monkeypatch):
    # 조용한 폴백 금지 — 모델 스냅샷이 캐시에 없으면 명시적으로 죽어야 한다.
    python_path = tmp_path / "python.exe"
    python_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(owsm_engine, "_default_owsm_python", lambda: python_path)
    monkeypatch.setattr(owsm_engine, "_find_snapshot", lambda: None)
    engine = owsm_engine.OwsmEngine()
    with pytest.raises(EngineNotAvailableError):
        engine.align(_audio(), [LyricLine(text="안녕", line_number=1)])


def test_default_owsm_python_platform_paths(monkeypatch):
    monkeypatch.setattr(owsm_engine.sys, "platform", "win32")
    win_path = owsm_engine._default_owsm_python()
    assert win_path.name == "python.exe"
    assert win_path.parent.name == "Scripts"

    monkeypatch.setattr(owsm_engine.sys, "platform", "linux")
    posix_path = owsm_engine._default_owsm_python()
    assert posix_path.name == "python3"
    assert posix_path.parent.name == "bin"


def test_worker_python_path_setting_overrides_default(tmp_path):
    override = tmp_path / "custom-python"
    settings = AlignmentSettings(owsm_python_path=str(override))
    engine = owsm_engine.OwsmEngine(config=settings)
    assert engine._worker_python() == override


def test_run_worker_wraps_nonzero_exit_in_alignment_error(monkeypatch):
    class _FakeCompleted:
        returncode = 1
        stdout = "some stdout"
        stderr = "boom"

    monkeypatch.setattr(owsm_engine.subprocess, "run", lambda *a, **k: _FakeCompleted())
    engine = owsm_engine.OwsmEngine()
    with pytest.raises(AlignmentError, match="owsm worker failed"):
        engine._run_worker({"lines": ["x"]})


def test_confidence_conversion_matches_worker_helper():
    assert _owsm_worker._confidence(0.0) == 1.0
    assert _owsm_worker._confidence(float("-inf")) == 0.0
    assert _owsm_worker._confidence(-1.0) == pytest.approx(math.exp(-1.0), abs=1e-6)
