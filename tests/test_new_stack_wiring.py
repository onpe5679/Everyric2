"""새 정렬 스택(3단계 라우팅 + owsm/omniasr 앵커 + 2패스 리파이너) 배선 회귀 테스트.

이식된 부품 자체(owsm_engine/omniasr_engine/refine_window/display_fixes)는 각자의
테스트 스위트(test_alignment_new_engines_wiring.py·test_refine_window.py·
test_display_fixes.py)가 이미 못박고 있다. 라우팅 자체(scripts/bench_adapters/routed.py의
line_log_conf_median·문턱값)도 벤치 쪽에서 실측됐다. 이 파일은 그것들을 ``everyric2/
server/worker.py``에 실제로 배선한 층만 검증한다: 설정 정합성 가드, 3단계 라우팅
(고속→구원→en 좌초 승급)의 분기 로직, refine_window(Path 계약) ↔ BaseAlignmentEngine.
emission_for(AudioData 계약) 사이의 배선 어댑터, 그리고 **조용한 구스택 폴백이 없는지**
(운영자 지시, 2026-08-03 정정 — 분리기/앵커/리파이너가 없거나 실패하면 예외가 그대로
올라가야 한다). 모델을 실제로 로드/추론하지 않는다 — 앵커·리파이너·분리기는 전부
페이크(호출 계약 수준 검증, 코디네이터 지시).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from everyric2.alignment.emission import EngineEmission
from everyric2.alignment.refine_window import PronSegmentSpan
from everyric2.audio.loader import AudioData
from everyric2.config.settings import (
    AlignmentSettings,
    AudioSettings,
    SegmentationSettings,
    Settings,
)
from everyric2.inference.prompt import LyricLine, SyncResult, WordSegment
from everyric2.server import worker


# ---------------------------------------------------------------------------
# 설정 정합성 가드 — Settings의 cross-field validator
# ---------------------------------------------------------------------------


class TestNewStackSeparatorConsistency:
    def test_legacy_engine_with_htdemucs_is_fine(self):
        Settings(alignment=AlignmentSettings(engine="ctc"), audio=AudioSettings(separator_backend="htdemucs"))

    def test_new_stack_engine_with_htdemucs_raises(self):
        with pytest.raises(ValueError, match="bs-polarformer-fp16"):
            Settings(
                alignment=AlignmentSettings(engine="owsm"),
                audio=AudioSettings(separator_backend="htdemucs"),
            )

    def test_omniasr_engine_with_htdemucs_also_raises(self):
        with pytest.raises(ValueError, match="bs-polarformer-fp16"):
            Settings(
                alignment=AlignmentSettings(engine="omniasr"),
                audio=AudioSettings(separator_backend="htdemucs"),
            )

    def test_new_stack_engine_with_polarformer_is_fine(self):
        Settings(
            alignment=AlignmentSettings(engine="owsm"),
            audio=AudioSettings(separator_backend="bs-polarformer-fp16"),
        )

    def test_legacy_engine_with_polarformer_is_fine(self):
        Settings(
            alignment=AlignmentSettings(engine="ctc"),
            audio=AudioSettings(separator_backend="bs-polarformer-fp16"),
        )


# ---------------------------------------------------------------------------
# 새 스택 게이트
# ---------------------------------------------------------------------------


class _FakeSettings:
    def __init__(self, engine: str) -> None:
        self.alignment = AlignmentSettings(engine=engine)


class TestNewStackGate:
    def test_disabled_for_legacy_engines(self):
        for engine in ("ctc", "nemo", "gpu-hybrid", "sofa"):
            assert worker._new_stack_enabled(_FakeSettings(engine)) is False

    def test_enabled_for_new_anchor_engines(self):
        for engine in ("owsm", "omniasr"):
            assert worker._new_stack_enabled(_FakeSettings(engine)) is True


# ---------------------------------------------------------------------------
# 라우팅 점수 — scripts/bench_adapters/routed.py::line_log_conf_median 이식
# ---------------------------------------------------------------------------


def _line(conf: float | None) -> SyncResult:
    return SyncResult(text="x", start_time=0.0, end_time=1.0, confidence=conf)


class TestLineLogConfMedian:
    def test_none_when_no_confidences(self):
        assert worker._line_log_conf_median([_line(None), _line(None)]) is None

    def test_median_of_log_confidences(self):
        import math

        results = [_line(0.5), _line(0.25), _line(0.125)]
        assert worker._line_log_conf_median(results) == pytest.approx(math.log(0.25))

    def test_zero_confidence_uses_log_floor_not_negative_infinity(self):
        import math

        score = worker._line_log_conf_median([_line(0.0)])
        assert score == pytest.approx(math.log(1e-6))
        assert math.isfinite(score)

    def test_ignores_lines_without_confidence(self):
        import math

        results = [_line(0.5), _line(None)]
        assert worker._line_log_conf_median(results) == pytest.approx(math.log(0.5))


# ---------------------------------------------------------------------------
# _PathBridgedRefiner — Path 계약(refine_window) <-> AudioData 계약(BaseAlignmentEngine)
# ---------------------------------------------------------------------------


class _RecordingEngine:
    def __init__(self, result: Any, available: bool = True) -> None:
        self.result = result
        self.received: Any = None
        self._available = available

    def is_available(self) -> bool:
        return self._available

    def emission_for(self, audio: Any) -> Any:
        self.received = audio
        return self.result


class _RecordingLoader:
    def __init__(self, audio: Any) -> None:
        self.audio = audio
        self.received_path: Path | None = None

    def load(self, path: Path) -> Any:
        self.received_path = path
        return self.audio


class TestPathBridgedRefiner:
    def test_loads_path_then_delegates_to_engine_with_audio_data(self):
        fake_audio = object()
        fake_emission = object()
        engine = _RecordingEngine(fake_emission)
        loader = _RecordingLoader(fake_audio)
        bridged = worker._PathBridgedRefiner(engine, loader)

        target = Path("some/vocals.wav")
        out = bridged.emission_for(target)

        assert loader.received_path == target
        assert engine.received is fake_audio
        assert out is fake_emission


# ---------------------------------------------------------------------------
# _pron_seg_to_wire — PronSegmentSpan -> 서버 wire PronSegment
# ---------------------------------------------------------------------------


class TestPronSegToWire:
    def test_minimal_span_omits_optional_keys(self):
        span = PronSegmentSpan(text="가", start=1.0, end=1.5)
        wire = worker._pron_seg_to_wire(span)
        assert wire == {"text": "가", "start": 1.0, "end": 1.5}

    def test_unresolved_and_confidence_and_word_end_included(self):
        span = PronSegmentSpan(
            text="ka", start=0.0, end=0.2, resolved=False, confidence=0.42, word_end=True
        )
        wire = worker._pron_seg_to_wire(span)
        assert wire == {
            "text": "ka",
            "start": 0.0,
            "end": 0.2,
            "resolved": False,
            "confidence": 0.42,
            "word_end": True,
        }


# ---------------------------------------------------------------------------
# 공용 픽스처
# ---------------------------------------------------------------------------


class _FakeAnchor:
    """EngineFactory.get_engine이 돌려주는 앵커를 흉내낸다 — align()만 기록·재생한다."""

    def __init__(self, results: list[SyncResult], available: bool = True) -> None:
        self._results = results
        self._available = available
        self.align_calls: list[dict[str, Any]] = []

    def is_available(self) -> bool:
        return self._available

    def align(self, audio, lyrics, language=None, progress_callback=None):
        self.align_calls.append({"audio": audio, "lyrics": lyrics, "language": language})
        return self._results


def _settings(**overrides) -> Any:
    kwargs: dict[str, Any] = {"engine": "owsm", "two_pass_enabled": False}
    kwargs.update(overrides)
    align = AlignmentSettings(**kwargs)
    return SimpleNamespace(
        alignment=align, segmentation=SegmentationSettings(), audio=AudioSettings()
    )


def _silence(seconds: float = 0.5, sr: int = 16000) -> AudioData:
    n = max(1, int(seconds * sr))
    return AudioData(waveform=np.zeros(n, dtype=np.float32), sample_rate=sr, duration=seconds)


class _FakeSepResult:
    def __init__(self, vocals: AudioData, accompaniment: AudioData) -> None:
        self.vocals = vocals
        self.accompaniment = accompaniment


class _FakeSeparator:
    def __init__(self, available: bool, sep_result: Any = None) -> None:
        self._available = available
        self._sep_result = sep_result
        self.config = SimpleNamespace(separator_backend="bs-polarformer-fp16")
        self.separate_calls = 0

    def is_available(self) -> bool:
        return self._available

    def separate(self, audio, use_gpu: bool = True):
        self.separate_calls += 1
        if self._sep_result is None:
            raise RuntimeError("separate() should not be called when unavailable")
        return self._sep_result


# ---------------------------------------------------------------------------
# _separate_stems_required — 구원 단계 전용, 조용한 폴백 금지
# ---------------------------------------------------------------------------


class TestSeparateStemsRequired:
    def test_raises_clearly_when_unavailable(self, monkeypatch):
        fake = _FakeSeparator(available=False)
        monkeypatch.setattr(
            "everyric2.audio.separator.get_shared_separator", lambda config=None: fake
        )
        with pytest.raises(RuntimeError, match="bs-polarformer-fp16"):
            worker._separate_stems_required(_silence(), _settings())
        assert fake.separate_calls == 0

    def test_returns_result_when_available(self, monkeypatch):
        sep = _FakeSepResult(_silence(), _silence())
        fake = _FakeSeparator(available=True, sep_result=sep)
        monkeypatch.setattr(
            "everyric2.audio.separator.get_shared_separator", lambda config=None: fake
        )
        out = worker._separate_stems_required(_silence(), _settings())
        assert out is sep
        assert fake.separate_calls == 1

    def test_propagates_separate_failure_without_swallowing(self, monkeypatch):
        class _RaisingSeparator(_FakeSeparator):
            def separate(self, audio, use_gpu: bool = True):
                raise ValueError("boom")

        fake = _RaisingSeparator(available=True)
        monkeypatch.setattr(
            "everyric2.audio.separator.get_shared_separator", lambda config=None: fake
        )
        with pytest.raises(ValueError, match="boom"):
            worker._separate_stems_required(_silence(), _settings())


# ---------------------------------------------------------------------------
# _run_fast_stage — 1단계, 분리 없음
# ---------------------------------------------------------------------------


class TestRunFastStage:
    def test_uses_omniasr_no_separation_no_pron_data(self, monkeypatch):
        results = [SyncResult(text="hi", start_time=0.0, end_time=1.0, confidence=0.5)]
        anchor = _FakeAnchor(results)
        recorded: dict[str, Any] = {}

        def fake_get_engine(engine_type, config=None):
            recorded["type"] = engine_type
            return anchor

        monkeypatch.setattr(
            "everyric2.alignment.factory.EngineFactory.get_engine", fake_get_engine
        )
        lyric_lines = [LyricLine(text="hi", line_number=1)]
        stack = worker._run_fast_stage(_silence(), lyric_lines, "en", _settings())

        assert recorded["type"] == "omniasr"
        assert stack.alignment_text == "omniasr-fast"
        assert stack.pron_data == {}
        assert stack.vad_regions is None
        assert stack.sep_result is None
        assert stack.adlib is None

    def test_unavailable_anchor_raises(self, monkeypatch):
        anchor = _FakeAnchor([], available=False)
        monkeypatch.setattr(
            "everyric2.alignment.factory.EngineFactory.get_engine",
            lambda engine_type, config=None: anchor,
        )
        with pytest.raises(RuntimeError, match="not available"):
            worker._run_fast_stage(
                _silence(), [LyricLine(text="a", line_number=1)], "en", _settings()
            )


# ---------------------------------------------------------------------------
# _run_rescue_stage — 2단계, 분리 필수 + 조용한 폴백 금지
# ---------------------------------------------------------------------------


class TestRunRescueStage:
    def test_separator_unavailable_raises_not_silently_falls_back(self, monkeypatch):
        anchor = _FakeAnchor([SyncResult(text="a", start_time=0.0, end_time=1.0)])
        monkeypatch.setattr(
            "everyric2.alignment.factory.EngineFactory.get_engine",
            lambda engine_type, config=None: anchor,
        )
        fake_sep = _FakeSeparator(available=False)
        monkeypatch.setattr(
            "everyric2.audio.separator.get_shared_separator", lambda config=None: fake_sep
        )
        with pytest.raises(RuntimeError, match="bs-polarformer-fp16"):
            worker._run_rescue_stage(
                _silence(), None, [LyricLine(text="a", line_number=1)], "ja",
                _settings(), lambda s: None, "owsm",
            )
        assert anchor.align_calls == []

    def test_picks_owsm_for_non_en_and_omniasr_for_en(self, monkeypatch):
        anchor_calls: list[str] = []

        def fake_get_engine(engine_type, config=None):
            anchor_calls.append(engine_type)
            return _FakeAnchor([SyncResult(text="a", start_time=0.0, end_time=0.5)])

        monkeypatch.setattr(
            "everyric2.alignment.factory.EngineFactory.get_engine", fake_get_engine
        )
        sep = _FakeSepResult(_silence(), _silence())
        settings = _settings(two_pass_enabled=False)
        worker._run_rescue_stage(
            _silence(), sep, [LyricLine(text="a", line_number=1)], "ja",
            settings, lambda s: None, "owsm",
        )
        worker._run_rescue_stage(
            _silence(), sep, [LyricLine(text="a", line_number=1)], "en",
            settings, lambda s: None, "omniasr",
        )
        assert anchor_calls == ["owsm", "omniasr"]

    def test_lazily_separates_when_sep_result_not_given(self, monkeypatch):
        anchor = _FakeAnchor([SyncResult(text="a", start_time=0.0, end_time=0.5)])
        monkeypatch.setattr(
            "everyric2.alignment.factory.EngineFactory.get_engine",
            lambda engine_type, config=None: anchor,
        )
        sep = _FakeSepResult(_silence(), _silence())
        fake_sep = _FakeSeparator(available=True, sep_result=sep)
        monkeypatch.setattr(
            "everyric2.audio.separator.get_shared_separator", lambda config=None: fake_sep
        )
        stack = worker._run_rescue_stage(
            _silence(), None, [LyricLine(text="a", line_number=1)], "ja",
            _settings(two_pass_enabled=False), lambda s: None, "owsm",
        )
        assert fake_sep.separate_calls == 1
        assert stack.sep_result is sep
        assert stack.alignment_text == "owsm-rescue"

    def test_two_pass_enabled_but_refiner_unavailable_raises(self, monkeypatch):
        anchor = _FakeAnchor([SyncResult(text="a", start_time=0.0, end_time=0.5)])
        refiner = _RecordingEngine(None, available=False)

        def fake_get_engine(engine_type, config=None):
            return anchor if engine_type == "owsm" else refiner

        monkeypatch.setattr(
            "everyric2.alignment.factory.EngineFactory.get_engine", fake_get_engine
        )
        sep = _FakeSepResult(_silence(), _silence())
        with pytest.raises(RuntimeError, match="refiner not available"):
            worker._run_rescue_stage(
                _silence(), sep, [LyricLine(text="a", line_number=1)], "ja",
                _settings(two_pass_enabled=True), lambda s: None, "owsm",
            )

    def test_two_pass_disabled_is_a_legitimate_choice_not_a_failure(self, monkeypatch):
        anchor = _FakeAnchor([SyncResult(text="a", start_time=0.0, end_time=0.5)])
        monkeypatch.setattr(
            "everyric2.alignment.factory.EngineFactory.get_engine",
            lambda engine_type, config=None: anchor,
        )
        sep = _FakeSepResult(_silence(), _silence())
        stack = worker._run_rescue_stage(
            _silence(), sep, [LyricLine(text="a", line_number=1)], "ja",
            _settings(two_pass_enabled=False), lambda s: None, "owsm",
        )
        assert stack.pron_data == {}
        assert stack.alignment_text == "owsm-rescue"

    def test_refine_populates_multi_script_pron_data(self, monkeypatch):
        vocab = {"가": 1}
        frame_sec = 0.02
        logits = torch.full((1, 4, 2), -8.0)
        logits[0, 1, 1] = 8.0
        emission = torch.log_softmax(logits, dim=-1)
        fake_emission = EngineEmission(
            emission=emission, blank_id=0, frame_sec=frame_sec, audio_sec=0.08, chunks=1,
            vocab=vocab,
        )
        anchor_result = SyncResult(text="가", start_time=0.0, end_time=0.08, confidence=0.9)
        anchor = _FakeAnchor([anchor_result])
        refiner_engine = _RecordingEngine(fake_emission)

        def fake_get_engine(engine_type, config=None):
            return anchor if engine_type == "owsm" else refiner_engine

        monkeypatch.setattr(
            "everyric2.alignment.factory.EngineFactory.get_engine", fake_get_engine
        )
        sep = _FakeSepResult(_silence(0.5), _silence(0.5))
        stack = worker._run_rescue_stage(
            _silence(0.5), sep, [LyricLine(text="가", line_number=1)], "ja",
            _settings(two_pass_enabled=True), lambda s: None, "owsm",
        )
        assert stack.alignment_text == "owsm-rescue-2pass"
        assert 0 in stack.pron_data
        entry = stack.pron_data[0]
        assert entry["pron"]["hangul"]
        assert entry["pronunciation"] == entry["pron"]["hangul"]
        assert entry["pron_segs"]["hangul"]
        assert entry["pron_segments"] == entry["pron_segs"]["hangul"]

    def test_refine_lines_failure_propagates_not_silently_dropped(self, monkeypatch):
        anchor = _FakeAnchor([SyncResult(text="가", start_time=0.0, end_time=0.08)])
        refiner_engine = _RecordingEngine(None)

        def fake_get_engine(engine_type, config=None):
            return anchor if engine_type == "owsm" else refiner_engine

        monkeypatch.setattr(
            "everyric2.alignment.factory.EngineFactory.get_engine", fake_get_engine
        )

        def _raise(*a, **kw):
            raise RuntimeError("refine exploded")

        monkeypatch.setattr("everyric2.alignment.refine_window.refine_lines", _raise)
        sep = _FakeSepResult(_silence(0.5), _silence(0.5))
        with pytest.raises(RuntimeError, match="refine exploded"):
            worker._run_rescue_stage(
                _silence(0.5), sep, [LyricLine(text="가", line_number=1)], "ja",
                _settings(two_pass_enabled=True), lambda s: None, "owsm",
            )


# ---------------------------------------------------------------------------
# _run_new_stack_alignment — 3단계 라우팅 분기
# ---------------------------------------------------------------------------


class TestRoutingDecision:
    def test_high_confidence_stays_on_fast_path(self, monkeypatch):
        results = [SyncResult(text="a", start_time=0.0, end_time=1.0, confidence=0.9)]
        anchor = _FakeAnchor(results)
        calls: list[str] = []

        def fake_get_engine(engine_type, config=None):
            calls.append(engine_type)
            return anchor

        monkeypatch.setattr(
            "everyric2.alignment.factory.EngineFactory.get_engine", fake_get_engine
        )
        stack = worker._run_new_stack_alignment(
            _silence(), None, [LyricLine(text="a", line_number=1)], "ja",
            _settings(), lambda s: None,
        )
        assert calls == ["omniasr"]
        assert stack.alignment_text == "omniasr-fast"
        assert stack.routing_meta["route"] == "fast"

    def test_low_confidence_escalates_to_owsm_rescue(self, monkeypatch):
        fast_results = [SyncResult(text="a", start_time=0.0, end_time=1.0, confidence=1e-9)]
        fast_anchor = _FakeAnchor(fast_results)
        rescue_anchor = _FakeAnchor([SyncResult(text="a", start_time=0.0, end_time=1.0)])
        calls: list[str] = []

        def fake_get_engine(engine_type, config=None):
            calls.append(engine_type)
            return fast_anchor if len(calls) == 1 else rescue_anchor

        monkeypatch.setattr(
            "everyric2.alignment.factory.EngineFactory.get_engine", fake_get_engine
        )
        sep = _FakeSepResult(_silence(), _silence())
        monkeypatch.setattr(
            "everyric2.audio.separator.get_shared_separator",
            lambda config=None: _FakeSeparator(True, sep),
        )
        stack = worker._run_new_stack_alignment(
            _silence(), None, [LyricLine(text="a", line_number=1)], "ja",
            _settings(two_pass_enabled=False), lambda s: None,
        )
        assert calls == ["omniasr", "owsm"]
        assert stack.alignment_text == "owsm-rescue"
        assert stack.routing_meta["route"] == "rescue"

    def test_en_skips_fast_path_entirely(self, monkeypatch):
        rescue_anchor = _FakeAnchor([SyncResult(text="a", start_time=0.0, end_time=1.0)])
        calls: list[str] = []

        def fake_get_engine(engine_type, config=None):
            calls.append(engine_type)
            return rescue_anchor

        monkeypatch.setattr(
            "everyric2.alignment.factory.EngineFactory.get_engine", fake_get_engine
        )
        sep = _FakeSepResult(_silence(), _silence())
        monkeypatch.setattr(
            "everyric2.audio.separator.get_shared_separator",
            lambda config=None: _FakeSeparator(True, sep),
        )
        stack = worker._run_new_stack_alignment(
            _silence(), None, [LyricLine(text="a", line_number=1)], "en",
            _settings(two_pass_enabled=False), lambda s: None,
        )
        assert calls == ["omniasr"]  # 구원의 자기앵커 한 번뿐 — fast 단계 자체가 안 돌았다
        assert stack.alignment_text == "omniasr-rescue"
        assert stack.routing_meta["route"] == "forced"

    def test_en_stranded_escalation_adopted_when_it_improves(self, monkeypatch):
        # _stranded_count 자체(display_fixes._stranded_sites 경유)는 test_display_fixes.py가
        # 이미 검증한다 — 여기서는 _run_new_stack_alignment의 "승급 시도 -> 개선되면 채택"
        # 결정 로직만 본다. apply_stranded_corrections도 내부에서 _stranded_sites를 한 번 더
        # 부르므로(display_fixes.py 자체 로직), 그 호출까지 같은 이터레이터를 공유시키면
        # 순서가 꼬인다 — 그래서 _stranded_count 자체를 목으로 대체해 그 문제를 피한다.
        rescue_calls: list[str] = []

        def fake_get_engine(engine_type, config=None):
            rescue_calls.append(engine_type)
            return _FakeAnchor([SyncResult(text="a", start_time=0.0, end_time=1.0)])

        monkeypatch.setattr(
            "everyric2.alignment.factory.EngineFactory.get_engine", fake_get_engine
        )
        sep = _FakeSepResult(_silence(), _silence())
        monkeypatch.setattr(
            "everyric2.audio.separator.get_shared_separator",
            lambda config=None: _FakeSeparator(True, sep),
        )

        counts = iter([3, 1])  # 첫 구원(omniasr)=3 잔존 -> 승급 -> owsm=1 잔존(개선)
        monkeypatch.setattr(worker, "_stranded_count", lambda stack: next(counts))

        stack = worker._run_new_stack_alignment(
            _silence(), None, [LyricLine(text="a", line_number=1)], "en",
            _settings(two_pass_enabled=False), lambda s: None,
        )
        assert rescue_calls == ["omniasr", "owsm"]
        assert stack.alignment_text == "owsm-rescue-escalated"
        assert stack.routing_meta["route"] == "escalated"
        assert stack.routing_meta["stranded_before"] == 3
        assert stack.routing_meta["stranded_after"] == 1

    def test_en_stranded_escalation_rejected_when_it_does_not_improve(self, monkeypatch):
        rescue_calls: list[str] = []

        def fake_get_engine(engine_type, config=None):
            rescue_calls.append(engine_type)
            return _FakeAnchor([SyncResult(text="a", start_time=0.0, end_time=1.0)])

        monkeypatch.setattr(
            "everyric2.alignment.factory.EngineFactory.get_engine", fake_get_engine
        )
        sep = _FakeSepResult(_silence(), _silence())
        monkeypatch.setattr(
            "everyric2.audio.separator.get_shared_separator",
            lambda config=None: _FakeSeparator(True, sep),
        )
        counts = iter([2, 2])  # 승급해도 안 줄어듦 -> 기각
        monkeypatch.setattr(worker, "_stranded_count", lambda stack: next(counts))

        stack = worker._run_new_stack_alignment(
            _silence(), None, [LyricLine(text="a", line_number=1)], "en",
            _settings(two_pass_enabled=False), lambda s: None,
        )
        assert rescue_calls == ["omniasr", "owsm"]
        assert stack.alignment_text == "omniasr-rescue"
        assert stack.routing_meta["stranded_before"] == 2
        assert stack.routing_meta["stranded_after"] == 2


# ---------------------------------------------------------------------------
# _finish_new_stack_alignment — 응답 dict 조립(멜로디/품질 없이)
# ---------------------------------------------------------------------------


class TestFinishNewStackAlignment:
    def test_response_shape_routing_and_adlib_additive_field(self, monkeypatch):
        # ja + 고신뢰(0.9) -> 라우팅이 고속 단계에서 끝나 분리가 전혀 필요 없다(en은 강제
        # 구원이라 분리 목이 또 필요해진다 — 별도 관심사라 여기서는 안 섞는다).
        result = SyncResult(
            text="hi", start_time=0.0, end_time=1.0, confidence=0.9,
            word_segments=[WordSegment(word="hi", start=0.0, end=1.0, confidence=0.9)],
        )
        anchor = _FakeAnchor([result])
        monkeypatch.setattr(
            "everyric2.alignment.factory.EngineFactory.get_engine",
            lambda engine_type, config=None: anchor,
        )
        lyric_lines = [LyricLine(text="hi", line_number=1)]
        settings = _settings()
        audio = _silence(0.5)

        out = worker._finish_new_stack_alignment(
            audio, None, None, lyric_lines, "ja", settings, lambda s: None,
            gloss_folded=None, melody_extractor=None, f0_future=None, f0_executor=None,
        )

        assert out["timestamps"][0]["text"] == "hi"
        assert out["alignment_text"] == "omniasr-fast"
        assert out["debug"]["alignment_text"] == "omniasr-fast"
        assert out["debug"]["star_spans"] == []
        assert out["debug"]["routing"]["route"] == "fast"
        assert out["adlib"] is None
        assert "key" in out and "tempo" in out
