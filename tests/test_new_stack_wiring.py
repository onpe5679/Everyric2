"""새 정렬 스택(3단계 라우팅 + owsm/omniasr 앵커 + 2패스 리파이너) 배선 회귀 테스트.

이식된 부품 자체(owsm_engine/omniasr_engine/refine_window/display_fixes)는 각자의
테스트 스위트(test_alignment_new_engines_wiring.py·test_refine_window.py·
test_display_fixes.py)가 이미 못박고 있다. 라우팅 자체(scripts/bench_adapters/routed.py의
line_log_conf_median·문턱값)도 벤치 쪽에서 실측됐다. 이 파일은 그것들을 ``everyric2/
server/worker.py``에 실제로 배선한 층만 검증한다: 설정 정합성 가드, 3단계 라우팅
(fast→medium/heavy→en 좌초 heavy 승급)의 분기 로직, refine_window(Path 계약) ↔ BaseAlignmentEngine.
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
# _warn_ignored_legacy_settings — 새 스택 + 안 배선된 레거시 스위치 켜짐 -> 경고 로그
# ---------------------------------------------------------------------------


class _FakeAlignmentSettings:
    def __init__(self, **flags: bool) -> None:
        self.engine = "owsm"
        self.caption_anchors = flags.get("caption_anchors", False)
        self.caption_scaffold = flags.get("caption_scaffold", False)
        self.star_prior = flags.get("star_prior", False)
        self.star_tokens = flags.get("star_tokens", False)


class TestWarnIgnoredLegacySettings:
    def test_no_warning_when_all_switches_off(self, caplog):
        settings = SimpleNamespace(alignment=_FakeAlignmentSettings())
        with caplog.at_level("WARNING"):
            worker._warn_ignored_legacy_settings(settings)
        assert caplog.records == []

    def test_warns_once_listing_every_ignored_switch(self, caplog):
        settings = SimpleNamespace(
            alignment=_FakeAlignmentSettings(caption_anchors=True, star_tokens=True)
        )
        with caplog.at_level("WARNING"):
            worker._warn_ignored_legacy_settings(settings)
        assert len(caplog.records) == 1
        message = caplog.records[0].getMessage()
        assert "caption_anchors" in message
        assert "star_tokens" in message
        # 안 켠 스위치는 언급하지 않는다
        assert "caption_scaffold(" not in message
        assert "star_prior(" not in message


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
# _separate_stems_required — medium/heavy 단계 전용, 조용한 폴백 금지
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
# _run_fast_stage — fast 깊이, 분리 없음
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
        assert stack.alignment_text == "fast"
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
# _run_deep_stage — medium/heavy 깊이, 분리 필수 + 조용한 폴백 금지
# ---------------------------------------------------------------------------


class TestRunDeepStage:
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
            worker._run_deep_stage(
                _silence(), None, [LyricLine(text="a", line_number=1)], "ja",
                _settings(), lambda s: None, "heavy",
            )
        assert anchor.align_calls == []

    def test_picks_owsm_for_heavy_and_omniasr_for_medium(self, monkeypatch):
        anchor_calls: list[str] = []

        def fake_get_engine(engine_type, config=None):
            anchor_calls.append(engine_type)
            return _FakeAnchor([SyncResult(text="a", start_time=0.0, end_time=0.5)])

        monkeypatch.setattr(
            "everyric2.alignment.factory.EngineFactory.get_engine", fake_get_engine
        )
        sep = _FakeSepResult(_silence(), _silence())
        settings = _settings(two_pass_enabled=False)
        worker._run_deep_stage(
            _silence(), sep, [LyricLine(text="a", line_number=1)], "ja",
            settings, lambda s: None, "heavy",
        )
        worker._run_deep_stage(
            _silence(), sep, [LyricLine(text="a", line_number=1)], "en",
            settings, lambda s: None, "medium",
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
        stack = worker._run_deep_stage(
            _silence(), None, [LyricLine(text="a", line_number=1)], "ja",
            _settings(two_pass_enabled=False), lambda s: None, "heavy",
        )
        assert fake_sep.separate_calls == 1
        assert stack.sep_result is sep
        assert stack.alignment_text == "heavy"

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
            worker._run_deep_stage(
                _silence(), sep, [LyricLine(text="a", line_number=1)], "ja",
                _settings(two_pass_enabled=True), lambda s: None, "heavy",
            )

    def test_two_pass_disabled_is_a_legitimate_choice_not_a_failure(self, monkeypatch):
        anchor = _FakeAnchor([SyncResult(text="a", start_time=0.0, end_time=0.5)])
        monkeypatch.setattr(
            "everyric2.alignment.factory.EngineFactory.get_engine",
            lambda engine_type, config=None: anchor,
        )
        sep = _FakeSepResult(_silence(), _silence())
        stack = worker._run_deep_stage(
            _silence(), sep, [LyricLine(text="a", line_number=1)], "ja",
            _settings(two_pass_enabled=False), lambda s: None, "heavy",
        )
        assert stack.pron_data == {}
        assert stack.alignment_text == "heavy"

    def test_refine_populates_multi_script_pron_data(self, monkeypatch):
        # "가"(순한글 한 글자)는 F1(2026-08-04 감사) 이후 refine_window가 owners 파생
        # 자체를 건너뛰는 입력이 됐다(한글 우세 → fallback_reason="non_derivable_script",
        # fast 경로에 위임) — 이 테스트의 실제 관심사(refine_lines가 낸 표기별 pron/
        # pron_segs가 pron_data 진입점까지 하나도 안 빠지고 실리는가, 결함 수정
        # 2026-08-03)와는 무관한 입력이었다. en 갈래(라틴)로 픽스처를 바꿔 같은 관심사를
        # 계속 지킨다 — "cat" -> CMU IPA "kat"(test_refine_window.py와 같은 패턴).
        vocab = {"k": 1, "a": 2, "t": 3}
        frame_sec = 0.02
        frames_per_char = 5
        token_ids: list[int | None] = []
        for ch in "kat":
            token_ids.extend([vocab[ch]] * frames_per_char)
        token_ids.extend([None] * 10)
        logits = torch.full((1, len(token_ids), len(vocab) + 1), -8.0)
        for t, tid in enumerate(token_ids):
            logits[0, t, tid if tid is not None else 0] = 8.0
        emission = torch.log_softmax(logits, dim=-1)
        line_end = len(token_ids) * frame_sec
        fake_emission = EngineEmission(
            emission=emission, blank_id=0, frame_sec=frame_sec, audio_sec=line_end, chunks=1,
            vocab=vocab,
        )
        anchor_result = SyncResult(text="cat", start_time=0.0, end_time=line_end, confidence=0.9)
        anchor = _FakeAnchor([anchor_result])
        refiner_engine = _RecordingEngine(fake_emission)

        def fake_get_engine(engine_type, config=None):
            return anchor if engine_type == "owsm" else refiner_engine

        monkeypatch.setattr(
            "everyric2.alignment.factory.EngineFactory.get_engine", fake_get_engine
        )
        sep = _FakeSepResult(_silence(0.5), _silence(0.5))
        stack = worker._run_deep_stage(
            _silence(0.5), sep, [LyricLine(text="cat", line_number=1)], "ja",
            _settings(two_pass_enabled=True), lambda s: None, "heavy",
        )
        assert stack.alignment_text == "heavy-2pass"
        assert 0 in stack.pron_data
        entry = stack.pron_data[0]
        assert entry["pron"]["hangul"]
        assert entry["pronunciation"] == entry["pron"]["hangul"]
        assert entry["pron_segs"]["hangul"]
        assert entry["pron_segments"] == entry["pron_segs"]["hangul"]
        # 결함 수정(2026-08-03): "표기 4종 중 3종이 저장 단계에서 유실된다"는 실사용자
        # 보고(weathergirl/M7VSEZOQIlg)를 못박는다 — refine_lines가 낸 표기별 pron/
        # pron_segs가 하나도 안 빠지고 pron_data 진입점(entry)까지 전부 실려야 한다.
        for key in ("hangul", "kana", "romaji", "en"):
            assert entry["pron"].get(key), f"pron[{key!r}] missing or empty"
            assert entry["pron_segs"].get(key), f"pron_segs[{key!r}] missing or empty"

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
            worker._run_deep_stage(
                _silence(0.5), sep, [LyricLine(text="가", line_number=1)], "ja",
                _settings(two_pass_enabled=True), lambda s: None, "heavy",
            )


# ---------------------------------------------------------------------------
# _run_deep_stage — 라인별 2패스 fallback_reason이 조용히 버려지지 않는지
#
# refine_window.refine_lines는 라인 단위 실패를 예외가 아니라 RefinedLine.fallback_reason
# 으로 신호한다(그 모듈의 "앵커·리파이너 계약" — 호출부가 이 신호를 보고 앵커 세그로
# 폴백하라는 뜻이다). 신호 자체가 나오는지는 test_refine_window.py가 이미 못박고 있다 —
# 여기서는 그 신호를 worker.py가 실제로 읽어 로그로 남기는지만 본다(감사 발견,
# 2026-08-04: 신호가 나와도 아무도 안 읽으면 "왜 이 줄만 근사 발음인지" 아무 데도 안 남는다).
# ---------------------------------------------------------------------------


class TestRunDeepStageLineFallbackVisibility:
    def test_line_fallback_reason_is_logged_not_silently_dropped(self, monkeypatch, caplog):
        from everyric2.alignment.refine_window import RefinedLine

        anchor_results = [
            SyncResult(text="가", start_time=0.0, end_time=0.5),
            SyncResult(text="나", start_time=0.5, end_time=1.0),
        ]
        anchor = _FakeAnchor(anchor_results)
        refiner_engine = _RecordingEngine(object())

        def fake_get_engine(engine_type, config=None):
            return anchor if engine_type == "owsm" else refiner_engine

        monkeypatch.setattr(
            "everyric2.alignment.factory.EngineFactory.get_engine", fake_get_engine
        )

        # 라인0은 정상 리파인, 라인1은 fallback_reason만 걸리고 pron/pron_segs가 비어 있다
        # (실제 refine_lines가 "window_shorter_than_targets" 등에서 만드는 모양 그대로).
        refined = [
            RefinedLine(start=0.0, end=0.5, pron={"hangul": "가"}, refined=True),
            RefinedLine(start=0.5, end=1.0, fallback_reason="window_shorter_than_targets"),
        ]
        monkeypatch.setattr(
            "everyric2.alignment.refine_window.refine_lines", lambda *a, **kw: refined
        )
        sep = _FakeSepResult(_silence(0.5), _silence(0.5))
        lyric_lines = [
            LyricLine(text="가", line_number=1),
            LyricLine(text="나", line_number=2),
        ]
        with caplog.at_level("WARNING"):
            stack = worker._run_deep_stage(
                _silence(0.5), sep, lyric_lines, "ja",
                _settings(two_pass_enabled=True), lambda s: None, "heavy",
            )

        # 정상 리파인 라인만 pron_data에 실린다 — fallback 라인은 조용히 빠지되(그 자체는
        # 설계된 동작, attach_pron_variants가 나중에 근사 발음을 채운다) ...
        assert 0 in stack.pron_data
        assert 1 not in stack.pron_data
        # ... 그 사실이 최소한 로그에는 남아야 한다 — 조용히 사라지면 안 된다.
        messages = [r.getMessage() for r in caplog.records]
        assert any("window_shorter_than_targets" in m for m in messages)

    def test_no_warning_when_every_line_refines_cleanly(self, monkeypatch, caplog):
        from everyric2.alignment.refine_window import RefinedLine

        anchor = _FakeAnchor([SyncResult(text="가", start_time=0.0, end_time=0.5)])
        refiner_engine = _RecordingEngine(object())

        def fake_get_engine(engine_type, config=None):
            return anchor if engine_type == "owsm" else refiner_engine

        monkeypatch.setattr(
            "everyric2.alignment.factory.EngineFactory.get_engine", fake_get_engine
        )
        refined = [RefinedLine(start=0.0, end=0.5, pron={"hangul": "가"}, refined=True)]
        monkeypatch.setattr(
            "everyric2.alignment.refine_window.refine_lines", lambda *a, **kw: refined
        )
        sep = _FakeSepResult(_silence(0.5), _silence(0.5))
        with caplog.at_level("WARNING"):
            worker._run_deep_stage(
                _silence(0.5), sep, [LyricLine(text="가", line_number=1)], "ja",
                _settings(two_pass_enabled=True), lambda s: None, "heavy",
            )
        messages = [r.getMessage() for r in caplog.records]
        assert not any("fell back to anchor-only" in m for m in messages)


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
        assert stack.alignment_text == "fast"
        assert stack.routing_meta["route"] == "fast"

    def test_none_score_escalates_to_heavy_not_fast(self, monkeypatch):
        # 판정 불가(confidence를 하나도 못 구함, score=None)는 "확신 있는 정상곡"이
        # 아니라 안전한 쪽(heavy)으로 떨어져야 한다 — routed.py:228과 같은 방향
        # (`if score is not None and score >= threshold: return fast`). 2026-08-03
        # 실곡 검증에서 이 방향이 뒤집혀 있던 결함을 그대로 못박는다.
        fast_results = [SyncResult(text="a", start_time=0.0, end_time=1.0, confidence=None)]
        fast_anchor = _FakeAnchor(fast_results)
        deep_anchor = _FakeAnchor([SyncResult(text="a", start_time=0.0, end_time=1.0)])
        calls: list[str] = []

        def fake_get_engine(engine_type, config=None):
            calls.append(engine_type)
            return fast_anchor if len(calls) == 1 else deep_anchor

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
        assert calls == ["omniasr", "owsm"]  # 고속만으로 안 끝나고 heavy까지 갔다
        assert stack.alignment_text == "heavy"
        assert stack.routing_meta["route"] == "heavy"
        assert stack.routing_meta["line_log_conf_median"] is None

    @pytest.mark.parametrize(
        "log_conf_values,expect_heavy",
        [
            # 벤치 실측 극한곡 대역(熱異常·토스트·소실·시니컬·루프더룸): -13.82 ~ -12.02
            pytest.param([-13.0, -12.5, -12.9], True, id="extreme-band"),
            # 벤치 실측 정상곡 대역: -11.80 ~ -4.36
            pytest.param([-11.7, -8.0, -4.5], False, id="normal-band"),
        ],
    )
    def test_measured_bench_bands_route_correctly(
        self, monkeypatch, log_conf_values, expect_heavy
    ):
        import math

        fast_results = [
            SyncResult(text=f"line{i}", start_time=float(i), end_time=float(i + 1),
                       confidence=math.exp(v))
            for i, v in enumerate(log_conf_values)
        ]
        fast_anchor = _FakeAnchor(fast_results)
        deep_anchor = _FakeAnchor([SyncResult(text="a", start_time=0.0, end_time=1.0)])
        calls: list[str] = []

        def fake_get_engine(engine_type, config=None):
            calls.append(engine_type)
            return fast_anchor if len(calls) == 1 else deep_anchor

        monkeypatch.setattr(
            "everyric2.alignment.factory.EngineFactory.get_engine", fake_get_engine
        )
        sep = _FakeSepResult(_silence(), _silence())
        monkeypatch.setattr(
            "everyric2.audio.separator.get_shared_separator",
            lambda config=None: _FakeSeparator(True, sep),
        )
        stack = worker._run_new_stack_alignment(
            _silence(), None,
            [LyricLine(text=f"line{i}", line_number=i + 1) for i in range(len(log_conf_values))],
            "ja", _settings(two_pass_enabled=False), lambda s: None,
        )
        if expect_heavy:
            assert calls == ["omniasr", "owsm"]
            assert stack.alignment_text == "heavy"
        else:
            assert calls == ["omniasr"]
            assert stack.alignment_text == "fast"

    def test_low_confidence_escalates_to_heavy(self, monkeypatch):
        fast_results = [SyncResult(text="a", start_time=0.0, end_time=1.0, confidence=1e-9)]
        fast_anchor = _FakeAnchor(fast_results)
        deep_anchor = _FakeAnchor([SyncResult(text="a", start_time=0.0, end_time=1.0)])
        calls: list[str] = []

        def fake_get_engine(engine_type, config=None):
            calls.append(engine_type)
            return fast_anchor if len(calls) == 1 else deep_anchor

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
        assert stack.alignment_text == "heavy"
        assert stack.routing_meta["route"] == "heavy"

    def test_en_skips_fast_path_entirely(self, monkeypatch):
        deep_anchor = _FakeAnchor([SyncResult(text="a", start_time=0.0, end_time=1.0)])
        calls: list[str] = []

        def fake_get_engine(engine_type, config=None):
            calls.append(engine_type)
            return deep_anchor

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
        assert calls == ["omniasr"]  # medium의 자기앵커 한 번뿐 — fast 단계 자체가 안 돌았다
        assert stack.alignment_text == "medium"
        assert stack.routing_meta["route"] == "medium"

    def test_en_stranded_escalation_adopted_when_it_improves(self, monkeypatch):
        # _stranded_count 자체(display_fixes._stranded_sites 경유)는 test_display_fixes.py가
        # 이미 검증한다 — 여기서는 _run_new_stack_alignment의 "승급 시도 -> 개선되면 채택"
        # 결정 로직만 본다. apply_stranded_corrections도 내부에서 _stranded_sites를 한 번 더
        # 부르므로(display_fixes.py 자체 로직), 그 호출까지 같은 이터레이터를 공유시키면
        # 순서가 꼬인다 — 그래서 _stranded_count 자체를 목으로 대체해 그 문제를 피한다.
        depth_calls: list[str] = []

        def fake_get_engine(engine_type, config=None):
            depth_calls.append(engine_type)
            return _FakeAnchor([SyncResult(text="a", start_time=0.0, end_time=1.0)])

        monkeypatch.setattr(
            "everyric2.alignment.factory.EngineFactory.get_engine", fake_get_engine
        )
        sep = _FakeSepResult(_silence(), _silence())
        # 매 get_shared_separator 호출마다 새 인스턴스를 만들지 않는다 — medium->heavy
        # 승급이 분리를 재사용하는지 확인하려면 separate_calls 카운터가 호출 전체에
        # 걸쳐 하나로 이어져야 한다.
        fake_separator = _FakeSeparator(True, sep)
        monkeypatch.setattr(
            "everyric2.audio.separator.get_shared_separator",
            lambda config=None: fake_separator,
        )

        counts = iter([3, 1])  # 첫 medium(omniasr)=3 잔존 -> 승급 -> heavy(owsm)=1 잔존(개선)
        monkeypatch.setattr(worker, "_stranded_count", lambda stack: next(counts))

        stack = worker._run_new_stack_alignment(
            _silence(), None, [LyricLine(text="a", line_number=1)], "en",
            _settings(two_pass_enabled=False), lambda s: None,
        )
        assert depth_calls == ["omniasr", "owsm"]
        assert stack.alignment_text == "heavy-escalated"
        assert stack.routing_meta["route"] == "heavy"
        assert stack.routing_meta["stranded_before"] == 3
        assert stack.routing_meta["stranded_after"] == 1
        # medium->heavy 승급은 이미 분리된 스템(deep.sep_result)을 그대로 넘겨 받아야
        # 한다(운영자 지시, 2026-08-04) — 물리적으로 owsm 앵커만 추가로 돌면 된다. 분리를
        # 두 번 했다면 이 카운터가 2가 된다.
        assert fake_separator.separate_calls == 1

    def test_en_stranded_escalation_rejected_when_it_does_not_improve(self, monkeypatch):
        depth_calls: list[str] = []

        def fake_get_engine(engine_type, config=None):
            depth_calls.append(engine_type)
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
        assert depth_calls == ["omniasr", "owsm"]
        assert stack.alignment_text == "medium"
        assert stack.routing_meta["stranded_before"] == 2
        assert stack.routing_meta["stranded_after"] == 2


# ---------------------------------------------------------------------------
# _resolve_stack_language — 라벨이 비면 문자 계열 우세 판정 (weathergirl 결함, 2026-08-03)
#
# jobs.language는 실측상 자주 None이다(사용자 생성 잡 4/4). 라벨 하나가 비면 ① en 강제
# medium 진입 무산(전부 fast로 샘), ② 2패스 리파이너 language or "en" 오분기, ③ 응답
# language=None까지 세 layer가 동시에 갈라졌다 — 한 번 판정한 값이 세 곳에 같이 흘러야
# 한다.
# ---------------------------------------------------------------------------


class TestResolveStackLanguage:
    def test_label_passes_through_normalized(self):
        assert worker._resolve_stack_language(" JA ", []) == ("ja", "label")

    def test_regional_subtags_are_stripped_from_labels(self):
        # 코덱스 감사 Med: "EN-us"류는 라우팅(startswith)은 통과하지만 OWSM 언어 심볼
        # 매칭(정확 키)이 실패해 무언어 정렬로 조용히 저하됐다 — 기본 코드만 남긴다.
        assert worker._resolve_stack_language("EN-us", []) == ("en", "label")
        assert worker._resolve_stack_language("zh_TW", []) == ("zh", "label")

    def test_empty_label_pure_latin_resolves_en(self):
        lines = [LyricLine(text="the weathergirl says sunshine again", line_number=1)]
        assert worker._resolve_stack_language(None, lines) == ("en", "script_census")

    def test_empty_label_latin_majority_with_kana_resolves_ja(self):
        # numb numb 실측: 라틴 53.7%인데 ja 곡 — "많은 쪽"이 아니라 vocab이 "덮는 쪽"
        # 판정이어야 en 오진이 없다(ctc_engine._pick_by_coverage).
        lines = [LyricLine(text="ナムナム baby yeah we go party tonight", line_number=1)]
        assert worker._resolve_stack_language(None, lines) == ("ja", "script_census")

    def test_empty_label_hangul_resolves_ko(self):
        lines = [LyricLine(text="보랏빛 기척", line_number=1)]
        assert worker._resolve_stack_language(None, lines) == ("ko", "script_census")


class TestRoutingLanguageResolution:
    def test_none_language_with_latin_lyrics_starts_at_medium(self, monkeypatch):
        # weathergirl(M7VSEZOQIlg) 실측 결함: language=None이면 "".startswith("en")=False로
        # 강제 medium 진입이 무산돼 en 곡이 전부 fast로 샜다.
        deep_anchor = _FakeAnchor([SyncResult(text="a", start_time=0.0, end_time=1.0)])
        calls: list[str] = []

        def fake_get_engine(engine_type, config=None):
            calls.append(engine_type)
            return deep_anchor

        monkeypatch.setattr(
            "everyric2.alignment.factory.EngineFactory.get_engine", fake_get_engine
        )
        sep = _FakeSepResult(_silence(), _silence())
        monkeypatch.setattr(
            "everyric2.audio.separator.get_shared_separator",
            lambda config=None: _FakeSeparator(True, sep),
        )
        stack = worker._run_new_stack_alignment(
            _silence(), None,
            [LyricLine(text="the weathergirl says sunshine again", line_number=1)],
            None, _settings(two_pass_enabled=False), lambda s: None,
        )
        assert calls == ["omniasr"]  # medium 자기앵커 한 번뿐 — fast 단계가 아예 안 돌았다
        assert stack.alignment_text == "medium"
        assert stack.routing_meta["route"] == "medium"
        assert stack.routing_meta["language"] == "en"
        assert stack.routing_meta["language_source"] == "script_census"

    def test_none_language_with_kana_mixed_lyrics_enters_fast_as_ja(self, monkeypatch):
        results = [SyncResult(text="a", start_time=0.0, end_time=1.0, confidence=0.9)]
        anchor = _FakeAnchor(results)
        monkeypatch.setattr(
            "everyric2.alignment.factory.EngineFactory.get_engine",
            lambda engine_type, config=None: anchor,
        )
        stack = worker._run_new_stack_alignment(
            _silence(), None,
            [LyricLine(text="ナムナム baby yeah we go party tonight", line_number=1)],
            None, _settings(), lambda s: None,
        )
        assert stack.alignment_text == "fast"
        assert stack.routing_meta["language"] == "ja"
        # 판정된 언어가 앵커까지 흘러야 한다 — 리파이너 분기(language or "en")와 owsm
        # 언어 심볼 선택의 재료이기도 하다.
        assert anchor.align_calls[0]["language"] == "ja"

    def test_label_wins_over_census(self, monkeypatch):
        # 라벨이 있으면 가사가 라틴이어도 추정이 덮지 않는다.
        results = [SyncResult(text="a", start_time=0.0, end_time=1.0, confidence=0.9)]
        anchor = _FakeAnchor(results)
        monkeypatch.setattr(
            "everyric2.alignment.factory.EngineFactory.get_engine",
            lambda engine_type, config=None: anchor,
        )
        stack = worker._run_new_stack_alignment(
            _silence(), None,
            [LyricLine(text="pure latin lyrics only here", line_number=1)],
            "ja", _settings(), lambda s: None,
        )
        assert stack.alignment_text == "fast"
        assert stack.routing_meta["language"] == "ja"
        assert stack.routing_meta["language_source"] == "label"


class TestMinDepthOverride:
    """분석 깊이 하한(확장 "분석 깊이 올리기" 버튼) — 라우터를 건너뛰고 요청 깊이 그대로."""

    def _mock_engines(self, monkeypatch):
        calls: list[str] = []

        def fake_get_engine(engine_type, config=None):
            calls.append(engine_type)
            return _FakeAnchor(
                [SyncResult(text="a", start_time=0.0, end_time=1.0, confidence=0.9)]
            )

        monkeypatch.setattr(
            "everyric2.alignment.factory.EngineFactory.get_engine", fake_get_engine
        )
        sep = _FakeSepResult(_silence(), _silence())
        monkeypatch.setattr(
            "everyric2.audio.separator.get_shared_separator",
            lambda config=None: _FakeSeparator(True, sep),
        )
        return calls

    def test_min_depth_heavy_skips_router_even_for_confident_ja(self, monkeypatch):
        # 고신뢰라 fast로 끝났을 곡도 min_depth=heavy면 라우터 없이 heavy로 바로 간다 —
        # 버튼의 배지 숫자와 결과 깊이가 일치해야 한다(예측 가능성).
        calls = self._mock_engines(monkeypatch)
        stack = worker._run_new_stack_alignment(
            _silence(), None, [LyricLine(text="a", line_number=1)], "ja",
            _settings(two_pass_enabled=False), lambda s: None, min_depth="heavy",
        )
        assert calls == ["owsm"]  # fast 단계(omniasr) 자체가 안 돌았다
        assert stack.alignment_text == "heavy"
        assert stack.routing_meta["route"] == "heavy"
        assert stack.routing_meta["requested_min_depth"] == "heavy"

    def test_min_depth_medium_runs_medium_even_for_ja(self, monkeypatch):
        # ja의 기본 사다리는 fast→heavy로 medium을 건너뛰지만, 명시 요청은 그대로 존중한다
        calls = self._mock_engines(monkeypatch)
        stack = worker._run_new_stack_alignment(
            _silence(), None, [LyricLine(text="a", line_number=1)], "ja",
            _settings(two_pass_enabled=False), lambda s: None, min_depth="medium",
        )
        assert calls == ["omniasr"]
        assert stack.alignment_text == "medium"
        assert stack.routing_meta["requested_min_depth"] == "medium"

    def test_no_min_depth_keeps_router(self, monkeypatch):
        calls = self._mock_engines(monkeypatch)
        stack = worker._run_new_stack_alignment(
            _silence(), None, [LyricLine(text="a", line_number=1)], "ja",
            _settings(), lambda s: None,
        )
        assert calls == ["omniasr"]
        assert stack.alignment_text == "fast"
        assert "requested_min_depth" not in stack.routing_meta


class TestFinishNewStackLanguageFallback:
    def test_payload_language_falls_back_to_census_when_label_missing(self, monkeypatch):
        # 새 앵커(owsm/omniasr)는 _current_language를 노출하지 않아 기존 폴백이 구조적으로
        # 죽어 있었다 — language=None 잡의 응답 language가 None으로 남았다(실측: 사용자
        # 생성 잡 4/4). 라우팅이 판정한 언어가 응답까지 흘러야 한다.
        result = SyncResult(text="こんにちは", start_time=0.0, end_time=1.0, confidence=0.9)
        anchor = _FakeAnchor([result])
        monkeypatch.setattr(
            "everyric2.alignment.factory.EngineFactory.get_engine",
            lambda engine_type, config=None: anchor,
        )
        out = worker._finish_new_stack_alignment(
            _silence(0.5), None, None,
            [LyricLine(text="こんにちは", line_number=1)],
            None, _settings(), lambda s: None,
            gloss_folded=None, melody_extractor=None, f0_future=None, f0_executor=None,
        )
        assert out["language"] == "ja"
        assert out["debug"]["routing"]["language_source"] == "script_census"

    def test_payload_language_keeps_label_when_present(self, monkeypatch):
        result = SyncResult(text="hi", start_time=0.0, end_time=1.0, confidence=0.9)
        anchor = _FakeAnchor([result])
        monkeypatch.setattr(
            "everyric2.alignment.factory.EngineFactory.get_engine",
            lambda engine_type, config=None: anchor,
        )
        out = worker._finish_new_stack_alignment(
            _silence(0.5), None, None, [LyricLine(text="hi", line_number=1)],
            "ja", _settings(), lambda s: None,
            gloss_folded=None, melody_extractor=None, f0_future=None, f0_executor=None,
        )
        assert out["language"] == "ja"


# ---------------------------------------------------------------------------
# _finish_new_stack_alignment — 응답 dict 조립(멜로디/품질 없이)
# ---------------------------------------------------------------------------


class TestFinishNewStackAlignment:
    def test_response_shape_routing_and_adlib_additive_field(self, monkeypatch):
        # ja + 고신뢰(0.9) -> 라우팅이 fast 깊이에서 끝나 분리가 전혀 필요 없다(en은 강제로
        # medium부터 시작해 분리 목이 또 필요해진다 — 별도 관심사라 여기서는 안 섞는다).
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
        assert out["alignment_text"] == "fast"
        assert out["debug"]["alignment_text"] == "fast"
        assert out["debug"]["star_spans"] == []
        assert out["debug"]["routing"]["route"] == "fast"
        assert out["adlib"] is None
        assert "key" in out and "tempo" in out


# ---------------------------------------------------------------------------
# 번역 레이어 배선 — available_langs/translations_by_lang/translation_lang은
# _run_alignment의 반환값이 아니라 sync.py._apply_translation_lang/
# _build_translations_by_lang이 **조회 시점에** DB(translation_layers 테이블 + 저장된
# timestamps)에서 계산한다(server/api/sync.py 참고) — 그 계산은 어느 정렬 스택이
# timestamps를 만들었는지 모른다. 유일한 전제는 "세그의 text가 원문 가사 줄과 정확히
# 같다"는 것이다: merge_line_meta(run_pipeline_core가 두 스택 공통으로 부른다)와
# translation_layer_lines(레이어 저장 재료)가 둘 다 텍스트로 매칭하기 때문이다. 이
# 절이 새 스택이 그 전제를 지키는지만 못박는다(감사 결함③, 2026-08-03).
# ---------------------------------------------------------------------------


class TestFinishNewStackAlignmentTranslationWiring:
    def test_segments_carry_original_line_text_for_translation_layer_matching(self, monkeypatch):
        results = [
            SyncResult(text="Hello", start_time=0.0, end_time=1.0, confidence=0.9),
            SyncResult(text="World", start_time=1.0, end_time=2.0, confidence=0.9),
        ]
        anchor = _FakeAnchor(results)
        monkeypatch.setattr(
            "everyric2.alignment.factory.EngineFactory.get_engine",
            lambda engine_type, config=None: anchor,
        )
        lyric_lines = [
            LyricLine(text="Hello", line_number=1),
            LyricLine(text="World", line_number=2),
        ]
        out = worker._finish_new_stack_alignment(
            _silence(0.5), None, None, lyric_lines, "ja", _settings(), lambda s: None,
            gloss_folded=None, melody_extractor=None, f0_future=None, f0_executor=None,
        )

        # merge_line_meta(구/신 두 경로가 run_pipeline_core에서 공유하는 병합 함수)가
        # 텍스트로 매칭하므로, 새 스택 세그의 text가 원문 가사 줄과 정확히 같아야
        # 번역이 실제로 붙는다.
        line_meta = [
            {"text": "Hello", "translation": "안녕"},
            {"text": "World", "translation": "세상"},
        ]
        merged = worker.merge_line_meta(out["timestamps"], line_meta)
        assert merged == 2
        assert [seg["translation"] for seg in out["timestamps"]] == ["안녕", "세상"]

        # translations_by_lang(sync.py._build_translations_by_lang)의 저장 재료인
        # translation_layer_lines도 같은 텍스트 매칭에 기댄다 — (text, translation) 쌍이
        # 실제 라인 텍스트·순서·개수와 정확히 일치해야 나중에 lang= 조회가 그 레이어로
        # 세그를 되찾을 수 있다.
        pairs = worker.translation_layer_lines(out["timestamps"])
        assert pairs == [
            {"text": "Hello", "translation": "안녕"},
            {"text": "World", "translation": "세상"},
        ]


# ---------------------------------------------------------------------------
# 단계 보고(report) — STAGE_WINDOWS에 등록된 이름만 쓰는지
# ---------------------------------------------------------------------------


class TestStageReporting:
    """새 스택이 report()로 내보내는 단계명이 전부 STAGE_WINDOWS에 등록돼 있는지(운영자
    지시, 2026-08-04 실곡 검증). 미등록 이름은 ``_stage_monitor``의 기본 창(36,88)에
    걸려 진행률이 88%에서 멈췄다 100으로 점프한다 — 새 하위 단계 이름을 새로 등록하는
    대신 기존 어휘("보컬 분리"·"전사 정렬")로 통일하는 쪽을 택했다(권고안 채택 근거는
    worker.py의 관련 report() 호출부 주석 참고). 이 테스트는 "새 창을 등록할 필요가
    없어졌다"는 것 자체를 못박는다 — 동시에 확장 1.5.5의 STAGE_LABEL_KEYS도 이 두 이름은
    이미 알고 있으므로(everyric2-chrome/src/content.ts) 별도 확장 수정이 필요 없다는
    것의 근거이기도 하다.
    """

    _REGISTERED_STAGES = frozenset(worker.STAGE_WINDOWS) | {worker.LINE_META_WAIT_STAGE}

    def test_fast_route_reports_no_separation_stage(self, monkeypatch):
        results = [SyncResult(text="a", start_time=0.0, end_time=1.0, confidence=0.9)]
        anchor = _FakeAnchor(results)
        monkeypatch.setattr(
            "everyric2.alignment.factory.EngineFactory.get_engine",
            lambda engine_type, config=None: anchor,
        )
        seen: list[str] = []
        worker._run_new_stack_alignment(
            _silence(), None, [LyricLine(text="a", line_number=1)], "ja",
            _settings(), seen.append,
        )
        assert seen, "fast route reported nothing"
        assert set(seen) <= self._REGISTERED_STAGES, seen
        # 고속 경로는 분리를 아예 안 한다 — 그 단계를 표시하면 안 된다(운영자 지시).
        assert "보컬 분리" not in seen
        assert "전사 정렬" in seen

    def test_deep_route_reports_only_registered_stages(self, monkeypatch):
        fast_results = [SyncResult(text="a", start_time=0.0, end_time=1.0, confidence=1e-9)]
        fast_anchor = _FakeAnchor(fast_results)
        deep_anchor = _FakeAnchor([SyncResult(text="a", start_time=0.0, end_time=1.0)])
        calls: list[str] = []

        def fake_get_engine(engine_type, config=None):
            calls.append(engine_type)
            return fast_anchor if len(calls) == 1 else deep_anchor

        monkeypatch.setattr(
            "everyric2.alignment.factory.EngineFactory.get_engine", fake_get_engine
        )
        sep = _FakeSepResult(_silence(), _silence())
        monkeypatch.setattr(
            "everyric2.audio.separator.get_shared_separator",
            lambda config=None: _FakeSeparator(True, sep),
        )
        seen: list[str] = []
        worker._run_new_stack_alignment(
            _silence(), None, [LyricLine(text="a", line_number=1)], "ja",
            _settings(two_pass_enabled=False), seen.append,
        )
        # 실제 순서는 ["전사 정렬"(고속 시도) -> "보컬 분리"(heavy 진입) -> "전사 정렬"
        # (heavy 앵커)]다 — 고속 시도 자체도 "정렬"이라 먼저 한 번 나오는 게 맞다. 순서
        # 자체보다 **전부 등록된 이름인가**가 핵심이다: _stage_monitor의
        # ``progress = max(progress, lo)``가 등록된 창 사이에서는 항상 비감소를
        # 보장하므로(각 창의 lo가 이전 진행보다 낮아도 max가 막는다), 등록만 돼 있으면
        # 순서와 무관하게 88% 정체 버그는 재발하지 않는다.
        assert set(seen) <= self._REGISTERED_STAGES, seen
        assert "보컬 분리" in seen  # heavy는 실제로 분리한다
        assert "전사 정렬" in seen

    def test_two_pass_sub_stage_reuses_registered_alignment_stage_name(self, monkeypatch):
        # 2패스 리파인 진입("음절 재정렬"이었던 자리)도 미등록 이름을 새로 안 만들고
        # "전사 정렬"을 재사용한다 — 같은 창(50,72) 안에서 진행률이 이어진다.
        vocab = {"가": 1}
        logits = torch.full((1, 4, 2), -8.0)
        logits[0, 1, 1] = 8.0
        emission = torch.log_softmax(logits, dim=-1)
        fake_emission = EngineEmission(
            emission=emission, blank_id=0, frame_sec=0.02, audio_sec=0.08, chunks=1,
            vocab=vocab,
        )
        anchor = _FakeAnchor([SyncResult(text="가", start_time=0.0, end_time=0.08)])
        anchor.emission_for = lambda audio: fake_emission  # depth="medium" -> omniasr 자기앵커 겸 리파이너
        monkeypatch.setattr(
            "everyric2.alignment.factory.EngineFactory.get_engine",
            lambda engine_type, config=None: anchor,
        )
        sep = _FakeSepResult(_silence(0.5), _silence(0.5))
        seen: list[str] = []
        worker._run_deep_stage(
            _silence(0.5), sep, [LyricLine(text="가", line_number=1)], "en",
            _settings(two_pass_enabled=True), seen.append, "medium",
        )
        assert set(seen) <= self._REGISTERED_STAGES, seen
        # 앵커 정렬 진입 + 2패스 진입, 둘 다 같은 등록된 이름으로 두 번 나온다.
        assert seen.count("전사 정렬") >= 2
