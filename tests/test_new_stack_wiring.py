"""새 정렬 스택(owsm/omniasr 앵커 + 2패스 리파이너) 배선 회귀 테스트.

이식된 부품 자체(owsm_engine/omniasr_engine/refine_window/display_fixes)는 각자의
테스트 스위트(test_alignment_new_engines_wiring.py·test_refine_window.py·
test_display_fixes.py)가 이미 못박고 있다 — 이 파일은 그것들을 ``everyric2/server/
worker.py``에 실제로 배선한 층만 검증한다: 설정 정합성 가드, 언어별 앵커 라우팅,
``refine_window.SyllableRefiner``(Path 계약) ↔ ``BaseAlignmentEngine.emission_for``
(AudioData 계약) 사이의 배선 어댑터, 그리고 ``_run_new_stack_alignment``/
``_finish_new_stack_alignment``이 앵커·리파이너 호출 결과를 응답 계약(pron/pron_segs/
adlib 등)으로 조립하는 흐름. 모델을 실제로 로드/추론하지 않는다 — 앵커·리파이너는
전부 페이크(호출 계약 수준 검증, 코디네이터 지시).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from everyric2.alignment.emission import EngineEmission
from everyric2.alignment.refine_window import PronSegmentSpan
from everyric2.audio.loader import AudioData
from everyric2.config.settings import AlignmentSettings, AudioSettings, Settings
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
        # 분리기만 새 백엔드를 골라도 무방하다 — 정합성 요구는 편도(새 앵커 → 새 분리기)다.
        Settings(
            alignment=AlignmentSettings(engine="ctc"),
            audio=AudioSettings(separator_backend="bs-polarformer-fp16"),
        )


# ---------------------------------------------------------------------------
# 새 스택 게이트 · 언어별 앵커 라우팅
# ---------------------------------------------------------------------------


class _FakeSettings:
    def __init__(self, engine: str) -> None:
        self.alignment = AlignmentSettings(engine=engine)


class TestNewStackGateAndRouting:
    def test_disabled_for_legacy_engines(self):
        for engine in ("ctc", "nemo", "gpu-hybrid", "sofa"):
            assert worker._new_stack_enabled(_FakeSettings(engine)) is False

    def test_enabled_for_new_anchor_engines(self):
        for engine in ("owsm", "omniasr"):
            assert worker._new_stack_enabled(_FakeSettings(engine)) is True

    def test_ja_routes_to_owsm(self):
        assert worker._new_stack_anchor_type("ja") == "owsm"

    @pytest.mark.parametrize("language", ["en", "ko", "zh", "auto", None, "", "JA"])
    def test_non_ja_routes_to_omniasr_except_case_insensitive_ja(self, language):
        # "JA"는 대소문자 무시 매칭이라 owsm으로 간다 — 그 밖은 전부 omniasr.
        expected = "owsm" if (language or "").strip().lower() == "ja" else "omniasr"
        assert worker._new_stack_anchor_type(language) == expected


# ---------------------------------------------------------------------------
# _PathBridgedRefiner — Path 계약(refine_window) <-> AudioData 계약(BaseAlignmentEngine)
# ---------------------------------------------------------------------------


class _RecordingEngine:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.received: Any = None

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
        # 브리지가 로더로 얻은 AudioData를 그대로 실제 엔진(BaseAlignmentEngine.emission_for
        # 계약)에 넘겼는지 — Path가 새지 않았는지가 핵심.
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
# _run_new_stack_alignment — 앵커 호출 · 라우팅 · 결과 조립
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
    from types import SimpleNamespace

    kwargs: dict[str, Any] = {"engine": "owsm", "two_pass_enabled": False}
    kwargs.update(overrides)
    align = AlignmentSettings(**kwargs)
    from everyric2.config.settings import SegmentationSettings

    return SimpleNamespace(alignment=align, segmentation=SegmentationSettings())


def _silence(seconds: float = 0.5, sr: int = 16000) -> AudioData:
    n = max(1, int(seconds * sr))
    return AudioData(waveform=np.zeros(n, dtype=np.float32), sample_rate=sr, duration=seconds)


class TestRunNewStackAlignmentAnchorOnly:
    """two_pass_enabled=False, 분리 결과 없음(vocals=None) — 가장 단순한 경로."""

    def test_picks_owsm_for_japanese(self, monkeypatch):
        anchor = _FakeAnchor([SyncResult(text="a", start_time=0.0, end_time=1.0)])
        recorded: dict[str, Any] = {}

        def fake_get_engine(engine_type, config=None):
            recorded["type"] = engine_type
            return anchor

        monkeypatch.setattr(
            "everyric2.alignment.factory.EngineFactory.get_engine", fake_get_engine
        )
        lyric_lines = [LyricLine(text="a", line_number=1)]
        stack = worker._run_new_stack_alignment(
            _silence(), _silence(), None, lyric_lines, "ja", _settings(), lambda s: None
        )
        assert recorded["type"] == "owsm"
        assert stack.engine is anchor
        assert stack.alignment_text == "owsm"
        assert stack.pron_data == {}
        assert stack.vad_regions is None
        assert stack.adlib is None
        assert len(anchor.align_calls) == 1
        assert anchor.align_calls[0]["language"] == "ja"

    def test_picks_omniasr_for_english(self, monkeypatch):
        anchor = _FakeAnchor([SyncResult(text="a", start_time=0.0, end_time=1.0)])
        recorded: dict[str, Any] = {}

        def fake_get_engine(engine_type, config=None):
            recorded["type"] = engine_type
            return anchor

        monkeypatch.setattr(
            "everyric2.alignment.factory.EngineFactory.get_engine", fake_get_engine
        )
        lyric_lines = [LyricLine(text="a", line_number=1)]
        stack = worker._run_new_stack_alignment(
            _silence(), _silence(), None, lyric_lines, "en", _settings(), lambda s: None
        )
        assert recorded["type"] == "omniasr"
        assert stack.alignment_text == "omniasr"

    def test_unavailable_anchor_raises(self, monkeypatch):
        anchor = _FakeAnchor([], available=False)
        monkeypatch.setattr(
            "everyric2.alignment.factory.EngineFactory.get_engine",
            lambda engine_type, config=None: anchor,
        )
        with pytest.raises(RuntimeError, match="not available"):
            worker._run_new_stack_alignment(
                _silence(), _silence(), None, [LyricLine(text="a", line_number=1)],
                "en", _settings(), lambda s: None,
            )


class TestRunNewStackAlignmentTwoPass:
    """two_pass_enabled=True — 리파이너까지 실제로(합성 emission, CPU) 태운다."""

    def test_refine_populates_multi_script_pron_data(self, monkeypatch, tmp_path):
        # "가" 한 글자를 강하게 지지하는 합성 emission — refine_lines의 forced_align이
        # 실제로 그 글자에 스팬을 낸다(test_refine_window.py와 같은 전략).
        vocab = {"가": 1}
        frame_sec = 0.02
        logits = torch.full((1, 4, 2), -8.0)
        logits[0, 1, 1] = 8.0  # blank=0 열, "가"=1 열
        emission = torch.log_softmax(logits, dim=-1)
        fake_emission = EngineEmission(
            emission=emission, blank_id=0, frame_sec=frame_sec, audio_sec=0.08, chunks=1,
            vocab=vocab,
        )

        anchor_result = SyncResult(text="가", start_time=0.0, end_time=0.08, confidence=0.9)
        anchor = _FakeAnchor([anchor_result])
        refiner_engine = _RecordingEngine(fake_emission)

        def fake_get_engine(engine_type, config=None):
            # 언어 "ja" -> 앵커 타입 "owsm"(_FakeAnchor), 리파이너는 항상 "omniasr" 별도 조회.
            return anchor if engine_type == "owsm" else refiner_engine

        monkeypatch.setattr(
            "everyric2.alignment.factory.EngineFactory.get_engine", fake_get_engine
        )

        vocals = _silence(0.5)
        accompaniment = _silence(0.5)

        class _SepResult:
            pass

        sep_result = _SepResult()
        sep_result.vocals = vocals
        sep_result.accompaniment = accompaniment

        lyric_lines = [LyricLine(text="가", line_number=1)]
        settings = _settings(two_pass_enabled=True)
        stack = worker._run_new_stack_alignment(
            vocals, vocals, sep_result, lyric_lines, "ja", settings, lambda s: None
        )

        assert stack.alignment_text == "owsm-2pass"
        assert 0 in stack.pron_data
        entry = stack.pron_data[0]
        # 표기별(pron/pron_segs)과 레거시 hangul 필드(pronunciation/pron_segments) 둘 다.
        assert "hangul" in entry["pron"]
        assert entry["pron"]["hangul"]
        assert entry["pronunciation"] == entry["pron"]["hangul"]
        assert entry["pron_segs"]["hangul"]
        assert entry["pron_segments"] == entry["pron_segs"]["hangul"]
        for seg in entry["pron_segments"]:
            assert set(seg) <= {"text", "start", "end", "resolved", "confidence", "word_end"}


# ---------------------------------------------------------------------------
# _finish_new_stack_alignment — 응답 dict 조립(멜로디/품질 없이)
# ---------------------------------------------------------------------------


class TestFinishNewStackAlignment:
    def test_response_shape_and_adlib_additive_field(self, monkeypatch):
        result = SyncResult(
            text="hi", start_time=0.0, end_time=1.0, confidence=0.8,
            word_segments=[WordSegment(word="hi", start=0.0, end=1.0, confidence=0.8)],
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
            audio, audio, None, None, lyric_lines, "en", settings, lambda s: None,
            gloss_folded=None, melody_extractor=None, f0_future=None, f0_executor=None,
        )

        assert out["timestamps"][0]["text"] == "hi"
        assert out["alignment_text"] == "omniasr"
        assert out["debug"]["alignment_text"] == "omniasr"
        assert out["debug"]["star_spans"] == []
        # adlib은 새 스택 전용 additive 필드 — 우세도 신호가 없으면(vocals=None) None.
        assert out["adlib"] is None
        assert "key" in out and "tempo" in out
