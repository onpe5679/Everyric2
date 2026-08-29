"""Adaptive 공개 엔진과 내부 OWSM/omniASR 앵커의 배선 계약을 검증한다.

사용자가 선택하는 엔진은 ``adaptive`` 하나이고, 앵커 선택은 라우터의 내부 구현이다.
"""

from __future__ import annotations

from everyric2.alignment.adaptive_engine import AdaptiveEngine
from everyric2.alignment.base import BaseAlignmentEngine
from everyric2.alignment.emission import EngineEmission
from everyric2.alignment.factory import EngineFactory
from everyric2.alignment.omniasr_engine import OmniASREngine
from everyric2.alignment.owsm_engine import OwsmEngine
from everyric2.audio.loader import AudioData
from everyric2.config.settings import AlignmentSettings, get_settings


def _audio() -> AudioData:
    return AudioData(waveform=None, sample_rate=16000, duration=1.0)  # type: ignore[arg-type]


def test_default_engine_is_adaptive():
    assert AlignmentSettings().engine == "adaptive"
    assert get_settings().alignment.engine == "adaptive"


def test_legacy_engine_is_not_a_valid_setting():
    import pytest

    with pytest.raises(Exception):
        AlignmentSettings(engine="ctc")


def test_factory_resolves_owsm_engine():
    engine = EngineFactory.get_engine("owsm")
    assert isinstance(engine, OwsmEngine)
    assert isinstance(engine, BaseAlignmentEngine)
    assert engine.get_engine_type() == "owsm"


def test_factory_resolves_omniasr_engine():
    engine = EngineFactory.get_engine("omniasr")
    assert isinstance(engine, OmniASREngine)
    assert isinstance(engine, BaseAlignmentEngine)
    assert engine.get_engine_type() == "omniasr"


def test_factory_default_engine_type_follows_settings():
    engine = EngineFactory.get_engine()
    assert isinstance(engine, AdaptiveEngine)


def test_get_available_engines_lists_only_adaptive():
    types = {entry["type"] for entry in EngineFactory.get_available_engines()}
    assert types == {"adaptive"}
    assert isinstance(EngineFactory.get_available_engines()[0]["available"], bool)


def test_engine_literal_accepts_only_adaptive():
    assert AlignmentSettings(engine="adaptive").engine == "adaptive"


def test_base_engine_default_emission_for_is_none():
    # emission_for 훅의 기본 구현 — 지원 안 하는 엔진은 전부 None을 돌려줘야 한다.
    audio = _audio()
    assert OwsmEngine().emission_for(audio) is None


def test_omniasr_emission_for_is_a_real_override():
    # OmniASREngine은 emission_for를 override한다 — 클래스 자체가 base 구현과 달라야 한다.
    assert OmniASREngine.emission_for is not BaseAlignmentEngine.emission_for
    assert OwsmEngine.emission_for is BaseAlignmentEngine.emission_for


def test_engine_emission_dataclass_shape():
    # 2패스 리파이너가 맞출 계약 — 필드가 조용히 사라지면 그쪽 이식이 깨진다.
    import torch

    payload = EngineEmission(
        emission=torch.zeros(1, 4, 3),
        blank_id=0,
        frame_sec=0.02,
        audio_sec=0.08,
        chunks=1,
        vocab={"a": 1, "b": 2},
    )
    assert payload.frame_of(0.04) == 2
    assert payload.vocab == {"a": 1, "b": 2}
