"""OWSM-CTC v4 1B / omniASR-CTC-300M 엔진의 배선 검증 — 등록·선택·기본값 불변.

새 엔진은 ``EngineFactory``에 등록만 되고 ``AlignmentSettings.engine``의 **기본값은
"ctc"로 그대로 남는다** — 배선 전환(기본 엔진 교체)은 이 작업의 범위 밖인 별도 작업이다.
이 스위트는 그 경계를 못박는다.
"""

from __future__ import annotations

from everyric2.alignment.base import BaseAlignmentEngine
from everyric2.alignment.emission import EngineEmission
from everyric2.alignment.factory import EngineFactory
from everyric2.alignment.nemo_engine import NeMoEngine
from everyric2.alignment.omniasr_engine import OmniASREngine
from everyric2.alignment.owsm_engine import OwsmEngine
from everyric2.alignment.sofa_engine import SOFAEngine
from everyric2.audio.loader import AudioData
from everyric2.config.settings import AlignmentSettings, get_settings


def _audio() -> AudioData:
    return AudioData(waveform=None, sample_rate=16000, duration=1.0)  # type: ignore[arg-type]


def test_default_engine_is_still_ctc():
    # 새 엔진 이식이 기존 배선을 조용히 바꾸면 안 된다.
    assert AlignmentSettings().engine == "ctc"
    assert get_settings().alignment.engine == "ctc"


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


def test_factory_default_engine_type_unaffected():
    # engine_type을 안 주면 여전히 설정 기본값(ctc)을 쓴다.
    from everyric2.alignment.ctc_engine import CTCEngine

    engine = EngineFactory.get_engine()
    assert isinstance(engine, CTCEngine)


def test_get_available_engines_lists_new_engines():
    types = {entry["type"] for entry in EngineFactory.get_available_engines()}
    assert "owsm" in types
    assert "omniasr" in types
    # "available" 키가 bool로 존재해야 한다(환경에 모델/venv가 없어도 예외로 죽지 않는다).
    for entry in EngineFactory.get_available_engines():
        if entry["type"] in ("owsm", "omniasr"):
            assert isinstance(entry["available"], bool)


def test_engine_literal_accepts_new_values():
    # pydantic Literal 검증 — owsm/omniasr가 유효한 설정값으로 받아들여져야 한다.
    assert AlignmentSettings(engine="owsm").engine == "owsm"
    assert AlignmentSettings(engine="omniasr").engine == "omniasr"


def test_base_engine_default_emission_for_is_none():
    # emission_for 훅의 기본 구현 — 지원 안 하는 엔진은 전부 None을 돌려줘야 한다.
    # 기존 엔진(NeMo/SOFA)이 override 없이 그대로 이 기본값을 쓰는지도 함께 확인해
    # base.py 수정이 기존 엔진의 동작을 바꾸지 않았다는 회귀 안전판이 된다.
    audio = _audio()
    assert NeMoEngine().emission_for(audio) is None
    assert SOFAEngine().emission_for(audio) is None
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
