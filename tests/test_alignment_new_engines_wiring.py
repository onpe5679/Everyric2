"""OWSM-CTC v4 1B / omniASR-CTC-300M 엔진의 배선 검증 — 등록·선택·기본값.

2026-08-03 배선 전환(everyric2/server/worker.py::_run_new_stack_alignment,
tests/test_new_stack_wiring.py) 이전에는 이 엔진들이 ``EngineFactory``에 등록만 되고
``AlignmentSettings.engine`` 기본값은 "ctc"로 남아 있었다. 배선이 실제로 태워진 지금은
기본값이 "owsm"(새 스택 켜짐의 동의어 — 언어별 실제 앵커는 owsm/omniasr로 갈린다,
AlignmentSettings.engine 필드 docstring 참고)이다 — 구스택(ctc 등)은 명시적으로 골라야
나온다(``EVERYRIC_ALIGNMENT_ENGINE=ctc``). 이 스위트는 그 경계를 못박는다.
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


def test_default_engine_is_now_owsm():
    # 배선 전환(2026-08-03) 이후 기본값 — "owsm" 리터럴 자체는 새 스택 스위치일 뿐이고
    # 실제 앵커는 언어별로 갈린다(worker._new_stack_anchor_type). 구스택은 여전히 명시적
    # 선택으로 그대로 나온다(아래 테스트).
    assert AlignmentSettings().engine == "owsm"
    assert get_settings().alignment.engine == "owsm"


def test_legacy_ctc_still_selectable_explicitly():
    assert AlignmentSettings(engine="ctc").engine == "ctc"


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
    # engine_type을 안 주면 설정 기본값을 쓴다 — 배선 전환 이후 그 기본값은 owsm이다.
    engine = EngineFactory.get_engine()
    assert isinstance(engine, OwsmEngine)


def test_factory_default_engine_type_honours_explicit_config():
    # 설정을 명시적으로 ctc로 주면 여전히 CTCEngine이 나온다(구스택 경로 그대로).
    from everyric2.alignment.ctc_engine import CTCEngine

    engine = EngineFactory.get_engine(config=AlignmentSettings(engine="ctc"))
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
