"""모델 웜 캐시(WS2-A) 테스트 — 지연 싱글턴 재사용/토글/로그를 검증한다.

GPU·실모델은 건드리지 않는다: 싱글턴 접근자는 인스턴스만 만들고 모델은 잡 처리 시점에
lazy 로드되므로, 단순 획득에는 어떤 torch 모델도 로드되지 않는다(API 전용 모드 계약의 근거).
"""

import logging

import pytest

from everyric2.alignment import ctc_engine as ctc_mod
from everyric2.audio import separator as sep_mod
from everyric2.audio.separator import get_shared_separator
from everyric2.config.settings import get_settings
from everyric2.melody import extractor as mel_mod

_MODULES = (
    (sep_mod, "_shared_separator"),
    (ctc_mod, "_shared_ctc_engine"),
    (mel_mod, "_shared_extractor"),
)


@pytest.fixture(autouse=True)
def _reset_singletons():
    for mod, attr in _MODULES:
        setattr(mod, attr, None)
    yield
    for mod, attr in _MODULES:
        setattr(mod, attr, None)


def _set_warm(value: bool) -> None:
    object.__setattr__(get_settings().server, "warm_models", value)


def _getters():
    from everyric2.alignment.ctc_engine import get_shared_ctc_engine
    from everyric2.melody.extractor import get_shared_extractor

    return [get_shared_separator, get_shared_ctc_engine, get_shared_extractor]


def test_warm_models_default_is_true():
    assert get_settings().server.warm_models is True


@pytest.mark.parametrize("idx", [0, 1, 2])
def test_warm_reuses_same_instance(idx):
    _set_warm(True)
    getter = _getters()[idx]
    a = getter()
    b = getter()
    assert a is b  # 두 번째 잡부터 재생성 0회


@pytest.mark.parametrize("idx", [0, 1, 2])
def test_warm_disabled_returns_fresh_instance(idx):
    _set_warm(False)
    getter = _getters()[idx]
    a = getter()
    b = getter()
    assert a is not b  # warm off면 기존처럼 잡마다 새 인스턴스


@pytest.mark.parametrize(
    "idx,name",
    [(0, "demucs"), (1, "ctc"), (2, "melody")],
)
def test_warm_reuse_logs_once(idx, name, caplog):
    _set_warm(True)
    getter = _getters()[idx]
    with caplog.at_level(logging.INFO):
        getter()  # 최초 생성 — 로그 없음
        getter()  # 재사용 — "warm model reuse: <name>"
    assert f"warm model reuse: {name}" in caplog.text


def test_shared_acquisition_does_not_load_models():
    # 싱글턴 획득만으로는 어떤 모델도 로드되지 않는다 — 잡 처리 시점에 lazy 로드
    # (API 전용 모드 프로세스에 모델이 상주하지 않게 하는 근거)
    _set_warm(True)
    from everyric2.alignment.ctc_engine import get_shared_ctc_engine
    from everyric2.melody.extractor import get_shared_extractor

    assert get_shared_extractor()._model is None
    assert get_shared_ctc_engine()._model is None
