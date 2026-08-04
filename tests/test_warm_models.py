"""모델 웜 캐시(WS2-A) 테스트 — 지연 싱글턴 재사용/토글/로그를 검증한다.

GPU·실모델은 건드리지 않는다: 싱글턴 접근자는 인스턴스만 만들고 모델은 잡 처리 시점에
lazy 로드되므로, 단순 획득에는 어떤 torch 모델도 로드되지 않는다(API 전용 모드 계약의 근거).
"""

import logging
import threading
import time

import pytest

from everyric2.alignment import ctc_engine as ctc_mod
from everyric2.alignment import omniasr_engine as omniasr_mod
from everyric2.alignment import owsm_engine as owsm_mod
from everyric2.audio import separator as sep_mod
from everyric2.audio.separator import get_shared_separator
from everyric2.config.settings import AlignmentSettings, get_settings
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


# ── 새 스택(owsm/omniasr) 웜 캐시 (2026-08-04) ────────────────────────────────────
#
# 위 세 엔진과 달리 owsm/omniasr는 **config 키를 쓰는 dict 캐시**다(단일 슬롯이 아니다) —
# 그래서 _MODULES/_reset_singletons(단일 슬롯을 None으로 되돌리는 규약)를 그대로 못 쓰고
# 별도 fixture로 dict를 {}로 되돌린다. 왜 dict인지: get_shared_owsm_engine/
# get_shared_omniasr_engine 함수 자신의 docstring(owsm_engine.py·omniasr_engine.py)에
# 근거를 남겨 뒀다 — owsm은 owsm_python_path·owsm_dtype이 실제로 적재 결과에 영향을 주고,
# omniasr는 지금은 config 무관이지만 안전 계약(다른 config면 새로 만든다)을 위해 dict
# 구조 자체는 유지한다.


@pytest.fixture(autouse=True)
def _reset_new_stack_caches():
    owsm_mod._shared_owsm_engines = {}
    omniasr_mod._shared_omniasr_engines = {}
    yield
    owsm_mod._shared_owsm_engines = {}
    omniasr_mod._shared_omniasr_engines = {}


def test_omniasr_warm_reuses_same_instance():
    _set_warm(True)
    a = omniasr_mod.get_shared_omniasr_engine(AlignmentSettings())
    b = omniasr_mod.get_shared_omniasr_engine(AlignmentSettings())
    assert a is b


def test_omniasr_warm_disabled_returns_fresh_instance():
    _set_warm(False)
    a = omniasr_mod.get_shared_omniasr_engine(AlignmentSettings())
    b = omniasr_mod.get_shared_omniasr_engine(AlignmentSettings())
    assert a is not b


def test_omniasr_warm_reuse_logs_once(caplog):
    _set_warm(True)
    with caplog.at_level(logging.INFO):
        omniasr_mod.get_shared_omniasr_engine()
        omniasr_mod.get_shared_omniasr_engine()
    assert "warm model reuse: omniasr" in caplog.text


def test_owsm_warm_reuses_same_instance_for_same_config_object():
    _set_warm(True)
    cfg = AlignmentSettings(owsm_python_path="/fake/py", owsm_dtype="bfloat16")
    a = owsm_mod.get_shared_owsm_engine(cfg)
    b = owsm_mod.get_shared_owsm_engine(cfg)
    assert a is b


def test_owsm_warm_reuses_same_instance_for_equivalent_distinct_config_objects():
    # 요청마다 새 AlignmentSettings 인스턴스가 생겨도, 캐시에 영향 주는 필드값(owsm_python_
    # path·owsm_dtype)이 같으면 재사용해야 한다 — 객체 identity가 아니라 값으로 키를 잡는다.
    _set_warm(True)
    cfg1 = AlignmentSettings(owsm_python_path="/fake/py", owsm_dtype="bfloat16")
    cfg2 = AlignmentSettings(owsm_python_path="/fake/py", owsm_dtype="bfloat16")
    a = owsm_mod.get_shared_owsm_engine(cfg1)
    b = owsm_mod.get_shared_owsm_engine(cfg2)
    assert a is b


def test_owsm_warm_creates_new_instance_for_different_python_path():
    _set_warm(True)
    cfg1 = AlignmentSettings(owsm_python_path="/fake/py1")
    cfg2 = AlignmentSettings(owsm_python_path="/fake/py2")
    a = owsm_mod.get_shared_owsm_engine(cfg1)
    b = owsm_mod.get_shared_owsm_engine(cfg2)
    assert a is not b, "설정(인터프리터 경로)이 다른데 웜 캐시를 재사용하면 옛 실행 환경이 물린다"


def test_owsm_warm_creates_new_instance_for_different_dtype():
    _set_warm(True)
    cfg1 = AlignmentSettings(owsm_dtype="bfloat16")
    cfg2 = AlignmentSettings(owsm_dtype="float32")
    a = owsm_mod.get_shared_owsm_engine(cfg1)
    b = owsm_mod.get_shared_owsm_engine(cfg2)
    assert a is not b


def test_owsm_warm_disabled_returns_fresh_instance():
    _set_warm(False)
    cfg = AlignmentSettings()
    a = owsm_mod.get_shared_owsm_engine(cfg)
    b = owsm_mod.get_shared_owsm_engine(cfg)
    assert a is not b


def test_owsm_warm_reuse_logs_once(caplog):
    _set_warm(True)
    cfg = AlignmentSettings()
    with caplog.at_level(logging.INFO):
        owsm_mod.get_shared_owsm_engine(cfg)
        owsm_mod.get_shared_owsm_engine(cfg)
    assert "warm model reuse: owsm" in caplog.text


def test_owsm_clear_shared_engine_drops_all_cached_configs():
    _set_warm(True)
    cfg1 = AlignmentSettings(owsm_python_path="/fake/py1")
    cfg2 = AlignmentSettings(owsm_python_path="/fake/py2")
    a1 = owsm_mod.get_shared_owsm_engine(cfg1)
    a2 = owsm_mod.get_shared_owsm_engine(cfg2)
    owsm_mod.clear_shared_owsm_engine()
    b1 = owsm_mod.get_shared_owsm_engine(cfg1)
    b2 = owsm_mod.get_shared_owsm_engine(cfg2)
    assert b1 is not a1
    assert b2 is not a2


def test_new_stack_shared_acquisition_does_not_load_models():
    # ctc/melody와 같은 계약 — 획득만으로는 무거운 것이 안 돈다. omniasr는 _model이 lazy라
    # 이걸로 직접 확인 가능하다. owsm은 애초에 인프로세스 모델이 없어(서브프로세스 격리)
    # 이 속성 자체가 없다 — is_available()/align() 호출 전까지는 프로세스도 안 뜬다는
    # 사실은 이 인스턴스 생성 코드가 subprocess.run을 부르지 않는 것으로 대신 확인한다.
    _set_warm(True)
    assert omniasr_mod.get_shared_omniasr_engine()._model is None


def test_concurrent_first_load_creates_only_one_instance_omniasr(monkeypatch):
    """동시 첫 적재 요청에도 인스턴스 생성이 1회만 — 스레드 안전성(락) 검증.

    생성자에 인위적 지연을 넣어 락이 없으면 반드시 여러 스레드가 겹쳐 들어가게 만든다
    (id() 비교만으로는 우연히 안 겹쳐서 거짓 통과할 수 있어, 생성 호출 횟수를 직접 센다)."""
    _set_warm(True)
    construct_count = 0
    count_lock = threading.Lock()
    original_init = omniasr_mod.OmniASREngine.__init__

    def slow_init(self, config=None):
        nonlocal construct_count
        with count_lock:
            construct_count += 1
        time.sleep(0.05)
        original_init(self, config)

    monkeypatch.setattr(omniasr_mod.OmniASREngine, "__init__", slow_init)

    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        omniasr_mod.get_shared_omniasr_engine()

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert construct_count == 1, f"동시 첫 적재에서 인스턴스가 {construct_count}번 생성됐다(락 실패)"


def test_concurrent_first_load_creates_only_one_instance_owsm(monkeypatch):
    _set_warm(True)
    construct_count = 0
    count_lock = threading.Lock()
    original_init = owsm_mod.OwsmEngine.__init__

    def slow_init(self, config=None):
        nonlocal construct_count
        with count_lock:
            construct_count += 1
        time.sleep(0.05)
        original_init(self, config)

    monkeypatch.setattr(owsm_mod.OwsmEngine, "__init__", slow_init)

    cfg = AlignmentSettings()
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        owsm_mod.get_shared_owsm_engine(cfg)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert construct_count == 1, f"동시 첫 적재에서 인스턴스가 {construct_count}번 생성됐다(락 실패)"


def test_get_engine_type_reflects_actual_engine_class():
    # sync_results.engine 저장 지점(worker.py)이 engine.get_engine_type()으로 값을
    # 결정한다 — 각 엔진이 자기 타입을 정확히 보고하는지 못박는다(GPU 없이 확인 가능,
    # 정적 메서드라 인스턴스 상태와 무관).
    assert omniasr_mod.OmniASREngine.get_engine_type() == "omniasr"
    assert owsm_mod.OwsmEngine.get_engine_type() == "owsm"
