"""멜로디 f0 병렬화(WS2-B) 테스트 — precompute_f0 → 주입이 인라인 계산과 동일함을 검증.

무거운 f0 추론(_infer_f0)만 정렬과 병렬로 떼어내는 리팩터라, 핵심은 '주입 경로가 재추론
없이 인라인과 똑같은 노트를 낸다'는 것. GPU·실모델은 건드리지 않고 _infer_f0를 합성 f0로
몽키패치해 순수 numpy 노트 로직만 태운다.
"""

import numpy as np

from everyric2.audio.loader import AudioData
from everyric2.config.settings import get_settings
from everyric2.melody.extractor import MelodyExtractor


def _fake_f0() -> tuple[np.ndarray, np.ndarray]:
    """0.5~1.5s 구간만 220Hz(MIDI 57) 유성, 나머지 무성(0)인 합성 f0."""
    times = np.arange(0.0, 3.0, 0.01)
    f0 = np.where((times >= 0.5) & (times < 1.5), 220.0, 0.0)
    return f0, times


def _dummy_audio() -> AudioData:
    # _infer_f0를 몽키패치하므로 파형 내용은 무의미 — 로더 계약만 만족시킨다
    return AudioData(waveform=np.zeros(16000, dtype=np.float32), sample_rate=16000, duration=1.0)


def test_inject_matches_inline_without_reinference(monkeypatch):
    ext = MelodyExtractor(get_settings().melody)
    f0, times = _fake_f0()
    calls = {"n": 0}

    def fake_infer(audio, vocals=None):
        calls["n"] += 1
        return f0.copy(), times.copy()

    monkeypatch.setattr(ext, "_infer_f0", fake_infer)
    audio = _dummy_audio()

    # 인라인 경로: annotate가 f0를 직접 추론(1회)해 노트를 붙인다
    ts_inline = [{"text": "a", "start": 0.5, "end": 1.5}]
    n_inline = ext.annotate_timestamps(audio, ts_inline)
    assert calls["n"] == 1
    assert n_inline == 1
    assert ts_inline[0].get("notes")
    assert ts_inline[0]["notes"][0]["midi"] == 57  # 220Hz = A3

    # 병렬 경로: precompute가 추론(1회), annotate(inject)는 재추론하지 않는다
    calls["n"] = 0
    pre = ext.precompute_f0(audio, None)
    assert calls["n"] == 1
    ts_inject = [{"text": "a", "start": 0.5, "end": 1.5}]
    n_inject = ext.annotate_timestamps(audio, ts_inject, precomputed_f0=pre)
    assert calls["n"] == 1  # annotate가 다시 추론하지 않았다 (주입 f0 사용)

    # 결과 동일성: 병렬로 미리 계산해 주입해도 노트가 인라인과 완전히 같아야 한다
    assert n_inject == n_inline
    assert ts_inject[0]["notes"] == ts_inline[0]["notes"]


def test_precompute_returns_raw_f0_and_times(monkeypatch):
    ext = MelodyExtractor(get_settings().melody)
    f0, times = _fake_f0()
    monkeypatch.setattr(ext, "_infer_f0", lambda audio, vocals=None: (f0.copy(), times.copy()))
    out_f0, out_times = ext.precompute_f0(_dummy_audio(), None)
    # precompute는 정렬 의존 후처리(마스킹/폴딩) 없이 원본 f0·times를 그대로 돌려준다
    assert np.array_equal(out_f0, f0)
    assert np.array_equal(out_times, times)


def test_annotate_without_precompute_still_works(monkeypatch):
    # precomputed_f0=None이면 기존처럼 내부에서 추론한다 (하위호환)
    ext = MelodyExtractor(get_settings().melody)
    f0, times = _fake_f0()
    monkeypatch.setattr(ext, "_infer_f0", lambda audio, vocals=None: (f0.copy(), times.copy()))
    ts = [{"text": "a", "start": 0.5, "end": 1.5}]
    assert ext.annotate_timestamps(_dummy_audio(), ts) == 1
