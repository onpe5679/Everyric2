"""반주 상관 함수(correlate_offset) 단위 테스트 — GPU/네트워크 없이 합성 신호만 쓴다.

부호 규약을 못 박는다: target이 reference보다 늦게 시작하면(커버가 원곡보다 인트로가 긺)
offset은 양수여야 한다 — 이는 GET /api/sync가 소스 타임스탬프를 t + offset으로 사상하는
방향(sync.py _shift_time)과 일치한다. 시프트 검출은 ±1 hop 오차 내, 무관 신호는 저신뢰.
"""

import numpy as np

from everyric2.audio.correlate import CORR_SR, HOP, correlate_offset

HOP_SEC = HOP / CORR_SR


def _click_track(sr: int, duration: float, seed: int) -> np.ndarray:
    """랜덤 간격 클릭 트랙 — 주기성이 없어 진짜 정렬이 유일 피크가 된다."""
    rng = np.random.default_rng(seed)
    n = int(duration * sr)
    y = np.zeros(n, dtype=np.float32)
    t = 0.3
    while t < duration - 0.3:
        idx = int(t * sr)
        # 짧은 감쇠 클릭 (온셋 스파이크)
        length = int(0.02 * sr)
        env = np.exp(-np.linspace(0, 6, length)).astype(np.float32)
        y[idx : idx + length] += env
        t += float(rng.uniform(0.4, 1.1))
    return y


def _delay(x: np.ndarray, sr: int, shift_sec: float) -> np.ndarray:
    """x를 shift_sec만큼 뒤로 민(양수=늦게 시작) 같은 길이 신호."""
    n = len(x)
    shift = int(round(shift_sec * sr))
    out = np.zeros(n, dtype=x.dtype)
    if shift >= 0:
        out[shift:] = x[: n - shift] if shift < n else 0
    else:
        out[: n + shift] = x[-shift:]
    return out


def test_correlate_detects_positive_shift():
    sr = CORR_SR
    ref = _click_track(sr, duration=30.0, seed=0)
    shift = 2.0  # target이 ref보다 2초 늦게 시작
    target = _delay(ref, sr, shift)
    res = correlate_offset(target, sr, ref, sr)
    assert abs(res.offset_sec - shift) <= HOP_SEC + 1e-6
    assert res.confidence >= 0.35


def test_correlate_detects_negative_shift():
    sr = CORR_SR
    ref = _click_track(sr, duration=30.0, seed=3)
    shift = -1.5  # target이 ref보다 1.5초 먼저 시작
    target = _delay(ref, sr, shift)
    res = correlate_offset(target, sr, ref, sr)
    assert abs(res.offset_sec - shift) <= HOP_SEC + 1e-6
    assert res.confidence >= 0.35


def test_correlate_zero_shift_identity():
    sr = CORR_SR
    ref = _click_track(sr, duration=20.0, seed=7)
    res = correlate_offset(ref.copy(), sr, ref, sr)
    assert abs(res.offset_sec) <= HOP_SEC + 1e-6
    assert res.confidence >= 0.35


def test_correlate_unrelated_signals_low_confidence():
    sr = CORR_SR
    rng = np.random.default_rng(11)
    a = rng.standard_normal(int(20 * sr)).astype(np.float32) * 0.1
    b = rng.standard_normal(int(20 * sr)).astype(np.float32) * 0.1
    res = correlate_offset(a, sr, b, sr)
    assert res.confidence < 0.35


def test_correlate_resamples_mismatched_sr():
    # 서로 다른 sr 입력도 공통 sr로 리샘플해 정렬한다
    ref = _click_track(CORR_SR, duration=20.0, seed=5)
    shift = 1.0
    target = _delay(ref, CORR_SR, shift)
    # target을 44100으로 업샘플한 뒤 넘겨도 offset은 초 단위로 동일해야 한다
    import librosa

    target_44k = librosa.resample(target, orig_sr=CORR_SR, target_sr=44100)
    res = correlate_offset(target_44k, 44100, ref, CORR_SR)
    assert abs(res.offset_sec - shift) <= 2 * HOP_SEC


def test_silent_signal_zero_confidence():
    sr = CORR_SR
    silent = np.zeros(int(10 * sr), dtype=np.float32)
    ref = _click_track(sr, duration=10.0, seed=1)
    res = correlate_offset(silent, sr, ref, sr)
    assert res.confidence == 0.0
