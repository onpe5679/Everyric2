"""반주 상관 함수(correlate_offset) 단위 테스트 — GPU/네트워크 없이 합성 신호만 쓴다.

부호 규약을 못 박는다: target이 reference보다 늦게 시작하면(커버가 원곡보다 인트로가 긺)
offset은 양수여야 한다 — 이는 GET /api/sync가 소스 타임스탬프를 t + offset으로 사상하는
방향(sync.py _shift_time)과 일치한다. 시프트 검출은 ±1 hop 오차 내.

지표 계약(2026-07-24 실측 캘리브레이션): confidence = 최고 피크 절대높이(동일 반주 0.93 /
무관 쌍 0.02 실측), offset_margin = 최고−이차 피크(오프셋 유일성 — 주기 신호에서 낮아짐).
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
    # 동일 신호 시프트 = 피크 높이(confidence)와 유일성(margin) 모두 높아야 한다
    assert res.confidence >= 0.55
    assert res.offset_margin >= 0.08


def test_correlate_detects_negative_shift():
    sr = CORR_SR
    ref = _click_track(sr, duration=30.0, seed=3)
    shift = -1.5  # target이 ref보다 1.5초 먼저 시작
    target = _delay(ref, sr, shift)
    res = correlate_offset(target, sr, ref, sr)
    assert abs(res.offset_sec - shift) <= HOP_SEC + 1e-6
    assert res.confidence >= 0.55
    assert res.offset_margin >= 0.08


def test_correlate_zero_shift_identity():
    sr = CORR_SR
    ref = _click_track(sr, duration=20.0, seed=7)
    res = correlate_offset(ref.copy(), sr, ref, sr)
    assert abs(res.offset_sec) <= HOP_SEC + 1e-6
    assert res.confidence >= 0.55


def test_correlate_unrelated_signals_low_confidence():
    # 서로 무관한 온셋 구조 + 서로 다른 길이 — 실제 무관 곡 쌍(실측 피크 0.02)의 모사.
    # 같은 길이의 흰소음 쌍은 프레이밍 에지 스파이크가 lag 0에서 가짜 상관을 만들므로
    # (그래서 _onset_envelope가 선두를 트림한다) 대표성 있는 신호로 검사한다.
    sr = CORR_SR
    a = _click_track(sr, duration=20.0, seed=11)
    b = _click_track(sr, duration=26.0, seed=42)
    res = correlate_offset(a, sr, b, sr)
    assert res.confidence < 0.35


def test_periodic_signal_high_peak_low_margin():
    """주기 신호(루프 곡 모사): 진짜 정렬 피크는 높지만 마디 간격 이차 피크 때문에
    margin이 낮다 — confidence가 아니라 margin 게이트가 이 모호성을 걸러야 한다."""
    sr = CORR_SR
    n = int(30.0 * sr)
    y = np.zeros(n, dtype=np.float32)
    interval = 0.7  # 고정 간격 클릭 = 강한 주기성
    t = 0.3
    while t < 30.0 - 0.3:
        idx = int(t * sr)
        length = int(0.02 * sr)
        y[idx : idx + length] += np.exp(-np.linspace(0, 6, length)).astype(np.float32)
        t += interval
    target = _delay(y, sr, 1.4)  # 정확히 2주기 시프트 — 이웃 주기와 혼동 유발 조건
    res = correlate_offset(target, sr, y, sr)
    assert res.confidence >= 0.55  # 같은 신호라 피크는 높다
    assert res.offset_margin < 0.08  # 하지만 오프셋은 모호하다 → 자동 링크 보류 대상


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
