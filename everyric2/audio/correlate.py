"""반주(인스트) 상관 기반 오프셋·신뢰도 추정 — 커버가 원곡과 같은 반주를 쓰는지 판정.

GPU가 필요 없는 순수 신호처리라 개발기(GPU 사용 금지)에서도 합성 신호로 단위 테스트할
수 있게 분리한다. 입력은 반주 파형 2개(임의 sr) → 공통 sr(22050 모노)로 리샘플 →
`librosa.onset.onset_strength` 엔벨로프(hop 512) → 정규화 크로스 코릴레이션 →
최고 피크 위치 = offset, confidence = 피크 대비 이차 피크 정규화 점수.

부호 규약: ``correlate_offset(target, ..., reference, ...)``이 돌려주는 offset은 target이
reference보다 얼마나 '늦은지'(초)다 — 즉 target(t) ≈ reference(t - offset). 링크 소비부
(GET /api/sync/{video_id})가 소스 타임스탬프를 ``t / rate + offset``으로 사상하므로
(sync.py ``_shift_time``), target = 커버(SyncLink.video_id), reference = 원곡
(SyncLink.source_video_id)으로 부르면 반환 offset이 그대로 ``SyncLink.offset_sec``가 된다
(t_cover = t_source + offset). 이 규약은 tests/test_correlate.py가 못 박는다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 상관 계산용 공통 표현 — 노래 오디오는 이 sr/hop이면 온셋 구조를 충분히 담고 계산도 가볍다
CORR_SR = 22050
HOP = 512
# 커버/원곡의 인트로 편차는 보통 수십 초 안 — 탐색을 ±90s로 제한해 곡 길이 배수 오탐을 줄인다
DEFAULT_MAX_LAG_SEC = 90.0
# 이차 피크를 셀 때 최고 피크 주변(초)을 제외 — 인접 lag은 거의 같은 정렬이라 배제해야
# '얼마나 뾰족한 매칭인가'를 잰다
_PEAK_EXCLUDE_SEC = 0.5
_EPS = 1e-9


@dataclass
class CorrelationResult:
    offset_sec: float
    confidence: float


def _onset_envelope(waveform: np.ndarray, sr: int) -> np.ndarray:
    """파형 → 온셋 강도 엔벨로프 (공통 sr로 리샘플 후 hop 512)."""
    import librosa

    y = np.asarray(waveform, dtype=np.float32)
    if y.ndim > 1:
        y = librosa.to_mono(y)
    if sr != CORR_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=CORR_SR)
    env = librosa.onset.onset_strength(y=y, sr=CORR_SR, hop_length=HOP)
    return np.asarray(env, dtype=np.float64)


def _normalize(env: np.ndarray) -> np.ndarray | None:
    """평균 제거 + 단위 노름 정규화. 무음/상수 신호(노름 0)면 None."""
    env = np.asarray(env, dtype=np.float64)
    env = env - env.mean()
    norm = float(np.linalg.norm(env))
    if norm < _EPS:
        return None
    return env / norm


def correlate_envelopes(
    env_target: np.ndarray,
    env_reference: np.ndarray,
    max_lag_frames: int,
    exclude_frames: int = 1,
) -> tuple[int, float]:
    """정규화된 두 엔벨로프의 크로스 코릴레이션 → (best_lag_frames, confidence).

    best_lag_frames는 target이 reference보다 늦은 프레임 수(양수 = target이 뒤). confidence는
    최고 피크와 (그 주변을 제외한) 이차 피크의 격차를 최고 피크로 정규화한 [0, 1) 점수 —
    뾰족한 유일 매칭이면 1에 가깝고, 평평한(무관) 상관이면 0에 가깝다. 순수 numpy/scipy라
    합성 신호로 직접 테스트한다."""
    from scipy.signal import correlate, correlation_lags

    a = _normalize(env_target)
    b = _normalize(env_reference)
    if a is None or b is None:
        return 0, 0.0

    corr = correlate(a, b, mode="full", method="fft")
    lags = correlation_lags(len(a), len(b), mode="full")

    # 탐색 범위를 ±max_lag_frames로 제한 (곡 길이 배수 등 먼 오정렬 배제)
    window = np.abs(lags) <= max(1, max_lag_frames)
    corr_w = corr[window]
    lags_w = lags[window]
    if corr_w.size == 0:
        return 0, 0.0

    best_i = int(np.argmax(corr_w))
    best_lag = int(lags_w[best_i])
    peak = float(corr_w[best_i])
    if peak <= 0.0:
        return best_lag, 0.0

    # 최고 피크 주변(같은 정렬)을 제외한 이차 피크 = 다음으로 그럴듯한 대안 정렬의 강도
    lo = max(0, best_i - exclude_frames)
    hi = min(corr_w.size, best_i + exclude_frames + 1)
    masked = corr_w.copy()
    masked[lo:hi] = -np.inf
    secondary = float(np.max(masked)) if np.isfinite(np.max(masked)) else 0.0
    secondary = max(0.0, secondary)

    # 정규화 엔벨로프(평균0·단위노름)의 크로스 코릴레이션은 값이 [-1, 1]이라, 최고 피크와
    # 이차 피크의 격차 자체가 이미 정규화된 신뢰도다: 강하고(peak↑) 유일한(secondary↓)
    # 매칭이면 1에 가깝고, 약하거나(무관 신호) 모호한(주기적 곡) 정렬이면 0에 가깝다.
    confidence = max(0.0, peak - secondary)
    return best_lag, confidence


def correlate_offset(
    target_waveform: np.ndarray,
    target_sr: int,
    reference_waveform: np.ndarray,
    reference_sr: int,
    max_lag_sec: float = DEFAULT_MAX_LAG_SEC,
) -> CorrelationResult:
    """두 반주 파형의 상관 오프셋·신뢰도. offset = target이 reference보다 늦은 초(부호 규약은
    모듈 docstring 참조: target = 커버, reference = 원곡 → SyncLink.offset_sec)."""
    env_t = _onset_envelope(target_waveform, target_sr)
    env_r = _onset_envelope(reference_waveform, reference_sr)
    max_lag_frames = int(round(max_lag_sec * CORR_SR / HOP))
    exclude_frames = max(1, int(round(_PEAK_EXCLUDE_SEC * CORR_SR / HOP)))
    best_lag, confidence = correlate_envelopes(
        env_t, env_r, max_lag_frames, exclude_frames=exclude_frames
    )
    offset_sec = round(best_lag * HOP / CORR_SR, 3)
    return CorrelationResult(offset_sec=offset_sec, confidence=round(confidence, 4))
