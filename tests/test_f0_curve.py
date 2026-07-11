"""downsample_f0_curve — 레인 디버그 오버레이용 RAW f0 다운샘플 회귀 테스트."""
import numpy as np

from everyric2.melody.extractor import F0Track, downsample_f0_curve


def test_downsample_to_20hz_with_unvoiced_none():
    # 100Hz 트랙 10초 → stride 5로 20Hz, 무성 프레임은 None으로 남는다
    times = (np.arange(1000) + 0.5) * 0.01
    midi = np.full(1000, 60.0)
    voiced = np.ones(1000, dtype=bool)
    voiced[500:520] = False
    curve = downsample_f0_curve(F0Track(times=times, midi=midi, voiced=voiced))

    assert curve is not None
    assert abs(curve["dt"] - 0.05) < 1e-6
    assert len(curve["midi"]) == 200
    assert curve["midi"][0] == 60.0
    assert curve["midi"][101] is None  # 원본 idx 505 = 무성


def test_downsample_caps_point_count():
    # 아주 긴 곡도 max_points를 넘지 않는다
    n = 100_000
    times = (np.arange(n) + 0.5) * 0.01
    track = F0Track(times=times, midi=np.full(n, 62.0), voiced=np.ones(n, dtype=bool))
    curve = downsample_f0_curve(track, max_points=5000)
    assert curve is not None
    assert len(curve["midi"]) <= 5000


def test_downsample_empty_track_returns_none():
    empty = F0Track(
        times=np.array([]), midi=np.array([]), voiced=np.array([], dtype=bool)
    )
    assert downsample_f0_curve(empty) is None
