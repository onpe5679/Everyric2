"""독음(ko) 정렬 star-swallow 가드 헬퍼 회귀 테스트.

VWVtIg5cdDU(初音ミクの消失) 실측: ko 정렬에서 star 하나가 간주 이후 실가창 21s를
삼켜 후반 라인이 ~40s 앞으로 압축됐다. 판정은 star 크기(熱異常도 20.7s를 정상
흡수)가 아니라 '간주 이후 발성 창을 어느 정렬이 채우는가'로 한다.
"""
from everyric2.audio.vad import VADResult, VocalRegion
from everyric2.inference.prompt import SyncResult
from everyric2.server.worker import (
    _lines_span_overlap,
    _post_interlude_window,
    _star_swallowed_vocal,
)


def _regions(*spans: tuple[float, float]) -> list[VocalRegion]:
    return [VocalRegion(start=s, end=e, energy=0.1) for s, e in spans]


def test_star_swallow_sums_overlap_and_takes_max_star():
    regions = _regions((10.0, 20.0), (30.0, 40.0))
    stars = [(5.0, 12.0), (15.0, 35.0)]  # 두 번째 star가 5+5=10s 삼킴
    assert _star_swallowed_vocal(stars, regions) == 10.0
    assert _star_swallowed_vocal([], regions) == 0.0


def test_post_interlude_window_anchors_on_largest_gap():
    # 갭: 20→24(4s), 30→42(12s) → 최대 갭 12s ≥ 5 → 창 = (42.0, 60.0)
    regions = _regions((0.0, 20.0), (24.0, 30.0), (42.0, 60.0))
    assert _post_interlude_window(regions, min_gap_sec=5.0) == (42.0, 60.0)


def test_post_interlude_window_none_when_no_big_gap():
    regions = _regions((0.0, 20.0), (21.0, 40.0))
    assert _post_interlude_window(regions, min_gap_sec=5.0) is None
    assert _post_interlude_window(_regions((0.0, 20.0)), min_gap_sec=5.0) is None


def test_lines_span_overlap_counts_only_window():
    lines = [
        SyncResult(text="앞", start_time=10.0, end_time=20.0),   # 창 밖
        SyncResult(text="걸침", start_time=38.0, end_time=46.0),  # 4s 겹침
        SyncResult(text="안", start_time=50.0, end_time=55.0),   # 5s 전체
    ]
    assert _lines_span_overlap(lines, (42.0, 60.0)) == 9.0
