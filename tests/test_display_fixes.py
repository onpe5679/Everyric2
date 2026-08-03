"""표시 보정층(우세도 기반) + 추임새 탐지 회귀 테스트.

포팅 원본: ``scripts/bench_adapters/postprocess.py``(장치 4종 + 좌초 탐지기)와
``scripts/karaoke_review.py``의 ``adlib_candidates``. 장치별로 (a) 발동해야 할 케이스에서
발동 (b) 발동하면 안 되는 케이스에서 무동작을 확인한다.

특히 ``_pull_disconnected_tails``의 «꼬리 ≤2세그» 가드는 오검출 1호(ロキ 8초 휴지 뒤
뒷소절 10세그를 잔해로 오판해 당겼던 사고)를 막은 것이라 반드시 못박는다
(``test_pull_disconnected_tails_guards_long_real_verse_after_long_rest``).
"""

from __future__ import annotations

import numpy as np
import pytest

from everyric2.alignment.display_fixes import (
    DominanceActivity,
    _clamp_pathological,
    _fold_stranded_repeats,
    _pull_disconnected_tails,
    _snap_silent_heads,
    _stranded_sites,
    adlib_candidates,
    apply_stranded_corrections,
    dominance_activity_from_waveforms,
)
from everyric2.audio.vad import VocalRegion
from everyric2.inference.prompt import SyncResult, WordSegment


# --------------------------------------------------------------------------
# 픽스처 헬퍼 — tests/test_clamp_timing.py · test_edge_fixes.py와 같은 관례
# --------------------------------------------------------------------------


def _line(text: str, start: float, end: float) -> SyncResult:
    return SyncResult(text=text, start_time=start, end_time=end)


def _wline(text: str, start: float, end: float, w0: float, w1: float) -> SyncResult:
    """글자 스팬이 [w0,w1]에 균등하게 깔린 라인 (라인 경계는 [start,end])."""
    step = (w1 - w0) / len(text)
    ws = [
        WordSegment(word=c, start=w0 + i * step, end=w0 + (i + 1) * step)
        for i, c in enumerate(text)
    ]
    return SyncResult(text=text, start_time=start, end_time=end, word_segments=ws)


def _line_with_segs(text: str, start: float, end: float, *spans: tuple[float, float]) -> SyncResult:
    segs = [WordSegment(word="x", start=s, end=e) for s, e in spans]
    return SyncResult(text=text, start_time=start, end_time=end, word_segments=segs)


def _region(start: float, end: float) -> VocalRegion:
    return VocalRegion(start=start, end=end, energy=0.1)


def _curve(total_sec: float, hop: float, base: float, spans=()) -> np.ndarray:
    """구간별 상수값 곡선. spans: [(start, end, value), ...] — 뒤가 앞을 덮어쓴다."""
    n = int(round(total_sec / hop))
    arr = np.full(n, base, dtype=np.float64)
    for s, e, v in spans:
        lo, hi = int(round(s / hop)), int(round(e / hop))
        arr[lo:hi] = v
    return arr


# ==========================================================================
# 1) _clamp_pathological — 병적으로 늘어진 라인 절단 (우세도 기반)
# ==========================================================================


def test_clamp_pathological_cuts_line_stranded_past_dominance_body():
    # 10초 라인, 실제 발성은 앞 3초뿐(커버리지 30%<50%) → body 리전 끝+0.3으로 절단
    line = _wline("가나다", 0.0, 10.0, 0.0, 3.0)
    regions = [_region(0.0, 3.0)]
    clamped = _clamp_pathological([line], regions)
    assert clamped == {0}
    assert line.end_time == pytest.approx(3.3)


def test_clamp_pathological_skips_short_lines():
    # 지속 8.0초는 "8초 초과"가 아니다(<=8.0 스킵) — 경계값
    line = _wline("가나다", 0.0, 8.0, 0.0, 3.0)
    regions = [_region(0.0, 3.0)]
    clamped = _clamp_pathological([line], regions)
    assert clamped == set()
    assert line.end_time == pytest.approx(8.0)


def test_clamp_pathological_skips_well_covered_lines():
    # 10초 라인이지만 발성 커버리지 60%>=50% → 정상 배치로 보고 건드리지 않는다
    line = _wline("가나다라마바", 0.0, 10.0, 0.0, 6.0)
    regions = [_region(0.0, 6.0)]
    clamped = _clamp_pathological([line], regions)
    assert clamped == set()
    assert line.end_time == pytest.approx(10.0)


# ==========================================================================
# 2) _fold_stranded_repeats — 간주에 좌초한 반복 렌디션 접기
# ==========================================================================


def test_fold_stranded_repeat_copies_previous_rhythm():
    prev = _line_with_segs(
        "같은가사", 10.0, 12.0, (10.0, 10.5), (10.5, 11.0), (11.0, 11.5), (11.5, 12.0)
    )
    stranded = _line_with_segs(
        "같은가사", 70.0, 77.0, (70.0, 73.5), (73.5, 75.0), (75.0, 76.0), (76.0, 77.0)
    )
    results = [prev, stranded]
    folded = _fold_stranded_repeats(results, regions=[])  # 간주 위 — 발성 커버리지 0
    assert folded == 1
    # 앞 렌디션 바로 뒤로 접힌다 (같은 길이 2.0초 유지)
    assert stranded.start_time == pytest.approx(12.0)
    assert stranded.end_time == pytest.approx(14.0)
    # 세그는 앞 렌디션의 리듬을 그대로 복사(같은 세그 수 → 비례 스케일 = 1.0)
    got = [(s.start, s.end) for s in stranded.word_segments]
    assert got == [
        pytest.approx((12.0, 12.5)),
        pytest.approx((12.5, 13.0)),
        pytest.approx((13.0, 13.5)),
        pytest.approx((13.5, 14.0)),
    ]


def test_fold_stranded_repeat_skips_different_text():
    prev = _line_with_segs("같은가사", 10.0, 12.0, (10.0, 11.0), (11.0, 12.0))
    other = _line_with_segs("다른가사", 70.0, 77.0, (70.0, 73.5), (73.5, 77.0))
    folded = _fold_stranded_repeats([prev, other], regions=[])
    assert folded == 0
    assert other.start_time == pytest.approx(70.0)


def test_fold_stranded_repeat_skips_when_line_has_real_vocal_coverage():
    # 좌초처럼 보여도 발성 커버리지가 30% 이상이면 정상 배치 — 건드리지 않는다
    prev = _line_with_segs("같은가사", 10.0, 12.0, (10.0, 11.0), (11.0, 12.0))
    covered = _line_with_segs("같은가사", 70.0, 77.0, (70.0, 73.5), (73.5, 77.0))
    regions = [_region(70.0, 75.0)]  # 7초 중 5초 커버 = 71%
    folded = _fold_stranded_repeats([prev, covered], regions)
    assert folded == 0
    assert covered.start_time == pytest.approx(70.0)


# ==========================================================================
# 3) _pull_disconnected_tails — 몸통과 끊긴 꼬리 세그 되당김
# ==========================================================================


def test_pull_disconnected_tail_two_segs_is_pulled():
    # rookie 0:14 원형: 몸통 [10,11] 뒤 3.3초 무음, 꼬리 2세그가 무음 위 좌초
    line = _line_with_segs("boo ing", 10.0, 14.7, (10.0, 11.0), (14.3, 14.5), (14.5, 14.7))
    pulled = _pull_disconnected_tails([line], regions=[])
    assert pulled == 1
    assert line.end_time == pytest.approx(11.3)
    assert line.word_segments[1].start == pytest.approx(11.3)
    assert line.word_segments[1].end == pytest.approx(11.3)
    assert line.word_segments[2].start == pytest.approx(11.3)
    assert line.word_segments[2].end == pytest.approx(11.3)
    # 몸통 세그는 손대지 않는다
    assert line.word_segments[0].start == pytest.approx(10.0)
    assert line.word_segments[0].end == pytest.approx(11.0)


def test_pull_disconnected_tails_guards_long_real_verse_after_long_rest():
    """오검출 1호(ロキ) — 긴 휴지 뒤 «진짜 뒷소절»(10세그+)은 잔해가 아니므로 당기지 않는다.

    はあ…(몸통, 2:36.8 끝) 뒤 8초 쉬고 寝言は寝て言えベイビー(2:45.8부터, 10세그)가
    이어지는 실제 가사 자리다. 꼬리 ≤2세그 가드가 없으면 이 10세그를 몸통 뒤로
    뭉개서 실제 가사 자리가 통째로 비워지고 추임새 띠로 오검출된다 — 반드시 막는다.
    """
    body = [(236.0, 236.8)]
    tail = [(245.8 + i * 0.5, 246.3 + i * 0.5) for i in range(10)]  # 10세그, 연속(간격 0)
    line = _line_with_segs("body + real verse", 236.0, 250.8, *body, *tail)
    original_segs = [(s.start, s.end) for s in line.word_segments]

    # 뒷소절이 실제로 소리 나고 있다는 사실도 같이 준다(가드는 세그 수만으로 막지만,
    # 이 조건이 빠졌다는 뜻은 아니라는 걸 보여주기 위해 함께 둔다)
    regions = [_region(245.8, 250.8)]

    pulled = _pull_disconnected_tails([line], regions)
    assert pulled == 0
    assert line.end_time == pytest.approx(250.8)
    assert [(s.start, s.end) for s in line.word_segments] == original_segs


def test_pull_disconnected_tail_skips_when_gap_is_short():
    line = _line_with_segs("x", 10.0, 12.0, (10.0, 11.0), (12.5, 12.6))
    # 몸통-꼬리 간격 1.5초 < 2.0초 문턱
    line2 = _line_with_segs("x", 10.0, 12.5, (10.0, 11.0), (12.5, 12.6))
    pulled = _pull_disconnected_tails([line2], regions=[])
    assert pulled == 0


def test_pull_disconnected_tail_skips_when_gap_has_real_vocal():
    # 몸통-꼬리 사이가 발성으로 대부분 채워져 있으면(연속 무음 <1.0초) 건드리지 않는다
    line = _line_with_segs("x", 10.0, 14.7, (10.0, 11.0), (14.3, 14.5), (14.5, 14.7))
    regions = [_region(11.2, 14.0)]  # 무음 최장 연속 = max(0.2, 0.7) = 0.7 < 1.0
    pulled = _pull_disconnected_tails([line], regions)
    assert pulled == 0


# ==========================================================================
# 4) _snap_silent_heads — 무음 위에 앉은 라인 머리를 첫 가창 온셋으로
# ==========================================================================


def test_snap_silent_head_moves_start_to_first_onset():
    # rookie 라인0 원형: 0~1.22초 디지털 무음(우세도 0 · -180dB), 1.22초부터 가창
    values = _curve(3.0, 0.01, base=0.0, spans=[(1.22, 3.0, 0.5)])
    db = _curve(3.0, 0.01, base=-180.0, spans=[(1.22, 3.0, -13.3)])
    activity = DominanceActivity(regions=[], values=values, hop=0.01, db=db)

    line = _line_with_segs("Boo", 0.0, 3.0, (0.0, 0.1), (0.1, 0.2), (0.2, 0.3))
    snapped = _snap_silent_heads([line], activity)

    assert snapped == 1
    assert line.start_time == pytest.approx(1.17)
    got = [(round(s.start, 2), round(s.end, 2)) for s in line.word_segments]
    assert got == [(1.17, 1.27), (1.27, 1.37), (1.37, 1.47)]


def test_snap_silent_head_skips_already_active_line():
    values = _curve(3.0, 0.01, base=0.0, spans=[(1.22, 3.0, 0.5)])
    db = _curve(3.0, 0.01, base=-180.0, spans=[(1.22, 3.0, -13.3)])
    activity = DominanceActivity(regions=[], values=values, hop=0.01, db=db)

    line = _line("X", 2.0, 3.0)  # 우세도 0.5 위에서 이미 시작
    snapped = _snap_silent_heads([line], activity)
    assert snapped == 0
    assert line.start_time == pytest.approx(2.0)


def test_snap_silent_head_guards_against_loud_non_dominant_lead_in():
    """결정적 가드 — 리드인에 «들리지만 우세하지 않은» 소리가 있으면 스냅하지 않는다.

    우세도만 보면(0.2~0.3초 구간의 낮은 우세도) 온셋 탐색이 이 구간을 그냥 지나쳐
    1.0초의 진짜 온셋에서 멈춘다. 하지만 그 리드인 안에 -10dB짜리 들리는 소리가 있다면
    거기는 "명백한 무음"이 아니다 — 스냅하면 그 소리를 잘라먹는다. 절대 크기 가드가
    없으면 이 케이스도 스냅돼 버린다(문서화된 소프트 어택 오검출).
    """
    values = _curve(3.0, 0.01, base=0.0, spans=[(1.0, 3.0, 0.5)])
    db = _curve(
        3.0, 0.01, base=-180.0, spans=[(0.2, 0.3, -10.0), (1.0, 3.0, -13.0)],
    )
    activity = DominanceActivity(regions=[], values=values, hop=0.01, db=db)

    line = _line_with_segs("Boo", 0.0, 3.0, (0.0, 0.1), (0.1, 0.2), (0.2, 0.3))
    snapped = _snap_silent_heads([line], activity)
    assert snapped == 0
    assert line.start_time == pytest.approx(0.0)


# ==========================================================================
# 5) _stranded_sites — 좌초 시그니처 탐지기 (이동 장치가 아니다)
# ==========================================================================


def test_stranded_sites_detects_long_gap_with_unexplained_voice():
    prev = _line("A", 90.0, 100.0)
    line = _line("B", 105.0, 110.0)
    regions = [_region(101.0, 103.0)]  # 갭(100~105) 안의 미설명 발성 2.0초 >=1.5
    assert _stranded_sites([prev, line], regions) == 1


def test_stranded_sites_ignores_short_gap():
    prev = _line("A", 90.0, 100.0)
    line = _line("B", 101.5, 105.0)  # 갭 1.5초 < 3.0
    assert _stranded_sites([prev, line], regions=[]) == 0


def test_stranded_sites_ignores_line_with_covered_head():
    prev = _line("A", 90.0, 100.0)
    line = _line("B", 105.0, 110.0)
    regions = [_region(105.0, 105.5)]  # 다음 라인 머리 1초 커버리지 50%>=0.2 → 정상 배치
    assert _stranded_sites([prev, line], regions) == 0


def test_stranded_sites_ignores_gap_without_enough_voice():
    prev = _line("A", 90.0, 100.0)
    line = _line("B", 105.0, 110.0)
    regions = [_region(101.0, 101.8)]  # 갭 안 발성 0.8초 < 1.5
    assert _stranded_sites([prev, line], regions) == 0


# ==========================================================================
# 6) apply_stranded_corrections — 접기 → 꼬리 되당김 → 머리 스냅 → 좌초 탐지
# ==========================================================================


def test_apply_stranded_corrections_runs_in_documented_order():
    prev = _line_with_segs(
        "같은가사", 10.0, 12.0, (10.0, 10.5), (10.5, 11.0), (11.0, 11.5), (11.5, 12.0)
    )
    stranded = _line_with_segs(
        "같은가사", 70.0, 77.0, (70.0, 73.5), (73.5, 75.0), (75.0, 76.0), (76.0, 77.0)
    )
    activity = DominanceActivity(
        regions=[],
        values=_curve(1.0, 0.01, base=0.0),
        hop=0.01,
        db=_curve(1.0, 0.01, base=-180.0),
    )
    stats = apply_stranded_corrections([prev, stranded], activity)
    assert stats["folded_lines"] == 1
    assert stats["pulled_tails"] == 0
    assert stats["snapped_heads"] == 0
    assert "stranded_sites" in stats
    # 접기가 먼저 돌아 stranded가 제자리로 옮겨졌다
    assert stranded.start_time == pytest.approx(12.0)


# ==========================================================================
# 7) dominance_activity_from_waveforms — 우세도 곡선을 region+dB로 감싸기
# ==========================================================================


def test_dominance_activity_from_waveforms_separates_interlude_from_singing():
    # test_star_prior.py의 우세도 실측 시나리오 재사용: 0~6초 간주(반주 우세), 6초~ 가창
    sr = 16000
    t = np.arange(12 * sr) / sr
    bleed = 0.05 * np.sin(2 * np.pi * 220 * t)
    voice = np.where(t >= 6.0, 0.5, 0.0) * np.sin(2 * np.pi * 220 * t)
    vocals = voice + bleed
    accomp = np.where(t < 6.0, 0.6, 0.2) * np.sin(2 * np.pi * 110 * t)

    activity = dominance_activity_from_waveforms(vocals, accomp, sr)
    assert activity is not None
    assert activity.regions, "가창 구간에서 region이 하나는 나와야 한다"
    # 모든 region은 간주(0~6초)가 아니라 가창(6초~) 쪽에 있어야 한다
    assert all(reg.start >= 5.5 for reg in activity.regions)
    assert activity.db is not None and len(activity.db) > 0


def test_dominance_activity_from_waveforms_returns_none_for_degenerate_input():
    assert dominance_activity_from_waveforms(np.zeros(5), np.zeros(5), 16000) is None


def test_dominance_activity_from_waveforms_returns_none_without_any_dominant_region():
    # 보컬이 끝까지 무음이면(반주만 있음) 0.35를 못 넘는 region이 하나도 없다
    sr = 16000
    t = np.arange(4 * sr) / sr
    vocals = np.zeros_like(t)
    accomp = 0.5 * np.sin(2 * np.pi * 110 * t)
    assert dominance_activity_from_waveforms(vocals, accomp, sr) is None


# ==========================================================================
# 8) adlib_candidates — 설명 안 된 가창 구간 (곡 단위, 응답 최상위용)
# ==========================================================================


def _adlib_activity(total_sec: float, value_spans, db_spans) -> DominanceActivity:
    hop = 0.1
    values = _curve(total_sec, hop, base=0.05, spans=value_spans)
    db = _curve(total_sec, hop, base=-40.0, spans=db_spans)
    return DominanceActivity(regions=[], values=values, hop=hop, db=db)


def test_adlib_candidate_detected_when_far_from_neighbors_dipped_and_loud():
    claimed = _line_with_segs("가사", 0.0, 0.0, (0.0, 2.0))
    activity = _adlib_activity(
        20.0, value_spans=[(6.0, 6.5, 0.5)], db_spans=[(6.0, 6.5, -15.0)],
    )
    out = adlib_candidates([claimed], activity)
    assert out is not None
    assert len(out) == 1
    t0, t1 = out[0]
    assert t0 == pytest.approx(6.0)
    assert t1 == pytest.approx(6.5)


def test_adlib_candidate_rejected_when_touching_following_lyric_onset():
    # 다음 라인 온셋(6.6)과 0.05초밖에 안 떨어짐 < ADLIB_MIN_TAIL_GAP(0.10) → 그 온셋의
    # 일부이지 추임새가 아니다
    lines = [
        _line_with_segs("가사1", 0.0, 0.0, (0.0, 2.0)),
        _line_with_segs("가사2", 0.0, 0.0, (6.6, 7.0)),
    ]
    activity = _adlib_activity(
        20.0, value_spans=[(6.0, 6.55, 0.5)], db_spans=[(6.0, 6.55, -15.0)],
    )
    assert adlib_candidates(lines, activity) is None


def test_adlib_candidate_rejected_when_touching_preceding_lyric_onset():
    # 직전 라인 온셋(5.8)과 0.2초밖에 안 떨어짐 < ADLIB_MIN_ONSET_GAP(0.35) → 그 음절의
    # 끝자락이지 추임새가 아니다
    lines = [_line_with_segs("가사", 0.0, 0.0, (5.8, 6.0))]
    activity = _adlib_activity(
        20.0, value_spans=[(6.0, 6.5, 0.5)], db_spans=[(6.0, 6.5, -15.0)],
    )
    assert adlib_candidates(lines, activity) is None


def test_adlib_candidate_rejected_without_dip_before_onset():
    # 후보 직전 0.4초 창이 이미 우세도 0.30으로 떠 있으면(딥이 없으면) 늘임음의 연속이다.
    # 딥 창 시작 여유를 0.6초 더 둔다 — int(x/hop) 절단(부동소수 오차, 원본 그대로 포팅)
    # 때문에 창이 의도한 지점보다 한 프레임 이를 수 있어, 경계에 딱 붙이면 테스트가
    # 우연히 그 절단 오차에 좌우된다.
    lines = [_line_with_segs("가사", 0.0, 0.0, (0.0, 2.0))]
    activity = _adlib_activity(
        20.0,
        value_spans=[(5.0, 6.0, 0.30), (6.0, 6.5, 0.5)],
        db_spans=[(6.0, 6.5, -15.0)],
    )
    assert adlib_candidates(lines, activity) is None


def test_adlib_candidate_rejected_when_too_quiet():
    # 우세도는 0.35를 넘지만(디지털 무음의 0/0 잡음) 절대 크기가 -30dB 밑이면 기각
    lines = [_line_with_segs("가사", 0.0, 0.0, (0.0, 2.0))]
    activity = _adlib_activity(20.0, value_spans=[(6.0, 6.5, 0.5)], db_spans=[])
    assert adlib_candidates(lines, activity) is None


def test_adlib_long_continuation_bypasses_onset_and_dip_rules():
    # ⓪ 예외 — 3초 넘게 이어지면(온셋 간격 0이어도) 늘임음/온셋 규칙을 면제하고 절대 크기만 본다
    lines = [_line_with_segs("가사", 0.0, 0.0, (0.0, 2.0))]
    activity = _adlib_activity(
        20.0, value_spans=[(2.0, 6.0, 0.5)], db_spans=[(2.0, 6.0, -15.0)],
    )
    out = adlib_candidates(lines, activity)
    assert out is not None
    t0, t1 = out[0]
    assert t0 == pytest.approx(2.0)
    assert t1 == pytest.approx(6.0)


def test_adlib_candidates_returns_none_when_no_segments():
    activity = _adlib_activity(20.0, value_spans=[(6.0, 6.5, 0.5)], db_spans=[(6.0, 6.5, -15.0)])
    assert adlib_candidates([_line("가사", 0.0, 2.0)], activity) is None
