"""독음(ko) 정렬 채택 파이프라인 테스트.

세 축을 검증한다.
- ``map_pron_alignment_to_line``: 발음 음절 정렬을 원문 글자에 역매핑 (다음절 한자 분할 포함).
- ``_pron_coverage``: 발음 커버리지 게이트 (>=0.9에서만 ko 경로).
- ``_snap_silence_undershoot``: 독음 정렬의 무음 언더슛(전이 라인 좌초) 교정.
"""
import pytest

from everyric2.audio.vad import VADResult, VocalRegion
from everyric2.inference.prompt import LyricLine, SyncResult, WordSegment
from everyric2.text.reading import map_pron_alignment_to_line
from everyric2.server.worker import (
    _pron_by_text,
    _pron_coverage,
    _shift_word_segments,
    _snap_silence_undershoot,
)


def _line(text: str, start: float, end: float, words=None) -> SyncResult:
    ws = [WordSegment(w, s, e) for w, s, e in (words or [])] or None
    return SyncResult(text=text, start_time=start, end_time=end, word_segments=ws)


def _vad(*spans: tuple[float, float]) -> VADResult:
    regions = [VocalRegion(start=s, end=e, energy=0.1) for s, e in spans]
    total = max((e for _, e in spans), default=0.0)
    return VADResult(
        regions=regions,
        sample_rate=16000,
        frame_duration=0.02,
        energy_threshold=0.01,
        total_duration=total,
    )


# --- 역매핑: 발음 음절 → 원문 글자 -----------------------------------------


def test_map_simple_hiragana_one_to_one():
    # ねこ = 네코, 음절 1:1 → 원문 글자 2개에 각 음절 시간이 그대로 실린다
    words, pron_segments = map_pron_alignment_to_line(
        "ねこ", "네코", [("네", 1.0, 1.2), ("코", 1.4, 1.6)]
    )
    assert [p["text"] for p in pron_segments] == ["네", "코"]
    assert words is not None
    assert [w["word"] for w in words] == ["ね", "こ"]
    assert words[0]["start"] == pytest.approx(1.0)
    assert words[1]["end"] == pytest.approx(1.6)


def test_map_multisyllable_kanji_splits_pron_but_merges_word():
    # 熱 = ネツ = 네츠: pron_segments는 음절 2개(노트 분할용), 원문 글자는 熱 하나로 병합
    words, pron_segments = map_pron_alignment_to_line(
        "熱", "네츠", [("네", 5.0, 5.1), ("츠", 5.3, 5.4)]
    )
    assert [p["text"] for p in pron_segments] == ["네", "츠"]
    assert len(pron_segments) == 2  # 음절 단위 → 熱을 두 노트로 쪼갤 수 있다
    assert words is not None
    assert [w["word"] for w in words] == ["熱"]  # 원문 글자는 1개
    assert words[0]["start"] == pytest.approx(5.0)
    assert words[0]["end"] == pytest.approx(5.4)  # 두 음절 구간 전체를 덮는다


def test_map_low_quality_alignment_drops_words_keeps_pron():
    # 발음이 원문과 안 맞으면(품질 미달) 원문 글자 매핑은 포기(None)하되,
    # 발음 음절 스팬(pron_segments)은 그대로 반환 → 라인 타이밍/발음 표시는 유지
    words, pron_segments = map_pron_alignment_to_line(
        "ねこ", "바보", [("바", 1.0, 1.2), ("보", 1.4, 1.6)]
    )
    assert words is None
    assert pron_segments is not None
    assert [p["text"] for p in pron_segments] == ["바", "보"]


def test_map_empty_inputs_return_none():
    assert map_pron_alignment_to_line("ねこ", "", []) == (None, None)
    assert map_pron_alignment_to_line("ねこ", "네코", []) == (None, None)


def test_map_pron_segments_are_monotonic():
    # 겹치는 입력 스팬이 들어와도 pron_segments는 단조 증가로 클램프된다
    words, pron_segments = map_pron_alignment_to_line(
        "ねこ", "네코", [("네", 1.0, 1.5), ("코", 1.2, 1.6)]
    )
    assert pron_segments[1]["start"] >= pron_segments[0]["end"]


# --- 글자별 confidence 역매핑 (회귀 수정) -----------------------------------


def test_map_confidence_distributed_per_char():
    # 음절별 conf가 다르면 글자 conf도 균일하지 않게 분배된다 (라인 균일 부여 회귀 방지).
    # 猫が: 猫=ねこ(네·코 두 음절, 한 글자), が=가 → 글자별로 conf가 갈린다.
    words, _pron = map_pron_alignment_to_line(
        "猫が", "네코가", [("네", 1.0, 1.2, 0.9), ("코", 1.4, 1.6, 0.9), ("가", 1.8, 2.0, 0.1)]
    )
    assert words is not None
    assert [w["word"] for w in words] == ["猫", "が"]
    assert words[0]["confidence"] == pytest.approx(0.9)  # 猫 = geomean(0.9, 0.9)
    assert words[1]["confidence"] == pytest.approx(0.1)  # が
    assert words[0]["confidence"] != words[1]["confidence"]  # 글자별로 다름


def test_map_pron_segments_carry_confidence():
    # pron_segments 각 음절에도 confidence가 실린다 (클라 미래 사용)
    _words, pron_segments = map_pron_alignment_to_line(
        "ねこ", "네코", [("네", 1.0, 1.2, 0.9), ("코", 1.4, 1.6, 0.2)]
    )
    assert [round(p["confidence"], 6) for p in pron_segments] == [0.9, 0.2]


def test_map_multisyllable_kanji_conf_is_geomean():
    # 熱=네츠: 두 음절 conf(0.9, 0.1)가 한 글자로 병합 → 글자 conf는 기하평균 sqrt(0.09)=0.3
    import math

    words, _pron = map_pron_alignment_to_line(
        "熱", "네츠", [("네", 5.0, 5.1, 0.9), ("츠", 5.3, 5.4, 0.1)]
    )
    assert words is not None and len(words) == 1
    assert words[0]["confidence"] == pytest.approx(math.exp((math.log(0.9) + math.log(0.1)) / 2))


def test_map_backward_compat_three_tuple_no_confidence():
    # conf 없는 3-튜플 입력은 기존대로 동작 — words에 confidence 키가 붙지 않는다
    words, pron_segments = map_pron_alignment_to_line(
        "ねこ", "네코", [("네", 1.0, 1.2), ("코", 1.4, 1.6)]
    )
    assert words is not None
    assert all("confidence" not in w for w in words)
    assert all("confidence" not in p for p in pron_segments)


# --- 커버리지 게이트 --------------------------------------------------------


def _lines(*texts: str):
    return [LyricLine(text=t, line_number=i + 1) for i, t in enumerate(texts)]


def test_coverage_full():
    meta = [
        {"text": "あ", "pronunciation": "아"},
        {"text": "い", "pronunciation": "이"},
    ]
    by_text = _pron_by_text(meta)
    assert _pron_coverage(_lines("あ", "い"), by_text) == pytest.approx(1.0)


def test_coverage_gate_below_threshold():
    # 4줄 중 3줄만 발음 → 0.75 < 0.9 (호출부에서 원문 폴백)
    meta = [
        {"text": "a", "pronunciation": "에이"},
        {"text": "b", "pronunciation": "비"},
        {"text": "c", "pronunciation": "시"},
        {"text": "d", "pronunciation": None},
    ]
    by_text = _pron_by_text(meta)
    assert _pron_coverage(_lines("a", "b", "c", "d"), by_text) == pytest.approx(0.75)


def test_coverage_ignores_blank_pron_and_missing_match():
    meta = [{"text": "a", "pronunciation": "  "}, {"text": "b", "pronunciation": "비"}]
    by_text = _pron_by_text(meta)
    # a는 공백 발음(무효), c는 메타 없음 → 3줄 중 1줄만 유효
    assert _pron_coverage(_lines("a", "b", "c"), by_text) == pytest.approx(1 / 3)


# --- 무음 언더슛 가드 -------------------------------------------------------


def test_undershoot_whole_line_snapped_forward():
    # 熱異常 L94 재현: 라인 전체가 간주 무음(146~147s)에 좌초, 실제 가창은 165s부터.
    # 길이 보존하며 통째로 다음 온셋(165 - 0.15)으로 이동한다.
    results = [
        _line("간주 앞", 128.0, 130.0),
        _line("전이 라인", 146.0, 147.0),  # 무음에 좌초 (dur 1.0)
        _line("가창 라인", 167.6, 169.0),
    ]
    vad = _vad((100.0, 131.0), (165.0, 200.0))
    clamped: set[int] = set()
    _snap_silence_undershoot(results, vad, clamped)
    assert clamped == {1}
    assert results[1].start_time == pytest.approx(164.85)  # 165.0 - 0.15
    assert results[1].end_time == pytest.approx(165.85)  # 길이 1.0 보존
    assert results[0].start_time == pytest.approx(128.0)  # 앞 라인 불변


def test_undershoot_start_only_snapped_when_line_overlaps_region():
    # 대부분 무음(커버리지<0.25)이지만 끝이 다음 온셋을 살짝 넘긴 라인은 start만 스냅한다
    results = [
        _line("앞", 100.0, 105.0),
        _line("걸친 라인", 160.0, 166.0),  # [160,166]∩[165,200]=1.0/6=0.17 → 좌초
    ]
    vad = _vad((100.0, 106.0), (165.0, 200.0))
    clamped: set[int] = set()
    _snap_silence_undershoot(results, vad, clamped)
    assert clamped == {1}
    assert results[1].start_time == pytest.approx(164.85)
    assert results[1].end_time == pytest.approx(166.0)  # 끝 유지(온셋 넘김)


def test_undershoot_healthy_line_not_touched():
    # 발성과 유의미하게 겹치는(커버리지>=0.25) 정상 라인은 건드리지 않는다
    results = [_line("정상", 166.0, 168.0)]
    vad = _vad((165.0, 200.0))
    clamped: set[int] = set()
    _snap_silence_undershoot(results, vad, clamped)
    assert clamped == set()
    assert results[0].start_time == pytest.approx(166.0)


def test_undershoot_partial_coverage_line_left_alone():
    # 절반이 발성 위(커버리지 0.5)면 언더슛이 아니므로 보수적으로 건드리지 않는다
    results = [_line("절반 걸침", 160.0, 170.0)]  # [160,170]∩[165,200]=5/10=0.5
    vad = _vad((165.0, 200.0))
    clamped: set[int] = set()
    _snap_silence_undershoot(results, vad, clamped)
    assert clamped == set()
    assert results[0].start_time == pytest.approx(160.0)


def test_undershoot_short_gap_not_touched():
    # 무음에 좌초했어도 다음 온셋까지 <1.5s면 온셋 직전 리드타임으로 보고 스냅하지 않는다
    results = [_line("리드타임", 163.6, 164.0)]  # 커버리지 0, 온셋(165)까지 1.4s
    vad = _vad((165.0, 200.0))
    clamped: set[int] = set()
    _snap_silence_undershoot(results, vad, clamped)
    assert clamped == set()
    assert results[0].start_time == pytest.approx(163.6)


def test_undershoot_skips_already_clamped():
    results = [_line("이미 처리됨", 146.0, 147.0)]
    vad = _vad((165.0, 200.0))
    clamped: set[int] = {0}
    _snap_silence_undershoot(results, vad, clamped)
    assert results[0].start_time == pytest.approx(146.0)  # 불변


def test_undershoot_no_following_region_not_touched():
    # 뒤에 발성 리전이 없으면(곡 끝 무음) 스냅할 온셋이 없다
    results = [_line("끝 무음", 250.0, 251.0)]
    vad = _vad((100.0, 200.0))
    clamped: set[int] = set()
    _snap_silence_undershoot(results, vad, clamped)
    assert clamped == set()
    assert results[0].start_time == pytest.approx(250.0)


def test_undershoot_not_applied_when_crossing_next_line():
    # 스냅 결과가 다음 라인 시작을 침범하면(목표 온셋 > 다음 라인 시작) 오탐으로 보고
    # 그 라인은 건드리지 않는다 (순서 역전 방지)
    results = [
        _line("좌초", 146.0, 147.0),
        _line("먼저 온 다음 줄", 150.0, 152.0),
    ]
    vad = _vad((165.0, 200.0))
    clamped: set[int] = set()
    _snap_silence_undershoot(results, vad, clamped)
    # 라인0 목표 온셋 164.85 >= 다음 라인 시작 150.0 → 라인0은 스킵
    assert 0 not in clamped
    assert results[0].start_time == pytest.approx(146.0)


def test_shift_word_segments_rescales_into_window():
    ws = [WordSegment("가", 146.0, 146.5), WordSegment("나", 146.5, 147.0)]
    _shift_word_segments(ws, 164.85, 165.85)
    assert ws[0].start == pytest.approx(164.85)
    assert ws[-1].end == pytest.approx(165.85)
    assert ws[0].end <= ws[1].start + 1e-9  # 순서 보존
