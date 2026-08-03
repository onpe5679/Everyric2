"""라인 메타(발음/번역) 매칭 회귀 테스트 — 정규화 키와 색인 우선순위.

실측 결손 2건:
  (6) `_normalize_line`이 공백만 접어, 구두점 앞뒤 띄어쓰기(`Are you ready ?`)나 전각/반각
      차이로 line_meta가 라인에 못 붙었다 (6줄 누락).
  (7) `by_text`가 동일 텍스트의 **첫 등장만** 채택해, 첫 등장이 비면 반복 후렴 전체가
      영구히 비었다 (후렴 5회 반복 곡에서 5회 전부 누락).
정규화가 과하면 반대로 서로 다른 라인이 같은 키가 돼 엉뚱한 메타가 붙으므로
(구두점 자체 제거·다른 낱말 병합) 그 위험도 함께 고정한다.
"""
from everyric2.server.worker import (
    _index_line_meta,
    _normalize_line,
    _pron_by_text,
    _pron_coverage,
    merge_line_meta,
)


class _Line:
    def __init__(self, text: str) -> None:
        self.text = text


# --- 수정 6: 표기차를 넘어 붙는다 --------------------------------------------


def test_punctuation_spacing_difference_still_matches():
    # 실측 누락 패턴: 메타 쪽만 물음표 앞에 공백이 있다
    segs = [{"text": "Are you ready?"}]
    meta = [{"text": "Are you ready ?", "pronunciation": "아 유 레디", "translation": "준비됐어?"}]
    assert merge_line_meta(segs, meta) == 1
    assert segs[0]["translation"] == "준비됐어?"
    assert segs[0]["pronunciation"] == "아 유 레디"


def test_fullwidth_halfwidth_and_ideographic_space_match():
    segs = [{"text": "行こう！"}, {"text": "ボク　は"}, {"text": "ＯＫ"}]
    meta = [
        {"text": "行こう!", "translation": "가자!"},
        {"text": "ボク は", "translation": "나는"},
        {"text": "OK", "translation": "좋아"},
    ]
    assert merge_line_meta(segs, meta) == 3
    assert [s["translation"] for s in segs] == ["가자!", "나는", "좋아"]


def test_zero_width_format_chars_are_ignored():
    segs = [{"text": "​誰も知らない"}]
    meta = [{"text": "誰も知らない", "translation": "아무도 모르는"}]
    assert merge_line_meta(segs, meta) == 1


# --- 과잉 정규화 방지: 서로 다른 라인이 같은 키가 되면 안 된다 -------------------


def test_different_punctuation_chars_do_not_collide():
    # 구두점까지 지우면 부호만 다른 별개 라인이 뭉쳐 엉뚱한 메타가 붙는다
    assert _normalize_line("行く。") != _normalize_line("行く？")
    segs = [{"text": "行く。"}, {"text": "行く？"}]
    meta = [
        {"text": "行く。", "translation": "간다"},
        {"text": "行く？", "translation": "갈까?"},
    ]
    assert merge_line_meta(segs, meta) == 2
    assert segs[0]["translation"] == "간다"
    assert segs[1]["translation"] == "갈까?"


def test_distinct_wording_does_not_collide():
    assert _normalize_line("君を待つ") != _normalize_line("君を待って")
    assert _normalize_line("Are you ready?") != _normalize_line("Are you steady?")
    # 대소문자는 접지 않는다 (다른 라인으로 남는다)
    assert _normalize_line("YES") != _normalize_line("yes")


def test_unmatched_line_is_left_untouched():
    segs = [{"text": "없는 라인"}]
    assert merge_line_meta(segs, [{"text": "다른 라인", "translation": "x"}]) == 0
    assert "translation" not in segs[0]


# --- 느슨한 매칭 폴백: 구두점 차이만으로 못 붙던 실측(2026-08-03) -----------------


def test_trailing_period_difference_falls_back_to_loose_match():
    # vg6pnvn1u10류 재현 — 엄격 키는 구두점을 보존해 서로 다른 키가 된다
    segs = [{"text": "I love you"}]
    meta = [{"text": "I love you.", "translation": "사랑해"}]
    assert merge_line_meta(segs, meta) == 1
    assert segs[0]["translation"] == "사랑해"


def test_strict_match_is_tried_before_the_loose_fallback():
    # 엄격 매칭이 성립하면 느슨한 폴백은 아예 안 쓰인다 — 뜻이 갈리는 구두점 쌍도
    # 각각 정확히 집힌다(둘 다 값이 있으면).
    segs = [{"text": "行く。"}, {"text": "行く？"}]
    meta = [
        {"text": "行く。", "translation": "간다"},
        {"text": "行く？", "translation": "갈까?"},
    ]
    assert merge_line_meta(segs, meta) == 2
    assert segs[0]["translation"] == "간다"
    assert segs[1]["translation"] == "갈까?"


def test_loose_fallback_does_not_collide_short_punctuation_only_lines():
    segs = [{"text": "?"}]
    meta = [{"text": "!", "translation": "완전히 다른 뜻"}]
    assert merge_line_meta(segs, meta) == 0
    assert "translation" not in segs[0]


# --- 수정 7: 값이 있는 첫 항목이 이긴다 ----------------------------------------


def test_repeated_chorus_prefers_first_entry_with_value():
    # 실측: 후렴 첫 등장이 비어 나머지 4회까지 전부 비었다
    meta = [
        {"text": "サビ", "pronunciation": "", "translation": ""},
        {"text": "サビ", "pronunciation": "사비", "translation": "후렴"},
        {"text": "サビ", "pronunciation": "다른", "translation": "다른 값"},
    ]
    segs = [{"text": "サビ"}, {"text": "サビ"}, {"text": "サビ"}]
    assert merge_line_meta(segs, meta) == 3
    assert all(s["translation"] == "후렴" for s in segs)  # 값 있는 첫 항목
    assert all(s["pronunciation"] == "사비" for s in segs)


def test_value_only_in_translation_still_wins_over_empty():
    meta = [
        {"text": "リフレイン", "pronunciation": None, "translation": None},
        {"text": "リフレイン", "translation": "후렴구"},
    ]
    assert _index_line_meta(meta)["リフレイン"]["translation"] == "후렴구"


def test_first_entry_wins_among_entries_that_all_have_values():
    meta = [
        {"text": "同じ", "translation": "첫 번째"},
        {"text": "同じ", "translation": "두 번째"},
    ]
    assert _index_line_meta(meta)["同じ"]["translation"] == "첫 번째"


def test_empty_text_entries_are_skipped():
    meta = [{"text": "", "translation": "x"}, {"text": "   ", "translation": "y"}]
    assert _index_line_meta(meta) == {}
    assert _index_line_meta(None) == {}


# --- 정렬 경로도 같은 색인 규칙을 쓴다 ------------------------------------------


def test_pron_index_and_coverage_share_the_value_first_rule():
    meta = [
        {"text": "サビ", "pronunciation": ""},
        {"text": "サビ", "pronunciation": "사비"},
        {"text": "Are you ready ?", "pronunciation": "아 유 레디"},
    ]
    by_text = _pron_by_text(meta)
    assert by_text[_normalize_line("サビ")]["pronunciation"] == "사비"
    # 커버리지도 함께 회복된다 — 예전에는 빈 첫 등장 때문에 독음 정렬 경로가 통째로 꺼졌다
    lines = [_Line("サビ"), _Line("サビ"), _Line("Are you ready?")]
    assert _pron_coverage(lines, by_text) == 1.0
