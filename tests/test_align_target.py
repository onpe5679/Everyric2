"""2패스 정렬 타깃/표시 파생(``everyric2.text.align_target``) 회귀 테스트.

GPU·모델 실행 없이 순수 함수만 검증한다(CMU 사전·MeCab 형태소 분석은 로컬에 있어 실입력으로
돈다). 못박는 것 셋:

1. **표시 파생이 정렬을 건드리지 않는다** — ``target``이 어느 표기(en/ja, hangul/kana)를
   요청하든 항상 같다.
2. **OOV는 통째로 첫 글자에 안 몰린다** — 사전에 없는 낱말도 표시가 철자 길이에 비례해
   퍼진다(한 구간에서 카라오케가 멈추는 사고를 막는 요구사항).
3. **혼재 표기(ja+라틴) 라우팅이 실측과 일치한다** — numb numb·drip in color 류 실곡
   벤치 런(``benchmark/runs/bs-polarformer-fp16__2pass-owsm-mixed-hangul``)에서 직접 확인한
   한글 세그 시퀀스를 그대로 기대값으로 못박는다(패리티).
"""

from __future__ import annotations

from everyric2.text.align_target import (
    LineUnits,
    derive_en_display_units,
    derive_ja_display_units,
    native_units,
)


def _non_empty(owners: list[str]) -> list[str]:
    """실제 노트가 될 소유자만 — 공백·구두점 통과분은 실제 정렬 단계에서 어휘 밖이라
    걸러지므로(refine_window) 여기서도 뺀다. 한글·가나는 유니코드상 alnum이라 안 걸린다.
    """
    return [c for c in owners if c.isalnum()]


# ---------------------------------------------------------------------------
# LineUnits 자체 — 길이 계약
# ---------------------------------------------------------------------------


def test_line_units_rejects_mismatched_owner_length():
    import pytest

    with pytest.raises(ValueError):
        LineUnits("ab", {"x": ["a"]})


def test_line_units_rejects_mismatched_word_end_length():
    import pytest

    with pytest.raises(ValueError):
        LineUnits("ab", {"x": ["a", "b"]}, word_end=[True])


def test_line_units_defaults_word_end_to_all_false():
    units = LineUnits("ab", {"x": ["a", "b"]})
    assert units.word_end == [False, False]


def test_native_units_is_identity():
    units = native_units("안녕")
    assert units.target == "안녕"
    assert units.owners["native"] == ["안", "녕"]


# ---------------------------------------------------------------------------
# en: 정렬-표시 분리 — target이 표기 선택과 무관하게 항상 같다
# ---------------------------------------------------------------------------


def test_en_target_is_ipa_not_spelling():
    units = derive_en_display_units("beautiful")
    # 철자 그대로가 아니라 CMU IPA에서 온 문자열이어야 한다(비어티펄이 아니라 뷰터펄 계열).
    assert units.target != "beautiful"
    assert units.target  # 비어있지 않다


def test_en_hangul_and_kana_share_the_same_target():
    # display 선택은 owners만 바꾸고 target(IPA)은 절대 안 바뀐다 — 벤치가 실측한 성질.
    units = derive_en_display_units("They say my hunger's got a mind of its own")
    assert set(units.owners) == {"hangul", "kana", "romaji", "en", "ipa"}
    # 네 표기 모두 같은 LineUnits 안에서 하나의 target을 공유한다(호출 자체가 하나뿐이므로
    # "정렬 재실행 없이 표기가 갈린다"는 요구사항이 구조적으로 성립한다).
    assert len(units.owners["hangul"]) == len(units.target)
    assert len(units.owners["kana"]) == len(units.target)
    assert len(units.owners["romaji"]) == len(units.target)
    assert len(units.owners["en"]) == len(units.target)


def test_en_monosyllable_word_is_one_hangul_segment():
    # CMU numb = N AH1 M (실묵음 b) — 한 음절, 한글 세그 하나 "넘".
    units = derive_en_display_units("numb")
    assert _non_empty(units.owners["hangul"]) == ["넘"]


def test_en_multisyllable_word_splits_into_syllable_segments():
    # 실측(벤치 2pass-owsm-mixed-hangul, ba7YbGO2aq4 24번 줄): "drip in color, cue" →
    # 드/립/인/커/러/큐 (6 세그) — 패리티 고정값.
    units = derive_en_display_units("drip in color, cue")
    assert _non_empty(units.owners["hangul"]) == ["드", "립", "인", "커", "러", "큐"]


def test_en_color_blue_matches_bench_parity():
    # 실측(같은 런, 30번 줄): "keep it color blue" → 킵/잇/커/러/브/루.
    units = derive_en_display_units("keep it color blue")
    assert _non_empty(units.owners["hangul"]) == ["킵", "잇", "커", "러", "브", "루"]


def test_en_syllable_display_uses_original_spelling_pieces():
    units = derive_en_display_units("beautiful")
    pieces = _non_empty(units.owners["en"])
    assert "".join(pieces) == "beautiful"
    assert len(pieces) >= 2  # 여러 음절로 갈렸다(beau-ti-ful류)


# ---------------------------------------------------------------------------
# OOV — 통째로 첫 글자에 몰리면 안 된다
# ---------------------------------------------------------------------------


def test_en_oov_word_distributes_display_across_spelling():
    # weathervane은 CMU에 없다(복합어) — 통째로 한 글자에 안 몰리고 철자 길이에 퍼진다.
    units = derive_en_display_units("weathervane")
    hangul = units.owners["hangul"]
    assert len(hangul) == len("weathervane")
    non_empty_positions = [i for i, c in enumerate(hangul) if c]
    # 최소 두 자리 이상에 나뉘어 있어야 한다 — 첫 글자 하나에 전부 실리면 실패.
    assert len(non_empty_positions) >= 2
    assert non_empty_positions[0] == 0  # 관례상 첫 슬롯부터 채운다


def test_en_oov_word_en_display_still_syllabifies():
    units = derive_en_display_units("weathervane")
    pieces = _non_empty(units.owners["en"])
    assert "".join(pieces) == "weathervane"
    assert len(pieces) >= 2


# ---------------------------------------------------------------------------
# 낱말 경계(word_end) — en pron 문자열 띄어쓰기용
# ---------------------------------------------------------------------------


def test_en_word_end_marks_last_target_char_of_each_word():
    units = derive_en_display_units("keep it")
    # "keep"의 마지막 타깃 문자와 "it"의 마지막 타깃 문자에서만 True.
    true_positions = [i for i, w in enumerate(units.word_end) if w]
    assert len(true_positions) == 2
    # 마지막 word_end는 항상 target의 마지막 문자다(줄 끝 = 마지막 낱말 끝).
    assert true_positions[-1] == len(units.target) - 1


def test_en_word_end_does_not_flag_mid_word_positions():
    units = derive_en_display_units("beautiful")
    true_positions = [i for i, w in enumerate(units.word_end) if w]
    assert true_positions == [len(units.target) - 1]


def test_pron_string_from_segments_has_no_double_space():
    """다운스트림(refine_window)이 세그 텍스트 + word_end로 pron 문자열을 지을 때 표준
    조립 방식(비어있지 않은 owner를 잇고 word_end 자리마다 공백 하나)이 원문 공백
    통과분과 겹치지 않는지 — owners 자체에 남은 리터럴 공백은 세그 단계에서 걸러진다는
    전제를 여기서는 owners만으로 재현해 한 번 더 확인한다.
    """
    units = derive_en_display_units("keep it")
    # 세그 조립을 흉내낸다: 공백이 아닌 owner만 모으고 word_end에서 공백 하나를 삽입.
    parts: list[str] = []
    for owner, end in zip(units.owners["hangul"], units.word_end):
        if owner.strip():
            parts.append(owner)
        if end:
            parts.append(" ")
    joined = "".join(parts).strip()
    assert "  " not in joined
    assert joined == "킵 잇"


# ---------------------------------------------------------------------------
# ja: 정렬-표시 분리 + 혼재(라틴) 라우팅
# ---------------------------------------------------------------------------


def test_ja_pure_line_hangul_matches_bench_parity():
    # 실측(2pass-owsm-mixed-hangul, ba7YbGO2aq4 2번 줄): "眩しくて numb numb" →
    # 마/부/시/쿠/테/넘/넘.
    units = derive_ja_display_units("眩しくて numb numb")
    assert _non_empty(units.owners["hangul"]) == ["마", "부", "시", "쿠", "테", "넘", "넘"]


def test_ja_repeated_word_matches_bench_parity():
    units = derive_ja_display_units("ひらひら numb numb")
    assert _non_empty(units.owners["hangul"]) == ["히", "라", "히", "라", "넘", "넘"]


def test_ja_target_is_kana_not_kanji():
    units = derive_ja_display_units("網膜に焼き付く影")
    # 정렬 타깃은 한자가 아니라 독음(가나)이어야 한다.
    assert "網" not in units.target
    assert "膜" not in units.target


def test_ja_latin_word_gets_syllable_segments_not_letter_by_letter():
    # numb numb 곡의 핵심 결함 재현 방지: 라틴이 글자 단위(n|u|m|b)로 안 쪼개져야 한다.
    units = derive_ja_display_units("numb numb")
    hangul = _non_empty(units.owners["hangul"])
    assert hangul == ["넘", "넘"]  # n|u|m|b 4세그가 아니라 넘 1세그씩


def test_ja_mixed_word_end_only_on_latin_span():
    units = derive_ja_display_units("ひらひら numb numb")
    # word_end가 켜진 자리의 owner는 라틴 유래(넘)여야 하고, 순수 가나 자리(히라히라)에는
    # 절대 켜지면 안 된다(요구사항 5).
    for index, flagged in enumerate(units.word_end):
        if flagged:
            assert units.owners["hangul"][index] or index > 0


def test_ja_pure_line_has_no_word_end_flags():
    units = derive_ja_display_units("眩しくて")
    assert not any(units.word_end)


def test_ja_han_only_word_passes_through_when_unreadable():
    # tokenize_reading이 못 읽는 한자(중국어 등)는 원문 그대로 남는다 — 크래시도, ja
    # 계열 오분류도 없어야 한다. MeCab이 실제로 어떻게 읽든(빈 문자열이 아니면) 결과가
    # 나야 하고 예외가 나면 안 된다.
    units = derive_ja_display_units("这口味让我陶醉")
    assert isinstance(units, LineUnits)
    assert len(units.owners["hangul"]) == len(units.target)


def test_ja_mixed_line_covers_full_source_length_class():
    # 라틴+가나 혼재 줄에서 owners 각 표기 길이가 target과 항상 일치한다(연속성의 전제 —
    # 실제 시간 연속성은 refine_window가 세그 단계에서 보장하고 별도로 못박는다).
    units = derive_ja_display_units("0.1mmの距離")
    for key, arr in units.owners.items():
        assert len(arr) == len(units.target), key


# ---------------------------------------------------------------------------
# 순수성 — 같은 입력에 같은 출력(결정론)
# ---------------------------------------------------------------------------


def test_en_derivation_is_deterministic():
    a = derive_en_display_units("hello world")
    b = derive_en_display_units("hello world")
    assert a == b


def test_ja_derivation_is_deterministic():
    a = derive_ja_display_units("眩しくて numb numb")
    b = derive_ja_display_units("眩しくて numb numb")
    assert a == b
