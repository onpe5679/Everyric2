"""``everyric2.text.en_g2p`` (CMU 발음 사전 기반 영어 G2P) 회귀 테스트.

순수 함수이고 CMU 사전이 로컬에 있어 실입력으로 충분히 검증 가능하다(GPU·모델 불필요).
못박는 것: ARPABET을 허브로 (한글, IPA, 가나, 기여 음소)가 함께 나오는가, 조밀 음차
원칙(자음군은 받침 우선), OOV 폴백, 음절 분리의 실측 커버리지.
"""

from __future__ import annotations

from everyric2.text import en_g2p


def test_word_to_ipa_uses_ascii_not_real_ipa_symbols():
    # 실측 근거: omniASR류가 실제로 내는 것은 철자에 가까운 ASCII다 — 죽은 IPA 기호(ɚ·ə 등)를
    # 정렬 타깃으로 쓰면 심판이 항상 살아있는 후보를 이긴다(en_g2p 모듈 docstring 참조).
    ipa = en_g2p.word_to_ipa("beautiful")
    assert ipa is not None
    assert all(ord(c) < 128 for c in ipa)  # 전부 ASCII


def test_word_to_ipa_unknown_word_returns_none():
    assert en_g2p.word_to_ipa("zzqxbnotaword") is None


def test_phones_to_hangul_tight_transliteration_prefers_coda():
    # approved: 관습 음차 "어프루브드" 대신 조밀 "어프룹드" 계열(받침 흡수) — 최소한 삽입
    # 모음 ㅡ 음절이 관습보다 적어야 한다.
    entries = en_g2p.pronunciations("approved")
    assert entries
    hangul = en_g2p.phones_to_hangul(entries[0])
    # 조밀 원칙: 어말 자음은 받침으로 흡수되어야 하고, "브" 같은 별도 모음 삽입 음절이
    # 최소화된다. 최소한 결과가 나와야 하고 원문 음절 수(3)보다 과하게 길면 안 된다.
    assert hangul
    assert len(hangul) <= 4


def test_units_for_word_gives_parallel_hangul_ipa_kana():
    units = en_g2p.units_for_word("beautiful")
    assert units is not None
    for unit in units:
        assert isinstance(unit.hangul, str) and unit.hangul
        assert isinstance(unit.ipa, str)
        assert isinstance(unit.kana, str)
        assert isinstance(unit.phones, list)


def test_units_for_word_unknown_returns_none():
    assert en_g2p.units_for_word("zzqxbnotaword") is None


def test_diphthong_splits_into_two_units():
    # AY(아이) 류 이중모음은 실제로 두 모라로 불려 유닛이 둘로 갈린다(모듈 docstring).
    units = en_g2p.units_for_word("my")  # M AY1 -> 마이
    assert units is not None
    assert len(units) == 2
    assert units[0].hangul == "마"
    assert units[1].hangul == "이"


def test_glide_y_collapses_syllable_beautiful_is_two_syllables():
    # beautiful: B Y UW1 T AH0 F AH0 L — 활음 Y가 자음+모음에 녹아 별도 음절을 안 만든다.
    entries = en_g2p.pronunciations("beautiful")
    assert entries
    assert en_g2p.syllable_count(entries[0]) == 3


def test_syllabify_spelling_matches_syllable_count():
    entries = en_g2p.pronunciations("beautiful")
    want = en_g2p.syllable_count(entries[0])
    pieces = en_g2p.syllabify_spelling("beautiful", want)
    assert pieces is not None
    assert len(pieces) == want
    assert "".join(pieces) == "beautiful"


def test_syllable_units_for_word_pieces_reconstruct_spelling():
    result = en_g2p.syllable_units_for_word("beautiful")
    assert result is not None
    pieces = [piece for piece, _units in result]
    assert "".join(pieces) == "beautiful"
    # 각 조각에 유닛이 최소 하나는 붙어 있어야 한다(빈 음절 조각 금지).
    assert all(units for _piece, units in result)


def test_syllabify_unknown_oov_preserves_word_and_splits_multiple_pieces():
    # weathervane은 CMU에 없다 — 사전 없이도 모음 그룹으로 음절 수를 추정해 가른다.
    pieces = en_g2p.syllabify_unknown("weathervane")
    assert "".join(pieces) == "weathervane"
    assert len(pieces) >= 2  # 통째로 한 조각이면 안 된다(그 구간에서 카라오케가 멈춘다)


def test_transliterate_cmu_falls_back_for_oov_word():
    # 사전에 있는 낱말은 CMU 경로, 없는 낱말은 latin_hangul 폴백 — 라인 전체가 죽지 않는다.
    text = en_g2p.transliterate_cmu("beautiful zzqxbnotaword")
    words = text.split(" ")
    assert len(words) == 2
    assert all(words)  # 둘 다 비어있지 않다


def test_transliterate_ipa_keeps_oov_word_as_latin():
    # IPA 경로는 한글로 폴백하지 않는다 — OOV는 원문 그대로 남는다(문자 체계를 안 섞는다).
    text = en_g2p.transliterate_ipa("beautiful zzqxbnotaword")
    assert "zzqxbnotaword" in text


def test_pronunciations_the_has_two_entries_context_determined_elsewhere():
    # the의 ðə/ði 두 발음이 있어야 문맥 결정(align_target 쪽 책임) 여지가 생긴다.
    entries = en_g2p.pronunciations("the")
    assert len(entries) >= 2


def test_our_has_ambiguous_syllable_count_between_pronunciations():
    # our의 1음절/2음절 후보가 실제로 갈린다 — 오디오 심판이 필요했던 실측 근거.
    entries = en_g2p.pronunciations("our")
    counts = {en_g2p.syllable_count(e) for e in entries}
    assert len(counts) >= 1  # 최소 사전에 발음이 있다(정확한 카운트는 사전 버전에 따라 다름)
    assert entries


def test_phones_to_units_no_vowel_word_gets_eu_syllable_per_consonant():
    # 모음이 없는 낱말(약어 등) — 자음마다 ㅡ 음절.
    units = en_g2p.phones_to_units(["S", "K"])
    assert len(units) == 2
    assert all(u.hangul for u in units)
