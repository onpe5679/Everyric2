"""가나→한글 결정적 변환 테스트 — 사용자 실측 오류 케이스가 회귀 기준.

실측: 쿤니다케(君 오독), 즈토(촉음 소실), 쯔우토(억지 장음), 지부가(ん 소실),
케나이데(消え 오독), 체ン바(가타카나 혼입).
"""

from everyric2.text.kana_hangul import (
    finalize_pronunciation,
    has_kana,
    has_kanji,
    kana_to_hangul,
)


class TestKanaToHangul:
    def test_sokuon_becomes_siot_batchim(self):
        assert kana_to_hangul("ずっと") == "즛토"
        assert kana_to_hangul("ずっと みえて") == "즛토 미에테"

    def test_n_becomes_nieun_batchim(self):
        assert kana_to_hangul("じぶんが") == "지분가"
        assert kana_to_hangul("しんぱい") == "신파이"

    def test_kimi_ni_dake(self):
        assert kana_to_hangul("きみにだけ") == "키미니다케"
        assert kana_to_hangul("きみ に だけ の") == "키미 니 다케 노"

    def test_kienaide(self):
        assert kana_to_hangul("きえないで") == "키에나이데"

    def test_katakana_normalized(self):
        assert kana_to_hangul("カエデノタネガ") == "카에데노타네가"

    def test_youon_digraphs(self):
        assert kana_to_hangul("ちゃんと") == "찬토"
        assert kana_to_hangul("しょうがない") == "쇼우가나이"
        assert kana_to_hangul("きゅう") == "큐우"

    def test_long_vowel_mark_repeats_vowel(self):
        assert kana_to_hangul("スーパー") == "스우파아"
        assert kana_to_hangul("メロディー") == "메로디이"

    def test_line_initial_sokuon_and_lone_n(self):
        assert kana_to_hangul("って") == "테"  # 붙일 음절 없음 → 촉음 무시
        assert kana_to_hangul("ん") == "응"

    def test_non_kana_passthrough(self):
        assert kana_to_hangul("とけい の はり が!") == "토케이 노 하리 가!"
        assert kana_to_hangul("abc 한글") == "abc 한글"

    def test_foreign_combos(self):
        assert kana_to_hangul("ふぁいと") == "파이토"
        assert kana_to_hangul("てぃあら") == "티아라"


class TestFinalize:
    def test_kana_input_converted(self):
        assert finalize_pronunciation("ずっと みえて") == "즛토 미에테"

    def test_hangul_input_kept(self):
        # 구형 프롬프트/비일본어 응답 — 이미 한글이면 그대로
        assert finalize_pronunciation("도케이노 하리가") == "도케이노 하리가"

    def test_mixed_kana_in_hangul_cleaned(self):
        # "체ン바" 류 혼입 — 가나만 변환되고 한글은 유지
        assert finalize_pronunciation("체ン바") == "첸바"

    def test_kanji_residual_falls_back_to_dictionary(self):
        out = finalize_pronunciation("時計 の はり が")
        assert out is not None
        assert not has_kanji(out)
        assert not has_kana(out)

    def test_empty_and_none(self):
        assert finalize_pronunciation(None) is None
        assert finalize_pronunciation("") == ""
