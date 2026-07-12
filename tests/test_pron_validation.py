"""독음 품질 가드 테스트 — 가나 혼입 감지·재시도 병합 + 가나 읽기 프롬프트 힌트."""
from everyric2.config.settings import TranslationSettings
from everyric2.server.api.translate import bad_pron_indices, merge_pron_retry
from everyric2.translation.translator import (
    GeminiTranslator,
    TranslationLine,
    _kana_readings,
)


def _line(pron):
    return TranslationLine(original="原文", translation="번역", pronunciation=pron)


class TestBadPronDetection:
    def test_detects_kana_leak(self):
        lines = [_line("도케이노 하리가"), _line("카에데노 타네가 체ン바"), _line(None)]
        assert bad_pron_indices(lines) == [1]

    def test_clean_hangul_passes(self):
        assert bad_pron_indices([_line("쿠루쿠루토"), _line("즛토 미에테")]) == []


class TestMergePronRetry:
    def test_replaces_with_clean_retry(self):
        lines = [_line("오츠 에코오 체ン바")]
        retry = [_line("오치루 에코오 첸바")]
        assert merge_pron_retry(lines, retry, [0]) == 1
        assert lines[0].pronunciation == "오치루 에코오 첸바"

    def test_strips_kana_when_retry_also_bad(self):
        lines = [_line("아주 긴 한글 독음 표기에 ン 하나")]
        retry = [_line("여전히 ガ 오염")]
        merge_pron_retry(lines, retry, [0])
        assert "ン" not in (lines[0].pronunciation or "")
        assert lines[0].pronunciation  # 소량 오염은 제거 후 유지

    def test_drops_pron_when_mostly_kana(self):
        lines = [_line("ほとんどカナだけの行 한")]
        merge_pron_retry(lines, None, [0])
        assert lines[0].pronunciation is None


class TestKanaReadingHints:
    def test_readings_for_japanese(self):
        assert _kana_readings("消えないで\nずっと見えて") == ["きえないで", "ずっとみえて"]

    def test_none_for_non_japanese(self):
        assert _kana_readings("hello world\nsecond line") is None

    def test_prompt_includes_hints_only_with_pronunciation(self):
        t = GeminiTranslator(TranslationSettings())
        p = t._build_prompt("消えないで", "ja", "ko", True)
        assert "KANA READINGS" in p and "きえないで" in p
        p2 = t._build_prompt("消えないで", "ja", "ko", False)
        assert "KANA" not in p2 and "pronunciation" not in p2
