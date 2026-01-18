import pytest

from everyric2.alignment.base import WordTimestamp
from everyric2.alignment.matcher import LyricsMatcher
from everyric2.inference.prompt import LyricLine


@pytest.fixture
def matcher():
    return LyricsMatcher()


def test_normalize_text_english(matcher):
    text = "Hello, World! 123"
    # Normalized: "hello world 123" (punctuation removed, lowercased)
    assert matcher.normalize_text(text) == "hello world 123"


def test_normalize_text_japanese(matcher):
    # Japanese punctuation "、" "！" should be removed
    text = "こんにちは、世界！"
    assert matcher.normalize_text(text) == "こんにちは世界"


def test_normalize_text_korean(matcher):
    text = "안녕하세요, 세계!"
    assert matcher.normalize_text(text) == "안녕하세요 세계"


def test_tokenize_english(matcher):
    text = "Hello World"
    # Splits by space
    assert matcher.tokenize(text) == ["hello", "world"]


def test_tokenize_japanese(matcher):
    # Contiguous Japanese characters are kept together
    text = "こんにちは世界"
    assert matcher.tokenize(text) == ["こんにちは世界"]

    # Mixed script: English and Japanese
    text2 = "Love世界"
    assert matcher.tokenize(text2) == ["love", "世界"]

    # With spaces
    text3 = "こんにちは 世界"
    assert matcher.tokenize(text3) == ["こんにちは", "世界"]


def test_tokenize_korean(matcher):
    # Mixed script: Korean and English
    text = "안녕하세요World"
    assert matcher.tokenize(text) == ["안녕하세요", "world"]

    # With spaces
    text2 = "안녕 하세요"
    assert matcher.tokenize(text2) == ["안녕", "하세요"]


def test_match_lyrics_to_words_exact(matcher):
    lyrics = [LyricLine(text="Hello", line_number=1), LyricLine(text="World", line_number=2)]
    words = [
        WordTimestamp(word="Hello", start=0.0, end=1.0),
        WordTimestamp(word="World", start=1.5, end=2.5),
    ]

    results = matcher.match_lyrics_to_words(lyrics, words, language="en")

    assert len(results) == 2
    assert results[0].text == "Hello"
    assert results[0].start_time == 0.0
    assert results[0].end_time == 1.0
    assert results[0].confidence > 0.9

    assert results[1].text == "World"
    assert results[1].start_time == 1.5
    assert results[1].end_time == 2.5
    assert results[1].confidence > 0.9


def test_match_lyrics_to_words_partial(matcher):
    lyrics = [LyricLine(text="Hello", line_number=1), LyricLine(text="World", line_number=2)]
    # "Brave" and "New" are extra words in audio between lyrics
    words = [
        WordTimestamp(word="Hello", start=0.0, end=1.0),
        WordTimestamp(word="Brave", start=1.0, end=1.5),
        WordTimestamp(word="New", start=1.5, end=2.0),
        WordTimestamp(word="World", start=2.0, end=3.0),
    ]

    results = matcher.match_lyrics_to_words(lyrics, words, language="en")

    assert len(results) == 2
    assert results[0].text == "Hello"
    assert results[0].start_time == 0.0
    assert results[0].end_time >= 1.0

    assert results[1].text == "World"
    assert results[1].start_time == 2.0
    assert results[1].end_time >= 3.0


def test_interpolate_timing(matcher):
    # Lyrics "B" is missing from words/audio
    lyrics = [
        LyricLine(text="A", line_number=1),
        LyricLine(text="B", line_number=2),
        LyricLine(text="C", line_number=3),
    ]
    words = [
        WordTimestamp(word="A", start=0.0, end=1.0),
        WordTimestamp(word="C", start=4.0, end=5.0),
    ]

    results = matcher.match_lyrics_to_words(lyrics, words, language="en")

    assert len(results) == 3
    # A matched
    assert results[0].start_time == 0.0
    assert results[0].end_time == 1.0

    # B interpolated
    # Gap between A.end (1.0) and C.start (4.0) is 3.0s
    # B should fill this gap
    assert results[1].text == "B"
    assert results[1].start_time == 1.0
    assert results[1].end_time == 4.0
    assert results[1].confidence == 0.0

    # C matched
    assert results[2].start_time == 4.0
    assert results[2].end_time == 5.0


def test_fallback_uniform_distribution(matcher):
    lyrics = [LyricLine(text="One", line_number=1), LyricLine(text="Two", line_number=2)]
    words = []  # No words found

    # Should fallback to uniform distribution over default 0-180s (if total_duration defaults to 180 when words is empty)
    results = matcher.match_lyrics_to_words(lyrics, words, language="en")

    assert len(results) == 2
    # 180 / 2 = 90s per line
    assert results[0].start_time == 0.0
    assert results[0].end_time == 90.0
    assert results[1].start_time == 90.0
    assert results[1].end_time == 180.0
