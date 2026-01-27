"""Tests for prompt builder and response parsing."""

import pytest

from everyric2.inference.prompt import LyricLine, PromptBuilder, SyncResult


class TestLyricLine:
    """Tests for LyricLine class."""

    def test_from_text_basic(self):
        """Test parsing lyrics from text."""
        text = """First line
Second line
Third line"""
        lines = LyricLine.from_text(text)

        assert len(lines) == 3
        assert lines[0].text == "First line"
        assert lines[0].line_number == 1
        assert lines[2].text == "Third line"
        assert lines[2].line_number == 3

    def test_from_text_empty_lines(self):
        """Test parsing with empty lines."""
        text = """First line

Second line

Third line"""
        lines = LyricLine.from_text(text)

        # Empty lines should be skipped
        assert len(lines) == 3
        # But line numbers should reflect original positions
        assert lines[0].line_number == 1
        assert lines[1].line_number == 3
        assert lines[2].line_number == 5

    def test_from_text_whitespace(self):
        """Test parsing with extra whitespace."""
        text = """  First line  
   Second line
Third line   """
        lines = LyricLine.from_text(text)

        # Whitespace should be stripped
        assert lines[0].text == "First line"
        assert lines[1].text == "Second line"


class TestSyncResult:
    """Tests for SyncResult class."""

    def test_to_dict(self):
        """Test converting to dictionary."""
        result = SyncResult(
            text="Test line",
            start_time=1.5,
            end_time=3.5,
            confidence=0.95,
            line_number=1,
        )
        d = result.to_dict()

        assert d["text"] == "Test line"
        assert d["start"] == 1.5
        assert d["end"] == 3.5
        assert d["confidence"] == 0.95
        assert d["line_number"] == 1

    def test_to_dict_minimal(self):
        """Test converting minimal result to dictionary."""
        result = SyncResult(text="Test", start_time=0, end_time=1)
        d = result.to_dict()

        assert d["text"] == "Test"
        assert d["confidence"] is None
        assert d["line_number"] is None


class TestPromptBuilder:
    """Tests for PromptBuilder class."""

    @pytest.fixture
    def builder(self) -> PromptBuilder:
        """Create a prompt builder."""
        return PromptBuilder()

    @pytest.fixture
    def sample_lyrics(self) -> list[LyricLine]:
        """Create sample lyrics."""
        return [
            LyricLine(text="First line", line_number=1),
            LyricLine(text="Second line", line_number=2),
            LyricLine(text="Third line", line_number=3),
        ]

    def test_build_lyrics_text(self, builder: PromptBuilder, sample_lyrics: list[LyricLine]):
        """Test building lyrics text."""
        text = builder.build_lyrics_text(sample_lyrics)

        assert "1. First line" in text
        assert "2. Second line" in text
        assert "3. Third line" in text

    def test_build_conversation(self, builder: PromptBuilder, sample_lyrics: list[LyricLine]):
        """Test building conversation structure."""
        conv = builder.build_conversation("/tmp/test.wav", sample_lyrics)

        assert len(conv) == 2  # system + user
        assert conv[0]["role"] == "system"
        assert conv[1]["role"] == "user"

        # User message should have audio and text
        user_content = conv[1]["content"]
        types = [c["type"] for c in user_content]
        assert "audio" in types
        assert "text" in types

    def test_parse_response_json(self, builder: PromptBuilder):
        """Test parsing JSON response."""
        response = """[
            {"text": "First line", "start": 1.0, "end": 2.5},
            {"text": "Second line", "start": 2.8, "end": 4.0}
        ]"""
        results = builder.parse_response(response)

        assert len(results) == 2
        assert results[0].text == "First line"
        assert results[0].start_time == 1.0
        assert results[0].end_time == 2.5

    def test_parse_response_with_markdown(self, builder: PromptBuilder):
        """Test parsing JSON wrapped in markdown code block."""
        response = """```json
[
    {"text": "First line", "start": 1.0, "end": 2.5}
]
```"""
        results = builder.parse_response(response)

        assert len(results) == 1
        assert results[0].text == "First line"

    def test_parse_response_with_extra_text(self, builder: PromptBuilder):
        """Test parsing JSON with surrounding text."""
        response = """Here are the synchronized lyrics:
[{"text": "Test line", "start": 0.5, "end": 1.5}]
Done!"""
        results = builder.parse_response(response)

        assert len(results) == 1
        assert results[0].text == "Test line"

    def test_parse_response_lrc_format(self, builder: PromptBuilder):
        """Test parsing LRC-style response."""
        response = """[00:05]First line
[00:10]Second line
[01:00]Third line"""
        results = builder.parse_response(response)

        assert len(results) == 3
        assert results[0].text == "First line"
        assert results[0].start_time == 5.0
        assert results[2].start_time == 60.0

    def test_parse_response_invalid(self, builder: PromptBuilder):
        """Test parsing invalid response raises error."""
        response = "This is not a valid response format at all."
        with pytest.raises(ValueError):
            builder.parse_response(response)
