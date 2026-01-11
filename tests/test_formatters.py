"""Tests for output formatters."""

import json

import pytest

from everyric2.inference.prompt import SyncResult
from everyric2.output.formatters import (
    ASSFormatter,
    FormatterFactory,
    JSONFormatter,
    LRCFormatter,
    SRTFormatter,
)


@pytest.fixture
def sample_results() -> list[SyncResult]:
    """Create sample sync results for testing."""
    return [
        SyncResult(text="First line of lyrics", start_time=5.23, end_time=8.45),
        SyncResult(text="Second line here", start_time=8.90, end_time=12.10),
        SyncResult(text="Third and final line", start_time=12.50, end_time=16.00),
    ]


class TestSRTFormatter:
    """Tests for SRT formatter."""

    def test_format_basic(self, sample_results: list[SyncResult]):
        """Test basic SRT formatting."""
        formatter = SRTFormatter()
        output = formatter.format(sample_results)

        assert "1\n" in output
        assert "00:00:05,230 -->" in output  # Start time
        assert "00:00:08,4" in output  # End time (allow for float rounding)
        assert "First line of lyrics" in output
        assert "2\n" in output
        assert "Second line here" in output

    def test_format_timestamp(self):
        """Test timestamp formatting."""
        formatter = SRTFormatter()

        # Test various timestamps
        assert formatter._format_time(0) == "00:00:00,000"
        assert formatter._format_time(5.23) == "00:00:05,230"
        assert formatter._format_time(65.5) == "00:01:05,500"
        assert formatter._format_time(3661.123) == "01:01:01,123"

    def test_extension(self):
        """Test file extension."""
        formatter = SRTFormatter()
        assert formatter.get_extension() == "srt"


class TestASSFormatter:
    """Tests for ASS formatter."""

    def test_format_has_header(self, sample_results: list[SyncResult]):
        """Test ASS output has proper header."""
        formatter = ASSFormatter()
        output = formatter.format(sample_results)

        assert "[Script Info]" in output
        assert "[V4+ Styles]" in output
        assert "[Events]" in output
        assert "Dialogue:" in output

    def test_format_with_metadata(self, sample_results: list[SyncResult]):
        """Test ASS with metadata."""
        formatter = ASSFormatter()
        output = formatter.format(sample_results, metadata={"title": "Test Song"})

        assert "Title: Test Song" in output

    def test_extension(self):
        """Test file extension."""
        formatter = ASSFormatter()
        assert formatter.get_extension() == "ass"


class TestLRCFormatter:
    """Tests for LRC formatter."""

    def test_format_basic(self, sample_results: list[SyncResult]):
        """Test basic LRC formatting."""
        formatter = LRCFormatter()
        output = formatter.format(sample_results)

        assert "[00:05.23]First line of lyrics" in output
        assert "[00:08.90]Second line here" in output

    def test_format_with_metadata(self, sample_results: list[SyncResult]):
        """Test LRC with metadata tags."""
        formatter = LRCFormatter()
        output = formatter.format(
            sample_results, metadata={"title": "Test Song", "artist": "Test Artist"}
        )

        assert "[ti:Test Song]" in output
        assert "[ar:Test Artist]" in output

    def test_extension(self):
        """Test file extension."""
        formatter = LRCFormatter()
        assert formatter.get_extension() == "lrc"


class TestJSONFormatter:
    """Tests for JSON formatter."""

    def test_format_valid_json(self, sample_results: list[SyncResult]):
        """Test output is valid JSON."""
        formatter = JSONFormatter()
        output = formatter.format(sample_results)

        # Should be valid JSON
        data = json.loads(output)
        assert "lyrics" in data
        assert "metadata" in data
        assert len(data["lyrics"]) == 3

    def test_format_structure(self, sample_results: list[SyncResult]):
        """Test JSON structure."""
        formatter = JSONFormatter()
        output = formatter.format(sample_results)
        data = json.loads(output)

        first = data["lyrics"][0]
        assert first["text"] == "First line of lyrics"
        assert first["start"] == 5.23
        assert first["end"] == 8.45

    def test_extension(self):
        """Test file extension."""
        formatter = JSONFormatter()
        assert formatter.get_extension() == "json"


class TestFormatterFactory:
    """Tests for formatter factory."""

    def test_get_supported_formats(self):
        """Test getting supported formats."""
        formats = FormatterFactory.get_supported_formats()
        assert "srt" in formats
        assert "ass" in formats
        assert "lrc" in formats
        assert "json" in formats

    def test_get_formatter_srt(self):
        """Test getting SRT formatter."""
        formatter = FormatterFactory.get_formatter("srt")
        assert isinstance(formatter, SRTFormatter)

    def test_get_formatter_case_insensitive(self):
        """Test format names are case insensitive."""
        formatter1 = FormatterFactory.get_formatter("SRT")
        formatter2 = FormatterFactory.get_formatter("srt")
        assert type(formatter1) == type(formatter2)

    def test_get_formatter_invalid(self):
        """Test getting invalid formatter raises error."""
        with pytest.raises(ValueError) as exc_info:
            FormatterFactory.get_formatter("invalid")
        assert "Unsupported format" in str(exc_info.value)
