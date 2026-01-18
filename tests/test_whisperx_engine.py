import sys
from unittest.mock import ANY, MagicMock, patch

import numpy as np
import pytest

from everyric2.alignment.base import EngineNotAvailableError
from everyric2.alignment.whisperx_engine import WhisperXEngine
from everyric2.audio.loader import AudioData
from everyric2.config.settings import AlignmentSettings
from everyric2.inference.prompt import LyricLine

# Fixtures


@pytest.fixture
def mock_whisperx():
    mock = MagicMock()
    # Setup default behaviors for common whisperx functions
    mock.load_model.return_value = MagicMock()
    mock.load_align_model.return_value = (MagicMock(), MagicMock())
    return mock


@pytest.fixture
def mock_torch():
    mock = MagicMock()
    mock.cuda.is_available.return_value = False
    return mock


@pytest.fixture
def engine_config():
    config = AlignmentSettings()
    # Set explicit defaults for testing
    config.whisperx_model = "base"
    config.whisperx_batch_size = 16
    config.whisperx_compute_type = "int8"
    return config


@pytest.fixture
def engine(engine_config):
    with patch("everyric2.alignment.whisperx_engine.AudioLoader"):
        return WhisperXEngine(engine_config)


@pytest.fixture
def mock_audio_data():
    return AudioData(
        waveform=np.zeros((2, 16000), dtype=np.float32), sample_rate=16000, duration=1.0
    )


# Tests


def test_is_available_true(engine, mock_whisperx):
    """Test that is_available returns True when whisperx can be imported."""
    with patch.dict(sys.modules, {"whisperx": mock_whisperx}):
        assert engine.is_available() is True


def test_is_available_false(engine):
    """Test that is_available returns False when whisperx cannot be imported."""
    # We use a trick to force ImportError by patching builtins.__import__
    # ensuring we only fail for 'whisperx'
    original_import = __import__

    def import_mock(name, *args, **kwargs):
        if name == "whisperx":
            raise ImportError("No module named 'whisperx'")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=import_mock):
        # We also need to ensure it's not in sys.modules, or import won't even be called
        with patch.dict(sys.modules):
            if "whisperx" in sys.modules:
                del sys.modules["whisperx"]
            assert engine.is_available() is False


def test_transcribe_basic(engine, mock_whisperx, mock_torch, mock_audio_data):
    """Test the basic transcription flow."""
    with patch.dict(sys.modules, {"whisperx": mock_whisperx, "torch": mock_torch}):
        # Mock internal model objects
        mock_model = MagicMock()
        mock_whisperx.load_model.return_value = mock_model

        mock_align_model = MagicMock()
        mock_align_metadata = MagicMock()
        mock_whisperx.load_align_model.return_value = (mock_align_model, mock_align_metadata)

        # Mock transcription return value
        mock_model.transcribe.return_value = {
            "segments": [{"text": "Hello world", "start": 0.0, "end": 1.0}],
            "language": "en",
        }

        # Mock alignment return value
        mock_whisperx.align.return_value = {
            "segments": [
                {
                    "text": "Hello world",
                    "start": 0.0,
                    "end": 1.0,
                    "words": [
                        {"word": "Hello", "start": 0.0, "end": 0.5, "score": 0.9},
                        {"word": "world", "start": 0.5, "end": 1.0, "score": 0.8},
                    ],
                }
            ]
        }

        # Configure the engine's audio loader mock
        mock_prepared = MagicMock()
        mock_prepared.waveform = np.zeros((1, 16000))
        engine.audio_loader.prepare_for_alignment.return_value = mock_prepared

        # Run transcribe
        result = engine.transcribe(mock_audio_data)

        # Assertions
        assert result.text == "Hello world"
        assert result.language == "en"
        assert len(result.words) == 2
        assert result.words[0].word == "Hello"
        assert result.words[0].start == 0.0
        assert result.words[0].end == 0.5
        assert result.words[1].word == "world"

        # Verify calls
        mock_whisperx.load_model.assert_called_once()
        mock_model.transcribe.assert_called_once_with(
            mock_prepared.waveform, batch_size=engine.config.whisperx_batch_size, language=None
        )
        mock_whisperx.load_align_model.assert_called_once()
        mock_whisperx.align.assert_called_once()


def test_transcribe_raises_error_if_not_available(engine):
    """Test that transcribe raises EngineNotAvailableError if whisperx is missing."""
    with patch.object(engine, "is_available", return_value=False):
        with pytest.raises(EngineNotAvailableError):
            engine.transcribe(AudioData(np.zeros(1), 16000, 1.0))


def test_language_detection(engine, mock_whisperx, mock_torch, mock_audio_data):
    """Test that language detection works and loads appropriate alignment model."""
    with patch.dict(sys.modules, {"whisperx": mock_whisperx, "torch": mock_torch}):
        mock_model = MagicMock()
        mock_whisperx.load_model.return_value = mock_model

        # Mock transcription to return Japanese
        mock_model.transcribe.return_value = {"segments": [], "language": "ja"}

        mock_whisperx.load_align_model.return_value = (MagicMock(), MagicMock())
        mock_whisperx.align.return_value = {"segments": []}

        engine.audio_loader.prepare_for_alignment.return_value.waveform = np.zeros(1)

        # Run with auto detection
        result = engine.transcribe(mock_audio_data, language="auto")

        assert result.language == "ja"
        # Verify it attempted to load the Japanese alignment model
        mock_whisperx.load_align_model.assert_called_with(language_code="ja", device=ANY)


def test_align_with_lyrics(engine, mock_audio_data):
    """Test the full align method including lyrics matching."""
    # We mock transcribe to avoid setting up all the whisperx mocks again
    mock_transcribe_result = MagicMock()
    mock_transcribe_result.words = []
    mock_transcribe_result.language = "en"

    engine.transcribe = MagicMock(return_value=mock_transcribe_result)

    # Fix: LyricLine takes text and line_number (required)
    lyrics = [LyricLine(text="Hello", line_number=1)]

    # Patch the LyricsMatcher where it is defined, so the import inside the method picks it up
    with patch("everyric2.alignment.matcher.LyricsMatcher") as MockMatcher:
        mock_matcher = MockMatcher.return_value
        expected_results = [MagicMock()]
        mock_matcher.match_lyrics_to_words.return_value = expected_results

        callback = MagicMock()
        results = engine.align(mock_audio_data, lyrics, progress_callback=callback)

        assert results == expected_results
        engine.transcribe.assert_called_once_with(mock_audio_data, None)
        mock_matcher.match_lyrics_to_words.assert_called_once_with(
            lyrics, mock_transcribe_result.words, mock_transcribe_result.language
        )
        assert callback.call_count == 3


def test_unload(engine, mock_whisperx, mock_torch):
    """Test that unload cleans up resources."""
    with patch.dict(sys.modules, {"whisperx": mock_whisperx, "torch": mock_torch}):
        # Manually set internal state
        engine._model = MagicMock()
        engine._align_model = MagicMock()
        engine._align_metadata = MagicMock()
        engine._current_language = "en"

        # Fix: ensure cuda.is_available returns True to trigger empty_cache call
        mock_torch.cuda.is_available.return_value = True

        with patch("gc.collect") as mock_gc:
            engine.unload()

            assert engine._model is None
            assert engine._align_model is None
            assert engine._align_metadata is None
            assert engine._current_language is None

            mock_gc.assert_called()
            mock_torch.cuda.empty_cache.assert_called()
