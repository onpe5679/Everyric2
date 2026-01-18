import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from everyric2.alignment.base import EngineNotAvailableError
from everyric2.alignment.faster_whisper_engine import FasterWhisperEngine
from everyric2.audio.loader import AudioData
from everyric2.config.settings import AlignmentSettings
from everyric2.inference.prompt import LyricLine


@pytest.fixture
def mock_faster_whisper():
    mock = MagicMock()
    mock.WhisperModel = MagicMock()
    return mock


@pytest.fixture
def mock_torch():
    mock = MagicMock()
    mock.cuda.is_available.return_value = False
    return mock


@pytest.fixture
def engine_config():
    config = AlignmentSettings()
    config.whisperx_model = "base"
    config.whisperx_compute_type = "int8"
    return config


@pytest.fixture
def engine(engine_config):
    with patch("everyric2.alignment.faster_whisper_engine.AudioLoader"):
        return FasterWhisperEngine(engine_config)


@pytest.fixture
def mock_audio_data():
    return AudioData(waveform=np.zeros((16000,), dtype=np.float32), sample_rate=16000, duration=1.0)


def test_is_available_true(engine, mock_faster_whisper):
    with patch.dict(sys.modules, {"faster_whisper": mock_faster_whisper}):
        assert engine.is_available() is True


def test_is_available_false(engine):
    original_import = __import__

    def import_mock(name, *args, **kwargs):
        if name == "faster_whisper":
            raise ImportError("No module named 'faster_whisper'")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=import_mock):
        with patch.dict(sys.modules):
            if "faster_whisper" in sys.modules:
                del sys.modules["faster_whisper"]
            assert engine.is_available() is False


def test_transcribe_basic(engine, mock_faster_whisper, mock_torch, mock_audio_data):
    with patch.dict(sys.modules, {"faster_whisper": mock_faster_whisper, "torch": mock_torch}):
        mock_model_instance = MagicMock()
        mock_faster_whisper.WhisperModel.return_value = mock_model_instance

        mock_segment = MagicMock()
        mock_segment.start = 0.0
        mock_segment.end = 1.0
        mock_segment.text = "Hello world"

        word1 = MagicMock()
        word1.word = "Hello"
        word1.start = 0.0
        word1.end = 0.5
        word1.probability = 0.9

        word2 = MagicMock()
        word2.word = "world"
        word2.start = 0.5
        word2.end = 1.0
        word2.probability = 0.8

        mock_segment.words = [word1, word2]

        mock_info = MagicMock()
        mock_info.language = "en"

        mock_model_instance.transcribe.return_value = (iter([mock_segment]), mock_info)

        mock_prepared = MagicMock()
        mock_prepared.waveform = np.zeros((16000,))
        engine.audio_loader.prepare_for_alignment.return_value = mock_prepared

        result = engine.transcribe(mock_audio_data)

        assert result.text == "Hello world"
        assert result.language == "en"
        assert len(result.words) == 2
        assert result.words[0].word == "Hello"
        assert result.words[0].start == 0.0
        assert result.words[0].end == 0.5
        assert result.words[1].word == "world"

        mock_faster_whisper.WhisperModel.assert_called_once()
        mock_model_instance.transcribe.assert_called_once()

        call_args = mock_model_instance.transcribe.call_args
        assert call_args[1]["word_timestamps"] is True
        assert call_args[1]["vad_filter"] is True


def test_transcribe_raises_error_if_not_available(engine):
    with patch.object(engine, "is_available", return_value=False):
        with pytest.raises(EngineNotAvailableError):
            engine.transcribe(AudioData(np.zeros(1), 16000, 1.0))


def test_language_detection(engine, mock_faster_whisper, mock_torch, mock_audio_data):
    with patch.dict(sys.modules, {"faster_whisper": mock_faster_whisper, "torch": mock_torch}):
        mock_model_instance = MagicMock()
        mock_faster_whisper.WhisperModel.return_value = mock_model_instance

        mock_info = MagicMock()
        mock_info.language = "ja"

        mock_model_instance.transcribe.return_value = (iter([]), mock_info)

        engine.audio_loader.prepare_for_alignment.return_value.waveform = np.zeros(1)

        result = engine.transcribe(mock_audio_data, language="auto")

        assert result.language == "ja"


def test_align_with_lyrics(engine, mock_audio_data):
    mock_transcribe_result = MagicMock()
    mock_transcribe_result.words = []
    mock_transcribe_result.language = "en"

    engine.transcribe = MagicMock(return_value=mock_transcribe_result)

    lyrics = [LyricLine(text="Hello", line_number=1)]

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


def test_unload(engine, mock_faster_whisper, mock_torch):
    with patch.dict(sys.modules, {"faster_whisper": mock_faster_whisper, "torch": mock_torch}):
        engine._model = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        with patch("gc.collect") as mock_gc:
            engine.unload()

            assert engine._model is None
            mock_gc.assert_called()
            mock_torch.cuda.empty_cache.assert_called()
