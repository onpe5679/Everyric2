import pytest
from unittest.mock import MagicMock, patch, call
from pathlib import Path
import subprocess

from everyric2.alignment.mfa_engine import MFAEngine
from everyric2.alignment.base import AlignmentError, EngineNotAvailableError
from everyric2.inference.prompt import LyricLine, SyncResult
from everyric2.audio.loader import AudioData


@pytest.fixture
def mock_audio_loader():
    with patch("everyric2.alignment.mfa_engine.AudioLoader") as mock:
        loader_instance = mock.return_value
        # return same audio object for prepare_for_alignment
        loader_instance.prepare_for_alignment.return_value = MagicMock()
        yield loader_instance


@pytest.fixture
def mfa_engine(mock_audio_loader):
    return MFAEngine()


@pytest.fixture
def sample_audio():
    audio = MagicMock(spec=AudioData)
    audio.duration = 10.0
    return audio


@pytest.fixture
def sample_lyrics():
    return [
        LyricLine(text="hello", line_number=1),
        LyricLine(text="world", line_number=2),
    ]


def test_is_available_true(mfa_engine):
    with patch("shutil.which", return_value="/usr/bin/mfa"):
        assert mfa_engine.is_available() is True


def test_is_available_false(mfa_engine):
    with patch("shutil.which", return_value=None):
        assert mfa_engine.is_available() is False


def test_check_models_installed_unsupported(mfa_engine):
    is_installed, missing = mfa_engine._check_models_installed("fr")
    assert is_installed is False
    assert "Language 'fr' not supported" in missing[0]


def test_check_models_installed_success(mfa_engine):
    # Mock subprocess.run to return installed models
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = [
            # First call for acoustic models
            MagicMock(stdout="english_mfa\nother_model"),
            # Second call for dictionary models
            MagicMock(stdout="english_us_arpa\nother_dict"),
        ]

        is_installed, missing = mfa_engine._check_models_installed("en")
        assert is_installed is True
        assert len(missing) == 0


def test_check_models_installed_missing(mfa_engine):
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = [
            # Acoustic missing
            MagicMock(stdout="other_model"),
            # Dictionary missing
            MagicMock(stdout="other_dict"),
        ]

        is_installed, missing = mfa_engine._check_models_installed("en")
        assert is_installed is False
        assert any("acoustic model" in m for m in missing)
        assert any("dictionary" in m for m in missing)


def test_check_models_installed_error(mfa_engine):
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(["cmd"], 30)):
        is_installed, missing = mfa_engine._check_models_installed("en")
        assert is_installed is False
        assert missing == ["MFA command failed"]


def test_align_creates_corpus(mfa_engine, sample_audio, sample_lyrics):
    # Setup
    with (
        patch.object(MFAEngine, "is_available", return_value=True),
        patch.object(MFAEngine, "_check_models_installed", return_value=(True, [])),
        patch("subprocess.run") as mock_run,
        patch("tempfile.TemporaryDirectory") as mock_temp_dir,
        patch("everyric2.alignment.mfa_engine.Path") as MockPath,
        patch.object(MFAEngine, "_parse_textgrid", return_value=[]),
    ):
        # Mock temp dir context
        mock_temp_dir.return_value.__enter__.return_value = "/tmp/mock"

        # Mock Path behavior
        mock_root = MockPath.return_value
        corpus_dir = mock_root / "corpus"
        output_dir = mock_root / "output"

        # When Path("/tmp/mock") is called, return our mock root
        def path_side_effect(arg):
            if arg == "/tmp/mock":
                return mock_root
            return MagicMock(spec=Path)

        MockPath.side_effect = path_side_effect

        # Mock file operations
        mock_audio_path = corpus_dir / "audio.wav"
        mock_lab_path = corpus_dir / "audio.lab"
        mock_textgrid_path = output_dir / "audio.TextGrid"
        mock_textgrid_path.exists.return_value = True

        # Call align
        mfa_engine.align(sample_audio, sample_lyrics, language="en")

        # Verify corpus creation
        corpus_dir.mkdir.assert_called()
        output_dir.mkdir.assert_called()

        # Verify audio preparation and saving
        mfa_engine.audio_loader.prepare_for_alignment.assert_called_with(
            sample_audio, target_sr=16000, normalize=True
        )
        # Verify to_file called on the prepared audio
        mfa_engine.audio_loader.prepare_for_alignment.return_value.to_file.assert_called_with(
            mock_audio_path
        )

        # Verify transcript writing
        mock_lab_path.write_text.assert_called_with("hello\nworld", encoding="utf-8")

        # Verify MFA command execution
        assert mock_run.call_count == 1
        cmd_args = mock_run.call_args[0][0]
        assert cmd_args[0] == "mfa"
        assert cmd_args[1] == "align"
        assert str(corpus_dir) in [str(a) for a in cmd_args]
        assert str(output_dir) in [str(a) for a in cmd_args]


def test_parse_textgrid(mfa_engine, sample_lyrics):
    # Mock praatio
    with patch.dict("sys.modules", {"praatio": MagicMock(), "praatio.textgrid": MagicMock()}):
        from praatio import textgrid

        # Setup mock TextGrid
        mock_tg = MagicMock()
        mock_tier = MagicMock()
        mock_entry1 = MagicMock()
        mock_entry1.label = "hello"
        mock_entry1.start = 0.0
        mock_entry1.end = 0.5
        mock_entry2 = MagicMock()
        mock_entry2.label = "world"
        mock_entry2.start = 0.6
        mock_entry2.end = 1.0

        mock_tier.entries = [mock_entry1, mock_entry2]
        mock_tg.tierNames = ["words"]
        mock_tg.getTier.return_value = mock_tier
        textgrid.openTextgrid.return_value = mock_tg

        # Mock LyricsMatcher
        with patch("everyric2.alignment.matcher.LyricsMatcher") as MockMatcher:
            mock_matcher_instance = MockMatcher.return_value
            expected_results = [
                SyncResult(text="hello", start_time=0.0, end_time=0.5, confidence=1.0),
                SyncResult(text="world", start_time=0.6, end_time=1.0, confidence=1.0),
            ]
            mock_matcher_instance.match_lyrics_to_words.return_value = expected_results

            # Call _parse_textgrid
            results = mfa_engine._parse_textgrid(Path("dummy.TextGrid"), sample_lyrics)

            assert results == expected_results
            mock_matcher_instance.match_lyrics_to_words.assert_called()


def test_align_not_available(mfa_engine, sample_audio, sample_lyrics):
    with patch.object(MFAEngine, "is_available", return_value=False):
        with pytest.raises(EngineNotAvailableError, match="MFA not installed"):
            mfa_engine.align(sample_audio, sample_lyrics)


def test_align_models_not_installed(mfa_engine, sample_audio, sample_lyrics):
    with (
        patch.object(MFAEngine, "is_available", return_value=True),
        patch.object(MFAEngine, "_check_models_installed", return_value=(False, ["missing"])),
    ):
        with pytest.raises(AlignmentError, match="MFA models not installed"):
            mfa_engine.align(sample_audio, sample_lyrics, language="en")


def test_align_subprocess_error(mfa_engine, sample_audio, sample_lyrics):
    with (
        patch.object(MFAEngine, "is_available", return_value=True),
        patch.object(MFAEngine, "_check_models_installed", return_value=(True, [])),
        patch("tempfile.TemporaryDirectory"),
        patch("everyric2.alignment.mfa_engine.Path"),
        patch(
            "subprocess.run", side_effect=subprocess.CalledProcessError(1, "mfa", stderr="Error!")
        ),
    ):
        with pytest.raises(AlignmentError, match="MFA alignment failed: Error!"):
            mfa_engine.align(sample_audio, sample_lyrics, language="en")


def test_align_timeout_error(mfa_engine, sample_audio, sample_lyrics):
    with (
        patch.object(MFAEngine, "is_available", return_value=True),
        patch.object(MFAEngine, "_check_models_installed", return_value=(True, [])),
        patch("tempfile.TemporaryDirectory"),
        patch("everyric2.alignment.mfa_engine.Path"),
        patch("subprocess.run", side_effect=subprocess.TimeoutExpired("mfa", 600)),
    ):
        with pytest.raises(AlignmentError, match="MFA alignment timed out"):
            mfa_engine.align(sample_audio, sample_lyrics, language="en")


def test_align_textgrid_missing(mfa_engine, sample_audio, sample_lyrics):
    with (
        patch.object(MFAEngine, "is_available", return_value=True),
        patch.object(MFAEngine, "_check_models_installed", return_value=(True, [])),
        patch("subprocess.run"),
        patch("tempfile.TemporaryDirectory") as mock_temp_dir,
        patch("everyric2.alignment.mfa_engine.Path") as MockPath,
    ):
        # Mock temp dir context
        mock_temp_dir.return_value.__enter__.return_value = "/tmp/mock"

        # Mock Path behavior
        mock_root = MockPath.return_value
        output_dir = mock_root / "output"

        # When Path("/tmp/mock") is called, return our mock root
        def path_side_effect(arg):
            if arg == "/tmp/mock":
                return mock_root
            return MagicMock(spec=Path)

        MockPath.side_effect = path_side_effect

        # Mock TextGrid existence (never found)
        # We need to ensure that output_dir / "audio.TextGrid" .exists() returns False
        # And output_dir.rglob returns empty

        (output_dir / "audio.TextGrid").exists.return_value = False
        (output_dir / "corpus" / "audio.TextGrid").exists.return_value = False
        output_dir.rglob.return_value = []

        with pytest.raises(AlignmentError, match="MFA output TextGrid not found"):
            mfa_engine.align(sample_audio, sample_lyrics, language="en")
