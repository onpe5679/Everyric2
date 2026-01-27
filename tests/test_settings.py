"""Tests for configuration settings."""

from pathlib import Path

from everyric2.config.settings import (
    AudioSettings,
    ModelSettings,
    OutputSettings,
    ServerSettings,
    Settings,
    get_settings,
    reset_settings,
)


class TestModelSettings:
    """Tests for model settings."""

    def test_default_values(self):
        """Test default model settings."""
        settings = ModelSettings()

        assert "Qwen" in settings.path
        assert settings.device_map == "auto"
        assert settings.use_flash_attention is True
        assert settings.max_audio_duration == 2400  # 40 minutes
        assert settings.chunk_duration == 1800  # 30 minutes

    def test_cache_dir_override(self, monkeypatch):
        """Test cache directory override via environment."""
        monkeypatch.setenv("EVERYRIC_MODEL_CACHE_DIR", "/mnt/d/huggingface_cache")
        settings = ModelSettings()

        assert settings.cache_dir == Path("/mnt/d/huggingface_cache")


class TestAudioSettings:
    """Tests for audio settings."""

    def test_default_values(self):
        """Test default audio settings."""
        settings = AudioSettings()

        assert settings.target_sample_rate == 24000  # Qwen-Omni native
        assert settings.demucs_model == "htdemucs"
        assert settings.temp_dir == Path("/tmp/everyric2")

    def test_temp_dir_created(self, tmp_path):
        """Test temp directory is created on validation."""
        test_dir = tmp_path / "everyric_test"

        # Monkey-patch the default
        settings = AudioSettings(temp_dir=test_dir)

        assert test_dir.exists()


class TestOutputSettings:
    """Tests for output settings."""

    def test_default_format(self):
        """Test default output format."""
        settings = OutputSettings()
        assert settings.default_format == "srt"

    def test_format_validation(self):
        """Test format validation."""
        # Valid formats should work
        for fmt in ["srt", "ass", "lrc", "json"]:
            settings = OutputSettings(default_format=fmt)
            assert settings.default_format == fmt


class TestServerSettings:
    """Tests for server settings."""

    def test_default_values(self):
        """Test default server settings."""
        settings = ServerSettings()

        assert settings.host == "0.0.0.0"
        assert settings.port == 8000
        assert settings.reload is False
        assert settings.workers == 1


class TestSettings:
    """Tests for main settings container."""

    def test_nested_settings(self):
        """Test nested settings are properly initialized."""
        settings = Settings()

        assert isinstance(settings.model, ModelSettings)
        assert isinstance(settings.audio, AudioSettings)
        assert isinstance(settings.output, OutputSettings)
        assert isinstance(settings.server, ServerSettings)

    def test_env_prefix(self, monkeypatch):
        """Test environment variable prefix."""
        monkeypatch.setenv("EVERYRIC_DEBUG", "true")

        reset_settings()  # Clear cached settings
        settings = get_settings()

        assert settings.debug is True

    def test_nested_env_vars(self, monkeypatch):
        """Test nested environment variables with delimiter."""
        monkeypatch.setenv("EVERYRIC_SERVER__PORT", "9000")

        reset_settings()
        settings = get_settings()

        assert settings.server.port == 9000


class TestGetSettings:
    """Tests for settings singleton."""

    def test_singleton(self):
        """Test settings returns same instance."""
        reset_settings()

        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2

    def test_reset_settings(self):
        """Test resetting settings creates new instance."""
        settings1 = get_settings()
        reset_settings()
        settings2 = get_settings()

        assert settings1 is not settings2
