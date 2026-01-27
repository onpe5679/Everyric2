"""Vocal separation using Demucs."""

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from everyric2.audio.loader import AudioData, AudioLoader
from everyric2.config.settings import AudioSettings, get_settings


class SeparationError(Exception):
    """Base exception for separation operations."""

    pass


class DemucsNotAvailableError(SeparationError):
    """Raised when Demucs is not installed."""

    pass


@dataclass
class SeparationResult:
    """Result of vocal separation."""

    vocals: AudioData
    accompaniment: AudioData
    original: AudioData


class VocalSeparator:
    """Separate vocals from music using Demucs."""

    AVAILABLE_MODELS = [
        "htdemucs",
        "htdemucs_ft",
        "htdemucs_6s",
        "mdx",
        "mdx_extra",
        "mdx_extra_q",
    ]

    def __init__(self, config: AudioSettings | None = None):
        """Initialize separator.

        Args:
            config: Audio settings. If None, uses global settings.
        """
        self.config = config or get_settings().audio
        self.loader = AudioLoader(config)
        self._demucs_available: bool | None = None

    def is_available(self) -> bool:
        """Check if Demucs is available.

        Returns:
            True if Demucs is installed and working.
        """
        if self._demucs_available is not None:
            return self._demucs_available

        try:
            import demucs  # noqa: F401

            self._demucs_available = True
        except ImportError:
            self._demucs_available = False

        return self._demucs_available

    def get_available_models(self) -> list[str]:
        """Get list of available Demucs models.

        Returns:
            List of model names.
        """
        return self.AVAILABLE_MODELS.copy()

    def separate(
        self,
        audio: AudioData,
        model: str | None = None,
        use_gpu: bool = True,
    ) -> SeparationResult:
        """Separate vocals from audio.

        Args:
            audio: Audio data to process.
            model: Demucs model name. Defaults to config setting.
            use_gpu: Whether to use GPU acceleration.

        Returns:
            SeparationResult with vocals and accompaniment.

        Raises:
            DemucsNotAvailableError: If Demucs is not installed.
            SeparationError: If separation fails.
        """
        if not self.is_available():
            raise DemucsNotAvailableError(
                "Demucs is not installed. Install with: pip install demucs"
            )

        model = model or self.config.demucs_model
        temp_dir = self.config.temp_dir
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Save input audio to temp file
        input_path = temp_dir / "demucs_input.wav"
        audio.to_file(input_path)

        # Output directory for Demucs
        output_dir = temp_dir / "demucs_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Build Demucs command
            cmd = [
                sys.executable,
                "-m",
                "demucs",
                "-n",
                model,
                "--two-stems",
                "vocals",
                "-o",
                str(output_dir),
            ]

            if not use_gpu:
                cmd.extend(["-d", "cpu"])

            cmd.append(str(input_path))

            # Run Demucs
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
            )

            if result.returncode != 0:
                # Try CPU fallback if GPU failed
                if use_gpu and "CUDA" in result.stderr:
                    return self.separate(audio, model, use_gpu=False)
                raise SeparationError(f"Demucs failed: {result.stderr}")

            # Find output files
            # Demucs outputs to: output_dir/model/input_name/vocals.wav, no_vocals.wav
            model_output_dir = output_dir / model / "demucs_input"

            vocals_path = model_output_dir / "vocals.wav"
            no_vocals_path = model_output_dir / "no_vocals.wav"

            if not vocals_path.exists() or not no_vocals_path.exists():
                raise SeparationError(f"Output files not found. Expected at: {model_output_dir}")

            # Load separated audio
            vocals = self.loader.load(vocals_path)
            accompaniment = self.loader.load(no_vocals_path)

            return SeparationResult(
                vocals=vocals,
                accompaniment=accompaniment,
                original=audio,
            )

        except subprocess.TimeoutExpired:
            raise SeparationError("Demucs timed out (>10 minutes)")
        except Exception as e:
            if isinstance(e, SeparationError):
                raise
            raise SeparationError(f"Separation failed: {e}") from e
        finally:
            # Cleanup input file
            if input_path.exists():
                input_path.unlink()

    def separate_file(
        self,
        audio_path: Path | str,
        model: str | None = None,
        use_gpu: bool = True,
    ) -> SeparationResult:
        """Separate vocals from audio file.

        Args:
            audio_path: Path to audio file.
            model: Demucs model name.
            use_gpu: Whether to use GPU.

        Returns:
            SeparationResult with vocals and accompaniment.
        """
        audio = self.loader.load(audio_path)
        return self.separate(audio, model, use_gpu)
