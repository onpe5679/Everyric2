"""Vocal separation using Demucs."""

import logging
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from everyric2.audio.loader import AudioData, AudioLoader
from everyric2.config.settings import AudioSettings, get_settings

logger = logging.getLogger(__name__)

# 웜 캐시 싱글턴 (WS2-A) — 프로세스 수명 동안 상주. 지연 생성이라 import만으로는 아무것도
# 로드하지 않는다 (API 전용 모드에서 절대 만들어지지 않게 하는 유일한 근거).
_shared_separator: "VocalSeparator | None" = None
_shared_separator_lock = threading.Lock()


def get_shared_separator(config: "AudioSettings | None" = None) -> "VocalSeparator":
    """웜 캐시된 VocalSeparator를 돌려준다 (EVERYRIC_SERVER_WARM_MODELS 기준).

    warm이 켜져 있으면 프로세스 수명 싱글턴을 재사용하고(두 번째 잡부터 재생성 0회), 재사용
    시 "warm model reuse: demucs" 1줄을 남긴다. 꺼져 있으면 매번 새 인스턴스(기존 동작).
    demucs는 서브프로세스로 도는 구조라 인스턴스 재사용이 인프로세스 모델 재로드를 없애지는
    않지만, 스펙의 싱글턴 규약을 동일하게 따른다."""
    if not get_settings().server.warm_models:
        return VocalSeparator(config)
    global _shared_separator
    with _shared_separator_lock:
        if _shared_separator is None:
            _shared_separator = VocalSeparator(config)
        else:
            logger.info("warm model reuse: demucs")
        return _shared_separator


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

            # Run Demucs — Windows 기본 콘솔 인코딩(cp949)로는 demucs의 유니코드
            # 진행 표시를 못 읽어 reader thread가 죽으므로 utf-8을 명시한다
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
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
