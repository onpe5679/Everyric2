"""Audio loading and preprocessing."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from everyric2.config.settings import AudioSettings, get_settings


class AudioError(Exception):
    """Base exception for audio operations."""

    pass


class UnsupportedFormatError(AudioError):
    """Raised when audio format is not supported."""

    pass


class CorruptedAudioError(AudioError):
    """Raised when audio file is corrupted."""

    pass


@dataclass
class AudioData:
    """Container for loaded audio data."""

    waveform: np.ndarray  # shape: (samples,) - mono
    sample_rate: int
    duration: float  # seconds
    source_path: Path | None = None

    @property
    def num_samples(self) -> int:
        """Get number of samples."""
        return len(self.waveform)

    def to_file(self, path: Path) -> Path:
        """Save audio to file."""
        sf.write(path, self.waveform, self.sample_rate)
        return path


@dataclass
class AudioChunk:
    """A chunk of audio with offset information."""

    audio: AudioData
    start_time: float  # seconds from original audio start
    end_time: float  # seconds from original audio start
    chunk_index: int
    total_chunks: int


class AudioLoader:
    """Load and preprocess audio files."""

    SUPPORTED_FORMATS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus", ".wma", ".aac"}

    def __init__(self, config: AudioSettings | None = None):
        """Initialize audio loader.

        Args:
            config: Audio settings. If None, uses global settings.
        """
        self.config = config or get_settings().audio

    def load(self, path: Path | str) -> AudioData:
        """Load audio file and convert to model input format.

        Args:
            path: Path to audio file.

        Returns:
            AudioData with waveform at target sample rate, mono.

        Raises:
            FileNotFoundError: If file doesn't exist.
            UnsupportedFormatError: If format is not supported.
            CorruptedAudioError: If file is corrupted.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise UnsupportedFormatError(
                f"Unsupported format: {path.suffix}. Supported: {', '.join(self.SUPPORTED_FORMATS)}"
            )

        try:
            # Load with librosa - automatically resamples and converts to mono
            waveform, sr = librosa.load(path, sr=self.config.target_sample_rate, mono=True)

            duration = len(waveform) / sr

            return AudioData(
                waveform=waveform,
                sample_rate=sr,
                duration=duration,
                source_path=path,
            )

        except Exception as e:
            raise CorruptedAudioError(f"Failed to load audio: {e}") from e

    def load_from_array(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        resample: bool = True,
    ) -> AudioData:
        """Load audio from numpy array.

        Args:
            waveform: Audio waveform array.
            sample_rate: Sample rate of the waveform.
            resample: Whether to resample to target sample rate.

        Returns:
            AudioData instance.
        """
        # Convert to mono if stereo
        if waveform.ndim == 2:
            waveform = librosa.to_mono(waveform)

        # Resample if needed
        if resample and sample_rate != self.config.target_sample_rate:
            waveform = librosa.resample(
                waveform,
                orig_sr=sample_rate,
                target_sr=self.config.target_sample_rate,
            )
            sample_rate = self.config.target_sample_rate

        duration = len(waveform) / sample_rate

        return AudioData(
            waveform=waveform,
            sample_rate=sample_rate,
            duration=duration,
            source_path=None,
        )

    def get_duration(self, path: Path | str) -> float:
        """Get audio duration without loading full file.

        Args:
            path: Path to audio file.

        Returns:
            Duration in seconds.
        """
        return librosa.get_duration(path=path)

    def chunk_audio(
        self,
        audio: AudioData,
        chunk_duration: int | None = None,
        overlap: int | None = None,
    ) -> Iterator[AudioChunk]:
        """Split audio into overlapping chunks.

        Args:
            audio: Audio data to chunk.
            chunk_duration: Duration of each chunk in seconds. Defaults to config.
            overlap: Overlap between chunks in seconds. Defaults to config.

        Yields:
            AudioChunk instances with offset information.
        """
        chunk_duration = chunk_duration or get_settings().model.chunk_duration
        overlap = overlap or get_settings().model.chunk_overlap

        if audio.duration <= chunk_duration:
            # No chunking needed
            yield AudioChunk(
                audio=audio,
                start_time=0.0,
                end_time=audio.duration,
                chunk_index=0,
                total_chunks=1,
            )
            return

        # Calculate chunk parameters
        chunk_samples = int(chunk_duration * audio.sample_rate)
        overlap_samples = int(overlap * audio.sample_rate)
        step_samples = chunk_samples - overlap_samples

        # Calculate total chunks
        total_samples = len(audio.waveform)
        total_chunks = max(1, int(np.ceil((total_samples - overlap_samples) / step_samples)))

        for i in range(total_chunks):
            start_sample = i * step_samples
            end_sample = min(start_sample + chunk_samples, total_samples)

            chunk_waveform = audio.waveform[start_sample:end_sample]
            start_time = start_sample / audio.sample_rate
            end_time = end_sample / audio.sample_rate

            yield AudioChunk(
                audio=AudioData(
                    waveform=chunk_waveform,
                    sample_rate=audio.sample_rate,
                    duration=end_time - start_time,
                    source_path=audio.source_path,
                ),
                start_time=start_time,
                end_time=end_time,
                chunk_index=i,
                total_chunks=total_chunks,
            )

    def save_temp(self, audio: AudioData, filename: str = "temp_audio.wav") -> Path:
        """Save audio to temporary directory.

        Args:
            audio: Audio data to save.
            filename: Output filename.

        Returns:
            Path to saved file.
        """
        self.config.temp_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.config.temp_dir / filename
        sf.write(output_path, audio.waveform, audio.sample_rate)
        return output_path

    def resample_for_alignment(
        self,
        audio: AudioData,
        target_sr: int = 16000,
    ) -> AudioData:
        if audio.sample_rate == target_sr:
            return audio

        resampled = librosa.resample(
            audio.waveform,
            orig_sr=audio.sample_rate,
            target_sr=target_sr,
        )
        return AudioData(
            waveform=resampled,
            sample_rate=int(target_sr),
            duration=len(resampled) / target_sr,
            source_path=audio.source_path,
        )

    def normalize_audio(
        self,
        audio: AudioData,
        target_db: float = -20.0,
    ) -> AudioData:
        rms = np.sqrt(np.mean(audio.waveform**2))
        if rms < 1e-10:
            return audio

        current_db = 20 * np.log10(rms + 1e-10)
        gain = 10 ** ((target_db - current_db) / 20)

        normalized = audio.waveform * gain
        normalized = np.clip(normalized, -1.0, 1.0)

        return AudioData(
            waveform=normalized,
            sample_rate=audio.sample_rate,
            duration=audio.duration,
            source_path=audio.source_path,
        )

    def prepare_for_alignment(
        self,
        audio: AudioData,
        target_sr: int = 16000,
        normalize: bool = True,
    ) -> AudioData:
        prepared = self.resample_for_alignment(audio, target_sr)
        if normalize:
            prepared = self.normalize_audio(prepared)
        return prepared
