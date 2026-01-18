"""SOFA-based forced alignment engine for singing voice.

SOFA (Singing-Oriented Forced Aligner) provides better alignment
for singing voice compared to speech-focused aligners like MFA.

Requires: lightning, einops, numba, textgrid
"""

import logging
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from everyric2.alignment.base import (
    AlignmentError,
    BaseAlignmentEngine,
    EngineNotAvailableError,
    TranscriptionResult,
    WordTimestamp,
)
from everyric2.alignment.matcher import MatchStats
from everyric2.audio.loader import AudioData
from everyric2.config.settings import AlignmentSettings
from everyric2.inference.prompt import LyricLine, SyncResult, WordSegment

logger = logging.getLogger(__name__)

# Model download URLs (English only - Japanese model URL is broken)
SOFA_MODELS = {
    "en": {
        "name": "tgm_en_v100",
        "url": "https://github.com/spicytigermeat/SOFA-Models/releases/download/v1.0.0_en/tgm_en_v100.ckpt",
        "dict_url": "https://raw.githubusercontent.com/spicytigermeat/SOFA-Models/main/tgm_sofa_dict.txt",
    },
}

CACHE_DIR = Path.home() / ".cache" / "everyric2" / "sofa"


def get_sofa_path() -> Path:
    """Get path to SOFA module, cloning if necessary."""
    sofa_path = CACHE_DIR / "SOFA"
    if not sofa_path.exists():
        logger.info("Cloning SOFA repository...")
        import subprocess

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/qiuqiao/SOFA.git", str(sofa_path)],
            check=True,
            capture_output=True,
        )
    return sofa_path


def download_model(language: str) -> tuple[Path, Path | None]:
    """Download SOFA model and dictionary for the given language."""
    import urllib.request

    if language not in SOFA_MODELS:
        raise AlignmentError(f"SOFA model not available for language: {language}")

    model_info = SOFA_MODELS[language]
    models_dir = CACHE_DIR / "models"
    dicts_dir = CACHE_DIR / "dictionaries"
    models_dir.mkdir(parents=True, exist_ok=True)
    dicts_dir.mkdir(parents=True, exist_ok=True)

    # Download model
    model_path = models_dir / f"{model_info['name']}.ckpt"
    if not model_path.exists():
        logger.info(f"Downloading SOFA model: {model_info['name']}...")
        urllib.request.urlretrieve(model_info["url"], model_path)
        logger.info(f"Model saved to: {model_path}")

    # Download dictionary
    dict_path = None
    if model_info.get("dict_url"):
        dict_path = dicts_dir / f"{model_info['name']}_dict.txt"
        if not dict_path.exists():
            logger.info(f"Downloading dictionary for {language}...")
            urllib.request.urlretrieve(model_info["dict_url"], dict_path)
            logger.info(f"Dictionary saved to: {dict_path}")

    return model_path, dict_path


class SimpleG2P:
    """Simple grapheme-to-phoneme converter using dictionary lookup."""

    def __init__(self, dict_path: Path | None = None):
        self.dictionary: dict[str, list[str]] = {}
        if dict_path and dict_path.exists():
            self._load_dictionary(dict_path)

    def _load_dictionary(self, dict_path: Path) -> None:
        """Load word-to-phoneme dictionary."""
        with open(dict_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or "\t" not in line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    word = parts[0].strip().lower()
                    phonemes = parts[1].strip().split()
                    self.dictionary[word] = phonemes

    def _normalize_word(self, word: str) -> str | None:
        """Normalize word and find it in dictionary.

        Handles contractions like "reachin'" -> "reaching".
        Returns the dictionary key if found, None otherwise.
        """
        word = word.lower().strip("'")

        # Direct lookup
        if word in self.dictionary:
            return word

        # Try adding 'g' for -in' contractions (singin' -> singing)
        if word.endswith("in"):
            expanded = word + "g"
            if expanded in self.dictionary:
                return expanded

        # Try removing trailing apostrophe variations
        for suffix in ["n't", "'s", "'d", "'ll", "'ve", "'re", "'m"]:
            if word.endswith(suffix.replace("'", "")):
                # Try base word
                base = word[: -len(suffix.replace("'", ""))]
                if base in self.dictionary:
                    return base

        return None

    def convert(self, text: str) -> tuple[list[str], list[str], list[int]]:
        """Convert text to phoneme sequence.

        Returns:
            Tuple of (ph_seq, word_seq, ph_idx_to_word_idx)
        """
        words = re.findall(r"[a-zA-Z']+", text.lower())

        ph_seq = ["SP"]  # Start with silence
        word_seq = []
        ph_idx_to_word_idx = [-1]  # SP doesn't belong to any word
        current_word_idx = 0  # Track actual index in word_seq

        for word in words:
            dict_key = self._normalize_word(word)
            if dict_key is None:
                logger.warning(f"Word '{word}' not in dictionary, skipping")
                continue

            word_seq.append(word)  # Keep original word for matching
            phonemes = self.dictionary[dict_key]

            for ph in phonemes:
                ph_seq.append(ph)
                ph_idx_to_word_idx.append(current_word_idx)

            # Add silence between words
            ph_seq.append("SP")
            ph_idx_to_word_idx.append(-1)
            current_word_idx += 1

        return ph_seq, word_seq, ph_idx_to_word_idx


class SOFAEngine(BaseAlignmentEngine):
    """SOFA-based forced alignment engine for singing voice."""

    SUPPORTED_LANGUAGES = ["en"]

    def __init__(self, config: AlignmentSettings | None = None):
        super().__init__(config)
        self._model = None
        self._g2p: SimpleG2P | None = None
        self._get_melspec = None
        self._current_lang: str | None = None
        self._device: torch.device | None = None
        self._sofa_path: Path | None = None
        self._last_word_timestamps: list[WordTimestamp] = []
        self._last_match_stats: MatchStats | None = None

    def is_available(self) -> bool:
        """Check if SOFA dependencies are available."""
        try:
            import lightning  # noqa: F401
            from einops import repeat  # noqa: F401

            return True
        except ImportError:
            return False

    def _get_device(self) -> torch.device:
        if self._device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self._device

    def _ensure_sofa_loaded(self) -> None:
        """Ensure SOFA module is in path."""
        if self._sofa_path is None:
            self._sofa_path = get_sofa_path()
            if str(self._sofa_path) not in sys.path:
                sys.path.insert(0, str(self._sofa_path))

    def _ensure_model_loaded(self, language: str) -> None:
        """Load SOFA model for the given language."""
        if self._model is not None and self._current_lang == language:
            return

        if not self.is_available():
            raise EngineNotAvailableError(
                "SOFA dependencies not installed. Install with: pip install lightning einops numba textgrid"
            )

        self._ensure_sofa_loaded()

        # Download model and dictionary
        model_path, dict_path = download_model(language)

        # Load G2P
        self._g2p = SimpleG2P(dict_path)

        # Import SOFA modules
        from modules.utils.get_melspec import MelSpecExtractor
        from train import LitForcedAlignmentTask

        # Load model
        device = self._get_device()
        logger.info(f"Loading SOFA model from: {model_path}")

        self._model = LitForcedAlignmentTask.load_from_checkpoint(str(model_path))
        self._model.to(device)
        self._model.eval()
        self._model.set_inference_mode("force")

        # Initialize melspec extractor
        melspec_config = self._model.melspec_config
        self._get_melspec = MelSpecExtractor(**melspec_config)

        self._current_lang = language
        logger.info(f"SOFA model loaded for language: {language}")

    def align(
        self,
        audio: AudioData,
        lyrics: list[LyricLine],
        language: str | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[SyncResult]:
        """Align lyrics to audio using SOFA.

        Args:
            audio: Audio data (waveform and sample rate)
            lyrics: List of lyric lines to align
            language: Language code (en, ja)
            progress_callback: Optional progress callback

        Returns:
            List of SyncResult with timestamps
        """
        from einops import repeat

        resolved_lang = self._resolve_language(language)
        if resolved_lang not in self.SUPPORTED_LANGUAGES:
            logger.warning(f"SOFA doesn't support {resolved_lang}, falling back to 'en'")
            resolved_lang = "en"

        self._ensure_model_loaded(resolved_lang)

        device = self._get_device()
        results: list[SyncResult] = []
        all_word_timestamps: list[WordTimestamp] = []

        # Combine all lyrics text
        full_text = " ".join([line.text for line in lyrics])

        # Convert to phonemes
        ph_seq, word_seq, ph_idx_to_word_idx = self._g2p.convert(full_text)

        if len(word_seq) == 0:
            logger.warning("No words found in dictionary for alignment")
            for i, line in enumerate(lyrics):
                results.append(
                    SyncResult(
                        text=line.text,
                        start_time=0.0,
                        end_time=audio.duration,
                        confidence=0.0,
                        line_number=i,
                    )
                )
            return results

        # Prepare audio
        # Resample to 44100 if needed
        target_sr = self._model.melspec_config["sample_rate"]
        waveform = audio.waveform
        if audio.sample_rate != target_sr:
            import torchaudio

            waveform = torchaudio.functional.resample(
                torch.from_numpy(waveform).float(), audio.sample_rate, target_sr
            ).numpy()

        # Convert to tensor
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=0)  # Mono
        waveform_tensor = torch.from_numpy(waveform).float().to(device)
        wav_length = len(waveform) / target_sr

        # Extract melspec
        melspec = self._get_melspec(waveform_tensor).detach().unsqueeze(0)
        melspec = (melspec - melspec.mean()) / melspec.std()
        melspec = repeat(
            melspec, "B C T -> B C (T N)", N=self._model.melspec_config["scale_factor"]
        )

        # Run alignment
        if progress_callback:
            progress_callback(1, 3)

        with torch.no_grad():
            (
                ph_seq_pred,
                ph_intervals,
                word_seq_pred,
                word_intervals,
                confidence,
                _,
                _,
            ) = self._model._infer_once(
                melspec,
                wav_length,
                ph_seq,
                word_seq,
                np.array(ph_idx_to_word_idx),
                return_ctc=False,
                return_plot=False,
            )

        if progress_callback:
            progress_callback(2, 3)

        # Convert to WordTimestamp with duration-based confidence
        # SOFA returns global confidence; use duration for per-word variation
        word_durations = [
            float(end - start) for _, (start, end) in zip(word_seq_pred, word_intervals)
        ]
        if word_durations:
            avg_duration = sum(word_durations) / len(word_durations)
            for word, (start, end) in zip(word_seq_pred, word_intervals):
                duration = float(end - start)
                # Duration ratio: 1.0 = average, <0.3 or >3.0 = suspicious
                duration_ratio = duration / avg_duration if avg_duration > 0 else 1.0
                # Convert to confidence: closer to 1.0 = higher confidence
                if duration_ratio < 1.0:
                    word_conf = float(confidence) * max(0.3, duration_ratio)
                else:
                    word_conf = float(confidence) * max(0.3, 2.0 - min(duration_ratio, 2.0))
                all_word_timestamps.append(
                    WordTimestamp(
                        word=str(word),
                        start=float(start),
                        end=float(end),
                        confidence=word_conf,
                    )
                )

        self._last_word_timestamps = all_word_timestamps

        # Match words to lyric lines
        results = self._match_words_to_lines(lyrics, all_word_timestamps, confidence)

        # Calculate match stats
        self._last_match_stats = MatchStats(
            total_lyrics=len(lyrics),
            matched_lyrics=sum(1 for r in results if r.word_segments),
            match_rate=float(confidence),
            avg_confidence=float(confidence),
        )

        if progress_callback:
            progress_callback(3, 3)

        return results

    def _match_words_to_lines(
        self,
        lyrics: list[LyricLine],
        word_timestamps: list[WordTimestamp],
        confidence: float,
    ) -> list[SyncResult]:
        """Match aligned words back to lyric lines."""
        results: list[SyncResult] = []
        word_idx = 0

        for line_idx, line in enumerate(lyrics):
            line_words = re.findall(r"[a-zA-Z']+", line.text.lower())
            line_word_segments: list[WordSegment] = []
            line_start = None
            line_end = None

            for target_word in line_words:
                # Find matching word in timestamps
                while word_idx < len(word_timestamps):
                    wt = word_timestamps[word_idx]
                    if wt.word.lower() == target_word.lower():
                        line_word_segments.append(
                            WordSegment(
                                word=wt.word,
                                start=wt.start,
                                end=wt.end,
                                confidence=wt.confidence,
                            )
                        )
                        if line_start is None:
                            line_start = wt.start
                        line_end = wt.end
                        word_idx += 1
                        break
                    word_idx += 1

            results.append(
                SyncResult(
                    text=line.text,
                    start_time=float(line_start) if line_start else 0.0,
                    end_time=float(line_end)
                    if line_end
                    else (results[-1].end_time if results else 0.0),
                    confidence=float(confidence),
                    line_number=line_idx,
                    word_segments=line_word_segments if line_word_segments else None,
                )
            )

        return results

    def transcribe(
        self,
        audio: AudioData,
        language: str | None = None,
    ) -> TranscriptionResult:
        """SOFA doesn't support transcription (forced alignment only)."""
        raise NotImplementedError("SOFA is a forced aligner and doesn't support transcription")

    @staticmethod
    def get_engine_type() -> Literal["sofa"]:
        return "sofa"

    def get_transcription_sets(
        self,
    ) -> list[tuple[list[WordTimestamp], MatchStats | None, str]]:
        if self._last_word_timestamps:
            return [(self._last_word_timestamps, self._last_match_stats, "sofa")]
        return []

    def get_last_transcription_data(
        self,
    ) -> tuple[list[WordTimestamp], MatchStats | None, str] | None:
        if self._last_word_timestamps:
            return (self._last_word_timestamps, self._last_match_stats, "sofa")
        return None
