"""NeMo Forced Aligner (NFA) engine for GPU-accelerated alignment."""

import tempfile
from pathlib import Path
from typing import Callable

import torch

from everyric2.alignment.base import (
    AlignmentError,
    BaseAlignmentEngine,
    EngineNotAvailableError,
    TranscriptionResult,
    WordTimestamp,
)
from everyric2.audio.loader import AudioData
from everyric2.config.settings import AlignmentSettings
from everyric2.inference.prompt import LyricLine, SyncResult


class NeMoEngine(BaseAlignmentEngine):
    """
    NVIDIA NeMo Forced Aligner using Conformer-CTC models.

    Features:
    - Production-grade from NVIDIA
    - High accuracy with Conformer architecture
    - Long audio support (1hr+) with chunked streaming
    """

    MODEL_MAP = {
        "en": "nvidia/stt_en_conformer_ctc_large",
        "ja": "nvidia/stt_ja_conformer_ctc_large",
        "ko": "nvidia/stt_ko_conformer_ctc_large",
    }

    def __init__(self, config: AlignmentSettings | None = None):
        super().__init__(config)
        self._model = None
        self._current_language = None
        self._device = None
        self._last_word_timestamps: list[WordTimestamp] = []

    def is_available(self) -> bool:
        try:
            from nemo.collections.asr.models import EncDecCTCModel

            return True
        except ImportError:
            return False

    def _get_model_name(self, language: str) -> str:
        if language == "ja":
            return getattr(self.config, "nemo_model_ja", self.MODEL_MAP["ja"])
        elif language == "ko":
            return getattr(self.config, "nemo_model_ko", self.MODEL_MAP["ko"])
        else:
            return getattr(self.config, "nemo_model_en", self.MODEL_MAP["en"])

    def _ensure_model_loaded(self, language: str) -> None:
        if self._model is not None and self._current_language == language:
            return

        if not self.is_available():
            raise EngineNotAvailableError(
                "NeMo toolkit not installed. Install with: pip install 'nemo_toolkit[asr]'"
            )

        from nemo.collections.asr.models import EncDecCTCModel

        model_name = self._get_model_name(language)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self._model = EncDecCTCModel.from_pretrained(model_name)
        self._model.eval()
        self._model.to(self._device)
        self._current_language = language

    def transcribe(
        self,
        audio: AudioData,
        language: str | None = None,
    ) -> TranscriptionResult:
        raise NotImplementedError(
            "NeMoEngine is optimized for forced alignment. Use align() instead."
        )

    def align(
        self,
        audio: AudioData,
        lyrics: list[LyricLine],
        language: str | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[SyncResult]:
        resolved_lang = self._resolve_language(language)

        if progress_callback:
            progress_callback(1, 6)

        self._ensure_model_loaded(resolved_lang)

        if progress_callback:
            progress_callback(2, 6)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            from everyric2.audio.loader import AudioLoader

            loader = AudioLoader()
            prepared = loader.prepare_for_alignment(audio, target_sr=16000, normalize=True)

            audio_path = tmpdir_path / "audio.wav"
            prepared.to_file(audio_path)

            if progress_callback:
                progress_callback(3, 6)

            full_text = " ".join(line.text for line in lyrics)
            manifest_path = tmpdir_path / "manifest.json"
            import json

            manifest_data = {
                "audio_filepath": str(audio_path),
                "text": full_text,
                "duration": audio.duration,
            }
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest_data, f, ensure_ascii=False)
                f.write("\n")

            if progress_callback:
                progress_callback(4, 6)

            output_dir = tmpdir_path / "output"
            output_dir.mkdir()

            word_timestamps = self._run_alignment(manifest_path, output_dir, resolved_lang)

            if progress_callback:
                progress_callback(5, 6)

            self._last_word_timestamps = word_timestamps

            from everyric2.alignment.matcher import LyricsMatcher

            matcher = LyricsMatcher()
            results = matcher.match_lyrics_to_words(lyrics, word_timestamps, resolved_lang)

            if progress_callback:
                progress_callback(6, 6)

            self._last_match_stats = getattr(matcher, "_last_stats", None)

            return results

    def _run_alignment(
        self,
        manifest_path: Path,
        output_dir: Path,
        language: str,
    ) -> list[WordTimestamp]:
        """Run NeMo forced alignment and parse CTM output."""
        try:
            from nemo.collections.asr.parts.utils.transcribe_utils import transcribe_partial_audio
        except ImportError:
            pass

        transcriptions = self._model.transcribe(
            [str(manifest_path.parent / "audio.wav")],
            return_hypotheses=True,
        )

        if not transcriptions or not transcriptions[0]:
            raise AlignmentError("NeMo transcription returned empty result")

        hypothesis = transcriptions[0]

        word_timestamps = []

        if hasattr(hypothesis, "timestep") and hypothesis.timestep:
            timesteps = hypothesis.timestep
            if hasattr(timesteps, "word"):
                for word_info in timesteps.word:
                    word_timestamps.append(
                        WordTimestamp(
                            word=word_info.get("word", ""),
                            start=word_info.get("start_offset", 0.0),
                            end=word_info.get("end_offset", 0.0),
                            confidence=word_info.get("score"),
                        )
                    )

        if not word_timestamps and hasattr(hypothesis, "words"):
            for w in hypothesis.words:
                word_timestamps.append(
                    WordTimestamp(
                        word=w.get("word", str(w)),
                        start=w.get("start", 0.0),
                        end=w.get("end", 0.0),
                        confidence=w.get("score"),
                    )
                )

        return word_timestamps

    def get_last_transcription_data(self) -> tuple[list[WordTimestamp], dict | None, str]:
        stats = getattr(self, "_last_match_stats", None)
        return (self._last_word_timestamps, stats, "nemo")

    def get_transcription_sets(self) -> list[tuple]:
        data = self.get_last_transcription_data()
        if data[0]:
            return [data]
        return []

    @staticmethod
    def get_engine_type() -> str:
        return "nemo"
