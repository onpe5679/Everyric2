import gc
from pathlib import Path
from typing import Callable, Literal

import numpy as np

from everyric2.alignment.base import (
    AlignmentError,
    BaseAlignmentEngine,
    EngineNotAvailableError,
    TranscriptionResult,
    WordTimestamp,
)
from everyric2.audio.loader import AudioData, AudioLoader
from everyric2.config.settings import AlignmentSettings
from everyric2.inference.prompt import LyricLine, SyncResult


class WhisperXEngine(BaseAlignmentEngine):
    ALIGNMENT_MODELS = {
        "en": "WAV2VEC2_ASR_BASE_960H",
        "ja": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
        "ko": "kresnik/wav2vec2-large-xlsr-korean",
    }

    def __init__(self, config: AlignmentSettings | None = None):
        super().__init__(config)
        self._model = None
        self._align_model = None
        self._align_metadata = None
        self._current_language = None
        self.audio_loader = AudioLoader()
        self._safe_globals_fixed = False
        self._last_matcher = None
        self._last_match_stats = None

    def _enable_tf32(self) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    def _fix_torch_safe_globals(self) -> None:
        if self._safe_globals_fixed:
            return
        import torch

        _original_load = torch.load

        def _patched_load(*args, **kwargs):
            if "weights_only" not in kwargs:
                kwargs["weights_only"] = False
            return _original_load(*args, **kwargs)

        torch.load = _patched_load
        self._safe_globals_fixed = True

    def is_available(self) -> bool:
        try:
            import whisperx

            return True
        except ImportError:
            return False

    def _ensure_model_loaded(self) -> None:
        if self._model is not None:
            return

        if not self.is_available():
            raise EngineNotAvailableError(
                "WhisperX not installed. Install with: pip install whisperx"
            )

        import torch
        import whisperx

        self._fix_torch_safe_globals()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = self.config.whisperx_compute_type
        if device == "cpu" and compute_type == "float16":
            compute_type = "float32"

        vad_method = getattr(self.config, "whisperx_vad_method", "silero")
        self._model = whisperx.load_model(
            self.config.whisperx_model,
            device,
            compute_type=compute_type,
            vad_method=vad_method,
        )
        self._device = device
        self._enable_tf32()

    def _ensure_align_model_loaded(self, language: str) -> None:
        if self._align_model is not None and self._current_language == language:
            return

        import whisperx

        if self._align_model is not None:
            del self._align_model
            del self._align_metadata
            gc.collect()

        self._align_model, self._align_metadata = whisperx.load_align_model(
            language_code=language,
            device=self._device,
        )
        self._current_language = language
        self._enable_tf32()

    def transcribe(
        self,
        audio: AudioData,
        language: str | None = None,
    ) -> TranscriptionResult:
        self._ensure_model_loaded()
        import whisperx

        prepared = self.audio_loader.prepare_for_alignment(
            audio,
            target_sr=16000,
            normalize=True,
        )

        result = self._model.transcribe(
            prepared.waveform,
            batch_size=self.config.whisperx_batch_size,
            language=language if language and language != "auto" else None,
        )

        detected_lang = result.get("language", language or "en")
        resolved_lang = self._resolve_language(detected_lang)

        self._ensure_align_model_loaded(resolved_lang)
        aligned = whisperx.align(
            result["segments"],
            self._align_model,
            self._align_metadata,
            prepared.waveform,
            self._device,
            return_char_alignments=False,
        )

        words = []
        for segment in aligned.get("segments", []):
            for word_data in segment.get("words", []):
                if "start" in word_data and "end" in word_data:
                    words.append(
                        WordTimestamp(
                            word=word_data.get("word", ""),
                            start=word_data["start"],
                            end=word_data["end"],
                            confidence=word_data.get("score"),
                        )
                    )

        full_text = " ".join(seg.get("text", "") for seg in result.get("segments", []))

        return TranscriptionResult(
            text=full_text.strip(),
            language=resolved_lang,
            segments=aligned.get("segments", []),
            words=words,
        )

    def align(
        self,
        audio: AudioData,
        lyrics: list[LyricLine],
        language: str | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[SyncResult]:
        if progress_callback:
            progress_callback(1, 3)

        transcription = self.transcribe(audio, language)

        if progress_callback:
            progress_callback(2, 3)

        from everyric2.alignment.matcher import LyricsMatcher

        matcher = LyricsMatcher()
        results = matcher.match_lyrics_to_words(
            lyrics,
            transcription.words,
            transcription.language,
        )
        self._last_matcher = matcher
        self._last_match_stats = matcher.last_match_stats

        if progress_callback:
            progress_callback(3, 3)

        return results

    def get_last_transcription_data(self):
        if self._last_matcher is None:
            return None, None, None
        return (
            self._last_matcher.last_transcription_words,
            self._last_match_stats,
            "whisperx",
        )

    def get_transcription_sets(self):
        words, stats, engine = self.get_last_transcription_data()
        if words:
            return [(words, stats, engine)]
        return []

    def unload(self) -> None:
        import torch

        if self._model is not None:
            del self._model
            self._model = None

        if self._align_model is not None:
            del self._align_model
            del self._align_metadata
            self._align_model = None
            self._align_metadata = None
            self._current_language = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def get_engine_type() -> Literal["whisperx", "qwen", "ctc", "nemo", "gpu-hybrid"]:
        return "whisperx"
