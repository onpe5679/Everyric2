import gc
from collections.abc import Callable
from typing import Literal

from everyric2.alignment.base import (
    BaseAlignmentEngine,
    EngineNotAvailableError,
    TranscriptionResult,
    WordTimestamp,
)
from everyric2.audio.loader import AudioData, AudioLoader
from everyric2.config.settings import AlignmentSettings
from everyric2.inference.prompt import LyricLine, SyncResult


class FasterWhisperEngine(BaseAlignmentEngine):
    def __init__(self, config: AlignmentSettings | None = None):
        super().__init__(config)
        self._model = None
        self.audio_loader = AudioLoader()
        self._device = None

    def is_available(self) -> bool:
        try:
            import faster_whisper  # noqa: F401

            return True
        except ImportError:
            return False

    def _ensure_model_loaded(self) -> None:
        if self._model is not None:
            return

        if not self.is_available():
            raise EngineNotAvailableError(
                "faster-whisper not installed. Install with: pip install faster-whisper"
            )

        import torch
        from faster_whisper import WhisperModel

        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = self.config.whisperx_compute_type

        if device == "cpu" and compute_type == "float16":
            compute_type = "float32"

        self._model = WhisperModel(
            self.config.whisperx_model,
            device=device,
            compute_type=compute_type,
        )
        self._device = device

    def transcribe(
        self,
        audio: AudioData,
        language: str | None = None,
    ) -> TranscriptionResult:
        self._ensure_model_loaded()

        prepared = self.audio_loader.prepare_for_alignment(
            audio,
            target_sr=16000,
            normalize=True,
        )

        segments, info = self._model.transcribe(
            prepared.waveform,
            language=language if language and language != "auto" else None,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        segment_list = list(segments)

        words = []
        full_text_parts = []
        segments_dict_list = []

        for segment in segment_list:
            full_text_parts.append(segment.text)

            seg_words = []
            if segment.words:
                for word in segment.words:
                    w_obj = WordTimestamp(
                        word=word.word,
                        start=word.start,
                        end=word.end,
                        confidence=word.probability,
                    )
                    words.append(w_obj)

                    seg_words.append(
                        {
                            "word": word.word,
                            "start": word.start,
                            "end": word.end,
                            "score": word.probability,
                        }
                    )

            segments_dict_list.append(
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "words": seg_words,
                }
            )

        full_text = " ".join(full_text_parts).strip()
        resolved_lang = self._resolve_language(info.language)

        return TranscriptionResult(
            text=full_text,
            language=resolved_lang,
            segments=segments_dict_list,
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

        if progress_callback:
            progress_callback(3, 3)

        return results

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None

        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def get_engine_type() -> Literal[
        "whisperx", "qwen", "ctc", "nemo", "gpu-hybrid", "faster_whisper"
    ]:
        return "faster_whisper"
