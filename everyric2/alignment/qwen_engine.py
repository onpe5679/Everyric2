from collections.abc import Callable
from typing import Literal

from everyric2.alignment.base import (
    BaseAlignmentEngine,
    EngineNotAvailableError,
    TranscriptionResult,
)
from everyric2.audio.loader import AudioData
from everyric2.config.settings import AlignmentSettings
from everyric2.inference.prompt import LyricLine, SyncResult


class QwenEngine(BaseAlignmentEngine):
    def __init__(self, config: AlignmentSettings | None = None):
        super().__init__(config)
        self._engine = None

    def is_available(self) -> bool:
        try:
            from everyric2.inference.qwen_omni_gguf import QwenOmniGGUFEngine  # noqa: F401

            return True
        except ImportError:
            return False

    def _ensure_engine_loaded(self) -> None:
        if self._engine is not None:
            return

        if not self.is_available():
            raise EngineNotAvailableError("Qwen-Omni engine not available")

        from everyric2.config.settings import get_settings
        from everyric2.inference.qwen_omni_gguf import QwenOmniGGUFEngine

        self._engine = QwenOmniGGUFEngine(get_settings().model)
        self._engine.load_model()

    def transcribe(
        self,
        audio: AudioData,
        language: str | None = None,
    ) -> TranscriptionResult:
        raise NotImplementedError(
            "Qwen engine does not support standalone transcription. "
            "Use align() with lyrics instead."
        )

    def align(
        self,
        audio: AudioData,
        lyrics: list[LyricLine],
        language: str | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[SyncResult]:
        self._ensure_engine_loaded()

        results = self._engine.sync_lyrics(
            audio,
            lyrics,
            progress_callback=progress_callback,
        )

        return results

    def unload(self) -> None:
        if self._engine is not None:
            self._engine.unload_model()
            self._engine = None

    @staticmethod
    def get_engine_type() -> Literal["whisperx", "qwen", "ctc", "nemo", "gpu-hybrid"]:
        return "qwen"
