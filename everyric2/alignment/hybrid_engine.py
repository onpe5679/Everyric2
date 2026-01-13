from typing import Callable, Literal

from everyric2.alignment.base import (
    AlignmentError,
    BaseAlignmentEngine,
    EngineNotAvailableError,
    TranscriptionResult,
)
from everyric2.alignment.mfa_engine import MFAEngine
from everyric2.alignment.whisperx_engine import WhisperXEngine
from everyric2.audio.loader import AudioData
from everyric2.config.settings import AlignmentSettings
from everyric2.inference.prompt import LyricLine, SyncResult


class HybridEngine(BaseAlignmentEngine):
    def __init__(self, config: AlignmentSettings | None = None):
        super().__init__(config)
        self.whisperx = WhisperXEngine(config)
        self.mfa = MFAEngine(config)
        self._last_engine = None

    def is_available(self) -> bool:
        return self.whisperx.is_available()

    def transcribe(
        self,
        audio: AudioData,
        language: str | None = None,
    ) -> TranscriptionResult:
        return self.whisperx.transcribe(audio, language)

    def align(
        self,
        audio: AudioData,
        lyrics: list[LyricLine],
        language: str | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[SyncResult]:
        resolved_lang = self._resolve_language(language)
        self._transcription_sets: list[tuple] = []

        wx_results = None
        try:
            wx_results = self._align_with_whisperx(audio, lyrics, resolved_lang, progress_callback)
            if hasattr(self.whisperx, "get_transcription_sets"):
                self._transcription_sets.extend(self.whisperx.get_transcription_sets())
            elif hasattr(self.whisperx, "get_last_transcription_data"):
                data = self.whisperx.get_last_transcription_data()
                if data and data[0]:
                    self._transcription_sets.append(
                        (data[0], data[1], data[2] if len(data) > 2 else "whisperx")
                    )
        except Exception:
            wx_results = None

        if self.mfa.is_available():
            models_ok, _ = self.mfa._check_models_installed(resolved_lang)
            if models_ok:
                try:
                    mfa_results = self._align_with_mfa(
                        audio, lyrics, resolved_lang, progress_callback
                    )
                    if hasattr(self.mfa, "get_transcription_sets"):
                        self._transcription_sets.extend(self.mfa.get_transcription_sets())
                    elif hasattr(self.mfa, "get_last_transcription_data"):
                        data = self.mfa.get_last_transcription_data()
                        if data and data[0]:
                            self._transcription_sets.append(
                                (data[0], data[1], data[2] if len(data) > 2 else "mfa")
                            )
                    self._last_engine = self.mfa
                    return mfa_results
                except Exception:
                    pass

        self._last_engine = self.whisperx
        return (
            wx_results
            if wx_results is not None
            else self._align_with_whisperx(audio, lyrics, resolved_lang, progress_callback)
        )

    def _align_with_mfa(
        self,
        audio: AudioData,
        lyrics: list[LyricLine],
        language: str,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[SyncResult]:
        def mfa_progress(current: int, total: int) -> None:
            if progress_callback:
                progress_callback(current, total)

        return self.mfa.align(audio, lyrics, language, mfa_progress)

    def _align_with_whisperx(
        self,
        audio: AudioData,
        lyrics: list[LyricLine],
        language: str,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[SyncResult]:
        def whisperx_progress(current: int, total: int) -> None:
            if progress_callback:
                progress_callback(current, total)

        return self.whisperx.align(audio, lyrics, language, whisperx_progress)

    def get_last_transcription_data(self):
        if self._last_engine and hasattr(self._last_engine, "get_last_transcription_data"):
            return self._last_engine.get_last_transcription_data()
        return None, None, None

    def get_transcription_sets(self):
        return getattr(self, "_transcription_sets", [])

    def unload(self) -> None:
        self.whisperx.unload()

    @staticmethod
    def get_engine_type() -> Literal["whisperx", "mfa", "hybrid", "qwen"]:
        return "hybrid"
