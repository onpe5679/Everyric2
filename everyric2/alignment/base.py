from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

from everyric2.audio.loader import AudioData
from everyric2.config.settings import AlignmentSettings
from everyric2.inference.prompt import LyricLine, SyncResult


class AlignmentError(Exception):
    pass


class EngineNotAvailableError(AlignmentError):
    pass


@dataclass
class WordTimestamp:
    word: str
    start: float
    end: float
    confidence: float | None = None


@dataclass
class TranscriptionResult:
    text: str
    language: str
    segments: list[dict] = field(default_factory=list)
    words: list[WordTimestamp] = field(default_factory=list)


class BaseAlignmentEngine(ABC):
    SUPPORTED_LANGUAGES: list[str] = ["en", "ja", "ko"]

    def __init__(self, config: AlignmentSettings | None = None):
        from everyric2.config.settings import get_settings

        self.config = config or get_settings().alignment

    @abstractmethod
    def is_available(self) -> bool:
        pass

    @abstractmethod
    def align(
        self,
        audio: AudioData,
        lyrics: list[LyricLine],
        language: str | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[SyncResult]:
        pass

    @abstractmethod
    def transcribe(
        self,
        audio: AudioData,
        language: str | None = None,
    ) -> TranscriptionResult:
        pass

    def detect_language(self, audio: AudioData) -> str:
        return "en"

    def _resolve_language(self, language: str | None) -> str:
        if language and language != "auto":
            return language
        return self.config.language if self.config.language != "auto" else "en"

    @staticmethod
    def get_engine_type() -> Literal["whisperx", "qwen", "ctc", "nemo", "gpu-hybrid"]:
        raise NotImplementedError
