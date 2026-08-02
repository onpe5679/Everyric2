from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from everyric2.alignment.emission import EngineEmission
from everyric2.audio.loader import AudioData
from everyric2.config.settings import AlignmentSettings
from everyric2.inference.prompt import LyricLine, SyncResult

if TYPE_CHECKING:
    from everyric2.alignment.matcher import MatchStats


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
    def get_engine_type() -> (
        Literal["whisperx", "qwen", "ctc", "nemo", "gpu-hybrid", "sofa", "owsm", "omniasr"]
    ):
        raise NotImplementedError

    # Optional methods - subclasses can override these
    def get_status_string(self) -> str | None:
        """Get current processing status string for progress display."""
        return None

    def emission_for(self, audio: AudioData) -> EngineEmission | None:
        """곡 전체 CTC emission 노출 (2패스 리파이너용). 기본은 미지원 — ``None``.

        지원하는 엔진(예: ``OmniASREngine``)만 override한다. 서브프로세스로 격리된 엔진은
        emission 텐서가 프로세스 경계를 못 넘으므로 구현하지 않는다 —
        ``everyric2/alignment/emission.py`` 모듈 docstring 참고.
        """
        return None

    def get_transcription_sets(
        self,
    ) -> list[tuple[list[WordTimestamp], "MatchStats | None", str]]:
        """Get all transcription data sets (for engines that produce multiple)."""
        return []

    def get_last_transcription_data(
        self,
    ) -> tuple[list[WordTimestamp], "MatchStats | None", str] | None:
        """Get the last transcription data (words, stats, engine_name)."""
        return None
