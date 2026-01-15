import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        self._progress_lock = threading.Lock()
        self._engine_status: dict[str, str] = {}

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
        self._engine_status = {"whisperx": "waiting", "mfa": "waiting"}

        mfa_available = False
        if self.mfa.is_available():
            models_ok, _ = self.mfa._check_models_installed(resolved_lang)
            mfa_available = models_ok

        if mfa_available:
            return self._align_parallel(audio, lyrics, resolved_lang, progress_callback)
        else:
            return self._align_whisperx_only(audio, lyrics, resolved_lang, progress_callback)

    def _align_parallel(
        self,
        audio: AudioData,
        lyrics: list[LyricLine],
        language: str,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[SyncResult]:
        wx_results = None
        mfa_results = None
        wx_error = None
        mfa_error = None

        def run_whisperx():
            nonlocal wx_results, wx_error
            try:
                self._update_status("whisperx", "running")
                self._report_progress(progress_callback)
                wx_results = self.whisperx.align(audio, lyrics, language, None)
                self._update_status("whisperx", "done")
                self._report_progress(progress_callback)
            except Exception as e:
                wx_error = e
                self._update_status("whisperx", "failed")
                self._report_progress(progress_callback)

        def run_mfa():
            nonlocal mfa_results, mfa_error
            try:
                self._update_status("mfa", "running")
                self._report_progress(progress_callback)
                mfa_results = self.mfa.align(audio, lyrics, language, None)
                self._update_status("mfa", "done")
                self._report_progress(progress_callback)
            except Exception as e:
                mfa_error = e
                self._update_status("mfa", "failed")
                self._report_progress(progress_callback)

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(run_whisperx), executor.submit(run_mfa)]
            for future in as_completed(futures):
                pass

        if wx_results is not None:
            self._collect_transcription_sets(self.whisperx, "whisperx")
        if mfa_results is not None:
            self._collect_transcription_sets(self.mfa, "mfa")

        if mfa_results is not None:
            self._last_engine = self.mfa
            return mfa_results
        elif wx_results is not None:
            self._last_engine = self.whisperx
            return wx_results
        else:
            raise AlignmentError("Both WhisperX and MFA failed")

    def _align_whisperx_only(
        self,
        audio: AudioData,
        lyrics: list[LyricLine],
        language: str,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[SyncResult]:
        self._update_status("whisperx", "running")
        self._report_progress(progress_callback)

        results = self.whisperx.align(audio, lyrics, language, progress_callback)
        self._collect_transcription_sets(self.whisperx, "whisperx")

        self._update_status("whisperx", "done")
        self._last_engine = self.whisperx
        return results

    def _collect_transcription_sets(self, engine, name: str) -> None:
        if hasattr(engine, "get_transcription_sets"):
            self._transcription_sets.extend(engine.get_transcription_sets())
        elif hasattr(engine, "get_last_transcription_data"):
            data = engine.get_last_transcription_data()
            if data and data[0]:
                self._transcription_sets.append(
                    (data[0], data[1], data[2] if len(data) > 2 else name)
                )

    def _update_status(self, engine: str, status: str) -> None:
        with self._progress_lock:
            self._engine_status[engine] = status

    def _report_progress(self, callback: Callable[[int, int], None] | None) -> None:
        if not callback:
            return
        with self._progress_lock:
            statuses = self._engine_status.copy()

        done_count = sum(1 for s in statuses.values() if s in ("done", "failed"))
        total = len(statuses)
        callback(done_count, total)

    def get_status_string(self) -> str:
        with self._progress_lock:
            statuses = self._engine_status.copy()

        parts = []
        for engine, status in statuses.items():
            if status == "running":
                parts.append(f"{engine}...")
            elif status == "done":
                parts.append(f"{engine} ✓")
            elif status == "failed":
                parts.append(f"{engine} ✗")
        return " | ".join(parts) if parts else "preparing..."

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
