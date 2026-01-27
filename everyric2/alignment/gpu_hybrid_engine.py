"""GPU Hybrid Engine combining CTC and NeMo for best accuracy."""

import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from everyric2.alignment.base import (
    AlignmentError,
    BaseAlignmentEngine,
    TranscriptionResult,
)
from everyric2.alignment.ctc_engine import CTCEngine
from everyric2.alignment.nemo_engine import NeMoEngine
from everyric2.audio.loader import AudioData
from everyric2.config.settings import AlignmentSettings
from everyric2.inference.prompt import LyricLine, SyncResult


class GPUHybridEngine(BaseAlignmentEngine):
    """
    GPU Hybrid Engine running CTC and NeMo in parallel.

    Selects best result based on confidence scores.
    Both engines run on GPU for maximum speed.
    """

    def __init__(self, config: AlignmentSettings | None = None):
        super().__init__(config)
        self.ctc = CTCEngine(config)
        self.nemo = NeMoEngine(config)
        self._transcription_sets: list[tuple] = []
        self._progress_lock = threading.Lock()
        self._engine_status: dict[str, str] = {}
        self._engine_step: dict[str, str] = {}

    def is_available(self) -> bool:
        return self.ctc.is_available() or self.nemo.is_available()

    def transcribe(
        self,
        audio: AudioData,
        language: str | None = None,
    ) -> TranscriptionResult:
        raise NotImplementedError("GPUHybridEngine is for forced alignment only.")

    def align(
        self,
        audio: AudioData,
        lyrics: list[LyricLine],
        language: str | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[SyncResult]:
        resolved_lang = self._resolve_language(language)
        self._transcription_sets = []
        self._engine_status = {}

        ctc_available = self.ctc.is_available()
        nemo_available = self.nemo.is_available()

        if ctc_available:
            self._engine_status["ctc"] = "waiting"
        if nemo_available:
            self._engine_status["nemo"] = "waiting"

        if not ctc_available and not nemo_available:
            raise AlignmentError(
                "No GPU alignment engine available. Install ctc-forced-aligner or nemo_toolkit[asr]"
            )

        if ctc_available and nemo_available:
            return self._align_parallel(audio, lyrics, resolved_lang, progress_callback)
        elif ctc_available:
            return self._align_ctc_only(audio, lyrics, resolved_lang, progress_callback)
        else:
            return self._align_nemo_only(audio, lyrics, resolved_lang, progress_callback)

    def _align_parallel(
        self,
        audio: AudioData,
        lyrics: list[LyricLine],
        language: str,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[SyncResult]:
        ctc_results = None
        nemo_results = None
        ctc_error = None
        nemo_error = None
        done_event = threading.Event()

        def ctc_progress(current: int, total: int) -> None:
            self._update_step("ctc", f"{current}/{total}")
            self._report_progress(progress_callback)

        def nemo_progress(current: int, total: int) -> None:
            self._update_step("nemo", f"{current}/{total}")
            self._report_progress(progress_callback)

        def run_ctc():
            nonlocal ctc_results, ctc_error
            try:
                self._update_status("ctc", "running")
                self._report_progress(progress_callback)
                ctc_results = self.ctc.align(audio, lyrics, language, ctc_progress)
                self._update_status("ctc", "done")
                self._report_progress(progress_callback)
            except Exception as e:
                ctc_error = e
                self._update_status("ctc", "failed")
                self._report_progress(progress_callback)

        def run_nemo():
            nonlocal nemo_results, nemo_error
            try:
                self._update_status("nemo", "running")
                self._report_progress(progress_callback)
                nemo_results = self.nemo.align(audio, lyrics, language, nemo_progress)
                self._update_status("nemo", "done")
                self._report_progress(progress_callback)
            except Exception as e:
                nemo_error = e
                self._update_status("nemo", "failed")
                self._report_progress(progress_callback)

        def progress_reporter():
            while not done_event.is_set():
                self._report_progress(progress_callback)
                time.sleep(0.5)

        with ThreadPoolExecutor(max_workers=3) as executor:
            reporter_future = executor.submit(progress_reporter)
            ctc_future = executor.submit(run_ctc)
            nemo_future = executor.submit(run_nemo)
            ctc_future.result()
            nemo_future.result()
            done_event.set()
            reporter_future.result()

        if ctc_results is not None:
            self._collect_transcription_sets(self.ctc, "ctc")
        if nemo_results is not None:
            self._collect_transcription_sets(self.nemo, "nemo")

        return self._select_best_result(ctc_results, nemo_results, ctc_error, nemo_error)

    def _select_best_result(
        self,
        ctc_results: list[SyncResult] | None,
        nemo_results: list[SyncResult] | None,
        ctc_error: Exception | None,
        nemo_error: Exception | None,
    ) -> list[SyncResult]:
        if ctc_results is None and nemo_results is None:
            errors = []
            if ctc_error:
                errors.append(f"CTC: {ctc_error}")
            if nemo_error:
                errors.append(f"NeMo: {nemo_error}")
            raise AlignmentError(f"Both GPU engines failed: {'; '.join(errors)}")

        if ctc_results is None:
            return nemo_results
        if nemo_results is None:
            return ctc_results

        ctc_stats = getattr(self.ctc, "_last_match_stats", None)
        nemo_stats = getattr(self.nemo, "_last_match_stats", None)

        ctc_score = ctc_stats.get("match_rate", 0) if ctc_stats else 0
        nemo_score = nemo_stats.get("match_rate", 0) if nemo_stats else 0

        if nemo_score >= ctc_score:
            return nemo_results
        return ctc_results

    def _align_ctc_only(
        self,
        audio: AudioData,
        lyrics: list[LyricLine],
        language: str,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[SyncResult]:
        self._update_status("ctc", "running")
        self._report_progress(progress_callback)
        results = self.ctc.align(audio, lyrics, language, progress_callback)
        self._collect_transcription_sets(self.ctc, "ctc")
        self._update_status("ctc", "done")
        return results

    def _align_nemo_only(
        self,
        audio: AudioData,
        lyrics: list[LyricLine],
        language: str,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[SyncResult]:
        self._update_status("nemo", "running")
        self._report_progress(progress_callback)
        results = self.nemo.align(audio, lyrics, language, progress_callback)
        self._collect_transcription_sets(self.nemo, "nemo")
        self._update_status("nemo", "done")
        return results

    def _collect_transcription_sets(self, engine, name: str) -> None:
        if hasattr(engine, "get_transcription_sets"):
            self._transcription_sets.extend(engine.get_transcription_sets())

    def _update_status(self, engine: str, status: str) -> None:
        with self._progress_lock:
            self._engine_status[engine] = status

    def _update_step(self, engine: str, step: str) -> None:
        with self._progress_lock:
            self._engine_step[engine] = step

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
            steps = self._engine_step.copy()

        parts = []
        for engine, status in statuses.items():
            step = steps.get(engine, "")
            if status == "running":
                if step:
                    parts.append(f"{engine}({step})")
                else:
                    parts.append(f"{engine}...")
            elif status == "done":
                parts.append(f"{engine}:done")
            elif status == "failed":
                parts.append(f"{engine}:fail")
            elif status == "waiting":
                parts.append(f"{engine}:wait")
        return " | ".join(parts) if parts else "preparing..."

    def get_transcription_sets(self) -> list[tuple]:
        return self._transcription_sets

    @staticmethod
    def get_engine_type() -> str:
        return "gpu-hybrid"
