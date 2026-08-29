"""Public facade for the single adaptive alignment pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

from everyric2.alignment.base import BaseAlignmentEngine, TranscriptionResult
from everyric2.audio.loader import AudioData
from everyric2.inference.prompt import LyricLine, SyncResult, WordSegment


def _sync_result(item: dict) -> SyncResult:
    words = [
        WordSegment(
            word=str(word.get("word", word.get("text", ""))),
            start=float(word["start"]),
            end=float(word["end"]),
            confidence=word.get("confidence"),
        )
        for word in item.get("words", item.get("word_segments", []))
    ]
    return SyncResult(
        text=str(item["text"]),
        start_time=float(item["start"]),
        end_time=float(item["end"]),
        confidence=item.get("confidence"),
        word_segments=words or None,
        translation=item.get("translation"),
        pronunciation=item.get("pronunciation"),
    )


class AdaptiveEngine(BaseAlignmentEngine):
    """Compatibility with the generic alignment interface, not with old engines."""

    def is_available(self) -> bool:
        try:
            from everyric2.local_runtime import RuntimePaths, health_report

            return bool(health_report(RuntimePaths.resolve())["ready"])
        except Exception:
            return False

    def align(
        self,
        audio: AudioData,
        lyrics: list[LyricLine],
        language: str | None = None,
        progress_callback=None,
    ) -> list[SyncResult]:
        from everyric2.server.worker import _run_alignment

        temporary: Path | None = None
        source = audio.source_path
        if source is None or not source.is_file():
            handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            handle.close()
            temporary = Path(handle.name)
            audio.to_file(temporary)
            source = temporary
        try:
            result = _run_alignment(
                str(source),
                "\n".join(line.text for line in lyrics),
                language,
                on_stage=((lambda _stage: progress_callback(0, 1)) if progress_callback else None),
                delete_audio=False,
            )
            return [_sync_result(item) for item in result.get("timestamps", [])]
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def transcribe(
        self,
        audio: AudioData,
        language: str | None = None,
    ) -> TranscriptionResult:
        raise NotImplementedError("The adaptive public pipeline aligns supplied lyrics")

    @staticmethod
    def get_engine_type() -> str:
        return "adaptive"
