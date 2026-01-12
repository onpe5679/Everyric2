"""Debug information collection."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import time
from pathlib import Path


def _serialize_value(v):
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, dict):
        return {k: _serialize_value(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_serialize_value(item) for item in v]
    return v


@dataclass
class StepTiming:
    name: str
    start_time: float
    end_time: float = 0.0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time if self.end_time else 0.0


@dataclass
class ChunkDebugInfo:
    """Debug info for a single chunk."""

    chunk_idx: int
    audio_start: float
    audio_end: float
    lyrics_start_idx: int
    lyrics_end_idx: int
    prompt: str
    response: str
    parsed_results: list[dict]
    processing_time: float


@dataclass
class DebugInfo:
    """Collects all debug information for a sync run."""

    source: str
    title: str | None
    command: str
    settings: dict[str, Any]

    original_lyrics: str = ""
    translated_lyrics: str | None = None

    audio_duration: float = 0.0
    audio_sample_rate: int = 0

    chunks: list[ChunkDebugInfo] = field(default_factory=list)
    steps: list[StepTiming] = field(default_factory=list)

    final_results: list[dict] = field(default_factory=list)

    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None
    _current_step: StepTiming | None = field(default=None, repr=False)

    errors: list[str] = field(default_factory=list)

    output_dir: Path | None = None
    _chunk_save_callback: Any = field(default=None, repr=False)

    def add_chunk(
        self,
        chunk_idx: int,
        audio_start: float,
        audio_end: float,
        lyrics_start_idx: int,
        lyrics_end_idx: int,
        prompt: str,
        response: str,
        parsed_results: list,
        processing_time: float,
    ) -> None:
        self.chunks.append(
            ChunkDebugInfo(
                chunk_idx=chunk_idx,
                audio_start=audio_start,
                audio_end=audio_end,
                lyrics_start_idx=lyrics_start_idx,
                lyrics_end_idx=lyrics_end_idx,
                prompt=prompt,
                response=response,
                parsed_results=[
                    {"text": r.text, "start": r.start_time, "end": r.end_time}
                    for r in parsed_results
                ],
                processing_time=processing_time,
            )
        )

    def add_error(self, error: str) -> None:
        self.errors.append(error)

    def start_step(self, name: str) -> None:
        if self._current_step:
            self.end_step()
        self._current_step = StepTiming(name=name, start_time=time.time())

    def end_step(self) -> None:
        if self._current_step:
            self._current_step.end_time = time.time()
            self.steps.append(self._current_step)
            self._current_step = None

    def finalize(self, results: list) -> None:
        self.end_time = datetime.now()
        self.final_results = [
            {"text": r.text, "start": r.start_time, "end": r.end_time} for r in results
        ]

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "title": self.title,
            "command": self.command,
            "settings": _serialize_value(self.settings),
            "original_lyrics": self.original_lyrics,
            "translated_lyrics": self.translated_lyrics,
            "audio_duration": self.audio_duration,
            "audio_sample_rate": self.audio_sample_rate,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_duration": (self.end_time - self.start_time).total_seconds()
            if self.end_time
            else None,
            "num_chunks": len(self.chunks),
            "steps": [{"name": s.name, "duration_seconds": s.duration} for s in self.steps],
            "chunks": [
                {
                    "chunk_idx": c.chunk_idx,
                    "audio_start": c.audio_start,
                    "audio_end": c.audio_end,
                    "lyrics_range": f"{c.lyrics_start_idx}-{c.lyrics_end_idx}",
                    "prompt": c.prompt,
                    "response": c.response,
                    "parsed_results": c.parsed_results,
                    "processing_time": c.processing_time,
                }
                for c in self.chunks
            ],
            "final_results": self.final_results,
            "errors": self.errors,
        }
