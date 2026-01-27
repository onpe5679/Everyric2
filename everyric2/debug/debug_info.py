import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


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
class DebugInfo:
    source: str
    title: str | None
    command: str
    settings: dict[str, Any]

    original_lyrics: str = ""
    translated_lyrics: str | None = None

    audio_duration: float = 0.0
    audio_sample_rate: int = 0

    steps: list[StepTiming] = field(default_factory=list)
    final_results: list[dict] = field(default_factory=list)

    transcription_words: list[dict] = field(default_factory=list)
    transcription_engine: str | None = None
    match_stats: dict[str, Any] = field(default_factory=dict)
    transcription_sets: list[dict] = field(default_factory=list)

    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None
    _current_step: StepTiming | None = field(default=None, repr=False)

    errors: list[str] = field(default_factory=list)
    output_dir: Path | None = None

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

    def add_transcription_data(
        self,
        words: list,
        match_stats: Any = None,
        engine_name: str | None = None,
    ) -> None:
        entry = {
            "engine": engine_name or "transcription",
            "words": [
                {
                    "word": w.word,
                    "start": w.start,
                    "end": w.end,
                    "confidence": w.confidence,
                }
                for w in words
            ],
            "match_stats": {
                "total_lyrics": getattr(match_stats, "total_lyrics", 0) if match_stats else 0,
                "matched_lyrics": getattr(match_stats, "matched_lyrics", 0) if match_stats else 0,
                "match_rate": getattr(match_stats, "match_rate", 0.0) if match_stats else 0.0,
                "avg_confidence": getattr(match_stats, "avg_confidence", 0.0)
                if match_stats
                else 0.0,
            },
        }
        self.transcription_sets.append(entry)

        if not self.transcription_words:
            self.transcription_words = entry["words"]
            self.match_stats = entry["match_stats"]
            self.transcription_engine = entry["engine"]

    def set_transcription_data(
        self,
        words: list,
        match_stats: Any = None,
        engine_name: str | None = None,
    ) -> None:
        self.add_transcription_data(words, match_stats, engine_name)

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
            "transcription_words": self.transcription_words,
            "transcription_engine": self.transcription_engine,
            "transcription_sets": self.transcription_sets,
            "match_stats": self.match_stats,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_duration": (self.end_time - self.start_time).total_seconds()
            if self.end_time
            else None,
            "steps": [{"name": s.name, "duration_seconds": s.duration} for s in self.steps],
            "final_results": self.final_results,
            "errors": self.errors,
        }
