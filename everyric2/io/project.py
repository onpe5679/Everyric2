import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from everyric2.inference.prompt import SyncResult, WordSegment


@dataclass
class ProjectMetadata:
    version: str = "1.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    source_audio: str | None = None
    source_lyrics: str | None = None
    language: str = "ja"
    engine: str = "ctc"
    audio_duration: float | None = None


@dataclass
class TranslationData:
    line_index: int
    original: str
    translation: str | None = None
    pronunciation: str | None = None


@dataclass
class AlignmentData:
    metadata: ProjectMetadata
    results: list[SyncResult]
    translations: list[TranslationData] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": asdict(self.metadata),
            "results": [self._sync_result_to_dict(r) for r in self.results],
            "translations": [asdict(t) for t in self.translations],
        }

    @staticmethod
    def _sync_result_to_dict(r: SyncResult) -> dict[str, Any]:
        d = {
            "text": r.text,
            "start_time": r.start_time,
            "end_time": r.end_time,
            "confidence": r.confidence,
            "line_number": r.line_number,
        }
        if r.word_segments:
            d["word_segments"] = [
                {"word": w.word, "start": w.start, "end": w.end, "confidence": w.confidence}
                for w in r.word_segments
            ]
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AlignmentData":
        metadata = ProjectMetadata(**data.get("metadata", {}))

        results = []
        for r in data.get("results", []):
            word_segments = None
            if "word_segments" in r and r["word_segments"]:
                word_segments = [
                    WordSegment(w["word"], w["start"], w["end"], w.get("confidence", 1.0))
                    for w in r["word_segments"]
                ]
            results.append(
                SyncResult(
                    text=r["text"],
                    start_time=r["start_time"],
                    end_time=r["end_time"],
                    confidence=r.get("confidence", 1.0),
                    line_number=r.get("line_number"),
                    word_segments=word_segments,
                )
            )

        translations = [TranslationData(**t) for t in data.get("translations", [])]

        return cls(metadata=metadata, results=results, translations=translations)


class ProjectFile:
    EXTENSION = ".everyric.json"

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.data: AlignmentData | None = None

    def save(self, data: AlignmentData) -> Path:
        self.data = data
        content = json.dumps(data.to_dict(), ensure_ascii=False, indent=2)
        self.path.write_text(content, encoding="utf-8")
        return self.path

    def load(self) -> AlignmentData:
        content = self.path.read_text(encoding="utf-8")
        raw = json.loads(content)
        self.data = AlignmentData.from_dict(raw)
        return self.data

    @classmethod
    def from_sync_results(
        cls,
        results: list[SyncResult],
        output_path: Path,
        metadata: ProjectMetadata | None = None,
        translation_result: Any | None = None,
    ) -> "ProjectFile":
        if metadata is None:
            metadata = ProjectMetadata()

        translations = []
        if translation_result and hasattr(translation_result, "lines"):
            for i, line in enumerate(translation_result.lines):
                translations.append(
                    TranslationData(
                        line_index=i,
                        original=line.original,
                        translation=line.translation,
                        pronunciation=line.pronunciation,
                    )
                )

        alignment_data = AlignmentData(
            metadata=metadata,
            results=results,
            translations=translations,
        )

        project_path = output_path.with_suffix(cls.EXTENSION)
        project = cls(project_path)
        project.save(alignment_data)
        return project

    def get_line_results(self) -> list[SyncResult]:
        if not self.data:
            raise ValueError("No data loaded")
        return self.data.results

    def apply_translations(self, results: list[SyncResult]) -> list[SyncResult]:
        if not self.data or not self.data.translations:
            return results

        trans_map = {t.line_index: t for t in self.data.translations}

        for r in results:
            if r.line_number is not None and r.line_number in trans_map:
                t = trans_map[r.line_number]
                r.translation = t.translation
                r.pronunciation = t.pronunciation

        return results

    def get_translation_track(self) -> list[SyncResult]:
        if not self.data:
            raise ValueError("No data loaded")

        trans_map = {t.line_index: t for t in self.data.translations}
        track = []

        for r in self.data.results:
            if r.line_number is not None and r.line_number in trans_map:
                t = trans_map[r.line_number]
                track.append(
                    SyncResult(
                        text=t.translation or "",
                        start_time=r.start_time,
                        end_time=r.end_time,
                        confidence=r.confidence,
                        line_number=r.line_number,
                    )
                )

        return track
