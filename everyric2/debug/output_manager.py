"""Output folder and debug file management."""

import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from everyric2.audio.loader import AudioData


@dataclass
class RunContext:
    """Context for a single sync run."""

    run_id: str
    output_dir: Path
    title: str | None = None
    source: str | None = None
    command: str | None = None
    settings: dict[str, Any] = field(default_factory=dict)

    # Intermediate data
    prompts: list[str] = field(default_factory=list)
    llm_responses: list[str] = field(default_factory=list)
    chunk_results: list[dict] = field(default_factory=list)

    # Timing info
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None

    def add_prompt(self, prompt: str) -> None:
        self.prompts.append(prompt)

    def add_response(self, response: str) -> None:
        self.llm_responses.append(response)

    def add_chunk_result(
        self, chunk_idx: int, start_time: float, end_time: float, results: list
    ) -> None:
        self.chunk_results.append(
            {
                "chunk_idx": chunk_idx,
                "audio_start": start_time,
                "audio_end": end_time,
                "results": [
                    {"text": r.text, "start": r.start_time, "end": r.end_time} for r in results
                ],
            }
        )


class OutputManager:
    """Manages output folder structure and debug files."""

    def __init__(self, base_dir: Path | str = "output"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_run_context(
        self,
        title: str | None = None,
        source: str | None = None,
        command: str | None = None,
        settings: dict | None = None,
    ) -> RunContext:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if title:
            safe_title = self._sanitize_filename(title)[:50]
            folder_name = f"{timestamp}_{safe_title}"
        else:
            folder_name = timestamp

        output_dir = self.base_dir / folder_name
        output_dir.mkdir(parents=True, exist_ok=True)

        return RunContext(
            run_id=timestamp,
            output_dir=output_dir,
            title=title,
            source=source,
            command=command,
            settings=settings or {},
        )

    def _sanitize_filename(self, name: str) -> str:
        name = re.sub(r'[<>:"/\\|?*]', "_", name)
        name = re.sub(r"\s+", "_", name)
        return name.strip("_")

    def save_lyrics(
        self, ctx: RunContext, lyrics_text: str, filename: str = "lyrics_original.txt"
    ) -> Path:
        path = ctx.output_dir / filename
        path.write_text(lyrics_text, encoding="utf-8")
        return path

    def save_translated_lyrics(self, ctx: RunContext, translated: str) -> Path:
        return self.save_lyrics(ctx, translated, "lyrics_translated_ko.txt")

    def save_audio(
        self, ctx: RunContext, audio: AudioData, filename: str = "audio_original.wav"
    ) -> Path:
        path = ctx.output_dir / filename
        audio.to_file(path)
        return path

    def copy_audio_file(
        self, ctx: RunContext, source_path: Path, filename: str = "audio_original"
    ) -> Path:
        dest = ctx.output_dir / f"{filename}{source_path.suffix}"
        shutil.copy2(source_path, dest)
        return dest

    def _json_default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def save_settings(self, ctx: RunContext) -> Path:
        path = ctx.output_dir / "settings.json"
        data = {
            "run_id": ctx.run_id,
            "title": ctx.title,
            "source": ctx.source,
            "command": ctx.command,
            "settings": ctx.settings,
            "start_time": ctx.start_time.isoformat(),
        }
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False, default=self._json_default),
            encoding="utf-8",
        )
        return path

    def save_debug_info(self, ctx: RunContext) -> Path:
        ctx.end_time = datetime.now()
        path = ctx.output_dir / "debug_info.json"

        data = {
            "run_id": ctx.run_id,
            "title": ctx.title,
            "source": ctx.source,
            "command": ctx.command,
            "start_time": ctx.start_time.isoformat(),
            "end_time": ctx.end_time.isoformat(),
            "duration_seconds": (ctx.end_time - ctx.start_time).total_seconds(),
            "num_chunks": len(ctx.chunk_results),
            "prompts": ctx.prompts,
            "llm_responses": ctx.llm_responses,
            "chunk_results": ctx.chunk_results,
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def save_final_output(self, ctx: RunContext, content: str, format: str = "srt") -> Path:
        path = ctx.output_dir / f"output.{format}"
        path.write_text(content, encoding="utf-8")
        return path

    def save_translated_output(self, ctx: RunContext, content: str, format: str = "srt") -> Path:
        path = ctx.output_dir / f"output_translated.{format}"
        path.write_text(content, encoding="utf-8")
        return path

    def get_chunks_dir(self, ctx: RunContext) -> Path:
        chunks_dir = ctx.output_dir / "chunks"
        chunks_dir.mkdir(exist_ok=True)
        return chunks_dir

    def save_chunk_audio(
        self,
        ctx: RunContext,
        audio: AudioData,
        chunk_idx: int,
        start_time: float,
        end_time: float,
    ) -> Path:
        chunks_dir = self.get_chunks_dir(ctx)
        filename = f"chunk_{chunk_idx:03d}_{start_time:.1f}s-{end_time:.1f}s.wav"
        path = chunks_dir / filename
        audio.to_file(path)
        return path

    def save_chunk_prompt(self, ctx: RunContext, chunk_idx: int, prompt: str) -> Path:
        chunks_dir = self.get_chunks_dir(ctx)
        path = chunks_dir / f"chunk_{chunk_idx:03d}_prompt.txt"
        path.write_text(prompt, encoding="utf-8")
        return path

    def save_chunk_response(self, ctx: RunContext, chunk_idx: int, response: str) -> Path:
        chunks_dir = self.get_chunks_dir(ctx)
        path = chunks_dir / f"chunk_{chunk_idx:03d}_response.txt"
        path.write_text(response, encoding="utf-8")
        return path
