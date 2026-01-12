"""Batch processing for multiple song/lyrics pairs."""

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

from everyric2.audio.downloader import YouTubeDownloader
from everyric2.audio.loader import AudioLoader
from everyric2.config.settings import get_settings
from everyric2.inference.prompt import LyricLine
from everyric2.inference.qwen_omni import QwenOmniEngine
from everyric2.output.formatters import FormatterFactory


@dataclass
class TestCase:
    source: str
    lyrics: list[LyricLine]
    title: str | None = None
    formats: list[str] = field(default_factory=lambda: ["srt"])


@dataclass
class BatchConfig:
    output_dir: Path
    formats: list[str]
    tests: list[TestCase]
    resume: bool = False

    @classmethod
    def from_yaml(cls, path: Path) -> "BatchConfig":
        with open(path) as f:
            data = yaml.safe_load(f)

        output_dir = Path(data.get("output_dir", "./output"))
        formats = data.get("formats", ["srt"])

        tests = []
        for i, item in enumerate(data.get("tests", [])):
            source = item["source"]
            title = item.get("title")

            if "lyrics" in item:
                lyrics = LyricLine.from_text(item["lyrics"])
            elif "lyrics_file" in item:
                lyrics = LyricLine.from_file(item["lyrics_file"])
            else:
                raise ValueError(f"Test #{i + 1} must have 'lyrics' or 'lyrics_file'")

            tests.append(
                TestCase(
                    source=source,
                    lyrics=lyrics,
                    title=title,
                    formats=item.get("formats", formats),
                )
            )

        return cls(output_dir=output_dir, formats=formats, tests=tests)


class BatchRunner:
    def __init__(self, config: BatchConfig):
        self.config = config
        self.engine: QwenOmniEngine | None = None
        self.downloader = YouTubeDownloader()
        self.audio_loader = AudioLoader()
        self.completed_file = config.output_dir / ".completed"

    def _get_completed(self) -> set[str]:
        if not self.completed_file.exists():
            return set()
        return set(self.completed_file.read_text().strip().split("\n"))

    def _mark_completed(self, title: str) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.completed_file, "a") as f:
            f.write(f"{title}\n")

    def _sanitize_filename(self, name: str) -> str:
        return re.sub(r'[<>:"/\\|?*]', "-", name).strip()

    def _create_output_dir(self, title: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = self._sanitize_filename(title)
        output_dir = self.config.output_dir / f"{timestamp}_{safe_title}"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _resolve_audio(self, source: str) -> tuple[Path, str | None]:
        if source.startswith(("http://", "https://")):
            result = self.downloader.download(source)
            return result.audio_path, result.title
        return Path(source), None

    def run(
        self,
        progress_callback=None,
        log_callback=None,
    ) -> dict[str, Path]:
        results = {}
        completed = self._get_completed() if self.config.resume else set()
        tests_to_run = [t for t in self.config.tests if (t.title or t.source) not in completed]

        if not tests_to_run:
            if log_callback:
                log_callback("All tests already completed.")
            return results

        if log_callback:
            log_callback(f"Loading model (this takes ~10 minutes)...")

        self.engine = QwenOmniEngine()
        self.engine.load_model()

        if log_callback:
            log_callback(f"Model loaded. Running {len(tests_to_run)} tests...")

        for i, test in enumerate(tests_to_run):
            if progress_callback:
                progress_callback(i + 1, len(tests_to_run), test.title or test.source)

            try:
                output_dir, title = self._run_single(test, log_callback)
                results[title] = output_dir
                self._mark_completed(title)
            except Exception as e:
                title = test.title or test.source
                if log_callback:
                    log_callback(f"FAILED: {title} - {e}")
                results[title] = None

        return results

    def _run_single(self, test: TestCase, log_callback=None) -> tuple[Path, str]:
        audio_path, extracted_title = self._resolve_audio(test.source)
        title = test.title or extracted_title or Path(test.source).stem

        if log_callback:
            log_callback(f"Processing: {title}")

        output_dir = self._create_output_dir(title)
        log_file = output_dir / "run.log"

        def log(msg: str):
            with open(log_file, "a") as f:
                f.write(f"{datetime.now().isoformat()} - {msg}\n")
            if log_callback:
                log_callback(f"  {msg}")

        log(f"Source: {test.source}")
        log(f"Title: {title}")
        log(f"Lyrics: {len(test.lyrics)} lines")
        log(f"Audio resolved: {audio_path}")

        audio_data = self.audio_loader.load(audio_path)
        log(f"Audio loaded: {audio_data.duration:.1f}s")

        sync_results = self.engine.sync_lyrics(audio_data, test.lyrics)
        log(f"Sync complete: {len(sync_results)} results")

        for fmt in test.formats:
            formatter = FormatterFactory.get_formatter(fmt)
            output_path = output_dir / f"sync{formatter.extension}"
            content = formatter.format(sync_results)
            output_path.write_text(content, encoding="utf-8")
            log(f"Output: {output_path.name}")

        log("Done!")
        return output_dir, title
