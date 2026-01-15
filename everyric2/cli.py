"""Command-line interface for Everyric2."""

# Monkey-patch torch.load to fix weights_only=True issue with pyannote/whisperx
try:
    import torch

    _original_load = torch.load

    def _patched_load(*args, **kwargs):
        if "weights_only" in kwargs:
            kwargs["weights_only"] = False
        return _original_load(*args, **kwargs)

    torch.load = _patched_load
except ImportError:
    pass

from pathlib import Path
from typing import Annotated, Optional, cast, Literal

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from everyric2 import __version__
from everyric2.config.settings import get_settings
from everyric2.output.formatters import FormatterFactory

app = typer.Typer(
    name="everyric2",
    help="Lyrics synchronization using Qwen3-Omni multimodal LLM",
    no_args_is_help=True,
)
console = Console()


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"everyric2 version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", "-v", callback=version_callback, is_eager=True),
    ] = False,
) -> None:
    """Everyric2 - Lyrics synchronization using Qwen3-Omni."""
    pass


@app.command()
def sync(
    source: Annotated[str, typer.Argument(help="YouTube URL or local audio file path")],
    lyrics: Annotated[Path, typer.Argument(help="Path to lyrics text file")],
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output file path"),
    ] = None,
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format (srt, ass, lrc, json)"),
    ] = "srt",
    separate: Annotated[
        bool,
        typer.Option("--separate", "-s", help="Use Demucs vocal separation"),
    ] = False,
    translate: Annotated[
        bool,
        typer.Option("--translate", "-t", help="Translate lyrics to Korean"),
    ] = False,
    debug: Annotated[
        bool,
        typer.Option("--debug", "-d", help="Save debug files (prompts, responses, diagnostics)"),
    ] = False,
    engine: Annotated[
        str,
        typer.Option("--engine", "-e", help="Alignment engine (whisperx, mfa, hybrid, qwen)"),
    ] = "hybrid",
    language: Annotated[
        str,
        typer.Option("--language", "-l", help="Language (auto, en, ja, ko)"),
    ] = "auto",
    model: Annotated[
        Optional[str],
        typer.Option("--model", "-m", help="Model path override (for qwen engine)"),
    ] = None,
    cache_dir: Annotated[
        Optional[Path],
        typer.Option("--cache-dir", help="HuggingFace cache directory"),
    ] = None,
    chunk_duration: Annotated[
        int | None,
        typer.Option("--chunk-duration", "-c", help="Audio chunk duration in seconds (qwen only)"),
    ] = None,
) -> None:
    """Synchronize lyrics with audio.

    Examples:
        everyric2 sync song.mp3 lyrics.txt -o output.srt
        everyric2 sync song.mp3 lyrics.txt --translate --debug
        everyric2 sync "https://youtube.com/..." lyrics.txt -f lrc
    """
    import json
    import shutil
    import sys

    supported_formats = FormatterFactory.get_supported_formats()
    if format.lower() not in supported_formats:
        console.print(
            f"[red]Error:[/red] Unsupported format '{format}'. "
            f"Supported: {', '.join(supported_formats)}"
        )
        raise typer.Exit(1)

    if not lyrics.exists():
        console.print(f"[red]Error:[/red] Lyrics file not found: {lyrics}")
        raise typer.Exit(1)

    settings = get_settings()
    if model:
        settings.model.path = model
    if cache_dir:
        settings.model.cache_dir = cache_dir

    video_title: str | None = None
    audio_path: Path
    audio = None
    original_audio = None
    vocals_audio = None
    lyric_lines = None
    translated_text: str | None = None
    translated_results: list | None = None
    results = None
    debug_info = None
    run_ctx = None
    output_manager = None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading audio...", total=None)

        try:
            from everyric2.audio.downloader import YouTubeDownloader
            from everyric2.audio.loader import AudioLoader
            import time as time_module

            step_start = time_module.time()
            loader = AudioLoader()
            downloader = YouTubeDownloader()

            if downloader.validate_url(source):
                progress.update(task, description="Downloading from YouTube...")
                dl_result = downloader.download(source)
                audio_path = dl_result.audio_path
                video_title = dl_result.title
                console.print(f"[green]Downloaded:[/green] {video_title}")
            else:
                source_path = Path(source)
                if not source_path.exists():
                    console.print(f"[red]Error:[/red] Audio file not found: {source}")
                    raise typer.Exit(1)
                audio_path = source_path

            progress.update(task, description="Loading audio...")
            audio = loader.load(audio_path)
            original_audio = audio
            audio_load_time = time_module.time() - step_start
            console.print(
                f"[green]Audio loaded:[/green] {audio.duration:.1f}s ({audio_load_time:.1f}s)"
            )

        except Exception as e:
            console.print(f"[red]Error loading audio:[/red] {e}")
            raise typer.Exit(1)

        if debug:
            from everyric2.debug.output_manager import OutputManager
            from everyric2.debug.debug_info import DebugInfo, StepTiming

            output_manager = OutputManager()
            command_str = " ".join(sys.argv)
            run_ctx = output_manager.create_run_context(
                title=video_title,
                source=source,
                command=command_str,
                settings=settings.model_dump() if hasattr(settings, "model_dump") else {},
            )
            debug_info = DebugInfo(
                source=source,
                title=video_title,
                command=command_str,
                settings=settings.model_dump() if hasattr(settings, "model_dump") else {},
                output_dir=run_ctx.output_dir,
            )
            debug_info.audio_duration = audio.duration
            output_manager.save_settings(run_ctx)
            output_manager.save_audio(run_ctx, audio, "audio_original.wav")
            debug_info.steps.append(
                StepTiming(
                    name="audio_load", start_time=step_start, end_time=step_start + audio_load_time
                )
            )
            console.print(f"[cyan]Debug output:[/cyan] {run_ctx.output_dir}")

        if separate:
            progress.update(task, description="Separating vocals...")
            sep_start = time_module.time()
            try:
                from everyric2.audio.separator import VocalSeparator

                separator = VocalSeparator()
                if separator.is_available():
                    sep_result = separator.separate(audio)
                    vocals_audio = sep_result.vocals
                    audio = vocals_audio
                    sep_time = time_module.time() - sep_start
                    console.print(f"[green]Vocal separation complete[/green] ({sep_time:.1f}s)")
                    if debug and run_ctx and output_manager:
                        output_manager.save_audio(run_ctx, vocals_audio, "audio_vocals.wav")
                        if debug_info:
                            from everyric2.debug.debug_info import StepTiming

                            debug_info.steps.append(
                                StepTiming("vocal_separation", sep_start, time_module.time())
                            )
                else:
                    console.print("[yellow]Warning:[/yellow] Demucs not installed.")
            except Exception as e:
                console.print(f"[yellow]Warning:[/yellow] Vocal separation failed: {e}")

        progress.update(task, description="Loading lyrics...")
        try:
            from everyric2.inference.prompt import LyricLine

            lyric_lines = LyricLine.from_file(lyrics)
            lyrics_text = lyrics.read_text(encoding="utf-8")
            console.print(f"[green]Lyrics loaded:[/green] {len(lyric_lines)} lines")

            if debug and run_ctx and output_manager:
                output_manager.save_lyrics(run_ctx, lyrics_text, "lyrics_original.txt")
                if debug_info:
                    debug_info.original_lyrics = lyrics_text

        except Exception as e:
            console.print(f"[red]Error loading lyrics:[/red] {e}")
            raise typer.Exit(1)

        if translate:
            progress.update(task, description="Translating lyrics to Korean...")
            trans_start = time_module.time()
            try:
                from everyric2.translation.translator import LyricsTranslator

                translator = LyricsTranslator()
                translated_text = translator.translate(lyric_lines)
                trans_time = time_module.time() - trans_start
                console.print(f"[green]Translation complete[/green] ({trans_time:.1f}s)")

                if debug and run_ctx and output_manager:
                    output_manager.save_translated_lyrics(run_ctx, translated_text)
                    if debug_info:
                        debug_info.translated_lyrics = translated_text
                        from everyric2.debug.debug_info import StepTiming

                        debug_info.steps.append(
                            StepTiming("translation", trans_start, time_module.time())
                        )

            except Exception as e:
                console.print(f"[yellow]Warning:[/yellow] Translation failed: {e}")

        progress.update(task, description=f"Loading {engine} engine...")
        model_start = time_module.time()
        try:
            from everyric2.alignment.factory import EngineFactory

            alignment_engine = EngineFactory.get_engine(engine, settings.alignment)
            if not alignment_engine.is_available():
                console.print(f"[red]Error:[/red] Engine '{engine}' is not available.")
                console.print(
                    "[yellow]Hint:[/yellow] Try 'everyric2 engines' to see available engines."
                )
                raise typer.Exit(1)

            model_time = time_module.time() - model_start
            console.print(f"[green]Engine ready:[/green] {engine} ({model_time:.1f}s)")

            if debug_info:
                from everyric2.debug.debug_info import StepTiming

                debug_info.steps.append(StepTiming("engine_load", model_start, time_module.time()))

            def progress_callback(current: int, total: int) -> None:
                if hasattr(alignment_engine, "get_status_string"):
                    status = alignment_engine.get_status_string()
                    progress.update(task, description=f"Synchronizing... [{status}]")
                else:
                    progress.update(task, description=f"Synchronizing... (step {current}/{total})")

            progress.update(task, description="Synchronizing lyrics...")
            sync_start = time_module.time()
            results = alignment_engine.align(
                audio, lyric_lines, language=language, progress_callback=progress_callback
            )
            sync_time = time_module.time() - sync_start
            console.print(f"[green]Synchronized:[/green] {len(results)} lines ({sync_time:.1f}s)")

            if debug_info:
                from everyric2.debug.debug_info import StepTiming

                debug_info.steps.append(StepTiming("sync", sync_start, time_module.time()))

                if hasattr(alignment_engine, "get_transcription_sets"):
                    sets = alignment_engine.get_transcription_sets()
                    for words, stats, engine_name in sets:
                        if words:
                            debug_info.add_transcription_data(words, stats, engine_name)
                elif hasattr(alignment_engine, "get_last_transcription_data"):
                    words, stats, engine_name = alignment_engine.get_last_transcription_data()
                    if words:
                        debug_info.add_transcription_data(words, stats, engine_name)
                elif hasattr(alignment_engine, "get_last_transcription_data"):
                    words, stats, engine_name = alignment_engine.get_last_transcription_data()
                    if words:
                        debug_info.add_transcription_data(words, stats, engine_name)

        except Exception as e:
            console.print(f"[red]Error during synchronization:[/red] {e}")
            raise typer.Exit(1)

        progress.update(task, description="Saving output...")
        try:
            formatter = FormatterFactory.get_formatter(format)
            formatted = formatter.format(results)

            if debug and run_ctx:
                final_output = run_ctx.output_dir / f"output.{format}"
                final_output.write_text(formatted, encoding="utf-8")

                if translated_text and results:
                    from everyric2.inference.prompt import SyncResult

                    translated_lines = [
                        line for line in translated_text.strip().split("\n") if line.strip()
                    ]
                    translated_results = []
                    for i, result in enumerate(results):
                        trans_text = (
                            translated_lines[i] if i < len(translated_lines) else result.text
                        )
                        translated_results.append(
                            SyncResult(
                                text=trans_text,
                                start_time=result.start_time,
                                end_time=result.end_time,
                            )
                        )
                    translated_formatted = formatter.format(translated_results)
                    translated_output = run_ctx.output_dir / f"output_translated.{format}"
                    translated_output.write_text(translated_formatted, encoding="utf-8")
                    console.print(f"[green]Translated output:[/green] {translated_output}")

                if debug_info:
                    debug_info.finalize(results)
                    debug_json = run_ctx.output_dir / "debug_info.json"
                    debug_json.write_text(
                        json.dumps(debug_info.to_dict(), indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )

                if debug_info:
                    progress.update(task, description="Generating diagnostics...")
                    try:
                        from everyric2.debug.visualizer import DiagnosticsVisualizer

                        visualizer = DiagnosticsVisualizer()
                        diag_path = run_ctx.output_dir / "diagnostics.png"
                        visualizer.create_diagnostics(
                            debug_info,
                            results,
                            diag_path,
                            audio_waveform=original_audio.waveform
                            if original_audio
                            else audio.waveform,
                            vocals_waveform=vocals_audio.waveform if vocals_audio else None,
                            translated_results=translated_results,
                            sample_rate=audio.sample_rate,
                        )
                        console.print(f"[green]Diagnostics:[/green] {diag_path}")
                    except Exception as e:
                        console.print(f"[yellow]Warning:[/yellow] Diagnostics failed: {e}")

                console.print(f"[green]Saved:[/green] {final_output}")
            else:
                if output is None:
                    output = lyrics.with_suffix(f".{formatter.get_extension()}")
                output.write_text(formatted, encoding="utf-8")
                console.print(f"[green]Saved:[/green] {output}")

        except Exception as e:
            console.print(f"[red]Error saving output:[/red] {e}")
            raise typer.Exit(1)

        progress.update(task, description="Done!")


@app.command()
def batch(
    config_file: Annotated[Path, typer.Argument(help="YAML config file for batch tests")],
    resume: Annotated[
        bool,
        typer.Option("--resume", "-r", help="Resume from last completed test"),
    ] = False,
) -> None:
    """Run batch tests from YAML config.

    Example config (batch.yaml):
        output_dir: ./output
        formats: [srt, ass, lrc, json]
        tests:
          - title: "Song Name"
            source: "https://youtube.com/watch?v=..."
            lyrics: |
              First line
              Second line
          - title: "Another Song"
            source: "./song.mp3"
            lyrics_file: "./lyrics.txt"

    Usage:
        everyric2 batch batch.yaml
        everyric2 batch batch.yaml --resume
    """
    if not config_file.exists():
        console.print(f"[red]Error:[/red] Config file not found: {config_file}")
        raise typer.Exit(1)

    try:
        from everyric2.batch import BatchConfig, BatchRunner

        config = BatchConfig.from_yaml(config_file)
        config.resume = resume
        runner = BatchRunner(config)

        console.print(f"[cyan]Batch config loaded:[/cyan] {len(config.tests)} tests")
        console.print(f"[cyan]Output directory:[/cyan] {config.output_dir}")
        console.print(f"[cyan]Formats:[/cyan] {', '.join(config.formats)}")

        def progress_cb(current: int, total: int, title: str) -> None:
            console.print(f"\n[cyan][{current}/{total}][/cyan] {title}")

        def log_cb(msg: str) -> None:
            console.print(f"  {msg}")

        results = runner.run(progress_callback=progress_cb, log_callback=log_cb)

        console.print("\n[green]Batch complete![/green]")
        for title, path in results.items():
            if path:
                console.print(f"  [green]✓[/green] {title} → {path}")
            else:
                console.print(f"  [red]✗[/red] {title} (failed)")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def serve(
    host: Annotated[
        str,
        typer.Option("--host", "-h", help="Server host"),
    ] = "0.0.0.0",
    port: Annotated[
        int,
        typer.Option("--port", "-p", help="Server port"),
    ] = 8000,
    reload: Annotated[
        bool,
        typer.Option("--reload", "-r", help="Enable auto-reload"),
    ] = False,
    workers: Annotated[
        int,
        typer.Option("--workers", "-w", help="Number of workers"),
    ] = 1,
) -> None:
    """Start the API server.

    Example:
        everyric2 serve --port 8000
    """
    try:
        import uvicorn

        console.print(f"[green]Starting server on {host}:{port}[/green]")
        uvicorn.run(
            "everyric2.server:app",
            host=host,
            port=port,
            reload=reload,
            workers=workers if not reload else 1,
        )
    except ImportError:
        console.print(
            "[red]Error:[/red] uvicorn not installed. Install with: pip install uvicorn[standard]"
        )
        raise typer.Exit(1)


@app.command()
def engines() -> None:
    """List available alignment engines."""
    from everyric2.alignment.factory import EngineFactory

    table = Table(title="Available Alignment Engines")
    table.add_column("Engine", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Description")

    for eng in EngineFactory.get_available_engines():
        status = "[green]✓ Available[/green]" if eng["available"] else "[red]✗ Not installed[/red]"
        table.add_row(eng["type"], status, eng["description"])

    console.print(table)
    console.print(
        "\n[dim]Use --engine/-e option to select: everyric2 sync song.mp3 lyrics.txt -e whisperx[/dim]"
    )


@app.command()
def transcribe(
    source: Annotated[str, typer.Argument(help="YouTube URL or local audio file path")],
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output file path for transcribed lyrics"),
    ] = None,
    separate: Annotated[
        bool,
        typer.Option("--separate", "-s", help="Use Demucs vocal separation"),
    ] = False,
    language: Annotated[
        str,
        typer.Option("--language", "-l", help="Language (auto, en, ja, ko)"),
    ] = "auto",
) -> None:
    """Transcribe audio to text (no lyrics file needed)."""
    from everyric2.alignment.factory import EngineFactory
    from everyric2.audio.loader import AudioLoader

    settings = get_settings()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading audio...", total=None)

        try:
            from everyric2.audio.downloader import YouTubeDownloader

            loader = AudioLoader()
            downloader = YouTubeDownloader()

            if downloader.validate_url(source):
                progress.update(task, description="Downloading from YouTube...")
                dl_result = downloader.download(source)
                audio_path = dl_result.audio_path
                console.print(f"[green]Downloaded:[/green] {dl_result.title}")
            else:
                audio_path = Path(source)
                if not audio_path.exists():
                    console.print(f"[red]Error:[/red] Audio file not found: {source}")
                    raise typer.Exit(1)

            audio = loader.load(audio_path)
            console.print(f"[green]Audio loaded:[/green] {audio.duration:.1f}s")

        except Exception as e:
            console.print(f"[red]Error loading audio:[/red] {e}")
            raise typer.Exit(1)

        if separate:
            progress.update(task, description="Separating vocals...")
            try:
                from everyric2.audio.separator import VocalSeparator

                separator = VocalSeparator()
                if separator.is_available():
                    sep_result = separator.separate(audio)
                    audio = sep_result.vocals
                    console.print("[green]Vocal separation complete[/green]")
                else:
                    console.print("[yellow]Warning:[/yellow] Demucs not installed.")
            except Exception as e:
                console.print(f"[yellow]Warning:[/yellow] Vocal separation failed: {e}")

        progress.update(task, description="Transcribing...")
        try:
            engine = EngineFactory.get_engine("whisperx", settings.alignment)
            if not engine.is_available():
                console.print("[red]Error:[/red] WhisperX not available for transcription.")
                raise typer.Exit(1)

            result = engine.transcribe(audio, language=language)
            console.print(
                f"[green]Transcribed:[/green] {len(result.words)} words, language: {result.language}"
            )

            output_text = result.text
            if output:
                output.write_text(output_text, encoding="utf-8")
                console.print(f"[green]Saved:[/green] {output}")
            else:
                console.print("\n[cyan]Transcription:[/cyan]")
                console.print(output_text)

        except Exception as e:
            console.print(f"[red]Error during transcription:[/red] {e}")
            raise typer.Exit(1)


@app.command()
def formats() -> None:
    """List supported output formats."""
    table = Table(title="Supported Output Formats")
    table.add_column("Format", style="cyan")
    table.add_column("Extension", style="green")
    table.add_column("Description")

    formats_info = [
        ("srt", "SubRip subtitle format - widely supported"),
        ("ass", "Advanced SubStation Alpha - supports styling"),
        ("lrc", "LRC lyrics format - common for music players"),
        ("json", "JSON format - for programmatic access"),
    ]

    for fmt, desc in formats_info:
        formatter = FormatterFactory.get_formatter(fmt)
        table.add_row(fmt, f".{formatter.get_extension()}", desc)

    console.print(table)


@app.command()
def info(
    source: Annotated[str, typer.Argument(help="YouTube URL or local audio file")],
) -> None:
    """Show audio/video information."""
    from everyric2.audio.downloader import YouTubeDownloader
    from everyric2.audio.loader import AudioLoader

    downloader = YouTubeDownloader()

    if downloader.validate_url(source):
        # YouTube URL
        try:
            info = downloader.get_video_info(source)
            table = Table(title="Video Information")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Title", info.title)
            table.add_row("Duration", f"{info.duration:.1f}s ({info.duration / 60:.1f}m)")
            table.add_row("Channel", info.channel or "N/A")
            table.add_row("URL", info.url)

            console.print(table)

        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)
    else:
        # Local file
        try:
            loader = AudioLoader()
            path = Path(source)
            if not path.exists():
                console.print(f"[red]Error:[/red] File not found: {source}")
                raise typer.Exit(1)

            duration = loader.get_duration(path)

            table = Table(title="Audio Information")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("File", path.name)
            table.add_row("Path", str(path.absolute()))
            table.add_row("Duration", f"{duration:.1f}s ({duration / 60:.1f}m)")
            table.add_row("Format", path.suffix)

            console.print(table)

        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)


@app.command()
def config() -> None:
    """Show current configuration."""
    settings = get_settings()

    table = Table(title="Current Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    # Model settings
    table.add_row("Model Path", settings.model.path)
    table.add_row(
        "Cache Dir", str(settings.model.cache_dir) if settings.model.cache_dir else "Default"
    )
    table.add_row("Device Map", settings.model.device_map)
    table.add_row("Torch Dtype", settings.model.torch_dtype)
    table.add_row("Flash Attention", str(settings.model.use_flash_attention))
    table.add_row("Max Audio Duration", f"{settings.model.max_audio_duration}s")
    table.add_row("Chunk Duration", f"{settings.model.chunk_duration}s")

    # Audio settings
    table.add_row("Sample Rate", f"{settings.audio.target_sample_rate}Hz")
    table.add_row("Demucs Model", settings.audio.demucs_model)
    table.add_row("Temp Dir", str(settings.audio.temp_dir))

    # Output settings
    table.add_row("Default Format", settings.output.default_format)

    # Server settings
    table.add_row("Server Host", settings.server.host)
    table.add_row("Server Port", str(settings.server.port))

    console.print(table)

    console.print("\n[dim]Set environment variables with EVERYRIC_ prefix to override.[/dim]")
    console.print("[dim]Example: EVERYRIC_MODEL__CACHE_DIR=/mnt/d/huggingface_cache[/dim]")


if __name__ == "__main__":
    app()
