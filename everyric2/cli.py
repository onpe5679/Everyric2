"""Command-line interface for Everyric2."""

from pathlib import Path
from typing import Annotated, Optional

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
    model: Annotated[
        Optional[str],
        typer.Option("--model", "-m", help="Model path override"),
    ] = None,
    cache_dir: Annotated[
        Optional[Path],
        typer.Option("--cache-dir", help="HuggingFace cache directory"),
    ] = None,
) -> None:
    """Synchronize lyrics with audio.

    Examples:
        everyric2 sync "https://youtube.com/watch?v=..." lyrics.txt -o output.srt
        everyric2 sync song.mp3 lyrics.txt --separate -f ass
        everyric2 sync audio.wav lyrics.txt -f lrc --cache-dir /mnt/d/huggingface_cache
    """
    # Validate format
    supported_formats = FormatterFactory.get_supported_formats()
    if format.lower() not in supported_formats:
        console.print(
            f"[red]Error:[/red] Unsupported format '{format}'. "
            f"Supported: {', '.join(supported_formats)}"
        )
        raise typer.Exit(1)

    # Validate lyrics file
    if not lyrics.exists():
        console.print(f"[red]Error:[/red] Lyrics file not found: {lyrics}")
        raise typer.Exit(1)

    # Determine output path
    if output is None:
        formatter = FormatterFactory.get_formatter(format)
        output = lyrics.with_suffix(f".{formatter.get_extension()}")

    settings = get_settings()

    # Override settings if provided
    if model:
        settings.model.path = model
    if cache_dir:
        settings.model.cache_dir = cache_dir

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Step 1: Load/download audio
        task = progress.add_task("Loading audio...", total=None)

        try:
            from everyric2.audio.downloader import YouTubeDownloader
            from everyric2.audio.loader import AudioLoader

            loader = AudioLoader()
            audio_path: Path

            # Check if source is YouTube URL
            downloader = YouTubeDownloader()
            if downloader.validate_url(source):
                progress.update(task, description="Downloading from YouTube...")
                result = downloader.download(source)
                audio_path = result.audio_path
                console.print(f"[green]Downloaded:[/green] {result.title}")
            else:
                # Local file
                source_path = Path(source)
                if not source_path.exists():
                    console.print(f"[red]Error:[/red] Audio file not found: {source}")
                    raise typer.Exit(1)
                audio_path = source_path

            progress.update(task, description="Loading audio...")
            audio = loader.load(audio_path)
            console.print(f"[green]Audio loaded:[/green] {audio.duration:.1f}s")

        except Exception as e:
            console.print(f"[red]Error loading audio:[/red] {e}")
            raise typer.Exit(1)

        # Step 2: Vocal separation (optional)
        if separate:
            progress.update(task, description="Separating vocals...")
            try:
                from everyric2.audio.separator import VocalSeparator

                separator = VocalSeparator()
                if separator.is_available():
                    result = separator.separate(audio)
                    audio = result.vocals
                    console.print("[green]Vocal separation complete[/green]")
                else:
                    console.print(
                        "[yellow]Warning:[/yellow] Demucs not installed. Skipping vocal separation."
                    )
            except Exception as e:
                console.print(f"[yellow]Warning:[/yellow] Vocal separation failed: {e}")

        # Step 3: Load lyrics
        progress.update(task, description="Loading lyrics...")
        try:
            from everyric2.inference.prompt import LyricLine

            lyric_lines = LyricLine.from_file(lyrics)
            console.print(f"[green]Lyrics loaded:[/green] {len(lyric_lines)} lines")
        except Exception as e:
            console.print(f"[red]Error loading lyrics:[/red] {e}")
            raise typer.Exit(1)

        # Step 4: Synchronize
        progress.update(task, description="Loading model...")
        try:
            from everyric2.inference.qwen_omni import QwenOmniEngine

            engine = QwenOmniEngine(settings.model)
            engine.load_model()
            console.print("[green]Model loaded[/green]")

            def progress_callback(current: int, total: int) -> None:
                progress.update(task, description=f"Synchronizing... (chunk {current}/{total})")

            progress.update(task, description="Synchronizing lyrics...")
            results = engine.sync_lyrics(audio, lyric_lines, progress_callback=progress_callback)
            console.print(f"[green]Synchronized:[/green] {len(results)} lines")

        except Exception as e:
            console.print(f"[red]Error during synchronization:[/red] {e}")
            raise typer.Exit(1)

        # Step 5: Format and save
        progress.update(task, description="Saving output...")
        try:
            formatter = FormatterFactory.get_formatter(format)
            formatted = formatter.format(results)
            output.write_text(formatted, encoding="utf-8")
            console.print(f"[green]Saved:[/green] {output}")

        except Exception as e:
            console.print(f"[red]Error saving output:[/red] {e}")
            raise typer.Exit(1)

        progress.update(task, description="Done!")


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
