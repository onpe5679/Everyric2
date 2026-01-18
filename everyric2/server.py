"""FastAPI server for Everyric2 API."""

import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from everyric2 import __version__
from everyric2.config.settings import get_settings
from everyric2.inference.prompt import LyricLine
from everyric2.output.formatters import FormatterFactory

# Global engine instance (loaded on startup)
_engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global _engine

    # Startup: Load model
    settings = get_settings()

    # Only preload if not in debug mode (faster development)
    if not settings.debug:
        try:
            from everyric2.inference.qwen_omni import QwenOmniEngine

            _engine = QwenOmniEngine(settings.model)
            _engine.load_model()
            print(f"Model loaded: {settings.model.path}")
        except Exception as e:
            print(f"Warning: Failed to preload model: {e}")
            print("Model will be loaded on first request.")

    yield

    # Shutdown: Cleanup
    if _engine is not None:
        _engine.unload_model()


app = FastAPI(
    title="Everyric2 API",
    description="Lyrics synchronization using Qwen3-Omni multimodal LLM",
    version=__version__,
    lifespan=lifespan,
)

# CORS middleware for Chrome extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request/Response Models ---


class SyncRequest(BaseModel):
    """Request model for sync endpoint."""

    lyrics: str
    youtube_url: str | None = None
    format: str = "json"


class SyncResponse(BaseModel):
    """Response model for sync endpoint."""

    success: bool
    message: str
    results: list[dict[str, Any]] | None = None
    formatted: str | None = None


class HealthResponse(BaseModel):
    """Response model for health endpoint."""

    status: str
    version: str
    model_loaded: bool
    gpu_available: bool
    memory_usage: dict[str, Any] | None = None


class FormatsResponse(BaseModel):
    """Response model for formats endpoint."""

    formats: list[str]


# --- Helper Functions ---


def get_engine():
    """Get or create engine instance."""
    global _engine
    if _engine is None:
        from everyric2.inference.qwen_omni import QwenOmniEngine

        settings = get_settings()
        _engine = QwenOmniEngine(settings.model)
    return _engine


async def cleanup_temp_file(path: Path):
    """Background task to cleanup temp files."""
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


# --- API Endpoints ---


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    import torch

    engine = get_engine() if _engine else None

    response = HealthResponse(
        status="healthy",
        version=__version__,
        model_loaded=engine.is_loaded if engine else False,
        gpu_available=torch.cuda.is_available(),
        memory_usage=engine.get_memory_usage() if engine and engine.is_loaded else None,
    )
    return response


@app.get("/formats", response_model=FormatsResponse)
async def list_formats():
    """List supported output formats."""
    return FormatsResponse(formats=FormatterFactory.get_supported_formats())


@app.post("/sync/upload", response_model=SyncResponse)
async def sync_with_upload(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(..., description="Audio file"),
    lyrics: str = Form(..., description="Lyrics text"),
    format: str = Form("json", description="Output format"),
    separate_vocals: bool = Form(False, description="Use vocal separation"),
):
    """Synchronize lyrics with uploaded audio file.

    Args:
        audio: Audio file upload
        lyrics: Lyrics text (one line per lyric)
        format: Output format (srt, ass, lrc, json)
        separate_vocals: Whether to apply vocal separation

    Returns:
        Synchronized lyrics in requested format
    """
    # Validate format
    supported = FormatterFactory.get_supported_formats()
    if format.lower() not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {format}. Supported: {', '.join(supported)}",
        )

    # Save uploaded file to temp
    suffix = Path(audio.filename).suffix if audio.filename else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    # Schedule cleanup
    background_tasks.add_task(cleanup_temp_file, tmp_path)

    try:
        from everyric2.audio.loader import AudioLoader

        # Load audio
        loader = AudioLoader()
        audio_data = loader.load(tmp_path)

        # Vocal separation if requested
        if separate_vocals:
            try:
                from everyric2.audio.separator import VocalSeparator

                separator = VocalSeparator()
                if separator.is_available():
                    result = separator.separate(audio_data)
                    audio_data = result.vocals
            except Exception as e:
                # Continue without separation
                print(f"Vocal separation skipped: {e}")

        # Parse lyrics
        lyric_lines = LyricLine.from_text(lyrics)

        # Get engine and sync
        engine = get_engine()
        if not engine.is_loaded:
            engine.load_model()

        results = engine.sync_lyrics(audio_data, lyric_lines)

        # Format output
        formatter = FormatterFactory.get_formatter(format)
        formatted = formatter.format(results)

        return SyncResponse(
            success=True,
            message=f"Synchronized {len(results)} lines",
            results=[r.to_dict() for r in results],
            formatted=formatted,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sync/youtube", response_model=SyncResponse)
async def sync_with_youtube(
    request: SyncRequest,
    background_tasks: BackgroundTasks,
):
    """Synchronize lyrics with YouTube video.

    Args:
        request: Sync request with YouTube URL and lyrics

    Returns:
        Synchronized lyrics in requested format
    """
    if not request.youtube_url:
        raise HTTPException(status_code=400, detail="youtube_url is required")

    # Validate format
    supported = FormatterFactory.get_supported_formats()
    if request.format.lower() not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {request.format}. Supported: {', '.join(supported)}",
        )

    try:
        from everyric2.audio.downloader import YouTubeDownloader
        from everyric2.audio.loader import AudioLoader

        # Download audio
        downloader = YouTubeDownloader()
        if not downloader.validate_url(request.youtube_url):
            raise HTTPException(status_code=400, detail="Invalid YouTube URL")

        download_result = downloader.download(request.youtube_url)
        background_tasks.add_task(cleanup_temp_file, download_result.audio_path)

        # Load audio
        loader = AudioLoader()
        audio_data = loader.load(download_result.audio_path)

        # Parse lyrics
        lyric_lines = LyricLine.from_text(request.lyrics)

        # Get engine and sync
        engine = get_engine()
        if not engine.is_loaded:
            engine.load_model()

        results = engine.sync_lyrics(audio_data, lyric_lines)

        # Format output
        formatter = FormatterFactory.get_formatter(request.format)
        metadata = {"title": download_result.title}
        formatted = formatter.format(results, metadata)

        return SyncResponse(
            success=True,
            message=f"Synchronized {len(results)} lines from '{download_result.title}'",
            results=[r.to_dict() for r in results],
            formatted=formatted,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/youtube/info")
async def get_youtube_info(url: str):
    """Get YouTube video information.

    Args:
        url: YouTube video URL

    Returns:
        Video metadata (title, duration, etc.)
    """
    try:
        from everyric2.audio.downloader import YouTubeDownloader

        downloader = YouTubeDownloader()
        if not downloader.validate_url(url):
            raise HTTPException(status_code=400, detail="Invalid YouTube URL")

        info = downloader.get_video_info(url)
        return {
            "title": info.title,
            "duration": info.duration,
            "channel": info.channel,
            "url": info.url,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/model/load")
async def load_model():
    """Manually load the model."""
    try:
        engine = get_engine()
        if engine.is_loaded:
            return {"message": "Model already loaded"}

        engine.load_model()
        return {"message": "Model loaded successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/model/unload")
async def unload_model():
    """Unload the model to free memory."""
    global _engine
    if _engine is not None and _engine.is_loaded:
        _engine.unload_model()
        return {"message": "Model unloaded"}
    return {"message": "Model not loaded"}


# Error handlers
@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    """Handle uncaught exceptions."""
    return JSONResponse(
        status_code=500,
        content={"success": False, "message": str(exc)},
    )
