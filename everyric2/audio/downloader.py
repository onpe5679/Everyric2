"""YouTube audio downloader using yt-dlp."""

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from everyric2.config.settings import AudioSettings, get_settings


class DownloadError(Exception):
    """Base exception for download operations."""

    pass


class InvalidURLError(DownloadError):
    """Raised when URL is invalid."""

    pass


class VideoUnavailableError(DownloadError):
    """Raised when video is unavailable."""

    pass


class DependencyError(DownloadError):
    """Raised when required dependency is missing."""

    pass


@dataclass
class VideoInfo:
    """YouTube video information."""

    title: str
    duration: float  # seconds
    url: str
    channel: str | None = None
    upload_date: str | None = None


@dataclass
class DownloadResult:
    """Result of audio download."""

    audio_path: Path
    title: str
    duration: float
    url: str


class YouTubeDownloader:
    """Download audio from YouTube using yt-dlp."""

    # YouTube URL patterns
    YOUTUBE_PATTERNS = [
        r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})",
        r"(?:https?://)?(?:www\.)?youtu\.be/([a-zA-Z0-9_-]{11})",
        r"(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})",
        r"(?:https?://)?(?:www\.)?youtube\.com/v/([a-zA-Z0-9_-]{11})",
        r"(?:https?://)?(?:music\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})",
    ]

    def __init__(self, config: AudioSettings | None = None):
        """Initialize downloader.

        Args:
            config: Audio settings. If None, uses global settings.
        """
        self.config = config or get_settings().audio
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """Check if required dependencies are available."""
        if not shutil.which("ffmpeg"):
            raise DependencyError(
                "ffmpeg is required but not found. Install it with: sudo apt install ffmpeg"
            )

    def validate_url(self, url: str) -> bool:
        """Check if URL is a valid YouTube URL.

        Args:
            url: URL to validate.

        Returns:
            True if valid YouTube URL.
        """
        return any(re.match(pattern, url) for pattern in self.YOUTUBE_PATTERNS)

    def extract_video_id(self, url: str) -> str | None:
        """Extract video ID from YouTube URL.

        Args:
            url: YouTube URL.

        Returns:
            Video ID or None if not found.
        """
        for pattern in self.YOUTUBE_PATTERNS:
            match = re.match(pattern, url)
            if match:
                return match.group(1)
        return None

    def get_video_info(self, url: str) -> VideoInfo:
        """Get video information without downloading.

        Args:
            url: YouTube URL.

        Returns:
            VideoInfo instance.

        Raises:
            InvalidURLError: If URL is invalid.
            VideoUnavailableError: If video is unavailable.
        """
        if not self.validate_url(url):
            raise InvalidURLError(f"Invalid YouTube URL: {url}")

        try:
            import yt_dlp

            ydl_opts: dict[str, Any] = {
                "quiet": True,
                "no_warnings": True,
                "extract_flat": False,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                if info is None:
                    raise VideoUnavailableError(f"Could not extract info: {url}")

                return VideoInfo(
                    title=info.get("title", "Unknown"),
                    duration=float(info.get("duration", 0)),
                    url=url,
                    channel=info.get("channel"),
                    upload_date=info.get("upload_date"),
                )

        except Exception as e:
            if "unavailable" in str(e).lower() or "private" in str(e).lower():
                raise VideoUnavailableError(f"Video unavailable: {url}") from e
            raise DownloadError(f"Failed to get video info: {e}") from e

    def download(
        self,
        url: str,
        output_dir: Path | None = None,
        filename: str | None = None,
    ) -> DownloadResult:
        """Download audio from YouTube URL.

        Args:
            url: YouTube URL.
            output_dir: Output directory. Defaults to temp dir.
            filename: Output filename (without extension). Defaults to video title.

        Returns:
            DownloadResult with path to downloaded audio.

        Raises:
            InvalidURLError: If URL is invalid.
            VideoUnavailableError: If video is unavailable.
            DownloadError: If download fails.
        """
        if not self.validate_url(url):
            raise InvalidURLError(f"Invalid YouTube URL: {url}")

        output_dir = output_dir or self.config.temp_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            import yt_dlp

            # Template for output filename
            if filename:
                outtmpl = str(output_dir / f"{filename}.%(ext)s")
            else:
                outtmpl = str(output_dir / "%(title)s.%(ext)s")

            ydl_opts: dict[str, Any] = {
                "format": "bestaudio/best",
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "wav",
                        "preferredquality": "192",
                    }
                ],
                "outtmpl": outtmpl,
                "quiet": True,
                "no_warnings": True,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                if info is None:
                    raise DownloadError(f"Failed to download: {url}")

                title = info.get("title", "Unknown")
                duration = float(info.get("duration", 0))

                # Find the downloaded file
                if filename:
                    audio_path = output_dir / f"{filename}.wav"
                else:
                    # Sanitize title for filename
                    safe_title = yt_dlp.utils.sanitize_filename(title)
                    audio_path = output_dir / f"{safe_title}.wav"

                if not audio_path.exists():
                    # Try to find any wav file in output dir
                    wav_files = list(output_dir.glob("*.wav"))
                    if wav_files:
                        audio_path = wav_files[0]
                    else:
                        raise DownloadError(f"Downloaded file not found: {audio_path}")

                return DownloadResult(
                    audio_path=audio_path,
                    title=title,
                    duration=duration,
                    url=url,
                )

        except yt_dlp.utils.DownloadError as e:
            if "unavailable" in str(e).lower() or "private" in str(e).lower():
                raise VideoUnavailableError(f"Video unavailable: {url}") from e
            raise DownloadError(f"Download failed: {e}") from e
        except Exception as e:
            raise DownloadError(f"Download failed: {e}") from e

    def cleanup(self, result: DownloadResult) -> None:
        """Clean up downloaded file.

        Args:
            result: Download result to clean up.
        """
        if result.audio_path.exists():
            result.audio_path.unlink()
