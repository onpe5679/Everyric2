"""Audio processing module."""

from everyric2.audio.downloader import DownloadResult, YouTubeDownloader
from everyric2.audio.loader import AudioChunk, AudioData, AudioLoader

__all__ = ["AudioLoader", "AudioData", "AudioChunk", "YouTubeDownloader", "DownloadResult"]
