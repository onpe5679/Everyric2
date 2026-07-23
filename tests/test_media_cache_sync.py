"""fetch_cached_audio_sync — 링크 검증의 캐시 우선 오디오 조달 (네트워크/ffmpeg 전부 모킹).

같은 호스트 워커가 동기 컨텍스트에서 캐시를 조달하고, 미설정/미스/추출 실패는 전부 None으로
떨어져 호출부(yt-dlp)가 이어받는 폴백 계약을 못 박는다.
"""

from everyric2.config.settings import get_settings
from everyric2.server import media_cache


def _set_url(url: str) -> None:
    object.__setattr__(get_settings().server, "media_cache_url", url)


def test_unset_url_returns_none():
    _set_url("")
    assert media_cache.fetch_cached_audio_sync("vid00000001", "t") is None


def test_hit_extracts_and_returns_path(monkeypatch, tmp_path):
    _set_url("http://cache.test")
    try:
        src = tmp_path / "vid.mp4"
        src.write_bytes(b"x")
        monkeypatch.setattr(
            media_cache, "_lookup", lambda url, key, vid: {"found": True, "path": str(src)}
        )
        monkeypatch.setattr(media_cache, "_run_ffmpeg", lambda s, d: True)
        out = media_cache.fetch_cached_audio_sync("vid00000001", "t")
        assert out is not None
        assert out.endswith(".m4a")
    finally:
        _set_url("")


def test_miss_returns_none(monkeypatch):
    _set_url("http://cache.test")
    try:
        monkeypatch.setattr(media_cache, "_lookup", lambda url, key, vid: {"found": False})
        assert media_cache.fetch_cached_audio_sync("vid00000001", "t") is None
    finally:
        _set_url("")


def test_extract_failure_returns_none(monkeypatch, tmp_path):
    _set_url("http://cache.test")
    try:
        src = tmp_path / "vid.mp4"
        src.write_bytes(b"x")
        monkeypatch.setattr(
            media_cache, "_lookup", lambda url, key, vid: {"found": True, "path": str(src)}
        )
        monkeypatch.setattr(media_cache, "_run_ffmpeg", lambda s, d: False)
        assert media_cache.fetch_cached_audio_sync("vid00000001", "t") is None
    finally:
        _set_url("")
