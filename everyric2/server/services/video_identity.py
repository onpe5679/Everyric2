"""YouTube video_id → 제목/채널의 짧고 실패 허용인 서버측 복구 경로."""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass

import requests

from everyric2.server import title_match

_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_HYPHEN_RE = re.compile(r"\s(?:-|–|—)\s")
_FEAT_RE = re.compile(r"(?:^|\s)(?:feat|ft)\.?\s*\S.*$", re.IGNORECASE)
_CACHE_TTL_SEC = 10 * 60
_NEGATIVE_TTL_SEC = 60
_CACHE_MAX = 512


@dataclass(frozen=True)
class VideoIdentity:
    title: str
    channel: str | None = None


_cache: dict[str, tuple[float, VideoIdentity | None]] = {}
_cache_lock = threading.Lock()


def fetch_video_identity(video_id: str) -> VideoIdentity | None:
    """YouTube oEmbed에서 공개 제목/채널을 읽는다. 실패는 캐시된 ``None``으로 닫는다."""
    if not _VIDEO_ID_RE.fullmatch(video_id):
        return None
    now = time.monotonic()
    with _cache_lock:
        cached = _cache.get(video_id)
        if cached and cached[0] > now:
            return cached[1]

    identity: VideoIdentity | None = None
    try:
        response = requests.get(
            "https://www.youtube.com/oembed",
            params={
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "format": "json",
            },
            headers={"User-Agent": "everyric2-video-identity/1.0"},
            timeout=2.0,
        )
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict):
            raw_title = data.get("title")
            raw_channel = data.get("author_name")
            if isinstance(raw_title, str) and raw_title.strip():
                identity = VideoIdentity(
                    title=raw_title.strip()[:300],
                    channel=(
                        raw_channel.strip()[:128]
                        if isinstance(raw_channel, str) and raw_channel.strip()
                        else None
                    ),
                )
    except (requests.RequestException, ValueError):
        identity = None

    ttl = _CACHE_TTL_SEC if identity is not None else _NEGATIVE_TTL_SEC
    with _cache_lock:
        for key, (expires_at, _) in list(_cache.items()):
            if expires_at <= now:
                _cache.pop(key, None)
        while len(_cache) >= _CACHE_MAX:
            _cache.pop(next(iter(_cache)))
        _cache[video_id] = (now + ttl, identity)
    return identity


def ordered_title_candidates(identity: VideoIdentity) -> list[str]:
    """공개 채널이 하이픈 어느 쪽과 같은지로 제목 방향을 정하고 대안을 보존한다."""
    raw = identity.title.strip()
    channel = (identity.channel or "").removesuffix(" - Topic").strip()
    split = _HYPHEN_RE.search(raw)
    if split and channel:
        left = raw[: split.start()].strip()
        right = raw[split.end() :].strip()
        right_title = _FEAT_RE.sub("", right).strip() or right
        channel_key = title_match.normalize_title(channel)
        left_key = title_match.normalize_title(left)
        right_key = title_match.normalize_title(right_title)
        if channel_key == right_key and channel_key != left_key:
            return [left]
        if channel_key == left_key and channel_key != right_key:
            return [right]
    return [raw]
