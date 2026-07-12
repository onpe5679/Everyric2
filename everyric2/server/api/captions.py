"""유튜브 자막 조회 API — 확장의 '자막에서 가사 가져오기' 소스.

워치 페이지에서 긁은 timedtext URL은 POT(proof-of-origin) 강제로 브라우저 플레이어
밖에서는 200 + 빈 본문을 돌려준다 (2026-07 실측). yt-dlp는 클라이언트 선택/POT을
내부에서 처리하고 계속 업데이트되므로, 자막은 서버가 yt-dlp로 받아 확장에 넘긴다.
"""

import json
import logging
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/captions", tags=["captions"])

# 경로 파라미터를 URL·파일 glob에 그대로 쓰므로 형식을 강제한다 (쿼리 인젝션 차단)
_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_LANG_RE = re.compile(r"^[A-Za-z0-9-]{1,16}$")


def _validate(video_id: str, lang: str | None = None) -> None:
    if not _VIDEO_ID_RE.match(video_id):
        raise HTTPException(status_code=422, detail="invalid video_id")
    if lang is not None and not _LANG_RE.match(lang):
        raise HTTPException(status_code=422, detail="invalid lang")

# 자동 생성 자막은 번역 대상 언어 ~150개가 전부 나열된다 — 원어 트랙('-orig')과
# 사용자가 실제로 쓸 번역만 노출한다
_AUTO_LANG_ALLOW = {"ja", "ko", "en"}


def _ydl_opts(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """다운로더와 동일한 쿠키/EJS/회선 옵션을 공유하는 yt-dlp 옵션."""
    from everyric2.audio.downloader import YouTubeDownloader

    opts: dict[str, Any] = {"quiet": True, "no_warnings": True, "skip_download": True}
    dl = YouTubeDownloader()
    dl._add_cookie_options(opts)
    dl._add_ejs_options(opts)
    dl._add_network_options(opts)
    if extra:
        opts.update(extra)
    return opts


def _track_label(entries: list[dict[str, Any]], lang: str, auto: bool) -> str:
    name = next((e.get("name") for e in entries if e.get("name")), None) or lang
    return f"{name} (자동 생성)" if auto else str(name)


def json3_events_to_lines(data: dict[str, Any]) -> list[dict[str, Any]]:
    """timedtext json3 → [{start, end, text}] — 빈 줄/음표만 있는 줄 제거,
    연속 중복(롤링 자막)은 첫 줄의 end를 늘려 병합한다."""
    lines: list[dict[str, Any]] = []
    for ev in data.get("events") or []:
        segs = ev.get("segs")
        if not segs:
            continue
        text = " ".join("".join(s.get("utf8", "") for s in segs).split())
        if not text or all(ch in "♪♫♬ " for ch in text):
            continue
        start = float(ev.get("tStartMs", 0)) / 1000.0
        end = start + float(ev.get("dDurationMs", 0)) / 1000.0
        if lines and lines[-1]["text"] == text:
            lines[-1]["end"] = max(lines[-1]["end"], round(end, 3))
            continue
        lines.append({"start": round(start, 3), "end": round(end, 3), "text": text})
    return lines


@router.get("/{video_id}")
def list_caption_tracks(video_id: str):
    """이 영상의 자막 트랙 목록 (업로더 자막 전체 + 자동 생성은 원어/ja·ko·en만)."""
    import yt_dlp

    _validate(video_id)
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        with yt_dlp.YoutubeDL(_ydl_opts()) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as e:
        logger.exception(f"caption listing failed for {video_id}")
        raise HTTPException(status_code=502, detail="caption listing failed") from e
    if not info:
        raise HTTPException(status_code=502, detail="caption listing failed: no info")

    tracks: list[dict[str, Any]] = []
    for lang, entries in (info.get("subtitles") or {}).items():
        if lang == "live_chat":
            continue  # 라이브 채팅 리플레이 — 자막이 아니다
        tracks.append({"lang": lang, "label": _track_label(entries, lang, False), "auto": False})
    for lang, entries in (info.get("automatic_captions") or {}).items():
        if not (lang.endswith("-orig") or lang in _AUTO_LANG_ALLOW):
            continue
        tracks.append({"lang": lang, "label": _track_label(entries, lang, True), "auto": True})
    # 업로더 자막 우선, 그 다음 자동 원어 — 지나치게 길어지지 않게 상한
    tracks.sort(key=lambda t: (t["auto"], not t["lang"].endswith("-orig")))
    return {"video_id": video_id, "tracks": tracks[:12]}


@router.get("/{video_id}/{lang}")
def get_caption_lines(video_id: str, lang: str, auto: bool = False):
    """선택한 트랙의 자막을 타이밍 포함 라인으로 반환 — yt-dlp가 json3로 받아 파싱."""
    import yt_dlp

    _validate(video_id, lang)
    url = f"https://www.youtube.com/watch?v={video_id}"
    tmp = Path(tempfile.mkdtemp(prefix="eycap-"))
    try:
        opts = _ydl_opts(
            {
                "writesubtitles": not auto,
                "writeautomaticsub": auto,
                "subtitleslangs": [lang],
                "subtitlesformat": "json3",
                "outtmpl": str(tmp / "cap.%(ext)s"),
            }
        )
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.extract_info(url, download=True)
        except Exception as e:
            logger.exception(f"caption download failed for {video_id}/{lang}")
            raise HTTPException(status_code=502, detail="caption download failed") from e

        files = sorted(tmp.glob("*.json3"))
        if not files:
            raise HTTPException(status_code=404, detail=f"no {lang} caption on this video")
        data = json.loads(files[0].read_text(encoding="utf-8"))
        lines = json3_events_to_lines(data)
        if not lines:
            raise HTTPException(status_code=404, detail="caption track is empty")
        return {"video_id": video_id, "lang": lang, "auto": auto, "lines": lines}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
