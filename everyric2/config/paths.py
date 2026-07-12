"""공유 파일 경로 — OS별 임시 디렉터리 기준.

예전에는 Path("/tmp/everyric2/...") 하드코딩이었는데, Windows pathlib에서 선행 "/"는
현재 드라이브 루트로 해석돼 실제로는 C:\\tmp\\everyric2\\...를 가리켰다 (진짜 임시
디렉터리가 아니라 정리도 안 되는 위치). tempfile.gettempdir() 기준으로 옮기되, 기존
설치 사용자의 쿠키를 잃지 않도록 새 경로에 없고 레거시에 있으면 레거시를 계속 읽는다 —
업로드는 항상 새 경로에 쓰므로 다음 업로드 때 자연 이관된다.
"""

import tempfile
from pathlib import Path

LEGACY_COOKIES_PATH = Path("/tmp/everyric2/youtube_cookies.txt")


def cookies_write_path() -> Path:
    """쿠키 업로드가 쓰는 경로 — 항상 새(진짜 임시 디렉터리) 위치."""
    return Path(tempfile.gettempdir()) / "everyric2" / "youtube_cookies.txt"


def cookies_read_path() -> Path:
    """쿠키를 읽을 경로 — 새 위치 우선, 없으면 레거시(기존 설치 호환)."""
    new = cookies_write_path()
    if not new.exists() and LEGACY_COOKIES_PATH.exists():
        return LEGACY_COOKIES_PATH
    return new
