"""``video_id`` 키 오디오 캐시 — 같은 영상을 두 번 받지 않기 위한 것.

기존 캐시는 ``(audio_hash, lyrics_hash)`` 키다. 해시를 구하려면 파일이 있어야 하고 파일을
구하려면 다운로드를 해야 하므로, **GPU 정렬은 아껐지만 유튜브 접촉은 한 번도 아끼지 못했다.**
2026-07-26 밤샘 배치 실측이 그 대가를 처음 숫자로 보여줬다 — 싱크 생성 182건에 유튜브
다운로드 275회, 같은 곡을 재처리하면 또 275회다.

부수 효과가 하나 더 있는데 이쪽이 실은 더 중요하다. ``audio_hash``는 파일 바이트 해시라서
**확보 경로에 의존한다**(미디어 캐시는 m4a 스트림카피, yt-dlp는 opus 우선 스트림카피 —
소스 코덱에 따라 opus/m4a/webm으로 갈리고, 디코드 안 되는 극소수만 wav로 트랜스코드
구제한다. ``worker.compute_audio_hash`` 참조). 같은 영상이 경로에 따라 다른 해시를 내면 해시 캐시가
미스해서 GPU 정렬을 다시 돈다. 이 캐시가 앞에 서면 같은 영상은 항상 같은 파일을 내므로
해시가 안정되고, 그래야 해시 캐시가 제 일을 한다.

## 보관 파일을 그대로 넘기지 않는다

파이프라인은 **입력 오디오를 지운다** — 과길이 초과, 취소 경로, 캐시 완결, 그리고
``_run_alignment``의 ``finally``가 각각 ``unlink``한다. 보관 원본을 넘기면 첫 잡이 캐시를
지워 버린다. 그래서 조회는 항상 작업용 **복사본**을 만들어 돌려준다. 하드링크가 디스크를
아끼지만 쓰지 않는다 — 누군가 나중에 오디오를 제자리에서 정규화하는 코드를 넣으면 캐시가
조용히 오염되고, 그 사고는 30MB 복사보다 훨씬 비싸다.

## 락은 프로세스 안에서만 병합한다

같은 영상에 동시 요청이 오면(공개 트래픽은 인기곡에 몰린다) 캐시는 두 번째 요청부터
듣지만 락은 첫 순간부터 듣는다. 여러 프로세스(서버 + 원격 워커)까지 병합하려면 파일 락이
필요한데, 프로세스마다 자기 캐시를 갖는 구조라 이득이 작아 넣지 않았다.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

# 캐시 파일명에 그대로 들어가는 값이라 형태를 강제한다. 유튜브 ID는 11자
# ``[A-Za-z0-9_-]``이지만 길이는 묶지 않는다(플랫폼이 늘어날 수 있다). 경로 구분자와
# ``..``이 절대 통과하지 못하게 하는 것이 요점이다.
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# 확장자도 파일명이 되므로 허용 목록으로 둔다. yt-dlp 경로는 opus 우선(소스에 따라
# m4a/webm으로도 갈리고, 디코드 실패 구제 시에만 wav), 미디어 캐시 경로는 m4a다.
_ALLOWED_EXT = frozenset({".wav", ".m4a", ".mp3", ".opus", ".ogg", ".flac", ".webm", ".aac"})

_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


class AudioCacheDisabled(Exception):
    """캐시가 꺼져 있거나 쓸 수 없을 때 — 호출부는 잡아서 다운로드로 넘어간다."""


def is_safe_id(video_id: str) -> bool:
    return bool(video_id) and bool(_SAFE_ID.match(video_id))


@contextmanager
def video_lock(video_id: str) -> Iterator[None]:
    """같은 영상의 확보를 프로세스 안에서 한 번으로 병합한다(single-flight).

    락을 잡은 쪽이 다운로드하고 캐시에 넣는다. 기다린 쪽은 락을 얻은 뒤 캐시를 다시
    조회하므로(호출부의 책임) 다운로드가 한 번으로 줄어든다. 락 객체는 영상마다 하나씩
    만들어 두고 지우지 않는다 — 하나가 수십 바이트이고, 지우려면 «지금 이 락을 기다리는
    사람이 없다»를 원자적으로 알아야 해서 얻는 것보다 잃는 게 많다.
    """
    if not is_safe_id(video_id):
        # 형태가 이상한 ID는 병합하지 않는다(캐시도 거부한다). 그냥 통과시킨다.
        yield
        return
    with _LOCKS_GUARD:
        lock = _LOCKS.setdefault(video_id, threading.Lock())
    with lock:
        yield


def _settings():
    from everyric2.config.settings import get_settings

    return get_settings().audio


def cache_dir() -> Path:
    """캐시 디렉터리. 캐시가 꺼져 있으면 ``AudioCacheDisabled``."""
    s = _settings()
    if not s.cache_enabled:
        raise AudioCacheDisabled("audio cache disabled")
    d = Path(s.cache_dir)
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise AudioCacheDisabled(f"cache dir unusable: {exc}") from exc
    return d


def find(video_id: str) -> Path | None:
    """보관된 파일 경로. 없거나 캐시가 꺼져 있으면 None."""
    if not is_safe_id(video_id):
        return None
    try:
        d = cache_dir()
    except AudioCacheDisabled:
        return None
    for ext in sorted(_ALLOWED_EXT):
        p = d / f"{video_id}{ext}"
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def take(video_id: str, dest_dir: Path, tag: str) -> Path | None:
    """캐시에서 **작업용 복사본**을 만들어 돌려준다. 미스면 None.

    돌려준 파일은 호출부(파이프라인)가 지워도 된다 — 캐시 원본은 그대로 남는다.
    조회에 성공하면 mtime을 지금으로 올려 LRU가 «최근 쓴 것»을 지키게 한다.
    """
    src = find(video_id)
    if src is None:
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{video_id}-{tag}{src.suffix}"
    try:
        shutil.copy2(src, dest)
    except OSError as exc:
        logger.info("캐시 오디오 복사 실패 — 다운로드로 넘어가요 (video %s): %s", video_id, exc)
        dest.unlink(missing_ok=True)
        return None
    try:
        os.utime(src, None)
    except OSError:
        pass  # mtime 갱신 실패는 LRU 정확도만 떨어뜨린다 — 캐시 히트를 버릴 이유가 아니다
    logger.info("오디오 캐시 히트 — 다운로드를 건너뜁니다 (video %s)", video_id)
    return dest


def put(video_id: str, src: Path) -> Path | None:
    """확보한 오디오를 캐시에 보관한다. 실패는 삼키고 None(보관 실패는 잡을 깨지 않는다).

    이미 보관돼 있으면 다시 쓰지 않는다 — 같은 영상을 두 경로로 받아 바이트가 다를 수
    있는데, 그때 뒤엣것으로 덮으면 ``audio_hash``가 흔들려 해시 캐시가 미스한다.
    **먼저 들어온 하나를 계속 쓰는 것**이 해시를 안정시킨다.
    """
    if not is_safe_id(video_id):
        return None
    src = Path(src)
    if src.suffix.lower() not in _ALLOWED_EXT:
        return None
    try:
        if not src.is_file() or src.stat().st_size == 0:
            return None
        d = cache_dir()
    except (AudioCacheDisabled, OSError):
        return None

    existing = find(video_id)
    if existing is not None:
        return existing

    dest = d / f"{video_id}{src.suffix.lower()}"
    # 같은 디렉터리에 임시 이름으로 쓰고 원자적으로 바꾼다 — 복사 중에 다른 잡이 반쪽
    # 파일을 히트하면 정렬이 잘린 오디오로 돌아간다.
    tmp = d / f".{video_id}.{os.getpid()}.part"
    try:
        shutil.copy2(src, tmp)
        os.replace(tmp, dest)
    except OSError as exc:
        logger.info("오디오 캐시 보관 실패 (video %s): %s", video_id, exc)
        tmp.unlink(missing_ok=True)
        return None
    prune()
    return dest


def stats() -> dict[str, int]:
    """파일 수와 총 바이트 — 로그·진단용."""
    try:
        d = cache_dir()
    except AudioCacheDisabled:
        return {"files": 0, "bytes": 0}
    files = [p for p in d.iterdir() if p.is_file() and not p.name.startswith(".")]
    return {"files": len(files), "bytes": sum(p.stat().st_size for p in files)}


def prune(max_bytes: int | None = None) -> int:
    """상한을 넘긴 만큼 오래 안 쓴 것부터 지운다. 지운 바이트를 돌려준다.

    wav 한 곡이 30MB대라 상한 없이 두면 디스크가 조용히 찬다. 상한 0/음수는 «무제한»이다
    (그것을 원하는 배포도 있다).
    """
    try:
        d = cache_dir()
    except AudioCacheDisabled:
        return 0
    if max_bytes is None:
        max_bytes = int(float(_settings().cache_max_gb) * 1024**3)
    if max_bytes <= 0:
        return 0

    entries = []
    total = 0
    for p in d.iterdir():
        if not p.is_file() or p.name.startswith("."):
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        entries.append((st.st_mtime, st.st_size, p))
        total += st.st_size
    if total <= max_bytes:
        return 0

    entries.sort()  # mtime 오름차순 — 오래 안 쓴 것이 앞
    freed = 0
    for _mtime, size, p in entries:
        if total - freed <= max_bytes:
            break
        try:
            p.unlink()
        except OSError:
            continue
        freed += size
    if freed:
        logger.info(
            "오디오 캐시 정리 — %.1fGB 중 %.1fGB 비웠어요 (상한 %.1fGB)",
            total / 1024**3, freed / 1024**3, max_bytes / 1024**3,
        )
    return freed
