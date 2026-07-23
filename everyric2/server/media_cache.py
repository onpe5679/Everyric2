"""외부 미디어 캐시 조회 + 오디오 추출 (mediacache/1 consumer, 중립).

동거 호스트가 이미 받아 둔 원본 미디어를 재다운로드 없이 재사용한다. 이 리포는 중립 계약
(GET {url}/lookup?platform=youtube&id=<id> → {found, path, ext, duration_sec})만 알고,
실제 외부 서비스 이름·스키마는 남기지 않는다. 히트 시 ffmpeg 스트림카피로 오디오만 추출해
잡 처리 주체(원격 워커/인프로세스)에게 넘긴다. 조회/추출 실패는 전부 조용히 yt-dlp 경로로
폴백한다(치명 아님). 과길이는 다운로드 없이 프리플라이트로 즉시 실패시킨다.

추출은 **전역 asyncio.Semaphore(1)**로 직렬화한다 — 동거 호스트의 CPU/NAS I/O 예산을 넘지
않기 위한 합의 조건. 추출 오디오는 워커 인증 뒤에만 존재하는 임시 파일이고 터미널 시 지운다
(외부 재서빙 엔드포인트 없음 — 저작권 규약).
"""

import asyncio
import logging
import os
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

_LOOKUP_TIMEOUT_SEC = 3.0
_FFMPEG_TIMEOUT_SEC = 120

# 동거 호스트 CPU/NAS I/O 예산 — 추출을 1개로 직렬화 (합의 조건)
_EXTRACT_SEMAPHORE: asyncio.Semaphore | None = None


def _extract_semaphore() -> asyncio.Semaphore:
    global _EXTRACT_SEMAPHORE
    if _EXTRACT_SEMAPHORE is None:
        _EXTRACT_SEMAPHORE = asyncio.Semaphore(1)
    return _EXTRACT_SEMAPHORE


def _lookup(url: str, key: str, video_id: str) -> dict:
    import requests

    resp = requests.get(
        f"{url.rstrip('/')}/lookup",
        params={"platform": "youtube", "id": video_id},
        headers={"Authorization": f"Bearer {key}"} if key else {},
        timeout=_LOOKUP_TIMEOUT_SEC,
    )
    resp.raise_for_status()
    return resp.json()


async def prepare_cached_audio(
    video_id: str, job_id: str, max_audio_sec: int
) -> tuple[str | None, str | None]:
    """미디어 캐시 조회 → (audio_path | None, fail_reason | None).

    - audio_path: 추출 성공 → 이 로컬 파일을 yt-dlp 대신 쓴다.
    - fail_reason: 과길이 등 사용자 노출 실패 문구 → 다운로드 없이 잡을 즉시 실패시킨다.
    - (None, None): 캐시 미설정/미스/오류/추출 실패 → 기존 yt-dlp 경로.
    """
    from everyric2.config.settings import get_settings

    server = get_settings().server
    if not server.media_cache_url:
        return None, None

    try:
        data = await asyncio.to_thread(
            _lookup, server.media_cache_url, server.media_cache_key, video_id
        )
    except Exception:
        logger.info("미디어 캐시 조회 실패 — yt-dlp로 폴백해요 (video %s)", video_id)
        return None, None

    if not data.get("found"):
        return None, None

    src = data.get("path")
    if not src or not os.path.isfile(src) or not os.access(src, os.R_OK):
        logger.info("미디어 캐시 경로가 없거나 읽을 수 없어 yt-dlp로 폴백해요 (video %s)", video_id)
        return None, None

    duration = data.get("duration_sec")
    if max_audio_sec > 0 and duration and duration > max_audio_sec:
        # 과길이는 추출·다운로드 없이 프리플라이트로 즉시 실패 (기존 과길이 문구 재사용)
        from everyric2.server.worker import over_length_message

        return None, over_length_message(float(duration), max_audio_sec)

    from everyric2.config.settings import get_settings as _get_settings

    dest = _get_settings().audio.temp_dir / f"{video_id}-{job_id[:8]}.m4a"
    ok = await _extract(src, dest)
    if not ok:
        logger.info("미디어 캐시 오디오 추출 실패 — yt-dlp로 폴백해요 (video %s)", video_id)
        return None, None
    logger.info("미디어 캐시 히트 — 추출 오디오 사용 (video %s)", video_id)
    return str(dest), None


def fetch_cached_audio_sync(video_id: str, tag: str) -> str | None:
    """워커 컨텍스트(동기)용 미디어 캐시 조달 — 조회+추출을 현재 스레드에서 수행.

    링크 검증처럼 서버 프로세스 밖(같은 호스트 워커)에서 쓴다. 워커는 잡을 한 번에 하나만
    처리하므로 추출이 자연 직렬이라 전역 세마포어는 생략한다. 미설정/미스/경로 비가독/추출
    실패는 전부 None → 호출부가 yt-dlp로 폴백. NAS가 이 호스트에 안 붙은 원격 워커는
    조회는 성공해도 경로 검사에서 미스가 나 자연 폴백된다."""
    from everyric2.config.settings import get_settings

    settings = get_settings()
    server = settings.server
    if not server.media_cache_url:
        return None
    try:
        data = _lookup(server.media_cache_url, server.media_cache_key, video_id)
    except Exception:
        logger.info("미디어 캐시 조회 실패 — yt-dlp로 폴백해요 (video %s)", video_id)
        return None
    if not data.get("found"):
        return None
    src = data.get("path")
    if not src or not os.path.isfile(src) or not os.access(src, os.R_OK):
        return None
    dest = settings.audio.temp_dir / f"linkcache-{tag}-{video_id}.m4a"
    if not _run_ffmpeg(src, dest):
        logger.info("미디어 캐시 오디오 추출 실패 — yt-dlp로 폴백해요 (video %s)", video_id)
        return None
    logger.info("미디어 캐시 히트 — 링크 검증 오디오 사용 (video %s)", video_id)
    return str(dest)


async def _extract(src: str, dest: Path) -> bool:
    async with _extract_semaphore():
        return await asyncio.to_thread(_run_ffmpeg, src, dest)


def _run_ffmpeg(src: str, dest: Path) -> bool:
    """ffmpeg 스트림카피로 오디오만 추출. POSIX에선 nice/ionice로 우선순위를 낮춘다
    (Windows엔 없으므로 ffmpeg 단독). 코덱 비호환 등 실패 시 False → yt-dlp 폴백."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd: list[str] = []
    if os.name == "posix":
        for tool, args in (("nice", ["-n", "19"]), ("ionice", ["-c", "3"])):
            if shutil.which(tool):
                cmd += [tool, *args]
    cmd += ["ffmpeg", "-y", "-i", str(src), "-vn", "-acodec", "copy", str(dest)]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=_FFMPEG_TIMEOUT_SEC)
    except Exception:
        dest.unlink(missing_ok=True)
        return False
    if result.returncode != 0 or not dest.exists():
        dest.unlink(missing_ok=True)
        return False
    return True
