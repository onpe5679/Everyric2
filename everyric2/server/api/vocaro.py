"""보카로 가사 위키 원제 매칭 API.

유튜브 영상 제목이 일본어 원제로 되어 있어 클라이언트의 한국어 독음 인덱스로는
찾지 못하는 경우, 서버 측 원제/한국어 겸용 인덱스(vocaro_index)로 슬러그를 답한다.

`EVERYRIC_SERVER_SONG_INDEX_URL`이 설정되면 로컬 인덱스 대신 외부 곡 인덱스(songindex/1)로
프록시한다 — 확장이 보는 응답 형태(VocaroMatchResponse)는 어느 경로든 동일하다. 이관 검증이
끝나면 별도 커밋으로 로컬 크롤러를 제거할 예정이라 이번 작업에선 지우지 않는다.
"""

import asyncio
import logging

from fastapi import APIRouter, BackgroundTasks, Query
from pydantic import BaseModel

from everyric2.config.settings import get_settings
from everyric2.server.vocaro_index import BASE_URL, build_index, index_status, is_building, match

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/vocaro", tags=["vocaro"])

# 업스트림 곡 인덱스 요청 타임아웃(초) — 확장 매칭은 대화형이라 짧게 잡고 실패 시 미발견 폴백
_UPSTREAM_TIMEOUT_SEC = 3.0


class VocaroMatchResponse(BaseModel):
    found: bool
    slug: str | None = None
    page_url: str | None = None
    ko: str | None = None
    ja: str | None = None
    status: str | None = None


class VocaroReindexResponse(BaseModel):
    status: str


class VocaroStatusResponse(BaseModel):
    built_at: str | None
    total: int
    with_ja: int
    building: bool


def _song_index_url() -> str:
    return get_settings().server.song_index_url.rstrip("/")


def _upstream_headers() -> dict[str, str]:
    key = get_settings().server.song_index_key
    return {"Authorization": f"Bearer {key}"} if key else {}


def _upstream_get(path: str, params: dict | None = None) -> dict:
    """외부 곡 인덱스 동기 GET (asyncio.to_thread로 감싸 호출). 의존성 추가 없이 requests 사용."""
    import requests

    resp = requests.get(
        f"{_song_index_url()}{path}",
        params=params or {},
        headers=_upstream_headers(),
        timeout=_UPSTREAM_TIMEOUT_SEC,
    )
    resp.raise_for_status()
    return resp.json()


@router.get("/match", response_model=VocaroMatchResponse)
async def match_title(background_tasks: BackgroundTasks, title: str = Query(..., min_length=1)):
    # 업스트림 모드: 외부 곡 인덱스로 프록시하고 응답을 1:1 매핑한다 (확장 응답 형태 무변경)
    if _song_index_url():
        try:
            data = await asyncio.to_thread(_upstream_get, "/match", {"title": title})
        except Exception as e:
            logger.info("외부 곡 인덱스 매칭 실패 — 미발견으로 폴백: %s", e)
            return VocaroMatchResponse(found=False, status="upstream_error")
        return VocaroMatchResponse(
            found=bool(data.get("found")),
            slug=data.get("slug"),
            page_url=data.get("page_url"),
            ko=data.get("ko"),
            ja=data.get("ja"),
            status=data.get("status"),
        )

    result = match(title)
    if result:
        return VocaroMatchResponse(
            found=True,
            slug=result.slug,
            page_url=f"{BASE_URL}/{result.slug}",
            ko=result.ko,
            ja=result.ja,
        )

    if index_status()["total"] == 0:
        # 인덱스가 아직 없으면 매칭 실패와 함께 백그라운드 빌드를 킥한다 (중복 킥은 락으로 방지)
        if not is_building():
            background_tasks.add_task(build_index)
        return VocaroMatchResponse(found=False, status="index_empty")

    return VocaroMatchResponse(found=False)


@router.post("/reindex", response_model=VocaroReindexResponse)
async def reindex(background_tasks: BackgroundTasks, force: bool = False):
    # 업스트림 모드에선 인덱스를 서버가 소유하지 않으므로 빌드 킥 없이 알린다
    if _song_index_url():
        return VocaroReindexResponse(status="upstream")
    if is_building():
        return VocaroReindexResponse(status="already_building")
    background_tasks.add_task(build_index, force=force)
    return VocaroReindexResponse(status="building")


@router.get("/status", response_model=VocaroStatusResponse)
async def status():
    # 업스트림 모드에선 외부 인덱스의 /status를 중계한다 (오류 시 비어 있는 상태로)
    if _song_index_url():
        try:
            data = await asyncio.to_thread(_upstream_get, "/status")
        except Exception as e:
            logger.info("외부 곡 인덱스 상태 조회 실패: %s", e)
            return VocaroStatusResponse(built_at=None, total=0, with_ja=0, building=False)
        return VocaroStatusResponse(
            built_at=data.get("built_at"),
            total=int(data.get("total", 0) or 0),
            with_ja=int(data.get("with_ja", 0) or 0),
            building=bool(data.get("building", False)),
        )
    return VocaroStatusResponse(**index_status())
