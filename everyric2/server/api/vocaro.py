"""보카로 가사 위키 원제 매칭 API.

유튜브 영상 제목이 일본어 원제로 되어 있어 클라이언트의 한국어 독음 인덱스로는
찾지 못하는 경우, 서버 측 원제/한국어 겸용 인덱스(vocaro_index)로 슬러그를 답한다.
"""

from fastapi import APIRouter, BackgroundTasks, Query
from pydantic import BaseModel

from everyric2.server.vocaro_index import BASE_URL, build_index, index_status, is_building, match

router = APIRouter(prefix="/api/vocaro", tags=["vocaro"])


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


@router.get("/match", response_model=VocaroMatchResponse)
async def match_title(background_tasks: BackgroundTasks, title: str = Query(..., min_length=1)):
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
    if is_building():
        return VocaroReindexResponse(status="already_building")
    background_tasks.add_task(build_index, force=force)
    return VocaroReindexResponse(status="building")


@router.get("/status", response_model=VocaroStatusResponse)
async def status():
    return VocaroStatusResponse(**index_status())
