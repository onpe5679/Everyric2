"""집계 조회 API — POST /api/stats/views.

확장의 로컬 기여 이력("내가 만든 N곡을 M명이 봤어요")은 자기가 만든 video_id 목록을
브라우저 로컬 스토리지에 들고 있다가, 이 엔드포인트로 배치 조회해 각 영상의 누적
조회수(SyncView, sync.py의 GET /api/sync/{video_id} 성공 시 증가)를 합산해 보여준다.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from everyric2.server.api.sync import _VIDEO_ID_RE
from everyric2.server.db.connection import get_session
from everyric2.server.db.repository import SyncViewRepository

router = APIRouter(prefix="/api/stats", tags=["stats"])

# 한 번에 물을 수 있는 영상 수 상한 — 로컬 기여 이력이 무한정 자라 요청 하나가 DB를
# 무제한 IN절로 두들기는 것을 막는다. 곡 하나 만드는 데도 시간이 걸리는 서비스 특성상
# 한 사용자가 100곡을 만들 일은 현실적으로 드물다.
_MAX_VIDEO_IDS = 100


class ViewsRequest(BaseModel):
    video_ids: list[str] = Field(max_length=_MAX_VIDEO_IDS)


class ViewsResponse(BaseModel):
    views: dict[str, int]


@router.post("/views", response_model=ViewsResponse)
async def get_view_counts(request: ViewsRequest):
    """요청한 video_id들의 누적 조회수 — 없는 영상(아직 아무도 안 봤거나 SyncView 행 자체가
    없음)은 0으로 채운다. 순서·중복은 신경 쓰지 않는다(dict 응답이라 호출부가 키로 찾는다)."""
    ids = [vid for vid in request.video_ids if _VIDEO_ID_RE.match(vid)]
    if len(ids) != len(request.video_ids):
        raise HTTPException(status_code=422, detail="invalid video_id in video_ids")

    async with get_session() as session:
        found = await SyncViewRepository(session).get_many(ids)
        return ViewsResponse(views={vid: found.get(vid, 0) for vid in ids})
