"""링크 후보 검증 API (link-jobs) — 범용 오케스트레이터가 X-API-Key로 호출한다.

"커버(video_id)가 원곡(source_video_id)과 같은 반주를 쓰는가"를 반주 상관으로 자동 판정해
SyncLink를 안전하게 자동 생성하기 위한 잡 큐다. 인증은 서버 전역 미들웨어(main.py)의
X-API-Key 검사로 처리한다 — 배포에서 EVERYRIC_SERVER_API_KEY가 설정되면 이 라우트도 키를
요구한다. 실제 판정(다운로드/분리/상관)은 원격 GPU 워커가 claim해 수행한다.
"""

import re

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from everyric2.server.db.connection import get_session
from everyric2.server.db.repository import LinkJobRepository

router = APIRouter(prefix="/api/link-jobs", tags=["link-jobs"])

_VIDEO_ID_PATTERN = r"^[A-Za-z0-9_-]{11}$"
_LINK_JOB_ID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


class LinkJobRequest(BaseModel):
    video_id: str = Field(pattern=_VIDEO_ID_PATTERN)
    source_video_id: str = Field(pattern=_VIDEO_ID_PATTERN)


class LinkJobCreateResponse(BaseModel):
    id: str


class LinkJobStatusResponse(BaseModel):
    status: str
    match: bool | None = None
    offset_sec: float | None = None
    confidence: float | None = None
    error: str | None = None


@router.post("", response_model=LinkJobCreateResponse)
async def create_link_job(request: LinkJobRequest):
    """링크 검증 잡 생성. 자기 자신 검증은 거부. 같은 쌍이 이미 진행 중(queued/processing)이면
    새 잡을 만들지 않고 그 id를 돌려준다(중복 방지)."""
    if request.video_id == request.source_video_id:
        raise HTTPException(status_code=400, detail="Cannot validate a video against itself")
    async with get_session() as session:
        repo = LinkJobRepository(session)
        active = await repo.get_active_pair(request.video_id, request.source_video_id)
        if active:
            return LinkJobCreateResponse(id=active.id)
        link_job = await repo.create(request.video_id, request.source_video_id)
        return LinkJobCreateResponse(id=link_job.id)


@router.get("/{link_job_id}", response_model=LinkJobStatusResponse)
async def get_link_job(link_job_id: str):
    if not _LINK_JOB_ID_RE.match(link_job_id):
        raise HTTPException(status_code=422, detail="invalid link_job_id")
    async with get_session() as session:
        link_job = await LinkJobRepository(session).get_by_id(link_job_id)
        if not link_job:
            raise HTTPException(status_code=404, detail="Link job not found")
        return LinkJobStatusResponse(
            status=link_job.status,
            match=link_job.match,
            offset_sec=link_job.offset_sec,
            confidence=link_job.confidence,
            error=link_job.error,
        )
