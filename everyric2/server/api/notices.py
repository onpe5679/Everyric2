"""운영 공지 API — 확장이 GET /api/notices로 조회해 배너·모달로 띄운다 (2026-08-04).

생성·비활성화는 어드민 전용이다. 인증은 이 레포에 이미 있는 관례를 그대로 재사용한다:
sync.py의 create_sync_link/reset_video_syncs가 쓰는 X-API-Key 헤더를
settings.server.admin_api_key와 비교하는 방식 — 새 헤더(X-Admin-Key 등)를 만들지 않는다.
"""

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from everyric2.config.settings import get_settings
from everyric2.server.db.connection import get_session
from everyric2.server.db.repository import NoticeRepository

router = APIRouter(prefix="/api/notices", tags=["notices"])

_LEVELS = ("info", "warning", "critical")


def _require_admin(x_api_key: str | None) -> None:
    """어드민 키 검사 — sync.py의 기존 관례(X-API-Key == admin_api_key)를 그대로 따른다.

    키 자체가 미설정이면 이 기능을 배포가 아예 켜지 않은 것으로 보아 503(기능 비활성) —
    403(키가 틀림)과 구분해야 운영자가 "설정을 안 했다"와 "키를 잘못 보냈다"를 가른다."""
    server = get_settings().server
    if not server.admin_api_key:
        raise HTTPException(status_code=503, detail="공지 관리 기능이 꺼져 있어요 (admin_api_key 미설정)")
    if x_api_key != server.admin_api_key:
        raise HTTPException(status_code=403, detail="어드민 키가 필요해요")


class NoticeItem(BaseModel):
    id: int
    title: str
    body: str
    level: str
    created_at: str | None = None
    ends_at: str | None = None


class NoticeListResponse(BaseModel):
    notices: list[NoticeItem]


class CreateNoticeRequest(BaseModel):
    title: str = Field(max_length=200)
    body: str = Field(max_length=10000)
    level: str = Field(default="info", pattern="^(info|warning|critical)$")
    ends_at: datetime | None = None


class NoticeMutationResponse(BaseModel):
    id: int
    ok: bool = True


@router.get("", response_model=NoticeListResponse)
async def list_notices():
    """지금 유효한 공지 목록(활성 + 아직 안 끝남), 최신순 최대 20건. 인증 불필요 — 확장이
    기동 때마다 조용히 조회하는 읽기 전용 경로다(생성·초기화 같은 GPU/파괴적 행위가 아니다)."""
    async with get_session() as session:
        notices = await NoticeRepository(session).list_active(limit=20)
        return NoticeListResponse(
            notices=[
                NoticeItem(
                    id=n.id,
                    title=n.title,
                    body=n.body,
                    level=n.level,
                    created_at=n.created_at.isoformat() if n.created_at else None,
                    ends_at=n.ends_at.isoformat() if n.ends_at else None,
                )
                for n in notices
            ]
        )


@router.post("", response_model=NoticeMutationResponse)
async def create_notice(
    request: CreateNoticeRequest,
    x_api_key: Annotated[str | None, Header()] = None,
):
    _require_admin(x_api_key)
    async with get_session() as session:
        notice = await NoticeRepository(session).create(
            title=request.title, body=request.body, level=request.level, ends_at=request.ends_at
        )
        return NoticeMutationResponse(id=notice.id)


@router.post("/{notice_id}/deactivate", response_model=NoticeMutationResponse)
async def deactivate_notice(
    notice_id: int,
    x_api_key: Annotated[str | None, Header()] = None,
):
    _require_admin(x_api_key)
    async with get_session() as session:
        found = await NoticeRepository(session).deactivate(notice_id)
        if not found:
            raise HTTPException(status_code=404, detail="공지를 찾을 수 없어요")
        return NoticeMutationResponse(id=notice_id)
