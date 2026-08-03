"""쿼터 조회 API — GET /api/limits/{video_id}.

실제 소비 지점(sync.py._check_action_limit/_check_destructive_limit)과 **같은 집계**
(ActionLogRepository.count_recent)를 읽기 전용으로 노출한다. 확장이 생성·초기화 버튼을
누르기 전에 "오늘 몇 번 남았는지" 미리 보여주려는 용도라, 이 조회 자체는 action_logs에
아무것도 남기지 않는다 — 남기면 조회 자체가 한도를 깎아먹는 모순이 생긴다.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from everyric2.config.settings import get_settings
from everyric2.server.api.sync import DAILY_GENERATE_LIMIT, _VIDEO_ID_RE
from everyric2.server.db.connection import get_session
from everyric2.server.db.repository import ActionLogRepository

router = APIRouter(prefix="/api/limits", tags=["limits"])


class _LimitDetail(BaseModel):
    limit: int
    used: int
    remaining: int


class LimitsResponse(BaseModel):
    # admin_api_key 미설정(로컬 단일사용자 기본)이면 한도가 아예 없다 — used=0/remaining=limit
    # 로 "무제한"을 값으로 표현해, 확장이 별도 null 분기 없이 숫자만 보고 그릴 수 있게 한다.
    enforced: bool
    generate: _LimitDetail
    destructive: _LimitDetail
    window_hours: int = 24


@router.get("/{video_id}", response_model=LimitsResponse)
async def get_limits(video_id: str):
    """이 영상의 24시간 쿼터 현황.

    destructive는 "reset"과 "regenerate(force)" 두 행위를 합쳐 하나로 보여준다 — 실제
    한도 검사는 이 둘을 (action, video_id)별로 **독립** 집계하므로(같은 daily_destructive_
    limit을 각자 따로 소비) 이론상 두 행위를 합쳐 limit의 2배까지 할 수 있다. 하지만 확장
    UI에는 배지 하나만 있으므로, 더 많이 소비된 쪽(worst case)을 used로 보여준다 — "적어도
    이만큼은 남아 있다"는 보수적 신호가 "실제보다 넉넉해 보여 한도 초과로 안내가 어긋나는"
    쪽보다 안전하다."""
    if not _VIDEO_ID_RE.match(video_id):
        raise HTTPException(status_code=422, detail="invalid video_id")

    server = get_settings().server
    enforced = bool(server.admin_api_key)
    destructive_limit = server.daily_destructive_limit

    async with get_session() as session:
        if not enforced:
            return LimitsResponse(
                enforced=False,
                generate=_LimitDetail(
                    limit=DAILY_GENERATE_LIMIT, used=0, remaining=DAILY_GENERATE_LIMIT
                ),
                destructive=_LimitDetail(
                    limit=destructive_limit, used=0, remaining=max(0, destructive_limit)
                ),
            )

        repo = ActionLogRepository(session)
        generate_used = await repo.count_recent("generate", video_id)
        reset_used = await repo.count_recent("reset", video_id)
        regenerate_used = await repo.count_recent("regenerate", video_id)
        destructive_used = max(reset_used, regenerate_used)

        return LimitsResponse(
            enforced=True,
            generate=_LimitDetail(
                limit=DAILY_GENERATE_LIMIT,
                used=generate_used,
                remaining=max(0, DAILY_GENERATE_LIMIT - generate_used),
            ),
            destructive=_LimitDetail(
                limit=destructive_limit,
                used=destructive_used,
                remaining=max(0, destructive_limit - destructive_used),
            ),
        )
