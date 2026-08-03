"""쿼터 조회 API — GET /api/limits/{video_id}.

실제 소비 지점(sync.py._check_action_limit/_check_destructive_limit)과 **같은 집계**
(ActionLogRepository.count_recent)를 읽기 전용으로 노출한다. 확장이 생성·초기화 버튼을
누르기 전에 "오늘 몇 번 남았는지" 미리 보여주려는 용도라, 이 조회 자체는 action_logs에
아무것도 남기지 않는다 — 남기면 조회 자체가 한도를 깎아먹는 모순이 생긴다.

next_reset_at(2026-08-04, 확장 "내 기여/남은 횟수" 페이지 상단 안내용 보강, additive):
**실제 기전은 고정 시각(자정 등) 리셋이 아니라 행위별 롤링 24시간 창**(count_recent가
`now - hours` 이후만 센다) — "매일 0시에 리셋" 같은 문구는 사실과 다르다. 대신 이 창
안에서 가장 오래된 기록이 창 밖으로 나가는(=카운트가 실제로 줄어드는) **정확한 다음
시각**을 계산해서 준다. 그래서 root에 resets_daily 같은 뭉뚱그린 불리언은 안 붙였다 —
window_hours(이미 있음)로 "약 24시간 주기"라는 사실은 전달되고, next_reset_at이 그 주기의
구체적인 다음 시점을 준다. 확장은 이 값으로 "N시간 후 리셋" 카운트다운을 그릴 수 있고,
"매일" 같은 문구를 쓰려면 window_hours==24라는 사실에 근거해 쓰면 된다(현재 유일한 값).

link·upgrade 버킷(2026-08-04, 운영자 지시 — "남은 횟수"에 커버 잇기·정렬 업그레이드도
표시):

- **link**: GET .../link-candidates가 실제로 GPU 잡을 제출하는 지점
  (sync.py._dispatch_candidate_followup)이 이미 action="link_candidates"로 독립 집계하고
  있었다 — generate/destructive와 겹치지 않는 진짜 카운터라 그대로 노출한다
  (limit=DAILY_LINK_CANDIDATE_LIMIT, 실측 3/24h).
- **upgrade**: 처음엔(같은 날 초판) "정렬 업그레이드"(min_depth, force 없는 재생성)가
  별도 액션 이름 없이 그대로 action="generate"로 로그돼, generate 값을 복사해 노출했다.
  운영자 정정("업그레이드도 당연히 생성 쿼터랑은 별개여야지")에 따라 sync.py 쪽에
  action="upgrade" 전용 카운터를 새로 분리했다(_check_upgrade_limit, daily_upgrade_limit
  설정값 — 기본 10회/24h). 이 파일은 이제 link와 동일하게 **진짜 독립 카운터**를 그대로
  노출한다 — generate 복사는 더 이상 하지 않는다.
"""

from datetime import timedelta

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from everyric2.config.settings import get_settings
from everyric2.server.api.sync import (
    DAILY_GENERATE_LIMIT,
    DAILY_LINK_CANDIDATE_LIMIT,
    _VIDEO_ID_RE,
)
from everyric2.server.db.connection import get_session
from everyric2.server.db.repository import ActionLogRepository

router = APIRouter(prefix="/api/limits", tags=["limits"])

# count_recent/oldest_recent 기본값(24)과 LimitsResponse.window_hours 기본값이 가리키는
# 같은 창 — 한 곳에서만 바꾸면 셋 다 같이 움직이게 상수로 뽑는다.
_WINDOW_HOURS = 24


class _LimitDetail(BaseModel):
    limit: int
    used: int
    remaining: int
    # 이 카운트가 다음으로 줄어드는 정확한 시각(UTC, tz 미표기 — 기존 created_at.isoformat()
    # 관례와 동일). used가 0(깎을 게 없음)이면 None. 계산 근거는 아래 _reset_at 참고.
    next_reset_at: str | None = None


class LimitsResponse(BaseModel):
    # admin_api_key 미설정(로컬 단일사용자 기본)이면 한도가 아예 없다 — used=0/remaining=limit
    # 로 "무제한"을 값으로 표현해, 확장이 별도 null 분기 없이 숫자만 보고 그릴 수 있게 한다.
    enforced: bool
    generate: _LimitDetail
    # 커버 잇기(다른 영상 싱크 연결) — action="link_candidates" 독립 집계, 모듈 docstring 참고.
    link: _LimitDetail
    # 정렬 업그레이드(분석 깊이 올리기) — action="upgrade" 독립 집계(daily_upgrade_limit).
    # 모듈 docstring "upgrade" 항목 참고.
    upgrade: _LimitDetail
    destructive: _LimitDetail
    window_hours: int = _WINDOW_HOURS


def _reset_at(oldest: object, hours: int) -> str | None:
    """가장 오래된 기록(oldest_recent 결과)이 창 밖으로 나가는 시각 — 그 순간이 카운트가
    실제로 줄어드는 다음 시각이다. 기록이 없으면(=count 0) 줄어들 게 없으니 None."""
    if oldest is None:
        return None
    return (oldest + timedelta(hours=hours)).isoformat()


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
    upgrade_limit = server.daily_upgrade_limit

    async with get_session() as session:
        if not enforced:
            return LimitsResponse(
                enforced=False,
                generate=_LimitDetail(
                    limit=DAILY_GENERATE_LIMIT, used=0, remaining=DAILY_GENERATE_LIMIT
                ),
                link=_LimitDetail(
                    limit=DAILY_LINK_CANDIDATE_LIMIT,
                    used=0,
                    remaining=DAILY_LINK_CANDIDATE_LIMIT,
                ),
                upgrade=_LimitDetail(
                    limit=upgrade_limit, used=0, remaining=max(0, upgrade_limit)
                ),
                destructive=_LimitDetail(
                    limit=destructive_limit, used=0, remaining=max(0, destructive_limit)
                ),
            )

        repo = ActionLogRepository(session)
        generate_used = await repo.count_recent("generate", video_id, hours=_WINDOW_HOURS)
        generate_oldest = await repo.oldest_recent("generate", video_id, hours=_WINDOW_HOURS)
        link_used = await repo.count_recent("link_candidates", video_id, hours=_WINDOW_HOURS)
        link_oldest = await repo.oldest_recent("link_candidates", video_id, hours=_WINDOW_HOURS)
        upgrade_used = await repo.count_recent("upgrade", video_id, hours=_WINDOW_HOURS)
        upgrade_oldest = await repo.oldest_recent("upgrade", video_id, hours=_WINDOW_HOURS)
        reset_used = await repo.count_recent("reset", video_id, hours=_WINDOW_HOURS)
        regenerate_used = await repo.count_recent("regenerate", video_id, hours=_WINDOW_HOURS)
        destructive_used = max(reset_used, regenerate_used)

        # destructive의 next_reset_at: used가 reset/regenerate 중 더 큰 쪽(worst case)이므로,
        # "언제 이 값이 실제로 줄어드는가"도 그 worst-case 쪽 기준이어야 한다. 두 행위가
        # **동률**이면(둘 다 == destructive_used) 어느 한쪽만 창을 벗어나서는 max()가 안
        # 줄어든다 — 둘 다 벗어나야 하므로 두 창-이탈 시각 중 **더 늦은 쪽**을 쓴다(코드
        # 근거는 db.repository.ActionLogRepository.oldest_recent 독스트링·이 함수 위
        # 모듈 docstring). 동률이 아니면 candidates가 하나뿐이라 그 값 그대로 쓴다.
        destructive_candidates: list[str] = []
        if reset_used == destructive_used:
            reset_oldest = await repo.oldest_recent("reset", video_id, hours=_WINDOW_HOURS)
            reset_reset_at = _reset_at(reset_oldest, _WINDOW_HOURS)
            if reset_reset_at is not None:
                destructive_candidates.append(reset_reset_at)
        if regenerate_used == destructive_used:
            regenerate_oldest = await repo.oldest_recent(
                "regenerate", video_id, hours=_WINDOW_HOURS
            )
            regenerate_reset_at = _reset_at(regenerate_oldest, _WINDOW_HOURS)
            if regenerate_reset_at is not None:
                destructive_candidates.append(regenerate_reset_at)
        destructive_reset_at = max(destructive_candidates) if destructive_candidates else None

        return LimitsResponse(
            enforced=True,
            generate=_LimitDetail(
                limit=DAILY_GENERATE_LIMIT,
                used=generate_used,
                remaining=max(0, DAILY_GENERATE_LIMIT - generate_used),
                next_reset_at=_reset_at(generate_oldest, _WINDOW_HOURS),
            ),
            link=_LimitDetail(
                limit=DAILY_LINK_CANDIDATE_LIMIT,
                used=link_used,
                remaining=max(0, DAILY_LINK_CANDIDATE_LIMIT - link_used),
                next_reset_at=_reset_at(link_oldest, _WINDOW_HOURS),
            ),
            upgrade=_LimitDetail(
                limit=upgrade_limit,
                used=upgrade_used,
                remaining=max(0, upgrade_limit - upgrade_used),
                next_reset_at=_reset_at(upgrade_oldest, _WINDOW_HOURS),
            ),
            destructive=_LimitDetail(
                limit=destructive_limit,
                used=destructive_used,
                remaining=max(0, destructive_limit - destructive_used),
                next_reset_at=destructive_reset_at,
            ),
        )
