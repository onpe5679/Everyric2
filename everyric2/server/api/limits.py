"""쿼터 조회 API — GET /api/limits/{video_id}.

실제 소비 지점(sync.py._check_action_limit)과 **같은 집계**(ActionLogRepository.count_recent)를
읽기 전용으로 노출한다. 확장이 생성·초기화 버튼을 누르기 전에 "오늘 몇 번 남았는지" 미리
보여주려는 용도라, 이 조회 자체는 action_logs에 아무것도 남기지 않는다 — 남기면 조회 자체가
한도를 깎아먹는 모순이 생긴다.

**집계 축은 이용자다**(운영자 결정 2026-08-10, docs/user-quota-spec.md §1). 경로에 video_id가
남아 있는 이유는 확장 무수정 조건 때문이다 — 확장은 지금 보고 있는 영상으로 이 경로를 부른다.
서버는 그 값을 형식 검증에만 쓰고 집계에는 쓰지 않는다: 같은 사람이 다른 영상으로 옮겨도
같은 개인 예산을 이어 쓴다. 이용자 식별자는 게이트웨이가 x-lyric-user 헤더로 넘긴다.

next_reset_at(2026-08-04, 확장 "내 기여/남은 횟수" 페이지 상단 안내용 보강, additive):
**실제 기전은 고정 시각(자정 등) 리셋이 아니라 행위별 롤링 24시간 창**(count_recent가
`now - hours` 이후만 센다) — "매일 0시에 리셋" 같은 문구는 사실과 다르다. 대신 이 창
안에서 가장 오래된 기록이 창 밖으로 나가는(=카운트가 실제로 줄어드는) **정확한 다음
시각**을 계산해서 준다. 그래서 root에 resets_daily 같은 뭉뚱그린 불리언은 안 붙였다 —
window_hours(이미 있음)로 "약 24시간 주기"라는 사실은 전달되고, next_reset_at이 그 주기의
구체적인 다음 시점을 준다.

link·upgrade 버킷(2026-08-04, 운영자 지시 — "남은 횟수"에 커버 잇기·정렬 업그레이드도
표시):

- **link**: 사람이 검색 시트에서 원곡 주소를 넣어 직접 잇는 행위(POST /api/sync/link,
  action="link"). 2026-08-10까지는 자동 후보 탐색(action="link_candidates")을 셌는데,
  그건 사용자가 요청한 적도 실패해도 볼 수도 없는 배경 요청이라 배지 라벨("커버 잇기")과
  내용이 어긋나 있었다 — 라벨이 가리키는 진짜 행위로 교체했다(같은 문서 §7). 옛 이름의
  기록은 액션 이름이 달라 자동으로 이 집계에서 빠진다.
- **upgrade**: action="upgrade" 전용 카운터. 어느 요청이 업그레이드인지는
  sync.py._is_upgrade_request가 판정한다(2026-08-10 재분류) — 그 전에는 확장의 깊이
  올리기 버튼이 파괴적 예산을 먹어 이 버킷이 영원히 만땅이었다.

destructive(2026-08-10): 초기화와 강제 재생성은 **하나의 합산 예산**이다(같은 문서 §2).
예전에는 두 행위가 각자 상한을 따로 소비하면서 표시만 max()로 뭉쳐, 한 예산처럼 보이는데
실제로는 두 예산이었다. 지금은 두 행위를 함께 세고, 회복 시각도 합계 기준(가장 이른 한
건이 창을 벗어나는 시점)으로 낸다.

한도**값**은 앞단이 준다(같은 문서 §6, 수신부는 api/quota_headers.py). 이 조회가 집행
경로(sync._check_action_limit)와 **같은 함수**로 값을 푸는 것이 계약이다 — 표시와 집행이
다른 값을 쓰면 "20/20인데 429"가 난다.
"""

from datetime import timedelta

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from everyric2.config.settings import get_settings
from everyric2.server.api.quota_headers import QuotaLimits, is_ops_actor
from everyric2.server.api.sync import (
    _VIDEO_ID_RE,
    DESTRUCTIVE_ACTIONS,
    _resolve_actor,
    request_limits,
)
from everyric2.server.db.connection import get_session
from everyric2.server.db.repository import ActionLogRepository

router = APIRouter(prefix="/api/limits", tags=["limits"])

# count_recent/oldest_recent 기본값(24)과 LimitsResponse.window_hours 기본값이 가리키는
# 같은 창 — 한 곳에서만 바꾸면 셋 다 같이 움직이게 상수로 뽑는다.
_WINDOW_HOURS = 24

# 버킷 하나만 무제한이고 나머지는 유한한 경우의 **표시용** 여유분.
#
# 확장에는 버킷별 무제한 표현이 없다(응답 계약: 무제한은 최상위 enforced=false 하나뿐).
# 그렇다고 무제한을 뜻하는 값(0·-1)을 limit에 실으면 확장이 `0 / 0` 소진으로 그린다 —
# 명시적으로 금지된 표현이다. 그래서 그 버킷만 "used + 여유분"으로 그려 **절대 소진으로
# 보이지 않게** 한다. 전 버킷이 무제한이면 이 값은 안 쓰이고 enforced=False로 나간다.
_UNLIMITED_DISPLAY_HEADROOM = 999


class _LimitDetail(BaseModel):
    limit: int
    used: int
    # 확장이 그대로 그리는 값이라 **음수가 되어서는 안 된다**(0으로 floor) — 음수를 보내면
    # 화면에 그대로 표시되고 소진 표시(빨간 강조)도 안 붙는다. 한도를 낮추거나 어드민이
    # 상한을 넘긴 뒤에는 used > limit이 정상적으로 발생한다.
    remaining: int
    # 이 카운트가 다음으로 줄어드는 정확한 시각(UTC, tz 미표기 — 기존 created_at.isoformat()
    # 관례와 동일). used가 0(깎을 게 없음)이면 None. 계산 근거는 아래 _reset_at 참고.
    next_reset_at: str | None = None


class LimitsResponse(BaseModel):
    # 필드 이름 네 개(generate·link·upgrade·destructive)는 확장이 그리는 이름이자 앞단
    # 한도 헤더의 키(quota_headers.BUCKETS)다 — 세 곳이 같은 어휘를 쓴다. 이름을 바꾸면
    # 확장 화면과 앞단 계약이 함께 깨진다(test_user_quota.py가 그 일치를 못박는다).
    #
    # enforced=False면 확장은 각 버킷을 "무제한 (사용 N회)"로 그린다. 세 경우에 False다:
    #   ① admin_api_key 미설정(로컬 단일사용자 기본) — 한도 자체가 없다.
    #   ② 면제 이용자 — 어드민 키(docs/user-quota-spec.md §5)와 운영 작업 식별자(§8).
    #      거절은 면제받지만 사용량은 기록되므로 used는 실제값이다. 강제(enforced=True)
    #      형태로 소진값을 주면 확장이 면제 이용자를 차단 상태로 그린다 — 확장 무수정
    #      조건에서 표시와 실제를 맞추는 유일한 방법이다.
    #   ③ 앞단이 **모든** 버킷을 무제한으로 준 경우(§6). **상한 0을 "무제한"의 뜻으로
    #      쓰지 않는다**: 확장은 그것을 0/0 소진으로 그린다.
    enforced: bool
    generate: _LimitDetail
    # 커버 잇기 — 사람이 직접 잇는 POST /api/sync/link(action="link"). 2026-08-10에 자동
    # 후보 탐색에서 교체했다(모듈 docstring 참고).
    link: _LimitDetail
    # 정렬 업그레이드(분석 깊이 올리기) — action="upgrade" 독립 집계.
    upgrade: _LimitDetail
    # 초기화 + 강제 재생성 **합산** 예산. 링크 해제는 포함하지 않는다 — 복구 행위라 한도를
    # 걸지 않는다(sync.DESTRUCTIVE_ACTIONS 주석 참고).
    destructive: _LimitDetail
    window_hours: int = _WINDOW_HOURS


def _reset_at(oldest: object, hours: int) -> str | None:
    """가장 오래된 기록(oldest_recent 결과)이 창 밖으로 나가는 시각 — 그 순간이 카운트가
    실제로 줄어드는 다음 시각이다. 기록이 없으면(=count 0) 줄어들 게 없으니 None."""
    if oldest is None:
        return None
    return (oldest + timedelta(hours=hours)).isoformat()


def _detail(limit: int | None, used: int, oldest: object) -> _LimitDetail:
    """버킷 하나 — remaining은 절대 음수가 되지 않는다(응답 계약).

    limit=None(무제한)은 값으로 표현할 수 없으므로 "used + 여유분"으로 그린다
    (_UNLIMITED_DISPLAY_HEADROOM 주석 참고). 0을 무제한의 뜻으로 쓰지 않는다."""
    shown = used + _UNLIMITED_DISPLAY_HEADROOM if limit is None else limit
    return _LimitDetail(
        limit=shown,
        used=used,
        remaining=max(0, shown - used),
        next_reset_at=_reset_at(oldest, _WINDOW_HOURS),
    )


@router.get("/{video_id}", response_model=LimitsResponse)
async def get_limits(
    video_id: str,
    x_api_key: str | None = Header(default=None),
    x_lyric_user: str | None = Header(default=None),
    x_lyric_limits: str | None = Header(default=None),
):
    """이 이용자의 24시간 쿼터 현황.

    video_id는 형식 검증만 한다 — 집계는 이용자 축이다(모듈 docstring). 한도를 강제하는
    배포에서 이용자 식별자가 없으면 400이다(sync._resolve_actor와 같은 fail-closed 규칙 —
    소비 경로는 거절하는데 조회만 통과시키면 화면이 실제와 다른 숫자를 그린다).

    한도값은 앞단이 준 x-lyric-limits를 그대로 쓴다(§6) — 집행 경로와 **같은 함수**
    (sync.request_limits)로 푸는 것이 계약이다. 표시와 집행이 다른 값을 쓰면 "20/20인데
    429"가 난다. 못 받으면 서버 기본값으로 내려앉고 그 사실이 경고로 남는다(집행 경로와
    같은 카운터를 공유하므로 샘플링 덕에 로그가 이 조회로 덮이지 않는다). 한도를 아예
    강제하지 않는 배포에서는 경고를 끈다 — 거기엔 무력화될 티어가 없다.
    """
    if not _VIDEO_ID_RE.match(video_id):
        raise HTTPException(status_code=422, detail="invalid video_id")

    server = get_settings().server
    enforced = bool(server.admin_api_key)
    limits = request_limits(x_lyric_limits, warn=enforced)

    if not enforced:
        # 한도가 아예 없는 배포 — 셀 것도 없다(소비 지점도 로그를 남기지 않는다).
        return LimitsResponse(
            enforced=False,
            generate=_detail(limits.generate, 0, None),
            link=_detail(limits.link, 0, None),
            upgrade=_detail(limits.upgrade, 0, None),
            destructive=_detail(limits.destructive, 0, None),
        )

    actor = _resolve_actor(x_lyric_user)
    # 거절만 면제받고 사용량은 기록되는 이용자들 — 어드민(§5)과 운영 작업 식별자(§8).
    # 그 사실을 그대로 낸다: "강제 아님 + 실제 사용량"(강제 형태로 소진값을 주면 확장이
    # 면제 이용자를 차단 상태로 그린다).
    exempt = x_api_key == server.admin_api_key or is_ops_actor(actor)

    async with get_session() as session:
        repo = ActionLogRepository(session)
        generate_used = await repo.count_recent("generate", actor, hours=_WINDOW_HOURS)
        generate_oldest = await repo.oldest_recent("generate", actor, hours=_WINDOW_HOURS)
        link_used = await repo.count_recent("link", actor, hours=_WINDOW_HOURS)
        link_oldest = await repo.oldest_recent("link", actor, hours=_WINDOW_HOURS)
        upgrade_used = await repo.count_recent("upgrade", actor, hours=_WINDOW_HOURS)
        upgrade_oldest = await repo.oldest_recent("upgrade", actor, hours=_WINDOW_HOURS)
        # 합산 예산이므로 두 행위를 한 번에 센다 — max()가 아니라 합계다. 회복 시각도
        # 합계 기준: 어느 한 건만 창을 벗어나도 값이 줄기 때문에 **가장 이른** 기록이 답이다.
        destructive_used = await repo.count_recent(DESTRUCTIVE_ACTIONS, actor, hours=_WINDOW_HOURS)
        destructive_oldest = await repo.oldest_recent(
            DESTRUCTIVE_ACTIONS, actor, hours=_WINDOW_HOURS
        )

        return LimitsResponse(
            enforced=_enforced(limits, exempt=exempt),
            generate=_detail(limits.generate, generate_used, generate_oldest),
            link=_detail(limits.link, link_used, link_oldest),
            upgrade=_detail(limits.upgrade, upgrade_used, upgrade_oldest),
            destructive=_detail(limits.destructive, destructive_used, destructive_oldest),
        )


def _enforced(limits: QuotaLimits, *, exempt: bool) -> bool:
    """이 응답을 "강제"로 낼 것인가.

    False가 되는 경우는 둘 뿐이다: ① 면제 이용자(어드민·운영 작업 — §5/§8), ② **모든**
    버킷이 무제한(앞단이 admin 티어에 주는 값 — §6). 일부만 무제한일 때 False로 내면 한도가
    살아 있는 버킷까지 확장이 무제한으로 그려 429가 예고 없이 튀어나온다 — 그 경우는 True를
    유지하고 무제한 버킷만 소진되지 않는 숫자로 그린다(_detail)."""
    return not exempt and not limits.all_unlimited
