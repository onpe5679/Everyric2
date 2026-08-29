import asyncio
import copy
import logging
import re
import time
from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Query
from pydantic import BaseModel, Field

from everyric2.config.settings import get_settings
from everyric2.server import media_cache, song_link, title_match

# 앞단(게이트웨이) 쿼터 헤더의 유일한 수신부 — 이름·형식·무제한 표현이 전부 거기 있다.
# 게이트웨이가 확정되면 이 모듈이 아니라 quota_headers.py를 고친다(docs §6).
from everyric2.server.api.quota_headers import QuotaLimits, is_ops_actor, resolve_limits

# 리스 회수는 워커 API가 소유한다 (레지스트리가 거기 있다). api/worker는 api/sync를
# 임포트하지 않으므로 순환이 없다 — 요청마다 함수 내 임포트를 반복하지 않게 최상위로 둔다.
from everyric2.server.api.worker import reclaim_expired_leases
from everyric2.server.db.connection import get_session
from everyric2.server.db.models import SyncFeedback, SyncResult
from everyric2.server.db.repository import (
    ActionLogRepository,
    JobRepository,
    LinkJobRepository,
    SyncLinkRepository,
    SyncRepository,
    SyncResultVersionRepository,
    SyncViewRepository,
    TranslationLayerRepository,
    VideoOffsetRepository,
    hash_lyrics,
)
from everyric2.server.text_fingerprint import (
    align_translation_lines,
    lines_fingerprint,
    normalize_line,
)

logger = logging.getLogger(__name__)

# 레거시 ko 번역 판정(F4, 2026-08-04 감사)용 — 완성형 한글 U+AC00~D7A3. 다른 모듈의
# 같은 계열 정규식(worker._HANGUL_CHAR_RE 등)과 같은 범위다.
_HANGUL_RE = re.compile("[가-힣]")


# ── 비싼 행위의 일일 상한 기본값 (action_logs 기반, 이용자·행위별 24시간) ──────────
#
# 파괴적 행위(daily_destructive_limit, 기본 4회 — 초기화+강제 재생성 합산)와 **같은 기전**을
# 상한만 달리해 재사용한다. 설정으로 올릴 후보라 값의 근거를 붙여 모듈 상수로 둔다.
#
# **집계 축은 이용자다**(운영자 결정 2026-08-10, docs/user-quota-spec.md §1). 예전에는
# (행위, 영상)이라 한 사람이 어떤 영상에서 상한을 다 쓰면 그 영상을 처음 여는 다른 사람이
# 0회로 시작했다. 아래 상수들의 "하루 N번"은 이제 **한 사람이 전체 영상을 통틀어** N번이다.
#
# 🔴 **이 값들은 기본값이다**(같은 문서 §6, 2026-08-10). 앞단(게이트웨이)이 이용자 티어에
# 맞는 상한을 헤더로 보내면 그 값이 이긴다 — 여기 상수는 앞단 값을 못 받았을 때만 쓰인다
# (_default_limits → quota_headers.resolve_limits). 티어에 준 값이 이 상수보다 크면
# 예전에는 작은 쪽이 먼저 걸려 영원히 도달할 수 없었다.

# POST /api/sync/generate — 이 경로에는 한도가 전혀 없었다. 가사를 한 글자만 바꾸면 매번 새
# lyrics_hash가 되어 캐시·합류를 모두 비켜 새 GPU 잡이 생긴다. 다만 이건 제품의 **주 경로**라
# 파괴적 행위와 같은 수준으로 잡으면 정상 사용이 망가진다: 오탈자 수정, 다른 가사 판본 시도,
# 실패 후 재시도로 한 영상에 여러 번 생성하는 것은 흔하다. 20회/24h는 그 여유를 크게 남기면서
# 무한 반복은 잘라낸다.
# 캐시 히트·진행 중 잡 합류는 GPU를 쓰지 않으므로 세지 않는다 (검사 위치가 잡 생성 직전).
DAILY_GENERATE_LIMIT = 20

# POST /api/sync/link — 사람이 검색 시트에서 원곡 주소를 넣어 **직접** 연결하는 행위.
# 확장 배지의 "커버 잇기"가 가리키는 것이 이것이다(docs/user-quota-spec.md §7).
#
# 2026-08-10까지 이 자리에 있던 것은 자동 후보 탐색(link-candidates)의 상한이었다. 그건
# 사용자가 요청한 적도, 실패해도 볼 수도 없는 배경 요청이라 라벨과 내용이 어긋났다 —
# 라벨이 가리키는 진짜 행위로 교체한다(_dispatch_candidate_followup 주석에 상한 제거 근거).
#
# 20회/24h: 이 경로는 GPU를 쓰지 않는다(DB upsert 한 번). 사람이 하루에 스무 곡을 손으로
# 이어 붙이는 것은 정상 사용의 상한 밖이고, 그 이상은 자동화된 반복이다. generate와 같은
# 자릿수로 둬 "한 사람이 하루에 손댈 수 있는 곡 수"라는 감각을 한 값으로 맞춘다.
DAILY_LINK_LIMIT = 20

# 하나의 예산을 나눠 쓰는 파괴적 행위들 — 초기화(reset)와 강제 재생성(regenerate).
# 예전에는 각자 daily_destructive_limit을 따로 소비해 실제로는 상한의 2배까지 가능했고,
# 화면에는 둘 중 큰 값 하나(max)만 보여 "한 예산처럼 보이는데 두 예산"이었다
# (docs/user-quota-spec.md §2). 지금은 두 행위의 **합계**가 하나의 상한을 쓴다.
#
# **링크 해제는 여기에 없다**(같은 문서 §2). 링크 해제는 잘못 연결된 링크를 되돌리는 복구
# 행위라, 한도로 막으면 잘못된 상태가 고쳐지지 않은 채 남는다 — 한도는 비용·가용성을
# 지키는 수단이지 정확성을 지키는 수단이 아니다.
DESTRUCTIVE_ACTIONS = ("reset", "regenerate")

# 가사 하한 — 이 줄 수를 못 넘기는 생성 요청은 잡을 만들지 않고 400으로 거절한다.
#
# 왜: lyrics에 최소 길이 제약이 없어 빈 가사가 무검증 통과하면 정렬 결과가 0줄이고, 그 0줄이
# completed로 저장돼 이후 같은 lyrics_hash는 **캐시 히트로 영구히 0줄**을 돌려준다
# (GET /api/sync/{id}가 found:true, timestamps:[]를 낸다 → 사용자에겐 "싱크가 있다"고 표시된
# 채 가사가 영원히 안 나오고, 같은 가사로는 다시 생성할 수도 없다).
#
# 값이 자막 경로(services.youtube_captions.MIN_LYRIC_LINES=3)와 다른 이유: 그 3줄은 «[음악]»
# 류 효과음 표기뿐인 자동자막을 걸러내는 값이고, /generate의 가사는 사용자가 의도해 붙여넣은
# 것이라 짧은 후크 한 줄도 정당한 입력이다. 영구 0줄 봉인은 1줄 하한으로 완전히 막힌다.
MIN_LYRICS_LINES = 1

# 가사 한 줄로 인정하는 조건 — 공백·구두점만 있는 줄은 세지 않는다 ("...\n---" 같은 입력이
# 하한을 통과해 0줄 정렬로 가는 것을 막는다). \w는 유니코드라 한글·가나·한자를 센다.
_LYRIC_WORD_RE = re.compile(r"\w")


def _validate_lyrics(lyrics: str) -> None:
    """생성 요청의 가사 하한 검사 — 미달이면 400 + 무엇을 하면 되는지 알리는 한국어 사유."""
    usable = sum(1 for line in lyrics.splitlines() if _LYRIC_WORD_RE.search(line))
    if usable < MIN_LYRICS_LINES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"가사에 글자가 있는 줄이 {MIN_LYRICS_LINES}줄 이상 필요해요 "
                "(지금은 비어 있거나 공백·기호뿐이에요). 가사를 붙여넣고 다시 시도해 주세요."
            ),
        )


# 429 사유에 쓰는 행위별 한국어 이름 — 이용자에게 "무엇의 한도인지"를 알린다.
# 파괴적 두 행위는 하나의 예산이므로 같은 문구를 쓴다(합산이라는 사실이 사유에 드러난다).
_ACTION_LABELS = {
    "generate": "싱크 생성",
    "upgrade": "정렬 업그레이드",
    "link": "커버 잇기",
    "reset": "초기화·강제 재생성",
    "regenerate": "초기화·강제 재생성",
}

# 행위 → 예산 버킷. 버킷 이름은 세 곳에서 같은 어휘여야 한다 — GET /api/limits 응답 필드
# (확장이 그리는 이름), 앞단 한도 헤더의 키(quota_headers.BUCKETS), 그리고 여기.
_ACTION_BUCKET = {
    "generate": "generate",
    "upgrade": "upgrade",
    "link": "link",
    "reset": "destructive",
    "regenerate": "destructive",
}

# 한 버킷을 **여러 행위가 나눠 쓰는** 경우 그 전체 집합. 검사는 합계로 하고 기록은 실제
# 행위 이름으로 남긴다 — 어느 행위였는지의 정보를 합산 예산에서도 잃지 않는다.
_BUCKET_ACTIONS: dict[str, tuple[str, ...]] = {"destructive": DESTRUCTIVE_ACTIONS}


def _default_limits() -> dict[str, int | None]:
    """앞단이 값을 안 줬을 때 내려앉을 **이 서버의** 기본값 — 버킷별(docs §6).

    모듈 상수를 호출 시점에 읽는다(테스트가 상수를 갈아끼워 경계를 재현한다)."""
    server = get_settings().server
    return {
        "generate": DAILY_GENERATE_LIMIT,
        "link": DAILY_LINK_LIMIT,
        "upgrade": server.daily_upgrade_limit,
        "destructive": server.daily_destructive_limit,
    }


def request_limits(quota_header: str | None, *, warn: bool = True) -> QuotaLimits:
    """이 요청에 집행할 버킷별 상한 — 앞단 값이 있으면 그 값, 없으면 기본값(docs §6).

    앞단 헤더의 이름·형식은 quota_headers.py 한 곳에 갇혀 있다. 여기는 **기본값을 대는
    자리**일 뿐이라 게이트웨이가 확정돼도 손댈 것이 없다."""
    return resolve_limits(quota_header, _default_limits(), warn=warn)


def _resolve_actor(actor: str | None) -> str:
    """한도 집행에 쓸 이용자 식별자를 확정한다 — **한도를 강제하는 배포에서만** 부른다.

    값은 게이트웨이가 x-lyric-user 헤더로 넘긴다(발급 키 이용자는 불변 키 ID, 익명
    이용자는 게이트웨이가 만든 솔트 해시 — docs/user-quota-spec.md §1). 서버는 게이트웨이
    뒤에만 있어야 하므로, 공개 배포에서 이 헤더가 없는 요청은 정상 경로로 도달할 수 없다.

    **부재·빈 값은 거절한다(fail-closed).** 영상 단위나 공용 이용자로 폴백하면 "한 사람의
    사용이 다른 사람을 정지시키는" 원래 문제로 그대로 되돌아간다(같은 문서 §1). 통과시키면
    한도가 아예 없는 우회로가 된다.

    상태 코드는 400이다: 401/403은 "당신의 자격 증명이 없거나 부족하다"는 뜻이라 확장이
    사용자에게 API 키 입력을 안내하게 되는데, 이 헤더는 사용자가 넣는 값이 아니라 앞단
    인프라가 붙이는 값이다 — 실제로 잘못된 것은 **요청이 도달한 경로**이므로 잘못된 요청
    (400)이 정직한 분류다. 어드민도 예외가 아니다: 어드민 면제는 §5에서 "거절만 면제,
    기록은 남긴다"로 정의되는데, actor가 없으면 기록 자체를 남길 수 없다.
    """
    # 문자열이 아닌 값은 없는 것으로 본다 — 라우트 코루틴을 직접 await하는 테스트 하네스가
    # 안 넘긴 인자에 Header(None) 기본값 객체를 그대로 실어 보낸다(quota_headers 참고).
    actor_id = actor.strip() if isinstance(actor, str) else ""
    if not actor_id:
        raise HTTPException(
            status_code=400,
            detail="이용자 식별 정보가 없는 요청이에요. 잠시 후 다시 시도해 주세요.",
        )
    return actor_id


async def _check_action_limit(
    session,
    action: str,
    video_id: str,
    actor: str | None,
    api_key: str | None,
    quota_header: str | None = None,
    *,
    record: bool = True,
) -> None:
    """(action, actor) 24시간 횟수 상한 — admin_api_key가 설정된 배포에서만.

    키가 미설정이면(로컬 사용) 제한 없음 — actor도 요구하지 않는다(기존 동작 유지).
    설정돼 있으면 actor를 요구하고(_resolve_actor, fail-closed), 초과면 429다.

    **상한값은 앞단이 준다**(docs/user-quota-spec.md §6). quota_header가 그 값을 실어 오고,
    없으면 이 서버의 기본값으로 내려앉는다(request_limits → quota_headers.resolve_limits가
    그 사실을 경고로 남긴다 — 조용한 티어 무력화 금지). 검사할 상한은 행위가 속한 **버킷**의
    값이다(_ACTION_BUCKET).

    **어드민은 거절만 면제받고 사용량은 기록된다**(같은 문서 §5). 예전에는 검사 자체를 조기
    return해 기록도 안 남아, 소유자 화면에는 실제로 쓴 뒤에도 한도가 항상 만땅으로 보였다
    (실측 2026-08-10). 지금은 카운트 비교만 건너뛴다. **운영 작업 식별자**(§8, `ops:` 접두)도
    같은 취급이다 — 대량 인제스트는 원래 상한 밖의 작업이지만 기록은 남는다.

    record=False: 검사만 하고 기록은 호출부가 나중에(_record_action) 남긴다. 실제로 일어난
    일이 없으면 예산을 소비하지 않아야 하는 경로(초기화 멱등성, 같은 문서 §4)를 위한 것이다.

    **한계(의도적)**: 익명 이용자의 식별자는 게이트웨이가 만든 값이라 그 층의 강도를
    넘어설 수 없다. 그 대신 이 층은 "누가 봐도 같은 사람"인 반복을 정확히 막는다 — 예전
    영상 단위 집계는 그조차 못 했다(영상만 바꾸면 예산이 새로 생겼다).
    """
    server = get_settings().server
    if not server.admin_api_key:
        return
    actor_id = _resolve_actor(actor)
    log_repo = ActionLogRepository(session)
    bucket = _ACTION_BUCKET[action]
    # 무제한(None)은 거절만 건너뛰고 기록은 남긴다 — 어드민과 같은 이유로, 사용량 표시가
    # 사실과 어긋나면 안 된다. 앞단의 "무제한"도 설정의 `0`(한도 비활성)도 여기서는 같은
    # None 하나다(quota_headers.QuotaLimits 참고).
    limit = request_limits(quota_header).get(bucket)
    exempt = api_key == server.admin_api_key or is_ops_actor(actor_id)
    if limit is not None and not exempt:
        used = await log_repo.count_recent(_BUCKET_ACTIONS.get(bucket, (action,)), actor_id)
        if used >= limit:
            label = _ACTION_LABELS.get(action, action)
            raise HTTPException(
                status_code=429,
                detail=(
                    f"{label} 일일 한도({limit}회/24시간)에 도달했어요. "
                    "가장 오래된 기록이 24시간을 지나면 다시 시도할 수 있어요."
                ),
            )
    if record:
        await log_repo.log(action, video_id, actor_id)


async def _record_action(
    session, action: str, video_id: str, actor: str | None, api_key: str | None
) -> None:
    """한도 검사를 이미 통과한 행위를 **실제로 일어난 뒤에** 기록한다.

    초기화 멱등성(docs/user-quota-spec.md §4): 지울 것이 없는 초기화는 GPU도 데이터도 건드리지
    않으므로 예산을 소비하면 안 된다. 그래서 검사(_check_action_limit(record=False))와 기록을
    분리하고, 실제로 지운 것이 있을 때만 이 함수를 부른다. api_key는 받지만 기록 여부를
    가르지 않는다 — 어드민도 기록 대상이다(§5).
    """
    if not get_settings().server.admin_api_key:
        return
    await ActionLogRepository(session).log(action, video_id, _resolve_actor(actor))


async def _check_destructive_limit(
    session,
    action: str,
    video_id: str,
    actor: str | None,
    api_key: str | None,
    quota_header: str | None = None,
    *,
    record: bool = True,
):
    """파괴적 행위 일일 한도 — 초기화와 강제 재생성이 **하나의 합산 예산**을 쓴다
    (destructive 버킷, 기본 4회/24h — 운영자 결정 2026-08-10, 종전 실질 총량 유지).

    예전에는 두 행위가 각자 상한을 따로 소비해 실제로는 상한의 2배까지 가능했다
    (docs/user-quota-spec.md §2). 두 행위가 같은 버킷(_ACTION_BUCKET)에 속해 _BUCKET_ACTIONS로
    함께 세이므로 그 구멍이 닫힌다."""
    await _check_action_limit(
        session, action, video_id, actor, api_key, quota_header, record=record
    )


async def _check_upgrade_limit(
    session,
    video_id: str,
    actor: str | None,
    api_key: str | None,
    quota_header: str | None = None,
) -> None:
    """정렬 업그레이드 일일 한도 — upgrade 버킷(기본 daily_upgrade_limit, 10회/24h).

    운영자 결정(2026-08-04): "업그레이드도 당연히 생성 쿼터랑은 별개여야지" — 이미 만든
    결과를 더 정밀하게 다시 뽑는 행위(fast→medium→heavy, 최대 2단계)가 새 싱크를 만드는
    generate 예산을 깎으면 안 된다. action 이름을 "generate"가 아니라 "upgrade"로 독립시켜
    ActionLog 집계 자체를 분리한다 — GET /api/limits가 이 이름을 그대로 읽어 upgrade
    버킷을 낸다(limits.py 참고).

    **어느 요청이 업그레이드인지는 _is_upgrade_request가 판정한다**(2026-08-10 재분류,
    docs/user-quota-spec.md §3): 확장의 API 래퍼가 재생성 요청에 force를 무조건 붙이는
    바람에 이 카운터는 도입 이후 한 번도 기록되지 않았고(실측 2026-08-10 전 기간 0건),
    업그레이드가 파괴적 예산을 대신 먹고 있었다.
    """
    await _check_action_limit(session, "upgrade", video_id, actor, api_key, quota_header)

router = APIRouter(prefix="/api/sync", tags=["sync"])

# 유튜브 video_id 형식 (captions.py와 동일 규칙) — 무제한 길이 문자열이 그대로 쿼리
# 파라미터/저장 키로 흘러드는 것을 차단한다
_VIDEO_ID_PATTERN = r"^[A-Za-z0-9_-]{11}$"
_VIDEO_ID_RE = re.compile(_VIDEO_ID_PATTERN)

# 생성 API의 check-then-act(기존 싱크/활성 잡 확인 → 잡 생성)를 직렬화한다 —
# 동시 다중 탭 요청이 근소하게 겹치면 같은 (video_id, lyrics_hash) 잡이 중복 생성됐다.
# 단일 프로세스 서버라 프로세스 내 락으로 충분하다.
_CREATE_LOCK = asyncio.Lock()


def _validate_video_id(video_id: str) -> None:
    if not _VIDEO_ID_RE.match(video_id):
        raise HTTPException(status_code=422, detail="invalid video_id")


async def _dispatch_job(
    job_id: str, background_tasks: BackgroundTasks, await_line_meta: bool = False
) -> None:
    """생성 잡을 처리 경로에 넘긴다.

    local_worker면 기존처럼 인프로세스로 처리(add_task → process_job)한다. False면 GPU
    없는 API 전용 서버로 보고, add_task 없이 status=queued로만 마킹해 원격 워커가 클레임
    하도록 둔다 (스태시 적재는 호출부가 이미 마쳤고, queue_position 표시도 그대로 동작).

    await_line_meta는 "번역·독음(line_meta)이 잡 생성 뒤에 따로 온다"는 예고다. 인프로세스
    워커는 다운로드·보컬 분리를 먼저 돌리고 정렬 진입 직전에 기다리므로 그 시간이 번역과
    겹친다. 원격 워커는 클레임 시점의 스태시만 받아 도중에 line_meta를 받을 수 없으니,
    대신 **큐 진입 자체를 line_meta 도착까지 늦춰** 조용히 원문 정렬로 떨어지는 것을 막는다
    (병렬 이득은 없고 기존과 동일한 총 소요 — 품질 회귀가 없는 쪽을 고른다)."""
    if get_settings().server.local_worker:
        from everyric2.server.worker import process_job, stash_line_meta_wait

        if await_line_meta:
            stash_line_meta_wait(job_id)
        background_tasks.add_task(process_job, job_id)
    elif await_line_meta:
        background_tasks.add_task(_queue_after_line_meta, job_id)
    else:
        async with get_session() as session:
            await JobRepository(session).update_status(job_id, "queued", progress=0)


async def _queue_after_line_meta(job_id: str) -> None:
    """line_meta가 도착(또는 상한 초과)한 뒤에 잡을 원격 워커 큐에 올린다.

    대기 중에는 status=processing + stage="번역 대기"로 둔다 — 확장이 무엇을 기다리는지
    보이고, queued가 아니라 워커의 get_oldest_queued에도 잡히지 않는다. 대기 중 취소되면
    큐에 올리지 않고 끝낸다. 상한은 유한하므로(LINE_META_WAIT_SEC) 확장이 아무것도 보내지
    않아도 잡은 결국 큐로 올라가 원문 정렬로 완주한다.

    **말미의 queued 쓰기는 조건부다.** 취소 확인(_consume_cancel)과 그 쓰기 사이에 취소
    요청이 들어오면 무조건 쓰기는 방금 failed가 된 잡을 queued로 되살린다 → 워커가 물어
    processing이 되고, 취소된 잡은 워커가 fail을 제출하지 않아 processing에 남고, 만료
    스윕이 다시 queued로 돌려 무한 진동한다. 아직 대기 중(processing)일 때만 쓴다."""
    from everyric2.server.worker import (
        LINE_META_WAIT_SEC,
        LINE_META_WAIT_STAGE,
        JobCancelled,
        _consume_cancel,
        await_line_meta_arrival,
    )

    async with get_session() as session:
        await JobRepository(session).update_status(
            job_id, "processing", progress=48, stage=LINE_META_WAIT_STAGE
        )
    try:
        arrived = await await_line_meta_arrival(job_id, LINE_META_WAIT_SEC)
    except JobCancelled:
        await _consume_cancel(job_id)
        return
    if await _consume_cancel(job_id):
        return
    if not arrived:
        logger.info(
            "Job %s: line_meta did not arrive within %.0fs; queueing for original-text alignment",
            job_id,
            LINE_META_WAIT_SEC,
        )
    async with get_session() as session:
        # 위 취소 확인 이후에 들어온 취소도 여기서 이긴다 — processing(대기 중)일 때만 쓴다
        queued = await JobRepository(session).update_status_if(
            job_id, "queued", expected=("processing",), progress=0
        )
    if not queued:
        # 대기 중 종결된 잡(취소·실패) — 되살리지 않는다. 되살리면 워커가 물고, 취소된 잡은
        # fail을 제출하지 않아 processing에 남고, 만료 스윕이 다시 queued로 돌려 진동한다.
        logger.info("Job %s: no longer waiting when line_meta finished; not queued", job_id)


class SyncLookupResponse(BaseModel):
    found: bool
    sync_id: str | None = None
    timestamps: list[dict[str, Any]] | None = None
    lyrics_source: str | None = None
    quality_score: float | None = None
    audio_hash: str | None = None
    language: str | None = None
    # 결함 #5 additive 필드 — 구버전 확장은 모르는 필드를 무시하므로 하위호환 유지.
    # engine_variant: MMS 강제 폴백 등 엔진 변형("mms" | None). engine_version: 이 싱크를
    # 만든 정렬 스택 식별자(models.ENGINE_VERSION) — NULL이면 이 컬럼이 생기기 전 구세대.
    engine_variant: str | None = None
    engine_version: str | None = None
    created_at: str | None = None
    # 곡 단위 진단 정보 (star 흡수 구간, VAD 발성 구간) — 확장 디버그 스트립용
    debug: dict[str, Any] | None = None
    # 가사 출처 표기 (예: 보카로 가사 위키) — 푸터 병기용
    attribution: dict[str, Any] | None = None
    # 곡 템포 {bpm, beat_offset} — 가라오케 레인 마디 창/비트 격자용
    tempo: dict[str, Any] | None = None
    # 곡 키 {tonic, mode, name, confidence} — 멜로디 분석의 K-S 추정, 레인 표시용
    key: dict[str, Any] | None = None
    # 다른 영상의 싱크를 오프셋과 함께 빌려 왔을 때만 채워진다 (자기 싱크가 있으면 None).
    # 클라이언트가 링크 상태 표시·해제 버튼을 띄우는 데 쓴다.
    linked: dict[str, Any] | None = None
    # 이 영상에 저장된 사용자 싱크 오프셋(초) — 클라이언트가 재생 시점에 적용.
    # 링크로 빌려온 싱크도 보는 영상 기준이라 영상마다 따로 저장된다.
    user_offset: float | None = None
    # 세그먼트 translation이 실제로 어느 언어인지 — lang 쿼리 파라미터를 준 요청에만
    # 의미가 있다. lang 없이 조회하면 항상 None(구버전 응답과 필드 단위 동일 유지).
    # lang="ko"인데 레이어가 없으면 레거시 저장분이 ko라는 이행 가정으로 "ko"를 낸다
    # (세그에 번역이 하나도 없으면 None). 레이어가 없는 비ko lang은 번역을 비우고 None.
    translation_lang: str | None = None
    # 이 싱크(지문 기준)로 실제 서빙 가능한 번역 언어 목록 — 레이어 테이블에 존재하는
    # target_lang + (세그에 레거시 ko 번역이 있으면 "ko" 포함), 중복 제거·정렬. lang
    # 지정/미지정 요청 모두 채운다 — 추가 필드라 구버전 클라이언트는 무시하면 그만이다.
    available_langs: list[str] | None = None
    # 언어별 번역 배열 — {lang: [세그 순서로 정렬된 번역, ...]}. 각 배열은 timestamps와
    # 길이가 같고, 매칭 안 된 세그는 None. 처음 로딩부터 (있는) 모든 언어를 다 받아두려는
    # 확장이 lang= 재조회 없이 즉시 언어를 전환할 수 있게 한다. 이 지문에 이미 있는
    # 레이어(+세그 레거시 ko) 뿐 아니라 크로스 지문 이관(다른 지문의 사람 번역을 지금
    # 세그에 재정렬해도 커버리지가 되는 언어)도 포함한다 — lang 지정/미지정 요청 모두
    # 채운다. 최대 _TRANSLATIONS_BY_LANG_MAX_LANGS개까지만(응답 크기 상한). found=False면
    # None. lang 파라미터의 기존 동작(legacy 슬롯 오버레이·translation_lang)은 이 필드와
    # 무관하게 그대로다 — 추가 필드라 구버전 클라이언트는 무시하면 그만이다.
    translations_by_lang: dict[str, list[str | None]] | None = None
    # 곡 단위 추임새 후보 [(start, end), ...] — 가사가 주장하지 않은 가창 구간(새 정렬
    # 스택의 display_fixes.adlib_candidates 전용, 레거시 스택은 이 필드를 채우지 않는다).
    # 판정이 아니라 후보다 — 화면에 띄워 귀로 확인하는 용도. additive 필드라 구버전
    # 확장은 무시하면 그만이다.
    adlib: list[list[float]] | None = None
    # F5(2026-08-04 감사, additive) — 이번 조회가 실제로 서빙한 번역의 출처. translation_
    # layers.attribution/origin은 저장만 되고 조회 응답에 안 나가 확장이 번역 출처 배지를
    # 세울 손잡이가 없었다. lang이 있고 그 언어 번역이 실제로 채워졌을 때만 채운다 —
    # translation_origin은 "llm"|"wiki"|"manual"|"caption"|"legacy"(TranslationLayer.origin과
    # 같은 값, 레이어 없이 세그 레거시 번역으로 서빙한 legacy ko 경로는 "legacy"). 레이어를
    # 못 찾았거나 lang 미지정이면 둘 다 None. 크로스 지문 이관 서빙은 이관 **원본** 레이어의
    # attribution/origin을 그대로 낸다(실제로 그 출처의 번역이기 때문).
    translation_attribution: dict[str, Any] | None = None
    translation_origin: str | None = None


class SyncPreviousVersionResponse(BaseModel):
    """이 영상 자기 싱크의 직전(재처리로 덮어써지기 전) 버전 — 확장의 A/B 고스트 비교용.

    SyncLookupResponse와 최대한 같은 모양을 쓴다(같은 필드명: timestamps=세그먼트 리스트,
    created_at=그 세대의 원래 생성 시각). 없으면(최초 생성뿐이었거나 아직 재처리된 적이
    없으면) found=false — GET /api/sync/{video_id}의 미존재 관례(404가 아니라 found=false)를
    그대로 따른다. 롤백 API는 이번 스코프 밖이라 이 응답에는 sync_id/롤백 액션이 없다."""

    found: bool
    timestamps: list[dict[str, Any]] | None = None
    language: str | None = None
    quality_score: float | None = None
    # 스냅샷된 행이 원래 만들어진 시각 (교체 전 세대의 생성 시각)
    created_at: str | None = None
    # 이 스냅샷이 찍힌(=재처리가 그 세대를 덮어쓴) 시각
    replaced_at: str | None = None
    # 스냅샷된 행의 lyrics_hash — 지금 화면의 싱크(현재 sync_results 행)와 대조해 "가사가
    # 같은 재정렬"(재생성 버튼 — A/B 스택 비교가 성립)인지 "가사 자체가 다른 새 생성"
    # (붙여넣기/검색 생성 — 줄이 대응하지 않아 비교가 무의미)인지 **확장이** 가리는 재료다.
    # 서버는 판정하지 않는다 — additive 필드라 구버전 확장은 무시하면 그만이다.
    lyrics_hash: str | None = None
    # 스냅샷된 세대의 엔진 정체 — 고스트 비교의 라벨("구: mms-htdemucs-1 → 신: ...").
    # engine_version이 NULL이면 스탬프 도입 이전 세대라는 뜻이다(그 자체가 정보).
    engine_variant: str | None = None
    engine_version: str | None = None


class LineMeta(BaseModel):
    """라인별 부가 정보 — 발음 표기/사람 번역 (보카로 가사 위키 등). 텍스트로 세그먼트에 매칭된다."""

    text: str
    pronunciation: str | None = None
    translation: str | None = None


class Attribution(BaseModel):
    """가사 출처 표기 (예: 보카로 가사 위키 CC BY) — 싱크에 저장돼 조회 시 그대로 반환된다."""

    name: str
    url: str | None = None
    # "CC BY-SA 4.0" 등 라이선스 문구 — miraheze(vocaloidlyrics.miraheze.org) 어댑터가 싣는다.
    license: str | None = None
    # 'vocaro' | 'miraheze' 등 — 확장이 attribution.name 정규식 대신 이 필드로 출처를 가른다
    # (구싱크에는 없다 — 그 경우 확장은 이름 문자열 폴백 판정을 유지한다).
    source_id: str | None = None


# ── 완성된 싱크에 이미 확보한 번역을 직접 저장 (POST /{video_id}/translations) ──────

# "llm"은 일부러 뺐다 — 그 origin은 POST /api/translate persist=true 전용이다(그쪽만
# 서버가 실제로 번역을 만든다). 이 엔드포인트는 **이미 존재하는** 사람 번역 텍스트를
# 그대로 옮기는 경로라 llm을 자처할 수 없다.
_TRANSLATION_LAYER_ORIGINS = ("caption", "wiki", "manual")

# 가사 하한(_validate_lyrics)과 같은 이유의 상한 — /api/translate의 400줄/1000자 관례를
# 그대로 따른다(TranslateRequest 검사 참고).
_MAX_TRANSLATION_LAYER_LINES = 400
_MAX_TRANSLATION_LAYER_LINE_CHARS = 1000

# lines의 text가 그 영상 최신 싱크의 세그 원문과 이 비율 미만으로 일치하면 저장을 거절한다
# — 엉뚱한 곡·다른 버전 가사에 번역이 붙는 사고를 막는다("절반 이상" 일치를 요구).
_MIN_TRANSLATION_LAYER_MATCH_RATIO = 0.5


class TranslationLayerLine(BaseModel):
    text: str
    translation: str


class SaveTranslationLayerRequest(BaseModel):
    target_lang: str = Field(max_length=8)
    lines: list[TranslationLayerLine]
    origin: str
    attribution: Attribution | None = None


class SaveTranslationLayerResponse(BaseModel):
    saved: bool
    matched: int
    total: int
    target_lang: str


def _validate_translation_layer_request(request: "SaveTranslationLayerRequest") -> None:
    """origin 화이트리스트 + 라인 상한 — 본문 검증. 매칭률 검사는 세그 원문이 있어야
    가능해 핸들러(세션 진입 후)에서 별도로 한다."""
    if request.origin not in _TRANSLATION_LAYER_ORIGINS:
        raise HTTPException(
            status_code=422,
            detail=f"origin은 {_TRANSLATION_LAYER_ORIGINS} 중 하나여야 해요 (받은 값: {request.origin!r})",
        )
    if not request.lines:
        raise HTTPException(status_code=422, detail="lines가 비어 있어요")
    if len(request.lines) > _MAX_TRANSLATION_LAYER_LINES or any(
        len(ln.text) > _MAX_TRANSLATION_LAYER_LINE_CHARS
        or len(ln.translation) > _MAX_TRANSLATION_LAYER_LINE_CHARS
        for ln in request.lines
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                f"라인이 너무 많거나 길어요 — 최대 {_MAX_TRANSLATION_LAYER_LINES}줄, "
                f"줄당 {_MAX_TRANSLATION_LAYER_LINE_CHARS}자까지 지원해요."
            ),
        )


class GenerateRequest(BaseModel):
    video_id: str = Field(pattern=_VIDEO_ID_PATTERN)
    lyrics: str
    lyrics_source: str = "user_input"
    language: str | None = None
    line_meta: list[LineMeta] | None = None
    attribution: Attribution | None = None
    # 영상 제목/아티스트 — 완성된 싱크에 함께 저장돼 커버 링크 후보 탐색의 단서가 된다.
    # 선택 필드라 예전 클라이언트 요청도 그대로 동작한다(제목 없이 저장).
    title: str | None = Field(default=None, max_length=256)
    artist: str | None = Field(default=None, max_length=128)
    # "line_meta는 아직이고 나중에 POST /api/sync/jobs/{job_id}/line-meta로 붙인다"는 예고.
    # 서버는 line_meta 없이 잡을 만들어 다운로드·보컬 분리를 즉시 시작하고, 정렬 진입 직전에
    # 상한을 둔 대기를 한 번 넣는다 — 클라이언트의 번역·독음 시간과 그만큼이 겹친다.
    # line_meta를 본문에 함께 실어 보내면 이 플래그는 무시된다(기다릴 것이 없다).
    line_meta_pending: bool = False
    # 요청자의 번역 대상 언어 — Job.target_lang으로 저장돼 워커의 레이어 기록·legacy 병기
    # 판정에 쓰인다. 안 싣는 구버전 확장은 "ko"(기존 동작).
    target_lang: str = Field(default="ko", max_length=8)
    # line_meta에 실린 번역의 언어. 새 확장은 항상 target_lang과 같게 보낸다 — 서버는
    # 현재 target_lang만 소비하지만, 계약을 요청 스키마에 명시해 두면(Extra 필드 무시에
    # 기대지 않고) 두 값이 갈라지는 미래 클라이언트를 스키마 수준에서 받아들일 수 있다.
    line_meta_lang: str = Field(default="ko", max_length=8)


class GenerateResponse(BaseModel):
    job_id: str
    status: str
    estimated_time: int = 15
    # 이 잡이 정렬 진입 전에 늦은 line_meta를 기다려 주는 상한(초) — 상한이지 보장은 아니다.
    # 0이면 나중에 붙여도 정렬에는 반영되지 않는다: line_meta를 본문에 이미 실어 보냈거나,
    # 플래그를 안 켰거나, status="completed"(이 경우 job_id는 잡이 아니라 완성된 싱크의 id라
    # /jobs/{id}/line-meta를 쓸 수 없다 — 번역이 끝나면 line_meta를 실어 /generate를 다시
    # 호출하면 기존 싱크에 병합된다).
    line_meta_wait_sec: float = 0.0


class SearchByAudioRequest(BaseModel):
    audio_hash: str


class CopySyncRequest(BaseModel):
    source_video_id: str = Field(pattern=_VIDEO_ID_PATTERN)
    target_video_id: str = Field(pattern=_VIDEO_ID_PATTERN)
    lyrics: str | None = None


class FeedbackRequest(BaseModel):
    """정렬 품질 별점 + 선택 오류 제보 (확장 별점 UI, 2026-08-03). 수집 전용 — 응답에
    영향을 주지 않는다."""

    video_id: str = Field(pattern=_VIDEO_ID_PATTERN)
    # 화면에 실제로 떠 있던 sync_results.id. 필드 자체가 없으면 구형 클라이언트라 최신 행으로
    # 폴백하고, 명시적 null이면 위키/LRCLIB 등 서버 싱크가 아닌 화면을 평가한 것으로 본다.
    sync_id: str | None = Field(default=None, max_length=36)
    rating: int = Field(ge=1, le=5)
    category: str | None = Field(default=None, pattern="^(timing|pronunciation|lyrics|other)$")
    comment: str | None = Field(default=None, max_length=1000)
    # 제출 시점 화면에 떠 있던 싱크의 분석 깊이(fast/medium/heavy). sync_id가 유효하면 서버가
    # 그 행에서 다시 확인하고, 행이 없거나 구세대라 깊이를 모를 때만 이 값을 쓴다.
    depth: str | None = Field(default=None, pattern="^(fast|medium|heavy)$")


class RegenerateRequest(BaseModel):
    video_id: str = Field(pattern=_VIDEO_ID_PATTERN)
    lyrics: str
    language: str | None = None
    force: bool = False
    # 분석 깊이 하한("medium"|"heavy") — 확장의 "분석 깊이 올리기" 버튼. 새 스택이
    # 라우팅 판정을 건너뛰고 이 깊이에서 시작한다(worker._PENDING_MIN_DEPTH 스태시로
    # 전달). 안 싣는 구버전/일반 재생성은 None = 기존 자동 라우팅 그대로.
    min_depth: str | None = Field(default=None, pattern="^(medium|heavy)$")
    line_meta: list[LineMeta] | None = None
    attribution: Attribution | None = None
    title: str | None = Field(default=None, max_length=256)
    artist: str | None = Field(default=None, max_length=128)
    # GenerateRequest.line_meta_pending과 동일 — 재생성도 번역과 병렬로 돌릴 수 있다
    line_meta_pending: bool = False
    # GenerateRequest와 동일 계약 — 재생성 요청자의 번역 언어
    target_lang: str = Field(default="ko", max_length=8)
    line_meta_lang: str = Field(default="ko", max_length=8)


def _merge_meta_into_sync(
    sync_result,
    line_meta: list[LineMeta] | None,
    attribution: Attribution | None = None,
    line_meta_lang: str = "ko",
) -> int:
    """이미 존재하는 싱크에 발음/번역 메타·출처를 병합 (세션 커밋은 호출부의 컨텍스트가 수행).

    반환값은 메타가 붙은 세그먼트 수 — 늦게 붙이는 경로(attach_line_meta)가 얼마나 매칭됐는지
    호출자에게 알려 주는 데 쓴다. 번역은 line_meta_lang이 "ko"일 때만 legacy 슬롯에 병합한다
    (워커 완료 경로의 resolve_layer_lang 판정과 같은 규칙 — 비ko 번역을 legacy에 밀어 넣으면
    한국어 사용자가 남의 언어를 받는다). 발음(한글)은 언어 무관하게 병합한다."""
    from everyric2.server.worker import merge_line_meta

    updated = dict(sync_result.timestamps)
    changed = False
    merged = 0
    if line_meta:
        segs = [dict(s) for s in updated.get("segments", [])]
        merged = merge_line_meta(
            segs,
            [m.model_dump() for m in line_meta],
            with_translation=(line_meta_lang == "ko"),
            language=sync_result.language,
        )
        if merged:
            updated["segments"] = segs
            changed = True
    if attribution is not None:
        updated["attribution"] = attribution.model_dump()
        changed = True
    if changed:
        # JSON 컬럼은 재할당해야 변경이 감지된다
        sync_result.timestamps = updated
    return merged


# ── 싱크 링크 (inst/커버 영상이 다른 영상의 전사를 오프셋과 함께 재사용) ───────────


class SyncLinkRequest(BaseModel):
    video_id: str = Field(pattern=_VIDEO_ID_PATTERN)
    source_video_id: str = Field(pattern=_VIDEO_ID_PATTERN)
    offset_sec: float = 0.0
    # 원곡 대비 재생 배속 (nightcore 1.25 등) — 고정 오프셋만으로는 배속이 다른 커버에서
    # 곡이 진행될수록 가사가 밀린다. 소스 시간 t → t/rate + offset으로 사상.
    rate: float = Field(default=1.0, ge=0.25, le=4.0)


class SyncLinkResponse(BaseModel):
    video_id: str
    source_video_id: str
    offset_sec: float
    rate: float = 1.0
    # 반주 상관 검증(link-jobs)을 통과한 링크인지 — 이 수동 API로 만든 링크는 항상 False
    verified: bool = False
    created_at: str | None = None


def _shift_time(value: Any, offset: float, rate: float = 1.0) -> Any:
    """숫자면 t/rate + offset 사상(과한 부동소수 잡음 방지로 반올림), 아니면 그대로.

    rate는 원곡 대비 재생 배속 — nightcore(1.25)처럼 시간축이 압축된 커버는 고정
    오프셋만으로는 뒤로 갈수록 밀린다. rate=1.0이면 기존과 동일한 순수 시프트."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return round(value / rate + offset, 4)
    return value


def _shift_sync_timestamps(
    timestamps: dict[str, Any], offset: float, rate: float = 1.0
) -> dict[str, Any]:
    """소스 싱크의 모든 시간 필드를 t/rate + offset으로 사상한 깊은 복사본을 만든다.

    세그먼트 start/end·words·notes·pron_segments·pron_segs(표기별), extra.debug의
    vad_regions/star_spans/f0_curve.t0, tempo.beat_offset, 세그먼트 debug.orig·
    debug.heard_spans까지 함께 옮긴다. attribution 등 시간이 아닌 필드는 그대로 둔다.
    offset은 음수(소스가 링크 영상보다 늦게 시작)도 된다. rate≠1이면 BPM(rate배)과
    f0_curve 샘플 간격(dt/rate)도 함께 보정한다."""
    data = copy.deepcopy(timestamps)
    rate = rate if rate and rate > 0 else 1.0

    def sh(value: Any) -> Any:
        return _shift_time(value, offset, rate)

    for seg in data.get("segments", []) or []:
        if seg.get("start") is not None:
            seg["start"] = sh(seg["start"])
        if seg.get("end") is not None:
            seg["end"] = sh(seg["end"])
        for w in seg.get("words") or []:
            if w.get("start") is not None:
                w["start"] = sh(w["start"])
            if w.get("end") is not None:
                w["end"] = sh(w["end"])
        for n in seg.get("notes") or []:
            if n.get("start") is not None:
                n["start"] = sh(n["start"])
            if n.get("end") is not None:
                n["end"] = sh(n["end"])
        for p in seg.get("pron_segments") or []:
            if p.get("start") is not None:
                p["start"] = sh(p["start"])
            if p.get("end") is not None:
                p["end"] = sh(p["end"])
        # 표기별 발음 음절 스팬 — {"hangul": [...], "romaji": [...], "kana": [...]}.
        # pron_segments(레거시 한글 전용)와 같은 수식으로, 표기마다 따로 옮긴다.
        pron_segs = seg.get("pron_segs")
        if isinstance(pron_segs, dict):
            for script_spans in pron_segs.values():
                for p in script_spans or []:
                    if p.get("start") is not None:
                        p["start"] = sh(p["start"])
                    if p.get("end") is not None:
                        p["end"] = sh(p["end"])
        dbg = seg.get("debug")
        if isinstance(dbg, dict):
            if isinstance(dbg.get("orig"), list) and len(dbg["orig"]) == 2:
                dbg["orig"] = [sh(dbg["orig"][0]), sh(dbg["orig"][1])]
            # heard_spans: [[글자, 시각], ...] — CTC가 들은 글자별 시각(디버그 스트립용).
            # 두 번째 원소만 시간이다.
            heard_spans = dbg.get("heard_spans")
            if isinstance(heard_spans, list):
                dbg["heard_spans"] = [
                    [span[0], sh(span[1])]
                    if isinstance(span, (list, tuple)) and len(span) >= 2
                    else span
                    for span in heard_spans
                ]

    debug = data.get("debug")
    if isinstance(debug, dict):
        for key in ("vad_regions", "star_spans"):
            arr = debug.get(key)
            if isinstance(arr, list):
                debug[key] = [
                    [sh(span[0]), sh(span[1]), *span[2:]]
                    for span in arr
                    if isinstance(span, (list, tuple)) and len(span) >= 2
                ]
        f0 = debug.get("f0_curve")
        if isinstance(f0, dict) and f0.get("t0") is not None:
            f0["t0"] = sh(f0["t0"])
            if rate != 1.0 and isinstance(f0.get("dt"), (int, float)):
                f0["dt"] = round(f0["dt"] / rate, 6)

    tempo = data.get("tempo")
    if isinstance(tempo, dict):
        if tempo.get("beat_offset") is not None:
            tempo["beat_offset"] = sh(tempo["beat_offset"])
        if rate != 1.0 and isinstance(tempo.get("bpm"), (int, float)):
            tempo["bpm"] = round(tempo["bpm"] * rate, 2)

    return data


def _build_sync_response(
    result, timestamps: dict[str, Any], linked: dict[str, Any] | None = None
) -> "SyncLookupResponse":
    """SyncResult + (원본 또는 시프트된) timestamps dict → 조회 응답. linked면 빌린 싱크."""
    return SyncLookupResponse(
        found=True,
        sync_id=result.id,
        timestamps=timestamps.get("segments", []),
        lyrics_source=result.engine,
        quality_score=result.quality_score,
        audio_hash=result.audio_hash,
        language=result.language,
        engine_variant=result.engine_variant,
        engine_version=result.engine_version,
        created_at=result.created_at.isoformat() if result.created_at else None,
        debug=timestamps.get("debug"),
        attribution=timestamps.get("attribution"),
        tempo=timestamps.get("tempo"),
        key=timestamps.get("key"),
        adlib=timestamps.get("adlib"),
        linked=linked,
    )


async def _bump_sync_views(session, video_id: str) -> None:
    """조회수 증가 — 실패해도 조회 자체(get_sync의 응답)는 절대 막지 않는다. 카운터는
    "내가 만든 N곡을 M명이 봤어요" 기여 이력용 부가 정보지, 싱크 조회의 핵심 계약이
    아니다(SyncView.__doc__ 참고)."""
    try:
        await SyncViewRepository(session).increment(video_id)
    except Exception:
        logger.exception("Failed to bump sync view count for video %s", video_id)


async def _persist_legacy_ko_layer(
    video_id: str,
    fingerprint: str,
    lines: list[dict[str, Any]],
    attribution: dict[str, Any] | None,
) -> None:
    """세그에 박혀 있던 레거시 ko 번역을 TranslationLayer(origin="legacy")로 백필한다.

    배포 이전(레이어 테이블이 생기기 전) 생성분은 ko 번역이 SyncResult.timestamps의
    세그먼트에만 있고 레이어가 없다. lang=en 같은 비ko 조회는 레이어가 없으면 세그
    translation을 비우므로(TranslationLayer.origin 주석 참고), 그 상태에서 재생성이 한 번
    이라도 일어나면 원래 있던 ko 번역(위키 사람 번역 포함)을 되살릴 방법이 사라진다 —
    두 호출부(`_apply_translation_lang`의 lang=ko 조회, `regenerate_sync`의 잡 생성 직전)가
    이 함수로 "레이어가 아직 없으면 지금 채운다"를 수행한다.

    BackgroundTasks로 스케줄되므로 요청을 처리하던 세션이 아니라 독립된 세션을 새로
    연다 — translate.py의 `_persist_translation_layer`와 같은 동기↔비동기 브리지 뒤
    저장 패턴이다. 실패해도 이미 나간 응답에는 영향이 없으므로 로그만 남기고 삼킨다.
    """
    try:
        async with get_session() as session:
            await TranslationLayerRepository(session).upsert_layer(
                video_id, fingerprint, "ko", lines=lines, attribution=attribution, origin="legacy"
            )
    except Exception:
        logger.exception("Failed to backfill legacy ko translation layer for video %s", video_id)


async def _schedule_ko_backfill_if_needed(
    session,
    background_tasks: BackgroundTasks,
    video_id: str,
    segments: list[dict[str, Any]],
    attribution: dict[str, Any] | None,
) -> None:
    """세그에 레거시 ko 번역이 있고 그 지문의 ko 레이어가 아직 없으면 백그라운드로 채운다.

    응답(조회든 재생성이든)을 늦추지 않으려고 upsert 자체는 `_persist_legacy_ko_layer`로
    미룬다 — 여기서는 "채울 필요가 있는가"만 판정하고 스케줄만 건다."""
    has_translation = any((seg.get("translation") or "").strip() for seg in segments)
    if not has_translation:
        return
    fingerprint = lines_fingerprint([seg.get("text", "") or "" for seg in segments])
    if await TranslationLayerRepository(session).get_layer(video_id, fingerprint, "ko") is not None:
        return
    lines = [
        {"text": seg.get("text", "") or "", "translation": seg.get("translation", "") or ""}
        for seg in segments
    ]
    background_tasks.add_task(_persist_legacy_ko_layer, video_id, fingerprint, lines, attribution)


async def _apply_translation_lang(
    session,
    video_id: str,
    resp: "SyncLookupResponse",
    lang: str | None,
    background_tasks: BackgroundTasks,
    link_source_video_id: str | None = None,
) -> "SyncLookupResponse":
    """조회 응답에 available_langs를 채우고, lang이 있으면 세그먼트 translation을 그
    언어로 맞춘다.

    **available_langs는 lang 유무와 무관하게 항상 채운다** — 추가 필드라 구버전 클라이언트
    호환에는 영향이 없다. 번역 치환은 **lang이 없으면 하지 않는다** — 구버전 클라이언트의
    응답이 필드 단위로 기존과 동일해야 한다는 전역 제약을 여기서 지킨다. video_id는 항상
    URL 경로의 값(자기 싱크든 링크로 빌린 싱크든)을 쓴다 — TranslationLayer는 (video_id,
    fingerprint, target_lang) 키라 보는 영상 기준으로 일관되게 조회해야 POST
    /api/translate의 persist=true 저장과 같은 키로 맞아떨어진다.

    link_source_video_id: 이 조회가 SyncLink로 빌려온 싱크일 때만 준다(자기 싱크 조회는
    None). 가사(따라서 fingerprint)가 원곡과 같으므로, **커버 자신의 video_id로 레이어가
    하나도 안 잡히면** 원곡의 video_id로 1회 폴백 조회한다(available_langs·lang 조회
    둘 다) — 안 그러면 원곡에 이미 있는 en 등 비ko 레이어를 커버 시청자는 영원히 못 본다
    (레이어 키가 video_id라 가사가 같아도 커버·원곡이 서로 다른 레이어로 격리되는 것이
    원인). **저장은 폴백하지 않는다** — ko 백필(`_schedule_ko_backfill_if_needed`)이
    새로 레이어를 만들 때는 여전히 보는 영상(커버) 기준을 지킨다(기존 원칙 불변, 아래
    호출부 참고).

    세그먼트는 원본 result.timestamps의 리스트를 직접 건드리지 않도록 얕은 복사본을
    만들어 교체한다 — JSON 컬럼은 재할당해야 변경이 감지되므로(다른 곳의 동일 주석 참고)
    이 자체가 SyncResult를 오염시키진 않지만, 세션 수명 동안 같은 ORM 객체가 재사용될
    가능성을 원천 차단하는 편이 안전하다.
    """
    if not resp.found:
        return resp
    segments = [dict(seg) for seg in (resp.timestamps or [])]
    fingerprint = lines_fingerprint([seg.get("text", "") or "" for seg in segments])
    has_legacy_translation = any((seg.get("translation") or "").strip() for seg in segments)

    repo = TranslationLayerRepository(session)
    layer_langs = await repo.list_layer_langs(video_id, fingerprint)
    if not layer_langs and link_source_video_id:
        layer_langs = await repo.list_layer_langs(link_source_video_id, fingerprint)
    available = set(layer_langs)
    # F4(2026-08-04 감사): 세그 translation 존재만으로 "ko"를 넣으면, 레거시 번역이
    # 실제로는 다른 언어인데 ko 레이어가 없는 곡에서 available_langs가 거짓으로 ko를
    # 광고한다(값을 더 정확하게 만들 뿐인 additive 보강 — 필드 존재 여부는 안 바뀐다).
    # 한글 문자가 실제로 있는 translation이 하나라도 있을 때만 ko로 친다.
    if any(_HANGUL_RE.search(seg.get("translation") or "") for seg in segments):
        available.add("ko")
    resp.available_langs = sorted(available)

    # segments가 아직 lang별로 덮이기 전(레거시 원본 상태)의 스냅샷으로 계산한다 —
    # 아래에서 특정 lang이 요청되면 segments["translation"]이 그 언어 값으로 덮이므로,
    # 그 뒤에 계산하면 legacy ko 항목이 엉뚱한 언어 값을 legacy로 착각해 담게 된다.
    resp.translations_by_lang = await _build_translations_by_lang(
        repo, video_id, link_source_video_id, fingerprint, layer_langs, segments, has_legacy_translation
    )

    if not lang:
        resp.timestamps = segments
        return resp

    layer = await repo.get_layer(video_id, fingerprint, lang)
    if layer is None and link_source_video_id:
        layer = await repo.get_layer(link_source_video_id, fingerprint, lang)

    # 미스이거나, 정확 매칭이 llm이면 — 다른 지문의 사람 번역이 더 나은 후보일 수 있다
    # (H7PR 프로드 실측: llm이 정확 매칭 자리를 차지해 "미스"가 아니게 되면서 이관이
    # 영원히 발동하지 않았다). 사람 origin의 정확 매칭은 시도하지 않는다 — 사람끼리는
    # 현재 지문이 최우선이다(재이관할 이유가 없다). 링크 조회는 커버 자신의 지문
    # 이력부터, 없으면 원곡의 지문 이력도 본다 — 정확 매칭의 own→link 폴백과 같은 우선순위.
    migrated = False
    migrated_attribution: dict[str, Any] | None = None
    migrated_origin: str | None = None
    if layer is None or layer.origin == "llm":
        for search_vid in (video_id, *((link_source_video_id,) if link_source_video_id else ())):
            ok, migrated_attribution, migrated_origin = await _try_cross_fingerprint_migration(
                repo, background_tasks, search_vid, video_id, fingerprint, lang, segments
            )
            if ok:
                migrated = True
                break

    if migrated:
        resp.translation_lang = lang
        # F5: 이관 서빙은 이관 원본 레이어(사람 origin)의 출처를 낸다 — 실제로 그
        # 출처의 번역이 재정렬돼 실린 것이기 때문이다.
        resp.translation_attribution = migrated_attribution
        resp.translation_origin = migrated_origin
    elif layer is not None:
        # 이관 후보가 없었거나(레이어가 애초에 사람 origin) 커버리지 미달 — 정확 매칭을
        # 그대로 서빙한다(merge_line_meta(worker.py)와 같은 색인 규칙 — 값이 있는 첫
        # 항목을 채택한다).
        by_text: dict[str, str] = {}
        for item in layer.lines or []:
            key = normalize_line(item.get("text", "") or "")
            if not key:
                continue
            value = item.get("translation", "") or ""
            if key not in by_text or (not by_text[key] and value):
                by_text[key] = value
        for seg in segments:
            seg["translation"] = by_text.get(normalize_line(seg.get("text", "") or ""), "")
        resp.translation_lang = lang
        resp.translation_attribution = layer.attribution
        resp.translation_origin = layer.origin
    elif lang == "ko":
        # 레이어가 없으면 저장된 레거시 번역이 ko라는 이행 가정 — 그대로 둔다.
        # 세그에 번역이 하나도 없으면 "ko"라고 우길 근거가 없으므로 None.
        resp.translation_lang = "ko" if has_legacy_translation else None
        if has_legacy_translation:
            # F5: 레이어가 없는 legacy 경로 — 곡 단위 가사 출처(resp.attribution, 이미
            # 이 응답에 채워져 있다)를 그대로 번역 출처로 낸다. origin="legacy"는
            # TranslationLayer.origin의 같은 값(모델 docstring 참고)과 일치시킨다.
            resp.translation_attribution = resp.attribution
            resp.translation_origin = "legacy"
            # 이번 조회로 레거시 ko가 노출되는 김에 레이어에 옮겨 백필한다 — 다음 번
            # lang=en 등 비ko 조회·재생성에서도 이 번역이 살아남게 한다.
            await _schedule_ko_backfill_if_needed(
                session, background_tasks, video_id, segments, resp.attribution
            )
    else:
        # 비ko이고 레이어가 없으면 레거시 값이 어느 언어인지 알 수 없다 — 비운다
        for seg in segments:
            seg["translation"] = ""
        resp.translation_lang = None
    resp.timestamps = segments
    return resp


# 크로스 지문 이관의 커버리지 문턱 — 저장(save_translation_layer)의 _MIN_TRANSLATION_
# LAYER_MATCH_RATIO와 같은 "절반 이상" 기준. 다른 값을 쓸 이유가 없어 상수를 공유한다.
_CROSS_FINGERPRINT_MIGRATION_MIN_COVERAGE = 0.5


async def _persist_migrated_translation_layer(
    video_id: str,
    fingerprint: str,
    target_lang: str,
    lines: list[dict[str, Any]],
    attribution: dict[str, Any] | None,
    origin: str,
) -> None:
    """크로스 지문 이관 결과를 새 지문으로 upsert — BackgroundTasks 전용, 독립 세션.

    이렇게 새 지문에 그대로 박아 두면 다음 조회부터는 정확 매칭(get_layer)으로 바로
    찾아 재정렬 비용을 다시 치르지 않는다. 실패해도 이미 나간 응답에는 영향이 없으므로
    로그만 남기고 삼킨다 — translate.py의 `_persist_translation_layer`와 같은 저장 패턴."""
    try:
        async with get_session() as session:
            await TranslationLayerRepository(session).upsert_layer(
                video_id, fingerprint, target_lang, lines=lines, attribution=attribution, origin=origin
            )
    except Exception:
        logger.exception("Failed to persist migrated translation layer for video %s", video_id)


async def _try_cross_fingerprint_migration(
    repo: TranslationLayerRepository,
    background_tasks: BackgroundTasks,
    search_video_id: str,
    store_video_id: str,
    fingerprint: str,
    lang: str,
    segments: list[dict[str, Any]],
) -> tuple[bool, dict[str, Any] | None, str | None]:
    """`search_video_id`의 다른 지문에 사람이 단 번역(wiki/caption/manual/legacy)이 있으면
    `align_translation_lines`로 지금 세그(`segments`)에 재정렬해 옮겨 쓴다.

    호출부는 두 상황에서 이 함수를 부른다: (a) 현재 지문에 `(video_id, fingerprint, lang)`
    레이어가 아예 없을 때(정확 매칭 미스), (b) 레이어가 있지만 origin이 "llm"일 때 — 사람
    번역이 있다면 그게 이겨야 한다(H7PR 프로드 실측: llm이 정확 매칭 자리를 차지해서
    "미스"가 아니게 되면 이관이 영원히 발동하지 않는 구멍이 있었다). llm origin은 대상이
    아니다 — 품질 보증이 없는 기계번역을 다른 줄 분할로 우격다짐 재정렬해 옮기느니
    재생성이 낫다.

    성공(재정렬 커버리지 50% 이상)하면 `segments`를 in-place로 채우고
    `(True, 이관 원본의 attribution, 이관 원본의 origin)`, 새 지문으로도 `store_video_id`
    키로 BackgroundTasks upsert(origin 유지)해 다음부터 정확 매칭으로 바로 찾게 한다 —
    **(b) 상황에서는 이 upsert가 기존 llm 레이어를 그대로 덮어쓴다**(`upsert_layer`가
    같은 (video, fingerprint, target_lang) 키를 통째로 교체하는 기존 동작 그대로 — 저장
    엔드포인트의 human-over-llm 규칙과 같은 원칙). `store_video_id`가 검색과 다를 수
    있는 이유(링크 조회): 저장은 언제나 **보는 영상** 기준을 지킨다(#6과 동일 원칙) —
    원곡에서 찾은 번역이어도 커버 video_id로 저장한다. attribution/origin을 함께 돌려주는
    이유(F5, 2026-08-04 감사): 호출부가 조회 응답의 `translation_attribution`/
    `translation_origin`에 실제로 서빙한 출처를 실어야 한다 — 새 지문 저장이 끝나기를
    기다리지 않고(백그라운드 태스크라 이 응답 시점엔 아직 커밋 전이다) 이미 손에 쥔
    `other`의 값을 그대로 돌려준다.

    후보가 없거나 커버리지 미달이면 `segments`를 건드리지 않고 `(False, None, None)` —
    호출부가 (b) 상황이었다면 기존 llm 레이어를 그대로 서빙하면 된다."""
    other = await repo.find_human_layer_other_fingerprint(search_video_id, lang, fingerprint)
    if other is None:
        return False, None, None
    seg_texts = [seg.get("text", "") or "" for seg in segments]
    if not seg_texts:
        return False, None, None
    remapped = align_translation_lines(seg_texts, other.lines or [])
    matched = sum(1 for value in remapped if value is not None)
    if matched / len(seg_texts) < _CROSS_FINGERPRINT_MIGRATION_MIN_COVERAGE:
        return False, None, None
    for seg, translation in zip(segments, remapped):
        seg["translation"] = translation or ""
    new_lines = [
        {"text": text, "translation": translation}
        for text, translation in zip(seg_texts, remapped)
        if translation is not None
    ]
    background_tasks.add_task(
        _persist_migrated_translation_layer,
        store_video_id,
        fingerprint,
        lang,
        new_lines,
        other.attribution,
        other.origin,
    )
    return True, other.attribution, other.origin


# translations_by_lang 응답 크기 상한 — 언어가 늘어날수록 응답이 커진다(세그당 배열
# 원소 하나씩). gzip 후엔 미미하지만(가라오케 표기 다중 문자열에 비하면 순수 텍스트
# 배열은 압축률이 높다) 무한정 담지 않도록 막아 둔다.
_TRANSLATIONS_BY_LANG_MAX_LANGS = 5


async def _find_cross_fingerprint_translation(
    repo: TranslationLayerRepository,
    video_id: str,
    link_source_video_id: str | None,
    fingerprint: str,
    lang: str,
    seg_texts: list[str],
) -> list[str | None] | None:
    """다른 지문의 사람 origin 레이어(wiki/caption/manual/legacy)를 찾아 `seg_texts`
    순서로 재정렬한 배열을 반환 — 커버리지 문턱(50%) 미달이거나 후보가 없으면 None.

    **읽기 전용**이다 — 저장은 호출부의 몫이다(`_try_cross_fingerprint_migration`은 같은
    탐색을 하되 성공 시 실제로 upsert까지 한다; 이 함수는 `translations_by_lang`처럼
    "보여주기만 하고 아직 아무도 안 쓴 언어까지 선제 저장은 안 한다"는 경로가 공유한다).
    링크 조회는 자기 video_id 지문 이력부터, 없으면 링크 소스의 지문 이력도 본다."""
    for search_vid in (video_id, *((link_source_video_id,) if link_source_video_id else ())):
        other = await repo.find_human_layer_other_fingerprint(search_vid, lang, fingerprint)
        if other is None:
            continue
        remapped = align_translation_lines(seg_texts, other.lines or [])
        matched = sum(1 for value in remapped if value is not None)
        if seg_texts and matched / len(seg_texts) >= _CROSS_FINGERPRINT_MIGRATION_MIN_COVERAGE:
            return remapped
    return None


async def _build_translations_by_lang(
    repo: TranslationLayerRepository,
    video_id: str,
    link_source_video_id: str | None,
    fingerprint: str,
    layer_langs: list[str],
    segments: list[dict[str, Any]],
    has_legacy_translation: bool,
) -> dict[str, list[str | None]]:
    """조회 응답의 translations_by_lang — 그 지문의 전 레이어(+legacy ko)를 세그 순서로
    정렬된 배열로 담는다. 확장이 lang= 재조회 없이 즉시 언어를 전환할 수 있게 한다.

    우선순위(비용이 싼 순서, `_TRANSLATIONS_BY_LANG_MAX_LANGS`까지):
    1. 이 지문(또는 링크 폴백)에 이미 있는 **사람 origin** 레이어, 또는 그 레이어가
       "llm"이면 다른 지문의 사람 번역이 재정렬 커버리지를 넘길 때 그걸로 대체 —
       `_apply_translation_lang`의 정확 매칭 서빙과 같은 우선순위(사람 크로스 지문 >
       정확 llm). llm 레이어이고 이관 후보가 없거나 커버리지 미달이면 그 llm 값 그대로.
    2. legacy ko — 1번에 "ko"가 없고(레이어 자체가 없음) 세그 자체에 레거시 번역이
       남아 있는 경우.
    3. 크로스 지문 이관 후보 — 1번에서 다뤄지지 않은(이 지문에 레이어 자체가 없는)
       언어 중, 다른 지문에 사람 origin 레이어가 있고 재정렬 커버리지가 문턱을 넘는 언어.

    **읽기 전용이다** — 1·3번에서 이관 가능함을 확인해도 여기서는 새 지문에 저장하지
    않는다(그건 실제로 그 언어가 lang= 요청된 순간 `_try_cross_fingerprint_migration`이
    한다 — 모든 lookup마다 아직 아무도 안 쓴 언어까지 선제적으로 써 두면 배경 작업이
    지나치게 늘어난다)."""
    seg_texts = [seg.get("text", "") or "" for seg in segments]
    result: dict[str, list[str | None]] = {}
    if not seg_texts:
        return result

    for lang_code in layer_langs:
        if len(result) >= _TRANSLATIONS_BY_LANG_MAX_LANGS:
            return result
        layer = await repo.get_layer(video_id, fingerprint, lang_code)
        if layer is None and link_source_video_id:
            layer = await repo.get_layer(link_source_video_id, fingerprint, lang_code)
        if layer is None:
            continue
        if layer.origin == "llm":
            migrated = await _find_cross_fingerprint_translation(
                repo, video_id, link_source_video_id, fingerprint, lang_code, seg_texts
            )
            if migrated is not None:
                result[lang_code] = [value or None for value in migrated]
                continue
        by_text: dict[str, str] = {}
        for item in layer.lines or []:
            key = normalize_line(item.get("text", "") or "")
            if not key:
                continue
            value = item.get("translation", "") or ""
            if key not in by_text or (not by_text[key] and value):
                by_text[key] = value
        result[lang_code] = [by_text.get(normalize_line(t)) or None for t in seg_texts]

    if (
        "ko" not in result
        and has_legacy_translation
        and len(result) < _TRANSLATIONS_BY_LANG_MAX_LANGS
    ):
        result["ko"] = [(seg.get("translation") or "").strip() or None for seg in segments]

    if len(result) < _TRANSLATIONS_BY_LANG_MAX_LANGS:
        for search_vid in (video_id, *((link_source_video_id,) if link_source_video_id else ())):
            candidate_langs = await repo.list_human_langs_other_fingerprint(search_vid, fingerprint)
            for lang_code in candidate_langs:
                if lang_code in result:
                    continue
                if len(result) >= _TRANSLATIONS_BY_LANG_MAX_LANGS:
                    return result
                migrated = await _find_cross_fingerprint_translation(
                    repo, video_id, link_source_video_id, fingerprint, lang_code, seg_texts
                )
                if migrated is not None:
                    result[lang_code] = [value or None for value in migrated]

    return result


async def _persist_pron_variants(sync_id: str, attached_segments: list[dict[str, Any]]) -> None:
    """`worker.attach_pron_variants` 결과를 SyncResult 행에 기회적으로 저장한다.

    BackgroundTasks로 스케줄되므로 요청 세션이 아니라 독립된 세션을 새로 연다 —
    translate.py의 `_persist_translation_layer`·이 파일의 `_persist_legacy_ko_layer`와
    같은 동기↔비동기 브리지 뒤 저장 패턴이다.

    행을 **다시 읽어** 그 시점의 세그먼트와 인덱스로 맞춰(요청 시점 스냅샷이 아니라) pron/
    pron_segs가 **아직 없는** 세그에만 병합한다 — 그 사이 다른 변경(번역 병합 등)이 있어도
    그 필드는 건드리지 않고, 이미 누군가 채웠으면 (멱등) 다시 쓰지 않는다. 세그 수가
    그 사이 달라졌으면(재생성 등) 인덱스 대응을 신뢰할 수 없어 통째로 포기한다 — 다음
    조회가 다시 시도한다. 실패해도 이미 나간 응답에는 영향이 없으므로 로그만 남기고 삼킨다.
    """
    try:
        async with get_session() as session:
            result = await session.get(SyncResult, sync_id)
            if result is None:
                return
            current = dict(result.timestamps or {})
            current_segments = list(current.get("segments", []) or [])
            if len(current_segments) != len(attached_segments):
                return
            changed = False
            new_segments = []
            for cur, attached in zip(current_segments, attached_segments):
                cur = dict(cur)
                if not cur.get("pron") and attached.get("pron"):
                    cur["pron"] = attached["pron"]
                    if attached.get("pron_segs"):
                        cur["pron_segs"] = attached["pron_segs"]
                    changed = True
                new_segments.append(cur)
            if not changed:
                return
            current["segments"] = new_segments
            result.timestamps = current  # JSON 컬럼 재할당 — 변경 감지 트리거
    except Exception:
        logger.exception("Failed to persist lazily attached pron variants for sync %s", sync_id)


async def _lazy_attach_pron_variants(
    sync_id: str, resp: "SyncLookupResponse", background_tasks: BackgroundTasks
) -> None:
    """구세대 싱크(표기별 발음 `pron` dict가 생기기 전에 만들어진 SyncResult)를 조회하는
    김에 `worker.attach_pron_variants`로 채워 넣는다 — 재생성 없이도 romaji·가나 표기가
    생긴다. 결정론 렌더(LLM 호출 없음)라 세그 수가 많은 곡도 비용이 낮다.

    **순서 의존— S5의 merge 가드(비한글 발음이 legacy `pronunciation`에 박히는 것을 막는
    `merge_line_meta` 가드, 커밋 ea145c0) 이후에만 안전하다.** 그 가드가 없으면 line_meta로
    들어온 romaji/가나 발음이 오염된 채로 `seg["pronunciation"]`(한글 전용 계약)에 박히고,
    그 위에 `attach_pron_variants`가 `pron["hangul"]`에 그 오염값을 얹어 **재생성 전까지
    지워지지 않는 영구 고착**을 만든다(감사 치명 #1 서버 잔여). 이 함수는 그 가드가 이미
    있다는 전제로 호출돼야 한다.

    fugashi(형태소 분석기)가 없는 배포는 통째로 건너뛴다 — 폴백(pykakasi) 독음은 ja 표기
    신뢰도가 낮은데, 여기서 만든 값은 DB에 영구 저장되므로(재생성 전까지 안 지워짐) 저품질
    값이 고착되면 안 된다.

    자기 싱크 조회 전용이다 — 링크로 빌려온 싱크(SyncLink 경유)는 세그가 시프트된
    사본이라 부르지 않는다(원곡을 직접 조회하면 원곡 행에 붙고, 그 뒤로는 `_shift_sync_
    timestamps`가 시프트해 커버 조회에도 자연히 실린다). 응답에는 즉시 반영하고(재조회
    없이 이번 요청부터 표기가 보이게), DB 저장은 BackgroundTasks로 미룬다(응답 지연
    방지 — translate persist·ko 백필과 같은 패턴).
    """
    from everyric2.text.ja_reading import reading_source

    segments = resp.timestamps or []

    def _needs_attach(seg: dict) -> bool:
        pron = seg.get("pron")
        if not pron:
            return True
        # 구세대 라틴 곡의 kana 단독 근사 — attach_pron_variants의 불완전 가드가
        # 빠진 표기(hangul/romaji/en/ipa)를 보완한다(표시값 E2E 실측 2026-08-03)
        return set(pron) == {"kana"}

    if not any(_needs_attach(seg) for seg in segments):
        return  # 전부 이미 완결 표기다 — 할 일 없음

    def _attach_all() -> bool:
        # **여기가 스레드인 것이 이 함수의 존재 이유다.** reading_source()의 첫 호출은
        # fugashi+UniDic 사전 로드(수 초)이고 attach 루프도 동기 CPU다 — async 핸들러에서
        # 인라인으로 돌리면 이벤트 루프가 통째로 멎어 /health까지 전부 타임아웃된다
        # (실측 2026-07-28: 재시작 직후 사용자 요청 클러스터가 1.5~8s 타임아웃 — gzip
        # 배포 재시작 7분 뒤 첫 lazy attach가 콜드 로드를 요청 경로에서 지불한 사고).
        if reading_source() != "fugashi":
            return False
        from everyric2.server.worker import attach_pron_variants

        for seg in segments:
            try:
                # 곡 언어를 넘긴다 — zh 곡의 순한자 라인이 ja 분기로 새지 않게 하는 게이트
                attach_pron_variants(seg, language=resp.language)
            except Exception:
                logger.exception("Lazy pron attach failed for a segment; leaving it as-is")
        return True

    import anyio

    attached = await anyio.to_thread.run_sync(_attach_all)
    if attached:
        background_tasks.add_task(_persist_pron_variants, sync_id, segments)


@router.post("/link", response_model=SyncLinkResponse)
async def create_sync_link(
    request: SyncLinkRequest,
    x_api_key: str | None = Header(default=None),
    # 이용자 축 한도의 식별자·상한 — 게이트웨이가 붙인다(_resolve_actor, request_limits)
    x_lyric_user: str | None = Header(default=None),
    x_lyric_limits: str | None = Header(default=None),
):
    """영상 video_id가 source_video_id의 싱크를 offset과 함께 빌려 쓰도록 링크(upsert).

    자기 자신 링크는 거부. source에 실제 싱크가 있어야 한다 — source가 그 자체로 링크만
    있고 자기 싱크가 없으면(링크의 링크) 거부한다(단순화: 1단계 링크만 허용).

    **이 경로는 검증이 없다** — 호출자가 준 오프셋(0 포함)을 그대로 박으므로 틀린 링크가
    코퍼스에 남을 수 있다(실제 사례 있음). 두 겹으로 완화한다: ① 만들어진 링크는 항상
    verified=False로 기록돼 자동 검증 링크(link-jobs 통과)와 조회 응답에서 구분되고,
    ② manual_link_requires_admin을 켠 배포에서는 어드민 키를 요구한다. 검증된 링크를
    원하면 POST /api/link-jobs(반주 상관 판정)를 쓴다.

    **확장 배지 "커버 잇기"가 세는 것이 이 행위다**(2026-08-10, docs/user-quota-spec.md §7).
    지금까지 그 배지가 세던 것은 자동 후보 탐색이었다 — 사용자가 요청한 적도, 실패해도
    볼 수도 없는 배경 요청이라 라벨과 내용이 어긋나 있었다. 반대로 사용자가 "커버 잇기"라고
    생각하는 이 경로에는 한도 검사도 기록도 전혀 없었다.

    검사 위치는 **실제로 링크가 만들어지기 직전**이다 — 자기 링크·소스 없음으로 거절되는
    요청은 아무것도 만들지 않으므로 예산을 먹으면 안 된다(생성 경로의 "GPU를 태우는 분기
    직전에만 센다"와 같은 규율)."""
    if request.video_id == request.source_video_id:
        raise HTTPException(status_code=400, detail="Cannot link a video to itself")

    server = get_settings().server
    if server.manual_link_requires_admin:
        if not server.admin_api_key or x_api_key != server.admin_api_key:
            raise HTTPException(
                status_code=403,
                detail="검증 없는 수동 링크는 어드민 키가 필요해요. "
                "자동 검증 링크는 /api/link-jobs로 요청해 주세요.",
            )

    async with get_session() as session:
        sync_repo = SyncRepository(session)
        source_syncs = await sync_repo.get_by_video(request.source_video_id)
        if not source_syncs:
            raise HTTPException(
                status_code=400,
                detail=f"Source video {request.source_video_id} has no sync to link",
            )
        await _check_action_limit(
            session, "link", request.video_id, x_lyric_user, x_api_key, x_lyric_limits
        )
        link_repo = SyncLinkRepository(session)
        link = await link_repo.upsert(
            request.video_id,
            request.source_video_id,
            request.offset_sec,
            request.rate,
            verified=False,
        )
        return SyncLinkResponse(
            video_id=link.video_id,
            source_video_id=link.source_video_id,
            offset_sec=link.offset_sec,
            rate=link.rate,
            verified=link.verified,
            created_at=link.created_at.isoformat() if link.created_at else None,
        )


@router.delete("/link/{video_id}")
async def delete_sync_link(video_id: str):
    _validate_video_id(video_id)
    async with get_session() as session:
        removed = await SyncLinkRepository(session).delete(video_id)
        return {"video_id": video_id, "removed": removed}


@router.get("/list")
async def list_available_syncs(limit: int = Query(50, ge=1, le=200)):
    """조회 가능한 싱크 목록 (확장의 링크 후보 선택용) — 영상별 1개, 최신순."""
    async with get_session() as session:
        results = await SyncRepository(session).get_all_unique_videos(limit=limit)
        items = []
        for r in results:
            ts = r.timestamps or {}
            segments = ts.get("segments", []) or []
            debug = ts.get("debug") or {}
            attribution = ts.get("attribution") or {}
            items.append(
                {
                    "video_id": r.video_id,
                    "first_line": segments[0].get("text", "") if segments else "",
                    "line_count": len(segments),
                    "attribution_name": attribution.get("name"),
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "alignment_text": debug.get("alignment_text"),
                }
            )
        # 확장 클라이언트(listSyncs)가 SyncListItem[] bare 배열을 기대 → 래핑하지 않는다
        return items


class UserOffsetRequest(BaseModel):
    offset_sec: float


@router.put("/offset/{video_id}")
async def save_user_offset(video_id: str, request: UserOffsetRequest):
    """이 영상에서 사용자가 조정한 싱크 오프셋(초)을 저장 — 다음 조회부터 함께 내려간다."""
    _validate_video_id(video_id)
    offset = max(-60.0, min(60.0, request.offset_sec))
    async with get_session() as session:
        await VideoOffsetRepository(session).upsert(video_id, offset)
    return {"video_id": video_id, "offset_sec": offset}


# ── 무다운로드 링크 재료 ①: 최근 조회의 가사 지문 기억 ───────────
#
# 확장은 GET /api/sync/{id}?lyrics_hash=…(가사 확보 시)를 먼저 부르고, 싱크가 없으면
# link-candidates를 잇달아 부른다. 확장 동결 제약으로 link-candidates 요청에 지문을 실을 수
# 없으므로, 두 호출을 서버 메모리로 잇는다 — 재시작 소실은 무해하다(다음 조회가 다시 채운다).
_RECENT_LYRICS_HASH: dict[str, tuple[str, float]] = {}
_LYRICS_HASH_TTL_SEC = 3600.0
_LYRICS_HASH_MAX = 4096


def _remember_lyrics_hash(video_id: str, lyrics_hash: str) -> None:
    now = time.time()
    if len(_RECENT_LYRICS_HASH) >= _LYRICS_HASH_MAX:
        cutoff = now - _LYRICS_HASH_TTL_SEC
        for k in [k for k, (_, ts) in _RECENT_LYRICS_HASH.items() if ts < cutoff]:
            _RECENT_LYRICS_HASH.pop(k, None)
        while len(_RECENT_LYRICS_HASH) >= _LYRICS_HASH_MAX:  # 전부 살아 있으면 오래된 것부터
            _RECENT_LYRICS_HASH.pop(next(iter(_RECENT_LYRICS_HASH)))
    _RECENT_LYRICS_HASH[video_id] = (lyrics_hash, now)


def _recall_lyrics_hash(video_id: str) -> str | None:
    row = _RECENT_LYRICS_HASH.get(video_id)
    if not row:
        return None
    h, ts = row
    if time.time() - ts > _LYRICS_HASH_TTL_SEC:
        _RECENT_LYRICS_HASH.pop(video_id, None)
        return None
    return h


class LinkCandidate(BaseModel):
    video_id: str
    title: str | None = None
    artist: str | None = None
    # 제목 유사도 (1.0 = 정규화 정확 일치). 같은 곡인지의 판정값이 아니라 후보 순위일 뿐이다
    score: float


class LinkCandidatesResponse(BaseModel):
    video_id: str
    # has_sync | linked | disabled | none | submitted | pending | cooldown
    status: str
    candidates: list[LinkCandidate] = []
    # 낸 후속 작업의 종류 — 클라이언트가 진행 상태를 어느 API로 폴링할지 가른다.
    # 오늘은 "link_validate"(반주 상관 검증) 하나뿐이다. _dispatch_candidate_followup 참고.
    followup: str | None = None
    # submitted/pending/cooldown일 때 해당 후속 작업의 id
    job_id: str | None = None


# ── 후보 확정 이후: 후속 작업 디스패치 (교체 지점) ────────────────

# 반주 상관 검증 잡 — 커버가 원곡과 같은 반주를 쓰는지 판정해 SyncLink를 만든다
FOLLOWUP_LINK_VALIDATE = "link_validate"


async def _dispatch_candidate_followup(
    session,
    video_id: str,
    candidate_video_id: str,
) -> tuple[str, str, str | None]:
    """후보를 확정한 뒤 **무엇을 제출할지** 결정하는 단일 교체 지점. (kind, status, job_id) 반환.

    status는 submitted | pending | cooldown.

    오늘의 구현은 반주 상관 검증 잡 하나다. 앞으로 "원곡의 가사·번역·독음을 재사용해
    이 영상 자체를 새로 정렬"처럼 다른 후속 작업으로 갈아끼울 수 있도록 제출 로직을 여기
    한 곳에 가둬 두었다 — 후보 탐색·제목 정규화·재제출 억제 정책은 이 함수를 바꿔도
    재작성할 필요가 없다. 두 경로를 조건부로 함께 쓰거나(예: 커버 음질이 나쁘면 링크,
    아니면 재정렬) 순차 폴백으로 확장하는 것도 이 함수 안에서 끝난다.

    교체 구현이 지켜야 할 계약:
      - 같은 (영상, 후보) 쌍의 재제출 억제를 반드시 자체적으로 유지할 것 — 진행 중이면
        pending, 최근에 끝난 이력이 있으면 cooldown. 이게 없으면 사용자가 같은 영상을
        열 때마다 GPU가 다시 돈다(현재 쿨다운 기준: link_retry_cooldown_days).
      - kind는 클라이언트가 진행 상태를 어느 API로 폴링할지 가르는 값이므로, 새 종류를
        도입하면 그 종류의 조회 경로도 함께 알려야 한다.

    **횟수 상한이 없는 이유**(2026-08-10, docs/user-quota-spec.md §7): 이 경로의 비용은
    횟수가 아니라 **구조**가 막는다 — `link_require_cached_pair`(기본 on)가 커버·원곡 양쪽이
    미디어 캐시에 있을 때만 잡을 만들고, `link_cache_only`(기본 on)가 캐시 미스에서 yt-dlp
    폴백을 금지한다. 실측(2026-08-10): 유튜브 접촉 0회, 검증 잡 전체 하루 2~3건.
    예전에 있던 횟수 상한은 사용자가 요청한 적도 없고 실패해도 볼 수 없는 배경 요청을
    사람의 예산에서 깎았고(확장은 이 조회의 실패를 조용히 무시한다 — content.ts
    probeLinkCandidates: `if (!data) return;`), 그 소진이 "커버 잇기" 배지로 보였다.

    🔴 **그 구조적 게이트를 끄는 것은 비용 상한을 없애는 결정과 같다**(같은 문서 §7).
    `link_require_cached_pair` 또는 `link_cache_only`를 off로 돌리려면 그때 횟수 상한을
    다시 논해야 한다 — 둘 중 하나라도 꺼지면 이 GET 하나가 다운로드 2회를 부를 수 있고,
    그때는 이 함수에 상한을 되돌려야 한다.

    기록도 남기지 않는다: 집행에 쓰이지 않는 값을 이용자 축으로 쌓는 것은 "한도 집행에
    필요한 최소 범위"(같은 문서 §1 개인정보 원칙)를 넘는다.
    """
    server = get_settings().server
    repo = LinkJobRepository(session)

    active = await repo.get_active_pair(video_id, candidate_video_id)
    if active:
        return FOLLOWUP_LINK_VALIDATE, "pending", active.id
    recent = await repo.get_recent_attempt(
        video_id, candidate_video_id, server.link_retry_cooldown_days
    )
    if recent:
        return FOLLOWUP_LINK_VALIDATE, "cooldown", recent.id
    # 억제(pending/cooldown)를 모두 통과해 실제로 GPU 잡을 만드는 지점. 횟수 상한은 여기
    # 없다 — 위 독스트링의 "횟수 상한이 없는 이유" 참고(구조적 게이트가 비용을 담당한다).
    link_job = await repo.create(video_id, candidate_video_id)
    return FOLLOWUP_LINK_VALIDATE, "submitted", link_job.id


@router.get("/{video_id}/link-candidates", response_model=LinkCandidatesResponse)
async def find_link_candidates(
    video_id: str,
    title: Annotated[str, Query(min_length=1, max_length=256)],
    artist: Annotated[str | None, Query(max_length=128)] = None,
    # 이용자 축 한도 헤더(x-lyric-user / x-lyric-limits)를 받지 않는다 — 이 경로는
    # 2026-08-10부터 어떤 예산도 소비하지 않는다(docs/user-quota-spec.md §7,
    # _dispatch_candidate_followup의 "횟수 상한이 없는 이유"). 받아 두고 안 쓰면 다음
    # 사람이 "여기도 세고 있나 보다"로 읽는다.
):
    """이 영상과 같은 곡일 만한 코퍼스 영상을 제목으로 찾고, 최상위 후보 1건에 대해
    후속 작업을 자동 제출한다 (무엇을 제출할지는 _dispatch_candidate_followup이 정한다 —
    오늘은 반주 상관 검증 잡). 응답의 followup이 그 종류를 알려준다.

    **제목 매칭은 후보 발견에만 쓴다.** 같은 곡인지의 최종 판정은 기존 반주 상관 게이트
    (link_match_threshold·link_min_offset_margin)가 그대로 담당하며, 이 엔드포인트가
    SyncLink를 직접 만드는 경로는 없다 — 제목이 맞았다는 이유만으로 링크가 생기지 않는다.
    그래서 매칭이 헐거워도 안전하다(오탐의 대가는 후속 작업 한 번).

    자기 싱크가 있거나 이미 링크가 있으면 후보 없이 즉시 반환한다. 같은 쌍을 최근
    link_retry_cooldown_days 안에 이미 시도했으면 재제출하지 않는다 — 사용자가 같은 영상을
    반복해 열 때마다 GPU를 다시 태우는 남용 경로를 막는다.

    **횟수 상한은 없다**(2026-08-10, docs/user-quota-spec.md §7): 이 요청은 사용자가 낸
    것이 아니라 확장이 배경으로 내는 것이라 사람의 예산에서 깎을 근거가 없고, 비용은
    (d)의 캐시 게이트가 구조적으로 막는다. 근거와 "그 게이트를 끄면 상한이 사라진다"는
    조건은 _dispatch_candidate_followup 독스트링에 있다."""
    _validate_video_id(video_id)
    server = get_settings().server

    async with get_session() as session:
        sync_repo = SyncRepository(session)

        # (a) 자기 싱크가 있으면 링크가 필요 없다 — 대신 비어 있던 제목을 이 기회에 채운다
        own = await sync_repo.get_by_video(video_id)
        if own:
            await sync_repo.set_title_if_missing(own[0], title, artist)
            return LinkCandidatesResponse(video_id=video_id, status="has_sync")
        if await SyncLinkRepository(session).get(video_id):
            return LinkCandidatesResponse(video_id=video_id, status="linked")

        # (b0) 무다운로드 0순위 — 플랫폼 관계 조회(songlink/1): 코퍼스 메타데이터에서
        # 파생된 커버↔원곡 관계, 조달 비용 0. 자동 파생이라(정답지 대비 74.5%) 후보일
        # 뿐이고, 원곡에 싱크가 있어야만 세운다(빌릴 것이 없으면 링크가 무의미하다).
        relation = await asyncio.to_thread(song_link.lookup_original, video_id)
        relation_hit = None
        if relation:
            rel_rows = await sync_repo.get_by_video(relation["original"]["id"])
            if rel_rows:
                relation_hit = rel_rows[0]

        # (b) 무다운로드 1순위 — 가사 지문: 같은 lyrics_hash의 싱크를 가진 다른 영상이
        # 있으면 제목 유사도와 무관하게 최상위 후보다. 지문은 직전 GET /api/sync가 서버
        # 메모리에 남긴 것을 이어받는다(확장 동결 — 요청 모양 불변). 판정이 아니라 후보다:
        # 링크에는 오프셋이 필요하고 그건 반주 상관만 계산할 수 있다.
        fingerprint_hit = None
        recalled = _recall_lyrics_hash(video_id)
        if recalled:
            fingerprint_hit = await sync_repo.find_other_video_by_lyrics_hash(
                recalled, video_id
            )

        # (c) 제목이 채워진 코퍼스를 전수 스캔해 상위 후보를 뽑는다 (자기 자신 제외).
        # rank_matches의 문서빈도 억제가 프로듀서명·가수명·잡토큰 조각 오탐을 거른다
        rows = await sync_repo.list_titled(limit=server.link_candidate_scan_limit)
        entries = [(r.video_id, r.title or "") for r in rows if r.video_id != video_id]
        ranked = title_match.rank_matches(
            title, entries, min_score=server.link_candidate_min_title_score, limit=5
        )
        by_video = {r.video_id: r for r in rows}
        candidates = [
            LinkCandidate(
                video_id=vid,
                title=by_video[vid].title,
                artist=by_video[vid].artist,
                score=score,
            )
            for vid, score in ranked
        ]
        def _prepend(cands: list[LinkCandidate], row, score: float) -> list[LinkCandidate]:
            return [
                LinkCandidate(
                    video_id=row.video_id, title=row.title, artist=row.artist, score=score
                ),
                *[c for c in cands if c.video_id != row.video_id],
            ][:5]

        # 우선순위: 지문(가사 문자열 일치 — 가장 강함) > 관계(자동 파생) > 제목 유사도.
        # 관계를 먼저 얹고 지문을 마지막에 얹어 지문이 맨 앞에 선다.
        if relation_hit is not None:
            conf = max(0.0, min(1.0, float(relation.get("confidence") or 0.0)))
            candidates = _prepend(candidates, relation_hit, conf)
        if fingerprint_hit is not None:
            candidates = _prepend(candidates, fingerprint_hit, 1.0)
        if not candidates:
            return LinkCandidatesResponse(video_id=video_id, status="none")
        if not server.auto_link_candidates:
            return LinkCandidatesResponse(
                video_id=video_id, status="disabled", candidates=candidates
            )

        # (d) 자동 제출 게이트 — 무다운로드 원칙. 상관을 다운로드 없이 돌릴 수 있는 경우
        # = 양쪽 오디오가 미디어 캐시에 있을 때뿐이므로, 그때만 잡을 만든다. 미스면 후보만
        # 반환하고 유튜브는 접촉하지 않는다 — «연결 실패는 허용되는 결과지만 유튜브 접촉은
        # 아니다»(unite 요청 2026-07-29). status는 확장이 이미 조용히 처리하는 none을 쓴다.
        # 실측: 공개 후 실사용자 영상의 캐시 적중률 11% — 이 게이트가 자동 제출의 89%를
        # 다운로드로 잇던 경로를 끊는다.
        if server.link_require_cached_pair:
            cover_cached, cand_cached = await asyncio.gather(
                asyncio.to_thread(media_cache.lookup_cached, video_id),
                asyncio.to_thread(media_cache.lookup_cached, candidates[0].video_id),
            )
            if not (cover_cached and cand_cached):
                return LinkCandidatesResponse(
                    video_id=video_id, status="none", candidates=candidates
                )

        # (e) 최상위 후보 1건만 제출한다 (여러 후보 순차 재시도는 넣지 않는다).
        # 무엇을 제출할지는 _dispatch_candidate_followup 한 곳에서만 정해진다
        kind, status, job_id = await _dispatch_candidate_followup(
            session, video_id, candidates[0].video_id
        )
        return LinkCandidatesResponse(
            video_id=video_id,
            status=status,
            candidates=candidates,
            followup=kind,
            job_id=job_id,
        )


# POST /api/sync/exists — video_id 최대 100개 (POST /api/stats/views와 같은 상한 근거:
# 배치 조회 하나가 무제한 IN절로 DB를 두들기는 것을 막는다).
_MAX_EXISTS_VIDEO_IDS = 100


class SyncExistsRequest(BaseModel):
    video_ids: list[str] = Field(max_length=_MAX_EXISTS_VIDEO_IDS)


class SyncExistsResponse(BaseModel):
    exists: dict[str, bool]


@router.post("/exists", response_model=SyncExistsResponse)
async def sync_exists(request: SyncExistsRequest):
    """요청한 video_id들의 싱크 존재 여부를 일괄 조회한다 (additive, 확장 개편의 영상별
    싱크 존재 배지용).

    **GET이 아니라 POST여야 한다** — `@router.get("/{video_id}")` 캐치올(바로 아래)이
    라우트 등록 순서상 GET /api/sync/exists를 "video_id=exists"로 그대로 삼켜 버린다.
    POST /api/stats/views(stats.py)와 같은 배치 조회 모양을 그대로 따른다.

    `sync_results`(자기 싱크)뿐 아니라 `sync_links`(빌려 온 싱크)도 존재로 친다 —
    `GET /api/sync/{video_id}`가 자기 싱크 없는 영상에도 링크 폴백을 내주므로(바로
    아래 `get_sync` 참고), 링크만 있는 영상을 false로 답하면 확장이 "싱크 없음" 배지를
    잘못 띄운다.

    쿼리는 video_id 열만 본다 — `timestamps` JSON 블롭은 절대 select하지 않는다(존재
    유무만 필요한 요청 하나가 곡 전체를 실어 나르면 안 된다). 응답은 요청 전체를 덮는
    dict다(요청하지 않은 video_id는 안 실린다) — 없는 영상은 False.
    """
    ids = [vid for vid in request.video_ids if _VIDEO_ID_RE.match(vid)]
    if len(ids) != len(request.video_ids):
        raise HTTPException(status_code=422, detail="invalid video_id in video_ids")

    async with get_session() as session:
        own = await SyncRepository(session).get_existing_video_ids(ids)
        linked = await SyncLinkRepository(session).get_existing_video_ids(ids)
        found = own | linked
        return SyncExistsResponse(exists={vid: vid in found for vid in ids})


@router.get("/{video_id}", response_model=SyncLookupResponse)
async def get_sync(
    video_id: str,
    lyrics_hash: str | None = None,
    title: Annotated[str | None, Query(max_length=256)] = None,
    artist: Annotated[str | None, Query(max_length=128)] = None,
    title_evidence: Annotated[
        str | None, Query(pattern="^channel_reversed$")
    ] = None,
    lang: Annotated[str | None, Query(max_length=8)] = None,
    # 기본값을 둔 이유: `BackgroundTasks | None = None`은 FastAPI가 이 타입을 더 이상
    # "시스템이 주입하는 특수 의존성"으로 인식하지 못하게 만들어 라우트 등록 자체가
    # FastAPIError로 깨진다(POST /api/translate에서 실측). 그렇다고 필수 인자로 두면
    # 이 함수를 직접 호출하는 기존 테스트 20여 곳(test_sync_link.py 등 — 이 작업의
    # 수정 허용 범위 밖)이 전부 깨진다. `BackgroundTasks()` 기본값은 타입 주석 자체는
    # Optional이 아니라서 FastAPI가 여전히 실제 요청마다 **새 인스턴스를 주입**하고
    # (기본값은 무시된다 — ASGI 요청으로 직접 검증함) 응답 후 정상적으로 실행해 준다.
    # 기본값 인스턴스는 이 함수를 직접 호출하는 기존 테스트가 인자를 생략했을 때만
    # 쓰이는 껍데기이고, 아무도 실행해 주지 않아 그 호출들의 백필 스케줄은 조용히
    # 버려진다(그 테스트들은 애초에 백필을 검증하지 않는다).
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """이 영상의 싱크를 조회한다. 자기 싱크 > 링크로 빌려온 싱크 순.

    title/artist는 선택적 기회적 백필용이다. 기본은 빈 값만 채우고, title_evidence가
    channel_reversed일 때만 과거의 뒤집힌 자동 메타데이터를 교정한다. 링크로 빌려온 싱크는
    소유자가 다른 영상(원곡)이라 커버 제목을 원곡 행에 새기지 않는다.

    lang은 선택이다 — 주면 세그먼트 translation을 그 언어의 TranslationLayer로 맞춰
    치환하고 응답의 translation_lang에 실제 반영된 언어를 담는다(규칙은
    _apply_translation_lang 참고). 안 주면 기존 필드 그대로다(구버전 클라이언트 호환).
    available_langs는 lang 유무와 무관하게 항상 채워진다."""
    _validate_video_id(video_id)
    if lyrics_hash:
        # 무다운로드 링크 재료 — 싱크 유무와 무관하게 남긴다(뒤이은 link-candidates가 쓴다)
        _remember_lyrics_hash(video_id, lyrics_hash)
    async with get_session() as session:
        repo = SyncRepository(session)
        user_offset = await VideoOffsetRepository(session).get(video_id)

        # 자기 싱크가 있으면 링크보다 우선한다
        if lyrics_hash:
            result = await repo.get_by_video_and_hash(video_id, lyrics_hash)
            if result:
                await repo.set_title_if_missing(
                    result,
                    title,
                    artist,
                    overwrite=title_evidence == "channel_reversed",
                )
                await _bump_sync_views(session, video_id)
                resp = _build_sync_response(result, result.timestamps)
                resp.user_offset = user_offset
                resp = await _apply_translation_lang(session, video_id, resp, lang, background_tasks)
                # 자기 싱크 조회에서만 부른다 — 구세대 싱크에 pron dict를 기회적으로
                # 채운다(감사 #3, S5 merge 가드 이후 전제 — _lazy_attach_pron_variants 참고)
                await _lazy_attach_pron_variants(result.id, resp, background_tasks)
                return resp
        else:
            results = await repo.get_by_video(video_id)
            if results:
                await repo.set_title_if_missing(
                    results[0],
                    title,
                    artist,
                    overwrite=title_evidence == "channel_reversed",
                )
                await _bump_sync_views(session, video_id)
                resp = _build_sync_response(results[0], results[0].timestamps)
                resp.user_offset = user_offset
                resp = await _apply_translation_lang(session, video_id, resp, lang, background_tasks)
                await _lazy_attach_pron_variants(results[0].id, resp, background_tasks)
                return resp

        # 자기 싱크가 없고 링크가 있으면 source 싱크를 offset 적용해 빌려 온다
        link = await SyncLinkRepository(session).get(video_id)
        if link:
            source_syncs = await repo.get_by_video(link.source_video_id)
            if source_syncs:
                src = source_syncs[0]
                link_rate = getattr(link, "rate", 1.0) or 1.0
                shifted = _shift_sync_timestamps(src.timestamps, link.offset_sec, link_rate)
                # 보는 영상(video_id) 기준 — 원곡(source_video_id)이 아니라 이 링크로 실제
                # 보고 있는 영상의 조회수를 늘린다(레이어 조회의 video_id 관례와 동일).
                await _bump_sync_views(session, video_id)
                resp = _build_sync_response(
                    src,
                    shifted,
                    linked={
                        "source_video_id": link.source_video_id,
                        "offset_sec": link.offset_sec,
                        "rate": link_rate,
                        # 반주 상관 검증을 통과한 링크인지 — 수동 링크(검증 없이 오프셋 지정)와
                        # 구분해 클라이언트가 신뢰도를 표시할 수 있게 한다
                        "verified": bool(getattr(link, "verified", False)),
                    },
                )
                resp.user_offset = user_offset
                # lang 레이어는 보는 영상(video_id) 기준으로 조회한다 — source_video_id가
                # 아니다. POST /api/translate persist=true도 항상 요청받은 video_id로
                # 저장하므로, 조회도 같은 키를 써야 서로 맞아떨어진다. 다만 커버 자신의
                # 레이어가 하나도 없으면 가사가 같은 원곡(source_video_id)으로 1회
                # 폴백한다 — 안 그러면 원곡에만 있는 비ko 레이어를 커버 시청자가 영원히
                # 못 본다(_apply_translation_lang의 link_source_video_id 주석 참고).
                return await _apply_translation_lang(
                    session, video_id, resp, lang, background_tasks, link.source_video_id
                )

        return SyncLookupResponse(found=False, user_offset=user_offset)


@router.get("/{video_id}/previous", response_model=SyncPreviousVersionResponse)
async def get_previous_sync_version(video_id: str):
    """이 영상 **자기 싱크**의 직전(재처리로 덮어써지기 전) 버전 스냅샷을 조회한다.

    확장의 재처리 전/후 A/B 고스트 비교용 — 롤백 자체는 이번 스코프 밖(확장 대개편 때
    설계). 링크(sync_links)로 빌려온 싱크는 다루지 않는다 — 여기서 보는 건 **이 video_id
    자신의** 이력뿐이고 링크 해소 로직과는 무관하다(빌려온 영상이 previous를 조회하면
    항상 found=false).

    스냅샷은 (video_id당 최신 1건) `SyncRepository.create()`가 새 sync_results 행을 넣기
    직전에 만든다 — 최초 생성뿐이었거나 아직 한 번도 재처리되지 않았으면 스냅샷이 없어
    found=false다. 미존재를 404가 아니라 found=false로 답하는 것은 기존 GET
    /api/sync/{video_id}와 같은 관례다."""
    _validate_video_id(video_id)
    async with get_session() as session:
        version = await SyncResultVersionRepository(session).get(video_id)
        if not version:
            return SyncPreviousVersionResponse(found=False)
        return SyncPreviousVersionResponse(
            found=True,
            timestamps=(version.timestamps or {}).get("segments", []),
            language=version.language,
            quality_score=version.quality_score,
            created_at=version.created_at.isoformat() if version.created_at else None,
            replaced_at=version.replaced_at.isoformat() if version.replaced_at else None,
            lyrics_hash=version.lyrics_hash,
            engine_variant=version.engine_variant,
            engine_version=version.engine_version,
        )


def _depth_of(result: "SyncResult") -> str | None:
    """sync_results 한 행의 timestamps.debug.routing.route — null-safe 추출. 새 정렬
    스택(라우팅 판정)이 만든 행만 채워진다 — 레거시 스택·라우팅 판정 자체가 없던 행은
    debug나 routing 키가 아예 없어 그대로 None으로 떨어진다."""
    debug = (result.timestamps or {}).get("debug") or {}
    routing = debug.get("routing") or {}
    return routing.get("route")


class SyncVersionSummary(BaseModel):
    """디버그 패널의 신구/깊이별 비교용 — sync_results 한 행의 요약(무거운 timestamps
    본문은 뺀다, 목록 하나가 최대 10건이라도 세그 전체를 다 실으면 무거워진다)."""

    id: str
    engine: str
    engine_variant: str | None = None
    engine_version: str | None = None
    language: str | None = None
    quality_score: float | None = None
    created_at: str | None = None
    # fast|medium|heavy — 새 스택 라우팅 판정(_depth_of). 레거시 행은 None.
    depth: str | None = None


class SyncVersionListResponse(BaseModel):
    versions: list[SyncVersionSummary]


@router.get("/{video_id}/versions", response_model=SyncVersionListResponse)
async def list_sync_versions(video_id: str):
    """이 영상의 sync_results 행 목록 — 최신순 최대 10건. SyncResult는 재생성마다 새 행을
    쌓는(UPDATE가 아니라 INSERT-only, models.SyncResult 독스트링) 이력이라, 이 목록이
    디버그 패널의 신구/깊이별(fast/medium/heavy) 비교 후보 전체다."""
    _validate_video_id(video_id)
    async with get_session() as session:
        results = await SyncRepository(session).get_by_video(video_id)
        return SyncVersionListResponse(
            versions=[
                SyncVersionSummary(
                    id=r.id,
                    engine=r.engine,
                    engine_variant=r.engine_variant,
                    engine_version=r.engine_version,
                    language=r.language,
                    quality_score=r.quality_score,
                    created_at=r.created_at.isoformat() if r.created_at else None,
                    depth=_depth_of(r),
                )
                for r in results[:10]
            ]
        )


class SyncVersionDetailResponse(BaseModel):
    """한 세대(sync_results 행 하나)의 전체 내용 — 목록에서 고른 두 세대를 실제로 그려
    비교하는 데 쓴다."""

    id: str
    timestamps: list[dict[str, Any]]
    language: str | None = None
    quality_score: float | None = None
    created_at: str | None = None
    engine_version: str | None = None
    depth: str | None = None


@router.get("/{video_id}/versions/{result_id}", response_model=SyncVersionDetailResponse)
async def get_sync_version_detail(video_id: str, result_id: str):
    """한 세대의 전체 timestamps(segments) — video_id가 그 행의 실제 소유자가 아니면
    404(다른 영상의 id로 남의 싱크 본문을 엿보는 것을 막는다)."""
    _validate_video_id(video_id)
    async with get_session() as session:
        result = await SyncRepository(session).get_by_id(result_id)
        if result is None or result.video_id != video_id:
            raise HTTPException(status_code=404, detail="해당 세대를 찾을 수 없어요")
        return SyncVersionDetailResponse(
            id=result.id,
            timestamps=(result.timestamps or {}).get("segments", []),
            language=result.language,
            quality_score=result.quality_score,
            created_at=result.created_at.isoformat() if result.created_at else None,
            engine_version=result.engine_version,
            depth=_depth_of(result),
        )


@router.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """정렬 품질 별점(1~5) + 선택 오류 제보 수집 (확장 별점 UI, 2026-08-03).

    신클라이언트가 화면에 실제로 떠 있던 ``sync_id``를 보내면 그 세대에 귀속한다. 필드가
    없는 구클라이언트만 제출 시점 최신 행으로 폴백한다. 명시적 null·삭제된 세대·무관한
    세대는 피드백 자체는 받되 sync_id=None으로 남겨 오귀속하지 않는다."""
    async with get_session() as session:
        repo = SyncRepository(session)
        attributed = None
        sync_id_supplied = "sync_id" in request.model_fields_set
        if sync_id_supplied:
            if request.sync_id:
                candidate = await repo.get_by_id(request.sync_id)
                if candidate is not None and candidate.video_id == request.video_id:
                    attributed = candidate
                elif candidate is not None:
                    # 링크 화면은 source 영상의 실제 싱크를 시프트해 보여준다. 현재 링크가 그
                    # source를 가리킬 때만 타 영상 행 귀속을 허용한다.
                    link = await SyncLinkRepository(session).get(request.video_id)
                    if link is not None and link.source_video_id == candidate.video_id:
                        attributed = candidate
        else:
            syncs = await repo.get_by_video(request.video_id)
            attributed = syncs[0] if syncs else None

        derived_depth = _depth_of(attributed) if attributed is not None else None
        # sync_id가 정확하면 행 메타가 정본이다. 필드가 없는 구형 클라이언트는 사용자가 본
        # 화면 depth를 이미 보내고 있었으므로 최신 행의 depth로 덮지 않는다.
        feedback_depth = (
            derived_depth or request.depth
            if sync_id_supplied and attributed is not None
            else request.depth or derived_depth
        )
        session.add(
            SyncFeedback(
                video_id=request.video_id,
                sync_id=attributed.id if attributed else None,
                rating=request.rating,
                category=request.category,
                comment=request.comment,
                engine_version=(
                    getattr(attributed, "engine_version", None) if attributed else None
                ),
                depth=feedback_depth,
            )
        )
        # 커밋은 get_session 컨텍스트가 수행한다 (이 모듈의 다른 쓰기 경로와 동일)
    return {"ok": True}


@router.post("/{video_id}/translations", response_model=SaveTranslationLayerResponse)
async def save_translation_layer(video_id: str, request: SaveTranslationLayerRequest):
    """완성된 싱크에 **이미 확보한** 번역(수동자막·위키 사람 번역 등)을 레이어로 직접 저장한다.

    다른 두 저장 경로와의 차이: `POST /api/translate`(persist=true)는 서버가 새로 번역해
    origin="llm"으로 저장하고, `POST /jobs/{job_id}/line-meta`는 진행 중인 생성 잡 전용
    (job_id 필수)이다. 이 엔드포인트는 **완성된 싱크를 보는 중(잡이 없음)**에 확장이 이미
    가진 번역 텍스트를 그대로 옮긴다 — 재번역도, 워커 개입도 없다.

    lines의 줄 분할이 이 영상 세그먼트의 줄 분할과 달라도(위키가 두 줄을 하나로 합쳤거나
    한 줄을 여럿으로 쪼갠 경우) `align_translation_lines`(순서 보존 결합 매칭, text_
    fingerprint.py)로 세그 텍스트 기준으로 재정렬해 저장한다 — 실증(H7PR): miraheze
    기반 새 싱크(36줄)와 vocaro 위키(42줄)의 줄 분할이 달라 줄 단위 동등 매칭이 50%를
    못 넘겨 위키 번역이 있는데도 LLM이 다시 불려 저품질 결과가 고착됐다. 엉뚱한 가사
    (다른 곡·다른 버전)가 붙는 사고를 막기 위해 재정렬 커버리지(세그 중 번역이 매겨진
    비율)가 절반 이상이어야 저장한다 — 미달이면 422와 함께 실제 커버리지를 알린다.
    **저장되는 lines는 항상 세그 자신의 텍스트로 재키잉된다** — 지문(fingerprint)이
    세그 텍스트 기준이므로, 원래 제출된(다른 분할의) 텍스트를 그대로 저장하면 이 조회가
    쓰는 지문과 어긋나 저장은 되는데 못 찾는 레이어가 된다.

    기존 레이어가 있으면 그 origin이 이번 요청과 같거나("caption"을 다시 "caption"으로
    갱신 등) 기존이 "llm"일 때만 교체한다. 그 외(예: 기존이 "wiki"인데 "caption"으로
    요청)는 **에러가 아니라 saved=false로 조용히 거절**한다 — 사람이 확인한 위키 번역을
    자동 자막이 덮어쓰지 못하게 한다("llm"은 이 엔드포인트가 절대 만들지 않는 origin이라
    /api/translate persist 결과를 밀어내는 경로만 여기 남는다).

    링크로 빌려온 영상(자기 싱크가 없고 SyncLink만 있는 경우)도 지원한다 — 세그먼트는
    링크 소스에서 가져오되(매칭·지문 계산용), 레이어는 **요청받은 video_id**를 키로 쓴다
    (source_video_id 아님) — GET 조회·POST /api/translate persist와 같은 키 관례를
    지켜야 서로 찾아진다. 우선순위는 get_sync와 동일하게 자기 싱크 > 링크."""
    _validate_video_id(video_id)
    _validate_translation_layer_request(request)

    async with get_session() as session:
        sync_repo = SyncRepository(session)
        own = await sync_repo.get_by_video(video_id)
        if own:
            segments = (own[0].timestamps or {}).get("segments", []) or []
        else:
            link = await SyncLinkRepository(session).get(video_id)
            source_syncs = (
                await sync_repo.get_by_video(link.source_video_id) if link else None
            )
            if not source_syncs:
                raise HTTPException(status_code=404, detail="이 영상의 싱크를 찾을 수 없어요")
            segments = (source_syncs[0].timestamps or {}).get("segments", []) or []

        seg_texts = [seg.get("text", "") or "" for seg in segments]
        submitted = [{"text": ln.text, "translation": ln.translation} for ln in request.lines]
        remapped = align_translation_lines(seg_texts, submitted)

        total = len(seg_texts)
        matched = sum(1 for value in remapped if value is not None)
        if total == 0 or matched / total < _MIN_TRANSLATION_LAYER_MATCH_RATIO:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"저장하려는 번역 라인이 이 영상의 가사와 거의 안 맞아요 "
                    f"({matched}/{total}줄 일치) — 다른 곡·다른 버전 가사가 아닌지 확인해 주세요."
                ),
            )

        fingerprint = lines_fingerprint(seg_texts)
        repo = TranslationLayerRepository(session)
        existing = await repo.get_layer(video_id, fingerprint, request.target_lang)
        if existing is not None and existing.origin not in (request.origin, "llm"):
            return SaveTranslationLayerResponse(
                saved=False, matched=matched, total=total, target_lang=request.target_lang
            )

        saved_layer = await repo.upsert_layer(
            video_id,
            fingerprint,
            request.target_lang,
            # 세그 텍스트로 재키잉 — 지문과 정합되게(위 docstring 참고)
            lines=[
                {"text": text, "translation": translation}
                for text, translation in zip(seg_texts, remapped)
                if translation is not None
            ],
            attribution=request.attribution.model_dump() if request.attribution else None,
            origin=request.origin,
        )
        # None = 라벨-내용 언어 불일치로 거부됨(repository 가드) — 저장한 척하지 않는다
        return SaveTranslationLayerResponse(
            saved=saved_layer is not None, matched=matched, total=total, target_lang=request.target_lang
        )


@router.delete("/{video_id}")
async def reset_video_syncs(
    video_id: str,
    x_api_key: str | None = Header(default=None),
    x_lyric_user: str | None = Header(default=None),
    x_lyric_limits: str | None = Header(default=None),
):
    """이 영상의 서버 싱크를 전부 삭제(초기화) — 잘못 붙여넣은 가사 등에서 새로 시작.

    이 영상이 소유자이거나 소스인 링크도 함께 제거한다 ("/link/{video_id}"가 먼저
    선언돼 있어 링크 삭제 경로와 충돌하지 않는다). 공개 배포에선 파괴적 행위 합산 예산
    (초기화+강제 재생성)이 걸린다.

    **멱등하다**(docs/user-quota-spec.md §4): 응답이 유실돼 클라이언트가 재시도하면 지울
    것은 이미 없어 아무 일도 일어나지 않는데 예산만 줄었다. 검사는 삭제 전에 하되(초과한
    요청은 지우지도 못하게 막아야 한다) **기록은 실제로 지운 것이 있을 때만** 남긴다."""
    _validate_video_id(video_id)
    async with get_session() as session:
        await _check_destructive_limit(
            session, "reset", video_id, x_lyric_user, x_api_key, x_lyric_limits, record=False
        )
        removed_syncs = await SyncRepository(session).delete_by_video(video_id)
        removed_links = await SyncLinkRepository(session).delete_involving(video_id)
        if removed_syncs or removed_links:
            await _record_action(session, "reset", video_id, x_lyric_user, x_api_key)
        return {
            "video_id": video_id,
            "removed_syncs": removed_syncs,
            "removed_links": removed_links,
        }


@router.post("/generate", response_model=GenerateResponse)
async def generate_sync(
    request: GenerateRequest,
    background_tasks: BackgroundTasks,
    x_api_key: str | None = Header(default=None),
    x_lyric_user: str | None = Header(default=None),
    x_lyric_limits: str | None = Header(default=None),
):
    """가사로 싱크 생성 잡을 만든다 (기존 싱크가 있으면 즉시 completed).

    line_meta(발음/번역)는 **선택**이다. 본문에 실어 보내는 기존 경로가 그대로 동작하고,
    아직 번역이 안 끝났으면 line_meta_pending=true로 잡을 먼저 만들어 다운로드·보컬 분리를
    선행시킨 뒤 POST /api/sync/jobs/{job_id}/line-meta로 붙일 수 있다 (응답의
    line_meta_wait_sec이 서버가 실제로 기다려 주는 상한).

    공개 배포(admin_api_key 설정)에서는 새 잡을 만드는 요청에 DAILY_GENERATE_LIMIT이
    걸린다 — 캐시 히트·진행 중 잡 합류는 GPU를 쓰지 않으므로 세지 않는다."""
    from everyric2.server.worker import LINE_META_WAIT_SEC

    _validate_lyrics(request.lyrics)
    lyrics_hash_value = hash_lyrics(request.lyrics)
    # 본문에 line_meta가 이미 있으면 기다릴 것이 없다 — 플래그보다 실제 값이 우선
    await_line_meta = request.line_meta_pending and not request.line_meta
    wait_sec = LINE_META_WAIT_SEC if await_line_meta else 0.0

    # 활성 잡을 보기 **전에** 만료 리스를 회수한다 (락 밖 — 중첩 세션 금지).
    # 죽은 워커가 물고 있던 잡은 processing에 남아 get_active_by_video에 활성으로 잡히고,
    # 그러면 이 요청이 죽은 잡에 합류해 그 (영상, 가사)는 재기동까지 재생성 불가가 된다.
    # 주기 스윕이 이미 그 일을 하지만 여기서 한 번 더 하는 것이 **이벤트 루프에 의존하지
    # 않는 방어선**이다 — 주기 태스크가 아직 안 돌았거나(간격 이내) 죽어 있어도 봉인이 풀린다.
    await reclaim_expired_leases()

    # 확인(기존 싱크/활성 잡)→생성 사이에 다른 요청이 끼면 중복 잡이 생긴다 — 직렬화
    async with _CREATE_LOCK, get_session() as session:
        sync_repo = SyncRepository(session)
        existing = await sync_repo.get_by_video_and_hash(request.video_id, lyrics_hash_value)
        if existing:
            # 정렬은 재사용하되 새로 들어온 발음/번역 메타·출처는 반영한다
            if request.line_meta or request.attribution:
                _merge_meta_into_sync(existing, request.line_meta, request.attribution)
            # 제목이 비어 있던 기존 싱크는 이 기회에 채운다 (기존 값은 덮어쓰지 않는다)
            await sync_repo.set_title_if_missing(existing, request.title, request.artist)
            return GenerateResponse(
                job_id=existing.id,
                status="completed",
                estimated_time=0,
            )

        job_repo = JobRepository(session)
        # 같은 영상·가사로 이미 돌고 있는 잡이 있으면 새 잡을 만들지 않고 합류한다 —
        # 버튼 연타로 동일 잡이 중복 생성되면 같은 임시 오디오 파일을 두 작업이 잡아
        # Windows에서 WinError 32(파일 사용 중)로 다운로드가 깨진다
        active = await job_repo.get_active_by_video(request.video_id, lyrics_hash_value)
        if active:
            from everyric2.server.worker import (
                stash_attribution,
                stash_line_meta,
                stash_title,
            )

            if request.line_meta:
                stash_line_meta(
                    active.id,
                    [m.model_dump() for m in request.line_meta],
                    request.line_meta_lang,
                )
            if request.attribution:
                stash_attribution(active.id, request.attribution.model_dump())
            stash_title(active.id, request.title, request.artist)
            # 합류한 잡이 이미 정렬에 들어갔을 수도 있으니 상한은 어디까지나 상한이다 —
            # line-meta 붙이기는 받아 주고(늦으면 완성된 싱크에 병합된다) 값은 그대로 알린다
            return GenerateResponse(
                job_id=active.id,
                status="processing",
                estimated_time=15,
                line_meta_wait_sec=wait_sec,
            )
        # 여기부터가 실제로 GPU를 태우는 유일한 분기 — 한도 검사는 이 지점이어야 한다
        # (위의 캐시 히트·합류에서 예산을 먹으면 정상 사용이 헛되게 소모된다)
        await _check_action_limit(
            session, "generate", request.video_id, x_lyric_user, x_api_key, x_lyric_limits
        )
        job = await job_repo.create(
            video_id=request.video_id,
            lyrics=request.lyrics,
            language=request.language,
            target_lang=request.target_lang,
        )
        job_id = job.id

    from everyric2.server.worker import stash_attribution, stash_line_meta, stash_title

    if request.line_meta:
        stash_line_meta(
            job_id, [m.model_dump() for m in request.line_meta], request.line_meta_lang
        )
    if request.attribution:
        stash_attribution(job_id, request.attribution.model_dump())
    stash_title(job_id, request.title, request.artist)
    await _dispatch_job(job_id, background_tasks, await_line_meta=await_line_meta)

    return GenerateResponse(
        job_id=job_id,
        status="processing",
        estimated_time=15,
        line_meta_wait_sec=wait_sec,
    )


class LineMetaAttachRequest(BaseModel):
    """진행 중인 생성 잡에 나중에 붙이는 번역·독음.

    **line_meta를 빈 배열로 보내면 "붙일 것이 없음"이 확정**돼 워커가 즉시 원문 정렬로
    진행한다 — 클라이언트가 번역에 실패했을 때 반드시 이걸 한 번 보내야 잡이 대기 상한까지
    헛되게 서 있지 않는다.
    """

    line_meta: list[LineMeta] = Field(default_factory=list)
    attribution: Attribution | None = None
    title: str | None = Field(default=None, max_length=256)
    artist: str | None = Field(default=None, max_length=128)
    # line_meta에 실린 번역의 언어 — GenerateRequest.line_meta_lang과 같은 계약.
    # 워커의 resolve_layer_lang이 레이어 언어·legacy 병기 판정에 쓴다.
    line_meta_lang: str = Field(default="ko", max_length=8)


class LineMetaAttachResponse(BaseModel):
    job_id: str
    # 잡의 현재 상태 (pending | queued | processing | completed | failed)
    status: str
    # stashed = 잡이 아직 진행 중 → 정렬(또는 최소한 결과 저장)에 반영된다
    # merged  = 잡이 이미 완료돼 완성된 싱크에 직접 병합했다
    # dropped = 잡이 실패/취소됐거나 완료 싱크를 찾지 못해 아무것도 하지 않았다
    applied: str
    # merged일 때 메타가 붙은 세그먼트 수
    merged_segments: int = 0


async def _merge_meta_into_completed_job(
    session,
    job,
    line_meta: list[LineMeta],
    attribution: Attribution | None,
    title: str | None,
    artist: str | None,
    line_meta_lang: str = "ko",
) -> LineMetaAttachResponse:
    """완료된 잡의 싱크에 메타를 직접 병합 — merged, 싱크를 못 찾으면 dropped.

    정렬은 다시 하지 않는다 (캐시 히트로 몇 초 만에 끝난 잡이 대표적).
    번역은 워커 완료 경로와 같은 규칙으로 언어 레이어에도 기록한다 — 이 경로만 빠지면
    늦게 붙인 번역이 legacy(ko 한정)에만 남고 언어별 조회가 영영 못 찾는다.
    """
    from everyric2.server.worker import (
        layer_origin,
        record_translation_layer,
        translation_layer_lines,
    )

    sync_repo = SyncRepository(session)
    existing = await sync_repo.get_by_video_and_hash(job.video_id, job.lyrics_hash)
    if existing is None:
        return LineMetaAttachResponse(job_id=job.id, status=job.status, applied="dropped")
    merged = _merge_meta_into_sync(existing, line_meta, attribution, line_meta_lang)
    attr_dump = attribution.model_dump() if attribution else None
    await record_translation_layer(
        session,
        job.video_id,
        [s.get("text") or "" for s in existing.timestamps.get("segments", [])],
        translation_layer_lines([m.model_dump() for m in line_meta]),
        line_meta_lang,
        origin=layer_origin(attr_dump),
        attribution=attr_dump,
    )
    await sync_repo.set_title_if_missing(existing, title, artist)
    return LineMetaAttachResponse(
        job_id=job.id, status=job.status, applied="merged", merged_segments=merged
    )


async def _attach_line_meta_to_job(
    job_id: str,
    line_meta: list[LineMeta],
    attribution: Attribution | None = None,
    title: str | None = None,
    artist: str | None = None,
    line_meta_lang: str = "ko",
) -> LineMetaAttachResponse | None:
    """번역·독음을 잡에 붙이는 실제 동작 — HTTP 엔드포인트와 서버 내부 생성 경로의 공용 몸통.

    잡을 찾지 못하면 None. HTTP 계약(404)으로 바꾸는 것은 엔드포인트의 몫이고, 내부
    호출자에게는 예외가 아니라 값으로 알려야 한다(백그라운드에서 올린 예외는 아무도 안 본다).
    """
    from everyric2.server.worker import stash_attribution, stash_line_meta, stash_title

    async with get_session() as session:
        job = await JobRepository(session).get_by_id(job_id)
        if not job:
            return None

        if job.status == "completed":
            return await _merge_meta_into_completed_job(
                session, job, line_meta, attribution, title, artist, line_meta_lang
            )

        if job.status == "failed":
            # 취소·실패한 잡 — 스태시를 남기면 정리 지점 없이 새므로 아무것도 하지 않는다
            return LineMetaAttachResponse(job_id=job_id, status=job.status, applied="dropped")

        job_status = job.status

    # 빈 배열도 그대로 넣는다 — 스태시 키의 존재 자체가 워커에게 "도착 확정" 신호다
    stash_line_meta(job_id, [m.model_dump() for m in line_meta], line_meta_lang)
    if attribution:
        stash_attribution(job_id, attribution.model_dump())
    stash_title(job_id, title, artist)

    # 스태시를 쓴 **뒤에 상태를 다시 읽는다**. 위 읽기와 이 쓰기 사이에 잡이 종결되면(캐시
    # 히트 완료·취소·실패) 스태시를 거둘 주체가 사라져 프로세스 수명 동안 영구 잔류하고
    # (누수), 메타는 싱크에 병합되지도 않는데 응답은 applied="stashed"라 사실과 다르다.
    # 종결됐으면 스태시를 회수하고 완료 싱크에 직접 병합(merged)하거나 버린다(dropped) —
    # 실제로 일어난 일을 응답에 담는다.
    #
    # 재확인 이후에 종결되는 창은 남는다(터미널 처리는 워커 쪽 코드가 소유해 여기서 같은 락을
    # 걸 수 없다). 다만 그 경우 스태시는 워커의 터미널 정리(_pop_stashes)가 거두므로 누수는
    # 되지 않고, 남는 것은 "stashed로 답했는데 반영되지 못했다"는 좁은 창뿐이다.
    async with get_session() as session:
        job = await JobRepository(session).get_by_id(job_id)
        if job is not None and job.status in ("completed", "failed"):
            from everyric2.server.api.worker import _pop_stashes

            _pop_stashes(job_id)
            if job.status == "failed":
                return LineMetaAttachResponse(
                    job_id=job_id, status=job.status, applied="dropped"
                )
            return await _merge_meta_into_completed_job(
                session, job, line_meta, attribution, title, artist, line_meta_lang
            )

    return LineMetaAttachResponse(job_id=job_id, status=job_status, applied="stashed")


@router.post("/jobs/{job_id}/line-meta", response_model=LineMetaAttachResponse)
async def attach_line_meta(job_id: str, request: LineMetaAttachRequest):
    """생성 잡에 번역·독음(line_meta)을 나중에 붙인다 — 번역을 다운로드·분리와 겹치는 경로.

    호출 순서: POST /api/sync/generate (line_meta_pending=true, line_meta 없이) → job_id 확보
    → 클라이언트가 번역·독음을 만드는 동안 서버는 다운로드·보컬 분리를 진행 → 이 엔드포인트
    → GET /api/job/{job_id} 폴링.

    잡이 아직 정렬 전이면 그 발음 텍스트로 정렬이 이뤄지고(독음 정렬), 대기 상한을 넘겨 이미
    원문으로 정렬됐거나 잡이 끝났으면 발음·번역 텍스트만 결과에 병합된다. 어느 쪽이든 호출은
    성공하며 applied가 무엇이 일어났는지 알린다 — 클라이언트는 분기할 필요가 없다.
    """
    from everyric2.server.api.job import _validate_job_id

    _validate_job_id(job_id)
    applied = await _attach_line_meta_to_job(
        job_id,
        request.line_meta,
        request.attribution,
        request.title,
        request.artist,
        request.line_meta_lang,
    )
    if applied is None:
        raise HTTPException(status_code=404, detail="잡을 찾을 수 없어요")
    return applied


class GenerateFromCaptionRequest(BaseModel):
    """video_id만으로 싱크를 만든다 — 가사는 서버가 유튜브 자막에서 조달한다.

    title/artist는 선택이다(커버 링크 후보 탐색의 단서로 함께 저장될 뿐, 자막 판정에는
    쓰이지 않는다). **자막 트랙을 고르는 필드는 일부러 두지 않았다** — 원어 판정은
    전적으로 서버 몫이고, 사용자가 고르는 단계를 없애는 것이 이 엔드포인트의 목적이다.
    """

    video_id: str = Field(pattern=_VIDEO_ID_PATTERN)
    title: str | None = Field(default=None, max_length=256)
    artist: str | None = Field(default=None, max_length=128)


class CaptionGenerateResponse(GenerateResponse):
    """생성 응답 + 실제로 어떤 자막을 왜 썼는지 — 클라이언트 표시·로그 확인용."""

    lang: str
    auto: bool
    track_label: str
    # 원어 판정 근거 (asr_orig | asr_only | video_language | sole_manual)
    reason: str
    line_count: int


# 가나(U+3040–U+30FF)와 CJK 한자(U+3400–U+9FFF) — 한글은 포함되지 않는다
_CJK_RE = re.compile(r"[぀-ヿ㐀-鿿]")


def _expects_pronunciation(lines: list[str]) -> bool:
    """이 원문에 발음표기(한글 독음)가 의미가 있는가.

    확장의 expectsPronunciation(content.ts)과 같은 규칙·같은 임계(5자)를 쓴다 — 한국어 곡에
    한글 독음을 붙이는 건 무의미하고 LLM 시간만 늘리며, 임계를 두면 제목의 한자 한두 자
    같은 잡음으로는 켜지지 않는다. 두 경로가 다른 규칙을 쓰면 같은 곡이 어디서 생성됐는지에
    따라 독음이 있다가 없어진다.
    """
    return len(_CJK_RE.findall("".join(lines))) >= 5


async def _translate_and_attach_line_meta(
    job_id: str,
    lines: list[str],
    source_lang: str | None,
    video_id: str,
    title: str | None,
    artist: str | None,
    human_translations: list[str] | None = None,
) -> None:
    """자막 가사의 번역·독음을 만들어 잡에 붙인다 (서버가 가사를 조달한 경로 전용).

    **어떤 이유로 실패해도 반드시 한 번은 붙인다** — 빈 리스트가 "붙일 것 없음" 확정 신호라
    (worker._PENDING_LINE_META 규약) 아무것도 안 붙이면 워커가 정렬 진입 직전에 대기 상한
    (LINE_META_WAIT_SEC)을 통째로 헛되게 태운다. 그래서 예외는 여기서 끝내고 로그로만 남긴다.

    `human_translations`는 같은 영상의 한국어 수동 자막에서 온 사람 번역이다(`lines`와 같은
    길이, 빈 문자열은 «그 줄에는 없음»). 있으면 **그 줄의 기계 번역을 덮는다** — 사람이 옮긴
    번역이 더 낫고, 같은 영상 자막이라 맥락도 맞다. 독음은 사람 자막에 없으므로 여전히 만든다.
    """
    from starlette.concurrency import run_in_threadpool

    from everyric2.server.api.translate import TranslateRequest, translate_lyrics

    human = human_translations if human_translations and any(human_translations) else None
    wants_pron = _expects_pronunciation(lines)

    # 독음이 필요 없고 번역은 사람 것이 있으면 LLM을 부를 이유가 없다 — 한국어 원문 곡에
    # 한국어 자막이 붙어 있는 경우가 아니라(그때는 번역 자체를 안 만든다), 원문이 라틴 문자인
    # 곡에 한국어 팬 자막이 있는 경우가 여기 걸린다.
    if human and not wants_pron:
        meta = [
            LineMeta(text=src, pronunciation=None, translation=tr)
            for src, tr in zip(lines, human)
            if tr
        ]
        applied = await _attach_line_meta_to_job(job_id, meta)
        logger.info(
            "Job %s: caption line_meta from human captions only (%d/%d lines, applied=%s)",
            job_id, len(meta), len(lines), getattr(applied, "applied", None),
        )
        return

    meta: list[LineMeta] = []
    try:
        # 엔진 선택(EVERYRIC_TRANSLATE_ENGINE)·톤·가나 오염 재시도까지 /api/translate와 완전히
        # 같은 경로를 쓴다 — 별도 호출을 만들면 두 경로의 번역 품질이 조용히 갈린다.
        # 동기 LLM 호출(수십 초)이라 이벤트 루프 밖으로 내보낸다: 같은 루프에서 이 잡의
        # 다운로드·보컬 분리가 돌고 있다.
        from fastapi import BackgroundTasks

        result = await run_in_threadpool(
            translate_lyrics,
            TranslateRequest(
                text="\n".join(lines),
                source_lang=source_lang or "auto",
                include_pronunciation=wants_pron,
                title=title,
                artist=artist,
                video_id=video_id,
            ),
            # persist를 안 쓰므로(기본 False) 실행될 일 없는 껍데기 — translate_lyrics가
            # BackgroundTasks를 필수로 받게 되어(POST /api/translate의 persist 브리지)
            # 직접 호출하는 이 경로도 인스턴스를 함께 넘겨야 한다.
            BackgroundTasks(),
        )
        # LLM이 echo한 original이 아니라 넘긴 원문으로 text를 채운다 — 병합(merge_line_meta)은
        # 정규화 텍스트 매칭이라 한 글자만 달라도 그 줄은 붙지 않는다. 줄 수가 같아 인덱스로
        # 대응시킬 수 있고, 짧은 응답이 와도 zip이 남는 줄을 조용히 버린다.
        for i, (src, line) in enumerate(zip(lines, result.lines)):
            pron = (line.pronunciation or "").strip() or None
            trans = (line.translation or "").strip() or None
            # 사람 번역이 그 줄에 있으면 기계 번역을 덮는다
            if human and i < len(human) and human[i]:
                trans = human[i]
            if pron or trans:
                meta.append(LineMeta(text=src, pronunciation=pron, translation=trans))
    except Exception:
        logger.exception("Job %s: caption line_meta translation failed", job_id)
        # LLM이 죽어도 사람 번역은 살아 있다 — 그것만이라도 붙인다
        if human:
            meta = [
                LineMeta(text=src, pronunciation=None, translation=tr)
                for src, tr in zip(lines, human)
                if tr
            ]

    applied = await _attach_line_meta_to_job(job_id, meta)
    if applied is None:
        logger.warning("Job %s: vanished before caption line_meta could be attached", job_id)
    else:
        logger.info(
            "Job %s: caption line_meta attached (%d/%d lines, applied=%s)",
            job_id,
            len(meta),
            len(lines),
            applied.applied,
        )


async def _process_caption_job(
    job_id: str,
    lines: list[str],
    source_lang: str | None,
    video_id: str,
    title: str | None,
    artist: str | None,
    pipeline: BackgroundTasks,
    human_translations: list[str] | None = None,
) -> None:
    """번역·독음 생성과 잡 처리(인프로세스 파이프라인 또는 원격 큐 진입)를 **동시에** 돌린다.

    둘을 각각 add_task로 걸면 안 된다: Starlette의 BackgroundTasks는 등록 순서대로 하나씩
    await하므로 먼저 걸린 잡 처리가 아직 시작조차 안 한 번역을 대기 상한까지 기다리고
    (원격 경로에선 그사이 큐에 올라가 번역을 통째로 놓친다), line_meta_pending이 노리는
    "다운로드·보컬 분리와 번역이 겹친다"가 성립하지 않는다.
    """
    await asyncio.gather(
        _translate_and_attach_line_meta(
            job_id, lines, source_lang, video_id, title, artist, human_translations
        ),
        pipeline(),
    )


@router.post("/generate-from-caption", response_model=CaptionGenerateResponse)
async def generate_sync_from_caption(
    request: GenerateFromCaptionRequest,
    background_tasks: BackgroundTasks,
    x_api_key: str | None = Header(default=None),
    x_lyric_user: str | None = Header(default=None),
    x_lyric_limits: str | None = Header(default=None),
):
    """video_id만으로 유튜브 자막을 조달해 싱크 생성 잡을 만든다.

    자막 사용 가능 여부 판정 → 원어 트랙 자동 선택 → 본문 취득 → 가사 텍스트 구성까지
    서버가 하고, 그 뒤는 **/generate와 완전히 같은 경로**로 넘긴다(중복 싱크 재사용,
    활성 잡 합류, 큐 적재가 그대로 적용된다 — 여기서 복제하지 않는다).

    자막 타임스탬프는 버린다. 자막 타이밍은 가사 표시용이라 발성 시점과 어긋나고,
    정렬은 어차피 CTC가 오디오에서 새로 잡는다.

    **번역·독음도 이 경로에서는 서버가 만든다.** 클라이언트는 자기가 본 자막 라인 분할만
    알지, 정렬에 실제로 쓰이는 분할(clean_caption_lines·merge_rolling)은 서버 쪽이다.
    line_meta 병합은 정규화 텍스트 매칭이라 분할이 어긋나면 한 줄도 붙지 않는다 — 가사를
    조달한 쪽이 메타도 소유해야 번역·독음이 실제로 싱크에 남는다.

    실패는 detail={code, message}로 나간다. 4xx는 이 영상이 자막으로는 불가능하다는
    확정 판정이므로 클라이언트는 가사 직접 붙여넣기로 안내하면 된다. 5xx는 조달 실패라
    재시도 가치가 있다.
    """
    from starlette.concurrency import run_in_threadpool

    from everyric2.server.services.youtube_captions import (
        CaptionUnavailable,
        fetch_lyrics_from_captions,
    )

    try:
        # yt-dlp는 블로킹 IO라 이벤트 루프 밖으로 내보낸다 (extract + 트랙 다운로드 2회)
        found = await run_in_threadpool(fetch_lyrics_from_captions, request.video_id)
    except CaptionUnavailable as e:
        message = e.message
        if e.terminal:
            message = f"{message} — 가사를 직접 붙여넣어 주세요"
        raise HTTPException(
            status_code=e.http_status, detail={"code": e.code, "message": message}
        ) from e

    track = found.track
    # /generate가 등록하는 잡 처리 작업을 별도 컨테이너로 받는다 — background_tasks에 그대로
    # 얹으면 번역 작업과 순차 실행돼 겹치지 않는다 (_process_caption_job 참고)
    pipeline = BackgroundTasks()
    base = await generate_sync(
        GenerateRequest(
            video_id=request.video_id,
            lyrics=found.text,
            lyrics_source="youtube_caption",
            # CTC가 다루는 언어일 때만 지정한다 — 그 밖(gl/fil 등)은 엔진의 텍스트
            # 기반 자동 판정이 더 낫다 (services.youtube_captions.ALIGNABLE_LANGS)
            language=found.align_language,
            attribution=Attribution(
                name=f"유튜브 자막 · {track.label}",
                url=f"https://www.youtube.com/watch?v={request.video_id}",
            ),
            title=request.title,
            artist=request.artist,
            # 번역·독음은 아래 백그라운드 작업이 만들어 붙인다 — 정렬 진입 직전까지 기다려
            # 주므로 다운로드·보컬 분리와 겹치고, 독음이 붙으면 독음 정렬 경로를 탄다
            line_meta_pending=True,
        ),
        pipeline,
        # 어드민 키는 이 경로에서도 일일 생성 상한을 면제받아야 한다 (같은 검사를 재사용)
        x_api_key,
        # 이용자 축 한도도 같은 검사를 그대로 탄다 — 자막 경로만 예산을 비켜 가면
        # "가사 붙여넣기 대신 자막으로 부르면 무제한"이 된다
        x_lyric_user,
        # 앞단이 준 상한도 함께 넘긴다 — 안 넘기면 이 경로만 서버 기본값으로 집행돼
        # 티어 차등이 자막 경로에서만 사라진다(§6)
        x_lyric_limits,
    )
    if base.status != "completed":
        # completed는 같은 자막 가사의 싱크를 그대로 재사용한 경우다 — job_id가 잡이 아니라
        # 싱크 id라 붙일 잡이 없고, generate_sync도 처리 작업을 등록하지 않는다(pipeline 비어
        # 있음). line_meta_pending도 그 경로에서 이미 무시된다(응답의 wait_sec=0).
        background_tasks.add_task(
            _process_caption_job,
            base.job_id,
            found.lines,
            track.language,
            request.video_id,
            request.title,
            request.artist,
            pipeline,
            found.translations,
        )
    return CaptionGenerateResponse(
        **base.model_dump(),
        lang=track.lang,
        auto=track.auto,
        track_label=track.label,
        reason=track.reason,
        line_count=len(found.lines),
    )


@router.post("/search-by-audio", response_model=SyncLookupResponse)
async def search_by_audio_hash(request: SearchByAudioRequest):
    async with get_session() as session:
        repo = SyncRepository(session)
        result = await repo.get_by_audio_hash(request.audio_hash)
        if result:
            return SyncLookupResponse(
                found=True,
                sync_id=result.id,
                timestamps=result.timestamps.get("segments", []),
                lyrics_source=result.engine,
                quality_score=result.quality_score,
                audio_hash=result.audio_hash,
                language=result.language,
                engine_variant=result.engine_variant,
                engine_version=result.engine_version,
                created_at=result.created_at.isoformat() if result.created_at else None,
            )
        return SyncLookupResponse(found=False)


@router.get("/list/{video_id}")
async def list_syncs_for_video(video_id: str):
    async with get_session() as session:
        repo = SyncRepository(session)
        results = await repo.get_by_video(video_id)
        return {
            "video_id": video_id,
            "syncs": [
                {
                    "sync_id": r.id,
                    "lyrics_hash": r.lyrics_hash,
                    "audio_hash": r.audio_hash,
                    "quality_score": r.quality_score,
                    "language": r.language,
                    "engine_variant": r.engine_variant,
                    "engine_version": r.engine_version,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in results
            ],
        }


class SearchSyncRequest(BaseModel):
    title: str | None = None
    artist: str | None = None
    limit: int = 10


@router.post("/search")
async def search_available_syncs(request: SearchSyncRequest):
    async with get_session() as session:
        repo = SyncRepository(session)
        results = await repo.get_all_unique_videos(limit=request.limit * 3)
        return {
            "syncs": [
                {
                    "video_id": r.video_id,
                    "audio_hash": r.audio_hash,
                    "quality_score": r.quality_score,
                    "language": r.language,
                    "engine_variant": r.engine_variant,
                    "engine_version": r.engine_version,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "lyrics_preview": _get_lyrics_preview(r.timestamps),
                }
                for r in results
            ]
        }


def _get_lyrics_preview(timestamps: dict) -> str:
    segments = timestamps.get("segments", [])
    if not segments:
        return ""
    texts = [s.get("text", "") for s in segments[:3]]
    return " / ".join(texts)[:100]


def _is_upgrade_request(request: "RegenerateRequest", latest_syncs: list) -> bool:
    """이 재생성 요청이 **정렬 업그레이드**인가(파괴적 재생성이 아니라).

    배경(docs/user-quota-spec.md §3): 확장의 API 래퍼가 재생성 요청에 force를 **무조건**
    붙인다. 서버가 force만 보고 파괴적 행위로 판정하는 바람에, 확장에서 그 경로를 부르는
    유일한 버튼인 "정렬 깊이 올리기"가 파괴적 예산을 먹었고 업그레이드 카운터는 도입 이후
    한 번도 기록되지 않았다(실측 2026-08-10 전 기간 0건).

    판정 세 갈래:
      ① min_depth가 있으면 업그레이드다 — force 유무와 무관하다(래퍼가 붙인 force는 요청의
         의도가 아니다).
      ② force인데 min_depth가 없고, 그 영상의 **최신 싱크가 스탬프 이전 세대**
         (engine_version IS NULL)면 업그레이드다 — 구세대 싱크용 업그레이드 버튼은 깊이
         인자 없이 호출하므로 ①에 걸리지 않는다.
      ③ 그 외 force는 파괴적 재생성이다.

    ②가 그 버튼보다 **넓다는 것은 인정된 결정**이다(운영자 2026-08-10): 같은 형태의 직접
    API 호출도 업그레이드로 분류된다. 확장에는 강제 재생성 UI가 없어 일반 이용자는 그 경로를
    탈 수 없고, 오분류가 나도 요청은 시킨 대로 수행되며 달라지는 것은 어느 예산이 줄어드는가
    뿐이다. 게다가 구세대 싱크는 한 번 업그레이드되면 스탬프가 찍혀 조건에서 빠지므로 이
    느슨함은 시간이 지나면 사라진다.

    latest_syncs는 호출부가 이미 (레거시 ko 백필 판정을 위해) 읽어 둔 최신순 목록이다 —
    같은 것을 다시 읽지 않는다."""
    if request.min_depth:
        return True
    if not request.force:
        return False
    return bool(latest_syncs) and latest_syncs[0].engine_version is None


@router.post("/regenerate", response_model=GenerateResponse)
async def regenerate_sync(
    request: RegenerateRequest,
    background_tasks: BackgroundTasks,
    x_api_key: str | None = Header(default=None),
    x_lyric_user: str | None = Header(default=None),
    x_lyric_limits: str | None = Header(default=None),
):
    from everyric2.server.worker import LINE_META_WAIT_SEC

    _validate_lyrics(request.lyrics)
    lyrics_hash_value = hash_lyrics(request.lyrics)
    await_line_meta = request.line_meta_pending and not request.line_meta
    wait_sec = LINE_META_WAIT_SEC if await_line_meta else 0.0

    # 재생성도 죽은 잡에 합류해 봉인될 수 있다 — 생성 경로와 같은 회수를 먼저 한다
    await reclaim_expired_leases()

    # 확인(기존 싱크/활성 잡)→생성 사이에 다른 요청이 끼면 중복 잡이 생긴다 — 직렬화
    async with _CREATE_LOCK, get_session() as session:
        sync_repo = SyncRepository(session)

        # 잡 생성 전 — 이 영상의 **최신** 싱크(새 lyrics_hash와 무관하게)에 레거시(세그
        # 전용) ko 번역이 남아 있고 아직 레이어가 없으면 지금 백필한다. 안 하면 이번
        # 재생성(특히 force)으로 그 세그 자체가 새 싱크로 갈아끼워지면서 옛 ko 번역
        # (위키 사람 번역 포함)을 되살릴 방법이 사라진다 — TranslationLayer.origin="legacy"
        # 주석 참고.
        latest_syncs = await sync_repo.get_by_video(request.video_id)
        if latest_syncs:
            latest_timestamps = latest_syncs[0].timestamps or {}
            await _schedule_ko_backfill_if_needed(
                session,
                background_tasks,
                request.video_id,
                latest_timestamps.get("segments", []) or [],
                latest_timestamps.get("attribution"),
            )

        # **한도 검사는 여기 없다** — 아래 두 조기 반환(캐시 히트·활성 잡 합류)을 지난
        # 뒤에야 GPU를 태우기 때문이다. 예전에는 force 검사가 이 위치(합류 판정 앞)라
        # 이미 도는 잡에 합류하는 재시도(버튼 연타)도 파괴적 예산을 깎았다
        # (docs/user-quota-spec.md §3).
        if not request.force and not request.min_depth:
            # min_depth(깊이 하한) 요청은 같은 가사의 기존 싱크가 **있어야** 성립하는
            # 재분석이다 — 이 조기 반환에 걸리면 아무 일도 안 일어나므로 건너뛴다.
            existing = await sync_repo.get_by_video_and_hash(request.video_id, lyrics_hash_value)
            if existing:
                if request.line_meta or request.attribution:
                    _merge_meta_into_sync(existing, request.line_meta, request.attribution)
                await sync_repo.set_title_if_missing(existing, request.title, request.artist)
                return GenerateResponse(
                    job_id=existing.id,
                    status="completed",
                    estimated_time=0,
                )

        job_repo = JobRepository(session)
        # 재생성도 같은 잡 진행 중이면 합류 — 연타가 동시 다운로드(WinError 32)를 만들지 않게
        active = await job_repo.get_active_by_video(request.video_id, lyrics_hash_value)
        if active:
            if request.min_depth:
                # 코덱스 감사 High(2026-08-03): 이 조기 반환이 min_depth 스태시보다 앞이라
                # 깊이 상향 요청이 조용히 무시됐다(재생성 연타 직후 깊이 버튼). 활성 잡에
                # best-effort로 스태시한다 — 잡이 아직 정렬 캡처 전(다운로드/분리 중)이면
                # 그대로 적용되고, 이미 지났으면 잡 터미널의 _pop_stashes가 걷어간다.
                # 적용 보장은 못 하지만(그 한계는 결과의 depth 배지가 정직하게 보여준다)
                # "말없이 버림"보다는 낫다.
                from everyric2.server.worker import stash_force as _stash_force
                from everyric2.server.worker import stash_min_depth as _stash_min_depth

                _stash_force(active.id)
                _stash_min_depth(active.id, request.min_depth)
            return GenerateResponse(
                job_id=active.id,
                status="processing",
                estimated_time=15,
                line_meta_wait_sec=wait_sec,
            )
        # ── 여기서부터 실제로 GPU를 태운다 — 한도는 이 지점에서 한 번만 센다 ──
        # 어느 예산을 쓰는지는 세 갈래 판정이 정한다(_is_upgrade_request 독스트링).
        # 한 요청은 반드시 **하나의** 예산만 소비한다.
        if _is_upgrade_request(request, latest_syncs):
            # 정렬 업그레이드는 generate·파괴적 예산과 별개(운영자 결정 2026-08-04·08-10)
            await _check_upgrade_limit(
                session, request.video_id, x_lyric_user, x_api_key, x_lyric_limits
            )
        elif request.force:
            # 강제 재생성은 GPU 수십 초를 태우는 파괴적 행위 — 초기화와 합산 예산을 쓴다
            await _check_destructive_limit(
                session, "regenerate", request.video_id, x_lyric_user, x_api_key, x_lyric_limits
            )
        else:
            # 비force·비업그레이드 재생성은 GPU 소비가 /generate와 같으므로 같은 상한을 쓴다
            await _check_action_limit(
                session,
                "generate",
                request.video_id,
                x_lyric_user,
                x_api_key,
                x_lyric_limits,
            )
        job = await job_repo.create(
            video_id=request.video_id,
            lyrics=request.lyrics,
            language=request.language,
            target_lang=request.target_lang,
        )
        job_id = job.id

    from everyric2.server.worker import (
        stash_attribution,
        stash_force,
        stash_line_meta,
        stash_min_depth,
        stash_title,
    )

    if request.force:
        # 워커의 (audio_hash, lyrics_hash) 재사용 검사까지 건너뛰어야 진짜 재생성이 된다
        stash_force(job_id)
    if request.min_depth:
        # 깊이 하한 요청은 같은 (audio_hash, lyrics_hash)의 캐시 재사용에 막히면 아무
        # 일도 안 일어난다 — force와 같은 우회가 함께 필요하다. force가 이미 켜져
        # 있으면(위) 중복 무해.
        stash_force(job_id)
        stash_min_depth(job_id, request.min_depth)
    if request.line_meta:
        stash_line_meta(
            job_id, [m.model_dump() for m in request.line_meta], request.line_meta_lang
        )
    if request.attribution:
        stash_attribution(job_id, request.attribution.model_dump())
    stash_title(job_id, request.title, request.artist)
    await _dispatch_job(job_id, background_tasks, await_line_meta=await_line_meta)

    return GenerateResponse(
        job_id=job_id,
        status="processing",
        estimated_time=15,
        line_meta_wait_sec=wait_sec,
    )
