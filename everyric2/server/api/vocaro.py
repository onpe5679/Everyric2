"""보카로 가사 위키 원제 매칭 API.

유튜브 영상 제목이 일본어 원제로 되어 있어 클라이언트의 한국어 독음 인덱스로는
찾지 못하는 경우, 서버 측 원제/한국어 겸용 인덱스(vocaro_index)로 슬러그를 답한다.

`EVERYRIC_SERVER_SONG_INDEX_URL`이 설정되면 로컬 인덱스 대신 외부 곡 인덱스(songindex/1)로
프록시한다 — 확장이 보는 응답 형태(VocaroMatchResponse)는 어느 경로든 동일하다. 이관 검증이
끝나면 별도 커밋으로 로컬 크롤러를 제거할 예정이라 이번 작업에선 지우지 않는다.
"""

import asyncio
import logging
import math
import re
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Query
from pydantic import BaseModel, Field

from everyric2.config.settings import get_settings
from everyric2.server import title_match
from everyric2.server.services.video_identity import (
    VideoIdentity,
    fetch_video_identity,
    ordered_title_candidates,
)
from everyric2.server.vocaro_index import (
    BASE_URL,
    MatchDecision,
    SongEntry,
    build_index,
    corroborated_raw_candidates,
    index_status,
    is_building,
    match_with_evidence,
    search_matches,
    strong_candidates,
)
from everyric2.sources import vocaro as vocaro_source

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/vocaro", tags=["vocaro"])

# 업스트림 곡 인덱스 요청 타임아웃(초) — 확장 매칭은 대화형이라 짧게 잡고 실패 시 미발견 폴백
_UPSTREAM_TIMEOUT_SEC = 3.0
MATCHER_VERSION = "identity-1"
_VIDEO_ID_LOOKUP_SEMAPHORE = asyncio.Semaphore(4)
_VIDEO_ID_LOOKUP_TIMEOUT_SEC = 2.5


class VocaroMatchCandidate(BaseModel):
    slug: str
    page_url: str
    ko: str | None = None
    ja: str | None = None
    producer: str | None = None
    score: float


class VocaroMatchResponse(BaseModel):
    found: bool
    slug: str | None = None
    page_url: str | None = None
    ko: str | None = None
    ja: str | None = None
    status: str | None = None
    reason: str | None = None
    matcher_version: str = MATCHER_VERSION
    upstream_matcher_version: str | None = None
    candidate_count: int = 0
    matched_query: str | None = None
    evidence_source: str | None = None
    resolved_title: str | None = None
    resolved_channel: str | None = None
    candidates: list[VocaroMatchCandidate] = Field(default_factory=list)


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


def _title_roots(title: str, candidates: list[str] | None) -> list[str]:
    roots: list[str] = []
    for value in [*(candidates or []), title]:
        value = value.strip()
        if value and value not in roots:
            roots.append(value)
    return roots[:4]


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _positive_int(value: object, default: int = 1) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def _entry_producer(entry: SongEntry) -> str | None:
    for value in (entry.ko, entry.ja):
        if value and "/" in value:
            producer = value.rsplit("/", 1)[1].strip()
            if producer:
                return producer
    return None


def _upstream_candidates(data: dict) -> list[VocaroMatchCandidate]:
    raw_candidates = data.get("candidates")
    if not isinstance(raw_candidates, list):
        return []
    candidates: list[VocaroMatchCandidate] = []
    for raw in raw_candidates[:5]:
        if not isinstance(raw, dict):
            continue
        slug = _optional_string(raw.get("slug"))
        if not slug or not slug.strip():
            continue
        try:
            score = float(raw.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        if not math.isfinite(score):
            score = 0.0
        ko = _optional_string(raw.get("ko"))
        ja = _optional_string(raw.get("ja"))
        producer = _optional_string(raw.get("producer")) or _entry_producer(
            SongEntry(slug=slug, ko=ko or "", ja=ja)
        )
        candidates.append(
            VocaroMatchCandidate(
                slug=slug,
                page_url=_optional_string(raw.get("page_url")) or f"{BASE_URL}/{slug}",
                ko=ko,
                ja=ja,
                producer=producer,
                score=score,
            )
        )
    return candidates


def _upstream_result_matches_title(data: dict, roots: list[str]) -> bool:
    """구형 업스트림의 ``found=true``를 기호 보존 정확 일치로 다시 검증한다.

    외부 songindex가 아직 느슨한 매처여도 ``S.C.R.E.A.M → SCREAM`` 같은 결과가 이 서버를
    통과하지 않게 하는 호환 경계다. ja/ko가 없으면 슬러그의 하이픈을 공백으로 본 별칭까지
    검증하되, 포함 일치는 허용하지 않는다.
    """
    fields = [data.get("ja"), data.get("ko")]
    slug = data.get("slug")
    if isinstance(slug, str):
        fields.append(slug.replace("-", " "))
    field_keys = {
        title_match.identity_key(field)
        for field in fields
        if isinstance(field, str) and title_match.identity_key(field)
    }
    # 후보 순서는 채널 근거가 확정한 우선순위다. 구형 업스트림이 원래 title만 보고 2순위
    # 결과를 돌려도 여기서 받아들이지 않고, 로컬 identity 매처가 1순위부터 다시 판정한다.
    preferred_roots = roots[:1]
    return any(
        title_match.identity_key(candidate) in field_keys
        for root in preferred_roots
        for candidate in strong_candidates(root)
    )


def _upstream_result_matches_video_identity(
    data: dict,
    resolved_title: str | None,
    resolved_channel: str | None,
) -> bool:
    if not resolved_title:
        return False
    fields = [data.get("ja"), data.get("ko")]
    slug = data.get("slug")
    if isinstance(slug, str):
        fields.append(slug.replace("-", " "))
    field_keys = {
        title_match.identity_key(field)
        for field in fields
        if isinstance(field, str) and title_match.identity_key(field)
    }
    candidates = corroborated_raw_candidates(
        resolved_title,
        artist=resolved_channel,
        channel=resolved_channel,
    )
    return any(title_match.identity_key(candidate) in field_keys for candidate in candidates)


def _upstream_consistent_with_local(data: dict, decision: MatchDecision) -> bool:
    slug = data.get("slug")
    if not isinstance(slug, str):
        return False
    upstream_matcher = _optional_string(data.get("matcher_version")) or ""
    if decision.status in {"ambiguous", "not_found", "index_empty"}:
        # 로컬이 교차검증할 항목을 못 댔는데 구 songindex의 제목 하나만 보고 고른 결과를
        # identity-1로 포장하면 신클라이언트의 안전 경계를 우회한다. 증거 계약을 구현한
        # 새 업스트림만 로컬보다 새 곡이거나 더 풍부한 producer/channel 판정임을 주장한다.
        return upstream_matcher.startswith("identity-")
    if decision.entry is not None:
        return decision.entry.slug == slug
    return True


def _decision_matches_strong_raw(
    decision: MatchDecision,
    raw_title: str | None,
    *,
    artist: str | None,
    channel: str | None,
) -> bool:
    """현재 hit가 raw 제목 전체의 비파괴 변형으로도 확인되는가."""
    if decision.entry is None or not raw_title:
        return False
    entry_keys = {
        title_match.identity_key(field)
        for field in (
            decision.entry.ja,
            decision.entry.ko,
            decision.entry.slug.replace("-", " "),
        )
        if field and title_match.identity_key(field)
    }
    return any(
        title_match.identity_key(candidate) in entry_keys
        for candidate in corroborated_raw_candidates(
            raw_title,
            artist=artist,
            channel=channel,
        )
    )


async def _bounded_video_identity(video_id: str) -> VideoIdentity | None:
    async def fetch() -> VideoIdentity | None:
        async with _VIDEO_ID_LOOKUP_SEMAPHORE:
            return await asyncio.to_thread(fetch_video_identity, video_id)

    try:
        # semaphore 대기까지 전체 예산에 포함한다. 폭주 시 큐에서 6s 클라이언트 타임아웃을
        # 넘기는 대신 2.5s 안에 안전 미스로 닫는다.
        return await asyncio.wait_for(fetch(), timeout=_VIDEO_ID_LOOKUP_TIMEOUT_SEC)
    except TimeoutError:
        return None


async def _decision_after_video_check(
    decision: MatchDecision,
    identity_task: asyncio.Task[VideoIdentity | None] | None,
) -> tuple[MatchDecision, str, str | None, str | None]:
    if identity_task is None:
        return decision, "client", None, None
    identity = await identity_task
    if identity is None:
        if decision.entry is not None:
            decision = MatchDecision(
                status="ambiguous",
                reason="video_id_identity_unavailable",
                candidate_count=decision.candidate_count,
                matched_query=decision.matched_query,
            )
        return decision, "client", None, None

    metadata_roots = ordered_title_candidates(identity)
    metadata_decision = await asyncio.to_thread(
        match_with_evidence,
        metadata_roots[0],
        title_candidates=metadata_roots,
        raw_title=identity.title,
        artist=identity.channel,
        channel=identity.channel,
    )
    return metadata_decision, "youtube_oembed", identity.title, identity.channel


def _accepted_upstream_response(
    data: dict,
    *,
    roots: list[str],
    decision: MatchDecision,
    decision_source: str,
    resolved_title: str | None,
    resolved_channel: str | None,
) -> VocaroMatchResponse | None:
    slug = data.get("slug")
    request_matches = _upstream_result_matches_title(data, roots)
    video_matches = bool(
        decision_source == "youtube_oembed"
        and decision.entry is not None
        and isinstance(slug, str)
        and slug == decision.entry.slug
    )
    video_title_matches = _upstream_result_matches_video_identity(
        data,
        resolved_title,
        resolved_channel,
    )
    evidence_matches = (
        video_matches or video_title_matches
        if decision_source == "youtube_oembed"
        else request_matches
    )
    if not (
        data.get("found")
        and data.get("status") in (None, "matched")
        and isinstance(slug, str)
        and slug.strip()
        and evidence_matches
        and _upstream_consistent_with_local(data, decision)
    ):
        return None
    candidates = _upstream_candidates(data)
    return VocaroMatchResponse(
        found=True,
        slug=slug,
        page_url=_optional_string(data.get("page_url")),
        ko=_optional_string(data.get("ko")),
        ja=_optional_string(data.get("ja")),
        status="matched",
        reason=_optional_string(data.get("reason")) or "upstream_exact_title",
        matcher_version=MATCHER_VERSION,
        upstream_matcher_version=_optional_string(data.get("matcher_version")),
        candidate_count=(
            len(candidates) if candidates else _positive_int(data.get("candidate_count"))
        ),
        matched_query=_optional_string(data.get("matched_query")),
        evidence_source=(
            "youtube_oembed" if decision_source == "youtube_oembed" else "song_index"
        ),
        resolved_title=resolved_title,
        resolved_channel=resolved_channel,
        candidates=candidates,
    )


async def _manual_search_response(title: str) -> VocaroMatchResponse | None:
    ranked = await asyncio.to_thread(search_matches, title, 5)
    if not ranked:
        return None
    manual = ranked[0][0]
    return VocaroMatchResponse(
        found=True,
        slug=manual.slug,
        page_url=f"{BASE_URL}/{manual.slug}",
        ko=manual.ko,
        ja=manual.ja,
        status="matched",
        reason="manual_search_fuzzy",
        candidate_count=len(ranked),
        evidence_source="manual_search",
        candidates=[
            VocaroMatchCandidate(
                slug=entry.slug,
                page_url=f"{BASE_URL}/{entry.slug}",
                ko=entry.ko,
                ja=entry.ja,
                producer=_entry_producer(entry),
                score=score,
            )
            for entry, score in ranked
        ],
    )


@router.get("/match", response_model=VocaroMatchResponse)
async def match_title(
    background_tasks: BackgroundTasks,
    title: Annotated[str, Query(min_length=1, max_length=256)],
    candidate: Annotated[list[str] | None, Query(max_length=256)] = None,
    raw_title: Annotated[str | None, Query(max_length=300)] = None,
    artist: Annotated[str | None, Query(max_length=128)] = None,
    channel: Annotated[str | None, Query(max_length=128)] = None,
    video_id: Annotated[
        str | None, Query(pattern=r"^[A-Za-z0-9_-]{11}$")
    ] = None,
    mode: Annotated[str | None, Query(pattern="^search$")] = None,
):
    roots = _title_roots(title, candidate)
    upstream_params: dict[str, object] = {"title": title}
    if candidate:
        upstream_params["candidate"] = candidate[:4]
    for key, value in (
        ("raw_title", raw_title),
        ("artist", artist),
        ("channel", channel),
        ("video_id", video_id),
        ("mode", mode),
    ):
        if value:
            upstream_params[key] = value

    decision = await asyncio.to_thread(
        match_with_evidence,
        title,
        title_candidates=roots,
        raw_title=raw_title,
        artist=artist,
        channel=channel,
    )

    needs_video_check = bool(
        mode is None
        and video_id
        and (
            decision.entry is None
            or not raw_title
            or decision.reason == "raw_evidence_title"
            or (
                raw_title
                and not _decision_matches_strong_raw(
                    decision,
                    raw_title,
                    artist=artist,
                    channel=channel,
                )
            )
        )
    )
    identity_task = (
        asyncio.create_task(_bounded_video_identity(video_id))
        if needs_video_check and video_id
        else None
    )

    # 업스트림과 video ID 복구는 서로 독립 네트워크 I/O라 겹쳐 실행한다. 최악 시간은
    # 3.0s+2.5s의 합이 아니라 둘 중 긴 쪽이 되어 Chrome의 6s 계약 안에 여유가 생긴다.
    upstream_task = (
        asyncio.create_task(asyncio.to_thread(_upstream_get, "/match", upstream_params))
        if _song_index_url()
        else None
    )
    data: dict | None = None
    if upstream_task is not None:
        try:
            upstream_result = await upstream_task
            if isinstance(upstream_result, dict):
                data = upstream_result
            else:
                logger.info("외부 곡 인덱스 응답이 객체가 아님 — 로컬 인덱스로 폴백")
        except Exception as e:
            logger.info("외부 곡 인덱스 매칭 실패 — 로컬 인덱스로 폴백: %s", e)

    decision, decision_source, resolved_title, resolved_channel = (
        await _decision_after_video_check(decision, identity_task)
    )

    if data is not None:
        accepted = _accepted_upstream_response(
            data,
            roots=roots,
            decision=decision,
            decision_source=decision_source,
            resolved_title=resolved_title,
            resolved_channel=resolved_channel,
        )
        if accepted is not None:
            return accepted
        if data.get("found"):
            logger.info("외부 곡 인덱스의 느슨한 제목 매칭을 기각: %s", title)

    if decision.entry is None and mode == "search":
        manual = await _manual_search_response(title)
        if manual is not None:
            return manual
    if decision.entry:
        result = decision.entry
        return VocaroMatchResponse(
            found=True,
            slug=result.slug,
            page_url=f"{BASE_URL}/{result.slug}",
            ko=result.ko,
            ja=result.ja,
            status=decision.status,
            reason=(
                f"video_id_{decision.reason}"
                if decision_source == "youtube_oembed"
                else decision.reason
            ),
            candidate_count=decision.candidate_count,
            matched_query=decision.matched_query,
            evidence_source=decision_source,
            resolved_title=resolved_title,
            resolved_channel=resolved_channel,
        )

    if decision.status == "index_empty":
        # 인덱스가 아직 없으면 매칭 실패와 함께 백그라운드 빌드를 킥한다 (중복 킥은 락으로 방지)
        if not is_building():
            background_tasks.add_task(build_index)
        return VocaroMatchResponse(
            found=False,
            status="index_empty",
            reason=decision.reason,
            evidence_source=decision_source,
            resolved_title=resolved_title,
            resolved_channel=resolved_channel,
        )

    return VocaroMatchResponse(
        found=False,
        status=decision.status,
        reason=decision.reason,
        candidate_count=decision.candidate_count,
        matched_query=decision.matched_query,
        evidence_source=decision_source,
        resolved_title=resolved_title,
        resolved_channel=resolved_channel,
    )


# ── 위키 페이지 프록시 (확장 1.5.5+) ─────────────────────────────
#
# 확장이 vocaro.wikidot.com host 권한을 제거하면서(스토어 심사 부담 축소 — 사용자 결정)
# 위키 조회가 이 두 경로로 옮겨 왔다. 위키는 CORS 헤더를 보내지 않아 확장이 권한 없이
# 직접 읽을 수 없다(실측 2026-07-28). 조회는 예의 있는 공유 조회기(호출 간격·백오프,
# sources.base.WikiFetcher)를 스레드에서 태운다. 실패·비가사 페이지는 4xx가 아니라
# found=false다 — 확장은 조용히 다음 가사 소스로 넘어간다.

#: 곡 슬러그 — 번역자가 손으로 짓는 소문자·숫자·하이픈. 인덱스 href 규칙('#'·':' 제외)과
#: 확장 guessSlug가 만드는 값의 합집합만 허용한다.
_PAGE_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,199}$")
#: 수록곡 일람 페이지명 — allsongs-{a..z} | allsongs-h{1..14}(한글 초성) | num | symbols
_INDEX_PAGE_RE = re.compile(r"^allsongs-(?:[a-z]|h(?:1[0-4]|[1-9])|num|symbols)$")


class VocaroPageLine(BaseModel):
    text: str
    pronunciation: str | None = None
    translation: str | None = None


class VocaroPageResponse(BaseModel):
    found: bool
    slug: str | None = None
    page_url: str | None = None
    page_title: str | None = None
    lines: list[VocaroPageLine] = []


class VocaroIndexEntryModel(BaseModel):
    slug: str
    title: str


class VocaroIndexResponse(BaseModel):
    found: bool
    entries: list[VocaroIndexEntryModel] = []


@router.get("/page", response_model=VocaroPageResponse)
async def song_page(
    slug: str = Query(..., min_length=1, max_length=200),
    hint: str | None = Query(None, max_length=300),
):
    """슬러그로 곡 페이지를 받아 파싱해 준다 — 원문/발음/번역 줄 목록 + 출처.

    ``hint``(영상 제목)는 한 페이지에 여러 버전 가사가 실린 경우(원곡/리믹스 등)
    맞는 버전의 표를 고르는 데 쓴다. 없으면 첫 표 — 구버전 확장과 동작 동일.
    """
    if not _PAGE_SLUG_RE.match(slug):
        return VocaroPageResponse(found=False)
    song = await asyncio.to_thread(vocaro_source.fetch_song, slug, None, hint)
    if song is None:
        return VocaroPageResponse(found=False)
    return VocaroPageResponse(
        found=True,
        slug=song.slug,
        page_url=song.page_url,
        page_title=song.page_title,
        lines=[
            VocaroPageLine(text=ln.text, pronunciation=ln.pronunciation, translation=ln.translation)
            for ln in song.lines
        ],
    )


@router.get("/index", response_model=VocaroIndexResponse)
async def index_listing(page: str = Query(..., min_length=1, max_length=32)):
    """수록곡 일람 페이지의 (slug, 제목) 쌍 — 확장의 한국어 독음 제목 매칭용."""
    if not _INDEX_PAGE_RE.match(page):
        return VocaroIndexResponse(found=False)
    entries = await asyncio.to_thread(vocaro_source.fetch_index_entries, page)
    if entries is None:
        return VocaroIndexResponse(found=False)
    return VocaroIndexResponse(
        found=True,
        entries=[VocaroIndexEntryModel(slug=slug, title=title) for slug, title in entries],
    )


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
