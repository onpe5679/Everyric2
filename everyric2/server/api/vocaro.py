"""보카로 가사 위키 원제 매칭 API.

유튜브 영상 제목이 일본어 원제로 되어 있어 클라이언트의 한국어 독음 인덱스로는
찾지 못하는 경우, 서버 측 원제/한국어 겸용 인덱스(vocaro_index)로 슬러그를 답한다.

`EVERYRIC_SERVER_SONG_INDEX_URL`이 설정되면 로컬 인덱스 대신 외부 곡 인덱스(songindex/1)로
프록시한다 — 확장이 보는 응답 형태(VocaroMatchResponse)는 어느 경로든 동일하다. 이관 검증이
끝나면 별도 커밋으로 로컬 크롤러를 제거할 예정이라 이번 작업에선 지우지 않는다.
"""

import asyncio
import logging
import re

from fastapi import APIRouter, BackgroundTasks, Query
from pydantic import BaseModel

from everyric2.config.settings import get_settings
from everyric2.server.vocaro_index import BASE_URL, build_index, index_status, is_building, match
from everyric2.sources import vocaro as vocaro_source

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
    # 업스트림 모드: 외부 곡 인덱스를 1차로 묻되, **미발견·오류는 확정이 아니다** — 로컬
    # 크롤 인덱스(6,550곡)로 폴백한다. 실측(2026-07-29, iTKM_3mdBsM «ホロウ»): 업스트림
    # 인덱스는 로컬보다 훨씬 작아, 미발견을 그대로 돌려주면 로컬이 찾을 수 있는 곡까지
    # 못 찾은 것으로 확정돼 확장이 자막 폴백(오검출 ASR·번역 자막)으로 밀려난다.
    if _song_index_url():
        try:
            data = await asyncio.to_thread(_upstream_get, "/match", {"title": title})
        except Exception as e:
            logger.info("외부 곡 인덱스 매칭 실패 — 로컬 인덱스로 폴백: %s", e)
            data = None
        if data and data.get("found"):
            return VocaroMatchResponse(
                found=True,
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


# ── 위키 페이지 프록시 (확장 1.5.5+) ─────────────────────────────
#
# 확장이 vocaro.wikidot.com host 권한을 제거하면서(스토어 심사 부담 축소 — 사용자 결정)
# 위키 조회가 이 두 경로로 옮겨 왔다. 위키는 CORS 헤더를 보내지 않아 확장이 권한 없이
# 직접 읽을 수 없다(실측 2026-07-28). 조회는 예의 있는 공유 조회기(호출 간격·백오프,
# sources.base.WikiFetcher)를 스레드에서 태운다. 실패·비가사 페이지는 4xx가 아니라
# found=false다 — 확장은 조용히 다음 가사 소스로 넘어간다.

#: 곡 슬러그 — 번역자가 손으로 짓는 소문자·숫자·하이픈. 인덱스 href 규칙('#'·':' 제외)과
#: 확장 guessSlug가 만드는 값의 합집합만 허용한다.
_PAGE_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,199}$")
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
    slug: str = Query(..., min_length=2, max_length=200),
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
