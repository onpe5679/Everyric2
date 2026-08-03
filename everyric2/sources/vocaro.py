"""보카로 가사 위키(vocaro.wikidot.com) 곡 페이지 파서 — 서버측 포트.

확장 ``everyric2-chrome/src/lib/vocaro.ts``의 ``parseSongPage``/``fetchSongPage``를 그대로
옮긴 것이다. 규칙을 두 곳에 적는 대신 **동작을 같게** 유지한다 — 확장이 붙인 발음/번역과
서버가 대량으로 붙인 발음/번역이 같은 줄 나눔이어야 line_meta 텍스트 매칭이 맞는다.

- 사이트가 HTTPS를 지원하지 않아 평문 http로 접근한다(https는 301로 되돌린다).
- 가사 표는 공식 가이드상 **원문/발음/번역 3행 1세트**다. 2행 1세트(발음 없이 번역만)인
  페이지도 있어 행 수로 판별한다.
- 라이선스: 위키 편집 콘텐츠는 CC BY 4.0(출처 표기 필요), 인용된 원문 가사의 저작권은
  원저작자에게 있다 — 저장·표시 경로에서 출처 링크를 항상 함께 싣는다.

슬러그 → 곡 매칭은 여기서 하지 않는다. 원제(일본어) 매칭은 서버가 이미 크롤해 둔
:mod:`everyric2.server.vocaro_index`(``models/vocaro_index.json``)가 유일한 안정 경로다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from everyric2.sources.base import SourceLine, WikiFetcher, cell_text

BASE_URL = "http://vocaro.wikidot.com"
LICENSE = "CC BY 4.0"
#: 확장이 출처 배지에 쓰는 이름과 같게 맞춘다 (everyric2-chrome/src/content.ts)
ATTRIBUTION_NAME = "보카로 가사 위키"
SOURCE_ID = "vocaro"


@dataclass(frozen=True)
class VocaroSong:
    slug: str
    page_url: str
    page_title: str
    lines: list[SourceLine]

    def attribution(self) -> dict[str, str]:
        """서버 ``attribution`` 계약(name/url/license/source_id)."""
        return {
            "name": ATTRIBUTION_NAME,
            "url": self.page_url,
            "license": LICENSE,
            "source_id": SOURCE_ID,
        }

    def original_text(self) -> str:
        """원문(일본어) 가사 본문 — 빈 줄은 빼고 줄바꿈으로 잇는다."""
        return "\n".join(ln.text for ln in self.lines if ln.text)

    def line_meta(self) -> list[dict[str, str | None]]:
        """발음이나 번역이 실제로 있는 줄만 골라 ``line_meta``로."""
        return [ln.as_dict() for ln in self.lines if ln.pronunciation or ln.translation]


def page_url(slug: str) -> str:
    return f"{BASE_URL}/{slug}"


# ── 파싱 (vocaro.ts parseSongPage와 동일 규칙) ─────────────────────

_TABLE_RE = re.compile(r'<table class="wiki-content-table">([\s\S]*?)</table>')
_ROW_RE = re.compile(r"<tr[^>]*>([\s\S]*?)</tr>")
_TITLE_CELL_RE = re.compile(r'<th[^>]*class="[^"]*title-cell[^"]*"[^>]*>([\s\S]*?)</th>')
_HEADING_RE = re.compile(r"<h([1-6])[^>]*>([\s\S]*?)</h\1>")


def _normalize_variant(s: str) -> str:
    """버전 헤딩/영상 제목 비교용 정규화 — 확장 normalizeTitle과 같은 정신(NFKC·casefold·문자숫자만)."""
    import unicodedata

    return "".join(ch for ch in unicodedata.normalize("NFKC", s).casefold() if ch.isalnum())


#: 흔한 버전 표기의 언어 간 동의어 묶음 — 정규화 후 값(공백·괄호 제거)으로 적는다.
#: 실측(2026-08-04, qXkkhP0d_iM «秋の未確認生物(long ver) / 音街ウナ» →
#: /cryptid-of-autumn): 위키 헤딩은 한국어("긴 버전"/"짧은 버전")인데 유튜브 제목의
#: 버전 표기는 영어("long ver")뿐이라 순수 부분열 포함(label in hint)이 절대 못
#: 만난다 — 문자 자체가 다른 언어라 "포함"이라는 개념이 성립하지 않는다. 헤딩이
#: 밴드명·리믹서명처럼 고유명사면(예: "Best Friend Remix") 원래 로직대로 원문
#: 그대로 힌트에 실리므로 이 동의어 묶음은 흔한 버전 낱말에만 좁게 잡는다 — 넓게
#: 잡으면(예: "long" 단독) 우연 포함(곡명에 "Longing" 등)의 오탐 위험이 커진다.
_VARIANT_SYNONYM_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"longver", "긴버전", "롱버전"}),
    frozenset({"shortver", "짧은버전", "숏버전", "쇼트버전"}),
    frozenset({"fullver", "풀버전"}),
    frozenset({"tvsize", "tvver", "tv사이즈", "tv버전", "티비사이즈", "티비버전"}),
    frozenset({"inst", "instrumental", "인스트", "반주"}),
    frozenset({"originalver", "오리지널", "원곡", "본편"}),
)


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _hint_tokens(variant_hint: str) -> frozenset[str]:
    """힌트(영상 제목 원문)를 비영숫자 경계로 쪼갠 토큰 집합 — 개별 토큰 + 인접 토큰
    연접(2개까지)을 후보로 낸다.

    실측(2026-08-04 A4 감사): 힌트 전체를 부분열로 뭉쳐 검사하던 예전 방식은 "against"·
    "instant"·"institute"(전부 "inst"를 부분열로 포함)가 "인스트" 그룹과, "Long Verse"
    ("longver"를 부분열로 포함)가 "긴버전" 그룹과 우연히 충돌했다(실제 함수 호출로
    재현). 토큰 경계로 쪼개 **정확 일치**만 인정하면 이 우연 포함이 사라진다 —
    "against"는 통짜 토큰이라 "inst"와 절대 같아질 수 없다.

    다만 실측 성공 사례(cryptid-of-autumn, 힌트 "(long ver)")는 "long"+"ver"가 공백으로
    갈린 **두 토큰**이 합쳐 하나의 버전 표기를 이룬다 — 그래서 인접한 두 토큰의 연접도
    후보에 넣는다(3단어 이상 연접은 조합 폭발 위험 대비 다루지 않는다 — 실측 사례가
    2단어 연접뿐이다).
    """
    raw_tokens = _TOKEN_RE.findall(variant_hint)
    norm_tokens = [_normalize_variant(t) for t in raw_tokens]
    tokens = {t for t in norm_tokens if t}
    for i in range(len(norm_tokens) - 1):
        combo = norm_tokens[i] + norm_tokens[i + 1]
        if combo:
            tokens.add(combo)
    return frozenset(tokens)


def _synonym_hint_match(label: str, hint_tokens: frozenset[str]) -> bool:
    """라벨이 흔한 버전 낱말이면, 힌트 쪽 토큰에 실린 다른 언어 동의어로도 대응을 확인한다.

    label 자체가 어느 동의어 묶음의 원소일 때만 동작 — 밴드명 등 그 외 라벨은
    기존 부분열 포함 검사만 탄다(이 함수는 항상 False). hint_tokens는 :func:`_hint_tokens`
    가 낸 **토큰 집합**이라 부분열이 아니라 정확 일치로 대조한다.
    """
    for group in _VARIANT_SYNONYM_GROUPS:
        if label in group:
            return any(token != label and token in hint_tokens for token in group)
    return False


def _pick_table(page_html: str, variant_hint: str | None) -> str:
    """가사 표가 여러 개인 페이지에서 힌트(영상 제목)에 맞는 버전의 표를 고른다.

    한 페이지에 원곡과 리믹스 가사가 헤딩("오리지널"/"Best Friend Remix",
    "긴 버전"/"짧은 버전" 등)으로 나뉘어 실리는 경우가 있다(실측: /monitoring,
    /cryptid-of-autumn — 헤딩 레벨은 h1·h2 등 제각각이라 _HEADING_RE는 h1~h6을
    다 본다). 예전 코드는 무조건 첫 표를 집어 롱버전 영상에 짧은 버전 가사를
    붙였다. 각 표의 **직전 헤딩**을 정규화해 힌트에 통째로 포함되거나(고유명사
    헤딩) 언어 간 버전 동의어가 대응되면(_synonym_hint_match — "긴 버전" ↔
    "long ver") 고르고(여럿이면 가장 긴 헤딩), 없으면 첫 표 — 표가 하나인
    대다수 페이지와 힌트 없는 구버전 호출은 동작이 그대로다.

    호출 전제: page_html에 표가 최소 1개 있다 (parse_song_page가 먼저 검사).
    """
    tables = list(_TABLE_RE.finditer(page_html))
    if len(tables) == 1 or not variant_hint:
        return tables[0].group(1)
    hint = _normalize_variant(variant_hint)
    if not hint:
        return tables[0].group(1)
    hint_tokens = _hint_tokens(variant_hint)
    best: tuple[int, str] | None = None
    prev_end = 0
    for m in tables:
        headings = _HEADING_RE.findall(page_html[prev_end : m.start()])
        prev_end = m.end()
        if not headings:
            continue  # 직전 표와 사이에 헤딩이 없으면 같은 버전의 연속 표 — 후보 아님
        label = _normalize_variant(re.sub(r"<[^>]+>", "", headings[-1][1]))
        # 2자 미만 라벨은 우연 포함이 너무 쉽다 (예: "2")
        if len(label) < 2:
            continue
        matched = label in hint or _synonym_hint_match(label, hint_tokens)
        if matched and (best is None or len(label) > best[0]):
            best = (len(label), m.group(1))
    return best[1] if best else tables[0].group(1)


def parse_song_page(
    page_html: str, variant_hint: str | None = None
) -> tuple[str, list[SourceLine]] | None:
    """곡 페이지 HTML → (원제, 가사 줄 목록). 가사 표가 없으면 None.

    행 수로 세트 크기를 판별한다 — 3의 배수면 원문/발음/번역, 아니고 2의 배수면
    원문/번역, 둘 다 아니면 전부 원문으로 본다(어긋난 표를 조용히 버리지 않는다).

    6행처럼 3과 2의 배수를 겸하는 표는 3행 세트로 읽는다 — vocaro.ts와 같은 판정
    순서다. 발음 행을 번역으로 오인하는 쪽보다 발음이 있는데 못 읽는 쪽이 드물다.

    ``variant_hint``(영상 제목)가 있으면 버전 헤딩이 힌트와 맞는 표를 고른다 —
    :func:`_pick_table` 참고.
    """
    if not _TABLE_RE.search(page_html):
        return None

    rows = [
        cell_text(m.group(1), drop_ruby=True)
        for m in _ROW_RE.finditer(_pick_table(page_html, variant_hint))
    ]

    lines: list[SourceLine]
    if rows and len(rows) % 3 == 0:
        lines = [
            SourceLine(
                text=rows[i],
                pronunciation=rows[i + 1] or None,
                translation=rows[i + 2] or None,
            )
            for i in range(0, len(rows), 3)
        ]
    elif rows and len(rows) % 2 == 0:
        lines = [
            SourceLine(text=rows[i], translation=rows[i + 1] or None)
            for i in range(0, len(rows), 2)
        ]
    else:
        lines = [SourceLine(text=r) for r in rows]

    lines = [ln for ln in lines if ln.text]
    title_match = _TITLE_CELL_RE.search(page_html)
    title = cell_text(title_match.group(1), drop_ruby=True) if title_match else ""
    return title, lines


# ── 조회 ──────────────────────────────────────────────────────────

_default_fetcher: WikiFetcher | None = None


def _fetcher() -> WikiFetcher:
    global _default_fetcher
    if _default_fetcher is None:
        _default_fetcher = WikiFetcher()
    return _default_fetcher


def fetch_song(
    slug: str, fetcher: WikiFetcher | None = None, variant_hint: str | None = None
) -> VocaroSong | None:
    """슬러그로 곡 페이지를 받아 파싱. 요청 실패·가사 표 없음이면 None (요청 1회).

    ``variant_hint``(영상 제목)는 한 페이지에 여러 버전 가사가 실린 경우의 표 선택용.
    """
    url = page_url(slug)
    page_html = (fetcher or _fetcher()).get_text(url)
    if not page_html:
        return None
    parsed = parse_song_page(page_html, variant_hint)
    if parsed is None:
        return None
    title, lines = parsed
    if not lines:
        return None
    return VocaroSong(slug=slug, page_url=url, page_title=title or slug, lines=lines)


# ── 수록곡 일람 (allsongs-*) ──────────────────────────────────────

# 확장 vocaro.ts parseIndexEntries와 같은 규칙: <li><a href="/slug">제목</a></li> 쌍만.
# href의 '#'(페이지 내 앵커)·':'(위키 시스템 페이지)은 곡 페이지가 아니므로 제외한다.
_INDEX_ENTRY_RE = re.compile(r'<li>\s*<a\s+href="/([^"#:]+)"[^>]*>([^<]+)</a>\s*</li>')


def parse_index_entries(page_html: str) -> list[tuple[str, str]]:
    """수록곡 일람 페이지 HTML → (slug, 제목) 쌍 목록.

    제목의 HTML 엔티티(&amp; 등)는 풀어 준다 — 사용자 검색어(실제 '&')와 대조되는
    값이므로 원문 엔티티를 남기면 그 제목은 영영 매칭되지 않는다.
    """
    import html as _html

    return [
        (m.group(1), _html.unescape(m.group(2)).strip())
        for m in _INDEX_ENTRY_RE.finditer(page_html)
        if m.group(2).strip()
        # 다른 일람·시스템·가이드 페이지로의 내비게이션 링크는 곡이 아니다 (vocaro.ts와 동일)
        and not m.group(1).startswith(("allsongs", "system", "guide"))
    ]


def fetch_index_entries(page: str, fetcher: WikiFetcher | None = None) -> list[tuple[str, str]] | None:
    """일람 페이지를 받아 파싱. 요청 실패는 None — 빈 목록(항목 없음)과 구분한다."""
    page_html = (fetcher or _fetcher()).get_text(f"{BASE_URL}/{page}")
    if page_html is None:
        return None
    return parse_index_entries(page_html)


def fetch_song_page(slug: str, fetcher: WikiFetcher | None = None) -> list[SourceLine]:
    """가사 줄만 필요할 때. 못 받으면 빈 목록 — 출처 표기가 필요하면 :func:`fetch_song`."""
    song = fetch_song(slug, fetcher)
    return song.lines if song else []
