"""VocaloidLyrics Wiki(vocaloidlyrics.miraheze.org) 검색·가사 표 파서 — 서버측 포트.

확장 ``everyric2-chrome/src/lib/miraheze.ts``를 그대로 옮긴 것이다. 발음은 로마자,
번역은 영어다 — vocaro(한글 독음 + 한국어 번역)의 대칭축이라 두 위키를 합치면 한 곡에
ko·en·ja 세 언어가 모인다.

확장이 실곡 4곡 HTML을 대조해 확인한 규칙을 그대로 유지한다:

- 가사 표는 ``<table class="lyrics-table" id="lyrics-N">``. **한 페이지에 언어가 다른 표가
  여러 개 있을 수 있다**(실측: 만다린 커버 표) — 헤더에 ``<th class="lyrics-jp">``가 있는
  표만 고른다.
- 헤더 행의 ``<th>`` 개수(2 또는 3)가 실제 열 수다. 2열이면 번역이 없다(로마자만).
- 연 사이 빈 줄은 ``<tr><td><br /></td></tr>`` — 칸 1개 + 내용 없음 → 건너뛴다.
- 영어만 있는 삽입구는 ``<td colspan="N" class="merged">text</td>`` — 칸 1개 + 내용 있음 →
  원문/로마자 구분이 없으므로 text 하나로만 싣는다.
- ``action=parse&page=<제목>``은 제목에 공백+괄호가 섞이면 missingtitle로 실패한 실측이
  있다 — search가 이미 pageid를 주므로 ``action=parse&pageid=<id>``로 우회한다.

검색어는 유튜브 원제를 그대로 넣지 않는다(:func:`title_candidates` 주석 참고) — 부가
정보가 전문검색의 관련도를 흐려 진짜 곡 페이지가 상위에서 밀려난 실사용 사고가 있다.

라이선스: 위키 편집 콘텐츠는 CC BY-SA 4.0(출처 표기 필요).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from urllib.parse import quote, urlencode

from everyric2.sources.base import ASCII_LB, SourceLine, WikiFetcher, cell_text

BASE_URL = "https://vocaloidlyrics.miraheze.org"
API_URL = f"{BASE_URL}/w/api.php"
LICENSE = "CC BY-SA 4.0"
SOURCE_ID = "miraheze"

# MediaWiki 전문검색(list=search)은 제목이 아니라 본문 관련도로 정렬한다 — 짧고 흔한
# 제목일수록 진짜 곡 페이지가 밀려난다(실측: 곡 페이지가 2위·5위, 1위는 수록 앨범·
# 프로듀서 페이지). 그래서 순위와 무관하게 검색 후보의 제목이 요청 후보와 검증된
# 정규화 접두 관계인 페이지만 조회한다. 일치가 없으면 검색 1위로 물러나지 않는다.
SEARCH_LIMIT = 10


@dataclass(frozen=True)
class MirahezeSong:
    page_title: str
    url: str
    lines: list[SourceLine]
    has_translation: bool

    def attribution(self) -> dict[str, str]:
        """서버 ``attribution`` 계약. 이름은 확장의 출처 배지 문구와 같게 맞춘다."""
        return {
            "name": f"{self.page_title} — VocaloidLyrics Wiki",
            "url": self.url,
            "license": LICENSE,
            "source_id": SOURCE_ID,
        }

    def original_text(self) -> str:
        return "\n".join(ln.text for ln in self.lines if ln.text)

    def translation_lines(self) -> list[dict[str, str]]:
        """번역이 실제로 있는 줄만 ``{text, translation}``으로 — 번역 레이어 본문 계약."""
        return [
            {"text": ln.text, "translation": ln.translation}
            for ln in self.lines
            if ln.text and ln.translation
        ]


# ── 검색어 후보 생성 (miraheze.ts titleCandidates 포트) ────────────

# 대표 구분자 — 유튜브 보카로 관례상 곡명 다음에 아티스트·가수 표기가 붙는 자리.
# feat./ft.는 대소문자·마침표 유무를 가리지 않는다. ASCII 낱말 경계(ASCII_LB)를 쓰는
# 이유는 base.py 머리말 참고 — 파이썬 \b는 «곡名feat.가수»에서 매칭에 실패한다.
_TITLE_SEPARATOR_RE = re.compile(
    rf"/|｜|\||\s-\s|〜|{ASCII_LB}feat\.?|{ASCII_LB}ft\.?", re.IGNORECASE
)

# 【】·[]·()·（）로 감싼 장식(MV·Official Music Video 등) 한 덩어리.
_DECORATION_RE = re.compile(r"[【\[（(][^】\]）)]*[】\]）)]")


def strip_before_separator(raw: str) -> str | None:
    """구분자 **앞** 조각(=곡명). 구분자가 없거나 맨 앞에 있으면 None."""
    m = _TITLE_SEPARATOR_RE.search(raw)
    if not m or m.start() == 0:
        return None
    return raw[: m.start()].strip() or None


def strip_decorations(raw: str) -> str | None:
    """장식 구간을 제거한 판. 장식이 없어 원문과 같으면 None."""
    cleaned = re.sub(r"\s+", " ", _DECORATION_RE.sub(" ", raw)).strip()
    return cleaned if cleaned and cleaned != raw else None


def title_candidates(raw: str) -> list[str]:
    """검색 전 제목 후보를 순서대로: ① 구분자 앞 조각, ② 장식 제거판, ③ 원문 그대로.

    실사용 사고(2026-07): 유튜브 원제를 그대로 검색하면 부가 정보(아티스트·feat.·장식)가
    전문검색의 관련도를 흐려 진짜 곡 페이지가 상위 10위 밖으로 밀리고, 접두 일치도 실패해
    프로듀서 페이지를 그대로 채택했다. 관례상 곡명은 구분자 앞에 오므로 그 조각을 먼저 쓴다.
    """
    trimmed = raw.strip()
    out: list[str] = []
    for value in (strip_before_separator(trimmed), strip_decorations(trimmed), trimmed):
        if value and value not in out:
            out.append(value)
    return out[:3]


def _normalize_match_title(value: str) -> str:
    """MediaWiki 제목 비교용 정규화 — NFKC + 소문자 + 공백 접기.

    문자를 전부 제거하지 않는다. 접두 뒤의 문자가 괄호·슬래시 같은
    Miraheze 곡 페이지 접미인지 검증해 ``Wonder``가 ``Wonderland``를 통과하지
    않게 하기 위해서다.
    """
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value).lower()).strip()


def page_title_matches_candidate(candidate: str, page_title: str) -> bool:
    """검색 후보와 페이지 제목이 자동 채택해도 되는 접두 관계인지.

    정확 일치 또는 ``ロキ (Roki)``·``フラジール/nulut``처럼 곡명 뒤에
    로마자·프로듀서 식별자가 붙은 모양만 통과한다. 제목을 정규화한 뒤
    다음 문자가 일반 문자인 부분열(``Wonderland`` 등)은 거절한다.
    """
    candidate_norm = _normalize_match_title(candidate)
    page_norm = _normalize_match_title(page_title)
    if not candidate_norm or not page_norm:
        return False
    if page_norm == candidate_norm:
        return True
    if not page_norm.startswith(candidate_norm):
        return False
    return page_norm[len(candidate_norm)] in " (/"


def producer_from_page_title(page_title: str) -> str | None:
    """「フラジール (Fragile)/nulut」처럼 제목 뒤에 붙은 프로듀서 이름. 없으면 None.

    이 위키의 동명이곡 구분 관례다. 대량 인제스트가 영상 후보를 검증할 때 «채널명이
    프로듀서와 같은가»를 볼 수 있는 유일한 단서라 여기서 꺼내 준다.
    """
    if "/" not in page_title:
        return None
    return page_title.rsplit("/", 1)[1].strip() or None


# ── 가사 표 파싱 ──────────────────────────────────────────────────

_LYRICS_TABLE_RE = re.compile(r'<table class="lyrics-table"[^>]*>([\s\S]*?)</table>')
_HEADER_ROW_RE = re.compile(r'<tr class="lyrics-table-header">([\s\S]*?)</tr>')
_ROW_RE = re.compile(r"<tr(?:\s[^>]*)?>([\s\S]*?)</tr>")
_CELL_RE = re.compile(r"<td[^>]*>([\s\S]*?)</td>")
_TH_RE = re.compile(r"<th")


def _find_japanese_lyrics_table(page_html: str) -> str | None:
    """lyrics-table들 중 헤더에 lyrics-jp가 있는 표(원문이 일본어인 표)."""
    for m in _LYRICS_TABLE_RE.finditer(page_html):
        if 'class="lyrics-jp"' in m.group(1):
            return m.group(1)
    return None


def _row_to_lines(cells: list[str]) -> list[SourceLine]:
    """행 하나(칸 목록)를 줄 0~1개로 — 빈 연 구분·삽입구·정상 행을 여기서 가른다."""
    if not cells:
        return []
    if len(cells) == 1:
        # <td><br /></td> 단독(내용 없음) = 연 사이 공백 줄, 가사가 아니다
        if not cells[0]:
            return []
        # <td colspan="N" class="merged">text</td> = 원문/로마자 구분 없는 삽입구
        return [SourceLine(text=cells[0])]
    text, pronunciation, translation = (cells + ["", ""])[:3]
    if not text:
        return []
    return [SourceLine(text=text, pronunciation=pronunciation or None, translation=translation or None)]


def parse_lyrics_table(page_html: str) -> tuple[list[SourceLine], bool] | None:
    """파싱된 위키 HTML → (가사 줄 목록, 번역 열 유무). 일본어 가사 표가 없으면 None."""
    table = _find_japanese_lyrics_table(page_html)
    if not table:
        return None

    header = _HEADER_ROW_RE.search(table)
    if not header:
        return None
    header_count = len(_TH_RE.findall(header.group(1)))
    if header_count < 2:  # 최소 원문 + 로마자
        return None

    lines: list[SourceLine] = []
    for row in _ROW_RE.finditer(table[header.end() :]):
        cells = [cell_text(c.group(1)) for c in _CELL_RE.finditer(row.group(1))]
        lines.extend(_row_to_lines(cells))
    return lines, header_count >= 3


# ── 검색 + 조회 ───────────────────────────────────────────────────

_default_fetcher: WikiFetcher | None = None


def _fetcher() -> WikiFetcher:
    global _default_fetcher
    if _default_fetcher is None:
        _default_fetcher = WikiFetcher()
    return _default_fetcher


def _search_top_hit(title: str, fetcher: WikiFetcher) -> tuple[int, str] | None:
    """(pageid, 제목). 정규화 접두 검증을 통과한 첫 검색 결과만 돌려준다."""
    query = urlencode(
        {
            "action": "query",
            "list": "search",
            "srsearch": title,
            "srlimit": str(SEARCH_LIMIT),
            "format": "json",
        }
    )
    data = fetcher.get_json(f"{API_URL}?{query}")
    hits = ((data or {}).get("query") or {}).get("search") or []
    if not hits:
        return None
    for hit in hits:
        hit_title = hit.get("title") or ""
        if page_title_matches_candidate(title, hit_title):
            return int(hit["pageid"]), hit_title
    return None


def _fetch_parsed_page(pageid: int, fetcher: WikiFetcher) -> tuple[str, str] | None:
    """pageid로 정규 페이지 제목과 파싱 HTML을 함께 읽는다."""
    query = urlencode(
        {"action": "parse", "pageid": str(pageid), "prop": "text", "format": "json"}
    )
    data = fetcher.get_json(f"{API_URL}?{query}")
    parsed = (data or {}).get("parse") or {}
    page_title = parsed.get("title")
    text = parsed.get("text")
    if isinstance(text, dict):
        text = text.get("*")
    if not isinstance(page_title, str) or not page_title or not isinstance(text, str):
        return None
    return page_title, text


# JS ``encodeURI``가 손대지 않는 문자들 — 확장과 같은 URL을 만들기 위한 safe 집합.
_ENCODE_URI_SAFE = ";,/?:@&=+$-_.!~*'()#"


def wiki_url(page_title: str) -> str:
    """MediaWiki 문서 URL — 공백만 '_'로 바꾸고 괄호·'*'·'/'는 그대로 남긴다.

    확장이 ``encodeURI``를 쓰는 것과 같은 규칙이다(``encodeURIComponent``를 쓰면 '/'까지
    %2F로 깨져 문서를 못 찾는다).
    """
    path = quote(page_title.replace(" ", "_"), safe=_ENCODE_URI_SAFE)
    return f"{BASE_URL}/wiki/{path}"


def lookup(title: str, fetcher: WikiFetcher | None = None) -> MirahezeSong | None:
    """제목으로 곡 페이지를 찾아 가사(원문 + 로마자 + [영어 번역])를 반환. 없으면 None.

    :func:`title_candidates`가 낸 후보를 순서대로 시도한다 — 검색해서 접두 일치 히트를
    얻고, 실제 조회한 정규 페이지 제목도 같은 접두 검증을 통과하며 일본어 가사 표가
    있을 때만 채택한다. 하나라도 어긋나면 다음 후보로 넘어가고, 검증되지 않은 검색
    1위로는 절대 물러나지 않는다.
    """
    trimmed = title.strip()
    if not trimmed:
        return None
    fetch = fetcher or _fetcher()

    candidates = title_candidates(trimmed)
    for candidate in candidates:
        hit = _search_top_hit(candidate, fetch)
        if hit is None:
            continue
        pageid, _search_title = hit
        fetched = _fetch_parsed_page(pageid, fetch)
        if fetched is None:
            continue
        page_title, page_html = fetched
        if not page_title_matches_candidate(candidate, page_title):
            continue
        parsed = parse_lyrics_table(page_html)
        if parsed is None or not parsed[0]:
            continue  # 이 후보의 채택 페이지엔 가사 표가 없다 — 다음 후보로
        lines, has_translation = parsed
        return MirahezeSong(
            page_title=page_title,
            url=wiki_url(page_title),
            lines=lines,
            has_translation=has_translation,
        )
    return None
