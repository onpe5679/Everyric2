"""보카로 가사 위키(vocaro.wikidot.com) 원제 매칭 인덱스.

유튜브 영상 제목이 일본어 원제로 되어 있으면 클라이언트(everyric2-chrome)의
'수록곡 일람' 초성 인덱스(한국어 독음 기준)로는 곡을 찾지 못한다. 이 모듈은
전체 42개 인덱스 페이지에서 슬러그/한국어 제목을 모으고, 각 곡 페이지의
title-cell(원제, 일본어)까지 채운 인덱스를 서버에 저장해 원제 → 슬러그
매칭을 가능하게 한다.

파서 규칙은 everyric2-chrome/src/lib/vocaro.ts의 parseIndexEntries /
parseSongPage / findMatch 로직과 동일하게 맞춘다.
"""

from __future__ import annotations

import html
import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import mkstemp

import requests

from everyric2.server import title_match

logger = logging.getLogger(__name__)

BASE_URL = "http://vocaro.wikidot.com"

# everyric2/server/vocaro_index.py -> parents[2]는 저장소 루트 (models/rmvpe 경로 계산과 동일 관례)
INDEX_PATH = Path(__file__).resolve().parents[2] / "models" / "vocaro_index.json"

# 한글 초성(h1~h14) + 영문(a~z) + 숫자/기호 = 총 42개 '수록곡 일람' 페이지
INDEX_PAGES = (
    [f"allsongs-h{i}" for i in range(1, 15)]
    + [f"allsongs-{c}" for c in "abcdefghijklmnopqrstuvwxyz"]
    + ["allsongs-num", "allsongs-symbols"]
)

SONG_FETCH_CONCURRENCY = 6
REQUEST_TIMEOUT_SEC = 8.0
LOG_PROGRESS_EVERY = 500
EXCLUDED_SLUG_PREFIXES = ("allsongs", "system", "guide")


@dataclass
class SongEntry:
    slug: str
    ko: str
    ja: str | None = None


# ── 모듈 전역 캐시 (프로세스 메모리) ─────────────────────────────
_state_lock = threading.Lock()
_building = False
_cache: list[SongEntry] | None = None
_built_at: str | None = None

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "everyric2-vocaro-index/1.0 (lyrics sync helper)"})


# ── 공개 API ──────────────────────────────────────────────────────

# 제목 정규화·후보 생성 규칙은 링크 후보 탐색(api/sync)과 공유한다 — title_match 단일 출처.
_candidate_queries = title_match.candidate_queries
_normalize_title = title_match.normalize_title


# 잘 알려진 보컬로이드/음성합성 보컬명 — 유튜브 영상 제목의 "곡명 / 보컬명" 관례에서
# 보컬명 자체가 위키에 실재하는 곡 제목과 우연히 같아 오매칭되는 것을 막는 재료다
# (2026-08-03 실측: `match('depresso. / 初音ミク')`가 depresso.(색인에 없음) 대신
# ryo의 곡 「初音ミク」에 매칭됐다 — 보컬명이 곡 후보 조각으로 쪼개져 정확 일치했다).
# 대소문자·전각은 _normalize_title이 흡수하므로 한 표기만 있어도 되지만, 로마자 표기가
# 흔한 것들은 함께 적어 둔다.
_KNOWN_VOCAL_NAMES = (
    "初音ミク", "Hatsune Miku",
    "鏡音リン", "Kagamine Rin",
    "鏡音レン", "Kagamine Len",
    "巡音ルカ", "Megurine Luka",
    "GUMI", "グミ",
    "IA",
    "KAITO", "カイト",
    "MEIKO", "メイコ",
    "重音テト", "Kasane Teto",
    "可不", "Kafu",
    "flower", "v_flower", "v flower",
    "歌愛ユキ", "Kaai Yuki",
    "星界", "Seikai",
    "裏命",
    "知声", "Chise",
)
_KNOWN_VOCAL_KEYS = frozenset(_normalize_title(name) for name in _KNOWN_VOCAL_NAMES)


def _is_vocal_only_fragment(q: str, full_norm: str) -> bool:
    """``q``가 잘 알려진 보컬명이고, 쿼리 전체가 그 보컬명만은 아닌가.

    참이면 이 후보는 "그 보컬이 부른 어떤 곡"이 아니라 "쿼리 속 보컬명이 위키의 다른
    곡 제목과 우연히 같다"는 뜻이다. 쿼리 전체가 보컬명뿐이면(``q == full_norm`` —
    진짜 그 곡을 찾는 경우) False라 정상적으로 매칭을 허용한다.
    """
    return q in _KNOWN_VOCAL_KEYS and q != full_norm


def match(title: str) -> SongEntry | None:
    """제목(원제 또는 한국어 독음 어느 쪽이든)으로 위키 곡 항목을 찾는다.

    유튜브 풀 제목("熱異常 - いよわ feat.初音ミク" 등)도 구분자 분해 후보로 재시도한다.
    ① 후보 순서대로 정규화 정확 일치, ② 후보 순서대로 상호 포함 + 길이비 >= 0.5
    (vocaro.ts findMatch와 동일 기준). 인덱스가 아직 구축되지 않았으면 None.
    """
    _ensure_loaded()
    entries = _cache or []
    if not entries:
        return None

    queries = _candidate_queries(title)
    if not queries:
        return None

    # 포함 매칭의 아티스트 토큰 가드 재료이자(기존) 보컬명 가드 재료(신규) — 풀 쿼리
    # 정규화본. 이 위치로 옮겼다 — 두 매칭 단계(정확 일치·포함) 모두 이 값이 필요하다.
    full_norm = _normalize_title(title)

    for q in queries:
        if _is_vocal_only_fragment(q, full_norm):
            # 쿼리 속 보컬명(初音ミク 등)이 위키의 다른 곡 제목과 우연히 같다 — 쿼리에
            # 남은 다른 토큰(depresso. 등)이 진짜 찾는 곡이다. _KNOWN_VOCAL_KEYS 문서 참고.
            continue
        hits: list[SongEntry] = []
        for entry in entries:
            # 슬러그가 3순위 별칭인 이유(2026-08-03 실측): 인덱스는 ko/ja 제목만 갖는데
            # 영상 제목이 영문 전사("Candy Cookie Chocolate")인 곡은 어느 쪽에도 안
            # 걸린다 — 위키 슬러그(candy-cookie-chocolate)가 바로 그 영문 전사다.
            for field in (entry.ja, entry.ko, entry.slug.replace("-", " ")):
                if field and _normalize_title(field) == q:
                    hits.append(entry)
                    break
        if not hits:
            continue
        if len(hits) == 1:
            return hits[0]
        # 동명이곡(2026-08-03 실측: シンデレラ가 ZIG판/DECO*27판 둘) — 제목만으로는 못
        # 가르므로 풀 쿼리의 **다른** 후보 토큰(아티스트 등)이 항목의 ko/ja/슬러그에
        # 나타나는 수로 가른다. 전부 0이면 기존처럼 인덱스 순서 첫 항목(결정론 유지).
        def _artist_bonus(entry: SongEntry) -> int:
            hay = _normalize_title(
                " ".join(x for x in (entry.ko or "", entry.ja or "", entry.slug.replace("-", " ")))
            )
            return sum(
                1
                for other in queries
                if other != q and len(other) >= 3 and other in hay
            )

        return max(hits, key=_artist_bonus)

    # full_norm은 위에서 이미 계산했다(정확 일치 패스의 보컬명 가드와 공유) — q ⊂ n
    # 방향에서 n의 나머지(제목부)가 풀 쿼리 어디에도 없으면, 겹친 것은 아티스트 이름뿐
    # 이라는 뜻이다(아래 아티스트 토큰 가드).
    for q in queries:
        if _is_vocal_only_fragment(q, full_norm):
            continue
        best: tuple[int, SongEntry] | None = None
        for entry in entries:
            # 슬러그 별칭은 **정확 일치 패스에만** 둔다. 포함 매칭에 넣으면 동명이곡
            # 넘버링 슬러그(melt-2 → "melt2")가 rest="2"(2자 미만)로 아티스트 토큰
            # 가드를 그냥 통과해 오탐 표면이 넓어진다(엣지 감사 #8). 장식 제목
            # ("… (Official MV)")은 candidate_queries가 괄호를 벗긴 후보를 이미
            # 만들므로 정확 일치 패스가 잡는다.
            for field in (entry.ja, entry.ko):
                if not field:
                    continue
                n = _normalize_title(field)
                if len(n) < 2:
                    continue
                if _is_vocal_only_fragment(n, full_norm):
                    # q 자체가 아니라 **항목 필드**가 보컬명뿐인 경우 — 쿼리가 그 보컬명을
                    # 포함하는 더 긴 문자열(q == full_norm, 위쪽 q 레벨 가드는 안 걸림)이면
                    # 여기서 걸린다("어떤곡 / 鏡音リン" 같은 실측 2호, 2026-08-03).
                    continue
                ratio = min(len(q), len(n)) / max(len(q), len(n))
                if not ((q in n or n in q) and ratio >= 0.5):
                    continue
                if n in q and n != q and ratio <= 0.5:
                    # 역방향 오매핑(2026-08-03 실측 2호): 3글자 곡 "Dec."이 아티스트
                    # 후보 "deco27" **안에** 포함(비율 정확히 0.5)돼 붙었다. 항목 제목이
                    # 후보의 절반 이하만 덮는 포함은 우연 일치가 지배한다 — 이 방향은
                    # 엄격 초과만 허용한다(정확 일치는 ① 패스가 이미 잡는다).
                    continue
                if q in n and n != q and q != full_norm:
                    # 실측 오매핑(2026-08-03): 쿼리 "DECO*27 - ダミーロマンス feat…"의
                    # 아티스트 후보 "deco27"이 인덱스 ko 필드 "신데렐라/DECO*27"에
                    # 포함돼 **다른 곡**에 붙었다. q가 다구획 쿼리의 부분 후보일 때
                    # (q != full_norm — 사용자가 친 문자열 전체가 아닐 때), 포함
                    # 매칭이 정당하려면 n에서 q를 뺀 나머지(그 항목의 실제 제목부)가
                    # 풀 쿼리 안에도 있어야 한다 — 없다면 겹친 건 공유 토큰(아티스트)
                    # 뿐이므로 기각한다. 쿼리 전체가 위키 제목의 부분 문자열인 경우
                    # (q == full_norm)는 기존처럼 정당한 부분 제목 검색이다.
                    rest = n.replace(q, "", 1)
                    if len(rest) >= 2 and rest not in full_norm:
                        continue
                if best is None or len(n) > best[0]:
                    best = (len(n), entry)
        if best:
            return best[1]
    return None


def index_status() -> dict:
    """현재 인덱스 상태 요약."""
    _ensure_loaded()
    entries = _cache or []
    return {
        "built_at": _built_at,
        "total": len(entries),
        "with_ja": sum(1 for e in entries if e.ja),
        "building": _building,
    }


def is_building() -> bool:
    return _building


def build_index(force: bool = False) -> dict:
    """인덱스를 (증분) 구축한다.

    1) 42개 '수록곡 일람' 페이지에서 slug/한국어 제목을 모으고(dedup),
    2) force=False면 기존에 이미 확보된 슬러그는 건너뛰고, 새 슬러그만 곡 페이지를
       fetch해 title-cell(원제)을 채운다 — 재실행 시 새 곡만 크롤하는 증분 방식.
    3) force=True면 기존 캐시를 무시하고 전량 재수집한다.

    동시 빌드 요청은 무시(락)하고, 완료되면 JSON을 원자적으로 저장한다.
    """
    global _building, _cache, _built_at

    with _state_lock:
        if _building:
            logger.info("vocaro_index: 이미 빌드가 진행 중이라 요청을 무시합니다")
            return {"status": "already_building"}
        _building = True

    start = time.monotonic()
    try:
        _ensure_loaded()
        existing_by_slug: dict[str, SongEntry] = {} if force else {e.slug: e for e in (_cache or [])}

        # 1) 인덱스 페이지 수집 (42개, 순차 — 곡 페이지 단계와 합쳐도 동시 요청이 6을 넘지 않게)
        collected: dict[str, str] = {}
        for page in INDEX_PAGES:
            page_html = _fetch(f"{BASE_URL}/{page}")
            if page_html is None:
                logger.warning("vocaro_index: 인덱스 페이지 요청 실패 - %s", page)
                continue
            for slug, ko in _parse_index_entries(page_html):
                collected.setdefault(slug, ko)
        logger.info("vocaro_index: 인덱스 페이지 수집 완료 - 슬러그 %d개", len(collected))

        # 2) 원제 미확보 슬러그만 곡 페이지에서 fetch (동시성 SONG_FETCH_CONCURRENCY)
        new_slugs = [slug for slug in collected if slug not in existing_by_slug]
        fetched = 0
        failed = 0
        new_entries: dict[str, SongEntry] = {}

        with ThreadPoolExecutor(max_workers=SONG_FETCH_CONCURRENCY) as pool:
            futures = {pool.submit(_fetch_ja, slug): slug for slug in new_slugs}
            for future in as_completed(futures):
                slug = futures[future]
                try:
                    ja = future.result()
                except Exception as e:  # 개별 곡 페이지 실패는 skip하고 계속
                    logger.warning("vocaro_index: 곡 페이지 처리 실패 - %s (%s)", slug, e)
                    ja = None
                if ja is None:
                    failed += 1
                new_entries[slug] = SongEntry(slug=slug, ko=collected[slug], ja=ja)
                fetched += 1
                if fetched % LOG_PROGRESS_EVERY == 0:
                    logger.info(
                        "vocaro_index: 진행 %d/%d (실패 %d)", fetched, len(new_slugs), failed
                    )

        merged = dict(existing_by_slug)
        merged.update(new_entries)
        entries_list = list(merged.values())
        built_at = datetime.now(timezone.utc).isoformat()
        _save_to_disk(built_at, entries_list)

        with _state_lock:
            _cache = entries_list
            _built_at = built_at

        elapsed = time.monotonic() - start
        with_ja = sum(1 for e in entries_list if e.ja)
        logger.info(
            "vocaro_index: 빌드 완료 - 총 %d곡, 원제 확보 %d곡, 신규 %d곡, 실패 %d건, %.1f초",
            len(entries_list), with_ja, len(new_slugs), failed, elapsed,
        )
        return {
            "status": "done",
            "total": len(entries_list),
            "with_ja": with_ja,
            "new": len(new_slugs),
            "failed": failed,
            "elapsed_sec": round(elapsed, 1),
        }
    finally:
        with _state_lock:
            _building = False


# ── 캐시 로드/저장 ────────────────────────────────────────────────

def _ensure_loaded() -> None:
    global _cache, _built_at
    if _cache is not None:
        return
    with _state_lock:
        if _cache is None:
            _built_at, _cache = _load_from_disk()


def _load_from_disk() -> tuple[str | None, list[SongEntry]]:
    if not INDEX_PATH.exists():
        return None, []
    try:
        data = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("vocaro_index: 저장된 인덱스 로드 실패, 빈 인덱스로 시작 - %s", e)
        return None, []
    entries = [SongEntry(**e) for e in data.get("entries", [])]
    return data.get("built_at"), entries


def _save_to_disk(built_at: str, entries: list[SongEntry]) -> None:
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"built_at": built_at, "entries": [asdict(e) for e in entries]}
    fd, tmp_path = mkstemp(dir=INDEX_PATH.parent, prefix=".vocaro_index_", suffix=".tmp")
    try:
        with open(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        Path(tmp_path).replace(INDEX_PATH)  # 원자적 교체
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise


# ── 네트워크 ──────────────────────────────────────────────────────

def _fetch(url: str) -> str | None:
    try:
        resp = _SESSION.get(url, timeout=REQUEST_TIMEOUT_SEC)
        if resp.status_code != 200:
            return None
        if resp.encoding is None or resp.encoding.lower() == "iso-8859-1":
            resp.encoding = resp.apparent_encoding or "utf-8"
        return resp.text
    except requests.RequestException:
        return None


def _fetch_ja(slug: str) -> str | None:
    page_html = _fetch(f"{BASE_URL}/{slug}")
    if page_html is None:
        return None
    return _parse_title_cell(page_html)


# ── 파싱 (vocaro.ts와 동일 규칙) ─────────────────────────────────

_INDEX_ITEM_RE = re.compile(r'<li>\s*<a\s+href="/([^"#:]+)"[^>]*>([^<]+)</a>\s*</li>')
_TITLE_CELL_RE = re.compile(r'<th[^>]*class="[^"]*title-cell[^"]*"[^>]*>([\s\S]*?)</th>')
_RT_SPAN_RE = re.compile(r'<span class="rt">[\s\S]*?</span>')
_BR_RE = re.compile(r"<br\s*/?>")
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _parse_index_entries(page_html: str) -> list[tuple[str, str]]:
    """'수록곡 일람' 페이지에서 (slug, 한국어제목) 쌍을 추출."""
    out: list[tuple[str, str]] = []
    for m in _INDEX_ITEM_RE.finditer(page_html):
        slug, raw_title = m.group(1), m.group(2)
        if slug.startswith(EXCLUDED_SLUG_PREFIXES):
            continue
        title = html.unescape(raw_title).strip()
        if title:
            out.append((slug, title))
    return out


def _parse_title_cell(page_html: str) -> str | None:
    """곡 페이지 HTML에서 title-cell(원제, 일본어) 텍스트를 추출. 없으면 None."""
    m = _TITLE_CELL_RE.search(page_html)
    if not m:
        return None
    title = _cell_text(m.group(1))
    return title or None


def _cell_text(cell_html: str) -> str:
    text = _RT_SPAN_RE.sub("", cell_html)  # 후리가나 읽기는 원문에서 제외
    text = _BR_RE.sub(" ", text)
    text = _TAG_RE.sub("", text)
    text = html.unescape(text)
    return _WS_RE.sub(" ", text).strip()
