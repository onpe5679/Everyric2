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
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import mkstemp

import requests

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

# 유튜브 곡 제목 관례("곡명 / 아티스트", "곡명 - 아티스트 feat.가수", "【가수】곡명" 등)의
# 구분자 — 하이픈은 양옆 공백이 있을 때만 분리해 합성어를 보존한다
_TITLE_SPLIT_RE = re.compile(r"\s*[/／|｜–—―~〜]\s*|\s+-\s+|\s*[「」『』【】\[\]]\s*")
_FEAT_RE = re.compile(r"(?:^|\s)(?:feat|ft)\.?\s*\S.*$", re.IGNORECASE)


# 괄호로 묶인 가수/독음 병기 («【初音ミク】곡명», «곡명 (아쿠노)») — 통째로 걷어낸 변형도 후보에 넣는다
_BRACKETED_RE = re.compile(r"【[^】]*】|「[^」]*」|『[^』]*』|\[[^\]]*\]|\([^)]*\)|（[^）]*）")


def _candidate_queries(title: str) -> list[str]:
    """풀 제목에서 곡명 후보를 정규화 형태로 생성.

    순서: 원문 → feat 제거 → 괄호 세그먼트 제거 → 각 변형의 구분자 조각(왼쪽 우선).
    괄호 제거 변형을 조각보다 먼저 두어 «【가수】곡명» 류에서 가수명이 곡명보다
    먼저 매칭되는 오탐을 막는다.
    """
    seen: set[str] = set()
    out: list[str] = []

    def add(raw: str) -> None:
        q = _normalize_title(raw)
        if len(q) >= 2 and q not in seen:
            seen.add(q)
            out.append(q)

    stripped = _BRACKETED_RE.sub(" ", title)
    variants = [title, _FEAT_RE.sub("", title), stripped, _FEAT_RE.sub("", stripped)]
    for v in variants:
        add(v)
    for v in variants:
        for part in _TITLE_SPLIT_RE.split(v):
            add(part)
            add(_FEAT_RE.sub("", part))
    return out


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

    for q in queries:
        for entry in entries:
            for field in (entry.ja, entry.ko):
                if field and _normalize_title(field) == q:
                    return entry

    for q in queries:
        best: tuple[int, SongEntry] | None = None
        for entry in entries:
            for field in (entry.ja, entry.ko):
                if not field:
                    continue
                n = _normalize_title(field)
                if len(n) < 2:
                    continue
                if (q in n or n in q) and min(len(q), len(n)) / max(len(q), len(n)) >= 0.5:
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


def _normalize_title(title: str) -> str:
    t = unicodedata.normalize("NFKC", title.lower())
    return "".join(ch for ch in t if ch.isalnum())
