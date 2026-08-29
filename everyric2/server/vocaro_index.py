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
from typing import Literal

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


@dataclass(frozen=True)
class MatchDecision:
    """자동 채택 가능한 곡 식별 판정.

    ``entry``가 없는 이유를 ``ambiguous``와 ``not_found``로 구분해, 클라이언트가 모호한
    결과를 느슨한 로컬 폴백으로 다시 살려내지 않게 한다.
    """

    status: Literal["matched", "ambiguous", "not_found", "index_empty"]
    reason: str
    entry: SongEntry | None = None
    candidate_count: int = 0
    matched_query: str | None = None


# ── 모듈 전역 캐시 (프로세스 메모리) ─────────────────────────────
_state_lock = threading.Lock()
_building = False
_cache: list[SongEntry] | None = None
_built_at: str | None = None
_confusable_source: list[SongEntry] | None = None
_confusable_by_key: dict[str, list[SongEntry]] = {}

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "everyric2-vocaro-index/1.0 (lyrics sync helper)"})


# ── 공개 API ──────────────────────────────────────────────────────

# 제목 정규화·후보 생성 규칙은 링크 후보 탐색(api/sync)과 공유한다 — title_match 단일 출처.
_normalize_title = title_match.normalize_title
_identity_key = title_match.identity_key


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


def _entry_identity_keys(entry: SongEntry) -> set[str]:
    return {
        _identity_key(field)
        for field in (entry.ja, entry.ko, entry.slug.replace("-", " "))
        if field and _identity_key(field)
    }


def _confusable_entries(entries: list[SongEntry], key: str) -> list[SongEntry]:
    """기호 제거/영문 단복수 한 글자 차이로 다른 실재 제목이 되는 항목군."""
    global _confusable_by_key, _confusable_source
    if _confusable_source is not entries:
        identity_entries: dict[str, dict[str, SongEntry]] = {}
        loose_groups: dict[tuple[str, str], set[str]] = {}
        for entry in entries:
            for field in (entry.ja, entry.ko, entry.slug.replace("-", " ")):
                if not field:
                    continue
                identity = _identity_key(field)
                loose = _normalize_title(field)
                if not identity or not loose:
                    continue
                identity_entries.setdefault(identity, {})[entry.slug] = entry
                loose_groups.setdefault(("punct", loose), set()).add(identity)
                if loose.isascii() and loose.isalnum():
                    plural_family = loose[:-1] if loose.endswith("s") and len(loose) > 1 else loose
                    loose_groups.setdefault(("plural", plural_family), set()).add(identity)

        built: dict[str, dict[str, SongEntry]] = {}
        for identities in loose_groups.values():
            slugs = {
                slug
                for identity in identities
                for slug in identity_entries.get(identity, {})
            }
            if len(identities) < 2 or len(slugs) < 2:
                continue
            for identity in identities:
                bucket = built.setdefault(identity, {})
                for other in identities:
                    bucket.update(identity_entries.get(other, {}))
        _confusable_by_key = {
            identity: list(by_slug.values()) for identity, by_slug in built.items()
        }
        _confusable_source = entries
    return _confusable_by_key.get(key, [])


def _disambiguate_exact_hits(
    hits: list[SongEntry],
    *,
    artist: str | None,
    channel: str | None,
    context: list[str],
) -> SongEntry | None:
    """동명이곡을 프로듀서/채널 단서로 하나까지 좁힌다. 동점은 임의 채택하지 않는다."""
    hints = [_normalize_title(value) for value in (artist, channel, *context) if value]
    # IA·U·M 같은 1~2자 보컬/아티스트 표기는 긴 슬러그 안에 우연히 너무 자주 나타난다.
    # 동명이곡 자동 선택 근거로는 3자 이상만 쓴다. 짧은 단서는 선택하지 않고 ambiguous가 안전하다.
    hints = [hint for hint in hints if len(hint) >= 3]
    if not hints:
        return None

    scored: list[tuple[int, SongEntry]] = []
    for entry in hits:
        producer_keys = {
            _normalize_title(value.rsplit("/", 1)[1])
            for value in (entry.ko, entry.ja)
            if value and "/" in value and value.rsplit("/", 1)[1].strip()
        }
        # 전체 제목/slug의 부분문자열은 `Eve`→`Eleven...` 같은 우연 일치를 만든다.
        # 위키가 명시한 `/producer` suffix와 정확히 같은 단서만 동명이곡 선택에 쓴다.
        scored.append((sum(1 for hint in hints if hint in producer_keys), entry))
    best_score = max(score for score, _ in scored)
    best = [entry for score, entry in scored if score == best_score and score > 0]
    return best[0] if len(best) == 1 else None


_FEAT_SUFFIX_RE = re.compile(r"(?:^|\s)(?:feat|ft)\.?\s*\S.*$", re.IGNORECASE)
_FEAT_VOCAL_RE = re.compile(r"(?:^|\s)(?:feat|ft)\.?\s*(\S.*)$", re.IGNORECASE)
_SAFE_ROMAN_ALIAS_RE = re.compile(r"^(.+?)\s*[（(]([A-Za-z0-9 '\-–—]+)[）)]\s*$")
_BRACKET_GROUP_RE = re.compile(
    r"【(?P<corner>[^】]*)】|\[(?P<square>[^\]]*)\]|（(?P<wide>[^）]*)）|\((?P<round>[^)]*)\)"
)
_RAW_SPLIT_RE = re.compile(r"\s*[/／|｜ㅣ–—―~〜]\s*|\s+-\s+")


def _dedupe_candidates(candidates: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = _identity_key(candidate)
        if key and key not in seen:
            seen.add(key)
            out.append(candidate.strip())
    return out


def _strip_safe_bracket_noise(title: str) -> str:
    """잡표기/알려진 보컬 괄호만 제거하고 버전·부제 괄호는 보존한다."""

    def replace(match: re.Match[str]) -> str:
        content = next((group for group in match.groups() if group is not None), "")
        key = _normalize_title(content)
        leftover = _normalize_title(title_match.strip_noise_tokens(content))
        # `(^_-)-☆` 같은 표정 기호는 정규화가 비어도 제목의 일부다. 실제 잡토큰이
        # 제거되어 원래 영숫자 key가 사라진 경우와 빈 괄호만 걷는다.
        is_noise = (not content.strip()) or (bool(key) and not leftover)
        return " " if (is_noise or key in _KNOWN_VOCAL_KEYS) else match.group(0)

    return _BRACKET_GROUP_RE.sub(replace, title)


def strong_candidates(title: str) -> list[str]:
    """제목 전체를 보존한 채 제거 근거가 명확한 잡표기만 걷은 후보."""
    trimmed = title.strip()
    no_feat = _FEAT_SUFFIX_RE.sub("", trimmed).strip()
    no_noise = _strip_safe_bracket_noise(trimmed).strip()
    candidates = [trimmed, no_feat, no_noise, _FEAT_SUFFIX_RE.sub("", no_noise).strip()]
    return _dedupe_candidates([candidate for candidate in candidates if candidate])


def corroborated_raw_candidates(
    raw_title: str,
    *,
    artist: str | None,
    channel: str | None,
) -> list[str]:
    """반대편 조각이 실제 artist/channel/보컬일 때만 raw 구분자 조각을 연다."""
    candidates = strong_candidates(raw_title)
    evidence = {_normalize_title(value) for value in (artist, channel) if value}
    parts = [part.strip() for part in _RAW_SPLIT_RE.split(raw_title) if part.strip()]
    part_keys = [
        {_normalize_title(value) for value in strong_candidates(part)}
        for part in parts
    ]
    corroborated = {
        index
        for index, keys in enumerate(part_keys)
        if any(key in evidence or key in _KNOWN_VOCAL_KEYS for key in keys)
        or (
            (feat_match := _FEAT_VOCAL_RE.search(parts[index])) is not None
            and _normalize_title(feat_match.group(1)) in _KNOWN_VOCAL_KEYS
        )
    }
    # 셋 이상 조각에서 한 보컬명만 확인됐다고 나머지 전부를 곡명으로 열면
    # `Unknown / Hatsune Miku / ロキ`가 실제 다른 곡 ロキ로 떨어진다. 증거가 아닌 조각이
    # 정확히 하나일 때만 그 한 조각을 곡명으로 연다(`Song / Producer / Miku`는 유지).
    uncorroborated = set(range(len(parts))) - corroborated
    if corroborated and len(uncorroborated) == 1:
        for index, part in enumerate(parts):
            if index not in uncorroborated:
                continue
            for candidate in strong_candidates(part):
                normalized = _normalize_title(candidate)
                # M/S/U 같은 한 글자 조각은 반대편 증거가 있어도 임의 곡명으로 채택하지 않는다.
                if len(normalized) >= 2 or (not normalized and _identity_key(candidate)):
                    candidates.append(candidate)
    return _dedupe_candidates(candidates)


def _parenthetical_alias_hits(title: str, entries: list[SongEntry]) -> tuple[str, list[SongEntry]] | None:
    """`원제 (Roman)`의 Roman이 같은 항목의 실제 별칭일 때만 괄호를 제거한다."""
    match = _SAFE_ROMAN_ALIAS_RE.match(title.strip())
    if not match:
        return None
    base = match.group(1).strip()
    alias = _normalize_title(match.group(2))
    base_key = _identity_key(base)
    hits = [
        entry
        for entry in entries
        if base_key in _entry_identity_keys(entry)
        and alias
        in {
            _normalize_title(field)
            for field in (entry.ja, entry.ko, entry.slug.replace("-", " "))
            if field
        }
    ]
    return (base, hits) if hits else None


def match_with_evidence(
    title: str,
    *,
    title_candidates: list[str] | None = None,
    raw_title: str | None = None,
    artist: str | None = None,
    channel: str | None = None,
) -> MatchDecision:
    """제목과 보존된 영상 단서로 자동 채택 가능한 항목만 반환한다.

    기존 포함 매칭은 검색 후보를 보여주는 데는 유용하지만, 가사를 자동 채택하는 자리에서는
    ``STARGAZERS → StargazeR``·``ワンダー → スチールワンダー`` 같은 오답을 만든다.
    여기서는 후보 변형과 인덱스 필드가 기호까지 정확히 같을 때만 채택한다. Chrome이 채널
    근거로 제목 방향을 뒤집은 경우 ``title_candidates``의 순서가 그 근거의 우선순위다.
    """
    _ensure_loaded()
    entries = _cache or []
    if not entries:
        return MatchDecision(status="index_empty", reason="index_empty")

    roots: list[str] = []
    for value in [*(title_candidates or []), title]:
        value = value.strip()
        if value and value not in roots:
            roots.append(value)

    # title/title_candidates는 이미 Chrome이 곡명으로 정리한 값이다. 자동 경로에서 다시 임의로
    # 괄호·구분자를 잘라 짧은 다른 곡으로 내려가지 않는다.
    candidate_groups = [strong_candidates(root) for root in roots]

    # 새 클라이언트는 방향이 검증된 후보를 보낸다. 구 클라이언트의 raw_title은 기본 제목이
    # 미스일 때만 보조하되, 이미 artist/channel로 식별된 조각은 곡명 후보에서 제외한다.
    # raw 후보는 여기서 한 번만 분해한다 — 다시 root로 넣으면 full 제목을 재분해하는 과정에서
    # 제외했던 아티스트 조각이 되살아난다.
    if raw_title:
        excluded = {_normalize_title(value) for value in (artist, channel) if value}
        source_candidates = corroborated_raw_candidates(
            raw_title,
            artist=artist,
            channel=channel,
        )
        raw_candidates = [
            value
            for value in source_candidates
            if _normalize_title(value) not in excluded
            and not _is_vocal_only_fragment(
                _normalize_title(value), _normalize_title(raw_title)
            )
        ]
        if raw_candidates:
            candidate_groups.append(raw_candidates)

    for root_index, candidates in enumerate(candidate_groups):
        for candidate in candidates:
            key = _identity_key(candidate)
            if not key:
                continue
            hits = [entry for entry in entries if key in _entry_identity_keys(entry)]
            if not hits:
                continue
            if len(hits) == 1:
                confusable = _confusable_entries(entries, key)
                if len(confusable) > 1:
                    chosen = _disambiguate_exact_hits(
                        confusable,
                        artist=artist,
                        channel=channel,
                        context=[],
                    )
                    # producer/channel은 exact 제목을 확인할 수는 있어도 다른 실재 제목으로
                    # 뒤집을 수 없다. 채널은 업로더·커버러일 수 있으므로 충돌은 보류한다.
                    if chosen is hits[0]:
                        return MatchDecision(
                            status="matched",
                            reason="confusable_disambiguated",
                            entry=chosen,
                            candidate_count=len(confusable),
                            matched_query=candidate,
                        )
                    return MatchDecision(
                        status="ambiguous",
                        reason="confusable_title",
                        candidate_count=len(confusable),
                        matched_query=candidate,
                    )
                if root_index == 0:
                    reason = "exact_title"
                elif raw_title and root_index >= len(roots):
                    reason = "raw_evidence_title"
                else:
                    reason = "evidence_title"
                return MatchDecision(
                    status="matched",
                    reason=reason,
                    entry=hits[0],
                    candidate_count=1,
                    matched_query=candidate,
                )
            context = [value for value in candidates if value != candidate]
            chosen = _disambiguate_exact_hits(
                hits,
                artist=artist,
                channel=channel,
                context=context,
            )
            if chosen is not None:
                return MatchDecision(
                    status="matched",
                    reason="identity_disambiguated",
                    entry=chosen,
                    candidate_count=len(hits),
                    matched_query=candidate,
                )
            return MatchDecision(
                status="ambiguous",
                reason="duplicate_title",
                candidate_count=len(hits),
                matched_query=candidate,
            )

    # `(Roki)` 같은 로마자 괄호는 실제 같은 항목의 slug/다른 별칭과도 일치할 때만
    # 제거한다. `(Remix)`·`(.Type.L)`·`(2024)`는 별칭 증거가 없어 여기서 열리지 않는다.
    for root in roots:
        alias_result = _parenthetical_alias_hits(root, entries)
        if alias_result is None:
            continue
        candidate, hits = alias_result
        if len(hits) == 1:
            return MatchDecision(
                status="matched",
                reason="parenthetical_alias",
                entry=hits[0],
                candidate_count=1,
                matched_query=candidate,
            )
        chosen = _disambiguate_exact_hits(
            hits,
            artist=artist,
            channel=channel,
            context=[],
        )
        if chosen is not None:
            return MatchDecision(
                status="matched",
                reason="parenthetical_alias_disambiguated",
                entry=chosen,
                candidate_count=len(hits),
                matched_query=candidate,
            )
        return MatchDecision(
            status="ambiguous",
            reason="duplicate_parenthetical_alias",
            candidate_count=len(hits),
            matched_query=candidate,
        )

    return MatchDecision(status="not_found", reason="no_exact_title")


def match(title: str) -> SongEntry | None:
    """제목 하나로 자동 채택해도 안전한 위키 곡 항목을 찾는다.

    상세 판정이 필요한 API는 :func:`match_with_evidence`를 직접 사용한다.
    """
    return match_with_evidence(title).entry


def search_matches(title: str, limit: int = 5) -> list[tuple[SongEntry, float]]:
    """사용자가 직접 고르는 수동 검색용 느슨한 후보를 점수순으로 반환."""
    _ensure_loaded()
    entries = _cache or []
    return title_match.rank_matches(
        title,
        [
            (
                entry,
                " / ".join(
                    value
                    for value in (entry.ja, entry.ko, entry.slug.replace("-", " "))
                    if value
                ),
            )
            for entry in entries
        ],
        min_score=0.5,
        limit=limit,
    )


def search_match(title: str) -> SongEntry | None:
    """레거시 단일 후보 호환용. 신규 API는 :func:`search_matches` 전체를 노출한다.

    자동 채택의 :func:`match_with_evidence`와 의도적으로 분리한다. 포함/잡표기 유사도는
    여기서만 허용되며, 결과는 Chrome 검색 시트에 후보로 보일 뿐 자동 가사로 채택되지 않는다.
    """
    ranked = search_matches(title, limit=1)
    return ranked[0][0] if ranked else None


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
