"""보카로 곡 대량 인제스트 — 위키 코퍼스를 돌며 곡마다 영상을 찾아 싱크를 만든다.

한 번 호출 = 한 청크(기본 25곡). 처리 후 요약 JSON을 stdout에 찍고 끝난다 — 총괄이
``--offset``을 밀며 청크를 이어 돌린다. 상태는 파일에 남아 재실행이 멱등하다.

곡 하나의 파이프라인::

    ① 코퍼스        models/vocaro_index.json 엔트리 (slug / ko / ja)
    ② 중복 스킵     상태 파일 + DB(sync_results)의 제목 대조
    ③ miraheze      영어 번역 + 프로듀서 이름(다음 단계의 채널 대조에 쓴다)
    ④ 영상 해석     yt-dlp ytsearch (메타만) → 후보 검증 → 확신 없으면 스킵
                    확정된 video_id로 중복 대조를 한 번 더
    ⑤ vocaro        한글 독음 + 한국어 번역 (영상이 정해진 뒤에 받는다 — 스킵될 곡에
                    위키 요청을 쓰지 않는다)
    ⑥ 가사 확보     원문 언어 수동 자막 → (없으면) vocaro 원문 → (없으면) miraheze 원문
    ⑦ 제출          POST /api/sync/generate (line_meta = ⑤) → 완료까지 폴링 (상한 12분)
    ⑧ en 레이어     miraheze 영어 번역을 POST /api/sync/{vid}/translations (origin=wiki)

**가사에 자동 생성(ASR) 자막은 절대 쓰지 않는다.** 수동 트랙만 열거하고
(``manual_track_keys``), 받을 때도 ``auto=False``만 쓰고, 받은 본문의 문자 구성이 곡의
원문 언어와 맞는지까지 확인한다(:func:`caption_lyrics`) — 일본어 곡에 한국어 팬 번역
트랙이 원문으로 들어오면 정렬이 파국적으로 무너진다.

서버 자신에서 실행한다 (127.0.0.1 — localhost는 요청당 2초 IPv6 스톨).
admin 키는 환경변수 ``EVERYRIC_SERVER_ADMIN_API_KEY``로 받고 절대 출력하지 않는다.

    set -a; source <ENV_FILE>; set +a
    # 먼저 첫 25곡을 검수 (제출 없음 — 영상 해석·가사 확보까지만)
    .venv/bin/python scripts/bulk_ingest.py --dry-run --limit 25
    # 실제 인제스트, 청크를 이어서
    .venv/bin/python scripts/bulk_ingest.py --limit 25 --offset 0
    .venv/bin/python scripts/bulk_ingest.py --limit 25 --offset 25

종료 코드: 0 = 청크 정상 종료(개별 곡 실패는 요약에), 2 = 설정/사용법 오류,
3 = 체인 중단(yt-dlp 403 연속 — 쿠키 갱신이나 yt-dlp 업데이트가 필요하다).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from everyric2.server import title_match  # noqa: E402
from everyric2.server.api.quota_headers import (  # noqa: E402
    ACTOR_HEADER,
    OPS_ACTOR_BULK_INGEST,
)
from everyric2.server.services import youtube_captions as yc  # noqa: E402
from everyric2.sources import miraheze, vocaro  # noqa: E402
from everyric2.sources.base import ASCII_LB, ASCII_RB, WikiFetcher  # noqa: E402

logger = logging.getLogger("bulk_ingest")

DEFAULT_BASE_URL = "http://127.0.0.1:8300"
DEFAULT_STATE_PATH = "bench/out/bulk_state.json"
DEFAULT_CORPUS_PATH = "models/vocaro_index.json"
DEFAULT_DB_PATH = "everyric2.db"

#: 재생성 금지 영상 — 손으로 고친 싱크가 있어 덮어쓰면 안 된다
BLOCKED_VIDEO_IDS = frozenset({"b2NTglk9tvI", "BiQsFYrzKKM"})

#: 잡 하나를 기다리는 상한. 다운로드 + 분리 + 정렬 + 저장까지 여유를 둔 값이다
DEFAULT_JOB_TIMEOUT_SEC = 720.0

#: 서버가 번역 레이어 한 번에 받는 줄 수 상한 (api/sync.py _MAX_TRANSLATION_LAYER_LINES)
MAX_TRANSLATION_LAYER_LINES = 400

#: yt-dlp가 이 횟수만큼 연속으로 403/차단을 맞으면 체인을 세운다. 실측(메모리): 403은
#: 곡 탓이 아니라 이 서버 공인 IP·쿠키 탓이라 계속 돌려도 전부 실패하고 흔적만 남는다
YTDLP_BLOCK_ABORT_STREAK = 3


# ── 곡 목록 ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Song:
    slug: str
    ko: str
    ja: str | None
    #: 인계(handoff) 모드 — 영상이 이미 정해진 곡. 검색을 건너뛰고 이 영상 하나를
    #: 단일 후보로 기존 채점(pick_video)에 태운다. 채점을 생략하지 않는 이유:
    #: 제목 부분일치로 얻은 슬러그가 오매칭이면 엉뚱한 가사를 정렬하게 되기 때문.
    pinned_video_id: str | None = None

    @property
    def original_title(self) -> str:
        """검색·대조에 쓰는 원제. 원제가 없으면 한국어 제목으로 물러난다."""
        return self.ja or self.ko


def load_corpus(path: Path) -> list[Song]:
    """``models/vocaro_index.json``을 곡 목록으로. 순서는 파일 순서를 그대로 지킨다.

    청크를 이어 돌리는 ``--offset``이 안정된 순서를 전제로 하므로 정렬하지 않는다.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        Song(slug=e["slug"], ko=e.get("ko") or "", ja=e.get("ja") or None)
        for e in data.get("entries", [])
        if e.get("slug")
    ]


def load_skip_file(path: Path) -> tuple[set[str], set[str]]:
    """(슬러그, 영상 id) 스킵 집합. 목록만 준 JSON은 슬러그로 읽는다."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return {str(x) for x in data}, set()
    return (
        {str(x) for x in (data.get("slugs") or [])},
        {str(x) for x in (data.get("video_ids") or [])},
    )


def load_handoff_songs(path: Path, corpus: list[Song]) -> list[Song]:
    """인계 프로브 JSONL(video_id·slug·match)을 영상 고정 곡 목록으로.

    받는 형식은 probe688 계열 산출물 — 한 줄에 ``{"video_id", "slug", "match", ...}``.
    슬러그 매칭이 된 행만 쓴다(``match``가 exact/substring). 같은 슬러그가 여러 영상에
    걸리면 첫 행만 남긴다 — 곡당 싱크 하나가 목표라 나머지는 중복이다.
    """
    by_slug = {s.slug: s for s in corpus}
    out: list[Song] = []
    seen_slugs: set[str] = set()
    seen_videos: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        slug = row.get("slug")
        video_id = row.get("video_id")
        if not slug or not video_id or row.get("match") not in ("exact", "substring"):
            continue
        base = by_slug.get(slug)
        if base is None or slug in seen_slugs or video_id in seen_videos:
            continue
        seen_slugs.add(slug)
        seen_videos.add(video_id)
        out.append(
            Song(slug=base.slug, ko=base.ko, ja=base.ja, pinned_video_id=str(video_id))
        )
    return out


# ── 처리 이력 (재실행 멱등) ────────────────────────────────────────

#: 다시 시도해도 같은 답이 나오는 스킵 사유 — ``--retry-skipped``로도 되돌리지 않는다
TERMINAL_SKIPS = frozenset(
    {"blocklisted", "already_ingested", "no_original_title", "title_too_short"}
)


@dataclass
class SongRecord:
    slug: str
    status: str  # ok | skipped | failed
    reason: str = ""
    video_id: str | None = None
    sync_id: str | None = None
    lyrics_source: str | None = None
    en_layer: str | None = None
    at: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "status": self.status,
            "reason": self.reason,
            "video_id": self.video_id,
            "sync_id": self.sync_id,
            "lyrics_source": self.lyrics_source,
            "en_layer": self.en_layer,
            "at": self.at,
            "evidence": self.evidence,
        }


class BulkState:
    """슬러그별 처리 결과를 담는 상태 파일.

    ``dry_runs``는 진단용으로만 쌓고 **스킵 판정에는 쓰지 않는다** — 총괄이 첫 청크를
    dry-run으로 검수한 뒤 같은 구간을 실제로 돌릴 때 전부 "이미 처리됨"으로 건너뛰면
    검수가 인제스트를 막아 버린다.
    """

    def __init__(self, path: Path, retry_skipped: bool = False) -> None:
        self.path = path
        self.retry_skipped = retry_skipped
        self.songs: dict[str, dict[str, Any]] = {}
        self.dry_runs: dict[str, dict[str, Any]] = {}
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                logger.warning("상태 파일을 읽지 못해 빈 상태로 시작합니다 - %s", path)
                data = {}
            self.songs = dict(data.get("songs") or {})
            self.dry_runs = dict(data.get("dry_runs") or {})

    def done_reason(self, slug: str) -> str | None:
        """이 슬러그를 건너뛸 사유. 다시 시도해야 하면 None.

        성공은 언제나 건너뛴다. 실패는 대개 일시적(잡 타임아웃·네트워크)이라 다시
        시도한다. 스킵은 기본적으로 건너뛰고, ``--retry-skipped``면 종료성 사유만 남긴다.
        """
        rec = self.songs.get(slug)
        if not rec:
            return None
        status = rec.get("status")
        if status == "ok":
            return "already_done"
        if status == "skipped":
            reason = str(rec.get("reason") or "")
            if self.retry_skipped and reason not in TERMINAL_SKIPS:
                return None
            return f"already_skipped:{reason}" if reason else "already_skipped"
        return None

    def record(self, rec: SongRecord) -> None:
        self.songs[rec.slug] = rec.as_dict()

    def record_dry_run(self, rec: SongRecord) -> None:
        self.dry_runs[rec.slug] = rec.as_dict()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "updated_at": _now_iso(),
            "songs": self.songs,
            "dry_runs": self.dry_runs,
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.path)  # 원자적 교체 — 중간에 죽어도 이력이 반쪽으로 남지 않는다


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── 기존 싱크 대조 ─────────────────────────────────────────────────


@dataclass
class ExistingCorpus:
    """이미 싱크가 있는 영상들. DB를 못 열면 비어 있고, 그때는 서버가 최종 방어선이다.

    ``POST /api/sync/generate``는 ``(video_id, lyrics_hash)``가 같으면 새 잡을 만들지 않고
    기존 싱크를 그대로 돌려주므로, 이 대조가 새는 것이 재생성 사고로 번지지는 않는다.
    """

    video_ids: set[str] = field(default_factory=set)
    titled: list[tuple[str, str]] = field(default_factory=list)  # (video_id, 저장된 제목)
    available: bool = True

    def has_video(self, video_id: str) -> bool:
        return video_id in self.video_ids

    def title_hit(self, title: str, min_score: float = 0.9) -> str | None:
        """제목이 기존 싱크와 사실상 같은 곡을 가리키면 그 video_id.

        ``title_match``의 정규화(대소문자·기호·잡토큰 제거)를 그대로 쓴다. 0.9는 "정확
        일치이거나 그에 준하는 포함"만 통과시키는 값이다 — 여기서 오탐하면 새 곡을
        영구히 건너뛰므로 느슨하게 잡지 않는다.

        **제목만 본다(아티스트를 붙이지 않는다).** ``match_score``는 두 제목의 후보
        변형끼리 비교하고 길이비 0.5 미만을 버리므로, 「곡명」에 채널명까지 이어 붙이면
        길이비가 무너져 진짜 같은 곡도 매칭되지 않는다(실측: 0.44). 유튜브 제목은 관례상
        구분자로 「곡명 / 프로듀서」가 나뉘어 있어 제목 하나로도 곡명 조각이 잡힌다.

        구분자 없이 공백으로만 이어진 제목은 여기서 놓친다 — 그 경우는 영상 id 대조와
        서버의 ``(video_id, lyrics_hash)`` 중복 판정이 받아 낸다.
        """
        for video_id, entry_title in self.titled:
            hit = title_match.match_score(title, entry_title)
            if hit and hit[0] >= min_score:
                return video_id
        return None


def load_existing_corpus(db_path: Path) -> ExistingCorpus:
    if not db_path.exists():
        logger.warning(
            "DB를 찾을 수 없어 제목 대조 없이 진행합니다 (--db로 경로 지정) - %s", db_path
        )
        return ExistingCorpus(available=False)
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        video_ids = {r[0] for r in con.execute("SELECT DISTINCT video_id FROM sync_results")}
        titled = [
            (r[0], r[1])
            for r in con.execute(
                "SELECT video_id, title FROM sync_results WHERE title IS NOT NULL"
            )
            if r[1]
        ]
    except sqlite3.Error as e:
        logger.warning("DB 조회 실패, 제목 대조 없이 진행합니다 - %s", e)
        return ExistingCorpus(available=False)
    finally:
        con.close()
    return ExistingCorpus(video_ids=video_ids, titled=titled)


# ── 영상 후보 검증 ─────────────────────────────────────────────────

MIN_DURATION_SEC = 60.0
MAX_DURATION_SEC = 600.0

#: 이 점수를 넘겨야 채택한다. 제목 포함(1.0) 하나로는 못 넘는 값이다 — 제목 말고도
#: 독립된 근거(공식 채널/『Topic』/프로듀서 채널/공식 마커+조회수)가 하나는 있어야 한다.
#: 오매칭 1곡이 스킵 10곡보다 해롭다.
ACCEPT_SCORE = 2.5

#: 정규화 길이가 이보다 짧은 제목(「ロキ」 등)은 포함 매칭이 우연에 너무 약하다 —
#: 영상 제목의 곡명 자리와 **정확히** 같을 때만 인정한다
SHORT_TITLE_NORM_LEN = 3


def _ascii_word(*words: str) -> str:
    """ASCII 낱말 경계를 붙인 대안 묶음 — 「soft」 속 「ft」 같은 오매칭을 막는다."""
    return "|".join(f"{ASCII_LB}{w}{ASCII_RB}" for w in words)


# 원곡이 아닌 업로드 — 하나라도 걸리면 후보에서 배제한다. 커버·리믹스는 오디오가 달라
# 원곡 가사 정렬이 어긋나고, 替え歌(가사 바꿔 부르기)·空耳는 가사 자체가 다르다.
_NON_ORIGINAL_RE = re.compile(
    r"歌ってみた|唄ってみた|弾いてみた|叩いてみた|踊ってみた|弾き語り|替え歌|空耳|逆再生"
    r"|カラオケ|オルゴール|合唱|メドレー|マッシュアップ|リミックス|アレンジ|耳コピ"
    r"|試聴|予告|ライブ|生演奏"
    r"|커버|불러봤|불러본|노래방|메들리|리믹스"
    r"|" + _ascii_word(
        "cover", "covered", "karaoke", "instrumental", "inst", "off",
        "remix", "arrange", "arranged", "mashup", "medley", "nightcore",
        "acoustic", "piano", "live", "teaser", "preview", "mmd", "utau",
    ),
    re.IGNORECASE,
)

# 공식/원곡 업로드 마커 — 가점.
_OFFICIAL_RE = re.compile(
    r"オリジナル曲?|本家|ボカロオリジナル|公式"
    r"|music\s*video|official"
    r"|" + _ascii_word("mv", "pv"),
    re.IGNORECASE,
)

# 보컬로이드/합성음성 가수 표기 — 보카로 원곡임을 뒷받침하는 약한 가점.
_SYNTH_SINGER_RE = re.compile(
    r"初音ミク|鏡音リン|鏡音レン|巡音ルカ|MEIKO|KAITO|重音テト|結月ゆかり|可不|星界|裏命"
    r"|ボーカロイド|ボカロ|ミク"
    r"|hatsune\s*miku|megurine|kagamine|vocaloid|synthesizer\s*v|synthv|cevio"
    r"|" + _ascii_word("miku", "gumi", "flower", "ia"),
    re.IGNORECASE,
)

# 유튜브가 음반사 배급 음원에 자동 생성해 주는 채널 — 「アーティスト名 - Topic」.
# 이 채널의 업로드는 배급된 공식 음원 그 자체라 가장 믿을 만한 후보다.
_TOPIC_CHANNEL_RE = re.compile(r"(?:\s-\s*Topic|・トピック|\s-\s*トピック)\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class VideoCandidate:
    video_id: str
    title: str
    channel: str
    duration: float | None
    view_count: int | None


def candidate_from_json(entry: dict[str, Any]) -> VideoCandidate | None:
    """yt-dlp ``--dump-json`` 항목 하나를 후보로. id/제목이 없으면 None."""
    video_id = entry.get("id")
    title = entry.get("title")
    if not video_id or not title:
        return None
    duration = entry.get("duration")
    views = entry.get("view_count")
    return VideoCandidate(
        video_id=str(video_id),
        title=str(title),
        channel=str(entry.get("channel") or entry.get("uploader") or ""),
        duration=float(duration) if isinstance(duration, (int, float)) else None,
        view_count=int(views) if isinstance(views, (int, float)) else None,
    )


@dataclass
class CandidateScore:
    candidate: VideoCandidate
    score: float = 0.0
    rejected: str | None = None
    signals: list[str] = field(default_factory=list)

    def as_evidence(self) -> dict[str, Any]:
        return {
            "video_id": self.candidate.video_id,
            "title": self.candidate.title,
            "channel": self.candidate.channel,
            "duration": self.candidate.duration,
            "views": self.candidate.view_count,
            "score": round(self.score, 2),
            "rejected": self.rejected,
            "signals": self.signals,
        }


def video_title_head(title: str) -> str:
    """영상 제목에서 곡명이 있을 자리 — 장식을 걷고 구분자 앞 조각을 취한다.

    유튜브 보카로 관례상 곡명은 「곡명 / 프로듀서 feat. 가수」처럼 구분자 앞에 온다.
    miraheze 어댑터가 검색어를 만들 때 쓰는 규칙과 같은 함수를 쓴다.
    """
    stripped = miraheze.strip_decorations(title) or title
    return miraheze.strip_before_separator(stripped) or stripped.strip()


def score_candidate(
    song_title: str, cand: VideoCandidate, producer: str | None = None
) -> CandidateScore:
    """후보 하나를 채점한다 — 배제 사유가 있으면 ``rejected``에 담아 점수 없이 돌려준다.

    순수 함수다(네트워크 없음) — 규칙을 테스트로 고정하기 위해서다.
    """
    out = CandidateScore(candidate=cand)

    if cand.video_id in BLOCKED_VIDEO_IDS:
        out.rejected = "blocklisted"
        return out
    if cand.duration is None:
        # 길이를 모르면 곡인지 낚시인지 가릴 수 없다 — 확신 없으면 스킵
        out.rejected = "no_duration"
        return out
    if not (MIN_DURATION_SEC <= cand.duration <= MAX_DURATION_SEC):
        out.rejected = "duration_out_of_range"
        return out
    if _NON_ORIGINAL_RE.search(cand.title):
        out.rejected = "not_original_upload"
        return out

    norm_song = title_match.normalize_title(song_title)
    norm_title = title_match.normalize_title(cand.title)
    norm_head = title_match.normalize_title(video_title_head(cand.title))
    exact_head = bool(norm_song) and norm_head == norm_song
    contained = bool(norm_song) and norm_song in norm_title

    if len(norm_song) < SHORT_TITLE_NORM_LEN:
        if not exact_head:
            out.rejected = "short_title_needs_exact_match"
            return out
    elif not contained:
        out.rejected = "title_mismatch"
        return out

    out.score = 1.0
    out.signals.append("title_exact_head" if exact_head else "title_contained")
    if exact_head:
        out.score += 1.5

    if _TOPIC_CHANNEL_RE.search(cand.channel):
        out.score += 2.0
        out.signals.append("topic_channel")
    elif producer and title_match.normalize_title(producer) and (
        title_match.normalize_title(producer) in title_match.normalize_title(cand.channel)
    ):
        # miraheze 문서 제목의 「곡명/프로듀서」에서 얻은 이름 — 채널이 그 프로듀서면
        # 본인 업로드다(보카로 곡의 원본 업로더는 거의 언제나 프로듀서 본인이다)
        out.score += 2.0
        out.signals.append("producer_channel")

    if _OFFICIAL_RE.search(cand.title):
        out.score += 1.0
        out.signals.append("official_marker")
    if _SYNTH_SINGER_RE.search(cand.title) or _SYNTH_SINGER_RE.search(cand.channel):
        out.score += 0.5
        out.signals.append("synth_singer")

    if cand.view_count is not None:
        if cand.view_count >= 100_000:
            out.score += 0.5
            out.signals.append("views_100k")
        elif cand.view_count >= 10_000:
            out.score += 0.25
            out.signals.append("views_10k")

    if out.score < ACCEPT_SCORE:
        out.rejected = "below_accept_score"
    return out


def pick_video(
    song_title: str, entries: list[dict[str, Any]], producer: str | None = None
) -> tuple[CandidateScore | None, list[dict[str, Any]]]:
    """(채택 후보 또는 None, 후보 전원의 판정 근거). 동점이면 조회수 많은 쪽을 쓴다."""
    scored = [
        score_candidate(song_title, cand, producer)
        for cand in (candidate_from_json(e) for e in entries)
        if cand is not None
    ]
    accepted = [s for s in scored if s.rejected is None]
    accepted.sort(key=lambda s: (-s.score, -(s.candidate.view_count or 0)))
    evidence = [s.as_evidence() for s in scored]
    return (accepted[0] if accepted else None), evidence


# ── yt-dlp 검색 ────────────────────────────────────────────────────

# 403/429/봇 확인 요구 — 곡 탓이 아니라 이 서버 출구·쿠키 탓인 실패. 문구로 가른다
# (yt-dlp는 종료 코드로 구분해 주지 않는다).
_YTDLP_BLOCK_RE = re.compile(
    r"http error 403|403:?\s*forbidden|http error 429|too many requests"
    r"|sign in to confirm|not a bot|blocked it in your country",
    re.IGNORECASE,
)


class YtdlpBlockedError(Exception):
    """유튜브가 이 서버의 요청을 막았다 — 다음 곡으로 넘어가도 같은 결과다."""


class YtdlpSearch:
    """``ytsearchN:`` 메타 조회. 다운로드는 하지 않는다(``--dump-json --no-download``).

    쿠키는 배포가 이미 쓰는 파일을 그대로 물려 준다 — 검색도 봇 확인에 걸리고, 실측상
    403은 쿠키/버전 문제였다.
    """

    def __init__(
        self,
        binary: str = "yt-dlp",
        count: int = 5,
        timeout_sec: float = 90.0,
        cookie_file: Path | None = None,
        runner: Callable[[list[str], float], subprocess.CompletedProcess[str]] | None = None,
    ) -> None:
        self.binary = binary
        self.count = count
        self.timeout_sec = timeout_sec
        self.cookie_file = cookie_file
        self._runner = runner or _run_command
        self.block_streak = 0

    def search(self, query: str) -> list[dict[str, Any]]:
        return self._dump_json(f"ytsearch{self.count}:{query}")

    def fetch_video(self, video_id: str) -> list[dict[str, Any]]:
        """영상 하나의 메타만 — 인계 모드에서 검색 대신 단일 후보를 만든다."""
        return self._dump_json(f"https://www.youtube.com/watch?v={video_id}")

    def _dump_json(self, target: str) -> list[dict[str, Any]]:
        cmd = [
            self.binary,
            target,
            "--dump-json",
            "--no-download",
            "--no-warnings",
            "--ignore-config",
            "--flat-playlist",
        ]
        if self.cookie_file:
            cmd += ["--cookies", str(self.cookie_file)]
        try:
            proc = self._runner(cmd, self.timeout_sec)
        except FileNotFoundError as e:
            raise SystemExit(f"yt-dlp를 실행할 수 없습니다 ({self.binary}): {e}") from e
        except subprocess.TimeoutExpired:
            logger.warning("yt-dlp 메타 조회 시간초과 - %s", target)
            return []

        stderr = proc.stderr or ""
        if _YTDLP_BLOCK_RE.search(stderr):
            self.block_streak += 1
            raise YtdlpBlockedError(stderr.strip().splitlines()[-1] if stderr.strip() else "blocked")

        entries: list[dict[str, Any]] = []
        for line in (proc.stdout or "").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed, dict):
                continue
            # 대개 영상 한 줄씩 나오지만, 플레이리스트 객체 하나로 나오는 판도 있다
            nested = parsed.get("entries")
            if isinstance(nested, list):
                entries.extend(e for e in nested if isinstance(e, dict))
            else:
                entries.append(parsed)
        if entries:
            self.block_streak = 0
        return entries


def _run_command(cmd: list[str], timeout_sec: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - 인자는 코드가 조립하고 셸을 거치지 않는다
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_sec,
        check=False,
    )


# ── 가사 확보 ──────────────────────────────────────────────────────

#: 원문 언어와 다른 문자가 이 비율을 넘으면 번역이 섞인 트랙으로 본다.
#: ``verify_track_body``는 «가나 >= 한글»이면 ja로 인정하므로 절반씩 섞인 이중 자막을
#: 통과시킨다 — 원문만 필요한 이 경로에는 그 여유가 없다.
MAX_FOREIGN_SCRIPT_RATIO = 0.15

_FOREIGN_SCRIPT_BY_LANG = {"ja": "hangul", "ko": "kana"}


@dataclass
class CaptionLyrics:
    lines: list[str]
    track_key: str
    track_lang: str
    evidence: dict[str, Any]


#: 이 코퍼스(vocaro.wikidot.com = 일본 보카로곡 한국어 번역 위키)의 기본 원문 언어.
#: 문자 구성이 다른 답을 주지 않을 때 여기로 물러난다.
CORPUS_DEFAULT_LANG = "ja"


def original_language(title: str) -> str:
    """위키 원제의 문자 구성으로 곡의 원문 언어를 판정 — 'ja' | 'ko'.

    영상 제목·채널명으로 보는 ``title_script_hint``보다 믿을 수 있다 — 유튜브 쪽 제목은
    번역·독음으로 올라와 있을 수 있다.

    ``body_language``의 답을 그대로 쓰지 않고 두 경우를 코퍼스 지식으로 덮는다:

    - **한자만 있는 제목**(「仮死化」 류)은 ``body_language``가 'zh'로 읽는다. 그러면 ja
      트랙이 언어 코드 불일치로 전부 걸러져 이 곡들의 자막 경로가 조용히 죽는다.
    - **라틴만 있는 제목**(「!mperfection」 류)은 None이다 — 판정 근거가 없다는 뜻이지
      일본어가 아니라는 뜻이 아니다.

    한글이 우세한 제목만 'ko'로 갈라진다(한국어 원곡). 잘못 'ja'로 봐도 대가는 작다 —
    본문 검사(:func:`caption_lyrics`)가 한 겹 더 있어 엉뚱한 트랙은 거기서 걸린다.
    """
    from everyric2.alignment.caption_anchors import script_counts

    lang = yc.body_language(script_counts(title))
    return lang if lang in ("ja", "ko") else CORPUS_DEFAULT_LANG


def caption_lyrics(
    video_id: str, expected_lang: str | None, probe_limit: int = 4
) -> tuple[CaptionLyrics | None, dict[str, Any]]:
    """수동 자막에서 **원문 언어 트랙만** 골라 (가사, 판정 근거)로. 없으면 (None, 근거).

    ASR 자막은 어느 경로로도 쓰지 않는다: 후보는 ``manual_track_keys``에 있는 키로만
    한정하고, 내려받을 때도 ``auto=False``만 쓴다.

    원문 트랙 판정은 세 겹이다 — ① 트랙 언어 코드가 원문 언어와 같은가, ② 받아 본
    **본문**의 문자 구성이 원문 언어와 맞는가(``verify_track_body``), ③ 원문 언어가
    아닌 문자가 섞여 있지 않은가(이중 자막 배제). 목록만 보고 고를 수 없다는 것이
    서버 자막 경로의 교훈이고, 여기서는 원문만 필요하므로 더 좁게 잡는다.

    근거는 실패해도 항상 돌려준다 — 왜 위키 원문으로 물러났는지가 상태 파일에 남아야
    사후에 오염 규모를 셀 수 있다.
    """
    evidence: dict[str, Any] = {"expected_lang": expected_lang, "tracks": []}
    try:
        info = yc.extract_caption_info(video_id)
    except yc.CaptionUnavailable as e:
        evidence["error"] = e.code
        logger.info("%s: 자막 목록 실패 (%s)", video_id, e.code)
        return None, evidence

    manual = set(yc.manual_track_keys(info))
    evidence["manual_tracks"] = sorted(manual)
    if not manual:
        evidence["error"] = "no_manual_captions"
        return None, evidence
    if expected_lang is None:
        # 원문 언어를 모르면 어떤 트랙이 원문인지 판정할 근거가 없다 — 위키 원문으로 간다
        evidence["error"] = "unknown_original_language"
        return None, evidence

    for key in yc.order_manual_tracks(info, expected_lang, probe_limit):
        if key not in manual or not yc.LANG_RE.match(key):
            continue
        probe: dict[str, Any] = {"track": key}
        evidence["tracks"].append(probe)
        if yc.base_lang(key) != expected_lang:
            probe["rejected"] = "lang_code_mismatch"
            continue
        try:
            raw = yc.download_track_lines(video_id, key, auto=False)
        except yc.CaptionUnavailable as e:
            probe["rejected"] = e.code
            continue

        lines = yc.clean_caption_lines(raw)
        probe["lines"] = len(lines)
        if len(lines) < yc.MIN_LYRIC_LINES:
            probe["rejected"] = "too_short"
            continue

        counts = yc.caption_script_counts(lines)
        probe["script"] = counts
        if yc.verify_track_body(expected_lang, counts) != expected_lang:
            probe["rejected"] = "body_mismatch"
            continue
        ratio = foreign_script_ratio(expected_lang, counts)
        probe["foreign_ratio"] = round(ratio, 3)
        if ratio > MAX_FOREIGN_SCRIPT_RATIO:
            probe["rejected"] = "mixed_script"
            continue

        probe["accepted"] = True
        return (
            CaptionLyrics(
                lines=lines, track_key=key, track_lang=expected_lang, evidence=evidence
            ),
            evidence,
        )

    evidence["error"] = "no_original_track"
    return None, evidence


def foreign_script_ratio(expected_lang: str, counts: dict[str, int]) -> float:
    """원문 언어가 아닌 CJK 문자의 비율 (한자는 일·중이 공유하므로 세지 않는다)."""
    foreign_key = _FOREIGN_SCRIPT_BY_LANG.get(expected_lang)
    if not foreign_key:
        return 0.0
    cjk = sum(counts.get(k, 0) for k in ("kana", "hangul", "han"))
    if cjk == 0:
        return 0.0
    return counts.get(foreign_key, 0) / cjk


# ── 서버 API ───────────────────────────────────────────────────────


class ApiError(Exception):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"HTTP {status}: {detail}")
        self.status = status
        self.detail = detail


class ApiClient:
    """서버 API 클라이언트. 로컬 파이프라인을 직접 부르지 않고 배포된 경로를 태운다.

    키는 이 배포가 모든 ``/api``에 요구하므로 조회에도 싣는다 (verify_regen.py와 같은 관례).

    **운영 작업 식별자를 함께 싣는다** (docs/user-quota-spec.md §8): 이 스크립트는 앞단
    게이트웨이를 우회해 서버를 직접 부르므로, 이용자 식별자가 없으면 §1의 fail-closed에
    걸려 400이다. 사람과 구분되는 고유 식별자(``ops:bulk_ingest``)를 써서 ① 통계가 사람
    예산에 섞이지 않고 ② 상한은 면제받되 **기록은 남는다**. 🔴 "어드민 키면 식별자 없이
    통과"로 풀지 않는다 — 그것은 면제라서 기록이 없는 상태를 운영 작업에 되살리는 것이다.
    """

    def __init__(self, base_url: str, api_key: str | None, timeout_sec: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_sec = timeout_sec

    def _request(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        req = urllib.request.Request(
            self.base_url + path, method="POST" if payload is not None else "GET"
        )
        req.add_header("Content-Type", "application/json")
        req.add_header(ACTOR_HEADER, OPS_ACTOR_BULK_INGEST)
        if self.api_key:
            req.add_header("x-api-key", self.api_key)
        body = json.dumps(payload, ensure_ascii=False).encode() if payload is not None else None
        try:
            with urllib.request.urlopen(req, body, timeout=self.timeout_sec) as resp:
                data = json.loads(resp.read() or b"{}")
        except urllib.error.HTTPError as e:
            raise ApiError(e.code, _http_error_detail(e)) from e
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            raise ApiError(0, str(e)) from e
        return data if isinstance(data, dict) else {}

    def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("/api/sync/generate", payload)

    def job(self, job_id: str) -> dict[str, Any]:
        return self._request(f"/api/job/{job_id}")

    def save_translation_layer(self, video_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request(f"/api/sync/{video_id}/translations", payload)

    def wait_job(
        self,
        job_id: str,
        timeout_sec: float,
        poll_sec: float,
        sleep: Callable[[float], None] = time.sleep,
    ) -> dict[str, Any]:
        """완료/실패까지 폴링. 상한을 넘기면 ``status='timeout'``으로 돌려준다."""
        deadline = time.monotonic() + timeout_sec
        last: dict[str, Any] = {}
        while time.monotonic() < deadline:
            try:
                last = self.job(job_id)
            except ApiError as e:
                # 잡 조회 실패는 서버 재기동 등 일시적일 수 있다 — 상한 안에서는 계속 본다
                logger.warning("잡 조회 실패, 재시도 - %s (%s)", job_id, e)
                sleep(poll_sec)
                continue
            if last.get("status") in ("completed", "failed"):
                return last
            sleep(poll_sec)
        return {**last, "status": "timeout"}


def _http_error_detail(e: urllib.error.HTTPError) -> str:
    try:
        body = json.loads(e.read() or b"{}")
    except (json.JSONDecodeError, OSError):
        return e.reason if isinstance(e.reason, str) else str(e.reason)
    detail = body.get("detail") if isinstance(body, dict) else None
    return str(detail) if detail else str(body)


# ── 곡 하나 처리 ───────────────────────────────────────────────────


@dataclass
class Runtime:
    """청크 하나를 도는 데 필요한 협력자 묶음 — 테스트가 통째로 갈아끼울 수 있게 모았다."""

    api: ApiClient
    search: YtdlpSearch
    state: BulkState
    existing: ExistingCorpus
    vocaro_fetcher: WikiFetcher
    miraheze_fetcher: WikiFetcher
    dry_run: bool = False
    job_timeout_sec: float = DEFAULT_JOB_TIMEOUT_SEC
    poll_sec: float = 5.0
    caption_probe_limit: int = 4
    skip_video_ids: set[str] = field(default_factory=set)
    sleep: Callable[[float], None] = time.sleep


def process_song(song: Song, rt: Runtime) -> SongRecord:
    """곡 하나를 끝까지 — 스킵/실패도 사유와 근거를 담은 기록으로 돌려준다.

    ``YtdlpBlockedError``만 밖으로 던진다(체인 중단 신호). 나머지 실패는 기록으로 접는다.
    """
    rec = SongRecord(slug=song.slug, status="skipped", at=_now_iso())
    if not song.ja:
        # 원제가 없으면 유튜브 검색어가 한국어 독음뿐이라 원곡을 찾을 수 없다
        rec.reason = "no_original_title"
        return rec
    if len(title_match.normalize_title(song.ja)) < 2:
        rec.reason = "title_too_short"
        return rec

    existing_video = rt.existing.title_hit(song.ja)
    if existing_video:
        rec.reason = "already_ingested"
        rec.video_id = existing_video
        rec.evidence["matched_by"] = "db_title"
        return rec

    # ① miraheze — 영어 번역과, 후보 검증에 쓸 프로듀서 이름을 여기서 얻는다.
    #    vocaro는 영상이 정해진 뒤에 받는다(아래) — 스킵될 곡에 위키 요청을 쓰지 않는다.
    en = miraheze.lookup(song.ja, rt.miraheze_fetcher)
    producer = miraheze.producer_from_page_title(en.page_title) if en else None
    rec.evidence["wiki"] = {
        "miraheze_page": en.page_title if en else None,
        "miraheze_translation": bool(en and en.has_translation),
        "producer": producer,
    }

    # ② 영상 해석 — 인계 곡은 정해진 영상 하나를 단일 후보로 같은 채점에 태운다
    if song.pinned_video_id:
        entries = rt.search.fetch_video(song.pinned_video_id)
        rec.evidence["pinned_video"] = song.pinned_video_id
    else:
        entries = rt.search.search(song.ja)
    chosen, evidence = pick_video(song.ja, entries, producer)
    rec.evidence["candidates"] = evidence
    if chosen is None:
        rec.reason = "no_confident_video" if entries else "no_search_result"
        return rec

    video_id = chosen.candidate.video_id
    rec.video_id = video_id
    rec.evidence["chosen"] = chosen.as_evidence()
    if video_id in BLOCKED_VIDEO_IDS or video_id in rt.skip_video_ids:
        rec.reason = "blocklisted"
        return rec
    if rt.existing.has_video(video_id):
        rec.reason = "already_ingested"
        rec.evidence["matched_by"] = "db_video_id"
        return rec

    # ③ vocaro — 발음(한글 독음)과 한국어 번역, 그리고 자막이 없을 때의 원문 폴백
    wiki = vocaro.fetch_song(song.slug, rt.vocaro_fetcher)
    rec.evidence["wiki"].update(
        {
            "vocaro_lines": len(wiki.lines) if wiki else 0,
            "vocaro_line_meta": len(wiki.line_meta()) if wiki else 0,
        }
    )

    # ④ 가사 — 원문 언어 수동 자막이 첫째, 위키 원문이 폴백
    expected_lang = original_language(song.ja)
    caption, caption_evidence = caption_lyrics(video_id, expected_lang, rt.caption_probe_limit)
    rec.evidence["caption"] = caption_evidence
    if caption is not None:
        lyrics_lines = caption.lines
        rec.lyrics_source = "caption"
    else:
        if wiki and wiki.lines:
            lyrics_lines = [ln.text for ln in wiki.lines if ln.text]
            rec.lyrics_source = "vocaro"
        elif en and en.lines:
            lyrics_lines = [ln.text for ln in en.lines if ln.text]
            rec.lyrics_source = "miraheze"
        else:
            rec.reason = "no_lyrics"
            return rec

    if len(lyrics_lines) < yc.MIN_LYRIC_LINES:
        rec.reason = "no_lyrics"
        rec.evidence["lyrics_lines"] = len(lyrics_lines)
        return rec
    rec.evidence["lyrics_lines"] = len(lyrics_lines)

    # 가사 소스와 무관한 혼합 스크립트 최종 검사 — 자막 경로는 트랙 단위로 이미 거르지만,
    # **위키 폴백**은 행수가 어긋난 가사 표를 파서가 «전부 원문»으로 살리는 방어적 폴백
    # (sources/vocaro.py parse_song_page) 탓에 원문·독음·번역 3줄이 통째로 가사가 될 수
    # 있다 — 실사고 2026-07-28: DAr03V5IIeQ(149줄)·ACuO7MoDdgc(52줄), 야간 544곡 중 2곡.
    # 그 표시는 확장에서 유효한 폴백이지만(없는 것보다 낫다) 인제스트 입력으로는 오염이다.
    counts = yc.caption_script_counts(lyrics_lines)
    ratio = foreign_script_ratio(expected_lang, counts)
    if ratio > MAX_FOREIGN_SCRIPT_RATIO:
        rec.reason = "mixed_script_lyrics"
        rec.evidence["lyrics_foreign_ratio"] = round(ratio, 3)
        return rec

    if rt.dry_run:
        rec.reason = "dry_run"
        return rec

    # ⑤ 제출
    payload: dict[str, Any] = {
        "video_id": video_id,
        "lyrics": "\n".join(lyrics_lines),
        "language": expected_lang,
        "target_lang": "ko",
        "line_meta_lang": "ko",
        "title": chosen.candidate.title[:256],
        # 채널명을 아티스트로 싣는다 — 커버 링크 후보 탐색의 단서이고, 다음 청크의
        # 중복 대조(ExistingCorpus.title_hit)가 볼 수 있는 유일한 제목 기록이다
        "artist": chosen.candidate.channel[:128] or None,
    }
    if wiki:
        line_meta = wiki.line_meta()
        if line_meta:
            payload["line_meta"] = line_meta
        payload["attribution"] = wiki.attribution()

    try:
        resp = rt.api.generate(payload)
    except ApiError as e:
        rec.status = "failed"
        rec.reason = f"generate_error:{e.status}"
        rec.evidence["error"] = e.detail
        return rec

    job_id = resp.get("job_id")
    status = resp.get("status")
    if not job_id:
        rec.status = "failed"
        rec.reason = "generate_no_job"
        rec.evidence["response"] = resp
        return rec

    if status == "completed":
        # 같은 (영상, 가사)의 싱크가 이미 있다 — 서버가 잡을 만들지 않고 그것을 돌려준 것.
        # 이때 job_id는 잡이 아니라 싱크 id라 폴링하면 404다.
        rec.status = "ok"
        rec.reason = "existing_sync"
        rec.sync_id = str(job_id)
    else:
        done = rt.api.wait_job(job_id, rt.job_timeout_sec, rt.poll_sec, rt.sleep)
        job_status = done.get("status")
        if job_status != "completed":
            rec.status = "failed"
            rec.reason = "job_timeout" if job_status == "timeout" else f"job_{job_status}"
            rec.evidence["job"] = {"job_id": job_id, "error": done.get("error")}
            return rec
        rec.status = "ok"
        rec.reason = ""
        rec.evidence["job"] = {"job_id": job_id}

    # ⑥ 영어 번역 레이어 — 서버의 순서 정렬 매처가 줄 나눔 차이를 흡수한다
    rec.en_layer = save_en_layer(rt.api, video_id, en)
    return rec


def save_en_layer(api: ApiClient, video_id: str, en: miraheze.MirahezeSong | None) -> str:
    """miraheze 영어 번역을 번역 레이어로 저장. 결과를 짧은 사유 문자열로 돌려준다.

    실패해도 곡 자체는 성공이다 — ko 싱크는 이미 저장돼 있고, en은 나중에 다시 붙일 수
    있는 덧layer다. 특히 422(일치율 50% 미달)는 위키 줄 나눔이 이 영상 가사와 다르다는
    뜻이라 재시도로 나아지지 않는다.
    """
    if en is None or not en.has_translation:
        return "no_source"
    lines = en.translation_lines()
    if not lines:
        return "no_translation_lines"
    payload = {
        "target_lang": "en",
        "origin": "wiki",
        "lines": lines[:MAX_TRANSLATION_LAYER_LINES],
        "attribution": en.attribution(),
    }
    try:
        resp = api.save_translation_layer(video_id, payload)
    except ApiError as e:
        if e.status == 422:
            return "low_match"
        return f"error:{e.status}"
    if not resp.get("saved"):
        return "refused"
    return f"saved:{resp.get('matched')}/{resp.get('total')}"


# ── 청크 실행 ──────────────────────────────────────────────────────


def run_chunk(songs: list[Song], rt: Runtime, sleep_between_sec: float = 6.0) -> dict[str, Any]:
    """곡 목록을 순차로(동시 잡 1) 처리하고 요약을 만든다.

    개별 곡의 예외는 그 곡의 실패로 접는다 — 한 곡의 사고가 청크 전체를 죽이면 밤새
    돈 결과를 전부 잃는다. yt-dlp 차단만 예외로, 연속 임계에 닿으면 체인을 세운다.
    """
    started = time.monotonic()
    records: list[SongRecord] = []
    aborted: str | None = None

    for i, song in enumerate(songs):
        done_reason = rt.state.done_reason(song.slug)
        if done_reason and not rt.dry_run:
            records.append(
                SongRecord(slug=song.slug, status="skipped", reason=done_reason, at=_now_iso())
            )
            continue

        try:
            rec = process_song(song, rt)
        except YtdlpBlockedError as e:
            rec = SongRecord(
                slug=song.slug,
                status="failed",
                reason="ytdlp_blocked",
                at=_now_iso(),
                evidence={"error": str(e), "streak": rt.search.block_streak},
            )
            records.append(rec)
            _persist(rt, rec)
            if rt.search.block_streak >= YTDLP_BLOCK_ABORT_STREAK:
                aborted = "ytdlp_blocked"
                logger.error(
                    "yt-dlp가 %d회 연속 차단됐습니다 — 체인을 중단합니다. "
                    "쿠키 갱신이나 yt-dlp 업데이트가 필요합니다.",
                    rt.search.block_streak,
                )
                break
            # 차단은 곡 탓이 아니라 이 서버 탓이다 — 다음 곡으로 넘어가기 전에 더 쉰다
            rt.sleep(sleep_between_sec * 2)
            continue
        except Exception as e:  # noqa: BLE001 - 곡 하나의 사고로 밤샘 청크를 잃지 않는다
            logger.exception("%s: 처리 중 예외", song.slug)
            rec = SongRecord(
                slug=song.slug,
                status="failed",
                reason="exception",
                at=_now_iso(),
                evidence={"error": f"{type(e).__name__}: {e}"},
            )

        records.append(rec)
        _persist(rt, rec)
        logger.info(
            "%s (%s): %s%s%s",
            song.slug,
            song.ja or song.ko,
            rec.status,
            f" [{rec.reason}]" if rec.reason else "",
            f" {rec.video_id}" if rec.video_id else "",
        )
        if i + 1 < len(songs) and sleep_between_sec > 0:
            rt.sleep(sleep_between_sec)

    rt.state.save()
    return summarize(records, aborted, time.monotonic() - started)


def _persist(rt: Runtime, rec: SongRecord) -> None:
    """곡마다 즉시 저장한다 — 중간에 죽어도 앞서 한 일을 잃지 않는다."""
    if rt.dry_run:
        rt.state.record_dry_run(rec)
    else:
        rt.state.record(rec)
    rt.state.save()


def summarize(
    records: list[SongRecord], aborted: str | None, elapsed_sec: float
) -> dict[str, Any]:
    skipped = Counter(r.reason or "unknown" for r in records if r.status == "skipped")
    failed = Counter(r.reason or "unknown" for r in records if r.status == "failed")
    return {
        "processed": len(records),
        "ok": sum(1 for r in records if r.status == "ok"),
        "skipped": dict(sorted(skipped.items())),
        "skipped_total": sum(skipped.values()),
        "failed": dict(sorted(failed.items())),
        "failed_total": sum(failed.values()),
        "aborted": aborted,
        "elapsed_sec": round(elapsed_sec, 1),
        "songs": [r.as_dict() for r in records],
    }


# ── 진입점 ─────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--limit", type=int, default=25, help="이 청크에서 처리할 곡 수 (기본 25)")
    ap.add_argument("--offset", type=int, default=0, help="코퍼스에서 건너뛸 곡 수")
    ap.add_argument("--corpus", default=DEFAULT_CORPUS_PATH)
    ap.add_argument(
        "--handoff-file",
        help="인계 프로브 JSONL(video_id·slug·match) — 코퍼스 순회 대신 이 목록의 "
        "슬러그 매칭 곡만, 영상을 고정해 처리한다",
    )
    ap.add_argument("--skip-file", help="슬러그/영상 id 스킵 목록 JSON")
    ap.add_argument("--state-file", default=DEFAULT_STATE_PATH)
    ap.add_argument("--db", default=DEFAULT_DB_PATH, help="중복 대조용 읽기 전용 DB")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="영상 해석·가사 확보까지만 하고 제출하지 않는다 (상태 파일의 dry_runs에만 기록)",
    )
    ap.add_argument("--sleep", type=float, default=6.0, help="곡 사이 대기(초)")
    ap.add_argument("--wiki-interval", type=float, default=1.0, help="위키 요청 최소 간격(초)")
    ap.add_argument("--job-timeout", type=float, default=DEFAULT_JOB_TIMEOUT_SEC)
    ap.add_argument("--poll", type=float, default=5.0, help="잡 상태 폴링 간격(초)")
    ap.add_argument("--search-count", type=int, default=5, help="ytsearchN의 N")
    ap.add_argument("--caption-probe-limit", type=int, default=4)
    ap.add_argument("--yt-dlp", default=os.environ.get("YT_DLP_BIN", "yt-dlp"))
    ap.add_argument(
        "--retry-skipped",
        action="store_true",
        help="지난 실행에서 스킵한 곡도 다시 시도한다 (종료성 사유는 제외)",
    )
    ap.add_argument("--summary-out", help="요약 JSON을 이 파일에도 쓴다")
    ap.add_argument("--verbose", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"코퍼스를 찾을 수 없습니다: {corpus_path}", file=sys.stderr)
        return 2
    api_key = os.environ.get("EVERYRIC_SERVER_ADMIN_API_KEY") or None
    if not api_key and not args.dry_run:
        print(
            "EVERYRIC_SERVER_ADMIN_API_KEY가 없습니다 — 이 배포는 모든 /api에 키를 요구합니다.",
            file=sys.stderr,
        )
        return 2

    songs = load_corpus(corpus_path)
    if args.handoff_file:
        songs = load_handoff_songs(Path(args.handoff_file), songs)
    skip_slugs: set[str] = set()
    skip_videos: set[str] = set()
    if args.skip_file:
        skip_slugs, skip_videos = load_skip_file(Path(args.skip_file))

    chunk = [
        s for s in songs[args.offset : args.offset + max(0, args.limit)] if s.slug not in skip_slugs
    ]
    logger.info(
        "%s %d곡 중 %d..%d 구간 %d곡 처리%s",
        "인계 목록" if args.handoff_file else "코퍼스",
        len(songs),
        args.offset,
        args.offset + args.limit,
        len(chunk),
        " (dry-run)" if args.dry_run else "",
    )

    rt = Runtime(
        api=ApiClient(args.base_url, api_key),
        search=YtdlpSearch(
            binary=args.yt_dlp, count=args.search_count, cookie_file=_cookie_file()
        ),
        state=BulkState(Path(args.state_file), retry_skipped=args.retry_skipped),
        existing=load_existing_corpus(Path(args.db)),
        vocaro_fetcher=WikiFetcher(min_interval_sec=args.wiki_interval),
        miraheze_fetcher=WikiFetcher(min_interval_sec=args.wiki_interval),
        dry_run=args.dry_run,
        job_timeout_sec=args.job_timeout,
        poll_sec=args.poll,
        caption_probe_limit=args.caption_probe_limit,
        skip_video_ids=skip_videos | BLOCKED_VIDEO_IDS,
    )

    summary = run_chunk(chunk, rt, sleep_between_sec=args.sleep)
    summary["chunk"] = {"offset": args.offset, "limit": args.limit, "corpus": len(songs)}
    text = json.dumps(summary, ensure_ascii=False, indent=1)
    print(text)
    if args.summary_out:
        out = Path(args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    return 3 if summary.get("aborted") else 0


def _cookie_file() -> Path | None:
    """배포가 쓰는 쿠키 파일 — 검색도 봇 확인에 걸리므로 같은 것을 물려 준다."""
    try:
        from everyric2.config.paths import cookies_read_path

        path = cookies_read_path()
    except Exception:  # noqa: BLE001 - 쿠키 없이도 검색은 대개 된다
        return None
    return path if path.exists() else None


if __name__ == "__main__":
    raise SystemExit(main())
