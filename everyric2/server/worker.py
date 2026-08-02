import asyncio
import hashlib
import logging
import math
import re
import shutil
import statistics
import subprocess
import tempfile
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# 잡별 라인 메타(발음/번역) 임시 저장소 — BackgroundTasks가 같은 프로세스에서 돌므로
# 인메모리로 충분하다 (프로세스가 죽으면 잡 실행 자체가 사라지므로 내구성 손해도 없음).
#
# **키의 존재 여부가 "도착 확정" 신호를 겸한다**: 키가 없으면 아직 안 온 것이고, 빈
# 리스트는 "붙일 메타가 없음이 확정"이다. 병렬 경로(_wait_for_line_meta)가 이 구분으로
# 무한 대기 없이 진행 여부를 정한다 — 별도 플래그 집합을 두지 않아 정리 지점이 늘지 않는다.
#
# **갱신은 단조적이다**: 값이 있는 메타가 이미 들어와 있으면 나중에 온 빈 리스트로 지우지
# 않는다 (stash_line_meta 참고). 빈 리스트의 "확정 신호" 계약은 그대로 유지되고, 비우는
# 방향만 막힌다.
_PENDING_LINE_META: dict[str, list[dict[str, Any]]] = {}
# line_meta에 실린 **번역의 언어**. 위 dict의 값에 합치지 않고 나란히 둔다 — 그 값은
# 원격 워커 claim 응답(api/worker._peek_line_meta → 응답의 line_meta 필드)에 그대로 실려
# 나가는 와이어 포맷이고, 테스트 여러 파일이 그것이 라인 dict의 리스트임을 직접 단언한다.
#
# **요청자의 언어(Job.target_lang)와 다른 값이다.** 가사 출처가 번역까지 들고 오면
# (vocaro=한국어) 요청자가 영어권이어도 세그에 실리는 번역은 한국어다. 레이어 언어와
# legacy 슬롯 유지 여부는 전부 이 값으로 정한다 (resolve_layer_lang 참고).
# 값이 없으면 "ko" — lang을 안 싣는 구버전 호출부의 기존 동작이다.
_PENDING_LINE_META_LANG: dict[str, str] = {}
# 강제 재생성 잡 — 동일 (audio_hash, lyrics_hash) 재사용을 건너뛰고 정렬을 다시 돌린다
_PENDING_FORCE: set[str] = set()


def _norm_lang(lang: str | None) -> str:
    """언어 코드 정규화 — 비었으면 "ko"(기존 동작이 기본값)."""
    return (lang or "ko").strip() or "ko"


def peek_line_meta_lang(job_id: str) -> str:
    """스태시된 line_meta 번역의 언어 조회 (pop 없음). 안 실렸으면 "ko"."""
    return _PENDING_LINE_META_LANG.get(job_id) or "ko"


def stash_line_meta(job_id: str, line_meta: list[dict[str, Any]], lang: str = "ko") -> None:
    """잡의 라인 메타(발음/번역)를 스태시한다 — **이미 있는 값을 빈 리스트로 지우지 않는다.**

    ``lang``은 ``line_meta``의 **번역**이 무슨 언어인가다(발음은 언어와 무관한 결정론
    한글 독음이라 해당 없음). 기본 "ko"라 lang을 안 넘기는 기존 호출부는 동작이 같다.

    재현(무조건 덮어쓰던 예전 규칙): ``line_meta_pending=true``로 잡 생성 → 확장이 번역에
    성공해 35줄을 attach → 클라이언트 재시도 로직이 같은 잡에 ``line_meta: []``를 재전송 →
    워커가 발음·번역 없이 원문만 정렬. 이번 세션에 고친 "자막 경로 0줄"과 결과가 같다.

    빈 리스트 자체는 거부할 수 없다 — 그것이 "붙일 메타가 없음 확정"이라는 도착 신호이고
    (_PENDING_LINE_META 주석, _wait_for_line_meta), 거부하면 번역 실패 잡이 상한까지
    120초를 헛되게 기다린다. 그래서 **비어 있지 않은 것을 비우는 방향만** 막는다. 반대 방향
    (빈 것 → 값 있는 것)과 값 있는 것끼리의 갱신은 그대로 허용한다.
    """
    if not line_meta and _PENDING_LINE_META.get(job_id):
        logger.info(
            f"Job {job_id}: ignored an empty line_meta re-send; keeping the "
            f"{len(_PENDING_LINE_META[job_id])} line(s) already stashed"
        )
        return  # 언어도 그대로 둔다 — 지키기로 한 그 메타의 언어이므로
    _PENDING_LINE_META[job_id] = line_meta
    # 기본값(ko)은 **저장하지 않는다** — 원격 워커 프로세스(cli.py)도 이 함수로 자기 프로세스
    # 전역에 메타를 스태시하는데 그쪽 정리 경로는 이 dict의 존재를 모른다. 기본값을 안 넣으면
    # lang을 넘기지 않는 그 경로에 남는 항목이 아예 없다. 조회는 부재를 ko로 읽는다.
    # ko로 되쓰는 경우 이전 비ko 값을 지워야 stale 언어가 남지 않는다.
    if (norm := _norm_lang(lang)) == "ko":
        _PENDING_LINE_META_LANG.pop(job_id, None)
    else:
        _PENDING_LINE_META_LANG[job_id] = norm


# ── line_meta 지연 도착 (번역·독음을 다운로드·분리와 병렬로) ─────────

# line_meta 도착 대기 상한(초)과 폴링 간격. 확장이 번역·독음을 만드는 동안 서버가
# 다운로드·보컬 분리·f0를 먼저 돌리는 병렬 경로에서 정렬 진입 직전에 쓴다.
# **상한은 반드시 유한하다** — 확장이 번역에 실패해 아무것도 보내지 않아도 잡이 영구히
# 걸리면 안 되고, 상한 초과는 line_meta 없는 원문 정렬(기존 동작)로 조용히 떨어진다.
LINE_META_WAIT_SEC = 120.0
LINE_META_POLL_SEC = 0.25
# 대기 구간의 사용자 표시 단계명 (STAGE_WINDOWS에 창을 함께 등록해야 진행률이 튀지 않는다)
LINE_META_WAIT_STAGE = "번역 대기"

# line_meta를 나중에 붙일 잡 — 확장이 잡 생성 시 line_meta_pending으로 예고하고,
# _dispatch_job(인프로세스 경로)이 넣는다. _process_job_inner가 JobInput으로 캡처하며
# 즉시 비우므로(_PENDING_FORCE와 같은 관례) 남아 새는 항목이 없다.
_PENDING_META_WAIT: set[str] = set()


def stash_line_meta_wait(job_id: str) -> None:
    _PENDING_META_WAIT.add(job_id)


class JobCancelled(Exception):
    """대기 중 취소 요청 감지 — 코어가 잡아 취소 마감 경로로 넘긴다."""


def _wait_for_line_meta(job_id: str, timeout: float) -> list[dict[str, Any]] | None:
    """늦게 오는 line_meta를 상한을 두고 기다린다 (정렬 스레드에서 동기 호출).

    도착했으면 그 값을, 상한을 넘었으면 None(원문 정렬 폴백)을 돌려준다. 빈 리스트로
    도착한 경우(확장이 번역 실패를 알린 경우)도 즉시 None으로 진행한다.
    폴링 간격마다 취소 요청을 확인해 대기 중에도 취소가 먹는다 (JobCancelled).
    """
    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        if job_id in _CANCEL_REQUESTED:
            raise JobCancelled(job_id)
        meta = _PENDING_LINE_META.get(job_id)
        if meta is not None:
            return meta or None
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            logger.info(
                f"Job {job_id}: line_meta did not arrive within {timeout:.0f}s; "
                f"aligning on the original text"
            )
            return None
        time.sleep(min(LINE_META_POLL_SEC, remaining))


async def await_line_meta_arrival(job_id: str, timeout: float) -> bool:
    """line_meta 도착을 상한을 두고 비동기로 기다린다 (도착 True / 상한 초과 False).

    _wait_for_line_meta의 이벤트 루프 버전 — GPU 없는 API 전용 서버가 원격 워커 큐 진입을
    늦출 때 쓴다. 취소 요청이면 JobCancelled.
    """
    loop = asyncio.get_event_loop()
    deadline = loop.time() + max(0.0, timeout)
    while True:
        if job_id in _CANCEL_REQUESTED:
            raise JobCancelled(job_id)
        if job_id in _PENDING_LINE_META:
            return True
        remaining = deadline - loop.time()
        if remaining <= 0:
            return False
        await asyncio.sleep(min(LINE_META_POLL_SEC, remaining))


# 잡별 가사 출처 표기 (예: 보카로 가사 위키) — 완성된 싱크에 함께 저장된다
_PENDING_ATTRIBUTION: dict[str, dict[str, Any]] = {}


def stash_attribution(job_id: str, attribution: dict[str, Any]) -> None:
    _PENDING_ATTRIBUTION[job_id] = attribution


def stash_force(job_id: str) -> None:
    _PENDING_FORCE.add(job_id)


# 잡별 영상 제목/아티스트 — 완성된 싱크에 함께 저장돼 커버 링크 후보 탐색의 단서가 된다.
# Job 테이블에 컬럼을 더하지 않고 라인 메타·출처와 같은 인메모리 스태시 관례를 따른다
# (인프로세스·원격 워커 두 경로 모두 저장은 서버 프로세스에서 일어난다).
_PENDING_TITLE: dict[str, tuple[str | None, str | None]] = {}


def stash_title(job_id: str, title: str | None, artist: str | None = None) -> None:
    if title or artist:
        _PENDING_TITLE[job_id] = (title, artist)


def peek_title(job_id: str) -> tuple[str | None, str | None]:
    return _PENDING_TITLE.get(job_id, (None, None))


# 사용자 취소 요청 잡 — 취소 API가 넣고, 워커가 단계 경계에서 확인해 중단한다.
# 이미 도는 CTC/demucs 스레드 자체는 중단하지 못하므로 '경계 취소'다
# (대기열 슬롯 진입·다운로드 직후·정렬 시작 전·저장 전). 확인 시 집합에서 제거된다.
_CANCEL_REQUESTED: set[str] = set()


def request_cancel(job_id: str) -> None:
    _CANCEL_REQUESTED.add(job_id)


async def _consume_cancel(job_id: str) -> bool:
    """취소 요청이 있으면 잡을 실패(취소) 상태로 마감하고 True."""
    if job_id not in _CANCEL_REQUESTED:
        return False
    _CANCEL_REQUESTED.discard(job_id)
    from everyric2.server.db.connection import get_session
    from everyric2.server.db.repository import JobRepository

    async with get_session() as session:
        await JobRepository(session).update_status(
            job_id, "failed", error="요청으로 취소했어요", failure_kind="cancelled"
        )
    logger.info(f"Job {job_id} cancelled at a stage boundary")
    return True


# 진행 단계 → 전역 진행률 창 (lo, hi). 단계 내부의 실제 진행 콜백은 없으므로
# 창 안에서 시간 기반으로 차오르고, job API가 창 기준 단계별 퍼센트를 계산한다.
STAGE_WINDOWS: dict[str, tuple[int, int]] = {
    "다운로드": (10, 34),
    "캐시 확인": (34, 36),
    "보컬 분리": (36, 50),
    # 번역·독음(line_meta) 지연 도착 대기 — 보컬 분리 뒤, 전사 정렬 앞의 좁은 창.
    # 창을 등록하지 않으면 _stage_monitor의 기본 창(36,88)이 걸려 진행률이 대기 중에
    # 88까지 치솟는다. 전사 정렬의 시작(50) 바로 앞에 둬 진행률이 되돌아가지 않게 한다.
    LINE_META_WAIT_STAGE: (48, 50),
    "전사 정렬": (50, 72),
    "타이밍 보정": (72, 80),
    "멜로디 분석": (80, 88),
    "저장": (90, 100),
}

# 동시 처리 슬롯 — 정렬(demucs+CTC+멜로디)은 GPU/RAM을 크게 쓰므로 상한 없이 병렬로
# 돌리면 OOM이 난다. 초과분은 status=queued로 대기하고 확장이 "대기열"로 표시한다.
_JOB_SEMAPHORE: asyncio.Semaphore | None = None


def _job_slot() -> asyncio.Semaphore:
    global _JOB_SEMAPHORE
    if _JOB_SEMAPHORE is None:
        from everyric2.config.settings import get_settings

        _JOB_SEMAPHORE = asyncio.Semaphore(max(1, get_settings().server.max_concurrent_jobs))
    return _JOB_SEMAPHORE


def _normalize_line(s: str) -> str:
    """라인 매칭용 정규화 키 — 유니코드 호환 정규화(NFKC) + 서식문자 제거 + 공백 전면 제거.

    라인 메타(발음/번역)는 가사 원문과 별도 경로로 들어와 표기가 미세하게 어긋난다.
    공백만 접던 예전 규칙은 구두점 앞뒤 띄어쓰기 차이(``Are you ready ?`` vs
    ``Are you ready?``)나 전각/반각 차이(``！`` vs ``!``)를 다른 라인으로 취급해
    실측 6줄이 메타를 못 받았다. NFKC가 전각/반각·호환 문자를 접고, 공백을 전부
    지워 띄어쓰기 위치 차이를 흡수한다.
    **구두점 자체는 지우지 않는다** — 지우면 ``行く。``와 ``行く？``처럼 부호만 다른
    별개 라인이 같은 키로 뭉쳐 엉뚱한 메타가 붙는다(과잉 정규화 위험).
    """
    t = unicodedata.normalize("NFKC", s)
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Cf")  # ZWSP·BOM 등 서식문자
    return re.sub(r"\s+", "", t)


def _meta_has_value(m: dict[str, Any]) -> bool:
    """라인 메타가 실제로 쓸 값(발음 또는 번역)을 담고 있는지."""
    return bool((m.get("pronunciation") or "").strip() or (m.get("translation") or "").strip())


def _index_line_meta(line_meta: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    """line_meta를 정규화 텍스트 → 메타로 색인 — **값이 있는 첫 항목**을 채택한다.

    예전 규칙(첫 등장 무조건 채택)은 반복 후렴에 치명적이었다: 후렴 첫 등장의 발음/번역이
    비어 있으면 그 빈 항목이 키를 선점해, 값이 채워진 나머지 반복분까지 영구히 비었다
    (실측: 후렴 5회 반복 곡에서 5회 전부 누락). 값이 있는 항목이 나오면 빈 항목을 대체한다.
    값 있는 항목끼리는 여전히 첫 등장이 이긴다(안정적 선택).
    """
    by_text: dict[str, dict[str, Any]] = {}
    for m in line_meta or []:
        t = _normalize_line(m.get("text", "") or "")
        if not t:
            continue
        cur = by_text.get(t)
        if cur is None or (not _meta_has_value(cur) and _meta_has_value(m)):
            by_text[t] = m
    return by_text


def _referee_switched(seg: dict[str, Any]) -> bool:
    """오디오 심판이 이 세그먼트의 독음을 기본값과 다른 후보로 바꿨는가 (debug 메타 근거)."""
    ref = (seg.get("debug") or {}).get("referee") or {}
    chosen = ref.get("chosen")
    return bool(chosen) and chosen != ref.get("default")


def merge_line_meta(
    timestamps: list[dict[str, Any]],
    line_meta: list[dict[str, Any]],
    *,
    with_translation: bool = True,
) -> int:
    """세그먼트에 발음/번역을 라인 텍스트 매칭으로 병합. 병합된 세그먼트 수를 반환.

    ``with_translation=False``면 번역만 건너뛰고 발음은 그대로 병합한다 — 독음은 언어와
    무관한 결정론 한글이지만 번역은 요청자의 언어라, **여러 언어 사용자가 공유하는 행**에
    비ko 번역을 legacy 슬롯으로 밀어 넣으면 안 되는 경로(캐시 재사용)를 위한 문이다.
    """
    by_text = _index_line_meta(line_meta)

    merged = 0
    for seg in timestamps:
        m = by_text.get(_normalize_line(seg.get("text", "") or ""))
        if not m:
            continue
        # seg["pronunciation"]은 한글 전용 legacy 계약이다(정렬 입력과 같은 계약 —
        # _alignable_pron 재사용). line_meta의 발음이 romaji·가나(비ko 사용자의 line_meta,
        # 또는 캐시에 섞여 들어온 값)면 병합을 스킵한다 — 안 그러면 legacy 슬롯에 로마자가
        # 박혀 attach_pron_variants가 그 위에 pron["hangul"]=romaji를 얹고, 재생성 없이는
        # 안 지워지는 오염으로 모든 ko 사용자가 한글 칸에서 로마자를 보게 된다(감사 치명 #1
        # 잔여 — _alignable_pron 가드는 정렬 입력만 막았고 이 병합 경로가 남아 있었다).
        alignable_pronunciation = _alignable_pron(m.get("pronunciation"))
        if alignable_pronunciation:
            # 오디오 심판이 고른 독음은 line_meta 값으로 되돌리지 않는다. line_meta의 발음이
            # 바로 심판이 **오디오 점수로 이미 진 기본값**이고, pron_segments(음절 스팬)는
            # 이긴 후보 기준이라 표기만 갈아끼우면 음절 수가 어긋난다(캐시 재사용·늦은 메타
            # 병합 경로에서 발생). 재병합은 표시 메타를 채우는 경로일 뿐 판정 지점이 아니다.
            if not _referee_switched(seg):
                seg["pronunciation"] = alignable_pronunciation
            _attach_pron_segments(seg)
        if with_translation and m.get("translation"):
            seg["translation"] = m["translation"]
        # 캐시 재사용·늦은 병합으로 들어온 세그먼트도 표기별 발음을 갖게 한다. 직렬화에서
        # 이미 붙였으면 멱등 가드가 지킨다. 심판 개입 라인은 여기에 이긴 읽기의 토큰 열이
        # 없으므로 attach가 romaji를 스스로 생략한다(기본 읽기로 렌더하면 표기가 어긋난다).
        attach_pron_variants(seg)
        merged += 1
    return merged


def _attach_pron_segments(seg: dict[str, Any]) -> None:
    """발음 음절별 타이밍 산출 — 정렬된 글자 타이밍 + 모라 분해 + DP 매칭.

    전사 모델을 다시 돌리지 않는다 (기존 CTC 글자 타이밍을 모라 수로 내부 분할).
    품질 미달/실패 시 필드를 남기지 않아 클라이언트가 그라데이션으로 폴백한다.
    이미 pron_segments가 있으면(독음 정렬 경로 산출값 등) DP 근사로 덮어쓰지 않는다 —
    캐시 재사용 시 라인 메타 재병합이 정확한 정렬 스팬을 훼손하는 것을 막는다.
    """
    if seg.get("pron_segments"):
        return
    pron = seg.get("pronunciation")
    words = seg.get("words")
    if not pron or not words:
        seg.pop("pron_segments", None)
        return
    try:
        from everyric2.text.reading import pron_segments_for_line

        char_spans = [
            (w.get("word", ""), float(w.get("start", 0.0)), float(w.get("end", 0.0)))
            for w in words
        ]
        segments = pron_segments_for_line(char_spans, seg.get("text", "") or "", pron)
        if segments:
            seg["pron_segments"] = segments
        else:
            seg.pop("pron_segments", None)
    except Exception:
        logger.exception("pron_segments computation failed; falling back to gradient fill")
        seg.pop("pron_segments", None)


# 일본어 글자 — reading._is_japanese_char와 같은 범위(U+3040~U+30FF 가나, U+3400~U+9FFF 한자).
# 문자 클래스의 경계 글자는 그 코드포인트의 실제 글자다(편집 시 치환 주의).
_JA_CHAR_RE = re.compile("[぀-ヿ㐀-鿿]")

# 한글 완성형 음절(U+AC00~D7A3). ko_reading._decompose_hangul과 같은 범위다.
_HANGUL_CHAR_RE = re.compile("[가-힣]")

# 라틴 알파벳 — ja/한글이 없는 세그에서 "라틴 곡"으로 분기할지 판별한다.
_LATIN_CHAR_RE = re.compile("[A-Za-z]")


def _hiragana_to_katakana(text: str) -> str:
    """히라가나 → 가타카나 정규화. ``kana_hangul._to_hiragana``(가타카나→히라가나)의
    역방향 한 줄이다. 범위 밖 문자(장음부 ー·공백·라틴·부호)는 그대로 통과한다.

    ja 곡 발음 표기(``pron.kana``)는 ``pron_style``의 가나 런 환전이 항등이라(가나
    읽기가 이미 히라가나 기준) 여기서 표시만 가타카나로 정규화한다 — 발음 전사가
    원문(가나 섞인 일본어)과 시각적으로 구분되는 편이 읽기 편하다.
    """
    return "".join(chr(ord(ch) + 0x60) if "ぁ" <= ch <= "ゖ" else ch for ch in text)


def _ja_mora_segments(
    seg: dict[str, Any],
    text: str,
    render_tokens: list[str],
    space_after: list[bool],
    tokens: list | None,
) -> list[dict[str, Any]] | None:
    """ja 곡 세그의 모라 시각 계산 — romaji·kana 두 표기가 공유한다.

    ``mora_segments_for_line``이 만드는 시각은 모라 경계에서 나오고 표기와 무관하다
    (모라 하나가 romaji로도, 가타카나로도 렌더될 뿐 구간은 하나다) — ``render_tokens``만
    표기별로 갈아 끼운다. 그래서 심판(``tokens``)이 바꾼 읽기도 두 표기 모두 자동으로
    같이 따라온다(같은 ``text_to_moras(text, tokens=tokens)``가 재료라서).

    모라 수가 어긋나면(글자 스팬이 다른 읽기에서 왔거나 시각 환산이 실패) None —
    표시 문자열만 남기고 확장이 그라데이션으로 폴백하는 편이, 틀린 시각으로 엉뚱한
    글자를 점등시키는 것보다 낫다(``_attach_pron_segments``의 실패 규약과 같다).
    """
    words = seg.get("words")
    if not words:
        return None
    try:
        from everyric2.text.reading import mora_segments_for_line

        char_spans = [
            (w.get("word", ""), float(w.get("start", 0.0)), float(w.get("end", 0.0)))
            for w in words
        ]
        mora_times = mora_segments_for_line(char_spans, text, tokens=tokens)
    except Exception:
        logger.exception("ja mora timing failed; keeping the display string only")
        return None
    if not mora_times or len(mora_times) != len(render_tokens):
        return None

    segments: list[dict[str, Any]] = []
    for (_, start, end), token, space in zip(mora_times, render_tokens, space_after):
        entry: dict[str, Any] = {"text": token, "start": round(start, 3), "end": round(end, 3)}
        if space:
            entry["space"] = True
        segments.append(entry)
    return segments


def _ko_char_time(seg: dict[str, Any], text: str) -> dict[int, tuple[float, float]] | None:
    """ko 곡 세그의 ``words``를 글자 인덱스별 (start, end)로 정리한다.

    ``_full_coverage_words``가 만드는 ``words``는 라인의 **모든 글자를 공백까지 포함해**
    1:1로 덮는다(실측: "옛날 머나먼 그 어느 마을엔" → words 15개, 그중 공백 4개) —
    비한글 원문 글자별 스팬 없이 통짜로 붙는 kana/romaji의 «전멸» 원인이었다. 여기서
    공백뿐인 항목을 걸러낸 뒤 원문의 비공백 글자와 순서대로 짝짓는다. 개수가 안
    맞거나(다른 줄의 words가 섞였거나 정렬 토큰이 여러 글자를 한 덩이로 묶은 예외
    케이스) 글자 자체가 어긋나면 None — 표시만 남기고 확장이 그라데이션으로 폴백한다
    (``_attach_pron_segments``와 같은 실패 규약).
    """
    words = seg.get("words")
    if not words:
        return None
    non_blank_words = [w for w in words if (w.get("word") or "").strip()]
    non_space_idx = [i for i, ch in enumerate(text) if not ch.isspace()]
    if len(non_blank_words) != len(non_space_idx):
        return None

    char_time: dict[int, tuple[float, float]] = {}
    for w, idx in zip(non_blank_words, non_space_idx):
        if w.get("word") != text[idx]:
            return None
        try:
            char_time[idx] = (float(w.get("start", 0.0)), float(w.get("end", 0.0)))
        except (TypeError, ValueError):
            return None
    return char_time


def _ko_seg_gap_has_space(text: str, ce: int, next_cs: int) -> bool:
    """두 토큰의 글자 구간 사이에 공백이 있는가 — ko 분기 세그의 space 플래그 재료."""
    return any(ch.isspace() for ch in text[ce:next_cs])


def _kana_mora_segments_ko(seg: dict[str, Any], text: str) -> list[dict[str, Any]] | None:
    """ko 곡 세그: ``hangul_line_moras`` + ``_ko_char_time``으로 kana 모라 시각을 만든다.

    받침이 독립 가나가 되어 한 글자에 모라 2개가 붙으면(한→ハ+ン), 그 글자의 시간
    구간을 모라 개수만큼 균등 분할한다(``reading._build_mora_time``과 같은 방식).

    모라의 글자 구간이 한 글자보다 넓을 수 있다 — 라틴 런이 낱말 하나로 묶인
    모라(예: baby→(4,8), ``ko_reading.hangul_line_moras`` 참고)가 그렇다. 그래서
    시작 시각은 ``char_time[cs]``, 끝 시각은 ``char_time[ce-1]``에서 따로 가져와
    합친다(단일 글자 모라는 cs==ce-1이라 예전과 동일하게 동작한다).
    """
    char_time = _ko_char_time(seg, text)
    if char_time is None:
        return None

    try:
        from everyric2.text.ko_reading import hangul_line_moras

        moras = hangul_line_moras(text)
    except Exception:
        logger.exception("ko mora computation failed; keeping the display string only")
        return None
    if not moras:
        return None

    segments: list[dict[str, Any]] = []
    i, n = 0, len(moras)
    while i < n:
        j = i
        span = (moras[i][1], moras[i][2])
        while j + 1 < n and (moras[j + 1][1], moras[j + 1][2]) == span:
            j += 1
        cs, ce = span
        start_t = char_time.get(cs)
        end_t = char_time.get(ce - 1)
        if start_t is None or end_t is None:
            i = j + 1
            continue
        start, end = start_t[0], end_t[1]
        total = max(end - start, 0.0)
        count = j - i + 1
        for k in range(count):
            segments.append(
                {
                    "text": moras[i + k][0],
                    "start": round(start + total * k / count, 3),
                    "end": round(start + total * (k + 1) / count, 3),
                }
            )
        if j + 1 < n and _ko_seg_gap_has_space(text, ce, moras[j + 1][1]):
            segments[-1]["space"] = True
        i = j + 1

    if not segments:
        return None
    for idx in range(1, len(segments)):
        if segments[idx]["start"] < segments[idx - 1]["end"]:
            segments[idx]["start"] = segments[idx - 1]["end"]
        if segments[idx]["end"] < segments[idx]["start"]:
            segments[idx]["end"] = segments[idx]["start"]
    return segments


def _romaja_syllable_segments_ko(seg: dict[str, Any], text: str) -> list[dict[str, Any]] | None:
    """ko 곡 세그: ``hangul_line_romaja_syllables`` + ``_ko_char_time``으로 RR 시각을 만든다.

    kana와 달리 한 글자는 항상 로마자 음절 한 덩이다(받침이 독립 모라로 갈라지지
    않는다 — 한→han, 국→guk) — 그래서 글자 스팬을 그대로 옮겨 붙이면 되고, 균등
    분할이 필요 없다.
    """
    char_time = _ko_char_time(seg, text)
    if char_time is None:
        return None

    try:
        from everyric2.text.ko_reading import hangul_line_romaja_syllables

        syllables = hangul_line_romaja_syllables(text)
    except Exception:
        logger.exception("ko romaja timing failed; keeping the display string only")
        return None
    if not syllables:
        return None

    segments: list[dict[str, Any]] = []
    for idx, (token, cs, ce) in enumerate(syllables):
        t = char_time.get(cs)
        if t is None:
            continue
        segments.append({"text": token, "start": round(t[0], 3), "end": round(t[1], 3)})
        if idx + 1 < len(syllables) and _ko_seg_gap_has_space(text, ce, syllables[idx + 1][1]):
            segments[-1]["space"] = True
    if not segments:
        return None

    for idx in range(1, len(segments)):
        if segments[idx]["start"] < segments[idx - 1]["end"]:
            segments[idx]["start"] = segments[idx - 1]["end"]
        if segments[idx]["end"] < segments[idx]["start"]:
            segments[idx]["end"] = segments[idx]["start"]
    return segments


def _attach_ja_pron_variants(
    seg: dict[str, Any], text: str, *, referee_tokens: list | None = None
) -> None:
    """일본어 곡 세그: hangul(기존 ``pronunciation`` 값) + romaji + kana(가타카나 표시).

    ``pronunciation``이 없으면(비ko 사용자의 생성 요청 — 번역 API가 그 사용자 언어로
    번역만 만들고 발음은 ko 전용 결정론 경로라 line_meta에 한글 독음이 안 실린다)
    ``wiki_pronunciation(text)``로 서버가 직접 한글 독음을 만든다(감사 2차 E4). 정렬은
    이미 이 경우 원문 폴백이었으므로(``_alignable_pron``이 빈 값 취급) 표시만 새로
    생기는 것이라 안전하다 — legacy ``seg["pronunciation"]``에도 싣는다(구버전 확장
    호환). 형태소 분석기가 없거나 실패하면 조용히 스킵(pron dict 자체가 안 생긴다).

    ``referee_tokens``를 주면(오디오 심판이 이긴 후보의 토큰 열 —
    ``pron_style.candidate_token_sets``) 그 읽기로 romaji·kana 모라 시각을 만든다. 주지
    않았는데 심판이 이 줄의 독음을 바꿨으면(``_referee_switched``) romaji·kana를
    **아예 붙이지 않는다**: 기본 읽기로 렌더하면 화면의 한글 독음(심판이 오디오로 고른
    읽기)과 다른 낱말을 읽는 표기가 나란히 찍힌다. 표기가 없으면 확장이 한글로
    폴백하므로 손해는 표기뿐이다.
    """
    pron = (seg.get("pronunciation") or "").strip()
    if not pron:
        try:
            from everyric2.text.pron_style import wiki_pronunciation

            pron = wiki_pronunciation(text)
        except Exception:
            logger.exception("self-generated hangul pronunciation failed")
            return
        if not pron:
            return
        seg["pronunciation"] = pron

    seg["pron"] = {"hangul": pron}
    if referee_tokens is None and _referee_switched(seg):
        return

    try:
        from everyric2.text.pron_style import romaji_line

        rendered = romaji_line(text, tokens=referee_tokens)
    except Exception:
        logger.exception("romaji rendering failed; keeping the hangul-only pron dict")
        return
    if not rendered or not rendered[0]:
        return

    display, mora_tokens, space_after = rendered
    seg["pron"]["romaji"] = display
    romaji_segments = _ja_mora_segments(seg, text, mora_tokens, space_after, referee_tokens)
    if romaji_segments:
        seg.setdefault("pron_segs", {})["romaji"] = romaji_segments

    _attach_ja_kana_variant(seg, text, space_after, referee_tokens)

    pron_segs = seg.get("pron_segs") or {}
    # legacy pron_segments(독음 정렬의 실측 hangul 스팬)가 있으면 파생하지 않는다 —
    # 확장은 pron_segs.hangul을 legacy보다 우선하므로, 파생본이 실측을 가리게 된다
    if "hangul" not in pron_segs and not seg.get("pron_segments"):
        hangul_segments = _ja_hangul_segments_from_kana(seg)
        if hangul_segments:
            seg.setdefault("pron_segs", {})["hangul"] = hangul_segments


def _ja_hangul_segments_from_kana(seg: dict[str, Any]) -> list[dict[str, Any]] | None:
    """ja 곡 세그: 한글 음절 시각을 kana 모라 세그에서 파생 — 독음 정렬 없이도 hangul 카라오케.

    독음(ko) 정렬이 저신뢰로 밀려 ja 원문 정렬이 채택된 곡(합성보컬 posterior 바닥 —
    실측 2026-07-29 xvH0hNzMjhg)은 pron_segs가 romaji·kana뿐이라, 한국어 UI(기본 hangul
    표기)에서 음절 카라오케가 죽는다. hangul 표기는 같은 읽기(가나 모라 열)를 음절로
    합친 것이고, 받침으로 실현되는 모라(ン/ッ 등)는 그 음절에 2모라로 귀속된다
    (``hangul_line_moras``, 한→ハ+ン) — 그래서 모라 수가 kana 세그 수와 일치하면
    음절별 모라 그룹의 시간 스팬을 그대로 hangul 음절 시각으로 쓸 수 있다.

    불일치(장음 축약·표기 차이·라틴 음차 등)면 None — 틀린 시각으로 엉뚱한 글자를
    점등시키는 것보다 표기만 남기고 그라데이션 폴백이 낫다(``_ja_mora_segments``와
    같은 실패 규약).
    """
    hangul = ((seg.get("pron") or {}).get("hangul") or "").strip()
    kana_segs = (seg.get("pron_segs") or {}).get("kana")
    if not hangul or not kana_segs:
        return None
    try:
        from everyric2.text.ko_reading import hangul_line_moras

        moras = hangul_line_moras(hangul)
    except Exception:
        logger.exception("hangul mora decomposition failed; skipping hangul segs")
        return None
    if not moras or len(moras) != len(kana_segs):
        return None

    out: list[dict[str, Any]] = []
    i = 0
    while i < len(moras):
        _, cs, ce = moras[i]
        j = i
        # 같은 (char_start, char_end)를 공유하는 연속 모라 = 같은 한글 음절(받침 실현 포함)
        while j + 1 < len(moras) and moras[j + 1][1] == cs and moras[j + 1][2] == ce:
            j += 1
        entry: dict[str, Any] = {
            "text": hangul[cs:ce],
            "start": kana_segs[i]["start"],
            "end": kana_segs[j]["end"],
        }
        # 공백은 kana 모라 공백(문절)이 아니라 **hangul 표기 자신의** 공백 위치를 따른다 —
        # «표시=세그» 불변식(_rebuild == display)은 표기별로 성립해야 한다
        if ce < len(hangul) and hangul[ce].isspace():
            entry["space"] = True
        out.append(entry)
        i = j + 1
    return out or None


def _attach_ja_kana_variant(
    seg: dict[str, Any], text: str, space_after: list[bool], referee_tokens: list | None
) -> None:
    """ja 곡 세그: 가타카나 발음 표시(``pron.kana``) + 가능하면 모라 시각.

    표시=세그 단일 소스(감사 2차 M4, romaji와 같은 방식) — 카타카나 모라 열 하나를
    만들어 표시 문자열(공백 규칙까지 ``space_after``를 그대로 재사용)과 세그 둘 다에
    쓴다. 예전에는 표시를 ``wiki_pronunciation(text, script="kana")``(문절 띄어쓰기)로
    따로 만들었는데, 세그는 romaji와 같은 모라(토큰) 경계로 띄워 둘의 공백 위치가
    달랐다(NEKURA: 표시는 「アルバイトワ」로 붙는데 세그는 「ワ」 앞에 공백 플래그가
    있다) — 혼합 줄(한글이 섞인 줄)에서는 한 술 더 떠 ``wiki_pronunciation``이 한글
    구간을 렌더 못 해 표시에서만 빠질 위험까지 있었다. 이제는 세그 재료(카타카나
    모라 열)로 표시를 합성하므로 그 위험이 구조적으로 없다.

    카타카나 모라 열은 ``referee_tokens``가 있으면 그 토큰 열로, 없으면 ``romaji_line``이
    기본값일 때 쓰는 것과 **같은 phonetic=True 토큰화**로 ``text_to_moras``를 부른다 —
    ``text_to_moras(text)``의 무인자 기본값(phonetic=False)을 그냥 쓰면 は・を 같은
    조사가 표기 그대로(하·워)로 남아 romaji가 읽는 소리(wa·o)와 짝이 어긋난다(실측:
    NEKURA의 は가 로마자로는 "wa"인데 무인자 모라는 literal "ハ"를 낸다). 그래서
    기본값도 같은 phonetic 토큰화를 명시적으로 만들어 쓴다(``romaji_line``이 내부에서
    하는 것과 동일) — 한글이 섞인 줄의 모라(``reading.text_to_moras``의 M1 확장)도 이
    호출 하나로 같이 나온다.
    """
    try:
        from everyric2.text.ja_reading import tokenize_reading
        from everyric2.text.reading import text_to_moras

        mora_tokens_source = referee_tokens
        if mora_tokens_source is None:
            mora_tokens_source = tokenize_reading(text, phonetic=True, adopt_ruby=True)
        kana_tokens = [
            _hiragana_to_katakana(m.kana) for m in text_to_moras(text, tokens=mora_tokens_source)
        ]
    except Exception:
        logger.exception("kana mora tokens failed")
        return
    if not kana_tokens:
        return

    kana_display = "".join(
        tok + (" " if sp else "") for tok, sp in zip(kana_tokens, space_after)
    ).strip()
    if not kana_display:
        return
    seg["pron"]["kana"] = kana_display

    segments = _ja_mora_segments(seg, text, kana_tokens, space_after, referee_tokens)
    if segments:
        seg.setdefault("pron_segs", {})["kana"] = segments


def _ko_mixed_line_hangul(text: str) -> str | None:
    """ko 곡 혼합 줄(가나·한자가 섞인 줄)의 hangul 표시 — ja 런만 독음으로 환전한다.

    ko 곡은 원문=독음이라 순한글 줄엔 hangul 키를 안 만드는 게 공유 계약인데, 혼합
    줄(«사랑해 デス» 등)의 ja 부분을 원문 그대로 두면 ko 사용자가 그 글자를 못 읽는다
    (감사 2차 R2). 한글 부분은 원문 그대로 두고(이미 독음이다) ja 런만
    ``kana_hangul.finalize_pronunciation``(한자→가나→한글 체인, LLM 발음 필드 마감에
    쓰는 것과 같은 함수)으로 바꿔치기한다.

    ja 글자가 하나도 없으면 None — 호출부가 그때는 hangul 키를 아예 만들지 않는다
    (순한글 줄에 원문과 100% 같은 값을 또 저장하는 낭비 방지, 기존 설계 유지).

    공백은 ja 청크에 넣지 않는다 — ``finalize_pronunciation``이 끝에서
    ``" ".join(pron.split())``로 공백을 정규화하므로, 청크 선두 공백을 넣으면
    조용히 삼켜져 원문의 한글·ja 사이 띄어쓰기가 사라진다.
    """
    if not _JA_CHAR_RE.search(text):
        return None
    try:
        from everyric2.text.kana_hangul import finalize_pronunciation
    except Exception:
        logger.exception("ko mixed-line hangul rendering failed")
        return None

    out: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        if not buf:
            return
        chunk = "".join(buf)
        if _JA_CHAR_RE.search(chunk):
            try:
                out.append(finalize_pronunciation(chunk) or chunk)
            except Exception:
                logger.exception("ko mixed-line ja chunk rendering failed")
                out.append(chunk)
        else:
            out.append(chunk)
        buf.clear()

    for ch in text:
        if _HANGUL_CHAR_RE.match(ch) or ch.isspace():
            flush()
            out.append(ch)
        else:
            buf.append(ch)
    flush()
    return "".join(out)


def _attach_ko_pron_variants(seg: dict[str, Any], text: str) -> None:
    """한국어 곡 세그: 가타카나(일본어권)·RR 로마자(영어권) — 둘 다 결정론 생성.

    ``pronunciation``(독음) 필드가 없는 게 정상이다 — 원문 한글 자체가 표시이므로
    순한글 줄에는 "hangul" 표기 키를 만들지 않는다(공유 계약: 클라이언트는 script
    하나만 고른다). 다만 혼합 줄(ja 글자가 섞인 줄)은 예외다 — ``_ko_mixed_line_hangul``이
    ja 런만 독음으로 환전한 hangul 표시를 만든다(감사 2차 R2).

    두 표기(kana/romaji) 모두 시각 부착을 독립 시도한다 — 한쪽이 실패해도(예:
    words 불일치) 다른 쪽 타이밍은 살아남는다.
    """
    try:
        from everyric2.text.ko_reading import hangul_to_kana, hangul_to_romaja

        kana = hangul_to_kana(text)
        romaja = hangul_to_romaja(text)
    except Exception:
        logger.exception("ko pron rendering failed")
        return

    pron: dict[str, str] = {"kana": kana, "romaji": romaja}
    mixed_hangul = _ko_mixed_line_hangul(text)
    if mixed_hangul:
        pron["hangul"] = mixed_hangul
    seg["pron"] = pron

    kana_segments = _kana_mora_segments_ko(seg, text)
    if kana_segments:
        seg.setdefault("pron_segs", {})["kana"] = kana_segments

    romaja_segments = _romaja_syllable_segments_ko(seg, text)
    if romaja_segments:
        seg.setdefault("pron_segs", {})["romaji"] = romaja_segments


def _attach_latin_pron_variants(seg: dict[str, Any], text: str) -> None:
    """라틴(영어) 곡 세그: 일본어권 사용자를 위한 가나 발음만 붙인다.

    ``latin_hangul``의 느슨 음차를 거쳐 만든 결정론 근사라 글자 스팬을 신뢰할 근거가
    없다 — CTC 정렬 자체가 라틴 위에서 약하다는 것이 기존 실측이다(``latin_hangul``
    모듈 docstring). 그래서 ``pron_segs``는 붙이지 않고 표시 문자열만 남긴다.
    """
    try:
        from everyric2.text.ko_reading import latin_to_kana

        kana = latin_to_kana(text)
    except Exception:
        logger.exception("latin pron rendering failed")
        return

    seg["pron"] = {"kana": kana}


def attach_pron_variants(seg: dict[str, Any], *, referee_tokens: list | None = None) -> None:
    """세그먼트에 표기별 발음(``pron``)과 가능하면 모라 스팬(``pron_segs``)을 얹는다.

    기존 ``pronunciation``/``pron_segments``(한글, ja 곡 전용)는 손대지 않는다 — 구버전
    확장은 그 필드만 읽으므로 새 표기는 **추가 필드로만** 올라간다(공유 계약).

    곡 언어는 세그 원문 텍스트의 **문자 수 우세**로 판정한다(감사 2차 M2 — 예전엔
    "일본어 글자가 하나라도 있으면 ja"였는데, ko 곡의 혼합 줄(«사랑해 デス»처럼 일본어
    낱말이 섞인 줄)이 한글이 더 많은데도 ja 분기로 새서 그 줄만 이웃 줄과 다른 표기
    종류(kana/romaji 대신 hangul/romaji/kana)를 받았다. E4로 ja 분기가 자체 발음
    생성을 갖췄으니 "새면 발음이 빈다"는 문제는 없어졌지만, 표기 종류 불일치는
    우세 판정 없이는 안 없어진다):

    1. 일본어 글자(가나·한자, ``_JA_CHAR_RE``) 수가 한글(``_HANGUL_CHAR_RE``) 수
       이상이면 **ja 곡** — hangul(없으면 자체 생성)+romaji+kana.
    2. 그렇지 않고 한글이 있으면(즉 한글 수 > 일본어 수) **ko 곡** — 가타카나+RR
       로마자를 그 자리에서 결정론 생성한다(``pronunciation`` 필드 불필요 — 원문
       자체가 독음이다).
    3. 둘 다 없고 라틴 알파벳(``_LATIN_CHAR_RE``)이 있으면 **라틴 곡** — 일본어권
       사용자용 가나 근사만 표시로 붙인다.
    4. 셋 다 없으면(숫자·기호뿐인 줄 등) 아무것도 붙이지 않는다.

    멱등 — 이미 ``pron``이 있으면 아무것도 하지 않는다. 캐시 재사용·늦은 메타 병합이
    직렬화 때 만든(심판 판정을 반영한) 값을 덮지 않게 하는 가드다.
    """
    if seg.get("pron"):
        return
    text = seg.get("text") or ""
    if not text:
        return

    ja_n = len(_JA_CHAR_RE.findall(text))
    ko_n = len(_HANGUL_CHAR_RE.findall(text))
    if ja_n and ja_n >= ko_n:
        _attach_ja_pron_variants(seg, text, referee_tokens=referee_tokens)
    elif ko_n:
        _attach_ko_pron_variants(seg, text)
    elif _LATIN_CHAR_RE.search(text):
        _attach_latin_pron_variants(seg, text)


def _referee_token_set(text: str, chosen: str) -> list | None:
    """심판이 이긴 독음 문자열을 만든 토큰 열 — 다른 표기가 같은 읽기를 따르게 하는 다리.

    후보 문자열 목록은 ``_referee_candidates``가 쓰는 ``pronunciation_candidates``와 같은
    순서·같은 내용이다(둘 다 ``_pronunciation_candidates_with_tokens`` 하나를 쓴다). 심판이
    **바꾼** 줄에서만 부르므로 chosen은 결정론 후보 중 하나이고 거의 항상 찾힌다 — 못 찾으면
    (후보 상한을 8보다 크게 올린 설정 등) None을 돌려 기본 읽기로 조용히 떨어진다.
    """
    from everyric2.text.pron_style import candidate_token_sets

    rendered, token_sets = candidate_token_sets(text)
    for candidate, tokens in zip(rendered, token_sets):
        if candidate == chosen:
            return tokens
    return None


def job_target_lang(job: Any) -> str:
    """잡의 번역 대상 언어(요청자가 **원한** 언어). 컬럼이 없거나 비면 "ko".

    **레이어 언어·legacy 판정에 쓰지 마라** — 그 둘은 세그에 실제로 실린 번역의 언어로
    정한다(``resolve_layer_lang``). 요청자 언어는 조회 시 ``lang`` 파라미터로만 의미가 있고,
    여기서는 «원한 것과 받은 것이 다르다»를 진단 로그로 남기는 데만 쓴다.
    """
    return (getattr(job, "target_lang", None) or "ko").strip() or "ko"


def resolve_layer_lang(job: Any, job_id: str) -> str:
    """이 잡의 번역을 어느 언어로 기록할 것인가 = **세그에 실제로 실린 번역의 언어**.

    요청자 언어(``Job.target_lang``)로 정하면 안 된다. 실사용 사고: 영어 설정 사용자가
    vocaro(한국어 번역까지 들어 있는 위키) 가사로 생성하면 세그에 붙는 번역은 한국어인데,
    그것을 en 레이어에 기록하고 legacy 슬롯에서는 벗겨 버렸다 → ``lang=en`` 조회가
    한국어 번역을 en으로 내주고 ``translation_lang="en"``까지 붙어, 확장은 «내 언어 번역이
    있다»고 보고 영어를 영영 요청하지 않았다. 한국어 번역은 legacy에서도 사라져 ko
    사용자까지 잃었다.

    올바른 규칙은 «담긴 것에 맞는 라벨을 붙인다»다: 한국어 번역이면 ko 레이어에 넣고
    legacy 슬롯에도 남긴다(구버전·ko 사용자에게 유효). 그러면 ``lang=en`` 조회는 비어
    나가고(``translation_lang=None``) 확장이 영어 번역을 요청한다 — 원래 의도한 흐름이다.
    """
    meta_lang = peek_line_meta_lang(job_id)
    target = job_target_lang(job)
    if target != meta_lang:
        logger.info(
            f"Job {job_id}: line_meta translation is {meta_lang!r} but the requester asked for "
            f"{target!r}; recording the {meta_lang} layer only. The {target} translation is not "
            f"created here — a lang={target} lookup stays empty so the client requests it."
        )
    return meta_lang


def translation_layer_lines(items: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    """(원문, 번역) 쌍 목록 — ``TranslationLayer.lines``에 그대로 들어가는 형태.

    직렬화된 세그먼트와 line_meta 항목이 같은 ``text``/``translation`` 키를 쓰므로 두
    경로가 이 함수 하나를 공유한다. 번역이 빈 줄은 넣지 않는다(간주·공백 줄).
    """
    lines: list[dict[str, str]] = []
    for item in items or []:
        text = (item.get("text") or "").strip()
        translation = (item.get("translation") or "").strip()
        if text and translation:
            lines.append({"text": text, "translation": translation})
    return lines


def layer_origin(attribution: dict[str, Any] | None) -> str:
    """생성 시 레이어 origin 판정 — 생성 경로의 세그먼트 번역은 전부 line_meta(위키·수동·
    자막)에서 온다. 워커는 번역을 스스로 만들지 않으므로 여기서 "llm"이 나올 일은 없다
    ("llm"은 /api/translate persist 경로 전용). source_id(신형 attribution)가 정본이고,
    구형(vocaro url만 있던 시절)은 url 존재로, 자막 병합은 이름으로 근사한다."""
    if not attribution:
        return "manual"
    sid = (attribution.get("source_id") or "").strip()
    if sid in ("vocaro", "miraheze"):
        return "wiki"
    name = attribution.get("name") or ""
    if "자막" in name or "caption" in name.lower():
        return "caption"
    if attribution.get("url"):
        return "wiki"
    return "manual"


def peek_attribution(job_id: str) -> dict[str, Any] | None:
    """스태시된 출처 표기 조회 (pop 없음) — 원격 워커 결과 수신부(api/worker.py)가
    레이어 기록에 쓴다. peek_title과 같은 계약."""
    return _PENDING_ATTRIBUTION.get(job_id)


async def record_translation_layer(
    session: Any,
    video_id: str,
    segment_texts: list[str],
    lines: list[dict[str, str]],
    target_lang: str,
    *,
    origin: str = "llm",
    attribution: dict[str, Any] | None = None,
) -> bool:
    """생성 시 번역을 (video_id, 가사 지문, 언어) 레이어에 기록한다. 기록했으면 True.

    지문은 **세그먼트 원문 텍스트 전체**로 계산한다 — 조회(``GET /api/sync?lang=``)가 같은
    싱크의 세그먼트 텍스트로 지문을 만들어 찾으므로, 번역이 붙은 줄만으로 계산하면 저장은
    되는데 영영 못 찾는 레이어가 된다.

    실패해도 예외를 밖으로 던지지 않는다 — 레이어는 곁다리 기록이고, 이것 때문에 정렬을
    다 끝낸 잡이 failed로 마감되면 손해가 훨씬 크다.
    """
    if not lines or not segment_texts:
        return False
    try:
        from everyric2.server.db.repository import TranslationLayerRepository
        from everyric2.server.text_fingerprint import lines_fingerprint

        await TranslationLayerRepository(session).upsert_layer(
            video_id=video_id,
            fingerprint=lines_fingerprint(segment_texts),
            target_lang=target_lang,
            lines=lines,
            attribution=attribution,
            origin=origin,
        )
    except Exception:
        logger.exception(
            f"Translation layer record failed for {video_id} ({target_lang}); "
            "the sync itself is unaffected"
        )
        return False
    logger.info(
        f"Recorded {len(lines)} translated line(s) for {video_id} in the {target_lang} layer"
    )
    return True


def compute_audio_hash(file_path: Path) -> str:
    """확보한 오디오 파일의 md5 — 캐시 키(SyncResult.audio_hash, String(32))다.

    **파일 바이트 해시라 확보 경로에 의존한다** — 같은 영상이라도 미디어 캐시 경로(m4a
    스트림카피)와 yt-dlp 경로(wav 트랜스코드)는 다른 해시가 된다. 아래 `_acquire_audio`에
    이 비대칭을 왜 그냥 두는지(내용 기반 해시로 못 고치는 이유) 실측과 함께 적어 뒀다.
    """
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()


# 길이 프로브(ffprobe) 상한(초) — 헤더만 읽는 작업이라 정상이면 수십 ms다. 상한은 손상
# 파일/네트워크 경로에서 프로세스가 매달리는 것만 막는 안전판이고, 초과하면 None으로 떨어져
# 예전과 같은 "검사 생략" 동작이 된다 (과길이 검사는 있으면 좋은 가드지 필수 경로가 아니다).
_FFPROBE_TIMEOUT_SEC = 10.0


def _audio_duration_sec(file_path: str) -> float | None:
    """오디오 길이(초) — 헤더만 읽어 즉시 반환. 실패 시 None(상한 검사 생략).

    soundfile(libsndfile) 먼저, 실패하면 ffprobe로 폴백한다. **libsndfile은 m4a/AAC를 못
    읽는다** — 실측(libsndfile 1.2.2, ffmpeg AAC 5초 파일):
        sf.info('t.wav')  → OK
        sf.info('t.m4a')  → LibsndfileError: Error opening 't.m4a': Format not recognised.
    미디어 캐시 경로는 ``-acodec copy``로 **m4a**를 만들어 넘기므로(media_cache._run_ffmpeg),
    soundfile만 쓰던 예전 구현은 그 경로에서 항상 None을 돌려줬고 호출부가 ``if duration and``
    이라 **과길이 검사가 통째로 생략**됐다. 캐시 lookup이 ``duration_sec``를 안 주면
    프리플라이트(media_cache.prepare_cached_audio)도 건너뛰므로 상한이 완전히 사라져, 장시간
    영상이 GPU 슬롯을 점유할 수 있었다. ffprobe는 ffmpeg와 함께 설치되고 다운로더가 이미
    ffmpeg를 필수 의존성으로 검사하므로(downloader._check_dependencies) 새 의존성이 아니다.
    """
    try:
        import soundfile as sf

        info = sf.info(file_path)
        return float(info.frames) / float(info.samplerate or 1)
    except Exception:
        pass
    return _ffprobe_duration_sec(file_path)


def _ffprobe_duration_sec(file_path: str) -> float | None:
    """ffprobe로 컨테이너 길이(초)를 읽는다 — libsndfile이 못 읽는 포맷(m4a/AAC)용 폴백."""
    if not shutil.which("ffprobe"):
        return None
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                file_path,
            ],
            capture_output=True,
            timeout=_FFPROBE_TIMEOUT_SEC,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    try:
        # 스트림 길이를 못 아는 컨테이너는 'N/A'를 뱉는다 → float()이 실패해 None
        duration = float(proc.stdout.decode("utf-8", "replace").strip())
    except ValueError:
        return None
    return duration if duration > 0 else None


async def process_job(job_id: str) -> None:
    from everyric2.server.db.connection import get_session
    from everyric2.server.db.repository import JobRepository

    async with get_session() as session:
        job_repo = JobRepository(session)
        job = await job_repo.get_by_id(job_id)

        if not job:
            logger.error(f"Job not found: {job_id}")
            _PENDING_META_WAIT.discard(job_id)
            return

        # 슬롯이 다 차 있으면 대기열 — job API가 queue_position을 계산해 내려준다
        await job_repo.update_status(job_id, "queued", progress=0)

    slot = _job_slot()
    if slot.locked():
        logger.info(f"Job {job_id} waiting for a processing slot")
    async with slot:
        # 대기열에 있는 동안 취소된 잡은 슬롯을 잡자마자 놓아준다
        if await _consume_cancel(job_id):
            _PENDING_META_WAIT.discard(job_id)
            return
        await _process_job_inner(job_id, job)


async def _complete_from_cache_db(
    job_id: str, job, audio_hash: str, lyrics_hash_value: str
) -> bool:
    """(audio_hash, lyrics_hash)가 일치하는 기존 싱크가 있으면 정렬 없이 잡을 완료한다.

    교차 영상 재사용: 조회(GET /api/sync·job API)는 전부 video_id 컬럼 기반이라, 다른
    영상의 행을 재사용만 하면 이 영상은 completed인데 가사가 영영 안 뜨고 초기화(DELETE)도
    지울 행이 없어 복구 불가였다 (동일 오디오 재업로드/공식 오디오 실측). 이 영상 몫의
    행을 복사 생성하고, 대기 중인 발음/번역 메타·출처도 원본이 아닌 이 행에만 반영한다.
    재사용(완료)했으면 True. 오디오 파일 정리는 호출부 몫이다 (원격 워커는 서버에 파일이
    없고 로컬에서 지운다 — _try_complete_from_cache가 인프로세스용 래퍼)."""
    from everyric2.server.db.connection import get_session
    from everyric2.server.db.repository import JobRepository, SyncRepository

    async with get_session() as session:
        sync_repo = SyncRepository(session)
        existing = await sync_repo.get_by_audio_and_lyrics_hash(audio_hash, lyrics_hash_value)
        if not existing:
            return False
        meta = _PENDING_LINE_META.pop(job_id, None)
        attr = _PENDING_ATTRIBUTION.pop(job_id, None)
        title, artist = _PENDING_TITLE.pop(job_id, (None, None))
        target = existing
        if existing.video_id != job.video_id:
            src = dict(existing.timestamps)
            segments = [dict(s) for s in src.pop("segments", [])]
            target = await sync_repo.create(
                video_id=job.video_id,
                lyrics_hash=lyrics_hash_value,
                timestamps=segments,
                language=existing.language,
                engine=existing.engine,
                # 이건 새 정렬이 아니라 기존 행의 **복사**다 — create()의 engine_variant/
                # engine_version 기본값(None/현행 ENGINE_VERSION)에 맡기면 원본이 어떤
                # 변형·스택으로 만들어졌는지가 이 복사본에서 조용히 사라지거나(variant),
                # 실제로는 옛 스택이 만든 결과인데 지금 막 만든 것처럼 현행 스택으로
                # 잘못 표시된다(version) — 원본 값을 그대로 옮긴다.
                engine_variant=existing.engine_variant,
                engine_version=existing.engine_version,
                quality_score=existing.quality_score,
                audio_hash=audio_hash,
                extra=src,
                title=title,
                artist=artist,
            )
            # 이 로그는 **복사가 실제로 일어난 이 분기**에 있어야 한다 — 예전에는 복사가 없는
            # else(같은 영상 재사용)에 붙어 있어, 과거에 복구 불가 사고를 낸 교차 영상 복사
            # 경로가 정작 로그에 한 줄도 남지 않았다(사고 재발 시 추적 불가).
            logger.info(
                f"Job {job_id}: copied sync {existing.id} from video {existing.video_id} "
                f"(same audio+lyrics) into new row {target.id} for video {job.video_id}"
            )
        else:
            await sync_repo.set_title_if_missing(target, title, artist)
            logger.info(
                f"Job {job_id}: reusing this video's own sync {target.id} "
                f"(same audio+lyrics, no copy needed)"
            )
        meta_lang = resolve_layer_lang(job, job_id)
        updated = dict(target.timestamps)
        changed = False
        if meta:
            segs = [dict(s) for s in updated.get("segments", [])]
            # 비ko 번역은 legacy 슬롯에 넣지 않는다 — 이 행은 **이미 존재하던 싱크**라 다른
            # 언어 사용자의 번역이 들어 있을 수 있고, 그 위에 덮으면 그 사용자가 다음 조회에서
            # 남의 언어를 받는다. 언어별 값은 아래 레이어에만 남긴다. 기준은 요청자 언어가
            # 아니라 **이 메타에 실린 번역의 언어**다(resolve_layer_lang).
            # (발음은 언어 무관한 결정론 한글 독음이라 그대로 병합한다.)
            if merge_line_meta(segs, meta, with_translation=(meta_lang == "ko")):
                updated["segments"] = segs
                changed = True
        if attr is not None:
            updated["attribution"] = attr
            changed = True
        if changed:
            # JSON 컬럼은 재할당해야 변경이 감지된다
            target.timestamps = updated
        await record_translation_layer(
            session,
            job.video_id,
            [s.get("text") or "" for s in updated.get("segments", [])],
            translation_layer_lines(meta),
            meta_lang,
            origin=layer_origin(attr),
            attribution=attr,
        )
        await JobRepository(session).update_status(
            job_id, "completed", progress=100, result_id=target.id
        )
        logger.info(f"Job {job_id} reused existing sync (audio_hash match)")
        return True


async def _try_complete_from_cache(
    job_id: str, job, audio_hash: str, lyrics_hash_value: str, audio_path: str
) -> bool:
    """캐시 완결 시 다운로드한 오디오까지 정리하는 인프로세스 래퍼.

    원격 워커 경로는 서버에 오디오 파일이 없으므로 이 래퍼 대신 _complete_from_cache_db를
    직접 부르고, 로컬 오디오는 워커 쪽 hooks.cache_check가 지운다."""
    completed = await _complete_from_cache_db(job_id, job, audio_hash, lyrics_hash_value)
    if completed:
        Path(audio_path).unlink(missing_ok=True)
    return completed


class PipelineError(Exception):
    """사용자에게 보이는 파이프라인 실패 (예: 영상 과길이). str(e)가 실패 문구가 된다."""


def over_length_message(duration_sec: float, max_audio_sec: int) -> str:
    """과길이 영상 거부 문구 — run_pipeline(다운로드 후)과 미디어 캐시 프리플라이트가 공유한다."""
    return (
        f"영상이 너무 길어요 ({duration_sec / 60:.0f}분). 싱크 생성은 "
        f"{max_audio_sec // 60}분 이하의 노래 영상에서만 지원해요."
    )


def classify_job_failure(exc: BaseException) -> str | None:
    """예외 → jobs.failure_kind (MoRef 감사 #3). 취소는 다루지 않는다 — cancel API와
    _consume_cancel이 그 경로를 이미 각자 "cancelled"로 못 박는다.

    audio/downloader.py가 이미 분류해 놓은 실패(로그인요구·나이제한·봉쇄·영상불가·
    스로틀·네트워크 — DownloadError 계열이고 code가 "unknown"이 아님)는 우리 시스템 바깥
    요인이라 "external". ffmpeg·JS 런타임 미설치(DependencyError, 또는 code=="js_runtime")는
    DownloadError를 경유해도 **서버 구성 문제**라 "system"으로 남긴다 — downloader.py 자신의
    분류 (d)와 같은 성격이다. DownloadError이지만 패턴이 하나도 안 걸려 code="unknown"으로
    영문 원문이 그대로 노출된 실패는 외부 요인인지 우리 쪽 결함인지 이 함수가 판단할 근거가
    없어 None(억지 분류 금지). 그 외 전부(CTC/demucs 크래시 등 downloader와 무관한 예외)는
    "system" — 진짜 시스템 오류가 여기 모인다."""
    from everyric2.audio.downloader import DependencyError, DownloadError

    if isinstance(exc, DependencyError):
        return "system"
    if isinstance(exc, DownloadError):
        code = getattr(exc, "code", "unknown")
        if code == "unknown":
            return None
        if code == "js_runtime":
            return "system"
        return "external"
    return "system"


@dataclass
class JobInput:
    """run_pipeline 입력 — 인프로세스는 스태시를 peek해, 원격은 claim 응답으로 채운다.

    오디오 확보 우선순위(_acquire_audio): **video_id 오디오 캐시** > audio_path(인프로세스가
    미디어 캐시에서 추출해 둔 로컬 파일) > audio_url(원격 워커가 서버 캐시 파일을 HTTP로 받음)
    > yt-dlp 다운로드. 앞의 경로가 실패하면 뒤로 폴백한다. audio_hash는 어느 경로든 **받은
    파일 바이트**로 계산하므로 경로가 다르면 같은 영상도 다른 해시가 되는데, 캐시가 맨 앞에
    서면 한 영상이 항상 한 파일을 내므로 그 흔들림이 사라진다 (_acquire_audio의 주석 참고).
    """

    job_id: str
    video_id: str
    lyrics: str
    language: str | None = None
    line_meta: list[dict[str, Any]] | None = None
    attribution: dict[str, Any] | None = None
    force: bool = False
    max_audio_sec: int = 0
    # 미디어 캐시 연동 — 인프로세스는 로컬 파일 경로, 원격 워커는 인증 헤더 딸린 HTTP URL
    audio_path: str | None = None
    audio_url: str | None = None
    audio_url_headers: dict[str, str] | None = None
    # line_meta(번역·독음)가 잡 생성 뒤에 따로 도착하는 병렬 경로인지. True면 코어가 정렬
    # 진입 직전에 인메모리 스태시를 다시 확인하고 상한을 둔 대기를 한 번 넣는다.
    # 스태시는 서버 프로세스에만 있으므로 원격 워커 경로는 항상 False다 (기존 동작).
    await_line_meta: bool = False


@dataclass
class PipelineResult:
    """run_pipeline 성공 결과 — 인프로세스 저장 경로/원격 result 제출이 그대로 저장한다."""

    timestamps: list[dict[str, Any]]
    language: str | None
    quality_score: float | None
    audio_hash: str
    extra: dict[str, Any] | None
    # MMS 강제 폴백 등 엔진 변형 식별자 — None이면 변형 없음(결함 #5, ctc_engine.py의
    # _current_engine_variant를 그대로 옮긴다). SyncRepository.create(engine_variant=...)로 간다.
    engine_variant: str | None = None


class PipelineHooks(Protocol):
    """파이프라인 코어가 앞뒤(진행률·취소·캐시)를 위임하는 콜백 묶음.

    - report: 순수 진행률 보고(취소 소진 없음). 다운로드 틱·단계 모니터가 쓴다.
    - progress: 진행률 보고 + 취소 확인. False면 취소 요청됨 → 코어가 중단(None 반환).
    - cache_check: (audio_hash, lyrics) 캐시 완결 판정. True면 잡 완료·오디오 정리까지
      끝났으므로 코어가 정렬을 건너뛰고 중단한다.
    """

    async def report(self, progress: int, stage: str) -> None: ...

    async def progress(self, progress: int, stage: str) -> bool: ...

    async def cache_check(self, audio_hash: str, audio_path: str) -> bool: ...


class InProcessHooks:
    """서버 프로세스가 직접 처리할 때의 hooks — 기존 _set_progress/_consume_cancel/
    _try_complete_from_cache를 감싸 리팩터 전과 같은 관찰 동작을 낸다."""

    def __init__(self, job_id: str, job) -> None:
        self.job_id = job_id
        self.job = job

    async def report(self, progress: int, stage: str) -> None:
        await _set_progress(self.job_id, progress, stage)

    async def progress(self, progress: int, stage: str) -> bool:
        # _set_progress는 취소 대기 중이면 쓰기를 건너뛴다(가드). 이어 취소를 소진해
        # 리팩터 전의 "경계에서 취소 확인" 동작을 그대로 재현한다.
        await _set_progress(self.job_id, progress, stage)
        return not await _consume_cancel(self.job_id)

    async def cache_check(self, audio_hash: str, audio_path: str) -> bool:
        from everyric2.server.db.repository import hash_lyrics

        return await _try_complete_from_cache(
            self.job_id, self.job, audio_hash, hash_lyrics(self.job.lyrics), audio_path
        )


async def run_pipeline(job: JobInput, hooks: PipelineHooks) -> PipelineResult | None:
    """생성 파이프라인 코어 — 인프로세스 워커와 원격 워커가 공유한다.

    hooks.progress로 단계·진행률을 보고하고(False면 취소 → None 반환으로 중단),
    hooks.cache_check로 (audio_hash, lyrics) 캐시 완결을 판정한다(True면 정렬 생략,
    None 반환). 성공하면 PipelineResult를, 취소/캐시 완결이면 None을 돌려준다. 영상
    과길이 등 사용자 노출 실패는 PipelineError로 올린다. 관찰 가능한 동작(단계 문구·
    진행률 값·취소 경계·캐시 동작·실패 문구·틱/모니터 UX)은 리팩터 전과 동일하다."""
    if not await hooks.progress(10, "다운로드"):
        return None

    dl_ticker = asyncio.create_task(
        _tick_progress(hooks.report, start=10, cap=33, interval=2.0, stage="다운로드")
    )
    try:
        download_result = await asyncio.get_event_loop().run_in_executor(
            None, _acquire_audio, job
        )
    finally:
        dl_ticker.cancel()
    audio_hash = download_result["audio_hash"]
    audio_path = download_result["audio_path"]

    # 노래가 아닌 초장시간 영상(팟캐스트/라이브 다시보기)이 GPU 슬롯을 몇 시간씩 점유하는
    # 것을 막는다 — 상한 초과는 정렬 전에 친절하게 실패. 취소 경계(아래 progress)보다 먼저
    # 검사해 리팩터 전 "다운로드 직후 → 과길이 검사 → 캐시 확인" 순서를 보존한다.
    if job.max_audio_sec > 0:
        duration = _audio_duration_sec(audio_path)
        if duration and duration > job.max_audio_sec:
            Path(audio_path).unlink(missing_ok=True)
            raise PipelineError(over_length_message(duration, job.max_audio_sec))

    # 다운로드 완료 → 캐시 확인 (취소 경계 겸)
    if not await hooks.progress(35, "캐시 확인"):
        Path(audio_path).unlink(missing_ok=True)
        return None

    if not job.force and await hooks.cache_check(audio_hash, audio_path):
        # 캐시로 완결 — 오디오는 hooks.cache_check가 이미 정리했다
        return None

    # 캐시 미스 → 정렬 진입 (취소 경계 겸)
    if not await hooks.progress(36, "보컬 분리"):
        Path(audio_path).unlink(missing_ok=True)
        return None

    # line_meta 지연 도착 경로: 정렬 스레드가 CTC 진입 직전에 부르는 리졸버를 넘긴다.
    # 그 앞의 오디오 로드·CTC 모델 웜업·보컬 분리·f0 전곡 추론은 line_meta 없이 돌 수
    # 있으므로 클라이언트의 번역 시간과 그만큼이 겹친다. 리졸버가 무엇을 물어 왔는지는
    # 아래 재병합·출처 조립에서 다시 필요하니 상자에 담아 밖으로 꺼낸다.
    meta_box: dict[str, list[dict[str, Any]] | None] = {"line_meta": job.line_meta}

    def _resolve_line_meta() -> list[dict[str, Any]] | None:
        meta_box["line_meta"] = _wait_for_line_meta(job.job_id, LINE_META_WAIT_SEC)
        return meta_box["line_meta"]

    resolver = _resolve_line_meta if (job.await_line_meta and not job.line_meta) else None

    # 정렬(CTC+분리+보정+멜로디)은 수십 초 걸리는 단일 블록 — 정렬 스레드가 단계명을
    # stage_holder에 쓰고, 모니터가 단계 창 안에서 진행률을 차오르게 하며 보고한다
    stage_holder: dict[str, str] = {"stage": "보컬 분리"}
    monitor = asyncio.create_task(_stage_monitor(hooks.report, stage_holder, start=36))
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            _run_alignment,
            audio_path,
            job.lyrics,
            job.language,
            job.line_meta,
            lambda name: stage_holder.__setitem__("stage", name),
            resolver,
            # 자막 앵커 조달용 — 가사 출처와 무관하게 «이 영상»의 사람 자막 시각을 본다
            job.video_id,
        )
    except JobCancelled:
        # line_meta 대기 중 취소 — 오디오는 _run_alignment의 finally가 이미 지웠다.
        # 경계 progress로 취소를 소진해 다른 취소 경계와 동일하게 마감한다
        # (_set_progress는 취소 대기 중이면 쓰기를 건너뛰므로 failed가 되돌려지지 않는다).
        await hooks.progress(48, LINE_META_WAIT_STAGE)
        return None
    finally:
        monitor.cancel()

    # 정렬 완료, 저장 단계 (취소 경계 겸) — 오디오는 _run_alignment의 finally가 정리했다
    if not await hooks.progress(90, "저장"):
        return None

    line_meta = meta_box["line_meta"]
    attribution = job.attribution
    if job.await_line_meta:
        # 상한을 넘겨 원문으로 정렬한 뒤에 도착한 line_meta도 표시용으로는 살린다 —
        # 정렬 텍스트는 이미 원문이지만 발음·번역 텍스트를 버릴 이유는 없다.
        # (스태시는 서버 프로세스에만 있고 await_line_meta는 그 경로에서만 True다)
        line_meta = _PENDING_LINE_META.get(job.job_id) or line_meta
        attribution = _PENDING_ATTRIBUTION.get(job.job_id) or attribution

    # 독음 정렬 경로는 발음/번역/pron_segments를 이미 세그먼트에 붙였으므로 재병합 생략
    if line_meta and result.get("alignment_text") != "pronunciation":
        merged = merge_line_meta(result["timestamps"], line_meta)
        logger.info(f"Line meta merged on {merged} segments")

    return PipelineResult(
        timestamps=result["timestamps"],
        language=result.get("language"),
        quality_score=result.get("quality_score"),
        audio_hash=audio_hash,
        extra=_build_extra(result, attribution),
        engine_variant=result.get("engine_variant"),
    )


async def _process_job_inner(job_id: str, job) -> None:
    from everyric2.config.settings import get_settings as _get_settings
    from everyric2.server.db.connection import get_session
    from everyric2.server.db.repository import JobRepository, SyncRepository, hash_lyrics

    # 스태시(발음/번역 메타·출처·강제)를 peek해 코어 입력을 만든다. 정상 완료/실패 시
    # 아래에서 pop한다 (캐시 완결 경로는 _complete_from_cache_db가 이미 pop). force는
    # 코어 입력으로 캡처했으니 여기서 discard한다.
    force = job_id in _PENDING_FORCE
    _PENDING_FORCE.discard(job_id)
    # line_meta 지연 도착 예고도 같은 관례로 캡처 후 즉시 비운다
    await_meta = job_id in _PENDING_META_WAIT
    _PENDING_META_WAIT.discard(job_id)
    max_audio_sec = _get_settings().server.max_job_audio_sec

    # 슬롯 획득 직후 = 잡이 이 프로세스로 넘어오는 순간 → 미디어 캐시 조회(있으면 추출 사용).
    # 과길이 프리플라이트는 다운로드 없이 즉시 실패시킨다.
    from everyric2.server.media_cache import prepare_cached_audio

    cache_path, fail_reason = await prepare_cached_audio(job.video_id, job_id, max_audio_sec)
    if fail_reason:
        _PENDING_LINE_META.pop(job_id, None)
        _PENDING_LINE_META_LANG.pop(job_id, None)
        _PENDING_ATTRIBUTION.pop(job_id, None)
        _PENDING_TITLE.pop(job_id, None)
        async with get_session() as session:
            await JobRepository(session).update_status(job_id, "failed", error=fail_reason)
        logger.info(f"Job {job_id} rejected (media cache preflight): {fail_reason}")
        return

    job_input = JobInput(
        job_id=job_id,
        video_id=job.video_id,
        lyrics=job.lyrics,
        language=job.language,
        line_meta=_PENDING_LINE_META.get(job_id),
        attribution=_PENDING_ATTRIBUTION.get(job_id),
        force=force,
        max_audio_sec=max_audio_sec,
        audio_path=cache_path,
        await_line_meta=await_meta,
    )
    try:
        result = await run_pipeline(job_input, InProcessHooks(job_id, job))
        if result is None:
            # 취소 또는 캐시 완결 — 잡 상태·오디오 정리는 각 경로가 이미 끝냈다
            _PENDING_LINE_META.pop(job_id, None)
            _PENDING_LINE_META_LANG.pop(job_id, None)
            _PENDING_ATTRIBUTION.pop(job_id, None)
            _PENDING_TITLE.pop(job_id, None)
            return

        async with get_session() as session:
            job_repo = JobRepository(session)
            sync_repo = SyncRepository(session)

            # 번역 언어 분리: 생성 결과의 번역을 **그 번역의 언어** 레이어에 남기고, 그것이
            # ko가 아닐 때만 legacy 슬롯(seg["translation"])을 비운다. ko 번역을 legacy에
            # 남기는 이유는 구버전 확장과 ko 사용자에게 그대로 유효해서고, 비ko를 비우는
            # 이유는 lang 없이 조회한 다른 사용자가 남의 언어를 받지 않게 하기 위해서다.
            # lang을 안 싣는 구버전 생성 요청은 "ko"라 기존 동작 그대로다.
            meta_lang = resolve_layer_lang(job, job_id)
            job_attr = _PENDING_ATTRIBUTION.get(job_id)
            await record_translation_layer(
                session,
                job.video_id,
                [s.get("text") or "" for s in result.timestamps],
                translation_layer_lines(result.timestamps),
                meta_lang,
                origin=layer_origin(job_attr),
                attribution=job_attr,
            )
            if meta_lang != "ko":
                for seg in result.timestamps:
                    seg.pop("translation", None)

            title, artist = peek_title(job_id)
            sync_result = await sync_repo.create(
                video_id=job.video_id,
                lyrics_hash=hash_lyrics(job.lyrics),
                timestamps=result.timestamps,
                language=result.language,
                engine="ctc",
                engine_variant=result.engine_variant,
                quality_score=result.quality_score,
                audio_hash=result.audio_hash,
                extra=result.extra,
                title=title,
                artist=artist,
            )

            await job_repo.update_status(
                job_id, "completed", progress=100, result_id=sync_result.id
            )
            logger.info(f"Job completed: {job_id}")
        _PENDING_LINE_META.pop(job_id, None)
        _PENDING_LINE_META_LANG.pop(job_id, None)
        _PENDING_ATTRIBUTION.pop(job_id, None)
        _PENDING_TITLE.pop(job_id, None)

    except PipelineError as e:
        # 사용자 노출 실패 (과길이 등) — 친절한 한국어 문구를 그대로 보존
        _PENDING_LINE_META.pop(job_id, None)
        _PENDING_LINE_META_LANG.pop(job_id, None)
        _PENDING_ATTRIBUTION.pop(job_id, None)
        _PENDING_TITLE.pop(job_id, None)
        async with get_session() as session:
            await JobRepository(session).update_status(job_id, "failed", error=str(e))
        logger.info(f"Job {job_id} rejected: {e}")

    except Exception as e:
        logger.exception(f"Job failed: {job_id}")
        _PENDING_LINE_META.pop(job_id, None)
        _PENDING_LINE_META_LANG.pop(job_id, None)
        _PENDING_ATTRIBUTION.pop(job_id, None)
        _PENDING_TITLE.pop(job_id, None)
        _PENDING_FORCE.discard(job_id)
        async with get_session() as session:
            job_repo = JobRepository(session)
            await job_repo.update_status(
                job_id, "failed", error=str(e), failure_kind=classify_job_failure(e)
            )

    finally:
        # 잡 경계 VRAM 위생 (인프로세스 워커 경로) — 앨로케이터가 사재기한 활성 스파이크
        # 예약을 반환한다. 원격 워커는 cli._worker_loop가 같은 훅을 부른다.
        from everyric2.gpu_mem import reclaim_after_job

        await asyncio.get_event_loop().run_in_executor(None, reclaim_after_job)


async def _set_progress(job_id: str, progress: int, stage: str | None = None) -> None:
    from everyric2.server.db.connection import get_session
    from everyric2.server.db.repository import JobRepository

    # 취소 대기 중이면 진행률 갱신을 멈춘다 — 취소 API가 이미 failed로 마감했는데
    # 모니터가 processing으로 되돌려 쓰면 클라이언트가 failed↔processing 왕복을 본다
    if job_id in _CANCEL_REQUESTED:
        return
    async with get_session() as session:
        await JobRepository(session).update_status(
            job_id, "processing", progress=progress, stage=stage
        )


async def _tick_progress(
    report, start: int, cap: int, interval: float = 4.0, stage: str | None = None
) -> None:
    """긴 단계 동안 진행률을 cap까지 천천히 올린다 — 취소되면 그대로 멈춘다.

    report는 순수 진행률 보고 콜백(hooks.report) — 취소를 소진하지 않는다. 틱이 취소를
    소진해 버리면 경계의 progress가 취소를 못 보고 잡이 그대로 진행되므로 반드시 report다."""
    progress = start
    try:
        while progress < cap:
            await asyncio.sleep(interval)
            progress = min(cap, progress + 4)
            await report(progress, stage)
    except asyncio.CancelledError:
        pass


async def _stage_monitor(report, stage_holder: dict[str, str], start: int, interval: float = 2.0) -> None:
    """정렬 블록 동안 stage_holder의 현재 단계를 읽어 단계명+진행률을 report로 보고한다.

    단계가 바뀌면 그 단계 창의 시작으로 점프하고, 같은 단계가 유지되는 동안은
    틱마다 창 폭의 1/6씩 상한까지 차오른다 (내부 진행 콜백이 없는 근사치)."""
    progress = float(start)
    last_stage: str | None = None
    try:
        while True:
            await asyncio.sleep(interval)
            stage = stage_holder.get("stage")
            if not stage:
                continue
            lo, hi = STAGE_WINDOWS.get(stage, (36, 88))
            if stage != last_stage:
                last_stage = stage
                progress = max(progress, float(lo))
            else:
                progress = min(float(hi), progress + (hi - lo) / 6.0)
            await report(int(progress), stage)
    except asyncio.CancelledError:
        pass


def _acquire_audio(job: "JobInput") -> dict:
    """오디오 확보 — video_id 캐시 > audio_path(로컬 캐시 추출) > audio_url(서버 캐시 HTTP) > yt-dlp.

    **video_id 캐시가 맨 앞에 선다.** 그 뒤의 세 경로는 어느 것이든 성공하면 결과를 캐시에
    보관하므로, 같은 영상의 두 번째 요청부터는 유튜브에 닿지 않는다. 확보 전체를 영상별 락으로
    감싸 동시 요청을 한 번으로 병합한다(single-flight) — 캐시는 두 번째 요청부터 듣지만 락은
    첫 순간부터 듣고, 공개 트래픽은 인기곡에 동시에 몰린다.

    ``force``(강제 재생성)는 오디오 캐시를 무시하지 **않는다**. force가 뜻하는 것은 «같은
    (audio_hash, lyrics) 싱크를 재사용하지 말고 정렬을 다시 돌려라»이고, 그 정렬에 쓸 오디오를
    다시 받아 올 이유는 없다. 덕분에 트랙 선택 수정 같은 회귀 검증을 재다운로드 0으로 돌린다.

    앞선 캐시 경로가 실패하면 조용히 yt-dlp로 폴백한다(INFO 로그 1줄).

    **audio_hash는 확보 경로에 의존한다** — 예전 독스트링은 "같은 원본이면 해시도 같다"고
    단언했지만 성립하지 않는다. 미디어 캐시 경로는 ``-acodec copy``로 m4a를 만들고
    (media_cache._run_ffmpeg) yt-dlp 경로는 wav로 트랜스코드하므로, 같은 영상도 바이트가
    달라 다른 해시가 된다. 그래서 같은 영상을 두 경로로 처리하면 캐시가 미스해 GPU 정렬을
    다시 돌린다(교차 영상 재사용도 경로가 갈리면 못 잡는다).

    **내용 기반 해시로 고치지 않는 이유 (실측)**: 디코딩 PCM을 정규화해(16k mono s16) md5를
    떠도 경로 독립이 되지 않는다. 같은 AAC 스트림을 두고
        AAC → 16k mono            : pcm md5 = a235c4d8da8f5bc736db2329a6cc35db
        AAC → 44.1k wav → 16k mono: pcm md5 = 7f46eea06d76b9b1897c7b2e4f251de5
    로 갈렸다 (ffmpeg 실측, 5초 사인파). 중간 wav가 이미 s16으로 양자화돼 리샘플 순서가
    달라지기 때문이다. 컨테이너만 바꾸는 스트림카피는 안정적이었지만(``-acodec copy`` 전후
    pcm md5 동일), 실제 두 경로는 애초에 **서로 다른 인코딩**(yt-dlp bestaudio는 보통 opus,
    미디어 캐시는 원본 컨테이너의 스트림)에서 출발하므로 어떤 정확 해시로도 일치시킬 수 없다.
    일치시키려면 지문(chromaprint류)이 필요한데, 그것은 새 의존성이고 **오탐(다른 곡을 같다고
    보는 것)** 을 들여온다 — 미스는 GPU 재정렬(무해)이고 오탐은 남의 가사를 붙이는 사고라
    비대칭이 크다. 그래서 미스를 감수하고 바이트 해시를 유지한다.

    (위 비대칭은 video_id 캐시가 앞에 서면서 실질적으로 사라진다 — 한 영상은 한 파일이다.
    캐시가 꺼진 배포에서는 예전 그대로다.)"""
    from everyric2.audio import cache as audio_cache
    from everyric2.config.settings import get_settings

    temp_dir = get_settings().audio.temp_dir
    tag = job.job_id[:8]

    # 락 밖에서 먼저 본다 — 캐시에 있으면 남을 기다릴 이유가 없다
    hit = audio_cache.take(job.video_id, temp_dir, tag)
    if hit is not None:
        return {"audio_path": str(hit), "audio_hash": compute_audio_hash(hit)}

    with audio_cache.video_lock(job.video_id):
        # 락을 기다리는 동안 앞선 잡이 받아 뒀을 수 있다 (병합이 실제로 일어나는 지점)
        hit = audio_cache.take(job.video_id, temp_dir, tag)
        if hit is not None:
            return {"audio_path": str(hit), "audio_hash": compute_audio_hash(hit)}
        acquired = _acquire_audio_uncached(job)
        # 보관은 락 안에서 — 대기 중인 잡이 락을 얻는 순간 캐시가 이미 채워져 있어야 한다
        audio_cache.put(job.video_id, Path(acquired["audio_path"]))
    return acquired


def _acquire_audio_uncached(job: "JobInput") -> dict:
    """캐시를 거치지 않는 확보 경로 — audio_path > audio_url > yt-dlp."""
    # 인프로세스: 서버가 미디어 캐시에서 추출해 넘긴 로컬 파일 직사용
    if job.audio_path:
        p = Path(job.audio_path)
        if p.exists():
            try:
                return {"audio_path": str(p), "audio_hash": compute_audio_hash(p)}
            except Exception:
                logger.info("캐시 오디오 파일을 읽지 못해 yt-dlp로 폴백해요 (video %s)", job.video_id)
        else:
            logger.info("캐시 오디오 파일이 없어 yt-dlp로 폴백해요 (video %s)", job.video_id)
    # 원격 워커: 서버 캐시 파일을 인증 헤더로 HTTP 다운로드
    if job.audio_url:
        try:
            path = _http_download_audio(
                job.audio_url, job.audio_url_headers, job.video_id, job.job_id
            )
            return {"audio_path": str(path), "audio_hash": compute_audio_hash(path)}
        except Exception:
            logger.info("서버 캐시 오디오 받기 실패 — yt-dlp로 폴백해요 (video %s)", job.video_id)
    return _download_and_hash(job.video_id, job.job_id)


def _http_download_audio(
    url: str, headers: dict[str, str] | None, video_id: str, job_id: str
) -> Path:
    """서버 캐시 오디오를 HTTP로 받아 임시 파일에 저장 (원격 워커 전용). 의존성 없이 requests."""
    import requests

    from everyric2.config.settings import get_settings

    temp_dir = get_settings().audio.temp_dir
    temp_dir.mkdir(parents=True, exist_ok=True)
    dest = temp_dir / f"{video_id}-{job_id[:8]}-cache.m4a"
    with requests.get(url, headers=headers or {}, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    return dest


def _download_and_hash(video_id: str, job_id: str) -> dict:
    from everyric2.audio.downloader import YouTubeDownloader

    downloader = YouTubeDownloader()
    youtube_url = f"https://www.youtube.com/watch?v={video_id}"
    # 잡별 고유 파일명 — 기본 %(title)s 템플릿은 같은 영상의 동시 잡이 한 파일을 두고
    # 경합해 Windows에서 WinError 32(파일 사용 중)로 다운로드가 깨진다
    dl_result = downloader.download(youtube_url, filename=f"{video_id}-{job_id[:8]}")
    audio_hash = compute_audio_hash(dl_result.audio_path)

    return {
        "audio_path": str(dl_result.audio_path),
        "audio_hash": audio_hash,
    }


def _build_extra(result: dict[str, Any], attribution: dict[str, Any] | None) -> dict[str, Any] | None:
    """싱크 JSON의 segments 밖 부가정보(디버그 메타, 출처 표기, 템포, 키, 추임새) 조립."""
    extra: dict[str, Any] = {}
    if result.get("debug"):
        extra["debug"] = result["debug"]
    if result.get("tempo"):
        extra["tempo"] = result["tempo"]
    if result.get("key"):
        extra["key"] = result["key"]
    if result.get("adlib"):
        # 새 스택 전용 additive 필드 — 구간 배열 [[start,end],...] (레거시 응답엔 없다).
        extra["adlib"] = result["adlib"]
    if attribution is not None:
        extra["attribution"] = attribution
    return extra or None


def _estimate_tempo(audio) -> dict[str, Any] | None:
    """librosa로 BPM·첫 비트 시각 추정 — 가라오케 레인의 박자/마디 격자용.

    보컬로이드 곡은 대부분 고정 BPM이라 (bpm, beat_offset)만으로 전 곡 격자를
    재구성할 수 있다. 실패는 치명적이지 않으므로 None으로 조용히 폴백.
    """
    try:
        import librosa
        import numpy as np

        y = np.asarray(audio.waveform, dtype=np.float32)
        sr = int(audio.sample_rate)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, trim=False)
        bpm = float(np.atleast_1d(tempo)[0])
        if not (30.0 <= bpm <= 300.0) or len(beats) < 8:
            return None
        beat_times = librosa.frames_to_time(beats, sr=sr)
        # 첫 비트 위상 — 비트 간격의 중앙값으로 격자를 안정화
        interval = float(np.median(np.diff(beat_times)))
        if interval > 0:
            bpm = 60.0 / interval
        return {"bpm": round(bpm, 2), "beat_offset": round(float(beat_times[0]), 3)}
    except Exception:
        logger.exception("Tempo estimation failed; lane falls back to seconds grid")
        return None


def _separate_stems(audio):
    """demucs 분리 결과 전체 (실패/미설치 시 None) — 보컬은 정렬·VAD·f0가 쓰고,
    반주는 star 성형의 보컬 우세도 계산(star_prior.vocal_presence_from_stems)이 쓴다.

    분리기는 웜 캐시 싱글턴(get_shared_separator)에서 가져와 잡마다 재생성하지 않는다 (WS2-A)."""
    try:
        import torch

        from everyric2.audio.separator import get_shared_separator

        separator = get_shared_separator()
        if not separator.is_available():
            logger.info("demucs not installed; skipping VAD clamp / using mix for melody")
            return None
        return separator.separate(audio, use_gpu=torch.cuda.is_available())
    except Exception:
        logger.exception("Vocal separation failed; skipping VAD clamp")
        return None


def _separate_vocals(audio):
    """demucs 보컬 스템만 (기존 호출부 호환 래퍼)."""
    result = _separate_stems(audio)
    return result.vocals if result is not None else None


def _repeat_key(text: str) -> str:
    """반복행 판정용 키 — 공백/기호를 지우고 대소문자를 무시한 텍스트."""
    return re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE).casefold()


def _clamp_repeated_outliers(results, clamped: set[int]) -> None:
    """같은 가사가 3번 이상 반복될 때, 형제 라인 duration 중앙값 대비 병적으로 긴
    라인을 중앙값 길이로 잘라낸다(시작 유지, end = start + 중앙값).

    CTC가 같은 훅을 반복해 부를 때 글자를 특정 렌디션에 몰아 흩뿌려, 같은 텍스트의
    다른 반복 라인은 ~2초인데 한 라인만 7초 outlier로 늘어나는 케이스를 잡는다.
    형제가 2개 이하이거나 중앙값 자체가 비정상(<0.5s)이면 건드리지 않는다.
    """
    groups: dict[str, list[int]] = {}
    for i, r in enumerate(results):
        key = _repeat_key(r.text)
        if key:
            groups.setdefault(key, []).append(i)
    for idxs in groups.values():
        if len(idxs) < 3:
            continue
        median = statistics.median(results[i].end_time - results[i].start_time for i in idxs)
        if median < 0.5:
            continue
        limit = max(median * 2.5, 4.0)
        for i in idxs:
            if i in clamped:
                continue  # 기존 규칙이 이미 처리한 라인은 그대로 둔다
            r = results[i]
            if r.end_time - r.start_time > limit:
                r.end_time = r.start_time + median
                clamped.add(i)


def _starts_on_vocal_onset(line, regions, tol: float = 0.5) -> bool:
    """라인의 첫 글자가 실제 발성 리전의 온셋 위(±tol)에 얹혀 있는지.

    "발성 위에 있는가"(리전 안 어디든)가 아니라 "**온셋과 맞는가**"를 본다 — 늦게 잡힌
    라인도 그 시점엔 노래가 이어지고 있어 리전 '안'에는 들어가므로, 단순 겹침으로는
    정확한 시작과 늦은 시작을 못 가른다. 온셋 일치는 CTC가 실제 발성 시작을 물었다는 뜻이다.
    글자 타이밍이 없으면(word_segments 없음) 판정하지 않는다 — 라인 start는 이미
    보정 대상 값이라 근거가 될 수 없다.
    """
    toks = [w for w in (getattr(line, "word_segments", None) or []) if w.word]
    if not toks:
        return False
    first = toks[0].start
    return any(abs(first - reg.start) <= tol for reg in regions)


def _pull_post_interlude_starts(results, vad_result, clamped: set[int]) -> None:
    """긴 간주(직전 라인 end와 8초 이상 벌어짐) 뒤 첫 라인의 시작이 실제 보컬 시작보다
    늦게 잡히면, 라인이 속한 가창 블록의 시작으로 라인 start를 당긴다.

    앵커는 "간주 이후 첫 리전"이 아니라 **라인과 겹치는 첫 리전에서 뒤로(≤2s 간격)
    이어지는 리전 체인의 시작**이다 — 간주 초입의 고립된 잔향/애드립 리전(체인 밖)에
    끌려가 3배 가드에 걸리는 오탐을 막는다 (熱異常 실측: 40초 간주 초입 0.6초 잔향).
    end는 유지하고, 당긴 결과 duration이 원래의 3배를 넘으면 오탐으로 보고 건너뛴다.

    **첫 글자가 실제 발성 온셋 위에 얹혀 있으면(±0.5s) 당기지 않는다** — CTC가 잡은
    시작이 VAD 온셋과 맞아떨어졌다는 것은 그 시작이 이미 옳다는 음향 증거다. 이 가드가
    없어 정확한 raw를 뒤로 끌어당긴 실측 사고가 있었다 (G5hScSFkib4 idx6 +0.37→−3.30s,
    idx27 +0.27→−6.91s). 반대로 진짜 늦게 잡힌 라인(熱異常)은 첫 글자가 리전 온셋이 아니라
    이미 진행 중인 발성 한복판에 떨어져 이 가드에 안 걸린다.
    당길 때는 라인 start만이 아니라 **글자 스팬도 함께 옮긴다** — start만 바꾸면 라인 시작과
    글자 위치가 어긋난 채 남아 이후 단계가 '글자가 라인 밖'으로 오판한다.
    """
    all_regions = sorted(vad_result.regions, key=lambda reg: reg.start)
    for i in range(1, len(results)):
        r = results[i]
        prev_end = results[i - 1].end_time
        if r.start_time - prev_end < 8.0:
            continue
        if _starts_on_vocal_onset(r, all_regions):
            continue  # 첫 글자가 실제 발성 온셋 위 — raw 시작이 이미 옳다
        # 간주~라인 구간의 발성 리전 (시간순)
        regions = sorted(
            (reg for reg in vad_result.regions if reg.end > prev_end and reg.start < r.end_time),
            key=lambda reg: reg.start,
        )
        j = next((k for k, reg in enumerate(regions) if reg.end > r.start_time), None)
        if j is None:
            continue
        # 라인과 겹치는 첫 리전에서 뒤로 이어지는 가창 블록의 시작까지 역추적
        while j > 0 and regions[j].start - regions[j - 1].end <= 2.0:
            j -= 1
        anchor = regions[j].start
        if anchor > r.start_time - 1.5:
            continue
        new_start = anchor - 0.15
        orig_dur = r.end_time - r.start_time
        if r.end_time - new_start > 3.0 * orig_dur:
            continue  # 3배 초과로 늘어나면 오탐 — 적용하지 않는다
        r.start_time = new_start
        if r.word_segments:
            _shift_word_segments(r.word_segments, r.start_time, r.end_time)
        clamped.add(i)


def _extend_phrase_final_tails(results, vad_result, clamped: set[int]) -> None:
    """소절 끝(뒤에 0.3초 이상 갭) 라인의 끝을 실제 발성 끝까지 연장한다.

    CTC는 마지막 음절을 온셋에서 끊어 늘임음(held note)의 감쇠를 따라가지 않는다 —
    SRT/VAD 이중 실측으로 phrase-final 라인의 86~100%가 median 0.4~0.66초 일찍
    끝남이 확인됨. 라인 끝이 속한 VAD 리전의 끝까지(단 다음 라인 시작 -0.05초,
    캡 이내) 라인과 마지막 글자의 end를 함께 연장한다.
    캡은 적응형: 리전 꼬리가 3초 이내이고 다음 라인이 8초 안에 이어지면 진짜
    늘임음으로 보고 +2.5초까지, 그 밖(꼬리>3초 병합 리전 의심, 또는 간주 직전
    라인)은 +1.5초로 보수적으로 자른다. 간주 직전은 리전 끝이 잔향·악기 유입으로
    실제 발성보다 늦게 잡히는 데다 재실행 간 ±1초 가까이 흔들려서(커버 실측
    cue#37: 리전 꼬리 1.5→2.3s 변동으로 과연장 악화) 꼬리 길이만으로는 진짜
    늘임음과 구분할 수 없다 — 잔존 3건(사비 중간, 다음 줄 갭 ~3초)과 과연장
    1건(간주 앞, 갭 22초)을 가르는 신호는 다음 라인까지의 갭이었다.
    소절 중간(butted) 라인과 이미 클램프로 잘라낸 라인은 건드리지 않는다.
    """
    for i, r in enumerate(results):
        if i in clamped:
            continue
        next_start = results[i + 1].start_time if i + 1 < len(results) else float("inf")
        if next_start - r.end_time <= 0.3:
            continue  # butted — 다음 음절이 바로 이어지는 라인은 그대로
        region = next(
            (reg for reg in vad_result.regions if reg.start <= r.end_time < reg.end), None
        )
        if region is None:
            continue  # 라인 끝이 발성 리전 밖 — 따라갈 꼬리가 없다
        real_tail = region.end - r.end_time <= 3.0 and next_start - r.end_time < 8.0
        cap = 2.5 if real_tail else 1.5
        new_end = min(region.end, next_start - 0.05, r.end_time + cap)
        if new_end <= r.end_time + 0.05:
            continue
        r.end_time = new_end
        if r.word_segments:
            r.word_segments[-1].end = new_end


def _line_body_region(line, regions):
    """라인의 '정체'를 담은 발성 리전 — word_segments 질량이 가장 큰 리전.

    글자(word_segments)의 중점이 어느 리전 위에 떨어지는지로 질량을 세고(글자 수 가중),
    최대 질량 리전을 돌려준다. 중점을 쓰는 이유: CTC 잔해는 글자 스팬이 길이 0으로
    무너지는 일이 잦아 '겹침 길이'로는 질량이 전부 0이 된다 — 중점은 스팬이 무너져도
    위치를 보존한다. 어느 리전에도 안 떨어지는 글자(무음 위)는 세지 않는다.
    질량을 못 재면(word_segments 없음/전부 무음 위) 첫 리전으로 폴백한다.

    라인이 여러 리전에 걸칠 때 **첫 리전을 무조건 정체로 가정하면** 실제 내용이 뒤쪽
    리전에 있는 라인의 맞는 끝을 버리게 된다 (실측: OHcNQHbWrFY idx21 −35.69s,
    G5hScSFkib4 idx26 −13.57s — 둘 다 raw 끝이 정답에 근접했는데 클램프가 파괴).
    """
    if not regions:
        return None
    toks = [w for w in (getattr(line, "word_segments", None) or []) if w.word]
    if toks:
        mass = [0.0] * len(regions)
        for w in toks:
            mid = (w.start + w.end) / 2.0
            for k, reg in enumerate(regions):
                if reg.start <= mid <= reg.end:
                    mass[k] += len(w.word)
                    break
        best = max(range(len(regions)), key=lambda k: mass[k])  # 동률은 가장 이른 리전
        if mass[best] > 0:
            return regions[best]
    return regions[0]


def _diff_fixes(
    fixes: dict[int, list[str]],
    label: str,
    before: list[tuple[float, float]],
    results,
    tol: float = 0.01,
) -> None:
    """스테이지 전후 타이밍 diff로 어떤 규칙이 어떤 라인을 고쳤는지 라벨링 (디버그용)."""
    for i, r in enumerate(results):
        if abs(r.start_time - before[i][0]) > tol or abs(r.end_time - before[i][1]) > tol:
            labels = fixes.setdefault(i, [])
            if label not in labels:
                labels.append(label)


def _clamp_stretched_lines(results, vad_result, fixes: dict[int, list[str]] | None = None):
    """가사에 없는 반복 가창(라인 내부 퍼짐)으로 병적으로 길어진 라인을 잘라낸다.

    CTC는 같은 가사가 여러 번 불리면 글자들을 여러 렌디션에 걸쳐 흩뿌릴 수 있다
    (라인 사이 star로는 못 잡는 케이스). 지속 8초 초과 + 발성 커버리지 50% 미만인
    라인만 **글자 질량이 실린 발성 구간**(_line_body_region) 끝으로 클램프한다 —
    정상 라인은 건드리지 않는다.
    여기에 더해 반복행 outlier 클램프·간주 후 시작 앵커 당기기·소절 끝 늘임음
    연장을 함께 적용한다. fixes를 넘기면 규칙별 적용 라인을 라벨링한다(디버그).
    반환: (results, 클램프된 라인 인덱스 집합)
    """
    clamped: set[int] = set()
    before = [(r.start_time, r.end_time) for r in results] if fixes is not None else None
    for i, r in enumerate(results):
        dur = r.end_time - r.start_time
        if dur <= 8.0:
            continue
        regions = [
            reg for reg in vad_result.regions if reg.end > r.start_time and reg.start < r.end_time
        ]
        if not regions:
            continue
        vocal = sum(min(reg.end, r.end_time) - max(reg.start, r.start_time) for reg in regions)
        if vocal / dur >= 0.5:
            continue
        body = _line_body_region(r, regions)
        new_end = min(r.end_time, max(body.end + 0.3, r.start_time + 1.5))
        if new_end < r.end_time:
            r.end_time = new_end
            clamped.add(i)
    if fixes is not None and before is not None:
        _diff_fixes(fixes, "stretch", before, results)
        before = [(r.start_time, r.end_time) for r in results]
    # 반복행 형제 대비 outlier로 늘어난 라인 + 간주 뒤 늦게 시작한 라인도 보정
    _clamp_repeated_outliers(results, clamped)
    if fixes is not None and before is not None:
        _diff_fixes(fixes, "repeat", before, results)
        before = [(r.start_time, r.end_time) for r in results]
    _pull_post_interlude_starts(results, vad_result, clamped)
    if fixes is not None and before is not None:
        _diff_fixes(fixes, "pull", before, results)
        before = [(r.start_time, r.end_time) for r in results]
    # 소절 끝 늘임음은 실제 발성 끝까지 연장 (클램프된 라인 제외)
    _extend_phrase_final_tails(results, vad_result, clamped)
    if fixes is not None and before is not None:
        _diff_fixes(fixes, "tail", before, results)
    if clamped:
        logger.info(f"Clamped {len(clamped)} pathologically stretched lines")
    return results, clamped


def _snap_silence_undershoot(results, vad_result, clamped: set[int]) -> None:
    """무음(간주)에 좌초한 라인을 다음 발성 온셋으로 스냅한다 (독음 정렬 전용).

    독음(ko) 정렬의 유일한 실패 모드: 간주 전후 전이 라인이 실제 가창(간주 이후)보다
    앞선 무음 구간에 배치된다 (熱異常 L94: 실제 165.5s인데 간주 무음 146s로 언더슛).
    라인 창의 발성 커버리지가 매우 낮고(<0.25 — 순수 무음 또는 간주 초입 잔향 blip만
    스침) 다음 발성 리전까지 >=1.5s 벌어져 있으면 그 온셋 직전(-0.15s)으로 당긴다.
    라인 전체가 무음에 갇혔으면(끝이 다음 온셋 이전) 길이를 보존해 통째로 이동한다.

    ``_pull_post_interlude_starts``(늦게 잡힌 시작을 앞으로 당김)와 방향이 겹칠 수 있어
    **_clamp_stretched_lines 이전에** 돌려, 좌초 라인이 먼저 제자리(다음 온셋)를 잡게 한다
    — 그래야 뒤따르는 정상 라인이 간주 후 첫 라인으로 오인돼 도로 당겨지지 않는다.
    커버리지가 조금이라도 있으면(온셋 리드·늘임음 꼬리 포함) 보수적으로 건드리지 않는다.
    """
    regions = sorted(vad_result.regions, key=lambda reg: reg.start)
    if not regions:
        return

    def _cover(line) -> float:
        d = max(1e-6, line.end_time - line.start_time)
        return sum(
            max(0.0, min(reg.end, line.end_time) - max(reg.start, line.start_time))
            for reg in regions
        ) / d

    last_snapped = float("-inf")
    for i, r in enumerate(results):
        if i in clamped:
            continue
        s, e = r.start_time, r.end_time
        dur = max(1e-6, e - s)
        if _cover(r) >= 0.25:
            continue  # 라인이 발성과 유의미하게 겹침 → 정상 배치, 건드리지 않음
        nxt = next((reg for reg in regions if reg.start >= s), None)
        if nxt is None:
            continue  # 뒤에 발성 없음 (곡 끝 무음) → 스냅할 온셋 없음
        if nxt.start - s < 1.5:
            continue  # 온셋 직전의 짧은 리드타임은 정상
        new_start = nxt.start - 0.15
        # 침범 판정: 다음 라인이 발성 위(정상 배치)면 그 앞을 침범 못 하게 막는다. 다음 라인도
        # 무음에 좌초(곧 뒤로 스냅)면 현재 위치로 막지 않는다 — 좌초 '쌍/블록'의 첫 줄이 아직
        # 안 옮겨진 둘째 줄에 막혀 스킵되던 버그(리프라이즈 1번째 줄만 잔존) 방지.
        nl = results[i + 1] if i + 1 < len(results) else None
        next_line_start = (
            float("inf") if (nl is None or _cover(nl) < 0.25) else nl.start_time
        )
        if new_start >= next_line_start:
            continue  # 정상 배치된 다음 라인을 침범 → 오탐, 적용하지 않음
        new_start = max(new_start, last_snapped + 0.05)  # 연속 좌초 라인 순서 유지(겹침 방지)
        if new_start >= next_line_start:
            continue
        if e <= nxt.start:
            # 라인 전체가 무음에 갇힘 → 길이 보존하고 통째로 온셋으로 이동(다음 라인 앞까지)
            r.start_time = new_start
            r.end_time = min(new_start + dur, next_line_start)
            if r.word_segments:
                _shift_word_segments(r.word_segments, r.start_time, r.end_time)
        else:
            r.start_time = new_start  # 시작만 무음, 라인이 이미 리전에 걸침 → start만 스냅
        last_snapped = r.start_time
        clamped.add(i)


def _snap_post_interlude_leak(
    results,
    vad_result,
    clamped: set[int],
    min_gap_sec: float,
    min_char_rate: float,
    max_coverage: float = 0.3,
) -> None:
    """긴 간주 앞에 붕괴·밀집한 리프라이즈 블록을 간주 이후 발성에 재배치한다 (ja-free).

    합성보컬은 CTC posterior가 균일 바닥이라 음향 앵커가 전멸해, 긴 간주 뒤 리프라이즈
    블록이 간주를 통째로 건너뛰어 앞으로 붕괴한다 (熱異常 실측: 32.7s 간주 132.8→165.5
    직전에 리프라이즈 선두 idx51-52가 129.9/130.7로 크램, 이후 블록은 gap_end엔 닿았으나
    압축돼 중앙값 -14.75s). ja 대조(_leaked_runs)는 ja도 같이 붕괴해 무력하므로, 간주(무음)
    라는 하드 음향 사실에 앵커한다 — 유일하게 posterior에 오염되지 않은 신호다.

    각 긴 간주 [gs,ge]마다:
      1. gap_end 직후 첫 발성에 정착한 라인 k를 찾고, 그 앞에서 **VAD 발성 커버리지가 낮은
         (max_coverage 미만 = 무음 위에 떠 있는)** 라인만 역추적해 누출 클러스터로 삼는다.
         **발성 위에 정상 배치된 라인은 절대 이동하지 않는다** — 이 커버리지 게이트가 정상/누출을
         가르는 1차 신호다 (消失는 초고속이라 정상 라인도 간격<1.5s·고밀도 → 간격/char-rate
         만으로는 정상 라인을 오판, 발성 위 라인까지 밀어 회귀시켰다).
      2. 2차: 클러스터 중 한 줄이라도 불가능한 char rate(글자수/지속, >min_char_rate)여야 한다
         (강제정렬이 풀 가사를 0초 슬롯에 욱여넣은 잔해 신호).
      3. 클러스터부터 다음 간주(없으면 곡 끝) 전까지의 블록을 간주 이후 발성 리전들에
         리전 길이 비례로 재배치하되, **어떤 라인도 (간주 길이+여유) 이상 이동하지 않도록**
         상한을 두고 단조성을 유지한다.

    발성 위에 정상 배치된 라인들(消失 가창, 커버 등)은 클러스터가 비어 무변경.
    """
    if min_gap_sec <= 0:
        return
    regions = sorted(vad_result.regions, key=lambda reg: reg.start)
    if len(regions) < 2:
        return
    n = len(results)
    interludes = [
        (a.end, b.start) for a, b in zip(regions, regions[1:]) if b.start - a.end >= min_gap_sec
    ]
    if not interludes:
        return

    def _coverage(i: int) -> float:
        r = results[i]
        dur = max(1e-6, r.end_time - r.start_time)
        ov = sum(
            max(0.0, min(reg.end, r.end_time) - max(reg.start, r.start_time)) for reg in regions
        )
        return ov / dur

    def _char_rate(i: int) -> float:
        dur = max(0.1, results[i].end_time - results[i].start_time)
        return len(re.sub(r"\s", "", results[i].text or "")) / dur

    for idx, (gs, ge) in enumerate(interludes):
        seg_end = interludes[idx + 1][0] if idx + 1 < len(interludes) else float("inf")
        k = next((i for i in range(n) if results[i].start_time >= ge - 2.0), None)
        if not k:  # None 또는 0 — 간주 앞 라인이 없음
            continue
        # 1차 게이트: 발성 커버리지<max_coverage(무음 위)인 라인만 누출로. 발성 위(정상) 라인을
        # 만나면 멈춘다 — 이 커버리지 경계가 클러스터 범위를 정한다(간격 신호는 消失 초고속에서
        # 정상 라인을 오판하므로 쓰지 않는다).
        cluster: list[int] = []
        j = k - 1
        while j >= 1 and results[j].start_time < ge - 0.5 and _coverage(j) < max_coverage:
            cluster.append(j)
            j -= 1
        if not cluster:
            continue
        first_leaked = min(cluster)
        if results[first_leaked].start_time > ge - min_gap_sec:
            continue  # 뒤로 크게 밀리지 않음 — 대량 누출 아님
        # 2차 게이트: 불가능한 char rate (정상 빠른 구간 추가 방어)
        if max(_char_rate(i) for i in cluster) < min_char_rate:
            continue
        block_last = k
        while block_last + 1 < n and results[block_last + 1].start_time < seg_end:
            block_last += 1
        post_regions = [
            (reg.start, reg.end) for reg in regions if reg.start >= ge - 0.5 and reg.start < seg_end
        ]
        block = list(range(first_leaked, block_last + 1))
        m = len(block)
        vocal = sum(e - s for s, e in post_regions)
        if not post_regions or m < 2 or vocal <= 0:
            continue

        def _vocal_time(frac: float) -> float:
            target = frac * vocal
            acc = 0.0
            for s, e in post_regions:
                d = e - s
                if acc + d >= target:
                    return s + (target - acc)
                acc += d
            return post_regions[-1][1]

        # 이동량 상한(간주 길이+여유)과 단조성을 지키며 발성 리전에 비례 재배치
        move_cap = (ge - gs) + 5.0
        prev_start = -1.0
        for pos, i in enumerate(block):
            new_start = min(_vocal_time(pos / m), results[i].start_time + move_cap)
            new_start = max(new_start, prev_start)
            new_end = max(_vocal_time((pos + 1) / m), new_start + 0.1)
            prev_start = new_start
            results[i].start_time = new_start
            results[i].end_time = new_end
            if results[i].word_segments:
                _shift_word_segments(results[i].word_segments, new_start, new_end)
            clamped.add(i)
        logger.warning(
            f"Mass post-interlude leak at [{gs:.1f}-{ge:.1f}]s: re-spaced {m} lines "
            f"(idx {first_leaked}..{block_last}) across {vocal:.1f}s of post-interlude vocal; "
            f"leaked cluster {sorted(cluster)}"
        )


def _shift_word_segments(word_segments, new_start: float, new_end: float) -> None:
    """word_segments를 [new_start, new_end] 구간으로 선형 리스케일(제자리)."""
    if not word_segments:
        return
    old_start = word_segments[0].start
    old_end = word_segments[-1].end
    span = old_end - old_start
    target = new_end - new_start
    if span <= 0:
        n = len(word_segments)
        step = target / n if n else 0.0
        for k, w in enumerate(word_segments):
            w.start = new_start + step * k
            w.end = new_start + step * (k + 1)
        return
    for w in word_segments:
        w.start = new_start + (w.start - old_start) / span * target
        w.end = new_start + (w.end - old_start) / span * target


def _geomean(values: list[float]) -> float | None:
    xs = [v for v in values if v is not None and v > 0]
    if not xs:
        return None
    return math.exp(sum(math.log(v) for v in xs) / len(xs))


def _avg_line_confidence(results) -> float | None:
    """곡 단위 평균 라인 신뢰도 (quality_score와 동일 규칙: 라인 conf 또는 글자 conf 기하평균)."""
    vals: list[float] = []
    for r in results:
        c = r.confidence
        if c is None and r.word_segments:
            c = _geomean([w.confidence for w in r.word_segments])
        if c is not None:
            vals.append(c)
    return sum(vals) / len(vals) if vals else None


def _dual_align_should_run(ko_conf: float | None, threshold: float) -> bool:
    """ko 정렬 평균 신뢰도가 임계 미만이면 True — ja 교차정렬을 돌릴지의 비용 게이트.

    threshold<=0(비활성)이거나 ko_conf가 임계 이상이면 두 번째 정렬을 아예 돌리지 않는다.
    """
    return threshold > 0 and ko_conf is not None and ko_conf < threshold


def _original_align_needed(ko_conf: float | None, dual_conf: float, fuse_enabled: bool) -> bool:
    """독음(ko) 경로에서 원문(ja) 정렬을 한 번 돌려야 하는지 — 두 소비자의 합집합 비용 게이트.

    ① 융합(``fuse_enabled``)은 라인 내부 원문 글자 분포를 ja 실측으로 갈아끼우므로 **상시**
       ja가 필요하다, ② 이중정렬 안전망은 ko가 저신뢰일 때만 필요하다(``_dual_align_should_run``).
    한 번 돌린 결과를 역누출 가드까지 셋이 공유하므로 정렬 패스는 최대 1회다.
    """
    return bool(fuse_enabled) or _dual_align_should_run(ko_conf, dual_conf)


# ── 어댑터 vocab 크기와 conf 스케일 보정 ──────────────────────────────
#
# CTC 라인 conf는 프레임별 posterior의 기하평균이라 **어댑터 vocab 크기에 직접 의존한다** —
# vocab이 크면 확률 질량이 더 많은 토큰으로 흩어져 같은 정렬 품질에서도 conf가 낮아진다.
# 실측(dQw4w9WgXcQ, 같은 오디오·같은 가사, 어댑터만 교체):
#     eng(vocab 154)  conf 0.1289, match_rate 0.9979
#     kor(vocab 1330) conf 0.0492, match_rate 1.0000   ← 잔차는 **동일**, 매칭률은 오히려 상승
# 즉 2.6배 하락은 품질 저하가 아니라 순수한 스케일 차이다.
#
# 스케일 모델: conf ≈ V^(-α)로 보면 α = -ln(conf)/ln(V)가 vocab에 무관한 "첨예도" 지표다.
# 위 실측에서 α_eng = 2.0490/5.0370 = 0.4068, α_kor = 3.0119/7.1929 = 0.4187 — 2.9% 차이로
# 일치한다. 반면 선형 모델(conf ∝ 1/V)은 kor conf를 0.0149로 예측해 실측 0.0492와 3.3배
# 틀린다. 그래서 로그 스케일 보정을 쓴다. (근거는 실측 1쌍이므로 **판정을 뒤집는 새 임계를
# 도입하지 않고**, 어댑터가 같을 때 항등이 되는 보정으로만 쓴다 — 이번 세션에 이미 "곡 단위
# conf 임계로 정상 곡을 오폭한" 사고가 있었다.)
#
# vocab 크기는 facebook/mms-1b-all tokenizer vocab.json 실측값이며
# tests/fixtures/mms_adapter_script_census.json이 원본, test_gloss_and_conf_scale.py가 고정한다.
_ADAPTER_VOCAB_SIZE: dict[str, int] = {
    "eng": 154,
    "kor": 1330,
    "jpn": 2268,
    "cmn-script_simplified": 4495,
}


def _conf_alpha(conf: float | None, adapter: str | None) -> float | None:
    """어댑터 스케일에 무관한 첨예도 α = log_V(1/conf). 작을수록 좋다 (α=1이면 균일=우연 수준).

    vocab 크기를 모르는 어댑터거나 conf가 없으면 None (호출부는 원래 값으로 폴백한다).
    """
    size = _ADAPTER_VOCAB_SIZE.get(adapter or "")
    if conf is None or conf <= 0 or not size or size <= 1:
        return None
    return -math.log(conf) / math.log(size)


def _scale_free_quality(conf: float | None, adapter: str | None) -> float | None:
    """곡 간 비교가 가능한 스케일 무관 품질 e^(-α) (0~1, 클수록 좋다). 보고 전용.

    저장되는 ``quality_score``는 **원본 conf 그대로 유지한다** — 확장이 0.001 고정 임계로
    저신뢰 경고를 띄우고 있어 스케일을 갈아끼우면 그 경고가 조용히 죽는다. 대신 이 값을
    debug 메타로 함께 내려보내 어댑터가 다른 곡끼리도 비교할 수 있게 한다.
    """
    alpha = _conf_alpha(conf, adapter)
    return None if alpha is None else math.exp(-alpha)


# ── 정렬 커버리지: "정렬이 아예 안 됐는데 경고도 안 뜨는" 실패 차단 ──────
#
# 정렬된 글자가 0개인 줄은 ctc_engine이 [None, None, None]으로 남기고
# ``_interpolate_unaligned``가 앞뒤 줄 사이로 보간해 채우며, **전 줄이 실패하면 전체 구간에
# 균등 분배**한다 (ctc_engine.py의 "OOV 등으로 정렬된 글자가 0개 → 아래에서 이웃 사이로 보간"
# / "전부 실패면 전체 구간에 균등 분배"). 결과 타이밍은 그럴듯하게 생겼지만 오디오 근거가 0이다.
#
# 그런데 그 줄들은 ``word_segments=None``이라 라인 conf가 하나도 없어 avg_confidence=None →
# ``quality_score=None``이 되고, 확장의 저신뢰 경고는
#     data.qualityScore != null && data.qualityScore < 0.001
#     (everyric2-chrome/src/content.ts:649, 1197)
# 을 요구하므로 **발화하지 않는다**. 사용자는 균등 타이밍 + 무경고 + "생성 성공"을 본다 —
# 안 된 것을 됐다고 말하는, 이 프로젝트에서 가장 해로운 실패 형태다. 그래서 정렬이 실질적으로
# 실패했다는 사실을 결과(quality_score + debug)에 실어 보낸다.
#
# 판정은 conf 크기가 아니라 **정렬이 성립한 줄이 몇 개인가**로만 한다 — 곡 단위 conf 임계로
# 정상 곡을 오폭한 과거 사고(_ADAPTER_VOCAB_SIZE 위 주석 참고)를 되풀이하지 않기 위해서다.
#
# 하한 0.5는 튜닝값이 아니라 "표시 타이밍의 **과반**이 실측이 아니라 보간 산물"이라는 구조적
# 진술이다. 정상 곡은 커버리지가 1.0에 붙어 있어 이 값에 닿지 않는다.
# (설정으로 뺄 후보지만 settings.py는 이 작업 범위 밖이라 모듈 상수로 둔다.)
ALIGNED_LINE_RATIO_MIN = 0.5

# 커버리지 미달 곡에 싣는 quality_score. 0.0은 확장의 `< 0.001` 조건을 확실히 통과하고,
# `qualityScore: sync.quality_score ?? undefined`(background.ts:271)와 `!= null` 어느 쪽에도
# 걸려 사라지지 않는다(0은 null/undefined가 아니다). None으로 두면 경고가 죽는다.
FAILED_ALIGNMENT_QUALITY = 0.0


def _quality_with_coverage(
    measured_conf: float | None, aligned_lines: int, total_lines: int
) -> tuple[float | None, dict[str, Any]]:
    """(저장할 quality_score, 근거 메타) — 정렬 커버리지가 하한 미만이면 저신뢰로 확정한다.

    ``measured_conf``는 **정렬된 줄만의** 평균 conf다(실패 줄은 conf가 없어 분모에서 빠진다).
    그래서 40줄 중 2줄만 정렬돼도 그 2줄의 평균이 곡 점수로 올라가 실패가 감춰진다.
    커버리지가 ``ALIGNED_LINE_RATIO_MIN`` 미만이면 quality_score를 확정 저신뢰
    (``FAILED_ALIGNMENT_QUALITY``)로 덮고, 원래 측정값은 근거 메타에 남겨 버리지 않는다.
    하한 이상이면 측정값을 그대로 돌려준다 — 정상 곡의 저장값은 한 치도 바뀌지 않는다.
    """
    ratio = (aligned_lines / total_lines) if total_lines else 0.0
    meta: dict[str, Any] = {
        "aligned_lines": aligned_lines,
        "total_lines": total_lines,
        "ratio": round(ratio, 4),
        # 커버리지 미달로 quality_score를 덮었을 때 원래 측정값(정렬된 줄만의 평균)
        "measured_conf": None if measured_conf is None else round(measured_conf, 6),
    }
    if total_lines and ratio >= ALIGNED_LINE_RATIO_MIN:
        return measured_conf, meta
    meta["failed"] = True
    return FAILED_ALIGNMENT_QUALITY, meta


def _rescale_conf(conf: float | None, from_adapter: str | None, to_adapter: str | None):
    """``from_adapter``로 측정한 conf를 ``to_adapter`` 스케일로 옮긴다 (α 보존).

    어댑터가 같거나 vocab 크기를 모르면 **항등** — 어댑터가 일치하는 경우(영어 곡은 이제
    ko/ja 양쪽이 kor 어댑터다)의 동작은 한 치도 바뀌지 않는다.
    """
    alpha = _conf_alpha(conf, from_adapter)
    to_size = _ADAPTER_VOCAB_SIZE.get(to_adapter or "")
    if alpha is None or not to_size or to_size <= 1:
        return conf
    if _ADAPTER_VOCAB_SIZE.get(from_adapter or "") == to_size:
        return conf
    return math.exp(-alpha * math.log(to_size))


def _dual_align_prefers_original(ko_conf: float, ja_conf: float | None, min_ratio: float) -> bool:
    """ja 평균 신뢰도가 ko의 min_ratio배 이상이면 True(원문 채택). 바닥 근처 노이즈로

    뒤집히지 않게 명확한 마진을 요구한다 (熱異常: ja도 같이 붕괴 → 채택 안 됨).
    """
    return ja_conf is not None and ja_conf >= ko_conf * min_ratio


def _spread_uncovered_words(
    out: list[dict[str, Any]],
    covered: list[bool],
    line_start: float | None,
    line_end: float | None,
    outer_per_char: float = 0.3,
) -> None:
    """타이밍이 없는 글자(covered=False) 구간을 앞뒤 앵커 사이에 글자 수 비례로 분배(제자리).

    예전에는 이런 글자를 전부 ``start = end = prev_end``인 **길이 0 점**으로 채웠다. 독음
    정렬 곡은 역매핑 실패로 원문 글자의 55~69%가 이 상태였고, 3글자 이상이 같은 시각에
    시작하는 비율이 38~59%에 달해 화면에서 원문이 뭉텅이로 한꺼번에 점등했다
    (대조군인 원문 정렬 곡은 11% / 2%).

    앞 앵커는 직전 글자의 end, 뒤 앵커는 다음 글자의 start다. 라인 선두/말미 구간은 안쪽
    앵커가 한쪽뿐이라 라인 경계(line_start/line_end)를 바깥 앵커로 쓰되, 라인 경계는
    클램프·스냅이 잡은 값이라 토큰 스팬과 크게 어긋날 수 있어 **글자당 outer_per_char초**
    까지만 빌린다(라인 끝이 30초 뒤인 클램프 라인에서 말미 부호 하나가 30초 점등하는 것을
    막는다). 바깥 앵커가 없으면 예전대로 길이 0으로 둔다.
    타이밍 단조성은 유지된다 — 뒤 앵커가 앞 앵커보다 이르면(원 토큰이 비단조) 구간을
    앞 앵커에 붙인 길이 0으로 접어 예전 동작과 같아진다.
    """
    n = len(out)
    i = 0
    while i < n:
        if covered[i]:
            i += 1
            continue
        j = i
        while j < n and not covered[j]:
            j += 1
        length = j - i
        lo = out[i - 1]["end"] if i > 0 else None
        hi = out[j]["start"] if j < n else None
        if lo is None and hi is None:
            # 라인 전체가 미커버(토큰이 전부 스퓨리어스) — 안쪽 앵커가 아예 없으니 라인 스팬
            # 전체에 균등 분배한다(_resynth_word_segments와 같은 폴백). 상한을 두면 라인
            # 앞머리에 전부 몰려 blip이 그대로 남는다.
            lo = line_start if line_start is not None else 0.0
            hi = max(line_end, lo) if line_end is not None else lo
        elif lo is None:  # 라인 선두 구간 — 라인 시작에서 뒤로 빌린다
            lo = hi if line_start is None else max(line_start, hi - outer_per_char * length)
            lo = min(lo, hi)
        elif hi is None:  # 라인 말미 구간 — 라인 끝까지 앞으로 빌린다
            hi = lo if line_end is None else min(line_end, lo + outer_per_char * length)
            hi = max(hi, lo)
        else:
            hi = max(hi, lo)  # 원 토큰이 비단조면 길이 0으로 접는다(예전 동작)
        span = hi - lo
        for k in range(length):
            out[i + k]["start"] = lo + span * k / length
            out[i + k]["end"] = lo + span * (k + 1) / length
        i = j


def _full_coverage_words(
    text: str,
    word_segments,
    line_start: float | None = None,
    line_end: float | None = None,
) -> list[dict[str, Any]]:
    """라인 본문 글자를 1:1 완전히 덮는 words 목록 (직렬화용).

    정렬 word_segments는 정규화 텍스트 기준이라 본문의 공백·문장부호·표기 차이 글자를
    빠뜨려, ''.join(words)가 본문과 어긋나면 확장의 글자 매핑(indexOf)이 죽고 라인이 통짜로
    점등된다. 여기서 words[].word를 순서대로 이으면 **정확히 본문**이 되도록 재구성한다:
      - 방출하는 word는 항상 본문 부분문자열이라 join(words)==text가 구조적으로 보장된다.
      - 정렬 토큰이 덮는 글자는 그 토큰의 타이밍/신뢰도를 상속,
      - 토큰이 못 덮는 글자(공백·부호·표기 차이·괄호 독음 등)는 앞뒤 앵커 사이에 글자 수
        비례로 분배(confidence=None, _spread_uncovered_words). 토큰이 본문에 안 나타나면
        (표기 차이) 스퓨리어스로 버려 그 글자를 다음 토큰과 재평가한다. 타이밍은 단조 비감소.
    line_start/line_end를 주면 라인 선두·말미의 미커버 구간이 그 경계까지(글자당 상한 이내)
    퍼진다. 안 주면 예전처럼 안쪽 앵커에 붙은 길이 0이 된다.
    pron_segments는 건드리지 않는다(별도 부착).
    """
    tokens = [w for w in (word_segments or []) if w.word]
    if not tokens:
        return []
    out, covered = _scan_token_coverage(text, tokens)
    _spread_uncovered_words(out, covered, line_start, line_end)
    return out


def _scan_token_coverage(text: str, tokens) -> tuple[list[dict[str, Any]], list[bool]]:
    """본문 글자를 정렬 토큰에 순서대로 매칭 → (글자별 out 엔트리, 토큰 상속 여부).

    ``_full_coverage_words``의 매칭 스캔 본체. 융합 판정(``_measured_anchor_count``)이
    "어느 정렬의 토큰이 본문에서 실측 앵커를 더 많이 만드는가"를 **같은 규칙으로** 재려고
    공유한다 — 규칙이 갈라지면 비교가 실제 직렬화 결과와 어긋난다.
    """
    out: list[dict[str, Any]] = []
    covered: list[bool] = []  # out[k]가 정렬 토큰에서 타이밍을 상속했는지
    ti, wi, n, m = 0, 0, len(text), len(tokens)
    while ti < n:
        tk = tokens[wi] if wi < m else None
        if tk is not None and text.startswith(tk.word, ti):
            out.append(
                {
                    "word": text[ti : ti + len(tk.word)],
                    "start": tk.start,
                    "end": tk.end,
                    "confidence": tk.confidence,
                }
            )
            covered.append(True)
            ti += len(tk.word)
            wi += 1
        elif tk is not None and text.find(tk.word, ti, ti + len(tk.word) + 4) < 0:
            wi += 1  # 표기 차이 스퓨리어스 토큰 — 버리고 같은 글자를 다음 토큰과 재평가
        else:
            # 토큰이 못 덮는 글자 → 호출부가 앞뒤 앵커 사이에 비례 분배
            out.append({"word": text[ti], "start": None, "end": None, "confidence": None})
            covered.append(False)
            ti += 1
    return out, covered


def _measured_anchor_count(text: str, word_segments) -> int:
    """본문 ``text`` 위에서 이 정렬이 만드는 **서로 다른 시작 시각**의 개수 (직렬화와 동일 규칙).

    라인 내부 카라오케 해상도는 '몇 글자를 덮었나'가 아니라 '라인 안에 서로 다른 시각이 몇
    개 있나'다. 역매핑은 한 발음 음절의 스팬을 그 음절이 걸친 원문 글자 전부에 **그대로
    복사**하므로 글자 커버리지는 100%여도 같은 시각이 3~5개씩 겹친다 — 이것이 실측에서
    ko 정렬 곡의 3자 이상 동시 시작 38~59%(ja 정렬 곡 2%)로 나타난 그 현상이고, 화면에서
    원문이 뭉텅이로 점등하는 직접 원인이다. 그래서 글자 수가 아니라 이 값으로 잰다.
    본문에 안 나타나는 토큰은 직렬화에서 버려지므로(스퓨리어스) 여기서도 안 센다.
    """
    toks = [w for w in (word_segments or []) if w.word]
    if not toks:
        return 0
    out, covered = _scan_token_coverage(text, toks)
    return len({o["start"] for o, c in zip(out, covered) if c})


def _resynth_word_segments(word_segments, start: float, end: float) -> None:
    """word_segments 시간을 라인 [start,end]에 글자 수 균등 비례로 재합성(제자리, confidence=None).

    붕괴 곡은 라인 내부 CTC 분포(글자 뭉침 등)가 무의미하므로, 왜곡 분포를 선형 리스케일로
    보존하는 대신 글자 수 균등 비례로 다시 깐다. word 문자열은 그대로라 직렬화의
    _full_coverage_words 매핑은 동일하게 통과한다.
    """
    toks = [w for w in (word_segments or []) if w.word]
    total = sum(len(w.word) for w in toks)
    if total == 0 or end <= start:
        return
    span = end - start
    acc = 0
    for w in toks:
        w.start = start + span * acc / total
        acc += len(w.word)
        w.end = start + span * acc / total
        w.confidence = None


def _subdivide_clumped_words(results, skip=frozenset()) -> int:
    """역매핑이 한 음절 스팬을 여러 글자에 복사한 «뭉침»을 글자 수 균등으로 세분한다(제자리).

    동일 (start, end)를 공유하는 연속 글자 묶음마다 그 스팬을 균등 분할한다 — 묶음의
    첫 시작·마지막 끝(ko 실측 앵커)은 그대로이고 안쪽만 나뉜다. 뭉친 글자들이 동시에
    점등했다가 다음 묶음으로 순간이동하는 카라오케 체감(사용자 보고 2026-07-28)을 없애는
    표시 평활화이지 측정이 아니다 — 그래서 융합 가드(_measured_anchor_count)가 도는
    융합 **뒤에만** 불러야 한다(먼저 부르면 합성 앵커가 실측 앵커 수를 부풀려 융합을
    잘못 막는다). conf는 묶음 값을 유지한다. 반환: 세분이 일어난 줄 수.
    """
    changed = 0
    for i, r in enumerate(results):
        if i in skip or not r.word_segments:
            continue
        ws = r.word_segments
        touched = False
        j = 0
        while j < len(ws):
            k = j + 1
            while (
                k < len(ws)
                and abs(ws[k].start - ws[j].start) < 1e-9
                and abs(ws[k].end - ws[j].end) < 1e-9
            ):
                k += 1
            n = k - j
            if n > 1 and ws[j].end > ws[j].start:
                base, span = ws[j].start, (ws[j].end - ws[j].start) / n
                for m in range(n):
                    ws[j + m].start = base + span * m
                    ws[j + m].end = base + span * (m + 1)
                touched = True
            j = k
        if touched:
            changed += 1
    return changed


def _resynth_pron_segments(pron_segments, start: float, end: float) -> None:
    """pron_segments(음절 dict 목록) 시간을 [start,end]에 음절 수 균등 비례로 재합성(제자리)."""
    n = len(pron_segments or [])
    if n == 0 or end <= start:
        return
    span = end - start
    for k, s in enumerate(pron_segments):
        s["start"] = start + span * k / n
        s["end"] = start + span * (k + 1) / n


def _apply_caption_scaffold(
    results, pron_data, fixes, anchor_plan, song_conf, audio_sec: float, align_settings
) -> dict[str, Any]:
    """자막 스캐폴드 적용 — 붕괴 곡의 줄 시작을 자막 시각으로 고정한다 (결과 교체).

    제약(마스크) 경로가 두 번 실패한 뒤의 대안이고 근거·실패 이력은
    ``alignment/caption_scaffold.py`` 모듈 주석에 있다. 항상 판정 dict를 돌려준다 —
    발동하지 않은 경우에도 «왜»(skipped)를 남겨야 사후 감사가 된다(caption_anchors 규약).
    움직인 줄은 «scaffold» 보정 라벨을 받아 확장 디버그 레인에 고스트로 표시된다.
    """
    from everyric2.alignment.caption_scaffold import drift_seconds, scaffold_plan

    if anchor_plan is None:
        return {"applied": False, "skipped": "no_plan"}
    if not anchor_plan.line_spans:
        return {
            "applied": False,
            "skipped": anchor_plan.debug.get("span_skipped")
            or anchor_plan.debug.get("positive_skipped")
            or anchor_plan.debug.get("skipped")
            or "no_anchors",
        }
    spans = [(r.start_time, r.end_time) for r in results]
    drifts = sorted(drift_seconds(spans, anchor_plan.line_spans))
    drift_med = drifts[len(drifts) // 2] if drifts else None
    max_conf = float(getattr(align_settings, "caption_scaffold_max_conf", 0.002))
    min_drift = float(getattr(align_settings, "caption_scaffold_min_drift_sec", 3.0))
    by_conf = song_conf is not None and song_conf < max_conf
    by_drift = drift_med is not None and drift_med >= min_drift
    meta: dict[str, Any] = {
        "rate": anchor_plan.debug.get("rate"),
        "track": anchor_plan.debug.get("track"),
        "anchors": len(anchor_plan.line_spans),
        "drift_median": round(drift_med, 2) if drift_med is not None else None,
        "song_conf": round(song_conf, 6) if song_conf is not None else None,
        "gates": {"conf": by_conf, "drift": by_drift},
    }
    if not (by_conf or by_drift):
        meta.update({"applied": False, "skipped": "not_collapsed"})
        return meta

    plan = scaffold_plan(
        spans,
        anchor_plan.line_spans,
        audio_sec,
        float(getattr(align_settings, "caption_scaffold_tolerance_sec", 1.0)),
    )
    counts = {"caption": 0, "interp": 0, "kept": 0}
    moved = 0
    for i, (r, pl) in enumerate(zip(results, plan)):
        counts[pl.source] += 1
        if pl.source == "kept":
            continue
        r.start_time = pl.start
        r.end_time = pl.end
        # 줄 안 분포는 균등 재합성 — 붕괴 곡의 라인 내부 CTC 분포는 무의미하고,
        # 자막은 줄 해상도까지만 신뢰한다 (SRT 해상도 실측)
        _resynth_word_segments(r.word_segments, pl.start, pl.end)
        if pron_data is not None and (pron_data.get(i) or {}).get("pron_segments"):
            _resynth_pron_segments(pron_data[i]["pron_segments"], pl.start, pl.end)
        fixes.setdefault(i, []).append("scaffold")
        moved += 1
    meta.update({"applied": True, "moved": moved, "sources": counts})
    logger.warning(
        f"Caption scaffold applied: {moved} line(s) re-timed to caption baseline "
        f"(caption {counts['caption']}, interp {counts['interp']}, kept {counts['kept']}; "
        f"drift median {meta['drift_median']}s, song conf {meta['song_conf']}, "
        f"track {meta['track']} rate {meta['rate']})"
    )
    return meta


def _impossible_word_distribution(word_segments, start: float, end: float, max_char_rate: float) -> bool:
    """라인 내부 word 분포가 물리적으로 불가능한지 — CTC 잔해 판별 (구조 신호).

    붕괴 곡에서는 라인 conf가 위치 신호를 갖지 않으므로(실측 corr(라인 conf,|잔차|)=-0.19)
    conf 대신 구조로 가른다:
      ① 글자 뭉침 — 글자들이 라인 폭의 절반도 못 덮으면서 실효 발화 속도가
         max_char_rate를 넘는다(강제정렬이 글자를 한 구석에 욱여넣은 잔해).
         라인 자체가 짧은 초고속 랩은 글자가 라인 폭을 다 덮으므로 걸리지 않는다.
      ② 경계 이탈 — word가 라인 [start,end] 밖을 1초 넘게 가리킨다
         (경계는 스냅/가드가 잡은 값이라 내부가 밖을 가리키면 잔해).
      ③ 선두 글자 고립 — 첫 글자와 둘째 글자 사이가 라인 폭의 40%를 넘는다.
         동일 음절이 연속되면(실측 XKZIQlqVjjk 코러스: ``Approved``×4 → 독음
         ``어프루브드``×4) posterior가 평평해져 **선두 1글자만 일찍 걸리고 나머지 34자가
         뒤에 몰린다**. 이때 글자 폭은 라인 폭을 다 덮으므로 ①의 뭉침 검사가 통과해버린다.
         라인 conf(2.4e−05)는 곡 평균 임계(0.002)에 걸리지 않아 어디에서도 안 잡혔다.
         정상 라인은 글자가 고르게 퍼져(4글자 이상이면 균등 간격 ≤25%) 이 지표에 안 걸린다.
    """
    if max_char_rate <= 0:
        return False
    toks = [w for w in (word_segments or []) if w.word]
    if len(toks) < 4:
        return False
    first = min(w.start for w in toks)
    last = max(w.end for w in toks)
    if first < start - 1.0 or last > end + 1.0:
        return True
    span = end - start
    if span <= 0:
        return False
    if toks[1].start - toks[0].start > span * 0.4:
        return True  # ③ 선두 글자만 앞에 고립 — 나머지가 뒤로 몰린 잔해
    # ④ 낱말이 프레임 한두 개에 눌린 것 — ①은 라인 전체 폭만 보므로, 글자가 라인 폭을 다
    #    덮으면서 그 **안에서** 뭉친 경우를 통과시킨다. 실측(XKZIQlqVjjk `Approved`×4):
    #    구간이 [3.3, 0.02, 0.02, 0.02]초 — 8글자 낱말이 0.02초, 즉 **프레임 1개**에 소진된다.
    #    코퍼스의 반복 구절 편차 분포가 p95 103배·p100 165배로 꼬리가 긴데 그 꼬리가 전부
    #    이 형태다.
    #
    #    판정을 **비율이 아니라 프레임 수**로 하는 이유: max_char_rate(11글자/초)를 낱말
    #    단위로 적용해 봤더니 정상 가창을 잡았다(실측 오탐: `消去しても` 5글자/0.44s =
    #    11.4글자/초, 일본어 빠른 줄 11~15글자/초 다수). 그 기준값은 *라인 전체*가 0에 가까운
    #    슬롯에 뭉친 것을 재려고 정해진 값이라 낱말 단위에서는 정상 범위와 겹친다.
    #    반면 "여러 글자가 프레임 1~2개 안에 있다"는 것은 조정할 여지가 없는 물리적 불가능이다
    #    — 모델은 한 프레임에서 그 글자들을 볼 수 없다. 오탐들은 21~114프레임이라 분리가 크다.
    if _word_pressed_into_a_frame(toks):
        return True
    width = last - first
    if width >= span * 0.5:
        return False
    n_chars = sum(len(w.word) for w in toks)
    return width <= 0 or n_chars / width > max_char_rate


# CTC 프레임은 20ms다 — 두 프레임(40ms)까지를 "한 순간"으로 보고 여유 5ms를 둔다.
# 조정용 손잡이가 아니다: 이 값을 키우면 정상 가창을 잡기 시작한다(호출부 ④ 주석 참조).
_MIN_WORD_FRAMES_SEC = 0.045

# 눌린 낱말이 라인 글자의 이만큼을 차지할 때만 재합성한다.
# 존재 여부로 판정하면 안 되는 이유(실측): `All in my heart その期待感`에서 기능어 `in`
# 하나가 1프레임을 받는데, 그것 때문에 라인 전체를 균등 재합성하면 나머지 글자들의
# **맞는** CTC 분포를 버린다. 기능어가 삼켜지는 것은 흔하고 그 자체로 해롭지 않다.
# 반면 원래 문제(`Approved`×4 중 3회가 각 1프레임)는 라인의 75%가 시각을 못 받은 것이라
# 성질이 다르다. 1/3은 그 둘 사이를 가르는 값이고, 코퍼스 실측으로 확인한다.
_PRESSED_SHARE_MIN = 1.0 / 3.0


def _word_pressed_into_a_frame(toks) -> bool:
    """공백으로 갈린 낱말 중 **여러 글자가 프레임 한두 개에 눌린** 것이 있는가.

    글자 토큰을 공백에서 끊어 낱말로 묶고, 낱말이 차지한 구간이 프레임 두 개(약 45ms)
    이하인지 본다. 한 글자 낱말은 세지 않는다 — 짧은 조사·감탄사가 순간적으로 지나가는 것은
    정상이고 그것까지 잡으면 정상 라인을 재합성하게 된다.

    임계가 프레임 수인 근거는 호출부 ④ 주석에 있다: 비율(글자/초) 기준은 낱말 단위에서
    정상 가창과 겹쳤고(실측 오탐 다수), 프레임 기준은 실측 잔해(1프레임)와 정상(21~114
    프레임) 사이가 한 자릿수 배가 아니라 열 배 이상 벌어져 있다.
    """
    groups: list[list] = [[]]
    for w in toks:
        if not w.word.strip():
            if groups[-1]:
                groups.append([])
            continue
        groups[-1].append(w)
    total = pressed = 0
    for g in groups:
        n = sum(len(w.word) for w in g)
        if not n:
            continue
        total += n
        if n < 2:
            continue  # 한 글자 낱말은 세지 않는다 (위 독스트링)
        if max(w.end for w in g) - min(w.start for w in g) <= _MIN_WORD_FRAMES_SEC:
            pressed += n
    return total > 0 and pressed / total >= _PRESSED_SHARE_MIN


def _measured_vocal_window(
    pron_segments, start: float, end: float, n_chars: int, max_char_rate: float
) -> tuple[float, float] | None:
    """그 라인에서 **발음 음절이 실제로 차지한 구간** [첫 음절 start, 마지막 음절 end].

    원문 글자 융합(``_fuse_original_char_timing``)의 사상 목표다. 라인 경계 [start,end]를
    목표로 쓰면 안 되는 이유: ko 라인의 **끝은 실제 발성 종료보다 늦은 경우가 많다** (끝음
    연장 tail 보정, 다음 줄까지의 여백, 클램프가 남긴 마진). 그 경계에 선형 사상하면 원문
    글자가 그 여백까지 늘어나는데, 발음 음절(pron_segments)은 ko CTC 실측 시각 그대로라
    늘어나지 않는다 — 두 레인이 벌어져 화면에서는 "원문만 늦게 점등"으로 보인다.
    실측(재생성 6곡, 같은 라인 안에서 원문 글자 분위 시각 − 발음 음절 분위 시각): 여섯 곡
    전부 부호가 양수이고 **첫 글자는 차이 0인데 뒤로 갈수록 벌어진다** (25% +0.00~+0.29,
    75% +0.09~+0.44, 끝 +0.09~+0.39) — 라인 끝 여백까지 늘리는 선형 사상의 지문이다.
    발음 음절 구간에 사상하면 두 레인이 같은 봉투를 공유해 나란히 흐른다. 라인 경계 자체는
    표시·스크롤·노트가 걸려 있는 검증된 값이라 **불변**이고, 바뀌는 것은 라인 내부뿐이다.

    None(호출부는 기존처럼 라인 경계로 사상)인 경우:
      - 발음 음절이 없거나 시간이 없다 — 애초에 융합 대상이 아닌 한국어/영어 곡, 커버리지
        0.9 게이트를 통과했지만 발음 표기가 빠진 일부 라인, 누출 스플라이스가 무효화(None)한
        라인,
      - 음절 구간이 라인 경계 밖 1초를 넘는다 — 라인은 스냅/클램프로 옮겨졌는데
        pron_segments는 (results와 별개 자료구조라 함께 움직이지 않는다) 원래 자리에 남은
        stale 값이다. 1초 기준은 ``_impossible_word_distribution``의 경계 이탈 판정과 같은 잣대,
      - 라인 경계와 교차한 폭이 0 이하,
      - 그 폭에 라인 글자를 넣으면 max_char_rate를 넘는다 — 음절이 한두 개만 살아남아 구간이
        지나치게 좁은 경우('구간을 못 정하는 라인'). 그대로 쓰면 융합 결과가 곧바로 다음
        단계의 불가능 뭉침 게이트(``_impossible_word_distribution`` ①)에 걸려 균등 분배로
        덮이므로, 좁은 창에 욱여넣는 대신 기존 폴백을 택한다.
    """
    times = [
        (s["start"], s["end"])
        for s in (pron_segments or [])
        if s.get("start") is not None and s.get("end") is not None
    ]
    if not times:
        return None
    lo = min(t[0] for t in times)
    hi = max(t[1] for t in times)
    if lo < start - 1.0 or hi > end + 1.0:
        return None
    t0, t1 = max(lo, start), min(hi, end)
    if t1 <= t0:
        return None
    if max_char_rate > 0 and n_chars / (t1 - t0) > max_char_rate:
        return None
    return (t0, t1)


def _fuse_original_char_timing(
    results,
    ja_results,
    fixes,
    max_char_rate: float = 0.0,
    pron_data=None,
    label: str = "fuse",
    max_disagreement: float = 0.0,
) -> set[int]:
    """ko 라인 경계는 그대로 두고 **라인 내부 원문 글자 분포만** ja 정렬 실측값으로 교체.

    독음(ko) 경로에서 원문 글자는 오디오를 한 번도 만지지 않는다 — 스팬이 '정렬된 한글
    음절 → 모라 → 원문 글자'의 3단 역매핑 합성물이라(``map_pron_alignment_to_line``),
    라인 경계가 완벽해도 **내부 분포**가 뭉친다(실측: ko 정렬 곡은 원문 글자의 3자 이상
    동시 시작이 38~59%, ja 정렬 곡은 2%). 라인 승자 선택으로는 안 풀리는 이유가 이것이다.

    그래서 라인 단위로 융합한다: ko의 [start,end]와 pron_segments는 **불변**이고, ja가 같은
    라인 인덱스에서 잰 글자 스팬을 그 라인 안으로 **선형 사상**해 word_segments로 갈아끼운다.
    사상 구간은 ja 토큰의 실제 extent[min start, max end] → 그 라인의 **발음 음절 구간**
    (``_measured_vocal_window``: 첫 음절 start ~ 마지막 음절 end)이다. ja 라인 경계가 아니라
    토큰 extent를 쓰므로 결과가 항상 그 구간 안에 들어오고 구간을 꽉 덮는다.
    목표가 ko 라인 경계가 **아닌** 이유는 라인 끝이 실제 발성 종료보다 늦은 경우가 많아
    (끝음 연장 tail, 다음 줄까지의 여백) 경계에 사상하면 원문 글자만 그 여백까지 늘어나
    실측 그대로인 발음 음절보다 뒤로 밀리기 때문이다 — 근거와 실측은 ``_measured_vocal_window``.
    발음 음절이 없거나 구간을 못 정하는 라인은 그 함수가 None을 주고, 여기서 **예전처럼
    라인 경계로 사상**한다(폴백). 어느 쪽이든 글자 스팬은 라인 경계 안에 들어온다.

    다음 라인은 융합하지 않고 기존 역매핑을 유지한다:
      - ja 토큰이 없거나 라인 텍스트가 대응하지 않음(인덱스 어긋남 방어),
      - ja가 그 라인에서 붕괴(``_impossible_word_distribution`` — 뭉침/경계 이탈/선두 고립),
      - ja가 그 라인에서 만드는 실측 앵커가 역매핑보다 **적음**(``_measured_anchor_count``) —
        한자가 OOV로 빠져 ja 토큰이 한두 개만 남은 라인에서 해상도가 오히려 내려가는 것을
        막는다. 글자 커버리지가 아니라 앵커 수로 재는 이유는 역매핑이 한 음절 스팬을 여러
        글자에 복사해 **커버리지 100% + 앵커 1개**를 만들기 때문이다 — 그 라인이야말로
        융합이 노리는 대상이라 커버리지로 재면 정확히 반대로 걸러진다.

    라인 conf(``r.confidence``)는 그대로 두고 글자 conf만 ja 실측값을 싣는다 — 그 라인의
    타이밍이 ja에서 왔으니 글자 conf도 ja가 맞다. 결과적으로 ko가 아무 스팬도 못 만든
    라인(``word_segments`` None)은 직렬화 백필을 통해 quality_score에 ja conf로 기여하게
    되는데, quality_score는 보고용이고 파이프라인 게이트(재합성의 곡 단위 conf)는 호출부가
    **융합 전에** 확정하므로 판정에는 영향이 없다.

    ja_results는 제자리 수정하지 않는다(WordSegment를 새로 만든다) — 누출 스플라이스가
    ja 객체를 results에 그대로 꽂아둔 경우가 있어 공유 객체를 흔들면 안 된다.
    반환: 융합된 라인 인덱스 집합. fixes에 ``label``을 남긴다(확장 디버그 표시).
    """
    from everyric2.inference.prompt import WordSegment

    fused: set[int] = set()
    if not ja_results or len(ja_results) != len(results):
        return fused
    for i, r in enumerate(results):
        ja = ja_results[i]
        if ja.text != r.text:
            continue
        toks = [w for w in (ja.word_segments or []) if w.word]
        if not toks:
            continue
        if _impossible_word_distribution(toks, ja.start_time, ja.end_time, max_char_rate):
            continue
        lo = min(w.start for w in toks)
        hi = max(w.end for w in toks)
        if hi - lo <= 0 or r.end_time - r.start_time <= 0:
            continue
        if _measured_anchor_count(r.text, toks) < _measured_anchor_count(
            r.text, r.word_segments
        ):
            continue
        t0, t1 = _measured_vocal_window(
            ((pron_data or {}).get(i) or {}).get("pron_segments"),
            r.start_time,
            r.end_time,
            sum(len(w.word) for w in toks),
            max_char_rate,
        ) or (r.start_time, r.end_time)
        scale = (t1 - t0) / (hi - lo)
        new = [
            WordSegment(
                word=w.word,
                start=t0 + (w.start - lo) * scale,
                end=t0 + (w.end - lo) * scale,
                confidence=w.confidence,
            )
            for w in toks
        ]
        # 단조 비감소 보장 — ja가 비단조여도 직렬화 계약(단조)이 깨지지 않게 접는다.
        prev = t0
        for w in new:
            w.start = max(w.start, prev)
            w.end = max(w.end, w.start)
            prev = w.end
        # ja 실측이 ko 역매핑과 **크게** 어긋나는 라인은 융합하지 않는다. ko 음절은 가수가
        # 실제로 낸 소리의 발음 텍스트를 kor 어댑터로 잰 값이고, ja는 한자 OOV 치환·희소
        # 토큰 위의 측정이라 둘이 크게 갈리면 ja가 틀린 쪽이었다(사용자 청취 2026-07-28 +
        # 오프라인 대조: JW3N-HvU0MA 융합 25줄 중 8줄이 글자 시작 중앙값 0.35s 초과,
        # p90 0.76s — 그 줄들이 정확히 「한글 전사가 더 정확한」 줄들이다). 해상도(뭉침
        # 해소)는 융합의 존재 이유지만 정확도가 먼저다 — 크게 어긋나면 뭉치더라도 ko
        # 실측에 정박한 역매핑을 지킨다. 작은 어긋남은 ja의 세밀함이 이득이라 통과시킨다.
        if max_disagreement > 0 and r.word_segments:
            ko_starts: dict[str, list[float]] = {}
            for w in r.word_segments:
                ko_starts.setdefault(w.word, []).append(w.start)
            deltas: list[float] = []
            for w in new:
                lst = ko_starts.get(w.word)
                if lst:
                    deltas.append(abs(w.start - lst.pop(0)))
            if deltas:
                deltas.sort()
                if deltas[len(deltas) // 2] > max_disagreement:
                    continue
        r.word_segments = new
        fused.add(i)
        labels = fixes.setdefault(i, [])
        if label not in labels:
            labels.append(label)
    return fused


def _synthesize_collapsed_timing(
    results, pron_data, fixes, song_conf, threshold, max_char_rate=0.0
) -> set[int]:
    """붕괴 곡/누출 라인의 라인 내부(word/pron) 타이밍을 균등 비례 합성으로 교체 (보정 마지막).

    CTC가 그럴듯하게 잡은 라인은 보존하고 잔해만 교체한다. 대상 = 합집합:
      (a) 곡 avg conf < threshold — 예비 스위치(기본 0=비활성): 전 라인,
      (b) fixes에 'leak' 라벨(스플라이스/폴백/대량 스냅으로 옮겨진) 라인 —
          이동된 내부 분포는 더 이상 오디오와 대응하지 않는다,
      (c) 내부 분포가 물리적으로 불가능한 라인(_impossible_word_distribution,
          max_char_rate>0일 때).
    라인 경계 [start,end]는 유지(스냅/가드가 잡아둔 값)하고 내부만 재합성한다. 이미 무효화된
    (None) pron_segments는 그대로 둔다. 반환: 재합성된 라인 인덱스 집합.
    """
    low_conf = threshold > 0 and song_conf is not None and song_conf < threshold
    targets = {
        i
        for i, r in enumerate(results)
        if low_conf
        or "leak" in fixes.get(i, [])
        or _impossible_word_distribution(
            r.word_segments, r.start_time, r.end_time, max_char_rate
        )
    }
    for i in targets:
        r = results[i]
        # 라인 conf가 word 기하평균 백필에 의존하면 재합성(word conf=None) 후 사라진다 —
        # 직렬화 백필과 동일하게 재합성 전에 라인 conf를 확정해 quality_score를 보존한다.
        if r.confidence is None and r.word_segments:
            r.confidence = _geomean([w.confidence for w in r.word_segments])
        _resynth_word_segments(r.word_segments, r.start_time, r.end_time)
        if pron_data:
            pd = pron_data.get(i)
            if pd and pd.get("pron_segments"):
                _resynth_pron_segments(pd["pron_segments"], r.start_time, r.end_time)
    return targets


def _star_swallowed_vocal(star_spans, vad_regions) -> float:
    """단일 star span이 실제 VAD 발성 구간을 삼킨 최대 겹침 길이(초).

    독음(ko) 정렬이 초고속/간주 구간에서 실패하면, 와일드카드 star(log 1.0)가 실제
    후반 가창을 통째로 흡수하고 그 라인들을 앞으로 압축한다 (VWVtIg5cdDU 실측: star
    한 개가 후반 가창 21s를 삼킴). 다만 이 값만으로는 '실가사 압축(VWV)'과 '가사 없는
    브릿지 정상 흡수(熱異常도 20.7s 삼키지만 배치 정상)'를 못 가른다. 그래서 이건
    비용 게이트(값이 크면 ja와 대조)로만 쓰고, 최종 판정은 간주 이후 발성 구간의
    라인 점유를 ko/ja 비교하는 호출부(_post_interlude_window)가 한다.
    """
    def ov(s, r):
        return max(0.0, min(s[1], r[1]) - max(s[0], r[0]))

    regions = [(reg.start, reg.end) for reg in vad_regions]
    return max((sum(ov(s, r) for r in regions) for s in star_spans), default=0.0)


def _post_interlude_window(vad_regions, min_gap_sec: float) -> tuple[float, float] | None:
    """최대 간주(무음 갭) 이후의 발성 창 [gap_end, last_vocal_end].

    간주는 오디오가 고정하는 구조라 star(정렬마다 위치 변동)보다 안정적인 앵커다.
    연속 VAD 리전 사이 최대 갭이 min_gap_sec 이상이면 그 갭 끝~마지막 발성 끝을
    '간주 이후 창'으로 돌려준다. 큰 간주가 없으면 None(폴백 판단 안 함).
    """
    regions = sorted((reg.start, reg.end) for reg in vad_regions)
    if len(regions) < 2:
        return None
    best_gap, gap_end = 0.0, None
    for (_, e0), (s1, _) in zip(regions, regions[1:]):
        if s1 - e0 > best_gap:
            best_gap, gap_end = s1 - e0, s1
    if gap_end is None or best_gap < min_gap_sec:
        return None
    return (gap_end, regions[-1][1])


def _lines_span_overlap(results, span: tuple[float, float]) -> float:
    """results 라인들이 [span] 구간과 겹친 총 길이(초) — 그 구간의 '라인 점유량'."""
    lo, hi = span
    return sum(max(0.0, min(r.end_time, hi) - max(r.start_time, lo)) for r in results)


def _splice_alignments(ko_results, ja_results, post_win: tuple[float, float]) -> int | None:
    """ko 정렬이 간주 이후 블록을 앞으로 압축했을 때 전곡 ja 폴백 대신 스플라이스한다.

    간주 전 라인은 ko(독음 음절 타이밍) 유지, ja가 간주 이후에 배치한 첫 라인(k)부터는
    ja 타이밍으로 교체한다 (ko_results를 제자리 수정). 가사 순서는 고정이고 CTC 정렬은
    라인 순서 단조라, ja 기준 '간주 이후 첫 라인' 인덱스가 곧 텍스트상 후반 블록의 시작이다.
    경계에서 ko 마지막 유지 라인이 ja 첫 교체 라인을 침범하면 끝을 클램프한다.

    반환: 교체 시작 인덱스 k. 스플라이스가 성립하지 않으면 None (ko_results 무변경):
      - ja가 간주 이후에 아무 라인도 안 둠 (가드 오발 — 호출부가 전곡 폴백)
      - k == 0 (전곡 교체 = 전곡 폴백과 동일하므로 기존 경로에 맡김)
      - ko 유지 구간(k 이전)에 경계를 넘는 라인이 있음 (압축이 간주를 걸침 — 부분 보존 불가)
    """
    gap_end = post_win[0]
    k = next(
        (i for i, r in enumerate(ja_results) if r.start_time >= gap_end - 0.5),
        None,
    )
    if not k:  # None 또는 0
        return None
    bound = ja_results[k].start_time
    if any(r.start_time >= bound for r in ko_results[:k]):
        return None
    for i in range(k):
        r = ko_results[i]
        if r.end_time > bound:
            r.end_time = max(r.start_time + 0.01, bound)
            if r.word_segments:
                _shift_word_segments(r.word_segments, r.start_time, r.end_time)
    ko_results[k:] = ja_results[k:]
    return k


def _post_interlude_windows(vad_regions, min_gap_sec: float) -> list[tuple[float, float]]:
    """min_gap_sec 이상의 **모든** 간주 뒤 발성 블록 창 목록 [gap_end, block_end] (시간순).

    ``_post_interlude_window``(최대 간주 하나만)의 확장판. 최대 갭만 앵커하면 두 번째로
    큰 간주 뒤의 대사 블록(初音ミクの消失 나레이션1: 간주1 43.76→59.84 뒤)이 가드
    범위 밖으로 빠진다. 각 간주 i의 창 끝은 다음 간주의 시작(없으면 마지막 발성 끝)이라
    간주 사이 블록이 서로 겹치지 않는다. 큰 간주가 없으면 빈 목록.
    """
    regions = sorted((reg.start, reg.end) for reg in vad_regions)
    if len(regions) < 2:
        return []
    interludes: list[tuple[float, float]] = []  # (gap_start, gap_end)
    for (_, e0), (s1, _) in zip(regions, regions[1:]):
        if s1 - e0 >= min_gap_sec:
            interludes.append((e0, s1))
    if not interludes:
        return []
    last_end = regions[-1][1]
    windows: list[tuple[float, float]] = []
    for i, (_gs, ge) in enumerate(interludes):
        block_end = interludes[i + 1][0] if i + 1 < len(interludes) else last_end
        if block_end - ge > 0:
            windows.append((ge, block_end))
    return windows


# 비가창 판정 임계 — 라인 구간에서 발성이 차지하는 비율이 이 값 이하면 "무음 위에 떠 있다".
# 간주 누출 판정(mass_leak_max_coverage=0.3)보다 훨씬 엄격하게 잡는다: 여기서는 라인을
# **삭제**하므로 애매한 것은 건드리지 않아야 한다.
_NONVOCAL_MAX_RATIO = 0.05
# 앞뒤 각각 이 개수까지만 훑는다 (크레딧은 몇 줄이고, 그 이상 지우는 건 다른 문제다)
_EDGE_DROP_MAX = 4
# 이보다 적게 남을 수 있으면 아무것도 지우지 않는다 (짧은 곡을 통째로 비우지 않게)
_EDGE_DROP_MIN_KEEP = 4
# «Vocal : 初音ミク», «作詞：〇〇», «Music & Lyrics : 40mP» — 짧은 머리말 + 콜론 + 내용.
# 머리말에 글자가 하나라도 있어야 한다: «3:00»처럼 숫자만 있는 진짜 가사를 제외하기 위한 조건.
_CREDITISH_LINE_RE = re.compile(r"^(?=[^:：]*[^\W\d_])[^:：]{1,24}[:：]\s*\S")
_URLISH_RE = re.compile(r"https?://|www\.|@[A-Za-z0-9_]{3,}")


def _looks_non_lyric(text: str) -> bool:
    """가사로 보이지 않는 줄인가 (크레딧 표기·출처·URL·계정 표기)."""
    t = (text or "").strip()
    if not t:
        return True
    return bool(_URLISH_RE.search(t) or _CREDITISH_LINE_RE.match(t))


def _drop_nonvocal_nonlyric_edges(
    timestamps: list[dict[str, Any]],
    max_ratio: float = _NONVOCAL_MAX_RATIO,
    max_drop: int = _EDGE_DROP_MAX,
    min_keep: int = _EDGE_DROP_MIN_KEEP,
) -> list[str]:
    """가사 앞뒤의 비가창 줄을 제거하고, 지운 줄 텍스트를 돌려준다 (timestamps를 제자리 수정).

    자막에는 «Music & Lyrics : 40mP»처럼 크레딧이 가사 줄로 섞여 들어오고, 붙여넣기 가사에도
    «作詞：〇〇» 헤더가 함께 넘어온다. 이런 줄은 정렬에 들어가면 화면 첫 줄부터 「뮤우짓쿠 안도
    레릿쿠스」 같은 독음으로 박힌다(실측: TXzfQ0cP1P0).

    **두 근거가 동시에 성립할 때만** 버린다:
      ① 그 라인 구간에 발성이 사실상 없다 (VAD active_ratio ≤ max_ratio)
      ② 텍스트가 가사로 보이지 않는다 (_looks_non_lyric)

    하나만 보면 안 된다 — ①만 보면 CTC가 무음에 잘못 얹은 **진짜 첫 줄**을 지우고, ②만 보면
    노래 안에서 콜론이 든 가사를 지운다. 크레딧은 곡의 맨 앞/맨 뒤에 붙으므로 양 끝에서만
    안쪽으로 훑고, 조건이 깨지는 첫 줄에서 멈춘다.
    """
    dropped: list[str] = []

    def is_candidate(seg: dict[str, Any]) -> bool:
        ratio = (seg.get("debug") or {}).get("active_ratio")
        if ratio is None or ratio > max_ratio:
            return False
        return _looks_non_lyric(seg.get("text", ""))

    while (
        len(timestamps) > min_keep and len(dropped) < max_drop and is_candidate(timestamps[0])
    ):
        dropped.append(timestamps.pop(0).get("text", ""))
    while (
        len(timestamps) > min_keep and len(dropped) < max_drop and is_candidate(timestamps[-1])
    ):
        dropped.append(timestamps.pop().get("text", ""))
    return dropped


def _interlude_gaps(vad_regions, min_gap_sec: float) -> list[tuple[float, float]]:
    """VAD 리전 사이의 긴 무음 구간 [gap_start, gap_end] 목록 (시간순)."""
    regions = sorted((reg.start, reg.end) for reg in vad_regions or [])
    return [
        (e0, s1) for (_, e0), (s1, _) in zip(regions, regions[1:]) if s1 - e0 >= min_gap_sec
    ]


def _straddles_interlude(results, min_gap_sec: float, vad_regions=None) -> bool:
    """라인이 간주를 가로지르는 정황이 있으면 True — ja 교차검증 트리거용 값싼 신호.

    두 가지 형태를 모두 본다:
      ① **라인 사이 간극**이 min_gap_sec 이상 (간주를 사이에 두고 블록이 나뉜 경우),
      ② **한 라인의 스팬이 간주(무음) 구간을 min_gap_sec 이상 덮는 경우** (vad_regions 필요).
    ②가 없으면 역설이 생긴다: 문제 라인이 간주를 통째로 덮으면(실측 81.34→119.21)
    라인 사이 간극이 하나도 남지 않아 ①이 0개가 되고, **가장 의심스러운 곡에서 교차검증이
    아예 안 돌았다**. 덮는 라인 자체가 "정렬이 간주를 못 넘었다"는 직접 증거다.

    독음(ko) 정렬은 간주를 사이에 두고 라인 블록을 배치하므로, 이 간극은 대사/간주
    전이가 있다는 값싼 ko 전용 신호다. star 삼킴이 작아(무음 위 star) 삼킴 게이트를
    못 넘는 누출(初音ミクの消失: 삼킴 1.02s)도 여기서 ja 교차검증을 트리거하게 한다.
    실제 라인 이동은 아래 ja 대조(_leaked_runs)만이 결정하므로 이 게이트가 넓어도
    무해 케이스를 흔들지 않는다 — 두 번째 정렬을 도는 비용만 늘 뿐이다.
    """
    if any(
        results[i + 1].start_time - results[i].end_time >= min_gap_sec
        for i in range(len(results) - 1)
    ):
        return True
    gaps = _interlude_gaps(vad_regions, min_gap_sec) if vad_regions is not None else []
    return any(
        min(r.end_time, ge) - max(r.start_time, gs) >= min_gap_sec
        for gs, ge in gaps
        for r in results
    )


def _leaked_runs(
    ko_results,
    ja_results,
    windows: list[tuple[float, float]],
    lead_sec: float,
    onset_slack: float = 2.0,
    rejoin_tol: float = 3.0,
) -> list[list[int]]:
    """ko가 간주 이전으로 역누출한 라인들의 연속 런 목록 (ja와 라인 단위 대조).

    ko/ja는 같은 가사 라인을 정렬하므로 인덱스가 대응한다. 어떤 라인 i가 '누출 seed'가
    되는 조건: **ja가 그 라인을 간주 이후 창 안(±onset_slack)에 두는데 ko는 창의 gap_end
    보다 lead_sec 이상 앞에 둔다.** seed를 포함하는 '변위(|ja.start−ko.start|>rejoin_tol)
    연속 블록'을 하나의 런으로 돌려준다 — 선두 한 줄만 새는 경우(初音ミクの消失 idx46)와
    블록 전체가 압축된 경우(7/11 idx46-52)를 모두 같은 방식으로 포착하고, seed 없는
    변위(초고속 랩부의 ±2s 흔들림)는 런에서 배제해 외과적으로만 교체한다.
    """
    n = len(ko_results)
    if n == 0 or n != len(ja_results) or not windows:
        return []
    seed = [False] * n
    for w0, w1 in windows:
        for i in range(n):
            if (
                ko_results[i].start_time <= w0 - lead_sec
                and (w0 - onset_slack) <= ja_results[i].start_time <= (w1 + onset_slack)
            ):
                seed[i] = True
    displaced = [
        abs(ja_results[i].start_time - ko_results[i].start_time) > rejoin_tol for i in range(n)
    ]
    runs: list[list[int]] = []
    i = 0
    while i < n:
        if not displaced[i]:
            i += 1
            continue
        j = i
        while j < n and displaced[j]:
            j += 1
        block = list(range(i, j))
        if any(seed[k] for k in block):
            runs.append(block)
        i = j
    return runs


def _apply_leak_splice(ko_results, ja_results, idxs: list[int]) -> None:
    """누출 라인 idxs의 타이밍(start/end/word_segments)만 ja 정렬값으로 제자리 교체한다.

    가사 순서·인덱스가 대응하므로 텍스트/신뢰도 등 ko 라인 속성은 유지하고 타이밍만
    바꾼다. 교체 라인의 ko 발음 음절 스팬(pron_segments)은 압축된 값이라 호출부가 별도로
    무효화한다(캐시 재병합이 ja 타이밍 기반으로 복원). 런은 연속이고 ja는 단조라 경계는
    앞뒤 유지 라인과 자연히 정합한다.
    """
    for i in idxs:
        ja = ja_results[i]
        r = ko_results[i]
        r.start_time = ja.start_time
        r.end_time = ja.end_time
        r.word_segments = ja.word_segments


def _mark_leak_ghosts(
    raw_spans: list[tuple[float, float]],
    fixes: dict[int, list[str]],
    pre_spans: dict[int, tuple[float, float]],
    results,
    tol: float = 0.2,
) -> None:
    """ja 스플라이스/폴백으로 타이밍이 바뀐 라인에 확장 디버그 고스트(원 ko 위치)+"leak" 라벨.

    스플라이스/폴백 경로는 raw_spans/fixes를 리셋해 디버그 고스트가 사라진다. 이 함수를
    리셋 직후 호출하면, pre_spans[i]=(교체 전 ko start,end)와 유의하게(>tol) 이동한 라인만
    raw_spans[i]를 원 ko 위치로 되돌리고 fixes[i] 앞에 "leak"을 넣는다 — 디버그 모드에서
    "ko가 어디서 당겨졌는지" 고스트가 표시된다.
    """
    for i, span in pre_spans.items():
        r = results[i]
        if abs(r.start_time - span[0]) > tol or abs(r.end_time - span[1]) > tol:
            raw_spans[i] = span
            labels = fixes.setdefault(i, [])
            if "leak" not in labels:
                labels.insert(0, "leak")


# ── 번역·독음 병기 시트 감지 (정렬 입력에서만 제외, 표시용은 유지) ─────────
#
# 사용자가 가사 사이트에서 복사해 붙여넣을 때 원문만 넣지 않고 「원문 / 한글 독음 / 한국어
# 번역」이 병기된 시트를 통째로 넣는 경우가 있다. 노래하지 않는 줄에 타이밍을 맞추려 들면
# 한 보컬에 몇 배 길이의 토큰 열을 억지로 맞추게 되어 **가창 줄의 타이밍까지** 망가진다.
# 코퍼스 73곡 실측:
#   FxOfDVyITak — (가나, 한글, 한글) 3줄 블록이 74/74회 완벽히 반복. 입력의 2/3가 비가창.
#                 quality_score 0.0135로 코퍼스 최저권.
#   ba7YbGO2aq4 — 한글 8줄이 전부 직전 일본어 줄의 번역(ゆらゆら numb numb → 아스라이해
#                 numb numb), 8/8이 바로 뒤에 붙는 패턴. quality_score 0.0072.
#
# 확장에도 lib/tri-line.ts(3줄 묶음 파서)가 있어 클라이언트에서 걸러지는 경로가 이미 있다.
# 여기는 그게 통과된 입력·과거 데이터·외부 호출자를 위한 **독립 안전망**이라 중복돼도 무해하다.
# 원문 줄 판정("가나/한자를 포함하고 한글은 거의 없음")은 tri-line.ts의 isJa와 같다. 한글 줄
# 판정은 의도적으로 **다르다**: tri-line.ts의 isKo는 한글 비율 0.5를 요구하는데, 실측
# ba7YbGO2aq4의 번역 줄 "아스라이해 numb numb"는 0.38(라틴 차용어가 그대로 남는다)이라 그
# 기준으로는 잡히지 않는다. 여기서는 "한글을 담고 있고 가나가 없다"를 기준으로 삼는다.

_RE_KANA = re.compile(r"[぀-ヿ]")
_RE_HAN = re.compile(r"[一-鿿]")
_RE_HANGUL = re.compile(r"[가-힣]")

# 주기 반복 게이트 — 이제 **이것이 유일한 규칙이다.** period는 3만 본다: (원문, 독음, 번역)
# 처럼 원문 한 줄 뒤에 **한글 두 줄이 연속으로** 규칙적으로 붙는 구조는 실제 곡에 존재하지
# 않는다. 반면 period 2 (원문, 한글)의 완전 교대는 한·일 병창 곡이라는 실재 구조와 텍스트만
# 으로 구분이 불가능하고, 오폭하면 곡의 절반을 잃으므로 의도적으로 통과시킨다.
_GLOSS_CYCLE_PERIOD = 3
_GLOSS_MIN_CYCLES = 3  # 최소 3주기(9줄) — 두 세트짜리 우연 일치를 배제
_GLOSS_MIN_CONFORM = 0.9  # 주기 중 패턴 일치 비율 (실측 FxOfDVyITak은 1.0)
_GLOSS_MIN_COVER = 0.9  # 주기 구조가 전체 입력의 이 비율 이상을 덮어야 한다
#
# **삭제된 규칙 B(인접 종속)에 대해** — 2026-07-26. "소수 한글 줄 전부가 일어 줄 바로 뒤에
# 붙으면 그것들은 앞줄의 번역이다"는 규칙이 있었고, 유일한 근거가 ba7YbGO2aq4의 8줄이었다.
# **그 판정이 틀렸다.** 사용자가 그 곡을 직접 듣고 확인했다: 「아스라이해」「희미한」「미묘한」
# 「좋아해」는 실제로 노래에서 한국어로 불리는 가사다(그 곡은 일본어·영어·한국어가 실제
# 발성에 섞인다). 규칙이 세운 가정 — "실제로 노래되는 한국어 구간은 한글 줄이 연달아 나온다"
# — 이 거짓이었다: 이 곡은 한국어가 **한 줄씩** 섞여 불린다(ゆらゆら → ゆらゆら → 아스라이해).
#
# 코퍼스 68곡 전수 조사에서 규칙 B가 잡은 곡은 **이 한 곡뿐이었다.** 즉 유일한 사례가
# 오판이었고 다른 이득이 없었다. 그래서 규칙을 지웠다 — 임계값 조정으로는 고칠 수 없다.
# 이 형태(한 줄씩 섞인 진짜 다국어 가사)와 "일부 줄에만 번역을 붙인 시트"는 텍스트만으로
# 구별할 수 없고, 오폭 비용이 미검출 비용보다 훨씬 크다(이 곡에서는 8줄이 가사에서 사라지고
# 앞줄의 translation 필드를 덮어써, 그 줄의 실제 번역까지 잃었다).
#
# 한글 줄로 인정하는 최소 한글 비율. 「아스라이해 numb numb」가 0.38이라 0.5로는 놓친다
# (라틴 차용어가 그대로 남는다). 0.2까지 낮춰도 영어 줄에 한글 한 단어가 섞인 정도는 걸러진다.
_GLOSS_MIN_HANGUL_RATIO = 0.2
# **원문 줄은 'ja'(가나/한자)만 인정한다.** 'other'(라틴)까지 원문으로 허용하면
# (영어 훅, 한국어, 한국어) 반복 = 실제로 존재하는 한국어 곡 구조가 3줄 주기에 걸려 곡의
# 2/3를 잃는다. 실측된 오염 형태는 전부 일어 원문 + 한글 주석이고, 클라이언트 tri-line도
# isJa로 원문을 요구하므로 두 경로의 범위가 일치한다. (영어 원문 + 한글 병기 시트는 서버에서
# 잡지 않는다 — 정상 곡 오폭 비용이 미검출 비용보다 훨씬 크다.)
_GLOSS_ANCHOR_CLASS = "ja"


def _line_script_class(text: str) -> str:
    """라인을 스크립트 계열로 분류: 'ja'(가나/한자 위주) / 'ko'(한글 있음) / 'other'.

    비율은 모두 공백 제외 글자 수 기준. 'ja'는 tri-line.ts의 isJa와 같은 규칙이고, 'ko'는
    라틴 차용어가 섞인 실측 번역 줄("아스라이해 numb numb", 한글 0.38)을 놓치지 않도록
    비율 문턱을 낮게 둔다. 대신 한자보다 한글이 많을 것을 요구해, 한자 줄에 한글 한 글자가
    섞인 원문이 한글 줄로 넘어가지 않게 한다.
    """
    dense = re.sub(r"\s", "", text)
    if not dense:
        return "other"
    hangul = len(_RE_HANGUL.findall(dense))
    han = len(_RE_HAN.findall(dense))
    has_kana = bool(_RE_KANA.search(text))
    hangul_ratio = hangul / len(dense)
    if (has_kana or han) and hangul_ratio < 0.15:
        return "ja"
    if hangul and not has_kana and hangul >= han and hangul_ratio >= _GLOSS_MIN_HANGUL_RATIO:
        return "ko"
    return "other"


def detect_gloss_lines(texts: list[str]) -> dict[int, tuple[str, int]]:
    """번역·독음 병기 시트의 **비가창 줄**을 찾아 {줄 인덱스: (역할, 원문 줄 인덱스)}로 반환.

    역할은 ``"pronunciation"`` 또는 ``"translation"`` — 호출부가 그 텍스트를 원문 줄의
    표시용 메타로 접어 넣어 화면에서 사라지지 않게 한다.

    **보수적으로 설계했다.** 정상 곡의 라인을 하나라도 잘못 빼면 그 줄이 타이밍을 잃으므로,
    아래 규칙이 압도적으로 성립할 때만 발동하고 애매하면 빈 dict를 돌려준다.
    걸러낼 줄은 항상 한글 줄이고 원문 줄은 일어(가나/한자) 줄이어야 한다 — 실측된 오염 형태가
    "한국어 사용자가 일어 가사에 한글 독음·번역을 병기해 붙여넣는 것"이고, 원문에 라틴까지
    허용하면 정상 한국어 곡을 오폭한다(_GLOSS_ANCHOR_CLASS 주석 참조).

    규칙(주기 반복): (일어, 한글, 한글) 3줄 주기가 3주기 이상 반복되고, 전체 입력의 90%
    이상을 덮으며, 주기 일치율이 90% 이상. 2·3번째 줄을 독음·번역으로 본다.

    잡지 못하는 형태(의도된 한계): ① (일어, 한글) 완전 교대 시트는 한·일 병창 곡과 텍스트상
    구별이 불가능해 통과시킨다, ② 영어·한국어 원문에 한글 주석이 붙은 시트, ③ 한국어 곡에
    일본어 번역이 병기된 역방향 시트, ④ **일부 줄에만 번역을 붙인 시트** — 이것을 잡으려던
    규칙 B가 진짜 다국어 가창 곡을 오폭해 삭제됐다(상수 블록의 삭제 기록 참조). 클라이언트
    tri-line 경로나 사용자 정리에 맡긴다.
    """
    n = len(texts)
    if n < _GLOSS_MIN_CYCLES * _GLOSS_CYCLE_PERIOD:
        return {}
    classes = [_line_script_class(t) for t in texts]
    return _detect_gloss_by_cycle(classes, n)


def _detect_gloss_by_cycle(classes: list[str], n: int) -> dict[int, tuple[str, int]]:
    """규칙 A — (일어, 한글, 한글) 3줄 주기. offset을 훑는 이유는 맨 앞의 제목/표기 한 줄이
    주기를 밀어도 감지가 죽지 않게 하기 위함이다 (offset 앞 줄들은 손대지 않는다)."""
    period = _GLOSS_CYCLE_PERIOD
    for offset in range(period):
        starts = list(range(offset, n - period + 1, period))
        if len(starts) < _GLOSS_MIN_CYCLES:
            continue
        if period * len(starts) < _GLOSS_MIN_COVER * n:
            continue
        conform = [
            s
            for s in starts
            if classes[s] == _GLOSS_ANCHOR_CLASS
            and all(classes[s + k] == "ko" for k in range(1, period))
        ]
        if len(conform) < _GLOSS_MIN_CONFORM * len(starts):
            continue
        # 일치하지 않는 주기는 손대지 않는다 — 그 줄들은 그대로 정렬 입력에 남는다.
        out: dict[int, tuple[str, int]] = {}
        for s in conform:
            out[s + 1] = ("pronunciation", s)
            out[s + 2] = ("translation", s)
        return out
    return {}


def _split_gloss_lines(lyric_lines: list, enabled: bool) -> tuple[list, dict[int, dict[str, str]]]:
    """병기 시트의 비가창 줄을 **정렬 입력에서만** 뺀다.

    반환: (정렬에 쓸 라인 목록, {정렬 인덱스: {역할: 텍스트}}). 두 번째 값은 정렬이 끝난 뒤
    ``_fold_gloss_into_segments``가 원문 세그먼트의 표시용 메타로 접어 넣어, 사용자가
    붙여넣은 줄이 화면에서 사라지지 않게 한다.
    """
    if not enabled or not lyric_lines:
        return lyric_lines, {}
    gloss = detect_gloss_lines([ln.text for ln in lyric_lines])
    if not gloss:
        return lyric_lines, {}
    keep = [i for i in range(len(lyric_lines)) if i not in gloss]
    if not keep:
        # 이론상 도달 불가(원문 줄은 항상 남는다) — 그래도 전멸 입력을 정렬에 넘기지 않는다
        return lyric_lines, {}
    pos = {orig: k for k, orig in enumerate(keep)}
    folded: dict[int, dict[str, str]] = {}
    for i in sorted(gloss):
        role, anchor = gloss[i]
        k = pos.get(anchor)
        if k is None:
            continue
        folded.setdefault(k, {}).setdefault(role, lyric_lines[i].text)
    tally: dict[str, int] = {}
    for role, _ in gloss.values():
        tally[role] = tally.get(role, 0) + 1
    logger.info(
        f"Bilingual gloss sheet detected: excluding {len(gloss)}/{len(lyric_lines)} non-sung "
        f"line(s) from alignment input ({tally}); kept for display on their source line. "
        f"Excluded line indexes (0-based, first 12): {sorted(gloss)[:12]}"
    )
    return [lyric_lines[i] for i in keep], folded


def _fold_gloss_into_segments(
    timestamps: list[dict[str, Any]], folded: dict[int, dict[str, str]]
) -> int:
    """제외한 병기 줄을 원문 세그먼트의 pronunciation/translation으로 되붙인다.

    이미 값이 있으면 덮지 않는다 — 독음 정렬 경로(pron_data)나 위키 line_meta가 채운 값은
    실측 타이밍(pron_segments)을 동반하므로 붙여넣기 텍스트보다 우선한다.
    """
    n = 0
    for k, roles in folded.items():
        if k >= len(timestamps):
            continue
        seg = timestamps[k]
        for role, text in roles.items():
            if not seg.get(role):
                seg[role] = text
                n += 1
    return n


def _pron_by_text(line_meta: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    """line_meta를 정규화 텍스트 → 메타 dict로 색인 (merge_line_meta와 동일 규칙).

    색인 규칙은 ``_index_line_meta`` 하나로 공유한다 — 규칙이 갈라지면 같은 곡이
    정렬 경로(여기)와 직렬화 병합 경로(merge_line_meta)에서 다른 메타를 보게 된다.
    """
    return _index_line_meta(line_meta)


def _alignable_pron(pron: str | None) -> str:
    """정렬 입력으로 쓸 수 있는 발음만 통과 — **한글이 한 글자라도 있어야 한다**.

    독음(ko) 정렬은 kor 어댑터에 한글 텍스트를 넣는 계약이다. 다국어화 이후 비ko
    사용자의 line_meta엔 romaji·가타카나 발음이 실릴 수 있는데(번역 API의 결정론
    매트릭스), 라틴은 kor 어댑터에서 정렬되지 않고(latin_hangul.py 헤더 실측 —
    conf<0.01이 90~99%) 가나도 마찬가지다. 그런 발음은 «없음»과 동일 취급해
    게이트(coverage)와 정렬 입력 양쪽에서 원문 폴백을 태운다.
    """
    p = (pron or "").strip()
    return p if p and _HANGUL_CHAR_RE.search(p) else ""


def _pron_coverage(lyric_lines, by_text: dict[str, dict[str, Any]]) -> float:
    """정렬 가능한(한글) 발음 표기가 붙은 라인 비율 (0~1)."""
    if not lyric_lines:
        return 0.0
    have = 0
    for ln in lyric_lines:
        m = by_text.get(_normalize_line(ln.text))
        if m and _alignable_pron(m.get("pronunciation")):
            have += 1
    return have / len(lyric_lines)


def _referee_candidates(
    lyric_lines, pron_for_line: list[str], align_settings
) -> tuple[dict[int, list[str]], dict[int, float]]:
    """오디오 심판에 넘길 라인별 후보 독음과 라인별 마진.

    기본 후보는 line_meta의 발음(결정론 또는 위키·사람)이다 — 그것을 [0]에 두고
    ``pronunciation_candidates``의 대안을 뒤에 붙인다.

    **사람이 쓴 발음도 심판 대상에 포함하되 마진을 더 크게 요구한다.** 근거: 사람은 결정론
    경로(사람 발음 2,207줄 대비 82.1%)보다 옳을 확률이 훨씬 높으므로 기본값으로 둘 이유가
    분명하지만, 사람이 후리가나·아테지를 놓친 줄이 실제로 있다(涙（シル）류). 아예 제외하면
    그 줄은 영구히 못 고치고, 같은 마진으로 두면 사람 표기가 노이즈로 뒤집힌다 — 그래서
    포함 + 큰 마진이다. 사람이 쓴 발음인지는 별도 플래그 없이 판정된다: 결정론 기본값
    (``candidates[0]``)과 다르면 그 발음은 다른 출처(위키 병합·LLM)에서 온 것이다.
    """
    from everyric2.text.pron_style import pronunciation_candidates

    cands: dict[int, list[str]] = {}
    margins: dict[int, float] = {}
    limit = max(2, int(align_settings.pron_referee_max_candidates))
    for i, pron in enumerate(pron_for_line):
        if not pron:
            continue
        alts = pronunciation_candidates(lyric_lines[i].text, max_candidates=limit)
        merged = [pron] + [a for a in alts if a != pron]
        if len(merged) < 2:
            continue  # 후보 없음 → 심판을 돌리지 않는다 (비용 0)
        cands[i] = merged[:limit]
        human = bool(alts) and pron != alts[0]
        margins[i] = float(
            align_settings.pron_referee_human_margin
            if human
            else align_settings.pron_referee_margin
        )
    return cands, margins


def _log_referee_decisions(decisions: list[dict]) -> None:
    """심판 판정을 로그로 남긴다 — 조용히 바꾸면 사후 추적이 불가능하다."""
    if not decisions:
        return
    switched = [d for d in decisions if d.get("chosen") != d.get("default")]
    for d in switched:
        logger.info(
            f"Audio referee switched line {d['line']} pronunciation: "
            f"{d['default']!r} → {d['chosen']!r} (+{d['gain']:.4f} nats/token >= margin "
            f"{d['margin']}, window {d['frames']} frames; candidates {d['scores']})"
        )
    logger.info(
        f"Audio referee: {len(switched)}/{len(decisions)} scored line(s) switched reading "
        f"(lines with a single candidate were not scored)"
    )


def _caption_forbidden_spans(video_id: str | None, lyric_lines, audio_sec: float, align_settings):
    """사람이 만든 유튜브 자막에서 «가사 줄이 놓일 수 없는 구간»을 뽑는다 (블로킹 IO).

    앵커는 **있으면 좋은** 신호다 — 자막 조달·매칭이 어떤 이유로든 실패하면 금지 구간 없이
    (=기존 동작으로) 계속한다. 그래서 예외를 전부 삼키고 근거만 debug에 남긴다.

    가사 출처와 무관하게 동작해야 한다는 점이 중요하다. 사고 곡의 가사는 자막이 아니라
    위키에서 왔고, 앵커는 «타이밍 좌표계»로만 쓰이므로 자막 가사 경로와 독립이다.
    """
    from everyric2.alignment.caption_anchors import (
        AnchorPlan,
        derive_anchor_plan,
        script_lang_hint,
    )

    # 스캐폴드(caption_scaffold)도 같은 조달·매칭을 쓴다 — 제약(caption_anchors)이 꺼져
    # 있어도 스캐폴드가 켜져 있으면 계획은 만들고, 제약으로 쓸지는 호출부가 스위치별로 정한다.
    wanted = getattr(align_settings, "caption_anchors", False) or getattr(
        align_settings, "caption_scaffold", False
    )
    if not wanted or not video_id:
        return AnchorPlan(debug={"skipped": "disabled" if video_id else "no_video_id"})
    texts = [ln.text for ln in lyric_lines]
    try:
        from everyric2.server.services.youtube_captions import iter_manual_caption_events

        tracks = iter_manual_caption_events(
            video_id,
            script_lang_hint("\n".join(texts)),
            align_settings.caption_anchor_max_tracks,
        )
        plan = derive_anchor_plan(
            texts,
            tracks,
            min_match=align_settings.caption_anchor_min_match,
            min_gap_sec=align_settings.caption_anchor_min_gap_sec,
            margin_sec=align_settings.caption_anchor_margin_sec,
            audio_sec=audio_sec,
            max_forbidden_ratio=align_settings.caption_anchor_max_forbidden_ratio,
            positive_min_match=align_settings.caption_anchor_positive_min_match,
            span_min_match=getattr(align_settings, "caption_scaffold_min_match", None),
        )
    except Exception:
        logger.exception("Caption anchor derivation failed; aligning without anchors")
        return AnchorPlan(debug={"skipped": "error"})
    if plan.spans or plan.line_starts:
        logger.info(
            f"Caption anchors for {video_id}: track {plan.debug.get('track')} matched "
            f"{plan.debug.get('matched')}/{plan.debug.get('matchable')} line(s) "
            f"({plan.debug.get('rate')}); forbidding {plan.debug.get('spans')}, "
            f"anchoring {len(plan.line_starts)} line start(s)"
        )
    else:
        logger.info(
            f"Caption anchors for {video_id}: not used ({plan.debug.get('skipped')}); "
            f"{plan.debug}"
        )
    return plan


def _align_with_pronunciation(
    engine,
    audio,
    lyric_lines,
    by_text: dict[str, dict[str, Any]],
    align_settings=None,
    anchor_kw: dict[str, Any] | None = None,
):
    """독음(ko) 텍스트로 CTC 정렬 후 원문 라인에 역매핑.

    ``align_settings``를 주고 ``pron_referee``가 켜져 있으면 라인별 후보 독음을 함께 넘겨
    **오디오가 어느 독음이 맞는지 고르게** 한다 (``_referee_candidates``). 주지 않으면
    후보 없이 정렬하므로 기존 동작과 동일하다.

    반환: (results, pron_data)
      results: 원문 텍스트 SyncResult 목록 (타이밍/word_segments는 독음 정렬 역매핑값).
      pron_data: line_idx → {"pronunciation", "translation", "pron_segments", "heard",
          "heard_spans", "referee", "tokens"}.
    """
    from everyric2.inference.prompt import LyricLine, SyncResult, WordSegment
    from everyric2.text.reading import map_pron_alignment_to_line

    # **표시용 독음**과 **정렬 텍스트**를 갈라 둔다. 한때 같은 값을 썼는데, 독음이 없는 줄이
    # 빈 텍스트로 정렬 엔진에 들어가 그 줄이 정렬에서 통째로 빠졌다 — 타이밍은 앞뒤 사이로
    # 보간되어 줄 시작은 그럭저럭 맞지만 **줄 내부 음절 스팬(word_segments)을 잃어** 그 줄에서
    # 가라오케 채움이 죽는다.
    #
    # 실측(ba7YbGO2aq4, 일·영·한이 실제 발성에 섞인 곡): 한국어 가창 줄 5개(「희미한」×2,
    # 「미묘한」×2, 「좋아해」)가 그렇게 빠져 align_coverage 0.9359였고 그 5줄만 words가 없었다.
    # 독음이 없는 줄은 사실상 **한글 전용 줄**이다(일어는 가나/한자에서, 라틴은 음차에서 항상
    # 독음이 나온다). 한글은 kor 어댑터 vocab이 덮으므로(1261자) 원문을 그대로 정렬하면 된다.
    pron_for_line = [
        _alignable_pron((by_text.get(_normalize_line(ln.text)) or {}).get("pronunciation"))
        for ln in lyric_lines
    ]
    align_for_line = [pron or ln.text for pron, ln in zip(pron_for_line, lyric_lines)]
    pron_lines = [
        LyricLine(text=text, line_number=ln.line_number)
        for text, ln in zip(align_for_line, lyric_lines)
    ]
    referee_cands: dict[int, list[str]] = {}
    referee_margins: dict[int, float] = {}
    if align_settings is not None and getattr(align_settings, "pron_referee", False):
        referee_cands, referee_margins = _referee_candidates(
            lyric_lines, pron_for_line, align_settings
        )
    ko_results = engine.align(
        audio,
        pron_lines,
        language="ko",
        line_candidates=referee_cands or None,
        referee_margins=referee_margins or None,
        **(anchor_kw or {}),
    )
    # ko 정렬 직후에 포착한다 — 이후 ja 교차정렬이 엔진의 직전-정렬 기록을 덮는다
    # (pron_star_spans와 같은 사정).
    decisions = list(getattr(engine, "_last_referee", None) or [])
    heard = dict(getattr(engine, "_last_heard", None) or {})
    heard_spans = dict(getattr(engine, "_last_heard_spans", None) or {})
    _log_referee_decisions(decisions)
    by_line_decision = {d["line"]: d for d in decisions}

    results = []
    pron_data: dict[int, dict[str, Any]] = {}
    for i, (ln, kr) in enumerate(zip(lyric_lines, ko_results)):
        # 심판이 이 라인의 독음을 바꿨으면 SyncResult.text가 이긴 후보다 — 역매핑·표시가
        # 모두 **실제로 정렬된** 텍스트를 써야 음절 스팬과 발음 표기가 어긋나지 않는다.
        aligned = kr.text or align_for_line[i]
        ko_words = kr.word_segments or []
        # 음절별 confidence까지 함께 넘겨 글자별 conf 역매핑을 살린다 (라인 균일 부여 회귀 수정)
        spans = [(w.word, w.start, w.end, w.confidence) for w in ko_words]

        words = pron_segments = None
        if aligned and spans:
            if pron_for_line[i]:
                words, pron_segments = map_pron_alignment_to_line(ln.text, aligned, spans)
            else:
                # 독음이 없어 **원문을 그대로** 정렬한 줄이다 — 역매핑할 것이 없다.
                # map_pron_alignment_to_line은 「한글 음절 → 모라 → 원문 글자」 3단
                # 역매핑이라 원문이 이미 한글인 이 경우에는 아무 것도 내지 못한다(실측:
                # words=None). 정렬된 음절 스팬이 곧 원문 글자 스팬이므로 그대로 쓴다.
                # pron_segments는 **표시 독음의** 음절 스팬이라 여기서는 없는 것이 맞다.
                words = [
                    {"word": w, "start": s, "end": e, "confidence": c} for w, s, e, c in spans
                ]

        word_segments = (
            [WordSegment(word=w["word"], start=w["start"], end=w["end"]) for w in words]
            if words
            else None
        )
        line_conf = _geomean([w.confidence for w in ko_words])
        if word_segments:
            # 글자별 conf(reading이 음절 conf 기하평균으로 산출) 우선, 매핑 불가 글자는 라인 기하평균 폴백
            for ws, w in zip(word_segments, words):
                c = w.get("confidence")
                if c is None:
                    c = line_conf
                ws.confidence = round(c, 6) if c is not None else None

        results.append(
            SyncResult(
                line_number=ln.line_number,
                text=ln.text,
                start_time=kr.start_time,
                end_time=kr.end_time,
                confidence=round(line_conf, 6) if line_conf is not None else None,
                word_segments=word_segments,
            )
        )
        meta = by_text.get(_normalize_line(ln.text)) or {}
        decision = by_line_decision.get(i)
        pron_data[i] = {
            # 원래 독음이 없던 줄은 **표시할 독음이 없다** — 정렬에만 원문을 썼을 뿐이다.
            # 여기에 원문을 넣으면 한글 줄 아래에 같은 한글이 한 번 더 찍힌다. 심판은 독음이
            # 있는 줄에만 후보를 만들므로(`pronunciation_candidates`가 일본어를 요구한다)
            # 이 갈래에서 심판 결과를 잃을 일은 없다.
            "pronunciation": (aligned or None) if pron_for_line[i] else None,
            "translation": meta.get("translation"),
            "pron_segments": pron_segments,
            # 진단용: 모델이 이 라인 구간에서 실제로 「들은」 텍스트와 심판의 판정 근거.
            # 노래 ASR은 약해서 heard로 발음을 만들면 안 되지만, 정렬 텍스트와의 불일치는
            # posterior 크기에 무관한 라인 단위 품질 지표다(conf는 곡 단위로만 유효).
            "heard": heard.get(i) or None,
            "heard_spans": heard_spans.get(i) or None,
            "referee": (
                {k: v for k, v in decision.items() if k != "line"} if decision else None
            ),
            # 심판이 바꾼 줄의 **이긴 읽기**를 토큰 열로 실어 보낸다 — 직렬화에서
            # romaji 등 다른 표기가 같은 읽기를 따르게 하는 유일한 통로다(안 실으면
            # 기본 읽기로 렌더돼 화면의 한글 독음과 다른 낱말이 나란히 찍힌다).
            # 안 바뀐 줄은 기본 읽기 = romaji_line의 자체 토큰화라 실을 것이 없다.
            "tokens": (
                _referee_token_set(ln.text, aligned)
                if decision and decision.get("chosen") != decision.get("default")
                else None
            ),
        }
    return results, pron_data


def _align_original(engine, audio, lyric_lines, language: str | None, anchor_kw=None):
    """원문 텍스트 정렬 — 자막 앵커 제약을 독음 경로와 **똑같이** 걸어 준다.

    ja 교차정렬·폴백까지 같은 제약을 받아야 아래 누출 가드(_leaked_runs)가 같은 좌표계의
    두 정렬을 비교한다. 한쪽만 제약하면 가드가 «제약 때문에 생긴 차이»를 누출로 오독한다.

    앵커가 없으면 앵커 인자를 **아예 넘기지 않는다**(``anchor_kw``가 빈 dict) — 기본 경로의 호출이
    앵커 도입 전과 문자 그대로 같아야 한다(엔진 대역을 쓰는 호출부·테스트도 그대로 통과한다).
    """
    return engine.align(
        audio, lyric_lines, language=language or "auto", **(anchor_kw or {})
    )


def _anchor_kwargs(forbidden_spans, line_starts=None) -> dict[str, Any]:
    """앵커가 있을 때만 그 키워드를 만든다 (없으면 빈 dict = 앵커 도입 전과 같은 호출)."""
    kwargs: dict[str, Any] = {}
    if forbidden_spans:
        kwargs["forbidden_spans"] = forbidden_spans
    if line_starts:
        kwargs["line_starts"] = line_starts
    return kwargs


# ── 새 정렬 스택 배선 (owsm/omniasr 앵커 + 2패스 리파이너) ──────────────────────
#
# 이식된 부품(everyric2/alignment/{owsm_engine,omniasr_engine,refine_window}.py,
# everyric2/alignment/display_fixes.py, everyric2/text/align_target.py)을 _run_alignment에
# 배선한다. 레거시 ko/ja 이중정렬·star 토큰·pron_data DP 근사 경로(_align_with_pronunciation
# 이하)는 완전히 별개로 남겨 둔다 — 섞지 않는다. 새 스택은 앵커·리파이너가 이미 다표기
# 음절을 **실측**하므로, "DP 근사를 사후에 바로잡는" 레거시 후반 단계(ko/ja 융합·뭉침
# 세분화·붕괴 재합성)를 다시 태우면 오히려 실측값을 헤친다 — new_stack_active 가드로
# 그 넷을 건너뛴다(_run_alignment 본문 참고).


def _new_stack_enabled(settings) -> bool:
    """``settings.alignment.engine``이 새 앵커 스택(owsm/omniasr) 중 하나를 가리키면 True.

    둘 다 "새 스택 켜짐"의 동의어다 — 실제로 어느 모델이 도는지는 곡 언어별로
    ``_new_stack_anchor_type``이 정한다(예: engine="owsm"이어도 en 곡은 omniasr로 정렬된다
    — 아래 함수 docstring 참고). engine이 기존 값("ctc" 등)으로 남아 있는 한 이 함수는
    False이고 _run_alignment는 구스택(get_shared_ctc_engine)으로 그대로 정렬한다(폴백).
    """
    return settings.alignment.engine in ("owsm", "omniasr")


def _new_stack_anchor_type(language: str | None) -> str:
    """언어 → 새 스택 앵커 엔진 타입.

    코디네이터 확정 라우팅(2026-08-03, 모델 교체 이니셔티브): ja는 OWSM-CTC v4 1B(붕괴
    방어 실측 — owsm_engine.py 모듈 docstring), 그 밖은 전부 omniASR-CTC-300M(1,600+ 언어
    체크포인트라 언어 게이트가 없다 — omniasr_engine.py 모듈 docstring)로 떨어진다. ko도
    포함된다: omniASR이 이미 다국어 단일 모델이고, 2패스 리파이너가 어차피 omniasr을
    상주시키므로(``_run_new_stack_alignment``) 같은 모델을 앵커로 겸용하면 모델 로드가
    1회로 끝난다. «무분리 asr 빠른 경로 → 붕괴 의심 시 owsm 3단계 구원» 같은 성능
    라우팅(scripts/bench_adapters/routed.py)은 이 표의 범위 밖이다 — 별도 후속 작업.
    """
    return "owsm" if (language or "").strip().lower() == "ja" else "omniasr"


class _PathBridgedRefiner:
    """refine_window.SyllableRefiner 계약과 BaseAlignmentEngine.emission_for 계약 사이의
    배선층 어댑터.

    두 이식이 ``emission_for``의 입력 타입을 서로 다르게 확정해 뒀다: OmniASREngine은
    ``BaseAlignmentEngine.emission_for(self, audio: AudioData)``(base.py:88)를 그대로
    override하는데, ``refine_window.refine_lines``는 ``refiner.emission_for(vocals_path)``로
    **Path**를 넘긴다(``SyllableRefiner`` 프로토콜과 tests/test_refine_window.py의
    ``_FakeRefiner.emission_for(self, audio_path: Path)``가 그 계약을 고정한다). 실제
    엔진을 리파이너로 그대로 넘기면 AudioData 자리에 Path가 들어가 ``AudioLoader.
    prepare_for_alignment``이 즉시 깨진다. 양쪽 다 각자 테스트로 계약이 고정돼 있어 어느
    쪽 모듈도 못 고친다 — 배선 층에서 흡수한다.
    """

    def __init__(self, engine: Any, loader: Any) -> None:
        self._engine = engine
        self._loader = loader

    def emission_for(self, audio_path: Path) -> Any:
        audio = self._loader.load(audio_path)
        return self._engine.emission_for(audio)


def _pron_seg_to_wire(span: Any) -> dict[str, Any]:
    """``refine_window.PronSegmentSpan`` → 서버 wire ``PronSegment`` 딕셔너리
    (everyric2-chrome/src/types.ts ``PronSegment``와 필드 단위로 대응)."""
    out: dict[str, Any] = {"text": span.text, "start": span.start, "end": span.end}
    if not span.resolved:
        out["resolved"] = False
    if span.confidence is not None:
        out["confidence"] = span.confidence
    if span.word_end:
        # PronSegment 계약에 없는 추가 필드 — additive라 구버전 확장은 무시한다
        # (refine_window.py 모듈의 "표시(발음) 세그" 절 참고).
        out["word_end"] = True
    return out


def _write_stems_for_two_pass(sep_result: Any) -> tuple[Path, Path]:
    """분리 결과를 임시 vocals.wav/inst.wav 쌍으로 쓴다.

    ``refine_window._dominance_curve``가 ``vocals_path.with_name("inst.wav")``라는 형제
    파일 관례로 반주를 찾는다 — 분리기 자체의 산출 파일(예: bs-polarformer-fp16의
    work_dir)은 서버가 상주 프로세스라 요청마다 이미 정리돼 있으므로(polarformer_
    separator.py의 finally) 리파이너를 위해 별도로 다시 쓴다. 반환: (지워야 할 임시
    디렉터리, vocals 경로) — 호출부가 finally에서 디렉터리를 지운다.
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="two_pass_stems_"))
    vocals_path = tmp_dir / "vocals.wav"
    sep_result.vocals.to_file(vocals_path)
    sep_result.accompaniment.to_file(tmp_dir / "inst.wav")
    return tmp_dir, vocals_path


@dataclass
class _NewStackResult:
    """``_run_new_stack_alignment`` 반환 묶음 — ``_run_alignment``의 공용 꼬리(타임스탬프
    직렬화 → 멜로디 → 품질 → 응답 조립)가 레거시 분기와 **같은 이름**으로 기대하는 지역
    변수들에 1:1 대응한다."""

    results: list[Any]
    alignment_text: str
    pron_data: dict[int, dict[str, Any]]
    fixes: dict[int, list[str]]
    raw_spans: list[tuple[float, float]]
    vad_regions: list[tuple[float, float]] | None
    clamped_lines: set[int]
    engine: Any
    adlib: list[tuple[float, float]] | None


def _run_new_stack_alignment(
    audio: Any,
    align_audio: Any,
    sep_result: Any,
    lyric_lines: list[Any],
    language: str | None,
    settings: Any,
    report: Any,
) -> "_NewStackResult":
    """새 정렬 스택(owsm/omniasr 앵커 + 2패스 리파이너) 본체.

    실행 순서(각 단계 이유는 인라인 주석):
      1. 앵커 정렬(라인 경계 확정) → 2. 우세도 기반 병적 길이 절단(``_clamp_pathological``,
      display_fixes.py 모듈 docstring이 "worker.py 배선 시 더 이른 단계에서 별도로
      호출하라"고 지시한 자리) → 3. 기존 VAD 기반 보정(``TimingPostProcessor``·
      ``_clamp_stretched_lines`` — 구스택과 같은 함수를 재사용, 회귀 표면을 늘리지
      않는다) → 4. 우세도 기반 좌초 보정(``display_fixes.apply_stranded_corrections``) →
      5. 추임새 후보 계산 → 6. 2패스 음절 재정렬(``refine_window.refine_lines`` — 위에서
      **최종 확정된** 라인 경계 위에서만 돈다, refine_lines의 "라인 경계는 앵커가 정한다"
      불변식과 맞물린다: 경계가 이 시점 이후 다시 안 움직여야 그 위의 음절 스팬이
      어긋나지 않는다).

    캡션 앵커(caption_anchors/caption_scaffold)·star 토큰·ko/ja 이중정렬 안전망은 이
    경로에 배선하지 않는다 — 전부 레거시 CTC 엔진의 특정 실패 모드(균일 posterior 등)에
    맞춰진 장치라 새 앵커에는 전제가 성립하지 않는다(범위 밖, 후속 작업으로 남긴다).
    """
    from everyric2.alignment import display_fixes as df
    from everyric2.alignment.factory import EngineFactory
    from everyric2.alignment.refine_window import TwoPassRefineConfig, refine_lines
    from everyric2.alignment.timing_postprocess import TimingPostProcessor
    from everyric2.audio.loader import AudioLoader
    from everyric2.audio.vad import VocalActivityDetector

    anchor_type = _new_stack_anchor_type(language)
    anchor = EngineFactory.get_engine(anchor_type, settings.alignment)
    if not anchor.is_available():
        raise RuntimeError(f"{anchor_type} anchor engine not available")

    results = anchor.align(align_audio, lyric_lines, language=language)
    raw_spans = [(r.start_time, r.end_time) for r in results]
    fixes: dict[int, list[str]] = {}

    vocals = sep_result.vocals if sep_result is not None else None
    accompaniment = sep_result.accompaniment if sep_result is not None else None

    # 우세도(dominance) — display_fixes의 장치 전부와 늘이기 게이트가 공유하는 신호.
    # VAD는 분리 스템의 간주 블리드에 죽으므로(display_fixes.py 모듈 docstring) 따로
    # 만든다. 스템이 없으면(분리 실패/미설치) 이 신호에 기대는 장치는 전부 조용히
    # 건너뛴다 — display_fixes.py 모듈 docstring의 계약("우세도가 없으면 전부 조용히
    # 건너뛴다") 그대로.
    activity = None
    if vocals is not None and accompaniment is not None:
        activity = df.dominance_activity_from_waveforms(
            vocals.waveform, accompaniment.waveform, vocals.sample_rate
        )

    if activity is not None:
        df._clamp_pathological(results, activity.regions)
        _diff_fixes(fixes, "dom-clamp", raw_spans, results)
        raw_spans = [(r.start_time, r.end_time) for r in results]

    vad_regions: list[tuple[float, float]] | None = None
    clamped_lines: set[int] = set()
    if vocals is not None:
        try:
            vad_result = VocalActivityDetector().detect(vocals)
            pp = TimingPostProcessor(settings.segmentation, extend_to_vocal=False).process(
                results, vad_result, "line"
            )
            _diff_fixes(fixes, "pp", raw_spans, pp.results, tol=0.2)
            results = pp.results
            results, clamped_lines = _clamp_stretched_lines(results, vad_result, fixes=fixes)
            vad_regions = [(round(reg.start, 2), round(reg.end, 2)) for reg in vad_result.regions]
        except Exception:
            logger.exception("New-stack VAD timing post-process failed; keeping anchor timing")

    # 우세도 기반 좌초 보정은 VAD 보정 **뒤에** 건다 — VAD가 분리 스템의 간주 블리드로
    # 못 잡는 좌초를 우세도가 잡고(display_fixes.py 모듈 docstring), 이 둘로 최종
    # 확정된 경계 위에서만 아래 2패스가 음절을 다시 잡아야 그 스팬이 나중에 어긋나지
    # 않는다("실행 순서" 문단의 doctring 참고).
    if activity is not None:
        before_stranded = [(r.start_time, r.end_time) for r in results]
        df.apply_stranded_corrections(results, activity)
        _diff_fixes(fixes, "stranded", before_stranded, results)

    adlib: list[tuple[float, float]] | None = None
    if activity is not None:
        adlib = df.adlib_candidates(results, activity)

    pron_data: dict[int, dict[str, Any]] = {}
    if not settings.alignment.two_pass_enabled:
        logger.info("Two-pass refiner disabled (two_pass_enabled=False); anchor-only segments")
    elif sep_result is None:
        logger.info("Two-pass refiner needs separated vocals; skipping (no separation result)")
    else:
        report("음절 재정렬")
        refiner = (
            anchor
            if anchor_type == "omniasr"
            else EngineFactory.get_engine("omniasr", settings.alignment)
        )
        stems_dir: Path | None = None
        try:
            stems_dir, vocals_path = _write_stems_for_two_pass(sep_result)
            bridged = _PathBridgedRefiner(refiner, AudioLoader())
            lines_text = [ln.text for ln in lyric_lines]
            refined = refine_lines(
                results,
                lines_text,
                bridged,
                vocals_path,
                language=language or "en",
                config=TwoPassRefineConfig(),
            )
            for i, rl in enumerate(refined):
                entry: dict[str, Any] = {}
                if rl.pron.get("hangul"):
                    entry["pronunciation"] = rl.pron["hangul"]
                if rl.pron_segs.get("hangul"):
                    entry["pron_segments"] = [_pron_seg_to_wire(s) for s in rl.pron_segs["hangul"]]
                if rl.pron:
                    entry["pron"] = dict(rl.pron)
                if rl.pron_segs:
                    entry["pron_segs"] = {
                        key: [_pron_seg_to_wire(s) for s in segs] for key, segs in rl.pron_segs.items()
                    }
                if entry:
                    pron_data[i] = entry
        except Exception:
            logger.exception("Two-pass refine failed; falling back to anchor-only segments")
        finally:
            if stems_dir is not None:
                shutil.rmtree(stems_dir, ignore_errors=True)

    alignment_text = f"{anchor_type}-2pass" if pron_data else anchor_type

    return _NewStackResult(
        results=results,
        alignment_text=alignment_text,
        pron_data=pron_data,
        fixes=fixes,
        raw_spans=raw_spans,
        vad_regions=vad_regions,
        clamped_lines=clamped_lines,
        engine=anchor,
        adlib=adlib,
    )


def _finish_new_stack_alignment(
    audio: Any,
    align_audio: Any,
    sep_result: Any,
    vocals: Any,
    lyric_lines: list[Any],
    language: str | None,
    settings: Any,
    report: Any,
    gloss_folded: Any,
    melody_extractor: Any,
    f0_future: Any,
    f0_executor: Any,
) -> dict[str, Any]:
    """새 스택 정렬을 실행하고 ``_run_alignment``과 같은 모양의 응답 dict를 조립한다.

    타임스탬프 직렬화 → gloss 되붙이기 → 다표기 부착 → 멜로디 → 품질 → debug_meta 꼬리는
    레거시 분기(``_run_alignment`` 본문, ``report("전사 정렬")`` 이후)의 같은 단계를
    의도적으로 **복제**했다 — 둘 다 ``getattr(engine, ..., default)`` 가드 위주라 원래는
    공유해도 안전하지만, 레거시 쪽 수백 줄 블록을 들여쓰기 수술로 감싸는 것보다 이 함수
    하나로 **조기 반환**(``_run_alignment``의 try/finally가 f0_executor·anchor_executor
    정리를 그대로 수행한다)하는 편이 기존 경로를 한 글자도 건드리지 않는다 — 회귀 표면을
    0으로 유지하는 쪽을 골랐다. 두 사본이 갈리면(예: quality_score 계산 방식이 바뀌면)
    여기도 함께 고쳐야 한다는 뜻이니 그 사실을 남겨 둔다.

    캡션 앵커·star span·caption scaffold는 새 스택에 배선하지 않았으므로
    (``_run_new_stack_alignment`` docstring 참고) debug_meta에 그 키들이 아예 없다 —
    레거시 응답의 "기능이 꺼져 있어 없음"과 달리 "이 스택엔 그 개념이 없음"이다.
    """
    stack = _run_new_stack_alignment(
        audio, align_audio, sep_result, lyric_lines, language, settings, report
    )
    results = stack.results
    pron_data = stack.pron_data
    fixes = stack.fixes
    raw_spans = stack.raw_spans
    vad_regions = stack.vad_regions
    clamped_lines = stack.clamped_lines
    engine = stack.engine
    alignment_text = stack.alignment_text

    timestamps: list[dict[str, Any]] = []
    pron_referee_tokens: list[Any] = []
    for i, r in enumerate(results):
        seg: dict[str, Any] = {"text": r.text, "start": r.start_time, "end": r.end_time}
        line_conf = r.confidence
        if line_conf is None and r.word_segments:
            line_conf = _geomean([w.confidence for w in r.word_segments])
        if line_conf is not None:
            seg["confidence"] = round(line_conf, 6)
        if r.word_segments:
            seg["words"] = _full_coverage_words(r.text, r.word_segments, r.start_time, r.end_time)
        pd = pron_data.get(i) or {}
        if pd.get("pronunciation"):
            seg["pronunciation"] = pd["pronunciation"]
        if pd.get("pron_segments"):
            seg["pron_segments"] = pd["pron_segments"]
        # 다표기(pron/pron_segs) — 레거시엔 없는 새 필드지만 attach_pron_variants가
        # seg.get("pron")로 멱등 가드를 거니 순서를 맞출 필요가 없다(먼저 실어도 안전 —
        # 아래 attach_pron_variants가 그대로 스킵한다).
        if pd.get("pron"):
            seg["pron"] = pd["pron"]
        if pd.get("pron_segs"):
            seg["pron_segs"] = pd["pron_segs"]
        debug: dict[str, Any] = {}
        if vad_regions is not None:
            dur = max(0.001, r.end_time - r.start_time)
            vocal = sum(
                max(0.0, min(e, r.end_time) - max(s, r.start_time)) for s, e in vad_regions
            )
            debug["active_ratio"] = round(vocal / dur, 2)
            debug["clamped"] = i in clamped_lines
            fx = fixes.get(i)
            if fx:
                debug["orig"] = [round(raw_spans[i][0], 2), round(raw_spans[i][1], 2)]
                debug["fixes"] = fx
        if debug:
            seg["debug"] = debug
        pron_referee_tokens.append(None)
        timestamps.append(seg)

    if gloss_folded:
        attached = _fold_gloss_into_segments(timestamps, gloss_folded)
        logger.info(
            f"Re-attached {attached} excluded gloss line(s) to their source segment "
            f"for display (alignment input untouched)"
        )

    for seg, referee_tokens in zip(timestamps, pron_referee_tokens):
        attach_pron_variants(seg, referee_tokens=referee_tokens)

    if vad_regions is not None:
        for text in _drop_nonvocal_nonlyric_edges(timestamps):
            logger.info(f"Dropped non-vocal non-lyric edge line: {text!r}")

    report("멜로디 분석")
    f0_curve = None
    song_key = None
    if melody_extractor is not None:
        try:
            precomputed_f0 = f0_future.result() if f0_future is not None else None
            annotated = melody_extractor.annotate_timestamps(
                audio, timestamps, vocals=vocals, precomputed_f0=precomputed_f0
            )
            f0_curve = melody_extractor.last_f0_curve
            song_key = melody_extractor.last_key
            logger.info(f"Melody notes annotated on {annotated} spans")
        except Exception:
            logger.exception("Melody extraction failed; continuing without notes")
        finally:
            if f0_executor is not None:
                f0_executor.shutdown(wait=True)

    avg_confidence = None
    confidences = [t.get("confidence") for t in timestamps if t.get("confidence") is not None]
    if confidences:
        avg_confidence = sum(confidences) / len(confidences)

    quality_score, coverage_meta = _quality_with_coverage(
        avg_confidence, len(confidences), len(timestamps)
    )
    if coverage_meta.get("failed"):
        logger.warning(
            f"Alignment coverage too low: only {coverage_meta['aligned_lines']}/"
            f"{coverage_meta['total_lines']} line(s) have measured char timing "
            f"(the rest are interpolated); reporting quality_score="
            f"{quality_score} instead of {coverage_meta['measured_conf']}"
        )

    detected_lang = language
    engine_variant = getattr(engine, "_current_engine_variant", None)
    if hasattr(engine, "_current_language"):
        detected_lang = engine._current_language

    quality_adapter = getattr(engine, "_current_adapter", None)
    quality_norm = _scale_free_quality(avg_confidence, quality_adapter)
    debug_meta: dict[str, Any] = {
        "star_spans": [],
        "vad_regions": [list(v) for v in vad_regions] if vad_regions is not None else None,
        "alignment_text": alignment_text,
        "f0_curve": f0_curve,
        "quality_adapter": quality_adapter,
        "quality_norm": None if quality_norm is None else round(quality_norm, 6),
        "align_coverage": coverage_meta,
    }

    return {
        "timestamps": timestamps,
        "language": detected_lang,
        "engine_variant": engine_variant,
        "quality_score": quality_score,
        "debug": debug_meta,
        "alignment_text": alignment_text,
        "tempo": _estimate_tempo(audio),
        "key": song_key,
        # 곡 단위 추임새 후보 — 새 스택 전용 additive 필드(레거시 응답엔 없다). 프런트
        # 계약: [[start,end],...] (팀리드 지시 필드명 "adlib" 그대로).
        "adlib": [[round(s, 3), round(e, 3)] for s, e in stack.adlib] if stack.adlib else None,
    }


def _run_alignment(
    audio_path: str,
    lyrics: str,
    language: str | None,
    line_meta: list[dict[str, Any]] | None = None,
    on_stage: Any | None = None,
    line_meta_resolver: Any | None = None,
    video_id: str | None = None,
) -> dict:
    """정렬 본체. line_meta_resolver를 주면 **보컬 분리·f0 착수 뒤, CTC 진입 직전에** 한 번
    불러 line_meta를 늦게 받아온다 (번역·독음을 클라이언트가 병렬로 만드는 경로).
    리졸버가 None을 돌려주면(상한 초과) 원문 정렬로 그대로 진행한다.

    video_id를 주고 ``caption_anchors``가 켜져 있으면 사람이 만든 유튜브 자막의 타임스탬프에서
    «가사 줄이 놓일 수 없는 구간»을 뽑아 정렬에 제약으로 넣는다 (``_caption_forbidden_spans``).
    가사 출처와 무관한 별개 신호이므로 자막으로 만든 싱크가 아니어도 동작한다."""
    from everyric2.audio.loader import AudioLoader
    from everyric2.config.settings import get_settings
    from everyric2.inference.prompt import LyricLine

    def report(stage: str) -> None:
        # 진행 단계 보고 — 실패해도 정렬 자체는 계속한다
        if on_stage is not None:
            try:
                on_stage(stage)
            except Exception:
                pass

    settings = get_settings()
    loader = AudioLoader()
    audio_path_obj = Path(audio_path)

    # WS2-B 병렬 f0 실행기 — 정렬 도중 예외가 나도 outer finally가 반드시 정리하도록 밖에 둔다
    f0_executor = None
    # 자막 앵커 조달 실행기 — 같은 사정으로 밖에 둔다 (정렬 진입 전 예외 경로)
    anchor_executor = None

    try:
        audio = loader.load(audio_path_obj)
        lyric_lines = LyricLine.from_text(lyrics)

        # 번역·독음 병기 시트의 비가창 줄을 정렬 입력에서 제외한다 (표시용으로는 아래에서
        # 원문 세그먼트에 되붙인다). CTC 진입 전에 해야 _pron_coverage도 가창 줄만으로
        # 계산돼 병기 시트가 독음 정렬 경로를 커버리지 미달로 떨어뜨리지 않는다.
        lyric_lines, gloss_folded = _split_gloss_lines(
            lyric_lines, settings.alignment.exclude_gloss_lines
        )

        # 새 정렬 스택(owsm/omniasr 앵커) 게이트 — 켜져 있으면 구스택 CTC 엔진은 아예
        # 웜업하지 않는다(안 쓸 모델을 GPU에 올려 둘 이유가 없다). 이 값 하나로 아래
        # 두 지점(엔진 웜업 스킵, 앵커/구스택 이중정렬 분기)이 갈린다.
        new_stack_active = _new_stack_enabled(settings)

        engine = None
        if not new_stack_active:
            # CTC 엔진은 웜 캐시 싱글턴 — 같은 언어의 두 번째 잡부터 모델 재로드 0회 (WS2-A).
            # torch를 최상위 import하는 모듈이라 반드시 여기서 지연 import한다 (API 전용 모드
            # 프로세스에 torch가 딸려 들어오지 않게 — main.py 지연 임포트 계약).
            from everyric2.alignment.ctc_engine import get_shared_ctc_engine

            engine = get_shared_ctc_engine(settings.alignment)
            if not engine.is_available():
                raise RuntimeError("CTC engine not available")

        # 자막 앵커 조달은 네트워크 IO(트랙당 yt-dlp 1회)라 보컬 분리와 **겹쳐서** 돌린다 —
        # 정렬 진입 전에만 있으면 되므로 분리 시간에 그대로 숨는다(f0 병렬과 같은 방식).
        anchor_future = None
        if (
            settings.alignment.caption_anchors or settings.alignment.caption_scaffold
        ) and video_id:
            import concurrent.futures

            anchor_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            anchor_future = anchor_executor.submit(
                _caption_forbidden_spans,
                video_id,
                lyric_lines,
                audio.duration,
                settings.alignment,
            )

        # 보컬 스템 1회 분리 — 원 설계(CLI --separate)대로 정렬 입력으로 쓰고, 아래 VAD
        # 라인 경계 보정과 멜로디 f0 추출에 재사용한다. 반주가 빠진 스템은 CTC emission이
        # 훨씬 깨끗해 고밀도 믹스/이펙트 구간에서 정렬 품질이 오른다. 미설치/실패 시 믹스 폴백.
        report("보컬 분리")
        need_vocals = settings.melody.separate_vocals or settings.alignment.align_on_vocals
        sep_result = _separate_stems(audio) if need_vocals else None
        vocals = sep_result.vocals if sep_result is not None else None
        align_audio = (
            vocals if (vocals is not None and settings.alignment.align_on_vocals) else audio
        )

        # WS2-B: 멜로디 f0 전곡 추론을 CTC 정렬과 병렬로 시작한다 — f0 추론은 정렬 결과에
        # 무의존이라(전곡 신호만 처리) GPU 유휴를 줄인다. 노트 부착(annotate)은 정렬·타이밍
        # 보정이 끝난 뒤 이 f0를 주입해 수행한다. 진행 stage 표시 순서(보컬 분리→전사 정렬→
        # 타이밍 보정→멜로디 분석)는 그대로 — f0는 백그라운드라 보고 stage를 바꾸지 않는다.
        # 멜로디 실패는 비치명(노트 없이 계속)이므로 여기서 예외를 삼키지 않고 result()에서 처리.
        f0_future = None
        melody_extractor = None
        if settings.melody.enabled:
            from everyric2.melody.extractor import get_shared_extractor

            melody_extractor = get_shared_extractor(settings.melody)
            if melody_extractor.is_available():
                import concurrent.futures

                f0_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                f0_future = f0_executor.submit(melody_extractor.precompute_f0, audio, vocals)
            else:
                logger.warning("Melody enabled but torchfcpe is not installed; skipping")
                melody_extractor = None

        # line_meta(발음·번역)가 아직 안 왔으면 여기서 기다린다 — 위의 오디오 로드·CTC 모델
        # 웜업·보컬 분리는 이미 끝났고 f0 전곡 추론은 백그라운드에서 계속 돌므로, 대기 시간이
        # 그 작업들과 겹쳐 사라진다. 상한 초과는 None(원문 정렬 폴백), 취소는 JobCancelled.
        if line_meta_resolver is not None:
            report(LINE_META_WAIT_STAGE)
            line_meta = line_meta_resolver()

        # 자막 앵커 결과 수거 — 여기서만 기다린다(보컬 분리와 겹쳐 돌았다). 조달 실패는
        # 빈 계획이라 정렬은 기존 경로 그대로 간다.
        anchor_plan = None
        anchor_kw: dict[str, Any] = {}
        if anchor_future is not None:
            try:
                anchor_plan = anchor_future.result()
                # 제약별 스위치: 금지 구간·양성 제약 모두 스위치가 켜진 것만 정렬에 들어간다.
                # 스캐폴드만 켜진 배포에서는 계획은 있되 DP 제약은 0개다 — 스캐폴드는
                # 정렬 «결과»를 고치는 후처리라 여기서 아무것도 걸지 않는다.
                anchor_kw = _anchor_kwargs(
                    anchor_plan.spans if settings.alignment.caption_anchors else None,
                    anchor_plan.line_starts
                    if settings.alignment.caption_anchor_positive
                    else None,
                )
            except Exception:
                logger.exception("Caption anchor thread failed; aligning without anchors")
            finally:
                if anchor_executor is not None:
                    anchor_executor.shutdown(wait=True)
                    anchor_executor = None

        # star 성형: star 채널의 프레임별 가격 신호를 만든다 (star_prior.py).
        # 1순위는 보컬/반주 스템의 **우세도** — f0 유성 지시자는 분리 스템 위에서 죽는다
        # (실측: 사고 곡 간주의 f0 presence 0.979 vs 우세도 0.199, star_prior.py 주석).
        # 순수 numpy라 대기가 없다. 반주 스템이 없을 때만 f0 폴백을 기다린다.
        # 실패·미가용은 평평한 star로 계속한다(성형은 있으면 좋은 신호다). anchor_kw에
        # 싣는 이유: ko/ja/이중정렬/재정렬이 전부 이 dict를 쓰므로 모든 정렬이 같은
        # 가격을 본다 — 한쪽만 성형하면 누출 가드가 성형 차이를 누출로 오독한다.
        if settings.alignment.star_prior and settings.alignment.star_tokens and not new_stack_active:
            presence = None
            try:
                if sep_result is not None:
                    from everyric2.alignment.star_prior import vocal_presence_from_stems

                    presence = vocal_presence_from_stems(
                        sep_result.vocals.waveform,
                        sep_result.accompaniment.waveform,
                        sep_result.vocals.sample_rate,
                        settings.alignment.star_prior_smooth_sec,
                    )
                    src = "stem dominance"
                elif f0_future is not None:
                    t0 = time.monotonic()
                    f0_hz, f0_times = f0_future.result()
                    from everyric2.alignment.star_prior import vocal_presence_from_f0

                    presence = vocal_presence_from_f0(
                        f0_hz, f0_times, settings.alignment.star_prior_smooth_sec
                    )
                    src = f"f0 fallback (waited {time.monotonic() - t0:.1f}s)"
                else:
                    src = "no signal"
                if presence is None:
                    logger.info(f"Star prior: no usable presence ({src}); star stays flat")
                else:
                    anchor_kw["vocal_presence"] = presence
                    logger.info(
                        f"Star prior: presence from {src} ({len(presence[1])} frames)"
                    )
            except Exception:
                logger.exception("Star prior: presence derivation failed; star stays flat")

        report("전사 정렬")
        if new_stack_active:
            # 새 스택은 여기서 조기 반환한다 — 아래 ko/ja 이중정렬·star 토큰·pron_data DP
            # 근사 블록(레거시 전용, 수백 줄)은 건드리지 않고 그대로 폴백 경로로 남긴다.
            # try/finally(f0_executor·anchor_executor 정리)는 조기 return에도 그대로 돈다.
            return _finish_new_stack_alignment(
                audio,
                align_audio,
                sep_result,
                vocals,
                lyric_lines,
                language,
                settings,
                report,
                gloss_folded,
                melody_extractor,
                f0_future,
                f0_executor,
            )
        # 독음(ko) 정렬 경로: 커버리지가 충분하면 한국어 발음 텍스트+kor adapter로 정렬하고
        # 원문 라인에 역매핑한다. 미달/실패 시 원문 정렬로 폴백 (회귀 0).
        by_text = _pron_by_text(line_meta)
        coverage = _pron_coverage(lyric_lines, by_text)
        pron_data: dict[int, dict[str, Any]] | None = None
        alignment_text = "original"
        # 원문 정렬 경로의 「들은 것」 — 독음 경로는 ko 라인 창 기준이라 pron_data가 들고 간다.
        # ja를 나중에 채택하는 분기들은 이 dict를 비워 둔다(엉뚱한 라인 창의 텍스트를 보고하지
        # 않기 위해). 심판 진단은 독음 경로에서만 의미가 있으므로 손실이 없다.
        heard_by_line: dict[int, str] = {}
        heard_spans_by_line: dict[int, list] = {}
        if settings.alignment.use_pronunciation and coverage >= 0.9:
            try:
                results, pron_data = _align_with_pronunciation(
                    engine,
                    align_audio,
                    lyric_lines,
                    by_text,
                    settings.alignment,
                    anchor_kw=anchor_kw,
                )
                alignment_text = "pronunciation"
                logger.info(f"Pronunciation alignment used (coverage={coverage:.2f})")
            except Exception:
                logger.exception("Pronunciation alignment failed; falling back to original text")
                results = _align_original(
                    engine, align_audio, lyric_lines, language, anchor_kw
                )
                pron_data = None
                heard_by_line = dict(getattr(engine, "_last_heard", None) or {})
                heard_spans_by_line = dict(getattr(engine, "_last_heard_spans", None) or {})
        else:
            if settings.alignment.use_pronunciation:
                logger.info(f"Pronunciation coverage {coverage:.2f} < 0.9; using original text")
            results = _align_original(engine, align_audio, lyric_lines, language, anchor_kw)
            heard_by_line = dict(getattr(engine, "_last_heard", None) or {})
            heard_spans_by_line = dict(getattr(engine, "_last_heard_spans", None) or {})

        # 자막 앵커 판정은 **ko 정렬 직후** 포착한다 — 아래 ja 교차정렬이 엔진의 직전-정렬
        # 기록을 덮는다(pron_star_spans·심판 판정과 같은 사정).
        anchor_decision = getattr(engine, "_last_caption_anchor", None)

        # 독음 정렬의 star span (아래 VAD 확보 후 '발성 삼킴' 게이트에 쓴다) — 이중정렬/가드
        # ja가 engine._last_star_spans를 덮으므로 어떤 재정렬보다 먼저 ko 정렬 직후 포착한다.
        pron_star_spans = (
            list(getattr(engine, "_last_star_spans", []))
            if alignment_text == "pronunciation"
            else []
        )

        # 저신뢰 이중정렬 안전망: 커버리지≥0.9면 ko 정렬은 품질과 무관하게 유지되는데,
        # 합성보컬은 posterior가 균일 바닥이라 ko가 성공해도 곡 전체가 저품질일 수 있다.
        # ko 평균 신뢰도가 임계 미만이면 ja 원문 정렬도 돌려 명확히 나은 쪽을 채택한다
        # (熱異常: ja도 같이 붕괴 → margin 미달로 ko 유지; 발음 정렬만 나쁜 곡은 ja 채택).
        # 여기에 더해, 원문(ja) 정렬은 융합 스위치가 켜져 있으면 **상시** 돌린다 — ko 경로의 원문 글자
        # 타이밍은 역매핑 합성물이라 라인 내부 분포를 ja 실측으로 갈아끼우려면 항상 필요하다
        # (아래 _fuse_original_char_timing). 비용은 CTC 1패스(4.7분 곡 ~9s)뿐이다: ko/ja는
        # 같은 mms-1b-all 베이스라 언어 전환이 어댑터 스왑(0.23s)이다. 여기서 한 번 돌린
        # 결과를 이중정렬 안전망과 역누출 가드가 모두 재사용한다(중복 정렬 0).
        # 이 정렬에 실제로 쓰인 MMS 어댑터 — conf는 어댑터 vocab 크기에 스케일 의존하므로
        # 아래 ko/ja 교차비교와 debug 보고의 스케일 무관 품질 산출에 쓴다. 어댑터는 다음
        # align 호출에서 교체되므로 **각 정렬 직후에** 포착해야 한다.
        align_adapter = getattr(engine, "_current_adapter", None)

        ja_alignment = None
        ja_adapter = align_adapter
        if alignment_text == "pronunciation":
            ko_conf = _avg_line_confidence(results)
            dual_check = _dual_align_should_run(ko_conf, settings.alignment.dual_align_conf)
            if _original_align_needed(
                ko_conf,
                settings.alignment.dual_align_conf,
                settings.alignment.fuse_original_chars,
            ):
                try:
                    ja_alignment = _align_original(
                        engine, align_audio, lyric_lines, language, anchor_kw
                    )
                    ja_adapter = getattr(engine, "_current_adapter", None)
                except Exception:
                    logger.exception(
                        "Original-text (ja) alignment failed; keeping pronunciation alignment "
                        "as-is (no dual-align cross-check, no original-char fusion)"
                    )
            if dual_check and ja_alignment is not None:
                try:
                    ja_conf_raw = _avg_line_confidence(ja_alignment)
                    # ja는 다른 어댑터(jpn/cmn)로 측정될 수 있고 vocab 크기가 다르면 conf
                    # 스케일 자체가 달라 raw 비율 비교가 성립하지 않는다. ko 어댑터 스케일로
                    # 옮겨 비교한다 — 어댑터가 같으면 항등이라 기존 판정은 그대로다.
                    ja_conf = _rescale_conf(ja_conf_raw, ja_adapter, align_adapter)
                    if _dual_align_prefers_original(
                        ko_conf, ja_conf, settings.alignment.dual_align_min_ratio
                    ):
                        logger.warning(
                            f"Low-confidence pronunciation alignment (avg conf {ko_conf:.5f} < "
                            f"{settings.alignment.dual_align_conf}); original-text alignment scores "
                            f"{ja_conf:.5f} (adapter {ja_adapter} raw {ja_conf_raw}, rescaled to "
                            f"{align_adapter} scale) (>= "
                            f"{settings.alignment.dual_align_min_ratio}x) — adopting original text"
                        )
                        results = ja_alignment
                        alignment_text = "original"
                        pron_data = None
                    else:
                        ja_repr = f"{ja_conf:.5f}" if ja_conf is not None else "n/a"
                        logger.info(
                            f"Low-confidence pronunciation alignment (avg conf {ko_conf:.5f}) but "
                            f"original text scores {ja_repr} "
                            f"(< {settings.alignment.dual_align_min_ratio}x) — keeping pronunciation"
                        )
                except Exception:
                    logger.exception("Dual-align cross-check failed; keeping pronunciation alignment")

        # VAD로 라인 경계 보정 — 가사에 없는 추임새/간주로 늘어진 라인을 실제 발성 구간으로
        report("타이밍 보정")
        vad_regions: list[tuple[float, float]] | None = None
        clamped_lines: set[int] = set()
        # 보정 전 원본(raw CTC) 타이밍 + 규칙별 보정 라벨 — 확장 디버그 오버레이용
        raw_spans = [(r.start_time, r.end_time) for r in results]
        fixes: dict[int, list[str]] = {}
        if vocals is not None:
            try:
                from everyric2.alignment.timing_postprocess import TimingPostProcessor
                from everyric2.audio.vad import VocalActivityDetector

                vad_result = VocalActivityDetector().detect(vocals)
                # 독음 정렬이 실제 발성을 star 와일드카드로 삼켰는지 검사한다. star 하나가
                # 후반 가창을 통째로 흡수하면 그 라인들이 앞으로 압축·오배치된다
                # (VWVtIg5cdDU(初音ミクの消失) 실측: star 한 개가 후반 가창 21s를 삼켜
                # 후반 라인이 ~40s 앞으로 압축, 불가능한 음절 속도). 단 삼킴 크기만으론
                # 熱異常(브릿지에서 20.7s 삼키지만 배치 정상)과 못 가른다 — 그래서 이건
                # 비용 게이트로만 쓰고, 판정은 '간주 이후 발성 창을 어느 정렬이 채우는가'로
                # 한다. ko가 그 창을 비우고(라인을 앞으로 압축) ja가 크게 채우면 ja 폴백,
                # 둘이 비슷하면(熱異常: ja도 채움, 배치 차이는 국소) ko 유지.
                if alignment_text == "pronunciation" and settings.alignment.star_vocal_fallback_sec > 0:
                    swallowed = _star_swallowed_vocal(pron_star_spans, vad_result.regions)
                    windows = _post_interlude_windows(
                        vad_result.regions, settings.alignment.interlude_min_gap_sec
                    )
                    post_win = _post_interlude_window(
                        vad_result.regions, settings.alignment.interlude_min_gap_sec
                    )
                    # star가 실보컬을 크게 삼켰거나(전체 압축) 라인이 간주를 가로지르면
                    # (라인 사이 간극 또는 간주를 통째로 덮는 라인 — vad_regions로 판정)
                    # (선두 라인만 뒤로 새는 약한 누출도 가능) ja 정렬을 한 번 돌려 대조한다.
                    # 삼킴이 작아도(무음 위 star: 初音ミクの消失 1.02s) straddle이면 검사 —
                    # 실제 라인 이동은 아래 ja 대조만 결정하므로 게이트가 넓어도 안전하다.
                    run_cross_check = bool(windows) and (
                        swallowed >= settings.alignment.star_vocal_fallback_sec
                        or _straddles_interlude(
                            results,
                            settings.alignment.interlude_min_gap_sec,
                            vad_result.regions,
                        )
                    )
                    if run_cross_check:
                        # 이중정렬 안전망이 이미 돌린 ja가 있으면 재사용 (같은 audio/lyrics)
                        ja_candidate = (
                            ja_alignment
                            if ja_alignment is not None
                            else _align_original(
                                engine, align_audio, lyric_lines, language, anchor_kw
                            )
                        )
                        # 1) 역방향 누출(모든 간주): ja가 간주 이후에 두는데 ko가 앞으로 뺀
                        #    라인들의 변위 런 중 크게(>= leak_min) 밀린 것만 골라 그 라인들의
                        #    타이밍을 ja로 외과 교체한다. 선두 한 줄만 새는 경우(idx46)와
                        #    블록 전체 압축(idx46-52)을 같은 방식으로 포착, 정상 라인은 보존.
                        runs = _leaked_runs(
                            results,
                            ja_candidate,
                            windows,
                            settings.alignment.post_interlude_leak_lead_sec,
                        )
                        leaked = [
                            i
                            for run in runs
                            if max(
                                ja_candidate[k].start_time - results[k].start_time for k in run
                            )
                            >= settings.alignment.post_interlude_leak_min_sec
                            for i in run
                        ]
                        if leaked and settings.alignment.star_guard_splice:
                            # 교체된 라인의 ko pron_segments는 압축된 타이밍이라 무효 —
                            # 스팬만 버리면 발음·번역 텍스트는 남고, 캐시 재병합이 ja
                            # 타이밍 기반 DP 근사로 노트 스팬을 복원한다.
                            pre_leak = {i: (results[i].start_time, results[i].end_time) for i in leaked}
                            logger.warning(
                                f"Pronunciation alignment leaked {len(leaked)} line(s) back across "
                                f"{len(windows)} interlude(s) (idx {leaked[0]}..{leaked[-1]}; star "
                                f"swallowed {swallowed:.1f}s); splicing ja timing onto leaked lines"
                            )
                            _apply_leak_splice(results, ja_candidate, leaked)
                            alignment_text = "spliced"
                            if pron_data:
                                for idx in leaked:
                                    pd = pron_data.get(idx)
                                    if pd:
                                        pd["pron_segments"] = None
                            raw_spans = [(r.start_time, r.end_time) for r in results]
                            fixes = {}
                            _mark_leak_ghosts(raw_spans, fixes, pre_leak, results)
                        elif leaked:
                            # 스플라이스 비활성 — 누출 확인 시 전곡 원문 폴백
                            pre_leak = {
                                i: (results[i].start_time, results[i].end_time)
                                for i in range(len(results))
                            }
                            logger.warning(
                                f"Pronunciation alignment leaked {len(leaked)} line(s) across "
                                f"interlude(s) (splice disabled); falling back to original-text "
                                f"alignment"
                            )
                            results = ja_candidate
                            alignment_text = "original"
                            pron_data = None
                            raw_spans = [(r.start_time, r.end_time) for r in results]
                            fixes = {}
                            _mark_leak_ghosts(raw_spans, fixes, pre_leak, results)
                        elif swallowed >= settings.alignment.star_vocal_fallback_sec and post_win:
                            # 2) 역누출 런은 없지만 삼킴 게이트가 트립됐으면, 레거시 전창
                            #    점유 대조로 극단적 전체 압축(창을 통째로 비운 경우)을 안전망으로.
                            ko_fill = _lines_span_overlap(results, post_win)
                            ja_fill = _lines_span_overlap(ja_candidate, post_win)
                            if ja_fill - ko_fill >= settings.alignment.post_interlude_fill_margin_sec:
                                pre_leak = {
                                    i: (results[i].start_time, results[i].end_time)
                                    for i in range(len(results))
                                }
                                splice_k = (
                                    _splice_alignments(results, ja_candidate, post_win)
                                    if settings.alignment.star_guard_splice
                                    else None
                                )
                                if splice_k is not None:
                                    logger.warning(
                                        f"Pronunciation alignment vacated the post-interlude window "
                                        f"[{post_win[0]:.1f}-{post_win[1]:.1f}]s (ko fills "
                                        f"{ko_fill:.1f}s vs ja {ja_fill:.1f}s; star swallowed "
                                        f"{swallowed:.1f}s); splicing ko[:{splice_k}] + ja[{splice_k}:]"
                                    )
                                    alignment_text = "spliced"
                                    if pron_data:
                                        for idx in range(splice_k, len(results)):
                                            pd = pron_data.get(idx)
                                            if pd:
                                                pd["pron_segments"] = None
                                else:
                                    logger.warning(
                                        f"Pronunciation alignment vacated the post-interlude window "
                                        f"[{post_win[0]:.1f}-{post_win[1]:.1f}]s (ko fills "
                                        f"{ko_fill:.1f}s vs ja {ja_fill:.1f}s, +{ja_fill - ko_fill:.1f}s; "
                                        f"star swallowed {swallowed:.1f}s); falling back to "
                                        f"original-text alignment"
                                    )
                                    results = ja_candidate
                                    alignment_text = "original"
                                    pron_data = None
                                raw_spans = [(r.start_time, r.end_time) for r in results]
                                fixes = {}
                                _mark_leak_ghosts(raw_spans, fixes, pre_leak, results)
                            else:
                                logger.info(
                                    f"Star swallowed {swallowed:.1f}s but both alignments fill the "
                                    f"post-interlude window similarly (ko {ko_fill:.1f}s, ja "
                                    f"{ja_fill:.1f}s, +{ja_fill - ko_fill:.1f}s < "
                                    f"{settings.alignment.post_interlude_fill_margin_sec}s); keeping "
                                    f"pronunciation alignment"
                                )

                        else:
                            logger.info(
                                f"No reverse-leak across {len(windows)} interlude(s) "
                                f"(star swallowed {swallowed:.1f}s); keeping pronunciation alignment"
                            )
                # extend_to_vocal은 끄는다: 가사에 없는 반복 가창/애드립도 "보컬 활동"이라
                # 라인을 그쪽으로 늘려버린다 (star 토큰이 흡수해 둔 구간을 도로 끌어안는 역효과)
                pp = TimingPostProcessor(settings.segmentation, extend_to_vocal=False).process(
                    results, vad_result, "line"
                )
                # 0.2s 넘게 움직인 라인만 pp 라벨 — 미세 조정까지 고스트로 그리면 소음
                _diff_fixes(fixes, "pp", raw_spans, pp.results, tol=0.2)
                # 독음 정렬의 무음 언더슛(전이 라인이 간주에 좌초) 교정 — ko 경로에만 적용.
                # _clamp_stretched_lines(내부 _pull이 간주 후 첫 라인을 당김) **이전에** 돌려
                # 좌초 라인이 먼저 다음 온셋을 잡게 한다 (뒤 라인 오인 당김 방지).
                snapped: set[int] = set()
                if alignment_text == "pronunciation":
                    snap_before = [(r.start_time, r.end_time) for r in pp.results]
                    _snap_silence_undershoot(pp.results, vad_result, snapped)
                    # 합성보컬 대량 역누출(긴 간주 앞 리프라이즈 붕괴)은 ja 대조로 못 잡으므로
                    # 간주 무음에 앵커해 재배치 — 무음 언더슛 스냅 다음, 클램프 이전에 돌린다.
                    _snap_post_interlude_leak(
                        pp.results,
                        vad_result,
                        snapped,
                        settings.alignment.mass_leak_min_gap_sec,
                        settings.alignment.mass_leak_min_char_rate,
                        settings.alignment.mass_leak_max_coverage,
                    )
                    _diff_fixes(fixes, "snap", snap_before, pp.results)
                results, clamped_lines = _clamp_stretched_lines(pp.results, vad_result, fixes=fixes)
                clamped_lines |= snapped
                vad_regions = [(round(reg.start, 2), round(reg.end, 2)) for reg in vad_result.regions]
                logger.info(f"Timing post-process: {pp.stats}")
            except Exception:
                logger.exception("VAD timing post-process failed; using raw alignment")

        # 곡 단위 평균 신뢰도는 **융합 전에** 확정한다 — 융합은 글자 conf를 ja 실측값으로
        # 갈아끼우므로, 라인 conf가 비어 글자 conf 기하평균으로 백필되는 라인이 있으면
        # 아래 재합성의 곡 단위 게이트가 융합 여부에 따라 흔들린다.
        song_conf = _avg_line_confidence(results)

        # ko/ja 융합 (스냅·클램프로 라인 경계가 확정된 뒤, 재합성 직전): ko 라인 경계와
        # pron_segments는 그대로 두고 라인 **내부** 원문 글자 분포만 ja 실측값으로 교체한다.
        # 사상 목표 구간은 라인 경계가 아니라 그 라인의 발음 음절 구간이다 — pron_data를
        # 넘겨야 원문 글자와 발음 음절이 같은 봉투를 공유한다(_measured_vocal_window).
        # 이미 합성물(3단 역매핑)인 부분만 진짜 측정값으로 갈아끼우는 것이라 회귀 표면이 작다.
        # 재합성보다 **먼저** 도는 이유: 융합으로 분포가 정상화된 라인은 아래 구조 게이트
        # (_impossible_word_distribution)에 더 이상 안 걸려 균등 분배로 덮이지 않는다 —
        # 균등 분배는 더 나은 실측값이 없을 때의 폴백이어야 한다. 반대로 leak 라벨 라인은
        # 재합성이 그대로 덮으므로(기존 규칙 유지) 그 경로 동작은 바뀌지 않는다.
        fused_lines: set[int] = set()
        if pron_data is not None and settings.alignment.fuse_original_chars:
            fused_lines = _fuse_original_char_timing(
                results,
                ja_alignment,
                fixes,
                settings.alignment.mass_leak_min_char_rate,
                pron_data=pron_data,
                max_disagreement=settings.alignment.fuse_max_disagreement_sec,
            )
            if fused_lines:
                logger.info(
                    f"Fused measured original-text char timing into {len(fused_lines)}/"
                    f"{len(results)} line(s) (ko line bounds and pron_segments kept; chars "
                    f"mapped onto each line's measured pron-syllable window)"
                )

        # 융합되지 않은 줄의 역매핑 «뭉침»(한 음절 스팬이 여러 글자에 복사됨)을 글자 수
        # 균등으로 세분한다. 불일치 게이트로 ja 대신 역매핑이 남은 줄에서, 뭉친 글자들이
        # 한꺼번에 점등했다가 다음 그룹으로 순간이동하는 체감 회귀가 보고됐다(사용자
        # 2026-07-28). 그룹 «경계»는 ko 실측 그대로라 정확도는 불변이고 그룹 안에만
        # 흐름이 생긴다. 융합 줄은 ja 실측이라 건드리지 않는다.
        if pron_data is not None:
            subdivided = _subdivide_clumped_words(results, skip=fused_lines)
            if subdivided:
                logger.info(
                    f"Subdivided clumped back-mapped char spans on {subdivided} line(s)"
                )

        # 자막 스캐폴드 (모든 스냅·클램프 뒤, 재합성·직렬화 앞): 붕괴 곡의 줄 시작을
        # 사람이 찍은 자막 시각으로 고정한다 — DP 제약이 아니라 결과 교체다. 발동 여부와
        # 무관하게 판정 전체가 debug.caption_scaffold로 내려간다 (디버그 투명성).
        scaffold_meta: dict[str, Any] | None = None
        if settings.alignment.caption_scaffold:
            try:
                scaffold_meta = _apply_caption_scaffold(
                    results,
                    pron_data,
                    fixes,
                    anchor_plan,
                    song_conf,
                    audio.duration,
                    settings.alignment,
                )
            except Exception:
                logger.exception("Caption scaffold failed; keeping the aligned timing")
                scaffold_meta = {"applied": False, "skipped": "error"}

        # 보정 마지막(스냅·클램프 이후, 직렬화 전): 붕괴 곡/누출 라인의 라인 내부 word/pron
        # 타이밍을 균등 비례로 재합성한다. 라인 경계는 유지하고 무의미한 CTC 내부 분포만 폐기.
        # 멜로디 노트는 라인 스팬 기반(word/pron 무소비)이라 영향 없음.
        synth_lines = _synthesize_collapsed_timing(
            results,
            pron_data,
            fixes,
            song_conf,
            settings.alignment.synth_all_lines_conf,
            settings.alignment.mass_leak_min_char_rate,
        )
        if synth_lines:
            logger.info(
                f"Re-synthesized uniform intra-line timing on {len(synth_lines)} line(s)"
            )

        timestamps = []
        # attach_pron_variants 호출은 이 루프 밖(gloss 되붙이기 뒤)으로 미룬다 — referee_tokens만
        # 여기서 인덱스 순서대로 챙겨 둔다(아래 이유 참고).
        pron_referee_tokens: list[Any] = []
        for i, r in enumerate(results):
            seg = {
                "text": r.text,
                "start": r.start_time,
                "end": r.end_time,
            }
            # 원문 정렬 경로는 엔진이 라인 confidence를 안 채운다(word에만 존재) —
            # ko 경로와 동일하게 글자 conf의 기하평균으로 보충해 quality_score와
            # 레인/패널의 곡 단위 conf 통계가 모든 곡에서 동작하게 한다
            line_conf = r.confidence
            if line_conf is None and r.word_segments:
                line_conf = _geomean([w.confidence for w in r.word_segments])
            if line_conf is not None:
                seg["confidence"] = round(line_conf, 6)
            if r.word_segments:
                # 본문 글자를 1:1 완전히 덮도록 재구성 — join(words)==r.text 보장(공백·표기
                # 차이로 확장 글자 매핑이 죽어 통짜 점등되던 문제 해소). pron_segments는 불변.
                seg["words"] = _full_coverage_words(
                    r.text, r.word_segments, r.start_time, r.end_time
                )
            # 독음 정렬 경로: 발음 음절 스팬을 멜로디 앵커·발음 표시용으로 직접 부착한다
            # (기존 DP 근사 pron_segments 대신 — 실제 정렬 타이밍이라 더 정확).
            pd: dict[str, Any] = {}
            if pron_data is not None:
                pd = pron_data.get(i) or {}
                if pd.get("pronunciation"):
                    seg["pronunciation"] = pd["pronunciation"]
                if pd.get("translation"):
                    seg["translation"] = pd["translation"]
                if pd.get("pron_segments"):
                    seg["pron_segments"] = pd["pron_segments"]
            debug: dict[str, Any] = {}
            if vad_regions is not None:
                # 라인 구간 중 실제 발성 비율 + 클램프 여부 — 확장 디버그 스트립용
                dur = max(0.001, r.end_time - r.start_time)
                vocal = sum(
                    max(0.0, min(e, r.end_time) - max(s, r.start_time)) for s, e in vad_regions
                )
                debug["active_ratio"] = round(vocal / dur, 2)
                debug["clamped"] = i in clamped_lines
                # 보정된 라인은 보정 전 원본 타이밍 + 적용 규칙 라벨을 함께 내려준다.
                # 융합(fuse)처럼 라인 내부만 바꾸는 보정은 경계가 그대로라 고스트가 현재
                # 위치에 겹쳐 그려진다 — 정보를 버리지 않고 그대로 내려보내되, 겹치는
                # 고스트를 흐리게 그릴지는 클라이언트가 판단한다(확장 디버그 오버레이).
                fx = fixes.get(i)
                if fx:
                    debug["orig"] = [round(raw_spans[i][0], 2), round(raw_spans[i][1], 2)]
                    debug["fixes"] = fx
            # 「들은 것」(모델의 greedy 전사)과 심판 판정 근거 — 기존 debug 키와 같은 방식.
            # 실오디오 검증에서 후보별 점수를 못 보면 판정을 되짚을 수 없으므로 반드시 싣는다.
            heard = pd.get("heard") if pron_data is not None else heard_by_line.get(i)
            if heard:
                debug["heard"] = heard
            heard_spans = (
                pd.get("heard_spans") if pron_data is not None else heard_spans_by_line.get(i)
            )
            if heard_spans:
                debug["heard_spans"] = [[ch, round(t, 2)] for ch, t in heard_spans]
            if pd.get("referee"):
                debug["referee"] = pd["referee"]
            if debug:
                seg["debug"] = debug
            # attach_pron_variants는 여기서 바로 부르지 않는다 — E4(자체 한글 독음 생성,
            # 감사 2차)가 ``pronunciation``이 비어 있으면 즉시 채우는데, 병기 시트 줄(gloss)의
            # pronunciation은 이 루프가 다 끝난 뒤 ``_fold_gloss_into_segments``가 되붙인다.
            # 여기서 먼저 부르면 E4가 그 자리를 선점해 자체 생성값이 들어가고, 뒤이은 fold의
            # "이미 값이 있으면 안 덮는다" 가드에 걸려 사용자가 붙여넣은 진짜 발음이 영영
            # 안 붙는다. referee_tokens만 순서대로 챙겨 뒀다가 fold 이후에 일괄 호출한다.
            pron_referee_tokens.append(pd.get("tokens"))
            timestamps.append(seg)

        # 정렬 입력에서 뺀 병기 줄을 표시용으로 되붙인다 — 사용자가 붙여넣은 줄이 화면에서
        # 사라지면 안 된다. 독음 정렬/위키 line_meta가 이미 채운 값은 덮지 않는다.
        if gloss_folded:
            attached = _fold_gloss_into_segments(timestamps, gloss_folded)
            logger.info(
                f"Re-attached {attached} excluded gloss line(s) to their source segment "
                f"for display (alignment input untouched)"
            )

        # 세그 완성 + gloss 되붙이기까지 끝난 뒤: 표기별 발음을 얹는다(위 루프에서 미룬 이유
        # 참고). debug["referee"]는 이미 각 세그에 실려 있어 attach가 심판 개입 라인을
        # 알아본다 — 순서 의존은 그 필드뿐이라 여기로 옮겨도 무관하다.
        for seg, referee_tokens in zip(timestamps, pron_referee_tokens):
            attach_pron_variants(seg, referee_tokens=referee_tokens)

        # 앞뒤에 섞여 들어온 비가창 줄(크레딧·출처·URL) 제거 — 발성 근거와 텍스트 근거가
        # 함께 성립할 때만 버린다. 자세한 판정 근거는 _drop_nonvocal_nonlyric_edges 참고.
        if vad_regions is not None:
            for text in _drop_nonvocal_nonlyric_edges(timestamps):
                logger.info(f"Dropped non-vocal non-lyric edge line: {text!r}")

        # 가라오케용 음정(MIDI 노트) 주석 — 실패해도 싱크 생성 자체는 계속한다.
        # f0 전곡 추론은 위에서 정렬과 병렬로 이미 돌고 있다(f0_future) — 여기서 그 결과를
        # 받아 정렬 결과에 노트만 부착한다. 미가용/미설정은 위에서 이미 걸러 melody_extractor가
        # None이다(경고도 이미 1줄 기록). 멜로디 실패는 비치명 — 노트 없이 계속.
        report("멜로디 분석")
        f0_curve = None
        song_key = None
        if melody_extractor is not None:
            try:
                precomputed_f0 = f0_future.result() if f0_future is not None else None
                # vocal_regions는 넘기지 않는다 — extractor가 라인 스팬 합집합으로
                # 자체 마스킹한다 (VAD 마스크는 조용한 벌스 노트를 소실시킴)
                annotated = melody_extractor.annotate_timestamps(
                    audio, timestamps, vocals=vocals, precomputed_f0=precomputed_f0
                )
                # 디버그 오버레이용 RAW f0 곡선 (다운샘플, 옥타브 폴딩 전)
                f0_curve = melody_extractor.last_f0_curve
                # 곡 키 (K-S 추정) — 레인 표시용, 스냅 보정은 extractor 내부에서 완료
                song_key = melody_extractor.last_key
                logger.info(f"Melody notes annotated on {annotated} spans")
            except Exception:
                logger.exception("Melody extraction failed; continuing without notes")
            finally:
                if f0_executor is not None:
                    f0_executor.shutdown(wait=True)

        avg_confidence = None
        confidences = [t.get("confidence") for t in timestamps if t.get("confidence") is not None]
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)

        # 정렬이 실제로 성립한 줄 수 = conf가 실린 줄 수. 정렬 글자가 0개인 줄은 보간으로만
        # 채워져 conf가 없다(_quality_with_coverage 위 주석의 재현 경로 참고). 커버리지가
        # 과반 미만이면 quality_score를 확정 저신뢰로 덮어 확장 경고에 걸리게 한다.
        quality_score, coverage_meta = _quality_with_coverage(
            avg_confidence, len(confidences), len(timestamps)
        )
        if coverage_meta.get("failed"):
            logger.warning(
                f"Alignment coverage too low: only {coverage_meta['aligned_lines']}/"
                f"{coverage_meta['total_lines']} line(s) have measured char timing "
                f"(the rest are interpolated); reporting quality_score="
                f"{quality_score} instead of {coverage_meta['measured_conf']}"
            )

        # 결함 #5: DB로 흘러가는 language는 반드시 순수 언어여야 한다. 엔진의 _current_lang은
        # 내부 캐시 키라 force_mms 정렬이면 "{language}_mms"로 뭉쳐 있다(ctc_engine.py 654행) —
        # 그 값을 그대로 language 컬럼에 썼던 것이 결함의 원인이다. 대신 엔진이 따로 노출하는
        # 순수 언어(_current_language)와 변형(_current_engine_variant)을 각각 읽는다.
        detected_lang = language
        engine_variant = None
        if hasattr(engine, "_current_language"):
            detected_lang = engine._current_language
            engine_variant = getattr(engine, "_current_engine_variant", None)

        # 곡 단위 디버그 메타 — star가 흡수한 구간(가사 밖 가창)과 VAD 발성 구간,
        # 그리고 어떤 텍스트로 정렬했는지(원문 vs 독음) 클라 디버그 표시용.
        # 독음을 유지한 경우 debug star는 ko 정렬의 것이어야 한다 — star 가드가 교차검증용
        # ja 정렬을 돌리면 engine._last_star_spans가 ja star로 덮이므로 미리 포착해 둔
        # pron_star_spans(ko star)를 쓴다. 원문 정렬(폴백 포함)은 _last_star_spans가 맞고,
        # 스플라이스는 후반(교체 구간)을 지배하는 ja star(_last_star_spans)를 그대로 쓴다.
        final_star_source = pron_star_spans if alignment_text == "pronunciation" else getattr(
            engine, "_last_star_spans", []
        )
        star_spans = [list(s) for s in final_star_source]
        # 최종 결과를 실제로 측정한 어댑터. 누출 폴백/스플라이스로 ja를 채택하는 분기가
        # 여러 곳이라 분기마다 갱신하지 않고 최종 alignment_text로 판정한다 ("spliced"는
        # ko 라인 경계가 유지되는 혼합이라 ko 쪽으로 본다).
        final_adapter = ja_adapter if alignment_text == "original" else align_adapter
        quality_norm = _scale_free_quality(avg_confidence, final_adapter)
        debug_meta = {
            "star_spans": star_spans,
            "vad_regions": [list(v) for v in vad_regions] if vad_regions is not None else None,
            "alignment_text": alignment_text,
            # 음정 모델(RMVPE/FCPE) RAW f0 곡선 — 레인 디버그 오버레이용
            "f0_curve": f0_curve,
            # quality_score(=avg_confidence)는 어댑터 vocab 크기에 스케일 의존해 곡 간
            # 비교가 불가능하다(실측: 같은 곡 eng 0.1289 vs kor 0.0492, 잔차 동일).
            # 어느 어댑터로 측정한 값인지와 스케일 무관 지표를 함께 내려보내 비교 가능하게 한다.
            # quality_score 자체는 확장의 0.001 고정 임계 호환 때문에 원본 그대로 둔다.
            "quality_adapter": final_adapter,
            "quality_norm": None if quality_norm is None else round(quality_norm, 6),
            # 정렬이 실제로 성립한 줄 수/전체 — quality_score를 저신뢰로 덮었는지의 근거.
            # quality_norm은 여전히 **측정된 줄만의** 첨예도이므로, 이 값을 함께 봐야 해석된다.
            "align_coverage": coverage_meta,
        }
        # 자막 앵커: 어떤 트랙이 몇 줄 매칭됐고 어느 구간을 금지했고 그 제약이 채택됐는지.
        # 사후 감사가 안 되면 이 기능은 신뢰할 수 없다 — 앵커를 안 쓴 경우에도 «왜 안 썼는지»
        # (skipped)를 남긴다. 앵커 스위치가 꺼져 있으면 키 자체가 없다(기존 debug와 동일).
        if anchor_plan is not None:
            debug_meta["caption_anchors"] = {**anchor_plan.debug, "decision": anchor_decision}
        # 자막 스캐폴드 판정 — 발동(줄 수·출처별 개수·게이트 값)이든 스킵(사유)이든 그대로.
        # 스위치가 꺼져 있으면 키 자체가 없다 (caption_anchors와 동일 규약).
        if scaffold_meta is not None:
            debug_meta["caption_scaffold"] = scaffold_meta

        return {
            "timestamps": timestamps,
            "language": detected_lang,
            "engine_variant": engine_variant,
            "quality_score": quality_score,
            "debug": debug_meta,
            "alignment_text": alignment_text,
            # 가라오케 레인의 마디 단위 고정 창·비트 격자용 — 실패해도 None으로 계속
            "tempo": _estimate_tempo(audio),
            # 곡 키 (멜로디 분석 부산물) — 레인 좌상단 표시용
            "key": song_key,
        }
    finally:
        # 성공 경로는 멜로디 블록의 finally가 이미 shutdown(wait=True)했다 — 정렬 예외로
        # 여기로 빠졌을 때만 남은 f0 스레드를 정리한다(멱등, 실행 중 future는 기다리지 않음)
        if f0_executor is not None:
            f0_executor.shutdown(wait=False)
        # 앵커 수거 지점을 지났으면 이미 None이다 — 그 전에 예외로 빠진 경우만 정리한다
        if anchor_executor is not None:
            anchor_executor.shutdown(wait=False)
        audio_path_obj.unlink(missing_ok=True)
