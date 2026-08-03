import json
import logging
import os
import random
import re
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import requests
from dotenv import load_dotenv

from everyric2.config.settings import TranslationSettings, get_settings
from everyric2.inference.prompt import LyricLine
from everyric2.text.ja_reading import kana_reading, reading_source
from everyric2.text.kana_hangul import has_kana
from everyric2.text.ko_reading import hangul_to_kana, hangul_to_romaja
from everyric2.text.pron_style import romaji_line, wiki_pronunciation

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class TranslationLine:
    original: str
    translation: str
    pronunciation: str | None = None
    # NIM 응답이 잘려(max_tokens 소진) 복구/재분할 후에도 이 라인만 살려내지 못한 경우 True.
    # 전체 500 대신 원문만 담아 반환하되, 어떤 라인이 실패했는지 결과에 남긴다.
    failed: bool = False


@dataclass
class TranslationResult:
    lines: list[TranslationLine]
    source_lang: str
    target_lang: str
    engine: str
    tone: str
    # 원문 언어가 대상 언어와 같아 번역을 건너뛴 경우 True (translation은 전부 빈 문자열).
    # 클라이언트가 '번역 실패'와 '번역할 것이 없음'을 구분할 수 있게 결과에 남긴다.
    translation_skipped: bool = False


class TranslationBudget:
    """한 번역 요청(OpenAICompatibleTranslator.translate() 한 번) 안에서 NIM 왕복(실제 HTTP
    요청) 수·누적 소요시간을 추적하는 공유 예산.

    translate() 진입 시 한 번 만들어 재귀 호출 트리 전체(_translate_lines → _run_batches →
    _translate_batch → _retry_low_quality → _request_completion → _post_completion)에 그대로
    넘겨진다. 미스매치 복구(재귀 depth)·저품질 재요청·429 백오프는 각자 정당한 메커니즘이라
    손대지 않고, 이 예산만 그 위에 공용 브레이크로 얹는다 — exhausted()가 True면 호출자는 새
    NIM 왕복을 시작하지 않고 그 시점까지의 결과로 예외 없이 반환한다(부분 번역이 무번역보다
    낫다는 이 파일의 기존 원칙과 동일).

    스레드 세이프 — 배치가 ThreadPoolExecutor로 동시에 뛰므로 카운터에 락을 건다. 한도가
    0 이하면 그 축은 무제한으로 취급한다(설정 필드의 "0 disables" 관례를 따른다).
    """

    def __init__(self, max_round_trips: int, max_duration_sec: float):
        self._max_round_trips = max_round_trips
        self._max_duration_sec = max_duration_sec
        self._start = time.monotonic()
        self._round_trips = 0
        self._lock = threading.Lock()
        self._warned = False

    def record_round_trip(self) -> None:
        """실제 NIM HTTP 요청(429 재시도 포함) 하나가 나갈 때마다 호출한다."""
        with self._lock:
            self._round_trips += 1

    def exhausted(self) -> bool:
        with self._lock:
            trips = self._round_trips
        if self._max_round_trips > 0 and trips >= self._max_round_trips:
            return True
        return self._max_duration_sec > 0 and (
            time.monotonic() - self._start
        ) >= self._max_duration_sec

    def warn_once(self, log_prefix: str) -> None:
        """예산 소진을 로그에 정확히 한 번만 남긴다(재귀/병렬 경로에서 반복 호출돼도 무해)."""
        with self._lock:
            if self._warned:
                return
            self._warned = True
            trips = self._round_trips
        logger.warning(
            "%sTranslation budget exhausted (%d round trips, %.1fs elapsed) — "
            "returning the best-effort result so far",
            log_prefix,
            trips,
            time.monotonic() - self._start,
        )


_HANGUL_RE = re.compile(r"[가-힣]")
_ASCII_LETTER_RE = re.compile(r"[A-Za-z]")
_OTHER_LETTER_RE = re.compile(r"[^\x00-\x7F가-힣\s\W]")
_JA_CHAR_RE = re.compile(r"[぀-ヿ㐀-鿿]")
# 원문 대조용 정규화 — 공백·문장부호를 지운다 (모델이 원문을 되돌려줄 때의 사소한 정리 흡수)
_ALIGN_STRIP_RE = re.compile(r"[\s\W_]+", re.UNICODE)

# pykakasi가 문맥 없이 오독하거나 훈독이 갈리는 대표 항목 — 단일 확정값 대신 후보를
# 함께 제시해 모델이 문맥으로 고르게 한다. 실측(pykakasi 2.3):
#   今更止められない → いまさら"やめ"られない (정답 とめ), 涙を止める → なみだを"やめる",
#   風が止む → かぜが"とむ" (정답 やむ), 君にだけ → "くん"にだけ (노래에선 きみ)
# 문맥으로 판별 가능한 것만 담는다 — 行く(いく/ゆく)나 明日(あした/あす)처럼 텍스트만으로
# 정할 수 없는 것은 오히려 사전값보다 나빠질 수 있어 넣지 않는다.
# 표는 남겨 두지만 적용은 pykakasi 폴백 경로로 좁혔다(_build_prompt 참조).
# 일본어 곡의 한글 독음은 이제 LLM에 묻지 않으므로(_use_deterministic_pron) 이 표가 실제로
# 쓰이는 곳은 **발음을 LLM에 묻는 경로**뿐이다 — 형태소 분석기를 못 쓰는 환경(폴백)과
# 비일본어(중국어 등) 원문. 그 경로에서는 여전히 유효하다.
_AMBIGUOUS_READINGS: tuple[tuple[str, str], ...] = (
    ("君", "きみ (you) / くん (name suffix)"),
    ("止め", "とめ (stop something) / やめ (quit, give up)"),
    ("止む", "やむ (rain/wind ceasing)"),
    ("止ん", "やん (止んで = やんで)"),
    ("開く", "ひらく (~を開く) / あく (~が開く)"),
    ("空く", "あく (becomes vacant) / すく (お腹が空く)"),
)


def _kana_readings(text: str) -> list[str] | None:
    """일본어 원문 각 라인의 히라가나 읽기 — 발음 프롬프트의 참조.

    읽기 엔진은 everyric2.text.ja_reading이 단독 소유한다(형태소 분석 우선, pykakasi
    폴백). 스레드 안전(번역 배치가 병렬로 부른다)도 그 모듈이 락으로 책임진다.

    일본어 문자가 없거나 읽기 실패 시 None (힌트 없이 진행).
    라인 수·순서는 입력 텍스트의 줄과 1:1.
    """
    if not _JA_CHAR_RE.search(text):
        return None
    try:
        readings = [
            kana_reading(stripped) if (stripped := ln.strip()) else ""
            for ln in text.split("\n")
        ]
        return readings if any(readings) else None
    except Exception:
        logger.exception("kana reading hints failed; prompting without them")
        return None


def _reading_candidates(line: str) -> str:
    """라인에 다의어 훈독이 있으면 후보 목록 주석을 만든다 (없으면 빈 문자열).

    사전 참조값을 '정답'으로 박아두면 모델이 오독을 그대로 베낀다 — 갈리는 항목만
    후보를 함께 보여 문맥으로 고르게 한다. 적용 여부는 호출부가 읽기 엔진을 보고
    정한다(_build_prompt) — 여기는 표 조회만 한다.
    """
    hits = [f"{key}={cands}" for key, cands in _AMBIGUOUS_READINGS if key in line]
    return f"  [CANDIDATES: {'; '.join(hits)}]" if hits else ""


# 구조화(JSON) 응답 잘림(NIM max_tokens 소진) 복구 파라미터.
# - THRESHOLD 초과면 처음부터 배치로 나눠 요청(잘림 예방).
# - SIZE: 한 배치 라인 수. 8192 예산 안에서 30줄 발음 JSON은 안전(실측).
# - MAX_SPLIT_DEPTH: 잘림 복구 시 재귀 재분할 깊이 상한(요청 폭주 방지).
_PRON_BATCH_THRESHOLD = 60
_PRON_BATCH_SIZE = 30
# 발음 없는 번역 JSON은 라인당 출력이 절반 이하 — 배치를 크게 잡아 곡 맥락을 덜 끊는다
_TEXT_BATCH_THRESHOLD = 120
_TEXT_BATCH_SIZE = 60
_MAX_SPLIT_DEPTH = 4

# 429(rate limit) 백오프 상한 — Retry-After가 비상식적으로 크게 와도 한 번의 대기가
# 게이트웨이 타임아웃(600s) 여유를 갉아먹지 않게 자른다.
_RATE_LIMIT_MAX_WAIT_SEC = 30.0
# 동시에 던진 배치들이 같은 순간에 429를 받으면 백오프도 같은 순간에 풀려 그대로 다시
# 몰린다 — 대기에 이 비율만큼의 무작위 지연을 섞어 재시도를 흩뜨린다.
_RATE_LIMIT_JITTER = 0.25

# "저품질 배치" 판정 — 응답이 문법적으로 완전한데도 중간 구간의 번역/발음만 빈 값으로
# 오는 현상(실측: 48줄 중 19~34번이 통째로 빈 값, 잘림 로그는 한 건도 없음)을 잡는다.
_LOW_QUALITY_RATIO = 0.2
_LOW_QUALITY_MIN_LINES = 2
_LOW_QUALITY_MAX_RETRIES = 1
# 원문이 이보다 짧으면(구두점·감탄사 등) 번역이 비어도 품질 문제로 보지 않는다
_MEANINGFUL_ORIGINAL_CHARS = 2

# 번역 스킵 게이트(원문 언어 == 대상 언어)의 보수 임계. 오판하면 번역이 통째로
# 사라지므로 발음 게이트(_detect_lang_heuristic)보다 훨씬 엄격하게 잡는다.
_SKIP_MIN_LETTERS = 20
_SKIP_DOMINANT_RATIO = 0.85
_SKIP_FOREIGN_TOLERANCE = 0.05

TONE_PROMPTS = {
    "literal": "Translate literally, preserving the original meaning as closely as possible.",
    "natural": "Translate naturally so it sounds fluent to native speakers.",
    "poetic": "Translate poetically, maintaining rhythm, beauty, and artistic expression.",
    "casual": "Translate in casual, conversational language.",
    "formal": "Translate in formal, polite language.",
}


class BaseTranslator(ABC):
    # 로그 상관용 라벨(보통 video_id). 서버 로그에 곡이 안 남아 사후 추적이 불가능했다 —
    # API가 요청마다 주입한다. 인스턴스는 요청당 새로 만들어지므로 스레드 간 누수는 없다.
    log_label: str | None = None

    def __init__(self, settings: TranslationSettings | None = None):
        self.settings = settings or get_settings().translation

    def _log_prefix(self) -> str:
        return f"[{self.log_label}] " if self.log_label else ""

    @abstractmethod
    def translate(
        self,
        lyrics: list[LyricLine] | str,
        source_lang: str = "auto",
        target_lang: str | None = None,
        context: str | None = None,
    ) -> TranslationResult:
        pass

    def _build_prompt(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        include_pronunciation: bool,
        context: str | None = None,
    ) -> str:
        lang_names = {"ko": "Korean", "en": "English", "ja": "Japanese", "zh": "Chinese"}
        target = lang_names.get(target_lang, target_lang)
        tone_instruction = TONE_PROMPTS.get(self.settings.tone, TONE_PROMPTS["natural"])
        context_block = f"\nSong: {context}" if context else ""

        # 가사 맥락 지시 — 줄별 고립 직역(기계번역 톤)을 막고 곡 전체를 하나의 화자로 잇는다
        register_hint = (
            " Korean song lyrics use the plain intimate register (반말/해라체, e.g. ~해, ~야,"
            " ~잖아) — never formal endings like ~습니다/~어요 unless the original is"
            " explicitly formal."
            if target_lang == "ko"
            else ""
        )
        lyrics_guidance = (
            "These lines are the lyrics of ONE song, in order. Read the whole song first,"
            " then translate so the lines flow as a coherent song: keep one consistent"
            " speaker, emotional register and formality throughout, resolve omitted"
            " subjects/pronouns from surrounding lines, and prefer natural lyrical phrasing"
            f" over word-for-word rendering. Never translate a line in isolation.{register_hint}"
        )

        if include_pronunciation:
            # 일본어 원문이면 가나 읽기를 참조로 프롬프트에 심는다. 기계 읽기는 여전히
            # '정답'이 아니라 '참조 + 오독은 문맥으로 교정'으로 지시한다.
            reading_block = ""
            readings = _kana_readings(text)
            if readings:
                text_lines = text.split("\n")
                # 다의어 후보 주석은 사전 읽기(pykakasi) 폴백일 때만 붙인다. 형태소 분석은
                # 문맥으로 이미 とめ/やむ를 맞히는데 그 위에 "とめ/やめ 중 골라라"를 얹으면
                # 맞은 읽기를 모델이 다시 흔든다. 표(_AMBIGUOUS_READINGS)는 폴백 경로에서
                # 여전히 유효하므로 남겨 두고 적용 조건만 좁혔다.
                with_candidates = reading_source() == "pykakasi"
                numbered = "\n".join(
                    f"{i + 1}. {r}"
                    + (
                        _reading_candidates(text_lines[i] if i < len(text_lines) else "")
                        if with_candidates
                        else ""
                    )
                    for i, r in enumerate(readings)
                )
                reading_block = (
                    "\nREFERENCE READINGS (machine dictionary reading of each line, in order."
                    " The dictionary can misread context-dependent kanji — e.g. it may say"
                    " くん for 君 where the song sings きみ, or やめられない for 止められない"
                    " where the line means 'can't stop' (とめられない). Use these as a base and"
                    " correct such misreadings from the song's context. Where a line ends with"
                    " [CANDIDATES: ...], the dictionary reading is unreliable for that word —"
                    " pick the candidate that fits the meaning of the sentence):\n"
                    + numbered
                    + "\n"
                )
            if target_lang == "ko":
                # 한글 독음은 서버가 가나에서 결정적으로 변환한다(kana_hangul) — LLM에겐
                # 문맥 판단이 필요한 '한자→가나'만 맡긴다. LLM의 가나→한글 기계 전사는
                # 촉음/ん 소실 실수가 잦았다 (ずっと→즈토, じぶんが→지부가 실측).
                pron_rule = (
                    "2. The full kana reading (ひらがな) of the ORIGINAL line — how the line"
                    " is actually sung. Convert every kanji to kana. FOLLOW the REFERENCE"
                    " READINGS below; deviate where the dictionary misread a context-dependent"
                    " kanji (e.g. 君 sung きみ not くん; 今更止められない is いまさらとめられない"
                    " 'can't stop it', not やめられない 'can't quit') or where the line lists"
                    " CANDIDATES — those are kun-readings the dictionary cannot pick without"
                    " context, so choose by meaning. Write particles as pronounced"
                    " (は→わ, へ→え). Insert a space between sung phrases — a typical line"
                    " has 2-4 phrases (きみにだけ みえている) — but keep particles attached"
                    " to their word (きみにだけ, never きみ に だけ). Kana ONLY in this field."
                )
                pron_example = (
                    '[{"original": "時計の針が", "translation": "시곗바늘이",'
                    ' "pronunciation": "とけいの はりが"}]'
                )
                pron_note = (
                    "- pronunciation must be the kana reading of the ORIGINAL line"
                    " (hiragana, spaced by phrase) — never romanization, never a"
                    " translation, never Hangul"
                )
            else:
                pron_rule = "2. Romanized pronunciation of the ORIGINAL text (not the translation)"
                pron_example = '[{"original": "原文", "translation": "translation", "pronunciation": "genbun"}]'
                pron_note = "- pronunciation should be romanization of the ORIGINAL lyrics"
            return f"""Translate these song lyrics to {target}.
{tone_instruction}
{lyrics_guidance}{context_block}

For each line, provide:
1. The translation
{pron_rule}

Output as JSON array:
{pron_example}

IMPORTANT:
- Output exactly one object for EVERY input line, in the same order — including lines that
  are a title, a repeat, an ad-lib, or already in {target}. Never skip, merge or add lines.
- Copy "original" verbatim from the input line so the lines can be matched up
- Output ONLY the JSON array, no explanations
{pron_note}
{reading_block}
LYRICS:
{text}"""
        else:
            # 평문 줄바꿈 응답은 모델이 한 줄만 빠뜨려도 이후 전 라인이 한 칸씩 밀린다
            # (실측: 0번 줄이 제목이라 가사로 안 보고 건너뛴 곡 31줄 전체가 밀림).
            # 원문을 함께 돌려받아 라인 정합성을 검증할 수 있게 JSON으로 통일한다.
            return f"""Translate these song lyrics to {target}.
{tone_instruction}
{lyrics_guidance}{context_block}

Output as JSON array:
[{{"original": "原文", "translation": "translation"}}]

IMPORTANT:
- Output exactly one object for EVERY input line, in the same order — including lines that
  are a title, a repeat, an ad-lib, or already in {target}. Never skip, merge or add lines.
- Copy "original" verbatim from the input line so the lines can be matched up
- Output ONLY the JSON array, no explanations

LYRICS:
{text}"""

    # ── 응답 파싱: 원문 대조로 라인을 맞춘다(위치 매칭 금지) ──────────────────────

    def _extract_json_items(self, response: str) -> list[dict]:
        """(잘렸을 수도 있는) JSON 배열에서 완전한 객체까지만 순서대로 뽑는다.

        예외를 던지지 않는다 — 살릴 게 없으면 []를 돌려줘 호출자가 재요청/폴백하게 한다.
        """
        text = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        if text.startswith("```"):
            match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
            if match:
                text = match.group(1).strip()

        start = text.find("[")
        if start == -1:
            return []
        return [
            item for item in self._decode_json_objects(text, start + 1) if isinstance(item, dict)
        ]

    @staticmethod
    def _decode_json_objects(text: str, pos: int) -> list:
        """text[pos:]의 JSON 배열 원소를 raw_decode로 하나씩 읽어 완전한 값만 모은다.
        마지막 원소가 잘려 있으면 그 앞까지만 반환(truncation-safe)."""
        decoder = json.JSONDecoder()
        objs: list = []
        n = len(text)
        while pos < n:
            while pos < n and text[pos] in " \t\r\n,":
                pos += 1
            if pos >= n or text[pos] == "]":
                break
            try:
                obj, end = decoder.raw_decode(text, pos)
            except json.JSONDecodeError:
                break  # 마지막 원소가 잘림 — 여기서 중단
            objs.append(obj)
            pos = end
        return objs

    @staticmethod
    def _align_key(text: str) -> str:
        """정렬 대조용 정규화 — 공백·구두점 제거 + 소문자화.
        모델이 원문을 되돌려줄 때 흔히 하는 사소한 정리(공백 정돈, 문장부호 생략)를 흡수한다."""
        stripped = _ALIGN_STRIP_RE.sub("", text or "").casefold()
        if stripped:
            return stripped
        # 기호만으로 된 라인(…, ♪)은 정규화하면 빈 문자열이 되어 대조가 불가능해진다 —
        # 공백만 정리한 원문으로 대조한다 (빈 키는 '대조 불가'로만 남긴다)
        return "".join((text or "").split()).casefold()

    def _align_items(self, items: list[dict], lines: list[str]) -> list[TranslationLine | None]:
        """응답 항목을 원문 텍스트로 입력 라인에 맞춘다. 길이는 항상 len(lines).

        모델이 한 줄을 빠뜨리면 위치 매칭은 그 뒤 전 라인의 번역을 한 칸씩 민다
        (실증: 0번 줄이 '실리카겔 - APEX'라는 제목 텍스트여서 모델이 가사로 취급하지
        않고 누락 → 31줄 전체가 1칸씩 밀려 저장). 그래서 돌려받은 original을 입력과
        순서대로 대조하고, 못 맞춘 자리는 None으로 남겨 호출자가 재요청하게 한다.
        """
        keys = [self._align_key(x) for x in lines]
        slots: list[TranslationLine | None] = [None] * len(lines)
        cursor = 0
        matched = 0
        echoed = 0
        for item in items:
            raw_original = str(item.get("original") or "")
            key = self._align_key(raw_original)
            if key:
                echoed += 1
            idx = next((k for k in range(cursor, len(lines)) if keys[k] == key), None) if key else None
            if idx is None:
                continue
            slots[idx] = TranslationLine(
                original=lines[idx],
                translation=str(item.get("translation") or ""),
                pronunciation=item.get("pronunciation"),
            )
            matched += 1
            cursor = idx + 1

        if matched == 0 and items and (echoed == 0 or len(items) == len(lines)):
            # 대조할 근거가 없다 — 모델이 original을 아예 안 돌려줬거나(echoed==0) 통째로
            # 고쳐 썼는데 줄 수는 정확히 같은 경우. 순서대로 채우되 입력 길이를 넘지 않는다.
            logger.warning(
                "%sTranslation response carried no matchable originals (%d items, %d lines)"
                " — falling back to positional matching",
                self._log_prefix(),
                len(items),
                len(lines),
            )
            for i, item in enumerate(items[: len(lines)]):
                slots[i] = TranslationLine(
                    original=lines[i],
                    translation=str(item.get("translation") or ""),
                    pronunciation=item.get("pronunciation"),
                )
        return slots

    def _plain_text_slots(
        self, response: str, lines: list[str]
    ) -> list[TranslationLine | None]:
        """모델이 JSON 지시를 무시하고 평문 줄바꿈으로 답했을 때의 폴백.

        줄 수가 입력과 정확히 맞을 때만 위치로 짝짓는다. 예전 구현은 빈 줄을 버리고
        무조건 위치로 붙여서, 모델이 한 줄에 빈 응답을 주면 그 줄부터 배열이 하나 줄고
        이후 전 라인이 한 칸씩 밀렸다.
        """
        text = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        for prefix in ("TRANSLATION:", "Translation:", "번역:"):
            if text.startswith(prefix):
                text = text[len(prefix) :].strip()
        if not text:
            # 빈 응답을 '빈 줄 하나'로 읽어 성공 처리하면 안 된다 (잘림/필터링과 동일 취급)
            return [None] * len(lines)
        if text.startswith(("[", "{")):
            # 모델은 JSON을 시도했고 그 파싱이 실패해서 여기까지 온 것이다(잘림 등) —
            # 깨진 JSON 조각을 평문 번역으로 받아들이면 안 된다. 실측: 1줄 배치에서
            # 잘린 '[{"original":"ダメ","transl'이 줄 수가 맞아 번역으로 저장됐다.
            return [None] * len(lines)

        raw = [ln.strip() for ln in text.split("\n")]
        nonempty = [ln for ln in raw if ln]
        # 서론 한 줄("Here is the translation:")이 붙은 경우까지만 관용한다
        without_preamble = nonempty[1:] if nonempty and nonempty[0].endswith(":") else []
        for candidate in (raw, nonempty, without_preamble):
            if candidate and len(candidate) == len(lines):
                return [
                    TranslationLine(original=orig, translation=trans, pronunciation=None)
                    for orig, trans in zip(lines, candidate)
                ]

        logger.warning(
            "%sPlain-text response has %d lines for %d inputs — refusing positional match",
            self._log_prefix(),
            len(nonempty),
            len(lines),
        )
        return [None] * len(lines)

    def _parse_aligned(
        self, response: str, original_lines: list[str], include_pronunciation: bool
    ) -> list[TranslationLine]:
        """재요청 기전이 없는 엔진(Gemini)용 단발 파싱 — 맞춘 라인만 채우고 나머지는
        원문만 담아 failed로 남긴다. 밀린 번역을 저장하느니 빈 라인이 낫다."""
        slots = self._align_items(self._extract_json_items(response), original_lines)
        if all(slot is None for slot in slots) and not include_pronunciation:
            slots = self._plain_text_slots(response, original_lines)
        if all(slot is None for slot in slots):
            raise ValueError(f"Failed to parse translation response: {response[:200]}")
        return [
            slot if slot is not None else self._failed_line(orig)
            for slot, orig in zip(slots, original_lines)
        ]

    @staticmethod
    def _failed_line(original: str) -> TranslationLine:
        """복구 불가 라인 — 원문만 담고 translation/pronunciation은 비운 채 failed 표시."""
        return TranslationLine(
            original=original, translation="", pronunciation=None, failed=True
        )

    @staticmethod
    def _is_blank_output(
        line: TranslationLine | None, original: str, include_pronunciation: bool
    ) -> bool:
        """원문은 유의미한데 번역(·발음)이 빈 라인인가 — '저품질 배치' 판정용.

        실증: 48줄 중 19~34번의 번역·발음이 전부 빈 값인데 응답 JSON은 문법적으로 완전해
        잘림 복구 경로가 발동하지 않았다(서버 로그에 잘림 경고가 한 건도 없음).
        """
        if line is None or line.failed:
            return False
        if len(original.strip()) < _MEANINGFUL_ORIGINAL_CHARS:
            return False
        if not (line.translation or "").strip():
            return True
        return include_pronunciation and not (line.pronunciation or "").strip()

    # ── 언어 게이트 ─────────────────────────────────────────────────────────────

    def _detect_lang_heuristic(self, text: str) -> str:
        """한글/ASCII 비율 기반 언어 추정. source_lang="auto"일 때 발음 생략 게이트에만 쓰이는
        거친 휴리스틱이며 실제 번역 언어 감지에는 관여하지 않는다."""
        hangul = len(_HANGUL_RE.findall(text))
        ascii_letters = len(_ASCII_LETTER_RE.findall(text))
        other_letters = len(_OTHER_LETTER_RE.findall(text))
        total = hangul + ascii_letters + other_letters
        if total == 0:
            return "en"
        if hangul / total >= 0.3:
            return "ko"
        if ascii_letters / total >= 0.5:
            return "en"
        return "other"

    def _should_skip_pronunciation(
        self, text: str, source_lang: str, target_lang: str = "ko"
    ) -> bool:
        """발음표기가 대상 언어에 무의미하면 생략한다 — 매트릭스 대각선(곡 언어==대상
        언어) 우선, 그 다음 target=ko 전용의 기존 규칙.

        **매트릭스 대각선**: 곡 언어와 번역 대상 언어가 같으면(ko곡×ko유저, en곡×en유저,
        ja곡×ja유저) 무조건 생략한다. ja곡×ja유저는 가나가 있어도 생략된다 — target이
        ja면 "가나 독음"은 원문 그 자체라 무의미하다(아래 가나 예외는 target=ko 전용).

        **target=ko 전용의 기존 규칙**: 원문이 영어/한국어면 로마자/한글 발음표기가
        무의미하므로 생략한다. 번역 자체는 그대로 수행되고 pronunciation 필드만 비운다.
        target이 ko가 아니면 이 규칙은 적용하지 않는다 — 비ko 타깃(예: ko곡×en유저)은
        로마자 발음이 필요할 수 있어 원문 언어만으로 생략을 결정할 수 없다(ko_reading 등
        미래 경로를 이 게이트가 먼저 죽이지 않게 한다).

        **가나가 있으면(target=ko일 때만) 생략하지 않는다.** 이 규칙의 전제는 "원문이
        영어/한국어"인데, `_detect_lang_heuristic`은 글자 수로 판정하므로 **영어가 많이
        섞인 일본어 곡**을 영어로 오판한다. 실측: 라틴 7줄/일본어 3줄로 된 요청에서 판정이
        "en"이 되어 10줄 전부 발음이 None으로 나갔다(번역은 정상, `failed=False`). 하필
        라틴이 많은 곡이 정렬이 가장 나쁜 곡이라(라인 conf가 라틴 없는 줄의 1/10) 발음이 그
        줄에서 가장 필요한데, 그 곡만 발음을 못 받고 있었다. 가나가 한 글자라도 있으면
        원문은 영어가 아니고 한글 독음이 의미를 가지므로 전제가 성립하지 않는다.
        """
        target = self._norm_lang(target_lang)
        song_lang = self._norm_lang(source_lang)
        if song_lang in ("", "auto"):
            song_lang = self._detect_lang_heuristic(text)
            if song_lang == "other" and has_kana(text):
                song_lang = "ja"
        if target and song_lang == target:
            return True
        if target != "ko":
            return False
        if has_kana(text):
            return False
        lang = source_lang
        if lang == "auto":
            lang = self._detect_lang_heuristic(text)
        return lang in ("en", "ko")

    def _is_ja_source_for_deterministic_pron(self, text: str, source_lang: str) -> bool:
        """이 곡을 결정론 ja 발음 경로의 원문으로 볼 것인가.

        판정은 **곡 전체** 텍스트로 한다 — 라인 단위로 보면 한자만 있는 줄이 중국어로
        오판된다(UniDic은 중국어에 틀려서 중국어는 LLM 경로로 남긴다). 가나가 한 글자라도
        있거나 source_lang이 명시적으로 ja면 ja로 본다."""
        return has_kana(text) or self._norm_lang(source_lang) == "ja"

    def _deterministic_pron_fn(
        self, text: str, source_lang: str, target_lang: str
    ) -> Callable[[str], str | None] | None:
        """이 (원문, 대상 언어) 조합에 결정론 발음 렌더러가 있으면 그 함수를, 없으면 None.

        `_use_deterministic_pron`(게이트)과 `_apply_deterministic_pron`(실제 렌더)이
        이 한 곳만 보고 판단하게 해서 두 곳의 매트릭스가 갈리는 사고를 막는다.

        매트릭스 (곡 원문 × 사용자 대상 언어):
          ja × ko → `wiki_pronunciation` (기존, 무변경 — 한글 독음)
          ja × en → `romaji_line(text)[0]` (표시 문자열 — 비ko 타깃의 계약은 로마자)
          ja × ja → 대각선(`_should_skip_pronunciation`)이 먼저 걸러 발음 자체를 안 묻는다
                     — 이 함수까지 오지 않는다
          ko × ja → `ko_reading.hangul_to_kana`
          ko × en → `ko_reading.hangul_to_romaja`
          그 외(zh 등) → None(기존 LLM 자유서술 경로 유지)

        실측(보카로 위키 사람 발음, ja×ko 경로): 결정론 82.4% vs LLM 82.2%로 정확도는
        동등한데, LLM은 같은 줄을 실행마다 다르게 읽고(「縋って」를 3회 중 2회 오독) 조사
        は를 표층 그대로 "하"로 쓰는 실수를 반복한다. 발음을 프롬프트에서 빼면 출력이
        절반 이하로 줄어 번역 배치(_TEXT_BATCH_*)를 크게 잡을 수 있다는 이득도 있다 —
        결정론 경로를 타는 모든 셀에 이 이득이 동일하게 적용된다(llm_pron 계산이
        `_use_deterministic_pron` 결과 하나로 갈리므로 별도 배선이 필요 없다).

        ja 원문 경로만 형태소 분석기 가용성(`reading_source()=='fugashi'`)에 의존한다 —
        폴백(pykakasi) 독음은 신뢰도가 낮다(縋って→ついって). `ko_reading`은 자모 분해
        규칙 기반이라 그런 의존이 없다(무조건 쓸 수 있다).
        """
        target = self._norm_lang(target_lang)

        if self._is_ja_source_for_deterministic_pron(text, source_lang):
            if reading_source() != "fugashi":
                return None
            if target == "ko":
                return lambda t: wiki_pronunciation(t) or None
            if target == "en":
                def _romaji(t: str) -> str | None:
                    rendered = romaji_line(t)
                    return rendered[0] if rendered else None

                return _romaji
            return None

        if self._norm_lang(source_lang) == "ko":
            if target == "ja":
                return lambda t: hangul_to_kana(t) or None
            if target == "en":
                return lambda t: hangul_to_romaja(t) or None
            return None

        return None

    def _use_deterministic_pron(self, text: str, source_lang: str, target_lang: str) -> bool:
        """이 곡의 발음표기를 LLM 대신 결정론 엔진(`_deterministic_pron_fn`)으로 만드는가."""
        return self._deterministic_pron_fn(text, source_lang, target_lang) is not None

    def _apply_deterministic_pron(
        self, lines: list[TranslationLine], text: str, source_lang: str, target_lang: str
    ) -> None:
        """LLM이 돌려준 발음을 버리고 결정론 값으로 덮는다 (원문에서만 만든다).

        번역이 실패한 라인(failed)도 채운다 — 발음은 더 이상 LLM 응답에 의존하지 않으므로
        번역이 비었다고 독음까지 비울 이유가 없다. 렌더러는 `_deterministic_pron_fn`이
        게이트(`_use_deterministic_pron`)와 정확히 같은 판정으로 고른다 — 두 곳의 매트릭스가
        갈릴 수 없다.
        """
        renderer = self._deterministic_pron_fn(text, source_lang, target_lang)
        if renderer is None:  # 방어적 — 호출부가 deterministic_pron=True일 때만 부른다
            return
        for line in lines:
            line.pronunciation = renderer(line.original)

    def _detect_lang_confident(self, text: str) -> str | None:
        """번역 스킵 게이트 전용 언어 판정 — 확신할 때만 "ko"/"en", 아니면 None.

        _detect_lang_heuristic은 오판해도 발음만 빠지므로 임계가 느슨하다. 번역 스킵은
        오판하면 번역이 통째로 사라지므로, 글자 수가 충분하고 다른 문자 체계가 거의 없고
        한쪽이 압도적일 때만 판정한다.
        """
        hangul = len(_HANGUL_RE.findall(text))
        ascii_letters = len(_ASCII_LETTER_RE.findall(text))
        other_letters = len(_OTHER_LETTER_RE.findall(text))
        total = hangul + ascii_letters + other_letters
        if total < _SKIP_MIN_LETTERS:
            return None
        if other_letters / total > _SKIP_FOREIGN_TOLERANCE:
            return None
        if hangul / total >= _SKIP_DOMINANT_RATIO:
            return "ko"
        if ascii_letters / total >= _SKIP_DOMINANT_RATIO:
            return "en"
        return None

    @staticmethod
    def _norm_lang(code: str | None) -> str:
        """"ko-KR"/"KO" 같은 표기를 기본 코드로 정규화."""
        return (code or "").strip().lower().replace("_", "-").split("-")[0]

    def _should_skip_translation(self, text: str, source_lang: str, target_lang: str) -> bool:
        """원문 언어가 번역 대상 언어와 같으면 번역을 건너뛴다 (LLM 호출 자체를 안 한다).

        같은 언어로 '재번역'하면 원문과 다른 문장이 나온다 — 실증: 한국어 곡의
        "저기요 제가요 가슴이 떨려서"가 "저기, 나야. 가슴이 너무 떨려"로 바뀌어 저장됐다.
        자동 감지가 확실하지 않으면 스킵하지 않는다: 번역이 통째로 사라지는 쪽이 더 나쁘다.
        """
        target = self._norm_lang(target_lang)
        if not target:
            return False
        source = self._norm_lang(source_lang)
        if source and source != "auto":
            # 호출자가 언어를 명시했다면 그 판단을 따른다
            return source == target
        return self._detect_lang_confident(text) == target

    def _skipped_translation_result(
        self,
        original_lines: list[str],
        source_lang: str,
        target_lang: str,
        engine: str,
    ) -> TranslationResult:
        """번역 스킵 결과 — 번역 필드는 비운다. 원문을 그대로 넣으면 클라이언트가 원문과
        번역을 똑같이 두 줄로 표시한다."""
        logger.info(
            "%sSource language already %s — skipping translation for %d lines",
            self._log_prefix(),
            target_lang,
            len(original_lines),
        )
        return TranslationResult(
            lines=[
                TranslationLine(original=orig, translation="", pronunciation=None)
                for orig in original_lines
            ],
            source_lang=source_lang,
            target_lang=target_lang,
            engine=engine,
            tone=self.settings.tone,
            translation_skipped=True,
        )


class GeminiTranslator(BaseTranslator):
    def __init__(self, settings: TranslationSettings | None = None):
        super().__init__(settings)
        self.api_key = self.settings.api_key or os.getenv("GEMINI_API_KEY")
        self.model = self.settings.model
        self.api_url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        )

    def translate(
        self,
        lyrics: list[LyricLine] | str,
        source_lang: str = "auto",
        target_lang: str | None = None,
        context: str | None = None,
    ) -> TranslationResult:
        target_lang = target_lang or self.settings.target_language

        if isinstance(lyrics, list):
            text = "\n".join(line.text for line in lyrics)
            original_lines = [line.text for line in lyrics]
        else:
            text = lyrics
            original_lines = [line.strip() for line in text.split("\n") if line.strip()]

        if not text.strip():
            return TranslationResult([], source_lang, target_lang, "gemini", self.settings.tone)

        if not self.api_key:
            return self._fallback_result(original_lines, source_lang, target_lang)

        include_pron = self.settings.include_pronunciation and not self._should_skip_pronunciation(
            text, source_lang, target_lang
        )
        # 일본어 곡의 독음은 서버가 결정론적으로 만든다 — 모델에는 번역만 요청한다
        deterministic_pron = include_pron and self._use_deterministic_pron(
            text, source_lang, target_lang
        )
        llm_pron = include_pron and not deterministic_pron
        if not include_pron and self._should_skip_translation(text, source_lang, target_lang):
            return self._skipped_translation_result(
                original_lines, source_lang, target_lang, "gemini"
            )
        prompt = self._build_prompt(text, source_lang, target_lang, llm_pron, context)

        try:
            response = requests.post(
                self.api_url,
                # 키는 URL 쿼리가 아니라 헤더로 — URL은 예외 메시지·로그에 그대로 찍힌다
                headers={"x-goog-api-key": self.api_key},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": self.settings.temperature,
                        "maxOutputTokens": 8192,
                    },
                },
                timeout=self.settings.timeout,
            )

            if not response.ok:
                raise RuntimeError(f"API error: {response.status_code} - {response.text[:200]}")

            result = response.json()
            content = result["candidates"][0]["content"]["parts"][0]["text"]

            lines = self._parse_aligned(content, original_lines, llm_pron)
            if deterministic_pron:
                self._apply_deterministic_pron(lines, text, source_lang, target_lang)

            return TranslationResult(lines, source_lang, target_lang, "gemini", self.settings.tone)

        except requests.exceptions.ConnectionError:
            return self._fallback_result(original_lines, source_lang, target_lang)
        except Exception as e:
            raise RuntimeError(f"Translation failed: {e}") from e

    def _fallback_result(
        self, original_lines: list[str], source_lang: str, target_lang: str
    ) -> TranslationResult:
        # API 키가 없거나 연결이 안 되면 무료 웹 번역(deep-translator)으로 폴백.
        # 플레이스홀더 텍스트를 번역인 척 반환하면 클라이언트 UI에 그대로 노출되므로 금지 —
        # 여기서도 실패하면 예외를 올려 API가 5xx로 응답하게 한다(확장은 '번역 실패' 표시).
        from deep_translator import GoogleTranslator

        target = {"zh": "zh-CN"}.get(target_lang, target_lang)
        translator = GoogleTranslator(source="auto", target=target)

        translated = translator.translate("\n".join(original_lines)) or ""
        parts = [p.strip() for p in translated.split("\n")]
        if len(parts) != len(original_lines):
            # 웹 번역이 줄 수를 보존하지 못한 경우 — 줄 단위로 재시도 (느리지만 정확)
            parts = [(t or "").strip() for t in translator.translate_batch(original_lines)]

        lines = [
            TranslationLine(original=orig, translation=trans, pronunciation=None)
            for orig, trans in zip(original_lines, parts)
        ]
        return TranslationResult(lines, source_lang, target_lang, "google-web", self.settings.tone)


class OpenAICompatibleTranslator(BaseTranslator):
    def __init__(self, settings: TranslationSettings | None = None):
        super().__init__(settings)
        # 결과에 찍는 실제 백엔드 이름 — 자동 전환(gemini 설정→NIM) 시 settings.engine과 다르다
        self.engine_name = self.settings.engine
        self.api_key = self.settings.api_key or os.getenv("OPENAI_API_KEY") or "local-gen-ai"
        self.model = self.settings.model

        if self.settings.engine == "openai":
            self.api_url = "https://api.openai.com/v1/chat/completions"
        else:
            self.api_url = self.settings.api_url or "http://localhost:11434/v1/chat/completions"

    def translate(
        self,
        lyrics: list[LyricLine] | str,
        source_lang: str = "auto",
        target_lang: str | None = None,
        context: str | None = None,
    ) -> TranslationResult:
        target_lang = target_lang or self.settings.target_language

        if isinstance(lyrics, list):
            text = "\n".join(line.text for line in lyrics)
            original_lines = [line.text for line in lyrics]
        else:
            text = lyrics
            original_lines = [line.strip() for line in text.split("\n") if line.strip()]

        if not text.strip():
            return TranslationResult(
                [], source_lang, target_lang, self.engine_name, self.settings.tone
            )

        include_pron = self.settings.include_pronunciation and not self._should_skip_pronunciation(
            text, source_lang, target_lang
        )
        # 일본어 곡의 독음은 서버가 결정론적으로 만든다 — 모델에는 번역만 요청하므로
        # 라인당 출력이 절반 이하로 줄고 _TEXT_BATCH_*의 큰 배치를 쓸 수 있다
        deterministic_pron = include_pron and self._use_deterministic_pron(
            text, source_lang, target_lang
        )
        llm_pron = include_pron and not deterministic_pron
        skip_translation = self._should_skip_translation(text, source_lang, target_lang)
        if skip_translation and not include_pron:
            return self._skipped_translation_result(
                original_lines, source_lang, target_lang, self.engine_name
            )

        try:
            # 요청당 총예산 — 재귀적 미스매치 복구(depth 4까지)·저품질 배치 재요청이 특정
            # 영상에서 무제한으로 중첩되며 p95 25.4s·최대 57.5s를 만든 실측(외부 감사 #9)에 대한
            # 공용 브레이크. 개별 재시도 메커니즘(미스매치 복구·저품질 재요청·429 백오프)은
            # 그대로 두고, 이 요청 하나(translate() 한 번 호출)의 재귀 트리 전체에 공유 예산을
            # 실어 보낸다 — TranslationBudget 클래스 문서 참고.
            budget = TranslationBudget(
                self.settings.budget_max_round_trips, self.settings.budget_max_duration_sec
            )
            # 구조화 응답은 라인 수가 많으면 max_tokens에서 잘려 파싱이 통째로 실패한다(500).
            # 프롬프트/요청/정렬/복구를 _translate_lines에 위임해 잘림·누락·저품질을 감지하고,
            # 최악의 경우에도 원문만 담아 부분 성공으로 마감한다.
            lines = self._translate_lines(
                original_lines,
                source_lang,
                target_lang,
                context,
                include_pron=llm_pron,
                budget=budget,
            )
            if deterministic_pron:
                self._apply_deterministic_pron(lines, text, source_lang, target_lang)
            if skip_translation:
                # 원문 == 대상 언어인데 발음은 필요한 경우(ja→ja 등) — 발음만 남기고
                # 무의미한 '재번역' 결과는 버린다
                for line in lines:
                    line.translation = ""

            return TranslationResult(
                lines,
                source_lang,
                target_lang,
                self.engine_name,
                self.settings.tone,
                translation_skipped=skip_translation,
            )

        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(f"Connection failed to {self.api_url}: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Translation failed: {e}") from e

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    @staticmethod
    def _sleep(seconds: float) -> None:
        """백오프 대기 — 테스트가 실제로 기다리지 않도록 주입 지점을 분리해 둔다."""
        time.sleep(seconds)

    @staticmethod
    def _retry_delay(response, delay: float) -> float:
        """다음 429 재시도까지의 대기 — Retry-After를 존중하되 상한과 지터를 씌운다."""
        headers = getattr(response, "headers", None) or {}
        try:
            hinted = float(headers.get("Retry-After", ""))
        except (TypeError, ValueError):
            hinted = 0.0
        wait = min(max(delay, hinted), _RATE_LIMIT_MAX_WAIT_SEC)
        return wait + random.uniform(0.0, wait * _RATE_LIMIT_JITTER)

    def _post_completion(self, payload: dict, budget: "TranslationBudget | None" = None):
        """chat/completions POST + 429(rate limit) 지수 백오프 재시도.

        배치를 동시에 던지므로 분당 한도에 걸려 429가 올 수 있다(실측으로 확인된 건 동시
        8건까지 429 없음 — 지속 RPM 한도는 미확인이다). 429는 '조금 뒤엔 되는' 실패라
        상한 안에서 기다렸다 다시 던지고, 상한을 넘으면 429 응답을 그대로 돌려줘 기존
        실패 경로(API error 예외 → 배치 부분 실패 처리)를 타게 한다. 대기하는 동안 다른
        배치는 자기 스레드에서 계속 진행한다.

        budget이 있으면 실제로 나가는 요청마다(429 재시도 포함) record_round_trip으로 센다 —
        이 안의 재시도 정책 자체는 건드리지 않는다(요청당 총예산은 새 배치/재시도를
        *시작*할지를 _translate_batch 진입부에서 가른다).
        """
        retries = max(0, self.settings.rate_limit_retries)
        delay = max(0.0, self.settings.rate_limit_backoff_sec)
        for attempt in range(retries + 1):
            response = requests.post(
                self.api_url,
                json=payload,
                headers=self._headers(),
                timeout=self.settings.timeout,
            )
            if budget is not None:
                budget.record_round_trip()
            if response.status_code != 429 or attempt >= retries:
                return response
            wait = self._retry_delay(response, delay)
            logger.warning(
                "%sRate limited (429) — retrying in %.1fs (attempt %d/%d)",
                self._log_prefix(),
                wait,
                attempt + 1,
                retries,
            )
            self._sleep(wait)
            delay *= 2
        return response

    def _request_completion(
        self,
        prompt: str,
        *,
        allow_empty: bool = False,
        budget: "TranslationBudget | None" = None,
    ) -> tuple[str, str | None]:
        """단발 chat/completions 호출. (content, finish_reason)를 돌려준다.

        빈 응답(콘텐츠 필터/reasoning이 max_tokens 소진)은 1회 재시도한다. 그래도 비면
        allow_empty=False면 예외를, True면 ("", finish_reason)을 돌려줘 호출자가 잘림과
        동일하게 복구·재분할하도록 한다.
        """
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
            "stream": False,
        }
        payload.update(self._payload_extras())

        content = ""
        finish_reason: str | None = None
        for attempt in range(2):
            response = self._post_completion(payload, budget=budget)

            if not response.ok:
                raise RuntimeError(f"API error: {response.status_code} - {response.text[:200]}")

            result = response.json()
            choice = result["choices"][0]
            content = choice["message"].get("content") or ""
            finish_reason = choice.get("finish_reason")
            if content.strip():
                break
            logger.warning(
                "Empty completion content (attempt %d/2, finish_reason=%s); %s",
                attempt + 1,
                finish_reason,
                "retrying" if attempt == 0 else "giving up",
            )

        if not content.strip() and not allow_empty:
            raise RuntimeError(
                "Empty completion content (model may have spent max_tokens on reasoning)"
            )
        return content, finish_reason

    def _translate_lines(
        self,
        original_lines: list[str],
        source_lang: str,
        target_lang: str,
        context: str | None,
        *,
        include_pron: bool,
        budget: "TranslationBudget | None" = None,
    ) -> list[TranslationLine]:
        """긴 입력은 처음부터 배치로 나눠(잘림 예방) 각 배치를 복구 로직으로 처리한 뒤
        순서대로 이어붙인다. 발음 배치는 라인당 출력이 커서 더 잘게 나눈다.

        배치끼리는 의존이 없어(각자 자기 구간만 번역한다) 동시에 요청한다 — 순차 루프에서는
        번역 시간이 배치 수에 선형 비례했다(실측: 30줄 1배치 8.3s, 실사용 평균 20.9s·최대
        118.5s). 결합은 완료 순서가 아니라 배치 인덱스 순으로만 한다.
        """
        threshold, size = (
            (_PRON_BATCH_THRESHOLD, _PRON_BATCH_SIZE)
            if include_pron
            else (_TEXT_BATCH_THRESHOLD, _TEXT_BATCH_SIZE)
        )
        if len(original_lines) <= threshold:
            return self._translate_batch(
                original_lines,
                source_lang,
                target_lang,
                context,
                include_pron=include_pron,
                depth=0,
                budget=budget,
            )

        batches = [
            original_lines[start : start + size]
            for start in range(0, len(original_lines), size)
        ]
        return [
            line
            for batch in self._run_batches(
                batches,
                source_lang,
                target_lang,
                context,
                include_pron=include_pron,
                budget=budget,
            )
            for line in batch
        ]

    def _batch_concurrency(self) -> int:
        """동시에 던질 배치 수 (설정값, 최소 1)."""
        return max(1, self.settings.batch_concurrency)

    def _run_batches(
        self,
        batches: list[list[str]],
        source_lang: str,
        target_lang: str,
        context: str | None,
        *,
        include_pron: bool,
        budget: "TranslationBudget | None" = None,
    ) -> list[list[TranslationLine]]:
        """배치들을 설정된 동시성으로 실행하고 **입력 인덱스 순서**의 결과를 돌려준다.

        결합이 완료 순서를 타면 가사가 통째로 뒤섞이므로 결과는 인덱스로만 되꽂는다
        (submit 순서대로 future.result()를 기다리면 순서가 자동으로 보장된다).

        한 배치가 예외로 죽어도 나머지 결과는 살린다 — 그 배치의 라인만 원문+failed로
        마감해 배치 내부의 부분 실패 처리(_failed_line)와 같은 모양이 되게 한다. 다만
        **모든** 배치가 실패하면 번역이 통째로 빈 채 '성공'으로 저장되는 쪽이 더 나쁘므로
        첫 예외를 그대로 올려 기존 실패 경로(500)를 탄다. 배치가 하나뿐인 짧은 곡은
        이 규칙에 따라 예외가 그대로 전파돼 기존 동작과 동일하다.
        """
        results: list[list[TranslationLine]] = [[] for _ in batches]
        failures: list[Exception] = []

        def run(index: int) -> list[TranslationLine]:
            return self._translate_batch(
                batches[index],
                source_lang,
                target_lang,
                context,
                include_pron=include_pron,
                depth=0,
                budget=budget,
            )

        def collect(index: int, produce) -> None:
            # produce()는 항상 호출 스레드(메인)에서 실행/대기한다 — results·failures를
            # 건드리는 것도 메인 스레드뿐이라 별도 락이 필요 없다.
            try:
                results[index] = produce()
            except Exception as exc:
                logger.exception(
                    "%sTranslation batch %d/%d failed (%d lines) — keeping the other batches",
                    self._log_prefix(),
                    index + 1,
                    len(batches),
                    len(batches[index]),
                )
                failures.append(exc)
                results[index] = [self._failed_line(line) for line in batches[index]]

        workers = min(self._batch_concurrency(), len(batches))
        if workers <= 1:
            # 동시성 1 — 스레드를 만들지 않고 기존 순차 루프 그대로 돈다
            for i in range(len(batches)):
                collect(i, lambda i=i: run(i))
        else:
            with ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="translate-batch"
            ) as pool:
                futures = [pool.submit(run, i) for i in range(len(batches))]
                for i, future in enumerate(futures):
                    collect(i, future.result)

        if failures and len(failures) == len(batches):
            raise failures[0]
        return results

    def _translate_batch(
        self,
        lines: list[str],
        source_lang: str,
        target_lang: str,
        context: str | None,
        *,
        include_pron: bool,
        depth: int,
        quality_retries: int = 0,
        budget: "TranslationBudget | None" = None,
    ) -> list[TranslationLine]:
        """한 배치를 요청하고 응답을 원문 대조로 입력 라인에 맞춘다. 반환 길이 == len(lines).

        못 맞춘 라인(잘림·누락·빈 응답)은 ① 그 라인들만 재요청하고, 진전이 전혀 없으면
        ② 절반으로 나눠 재귀, ③ 깊이 한도를 넘거나 단일 라인도 실패하면 원문만 담고
        failed=True로 마감(전체 500 방지). 응답이 완전한데도 번역·발음이 빈 라인이 많으면
        ④ '저품질 배치'로 보고 그 라인들만 한 번 더 요청한다.

        budget이 이미 소진됐으면(요청당 총예산 — TranslationBudget 참고) 새 NIM 왕복을 만들지
        않고 이 배치 전체를 원문만 담은 failed 라인으로 즉시 마감한다. 재귀(미스매치 복구·절반
        분할·저품질 재요청)는 전부 이 함수를 다시 부르므로, 이 진입부 한 곳의 체크가 재귀
        트리 전체에서 예산을 강제한다.
        """
        if not lines:
            return []
        if budget is not None and budget.exhausted():
            budget.warn_once(self._log_prefix())
            return [self._failed_line(line) for line in lines]

        text = "\n".join(lines)
        prompt = self._build_prompt(text, source_lang, target_lang, include_pron, context)
        content, finish_reason = self._request_completion(prompt, allow_empty=True, budget=budget)

        slots = self._align_items(self._extract_json_items(content), lines)
        if all(slot is None for slot in slots) and not include_pron:
            # 모델이 JSON 지시를 무시하고 평문으로 답한 경우 — 줄 수가 맞을 때만 수용
            slots = self._plain_text_slots(content, lines)

        missing = [i for i, slot in enumerate(slots) if slot is None]
        if missing:
            logger.warning(
                "%sTranslation response incomplete (finish_reason=%s, content_len=%d, "
                "matched %d/%d lines, depth=%d) — recovering",
                self._log_prefix(),
                finish_reason,
                len(content),
                len(lines) - len(missing),
                len(lines),
                depth,
            )
            if depth >= _MAX_SPLIT_DEPTH:
                for i in missing:
                    slots[i] = self._failed_line(lines[i])
            elif len(missing) < len(lines):
                # 일부는 확보 — 못 맞춘 라인만 다시 요청해 제자리에 채운다
                retried = self._translate_batch(
                    [lines[i] for i in missing],
                    source_lang,
                    target_lang,
                    context,
                    include_pron=include_pron,
                    depth=depth + 1,
                    budget=budget,
                )
                for i, line in zip(missing, retried):
                    slots[i] = line
            elif len(lines) > 1:
                # 한 라인도 못 맞췄다 — 절반으로 쪼개 재귀
                mid = len(lines) // 2
                return self._translate_batch(
                    lines[:mid], source_lang, target_lang, context,
                    include_pron=include_pron, depth=depth + 1, budget=budget,
                ) + self._translate_batch(
                    lines[mid:], source_lang, target_lang, context,
                    include_pron=include_pron, depth=depth + 1, budget=budget,
                )
            else:
                slots[0] = self._failed_line(lines[0])

        # 복구를 거치면 빈 칸은 없지만, 길이 보장은 호출자 계약이라 방어적으로 채운다
        resolved = [
            slot if slot is not None else self._failed_line(lines[i])
            for i, slot in enumerate(slots)
        ]
        return self._retry_low_quality(
            resolved, lines, source_lang, target_lang, context,
            include_pron=include_pron, depth=depth, quality_retries=quality_retries,
            budget=budget,
        )

    def _retry_low_quality(
        self,
        resolved: list[TranslationLine],
        lines: list[str],
        source_lang: str,
        target_lang: str,
        context: str | None,
        *,
        include_pron: bool,
        depth: int,
        quality_retries: int,
        budget: "TranslationBudget | None" = None,
    ) -> list[TranslationLine]:
        """응답이 절단되지 않았는데도 중간 구간의 번역·발음만 비어 온 경우의 재요청.

        실증: 48줄 중 19~34번이 통째로 빈 값이었는데 잘림 로그는 한 건도 없었다 — 모델이
        문법적으로 완전한 JSON을 주면서 내용만 뭉갠 품질 문제라 잘림 복구가 발동하지 않았다.
        무한 재시도를 막기 위해 배치당 _LOW_QUALITY_MAX_RETRIES회로 제한한다.
        """
        blank = [
            i
            for i, line in enumerate(resolved)
            if self._is_blank_output(line, lines[i], include_pron)
        ]
        if (
            not blank
            or quality_retries >= _LOW_QUALITY_MAX_RETRIES
            or len(blank) < _LOW_QUALITY_MIN_LINES
            or len(blank) / len(lines) < _LOW_QUALITY_RATIO
        ):
            return resolved
        if budget is not None and budget.exhausted():
            # 예산 소진 — 이 재요청은 시작하지 않는다. resolved를 그대로 돌려준다(빈 값이지만
            # failed=False인 라인을 _translate_batch의 소진 마감으로 failed=True로 격하시키지
            # 않기 위해 여기서 먼저 걸러 재귀 호출 자체를 생략한다).
            budget.warn_once(self._log_prefix())
            return resolved

        logger.warning(
            "%sLow-quality translation batch: %d/%d lines came back empty (indices %s, "
            "depth=%d) — re-requesting those lines",
            self._log_prefix(),
            len(blank),
            len(lines),
            blank,
            depth,
        )
        retried = self._translate_batch(
            [lines[i] for i in blank],
            source_lang,
            target_lang,
            context,
            include_pron=include_pron,
            depth=depth,
            quality_retries=quality_retries + 1,
            budget=budget,
        )
        for i, line in zip(blank, retried):
            # 재요청도 비었으면 1차 결과를 유지한다 (failed 표시로 덮어쓰지 않는다)
            if not self._is_blank_output(line, lines[i], include_pron):
                resolved[i] = line
        return resolved

    def _payload_extras(self) -> dict:
        """엔진별 추가 페이로드 훅 — 기본은 없음."""
        return {}


class NvidiaTranslator(OpenAICompatibleTranslator):
    """NVIDIA NIM (OpenAI 호환 /v1/chat/completions) 백엔드.

    키 해석 순서: settings.api_key -> env NVIDIA_API_KEY -> 루트 nvapi.txt 파일.
    모델은 gemini 기본값(settings.model)과 섞이지 않도록 settings.nvidia_model을 쓴다.
    """

    NIM_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
    _KEY_FILE = Path(__file__).resolve().parents[2] / "nvapi.txt"

    def __init__(self, settings: TranslationSettings | None = None):
        # OpenAICompatibleTranslator.__init__을 건너뛰고 BaseTranslator.__init__만 호출해
        # OPENAI_API_KEY/로컬 기본 URL 등 다른 엔진 전용 로직이 섞이지 않게 한다.
        BaseTranslator.__init__(self, settings)
        # settings.engine이 "gemini"여도(키 부재 자동 전환) 결과에는 실제 백엔드를 찍는다
        self.engine_name = "nvidia"
        self.api_key = (
            self.settings.api_key or os.getenv("NVIDIA_API_KEY") or self._read_key_file()
        )
        self.model = self.settings.nvidia_model
        self.api_url = self.settings.api_url or self.NIM_API_URL

    def _read_key_file(self) -> str | None:
        try:
            return self._KEY_FILE.read_text(encoding="utf-8").strip() or None
        except OSError:
            return None

    def _payload_extras(self) -> dict:
        model = self.model.lower()
        # qwen3 계열은 reasoning 모델 — 사고 모드를 끄지 않으면 max_tokens를 사고에
        # 소진해 content가 비거나(빈 응답) 타임아웃이 난다. NIM qwen 챗 템플릿 스위치.
        if "qwen" in model:
            return {"chat_template_kwargs": {"thinking": False}}
        # gpt-oss도 reasoning 모델인데 thinking off 스위치가 없다 — effort를 최저로.
        # 기본 effort로는 30줄 가사에서 사고가 예산을 소진해 빈 응답/잘린 JSON이 났다.
        if "gpt-oss" in model:
            return {"reasoning_effort": "low"}
        return {}


class TranslatorFactory:
    @staticmethod
    def get_translator(settings: TranslationSettings | None = None) -> BaseTranslator:
        settings = settings or get_settings().translation

        if settings.engine == "gemini":
            # gemini 키가 없으면 웹 번역 폴백(번역만 가능, 발음표기 불가·기계번역 톤)으로
            # 조용히 격하된다 — NVIDIA 키(env 또는 루트 nvapi.txt)가 있으면 NIM으로 자동
            # 전환한다. env 없이 uvicorn만 띄운 서버에서 발음이 통째로 빠지는 사고 방지.
            if not (settings.api_key or os.getenv("GEMINI_API_KEY")):
                nvidia = NvidiaTranslator(settings)
                if nvidia.api_key:
                    logger.info(
                        "No Gemini API key; auto-switching translation engine to NVIDIA NIM"
                    )
                    return nvidia
            return GeminiTranslator(settings)
        elif settings.engine == "nvidia":
            return NvidiaTranslator(settings)
        elif settings.engine in ("openai", "local"):
            return OpenAICompatibleTranslator(settings)
        else:
            raise ValueError(f"Unknown translation engine: {settings.engine}")


class LyricsTranslator:
    def __init__(
        self,
        api_key: str | None = None,
        settings: TranslationSettings | None = None,
        log_label: str | None = None,
    ):
        if settings is None:
            settings = get_settings().translation
        if api_key:
            settings.api_key = api_key
        self._translator = TranslatorFactory.get_translator(settings)
        # 어떤 곡의 번역인지 서버 로그에 남긴다 (보통 video_id) — 없으면 라벨 없이 진행
        self._translator.log_label = log_label
        self.settings = settings

    def translate(
        self,
        lyrics: list[LyricLine] | str,
        source_lang: str = "auto",
        target_lang: str = "ko",
        context: str | None = None,
    ) -> str:
        result = self._translator.translate(lyrics, source_lang, target_lang, context)
        return "\n".join(line.translation for line in result.lines)

    def translate_with_pronunciation(
        self,
        lyrics: list[LyricLine] | str,
        source_lang: str = "auto",
        target_lang: str = "ko",
        context: str | None = None,
    ) -> TranslationResult:
        old_setting = self.settings.include_pronunciation
        self.settings.include_pronunciation = True
        try:
            result = self._translator.translate(lyrics, source_lang, target_lang, context)
        finally:
            self.settings.include_pronunciation = old_setting
        if target_lang == "ko":
            # LLM은 가나 독음까지만 책임진다 — 가나→한글은 결정적 변환으로 마감
            # (촉음=ㅅ받침, ん=ㄴ받침, 장음=모음 반복). 한글로 온 구형 응답은 그대로 둔다.
            from everyric2.text.kana_hangul import finalize_pronunciation

            for line in result.lines:
                line.pronunciation = finalize_pronunciation(line.pronunciation)
        return result
