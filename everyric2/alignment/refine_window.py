"""2패스 정렬 리파이너 — 앵커가 잡은 라인 창 안에서 경량 CTC로 음절을 다시 잡는다.

동기·전체 설계는 ``scripts/bench_adapters/two_pass.py``(2,408줄)에서 실측으로 정해졌다.
요지: UST 준정답 채점의 두 축(라인 시작 vs 음절 온셋)이 서로 다른 모델을 가리켰다 —
무거운 앵커(OWSM-CTC 등)는 라인 경계에서 붕괴가 없는 유일한 후보였고, 경량 단일 언어
모델은 음절 온셋에서 더 정확하지만(84%대) 어려운 곡에서 무너졌다(32~49%). 그래서
**라인 경계는 앵커가 정하고, 각 라인의 창 안에서만 경량 모델이 음절을 다시 잡는다.**
전역 DP를 창으로 쪼개면 경량 모델이 곡 전체에서 미끄러질 여지 자체가 사라지므로, 붕괴는
앵커가 막고 해상도는 경량 모델이 채우는 구조가 된다.

설계 불변식 넷(벤치와 동일):

* **라인 경계는 이 모듈이 절대 못 건드린다.** ``RefinedLine.start``/``end``는 앵커 값
  그대로다.
* **하한이 앵커 단독으로 고정된다.** 창이 토큰 수보다 짧거나, vocab에 없는 문자뿐이거나,
  DP가 실패하면 그 라인은 세그가 비고 ``fallback_reason``에 이유가 남는다 — 호출부가
  앵커 세그로 폴백할 수 있게 신호만 준다(이 모듈은 앵커 세그 자체를 모른다).
* **emission은 곡당 한 번.** 라인마다 forward를 다시 돌리지 않고 리파이너의
  ``emission_for``가 낸 전곡 emission을 프레임 축에서 잘라 쓴다.
* **표기는 전부 낸다.** 라인 하나를 한 번 정렬하면 그 스팬 위에 표기별(hangul/kana/
  romaji/en) 텍스트를 동시에 얹는다 — 재생 시점 사용자 설정이 어느 표기를 볼지 정하므로
  서버가 하나만 골라 구우면 안 된다(``everyric2.text.align_target`` 참조). 표기마다
  세그 개수가 다를 수 있고, 각 표기는 독립적으로 라인 전체를 빈틈없이 덮는다(연속성은
  ``_extend_segments``가 세그 끝을 다음 세그 시작까지 미는 것으로 보장된다 —
  ``tests/test_refine_window.py``의 연속성 테스트 참조).

앵커 계약(2026-08-03 코디네이터 확정, ``everyric2.alignment.emission.EngineEmission`` /
``BaseAlignmentEngine.emission_for``)은 모듈 하단 ``SyllableRefiner`` 프로토콜과 아래
"앵커·리파이너 계약" 절 참조. OWSM류 앵커는 ``emission_for``가 ``None``을 돌려주므로
(별도 venv 서브프로세스라 emission 텐서가 프로세스 경계를 못 넘는다) 이 모듈은 항상
**인프로세스 리파이너**(예: OmniASR)의 emission만 쓴다 — 앵커의 emission은 요구하지 않는다.
"""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from everyric2.alignment.emission import EngineEmission
    from everyric2.inference.prompt import SyncResult

logger = logging.getLogger(__name__)


# ── 앵커·리파이너 계약 ──────────────────────────────────────────────────────
#
# 이 모듈은 두 가지를 밖에서 받는다.
#
# ① 앵커 라인 창 — ``list[SyncResult]``, 입력 가사 줄과 1:1·순서 보존. ``start_time``/
#    ``end_time``만 쓴다(``confidence``는 통과시킬 뿐 재계산하지 않는다). 다른 어떤
#    앵커 엔진이든 ``BaseAlignmentEngine.align()``과 같은 모양만 지키면 된다 — 이 모듈은
#    앵커가 OWSM인지 OmniASR인지 모른다.
#
# ② 리파이너 emission — ``EngineEmission``(``everyric2/alignment/emission.py``, 앵커 팀
#    확정본):
#
#        @dataclass
#        class EngineEmission:
#            emission: Any        # [1, T, V] log-softmax (torch.Tensor)
#            blank_id: int
#            frame_sec: float
#            audio_sec: float
#            chunks: int
#            vocab: dict[str, int] = field(default_factory=dict)
#            def frame_of(self, seconds: float) -> int: ...
#
#    ``refiner.emission_for(vocals_path) -> EngineEmission | None``. ``None``이면(리파이너가
#    emission을 못 내는 앵커류 엔진이면) 그 곡 전체가 폴백이다 — ``refine_lines``가 모든
#    라인에 ``fallback_reason="refiner_emission_unavailable"``을 달아 돌려준다.
#
#    ``vocab``이 있으므로 이 모듈은 ``prepare_line_targets`` 같은 별도 메서드를 리파이너에
#    요구하지 않는다 — 정렬 타깃 문자열을 토큰 id로 바꾸는 일(``_tokenize_target``)은
#    이 모듈이 vocab 딕셔너리 하나로 직접 한다. 리파이너 구현체에 요구하는 표면적이
#    ``emission_for`` 하나뿐이라는 뜻이다.
class SyllableRefiner(Protocol):
    """2패스 경량 모델이 만족해야 하는 계약. ``BaseAlignmentEngine``과 같은 모양이다."""

    def emission_for(self, audio_path: Path) -> "EngineEmission | None": ...


# ── 표시(발음) 세그 — 서버 wire 계약(everyric2-chrome/src/types.ts)과 손실 없이 대응 ──
#
#     interface PronSegment { text: string; start: number; end: number;
#                              resolved?: boolean; confidence?: number }
#     pron?: Record<string, string>              // 표기 → 발음 문자열 전체
#     pron_segs?: Record<string, PronSegment[]>   // 표기 → 음절 스팬
#
# ``word_end``는 이 계약에 없는 **추가** 필드다 — wire가 additive라(필드 추가는 구버전
# 확장이 무시) 안전하다. en 표기(라틴 낱말)만 쓴다: 원문 공백이 CTC 어휘에 없어 정렬 후
# 세그에서 사라지므로, 세그 자체가 아니라 이 플래그로 낱말 경계를 옮긴다(``_join_pron``
# 참조). ja 표기는 이미 원문 공백이 소유자에 실려 있어 이 플래그가 항상 False다.
@dataclass
class PronSegmentSpan:
    """``PronSegment``(wire) 한 칸. ``word_end``는 서버 내부용 — 프런트에 그대로 나간다."""

    text: str
    start: float
    end: float
    resolved: bool = True
    confidence: float | None = None
    word_end: bool = False


@dataclass
class RefinedLine:
    """2패스 출력 한 줄. ``start``/``end``/``confidence``는 앵커 값 그대로다."""

    start: float
    end: float
    confidence: float | None = None
    pron: dict[str, str] = field(default_factory=dict)
    pron_segs: dict[str, list[PronSegmentSpan]] = field(default_factory=dict)
    refined: bool = False
    fallback_reason: str | None = None


@dataclass(frozen=True)
class TwoPassRefineConfig:
    """실측으로 정해진 문턱값들 — ``scripts/bench_adapters/two_pass.py``의 값을 그대로
    옮긴다. 바꾸지 마라(코디네이터·운영자 지시).
    """

    # 창을 앵커 라인 경계에 딱 맞추면 2패스가 실제 발화의 앞뒤를 창 밖에 두고 잘라먹는다.
    # 앵커의 라인 시작 오차 자체가 ≤0.15s 기준으로 재던 값이라 그와 같은 자릿수 여유를 준다.
    window_pad_sec: float = 0.2
    # 세그를 늘일 수 있는 최대 길이(초). UST 노트 15,503개에서 99.5퍼센타일 1.111s.
    seg_hold_max_sec: float = 1.5
    # 게이트가 "내내 부르고 있다"고 판정했을 때 풀어 주는 한도(초).
    seg_hold_max_held_sec: float = 3.0
    # 라인 꼬리를 최대 얼마나 밀 것인가(초).
    line_tail_max_sec: float = 2.0
    # 꼬리를 멈출 「끊김」의 최소 길이(초).
    line_tail_quiet_sec: float = 0.12
    # 짧아도 깊은 골에서 꼬리를 멈출 것인가.
    line_tail_deep: bool = True
    # 세그 사이 늘이기의 발성 문턱. **채택값(2026-08-02 스윕)** — ja 7곡에서 무음 위 점등
    # 104.2초 → 70.5초(−32%)에 구간 IoU는 오히려 49.84→49.89로 상승, 음절 축은 전 문턱에서
    # 완전 불변. 0.30(간주 판정용)·0.20 모두 이 값보다 못했다. 절대 바꾸지 마라.
    seg_voiced_level: float = 0.12
    # 세그 끝을 다음 세그 시작까지 늘릴 것인가(노래방 표시 규약).
    extend_segments: bool = True
    # 세그 사이 늘이기도 "발성이 이어지는 동안"만 할 것인가(``seg_voiced_level`` 문턱).
    extend_voiced_only: bool = True
    # 라인 마지막 세그의 끝을 우세도가 이어지는 동안 다음 라인 전까지 민다.
    extend_line_tails: bool = True
    # 한 시각에 뭉친 세그를 발성 구간에 펴 줄 것인가.
    spread_piles: bool = True
    # 반복 훅에서 렌디션을 건너뛴 자리를 되돌릴 것인가.
    respace_repeats: bool = True


# ---------------------------------------------------------------------------
# 정렬 타깃 → 토큰
# ---------------------------------------------------------------------------


def _tokenize_target(
    text: str, vocab: dict[str, int]
) -> tuple[list[int], list[tuple[int, int] | None]]:
    """정렬 타깃 문자열 → (토큰 id 열, 문자별 토큰 범위).

    ``ranges[i]``는 ``text[i]``가 차지하는 토큰 id 열의 半개구간이다. vocab에 없는 문자는
    ``None``(그 문자는 정렬에서 제외 — 공백·구두점이 전형이고, 그 시간은 이웃 세그가
    ``_extend_segments``로 흡수한다). 라틴은 소문자로 한 번 더 조회한다(어댑터 vocab이
    대문자를 안 갖는 사례가 실측돼 있다 — ``ctc_engine._resolve_token_char`` 참조).
    """
    token_ids: list[int] = []
    ranges: list[tuple[int, int] | None] = []
    for char in text:
        token_id = vocab.get(char)
        if token_id is None:
            lowered = char.lower()
            if lowered != char:
                token_id = vocab.get(lowered)
        if token_id is None:
            ranges.append(None)
            continue
        start = len(token_ids)
        token_ids.append(token_id)
        ranges.append((start, start + 1))
    return token_ids, ranges


def _derive_units(source: str, language: str):
    """언어 힌트 → ``align_target`` 파생 함수 선택. ja/ko 이외는 en 경로로 떨어진다
    (다국어 라인이라도 일단 라틴 회로가 있어야 라틴 구간이라도 건진다)."""
    from everyric2.text.align_target import derive_en_display_units, derive_ja_display_units

    if language == "ja":
        return derive_ja_display_units(source)
    return derive_en_display_units(source)


# ---------------------------------------------------------------------------
# 창 전체 점수 (참고용 — 심판은 이 모듈 범위 밖이다)
# ---------------------------------------------------------------------------


def _window_score(frame_scores: Any) -> float | None:
    """창 전체의 프레임당 평균 로그확률. blank 프레임까지 포함해 정규화한다 — 프로드와
    같은 정의(``ctc_engine._score_tokens``)."""
    if frame_scores is None or len(frame_scores) == 0:
        return None
    return float(frame_scores.sum()) / len(frame_scores)


# ---------------------------------------------------------------------------
# 세그 구성 — 표기 하나(owners 배열 하나)에 대해
# ---------------------------------------------------------------------------


def _build_segments(
    owners: list[str],
    ranges: list[tuple[int, int] | None],
    spans: Any,
    offset: float,
    frame_sec: float,
    word_end: list[bool],
) -> list[dict[str, Any]]:
    """표기 하나의 owners 배열 → 세그 목록(딕셔너리, 파이프라인 내부 표현).

    표시 글자가 없는 타깃 문자(빈 소유자) **또는 공백뿐인 통과분**은 앞 세그의 끝을
    거기까지 늘린다 — 세그 하나가 음절 전체(또는 낱말 사이 여백)를 덮게 되고, 그 결과로
    표기 사이에 시간 축이 끊기지 않는다(연속성 요구사항). 공백이 vocab에 없어 애초에
    정렬되지 않은 자리(``ranges[i] is None``)는 여기서 아무것도 안 하지만, 그 구간은
    ``_extend_segments``가 "다음 세그 시작까지 늘린다"로 다시 메운다 — 두 메커니즘이
    합쳐 어떤 경로로 빠지든 빈틈이 안 남는다.
    """
    segs: list[dict[str, Any]] = []
    for index, token_range in enumerate(ranges):
        if token_range is None:
            continue
        owner = owners[index] if index < len(owners) else ""
        lo, hi = token_range
        flagged = bool(index < len(word_end) and word_end[index])
        if not owner.strip():
            if segs:
                segs[-1]["end"] = round(offset + float(spans[hi - 1].end) * frame_sec, 3)
                if flagged:
                    segs[-1]["word_end"] = True
            continue
        segs.append(
            {
                "t": owner,
                "start": round(offset + float(spans[lo].start) * frame_sec, 3),
                "end": round(offset + float(spans[hi - 1].end) * frame_sec, 3),
                "word_end": flagged,
            }
        )
    return segs


def _join_pron(segs: list[dict[str, Any]]) -> str:
    """세그 목록 → 표기 전체 문자열. ``word_end``가 걸린 세그 뒤에 공백 하나를 넣는다.

    리터럴 공백 문자(라틴 낱말 사이 통과분)에 기대지 않는다 — 그 문자는 리파이너 vocab에
    없으면 세그 자체가 안 생겨 정보가 사라진다. ``word_end`` 플래그가 유일한 근거다.
    """
    parts: list[str] = []
    for seg in segs:
        parts.append(seg["t"])
        if seg.get("word_end"):
            parts.append(" ")
    return "".join(parts).strip()


# ---------------------------------------------------------------------------
# 뭉침 펴기
# ---------------------------------------------------------------------------


def _spread_piled_segments(segs: list[dict[str, Any]], presence: Any, frame_sec: float) -> int:
    """한 시각에 뭉친 세그를 그 앞 발성 구간에 편다(``two_pass.py`` 이식, 로직 불변).

    CTC가 같은 가사를 여러 렌디션에 걸쳐 흘릴 때 앞쪽 글자들이 스팬 길이 0으로 무너져
    한 프레임에 쌓이는 사고를 편다. 시작 시각만 고친다 — 끝은 ``_extend_segments``가
    다시 잡는다.
    """
    import numpy as np

    fixed = 0
    index = 0
    while index < len(segs) - 1:
        if abs(segs[index + 1]["start"] - segs[index]["start"]) > 1e-6:
            index += 1
            continue
        stop = index + 1
        while stop < len(segs) and abs(segs[stop]["start"] - segs[index]["start"]) <= 1e-6:
            stop += 1
        origin = segs[index]["start"]
        limit = segs[stop]["start"] if stop < len(segs) else segs[-1].get("end", origin)
        span = limit - origin
        count = stop - index
        if span > 1e-3 and count > 1:
            lo = max(int(origin / frame_sec), 0)
            hi = min(int(limit / frame_sec) + 1, len(presence))
            weights = presence[lo:hi] if hi > lo else None
            if weights is not None and len(weights) >= count and float(weights.sum()) > 0:
                cumulative = np.cumsum(weights) / float(weights.sum())
                for offset in range(1, count):
                    position = int(np.searchsorted(cumulative, offset / count))
                    segs[index + offset]["start"] = round((lo + position) * frame_sec, 3)
            else:
                step = span / count
                for offset in range(1, count):
                    segs[index + offset]["start"] = round(origin + offset * step, 3)
            for offset in range(count):
                segs[index + offset]["end"] = max(
                    segs[index + offset]["end"], segs[index + offset]["start"]
                )
            fixed += count - 1
        index = stop
    return fixed


# ---------------------------------------------------------------------------
# 늘이기 게이트 — 우세도 × 발화로 "언제까지 켜 둘지" 정한다
# ---------------------------------------------------------------------------

_HOLD_DOMINANCE = 0.30
_HOLD_QUIET_SEC = 0.12
_HOLD_SPEAK_SEC = 0.10
_HOLD_DEEP_LEVEL = 0.25
_HOLD_DEEP_SEC = 0.04

_DOMINANCE_CACHE: dict[str, Any] = {}


def _dominance_curve(vocals_path: Path) -> tuple[Any, float] | None:
    """보컬 우세도 곡선. ``star_prior.vocal_presence_from_stems``를 그대로 부른다 —
    복제하면 신호가 갈린다. ``inst.wav``(반주 스템)가 옆에 없으면 None(게이트 없이 진행)."""
    key = str(vocals_path)
    if key in _DOMINANCE_CACHE:
        return _DOMINANCE_CACHE[key]
    made = None
    instrumental = vocals_path.with_name("inst.wav")
    if instrumental.is_file():
        try:
            import librosa

            from everyric2.alignment.star_prior import vocal_presence_from_stems

            vocals, _ = librosa.load(str(vocals_path), sr=16_000, mono=True)
            accomp, _ = librosa.load(str(instrumental), sr=16_000, mono=True)
            curve = vocal_presence_from_stems(vocals, accomp, 16_000, smooth_sec=0.2, hop_sec=0.01)
            if curve is not None:
                made = (curve[1], 0.01)
        except Exception:
            logger.warning("우세도 계산 실패 — 늘이기 게이트 없이 진행", exc_info=True)
    _DOMINANCE_CACHE[key] = made
    return made


class ExtendGate:
    """세그를 어디까지 늘일지 시간 길이가 아니라 그 시간의 상태로 정한다.

    우세도(``rms(보컬)/(rms(보컬)+rms(반주))``)와 발화(``1 - p_blank``) 두 축을 쓴다.
    ``two_pass.py``의 ``_ExtendGate``를 그대로 옮긴 것이라 문턱 상수도 동일하다.
    """

    def __init__(
        self,
        dominance: Any,
        dom_hop: float,
        presence: Any,
        frame_sec: float,
        speak_level: float,
        quiet_sec: float = 0.12,
        deep: bool = False,
    ) -> None:
        self.dominance = dominance
        self.dom_hop = dom_hop
        self.presence = presence
        self.frame_sec = frame_sec
        self.speak_level = speak_level
        self.quiet_sec = quiet_sec
        self.deep = deep

    def limit(self, t0: float, t1: float) -> float:
        """[t0, t1) 안에서 늘이기를 멈춰야 할 시각. 멈출 이유가 없으면 ``t1``."""
        stops = [t1]
        if self.dominance is not None and t1 > t0:
            need = max(1, int(_HOLD_QUIET_SEC / self.dom_hop))
            lo, hi = int(t0 / self.dom_hop), min(len(self.dominance), int(t1 / self.dom_hop))
            run = None
            for index in range(lo, hi):
                if self.dominance[index] < _HOLD_DOMINANCE:
                    if run is None:
                        run = index
                    elif index - run + 1 >= need:
                        stops.append(run * self.dom_hop)
                        break
                else:
                    run = None
        if self.presence is not None and t1 > t0:
            need = max(1, int(_HOLD_SPEAK_SEC / self.frame_sec))
            lo, hi = int(t0 / self.frame_sec), min(len(self.presence), int(t1 / self.frame_sec))
            run = None
            for index in range(lo, hi):
                if self.presence[index] >= self.speak_level:
                    if run is None:
                        run = index
                    elif index - run + 1 >= need:
                        stops.append(run * self.frame_sec)
                        break
                else:
                    run = None
        return max(t0, min(stops))

    def voiced_reach(self, t0: float, t1: float, level: float | None = None) -> float:
        """``t0``부터 발성이 이어지는 동안 갈 수 있는 끝. 자르는 데 쓰지 않는다."""
        if self.dominance is None or t1 <= t0:
            return t0
        floor = _HOLD_DOMINANCE if level is None else level
        need = max(1, int(self.quiet_sec / self.dom_hop))
        deep_need = max(1, int(_HOLD_DEEP_SEC / self.dom_hop))
        lo, hi = int(t0 / self.dom_hop), min(len(self.dominance), int(t1 / self.dom_hop))
        run = deep_run = None
        for index in range(lo, hi):
            value = self.dominance[index]
            if self.deep and value < min(_HOLD_DEEP_LEVEL, floor):
                if deep_run is None:
                    deep_run = index
                elif index - deep_run + 1 >= deep_need:
                    return deep_run * self.dom_hop
            else:
                deep_run = None
            if value < floor:
                if run is None:
                    run = index
                elif index - run + 1 >= need:
                    return run * self.dom_hop
            else:
                run = None
        return (run * self.dom_hop) if run is not None else t1

    def held(self, t0: float, t1: float) -> bool:
        """그 구간 내내 "부르고 있는데 새 음소는 없다" — 늘임음."""
        if self.dominance is None or t1 <= t0:
            return False
        lo, hi = int(t0 / self.dom_hop), min(len(self.dominance), int(t1 / self.dom_hop))
        if hi - lo < 2:
            return False
        voiced = sum(1 for index in range(lo, hi) if self.dominance[index] >= _HOLD_DOMINANCE)
        return voiced >= 0.8 * (hi - lo)


def _extend_segments(
    segs: list[dict[str, Any]],
    line_end: float,
    hold_max: float,
    gate: "ExtendGate | None" = None,
    hold_max_held: float = 3.0,
    voiced_only: bool = False,
    voiced_level: float | None = None,
) -> int:
    """세그 끝을 다음 세그 시작까지 늘린다 — 노래방 표시 규약. **시작은 손대지 않는다.**

    ``hold_max``는 "간주에 흩어진 음절이 화면에서 쭉 늘어나는" 반대 증상을 막는 한도다.
    ``gate``가 있으면 그 상수는 오디오가 반박할 수 있는 기본값이 된다.
    """
    stretched = 0

    def reach(seg: dict[str, Any], boundary: float) -> float:
        cap = seg["start"] + hold_max
        if gate is None:
            return min(boundary, cap)
        if voiced_only:
            return min(
                cap,
                max(seg["end"], gate.voiced_reach(seg["end"], boundary, voiced_level)),
            )
        stop = gate.limit(seg["end"], boundary)
        if gate.held(seg["end"], stop):
            cap = seg["start"] + hold_max_held
        return min(boundary, stop, cap)

    for current, following in zip(segs, segs[1:]):
        target = reach(current, following["start"])
        if target > current["end"]:
            current["end"] = round(target, 3)
            stretched += 1
    if segs:
        target = reach(segs[-1], line_end)
        if target > segs[-1]["end"]:
            segs[-1]["end"] = round(target, 3)
            stretched += 1
    return stretched


def _extend_line_tails(
    lines: list[RefinedLine], key: str, gate: "ExtendGate", max_extra: float
) -> int:
    """라인 마지막 세그의 끝만 늘린다(표기 ``key`` 한정) — 자르지 않는다.

    다음 라인 시작(또는 ``line.end + max_extra``, 마지막 줄) 전까지, **우세도가 이어지는
    동안만** 민다. 발화 축은 안 본다 — 늘임음은 발화가 없는 구간이다.
    """
    moved = 0
    for index, line in enumerate(lines):
        segs = line.pron_segs.get(key)
        if not segs:
            continue
        boundary = lines[index + 1].start if index + 1 < len(lines) else line.end + max_extra
        ceiling = min(boundary, line.end + max_extra)
        if ceiling <= line.end + 0.05:
            continue
        stop = gate.voiced_reach(line.end, ceiling)
        if stop <= line.end + 0.05:
            continue
        if segs[-1].end < stop:
            segs[-1].end = round(stop, 3)
            moved += 1
    return moved


# ---------------------------------------------------------------------------
# 라인 재배치 — 반복 훅 · 창 겹침 보정
# ---------------------------------------------------------------------------


def _shift_line(line: RefinedLine, delta: float) -> None:
    """라인의 모든 표기 세그를 통째로 옮긴다(강체 이동). ``start``/``end``(앵커 값)는
    안 건드린다 — 반복 훅 보정은 세그 위치만 조정하는 국소 수정이다."""
    for segs in line.pron_segs.values():
        for seg in segs:
            seg.start = round(seg.start + delta, 3)
            seg.end = round(seg.end + delta, 3)


def _respace_repeated_lines(
    lines: list[RefinedLine], sources: list[str], min_run: int = 3, factor: float = 1.5
) -> int:
    """같은 가사가 연속 반복될 때 한 렌디션을 건너뛴 자리를 되돌린다(``two_pass.py`` 이식).

    간격 중앙값의 ``factor``배를 넘는 자리를 찾아 초과분만큼 뒤쪽을 당긴다. 라인
    ``start``(앵커 값)를 기준으로 간격을 재고, 세그는 강체 이동으로 따라간다.

    벤치 원본은 이 루프가 돌 때마다 ``line["start"]``도 함께 옮겨서(가변 딕셔너리) 다음
    반복이 "이미 고쳐졌다"를 자연히 보게 된다. 이 모듈은 ``RefinedLine.start``를 앵커
    값으로 고정해 두므로(불변식) 그 수렴 신호가 없다 — 대신 **로컬 사본**
    (``effective_starts``)에 보정을 누적해 같은 간격을 반복해서 다시 당기는 사고를 막는다.
    """
    fixed = 0
    index = 0
    while index < len(lines):
        text = sources[index].strip() if index < len(sources) else ""
        stop = index + 1
        while stop < len(lines) and text and sources[stop].strip() == text:
            stop += 1
        if text and stop - index >= min_run:
            effective_starts = [lines[k].start for k in range(index, stop)]
            for _ in range(stop - index):
                gaps = [b - a for a, b in zip(effective_starts, effective_starts[1:])]
                median = statistics.median(gaps) if gaps else 0.0
                if median <= 0.05:
                    break
                worst = max(range(len(gaps)), key=lambda k: gaps[k])
                excess = gaps[worst] - median
                if gaps[worst] <= median * factor or excess <= 0.10:
                    break
                for k in range(worst + 1, len(effective_starts)):
                    effective_starts[k] -= excess
                    _shift_line(lines[index + k], -excess)
                fixed += 1
        index = max(stop, index + 1)
    return fixed


def _enforce_monotonic(lines: list[RefinedLine], key: str) -> int:
    """라인 경계에서 세그가 뒤로 밀리는 것을 막는다(표기 ``key`` 한정).

    라인마다 ``[start-pad, end+pad]`` 창에서 따로 정렬하므로 그 pad가 인접 라인과
    겹치면 앞 라인의 마지막 세그가 다음 라인 첫 세그보다 늦게 끝날 수 있다. 겹친 구간을
    반으로 갈라 양쪽에 나눠 준다.
    """
    flat = [seg for line in lines for seg in line.pron_segs.get(key, [])]
    fixed = 0
    for previous, current in zip(flat, flat[1:]):
        if current.start >= previous.end:
            continue
        middle = (previous.end + current.start) / 2
        middle = max(middle, previous.start)
        previous.end = round(middle, 3)
        current.start = round(middle, 3)
        if current.end < current.start:
            current.end = current.start
        fixed += 1
    return fixed


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------


def refine_lines(
    anchor_lines: "list[SyncResult]",
    source_lines: list[str],
    refiner: SyllableRefiner,
    vocals_path: Path,
    *,
    language: str = "ja",
    config: TwoPassRefineConfig | None = None,
) -> list[RefinedLine]:
    """앵커 라인 창 안에서 경량 CTC로 표기별 음절 세그를 낸다.

    실패는 예외가 아니라 ``RefinedLine.fallback_reason``으로 신호한다 — 호출부(아직
    배선 전)가 실패한 라인에서 앵커의 원래 세그를 그대로 쓰도록 폴백하기 위해서다(이
    모듈은 하한을 보장할 뿐 앵커 세그 자체를 모른다 — "하한이 앵커 단독으로 고정된다"
    불변식).
    """
    import torch
    import torchaudio.functional as functional

    resolved_config = config or TwoPassRefineConfig()
    if len(anchor_lines) != len(source_lines):
        raise ValueError(
            f"앵커가 {len(source_lines)}줄 입력에 {len(anchor_lines)}줄을 돌려줬다"
        )

    lines = [
        RefinedLine(
            start=float(anchor.start_time),
            end=float(anchor.end_time),
            confidence=anchor.confidence,
        )
        for anchor in anchor_lines
    ]

    emission = refiner.emission_for(vocals_path)
    if emission is None:
        for line in lines:
            line.fallback_reason = "refiner_emission_unavailable"
        return lines

    vocab = emission.vocab
    frame_sec = emission.frame_sec
    total_frames = int(emission.emission.shape[1])
    device = emission.emission.device
    blank_id = emission.blank_id

    presence = None
    if resolved_config.spread_piles or resolved_config.extend_line_tails:
        try:
            presence = (1 - emission.emission[0][:, blank_id].exp()).float().cpu().numpy()
        except Exception:
            logger.warning("presence 계산 실패, 뭉침 펴기/늘이기 게이트 생략", exc_info=True)

    gate: ExtendGate | None = None
    if resolved_config.extend_line_tails or resolved_config.extend_voiced_only:
        import numpy as np

        made = _dominance_curve(vocals_path)
        if made is not None or presence is not None:
            speak_level = 2.0  # presence는 1을 못 넘으므로 "절대 안 걸림"
            if presence is not None:
                mask = np.zeros(len(presence), dtype=bool)
                for line in lines:
                    lo = max(0, int(line.start / frame_sec))
                    hi = min(len(presence), int(line.end / frame_sec))
                    if hi > lo:
                        mask[lo:hi] = True
                if bool(mask.any()):
                    speak_level = float(np.percentile(presence[mask], 90))
            gate = ExtendGate(
                made[0] if made else None,
                made[1] if made else 0.01,
                presence,
                frame_sec,
                speak_level,
                resolved_config.line_tail_quiet_sec,
                resolved_config.line_tail_deep,
            )
    seg_gate = gate if resolved_config.extend_voiced_only else None

    for line, source in zip(lines, source_lines):
        units = _derive_units(source, language)
        if not units.target:
            line.fallback_reason = "empty_derived_text"
            continue
        token_ids, ranges = _tokenize_target(units.target, vocab)
        if not token_ids:
            line.fallback_reason = "no_in_vocab_chars"
            continue
        start, end = line.start, line.end
        if end <= start:
            line.fallback_reason = "no_anchor_window"
            continue
        first = max(0, int((start - resolved_config.window_pad_sec) / frame_sec))
        last = min(
            total_frames, int(math.ceil((end + resolved_config.window_pad_sec) / frame_sec))
        )
        repeats = sum(1 for a, b in zip(token_ids, token_ids[1:]) if a == b)
        if last - first < len(token_ids) + repeats:
            line.fallback_reason = "window_shorter_than_targets"
            continue

        window = emission.emission[:, first:last, :]
        targets = torch.tensor([token_ids], dtype=torch.int32, device=device)
        try:
            aligned, scores = functional.forced_align(window, targets, blank=blank_id)
            spans = functional.merge_tokens(aligned[0], scores[0], blank=blank_id)
        except Exception:
            logger.warning("%s: 라인 재정렬 실패, 세그 없이 폴백", source[:24], exc_info=True)
            line.fallback_reason = "forced_align_failed"
            continue
        if len(spans) != len(token_ids):
            line.fallback_reason = "span_count_mismatch"
            continue

        offset = first * frame_sec
        for key, owners in units.owners.items():
            segs = _build_segments(owners, ranges, spans, offset, frame_sec, units.word_end)
            if not segs:
                continue
            if resolved_config.spread_piles and presence is not None:
                _spread_piled_segments(segs, presence, frame_sec)
            if resolved_config.extend_segments:
                _extend_segments(
                    segs,
                    end,
                    resolved_config.seg_hold_max_sec,
                    seg_gate,
                    resolved_config.seg_hold_max_held_sec,
                    resolved_config.extend_voiced_only,
                    resolved_config.seg_voiced_level,
                )
            spans_out = [
                PronSegmentSpan(
                    text=s["t"], start=s["start"], end=s["end"], word_end=bool(s.get("word_end"))
                )
                for s in segs
            ]
            line.pron_segs[key] = spans_out
            line.pron[key] = _join_pron(segs)
        line.refined = bool(line.pron_segs)

    if resolved_config.respace_repeats:
        _respace_repeated_lines(lines, source_lines)

    display_keys = {key for line in lines for key in line.pron_segs}
    for key in display_keys:
        _enforce_monotonic(lines, key)
    if resolved_config.extend_line_tails and gate is not None:
        for key in display_keys:
            _extend_line_tails(lines, key, gate, resolved_config.line_tail_max_sec)
            # 라인 꼬리를 민 뒤에는 pron 문자열이 안 바뀐다(텍스트는 그대로, 끝만 늘었다) —
            # 다시 조립할 필요가 없다.

    return lines
