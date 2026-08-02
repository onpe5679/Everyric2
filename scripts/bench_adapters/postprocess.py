"""프로드 VAD 보정층을 벤치 정렬기 위에 얹는 래퍼 어댑터.

## 왜 필요한가

``MMSBaselineAligner``의 docstring이 밝히듯 이 하네스는 그동안 프로드 워커가 정렬 **위에**
얹는 층(VAD 클램프·간주 앵커·늘임음 연장·붕괴 재합성)을 일부러 뺐다. 「정렬 모델 자체」를
재는 게 목적이었고, 보정층은 모델을 갈아끼워도 남는 층이라 후보 간 비교에는 영향이 없기
때문이다.

그런데 «최종 스택을 고른 뒤»에는 질문이 바뀐다 — 이 스택을 프로드에 넣으면 실제로 어떤
품질이 나오는가. 그러려면 보정층을 붙여서 재야 한다. 이 래퍼는 임의의 정렬기 출력에
프로드와 **같은 함수들**을 같은 순서로 적용한다(복제가 아니라 직접 import한다 — 프로드가
바뀌면 여기가 따라간다).

적용 순서는 ``worker.py``의 파이프라인 그대로다:

1. ``TimingPostProcessor(...).process(results, vad, "line")`` — 1차 정렬 후처리
2. ``_snap_silence_undershoot`` — 간주에 좌초한 전이 라인을 다음 발성 온셋으로 (독음 경로)
3. ``_clamp_stretched_lines`` — 병적으로 늘어진 라인 클램프. 내부에서 반복행 outlier
   클램프·간주 후 시작 당기기·소절 끝 늘임음 연장이 함께 돈다.

## VAD 입력이라는 함정

프로드는 **분리된 vocals**에 VAD를 건다. 무분리 경로(원곡 믹스)에 그대로 걸면 반주가
발성으로 잡혀 리전이 곡 전체로 번지고, 그러면 "발성 커버리지 50% 미만" 같은 판정이
무의미해진다. 그래서 이 래퍼는 **분리 스템이 있으면 그쪽에 VAD를 건다** — 라우팅의 빠른
경로처럼 정렬 자체는 무분리로 했더라도, 보정 판단만은 분리 신호로 내린다. 스템이 없으면
넘겨받은 파형을 쓰되 meta에 그 사실을 남긴다.

## 세그는 건드리지 않는다

라인이 움직여도 그 안의 음절 세그는 그대로 둔다. 프로드가 그렇게 한다 —
``TimingPostProcessor``는 ``word_segments``를 손대지 않고 넘기고, ``_resynth_word_segments``는
캡션 스캐폴드와 붕괴 재합성 두 자리에서만 불린다(worker.py:2629, 2937). 즉 VAD 클램프는
**라인 경계만 고치는 층**이다.

첫 구현에서 이 규약을 어기고 이동 라인마다 세그를 선형 리스케일했더니 음절 정확도가
88.8% → **43.5%**로 반토막 났다. 라인 경계는 추정치이고 세그는 CTC 실측이라, 추정치에 맞춰
실측을 옮기면 정보가 순손실이다.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.benchmark_alignment import AlignerAdapter, AlignOut

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
STEMS_ROOT = REPO_ROOT / "benchmark" / "stems"
# 보정 판단용 VAD를 걸 분리 스템 우선순위. 앞에 있는 것부터 찾는다.
VAD_STEM_PREFERENCE = ("bs-polarformer-fp16", "bs-polarformer", "htdemucs")


@dataclass(frozen=True)
class PostProcessConfig:
    name: str
    base: str
    # 독음(가나·한글) 입력으로 정렬한 경로에서만 도는 보정. 프로드도 alignment_text가
    # "pronunciation"일 때만 건다.
    pron_path: bool = True
    # 프로드 보정층 전체가 아니라 **병적 라인 절단 한 규칙만** 적용한다.
    clamp_only: bool = False
    note: str = ""


CONFIGS: tuple[PostProcessConfig, ...] = (
    PostProcessConfig(
        name="routed-2mode+pp",
        base="routed-2mode",
        note="2모드 라우팅 + 프로드 VAD 보정층",
    ),
    PostProcessConfig(
        name="2pass-owsm-omniasr+pp",
        base="2pass-owsm-omniasr",
        note="polar 2패스 + 프로드 VAD 보정층",
    ),
    # ★ja 채택 스택 + **라인 클램프만**. ``pron_path=False``라 무음 언더슛 스냅은 빠진다 —
    # 그건 간주에 좌초한 라인을 다음 온셋으로 통째로 옮기는 공격적인 보정이고, 지금 필요한
    # 것은 「병적으로 늘어진 라인을 자르고 소절 끝 늘임음을 실제 발성 끝까지 늘리는」 쪽이다
    # (사용자 지시 2026-08-02). ``_clamp_stretched_lines`` 안에 그 둘이 다 들어 있다.
    PostProcessConfig(
        name="2pass-owsm-mixed+pp",
        base="2pass-owsm-mixed",
        pron_path=False,
        clamp_only=True,
        note="ja 채택 스택 + 병적 라인 절단만",
    ),
    # ★en 채택 스택 + 병적 라인 절단. ja 7곡에서 437줄 중 0회 발동이라 무효로 판정했는데,
    # **그 판단이 ja 표본만으로 내려진 것이었다**. en에서는 명백히 발동해야 한다 — Madeon
    # 1:30~2:32의 비가창 62초(우세도 0.033)에 우리 레인이 38.0초를 덮는 반면 PROD는 0.2초다.
    # 원인은 라인28의 「윌」 한 세그가 32.19초를 먹은 것이고, 그 라인은 35.6초 지속에 발성
    # 커버리지 ~4%라 「8초 초과 + 커버리지 50% 미만」 조건에 정확히 걸린다(2026-08-02).
    PostProcessConfig(
        name="2pass-asr-ipa-hangul+pp",
        base="2pass-asr-ipa-hangul",
        pron_path=False,
        clamp_only=True,
        note="en 채택 스택 + 병적 라인 절단만",
    ),
    PostProcessConfig(
        name="2pass-asr-ipa-en+pp",
        base="2pass-asr-ipa-en",
        pron_path=False,
        clamp_only=True,
        note="en 원문 음절 + 병적 라인 절단만",
    ),
    PostProcessConfig(
        name="2pass-asr-ipa-phonetic+pp",
        base="2pass-asr-ipa-phonetic",
        pron_path=False,
        clamp_only=True,
        note="en IPA 전사 + 병적 라인 절단만",
    ),
    PostProcessConfig(
        name="omniasr-ctc+pp",
        base="omniasr-ctc",
        note="omniASR 단독 + 프로드 VAD 보정층",
    ),
)


def _resolve_base(name: str) -> Any:
    """기반 정렬기 인스턴스 — 하네스 레지스트리를 통째로 세워 찾는다."""
    from scripts.benchmark_alignment import ALIGNERS, _register_optional_aligners

    if name not in ALIGNERS:
        _register_optional_aligners()
    adapter = ALIGNERS.get(name)
    if adapter is None:
        raise ValueError(f"후처리 래퍼가 가리키는 정렬기가 없다: {name!r}")
    return adapter()


def _vad_source(vocals_path: Path) -> tuple[Path, str]:
    """VAD를 걸 파형 — 분리 스템이 있으면 그쪽, 없으면 넘겨받은 것."""
    video_id = vocals_path.parent.name
    for separator in VAD_STEM_PREFERENCE:
        candidate = STEMS_ROOT / separator / video_id / "vocals.wav"
        if candidate.is_file():
            return candidate, separator
    return vocals_path, "as-given"


# ── 우세도로 만든 발성 구간 ──
# 클램프 규칙(8초 초과 + 발성 커버리지 50% 미만)이 ja 437줄에서 0회, en에서도 0회 발동했다.
# 규칙이 필요 없어서가 아니라 **조건이 참이 될 수 없어서**다: VAD는 분리 스템 위에서 죽는다.
# Madeon 1:30~2:32는 청취로도 우세도로도 명백한 비가창인데(우세도 평균 0.033, 0.35 이상
# 프레임 3.9%) VAD는 그 구간을 「0:05.42~1:34.84 89초 통짜 발성」으로 읽어 라인28 커버리지가
# 0.779가 된다. ``star_prior.py``가 이미 실측해 둔 것과 같은 현상이다 — 간주 presence 0.979,
# 간주 RMS가 가창보다 3dB 낮을 뿐. **반주 대비 비율만이 3배 대비로 가른다.**
_DOMINANCE_LEVEL = 0.35
_DOMINANCE_MIN_SEC = 0.10


class _Region:
    __slots__ = ("start", "end")

    def __init__(self, start: float, end: float) -> None:
        self.start = start
        self.end = end


class _RegionSet:
    """``vad`` 자리에 그대로 끼울 수 있는 최소 형태."""

    def __init__(self, regions: list) -> None:
        self.regions = regions


def _dominance_activity(vocals_path: Path):
    """보컬 우세도 ≥ 0.35가 이어지는 구간. 못 만들면 None."""
    instrumental = vocals_path.with_name("inst.wav")
    if not instrumental.is_file():
        return None
    try:
        import librosa

        from everyric2.alignment.star_prior import vocal_presence_from_stems

        vocals, _ = librosa.load(str(vocals_path), sr=16_000, mono=True)
        accomp, _ = librosa.load(str(instrumental), sr=16_000, mono=True)
        made = vocal_presence_from_stems(vocals, accomp, 16_000, smooth_sec=0.2, hop_sec=0.01)
    except Exception:
        logger.warning("우세도 계산 실패 — VAD로 물러선다", exc_info=True)
        return None
    if made is None:
        return None
    values, hop = made[1], 0.01
    regions, run = [], None
    for index in range(len(values) + 1):
        active = index < len(values) and float(values[index]) >= _DOMINANCE_LEVEL
        if active and run is None:
            run = index
        elif not active and run is not None:
            if (index - run) * hop >= _DOMINANCE_MIN_SEC:
                regions.append(_Region(run * hop, index * hop))
            run = None
    if not regions:
        return None
    made_set = _RegionSet(regions)
    # 원곡선도 실어 둔다 — 머리 스냅(``_snap_silent_heads``)은 region(≥0.35·최소길이)보다
    # 고운 판정이 필요하다: 발성 바닥(0.12)과 절대 크기(dBFS)는 region으로는 못 본다.
    import numpy as np

    made_set.values = values
    made_set.hop = hop
    step = max(1, int(hop * 16_000))
    usable = (len(vocals) // step) * step
    frames = vocals[:usable].reshape(-1, step)
    made_set.db = 20 * np.log10(np.maximum(np.sqrt((frames**2).mean(axis=1)), 1e-9))
    return made_set


def _clamp_pathological(results, vad, line_body_region) -> set[int]:
    """``_clamp_stretched_lines``의 **첫 규칙만** — 병적으로 늘어진 라인 절단.

    원본은 네 규칙을 묶어 돈다(병적 절단 · 반복행 outlier · 간주 후 시작 당기기 · 소절 끝
    늘임음 연장). 묶음을 통째로 걸어 재 봤더니 ja 7곡 437줄 중 **423줄(96.8%)이 움직이는데
    정작 절단은 3줄**이었고, 지표는 라인 75.50→74.73 · 음절 74.40→73.61 · 구간
    61.47→59.50으로 내려갔다(2026-08-02). 라인이 중앙값 기준 25% 길어진 것이 원인이다.

    소절 끝 늘임음은 **세그 늘이기**가 이미 해결했다(구간 29.01→61.47%). 라인 단위로 또
    늘릴 이유가 없다. 그래서 잡을 3줄만 잡는 얇은 층으로 남긴다.

    조건과 절단 지점은 프로드와 같다 — 지속 8초 초과 + 발성 커버리지 50% 미만인 라인을
    글자 질량이 실린 발성 구간 끝으로 자른다. 정상 라인은 건드리지 않는다.
    """
    clamped: set[int] = set()
    for index, result in enumerate(results):
        duration = result.end_time - result.start_time
        if duration <= 8.0:
            continue
        regions = [
            reg for reg in vad.regions
            if reg.end > result.start_time and reg.start < result.end_time
        ]
        if not regions:
            continue
        vocal = sum(
            min(reg.end, result.end_time) - max(reg.start, result.start_time) for reg in regions
        )
        if vocal / duration >= 0.5:
            continue
        body = line_body_region(result, regions)
        if body is None:
            continue
        new_end = min(result.end_time, max(body.end + 0.3, result.start_time + 1.5))
        if new_end < result.end_time:
            result.end_time = new_end
            clamped.add(index)
    return clamped


def _coverage(regions: list, start: float, end: float) -> float:
    duration = max(1e-6, end - start)
    return sum(
        max(0.0, min(reg.end, end) - max(reg.start, start)) for reg in regions
    ) / duration


def _silent_run(regions: list, start: float, end: float) -> float:
    """[start, end] 안에서 발성 구간에 안 덮인 **최장 연속** 길이."""
    cursor, best = start, 0.0
    for reg in sorted(regions, key=lambda r: r.start):
        if reg.end <= start or reg.start >= end:
            continue
        best = max(best, reg.start - cursor)
        cursor = max(cursor, reg.end)
    return max(best, end - cursor)


def _fold_stranded_repeats(lines: list[dict[str, Any]], regions: list) -> int:
    """간주에 좌초한 **반복 렌디션**을 앞 렌디션 바로 뒤로 접는다.

    Madeon(The Prince) 라인29가 원형이다. 「This love will do」가 두 번 연속 불리는데
    (PROD 실측 1:25.27~1:26.74 · 1:26.74~1:30.19), 앵커 둘 다(owsm·omniasr) 두 번째
    렌디션을 62초 간주 위(2:02~2:14)에 앉혔다. emission을 자유 디코드해 보면 그 자리의
    신스 스탭이 「l」「o」(p 0.5~0.8)로 들린다 — love와 호환되는 가짜 증거가 실제로 있어서
    DP가 속은 것이고, 마지막 음절은 2:13.4의 짧은 발화성 소리(우세도 0.82)가 앵커였다.

    꼬리 클램프는 여기 무력하다 — **마지막 세그가 발성 위에** 있어 자를 무음 꼬리가 없다.
    라인 전체가 잘못 놓인 것이므로 라인 단위로 옮기는 수밖에 없다. 13곡 스캔에서 이 조건
    (지속 6초 이상 + 커버리지 0.30 미만 + 직전 라인과 같은 텍스트)에 걸리는 라인은 정확히
    그 하나였다(2026-08-02) — 오검출 0인 조건만 옮긴다.

    세그는 앞 렌디션의 리듬을 복사한다. 같은 텍스트를 같은 곡조로 부르므로 앞 렌디션의
    세그 오프셋이 우리가 가진 가장 좋은 추정이고, 세그 수가 다르면 비례 배치로 물러선다.
    """
    folded = 0
    for index in range(1, len(lines)):
        line, prev = lines[index], lines[index - 1]
        start, end = float(line["start"]), float(line["end"])
        if end - start < 6.0 or not line.get("segs"):
            continue
        if str(line.get("text") or "").strip() != str(prev.get("text") or "").strip():
            continue
        if not prev.get("segs") or float(prev["end"]) >= start:
            continue
        if _coverage(regions, start, end) >= 0.30:
            continue
        next_start = (
            float(lines[index + 1]["start"]) if index + 1 < len(lines) else float("inf")
        )
        prev_start, prev_end = float(prev["start"]), float(prev["end"])
        room = next_start - 0.05 - prev_end
        if room < 0.8:
            continue
        new_start = prev_end
        length = min(prev_end - prev_start, room)
        line["start"] = round(new_start, 3)
        line["end"] = round(new_start + length, 3)
        segs, pattern = line["segs"], prev["segs"]
        if len(segs) == len(pattern):
            scale = length / max(1e-6, prev_end - prev_start)
            for seg, ref in zip(segs, pattern):
                seg["start"] = round(new_start + (float(ref["start"]) - prev_start) * scale, 3)
                seg["end"] = round(new_start + (float(ref["end"]) - prev_start) * scale, 3)
        else:
            step = length / len(segs)
            for pos, seg in enumerate(segs):
                seg["start"] = round(new_start + pos * step, 3)
                seg["end"] = round(new_start + (pos + 1) * step, 3)
        line.setdefault("meta", {})["postprocessed"] = True
        folded += 1
    return folded


def _pull_disconnected_tails(lines: list[dict[str, Any]], regions: list) -> int:
    """몸통과 무음으로 끊긴 라인 꼬리 세그를 몸통 끝으로 되당긴다.

    rookie 0:14가 원형이다. 「booing」의 마지막 두 세그가 몸통(~0:11.0)에서 3.3초 떨어진
    0:14.3에 앉았는데, 그 자리는 가사에 없는 「데코」 캐치프레이즈다 — owsm 자유 디코드가
    「て」p=0.80 「こ」p=0.70으로 실제 가창(p 0.3~0.4)보다 또렷하게 들었다. 가사에 없는
    발화의 프레임을 DP가 가장 가까운 라인 꼬리에 먹인 것이다. butcher 2:08의 「라」「입」도
    같은 꼴이고(UST 실측 정위치 2:01.0, 7초 지각), Madeon 라인28의 「두」는 클램프가 길이를
    0으로 눌렀지만 **시작이 그대로 무음 위**(2:01.14)에 남아 있었다.

    판정: 라인 안 인접 세그 간격 ≥ 2초이고 그 사이 연속 무음 ≥ 1초. 13곡 스캔에서 5자리,
    UST·청취로 전부 결함 확정(2026-08-02). 꼬리 세그는 몸통 끝 +0.3초 지점으로 모으고
    라인 끝도 거기로 줄인다 — 세그 시작을 옮기는 드문 자리지만, 여기의 시작은 CTC 실측이
    아니라 남의 소리에 붙잡힌 잔해라는 것을 emission으로 확인했다.
    """
    pulled = 0
    for index, line in enumerate(lines):
        segs = line.get("segs") or []
        cut = None
        for k in range(len(segs) - 1):
            body_end = float(max(segs[k]["end"], segs[k]["start"]))
            tail_start = float(segs[k + 1]["start"])
            if tail_start - body_end < 2.0:
                continue
            if _silent_run(regions, body_end, tail_start) < 1.0:
                continue
            # 꼬리는 최대 2세그만 — 확정 결함 5건(rookie 「boo」「ing」·butcher 「라」「입」·
            # Madeon 「두」)은 전부 1~2세그였다. 소절 반쪽(10세그+)이 무음 뒤에 있으면 그건
            # 잔해가 아니라 **긴 휴지 뒤의 진짜 뒷소절**이다 — ロキ 「はあ… 寝言は寝て言え
            # ベイビー」(2:36.8 뒤 8초 쉬고 2:45.8부터 가창)를 당겼다가 진짜 가사 자리가
            # 통째로 비어 추임새 띠로 둔갑했다(사용자 청취, 2026-08-02). 오검출 1호.
            if len(segs) - (k + 1) > 2:
                continue
            cut = k
            break
        if cut is None:
            continue
        body_end = float(max(segs[cut]["end"], segs[cut]["start"]))
        next_start = (
            float(lines[index + 1]["start"]) if index + 1 < len(lines) else float("inf")
        )
        target = round(min(body_end + 0.3, next_start - 0.01), 3)
        target = max(target, body_end)
        for seg in segs[cut + 1 :]:
            seg["start"] = seg["end"] = target
        line["end"] = round(max(float(line["start"]), min(float(line["end"]), target)), 3)
        line.setdefault("meta", {})["postprocessed"] = True
        pulled += 1
    return pulled


def _snap_silent_heads(lines: list[dict[str, Any]], activity) -> int:
    """«명백한 무음» 위에 앉은 라인 머리를 첫 가창 온셋으로 되민다.

    rookie 라인0이 원형이다. 「Boo」가 0:00.00(우세도 0.00 · −180dBFS 디지털 무음)에서
    켜지는데 실제 소리는 1.22초 뒤에 난다. Black Wood 라인102 「Sleep.」(5:51.54, 앞 무음
    3.34초)도 같은 꼴 — 8차에서 «정렬기 레벨이라 표시층으로 못 잡는다»고 적었던 자리가
    사실은 이 조건으로 잡힌다.

    조건(14곡 스캔에서 3건 전부 실결함·오검출 0, 2026-08-02):
      · 시작 우세도 < 0.12(발성 바닥) · 시작부터 가창(≥0.35) 온셋까지 무음 런 ≥ 0.5초
      · 온셋이 라인 안에 있고 온셋 이후 커버리지 ≥ 0.4 — 통째로 무음에 앉은 좌초 라인
        (butcher 라인50이 그렇다)은 대상이 아니다. 그건 접기/블록 계열의 문제다.
      · 시작~온셋의 절대 크기 최대 < −30dBFS — **이 가드가 결정적이다.** 빼면 우세도만
        낮은 소프트 어택 6곳(토스트 1:31 등)이 걸려 진짜 온셋을 잘라먹는다.
    온셋은 «우세도 ≥ 0.35 **이면서 들리는(≥ −30dBFS)** 첫 프레임»이다 — 우세도는 비율이라
    디지털 무음에서 0/0 잡음으로 튄다(ロキ 인트로 0:00.31~0:02.04가 우세도 ≥0.35인데
    dB −90.6, 진짜 온셋은 0:17.66·−13.3dB — PROD 17.60과 일치). dB 축 없이 우세도만
    보면 가짜 구간에 걸려 멈춘다.
    시작을 옮기는 드문 자리다: 여기의 시작은 CTC 실측이 아니라 무음 위 잔해라는 것을
    우세도·절대 크기 두 축으로 확인한 경우만 옮긴다(``_pull_disconnected_tails``와 같은 원칙).
    """
    values = getattr(activity, "values", None)
    db = getattr(activity, "db", None)
    hop = getattr(activity, "hop", 0.01)
    if values is None or db is None:
        return 0
    snapped = 0
    total = len(values)
    audible_total = min(total, len(db))
    for line in lines:
        start, end = float(line["start"]), float(line["end"])
        k = int(start / hop)
        if k >= total or float(values[k]) >= 0.12:
            continue
        onset = k
        while onset < audible_total and (
            float(values[onset]) < _DOMINANCE_LEVEL or float(db[onset]) < -30.0
        ):
            onset += 1
        if onset >= audible_total or (onset - k) * hop < 0.5:
            continue
        onset_t = onset * hop
        if onset_t >= end:
            continue
        hi = min(total, int(end / hop))
        covered = sum(1 for x in values[onset:hi] if float(x) >= _DOMINANCE_LEVEL)
        if hi <= onset or covered / (hi - onset) < 0.4:
            continue
        if float(max(db[k : min(onset, len(db))], default=-999.0) if isinstance(db, list)
                 else db[k : min(onset, len(db))].max()) >= -30.0:
            continue
        new_start = round(max(start, onset_t - 0.05), 3)
        shift = new_start - start
        untouched = next(
            (float(s["start"]) for s in line.get("segs") or [] if float(s["start"]) >= new_start),
            end,
        )
        for seg in line.get("segs") or []:
            if float(seg["start"]) >= new_start:
                continue
            duration = max(0.0, float(seg["end"]) - float(seg["start"]))
            seg["start"] = round(min(float(seg["start"]) + shift, untouched), 3)
            seg["end"] = round(min(end, float(seg["start"]) + duration), 3)
        line["start"] = new_start
        line.setdefault("meta", {})["postprocessed"] = True
        snapped += 1
    return snapped


class PostProcessedAligner(AlignerAdapter):
    """기반 정렬기 출력에 프로드 보정층을 적용한다."""

    name: str = ""
    config: PostProcessConfig

    def __init__(self, config: PostProcessConfig | None = None) -> None:
        if config is None:
            config = self.config
        self.config = config
        self.name = config.name
        self._base: Any | None = None

    def align(self, vocals_path: Path, lyrics: str, language: str) -> AlignOut:
        started = time.perf_counter()
        out = self._base_aligner().align(vocals_path, lyrics, language)
        stats = self._apply(out.lines, vocals_path)
        out.elapsed_sec = round(time.perf_counter() - started, 2)
        out.meta = {**out.meta, "postprocess": stats}
        return out

    def _apply(self, lines: list[dict[str, Any]], vocals_path: Path) -> dict[str, Any]:
        from everyric2.audio.loader import AudioLoader
        from everyric2.audio.vad import VocalActivityDetector
        from everyric2.config.settings import get_settings
        from everyric2.inference.prompt import SyncResult, WordSegment
        from everyric2.server.worker import _clamp_stretched_lines, _snap_silence_undershoot

        settings = get_settings()
        source, stem_used = _vad_source(vocals_path)
        started = time.perf_counter()
        try:
            audio = AudioLoader(settings.audio).load(source)
            vad = VocalActivityDetector().detect(audio)
        except Exception as exc:
            logger.warning("%s: VAD 실패, 보정 생략 — %r", self.name, exc)
            return {"applied": False, "error": repr(exc)[:200]}

        originals = [(float(line["start"]), float(line["end"])) for line in lines]
        results = [
            SyncResult(
                text=str(line.get("text") or ""),
                start_time=float(line["start"]),
                end_time=float(line["end"]),
                confidence=line.get("confidence"),
                line_number=index,
                word_segments=[
                    WordSegment(word=str(seg.get("t") or ""), start=float(seg["start"]),
                                end=float(seg["end"]))
                    for seg in (line.get("segs") or [])
                ],
            )
            for index, line in enumerate(lines)
        ]

        snapped: set[int] = set()
        activity, activity_source = vad, "vad"
        if self.config.clamp_only:
            from everyric2.server.worker import _line_body_region

            made = _dominance_activity(source)
            if made is not None:
                activity, activity_source = made, "dominance"
            corrected = results
            clamped = _clamp_pathological(results, activity, _line_body_region)
        else:
            from everyric2.alignment.timing_postprocess import TimingPostProcessor

            # extend_to_vocal=False는 프로드와 같다 — star가 흡수해 둔 구간을 도로 끌어안는
            # 역효과를 막는다(worker.py 주석 참고).
            pp = TimingPostProcessor(settings.segmentation, extend_to_vocal=False).process(
                results, vad, "line"
            )
            if self.config.pron_path:
                try:
                    _snap_silence_undershoot(pp.results, vad, snapped)
                except Exception as exc:
                    logger.warning("%s: 무음 언더슛 스냅 실패 — %r", self.name, exc)
            corrected, clamped = _clamp_stretched_lines(pp.results, vad)

        moved = trimmed = 0
        for line, result, old in zip(lines, corrected, originals):
            new = (float(result.start_time), float(result.end_time))
            if abs(new[0] - old[0]) < 1e-6 and abs(new[1] - old[1]) < 1e-6:
                continue
            moved += 1
            # **세그는 건드리지 않는다.** 프로드도 그렇다 — ``TimingPostProcessor``는
            # word_segments를 그대로 넘기고, ``_resynth_word_segments``는 캡션 스캐폴드와
            # 붕괴 재합성 두 자리에서만 불린다(worker.py:2629, 2937). 라인이 움직였다고
            # 세그를 따라 옮기면 CTC **실측** 음절 타이밍을 라인 경계 추정치로 덮어쓰는 셈이라,
            # 실제로 음절 정확도가 88.8% → 43.5%로 반토막 났다(첫 구현 실측).
            line["start"], line["end"] = new
            line.setdefault("meta", {})["postprocessed"] = True
            # 세그 **시작**은 그대로 둔다(CTC 실측). 다만 표시용으로 다음 세그 시작까지
            # 늘여 둔 **끝**이 새 라인 끝을 넘으면 줄인다 — 안 그러면 클램프로 잘라낸 간주에
            # 하이라이트가 그대로 남는다. 우리가 만든 늘이기를 되돌리는 것이지 실측을
            # 옮기는 것이 아니다(``two_pass._extend_segments`` 참조).
            for seg in line.get("segs") or []:
                if seg["end"] > new[1]:
                    seg["end"] = round(max(seg["start"], new[1]), 3)
                    trimmed += 1

        # 좌초 보정 두 장치 — 둘 다 우세도 위에서만 판정한다(분리 스템에서 VAD가 죽는 것이
        # 이 층의 교훈이었다). 접기를 먼저 돌린다: 좌초 라인이 제자리로 가면 그 안의 끊긴
        # 꼬리는 함께 사라지므로, 순서를 바꾸면 접힐 라인의 꼬리를 먼저 뭉개 버린다.
        # 갭 되당김(`_pull_stranded_into_gap` — butcher L50을 간주 무음에서 직전 갭의
        # 미설명 발성으로 옮기는 장치)은 청취로 기각(2026-08-02): 유일 발동지 butcher에서
        # 2:07~2:12가 통째로 비어 2:11 「The slaughter's on」 시작이 앵커 단독보다 더
        # 어긋나 보였다. 블록 지각 곡은 손대지 않는 것이 낫다 — 잔해 이동은 이동한 라인만
        # 보면 이득이어도 «비워진 자리»가 새 증상이 된다.
        folded = pulled = snapped_heads = 0
        if self.config.clamp_only and activity_source == "dominance":
            folded = _fold_stranded_repeats(lines, activity.regions)
            pulled = _pull_disconnected_tails(lines, activity.regions)
            snapped_heads = _snap_silent_heads(lines, activity)

        return {
            "applied": True,
            "vad_stem": stem_used,
            "vad_regions": len(vad.regions),
            "activity_source": activity_source,
            "activity_regions": len(activity.regions),
            "moved_lines": moved,
            "trimmed_segments": trimmed,
            "clamped_lines": len(clamped),
            "snapped_lines": len(snapped),
            "folded_lines": folded,
            "pulled_tails": pulled,
            "snapped_heads": snapped_heads,
            "total_lines": len(lines),
            "elapsed_sec": round(time.perf_counter() - started, 3),
        }

    def _base_aligner(self) -> Any:
        if self._base is None:
            self._base = _resolve_base(self.config.base)
        return self._base


def register(aligner_registry: dict[str, type[AlignerAdapter]]) -> None:
    for spec in CONFIGS:
        aligner_registry[spec.name] = _config_class(spec)


def _config_class(spec: PostProcessConfig) -> type[PostProcessedAligner]:
    class ConfiguredPostProcessedAligner(PostProcessedAligner):
        name = spec.name
        config = spec

    ConfiguredPostProcessedAligner.__name__ = "PostProcessed_" + spec.name.replace("-", "_").replace("+", "_")
    ConfiguredPostProcessedAligner.__qualname__ = ConfiguredPostProcessedAligner.__name__
    return ConfiguredPostProcessedAligner
