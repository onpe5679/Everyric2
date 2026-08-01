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
        if self.config.clamp_only:
            from everyric2.server.worker import _line_body_region

            corrected = results
            clamped = _clamp_pathological(results, vad, _line_body_region)
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

        return {
            "applied": True,
            "vad_stem": stem_used,
            "vad_regions": len(vad.regions),
            "moved_lines": moved,
            "trimmed_segments": trimmed,
            "clamped_lines": len(clamped),
            "snapped_lines": len(snapped),
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
