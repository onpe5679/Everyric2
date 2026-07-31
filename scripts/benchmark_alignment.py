"""분리·정렬 후보 모델 광역 스윕 A/B 하네스 (모델 교체 이니셔티브 P1).

스펙: ``benchmark/HARNESS_SPEC.md``. 로컬 GPU 전용 — 서버·프로드 DB를 건드리지 않는다.

축은 둘이다. **분리기**(SeparatorAdapter)와 **정렬기**(AlignerAdapter)를 각각 레지스트리에
꽂고 조합으로 돌린다. 분리 결과는 ``(separator, video_id)``로 디스크 캐시되므로 정렬 후보를
추가로 돌려도 분리를 다시 하지 않는다.

기준(reference)은 ``benchmark/eval_set.json``의 ``baseline_timestamps`` — **프로드 현행 스택이
낸 라인 시각**이다. 사람이 검증한 정답이 아니라 후보들을 같은 자에 대는 **공통 비교 기준**일
뿐이다. 기준 자체가 밀려 있는 곡에서는 후보의 MAE가 커지는 것이 오히려 개선일 수 있으므로,
수치와 함께 반드시 검수 산출물(스템 wav·SRT·diff HTML)을 듣고 봐야 한다.

    ./.venv/Scripts/python.exe scripts/benchmark_alignment.py \
        --separators htdemucs --aligners mms-baseline --songs <vid1>,<vid2>

    # 오디오 확보 상태만 보기 / 계산 없이 runs 캐시만으로 REPORT.md 다시 쓰기
    ./.venv/Scripts/python.exe scripts/benchmark_alignment.py --list --strata ja
    ./.venv/Scripts/python.exe scripts/benchmark_alignment.py --report-only

후보 모델 추가 방법
-------------------
분리 후보: ``SeparatorAdapter``를 상속해 ``name``과 ``separate(audio_path, work_dir)``만
구현하고 ``SEPARATORS``에 등록한다. ``work_dir``에 ``vocals.wav``·``inst.wav``를 남기고
``SeparationOut``으로 그 경로와 소요 시간·VRAM을 돌려주면 된다(VRAM은 ``VramProbe``로 감싸면
자동). 스템 캐시·검수 트리·리포트는 이름만 보고 따라온다.

정렬 후보: ``AlignerAdapter``를 상속해 ``name``과 ``align(vocals_path, lyrics, language)``만
구현하고 ``ALIGNERS``에 등록한다. 돌려줄 ``AlignOut.lines``는 **가사 줄 순서 그대로**
``{text, start, end, confidence, measured}`` 목록이고, ``quality_score``는 붕괴 게이트가 쓰므로
비교 가능한 값이어야 한다(현행 규칙은 ``_quality_score`` 참고 — 라인 conf 평균 + 커버리지
하한). 지표·짝짓기·산출물은 손댈 필요가 없다.

인스턴스는 스윕 내내 하나만 만들어 재사용하므로 모델을 상주시켜도 된다. 무거운 import는
어댑터 메서드 안에서 지연 import할 것 — ``--help``와 ``--list``가 torch 없이 떠야 한다.
"""

from __future__ import annotations

import argparse
import difflib
import html
import json
import math
import os
import shutil
import statistics
import sys
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_BENCH_DIR = REPO_ROOT / "benchmark"

# PCO 창(초). MIREX 관행값이자 스펙 지정값.
PCO_WINDOW = 0.3
# 「밀린 라인」 판정 임계(초) — diff HTML 하이라이트와 초과 비율 지표가 공유한다.
LATE_THRESHOLD = 0.3

# 아래 두 값은 everyric2.server.worker의 quality_score 산출 규칙을 그대로 옮긴 것이다
# (ALIGNED_LINE_RATIO_MIN / FAILED_ALIGNMENT_QUALITY). worker를 import하지 않는 이유는
# 하네스를 서버 스택(fastapi·DB 설정)에서 독립시키기 위해서다 — 규칙이 바뀌면 여기도 고쳐야
# 한다. 확장이 저신뢰 경고를 띄우는 고정 임계(0.001)도 붕괴 게이트에서 같은 값을 쓴다.
COVERAGE_MIN = 0.5
FAILED_ALIGNMENT_QUALITY = 0.0
LOW_CONF_THRESHOLD = 0.001


# ──────────────────────────────────────────────────────────────────────────
# VRAM 계측
# ──────────────────────────────────────────────────────────────────────────


class VramProbe:
    """스테이지 동안의 VRAM 피크. 두 값을 따로 낸다.

    ``process_peak_mb``: torch allocator의 ``max_memory_allocated`` — 인프로세스 모델(CTC
    정렬 등)에 유효하고, 서브프로세스로 도는 분리기(demucs)에서는 0에 가깝다.
    ``device_peak_mb``: ``mem_get_info``를 폴링해 잡은 **디바이스 전체** 사용량 피크 —
    서브프로세스도 잡히지만 다른 프로세스의 사용량까지 섞인다. 후보 간 비교는 같은 스코프
    끼리만 해야 한다.

    두 값 모두 **상주 모델이 바닥값으로 깔린다**: ``reset_peak_memory_stats``는 피크를 0이
    아니라 «현재 할당량»으로 되돌리므로, 정렬 모델을 띄워 둔 뒤 재는 분리 스테이지의
    프로세스 피크에는 그 모델이 그대로 포함된다. 스테이지 증분이 아니라 그 시점의 상한으로
    읽어야 한다.
    """

    def __init__(self, interval: float = 0.25):
        self.interval = interval
        self.process_peak_mb: float | None = None
        self.device_peak_mb: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._device_peak_bytes = 0
        self._torch = None

    def _sample(self) -> None:
        torch = self._torch
        while not self._stop.is_set():
            try:
                free, total = torch.cuda.mem_get_info()
                self._device_peak_bytes = max(self._device_peak_bytes, total - free)
            except Exception:
                return
            self._stop.wait(self.interval)

    def __enter__(self) -> VramProbe:
        try:
            import torch

            if not torch.cuda.is_available():
                return self
            self._torch = torch
            torch.cuda.reset_peak_memory_stats()
            self._thread = threading.Thread(target=self._sample, daemon=True)
            self._thread.start()
        except Exception:
            self._torch = None
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._torch is None:
            return
        try:
            self.process_peak_mb = round(self._torch.cuda.max_memory_allocated() / 2**20, 1)
        except Exception:
            self.process_peak_mb = None
        if self._device_peak_bytes:
            self.device_peak_mb = round(self._device_peak_bytes / 2**20, 1)


# ──────────────────────────────────────────────────────────────────────────
# 어댑터 2축
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class SeparationOut:
    vocals_path: Path
    inst_path: Path
    elapsed_sec: float | None = None
    vram_peak_mb: float | None = None
    vram_device_peak_mb: float | None = None
    cached: bool = False
    note: str | None = None

    def meta(self) -> dict:
        return {
            "elapsed_sec": self.elapsed_sec,
            "vram_peak_mb": self.vram_peak_mb,
            "vram_device_peak_mb": self.vram_device_peak_mb,
            "note": self.note,
        }


@dataclass
class AlignOut:
    """정렬 후보 하나의 출력. ``lines``는 가사 줄과 1:1이고 순서를 유지한다."""

    lines: list[dict] = field(default_factory=list)
    elapsed_sec: float | None = None
    vram_peak_mb: float | None = None
    vram_device_peak_mb: float | None = None
    quality_score: float | None = None
    meta: dict = field(default_factory=dict)


class SeparatorAdapter:
    """분리 후보 하나. ``name``이 레지스트리 키이자 결과 트리의 디렉터리 이름이다."""

    name: str = ""

    def separate(self, audio_path: Path, work_dir: Path) -> SeparationOut:
        raise NotImplementedError


class AlignerAdapter:
    """정렬 후보 하나. 인스턴스는 스윕 내내 재사용되므로 모델을 상주시켜도 된다."""

    name: str = ""

    def align(self, vocals_path: Path, lyrics: str, language: str) -> AlignOut:
        raise NotImplementedError


class HtdemucsSeparator(SeparatorAdapter):
    """현행 프로드 분리 경로 (``everyric2.audio.separator.VocalSeparator``, demucs 서브프로세스).

    프로드와 **같은 입력 열화**를 유지한다: VocalSeparator는 오디오를 먼저
    ``AudioLoader``(기본 24kHz·모노)로 읽어 wav로 떨어뜨린 뒤 demucs에 넘긴다. 즉 분리기가
    보는 것은 원본이 아니라 24kHz 모노 다운믹스다. 여기서 원본을 직접 먹이면 기준선이
    프로드와 달라지므로 P1에서는 일부러 같은 경로를 쓴다 — 분리 후보 비교(P3)에서 이 열화
    자체를 축으로 뺄지는 그때 정한다.

    스템은 demucs가 낸 원본 wav(44.1kHz 스테레오)를 그대로 캐시로 옮긴다. 검수용 재생
    품질을 위해서다. 정렬기는 이 파일을 ``AudioLoader``로 다시 읽으므로(24kHz 모노) 프로드가
    정렬에 쓰는 파형과 동일하다.
    """

    name = "htdemucs"
    demucs_model = "htdemucs"

    def separate(self, audio_path: Path, work_dir: Path) -> SeparationOut:
        import torch

        from everyric2.audio.separator import VocalSeparator
        from everyric2.config.settings import get_settings

        work_dir.mkdir(parents=True, exist_ok=True)
        # temp_dir을 곡별 작업 디렉터리로 갈아끼운다 — VocalSeparator는 demucs 입출력에
        # 고정 파일명(demucs_input.wav)을 쓰므로 공유 temp_dir을 그대로 쓰면 병행 실행이
        # 서로를 덮는다. validator를 우회하는 model_copy라 mkdir은 위에서 직접 했다.
        audio_cfg = get_settings().audio.model_copy(update={"temp_dir": work_dir})
        separator = VocalSeparator(audio_cfg)
        if not separator.is_available():
            raise RuntimeError("demucs is not installed in this interpreter")

        with VramProbe() as probe:
            started = time.perf_counter()
            result = separator.separate_file(
                audio_path, model=self.demucs_model, use_gpu=torch.cuda.is_available()
            )
            elapsed = time.perf_counter() - started

        raw_dir = work_dir / "demucs_output" / self.demucs_model / "demucs_input"
        vocals_out = work_dir / "vocals.wav"
        inst_out = work_dir / "inst.wav"
        note = None
        if (raw_dir / "vocals.wav").exists() and (raw_dir / "no_vocals.wav").exists():
            shutil.move(str(raw_dir / "vocals.wav"), str(vocals_out))
            shutil.move(str(raw_dir / "no_vocals.wav"), str(inst_out))
        else:
            # demucs 출력 레이아웃이 바뀐 경우 — 로드된 AudioData로 떨어뜨리고 사실을 남긴다
            result.vocals.to_file(vocals_out)
            result.accompaniment.to_file(inst_out)
            note = "demucs raw stems not found; wrote loader-decoded (mono) stems instead"
        shutil.rmtree(work_dir / "demucs_output", ignore_errors=True)

        return SeparationOut(
            vocals_path=vocals_out,
            inst_path=inst_out,
            elapsed_sec=round(elapsed, 2),
            vram_peak_mb=probe.process_peak_mb,
            vram_device_peak_mb=probe.device_peak_mb,
            note=note,
        )


class NosepSeparator(SeparatorAdapter):
    """무분리 대조군 — 원곡 믹스를 그대로 vocals 자리에 놓는다.

    신 정렬기들이 분리 품질을 얼마나 타는지 재는 축. 정렬기가 실제로 보는 파형과 동일한
    ``AudioLoader`` 경로(기본 24kHz 모노)로 떨어뜨린다 — 분리 스킵의 효과만 남기고 입력
    열화 조건은 다른 분리기와 맞추기 위해서다. ``inst.wav``는 형식상 필요해 같은 길이의
    무음을 놓는다(뷰어의 원곡 재생은 별도 소스라 영향 없음).
    """

    name = "nosep"

    def separate(self, audio_path: Path, work_dir: Path) -> SeparationOut:
        import numpy as np

        from everyric2.audio.loader import AudioData, AudioLoader
        from everyric2.config.settings import get_settings

        work_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        audio = AudioLoader(get_settings().audio).load(audio_path)
        vocals_out = audio.to_file(work_dir / "vocals.wav")
        inst_out = AudioData(
            waveform=np.zeros_like(audio.waveform),
            sample_rate=audio.sample_rate,
            duration=audio.duration,
        ).to_file(work_dir / "inst.wav")
        return SeparationOut(
            vocals_path=vocals_out,
            inst_path=inst_out,
            elapsed_sec=round(time.perf_counter() - started, 2),
            note="no separation; vocals is the original mix (AudioLoader-decoded)",
        )


class MMSBaselineAligner(AlignerAdapter):
    """현행 프로드 정렬 엔진의 **맨몸 경로** (``CTCEngine.align``, MMS-1B-all 어댑터).

    프로드 워커가 이 위에 얹는 것들(자막 앵커·독음 이중정렬·심판·VAD 클램프·붕괴 재합성)은
    **일부러 뺐다**. 이 하네스가 재는 것은 「정렬 모델 자체」의 광역 성능이고, 워커의 보정은
    모델을 갈아끼워도 그대로 남는 층이기 때문이다. 따라서 기준선(baseline_timestamps, 보정
    전부 포함)과의 잔차에는 «모델 차이»와 «보정층 부재»가 섞여 있다 — 후보 간 비교는 공정하고
    (모두 같은 맨몸 조건), 기준선과의 절대 잔차는 그렇게 읽어야 한다.
    """

    name = "mms-baseline"

    def __init__(self) -> None:
        self._engine = None
        self._loader = None

    def _engine_and_loader(self):
        if self._engine is None:
            from everyric2.alignment.factory import EngineFactory
            from everyric2.audio.loader import AudioLoader
            from everyric2.config.settings import get_settings

            settings = get_settings()
            engine = EngineFactory.get_engine("ctc", settings.alignment)
            if not engine.is_available():
                raise RuntimeError("CTC engine unavailable (transformers/torchaudio missing)")
            self._engine = engine
            self._loader = AudioLoader(settings.audio)
        return self._engine, self._loader

    def align(self, vocals_path: Path, lyrics: str, language: str) -> AlignOut:
        from everyric2.inference.prompt import LyricLine

        engine, loader = self._engine_and_loader()
        audio = loader.load(vocals_path)
        lyric_lines = LyricLine.from_text(lyrics)
        if not lyric_lines:
            raise RuntimeError("lyrics produced zero lines")

        # 모델 로드를 타이머 **밖으로** 뺀다. 첫 곡의 정렬 시간에 MMS-1B 가중치 로드(5090
        # 실측 약 11초)가 섞여 있으면 「첫 곡이 6배 느린」 표가 나오고 후보 간 처리 시간
        # 비교가 무의미해진다. 언어를 명시해 부르므로 align()의 언어 해석은 이 호출과
        # 완전히 같고(auto 감지 분기를 타지 않는다), align()은 같은 언어면 재확인만 하고
        # 넘어가므로 정렬 결과는 한 치도 바뀌지 않는다. 사설 API라 실패는 삼키고 사실만 남긴다.
        load_sec = None
        if language and language != "auto":
            try:
                load_started = time.perf_counter()
                engine._ensure_model_loaded(language)
                load_sec = round(time.perf_counter() - load_started, 2)
            except Exception:
                load_sec = None

        with VramProbe() as probe:
            started = time.perf_counter()
            results = engine.align(audio, lyric_lines, language=language or "auto")
            elapsed = time.perf_counter() - started

        lines: list[dict] = []
        for r in results:
            conf = r.confidence
            if conf is None and r.word_segments:
                conf = _geomean([w.confidence for w in r.word_segments])
            lines.append(
                {
                    "text": r.text,
                    "start": r.start_time,
                    "end": r.end_time,
                    "confidence": None if conf is None else round(float(conf), 6),
                    # 실측 글자 타이밍이 있는 줄만 «정렬이 성립한» 줄이다. 나머지는
                    # ctc_engine._interpolate_unaligned가 이웃 사이로 채운 보간 산물이다.
                    "measured": bool(r.word_segments),
                    "chars": len(r.word_segments or []),
                    # 글자/음절 실측 스팬 — 음절 지표·검수 뷰어의 원료 (후보 어댑터와 같은 계약)
                    "segs": [
                        {"t": w.word, "start": round(float(w.start), 3), "end": round(float(w.end), 3)}
                        for w in (r.word_segments or [])
                    ],
                }
            )

        quality, coverage = _quality_score(lines)
        return AlignOut(
            lines=lines,
            elapsed_sec=round(elapsed, 2),
            vram_peak_mb=probe.process_peak_mb,
            vram_device_peak_mb=probe.device_peak_mb,
            quality_score=quality,
            meta={
                "audio_sec": round(audio.duration, 2),
                # 정렬 시간과 분리해 잰 모델 로드 시간. None이면 로드를 타이머 밖으로 못 빼서
                # elapsed_sec에 섞여 있다는 뜻이다 (해석에 필요하므로 반드시 확인할 것).
                "model_load_sec": load_sec,
                "engine_lang": getattr(engine, "_current_lang", None),
                "engine_adapter": getattr(engine, "_current_adapter", None),
                "coverage": coverage,
                "star_spans": len(getattr(engine, "_last_star_spans", []) or []),
            },
        )


SEPARATORS: dict[str, type[SeparatorAdapter]] = {
    HtdemucsSeparator.name: HtdemucsSeparator,
    NosepSeparator.name: NosepSeparator,
}
ALIGNERS: dict[str, type[AlignerAdapter]] = {
    MMSBaselineAligner.name: MMSBaselineAligner,
}


def _register_optional_aligners() -> None:
    """선택적 후보 배선(현재: HF CTC 10종, ko 5 + ja 5).

    이 스크립트를 직접 실행하면(``__name__ == "__main__"``) ``scripts.bench_adapters.hf_ctc``가
    다시 ``from scripts.benchmark_alignment import ...``로 이 모듈을 끌어오는데, 그 재-import는
    ``scripts.benchmark_alignment``라는 별도 모듈 이름으로 이 파일 전체를 처음부터 다시
    실행한다. 모듈 최상단(ALIGNERS 정의 직후)에서 바로 이 함수를 불렀다면 그 두 번째 실행에서도
    또 같은 배선을 시도해 순환 임포트(ImportError: partially initialized module)로 죽는다.
    ``main()``에서만 호출해 두 번째 실행 시엔 ``__name__``이 "__main__"이 아니므로
    호출 자체가 안 일어나게 막는다 — 순환을 끊는 지점이 여기다.
    """
    try:
        from scripts.bench_adapters.hf_ctc import register as _register_hf_ctc

        _register_hf_ctc(ALIGNERS)
    except Exception as exc:  # extras 없는 환경에서도 하네스 자체는 계속 쓸 수 있어야 한다
        print(f"[bench_adapters] hf_ctc 후보 배선 실패, 건너뜀: {exc!r}", file=sys.stderr)
    try:
        from scripts.bench_adapters.omni_ctc import register as _register_omni

        _register_omni(ALIGNERS)
    except Exception as exc:
        print(f"[bench_adapters] omniASR 후보 배선 실패, 건너뜀: {exc!r}", file=sys.stderr)
    try:
        from scripts.bench_adapters.owsm_ctc import register as _register_owsm

        _register_owsm(ALIGNERS)
    except Exception as exc:
        print(f"[bench_adapters] OWSM 후보 배선 실패, 건너뜀: {exc!r}", file=sys.stderr)
    try:
        from scripts.bench_adapters.separators_roformer import register as _register_sep

        _register_sep(SEPARATORS)
    except Exception as exc:
        print(f"[bench_adapters] 분리 후보 배선 실패, 건너뜀: {exc!r}", file=sys.stderr)
    # 신규 분리 후보군 — 품질파(BS-RoFormer 계열)·경량파(ONNX/경량 모델) 모듈은 분리 파일로
    # 유지한다(작업 팀이 달라 같은 파일 충돌을 피하려는 배선 결정).
    for _mod_name in ("separators_quality", "separators_light"):
        try:
            import importlib

            _mod = importlib.import_module(f"scripts.bench_adapters.{_mod_name}")
            _mod.register(SEPARATORS)
        except ModuleNotFoundError:
            pass  # 아직 미구현 — 정상
        except Exception as exc:
            print(f"[bench_adapters] {_mod_name} 배선 실패, 건너뜀: {exc!r}", file=sys.stderr)
    try:
        # 비-CTC 후보(C 트랙): 전용 venv 서브프로세스라 여기서는 임포트만 하고 무게는 워커가 진다
        from scripts.bench_adapters.qwen3_fa import register as _register_qwen3_fa

        _register_qwen3_fa(ALIGNERS)
    except Exception as exc:
        print(f"[bench_adapters] qwen3-fa 후보 배선 실패, 건너뜀: {exc!r}", file=sys.stderr)
    try:
        from scripts.bench_adapters.nemo_nfa import register as _register_nemo_nfa

        _register_nemo_nfa(ALIGNERS)
    except Exception as exc:
        print(f"[bench_adapters] nemo-nfa 후보 배선 실패, 건너뜀: {exc!r}", file=sys.stderr)
    try:
        from scripts.bench_adapters.parakeet_ctc_ja import register as _register_parakeet_ja

        _register_parakeet_ja(ALIGNERS)
    except Exception as exc:
        print(f"[bench_adapters] parakeet-ctc-ja 후보 배선 실패, 건너뜀: {exc!r}", file=sys.stderr)
    try:
        # nemo-nfa의 VRAM 절감 시도(bf16). nemo_nfa.py를 직접 건드리지 않고 별도 파일로
        # 분리했다 — 이유는 scripts/bench_adapters/nemo_nfa_bf16.py 모듈 docstring 참고.
        from scripts.bench_adapters.nemo_nfa_bf16 import register as _register_nemo_nfa_bf16

        _register_nemo_nfa_bf16(ALIGNERS)
    except Exception as exc:
        print(f"[bench_adapters] nemo-nfa-bf16 후보 배선 실패, 건너뜀: {exc!r}", file=sys.stderr)


# ──────────────────────────────────────────────────────────────────────────
# 품질 점수 (프로드 규칙 재현) · 지표
# ──────────────────────────────────────────────────────────────────────────


def _geomean(values: list[float | None]) -> float | None:
    xs = [float(v) for v in values if v is not None and v > 0]
    if not xs:
        return None
    return math.exp(sum(math.log(v) for v in xs) / len(xs))


def _quality_score(lines: list[dict]) -> tuple[float | None, dict]:
    """(quality_score, 커버리지 근거) — worker.py의 산출 규칙과 같다.

    라인 conf의 산술평균이고, **정렬이 성립한 줄만** 분모에 든다. 그래서 40줄 중 2줄만
    정렬돼도 그 2줄 평균이 곡 점수로 올라가 실패가 감춰진다 — 커버리지가 하한 미만이면
    확정 저신뢰(0.0)로 덮고 원래 측정값은 근거에 남긴다.
    """
    confs = [ln["confidence"] for ln in lines if ln.get("confidence") is not None]
    measured = sum(1 for ln in lines if ln.get("confidence") is not None)
    total = len(lines)
    avg = sum(confs) / len(confs) if confs else None
    ratio = (measured / total) if total else 0.0
    meta = {
        "aligned_lines": measured,
        "total_lines": total,
        "ratio": round(ratio, 4),
        "measured_conf": None if avg is None else round(avg, 6),
    }
    if total and ratio >= COVERAGE_MIN:
        return (None if avg is None else round(avg, 6)), meta
    meta["failed"] = True
    return FAILED_ALIGNMENT_QUALITY, meta


_PUNCT_CATS = {"P", "S", "Z", "C"}


def _pair_key(text: str) -> str:
    """줄 짝짓기용 정규화 키 — 공백·구두점·대소문자 차이를 지운다.

    후보 줄은 원 가사에서 만들고 기준선 줄은 프로드가 내보낸 텍스트인데, 프로드는 병기 줄
    되붙이기·비가창 줄 제거를 하므로 줄 수가 어긋날 수 있다. 인덱스로 짝지으면 한 줄
    어긋난 순간 그 뒤 전체가 오차로 잡히므로 텍스트로 짝짓는다.
    """
    out = []
    for ch in unicodedata.normalize("NFKC", text or "").lower():
        if unicodedata.category(ch)[0] in _PUNCT_CATS:
            continue
        out.append(ch)
    return "".join(out)


def _pair_lines(ref_lines: list[dict], est_lines: list[dict]) -> list[tuple[int, int]]:
    """(기준 인덱스, 후보 인덱스) 짝 목록. 텍스트 정규화 키의 최장 공통 블록으로 맞춘다."""
    ref_keys = [_pair_key(x.get("text", "")) for x in ref_lines]
    est_keys = [_pair_key(x.get("text", "")) for x in est_lines]
    # autojunk는 후렴 반복처럼 자주 나오는 줄을 «잡음»으로 버려 짝을 놓친다 — 반드시 끈다
    matcher = difflib.SequenceMatcher(None, ref_keys, est_keys, autojunk=False)
    pairs: list[tuple[int, int]] = []
    for i, j, size in matcher.get_matching_blocks():
        for k in range(size):
            if not ref_keys[i + k]:  # 구두점만 있는 줄은 아무 줄과도 같아 보인다
                continue
            pairs.append((i + k, j + k))
    return pairs


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def compute_metrics(baseline_lines: list[dict], est_lines: list[dict]) -> dict:
    """라인 온셋 잔차 지표. 기준은 baseline의 라인 start, 부호는 (후보 - 기준)."""
    pairs = _pair_lines(baseline_lines, est_lines)
    deltas: list[float] = []
    ref_starts: list[float] = []
    est_starts: list[float] = []
    for ri, ei in pairs:
        rs = baseline_lines[ri].get("start")
        es = est_lines[ei].get("start")
        if rs is None or es is None:
            continue
        ref_starts.append(float(rs))
        est_starts.append(float(es))
        deltas.append(float(es) - float(rs))

    out: dict[str, Any] = {
        "baseline_lines": len(baseline_lines),
        "candidate_lines": len(est_lines),
        "line_count_mismatch": len(baseline_lines) != len(est_lines),
        "paired": len(deltas),
        "pair_rate": round(len(deltas) / len(baseline_lines), 4) if baseline_lines else 0.0,
    }
    if not deltas:
        out["error"] = "no_paired_lines"
        return out

    abs_d = [abs(d) for d in deltas]
    out.update(
        {
            "mae": round(statistics.median(abs_d), 3),  # median absolute error
            "aae": round(statistics.fmean(abs_d), 3),  # average absolute error
            f"pco_{PCO_WINDOW}": round(sum(1 for d in abs_d if d <= PCO_WINDOW) / len(abs_d), 4),
            "p95_abs": round(_percentile(abs_d, 95), 3),
            "max_abs": round(max(abs_d), 3),
            f"over_{LATE_THRESHOLD}s_ratio": round(
                sum(1 for d in abs_d if d > LATE_THRESHOLD) / len(abs_d), 4
            ),
            # 지각 비대칭(앞선 오차가 늦은 오차보다 거슬린다)을 보려면 부호가 필요하다
            "signed_median": round(statistics.median(deltas), 3),
        }
    )

    # mir_eval을 «자로» 쓴다 — 위 수치와 어긋나면 그 사실이 드러나야 하므로 따로 싣는다.
    # 단조 증가·동일 길이를 요구하므로 실패할 수 있다(짝지은 부분열은 대개 단조지만
    # 보간 줄이 섞이면 깨질 수 있다). 실패는 사유를 남기고 계속한다.
    try:
        import numpy as np

        import mir_eval.alignment as me_align

        r = np.asarray(ref_starts, dtype=np.float64)
        e = np.asarray(est_starts, dtype=np.float64)
        mae, aae = me_align.absolute_error(r, e)
        out["mir_eval"] = {
            "mae": round(float(mae), 3),
            "aae": round(float(aae), 3),
            "percentage_correct": round(
                float(me_align.percentage_correct(r, e, window=PCO_WINDOW)), 4
            ),
            "percentage_correct_segments": round(
                float(me_align.percentage_correct_segments(r, e)), 4
            ),
            "karaoke_perceptual": round(float(me_align.karaoke_perceptual_metric(r, e)), 4),
        }
    except Exception as exc:
        out["mir_eval_error"] = repr(exc)[:200]

    # ── 음절(글자) 단위 지표 — 카라오케 품질의 실제 축 ──────────────
    # 기준: baseline 세그의 words(프로드 실측 글자 스팬), 후보: est 라인의 segs.
    # 라인 짝 안에서 글자 텍스트 키의 LCS로 다시 짝지어 온셋 잔차를 모은다. 둘 중 하나라도
    # 세그가 없으면 그 라인은 건너뛴다(구식 run json은 segs가 없어 라인 지표만 남는다).
    syl_deltas: list[float] = []
    for ri, ei in pairs:
        ref_segs = baseline_lines[ri].get("words") or []
        est_segs = est_lines[ei].get("segs") or []
        if not ref_segs or not est_segs:
            continue
        rkeys = [_pair_key(str(x.get("word") or x.get("t") or "")) for x in ref_segs]
        ekeys = [_pair_key(str(x.get("t") or "")) for x in est_segs]
        sm = difflib.SequenceMatcher(None, rkeys, ekeys, autojunk=False)
        for i, j, size in sm.get_matching_blocks():
            for k in range(size):
                if not rkeys[i + k]:
                    continue
                rs = ref_segs[i + k].get("start")
                es = est_segs[j + k].get("start")
                if rs is None or es is None:
                    continue
                syl_deltas.append(float(es) - float(rs))
    if syl_deltas:
        syl_abs = sorted(abs(d) for d in syl_deltas)
        out.update(
            {
                "syl_paired": len(syl_deltas),
                "syl_mae": round(statistics.median(syl_abs), 3),
                "syl_pco": round(sum(1 for a in syl_abs if a <= PCO_WINDOW) / len(syl_abs), 4),
                "syl_p95": round(_percentile(syl_abs, 95), 3),
            }
        )
    return out


def collapse_flags(run: dict) -> list[str]:
    """붕괴 게이트 — 「돌았지만 못 쓴다」를 수치 하나로 감추지 않기 위한 라벨들."""
    flags: list[str] = []
    if run.get("error"):
        return ["error"]
    q = run.get("quality_score")
    coverage = ((run.get("align_meta") or {}).get("coverage")) or {}
    if coverage.get("failed"):
        flags.append("coverage")
    if q is None:
        flags.append("no_quality")
    elif q < LOW_CONF_THRESHOLD:
        flags.append("low_conf")
    metrics = run.get("metrics") or {}
    if metrics.get("error"):
        flags.append("unpaired")
    elif metrics.get("pair_rate", 1.0) < 0.5:
        flags.append("low_pair_rate")
    if metrics.get("line_count_mismatch"):
        flags.append("line_count")
    return flags


# ──────────────────────────────────────────────────────────────────────────
# 산출물 (SRT · diff HTML · summary.md)
# ──────────────────────────────────────────────────────────────────────────

_FORBIDDEN_FS = '<>:"/\\|?*'


def slugify(title: str, limit: int = 40) -> str:
    """파일명 안전화 — 한글·일본어는 유지하고 금지문자·제어문자만 걷어낸다."""
    out = []
    for ch in (title or "").strip():
        if ch in _FORBIDDEN_FS or unicodedata.category(ch)[0] == "C":
            out.append("-")
        else:
            out.append(ch)
    slug = "".join(out).strip()
    slug = " ".join(slug.split())[:limit].strip(" .-")
    return slug or "untitled"


def _srt_time(seconds: float) -> str:
    total_ms = max(0, int(round(float(seconds) * 1000)))
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(path: Path, lines: list[dict]) -> int:
    """라인 목록 → SRT. start/end가 없는 줄은 건너뛰고 개수만 돌려준다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    blocks: list[str] = []
    idx = 0
    for ln in lines:
        start, end = ln.get("start"), ln.get("end")
        if start is None:
            continue
        if end is None or float(end) <= float(start):
            end = float(start) + 0.5
        idx += 1
        text = (ln.get("text") or "").strip() or "　"
        blocks.append(f"{idx}\n{_srt_time(start)} --> {_srt_time(end)}\n{text}\n")
    path.write_text("\n".join(blocks), encoding="utf-8")
    return idx


_DIFF_CSS = """
body{font:14px/1.5 -apple-system,'Segoe UI',Meiryo,'Malgun Gothic',sans-serif;margin:24px;color:#111}
h1{font-size:18px;margin:0 0 4px}h2{font-size:14px;margin:18px 0 6px;color:#444}
.meta{color:#555;margin-bottom:12px}
.meta b{color:#111}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid #e5e5e5;padding:4px 8px;text-align:left;vertical-align:top}
th{background:#fafafa;position:sticky;top:0}
td.num{text-align:right;white-space:nowrap}
tr.late{background:#fff1f0}
tr.miss{background:#f5f5f5;color:#888}
.flag{display:inline-block;background:#ffe8a3;border-radius:3px;padding:1px 6px;margin-right:4px}
"""


def write_diff_html(
    path: Path,
    song: dict,
    aligner: str,
    separator: str,
    baseline_lines: list[dict],
    est_lines: list[dict],
    metrics: dict,
    run: dict,
) -> None:
    """라인 표: text | baseline start | candidate start | Δ. |Δ|>0.3s 행은 하이라이트."""
    pair_map = dict(_pair_lines(baseline_lines, est_lines))
    flags = collapse_flags(run)
    head_bits = [
        f"<b>{html.escape(song.get('title') or '')}</b> — {html.escape(song.get('artist') or '')}",
        f"video_id <b>{html.escape(song['video_id'])}</b>",
        f"stratum <b>{html.escape(str(song.get('stratum')))}</b>",
        f"separator <b>{html.escape(separator)}</b> / aligner <b>{html.escape(aligner)}</b>",
    ]
    metric_bits = []
    for key in ("mae", "aae", f"pco_{PCO_WINDOW}", "p95_abs", "max_abs", f"over_{LATE_THRESHOLD}s_ratio", "signed_median"):
        if key in metrics:
            metric_bits.append(f"{key} <b>{metrics[key]}</b>")
    metric_bits.append(f"paired <b>{metrics.get('paired', 0)}</b>/{metrics.get('baseline_lines', 0)}")
    metric_bits.append(f"quality_score <b>{run.get('quality_score')}</b>")

    rows: list[str] = []
    for ri, ref in enumerate(baseline_lines):
        ei = pair_map.get(ri)
        est = est_lines[ei] if ei is not None else None
        rs = ref.get("start")
        es = est.get("start") if est else None
        cls = "miss"
        delta_txt = "—"
        if rs is not None and es is not None:
            delta = float(es) - float(rs)
            delta_txt = f"{delta:+.2f}"
            cls = "late" if abs(delta) > LATE_THRESHOLD else ""
        rows.append(
            "<tr class='{cls}'><td class='num'>{i}</td><td>{text}</td>"
            "<td class='num'>{rs}</td><td class='num'>{es}</td><td class='num'>{d}</td>"
            "<td>{cand}</td></tr>".format(
                cls=cls,
                i=ri + 1,
                text=html.escape(ref.get("text") or ""),
                rs="—" if rs is None else f"{float(rs):.2f}",
                es="—" if es is None else f"{float(es):.2f}",
                d=delta_txt,
                cand=html.escape((est or {}).get("text") or ""),
            )
        )

    doc = (
        "<!doctype html><html lang='ko'><head><meta charset='utf-8'>"
        f"<title>{html.escape(song['video_id'])} {html.escape(aligner)} vs baseline</title>"
        f"<style>{_DIFF_CSS}</style></head><body>"
        f"<h1>{' · '.join(head_bits)}</h1>"
        f"<div class='meta'>{' · '.join(metric_bits)}</div>"
        + (
            "<div class='meta'>"
            + "".join(f"<span class='flag'>{html.escape(f)}</span>" for f in flags)
            + "</div>"
            if flags
            else ""
        )
        + "<div class='meta'>기준선은 프로드 현행 스택의 라인 시각이다 — 검증된 정답이 아니라 "
        "공통 비교 기준이다. Δ가 크다고 후보가 나쁜 것도, 작다고 좋은 것도 아니므로 "
        "스템 wav와 함께 들어야 한다.</div>"
        "<table><thead><tr><th>#</th><th>baseline text</th><th>baseline start</th>"
        "<th>candidate start</th><th>Δ(s)</th><th>candidate text</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></body></html>"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc, encoding="utf-8")


def _link_or_copy(src: Path, dst: Path) -> None:
    """검수 트리에 스템·원본을 놓는다. 같은 볼륨이면 하드링크(공짜), 아니면 복사."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_song_summary(path: Path, song: dict, runs: list[dict]) -> None:
    lines = [
        f"# {song.get('title') or song['video_id']}",
        "",
        f"- video_id: `{song['video_id']}`",
        f"- artist: {song.get('artist') or '-'}",
        f"- language(eval_set): `{song.get('language')}` → 정렬 입력 `{base_language(song.get('language'))}`",
        f"- stratum: `{song.get('stratum')}`",
        f"- duration_est: {song.get('duration_est')}s / 가사 {song.get('line_count')}줄",
        f"- 프로드 baseline quality_score: {song.get('quality_score')}",
        "",
        "## 후보 조합",
        "",
        "| separator | aligner | run | MAE | AAE | PCO@0.3 | P95 | max | >0.3s | signed med | quality | 붕괴 | sep s | align s |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for run in runs:
        m = run.get("metrics") or {}
        flags = collapse_flags(run)
        lines.append(
            "| {sep} | {aln} | r{idx} | {mae} | {aae} | {pco} | {p95} | {mx} | {over} | {sm} | {q} | {fl} | {ss} | {as_} |".format(
                sep=run["separator"],
                aln=run["aligner"],
                idx=run["run_idx"],
                mae=m.get("mae", "—"),
                aae=m.get("aae", "—"),
                pco=m.get(f"pco_{PCO_WINDOW}", "—"),
                p95=m.get("p95_abs", "—"),
                mx=m.get("max_abs", "—"),
                over=m.get(f"over_{LATE_THRESHOLD}s_ratio", "—"),
                sm=m.get("signed_median", "—"),
                q=run.get("quality_score", "—"),
                fl=" ".join(flags) or "-",
                ss=(run.get("separation") or {}).get("elapsed_sec", "—"),
                as_=run.get("align_elapsed_sec", "—"),
            )
        )
    lines += ["", "## 특이사항", ""]
    noted = False
    for run in runs:
        m = run.get("metrics") or {}
        bits = []
        if run.get("error"):
            bits.append(f"실패: `{run['error']}`")
        if m.get("line_count_mismatch"):
            bits.append(
                f"라인 수 불일치 (기준 {m.get('baseline_lines')} vs 후보 {m.get('candidate_lines')}, "
                f"짝지어진 줄 {m.get('paired')})"
            )
        cov = ((run.get("align_meta") or {}).get("coverage")) or {}
        if cov.get("failed"):
            bits.append(
                f"정렬 커버리지 붕괴: 실측 타이밍이 있는 줄 {cov.get('aligned_lines')}/"
                f"{cov.get('total_lines')} (측정 conf {cov.get('measured_conf')})"
            )
        q = run.get("quality_score")
        if q is not None and q < LOW_CONF_THRESHOLD and not cov.get("failed"):
            bits.append(f"저신뢰 quality_score {q} (확장 경고 임계 {LOW_CONF_THRESHOLD} 미만)")
        if m.get("mir_eval_error"):
            bits.append(f"mir_eval 실패 → 자체 지표만: `{m['mir_eval_error']}`")
        if (run.get("separation") or {}).get("note"):
            bits.append(f"분리 주의: {run['separation']['note']}")
        if bits:
            noted = True
            lines.append(f"- **{run['separator']} × {run['aligner']} r{run['run_idx']}**")
            lines += [f"  - {b}" for b in bits]
    if not noted:
        lines.append("- 없음")
    lines += [
        "",
        "## 읽는 법",
        "",
        "기준선(`align/baseline-prod.srt`)은 프로드 현행 스택의 출력이다. 사람이 검증한 정답이",
        "아니므로 Δ는 «다름»이지 «틀림»이 아니다. 판단은 `audio/<separator>/vocals.wav`를 듣고",
        "diff HTML의 큰 Δ 줄을 실제로 확인해서 내린다.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────────
# 집계 리포트
# ──────────────────────────────────────────────────────────────────────────


def _median_or_none(values: list[float]) -> float | None:
    xs = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return round(statistics.median(xs), 3) if xs else None


def build_report(bench_dir: Path, eval_set: dict) -> str:
    songs = {s["video_id"]: s for s in eval_set["songs"]}
    runs_root = bench_dir / "runs"
    runs: list[dict] = []
    for combo_dir in sorted(runs_root.glob("*__*")):
        for run_path in sorted(combo_dir.glob("*.json")):
            try:
                runs.append(json.loads(run_path.read_text(encoding="utf-8")))
            except Exception:
                continue

    out = [
        "# 모델 교체 벤치 리포트",
        "",
        f"- 생성: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- runs 캐시: `{runs_root.as_posix()}` ({len(runs)}건)",
        f"- 평가 세트: {len(songs)}곡",
        "",
        "## 기준(reference)에 대하여",
        "",
        "지표의 기준은 **프로드 현행 스택이 낸 라인 시각**(`eval_set.json`의",
        "`baseline_timestamps`)이다. 사람이 검증한 정답이 아니다. 기준 자체가 밀려 있는 곡에서는",
        "후보의 MAE가 커지는 것이 오히려 개선일 수 있다. 이 표는 «어느 후보가 현행과 얼마나",
        "다른가»와 «붕괴하지 않는가»를 재는 자이고, 좋아졌는지는 검수 산출물(스템 wav·SRT·diff",
        "HTML)로 판단한다.",
        "",
        "또한 정렬 후보는 워커 보정층(자막 앵커·독음 이중정렬·심판·VAD 클램프·붕괴 재합성) 없이",
        "맨몸으로 돈다 — 후보끼리는 같은 조건이라 공정하지만, 기준선과의 절대 잔차에는 «보정층",
        "부재»가 섞여 있다.",
        "",
    ]

    if not runs:
        out += ["## 결과", "", "runs 캐시가 비어 있다 — 아직 실행된 조합이 없다.", ""]
        return "\n".join(out)

    combos: dict[tuple[str, str], list[dict]] = {}
    for run in runs:
        combos.setdefault((run["separator"], run["aligner"]), []).append(run)

    metric_cols = [
        ("mae", "MAE"),
        ("aae", "AAE"),
        (f"pco_{PCO_WINDOW}", "PCO@0.3"),
        ("p95_abs", "P95"),
        ("max_abs", "max"),
        (f"over_{LATE_THRESHOLD}s_ratio", ">0.3s"),
        ("signed_median", "signed med"),
    ]

    for (sep, aln), combo_runs in sorted(combos.items()):
        out += [f"## {sep} × {aln}", ""]
        by_stratum: dict[str, list[dict]] = {}
        for run in combo_runs:
            stratum = base_language(run.get("stratum") or run.get("language") or "?")
            by_stratum.setdefault(stratum, []).append(run)

        header = "| stratum | 곡 | " + " | ".join(label for _, label in metric_cols)
        header += " | quality med | 붕괴 곡 | sep s med | align s med |"
        out += [
            header,
            "|---|---|" + "---|" * (len(metric_cols) + 4),
        ]
        for stratum in sorted(by_stratum) + ["ALL"]:
            group = combo_runs if stratum == "ALL" else by_stratum[stratum]
            ok = [r for r in group if not r.get("error") and (r.get("metrics") or {}).get("mae") is not None]
            cells = [
                _median_or_none([(r.get("metrics") or {}).get(key) for r in ok])
                for key, _ in metric_cols
            ]
            collapsed = sorted({r["video_id"] for r in group if collapse_flags(r)})
            row = f"| {stratum} | {len({r['video_id'] for r in group})} | "
            row += " | ".join("—" if c is None else str(c) for c in cells)
            row += " | {q} | {c} | {ss} | {as_} |".format(
                q=_median_or_none([r.get("quality_score") for r in group]),
                c=len(collapsed),
                ss=_median_or_none([(r.get("separation") or {}).get("elapsed_sec") for r in group]),
                as_=_median_or_none([r.get("align_elapsed_sec") for r in group]),
            )
            out.append(row)
        out.append("")

        vram = [
            r.get("align_vram_peak_mb")
            for r in combo_runs
            if r.get("align_vram_peak_mb") is not None
        ]
        dev_vram = [
            (r.get("separation") or {}).get("vram_device_peak_mb")
            for r in combo_runs
            if (r.get("separation") or {}).get("vram_device_peak_mb") is not None
        ]
        out += [
            "- 각 칸은 **곡별 값의 중앙값**이다 — 붕괴 곡도 분모에 들어간다(중앙값이라 꼬리에 "
            "지배되지는 않지만, 붕괴 곡 수를 함께 보지 않으면 해석이 틀린다).",
            "- 정렬 시간에는 모델 로드가 빠져 있다(run json의 `align_meta.model_load_sec`로 별도 기록).",
            "- VRAM 피크(정렬, torch allocator) 중앙값: "
            + (f"{_median_or_none(vram)} MB" if vram else "—"),
            "- VRAM 피크(분리, 디바이스 전체 샘플링) 중앙값: "
            + (f"{_median_or_none(dev_vram)} MB" if dev_vram else "—"),
            "",
            "### 곡별",
            "",
            "| stratum | video_id | 제목 | MAE | AAE | PCO@0.3 | P95 | max | >0.3s | 짝 | quality | 붕괴 |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for run in sorted(
            combo_runs,
            key=lambda r: (
                base_language(r.get("stratum") or ""),
                -((r.get("metrics") or {}).get("mae") or -1),
            ),
        ):
            m = run.get("metrics") or {}
            song = songs.get(run["video_id"], {})
            title = (song.get("title") or "")[:32]
            out.append(
                "| {st} | `{vid}` | {title} | {mae} | {aae} | {pco} | {p95} | {mx} | {over} | {pr} | {q} | {fl} |".format(
                    st=base_language(run.get("stratum") or "?"),
                    vid=run["video_id"],
                    title=title,
                    mae=m.get("mae", "—"),
                    aae=m.get("aae", "—"),
                    pco=m.get(f"pco_{PCO_WINDOW}", "—"),
                    p95=m.get("p95_abs", "—"),
                    mx=m.get("max_abs", "—"),
                    over=m.get(f"over_{LATE_THRESHOLD}s_ratio", "—"),
                    pr=f"{m.get('paired', 0)}/{m.get('baseline_lines', 0)}",
                    q=run.get("quality_score", "—"),
                    fl=" ".join(collapse_flags(run)) or "-",
                )
            )
        out.append("")

    failures = [r for r in runs if r.get("error")]
    if failures:
        out += ["## 실패한 실행", ""]
        for run in failures:
            out.append(
                f"- `{run['video_id']}` {run['separator']} × {run['aligner']} "
                f"r{run['run_idx']}: `{run['error']}`"
            )
        out.append("")

    out += [
        "## 검수 산출물",
        "",
        "`benchmark/results/<stratum>/<video_id>__<제목>/` 아래에 원본·스템 wav, 후보별 SRT,",
        "기준선 SRT, diff HTML, 곡별 summary.md가 있다. 수치가 같아도 스템이 다르게 들릴 수",
        "있으므로 후보 채택 판단에는 둘 다 필요하다.",
        "",
    ]
    return "\n".join(out)


# ──────────────────────────────────────────────────────────────────────────
# 스윕 실행
# ──────────────────────────────────────────────────────────────────────────


def base_language(language: str | None) -> str:
    """eval_set의 `ja_mms` 같은 라벨에서 `_mms` 접미를 벗긴 기본 언어."""
    lang = (language or "").strip()
    return lang[: -len("_mms")] if lang.endswith("_mms") else lang


def baseline_segments(song: dict) -> list[dict]:
    bt = song.get("baseline_timestamps") or {}
    if isinstance(bt, list):
        return bt
    return bt.get("segments") or bt.get("timestamps") or []


def _pron_lines_from_segments(segs: list[dict], key: str) -> tuple[str, list[str]] | None:
    """baseline 세그먼트에서 (독음 입력 텍스트, 표시 텍스트 목록)을 짓는다.

    독음이 실린 세그가 80% 미만이면 None — 재현 불가로 보고 호출부가 다른 경로를 탄다."""
    pron_lines: list[str] = []
    display: list[str] = []
    have = 0
    for seg in segs:
        text = str(seg.get("text") or "").strip()
        pron = seg.get("pron") or {}
        line = str(pron.get(key) or "").strip()
        if not line and key == "hangul":
            line = str(seg.get("pronunciation") or "").strip()
        chosen = line or text
        if not chosen:
            continue
        if line:
            have += 1
        pron_lines.append(chosen)
        display.append(text or chosen)
    if not pron_lines or have / len(pron_lines) < 0.8:
        return None
    return "\n".join(pron_lines), display


def alignment_input(song: dict, mode: str = "auto") -> tuple[str | None, list[str] | None, str]:
    """프로드가 실제로 정렬한 입력 텍스트 재현 — (정렬 입력, 표시 텍스트 목록|None, 모드).

    ko(-경로)로 정렬된 곡의 가사가 한글이 아니면(일본어 원문 등) 프로드는 결정론 독음
    변환이 만든 **한글 독음**을 정렬했고, 그 산물이 baseline 세그먼트의 ``pron.hangul``이다.
    en 경로의 비라틴 가사도 같은 구조로 ``pron.romaji``다. 원문을 그대로 kor 어댑터에 넣으면
    vocab 커버리지가 0으로 무너져 «모델 성능»이 아니라 «입력 불일치»를 재게 된다 (기준선
    1차 실측에서 ko 층 18곡 중 10곡이 이 이유로 coverage 붕괴). 표시 텍스트 목록은 정렬
    결과 라인에 되씌워 기준선과의 텍스트 짝짓기(_pair_lines)를 살리는 용도다.
    독음이 실린 세그가 80% 미만이면 재현 불가로 보고 원문 그대로 간다.
    """
    lyrics = song.get("lyrics") or ""
    if mode == "pron-hangul":
        # 강제 독음 모드 — 프로드 재현이 아니라 «전 언어→한글 단일 경로» 아키텍처 탐색용.
        # 어떤 언어 곡이든 baseline의 pron.hangul(당시 심판 산출물)로 정렬한다. 미보유 곡은
        # None을 돌려 호출부가 건너뛴다.
        built = _pron_lines_from_segments(baseline_segments(song), "hangul")
        if built is None:
            return None, None, "pron_hangul_unavailable"
        return built[0], built[1], "forced_pron_hangul"

    if mode == "pron-hangul-local":
        # 리포 결정론 음차(latin_hangul·kana_hangul)로 독음을 직접 생성 — baseline에
        # 독음 산물이 없는 곡(en romaji 경로 곡·자막 기준선 곡)의 «전 언어→한글» 실측용.
        # 프로드 LLM 발음(심판) 경로와 달리 결정론 후보라 품질 하한 실측에 해당한다.
        import re
        import sys as _sys

        repo = str(Path(__file__).resolve().parents[1])
        if repo not in _sys.path:
            _sys.path.insert(0, repo)
        from everyric2.text.kana_hangul import finalize_pronunciation
        from everyric2.text.latin_hangul import transliterate_latin

        ja_re = re.compile(r"[぀-ヿㇰ-ㇿ㐀-鿿豈-﫿]")

        def _line_to_hangul(line: str) -> str:
            out: list[str] = []
            buf: list[str] = []

            def flush() -> None:
                if buf:
                    chunk = "".join(buf)
                    if ja_re.search(chunk):
                        try:
                            chunk = finalize_pronunciation(chunk) or chunk
                        except Exception:
                            pass
                    out.append(chunk)
                    buf.clear()

            for ch in line:
                if "가" <= ch <= "힣" or ch.isspace():
                    flush()
                    out.append(ch)
                else:
                    buf.append(ch)
            flush()
            return transliterate_latin("".join(out))

        display = [l for l in lyrics.splitlines() if l.strip()]
        if not display:
            return None, None, "pron_hangul_local_unavailable"
        return "\n".join(_line_to_hangul(l) for l in display), display, "forced_pron_hangul_local"

    lang = base_language(song.get("language"))
    key = {"ko": "hangul", "en": "romaji"}.get(lang)
    if not key:
        return lyrics, None, "raw"
    letters = [ch for ch in lyrics if ch.isalpha()]
    if key == "hangul":
        native = sum(1 for ch in letters if "가" <= ch <= "힣")
    else:
        native = sum(1 for ch in letters if ch.isascii())
    if not letters or native / len(letters) >= 0.5:
        return lyrics, None, "raw"
    built = _pron_lines_from_segments(baseline_segments(song), key)
    if built is None:
        return lyrics, None, "raw"
    return built[0], built[1], f"pron_{key}"


def load_eval_set(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        data = {"songs": data, "count": len(data)}
    return data


def select_songs(eval_set: dict, songs_arg: str, strata_arg: str | None) -> list[dict]:
    songs = eval_set["songs"]
    if songs_arg and songs_arg != "all":
        wanted = [v.strip() for v in songs_arg.split(",") if v.strip()]
        index = {s["video_id"]: s for s in songs}
        missing = [v for v in wanted if v not in index]
        if missing:
            raise SystemExit(f"eval_set에 없는 video_id: {', '.join(missing)}")
        songs = [index[v] for v in wanted]
    if strata_arg:
        wanted_strata = {s.strip() for s in strata_arg.split(",") if s.strip()}
        songs = [
            s
            for s in songs
            if s.get("stratum") in wanted_strata
            or base_language(s.get("stratum")) in wanted_strata
        ]
    return songs


def ensure_stems(
    adapter: SeparatorAdapter, song: dict, audio_path: Path, bench_dir: Path, force: bool
) -> SeparationOut:
    """스템 디스크 캐시 — `(separator, video_id)` 키. 정렬 후보들이 이걸 재사용한다."""
    stem_dir = bench_dir / "stems" / fs_name(adapter.name) / song["video_id"]
    vocals, inst = stem_dir / "vocals.wav", stem_dir / "inst.wav"
    meta_path = stem_dir / "meta.json"
    if not force and vocals.exists() and inst.exists():
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        return SeparationOut(
            vocals_path=vocals,
            inst_path=inst,
            elapsed_sec=meta.get("elapsed_sec"),
            vram_peak_mb=meta.get("vram_peak_mb"),
            vram_device_peak_mb=meta.get("vram_device_peak_mb"),
            cached=True,
            note=meta.get("note"),
        )

    shutil.rmtree(stem_dir, ignore_errors=True)
    out = adapter.separate(audio_path, stem_dir)
    if out.vocals_path != vocals:
        shutil.move(str(out.vocals_path), str(vocals))
        out.vocals_path = vocals
    if out.inst_path != inst:
        shutil.move(str(out.inst_path), str(inst))
        out.inst_path = inst
    meta_path.write_text(json.dumps(out.meta(), ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def fs_name(name: str) -> str:
    """어댑터 이름을 파일시스템 안전하게 — Windows는 `:`를 경로에 못 쓴다(hf:kkonjeong 실측).

    표시·CLI·run json에는 원래 이름을 유지하고 디렉터리·파일명에만 이걸 쓴다."""
    return name.replace(":", "-")


def run_path_for(bench_dir: Path, sep: str, aln: str, video_id: str, run_idx: int) -> Path:
    return bench_dir / "runs" / f"{fs_name(sep)}__{fs_name(aln)}" / f"{video_id}__r{run_idx}.json"


def artifacts_for_song(bench_dir: Path, song: dict) -> Path:
    return (
        bench_dir
        / "results"
        / base_language(song.get("stratum") or song.get("language") or "unknown")
        / f"{song['video_id']}__{slugify(song.get('title') or '')}"
    )


def sweep(args: argparse.Namespace) -> int:
    bench_dir = Path(args.bench_dir).resolve()
    eval_set = load_eval_set(Path(args.eval_set))
    songs = select_songs(eval_set, args.songs, args.strata)

    sep_names = [s.strip() for s in args.separators.split(",") if s.strip()]
    aln_names = [a.strip() for a in args.aligners.split(",") if a.strip()]
    for name in sep_names:
        if name not in SEPARATORS:
            raise SystemExit(f"미등록 분리 후보 '{name}' (등록됨: {', '.join(SEPARATORS)})")
    for name in aln_names:
        if name not in ALIGNERS:
            raise SystemExit(f"미등록 정렬 후보 '{name}' (등록됨: {', '.join(ALIGNERS)})")

    separators = {name: SEPARATORS[name]() for name in sep_names}
    aligners = {name: ALIGNERS[name]() for name in aln_names}

    audio_dir = Path(args.audio_dir) if args.audio_dir else bench_dir / "audio"
    print(f"벤치 디렉터리: {bench_dir}")
    print(f"오디오: {audio_dir}")
    print(f"곡 {len(songs)} × 분리 {len(sep_names)} × 정렬 {len(aln_names)} × 반복 {args.repeat}")

    executed = skipped = failed = 0
    for pos, song in enumerate(songs, start=1):
        video_id = song["video_id"]
        audio_path = next(
            (p for p in (audio_dir / f"{video_id}{ext}" for ext in (".m4a", ".wav", ".mp3", ".opus", ".webm")) if p.exists()),
            None,
        )
        label = f"[{pos}/{len(songs)}] {video_id} {(song.get('title') or '')[:28]}"
        if audio_path is None:
            print(f"{label}: 오디오 없음 → 건너뜀", flush=True)
            skipped += 1
            continue

        language = base_language(song.get("language"))
        baseline = baseline_segments(song)
        align_lyrics, display_texts, align_input_mode = alignment_input(song, args.input_mode)
        if align_lyrics is None:
            print(f"{label}: 강제 독음 입력 불가(pron.hangul 미보유) → 건너뜀", flush=True)
            skipped += 1
            continue
        if args.input_mode in ("pron-hangul", "pron-hangul-local"):
            # 독음은 한글이므로 어댑터에는 ko로 정렬시킨다. 러너 라벨에 @hangul을 붙여
            # 같은 정렬기의 raw 입력 런과 캐시·리포트가 절대 섞이지 않게 한다.
            language = "ko"
        aln_suffix = {"pron-hangul": "@hangul", "pron-hangul-local": "@hangul-local"}.get(
            args.input_mode, ""
        )
        song_runs: list[dict] = []

        for sep_name, sep_adapter in separators.items():
            try:
                separation = ensure_stems(sep_adapter, song, audio_path, bench_dir, args.force_separate)
            except Exception as exc:
                print(f"{label} {sep_name}: 분리 실패 {exc!r}", flush=True)
                failed += 1
                for aln_name in aln_names:
                    for run_idx in range(1, args.repeat + 1):
                        run = {
                            "video_id": video_id,
                            "separator": sep_name,
                            "aligner": aln_name,
                            "run_idx": run_idx,
                            "language": song.get("language"),
                            "align_language": language,
                            "stratum": song.get("stratum"),
                            "error": f"separation failed: {exc!r}"[:400],
                        }
                        _write_run(bench_dir, run)
                        song_runs.append(run)
                continue

            print(
                f"{label} {sep_name}: 스템 {'캐시' if separation.cached else f'{separation.elapsed_sec}s'}",
                flush=True,
            )

            for aln_name, aln_adapter in aligners.items():
                aln_label = f"{aln_name}{aln_suffix}"
                for run_idx in range(1, args.repeat + 1):
                    out_path = run_path_for(bench_dir, sep_name, aln_label, video_id, run_idx)
                    if out_path.exists() and not args.force:
                        try:
                            cached = json.loads(out_path.read_text(encoding="utf-8"))
                        except Exception:
                            cached = None
                        # 실패로 기록된 실행은 다시 시도한다 — 캐시가 실패를 굳히면 안 된다
                        # --refresh-segs: 음절 세그가 없는 구식 런은 캐시로 안 치고 다시 정렬
                        stale_segs = args.refresh_segs and cached and not any(
                            l.get("segs") for l in (cached.get("lines") or [])
                        )
                        if cached and not cached.get("error") and not stale_segs:
                            song_runs.append(cached)
                            skipped += 1
                            print(f"{label} {sep_name}×{aln_label} r{run_idx}: 캐시 재사용", flush=True)
                            continue

                    run: dict[str, Any] = {
                        "video_id": video_id,
                        "title": song.get("title"),
                        "separator": sep_name,
                        "aligner": aln_label,
                        "run_idx": run_idx,
                        "language": song.get("language"),
                        "align_language": language,
                        "stratum": song.get("stratum"),
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "separation": {**separation.meta(), "cached": separation.cached},
                    }
                    try:
                        run["align_input"] = align_input_mode
                        align_out = aln_adapter.align(separation.vocals_path, align_lyrics, language)
                        if display_texts:
                            if len(align_out.lines) == len(display_texts):
                                # 정렬은 독음으로 했지만 표시·짝짓기는 원문으로 — 프로드와 동일 구조
                                for ln, disp in zip(align_out.lines, display_texts):
                                    ln["aligned_text"] = ln["text"]
                                    ln["text"] = disp
                            else:
                                run["align_input_line_mismatch"] = (
                                    f"{len(align_out.lines)} != {len(display_texts)}"
                                )
                        run.update(
                            {
                                "align_elapsed_sec": align_out.elapsed_sec,
                                "align_vram_peak_mb": align_out.vram_peak_mb,
                                "align_vram_device_peak_mb": align_out.vram_device_peak_mb,
                                "quality_score": align_out.quality_score,
                                "align_meta": align_out.meta,
                                "lines": align_out.lines,
                                "metrics": compute_metrics(baseline, align_out.lines),
                            }
                        )
                        m = run["metrics"]
                        print(
                            f"{label} {sep_name}×{aln_label} r{run_idx}: "
                            f"{align_out.elapsed_sec}s MAE={m.get('mae')} AAE={m.get('aae')} "
                            f"PCO={m.get(f'pco_{PCO_WINDOW}')} q={align_out.quality_score} "
                            f"{' '.join(collapse_flags(run)) or 'ok'}",
                            flush=True,
                        )
                        executed += 1
                    except Exception as exc:
                        run["error"] = repr(exc)[:400]
                        failed += 1
                        print(f"{label} {sep_name}×{aln_label} r{run_idx}: 실패 {exc!r}", flush=True)
                        _free_cuda()
                    _write_run(bench_dir, run)
                    song_runs.append(run)

        if not args.no_artifacts and song_runs:
            try:
                write_artifacts(bench_dir, song, song_runs, audio_path, separators, baseline)
            except Exception as exc:
                print(f"{label}: 검수 산출물 생성 실패 {exc!r}", flush=True)

    report = build_report(bench_dir, eval_set)
    (bench_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(f"\n실행 {executed} · 캐시/건너뜀 {skipped} · 실패 {failed}")
    print(f"리포트: {(bench_dir / 'REPORT.md')}")
    return 0 if executed or skipped else 1


def _write_run(bench_dir: Path, run: dict) -> None:
    path = run_path_for(bench_dir, run["separator"], run["aligner"], run["video_id"], run["run_idx"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")


def _free_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def write_artifacts(
    bench_dir: Path,
    song: dict,
    song_runs: list[dict],
    audio_path: Path,
    separators: dict[str, SeparatorAdapter],
    baseline: list[dict],
) -> None:
    """사용자 검수 트리 — 수치만큼 중요한 산출물이다(듣고 볼 수 있어야 한다)."""
    root = artifacts_for_song(bench_dir, song)
    (root / "audio").mkdir(parents=True, exist_ok=True)
    _link_or_copy(audio_path, root / "audio" / f"original{audio_path.suffix}")

    for sep_name in separators:
        stem_dir = bench_dir / "stems" / fs_name(sep_name) / song["video_id"]
        for stem in ("vocals.wav", "inst.wav"):
            src = stem_dir / stem
            if src.exists():
                _link_or_copy(src, root / "audio" / fs_name(sep_name) / stem)

    write_srt(root / "align" / "baseline-prod.srt", baseline)
    for run in song_runs:
        if run.get("error") or not run.get("lines"):
            continue
        # 반복 실행은 첫 회(r1)만 검수 산출물로 남긴다 — 나머지는 runs json에 있다
        if run["run_idx"] != 1:
            continue
        stem = fs_name(f"{run['aligner']}__{run['separator']}")
        write_srt(root / "align" / f"{stem}.srt", run["lines"])
        write_diff_html(
            root / "align" / f"diff__{stem}__vs__baseline.html",
            song,
            run["aligner"],
            run["separator"],
            baseline,
            run["lines"],
            run.get("metrics") or {},
            run,
        )
    write_song_summary(root / "summary.md", song, song_runs)


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass

    _register_optional_aligners()

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--separators",
        default="htdemucs",
        help=f"쉼표 구분 분리 후보 (등록됨: {', '.join(SEPARATORS)})",
    )
    parser.add_argument(
        "--aligners",
        default="mms-baseline",
        help=f"쉼표 구분 정렬 후보 (등록됨: {', '.join(ALIGNERS)})",
    )
    parser.add_argument(
        "--input-mode",
        choices=("auto", "pron-hangul", "pron-hangul-local"),
        default="auto",
        help="auto: 프로드가 정렬한 입력 재현(기본). pron-hangul: 모든 곡을 baseline의 한글 "
        "독음(pron.hangul)으로 강제 정렬 — «전 언어→한글 단일 경로» 아키텍처 탐색용. "
        "러너 라벨에 @hangul이 붙어 raw 런과 캐시·리포트가 분리된다.",
    )
    parser.add_argument("--songs", default="all", help="video_id 쉼표 구분 또는 all")
    parser.add_argument("--strata", default=None, help="stratum 필터 (예: ja,ko — `_mms` 변형 포함)")
    parser.add_argument("--repeat", type=int, default=1, help="같은 조합 반복 횟수")
    parser.add_argument("--bench-dir", default=str(DEFAULT_BENCH_DIR))
    parser.add_argument("--eval-set", default=str(DEFAULT_BENCH_DIR / "eval_set.json"))
    parser.add_argument("--audio-dir", default=None, help="기본값: <bench-dir>/audio")
    parser.add_argument("--force", action="store_true", help="runs 캐시를 무시하고 다시 정렬")
    parser.add_argument(
        "--refresh-segs",
        action="store_true",
        help="음절 세그(lines[].segs)가 없는 구식 런만 다시 정렬 (있으면 캐시 재사용)",
    )
    parser.add_argument("--force-separate", action="store_true", help="스템 캐시를 무시하고 다시 분리")
    parser.add_argument("--no-artifacts", action="store_true", help="검수 트리 생성 생략")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="계산 없이 runs 캐시만으로 REPORT.md 재생성",
    )
    parser.add_argument("--list", action="store_true", help="평가 세트와 오디오 확보 상태만 출력")
    args = parser.parse_args()

    bench_dir = Path(args.bench_dir).resolve()
    if args.list:
        eval_set = load_eval_set(Path(args.eval_set))
        audio_dir = Path(args.audio_dir) if args.audio_dir else bench_dir / "audio"
        have = 0
        for song in select_songs(eval_set, args.songs, args.strata):
            exts = [
                ext
                for ext in (".m4a", ".wav", ".mp3", ".opus", ".webm")
                if (audio_dir / f"{song['video_id']}{ext}").exists()
            ]
            have += bool(exts)
            print(
                f"{song['video_id']}  {song.get('stratum'):<8} "
                f"{'AUDIO ' + exts[0] if exts else '-        '}  "
                f"{(song.get('title') or '')[:40]}"
            )
        print(f"\n오디오 확보: {have}곡")
        return 0

    if args.report_only:
        eval_set = load_eval_set(Path(args.eval_set))
        (bench_dir / "REPORT.md").parent.mkdir(parents=True, exist_ok=True)
        (bench_dir / "REPORT.md").write_text(build_report(bench_dir, eval_set), encoding="utf-8")
        print(f"리포트: {bench_dir / 'REPORT.md'}")
        return 0

    return sweep(args)


if __name__ == "__main__":
    raise SystemExit(main())
