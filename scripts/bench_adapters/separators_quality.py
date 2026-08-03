"""품질파(성능파) 보컬 분리 후보 어댑터 — BS-RoFormer 계열 3종 + fp16 변형.

근거 문서: ``docs/research/2026-07-31-acapella-quality-candidates.md`` Part 1.

``separators_roformer`` 와 같은 구조다. 무거운 의존성(audio-separator + CUDA torch)은
``benchmark/.venv-sep`` 에만 두고, 어댑터는 이 파일 자신을 그 인터프리터로 다시 실행해
분리를 시킨다. 메인 벤치 인터프리터는 건드리지 않는다.

등록 후보
---------

``bs-leap-xe``
    **BS-Roformer Leap Xe** (pcunwa, 2026-06-30 업로드).
    - 가중치: https://huggingface.co/pcunwa/BS-Roformer-Leap (``Xe/bs_leap_xe_voc.ckpt``)
    - 가중치 라이선스: **미확인** — HF 리포에 라이선스 태그가 없고 모델 카드도 비어 있다.
      명시적 NC 표기가 없어 실측 대상에는 포함하지만, 채택 시점에 저자 확인이 필요하다.
    - 코드 라이선스: 표준 ``bs_roformer`` 구조라 MSST(MIT) 구현으로 돌아간다.
    - SDR: MVSep multisong vocals **11.7577** (`unwa leap Xe`, 15위). 공개 가중치 중
      사실상 최상위이고, 기준선 HyperACE v2(11.3957) 대비 +0.36 dB.
    - config 실측: dim 256 / depth 16 / 90밴드 / ``inference.dim_t 1876``.

``bs-polarformer``
    **BS PolarFormer 오픈 가중치판** (ZFTurbo MSST v1.0.20 릴리스, fp16 체크포인트).
    - 가중치: https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/tag/v1.0.20
    - 가중치 라이선스: 미확인(저장소 릴리스 자산). 코드 라이선스: **MIT** (MSST).
    - SDR: multisong vocals **11.00** (저장소 보고). MVSep 서비스 전용 124밴드판
      (12.0230)과는 **다른 모델**이다 — 이쪽만 다운로드가 가능하다.
    - 아키텍처: BS-RoFormer + PoPE(극좌표 위치 임베딩, arXiv 2509.10534).
      ``use_pope: True`` 는 audio-separator 내장 BSRoformer가 모르는 인자라,
      MSST 원본 모델 코드(핀 커밋)와 ``PoPE-pytorch`` 를 워커에서 직접 쓴다.

``anvuew-ft1``
    **anvuew BS-RoFormer ft1** (2026-04-17).
    - 가중치: https://huggingface.co/anvuew/BS-RoFormer (``bs_roformer_ft1_anvuew_sdr_12.55.ckpt``)
    - 가중치 라이선스: **GPL-3.0** (HF에 명시). 신규 후보 중 유일하게 라이선스가 확정된
      보컬 본체 모델이다.
    - SDR: 파일명의 12.55는 **저자 자체 테스트셋** 기준이며 MVSep multisong이 아니다.
      Leap과 직접 줄세우면 안 되고, 같은 세트로 우리가 실측해야 한다.

``bs-hyperace-v2``
    **BS-Roformer Voc HyperACEv2** (pcunwa). 기존 조사 문서의 기준선 후보
    (MVSep multisong vocals 11.3957, 37위).
    - 가중치: https://huggingface.co/pcunwa/BS-Roformer-HyperACE (``v2_voc/bs_roformer_voc_hyperacev2.ckpt``)
    - 가중치 라이선스: **미확인**. HF 리포에 라이선스 태그가 없다. 모델 카드가
      "이 가중치는 anvuew의 BS-RoFormer 가중치에 기반한다"고 명시하는데, anvuew 본체는
      **GPL-3.0**(``anvuew-ft1`` 참조)이라 파생 체인상 GPL 상속 가능성이 있다. 명시적
      NC 표기는 없어 실측 대상에는 포함하지만, 상업적 채택 전 라이선스 확인이 필요하다.
    - 아키텍처: 표준 BS-RoFormer가 **아니다**. ``MaskEstimator`` 내부에 표준 per-band
      MLP 출력과 별개로 ``SegmModel``(인코더-디코더 + ``HyperACE`` hypergraph
      convolution 브랜치)을 더하는 커스텀 변형이다. audio-separator 내장 RoFormer도
      MSST 표준 ``bs_roformer.py`` 도 이 구조를 모른다 — HF 리포 자신의
      ``v2_voc/bs_roformer.py`` 를 커밋 고정으로 받아 쓴다(``bs-polarformer`` 의
      MSST 벤더링과 같은 패턴, 다만 소스가 MSST가 아니라 이 HF 리포 자신).
    - config 실측: dim 256 / depth 12 / 62밴드 / ``chunk_size 960000``(≈21.8초 @44.1kHz,
      Leap Xe보다도 크다) / ``inference.num_overlap 4`` / ``inference.batch_size 2``.

``bs-polarformer-fp16`` / ``bs-polarformer-ov2`` / ``bs-polarformer-fast``
    ``bs-polarformer`` 가속 실험용 별명 어댑터 3종(원본 ``bs-polarformer`` 스펙/클래스는
    불변, ``dataclasses.replace`` 로 파생). 가중치·config·라이선스·SDR은 원본과 동일.
    - ``-fp16``: ``kimft-melband-fp16`` 과 같은 ``_wrap_autocast_fp16`` 경로(가중치는
      fp32 유지, forward만 autocast). MSST 벤더링 로더(``_patch_loader_for_pope``)가
      ``separator.model_run`` 에 심는 모델을 그대로 감싼다 — 백엔드 무관하게 동작.
    - ``-ov2``: ``overlap_sec`` 을 8.0→0.0 으로 낮춘다. **주의**: 이름은 "overlap"이지만
      audio-separator ``MDXCSeparator.demix()`` 의 is_roformer 분기에서 이 값은
      MSST의 ``inference.num_overlap``(분할 횟수)과 무관한 **초 단위 홉 스텝**이다
      (``step = min(overlap_sec*sample_rate, chunk_size)``, 0 이하면 step=chunk_size).
      yaml의 ``inference.num_overlap: 2`` 는 이 경로에서 읽히지 않는다. 0.0은 step이
      chunk_size(≈12.77초)에 클램프되는 최대 축소치이고, 기존 8.0초 대비 포워드 패스가
      ~1.6배 줄어든다(4배 축소가 아니다 — 클램프 때문에 이게 이 다이얼의 상한이다).
    - ``-fast``: 위 두 변형의 조합.

``kimft-melband-fp16``
    기존 ``kimft-melband`` 와 **같은 체크포인트**를 fp16 autocast로 돌리는 양자화 실험용
    변형. 가중치·라이선스·SDR은 원본과 동일(KimberleyJSN/melbandroformer).
    VRAM·속도·품질 차이만 보려는 대조군이므로, 결과 해석 시 원본과 쌍으로 볼 것.

VRAM 주의
---------
Leap Xe는 ``chunk_size 881559``(≈20초)로 학습된 모델이고 추론 청크도 그만큼 크다.
3090 24GB 예산 판정에는 워커가 찍는 ``vram_alloc_peak_mb`` / ``vram_reserved_peak_mb``
를 쓸 것.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPO_ROOT / "benchmark"
MODEL_DIR = BENCHMARK_DIR / "models"
SEPARATOR_PYTHON = BENCHMARK_DIR / ".venv-sep" / "Scripts" / "python.exe"

# PolarFormer용 MSST 원본 모델 코드는 핀 커밋에서만 받는다. main을 따라가면 어느 날
# 모델 정의가 바뀌어 같은 체크포인트가 안 붙는 사고가 난다.
MSST_COMMIT = "e247dfe4abc1f17c69dff719207fe045dc04413a"
MSST_VENDOR_DIR = MODEL_DIR / f"msst_src_{MSST_COMMIT[:8]}"
MSST_RAW_BASE = f"https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/{MSST_COMMIT}/"
# bs_roformer.py 가 ``from models.bs_roformer.attend import Attend`` 로 절대 임포트를 하므로
# 패키지 경로를 그대로 재현해야 한다. 저장소의 __init__.py 는 BSConformer 등 쓰지 않는
# 모델까지 끌어와 의존성을 늘리므로 빈 파일로 대체한다.
MSST_FILES = ("models/bs_roformer/attend.py", "models/bs_roformer/bs_roformer.py")

# HyperACEv2 용 커스텀 모델 코드는 MSST가 아니라 HF 리포 자신에서 온다(PolarFormer와
# 다른 소스). attend.py 만 MSST 핀 커밋에서 재사용한다(HyperACE 코드가 같은 상대 경로로
# ``from models.bs_roformer.attend import Attend`` 를 임포트한다).
HYPERACE_COMMIT = "5b1f8283125d5e4a3614d0e3635a636e09c84059"
HYPERACE_VENDOR_DIR = MODEL_DIR / f"hyperace_src_{HYPERACE_COMMIT[:8]}"
HYPERACE_REPO = "pcunwa/BS-Roformer-HyperACE"
HYPERACE_MODEL_FILE = "v2_voc/bs_roformer.py"

BACKEND_AUDIO_SEPARATOR = "audio-separator"
BACKEND_MSST_POPE = "msst-pope"
BACKEND_HYPERACE = "hyperace-custom"


@dataclass(frozen=True)
class _Asset:
    """체크포인트/컨피그 하나. HF 리포는 HF 캐시(HF_HOME)로, 그 외는 URL 직다운."""

    local_name: str
    url: str | None = None
    hf_repo: str | None = None
    hf_path: str | None = None


@dataclass(frozen=True)
class _QualitySpec:
    adapter_name: str
    checkpoint: _Asset
    config: _Asset
    license: str
    source_url: str
    sdr_note: str
    backend: str = BACKEND_AUDIO_SEPARATOR
    autocast_fp16: bool = False
    # audio-separator MDXCSeparator.demix()의 is_roformer 분기가 실제로 쓰는 값이다.
    # 이름은 "overlap"이지만 MSST의 inference.num_overlap(분할 횟수)과는 다른 개념이다 —
    # ``desired_step = overlap_sec * sample_rate; step = min(desired_step, chunk_size)``
    # (0 이하면 step=chunk_size). 즉 초 단위 "홉 스텝"이고, 값이 커질수록 오버랩이
    # 줄어(포워드 패스가 줄어) 빨라진다. yaml의 inference.num_overlap 필드는 이 경로에서
    # 아예 읽히지 않는다. 기본값 8.0은 기존 전 후보 공통 하드코딩 값과 동일하다.
    overlap_sec: float = 8.0


BS_LEAP_XE = _QualitySpec(
    adapter_name="bs-leap-xe",
    checkpoint=_Asset(
        local_name="bs_leap_xe_voc.ckpt",
        hf_repo="pcunwa/BS-Roformer-Leap",
        hf_path="Xe/bs_leap_xe_voc.ckpt",
    ),
    config=_Asset(
        local_name="leap_xe_config_voc.yaml",
        hf_repo="pcunwa/BS-Roformer-Leap",
        hf_path="Xe/leap_xe_config_voc.yaml",
    ),
    license="weights: unverified (no HF license tag, no explicit NC); code path MIT",
    source_url="https://huggingface.co/pcunwa/BS-Roformer-Leap",
    sdr_note="MVSep multisong vocals 11.7577",
)

BS_POLARFORMER = _QualitySpec(
    adapter_name="bs-polarformer",
    checkpoint=_Asset(
        local_name="model_bs_polarformer_float16.ckpt",
        url=(
            "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/"
            "v1.0.20/model_bs_polarformer_float16.ckpt"
        ),
    ),
    config=_Asset(
        local_name="model_bs_polarformer_float16.yaml",
        url=(
            "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/"
            "v1.0.20/model_bs_polarformer_float16.yaml"
        ),
    ),
    # 2026-07-31 개정: 자가 학습·자가 배포 릴리스 자산(v1.0.20, LICENSE 이후)이라 저장소 MIT 상속 —
    # 근거 체인은 docs/research/2026-07-30-model-replacement/final-weights-licenses.md §1 개정 절.
    license="weights: MIT (MSST repo inheritance, self-trained release asset v1.0.20); code MIT (MSST) + MIT (PoPE-pytorch)",
    source_url="https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/tag/v1.0.20",
    sdr_note="MSST-reported multisong vocals 11.00",
    backend=BACKEND_MSST_POPE,
)

ANVUEW_FT1 = _QualitySpec(
    adapter_name="anvuew-ft1",
    checkpoint=_Asset(
        local_name="bs_roformer_ft1_anvuew_sdr_12.55.ckpt",
        hf_repo="anvuew/BS-RoFormer",
        hf_path="bs_roformer_ft1_anvuew_sdr_12.55.ckpt",
    ),
    config=_Asset(
        local_name="bs_roformer_anvuew_config.yaml",
        hf_repo="anvuew/BS-RoFormer",
        hf_path="config.yaml",
    ),
    license="GPL-3.0 (weights and code, stated on the HF model card)",
    source_url="https://huggingface.co/anvuew/BS-RoFormer",
    sdr_note="author-reported 12.55 on their own test set (NOT MVSep multisong)",
)

BS_HYPERACE_V2 = _QualitySpec(
    adapter_name="bs-hyperace-v2",
    checkpoint=_Asset(
        local_name="bs_roformer_voc_hyperacev2.ckpt",
        hf_repo=HYPERACE_REPO,
        hf_path="v2_voc/bs_roformer_voc_hyperacev2.ckpt",
    ),
    config=_Asset(
        local_name="bs_hyperacev2_voc_config.yaml",
        hf_repo=HYPERACE_REPO,
        hf_path="v2_voc/config.yaml",
    ),
    license=(
        "weights: unverified (no HF license tag); model card states derived from "
        "anvuew/BS-RoFormer (GPL-3.0) -> possible GPL inheritance, confirm before "
        "commercial use; custom bs_roformer.py has no license header"
    ),
    source_url="https://huggingface.co/pcunwa/BS-Roformer-HyperACE",
    sdr_note="MVSep multisong vocals 11.3957 (37위, 기존 조사 문서의 기준선 후보)",
    backend=BACKEND_HYPERACE,
)

KIMFT_MELBAND_FP16 = _QualitySpec(
    adapter_name="kimft-melband-fp16",
    checkpoint=_Asset(
        local_name="MelBandRoformer.ckpt",
        url=(
            "https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/"
            "MelBandRoformer.ckpt"
        ),
    ),
    config=_Asset(
        local_name="config_vocals_mel_band_roformer_kim.yaml",
        url=(
            "https://raw.githubusercontent.com/TRvlvr/application_data/main/"
            "mdx_model_data/mdx_c_configs/config_vocals_mel_band_roformer_kim.yaml"
        ),
    ),
    license="same as kimft-melband (KimberleyJSN/melbandroformer)",
    source_url="https://huggingface.co/KimberleyJSN/melbandroformer",
    sdr_note="same checkpoint as kimft-melband; fp16 autocast variant for VRAM/speed contrast",
    autocast_fp16=True,
)

# bs-polarformer 가속 실험 변형 3종. bs-polarformer 원본 스펙/클래스는 절대 건드리지
# 않고 ``dataclasses.replace`` 로 파생시킨다 — 기존 벤치 결과와의 비교 가능성을 지킨다.
BS_POLARFORMER_FP16 = replace(
    BS_POLARFORMER,
    adapter_name="bs-polarformer-fp16",
    autocast_fp16=True,
    # fp16 안내 문구는 _QualitySeparator.separate()가 autocast_fp16 플래그를 보고
    # 자동으로 note에 붙이므로(중복 방지) 여기서는 sdr_note를 원본 그대로 둔다.
)

BS_POLARFORMER_OV2 = replace(
    BS_POLARFORMER,
    adapter_name="bs-polarformer-ov2",
    overlap_sec=0.0,
    sdr_note=BS_POLARFORMER.sdr_note
    + "; zero-overlap variant (overlap_sec 8.0->0.0, step clamps to chunk_size ~12.77s,"
    " ~1.6x fewer forward passes than baseline)",
)

BS_POLARFORMER_FAST = replace(
    BS_POLARFORMER,
    adapter_name="bs-polarformer-fast",
    autocast_fp16=True,
    overlap_sec=0.0,
    sdr_note=BS_POLARFORMER.sdr_note + "; combined fp16 autocast + zero-overlap variant",
)

MODEL_SPECS = {
    spec.adapter_name: spec
    for spec in (
        BS_LEAP_XE,
        BS_POLARFORMER,
        ANVUEW_FT1,
        BS_HYPERACE_V2,
        KIMFT_MELBAND_FP16,
        BS_POLARFORMER_FP16,
        BS_POLARFORMER_OV2,
        BS_POLARFORMER_FAST,
    )
}


@dataclass
class _SeparationOut:
    """``benchmark_alignment.SeparationOut`` 과 구조만 맞춘 값 객체.

    하네스는 ``isinstance`` 가 아니라 속성과 ``meta()`` 만 본다. 여기서 로컬로 두면
    실행 중인 벤치 스크립트를 다른 모듈 이름으로 다시 임포트하는 일을 피할 수 있다.
    """

    vocals_path: Path
    inst_path: Path
    elapsed_sec: float | None = None
    vram_peak_mb: float | None = None
    vram_device_peak_mb: float | None = None
    cached: bool = False
    note: str | None = None

    def meta(self) -> dict[str, Any]:
        return {
            "elapsed_sec": self.elapsed_sec,
            "vram_peak_mb": self.vram_peak_mb,
            "vram_device_peak_mb": self.vram_device_peak_mb,
            "note": self.note,
        }


def _require_separator_python() -> Path:
    if SEPARATOR_PYTHON.is_file():
        return SEPARATOR_PYTHON
    raise RuntimeError(
        "품질파 분리 환경이 없다: "
        f"{SEPARATOR_PYTHON} 를 기대했다. 메인 .venv 는 건드리지 말고 "
        "benchmark/.venv-sep 에 audio-separator 스택을 설치할 것."
    )


def _prepare_benchmark_input(audio_path: Path, work_dir: Path) -> Path:
    """기존 Demucs/RoFormer 어댑터와 같은 24 kHz 모노 입력 경로를 쓴다(비교 가능성 유지)."""
    from everyric2.audio.loader import AudioLoader
    from everyric2.config.settings import get_settings

    prepared = work_dir / "quality_input.wav"
    AudioLoader(get_settings().audio).load(audio_path).to_file(prepared)
    return prepared


class _QualitySeparator:
    """격리된 audio-separator 설치를 감싸는 서브프로세스 어댑터."""

    spec: _QualitySpec
    name: str

    def __init__(self, spec: _QualitySpec):
        self.spec = spec
        self.name = spec.adapter_name

    def separate(self, audio_path: Path, work_dir: Path) -> _SeparationOut:
        python = _require_separator_python()
        work_dir.mkdir(parents=True, exist_ok=True)
        prepared_input = _prepare_benchmark_input(audio_path, work_dir)
        vocals = work_dir / "vocals.wav"
        inst = work_dir / "inst.wav"
        cmd = [
            str(python),
            str(Path(__file__).resolve()),
            "--worker",
            "--model",
            self.spec.adapter_name,
            "--audio",
            str(prepared_input),
            "--output-dir",
            str(work_dir),
            "--models-dir",
            str(MODEL_DIR),
        ]
        started = time.perf_counter()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=3600,
            )
        finally:
            prepared_input.unlink(missing_ok=True)
        elapsed = time.perf_counter() - started
        if result.returncode != 0:
            raise RuntimeError(
                f"{self.name} worker failed ({result.returncode}):\n"
                f"stdout:\n{result.stdout[-4000:]}\nstderr:\n{result.stderr[-4000:]}"
            )
        if not vocals.is_file() or not inst.is_file():
            raise RuntimeError(
                f"{self.name} worker returned success without both stems: "
                f"vocals={vocals.exists()}, inst={inst.exists()}"
            )
        # 워커가 마지막 줄에 찍는 JSON에서 프로세스 내부 VRAM 피크를 회수한다
        vram_alloc = vram_reserved = None
        for line in reversed((result.stdout or "").strip().splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    info = json.loads(line)
                except json.JSONDecodeError:
                    break
                vram_alloc = info.get("vram_alloc_peak_mb")
                vram_reserved = info.get("vram_reserved_peak_mb")
                break
        note = (
            "vocals predicted by single-target RoFormer; inst is mixture-minus-vocals residual"
            f"; license={self.spec.license}; sdr={self.spec.sdr_note}"
        )
        if self.spec.autocast_fp16:
            note += "; fp16 autocast forward (weights stay fp32)"
        if self.spec.overlap_sec != 8.0:
            note += f"; overlap_sec={self.spec.overlap_sec} (baseline=8.0)"
        if vram_reserved is not None:
            note += "; vram_device_peak_mb는 워커 reserved 피크"
        return _SeparationOut(
            vocals_path=vocals,
            inst_path=inst,
            elapsed_sec=round(elapsed, 2),
            vram_peak_mb=vram_alloc,
            vram_device_peak_mb=vram_reserved,
            note=note,
        )


class BSLeapXeSeparator(_QualitySeparator):
    """BS-Roformer Leap Xe (pcunwa) — 모듈 docstring의 ``bs-leap-xe`` 항목 참조."""

    name = BS_LEAP_XE.adapter_name

    def __init__(self) -> None:
        super().__init__(BS_LEAP_XE)


class BSPolarFormerSeparator(_QualitySeparator):
    """BS PolarFormer 오픈 가중치판 (ZFTurbo v1.0.20, fp16 ckpt, MIT 코드)."""

    name = BS_POLARFORMER.adapter_name

    def __init__(self) -> None:
        super().__init__(BS_POLARFORMER)


class AnvuewFT1Separator(_QualitySeparator):
    """anvuew BS-RoFormer ft1 (GPL-3.0) — 라이선스가 확정된 유일한 신규 후보."""

    name = ANVUEW_FT1.adapter_name

    def __init__(self) -> None:
        super().__init__(ANVUEW_FT1)


class BSHyperACEV2Separator(_QualitySeparator):
    """BS-Roformer Voc HyperACEv2 (pcunwa) — 모듈 docstring의 ``bs-hyperace-v2`` 항목 참조."""

    name = BS_HYPERACE_V2.adapter_name

    def __init__(self) -> None:
        super().__init__(BS_HYPERACE_V2)


class KimFTMelBandFP16Separator(_QualitySeparator):
    """kimft-melband과 같은 체크포인트의 fp16 autocast 변형(양자화 대조군)."""

    name = KIMFT_MELBAND_FP16.adapter_name

    def __init__(self) -> None:
        super().__init__(KIMFT_MELBAND_FP16)


class BSPolarFormerFP16Separator(_QualitySeparator):
    """bs-polarformer의 autocast fp16 forward 변형(가중치는 fp32 유지). 원본 bs-polarformer는 불변."""

    name = BS_POLARFORMER_FP16.adapter_name

    def __init__(self) -> None:
        super().__init__(BS_POLARFORMER_FP16)


class BSPolarFormerOV2Separator(_QualitySeparator):
    """bs-polarformer의 오버랩 축소(overlap_sec 8.0->0.0, zero-overlap) 변형. 원본 bs-polarformer는 불변."""

    name = BS_POLARFORMER_OV2.adapter_name

    def __init__(self) -> None:
        super().__init__(BS_POLARFORMER_OV2)


class BSPolarFormerFastSeparator(_QualitySeparator):
    """bs-polarformer의 fp16 autocast + zero-overlap 겸용 변형. 원본 bs-polarformer는 불변."""

    name = BS_POLARFORMER_FAST.adapter_name

    def __init__(self) -> None:
        super().__init__(BS_POLARFORMER_FAST)


def register(separator_registry: dict) -> None:
    """하네스 모듈을 임포트하지 않고 분리 후보 클래스를 등록한다."""
    separator_registry.update(
        {
            BSLeapXeSeparator.name: BSLeapXeSeparator,
            BSPolarFormerSeparator.name: BSPolarFormerSeparator,
            AnvuewFT1Separator.name: AnvuewFT1Separator,
            BSHyperACEV2Separator.name: BSHyperACEV2Separator,
            KimFTMelBandFP16Separator.name: KimFTMelBandFP16Separator,
            BSPolarFormerFP16Separator.name: BSPolarFormerFP16Separator,
            BSPolarFormerOV2Separator.name: BSPolarFormerOV2Separator,
            BSPolarFormerFastSeparator.name: BSPolarFormerFastSeparator,
        }
    )


# ---------------------------------------------------------------------------
# 아래는 전부 워커(benchmark/.venv-sep 인터프리터) 전용 코드다.
# ---------------------------------------------------------------------------


def _download(url: str, destination: Path) -> None:
    """한 번만 받고 마지막에 원자적으로 rename 한다(requests 는 sep venv 에만 있다)."""
    if destination.is_file() and destination.stat().st_size > 0:
        return

    import requests

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    partial.unlink(missing_ok=True)
    with requests.get(url, stream=True, timeout=(15, 120)) as response:
        response.raise_for_status()
        expected = int(response.headers.get("Content-Length", "0"))
        content_encoded = bool(response.headers.get("Content-Encoding"))
        written = 0
        with partial.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    written += len(chunk)
    # requests 는 인코딩된 응답을 투명하게 풀어주므로 Content-Length 와 디스크 바이트 수가
    # 항상 같지는 않다.
    if expected and not content_encoded and written != expected:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"incomplete download for {destination.name}: {written}/{expected} bytes")
    partial.replace(destination)


def _resolve_asset(asset: _Asset, models_dir: Path) -> Path:
    """HF 리포 자산은 HF 캐시(HF_HOME)로, 그 외는 models_dir 로 받는다."""
    if asset.hf_repo:
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(repo_id=asset.hf_repo, filename=asset.hf_path))
    if not asset.url:
        raise RuntimeError(f"asset {asset.local_name} has neither hf_repo nor url")
    destination = models_dir / asset.local_name
    _download(asset.url, destination)
    return destination


def _ensure_msst_sources() -> Path:
    """PolarFormer용 MSST 모델 코드를 핀 커밋에서 받아 임포트 가능한 경로를 돌려준다."""
    package_dir = MSST_VENDOR_DIR / "models" / "bs_roformer"
    package_dir.mkdir(parents=True, exist_ok=True)
    init_file = package_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text(
            "# MSST models.bs_roformer 의 축소 재현 — bs_roformer.py 의 절대 임포트를\n"
            "# 성립시키기 위한 빈 패키지 마커다(원본 __init__ 은 쓰지 않는 모델까지 끌어온다).\n",
            encoding="utf-8",
        )
    for relative in MSST_FILES:
        _download(MSST_RAW_BASE + relative, MSST_VENDOR_DIR / relative)
    return MSST_VENDOR_DIR


def _build_pope_model(model_config: dict, checkpoint: Path, device):
    """MSST 원본 BSRoformer(use_pope=True)를 만들고 체크포인트를 얹는다.

    audio-separator 내장 BSRoformer 는 ``use_pope`` 인자를 모르기 때문에 이 경로만
    MSST 구현을 쓴다. fp16 체크포인트는 ``load_state_dict`` 가 fp32 파라미터로
    캐스팅해 복사하므로 별도 변환이 필요 없다.
    """
    import torch

    vendor_dir = str(_ensure_msst_sources())
    if vendor_dir not in sys.path:
        sys.path.insert(0, vendor_dir)
    from models.bs_roformer.bs_roformer import BSRoformer  # type: ignore[import-not-found]

    model = BSRoformer(**model_config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    return model.to(device).eval()


def _ensure_hyperace_sources() -> Path:
    """HyperACEv2용 커스텀 모델 코드를 HF 리포 자신에서 커밋 고정으로 받는다.

    PolarFormer와 달리 소스가 MSST가 아니라 이 HF 리포(``pcunwa/BS-Roformer-HyperACE``)
    자신이다 — ``MaskEstimator`` 내부에 표준 MSST에는 없는 ``SegmModel``/``HyperACE``
    hypergraph 브랜치가 들어있는 커스텀 구현이라 표준 코드로는 체크포인트가 안 붙는다.
    ``attend.py`` 만 MSST 핀 커밋에서 재사용한다(같은 상대 임포트 경로를 쓴다).
    별도 벤더 디렉터리를 쓰는 이유는 PolarFormer의 ``models/bs_roformer/bs_roformer.py``
    와 파일 내용이 다르므로 같은 경로를 공유하면 다운로드 캐시가 서로를 덮어쓰기 때문이다.
    """
    from huggingface_hub import hf_hub_download

    package_dir = HYPERACE_VENDOR_DIR / "models" / "bs_roformer"
    package_dir.mkdir(parents=True, exist_ok=True)
    init_file = package_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text(
            "# HyperACEv2 models.bs_roformer 의 축소 재현 — bs_roformer.py 의 절대 임포트를\n"
            "# 성립시키기 위한 빈 패키지 마커다.\n",
            encoding="utf-8",
        )
    _download(MSST_RAW_BASE + "models/bs_roformer/attend.py", package_dir / "attend.py")
    model_file = package_dir / "bs_roformer.py"
    if not model_file.is_file() or model_file.stat().st_size == 0:
        source = Path(
            hf_hub_download(
                repo_id=HYPERACE_REPO,
                filename=HYPERACE_MODEL_FILE,
                revision=HYPERACE_COMMIT,
            )
        )
        model_file.write_bytes(source.read_bytes())
    return HYPERACE_VENDOR_DIR


def _build_hyperace_model(model_config: dict, checkpoint: Path, device):
    """HyperACEv2 커스텀 ``BSRoformer``(hypergraph MaskEstimator 포함)를 만들고 체크포인트를 얹는다."""
    import torch

    vendor_dir = str(_ensure_hyperace_sources())
    # PolarFormer 경로(MSST_VENDOR_DIR)와 같은 ``models`` 패키지명을 다른 물리 경로에서
    # 재정의하므로, 이전 실행에서 캐시된 모듈이 남아있지 않도록 먼저 비운다. 이 함수는
    # 워커 프로세스당 한 스펙만 처리하므로 다른 백엔드와 동시에 부딪힐 일은 없지만,
    # sys.path 삽입 순서에 의존하는 사고를 막기 위해 명시적으로 정리한다.
    import sys as _sys

    for name in list(_sys.modules):
        if name == "models" or name.startswith("models."):
            del _sys.modules[name]
    if vendor_dir not in sys.path:
        sys.path.insert(0, vendor_dir)
    from models.bs_roformer.bs_roformer import BSRoformer  # type: ignore[import-not-found]

    model = BSRoformer(**model_config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    return model.to(device).eval()


def _patch_loader_for_hyperace(model_config: dict, checkpoint: Path) -> None:
    """RoformerLoader 를 가로채 HyperACEv2 모델을 대신 돌려준다(``_patch_loader_for_pope`` 참조)."""
    from audio_separator.separator.roformer.model_loading_result import (
        ImplementationVersion,
        ModelLoadingResult,
    )
    from audio_separator.separator.roformer.roformer_loader import RoformerLoader

    def _load_model(self, model_path, config, device="cpu"):  # noqa: ANN001
        model = _build_hyperace_model(model_config, checkpoint, device)
        result = ModelLoadingResult.success_result(
            model=model,
            implementation=ImplementationVersion.NEW,
            config=config,
        )
        result.add_model_info("model_type", "bs_roformer_hyperace_v2")
        result.add_model_info("loading_method", "hyperace-vendored")
        result.add_model_info("device", str(device))
        return result

    RoformerLoader.load_model = _load_model


def _patch_loader_for_pope(model_config: dict, checkpoint: Path) -> None:
    """RoformerLoader 를 가로채 PolarFormer 모델을 대신 돌려준다.

    MDXCSeparator 의 청크 분할·overlap-add·스템 기록 파이프라인은 그대로 쓰고,
    모델 생성 지점 하나만 갈아끼우는 게 목적이다.
    """
    from audio_separator.separator.roformer.model_loading_result import (
        ImplementationVersion,
        ModelLoadingResult,
    )
    from audio_separator.separator.roformer.roformer_loader import RoformerLoader

    def _load_model(self, model_path, config, device="cpu"):  # noqa: ANN001
        model = _build_pope_model(model_config, checkpoint, device)
        result = ModelLoadingResult.success_result(
            model=model,
            implementation=ImplementationVersion.NEW,
            config=config,
        )
        result.add_model_info("model_type", "bs_roformer_pope")
        result.add_model_info("loading_method", "msst-vendored")
        result.add_model_info("device", str(device))
        return result

    RoformerLoader.load_model = _load_model


def _wrap_autocast_fp16(model):
    """가중치는 fp32로 두고 forward 만 fp16 autocast 로 돌리는 래퍼.

    ``model.half()`` 로 통째 캐스팅하면 STFT/iSTFT 가 half 를 거부하거나 정밀도가
    무너진다. autocast 는 STFT 를 fp32 로 유지한 채 matmul 만 fp16 으로 내려 실제
    속도·VRAM 이득은 그대로 가져간다.
    """
    import torch
    from torch import nn

    class _AutocastWrapper(nn.Module):
        def __init__(self, inner: nn.Module):
            super().__init__()
            self.inner = inner

        def forward(self, *args, **kwargs):
            with torch.autocast("cuda", dtype=torch.float16):
                out = self.inner(*args, **kwargs)
            return out.float() if torch.is_tensor(out) else out

    return _AutocastWrapper(model)


def _load_separator(spec: _QualitySpec, models_dir: Path, output_dir: Path, force_cpu: bool):
    """명시적 URL/HF 경로에서 audio-separator 의 RoFormer MDXC 백엔드를 세운다."""
    if force_cpu:
        # CPU 검증이 살아있는 GPU 를 건드리지 않도록 torch 임포트 전에 세팅해야 한다.
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    import torch
    import yaml
    from audio_separator.separator.architectures.mdxc_separator import MDXCSeparator

    checkpoint = _resolve_asset(spec.checkpoint, models_dir)
    config_path = _resolve_asset(spec.config, models_dir)
    with config_path.open(encoding="utf-8") as handle:
        model_data = yaml.load(handle, Loader=yaml.FullLoader)
    if not isinstance(model_data, dict):
        raise RuntimeError(f"invalid YAML configuration: {config_path}")
    model_data["is_roformer"] = True
    # 추론에서는 gradient checkpointing 이 이득이 없고(no_grad), 재계산 비용만 든다.
    # anvuew config 가 True 로 켜 두었으므로 여기서 내린다 — 가중치 형상과 무관하다.
    if isinstance(model_data.get("model"), dict):
        model_data["model"]["use_torch_checkpoint"] = False

    if spec.backend == BACKEND_MSST_POPE:
        _patch_loader_for_pope(dict(model_data["model"]), checkpoint)
    elif spec.backend == BACKEND_HYPERACE:
        _patch_loader_for_hyperace(dict(model_data["model"]), checkpoint)

    device = torch.device("cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger = logging.getLogger(f"everyric2.benchmark.{spec.adapter_name}")
    logger.setLevel(logging.WARNING)
    common_config = {
        "logger": logger,
        "log_level": logging.WARNING,
        "torch_device": device,
        "torch_device_cpu": torch.device("cpu"),
        "torch_device_mps": None,
        "onnx_execution_provider": None,
        "model_name": checkpoint.stem,
        "model_path": str(checkpoint),
        "model_data": model_data,
        "output_format": "WAV",
        "output_bitrate": None,
        "output_dir": str(output_dir),
        "normalization_threshold": 0.9,
        "amplification_threshold": 0.0,
        "output_single_stem": None,
        "invert_using_spec": False,
        "sample_rate": int(model_data["audio"]["sample_rate"]),
        "use_soundfile": True,
    }
    arch_config = {
        # segment_size/override_model_segment_size/batch_size/pitch_shift는 기존 RoFormer
        # 어댑터와 같은 값 — 후보 간 비교 가능성을 위해 건드리지 않는다. overlap만
        # spec.overlap_sec(기본 8.0, 기존과 동일)로 스펙별 오버라이드를 허용한다 —
        # 가속 실험 변형(예: bs-polarformer-ov2)에서만 값을 바꾼다.
        "segment_size": 256,
        "override_model_segment_size": False,
        "overlap": spec.overlap_sec,
        "batch_size": 1,
        "pitch_shift": 0,
    }
    separator = MDXCSeparator(common_config=common_config, arch_config=arch_config)
    if spec.autocast_fp16 and not force_cpu:
        if device.type != "cuda":
            raise RuntimeError(f"{spec.adapter_name} requires CUDA for fp16 autocast; device={device}")
        separator.model_run = _wrap_autocast_fp16(separator.model_run).to(device).eval()
    # force_cpu(=verify)에서는 래핑을 건너뛴다 — autocast 는 CUDA 전용이고, verify 는
    # 파라미터 형상만 보는 경로라 fp32 그대로도 검사 목적을 만족한다.
    return separator, device


def _worker(args: argparse.Namespace) -> int:
    spec = MODEL_SPECS[args.model]
    separator, device = _load_separator(
        spec,
        Path(args.models_dir),
        Path(args.output_dir),
        force_cpu=args.action == "verify",
    )
    parameters = list(separator.model_run.parameters())
    if args.action == "verify":
        print(
            json.dumps(
                {
                    "model": spec.adapter_name,
                    "device": str(device),
                    "backend": spec.backend,
                    "parameters": sum(parameter.numel() for parameter in parameters),
                    "first_parameter_shape": list(parameters[0].shape),
                    "license": spec.license,
                }
            )
        )
        return 0

    output_dir = Path(args.output_dir)
    # 스템 키는 모델 카드마다 다르다 — Leap/PolarFormer 는 여집합을 "other", anvuew 는
    # "instrument" 로 부른다. 흔한 표기를 전부 걸고, 모르는 키는 audio-separator 가
    # 조용히 무시한다.
    outputs = separator.separate(
        args.audio,
        custom_output_names={
            "vocals": "vocals",
            "Vocals": "vocals",
            "other": "inst",
            "Other": "inst",
            "instrument": "inst",
            "Instrument": "inst",
            "instrumental": "inst",
            "Instrumental": "inst",
        },
    )
    # 그래도 남는 기본 파일명은 이름 휴리스틱으로 표준 이름에 맞춰 리네임한다.
    vocals_path = output_dir / "vocals.wav"
    inst_path = output_dir / "inst.wav"
    for raw in map(Path, outputs):
        path = raw if raw.is_absolute() else output_dir / raw
        path = path.resolve()
        if path in (vocals_path.resolve(), inst_path.resolve()) or not path.exists():
            continue
        low = path.name.lower()
        if "vocal" in low and not vocals_path.exists():
            path.replace(vocals_path)
        elif ("inst" in low or "other" in low or "no_vocal" in low or "accompaniment" in low) and not inst_path.exists():
            path.replace(inst_path)
    if not (vocals_path.exists() and inst_path.exists()):
        raise RuntimeError(f"unexpected RoFormer output paths: {outputs}")
    # 워커 프로세스 내부 VRAM 피크 — 곡 길이별 분리 VRAM 곡선(3090 예산 판정)의 원료.
    vram: dict[str, float] = {}
    try:
        import torch

        if torch.cuda.is_available():
            vram = {
                "vram_alloc_peak_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
                "vram_reserved_peak_mb": round(torch.cuda.max_memory_reserved() / 2**20, 1),
            }
    except Exception:
        pass
    print(json.dumps({"model": spec.adapter_name, "device": str(device), "outputs": outputs, **vram}))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Internal worker for quality-tier separator adapters")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model", choices=sorted(MODEL_SPECS))
    parser.add_argument("--action", choices=("separate", "verify"), default="separate")
    parser.add_argument("--audio")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--models-dir", default=str(MODEL_DIR))
    args = parser.parse_args()
    if not args.worker:
        parser.error("this module is invoked by the benchmark adapter; pass --worker")
    if args.action == "separate" and not args.audio:
        parser.error("--audio is required for separation")
    return args


if __name__ == "__main__":
    raise SystemExit(_worker(_parse_args()))
