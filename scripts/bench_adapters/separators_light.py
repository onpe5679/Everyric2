"""경량파(클라이언트/저비용 서버) 보컬 분리 후보 어댑터.

조사 근거: ``docs/research/2026-07-31-acapella-quality-candidates.md`` Part 2.
용도는 「브라우저 WebGPU / 로컬 CPU에서도 돌 만한 분리기가 정렬 입력으로 쓸 만한가」를
실측하는 것이다. 품질파(BS-RoFormer 계열)와 달리 **SDR 상한이 아니라 RTF·모델 크기 대비
정렬 손실**이 판정 축이다.

등록 어댑터
-----------

===================== ================================================== ==================
어댑터명              모델 / 출처                                        라이선스
===================== ================================================== ==================
``demucs-onnx``       HT-Demucs FT vocals 스페셜리스트 ONNX fp32          MIT
                      (StemSplit ``demucs-onnx``,
                      HF ``StemSplitio/htdemucs-ft-vocals-onnx``)
                      MUSDB vocals SDR 9.19 / 316 MB / 저자보고 RTF 0.20
                      (M4 Pro CPU, 단일 스템)
``demucs-onnx-fp16``  같은 그래프의 fp16 가중치 변종                      MIT
                      (``precision="fp16weights"``). 다운로드 크기 약
                      1.9배 감소, 저자보고 런타임·RAM은 fp32와 동일,
                      fp32 대비 최대 절대오차 ~6e-5.
``demucs-onnx-int8``  위 fp32 그래프를 onnxruntime 동적 양자화            MIT
                      (``quantize_dynamic``, QInt8)한 변종. 우리가 만든
                      산물이라 상류 품질 보고가 없다 — 순수 실측 대상.
                      CPU EP 전용(양자화 연산자는 CUDA EP 미지원).
                      **실측 결론은 부정적**: 크기는 302→188MB(1.6배)
                      밖에 안 줄고 CPU 속도는 RTF 0.24 → 0.77로 **3배
                      느려졌으며**, 같은 곡 omniasr 정렬 MAE가
                      0.079 → 0.198로 나빠졌다. 경량화 수단으로 쓰지 말 것.
``mini-bsrofo-18m``   HiDolen/Mini-BS-RoFormer-18M (transformers          MIT
                      custom_code, 17.9M 파라미터). MUSDB18HQ val
                      vocals SDR 10.03 — 척도가 MVSep multisong과 달라
                      품질파 표와 직접 줄세우면 안 된다.
``umx-l``             Open-Unmix UMX-L (openunmix, torch.hub 가중치)      MIT
                      구세대 대조군(vocals ~7급). 경량축 하한이 어디서
                      무너지는지 보기 위한 축이지 채택 후보가 아니다.
===================== ================================================== ==================

**범위 밖으로 남긴 것**: ``BSRoformer.cpp``(chenmozhijin, MIT, GGML)는 C++ 빌드 체인
(CMake + CUDA/Vulkan 백엔드)이 필요해 이번 배선에서 제외했다. 공개된 RTF·곡당 처리시간
보고가 아예 없어서 도입하려면 빌드부터 실측까지가 별건 작업이다. ``Mini-BS-RoFormer-V2-46.8M``은
성능(vocals 10.86)·연산량 모두 18M판보다 낫지만 **CC-BY-NC-4.0**이라 정책상 제외했다.

의존성 격리
-----------

onnxruntime-gpu / transformers / openunmix는 **``benchmark/.venv-light`` 전용 venv**에만 있고,
메인 벤치 인터프리터는 건드리지 않는다(``.venv-sep``/``.venv-owsm`` 워커 패턴과 동일).
이 파일이 그 인터프리터에서 자기 자신을 ``--worker``로 다시 실행한다.

실행 프로바이더(EP)
-------------------

ONNX 경로는 ``CUDA → DML → CPU`` 순으로 실제 사용 가능한 EP를 고른다. 어느 EP로 돌았는지는
세션에서 되읽어 ``SeparationOut.note``에 남긴다 — RTX 5090(sm_120)처럼 EP 빌드가 하드웨어를
못 따라오면 조용히 CPU로 떨어지므로, 시간 수치를 EP 없이 읽으면 안 된다. onnxruntime-gpu의
CUDA EP는 torch가 들고 있는 cuDNN/cuBLAS DLL을 쓰므로 워커가 ``torch/lib``을 DLL 검색 경로에
먼저 넣는다.

입력 계약
---------

기존 분리 어댑터(htdemucs, RoFormer 계열)와 **같은 입력 열화**를 쓴다: ``AudioLoader``로
24kHz 모노 wav를 만든 뒤 워커가 44.1kHz 스테레오로 되올려 모델에 먹인다. 원본을 직접 먹이면
기준선이 다른 분리 후보와 달라져 비교가 깨진다.

CPU 시간 측정
-------------

클라이언트 RTF 추정용 CPU 실측은 **기본 꺼져 있다**(전수 스윕에 곡마다 CPU 재실행을 붙이면
스윕이 몇 배로 늘어난다). ``LIGHT_CPU_PROBE_SEC=30`` 처럼 환경변수로 초를 주면 그 길이만큼
앞부분을 **별도 워커 프로세스**에서 CPU로 한 번 더 돌려 ``cpu_probe`` 수치를 note에 싣는다.
같은 프로세스에서 GPU 실행 직후 CPU로 재실행하면 mini-bsrofo가 접근 위반으로 죽는다(실측).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPO_ROOT / "benchmark"
LIGHT_PYTHON = BENCHMARK_DIR / ".venv-light" / "Scripts" / "python.exe"
LIGHT_MODEL_DIR = BENCHMARK_DIR / "models" / "light"

MODEL_SAMPLE_RATE = 44100
MINI_BSROFO_REPO = "HiDolen/Mini-BS-RoFormer-18M"


@dataclass(frozen=True)
class _LightSpec:
    """어댑터 하나의 워커 파라미터. ``kind``가 워커의 분기 키다."""

    adapter_name: str
    kind: str  # "demucs_onnx" | "mini_bsrofo" | "umx"
    precision: str = "fp32"  # demucs_onnx: fp32 | fp16weights | int8dyn
    license: str = "MIT"
    source: str = ""


SPECS: dict[str, _LightSpec] = {
    spec.adapter_name: spec
    for spec in (
        _LightSpec(
            adapter_name="demucs-onnx",
            kind="demucs_onnx",
            precision="fp32",
            source="StemSplitio/htdemucs-ft-vocals-onnx (MUSDB vocals SDR 9.19)",
        ),
        _LightSpec(
            adapter_name="demucs-onnx-fp16",
            kind="demucs_onnx",
            precision="fp16weights",
            source="StemSplitio/htdemucs-ft-vocals-onnx fp16weights",
        ),
        _LightSpec(
            adapter_name="demucs-onnx-int8",
            kind="demucs_onnx",
            precision="int8dyn",
            source="onnxruntime quantize_dynamic(QInt8) of htdemucs_ft_vocals fp32",
        ),
        _LightSpec(
            adapter_name="mini-bsrofo-18m",
            kind="mini_bsrofo",
            source=f"{MINI_BSROFO_REPO} (MUSDB18HQ val vocals SDR 10.03, 17.9M params)",
        ),
        _LightSpec(
            adapter_name="umx-l",
            kind="umx",
            source="openunmix umxl (구세대 대조군)",
        ),
    )
}


@dataclass
class _SeparationOut:
    """``benchmark_alignment.SeparationOut``의 구조적 등가물.

    하네스는 속성과 ``meta()``만 쓰고 ``isinstance``를 보지 않는다. 실행 중인 하네스 모듈을
    다른 이름으로 재-import 하지 않기 위해 여기 로컬로 둔다(RoFormer 어댑터와 같은 이유).
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


def _require_light_python() -> Path:
    if LIGHT_PYTHON.is_file():
        return LIGHT_PYTHON
    raise RuntimeError(
        "경량 분리 후보 전용 환경이 없다: "
        f"{LIGHT_PYTHON} 가 필요하다. `python -m venv benchmark/.venv-light` 로 만들고 "
        "onnxruntime-gpu / demucs-onnx / transformers / openunmix 를 거기에만 설치할 것 "
        "(메인 .venv에 설치 금지)."
    )


def _prepare_benchmark_input(audio_path: Path, work_dir: Path) -> Path:
    """기존 분리 어댑터와 같은 24kHz 모노 입력 경로."""
    from everyric2.audio.loader import AudioLoader
    from everyric2.config.settings import get_settings

    prepared = work_dir / "light_input.wav"
    AudioLoader(get_settings().audio).load(audio_path).to_file(prepared)
    return prepared


class _LightSeparator:
    """전용 venv 워커를 부르는 서브프로세스 어댑터."""

    spec: _LightSpec
    name: str

    def __init__(self, spec: _LightSpec):
        self.spec = spec
        self.name = spec.adapter_name

    def separate(self, audio_path: Path, work_dir: Path) -> _SeparationOut:
        python = _require_light_python()
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
            str(LIGHT_MODEL_DIR),
        ]
        env = dict(os.environ)
        env.setdefault("PYTHONIOENCODING", "utf-8")
        started = time.perf_counter()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=7200,
                env=env,
            )
            wall = time.perf_counter() - started
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
            info = _last_json_line(result.stdout)
            info.update(self._cpu_probe(python, prepared_input, work_dir, env))
        finally:
            prepared_input.unlink(missing_ok=True)
        # 워커가 잰 순수 분리 시간이 있으면 그걸 쓴다 — 서브프로세스 wall에는 인터프리터
        # 기동·모델 로드가 섞여 있어 후보 간 비교에 못 쓴다.
        elapsed = info.get("separate_sec")
        return _SeparationOut(
            vocals_path=vocals,
            inst_path=inst,
            elapsed_sec=round(float(elapsed), 2) if elapsed is not None else round(wall, 2),
            vram_peak_mb=info.get("vram_peak_mb"),
            vram_device_peak_mb=info.get("vram_device_peak_mb"),
            note=self._note(info, wall),
        )

    def _cpu_probe(
        self, python: Path, prepared_input: Path, work_dir: Path, env: dict[str, str]
    ) -> dict[str, Any]:
        """클라이언트 RTF 추정용 CPU 실측 — ``LIGHT_CPU_PROBE_SEC``로 켠다(기본 꺼짐).

        **반드시 별도 프로세스**여야 한다. 같은 워커 안에서 GPU 실행 뒤에 CPU로 다시 돌리면
        mini-bsrofo가 접근 위반(0xC0000005)으로 죽는다(실측). 겸사겸사 프로브가 죽어도 본
        분리 결과는 살아남는다. 전수 스윕에 곡마다 CPU 재실행을 붙이면 스윕이 몇 배가 되므로
        기본값은 0(꺼짐)이다.
        """
        try:
            probe_sec = float(os.environ.get("LIGHT_CPU_PROBE_SEC", "0") or 0)
        except ValueError:
            probe_sec = 0.0
        if probe_sec <= 0:
            return {}
        probe_dir = work_dir / "_cpu_probe"
        shutil.rmtree(probe_dir, ignore_errors=True)
        probe_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(python),
            str(Path(__file__).resolve()),
            "--worker",
            "--model",
            self.spec.adapter_name,
            "--audio",
            str(prepared_input),
            "--output-dir",
            str(probe_dir),
            "--models-dir",
            str(LIGHT_MODEL_DIR),
            "--cpu",
            "--limit-sec",
            str(probe_sec),
        ]
        try:
            started = time.perf_counter()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=7200,
                env=env,
            )
            wall = time.perf_counter() - started
            if result.returncode != 0:
                return {"cpu_probe_error": f"exit {result.returncode}: {result.stderr.strip()[-160:]}"}
            probe = _last_json_line(result.stdout)
            return {
                "cpu_probe_sec": probe.get("separate_sec", round(wall, 2)),
                "cpu_probe_audio_sec": probe.get("audio_sec"),
                "cpu_probe_ep": probe.get("ep", "cpu"),
            }
        except Exception as exc:  # 프로브 실패가 본 실행을 무효화하면 안 된다
            return {"cpu_probe_error": repr(exc)[:160]}
        finally:
            shutil.rmtree(probe_dir, ignore_errors=True)

    def _note(self, info: dict[str, Any], wall: float) -> str:
        parts = [
            f"{self.spec.source}; license={self.spec.license}",
            "vocals는 단일 타깃 예측, inst는 mixture-minus-vocals 잔차",
            f"ep={info.get('ep', 'unknown')}",
        ]
        if info.get("model_mb") is not None:
            parts.append(f"model={info['model_mb']}MB")
        if info.get("load_sec") is not None:
            parts.append(f"load={info['load_sec']}s")
        if info.get("audio_sec") is not None and info.get("separate_sec") is not None:
            rtf = info["separate_sec"] / max(info["audio_sec"], 1e-6)
            parts.append(f"rtf={rtf:.3f}(audio {info['audio_sec']:.1f}s)")
        if info.get("cpu_probe_sec") is not None:
            probe_audio = info.get("cpu_probe_audio_sec") or 0.0
            cpu_rtf = info["cpu_probe_sec"] / max(probe_audio, 1e-6)
            parts.append(
                f"cpu_probe={info['cpu_probe_sec']:.1f}s/{probe_audio:.0f}s(rtf={cpu_rtf:.3f}, "
                f"ep={info.get('cpu_probe_ep', 'CPUExecutionProvider')})"
            )
        if info.get("cpu_probe_error"):
            parts.append(f"cpu_probe 실패: {info['cpu_probe_error']}")
        if info.get("cudnn"):
            parts.append(f"cudnn={info['cudnn']}")
        if info.get("vram_note"):
            parts.append(str(info["vram_note"]))
        if info.get("extra"):
            parts.append(str(info["extra"]))
        parts.append(f"worker_wall={wall:.1f}s")
        return "; ".join(parts)


def _last_json_line(stdout: str) -> dict[str, Any]:
    for line in reversed((stdout or "").strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                return {}
    return {}


class DemucsOnnxSeparator(_LightSeparator):
    name = "demucs-onnx"

    def __init__(self) -> None:
        super().__init__(SPECS["demucs-onnx"])


class DemucsOnnxFp16Separator(_LightSeparator):
    name = "demucs-onnx-fp16"

    def __init__(self) -> None:
        super().__init__(SPECS["demucs-onnx-fp16"])


class DemucsOnnxInt8Separator(_LightSeparator):
    name = "demucs-onnx-int8"

    def __init__(self) -> None:
        super().__init__(SPECS["demucs-onnx-int8"])


class MiniBSRoformer18MSeparator(_LightSeparator):
    name = "mini-bsrofo-18m"

    def __init__(self) -> None:
        super().__init__(SPECS["mini-bsrofo-18m"])


class UmxLSeparator(_LightSeparator):
    name = "umx-l"

    def __init__(self) -> None:
        super().__init__(SPECS["umx-l"])


def register(separator_registry: dict) -> None:
    """하네스 모듈을 import 하지 않고 분리 후보를 등록한다."""
    separator_registry.update(
        {
            DemucsOnnxSeparator.name: DemucsOnnxSeparator,
            DemucsOnnxFp16Separator.name: DemucsOnnxFp16Separator,
            DemucsOnnxInt8Separator.name: DemucsOnnxInt8Separator,
            MiniBSRoformer18MSeparator.name: MiniBSRoformer18MSeparator,
            UmxLSeparator.name: UmxLSeparator,
        }
    )


# ──────────────────────────────────────────────────────────────────────────
# 이하 워커 — benchmark/.venv-light 인터프리터에서만 실행된다
# ──────────────────────────────────────────────────────────────────────────


def _prepare_cuda_dlls() -> str | None:
    """onnxruntime-gpu의 CUDA EP가 쓸 cuDNN/cuBLAS를 로드 시점에 확정한다.

    onnxruntime을 import 하기 **전에** 불러야 한다 — DLL 검색 경로는 로드 시점에 굳는다.

    실측(RTX 5090 / ORT 1.28): torch 2.8 번들 cuDNN 9.10만 걸리면 htdemucs의 ConvTranspose가
    ``CUDNN_FE failure 11: CUDNN_BACKEND_API_FAILED``로 죽는다. pip ``nvidia-cudnn-cu12`` 9.24를
    **명시적으로 먼저 preload** 하면 통과한다. torch/lib에도 같은 이름의 DLL이 있어 검색 경로
    순서에 맡기면 어느 쪽이 물릴지 불확정이므로, 전체 경로로 미리 로드해 프로세스에 못박는다
    (torch는 이미 로드된 모듈을 그대로 쓰며 실측상 conv 동작에 문제없다).
    """
    loaded: str | None = None
    try:
        import ctypes
        import site

        for root in map(Path, site.getsitepackages()):
            nvidia = root / "nvidia"
            if not nvidia.is_dir():
                continue
            for sub in ("cudnn", "cublas", "cuda_runtime", "cuda_nvrtc"):
                bin_dir = nvidia / sub / "bin"
                if bin_dir.is_dir():
                    os.add_dll_directory(str(bin_dir))
            cudnn = nvidia / "cudnn" / "bin" / "cudnn64_9.dll"
            if cudnn.is_file() and loaded is None:
                ctypes.WinDLL(str(cudnn))
                loaded = str(cudnn)
    except Exception:
        loaded = None
    try:
        import torch

        lib = Path(torch.__file__).resolve().parent / "lib"
        if lib.is_dir():
            os.add_dll_directory(str(lib))
    except Exception:
        pass
    return loaded


class _GpuMemSampler:
    """장치 전체 VRAM 사용량을 표본해 「이 프로세스가 늘린 만큼」을 추정한다.

    ONNX Runtime은 torch 할당자를 안 쓰므로 ``torch.cuda.max_memory_allocated``에 잡히지 않고,
    이 머신(WDDM)에서는 ``nvidia-smi --query-compute-apps``의 per-PID 사용량이 ``[N/A]``로
    나온다(실측). 남는 수단은 장치 사용량의 **기준선 대비 증가분**뿐인데, 지금 GPU를 함께 쓰는
    다른 벤치 체인이 있으면 오염된다 — 그래서 절대 피크와 증가분을 함께 남기고 note에 조건을
    명시한다. 절대값을 3090 예산 판정에 그대로 쓰면 안 된다.
    """

    def __init__(self, interval: float = 0.3):
        self.interval = interval
        self.baseline_mb: float | None = None
        self.peak_mb: float | None = None
        self.delta_mb: float | None = None
        self.note: str | None = None
        self._stop = None
        self._thread = None
        self._torch: Any = None

    def _sample(self) -> None:
        peak = self.baseline_mb or 0.0
        while not self._stop.is_set():
            try:
                free, total = self._torch.cuda.mem_get_info()
                peak = max(peak, (total - free) / 2**20)
            except Exception:
                self.note = "vram=장치 표본 실패"
                return
            self._stop.wait(self.interval)
        self.peak_mb = round(peak, 1)
        if self.baseline_mb is not None:
            self.delta_mb = round(max(peak - self.baseline_mb, 0.0), 1)

    def __enter__(self) -> _GpuMemSampler:
        import threading

        try:
            import torch

            if not torch.cuda.is_available():
                self.note = "vram=CUDA 미가용"
                return self
            self._torch = torch
            free, total = torch.cuda.mem_get_info()
            self.baseline_mb = round((total - free) / 2**20, 1)
        except Exception:
            self.note = "vram=장치 표본 불가"
            return self
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._stop is not None:
            self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)


def _load_mix_44k_stereo(path: Path):
    """24kHz 모노 벤치 입력을 모델이 요구하는 44.1kHz 스테레오로 되올린다."""
    import numpy as np
    import soundfile as sf
    import soxr

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = data.T  # (channels, samples)
    if sr != MODEL_SAMPLE_RATE:
        audio = soxr.resample(audio.T, sr, MODEL_SAMPLE_RATE).T.astype("float32", copy=False)
    if audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
    elif audio.shape[0] > 2:
        audio = audio[:2]
    return np.ascontiguousarray(audio, dtype="float32")


def _write_stems(output_dir: Path, mix, vocals):
    """vocals와 mixture-minus-vocals 잔차를 44.1kHz 스테레오로 떨군다."""
    import numpy as np
    import soundfile as sf

    vocals = np.asarray(vocals, dtype="float32")
    if vocals.ndim == 1:
        vocals = np.stack([vocals, vocals], axis=0)
    if vocals.shape[0] > vocals.shape[1]:  # (samples, channels)로 온 경우
        vocals = vocals.T
    if vocals.shape[0] == 1:
        vocals = np.repeat(vocals, 2, axis=0)
    length = min(mix.shape[1], vocals.shape[1])
    vocals = vocals[:2, :length]
    # NaN/Inf 스템을 그냥 쓰면 정렬기는 정상 종료하고 **의미 없는 수치**를 남긴다(umx-l의
    # 위너 EM NaN 사고: MAE 85초짜리 «성공» 런). 여기서 크게 실패시키는 편이 낫다.
    if not np.isfinite(vocals).all():
        bad = int((~np.isfinite(vocals)).sum())
        raise RuntimeError(f"분리 결과에 비유한값 {bad}개 — 스템을 쓰지 않고 중단한다")
    inst = mix[:2, :length] - vocals
    vocals_path = output_dir / "vocals.wav"
    inst_path = output_dir / "inst.wav"
    sf.write(str(vocals_path), vocals.T, MODEL_SAMPLE_RATE, subtype="FLOAT")
    sf.write(str(inst_path), inst.T, MODEL_SAMPLE_RATE, subtype="FLOAT")
    return vocals_path, inst_path


def _write_temp_mix(output_dir: Path, mix) -> Path:
    """demucs-onnx는 파일 경로만 받는다 — 되올린 믹스를 임시 wav로 넘긴다."""
    import soundfile as sf

    path = output_dir / "_light_mix_44k.wav"
    sf.write(str(path), mix.T, MODEL_SAMPLE_RATE, subtype="FLOAT")
    return path


def _pick_onnx_providers(force_cpu: bool = False) -> list[str]:
    import onnxruntime as ort

    if force_cpu:
        return ["CPUExecutionProvider"]
    available = ort.get_available_providers()
    for candidate in ("CUDAExecutionProvider", "DmlExecutionProvider"):
        if candidate in available:
            return [candidate, "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _active_onnx_ep(default: str) -> str:
    """세션 풀에서 실제로 물린 EP를 되읽는다 — 요청과 실제는 다를 수 있다."""
    try:
        import demucs_onnx

        pool = demucs_onnx.session_pool()
        sessions = list(getattr(pool, "_sessions", {}).values())
        if sessions:
            providers = sessions[0].get_providers()
            if providers:
                return providers[0]
    except Exception:
        pass
    return default


def _quantized_vocals_model(models_dir: Path) -> tuple[Path, float]:
    """htdemucs_ft vocals 스페셜리스트를 동적 양자화(QInt8)해 캐시한다."""
    from demucs_onnx.inference import download_stem_model

    source = Path(download_stem_model("vocals", precision="fp32"))
    models_dir.mkdir(parents=True, exist_ok=True)
    target = models_dir / "htdemucs_ft_vocals_int8dyn.onnx"
    if target.is_file() and target.stat().st_size > 0:
        return target, 0.0
    from onnxruntime.quantization import QuantType, quantize_dynamic

    started = time.perf_counter()
    partial = target.with_suffix(".onnx.part")
    partial.unlink(missing_ok=True)
    quantize_dynamic(str(source), str(partial), weight_type=QuantType.QInt8)
    partial.replace(target)
    return target, round(time.perf_counter() - started, 1)


def _use_lean_cpu_session(inference: Any) -> None:
    """int8 세션의 CPU 메모리 아레나를 끈다.

    동적 양자화 그래프는 ConvInteger/MatMulInteger의 int32 중간 텐서 때문에 아레나가 크게
    부풀고, 이 머신처럼 커밋 여유가 좁으면(다른 앱이 커밋을 점유한 상태) 2MB짜리 numpy
    할당조차 실패한다(실측). 아레나를 끄면 느려지는 대신 상주 메모리가 내려간다.
    """
    import onnxruntime as ort

    def _lean_session(onnx_path: Any, providers: Any) -> Any:
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        return ort.InferenceSession(str(onnx_path), sess_options=options, providers=list(providers))

    inference._make_session = _lean_session  # type: ignore[assignment]


def _run_demucs_onnx(
    spec: _LightSpec, mix, output_dir: Path, models_dir: Path, force_cpu: bool
) -> dict[str, Any]:
    import demucs_onnx
    import demucs_onnx.inference as inference

    info: dict[str, Any] = {}
    precision = "fp32" if spec.precision == "int8dyn" else spec.precision
    # 양자화 그래프는 CPU EP에서만 의미가 있다(ConvInteger/MatMulInteger가 CUDA EP 미지원).
    providers = _pick_onnx_providers(force_cpu or spec.precision == "int8dyn")

    load_started = time.perf_counter()
    if spec.precision == "int8dyn":
        quantized, quantize_sec = _quantized_vocals_model(models_dir)
        info["model_mb"] = round(quantized.stat().st_size / 2**20, 1)
        if quantize_sec:
            info["extra"] = f"quantize_dynamic {quantize_sec}s (1회, 캐시됨)"
        original = inference.download_stem_model

        def _patched(stem: str, **kwargs: Any) -> Path:
            if stem == "vocals":
                return quantized
            return original(stem, **kwargs)

        inference.download_stem_model = _patched  # type: ignore[assignment]
        _use_lean_cpu_session(inference)
    else:
        model_path = inference.download_stem_model("vocals", precision=precision)
        info["model_mb"] = round(Path(model_path).stat().st_size / 2**20, 1)
    # 세션 생성(그래프 컴파일)까지 타이머 밖에서 끝낸다 — 안에 두면 첫 곡만 몇 초 비싸져서
    # 후보 간 RTF 비교가 깨진다. separate()는 풀에 있는 세션을 그대로 재사용한다.
    demucs_onnx.prewarm(["htdemucs_ft_vocals"], precision=precision, providers=providers)
    info["load_sec"] = round(time.perf_counter() - load_started, 2)

    temp_mix = _write_temp_mix(output_dir, mix)
    try:
        started = time.perf_counter()
        vocals = demucs_onnx.separate_stem(
            str(temp_mix),
            "vocals",
            None,
            providers=providers,
            precision=precision,
            verbose=False,
            progress=False,
        )
        info["separate_sec"] = round(time.perf_counter() - started, 2)
    finally:
        temp_mix.unlink(missing_ok=True)
    info["ep"] = _active_onnx_ep(providers[0])
    info["vocals"] = vocals
    return info


def _run_mini_bsrofo(mix, force_cpu: bool) -> dict[str, Any]:
    import torch
    from transformers import AutoModel

    info: dict[str, Any] = {}
    device = "cpu" if force_cpu or not torch.cuda.is_available() else "cuda"
    load_started = time.perf_counter()
    model = AutoModel.from_pretrained(MINI_BSROFO_REPO, trust_remote_code=True)
    model = model.to(device).eval()
    info["load_sec"] = round(time.perf_counter() - load_started, 2)
    params = sum(p.numel() for p in model.parameters())
    info["extra"] = f"{params / 1e6:.1f}M params"
    info["model_mb"] = round(params * 4 / 2**20, 1)
    info["ep"] = f"torch:{device}"

    waveform = torch.from_numpy(mix).to(device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.no_grad():
        stems = model.separate(
            waveform,
            chunk_size=MODEL_SAMPLE_RATE * 6,
            overlap_size=MODEL_SAMPLE_RATE * 3,
            gap_size=0,
            batch_size=2,
            verbose=False,
        )
    info["separate_sec"] = round(time.perf_counter() - started, 2)
    # 모델 카드 기준 스템 순서는 (bass, drums, other, vocal) — 마지막 행이 보컬이다.
    info["vocals"] = stems[-1].detach().float().cpu().numpy()
    if device == "cuda":
        info["torch_alloc_peak_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    del stems, waveform, model
    if device == "cuda":
        torch.cuda.empty_cache()
    return info


def _run_umx(mix, force_cpu: bool) -> dict[str, Any]:
    import torch
    from openunmix import predict, utils

    info: dict[str, Any] = {}
    device = torch.device("cpu" if force_cpu or not torch.cuda.is_available() else "cuda")
    info["ep"] = f"torch:{device.type}"
    # 분리기 생성을 타이머 **밖으로** 뺀다 — 첫 호출은 zenodo 가중치 다운로드(실측 4분 44초)를
    # 포함해서, 안에 두면 분리 시간이 아니라 회선 속도를 재게 된다.
    load_started = time.perf_counter()
    # niter=0 필수. load_separator의 기본 niter=1은 다채널 위너 필터 EM을 한 번 돌리는데,
    # 우리 입력은 24kHz **모노**를 스테레오로 복제한 파형이라 좌우가 완전히 동일하다 →
    # 공간 공분산 행렬이 특이행렬이 되어 역행렬에서 NaN이 나온다(실측: 스템 전체가 NaN,
    # 정렬은 그대로 돌아 MAE 85초짜리 «성공» 런을 만들었다). niter=0이면 EM을 건너뛰고
    # 크기 추정 × 믹스 위상으로 끝난다.
    separator = utils.load_separator(
        model_str_or_path="umxl",
        targets=["vocals"],
        niter=0,
        residual=True,
        device=device,
        pretrained=True,
    )
    separator.freeze()
    separator.to(device)
    info["load_sec"] = round(time.perf_counter() - load_started, 2)
    params = sum(p.numel() for p in separator.parameters())
    info["extra"] = f"{params / 1e6:.1f}M params (vocals 타깃만 로드)"
    info["model_mb"] = round(params * 4 / 2**20, 1)

    waveform = torch.from_numpy(mix).unsqueeze(0)  # (batch, channels, samples)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    estimates = predict.separate(
        audio=waveform,
        rate=MODEL_SAMPLE_RATE,
        separator=separator,
        device=device,
    )
    info["separate_sec"] = round(time.perf_counter() - started, 2)
    info["vocals"] = estimates["vocals"][0].detach().float().cpu().numpy()
    if device.type == "cuda":
        info["torch_alloc_peak_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        torch.cuda.empty_cache()
    return info


def _dispatch(spec: _LightSpec, mix, output_dir: Path, models_dir: Path, force_cpu: bool):
    if spec.kind == "demucs_onnx":
        return _run_demucs_onnx(spec, mix, output_dir, models_dir, force_cpu)
    if spec.kind == "mini_bsrofo":
        return _run_mini_bsrofo(mix, force_cpu)
    if spec.kind == "umx":
        return _run_umx(mix, force_cpu)
    raise RuntimeError(f"알 수 없는 분리 후보 종류: {spec.kind}")


def _worker(args: argparse.Namespace) -> int:
    cudnn = _prepare_cuda_dlls()
    spec = SPECS[args.model]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(args.models_dir)

    mix = _load_mix_44k_stereo(Path(args.audio))
    if args.limit_sec:
        mix = mix[:, : int(args.limit_sec * MODEL_SAMPLE_RATE)]
    audio_sec = mix.shape[1] / MODEL_SAMPLE_RATE

    with _GpuMemSampler() as sampler:
        info = _dispatch(spec, mix, output_dir, models_dir, force_cpu=args.cpu)
    vocals = info.pop("vocals")
    _write_stems(output_dir, mix, vocals)

    torch_peak = info.pop("torch_alloc_peak_mb", None)
    vram_note = sampler.note
    if vram_note is None:
        vram_note = (
            "vram_peak_mb=torch allocator 피크"
            if torch_peak is not None
            else "vram_peak_mb=장치 사용량 증가분 추정(공유 GPU면 오염됨)"
        )
        if sampler.baseline_mb is not None:
            vram_note += f", device_baseline={sampler.baseline_mb}MB"
    payload: dict[str, Any] = {
        "model": spec.adapter_name,
        "audio_sec": round(audio_sec, 2),
        "vram_peak_mb": torch_peak if torch_peak is not None else sampler.delta_mb,
        "vram_device_peak_mb": sampler.peak_mb,
        "vram_note": vram_note,
        **info,
    }
    if cudnn and spec.kind == "demucs_onnx":
        try:
            from importlib.metadata import version

            payload["cudnn"] = f"nvidia-cudnn-cu12 {version('nvidia-cudnn-cu12')} (preload)"
        except Exception:
            payload["cudnn"] = "preloaded"

    print(json.dumps(payload, ensure_ascii=False))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Internal worker for lightweight separator adapters")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model", choices=sorted(SPECS), required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--models-dir", default=str(LIGHT_MODEL_DIR))
    parser.add_argument("--cpu", action="store_true", help="CPU만 사용(실행 프로바이더 강제)")
    parser.add_argument("--limit-sec", type=float, default=0.0, help="앞에서 N초만 처리(CPU 프로브용)")
    args = parser.parse_args()
    if not args.worker:
        parser.error("이 모듈은 벤치 어댑터가 호출한다 — --worker 를 붙일 것")
    return args


if __name__ == "__main__":
    raise SystemExit(_worker(_parse_args()))
