"""RoFormer-family separator adapters for the alignment benchmark.

This module deliberately keeps RoFormer dependencies out of the benchmark interpreter.
``audio-separator`` and its CUDA PyTorch build live in ``benchmark/.venv-sep``;
the adapters invoke this file again in that interpreter for model download and
separation.  The main benchmark venv therefore remains untouched.

The RoFormer checkpoints are single-target vocal models.  Their instrumental
stem is the mixture-minus-vocals residual and is reported as such in the
``note`` field returned to the harness.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPO_ROOT / "benchmark"
MODEL_DIR = BENCHMARK_DIR / "models"
SEPARATOR_PYTHON = BENCHMARK_DIR / ".venv-sep" / "Scripts" / "python.exe"


@dataclass(frozen=True)
class _ModelSpec:
    adapter_name: str
    checkpoint_name: str
    checkpoint_url: str
    config_name: str
    config_url: str


KIMFT_MELBAND = _ModelSpec(
    adapter_name="kimft-melband",
    checkpoint_name="MelBandRoformer.ckpt",
    checkpoint_url=(
        "https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/"
        "MelBandRoformer.ckpt"
    ),
    config_name="config_vocals_mel_band_roformer_kim.yaml",
    config_url=(
        "https://raw.githubusercontent.com/TRvlvr/application_data/main/"
        "mdx_model_data/mdx_c_configs/config_vocals_mel_band_roformer_kim.yaml"
    ),
)

BS_VIPERX = _ModelSpec(
    adapter_name="bs-viperx",
    checkpoint_name="model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    checkpoint_url=(
        "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/"
        "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
    ),
    config_name="model_bs_roformer_ep_317_sdr_12.9755.yaml",
    config_url=(
        "https://raw.githubusercontent.com/TRvlvr/application_data/main/"
        "mdx_model_data/mdx_c_configs/model_bs_roformer_ep_317_sdr_12.9755.yaml"
    ),
)

MODEL_SPECS = {
    KIMFT_MELBAND.adapter_name: KIMFT_MELBAND,
    BS_VIPERX.adapter_name: BS_VIPERX,
}


@dataclass
class _SeparationOut:
    """Structural equivalent of benchmark_alignment.SeparationOut.

    The harness uses attributes and ``meta()``, rather than ``isinstance``.
    Keeping this local avoids importing the running benchmark script as a
    second module under a different name.
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


class _VramProbe:
    """Match the harness' subprocess-aware device-memory sampling contract."""

    def __init__(self, interval: float = 0.25):
        self.interval = interval
        self.process_peak_mb: float | None = None
        self.device_peak_mb: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._torch: Any = None
        self._device_peak_bytes = 0

    def _sample(self) -> None:
        while not self._stop.is_set():
            try:
                free, total = self._torch.cuda.mem_get_info()
                self._device_peak_bytes = max(self._device_peak_bytes, total - free)
            except Exception:
                return
            self._stop.wait(self.interval)

    def __enter__(self) -> _VramProbe:
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


def _require_separator_python() -> Path:
    if SEPARATOR_PYTHON.is_file():
        return SEPARATOR_PYTHON
    raise RuntimeError(
        "RoFormer separator environment is missing: "
        f"expected {SEPARATOR_PYTHON}. Install audio-separator in benchmark/.venv-sep "
        "without modifying the main .venv."
    )


def _prepare_benchmark_input(audio_path: Path, work_dir: Path) -> Path:
    """Use the same 24 kHz mono input path as the existing Demucs benchmark adapter."""
    from everyric2.audio.loader import AudioLoader
    from everyric2.config.settings import get_settings

    prepared = work_dir / "roformer_input.wav"
    AudioLoader(get_settings().audio).load(audio_path).to_file(prepared)
    return prepared


class _RoformerSeparator:
    """Subprocess adapter around the isolated audio-separator installation."""

    spec: _ModelSpec
    name: str

    def __init__(self, spec: _ModelSpec):
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
                timeout=1800,
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
        return _SeparationOut(
            vocals_path=vocals,
            inst_path=inst,
            elapsed_sec=round(elapsed, 2),
            vram_peak_mb=vram_alloc,
            vram_device_peak_mb=vram_reserved,
            note="vocals predicted by single-target RoFormer; inst is mixture-minus-vocals residual"
            + ("; vram_device_peak_mb는 워커 reserved 피크" if vram_reserved is not None else ""),
        )


class KimFTMelBandSeparator(_RoformerSeparator):
    name = KIMFT_MELBAND.adapter_name

    def __init__(self) -> None:
        super().__init__(KIMFT_MELBAND)


class BSViperxSeparator(_RoformerSeparator):
    name = BS_VIPERX.adapter_name

    def __init__(self) -> None:
        super().__init__(BS_VIPERX)


class HtdemucsFTSeparator:
    """Existing Demucs subprocess path with only ``-n htdemucs_ft`` changed."""

    name = "htdemucs_ft"
    demucs_model = "htdemucs_ft"

    def separate(self, audio_path: Path, work_dir: Path) -> _SeparationOut:
        import torch

        from everyric2.audio.separator import VocalSeparator
        from everyric2.config.settings import get_settings

        work_dir.mkdir(parents=True, exist_ok=True)
        audio_cfg = get_settings().audio.model_copy(update={"temp_dir": work_dir})
        separator = VocalSeparator(audio_cfg)
        if not separator.is_available():
            raise RuntimeError("demucs is not installed in the benchmark interpreter")

        with _VramProbe() as probe:
            started = time.perf_counter()
            result = separator.separate_file(
                audio_path,
                model=self.demucs_model,
                use_gpu=torch.cuda.is_available(),
            )
            elapsed = time.perf_counter() - started
        raw_dir = work_dir / "demucs_output" / self.demucs_model / "demucs_input"
        vocals = work_dir / "vocals.wav"
        inst = work_dir / "inst.wav"
        note = None
        try:
            if (raw_dir / "vocals.wav").is_file() and (raw_dir / "no_vocals.wav").is_file():
                shutil.move(str(raw_dir / "vocals.wav"), str(vocals))
                shutil.move(str(raw_dir / "no_vocals.wav"), str(inst))
            else:
                result.vocals.to_file(vocals)
                result.accompaniment.to_file(inst)
                note = "demucs raw stems not found; wrote loader-decoded (mono) stems instead"
        finally:
            shutil.rmtree(work_dir / "demucs_output", ignore_errors=True)
        return _SeparationOut(
            vocals_path=vocals,
            inst_path=inst,
            elapsed_sec=round(elapsed, 2),
            vram_peak_mb=probe.process_peak_mb,
            vram_device_peak_mb=probe.device_peak_mb,
            note=note,
        )


def register(separator_registry: dict) -> None:
    """Register benchmark separator classes without importing the harness module."""
    separator_registry.update(
        {
            KimFTMelBandSeparator.name: KimFTMelBandSeparator,
            BSViperxSeparator.name: BSViperxSeparator,
            HtdemucsFTSeparator.name: HtdemucsFTSeparator,
        }
    )


def _download(url: str, destination: Path) -> None:
    """Download once with an atomic final rename; requests is isolated in the sep venv."""
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
    # requests transparently decompresses encoded HTTP responses, so their
    # Content-Length is not comparable to the bytes written to disk.
    if expected and not content_encoded and written != expected:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"incomplete download for {destination.name}: {written}/{expected} bytes")
    partial.replace(destination)


def _load_roformer(spec: _ModelSpec, models_dir: Path, output_dir: Path, force_cpu: bool):
    """Instantiate audio-separator's RoFormer MDXC backend from explicit model URLs."""
    if force_cpu:
        # Must be set before importing torch so CPU validation cannot touch the active GPU.
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    import torch
    import yaml
    from audio_separator.separator.architectures.mdxc_separator import MDXCSeparator

    checkpoint = models_dir / spec.checkpoint_name
    config_path = models_dir / spec.config_name
    _download(spec.checkpoint_url, checkpoint)
    _download(spec.config_url, config_path)
    with config_path.open(encoding="utf-8") as handle:
        model_data = yaml.load(handle, Loader=yaml.FullLoader)
    if not isinstance(model_data, dict):
        raise RuntimeError(f"invalid YAML configuration: {config_path}")
    model_data["is_roformer"] = True

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
        "segment_size": 256,
        "override_model_segment_size": False,
        # RoFormer backend clamps an over-large overlap step to one model chunk.
        "overlap": 8,
        "batch_size": 1,
        "pitch_shift": 0,
    }
    return MDXCSeparator(common_config=common_config, arch_config=arch_config), device


def _worker(args: argparse.Namespace) -> int:
    spec = MODEL_SPECS[args.model]
    separator, device = _load_roformer(
        spec,
        Path(args.models_dir),
        Path(args.output_dir),
        force_cpu=args.action == "verify",
    )
    first_parameter = next(separator.model_run.parameters())
    if args.action == "verify":
        print(
            json.dumps(
                {
                    "model": spec.adapter_name,
                    "device": str(device),
                    "parameters": sum(parameter.numel() for parameter in separator.model_run.parameters()),
                    "first_parameter_shape": list(first_parameter.shape),
                }
            )
        )
        return 0

    output_dir = Path(args.output_dir)
    # 스템 키는 모델 카드마다 다르다 — Kim FT MelBand는 보컬 여집합 스템을 "Instrumental"로
    # 부르므로(실측: 기본 파일명 `..._(Instrumental)_MelBandRoformer.wav`로 떨어짐) 흔한 표기를
    # 전부 건다. 모르는 키는 audio-separator가 조용히 무시한다.
    outputs = separator.separate(
        args.audio,
        custom_output_names={
            "vocals": "vocals",
            "Vocals": "vocals",
            "other": "inst",
            "Other": "inst",
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
        elif ("inst" in low or "no_vocal" in low or "accompaniment" in low) and not inst_path.exists():
            path.replace(inst_path)
    if not (vocals_path.exists() and inst_path.exists()):
        raise RuntimeError(f"unexpected RoFormer output paths: {outputs}")
    # 워커 프로세스 내부 VRAM 피크 — 곡 길이별 분리 VRAM 곡선(3090 예산 판정)의 원료.
    # allocator 피크(alloc)와 예약 피크(reserved, OS 관점 상한)를 함께 싣는다.
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
    parser = argparse.ArgumentParser(description="Internal worker for RoFormer benchmark adapters")
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
