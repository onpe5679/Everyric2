"""BS-PolarFormer(fp16) 보컬 분리 백엔드 — 프로드 이식.

벤치 실측(scripts/bench_adapters/separators_quality.py::BS_POLARFORMER_FP16, 근거 문서
docs/research/2026-07-30-model-replacement/ust-precision-comparison.md)에서 극한곡(합성보컬 등)
정렬 정확도가 무분리 대비 +26.7pp(47.6 -> 74.3) 오른 후보를 everyric2 서버로 옮긴 것이다.
아키텍처는 BS-RoFormer + PoPE(극좌표 위치 임베딩, arXiv 2509.10534) — audio-separator 내장
BSRoformer가 모르는 ``use_pope`` 인자라 MSST(Music-Source-Separation-Training) 원본 모델 코드를
핀 커밋에서 받아 쓰고, ``audio_separator``의 ``RoformerLoader.load_model``을 가로채 그 모델을
대신 돌려준다. forward만 fp16 autocast로 돌리고 가중치는 fp32로 유지한다(``model.half()``로
통째 캐스팅하면 STFT/iSTFT가 half를 거부하거나 정밀도가 무너진다).

벤치와의 결정적 차이 — 모델 조달
---------------------------------
벤치는 격리된 ``benchmark/.venv-sep`` 서브프로세스에서 매 분리마다 새로 실행되므로, 체크포인트/
설정/MSST 모델 코드가 없으면 그 자리에서 내려받고(``_download``, ``_resolve_asset``,
``_ensure_msst_sources``), ``RoformerLoader`` monkeypatch도 프로세스가 매번 새로 뜨니 영구히
남아도 안전하다. everyric2 서버는 상주 프로세스이고 사용자 요청 경로다:

- 자산이 없다고 그 자리에서 수 GB를 받기 시작하면 요청이 임의로 오래 막힌다.
- monkeypatch가 프로세스 수명 내내 남으면 이후 audio_separator를 쓰는 다른 경로(현재는 없지만)를
  전부 오염시킨다.
- 어느 분리기가 실제로 돌았는지 불명확한 폴백은 정렬 결과 해석을 불가능하게 만든다
  (VocalSeparator._separate_polarformer 쪽 정책 — everyric2/audio/separator.py).

그래서 이 모듈은 자산을 **절대 다운로드하지 않는다** — 없으면 ``require_available()``이
무엇이 없는지 나열한 명확한 예외를 던진다. monkeypatch도 컨텍스트 매니저로 감싸 분리가 끝나면
``RoformerLoader.load_model``을 원래대로 되돌린다.
"""

from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

BACKEND_NAME = "bs-polarformer-fp16"

# scripts/bench_adapters/separators_quality.py::BS_POLARFORMER의 체크포인트/설정 자산과
# 같은 파일명 — 벤치가 이미 내려받아 둔 benchmark/models 디렉터리를 그대로
# AudioSettings.separator_model_dir로 가리켜도 동작하게 하기 위함이다.
_CHECKPOINT_NAME = "model_bs_polarformer_float16.ckpt"
_CONFIG_NAME = "model_bs_polarformer_float16.yaml"

# PolarFormer 모델 코드(BSRoformer, use_pope=True)는 MSST 원본 저장소의 핀 커밋에서만 받는다.
# main을 따라가면 어느 날 모델 정의가 바뀌어 같은 체크포인트가 안 붙는 사고가 난다 — 벤치와
# 동일 커밋을 써서 비교 가능성을 지킨다(scripts/bench_adapters/separators_quality.py::MSST_COMMIT).
_MSST_COMMIT = "e247dfe4abc1f17c69dff719207fe045dc04413a"
_MSST_VENDOR_SUBDIR = f"msst_src_{_MSST_COMMIT[:8]}"
_MSST_PACKAGE_RELATIVE = Path("models") / "bs_roformer"
_MSST_ATTEND_FILE = "attend.py"
_MSST_MODEL_FILE = "bs_roformer.py"

# audio-separator MDXCSeparator.demix()의 is_roformer 분기가 실제로 쓰는 "overlap"(초 단위 홉
# 스텝). 기존 RoFormer 어댑터들과 같은 값 — 비교 가능성을 위해 건드리지 않는다.
_OVERLAP_SEC = 8.0
_SEGMENT_SIZE = 256
_BATCH_SIZE = 1


class PolarFormerBackendError(Exception):
    """bs-polarformer-fp16 백엔드 관련 오류의 기반 클래스."""


class PolarFormerUnavailableError(PolarFormerBackendError):
    """의존성 또는 모델 자산이 없어(혹은 CUDA가 없어) 이 백엔드를 쓸 수 없을 때.

    조용한 폴백 금지 정책의 핵심 예외 — 호출자는 이걸 잡아 다른 분리기로 조용히 넘어가지
    말고 표면화해야 한다(everyric2/audio/separator.py::VocalSeparator가 SeparatorBackend
    UnavailableError로 감싸 그대로 전파한다).
    """


@dataclass
class PolarFormerOutput:
    """분리 결과 파일 경로 + 계측치.

    everyric2 서버 계약(SeparationResult)으로의 변환은 호출자(everyric2.audio.separator)의
    몫이다 — 이 모듈은 벤치의 SeparationOut을 흉내내지 않는다(과제 지시: 벤치 어댑터는 로직
    참조용이지 그대로 복붙할 대상이 아니다).
    """

    vocals_path: Path
    inst_path: Path
    elapsed_sec: float
    vram_alloc_peak_mb: float | None = None
    vram_reserved_peak_mb: float | None = None


def _asset_paths(models_dir: Path) -> tuple[Path, Path]:
    return models_dir / _CHECKPOINT_NAME, models_dir / _CONFIG_NAME


def _vendor_paths(models_dir: Path) -> tuple[Path, Path, Path, Path]:
    """(벤더 루트, __init__.py 마커, attend.py, bs_roformer.py)."""
    vendor_dir = models_dir / _MSST_VENDOR_SUBDIR
    package_dir = vendor_dir / _MSST_PACKAGE_RELATIVE
    return (
        vendor_dir,
        package_dir / "__init__.py",
        package_dir / _MSST_ATTEND_FILE,
        package_dir / _MSST_MODEL_FILE,
    )


def _is_nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _missing_reasons(models_dir: Path) -> list[str]:
    """없는 자산/의존성을 사람이 읽을 문장 목록으로 돌려준다(빈 리스트=전부 있음).

    __init__.py 마커는 검사하지 않는다 — 내용 없는 순수 패키징 절차물이라 필요하면
    ``_ensure_vendor_package_marker``가 그 자리에서 만든다(다운로드가 아니다).
    """
    reasons: list[str] = []
    checkpoint, config = _asset_paths(models_dir)
    for label, path in (("checkpoint", checkpoint), ("config", config)):
        if not _is_nonempty_file(path):
            reasons.append(f"{label} missing: {path}")
    _, _init, attend, model_file = _vendor_paths(models_dir)
    for label, path in (("MSST attend.py", attend), ("MSST bs_roformer.py", model_file)):
        if not _is_nonempty_file(path):
            reasons.append(f"{label} missing: {path}")
    try:
        import audio_separator  # noqa: F401
    except ImportError as exc:
        reasons.append(f"'audio-separator' package not importable: {exc}")
    return reasons


def dependencies_and_assets_available(models_dir: Path) -> bool:
    """의존성 + 모델 자산이 모두 있으면 True.

    절대 다운로드를 시도하지 않는다 — 모듈 docstring의 벤치와의 차이 참조. CUDA 존재 여부는
    여기서 검사하지 않는다(장치 상태는 매 요청 시점에 바뀔 수 있어 "가용" 캐시로 두기보다
    ``separate()`` 호출 시점에 바로 검사하는 편이 정확하다).
    """
    return not _missing_reasons(models_dir)


def require_available(models_dir: Path) -> None:
    """자산/의존성이 없으면 무엇이 없는지 나열한 PolarFormerUnavailableError를 던진다."""
    reasons = _missing_reasons(models_dir)
    if reasons:
        raise PolarFormerUnavailableError(
            f"{BACKEND_NAME} backend is not provisioned. Missing:\n  - "
            + "\n  - ".join(reasons)
            + f"\nProvision the checkpoint/config under {models_dir} and the MSST source at "
            f"pinned commit {_MSST_COMMIT} (see scripts/bench_adapters/separators_quality.py "
            "BS_POLARFORMER for the exact upstream URLs) before selecting "
            f"separator_backend={BACKEND_NAME}. This backend never downloads assets itself."
        )


def _ensure_vendor_package_marker(models_dir: Path) -> Path:
    """빈 ``__init__.py`` 마커만 필요 시 만든다.

    실제 모델 코드(attend.py/bs_roformer.py)는 사전 조달 대상이라 여기서 받지 않는다 —
    마커는 내용이 없어 "어느 모델이 돌았는지" 판단에 영향을 주지 않는 순수 파이썬 패키징
    절차(``models.bs_roformer`` 를 임포트 가능한 패키지로 만들 뿐)라 다운로드로 취급하지 않는다.
    """
    vendor_dir, init_file, _, _ = _vendor_paths(models_dir)
    init_file.parent.mkdir(parents=True, exist_ok=True)
    if not init_file.exists():
        init_file.write_text(
            "# MSST models.bs_roformer 축소 재현 — bs_roformer.py의 절대 임포트"
            "(from models.bs_roformer.attend import Attend)를 성립시키기 위한 빈 패키지 마커.\n",
            encoding="utf-8",
        )
    return vendor_dir


def _wrap_autocast_fp16(model: Any) -> Any:
    """가중치는 fp32로 두고 forward만 fp16 autocast로 돌리는 래퍼.

    scripts/bench_adapters/separators_quality.py::_wrap_autocast_fp16과 동일 로직.
    ``model.half()``로 통째 캐스팅하면 STFT/iSTFT가 half를 거부하거나 정밀도가 무너진다 —
    autocast는 STFT를 fp32로 유지한 채 matmul만 fp16으로 내려 실제 속도·VRAM 이득을 가져간다.
    """
    import torch
    from torch import nn

    class _AutocastWrapper(nn.Module):
        def __init__(self, inner: nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            with torch.autocast("cuda", dtype=torch.float16):
                out = self.inner(*args, **kwargs)
            return out.float() if torch.is_tensor(out) else out

    return _AutocastWrapper(model)


def _build_pope_model(model_config: dict, checkpoint: Path, device: Any, vendor_dir: Path) -> Any:
    """MSST 원본 BSRoformer(use_pope=True)를 만들고 체크포인트를 얹는다.

    audio-separator 내장 BSRoformer는 ``use_pope`` 인자를 모르기 때문에 이 경로만 MSST
    구현을 쓴다. fp16 체크포인트는 ``load_state_dict``가 fp32 파라미터로 캐스팅해 복사하므로
    별도 변환이 필요 없다.
    """
    import sys

    import torch

    vendor_dir_str = str(vendor_dir)
    if vendor_dir_str not in sys.path:
        sys.path.insert(0, vendor_dir_str)
    from models.bs_roformer.bs_roformer import BSRoformer  # type: ignore[import-not-found]

    model = BSRoformer(**model_config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    return model.to(device).eval()


@contextlib.contextmanager
def _patched_loader_for_pope(model_config: dict, checkpoint: Path, vendor_dir: Path):
    """``RoformerLoader.load_model``을 PolarFormer 전용 로더로 임시 교체한다.

    벤치(scripts/bench_adapters/separators_quality.py::_patch_loader_for_pope)는 분리마다
    새 서브프로세스라 patch가 영구히 남아도 안전하다. everyric2 서버는 상주 프로세스라 이
    patch가 계속 남으면 이후 audio_separator를 쓰는 다른 어떤 경로도 PolarFormer 로더로
    오염된다 — 그래서 컨텍스트 종료 시(성공/실패 무관) 원래 메서드로 반드시 되돌린다.
    """
    from audio_separator.separator.roformer.model_loading_result import (
        ImplementationVersion,
        ModelLoadingResult,
    )
    from audio_separator.separator.roformer.roformer_loader import RoformerLoader

    original_load_model = RoformerLoader.load_model

    def _load_model(self: Any, model_path: Any, config: Any, device: Any = "cpu") -> Any:
        model = _build_pope_model(model_config, checkpoint, device, vendor_dir)
        result = ModelLoadingResult.success_result(
            model=model, implementation=ImplementationVersion.NEW, config=config
        )
        result.add_model_info("model_type", "bs_roformer_pope")
        result.add_model_info("loading_method", "msst-vendored")
        result.add_model_info("device", str(device))
        return result

    RoformerLoader.load_model = _load_model
    try:
        yield
    finally:
        RoformerLoader.load_model = original_load_model


class _VramProbe:
    """CUDA 프로세스 할당 피크 계측 — 로그 전용(everyric2/audio/separator.py의 기존 htdemucs
    경로에는 VRAM 계측이 없어 SeparationResult 계약에는 반영하지 않는다).

    벤치(scripts/bench_adapters/separators_quality.py::_worker)는 alloc/reserved 두 피크를
    찍는다; 여기서는 같은 두 지표를 쓰되 벤치의 장치 전역(mem_get_info) 백그라운드 샘플링
    스레드는 들이지 않는다 — 다른 프로세스의 VRAM 사용까지 보는 것은 이 서버 로그의 목적을
    넘는다(그 계측이 필요해지면 scripts/bench_adapters/separators_quality.py._VramProbe와
    같은 이름의 클래스가 있으니 참조).
    """

    def __init__(self) -> None:
        self.alloc_peak_mb: float | None = None
        self.reserved_peak_mb: float | None = None
        self._torch: Any = None

    def __enter__(self) -> _VramProbe:
        try:
            import torch

            if not torch.cuda.is_available():
                return self
            self._torch = torch
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            self._torch = None
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._torch is None:
            return
        try:
            self.alloc_peak_mb = round(self._torch.cuda.max_memory_allocated() / 2**20, 1)
            self.reserved_peak_mb = round(self._torch.cuda.max_memory_reserved() / 2**20, 1)
        except Exception:
            pass


def _normalize_outputs(outputs: list, output_dir: Path) -> tuple[Path, Path]:
    """audio-separator가 돌려준 출력 파일명을 vocals.wav/inst.wav로 표준화한다.

    스템 키는 모델 카드마다 다르다 — custom_output_names로 대부분 잡히지만, 그래도 남는
    기본 파일명은 이름 휴리스틱으로 리네임한다(scripts/bench_adapters/separators_quality.py
    ::_worker와 동일 로직).
    """
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
        elif (
            "inst" in low or "other" in low or "no_vocal" in low or "accompaniment" in low
        ) and not inst_path.exists():
            path.replace(inst_path)
    return vocals_path, inst_path


def separate(
    input_path: Path,
    work_dir: Path,
    models_dir: Path,
    use_gpu: bool = True,
) -> PolarFormerOutput:
    """bs-polarformer-fp16으로 input_path를 분리해 work_dir에 vocals.wav/inst.wav를 쓴다.

    자산/CUDA 부재는 여기서 다시 검사한다(``require_available``) — 이중 호출 비용은 파일시스템
    stat 몇 개뿐이라, "먼저 확인했으니 여기선 안 본다"는 취약한 계약보다 이 함수 자체가
    스스로 안전한 편이 낫다.
    """
    import torch

    if not (use_gpu and torch.cuda.is_available()):
        raise PolarFormerUnavailableError(
            f"{BACKEND_NAME} requires CUDA (fp16 autocast forward has no CPU path); "
            f"use_gpu={use_gpu}, torch.cuda.is_available()={torch.cuda.is_available()}"
        )
    require_available(models_dir)

    import yaml

    device = torch.device("cuda")
    work_dir.mkdir(parents=True, exist_ok=True)
    checkpoint, config_path = _asset_paths(models_dir)
    with config_path.open(encoding="utf-8") as handle:
        model_data = yaml.load(handle, Loader=yaml.FullLoader)
    if not isinstance(model_data, dict):
        raise PolarFormerBackendError(f"invalid YAML configuration: {config_path}")
    model_data["is_roformer"] = True
    # 추론에서는 gradient checkpointing이 이득이 없고(no_grad), 재계산 비용만 든다.
    if isinstance(model_data.get("model"), dict):
        model_data["model"]["use_torch_checkpoint"] = False

    vendor_dir = _ensure_vendor_package_marker(models_dir)

    worker_logger = logging.getLogger(f"everyric2.audio.{BACKEND_NAME}")
    worker_logger.setLevel(logging.WARNING)
    common_config = {
        "logger": worker_logger,
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
        "output_dir": str(work_dir),
        "normalization_threshold": 0.9,
        "amplification_threshold": 0.0,
        "output_single_stem": None,
        "invert_using_spec": False,
        "sample_rate": int(model_data["audio"]["sample_rate"]),
        "use_soundfile": True,
    }
    arch_config = {
        "segment_size": _SEGMENT_SIZE,
        "override_model_segment_size": False,
        "overlap": _OVERLAP_SEC,
        "batch_size": _BATCH_SIZE,
        "pitch_shift": 0,
    }

    from audio_separator.separator.architectures.mdxc_separator import MDXCSeparator

    started = time.perf_counter()
    with (
        _VramProbe() as probe,
        _patched_loader_for_pope(dict(model_data["model"]), checkpoint, vendor_dir),
    ):
        mdxc_separator = MDXCSeparator(common_config=common_config, arch_config=arch_config)
        mdxc_separator.model_run = _wrap_autocast_fp16(mdxc_separator.model_run).to(device).eval()
        outputs = mdxc_separator.separate(
            str(input_path),
            custom_output_names={
                "vocals": "vocals",
                "Vocals": "vocals",
                "other": "inst",
                "Other": "inst",
                "instrumental": "inst",
                "Instrumental": "inst",
            },
        )
    elapsed = time.perf_counter() - started

    vocals_path, inst_path = _normalize_outputs(outputs, work_dir)
    if not (vocals_path.exists() and inst_path.exists()):
        raise PolarFormerBackendError(f"unexpected {BACKEND_NAME} output paths: {outputs}")

    logger.info(
        "separator backend used: %s elapsed_sec=%.2f vram_alloc_peak_mb=%s vram_reserved_peak_mb=%s",
        BACKEND_NAME,
        elapsed,
        probe.alloc_peak_mb,
        probe.reserved_peak_mb,
    )
    return PolarFormerOutput(
        vocals_path=vocals_path,
        inst_path=inst_path,
        elapsed_sec=round(elapsed, 2),
        vram_alloc_peak_mb=probe.alloc_peak_mb,
        vram_reserved_peak_mb=probe.reserved_peak_mb,
    )
