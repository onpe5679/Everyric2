"""Local-only runtime and model provisioning for Everyric Studio.

The CEP panel owns the user experience; this module owns deterministic local state.  It never
contacts an Everyric server and never writes into the installed CEP extension.  Downloads are
staged under the user data root, pinned by ``runtime_manifest.json``, and activated only after a
health check succeeds.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.resources
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BRIDGE_PROTOCOL = "everyric-local-bridge/v1"
RESULT_SCHEMA = "sync-document/v3"
STATE_SCHEMA = 1


def _manifest() -> dict[str, Any]:
    resource = importlib.resources.files("everyric2").joinpath("runtime_manifest.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def _default_root() -> Path:
    explicit = os.environ.get("EVERYRIC_HOME")
    if explicit:
        return Path(explicit).expanduser()
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "Everyric"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Everyric"
    base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "everyric"


@dataclass(frozen=True)
class RuntimePaths:
    root: Path
    models: Path
    hf_home: Path
    owsm_venv: Path
    state: Path

    @classmethod
    def resolve(cls, root: Path | None = None, models: Path | None = None) -> RuntimePaths:
        root = (root or _default_root()).expanduser().resolve()
        models = (
            (models or Path(os.environ.get("EVERYRIC_MODEL_DIR", root / "models")))
            .expanduser()
            .resolve()
        )
        return cls(
            root=root,
            models=models,
            hf_home=root / "huggingface",
            owsm_venv=root / "owsm-venv",
            state=root / "install-state.json",
        )

    @property
    def owsm_python(self) -> Path:
        if sys.platform == "win32":
            return self.owsm_venv / "Scripts" / "python.exe"
        return self.owsm_venv / "bin" / "python3"


def configure_environment(paths: RuntimePaths) -> None:
    """Point all engine components at the managed local state."""
    os.environ["EVERYRIC_HOME"] = str(paths.root)
    os.environ["EVERYRIC_MODEL_DIR"] = str(paths.models)
    os.environ["HF_HOME"] = str(paths.hf_home)
    os.environ["EVERYRIC_AUDIO_SEPARATOR_MODEL_DIR"] = str(paths.models)
    os.environ["EVERYRIC_ALIGNMENT_OWSM_PYTHON_PATH"] = str(paths.owsm_python)
    os.environ["EVERYRIC_ALIGNMENT_ENGINE"] = "adaptive"


def _platform_key() -> str:
    machine = platform.machine().lower()
    arch = (
        "x64"
        if machine in {"amd64", "x86_64"}
        else "arm64"
        if machine in {"arm64", "aarch64"}
        else machine
    )
    return f"{sys.platform}-{arch}"


def _emit(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=False), flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, target: Path, expected_sha256: str, expected_size: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if (
        target.is_file()
        and target.stat().st_size == expected_size
        and _sha256(target) == expected_sha256
    ):
        _emit("download-skip", file=target.name, bytes=expected_size)
        return

    part = target.with_suffix(target.suffix + ".part")
    start = part.stat().st_size if part.exists() else 0
    if start >= expected_size:
        if start == expected_size and _sha256(part) == expected_sha256:
            part.replace(target)
            return
        part.unlink()
        start = 0
    request = urllib.request.Request(url)
    if start:
        request.add_header("Range", f"bytes={start}-")
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - manifest URLs are pinned HTTPS
        if start and getattr(response, "status", 200) != 206:
            part.unlink(missing_ok=True)
            start = 0
        mode = "ab" if start else "wb"
        downloaded = start
        with part.open(mode) as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
                downloaded += len(chunk)
                _emit("download-progress", file=target.name, bytes=downloaded, total=expected_size)

    if part.stat().st_size != expected_size:
        raise RuntimeError(
            f"download size mismatch for {target.name}: {part.stat().st_size} != {expected_size}"
        )
    actual = _sha256(part)
    if actual != expected_sha256:
        raise RuntimeError(f"SHA-256 mismatch for {target.name}: {actual} != {expected_sha256}")
    part.replace(target)
    _emit("download-complete", file=target.name, bytes=expected_size)


def _snapshot(repo: dict[str, Any], paths: RuntimePaths) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface-hub is required by the desktop runtime") from exc

    _emit("model-download", model=repo["repoId"], revision=repo["revision"])
    snapshot_download(
        repo_id=repo["repoId"],
        revision=repo["revision"],
        allow_patterns=list(repo["allowPatterns"]),
        cache_dir=str(paths.hf_home / "hub"),
    )
    _emit("model-complete", model=repo["repoId"])


def _ensure_owsm_environment(paths: RuntimePaths) -> None:
    if paths.owsm_python.is_file():
        probe = subprocess.run(
            [
                str(paths.owsm_python),
                "-c",
                "import espnet2, sentencepiece, torch; "
                "assert torch.cuda.is_available(); "
                "assert (torch.ones(1, device='cuda') * 2).cpu().item() == 2",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode == 0:
            _emit("owsm-environment-ready", python=str(paths.owsm_python))
            return

    uv = os.environ.get("EVERYRIC_UV_PATH") or shutil.which("uv")
    if not uv:
        raise RuntimeError("uv is required to create the isolated OWSM environment")
    paths.owsm_venv.parent.mkdir(parents=True, exist_ok=True)
    _emit("owsm-environment-create", path=str(paths.owsm_venv))
    subprocess.run([uv, "venv", str(paths.owsm_venv), "--python", "3.11"], check=True)
    subprocess.run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(paths.owsm_python),
            "--index-url",
            "https://download.pytorch.org/whl/cu128",
            "torch==2.8.0+cu128",
            "torchaudio==2.8.0+cu128",
        ],
        check=True,
    )
    subprocess.run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(paths.owsm_python),
            "--extra-index-url",
            "https://download.pytorch.org/whl/cu128",
            "espnet==202511",
            "sentencepiece==0.2.0",
        ],
        check=True,
    )
    _emit("owsm-environment-ready", python=str(paths.owsm_python))


def provision(paths: RuntimePaths) -> dict[str, Any]:
    manifest = _manifest()
    platform_info = manifest["platforms"].get(_platform_key())
    if not platform_info or not platform_info.get("local"):
        reason = (platform_info or {}).get("reason", "No local runtime bundle is defined")
        raise RuntimeError(f"local engine is unsupported on {_platform_key()}: {reason}")

    for directory in (paths.root, paths.models, paths.hf_home):
        directory.mkdir(parents=True, exist_ok=True)
    configure_environment(paths)

    models = manifest["models"]
    _snapshot(models["omniasr"], paths)
    _snapshot(models["owsm"], paths)

    polar = models["polarformer"]
    for item in polar["files"]:
        _download(item["url"], paths.models / item["name"], item["sha256"], item["size"])
    vendor = polar["vendor"]
    vendor_root = paths.models / f"msst_src_{vendor['commit'][:8]}"
    base = f"https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/{vendor['commit']}/"
    for item in vendor["files"]:
        _download(
            base + item["path"],
            vendor_root / item["path"],
            item["sha256"],
            item["size"],
        )
    package_dir = vendor_root / "models" / "bs_roformer"
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "__init__.py").touch()

    _ensure_owsm_environment(paths)
    report = health_report(paths)
    if not report["ready"]:
        raise RuntimeError("runtime health check failed: " + "; ".join(report["problems"]))

    state = {
        "schemaVersion": STATE_SCHEMA,
        "engineVersion": manifest["engineVersion"],
        "bridgeProtocol": BRIDGE_PROTOCOL,
        "resultSchema": RESULT_SCHEMA,
        "platform": _platform_key(),
        "modelDir": str(paths.models),
        "healthy": True,
    }
    pending_state = paths.state.with_suffix(".json.tmp")
    pending_state.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending_state.replace(paths.state)
    _emit("ready", state=state)
    return state


def _nvidia_smi() -> dict[str, Any] | None:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    name, total, free, driver = [part.strip() for part in result.stdout.splitlines()[0].split(",")]
    return {"name": name, "vramTotalMb": int(total), "vramFreeMb": int(free), "driver": driver}


def health_report(paths: RuntimePaths) -> dict[str, Any]:
    configure_environment(paths)
    problems: list[str] = []
    manifest = _manifest()
    platform_info = manifest["platforms"].get(_platform_key())
    if not platform_info or not platform_info.get("local"):
        problems.append(
            (platform_info or {}).get("reason", f"No local runtime for {_platform_key()}")
        )
    gpu = _nvidia_smi()
    minimum_vram = int((platform_info or {}).get("minimumVramMb", 0))
    if gpu is None:
        problems.append("NVIDIA GPU or driver was not detected")
    elif minimum_vram and gpu["vramTotalMb"] < minimum_vram:
        problems.append(f"GPU VRAM {gpu['vramTotalMb']}MB is below the {minimum_vram}MB minimum")
    cuda: dict[str, Any] = {"available": False}
    try:
        import torch

        cuda = {
            "available": bool(torch.cuda.is_available()),
            "torch": torch.__version__,
            "runtime": torch.version.cuda,
        }
        if torch.cuda.is_available():
            tensor = torch.ones(1, device="cuda") * 2
            cuda["smoke"] = float(tensor.cpu().item()) == 2.0
        else:
            problems.append("PyTorch CUDA is unavailable")
    except Exception as exc:
        problems.append(f"PyTorch CUDA probe failed: {exc}")

    try:
        from everyric2.alignment.omniasr_engine import OmniASREngine

        if not OmniASREngine().is_available():
            problems.append("omniASR model is not provisioned")
    except Exception as exc:
        problems.append(f"omniASR probe failed: {exc}")

    try:
        from everyric2.audio.polarformer_separator import dependencies_and_assets_available

        if not dependencies_and_assets_available(paths.models):
            problems.append("BS-PolarFormer dependencies or assets are missing")
    except Exception as exc:
        problems.append(f"BS-PolarFormer probe failed: {exc}")

    try:
        from everyric2.alignment.owsm_engine import _find_snapshot

        if _find_snapshot() is None:
            problems.append("OWSM model is not provisioned")
    except Exception as exc:
        problems.append(f"OWSM model probe failed: {exc}")

    if not paths.owsm_python.is_file():
        problems.append("OWSM isolated Python is missing")
    else:
        probe = subprocess.run(
            [
                str(paths.owsm_python),
                "-c",
                "import espnet2, sentencepiece, torch; "
                "assert torch.cuda.is_available(); "
                "assert (torch.ones(1, device='cuda') * 2).cpu().item() == 2",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode != 0:
            problems.append("OWSM isolated environment is unhealthy")

    return {
        "ready": not problems,
        "protocol": BRIDGE_PROTOCOL,
        "resultSchema": RESULT_SCHEMA,
        "engineVersion": manifest["engineVersion"],
        "platform": _platform_key(),
        "paths": {
            "root": str(paths.root),
            "models": str(paths.models),
            "hfHome": str(paths.hf_home),
            "owsmPython": str(paths.owsm_python),
        },
        "gpu": gpu,
        "cuda": cuda,
        "problems": problems,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="everyric2-runtime")
    parser.add_argument("command", choices=("status", "health", "provision", "repair"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--models", type=Path)
    args = parser.parse_args(argv)
    paths = RuntimePaths.resolve(args.root, args.models)
    try:
        if args.command in {"provision", "repair"}:
            provision(paths)
        else:
            report = health_report(paths)
            print(json.dumps(report, ensure_ascii=False))
            if args.command == "health" and not report["ready"]:
                return 2
        return 0
    except Exception as exc:
        _emit("error", message=str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
