"""One-process-per-job local bridge used by Everyric Studio.

The wire format is newline-delimited JSON so CEP can stream progress and terminate the child
process for cancellation.  No endpoint, token, upload, or network fallback exists here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from everyric2.local_runtime import (
    BRIDGE_PROTOCOL,
    RESULT_SCHEMA,
    RuntimePaths,
    configure_environment,
    health_report,
)


def _emit(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=False), flush=True)


def _read_request(path: Path) -> dict[str, Any]:
    request = json.loads(path.read_text(encoding="utf-8"))
    if request.get("protocol") not in (None, BRIDGE_PROTOCOL):
        raise ValueError(f"unsupported bridge protocol: {request.get('protocol')!r}")
    audio = Path(str(request.get("audioPath", ""))).expanduser().resolve()
    if not audio.is_file():
        raise ValueError(f"audioPath is not a file: {audio}")
    lyrics = request.get("lyrics")
    if not isinstance(lyrics, str) or not lyrics.strip():
        raise ValueError("lyrics must be a non-empty string")
    request["audioPath"] = str(audio)
    return request


def _paths(request: dict[str, Any]) -> RuntimePaths:
    root = Path(request["runtimeRoot"]) if request.get("runtimeRoot") else None
    models = Path(request["modelDir"]) if request.get("modelDir") else None
    return RuntimePaths.resolve(root, models)


def capabilities(request: dict[str, Any] | None = None) -> int:
    request = request or {}
    paths = _paths(request)
    configure_environment(paths)
    _emit("hello", protocol=BRIDGE_PROTOCOL, resultSchema=RESULT_SCHEMA)
    _emit("capabilities", **health_report(paths))
    return 0


def align(request_path: Path) -> int:
    request = _read_request(request_path)
    paths = _paths(request)
    configure_environment(paths)
    report = health_report(paths)
    if not report["ready"]:
        raise RuntimeError("local runtime is not ready: " + "; ".join(report["problems"]))

    _emit("hello", protocol=BRIDGE_PROTOCOL, resultSchema=RESULT_SCHEMA)

    # Import after configuring model/cache paths.  The worker module is intentionally reused as
    # the current adaptive core; the bridge contract prevents CEP from depending on server APIs.
    from everyric2.server.worker import _run_alignment

    result = _run_alignment(
        request["audioPath"],
        request["lyrics"],
        request.get("language") or None,
        line_meta=request.get("lineMeta"),
        on_stage=lambda stage: _emit("progress", stage=stage),
        min_depth=request.get("minDepth"),
        on_depth=lambda depth: _emit("depth", depth=depth),
        delete_audio=False,
    )
    document = {
        "schema": RESULT_SCHEMA,
        "engineVersion": report["engineVersion"],
        "result": result,
    }
    _emit("result", document=document)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="everyric2-bridge")
    subparsers = parser.add_subparsers(dest="command", required=True)
    capabilities_parser = subparsers.add_parser("capabilities")
    capabilities_parser.add_argument("--request", type=Path)
    align_parser = subparsers.add_parser("align")
    align_parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        if args.command == "capabilities":
            request = json.loads(args.request.read_text(encoding="utf-8")) if args.request else None
            return capabilities(request)
        return align(args.request)
    except Exception as exc:
        _emit("error", message=str(exc), exception=type(exc).__name__)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
