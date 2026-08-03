"""NeMo Forced Aligner (NFA) bf16 variant -- VRAM-reduction attempt for ``nemo-nfa``.

Split into its own file rather than editing ``nemo_nfa.py`` in place: a Japanese-native NFA
effort was working in the same session, and ``nemo_nfa.py``'s language gating / lyrics handling
was the likely surface for that work. This module only *imports* nemo_nfa's language-agnostic
helpers (lyric splitting, line folding, chunk loading) -- read-only, no shared lines edited --
while the worker subprocess, which re-invokes ``Path(__file__)`` as
``python <this file> --worker``, gets its own copy here so it loads Conformer-CTC weights in
bfloat16 instead of ``nemo_nfa.py``'s fp32. The fp32 candidate's file, worker, and cache are
untouched by anything in this module.

Cast pattern mirrors ``owsm-ctc-v4-1b-bf16``/``omniasr-ctc-bf16``: restore the checkpoint on
CPU, cast to the target dtype there, move once to the GPU, ``empty_cache()``. ``input_signal``
stays fp32 -- NeMo's mel-spectrogram preprocessor runs ``torch.stft`` internally, which does not
accept half-precision input -- so only the Conformer encoder/decoder weights are cast; the
emission is cast back to fp32 before ``forced_align`` for the same near-blank-resolution reason
``hf_ctc._log_softmax`` casts.

Measured on the two required verification songs (``kimft-melband-fp16`` separator, run via
``--input-mode pron-hangul-local`` since nemo-nfa is ko-only):

* ``owv06htaoI8`` (ko): MAE 0.071s (fp32) -> 0.075s (bf16), AAE 0.348 -> 0.297, PCO 0.9206 ->
  0.9048 -- within tolerance. ``align_vram_peak_mb`` 1101.9 -> 558.2 (-49%),
  ``align_vram_device_peak_mb`` ~1352 -> ~718 (-47%).
* ``AOhFzDN3eMI`` (en): not a meaningful precision comparison -- nemo-nfa is ko-only, so even
  the fp32 baseline through the hangul-local workaround is already low_conf (MAE 0.711s); bf16's
  0.761s is dwarfed by that pre-existing collapse and isn't attributable to the dtype change.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


def _load_nemo_nfa_module() -> Any:
    """Import ``nemo_nfa.py`` by file path rather than as ``scripts.bench_adapters.nemo_nfa``.

    ``benchmark/.venv-nemo`` (where the worker subprocess below actually runs) has an unrelated
    pip-installed package that squats the name ``scripts`` in its site-packages -- a regular
    package (has ``__init__.py``) always wins name resolution over the repo's ``scripts``
    namespace package (no ``__init__.py``), regardless of ``sys.path`` order, so the normal
    ``from scripts.bench_adapters.nemo_nfa import ...`` silently resolves to the wrong module
    (or fails) inside that venv. File-path loading -- the same trick ``nemo_nfa.py``'s own
    ``_load_chunking`` already uses for ``everyric2.audio.chunking`` -- sidesteps the collision
    entirely and works identically in the main .venv and in benchmark/.venv-nemo. Safe because
    ``nemo_nfa.py``'s module-level imports are stdlib-only (torch/nemo are deferred inside
    functions), so it loads cleanly in either interpreter.
    """
    import importlib.util

    path = Path(__file__).resolve().parent / "nemo_nfa.py"
    spec = importlib.util.spec_from_file_location("_nemo_nfa_bf16_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load nemo_nfa helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_nemo_nfa = _load_nemo_nfa_module()
ALIGN_CHUNK_OVERLAP_SEC = _nemo_nfa.ALIGN_CHUNK_OVERLAP_SEC
ALIGN_CHUNK_SEC = _nemo_nfa.ALIGN_CHUNK_SEC
HF_FILENAME = _nemo_nfa.HF_FILENAME
HF_REPO_ID = _nemo_nfa.HF_REPO_ID
LOW_CONF_COVERAGE = _nemo_nfa.LOW_CONF_COVERAGE
REPO_ROOT = _nemo_nfa.REPO_ROOT
SUPPORTED_LANGUAGE = _nemo_nfa.SUPPORTED_LANGUAGE
TARGET_SAMPLE_RATE = _nemo_nfa.TARGET_SAMPLE_RATE
_base_language = _nemo_nfa._base_language
_is_alignment_character = _nemo_nfa._is_alignment_character
_line_results = _nemo_nfa._line_results
_load_chunking = _nemo_nfa._load_chunking
_quality_score = _nemo_nfa._quality_score
_require_python = _nemo_nfa._require_python
_split_lyrics = _nemo_nfa._split_lyrics

NAME = "nemo-nfa-bf16"
DTYPE = "bfloat16"


class NemoNFABf16Aligner:
    """VRAM-reduction attempt: bfloat16 Conformer-CTC weights.

    Structurally mirrors ``nemo_nfa.NemoNFAAligner.align``, but the worker subprocess below
    re-invokes *this* file, not ``nemo_nfa.py`` -- the fp32 candidate's worker never runs any
    code from here.
    """

    name = NAME

    def align(self, vocals_path: Path, lyrics: str, language: str) -> Any:
        from scripts.benchmark_alignment import AlignOut

        base = _base_language(language or "")
        if base and base != SUPPORTED_LANGUAGE:
            raise ValueError(f"{self.name} is a {SUPPORTED_LANGUAGE} candidate, not {language!r}")

        lines = _split_lyrics(lyrics)
        if not lines:
            raise ValueError("lyrics produced zero non-empty lines")

        python = _require_python()
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="nemonfabf16_") as tmp:
            request_path = Path(tmp) / "request.json"
            response_path = Path(tmp) / "response.json"
            request_path.write_text(
                json.dumps(
                    {
                        "audio": str(vocals_path),
                        "lines": lines,
                        "chunk_sec": ALIGN_CHUNK_SEC,
                        "overlap_sec": ALIGN_CHUNK_OVERLAP_SEC,
                        "repo_root": str(REPO_ROOT),
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            status = _run_worker(python, request_path, response_path, self.name)
            payload = json.loads(response_path.read_text(encoding="utf-8"))

        if payload.get("error"):
            raise RuntimeError(f"{self.name}: {payload['error']}")

        elapsed = time.perf_counter() - started
        line_results = _line_results(lines, payload["lines"], payload["audio_sec"])
        quality_score, quality_meta = _quality_score(line_results)
        return AlignOut(
            lines=line_results,
            elapsed_sec=round(elapsed, 2),
            vram_peak_mb=status.get("vram_alloc_peak_mb"),
            vram_device_peak_mb=status.get("vram_reserved_peak_mb"),
            quality_score=quality_score,
            meta={
                "model": HF_REPO_ID,
                "adapter": self.name,
                "language": SUPPORTED_LANGUAGE,
                "architecture": "NeMo EncDecCTCModelBPE (Conformer-CTC) + Viterbi forced alignment",
                "alignment_backend": payload["backend"],
                "vocab_unit": "SentencePiece BPE, targets built one source character at a time",
                "vocab_size": payload["vocab_size"],
                "frame_stride_sec": payload["frame_stride_sec"],
                "align_chunks": payload["chunks"],
                "align_chunk_sec": ALIGN_CHUNK_SEC,
                "align_chunk_overlap_sec": ALIGN_CHUNK_OVERLAP_SEC,
                "audio_sec": payload["audio_sec"],
                "model_load_sec": payload["load_sec"],
                "worker_align_sec": payload["align_sec"],
                "coverage_threshold": LOW_CONF_COVERAGE,
                "coverage_denominator": "letter/number-like source characters",
                "quality": quality_meta,
                "license": "Apache-2.0",
                "dtype": DTYPE,
            },
        )


def register(aligner_registry: dict) -> None:
    """Register the bf16 NeMo NFA variant in a harness aligner registry."""
    aligner_registry[NemoNFABf16Aligner.name] = NemoNFABf16Aligner


def _run_worker(python: Path, request: Path, response: Path, name: str) -> dict:
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    # NeMo chatters on import; the adapter only reads the trailing status JSON, but a
    # quiet worker keeps sweep logs legible.
    env.setdefault("NEMO_TESTING", "1")
    result = subprocess.run(
        [str(python), str(Path(__file__).resolve()), "--worker",
         "--request", str(request), "--response", str(response)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=1800, env=env,
    )
    if result.returncode != 0 or not response.is_file():
        raise RuntimeError(
            f"{name} worker failed ({result.returncode}):\n"
            f"stdout:\n{(result.stdout or '')[-4000:]}\nstderr:\n{(result.stderr or '')[-4000:]}"
        )
    for line in reversed((result.stdout or "").strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                break
    return {}


# ──────────────────────────────────────────────────────────────────────────
# Worker (runs inside benchmark/.venv-nemo -- nemo_toolkit[asr], torch cu128)
# ──────────────────────────────────────────────────────────────────────────


def _worker_align(request: dict) -> dict:
    import torch
    import torchaudio
    import torchaudio.functional as functional
    from huggingface_hub import hf_hub_download
    from nemo.collections.asr.models import ASRModel

    chunking = _load_chunking(request["repo_root"])
    lines: list[str] = request["lines"]
    chunk_sec = float(request["chunk_sec"])
    overlap_sec = float(request["overlap_sec"])

    waveform, sample_rate = torchaudio.load(request["audio"])
    waveform = waveform.mean(dim=0)
    if sample_rate != TARGET_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, TARGET_SAMPLE_RATE)
    waveform = waveform.to(torch.float32).contiguous()
    audio_sec = waveform.numel() / TARGET_SAMPLE_RATE

    load_started = time.perf_counter()
    checkpoint = hf_hub_download(HF_REPO_ID, HF_FILENAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Restore on CPU and only then move to the GPU. NeMo's ``restore_from`` deserializes fp32
    # weights regardless of ``map_location``, so requesting cuda directly still puts a full fp32
    # copy on the card before any cast happens. Casting on CPU first (same order as
    # owsm-ctc-v4-1b-bf16/omniasr-ctc-bf16) means the GPU only ever sees one copy of the weights.
    model = ASRModel.restore_from(checkpoint, map_location="cpu")
    model.eval()
    model = model.to(dtype=torch.bfloat16)
    model = model.to(device)
    if device == "cuda":
        torch.cuda.empty_cache()
    load_sec = round(time.perf_counter() - load_started, 2)

    tokenizer = model.tokenizer
    vocab_size = int(model.decoder.num_classes_with_blank)
    # NeMo CTC reserves the final index for blank (``num_classes_with_blank - 1``).
    blank_id = vocab_size - 1

    def encode_char(char: str) -> list[int] | None:
        """Token ids for one source character, or None when it is out of vocabulary."""
        ids = tokenizer.text_to_ids(char)
        pieces = tokenizer.ids_to_tokens(ids) if ids else []
        kept = [i for i, piece in zip(ids, pieces) if piece != "▁"]
        if not kept:
            return None
        unk = getattr(tokenizer, "unk_id", None)
        if unk is not None and any(i == unk for i in kept):
            return None
        return kept

    # ── targets: one entry per alignable source character ──────────────────
    token_ids: list[int] = []
    char_ranges: list[list[tuple[str, int, int] | None]] = []
    totals: list[int] = []
    for line in lines:
        entries: list[tuple[str, int, int] | None] = []
        total = 0
        for char in line:
            if not _is_alignment_character(char):
                continue
            total += 1
            ids = encode_char(char)
            if ids is None:
                entries.append(None)
                continue
            first = len(token_ids)
            token_ids.extend(ids)
            entries.append((char, first, len(token_ids)))
        char_ranges.append(entries)
        totals.append(total)
    if not token_ids:
        return {"error": "no in-vocabulary lyric characters"}

    # ── chunked emission ───────────────────────────────────────────────────
    def forward(chunk: torch.Tensor) -> torch.Tensor:
        # input_signal stays fp32: NeMo's preprocessor runs torch.stft internally, which does
        # not accept half-precision input, so only the encoder/decoder weights are cast.
        signal = chunk.unsqueeze(0).to(device=device, dtype=torch.float32)
        length = torch.tensor([chunk.numel()], device=device)
        with torch.inference_mode():
            log_probs, _, _ = model(input_signal=signal, input_signal_length=length)
        # Normalize back to fp32 for the same reason hf_ctc._log_softmax does: forced_align's
        # DP needs full resolution near the blank log-probability.
        return log_probs.float()

    total_samples = int(waveform.numel())
    windows = chunking.plan_chunk_windows(
        total_samples,
        int(chunk_sec * TARGET_SAMPLE_RATE),
        int(overlap_sec * TARGET_SAMPLE_RATE),
    )
    if len(windows) == 1:
        emission = forward(waveform)
    else:
        pieces = [forward(waveform[s:e].contiguous()).cpu() for s, e in windows]
        emission = chunking.stitch_chunk_outputs(pieces, windows, total_samples, frame_axis=1)

    if max(token_ids) >= emission.shape[-1]:
        return {"error": f"token id {max(token_ids)} exceeds emission width {emission.shape[-1]}"}

    targets = torch.tensor([token_ids], dtype=torch.int32, device=emission.device)
    aligned_tokens, scores = functional.forced_align(emission, targets, blank=blank_id)
    spans = functional.merge_tokens(aligned_tokens[0], scores[0], blank=blank_id)

    ratio = total_samples / emission.shape[1] / TARGET_SAMPLE_RATE
    out_lines: list[dict] = []
    for entries, total in zip(char_ranges, totals):
        segs: list[dict] = []
        span_scores: list[float] = []
        for entry in entries:
            if entry is None:
                continue
            char, first, last = entry
            covered = spans[first:last]
            if not covered:
                continue
            span_scores.extend(float(span.score) for span in covered)
            segs.append({
                "t": char,
                "start": float(covered[0].start) * ratio,
                "end": float(covered[-1].end) * ratio,
            })
        log_score = (sum(span_scores) / len(span_scores)) if span_scores else None
        out_lines.append({"segs": segs, "total_chars": total, "log_score": log_score})

    return {
        "lines": out_lines,
        "audio_sec": round(audio_sec, 3),
        "chunks": len(windows),
        "load_sec": load_sec,
        "vocab_size": vocab_size,
        "frame_stride_sec": round(ratio, 5),
        "backend": "torchaudio.functional.forced_align (Viterbi) over NeMo CTC log-probs",
    }


def _worker(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Internal worker for the NeMo NFA bf16 benchmark adapter")
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    args = parser.parse_args(argv)

    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    started = time.perf_counter()
    try:
        payload = _worker_align(request)
    except Exception as exc:  # surfaced to the harness as a run error, not a silent zero
        payload = {"error": f"{type(exc).__name__}: {exc}"[:600]}
    payload.setdefault("load_sec", 0.0)
    payload["align_sec"] = round(time.perf_counter() - started - payload["load_sec"], 2)
    Path(args.response).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    status: dict[str, Any] = {"model": HF_REPO_ID, "chunks": payload.get("chunks")}
    try:
        import torch

        if torch.cuda.is_available():
            status["vram_alloc_peak_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
            status["vram_reserved_peak_mb"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
    except Exception:
        pass
    print(json.dumps(status))
    return 0


if __name__ == "__main__":
    raise SystemExit(_worker(sys.argv[1:]))
