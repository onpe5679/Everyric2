"""Parakeet TDT-CTC 0.6B (ja) adapter -- native Japanese CTC alignment, no Hangul detour.

Every existing ja route through this harness rewrites the lyric text into Hangul reading
first (``nemo-nfa@hangul-local``, the current listening winner) or leans on a multilingual
model whose ja head is one of many (``owsm-ctc-v4-1b``).  This candidate is the first ja
*native* acoustic model in the harness: text stays Japanese (kanji + kana, exactly as the
lyric line reads) and only the model changes.

Model: ``nvidia/parakeet-tdt_ctc-0.6b-ja`` -- CC-BY-4.0, a Hybrid FastConformer TDT-CTC
(``EncDecHybridRNNTCTCBPEModel``) trained on ReazonSpeech v2.0 (~35k h Japanese).  The
"hybrid" part is what makes this candidate exist at all: its primary decoder is a TDT
transducer, not CTC, but NeMo trains an auxiliary CTC head on the same encoder
(``self.ctc_decoder``), and ``model.change_decoding_strategy(decoder_type="ctc")`` switches
inference onto it.  That CTC head is what this adapter feeds into
``torchaudio.functional.forced_align`` -- the same Viterbi decoder ``nemo_nfa``/``hf_ctc``
use, so a difference in the numbers is attributable to the acoustic model, not the search.

Two other candidates were considered and ruled out before this one (see
``docs/research/2026-07-30-model-replacement/`` if that survey doc exists, otherwise this
docstring is the record):

* ``reazon-research/reazonspeech-nemo-v2`` -- NeMo-hosted, but it is a **subword RNN-T**
  (Longformer-attention FastConformer, no CTC head at all).  Apache-2.0, ReazonSpeech v2.0,
  same training data as this candidate.  Without a CTC posterior there is nothing for
  ``forced_align`` to consume; using it would mean writing a transducer forced-aligner from
  scratch, which is a different project, not a drop-in NFA candidate.
* A native ja head inside NeMo itself (``stt_ja_*``) -- does not exist.  NVIDIA has never
  published a Korean *or* Japanese Conformer-CTC checkpoint on NGC; ``nemo-nfa``'s ko model
  is community-supplied for the same reason.  Parakeet-ja is the first NVIDIA-published ja
  checkpoint with a usable CTC path.

Architecture detail that drives the worker's shape: unlike ``nemo-nfa``'s pure-CTC
``EncDecCTCModelBPE`` where ``model(...)`` returns log-probs directly, this hybrid model's
``forward()`` only runs the shared encoder and returns ``(encoded, encoded_len)`` -- CTC
log-probs are a second call, ``model.ctc_decoder(encoder_output=encoded)``.  Frame stride is
~80ms (measured: 0.0794s/10s chunk, 0.0799s/60s chunk -- FastConformer's native 8x
subsampling), twice as coarse as ``nemo-nfa``'s 40ms 4x-subsampled ko model.  Vocabulary is a
3,072-token SentencePiece BPE (3,073 with blank); targets are built one source character at a
time exactly as ``nemo_nfa`` does (a lone Japanese character almost always tokenizes to
``['▁', char]`` and the boundary marker is dropped), which keeps ``segs`` per-character so it
pairs against the same syllable metric the Hangul-route candidates report.

Language gating is deliberately *not* a hard raise.  ``nemo-nfa`` raises on a language
mismatch because it is reused across every stratum via ``--input-mode pron-hangul-local``, so
the gate protects against silently running the ko model over un-transliterated text. This
candidate is only ever invoked directly by name against explicitly chosen songs, and the two
songs it must be validated against (``owv06htaoI8``, ``N522kBMyoCk``) carry a *wrong*
``language`` field in ``eval_set.json`` (``en_mms``/``ko``) -- both are Japanese Vocaloid
songs whose YouTube auto-caption language was misdetected because the vocals are synthetic
(the same defect the extension's caption-track picker was patched for). Raising on that field
would reproduce the exact failure ``hf-reazon-hubert-base`` hit on these two songs (see
``benchmark/REPORT.md``: ``"is a ja candidate, not 'ko'"`` / ``"not 'en'"``). The requested
language is recorded in ``meta`` instead, so a real mismatch is visible in the report without
blocking the one comparison this candidate exists to make.

Environment: reuses ``benchmark/.venv-nemo`` (nemo_toolkit[asr]==2.7.3, already installed for
``nemo-nfa`` -- no new packages).  Loading is a plain ``ASRModel.from_pretrained(HF_REPO_ID)``
rather than ``nemo_nfa``'s ``hf_hub_download`` + ``restore_from``: this checkpoint's HF repo
already carries NeMo's own hub metadata, so ``from_pretrained`` resolves it directly.
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

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPO_ROOT / "benchmark"
NEMO_PYTHON = BENCHMARK_DIR / ".venv-nemo" / "Scripts" / "python.exe"

HF_REPO_ID = "nvidia/parakeet-tdt_ctc-0.6b-ja"
ADAPTER_NAME = "parakeet-ctc-ja"

ALIGN_CHUNK_SEC = 60.0
ALIGN_CHUNK_OVERLAP_SEC = 5.0
TARGET_SAMPLE_RATE = 16_000

SUPPORTED_LANGUAGE = "ja"


class ParakeetCtcJaAligner:
    """Subprocess adapter around the isolated ``benchmark/.venv-nemo`` installation."""

    name = ADAPTER_NAME

    def align(self, vocals_path: Path, lyrics: str, language: str) -> Any:
        from scripts.bench_adapters.hf_ctc import LOW_CONF_COVERAGE, _base_language, _split_lyrics
        from scripts.benchmark_alignment import AlignOut

        lines = _split_lyrics(lyrics)
        if not lines:
            raise ValueError("lyrics produced zero non-empty lines")

        requested_language = _base_language(language or "")
        language_mismatch = bool(requested_language) and requested_language != SUPPORTED_LANGUAGE

        python = _require_python()
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="parakeetja_") as tmp:
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
        line_results = _line_results(lines, payload["lines"], payload["audio_sec"], LOW_CONF_COVERAGE)
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
                "language_requested": requested_language or None,
                "language_mismatch_warning": language_mismatch,
                "architecture": "NeMo EncDecHybridRNNTCTCBPEModel (FastConformer TDT+CTC), aux CTC head + Viterbi forced alignment",
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
                "license": "CC-BY-4.0",
                "training_data": "ReazonSpeech v2.0 (~35k h Japanese)",
            },
        )


def register(aligner_registry: dict) -> None:
    """Register the Parakeet-ja CTC candidate in a harness aligner registry."""
    aligner_registry[ParakeetCtcJaAligner.name] = ParakeetCtcJaAligner


def _require_python() -> Path:
    if NEMO_PYTHON.is_file():
        return NEMO_PYTHON
    raise RuntimeError(
        "NeMo environment is missing: "
        f"expected {NEMO_PYTHON}. This candidate reuses nemo_nfa's benchmark/.venv-nemo "
        "without modifying the main .venv."
    )


def _run_worker(python: Path, request: Path, response: Path, name: str) -> dict:
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
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


def _line_results(
    lines: list[str], worker_lines: list[dict], audio_sec: float, low_conf_coverage: float
) -> list[dict]:
    """Fold worker per-character spans into the harness' one-result-per-lyric-line contract."""
    line_times: list[tuple[float, float, float | None, int] | None] = []
    line_segs: list[list[dict]] = []
    for entry in worker_lines:
        segs = [
            {"t": seg["t"], "start": round(float(seg["start"]), 3), "end": round(float(seg["end"]), 3)}
            for seg in entry["segs"]
        ]
        if segs:
            confidence = _confidence(float(entry["log_score"]))
            line_times.append((segs[0]["start"], segs[-1]["end"], confidence, len(segs)))
        else:
            line_times.append(None)
        line_segs.append(segs)

    interpolated = _interpolate_times(line_times, audio_sec)
    results: list[dict] = []
    for text, entry, timing, segs in zip(lines, worker_lines, interpolated, line_segs):
        start, end, confidence, matched = timing
        total = int(entry["total_chars"])
        coverage = (matched / total) if total else 0.0
        results.append(
            {
                "text": text,
                "start": start,
                "end": end,
                "segs": segs,
                "confidence": None if confidence is None else round(confidence, 6),
                "measured": matched > 0 and confidence is not None,
                "chars": matched,
                "coverage": round(coverage, 6),
                "low_conf": coverage < low_conf_coverage,
                "meta": {
                    "model": ParakeetCtcJaAligner.name,
                    "vocab_unit": "SentencePiece BPE per source character",
                    "matched_chars": matched,
                    "total_chars": total,
                    "coverage": round(coverage, 6),
                    "low_conf": coverage < low_conf_coverage,
                },
            }
        )
    return results


def _confidence(mean_log_score: float) -> float | None:
    import math

    if not math.isfinite(mean_log_score):
        return None
    return math.exp(min(0.0, mean_log_score))


def _interpolate_times(
    timings: list[tuple[float, float, float | None, int] | None], audio_length: float
) -> list[tuple[float, float, float | None, int]]:
    """Fill unmeasured lines between measured neighbours (same rule as hf_ctc/nemo_nfa)."""
    result = list(timings)
    i = 0
    while i < len(result):
        if result[i] is not None:
            i += 1
            continue
        start = i
        end = i
        while end + 1 < len(result) and result[end + 1] is None:
            end += 1
        previous_end = result[start - 1][1] if start > 0 and result[start - 1] else 0.0
        next_start = (
            result[end + 1][0] if end + 1 < len(result) and result[end + 1] else audio_length
        )
        slot = max(max(0.0, next_start - previous_end) / (end - start + 1), 0.1)
        for index in range(start, end + 1):
            offset = index - start
            result[index] = (previous_end + offset * slot, previous_end + (offset + 1) * slot, None, 0)
        i = end + 1
    return [item or (0.0, 0.0, None, 0) for item in result]


def _quality_score(lines: list[dict]) -> tuple[float | None, dict]:
    confs = [float(line["confidence"]) for line in lines if line.get("confidence") is not None]
    measured = len(confs)
    total = len(lines)
    average = sum(confs) / measured if confs else None
    ratio = measured / total if total else 0.0
    meta: dict[str, Any] = {
        "aligned_lines": measured,
        "total_lines": total,
        "ratio": round(ratio, 4),
        "measured_conf": None if average is None else round(average, 6),
    }
    if total and ratio >= 0.5:
        return (None if average is None else round(average, 6)), meta
    meta["failed"] = True
    return 0.0, meta


# ──────────────────────────────────────────────────────────────────────────
# Worker (runs inside benchmark/.venv-nemo -- nemo_toolkit[asr], torch cu128)
# ──────────────────────────────────────────────────────────────────────────


def _load_chunking(repo_root: str):
    """Import the repo's chunk planner by file path (see ``nemo_nfa`` for why by-path)."""
    import importlib.util

    path = Path(repo_root) / "everyric2" / "audio" / "chunking.py"
    spec = importlib.util.spec_from_file_location("_bench_chunking_ja", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load chunking helper from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _is_alignment_character(char: str) -> bool:
    import unicodedata

    if char.isspace():
        return False
    category = unicodedata.category(char)
    if category[0] in {"P", "S", "C", "Z"}:
        return False
    return category[0] in {"L", "M", "N"} or 0x4E00 <= ord(char) <= 0x9FFF


def _worker_align(request: dict) -> dict:
    import torch
    import torchaudio
    import torchaudio.functional as functional
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ASRModel.from_pretrained(HF_REPO_ID, map_location=device)
    if int(model.cfg.sample_rate) != TARGET_SAMPLE_RATE:
        return {"error": f"model expects {model.cfg.sample_rate}Hz, worker hardcodes {TARGET_SAMPLE_RATE}Hz"}
    model.change_decoding_strategy(decoder_type="ctc")
    model.eval()
    load_sec = round(time.perf_counter() - load_started, 2)

    tokenizer = model.tokenizer
    ctc_decoder = model.ctc_decoder
    vocab_size = int(ctc_decoder.num_classes_with_blank)
    blank_id = vocab_size - 1

    def encode_char(char: str) -> list[int] | None:
        """Token ids for one source character, or None when it is out of vocabulary.

        Same rule as ``nemo_nfa.encode_char``: drop the bare word-boundary piece and keep
        only the character's own SentencePiece pieces, so targets stay per-character.
        """
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
        signal = chunk.unsqueeze(0).to(device)
        length = torch.tensor([chunk.numel()], device=device)
        with torch.inference_mode():
            encoded, _ = model(input_signal=signal, input_signal_length=length)
            log_probs = model.ctc_decoder(encoder_output=encoded)
        return log_probs

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
        "backend": "torchaudio.functional.forced_align (Viterbi) over NeMo hybrid TDT-CTC aux-CTC log-probs",
    }


def _worker(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Internal worker for the Parakeet-ja CTC benchmark adapter")
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
