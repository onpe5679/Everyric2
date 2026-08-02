"""NeMo Forced Aligner (NFA) adapter for the alignment benchmark (track C candidate).

What this is, precisely.  NFA is not an alignment *algorithm* distinct from what this
harness already runs -- it is NeMo's CTC acoustic model plus Viterbi forced alignment over
that model's log-probs.  The novelty a track-C benchmark can actually extract from it is
therefore the **acoustic model family**: a Conformer-CTC encoder with a SentencePiece BPE
vocabulary, trained on ~14k hours of Korean, versus the wav2vec2/HuBERT character- and
jamo-vocabulary models the ``hf-*`` candidates use.  The search itself is deliberately the
same ``torchaudio.functional.forced_align`` that ``hf_ctc`` uses, so a difference in the
numbers is attributable to the model rather than to the decoder.

Model: ``SungBeom/stt_kr_conformer_ctc_medium`` -- Apache-2.0, ``EncDecCTCModelBPE``,
Conformer-CTC (d_model 512, 18 layers), 2,047-token SentencePiece BPE of which 1,814 are
single characters, 10 ms window stride with 4x striding subsampling, i.e. a **40 ms frame
stride**.  That sits between Qwen3-FA's 80 ms quantisation and the wav2vec2 candidates'
20 ms stride, and it is the resolution floor on this candidate's syllable metrics.

Language: Korean only.  NVIDIA publishes no Korean ASR checkpoint (their ``stt_*`` line
has no ko entry), so this is a community Apache-2.0 model, which settles the open question
the 7/26 survey and the C-track report disagreed on: NFA's *tooling* is language-agnostic,
but its Korean *model* is community-supplied, not NVIDIA's.  ja/en/zh songs reach this
candidate through the harness' ``--input-mode pron-hangul-local``, which rewrites any
language into Hangul reading before alignment.

Environment (isolated -- the main ``.venv`` must not be touched)::

    C:\\Users\\user\\AppData\\Roaming\\uv\\python\\cpython-3.12.11-windows-x86_64-none\\python.exe ^
        -m venv benchmark\\.venv-nemo
    benchmark\\.venv-nemo\\Scripts\\python.exe -m pip install ^
        torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
    benchmark\\.venv-nemo\\Scripts\\python.exe -m pip install "nemo_toolkit[asr]==2.7.3"

Installing torch from the cu128 index *first* matters: ``nemo_toolkit`` only pins
``torch>=2.6.0``, so a bare install resolves a default PyPI wheel instead of the CUDA one.
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

HF_REPO_ID = "SungBeom/stt_kr_conformer_ctc_medium"
HF_FILENAME = "stt_kr_conformer_ctc_medium.nemo"

# Same window geometry as hf_ctc.ALIGN_CHUNK_SEC/ALIGN_CHUNK_OVERLAP_SEC, and the same
# stitching helper, so peak VRAM is bounded by chunk length rather than song length.
ALIGN_CHUNK_SEC = 60.0
ALIGN_CHUNK_OVERLAP_SEC = 5.0
TARGET_SAMPLE_RATE = 16_000
LOW_CONF_COVERAGE = 0.90

SUPPORTED_LANGUAGE = "ko"


def _split_lyrics(lyrics: str) -> list[str]:
    return [line.strip() for line in lyrics.strip().splitlines() if line.strip()]


def _base_language(language: str) -> str:
    return language.removesuffix("_mms").strip().lower()


class NemoNFAAligner:
    """Subprocess adapter around the isolated ``benchmark/.venv-nemo`` installation."""

    name = "nemo-nfa"

    def align(self, vocals_path: Path, lyrics: str, language: str) -> Any:
        from scripts.benchmark_alignment import AlignOut

        base = _base_language(language or "")
        if base and base != SUPPORTED_LANGUAGE:
            raise ValueError(
                f"{self.name} is a {SUPPORTED_LANGUAGE} candidate, not {language!r}"
            )

        lines = _split_lyrics(lyrics)
        if not lines:
            raise ValueError("lyrics produced zero non-empty lines")

        python = _require_python()
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="nemonfa_") as tmp:
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
            },
        )


def register(aligner_registry: dict) -> None:
    """Register the NeMo forced-alignment candidate in a harness aligner registry."""
    aligner_registry[NemoNFAAligner.name] = NemoNFAAligner


def _require_python() -> Path:
    if NEMO_PYTHON.is_file():
        return NEMO_PYTHON
    raise RuntimeError(
        "NeMo environment is missing: "
        f"expected {NEMO_PYTHON}. Create it per this module's docstring without "
        "modifying the main .venv."
    )


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


def _line_results(lines: list[str], worker_lines: list[dict], audio_sec: float) -> list[dict]:
    """Fold worker per-character spans into the harness' one-result-per-lyric-line contract."""
    line_times: list[tuple[float, float, float | None, int] | None] = []
    line_segs: list[list[dict]] = []
    for entry in worker_lines:
        segs = [
            {"t": seg["t"], "start": round(float(seg["start"]), 3), "end": round(float(seg["end"]), 3)}
            for seg in entry["segs"]
        ]
        if segs:
            # ``forced_align`` scores are log-probabilities because the emission is a
            # log-softmax; the harness' quality score and its low-confidence gate both read
            # confidence as a probability, so exponentiate exactly as hf_ctc does.
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
                "low_conf": coverage < LOW_CONF_COVERAGE,
                "meta": {
                    "model": NemoNFAAligner.name,
                    "vocab_unit": "SentencePiece BPE per source character",
                    "matched_chars": matched,
                    "total_chars": total,
                    "coverage": round(coverage, 6),
                    "low_conf": coverage < LOW_CONF_COVERAGE,
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
    """Fill unmeasured lines between measured neighbours (same rule as hf_ctc)."""
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
    """Import the repo's chunk planner by file path.

    ``everyric2.audio.chunking`` is pure numpy, but importing it as a package would drag
    ``everyric2/__init__`` and the settings stack into a venv that has none of it.  Loading
    the single module file keeps the worker's dependency surface at numpy while still using
    the *same* window/stitch code as the hf_ctc candidates -- the comparison depends on it.
    """
    import importlib.util

    path = Path(repo_root) / "everyric2" / "audio" / "chunking.py"
    spec = importlib.util.spec_from_file_location("_bench_chunking", path)
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
    model = ASRModel.restore_from(checkpoint, map_location=device)
    model.eval()
    load_sec = round(time.perf_counter() - load_started, 2)

    tokenizer = model.tokenizer
    vocab_size = int(model.decoder.num_classes_with_blank)
    # NeMo CTC reserves the final index for blank (``num_classes_with_blank - 1``).
    blank_id = vocab_size - 1

    def encode_char(char: str) -> list[int] | None:
        """Token ids for one source character, or None when it is out of vocabulary.

        SentencePiece prefixes a standalone piece with the word-boundary marker; that marker
        is not an acoustic event, so it is dropped and only the character's own pieces are
        used as alignment targets.  Building targets per character (rather than letting BPE
        merge ``어떤`` into one piece) is what makes ``segs`` per-syllable, which is the
        grain the harness' syllable metric pairs against the baseline's character spans.
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
            log_probs, _, _ = model(input_signal=signal, input_signal_length=length)
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
        # Mean over every merged span in the line -- the same denominator hf_ctc uses, so
        # the two candidates' confidences stay on one scale.
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

    parser = argparse.ArgumentParser(description="Internal worker for the NeMo NFA benchmark adapter")
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
