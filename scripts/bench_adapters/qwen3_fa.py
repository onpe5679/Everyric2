"""Qwen3-ForcedAligner adapter for the alignment benchmark (track C candidate).

Architecture note -- this is deliberately *not* a CTC aligner.  Every other candidate in
this harness (``mms-baseline``, the ten ``hf-*`` models) computes a per-frame emission
matrix and runs Viterbi forced alignment over it.  Qwen3-ForcedAligner is a
non-autoregressive **token-classification** aligner built on the Qwen3-ASR audio encoder:
the audio and the transcript units go into one sequence, a ``<timestamp>`` marker is
emitted around each unit, and a 3,902-way classification head predicts the timestamp
*class* at each marker position.  There is no blank symbol, no emission matrix, and no
Viterbi search, so it can neither collapse the way CTC does on synthetic vocals nor
guarantee monotonicity by construction; starts are clamped non-decreasing at the end and
the number of inversions that needed fixing is reported as ``monotonic_fixes``.

It is run in two passes -- one whole-song *anchor* forward, then per-group *refine*
forwards over the audio each group was anchored to.  The reason is that this model
stretches whatever text it is given across whatever audio it is given, so both halves of a
naive window scheme go wrong; see the constants block below for the measurements that
forced this design.

Resolution ceiling: ``timestamp_segment_time`` is 80 ms and the head has 3,902 classes,
so timestamps are quantised to 80 ms and no unit can be placed past 3902*0.08 = 312 s.
That 80 ms floor is coarser than the wav2vec2 CTC candidates' 20 ms frame stride, and it
bounds the best MAE this candidate can reach -- read its syllable-level numbers with that
in mind.

License: Apache-2.0 (``Qwen/Qwen3-ForcedAligner-0.6B-hf`` model card).  Commercially
usable; not an NC candidate.

Supported languages (11, model card): Chinese, English, Cantonese, French, German,
Italian, Japanese, Korean, Portuguese, Russian, Spanish.  The evaluation set's ko/ja/en/zh
are all covered, so this adapter needs no ``--input-mode pron-hangul-local`` detour --
though it still works under one, because forced Hangul input is just Korean text.

Environment (isolated -- the main ``.venv`` must not be touched)::

    C:\\Users\\user\\AppData\\Roaming\\uv\\python\\cpython-3.12.11-windows-x86_64-none\\python.exe ^
        -m venv benchmark\\.venv-qwen3fa
    benchmark\\.venv-qwen3fa\\Scripts\\python.exe -m pip install ^
        torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
    benchmark\\.venv-qwen3fa\\Scripts\\python.exe -m pip install ^
        transformers==5.14.1 accelerate soundfile librosa soynlp nagisa

``transformers>=5`` is required: ``Qwen3ASRForTokenClassification`` does not exist in the
main venv's 4.57.6.  The model card still says "install from source", but 5.14.1 ships it.
``soynlp``/``nagisa`` are only needed by the ``native`` unit mode below.
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
QWEN3FA_PYTHON = BENCHMARK_DIR / ".venv-qwen3fa" / "Scripts" / "python.exe"

MODEL_ID = "Qwen/Qwen3-ForcedAligner-0.6B-hf"

# Two-pass geometry.  An earlier revision walked the song greedily -- 60 s window, feed all
# remaining text, keep whatever landed before the window's end -- on the measured assumption
# that surplus text piles up *at* the boundary rather than smearing over the real units.
# That assumption only holds for dense lyrics.  On sparse songs the model instead emits
# degenerate output (measured on 3rkgLcItbyE, 1.36 units/s: a 60 s window fed all 347 units
# returned timestamps running to 103.5 s -- past the window's own audio -- in an arithmetic
# progression), so the greedy walk swallowed the whole transcript in one or two windows and
# every later line collapsed to the front of the song (MAE 162.6 s).
#
# The fix is to never ask the model to align text against audio it does not belong to.  That
# needs the text->audio assignment up front, which is what the anchor pass produces: one
# whole-song forward, coarse but globally ordered (measured MAE 10.1 s / 3.3 s on the two
# worst songs).  The refine pass then re-aligns each audio segment against *only* the units
# the anchor placed there, which is the matched-input regime the model is good at.
#
# The anchor's whole-song forward is the peak-VRAM step, and it is inherently bounded: the
# timestamp head has 3,902 classes at 80 ms, so no unit can be addressed past 312 s and the
# anchor is capped below that.  Measured peak 6.9 GB at 254 s -- inside the 9 GB gate.
ANCHOR_MAX_SEC = 300.0
# How much anchored content one refine group covers, and the ceiling on the audio slack
# added around that group's own extent.  Anchor error larger than the slack is unrecoverable.
REFINE_SEGMENT_SEC = 45.0
REFINE_MARGIN_SEC = 15.0

# The anchor forward has one systematic failure: this model lays text out at a natural
# *speech* rate and does not stretch it to fill audio that is longer than the words would
# normally take.  Sung lines are held far longer than spoken ones, so on many songs the whole
# lyric gets packed into the front of the track (measured on HyBxn5gzpn0: 122 units placed
# inside 0.7-50.4 s of a 196 s song -- about 3 units/s, conversational pace -- while every
# other candidate in the harness spans it to ~180 s).
#
# Energy on the *separated vocals* stem recovers the extent the anchor lost.  The test is how
# much of the song's singing the anchor actually spans, not how wide the anchor looks: on
# LaEgpNBt-bQ the anchor reached 102 s of a 153 s-sung track and so passed a span-ratio check
# at 0.67 while still ignoring half the song.  Coverage catches that, because the half it
# ignored is sung.  A collapsed anchor is discarded and rebuilt by spreading the units evenly
# over the frames that actually contain singing, which skips instrumental gaps rather than
# merely rescaling over them.
ANCHOR_COVERAGE_MIN = 0.75
# A refine step is rejected when its distinct timestamps fall below this share of its unit
# count.  Measured on LaEgpNBt-bQ's five steps: the four that collapsed scored 0.02-0.15, the
# one that aligned scored 0.51, so the classes are far enough apart that the exact cut is not
# load-bearing.  The minimum size keeps tiny steps, where a low ratio is unremarkable, out.
STEP_MIN_DISTINCT_RATIO = 0.3
STEP_MIN_UNITS = 8
VAD_FRAME_SEC = 0.02
# Threshold relative to this stem's own loud level; separated-vocal levels vary per song, so
# an absolute floor would misfire.
VAD_LEVEL_RATIO = 0.06

# Alignment unit grain.  ``fine`` (default) gives one unit per CJK/kana/Hangul character,
# which is what ``segs`` needs to be comparable with the syllable spans the CTC candidates
# produce.  ``native`` defers to ``Qwen3ASRProcessor.split_words_for_alignment``, i.e. the
# tokenisation the model was trained and benchmarked with (soynlp for Korean, nagisa for
# Japanese, per-character for Chinese, whitespace elsewhere) -- word-grained, so its segs
# are words rather than syllables.  Measured on both smoke songs, ``fine`` is the better
# of the two on line MAE as well as being the finer grain, so it is the default; the knob
# exists so that "our unit choice handicapped the model" stays a falsifiable claim.
UNIT_MODE = os.environ.get("QWEN3FA_UNIT_MODE", "fine")

# Refine can be switched off (``QWEN3FA_REFINE=0``) to run the anchor pass alone, i.e. the
# model's own designed one-shot mode.  Kept as a knob because which one wins is song-shaped:
# refine is what rescues long songs, and on short ones the anchor is already close.
REFINE_ENABLED = os.environ.get("QWEN3FA_REFINE", "1") not in ("0", "false", "False")
# Timestamp head geometry (config.json / processor defaults); asserted in the worker.
TIMESTAMP_SEGMENT_MS = 80.0
MAX_ADDRESSABLE_SEC = 312.0

LOW_CONF_COVERAGE = 0.90

# Model-card language names, keyed by the harness' base language code.
LANGUAGE_NAMES: dict[str, str] = {
    "zh": "Chinese",
    "en": "English",
    "yue": "Cantonese",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "ru": "Russian",
    "es": "Spanish",
}

_UNIT_GRAIN = {
    "fine": "per-character for CJK/kana/Hangul, per-word for Latin",
    "native": "Qwen3ASRProcessor.split_words_for_alignment (word-grained)",
}

_CJK_RANGES = (
    (0x3040, 0x309F),  # hiragana
    (0x30A0, 0x30FF),  # katakana
    (0x31F0, 0x31FF),  # katakana phonetic extensions
    (0x3400, 0x4DBF),  # CJK ext A
    (0x4E00, 0x9FFF),  # CJK unified
    (0xF900, 0xFAFF),  # CJK compatibility ideographs
    (0xAC00, 0xD7A3),  # Hangul syllables
)


def _is_cjk(char: str) -> bool:
    code = ord(char)
    return any(low <= code <= high for low, high in _CJK_RANGES)


def _split_lyrics(lyrics: str) -> list[str]:
    return [line.strip() for line in lyrics.strip().splitlines() if line.strip()]


def _base_language(language: str) -> str:
    return language.removesuffix("_mms").strip().lower()


def split_units(line: str) -> list[str]:
    """Split one lyric line into alignment units, finest-first.

    Each CJK/kana/Hangul character becomes its own unit -- that is the karaoke-relevant
    grain and it is what fills ``segs``.  Latin/Cyrillic/digit runs stay whole words,
    because splitting ``"One"`` into ``O``/``n``/``e`` asks the model for sub-phonemic
    boundaries it was never trained to place.  Punctuation and whitespace are separators,
    never units -- the same rule ``Qwen3ASRProcessor.split_words_for_alignment`` applies,
    including its one exception: an apostrophe is a word character, so ``"Don't"`` stays a
    single unit instead of becoming ``Don`` + ``t``.
    """
    units: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            word = "".join(buffer)
            buffer.clear()
            # A run of bare apostrophes carries no acoustic content to align against.
            if any(char.isalnum() for char in word):
                units.append(word)

    for char in line:
        if _is_cjk(char):
            flush()
            units.append(char)
        elif char == "'" or char.isalnum():
            buffer.append(char)
        else:
            flush()
    flush()
    return units


class Qwen3ForcedAligner:
    """Subprocess adapter around the isolated ``benchmark/.venv-qwen3fa`` installation.

    The harness keeps one instance for the whole sweep, but each ``align`` call spawns a
    fresh worker: model load is ~1.6 s against a ~0.3 s alignment, so a resident worker
    would buy little and would hold ~2 GB of VRAM across the separation stages.
    """

    name = "qwen3-fa"

    def align(self, vocals_path: Path, lyrics: str, language: str) -> Any:
        from scripts.benchmark_alignment import AlignOut

        base = _base_language(language or "")
        if base and base not in LANGUAGE_NAMES:
            raise ValueError(
                f"{self.name} is a {'/'.join(sorted(LANGUAGE_NAMES))} candidate, not {language!r}"
            )
        language_name = LANGUAGE_NAMES.get(base or "ko", "Korean")

        lines = _split_lyrics(lyrics)
        if not lines:
            raise ValueError("lyrics produced zero non-empty lines")
        line_units = [split_units(line) for line in lines]
        if not any(line_units):
            raise ValueError(f"{self.name} found no alignable characters in the lyrics")

        python = _require_python()
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="qwen3fa_") as tmp:
            request_path = Path(tmp) / "request.json"
            response_path = Path(tmp) / "response.json"
            request_path.write_text(
                json.dumps(
                    {
                        "audio": str(vocals_path),
                        "language": language_name,
                        "lines": lines,
                        "line_units": line_units,
                        "unit_mode": UNIT_MODE,
                        "anchor_max_sec": ANCHOR_MAX_SEC,
                        "refine_enabled": REFINE_ENABLED,
                        "refine_segment_sec": REFINE_SEGMENT_SEC,
                        "refine_margin_sec": REFINE_MARGIN_SEC,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            status = _run_worker(python, request_path, response_path, self.name)
            payload = json.loads(response_path.read_text(encoding="utf-8"))

        elapsed = time.perf_counter() - started
        # The worker echoes back the units it actually used: under ``native`` it re-splits
        # with the model's own tokeniser, so its counts -- not ours -- define the folding.
        used_units = payload.get("line_units") or line_units
        line_results = _line_results(lines, used_units, payload["units"], payload["audio_sec"])
        quality_score, quality_meta = _quality_score(line_results)
        return AlignOut(
            lines=line_results,
            elapsed_sec=round(elapsed, 2),
            vram_peak_mb=status.get("vram_alloc_peak_mb"),
            vram_device_peak_mb=status.get("vram_reserved_peak_mb"),
            quality_score=quality_score,
            meta={
                "model": MODEL_ID,
                "adapter": self.name,
                "language": language_name,
                "architecture": "non-CTC: Qwen3-ASR audio encoder + timestamp token classification head",
                "unit_mode": UNIT_MODE,
                "unit_grain": _UNIT_GRAIN[UNIT_MODE],
                "timestamp_segment_ms": TIMESTAMP_SEGMENT_MS,
                "timestamp_quantisation_sec": TIMESTAMP_SEGMENT_MS / 1000.0,
                "align_chunks": payload["chunks"],
                "refine_enabled": REFINE_ENABLED,
                "anchor_passes": payload["anchor_passes"],
                "anchor_collapsed": payload["anchor_collapsed"],
                "anchor_span_sec": payload["anchor_span_sec"],
                "anchor_sung_coverage": payload["anchor_sung_coverage"],
                "sung_sec": payload["sung_sec"],
                "refine_passes": payload["refine_passes"],
                "refine_fallbacks": payload["refine_fallbacks"],
                "monotonic_fixes": payload["monotonic_fixes"],
                "anchor_max_sec": ANCHOR_MAX_SEC,
                "refine_segment_sec": REFINE_SEGMENT_SEC,
                "refine_margin_sec": REFINE_MARGIN_SEC,
                "audio_sec": payload["audio_sec"],
                "model_load_sec": payload["load_sec"],
                "worker_align_sec": payload["align_sec"],
                "monotonic": payload["monotonic"],
                "total_units": len(payload["units"]),
                "coverage_threshold": LOW_CONF_COVERAGE,
                "quality": quality_meta,
                "license": "Apache-2.0",
            },
        )


def register(aligner_registry: dict) -> None:
    """Register the Qwen3 forced-alignment candidate in a harness aligner registry."""
    aligner_registry[Qwen3ForcedAligner.name] = Qwen3ForcedAligner


def _require_python() -> Path:
    if QWEN3FA_PYTHON.is_file():
        return QWEN3FA_PYTHON
    raise RuntimeError(
        "Qwen3-ForcedAligner environment is missing: "
        f"expected {QWEN3FA_PYTHON}. Create it per this module's docstring without "
        "modifying the main .venv."
    )


def _run_worker(python: Path, request: Path, response: Path, name: str) -> dict:
    """Run the worker and recover its trailing status JSON (repo subprocess convention)."""
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
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
    lines: list[str], line_units: list[list[str]], units: list[dict], audio_sec: float
) -> list[dict]:
    """Fold flat unit timestamps back into one result per lyric line (harness contract)."""
    cursor = 0
    line_times: list[tuple[float, float, float | None, int] | None] = []
    line_segs: list[list[dict]] = []
    for own_units in line_units:
        segs: list[dict] = []
        scores: list[float] = []
        for _ in own_units:
            unit = units[cursor]
            cursor += 1
            start, end = float(unit["start"]), float(unit["end"])
            segs.append({"t": unit["t"], "start": round(start, 3), "end": round(end, 3)})
            scores.append(float(unit["score"]))
        if segs:
            confidence = sum(scores) / len(scores) if scores else None
            line_times.append((segs[0]["start"], segs[-1]["end"], confidence, len(segs)))
        else:
            line_times.append(None)
        line_segs.append(segs)

    interpolated = _interpolate_times(line_times, audio_sec)
    results: list[dict] = []
    for text, own_units, timing, segs in zip(lines, line_units, interpolated, line_segs):
        start, end, confidence, matched = timing
        total = len(own_units)
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
                    "model": Qwen3ForcedAligner.name,
                    "unit_grain": _UNIT_GRAIN[UNIT_MODE],
                    "matched_chars": matched,
                    "total_chars": total,
                    "coverage": round(coverage, 6),
                    "low_conf": coverage < LOW_CONF_COVERAGE,
                },
            }
        )
    return results


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
    """Same shape as hf_ctc/worker: mean line confidence, floored to 0.0 on low coverage."""
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
# Worker (runs inside benchmark/.venv-qwen3fa -- transformers 5.x, torch cu128)
# ──────────────────────────────────────────────────────────────────────────


def _worker_align(request: dict) -> dict:
    import torch
    import torchaudio
    from transformers import AutoModelForTokenClassification, AutoProcessor

    audio_path = request["audio"]
    line_units: list[list[str]] = request["line_units"]
    anchor_max_sec = float(request["anchor_max_sec"])
    segment_sec = float(request["refine_segment_sec"])
    margin_sec = float(request["refine_margin_sec"])
    unit_mode = request.get("unit_mode", "fine")

    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = waveform.mean(dim=0)
    if sample_rate != 16_000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16_000)
        sample_rate = 16_000
    audio_sec = waveform.numel() / sample_rate

    load_started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = (
        AutoModelForTokenClassification.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
        .to("cuda" if torch.cuda.is_available() else "cpu")
        .eval()
    )
    load_sec = round(time.perf_counter() - load_started, 2)
    device = next(model.parameters()).device
    timestamp_token_id = int(model.config.timestamp_token_id)
    segment_ms = float(processor.timestamp_segment_time)
    if abs(segment_ms - TIMESTAMP_SEGMENT_MS) > 1e-6:
        # The 80 ms figure is baked into this module's docstring and reported meta; if a
        # future checkpoint changes it, the recorded resolution claim would silently lie.
        raise RuntimeError(f"unexpected timestamp_segment_time: {segment_ms} != {TIMESTAMP_SEGMENT_MS}")

    if unit_mode == "native":
        # Split per line, not over the whole lyric, so each unit keeps a known owning line.
        line_units = [
            processor.split_words_for_alignment(line, request["language"])
            for line in request["lines"]
        ]
    elif unit_mode != "fine":
        raise ValueError(f"unknown unit_mode: {unit_mode!r}")

    flat: list[str] = [unit for units in line_units for unit in units]
    if not flat:
        raise RuntimeError(f"unit_mode={unit_mode} produced zero alignment units")

    def align_window(audio, units: list[str]) -> list[dict]:
        conversation = [[{
            "role": "user",
            "content": [{"type": "audio", "audio": audio.numpy()}]
            + [{"type": "text", "text": unit} for unit in units],
        }]]
        inputs = processor.apply_chat_template(
            conversation, tokenize=True, return_dict=True
        ).to(device, torch.bfloat16)
        with torch.inference_mode():
            outputs = model(**inputs)
        logits = outputs.logits
        stamped = processor.decode_forced_alignment(
            logits=logits, input_ids=inputs["input_ids"], word_lists=[units],
            timestamp_token_id=timestamp_token_id,
        )[0]
        # Confidence proxy: softmax probability of the argmax timestamp class.  There is no
        # CTC score to borrow here, and the harness' collapse gate needs *some* per-line
        # signal; a unit whose two markers are both confidently classified is the closest
        # analogue.  Reported as ``score`` and averaged per line.
        mask = inputs["input_ids"][0] == timestamp_token_id
        probs = torch.softmax(logits[0][mask].float(), dim=-1).max(dim=-1).values.tolist()
        for index, item in enumerate(stamped):
            pair = probs[index * 2: index * 2 + 2]
            item["score"] = sum(pair) / len(pair) if pair else 0.0
        return stamped

    # The anchor runs on the same character units as the refine pass.  Anchoring on whole
    # lines instead was tried to shrink the peak-VRAM forward -- it cuts the text side of the
    # sequence by an order of magnitude (VWVtIg5cdDU: 1,145 units -> 92 lines) -- but a line
    # is far longer than the word- and character-sized units this model was trained to stamp,
    # and the anchors came back worse where it mattered (szyPY8nbBF4 MAE 3.3 -> 22.8,
    # hFTs6HbtxbE 0.40 -> 0.64).  Peak VRAM is the price of a usable anchor here.
    anchor, anchor_passes = _anchor_pass(
        align_window, waveform, flat, sample_rate, audio_sec, anchor_max_sec
    )

    active, frame_sec = _vocal_activity(waveform, sample_rate)
    sung_sec = float(active.sum().item()) * frame_sec
    anchor_span = (anchor[-1][1] - anchor[0][0]) if anchor else 0.0
    coverage = _sung_coverage(active, frame_sec, anchor[0][0], anchor[-1][1]) if anchor else 1.0
    anchor_collapsed = bool(sung_sec > 0.0 and coverage < ANCHOR_COVERAGE_MIN)
    if anchor_collapsed:
        anchor = _spread_over_activity(len(flat), active, frame_sec)
    if request.get("refine_enabled", True):
        # Exactly one round.  A rebuilt anchor is systematically early -- the energy gate counts
        # intro ad-libs and backing vocals as sung, so spreading the lyric over every sung frame
        # starts it before the first real line (measured median residual -22 s on LaEgpNBt-bQ)
        # -- and that is further than one round's margin can travel.  Neither obvious remedy
        # works, and both were measured: widening the first round to a 3x margin re-creates the
        # too-much-audio-for-the-text compression (HyBxn5gzpn0 9.2 -> 23.1 s MAE), and repeating
        # the normal margin diverges rather than converging (9.2 -> 37.7 over three rounds),
        # because a refine step redistributes its text inside the window it is handed rather
        # than searching for a better window.  The residual error is left in place and reported
        # via ``anchor_collapsed`` instead of being chased with a mechanism that makes it worse.
        refined, refine_passes, fallbacks = _refine_pass(
            align_window, waveform, flat, sample_rate, audio_sec, anchor, segment_sec, margin_sec
        )
    else:
        refined, refine_passes, fallbacks = anchor, 0, 0

    # Seams between refine segments are the one place ordering can invert, so clamp starts
    # to be non-decreasing.  Karaoke rendering and the harness' pairing both assume order.
    results: list[dict] = []
    monotonic_fixes = 0
    previous = 0.0
    for unit, (start, end, score) in zip(flat, refined):
        if start < previous:
            monotonic_fixes += 1
        start = min(max(start, previous), audio_sec)
        end = min(max(end, start), audio_sec)
        previous = start
        results.append({
            "t": unit,
            "start": round(start, 3),
            "end": round(end, 3),
            "score": round(float(score), 6),
        })

    return {
        "units": results,
        "line_units": line_units,
        "audio_sec": round(audio_sec, 3),
        "chunks": anchor_passes + refine_passes,
        "anchor_passes": anchor_passes,
        "anchor_collapsed": anchor_collapsed,
        "anchor_span_sec": round(anchor_span, 2),
        "anchor_sung_coverage": round(coverage, 4),
        "sung_sec": round(sung_sec, 2),
        "refine_passes": refine_passes,
        "refine_fallbacks": fallbacks,
        "monotonic_fixes": monotonic_fixes,
        "load_sec": load_sec,
        "monotonic": monotonic_fixes == 0,
    }


def _vocal_activity(waveform, sample_rate):
    """(per-frame sung/not-sung mask, frame duration) from the separated vocals stem.

    Deliberately a plain energy gate rather than a learned VAD: the input is already a
    vocals-only stem, so anything above the stem's own noise floor is singing, and the
    adapter must not grow a second model dependency to answer a question this cheap.
    """
    import torch

    samples = max(1, int(VAD_FRAME_SEC * sample_rate))
    usable = (waveform.numel() // samples) * samples
    if usable < samples:
        return torch.zeros(0, dtype=torch.bool), VAD_FRAME_SEC
    frames = waveform[:usable].reshape(-1, samples).float()
    rms = torch.sqrt((frames ** 2).mean(dim=1) + 1e-12)
    loud = float(torch.quantile(rms, 0.995))
    return rms > max(loud * VAD_LEVEL_RATIO, 1e-4), VAD_FRAME_SEC


def _step_degenerate(stamped, span):
    """Has this refine step actually aligned anything, or just emitted filler?

    Two signatures, both measured on LaEgpNBt-bQ.  Timestamps running past the step's own
    audio are the arithmetic-progression garbage described at the top of this module.  The
    quieter and more damaging one is collapse: a step handed 133 units returned a single
    distinct timestamp -- every unit at 0.0 -- and because zero is inside the window, an
    end-time check waves it through.  It then overwrites a whole group of anchor times with
    one value, which is what produced 512 segs holding only 49 distinct starts.
    """
    if not stamped:
        return True
    if max(item["end_time"] for item in stamped) > span + 2.0:
        return True
    distinct = len({round(item["start_time"], 3) for item in stamped})
    return len(stamped) >= STEP_MIN_UNITS and distinct < STEP_MIN_DISTINCT_RATIO * len(stamped)


def _sung_coverage(active, frame_sec, start, end):
    """Fraction of the song's singing that falls inside ``[start, end]``."""
    total = float(active.sum().item())
    if total <= 0:
        return 1.0
    low = max(0, int(start / frame_sec))
    high = min(int(active.numel()), int(end / frame_sec) + 1)
    if high <= low:
        return 0.0
    return float(active[low:high].sum().item()) / total


def _spread_over_activity(count, active, frame_sec):
    """Place ``count`` units evenly across the frames that contain singing.

    Used only to replace an anchor that collapsed.  The even spread is a prior -- lyrics run
    at a roughly steady rate *while someone is singing*, which is not true against wall-clock
    time once intros and instrumental breaks are in the picture -- and the refine pass is what
    turns it into real timings.  Positions come from unit index rather than from the discarded
    anchor's times, which are exactly the numbers that were found untrustworthy.
    """
    import torch

    indices = torch.nonzero(active).flatten()
    total = int(indices.numel())
    if total == 0 or count == 0:
        return [(0.0, 0.0, 0.0) for _ in range(count)]
    spread: list[tuple[float, float, float]] = []
    for position in range(count):
        low = min(int(position * total / count), total - 1)
        high = min(max(int((position + 1) * total / count), low + 1), total)
        start = float(indices[low].item()) * frame_sec
        end = float(indices[high - 1].item()) * frame_sec + frame_sec
        spread.append((start, max(end, start + frame_sec), 0.0))
    return spread


def _anchor_pass(align_window, waveform, flat, sample_rate, audio_sec, anchor_max_sec):
    """Coarse absolute times for every unit -- one forward for any song under the cap.

    Long enough songs cannot be anchored in one pass (the timestamp head tops out at 312 s),
    so they fall back to sequential super-windows with the transcript split proportionally
    by duration.  That split is a crude prior, but it only has to be good enough for the
    refine pass to sort out, and no song in the evaluation set reaches it.
    """
    times: list[tuple[float, float, float]] = []
    passes = 0
    cursor = 0
    offset = 0.0
    while cursor < len(flat):
        span_end = min(audio_sec, offset + anchor_max_sec)
        remaining_units = len(flat) - cursor
        remaining_sec = max(audio_sec - offset, 1e-6)
        if span_end >= audio_sec - 1e-3:
            take = remaining_units
        else:
            take = max(1, round(remaining_units * (span_end - offset) / remaining_sec))
        chunk = waveform[int(offset * sample_rate): int(span_end * sample_rate)]
        stamped = align_window(chunk, flat[cursor: cursor + take])
        passes += 1
        for item in stamped:
            times.append((
                offset + item["start_time"], offset + item["end_time"], item["score"]
            ))
        cursor += take
        offset = span_end
    return times, passes


def _refine_pass(
    align_window, waveform, flat, sample_rate, audio_sec, anchor, segment_sec, margin_sec
):
    """Re-align consecutive groups of units against the audio the anchor says they occupy.

    The window is derived from the group's own anchored extent rather than from a fixed time
    grid, because this model stretches whatever text it is given across whatever audio it is
    given.  A fixed grid breaks that both ways: a segment holding 20 s of singing but handed
    150 s of audio smears its units across all of it (measured: a 90 s grid with 30 s margins
    left 127 ordering inversions on 3rkgLcItbyE and pushed MAE from the anchor's 10.1 s to
    16.4 s).  Sizing the audio to the text keeps every refine call in the matched-input
    regime, which is the only regime this model is reliable in.

    A step whose output runs past its own audio is the degenerate signature described at the
    top of this module; those steps keep the anchor's times instead of overwriting good data
    with garbage, and the count is reported as ``refine_fallbacks``.
    """
    refined = list(anchor)
    passes = 0
    fallbacks = 0
    first = 0
    while first < len(anchor):
        last = first
        group_start = anchor[first][0]
        while last + 1 < len(anchor) and anchor[last + 1][1] - group_start < segment_sec:
            last += 1
        group_end = max(anchor[index][1] for index in range(first, last + 1))
        # Margin scales with the group so a long span gets proportionally more slack, but
        # never so much that the group's text stops filling the audio it is aligned against.
        margin = min(max(0.2 * (group_end - group_start), 3.0), margin_sec)
        audio_start = max(0.0, group_start - margin)
        audio_stop = min(audio_sec, group_end + margin)
        if audio_stop - audio_start < 1.0:
            first = last + 1
            continue
        chunk = waveform[int(audio_start * sample_rate): int(audio_stop * sample_rate)]
        stamped = align_window(chunk, flat[first: last + 1])
        passes += 1
        span = audio_stop - audio_start
        if _step_degenerate(stamped, span):
            fallbacks += 1
        else:
            for index, item in zip(range(first, last + 1), stamped):
                refined[index] = (
                    audio_start + item["start_time"],
                    audio_start + item["end_time"],
                    item["score"],
                )
        first = last + 1
    return refined, passes, fallbacks


def _worker(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Internal worker for the Qwen3-FA benchmark adapter")
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    args = parser.parse_args(argv)

    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    started = time.perf_counter()
    payload = _worker_align(request)
    payload["align_sec"] = round(time.perf_counter() - started - payload["load_sec"], 2)
    Path(args.response).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    status: dict[str, Any] = {"model": MODEL_ID, "chunks": payload["chunks"]}
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
