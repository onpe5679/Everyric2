"""OWSM-CTC v4 1B adapter for the alignment benchmark.

``espnet/owsm_ctc_v4_1B`` is an ESPnet checkpoint (``config.yaml`` + ``.pth`` + a 50k
SentencePiece model), not a Transformers repository, and ESPnet pins a dependency set that
conflicts with the benchmark interpreter.  Following ``separators_roformer``'s pattern, ESPnet
lives in ``benchmark/.venv-owsm`` and this file re-invokes itself there as a subprocess
worker; the benchmark venv stays untouched.

Model shape that drives the design:

* Encoder-only E-Branchformer, 27 blocks, ``conv2d8`` subsampling.  One CTC frame is ~80ms
  (12.47 frames/s measured), roughly 4x coarser than the wav2vec2 candidates' 20ms.
* The encoder is conditioned on a two-token prefix ``[<lang>, <asr>]`` whose states are
  **prepended to the encoder output**, so the first two frames are not audio.  The worker
  computes the expected audio frame count from the frontend/subsampling arithmetic and refuses
  to continue unless the surplus is exactly the prefix length -- a silent failure here would
  shift every timestamp by ~160ms.
* Training used a fixed 30s buffer (``preprocessor_conf.speech_length``), so the worker chunks
  at that length rather than the module-wide 60s default and pads the tail chunk.

Alignment unit: the vocabulary is a 50k unigram SentencePiece model, so unlike the
single-character candidates a token can span several lyric characters (``▁안녕하세요`` is one
token).  The worker therefore aligns the model's **native** tokenization -- which is what the
CTC head was trained to emit, and the only way to get honest posteriors out of it -- and then
splits each token's measured span across the alignable source characters it covers,
proportionally.  Line boundaries are always measured token boundaries; character boundaries
inside a multi-character token are interpolated, and ``meta.preprocessing`` says so.

Language support: multilingual.  ko/ja/en/zh map onto ``<kor>``/``<jpn>``/``<eng>``/``<cmn>``;
anything else falls back to ``<nolang>``, so no stratum is gated out.

License: CC-BY-4.0 (model card).  Commercial use is permitted with attribution -- unlike the
CC-BY-NC-4.0 MMS baseline.  The training corpus ``espnet/yodas_owsmv4`` is separately licensed.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPO_ROOT / "benchmark"
OWSM_PYTHON = BENCHMARK_DIR / ".venv-owsm" / "Scripts" / "python.exe"

MODEL_ID = "espnet/owsm_ctc_v4_1B"
ADAPTER_NAME = "owsm-ctc-v4-1b"
EXP_DIR_NAME = "s2t_train_owsmctc_ebf27_conv2d8_size1024_mel128_bs320_raw_bpe50000"
MODEL_FILE_NAME = "valid.total_count.ave_5best.till70epoch.pth"

ALIGN_CHUNK_OVERLAP_SEC = 5.0

# OWSM language symbols for the benchmark's strata. ``<nolang>`` is the model's own
# "unspecified" symbol, so an unmapped language degrades instead of failing.
LANGUAGE_SYMBOLS = {"ko": "<kor>", "ja": "<jpn>", "en": "<eng>", "zh": "<cmn>"}
DEFAULT_LANGUAGE_SYMBOL = "<nolang>"

PREPROCESSING_LABEL = (
    "native 50k SentencePiece tokenization; character seg boundaries interpolated "
    "within multi-character tokens"
)


def _find_snapshot() -> Path:
    """Locate the cached ``espnet/owsm_ctc_v4_1B`` snapshot root."""

    import os

    roots: list[Path] = []
    for variable in ("TRANSFORMERS_CACHE", "HF_HOME"):
        value = os.environ.get(variable)
        if not value:
            continue
        root = Path(value)
        if variable == "HF_HOME" and root.name.lower() != "hub":
            root = root / "hub"
        if root not in roots:
            roots.append(root)

    slug = "models--" + MODEL_ID.replace("/", "--")
    matches: list[Path] = []
    for root in roots:
        matches.extend(root.glob(f"{slug}/snapshots/*/exp/{EXP_DIR_NAME}/{MODEL_FILE_NAME}"))
    if not matches:
        raise FileNotFoundError(
            f"{MODEL_ID} was not found under HF_HOME/TRANSFORMERS_CACHE; "
            "download it before running this candidate"
        )
    newest = max(matches, key=lambda path: path.stat().st_mtime)
    return newest.parents[2]


class OwsmCTCAligner:
    """OWSM-CTC forced aligner driven through the isolated ESPnet interpreter.

    Deliberately not a subclass of ``benchmark_alignment.AlignerAdapter``: this file is also
    executed as ``__main__`` inside ``benchmark/.venv-owsm``, where the harness' imports do not
    resolve.  The harness reads attributes off the returned object rather than type-checking
    it, so every harness import stays inside the parent-only code paths below.
    """

    name = ADAPTER_NAME

    # VRAM knobs. The default reproduces the model's own inference recipe: fp32 weights and the
    # 30s training buffer. ``buffer_sec=None`` means "whatever preprocessor_conf.speech_length
    # says"; setting it shorter takes the encoder off its training length, which is a quality
    # risk, but is the only lever that shrinks activations -- peak VRAM is a function of the
    # padded buffer, not of song length (every song in the fp32 sweep reported the same peak).
    dtype: str = "float32"
    buffer_sec: float | None = None
    # 인코더 청크를 한 번에 몇 개씩 태울지. 1이 기존 순차 경로와 완전히 동일하다.
    # 30초 버퍼 하나로는 카드가 놀아서(279초 곡 12청크 = 11.66초, 정렬 총시간의 88%)
    # 배치가 활성화 메모리를 주고 그 유휴 시간을 산다.
    batch_size: int = 1
    # torch's caching allocator reserves far more than it allocates here (fp32: 8,968MB reserved
    # vs 5,165MB allocated). Expandable segments hand that fragmentation back without touching a
    # single number the model computes, so it is the one lever with no quality risk at all.
    expandable_segments: bool = False

    def __init__(self) -> None:
        self._snapshot: Path | None = None

    @staticmethod
    def _dominance(vocals_path: Path) -> list[float] | None:
        """보컬 우세도 곡선 — star를 넣을 자리(발성이 끊긴 곳)를 가르는 신호.

        f0 유성도·스템 RMS·VAD는 전부 분리 스템 위에서 죽는다(간주 presence 0.979,
        간주 RMS −23.4dB vs 가창 −19.8dB로 3dB 차이 — ``alignment/star_prior.py`` 실측).
        **반주 대비** 비율만이 3배 대비로 갈랐다(간주 0.199 · 가창 0.36~0.68). 프로드의
        측정된 구현을 그대로 부른다 — 복제하면 신호가 갈린다.

        smooth 0.2s는 기본값(0.4s)보다 짧다. 기본값은 star 가격 성형용이라 경계를 뭉개도
        되지만, 여기서는 «끊겼나»를 봐야 하므로 짧은 쉼이 뭉개지면 안 된다.
        """
        instrumental = vocals_path.with_name("inst.wav")
        if not instrumental.is_file():
            return None
        try:
            import librosa

            from everyric2.alignment.star_prior import vocal_presence_from_stems

            vocals, _ = librosa.load(str(vocals_path), sr=16_000, mono=True)
            accomp, _ = librosa.load(str(instrumental), sr=16_000, mono=True)
            made = vocal_presence_from_stems(vocals, accomp, 16_000, smooth_sec=0.2, hop_sec=0.01)
        except Exception:
            logger.warning("우세도 계산 실패 — star 자리 선별 없이 진행", exc_info=True)
            return None
        if made is None:
            return None
        return [round(float(x), 4) for x in made[1]]

    def align(self, vocals_path: Path, lyrics: str, language: str) -> Any:
        from scripts.bench_adapters.hf_ctc import (
            LOW_CONF_COVERAGE,
            _base_language,
            _quality_score,
            _split_lyrics,
        )
        from scripts.benchmark_alignment import AlignOut

        lines = _split_lyrics(lyrics)
        if not lines:
            raise ValueError("lyrics produced zero non-empty lines")
        if not OWSM_PYTHON.is_file():
            raise RuntimeError(
                "OWSM environment is missing: expected "
                f"{OWSM_PYTHON}. Install espnet in benchmark/.venv-owsm "
                "without modifying the main .venv."
            )
        if self._snapshot is None:
            self._snapshot = _find_snapshot()

        lang_sym = LANGUAGE_SYMBOLS.get(_base_language(language or ""), DEFAULT_LANGUAGE_SYMBOL)
        payload = {
            "star": bool(getattr(self, "star_tokens", False)),
            "star_mode": getattr(self, "star_mode", "all"),
            "presence": (
                self._dominance(vocals_path)
                if getattr(self, "star_mode", "all") == "gap"
                else None
            ),
            "presence_hop": 0.01,
            "vocals_path": str(vocals_path),
            "lines": lines,
            "lang_sym": lang_sym,
            "snapshot": str(self._snapshot),
            "overlap_sec": ALIGN_CHUNK_OVERLAP_SEC,
            "dtype": self.dtype,
            "buffer_sec": self.buffer_sec,
            "batch_size": self.batch_size,
            "emit_curves": bool(getattr(self, "emit_curves", False)),
        }

        started = time.perf_counter()
        result = self._run_worker(payload)
        elapsed = time.perf_counter() - started

        line_results = self._line_results(lines, result)
        quality_score, quality_meta = _quality_score(line_results)
        return AlignOut(
            lines=line_results,
            elapsed_sec=round(elapsed, 2),
            vram_peak_mb=result.get("vram_peak_mb"),
            vram_device_peak_mb=result.get("vram_reserved_peak_mb"),
            quality_score=quality_score,
            meta={
                "model": MODEL_ID,
                "adapter": self.name,
                "language": "multilingual",
                "lang_sym": lang_sym,
                "vocab_unit": "bpe subword (50k unigram SentencePiece)",
                "vocab_size": result.get("vocab_size"),
                "model_load_sec": result.get("load_sec"),
                "audio_sec": result.get("audio_sec"),
                "sample_rate": 16_000,
                "coverage_threshold": LOW_CONF_COVERAGE,
                "coverage_denominator": "letter/number-like source characters",
                "dtype": self.dtype,
                "expandable_segments": self.expandable_segments,
                "load_reserved_peak_mb": result.get("load_reserved_peak_mb"),
                "align_chunks": result.get("chunks"),
                "align_chunk_sec": result.get("chunk_sec"),
                "frame_sec": result.get("frame_sec"),
                "preprocessing": PREPROCESSING_LABEL,
                # 줄 사이 star가 실제로 얼마를 흡수했는가 — 채택 판단의 핵심 수치다.
                "star_tokens": result.get("star_tokens"),
                "star_frames": result.get("star_frames"),
                "star_mode": result.get("star_mode"),
                "star_slots": result.get("star_slots"),
                "quality": quality_meta,
                "worker_python": str(OWSM_PYTHON),
            },
        )

    def _run_worker(self, payload: dict[str, Any]) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="owsm_bench_") as tmp:
            request = Path(tmp) / "request.json"
            response = Path(tmp) / "response.json"
            request.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            command = [
                str(OWSM_PYTHON),
                str(Path(__file__).resolve()),
                "--worker",
                "--request",
                str(request),
                "--response",
                str(response),
            ]
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=3600,
                env=self._worker_env(),
            )
            if completed.returncode != 0 or not response.is_file():
                raise RuntimeError(
                    f"{self.name} worker failed ({completed.returncode}):\n"
                    f"stdout:\n{(completed.stdout or '')[-4000:]}\n"
                    f"stderr:\n{(completed.stderr or '')[-4000:]}"
                )
            return json.loads(response.read_text(encoding="utf-8"))

    def _worker_env(self) -> dict[str, str]:
        import os

        env = dict(os.environ)
        # The worker prints structured errors and ESPnet logs non-ASCII model paths.
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        if self.expandable_segments:
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        return env

    def _line_results(self, lines: list[str], result: dict[str, Any]) -> list[dict[str, Any]]:
        """Assemble the harness line contract from the worker's per-line spans."""

        from scripts.bench_adapters.hf_ctc import (
            LOW_CONF_COVERAGE,
            _confidence,
            _interpolate_times,
        )

        worker_lines = result["lines"]
        if len(worker_lines) != len(lines):
            raise RuntimeError(
                f"{self.name} worker returned {len(worker_lines)} lines for {len(lines)} inputs"
            )
        audio_length = float(result.get("audio_sec") or 0.0)

        timings: list[tuple[float, float, float | None, int] | None] = []
        for entry in worker_lines:
            segs = entry.get("segs") or []
            if not segs:
                timings.append(None)
                continue
            score = entry.get("mean_log_score")
            confidence = _confidence(float(score)) if score is not None else None
            timings.append(
                (
                    float(segs[0]["start"]),
                    float(segs[-1]["end"]),
                    confidence,
                    int(entry.get("matched_chars") or 0),
                )
            )

        interpolated = _interpolate_times(timings, audio_length)
        results: list[dict[str, Any]] = []
        for text, entry, timing in zip(lines, worker_lines, interpolated):
            start, end, confidence, matched = timing
            total = int(entry.get("total_chars") or 0)
            coverage = (matched / total) if total else 0.0
            results.append(
                {
                    "text": text,
                    "start": start,
                    "end": end,
                    "segs": entry.get("segs") or [],
                    "confidence": None if confidence is None else round(confidence, 6),
                    "measured": matched > 0 and confidence is not None,
                    "chars": matched,
                    "coverage": round(coverage, 6),
                    "low_conf": coverage < LOW_CONF_COVERAGE,
                    "meta": {
                        "model": self.name,
                        "vocab_unit": "bpe subword (50k unigram SentencePiece)",
                        "preprocessing": PREPROCESSING_LABEL,
                        "matched_chars": matched,
                        "total_chars": total,
                        "coverage": round(coverage, 6),
                        "low_conf": coverage < LOW_CONF_COVERAGE,
                        "tokens": int(entry.get("tokens") or 0),
                    },
                }
            )
        return results


# VRAM-reduction variants. The fp32/30s default stays under the plain name so its existing
# sweep results keep their meaning; each variant gets its own adapter name so the harness
# caches it in a separate runs/ directory instead of overwriting the fp32 evidence.
# Two axes were measured and dropped rather than registered:
#   float16 -- the encoder overflows. The fp32-cast log_softmax still yields non-finite
#     emissions and ``forced_align`` dies on a garbage DP index. bfloat16 keeps fp32's exponent
#     range and is the only safe half-precision choice here.
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments -- unsupported on Windows and silently ignored;
#     reserved peak came back byte-identical (8,968MB) while the run got slower.
VRAM_VARIANTS: tuple[dict[str, Any], ...] = (
    # Config-identical to the base adapter. It exists only so the allocator-hygiene effect
    # (CPU-side dtype cast + post-load empty_cache) can be measured on fp32 without
    # overwriting the pre-fix baseline runs the sweep already recorded under the plain name.
    {"suffix": "fp32"},
    {"suffix": "bf16", "dtype": "bfloat16"},
    # 인코더 배치 — 실측 결과 **이득 없음**(279초 곡 11.13s → b4 10.47s, b8 10.67s)에
    # VRAM만 2,622 → 4,573/5,653MB로 늘었다. 30초 버퍼로도 카드는 이미 포화 상태였다는 뜻이라
    # 이 축은 닫혔다. 재실험 없이 되풀이하지 않도록 변형은 남겨 둔다.
    {"suffix": "bf16-b4", "dtype": "bfloat16", "batch_size": 4},
    {"suffix": "bf16-b8", "dtype": "bfloat16", "batch_size": 8},
    # 버퍼 길이 — 연산량 자체를 줄이는 축이다. 30초 버퍼 + 5초 겹침이면 279초 곡이 12청크
    # (=360초 분량)로 불어나 원곡보다 81초를 더 계산한다. 버퍼를 키우면 청크 수와 겹침 낭비가
    # 같이 준다. 대가는 훈련 길이(30초) 이탈이라 품질을 반드시 UST로 재확인해야 한다.
    {"suffix": "bf16-60s", "dtype": "bfloat16", "buffer_sec": 60.0},
    {"suffix": "bf16-90s", "dtype": "bfloat16", "buffer_sec": 90.0},
    {"suffix": "bf16-20s", "dtype": "bfloat16", "buffer_sec": 20.0},
    {"suffix": "bf16-15s", "dtype": "bfloat16", "buffer_sec": 15.0},
    {"suffix": "20s", "buffer_sec": 20.0},
    # ★줄 사이 star — 추임새·애드립·반복 후렴 흡수. 채택 스택(bf16)에 얹은 실험 변형이다.
    {"suffix": "bf16-star", "dtype": "bfloat16", "star_tokens": True},
    # ★끊긴 자리에만 star — 「명백히 계속 부르는」 줄 사이에는 안 넣는다(사용자 제안 2026-08-02).
    # 이건 기각된 ``star_prior``와 **구조가 다르다**: 그쪽은 star를 어디에나 두고 프레임별
    # 가격만 매겨 blank로 동점이 우회했지만, 여기서는 토큰 자체를 안 넣으므로 우회할 대상이 없다.
    {"suffix": "bf16-stargap", "dtype": "bfloat16", "star_tokens": True, "star_mode": "gap"},
)


def _variant_class(spec: dict[str, Any]) -> type[OwsmCTCAligner]:
    class VariantOwsmCTCAligner(OwsmCTCAligner):
        name = f"{ADAPTER_NAME}-{spec['suffix']}"
        dtype = spec.get("dtype", "float32")
        buffer_sec = spec.get("buffer_sec")
        expandable_segments = spec.get("expandable_segments", False)
        batch_size = spec.get("batch_size", 1)
        star_tokens = spec.get("star_tokens", False)
        star_mode = spec.get("star_mode", "all")

    VariantOwsmCTCAligner.__name__ = "OwsmCTC_" + spec["suffix"].replace("-", "_")
    VariantOwsmCTCAligner.__qualname__ = VariantOwsmCTCAligner.__name__
    return VariantOwsmCTCAligner


def register(aligner_registry: dict) -> None:
    """Register the OWSM-CTC candidate without importing the harness module."""

    aligner_registry[OwsmCTCAligner.name] = OwsmCTCAligner
    for spec in VRAM_VARIANTS:
        variant = _variant_class(spec)
        aligner_registry[variant.name] = variant


# ──────────────────────────────────────────────────────────────────────────
# Worker — runs inside benchmark/.venv-owsm, never in the benchmark interpreter
# ──────────────────────────────────────────────────────────────────────────


def _worker_is_alignment_character(char: str) -> bool:
    """Same alignment-target rule as ``hf_ctc``, restated for the isolated interpreter."""

    import unicodedata

    if char.isspace():
        return False
    category = unicodedata.category(char)
    if category[0] in {"P", "S", "C", "Z"}:
        return False
    return category[0] in {"L", "M", "N"} or 0x4E00 <= ord(char) <= 0x9FFF


def _load_chunking_module() -> Any:
    """Import ``everyric2.audio.chunking`` by path.

    ``everyric2.audio.__init__`` pulls in the downloader/loader stack, which is not installed
    in the OWSM venv, so the package import path is not usable here.  The chunking module
    itself only needs numpy, and loading it by file keeps one implementation of the
    overlap-crop stitching rather than a divergent copy.
    """

    import importlib.util

    path = REPO_ROOT / "everyric2" / "audio" / "chunking.py"
    spec = importlib.util.spec_from_file_location("_owsm_bench_chunking", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load chunking helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _expected_audio_frames(n_samples: int, hop_length: int = 160) -> int:
    """Encoder frames the audio alone should occupy: log-mel frames then ``conv2d8``.

    The default ESPnet frontend is a centered STFT (one frame per hop, plus one), and
    ``Conv2dSubsampling8`` applies three kernel-3/stride-2 convolutions along time.
    """

    frames = n_samples // hop_length + 1
    for _ in range(3):
        frames = (frames - 3) // 2 + 1
    return frames


# 「끊김」 판정. 우세도가 ``_STAR_GAP_LEVEL`` 아래인 프레임이 ``_STAR_GAP_MIN_FRAMES``(10ms
# 격자) 이상 이어져야 star를 놓는다. 실측 근거(``star_prior``): 간주 우세도 0.199 · 가창
# 0.36~0.68이라 0.3이 그 사이다. 100ms는 숨 쉬는 시간 — 그보다 짧은 골은 무성 자음이다.
_STAR_GAP_LEVEL = 0.30
_STAR_GAP_MIN_FRAMES = 10


def _worker_main(request_path: Path, response_path: Path) -> int:
    import os

    import numpy as np
    import torch

    payload = json.loads(request_path.read_text(encoding="utf-8"))
    snapshot = Path(payload["snapshot"])
    exp_dir = snapshot / "exp" / EXP_DIR_NAME

    # config.yaml stores feats_stats.npz as a snapshot-relative path.
    os.chdir(snapshot)

    from espnet2.bin.s2t_inference_ctc import Speech2TextGreedySearch

    started = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = str(payload.get("dtype") or "float32")
    # Build on CPU and only then move to the GPU. ESPnet casts to ``dtype`` after materializing
    # fp32 weights, so constructing directly on cuda puts a 4GB fp32 copy and the cast result on
    # the card at the same time -- measured 7,732MB reserved during load, which dwarfs the
    # ~3GB the alignment itself needs. Casting on CPU means the GPU only ever sees ``dtype``.
    s2t = Speech2TextGreedySearch(
        s2t_train_config=str(exp_dir / "config.yaml"),
        s2t_model_file=str(exp_dir / MODEL_FILE_NAME),
        bpemodel=str(snapshot / "data" / "token_list" / "bpe_unigram50000" / "bpe.model"),
        device="cpu",
        dtype=dtype,
        lang_sym=payload["lang_sym"],
        task_sym="<asr>",
        use_flash_attn=False,
    )
    if device != "cpu":
        s2t.s2t_model.to(device)
        s2t.device = device
    load_sec = round(time.perf_counter() - started, 2)

    model = s2t.s2t_model
    torch_dtype = getattr(torch, dtype)
    token_list = list(s2t.s2t_train_args.token_list)
    token_to_id = {token: index for index, token in enumerate(token_list)}
    blank_id = int(model.blank_id)
    # The padded buffer the encoder actually sees. Defaults to the training length.
    chunk_sec = float(
        payload.get("buffer_sec") or s2t.s2t_train_args.preprocessor_conf["speech_length"]
    )

    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(
        model_file=str(snapshot / "data" / "token_list" / "bpe_unigram50000" / "bpe.model")
    )

    # ── targets: native SentencePiece pieces, mapped back to source characters ──
    lines: list[str] = payload["lines"]
    target_ids: list[int] = []
    # (line index, [character indices this token covers])
    token_owners: list[tuple[int, list[int]]] = []
    line_totals: list[int] = []
    for line_index, line in enumerate(lines):
        alignable = [i for i, char in enumerate(line) if _worker_is_alignment_character(char)]
        line_totals.append(len(alignable))
        alignable_set = set(alignable)
        for piece in sp.encode(line, out_type="immutable_proto").pieces:
            token_id = token_to_id.get(piece.piece)
            if token_id is None or token_id == blank_id:
                continue
            covered = [i for i in range(piece.begin, piece.end) if i in alignable_set]
            if not covered:
                continue
            target_ids.append(token_id)
            token_owners.append((line_index, covered))

    if not target_ids:
        raise RuntimeError("OWSM found no in-vocabulary lyric tokens")

    # ── audio ──
    import torchaudio

    waveform, sample_rate = torchaudio.load(payload["vocals_path"])
    waveform = waveform.mean(dim=0)
    if sample_rate != 16_000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16_000)
    waveform = waveform.to(dtype=torch.float32, device="cpu").contiguous()
    n_samples = int(waveform.numel())
    audio_sec = n_samples / 16_000

    chunking = _load_chunking_module()
    windows = chunking.plan_chunk_windows(
        n_samples, int(chunk_sec * 16_000), int(float(payload["overlap_sec"]) * 16_000)
    )
    buffer_samples = int(chunk_sec * 16_000)

    # Only the blank and the target tokens matter to the CTC DP, so the 50,002-wide emission is
    # gathered down to the columns actually referenced. Keeping the full width would cost
    # ~440MB of stitched CPU tensor on a 3-minute song for no change in the alignment.
    compact_tokens = [blank_id] + sorted({t for t in target_ids if t != blank_id})
    compact_index = {token: position for position, token in enumerate(compact_tokens)}
    column_index = torch.tensor(compact_tokens, dtype=torch.long, device=device)
    compact_targets = [compact_index[t] for t in target_ids]

    # ── 줄 사이 와일드카드 star ──
    # 가사에 없는 가창(추임새·애드립·반복 후렴)을 흡수시켜 이웃 라인이 그 위로 늘어나는 것을
    # 막는다. 프로드가 하는 일이고(``star_tokens`` 기본 켬) 하네스에는 없었다.
    #
    # ``star_mode``:
    #   "all" — 모든 줄 사이. 실측에서 양날이었다 — 인트로 애드립은 옳게 흡수했지만
    #           (ルーキー 0:07 6.65s → 2.73s) 끊김 없이 이어지는 반복 가창에서는 한 사이클을
    #           통째로 먹었다(熱異常 2:07 `黒い星が` 0.47초 건너뜀).
    #   "gap"  — **발성이 끊긴 줄 사이에만** 넣는다. star가 log(1)=0이라 어디서든 실제 토큰보다
    #           싼 것이 원인이므로, 애초에 놓을 자리를 고른다. 성형(``star_prior``)이 실패한
    #           이유(blank가 무료라 동점이 우회)가 여기엔 적용되지 않는다 — 토큰이 없으면
    #           우회할 대상도 없다.
    use_star = bool(payload.get("star"))
    star_mode = payload.get("star_mode") or "all"
    presence = payload.get("presence")
    star_id = len(compact_tokens)

    text_prev = torch.tensor([[model.na]], dtype=torch.long, device=device)
    text_prev_lengths = text_prev.new_full([1], dtype=torch.long, fill_value=1)
    prefix = torch.tensor(
        [[token_to_id[payload["lang_sym"]], token_to_id["<asr>"]]], dtype=torch.long, device=device
    )
    prefix_lengths = prefix.new_full([1], dtype=torch.long, fill_value=prefix.size(1))

    load_reserved_peak = None
    if device == "cuda":
        # ESPnet materializes fp32 weights and only then casts to ``dtype``, so for a half
        # precision run both copies are live at once and the caching allocator holds on to the
        # fp32-sized segments forever after. That load transient is real but one-off; releasing
        # it here is what lets the steady-state alignment footprint actually reflect ``dtype``.
        # Both numbers are reported: the transient still has to fit on the card.
        load_reserved_peak = round(torch.cuda.max_memory_reserved() / 2**20, 1)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Chunks are all padded to the same ``buffer_samples`` length, so several of them can ride
    # one encoder call. Measured on a 279s song (12 chunks): the sequential loop spent 11.66s
    # of the 13.26s total alignment time -- the encoder was latency-bound, not throughput-bound,
    # because each 30s buffer is far too small to saturate the card. Batching trades activation
    # memory for that idle time; ``batch_size`` is the knob and 1 reproduces the old path
    # exactly (same tensors, same order), which is how the equivalence check is run.
    batch_size = max(1, int(payload.get("batch_size") or 1))
    buffer_frames = _expected_audio_frames(buffer_samples)
    want_curves = bool(payload.get("emit_curves"))
    pieces: list[Any] = []
    free_pieces: list[Any] = []
    for offset in range(0, len(windows), batch_size):
        batch = windows[offset : offset + batch_size]
        segments, reals = [], []
        for start, end in batch:
            segment = waveform[start:end]
            real = int(segment.numel())
            reals.append(real)
            if real < buffer_samples:
                segment = torch.nn.functional.pad(segment, (0, buffer_samples - real))
            segments.append(segment)
        count = len(segments)
        speech = torch.stack(segments).to(device=device, dtype=torch_dtype)
        speech_lengths = torch.full([count], speech.size(1), dtype=torch.long, device=device)
        with torch.no_grad():
            enc, _ = model.encode(
                speech=speech,
                speech_lengths=speech_lengths,
                # ``repeat`` rather than ``expand``: the encoder may write into these, and a
                # broadcast view would alias every row of the batch onto one buffer.
                text_prev=text_prev.repeat(count, 1),
                text_prev_lengths=text_prev_lengths.repeat(count),
                prefix=prefix.repeat(count, 1),
                prefix_lengths=prefix_lengths.repeat(count),
            )
            if isinstance(enc, tuple):
                enc = enc[0]
            # Equivalent to ``model.ctc.log_softmax`` except the normalization runs in fp32.
            # A 50,002-way log_softmax in fp16 loses resolution exactly where the CTC DP needs
            # it -- among the near-blank log-probabilities that decide the alignment path.
            logp = torch.log_softmax(model.ctc.ctc_lo(enc).float(), dim=-1)

        # The prefix states sit in front of the audio frames; dropping the wrong count would
        # bias every timestamp, so verify rather than assume.
        surplus = int(logp.shape[1]) - buffer_frames
        if surplus != int(prefix.size(1)):
            raise RuntimeError(
                f"unexpected OWSM encoder length: got {logp.shape[1]} frames, expected "
                f"{buffer_frames} audio frames + {int(prefix.size(1))} prefix frames"
            )
        sliced = logp[:, surplus:, :]
        compact = torch.index_select(sliced, 2, column_index).cpu()
        # 자유 디코드 = 어휘 **전체**의 argmax. 「모델이 실제로 무엇을 들었나」이고, 강제 정렬
        # 점수와 달리 가사에 매이지 않는다. compact로 줄이고 나면 영영 복원할 수 없으므로
        # 여기서만 뽑을 수 있다. 비용은 마지막 축 max 한 번.
        free = None
        if want_curves:
            top_logp, top_id = sliced.max(dim=-1)
            free = torch.stack([top_logp, top_id.to(top_logp.dtype)], dim=-1).cpu()
        for row, real in enumerate(reals):
            valid = max(1, min(buffer_frames, round(buffer_frames * real / buffer_samples)))
            pieces.append(compact[row : row + 1, :valid, :])
            if free is not None:
                free_pieces.append(free[row : row + 1, :valid, :])

    emission = (
        pieces[0] if len(pieces) == 1 else chunking.stitch_chunk_outputs(pieces, windows, n_samples, frame_axis=1)
    )
    emission = emission.float().contiguous()

    import torchaudio.functional as functional

    def _align(target_list: list[int], with_star: bool):
        source = emission
        if with_star:
            # star는 log(1.0)=0 — ``forced_align``이 정규화된 log_probs를 기대하므로 이 트릭은
            # log_softmax **이후에만** 성립한다.
            source = torch.cat(
                [
                    emission,
                    torch.zeros(
                        (emission.shape[0], emission.shape[1], 1), dtype=emission.dtype
                    ),
                ],
                dim=-1,
            ).contiguous()
        tensor = torch.tensor([target_list], dtype=torch.int32)
        tokens, scores = functional.forced_align(source, tensor, blank=0)
        return functional.merge_tokens(tokens[0], scores[0], blank=0)

    def _insert_stars(slots: set[int]) -> tuple[list[int], list[Any]]:
        """``slots``에 든 라인 번호 **앞에** star를 꽂은 (타깃, 소유자)."""
        out_targets: list[int] = []
        out_owners: list[Any] = []
        previous_line: int | None = None
        for token, owner in zip(compact_targets, token_owners):
            if previous_line is not None and owner[0] != previous_line and owner[0] in slots:
                out_targets.append(star_id)
                out_owners.append(None)  # star는 어느 라인의 몫도 아니다
            out_targets.append(token)
            out_owners.append(owner)
            previous_line = owner[0]
        return out_targets, out_owners

    star_slots: set[int] = set()
    if use_star:
        if star_mode == "gap" and presence:
            # ① star 없이 한 번 정렬해 라인 경계를 얻는다. 인코더는 이미 돌았으므로 DP만 든다.
            base_spans = _align(compact_targets, False)
            bounds: dict[int, list[int]] = {}
            for span, owner in zip(base_spans, token_owners):
                edge = bounds.setdefault(owner[0], [int(span.start), int(span.end)])
                edge[0] = min(edge[0], int(span.start))
                edge[1] = max(edge[1], int(span.end))
            # ② 인접 라인 사이가 «끊겼는가» — 우세도가 바닥을 친 프레임이 충분히 이어지는가.
            frame_sec = n_samples / int(emission.shape[1]) / 16_000
            hop = float(payload.get("presence_hop") or 0.01)
            ordered = sorted(bounds)
            for previous_line, next_line in zip(ordered, ordered[1:]):
                lo = int(bounds[previous_line][1] * frame_sec / hop)
                hi = int(bounds[next_line][0] * frame_sec / hop)
                if hi - lo < _STAR_GAP_MIN_FRAMES:
                    continue
                quiet = sum(1 for value in presence[lo:hi] if value < _STAR_GAP_LEVEL)
                if quiet >= _STAR_GAP_MIN_FRAMES:
                    star_slots.add(next_line)
        else:
            star_slots = {owner[0] for owner in token_owners}
        if star_slots:
            compact_targets, token_owners = _insert_stars(star_slots)

    token_spans = _align(compact_targets, bool(use_star and star_slots))
    if len(token_spans) != len(compact_targets):
        raise RuntimeError(
            f"OWSM produced {len(token_spans)} spans for {len(compact_targets)} target tokens"
        )

    ratio = n_samples / int(emission.shape[1]) / 16_000

    # ── spans -> per-character segs ──
    out_lines: list[dict[str, Any]] = [
        {"segs": [], "total_chars": total, "matched_chars": 0, "tokens": 0, "_scores": []}
        for total in line_totals
    ]
    star_frames = 0
    for span, owner in zip(token_spans, token_owners):
        if owner is None:  # star가 흡수한 구간 — 어느 라인에도 안 준다
            star_frames += int(span.end) - int(span.start)
            continue
        line_index, covered = owner
        start_sec = float(span.start) * ratio
        end_sec = float(span.end) * ratio
        entry = out_lines[line_index]
        entry["tokens"] += 1
        entry["_scores"].append(float(span.score))
        step = (end_sec - start_sec) / len(covered)
        for position, char_index in enumerate(covered):
            entry["segs"].append(
                {
                    "t": lines[line_index][char_index],
                    "start": round(start_sec + position * step, 3),
                    "end": round(start_sec + (position + 1) * step, 3),
                }
            )
        entry["matched_chars"] += len(covered)

    for entry in out_lines:
        scores = entry.pop("_scores")
        entry["segs"].sort(key=lambda seg: seg["start"])
        entry["mean_log_score"] = (sum(scores) / len(scores)) if scores else None

    # ── 추임새 진단용 프레임 곡선 ──
    # log_softmax는 어휘 **전체**에서 돌고 나서 gather했으므로, 압축된 열들은 여전히 전체
    # 정규화 상태다. 따라서 프레임마다
    #     p_기타 = 1 − p_blank − Σp_가사토큰
    # 이 「모델이 가사에 없는 무언가를 들었다」는 질량이 된다 — 추가 연산 없이 얻는 신호.
    curves = None
    if want_curves:
        free = (
            free_pieces[0]
            if len(free_pieces) == 1
            else chunking.stitch_chunk_outputs(free_pieces, windows, n_samples, frame_axis=1)
        )
        probs = emission[0].exp()
        top_ids = free[0, :, 1].long().tolist()
        curves = {
            "blank": [round(float(x), 5) for x in probs[:, 0].tolist()],
            "lyric": [round(float(x), 5) for x in probs[:, 1:].sum(dim=1).tolist()],
            "top_p": [round(float(x), 5) for x in free[0, :, 0].float().exp().tolist()],
            "top_id": top_ids,
            "vocab": {str(i): token_list[i] for i in sorted(set(top_ids))},
        }

    response = {
        "audio_sec": round(audio_sec, 3),
        "frames": int(emission.shape[1]),
        "frame_sec": round(ratio, 6),
        "chunks": len(windows),
        "chunk_sec": chunk_sec,
        "load_sec": load_sec,
        "dtype": dtype,
        "vocab_size": len(token_list),
        "compact_vocab_size": len(compact_tokens),
        "target_tokens": len(compact_targets),
        "star_tokens": sum(1 for owner in token_owners if owner is None),
        "star_mode": star_mode if use_star else None,
        "star_slots": len(star_slots),
        "star_frames": star_frames,
        "lines": out_lines,
    }
    if curves is not None:
        response["curves"] = curves

    # ── 압축 emission 덤프(진단 전용) ──
    # 인코더는 정렬 시간의 88%다. 탐지 규칙을 바꿔 가며 재실험하려면 emission을 한 번 남겨
    # 두고 DP만 다시 도는 편이 유일하게 현실적이다. 열은 [blank, *정렬 타깃 토큰]이고
    # ``compact_targets``의 인덱스가 그대로 이 열을 가리킨다.
    dump_dir = payload.get("dump_dir")
    if dump_dir:
        import numpy as np

        target_dir = Path(dump_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        np.save(target_dir / "emission.npy", emission[0].numpy().astype(np.float16))
        (target_dir / "targets.json").write_text(
            json.dumps(
                {
                    "compact_targets": compact_targets,
                    "compact_tokens": compact_tokens,
                    "token_pieces": [token_list[t] for t in compact_tokens],
                    "token_owners": [None if o is None else [o[0], o[1]] for o in token_owners],
                    "lines": lines,
                    "frame_sec": ratio,
                    "frames": int(emission.shape[1]),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        response["dump_dir"] = str(target_dir)
    if device == "cuda":
        response["vram_peak_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        response["vram_reserved_peak_mb"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
        response["load_reserved_peak_mb"] = load_reserved_peak

    response_path.write_text(json.dumps(response, ensure_ascii=False), encoding="utf-8")
    return 0


def _parse_args() -> Any:
    import argparse

    parser = argparse.ArgumentParser(description="Internal worker for the OWSM-CTC benchmark adapter")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    args = parser.parse_args()
    if not args.worker:
        parser.error("this module is invoked by the benchmark adapter; pass --worker")
    return args


if __name__ == "__main__":
    _args = _parse_args()
    raise SystemExit(_worker_main(Path(_args.request), Path(_args.response)))
