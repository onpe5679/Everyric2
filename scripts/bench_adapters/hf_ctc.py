"""Standalone Hugging Face CTC adapters for the alignment benchmark.

This module deliberately does not participate in the production alignment path.  The
benchmark session can opt in with::

    from scripts.bench_adapters.hf_ctc import register
    register(ALIGNERS)

The heavy imports (torch, torchaudio, transformers, and pykakasi) are all delayed until
an adapter is used.  That keeps ``benchmark_alignment.py --help`` and ``--list`` cheap.
"""

from __future__ import annotations

import json
import logging
import math
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.benchmark_alignment import AlignOut, AlignerAdapter, VramProbe

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16_000
LOW_CONF_COVERAGE = 0.90

# Whole-song forward passes make encoder activation memory grow with audio length -- a 5090
# smoke test hit 31.8/32.6 GB. everyric2.alignment.ctc_engine's align_chunk_sec default
# (360.0s, tuned for MMS-1B on a 24GB 3090) is NOT safe here: measured directly on this repo's
# candidates, hf-ko-42maru alone used 22.7GB process VRAM (32.6GB device -- essentially the
# whole 5090) unchunked on an ordinary 176s song, while hf-kkonjeong used 3.1GB and
# hf-jonatasgrosman-ja used 8.8GB on a 294s song -- VRAM-per-second-of-audio varies by roughly
# an order of magnitude across these HF architectures (attention-heavy Conformer vs. CNN-heavy
# wav2vec2), so one global constant has to be sized for the worst case, not the average.
# Re-measured with the actual candidates at smaller chunk sizes on that same 176s hf-ko-42maru
# case: 90s -> 6.5GB, 60s -> 3.3GB, 45s -> 2.2GB. 60s keeps every candidate comfortably under
# the task's 9GB VRAM promotion gate with margin, independent of total song length (peak scales
# with chunk length only -- see stitch_chunk_outputs), while staying long enough that most
# individual lyric lines fall inside one chunk's confident-middle region.
ALIGN_CHUNK_SEC = 60.0
ALIGN_CHUNK_OVERLAP_SEC = 5.0
_HIRAGANA_CONVERTER: Any | None = None

# NFD emits canonical jamo (ᄀ/ᅡ/ᆨ), while Kkonjeong and HJLee use compatibility
# jamo (ㄱ/ㅏ/ㄱ). NFKC does not perform this cross-block conversion, so keep the
# mapping explicit. The final-consonant mapping is spelled out because those
# compatibility characters do not carry canonical decompositions.
_CHOSEONG_COMPAT = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_JUNGSEONG_COMPAT = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_JONGSEONG_COMPAT = "ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"
_MODERN_TO_COMPAT_JAMO: dict[str, str] = {
    **{chr(0x1100 + i): ch for i, ch in enumerate(_CHOSEONG_COMPAT)},
    **{chr(0x1161 + i): ch for i, ch in enumerate(_JUNGSEONG_COMPAT)},
    **{chr(0x11A8 + i): ch for i, ch in enumerate(_JONGSEONG_COMPAT)},
}

# MFA-style 43-entry Korean phone vocabulary for slplab/wav2vec2-xls-r-300m_phone-mfa_korean
# (Wav2Vec2PhonemeCTCTokenizer; vocab ids 0-42 read directly from the cached vocab.json --
# 40 phone symbols + | + [UNK] + [PAD]). The 21 jungseong letters collapse to 17 phones via
# the three standard modern-Seoul vowel mergers (ㅐ/ㅔ, ㅒ/ㅖ, ㅙ/ㅚ/ㅞ all share one token);
# the 27 jongseong letters neutralize to the 7 codas of 표준 발음법 대표음 규칙. This text is
# a Hangul transliteration of Japanese lyrics (see the pron-hangul-local input mode), so
# Korean liaison/tensification/nasalization rules do not apply -- each syllable block
# converts independently of its neighbours.
_CHOSEONG_TO_PHONE_MFA: dict[str, str | None] = {
    "ㄱ": "G", "ㄲ": "GG", "ㄴ": "N", "ㄷ": "D", "ㄸ": "DD", "ㄹ": "R", "ㅁ": "M",
    "ㅂ": "B", "ㅃ": "BB", "ㅅ": "S", "ㅆ": "SS", "ㅇ": None, "ㅈ": "J", "ㅉ": "JJ",
    "ㅊ": "CHh", "ㅋ": "Kh", "ㅌ": "Th", "ㅍ": "Ph", "ㅎ": "H",
}
_JUNGSEONG_TO_PHONE_MFA: dict[str, str] = {
    "ㅏ": "A", "ㅐ": "E", "ㅑ": "iA", "ㅒ": "iE", "ㅓ": "EO", "ㅔ": "E", "ㅕ": "iEO",
    "ㅖ": "iE", "ㅗ": "O", "ㅘ": "oA", "ㅙ": "oE", "ㅚ": "oE", "ㅛ": "iO", "ㅜ": "U",
    "ㅝ": "uEO", "ㅞ": "oE", "ㅟ": "uI", "ㅠ": "iU", "ㅡ": "EU", "ㅢ": "euI", "ㅣ": "I",
}
_JONGSEONG_TO_PHONE_MFA: dict[str, str] = {
    "ㄱ": "k", "ㄲ": "k", "ㄳ": "k", "ㄴ": "N", "ㄵ": "N", "ㄶ": "N", "ㄷ": "t",
    "ㄹ": "L", "ㄺ": "k", "ㄻ": "M", "ㄼ": "L", "ㄽ": "L", "ㄾ": "L", "ㄿ": "p",
    "ㅀ": "L", "ㅁ": "M", "ㅂ": "p", "ㅄ": "p", "ㅅ": "t", "ㅆ": "t", "ㅇ": "NG",
    "ㅈ": "t", "ㅊ": "t", "ㅋ": "k", "ㅌ": "t", "ㅍ": "p", "ㅎ": "t",
}


@dataclass(frozen=True)
class HFCTCModelConfig:
    """Configuration and observed tokenizer family for one HF candidate."""

    name: str
    model_id: str
    language: str
    vocab_unit: str
    # ``compat_jamo`` uses U+3130 compatibility jamo; ``modern_jamo`` uses
    # U+1100/U+1160/U+11A8 canonical jamo as emitted by NFD.
    jamo_style: str | None = None
    # Selects a Hangul-syllable-to-phone decomposition table (see ``_hangul_phonemes``),
    # analogous to ``jamo_style`` but targeting a phone-level rather than jamo-level vocab.
    phoneme_style: str | None = None
    hiragana_conversion: bool = False
    vocab_note: str = ""
    # Multilingual candidates carry every benchmark stratum, so ``align`` must not reject a
    # song for disagreeing with ``language``; that field stays only as a metadata label.
    multilingual: bool = False
    # Per-candidate override of the module-wide chunk length. Models trained on a fixed
    # utterance budget (OWSM-CTC's 30s buffer, omniASR's short corpus) align better when the
    # inference window matches training rather than the 60s VRAM-driven default.
    align_chunk_sec: float | None = None


CANDIDATES: tuple[HFCTCModelConfig, ...] = (
    HFCTCModelConfig(
        name="hf-en-960h",
        model_id="facebook/wav2vec2-base-960h",
        language="en",
        vocab_unit="char",
        vocab_note=(
            "32 entries: uppercase A-Z, apostrophe, | delimiter, specials. "
            "영어 네이티브 대조군 — 다국어 후보들의 en 붕괴가 스크립트 prior 문제인지 검증용. "
            "라이선스 Apache-2.0."
        ),
    ),
    HFCTCModelConfig(
        name="hf-kkonjeong",
        model_id="Kkonjeong/wav2vec2-base-korean",
        language="ko",
        vocab_unit="jamo",
        jamo_style="compat_jamo",
        vocab_note="54 entries: compatibility jamo plus [PAD]/[UNK] and a space token.",
    ),
    # Chunk-stitch ablation. Every hf_ctc candidate caps near 73% regardless of vocab size or
    # parameter count, which points at this adapter rather than at the acoustic models; the
    # 60s window plus 5s-overlap stitching is the most likely shared ceiling. kkonjeong measured
    # 3.1GB unchunked on a 294s song, so a whole-song window is safe for this one candidate.
    HFCTCModelConfig(
        name="hf-kkonjeong-nochunk",
        model_id="Kkonjeong/wav2vec2-base-korean",
        language="ko",
        vocab_unit="jamo",
        jamo_style="compat_jamo",
        align_chunk_sec=1200.0,
        vocab_note="hf-kkonjeong with a whole-song alignment window (chunk-stitch ablation).",
    ),
    HFCTCModelConfig(
        name="hf-kresnik",
        model_id="kresnik/wav2vec2-large-xlsr-korean",
        language="ko",
        vocab_unit="syllable",
        vocab_note="1,205 entries: 1,202 Hangul syllables plus |, [UNK], and [PAD].",
    ),
    # Capacity-scaling control for hf-kkonjeong (94.4M) / hf-kresnik (300M): same Zeroth-Korean
    # training data, 1B parameters. Isolates "does nemo win on model size?" from data volume.
    # Syllable vocab shares kresnik's Japanese-you'on OOV risk (e.g. 캬 is absent) -- check
    # coverage before reading the score.
    HFCTCModelConfig(
        name="hf-anantoj-ko-1b",
        model_id="anantoj/wav2vec2-xls-r-1b-korean",
        language="ko",
        vocab_unit="syllable",
        vocab_note="1,205 entries: Hangul syllables plus |, [UNK], and [PAD] (same set as kresnik).",
    ),
    HFCTCModelConfig(
        name="hf-xls-r-ko",
        model_id="w11wo/wav2vec2-xls-r-300m-korean",
        language="ko",
        vocab_unit="syllable",
        vocab_note="1,205 entries: Hangul syllable vocabulary with |, [UNK], and [PAD].",
    ),
    HFCTCModelConfig(
        name="hf-hjlee-ko",
        model_id="thisisHJLee/wav2vec2-large-xls-r-300m-korean-g",
        language="ko",
        vocab_unit="jamo",
        jamo_style="compat_jamo",
        vocab_note="51 entries: compatibility jamo, |, [PAD], and [UNK].",
    ),
    HFCTCModelConfig(
        name="hf-ko-42maru",
        model_id="42MARU/ko-42maru-wav2vec2-conformer-del-1s",
        language="ko",
        vocab_unit="jamo",
        jamo_style="modern_jamo",
        vocab_note="72 entries: canonical choseong/jungseong/jongseong jamo plus CTC specials and |.",
    ),
    # Capacity-scaling companion to hf-kkonjeong (54-way jamo) and hf-kresnik (1,205-way
    # syllable) at the opposite vocab-size extreme: 43-way phone. Tests whether a smaller
    # CTC output alphabet concentrates probability mass on target tokens and survives the
    # blank-vs-target competition better than a larger one (see hf-kresnik/hf-anantoj-ko-1b
    # notes on Japanese-you'on OOV under a syllable vocab -- this candidate has zero OOV risk
    # by construction since every Hangul syllable decomposes onto in-vocab phones).
    HFCTCModelConfig(
        name="hf-slplab-phone-mfa",
        model_id="slplab/wav2vec2-xls-r-300m_phone-mfa_korean",
        language="ko",
        vocab_unit="phoneme",
        phoneme_style="mfa_ko_43",
        vocab_note=(
            "43 entries: 40 MFA-style Korean phone symbols (7-way coda neutralization; "
            "jungseong mergers ㅐ/ㅔ, ㅒ/ㅖ, ㅙ/ㅚ/ㅞ) plus |, [UNK], [PAD]. PER 3.88% on "
            "Zeroth-Korean per model card. Apache-2.0."
        ),
    ),
    HFCTCModelConfig(
        name="hf-jonatasgrosman-ja",
        model_id="jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
        language="ja",
        vocab_unit="syllable",
        vocab_note="2,341 single-character entries covering hiragana, katakana, Han, and a small Latin set.",
    ),
    HFCTCModelConfig(
        name="hf-reazon-hubert-base",
        model_id="reazon-research/japanese-hubert-base-k2-rs35kh",
        language="ja",
        vocab_unit="syllable",
        vocab_note="2,341 single-character entries covering hiragana, katakana, Han, and specials.",
    ),
    HFCTCModelConfig(
        name="hf-reazon-wav2vec2-base",
        model_id="reazon-research/japanese-wav2vec2-base-rs35kh",
        language="ja",
        vocab_unit="other: mixed character/SentencePiece",
        vocab_note="3,000 entries: Japanese/Latin single characters plus multi-character ▁ subwords.",
    ),
    HFCTCModelConfig(
        name="hf-reazon-wav2vec2-large",
        model_id="reazon-research/japanese-wav2vec2-large-rs35kh",
        language="ja",
        vocab_unit="other: mixed character/SentencePiece",
        vocab_note="3,000 entries: same mixed single-character and ▁ subword vocabulary as base.",
    ),
    HFCTCModelConfig(
        name="hf-ttop324-ja",
        model_id="ttop324/wav2vec2-live-japanese",
        language="ja",
        vocab_unit="hiragana",
        hiragana_conversion=True,
        vocab_note="100 entries: hiragana, one prolonged-sound mark, a small Latin set, 々, and CTC specials; no Han.",
    ),
)


@dataclass
class _PreparedUnit:
    """One source character and the one-or-more CTC tokens used to align it."""

    source: str
    tokens: list[str]
    alignable: bool
    matched: bool


@dataclass
class _PreparedLine:
    text: str
    units: list[_PreparedUnit]
    converted: bool = False

    @property
    def total_chars(self) -> int:
        return sum(unit.alignable for unit in self.units)

    @property
    def matched_chars(self) -> int:
        return sum(unit.alignable and unit.matched for unit in self.units)

    @property
    def coverage(self) -> float:
        if self.total_chars == 0:
            return 0.0
        return self.matched_chars / self.total_chars


class HFCTCAligner(AlignerAdapter):
    """General-purpose HF CTC forced aligner used only by the benchmark harness."""

    name: str = ""
    model_id: str = ""
    model_config: HFCTCModelConfig

    def __init__(self, config: HFCTCModelConfig | None = None) -> None:
        if config is None:
            config = self.model_config
        self.model_config = config
        self.name = config.name
        self.model_id = config.model_id
        self._processor: Any | None = None
        self._model: Any | None = None
        self._vocab: dict[str, int] | None = None
        self._vocab_path: Path | None = None
        self._load_seconds: float | None = None

    def inspect_vocab(self) -> dict[str, Any]:
        """Read the cached vocab without loading model weights or running inference."""

        vocab = self._ensure_vocab()
        counts: dict[str, int] = {}
        for token in vocab:
            kind = _token_kind(token)
            counts[kind] = counts.get(kind, 0) + 1
        return {
            "model": self.model_id,
            "adapter": self.name,
            "vocab_unit": self.model_config.vocab_unit,
            "vocab_size": len(vocab),
            "vocab_path": None if self._vocab_path is None else str(self._vocab_path),
            "token_kinds": counts,
            "vocab_note": self.model_config.vocab_note,
        }

    def preprocess_lines(self, lyrics: str) -> list[dict[str, Any]]:
        """Prepare lyrics and return coverage details without model inference.

        Coverage counts only letter/number-like lyric characters. Whitespace and
        punctuation are separators rather than CTC targets. A jamo candidate counts a
        source Hangul syllable as matched only when all of its decomposed jamo tokens
        are present, which keeps the metric in source-character units.
        """

        prepared = self._prepare_lines(_split_lyrics(lyrics))
        return [self._line_metadata(line) for line in prepared]

    def align(self, vocals_path: Path, lyrics: str, language: str) -> AlignOut:
        """Run HF CTC emission (GPU if available, else CPU) and return one timed result per lyric line."""

        if (
            not self.model_config.multilingual
            and language
            and _base_language(language) != self.model_config.language
        ):
            raise ValueError(
                f"{self.name} is a {self.model_config.language} candidate, not {language!r}"
            )

        lines = _split_lyrics(lyrics)
        if not lines:
            raise ValueError("lyrics produced zero non-empty lines")
        prepared = self._prepare_lines(lines)
        valid_units = [unit for line in prepared for unit in line.units if unit.matched]
        if not valid_units:
            raise ValueError(f"{self.name} found no in-vocabulary lyric characters")

        # Model load is a one-time, cached cost (see ``_ensure_model``) -- keep it outside the
        # probe so a candidate's first-call VRAM spike doesn't get attributed to "alignment".
        processor, model = self._ensure_model()
        device = next(model.parameters()).device

        started = time.perf_counter()
        waveform, sample_rate = self._load_audio(vocals_path)

        token_ids: list[int] = []
        unit_token_ranges: list[tuple[int, int] | None] = []
        for line in prepared:
            for unit in line.units:
                if not unit.matched:
                    unit_token_ranges.append(None)
                    continue
                first = len(token_ids)
                token_ids.extend(self._vocab[token] for token in unit.tokens)  # type: ignore[index]
                unit_token_ranges.append((first, len(token_ids)))

        import torch
        import torchaudio.functional as functional

        with VramProbe() as probe:
            emission, n_chunks = self._chunked_emission(processor, model, waveform, sample_rate, device)

            if max(token_ids, default=-1) >= emission.shape[-1]:
                raise RuntimeError(
                    f"{self.name} vocab id exceeds model output width: "
                    f"max_id={max(token_ids)} width={emission.shape[-1]}"
                )

            blank_id = self._blank_id(model)
            targets = torch.tensor([token_ids], dtype=torch.int32, device=emission.device)
            try:
                aligned_tokens, alignment_scores = functional.forced_align(
                    emission, targets, blank=blank_id
                )
                token_spans = functional.merge_tokens(
                    aligned_tokens[0], alignment_scores[0], blank=blank_id
                )
            except Exception as exc:
                raise RuntimeError(f"{self.name} forced alignment failed: {exc}") from exc

        ratio = waveform.numel() / emission.shape[1] / TARGET_SAMPLE_RATE
        line_results = self._line_results(
            prepared, unit_token_ranges, token_spans, ratio, waveform.numel() / TARGET_SAMPLE_RATE
        )
        quality_score, quality_meta = _quality_score(line_results)
        elapsed = time.perf_counter() - started
        return AlignOut(
            lines=line_results,
            elapsed_sec=round(elapsed, 2),
            vram_peak_mb=probe.process_peak_mb,
            vram_device_peak_mb=probe.device_peak_mb,
            quality_score=quality_score,
            meta={
                "model": self.model_id,
                "adapter": self.name,
                "language": self.model_config.language,
                "vocab_unit": self.model_config.vocab_unit,
                "vocab_size": len(self._ensure_vocab()),
                "vocab_path": None if self._vocab_path is None else str(self._vocab_path),
                "model_load_sec": self._load_seconds,
                "audio_sec": round(waveform.numel() / TARGET_SAMPLE_RATE, 3),
                "sample_rate": sample_rate,
                "coverage_threshold": LOW_CONF_COVERAGE,
                "coverage_denominator": "letter/number-like source characters",
                "align_chunks": n_chunks,
                "align_chunk_sec": self._chunk_sec(),
                "preprocessing": _preprocessing_label(self.model_config),
                "quality": quality_meta,
                "hiragana_conversion_approximation": self.model_config.hiragana_conversion,
            },
        )

    def _chunk_sec(self) -> float:
        override = self.model_config.align_chunk_sec
        return ALIGN_CHUNK_SEC if override is None else float(override)

    def _chunked_emission(
        self, processor: Any, model: Any, waveform: Any, sample_rate: int, device: Any
    ) -> tuple[Any, int]:
        """(Log-softmax CTC emission [1, T, V], chunk count) -- peak VRAM bounded by chunk length.

        Splits the waveform into overlapping windows (``ALIGN_CHUNK_SEC``/
        ``ALIGN_CHUNK_OVERLAP_SEC``, mirrors ctc_engine's ``align_chunk_sec``/
        ``align_chunk_overlap_sec``), runs a separate forward pass per window, moves each
        chunk's emission to CPU immediately, then stitches the confident-middle slice of each
        chunk back into one full-length sequence with
        ``everyric2.audio.chunking.stitch_chunk_outputs``. A single window (short audio, or
        chunking disabled) returns the emission on ``device`` unmoved -- identical to the old
        unchunked path, so short-song results and speed are unaffected.
        """
        from everyric2.audio.chunking import plan_chunk_windows, stitch_chunk_outputs

        n = int(waveform.numel())
        windows = plan_chunk_windows(
            n,
            int(self._chunk_sec() * TARGET_SAMPLE_RATE),
            int(ALIGN_CHUNK_OVERLAP_SEC * TARGET_SAMPLE_RATE),
        )

        def _forward(chunk: Any) -> Any:
            processor_inputs = processor(
                chunk.numpy(), sampling_rate=sample_rate, return_tensors="pt", padding=True
            )
            # Half-precision VRAM-reduction variants (e.g. omniasr-ctc-bf16) cast the model's
            # own parameters but leave the feature extractor's fp32 output alone; matching the
            # floating inputs to the model's dtype here is a no-op for every fp32 candidate
            # (dtype already matches) and keeps this shared path from throwing a scalar-type
            # mismatch for a cast one.
            model_dtype = next(model.parameters()).dtype
            model_inputs: dict[str, Any] = {}
            for key, value in processor_inputs.items():
                if not hasattr(value, "to"):
                    continue
                if value.is_floating_point() and value.dtype != model_dtype:
                    value = value.to(dtype=model_dtype)
                model_inputs[key] = value.to(device)
            with _torch_inference_mode():
                logits = model(**model_inputs).logits
            return _log_softmax(logits)

        if len(windows) == 1:
            return _forward(waveform), 1

        pieces = [_forward(waveform[s:e].contiguous()).cpu() for s, e in windows]
        return stitch_chunk_outputs(pieces, windows, n, frame_axis=1), len(windows)

    def _ensure_vocab(self) -> dict[str, int]:
        if self._vocab is not None:
            return self._vocab
        path = _find_cached_file(self.model_id, "vocab.json")
        if path is None:
            raise FileNotFoundError(
                f"vocab.json for {self.model_id} was not found under HF_HOME/TRANSFORMERS_CACHE"
            )
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"vocab.json is not an object: {path}")
        self._vocab = {str(token): int(token_id) for token, token_id in raw.items()}
        self._vocab_path = path
        return self._vocab

    def _ensure_processor(self) -> Any:
        if self._processor is None:
            from transformers import AutoProcessor

            self._ensure_vocab()
            self._processor = AutoProcessor.from_pretrained(
                self.model_id, local_files_only=True
            )
        return self._processor

    def _ensure_model(self) -> tuple[Any, Any]:
        if self._model is not None:
            return self._ensure_processor(), self._model

        import torch
        from transformers import AutoModelForCTC

        started = time.perf_counter()
        processor = self._ensure_processor()
        self._model = AutoModelForCTC.from_pretrained(
            self.model_id, local_files_only=True, torch_dtype=torch.float32
        )
        self._model.to("cuda" if torch.cuda.is_available() else "cpu")
        self._model.eval()
        self._load_seconds = round(time.perf_counter() - started, 2)
        return processor, self._model

    def _load_audio(self, path: Path) -> tuple[Any, int]:
        import torch
        import torchaudio

        waveform, sample_rate = torchaudio.load(str(path))
        if waveform.ndim != 2 or waveform.shape[0] == 0:
            raise ValueError(f"expected audio shaped [channels, samples], got {waveform.shape}")
        waveform = waveform.mean(dim=0)
        if sample_rate != TARGET_SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sample_rate, TARGET_SAMPLE_RATE)
            sample_rate = TARGET_SAMPLE_RATE
        return waveform.to(dtype=torch.float32, device="cpu").contiguous(), sample_rate

    def _prepare_lines(self, lines: list[str]) -> list[_PreparedLine]:
        vocab = self._ensure_vocab()
        prepared: list[_PreparedLine] = []
        for line in lines:
            units: list[_PreparedUnit] = []
            converted = False
            for char in line:
                if self.model_config.hiragana_conversion:
                    source_tokens = _to_hiragana(char)
                    converted = converted or source_tokens != [char]
                elif self.model_config.jamo_style:
                    source_tokens = _hangul_tokens(char, self.model_config.jamo_style)
                elif self.model_config.phoneme_style:
                    source_tokens = _hangul_phonemes(char, self.model_config.phoneme_style)
                else:
                    source_tokens = [char]

                alignable = _is_alignment_character(char)
                if not alignable:
                    units.append(_PreparedUnit(char, [], False, False))
                    continue
                normalized = [_lookup_token(token, vocab) for token in source_tokens]
                tokens = [token for token in normalized if token is not None]
                matched = len(tokens) == len(source_tokens) and bool(tokens)
                units.append(_PreparedUnit(char, tokens if matched else [], True, matched))
            prepared.append(_PreparedLine(line, units, converted))
        return prepared

    def _line_results(
        self,
        prepared: list[_PreparedLine],
        unit_token_ranges: list[tuple[int, int] | None],
        token_spans: list[Any],
        ratio: float,
        audio_length: float,
    ) -> list[dict[str, Any]]:
        line_times: list[tuple[float, float, float, int] | None] = []
        line_segs: list[list[dict[str, Any]]] = []
        cursor = 0
        for line in prepared:
            starts: list[float] = []
            ends: list[float] = []
            scores: list[float] = []
            segs: list[dict[str, Any]] = []
            for unit in line.units:
                token_range = unit_token_ranges[cursor]
                cursor += 1
                if token_range is None:
                    continue
                start_idx, end_idx = token_range
                if end_idx > len(token_spans):
                    logger.warning(
                        "%s returned %d merged spans for %d target tokens; line timing is incomplete",
                        self.name,
                        len(token_spans),
                        end_idx,
                    )
                    continue
                spans = token_spans[start_idx:end_idx]
                if not spans:
                    continue
                u_start = float(spans[0].start) * ratio
                u_end = float(spans[-1].end) * ratio
                starts.append(u_start)
                ends.append(u_end)
                scores.extend(float(span.score) for span in spans)
                # 음절(원문 글자) 단위 실측 스팬 — 카라오케 품질 비교·검수 뷰어의 원료.
                # 자모 vocab 모델은 한 글자의 자모 토큰 전체를 묶은 스팬이라 곧 음절 스팬이다.
                segs.append({"t": unit.source, "start": round(u_start, 3), "end": round(u_end, 3)})
            if starts:
                mean_log_score = sum(scores) / len(scores) if scores else -math.inf
                line_times.append((starts[0], ends[-1], _confidence(mean_log_score), line.matched_chars))
            else:
                line_times.append(None)
            line_segs.append(segs)

        interpolated = _interpolate_times(line_times, audio_length)
        results: list[dict[str, Any]] = []
        for line, timing, segs in zip(prepared, interpolated, line_segs):
            start, end, confidence, matched = timing
            coverage = line.coverage
            results.append(
                {
                    "text": line.text,
                    "start": start,
                    "end": end,
                    "segs": segs,
                    "confidence": None if confidence is None else round(confidence, 6),
                    "measured": matched > 0 and confidence is not None,
                    "chars": matched,
                    "coverage": round(coverage, 6),
                    "low_conf": coverage < LOW_CONF_COVERAGE,
                    "meta": {
                        "model": self.name,
                        "vocab_unit": self.model_config.vocab_unit,
                        "preprocessing": _preprocessing_label(self.model_config),
                        "matched_chars": matched,
                        "total_chars": line.total_chars,
                        "coverage": round(coverage, 6),
                        "low_conf": coverage < LOW_CONF_COVERAGE,
                        "hiragana_conversion_approximation": self.model_config.hiragana_conversion,
                    },
                }
            )
        return results

    def _line_metadata(self, line: _PreparedLine) -> dict[str, Any]:
        return {
            "text": line.text,
            "matched_chars": line.matched_chars,
            "total_chars": line.total_chars,
            "coverage": round(line.coverage, 6),
            "low_conf": line.coverage < LOW_CONF_COVERAGE,
            "vocab_unit": self.model_config.vocab_unit,
            "preprocessing": _preprocessing_label(self.model_config),
            "hiragana_conversion_approximation": self.model_config.hiragana_conversion,
            "converted": line.converted,
        }

    @staticmethod
    def _blank_id(model: Any) -> int:
        blank_id = getattr(getattr(model, "config", None), "pad_token_id", None)
        if blank_id is None:
            raise ValueError("CTC model has no pad_token_id to use as the blank token")
        return int(blank_id)


def register(aligner_registry: dict[str, type[AlignerAdapter]]) -> None:
    """Register all ten HF candidates in a harness aligner registry."""

    for config in CANDIDATES:
        aligner_registry[config.name] = _candidate_class(config)


def _candidate_class(config: HFCTCModelConfig) -> type[HFCTCAligner]:
    class CandidateHFCTCAligner(HFCTCAligner):
        name = config.name
        model_id = config.model_id
        model_config = config

    CandidateHFCTCAligner.__name__ = "HFCTC_" + config.name.removeprefix("hf-").replace("-", "_")
    CandidateHFCTCAligner.__qualname__ = CandidateHFCTCAligner.__name__
    return CandidateHFCTCAligner


def _split_lyrics(lyrics: str) -> list[str]:
    return [line.strip() for line in lyrics.strip().splitlines() if line.strip()]


def _base_language(language: str) -> str:
    return language.removesuffix("_mms").strip().lower()


def _preprocessing_label(config: HFCTCModelConfig) -> str:
    if config.hiragana_conversion:
        return "pykakasi per-source-character hiragana conversion (approximation)"
    if config.jamo_style == "compat_jamo":
        return "Hangul syllable decomposition to compatibility jamo"
    if config.jamo_style == "modern_jamo":
        return "Hangul syllable NFD decomposition to canonical jamo"
    if config.phoneme_style:
        return "Hangul syllable decomposition to MFA-style Korean phones (7-way coda neutralization)"
    if config.vocab_unit.startswith("other:"):
        return "direct in-vocab single-character targets; OOV characters skipped"
    return "direct in-vocab character targets; OOV characters skipped"


def _token_kind(token: str) -> str:
    if token == "|" or token.startswith("<") or token.startswith("["):
        return "special"
    if len(token) != 1:
        return "subword"
    code = ord(token)
    if 0xAC00 <= code <= 0xD7A3:
        return "hangul_syllable"
    if (0x1100 <= code <= 0x11FF) or (0x3130 <= code <= 0x318F):
        return "jamo"
    if 0x3040 <= code <= 0x309F:
        return "hiragana"
    if 0x30A0 <= code <= 0x30FF:
        return "katakana"
    if 0x4E00 <= code <= 0x9FFF:
        return "han"
    return "other_char"


def _find_cached_file(model_id: str, filename: str) -> Path | None:
    """Find the newest snapshot file using the configured HF cache roots."""

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

    slug = "models--" + model_id.replace("/", "--")
    matches: list[Path] = []
    for root in roots:
        matches.extend(root.glob(f"{slug}/snapshots/*/{filename}"))
    return max(matches, key=lambda path: path.stat().st_mtime) if matches else None


def _hangul_tokens(char: str, style: str) -> list[str]:
    if not (0xAC00 <= ord(char) <= 0xD7A3):
        return [char]
    decomposed = unicodedata.normalize("NFD", char)
    if style == "modern_jamo":
        return list(decomposed)
    if style == "compat_jamo":
        # NFKC does NOT do this conversion: Unicode's compatibility decomposition maps
        # compat jamo (U+3131-...) -> modern/canonical jamo (U+1100-...), i.e. the
        # opposite direction. Running NFKC on the already-canonical output of NFD is a
        # no-op, so every compat_jamo candidate (Kkonjeong, thisisHJLee) previously saw
        # 0% vocab coverage. Map explicitly using the standard KS X 1001 jamo ordering.
        return [_MODERN_TO_COMPAT_JAMO.get(token, token) for token in decomposed]
    raise ValueError(f"unknown jamo style: {style}")


def _hangul_phonemes(char: str, style: str) -> list[str]:
    """One Hangul syllable -> its MFA-style phone tokens (see the module-level phone tables).

    Uses the algorithmic choseong/jungseong/jongseong index decomposition (not NFD) so the
    result indexes directly into ``_CHOSEONG_COMPAT``/``_JUNGSEONG_COMPAT``/``_JONGSEONG_COMPAT``
    -- the same KS X 1001 ordering already used by the compat-jamo candidates above. A null
    onset (ㅇ) emits no token; a syllable with no coda emits none for that slot.
    """
    if not (0xAC00 <= ord(char) <= 0xD7A3):
        return [char]
    if style != "mfa_ko_43":
        raise ValueError(f"unknown phoneme style: {style}")
    code = ord(char) - 0xAC00
    cho_idx, remainder = divmod(code, 21 * 28)
    jung_idx, jong_idx = divmod(remainder, 28)
    tokens: list[str] = []
    onset = _CHOSEONG_TO_PHONE_MFA[_CHOSEONG_COMPAT[cho_idx]]
    if onset is not None:
        tokens.append(onset)
    tokens.append(_JUNGSEONG_TO_PHONE_MFA[_JUNGSEONG_COMPAT[jung_idx]])
    if jong_idx:
        tokens.append(_JONGSEONG_TO_PHONE_MFA[_JONGSEONG_COMPAT[jong_idx - 1]])
    return tokens


def _to_hiragana(char: str) -> list[str]:
    """Convert one source character with pykakasi, preserving per-char timing groups."""

    global _HIRAGANA_CONVERTER

    import pykakasi

    if _HIRAGANA_CONVERTER is None:
        _HIRAGANA_CONVERTER = pykakasi.kakasi()
    converter = _HIRAGANA_CONVERTER
    if hasattr(converter, "convert"):
        converted = converter.convert(char)
        return list("".join(item.get("hira", item.get("orig", "")) for item in converted))
    legacy = converter.getConverter()
    return list(legacy.do(char))


def _lookup_token(token: str, vocab: dict[str, int]) -> str | None:
    if token in vocab:
        return token
    lowered = token.lower()
    if lowered in vocab:
        return lowered
    # 960h처럼 대문자 vocab인 영어 모델용 — 자모·가나에는 upper()가 항등이라 무영향.
    uppered = token.upper()
    if uppered in vocab:
        return uppered
    return None


def _is_alignment_character(char: str) -> bool:
    if char.isspace():
        return False
    category = unicodedata.category(char)
    if category[0] in {"P", "S", "C", "Z"}:
        return False
    return category[0] in {"L", "M", "N"} or 0x4E00 <= ord(char) <= 0x9FFF


def _confidence(mean_log_score: float) -> float | None:
    if not math.isfinite(mean_log_score):
        return None
    return math.exp(min(0.0, mean_log_score))


def _interpolate_times(
    timings: list[tuple[float, float, float | None, int] | None], audio_length: float
) -> list[tuple[float, float, float | None, int]]:
    """Fill unmeasured lines between measured neighbours while preserving order."""

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
            result[end + 1][0]
            if end + 1 < len(result) and result[end + 1]
            else audio_length
        )
        slot = max(0.0, next_start - previous_end) / (end - start + 1)
        slot = max(slot, 0.1)
        for index in range(start, end + 1):
            offset = index - start
            result[index] = (previous_end + offset * slot, previous_end + (offset + 1) * slot, None, 0)
        i = end + 1
    return [item or (0.0, 0.0, None, 0) for item in result]


def _quality_score(lines: list[dict[str, Any]]) -> tuple[float | None, dict[str, Any]]:
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


class _torch_inference_mode:
    """Small lazy context wrapper so importing this module never imports torch."""

    def __enter__(self) -> Any:
        import torch

        self._context = torch.inference_mode()
        return self._context.__enter__()

    def __exit__(self, *exc: Any) -> Any:
        return self._context.__exit__(*exc)


def _log_softmax(logits: Any) -> Any:
    import torch

    # A half-precision model's logits get normalized in fp32 -- a bf16 log_softmax over a
    # multi-thousand-wide vocabulary loses resolution exactly where forced_align's DP needs it
    # (near-blank log-probabilities). ``.float()`` is a no-op for every existing fp32 candidate.
    return torch.log_softmax(logits.float(), dim=-1).contiguous()
