"""Meta Omnilingual ASR CTC adapter for the alignment benchmark.

``facebook/omniASR-CTC-300M`` ships as a fairseq2 checkpoint (a flat ``{"model": state_dict}``
pickle plus a SentencePiece tokenizer), not as a Transformers repository.  fairseq2 itself has
no supported Windows build, so this module does not depend on it: the checkpoint's parameter
layout is an ordinary wav2vec2-large (7-layer conv feature extractor, 24 pre-norm transformer
blocks, convolutional position embedding, one linear CTC head), and ``_convert_state_dict``
renames it onto ``transformers.Wav2Vec2ForCTC``.  The port is exact rather than approximate:
every checkpoint tensor maps onto exactly one Transformers parameter, no Transformers
parameter is left unfilled except the training-only ``masked_spec_embed``, and the resulting
parameter count is 325,494,996 -- the figure printed on the model card.

Adapter behaviour therefore rides on ``hf_ctc.HFCTCAligner`` unchanged: same 30s/5s overlapping
chunked emission, same ``forced_align``/``merge_tokens`` span extraction, same per-source-
character ``lines[].segs``.

Vocabulary: 9,812 SentencePiece pieces, and -- unusually for a SentencePiece model -- every
one of them is a single character (5,750 Han, 1,220 Hangul syllables, 80 hiragana, 80
katakana, plus Latin/Cyrillic/Arabic/etc.).  There are no multi-character subwords and no
``U+2581``-prefixed word pieces, so a lyric character maps to exactly one CTC token and the
character spans this adapter reports are measured, not interpolated.  Latin coverage is
lowercase-only; ``hf_ctc._lookup_token`` already retries ``str.lower()``.

Language support: multilingual (1,600+ languages), so no language gate -- every benchmark
stratum (ko/ja/en/zh) is alignable.  The CTC checkpoints take no language-conditioning input.

License: Apache-2.0 for both the model weights and the omnilingual-asr code
(https://github.com/facebookresearch/omnilingual-asr/blob/main/LICENSE); the training corpus
``facebook/omnilingual-asr-corpus`` is CC-BY-4.0.  Commercial use is permitted -- unlike the
CC-BY-NC-4.0 MMS baseline this benchmark is trying to replace.
"""

from __future__ import annotations

import dataclasses
import re
import time
from pathlib import Path
from typing import Any

from scripts.bench_adapters.hf_ctc import (
    HFCTCAligner,
    HFCTCModelConfig,
    _find_cached_file,
)
from scripts.benchmark_alignment import AlignerAdapter

MODEL_ID = "facebook/omniASR-CTC-300M"
CHECKPOINT_NAME = "omniASR-CTC-300M.pt"
TOKENIZER_NAME = "omniASR_tokenizer.model"

# The CTC blank. ``<s>`` sits at index 0 of the SentencePiece model and is the token that
# dominates the frame-level argmax, which is the observable signature of the blank.
BLANK_ID = 0

OMNI_ASR_CTC_300M = HFCTCModelConfig(
    name="omniasr-ctc",
    model_id=MODEL_ID,
    language="multilingual",
    vocab_unit="character",
    multilingual=True,
    # The omnilingual-asr corpus is short-utterance speech and the official VRAM/RTF figures
    # are quoted at 30s; keeping the window at training length avoids asking a model for
    # positional extrapolation it was never trained to do.
    align_chunk_sec=30.0,
    vocab_note=(
        "9,812 SentencePiece pieces, all single characters (5,750 Han, 1,220 Hangul "
        "syllables, 80 hiragana, 80 katakana, Latin lowercase, plus other scripts); "
        "no multi-character subwords."
    ),
)


def _read_sentencepiece_pieces(path: Path) -> list[str]:
    """Return the piece strings of a SentencePiece ``ModelProto`` in index order.

    The benchmark interpreter has no ``sentencepiece`` package and this adapter needs only the
    piece table, so the protobuf is walked directly.  ``ModelProto`` field 1 is the repeated
    ``SentencePiece`` message and its field 1 is the piece string; every other field is
    skipped by wire type.
    """

    def read_varint(buf: bytes, i: int) -> tuple[int, int]:
        shift = 0
        value = 0
        while True:
            byte = buf[i]
            i += 1
            value |= (byte & 0x7F) << shift
            if not byte & 0x80:
                return value, i
            shift += 7

    def skip(buf: bytes, i: int, wire: int) -> int:
        if wire == 0:
            _, i = read_varint(buf, i)
            return i
        if wire == 1:
            return i + 8
        if wire == 5:
            return i + 4
        if wire == 2:
            length, i = read_varint(buf, i)
            return i + length
        raise ValueError(f"unsupported protobuf wire type {wire} in {path}")

    buf = path.read_bytes()
    pieces: list[str] = []
    i = 0
    while i < len(buf):
        key, i = read_varint(buf, i)
        field, wire = key >> 3, key & 7
        if field != 1 or wire != 2:
            i = skip(buf, i, wire)
            continue
        length, i = read_varint(buf, i)
        payload = buf[i : i + length]
        i += length
        j = 0
        piece: str | None = None
        while j < len(payload):
            inner_key, j = read_varint(payload, j)
            inner_field, inner_wire = inner_key >> 3, inner_key & 7
            if inner_field == 1 and inner_wire == 2:
                inner_length, j = read_varint(payload, j)
                piece = payload[j : j + inner_length].decode("utf-8", "replace")
                j += inner_length
                continue
            j = skip(payload, j, inner_wire)
        if piece is None:
            raise ValueError(f"SentencePiece entry without a piece string in {path}")
        pieces.append(piece)
    return pieces


def _convert_state_dict(raw: dict[str, Any]) -> dict[str, Any]:
    """Rename fairseq2 wav2vec2 parameters onto ``Wav2Vec2ForCTC`` parameter names."""

    converted: dict[str, Any] = {}
    for key, tensor in raw.items():
        match = re.match(r"encoder_frontend\.feature_extractor\.layers\.(\d+)\.(.*)", key)
        if match:
            converted[
                f"wav2vec2.feature_extractor.conv_layers.{match.group(1)}.{match.group(2)}"
            ] = tensor
            continue
        if key.startswith("encoder_frontend.post_extract_layer_norm."):
            converted["wav2vec2.feature_projection.layer_norm." + key.rsplit(".", 1)[1]] = tensor
            continue
        if key.startswith("encoder_frontend.model_dim_proj."):
            converted["wav2vec2.feature_projection.projection." + key.rsplit(".", 1)[1]] = tensor
            continue
        if key.startswith("encoder_frontend.pos_encoder.conv."):
            # Transformers registers the positional conv with torch's weight-norm
            # parametrization, so the fairseq2 g/v pair lands on original0/original1.
            suffix = {
                "bias": "bias",
                "weight_g": "parametrizations.weight.original0",
                "weight_v": "parametrizations.weight.original1",
            }[key.rsplit(".", 1)[1]]
            converted["wav2vec2.encoder.pos_conv_embed.conv." + suffix] = tensor
            continue
        match = re.match(r"encoder\.layers\.(\d+)\.(.*)", key)
        if match:
            rest = (
                match.group(2)
                .replace("self_attn_layer_norm.", "layer_norm.")
                .replace("self_attn.output_proj.", "attention.out_proj.")
                .replace("self_attn.", "attention.")
                .replace("ffn_layer_norm.", "final_layer_norm.")
                .replace("ffn.inner_proj.", "feed_forward.intermediate_dense.")
                .replace("ffn.output_proj.", "feed_forward.output_dense.")
            )
            converted[f"wav2vec2.encoder.layers.{match.group(1)}.{rest}"] = tensor
            continue
        if key.startswith("encoder.layer_norm."):
            converted["wav2vec2.encoder.layer_norm." + key.rsplit(".", 1)[1]] = tensor
            continue
        if key.startswith("final_proj."):
            converted["lm_head." + key.rsplit(".", 1)[1]] = tensor
            continue
        raise KeyError(f"unmapped omniASR checkpoint parameter: {key}")
    return converted


class OmniASRCTCAligner(HFCTCAligner):
    """omniASR-CTC-300M forced aligner, loaded through an in-process fairseq2->HF port."""

    name = OMNI_ASR_CTC_300M.name
    model_id = OMNI_ASR_CTC_300M.model_id
    model_config = OMNI_ASR_CTC_300M
    # VRAM knob for half-precision variants (see OmniASRCTCBf16Aligner below). "float32"
    # reproduces this class's existing cached results bit-for-bit.
    dtype: str = "float32"

    def _resolve(self, filename: str) -> Path:
        path = _find_cached_file(self.model_id, filename)
        if path is not None:
            return path
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(self.model_id, filename))

    def _ensure_vocab(self) -> dict[str, int]:
        if self._vocab is not None:
            return self._vocab
        path = self._resolve(TOKENIZER_NAME)
        pieces = _read_sentencepiece_pieces(path)
        # Index order is the CTC output order: the head is 9,812 wide, exactly the piece count.
        self._vocab = {piece: index for index, piece in enumerate(pieces)}
        self._vocab_path = path
        return self._vocab

    def _ensure_processor(self) -> Any:
        if self._processor is None:
            from transformers import Wav2Vec2FeatureExtractor

            # Mirrors the checkpoint's documented preprocessing: raw 16 kHz waveform,
            # zero-mean/unit-variance normalized, no attention mask.
            self._processor = Wav2Vec2FeatureExtractor(
                feature_size=1,
                sampling_rate=16_000,
                padding_value=0.0,
                do_normalize=True,
                return_attention_mask=False,
            )
        return self._processor

    def _ensure_model(self) -> tuple[Any, Any]:
        if self._model is not None:
            return self._ensure_processor(), self._model

        import torch
        from transformers import Wav2Vec2Config, Wav2Vec2ForCTC

        started = time.perf_counter()
        processor = self._ensure_processor()
        vocab_size = len(self._ensure_vocab())
        config = Wav2Vec2Config(
            vocab_size=vocab_size,
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=4096,
            hidden_act="gelu",
            feat_extract_norm="layer",
            feat_extract_activation="gelu",
            conv_dim=(512,) * 7,
            conv_stride=(5, 2, 2, 2, 2, 2, 2),
            conv_kernel=(10, 3, 3, 3, 3, 2, 2),
            conv_bias=True,
            num_conv_pos_embeddings=128,
            num_conv_pos_embedding_groups=16,
            do_stable_layer_norm=True,
            apply_spec_augment=False,
            layer_norm_eps=1e-5,
            pad_token_id=BLANK_ID,
        )
        model = Wav2Vec2ForCTC(config)
        raw = torch.load(self._resolve(CHECKPOINT_NAME), map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(_convert_state_dict(raw["model"]), strict=False)
        # ``masked_spec_embed`` only participates in SpecAugment during training and has no
        # counterpart in an inference checkpoint. Anything else missing means the port drifted.
        if unexpected or [key for key in missing if key != "wav2vec2.masked_spec_embed"]:
            raise RuntimeError(
                f"{self.name} state dict port mismatch: missing={missing} unexpected={unexpected}"
            )
        # Cast on CPU before the GPU move (same order owsm_ctc's bf16 variant uses): the state
        # dict above is fp32 on CPU, so casting here means the GPU only ever sees one copy of
        # the weights at the target dtype instead of a transient fp32 copy plus the cast one.
        torch_dtype = getattr(torch, self.dtype)
        if torch_dtype is not torch.float32:
            model = model.to(dtype=torch_dtype)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        if device == "cuda":
            torch.cuda.empty_cache()
        self._model = model
        self._load_seconds = round(time.perf_counter() - started, 2)
        return processor, self._model


# A distinct HFCTCModelConfig per variant, with only ``name`` changed: HFCTCAligner.__init__
# sets ``self.name`` from ``config.name`` (not from the class attribute), so a variant that
# reused OMNI_ASR_CTC_300M outright would report itself as "omniasr-ctc" and collide with the
# fp32 candidate's cached runs.
OMNI_ASR_CTC_300M_BF16 = dataclasses.replace(OMNI_ASR_CTC_300M, name="omniasr-ctc-bf16")
OMNI_ASR_CTC_300M_FP16 = dataclasses.replace(OMNI_ASR_CTC_300M, name="omniasr-ctc-fp16")


class OmniASRCTCBf16Aligner(OmniASRCTCAligner):
    """VRAM-reduction variant: bfloat16 weights, cast on CPU before the GPU move.

    Registered under its own name so the harness caches its sweep results separately from the
    fp32 ``omniasr-ctc`` evidence rather than overwriting it.
    """

    name = "omniasr-ctc-bf16"
    model_config = OMNI_ASR_CTC_300M_BF16
    dtype = "bfloat16"


class OmniASRCTCFp16Aligner(OmniASRCTCAligner):
    """VRAM-reduction variant: float16 weights.

    Unlike owsm-ctc-v4-1b (whose E-Branchformer encoder overflows fp16 even with an fp32-cast
    log_softmax), this wav2vec2-style CNN+attention encoder measured clean on both verification
    songs -- quality_score within 0.0002 of fp32 and identical VRAM to the bf16 variant. Kept
    alongside bf16 rather than replacing it since only two songs were checked.
    """

    name = "omniasr-ctc-fp16"
    model_config = OMNI_ASR_CTC_300M_FP16
    dtype = "float16"


def register(aligner_registry: dict[str, type[AlignerAdapter]]) -> None:
    """Register the omniASR CTC candidate and its VRAM-reduction variants in a harness registry."""

    aligner_registry[OmniASRCTCAligner.name] = OmniASRCTCAligner
    aligner_registry[OmniASRCTCBf16Aligner.name] = OmniASRCTCBf16Aligner
    aligner_registry[OmniASRCTCFp16Aligner.name] = OmniASRCTCFp16Aligner
