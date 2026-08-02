"""OmniASR-CTC 정렬 엔진 — ``facebook/omniASR-CTC-300M``.

벤치(``scripts/bench_adapters/omni_ctc.py``)에서 en 스택의 앵커이자 2패스 리파이너로
채택된 모델을 서버 계약(``BaseAlignmentEngine``)으로 다시 구현한다. 벤치 어댑터를 그대로
복붙하지 않는다 — 벤치는 ``AlignOut(lines=[{text,start,end,confidence,measured}])``을
돌려주지만 서버는 ``list[SyncResult]``(각 줄이 ``WordSegment`` 글자별 스팬을 담는다)를
쓴다. 아래는 그 차이를 메운 재구현이다.

인프로세스로 돈다 — ``OwsmEngine``(서브프로세스 격리)과 대비되는 지점이다. ``omniASR-CTC-300M``은
fairseq2 체크포인트(``{"model": state_dict}`` 피클 + SentencePiece 토크나이저)지만, 그
파라미터 레이아웃이 평범한 wav2vec2-large(7층 conv 피처 추출기, 24개 pre-norm 트랜스포머
블록, conv 위치 임베딩, 선형 CTC head)와 **정확히** 대응한다 — ``_convert_state_dict``가
그 이름만 ``transformers.Wav2Vec2ForCTC``로 바꿔 얹는다(근사 이식이 아니라 텐서 하나마다
파라미터 하나가 대응하는 정확 매핑, 학습 전용 ``masked_spec_embed``만 비어 있다). fairseq2
자체는 Windows 지원 빌드가 없어 이 포트가 필요하지만, 일단 이식되면 서버가 이미 상주시키는
``CTCEngine``과 같은 transformers/torchaudio 스택 위에서 돌기 때문에 별도 venv나
서브프로세스가 필요 없다.

vocab: 9,812 SentencePiece 조각, 이례적으로 **전부 단일 글자**(한자 5,750·한글 음절 1,220·
히라가나 80·가타카나 80 등). 다중 글자 서브워드가 없으므로 가사 글자 하나가 CTC 토큰
정확히 하나에 대응하고, 이 엔진이 내는 글자 스팬은 (OWSM의 서브워드 보간과 달리) 전부
**실측**이다. 라틴 커버리지는 소문자만이라 대문자는 소문자로 한 번 더 조회한다.

언어: 다국어(1,600+ 언어) 체크포인트라 언어 게이트가 없다 — ko/ja/en/zh 전부 이 한 모델로
정렬한다. ``language`` 인자는 시그니처 호환을 위해 받되 모델 선택에 쓰지 않는다.

라이선스: 가중치·코드 모두 Apache-2.0
(https://github.com/facebookresearch/omnilingual-asr/blob/main/LICENSE). 상업적 사용
가능 — 기존 CC-BY-NC-4.0 MMS 기준선과 다르다.

모델 조달: HF 캐시(``TRANSFORMERS_CACHE``/``HF_HOME``) 스냅샷 탐색만 한다
(``everyric2.alignment.hf_cache.find_cached_file``). 캐시에 없으면 조용히 네트워크로
받아오지 않고 ``EngineNotAvailableError``를 던진다 — 그 헬퍼 모듈의 docstring 참고.
"""

from __future__ import annotations

import math
import re
import time
import unicodedata
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import torch
import torchaudio.functional as F

from everyric2.alignment.base import (
    AlignmentError,
    BaseAlignmentEngine,
    EngineNotAvailableError,
    TranscriptionResult,
    WordTimestamp,
)
from everyric2.alignment.emission import EngineEmission
from everyric2.alignment.hf_cache import find_cached_file
from everyric2.audio.loader import AudioData
from everyric2.config.settings import AlignmentSettings
from everyric2.inference.prompt import LyricLine, SyncResult, WordSegment

MODEL_ID = "facebook/omniASR-CTC-300M"
CHECKPOINT_NAME = "omniASR-CTC-300M.pt"
TOKENIZER_NAME = "omniASR_tokenizer.model"

# CTC blank. ``<s>``가 SentencePiece 모델의 인덱스 0에 있고, 프레임 단위 argmax를 지배하는
# 토큰이 곧 blank라는 표시다(벤치 실측).
BLANK_ID = 0

TARGET_SAMPLE_RATE = 16_000
# omniASR 학습 말뭉치가 짧은 발화 위주라 공식 VRAM/RTF 수치가 30초 기준이다 — 그 학습
# 길이를 넘겨 위치 외삽을 시키지 않는다(벤치와 같은 값).
ALIGN_CHUNK_SEC = 30.0
ALIGN_CHUNK_OVERLAP_SEC = 5.0


def _read_sentencepiece_pieces(path: Path) -> list[str]:
    """SentencePiece ``ModelProto``의 piece 문자열을 인덱스 순서로 읽는다.

    ``sentencepiece`` 패키지 없이도 vocab을 읽어야 하므로(서버 .venv에 그 패키지가 없을 수
    있다) protobuf를 직접 훑는다. ``ModelProto`` 필드 1이 반복되는 ``SentencePiece``
    메시지고 그 필드 1이 piece 문자열이다 — 그 밖은 wire type으로 건너뛴다.
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
        field_no, wire = key >> 3, key & 7
        if field_no != 1 or wire != 2:
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
    """fairseq2 wav2vec2 파라미터 이름을 ``Wav2Vec2ForCTC`` 파라미터 이름으로 바꾼다."""

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


def _is_alignment_character(char: str) -> bool:
    """이 글자가 정렬 타깃(=CTC 토큰 하나)이 될 수 있는가. 공백·구두점·기호는 제외."""
    if char.isspace():
        return False
    category = unicodedata.category(char)
    if category[0] in {"P", "S", "C", "Z"}:
        return False
    return category[0] in {"L", "M", "N"} or 0x4E00 <= ord(char) <= 0x9FFF


def _lookup_char(char: str, vocab: dict[str, int]) -> str | None:
    """정렬 토큰으로 조회할 글자. vocab에 원형이 없으면 소문자로 한 번 더(라틴 대문자용)."""
    if char in vocab:
        return char
    lowered = char.lower()
    if lowered != char and lowered in vocab:
        return lowered
    return None


def _prepare_targets(
    lyrics: list[LyricLine], vocab: dict[str, int]
) -> tuple[list[int], list[list[tuple[int, int] | None]]]:
    """가사 줄들 -> (전곡 타깃 토큰열, 줄마다 글자별 [시작, 끝) 토큰 구간).

    전곡을 하나의 타깃열로 이어 붙여 강제정렬을 **한 번만** 돌린다(줄마다 다시 돌리지
    않는다) — 벤치의 ``HFCTCAligner.align``과 같은 구조다. 이 vocab은 다중 글자 서브워드가
    없으므로 글자 하나가 토큰 하나이고, 구간은 항상 길이 1이다.
    """
    token_ids: list[int] = []
    ranges: list[list[tuple[int, int] | None]] = []
    for line in lyrics:
        line_ranges: list[tuple[int, int] | None] = []
        for char in line.text:
            if not _is_alignment_character(char):
                line_ranges.append(None)
                continue
            token = _lookup_char(char, vocab)
            if token is None:
                line_ranges.append(None)
                continue
            first = len(token_ids)
            token_ids.append(vocab[token])
            line_ranges.append((first, first + 1))
        ranges.append(line_ranges)
    return token_ids, ranges


def _confidence(log_score: float) -> float:
    """CTC 평균 로그확률(음수) -> 0~1 신뢰도. emission이 log_softmax라 exp로 되돌린다."""
    if not math.isfinite(log_score):
        return 0.0
    return round(math.exp(min(0.0, log_score)), 6)


def _line_confidence(word_confidences: list[float | None]) -> float | None:
    """줄 전체 신뢰도 — 글자별(``WordSegment``) confidence의 기하평균.

    ``scripts/bench_adapters/hf_ctc.py::_confidence``가 라인 신뢰도를 내는 공식과
    수학적으로 같다(그쪽은 원시 로그점수 평균 뒤 한 번만 exp — ``exp(mean(log_p))`` —
    여기는 이미 exp를 거친 글자별 confidence의 기하평균 — ``exp(mean(log(exp(log_p))))``,
    같은 값이다). ``everyric2.server.worker._geomean``도 같은 공식을 쓴다 — 라인 conf가
    비어 있을 때(과거엔 이 함수가 없어 늘 그랬다) 그쪽이 폴백으로 다시 계산해 왔는데,
    라우팅(``worker._line_log_conf_median``)처럼 **정렬 직후** 라인 단위 신호가 필요한
    호출부는 그 폴백보다 먼저 실행돼 언제나 ``None``을 봤다 — 실측 버그(2026-08-03,
    코디네이터 실곡 검증). 엔진 자신이 채워야 모든 소비처가 일관되게 값을 본다.
    """
    values = [v for v in word_confidences if v is not None and v > 0]
    if not values:
        return None
    return round(math.exp(sum(math.log(v) for v in values) / len(values)), 6)


def _interpolate_line_times(
    times: list[tuple[float, float] | None], audio_length: float
) -> list[tuple[float, float]]:
    """정렬된 글자가 0개인 줄(전부 OOV)을 앞뒤 정렬 줄 사이 간격에 균등 배분(순서 보존)."""
    result = list(times)
    n = len(result)
    i = 0
    while i < n:
        if result[i] is not None:
            i += 1
            continue
        start = i
        end = i
        while end + 1 < n and result[end + 1] is None:
            end += 1
        prev_end = result[start - 1][1] if start > 0 and result[start - 1] else 0.0
        next_start = result[end + 1][0] if end + 1 < n and result[end + 1] else audio_length
        available = max(0.0, next_start - prev_end)
        count = end - start + 1
        seg = max(available / count if count else 0.0, 0.1)
        for j in range(start, end + 1):
            offset = j - start
            result[j] = (prev_end + offset * seg, prev_end + (offset + 1) * seg)
        i = end + 1
    return [t if t is not None else (0.0, 0.0) for t in result]


def _line_results(
    lyrics: list[LyricLine],
    ranges: list[list[tuple[int, int] | None]],
    token_spans: list[Any],
    ratio: float,
    audio_length: float,
) -> list[SyncResult]:
    """강제정렬 스팬 -> 줄별 ``SyncResult``(글자별 ``WordSegment`` 포함)."""
    line_times: list[tuple[float, float] | None] = []
    line_segments: list[list[WordSegment]] = []
    for line, line_ranges in zip(lyrics, ranges):
        starts: list[float] = []
        ends: list[float] = []
        segs: list[WordSegment] = []
        for char, token_range in zip(line.text, line_ranges):
            if token_range is None:
                continue
            index, _ = token_range
            if index >= len(token_spans):
                continue
            span = token_spans[index]
            start = float(span.start) * ratio
            end = float(span.end) * ratio
            starts.append(start)
            ends.append(end)
            segs.append(
                WordSegment(
                    word=char,
                    start=round(start, 3),
                    end=round(end, 3),
                    confidence=_confidence(float(span.score)),
                )
            )
        line_times.append((starts[0], ends[-1]) if starts else None)
        line_segments.append(segs)

    interpolated = _interpolate_line_times(line_times, audio_length)
    results: list[SyncResult] = []
    for line, (start, end), segs in zip(lyrics, interpolated, line_segments):
        results.append(
            SyncResult(
                line_number=line.line_number,
                text=line.text,
                start_time=start,
                end_time=end,
                word_segments=segs or None,
                confidence=_line_confidence([s.confidence for s in segs]) if segs else None,
            )
        )
    return results


class OmniASREngine(BaseAlignmentEngine):
    """omniASR-CTC-300M 강제 정렬 엔진 — 인프로세스, 다국어 단일 모델."""

    def __init__(self, config: AlignmentSettings | None = None) -> None:
        super().__init__(config)
        self._model: Any | None = None
        self._processor: Any | None = None
        self._vocab: dict[str, int] | None = None
        self._load_seconds: float | None = None
        self._last_word_timestamps: list[WordTimestamp] = []

    def is_available(self) -> bool:
        try:
            import torchaudio  # noqa: F401
            from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC  # noqa: F401
        except ImportError:
            return False
        checkpoint = find_cached_file(MODEL_ID, CHECKPOINT_NAME)
        tokenizer = find_cached_file(MODEL_ID, TOKENIZER_NAME)
        return checkpoint is not None and tokenizer is not None

    def _ensure_vocab(self) -> dict[str, int]:
        if self._vocab is not None:
            return self._vocab
        path = find_cached_file(MODEL_ID, TOKENIZER_NAME)
        if path is None:
            raise EngineNotAvailableError(
                f"{TOKENIZER_NAME} for {MODEL_ID} was not found under "
                "HF_HOME/TRANSFORMERS_CACHE; provision it before using the omniasr engine"
            )
        pieces = _read_sentencepiece_pieces(path)
        self._vocab = {piece: index for index, piece in enumerate(pieces)}
        return self._vocab

    def _ensure_model(self) -> tuple[Any, Any]:
        if self._model is not None:
            return self._processor, self._model

        checkpoint_path = find_cached_file(MODEL_ID, CHECKPOINT_NAME)
        if checkpoint_path is None:
            raise EngineNotAvailableError(
                f"{CHECKPOINT_NAME} for {MODEL_ID} was not found under "
                "HF_HOME/TRANSFORMERS_CACHE; provision it before using the omniasr engine"
            )

        from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

        started = time.perf_counter()
        vocab = self._ensure_vocab()
        processor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=TARGET_SAMPLE_RATE,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=False,
        )
        config = Wav2Vec2Config(
            vocab_size=len(vocab),
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
        raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(_convert_state_dict(raw["model"]), strict=False)
        # ``masked_spec_embed``는 학습 중 SpecAugment 전용이라 추론 체크포인트에 대응물이
        # 없다. 그 밖에 빠지거나 남는 파라미터가 있으면 포트가 어긋난 것이다.
        if unexpected or [key for key in missing if key != "wav2vec2.masked_spec_embed"]:
            raise AlignmentError(
                f"omniasr checkpoint state dict port mismatch: "
                f"missing={missing} unexpected={unexpected}"
            )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()

        self._processor = processor
        self._model = model
        self._load_seconds = round(time.perf_counter() - started, 2)
        return self._processor, self._model

    def _chunked_emission(
        self, waveform: torch.Tensor, processor: Any, model: Any, device: Any
    ) -> tuple[torch.Tensor, int]:
        """([1, T, V] log-softmax emission, 청크 수). 피크 VRAM은 청크 길이로 상한된다."""
        from everyric2.audio.chunking import plan_chunk_windows, stitch_chunk_outputs

        n = int(waveform.shape[0])
        windows = plan_chunk_windows(
            n,
            int(ALIGN_CHUNK_SEC * TARGET_SAMPLE_RATE),
            int(ALIGN_CHUNK_OVERLAP_SEC * TARGET_SAMPLE_RATE),
        )

        def _forward(chunk: torch.Tensor) -> torch.Tensor:
            inputs = processor(
                chunk.numpy(), sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt", padding=True
            )
            input_values = inputs.input_values.to(device=device)
            with torch.inference_mode():
                logits = model(input_values=input_values).logits
            return torch.log_softmax(logits.float(), dim=-1).contiguous()

        if len(windows) == 1:
            return _forward(waveform), 1

        pieces = [_forward(waveform[s:e].contiguous()).cpu() for s, e in windows]
        return stitch_chunk_outputs(pieces, windows, n, frame_axis=1), len(windows)

    def _prepare_waveform(self, audio: AudioData) -> torch.Tensor:
        from everyric2.audio.loader import AudioLoader

        loader = AudioLoader()
        prepared = loader.prepare_for_alignment(audio, target_sr=TARGET_SAMPLE_RATE, normalize=True)
        waveform = torch.from_numpy(prepared.waveform.astype("float32"))
        if waveform.dim() == 2:
            waveform = waveform.mean(dim=0)
        return waveform

    def transcribe(
        self,
        audio: AudioData,
        language: str | None = None,
    ) -> TranscriptionResult:
        raise NotImplementedError(
            "OmniASREngine does not support transcription. Use for forced alignment only."
        )

    def align(
        self,
        audio: AudioData,
        lyrics: list[LyricLine],
        language: str | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[SyncResult]:
        if not lyrics:
            raise AlignmentError("no lyric lines to align")

        processor, model = self._ensure_model()
        vocab = self._ensure_vocab()
        device = next(model.parameters()).device

        if progress_callback:
            progress_callback(1, 4)

        waveform = self._prepare_waveform(audio)
        token_ids, ranges = _prepare_targets(lyrics, vocab)
        if not token_ids:
            raise AlignmentError(f"{self.name}: no in-vocabulary lyric characters found")

        if progress_callback:
            progress_callback(2, 4)

        emission, _n_chunks = self._chunked_emission(waveform, processor, model, device)

        if progress_callback:
            progress_callback(3, 4)

        targets = torch.tensor([token_ids], dtype=torch.int32, device=emission.device)
        try:
            aligned_tokens, scores = F.forced_align(emission, targets, blank=BLANK_ID)
            token_spans = F.merge_tokens(aligned_tokens[0], scores[0], blank=BLANK_ID)
        except Exception as exc:
            raise AlignmentError(f"{self.name} forced alignment failed: {exc}") from exc

        audio_length = waveform.shape[0] / TARGET_SAMPLE_RATE
        ratio = audio_length / int(emission.shape[1])
        results = _line_results(lyrics, ranges, token_spans, ratio, audio_length)

        self._last_word_timestamps = [
            WordTimestamp(word=seg.word, start=seg.start, end=seg.end, confidence=seg.confidence)
            for result in results
            if result.word_segments
            for seg in result.word_segments
        ]

        if progress_callback:
            progress_callback(4, 4)

        return results

    def emission_for(self, audio: AudioData) -> EngineEmission:
        """곡 전체 emission 노출 — 2패스 리파이너가 라인 창만큼 프레임 축을 잘라 쓰는 입력.

        ``align``이 쓰는 경로(``_ensure_model`` -> ``_prepare_waveform`` ->
        ``_chunked_emission``)를 그대로 조합할 뿐이라 정렬 결과와 같은 emission이 나온다.
        모델 적재는 캐시되므로(``_ensure_model``) 첫 호출만 무겁다. 계약은
        ``everyric2.alignment.emission.EngineEmission`` 참고.
        """
        processor, model = self._ensure_model()
        vocab = self._ensure_vocab()
        device = next(model.parameters()).device

        waveform = self._prepare_waveform(audio)
        emission, n_chunks = self._chunked_emission(waveform, processor, model, device)
        audio_length = waveform.shape[0] / TARGET_SAMPLE_RATE
        ratio = audio_length / int(emission.shape[1])
        return EngineEmission(
            emission=emission,
            blank_id=BLANK_ID,
            frame_sec=ratio,
            audio_sec=audio_length,
            chunks=n_chunks,
            vocab=dict(vocab),
        )

    def get_last_transcription_data(
        self,
    ) -> tuple[list[WordTimestamp], None, str]:
        return (self._last_word_timestamps, None, "omniasr")

    def get_transcription_sets(self) -> list[tuple[list[WordTimestamp], None, str]]:
        data = self.get_last_transcription_data()
        if data[0]:
            return [data]
        return []

    @staticmethod
    def get_engine_type() -> Literal["omniasr"]:
        return "omniasr"
