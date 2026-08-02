"""``OmniASREngine`` 서버 이식 검증 — 인프로세스 CTC 정렬기의 계약·가용성·emission 노출.

GPU도 실제 가중치도 없이 검증 가능하도록 모델 로드·추론은 전부 목(mock)한다(무거운 연산
금지 규약). 실제 forward pass·``forced_align``은 이 스위트 범위 밖이다 — 서버 계약
(``BaseAlignmentEngine`` 구현, ``EngineNotAvailableError`` 처리, ``emission_for`` 계약,
fairseq2 -> ``Wav2Vec2ForCTC`` 상태 딕셔너리 이식)만 못박는다.
"""

from __future__ import annotations

import math

import pytest
import torch

from everyric2.alignment import omniasr_engine
from everyric2.alignment.base import AlignmentError, EngineNotAvailableError
from everyric2.alignment.emission import EngineEmission
from everyric2.audio.loader import AudioData
from everyric2.inference.prompt import LyricLine


def _audio() -> AudioData:
    return AudioData(waveform=None, sample_rate=16000, duration=1.0)  # type: ignore[arg-type]


class _FakeParam:
    device = torch.device("cpu")


class _FakeModel:
    def parameters(self):
        return iter([_FakeParam()])


# ── 계약: get_engine_type / transcribe / is_available ─────────────────────────────


def test_get_engine_type():
    assert omniasr_engine.OmniASREngine.get_engine_type() == "omniasr"


def test_transcribe_not_implemented():
    engine = omniasr_engine.OmniASREngine()
    with pytest.raises(NotImplementedError):
        engine.transcribe(_audio())


def test_is_available_false_without_cached_weights(monkeypatch):
    monkeypatch.setattr(omniasr_engine, "find_cached_file", lambda *a, **k: None)
    assert omniasr_engine.OmniASREngine().is_available() is False


def test_is_available_true_with_cached_weights(tmp_path, monkeypatch):
    checkpoint = tmp_path / omniasr_engine.CHECKPOINT_NAME
    tokenizer = tmp_path / omniasr_engine.TOKENIZER_NAME
    checkpoint.write_bytes(b"")
    tokenizer.write_bytes(b"")

    def _fake_find(model_id: str, filename: str):
        return {
            omniasr_engine.CHECKPOINT_NAME: checkpoint,
            omniasr_engine.TOKENIZER_NAME: tokenizer,
        }[filename]

    monkeypatch.setattr(omniasr_engine, "find_cached_file", _fake_find)
    assert omniasr_engine.OmniASREngine().is_available() is True


def test_is_available_false_with_partial_cache(tmp_path, monkeypatch):
    # 체크포인트만 있고 토크나이저가 없는 경우도 미가용 — 둘 다 있어야 한다.
    checkpoint = tmp_path / omniasr_engine.CHECKPOINT_NAME
    checkpoint.write_bytes(b"")

    def _fake_find(model_id: str, filename: str):
        return checkpoint if filename == omniasr_engine.CHECKPOINT_NAME else None

    monkeypatch.setattr(omniasr_engine, "find_cached_file", _fake_find)
    assert omniasr_engine.OmniASREngine().is_available() is False


# ── 조용한 폴백 금지: 캐시에 없으면 EngineNotAvailableError ───────────────────────────


def test_ensure_vocab_raises_engine_not_available_without_tokenizer(monkeypatch):
    monkeypatch.setattr(omniasr_engine, "find_cached_file", lambda *a, **k: None)
    engine = omniasr_engine.OmniASREngine()
    with pytest.raises(EngineNotAvailableError):
        engine._ensure_vocab()


def test_ensure_model_raises_engine_not_available_without_checkpoint(monkeypatch, tmp_path):
    tokenizer = tmp_path / omniasr_engine.TOKENIZER_NAME
    tokenizer.write_bytes(b"")

    def _fake_find(model_id: str, filename: str):
        return tokenizer if filename == omniasr_engine.TOKENIZER_NAME else None

    monkeypatch.setattr(omniasr_engine, "find_cached_file", _fake_find)
    monkeypatch.setattr(
        omniasr_engine, "_read_sentencepiece_pieces", lambda path: ["<s>", "a", "b"]
    )
    engine = omniasr_engine.OmniASREngine()
    with pytest.raises(EngineNotAvailableError):
        engine._ensure_model()


def test_align_raises_engine_not_available_without_weights(monkeypatch):
    monkeypatch.setattr(omniasr_engine, "find_cached_file", lambda *a, **k: None)
    engine = omniasr_engine.OmniASREngine()
    with pytest.raises(EngineNotAvailableError):
        engine.align(_audio(), [LyricLine(text="hello", line_number=1)])


def test_align_rejects_empty_lyrics():
    engine = omniasr_engine.OmniASREngine()
    with pytest.raises(AlignmentError):
        engine.align(_audio(), [])


# ── fairseq2 -> Wav2Vec2ForCTC 상태 딕셔너리 이식 (순수 함수, 실제 가중치 불필요) ──────


def test_convert_state_dict_maps_known_patterns():
    raw = {
        "encoder_frontend.feature_extractor.layers.0.conv.weight": "A",
        "encoder_frontend.post_extract_layer_norm.weight": "B",
        "encoder_frontend.model_dim_proj.weight": "C",
        "encoder_frontend.pos_encoder.conv.bias": "D",
        "encoder_frontend.pos_encoder.conv.weight_g": "E",
        "encoder_frontend.pos_encoder.conv.weight_v": "F",
        "encoder.layers.0.self_attn_layer_norm.weight": "G",
        "encoder.layers.0.self_attn.output_proj.weight": "H",
        "encoder.layers.0.self_attn.k_proj.weight": "I",
        "encoder.layers.0.ffn_layer_norm.weight": "J",
        "encoder.layers.0.ffn.inner_proj.weight": "K",
        "encoder.layers.0.ffn.output_proj.weight": "L",
        "encoder.layer_norm.weight": "M",
        "final_proj.weight": "N",
    }
    converted = omniasr_engine._convert_state_dict(raw)

    assert converted["wav2vec2.feature_extractor.conv_layers.0.conv.weight"] == "A"
    assert converted["wav2vec2.feature_projection.layer_norm.weight"] == "B"
    assert converted["wav2vec2.feature_projection.projection.weight"] == "C"
    assert converted["wav2vec2.encoder.pos_conv_embed.conv.bias"] == "D"
    assert (
        converted["wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original0"]
        == "E"
    )
    assert (
        converted["wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original1"]
        == "F"
    )
    assert converted["wav2vec2.encoder.layers.0.layer_norm.weight"] == "G"
    assert converted["wav2vec2.encoder.layers.0.attention.out_proj.weight"] == "H"
    assert converted["wav2vec2.encoder.layers.0.attention.k_proj.weight"] == "I"
    assert converted["wav2vec2.encoder.layers.0.final_layer_norm.weight"] == "J"
    assert converted["wav2vec2.encoder.layers.0.feed_forward.intermediate_dense.weight"] == "K"
    assert converted["wav2vec2.encoder.layers.0.feed_forward.output_dense.weight"] == "L"
    assert converted["wav2vec2.encoder.layer_norm.weight"] == "M"
    assert converted["lm_head.weight"] == "N"


def test_convert_state_dict_raises_on_unmapped_key():
    with pytest.raises(KeyError):
        omniasr_engine._convert_state_dict({"totally.unknown.key": "x"})


# ── 글자 -> 토큰 타깃 준비 (순수 함수) ────────────────────────────────────────────


def test_lookup_char_prefers_exact_then_lowercase():
    vocab = {"a": 0, "가": 1}
    assert omniasr_engine._lookup_char("a", vocab) == "a"
    assert omniasr_engine._lookup_char("A", vocab) == "a"
    assert omniasr_engine._lookup_char("가", vocab) == "가"
    assert omniasr_engine._lookup_char("z", vocab) is None


def test_prepare_targets_skips_punctuation_and_oov():
    vocab = {"가": 0, "나": 1}
    lyrics = [LyricLine(text="가, 나!", line_number=1)]
    token_ids, ranges = omniasr_engine._prepare_targets(lyrics, vocab)

    assert token_ids == [0, 1]
    assert ranges[0] == [(0, 1), None, None, (1, 2), None]


def test_confidence_converts_log_score():
    assert omniasr_engine._confidence(0.0) == 1.0
    assert omniasr_engine._confidence(float("-inf")) == 0.0
    assert omniasr_engine._confidence(-1.0) == pytest.approx(math.exp(-1.0), abs=1e-6)


# ── 강제정렬 스팬 -> SyncResult/WordSegment (순수 함수, 합성 스팬으로 검증) ────────────


class _FakeSpan:
    def __init__(self, start: int, end: int, score: float) -> None:
        self.start = start
        self.end = end
        self.score = score


def test_line_results_builds_word_segments_and_interpolates():
    lyrics = [
        LyricLine(text="가나", line_number=1),
        LyricLine(text="", line_number=2),  # 아무 것도 안 매칭 -> 보간 대상
        LyricLine(text="다", line_number=3),
    ]
    ranges = [[(0, 1), (1, 2)], [], [(2, 3)]]
    token_spans = [
        _FakeSpan(0, 5, -0.1),
        _FakeSpan(5, 10, -0.2),
        _FakeSpan(20, 25, -0.05),
    ]

    results = omniasr_engine._line_results(lyrics, ranges, token_spans, ratio=0.02, audio_length=1.0)

    assert len(results) == 3
    assert [w.word for w in results[0].word_segments] == ["가", "나"]
    assert results[0].word_segments[0].confidence == omniasr_engine._confidence(-0.1)
    assert results[0].start_time == pytest.approx(0.0)
    assert results[0].end_time == pytest.approx(0.2)
    assert results[2].start_time == pytest.approx(0.4)
    # 보간된 중간 줄이 순서를 지킨다 (역전·겹침 없음)
    assert results[0].end_time <= results[1].start_time <= results[1].end_time <= results[2].start_time


# ── emission 노출 계약 (2패스 리파이너용) ─────────────────────────────────────────


def test_emission_for_returns_engine_emission_contract(monkeypatch):
    engine = omniasr_engine.OmniASREngine()
    fake_emission = torch.zeros(1, 10, 5)

    monkeypatch.setattr(engine, "_ensure_model", lambda: (object(), _FakeModel()))
    monkeypatch.setattr(engine, "_ensure_vocab", lambda: {"a": 0, "b": 1})
    monkeypatch.setattr(engine, "_prepare_waveform", lambda audio: torch.zeros(16000))
    monkeypatch.setattr(
        engine,
        "_chunked_emission",
        lambda waveform, processor, model, device: (fake_emission, 1),
    )

    result = engine.emission_for(_audio())

    assert isinstance(result, EngineEmission)
    assert result.emission is fake_emission
    assert result.blank_id == omniasr_engine.BLANK_ID
    assert result.chunks == 1
    assert result.vocab == {"a": 0, "b": 1}
    assert result.audio_sec == pytest.approx(1.0)
    assert result.frame_sec == pytest.approx(0.1)  # 1.0초 / 10프레임
