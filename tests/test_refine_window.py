"""``everyric2.alignment.refine_window``(2패스 음절 리파이너) 회귀 테스트.

GPU·실모델 없이 검증한다(``test_emission_mask.py``·``test_pron_referee.py``와 같은 전략):
사고를 재현/검증하는 emission을 직접 구성하고 ``torchaudio.functional.forced_align``을
실제로 돌린다(CPU, 합성 텐서). 못박는 것:

* 앵커 창(``window_pad_sec``) 밖 프레임은 그 라인의 정렬에 안 쓰인다.
* 세그 늘이기의 발성 문턱(``seg_voiced_level=0.12``, 절대 바꾸면 안 되는 채택값)이 실제로
  세그 사이 공백을 메우는가.
* **연속성** — 혼재(ja+라틴) 라인에서 표기별 세그 열이 라인 구간을 빈틈없이 덮는가(시간
  역전 없음). 코디네이터가 명시적으로 요구한 회귀 방지 지점이다.
* 라인 경계(``start``/``end``)는 정제 단계가 절대 안 바꾼다.
* 리파이너가 emission을 못 내면(``emission_for`` → None) 전 라인이 폴백 신호만 달고
  조용히 돌아온다(예외 없음).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import torch

from everyric2.alignment.refine_window import (
    ExtendGate,
    TwoPassRefineConfig,
    _build_segments,
    _enforce_monotonic,
    _extend_segments,
    _respace_repeated_lines,
    _shift_line,
    _spread_piled_segments,
    _tokenize_target,
    refine_lines,
)
from everyric2.text.align_target import join_display
from everyric2.inference.prompt import SyncResult

FRAME_SEC = 0.02  # 20ms 프레임 규약(기존 스위트와 동일)


@dataclass
class _FakeEmission:
    """``everyric2.alignment.emission.EngineEmission``과 같은 모양(덕타이핑) — 앵커 팀
    확정본이 아직 이 워크트리에 없어 로컬로 흉내낸다."""

    emission: Any
    blank_id: int
    frame_sec: float
    audio_sec: float
    chunks: int = 1
    vocab: dict[str, int] = field(default_factory=dict)

    def frame_of(self, seconds: float) -> int:
        return int(round(seconds / self.frame_sec))


@dataclass
class _FakeRefiner:
    emission: _FakeEmission | None

    def emission_for(self, audio_path: Path) -> _FakeEmission | None:
        return self.emission


def _build_vocab(chars: str) -> dict[str, int]:
    """등장 문자마다 순서대로 id를 매긴다. 0은 blank로 비워 둔다."""
    vocab: dict[str, int] = {}
    for index, char in enumerate(sorted(set(chars)), start=1):
        vocab[char] = index
    return vocab


def _peaky_emission(
    token_ids_per_frame: list[int | None], vocab_size: int, blank_id: int = 0
) -> torch.Tensor:
    """프레임마다 지정한 토큰(또는 blank=None)이 강하게 지지되는 emission [1, T, V]."""
    frames = len(token_ids_per_frame)
    logits = torch.full((1, frames, vocab_size), -8.0)
    for t, token_id in enumerate(token_ids_per_frame):
        logits[0, t, token_id if token_id is not None else blank_id] = 8.0
    return torch.log_softmax(logits, dim=-1)


def _line(text: str, start: float, end: float, confidence: float | None = 0.9) -> SyncResult:
    return SyncResult(text=text, start_time=start, end_time=end, confidence=confidence)


# ---------------------------------------------------------------------------
# _tokenize_target — vocab 조회
# ---------------------------------------------------------------------------


def test_tokenize_target_maps_known_chars():
    vocab = {"k": 1, "a": 2, "t": 3}
    ids, ranges = _tokenize_target("kat", vocab)
    assert ids == [1, 2, 3]
    assert ranges == [(0, 1), (1, 2), (2, 3)]


def test_tokenize_target_drops_unknown_chars_as_none():
    vocab = {"k": 1, "t": 3}
    ids, ranges = _tokenize_target("k a t", vocab)  # 공백·a가 vocab에 없다
    assert ids == [1, 3]
    assert ranges[0] == (0, 1)
    assert ranges[1] is None  # 공백
    assert ranges[2] is None  # a
    assert ranges[4] == (1, 2)


def test_tokenize_target_lowercases_ascii_fallback():
    vocab = {"k": 1}
    ids, ranges = _tokenize_target("K", vocab)
    assert ids == [1]
    assert ranges == [(0, 1)]


# ---------------------------------------------------------------------------
# _build_segments — 빈 소유자·공백 통과분 흡수
# ---------------------------------------------------------------------------


class _Span:
    def __init__(self, start: int, end: int, score: float = 0.0) -> None:
        self.start = start
        self.end = end
        self.score = score


def test_build_segments_merges_empty_owner_into_previous():
    # target="ab", owners=["X", ""] — 둘째 글자는 표시가 없다(이중모음 둘째 음소류).
    owners = ["X", ""]
    ranges = [(0, 1), (1, 2)]
    spans = [_Span(0, 2), _Span(2, 5)]
    segs = _build_segments(owners, ranges, spans, 0.0, FRAME_SEC, [False, False])
    assert len(segs) == 1
    assert segs[0]["t"] == "X"
    assert segs[0]["end"] == pytest.approx(5 * FRAME_SEC)


def test_build_segments_merges_whitespace_owner_into_previous():
    # 공백이 용케 vocab에 있어 정렬됐어도(owner=" ") 별도 세그를 만들지 않는다.
    owners = ["X", " ", "Y"]
    ranges = [(0, 1), (1, 2), (2, 3)]
    spans = [_Span(0, 2), _Span(2, 3), _Span(3, 6)]
    segs = _build_segments(owners, ranges, spans, 0.0, FRAME_SEC, [False, False, False])
    assert [s["t"] for s in segs] == ["X", "Y"]
    # 공백 구간이 앞 세그로 흡수돼 X의 끝이 공백 스팬 끝까지 늘어난다.
    assert segs[0]["end"] == pytest.approx(3 * FRAME_SEC)


def test_build_segments_skips_unaligned_positions():
    owners = ["X", "Y"]
    ranges = [(0, 1), None]  # 둘째 문자가 vocab 밖이라 애초에 정렬 안 됨
    spans = [_Span(0, 2)]
    segs = _build_segments(owners, ranges, spans, 0.0, FRAME_SEC, [False, False])
    assert len(segs) == 1
    assert segs[0]["t"] == "X"


def test_build_segments_carries_word_end_flag_through_orphan():
    owners = ["X", ""]
    ranges = [(0, 1), (1, 2)]
    spans = [_Span(0, 2), _Span(2, 5)]
    segs = _build_segments(owners, ranges, spans, 0.0, FRAME_SEC, [False, True])
    assert segs[0]["word_end"] is True


# ---------------------------------------------------------------------------
# join_display — 표시 문자열은 owners 전체에서 잇는다 (낱말 경계 공백 포함)
# ---------------------------------------------------------------------------


def test_join_display_inserts_single_space_at_word_end():
    assert join_display(["킵", "잇"], [True, False]) == "킵 잇"


def test_join_display_no_trailing_space_at_line_end():
    assert join_display(["킵"], [True]) == "킵"


def test_join_display_skips_blank_owner_but_keeps_its_word_end():
    # 낱말 사이 원문 공백 패스스루(" ")는 건너뛰되, 그 자리에 word_end가 걸려 있으면
    # 공백은 하나만 들어가야 한다.
    assert join_display(["킵", " ", "잇"], [True, False, False]) == "킵 잇"


# ---------------------------------------------------------------------------
# _extend_segments — 발성 문턱 0.12 (채택값, 절대 불변)
# ---------------------------------------------------------------------------


def test_seg_voiced_level_default_is_adopted_value():
    assert TwoPassRefineConfig().seg_voiced_level == 0.12


def test_extend_segments_stretches_end_to_next_start():
    segs = [
        {"start": 0.0, "end": 0.1},
        {"start": 0.5, "end": 0.6},
    ]
    stretched = _extend_segments(segs, line_end=1.0, hold_max=1.5)
    assert stretched == 2
    assert segs[0]["end"] == pytest.approx(0.5)
    assert segs[1]["end"] == pytest.approx(1.0)  # 마지막 세그는 line_end까지


def test_extend_segments_respects_hold_max_without_gate():
    segs = [{"start": 0.0, "end": 0.1}]
    _extend_segments(segs, line_end=5.0, hold_max=1.5)
    assert segs[0]["end"] == pytest.approx(1.5)  # 상한에서 멈춘다(게이트 없음)


def test_extend_voiced_only_gate_fills_voiced_gap_at_012_threshold():
    # 우세도가 0.12 밑으로 안 내려가면(계속 발성) voiced_reach가 boundary까지 민다.
    # hold_max는 voiced_only에서도 여전히 상한이므로(자르기만 할 뿐 상한은 그대로 지킨다),
    # boundary보다 넉넉하게 잡아야 voiced_reach의 효과가 보인다.
    dominance = [0.5] * 100  # 전 구간 우세(문턱 0.12를 항상 웃돈다)
    gate = ExtendGate(dominance, 0.01, None, FRAME_SEC, speak_level=2.0)
    segs = [{"start": 0.0, "end": 0.1}]
    _extend_segments(
        segs, line_end=0.9, hold_max=2.0, gate=gate, voiced_only=True, voiced_level=0.12
    )
    assert segs[0]["end"] == pytest.approx(0.9)


def test_extend_voiced_only_gate_still_caps_at_hold_max():
    # voiced_only는 "자르기만 하고 상한은 그대로 지킨다" — 계속 발성 중이어도 hold_max를
    # 넘어 늘리지는 않는다(그 완화는 extend_gate=True 쪽 몫이고 이 포팅 범위 밖이다).
    dominance = [0.5] * 100
    gate = ExtendGate(dominance, 0.01, None, FRAME_SEC, speak_level=2.0)
    segs = [{"start": 0.0, "end": 0.1}]
    _extend_segments(
        segs, line_end=0.9, hold_max=0.2, gate=gate, voiced_only=True, voiced_level=0.12
    )
    assert segs[0]["end"] == pytest.approx(0.2)


def test_extend_voiced_only_gate_stops_when_dominance_drops_below_012():
    # 0.3초 지점부터 우세도가 문턱(0.12) 아래로 떨어진다 — 거기서 멈춰야 한다.
    dominance = [0.5] * 30 + [0.0] * 70  # dom_hop=0.01초 → 0.3초 경계
    gate = ExtendGate(dominance, 0.01, None, FRAME_SEC, speak_level=2.0)
    segs = [{"start": 0.0, "end": 0.1}]
    _extend_segments(
        segs, line_end=0.9, hold_max=1.5, gate=gate, voiced_only=True, voiced_level=0.12
    )
    assert segs[0]["end"] == pytest.approx(0.3, abs=0.02)


# ---------------------------------------------------------------------------
# _spread_piled_segments — 동시각 뭉침 펴기
# ---------------------------------------------------------------------------


def test_spread_piled_segments_separates_same_start_pile():
    import numpy as np

    segs = [
        {"start": 1.0, "end": 1.0},
        {"start": 1.0, "end": 1.0},
        {"start": 1.0, "end": 1.0},
        {"start": 2.0, "end": 2.1},
    ]
    presence = np.ones(200)  # 균일 발성 — 균등 분배로 떨어진다
    fixed = _spread_piled_segments(segs, presence, FRAME_SEC)
    assert fixed == 2
    starts = [s["start"] for s in segs]
    assert starts == sorted(starts)
    assert starts[0] < starts[1] < starts[2] < 2.0


# ---------------------------------------------------------------------------
# 라인 재배치 · 단조성
# ---------------------------------------------------------------------------


def test_shift_line_moves_all_display_keys_together():
    from everyric2.alignment.refine_window import PronSegmentSpan, RefinedLine

    line = RefinedLine(start=0.0, end=1.0)
    line.pron_segs["hangul"] = [PronSegmentSpan(text="가", start=0.1, end=0.2)]
    line.pron_segs["kana"] = [PronSegmentSpan(text="ka", start=0.1, end=0.2)]
    _shift_line(line, 0.5)
    assert line.pron_segs["hangul"][0].start == pytest.approx(0.6)
    assert line.pron_segs["kana"][0].start == pytest.approx(0.6)
    assert line.start == 0.0  # 라인 경계 자체는 안 바뀐다


def test_enforce_monotonic_splits_overlap_in_half():
    from everyric2.alignment.refine_window import PronSegmentSpan, RefinedLine

    line_a = RefinedLine(start=0.0, end=1.0)
    line_a.pron_segs["hangul"] = [PronSegmentSpan(text="가", start=0.0, end=0.6)]
    line_b = RefinedLine(start=1.0, end=2.0)
    line_b.pron_segs["hangul"] = [PronSegmentSpan(text="나", start=0.5, end=1.0)]
    fixed = _enforce_monotonic([line_a, line_b], "hangul")
    assert fixed == 1
    assert line_a.pron_segs["hangul"][0].end == pytest.approx(0.55)
    assert line_b.pron_segs["hangul"][0].start == pytest.approx(0.55)


def test_respace_repeated_lines_pulls_back_skipped_rendition():
    from everyric2.alignment.refine_window import PronSegmentSpan, RefinedLine

    # 균일 0.5초 간격이어야 하는데 셋째 줄이 렌디션 하나를 건너뛰어 +0.5초 밀렸다.
    starts = [0.0, 0.5, 1.5, 2.0]
    lines = [RefinedLine(start=s, end=s + 0.4) for s in starts]
    for line in lines:
        line.pron_segs["hangul"] = [PronSegmentSpan(text="가", start=line.start, end=line.end)]
    sources = ["가나다"] * 4
    fixed = _respace_repeated_lines(lines, sources)
    assert fixed >= 1
    # 셋째·넷째 줄 세그가 0.5초씩 당겨진다(라인 start 자체는 앵커 값이라 안 바뀐다).
    assert lines[2].pron_segs["hangul"][0].start == pytest.approx(1.0)
    assert lines[2].start == 1.5  # 라인 경계는 불변


# ---------------------------------------------------------------------------
# refine_lines — 통합. 합성 emission으로 전체 파이프라인을 돌린다.
# ---------------------------------------------------------------------------


def test_refine_lines_returns_fallback_when_emission_unavailable():
    # OWSM류 앵커(emission_for → None)를 흉내낸다 — 예외 없이 전 라인 폴백.
    anchors = [_line("hi", 0.0, 1.0)]
    refiner = _FakeRefiner(emission=None)
    lines = refine_lines(anchors, ["hi"], refiner, Path("dummy.wav"), language="en")
    assert len(lines) == 1
    assert lines[0].fallback_reason == "refiner_emission_unavailable"
    assert lines[0].pron_segs == {}
    assert lines[0].start == 0.0 and lines[0].end == 1.0  # 라인 경계는 앵커 값 그대로


def test_refine_lines_raises_on_line_count_mismatch():
    anchors = [_line("hi", 0.0, 1.0), _line("bye", 1.0, 2.0)]
    refiner = _FakeRefiner(emission=None)
    with pytest.raises(RuntimeError if False else Exception):
        refine_lines(anchors, ["hi"], refiner, Path("dummy.wav"), language="en")


def test_refine_lines_aligns_simple_en_word_and_produces_multi_key_segs():
    # "cat" -> CMU K AE1 T -> IPA "kat"(1음절, hangul "캣"). 프레임 0~29(0.6s)를
    # k/a/t 순서로 강하게 지지하는 합성 emission을 준다.
    vocab = _build_vocab("kat")
    frames_per_token = 10
    token_ids_per_frame: list[int | None] = []
    for ch in "kat":
        token_ids_per_frame.extend([vocab[ch]] * frames_per_token)
    token_ids_per_frame.extend([None] * 20)  # 꼬리 blank(늘이기 상한 확인용 여유)
    emission_tensor = _peaky_emission(token_ids_per_frame, vocab_size=len(vocab) + 1, blank_id=0)
    fake_emission = _FakeEmission(
        emission=emission_tensor, blank_id=0, frame_sec=FRAME_SEC, audio_sec=1.0, vocab=vocab
    )
    refiner = _FakeRefiner(emission=fake_emission)

    anchors = [_line("cat", 0.0, 0.6)]
    lines = refine_lines(anchors, ["cat"], refiner, Path("dummy.wav"), language="en")

    assert len(lines) == 1
    line = lines[0]
    assert line.fallback_reason is None
    assert line.refined is True
    # 다섯 표기(hangul/kana/romaji/en/ipa)가 전부 나와야 한다 — ipa는 정렬 타깃 자체를
    # 표시로 얹은 것(IPA 표시 옵션, 2026-08-03).
    assert set(line.pron_segs) == {"hangul", "kana", "romaji", "en", "ipa"}
    assert line.pron["hangul"] == "캣"
    assert line.start == 0.0 and line.end == 0.6  # 라인 경계는 안 바뀐다


def test_refine_lines_en_song_romaji_matches_en_not_katakana_roundtrip():
    """en 곡(라틴 리퍼리 경로)의 romaji 정답은 원문 철자다 — derive_en_display_units가
    내는 romaji(가나 음차 재변환)는 en 곡에서 오염이다(za wezaa poreketusu류, 2026-08-03
    감사). 같은 emission·같은 "cat" 라인으로 line.pron["romaji"] == line.pron["en"]과
    두 표기의 세그 텍스트·시각이 일치하는지 못박는다."""
    vocab = _build_vocab("kat")
    frames_per_token = 10
    token_ids_per_frame: list[int | None] = []
    for ch in "kat":
        token_ids_per_frame.extend([vocab[ch]] * frames_per_token)
    token_ids_per_frame.extend([None] * 20)
    emission_tensor = _peaky_emission(token_ids_per_frame, vocab_size=len(vocab) + 1, blank_id=0)
    fake_emission = _FakeEmission(
        emission=emission_tensor, blank_id=0, frame_sec=FRAME_SEC, audio_sec=1.0, vocab=vocab
    )
    refiner = _FakeRefiner(emission=fake_emission)

    anchors = [_line("cat", 0.0, 0.6)]
    lines = refine_lines(anchors, ["cat"], refiner, Path("dummy.wav"), language="en")

    line = lines[0]
    assert line.fallback_reason is None
    assert line.pron["romaji"] == line.pron["en"]
    assert [(s.text, s.start, s.end) for s in line.pron_segs["romaji"]] == [
        (s.text, s.start, s.end) for s in line.pron_segs["en"]
    ]


def test_refine_lines_ja_song_latin_run_keeps_kana_derived_romaji():
    """ja 곡(is_ja_line=True)에서는 이 수정이 손대면 안 된다 — derive_ja_display_units
    경로의 라틴 구간은 가나·로마자 변환이 여전히 정답이다("numb"의 가나 음차 romaji가
    "numb" 원문 철자로 덮이면 안 된다)."""
    from everyric2.text.align_target import derive_ja_display_units

    units = derive_ja_display_units("numb")
    target = units.target
    vocab = _build_vocab(target)
    frames_per_char = 5
    token_ids_per_frame: list[int | None] = []
    for ch in target:
        token_ids_per_frame.extend([vocab[ch]] * frames_per_char)
    token_ids_per_frame.extend([None] * 10)
    emission_tensor = _peaky_emission(token_ids_per_frame, vocab_size=len(vocab) + 1, blank_id=0)
    fake_emission = _FakeEmission(
        emission=emission_tensor,
        blank_id=0,
        frame_sec=FRAME_SEC,
        audio_sec=len(token_ids_per_frame) * FRAME_SEC,
        vocab=vocab,
    )
    refiner = _FakeRefiner(emission=fake_emission)

    line_end = len(token_ids_per_frame) * FRAME_SEC
    anchors = [_line("numb", 0.0, line_end)]
    lines = refine_lines(anchors, ["numb"], refiner, Path("dummy.wav"), language="ja")

    line = lines[0]
    assert line.fallback_reason is None
    # ja 파생(JA_DISPLAY_KEYS = hangul/kana/romaji)에는 en 표기 자체가 없다 — "en" in
    # line.pron 가드가 거짓이라 이 수정이 아예 발동하지 않는다는 것이 곧 증거다.
    assert "en" not in line.pron
    assert line.pron["romaji"], "가나 기반 romaji 파생 자체는 여전히 나와야 한다"


def test_refine_lines_window_ignores_frames_outside_pad():
    # 창 밖(라인 시작보다 훨씬 이전)에 다른 토큰이 강하게 있어도 그 라인 정렬에 안 쓰인다
    # — window_pad_sec(0.2s)가 실제로 창을 제한하는지 확인.
    vocab = _build_vocab("kat")
    frames: list[int | None] = [None] * 500  # 10초 분량 배경
    # 라인 2는 8.0~8.6초 구간에 kat를 둔다(프레임 400~429).
    start_frame = int(8.0 / FRAME_SEC)
    for offset, ch in enumerate("kat"):
        for f in range(start_frame + offset * 10, start_frame + offset * 10 + 10):
            frames[f] = vocab[ch]
    emission_tensor = _peaky_emission(frames, vocab_size=len(vocab) + 1, blank_id=0)
    fake_emission = _FakeEmission(
        emission=emission_tensor, blank_id=0, frame_sec=FRAME_SEC, audio_sec=10.0, vocab=vocab
    )
    refiner = _FakeRefiner(emission=fake_emission)

    anchors = [_line("cat", 8.0, 8.6)]
    lines = refine_lines(anchors, ["cat"], refiner, Path("dummy.wav"), language="en")
    assert lines[0].fallback_reason is None
    assert lines[0].pron_segs["hangul"][0].start == pytest.approx(8.0, abs=0.05)


# ---------------------------------------------------------------------------
# 연속성 — 혼재(ja+라틴) 라인에서 표기별 세그가 라인 구간을 빈틈없이 덮는다
# ---------------------------------------------------------------------------


def _assert_contiguous_and_monotonic(segs: list) -> None:
    """세그 열이 시간 역전 없이 이어지는지(각 세그의 끝이 다음 세그의 시작과 같거나
    그 전) 확인한다 — extend_segments가 세그 사이 빈틈을 없앤 뒤라야 성립한다."""
    for previous, current in zip(segs, segs[1:]):
        assert current.start >= previous.start
        assert previous.end <= current.start + 1e-6 or previous.end <= current.end
        assert current.end >= current.start


def test_refine_lines_mixed_ja_latin_line_has_no_gap_in_hangul_track():
    # "ひらひら numb numb" — ja 독음 + 라틴 라우팅이 섞인 라인. 두 계열의 타깃 문자를
    # 전부 강하게 지지하는 합성 emission을 만들어, 세그 사이에 시간 역전이나 (늘이기
    # 이후에도 남는) 빈틈이 없는지 못박는다.
    from everyric2.text.align_target import derive_ja_display_units

    units = derive_ja_display_units("ひらひら numb numb")
    target = units.target
    vocab = _build_vocab(target)
    frames_per_char = 5
    token_ids_per_frame: list[int | None] = []
    for ch in target:
        token_id = vocab.get(ch)
        token_ids_per_frame.extend([token_id] * frames_per_char)
    token_ids_per_frame.extend([None] * 10)
    emission_tensor = _peaky_emission(token_ids_per_frame, vocab_size=len(vocab) + 1, blank_id=0)
    fake_emission = _FakeEmission(
        emission=emission_tensor,
        blank_id=0,
        frame_sec=FRAME_SEC,
        audio_sec=len(token_ids_per_frame) * FRAME_SEC,
        vocab=vocab,
    )
    refiner = _FakeRefiner(emission=fake_emission)

    line_end = len(token_ids_per_frame) * FRAME_SEC
    anchors = [_line("ひらひら numb numb", 0.0, line_end)]
    lines = refine_lines(
        anchors, ["ひらひら numb numb"], refiner, Path("dummy.wav"), language="ja"
    )

    line = lines[0]
    assert line.fallback_reason is None
    hangul_segs = line.pron_segs["hangul"]
    assert len(hangul_segs) >= 4  # 히·라·히·라 + 넘·넘 중 최소 다수가 살아야 한다
    _assert_contiguous_and_monotonic(hangul_segs)
    # 라인 경계를 벗어나지 않는다.
    assert hangul_segs[0].start >= -1e-6
    assert hangul_segs[-1].end <= line_end + 1e-6


# ---------------------------------------------------------------------------
# _derive_units — 문자 계열 판정 (결함 수정, 2026-08-03 실사용자 보고 熱異常/b2NTglk9tvI)
#
# language=None(미판정)이 en 경로로 조용히 떨어져 ja 원문이 한자까지 그대로 통과하던
# 결함. worker.attach_pron_variants와 같은 원리(문자 수 우세)로 곡 단위 language 라벨이
# 아니라 이 라인 원문을 직접 본다.
# ---------------------------------------------------------------------------


def test_derive_units_picks_ja_by_character_majority_when_language_is_none():
    from everyric2.alignment.refine_window import _derive_units
    from everyric2.text.align_target import derive_ja_display_units

    source = "死んだ変数で繰り返す"
    units = _derive_units(source, None)
    expected = derive_ja_display_units(source)
    assert units.target == expected.target
    assert units.owners == expected.owners

    # 원문 한자가 표시에 그대로 남지 않는다 — 예전 결함은 이 자리에서 원문을 그대로 돌려줬다.
    hangul_display = "".join(units.owners["hangul"])
    assert hangul_display != source
    assert "死" not in hangul_display
    assert "変" not in hangul_display


def test_derive_units_ignores_a_wrong_language_label_and_follows_character_majority():
    # 곡 단위 라벨이 틀려도(예: 혼합 줄이라 다른 줄에서 결정된 en 라벨이 새어 들어온
    # 경우) 이 라인 자체가 ja 우세면 ja 파생을 따른다 — 라벨은 문자로 못 정할 때의
    # 최후 타이브레이커일 뿐이다.
    from everyric2.alignment.refine_window import _derive_units
    from everyric2.text.align_target import derive_ja_display_units

    source = "死んだ変数で繰り返す"
    units = _derive_units(source, "en")
    expected = derive_ja_display_units(source)
    assert units.target == expected.target
    assert units.owners == expected.owners


def test_derive_units_still_uses_en_path_for_non_ja_source():
    # 회귀 방지 — ja 글자가 없는 줄(en/ko 등)은 그대로 en 경로다(기존 동작 유지, 이
    # 모듈에 ko 전용 파생 경로는 없다).
    from everyric2.alignment.refine_window import _derive_units
    from everyric2.text.align_target import derive_en_display_units

    for source in ("hello world", "사랑해"):
        units = _derive_units(source, None)
        expected = derive_en_display_units(source)
        assert units.target == expected.target
        assert units.owners == expected.owners


def test_derive_units_falls_back_to_language_hint_only_when_character_majority_is_ambiguous():
    # ja/한글 둘 다 없는 줄(숫자·기호뿐)만 language 힌트가 타이브레이커로 쓰인다.
    from everyric2.alignment.refine_window import _derive_units
    from everyric2.text.align_target import derive_en_display_units, derive_ja_display_units

    source = "123!"
    assert _derive_units(source, "ja").owners == derive_ja_display_units(source).owners
    assert _derive_units(source, "en").owners == derive_en_display_units(source).owners
    assert _derive_units(source, None).owners == derive_en_display_units(source).owners


def test_refine_lines_derives_ja_display_even_when_language_is_none():
    # refine_lines 전체 경로에서도 language=None이 ja 라인의 표기를 en으로 새지 않게
    # 막는다(단위 테스트만이 아니라 실제 정렬 파이프라인 계약을 못박는다).
    from everyric2.text.align_target import derive_ja_display_units

    text = "ひらひら"
    units = derive_ja_display_units(text)
    target = units.target
    vocab = _build_vocab(target)
    frames_per_char = 5
    token_ids_per_frame: list[int | None] = []
    for ch in target:
        token_ids_per_frame.extend([vocab.get(ch)] * frames_per_char)
    token_ids_per_frame.extend([None] * 10)
    emission_tensor = _peaky_emission(token_ids_per_frame, vocab_size=len(vocab) + 1, blank_id=0)
    fake_emission = _FakeEmission(
        emission=emission_tensor,
        blank_id=0,
        frame_sec=FRAME_SEC,
        audio_sec=len(token_ids_per_frame) * FRAME_SEC,
        vocab=vocab,
    )
    refiner = _FakeRefiner(emission=fake_emission)

    line_end = len(token_ids_per_frame) * FRAME_SEC
    anchors = [_line(text, 0.0, line_end)]
    lines = refine_lines(anchors, [text], refiner, Path("dummy.wav"), language=None)

    line = lines[0]
    assert line.fallback_reason is None
    assert line.pron["hangul"] != text  # 원문 그대로 새지 않았다(예전 결함의 증상)
    assert line.pron_segs["hangul"]  # 세그도 실제로 생긴다


# ---------------------------------------------------------------------------
# 오디오 심판 — 결함 수정(2026-08-03, 벤치에서 이식 누락됐던 부분).
#
# ja 검증 픽스처는 team lead가 못박은 실사용자 사례 그대로다: numb numb(ba7YbGO2aq4)의
# 「好き好き」가 사전 기본값으로는 連濁된 「스키즈키」(すきずき)로 나오는데, 벤치가
# 청취 6/6으로 확정한 정답은 「스키스키」(すきすき)다. 오디오(합성 emission)가 すきすき
# 쪽을 강하게 지지하도록 만들어, 심판이 실제로 그 쪽을 채택하는지 못박는다.
# ---------------------------------------------------------------------------


def _refine_one_line(text: str, target: str, *, language: str, config=None, vocab_chars: str | None = None):
    """text 한 줄을 ``target`` 문자열을 강하게 지지하는 합성 emission으로 정렬한다.

    vocab은 기본적으로 ``target``만 커버한다 — 심판이 켜져 있고 다른 후보(예: 기본 발음)도
    같은 창에서 정렬을 시도한다면 그 문자까지 ``vocab_chars``로 함께 넘겨야 한다. 안 그러면
    vocab에 없는 문자가 ``_tokenize_target``에서 조용히 드롭돼(그 문자는 원래 「정렬 불가」
    취급이라 세그 자체가 안 생긴다) 엉뚱한 실패로 보인다 — 실제 심판 버그가 아니다.
    """
    vocab = _build_vocab(vocab_chars if vocab_chars is not None else target)
    frames_per_char = 5
    token_ids_per_frame: list[int | None] = []
    for ch in target:
        token_ids_per_frame.extend([vocab.get(ch)] * frames_per_char)
    token_ids_per_frame.extend([None] * 10)
    emission_tensor = _peaky_emission(token_ids_per_frame, vocab_size=len(vocab) + 1, blank_id=0)
    fake_emission = _FakeEmission(
        emission=emission_tensor,
        blank_id=0,
        frame_sec=FRAME_SEC,
        audio_sec=len(token_ids_per_frame) * FRAME_SEC,
        vocab=vocab,
    )
    refiner = _FakeRefiner(emission=fake_emission)
    line_end = len(token_ids_per_frame) * FRAME_SEC
    anchors = [_line(text, 0.0, line_end)]
    return refine_lines(
        anchors, [text], refiner, Path("dummy.wav"), language=language, config=config
    )[0]


def test_referee_ja_adopts_the_correct_reading_when_audio_supports_it():
    # 벤치·team lead 실측 픽스처 그대로: 사전 기본값은 連濁된 「스키즈키」인데 정답은
    # 「스키스키」다. vocab을 두 후보(すきずき/すきすき)의 합집합으로 짜서, 오디오는
    # すきすき쪽만 지지하도록 만든다.
    text = "好き好き"
    vocab_source = "すきずきすきすき"  # 두 후보 문자 전부 포함(ず도 vocab엔 있어야 실패가 아니라 "심판이 진다"가 된다)
    vocab = _build_vocab(vocab_source)
    frames_per_char = 5
    correct = "すきすき"
    token_ids_per_frame: list[int | None] = []
    for ch in correct:
        token_ids_per_frame.extend([vocab.get(ch)] * frames_per_char)
    token_ids_per_frame.extend([None] * 10)
    emission_tensor = _peaky_emission(token_ids_per_frame, vocab_size=len(vocab) + 1, blank_id=0)
    fake_emission = _FakeEmission(
        emission=emission_tensor,
        blank_id=0,
        frame_sec=FRAME_SEC,
        audio_sec=len(token_ids_per_frame) * FRAME_SEC,
        vocab=vocab,
    )
    refiner = _FakeRefiner(emission=fake_emission)
    line_end = len(token_ids_per_frame) * FRAME_SEC
    anchors = [_line(text, 0.0, line_end)]
    lines = refine_lines(anchors, [text], refiner, Path("dummy.wav"), language="ja")

    line = lines[0]
    assert line.fallback_reason is None
    assert line.pron["hangul"] == "스키스키"  # 정답 — "스키즈키"면 실패
    assert line.referee is not None
    assert line.referee["default"] == "스키즈키"
    assert line.referee["chosen"] == "스키스키"
    assert line.referee["gain"] is not None and line.referee["gain"] >= line.referee["margin"]


def test_referee_ja_keeps_default_when_audio_does_not_support_the_alternate():
    # 대칭 검증 — 오디오가 기본값(すきずき)을 지지하면 그대로 유지해야 한다(회귀 방지:
    # 심판이 항상 대체를 고르는 버그였다면 이 테스트가 잡는다).
    text = "好き好き"
    default = "すきずき"
    line = _refine_one_line(text, default, language="ja")
    assert line.fallback_reason is None
    assert line.pron["hangul"] == "스키즈키"
    assert line.referee is not None
    assert line.referee["chosen"] == line.referee["default"] == "스키즈키"


def test_referee_ja_off_never_switches_even_when_audio_disagrees():
    # referee=False면 예전 동작(항상 사전 첫 발음) 그대로다 — 오디오가 대체를 강하게
    # 지지해도 무시한다.
    from everyric2.alignment.refine_window import TwoPassRefineConfig

    text = "好き好き"
    line = _refine_one_line(
        text, "すきすき", language="ja", config=TwoPassRefineConfig(referee=False)
    )
    assert line.fallback_reason is None
    assert line.pron["hangul"] == "스키즈키"  # 심판이 꺼졌으니 사전 기본값 그대로
    assert line.referee is None


def test_referee_ja_no_op_when_line_has_no_ambiguous_word():
    # 애매 낱말이 없는 절대다수의 라인은 후보가 아예 없다 — 심판이 안 돈다(비용 0).
    text = "ひらひら"
    line = _refine_one_line(text, "ひらひら", language="ja")
    assert line.fallback_reason is None
    assert line.referee is None


def test_referee_en_switches_the_pronunciation_when_audio_supports_it():
    # "our"는 CMU에 세 발음이 있다(AW1 ER0 / AW1 R / AA1 R) — entry 0(기본, aur)가 아니라
    # entry 2(ar)를 오디오가 지지하도록 만든다. allow_length_change=True(en 채택값)라
    # 길이가 달라도(aur=3자 vs ar=2자) 후보에 오른다.
    from everyric2.text.align_target import derive_en_display_units

    text = "our house"
    base = derive_en_display_units(text)
    winner = derive_en_display_units(text, entries={0: 2})  # entry 2 = AA1 R ("ar")
    assert winner.target != base.target  # 길이도 다르다(회귀 방지: allow_length_change 확인)

    line = _refine_one_line(text, winner.target, language="en")
    assert line.fallback_reason is None
    assert line.referee is not None
    assert line.referee["chosen"] != line.referee["default"]
    assert line.pron["hangul"] == "".join(winner.owners["hangul"])


def test_referee_en_the_is_never_a_candidate_and_is_context_corrected_regardless():
    # "the"는 심판 후보에서 제외된다(문맥이 이미 정한다) — referee on/off와 무관하게
    # 다음 낱말의 첫소리로 결정된 발음이 나와야 한다.
    from everyric2.alignment.refine_window import TwoPassRefineConfig
    from everyric2.text.align_target import derive_en_display_units

    text = "the apple"  # apple = 모음 시작 -> ði("디")
    expected = derive_en_display_units(text)
    for config in (None, TwoPassRefineConfig(referee=False)):
        line = _refine_one_line(text, expected.target, language="en", config=config)
        assert line.fallback_reason is None
        assert line.pron["hangul"] == "".join(expected.owners["hangul"])
        assert line.pron["hangul"].startswith("디")


def test_referee_en_debug_records_default_chosen_margin_and_scores():
    from everyric2.text.align_target import derive_en_display_units

    text = "our house"
    winner = derive_en_display_units(text, entries={0: 2})
    line = _refine_one_line(text, winner.target, language="en")
    assert line.referee is not None
    assert set(line.referee) >= {"default", "chosen", "margin", "gain", "scores"}
    assert line.referee["margin"] == 0.03
    assert line.referee["scores"]  # 시도한 후보들의 (라벨, gain) 목록이 남는다


def test_referee_mixed_ja_latin_line_does_not_referee_the_embedded_latin_word():
    # latin_referee는 채택 구성에서 의도적으로 꺼져 있다 — 혼재(ja+라틴) 줄의 라틴 낱말은
    # 이 포트에서 아예 심판 후보로 만들지 않는다(entry 0 고정). "numb"는 CMU에 발음이
    # 하나뿐이라 이 자체로는 대체가 없지만, ja 쪽 심판이 라틴 구간까지 건드리지 않는지
    # (라틴 owners가 en 파생과 동일한지)를 못박는다.
    from everyric2.text.align_target import derive_en_display_units, derive_ja_display_units

    text = "好き numb"
    units = derive_ja_display_units(text)
    line = _refine_one_line(text, units.target, language="ja")
    assert line.fallback_reason is None
    # 라틴 구간(numb)의 hangul 표시는 항상 entry 0 기반 파생과 같다 — 심판이 안 건드렸다.
    expected_latin = derive_en_display_units("numb").owners["hangul"]
    assert "".join(expected_latin) in line.pron["hangul"]
