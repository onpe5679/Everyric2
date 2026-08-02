"""노트 양자화(옥타브 폴딩·경계 결정) 회귀 고정.

여기 있는 케이스는 전부 scripts/bench_melody_ust.py(UST 12곡 준정답 벤치)에서
실제로 점수를 깎던 실패를 최소 재현으로 굳힌 것이다. 벤치는 오디오가 필요 없지만
UST 파일을 읽어야 하므로 CI에서 못 돌린다 — 그 결론만 여기로 옮겨 놓는다.
"""

import numpy as np
import pytest

from everyric2.melody.extractor import (
    F0Track,
    MelodyExtractor,
    _agreed_octave_shift,
    _fold_notes_to_local_median,
    fold_line_octaves,
    notes_from_anchor_spans,
    octave_fold_shift,
)

FRAME_DT = 0.01


def make_track(spec: list[tuple[float, float | None]], dt: float = FRAME_DT) -> F0Track:
    """(길이초, midi|None=무성음) 목록으로 합성 F0Track 생성."""
    frames: list[float] = []
    for dur, midi in spec:
        frames.extend([np.nan if midi is None else midi] * int(round(dur / dt)))
    arr = np.array(frames, dtype=np.float64)
    return F0Track(times=(np.arange(len(arr)) + 0.5) * dt, midi=arr, voiced=~np.isnan(arr))


class TestOctaveFoldShift:
    """옥타브 인공물만 접고 진짜 도약은 남긴다 — 잔차 게이팅."""

    @pytest.mark.parametrize("dev,expected", [(-12.0, 12.0), (12.0, -12.0), (-24.0, 24.0)])
    def test_exact_octave_deviation_folds(self, dev, expected):
        assert octave_fold_shift(np.array([dev]))[0] == expected

    @pytest.mark.parametrize("dev", [-16.0, 16.0, -19.0, 9.0, -10.0])
    def test_non_octave_interval_preserved(self, dev):
        # 16·19반음은 옥타브의 배수가 아니다 — 음역이 넓은 라인의 진짜 도약이라 보존
        assert octave_fold_shift(np.array([dev]))[0] == 0.0

    def test_small_deviation_ignored(self):
        assert octave_fold_shift(np.array([-3.0]))[0] == 0.0

    def test_noise_tolerance(self):
        # f0 잡음으로 -12가 -12.8까지 흔들려도 인공물로 인정 (잔차 1.5 이내)
        assert octave_fold_shift(np.array([-12.8]))[0] == 12.0
        assert octave_fold_shift(np.array([-14.0]))[0] == 0.0


class TestAgreedOctaveShift:
    """앞뒤 문맥이 둘 다 같은 이탈을 가리켜야 접는다."""

    def test_transient_excursion_folds(self):
        # 앞뒤 모두 60인데 혼자 48 — 잠깐 내려갔다 돌아오는 서브하모닉 락온
        shift = _agreed_octave_shift(np.array([48.0]), np.array([60.0]), np.array([60.0]))
        assert shift[0] == 12.0

    def test_sustained_leap_preserved(self):
        # 앞은 60, 뒤는 48 — 실제로 옥타브 내려가 머무르는 멜로디라 접으면 안 된다
        shift = _agreed_octave_shift(np.array([48.0]), np.array([60.0]), np.array([48.0]))
        assert shift[0] == 0.0


class TestFoldLineOctaves:
    def test_wide_range_line_preserved(self):
        # 한 프레이즈가 2옥타브를 오가는 곡이 실제로 있다 (numb numb: 라인 음역 23~26반음).
        # 고정 14반음 창 폴딩은 이런 라인을 통째로 망가뜨렸다 — 벤치 잡음 0에서 정답 695노트 파괴.
        track = make_track([(0.5, 56.0), (0.5, 63.0), (0.5, 72.0), (0.5, 79.0), (0.5, 68.0)])
        before = track.midi.copy()
        fold_line_octaves(track, [(0.0, 2.5)])
        assert np.array_equal(track.midi, before)

    def test_transient_subharmonic_lock_folded(self):
        # 라인 중간 0.3초가 -12로 잠긴 뒤 원 궤적으로 복귀 — 이건 인공물이라 접어야 한다
        track = make_track([(1.0, 67.0), (0.3, 55.0), (1.0, 67.0)])
        folded = fold_line_octaves(track, [(0.0, 2.3)])
        assert folded > 0
        assert track.midi[track.voiced].min() >= 66.0

    def test_low_verse_section_not_raised(self):
        # 옥타브 낮은 벌스가 여러 라인 이어지는 경우 — 이웃도 낮으므로 끌어올리면 안 된다
        spec = [(0.5, 72.0), (0.2, None)] * 2 + [(0.5, 60.0), (0.2, None)] * 3
        track = make_track(spec)
        spans = [(i * 0.7, i * 0.7 + 0.5) for i in range(5)]
        before = track.midi.copy()
        fold_line_octaves(track, spans)
        assert np.array_equal(track.midi, before, equal_nan=True)

    def test_double_folded_line_still_rescued(self):
        # 이중 폴딩(-24)은 어떤 곡의 음역으로도 설명되지 않는 인공물이라 전역 가드가 잡는다
        spec = [(0.5, 60.0), (0.2, None)] * 3 + [(0.5, 36.0), (0.2, None)] + [(0.5, 60.0)]
        track = make_track(spec)
        spans = [(i * 0.7, i * 0.7 + 0.5) for i in range(5)]
        fold_line_octaves(track, spans)
        assert float(np.median(track.midi[(track.times >= 2.1) & (track.times < 2.6)])) == 60.0


class TestFoldNotesToLocalMedian:
    def _notes(self, midis: list[int]) -> list[dict]:
        return [
            {"midi": m, "start": i * 0.3, "end": i * 0.3 + 0.3, "confidence": 0.9}
            for i, m in enumerate(midis)
        ]

    def test_isolated_octave_outlier_folded(self):
        notes = self._notes([67, 67, 55, 67, 67])
        _fold_notes_to_local_median(notes)
        assert notes[2]["midi"] == 67

    def test_wide_but_non_octave_leap_preserved(self):
        # 라인 중앙값에서 16반음 떨어진 진짜 고음 — 라인 중앙값 ±9 규칙은 이걸 접었다
        notes = self._notes([60, 60, 76, 60, 60])
        _fold_notes_to_local_median(notes)
        assert notes[2]["midi"] == 76

    def test_register_change_preserved(self):
        # 라인 후반이 통째로 옥타브 위로 올라가 머무르는 경우 (뒤 문맥이 일치)
        notes = self._notes([60, 60, 60, 72, 72, 72])
        _fold_notes_to_local_median(notes)
        assert [n["midi"] for n in notes] == [60, 60, 60, 72, 72, 72]

    def test_isolated_exact_octave_ornament_is_folded_known_limit(self):
        """알려진 한계: 앞뒤와 정확히 12반음 차이 나는 **단발** 노트는 되돌려진다.

        음고 정보만으로는 "한 음만 옥타브 위로 튀는 장식음"과 "서브하모닉 락온"이
        구분되지 않는다 — 둘 다 앞뒤 문맥과 정확히 ±12 차이 나는 일시적 이탈이다.
        실제 f0에는 가창의 글라이드가 남아 구분 단서가 되지만, 그 판정은 f0 백엔드의
        전이 구간 해상도에 의존하므로 이 계층에서 확정하지 않는다. UST 12곡 벤치
        기준 이 오폴딩은 정답의 0.3%이고, 되돌리기 전(고정 창 폴딩)은 11%였다.
        """
        notes = self._notes([67, 67, 79, 67, 67])
        _fold_notes_to_local_median(notes)
        assert notes[2]["midi"] == 67


class TestAnchorBoundaries:
    def test_boundary_snaps_to_pitch_transition(self):
        # f0는 0.50s에서 60→64로 바뀌는데 정렬 앵커는 0.44s로 어긋나 있다.
        # 경계를 f0 전이점으로 당겨야 노트 온셋 정확도가 정렬 오차에 묶이지 않는다.
        track = make_track([(0.5, 60.0), (0.5, 64.0)])
        notes = notes_from_anchor_spans(track, [(0.0, 0.44), (0.44, 1.0)])
        assert [n["midi"] for n in notes] == [60, 64]
        assert abs(notes[1]["start"] - 0.5) <= 0.02

    def test_boundary_snap_bounded_by_window(self):
        # 전이점이 창(기본 60ms) 밖이면 정렬 결과를 존중해 크게 움직이지 않는다
        track = make_track([(0.8, 60.0), (0.4, 64.0)])
        notes = notes_from_anchor_spans(track, [(0.0, 0.5), (0.5, 1.2)])
        assert notes[1]["start"] <= 0.5 + 0.07

    def test_same_pitch_boundary_untouched(self):
        # 두 음절이 같은 음이면 옮길 f0 근거가 없다 — 앵커 그대로 (리듬 표시 유지)
        track = make_track([(0.6, 60.0)])
        notes = notes_from_anchor_spans(track, [(0.0, 0.3), (0.3, 0.6)])
        assert [n["start"] for n in notes] == [0.0, 0.3]
        assert len(notes) == 2

    def test_note_end_trimmed_to_voiced_extent(self):
        # 앵커 끝은 다음 글자 시작까지 늘어나 쉼표를 넘는다 — 노트 막대는 발성까지만
        track = make_track([(0.3, 62.0), (0.5, None)])
        notes = notes_from_anchor_spans(track, [(0.0, 0.8)])
        assert len(notes) == 1
        assert notes[0]["end"] == pytest.approx(0.3, abs=0.02)

    def test_note_start_stays_on_anchor(self):
        # 자음 때문에 발성이 늦게 시작해도 노트 시작은 앵커(가사)에 잠긴 채로 둔다
        track = make_track([(0.1, None), (0.4, 65.0)])
        notes = notes_from_anchor_spans(track, [(0.0, 0.5)])
        assert notes[0]["start"] == 0.0


class TestQuantizeAccuracyEndToEnd:
    """정답 노트열 → 합성 f0 → 파이프라인 전체 → 노트 단위 일치율.

    scripts/bench_melody_ust.py를 UST 없이 축소 재현한 것. 여기서 깨지면 벤치도 깨진다.
    """

    def _truth(self) -> list[tuple[int, float, float]]:
        # 넓은 음역(라인 폭 최대 21반음) + 저음 라인 — 폴딩이 잘못되면 여기서 무너진다.
        # 정확히 12반음짜리 단발 장식음은 일부러 뺐다: 그건 음고만으로 락온과 구분이
        # 안 되는 알려진 한계라 별도 테스트에서 명시적으로 다룬다.
        pattern = [60, 64, 67, 72, 67, 64, 60, 55, 60, 67, 76, 67]
        out: list[tuple[int, float, float]] = []
        t = 0.0
        for i, m in enumerate(pattern * 3):
            out.append((m, t, t + 0.3))
            t += 0.3
            if i % 6 == 5:  # 프레이즈 경계
                t += 0.5
        return out

    def test_matches_truth_notes(self):
        truth = self._truth()
        total = truth[-1][2] + 0.5
        n = int(total / FRAME_DT)
        times = (np.arange(n) + 0.5) * FRAME_DT
        midi = np.full(n, np.nan)
        for m, s, e in truth:
            midi[int(s / FRAME_DT) : int(e / FRAME_DT)] = m
        rng = np.random.default_rng(0)
        midi += rng.normal(0.0, 0.08, n)  # 프레임 지터
        voiced = np.isfinite(midi)
        hz = np.where(voiced, 440.0 * np.power(2.0, (midi - 69.0) / 12.0), 0.0)

        segments: list[dict] = []
        for m, s, e in truth:
            if segments and s - segments[-1]["end"] <= 0.35:
                segments[-1]["end"] = e
                segments[-1]["words"].append({"word": "x", "start": s, "end": s + 0.05})
            else:
                segments.append(
                    {"text": "x", "start": s, "end": e,
                     "words": [{"word": "x", "start": s, "end": s + 0.05}]}
                )

        MelodyExtractor().annotate_timestamps(
            None, segments, precomputed_f0=(np.nan_to_num(hz), times)
        )
        pred = [n for seg in segments for n in seg.get("notes", [])]
        matched = sum(
            1
            for m, s, _e in truth
            if any(abs(p["start"] - s) <= 0.08 and abs(p["midi"] - m) <= 1 for p in pred)
        )
        assert matched / len(truth) >= 0.95
