"""Melody extraction (f0 → MIDI notes) tests."""

import numpy as np
import pytest

from everyric2.melody.extractor import (
    F0Track,
    MelodyExtractor,
    anchor_spans_from_words,
    hz_to_midi,
    notes_for_span,
    notes_from_anchor_spans,
    snap_octave_jumps,
)

FRAME_DT = 0.01


def make_track(spec: list[tuple[float, float | None]]) -> F0Track:
    """(길이초, midi|None=무성음) 목록으로 합성 F0Track 생성."""
    midi_frames: list[float] = []
    for dur, midi in spec:
        n = int(round(dur / FRAME_DT))
        midi_frames.extend([midi if midi is not None else np.nan] * n)
    midi_arr = np.array(midi_frames, dtype=np.float64)
    voiced = ~np.isnan(midi_arr)
    times = (np.arange(len(midi_arr)) + 0.5) * FRAME_DT
    return F0Track(times=times, midi=midi_arr, voiced=voiced)


class TestNotesFromAnchors:
    def test_notes_cut_at_anchor_boundaries(self):
        # 음이 바뀌는 두 음절 — 노트 경계가 앵커(음절) 경계와 정확히 일치해야 한다
        track = make_track([(0.3, 60.0), (0.3, 64.0)])
        notes = notes_from_anchor_spans(track, [(0.0, 0.3), (0.3, 0.6)])
        assert [n["midi"] for n in notes] == [60, 64]
        assert notes[0]["start"] == 0.0
        assert notes[1]["start"] == 0.3

    def test_same_pitch_anchors_stay_separate(self):
        # 같은 음이어도 음절마다 별도 노트 — 노래방 악보처럼 리듬이 보여야 한다
        track = make_track([(0.6, 60.0)])
        notes = notes_from_anchor_spans(track, [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6)])
        assert len(notes) == 3
        assert [n["start"] for n in notes] == [0.0, 0.2, 0.4]
        assert all(n["midi"] == 60 for n in notes)

    def test_unvoiced_anchor_skipped(self):
        track = make_track([(0.3, 60.0), (0.3, None)])
        notes = notes_from_anchor_spans(track, [(0.0, 0.3), (0.3, 0.6)])
        assert len(notes) == 1 and notes[0]["midi"] == 60

    def test_long_melisma_splits_inside_anchor(self):
        # 길게 끄는 음절(1.5s)은 내부 run 분할 허용 — 단 첫 노트는 앵커 시작에 스냅
        track = make_track([(0.8, 60.0), (0.7, 65.0)])
        notes = notes_from_anchor_spans(track, [(0.0, 1.5)])
        assert len(notes) == 2
        assert notes[0]["start"] == 0.0
        assert [n["midi"] for n in notes] == [60, 65]

    def test_anchor_spans_from_words(self):
        words = [
            {"word": "a", "start": 0.0, "end": 0.05},
            {"word": "b", "start": 0.5, "end": 0.55},
        ]
        spans = anchor_spans_from_words(words, 3.0)
        assert spans[0] == (0.0, 0.5)  # 다음 글자 시작까지 확장
        assert spans[1][0] == 0.5
        assert abs(spans[1][1] - 2.05) < 1e-9  # 라인 끝 전이라도 +1.5s 상한


class TestSnapOctaveJumps:
    def test_octave_lock_on_folded_back(self):
        # 62에서 노래하다 트래커가 한 옥타브 아래(50)로 잠기는 실측 실패 모드
        track = make_track([(0.3, 62.0), (0.3, 50.0), (0.3, 63.0)])
        snapped = snap_octave_jumps(track)
        assert snapped > 0
        assert np.all(track.midi[track.voiced] >= 60.0)  # 50 구간이 62로 접힘

    def test_double_octave_folded(self):
        # +24반음(4배음) 튐도 접힌다
        track = make_track([(0.3, 60.0), (0.2, 84.0), (0.3, 61.0)])
        snap_octave_jumps(track)
        assert track.midi[track.voiced].max() < 70.0

    def test_genuine_leap_kept(self):
        # 완전5도(7반음) 이내 실제 도약은 보존
        track = make_track([(0.3, 60.0), (0.3, 66.0)])
        snapped = snap_octave_jumps(track)
        assert snapped == 0
        assert set(np.round(track.midi[track.voiced]).astype(int)) == {60, 66}

    def test_reset_after_long_silence(self):
        # 0.5s 넘는 무성 구간 뒤에는 기준이 리셋돼 스냅하지 않는다 (프레이즈 경계)
        track = make_track([(0.3, 72.0), (0.8, None), (0.3, 55.0)])
        snapped = snap_octave_jumps(track)
        assert snapped == 0
        assert 55.0 in track.midi[track.voiced]

    def test_snapped_value_becomes_new_reference(self):
        # 스냅된 값이 다음 프레임의 기준이 되어 연쇄 구간 전체가 일관되게 접힌다
        track = make_track([(0.2, 62.0), (0.4, 50.0), (0.2, 50.0)])
        snap_octave_jumps(track)
        vals = set(np.round(track.midi[track.voiced]).astype(int))
        assert vals == {62}


class TestHzToMidi:
    def test_a4_is_69(self):
        assert hz_to_midi(np.array([440.0]))[0] == pytest.approx(69.0)

    def test_octaves(self):
        assert hz_to_midi(np.array([220.0]))[0] == pytest.approx(57.0)
        assert hz_to_midi(np.array([880.0]))[0] == pytest.approx(81.0)

    def test_unvoiced_is_nan(self):
        assert np.isnan(hz_to_midi(np.array([0.0]))[0])


class TestNotesForSpan:
    def test_steady_note(self):
        track = make_track([(1.0, 60.0)])
        notes = notes_for_span(track, 0.0, 1.0)
        assert len(notes) == 1
        assert notes[0]["midi"] == 60
        assert notes[0]["confidence"] > 0.9

    def test_two_notes(self):
        track = make_track([(0.5, 60.0), (0.5, 64.0)])
        notes = notes_for_span(track, 0.0, 1.0)
        assert [n["midi"] for n in notes] == [60, 64]
        assert notes[0]["end"] <= notes[1]["start"]

    def test_unvoiced_span_returns_empty(self):
        track = make_track([(1.0, None)])
        assert notes_for_span(track, 0.0, 1.0) == []

    def test_low_voiced_ratio_returns_empty(self):
        track = make_track([(0.05, 60.0), (0.95, None)])
        assert notes_for_span(track, 0.0, 1.0) == []

    def test_single_frame_blip_is_absorbed(self):
        track = make_track([(0.4, 60.0), (0.01, 61.0), (0.4, 60.0)])
        notes = notes_for_span(track, 0.0, 0.81)
        assert [n["midi"] for n in notes] == [60]

    def test_short_consonant_gap_stays_one_note(self):
        track = make_track([(0.3, 60.0), (0.05, None), (0.3, 60.0)])
        notes = notes_for_span(track, 0.0, 0.65)
        assert [n["midi"] for n in notes] == [60]

    def test_rapid_alternation_falls_back_to_mode(self):
        # 반음 사이를 프레임 단위로 오가는 비브라토 — 안정 run이 없어 최빈값 폴백
        spec: list[tuple[float, float | None]] = [(0.02, 60.0), (0.02, 61.0)] * 15
        track = make_track(spec)
        notes = notes_for_span(track, 0.0, 0.6)
        assert len(notes) == 1
        assert notes[0]["midi"] in (60, 61)

    def test_span_outside_track(self):
        track = make_track([(1.0, 60.0)])
        assert notes_for_span(track, 5.0, 6.0) == []


class TestAnnotateTimestamps:
    def _extractor_with_track(self, monkeypatch, track: F0Track) -> MelodyExtractor:
        extractor = MelodyExtractor()
        monkeypatch.setattr(extractor, "extract_f0", lambda _audio, **_kw: track)
        return extractor

    def test_segment_notes_split_by_pitch(self, monkeypatch):
        # 라인 안에서 피치가 60→64로 바뀌면 노트도 2개로 분할된다
        track = make_track([(0.5, 60.0), (0.5, 64.0)])
        extractor = self._extractor_with_track(monkeypatch, track)
        timestamps = [
            {
                "text": "가나",
                "start": 0.0,
                "end": 1.0,
                # CTC word_segments는 글자 단위 초단기 span — 노트는 세그먼트 레벨에만 붙는다
                "words": [
                    {"word": "가", "start": 0.0, "end": 0.06},
                    {"word": "나", "start": 0.5, "end": 0.56},
                ],
            }
        ]
        count = extractor.annotate_timestamps(object(), timestamps)
        assert count == 1
        assert [n["midi"] for n in timestamps[0]["notes"]] == [60, 64]
        assert "notes" not in timestamps[0]["words"][0]

    def test_segment_without_words_gets_notes(self, monkeypatch):
        track = make_track([(1.0, 67.0)])
        extractor = self._extractor_with_track(monkeypatch, track)
        timestamps = [{"text": "라", "start": 0.0, "end": 1.0}]
        count = extractor.annotate_timestamps(object(), timestamps)
        assert count == 1
        assert timestamps[0]["notes"][0]["midi"] == 67

    def test_unvoiced_segment_left_untouched(self, monkeypatch):
        track = make_track([(1.0, None)])
        extractor = self._extractor_with_track(monkeypatch, track)
        timestamps = [{"text": "쉿", "start": 0.0, "end": 1.0}]
        count = extractor.annotate_timestamps(object(), timestamps)
        assert count == 0
        assert "notes" not in timestamps[0]


@pytest.mark.skipif(
    not MelodyExtractor().is_available(), reason="torchfcpe not installed"
)
class TestFcpeIntegration:
    def test_sine_wave_maps_to_a4(self):
        from everyric2.audio.loader import AudioData
        from everyric2.config.settings import MelodySettings
        from everyric2.melody.extractor import MELODY_SAMPLE_RATE

        sr = MELODY_SAMPLE_RATE
        t = np.arange(int(sr * 1.2)) / sr
        waveform = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
        audio = AudioData(waveform=waveform, sample_rate=sr, duration=1.2)

        # FCPE만 검증 — 합성 사인파에 demucs 분리를 태울 이유가 없다
        extractor = MelodyExtractor(MelodySettings(separate_vocals=False, f0_model="fcpe"))
        timestamps = [{"text": "아", "start": 0.1, "end": 1.1}]
        count = extractor.annotate_timestamps(audio, timestamps)
        assert count == 1
        assert timestamps[0]["notes"][0]["midi"] == 69  # A4


def _rmvpe_weights_available() -> bool:
    from everyric2.config.settings import MelodySettings

    return MelodySettings(f0_model="rmvpe").rmvpe_model_path.exists()


@pytest.mark.skipif(
    not (MelodyExtractor().is_available() and _rmvpe_weights_available()),
    reason="rmvpe.pt weights not downloaded (see MelodySettings.rmvpe_model_path)",
)
class TestRmvpeIntegration:
    def test_sine_wave_maps_to_a4(self):
        # RMVPE 백엔드로 A/B 검증한 것과 동일한 사인파 회귀 — FCPE 통합 테스트의 RMVPE 버전.
        from everyric2.audio.loader import AudioData
        from everyric2.config.settings import MelodySettings
        from everyric2.melody.extractor import MELODY_SAMPLE_RATE

        sr = MELODY_SAMPLE_RATE
        t = np.arange(int(sr * 1.2)) / sr
        waveform = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
        audio = AudioData(waveform=waveform, sample_rate=sr, duration=1.2)

        extractor = MelodyExtractor(MelodySettings(separate_vocals=False, f0_model="rmvpe"))
        timestamps = [{"text": "아", "start": 0.1, "end": 1.1}]
        count = extractor.annotate_timestamps(audio, timestamps)
        assert count == 1
        assert timestamps[0]["notes"][0]["midi"] == 69  # A4
        assert extractor._backend == "rmvpe"  # FCPE로 조용히 폴백되지 않았는지 확인
