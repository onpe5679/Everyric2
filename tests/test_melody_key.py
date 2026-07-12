"""곡 키 추정(K-S)·스케일 스냅·옥타브 전역 가드 테스트 — 합성 F0Track 기반."""
import numpy as np

from everyric2.melody.extractor import (
    F0Track,
    estimate_key,
    fold_line_octaves,
    snap_notes_to_key,
)


def _track(midi: np.ndarray, dt: float = 0.02) -> F0Track:
    times = np.arange(len(midi)) * dt
    return F0Track(times=times, midi=midi.astype(float), voiced=np.ones(len(midi), dtype=bool))


def _c_major_track() -> F0Track:
    # C장조 스케일 음을 프레임 40개씩 유지 (길게 유지되는 음이 히스토그램을 지배)
    notes = [60, 62, 64, 65, 67, 69, 71, 67, 64, 60]
    return _track(np.concatenate([np.full(40, m) for m in notes]))


class TestEstimateKey:
    def test_c_major_detected(self):
        key = estimate_key(_c_major_track())
        assert key is not None
        assert key["tonic"] == 0
        assert key["mode"] == "major"
        assert key["name"] == "C"
        assert key["confidence"] > 0.8

    def test_a_minor_detected(self):
        # A 자연단음계 — 라(57) 중심으로 유지
        notes = [57, 59, 60, 62, 64, 65, 67, 64, 60, 57]
        key = estimate_key(_track(np.concatenate([np.full(40, m) for m in notes])))
        assert key is not None
        assert key["tonic"] == 9
        assert key["mode"] == "minor"
        assert key["name"] == "Am"

    def test_too_few_frames_returns_none(self):
        assert estimate_key(_track(np.full(10, 60.0))) is None


class TestSnapNotesToKey:
    def _key(self):
        return {"tonic": 0, "mode": "major", "name": "C", "confidence": 0.9}

    def test_boundary_note_snaps_to_scale(self):
        # f0 중심 63.45 → 63(D#, 스케일 밖)으로 양자화된 노트가 64(E)로 스냅
        t = np.arange(0.0, 0.5, 0.01)
        track = F0Track(times=t, midi=np.full(len(t), 63.45), voiced=np.ones(len(t), dtype=bool))
        segs = [{"notes": [{"midi": 63, "start": 0.0, "end": 0.5}]}]
        assert snap_notes_to_key(segs, track, self._key()) == 1
        assert segs[0]["notes"][0]["midi"] == 64

    def test_solid_chromatic_note_preserved(self):
        # f0 중심이 63.02에 확실히 앉은 반음계 경과음은 건드리지 않는다
        t = np.arange(0.0, 0.5, 0.01)
        track = F0Track(times=t, midi=np.full(len(t), 63.02), voiced=np.ones(len(t), dtype=bool))
        segs = [{"notes": [{"midi": 63, "start": 0.0, "end": 0.5}]}]
        assert snap_notes_to_key(segs, track, self._key()) == 0
        assert segs[0]["notes"][0]["midi"] == 63

    def test_in_scale_note_untouched(self):
        t = np.arange(0.0, 0.5, 0.01)
        track = F0Track(times=t, midi=np.full(len(t), 64.4), voiced=np.ones(len(t), dtype=bool))
        segs = [{"notes": [{"midi": 64, "start": 0.0, "end": 0.5}]}]
        assert snap_notes_to_key(segs, track, self._key()) == 0


class TestGlobalOctaveGuard:
    def test_two_octave_drop_folds_back(self):
        # 정상 라인 5개(60) + 이중 폴딩으로 -24 추락한 라인 1개 → 전곡 기준(60)으로 복귀
        frames = [np.full(50, 60.0) for _ in range(5)] + [np.full(50, 36.0)]
        track = _track(np.concatenate(frames))
        spans = [(i * 1.0, (i + 1) * 1.0) for i in range(6)]
        folded = fold_line_octaves(track, spans)
        assert folded >= 50
        assert float(np.median(track.midi[250:])) == 60.0
