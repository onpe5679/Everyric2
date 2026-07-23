"""Vocal melody extraction: f0 estimation → per-syllable MIDI notes.

파이프라인:
  1. 곡 전체 오디오에 f0 백엔드(FCPE 또는 RMVPE)를 1회 통과시켜 프레임 단위
     f0(Hz)를 얻는다 (unvoiced=0).
  2. 정렬 결과의 각 단어/음절 [start, end) 구간에서 f0 프레임을 잘라
     MIDI 반음으로 양자화하고, 안정 구간(run)별로 대표 노트를 만든다.

f0 백엔드는 MelodySettings.f0_model로 선택한다 (기본 rmvpe, 가중치 없으면 FCPE로
자동 폴백). 두 백엔드 모두 폴리포닉 믹스에서도 동작하지만, 반주가 큰 곡에서는
기타/베이스 피치가 노트에 섞인다 — 그래서 기본적으로 demucs로 보컬을 분리한 뒤
f0를 뽑는다 (EVERYRIC_MELODY_SEPARATE_VOCALS, demucs 미설치·실패 시 믹스로 폴백).
torchfcpe 미설치 시 조용히 비활성화된다 (RMVPE만으로는 폴백 경로가 없어 최소
torchfcpe는 필요).
"""

import logging
import threading
from dataclasses import dataclass

import numpy as np

from everyric2.audio.loader import AudioData
from everyric2.config.settings import MelodySettings, get_settings

logger = logging.getLogger(__name__)

MELODY_SAMPLE_RATE = 16000


@dataclass
class F0Track:
    """프레임 단위 f0 트랙 (시간축은 초)."""

    times: np.ndarray  # (frames,) 각 프레임 중심 시각
    midi: np.ndarray  # (frames,) float MIDI, unvoiced는 NaN
    voiced: np.ndarray  # (frames,) bool


def hz_to_midi(hz: np.ndarray) -> np.ndarray:
    """Hz → float MIDI (0 이하 입력은 NaN)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        midi = 69.0 + 12.0 * np.log2(np.asarray(hz, dtype=np.float64) / 440.0)
    return np.where(np.asarray(hz) > 0, midi, np.nan)


# Krumhansl-Schmuckler 키 프로파일 (Krumhansl & Kessler 1982) — pitch class 0 = 으뜸음
_KS_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_KS_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
_KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
_MINOR_SCALE = (0, 2, 3, 5, 7, 8, 10)  # 자연 단음계


def estimate_key(track: F0Track) -> dict | None:
    """유성 프레임의 pitch-class 히스토그램을 K-S 프로파일과 상관해 곡 키를 추정.

    프레임 수로 자연 가중되므로 길게 유지되는 음(구조적으로 중요한 음)이 더 세게
    반영된다. 반환: {"tonic": 0~11, "mode": major|minor, "name": "G#m", "confidence": r}.
    유성 프레임이 너무 적으면 None (간주 위주 곡 등 — 표시·보정 모두 생략).
    """
    voiced = track.voiced & np.isfinite(track.midi)
    if int(voiced.sum()) < 50:
        return None
    pcs = np.round(track.midi[voiced]).astype(int) % 12
    hist = np.bincount(pcs, minlength=12).astype(np.float64)
    if hist.sum() <= 0 or np.count_nonzero(hist) < 3:
        return None
    best: tuple[float, int, str] | None = None
    for mode, profile in (("major", _KS_MAJOR), ("minor", _KS_MINOR)):
        prof = np.asarray(profile, dtype=np.float64)
        for tonic in range(12):
            r = float(np.corrcoef(np.roll(hist, -tonic), prof)[0, 1])
            if not np.isfinite(r):
                continue
            if best is None or r > best[0]:
                best = (r, tonic, mode)
    if best is None:
        return None
    r, tonic, mode = best
    return {
        "tonic": tonic,
        "mode": mode,
        "name": _KEY_NAMES[tonic] + ("m" if mode == "minor" else ""),
        "confidence": round(max(0.0, r), 3),
    }


def _scale_pitch_classes(key: dict) -> set[int]:
    base = _MINOR_SCALE if key.get("mode") == "minor" else _MAJOR_SCALE
    return {(int(key["tonic"]) + d) % 12 for d in base}


def snap_notes_to_key(
    timestamps: list[dict], track: F0Track, key: dict, *, max_dev: float = 0.6
) -> int:
    """스케일 밖 노트 중 반올림 경계가 애매한 것만 이웃 스케일음으로 스냅 (제자리 수정).

    노트 반음은 f0의 최빈/중앙 반올림이라 실제 f0 중심이 x.4~x.6 사이에 걸치면
    반쯤 무작위로 이웃 반음에 떨어진다 — 그 애매한 경우에 한해 곡 키의 스케일음을
    타이브레이커로 쓴다. f0 중심이 원 노트에 명백히 가까운 진짜 반음계 경과음은
    보존한다 (스케일 밖 + 이웃이 f0 중심에서 max_dev 초과면 스냅 안 함).
    반환: 스냅한 노트 수.
    """
    scale = _scale_pitch_classes(key)
    snapped = 0
    for seg in timestamps:
        for n in seg.get("notes") or []:
            midi = int(n["midi"])
            if midi % 12 in scale:
                continue
            mask = (track.times >= n["start"]) & (track.times < n["end"]) & track.voiced
            if int(mask.sum()) < 3:
                continue
            center = float(np.nanmedian(track.midi[mask]))
            cands = [
                c for c in (midi - 1, midi + 1)
                if c % 12 in scale and abs(c - center) <= max_dev
            ]
            if not cands:
                continue
            cand = min(cands, key=lambda c: abs(c - center))
            # 이웃 후보가 f0 중심에서 원 노트보다 확연히 멀면 증거를 거스르는 것 — 보존
            if abs(cand - center) > abs(midi - center) + 0.25:
                continue
            n["midi"] = int(cand)
            snapped += 1
    return snapped


def downsample_f0_curve(
    track: F0Track, target_dt: float = 0.05, max_points: int = 12000
) -> dict | None:
    """f0 트랙을 확장 디버그 오버레이용 균일 곡선으로 다운샘플 (~20Hz).

    옥타브 폴딩 이전의 RAW 트랙을 넘겨야 모델의 서브하모닉 락온 같은 원본 거동이
    보인다. unvoiced 프레임은 None — 클라이언트가 선을 끊는 신호로 쓴다.
    """
    n = len(track.times)
    if n == 0:
        return None
    frame_dt = float(track.times[1] - track.times[0]) if n > 1 else target_dt
    if frame_dt <= 0:
        return None
    stride = max(1, round(target_dt / frame_dt))
    if n / stride > max_points:
        stride = int(np.ceil(n / max_points))
    midi = [
        round(float(track.midi[i]), 1)
        if bool(track.voiced[i]) and np.isfinite(track.midi[i])
        else None
        for i in range(0, n, stride)
    ]
    return {
        "t0": round(float(track.times[0]), 3),
        "dt": round(frame_dt * stride, 4),
        "midi": midi,
    }


def snap_octave_jumps(
    track: F0Track,
    *,
    max_jump: float = 7.0,
    reset_gap_sec: float = 0.5,
) -> int:
    """옥타브/배음 락온 보정 — midi 배열을 제자리 수정하고 스냅한 프레임 수를 반환.

    FCPE(local_argmax)는 배음 사이를 수백 ms 단위로 오가며 잠기는 실패 모드가 있다
    (로키 실측: 정답 옥타브와 -12반음 지점이 히스토그램상 대등한 쌍봉).
    직전 유성 프레임과 max_jump 반음을 초과해 차이 나는 프레임을 ±12반음 단위로
    접어 궤적 연속성을 강제한다 — 노트 레벨 ≥7반음 도약 37%→5% 실측.
    reset_gap_sec 이상 무성 구간이 지나면 기준을 리셋한다 (프레이즈 경계).
    """
    prev_midi: float | None = None
    prev_t: float | None = None
    snapped = 0
    for i in range(len(track.times)):
        if not track.voiced[i]:
            continue
        t = float(track.times[i])
        m = float(track.midi[i])
        if prev_midi is not None and prev_t is not None and t - prev_t <= reset_gap_sec:
            diff = m - prev_midi
            if abs(diff) > max_jump:
                # ±12k 시프트 중 직전 값에 가장 가까워지는 후보로 접는다
                k = round(diff / 12.0)
                candidate = m - 12.0 * k
                if abs(candidate - prev_midi) < abs(diff):
                    track.midi[i] = candidate
                    m = candidate
                    snapped += 1
        prev_midi = m
        prev_t = t
    return snapped


def notes_for_span(
    track: F0Track,
    start: float,
    end: float,
    *,
    min_note_sec: float = 0.1,
    max_gap_sec: float = 0.08,
    min_voiced_ratio: float = 0.15,
) -> list[dict]:
    """[start, end) 구간의 f0를 반음 양자화해 안정 노트 목록을 만든다.

    반환 노트: {"midi": int, "start": s, "end": s, "confidence": 0~1}
    유성음 비율이 너무 낮거나 안정 구간이 없으면 빈 목록.
    """
    mask = (track.times >= start) & (track.times < end)
    if not mask.any():
        return []
    times = track.times[mask]
    midi = track.midi[mask]
    voiced = track.voiced[mask]

    n_voiced = int(voiced.sum())
    if n_voiced < 3 or n_voiced / len(voiced) < min_voiced_ratio:
        return []

    rounded = np.where(voiced, np.round(midi), np.nan)

    # 유성음 프레임을 훑으며 같은 반음이 이어지는 run으로 묶는다.
    # 짧은 무성음 공백(max_gap_sec)은 같은 run으로 잇는다 (자음/숨 등).
    runs: list[dict] = []  # {midi, start, end, frames}
    current: dict | None = None
    last_voiced_t: float | None = None
    for i in range(len(times)):
        if not voiced[i]:
            if (
                current is not None
                and last_voiced_t is not None
                and times[i] - last_voiced_t > max_gap_sec
            ):
                runs.append(current)
                current = None
            continue
        note = int(rounded[i])
        if current is not None and (
            note != current["midi"]
            or (last_voiced_t is not None and times[i] - last_voiced_t > max_gap_sec)
        ):
            runs.append(current)
            current = None
        if current is None:
            current = {"midi": note, "start": float(times[i]), "end": float(times[i]), "frames": 0}
        current["end"] = float(times[i])
        current["frames"] += 1
        last_voiced_t = float(times[i])
    if current is not None:
        runs.append(current)

    # 너무 짧은 run(단발 흔들림) 제거 후, 같은 반음의 인접 run 병합
    stable = [r for r in runs if r["end"] - r["start"] >= min_note_sec]
    merged: list[dict] = []
    for r in stable:
        prev = merged[-1] if merged else None
        if prev is not None and prev["midi"] == r["midi"] and r["start"] - prev["end"] <= 0.15:
            prev["end"] = r["end"]
            prev["frames"] += r["frames"]
        else:
            merged.append(dict(r))

    # 전부 탈락했으면(비브라토 심함 등) 최빈 반음 하나로 폴백
    if not merged:
        values, counts = np.unique(rounded[voiced].astype(int), return_counts=True)
        mode_midi = int(values[np.argmax(counts)])
        v_times = times[voiced]
        span_start, span_end = float(v_times[0]), float(v_times[-1])
        if span_end - span_start < min_note_sec:
            return []
        merged = [
            {
                "midi": mode_midi,
                "start": span_start,
                "end": span_end,
                "frames": int(counts.max()),
            }
        ]

    notes = []
    for r in merged:
        in_run = (times >= r["start"]) & (times <= r["end"]) & voiced
        agree = float(np.mean(rounded[in_run] == r["midi"])) if in_run.any() else 0.0
        notes.append(
            {
                "midi": r["midi"],
                "start": round(r["start"], 3),
                "end": round(r["end"], 3),
                "confidence": round(agree * (n_voiced / len(voiced)), 3),
            }
        )
    return notes


def fold_line_octaves(
    track: F0Track,
    spans: list[tuple[float, float]],
    *,
    window: float = 14.0,
    global_guard: float = 9.0,
) -> int:
    """라인별 지배 옥타브 창으로 f0를 접는다 — 체인 스냅의 구조적 함정 대체.

    프레임 체인 스냅(직전 프레임 기준 ±12 접기)은 리셋 직후 첫 프레임이
    서브하모닉이면 라인 전체가 저옥타브에 갇히고, 접힌 값이 다시 기준이 되어
    이중 폴딩(-24)까지 발생한다 (로키 벌스 실측: 라인 중앙값 59→26).
    여기서는 라인마다 (1) 유성 프레임이 가장 많이 모이는 window 반음 창을 찾아
    창 밖 프레임을 ±12k로 창 안에 접고, (2) 라인 전체가 전곡 기준(라인 중앙값들의
    중앙값)보다 global_guard 반음 이상 벗어난 서브하모닉/배음이면 라인 통째로
    ±12 이동한다. 반환: 접힌 프레임 수.
    """
    folded = 0
    line_medians: list[float | None] = []
    span_indices: list[np.ndarray] = []
    for s, e in spans:
        mask = (track.times >= s) & (track.times < e) & track.voiced
        idx = np.where(mask)[0]
        span_indices.append(idx)
        if len(idx) < 5:
            line_medians.append(None)
            continue
        vals = track.midi[idx]
        lo = int(np.floor(np.nanmin(vals)))
        hi = int(np.ceil(np.nanmax(vals)))
        best_w, best_c = lo, -1
        for w in range(lo, max(lo, hi - int(window)) + 1):
            c = int(((vals >= w) & (vals < w + window)).sum())
            if c > best_c:
                best_c, best_w = c, w
        center = best_w + window / 2
        for i in idx:
            m = float(track.midi[i])
            if best_w <= m < best_w + window:
                continue
            k = round((m - center) / 12.0)
            cand = m - 12.0 * k
            if best_w - 1 <= cand < best_w + window + 1:
                track.midi[i] = cand
                folded += 1
        line_medians.append(float(np.nanmedian(track.midi[idx])))

    valid = [m for m in line_medians if m is not None]
    if valid:
        g = float(np.median(valid))
        for idx, m in zip(span_indices, line_medians):
            if m is None or len(idx) == 0:
                continue
            # 2옥타브(이중 폴딩·서브서브하모닉)까지 잡도록 개선이 될 때까지 반복 이동
            shift = 0.0
            cur = m
            for _ in range(2):
                if g - cur >= global_guard and abs(cur + 12 - g) < abs(cur - g):
                    shift += 12.0
                    cur += 12.0
                elif cur - g >= global_guard and abs(cur - 12 - g) < abs(cur - g):
                    shift -= 12.0
                    cur -= 12.0
                else:
                    break
            if shift:
                track.midi[idx] += shift
                folded += len(idx)
    return folded


def notes_from_anchor_spans(
    track: F0Track,
    anchors: list[tuple[float, float]],
    *,
    min_note_sec: float = 0.08,
    max_gap_sec: float = 0.12,
    min_voiced_ratio: float = 0.15,
    long_span_sec: float = 1.0,
) -> list[dict]:
    """정렬된 음절(글자) 앵커 경계에서 노트를 자른다 — 노트 타이밍이 가사와 잠긴다.

    자유 f0 안정 run 분할은 리듬이 가사 하이라이트와 따로 놀았다. 여기서는 각 앵커
    [start, end) 구간에서 가장 오래 유지된 반음을 그 음절의 노트로 삼는다.
    같은 음이 이어져도 음절마다 별도 노트를 유지한다 — 노래방 악보처럼 음절 단위
    리듬이 보여야 하기 때문 (병합하면 통짜 긴 막대가 되어 리듬 정보가 사라진다).
    길게 끄는 음절(멜리스마, long_span_sec 초과)만 내부 run 분할을 허용하되
    첫 노트 시작은 앵커 시작에 스냅한다.
    """
    raw: list[dict] = []
    for a0, a1 in anchors:
        if a1 <= a0:
            continue
        in_span = (track.times >= a0) & (track.times < a1)
        mask = in_span & track.voiced
        n_voiced = int(mask.sum())
        if n_voiced < 3:
            continue
        if a1 - a0 > long_span_sec:
            sub = notes_for_span(
                track,
                a0,
                a1,
                min_note_sec=min_note_sec,
                max_gap_sec=max_gap_sec,
                min_voiced_ratio=min_voiced_ratio,
            )
            if sub:
                sub[0]["start"] = round(a0, 3)
                raw.extend(sub)
                continue
        # 최빈 반음 = 스팬에서 가장 오래 유지된 음 (모음 정상 상태).
        # 중앙값은 음절 시작부(자음/브레시 온셋)의 서브하모닉 프레임에 쉽게 오염된다.
        rounded = np.round(track.midi[mask]).astype(int)
        values, counts = np.unique(rounded, return_counts=True)
        midi = int(values[np.argmax(counts)])
        conf = round(float(counts.max()) / max(1, int(in_span.sum())), 3)
        raw.append({"midi": midi, "start": round(a0, 3), "end": round(a1, 3), "confidence": conf})

    return raw


def _fold_notes_to_line_median(notes: list[dict], *, threshold: float = 9.0) -> None:
    """라인 노트 중앙값 기준 옥타브 이탈 노트를 접는다 (제자리 수정).

    한 라인 안에서 인접 음절이 옥타브를 오가는 멜로디는 사실상 없다 — threshold
    반음을 초과해 벗어난 노트가 ±12 이동으로 중앙값에 가까워지면 접는다.
    실제 고음 이탈(±9 이내)은 보존된다.
    """
    if len(notes) < 3:
        return
    midis = sorted(n["midi"] for n in notes)
    med = midis[len(midis) // 2]
    for n in notes:
        d = n["midi"] - med
        if abs(d) <= threshold:
            continue
        k = round(d / 12.0)
        cand = n["midi"] - 12 * k
        if abs(cand - med) < abs(d):
            n["midi"] = int(cand)


def anchor_spans_from_words(words: list[dict], seg_end: float) -> list[tuple[float, float]]:
    """글자 타이밍을 음절 앵커로 변환 — 각 글자는 다음 글자 시작(또는 라인 끝)까지 노래된다.

    CTC 글자 span은 30~80ms로 짧고 끌리는 모음은 blank로 빠지므로, 앵커 끝은
    다음 글자 시작까지 확장하되 무가창 간주로 새지 않게 최대 1.5초로 제한한다.
    """
    spans: list[tuple[float, float]] = []
    starts = [float(w.get("start", 0.0)) for w in words]
    for i, w in enumerate(words):
        s = float(w.get("start", 0.0))
        e = float(w.get("end", s))
        next_start = starts[i + 1] if i + 1 < len(words) else seg_end
        cap = max(e, s + 0.05) + 1.5
        spans.append((s, max(s, min(next_start, seg_end, cap))))
    return spans


class MelodyExtractor:
    """FCPE 기반 멜로디 추출기. 모델은 최초 사용 시 lazy 로드."""

    def __init__(self, config: MelodySettings | None = None):
        self.config = config or get_settings().melody
        self._model = None
        self._backend: str | None = None  # "fcpe" | "rmvpe", set once _get_model() runs
        # annotate_timestamps가 채우는 디버그용 RAW f0 곡선 (다운샘플, 폴딩 전)
        self.last_f0_curve: dict | None = None
        # annotate_timestamps가 채우는 곡 키 추정 결과 — 싱크에 저장돼 레인에 표시된다
        self.last_key: dict | None = None

    def is_available(self) -> bool:
        try:
            import torchfcpe  # noqa: F401

            return True
        except ImportError:
            return False

    def _get_model(self):
        if self._model is None:
            import torch

            device = self.config.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"

            if self.config.f0_model == "rmvpe":
                try:
                    from everyric2.melody.rmvpe import RMVPEPredictor

                    model_path = self.config.rmvpe_model_path
                    if not model_path.exists():
                        raise FileNotFoundError(str(model_path))
                    self._model = RMVPEPredictor(str(model_path), device=device)
                    self._backend = "rmvpe"
                except Exception:
                    logger.warning(
                        "RMVPE backend unavailable (weights missing or load failed); "
                        "falling back to FCPE",
                        exc_info=True,
                    )

            if self._model is None:
                from torchfcpe import spawn_bundled_infer_model

                self._model = spawn_bundled_infer_model(device=device)
                self._backend = "fcpe"
        return self._model

    def _maybe_separate(self, audio: AudioData) -> AudioData:
        """설정이 켜져 있으면 demucs로 보컬만 분리해 반환. 실패하면 원본 믹스."""
        if not self.config.separate_vocals:
            return audio
        try:
            import torch

            from everyric2.audio.separator import VocalSeparator

            separator = VocalSeparator()
            if not separator.is_available():
                logger.info("demucs not installed; extracting f0 from the mix")
                return audio
            result = separator.separate(audio, use_gpu=torch.cuda.is_available())
            logger.info("Vocal separation done; extracting f0 from vocals stem")
            return result.vocals
        except Exception:
            logger.exception("Vocal separation failed; falling back to the mix")
            return audio

    def _infer_f0(
        self, audio: AudioData, vocals: AudioData | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """곡 전체에 f0 백엔드를 1회 통과시켜 (f0_hz, times)를 낸다 — 정렬 결과에 무의존.

        이 부분이 파이프라인에서 무거운(GPU) 단계라 WS2-B가 CTC 정렬과 병렬로 돌린다
        (precompute_f0). vocals가 주어지면 이미 분리된 스템으로 간주하고 재분리를 건너뛴다.
        vocal_regions 마스킹·옥타브 스냅 같은 정렬 의존 후처리는 여기서 하지 않는다."""
        import librosa
        import torch

        audio = vocals if vocals is not None else self._maybe_separate(audio)
        waveform = audio.waveform
        if audio.sample_rate != MELODY_SAMPLE_RATE:
            waveform = librosa.resample(
                waveform, orig_sr=audio.sample_rate, target_sr=MELODY_SAMPLE_RATE
            )

        model = self._get_model()
        if self._backend == "rmvpe":
            f0 = model.infer(waveform, threshold=self.config.rmvpe_threshold)
        else:
            audio_t = torch.from_numpy(np.ascontiguousarray(waveform, dtype=np.float32))
            audio_t = audio_t.unsqueeze(0).unsqueeze(-1)
            with torch.no_grad():
                f0 = model.infer(
                    audio_t,
                    sr=MELODY_SAMPLE_RATE,
                    decoder_mode="local_argmax",
                    threshold=self.config.threshold,
                    interp_uv=False,
                )
            f0 = f0.squeeze().cpu().numpy().astype(np.float64)

        f0 = np.asarray(f0, dtype=np.float64)
        duration = len(waveform) / MELODY_SAMPLE_RATE
        frame_dt = duration / max(1, len(f0))
        times = (np.arange(len(f0)) + 0.5) * frame_dt
        return f0, times

    def precompute_f0(
        self, audio: AudioData, vocals: AudioData | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """f0 전곡 추론만 수행(정렬 무의존) → (f0_hz, times). CTC 정렬과 병렬 실행용 (WS2-B).

        결과를 annotate_timestamps(..., precomputed_f0=...)로 주입하면 재추론 없이 노트를
        부착한다. 모델 로드/추론은 이 호출에서 일어나므로 별도 스레드에서 부르면 정렬과 겹친다.
        """
        return self._infer_f0(audio, vocals=vocals)

    def extract_f0(
        self,
        audio: AudioData,
        vocals: AudioData | None = None,
        vocal_regions: list[tuple[float, float]] | None = None,
        apply_snap: bool | None = None,
        precomputed: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> F0Track:
        """곡 전체에서 프레임 단위 f0 트랙을 뽑는다 (분리 옵션 적용 후 f0 백엔드 1회 추론).

        precomputed=(f0_hz, times)가 주어지면 재추론을 건너뛰고 그 값으로 트랙을 구성한다
        (WS2-B: 정렬과 병렬로 미리 계산한 f0 주입). vocals가 주어지면 이미 분리된 보컬 스템으로
        간주하고 재분리를 건너뛴다 (워커가 VAD용으로 분리한 스템을 재사용 — demucs 이중 실행
        방지). vocal_regions(VAD 발성 구간)가 주어지면 구간 밖 프레임을 무성 처리한다 — 분리
        잔여 노이즈가 라인 사이를 '유성'으로 이어버리면 옥타브 스냅의 리셋이 막혀 저음 기준이
        라인 경계를 넘어 전파되는 실측 실패 모드를 차단한다.
        """
        f0, times = precomputed if precomputed is not None else self._infer_f0(audio, vocals=vocals)
        f0 = np.asarray(f0, dtype=np.float64)
        voiced = (f0 >= self.config.f0_min) & (f0 <= self.config.f0_max)
        if vocal_regions:
            in_vocal = np.zeros_like(voiced)
            for s, e in vocal_regions:
                in_vocal |= (times >= s) & (times < e)
            voiced &= in_vocal
        midi = hz_to_midi(np.where(voiced, f0, 0.0))
        track = F0Track(times=times, midi=midi, voiced=voiced)
        # annotate_timestamps는 라인별 창 폴딩(fold_line_octaves)을 쓰므로 체인 스냅을 끈다
        if self.config.octave_snap if apply_snap is None else apply_snap:
            snapped = snap_octave_jumps(track)
            if snapped:
                logger.info(f"Octave snap folded {snapped} frames")
        return track

    def annotate_timestamps(
        self,
        audio: AudioData,
        timestamps: list[dict],
        vocals: AudioData | None = None,
        vocal_regions: list[tuple[float, float]] | None = None,
        precomputed_f0: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> int:
        """정렬 결과(worker 포맷)의 각 세그먼트(라인)에 notes를 붙인다.

        노트는 라인 [start, end) 구간에서 피치 안정 run 단위로 분할되므로
        단어/글자 경계와 무관하게 멜리스마·이음도 자연스럽게 나뉜다.
        (CTC의 word_segments는 글자 단위 span이라 노트 산출에는 너무 짧다.)
        vocals: 호출부가 이미 분리해 둔 보컬 스템 (있으면 재분리 생략).
        precomputed_f0: precompute_f0가 정렬과 병렬로 미리 계산한 (f0_hz, times) (WS2-B).
        주어지면 f0 재추론을 건너뛰고 정렬 의존 후처리(마스킹·폴딩·노트 부착)만 수행한다.
        반환값: notes가 붙은 세그먼트 수.
        """
        # 스냅 오염 방지 마스크는 라인 스팬 합집합이 기본 — VAD 구간을 쓰면 전곡 RMS
        # 퍼센타일 기준이라 조용한 벌스 라인까지 무성 처리돼 노트가 통째로 사라진다
        # (실측: 4/45 라인 소실). 라인 사이 노이즈 차단이 목적이므로 라인 스팬이면 충분.
        if vocal_regions is None:
            vocal_regions = [
                (float(s["start"]) - 0.15, float(s["end"]) + 0.15)
                for s in timestamps
                if s.get("start") is not None and s.get("end") is not None
            ]
        # 체인 스냅 대신 라인별 지배 옥타브 창 폴딩 — 라인 단위라 오염 전파가 없다.
        # precomputed_f0가 있으면 재추론 없이 주입값으로 트랙을 만든다 (정렬과 병렬 계산 결과).
        track = self.extract_f0(
            audio,
            vocals=vocals,
            vocal_regions=vocal_regions,
            apply_snap=False,
            precomputed=precomputed_f0,
        )
        # 디버그 오버레이용 RAW 곡선 — 폴딩 전에 캡처해야 모델 원본 거동이 보인다
        self.last_f0_curve = downsample_f0_curve(track)
        if self.config.octave_snap and vocal_regions:
            folded = fold_line_octaves(track, vocal_regions)
            if folded:
                logger.info(f"Per-line octave fold adjusted {folded} frames")
        kwargs = {
            "min_note_sec": self.config.min_note_sec,
            "max_gap_sec": self.config.max_gap_sec,
            "min_voiced_ratio": self.config.min_voiced_ratio,
        }
        count = 0
        for seg in timestamps:
            start, end = seg.get("start"), seg.get("end")
            if start is None or end is None or end <= start:
                continue
            notes: list[dict] = []
            # 독음 정렬 곡은 발음 음절 스팬(pron_segments)을 앵커로 우선한다 — 다음절 한자
            # (熱=ネツ)가 원문 글자 span으로는 노트 1개로 뭉치지만, 음절 스팬이면 음절마다
            # 별도 노트로 쪼개진다 (사용자 요구: 다음절 한자 노트 분할).
            words = seg.get("words")
            anchor_source = seg.get("pron_segments") or words
            if self.config.anchor_to_words and anchor_source:
                # 노트를 정렬된 음절 경계에서 자른다 — 가사 하이라이트와 타이밍 일치
                anchors = anchor_spans_from_words(anchor_source, float(end))
                notes = notes_from_anchor_spans(
                    track,
                    anchors,
                    min_note_sec=self.config.min_note_sec,
                    max_gap_sec=self.config.max_gap_sec,
                    min_voiced_ratio=self.config.min_voiced_ratio,
                )
                # 라인 창(14반음)은 서브하모닉(-12)과 실음이 공존할 수 있어,
                # 라인 노트 중앙값에서 9반음 초과 이탈만 ±12 접어 마무리한다
                _fold_notes_to_line_median(notes)
            if not notes:
                notes = notes_for_span(track, float(start), float(end), **kwargs)
            if notes:
                seg["notes"] = notes
                count += 1

        # 전역 저음 이상치 필터: 옥타브 스냅이 잘못된 저음 기준에 잠겨 전파되거나
        # 분리 잔여물(베이스/노이즈)이 만든, 곡 멜로디 대역에서 한참 벗어난 노트를 버린다.
        # 기준은 노트가 아니라 f0 트랙의 유성 프레임 중앙값 — 지속 시간으로 자연 가중되어
        # 온셋 파편 노트가 많아져도 오염되지 않는다.
        all_midis = [n["midi"] for s in timestamps for n in s.get("notes", [])]
        if all_midis and track.voiced.any():
            median = float(np.nanmedian(track.midi[track.voiced]))
            # -14: 벌스가 후렴보다 실제로 5~8반음 낮은 곡에서 진짜 노트를 먹지 않도록
            # 옥타브(-12) 이상 벗어난 것만 이상치로 취급한다
            floor = median - 14
            dropped = 0
            for seg in timestamps:
                notes = seg.get("notes")
                if not notes:
                    continue
                kept = [n for n in notes if n["midi"] >= floor]
                dropped += len(notes) - len(kept)
                if kept:
                    seg["notes"] = kept
                else:
                    del seg["notes"]
                    count -= 1
            if dropped:
                logger.info(f"Dropped {dropped} low-outlier notes (< median-10 = {floor})")

        # 곡 키 추정 + 스케일 기반 반음 타이브레이크 — 표시는 항상, 스냅은 상관이
        # 충분히 높을 때만 (조성이 약한 곡에서 엉뚱한 스케일로 노트를 옮기지 않게)
        self.last_key = None
        if self.config.key_detect:
            try:
                self.last_key = estimate_key(track)
                if self.last_key:
                    logger.info(
                        f"Estimated key: {self.last_key['name']} "
                        f"(r={self.last_key['confidence']})"
                    )
                    if self.config.key_snap and self.last_key["confidence"] >= 0.6:
                        snapped = snap_notes_to_key(timestamps, track, self.last_key)
                        if snapped:
                            logger.info(f"Key snap adjusted {snapped} boundary notes")
            except Exception:
                logger.exception("Key estimation failed; continuing without key")
                self.last_key = None
        return count


# 웜 캐시 싱글턴 (WS2-A) — 프로세스 수명 동안 MelodyExtractor(와 그 안에 lazy 로드된 f0
# 백엔드 모델)를 상주시킨다. 지연 생성이라 import만으로는 아무것도 로드하지 않는다.
_shared_extractor: "MelodyExtractor | None" = None
_shared_extractor_lock = threading.Lock()


def get_shared_extractor(config: MelodySettings | None = None) -> "MelodyExtractor":
    """웜 캐시된 MelodyExtractor를 돌려준다 (EVERYRIC_SERVER_WARM_MODELS 기준).

    _get_model이 로드한 f0 백엔드(RMVPE/FCPE)는 인스턴스에 상주하므로, 같은 추출기를 재사용하면
    두 번째 잡부터 f0 모델 재로드가 0회다. 재사용 시 "warm model reuse: melody" 1줄. warm이
    꺼져 있으면 매번 새 인스턴스(기존 동작). 잡별 상태(last_f0_curve/last_key)는 annotate가
    매번 덮어쓰므로 직렬 잡 처리(max_concurrent_jobs=1·순차 워커 루프) 전제에서 안전하다."""
    if not get_settings().server.warm_models:
        return MelodyExtractor(config)
    global _shared_extractor
    with _shared_extractor_lock:
        if _shared_extractor is None:
            _shared_extractor = MelodyExtractor(config)
        else:
            logger.info("warm model reuse: melody")
        return _shared_extractor


def clear_shared_extractor() -> None:
    """웜 캐시 해제 (VRAM 가드용) — 다음 요청에서 지연 재생성된다."""
    global _shared_extractor
    with _shared_extractor_lock:
        _shared_extractor = None
