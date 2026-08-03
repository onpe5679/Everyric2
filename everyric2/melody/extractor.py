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


OCTAVE_MIN_DEV = 8.0
OCTAVE_MAX_RESIDUAL = 1.5


def octave_fold_shift(
    dev: np.ndarray,
    *,
    min_dev: float = OCTAVE_MIN_DEV,
    max_residual: float = OCTAVE_MAX_RESIDUAL,
) -> np.ndarray:
    """기준 대비 편차(반음)를 되돌릴 ±12k 이동량 — 옥타브 인공물에만 반응한다.

    서브하모닉/배음 락온은 **정확히** ±12의 배수만큼 어긋난다 (f0 잡음 때문에 ±1반음
    정도 흔들릴 뿐). 반면 실제 멜로디의 넓은 도약은 16·19반음처럼 12의 배수가 아닌
    임의 음정이다. 그래서 "많이 벗어났으니 접는다"가 아니라 "12의 배수 근처로
    벗어났을 때만 접는다"가 옳은 판정이다 — 편차가 min_dev 이상이면서 가장 가까운
    12의 배수와의 잔차가 max_residual 이내일 때만 이동량을 낸다 (아니면 0).

    벤치 실측(scripts/bench_melody_ust.py, UST 12곡 7502노트): 잔차 게이팅 없는
    구버전은 잡음 없는 f0에서도 정답 노트 811개를 ±12/±24로 파괴했다 (음역이 넓은
    라인이 통째로 접힘). 잔차 게이팅 후 같은 조건에서 파괴 0.
    """
    dev = np.asarray(dev, dtype=np.float64)
    k = np.round(dev / 12.0)
    ok = (np.abs(dev) >= min_dev) & (k != 0) & (np.abs(dev - 12.0 * k) <= max_residual)
    return np.where(ok, -12.0 * k, 0.0)


def _side_medians(vals: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    """각 원소의 **직전 width개**와 **직후 width개** 중앙값을 따로 낸다 (끝은 edge 패딩).

    앞뒤를 합친 중앙창을 쓰면 진짜 도약 직후 프레임에서 창이 도약 이전 값에 지배돼
    기준이 지연된다 — 옥타브 도약이면 그 지연이 그대로 "12반음 이탈"로 보여 정답을
    파괴한다. 좌우를 분리해야 "도약(한쪽만 다름)"과 "이탈(양쪽 다 다름)"이 갈린다.
    """
    n = len(vals)
    width = max(1, min(width, n))
    pad = np.pad(vals, width, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(pad, width)
    # win[i]는 pad[i:i+width] — prev는 원소 i 직전 width개, next는 직후 width개
    prev = np.median(win[:n], axis=1)
    nxt = np.median(win[width + 1 : width + 1 + n], axis=1)
    return prev, nxt


def _agreed_octave_shift(
    vals: np.ndarray,
    prev_ref: np.ndarray,
    next_ref: np.ndarray,
    *,
    min_dev: float = OCTAVE_MIN_DEV,
) -> np.ndarray:
    """앞뒤 기준이 **둘 다** 같은 ±12k 이탈을 가리킬 때만 그 이동량을 낸다.

    서브하모닉 락온은 궤적에서 잠깐 벗어났다 **되돌아오는** 이탈이라 앞뒤 문맥 모두와
    옥타브만큼 어긋난다. 반대로 멜로디가 진짜로 한 옥타브 올라가 머무르면 뒤 문맥은
    새 음역과 일치하므로 판정이 갈려 접지 않는다. 이 비대칭이 둘을 가르는 유일한
    신호다 — 편차 크기만으로는 구분되지 않는다.
    """
    sp = octave_fold_shift(vals - prev_ref, min_dev=min_dev)
    sn = octave_fold_shift(vals - next_ref, min_dev=min_dev)
    return np.where((sp != 0.0) & (sp == sn), sp, 0.0)


def fold_line_octaves(
    track: F0Track,
    spans: list[tuple[float, float]],
    *,
    context_sec: float = 0.8,
    global_guard: float = 20.0,
    context_lines: int = 3,
) -> int:
    """라인 안의 국소 궤적에서 옥타브만큼 튄 프레임을 되돌린다. 반환: 접힌 프레임 수.

    프레임 체인 스냅(직전 프레임 기준 ±12 접기)은 리셋 직후 첫 프레임이
    서브하모닉이면 라인 전체가 저옥타브에 갇히고, 접힌 값이 다시 기준이 되어
    이중 폴딩(-24)까지 발생한다 (로키 벌스 실측: 라인 중앙값 59→26). 그렇다고
    라인의 지배 옥타브 창(고정 14반음)을 기준으로 삼으면 이번엔 **음역이 넓은 라인이
    통째로 망가진다** — 한 프레이즈가 2옥타브를 오가는 곡이 실제로 있다
    (numb numb: 라인 음역 23~26반음, 32라인 중 6라인).

    그래서 기준은 (1) 각 프레임의 **직전/직후 context_sec 중앙값 두 개**이고, 둘 다
    같은 ±12k 이탈을 가리킬 때만 접는다 (_agreed_octave_shift). 국소 기준은 넓은
    음역도 따라가고, 좌우 분리 덕에 "잠깐 튀었다 돌아오는 락온"과 "올라가서 머무르는
    진짜 도약"이 갈린다. 이동량은 octave_fold_shift의 잔차 게이팅을 거쳐 12의 배수
    근처 이탈에만 적용된다.
    (2) 라인 전체가 잠긴 경우는 라인 안에 기준이 없으므로 라인 중앙값을 **앞뒤
    context_lines개 이웃 라인 + 전곡 중앙값**과 비교해 셋 다 같은 판정일 때만 라인을
    통째로 이동한다. 다만 이 판정은 global_guard=20반음부터, 즉 **2옥타브 이상 이탈
    (이중 폴딩 인공물)에만** 건다 — 한 옥타브짜리 라인 이탈은 실제로 옥타브 낮게
    부르는 벌스와 음고 증거만으로는 구분이 불가능하다는 것이 벤치 결론이다
    (1옥타브까지 걸면 구제 0 · 파괴 162노트). 라인 통째 락온의 실제 복구는 f0 백엔드의
    salience 같은 별도 증거가 필요하며 이 계층의 문제가 아니다.
    """
    folded = 0
    line_medians: list[float | None] = []
    span_indices: list[np.ndarray] = []
    frame_dt = float(track.times[1] - track.times[0]) if len(track.times) > 1 else 0.01
    half = max(1, int(round(context_sec / max(frame_dt, 1e-6))))
    for s, e in spans:
        mask = (track.times >= s) & (track.times < e) & track.voiced
        idx = np.where(mask)[0]
        span_indices.append(idx)
        if len(idx) < 5:
            line_medians.append(None)
            continue
        vals = track.midi[idx]
        prev_ref, next_ref = _side_medians(vals, half)
        shift = _agreed_octave_shift(vals, prev_ref, next_ref)
        hit = shift != 0.0
        if hit.any():
            track.midi[idx[hit]] += shift[hit]
            folded += int(hit.sum())
        line_medians.append(float(np.nanmedian(track.midi[idx])))

    # 라인 통째 락온: 라인 안에는 기준이 없으니 **이웃 라인**을 문맥으로 쓴다.
    # 전곡 중앙값 하나를 기준으로 삼으면 실제로 옥타브 낮게 부르는 벌스 구간이
    # 통째로 끌어올려진다 (벤치 실측: 그 하나 때문에 정답 185노트 파괴 — numb numb·
    # みむかｩわ처럼 곡 음역이 2옥타브를 넘는 곡에서 집중 발생).
    valid_idx = [i for i, m in enumerate(line_medians) if m is not None]
    if valid_idx:
        song_ref = float(np.median([line_medians[j] for j in valid_idx]))
    for pos, i in enumerate(valid_idx):
        idx = span_indices[i]
        if len(idx) == 0:
            continue
        before = [line_medians[j] for j in valid_idx[max(0, pos - context_lines) : pos]]
        after = [line_medians[j] for j in valid_idx[pos + 1 : pos + 1 + context_lines]]
        if not before and not after:
            continue
        # 첫/마지막 라인은 한쪽 문맥밖에 없으므로 그쪽만으로 판정한다
        prev_ref = float(np.median(before or after))
        next_ref = float(np.median(after or before))
        cur = np.array([float(line_medians[i])])
        shift = float(
            _agreed_octave_shift(
                cur, np.array([prev_ref]), np.array([next_ref]), min_dev=global_guard
            )[0]
        )
        # 이웃 두 쪽에 더해 **전곡 기준**까지 같은 판정이어야 라인을 통째로 옮긴다.
        # 높은 후렴 사이에 낀 진짜 저음 라인은 이웃과의 차가 우연히 12에 걸릴 수 있지만
        # 전곡 중앙값과는 12의 배수로 떨어지지 않는다 (실측: 花めかない 162노트 오폴딩).
        if shift and shift != float(
            octave_fold_shift(cur - song_ref, min_dev=global_guard)[0]
        ):
            shift = 0.0
        if shift:
            track.midi[idx] += shift
            folded += len(idx)
    return folded


def _span_pitch(track: F0Track, i0: int, i1: int) -> tuple[int, float, int] | None:
    """프레임 구간 [i0, i1)의 대표 반음 → (midi, confidence, 유성 프레임 수).

    최빈 반음 = 구간에서 가장 오래 유지된 음 (모음 정상 상태). 중앙값은 음절 시작부
    (자음/브레시 온셋)의 서브하모닉 프레임에 쉽게 오염된다.
    """
    if i1 <= i0:
        return None
    voiced = track.voiced[i0:i1]
    n_voiced = int(voiced.sum())
    if n_voiced < 3:
        return None
    rounded = np.round(track.midi[i0:i1][voiced]).astype(int)
    values, counts = np.unique(rounded, return_counts=True)
    midi = int(values[np.argmax(counts)])
    conf = float(counts.max()) / max(1, i1 - i0)
    return midi, conf, n_voiced


def _refine_boundary(
    track: F0Track, i0: int, ib: int, i1: int, pa: int, pb: int, window: int
) -> int:
    """앵커 경계 ib를 두 음(pa→pb)이 실제로 갈리는 f0 전이점으로 옮긴다.

    정렬(CTC) 앵커는 자음 길이·발음 사전 때문에 실제 음 전환보다 수십 ms 앞뒤로
    어긋난다. f0에는 전환 지점이 그대로 남아 있으므로, 경계 후보를 창 안에서 훑어
    "앞은 pa에 가깝고 뒤는 pb에 가까운" 프레임 수가 최대가 되는 지점을 고른다.
    창(window) 밖으로는 절대 나가지 않아 가사-노트 잠금은 유지된다.
    """
    lo = max(i0 + 1, ib - window)
    hi = min(i1 - 1, ib + window)
    if hi <= lo:
        return ib
    seg = track.midi[lo:hi]
    ok = track.voiced[lo:hi] & np.isfinite(seg)
    # near_a[i] = 그 프레임이 pb보다 pa에 가까운가
    near_a = np.zeros(len(seg), dtype=np.int32)
    near_a[ok] = (np.abs(seg[ok] - pa) <= np.abs(seg[ok] - pb)).astype(np.int32)
    voiced_n = ok.astype(np.int32)
    # 분할점 s(로컬 인덱스)에서의 점수 = [0,s)에서 pa에 가까운 수 + [s,end)에서 pb에 가까운 수
    pre_a = np.concatenate(([0], np.cumsum(near_a)))
    pre_v = np.concatenate(([0], np.cumsum(voiced_n)))
    total_v = int(pre_v[-1])
    if total_v < 4:
        return ib
    scores = pre_a + (total_v - pre_v) - (pre_a[-1] - pre_a)
    best = int(np.argmax(scores))
    # 원 경계보다 확실히 나을 때만 옮긴다 (증거가 팽팽하면 정렬 결과를 존중)
    if scores[best] - scores[ib - lo] < 2:
        return ib
    return lo + best


def notes_from_anchor_spans(
    track: F0Track,
    anchors: list[tuple[float, float]],
    *,
    min_note_sec: float = 0.08,
    max_gap_sec: float = 0.12,
    min_voiced_ratio: float = 0.15,
    long_span_sec: float = 1.0,
    boundary_snap_sec: float = 0.06,
    trim_tail: bool = True,
) -> list[dict]:
    """정렬된 음절(글자) 앵커 경계에서 노트를 자른다 — 노트 타이밍이 가사와 잠긴다.

    자유 f0 안정 run 분할은 리듬이 가사 하이라이트와 따로 놀았다. 여기서는 각 앵커
    [start, end) 구간에서 가장 오래 유지된 반음을 그 음절의 노트로 삼는다.
    같은 음이 이어져도 음절마다 별도 노트를 유지한다 — 노래방 악보처럼 음절 단위
    리듬이 보여야 하기 때문 (병합하면 통짜 긴 막대가 되어 리듬 정보가 사라진다).
    길게 끄는 음절(멜리스마, long_span_sec 초과)만 내부 run 분할을 허용하되
    첫 노트 시작은 앵커 시작에 스냅한다.

    앵커 경계를 그대로 쓰면 노트 경계 정확도가 정렬 정확도에 그대로 묶인다. 그래서
    (1) 음이 바뀌는 인접 앵커 사이 경계는 boundary_snap_sec 안에서 f0 전이점으로
    미세 조정하고, (2) 앵커 끝이 발성보다 길게 잡힌 경우(다음 글자까지 확장하다
    쉼표를 넘은 경우) 노트 끝을 마지막 유성 프레임까지 줄인다. 둘 다 창 안 조정이라
    가사-노트 잠금은 유지된다.
    """
    times = track.times
    n_frames = len(times)
    if n_frames == 0:
        return []
    frame_dt = float(times[1] - times[0]) if n_frames > 1 else 0.01
    snap_w = max(0, int(round(boundary_snap_sec / max(frame_dt, 1e-6))))

    # 1패스: 앵커별 프레임 구간과 대표 반음 (경계 조정 전)
    spans: list[dict] = []
    for a0, a1 in anchors:
        if a1 <= a0:
            continue
        i0 = int(np.searchsorted(times, a0, side="left"))
        i1 = int(np.searchsorted(times, a1, side="left"))
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
                spans.append({"melisma": sub, "i0": i0, "i1": i1})
                continue
        pitch = _span_pitch(track, i0, i1)
        if pitch is None:
            continue
        spans.append({"midi": pitch[0], "conf": pitch[1], "i0": i0, "i1": i1, "a0": a0, "a1": a1})

    # 2패스: 음이 바뀌는 인접 앵커의 공유 경계를 f0 전이점으로 옮긴다
    if snap_w > 0:
        for prev, cur in zip(spans, spans[1:]):
            if "melisma" in prev or "melisma" in cur:
                continue
            if prev["i1"] != cur["i0"] or prev["midi"] == cur["midi"]:
                continue  # 붙어 있지 않거나(쉼표 사이) 같은 음이면 옮길 근거가 없다
            nb = _refine_boundary(
                track, prev["i0"], cur["i0"], cur["i1"], prev["midi"], cur["midi"], snap_w
            )
            if nb == cur["i0"]:
                continue
            prev["i1"] = cur["i0"] = nb
            prev["a1"] = cur["a0"] = float(times[nb])

    # 3패스: 조정된 구간으로 반음 재계산 + 발성 끝으로 트림
    raw: list[dict] = []
    for sp in spans:
        if "melisma" in sp:
            raw.extend(sp["melisma"])
            continue
        pitch = _span_pitch(track, sp["i0"], sp["i1"])
        if pitch is None:
            continue
        midi, conf, _ = pitch
        start, end = float(sp["a0"]), float(sp["a1"])
        if trim_tail:
            # 앵커 끝은 다음 글자 시작까지 늘어나 있어 쉼표·간주를 넘길 수 있다.
            # 마지막 유성 프레임까지 줄이면 노트 막대가 실제 발성 길이와 맞는다.
            voiced_idx = np.flatnonzero(track.voiced[sp["i0"] : sp["i1"]])
            if len(voiced_idx):
                last = sp["i0"] + int(voiced_idx[-1])
                end = min(end, float(times[last]) + frame_dt)
        if end <= start:
            end = start + frame_dt
        raw.append(
            {
                "midi": midi,
                "start": round(start, 3),
                "end": round(end, 3),
                "confidence": round(conf, 3),
            }
        )
    return raw


def _fold_notes_to_local_median(notes: list[dict], *, context: int = 3) -> None:
    """프레임 폴딩이 놓친 옥타브 이탈 노트를 이웃 노트 문맥으로 마무리한다 (제자리 수정).

    기준은 라인 전체 중앙값이 아니라 **자기를 뺀 앞 context개 / 뒤 context개 중앙값
    두 개**이고, 둘 다 같은 ±12k 이탈을 가리킬 때만 접는다. 라인 중앙값을 쓰면 음역이
    넓은 라인(한 프레이즈가 2옥타브를 오가는 곡이 실제로 있다)에서 양 끝 노트가 서로를
    이상치로 만들고, 좌우를 합치면 옥타브 도약 직후 노트가 이탈로 오판된다.
    """
    if len(notes) < 3:
        return
    midis = [n["midi"] for n in notes]
    for i, n in enumerate(notes):
        before = midis[max(0, i - context) : i]
        after = midis[i + 1 : i + 1 + context]
        if not before or not after:
            continue
        cur = np.array([float(n["midi"])])
        shift = float(
            _agreed_octave_shift(
                cur, np.array([float(np.median(before))]), np.array([float(np.median(after))])
            )[0]
        )
        if shift:
            n["midi"] = int(n["midi"] + shift)


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
        vocal_regions 마스킹·옥타브 스냅 같은 정렬 의존 후처리는 여기서 하지 않는다.

        긴 오디오는 겹침 청크로 나눠 청크별 추론 후 f0 배열을 시간축으로 스티칭한다 —
        f0 추론은 CTC 정렬과 병렬로 GPU를 써 활성 피크가 합쳐지므로(WS2-B), 통짜 forward의
        길이 비례 활성값이 긴 곡 OOM에 기여한다(실사고 2026-07-24). 단일 청크(짧은 곡·비활성)는
        통짜와 완전히 동일하다 (MelodySettings.chunk_sec)."""
        import librosa

        audio = vocals if vocals is not None else self._maybe_separate(audio)
        waveform = audio.waveform
        if audio.sample_rate != MELODY_SAMPLE_RATE:
            waveform = librosa.resample(
                waveform, orig_sr=audio.sample_rate, target_sr=MELODY_SAMPLE_RATE
            )
        waveform = np.ascontiguousarray(waveform, dtype=np.float32)

        from everyric2.audio.chunking import plan_chunk_windows, stitch_chunk_outputs

        n = len(waveform)
        chunk_sec = getattr(self.config, "chunk_sec", 0.0) or 0.0
        overlap_sec = getattr(self.config, "chunk_overlap_sec", 5.0) or 0.0
        windows = plan_chunk_windows(
            n, int(chunk_sec * MELODY_SAMPLE_RATE), int(overlap_sec * MELODY_SAMPLE_RATE)
        )
        if len(windows) == 1:
            f0 = self._infer_f0_chunk(waveform)
        else:
            pieces = [
                self._infer_f0_chunk(np.ascontiguousarray(waveform[s:e])) for s, e in windows
            ]
            f0 = stitch_chunk_outputs(pieces, windows, n, frame_axis=0)

        f0 = np.asarray(f0, dtype=np.float64)
        duration = len(waveform) / MELODY_SAMPLE_RATE
        frame_dt = duration / max(1, len(f0))
        times = (np.arange(len(f0)) + 0.5) * frame_dt
        return f0, times

    def _infer_f0_chunk(self, waveform: np.ndarray) -> np.ndarray:
        """단일 파형 청크(16kHz mono np.ndarray) → 백엔드 f0(Hz) 배열 (10ms hop, unvoiced=0)."""
        import torch

        model = self._get_model()
        if self._backend == "rmvpe":
            return np.asarray(
                model.infer(waveform, threshold=self.config.rmvpe_threshold), dtype=np.float64
            )
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
        return f0.squeeze().cpu().numpy().astype(np.float64)

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
                # 프레임 폴딩이 놓친 잔여 옥타브 이탈을 노트 레벨에서 마무리한다
                # (이웃 노트 기준, 12의 배수 근처 이탈만)
                _fold_notes_to_local_median(notes)
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
