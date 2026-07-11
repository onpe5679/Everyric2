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
            shift = 0.0
            if g - m >= global_guard and abs(m + 12 - g) < abs(m - g):
                shift = 12.0
            elif m - g >= global_guard and abs(m - 12 - g) < abs(m - g):
                shift = -12.0
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

    def extract_f0(
        self,
        audio: AudioData,
        vocals: AudioData | None = None,
        vocal_regions: list[tuple[float, float]] | None = None,
        apply_snap: bool | None = None,
    ) -> F0Track:
        """곡 전체에서 프레임 단위 f0 트랙을 뽑는다 (분리 옵션 적용 후 FCPE 1회 추론).

        vocals가 주어지면 이미 분리된 보컬 스템으로 간주하고 재분리를 건너뛴다
        (워커가 VAD용으로 분리한 스템을 재사용 — demucs 이중 실행 방지).
        vocal_regions(VAD 발성 구간)가 주어지면 구간 밖 프레임을 무성 처리한다 —
        분리 잔여 노이즈가 라인 사이를 '유성'으로 이어버리면 옥타브 스냅의 리셋이
        막혀 저음 기준이 라인 경계를 넘어 전파되는 실측 실패 모드를 차단한다.
        """
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

        duration = len(waveform) / MELODY_SAMPLE_RATE
        frame_dt = duration / max(1, len(f0))
        times = (np.arange(len(f0)) + 0.5) * frame_dt
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
    ) -> int:
        """정렬 결과(worker 포맷)의 각 세그먼트(라인)에 notes를 붙인다.

        노트는 라인 [start, end) 구간에서 피치 안정 run 단위로 분할되므로
        단어/글자 경계와 무관하게 멜리스마·이음도 자연스럽게 나뉜다.
        (CTC의 word_segments는 글자 단위 span이라 노트 산출에는 너무 짧다.)
        vocals: 호출부가 이미 분리해 둔 보컬 스템 (있으면 재분리 생략).
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
        # 체인 스냅 대신 라인별 지배 옥타브 창 폴딩 — 라인 단위라 오염 전파가 없다
        track = self.extract_f0(audio, vocals=vocals, vocal_regions=vocal_regions, apply_snap=False)
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
        return count
