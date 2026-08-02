"""UST 준정답 대비 노트 양자화 정확도 벤치 (오디오 무의존, 순수 계산).

목적: 가라오케 노트의 **음고와 노트 경계**가 UST 채보와 얼마나 일치하는지 재는 것.
표기(가사)는 여기서 재지 않는다 — UST 가사는 채보자마다 달라 불변이 아니다.

동작:
  1. UST/USTX에서 정답 노트열(MIDI 음고 + 시작/끝)을 읽는다.
  2. 정답 노트열에서 프레임 f0 곡선을 **역생성**한다 (포르타멘토·비브라토·프레임
     지터·자음 무성 구간·서브하모닉 락온·전역 디튠을 옵션으로 주입).
  3. 그 f0를 `MelodyExtractor.annotate_timestamps(precomputed_f0=...)`에 그대로
     주입해 **현행 프로덕션 양자화 경로 전체**(라인별 옥타브 폴딩 → 앵커 노트 →
     라인 중앙값 폴딩 → 저음 이상치 필터 → 키 스냅)를 통과시킨다.
  4. 산출 노트열과 정답 노트열을 노트 단위로 매칭해 일치율과 오류 유형을 낸다.

f0 추출(RMVPE/FCPE) 자체는 재지 않는다 — 그건 오디오가 필요한 서버 작업이다.
여기서 재는 것은 "f0가 이 정도 품질로 들어왔을 때 양자화가 정답을 얼마나 보존하는가"다.

실행 예:
  PYTHONPATH=. python scripts/bench_melody_ust.py --profile realistic
  PYTHONPATH=. python scripts/bench_melody_ust.py --profile all --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from everyric2.config.settings import MelodySettings  # noqa: E402
from everyric2.melody.extractor import MelodyExtractor  # noqa: E402

USTS_ROOT = REPO_ROOT / "benchmark" / "usts"
FRAME_DT = 0.01  # 프로덕션 f0 백엔드와 동일한 10ms hop
PHRASE_GAP = 0.35  # 이 이상 쉬면 프레이즈(라인) 경계 — karaoke_review와 동일 기준

# 쉼표/브레스 토큰. 가사 필터가 아니라 **발성 여부** 판정용이라 karaoke_review의
# _ust_lyric보다 좁다 — 영어 가사(ASCII) 노트도 음고 정답으로 살려야 한다.
_REST_TOKENS = {"r", "br", "v", "cl", "sil", "pau", "vf", "", "-", "+", "aa", "息", "R息"}
_PITCH_SUFFIX = re.compile(r"[A-G]#?-?\d+$")
# 연장 표기 — 앞 음절을 끄는 노트라 정렬기가 별도 앵커를 만들지 못한다(멜리스마)
_CONTINUATION = {"ー", "-", "+", "ー ", "*"}


@dataclass
class TruthNote:
    midi: int
    start: float
    end: float
    lyric: str
    continuation: bool = False


@dataclass
class SongSpec:
    label: str
    path: str
    kind: str = "ust"  # ust | ustx
    track: str = "lead"


# benchmark/usts/INVENTORY.md의 "즉시 투입 가능" 판정 곡 — 제목·시간범위 정합이
# 확인됐고 파싱이 정상인 것만. 커버리지 결손·인코딩 파손 곡은 제외한다.
SONGS: list[SongSpec] = [
    SongSpec("deep_sea_girl", "Deep Sea Girl CV UST/Deep sea girl.ust"),
    SongSpec("nekomimi_archive", "Nekomimi Archive UST/NekomimiArchive_main.ust"),
    SongSpec("pseudo_hope", "Pseudo-Hope Syndrome USTs by Levin/main.ust"),
    SongSpec("teikoku_shojo", "TeikokuShojo_ust/TeikokuShojo.ust"),
    SongSpec("kikuo_yurushite", "[2023-02-03] 君が死んでも許してあげるよ - Kikuo/"
             "君が死んでも許してあげるよ main.ust"),
    SongSpec("dame_ningen", "dame ningen da/main.ust"),
    SongSpec("hanamekanai", "花めかない UST/main.ust"),
    SongSpec("alien_alien", "Alien Alien UST by Zansatsu/CV/Alien Alien Main.ust"),
    SongSpec("numb_numb", "numb numb USTX by spayde-173P/ustx/numb numb untuned.ustx", "ustx"),
    SongSpec("rookie", "rookie/rookie.ustx", "ustx"),
    SongSpec("mimukawa", "みむかｩわナイストライ by さっぱりあんずジャム/みむかｩわナイストライ.ustx",
             "ustx"),
    SongSpec("inasena_girl", "君はいなせなガール ust/IA.ust"),
]


# ---------------------------------------------------------------- UST 파싱


def _is_rest(raw: str) -> bool:
    clean = _PITCH_SUFFIX.sub("", str(raw).strip()).strip()
    return clean.lower() in _REST_TOKENS


def _is_continuation(raw: str) -> bool:
    return _PITCH_SUFFIX.sub("", str(raw).strip()).strip() in _CONTINUATION


def parse_ust_notes(path: Path) -> list[TruthNote]:
    """UTAU .ust → 정답 노트열. karaoke_review의 파서와 달리 NoteNum(음고)을 살린다."""
    raw = path.read_bytes()
    text = None
    for enc in ("cp932", "utf-8-sig", "utf-8"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise ValueError(f"UST decode failed: {path}")

    tempo = 120.0
    for line in text.splitlines():
        if line.startswith("Tempo="):
            try:
                tempo = float(line.split("=", 1)[1].replace(",", "."))
            except ValueError:
                pass
            break

    entries: list[dict[str, str]] = []
    cur: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            entries.append(cur)
            cur = {"_section": line[1:-1]}
        elif "=" in line:
            key, _, value = line.partition("=")
            cur[key] = value
    entries.append(cur)

    notes: list[TruthNote] = []
    t = 0.0
    for cur in entries:
        section = cur.get("_section", "")
        if not section.startswith("#") or section in ("#SETTING", "#VERSION", "#TRACKEND"):
            continue
        if "Tempo" in cur:
            try:
                tempo = float(cur["Tempo"].replace(",", "."))
            except ValueError:
                pass
        try:
            length = int(cur.get("Length", "0"))
        except ValueError:
            length = 0
        dur = length * 60.0 / (tempo * 480)
        lyric = cur.get("Lyric", "")
        try:
            midi = int(cur.get("NoteNum", "-1"))
        except ValueError:
            midi = -1
        if not _is_rest(lyric) and 0 <= midi <= 127 and dur > 0:
            notes.append(
                TruthNote(midi, t, t + dur, lyric.strip(), _is_continuation(lyric))
            )
        t += dur
    return notes


def parse_ustx_notes(path: Path, track: str = "lead") -> list[TruthNote]:
    """OpenUtau .ustx → 정답 노트열 (lead 파트 합집합). 템포맵·파트 오프셋 반영."""
    import yaml

    doc = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    resolution = float(doc.get("resolution") or 480)
    tempos = sorted(
        (int(t.get("position") or 0), float(t.get("bpm") or doc.get("bpm") or 120))
        for t in (doc.get("tempos") or [{"position": 0, "bpm": doc.get("bpm") or 120}])
    )

    def tick_to_sec(tick: float) -> float:
        sec = 0.0
        for i, (pos, bpm) in enumerate(tempos):
            nxt = tempos[i + 1][0] if i + 1 < len(tempos) else None
            if nxt is None or tick <= nxt:
                return sec + (tick - pos) * 60.0 / (bpm * resolution)
            sec += (nxt - pos) * 60.0 / (bpm * resolution)
        return sec

    notes: list[TruthNote] = []
    for part in doc.get("voice_parts") or []:
        name = str(part.get("name") or "").lower()
        label = "harm" if "harm" in name else "lead"
        if label != track:
            continue
        base = int(part.get("position") or 0)
        for note in part.get("notes") or []:
            lyric = str(note.get("lyric") or "")
            if _is_rest(lyric):
                continue
            try:
                midi = int(note.get("tone"))
            except (TypeError, ValueError):
                continue
            tick = base + int(note.get("position") or 0)
            start = tick_to_sec(tick)
            end = tick_to_sec(tick + int(note.get("duration") or 0))
            if end > start and 0 <= midi <= 127:
                notes.append(TruthNote(midi, start, end, lyric.strip(), _is_continuation(lyric)))
    notes.sort(key=lambda n: (n.start, n.end))
    # 같은 시각에 겹치는 노트(파트 내부 하모니)는 최고음만 남긴다 — 리드 멜로디 단선율 가정
    dedup: list[TruthNote] = []
    for n in notes:
        if dedup and n.start < dedup[-1].end - 1e-6:
            if n.midi > dedup[-1].midi:
                dedup[-1] = n
            continue
        dedup.append(n)
    return dedup


def load_truth(spec: SongSpec) -> list[TruthNote]:
    path = USTS_ROOT / spec.path
    if not path.exists():
        raise FileNotFoundError(str(path))
    if spec.kind == "ustx":
        return parse_ustx_notes(path, spec.track)
    return parse_ust_notes(path)


# ------------------------------------------------------------ f0 역생성


@dataclass
class SynthProfile:
    """정답 노트열 → f0 곡선 역생성 파라미터. 값이 클수록 가혹한 조건."""

    name: str
    detune_cents: float = 0.0  # 곡 전체 튜닝 이탈 (A440 미준수·플랫 창법)
    portamento_ms: float = 0.0  # 음 사이 글라이드 폭 (경계 중심)
    vibrato_semitones: float = 0.0  # 비브라토 진폭 (peak)
    vibrato_rate_hz: float = 5.5
    vibrato_min_note: float = 0.35  # 이보다 긴 노트에만 비브라토
    jitter_semitones: float = 0.0  # 프레임 백색잡음 표준편차
    drift_semitones: float = 0.0  # 저주파 피치 흔들림 진폭
    onset_unvoiced_ms: float = 0.0  # 자음 구간 무성 처리 (노트 시작)
    dropout_rate: float = 0.0  # 노트 중간 무성 결손 확률
    octave_error_rate: float = 0.0  # 서브하모닉(-12) 락온 구간 비율
    line_lock_rate: float = 0.0  # 프레이즈 통째가 서브하모닉에 잠기는 비율
    anchor_jitter_ms: float = 0.0  # 정렬(CTC) 앵커 시작 지터 표준편차
    merge_continuation: bool = False  # 연장 노트를 앞 앵커에 흡수 (멜리스마 재현)


PROFILES: dict[str, SynthProfile] = {
    # 양자화 로직 자체의 무결성 확인용 — 여기서 100%가 아니면 순수 로직 결함이다
    "clean": SynthProfile("clean"),
    # 실제 가창의 기본 성질만 (글라이드·비브라토·약한 지터)
    "sung": SynthProfile(
        "sung",
        portamento_ms=60.0,
        vibrato_semitones=0.35,
        jitter_semitones=0.06,
        drift_semitones=0.08,
        onset_unvoiced_ms=25.0,
    ),
    # 프로덕션 실측 조건 근사: 디튠 + 정렬 지터 + 멜리스마 + 약한 옥타브 락온
    "realistic": SynthProfile(
        "realistic",
        detune_cents=-28.0,
        portamento_ms=70.0,
        vibrato_semitones=0.45,
        jitter_semitones=0.10,
        drift_semitones=0.12,
        onset_unvoiced_ms=35.0,
        dropout_rate=0.03,
        octave_error_rate=0.02,
        anchor_jitter_ms=25.0,
        merge_continuation=True,
    ),
    # 열악 조건 (합성보컬·분리 잔여물)
    "hard": SynthProfile(
        "hard",
        detune_cents=42.0,
        portamento_ms=110.0,
        vibrato_semitones=0.7,
        jitter_semitones=0.18,
        drift_semitones=0.2,
        onset_unvoiced_ms=45.0,
        dropout_rate=0.07,
        octave_error_rate=0.06,
        anchor_jitter_ms=45.0,
        merge_continuation=True,
    ),
}


def synth_f0(
    truth: list[TruthNote], profile: SynthProfile, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """정답 노트열 → (f0_hz, times). unvoiced는 0Hz — 프로덕션 f0 배열과 동일 규약."""
    if not truth:
        return np.zeros(0), np.zeros(0)
    duration = truth[-1].end + 1.0
    n = int(math.ceil(duration / FRAME_DT))
    times = (np.arange(n) + 0.5) * FRAME_DT
    midi = np.full(n, np.nan)

    detune = profile.detune_cents / 100.0
    for note in truth:
        i0 = int(np.searchsorted(times, note.start))
        i1 = int(np.searchsorted(times, note.end))
        if i1 <= i0:
            i1 = min(n, i0 + 1)
        midi[i0:i1] = note.midi + detune

    # 포르타멘토: 인접(붙어 있는) 노트 경계를 중심으로 선형 글라이드
    if profile.portamento_ms > 0:
        half = profile.portamento_ms / 2000.0
        for prev, nxt in zip(truth, truth[1:]):
            if nxt.start - prev.end > 0.03 or prev.midi == nxt.midi:
                continue
            b = nxt.start
            i0 = int(np.searchsorted(times, b - half))
            i1 = int(np.searchsorted(times, b + half))
            if i1 <= i0:
                continue
            ramp = np.linspace(0.0, 1.0, i1 - i0)
            midi[i0:i1] = (prev.midi + detune) * (1 - ramp) + (nxt.midi + detune) * ramp

    # 비브라토: 긴 노트의 후반부에만 (실제 창법과 동일하게 어택 직후엔 없다)
    if profile.vibrato_semitones > 0:
        for note in truth:
            if note.end - note.start < profile.vibrato_min_note:
                continue
            onset = note.start + (note.end - note.start) * 0.4
            i0 = int(np.searchsorted(times, onset))
            i1 = int(np.searchsorted(times, note.end))
            if i1 <= i0:
                continue
            phase = rng.uniform(0, 2 * np.pi)
            t = times[i0:i1] - onset
            midi[i0:i1] += profile.vibrato_semitones * np.sin(
                2 * np.pi * profile.vibrato_rate_hz * t + phase
            )

    if profile.drift_semitones > 0:
        walk = rng.normal(0.0, 1.0, n)
        k = 101
        kernel = np.ones(k) / k
        smooth = np.convolve(walk, kernel, mode="same")
        denom = np.std(smooth)
        if denom > 0:
            midi += profile.drift_semitones * smooth / denom

    if profile.jitter_semitones > 0:
        midi += rng.normal(0.0, profile.jitter_semitones, n)

    voiced = np.isfinite(midi)

    # 자음 어택: 노트 시작 몇 프레임을 무성 처리
    if profile.onset_unvoiced_ms > 0:
        k = max(1, int(round(profile.onset_unvoiced_ms / 1000.0 / FRAME_DT)))
        for note in truth:
            i0 = int(np.searchsorted(times, note.start))
            voiced[i0 : i0 + k] = False

    # 노트 중간 결손 (분리 잔여물·숨)
    if profile.dropout_rate > 0:
        for note in truth:
            if rng.random() >= profile.dropout_rate:
                continue
            i0 = int(np.searchsorted(times, note.start))
            i1 = int(np.searchsorted(times, note.end))
            if i1 - i0 < 6:
                continue
            c = rng.integers(i0 + 2, max(i0 + 3, i1 - 2))
            voiced[c : c + 3] = False

    # 서브하모닉 락온: **노트 경계와 무관한** 시간 구간을 -12로 잠근다.
    # 실제 f0 트래커의 락온은 노트 단위로 딱 떨어지지 않고 0.1~1.5초짜리 구간이
    # 음절 중간에서 시작해 중간에서 풀린다 — 노트 단위로 주입하면 앵커 최빈값
    # 투표가 절대 못 고치는 최악만 재현돼 폴딩의 가치가 과대평가된다.
    if profile.octave_error_rate > 0:
        span_total = truth[-1].end - truth[0].start
        locked = 0.0
        target = span_total * profile.octave_error_rate
        while locked < target:
            dur = float(rng.uniform(0.1, 1.5))
            t0 = float(rng.uniform(truth[0].start, truth[-1].end))
            i0 = int(np.searchsorted(times, t0))
            i1 = int(np.searchsorted(times, t0 + dur))
            midi[i0:i1] -= 12.0
            locked += dur

    # 프레이즈 통째 락온: 조용한 벌스 한 줄이 통째로 서브하모닉에 잠기는 실패 모드.
    # 라인 안에는 기준이 없어 프레임/노트 문맥으로는 못 고친다 — 전역 가드의 존재 이유라
    # 벤치에 이 모드가 없으면 가드의 가치를 0으로 오판하게 된다.
    if profile.line_lock_rate > 0:
        phrases: list[list[TruthNote]] = []
        for note in truth:
            if phrases and note.start - phrases[-1][-1].end <= PHRASE_GAP:
                phrases[-1].append(note)
            else:
                phrases.append([note])
        for phrase in phrases:
            if rng.random() >= profile.line_lock_rate:
                continue
            i0 = int(np.searchsorted(times, phrase[0].start))
            i1 = int(np.searchsorted(times, phrase[-1].end))
            midi[i0:i1] -= 12.0

    hz = np.where(voiced & np.isfinite(midi), 440.0 * np.power(2.0, (midi - 69.0) / 12.0), 0.0)
    return np.nan_to_num(hz, nan=0.0), times


# ------------------------------------------------- 정답 → 라인/앵커 (정렬 결과 모사)


def build_segments(
    truth: list[TruthNote], profile: SynthProfile, rng: np.random.Generator
) -> list[dict]:
    """정답 노트열 → worker 포맷 timestamps (라인 + 글자 앵커).

    실제로는 CTC 정렬이 만드는 입력이다. 여기서는 정답 경계에 지터를 준 것으로 모사하며,
    연장 표기(ー 등) 노트는 정렬기가 별도 앵커를 만들지 못하므로 앞 앵커에 흡수시킨다
    (merge_continuation) — 한 앵커가 여러 정답 노트를 덮는 멜리스마 상황 재현.
    """
    phrases: list[list[TruthNote]] = []
    for note in truth:
        if phrases and note.start - phrases[-1][-1].end <= PHRASE_GAP:
            phrases[-1].append(note)
        else:
            phrases.append([note])

    segments: list[dict] = []
    for phrase in phrases:
        words: list[dict] = []
        for note in phrase:
            if (
                profile.merge_continuation
                and note.continuation
                and words
            ):
                continue  # 앞 글자에 흡수 — 앵커는 다음 글자 시작까지 자동 확장된다
            s = note.start
            if profile.anchor_jitter_ms > 0:
                s += float(rng.normal(0.0, profile.anchor_jitter_ms / 1000.0))
            s = max(0.0, s)
            words.append({"word": note.lyric or "x", "start": round(s, 3),
                          "end": round(s + 0.05, 3)})
        if not words:
            continue
        words.sort(key=lambda w: w["start"])
        segments.append(
            {
                "text": "".join(w["word"] for w in words),
                "start": round(min(words[0]["start"], phrase[0].start), 3),
                "end": round(phrase[-1].end, 3),
                "words": words,
            }
        )
    return segments


# ---------------------------------------------------------------- 채점


@dataclass
class Score:
    truth_n: int = 0
    pred_n: int = 0
    matched: int = 0  # 온셋 ±tol AND 음고 ±tol
    onset_ok_pitch_off: int = 0  # 온셋은 맞았는데 음고가 틀림 (양자화 오류)
    onset_ok_octave: int = 0  # 그 중 옥타브급 이탈
    onset_off: int = 0  # 겹치는 노트는 있으나 온셋이 허용 밖 (경계 지터)
    missed: int = 0  # 대응 노트 없음 (병합·소실)
    spurious: int = 0  # 정답에 없는 산출 노트 (과분할)
    offset_ok: int = 0  # 매칭 노트 중 끝 경계도 ±tol
    abs_pitch_err: list[float] = field(default_factory=list)
    onset_err: list[float] = field(default_factory=list)

    def merge(self, other: "Score") -> None:
        self.truth_n += other.truth_n
        self.pred_n += other.pred_n
        self.matched += other.matched
        self.onset_ok_pitch_off += other.onset_ok_pitch_off
        self.onset_ok_octave += other.onset_ok_octave
        self.onset_off += other.onset_off
        self.missed += other.missed
        self.spurious += other.spurious
        self.offset_ok += other.offset_ok
        self.abs_pitch_err.extend(other.abs_pitch_err)
        self.onset_err.extend(other.onset_err)

    @property
    def recall(self) -> float:
        return self.matched / self.truth_n if self.truth_n else 0.0

    @property
    def precision(self) -> float:
        return self.matched / self.pred_n if self.pred_n else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def as_dict(self) -> dict:
        return {
            "truth_n": self.truth_n,
            "pred_n": self.pred_n,
            "matched": self.matched,
            "recall": round(self.recall, 4),
            "precision": round(self.precision, 4),
            "f1": round(self.f1, 4),
            "offset_ok_rate": round(self.offset_ok / self.matched, 4) if self.matched else 0.0,
            "err_pitch_off": self.onset_ok_pitch_off,
            "err_octave": self.onset_ok_octave,
            "err_onset_off": self.onset_off,
            "err_missed": self.missed,
            "err_spurious": self.spurious,
            "median_abs_pitch_err": round(
                float(np.median(self.abs_pitch_err)), 3
            ) if self.abs_pitch_err else 0.0,
            "median_abs_onset_err_ms": round(
                float(np.median(np.abs(self.onset_err))) * 1000, 1
            ) if self.onset_err else 0.0,
        }


def score_notes(
    truth: list[TruthNote],
    pred: list[dict],
    *,
    onset_tol: float = 0.08,
    pitch_tol: float = 1.0,
    ignore_spans: list[tuple[float, float]] | None = None,
) -> Score:
    """정답/산출 노트열을 온셋 기준으로 1:1 그리디 매칭해 채점.

    매칭 규칙: 온셋 차 ≤ onset_tol인 미사용 산출 노트 중 (음고 오차, 온셋 오차) 최소.
    이렇게 하면 "옆 노트를 훔쳐 억지로 맞추는" 경우가 음고 오류로 정직하게 남는다.
    ignore_spans 안에서 시작하는 미매칭 산출 노트는 과분할로 세지 않는다 (채점 대상에서
    뺀 정답 노트 자리).
    """
    def _ignored(p: dict) -> bool:
        return bool(ignore_spans) and any(a <= p["start"] < b for a, b in ignore_spans)

    s = Score(truth_n=len(truth), pred_n=len(pred))
    if not truth:
        s.pred_n = s.spurious = sum(1 for p in pred if not _ignored(p))
        return s
    pred_sorted = sorted(pred, key=lambda p: p["start"])
    starts = np.array([p["start"] for p in pred_sorted]) if pred_sorted else np.zeros(0)
    used = [False] * len(pred_sorted)

    for t in truth:
        lo = int(np.searchsorted(starts, t.start - onset_tol))
        hi = int(np.searchsorted(starts, t.start + onset_tol, side="right"))
        cands = [i for i in range(lo, hi) if not used[i]]
        if cands:
            best = min(
                cands,
                key=lambda i: (
                    abs(pred_sorted[i]["midi"] - t.midi),
                    abs(pred_sorted[i]["start"] - t.start),
                ),
            )
            used[best] = True
            p = pred_sorted[best]
            d = p["midi"] - t.midi
            s.abs_pitch_err.append(abs(d))
            s.onset_err.append(p["start"] - t.start)
            if abs(d) <= pitch_tol:
                s.matched += 1
                if abs(p["end"] - t.end) <= onset_tol:
                    s.offset_ok += 1
            elif abs(d) >= 7:
                s.onset_ok_octave += 1
            else:
                s.onset_ok_pitch_off += 1
            continue
        # 온셋 허용 밖이라도 시간적으로 겹치는 산출 노트가 있으면 "경계 어긋남"
        overlap = [
            i
            for i in range(len(pred_sorted))
            if not used[i] and pred_sorted[i]["start"] < t.end and pred_sorted[i]["end"] > t.start
        ]
        if overlap:
            used[overlap[0]] = True
            s.onset_off += 1
        else:
            s.missed += 1
    # 무시 스팬의 **미매칭** 산출만 분모·과분할에서 뺀다. 매칭된 것까지 빼면
    # matched > pred_n이 되어 정밀도가 1을 넘는다.
    ignored_unmatched = sum(
        1 for i, u in enumerate(used) if not u and _ignored(pred_sorted[i])
    )
    s.pred_n -= ignored_unmatched
    s.spurious = sum(1 for u in used if not u) - ignored_unmatched
    return s


# ---------------------------------------------------------------- 실행


def run_song(
    spec: SongSpec,
    profile: SynthProfile,
    settings: MelodySettings,
    seed: int,
    *,
    onset_tol: float = 0.08,
    pitch_tol: float = 1.0,
    limit_sec: float = 0.0,
    min_truth_note: float = 0.0,
) -> tuple[Score, dict]:
    rng = np.random.default_rng(seed)
    truth = load_truth(spec)
    if limit_sec > 0:
        truth = [n for n in truth if n.end <= limit_sec]
    if not truth:
        raise ValueError(f"no truth notes: {spec.label}")

    hz, times = synth_f0(truth, profile, rng)
    segments = build_segments(truth, profile, rng)

    extractor = MelodyExtractor(settings)
    extractor.annotate_timestamps(None, segments, precomputed_f0=(hz, times))
    pred = [n for seg in segments for n in seg.get("notes", [])]

    # 채점 대상에서만 초단기 노트를 뺀다 (합성·정렬 입력에는 그대로 남긴다).
    # UST에는 자음 코다용 13~50ms 노트가 섞여 있는데(전체의 4%) 10ms hop f0로는
    # 표현 자체가 불가능하고 노래방 막대로 보여줄 수도 없다 — 이걸 정답에 넣고 재면
    # 도달 불가능한 천장을 만든다.
    scored, ignore = truth, None
    if min_truth_note:
        scored = [n for n in truth if n.end - n.start >= min_truth_note]
        # 제외한 정답 노트 자리의 산출 노트는 과분할로 세지 않는다 — 채점하지 않기로
        # 한 노트를 맞힌 것이 정밀도를 깎으면 안 된다.
        ignore = [(n.start, n.end) for n in truth if n.end - n.start < min_truth_note]
    score = score_notes(
        scored, pred, onset_tol=onset_tol, pitch_tol=pitch_tol, ignore_spans=ignore
    )
    meta = {"lines": len(segments), "key": extractor.last_key}
    return score, meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default="realistic",
                    help="clean|sung|realistic|hard|all (쉼표 구분 가능)")
    ap.add_argument("--songs", default="", help="쉼표 구분 label 필터")
    ap.add_argument("--seed", type=int, default=20260803)
    ap.add_argument("--onset-tol", type=float, default=0.08)
    ap.add_argument("--pitch-tol", type=float, default=1.0)
    ap.add_argument("--limit-sec", type=float, default=0.0, help="곡당 앞부분만 (빠른 반복용)")
    ap.add_argument("--min-truth-note", type=float, default=0.0,
                    help="이보다 짧은 정답 노트는 채점 제외 (0.08 권장 — 자음 코다용 초단기 노트)")
    ap.add_argument("--json", default="", help="결과 JSON 저장 경로")
    ap.add_argument("--per-song", action="store_true", help="곡별 표 출력")
    args = ap.parse_args()

    names = (
        list(PROFILES)
        if args.profile == "all"
        else [p.strip() for p in args.profile.split(",") if p.strip()]
    )
    wanted = {s.strip() for s in args.songs.split(",") if s.strip()}
    songs = [s for s in SONGS if not wanted or s.label in wanted]

    # 벤치는 f0 품질이 아니라 양자화를 재므로 분리·백엔드 설정은 무의미 (precomputed 경로)
    settings = MelodySettings(separate_vocals=False)

    out: dict = {"profiles": {}, "config": {
        "onset_tol": args.onset_tol, "pitch_tol": args.pitch_tol, "seed": args.seed,
        "songs": [s.label for s in songs],
    }}
    for name in names:
        profile = PROFILES[name]
        total = Score()
        rows: list[tuple[str, Score]] = []
        for i, spec in enumerate(songs):
            try:
                score, _meta = run_song(
                    spec, profile, settings, args.seed + i,
                    onset_tol=args.onset_tol, pitch_tol=args.pitch_tol,
                    limit_sec=args.limit_sec, min_truth_note=args.min_truth_note,
                )
            except Exception as exc:  # 개별 곡 파싱 실패가 전체 벤치를 죽이지 않게
                print(f"  [skip] {spec.label}: {type(exc).__name__}: {exc}")
                continue
            rows.append((spec.label, score))
            total.merge(score)

        out["profiles"][name] = {
            "total": total.as_dict(),
            "songs": {label: sc.as_dict() for label, sc in rows},
        }
        d = total.as_dict()
        print(f"\n=== profile: {name} ({len(rows)} songs) ===")
        print(
            f"  F1 {d['f1']:.3f}  recall {d['recall']:.3f}  precision {d['precision']:.3f}"
            f"  (truth {d['truth_n']}, pred {d['pred_n']})"
        )
        print(
            f"  오류 분해: 음고오류 {d['err_pitch_off']}  옥타브 {d['err_octave']}"
            f"  경계이탈 {d['err_onset_off']}  소실 {d['err_missed']}  과분할 {d['err_spurious']}"
        )
        print(
            f"  중앙값 음고오차 {d['median_abs_pitch_err']}반음  "
            f"온셋오차 {d['median_abs_onset_err_ms']}ms  끝경계일치 {d['offset_ok_rate']:.3f}"
        )
        if args.per_song:
            for label, sc in sorted(rows, key=lambda r: r[1].f1):
                sd = sc.as_dict()
                print(
                    f"    {label:<18} F1 {sd['f1']:.3f}  R {sd['recall']:.3f}"
                    f"  P {sd['precision']:.3f}  음고 {sd['err_pitch_off']}"
                    f"  옥타브 {sd['err_octave']}  경계 {sd['err_onset_off']}"
                    f"  소실 {sd['err_missed']}  과분할 {sd['err_spurious']}"
                )

    if args.json:
        Path(args.json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
