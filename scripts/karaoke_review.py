"""Generate a file://-safe single-page karaoke alignment review viewer.

The viewer loads one song at a time with local ``<script>`` tags.  This avoids
``fetch`` CORS restrictions while keeping the browser page completely offline.
"""

from __future__ import annotations

import argparse
import bisect
import colorsys
import json
import re
from collections.abc import Iterable
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "benchmark"
# 트랙 색 — 곡·뷰어가 달라도 같은 조합은 항상 같은 색이 나오게 결정론으로 만든다.
# 색상(hue)=전사 라우트, 밝기(lightness)=분리기, 채도(saturation)=음차(@hangul*) 여부.
ALIGNER_HUE = {
    "omniasr-ctc": 210 / 360,          # 파랑 계열
    "owsm-ctc-v4-1b": 172 / 360,       # 청록 — 구세대 fp32, bf16 현역과 같은 초록 계열이되 구분
    "owsm-ctc-v4-1b-bf16": 150 / 360,  # 초록 계열
    "nemo-nfa": 25 / 360,              # 주황 계열
    "hf-kkonjeong": 320 / 360,         # 자홍 — ko CTC 계열(nemo와 같은 음차 구조, 다른 모델)
}
# 레인 왼쪽 세로 띠 — 분리기를 색으로 즉시 구분(특히 무분리). 정렬기 색과 독립된 축.
SEP_BAND = {
    "nosep": "#e0a03c",              # 주황 — 무분리(분리기 없음)
    "bs-polarformer": "#4f9bd9", "bs-polarformer-fp16": "#4f9bd9",   # 파랑
    "kimft-melband": "#63c88a", "kimft-melband-fp16": "#63c88a",     # 초록
    "umx-l": "#9a7bd0", "htdemucs": "#8a8f99", "demucs-onnx-fp16": "#8a8f99",
}
SEP_LIGHTNESS = {
    "kimft-melband": 0.55, "kimft-melband-fp16": 0.55,  # 주력 — 가장 잘 보이는 중간 밝기
    "bs-leap-xe": 0.42, "bs-polarformer": 0.48, "bs-polarformer-fp16": 0.48,
    "anvuew-ft1": 0.63, "demucs-onnx-fp16": 0.70, "htdemucs": 0.70,
    "umx-l": 0.77, "nosep": 0.35,
}


def _track_color(separator: str, aligner: str) -> str:
    base, _, suffix = aligner.partition("@")
    light = SEP_LIGHTNESS.get(separator, 0.60)
    if base == "mms-baseline":
        hue, sat = 0.0, 0.06  # 베이스라인은 무채색
    else:
        hue = ALIGNER_HUE.get(base, 300 / 360)
        sat = 0.38 if suffix else 0.72  # 음차 모드는 채도를 낮춰 같은 계열의 변형으로 보이게
    r, g, b = colorsys.hls_to_rgb(hue, light, sat)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def _number(value: object) -> float | None:
    """Return a usable timestamp, or ``None`` when the source is malformed."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 and result != float("inf") else None


def _short_label(text: str, limit: int = 30) -> str:
    clean = " ".join(text.split())
    return clean if len(clean) <= limit else f"{clean[:limit - 1]}…"


def _normalise_lines(items: Iterable[dict]) -> list[dict]:
    lines: list[dict] = []
    for item in items:
        start, end = _number(item.get("start")), _number(item.get("end"))
        if start is None or end is None or end < start:
            continue
        text = str(item.get("text") or "")
        lines.append({"text": text, "label": _short_label(text), "start": start, "end": end})
    return sorted(lines, key=lambda line: (line["start"], line["end"]))


def _normalise_segs(items: Iterable[dict]) -> list[dict]:
    """Normalise word/syllable spans from prod and every candidate run."""
    segs: list[dict] = []
    for item in items:
        start, end = _number(item.get("start")), _number(item.get("end"))
        if start is None or end is None or end < start:
            continue
        text = str(item.get("t") or item.get("word") or item.get("text") or "")
        segs.append({"t": text, "start": start, "end": end})
    return sorted(segs, key=lambda seg: (seg["start"], seg["end"]))


def _attach_prod_matches(segs: list[dict], prod_segs: list[dict]) -> None:
    """Attach nearest matching PROD span for the tooltip delta, in O(n log n)."""
    by_text: dict[str, list[dict]] = {}
    for prod in prod_segs:
        if prod["t"].strip():
            by_text.setdefault(prod["t"], []).append(prod)
    starts = {text: [span["start"] for span in spans] for text, spans in by_text.items()}
    for seg in segs:
        candidates = by_text.get(seg["t"])
        if not candidates:
            continue
        index = bisect.bisect_left(starts[seg["t"]], seg["start"])
        nearby = candidates[max(0, index - 1) : min(len(candidates), index + 2)]
        closest = min(
            nearby,
            key=lambda prod: (
                abs((prod["start"] + prod["end"]) - (seg["start"] + seg["end"])),
                abs(prod["start"] - seg["start"]),
            ),
        )
        seg["prod"] = [closest["start"], closest["end"]]


def _parse_srt_time(value: str) -> float:
    match = re.fullmatch(r"(\d+):(\d{2}):(\d{2})[,.](\d{3})", value.strip())
    if match is None:
        raise ValueError(f"Invalid SRT timestamp: {value!r}")
    hours, minutes, seconds, milliseconds = (int(part) for part in match.groups())
    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000


def parse_srt(path: Path) -> list[dict]:
    """Parse an SRT into reference-line blocks; SRT has no syllable spans."""
    lines: list[dict] = []
    for block in re.split(r"\r?\n\s*\r?\n", path.read_text(encoding="utf-8-sig").strip()):
        rows = [row.strip() for row in block.splitlines() if row.strip()]
        time_row = next((row for row in rows if "-->" in row), None)
        if time_row is None:
            continue
        try:
            start_text, end_text = (part.strip().split()[0] for part in time_row.split("-->", 1))
            start, end = _parse_srt_time(start_text), _parse_srt_time(end_text)
        except (IndexError, ValueError):
            continue
        caption = " ".join(rows[rows.index(time_row) + 1 :])
        if caption and end >= start:
            lines.append({"text": caption, "label": _short_label(caption), "start": start, "end": end})
    return lines


# 브레스·무음 토큰. 공백 유무로 휴지를 판정하면 VCV 로마자("- ke")가 전멸하므로 토큰으로 가른다.
_UST_NON_LYRIC = {"r", "br", "v", "cl", "sil", "pau", "vf", "-", "+"}
# utaformatix 등으로 변환된 ustx는 음이름이 가사 뒤에 붙어 나온다("- keD#4", "へA#3").
_UST_PITCH_SUFFIX = re.compile(r"[A-G]#?-?\d+$")


def _ust_lyric(raw: str) -> str | None:
    """UST/USTX 가사 토큰 → 발음 음절. 휴지·브레스면 None.

    처리 순서: 음이름 꼬리 제거 → VCV 선행 모음 접두사 제거 → 고정 휴지 토큰 판정.
    """
    clean = str(raw).strip()
    if not clean:
        return None
    clean = _UST_PITCH_SUFFIX.sub("", clean).strip()
    body = re.sub(r"^[\x00-\x7f]+\s+", "", clean)
    if body:
        clean = body
    if not clean or clean.lower() in _UST_NON_LYRIC or not any(ch.isalpha() for ch in clean):
        return None
    return clean


def parse_ust_track(path: Path, offset: float = 0.0) -> tuple[list[dict], list[dict]]:
    """UTAU UST → (프레이즈 라인, 노트 세그). 팬 채보 준정답 레인용.

    틱→초: 노트 Length(480틱=4분음표)와 Tempo(전역 + 노트별 오버라이드)로 결정론 환산.
    프레이즈 경계: 휴지(R) 또는 0.35s 이상 갭. offset은 UST→오디오 전역 이동(초).
    """
    raw = path.read_bytes()
    text = None
    for enc in ("cp932", "utf-8-sig", "utf-8"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise SystemExit(f"UST decode failed: {path}")
    tempo = 120.0
    for line in text.splitlines():
        if line.startswith("Tempo="):
            try:
                tempo = float(line.split("=", 1)[1].replace(",", "."))
            except ValueError:
                pass
            break
    notes: list[tuple[int, str, float | None]] = []
    cur: dict[str, str] = {}

    def flush() -> None:
        section = cur.get("_section", "")
        if section.startswith("#") and section not in ("#SETTING", "#VERSION", "#TRACKEND"):
            note_tempo = None
            if "Tempo" in cur:
                try:
                    note_tempo = float(cur["Tempo"].replace(",", "."))
                except ValueError:
                    note_tempo = None
            try:
                length = int(cur.get("Length", "0"))
            except ValueError:
                length = 0
            notes.append((length, cur.get("Lyric", ""), note_tempo))

    for line in text.splitlines():
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            flush()
            cur = {"_section": line[1:-1]}
        elif "=" in line:
            key, _, value = line.partition("=")
            cur[key] = value
    flush()

    segs: list[dict] = []
    t = 0.0
    for length, lyric, note_tempo in notes:
        if note_tempo:
            tempo = note_tempo
        dur = length * 60.0 / (tempo * 480)
        clean = _ust_lyric(lyric)
        if clean:
            segs.append({"t": clean, "start": t + offset, "end": t + dur + offset})
        t += dur

    lines: list[dict] = []
    for seg in segs:
        if lines and seg["start"] - lines[-1]["end"] <= 0.35:
            lines[-1]["end"] = max(lines[-1]["end"], seg["end"])
            lines[-1]["text"] += seg["t"]
        else:
            lines.append({"text": seg["t"], "start": seg["start"], "end": seg["end"]})
    return _normalise_lines(lines), _normalise_segs(segs)


def parse_ustx_tracks(path: Path, offset: float = 0.0) -> list[tuple[str, list[dict], list[dict]]]:
    """OpenUtau USTX → [(레인 라벨, 라인, 세그)] — 파트를 리드/하모리 두 묶음으로.

    ustx는 YAML(BOM)이고 템포맵·파트 절대 위치(position, 틱)가 명시돼 있어 UST보다 정확하다.
    파트명에 'harm'이 들어가면 하모리 묶음, 나머지는 리드 묶음. '+' 등 ASCII 전용 가사
    (연장·브레스 표기)는 UST와 같은 기준으로 제외한다.
    """
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

    groups: dict[str, list[dict]] = {}
    for part in doc.get("voice_parts") or []:
        base = int(part.get("position") or 0)
        label = "harm" if "harm" in str(part.get("name") or "").lower() else "lead"
        for note in part.get("notes") or []:
            lyric = _ust_lyric(note.get("lyric") or "")
            if not lyric:
                continue
            tick = base + int(note.get("position") or 0)
            groups.setdefault(label, []).append({
                "t": lyric,
                "start": tick_to_sec(tick) + offset,
                "end": tick_to_sec(tick + int(note.get("duration") or 0)) + offset,
            })

    out: list[tuple[str, list[dict], list[dict]]] = []
    for label in ("lead", "harm"):
        segs = sorted(groups.get(label) or [], key=lambda s: (s["start"], s["end"]))
        if not segs:
            continue
        lines: list[dict] = []
        for seg in segs:
            if lines and seg["start"] - lines[-1]["end"] <= 0.35:
                lines[-1]["end"] = max(lines[-1]["end"], seg["end"])
                lines[-1]["text"] += seg["t"]
            else:
                lines.append({"text": seg["t"], "start": seg["start"], "end": seg["end"]})
        out.append((label, _normalise_lines(lines), _normalise_segs(segs)))
    return out


# E1에서 관문을 통과한 클린 융합(UST 17곡 ≤0.15s 80.1%) — 뷰어에서 직접 듣기 위한 합성 레인.
# 3다리 홀수라 라인 시작의 중앙값은 «실제 어느 한 레인의 값»이다. 그 레인의 라인과 음절을
# 그대로 가져오므로 없는 타이밍을 지어내지 않는다(짝수 다리였다면 평균이라 조작이 된다).
FUSION_SEPARATOR = "bs-polarformer-fp16"
FUSION_LANES = ("omniasr-ctc", "hf-slplab-phone-mfa@hangul-local", "hf-kkonjeong@hangul-local")


def _fused_track(runs: list[tuple[Path, dict]], prod_segs: list[dict]) -> dict | None:
    by_lane: dict[str, dict] = {}
    for path, run in runs:
        separator, _, lane = path.parent.name.partition("__")
        if separator == FUSION_SEPARATOR and lane in FUSION_LANES:
            by_lane[lane] = run
    if len(by_lane) != len(FUSION_LANES):
        return None
    ordered = [by_lane[lane]["lines"] for lane in FUSION_LANES]
    count = min(len(lines) for lines in ordered)
    if count < 5:
        return None
    picked: list[dict] = []
    for i in range(count):
        # 시작 시각 기준 가운데 레인을 고른다 = 중앙값을 낸 레인
        mid = sorted(ordered, key=lambda lines: _number(lines[i].get("start")) or 0.0)[1]
        picked.append(mid[i])
    segs = _normalise_segs(seg for line in picked for seg in (line.get("segs") or []))
    _attach_prod_matches(segs, prod_segs)
    return {
        "name": "융합 · 클린3 중앙값",
        "color": "#ffd166",
        "band": SEP_BAND.get(FUSION_SEPARATOR, ""),
        "lines": _normalise_lines(picked),
        "segs": segs,
        "no_segs": not segs,
    }


def _baseline_segments(song: dict) -> list[dict]:
    timestamps = song.get("baseline_timestamps") or {}
    return timestamps if isinstance(timestamps, list) else list(
        timestamps.get("segments") or timestamps.get("timestamps") or []
    )


def _song_root(song: dict) -> Path | None:
    """Locate the already-created artifact folder without importing benchmark code."""
    matches = sorted((BENCH / "results").glob(f"*/{song['video_id']}__*"))
    return matches[0] if matches else None


def _audio_paths(root: Path) -> list[dict] | None:
    """모든 분리기 스템을 소스 목록으로 — 뷰어에서 분리기별 청취 비교용."""
    audio_dir = root / "audio"
    originals = sorted(audio_dir.glob("original.*")) if audio_dir.is_dir() else []
    if not originals:
        return None
    sources = [{"name": "original mix", "path": f"audio/{originals[0].name}"}]
    has_fp16_stems = (audio_dir / "kimft-melband-fp16" / "vocals.wav").is_file()
    has_onnx_stems = (audio_dir / "demucs-onnx-fp16" / "vocals.wav").is_file()
    for sep_dir in sorted(path for path in audio_dir.iterdir() if path.is_dir()):
        if sep_dir.name in VIEWER_SEPARATOR_EXCLUDE:
            continue
        if sep_dir.name == "kimft-melband" and has_fp16_stems:
            continue
        if sep_dir.name == "htdemucs" and has_onnx_stems:
            continue
        for stem in ("vocals", "inst"):
            if (sep_dir / f"{stem}.wav").is_file():
                sources.append(
                    {"name": f"{sep_dir.name} · {stem}", "path": f"audio/{sep_dir.name}/{stem}.wav"}
                )
    return sources


# 뷰어 레인은 베이스라인 + 현역 후보만 — 탈락 후보(qwen3, 구 hf 계열, owsm 실험 변형)는
# 런 캐시는 남기되 표시에서 뺀다. 기준은 combo 디렉터리명의 base 정렬기(@suffix 제거).
VIEWER_ALIGNERS = {
    "mms-baseline",
    "omniasr-ctc",
    "owsm-ctc-v4-1b",
    "owsm-ctc-v4-1b-bf16",
    # nemo-nfa 제외(사용자 지시 2026-08-01) — ko 체크포인트가 Riva EULA 위반 정황이라 채택 불가.
    # 성능 최상위였으나 권원이 없어 비교 대상에서도 뺀다(런 캐시는 보존).
    # ko CTC + 한글 음차 경로 — 극한곡에서 omniasr 대비 +8.8pp(2026-08-01 UST 실측)로 복권
    "hf-kkonjeong",
    # ja 네이티브 — 구 프론티어 재평가에서 UST 73.3%로 생존(2026-08-01)
    "hf-reazon-hubert-base",
    # 음소 43 vocab — vocab 축소 가설 검증용 클린 후보(Apache-2.0)
    "hf-slplab-phone-mfa",
}

# 분리기 필터 — mini-bsrofo-18m은 분리 품질 미달로 전면 제외(청취 판정),
# bs-leap-xe·anvuew-ft1·bs-hyperace-v2는 사용자 판정으로 전면 제외(2026-07-31),
# demucs-onnx(fp32)는 fp16과 스템 동일·int8은 실측 실패라 구현 단계 테스트 런만 남은 상태.
# kimft-melband(fp32)은 fp16 변형과 결과가 동일해 fp16이 있는 자리에서는 숨긴다
# (fp16 런이 없는 비서브셋 곡에서는 폴백으로 유지).
VIEWER_SEPARATOR_EXCLUDE = {
    "mini-bsrofo-18m", "bs-leap-xe", "anvuew-ft1", "bs-hyperace-v2",
    "demucs-onnx", "demucs-onnx-int8",
    # demucs 계열 전면 제외(사용자 지시 2026-08-01) — 가중치 라이선스 비MIT 확정과도 부합
    "demucs-onnx-fp16", "htdemucs",
    # 폴라포머 속도 실험 변형 — 2곡에만 있고 ov2는 청크 seam으로 기각됨(비교를 흐린다)
    "bs-polarformer-fast", "bs-polarformer-ov2",
    # umx-l 기각(2026-08-01) — 짝지은 비교에서 세 라우트 모두 무분리에 패배(-5.5~-8.0pp)하면서
    # 곡당 1.7s·VRAM 4.3GB(분리기 최대)를 쓴다. 무분리에 열등 지배.
    "umx-l",
    # kimft 제외(사용자 지시 2026-08-01) — polar와 품질 무승부(정상 116쌍 MAE 차 중앙 0.0000s)인데
    # VRAM 3,062MB로 polar(2,294)보다 무거워 3090 9GB 상주 예산을 넘긴다.
    "kimft-melband", "kimft-melband-fp16",
}

# 오입력 확정 곡(2026-07-31, collapse_annotations.json) — 가사·음원 불일치라 어떤 레인도
# 무의미해 뷰어 목록에서 제외한다. hard_audio 2곡(토스트·BUTCHER)은 정당한 난곡이라 유지.
VIEWER_EXCLUDE_SONGS = {
    "emrt46SRyYs", "HyBxn5gzpn0", "LaEgpNBt-bQ", "-glcrfq-Sw4", "uGWsbwZtmeY", "JcvWe8EHdek",
}


def build_tracks(
    song: dict,
    srt_path: Path | None = None,
    ust_specs: list[tuple[Path, float]] | None = None,
    ust_shifts: dict[str, dict] | None = None,
) -> list[dict]:
    baseline = _baseline_segments(song)
    prod_segs = _normalise_segs(word for line in baseline for word in (line.get("words") or []))
    _attach_prod_matches(prod_segs, prod_segs)
    tracks = [{"name": "PROD", "color": "#aeb6c2", "lines": _normalise_lines(baseline), "segs": prod_segs}]
    if srt_path is not None:
        tracks.append(
            {
                "name": f"CAPTION · {srt_path.name}",
                "color": "#ffd166",
                "lines": parse_srt(srt_path),
                "segs": [],
                "line_reference": True,
            }
        )
    for ust_path, ust_offset in ust_specs or []:
        if ust_path.suffix.lower() == ".ustx":
            lanes = [(f"{ust_path.stem} · {label}", lines, segs)
                     for label, lines, segs in parse_ustx_tracks(ust_path, ust_offset)]
        else:
            u_lines, u_segs = parse_ust_track(ust_path, ust_offset)
            lanes = [(ust_path.stem, u_lines, u_segs)]
        for lane_label, u_lines, u_segs in lanes:
            # E키 내보내기 JSON(소절별 오프셋) 베이크 — 그룹핑(=인덱스 부여) 후에 적용해야
            # 뷰어에서 본 소절 인덱스와 일치한다. 적용 후 재정렬.
            shift = (ust_shifts or {}).get(lane_label)
            shifted_phrases = 0
            if shift and (shift["lane"] or shift["phrases"]):
                lane_delta = shift["lane"]
                pre_starts = [line["start"] for line in u_lines]

                def _line_idx(t: float) -> int:
                    j = bisect.bisect_right(pre_starts, t + 1e-4) - 1
                    return 0 if j < 0 else j

                for k, line in enumerate(u_lines):
                    d = lane_delta + shift["phrases"].get(k, 0.0)
                    line["start"] += d
                    line["end"] += d
                for seg in u_segs:
                    d = lane_delta + shift["phrases"].get(_line_idx(seg["start"]), 0.0)
                    seg["start"] += d
                    seg["end"] += d
                u_lines.sort(key=lambda line: (line["start"], line["end"]))
                u_segs.sort(key=lambda seg: (seg["start"], seg["end"]))
                shifted_phrases = len(shift["phrases"])
            _attach_prod_matches(u_segs, prod_segs)
            tracks.append(
                {
                    "name": f"UST · {lane_label}" + (f" (δ{ust_offset:+.2f}s)" if ust_offset else "")
                    + (f" (소절보정 {shifted_phrases})" if shifted_phrases else ""),
                    "color": "#f2a2c8",
                    "lines": u_lines,
                    "segs": u_segs,
                    "ust": True,  # 뷰어에서 Shift+드래그 오프셋 조정 대상
                }
            )
    # allin1 구조 분석 결과가 있으면 참조 레인으로 자동 표시 (간주/솔로 마스킹 가설 눈 검증용)
    allin1_path = BENCH / "allin1" / f"{song['video_id']}.json"
    if allin1_path.is_file():
        try:
            aj = json.loads(allin1_path.read_text(encoding="utf-8"))
            seg_lines = [
                {"text": str(s.get("label") or "?"), "start": s.get("start"), "end": s.get("end")}
                for s in (aj.get("segments") or [])
            ]
            if seg_lines:
                tracks.append(
                    {
                        "name": f"ALLIN1 구간 (BPM {aj.get('bpm')})",
                        "color": "#7bd88f",
                        "lines": _normalise_lines(seg_lines),
                        "segs": [],
                        "line_reference": True,
                    }
                )
        except (OSError, json.JSONDecodeError) as error:
            print(f"allin1 결과 무시 ({allin1_path}): {error}")

    video_id = song["video_id"]
    runs: list[tuple[Path, dict]] = []
    # fp32→fp16 대체: fp16 런이 같은 레인을 제공하면 fp32 변형은 숨긴다(결과 동일 실측).
    FP16_REPLACES = {"kimft-melband-fp16": "kimft-melband", "bs-polarformer-fp16": "bs-polarformer"}
    fp16_lanes: dict[str, set[str]] = {}  # fp32 분리기명 → fp16이 커버한 레인들
    for path in sorted((BENCH / "runs").glob(f"*/{video_id}__r1.json")):
        separator, _, aligner = path.parent.name.partition("__")
        if aligner.partition("@")[0] not in VIEWER_ALIGNERS or separator in VIEWER_SEPARATOR_EXCLUDE:
            continue
        try:
            run = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            print(f"Skipping invalid run {path}: {error}")
            continue
        if not run.get("error") and run.get("lines"):
            runs.append((path, run))
            if separator in FP16_REPLACES:
                fp16_lanes.setdefault(FP16_REPLACES[separator], set()).add(aligner)
    runs = [
        (path, run)
        for path, run in runs
        if path.parent.name.partition("__")[2] not in fp16_lanes.get(path.parent.name.partition("__")[0], set())
    ]
    # htdemucs는 demucs-onnx-fp16(htdemucs_ft 보컬 타깃 ONNX)에 전 라우트 지표로 밀리고
    # 스템도 청취상 구분이 안 돼(사용자 판정) 같은 레인을 다른 분리기가 제공하면 숨긴다.
    # 유일 커버리지(예: mms-baseline@hangul은 htdemucs에만 있음)는 남긴다.
    covered: dict[str, set[str]] = {}
    for path, _ in runs:
        sep_name, _, lane = path.parent.name.partition("__")
        covered.setdefault(lane, set()).add(sep_name)
    runs = [
        (path, run)
        for path, run in runs
        if not (
            path.parent.name.partition("__")[0] == "htdemucs"
            and covered[path.parent.name.partition("__")[2]] - {"htdemucs"}
        )
    ]

    for path, run in runs:
        separator, _, lane = path.parent.name.partition("__")
        aligner = str(run.get("aligner") or lane or path.parent.name)
        # 항상 "분리기 · 정렬기" — 첫 조합만 접두사를 생략하던 규칙은 곡마다 다른 레인이
        # 무접두사 이름을 차지해 뷰어 간 비교를 흐렸다.
        name = f"{separator} · {aligner}" if lane else aligner
        segs = _normalise_segs(seg for line in run["lines"] for seg in (line.get("segs") or []))
        _attach_prod_matches(segs, prod_segs)
        tracks.append(
            {
                "name": name,
                "color": _track_color(separator if lane else "", aligner),
                "band": SEP_BAND.get(separator, ""),  # 레인 왼쪽 세로 띠 = 분리기 구분
                "lines": _normalise_lines(run["lines"]),
                "segs": segs,
                "no_segs": not segs,
            }
        )
    fused = _fused_track(runs, prod_segs)
    if fused:
        tracks.append(fused)
    return tracks


UST_TRUTH_EXCLUDE = {"s4kAOHUSvT8": [(58.0, 78.0)]}  # 토스트 간주 '아아아' — 실제 가사와 다름(사용자 확인)

# 한글 → 로마자. UST 채보자들이 한국어 곡도 로마자 CV로 적는 경우가 많아(catch catch 실측),
# 텍스트 앵커를 붙이려면 한글 쪽을 같은 표기로 내려야 한다. 표기 관습이 제각각이라
# 엄밀한 국어 로마자법이 아니라 느슨한 대조용 표기다(ㅐ/ㅔ 병합, 종성 대표음).
_HANGUL_CHO = ("g", "kk", "n", "d", "tt", "r", "m", "b", "pp", "s", "ss", "", "j", "jj",
               "ch", "k", "t", "p", "h")
_HANGUL_JUNG = ("a", "e", "ya", "ye", "eo", "e", "yeo", "ye", "o", "wa", "we", "we", "yo",
                "u", "wo", "we", "wi", "yu", "eu", "ui", "i")
_HANGUL_JONG = ("", "k", "k", "k", "n", "n", "n", "t", "l", "k", "m", "l", "l", "l", "p",
                "l", "m", "p", "p", "t", "t", "ng", "t", "t", "k", "t", "p", "t")


def hangul_to_roman(text: str) -> str:
    out: list[str] = []
    for ch in str(text):
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:
            idx = code - 0xAC00
            out.append(_HANGUL_CHO[idx // 588] + _HANGUL_JUNG[(idx % 588) // 28]
                       + _HANGUL_JONG[idx % 28])
        else:
            out.append(ch)
    return "".join(out)


def _ust_truth_anchors(video_id: str, tracks: list[dict]) -> dict | None:
    """UST 레인 → 라인별 정답 시각(텍스트 앵커, ust_line_judge v3와 동일 방법) — 레인 채점용.

    비-harm UST 노트와 PROD 라인 텍스트를 로마자 정규화해 전역 단조 정렬, 라인 문자 40%+
    매칭 시 앵커. 앵커 커버리지 40% 미만(en↔가나 등)이면 채점 무의미라 None.
    """
    notes = []
    for track in tracks:
        if track.get("ust") and "harm" not in track["name"].lower():
            notes += [(seg["start"], seg["t"]) for seg in track["segs"]]
    if not notes:
        return None
    prod = next((t for t in tracks if t["name"] == "PROD"), None)
    if prod is None or not prod.get("lines"):
        return None
    try:
        import difflib

        import pykakasi
    except ImportError:
        return None
    kks = getattr(_ust_truth_anchors, "_kks", None)
    if kks is None:
        kks = _ust_truth_anchors._kks = pykakasi.kakasi()

    def norm(text: str) -> str:
        out = "".join((item.get("hepburn") or "") for item in kks.convert(hangul_to_roman(text)))
        return re.sub(r"[^a-z0-9]", "", out.lower())

    notes.sort()
    for lo, hi in UST_TRUTH_EXCLUDE.get(video_id, []):
        notes = [n for n in notes if not (lo <= n[0] <= hi)]
    n_chars, n_time = [], []
    for start, txt in notes:
        r = norm(txt)
        n_chars.append(r)
        n_time += [start] * len(r)
    l_ranges, l_parts, pos = [], [], 0
    for line in prod["lines"]:
        r = norm(line["text"])
        l_ranges.append((pos, pos + len(r)))
        l_parts.append(r)
        pos += len(r)
    char_map = {}
    matcher = difflib.SequenceMatcher(None, "".join(l_parts), "".join(n_chars), autojunk=False)
    for blk in matcher.get_matching_blocks():
        for k in range(blk.size):
            char_map[blk.a + k] = blk.b + k
    anchors = {}
    for i, (a0, a1) in enumerate(l_ranges):
        hits = [char_map[c] for c in range(a0, a1) if c in char_map]
        if len(hits) >= max(2, (a1 - a0) * 0.4):
            anchors[str(i)] = round(n_time[hits[0]], 3)
    if len(anchors) / max(len(prod["lines"]), 1) < 0.4:
        return None
    return {"anchors": anchors, "lines": len(prod["lines"])}


def _json_for_script(data: object) -> str:
    """Serialize safely inside a script tag/assignment without external resources."""
    return json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c").replace(
        ">", "\\u003e"
    ).replace("&", "\\u0026")


SINGLE_PAGE_HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Karaoke alignment review</title>
<style>
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;min-height:100vh;background:#14161a;color:#dde1e7;font:13px system-ui,"Malgun Gothic",sans-serif;display:flex;flex-direction:column}#top{background:#1d2026;border-bottom:1px solid #343942;padding:9px 12px;z-index:4}h1{font-size:15px;margin:0 0 7px}#song-controls,#controls,#tracks-toggle{display:flex;gap:8px;align-items:center;flex-wrap:wrap}#tracks-toggle{margin-top:7px;font-size:11px}input,select,button{background:#2b3039;color:#e5e9ef;border:1px solid #4b5360;border-radius:4px;padding:4px 7px}#song-filter{min-width:190px}#song-select{min-width:min(620px,70vw);max-width:70vw}button{cursor:pointer}button:hover{background:#3c434e}audio{height:27px;max-width:330px}label{display:flex;align-items:center;gap:4px;cursor:pointer}.swatch{width:10px;height:10px;border-radius:2px;display:inline-block}.tag{color:#9ca8b8}.warn{color:#ffd166}.error{color:#ff8791}#wrap{position:relative;flex:1;overflow:auto;min-height:280px;background:#14161a}#timeline{position:relative;min-height:100%}#cv{position:sticky;left:0;top:0;display:block;z-index:1;cursor:crosshair}#tip{display:none;position:fixed;z-index:10;max-width:310px;padding:7px 9px;border:1px solid #667084;border-radius:5px;background:#11151bcc;color:#f2f5f8;font-size:12px;line-height:1.45;pointer-events:none;white-space:pre-line;box-shadow:0 4px 16px #0008}.hint{color:#9ca8b8;font-size:11px}kbd{padding:1px 4px;border:1px solid #515966;border-radius:3px;background:#292e36;font:11px monospace}
</style></head><body>
<div id="top"><h1 id="page-title">Karaoke alignment review</h1><div id="song-controls"><input id="song-filter" type="search" placeholder="Filter stratum, title, quality, ID"><label title="UST 준정답 레인이 있는 곡만"><input id="ust-only" type="checkbox">UST만</label><select id="song-select" aria-label="Song selector"></select><span id="song-status" class="tag"></span></div><div id="controls"><audio id="au" controls preload="metadata"></audio><select id="audiosrc" title="오디오 소스"></select><button id="zi">Zoom +</button><button id="zo">Zoom −</button><select id="viewmode" title="표시 모드"><option value="both">음절+라인</option><option value="segs">음절만</option><option value="lines">라인만</option></select><select id="followmode" title="화면 따라가기"><option value="page">따라가기: 페이지 넘김</option><option value="pin">따라가기: 헤드 고정</option><option value="off">따라가기: 끄기</option></select><span id="pos" class="tag"></span><span class="hint"><kbd>Space</kbd> play/pause · <kbd>←</kbd>/<kbd>→</kbd> seek 2s · click timeline to seek · 휠 zoom</span></div><div id="tracks-toggle"></div></div>
<div id="wrap"><div id="timeline"><canvas id="cv"></canvas></div></div><div id="tip"></div>
<script src="data/index.js"></script><script>
const ROW_H=50,LABEL_W=220;window.__SONG_DATA__=window.__SONG_DATA__||{};
const index=Array.isArray(window.__SONG_INDEX__)?window.__SONG_INDEX__:[];
let DATA=null,dur=1,pps=42,scheduled=false,activeId=null,SOURCES=[],visible=[],MODE='both',RENDER_TS=0,PAGE_TS=0,OFFS={},LOFFS={},USTDRAG=null,SEL=null,MARQ=null;
function lineOff(ti,li){const m=LOFFS[ti];return m?(m[li]||0):0}
function segEff(track,ti,si){return (OFFS[ti]||0)+(track._segLine?lineOff(ti,track._segLine[si]):0)}
const $=s=>document.querySelector(s),au=$('#au'),wrap=$('#wrap'),timeline=$('#timeline'),cv=$('#cv'),tip=$('#tip'),pos=$('#pos'),select=$('#song-select'),filter=$('#song-filter'),status=$('#song-status'),toggles=$('#tracks-toggle'),ctx=cv.getContext('2d');
const esc=v=>String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const stamp=v=>`${Math.floor(v/60)}:${String(Math.floor(v%60)).padStart(2,'0')}.${String(Math.round((v%1)*1000)).padStart(3,'0')}`;
const signed=v=>`${v>=0?'+':''}${v.toFixed(3)}s`;
const K1={'あ':'아','い':'이','う':'우','え':'에','お':'오','か':'카','き':'키','く':'쿠','け':'케','こ':'코','さ':'사','し':'시','す':'스','せ':'세','そ':'소','た':'타','ち':'치','つ':'츠','て':'테','と':'토','な':'나','に':'니','ぬ':'누','ね':'네','の':'노','は':'하','ひ':'히','ふ':'후','へ':'헤','ほ':'호','ま':'마','み':'미','む':'무','め':'메','も':'모','や':'야','ゆ':'유','よ':'요','ら':'라','り':'리','る':'루','れ':'레','ろ':'로','わ':'와','ゐ':'이','ゑ':'에','を':'오','ん':'응','が':'가','ぎ':'기','ぐ':'구','げ':'게','ご':'고','ざ':'자','じ':'지','ず':'즈','ぜ':'제','ぞ':'조','だ':'다','ぢ':'지','づ':'즈','で':'데','ど':'도','ば':'바','び':'비','ぶ':'부','べ':'베','ぼ':'보','ぱ':'파','ぴ':'피','ぷ':'푸','ぺ':'페','ぽ':'포','ぁ':'아','ぃ':'이','ぅ':'우','ぇ':'에','ぉ':'오','ゃ':'야','ゅ':'유','ょ':'요','っ':'ㅅ','ゔ':'부'};
const K2={'きゃ':'캬','きゅ':'큐','きょ':'쿄','しゃ':'샤','しゅ':'슈','しょ':'쇼','ちゃ':'차','ちゅ':'츄','ちょ':'초','にゃ':'냐','にゅ':'뉴','にょ':'뇨','ひゃ':'햐','ひゅ':'휴','ひょ':'효','みゃ':'먀','みゅ':'뮤','みょ':'묘','りゃ':'랴','りゅ':'류','りょ':'료','ぎゃ':'갸','ぎゅ':'규','ぎょ':'교','じゃ':'자','じゅ':'주','じょ':'조','ぢゃ':'자','ぢゅ':'주','ぢょ':'조','びゃ':'뱌','びゅ':'뷰','びょ':'뵤','ぴゃ':'퍄','ぴゅ':'퓨','ぴょ':'표','ふぁ':'화','ふぃ':'휘','ふぇ':'훼','ふぉ':'훠','うぃ':'위','うぇ':'웨','てぃ':'티','でぃ':'디','とぅ':'투','どぅ':'두','しぇ':'셰','ちぇ':'체','じぇ':'제'};
function toReadable(text){const h=String(text).replace(/[ァ-ヶ]/g,c=>String.fromCharCode(c.charCodeAt(0)-0x60));let out='';for(let i=0;i<h.length;i++){const two=h.slice(i,i+2);if(K2[two]){out+=K2[two];i++;continue}const ch=h[i];if(ch==='ー'){out+='-';continue}out+=(K1[ch]!==undefined?K1[ch]:ch)}return out}
const songLabel=s=>`[${s.stratum}] ${s.title}${s.ust_tracks?` · UST${s.ust_tracks>1?'×'+s.ust_tracks:''}`:''} · q ${Number(s.quality_score||0).toFixed(4)} · ${s.has_syllable_spans?'syllables':'no syllables'} · ${s.video_id}`;
function fillSelector(){const query=filter.value.trim().toLowerCase(),previous=select.value,ustOnly=$('#ust-only').checked;select.replaceChildren();const songs=index.filter(s=>(!ustOnly||s.ust_tracks)&&songLabel(s).toLowerCase().includes(query));for(const song of songs){const option=document.createElement('option');option.value=song.video_id;option.textContent=songLabel(song);select.append(option)}if(songs.some(s=>s.video_id===previous))select.value=previous;else if(songs.length)select.value=songs[0].video_id;status.textContent=songs.length?`${songs.length} songs`:'No matching songs'}
function activeRows(){return DATA?DATA.tracks.map((_,i)=>i).filter(i=>visible[i]):[]}
function lowerBound(items,time){let low=0,high=items.length;while(low<high){const mid=(low+high)>>1;if(items[mid].start<time)low=mid+1;else high=mid}return low}
function queueRender(){if(!scheduled){scheduled=true;requestAnimationFrame(()=>{scheduled=false;render()})}}
function render(){if(!DATA)return;const rows=activeRows(),width=Math.max(1,wrap.clientWidth),height=Math.max(1,wrap.clientHeight),dpr=devicePixelRatio||1;if(cv.width!==Math.ceil(width*dpr)||cv.height!==Math.ceil(height*dpr)){cv.width=Math.ceil(width*dpr);cv.height=Math.ceil(height*dpr);cv.style.width=width+'px';cv.style.height=height+'px'}const contentWidth=Math.max(width,LABEL_W+Math.ceil(dur*pps)),contentHeight=Math.max(height,rows.length*ROW_H);timeline.style.width=contentWidth+'px';timeline.style.height=contentHeight+'px';ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,width,height);ctx.fillStyle='#14161a';ctx.fillRect(0,0,width,height);const followMode=$('#followmode').value,span=Math.max(1,width-LABEL_W);let timeScroll;if(!au.paused&&followMode==='pin'){timeScroll=Math.max(0,au.currentTime*pps-span*.38)}else if(!au.paused&&followMode==='page'){const headX=au.currentTime*pps;if(headX<PAGE_TS||headX>PAGE_TS+span*.85)PAGE_TS=Math.max(0,headX-span*.15);timeScroll=PAGE_TS}else{timeScroll=Math.max(0,wrap.scrollLeft-LABEL_W)}RENDER_TS=timeScroll;const viewStart=timeScroll/pps,viewEnd=(timeScroll+Math.max(0,width-LABEL_W))/pps,yScroll=wrap.scrollTop;ctx.fillStyle='#1d2026';ctx.fillRect(0,0,LABEL_W,height);ctx.strokeStyle='#303640';ctx.beginPath();ctx.moveTo(LABEL_W-.5,0);ctx.lineTo(LABEL_W-.5,height);ctx.stroke();ctx.font='10px system-ui';ctx.fillStyle='#788392';ctx.strokeStyle='#292e36';for(let sec=Math.floor(viewStart/10)*10;sec<=viewEnd+10;sec+=10){const x=LABEL_W+sec*pps-timeScroll;ctx.beginPath();ctx.moveTo(x+.5,0);ctx.lineTo(x+.5,height);ctx.stroke();ctx.fillText(stamp(sec),x+3,11)}rows.forEach((trackIndex,row)=>{const track=DATA.tracks[trackIndex],top=row*ROW_H-yScroll,bottom=top+ROW_H;if(bottom<0||top>height)return;ctx.save();ctx.beginPath();ctx.rect(0,top,LABEL_W-4,ROW_H);ctx.clip();if(track.band){ctx.fillStyle=track.band;ctx.fillRect(0,top+1,4,ROW_H-2)}ctx.fillStyle=track.color;ctx.font='12px system-ui';ctx.fillText(track.name,8,top+14);ctx.fillStyle='#788392';ctx.font='9px system-ui';ctx.fillText(track.line_reference?'caption mapping':track.no_segs?'no measured syllables'+ustScore(track):track.ust?`Shift=레인 Ctrl=범위선택 Alt=이동 · δadj ${((OFFS[trackIndex]||0)>=0?'+':'')+(OFFS[trackIndex]||0).toFixed(2)}s${Object.keys(LOFFS[trackIndex]||{}).length?` (+소절 ${Object.keys(LOFFS[trackIndex]).length})`:''} · dblclick reset · E=복사`:`${track.segs.length} syllable spans${ustScore(track)}`,8,top+26);const nowT=au.currentTime-(track.ust?OFFS[trackIndex]||0:0);let liveTxt='';if(track.segs.length){const s=track.segs[lowerBound(track.segs,nowT)-1];if(s)liveTxt=s.t}else if(track.lines.length){const l=track.lines[lowerBound(track.lines,nowT)-1];if(l&&nowT<=l.end+1)liveTxt=l.label}if(liveTxt){const conv=toReadable(liveTxt),subst=conv!==liveTxt;ctx.font='bold 15px system-ui';const convW=ctx.measureText(conv).width;ctx.fillStyle=subst?'#a9b6c9':'#f2f5f8';ctx.fillText(conv,8,top+45);if(subst){ctx.font='10px system-ui';ctx.fillStyle='#5a6472';ctx.fillText(liveTxt,14+convW,top+45)}}ctx.restore();ctx.strokeStyle='#303640';ctx.beginPath();ctx.moveTo(0,bottom-.5);ctx.lineTo(width,bottom-.5);ctx.stroke();const toff=track.ust?OFFS[trackIndex]||0:0;const noSegTrack=track.line_reference||track.no_segs,drawLines=MODE!=='segs'||noSegTrack,drawSegs=MODE!=='lines'&&!noSegTrack;if(drawLines){const big=!drawSegs;const lineStart=track.ust?0:Math.max(0,lowerBound(track.lines,viewStart-toff)-1);for(let i=lineStart;i<track.lines.length&&(track.ust||track.lines[i].start+toff<=viewEnd);i++){const line=track.lines[i],leff=track.ust?toff+lineOff(trackIndex,i):toff;if(track.ust&&(line.end+leff<viewStart||line.start+leff>viewEnd))continue;const x=LABEL_W+(line.start+leff)*pps-timeScroll,w=Math.max(1,(line.end-line.start)*pps),ly=big?top+10:top+5,lh=big?30:9;ctx.fillStyle=track.color+(big?'55':'22');ctx.fillRect(x,ly,w,lh);ctx.strokeStyle=track.color+'88';ctx.strokeRect(x+.5,ly+.5,Math.max(1,w-1),lh-1);if(w>=24){ctx.save();ctx.beginPath();ctx.rect(x+1,ly,w-2,lh);ctx.clip();ctx.fillStyle=big?'#eef2f7':'#c7cede';ctx.font=big?'11px system-ui':'9px system-ui';ctx.fillText(line.label,x+3,big?ly+19:ly+8);ctx.restore()}}}if(drawSegs){const st=MODE==='both'?top+17:top+13,sh=MODE==='both'?23:25;const segStart=track.ust?0:Math.max(0,lowerBound(track.segs,viewStart-toff)-1);ctx.fillStyle=track.color;for(let i=segStart;i<track.segs.length&&(track.ust||track.segs[i].start+toff<=viewEnd);i++){const seg=track.segs[i],seff=track.ust?segEff(track,trackIndex,i):toff;if(track.ust&&(seg.end+seff<viewStart||seg.start+seff>viewEnd))continue;const x=LABEL_W+(seg.start+seff)*pps-timeScroll,w=Math.max(2,(seg.end-seg.start)*pps);ctx.fillRect(x,st,w,sh);if(track.ust&&SEL&&SEL.ti===trackIndex&&track._segLine&&SEL.set.has(track._segLine[i])){ctx.strokeStyle='#ffffff';ctx.strokeRect(x+.5,st+.5,Math.max(1,w-1),sh-1)}if(w>=11){ctx.fillStyle='#101318';ctx.font='11px system-ui';ctx.fillText(seg.t,x+2,st+sh-7);ctx.fillStyle=track.color}}}});if(MARQ){const rowIdx=rows.indexOf(MARQ.ti);if(rowIdx>=0){const lo=Math.min(MARQ.t0,MARQ.t1),hi=Math.max(MARQ.t0,MARQ.t1),mx=LABEL_W+lo*pps-timeScroll,mw=Math.max(1,(hi-lo)*pps),my=rowIdx*ROW_H-yScroll;ctx.fillStyle='#8ab4ff22';ctx.fillRect(mx,my,mw,ROW_H);ctx.strokeStyle='#8ab4ff';ctx.strokeRect(mx+.5,my+.5,mw-1,ROW_H-1)}}const playX=LABEL_W+au.currentTime*pps-timeScroll;if(playX>=LABEL_W&&playX<=width){ctx.strokeStyle='#ff5c67';ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(playX+.5,0);ctx.lineTo(playX+.5,height);ctx.stroke();ctx.lineWidth=1}}
function ustScoreVal(track){if(!DATA||!DATA.ust_truth||track.ust||track.line_reference||!track.lines)return null;const anch=DATA.ust_truth.anchors;let adj=0;for(let i=0;i<DATA.tracks.length;i++){const u=DATA.tracks[i];if(u.ust&&!/harm/i.test(u.name)){adj=OFFS[i]||0;break}}let n=0,hit=0;for(const k in anch){const line=track.lines[+k];if(!line)continue;n++;if(Math.abs(line.start-(anch[k]+adj))<=0.15)hit++}return n?hit/n:null}
function ustScore(track){const v=ustScoreVal(track);return v==null?'':` · UST ${Math.round(100*v)}%`}
// UST 준정답이 있는 곡은 레인을 채점 순으로 — 정답(UST)·참조 레인을 위에, 후보는 적중률 내림차순
function sortTracksByUst(song){if(!song.ust_truth)return;const grp=t=>t.ust?2:t.line_reference?1:0;song.tracks.sort((a,b)=>{const ga=grp(a),gb=grp(b);if(ga!==gb)return gb-ga;const va=ustScoreVal(a),vb=ustScoreVal(b);return (vb==null?-1:vb)-(va==null?-1:va)})}
function renderTrackToggles(){toggles.replaceChildren();DATA.tracks.forEach((track,i)=>{const label=document.createElement('label'),box=document.createElement('input');box.type='checkbox';box.checked=visible[i];box.addEventListener('change',()=>{visible[i]=box.checked;queueRender()});label.append(box);if(track.band){const bar=document.createElement('span');bar.className='swatch';bar.style.background=track.band;bar.style.width='4px';label.append(bar)}const swatch=document.createElement('span');swatch.className='swatch';swatch.style.background=track.color;label.append(swatch,document.createTextNode(track.name+(track.no_segs?' (no segs)':track.line_reference?' (caption)':'')));toggles.append(label)})}
function pointAt(event){const rect=cv.getBoundingClientRect();return{x:event.clientX-rect.left,y:event.clientY-rect.top,time:(RENDER_TS+event.clientX-rect.left-LABEL_W)/pps}}
function findSpan(track,time){if(!track?.segs.length)return null;const index=lowerBound(track.segs,time);for(const candidate of [index,index-1]){const seg=track.segs[candidate];if(seg&&time>=seg.start-4/pps&&time<=seg.end+4/pps)return seg}return null}
function findLine(track,time){if(!track?.lines?.length)return null;const index=lowerBound(track.lines,time);for(const candidate of [index,index-1]){const line=track.lines[candidate];if(line&&time>=line.start-2/pps&&time<=line.end+2/pps)return line}return null}
function showTip(event,html){tip.innerHTML=html;tip.style.display='block';tip.style.left=Math.min(event.clientX+14,innerWidth-325)+'px';tip.style.top=Math.min(event.clientY+14,innerHeight-90)+'px'}
cv.addEventListener('mousemove',event=>{if(MARQ){MARQ.t1=pointAt(event).time;queueRender();return}if(USTDRAG){const d=(event.clientX-USTDRAG.x)/pps;if(USTDRAG.lis){const m=LOFFS[USTDRAG.ti]=LOFFS[USTDRAG.ti]||{};USTDRAG.lis.forEach(k=>m[k]=USTDRAG.bases[k]+d)}else if(USTDRAG.li!=null){(LOFFS[USTDRAG.ti]=LOFFS[USTDRAG.ti]||{})[USTDRAG.li]=USTDRAG.base+d}else{OFFS[USTDRAG.ti]=USTDRAG.base+d}queueRender();return}const point=pointAt(event),ti=activeRows()[Math.floor((wrap.scrollTop+point.y)/ROW_H)],track=DATA?.tracks[ti];if(!track){tip.style.display='none';return}const toff=track.ust?OFFS[ti]||0:0;let seg=null,segE=toff;if(MODE!=='lines'){if(track.ust){for(let k=0;k<track.segs.length;k++){const e=segEff(track,ti,k);if(point.time>=track.segs[k].start+e-4/pps&&point.time<=track.segs[k].end+e+4/pps){seg=track.segs[k];segE=e;break}}}else{seg=findSpan(track,point.time-toff)}}if(seg){const prod=seg.prod,delta=prod?`PROD: ${stamp(prod[0])}–${stamp(prod[1])}\nΔ start ${signed(seg.start+segE-prod[0])} · Δ end ${signed(seg.end+segE-prod[1])}`:'PROD: no same-character span';showTip(event,`<b>${esc(track.name)} · ${esc(seg.t||'(empty)')}</b>\n${stamp(seg.start+segE)}–${stamp(seg.end+segE)}\n${delta}`);return}let line=null,lineE=toff;if(track.ust){for(let k=0;k<track.lines.length;k++){const e=toff+lineOff(ti,k);if(point.time>=track.lines[k].start+e-2/pps&&point.time<=track.lines[k].end+e+2/pps){line=track.lines[k];lineE=e;break}}}else{line=findLine(track,point.time-toff)}if(!line){tip.style.display='none';return}const text=line.text.length>140?line.text.slice(0,139)+'…':line.text;showTip(event,`<b>${esc(track.name)} · line</b>\n${esc(text)}\n${stamp(line.start+lineE)}–${stamp(line.end+lineE)} · ${(line.end-line.start).toFixed(2)}s`)});cv.addEventListener('mouseleave',()=>tip.style.display='none');function saveOff(ti){const t=DATA&&DATA.tracks[ti];if(!t)return;try{localStorage.setItem('ustoff:'+activeId+':'+t.name,String(OFFS[ti]||0));localStorage.setItem('ustloff:'+activeId+':'+t.name,JSON.stringify(LOFFS[ti]||{}))}catch(e){}}
cv.addEventListener('mousedown',event=>{if(!(event.shiftKey||event.altKey||event.ctrlKey)||!DATA)return;const point=pointAt(event),ti=activeRows()[Math.floor((wrap.scrollTop+point.y)/ROW_H)],track=DATA.tracks[ti];if(!track||!track.ust)return;event.preventDefault();
if(event.ctrlKey){MARQ={ti,t0:point.time,t1:point.time};return}
if(event.altKey){const toff=OFFS[ti]||0;let li=-1;for(let k=0;k<track.lines.length;k++){const e=toff+lineOff(ti,k);if(point.time>=track.lines[k].start+e-0.2&&point.time<=track.lines[k].end+e+0.2){li=k;break}}
if(SEL&&SEL.ti===ti&&(li<0||SEL.set.has(li))){const bases={};SEL.set.forEach(k=>bases[k]=lineOff(ti,k));USTDRAG={ti,lis:[...SEL.set],bases,x:event.clientX};return}
if(li<0)return;USTDRAG={ti,li,x:event.clientX,base:lineOff(ti,li)}}
else{USTDRAG={ti,x:event.clientX,base:OFFS[ti]||0}}});
window.addEventListener('mouseup',()=>{if(MARQ){const track=DATA&&DATA.tracks[MARQ.ti];if(track){const lo=Math.min(MARQ.t0,MARQ.t1),hi=Math.max(MARQ.t0,MARQ.t1),toff=OFFS[MARQ.ti]||0,set=new Set();track.lines.forEach((l,k)=>{const e=toff+lineOff(MARQ.ti,k);if(l.start+e<=hi&&l.end+e>=lo)set.add(k)});SEL=set.size?{ti:MARQ.ti,set}:null}MARQ=null;queueRender();return}if(!USTDRAG)return;saveOff(USTDRAG.ti);USTDRAG=null;queueRender()});
cv.addEventListener('dblclick',event=>{const point=pointAt(event),ti=activeRows()[Math.floor((wrap.scrollTop+point.y)/ROW_H)],track=DATA?.tracks[ti];if(!track||!track.ust)return;OFFS[ti]=0;delete LOFFS[ti];saveOff(ti);queueRender()});
cv.addEventListener('click',event=>{if(event.shiftKey||event.ctrlKey||event.altKey)return;const point=pointAt(event);if(point.x>=LABEL_W){au.currentTime=Math.max(0,Math.min(dur,point.time));queueRender()}});wrap.addEventListener('scroll',queueRender,{passive:true});wrap.addEventListener('wheel',event=>{if(!DATA)return;event.preventDefault();const rect=cv.getBoundingClientRect(),cursorX=event.clientX-rect.left,anchorTime=(RENDER_TS+cursorX-LABEL_W)/pps,old=pps;pps=Math.min(500,Math.max(5,pps*Math.pow(1.0015,-event.deltaY)));const ratio=pps/old;PAGE_TS=Math.max(0,PAGE_TS*ratio);if(au.paused||$('#followmode').value==='off'){const ts=anchorTime*pps-(cursorX-LABEL_W);wrap.scrollLeft=ts>0?ts+LABEL_W:0}queueRender()},{passive:false});new ResizeObserver(queueRender).observe(wrap);
$('#zi').onclick=()=>{pps=Math.min(pps*1.5,500);queueRender()};$('#zo').onclick=()=>{pps=Math.max(pps/1.5,5);queueRender()};$('#viewmode').onchange=event=>{MODE=event.target.value;queueRender()};$('#followmode').onchange=()=>queueRender();$('#audiosrc').onchange=function(){const time=au.currentTime,playing=!au.paused,source=SOURCES[+this.value];if(!source)return;au.setAttribute('src',dataUrl(source.path));au.load();au.addEventListener('loadedmetadata',()=>{au.currentTime=time;if(playing)au.play()},{once:true})};document.addEventListener('keydown',event=>{if(['INPUT','TEXTAREA','SELECT'].includes(event.target.tagName))return;if(event.code==='Space'){event.preventDefault();au.paused?au.play():au.pause()}else if(event.code==='ArrowLeft')au.currentTime=Math.max(0,au.currentTime-2);else if(event.code==='ArrowRight')au.currentTime=Math.min(dur,au.currentTime+2);else if(event.code==='Escape'){SEL=null;queueRender()}else if(event.code==='KeyE'&&DATA){const parts=[];DATA.tracks.forEach((t,i)=>{if(t.ust)parts.push({track:t.name,lane_offset:+(OFFS[i]||0).toFixed(3),phrase_offsets:Object.fromEntries(Object.entries(LOFFS[i]||{}).filter(([,v])=>Math.abs(v)>1e-3).map(([k,v])=>[k,+v.toFixed(3)]))})});if(parts.length){const txt=JSON.stringify({song:activeId,ust:parts});if(navigator.clipboard&&navigator.clipboard.writeText){navigator.clipboard.writeText(txt).then(()=>{status.textContent='UST 오프셋 JSON 복사됨'})}else{console.log(txt);status.textContent='UST 오프셋 콘솔 출력'}}}});
function dataUrl(path){return new URL(`data/${path}`,location.href).href}
function loadSong(videoId){if(!videoId||videoId===activeId)return;const apply=()=>{const song=window.__SONG_DATA__[videoId];if(!song){status.textContent=`Could not load ${videoId}`;status.className='error';return}activeId=videoId;DATA=song;dur=Math.max(Number(song.duration)||0,...song.tracks.flatMap(track=>[...track.lines,...track.segs].map(item=>item.end)),1);SOURCES=(song.audio&&song.audio.sources)||[];const srcSel=$('#audiosrc');srcSel.replaceChildren();SOURCES.forEach((source,i)=>{const option=document.createElement('option');option.value=i;option.textContent=source.name;srcSel.append(option)});srcSel.value='0';au.pause();if(SOURCES.length)au.setAttribute('src',dataUrl(SOURCES[0].path));au.load();$('#page-title').textContent=`[${song.stratum}] ${song.title} · Karaoke alignment review`;OFFS={};LOFFS={};sortTracksByUst(song);visible=song.tracks.map(()=>true);song.tracks.forEach((t,i)=>{if(t.ust){const v=parseFloat(localStorage.getItem('ustoff:'+videoId+':'+t.name));if(Number.isFinite(v)&&v)OFFS[i]=v;try{const m=JSON.parse(localStorage.getItem('ustloff:'+videoId+':'+t.name)||'null');if(m&&typeof m==='object'&&Object.keys(m).length)LOFFS[i]=m}catch(e){}t._segLine=t.segs.map(s=>{let j=lowerBound(t.lines,s.start+1e-4)-1;return j<0?0:j})}});renderTrackToggles();status.textContent=song.no_segs_tracks?`${song.no_segs_tracks} track(s) without syllables`:'Syllable spans available';status.className=song.no_segs_tracks?'warn':'tag';wrap.scrollLeft=0;wrap.scrollTop=0;PAGE_TS=0;queueRender()};if(window.__SONG_DATA__[videoId]){apply();return}const script=document.createElement('script');script.src=`data/${encodeURIComponent(videoId)}.js`;script.onload=apply;script.onerror=()=>{status.textContent=`Failed to load data/${videoId}.js`;status.className='error'};document.head.append(script)}
filter.addEventListener('input',fillSelector);$('#ust-only').addEventListener('change',()=>{try{localStorage.setItem('ustonly',$('#ust-only').checked?'1':'')}catch(e){}fillSelector()});try{$('#ust-only').checked=!!localStorage.getItem('ustonly')}catch(e){}select.addEventListener('change',()=>loadSong(select.value));au.addEventListener('loadedmetadata',()=>{if(Number.isFinite(au.duration)&&au.duration>0){dur=au.duration;queueRender()}});function tick(){pos.textContent=`${stamp(au.currentTime)} / ${stamp(dur)} · ${pps.toFixed(0)} px/s`;if(DATA&&!au.paused&&$('#followmode').value!=='off'){wrap.scrollLeft=RENDER_TS>0?RENDER_TS+LABEL_W:0}queueRender();requestAnimationFrame(tick)}fillSelector();const wanted=new URLSearchParams(location.search).get('song');if(wanted&&[...select.options].some(option=>option.value===wanted))select.value=wanted;if(select.value)loadSong(select.value);tick();
</script></body></html>'''


def _srt_mapping(specs: list[str], song_ids: list[str] | None) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for spec in specs:
        if "=" in spec:
            video_id, value = spec.split("=", 1)
            mapping[video_id.strip()] = Path(value).expanduser()
        elif song_ids and len(song_ids) == 1:
            mapping[song_ids[0]] = Path(spec).expanduser()
        else:
            raise SystemExit("--srt must be VIDEO_ID=PATH unless one --song is selected.")
    for video_id, path in mapping.items():
        resolved = path if path.is_absolute() else (REPO / path).resolve()
        if not resolved.is_file():
            raise SystemExit(f"SRT file not found for {video_id}: {resolved}")
        mapping[video_id] = resolved
    return mapping


def _ust_mapping(specs: list[str], song_ids: list[str] | None) -> dict[str, list[tuple[Path, float]]]:
    """--ust "[VIDEO_ID=]PATH[@OFFSET]" — 곡당 여러 UST(메인·하모리) 허용."""
    mapping: dict[str, list[tuple[Path, float]]] = {}
    for spec in specs:
        if "=" in spec:
            video_id, value = spec.split("=", 1)
            video_id = video_id.strip()
        elif song_ids and len(song_ids) == 1:
            video_id, value = song_ids[0], spec
        else:
            raise SystemExit("--ust must be VIDEO_ID=PATH[@OFFSET] unless one --song is selected.")
        offset = 0.0
        if "@" in value:
            candidate, _, tail = value.rpartition("@")
            try:
                offset = float(tail)
                value = candidate
            except ValueError:
                pass  # 경로에 @가 든 경우 — 오프셋 아님
        path = Path(value).expanduser()
        resolved = path if path.is_absolute() else (REPO / path).resolve()
        if not resolved.is_file():
            raise SystemExit(f"UST file not found for {video_id}: {resolved}")
        mapping.setdefault(video_id, []).append((resolved, offset))
    return mapping


def _ust_shifts_mapping(specs: list[str], song_ids: list[str] | None) -> dict[str, dict[str, dict]]:
    """--ust-shifts "[VIDEO_ID=]PATH" — 뷰어 E키로 내보낸 오프셋 JSON을 베이크용으로 읽는다."""
    mapping: dict[str, dict[str, dict]] = {}
    for spec in specs:
        if "=" in spec:
            video_id, value = spec.split("=", 1)
            video_id = video_id.strip()
        elif song_ids and len(song_ids) == 1:
            video_id, value = song_ids[0], spec
        else:
            raise SystemExit("--ust-shifts must be VIDEO_ID=PATH unless one --song is selected.")
        path = Path(value).expanduser()
        resolved = path if path.is_absolute() else (REPO / path).resolve()
        if not resolved.is_file():
            raise SystemExit(f"UST shifts file not found for {video_id}: {resolved}")
        data = json.loads(resolved.read_text(encoding="utf-8"))
        for entry in data.get("ust") or []:
            core = re.sub(r"\s*\(δ[^)]*\)\s*$", "", str(entry.get("track") or ""))
            core = re.sub(r"^UST · ", "", core)
            core = re.sub(r"\s*\(소절보정 \d+\)\s*$", "", core)
            mapping.setdefault(video_id, {})[core] = {
                "lane": float(entry.get("lane_offset") or 0.0),
                "phrases": {int(k): float(v) for k, v in (entry.get("phrase_offsets") or {}).items()},
            }
    return mapping


def generate_song_data(
    song: dict,
    srt_path: Path | None = None,
    ust_specs: list[tuple[Path, float]] | None = None,
    ust_shifts: dict[str, dict] | None = None,
) -> tuple[Path, dict] | None:
    """Write one local script assignment; no media/data is copied or downloaded."""
    root = _song_root(song)
    if root is None:
        return None
    audio = _audio_paths(root)
    if audio is None:
        return None
    tracks = build_tracks(song, srt_path, ust_specs, ust_shifts)
    latest = max((item["end"] for track in tracks for item in track["lines"] + track["segs"]), default=0.0)
    relative_root = root.relative_to(BENCH / "results").as_posix()
    payload = {
        "video_id": song["video_id"],
        "stratum": str(song.get("stratum") or song.get("language") or "unknown"),
        "title": str(song.get("title") or song["video_id"]),
        "quality_score": float(song.get("quality_score") or 0.0),
        "duration": max(float(song.get("duration_est") or 0.0), latest, 1.0),
        "audio": {
            "sources": [
                {"name": source["name"], "path": f"../{relative_root}/{source['path']}"}
                for source in audio
            ]
        },
        "tracks": tracks,
        "no_segs_tracks": sum(1 for track in tracks if track.get("no_segs")),
    }
    truth = _ust_truth_anchors(song["video_id"], tracks)
    if truth:
        payload["ust_truth"] = truth
    output = BENCH / "results" / "data" / f"{song['video_id']}.js"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "window.__SONG_DATA__ = window.__SONG_DATA__ || {};\n"
        f'window.__SONG_DATA__["{song["video_id"]}"] = {_json_for_script(payload)};\n',
        encoding="utf-8",
    )
    # 곡 폴더의 구세대 viewer.html을 SPA 리다이렉트로 교체 — 낡은 뷰어를 열 길을 없앤다
    prefix = "../" * len(root.relative_to(BENCH / "results").parts)
    target = f"{prefix}viewer.html?song={song['video_id']}"
    (root / "viewer.html").write_text(
        f'<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0; url={target}">'
        f'<title>Redirecting to karaoke viewer</title><a href="{target}">Open single-page viewer</a>',
        encoding="utf-8",
    )
    return output, {
        "video_id": payload["video_id"], "stratum": payload["stratum"], "title": payload["title"],
        "quality_score": payload["quality_score"], "duration": payload["duration"],
        "data_path": f"data/{song['video_id']}.js", "span_count": sum(len(t["segs"]) for t in tracks),
        "has_syllable_spans": any(track["segs"] for track in tracks),
        "no_segs_tracks": payload["no_segs_tracks"],
        "subtitle_tracks": sum(1 for track in tracks if track.get("line_reference")),
        "ust_tracks": sum(1 for track in tracks if track.get("ust")),
    }


def _read_existing_index() -> dict[str, dict]:
    path = BENCH / "results" / "data" / "index.js"
    if not path.is_file():
        return {}
    match = re.fullmatch(r"\s*window\.__SONG_INDEX__\s*=\s*(.*?)\s*;\s*", path.read_text(encoding="utf-8"), re.S)
    if match is None:
        return {}
    try:
        entries = json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}
    return {entry["video_id"]: entry for entry in entries if isinstance(entry, dict) and entry.get("video_id")}


def write_outputs(entries: dict[str, dict]) -> tuple[Path, Path, Path]:
    results = BENCH / "results"
    viewer = results / "viewer.html"
    viewer.parent.mkdir(parents=True, exist_ok=True)
    viewer.write_text(SINGLE_PAGE_HTML, encoding="utf-8")
    ordered = sorted(entries.values(), key=lambda entry: (entry["stratum"], entry["title"].casefold()))
    index_data = results / "data" / "index.js"
    index_data.parent.mkdir(parents=True, exist_ok=True)
    index_data.write_text(f"window.__SONG_INDEX__ = {_json_for_script(ordered)};\n", encoding="utf-8")
    old_index = results / "index.html"
    old_index.write_text(
        "<!doctype html><meta charset=\"utf-8\"><meta http-equiv=\"refresh\" content=\"0; url=viewer.html\">"
        "<title>Redirecting to karaoke viewer</title><a href=\"viewer.html\">Open karaoke viewer</a>",
        encoding="utf-8",
    )
    return viewer, index_data, old_index


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate an offline single-page karaoke review viewer.")
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--song", action="append", metavar="VIDEO_ID", help="Regenerate one song (repeatable).")
    selector.add_argument("--all", action="store_true", help="Generate data for every eval-set song with audio.")
    parser.add_argument("--eval-set", default=str(BENCH / "eval_set.json"))
    parser.add_argument("--srt", action="append", default=[], metavar="[VIDEO_ID=]PATH", help="Add caption mapping (repeatable).")
    parser.add_argument("--ust", action="append", default=[], metavar="[VIDEO_ID=]PATH[@OFFSET]",
                        help="Add UTAU UST reference lane (repeatable; OFFSET seconds shifts UST onto the audio).")
    parser.add_argument("--ust-shifts", action="append", default=[], metavar="[VIDEO_ID=]PATH",
                        help="Bake viewer-exported (E key) per-phrase offset JSON into the UST lanes.")
    args = parser.parse_args()
    eval_set = json.loads(Path(args.eval_set).read_text(encoding="utf-8"))
    all_songs = list(eval_set.get("songs") or [])
    by_id = {song.get("video_id"): song for song in all_songs}
    selected_ids = list(dict.fromkeys(args.song or []))
    if args.all:
        songs = all_songs
    else:
        missing = [video_id for video_id in selected_ids if video_id not in by_id]
        if missing:
            raise SystemExit(f"Song not found in eval set: {', '.join(missing)}")
        songs = [by_id[video_id] for video_id in selected_ids]
    srt_by_song = _srt_mapping(args.srt, selected_ids or None)
    ust_by_song = _ust_mapping(args.ust, selected_ids or None)
    shifts_by_song = _ust_shifts_mapping(args.ust_shifts, selected_ids or None)
    entries = {} if args.all else _read_existing_index()
    generated = 0
    for song in songs:
        video_id = song["video_id"]
        if video_id in VIEWER_EXCLUDE_SONGS:
            print(f"오입력 확정 곡 — 뷰어에서 제외: {video_id}")
            continue
        result = generate_song_data(song, srt_by_song.get(video_id), ust_by_song.get(video_id),
                                    shifts_by_song.get(video_id))
        if result is None:
            print(f"Skipped (audio/artifacts missing): {video_id}")
            continue
        output, entry = result
        entries[video_id] = entry
        generated += 1
        print(f"{output} spans={entry['span_count']}")
    entries = {vid: entry for vid, entry in entries.items() if vid not in VIEWER_EXCLUDE_SONGS}
    viewer, index_data, old_index = write_outputs(entries)
    print(viewer)
    print(index_data)
    print(old_index)
    print(f"data={generated} skipped={len(songs) - generated}")
    return 0 if generated else 1


if __name__ == "__main__":
    raise SystemExit(main())
