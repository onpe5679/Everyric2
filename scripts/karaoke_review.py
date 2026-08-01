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
#
# 각도는 «계열이 구역으로 읽히게»가 아니라 **지각 거리(ΔE)로** 깔았다. 같은 간격으로 벌려도
# 눈에는 안 갈린다 — 초록~청록 구간은 둔하고 빨강~노랑은 민감해서, 19°씩 고르게 벌린 배치가
# 실측 ΔE 9.6까지 붙는 쌍을 남겼다(2pass-en-ipa ↔ -ipa-hangul).
#
# 계열 소속감은 **세그 바 모양**(`_track_shape`: 단독 민바 / 2패스 흰 캡 / 라우팅 테두리)이
# 진다. 그래서 색은 «같은 모양 안에서 유일할 것»만 책임지면 되고, 그 완화 덕에 그룹당 색
# 공간이 넓어져 같은 모양 쌍은 전부 ΔE 15 이상으로 벌었다. 모양이 다른 쌍은 최소 ΔE 12.7까지
# 붙지만 모양이 갈라 준다(단, 완전히 겹치면 「같은 것의 변형」으로 오독되므로 면제는 아니다 —
# 좁은 세그 바에서는 장식이 생략된다).
#
# 어댑터를 추가할 때는 빈 각도를 눈대중으로 고르지 말 것. 같은 모양 그룹 안에서 ΔE가 최대가
# 되는 자리를 찾아야 한다(고정 5종은 사용자에게 익은 색이라 움직이지 않는다).
ALIGNER_HUE = {
    "2pass-owsm-omniasr": 4 / 360,
    # ja 독음 쌍 — 짝끼리 나란히 보도록 가까이 두되 2pass-owsm-omniasr(4°)와는 띄운다.
    "2pass-owsm-reading": 64 / 360,        # 2패스 · ja 가나 독음 + n-best 심판(기각판)
    "2pass-owsm-reading-noref": 80 / 360,  # 2패스 · 표기형 독음
    "2pass-owsm-reading-joshi": 96 / 360,  # 2패스 · 조사만 발음형
    "2pass-owsm-reading-phon": 108 / 360,  # 2패스 · 전체 발음형
    # ★심판 비교 쌍 — 짝끼리 붙여 둔다
    "2pass-owsm-prod": 16 / 360,
    "2pass-owsm-mixed": 120 / 360,         # 2패스 · ★혼합 표기(ja+라틴+한글)
    "2pass-owsm-mixed-hangul": 156 / 360,  # 2패스 · ★한글 표시층           # 2패스 · ★프로드 독음 + 심판
    "2pass-owsm-prod-noref": 28 / 360,     # 2패스 · 같은 독음, 심판 끔       # 2패스
    "nemo-nfa": 25 / 360,                # 단독 · 고정(현재 뷰어 제외 — 자리는 비워 둔다)
    "2pass-en-hangul": 36 / 360,         # 2패스 · 철자 기반 음차(뷰어 제외 — 자리는 비워 둔다)
    # ★원문 영어 층 — 2pass-asr-ipa-hangul과 **같은 정렬**을 원문 철자 음절로 묶은 레인.
    # 제외된 2pass-en-hangul 자리를 물려받는다(en 전용 경로라 같은 곡에 같이 뜨지 않는다).
    "2pass-asr-ipa-en": 36 / 360,        # 2패스 · ★en 원문 음절 스팬
    # 가나 레인이 비운 자리(204°)를 IPA 전사 레인이 물려받는다.
    "2pass-asr-ipa-phonetic": 204 / 360,  # 2패스 · ★IPA 전사(음절 단위)
    # 심판 대조군은 각자의 짝과 **나란히 놓고 비교**하는 레인이라 짝의 색에서 조금만 띄운다
    # — 계열이 갈리면 «다른 실험»으로 읽혀 대조가 안 된다(hangul 308°↔292°, en 36°↔20°).
    "2pass-asr-ipa-hangul-noref": 292 / 360,  # 2패스 · 심판 끈 대조군(한글)
    "2pass-asr-ipa-en-noref": 20 / 360,       # 2패스 · 심판 끈 대조군(영어 원문)
    "2pass-asr-ipa-en-energy": 28 / 360,      # 2패스 · 강도 봉우리 켠 대조군(36°와 20° 사이)
    "routed-2mode+pp": 46 / 360,         # 라우팅
    # ★en 채택 후보 — IPA 정렬을 **앵커 없이** 단독으로 한다. owsm 2패스 대비 정렬 22배 빠르고
    # MAE는 절반(0.08 vs 0.15)이라 앵커가 필요 없음이 실측으로 드러났다.
    "omniasr-ctc-ipa-hangul": 56 / 360,  # 단독 · IPA 정렬 → 한글 표시
    "2pass-owsm-reazon": 72 / 360,       # 2패스
    "routed-2mode": 90 / 360,            # 라우팅 · 기준 경로
    # 58°에 뒀다가 polar 분리 밝기에서 2pass-owsm-reazon(72°)과 ΔE 9.2까지 붙었다.
    # 대표 밝기(nosep) 하나로만 자리를 고르면 이렇게 다른 분리기에서 충돌한다 — 전 밝기·채도
    # 조합을 훑어 다시 고른 자리가 112°다.
    "hf-slplab-phone-mfa": 112 / 360,    # 단독 · 음소 43 vocab
    "2pass-en-ipa-hangul": 134 / 360,    # 2패스 · ★IPA 정렬 → 한글 표시
    "owsm-ctc-v4-1b-bf16": 150 / 360,    # 단독 · 고정(초록 — 앵커 현역)
    "2pass-en-ipa": 162 / 360,           # 2패스 · IPA 음소 그대로(묶기 전)
    "owsm-ctc-v4-1b": 172 / 360,         # 단독 · 고정(청록 — 구세대 fp32)
    "2pass-owsm-kkonjeong": 188 / 360,   # 2패스
    "routed-2mode-nosanchor": 198 / 360,  # 라우팅
    "2pass-asr-ipa-kana": 204 / 360,     # 2패스 · ★en 채택판의 가나 표시
    "omniasr-ctc": 210 / 360,            # 단독 · 고정(파랑 — 다국어 주력)
    "2pass-en-cmu": 218 / 360,           # 2패스 · CMU → 한글 음차
    "routed-2mode-safe": 226 / 360,      # 라우팅 · 임계 −11.0
    "hf-reazon-hubert-base": 236 / 360,  # 단독 · ja 네이티브
    "2pass-en-kana": 272 / 360,          # 2패스 · 철자 기반 음차
    "routed-2mode-lang": 296 / 360,      # 라우팅 · ★언어 배선(en 강제 구원)
    # ★en 최종 채택 — asr이 1패스로 라인 창을 잡고 2패스에서 그 창 안만 IPA로 음절 분리.
    # owsm 앵커판(2pass-en-ipa-*)보다 정렬 12배 빠르고 MAE 절반이다.
    "2pass-asr-ipa-hangul": 308 / 360,   # 2패스 · ★en 채택
    "hf-kkonjeong": 320 / 360,           # 단독 · 고정(자홍 — ko CTC)
    "2pass-en-ipa-kana": 332 / 360,      # 2패스 · ★IPA 정렬 → 가나 표시
    "routed-2mode-lang-kana": 344 / 360,  # 라우팅 · 위의 가나 표시판
    "omniasr-ctc-ipa-kana": 352 / 360,   # 단독 · 위와 같은 정렬, 표시만 가나
}
# 레인 왼쪽 세로 띠 — **연산 깊이**를 색으로. 분리기 종류를 구분하던 자리인데, 후보가
# polar 하나로 좁혀지면서 그 축이 «분리했나 안 했나»만 남았고, 실제로 읽고 싶은 것은
# 「이 레인이 시간을 얼마나 들이부었나」이기 때문이다(사용자 지시 2026-08-01).
#
# 계단을 가르는 축은 **분리 여부**와 **owsm 앵커 여부** 둘이다. 「2패스인가」로 가르면 안 된다
# — 같은 2패스라도 앵커가 자기 자신(asr)이면 +0.7~3.2초지만 owsm이면 +17초다. 실측(en 5곡,
# 정렬 시간 · polar 분리는 +10초):
#   무분리+단독 0.9 · 무분리+asr앵커 4.1  │ 분리+단독 10.8 · 분리+asr앵커 11.5 · 무분리+owsm 16.3 │ 분리+owsm 28.1
#   └─────── 1 얕음 ───────┘              └──────────────── 2 중간 ────────────────┘              └─ 3 깊음 ─┘
# ja는 계단마다 오르고(극한 음절 47.4 → 73.5 → 79.5) en은 2단계에서 포화한다 — 그 차이를
# 눈으로 보라고 색을 나눈다.
DEPTH_BAND = {
    1: "#4fb87a",  # 초록 — 싸다
    2: "#e0a03c",  # 주황
    3: "#d9534f",  # 빨강 — 비싸다
}


def _depth_band(separator: str, aligner: str, run: dict) -> str:
    """이 레인이 실제로 지른 연산 깊이 → 띠 색.

    라우팅 레인은 **곡마다 경로가 갈리므로** 설정이 아니라 그 곡에서 실제로 탄 경로를 본다
    (같은 레인이 어떤 곡에서는 초록, 어떤 곡에서는 빨강으로 뜬다 — 그게 라우팅의 요점이다).
    """
    meta = run.get("align_meta") or {}
    routing = meta.get("routing") or {}
    if routing.get("route"):
        if routing["route"] == "fast":
            return DEPTH_BAND[1]
        separated = bool(routing.get("rescue_separator"))
        heavy_anchor = "owsm" in str(routing.get("rescue_aligner") or "")
    else:
        separated = separator not in ("", "nosep")
        heavy_anchor = "owsm" in str((meta.get("two_pass") or {}).get("anchor") or "")
    if separated and heavy_anchor:
        return DEPTH_BAND[3]
    if separated or heavy_anchor:
        return DEPTH_BAND[2]
    return DEPTH_BAND[1]
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


def _track_shape(aligner: str) -> str:
    """세그 바 장식 힌트 — 색 축 하나로는 20개 넘는 레인을 못 가르므로 모양 축을 하나 더 준다.

    구조가 다른 세 부류(단독 1패스 / 2패스 / 라우팅)를 갈라, 색이 비슷하게 보여도
    «어느 계열인지»는 항상 읽히게 한다. 빈 문자열이면 장식 없음(단독 1패스 현행 그대로).
    """
    base = aligner.partition("@")[0]
    if base.startswith("routed-"):
        return "routed"
    if base.startswith("2pass-"):
        return "twopass"
    return ""


def _song_language(song: dict) -> str:
    """곡의 기본 언어 — 뷰어가 stratum을 우선 읽는 규칙(payload["stratum"])과 같은 순서로 본다.

    eval_set의 ``ja_mms`` 같은 라벨은 접미를 벗겨야 언어로 쓸 수 있다(benchmark_alignment의
    ``base_language``와 같은 규칙 — 이 스크립트는 그 모듈을 임포트하지 않으므로 여기 둔다).
    """
    lang = str(song.get("stratum") or song.get("language") or "").strip()
    return lang[: -len("_mms")] if lang.endswith("_mms") else lang


def _compact_by_default(separator: str, aligner: str, language: str) -> bool:
    """기본으로 절반 높이(참고용)로 내려 둘 레인 — 결론이 난 비교는 «있되 자리를 덜 먹게» 둔다.

    숨기기가 아니라 단축인 이유: 체크박스를 끈 채 시작해 봤더니 «안 보이면 없는 것과 같다»는
    판정이었다(사용자, 2026-08-01). 결론의 근거는 화면에 남아 있어야 그 결론을 되짚을 수 있다.
    """
    base = aligner.partition("@")[0]
    if base == "routed-2mode+pp":
        # VAD 보정층 이식은 세그를 깨고 라인 지표도 낮춰 기각됐다(「VAD 보정층 이식 — 기각」).
        return True
    if base == "routed-2mode":
        # 임계 −11.5판. 2026-08-01에 −11.0(routed-2mode-safe)으로 확정돼 비교 대상 자리가 끝났다.
        return True
    if language == "en":
        if separator.startswith("bs-polarformer") and not base.startswith("2pass-asr-ipa-"):
            # en 채택은 **분리 + asr 자기앵커 창 + IPA**다. 그 조합만 주력으로 남기고 같은
            # 분리기의 나머지 레인은 참고용으로 내린다.
            return True
    # owsm 앵커를 붙인 en 2패스는 값을 못 한다 — 세 구성 대조(2026-08-01)에서 앵커를 빼면
    # 정렬이 22배 빨라지는데(18.1초 → 0.8초) MAE는 오히려 절반이다(0.15 → 0.08). 앵커의 원래
    # 근거였던 「라틴은 CTC가 약해 라인이 무너진다」가 타깃을 IPA로 바꾸면서 사라졌다.
    return base.startswith("2pass-en-")


def _drop_lane(aligner: str, language: str) -> bool:
    """뷰어에 아예 안 그릴 레인. 단축(_compact_by_default)과의 경계가 있다 —

    결론의 근거로 **되짚을 일이 남았으면** 단축, 그 근거를 다른 레인이 이미 더 잘 보여주면
    제거다. 단축은 자리를 절반으로 줄일 뿐 레인 수는 그대로라, 「레인이 너무 많다」는 문제
    자체는 단축으로 안 풀린다(사용자, 2026-08-01).
    """
    # 지금은 VIEWER_ALIGNERS에서 거르는 것으로 충분해 비어 있다. 레인별(정렬기 이름만으로는
    # 못 가르는) 제거가 다시 필요해지면 여기에 조건을 둔다 — 예전에 창 없는 IPA 대조군과
    # 음차 1·2세대를 여기서 뺐고, 지금은 그 정렬기들이 아예 후보 목록에 없다.
    del aligner, language
    return False


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
# 끄기(사용자 지시 2026-08-01) — 레인이 많아 시인성을 깎는데, 융합은 2모드 라우팅 채택으로
# 후보 자리에서 내려왔다. 세 레인을 **전부** 돌려야 나오는 합성이라 곡당 4.3초짜리 라우팅과는
# 비용 비교 자체가 성립하지 않는다. 계산 로직은 그대로 두므로 다시 볼 일이 생기면 True로 켠다.
SHOW_FUSED_LANE = False


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
        # 융합은 세 레인을 전부 돌려야 나오므로 가장 깊은 칸이다(현재 꺼져 있다).
        "band": DEPTH_BAND[3],
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


# 2패스 레인 표시용 짧은 모델명 — 레인 라벨은 폭이 좁아 정식 어댑터명이 다 안 들어간다.
_SHORT_MODEL = {
    "owsm-ctc-v4-1b-bf16": "owsm",
    "owsm-ctc-v4-1b": "owsm-fp32",
    "omniasr-ctc": "omniasr",
    "hf-reazon-hubert-base": "reazon",
    "hf-kkonjeong": "kkonjeong",
}


def _lane_label(aligner: str, run: dict) -> str:
    """레인 표시명 — 2패스는 «입력 표기»와 «경량 모델이 실제로 본 표기»가 다를 수 있다.

    `@kana` 같은 접미사는 하네스가 **입력 모드**에 붙이는 라벨이라 어댑터 바깥의 소스 가사만
    가리킨다. 2패스는 그 안에서 경량 모델 몫만 다시 음차하는데, ko 모델은 애초에 가나를 못
    먹으므로(kkonjeong vocab 54개 중 가나 토큰 0개 — 그대로 넣으면 정렬 타깃이 0개다) 이름만
    보면 "kkonjeong이 가나로 정렬했다"로 읽힌다. 실제 표기를 화살표로 덧붙여 그 오독을 막는다.

    판정은 설정값이 아니라 **실측 변환 줄 수**(``script_converted_lines``)로 한다. 같은 hangul
    설정도 ko 곡에서는 변환이 항등이라(입력이 이미 한글) 붙이면 거짓이 되기 때문이다.
    그 필드가 없는 구식 런은 설정값 대조로 폴백한다.
    """
    meta = run.get("align_meta") or {}
    # 라우팅 레인은 **곡마다 다른 경로**를 탄다. 어느 쪽으로 갔는지가 그 곡 결과를 읽는
    # 전제이므로 레인 이름에 드러낸다(fast=무분리 ASR, rescue=분리+앵커 2패스).
    routing = meta.get("routing") or {}
    if routing.get("route"):
        score = routing.get("line_log_conf_median")
        return f"{aligner} [{routing['route']}{'' if score is None else f' {score:g}'}]"
    two_pass = meta.get("two_pass") or {}
    if not two_pass:
        return aligner
    label = aligner
    script = two_pass.get("refiner_script")
    # 어댑터 이름이 이미 그 표기를 말하고 있으면(2pass-en-hangul ← latin-hangul) 덧붙이지 않는다.
    if script and script != "native" and script.removeprefix("latin-") not in aligner:
        converted = two_pass.get("script_converted_lines")
        if converted is None:  # 구식 런 — 입력 접미사와 설정값이 다르면 변환된 것으로 본다
            converted = script != aligner.partition("@")[2]
        if converted:
            label = f"{label}→{script}"
    # 어떤 «모델 조합»인지는 레인 이름만으로 알 수 없는 경우가 있다(2pass-en-kana는 음차
    # 방식만 말한다). 앵커·경량 모델을 짧은 이름으로 덧붙여 조합을 항상 읽히게 한다.
    pair = " + ".join(
        _SHORT_MODEL.get(m, m) for m in (two_pass.get("anchor"), two_pass.get("refiner")) if m
    )
    if pair and not all(part in aligner for part in pair.split(" + ")):
        label = f"{label} [{pair}]"
    return label


# 뷰어 레인은 베이스라인 + 현역 후보만 — 탈락 후보(qwen3, 구 hf 계열, owsm 실험 변형)는
# 런 캐시는 남기되 표시에서 뺀다. 기준은 combo 디렉터리명의 base 정렬기(@suffix 제거).
VIEWER_ALIGNERS = {
    # ── 기준선 ──────────────────────────────────────────────────────────────
    "omniasr-ctc",
    "owsm-ctc-v4-1b-bf16",
    # ── ja 채택 스택 ────────────────────────────────────────────────────────
    "2pass-owsm-omniasr",   # 다국어 단일 경로(ja 17곡 음절 86.7%)
    "2pass-owsm-mixed",     # ★프로드 독음 + 라틴 음절화 + 장음 + 심판
    "2pass-owsm-mixed-hangul",
    # 라인 클램프층(간주 좌초 스냅 제외) — 병적 라인 절단 + 소절 끝 늘임음 연장.
    "2pass-owsm-mixed+pp",  # ★위와 정렬이 같고 표시만 한글 — 한국어 사용자층
    # ── en 채택 스택 ────────────────────────────────────────────────────────
    # 같은 정렬(ASCII 음소 타깃 + asr 자기 라인 창 + 심판)을 세 해상도로 본다.
    "2pass-asr-ipa-en",         # 원문 음절
    "2pass-asr-ipa-hangul",     # 한글 발음
    "2pass-asr-ipa-phonetic",   # 음소 전사 자체
    # ── 라우팅(진행 중) ─────────────────────────────────────────────────────
    "routed-2mode",
    "routed-2mode+pp",
    "routed-2mode-safe",
    "routed-2mode-lang",
}

# 판정이 끝나 뷰어에서 내린 레인 — 이름과 이유만 남긴다. 위 집합으로 옮기면 즉시 돌아온다.
RETIRED_ALIGNERS = {
    "mms-baseline": "하차 확정",
    "owsm-ctc-v4-1b": "bf16이 대체(품질 동일·VRAM 절반)",
    "2pass-owsm-reading": "조사/발음형 축 — 차이 없음으로 기각(71.91/71.80/72.41)",
    "2pass-owsm-reading-noref": "위와 같은 축",
    "2pass-owsm-reading-joshi": "위와 같은 축",
    "2pass-owsm-reading-phon": "위와 같은 축",
    "2pass-owsm-prod": "2pass-owsm-mixed가 대체(라틴 음절화 추가)",
    "2pass-owsm-prod-noref": "ja 심판 채택 확정 — 청취 6/6",
    "2pass-owsm-mixed-nolong": "장음 축 — 중립 확정",
    "2pass-owsm-mixed-en": "라틴 낱말 심판 — color 12건 오답으로 기각",
    "2pass-owsm-mixed+pp": "프로드 라인 보정층 — 병적 절단 규칙이 437줄 중 0회 발동(무효)",
    "2pass-asr-ipa-hangul-noref": "en 심판 채택 확정 — 정답 4/17 → 14/17",
    "2pass-asr-ipa-en-noref": "위와 같음",
    "2pass-asr-ipa-en-energy": "오디오 강도 봉우리 — 14/17 → 11~12/17로 기각",
    "2pass-en-ipa": "owsm 앵커 en 2패스 — asr 자기앵커가 대체(정렬 22배·MAE 절반)",
    "2pass-en-ipa-hangul": "위와 같음",
}

# 표기까지 포함한 레인 단위 제외 — **모국어가 아닌 표기를 강요한 레인**을 뺀다.
# 실측(2026-08-01): 한글 강제는 다국어·ja 네이티브 모델의 라인 정확도를 크게 깎는다
# (owsm 76.0 → 63.9, omniasr 73.0 → 69.6). 같은 독음을 **가나**로 펼치면 반대로 오르므로
# (owsm 79.7, omniasr 78.5) 문제는 독음 확장이 아니라 낯선 표기 강요다.
# ko 모델(kkonjeong·slplab·kresnik·hjlee)의 @hangul-local은 모국어 표기이므로 유지한다.
VIEWER_LANE_EXCLUDE = {
    "owsm-ctc-v4-1b-bf16@hangul-local",
    "owsm-ctc-v4-1b-bf16@hangul",
    "owsm-ctc-v4-1b@hangul-local",
    "owsm-ctc-v4-1b@hangul",
    "omniasr-ctc@hangul-local",
    "omniasr-ctc@hangul",
    "hf-reazon-hubert-base@hangul-local",
    "hf-reazon-hubert-base@hangul",
    "mms-baseline@hangul-local",
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


def allin1_sections(song: dict) -> dict | None:
    """allin1 구조 분석 → **배경 구간**. 레인이 아니라 캔버스 전체에 깔리는 세로선이 된다.

    원래는 참조 레인 하나를 차지했는데(간주/솔로 마스킹 가설 눈 검증용), 구간 경계는 «어느
    레인의 값»이 아니라 **모든 레인에 공통인 시간축 눈금**이다. 레인으로 두면 그 한 줄에서만
    보이고 다른 레인의 세그와 세로로 맞춰 보려면 눈이 그 줄까지 오갔다 해야 한다. 세로선으로
    깔면 모든 레인을 한 번에 가로지르므로 「이 붕괴가 간주 구간인가」를 바로 읽는다.
    """
    path = BENCH / "allin1" / f"{song['video_id']}.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"allin1 결과 무시 ({path}): {error}")
        return None
    spans = [
        {"t": round(float(s["start"]), 3), "e": round(float(s["end"]), 3),
         "l": str(s.get("label") or "?")}
        for s in (data.get("segments") or [])
        if s.get("start") is not None and s.get("end") is not None
    ]
    return {"bpm": data.get("bpm"), "spans": spans} if spans else None


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
                    # 준정답 레인은 «맞았는지 대보는» 기준이라 늘 떠 있어야 하지만, 후보 레인처럼
                    # 자세히 들여다볼 일은 없다. 전체 높이를 계속 먹을 값을 못 한다(사용자, 2026-08-01).
                    "compact": True,
                }
            )

    video_id = song["video_id"]
    language = _song_language(song)  # 어느 레인을 참고용으로 내릴지는 곡 언어에 따라 갈린다
    runs: list[tuple[Path, dict]] = []
    # fp32→fp16 대체: fp16 런이 같은 레인을 제공하면 fp32 변형은 숨긴다(결과 동일 실측).
    FP16_REPLACES = {"kimft-melband-fp16": "kimft-melband", "bs-polarformer-fp16": "bs-polarformer"}
    fp16_lanes: dict[str, set[str]] = {}  # fp32 분리기명 → fp16이 커버한 레인들
    for path in sorted((BENCH / "runs").glob(f"*/{video_id}__r1.json")):
        separator, _, aligner = path.parent.name.partition("__")
        if (aligner.partition("@")[0] not in VIEWER_ALIGNERS
                or aligner in VIEWER_LANE_EXCLUDE
                or separator in VIEWER_SEPARATOR_EXCLUDE):
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
    # 정렬기 쪽 정밀도 중복도 같은 규칙 — 같은 분리기·같은 표기에서 저정밀 런이 자리를 채우면
    # fp32 변형은 숨긴다(품질 무회귀 실측 완료). 표기 접미사(@hangul-local 등)까지 맞춰 본다.
    ALIGNER_PRECISION_REPLACES = {"owsm-ctc-v4-1b": "owsm-ctc-v4-1b-bf16"}
    present = {path.parent.name for path, _ in runs}

    def _superseded(dirname: str) -> bool:
        separator, _, lane = dirname.partition("__")
        base, at, suffix = lane.partition("@")
        better = ALIGNER_PRECISION_REPLACES.get(base)
        return bool(better) and f"{separator}__{better}{at}{suffix}" in present

    runs = [(path, run) for path, run in runs if not _superseded(path.parent.name)]
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
        # 신호를 묻지 않는 라우팅(route=forced)은 늘 같은 경로를 타므로 그 구원 레인과
        # **바이트 단위로 같다**(en 5곡 실측: 세그·라인 최대 Δ 0.0000). 레인 하나를 더 그려도
        # 얻는 정보가 없다. fast/rescue는 곡마다 갈리므로 그대로 둔다 — 그게 라우팅의 요점이다.
        if ((run.get("align_meta") or {}).get("routing") or {}).get("route") == "forced":
            continue
        if _drop_lane(aligner, language):
            continue
        # 항상 "분리기 · 정렬기" — 첫 조합만 접두사를 생략하던 규칙은 곡마다 다른 레인이
        # 무접두사 이름을 차지해 뷰어 간 비교를 흐렸다.
        label = _lane_label(aligner, run)
        name = f"{separator} · {label}" if lane else label
        segs = _normalise_segs(seg for line in run["lines"] for seg in (line.get("segs") or []))
        _attach_prod_matches(segs, prod_segs)
        track = {
            "name": name,
            "color": _track_color(separator if lane else "", aligner),
            "band": _depth_band(separator, aligner, run),  # 레인 왼쪽 세로 띠 = 연산 깊이
            "lines": _normalise_lines(run["lines"]),
            "segs": segs,
            "no_segs": not segs,
        }
        # 없을 때가 기본값(장식 없음·전체 높이)이라 해당할 때만 붙인다 — 데이터 파일도 그만큼 짧아진다.
        shape = _track_shape(aligner)
        if shape:
            track["shape"] = shape
        if _compact_by_default(separator, aligner, language):
            track["compact"] = True
        tracks.append(track)
    if SHOW_FUSED_LANE:
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


def _segs_by_line(track: dict) -> list[list[dict]]:
    """트랙의 세그를 라인별로 묶는다. 라인 배열은 입력 가사와 1:1이라 인덱스가 곧 라인 번호다."""
    groups: list[list[dict]] = []
    segs = track.get("segs") or []
    cursor = 0
    for line in track.get("lines") or []:
        group: list[dict] = []
        while cursor < len(segs) and segs[cursor]["start"] < line["end"] + 1e-6:
            if segs[cursor]["start"] >= line["start"] - 0.5:
                group.append(segs[cursor])
            cursor += 1
        groups.append(group)
    return groups


def _ust_truth_anchors(video_id: str, tracks: list[dict]) -> dict | None:
    """UST 레인 → 라인별 정답 시각(텍스트 앵커, ust_line_judge v3와 동일 방법) — 레인 채점용.

    비-harm UST 노트와 PROD 라인 텍스트를 로마자 정규화해 전역 단조 정렬, 라인 문자 40%+
    매칭 시 앵커. 앵커 커버리지 40% 미만(en↔가나 등)이면 채점 무의미라 None.

    **세그 매칭은 라인 안에서만 한다.** 예전에는 곡 전체를 한 줄로 이어 붙여 difflib을
    돌렸는데, 후렴이 반복되는 곡에서 정답이 통째로 다른 후렴으로 미끄러졌다 — 심판이 독음을
    `すきずき`→`すきすき`로 고쳐 문자열이 더 자기유사해지자 0:36 세그의 정답이 **2:01(85초
    밖)** 에 붙었고, 옳은 쪽이 0/13·틀린 쪽이 13/13으로 나왔다(2026-08-02). 라인 앵커는
    라인 텍스트가 길고 구별되므로 전역으로 잡아도 안전하고, 세그 매칭만 그 라인의 노트
    범위로 가두면 미끄러짐이 원천 차단된다.

    세그마다 시작(``truth``)과 **끝**(``truth_end``)을 함께 붙인다. 시작만 보는 지표는
    「세그가 음소 하나 길이로 잘린」 결함을 통과시킨다 — 라틴 세그가 19ms로 잘려 하이라이트가
    음절 중간에 꺼지는데도 시작이 맞아 +3.75pp가 나왔다(2026-08-02).
    """
    notes = []
    for track in tracks:
        if track.get("ust") and "harm" not in track["name"].lower():
            notes += [(seg["start"], seg.get("end"), seg["t"]) for seg in track["segs"]]
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
    n_chars, n_time, n_end, n_note, n_offset = [], [], [], [], [0]
    for index, (start, end, txt) in enumerate(notes):
        r = norm(txt)
        n_chars.append(r)
        n_time += [start] * len(r)
        n_end += [end if end is not None else start] * len(r)
        n_note += [index] * len(r)
        n_offset.append(n_offset[-1] + len(r))
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
    first_note: dict[int, int] = {}
    for i, (a0, a1) in enumerate(l_ranges):
        hits = [char_map[c] for c in range(a0, a1) if c in char_map]
        if len(hits) >= max(2, (a1 - a0) * 0.4):
            anchors[str(i)] = round(n_time[hits[0]], 3)
            first_note[i] = n_note[hits[0]]
    if len(anchors) / max(len(prod["lines"]), 1) < 0.4:
        return None

    # 라인이 덮는 **노트 범위**는 «이 라인의 첫 노트 ~ 다음 라인의 첫 노트»로 자른다.
    # 한때 그 라인에 걸린 hits의 min/max로 잡았는데, 후렴이 반복되는 곡에서 한 라인의 hits가
    # 곡 전체에 흩어져 범위가 수백 노트로 부풀었다(numb numb 정답 1,094 — 실제는 그 1/3).
    # 앵커가 연속으로 잡힌 구간에서만 범위를 준다 — 사이에 앵커 없는 라인이 있으면 그 라인의
    # 노트가 앞 라인 몫으로 섞이므로 아예 채점에서 뺀다.
    ordered = sorted(first_note)
    line_notes: dict[int, tuple[int, int]] = {}
    for position, line_index in enumerate(ordered):
        nxt = ordered[position + 1] if position + 1 < len(ordered) else None
        if nxt is None:
            line_notes[line_index] = (first_note[line_index], len(notes))
        elif nxt == line_index + 1:
            line_notes[line_index] = (first_note[line_index], first_note[nxt])

    # 음절 축 — 후보 세그마다 대응 UST 노트의 시작·끝을 붙인다. 라인 시작만 보는 지표는 BPE
    # 보간 손실(owsm)을 못 잡는다: 라인 시작은 항상 토큰 경계라 오차가 안 나타난다.
    joined = "".join(n_chars)
    for track in tracks:
        if track.get("ust") or track.get("line_reference") or not track.get("segs"):
            continue
        for line_index, group in enumerate(_segs_by_line(track)):
            span = line_notes.get(line_index)
            if not span or not group:
                continue
            c0, c1 = n_offset[span[0]], n_offset[span[1]]
            local = joined[c0:c1]
            if not local:
                continue
            s_ranges, s_parts, pos = [], [], 0
            for seg in group:
                r = norm(seg.get("t", ""))
                s_ranges.append((pos, pos + len(r)))
                s_parts.append(r)
                pos += len(r)
            smap = {}
            matcher = difflib.SequenceMatcher(None, "".join(s_parts), local, autojunk=False)
            for blk in matcher.get_matching_blocks():
                for k in range(blk.size):
                    smap[blk.a + k] = c0 + blk.b + k
            for seg, (a0, a1) in zip(group, s_ranges):
                hits = [smap[c] for c in range(a0, a1) if c in smap]
                if hits:
                    seg["truth"] = round(n_time[min(hits)], 3)
                    seg["truth_end"] = round(n_end[max(hits)], 3)
    # 라인별 **정답 칸 수**. 「몇 개로 나눴나」가 이 작업의 본체인데 시작 시각 지표는 그걸
    # 약하게만 잡는다 — ``numb``을 ``n|u|m|b`` 넷으로 쪼개도 넷 중 하나는 시작이 맞는다.
    # 세그 길이로는 못 잰다: CTC 스팬은 본래 뾰족해서 프로드를 포함한 **모든** 경로가 20ms이고
    # (UST 노트는 219ms) 그건 규약 차이지 결함이 아니다(2026-08-02 실측).
    note_counts = {str(i): span[1] - span[0] for i, span in line_notes.items()}
    return {"anchors": anchors, "lines": len(prod["lines"]), "note_counts": note_counts}


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
// 결론이 난 레인(track.compact)은 절반 높이로 그린다 — 지워 버리면 결론의 근거까지 사라진다.
// 행 높이가 두 가지가 되는 순간 «y ÷ ROW_H»로 행을 찾던 역산이 전부 무효라, 켜진 행의 높이를
// 누적한 경계 배열을 만들어 두고 좌표 변환은 전부 여기를 거친다.
function rowHeight(ti){const t=DATA&&DATA.tracks[ti];return t&&t.compact?ROW_H/2:ROW_H}
let LAY=null;
// 레이아웃이 바뀌는 건 곡 교체와 체크박스 토글뿐이라 그때만 버린다(매 프레임 다시 쌓지 않는다).
function invalidateLayout(){LAY=null}
function layout(){if(LAY)return LAY;const rows=activeRows(),tops=[],hs=[];let y=0;for(const ti of rows){const h=rowHeight(ti);tops.push(y);hs.push(h);y+=h}return LAY={rows,tops,hs,total:y}}
// 마우스 y(스크롤 포함 콘텐츠 좌표) → 트랙 인덱스. 행이 수십 개라 선형 탐색으로 충분하고,
// 범위 밖이면 undefined — 호출부의 `DATA.tracks[ti]` 가드가 예전과 똑같이 걸린다.
function rowAt(y){const l=layout();for(let i=0;i<l.rows.length;i++)if(y>=l.tops[i]&&y<l.tops[i]+l.hs[i])return l.rows[i];return undefined}
function lowerBound(items,time){let low=0,high=items.length;while(low<high){const mid=(low+high)>>1;if(items[mid].start<time)low=mid+1;else high=mid}return low}
function queueRender(){if(!scheduled){scheduled=true;requestAnimationFrame(()=>{scheduled=false;render()})}}
// ALLIN1 구조 구간 — 레인 하나를 먹던 것을 «모든 레인을 가로지르는» 세로선으로 바꿨다.
// 구간 경계는 어느 한 레인의 값이 아니라 시간축 공통 눈금이라, 레인으로 두면 다른 레인의
// 세그와 세로로 맞춰 보려고 눈이 그 줄까지 오갔다 해야 했다.
// pass 0 = 세그 «아래»(선), pass 1 = 세그 «위»(라벨). 선까지 위에 그리면 구간 정보가 정렬
// 결과를 가려 주객이 뒤바뀐다. 라벨은 반대로 가려지면 못 읽으니 위로 올린다.
function drawSections(pass,viewStart,viewEnd,timeScroll,width,height){const S=DATA.sections;if(!S)return;if(pass)ctx.font='9px system-ui';for(const sp of S.spans){if(sp.e<viewStart||sp.t>viewEnd)continue;const x=LABEL_W+sp.t*pps-timeScroll;if(x>width)continue;if(!pass){if(x<LABEL_W)continue;
// 노래 구간은 진하게, 비가창 구간(intro/inst/outro)은 흐리게 — 「이 붕괴가 간주에 걸렸나」를 색으로 읽는다.
ctx.strokeStyle=/verse|chorus|bridge/i.test(sp.l)?'#7bd88f55':'#7bd88f22';ctx.beginPath();ctx.moveTo(x+.5,0);ctx.lineTo(x+.5,height);ctx.stroke()}else{const lx=Math.max(LABEL_W+2,x+3);if(lx>width-10)continue;const w=ctx.measureText(sp.l).width;
// 10초 눈금(y=11) 아래 줄에 둔다. 첫 레인 위에 겹치므로 배경을 깔지 않으면 세그와 섞여 못 읽는다.
ctx.fillStyle='#14161ad0';ctx.fillRect(lx-2,14,w+4,11);ctx.fillStyle='#7bd88f';ctx.fillText(sp.l,lx,23)}}}
// 세그 바 장식(track.shape): 2패스=윗변 2px 밝은 캡, 라우팅=어두운 테두리, 단독=민바.
// 바는 2px까지 좁아지므로 «색을 통째로 덮지 않는» 장식만 쓴다. 캡과 테두리의 위·아래 변은
// 높이만 먹으므로 폭과 무관하게 안전하고, **세로 변은 색이 4px 이상 남는 폭(w>=6)에서만**
// 그린다 — 그래서 좁을 때 라우팅은 가로 두 줄로, 넓어지면 닫힌 타일로 보인다.
// 행 안의 좌표는 전부 행 높이 비례(k)로 잡는다 — 고정 픽셀을 쓰면 절반 높이 행에서 바가 행
// 밖으로 삐져나간다. 라벨은 비례로 안 되는 축이라(글자는 반으로 줄면 못 읽는다) compact 행에서만
// 세 줄 중 이름만 남긴다. 부제·현재 글자를 잃어도 «어느 레인이 여기 있다»는 읽혀야 하기 때문.
function render(){if(!DATA)return;const L=layout(),rows=L.rows,width=Math.max(1,wrap.clientWidth),height=Math.max(1,wrap.clientHeight),dpr=devicePixelRatio||1;if(cv.width!==Math.ceil(width*dpr)||cv.height!==Math.ceil(height*dpr)){cv.width=Math.ceil(width*dpr);cv.height=Math.ceil(height*dpr);cv.style.width=width+'px';cv.style.height=height+'px'}const contentWidth=Math.max(width,LABEL_W+Math.ceil(dur*pps)),contentHeight=Math.max(height,L.total);timeline.style.width=contentWidth+'px';timeline.style.height=contentHeight+'px';ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,width,height);ctx.fillStyle='#14161a';ctx.fillRect(0,0,width,height);const followMode=$('#followmode').value,span=Math.max(1,width-LABEL_W);let timeScroll;if(!au.paused&&followMode==='pin'){timeScroll=Math.max(0,au.currentTime*pps-span*.38)}else if(!au.paused&&followMode==='page'){const headX=au.currentTime*pps;if(headX<PAGE_TS||headX>PAGE_TS+span*.85)PAGE_TS=Math.max(0,headX-span*.15);timeScroll=PAGE_TS}else{timeScroll=Math.max(0,wrap.scrollLeft-LABEL_W)}RENDER_TS=timeScroll;const viewStart=timeScroll/pps,viewEnd=(timeScroll+Math.max(0,width-LABEL_W))/pps,yScroll=wrap.scrollTop;ctx.fillStyle='#1d2026';ctx.fillRect(0,0,LABEL_W,height);ctx.strokeStyle='#303640';ctx.beginPath();ctx.moveTo(LABEL_W-.5,0);ctx.lineTo(LABEL_W-.5,height);ctx.stroke();ctx.font='10px system-ui';ctx.fillStyle='#788392';ctx.strokeStyle='#292e36';for(let sec=Math.floor(viewStart/10)*10;sec<=viewEnd+10;sec+=10){const x=LABEL_W+sec*pps-timeScroll;ctx.beginPath();ctx.moveTo(x+.5,0);ctx.lineTo(x+.5,height);ctx.stroke();ctx.fillText(stamp(sec),x+3,11)}drawSections(0,viewStart,viewEnd,timeScroll,width,height);rows.forEach((trackIndex,row)=>{const track=DATA.tracks[trackIndex],rh=L.hs[row],top=L.tops[row]-yScroll,bottom=top+rh,k=rh/ROW_H;if(bottom<0||top>height)return;ctx.save();ctx.beginPath();ctx.rect(0,top,LABEL_W-4,rh);ctx.clip();if(track.band){ctx.fillStyle=track.band;ctx.fillRect(0,top+1,4,rh-2)}ctx.fillStyle=track.color;const nowT=au.currentTime-(track.ust?OFFS[trackIndex]||0:0);let liveTxt='';if(track.segs.length){const s=track.segs[lowerBound(track.segs,nowT)-1];if(s)liveTxt=s.t}else if(track.lines.length){const l=track.lines[lowerBound(track.lines,nowT)-1];if(l&&nowT<=l.end+1)liveTxt=l.label}if(track.compact){ctx.font='9px system-ui';ctx.fillText(track.name,8,top+10);if(track._score){const nameW=ctx.measureText(track.name).width;ctx.fillStyle='#788392';ctx.fillText(track._score,11+nameW,top+10)}if(liveTxt){const conv=toReadable(liveTxt);ctx.fillStyle='#f2f5f8';ctx.font='bold 11px system-ui';ctx.fillText(conv,8,top+22)}}else{ctx.font='12px system-ui';ctx.fillText(track.name,8,top+14);ctx.fillStyle='#788392';ctx.font='9px system-ui';ctx.fillText(track.line_reference?'caption mapping':track.no_segs?'no measured syllables'+(track._score||''):track.ust?`Shift=레인 Ctrl=범위선택 Alt=이동 · δadj ${((OFFS[trackIndex]||0)>=0?'+':'')+(OFFS[trackIndex]||0).toFixed(2)}s${Object.keys(LOFFS[trackIndex]||{}).length?` (+소절 ${Object.keys(LOFFS[trackIndex]).length})`:''} · dblclick reset · E=복사`:(track._score?`${track._score.slice(3)} · ${track.segs.length}조각`:`${track.segs.length} syllable spans`),8,top+26);if(liveTxt){const conv=toReadable(liveTxt),subst=conv!==liveTxt;ctx.font='bold 15px system-ui';const convW=ctx.measureText(conv).width;ctx.fillStyle=subst?'#a9b6c9':'#f2f5f8';ctx.fillText(conv,8,top+45);if(subst){ctx.font='10px system-ui';ctx.fillStyle='#5a6472';ctx.fillText(liveTxt,14+convW,top+45)}}}ctx.restore();ctx.strokeStyle='#303640';ctx.beginPath();ctx.moveTo(0,bottom-.5);ctx.lineTo(width,bottom-.5);ctx.stroke();const toff=track.ust?OFFS[trackIndex]||0:0;const noSegTrack=track.line_reference||track.no_segs,drawLines=MODE!=='segs'||noSegTrack,drawSegs=MODE!=='lines'&&!noSegTrack;if(drawLines){const big=!drawSegs;const lineStart=track.ust?0:Math.max(0,lowerBound(track.lines,viewStart-toff)-1);for(let i=lineStart;i<track.lines.length&&(track.ust||track.lines[i].start+toff<=viewEnd);i++){const line=track.lines[i],leff=track.ust?toff+lineOff(trackIndex,i):toff;if(track.ust&&(line.end+leff<viewStart||line.start+leff>viewEnd))continue;const x=LABEL_W+(line.start+leff)*pps-timeScroll,w=Math.max(1,(line.end-line.start)*pps),ly=top+(big?10:5)*k,lh=(big?30:9)*k;ctx.fillStyle=track.color+(big?'55':'22');ctx.fillRect(x,ly,w,lh);ctx.strokeStyle=track.color+'88';ctx.strokeRect(x+.5,ly+.5,Math.max(1,w-1),lh-1);if(w>=24){ctx.save();ctx.beginPath();ctx.rect(x+1,ly,w-2,lh);ctx.clip();ctx.fillStyle=big?'#eef2f7':'#c7cede';ctx.font=`${Math.max(8,Math.round((big?11:9)*k))}px system-ui`;ctx.fillText(line.label,x+3,ly+(big?19:8)*k);ctx.restore()}}}if(drawSegs){const st=top+(MODE==='both'?17:13)*k,sh=(MODE==='both'?23:25)*k;const segStart=track.ust?0:Math.max(0,lowerBound(track.segs,viewStart-toff)-1),shape=track.shape||'';ctx.fillStyle=track.color;for(let i=segStart;i<track.segs.length&&(track.ust||track.segs[i].start+toff<=viewEnd);i++){const seg=track.segs[i],seff=track.ust?segEff(track,trackIndex,i):toff;if(track.ust&&(seg.end+seff<viewStart||seg.start+seff>viewEnd))continue;const x=LABEL_W+(seg.start+seff)*pps-timeScroll,w=Math.max(2,(seg.end-seg.start)*pps);ctx.fillRect(x,st,w,sh);if(shape==='twopass'){ctx.fillStyle='#ffffff99';ctx.fillRect(x,st,w,2);ctx.fillStyle=track.color}else if(shape==='routed'){ctx.fillStyle='#0b0e12';ctx.fillRect(x,st,w,1);ctx.fillRect(x,st+sh-1,w,1);if(w>=6){ctx.fillRect(x,st,1,sh);ctx.fillRect(x+w-1,st,1,sh)}ctx.fillStyle=track.color}if(track.ust&&SEL&&SEL.ti===trackIndex&&track._segLine&&SEL.set.has(track._segLine[i])){ctx.strokeStyle='#ffffff';ctx.strokeRect(x+.5,st+.5,Math.max(1,w-1),sh-1)}if(w>=11){ctx.fillStyle='#101318';ctx.font=`${Math.max(8,Math.round(11*k))}px system-ui`;ctx.fillText(seg.t,x+2,st+sh-7*k);ctx.fillStyle=track.color}}}});drawSections(1,viewStart,viewEnd,timeScroll,width,height);if(MARQ){const rowIdx=rows.indexOf(MARQ.ti);if(rowIdx>=0){const lo=Math.min(MARQ.t0,MARQ.t1),hi=Math.max(MARQ.t0,MARQ.t1),mx=LABEL_W+lo*pps-timeScroll,mw=Math.max(1,(hi-lo)*pps),my=L.tops[rowIdx]-yScroll,mh=L.hs[rowIdx];ctx.fillStyle='#8ab4ff22';ctx.fillRect(mx,my,mw,mh);ctx.strokeStyle='#8ab4ff';ctx.strokeRect(mx+.5,my+.5,mw-1,mh-1)}}const playX=LABEL_W+au.currentTime*pps-timeScroll;if(playX>=LABEL_W&&playX<=width){ctx.strokeStyle='#ff5c67';ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(playX+.5,0);ctx.lineTo(playX+.5,height);ctx.stroke();ctx.lineWidth=1}}
function ustScoreVal(track){if(!DATA||!DATA.ust_truth||track.ust||track.line_reference||!track.lines)return null;const anch=DATA.ust_truth.anchors;let adj=0;for(let i=0;i<DATA.tracks.length;i++){const u=DATA.tracks[i];if(u.ust&&!/harm/i.test(u.name)){adj=OFFS[i]||0;break}}let n=0,hit=0;for(const k in anch){const line=track.lines[+k];if(!line)continue;n++;if(Math.abs(line.start-(anch[k]+adj))<=0.15)hit++}return n?hit/n:null}
function ustScore(track){const v=ustScoreVal(track);return v==null?'':` · 라인 ${Math.round(100*v)}%`+syllScore(track)}
// 음절 축 — 세그에 박아둔 대응 UST 노트 시각과 비교. 라인 채점이 못 보는 BPE 보간 손실이 여기 보인다.
// 음절% = 세그 시작이 정답 노트 시작과 0.10s 안. 분절% = **라인의 칸 수**가 정답
// 노트 수와 같은 라인의 비율. 「몇 개로 나눴나」는 시작 시각으로는 약하게만 잡힌다 —
// numb을 n|u|m|b 넷으로 쪼개도 넷 중 하나는 시작이 맞는다(2026-08-02).
function syllScore(track){if(!DATA||!DATA.ust_truth||!track.segs||!track.segs.length)return'';let adj=0;for(let i=0;i<DATA.tracks.length;i++){const u=DATA.tracks[i];if(u.ust&&!/harm/i.test(u.name)){adj=OFFS[i]||0;break}}let n=0,hit=0;for(const s of track.segs){if(s.truth==null)continue;n++;if(Math.abs(s.start-(s.truth+adj))<=0.10)hit++}if(!n)return'';let out=` · 음절 ${Math.round(100*hit/n)}%`;const counts=DATA.ust_truth.note_counts;if(counts&&track.lines){let lines=0,same=0,k=0;for(let i=0;i<track.lines.length;i++){const ln=track.lines[i];let c=0;while(k<track.segs.length&&track.segs[k].start<ln.end+1e-6){if(track.segs[k].start>=ln.start-0.5)c++;k++}const want=counts[String(i)];if(want==null||!c)continue;lines++;if(c===want)same++}if(lines)out+=` · 분절 ${Math.round(100*same/lines)}%`}return out}
// 채점은 세그 전량을 훑으므로 매 프레임 계산하면 렌더가 죽는다. 곡 로드·드래그 종료 때만 갱신한다.
function refreshScores(){if(DATA)DATA.tracks.forEach(t=>{t._score=ustScore(t)})}
// UST 준정답이 있는 곡은 레인을 채점 순으로 — 정답(UST)·참조 레인을 위에, 후보는 적중률 내림차순
function sortTracksByUst(song){if(!song.ust_truth)return;const grp=t=>t.ust?2:t.line_reference?1:0;song.tracks.sort((a,b)=>{const ga=grp(a),gb=grp(b);if(ga!==gb)return gb-ga;const va=ustScoreVal(a),vb=ustScoreVal(b);return (vb==null?-1:vb)-(va==null?-1:va)})}
function renderTrackToggles(){toggles.replaceChildren();DATA.tracks.forEach((track,i)=>{const label=document.createElement('label'),box=document.createElement('input');box.type='checkbox';box.checked=visible[i];box.addEventListener('change',()=>{visible[i]=box.checked;invalidateLayout();queueRender()});label.append(box);if(track.band){const bar=document.createElement('span');bar.className='swatch';bar.style.background=track.band;bar.style.width='4px';label.append(bar)}const swatch=document.createElement('span');swatch.className='swatch';swatch.style.background=track.color;label.append(swatch,document.createTextNode(track.name+(track.no_segs?' (no segs)':track.line_reference?' (caption)':'')));toggles.append(label)})}
function pointAt(event){const rect=cv.getBoundingClientRect();return{x:event.clientX-rect.left,y:event.clientY-rect.top,time:(RENDER_TS+event.clientX-rect.left-LABEL_W)/pps}}
function findSpan(track,time){if(!track?.segs.length)return null;const index=lowerBound(track.segs,time);for(const candidate of [index,index-1]){const seg=track.segs[candidate];if(seg&&time>=seg.start-4/pps&&time<=seg.end+4/pps)return seg}return null}
function findLine(track,time){if(!track?.lines?.length)return null;const index=lowerBound(track.lines,time);for(const candidate of [index,index-1]){const line=track.lines[candidate];if(line&&time>=line.start-2/pps&&time<=line.end+2/pps)return line}return null}
function showTip(event,html){tip.innerHTML=html;tip.style.display='block';tip.style.left=Math.min(event.clientX+14,innerWidth-325)+'px';tip.style.top=Math.min(event.clientY+14,innerHeight-90)+'px'}
cv.addEventListener('mousemove',event=>{if(MARQ){MARQ.t1=pointAt(event).time;queueRender();return}if(USTDRAG){const d=(event.clientX-USTDRAG.x)/pps;if(USTDRAG.lis){const m=LOFFS[USTDRAG.ti]=LOFFS[USTDRAG.ti]||{};USTDRAG.lis.forEach(k=>m[k]=USTDRAG.bases[k]+d)}else if(USTDRAG.li!=null){(LOFFS[USTDRAG.ti]=LOFFS[USTDRAG.ti]||{})[USTDRAG.li]=USTDRAG.base+d}else{OFFS[USTDRAG.ti]=USTDRAG.base+d}queueRender();return}const point=pointAt(event),ti=rowAt(wrap.scrollTop+point.y),track=DATA?.tracks[ti];if(!track){tip.style.display='none';return}const toff=track.ust?OFFS[ti]||0:0;let seg=null,segE=toff;if(MODE!=='lines'){if(track.ust){for(let k=0;k<track.segs.length;k++){const e=segEff(track,ti,k);if(point.time>=track.segs[k].start+e-4/pps&&point.time<=track.segs[k].end+e+4/pps){seg=track.segs[k];segE=e;break}}}else{seg=findSpan(track,point.time-toff)}}if(seg){const prod=seg.prod,delta=prod?`PROD: ${stamp(prod[0])}–${stamp(prod[1])}\nΔ start ${signed(seg.start+segE-prod[0])} · Δ end ${signed(seg.end+segE-prod[1])}`:'PROD: no same-character span';showTip(event,`<b>${esc(track.name)} · ${esc(seg.t||'(empty)')}</b>\n${stamp(seg.start+segE)}–${stamp(seg.end+segE)}\n${delta}`);return}let line=null,lineE=toff;if(track.ust){for(let k=0;k<track.lines.length;k++){const e=toff+lineOff(ti,k);if(point.time>=track.lines[k].start+e-2/pps&&point.time<=track.lines[k].end+e+2/pps){line=track.lines[k];lineE=e;break}}}else{line=findLine(track,point.time-toff)}if(!line){tip.style.display='none';return}const text=line.text.length>140?line.text.slice(0,139)+'…':line.text;showTip(event,`<b>${esc(track.name)} · line</b>\n${esc(text)}\n${stamp(line.start+lineE)}–${stamp(line.end+lineE)} · ${(line.end-line.start).toFixed(2)}s`)});cv.addEventListener('mouseleave',()=>tip.style.display='none');function saveOff(ti){const t=DATA&&DATA.tracks[ti];if(!t)return;try{localStorage.setItem('ustoff:'+activeId+':'+t.name,String(OFFS[ti]||0));localStorage.setItem('ustloff:'+activeId+':'+t.name,JSON.stringify(LOFFS[ti]||{}))}catch(e){}}
cv.addEventListener('mousedown',event=>{if(!(event.shiftKey||event.altKey||event.ctrlKey)||!DATA)return;const point=pointAt(event),ti=rowAt(wrap.scrollTop+point.y),track=DATA.tracks[ti];if(!track||!track.ust)return;event.preventDefault();
if(event.ctrlKey){MARQ={ti,t0:point.time,t1:point.time};return}
if(event.altKey){const toff=OFFS[ti]||0;let li=-1;for(let k=0;k<track.lines.length;k++){const e=toff+lineOff(ti,k);if(point.time>=track.lines[k].start+e-0.2&&point.time<=track.lines[k].end+e+0.2){li=k;break}}
if(SEL&&SEL.ti===ti&&(li<0||SEL.set.has(li))){const bases={};SEL.set.forEach(k=>bases[k]=lineOff(ti,k));USTDRAG={ti,lis:[...SEL.set],bases,x:event.clientX};return}
if(li<0)return;USTDRAG={ti,li,x:event.clientX,base:lineOff(ti,li)}}
else{USTDRAG={ti,x:event.clientX,base:OFFS[ti]||0}}});
window.addEventListener('mouseup',()=>{if(MARQ){const track=DATA&&DATA.tracks[MARQ.ti];if(track){const lo=Math.min(MARQ.t0,MARQ.t1),hi=Math.max(MARQ.t0,MARQ.t1),toff=OFFS[MARQ.ti]||0,set=new Set();track.lines.forEach((l,k)=>{const e=toff+lineOff(MARQ.ti,k);if(l.start+e<=hi&&l.end+e>=lo)set.add(k)});SEL=set.size?{ti:MARQ.ti,set}:null}MARQ=null;queueRender();return}if(!USTDRAG)return;saveOff(USTDRAG.ti);USTDRAG=null;refreshScores();queueRender()});
cv.addEventListener('dblclick',event=>{const point=pointAt(event),ti=rowAt(wrap.scrollTop+point.y),track=DATA?.tracks[ti];if(!track||!track.ust)return;OFFS[ti]=0;delete LOFFS[ti];saveOff(ti);refreshScores();queueRender()});
cv.addEventListener('click',event=>{if(event.shiftKey||event.ctrlKey||event.altKey)return;const point=pointAt(event);if(point.x>=LABEL_W){au.currentTime=Math.max(0,Math.min(dur,point.time));queueRender()}});wrap.addEventListener('scroll',queueRender,{passive:true});wrap.addEventListener('wheel',event=>{if(!DATA)return;event.preventDefault();const rect=cv.getBoundingClientRect(),cursorX=event.clientX-rect.left,anchorTime=(RENDER_TS+cursorX-LABEL_W)/pps,old=pps;pps=Math.min(500,Math.max(5,pps*Math.pow(1.0015,-event.deltaY)));const ratio=pps/old;PAGE_TS=Math.max(0,PAGE_TS*ratio);if(au.paused||$('#followmode').value==='off'){const ts=anchorTime*pps-(cursorX-LABEL_W);wrap.scrollLeft=ts>0?ts+LABEL_W:0}queueRender()},{passive:false});new ResizeObserver(queueRender).observe(wrap);
$('#zi').onclick=()=>{pps=Math.min(pps*1.5,500);queueRender()};$('#zo').onclick=()=>{pps=Math.max(pps/1.5,5);queueRender()};$('#viewmode').onchange=event=>{MODE=event.target.value;queueRender()};$('#followmode').onchange=()=>queueRender();$('#audiosrc').onchange=function(){const time=au.currentTime,playing=!au.paused,source=SOURCES[+this.value];if(!source)return;au.setAttribute('src',dataUrl(source.path));au.load();au.addEventListener('loadedmetadata',()=>{au.currentTime=time;if(playing)au.play()},{once:true})};document.addEventListener('keydown',event=>{if(['INPUT','TEXTAREA','SELECT'].includes(event.target.tagName))return;if(event.code==='Space'){event.preventDefault();au.paused?au.play():au.pause()}else if(event.code==='ArrowLeft')au.currentTime=Math.max(0,au.currentTime-2);else if(event.code==='ArrowRight')au.currentTime=Math.min(dur,au.currentTime+2);else if(event.code==='Escape'){SEL=null;queueRender()}else if(event.code==='KeyE'&&DATA){const parts=[];DATA.tracks.forEach((t,i)=>{if(t.ust)parts.push({track:t.name,lane_offset:+(OFFS[i]||0).toFixed(3),phrase_offsets:Object.fromEntries(Object.entries(LOFFS[i]||{}).filter(([,v])=>Math.abs(v)>1e-3).map(([k,v])=>[k,+v.toFixed(3)]))})});if(parts.length){const txt=JSON.stringify({song:activeId,ust:parts});if(navigator.clipboard&&navigator.clipboard.writeText){navigator.clipboard.writeText(txt).then(()=>{status.textContent='UST 오프셋 JSON 복사됨'})}else{console.log(txt);status.textContent='UST 오프셋 콘솔 출력'}}}});
function dataUrl(path){return new URL(`data/${path}`,location.href).href}
function loadSong(videoId){if(!videoId||videoId===activeId)return;const apply=()=>{const song=window.__SONG_DATA__[videoId];if(!song){status.textContent=`Could not load ${videoId}`;status.className='error';return}activeId=videoId;DATA=song;dur=Math.max(Number(song.duration)||0,...song.tracks.flatMap(track=>[...track.lines,...track.segs].map(item=>item.end)),1);SOURCES=(song.audio&&song.audio.sources)||[];const srcSel=$('#audiosrc');srcSel.replaceChildren();SOURCES.forEach((source,i)=>{const option=document.createElement('option');option.value=i;option.textContent=source.name;srcSel.append(option)});srcSel.value='0';au.pause();if(SOURCES.length)au.setAttribute('src',dataUrl(SOURCES[0].path));au.load();$('#page-title').textContent=`[${song.stratum}] ${song.title} · Karaoke alignment review`;OFFS={};LOFFS={};sortTracksByUst(song);visible=song.tracks.map(()=>true);invalidateLayout();song.tracks.forEach((t,i)=>{if(t.ust){const v=parseFloat(localStorage.getItem('ustoff:'+videoId+':'+t.name));if(Number.isFinite(v)&&v)OFFS[i]=v;try{const m=JSON.parse(localStorage.getItem('ustloff:'+videoId+':'+t.name)||'null');if(m&&typeof m==='object'&&Object.keys(m).length)LOFFS[i]=m}catch(e){}t._segLine=t.segs.map(s=>{let j=lowerBound(t.lines,s.start+1e-4)-1;return j<0?0:j})}});refreshScores();renderTrackToggles();status.textContent=song.no_segs_tracks?`${song.no_segs_tracks} track(s) without syllables`:'Syllable spans available';status.className=song.no_segs_tracks?'warn':'tag';wrap.scrollLeft=0;wrap.scrollTop=0;PAGE_TS=0;queueRender()};if(window.__SONG_DATA__[videoId]){apply();return}const script=document.createElement('script');script.src=`data/${encodeURIComponent(videoId)}.js`;script.onload=apply;script.onerror=()=>{status.textContent=`Failed to load data/${videoId}.js`;status.className='error'};document.head.append(script)}
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
    sections = allin1_sections(song)
    latest = max((item["end"] for track in tracks for item in track["lines"] + track["segs"]), default=0.0)
    # 구간은 레인을 안 쓰지만 곡 끝까지 덮으므로 길이 계산에는 여전히 들어가야 한다 —
    # 레인이던 시절엔 위 max가 알아서 봤다.
    if sections:
        latest = max(latest, max(span["e"] for span in sections["spans"]))
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
    if sections:
        payload["sections"] = sections
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
