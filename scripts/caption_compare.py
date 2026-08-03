"""유튜브 수동 자막 타이밍 vs 벤치 정렬 결과 대조 — 독립(제3) 기준 축.

벤치의 기본 기준(프로드 싱크)은 MMS 출력이라 «MMS와 얼마나 같나»를 재는 순환성이 있다.
업로더 수동 자막의 라인 타이밍은 사람이 만든 독립 기준이므로, 이 스크립트는 한 곡의
모든 벤치 런(+프로드 기준선 자체)을 자막 타이밍에 대해 채점해 나란히 보여준다.

두 매칭 경로:
- direct: 자막 텍스트가 원문 가사(ja) — 자막 줄 ↔ 정렬 줄을 텍스트 정규화 키의 LCS로 짝짓기
  (반복 구절을 자막 제작자가 합쳐버린 경우는 LCS가 소화 — 사용자 실측 주의사항)
- via-wiki: 자막이 번역(ko 등) — 자막 줄 ↔ 위키 라인별 번역 ↔ 위키 원문 줄 ↔ 정렬 줄
  (보카로 위키는 원문·번역이 라인 병렬이라 다리가 된다)

주의: 다른 업로드(커버 등)의 자막을 쓸 때는 두 영상의 길이가 같아야 타임라인 비교가
성립한다(熱異常 실측: 원본·커버 둘 다 241초). signed median이 크게 뜨면 상수 오프셋 의심.

    ./.venv/Scripts/python.exe scripts/caption_compare.py --song b2NTglk9tvI \
        --srt benchmark/captions/hDhjRh-Gt4g.ja.srt
    ./.venv/Scripts/python.exe scripts/caption_compare.py --song icBDYkfxpMs \
        --srt benchmark/captions/icBDYkfxpMs.ko.srt --wiki benchmark/captions/icbd_wiki.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "benchmark"

# benchmark_alignment의 텍스트 짝짓기(_pair_lines·_pair_key)를 그대로 재사용한다 —
# 자막 대조가 벤치 지표와 다른 규칙으로 짝지으면 두 축의 차이가 «매칭 규칙 차이»에 오염된다.
_spec = importlib.util.spec_from_file_location("bench", REPO / "scripts" / "benchmark_alignment.py")
_bench = importlib.util.module_from_spec(_spec)
sys.modules["bench"] = _bench
_spec.loader.exec_module(_bench)

_TS = re.compile(r"(\d+):(\d\d):(\d\d)[,.](\d{1,3})")


def parse_srt(path: Path) -> list[dict]:
    """SRT → [{start, end, text}] (멀티라인 블록은 공백으로 합침)."""
    blocks = re.split(r"\r?\n\r?\n", path.read_text(encoding="utf-8-sig"))
    out = []
    for b in blocks:
        lines = [ln.strip() for ln in b.strip().splitlines() if ln.strip()]
        if len(lines) < 2 or "-->" not in "".join(lines[:2]):
            continue
        ts_line = lines[1] if "-->" in lines[1] else lines[0]
        m = _TS.findall(ts_line)
        if len(m) < 2:
            continue
        def sec(t):
            h, mi, s, ms = (int(x) for x in t)
            return h * 3600 + mi * 60 + s + ms / 1000.0
        text_lines = lines[2:] if "-->" in lines[1] else lines[1:]
        text = " ".join(text_lines).strip()
        if text:
            out.append({"start": sec(m[0]), "end": sec(m[1]), "text": text})
    return out


def match_starts(ref_texts: list[str], ref_starts: list[float], est_lines: list[dict]) -> list[tuple[float, float, int]]:
    """(자막 시각, 후보 시각, 후보 줄 idx) 짝 목록 — bench의 LCS 짝짓기 재사용."""
    ref = [{"text": t, "start": s} for t, s in zip(ref_texts, ref_starts)]
    pairs = _bench._pair_lines(ref, est_lines)
    out = []
    for ri, ei in pairs:
        rs, es = ref[ri]["start"], est_lines[ei].get("start")
        if rs is None or es is None:
            continue
        out.append((float(rs), float(es), ei))
    return out


def metrics(pairs: list[tuple[float, float, int]]) -> dict:
    if not pairs:
        return {"matched": 0}
    deltas = [e - r for r, e, _ in pairs]
    ab = sorted(abs(d) for d in deltas)
    return {
        "matched": len(pairs),
        "mae": round(statistics.median(ab), 3),
        "mean": round(sum(ab) / len(ab), 3),
        "p95": round(_bench._percentile(ab, 95), 3),
        "pco03": round(sum(1 for a in ab if a <= 0.3) / len(ab), 4),
        "signed_med": round(statistics.median(deltas), 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--song", required=True, help="벤치 평가 세트의 video_id")
    ap.add_argument("--srt", required=True, help="수동 자막 SRT 경로")
    ap.add_argument("--wiki", default=None, help="번역 자막일 때: 위키 페이지 JSON(라인 병렬 번역)")
    ap.add_argument("--eval-set", default=str(BENCH / "eval_set.json"))
    args = ap.parse_args()

    eval_set = json.loads(Path(args.eval_set).read_text(encoding="utf-8"))
    song = next((s for s in eval_set["songs"] if s["video_id"] == args.song), None)
    if song is None:
        raise SystemExit(f"eval_set에 없음: {args.song}")

    caps = parse_srt(Path(args.srt))
    if not caps:
        raise SystemExit("자막 파싱 결과 0줄")

    # 기준 라인(원문 텍스트, 자막 시각) 구성
    if args.wiki:
        wiki = json.loads(Path(args.wiki).read_text(encoding="utf-8"))
        wl = [ln for ln in wiki["lines"] if (ln.get("text") or "").strip()]
        # 자막(번역) ↔ 위키 번역 짝짓기 → 위키 원문 줄에 자막 시각 부여
        tr_lines = [{"text": ln.get("translation") or "", "start": None} for ln in wl]
        cap_est = [{"text": c["text"], "start": c["start"]} for c in caps]
        bridge = _bench._pair_lines(tr_lines, cap_est)
        ref_texts, ref_starts = [], []
        for wi, ci in bridge:
            ref_texts.append(wl[wi]["text"])
            ref_starts.append(caps[ci]["start"])
        note = f"via-wiki: 자막 {len(caps)}줄 중 {len(bridge)}줄이 위키 번역과 짝지어짐"
    else:
        ref_texts = [c["text"] for c in caps]
        ref_starts = [c["start"] for c in caps]
        note = f"direct: 자막 {len(caps)}줄"

    rows = []
    # 프로드 기준선 자체도 자막 축에서 채점 — 순환성 밖에서 현행이 어디쯤인지
    baseline = _bench.baseline_segments(song)
    rows.append(("PROD-baseline", metrics(match_starts(ref_texts, ref_starts, baseline))))

    for run_path in sorted((BENCH / "runs").glob(f"*/{args.song}__r1.json")):
        r = json.loads(run_path.read_text(encoding="utf-8"))
        if r.get("error") or not r.get("lines"):
            continue
        label = f"{r['aligner']} ({r['separator']})"
        rows.append((label, metrics(match_starts(ref_texts, ref_starts, r["lines"]))))

    rows.sort(key=lambda x: (x[1].get("mae") is None, x[1].get("mae", 9e9)))
    title = (song.get("title") or args.song)[:60]
    out = [f"# 자막 타이밍 대조 — {title}", "",
           f"- 자막: `{Path(args.srt).name}` / {note}",
           "- MAE·P95·PCO는 **자막 시각 기준** (프로드 기준선과 독립). signed med가 크면 상수 오프셋 의심.", "",
           "| 조합 | 짝 | MAE | mean | P95 | PCO@0.3 | signed med |",
           "|---|---|---|---|---|---|---|"]
    for label, m in rows:
        if m.get("matched", 0) == 0:
            out.append(f"| {label} | 0 | - | - | - | - | - |")
            continue
        out.append(f"| {label} | {m['matched']} | {m['mae']} | {m['mean']} | {m['p95']} | {m['pco03']} | {m['signed_med']} |")
    report = "\n".join(out)
    dest = BENCH / f"caption_compare_{args.song}.md"
    dest.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\n저장: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
