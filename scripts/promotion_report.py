"""후보 정렬기 승격(promotion) 판정 리포트 — collapse_annotations.json 조인 (P2 B 트랙, task #4).

``benchmark/REPORT.md``(``benchmark_alignment.py``의 산출물)와 **별도 파일**이다. baseline
스윕(``benchmark_alignment.py``)이 실행 중인 동안에도 그 스크립트를 고치지 않고 안전하게
얹기 위해서다 — 이 스크립트는 ``benchmark/runs/**/*.json``·``eval_set.json``·
``collapse_annotations.json``을 **읽기만** 하고 ``benchmark/PROMOTION.md``만 쓴다
(``REPORT.md``는 하네스 소유라 건드리지 않는다).

승격 조건(task #4 원문):
- MAE·P95 비악화 — 후보 median <= 기준 median * (1 + REL_TOL). 판정은 전체(ALL) 집계
  기준이다(층별 표는 참고용 — 곡 수가 적은 층은 표본 잡음이 커서 승격 판정에 직접 쓰지
  않는다). REL_TOL은 부동소수·측정 잡음을 흡수하기 위한 값이고 task가 명시한 수치는 아니다.
- hard_audio 붕괴 곡(``HARD_AUDIO_SONGS``) 비악화 — 기준이 이미 붕괴한 곡은 후보가 붕괴해도
  악화가 아니다. 기준이 안 붕괴했는데 후보가 붕괴하면 악화.
- VRAM(정렬 스테이지, torch allocator process peak) 9GB 이내.

``collapse_annotations.json``의 cause가 ``input_lyrics``/``input_caption``인 곡은 모델이 못
고치는 입력 오염이므로 붕괴 게이트 채점에서 빼고 별도 열로만 보고한다(``cause`` 필드는
사용자 수동 청취 실측 — ``benchmark/collapse_annotations.json`` 헤더 주석 참고).

층별 표는 ``mixed``(코드스위칭)를 포함해 ``benchmark_alignment.py``의 stratum 분류를 그대로
따른다 — 단일 언어 후보가 코드스위칭 곡에서 어떻게 무너지는지 한눈에 보려는 목적.

    ./.venv/Scripts/python.exe scripts/promotion_report.py
    ./.venv/Scripts/python.exe scripts/promotion_report.py --baseline htdemucs__mms-baseline
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# 붕괴 판정·stratum 정규화 로직은 benchmark_alignment.py의 것을 그대로 재사용한다(read-only
# import — 그 스크립트를 고치지 않으므로 실행 중인 baseline 스윕과 충돌하지 않는다). 규칙이
# 바뀌면 여기도 따라 바뀐다.
from benchmark_alignment import base_language, collapse_flags  # noqa: E402

DEFAULT_BENCH_DIR = REPO_ROOT / "benchmark"

# MAE/P95 비악화 판정 상대 허용오차 — task가 명시한 값이 아니라 잡음 흡수용 기본값.
REL_TOL = 0.02
VRAM_LIMIT_MB = 9 * 1024
HARD_AUDIO_SONGS = {"s4kAOHUSvT8", "vjBFftpQxxM"}
INPUT_CONTAMINATION_CAUSES = {"input_lyrics", "input_caption"}


def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def load_annotations(bench_dir: Path) -> dict:
    raw = load_json(bench_dir / "collapse_annotations.json", {})
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def load_runs(bench_dir: Path) -> dict[tuple[str, str], list[dict]]:
    combos: dict[tuple[str, str], list[dict]] = {}
    runs_root = bench_dir / "runs"
    for combo_dir in sorted(runs_root.glob("*__*")):
        sep, _, aln = combo_dir.name.partition("__")
        for run_path in sorted(combo_dir.glob("*.json")):
            try:
                run = json.loads(run_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            combos.setdefault((sep, aln), []).append(run)
    return combos


def _median(values: list[float | None]) -> float | None:
    xs = [v for v in values if v is not None]
    return round(statistics.median(xs), 3) if xs else None


def is_input_contaminated(video_id: str, annotations: dict) -> bool:
    return (annotations.get(video_id) or {}).get("cause") in INPUT_CONTAMINATION_CAUSES


def run_stratum(run: dict) -> str:
    return base_language(run.get("stratum") or run.get("language") or "?")


def _aggregate(runs: list[dict]) -> dict:
    ok = [r for r in runs if not r.get("error") and (r.get("metrics") or {}).get("mae") is not None]
    vram = [r.get("align_vram_peak_mb") for r in runs if r.get("align_vram_peak_mb") is not None]
    return {
        "n_songs": len({r["video_id"] for r in runs}),
        "mae": _median([(r.get("metrics") or {}).get("mae") for r in ok]),
        "p95": _median([(r.get("metrics") or {}).get("p95_abs") for r in ok]),
        "vram_peak_max": round(max(vram), 1) if vram else None,
        "vram_peak_median": _median(vram),
        "collapsed_songs": sorted({r["video_id"] for r in runs if collapse_flags(r)}),
    }


def combo_stats(runs: list[dict], annotations: dict) -> dict:
    """게이트 대상(clean) 런과 입력오염(excluded) 런을 나눠 집계한다."""
    clean = [r for r in runs if not is_input_contaminated(r["video_id"], annotations)]
    excluded = [r for r in runs if is_input_contaminated(r["video_id"], annotations)]
    return {
        "clean": _aggregate(clean),
        "excluded": _aggregate(excluded),
        "excluded_video_ids": sorted({r["video_id"] for r in excluded}),
    }


def stratum_breakdown(runs: list[dict], annotations: dict) -> dict[str, dict]:
    """clean(입력오염 제외) 런만 stratum별로 나눠 집계 — mixed 포함. 표시 전용."""
    clean = [r for r in runs if not is_input_contaminated(r["video_id"], annotations)]
    by_stratum: dict[str, list[dict]] = {}
    for r in clean:
        by_stratum.setdefault(run_stratum(r), []).append(r)
    return {stratum: _aggregate(group) for stratum, group in sorted(by_stratum.items())}


def hard_audio_check(runs: list[dict], baseline_runs: list[dict]) -> dict:
    by_song = {r["video_id"]: r for r in runs}
    base_by_song = {r["video_id"]: r for r in baseline_runs}
    out: dict[str, dict] = {}
    for vid in sorted(HARD_AUDIO_SONGS):
        cand = by_song.get(vid)
        base = base_by_song.get(vid)
        cand_flags = collapse_flags(cand) if cand else ["missing"]
        base_flags = collapse_flags(base) if base else ["missing"]
        degraded = bool(cand_flags) and not bool(base_flags)
        out[vid] = {
            "baseline_flags": base_flags,
            "candidate_flags": cand_flags,
            "degraded": degraded,
            "baseline_mae": (base.get("metrics") or {}).get("mae") if base else None,
            "candidate_mae": (cand.get("metrics") or {}).get("mae") if cand else None,
        }
    return out


def promotion_verdict(candidate_clean: dict, baseline_clean: dict, hard_audio: dict) -> dict:
    reasons: list[str] = []

    mae_ok = (
        candidate_clean["mae"] is not None
        and baseline_clean["mae"] is not None
        and candidate_clean["mae"] <= baseline_clean["mae"] * (1 + REL_TOL)
    )
    if not mae_ok:
        reasons.append(f"MAE 악화 (후보 {candidate_clean['mae']} > 기준 {baseline_clean['mae']}×{1+REL_TOL:.2f})")

    p95_ok = (
        candidate_clean["p95"] is not None
        and baseline_clean["p95"] is not None
        and candidate_clean["p95"] <= baseline_clean["p95"] * (1 + REL_TOL)
    )
    if not p95_ok:
        reasons.append(f"P95 악화 (후보 {candidate_clean['p95']} > 기준 {baseline_clean['p95']}×{1+REL_TOL:.2f})")

    hard_audio_ok = not any(v["degraded"] for v in hard_audio.values())
    if not hard_audio_ok:
        bad = [vid for vid, v in hard_audio.items() if v["degraded"]]
        reasons.append(f"hard_audio 붕괴곡 악화: {', '.join(bad)}")

    vram_ok = candidate_clean["vram_peak_max"] is None or candidate_clean["vram_peak_max"] <= VRAM_LIMIT_MB
    if not vram_ok:
        reasons.append(f"VRAM 초과 ({candidate_clean['vram_peak_max']} MB > {VRAM_LIMIT_MB} MB)")

    return {
        "passed": mae_ok and p95_ok and hard_audio_ok and vram_ok,
        "reasons": reasons,
        "mae_ok": mae_ok,
        "p95_ok": p95_ok,
        "hard_audio_ok": hard_audio_ok,
        "vram_ok": vram_ok,
    }


def _stratum_table(breakdown: dict[str, dict]) -> list[str]:
    lines = ["| stratum | 곡 | MAE | P95 | VRAM peak(med, MB) | 붕괴 |", "|---|---|---|---|---|---|"]
    for stratum, agg in breakdown.items():
        lines.append(
            "| {st} | {n} | {mae} | {p95} | {vmed} | {c} |".format(
                st=stratum,
                n=agg["n_songs"],
                mae=agg["mae"],
                p95=agg["p95"],
                vmed=agg["vram_peak_median"],
                c=len(agg["collapsed_songs"]),
            )
        )
    return lines


def build_report(bench_dir: Path, baseline_combo: tuple[str, str]) -> str:
    annotations = load_annotations(bench_dir)
    combos = load_runs(bench_dir)
    n_excluded_total = sum(1 for v in annotations.values() if v.get("cause") in INPUT_CONTAMINATION_CAUSES)

    out = [
        "# 정렬 후보 승격(promotion) 리포트",
        "",
        "`benchmark/REPORT.md`(하네스 원 산출물)와 별도 파일이다 — baseline 스윕이 실행 중인",
        "동안에도 `scripts/benchmark_alignment.py`를 건드리지 않고 runs 캐시만 읽어서 만든다.",
        "",
        f"- 기준 조합: `{baseline_combo[0]}__{baseline_combo[1]}`",
        f"- 승격 조건(전체 집계 기준): MAE·P95 비악화(상대허용 {REL_TOL:.0%} — task 지정값 아님, "
        f"측정 잡음 흡수용) + hard_audio 붕괴 곡({', '.join(sorted(HARD_AUDIO_SONGS))}) 비악화 "
        f"+ VRAM(정렬 스테이지 process peak) {VRAM_LIMIT_MB} MB 이내",
        f"- 붕괴 게이트 제외(입력 오염, `collapse_annotations.json` 조인): {n_excluded_total}곡 — "
        "별도 열로만 보고, 승격 판정에는 반영하지 않는다",
        "- 층별 표는 `mixed`(코드스위칭)를 포함한다 — 단일 언어 후보의 코드스위칭 거동을 보는 참고용이고, "
        "승격 판정 자체는 전체(ALL) 집계로 한다.",
        "",
    ]

    if baseline_combo not in combos:
        out += [
            "## 결과",
            "",
            f"기준 조합 `{baseline_combo[0]}__{baseline_combo[1]}`의 runs가 없다 — "
            "아직 baseline 스윕이 끝나지 않았거나 `--baseline` 값이 다르다.",
            "",
        ]
        return "\n".join(out)

    baseline_runs = combos[baseline_combo]
    baseline_stats = combo_stats(baseline_runs, annotations)
    baseline_strata = stratum_breakdown(baseline_runs, annotations)

    out += [
        "## 기준선",
        "",
        f"- clean(게이트 대상, ALL) {baseline_stats['clean']['n_songs']}곡: "
        f"MAE {baseline_stats['clean']['mae']} · P95 {baseline_stats['clean']['p95']} · "
        f"VRAM peak(max) {baseline_stats['clean']['vram_peak_max']} MB · "
        f"붕괴 {len(baseline_stats['clean']['collapsed_songs'])}곡",
        f"- 입력오염 제외 {baseline_stats['excluded']['n_songs']}곡 (별도 열): "
        f"MAE {baseline_stats['excluded']['mae']} · P95 {baseline_stats['excluded']['p95']}",
        "",
        "### 기준선 — 층별(mixed 포함)",
        "",
        *_stratum_table(baseline_strata),
        "",
        "## 후보 — 전체(ALL) 요약 및 승격 판정",
        "",
        "| separator | aligner | clean 곡 | MAE | P95 | VRAM peak(max/med, MB) | hard_audio | 입력오염 제외곡 MAE | 승격 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    candidates = sorted(k for k in combos if k != baseline_combo)
    if not candidates:
        out += ["", "*(후보 조합 없음 — 아직 후보 정렬기가 배선/실행되지 않았다)*", ""]
    for combo in candidates:
        runs = combos[combo]
        stats = combo_stats(runs, annotations)
        hard_audio = hard_audio_check(runs, baseline_runs)
        verdict = promotion_verdict(stats["clean"], baseline_stats["clean"], hard_audio)
        ha_cell = " ".join(f"{vid}:{'악화' if v['degraded'] else 'ok'}" for vid, v in hard_audio.items())
        out.append(
            "| {sep} | {aln} | {n} | {mae} | {p95} | {vmax}/{vmed} | {ha} | {exm} | {verdict} |".format(
                sep=combo[0],
                aln=combo[1],
                n=stats["clean"]["n_songs"],
                mae=stats["clean"]["mae"],
                p95=stats["clean"]["p95"],
                vmax=stats["clean"]["vram_peak_max"],
                vmed=stats["clean"]["vram_peak_median"],
                ha=ha_cell,
                exm=stats["excluded"]["mae"],
                verdict="PASS" if verdict["passed"] else "FAIL: " + "; ".join(verdict["reasons"]),
            )
        )
    out.append("")

    if candidates:
        out += ["## 후보 — 층별 상세(mixed 포함, 참고용)", ""]
        for combo in candidates:
            runs = combos[combo]
            strata = stratum_breakdown(runs, annotations)
            out += [f"### {combo[0]} × {combo[1]}", "", *_stratum_table(strata), ""]

    if annotations:
        out += [
            "## 붕괴 게이트 제외 곡 (입력 오염 — 모델 A/B 신호 아님)",
            "",
            "| video_id | cause | note |",
            "|---|---|---|",
        ]
        for vid, meta in sorted(annotations.items()):
            if meta.get("cause") in INPUT_CONTAMINATION_CAUSES:
                out.append(f"| `{vid}` | {meta.get('cause')} | {meta.get('note', '')} |")
        out.append("")

    return "\n".join(out)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench-dir", default=str(DEFAULT_BENCH_DIR))
    parser.add_argument("--baseline", default="htdemucs__mms-baseline", help="기준 조합 (sep__aln)")
    parser.add_argument("--out", default=None, help="기본값: <bench-dir>/PROMOTION.md")
    args = parser.parse_args()

    bench_dir = Path(args.bench_dir).resolve()
    sep, _, aln = args.baseline.partition("__")
    report = build_report(bench_dir, (sep, aln))
    out_path = Path(args.out) if args.out else bench_dir / "PROMOTION.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"리포트: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
