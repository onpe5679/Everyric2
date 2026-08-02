"""곡 길이별 분리 VRAM 곡선 실측 — 3090(9GB 예산) 이식 판정의 결정 데이터.

서비스 상한: max_audio_duration 2,400s, 파이프라인 청크 1,800s → 분리기는 최악
1,800s 입력을 받는다. 실곡을 타일링한 합성 장곡으로 길이 축을 만들고, 분리기별로
깨끗한 GPU 상태(다른 모델 미상주)에서 시간·VRAM을 잰다.

- htdemucs: 서브프로세스라 디바이스 폴링(VramProbe)이 유일한 관찰 수단 —
  측정 전 유휴 사용량을 빼서 delta도 남긴다.
- kimft-melband: 워커가 stdout JSON으로 내부 alloc/reserved 피크를 보고한다.

    ./.venv/Scripts/python.exe scripts/vram_curve.py --durations 240,600,1200,1800

산출: benchmark/VRAM_CURVE.md + vram_curve.json. 합성 오디오·스템은 측정 후 삭제.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "benchmark"
WORK = BENCH / "vramcurve"

_spec = importlib.util.spec_from_file_location("bench", REPO / "scripts" / "benchmark_alignment.py")
_bench = importlib.util.module_from_spec(_spec)
sys.modules["bench"] = _bench
_spec.loader.exec_module(_bench)

SOURCE_SONG = "hFTs6HbtxbE"  # QWER 고민중독 — 평범한 실곡 (타일링 원료)


def synth_audio(duration: int) -> Path:
    """실곡을 타일링해 duration초 wav(24kHz 모노)를 만든다 — ffmpeg 불요."""
    import numpy as np
    import soundfile as sf

    dest = WORK / "audio" / f"tile_{duration}s.wav"
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    src = next(iter((BENCH / "audio").glob(f"{SOURCE_SONG}.*")))
    import librosa

    y, sr = librosa.load(str(src), sr=24000, mono=True)
    need = duration * sr
    reps = int(need // len(y)) + 1
    tiled = np.tile(y, reps)[:need]
    sf.write(str(dest), tiled, sr)
    return dest


def idle_device_mb() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
        return round((total - free) / 2**20, 1)
    except Exception:
        return None


def _smi_used_mb() -> float | None:
    """nvidia-smi 전역 memory.used(MiB) — WDDM에서 타 프로세스(demucs 서브프로세스)까지
    보이는 유일한 관찰 수단. torch mem_get_info는 자기 프로세스 컨텍스트만 반영한다."""
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        return float(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


class SmiProbe:
    """분리 실행 동안 nvidia-smi 전역 사용량 피크를 폴링한다.

    타 프로세스(브라우저·DWM) 사용량이 섞이므로 idle 스냅숏 대비 delta로 판정한다."""

    def __init__(self, interval: float = 0.3):
        self.interval = interval
        self.peak_mb: float | None = None
        self._stop = False
        self._thread = None

    def _loop(self):
        while not self._stop:
            used = _smi_used_mb()
            if used is not None and (self.peak_mb is None or used > self.peak_mb):
                self.peak_mb = used
            import time as _t

            _t.sleep(self.interval)

    def __enter__(self):
        import threading

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop = True
        if self._thread:
            self._thread.join(timeout=3)
        return False


def measure(sep_name: str, audio: Path, duration: int) -> dict:
    adapter = _bench.SEPARATORS[sep_name]()
    work = WORK / sep_name / f"{duration}s"
    shutil.rmtree(work, ignore_errors=True)
    idle = idle_device_mb()
    smi_idle = _smi_used_mb()
    row: dict = {
        "separator": sep_name, "duration_s": duration,
        "idle_device_mb": idle, "smi_idle_mb": smi_idle,
    }
    started = time.perf_counter()
    try:
        with SmiProbe(interval=0.3) as smi, _bench.VramProbe(interval=0.2) as probe:
            out = adapter.separate(audio, work)
        row.update(
            {
                "elapsed_s": round(time.perf_counter() - started, 1),
                "device_peak_mb": probe.device_peak_mb,
                "device_delta_mb": (
                    round(probe.device_peak_mb - idle, 1)
                    if probe.device_peak_mb is not None and idle is not None
                    else None
                ),
                "smi_peak_mb": smi.peak_mb,
                "smi_delta_mb": (
                    round(smi.peak_mb - smi_idle, 1)
                    if smi.peak_mb is not None and smi_idle is not None
                    else None
                ),
                "worker_alloc_peak_mb": out.vram_peak_mb,
                "worker_reserved_peak_mb": out.vram_device_peak_mb,
            }
        )
    except Exception as exc:
        row["error"] = repr(exc)[:300]
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--separators", default="htdemucs,kimft-melband")
    ap.add_argument("--durations", default="240,600,1200,1800")
    args = ap.parse_args()
    _bench._register_optional_aligners()

    seps = [s.strip() for s in args.separators.split(",") if s.strip()]
    durs = [int(d) for d in args.durations.split(",")]
    for s in seps:
        if s not in _bench.SEPARATORS:
            raise SystemExit(f"미등록 분리기: {s} (등록: {', '.join(_bench.SEPARATORS)})")

    rows = []
    for dur in durs:
        audio = synth_audio(dur)
        for sep in seps:
            print(f"[{dur}s × {sep}] 측정 중...", flush=True)
            row = measure(sep, audio, dur)
            rows.append(row)
            print(f"  -> {json.dumps(row, ensure_ascii=False)}", flush=True)

    # 기존 결과와 병합 — 이번에 안 잰 (분리기, 길이) 행은 보존한다
    json_path = BENCH / "vram_curve.json"
    if json_path.exists():
        try:
            prev = json.loads(json_path.read_text(encoding="utf-8"))
            fresh = {(r["separator"], r["duration_s"]) for r in rows}
            rows = [r for r in prev if (r["separator"], r["duration_s"]) not in fresh] + rows
            rows.sort(key=lambda r: (r["duration_s"], r["separator"]))
        except Exception:
            pass
    json_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    md = ["# 곡 길이별 분리 VRAM 곡선 (5090 실측)", "",
          f"- 원료 곡: {SOURCE_SONG} 타일링 (24kHz 모노 wav) · 측정 시각: {time.strftime('%Y-%m-%d %H:%M')}",
          "- «smi delta»: nvidia-smi 전역 사용량의 유휴 대비 증가분 — 서브프로세스(demucs)까지"
          " 보이는 유일한 수단. 타 프로세스 활동이 섞일 수 있어 ±수백MB 노이즈 감안.",
          "- torch 디바이스 폴링은 WDDM에서 타 프로세스가 안 보여 서브프로세스 측정엔 무효"
          " (구버전 표의 htdemucs 0MB가 그 흔적).",
          "- kimft는 워커 내부 alloc(실사용)/reserved(OS 관점 상한) 피크가 정확한 수치다.",
          "- 3090 이식 판정 기준: 예산 9GB — reserved/delta가 길이에 따라 자라는지가 관전 포인트.", "",
          "| 길이 | 분리기 | 시간 | smi delta | 워커 alloc 피크 | 워커 reserved 피크 | 오류 |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(
            f"| {r['duration_s']}s | {r['separator']} | {r.get('elapsed_s', '-')}s "
            f"| {r.get('smi_delta_mb', '-')}MB | {r.get('worker_alloc_peak_mb', '-')}MB "
            f"| {r.get('worker_reserved_peak_mb', '-')}MB | {(r.get('error') or '')[:60]} |"
        )
    (BENCH / "VRAM_CURVE.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("저장: benchmark/VRAM_CURVE.md, vram_curve.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
