"""실제 쓰기 경로 검증 — regenerate(force)로 새 싱크를 만들고 이전 싱크·자막 기준과 비교한다.

릴리스 검증 기준(«쓰기 경로 통과 없이는 릴리스 금지»)의 자동화다. 오프라인 벤치
(align_bench.py)는 엔진만 태우므로, 배포된 서버의 진짜 경로(다운로드 캐시 → 분리 →
정렬 → 저장 → line_meta 병합)를 이것으로 태운다.

발음 소실 함정: regenerate만 부르면 발음이 날아간다(발음은 /api/translate → line_meta
경로다). 그래서 **이전 싱크의 세그먼트에서 line_meta를 뽑아 본문에 실어 보낸다** —
새 싱크에도 발음·번역이 붙어 있어야 통과다.

서버 자체에서 실행한다 (127.0.0.1 — localhost는 IPv6 스톨). admin 키는 환경변수
EVERYRIC_SERVER_ADMIN_API_KEY로 받는다 — 값을 출력하지 않는다.

    set -a; source <ENV_FILE>; set +a
    .venv/bin/python scripts/verify_regen.py --video-id zyRt-nBM3dY ...
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
import urllib.request
from pathlib import Path

from everyric2.server.api.quota_headers import ACTOR_HEADER, OPS_ACTOR_VERIFY_REGEN

BASE = "http://127.0.0.1:8300"


def _api(path: str, payload: dict | None = None, key: str | None = None) -> dict:
    req = urllib.request.Request(BASE + path, method="POST" if payload is not None else "GET")
    req.add_header("Content-Type", "application/json")
    # 앞단 게이트웨이를 우회해 직접 부르므로 **운영 작업 식별자**를 싣는다
    # (docs/user-quota-spec.md §8). 없으면 §1의 fail-closed에 걸려 400이고, 사람 식별자를
    # 빌려 쓰면 그 사람의 예산·통계가 검증 실행으로 오염된다. 상한은 면제, 기록은 남는다.
    req.add_header(ACTOR_HEADER, OPS_ACTOR_VERIFY_REGEN)
    if key:
        req.add_header("x-api-key", key)
    data = json.dumps(payload).encode() if payload is not None else None
    with urllib.request.urlopen(req, data, timeout=60) as r:
        return json.loads(r.read())


def _latest_sync(con, vid: str) -> dict | None:
    row = con.execute(
        "SELECT id, timestamps, quality_score, created_at FROM sync_results "
        "WHERE video_id=? ORDER BY created_at DESC LIMIT 1",
        (vid,),
    ).fetchone()
    if not row:
        return None
    ts = json.loads(row[1])
    return {
        "id": row[0],
        "segments": ts.get("segments") or [],
        "debug": ts.get("debug") or {},
        "quality": row[2],
        "created_at": row[3],
    }


def _caption_reference(vid: str, lines: list[str], captions_dir: Path) -> dict[int, float]:
    from everyric2.alignment.caption_anchors import MIN_KEY_LEN, anchor_key, match_anchors

    path = captions_dir / f"{vid}.json"
    if not path.exists():
        return {}
    tracks = json.loads(path.read_text(encoding="utf-8"))
    matchable = sum(1 for t in lines if len(anchor_key(t)) >= MIN_KEY_LEN) or 1
    best_rate, best = 0.0, []
    for _lang, events in tracks:
        anchors = match_anchors(lines, events)
        rate = len(anchors) / matchable
        if rate > best_rate:
            best_rate, best = rate, anchors
    return {a.line_idx: a.start for a in best} if best_rate >= 0.5 else {}


def _vs_ref(segments: list[dict], ref: dict[int, float]) -> dict | None:
    import numpy as np

    errs = [
        segments[i]["start"] - t
        for i, t in ref.items()
        if i < len(segments) and segments[i].get("start") is not None
    ]
    if len(errs) < 5:
        return None
    a = np.abs(np.array(errs))
    return {
        "n": len(errs),
        "mae": round(float(np.median(a)), 3),
        "aae": round(float(np.mean(a)), 3),
        "pco_0.3": round(float(np.mean(a <= 0.3)), 3),
        "signed_median": round(float(np.median(errs)), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video-id", action="append", required=True)
    ap.add_argument("--db", default="everyric2.db")
    ap.add_argument("--captions-dir", default="bench/captions")
    ap.add_argument("--timeout", type=float, default=420.0)
    args = ap.parse_args()

    key = os.environ.get("EVERYRIC_SERVER_ADMIN_API_KEY") or None
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    out: dict[str, dict] = {}

    for vid in args.video_id:
        before = _latest_sync(con, vid)
        if before is None:
            print(f"{vid}: 기존 싱크 없음 — 건너뜀")
            continue
        lyrics_row = con.execute(
            "SELECT lyrics FROM jobs WHERE video_id=? AND status='completed' "
            "ORDER BY created_at DESC LIMIT 1",
            (vid,),
        ).fetchone()
        lyrics = lyrics_row[0]

        seen: set[str] = set()
        line_meta = []
        for s in before["segments"]:
            text = s.get("text") or ""
            if text in seen:
                continue
            seen.add(text)
            if s.get("pronunciation") or s.get("translation"):
                line_meta.append(
                    {
                        "text": text,
                        "pronunciation": s.get("pronunciation"),
                        "translation": s.get("translation"),
                    }
                )

        resp = _api(
            "/api/sync/regenerate",
            {"video_id": vid, "lyrics": lyrics, "force": True, "line_meta": line_meta or None},
            key,
        )
        job_id = resp.get("job_id")
        print(f"{vid}: job {job_id} ({resp.get('status')}), line_meta {len(line_meta)}줄 동봉")
        if not job_id:
            out[vid] = {"error": f"no job: {resp}"}
            continue

        deadline = time.monotonic() + args.timeout
        status = None
        while time.monotonic() < deadline:
            # 이 배포는 모든 /api에 키를 요구한다 (health까지 401) — 조회에도 싣는다
            st = _api(f"/api/job/{job_id}", key=key)
            status = st.get("status")
            if status in ("completed", "failed"):
                break
            time.sleep(5)
        if status != "completed":
            out[vid] = {"error": f"job {status}"}
            print(f"{vid}: 실패/시간초과 — {status}")
            continue

        # 새 커넥션으로 최신 행을 읽는다 (ro 커넥션 캐시 회피)
        con2 = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
        after = _latest_sync(con2, vid)
        lines = [ln for ln in lyrics.splitlines() if ln.strip()]
        ref = _caption_reference(vid, lines, Path(args.captions_dir))

        def _meta_counts(segs):
            return (
                sum(1 for s in segs if s.get("pronunciation")),
                sum(1 for s in segs if s.get("translation")),
            )

        b_pron, b_tr = _meta_counts(before["segments"])
        a_pron, a_tr = _meta_counts(after["segments"])
        result = {
            "new_sync": after["id"] != before["id"],
            "segments": [len(before["segments"]), len(after["segments"])],
            "quality": [before["quality"], after["quality"]],
            "pron": [b_pron, a_pron],
            "translation": [b_tr, a_tr],
            "vs_ref_before": _vs_ref(before["segments"], ref),
            "vs_ref_after": _vs_ref(after["segments"], ref),
            "star_spans_after": (after["debug"].get("star_spans") or [])[:6],
        }
        out[vid] = result
        print(vid, json.dumps(result, ensure_ascii=False))

    Path("bench/out").mkdir(parents=True, exist_ok=True)
    Path("bench/out/verify_regen.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print("saved: bench/out/verify_regen.json")


if __name__ == "__main__":
    main()
