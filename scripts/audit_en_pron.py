#!/usr/bin/env python
"""완료된 싱크의 **영어 라인 한글 발음**을 전수 역검사한다.

``everyric2.text.pron_backcheck``를 DB 전체에 돌려 「원문 IPA에서 나올 수 없는 표기」를
곡별·라인별로 모은다. 판정 규칙과 그 한계는 그 모듈 문서에 있고, 여기서는 **어디서 무엇을
꺼내는가**만 정한다.

DB 경로는 서버와 같은 곳을 본다 — ``everyric2.server.db.connection.DATABASE_URL``
(환경변수 ``DATABASE_URL``, 기본 ``sqlite+aiosqlite:///./everyric2.db``)을 그대로 읽는다.
읽기 전용(``mode=ro``)으로 연다: 감사 도구가 운영 DB를 건드릴 이유가 없고, 서버가 WAL로
돌고 있어도 안전하다.

**손상 행을 건너뛴다.** 로컬 개발 DB가 실제로 손상돼 있었다(``PRAGMA quick_check``가
b-tree 순환을 보고한다). 통짜 ``SELECT``는 한 행 때문에 전부 실패하므로 rowid로 하나씩
읽고 실패한 행을 세어 리포트에 남긴다 — 조용히 빠뜨리면 「전수」가 거짓말이 된다.

발음의 출처는 세그먼트마다 두 자리다(``server/api/sync.py``·``worker.attach_pron_variants``):
``seg["pron"]["hangul"]``(표기별 dict, 신형)와 ``seg["pronunciation"]``(레거시 한글 전용).
앞을 우선하고 없으면 뒤를 쓴다.

사용:
    PYTHONPATH=. python scripts/audit_en_pron.py
    PYTHONPATH=. python scripts/audit_en_pron.py --json report.json --db other.db
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from everyric2.text.pron_backcheck import check_line


def _resolve_db_path(override: str | None) -> Path:
    """서버와 같은 규칙으로 sqlite 파일 경로를 정한다."""
    if override:
        return Path(override)
    from everyric2.server.db.connection import DATABASE_URL

    if not DATABASE_URL.startswith("sqlite"):
        raise SystemExit(
            f"sqlite가 아닌 DB는 이 도구가 읽지 않는다: {DATABASE_URL.split('://')[0]}://…\n"
            "--db로 sqlite 파일을 직접 주거나, 서버에서 덤프를 받아서 돌려라."
        )
    # "sqlite+aiosqlite:///./everyric2.db" → "./everyric2.db"
    return Path(DATABASE_URL.split(":///", 1)[1])


def _pron_of(seg: dict[str, Any]) -> str:
    """세그먼트의 한글 발음 — 신형 ``pron.hangul`` 우선, 없으면 레거시 ``pronunciation``."""
    pron = seg.get("pron")
    if isinstance(pron, dict):
        hangul = (pron.get("hangul") or "").strip()
        if hangul:
            return hangul
    return (seg.get("pronunciation") or "").strip()


# 리포트에 쓰는 컬럼. 이 중 **실제로 존재하는 것만** 고른다 — DB가 서버 코드보다 오래돼
# 컬럼이 없을 수 있다(engine_version은 나중에 생겼고 구 DB에는 없다). 서버의 init_db도
# 같은 방식으로 PRAGMA table_info를 보고 컬럼 유무를 판단한다.
_WANTED_COLUMNS = (
    "id", "video_id", "language", "engine_variant", "engine_version",
    "title", "timestamps", "created_at",
)


def _iter_rows(conn: sqlite3.Connection) -> tuple[list[dict[str, Any]], int, list[str]]:
    """sync_results를 rowid 하나씩 읽는다. (읽힌 행, 손상으로 못 읽은 행 수, 없는 컬럼).

    한 행씩 읽는 이유는 손상 때문이다: 통짜 ``SELECT``는 깨진 b-tree 페이지 하나에 걸리면
    전부 실패한다. 다만 **손상만** 건너뛴다 — ``OperationalError``(컬럼 없음 등)까지 같이
    삼키면 스키마가 안 맞는 것을 「행 손상」으로 뒤집어씌워 리포트가 거짓말을 한다.
    """
    present = {row[1] for row in conn.execute("PRAGMA table_info(sync_results)")}
    missing = [c for c in _WANTED_COLUMNS if c not in present]
    columns = [c for c in _WANTED_COLUMNS if c in present]
    if "timestamps" not in columns:
        raise SystemExit("sync_results에 timestamps 컬럼이 없다 — 이 DB는 조사 대상이 아니다.")

    rowids = [r[0] for r in conn.execute("SELECT rowid FROM sync_results")]
    select = f"SELECT {', '.join(columns)} FROM sync_results WHERE rowid=?"

    rows: list[dict[str, Any]] = []
    broken = 0
    for rowid in rowids:
        try:
            row = conn.execute(select, (rowid,)).fetchone()
        except sqlite3.OperationalError:
            raise  # 스키마·질의 오류는 손상이 아니다. 조용히 세지 말고 터뜨린다.
        except sqlite3.DatabaseError:
            broken += 1
            continue
        if row is not None:
            rows.append({c: row[c] for c in columns})
    return rows, broken, missing


def audit(db_path: Path) -> dict[str, Any]:
    if not db_path.exists():
        raise SystemExit(f"DB 파일이 없다: {db_path.resolve()}")

    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows, broken, missing_columns = _iter_rows(conn)
    finally:
        conn.close()

    songs: list[dict[str, Any]] = []
    totals = Counter()
    skip_reasons = Counter()
    part_counts = Counter()

    for row in rows:
        totals["싱크"] += 1
        raw = row.get("timestamps")
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, ValueError):
            totals["timestamps 파싱 실패"] += 1
            continue
        segments = (data or {}).get("segments") or []

        findings: list[dict[str, Any]] = []
        judged = 0
        for index, seg in enumerate(segments):
            text = (seg.get("text") or "").strip()
            pron = _pron_of(seg)
            if not text or not pron:
                continue
            totals["발음 있는 라인"] += 1
            verdict = check_line(text, pron)
            if verdict.scope == "skipped":
                skip_reasons[verdict.skip_reason or "(사유 없음)"] += 1
                totals["판정 제외"] += 1
                continue
            judged += 1
            totals["판정한 라인"] += 1
            totals["판정한 음절"] += verdict.checked
            if verdict.impossible:
                totals["불가능 라인"] += 1
                for item in verdict.impossible:
                    part_counts[f"{item.part} {item.value}"] += 1
                findings.append(
                    {
                        "line": index,
                        "text": verdict.text,
                        "pron": verdict.pron,
                        "scope": verdict.scope,
                        "impossible": [
                            {
                                "syllable": f.syllable,
                                "position": f.position,
                                "word": f.word,
                                "part": f.part,
                                "value": f.value,
                                "reason": f.reason,
                            }
                            for f in verdict.impossible
                        ],
                    }
                )

        if judged:
            totals["영어 라인이 있는 싱크"] += 1
        if findings:
            songs.append(
                {
                    "sync_id": row.get("id"),
                    "video_id": row.get("video_id"),
                    "language": row.get("language"),
                    "engine_version": row.get("engine_version"),
                    "title": row.get("title"),
                    "judged_lines": judged,
                    "lines": findings,
                }
            )

    return {
        "db": str(db_path.resolve()),
        "unreadable_rows": broken,
        "missing_columns": missing_columns,
        "totals": dict(totals),
        "skip_reasons": dict(skip_reasons),
        "part_counts": dict(part_counts),
        "songs": songs,
    }


def _print_report(report: dict[str, Any]) -> None:
    totals = report["totals"]
    print(f"DB: {report['db']}")
    if report["unreadable_rows"]:
        print(f"경고: 손상으로 읽지 못한 sync_results 행 {report['unreadable_rows']}개 — "
              "이 리포트는 그만큼 전수가 아니다")
    if report["missing_columns"]:
        print(f"참고: 이 DB에 없는 컬럼 {', '.join(report['missing_columns'])} "
              "— 서버 코드보다 오래된 DB다(판정 자체에는 영향 없음)")
    print()

    if not totals.get("싱크"):
        print("sync_results에 행이 없다. 조사할 싱크가 없으므로 여기서 끝낸다.")
        return

    print(f"싱크 {totals.get('싱크', 0)}개 · 발음이 붙은 라인 {totals.get('발음 있는 라인', 0)}개")
    print(f"  판정한 라인 {totals.get('판정한 라인', 0)}개 "
          f"(음절 {totals.get('판정한 음절', 0)}개) / 판정 제외 {totals.get('판정 제외', 0)}개")
    print(f"  영어 라인이 있는 싱크 {totals.get('영어 라인이 있는 싱크', 0)}개")
    print()

    if not totals.get("판정한 라인"):
        print("판정 가능한 영어 라인이 하나도 없었다 — 불가능 표기를 찾은 것이 아니라")
        print("**조사 대상 자체가 없었다.** 제외 사유는 아래와 같다:")
        for reason, count in sorted(report["skip_reasons"].items(), key=lambda kv: -kv[1]):
            print(f"  {count:6d}  {reason}")
        return

    flagged = totals.get("불가능 라인", 0)
    rate = flagged / totals["판정한 라인"] * 100
    print(f"불가능 표기가 있는 라인: {flagged}개 ({rate:.2f}%)")
    if report["part_counts"]:
        print("  성분별:", ", ".join(
            f"{k} {v}" for k, v in sorted(report["part_counts"].items(), key=lambda kv: -kv[1])
        ))
    print()

    for song in report["songs"]:
        label = song["title"] or song["video_id"]
        print(f"── {label} ({song['video_id']}, lang={song['language']}, "
              f"engine={song['engine_version']}) — {len(song['lines'])}줄")
        for line in song["lines"]:
            print(f"   [{line['line']}] {line['text']}")
            print(f"        발음: {line['pron']}   (판정 범위: {line['scope']})")
            for item in line["impossible"]:
                owner = f" · 낱말 {item['word']}" if item["word"] else ""
                print(f"        └ {item['syllable']} — {item['reason']}{owner}")
        print()

    if report["skip_reasons"]:
        print("판정 제외 사유:")
        for reason, count in sorted(report["skip_reasons"].items(), key=lambda kv: -kv[1]):
            print(f"  {count:6d}  {reason}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", help="sqlite 파일 경로 (기본: 서버와 같은 DATABASE_URL)")
    parser.add_argument("--json", help="리포트를 이 경로에 JSON으로 쓴다")
    args = parser.parse_args()

    report = audit(_resolve_db_path(args.db))
    _print_report(report)
    if args.json:
        Path(args.json).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nJSON 리포트: {Path(args.json).resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
