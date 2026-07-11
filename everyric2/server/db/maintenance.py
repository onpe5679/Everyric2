"""구 파이프라인 캐시 싱크 정리 유틸.

신 파이프라인 싱크는 곡 단위 debug 메타에 ``alignment_text``("pronunciation" | "original")를
기록한다. 이 필드가 없는 행은 구 파이프라인 산출물이므로 정리 대상이다.

기본은 dry-run(카운트/목록만 출력) — 실제 삭제는 ``--apply`` 를 줘야 한다. 실행 중인
서버가 있으면 SQLite 파일 락과 캐시 불일치가 생길 수 있으니 **서버 재기동/정지 후** 돌리고,
실행 전 DB는 git 체크포인트 등으로 백업해 둘 것.

    uv run python -m everyric2.server.db.maintenance            # dry-run (기본)
    uv run python -m everyric2.server.db.maintenance --apply    # 실제 삭제
"""
from __future__ import annotations

import argparse
import asyncio
from typing import Any

from sqlalchemy import select

from everyric2.server.db.models import SyncResult


def is_old_pipeline(timestamps: Any) -> bool:
    """싱크 timestamps(JSON)가 구 파이프라인 산출물인지 — debug.alignment_text 부재 여부."""
    if not isinstance(timestamps, dict):
        return True
    debug = timestamps.get("debug")
    if not isinstance(debug, dict):
        return True
    return not debug.get("alignment_text")


async def classify_syncs(session) -> tuple[list[SyncResult], list[SyncResult]]:
    """(구 파이프라인 행, 신 파이프라인 행)으로 분류. 데이터셋이 작아 전 행 로드로 충분."""
    rows = list((await session.execute(select(SyncResult))).scalars().all())
    old = [r for r in rows if is_old_pipeline(r.timestamps)]
    new = [r for r in rows if not is_old_pipeline(r.timestamps)]
    return old, new


async def reset_old_pipeline_syncs(session, apply: bool = False) -> dict[str, Any]:
    """구 파이프라인 싱크 행을 (apply=True면) 삭제. 신 파이프라인 행은 보존.

    반환: {"old": [...], "kept": [...], "applied": bool} — old/kept는 (video_id, id) 목록.
    apply=False(dry-run)면 아무것도 지우지 않는다.
    """
    old, new = await classify_syncs(session)
    if apply:
        for r in old:
            await session.delete(r)
        await session.flush()
    return {
        "old": [(r.video_id, r.id) for r in old],
        "kept": [(r.video_id, r.id) for r in new],
        "applied": apply,
    }


async def _run(apply: bool) -> None:
    from everyric2.server.db.connection import get_session

    async with get_session() as session:
        result = await reset_old_pipeline_syncs(session, apply=apply)
    mode = "APPLY (삭제 실행)" if apply else "DRY-RUN (미삭제)"
    print(f"[{mode}] 구 파이프라인 {len(result['old'])}행, 신 파이프라인(유지) {len(result['kept'])}행")
    print("삭제 대상 (video_id):", [v for v, _ in result["old"]])
    print("유지 (video_id):", [v for v, _ in result["kept"]])
    if not apply and result["old"]:
        print("실제 삭제하려면 --apply 를 붙여 다시 실행하세요 (서버 재기동/정지 후, DB 백업 후).")


def main() -> None:
    parser = argparse.ArgumentParser(description="구 파이프라인 캐시 싱크 정리")
    parser.add_argument(
        "--apply", action="store_true",
        help="실제로 삭제한다 (미지정 시 dry-run). 서버 재기동/정지 + DB 백업 후 실행 권장.",
    )
    args = parser.parse_args()
    asyncio.run(_run(args.apply))


if __name__ == "__main__":
    main()
