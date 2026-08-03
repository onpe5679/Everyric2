"""processing 고아 잡 TTL 리퍼 — 진행 하트비트(Job.updated_at)가 끊긴 잡을 주기 회수한다.

server/api/worker.py의 리스 스위퍼(_sweep_expired_leases)는 **원격 워커가 리스를 쥔 잡만**
커버한다(_LEASES 인메모리 레지스트리, 만료 기준 worker_lease_sec 기본 120s). 그 파일의 주석이
명시하듯 인프로세스 워커(local_worker=true)와 _queue_after_line_meta의 "번역 대기" 구간
(worker.LINE_META_WAIT_STAGE)은 리스 없이 정상적으로 processing이라 리스 스위퍼의 대상이
아니다 — 그 두 경로에서 진행 보고가 끊기면(원인 불문: 프로세스 행, GPU 드라이버 정지 등) 잡은
영구히 processing에 남는다.

실측(2026-08, 외부 감사 #7): stage="번역 대기", progress=48로 6.3시간 정체한 잡 하나가 확장
폴링을 48시간 동안 10,779회(전체 트래픽 4%) 발생시켰다. 서버 프로세스 자체는 죽지 않아서
재기동 시 좀비 정리(db/connection.py init_db의 UPDATE ... WHERE status IN (...))도 발화하지
않았다 — 그 정리는 **서버가 다시 뜰 때 한 번**만 돈다.

이 모듈은 서버가 켜져 있는 동안 주기적으로 "진행이 오래 끊긴 processing 잡"을 찾아 회수한다.
판정 기준은 created_at(시작 시각)이 아니라 updated_at(마지막 진행 갱신) — Job.updated_at은
onupdate=func.now()라 JobRepository.update_status/update_status_if 호출마다 갱신되고, 정상
진행 중인 잡은 2~4초 간격으로 그 호출을 받는다(worker.py의 _tick_progress/_stage_monitor).
그래서 오래 걸리는 정상 잡(수 분 규모)은 건드리지 않고, 하트비트 자체가 끊긴 잡만 걸린다.
"""

import asyncio
import contextlib
import logging
from datetime import datetime, timedelta, timezone

from everyric2.config.settings import get_settings
from everyric2.server.db.connection import get_session
from everyric2.server.db.repository import JobRepository

logger = logging.getLogger(__name__)

# 사용자에게 그대로 노출되는 회수 사유 — 확장이 이 문자열을 에러 메시지로 표시한다
ORPHAN_RECOVERY_MESSAGE = "서버가 정체된 작업을 회수했어요 — 다시 시도해 주세요"

# 스윕 주기(초). TTL(기본 45~60분)보다 훨씬 촘촘할 필요는 없다 — 회수 지연 상한이 이
# 간격이므로, 리스 스위퍼(20s, worker_lease_sec=120s 대비)보다 훨씬 느슨하게 잡아도 시간
# 단위 TTL에서는 무시할 수 있는 오차다. 처리할 고아 잡이 없으면 쿼리 하나로 끝난다.
ORPHAN_SWEEP_INTERVAL_SEC = 300.0

_SWEEPER_TASK: asyncio.Task | None = None


async def sweep_orphan_jobs() -> int:
    """updated_at이 TTL을 넘긴 processing 잡을 failed로 회수 — 회수한 수를 반환.

    TTL(server.orphan_job_ttl_min)이 0 이하면 비활성으로 보고 즉시 0. 회수는
    JobRepository.update_status_if(expected=("processing",))로 조건부 쓰기를 써서, 조회와
    쓰기 사이에 그 잡이 다른 경로(정상 완료·취소)로 이미 상태를 벗어난 경우를 덮어쓰지
    않는다 — repository의 update_status_if 문서화 참고(무조건 쓰기는 취소와 경합한다)."""
    ttl_min = get_settings().server.orphan_job_ttl_min
    if ttl_min <= 0:
        return 0
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=ttl_min)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    reaped = 0
    async with get_session() as session:
        repo = JobRepository(session)
        stale = await repo.get_stale_processing(cutoff)
        for job in stale:
            stuck_for = now - job.updated_at
            ok = await repo.update_status_if(
                job.id,
                "failed",
                expected=("processing",),
                error=ORPHAN_RECOVERY_MESSAGE,
            )
            if ok:
                reaped += 1
                logger.warning(
                    "Reaped orphan job %s stuck in processing for %s "
                    "(stage=%s, progress=%s)",
                    job.id,
                    stuck_for,
                    job.stage,
                    job.progress,
                )
    return reaped


async def _orphan_sweep_loop() -> None:
    """ORPHAN_SWEEP_INTERVAL_SEC마다 sweep_orphan_jobs를 돈다 (취소로만 끝난다).

    한 번의 실패로 루프가 죽으면 이 안전망 전체가 사라지므로, 스윕 예외는 로그로 남기고
    다음 주기에 재시도한다(취소는 그대로 전파해야 종료가 된다) — api/worker.py의
    _lease_sweep_loop와 같은 규약."""
    while True:
        await asyncio.sleep(ORPHAN_SWEEP_INTERVAL_SEC)
        try:
            await sweep_orphan_jobs()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Orphan job sweep failed; retrying at the next interval")


def start_orphan_sweeper() -> None:
    """주기 스윕 시작 — **앱 lifespan에서만 호출한다**. 멱등.

    임포트 시점에 태스크를 만들지 않는 이유는 api/worker.py의 start_lease_sweeper와 같다 —
    앱을 띄우지 않고 라우트 코루틴을 직접 await하는 이 레포의 서버 테스트가 실행 중인 루프
    없이 태스크를 만들어 실패하거나 태스크가 남는 것을 막는다."""
    global _SWEEPER_TASK
    if _SWEEPER_TASK is None or _SWEEPER_TASK.done():
        _SWEEPER_TASK = asyncio.create_task(_orphan_sweep_loop())


async def stop_orphan_sweeper() -> None:
    """주기 스윕 취소 + 종료 확인 — lifespan 종료에서 반드시 호출한다(태스크 누수 금지)."""
    global _SWEEPER_TASK
    task = _SWEEPER_TASK
    _SWEEPER_TASK = None
    if task is None:
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
