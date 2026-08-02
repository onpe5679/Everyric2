import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from everyric2.server.db.models import Base

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./everyric2.db")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)
elif DATABASE_URL.startswith("postgresql://") and "+asyncpg" not in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

# SQLite: 잦은 잡 진행률 update + 다중 탭 폴링이 겹치면 기본 설정으로는
# "database is locked"가 난다 — busy timeout과 WAL로 동시성을 확보한다
_connect_args = {"timeout": 30} if DATABASE_URL.startswith("sqlite") else {}
engine = create_async_engine(DATABASE_URL, echo=False, connect_args=_connect_args)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # create_all은 기존 테이블에 새 컬럼을 추가하지 않는다 — 가벼운 수동 보강 (SQLite 전용)
        if DATABASE_URL.startswith("sqlite"):
            from sqlalchemy import text

            await conn.execute(text("PRAGMA journal_mode=WAL"))
            cols = {row[1] for row in await conn.execute(text("PRAGMA table_info(jobs)"))}
            if "stage" not in cols:
                await conn.execute(text("ALTER TABLE jobs ADD COLUMN stage VARCHAR(24)"))
            if "target_lang" not in cols:
                await conn.execute(
                    text("ALTER TABLE jobs ADD COLUMN target_lang VARCHAR(8) DEFAULT 'ko'")
                )
            if "failure_kind" not in cols:
                await conn.execute(text("ALTER TABLE jobs ADD COLUMN failure_kind VARCHAR(16)"))
            link_cols = {
                row[1] for row in await conn.execute(text("PRAGMA table_info(sync_links)"))
            }
            if link_cols and "rate" not in link_cols:
                await conn.execute(
                    text("ALTER TABLE sync_links ADD COLUMN rate FLOAT DEFAULT 1.0")
                )
            if link_cols and "verified" not in link_cols:
                await conn.execute(
                    text("ALTER TABLE sync_links ADD COLUMN verified BOOLEAN DEFAULT 0")
                )
                # 일회성 백필: 같은 쌍으로 match=true 검증 잡이 끝난 링크는 자동 생성분이다.
                # 나머지(수동 링크 API로 박힌 것)는 0으로 남겨 미검증으로 표시한다.
                await conn.execute(
                    text(
                        "UPDATE sync_links SET verified=1 WHERE EXISTS ("
                        "  SELECT 1 FROM link_jobs lj"
                        "  WHERE lj.video_id = sync_links.video_id"
                        "    AND lj.source_video_id = sync_links.source_video_id"
                        "    AND lj.status = 'done' AND lj.match = 1)"
                    )
                )
            sync_cols = {
                row[1] for row in await conn.execute(text("PRAGMA table_info(sync_results)"))
            }
            # 제목/아티스트: 커버 링크 후보 탐색용. 기존 행은 NULL로 남고 조회 시 백필된다
            if sync_cols and "title" not in sync_cols:
                await conn.execute(text("ALTER TABLE sync_results ADD COLUMN title VARCHAR(256)"))
            if sync_cols and "artist" not in sync_cols:
                await conn.execute(text("ALTER TABLE sync_results ADD COLUMN artist VARCHAR(128)"))
        # 서버가 죽으며 남긴 좀비 잡(pending/processing) 정리 — 방치하면 같은 영상의
        # 생성 요청이 죽은 잡에 합류해 영구 "전사 중"에 갇힌다
        from sqlalchemy import text as _text

        # failure_kind='system' — 서버 프로세스 자체가 죽어 중단된 것이지 사용자 취소도
        # 다운로드의 외부 요인도 아니다(MoRef 감사 #3).
        result = await conn.execute(
            _text(
                "UPDATE jobs SET status='failed', failure_kind='system', "
                "error='서버 재시작으로 중단된 작업이에요. 다시 생성해 주세요.' "
                "WHERE status IN ('pending', 'processing', 'queued')"
            )
        )
        if result.rowcount:
            import logging

            logging.getLogger(__name__).warning(
                f"Reset {result.rowcount} zombie jobs left from a previous run"
            )
        # 링크 검증 잡도 같은 좀비를 남긴다 — 이 정리가 없으면 processing으로 영구 잔류하고
        # 리스 레지스트리(_LEASES)는 재기동으로 유실돼 만료 스윕도 되돌릴 수 없다.
        # get_active_pair가 그 잡을 활성(queued/processing)으로 보므로 같은 (커버, 원곡) 쌍은
        # 영구 pending이 되어 그 커버는 다시는 링크를 얻지 못한다.
        #
        # **failed가 아니라 queued로 되돌린다.** sync 잡을 failed로 마감하는 이유는 인메모리
        # 스태시(line_meta·출처·제목·force)가 재기동으로 유실돼 그대로 재개하면 번역·독음이
        # 조용히 사라지기 때문인데, 링크 잡은 행 안의 (video_id, source_video_id)만으로
        # 완결돼 유실될 상태가 없다. 오히려 failed가 더 해롭다: get_recent_attempt가 최근
        # failed를 쿨다운(link_retry_cooldown_days, 기본 14일)으로 세어 같은 쌍의 자동 재제출을
        # 막으므로 그 커버는 2주간 링크를 못 얻는다. queued면 워커가 다시 물어 완주한다.
        # (queued 링크 잡은 손대지 않는다 — 아직 아무도 물지 않았으므로 그대로 유효하다.)
        link_result = await conn.execute(
            _text("UPDATE link_jobs SET status='queued' WHERE status='processing'")
        )
        if link_result.rowcount:
            import logging

            logging.getLogger(__name__).warning(
                f"Requeued {link_result.rowcount} zombie link jobs left from a previous run"
            )


async def close_db():
    await engine.dispose()


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
