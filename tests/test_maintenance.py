"""구 파이프라인 캐시 싱크 정리 테스트.

격리된 in-memory SQLite에서 reset_old_pipeline_syncs를 직접 호출한다(실 DB 무영향).
신 파이프라인은 debug.alignment_text가 있고, 구 파이프라인은 없다.
"""
import asyncio

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from everyric2.server.db.maintenance import (
    is_old_pipeline,
    reset_old_pipeline_syncs,
)
from everyric2.server.db.models import Base, SyncResult
from everyric2.server.db.repository import SyncRepository


def test_is_old_pipeline_classification():
    # 신 파이프라인: debug.alignment_text 존재
    assert is_old_pipeline({"segments": [], "debug": {"alignment_text": "pronunciation"}}) is False
    assert is_old_pipeline({"segments": [], "debug": {"alignment_text": "original"}}) is False
    # 구 파이프라인: debug 없음 / alignment_text 없음 / 빈 값 / dict 아님
    assert is_old_pipeline({"segments": []}) is True
    assert is_old_pipeline({"segments": [], "debug": {"vad_regions": []}}) is True
    assert is_old_pipeline({"segments": [], "debug": {"alignment_text": None}}) is True
    assert is_old_pipeline({"segments": [], "debug": {"alignment_text": ""}}) is True
    assert is_old_pipeline(None) is True


async def _seed(sessionmaker):
    async with sessionmaker() as s:
        repo = SyncRepository(s)
        # 신 파이프라인 2곡 (유지)
        await repo.create("NEW1", "h", [{"text": "a"}], extra={"debug": {"alignment_text": "pronunciation"}})
        await repo.create("NEW2", "h", [{"text": "b"}], extra={"debug": {"alignment_text": "original"}})
        # 구 파이프라인 3곡 (삭제 대상)
        await repo.create("OLD1", "h", [{"text": "c"}])  # debug 없음
        await repo.create("OLD2", "h", [{"text": "d"}], extra={"debug": {"vad_regions": []}})
        await repo.create("OLD3", "h", [{"text": "e"}], extra={"tempo": {"bpm": 120}})
        await s.commit()


async def _count(sessionmaker) -> int:
    async with sessionmaker() as s:
        return (await s.execute(select(func.count()).select_from(SyncResult))).scalar_one()


async def _with_db(fn):
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    try:
        await _seed(sm)
        await fn(sm)
    finally:
        await engine.dispose()


def test_dry_run_does_not_delete():
    async def body():
        async def check(sm):
            async with sm() as s:
                result = await reset_old_pipeline_syncs(s, apply=False)
                await s.commit()
            assert result["applied"] is False
            assert sorted(v for v, _ in result["old"]) == ["OLD1", "OLD2", "OLD3"]
            assert sorted(v for v, _ in result["kept"]) == ["NEW1", "NEW2"]
            assert await _count(sm) == 5  # 아무것도 안 지움

        await _with_db(check)

    asyncio.run(body())


def test_apply_deletes_only_old_pipeline():
    async def body():
        async def check(sm):
            async with sm() as s:
                result = await reset_old_pipeline_syncs(s, apply=True)
                await s.commit()
            assert result["applied"] is True
            assert len(result["old"]) == 3 and len(result["kept"]) == 2
            assert await _count(sm) == 2  # 신 파이프라인 2곡만 남음
            async with sm() as s:
                remaining = sorted(
                    (await s.execute(select(SyncResult.video_id))).scalars().all()
                )
            assert remaining == ["NEW1", "NEW2"]

        await _with_db(check)

    asyncio.run(body())
