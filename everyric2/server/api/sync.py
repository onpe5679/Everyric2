from typing import Any

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel

from everyric2.server.db.connection import get_session
from everyric2.server.db.repository import JobRepository, SyncRepository, hash_lyrics

router = APIRouter(prefix="/api/sync", tags=["sync"])


class SyncLookupResponse(BaseModel):
    found: bool
    sync_id: str | None = None
    timestamps: list[dict[str, Any]] | None = None
    lyrics_source: str | None = None
    quality_score: float | None = None
    audio_hash: str | None = None
    language: str | None = None
    created_at: str | None = None
    # 곡 단위 진단 정보 (star 흡수 구간, VAD 발성 구간) — 확장 디버그 스트립용
    debug: dict[str, Any] | None = None
    # 가사 출처 표기 (예: 보카로 가사 위키) — 푸터 병기용
    attribution: dict[str, Any] | None = None
    # 곡 템포 {bpm, beat_offset} — 가라오케 레인 마디 창/비트 격자용
    tempo: dict[str, Any] | None = None


class LineMeta(BaseModel):
    """라인별 부가 정보 — 발음 표기/사람 번역 (보카로 가사 위키 등). 텍스트로 세그먼트에 매칭된다."""

    text: str
    pronunciation: str | None = None
    translation: str | None = None


class Attribution(BaseModel):
    """가사 출처 표기 (예: 보카로 가사 위키 CC BY) — 싱크에 저장돼 조회 시 그대로 반환된다."""

    name: str
    url: str | None = None


class GenerateRequest(BaseModel):
    video_id: str
    lyrics: str
    lyrics_source: str = "user_input"
    language: str | None = None
    line_meta: list[LineMeta] | None = None
    attribution: Attribution | None = None


class GenerateResponse(BaseModel):
    job_id: str
    status: str
    estimated_time: int = 15


class SearchByAudioRequest(BaseModel):
    audio_hash: str


class CopySyncRequest(BaseModel):
    source_video_id: str
    target_video_id: str
    lyrics: str | None = None


class RegenerateRequest(BaseModel):
    video_id: str
    lyrics: str
    language: str | None = None
    force: bool = False
    line_meta: list[LineMeta] | None = None
    attribution: Attribution | None = None


def _merge_meta_into_sync(
    sync_result, line_meta: list[LineMeta] | None, attribution: Attribution | None = None
) -> None:
    """이미 존재하는 싱크에 발음/번역 메타·출처를 병합 (세션 커밋은 호출부의 컨텍스트가 수행)."""
    from everyric2.server.worker import merge_line_meta

    updated = dict(sync_result.timestamps)
    changed = False
    if line_meta:
        segs = [dict(s) for s in updated.get("segments", [])]
        if merge_line_meta(segs, [m.model_dump() for m in line_meta]):
            updated["segments"] = segs
            changed = True
    if attribution is not None:
        updated["attribution"] = attribution.model_dump()
        changed = True
    if changed:
        # JSON 컬럼은 재할당해야 변경이 감지된다
        sync_result.timestamps = updated


@router.get("/{video_id}", response_model=SyncLookupResponse)
async def get_sync(video_id: str, lyrics_hash: str | None = None):
    async with get_session() as session:
        repo = SyncRepository(session)

        if lyrics_hash:
            result = await repo.get_by_video_and_hash(video_id, lyrics_hash)
            if result:
                return SyncLookupResponse(
                    found=True,
                    sync_id=result.id,
                    timestamps=result.timestamps.get("segments", []),
                    lyrics_source=result.engine,
                    quality_score=result.quality_score,
                    audio_hash=result.audio_hash,
                    language=result.language,
                    created_at=result.created_at.isoformat() if result.created_at else None,
                    debug=result.timestamps.get("debug"),
                    attribution=result.timestamps.get("attribution"),
                    tempo=result.timestamps.get("tempo"),
                )
        else:
            results = await repo.get_by_video(video_id)
            if results:
                result = results[0]
                return SyncLookupResponse(
                    found=True,
                    sync_id=result.id,
                    timestamps=result.timestamps.get("segments", []),
                    lyrics_source=result.engine,
                    quality_score=result.quality_score,
                    audio_hash=result.audio_hash,
                    language=result.language,
                    created_at=result.created_at.isoformat() if result.created_at else None,
                    debug=result.timestamps.get("debug"),
                    attribution=result.timestamps.get("attribution"),
                    tempo=result.timestamps.get("tempo"),
                )

        return SyncLookupResponse(found=False)


@router.post("/generate", response_model=GenerateResponse)
async def generate_sync(request: GenerateRequest, background_tasks: BackgroundTasks):
    lyrics_hash_value = hash_lyrics(request.lyrics)

    async with get_session() as session:
        sync_repo = SyncRepository(session)
        existing = await sync_repo.get_by_video_and_hash(request.video_id, lyrics_hash_value)
        if existing:
            # 정렬은 재사용하되 새로 들어온 발음/번역 메타·출처는 반영한다
            if request.line_meta or request.attribution:
                _merge_meta_into_sync(existing, request.line_meta, request.attribution)
            return GenerateResponse(
                job_id=existing.id,
                status="completed",
                estimated_time=0,
            )

        job_repo = JobRepository(session)
        job = await job_repo.create(
            video_id=request.video_id,
            lyrics=request.lyrics,
            language=request.language,
        )
        job_id = job.id

    from everyric2.server.worker import process_job, stash_attribution, stash_line_meta

    if request.line_meta:
        stash_line_meta(job_id, [m.model_dump() for m in request.line_meta])
    if request.attribution:
        stash_attribution(job_id, request.attribution.model_dump())
    background_tasks.add_task(process_job, job_id)

    return GenerateResponse(
        job_id=job_id,
        status="processing",
        estimated_time=15,
    )


@router.post("/search-by-audio", response_model=SyncLookupResponse)
async def search_by_audio_hash(request: SearchByAudioRequest):
    async with get_session() as session:
        repo = SyncRepository(session)
        result = await repo.get_by_audio_hash(request.audio_hash)
        if result:
            return SyncLookupResponse(
                found=True,
                sync_id=result.id,
                timestamps=result.timestamps.get("segments", []),
                lyrics_source=result.engine,
                quality_score=result.quality_score,
                audio_hash=result.audio_hash,
                language=result.language,
                created_at=result.created_at.isoformat() if result.created_at else None,
            )
        return SyncLookupResponse(found=False)


@router.get("/list/{video_id}")
async def list_syncs_for_video(video_id: str):
    async with get_session() as session:
        repo = SyncRepository(session)
        results = await repo.get_by_video(video_id)
        return {
            "video_id": video_id,
            "syncs": [
                {
                    "sync_id": r.id,
                    "lyrics_hash": r.lyrics_hash,
                    "audio_hash": r.audio_hash,
                    "quality_score": r.quality_score,
                    "language": r.language,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in results
            ],
        }


class SearchSyncRequest(BaseModel):
    title: str | None = None
    artist: str | None = None
    limit: int = 10


@router.post("/search")
async def search_available_syncs(request: SearchSyncRequest):
    async with get_session() as session:
        repo = SyncRepository(session)
        results = await repo.get_all_unique_videos(limit=request.limit * 3)
        return {
            "syncs": [
                {
                    "video_id": r.video_id,
                    "audio_hash": r.audio_hash,
                    "quality_score": r.quality_score,
                    "language": r.language,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "lyrics_preview": _get_lyrics_preview(r.timestamps),
                }
                for r in results
            ]
        }


def _get_lyrics_preview(timestamps: dict) -> str:
    segments = timestamps.get("segments", [])
    if not segments:
        return ""
    texts = [s.get("text", "") for s in segments[:3]]
    return " / ".join(texts)[:100]


@router.post("/regenerate", response_model=GenerateResponse)
async def regenerate_sync(request: RegenerateRequest, background_tasks: BackgroundTasks):
    lyrics_hash_value = hash_lyrics(request.lyrics)

    async with get_session() as session:
        if not request.force:
            sync_repo = SyncRepository(session)
            existing = await sync_repo.get_by_video_and_hash(request.video_id, lyrics_hash_value)
            if existing:
                if request.line_meta or request.attribution:
                    _merge_meta_into_sync(existing, request.line_meta, request.attribution)
                return GenerateResponse(
                    job_id=existing.id,
                    status="completed",
                    estimated_time=0,
                )

        job_repo = JobRepository(session)
        job = await job_repo.create(
            video_id=request.video_id,
            lyrics=request.lyrics,
            language=request.language,
        )
        job_id = job.id

    from everyric2.server.worker import process_job, stash_attribution, stash_force, stash_line_meta

    if request.force:
        # 워커의 (audio_hash, lyrics_hash) 재사용 검사까지 건너뛰어야 진짜 재생성이 된다
        stash_force(job_id)
    if request.line_meta:
        stash_line_meta(job_id, [m.model_dump() for m in request.line_meta])
    if request.attribution:
        stash_attribution(job_id, request.attribution.model_dump())
    background_tasks.add_task(process_job, job_id)

    return GenerateResponse(
        job_id=job_id,
        status="processing",
        estimated_time=15,
    )
