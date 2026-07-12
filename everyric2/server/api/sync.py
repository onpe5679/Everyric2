import copy
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Query
from pydantic import BaseModel

from everyric2.config.settings import get_settings
from everyric2.server.db.connection import get_session
from everyric2.server.db.repository import (
    ActionLogRepository,
    JobRepository,
    SyncLinkRepository,
    SyncRepository,
    VideoOffsetRepository,
    hash_lyrics,
)


async def _check_destructive_limit(session, action: str, video_id: str, api_key: str | None):
    """파괴적 행위(강제 재생성·초기화) 일일 한도 — admin_api_key가 설정된 배포에서만.

    키가 미설정이면(로컬 사용) 제한 없음. 키 보유 요청은 어드민으로 보고 통과.
    통과 시 로그를 남겨 다음 검사에 반영한다.
    """
    server = get_settings().server
    limit = server.daily_destructive_limit
    if not server.admin_api_key or limit <= 0 or api_key == server.admin_api_key:
        return
    log_repo = ActionLogRepository(session)
    if await log_repo.count_recent(action, video_id) >= limit:
        raise HTTPException(
            status_code=429,
            detail=f"이 영상의 {action} 일일 한도({limit}회/24시간)에 도달했어요. 내일 다시 시도해 주세요.",
        )
    await log_repo.log(action, video_id)

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
    # 곡 키 {tonic, mode, name, confidence} — 멜로디 분석의 K-S 추정, 레인 표시용
    key: dict[str, Any] | None = None
    # 다른 영상의 싱크를 오프셋과 함께 빌려 왔을 때만 채워진다 (자기 싱크가 있으면 None).
    # 클라이언트가 링크 상태 표시·해제 버튼을 띄우는 데 쓴다.
    linked: dict[str, Any] | None = None
    # 이 영상에 저장된 사용자 싱크 오프셋(초) — 클라이언트가 재생 시점에 적용.
    # 링크로 빌려온 싱크도 보는 영상 기준이라 영상마다 따로 저장된다.
    user_offset: float | None = None


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


# ── 싱크 링크 (inst/커버 영상이 다른 영상의 전사를 오프셋과 함께 재사용) ───────────


class SyncLinkRequest(BaseModel):
    video_id: str
    source_video_id: str
    offset_sec: float = 0.0


class SyncLinkResponse(BaseModel):
    video_id: str
    source_video_id: str
    offset_sec: float
    created_at: str | None = None


def _shift_time(value: Any, offset: float) -> Any:
    """숫자면 offset 시프트(과한 부동소수 잡음 방지로 반올림), 아니면 그대로."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return round(value + offset, 4)
    return value


def _shift_sync_timestamps(timestamps: dict[str, Any], offset: float) -> dict[str, Any]:
    """소스 싱크의 모든 시간 필드를 offset만큼 시프트한 깊은 복사본을 만든다.

    세그먼트 start/end·words·notes·pron_segments, extra.debug의 vad_regions/star_spans/
    f0_curve.t0, tempo.beat_offset, 세그먼트 debug.orig까지 함께 옮긴다. attribution 등
    시간이 아닌 필드는 그대로 둔다. offset은 음수(소스가 링크 영상보다 늦게 시작)도 된다."""
    data = copy.deepcopy(timestamps)

    for seg in data.get("segments", []) or []:
        if seg.get("start") is not None:
            seg["start"] = _shift_time(seg["start"], offset)
        if seg.get("end") is not None:
            seg["end"] = _shift_time(seg["end"], offset)
        for w in seg.get("words") or []:
            if w.get("start") is not None:
                w["start"] = _shift_time(w["start"], offset)
            if w.get("end") is not None:
                w["end"] = _shift_time(w["end"], offset)
        for n in seg.get("notes") or []:
            if n.get("start") is not None:
                n["start"] = _shift_time(n["start"], offset)
            if n.get("end") is not None:
                n["end"] = _shift_time(n["end"], offset)
        for p in seg.get("pron_segments") or []:
            if p.get("start") is not None:
                p["start"] = _shift_time(p["start"], offset)
            if p.get("end") is not None:
                p["end"] = _shift_time(p["end"], offset)
        dbg = seg.get("debug")
        if isinstance(dbg, dict) and isinstance(dbg.get("orig"), list) and len(dbg["orig"]) == 2:
            dbg["orig"] = [_shift_time(dbg["orig"][0], offset), _shift_time(dbg["orig"][1], offset)]

    debug = data.get("debug")
    if isinstance(debug, dict):
        for key in ("vad_regions", "star_spans"):
            arr = debug.get(key)
            if isinstance(arr, list):
                debug[key] = [
                    [_shift_time(span[0], offset), _shift_time(span[1], offset), *span[2:]]
                    for span in arr
                    if isinstance(span, (list, tuple)) and len(span) >= 2
                ]
        f0 = debug.get("f0_curve")
        if isinstance(f0, dict) and f0.get("t0") is not None:
            f0["t0"] = _shift_time(f0["t0"], offset)

    tempo = data.get("tempo")
    if isinstance(tempo, dict) and tempo.get("beat_offset") is not None:
        tempo["beat_offset"] = _shift_time(tempo["beat_offset"], offset)

    return data


def _build_sync_response(
    result, timestamps: dict[str, Any], linked: dict[str, Any] | None = None
) -> "SyncLookupResponse":
    """SyncResult + (원본 또는 시프트된) timestamps dict → 조회 응답. linked면 빌린 싱크."""
    return SyncLookupResponse(
        found=True,
        sync_id=result.id,
        timestamps=timestamps.get("segments", []),
        lyrics_source=result.engine,
        quality_score=result.quality_score,
        audio_hash=result.audio_hash,
        language=result.language,
        created_at=result.created_at.isoformat() if result.created_at else None,
        debug=timestamps.get("debug"),
        attribution=timestamps.get("attribution"),
        tempo=timestamps.get("tempo"),
        key=timestamps.get("key"),
        linked=linked,
    )


@router.post("/link", response_model=SyncLinkResponse)
async def create_sync_link(request: SyncLinkRequest):
    """영상 video_id가 source_video_id의 싱크를 offset과 함께 빌려 쓰도록 링크(upsert).

    자기 자신 링크는 거부. source에 실제 싱크가 있어야 한다 — source가 그 자체로 링크만
    있고 자기 싱크가 없으면(링크의 링크) 거부한다(단순화: 1단계 링크만 허용)."""
    if request.video_id == request.source_video_id:
        raise HTTPException(status_code=400, detail="Cannot link a video to itself")

    async with get_session() as session:
        sync_repo = SyncRepository(session)
        source_syncs = await sync_repo.get_by_video(request.source_video_id)
        if not source_syncs:
            raise HTTPException(
                status_code=400,
                detail=f"Source video {request.source_video_id} has no sync to link",
            )
        link_repo = SyncLinkRepository(session)
        link = await link_repo.upsert(
            request.video_id, request.source_video_id, request.offset_sec
        )
        return SyncLinkResponse(
            video_id=link.video_id,
            source_video_id=link.source_video_id,
            offset_sec=link.offset_sec,
            created_at=link.created_at.isoformat() if link.created_at else None,
        )


@router.delete("/link/{video_id}")
async def delete_sync_link(video_id: str):
    async with get_session() as session:
        removed = await SyncLinkRepository(session).delete(video_id)
        return {"video_id": video_id, "removed": removed}


@router.get("/list")
async def list_available_syncs(limit: int = Query(50, ge=1, le=200)):
    """조회 가능한 싱크 목록 (확장의 링크 후보 선택용) — 영상별 1개, 최신순."""
    async with get_session() as session:
        results = await SyncRepository(session).get_all_unique_videos(limit=limit)
        items = []
        for r in results:
            ts = r.timestamps or {}
            segments = ts.get("segments", []) or []
            debug = ts.get("debug") or {}
            attribution = ts.get("attribution") or {}
            items.append(
                {
                    "video_id": r.video_id,
                    "first_line": segments[0].get("text", "") if segments else "",
                    "line_count": len(segments),
                    "attribution_name": attribution.get("name"),
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "alignment_text": debug.get("alignment_text"),
                }
            )
        # 확장 클라이언트(listSyncs)가 SyncListItem[] bare 배열을 기대 → 래핑하지 않는다
        return items


class UserOffsetRequest(BaseModel):
    offset_sec: float


@router.put("/offset/{video_id}")
async def save_user_offset(video_id: str, request: UserOffsetRequest):
    """이 영상에서 사용자가 조정한 싱크 오프셋(초)을 저장 — 다음 조회부터 함께 내려간다."""
    offset = max(-60.0, min(60.0, request.offset_sec))
    async with get_session() as session:
        await VideoOffsetRepository(session).upsert(video_id, offset)
    return {"video_id": video_id, "offset_sec": offset}


@router.get("/{video_id}", response_model=SyncLookupResponse)
async def get_sync(video_id: str, lyrics_hash: str | None = None):
    async with get_session() as session:
        repo = SyncRepository(session)
        user_offset = await VideoOffsetRepository(session).get(video_id)

        # 자기 싱크가 있으면 링크보다 우선한다
        if lyrics_hash:
            result = await repo.get_by_video_and_hash(video_id, lyrics_hash)
            if result:
                resp = _build_sync_response(result, result.timestamps)
                resp.user_offset = user_offset
                return resp
        else:
            results = await repo.get_by_video(video_id)
            if results:
                resp = _build_sync_response(results[0], results[0].timestamps)
                resp.user_offset = user_offset
                return resp

        # 자기 싱크가 없고 링크가 있으면 source 싱크를 offset 적용해 빌려 온다
        link = await SyncLinkRepository(session).get(video_id)
        if link:
            source_syncs = await repo.get_by_video(link.source_video_id)
            if source_syncs:
                src = source_syncs[0]
                shifted = _shift_sync_timestamps(src.timestamps, link.offset_sec)
                resp = _build_sync_response(
                    src,
                    shifted,
                    linked={
                        "source_video_id": link.source_video_id,
                        "offset_sec": link.offset_sec,
                    },
                )
                resp.user_offset = user_offset
                return resp

        return SyncLookupResponse(found=False, user_offset=user_offset)


@router.delete("/{video_id}")
async def reset_video_syncs(video_id: str, x_api_key: str | None = Header(default=None)):
    """이 영상의 서버 싱크를 전부 삭제(초기화) — 잘못 붙여넣은 가사 등에서 새로 시작.

    이 영상이 소유자이거나 소스인 링크도 함께 제거한다 ("/link/{video_id}"가 먼저
    선언돼 있어 링크 삭제 경로와 충돌하지 않는다). 공개 배포에선 일일 한도 적용."""
    async with get_session() as session:
        await _check_destructive_limit(session, "reset", video_id, x_api_key)
        removed_syncs = await SyncRepository(session).delete_by_video(video_id)
        removed_links = await SyncLinkRepository(session).delete_involving(video_id)
        return {
            "video_id": video_id,
            "removed_syncs": removed_syncs,
            "removed_links": removed_links,
        }


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
        # 같은 영상·가사로 이미 돌고 있는 잡이 있으면 새 잡을 만들지 않고 합류한다 —
        # 버튼 연타로 동일 잡이 중복 생성되면 같은 임시 오디오 파일을 두 작업이 잡아
        # Windows에서 WinError 32(파일 사용 중)로 다운로드가 깨진다
        active = await job_repo.get_active_by_video(request.video_id, lyrics_hash_value)
        if active:
            if request.line_meta or request.attribution:
                from everyric2.server.worker import stash_attribution, stash_line_meta

                if request.line_meta:
                    stash_line_meta(active.id, [m.model_dump() for m in request.line_meta])
                if request.attribution:
                    stash_attribution(active.id, request.attribution.model_dump())
            return GenerateResponse(job_id=active.id, status="processing", estimated_time=15)
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
async def regenerate_sync(
    request: RegenerateRequest,
    background_tasks: BackgroundTasks,
    x_api_key: str | None = Header(default=None),
):
    lyrics_hash_value = hash_lyrics(request.lyrics)

    async with get_session() as session:
        if request.force:
            # 강제 재생성은 GPU 수십 초를 태우는 파괴적 행위 — 공개 배포에선 일일 한도 적용
            await _check_destructive_limit(session, "regenerate", request.video_id, x_api_key)
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
        # 재생성도 같은 잡 진행 중이면 합류 — 연타가 동시 다운로드(WinError 32)를 만들지 않게
        active = await job_repo.get_active_by_video(request.video_id, lyrics_hash_value)
        if active:
            return GenerateResponse(job_id=active.id, status="processing", estimated_time=15)
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
