import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import Row, delete, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from everyric2.server.db.models import (
    ENGINE_VERSION,
    ActionLog,
    Job,
    JobMetric,
    LinkJob,
    Notice,
    SyncLink,
    SyncResult,
    SyncResultVersion,
    SyncView,
    TranslationLayer,
    VideoOffset,
)


def hash_lyrics(lyrics: str) -> str:
    return hashlib.sha256(lyrics.strip().encode()).hexdigest()[:16]


_layer_logger = logging.getLogger(__name__)


def layer_content_lang_mismatch(target_lang: str, lines: list[dict[str, Any]]) -> bool:
    """레이어 라벨과 번역 본문의 언어가 명백히 어긋나는가 — 오염 저장을 막는 판정.

    실제 사고(2026-07-28, OHcNQHbWrFY): ``lineMetaLang``이 빠진 생성 요청이 서버 기본값
    ko로 폴백해 **영어 번역이 (ko, manual) 레이어로 저장**됐고, 크로스 지문 이관이 사람
    origin이라 신뢰해 새 지문까지 복사했다 — ko 사용자 화면에 영어가 떴다.

    문턱은 보수적이다: 본문이 짧으면(판정 근거 부족) 통과, 외래어·고유명사가 섞인 정상
    번역도 통과 — ko 라벨인데 한글이 사실상 없거나(5% 미만), en·ja 라벨인데 한글이
    과반인 명백한 오염만 막는다. (en 본문이 ja로 라벨되는 류는 이 사고 경로에서 나올 수
    없어 — 폴백 기본값이 ko 하나다 — 범위 밖으로 둔다.)
    """
    text = " ".join((line.get("translation") or "") for line in lines)
    visible = sum(1 for ch in text if not ch.isspace())
    if visible < 20:
        return False
    hangul_ratio = sum(1 for ch in text if "가" <= ch <= "힣") / visible
    if target_lang == "ko":
        return hangul_ratio < 0.05
    if target_lang in ("en", "ja"):
        return hangul_ratio > 0.5
    return False


class SyncRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_id(self, sync_id: str) -> SyncResult | None:
        """싱크 id로 정확히 한 건 — 잡이 실제로 만든 결과(job.result_id)를 짚을 때 쓴다."""
        result = await self.session.execute(select(SyncResult).where(SyncResult.id == sync_id))
        return result.scalar_one_or_none()

    async def get_by_video_and_hash(self, video_id: str, lyrics_hash: str) -> SyncResult | None:
        # force 재생성은 같은 (video_id, lyrics_hash) 행을 여러 개 만들 수 있다 — 최신 우선
        result = await self.session.execute(
            select(SyncResult)
            .where(
                SyncResult.video_id == video_id,
                SyncResult.lyrics_hash == lyrics_hash,
            )
            .order_by(SyncResult.created_at.desc())
        )
        return result.scalars().first()

    async def find_other_video_by_lyrics_hash(
        self, lyrics_hash: str, exclude_video_id: str
    ) -> SyncResult | None:
        """같은 가사 지문(lyrics_hash)의 싱크를 가진 **다른** 영상 — 무다운로드 링크 후보.

        커버가 원곡과 같은 위키/LRCLIB 가사를 쓰면 확장이 보내는 lyrics_hash가 문자열
        수준에서 일치한다 — 제목 유사도보다 훨씬 강한 «같은 곡» 신호다(다운로드 0).
        최신 행 우선. 판정이 아니라 후보다 — 오프셋은 여전히 반주 상관이 계산해야 한다.
        """
        result = await self.session.execute(
            select(SyncResult)
            .where(
                SyncResult.lyrics_hash == lyrics_hash,
                SyncResult.video_id != exclude_video_id,
            )
            .order_by(SyncResult.created_at.desc())
        )
        return result.scalars().first()

    async def get_by_audio_hash(self, audio_hash: str) -> SyncResult | None:
        result = await self.session.execute(
            select(SyncResult)
            .where(SyncResult.audio_hash == audio_hash)
            .order_by(SyncResult.created_at.desc())
        )
        return result.scalar_one_or_none()

    async def get_by_audio_and_lyrics_hash(
        self, audio_hash: str, lyrics_hash: str
    ) -> SyncResult | None:
        # force 재생성으로 동일 해시 행이 복수 존재할 수 있다 — 최신 우선
        result = await self.session.execute(
            select(SyncResult)
            .where(
                SyncResult.audio_hash == audio_hash,
                SyncResult.lyrics_hash == lyrics_hash,
            )
            .order_by(SyncResult.created_at.desc())
        )
        return result.scalars().first()

    async def get_by_video(self, video_id: str) -> list[SyncResult]:
        result = await self.session.execute(
            select(SyncResult)
            .where(SyncResult.video_id == video_id)
            .order_by(SyncResult.created_at.desc())
        )
        return list(result.scalars().all())

    async def get_existing_video_ids(self, video_ids: list[str]) -> set[str]:
        """요청받은 video_id 중 자기 싱크(sync_results 행)가 있는 것만 — POST
        /api/sync/exists 배치 조회 전용. video_id 열만 select한다(timestamps JSON
        블롭은 절대 읽지 않는다 — 존재 유무만 필요한 요청 하나가 곡 전체를 실어 나르면
        안 된다)."""
        if not video_ids:
            return set()
        result = await self.session.execute(
            select(SyncResult.video_id).where(SyncResult.video_id.in_(video_ids)).distinct()
        )
        return set(result.scalars().all())

    async def get_all_unique_videos(self, limit: int = 50) -> list[SyncResult]:
        """Get one sync result per unique video_id, ordered by most recent."""
        from sqlalchemy import func

        # Subquery to get max created_at for each video_id
        subquery = (
            select(SyncResult.video_id, func.max(SyncResult.created_at).label("max_created"))
            .group_by(SyncResult.video_id)
            .subquery()
        )

        # Join to get full SyncResult rows
        result = await self.session.execute(
            select(SyncResult)
            .join(
                subquery,
                (SyncResult.video_id == subquery.c.video_id)
                & (SyncResult.created_at == subquery.c.max_created),
            )
            .order_by(SyncResult.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def list_titled(self, limit: int = 500) -> list[Row[tuple[str, str | None, str | None]]]:
        """title이 채워진 싱크를 영상별 1건(최신)으로 — 링크 후보 전수 스캔용.

        created_at이 초 단위 문자열이라 같은 초에 만들어진 동일 영상 행이 둘 다 걸릴 수
        있어 파이썬에서 한 번 더 dedupe한다.

        호출부(server/api/sync.py의 find_link_candidates)는 video_id/title/artist 세
        속성만 쓴다 — ORM 엔티티 전체(특히 대형 JSON `timestamps` 컬럼)를 끌어와 매 요청
        수백MB를 역직렬화하며 이벤트루프를 통째로 블로킹하던 문제(실측 5~7초)가 있어
        필요한 세 컬럼만 select한다. 다중 컬럼 select라 `.scalars()`를 걸면 0번째 컬럼만
        남으므로 반드시 `.all()`을 그대로 쓴다 — Row는 속성 접근이 되어 호출부는 그대로다."""
        subquery = (
            select(SyncResult.video_id, func.max(SyncResult.created_at).label("max_created"))
            .where(SyncResult.title.is_not(None))
            .group_by(SyncResult.video_id)
            .subquery()
        )
        result = await self.session.execute(
            select(SyncResult.video_id, SyncResult.title, SyncResult.artist)
            .join(
                subquery,
                (SyncResult.video_id == subquery.c.video_id)
                & (SyncResult.created_at == subquery.c.max_created),
            )
            .where(SyncResult.title.is_not(None))
            .order_by(SyncResult.created_at.desc())
            .limit(limit * 2)
        )
        seen: set[str] = set()
        rows: list[Row[tuple[str, str | None, str | None]]] = []
        for row in result.all():
            if row.video_id in seen:
                continue
            seen.add(row.video_id)
            rows.append(row)
            if len(rows) >= limit:
                break
        return rows

    async def set_title_if_missing(
        self, sync_result: SyncResult, title: str | None, artist: str | None = None
    ) -> bool:
        """title이 비어 있을 때만 조용히 채운다 (기존 값은 절대 덮어쓰지 않는다).

        기회적 백필용 — 조회 요청이 제목을 실어 보내면 재생성 없이 기존 코퍼스에 제목이
        쌓인다. 채웠으면 True."""
        if not title or sync_result.title:
            return False
        sync_result.title = title.strip()[:256]
        if artist and not sync_result.artist:
            sync_result.artist = artist.strip()[:128]
        await self.session.flush()
        return True

    async def delete_by_video(self, video_id: str) -> int:
        """이 영상의 모든 싱크 삭제(초기화) — 잘못 붙여넣은 가사 등에서 완전히 새로 시작.
        삭제된 행 수를 반환한다(반환값은 sync_results 삭제 건수만 — 아래 스냅샷 삭제는
        포함하지 않는다. reset_video_syncs의 removed_syncs 응답 계약을 그대로 유지한다).

        같은 트랜잭션에서 sync_result_versions의 스냅샷도 함께 지운다 — 스냅샷은
        "재처리가 덮어쓴 직전 버전"을 보여주는 재료인데, 사용자가 초기화로 sync_results를
        통째로 지운 뒤에도 그 스냅샷이 남으면 GET /api/sync/{video_id}/previous가
        사용자가 지우라고 한 옛 내용을 계속 돌려준다(재생성 후에도 create()는 지워진
        video_id에 "기존 행 없음=최초 생성"으로 보아 새 스냅샷을 안 만드므로, 고아
        스냅샷은 재처리로도 자연 정리되지 않고 영구 잔류한다). 이 삭제는 video_id 하나의
        PK 행 하나만 건드리므로(SyncResultVersion.video_id가 PK) 전체 스캔이 아니다.

        번역 레이어와 사용자 오프셋도 같은 논리로 함께 지운다(엣지 감사 4.1, 실측: 로컬
        DB에 싱크 없는 고아 레이어 1건). 레이어 키가 (video_id, **지문**, lang)이라,
        초기화 뒤 같은 가사로 다시 만들면 지문이 그대로 맞아 **지우라고 한 옛 번역이
        되살아난다** — 오염된 번역이 저장된 경우 초기화로 복구할 수 없는 상태가 된다.
        오프셋도 지워진 싱크의 타이밍에 맞춘 값이라 새 싱크에 적용하면 어긋난다.

        피드백·잡·행위 로그는 **남긴다** — 운영자에게 간 제보와 쿼터 계산의 재료라
        사용자 콘텐츠가 아니고, 지우면 초기화가 일일 한도 우회 수단이 된다."""
        await self.session.execute(
            delete(SyncResultVersion).where(SyncResultVersion.video_id == video_id)
        )
        await self.session.execute(
            delete(TranslationLayer).where(TranslationLayer.video_id == video_id)
        )
        await self.session.execute(delete(VideoOffset).where(VideoOffset.video_id == video_id))
        result = await self.session.execute(
            delete(SyncResult).where(SyncResult.video_id == video_id)
        )
        return result.rowcount or 0

    async def create(
        self,
        video_id: str,
        lyrics_hash: str,
        timestamps: list[dict[str, Any]],
        language: str | None = None,
        engine: str = "ctc",
        engine_variant: str | None = None,
        # 새로 만드는 싱크는 전부 현행 스택 식별자를 새긴다(결함 #5 부수 작업) — 호출부가
        # 일일이 넘기지 않아도 되게 기본값을 ENGINE_VERSION으로 둔다. 옛 스택 흔적을 남기고
        # 싶은 백필/마이그레이션 스크립트만 명시적으로 None을 넘기면 된다.
        engine_version: str | None = ENGINE_VERSION,
        quality_score: float | None = None,
        audio_hash: str | None = None,
        extra: dict[str, Any] | None = None,
        title: str | None = None,
        artist: str | None = None,
    ) -> SyncResult:
        payload = {"segments": timestamps, **(extra or {})}
        # 이 video_id의 기존 최신 행이 있으면(=이 새 행이 조회 우선순위에서 그 행을
        # 가리게 되면) 새로 넣기 **직전**에 스냅샷한다 — 모든 저장 경로(인프로세스
        # worker._process_job_inner, 원격 워커 api/worker.submit_result, 캐시 재사용의
        # 교차 영상 복사 _complete_from_cache_db)가 이 메서드 하나로 수렴하므로 여기가
        # "덮어쓰기" 지점의 유일한 문이다. SyncResultVersion.__doc__ 참고.
        await self._snapshot_previous_version(video_id, payload)
        sync_result = SyncResult(
            video_id=video_id,
            lyrics_hash=lyrics_hash,
            audio_hash=audio_hash,
            # extra: segments 밖의 곡 단위 부가정보 (예: {"debug": {...}})
            timestamps=payload,
            language=language,
            engine=engine,
            engine_variant=engine_variant,
            engine_version=engine_version,
            quality_score=quality_score,
            title=(title.strip()[:256] if title else None),
            artist=(artist.strip()[:128] if artist else None),
        )
        self.session.add(sync_result)
        await self.session.flush()
        return sync_result

    async def _snapshot_previous_version(
        self, video_id: str, new_payload: dict[str, Any]
    ) -> None:
        """새 행이 가리게 될 이 video_id의 기존 최신 행을 sync_result_versions로 옮긴다.

        기존 행이 없으면(최초 생성) 아무것도 하지 않는다 — 비교할 "직전"이 없다.

        기존 행의 timestamps가 새 payload와 **완전히 같으면**(캐시 재사용의 교차 영상
        복사가 다른 video_id의 동일 내용을 그대로 옮겨 담는 경우 등) 스냅샷도 만들지
        않는다 — 실질적으로 아무것도 안 바뀐 "교체"를 버전으로 남기면 고스트 A/B 비교가
        내용이 같은 두 스냅샷을 나란히 보여주는 무의미한 diff가 된다. 이미 이 조회
        하나로 두 값을 다 들고 있으므로 비교 비용은 이 video_id 한 건에 한정된다(전체
        스캔이나 다른 행의 하이드레이션은 없다).
        """
        result = await self.session.execute(
            select(SyncResult)
            .where(SyncResult.video_id == video_id)
            .order_by(SyncResult.created_at.desc())
            .limit(1)
        )
        previous = result.scalars().first()
        if previous is None or previous.timestamps == new_payload:
            return
        await SyncResultVersionRepository(self.session).snapshot(previous)


class SyncResultVersionRepository:
    """직전 버전 스냅샷 CRUD — video_id가 PK라 video_id당 최신 1건만 존재한다.

    SyncLinkRepository.upsert·VideoOffsetRepository.upsert와 같은 관례(조회 후 있으면
    필드 교체, 없으면 새로 삽입 후 flush)를 따른다 — DELETE 후 INSERT가 아니라 같은
    PK 행을 UPDATE하는 편이 "video_id당 최신 1건" 불변식을 SQLite 레벨(PK 유일성)에서
    그대로 보장하면서 왕복도 하나 더 줄인다.
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get(self, video_id: str) -> SyncResultVersion | None:
        result = await self.session.execute(
            select(SyncResultVersion).where(SyncResultVersion.video_id == video_id)
        )
        return result.scalar_one_or_none()

    async def snapshot(self, previous: SyncResult) -> None:
        """덮어써지기 직전의 sync_results 행 하나를 스냅샷한다 (video_id당 최신 1건,
        기존 스냅샷이 있으면 이번 것으로 교체 — replaced_at은 onupdate로 자동 갱신)."""
        existing = await self.get(previous.video_id)
        if existing:
            existing.lyrics_hash = previous.lyrics_hash
            existing.timestamps = previous.timestamps
            existing.language = previous.language
            existing.engine = previous.engine
            existing.engine_variant = previous.engine_variant
            existing.engine_version = previous.engine_version
            existing.quality_score = previous.quality_score
            existing.created_at = previous.created_at
        else:
            self.session.add(
                SyncResultVersion(
                    video_id=previous.video_id,
                    lyrics_hash=previous.lyrics_hash,
                    timestamps=previous.timestamps,
                    language=previous.language,
                    engine=previous.engine,
                    engine_variant=previous.engine_variant,
                    engine_version=previous.engine_version,
                    quality_score=previous.quality_score,
                    created_at=previous.created_at,
                )
            )
        await self.session.flush()


class JobRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_id(self, job_id: str) -> Job | None:
        result = await self.session.execute(select(Job).where(Job.id == job_id))
        return result.scalar_one_or_none()

    async def get_pending(self, limit: int = 10) -> list[Job]:
        result = await self.session.execute(
            select(Job).where(Job.status == "pending").order_by(Job.created_at).limit(limit)
        )
        return list(result.scalars().all())

    async def get_oldest_queued(self) -> Job | None:
        """가장 오래 대기(queued)한 잡 — 원격 워커 claim이 FIFO로 하나씩 가져간다."""
        result = await self.session.execute(
            select(Job).where(Job.status == "queued").order_by(Job.created_at).limit(1)
        )
        return result.scalar_one_or_none()

    async def get_stale_processing(self, cutoff: datetime) -> list[Job]:
        """updated_at이 cutoff보다 오래된 processing 잡 — 고아 잡 TTL 리퍼(orphan_reaper)가
        회수 대상을 고르는 데 쓴다.

        기준은 created_at(시작 시각)이 아니라 updated_at(마지막 진행 갱신)이다 — 정상 진행
        중인 긴 잡은 2~4초 간격으로 진행률을 보고해(worker._tick_progress/_stage_monitor)
        updated_at이 계속 갱신되므로, 오래 걸리는 정상 잡을 실수로 회수하지 않는다."""
        result = await self.session.execute(
            select(Job).where(Job.status == "processing", Job.updated_at < cutoff)
        )
        return list(result.scalars().all())

    async def get_active_by_video(self, video_id: str, lyrics_hash: str) -> Job | None:
        """같은 영상·같은 가사로 이미 진행 중(pending/processing)인 잡 — 중복 생성 차단용.

        같은 잡이 2개 돌면 같은 임시 오디오 파일을 두 프로세스가 잡아 Windows에서
        WinError 32(파일 잠금)로 다운로드가 깨진다 — 생성 요청은 진행 중 잡에 합류시킨다.
        """
        result = await self.session.execute(
            select(Job)
            .where(
                Job.video_id == video_id,
                Job.lyrics_hash == lyrics_hash,
                Job.status.in_(["pending", "queued", "processing"]),
            )
            .order_by(Job.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def count_queued_before(self, created_at, exclude_id: str | None = None) -> int:
        """대기열 순번 계산 — 나보다 먼저 등록된 대기(queued) 잡 수.

        created_at은 server_default=func.now()라 SQLite에 초 단위 문자열로 저장되는데,
        파이썬 datetime 바인딩은 마이크로초까지 붙어 문자열 비교에서 자기 자신이
        "나보다 먼저"로 세어졌다 (첫 대기 잡이 대기열 2번으로 표시). `<=` + 자기 id
        제외로 바로잡는다 — 같은 초의 다른 잡끼리 순번을 공유하는 건 허용."""
        conditions = [Job.status == "queued", Job.created_at <= created_at]
        if exclude_id is not None:
            conditions.append(Job.id != exclude_id)
        result = await self.session.execute(
            select(func.count()).select_from(Job).where(*conditions)
        )
        return int(result.scalar_one())

    async def create(
        self,
        video_id: str,
        lyrics: str,
        language: str | None = None,
        target_lang: str = "ko",
    ) -> Job:
        lyrics_hash = hash_lyrics(lyrics)
        job = Job(
            video_id=video_id,
            lyrics=lyrics,
            lyrics_hash=lyrics_hash,
            language=language,
            # 요청자의 번역 언어 — 조회(lang 파라미터)용 기록. 레이어 언어와 legacy 병기
            # 판정은 line_meta_lang(worker.resolve_layer_lang)이 하고, 이 값은 두 값이
            # 어긋날 때 진단 로그에만 쓰인다 (Job.target_lang 주석 참조)
            target_lang=(target_lang or "ko").strip() or "ko",
        )
        self.session.add(job)
        await self.session.flush()
        return job

    async def update_status(
        self,
        job_id: str,
        status: str,
        progress: int | None = None,
        result_id: str | None = None,
        error: str | None = None,
        stage: str | None = None,
        failure_kind: str | None = None,
    ) -> None:
        """failure_kind는 status="failed" 쓰기 지점만 넘긴다 — "cancelled"/"external"/"system".
        분류가 애매한 실패(과길이 등 정책 거절)는 넘기지 않아 컬럼이 NULL로 남는다(억지
        분류 금지, MoRef 감사 #3). 완료·진행 갱신 등 나머지 호출부는 그대로 생략한다."""
        values: dict[str, Any] = {"status": status}
        if progress is not None:
            values["progress"] = progress
        if result_id is not None:
            values["result_id"] = result_id
        if error is not None:
            values["error"] = error
        if stage is not None:
            values["stage"] = stage
        if failure_kind is not None:
            values["failure_kind"] = failure_kind

        await self.session.execute(update(Job).where(Job.id == job_id).values(**values))

    async def update_status_if(
        self,
        job_id: str,
        status: str,
        expected: tuple[str, ...],
        progress: int | None = None,
        error: str | None = None,
        stage: str | None = None,
    ) -> bool:
        """현재 status가 expected 안에 있을 때만 상태를 쓴다 (조건부 UPDATE). 썼으면 True.

        무조건 쓰기(update_status)는 취소와 경합한다: 취소 확인과 상태 쓰기 사이에 취소 API가
        끼면 방금 failed로 마킹된 잡이 되살아난다. 실제 사례가 sync.py의
        _queue_after_line_meta 말미다 — 되살아난 queued를 워커가 물어 processing이 되고,
        취소된 잡은 워커가 fail을 제출하지 않으므로 processing에 남고, 만료 스윕이 다시
        queued로 되돌려 **무한 진동**한다. WHERE에 현재 상태를 넣어 읽기-쓰기를 한 문장으로
        만들면 그 창이 사라진다.
        """
        values: dict[str, Any] = {"status": status}
        if progress is not None:
            values["progress"] = progress
        if error is not None:
            values["error"] = error
        if stage is not None:
            values["stage"] = stage

        result = await self.session.execute(
            update(Job).where(Job.id == job_id, Job.status.in_(expected)).values(**values)
        )
        return bool(result.rowcount)


class VideoOffsetRepository:
    """영상별 사용자 싱크 오프셋 upsert/조회."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get(self, video_id: str) -> float | None:
        result = await self.session.execute(
            select(VideoOffset).where(VideoOffset.video_id == video_id)
        )
        row = result.scalar_one_or_none()
        return row.offset_sec if row else None

    async def upsert(self, video_id: str, offset_sec: float) -> None:
        result = await self.session.execute(
            select(VideoOffset).where(VideoOffset.video_id == video_id)
        )
        row = result.scalar_one_or_none()
        if row:
            row.offset_sec = offset_sec
        else:
            self.session.add(VideoOffset(video_id=video_id, offset_sec=offset_sec))
        await self.session.flush()


class ActionLogRepository:
    """파괴적 행위 로그 — 영상·행위별 최근 24시간 횟수로 일일 한도를 검사한다."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def log(self, action: str, video_id: str) -> None:
        self.session.add(ActionLog(action=action, video_id=video_id))
        await self.session.flush()

    async def count_recent(self, action: str, video_id: str, hours: int = 24) -> int:
        since = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=hours)
        result = await self.session.execute(
            select(func.count())
            .select_from(ActionLog)
            .where(
                ActionLog.action == action,
                ActionLog.video_id == video_id,
                ActionLog.created_at >= since,
            )
        )
        return int(result.scalar_one())


class SyncLinkRepository:
    """싱크 링크 CRUD (video_id 고유 → PK 기반 upsert)."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get(self, video_id: str) -> SyncLink | None:
        result = await self.session.execute(
            select(SyncLink).where(SyncLink.video_id == video_id)
        )
        return result.scalar_one_or_none()

    async def get_existing_video_ids(self, video_ids: list[str]) -> set[str]:
        """요청받은 video_id 중 링크(다른 영상의 싱크를 빌려 쓰는 행)가 있는 것만 — POST
        /api/sync/exists 배치 조회 전용. `get_sync`가 자기 싱크 없는 영상에도 링크
        폴백을 내주므로(GET /api/sync/{video_id} 참고) 링크만 있는 영상도 존재로 친다."""
        if not video_ids:
            return set()
        result = await self.session.execute(
            select(SyncLink.video_id).where(SyncLink.video_id.in_(video_ids))
        )
        return set(result.scalars().all())

    async def delete_involving(self, video_id: str) -> int:
        """이 영상이 소유자이거나 소스인 링크 전부 삭제 — 싱크 초기화 시 정합성 유지
        (소스 싱크가 사라진 링크를 남겨두면 빌려 쓰던 영상의 조회가 깨진다)."""
        result = await self.session.execute(
            delete(SyncLink).where(
                or_(SyncLink.video_id == video_id, SyncLink.source_video_id == video_id)
            )
        )
        return result.rowcount or 0

    async def upsert(
        self,
        video_id: str,
        source_video_id: str,
        offset_sec: float,
        rate: float = 1.0,
        verified: bool = False,
    ) -> SyncLink:
        """verified=True는 반주 상관 검증(link-jobs)을 통과한 자동 링크만 쓴다 —
        수동 링크 API는 검증 없이 오프셋을 박으므로 항상 False로 남는다."""
        existing = await self.get(video_id)
        if existing:
            existing.source_video_id = source_video_id
            existing.offset_sec = offset_sec
            # 신규 삽입 시 rate가 누락돼 배속 링크가 1.0으로 저장되던 버그를 함께 바로잡는다
            existing.rate = rate
            existing.verified = verified
            await self.session.flush()
            return existing
        link = SyncLink(
            video_id=video_id,
            source_video_id=source_video_id,
            offset_sec=offset_sec,
            rate=rate,
            verified=verified,
        )
        self.session.add(link)
        await self.session.flush()
        return link

    async def delete(self, video_id: str) -> bool:
        existing = await self.get(video_id)
        if not existing:
            return False
        await self.session.delete(existing)
        await self.session.flush()
        return True


class TranslationLayerRepository:
    """언어별 번역 레이어 CRUD — (video_id, fingerprint, target_lang) 유니크 기반 upsert.

    SyncLinkRepository.upsert와 같은 모양(get 후 있으면 필드 교체, 없으면 새로 만들고
    flush)을 따른다. 유니크 충돌 시 lines/attribution/origin을 통째로 새 값으로 바꾼다 —
    부분 병합은 하지 않는다(레이어는 항상 한 번의 번역 호출 결과 전체를 담는다).
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_layer(
        self, video_id: str, fingerprint: str, target_lang: str
    ) -> TranslationLayer | None:
        result = await self.session.execute(
            select(TranslationLayer).where(
                TranslationLayer.video_id == video_id,
                TranslationLayer.fingerprint == fingerprint,
                TranslationLayer.target_lang == target_lang,
            )
        )
        return result.scalar_one_or_none()

    async def upsert_layer(
        self,
        video_id: str,
        fingerprint: str,
        target_lang: str,
        lines: list[dict[str, Any]],
        attribution: dict[str, Any] | None,
        origin: str,
    ) -> TranslationLayer | None:
        # 모든 레이어 쓰기(생성·지연 첨부·POST·LLM persist·이관 재기록)가 이 한 곳을
        # 지난다 — 라벨-내용 언어 불일치는 여기서 일괄 거부한다(None 반환, 로그).
        if layer_content_lang_mismatch(target_lang, lines):
            _layer_logger.warning(
                "translation layer refused: content language mismatches target %s "
                "(video %s, origin %s) — mislabeled line_meta?",
                target_lang,
                video_id,
                origin,
            )
            return None
        existing = await self.get_layer(video_id, fingerprint, target_lang)
        if existing:
            existing.lines = lines
            existing.attribution = attribution
            existing.origin = origin
            await self.session.flush()
            return existing
        layer = TranslationLayer(
            video_id=video_id,
            fingerprint=fingerprint,
            target_lang=target_lang,
            lines=lines,
            attribution=attribution,
            origin=origin,
        )
        self.session.add(layer)
        await self.session.flush()
        return layer

    async def list_layer_langs(self, video_id: str, fingerprint: str) -> list[str]:
        """이 (video_id, fingerprint)에 존재하는 레이어들의 target_lang 목록 — 정렬·중복 제거.

        조회 응답의 available_langs가 이걸 쓴다 — "이 싱크로 지금 서빙 가능한 번역 언어가
        뭐가 있나"를 클라이언트가 알 수 있게 한다."""
        result = await self.session.execute(
            select(TranslationLayer.target_lang)
            .where(
                TranslationLayer.video_id == video_id,
                TranslationLayer.fingerprint == fingerprint,
            )
            .distinct()
        )
        return sorted(result.scalars().all())

    # 재생성·가사 수정으로 지문이 바뀌어도 사람이 확인한 번역은 살아남아야 한다 — 크로스
    # 지문 이관(_apply_translation_lang) 전용. llm은 대상이 아니다: 품질 보증이 없는
    # 기계번역을 다른 줄 분할로 우격다짐 재정렬해 옮기느니 재생성이 낫다.
    HUMAN_ORIGINS: tuple[str, ...] = ("wiki", "caption", "manual", "legacy")

    async def find_human_layer_other_fingerprint(
        self, video_id: str, target_lang: str, exclude_fingerprint: str
    ) -> TranslationLayer | None:
        """같은 (video_id, target_lang)의 **다른** 지문에 있는 사람 origin 레이어 중 가장
        최근 것 — 없으면 None. 재생성·가사 오탈자 수정으로 지문이 바뀐 뒤에도, 예전 지문에
        남아 있는 위키·자막·수동 번역을 찾아내는 데 쓴다(align_translation_lines로 새
        세그에 재정렬하는 것은 호출부의 몫 — 이 메서드는 후보만 찾는다)."""
        result = await self.session.execute(
            select(TranslationLayer)
            .where(
                TranslationLayer.video_id == video_id,
                TranslationLayer.target_lang == target_lang,
                TranslationLayer.fingerprint != exclude_fingerprint,
                TranslationLayer.origin.in_(self.HUMAN_ORIGINS),
            )
            .order_by(TranslationLayer.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def list_human_langs_other_fingerprint(
        self, video_id: str, exclude_fingerprint: str
    ) -> list[str]:
        """이 영상의 **다른** 지문에 사람 origin 레이어가 있는 target_lang 목록 — 정렬·
        중복 제거. 크로스 지문 이관 후보 탐색(어떤 언어를 재정렬해 볼 가치가 있는지)에
        쓴다 — `find_human_layer_other_fingerprint`는 language 하나를 안다는 전제로
        레이어 자체를 찾지만, 이 메서드는 "볼 가치가 있는 language가 뭐가 있나"부터
        답한다."""
        result = await self.session.execute(
            select(TranslationLayer.target_lang)
            .where(
                TranslationLayer.video_id == video_id,
                TranslationLayer.fingerprint != exclude_fingerprint,
                TranslationLayer.origin.in_(self.HUMAN_ORIGINS),
            )
            .distinct()
        )
        return sorted(result.scalars().all())


class LinkJobRepository:
    """링크 검증 잡 CRUD — 중복 쌍 병합·FIFO claim·결과 마감."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_id(self, link_job_id: str) -> LinkJob | None:
        result = await self.session.execute(select(LinkJob).where(LinkJob.id == link_job_id))
        return result.scalar_one_or_none()

    async def get_active_pair(self, video_id: str, source_video_id: str) -> LinkJob | None:
        """같은 (video_id, source_video_id)로 이미 진행 중(queued/processing)인 잡 — 중복 방지."""
        result = await self.session.execute(
            select(LinkJob)
            .where(
                LinkJob.video_id == video_id,
                LinkJob.source_video_id == source_video_id,
                LinkJob.status.in_(["queued", "processing"]),
            )
            .order_by(LinkJob.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def get_recent_attempt(
        self, video_id: str, source_video_id: str, days: int
    ) -> LinkJob | None:
        """최근 N일 안에 끝난(done/failed/declined) 같은 쌍의 잡 — 자동 재제출 쿨다운용.

        get_active_pair는 진행 중(queued/processing) 중복만 막는다. 그래서 완료·실패한
        쌍은 사용자가 그 영상을 열 때마다 다시 제출돼 GPU를 반복해 태울 수 있다 (온디맨드
        자동 제출 경로가 생기며 실제 남용 경로가 됐다). 이력이 있으면 그 잡을 돌려준다.
        days<=0이면 쿨다운 비활성으로 보고 항상 None.

        declined(무다운로드 정책 종결 — MoRef 감사 #4)도 "끝난 잡"이라 여기 포함한다: 빼면
        캐시 미스로 거절된 쌍이 그 영상을 열 때마다 매번 새 링크 잡을 만들어 워커 claim
        왕복만 반복하게 된다(다운로드가 없어 GPU 비용은 없지만 같은 남용 경로다)."""
        if days <= 0:
            return None
        since = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days)
        result = await self.session.execute(
            select(LinkJob)
            .where(
                LinkJob.video_id == video_id,
                LinkJob.source_video_id == source_video_id,
                LinkJob.status.in_(["done", "failed", "declined"]),
                LinkJob.created_at >= since,
            )
            .order_by(LinkJob.created_at.desc())
            .limit(1)
        )
        return result.scalars().first()

    async def get_oldest_queued(self) -> LinkJob | None:
        """가장 오래 대기(queued)한 링크 잡 — 워커 claim이 sync 잡 다음으로 FIFO 소비한다."""
        result = await self.session.execute(
            select(LinkJob).where(LinkJob.status == "queued").order_by(LinkJob.created_at).limit(1)
        )
        return result.scalar_one_or_none()

    async def create(self, video_id: str, source_video_id: str) -> LinkJob:
        link_job = LinkJob(video_id=video_id, source_video_id=source_video_id, status="queued")
        self.session.add(link_job)
        await self.session.flush()
        return link_job

    async def update_status(self, link_job_id: str, status: str) -> None:
        await self.session.execute(
            update(LinkJob).where(LinkJob.id == link_job_id).values(status=status)
        )

    async def mark_done(
        self, link_job_id: str, match: bool, offset_sec: float, confidence: float
    ) -> None:
        await self.session.execute(
            update(LinkJob)
            .where(LinkJob.id == link_job_id)
            .values(
                status="done", match=match, offset_sec=offset_sec, confidence=confidence, error=None
            )
        )

    async def mark_failed(self, link_job_id: str, error: str) -> None:
        await self.session.execute(
            update(LinkJob).where(LinkJob.id == link_job_id).values(status="failed", error=error)
        )

    async def mark_declined(self, link_job_id: str, error: str) -> None:
        """무다운로드 원칙에 따른 정책적 종결(예: cache_miss_no_download) — 오류가 아니므로
        failed와 갈라 기록한다 (MoRef 감사 #4). error 자유텍스트는 그대로 보존한다."""
        await self.session.execute(
            update(LinkJob)
            .where(LinkJob.id == link_job_id)
            .values(status="declined", error=error)
        )


class NoticeRepository:
    """운영 공지 CRUD — 목록은 활성분만, 생성·비활성화는 어드민 전용(api/notices.py가 키를 검사)."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def list_active(self, limit: int = 20) -> list[Notice]:
        """활성(active=True) + 아직 안 끝난(ends_at NULL 또는 미래) 공지, 최신순."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        result = await self.session.execute(
            select(Notice)
            .where(Notice.active.is_(True), or_(Notice.ends_at.is_(None), Notice.ends_at > now))
            .order_by(Notice.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_by_id(self, notice_id: int) -> Notice | None:
        result = await self.session.execute(select(Notice).where(Notice.id == notice_id))
        return result.scalar_one_or_none()

    async def create(
        self, title: str, body: str, level: str, ends_at: datetime | None = None
    ) -> Notice:
        notice = Notice(title=title, body=body, level=level, ends_at=ends_at)
        self.session.add(notice)
        await self.session.flush()
        return notice

    async def deactivate(self, notice_id: int) -> bool:
        notice = await self.get_by_id(notice_id)
        if notice is None:
            return False
        notice.active = False
        await self.session.flush()
        return True


class SyncViewRepository:
    """영상별 조회수 카운터 — video_id가 PK라 upsert(있으면 +1, 없으면 1로 생성)."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def increment(self, video_id: str) -> None:
        result = await self.session.execute(select(SyncView).where(SyncView.video_id == video_id))
        row = result.scalar_one_or_none()
        if row:
            row.views += 1
        else:
            self.session.add(SyncView(video_id=video_id, views=1))
        await self.session.flush()

    async def get_many(self, video_ids: list[str]) -> dict[str, int]:
        """요청받은 video_id 중 실제로 조회수가 있는 것만 dict로 — 없는 건 응답 조립부가 0으로
        채운다(SyncView 행이 아예 없는 영상은 여기 안 잡힌다)."""
        if not video_ids:
            return {}
        result = await self.session.execute(
            select(SyncView.video_id, SyncView.views).where(SyncView.video_id.in_(video_ids))
        )
        return {row.video_id: row.views for row in result.all()}


class JobMetricRepository:
    """완료된 잡의 처리 시간·깊이 이력 — ETA 산출(GET /api/job/{id})의 유일한 재료.

    실패한 잡은 여기 안 온다(호출부가 성공 경로에서만 record를 부른다) — 완주 못 한
    소요 시간을 섞으면 ETA가 낙관적으로 왜곡된다(models.JobMetric 독스트링 참고)."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def record(
        self, job_id: str, video_id: str, depth: str | None, duration_sec: float
    ) -> None:
        self.session.add(
            JobMetric(job_id=job_id, video_id=video_id, depth=depth, duration_sec=duration_sec)
        )
        await self.session.flush()

    async def recent_durations(self, depth: str | None, limit: int = 20) -> list[float]:
        """최근 limit건의 duration_sec — depth를 주면 그 깊이만(라우팅이 결정한 깊이별
        ETA), None이면 전체(깊이 미상·큐 대기열 ETA의 all-depth 폴백)."""
        stmt = select(JobMetric.duration_sec).order_by(JobMetric.created_at.desc()).limit(limit)
        if depth is not None:
            stmt = stmt.where(JobMetric.depth == depth)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
