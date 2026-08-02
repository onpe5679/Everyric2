from datetime import datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import JSON, Boolean, DateTime, Float, String, Text, UniqueConstraint, func
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(AsyncAttrs, DeclarativeBase):
    pass


class SyncResult(Base):
    __tablename__ = "sync_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    video_id: Mapped[str] = mapped_column(String(32), index=True)
    lyrics_hash: Mapped[str] = mapped_column(String(64), index=True)
    audio_hash: Mapped[str | None] = mapped_column(String(32), index=True)
    timestamps: Mapped[dict[str, Any]] = mapped_column(JSON)
    # 이것은 **원문 언어**다(가사가 무슨 말인가). 번역 대상 언어가 아니다.
    #
    # ⚠ 알려진 한계 — 번역·발음은 `timestamps`의 세그먼트에 박혀 저장되는데(생성 시
    # merge_line_meta), **어느 언어로 번역했는지는 어디에도 기록되지 않는다.** 그래서 같은
    # 영상을 모국어가 다른 두 사용자가 보면 먼저 만든 쪽의 언어가 그대로 내려간다: 한국인이
    # 만든 싱크를 일본인이 열면 한국어 번역과 한글 독음을 받는다(확장의 loadTranslations가
    # "번역이 다 있다"고 판정해 자기 언어로 재요청하지 않는다 — 그쪽 주석에 경로가 있다).
    #
    # 지금은 한국어권 사용자만 대상이라 두었다. 다국어로 넓힐 때 고칠 자리는 번역을
    # (video_id, target_lang) 키로 분리하는 것이고, 그 전 단계 임시책은 timestamps에 번역
    # 언어를 적고 클라이언트가 비교하게 하는 것이다(스키마 변경 없이 가능).
    language: Mapped[str | None] = mapped_column(String(8))
    engine: Mapped[str] = mapped_column(String(16), default="ctc")
    quality_score: Mapped[float | None] = mapped_column(Float)
    # 영상 제목/아티스트 — 커버 링크 후보 탐색이 코퍼스에서 같은 곡을 찾는 유일한 단서다
    # (video_id만으로는 곡을 식별할 수 없다). 기존 행은 NULL로 남고, 조회 시 기회적으로
    # 백필된다(SyncRepository.set_title_if_missing). 매칭은 title_match가 정규화해 수행하므로
    # 유튜브 풀 제목을 그대로 저장해도 된다.
    title: Mapped[str | None] = mapped_column(String(256))
    artist: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    video_id: Mapped[str] = mapped_column(String(32), index=True)
    lyrics: Mapped[str] = mapped_column(Text)
    lyrics_hash: Mapped[str] = mapped_column(String(64))
    language: Mapped[str | None] = mapped_column(String(8))
    # 요청자의 **번역 대상 언어**. 바로 위 language는 원문 언어다(SyncResult.language 주석).
    # 생성 결과 번역은 이 언어의 TranslationLayer에 기록되고, "ko"일 때만 세그먼트의 legacy
    # translation 슬롯에도 병기된다 — 구버전 확장은 그 슬롯만 읽으므로 남의 언어를 넣으면
    # 한국어 사용자가 영어 번역을 받는다. 값을 안 주는 구버전 생성 요청은 "ko"(기존 동작).
    target_lang: Mapped[str] = mapped_column(String(8), default="ko", server_default="ko")
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    result_id: Mapped[str | None] = mapped_column(String(36))
    error: Mapped[str | None] = mapped_column(Text)
    # status="failed"가 뭉뚱그리던 사건 구분 (MoRef 감사 #3) — "cancelled"(사용자 취소) /
    # "external"(다운로드가 이미 분류한 외부 요인: 로그인요구·봉쇄·영상불가 등,
    # audio/downloader.py의 DownloadError 계열) / "system"(그 외 예외 — 진짜 시스템 오류).
    # error 자유텍스트는 그대로 두고 이 컬럼은 집계·모니터링용 구조화 값만 더한다.
    # NULL은 "아직 실패하지 않음"과 "실패했지만 분류가 애매해 억지로 넣지 않음"(예: 영상
    # 과길이 정책 거절 — downloader 분류도 시스템 결함도 아니다) 둘 다를 뜻한다.
    failure_kind: Mapped[str | None] = mapped_column(String(16))
    progress: Mapped[int] = mapped_column(default=0)
    # 현재 진행 단계명 (다운로드/전사 정렬/보컬 분리/…) — 확장 진행 칩 표시용
    stage: Mapped[str | None] = mapped_column(String(24))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class VideoOffset(Base):
    """영상별 사용자 싱크 오프셋(초) — 확장의 ±0.1s 조정을 서버에 영구 저장.

    보는 영상(video_id) 기준이라, 링크로 빌려온 싱크(inst/커버)도 영상마다
    서로 다른 오프셋을 따로 저장할 수 있다. 타임스탬프 자체는 건드리지 않고
    클라이언트가 재생 시점에 적용한다.
    """

    __tablename__ = "video_offsets"

    video_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    offset_sec: Mapped[float] = mapped_column(Float, default=0.0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class ActionLog(Base):
    """파괴적 행위(강제 재생성·초기화) 기록 — 일일 한도 검사용 (공개 배포 대비)."""

    __tablename__ = "action_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    action: Mapped[str] = mapped_column(String(16), index=True)
    video_id: Mapped[str] = mapped_column(String(32), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class SyncLink(Base):
    """다른 영상의 싱크를 오프셋과 함께 재사용하는 링크 (inst/커버 영상용).

    video_id는 링크의 소유자(PK=고유). source_video_id의 실제 싱크를 offset_sec만큼
    시프트해 조회 시 대신 내려준다. 자기 싱크가 있으면 링크보다 우선한다.
    """

    __tablename__ = "sync_links"

    video_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    source_video_id: Mapped[str] = mapped_column(String(32), index=True)
    offset_sec: Mapped[float] = mapped_column(Float, default=0.0)
    # 원곡 대비 재생 배속 (nightcore 1.25 등) — 소스 시간 t를 t/rate + offset으로 사상.
    # 1.0이면 순수 시프트(기존 동작).
    rate: Mapped[float] = mapped_column(Float, default=1.0, server_default="1.0")
    # 반주 상관 검증(link-jobs)을 통과해 만들어진 링크인지. 수동 링크 API(POST /api/sync/link)는
    # 검증 없이 임의 오프셋(0 포함)을 박을 수 있어 코퍼스에 틀린 링크가 남은 전례가 있다 —
    # 조회 응답의 linked.verified로 내려보내 클라이언트가 구분할 수 있게 한다.
    verified: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class TranslationLayer(Base):
    """언어별 번역의 정석 저장소 — (video_id, fingerprint, target_lang) 유니크.

    번역·발음은 지금 SyncResult.timestamps의 세그먼트에 박혀 저장되는데, 그 구조는 "어느
    언어로 번역했는지"를 어디에도 기록하지 않는다(SyncResult.language 주석 참조 — 그건
    원문 언어지 번역 대상 언어가 아니다). 그래서 모국어가 다른 두 사용자가 같은 영상을
    보면 먼저 만든 쪽의 번역 언어가 그대로 내려간다. 이 테이블이 그 언어 슬롯을 분리한다.

    키에 video_id뿐 아니라 fingerprint(가사 원문 지문, lines_fingerprint)를 넣은 이유:
    **가사가 같으면 싱크가 재생성되어도 번역이 살아남는다.** video_id만 키였다면 오탈자
    수정 없는 순수 재생성(force)에서도 SyncResult가 새 행을 만들 때마다 같은 번역을 LLM에
    다시 물어야 했을 것이다. fingerprint를 SyncResult 자체가 아니라 별도로 계산해 두는
    이유는 여러 SyncResult(엔진 버전이 다른 재생성 등)가 같은 가사를 공유할 수 있어서다.

    origin="legacy": 이 레이어 테이블이 생기기 전(배포 이전)에 만들어진 싱크는 ko 번역이
    SyncResult.timestamps의 세그먼트에만 박혀 있고 레이어가 없다. lang=en 등 비ko 조회는
    그 세그 translation을 비우므로(레이어가 없으면 어느 언어인지 알 수 없다), 그 상태에서
    재생성(특히 force)이 한 번이라도 일어나면 원래 세그에 있던 ko 번역(위키 사람 번역
    포함)을 되살릴 방법이 사라진다. sync.py의 GET 조회(lang=ko)와 재생성 직전 두 지점이
    이 사고를 막기 위해 세그의 레거시 번역을 이 origin으로 레이어에 옮겨 백필한다 — "llm"이
    아닌 이유는 실제로 LLM이 만든 값이 아닐 수 있어서다(사람 위키 번역이 세그에 병합된
    경우 포함, attribution이 있으면 그 출처를 그대로 보존한다).
    """

    __tablename__ = "translation_layers"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    video_id: Mapped[str] = mapped_column(String(32), index=True)
    fingerprint: Mapped[str] = mapped_column(String(32), index=True)
    target_lang: Mapped[str] = mapped_column(String(8))
    # [{"text": str, "translation": str}, ...] — 원문 라인과 번역의 병렬 배열
    lines: Mapped[list[Any]] = mapped_column(JSON)
    # {"name","url","license","source_id"} — 위키 등 출처 표기. LLM 번역은 None.
    attribution: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    origin: Mapped[str] = mapped_column(String(16))  # "llm"|"wiki"|"manual"|"caption"|"legacy"
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    __table_args__ = (UniqueConstraint("video_id", "fingerprint", "target_lang"),)


class LinkJob(Base):
    """링크 후보 검증 잡 — "커버(video_id)가 원곡(source_video_id)과 같은 반주를 쓰는가"를
    반주 상관으로 자동 판정한다. 외부 오케스트레이터가 POST /api/link-jobs로 넣고, 원격 GPU
    워커가 claim해 처리한다. match=true면 워커 result 수신부가 SyncLink를 자동 생성한다.

    created_at은 server_default라 SQLite에 초 단위 문자열로 저장된다 — 카운트/순번 비교 시
    마이크로초 바인딩 off-by-one 교훈(JobRepository.count_queued_before) 주의.

    status: "queued" → "processing" → "done"(match 유무 무관, 정상 판정 완료) 또는
    "failed"(진짜 오류) 또는 "declined"(MoRef 감사 #4 — cli.py가 무다운로드 원칙에 따라
    캐시 미스로 판정 자체를 포기한 정책적 종결. cache_miss_no_download류가 여기 해당하며,
    이건 오류가 아니므로 failed와 갈라 집계한다). 세 상태 모두 "끝난 잡"으로 취급되어
    재제출 쿨다운(LinkJobRepository.get_recent_attempt) 대상이다.
    """

    __tablename__ = "link_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    video_id: Mapped[str] = mapped_column(String(32), index=True)
    source_video_id: Mapped[str] = mapped_column(String(32), index=True)
    status: Mapped[str] = mapped_column(String(16), default="queued", index=True)
    match: Mapped[bool | None] = mapped_column(Boolean)
    offset_sec: Mapped[float | None] = mapped_column(Float)
    confidence: Mapped[float | None] = mapped_column(Float)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class SyncResultVersion(Base):
    """sync_results가 재처리(모델 스택 교체 등)로 덮어써지기 **직전**의 1세대 스냅샷.

    확장의 재처리 전/후 "고스트" A/B 비교 + (향후) 롤백의 재료다. video_id를 PK로 써서
    video_id당 최신 직전 버전 1건만 유지한다 — 여러 세대를 쌓지 않는 이유는 sync_results
    자체가 행당 평균 272KB(JSON, SyncRepository.create 참고)라, 재처리마다 무한히 누적되면
    이 서버(SQLite 단일 파일)의 저장·백업 비용이 재처리 횟수에 비례해 무한정 자라기
    때문이다. 롤백 API는 이번 스코프 밖(확장 대개편 때 설계) — 여기서는 조회만 제공한다
    (GET /api/sync/{video_id}/previous, server/api/sync.py).

    **스냅샷 시점**: `sync_results`는 UPDATE가 아니라 매 처리마다 새 행을 INSERT하는
    구조라(같은 video_id에 여러 행이 공존하고, 조회는 created_at 최신 행을 우선한다 —
    SyncRepository.get_by_video 등 참고) DB 레벨의 "덮어쓰기"(UPDATE)는 일어나지 않는다.
    여기서 "덮어쓰기"는 **"새 행이 조회 우선순위에서 기존 행을 가린다"**는 제품 레벨
    사건이고, 그 사건이 일어나는 유일한 지점은 `SyncRepository.create()`다(두 저장 경로 —
    인프로세스 `worker._process_job_inner`, 원격 워커 `api/worker.submit_result` — 가 모두
    이 메서드 하나로 수렴한다). 그래서 스냅샷도 그 안에서 찍는다: 새 행을 만들기 직전에
    같은 video_id의 기존 최신 행을 여기로 옮겨 담는다. 최초 생성(기존 행 없음)은
    스냅샷을 만들지 않는다 — 비교할 "직전"이 없다.
    """

    __tablename__ = "sync_result_versions"

    video_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    # sync_results.timestamps와 같은 통짜 JSON(스냅샷 시점의 segments+extra 전체) — 고스트
    # 비교가 세그먼트 타이밍·번역·발음까지 그대로 다시 그릴 수 있어야 한다
    timestamps: Mapped[dict[str, Any]] = mapped_column(JSON)
    language: Mapped[str | None] = mapped_column(String(8))
    engine: Mapped[str] = mapped_column(String(16), default="ctc")
    quality_score: Mapped[float | None] = mapped_column(Float)
    # 스냅샷된 sync_results 행이 **원래** 만들어진 시각 (그 세대의 생성 시각 — 고스트 라벨용).
    # server_default를 쓰지 않는다: 이 값은 지금이 아니라 옮겨 담는 원본 행의 created_at을
    # 그대로 옮긴 것이다.
    created_at: Mapped[datetime] = mapped_column(DateTime)
    # 이 스냅샷이 찍힌(=새 행이 덮어쓴) 시각. 갱신(두 번째 이후 교체)에서도 onupdate로
    # 자동 갱신된다 — VideoOffset.updated_at과 같은 관례(레포지토리가 직접 손대지 않는다).
    replaced_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )
