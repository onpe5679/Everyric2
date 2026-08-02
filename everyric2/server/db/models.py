from datetime import datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import JSON, Boolean, DateTime, Float, String, Text, UniqueConstraint, func
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(AsyncAttrs, DeclarativeBase):
    pass


# 현행 정렬 스택 식별자 — 새로 만드는 SyncResult에 새긴다(SyncRepository.create 기본값,
# ENGINE_VERSION을 그대로 engine_version 컬럼에 stamp). 기존 행은 engine_version=NULL로
# 남아 "이 스탬프가 생기기 전(구세대) 스택으로 만들어졌다"는 뜻이 된다 — 소급 백필하지
# 않는다(#5 백필은 language/engine_variant 분리에만 해당, engine_version은 별개).
#
# 모델 교체 이니셔티브(bench/model-replacement-owsm, docs/research/2026-07-30-*)가 스택을
# 갈아끼우면 이 문자열을 그 스택 식별자로 바꿔라 — 배포 시점의 git 커밋이 실제 버전 경계다.
ENGINE_VERSION = "mms-htdemucs-1"


class SyncResult(Base):
    __tablename__ = "sync_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    video_id: Mapped[str] = mapped_column(String(32), index=True)
    lyrics_hash: Mapped[str] = mapped_column(String(64), index=True)
    audio_hash: Mapped[str | None] = mapped_column(String(32), index=True)
    timestamps: Mapped[dict[str, Any]] = mapped_column(JSON)
    # 이것은 **원문 언어**다(가사가 무슨 말인가). 번역 대상 언어가 아니다. **순수 언어
    # 코드만 들어간다** ("ja", "ko" 등) — 엔진 변형·폴백 여부는 아래 engine_variant가 진다.
    #
    # 결함 #5(2026-08 수정): 예전에는 MMS 강제 폴백(force_mms) 정렬 결과가 이 컬럼에
    # "ja_mms"·"ko_mms"처럼 언어와 엔진 변형이 뭉쳐 저장됐다(ctc_engine._ensure_model_loaded의
    # cache_key를 worker._run_alignment가 그대로 detected_lang에 흘렸다). 그래서
    # `WHERE language='ja'` 같은 순수 언어 필터가 실제 ja 곡의 19.8%를 누락했다. 지금은
    # 엔진이 순수 언어(_current_language)와 변형(_current_engine_variant)을 따로 노출해
    # 이 컬럼은 항상 순수 언어만 받는다 — MMS 폴백이었으면 engine_variant='mms'로 대신 남는다.
    # 기존 "{lang}_mms" 행은 마이그레이션(connection.py init_db)이 소급 분리했다
    # (engine_variant NULL 컬럼이 처음 생기는 시점에 1회, 멱등).
    #
    # ✓ 해결됨(2026-07-28) — 번역 대상 언어 분리: 예전엔 번역·발음이 `timestamps`의
    # 세그먼트에 박혀 저장돼(생성 시 merge_line_meta) **어느 언어로 번역했는지 기록이
    # 없었다.** 지금은 TranslationLayer 테이블(아래, (video_id, fingerprint, target_lang)
    # 유니크)이 그 언어 슬롯을 따로 갖는다(커밋 6a0e614·76292af·1f1ab0b). 세그의 legacy
    # translation 슬롯은 ko 하위호환용으로만 남아 있다 — TranslationLayer 독스트링 참고.
    language: Mapped[str | None] = mapped_column(String(8))
    engine: Mapped[str] = mapped_column(String(16), default="ctc")
    # 이 정렬에 실제로 쓰인 엔진의 변형/폴백 식별자. 지금 유일하게 쓰이는 값은 "mms"
    # (force_mms 강제 폴백 — 예전엔 language에 "{lang}_mms"로 뭉쳐 저장되던 값, 결함 #5).
    # None이면 language의 기본 어댑터를 그대로 썼다는 뜻(변형 없음). 기존(마이그레이션
    # 이전) 행도 NULL이다 — "변형 없음"과 "몰라서 못 남김"을 구분하지 않는다(백필 대상은
    # language가 "_mms" 접미였던 행뿐이었고, 그 행들은 이미 'mms'로 채워졌다).
    engine_variant: Mapped[str | None] = mapped_column(String(16))
    # 이 싱크를 만든 정렬 스택의 식별자 — 생성 시점의 ENGINE_VERSION(위 모듈 상수)을 그대로
    # 새긴다. NULL이면 이 컬럼이 생기기 전(구세대) 스택으로 만들어졌다는 뜻 — 소급 백필하지
    # 않는다(모델 교체 전후 스택을 곡 단위로 구분하는 것이 목적이라, 과거 값을 되짚어
    # 채우면 오히려 거짓 정보가 된다).
    engine_version: Mapped[str | None] = mapped_column(String(32))
    # CTC 디코딩 자기확신도(정렬된 줄만의 평균 로그확률류 conf, worker._quality_with_coverage가
    # 계산) — **사람이 매긴 정렬 품질 평가가 아니다.** 결함 #6(감사 확정): 이 값은 어댑터
    # vocab 크기에 스케일 의존적이라 **곡 간 비교·정렬에 쓰면 결함**이다(실측: 같은 곡이
    # eng 어댑터로는 0.1289, kor 어댑터로는 0.0492 — 잔차 정보량은 같은데 스케일만 다르다).
    # 곡 간 비교가 필요하면 이 값이 아니라 debug.quality_norm(스케일 무관 e^(-α),
    # worker._scale_free_quality)을 써라. 유일한 소비처인 크롬 확장은 절대 임계
    # `<0.001`만 검사한다(background.ts) — 그 검사는 "정렬이 사실상 실패했다"는 이진
    # 신호로만 이 값을 쓰므로 스케일 의존성과 설계 의도가 맞아떨어진다. config/settings.py의
    # caption_anchors 필드 설명 근처에도 같은 경고가 있다(그 실험이 이 스케일 의존성 때문에
    # 실패했다).
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
