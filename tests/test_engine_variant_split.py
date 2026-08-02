"""결함 #5(language/engine_variant 분리) + 결함 #6(quality_score 문서화 부수 확인) 회귀 테스트.

외부 감사(MoRef)가 코드로 확정한 결함 #5: MMS 강제 폴백(force_mms) 정렬 결과가
``sync_results.language``에 "{순수언어}_mms"로 뭉쳐 저장돼(ctc_engine.py의 cache_key를
worker.py가 그대로 detected_lang에 흘렸다), ``WHERE language='ja'`` 같은 순수 언어 필터가
실제 ja 곡을 누락시켰다(실측 19.8%). 이 파일이 못박는 계약:

① 엔진(ctc_engine.CTCEngine)이 순수 언어와 변형을 별도 속성으로 노출한다 — 캐시 키
   (_current_lang)는 내부용으로 그대로 두고 건드리지 않는다(기존 test_adapter_swap.py의
   test_force_mms_keeps_cache_key_distinct가 그 값을 이미 못박고 있다).
② worker._run_alignment가 그 둘을 읽어 result["language"]는 항상 순수 언어, 새 키
   result["engine_variant"]에 변형을 따로 담는다.
③ SyncRepository.create가 engine_variant를 그대로 저장하고, engine_version은 기본값으로
   현행 스택 식별자(ENGINE_VERSION)를 새로 만드는 모든 행에 자동으로 새긴다.
④ connection.init_db의 소급 백필이 기존 "{lang}_mms" 행을 순수 언어 + engine_variant='mms'로
   가르고, 멱등이며(재실행 안전), engine_version은 소급 채우지 않는다(구세대 표시로 NULL 유지).
⑤ GET 응답 모델(SyncLookupResponse)에 engine_variant/engine_version이 additive로 붙는다 —
   기존 필드는 손대지 않고, 없어도(구버전 호출부) 기본값 None으로 안전하다.
"""

import asyncio
import contextlib

import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from everyric2.server.db import connection as db_conn
from everyric2.server.db.models import ENGINE_VERSION, Base, SyncResult
from everyric2.server.db.repository import SyncRepository, hash_lyrics

VID_JA = "JAJAJAJAJA1"
VID_KO = "KOKOKOKOKO1"


@contextlib.asynccontextmanager
async def _db():
    """격리된 in-memory SQLite — 기존 서버 테스트 규약(test_sync_link.py 등)과 동일."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    orig = db_conn.async_session
    db_conn.async_session = sm
    try:
        yield sm
    finally:
        db_conn.async_session = orig
        await engine.dispose()


# ── ① 엔진이 순수 언어와 변형을 따로 노출한다 ──────────────────────────


class _StubTokenizer:
    def __init__(self):
        self.target_lang = None
        self.pad_token_id = 0

    def set_target_lang(self, code):
        self.target_lang = code

    def get_vocab(self):
        return {f"t{i}": i for i in range(32)}


class _StubProcessor:
    def __init__(self):
        self.tokenizer = _StubTokenizer()

    @classmethod
    def from_pretrained(cls, name):
        return cls()


class _StubModel:
    def __init__(self):
        self.adapter_calls = []

    @classmethod
    def from_pretrained(cls, name):
        return cls()

    def to(self, device):
        return self

    def load_adapter(self, code):
        self.adapter_calls.append(code)

    def eval(self):
        return self


@pytest.fixture
def stubbed_engine(monkeypatch):
    """test_adapter_swap.py의 stubbed 픽스처와 같은 기법 — 가중치 로더만 스텁으로 갈아끼운다."""
    import torch
    import transformers

    from everyric2.alignment.ctc_engine import CTCEngine
    from everyric2.config.settings import AlignmentSettings

    monkeypatch.setattr(transformers.Wav2Vec2ForCTC, "from_pretrained", _StubModel.from_pretrained)
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", _StubProcessor.from_pretrained)
    engine = CTCEngine(AlignmentSettings())
    engine._device = torch.device("cpu")  # 로컬 GPU를 건드리지 않는다
    return engine


def test_force_mms_exposes_pure_language_and_mms_variant(stubbed_engine):
    """force_mms=True로 로드하면 캐시 키는 여전히 '{lang}_mms'지만(내부용, 건드리지 않음),
    새 속성은 순수 언어와 변형을 따로 준다 — 이게 worker.py가 DB에 쓸 값이다."""
    stubbed_engine._ensure_model_loaded("ja", force_mms=True)
    assert stubbed_engine._current_lang == "ja_mms"  # 캐시 키는 그대로 (기존 계약)
    assert stubbed_engine._current_language == "ja"
    assert stubbed_engine._current_engine_variant == "mms"


def test_normal_load_has_no_engine_variant(stubbed_engine):
    """force_mms 없이 로드하면 변형이 없다 — None이지 빈 문자열이 아니다."""
    stubbed_engine._ensure_model_loaded("ja", force_mms=False)
    assert stubbed_engine._current_lang == "ja"
    assert stubbed_engine._current_language == "ja"
    assert stubbed_engine._current_engine_variant is None


def test_adapter_swap_path_also_updates_the_new_attributes(stubbed_engine):
    """어댑터 스왑(베이스 유지, 690행 조기 반환 분기)도 새 속성을 갱신해야 한다 —
    전체 재로드 경로(721행)만 갱신하고 스왑 경로를 빠뜨리면 두 번째 언어부터 값이 굳는다."""
    stubbed_engine._ensure_model_loaded("ko", force_mms=True)
    assert stubbed_engine._current_engine_variant == "mms"
    # 같은 베이스(MMS)로 어댑터만 바꿔 변형 없는 언어로 전환
    stubbed_engine._ensure_model_loaded("en", force_mms=False)
    assert stubbed_engine._current_language == "en"
    assert stubbed_engine._current_engine_variant is None
    # 다시 force_mms로 전환 — 스왑 경로에서도 변형이 살아나야 한다
    stubbed_engine._ensure_model_loaded("ja", force_mms=True)
    assert stubbed_engine._current_language == "ja"
    assert stubbed_engine._current_engine_variant == "mms"


# ── ② worker._run_alignment이 engine_variant를 결과 딕셔너리에 싣는다 ──────


class _FakeAlignEngine:
    """CTC 엔진 대역 — _current_language/_current_engine_variant를 실제 엔진처럼 노출한다.

    test_worker_pipeline_defects.py의 _UnalignedEngine과 같은 자리지만, 그쪽은 결함 #5 이전
    속성(_current_lang만)으로 짜여 있어 그 테스트들이 여전히 통과함을 별도로 확인했다
    (hasattr(engine, "_current_language") 가드가 구식 대역과도 하위호환된다)."""

    def __init__(self, language: str, variant: str | None):
        self._current_adapter = "jpn"
        self._current_lang = f"{language}_mms" if variant else language
        self._current_language = language
        self._current_engine_variant = variant
        self._last_star_spans: list = []

    def is_available(self) -> bool:
        return True

    def align(self, audio, lyrics, language=None, progress_callback=None):
        from everyric2.inference.prompt import SyncResult, WordSegment

        out = []
        for k, ln in enumerate(lyrics):
            out.append(
                SyncResult(
                    text=ln.text,
                    start_time=float(k),
                    end_time=float(k) + 1.0,
                    confidence=0.05,
                    line_number=ln.line_number,
                    word_segments=[
                        WordSegment(word=ln.text[:1] or "x", start=float(k), end=float(k) + 0.5, confidence=0.05)
                    ],
                )
            )
        return out


def _run_alignment_with_fake_engine(monkeypatch, tmp_path, language: str, variant: str | None):
    import numpy as np

    from everyric2.alignment import ctc_engine as ctc_mod
    from everyric2.audio import loader as loader_mod
    from everyric2.config.settings import get_settings
    from everyric2.server import worker as worker_mod

    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"fake")
    fake_audio = loader_mod.AudioData(
        waveform=np.zeros(16000, dtype="float32"), sample_rate=16000, duration=1.0
    )

    class _FakeLoader:
        def load(self, path):
            return fake_audio

    engine = _FakeAlignEngine(language, variant)
    monkeypatch.setattr(loader_mod, "AudioLoader", _FakeLoader)
    monkeypatch.setattr(ctc_mod, "get_shared_ctc_engine", lambda _s: engine)
    monkeypatch.setattr(worker_mod, "_separate_vocals", lambda _a: None)
    monkeypatch.setattr(worker_mod, "_estimate_tempo", lambda _a: None)

    settings = get_settings()
    saved_melody = settings.melody.enabled
    object.__setattr__(settings.melody, "enabled", False)
    try:
        lyrics = "\n".join(f"テスト行{i}" for i in range(3))
        return worker_mod._run_alignment(str(audio_file), lyrics, language)
    finally:
        object.__setattr__(settings.melody, "enabled", saved_melody)


def test_run_alignment_reports_pure_language_and_mms_variant(monkeypatch, tmp_path):
    """force_mms 정렬 결과는 language가 순수하고 engine_variant='mms'다 — 결함 #5 이전에는
    language 자체가 'ja_mms'였다."""
    result = _run_alignment_with_fake_engine(monkeypatch, tmp_path, "ja", "mms")
    assert result["language"] == "ja"
    assert result["engine_variant"] == "mms"


def test_run_alignment_reports_no_variant_when_not_mms(monkeypatch, tmp_path):
    result = _run_alignment_with_fake_engine(monkeypatch, tmp_path, "ko", None)
    assert result["language"] == "ko"
    assert result["engine_variant"] is None


# ── ③ SyncRepository.create가 engine_variant/engine_version을 저장한다 ─────


def test_repository_create_stores_variant_and_stamps_current_engine_version():
    async def body():
        async with _db() as sm:
            async with sm() as s:
                await SyncRepository(s).create(
                    video_id=VID_JA,
                    lyrics_hash="h1",
                    timestamps=[{"text": "x", "start": 0.0, "end": 1.0}],
                    language="ja",
                    engine_variant="mms",
                )
                await s.commit()
            async with sm() as s:
                row = await SyncRepository(s).get_by_video_and_hash(VID_JA, "h1")
                assert row.language == "ja"
                assert row.engine_variant == "mms"
                # 호출부가 명시하지 않아도 현행 스택 식별자가 자동으로 새겨진다
                assert row.engine_version == ENGINE_VERSION

    asyncio.run(body())


def test_repository_create_without_variant_leaves_it_none():
    async def body():
        async with _db() as sm:
            async with sm() as s:
                await SyncRepository(s).create(
                    video_id=VID_KO,
                    lyrics_hash="h1",
                    timestamps=[{"text": "x", "start": 0.0, "end": 1.0}],
                    language="ko",
                )
                await s.commit()
            async with sm() as s:
                row = await SyncRepository(s).get_by_video_and_hash(VID_KO, "h1")
                assert row.engine_variant is None
                assert row.engine_version == ENGINE_VERSION

    asyncio.run(body())


def test_repository_create_can_opt_out_of_the_version_stamp():
    """백필/마이그레이션 스크립트가 옛 스택 흔적을 남기고 싶을 때를 위한 탈출구."""

    async def body():
        async with _db() as sm:
            async with sm() as s:
                await SyncRepository(s).create(
                    video_id=VID_JA,
                    lyrics_hash="h2",
                    timestamps=[{"text": "x", "start": 0.0, "end": 1.0}],
                    engine_version=None,
                )
                await s.commit()
            async with sm() as s:
                row = await SyncRepository(s).get_by_video_and_hash(VID_JA, "h2")
                assert row.engine_version is None

    asyncio.run(body())


# ── ④ 소급 백필: "{lang}_mms" 행을 순수 언어 + engine_variant='mms'로 가른다 ──


def test_init_db_backfills_legacy_mms_suffixed_language(monkeypatch):
    """구세대 스키마(engine_variant/engine_version 없음, title/artist는 이미 있는 상태 —
    실제 프로덕션이 지금 이 모양이다)에서 재기동하면 "{lang}_mms" 행이 갈라져야 한다.

    멱등성도 함께 검증한다: init_db를 두 번 불러도 값이 흔들리지 않아야 한다(재실행 안전).
    """
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(db_conn, "engine", engine)
    monkeypatch.setattr(db_conn, "async_session", sm)

    async def body():
        # engine_variant/engine_version이 없는 구세대 sync_results — 이미 title/artist
        # 마이그레이션은 거친 상태를 흉내낸다(create_all이 이미 존재하는 테이블은 건드리지
        # 않으므로 이 수동 스키마가 init_db 이후에도 "구세대 컬럼 목록"으로 남는다).
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "CREATE TABLE sync_results ("
                    "id VARCHAR(36) PRIMARY KEY, video_id VARCHAR(32), lyrics_hash VARCHAR(64), "
                    "audio_hash VARCHAR(32), timestamps JSON, language VARCHAR(8), "
                    "engine VARCHAR(16), quality_score FLOAT, title VARCHAR(256), "
                    "artist VARCHAR(128), created_at DATETIME)"
                )
            )
            for row_id, vid, lang in [
                ("r1", "VID1VID1VID", "ja_mms"),
                ("r2", "VID2VID2VID", "ko_mms"),
                ("r3", "VID3VID3VID", "en"),
                ("r4", "VID4VID4VID", None),
            ]:
                await conn.execute(
                    text(
                        "INSERT INTO sync_results "
                        "(id, video_id, lyrics_hash, timestamps, language, engine) "
                        "VALUES (:id, :vid, :h, '{}', :lang, 'ctc')"
                    ),
                    {"id": row_id, "vid": vid, "h": f"h-{row_id}", "lang": lang},
                )

        await db_conn.init_db()

        async def _rows_by_id(session):
            result = await session.execute(select(SyncResult))
            return {r.id: r for r in result.scalars().all()}

        async with sm() as s:
            rows = await _rows_by_id(s)

        assert rows["r1"].language == "ja"
        assert rows["r1"].engine_variant == "mms"
        assert rows["r2"].language == "ko"
        assert rows["r2"].engine_variant == "mms"
        # 순수 언어였던 행은 손대지 않는다
        assert rows["r3"].language == "en"
        assert rows["r3"].engine_variant is None
        # NULL 언어 행도 크래시 없이 그대로
        assert rows["r4"].language is None
        assert rows["r4"].engine_variant is None
        # engine_version은 소급 백필 대상이 아니다 — 기존 행은 구세대 표시로 NULL 유지
        assert rows["r1"].engine_version is None
        assert rows["r3"].engine_version is None

        # 멱등성: 두 번째 init_db 호출(서버 재기동 재현)이 값을 흔들지 않는다
        await db_conn.init_db()
        async with sm() as s:
            rows_again = await _rows_by_id(s)
        assert rows_again["r1"].language == "ja"
        assert rows_again["r1"].engine_variant == "mms"
        assert rows_again["r2"].language == "ko"
        assert rows_again["r2"].engine_variant == "mms"
        assert rows_again["r3"].language == "en"
        assert rows_again["r3"].engine_variant is None

    try:
        asyncio.run(body())
    finally:
        asyncio.run(engine.dispose())


# ── ⑤ GET 응답 모델(SyncLookupResponse)에 additive 필드가 붙는다 ──────────


def test_sync_lookup_response_defaults_engine_fields_to_none_when_absent():
    """구버전 호출부(engine_variant/engine_version을 안 주는 코드)와도 여전히 만들어져야
    한다 — 이게 '추가 필드라 구버전 확장은 무시하면 그만'의 서버 쪽 절반이다."""
    from everyric2.server.api.sync import SyncLookupResponse

    resp = SyncLookupResponse(found=False)
    assert resp.engine_variant is None
    assert resp.engine_version is None
    # 기존 필드는 이 추가로 전혀 바뀌지 않았다
    assert resp.found is False
    assert resp.language is None


# ── ⑥ 교차 영상 캐시 복사가 engine_variant/engine_version을 원본 그대로 옮긴다ㅡ


def test_cross_video_cache_copy_preserves_engine_variant_and_version_from_the_original():
    """_complete_from_cache_db는 새 정렬이 아니라 기존 행의 **복사**다. create()의
    engine_variant/engine_version 기본값(None/현행 ENGINE_VERSION)에 맡기면 mms로 만들어진
    원본의 변형 정보가 복사본에서 조용히 사라지거나, 실제로는 옛 스택 산물인데 방금 만든
    것처럼 현행 스택으로 잘못 표시된다 — 원본 값이 그대로 옮겨져야 한다."""

    async def body():
        from everyric2.server.db.repository import JobRepository, SyncRepository, hash_lyrics
        from everyric2.server.worker import _try_complete_from_cache

        lyrics = "라인1\n라인2"

        async with _db() as sm:
            async with sm() as s:
                await SyncRepository(s).create(
                    video_id="SRCSRCSRC01",
                    lyrics_hash=hash_lyrics(lyrics),
                    timestamps=[{"text": "라인1", "start": 1.0, "end": 2.0}],
                    audio_hash="hashA",
                    language="ja",
                    engine_variant="mms",
                    engine_version="old-stack-0",  # 옛 스택 산물임을 흉내낸다
                )
                job = await JobRepository(s).create(video_id="DSTDSTDST01", lyrics=lyrics)
                await s.commit()

            import tempfile
            from pathlib import Path

            with tempfile.TemporaryDirectory() as tmp:
                audio = Path(tmp) / "a.wav"
                audio.write_bytes(b"x")
                ok = await _try_complete_from_cache(
                    job.id, job, "hashA", hash_lyrics(lyrics), str(audio)
                )
                assert ok is True

            async with sm() as s:
                copied = await SyncRepository(s).get_by_video_and_hash(
                    "DSTDSTDST01", hash_lyrics(lyrics)
                )
                assert copied.language == "ja"
                assert copied.engine_variant == "mms"
                # 현행 ENGINE_VERSION으로 덮이지 않고 원본의 옛 스택 식별자를 그대로 옮긴다
                assert copied.engine_version == "old-stack-0"
                assert copied.engine_version != ENGINE_VERSION

    asyncio.run(body())


def test_get_sync_end_to_end_surfaces_engine_variant_and_version():
    """생성 → 저장 → 조회 전 구간에서 engine_variant/engine_version이 새지 않는다."""

    async def body():
        async with _db() as sm:
            async with sm() as s:
                lyrics_hash = hash_lyrics("가사")
                await SyncRepository(s).create(
                    video_id=VID_JA,
                    lyrics_hash=lyrics_hash,
                    timestamps=[{"text": "가사", "start": 0.0, "end": 1.0}],
                    language="ja",
                    engine_variant="mms",
                )
                await s.commit()

            from everyric2.server.api.sync import get_sync

            resp = await get_sync(VID_JA, lyrics_hash=lyrics_hash)
            assert resp.found is True
            assert resp.language == "ja"
            assert resp.engine_variant == "mms"
            assert resp.engine_version == ENGINE_VERSION

    asyncio.run(body())
