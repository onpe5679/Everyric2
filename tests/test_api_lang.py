"""GET /api/sync?lang= 서빙 + POST /api/translate persist= 저장 (Task 9).

실 DB를 건드리지 않도록 격리된 in-memory SQLite로 connection.async_session을 몽키패치하고
라우트 핸들러(코루틴)를 직접 호출한다 — test_sync_link.py·test_caption_line_meta.py와 같은
서버 테스트 규약이다. translate_lyrics는 동기(plain def) + BackgroundTasks 브리지라
`BackgroundTasks()`를 만들어 넘기고 `await bg()`로 큐에 쌓인 저장 작업을 직접 실행한다
(Starlette가 응답 뒤에 하는 일과 동일 — test_caption_line_meta.py의 `_run_background`
패턴). LLM은 절대 호출하지 않는다 — `LyricsTranslator`를 통째로 목으로 갈아끼운다.

4개 핵심 시나리오(계획 Task 9 Step 1) + ko 레거시 이행 가정 + SyncLink 경유 조회에도
같은 규칙이 적용되는지를 덧붙인다.
"""

import asyncio
import contextlib

import pytest
from fastapi import BackgroundTasks, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from everyric2.config.settings import get_settings
from everyric2.server.api import translate as translate_api
from everyric2.server.api.sync import (
    Attribution,
    RegenerateRequest,
    SaveTranslationLayerRequest,
    SaveTranslationLayerResponse,
    SyncLinkRequest,
    TranslationLayerLine,
    create_sync_link,
    get_sync,
    regenerate_sync,
    save_translation_layer,
)
from everyric2.server.api.translate import TranslateRequest, translate_lyrics
from everyric2.server.db import connection as db_conn
from everyric2.server.db.models import Base, SyncResult
from everyric2.server.db.repository import SyncRepository, TranslationLayerRepository
from everyric2.server.text_fingerprint import lines_fingerprint

VIDEO = "LANGVID0001"
LINES = ["첫 줄", "둘째 줄"]
FP = lines_fingerprint(LINES)


@contextlib.asynccontextmanager
async def _env(**server_overrides):
    """in-memory SQLite로 몽키패치 + 선택적 서버 설정 오버라이드(local_worker 등).

    test_caption_line_meta.py의 _env(**server_overrides)와 같은 패턴 — regenerate_sync는
    local_worker=True(기본값)면 백그라운드로 실 워커 파이프라인(process_job)까지 큐잉하려
    들어 유닛 테스트에서 await하면 안 된다. local_worker=False로 두면 상태만 queued로
    마킹하고 끝나 백필 태스크만 안전하게 실행할 수 있다."""
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
    server = get_settings().server
    saved = {k: getattr(server, k) for k in server_overrides}
    for k, v in server_overrides.items():
        object.__setattr__(server, k, v)
    try:
        yield sm
    finally:
        db_conn.async_session = orig
        for k, v in saved.items():
            object.__setattr__(server, k, v)
        await engine.dispose()


def _seed_segments(translations: list[str]) -> list[dict]:
    return [
        {"text": text, "translation": tr, "start": float(i), "end": float(i) + 1.0}
        for i, (text, tr) in enumerate(zip(LINES, translations))
    ]


async def _run_background(bg: BackgroundTasks) -> None:
    """Starlette가 응답 후에 하는 일 — 등록된 작업을 순서대로 실행한다."""
    await bg()


# ── translate_lyrics를 위한 LLM 없는 대역 (persist 테스트 전용) ──────────────


class _FakeLine:
    def __init__(self, original: str, translation: str):
        self.original = original
        self.translation = translation
        self.pronunciation = None
        self.failed = False


class _FakeResult:
    def __init__(self, lines, source_lang, target_lang, translation_skipped=False):
        self.lines = lines
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.engine = "fake"
        self.translation_skipped = translation_skipped


class _FakeInnerTranslator:
    def translate(self, text, source_lang, target_lang, context=None):
        originals = text.split("\n")
        if source_lang == target_lang:
            lines = [_FakeLine(o, "") for o in originals]
            return _FakeResult(lines, source_lang, target_lang, translation_skipped=True)
        lines = [_FakeLine(o, f"[{target_lang}] {o}") for o in originals]
        return _FakeResult(lines, source_lang, target_lang)


class _FakeLyricsTranslator:
    def __init__(self, settings=None, log_label=None):
        self._translator = _FakeInnerTranslator()

    def translate_with_pronunciation(self, text, source_lang, target_lang, context=None):
        return self._translator.translate(text, source_lang, target_lang, context)


def _patch_translator(monkeypatch):
    monkeypatch.setattr(translate_api, "LyricsTranslator", _FakeLyricsTranslator)


def _blankable_translator(blank_indices: set[int]):
    """지정한 인덱스의 라인만(또는 전부) 빈 번역으로 돌려주는 LyricsTranslator 대역 —
    부분/전체 번역 실패 재현용(#5 감사 항목)."""

    class _Inner:
        def translate(self, text, source_lang, target_lang, context=None):
            originals = text.split("\n")
            lines = [
                _FakeLine(o, "" if i in blank_indices else f"[{target_lang}] {o}")
                for i, o in enumerate(originals)
            ]
            return _FakeResult(lines, source_lang, target_lang)

    class _Translator:
        def __init__(self, settings=None, log_label=None):
            self._translator = _Inner()

        def translate_with_pronunciation(self, text, source_lang, target_lang, context=None):
            return self._translator.translate(text, source_lang, target_lang, context)

    return _Translator


# ── (1) lang 없이 조회 = 기존 응답 그대로 ─────────────────────────────


def test_lookup_without_lang_is_unchanged():
    async def body():
        async with _env() as sm:
            async with sm() as s:
                await SyncRepository(s).create(
                    video_id=VIDEO,
                    lyrics_hash="h1",
                    timestamps=_seed_segments(["레거시 번역 1", ""]),
                )
                await s.commit()

            resp = await get_sync(VIDEO)
            assert resp.found is True
            assert resp.translation_lang is None
            assert [seg["translation"] for seg in resp.timestamps] == ["레거시 번역 1", ""]
            assert [seg["text"] for seg in resp.timestamps] == LINES

    asyncio.run(body())


# ── (2) 레이어 upsert 후 lang=en 조회 → translation 교체 + translation_lang="en" ──


def test_lookup_with_lang_replaces_translation_from_layer():
    async def body():
        async with _env() as sm:
            async with sm() as s:
                await SyncRepository(s).create(
                    video_id=VIDEO,
                    lyrics_hash="h1",
                    timestamps=_seed_segments(["레거시 번역 1", "레거시 번역 2"]),
                )
                await TranslationLayerRepository(s).upsert_layer(
                    VIDEO,
                    FP,
                    "en",
                    lines=[
                        {"text": "첫 줄", "translation": "First line"},
                        {"text": "둘째 줄", "translation": "Second line"},
                    ],
                    attribution=None,
                    origin="llm",
                )
                await s.commit()

            resp = await get_sync(VIDEO, lang="en")
            assert resp.translation_lang == "en"
            assert [seg["translation"] for seg in resp.timestamps] == [
                "First line",
                "Second line",
            ]

            # 레이어 조회가 원본 SyncResult.timestamps를 오염시키지 않았는지 확인 —
            # lang 없이 다시 조회하면 레거시 번역이 그대로 남아 있어야 한다.
            legacy = await get_sync(VIDEO)
            assert [seg["translation"] for seg in legacy.timestamps] == [
                "레거시 번역 1",
                "레거시 번역 2",
            ]
            assert legacy.translation_lang is None

    asyncio.run(body())


# ── (3) lang=fr 레이어 없음 → translation 전부 빈 값 + translation_lang None ──


def test_lookup_with_lang_and_no_layer_blanks_out_non_ko():
    async def body():
        async with _env() as sm:
            async with sm() as s:
                await SyncRepository(s).create(
                    video_id=VIDEO,
                    lyrics_hash="h1",
                    timestamps=_seed_segments(["레거시 번역 1", "레거시 번역 2"]),
                )
                await s.commit()

            resp = await get_sync(VIDEO, lang="fr")
            assert resp.translation_lang is None
            assert all(seg["translation"] == "" for seg in resp.timestamps)

    asyncio.run(body())


# ── ko 레거시 이행 가정: 레이어 없는 lang="ko"는 저장분을 ko로 간주하고 유지한다 ──


def test_lookup_with_lang_ko_and_no_layer_keeps_legacy_translation():
    async def body():
        async with _env() as sm:
            async with sm() as s:
                await SyncRepository(s).create(
                    video_id=VIDEO,
                    lyrics_hash="h1",
                    timestamps=_seed_segments(["레거시 번역 1", ""]),
                )
                await s.commit()

            resp = await get_sync(VIDEO, lang="ko")
            assert resp.translation_lang == "ko"
            assert [seg["translation"] for seg in resp.timestamps] == ["레거시 번역 1", ""]

    asyncio.run(body())


def test_lookup_with_lang_ko_and_no_translation_anywhere_yields_none():
    async def body():
        async with _env() as sm:
            async with sm() as s:
                await SyncRepository(s).create(
                    video_id=VIDEO,
                    lyrics_hash="h1",
                    timestamps=_seed_segments(["", ""]),
                )
                await s.commit()

            resp = await get_sync(VIDEO, lang="ko")
            # 번역이 하나도 없으면 "ko"라고 우길 근거가 없다
            assert resp.translation_lang is None

    asyncio.run(body())


# ── F5(2026-08-04 감사, additive): translation_attribution/translation_origin ──


def test_translation_attribution_and_origin_reflect_the_exact_match_layer():
    async def body():
        async with _env() as sm:
            async with sm() as s:
                await SyncRepository(s).create(
                    video_id=VIDEO, lyrics_hash="h1", timestamps=_seed_segments(["", ""])
                )
                await TranslationLayerRepository(s).upsert_layer(
                    VIDEO,
                    FP,
                    "en",
                    lines=[
                        {"text": "첫 줄", "translation": "First line"},
                        {"text": "둘째 줄", "translation": "Second line"},
                    ],
                    attribution={"name": "위키", "url": None, "license": None, "source_id": "wiki"},
                    origin="wiki",
                )
                await s.commit()

            resp = await get_sync(VIDEO, lang="en")
            assert resp.translation_lang == "en"
            assert resp.translation_origin == "wiki"
            assert resp.translation_attribution == {
                "name": "위키", "url": None, "license": None, "source_id": "wiki",
            }

    asyncio.run(body())


def test_translation_origin_is_legacy_when_serving_ko_without_a_layer():
    async def body():
        async with _env() as sm:
            async with sm() as s:
                await SyncRepository(s).create(
                    video_id=VIDEO,
                    lyrics_hash="h1",
                    timestamps=_seed_segments(["레거시 번역 1", ""]),
                )
                await s.commit()

            resp = await get_sync(VIDEO, lang="ko")
            assert resp.translation_lang == "ko"
            assert resp.translation_origin == "legacy"
            # 레이어가 없는 legacy 경로는 곡 단위 가사 출처(resp.attribution)를 그대로 낸다.
            assert resp.translation_attribution == resp.attribution

    asyncio.run(body())


def test_translation_attribution_and_origin_are_none_without_lang():
    async def body():
        async with _env() as sm:
            async with sm() as s:
                await SyncRepository(s).create(
                    video_id=VIDEO,
                    lyrics_hash="h1",
                    timestamps=_seed_segments(["레거시 번역 1", ""]),
                )
                await s.commit()

            resp = await get_sync(VIDEO)
            assert resp.translation_origin is None
            assert resp.translation_attribution is None

    asyncio.run(body())


def test_translation_attribution_and_origin_are_none_when_lang_has_no_translation():
    async def body():
        async with _env() as sm:
            async with sm() as s:
                await SyncRepository(s).create(
                    video_id=VIDEO,
                    lyrics_hash="h1",
                    timestamps=_seed_segments(["레거시 번역 1", "레거시 번역 2"]),
                )
                await s.commit()

            resp = await get_sync(VIDEO, lang="fr")
            assert resp.translation_lang is None
            assert resp.translation_origin is None
            assert resp.translation_attribution is None

    asyncio.run(body())


# ── (4) POST /api/translate persist=true → 레이어 생김 ─────────────────


def test_translate_persist_creates_translation_layer(monkeypatch):
    _patch_translator(monkeypatch)

    async def body():
        async with _env() as sm:
            bg = BackgroundTasks()
            resp = translate_lyrics(
                TranslateRequest(
                    text="\n".join(LINES),
                    source_lang="ko",
                    target_lang="ja",
                    video_id=VIDEO,
                    persist=True,
                ),
                bg,
            )
            assert resp.lines[0].translation == "[ja] 첫 줄"
            await _run_background(bg)

            async with sm() as s:
                layer = await TranslationLayerRepository(s).get_layer(VIDEO, FP, "ja")
                assert layer is not None
                assert layer.origin == "llm"
                assert layer.lines == [
                    {"text": "첫 줄", "translation": "[ja] 첫 줄"},
                    {"text": "둘째 줄", "translation": "[ja] 둘째 줄"},
                ]

    asyncio.run(body())


def test_translate_persist_false_does_not_create_a_layer(monkeypatch):
    _patch_translator(monkeypatch)

    async def body():
        async with _env() as sm:
            bg = BackgroundTasks()
            translate_lyrics(
                TranslateRequest(
                    text="\n".join(LINES),
                    source_lang="ko",
                    target_lang="ja",
                    video_id=VIDEO,
                    persist=False,
                ),
                bg,
            )
            await _run_background(bg)

            async with sm() as s:
                assert await TranslationLayerRepository(s).get_layer(VIDEO, FP, "ja") is None

    asyncio.run(body())


def test_translate_persist_true_without_video_id_does_not_crash(monkeypatch):
    _patch_translator(monkeypatch)

    async def body():
        async with _env():
            bg = BackgroundTasks()
            resp = translate_lyrics(
                TranslateRequest(
                    text="\n".join(LINES), source_lang="ko", target_lang="ja", persist=True
                ),
                bg,
            )
            assert resp.lines  # 정상 응답 — video_id 없으면 저장 시도만 생략
            await _run_background(bg)  # 큐가 비어 있어야 한다(예외 없이 통과)

    asyncio.run(body())


def test_translate_persist_skips_when_translation_was_skipped(monkeypatch):
    # source == target(대각선)이면 번역 필드가 전부 빈 값이라 저장할 것이 없다 —
    # 레이어에 빈 값을 얹으면 기존 실제 번역을 덮어쓸 위험만 있으므로 저장하지 않는다.
    _patch_translator(monkeypatch)

    async def body():
        async with _env() as sm:
            bg = BackgroundTasks()
            resp = translate_lyrics(
                TranslateRequest(
                    text="\n".join(LINES),
                    source_lang="ko",
                    target_lang="ko",
                    video_id=VIDEO,
                    persist=True,
                ),
                bg,
            )
            assert resp.translation_skipped is True
            await _run_background(bg)

            async with sm() as s:
                assert await TranslationLayerRepository(s).get_layer(VIDEO, FP, "ko") is None

    asyncio.run(body())


# ── 엔드투엔드: persist로 저장한 것을 lang 조회가 실제로 되찾는다 ─────────


def test_persist_then_lookup_round_trip(monkeypatch):
    _patch_translator(monkeypatch)

    async def body():
        async with _env() as sm:
            async with sm() as s:
                await SyncRepository(s).create(
                    video_id=VIDEO, lyrics_hash="h1", timestamps=_seed_segments(["", ""])
                )
                await s.commit()

            bg = BackgroundTasks()
            translate_lyrics(
                TranslateRequest(
                    text="\n".join(LINES),
                    source_lang="ko",
                    target_lang="ja",
                    video_id=VIDEO,
                    persist=True,
                ),
                bg,
            )
            await _run_background(bg)

            resp = await get_sync(VIDEO, lang="ja")
            assert resp.translation_lang == "ja"
            assert [seg["translation"] for seg in resp.timestamps] == [
                "[ja] 첫 줄",
                "[ja] 둘째 줄",
            ]

    asyncio.run(body())


# ── 링크 경유 조회(SyncLink)도 같은 규칙이 적용되는지 ────────────────────


def test_linked_sync_lookup_applies_lang_keyed_by_requested_video_id():
    """링크로 빌려온 싱크도 lang 규칙이 적용된다 — 레이어 키는 source가 아니라
    **요청받은 video_id**(링크를 단 영상)를 쓴다. POST /api/translate persist=true도
    항상 요청 video_id로 저장하므로, 조회도 같은 키라야 서로 맞아떨어진다."""

    async def body():
        async with _env() as sm:
            async with sm() as s:
                await SyncRepository(s).create(
                    video_id="SRCSRCSRC01",
                    lyrics_hash="h1",
                    timestamps=_seed_segments(["소스 레거시", "소스 레거시2"]),
                )
                await s.commit()

            await create_sync_link(
                SyncLinkRequest(
                    video_id="DSTDSTDST01", source_video_id="SRCSRCSRC01", offset_sec=5.0
                )
            )

            async with sm() as s:
                # 레이어는 링크를 단 영상(DST) 키로 저장 — SRC 키로는 안 만든다
                await TranslationLayerRepository(s).upsert_layer(
                    "DSTDSTDST01",
                    FP,
                    "en",
                    lines=[
                        {"text": "첫 줄", "translation": "Linked first"},
                        {"text": "둘째 줄", "translation": "Linked second"},
                    ],
                    attribution=None,
                    origin="llm",
                )
                await s.commit()

            resp = await get_sync("DSTDSTDST01", lang="en")
            assert resp.linked is not None  # 여전히 링크 응답이다(시간이 시프트됨)
            assert resp.translation_lang == "en"
            assert [seg["translation"] for seg in resp.timestamps] == [
                "Linked first",
                "Linked second",
            ]
            # 시간은 여전히 시프트된 값(오프셋 5.0) — lang 처리가 시간 필드를 안 건드렸다
            assert resp.timestamps[0]["start"] == 5.0

            # SRC 자체를 lang=en으로 조회하면 그 레이어가 없으므로(DST 키로만 있다) 비워진다
            src_resp = await get_sync("SRCSRCSRC01", lang="en")
            assert src_resp.translation_lang is None
            assert all(seg["translation"] == "" for seg in src_resp.timestamps)

    asyncio.run(body())


# ── ko 자동 백필: 실사고 시나리오 — 배포 이전 생성분의 ko 번역(위키 사람 번역 포함)이
# en 유저의 재생성으로 서빙에서 사라지는 것을 막는다 ────────────────────────


def test_ko_lookup_backfills_layer_and_survives_regeneration_without_legacy_segments():
    """레이어 없는 옛 싱크를 lang=ko로 읽으면 (a) 레거시 그대로 서빙되고 (b) 백그라운드로
    레이어에 옮겨 백필된다. 그 뒤 "en 유저가 재생성해 새 싱크의 세그엔 ko 번역이 아예
    없는" 상황을 흉내 내도, 미리 백필된 레이어 덕분에 lang=ko 조회가 여전히 그 번역을
    돌려준다 — 백필이 없었다면 이 지점에서 translation_lang이 None이 됐을 것이다."""

    async def body():
        async with _env() as sm:
            async with sm() as s:
                await SyncRepository(s).create(
                    video_id=VIDEO,
                    lyrics_hash="h1",
                    timestamps=_seed_segments(["레거시 번역 1", "레거시 번역 2"]),
                )
                await s.commit()

            # (1) lang=ko 조회 — 레거시 그대로 서빙 + 백그라운드 백필 스케줄
            bg = BackgroundTasks()
            resp = await get_sync(VIDEO, lang="ko", background_tasks=bg)
            assert resp.translation_lang == "ko"
            assert [seg["translation"] for seg in resp.timestamps] == [
                "레거시 번역 1",
                "레거시 번역 2",
            ]
            await _run_background(bg)

            async with sm() as s:
                layer = await TranslationLayerRepository(s).get_layer(VIDEO, FP, "ko")
                assert layer is not None
                assert layer.origin == "legacy"
                assert layer.lines == [
                    {"text": "첫 줄", "translation": "레거시 번역 1"},
                    {"text": "둘째 줄", "translation": "레거시 번역 2"},
                ]

            # (2) en 유저의 재생성을 흉내 — 같은 가사지만 새 싱크의 세그엔 ko 번역이 없다.
            # **delete_by_video로 흉내 내면 안 된다**: sync_results는 INSERT 전용에
            # 최신 우선이라 재생성은 행을 지우지 않고 새로 쌓을 뿐이고, delete_by_video는
            # 사용자의 "초기화"(번역 레이어·오프셋까지 함께 버린다)라 의미가 정반대다.
            # 예전엔 이 자리에서 delete를 불러, 초기화가 레이어를 남기던 결함(엣지 감사
            # 4.1)이 고쳐지자 이 테스트가 깨졌다 — 깨진 쪽은 흉내 방식이었다.
            async with sm() as s:
                repo = SyncRepository(s)
                await repo.create(
                    video_id=VIDEO, lyrics_hash="h1", timestamps=_seed_segments(["", ""])
                )
                await s.commit()

            # (3) 백필된 레이어가 살아 있어 lang=ko가 여전히 복원된다
            recovered = await get_sync(VIDEO, lang="ko")
            assert recovered.translation_lang == "ko"
            assert [seg["translation"] for seg in recovered.timestamps] == [
                "레거시 번역 1",
                "레거시 번역 2",
            ]

    asyncio.run(body())


def test_ko_lookup_without_legacy_translation_schedules_nothing():
    # 세그에 번역이 하나도 없으면 백필할 것도 없다 — 빈 레이어를 만들지 않는다
    async def body():
        async with _env() as sm:
            async with sm() as s:
                await SyncRepository(s).create(
                    video_id=VIDEO, lyrics_hash="h1", timestamps=_seed_segments(["", ""])
                )
                await s.commit()

            bg = BackgroundTasks()
            resp = await get_sync(VIDEO, lang="ko", background_tasks=bg)
            assert resp.translation_lang is None
            await _run_background(bg)

            async with sm() as s:
                assert await TranslationLayerRepository(s).get_layer(VIDEO, FP, "ko") is None

    asyncio.run(body())


# ── 재생성 직전 ko 백필 ───────────────────────────────────────────────


def test_regenerate_backfills_ko_layer_before_creating_job():
    """재생성 잡을 만들기 전에, 이 영상의 최신 싱크에 남아 있는 레거시 ko 번역을 레이어로
    옮긴다 — 재생성(특히 force)이 그 세그 자체를 갈아끼우기 전에 선수를 친다.

    local_worker=False로 둔다 — 기본값(True)이면 _dispatch_job이 실 워커 파이프라인
    (process_job)까지 background_tasks에 얹어, 백필만 확인하려는 이 테스트가 await bg()
    할 때 실제 다운로드·정렬을 돌리려 든다."""

    async def body():
        async with _env(local_worker=False) as sm:
            async with sm() as s:
                await SyncRepository(s).create(
                    video_id=VIDEO,
                    lyrics_hash="oldhash",
                    timestamps=_seed_segments(["레거시 번역 1", "레거시 번역 2"]),
                )
                await s.commit()

            bg = BackgroundTasks()
            await regenerate_sync(
                RegenerateRequest(video_id=VIDEO, lyrics="완전히 다른 새 가사", force=True),
                bg,
                x_api_key=None,
            )
            await _run_background(bg)

            async with sm() as s:
                layer = await TranslationLayerRepository(s).get_layer(VIDEO, FP, "ko")
                assert layer is not None
                assert layer.origin == "legacy"
                assert layer.lines == [
                    {"text": "첫 줄", "translation": "레거시 번역 1"},
                    {"text": "둘째 줄", "translation": "레거시 번역 2"},
                ]

    asyncio.run(body())


def test_regenerate_does_not_backfill_when_ko_layer_already_exists():
    # 이미 레이어가 있으면(예: 이전 요청이 이미 백필함) 다시 만들 필요가 없다 — origin이
    # "llm"이던 것을 "legacy"로 잘못 덮어쓰지 않는지 확인한다.
    async def body():
        async with _env(local_worker=False) as sm:
            async with sm() as s:
                await SyncRepository(s).create(
                    video_id=VIDEO,
                    lyrics_hash="oldhash",
                    timestamps=_seed_segments(["레거시 번역 1", "레거시 번역 2"]),
                )
                await TranslationLayerRepository(s).upsert_layer(
                    VIDEO,
                    FP,
                    "ko",
                    lines=[{"text": "첫 줄", "translation": "이미 있던 정식 번역"}],
                    attribution=None,
                    origin="llm",
                )
                await s.commit()

            bg = BackgroundTasks()
            await regenerate_sync(
                RegenerateRequest(video_id=VIDEO, lyrics="완전히 다른 새 가사", force=True),
                bg,
                x_api_key=None,
            )
            await _run_background(bg)

            async with sm() as s:
                layer = await TranslationLayerRepository(s).get_layer(VIDEO, FP, "ko")
                assert layer.origin == "llm"  # 그대로 — legacy로 덮어써지지 않았다

    asyncio.run(body())


# ── available_langs ────────────────────────────────────────────────


def test_available_langs_includes_stored_layers_and_legacy_ko():
    async def body():
        async with _env() as sm:
            async with sm() as s:
                await SyncRepository(s).create(
                    video_id=VIDEO,
                    lyrics_hash="h1",
                    timestamps=_seed_segments(["레거시 번역 1", "레거시 번역 2"]),
                )
                await TranslationLayerRepository(s).upsert_layer(
                    VIDEO,
                    FP,
                    "en",
                    lines=[{"text": "첫 줄", "translation": "First"}],
                    attribution=None,
                    origin="llm",
                )
                await TranslationLayerRepository(s).upsert_layer(
                    VIDEO,
                    FP,
                    "ja",
                    lines=[{"text": "첫 줄", "translation": "最初"}],
                    attribution=None,
                    origin="llm",
                )
                await s.commit()

            # lang 미지정 조회도 available_langs를 채운다 — 정렬 + 레거시 ko 포함 + 중복 제거
            resp = await get_sync(VIDEO)
            assert resp.available_langs == ["en", "ja", "ko"]

            # lang 지정 조회도 마찬가지로 채운다
            resp_en = await get_sync(VIDEO, lang="en")
            assert resp_en.available_langs == ["en", "ja", "ko"]

    asyncio.run(body())


def test_available_langs_excludes_ko_without_legacy_translation():
    async def body():
        async with _env() as sm:
            async with sm() as s:
                await SyncRepository(s).create(
                    video_id=VIDEO, lyrics_hash="h1", timestamps=_seed_segments(["", ""])
                )
                await TranslationLayerRepository(s).upsert_layer(
                    VIDEO,
                    FP,
                    "en",
                    lines=[{"text": "첫 줄", "translation": "First"}],
                    attribution=None,
                    origin="llm",
                )
                await s.commit()

            resp = await get_sync(VIDEO)
            assert resp.available_langs == ["en"]

    asyncio.run(body())


def test_available_langs_is_none_when_sync_not_found():
    async def body():
        async with _env():
            resp = await get_sync("NOSYNCNOS01")
            assert resp.found is False
            assert resp.available_langs is None

    asyncio.run(body())


def test_available_langs_excludes_ko_when_legacy_translation_has_no_hangul():
    """F4(2026-08-04 감사) — 레거시 translation 필드에 한글이 전혀 없는 텍스트가 남아
    있어도(다른 언어가 실수로 legacy 슬롯에 들어간 경우 등) available_langs가 거짓으로
    "ko"를 광고하면 안 된다. 존재 여부가 아니라 실제 한글 문자로 판정한다."""

    async def body():
        async with _env() as sm:
            async with sm() as s:
                await SyncRepository(s).create(
                    video_id=VIDEO,
                    lyrics_hash="h1",
                    timestamps=_seed_segments(["Not Korean at all", "second line, still not"]),
                )
                await s.commit()

            resp = await get_sync(VIDEO)
            assert resp.available_langs == []

    asyncio.run(body())


# ── POST /{video_id}/translations — 완성된 싱크에 이미 확보한 번역 직접 저장 ──────


def _layer_lines(pairs: list[tuple[str, str]]) -> list[TranslationLayerLine]:
    return [TranslationLayerLine(text=t, translation=tr) for t, tr in pairs]


def test_save_translation_layer_then_lang_lookup_reflects_it():
    """정상 저장 → lang 조회에 그대로 반영된다."""

    async def body():
        async with _env() as sm:
            async with sm() as s:
                await SyncRepository(s).create(
                    video_id=VIDEO, lyrics_hash="h1", timestamps=_seed_segments(["", ""])
                )
                await s.commit()

            resp = await save_translation_layer(
                VIDEO,
                SaveTranslationLayerRequest(
                    target_lang="en",
                    lines=_layer_lines([("첫 줄", "First line"), ("둘째 줄", "Second line")]),
                    origin="caption",
                ),
            )
            assert resp == SaveTranslationLayerResponse(
                saved=True, matched=2, total=2, target_lang="en"
            )

            async with sm() as s:
                layer = await TranslationLayerRepository(s).get_layer(VIDEO, FP, "en")
                assert layer is not None
                assert layer.origin == "caption"
                assert layer.lines == [
                    {"text": "첫 줄", "translation": "First line"},
                    {"text": "둘째 줄", "translation": "Second line"},
                ]

            lookup = await get_sync(VIDEO, lang="en")
            assert lookup.translation_lang == "en"
            assert [seg["translation"] for seg in lookup.timestamps] == [
                "First line",
                "Second line",
            ]

    asyncio.run(body())


def test_save_translation_layer_stores_attribution():
    async def body():
        async with _env() as sm:
            async with sm() as s:
                await SyncRepository(s).create(
                    video_id=VIDEO, lyrics_hash="h1", timestamps=_seed_segments(["", ""])
                )
                await s.commit()

            await save_translation_layer(
                VIDEO,
                SaveTranslationLayerRequest(
                    target_lang="en",
                    lines=_layer_lines([("첫 줄", "First line"), ("둘째 줄", "Second line")]),
                    origin="wiki",
                    attribution=Attribution(
                        name="테스트 위키", url="https://example.test", license="CC BY-SA 4.0"
                    ),
                ),
            )

            async with sm() as s:
                layer = await TranslationLayerRepository(s).get_layer(VIDEO, FP, "en")
                assert layer.attribution == {
                    "name": "테스트 위키",
                    "url": "https://example.test",
                    "license": "CC BY-SA 4.0",
                    "source_id": None,
                }

    asyncio.run(body())


def test_save_translation_layer_rejects_low_match_rate():
    """재정렬 커버리지(세그 중 번역이 매겨진 비율)가 절반 미만이면 422 — 엉뚱한 가사 방지.

    커버리지는 **세그 개수** 기준이다(align_translation_lines 도입 이후 — 예전엔 제출된
    lines 개수 기준이었다). 4개 세그 중 1개만 매칭되도록 구성해 25%로 명백히 미달시킨다."""

    async def body():
        async with _env() as sm:
            async with sm() as s:
                await SyncRepository(s).create(
                    video_id=VIDEO,
                    lyrics_hash="h1",
                    timestamps=[
                        {"text": "첫 줄", "translation": "", "start": 0.0, "end": 1.0},
                        {"text": "둘째 줄", "translation": "", "start": 1.0, "end": 2.0},
                        {"text": "셋째 줄", "translation": "", "start": 2.0, "end": 3.0},
                        {"text": "넷째 줄", "translation": "", "start": 3.0, "end": 4.0},
                    ],
                )
                await s.commit()

            with pytest.raises(HTTPException) as exc:
                await save_translation_layer(
                    VIDEO,
                    SaveTranslationLayerRequest(
                        target_lang="en",
                        # 첫 줄만 이 영상의 가사와 일치 — 나머지 3개 세그는 대응 없음(1/4=25%)
                        lines=_layer_lines(
                            [("첫 줄", "First line"), ("전혀 다른 곡의 가사", "Wrong song")]
                        ),
                        origin="caption",
                    ),
                )
            assert exc.value.status_code == 422

            fp4 = lines_fingerprint(["첫 줄", "둘째 줄", "셋째 줄", "넷째 줄"])
            async with sm() as s:
                assert await TranslationLayerRepository(s).get_layer(VIDEO, fp4, "en") is None

    asyncio.run(body())


def test_save_translation_layer_accepts_exactly_half_match_rate():
    # "절반 이상"은 정확히 50%도 포함한다(경계값)
    async def body():
        async with _env() as sm:
            async with sm() as s:
                await SyncRepository(s).create(
                    video_id=VIDEO, lyrics_hash="h1", timestamps=_seed_segments(["", ""])
                )
                await s.commit()

            resp = await save_translation_layer(
                VIDEO,
                SaveTranslationLayerRequest(
                    target_lang="en",
                    lines=_layer_lines([("첫 줄", "First line"), ("전혀 다른 가사", "Wrong")]),
                    origin="caption",
                ),
            )
            assert resp.saved is True
            assert (resp.matched, resp.total) == (1, 2)

    asyncio.run(body())


def test_save_translation_layer_rejects_unknown_origin():
    async def body():
        async with _env() as sm:
            async with sm() as s:
                await SyncRepository(s).create(
                    video_id=VIDEO, lyrics_hash="h1", timestamps=_seed_segments(["", ""])
                )
                await s.commit()

            with pytest.raises(HTTPException) as exc:
                await save_translation_layer(
                    VIDEO,
                    SaveTranslationLayerRequest(
                        target_lang="en",
                        lines=_layer_lines([("첫 줄", "First line"), ("둘째 줄", "Second")]),
                        origin="llm",  # 화이트리스트 밖 — 이 엔드포인트는 llm을 못 만든다
                    ),
                )
            assert exc.value.status_code == 422

    asyncio.run(body())


def test_save_translation_layer_overwrites_llm_layer_with_caption():
    """기존 레이어가 origin="llm"이면 caption이 교체할 수 있다."""

    async def body():
        async with _env() as sm:
            async with sm() as s:
                await SyncRepository(s).create(
                    video_id=VIDEO, lyrics_hash="h1", timestamps=_seed_segments(["", ""])
                )
                await TranslationLayerRepository(s).upsert_layer(
                    VIDEO,
                    FP,
                    "en",
                    lines=[{"text": "첫 줄", "translation": "machine translated"}],
                    attribution=None,
                    origin="llm",
                )
                await s.commit()

            resp = await save_translation_layer(
                VIDEO,
                SaveTranslationLayerRequest(
                    target_lang="en",
                    lines=_layer_lines([("첫 줄", "human caption"), ("둘째 줄", "second line")]),
                    origin="caption",
                ),
            )
            assert resp.saved is True

            async with sm() as s:
                layer = await TranslationLayerRepository(s).get_layer(VIDEO, FP, "en")
                assert layer.origin == "caption"
                assert layer.lines[0]["translation"] == "human caption"

    asyncio.run(body())


def test_save_translation_layer_re_updates_same_origin():
    """기존이 caption이고 새 요청도 caption이면(재수집·정정 등) 교체된다."""

    async def body():
        async with _env() as sm:
            async with sm() as s:
                await SyncRepository(s).create(
                    video_id=VIDEO, lyrics_hash="h1", timestamps=_seed_segments(["", ""])
                )
                await TranslationLayerRepository(s).upsert_layer(
                    VIDEO,
                    FP,
                    "en",
                    lines=[{"text": "첫 줄", "translation": "old caption text"}],
                    attribution=None,
                    origin="caption",
                )
                await s.commit()

            resp = await save_translation_layer(
                VIDEO,
                SaveTranslationLayerRequest(
                    target_lang="en",
                    lines=_layer_lines(
                        [("첫 줄", "corrected caption text"), ("둘째 줄", "second line")]
                    ),
                    origin="caption",
                ),
            )
            assert resp.saved is True

            async with sm() as s:
                layer = await TranslationLayerRepository(s).get_layer(VIDEO, FP, "en")
                assert layer.lines[0]["translation"] == "corrected caption text"

    asyncio.run(body())


def test_save_translation_layer_refuses_to_overwrite_a_different_human_origin():
    """기존이 "wiki"인데 "caption"으로 요청 — 사람이 확인한 위키 번역을 자동 자막이
    덮어쓰면 안 된다. 에러가 아니라 saved=false로 조용히 거절한다."""

    async def body():
        async with _env() as sm:
            async with sm() as s:
                await SyncRepository(s).create(
                    video_id=VIDEO, lyrics_hash="h1", timestamps=_seed_segments(["", ""])
                )
                await TranslationLayerRepository(s).upsert_layer(
                    VIDEO,
                    FP,
                    "en",
                    lines=[{"text": "첫 줄", "translation": "trusted wiki translation"}],
                    attribution={"name": "위키", "url": None, "license": None, "source_id": "wiki"},
                    origin="wiki",
                )
                await s.commit()

            resp = await save_translation_layer(
                VIDEO,
                SaveTranslationLayerRequest(
                    target_lang="en",
                    lines=_layer_lines([("첫 줄", "auto caption text"), ("둘째 줄", "second")]),
                    origin="caption",
                ),
            )
            assert resp.saved is False
            assert (resp.matched, resp.total) == (2, 2)  # 매칭률 계산은 그대로 응답에 실린다

            async with sm() as s:
                layer = await TranslationLayerRepository(s).get_layer(VIDEO, FP, "en")
                assert layer.origin == "wiki"  # 안 바뀜
                assert layer.lines[0]["translation"] == "trusted wiki translation"

    asyncio.run(body())


def test_save_translation_layer_uses_requested_video_id_not_source_for_linked_video():
    """링크로 빌려온 영상도 요청받은 video_id를 레이어 키로 쓴다 — source_video_id가 아니다."""

    async def body():
        async with _env() as sm:
            async with sm() as s:
                await SyncRepository(s).create(
                    video_id="SRCSRCSRC01",
                    lyrics_hash="h1",
                    timestamps=_seed_segments(["", ""]),
                )
                await s.commit()

            await create_sync_link(
                SyncLinkRequest(
                    video_id="DSTDSTDST01", source_video_id="SRCSRCSRC01", offset_sec=5.0
                )
            )

            resp = await save_translation_layer(
                "DSTDSTDST01",
                SaveTranslationLayerRequest(
                    target_lang="en",
                    lines=_layer_lines([("첫 줄", "First"), ("둘째 줄", "Second")]),
                    origin="caption",
                ),
            )
            assert resp.saved is True

            async with sm() as s:
                repo = TranslationLayerRepository(s)
                assert await repo.get_layer("DSTDSTDST01", FP, "en") is not None
                assert await repo.get_layer("SRCSRCSRC01", FP, "en") is None

            lookup = await get_sync("DSTDSTDST01", lang="en")
            assert lookup.linked is not None
            assert lookup.translation_lang == "en"
            assert [seg["translation"] for seg in lookup.timestamps] == ["First", "Second"]

    asyncio.run(body())


def test_save_translation_layer_404_when_no_sync():
    async def body():
        async with _env():
            with pytest.raises(HTTPException) as exc:
                await save_translation_layer(
                    "NOSYNCNOS01",
                    SaveTranslationLayerRequest(
                        target_lang="en",
                        lines=_layer_lines([("첫 줄", "First"), ("둘째 줄", "Second")]),
                        origin="caption",
                    ),
                )
            assert exc.value.status_code == 404

    asyncio.run(body())


# ── 감사 #5: 빈 번역 persist가 좋은 레이어를 파괴하면 안 된다 ────────────────


def test_translate_persist_filters_blank_translation_lines(monkeypatch):
    # 부분 실패 — 둘째 줄만 빈 번역으로 돌아왔다. 저장되는 lines에서 그 줄은 빠져야 한다
    # (worker.translation_layer_lines와 같은 규칙: 텍스트·번역이 둘 다 있어야 남긴다).
    monkeypatch.setattr(translate_api, "LyricsTranslator", _blankable_translator({1}))

    async def body():
        async with _env() as sm:
            bg = BackgroundTasks()
            translate_lyrics(
                TranslateRequest(
                    text="\n".join(LINES),
                    source_lang="ko",
                    target_lang="en",
                    video_id=VIDEO,
                    persist=True,
                ),
                bg,
            )
            await _run_background(bg)

            async with sm() as s:
                layer = await TranslationLayerRepository(s).get_layer(VIDEO, FP, "en")
                assert layer is not None
                assert layer.lines == [{"text": "첫 줄", "translation": "[en] 첫 줄"}]

    asyncio.run(body())


def test_translate_persist_all_blank_does_not_destroy_existing_layer(monkeypatch):
    """재현: 전체 실패(또는 재시도 후에도 전부 빈 값) 응답을 persist=true로 부르면,
    필터링 후 남는 줄이 0이라 저장 자체를 건너뛰어야 한다 — 기존의 완전한 레이어가
    빈 값으로 통째로 덮이면 안 된다(upsert_layer는 부분 병합 없이 전체 교체이므로,
    걸러내지 않고 그대로 저장하면 기존 좋은 데이터가 사라진다)."""
    monkeypatch.setattr(translate_api, "LyricsTranslator", _blankable_translator({0, 1}))

    async def body():
        async with _env() as sm:
            async with sm() as s:
                await TranslationLayerRepository(s).upsert_layer(
                    VIDEO,
                    FP,
                    "en",
                    lines=[
                        {"text": "첫 줄", "translation": "Good existing first"},
                        {"text": "둘째 줄", "translation": "Good existing second"},
                    ],
                    attribution=None,
                    origin="llm",
                )
                await s.commit()

            bg = BackgroundTasks()
            translate_lyrics(
                TranslateRequest(
                    text="\n".join(LINES),
                    source_lang="ko",
                    target_lang="en",
                    video_id=VIDEO,
                    persist=True,
                ),
                bg,
            )
            await _run_background(bg)

            async with sm() as s:
                layer = await TranslationLayerRepository(s).get_layer(VIDEO, FP, "en")
                # 저장을 건너뛰었으므로 기존 값 그대로다
                assert layer.lines == [
                    {"text": "첫 줄", "translation": "Good existing first"},
                    {"text": "둘째 줄", "translation": "Good existing second"},
                ]

    asyncio.run(body())


def test_translate_persist_keeps_fingerprint_over_full_text_when_filtering(monkeypatch):
    # 필터링된 부분집합이 아니라 texts 전체로 지문을 계산해야 조회가 찾을 수 있다
    monkeypatch.setattr(translate_api, "LyricsTranslator", _blankable_translator({0}))

    async def body():
        async with _env() as sm:
            async with sm() as s:
                await SyncRepository(s).create(
                    video_id=VIDEO, lyrics_hash="h1", timestamps=_seed_segments(["", ""])
                )
                await s.commit()

            bg = BackgroundTasks()
            translate_lyrics(
                TranslateRequest(
                    text="\n".join(LINES),
                    source_lang="ko",
                    target_lang="en",
                    video_id=VIDEO,
                    persist=True,
                ),
                bg,
            )
            await _run_background(bg)

            # lang= 조회는 세그먼트 원문 전체(2줄)로 지문을 만든다 — 저장 쪽이 필터링된
            # 1줄만으로 지문을 계산했다면 여기서 못 찾는다
            resp = await get_sync(VIDEO, lang="en")
            assert resp.translation_lang == "en"
            assert resp.timestamps[1]["translation"] == "[en] 둘째 줄"

    asyncio.run(body())


# ── 감사 #6: SyncLink 커버가 원곡의 비ko 레이어를 볼 수 있어야 한다 ─────────────


def test_linked_video_falls_back_to_source_layer_when_cover_has_none():
    """원곡에만 en 레이어가 있으면, 커버(자기 레이어 없음) lang=en 조회가 그걸 받는다."""

    async def body():
        async with _env() as sm:
            async with sm() as s:
                await SyncRepository(s).create(
                    video_id="SRCSRCSRC01", lyrics_hash="h1", timestamps=_seed_segments(["", ""])
                )
                await TranslationLayerRepository(s).upsert_layer(
                    "SRCSRCSRC01",
                    FP,
                    "en",
                    lines=[
                        {"text": "첫 줄", "translation": "Original's English"},
                        {"text": "둘째 줄", "translation": "Second"},
                    ],
                    attribution=None,
                    origin="llm",
                )
                await s.commit()

            await create_sync_link(
                SyncLinkRequest(
                    video_id="DSTDSTDST01", source_video_id="SRCSRCSRC01", offset_sec=5.0
                )
            )

            resp = await get_sync("DSTDSTDST01", lang="en")
            assert resp.translation_lang == "en"
            assert [seg["translation"] for seg in resp.timestamps] == [
                "Original's English",
                "Second",
            ]
            assert "en" in resp.available_langs

    asyncio.run(body())


def test_linked_video_prefers_its_own_layer_over_source_fallback():
    # 커버 자신의 레이어가 있으면 그걸 쓴다 — 폴백은 "미스"일 때만
    async def body():
        async with _env() as sm:
            async with sm() as s:
                await SyncRepository(s).create(
                    video_id="SRCSRCSRC01", lyrics_hash="h1", timestamps=_seed_segments(["", ""])
                )
                await TranslationLayerRepository(s).upsert_layer(
                    "SRCSRCSRC01",
                    FP,
                    "en",
                    lines=[{"text": "첫 줄", "translation": "Source English"}],
                    attribution=None,
                    origin="llm",
                )
                await TranslationLayerRepository(s).upsert_layer(
                    "DSTDSTDST01",
                    FP,
                    "en",
                    lines=[{"text": "첫 줄", "translation": "Cover's own English"}],
                    attribution=None,
                    origin="caption",
                )
                await s.commit()

            await create_sync_link(
                SyncLinkRequest(
                    video_id="DSTDSTDST01", source_video_id="SRCSRCSRC01", offset_sec=0.0
                )
            )

            resp = await get_sync("DSTDSTDST01", lang="en")
            assert resp.timestamps[0]["translation"] == "Cover's own English"

    asyncio.run(body())


def test_linked_video_available_langs_falls_back_to_source():
    async def body():
        async with _env() as sm:
            async with sm() as s:
                await SyncRepository(s).create(
                    video_id="SRCSRCSRC01", lyrics_hash="h1", timestamps=_seed_segments(["", ""])
                )
                await TranslationLayerRepository(s).upsert_layer(
                    "SRCSRCSRC01",
                    FP,
                    "ja",
                    lines=[{"text": "첫 줄", "translation": "最初"}],
                    attribution=None,
                    origin="llm",
                )
                await s.commit()

            await create_sync_link(
                SyncLinkRequest(
                    video_id="DSTDSTDST01", source_video_id="SRCSRCSRC01", offset_sec=0.0
                )
            )

            # lang 미지정 조회도 available_langs는 폴백을 포함해 채운다
            resp = await get_sync("DSTDSTDST01")
            assert resp.available_langs == ["ja"]

    asyncio.run(body())


def test_linked_video_available_langs_does_not_merge_when_cover_has_its_own():
    # 커버가 자기 레이어를 하나라도 가지면(예: en) "미스"가 아니므로 폴백하지 않는다 —
    # 원곡의 다른 언어(ja)는 안 섞인다(1회 폴백 = on-miss 폴백이지 합집합이 아니다)
    async def body():
        async with _env() as sm:
            async with sm() as s:
                await SyncRepository(s).create(
                    video_id="SRCSRCSRC01", lyrics_hash="h1", timestamps=_seed_segments(["", ""])
                )
                await TranslationLayerRepository(s).upsert_layer(
                    "SRCSRCSRC01",
                    FP,
                    "ja",
                    lines=[{"text": "첫 줄", "translation": "最初"}],
                    attribution=None,
                    origin="llm",
                )
                await TranslationLayerRepository(s).upsert_layer(
                    "DSTDSTDST01",
                    FP,
                    "en",
                    lines=[{"text": "첫 줄", "translation": "First"}],
                    attribution=None,
                    origin="caption",
                )
                await s.commit()

            await create_sync_link(
                SyncLinkRequest(
                    video_id="DSTDSTDST01", source_video_id="SRCSRCSRC01", offset_sec=0.0
                )
            )

            resp = await get_sync("DSTDSTDST01")
            assert resp.available_langs == ["en"]

    asyncio.run(body())


def test_own_sync_lookup_does_not_fall_back_to_anything():
    # 자기 싱크가 있는 영상은 link_source_video_id가 None이라 폴백 자체가 없다 — 회귀 방지
    async def body():
        async with _env() as sm:
            async with sm() as s:
                await SyncRepository(s).create(
                    video_id=VIDEO, lyrics_hash="h1", timestamps=_seed_segments(["", ""])
                )
                await s.commit()

            resp = await get_sync(VIDEO, lang="en")
            assert resp.translation_lang is None
            assert resp.available_langs == []

    asyncio.run(body())


# ── 감사 #7·#10: 링크 시프트가 pron_segs·heard_spans도 옮겨야 한다 ─────────────


def _segment_with_pron_and_heard(**overrides) -> dict:
    seg = {
        "text": "テスト",
        "start": 2.0,
        "end": 4.0,
        "pron_segs": {
            "romaji": [
                {"text": "te", "start": 2.0, "end": 3.0},
                {"text": "su", "start": 3.0, "end": 4.0},
            ],
            "kana": [{"text": "テ", "start": 2.0, "end": 3.0}],
        },
        "debug": {"heard_spans": [["テ", 2.0], ["ス", 3.0]]},
    }
    seg.update(overrides)
    return seg


def test_link_shift_applies_to_pron_segs_and_heard_spans():
    async def body():
        async with _env() as sm:
            async with sm() as s:
                await SyncRepository(s).create(
                    video_id="SRCSRCSRC01",
                    lyrics_hash="h1",
                    timestamps=[_segment_with_pron_and_heard()],
                )
                await s.commit()

            await create_sync_link(
                SyncLinkRequest(
                    video_id="DSTDSTDST01", source_video_id="SRCSRCSRC01", offset_sec=10.0
                )
            )

            resp = await get_sync("DSTDSTDST01")
            seg = resp.timestamps[0]
            assert seg["pron_segs"]["romaji"] == [
                {"text": "te", "start": 12.0, "end": 13.0},
                {"text": "su", "start": 13.0, "end": 14.0},
            ]
            assert seg["pron_segs"]["kana"] == [{"text": "テ", "start": 12.0, "end": 13.0}]
            assert seg["debug"]["heard_spans"] == [["テ", 12.0], ["ス", 13.0]]

    asyncio.run(body())


def test_link_shift_with_rate_applies_to_pron_segs_and_heard_spans():
    async def body():
        async with _env() as sm:
            async with sm() as s:
                await SyncRepository(s).create(
                    video_id="SRCSRCSRC01",
                    lyrics_hash="h1",
                    timestamps=[_segment_with_pron_and_heard()],
                )
                await s.commit()

            await create_sync_link(
                SyncLinkRequest(
                    video_id="DSTDSTDST01",
                    source_video_id="SRCSRCSRC01",
                    offset_sec=1.0,
                    rate=2.0,
                )
            )

            resp = await get_sync("DSTDSTDST01")
            seg = resp.timestamps[0]
            # sh(t) = t/rate + offset — pron_segments·words와 동일 수식
            assert seg["pron_segs"]["romaji"][0]["start"] == 2.0  # 2.0/2.0 + 1.0
            assert seg["pron_segs"]["romaji"][0]["end"] == 2.5  # 3.0/2.0 + 1.0
            assert seg["debug"]["heard_spans"][0] == ["テ", 2.0]
            assert seg["debug"]["heard_spans"][1] == ["ス", 2.5]

    asyncio.run(body())


def test_link_shift_pron_segs_and_heard_spans_are_optional():
    # 두 필드가 없는 세그먼트(구형 싱크)에서 시프트가 예외 없이 통과해야 한다
    async def body():
        async with _env() as sm:
            async with sm() as s:
                await SyncRepository(s).create(
                    video_id="SRCSRCSRC01",
                    lyrics_hash="h1",
                    timestamps=[{"text": "가사", "start": 1.0, "end": 2.0}],
                )
                await s.commit()

            await create_sync_link(
                SyncLinkRequest(
                    video_id="DSTDSTDST01", source_video_id="SRCSRCSRC01", offset_sec=5.0
                )
            )

            resp = await get_sync("DSTDSTDST01")
            assert resp.timestamps[0]["start"] == 6.0

    asyncio.run(body())


# ── 감사 #3: 구세대 싱크(pron dict 없음)를 조회 시 기회적으로 채운다 ────────────
#
# **순서 의존**: S5의 merge 가드(비한글 발음이 legacy pronunciation에 박히는 것을 막는
# merge_line_meta 가드, 커밋 ea145c0)가 이미 마스터에 있다는 전제로만 안전하다 — 이
# 스위트를 실행하는 시점에는 그 가드가 이미 커밋돼 있음을 확인했다(worker.py 상태로
# 검증). 그 가드 없이 이 기능만 먼저 배포되면, 오염된(비한글) line_meta 발음이 legacy
# seg["pronunciation"]에 박힌 채 attach_pron_variants가 그 값을 pron.hangul에 영구
# 고착시킬 수 있다(감사 치명 #1 서버 잔여).


def _legacy_ja_segment() -> dict:
    # 구세대 싱크의 특징: 기존 hangul 독음(pronunciation)은 있지만 표기별 pron dict가
    # pron dict 기능이 생기기 전에 만들어져 없다.
    return {"text": "元気", "start": 0.0, "end": 1.0, "pronunciation": "겐키"}


def test_lazy_attach_fills_pron_dict_and_reflects_in_the_immediate_response():
    from everyric2.text.pron_style import romaji_line

    async def body():
        async with _env() as sm:
            async with sm() as s:
                await SyncRepository(s).create(
                    video_id=VIDEO, lyrics_hash="h1", timestamps=[_legacy_ja_segment()]
                )
                await s.commit()

            # 재조회 없이 이번 응답부터 즉시 반영된다
            resp = await get_sync(VIDEO)
            seg = resp.timestamps[0]
            assert seg["pron"]["hangul"] == "겐키"
            assert seg["pron"]["romaji"] == romaji_line("元気")[0]
            assert "kana" in seg["pron"]

    asyncio.run(body())


def test_lazy_attach_persists_to_the_row_via_background_task():
    async def body():
        async with _env() as sm:
            async with sm() as s:
                sync = await SyncRepository(s).create(
                    video_id=VIDEO, lyrics_hash="h1", timestamps=[_legacy_ja_segment()]
                )
                sync_id = sync.id
                await s.commit()

            bg = BackgroundTasks()
            await get_sync(VIDEO, background_tasks=bg)
            await _run_background(bg)

            async with sm() as s:
                stored = await s.get(SyncResult, sync_id)
                stored_seg = stored.timestamps["segments"][0]
                assert stored_seg["pron"]["hangul"] == "겐키"
                assert "romaji" in stored_seg["pron"]

            # 저장 후 재조회에도 그대로 남아 있다(재계산 없이 저장분을 그대로 서빙)
            resp = await get_sync(VIDEO)
            assert resp.timestamps[0]["pron"]["hangul"] == "겐키"

    asyncio.run(body())


def test_lazy_attach_is_idempotent_and_does_not_recompute_existing_pron():
    # 이미 pron이 있는 세그는(예: 심판 판정을 반영해 직렬화 때 만든 값) 손대지 않는다
    async def body():
        async with _env() as sm:
            seg = _legacy_ja_segment()
            seg["pron"] = {"hangul": "겐키", "romaji": "already there"}
            async with sm() as s:
                await SyncRepository(s).create(video_id=VIDEO, lyrics_hash="h1", timestamps=[seg])
                await s.commit()

            resp = await get_sync(VIDEO)
            assert resp.timestamps[0]["pron"] == {"hangul": "겐키", "romaji": "already there"}

    asyncio.run(body())


def test_lazy_attach_skips_entirely_without_fugashi(monkeypatch):
    # fugashi 미가용(pykakasi 폴백)이면 통째로 건너뛴다 — 저품질 값이 DB에 영구 고착되지
    # 않게 한다. 지역 임포트라 원본(everyric2.text.ja_reading)을 패치해야 한다.
    monkeypatch.setattr("everyric2.text.ja_reading.reading_source", lambda: "pykakasi")

    async def body():
        async with _env() as sm:
            async with sm() as s:
                await SyncRepository(s).create(
                    video_id=VIDEO, lyrics_hash="h1", timestamps=[_legacy_ja_segment()]
                )
                await s.commit()

            resp = await get_sync(VIDEO)
            assert "pron" not in resp.timestamps[0]

    asyncio.run(body())


def test_lazy_attach_does_not_apply_to_linked_lookup():
    # 링크로 빌려온 싱크는 시프트된 사본이라 lazy attach를 부르지 않는다 — 원곡을 직접
    # 조회하면 원곡 행에 붙고, 그 뒤로는 시프트(#7)가 커버 조회에도 자연히 실어 준다.
    async def body():
        async with _env() as sm:
            async with sm() as s:
                await SyncRepository(s).create(
                    video_id="SRCSRCSRC01", lyrics_hash="h1", timestamps=[_legacy_ja_segment()]
                )
                await s.commit()

            await create_sync_link(
                SyncLinkRequest(
                    video_id="DSTDSTDST01", source_video_id="SRCSRCSRC01", offset_sec=5.0
                )
            )

            resp = await get_sync("DSTDSTDST01")
            assert "pron" not in resp.timestamps[0]

    asyncio.run(body())


def test_lazy_attach_persist_gives_up_if_segment_count_changed_meanwhile():
    # 백그라운드 실행 전에 행의 세그 구성이 바뀌면(재생성 등) 인덱스 대응을 신뢰할 수
    # 없어 포기한다 — 엉뚱한 세그에 값을 붙이는 사고보다 다음 조회에 다시 시도하는 편이 낫다
    async def body():
        async with _env() as sm:
            async with sm() as s:
                sync = await SyncRepository(s).create(
                    video_id=VIDEO, lyrics_hash="h1", timestamps=[_legacy_ja_segment()]
                )
                sync_id = sync.id
                await s.commit()

            bg = BackgroundTasks()
            await get_sync(VIDEO, background_tasks=bg)

            async with sm() as s:
                stored = await s.get(SyncResult, sync_id)
                updated = dict(stored.timestamps)
                updated["segments"] = [_legacy_ja_segment(), _legacy_ja_segment()]
                stored.timestamps = updated
                await s.commit()

            await _run_background(bg)  # 예외 없이 통과해야 한다

            async with sm() as s:
                stored = await s.get(SyncResult, sync_id)
                assert len(stored.timestamps["segments"]) == 2
                assert "pron" not in stored.timestamps["segments"][0]

    asyncio.run(body())


# ── H7PR 실증: 순서 정렬 매칭(A) + 저장 재매핑(B) + 크로스 지문 이관(C) ─────────
#
# 실화: miraheze 기반 새 싱크(36줄)와 vocaro 위키 페이지(42줄)의 줄 분할이 달라 줄 단위
# 동등 매칭이 50%를 못 넘겼다 — 위키 번역이 있는데도 LLM이 다시 불려 저품질 번역이
# 고착됐다. 아래는 축소된 재현 픽스처(합침 1건 + 쪼갬 1건 + 정확 매칭 1건)로 실제
# 42/36줄 규모를 그대로 옮기는 대신 같은 부류의 분할 불일치를 검증한다.


# ── A: align_translation_lines 단위 테스트 ───────────────────────────────


def test_align_merges_when_wiki_line_matches_several_seg_lines():
    # 위키가 합침: 위키 한 줄 "AB"가 세그 "A"+"B"의 연결과 같다 — 번역은 첫 세그에만
    from everyric2.server.text_fingerprint import align_translation_lines

    wiki = [{"text": "AB", "translation": "T_AB"}, {"text": "C", "translation": "T_C"}]
    segs = ["A", "B", "C"]
    assert align_translation_lines(segs, wiki) == ["T_AB", None, "T_C"]


def test_align_splits_when_seg_line_matches_several_wiki_lines():
    # 위키가 쪼갬: 세그 한 줄 "CD"가 위키 "C"+"D"의 연결과 같다 — 번역을 이어 붙여 배정
    from everyric2.server.text_fingerprint import align_translation_lines

    wiki = [
        {"text": "AB", "translation": "T_AB"},
        {"text": "C", "translation": "T_C"},
        {"text": "D", "translation": "T_D"},
        {"text": "E", "translation": "T_E"},
    ]
    segs = ["A", "B", "CD", "E"]
    assert align_translation_lines(segs, wiki) == ["T_AB", None, "T_CT_D", "T_E"]


def test_align_resyncs_past_an_extra_wiki_only_line():
    # 위키에만 있는 여분 줄("intro")은 버려지고, 그 다음부터 다시 정상 매칭된다
    from everyric2.server.text_fingerprint import align_translation_lines

    wiki = [
        {"text": "intro", "translation": "T_intro"},
        {"text": "hello", "translation": "T_hello"},
        {"text": "world", "translation": "T_world"},
    ]
    segs = ["hello", "world"]
    assert align_translation_lines(segs, wiki) == ["T_hello", "T_world"]


def test_align_gives_up_on_remaining_segs_when_no_anchor_found():
    from everyric2.server.text_fingerprint import align_translation_lines

    wiki = [{"text": "completely unrelated", "translation": "T_x"}]
    segs = ["hello", "world"]
    assert align_translation_lines(segs, wiki) == [None, None]


def test_align_handles_empty_inputs():
    from everyric2.server.text_fingerprint import align_translation_lines

    assert align_translation_lines([], [{"text": "a", "translation": "A"}]) == []
    assert align_translation_lines(["a", "b"], []) == [None, None]
    assert align_translation_lines([], []) == []


def test_align_exact_match_is_unaffected_by_normalize_line_whitespace():
    # 정확 매칭은 normalize_line 정규화를 그대로 쓴다 — 공백 위치 차이는 무시된다
    from everyric2.server.text_fingerprint import align_translation_lines

    wiki = [{"text": "Are  you ready ?", "translation": "준비됐어?"}]
    segs = ["Are you ready?"]
    assert align_translation_lines(segs, wiki) == ["준비됐어?"]


# ── A2: 느슨한 매칭 폴백 — 구두점 차이만으로 못 붙던 실측 2건 (2026-08-03) ──────────


def test_align_falls_back_to_loose_match_on_trailing_punctuation_difference():
    """vg6pnvn1u10 idx 0 재현: "I love you." vs "I love you" — 엄격 키는 마침표를
    보존해 서로 다른 키가 된다. 느슨한 키(구두점 제거)가 폴백으로 붙여야 한다."""
    from everyric2.server.text_fingerprint import align_translation_lines

    wiki = [{"text": "I love you.", "translation": "사랑해"}]
    segs = ["I love you"]
    assert align_translation_lines(segs, wiki) == ["사랑해"]


def test_align_combine_falls_back_to_loose_match_when_a_piece_has_punctuation():
    """OVwCr2MESfo류 재현: 위키가 합친 한 줄("ドラマを見るのが好きだった。")과 세그가
    쪼갠 두 줄("ドラマを見るのが" + "好きだった")을 엄격 결합("ドラマを見るのが" +
    "好きだった" = "ドラマを見るのが好きだった", 위키 쪽엔 마침표가 남아 있다)이
    못 붙일 때 느슨한 결합이 대신 붙는다."""
    from everyric2.server.text_fingerprint import align_translation_lines

    wiki = [
        {"text": "ドラマを見るのが好きだった。", "translation": "드라마 보는 걸 좋아했어"},
        {"text": "次の行", "translation": "다음 줄"},
    ]
    segs = ["ドラマを見るのが", "好きだった", "次の行"]
    assert align_translation_lines(segs, wiki) == [
        "드라마 보는 걸 좋아했어", None, "다음 줄",
    ]


def test_align_split_falls_back_to_loose_match_when_a_piece_has_punctuation():
    # 결합 방향(세그가 합쳐 있고 위키가 쪼갠 경우)도 동일하게 느슨한 폴백이 붙는다
    from everyric2.server.text_fingerprint import align_translation_lines

    wiki = [
        {"text": "ドラマを見るのが", "translation": "T1"},
        {"text": "好きだった。", "translation": "T2"},
    ]
    segs = ["ドラマを見るのが好きだった"]
    assert align_translation_lines(segs, wiki) == ["T1T2"]


def test_align_does_not_collide_short_punctuation_only_fragments():
    """느슨한 키가 2자 미만이 되는 조각(감탄사·기호뿐)은 매칭에서 제외된다 — 무관한
    다른 짧은 조각과 우연히 같은 키가 되는 사고를 막는다."""
    from everyric2.server.text_fingerprint import align_translation_lines

    wiki = [{"text": "!", "translation": "완전히 다른 뜻"}]
    segs = ["?"]
    assert align_translation_lines(segs, wiki) == [None]


def test_align_strict_punctuation_pairs_still_disambiguate_when_both_present():
    """엄격 키가 먼저 시도된다 — 뜻이 갈리는 구두점 쌍(둘 다 값이 있으면)은 느슨한
    폴백까지 가지 않고 엄격 매칭이 각각 정확히 집어낸다."""
    from everyric2.server.text_fingerprint import align_translation_lines

    wiki = [
        {"text": "行く。", "translation": "간다"},
        {"text": "行く？", "translation": "갈까?"},
    ]
    segs = ["行く。", "行く？"]
    assert align_translation_lines(segs, wiki) == ["간다", "갈까?"]


def test_loose_normalize_line_strips_punctuation_and_guards_short_keys():
    from everyric2.server.text_fingerprint import loose_normalize_line

    assert loose_normalize_line("I love you.") == loose_normalize_line("I love you")
    assert loose_normalize_line("行く。") == loose_normalize_line("行く？")
    assert loose_normalize_line("!") is None  # 2자 미만은 매칭 불가
    assert loose_normalize_line("") is None


# ── B: 저장 엔드포인트가 재정렬을 쓰는지(세그 텍스트로 재키잉) ────────────────


def test_save_translation_layer_remaps_split_and_merged_lines_to_segment_text():
    """H7PR 축소 재현 — 제출된 lines의 줄 분할이 세그와 달라도(합침+쪼갬 혼재) 재정렬해
    저장하고, 저장되는 lines는 세그 자신의 텍스트로 재키잉된다(지문과 정합)."""

    async def body():
        async with _env() as sm:
            seg_texts = ["A", "B", "CD", "E"]
            async with sm() as s:
                await SyncRepository(s).create(
                    video_id=VIDEO,
                    lyrics_hash="h1",
                    timestamps=[{"text": t, "translation": ""} for t in seg_texts],
                )
                await s.commit()

            resp = await save_translation_layer(
                VIDEO,
                SaveTranslationLayerRequest(
                    target_lang="en",
                    lines=_layer_lines(
                        [("AB", "T_AB"), ("C", "T_C"), ("D", "T_D"), ("E", "T_E")]
                    ),
                    origin="wiki",
                ),
            )
            assert resp.saved is True
            assert (resp.matched, resp.total) == (3, 4)  # A·B는 합쳐서 1매칭, CD·E는 각 1매칭

            fp = lines_fingerprint(seg_texts)
            async with sm() as s:
                layer = await TranslationLayerRepository(s).get_layer(VIDEO, fp, "en")
                # 저장된 lines는 세그 텍스트("A","CD","E") 기준 — 제출 원문("AB","C","D")이 아니다
                assert layer.lines == [
                    {"text": "A", "translation": "T_AB"},
                    {"text": "CD", "translation": "T_CT_D"},
                    {"text": "E", "translation": "T_E"},
                ]

            # 지문이 세그 기준이므로 lang 조회가 바로 찾는다(재정렬 없이)
            lookup = await get_sync(VIDEO, lang="en")
            assert lookup.translation_lang == "en"
            assert lookup.timestamps[0]["translation"] == "T_AB"
            assert lookup.timestamps[1]["translation"] == ""  # B는 매칭 없음(합침의 나머지)
            assert lookup.timestamps[2]["translation"] == "T_CT_D"
            assert lookup.timestamps[3]["translation"] == "T_E"

    asyncio.run(body())


# ── C: 크로스 지문 이관 ───────────────────────────────────────────────


def test_cross_fingerprint_migration_serves_and_persists_human_layer_under_new_fingerprint():
    """실화 재현: 옛 지문(42줄류 분할)에 위키 ko 번역이 있고, 재생성으로 새 지문(36줄류
    분할)이 됐다 — 새 지문 조회가 예전 위키 번역을 찾아 재정렬해 서빙하고, 다음부터
    바로 찾도록 새 지문에도 저장(BackgroundTasks)한다."""

    async def body():
        async with _env() as sm:
            old_seg_texts = ["one", "two", "three"]  # 예전(위키) 분할
            new_seg_texts = ["onetwo", "three"]  # 새 싱크의 분할(합침)

            async with sm() as s:
                old_fp = lines_fingerprint(old_seg_texts)
                await TranslationLayerRepository(s).upsert_layer(
                    VIDEO,
                    old_fp,
                    "ko",
                    lines=[
                        {"text": "one", "translation": "하나"},
                        {"text": "two", "translation": "둘"},
                        {"text": "three", "translation": "셋"},
                    ],
                    attribution={"name": "위키", "url": None, "license": None, "source_id": "wiki"},
                    origin="wiki",
                )
                await SyncRepository(s).create(
                    video_id=VIDEO,
                    lyrics_hash="h_new",
                    timestamps=[{"text": t, "translation": ""} for t in new_seg_texts],
                )
                await s.commit()

            bg = BackgroundTasks()
            resp = await get_sync(VIDEO, lang="ko", background_tasks=bg)
            assert resp.translation_lang == "ko"
            assert resp.timestamps[0]["translation"] == "하나둘"  # "one"+"two" 합침 → 이어붙임
            assert resp.timestamps[1]["translation"] == "셋"

            await _run_background(bg)

            new_fp = lines_fingerprint(new_seg_texts)
            async with sm() as s:
                migrated_layer = await TranslationLayerRepository(s).get_layer(VIDEO, new_fp, "ko")
                assert migrated_layer is not None
                assert migrated_layer.origin == "wiki"  # origin 유지
                assert migrated_layer.attribution == {
                    "name": "위키", "url": None, "license": None, "source_id": "wiki",
                }
                assert migrated_layer.lines == [
                    {"text": "onetwo", "translation": "하나둘"},
                    {"text": "three", "translation": "셋"},
                ]

            # 이관 후에는 정확 매칭으로 바로 찾는다 — 재정렬을 다시 치르지 않는다
            resp2 = await get_sync(VIDEO, lang="ko")
            assert resp2.translation_lang == "ko"
            assert resp2.timestamps[0]["translation"] == "하나둘"

    asyncio.run(body())


def test_cross_fingerprint_migration_reports_the_migrated_source_attribution_and_origin():
    """F5(2026-08-04 감사) — 이관 서빙 응답의 translation_attribution/translation_origin은
    이관 **원본**(다른 지문의 사람 origin 레이어)의 값을 낸다 — 실제로 그 출처의
    번역이 재정렬돼 실린 것이기 때문이다."""

    async def body():
        async with _env() as sm:
            old_seg_texts = ["one", "two", "three"]
            new_seg_texts = ["onetwo", "three"]

            async with sm() as s:
                old_fp = lines_fingerprint(old_seg_texts)
                await TranslationLayerRepository(s).upsert_layer(
                    VIDEO,
                    old_fp,
                    "ko",
                    lines=[
                        {"text": "one", "translation": "하나"},
                        {"text": "two", "translation": "둘"},
                        {"text": "three", "translation": "셋"},
                    ],
                    attribution={"name": "위키", "url": None, "license": None, "source_id": "wiki"},
                    origin="wiki",
                )
                await SyncRepository(s).create(
                    video_id=VIDEO,
                    lyrics_hash="h_new",
                    timestamps=[{"text": t, "translation": ""} for t in new_seg_texts],
                )
                await s.commit()

            resp = await get_sync(VIDEO, lang="ko", background_tasks=BackgroundTasks())
            assert resp.translation_lang == "ko"
            assert resp.translation_origin == "wiki"
            assert resp.translation_attribution == {
                "name": "위키", "url": None, "license": None, "source_id": "wiki",
            }

    asyncio.run(body())


def test_cross_fingerprint_migration_does_not_migrate_llm_origin():
    # llm origin은 이관 대상이 아니다 — 품질 보증이 없어 재생성이 낫다. 새 싱크는 옛
    # 지문(2줄)과 다른 지문이 되도록 한 줄로 합쳐 둔다(그래도 align은 성공할 수 있는
    # 정도 — 검증 대상은 "성공해도 llm은 이관 안 된다"는 origin 필터 자체다).
    async def body():
        async with _env() as sm:
            old_seg_texts = ["one", "two"]

            async with sm() as s:
                old_fp = lines_fingerprint(old_seg_texts)
                await TranslationLayerRepository(s).upsert_layer(
                    VIDEO,
                    old_fp,
                    "ko",
                    lines=[
                        {"text": "one", "translation": "머신원"},
                        {"text": "two", "translation": "머신투"},
                    ],
                    attribution=None,
                    origin="llm",
                )
                await SyncRepository(s).create(
                    video_id=VIDEO,
                    lyrics_hash="h_new",
                    timestamps=[{"text": "onetwo", "translation": ""}],
                )
                await s.commit()

            resp = await get_sync(VIDEO, lang="ko")
            # llm 레이어는 이관되지 않는다 — 세그에 레거시 번역도 없으므로 translation_lang None
            assert resp.translation_lang is None
            assert resp.timestamps[0]["translation"] == ""

    asyncio.run(body())


def test_cross_fingerprint_migration_skipped_when_coverage_too_low():
    async def body():
        async with _env() as sm:
            old_seg_texts = ["one", "two", "three", "four"]

            async with sm() as s:
                old_fp = lines_fingerprint(old_seg_texts)
                await TranslationLayerRepository(s).upsert_layer(
                    VIDEO,
                    old_fp,
                    "ko",
                    lines=[{"text": t, "translation": f"번역_{t}"} for t in old_seg_texts],
                    attribution=None,
                    origin="wiki",
                )
                # 새 싱크는 완전히 다른 가사 — 정렬이 하나도 안 맞는다
                await SyncRepository(s).create(
                    video_id=VIDEO,
                    lyrics_hash="h_new",
                    timestamps=[
                        {"text": "완전히 다른 가사 줄 하나", "translation": ""},
                        {"text": "완전히 다른 가사 줄 둘", "translation": ""},
                    ],
                )
                await s.commit()

            resp = await get_sync(VIDEO, lang="ko")
            assert resp.translation_lang is None
            assert all(seg["translation"] == "" for seg in resp.timestamps)

    asyncio.run(body())


def test_cross_fingerprint_migration_prefers_exact_match_when_current_fingerprint_has_layer():
    # 지금 지문에 이미 레이어가 있으면 이관 시도 자체를 안 한다(그럴 필요가 없다)
    async def body():
        async with _env() as sm:
            seg_texts = ["one", "two"]
            async with sm() as s:
                fp = lines_fingerprint(seg_texts)
                await TranslationLayerRepository(s).upsert_layer(
                    VIDEO, fp, "ko",
                    lines=[{"text": "one", "translation": "정확매칭하나"}],
                    attribution=None, origin="wiki",
                )
                # 다른 지문에도 사람 레이어가 있지만(이관 후보), 현재 지문이 이미 있으므로 무시돼야 한다
                await TranslationLayerRepository(s).upsert_layer(
                    VIDEO, "deadbeef" * 4, "ko",
                    lines=[{"text": "one", "translation": "이관후보"}],
                    attribution=None, origin="wiki",
                )
                await SyncRepository(s).create(
                    video_id=VIDEO, lyrics_hash="h1",
                    timestamps=[{"text": t, "translation": ""} for t in seg_texts],
                )
                await s.commit()

            resp = await get_sync(VIDEO, lang="ko")
            assert resp.timestamps[0]["translation"] == "정확매칭하나"

    asyncio.run(body())


# ── translations_by_lang — 보유 언어 전부를 lookup에 동봉 (사용자 채택안) ─────────


def test_translations_by_lang_includes_all_layers_matching_segment_order():
    """2언어 레이어 곡의 lookup에 두 배열이 동봉되고, 각 배열은 세그 길이와 같으며
    매칭 안 된 세그는 None으로 채워진다."""

    async def body():
        async with _env() as sm:
            async with sm() as s:
                await SyncRepository(s).create(
                    video_id=VIDEO, lyrics_hash="h1", timestamps=_seed_segments(["", ""])
                )
                await TranslationLayerRepository(s).upsert_layer(
                    VIDEO,
                    FP,
                    "en",
                    lines=[{"text": "첫 줄", "translation": "First line"}],  # 둘째 줄은 없음
                    attribution=None,
                    origin="llm",
                )
                await TranslationLayerRepository(s).upsert_layer(
                    VIDEO,
                    FP,
                    "ja",
                    lines=[
                        {"text": "첫 줄", "translation": "最初の行"},
                        {"text": "둘째 줄", "translation": "二番目の行"},
                    ],
                    attribution=None,
                    origin="wiki",
                )
                await s.commit()

            resp = await get_sync(VIDEO)  # lang 미지정 — 그래도 동봉된다
            assert resp.translations_by_lang == {
                "en": ["First line", None],  # 둘째 줄은 레이어에 없어 None
                "ja": ["最初の行", "二番目の行"],
            }

    asyncio.run(body())


def test_translations_by_lang_present_regardless_of_lang_param():
    # lang= 지정 요청에도 동봉되고, 요청한 lang과 무관하게 다른 언어도 함께 담긴다
    async def body():
        async with _env() as sm:
            async with sm() as s:
                await SyncRepository(s).create(
                    video_id=VIDEO, lyrics_hash="h1", timestamps=_seed_segments(["", ""])
                )
                await TranslationLayerRepository(s).upsert_layer(
                    VIDEO, FP, "en",
                    lines=[
                        {"text": "첫 줄", "translation": "First"},
                        {"text": "둘째 줄", "translation": "Second"},
                    ],
                    attribution=None, origin="llm",
                )
                await TranslationLayerRepository(s).upsert_layer(
                    VIDEO, FP, "ja",
                    lines=[{"text": "첫 줄", "translation": "最初"}],
                    attribution=None, origin="wiki",
                )
                await s.commit()

            resp = await get_sync(VIDEO, lang="en")
            # 요청한 lang(en)에 따라 timestamps.translation은 en으로 덮이지만,
            # translations_by_lang은 두 언어 모두 담는다
            assert resp.translation_lang == "en"
            assert [seg["translation"] for seg in resp.timestamps] == ["First", "Second"]
            assert resp.translations_by_lang == {
                "en": ["First", "Second"],
                "ja": ["最初", None],
            }

    asyncio.run(body())


def test_translations_by_lang_includes_legacy_ko_when_no_ko_layer():
    async def body():
        async with _env() as sm:
            async with sm() as s:
                await SyncRepository(s).create(
                    video_id=VIDEO, lyrics_hash="h1",
                    timestamps=_seed_segments(["레거시 번역 1", ""]),
                )
                await s.commit()

            resp = await get_sync(VIDEO)
            assert resp.translations_by_lang == {"ko": ["레거시 번역 1", None]}

    asyncio.run(body())


def test_translations_by_lang_ko_layer_takes_precedence_over_legacy():
    # ko 레이어가 있으면 그걸 쓴다 — 세그의 레거시 텍스트로 덮지 않는다
    async def body():
        async with _env() as sm:
            async with sm() as s:
                await SyncRepository(s).create(
                    video_id=VIDEO, lyrics_hash="h1",
                    timestamps=_seed_segments(["레거시 번역 1", ""]),
                )
                await TranslationLayerRepository(s).upsert_layer(
                    VIDEO, FP, "ko",
                    lines=[{"text": "첫 줄", "translation": "정식 레이어 번역"}],
                    attribution=None, origin="wiki",
                )
                await s.commit()

            resp = await get_sync(VIDEO)
            assert resp.translations_by_lang == {"ko": ["정식 레이어 번역", None]}

    asyncio.run(body())


def test_translations_by_lang_cross_fingerprint_candidate_when_fingerprints_differ():
    async def body():
        async with _env() as sm:
            old_seg_texts = ["hello", "world"]
            new_seg_texts = ["hello world"]  # 세그 개수가 달라 지문도 다르다(합침)

            async with sm() as s:
                old_fp = lines_fingerprint(old_seg_texts)
                await TranslationLayerRepository(s).upsert_layer(
                    VIDEO, old_fp, "en",
                    lines=[
                        {"text": "hello", "translation": "Hello"},
                        {"text": "world", "translation": "World"},
                    ],
                    attribution=None, origin="wiki",
                )
                await SyncRepository(s).create(
                    video_id=VIDEO, lyrics_hash="h_new",
                    timestamps=[{"text": t, "translation": ""} for t in new_seg_texts],
                )
                await s.commit()

            new_fp = lines_fingerprint(new_seg_texts)
            assert old_fp != new_fp

            resp = await get_sync(VIDEO)
            # "hello"+"world" 결합이 세그 "hello world"와 일치(위키가 쪼갬 반대 — 위키가
            # 더 잘게 쪼개져 있고 세그가 합쳐졌으므로 실제로는 "위키가 합침" 케이스와 대칭인
            # "세그가 합침" — align_translation_lines 관점에서는 위키 여러 줄이 세그 하나와
            # 일치하는 "위키가 쪼갬" 케이스로 해석된다(번역을 이어 붙여 배정)
            assert resp.translations_by_lang == {"en": ["HelloWorld"]}

            # 읽기 전용 확인 — 이 조회만으로는 새 지문에 저장되지 않는다(lang= 요청 전까지)
            async with sm() as s:
                assert await TranslationLayerRepository(s).get_layer(VIDEO, new_fp, "en") is None

    asyncio.run(body())


def test_translations_by_lang_is_capped_at_five_languages():
    async def body():
        async with _env() as sm:
            async with sm() as s:
                await SyncRepository(s).create(
                    video_id=VIDEO, lyrics_hash="h1", timestamps=_seed_segments(["", ""])
                )
                for lang_code in ["en", "ja", "zh", "fr", "de", "es", "it"]:  # 7개 — 5개로 잘려야 한다
                    await TranslationLayerRepository(s).upsert_layer(
                        VIDEO, FP, lang_code,
                        lines=[{"text": "첫 줄", "translation": f"T-{lang_code}"}],
                        attribution=None, origin="llm",
                    )
                await s.commit()

            resp = await get_sync(VIDEO)
            assert len(resp.translations_by_lang) == 5

    asyncio.run(body())


def test_translations_by_lang_is_none_when_sync_not_found():
    async def body():
        async with _env():
            resp = await get_sync("NOSYNCNOS01")
            assert resp.found is False
            assert resp.translations_by_lang is None

    asyncio.run(body())


# ── 사람 크로스 지문 번역이 현재 지문의 llm 정확 매칭을 이긴다 (H7PR 프로드 실측) ────
#
# 구멍: 이관(C)은 정확 매칭 "미스"에만 발동했다. 그런데 llm 레이어가 현재 지문 자리를
# 이미 차지하고 있으면 get_layer가 값을 돌려주므로 "미스"가 아니다 — 그래서 다른 지문의
# (ko,wiki,42) 같은 사람 번역이 (ko,llm,36) 자리를 절대 못 이겼다. 아래는 그 재현.


def test_human_cross_fingerprint_layer_wins_over_exact_llm_match():
    """현재 지문에 llm, 다른 지문에 wiki(줄 분할이 다름) — lang 조회가 wiki 값을 서빙하고
    같은 (video,fp,lang) 자리를 origin=wiki로 교체한다. 재조회는 정확 매칭으로 바로 찾는다."""

    async def body():
        async with _env() as sm:
            old_seg_texts = ["one", "two", "three"]  # 위키(옛 분할, 42줄류)
            new_seg_texts = ["onetwo", "three"]  # 현재 싱크(새 분할, 36줄류 — 합침)

            async with sm() as s:
                # 현재 지문 자리를 llm이 이미 차지하고 있다 — 이게 구멍의 원인이었다
                new_fp = lines_fingerprint(new_seg_texts)
                await TranslationLayerRepository(s).upsert_layer(
                    VIDEO,
                    new_fp,
                    "ko",
                    lines=[
                        {"text": "onetwo", "translation": "머신하나둘"},
                        {"text": "three", "translation": "머신셋"},
                    ],
                    attribution=None,
                    origin="llm",
                )
                # 다른(옛) 지문에 사람이 확인한 위키 번역이 살아 있다
                old_fp = lines_fingerprint(old_seg_texts)
                await TranslationLayerRepository(s).upsert_layer(
                    VIDEO,
                    old_fp,
                    "ko",
                    lines=[
                        {"text": "one", "translation": "하나"},
                        {"text": "two", "translation": "둘"},
                        {"text": "three", "translation": "셋"},
                    ],
                    attribution={"name": "위키", "url": None, "license": None, "source_id": "wiki"},
                    origin="wiki",
                )
                await SyncRepository(s).create(
                    video_id=VIDEO,
                    lyrics_hash="h_new",
                    timestamps=[{"text": t, "translation": ""} for t in new_seg_texts],
                )
                await s.commit()

            bg = BackgroundTasks()
            resp = await get_sync(VIDEO, lang="ko", background_tasks=bg)
            # 머신 번역("머신하나둘"/"머신셋")이 아니라 위키 값("one"+"two" 합침 이어붙임)이 나온다
            assert resp.translation_lang == "ko"
            assert resp.timestamps[0]["translation"] == "하나둘"
            assert resp.timestamps[1]["translation"] == "셋"

            await _run_background(bg)

            # 같은 (video, new_fp, "ko") 자리가 llm → wiki로 교체됐다(새 행이 아니다)
            async with sm() as s:
                repo = TranslationLayerRepository(s)
                layer = await repo.get_layer(VIDEO, new_fp, "ko")
                assert layer is not None
                assert layer.origin == "wiki"
                assert layer.attribution == {
                    "name": "위키", "url": None, "license": None, "source_id": "wiki",
                }
                assert layer.lines == [
                    {"text": "onetwo", "translation": "하나둘"},
                    {"text": "three", "translation": "셋"},
                ]

            # 재조회는 정확 매칭으로 바로 찾는다(이관 재시도 없이)
            resp2 = await get_sync(VIDEO, lang="ko")
            assert resp2.translation_lang == "ko"
            assert resp2.timestamps[0]["translation"] == "하나둘"

    asyncio.run(body())


def test_llm_only_with_no_human_candidate_keeps_existing_behavior():
    # 사람 후보가 아예 없으면(다른 지문에도 llm만 있거나 아무것도 없음) 기존 llm을 그대로 서빙
    async def body():
        async with _env() as sm:
            seg_texts = ["one", "two"]
            async with sm() as s:
                fp = lines_fingerprint(seg_texts)
                await TranslationLayerRepository(s).upsert_layer(
                    VIDEO, fp, "ko",
                    lines=[
                        {"text": "one", "translation": "머신하나"},
                        {"text": "two", "translation": "머신둘"},
                    ],
                    attribution=None, origin="llm",
                )
                await SyncRepository(s).create(
                    video_id=VIDEO, lyrics_hash="h1",
                    timestamps=[{"text": t, "translation": ""} for t in seg_texts],
                )
                await s.commit()

            resp = await get_sync(VIDEO, lang="ko")
            assert resp.translation_lang == "ko"
            assert resp.timestamps[0]["translation"] == "머신하나"
            assert resp.timestamps[1]["translation"] == "머신둘"

            async with sm() as s:
                layer = await TranslationLayerRepository(s).get_layer(VIDEO, fp, "ko")
                assert layer.origin == "llm"  # 안 바뀜

    asyncio.run(body())


def test_human_cross_fingerprint_does_not_challenge_exact_human_match():
    # 정확 매칭 자체가 이미 사람 origin이면 이관을 시도조차 하지 않는다(현재 지문 우선)
    async def body():
        async with _env() as sm:
            seg_texts = ["one", "two"]
            async with sm() as s:
                fp = lines_fingerprint(seg_texts)
                await TranslationLayerRepository(s).upsert_layer(
                    VIDEO, fp, "ko",
                    lines=[{"text": "one", "translation": "현재캡션"}],
                    attribution=None, origin="caption",
                )
                # 다른 지문에 더 "그럴듯해 보이는" 위키가 있어도 무시돼야 한다
                await TranslationLayerRepository(s).upsert_layer(
                    VIDEO, "cafebabe" * 4, "ko",
                    lines=[{"text": "one", "translation": "다른지문위키"}],
                    attribution=None, origin="wiki",
                )
                await SyncRepository(s).create(
                    video_id=VIDEO, lyrics_hash="h1",
                    timestamps=[{"text": t, "translation": ""} for t in seg_texts],
                )
                await s.commit()

            resp = await get_sync(VIDEO, lang="ko")
            assert resp.timestamps[0]["translation"] == "현재캡션"

    asyncio.run(body())


def test_translations_by_lang_prefers_human_cross_fingerprint_over_exact_llm():
    """translations_by_lang도 같은 우선순위를 따른다 — 읽기 전용(저장은 안 함)."""

    async def body():
        async with _env() as sm:
            old_seg_texts = ["one", "two"]
            new_seg_texts = ["onetwo"]

            async with sm() as s:
                new_fp = lines_fingerprint(new_seg_texts)
                await TranslationLayerRepository(s).upsert_layer(
                    VIDEO, new_fp, "ko",
                    lines=[{"text": "onetwo", "translation": "머신하나둘"}],
                    attribution=None, origin="llm",
                )
                old_fp = lines_fingerprint(old_seg_texts)
                await TranslationLayerRepository(s).upsert_layer(
                    VIDEO, old_fp, "ko",
                    lines=[
                        {"text": "one", "translation": "하나"},
                        {"text": "two", "translation": "둘"},
                    ],
                    attribution=None, origin="wiki",
                )
                await SyncRepository(s).create(
                    video_id=VIDEO, lyrics_hash="h_new",
                    timestamps=[{"text": t, "translation": ""} for t in new_seg_texts],
                )
                await s.commit()

            resp = await get_sync(VIDEO)  # lang 미지정
            assert resp.translations_by_lang == {"ko": ["하나둘"]}

            # 읽기 전용 확인 — 새 지문 자리의 llm 레이어는 그대로 남아 있다(교체되지
            # 않는다). lang= 요청이 실제로 들어올 때만(_apply_translation_lang의 메인
            # 서빙 경로) 그 자리가 사람 origin으로 교체된다.
            async with sm() as s:
                layer = await TranslationLayerRepository(s).get_layer(VIDEO, new_fp, "ko")
                assert layer is not None
                assert layer.origin == "llm"
                assert layer.lines == [{"text": "onetwo", "translation": "머신하나둘"}]

    asyncio.run(body())
