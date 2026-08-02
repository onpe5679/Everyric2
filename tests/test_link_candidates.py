"""커버 온디맨드 자동 연결 — 제목 매칭·후보 선정·쿨다운·제목 백필 테스트.

기존 서버 테스트 규약을 그대로 따른다: 격리된 in-memory SQLite로 connection.async_session을
몽키패치하고 라우트 코루틴을 직접 await(asyncio.run). httpx/TestClient는 쓰지 않는다.

여기서 검증하는 계약의 핵심: **제목은 후보 발견에만 쓰이고, SyncLink는 절대 만들지 않는다.**
제목이 완벽히 일치해도 결과는 '검증 잡 제출'까지이며 링크 생성은 반주 상관 판정의 몫이다.
"""

import asyncio
import contextlib
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import BackgroundTasks, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from everyric2.config.settings import get_settings
from everyric2.server import title_match
from everyric2.server import worker as worker_core
from everyric2.server.api.sync import (
    GenerateRequest,
    SyncLinkRequest,
    create_sync_link,
    find_link_candidates,
    generate_sync,
    get_sync,
)
from everyric2.server.db import connection as db_conn
from everyric2.server.db.models import Base, LinkJob
from everyric2.server.db.repository import (
    LinkJobRepository,
    SyncLinkRepository,
    SyncRepository,
)

COVER = "COVERvideo1"
SOURCE = "SOURCEvid01"
OTHER = "OTHERvid001"

SOURCE_TITLE = "熱異常 / いよわ feat.初音ミク"
COVER_TITLE = "熱異常 歌ってみた【足立レイ】"


@contextlib.asynccontextmanager
async def _env(**server_overrides):
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
    worker_core._PENDING_TITLE.clear()
    try:
        yield sm
    finally:
        db_conn.async_session = orig
        for k, v in saved.items():
            object.__setattr__(server, k, v)
        worker_core._PENDING_TITLE.clear()
        await engine.dispose()


async def _seed_sync(sm, video_id, title=None, artist=None, lyrics_hash="h1"):
    async with sm() as s:
        row = await SyncRepository(s).create(
            video_id=video_id,
            lyrics_hash=lyrics_hash,
            timestamps=[{"text": "라인", "start": 1.0, "end": 2.0}],
            engine="ctc",
            title=title,
            artist=artist,
        )
        await s.commit()
        return row.id


async def _count_link_jobs(sm) -> int:
    async with sm() as s:
        return len((await s.execute(select(LinkJob))).scalars().all())


@pytest.fixture(autouse=True)
def _cached_pair_everywhere(monkeypatch):
    """기본값: 미디어 캐시 완비로 가정 — 캐시 게이트(link_require_cached_pair)가 기본 켜짐이라
    기존 제출 계열 테스트가 게이트에 걸리지 않게 한다. 게이트 자체를 검증하는 테스트만
    개별적으로 미스를 흉내 낸다. 가사 지문 메모리도 테스트 간 격리한다."""
    from everyric2.server.api import sync as sync_api

    monkeypatch.setattr(sync_api.media_cache, "lookup_cached", lambda _vid: True)
    sync_api._RECENT_LYRICS_HASH.clear()
    yield
    sync_api._RECENT_LYRICS_HASH.clear()


# ── title_match: 정규화·잡토큰 제거 ───────────────────────────────


def test_normalize_title_strips_spaces_and_symbols():
    assert title_match.normalize_title("Roki ロキ!") == "rokiロキ"


def test_strip_noise_tokens_removes_upload_conventions():
    for raw, kept in [
        ("熱異常 Official MV", "熱異常"),
        ("熱異常 歌ってみた", "熱異常"),
        ("熱異常 (Instrumental)", "熱異常"),
        ("熱異常 off vocal", "熱異常"),
        ("熱異常 커버", "熱異常"),
        ("Roki Music Video", "roki"),
    ]:
        assert title_match.normalize_title(title_match.strip_noise_tokens(raw)) == kept


def test_candidate_queries_keeps_index_behaviour_without_noise_drop():
    # drop_noise 기본값 False는 곡 인덱스 매칭의 기존 동작을 그대로 보존한다
    plain = title_match.candidate_queries("熱異常 Official MV")
    assert "熱異常" not in plain
    with_noise_drop = title_match.candidate_queries("熱異常 Official MV", drop_noise=True)
    assert "熱異常" in with_noise_drop


# ── title_match: 커버 표기 확장(실측, 2026-07) ────────────────────
#
# 사용자 제보(«커버를 이을 때 한국어나 영어 cover인 걸 잘 인식 못해»)를 실제 유튜브
# 커버 제목 표본(oEmbed 조회 가능한 공개 영상)으로 재현·확정했다. 두 표에 걸쳐 있다:
# (1) カバー(가타카나)·불러보/불러봤 활용형이 잡토큰 목록에 없어 축약되지 않았다.
# (2) 한국어 커버 제목이 세로줄(｜) 대신 흔히 쓰는 ㅣ(U+3163, 한글 자모 "이")가 구분자
#     집합에 없어, 곡명이 뒤따르는 장식과 통째로 붙어 다른 후보와 매칭이 안 됐다.


def test_strip_noise_tokens_removes_katakana_and_korean_cover_conjugations():
    for raw, kept in [
        ("熱異常 カバー", "熱異常"),
        ("熱異常 불러보았다", "熱異常"),  # 문어체 과거(비축약)
        ("熱異常 불러봤음", "熱異常"),  # 구어체 과거+명사형(축약)
        ("熱異常 불러봄", "熱異常"),  # 명사형
        ("熱異常 불러본", "熱異常"),  # 관형형
        ("熱異常 불러보는", "熱異常"),  # 현재 관형형
        ("熱異常 불러봤다", "熱異常"),  # 기존 항목 — 회귀 확인
    ]:
        assert title_match.normalize_title(title_match.strip_noise_tokens(raw)) == kept


def test_ㅣ_separator_isolates_the_song_name_from_trailing_decoration():
    # 실측: 「【MV】 로키 (ROKI) ㅣ한국어 Coverㅣ【레볼루션 하트】」— ㅣ가 구분자가 아니면
    # "로키"가 뒤 장식과 통째로 붙어 다른 후보와 절대 매칭되지 않는다(길이비가 항상 낮다).
    candidates = title_match.candidate_queries(
        "【MV】 로키 (ROKI) ㅣ한국어 Coverㅣ【레볼루션 하트】", drop_noise=True
    )
    assert "로키" in candidates


def test_two_real_korean_cover_titles_of_the_same_song_now_match():
    """실측 재현 — 사고 그 자체: 실제 한국어 커버 제목 두 개(서로 다른 업로더, 같은 곡)가
    수정 전에는 서로 매칭되지 않았다(None). ㅣ 구분자 인식이 원인이었다."""
    a = "【MV】 로키 (ROKI) ㅣ한국어 Coverㅣ【레볼루션 하트】"
    b = "【Han＆Guriri】로키/ROKI 한국어 커버 (ロキ Korean cover)"
    score, _ = title_match.match_score(a, b)
    assert score == 1.0


@pytest.mark.parametrize(
    "cover_title",
    [
        "ロキ/白上フブキ&影山シエン(Cover)",
        "ロキ / 輪堂千速 ( cover )",
        "【下田麻美】ロキ／みきとP《歌ってみた》",
        "ロキ 歌ってみた / ナナヲアカリ",
        "【歌ってみた】ロキ / みきとP【天神子兎音cover】",
    ],
)
def test_real_japanese_cover_titles_match_the_original(cover_title):
    # 실제 「ロキ」 커버 업로드 제목 표본(가타카나 cover·괄호 cover·歌ってみた 조합)이
    # 위키 원제와 정확히 일치한다(score=1.0) — 회귀 없이 여전히 잘 잡힌다.
    score, _ = title_match.match_score(cover_title, "ロキ (Roki)")
    assert score == 1.0


# ── title_match: 매칭 점수 ────────────────────────────────────────


def test_match_score_exact_after_noise_strip():
    score, _ = title_match.match_score("熱異常 Official MV", "熱異常")
    assert score == 1.0


def test_match_score_matches_cover_against_original_full_title():
    score, _ = title_match.match_score(COVER_TITLE, SOURCE_TITLE)
    assert score == 1.0


def test_match_score_unrelated_titles_return_none():
    assert title_match.match_score("熱異常", "全然違う曲名です") is None


def test_match_score_artist_fragment_overlap_stays_below_default_threshold():
    """«【初音ミク】곡명» 류에서 가수명 조각이 다른 곡 제목에 포함되며 생기는 오탐 —
    점수가 기본 하한(0.6) 아래라 후보로 승격되지 않는다."""
    hit = title_match.match_score("【初音ミク】熱異常", "初音ミクの消失")
    assert hit is not None
    assert hit[0] < 0.6


def test_rank_matches_orders_exact_before_partial_and_honours_min_score():
    entries = [("v_partial", "初音ミクの消失"), ("v_exact", "熱異常")]
    ranked = title_match.rank_matches("【初音ミク】熱異常", entries, min_score=0.5)
    assert ranked[0][0] == "v_exact"
    assert ranked[0][1] == 1.0
    # 하한을 올리면 부분 겹침 후보가 탈락한다
    tight = title_match.rank_matches("【初音ミク】熱異常", entries, min_score=0.6)
    assert [key for key, _ in tight] == ["v_exact"]


def test_strip_noise_tokens_normalizes_decorative_cover_font():
    """𝖢𝖮𝖵𝖤𝖱(수학 산세리프)는 NFKC 후에야 cover다 — 정규화 전에 잡토큰을 제거하던 순서
    탓에 살아남아 헛 후보를 만들었다 (실측 unite 2026-07-29)."""
    assert title_match.normalize_title(title_match.strip_noise_tokens("로키 𝖢𝖮𝖵𝖤𝖱")) == "로키"


def test_candidate_queries_drops_pure_noise_fragments_only_in_drop_noise_mode():
    """«[MV] A곡»과 «[MV] B곡»이 mv==mv로 만점 매칭되던 경로 차단 — 잡토큰만으로 이루어진
    조각은 후보에서 뺀다. 인덱스 경로(drop_noise=False)의 기존 동작은 그대로 보존한다."""
    link_mode = title_match.candidate_queries("[MV] 熱異常", drop_noise=True)
    assert "mv" not in link_mode
    assert "熱異常" in link_mode
    index_mode = title_match.candidate_queries("[MV] 熱異常", drop_noise=False)
    assert "mv" in index_mode


def test_rank_matches_df_suppresses_shared_producer_and_singer_fragments():
    """실측 오탐(unite 2026-07-29): «ARTIST - 곡명» 관례에서 아티스트 조각이 제목 앞머리라
    priority까지 낮아 만점 오탐이 정탐을 이겼다. 코퍼스 여러 곡이 공유하는 조각(프로듀서명·
    가수명)은 문서빈도 억제가 매칭에서 뺀다 — 진짜 커버쌍(공유 조각 없음)은 그대로 산다."""
    entries = [
        ("v1", "DECO*27 - ルーキー feat. 初音ミク"),
        ("v2", "DECO*27 - モニタリング feat. 初音ミク"),
        ("v3", "DECO*27 - ラビットホール feat. 初音ミク"),
        ("v4", "DECO*27 - ヴァンパイア feat. 初音ミク"),
        ("v5", "DECO*27 - サラマンダー feat. 初音ミク"),
        ("v6", "Orangestar - DAYBREAK FRONTLINE (feat. IA)"),
    ]
    wrong = title_match.rank_matches("DECO*27 - 勘違い性反希望症 feat. 初音ミク", entries)
    assert wrong == []
    right = title_match.rank_matches("DAYBREAK FRONTLINE / 텐코 시부키 cover", entries)
    assert [k for k, _ in right] == ["v6"]
    # 억제를 끄면 예전 오탐이 재현된다 — 이 테스트가 지키는 것이 무엇인지의 대조군
    legacy = title_match.rank_matches(
        "DECO*27 - 勘違い性反希望症 feat. 初音ミク", entries, max_fragment_df=None
    )
    assert legacy and legacy[0][1] == 1.0


# ── 후보 탐색 엔드포인트 ──────────────────────────────────────────


def test_candidates_submits_link_job_for_top_match():
    async def body():
        async with _env() as sm:
            await _seed_sync(sm, SOURCE, title=SOURCE_TITLE)
            resp = await find_link_candidates(COVER, title=COVER_TITLE)
            assert resp.status == "submitted"
            assert [c.video_id for c in resp.candidates] == [SOURCE]
            assert resp.candidates[0].score == 1.0
            assert resp.followup == "link_validate"
            assert resp.job_id
            async with sm() as s:
                lj = await LinkJobRepository(s).get_by_id(resp.job_id)
                assert lj.video_id == COVER and lj.source_video_id == SOURCE
                assert lj.status == "queued"
                # 제목이 완벽히 일치해도 링크는 만들어지지 않는다 — 판정은 반주 상관의 몫
                assert await SyncLinkRepository(s).get(COVER) is None

    asyncio.run(body())


def test_candidates_excludes_self_and_unrelated_titles():
    async def body():
        async with _env() as sm:
            # 자기 자신이 코퍼스에 제목만 있는 다른 lyrics_hash로 들어 있어도 후보가 되면 안 된다
            await _seed_sync(sm, OTHER, title="全然違う曲名です")
            resp = await find_link_candidates(COVER, title=COVER_TITLE)
            assert resp.status == "none"
            assert resp.candidates == []
            assert await _count_link_jobs(sm) == 0

    asyncio.run(body())


def test_candidates_skips_when_video_has_own_sync():
    async def body():
        async with _env() as sm:
            await _seed_sync(sm, SOURCE, title=SOURCE_TITLE)
            await _seed_sync(sm, COVER, title=None)
            resp = await find_link_candidates(COVER, title=COVER_TITLE, artist="足立レイ")
            assert resp.status == "has_sync"
            assert resp.candidates == []
            assert await _count_link_jobs(sm) == 0
            # 자기 싱크가 있는 경우 이 호출은 제목 백필로 쓰인다
            async with sm() as s:
                rows = await SyncRepository(s).get_by_video(COVER)
                assert rows[0].title == COVER_TITLE
                assert rows[0].artist == "足立レイ"

    asyncio.run(body())


def test_candidates_skips_when_link_already_exists():
    async def body():
        async with _env() as sm:
            await _seed_sync(sm, SOURCE, title=SOURCE_TITLE)
            async with sm() as s:
                await SyncLinkRepository(s).upsert(COVER, SOURCE, 1.0, verified=True)
                await s.commit()
            resp = await find_link_candidates(COVER, title=COVER_TITLE)
            assert resp.status == "linked"
            assert await _count_link_jobs(sm) == 0

    asyncio.run(body())


def test_candidates_disabled_returns_candidates_without_job():
    async def body():
        async with _env(auto_link_candidates=False) as sm:
            await _seed_sync(sm, SOURCE, title=SOURCE_TITLE)
            resp = await find_link_candidates(COVER, title=COVER_TITLE)
            assert resp.status == "disabled"
            assert [c.video_id for c in resp.candidates] == [SOURCE]
            assert resp.job_id is None
            assert resp.followup is None  # 아무 후속 작업도 내지 않았다
            assert await _count_link_jobs(sm) == 0

    asyncio.run(body())


def test_candidates_rejects_malformed_video_id():
    async def body():
        async with _env():
            with pytest.raises(HTTPException) as e:
                await find_link_candidates("too-short", title=COVER_TITLE)
            assert e.value.status_code == 422

    asyncio.run(body())


# ── 재제출 억제: 진행 중 병합 + 완료/실패 쿨다운 ──────────────────


def test_candidates_merges_into_active_job():
    async def body():
        async with _env() as sm:
            await _seed_sync(sm, SOURCE, title=SOURCE_TITLE)
            first = await find_link_candidates(COVER, title=COVER_TITLE)
            second = await find_link_candidates(COVER, title=COVER_TITLE)
            assert second.status == "pending"
            assert second.job_id == first.job_id
            assert await _count_link_jobs(sm) == 1

    asyncio.run(body())


@pytest.mark.parametrize("finished_status", ["done", "failed", "declined"])
def test_candidates_respect_cooldown_after_finished_attempt(finished_status):
    async def body():
        async with _env(link_retry_cooldown_days=14) as sm:
            await _seed_sync(sm, SOURCE, title=SOURCE_TITLE)
            async with sm() as s:
                lj = await LinkJobRepository(s).create(COVER, SOURCE)
                lj.status = finished_status
                await s.commit()
                finished_id = lj.id

            resp = await find_link_candidates(COVER, title=COVER_TITLE)
            assert resp.status == "cooldown"
            assert resp.job_id == finished_id
            # 새 잡을 만들지 않는다 — 같은 영상을 반복해 열어도 GPU가 다시 돌지 않는다
            assert await _count_link_jobs(sm) == 1

    asyncio.run(body())


def test_candidates_resubmit_after_cooldown_expires():
    async def body():
        async with _env(link_retry_cooldown_days=14) as sm:
            await _seed_sync(sm, SOURCE, title=SOURCE_TITLE)
            stale = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=30)
            async with sm() as s:
                s.add(
                    LinkJob(
                        video_id=COVER,
                        source_video_id=SOURCE,
                        status="done",
                        match=False,
                        created_at=stale,
                    )
                )
                await s.commit()

            resp = await find_link_candidates(COVER, title=COVER_TITLE)
            assert resp.status == "submitted"
            assert await _count_link_jobs(sm) == 2

    asyncio.run(body())


def test_cooldown_disabled_when_days_zero():
    async def body():
        async with _env(link_retry_cooldown_days=0) as sm:
            await _seed_sync(sm, SOURCE, title=SOURCE_TITLE)
            async with sm() as s:
                lj = await LinkJobRepository(s).create(COVER, SOURCE)
                lj.status = "done"
                await s.commit()

            resp = await find_link_candidates(COVER, title=COVER_TITLE)
            assert resp.status == "submitted"
            assert await _count_link_jobs(sm) == 2

    asyncio.run(body())


def test_cooldown_only_applies_to_the_same_pair():
    async def body():
        async with _env(link_retry_cooldown_days=14) as sm:
            await _seed_sync(sm, SOURCE, title=SOURCE_TITLE)
            async with sm() as s:
                lj = await LinkJobRepository(s).create(COVER, OTHER)
                lj.status = "done"
                await s.commit()

            resp = await find_link_candidates(COVER, title=COVER_TITLE)
            assert resp.status == "submitted"

    asyncio.run(body())


# ── 제목 백필 ─────────────────────────────────────────────────────


def test_get_sync_backfills_missing_title():
    async def body():
        async with _env() as sm:
            await _seed_sync(sm, SOURCE, title=None)
            resp = await get_sync(SOURCE, title=SOURCE_TITLE, artist="いよわ")
            assert resp.found is True
            async with sm() as s:
                row = (await SyncRepository(s).get_by_video(SOURCE))[0]
                assert row.title == SOURCE_TITLE
                assert row.artist == "いよわ"

    asyncio.run(body())


def test_get_sync_never_overwrites_existing_title():
    async def body():
        async with _env() as sm:
            await _seed_sync(sm, SOURCE, title="원래 제목", artist="원래 아티스트")
            await get_sync(SOURCE, title="다른 제목", artist="다른 아티스트")
            async with sm() as s:
                row = (await SyncRepository(s).get_by_video(SOURCE))[0]
                assert row.title == "원래 제목"
                assert row.artist == "원래 아티스트"

    asyncio.run(body())


def test_get_sync_backfill_targets_hash_matched_row():
    async def body():
        async with _env() as sm:
            await _seed_sync(sm, SOURCE, title=None, lyrics_hash="hh")
            await get_sync(SOURCE, lyrics_hash="hh", title=SOURCE_TITLE)
            async with sm() as s:
                row = (await SyncRepository(s).get_by_video(SOURCE))[0]
                assert row.title == SOURCE_TITLE

    asyncio.run(body())


def test_get_sync_does_not_stamp_cover_title_onto_linked_source():
    async def body():
        async with _env() as sm:
            await _seed_sync(sm, SOURCE, title=None)
            async with sm() as s:
                await SyncLinkRepository(s).upsert(COVER, SOURCE, 2.0, verified=True)
                await s.commit()
            resp = await get_sync(COVER, title=COVER_TITLE)
            assert resp.linked["source_video_id"] == SOURCE
            async with sm() as s:
                row = (await SyncRepository(s).get_by_video(SOURCE))[0]
                assert row.title is None  # 커버 제목이 원곡 행에 새겨지면 안 된다

    asyncio.run(body())


def test_generate_backfills_title_on_existing_sync():
    async def body():
        async with _env() as sm:
            from everyric2.server.db.repository import hash_lyrics

            lyrics = "가사 한 줄"
            await _seed_sync(sm, SOURCE, title=None, lyrics_hash=hash_lyrics(lyrics))
            resp = await generate_sync(
                GenerateRequest(
                    video_id=SOURCE, lyrics=lyrics, title=SOURCE_TITLE, artist="いよわ"
                ),
                BackgroundTasks(),
            )
            assert resp.status == "completed"
            async with sm() as s:
                row = (await SyncRepository(s).get_by_video(SOURCE))[0]
                assert row.title == SOURCE_TITLE

    asyncio.run(body())


def test_generate_stashes_title_for_new_job():
    async def body():
        async with _env(local_worker=False) as sm:
            resp = await generate_sync(
                GenerateRequest(
                    video_id=COVER, lyrics="새 가사", title=COVER_TITLE, artist="足立レイ"
                ),
                BackgroundTasks(),
            )
            assert resp.status == "processing"
            assert worker_core.peek_title(resp.job_id) == (COVER_TITLE, "足立レイ")
            _ = sm

    asyncio.run(body())


# ── 수동 링크 안전장치 ────────────────────────────────────────────


def test_manual_link_is_recorded_unverified():
    async def body():
        async with _env() as sm:
            await _seed_sync(sm, SOURCE, title=SOURCE_TITLE)
            link = await create_sync_link(
                SyncLinkRequest(video_id=COVER, source_video_id=SOURCE, offset_sec=0.0)
            )
            assert link.verified is False
            resp = await get_sync(COVER)
            assert resp.linked["verified"] is False

    asyncio.run(body())


def test_manual_link_admin_gate_rejects_without_key():
    async def body():
        async with _env(manual_link_requires_admin=True, admin_api_key="adminkey") as sm:
            await _seed_sync(sm, SOURCE, title=SOURCE_TITLE)
            with pytest.raises(HTTPException) as e:
                await create_sync_link(
                    SyncLinkRequest(video_id=COVER, source_video_id=SOURCE, offset_sec=0.0)
                )
            assert e.value.status_code == 403
            # 어드민 키를 제시하면 통과한다
            link = await create_sync_link(
                SyncLinkRequest(video_id=COVER, source_video_id=SOURCE, offset_sec=0.0),
                x_api_key="adminkey",
            )
            assert link.verified is False

    asyncio.run(body())


def test_manual_link_gate_off_by_default():
    async def body():
        async with _env() as sm:
            await _seed_sync(sm, SOURCE, title=SOURCE_TITLE)
            link = await create_sync_link(
                SyncLinkRequest(video_id=COVER, source_video_id=SOURCE, offset_sec=0.0)
            )
            assert link.source_video_id == SOURCE

    asyncio.run(body())


def test_synclink_upsert_persists_rate_on_insert():
    """신규 삽입 시 rate가 누락돼 배속 링크가 1.0으로 저장되던 회귀 방지."""

    async def body():
        async with _env() as sm:
            async with sm() as s:
                link = await SyncLinkRepository(s).upsert(COVER, SOURCE, 1.5, rate=1.25)
                assert link.rate == 1.25
                await s.commit()
            async with sm() as s:
                assert (await SyncLinkRepository(s).get(COVER)).rate == 1.25

    asyncio.run(body())


# ── 무다운로드 재설계 (unite 요청 2026-07-29) ─────────────────────


def test_fingerprint_candidate_outranks_title_matching():
    """② 가사 지문 — 직전 GET /api/sync가 남긴 lyrics_hash와 같은 지문의 싱크를 가진 다른
    영상은 제목이 전혀 달라도 최상위 후보가 된다 (다운로드 0, 확장 요청 모양 불변)."""

    async def body():
        async with _env(auto_link_candidates=False) as sm:
            await _seed_sync(sm, SOURCE, title="全然違う表記の原曲", lyrics_hash="FPHASH")
            with contextlib.suppress(HTTPException):
                await get_sync(COVER, lyrics_hash="FPHASH")
            resp = await find_link_candidates(COVER, title="제목으로는 절대 못 맞출 문자열")
            assert resp.status == "disabled"
            assert resp.candidates[0].video_id == SOURCE
            assert resp.candidates[0].score == 1.0

    asyncio.run(body())


def test_fingerprint_recall_expires_after_ttl():
    """TTL이 지난 지문 기억은 후보를 만들지 않는다 — 오래전 다른 가사로 조회한 흔적이
    엉뚱한 원곡을 물어 오면 안 된다."""

    async def body():
        async with _env(auto_link_candidates=False) as sm:
            await _seed_sync(sm, SOURCE, title=None, lyrics_hash="FPHASH")
            from everyric2.server.api import sync as sync_api

            sync_api._RECENT_LYRICS_HASH[COVER] = ("FPHASH", 0.0)  # 아주 오래된 기억
            resp = await find_link_candidates(COVER, title="무관한 제목")
            assert resp.candidates == []

    asyncio.run(body())


def test_relation_candidate_requires_an_existing_sync(monkeypatch):
    """①(songlink/1) — 관계가 가리키는 원곡에 싱크가 있어야만 후보다(빌릴 것이 없으면
    링크가 무의미). 관계는 판정이 아니라 후보라 score는 응답 confidence를 그대로 쓴다."""

    async def body():
        async with _env(auto_link_candidates=False) as sm:
            from everyric2.server.api import sync as sync_api

            monkeypatch.setattr(
                sync_api.song_link,
                "lookup_original",
                lambda _vid: {
                    "found": True,
                    "original": {"platform": "youtube", "id": SOURCE},
                    "confidence": 0.74,
                },
            )
            resp = await find_link_candidates(COVER, title="무관한 제목")
            assert resp.candidates == []  # 원곡 싱크 없음 → 후보 아님
            await _seed_sync(sm, SOURCE, title="아무 표기")
            resp2 = await find_link_candidates(COVER, title="무관한 제목")
            assert resp2.candidates[0].video_id == SOURCE
            assert resp2.candidates[0].score == 0.74

    asyncio.run(body())


def test_fingerprint_outranks_relation_candidate(monkeypatch):
    """우선순위 — 가사 지문(문자열 일치)이 자동 파생 관계보다 앞선다."""

    async def body():
        async with _env(auto_link_candidates=False) as sm:
            from everyric2.server.api import sync as sync_api

            await _seed_sync(sm, SOURCE, title=None, lyrics_hash="FPHASH")
            await _seed_sync(sm, OTHER, title=None, lyrics_hash="unrelated")
            monkeypatch.setattr(
                sync_api.song_link,
                "lookup_original",
                lambda _vid: {
                    "found": True,
                    "original": {"platform": "youtube", "id": OTHER},
                    "confidence": 1.0,
                },
            )
            with contextlib.suppress(HTTPException):
                await get_sync(COVER, lyrics_hash="FPHASH")
            resp = await find_link_candidates(COVER, title="무관한 제목")
            assert [c.video_id for c in resp.candidates][:2] == [SOURCE, OTHER]

    asyncio.run(body())


def test_cache_gate_blocks_submission_without_cached_pair(monkeypatch):
    """④ 캐시 쌍 게이트 — 한쪽이라도 미디어 캐시에 없으면 잡을 만들지 않는다(후보만 반환).
    실측: 실사용자 영상 캐시 적중 11% — 이 게이트가 자동 제출의 89%가 다운로드로 이어지던
    경로를 끊는다. status는 확장이 이미 조용히 처리하는 none을 쓴다."""

    async def body():
        async with _env() as sm:
            await _seed_sync(sm, SOURCE, title=SOURCE_TITLE)
            from everyric2.server.api import sync as sync_api

            monkeypatch.setattr(sync_api.media_cache, "lookup_cached", lambda _vid: False)
            resp = await find_link_candidates(COVER, title=COVER_TITLE)
            assert resp.status == "none"
            assert [c.video_id for c in resp.candidates] == [SOURCE]
            assert await _count_link_jobs(sm) == 0

    asyncio.run(body())


def test_cache_gate_off_restores_legacy_submission(monkeypatch):
    async def body():
        async with _env(link_require_cached_pair=False) as sm:
            await _seed_sync(sm, SOURCE, title=SOURCE_TITLE)
            from everyric2.server.api import sync as sync_api

            # 게이트를 끄면 캐시 조회 자체를 하지 않는다
            monkeypatch.setattr(
                sync_api.media_cache,
                "lookup_cached",
                lambda _vid: (_ for _ in ()).throw(AssertionError("게이트 off인데 캐시 조회")),
            )
            resp = await find_link_candidates(COVER, title=COVER_TITLE)
            assert resp.status == "submitted"
            assert await _count_link_jobs(sm) == 1

    asyncio.run(body())
