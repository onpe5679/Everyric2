"""곡 인덱스 프록시(WS1-C) 테스트 — 외부 곡 인덱스로의 업스트림 분기만 검증한다.

네트워크를 타지 않도록 _upstream_get(동기 requests 호출)을 몽키패치하고 라우트 코루틴을
직접 await한다(asyncio.run). song_index_url 설정 유/무/오류 세 분기를 못 박는다.
"""

import asyncio

from fastapi import BackgroundTasks

from everyric2.config.settings import get_settings
from everyric2.server import vocaro_index as vi
from everyric2.server.api import vocaro as vocaro_api
from everyric2.server.api.vocaro import match_title, reindex, status
from everyric2.server.services.video_identity import VideoIdentity
from everyric2.server.vocaro_index import MatchDecision, SongEntry


def _matched(entry: SongEntry) -> MatchDecision:
    return MatchDecision(status="matched", reason="exact_title", entry=entry, candidate_count=1)


def _set_url(url: str) -> None:
    object.__setattr__(get_settings().server, "song_index_url", url)


# ── url 미설정: 기존 로컬 인덱스 경로 그대로 ──────────────────────


def test_match_uses_local_index_when_url_unset(monkeypatch):
    # song_index_url 기본값("")이면 로컬 match()로 슬러그를 답하고 page_url은 BASE_URL 기반
    monkeypatch.setattr(
        vocaro_api,
        "match_with_evidence",
        lambda title, **kwargs: _matched(SongEntry(slug="roki", ko="로키", ja="ロキ")),
    )
    resp = asyncio.run(match_title(BackgroundTasks(), title="ロキ"))
    assert resp.found is True
    assert resp.slug == "roki"
    assert resp.page_url == f"{vi.BASE_URL}/roki"


# ── url 설정: 업스트림 프록시 ─────────────────────────────────────


def test_match_proxies_upstream_and_maps_1to1(monkeypatch):
    _set_url("http://idx.test")
    captured = {}

    def fake_get(path, params=None):
        captured["path"] = path
        captured["params"] = params
        return {
            "found": True,
            "slug": "roki",
            "page_url": "http://idx.test/pages/roki",
            "ko": "로키",
            "ja": "ロキ",
        }

    monkeypatch.setattr(vocaro_api, "_upstream_get", fake_get)
    resp = asyncio.run(match_title(BackgroundTasks(), title="ロキ"))
    assert captured["path"] == "/match"
    assert captured["params"] == {"title": "ロキ"}
    assert resp.found is True
    assert resp.slug == "roki"
    # page_url은 업스트림 값을 그대로 쓴다 (로컬 BASE_URL로 재구성하지 않음)
    assert resp.page_url == "http://idx.test/pages/roki"
    assert resp.ko == "로키" and resp.ja == "ロキ"


def test_match_forwards_additive_identity_evidence(monkeypatch):
    _set_url("http://idx.test")
    captured = {}

    def fake_get(path, params=None):
        captured["params"] = params
        return {
            "found": True,
            "slug": "polaris",
            "ja": "POLARIS",
            "matcher_version": "identity-1",
        }

    monkeypatch.setattr(vocaro_api, "_upstream_get", fake_get)
    resp = asyncio.run(
        match_title(
            BackgroundTasks(),
            title="Patterns ft. @rino",
            candidate=["POLARIS"],
            raw_title="POLARIS - Patterns ft. @rino",
            artist="POLARIS",
            channel="Patterns",
            video_id="bEV_tH_yrIc",
        )
    )
    assert resp.found is True
    assert resp.slug == "polaris"
    assert captured["params"] == {
        "title": "Patterns ft. @rino",
        "candidate": ["POLARIS"],
        "raw_title": "POLARIS - Patterns ft. @rino",
        "artist": "POLARIS",
        "channel": "Patterns",
        "video_id": "bEV_tH_yrIc",
    }


def test_legacy_upstream_cannot_override_channel_backed_candidate(monkeypatch):
    _set_url("http://idx.test")
    monkeypatch.setattr(
        vocaro_api,
        "_upstream_get",
        lambda path, params=None: {"found": True, "slug": "patterns", "ja": "Patterns"},
    )
    local = SongEntry(slug="polaris", ko="POLARIS", ja="POLARIS")
    monkeypatch.setattr(
        vocaro_api,
        "match_with_evidence",
        lambda title, **kwargs: _matched(local),
    )
    resp = asyncio.run(
        match_title(
            BackgroundTasks(),
            title="Patterns ft. @rino",
            candidate=["POLARIS"],
            raw_title="POLARIS - Patterns ft. @rino",
            artist="POLARIS",
            channel="Patterns",
        )
    )
    assert resp.found is True
    assert resp.slug == "polaris"
    assert resp.matcher_version == "identity-1"


def test_legacy_upstream_cannot_pick_one_local_homonym(monkeypatch):
    _set_url("http://idx.test")
    monkeypatch.setattr(
        vocaro_api,
        "_upstream_get",
        lambda path, params=None: {
            "found": True,
            "slug": "scream-umetora",
            "ja": "Scream",
        },
    )
    monkeypatch.setattr(
        vocaro_api,
        "match_with_evidence",
        lambda title, **kwargs: MatchDecision(
            status="ambiguous",
            reason="duplicate_title",
            candidate_count=2,
            matched_query="Scream",
        ),
    )
    resp = asyncio.run(
        match_title(
            BackgroundTasks(),
            title="Scream",
            artist="Naoki",
            channel="Naoki",
        )
    )
    assert resp.found is False
    assert resp.status == "ambiguous"
    assert resp.matcher_version == "identity-1"


def test_legacy_upstream_cannot_be_relabelled_when_local_has_no_entry(monkeypatch):
    _set_url("http://idx.test")
    monkeypatch.setattr(
        vocaro_api,
        "_upstream_get",
        lambda path, params=None: {
            "found": True,
            "slug": "new-song-wrong-producer",
            "ja": "New Song",
        },
    )
    monkeypatch.setattr(
        vocaro_api,
        "match_with_evidence",
        lambda title, **kwargs: MatchDecision(status="not_found", reason="no_exact_title"),
    )
    resp = asyncio.run(match_title(BackgroundTasks(), title="New Song", artist="Right Producer"))
    assert resp.found is False
    assert resp.status == "not_found"


def test_loose_legacy_upstream_hit_is_rejected(monkeypatch):
    _set_url("http://idx.test")
    monkeypatch.setattr(
        vocaro_api,
        "_upstream_get",
        lambda path, params=None: {"found": True, "slug": "scream", "ja": "SCREAM"},
    )
    monkeypatch.setattr(
        vocaro_api,
        "match_with_evidence",
        lambda title, **kwargs: MatchDecision(status="not_found", reason="no_exact_title"),
    )
    resp = asyncio.run(match_title(BackgroundTasks(), title="S.C.R.E.A.M"))
    assert resp.found is False
    assert resp.status == "not_found"
    assert resp.matcher_version == "identity-1"


def test_upstream_found_without_slug_cannot_escape_response_invariant(monkeypatch):
    _set_url("http://idx.test")
    monkeypatch.setattr(
        vocaro_api,
        "_upstream_get",
        lambda path, params=None: {"found": True, "ja": "ロキ"},
    )
    monkeypatch.setattr(
        vocaro_api,
        "match_with_evidence",
        lambda title, **kwargs: MatchDecision(status="not_found", reason="no_exact_title"),
    )
    resp = asyncio.run(match_title(BackgroundTasks(), title="ロキ"))
    assert resp.found is False
    assert resp.slug is None


def test_upstream_found_with_nonmatched_status_is_rejected(monkeypatch):
    _set_url("http://idx.test")
    monkeypatch.setattr(
        vocaro_api,
        "_upstream_get",
        lambda path, params=None: {
            "found": True,
            "status": "ambiguous",
            "slug": "roki",
            "ja": "ロキ",
        },
    )
    monkeypatch.setattr(
        vocaro_api,
        "match_with_evidence",
        lambda title, **kwargs: MatchDecision(status="ambiguous", reason="duplicate_title"),
    )
    resp = asyncio.run(match_title(BackgroundTasks(), title="ロキ"))
    assert resp.found is False
    assert resp.status == "ambiguous"


def test_non_object_upstream_payload_falls_back_without_500(monkeypatch):
    _set_url("http://idx.test")
    monkeypatch.setattr(vocaro_api, "_upstream_get", lambda path, params=None: ["bad"])
    monkeypatch.setattr(
        vocaro_api,
        "match_with_evidence",
        lambda title, **kwargs: MatchDecision(status="not_found", reason="no_exact_title"),
    )
    resp = asyncio.run(match_title(BackgroundTasks(), title="없는곡"))
    assert resp.found is False
    assert resp.status == "not_found"


def test_malformed_optional_upstream_fields_are_sanitized(monkeypatch):
    _set_url("http://idx.test")
    monkeypatch.setattr(
        vocaro_api,
        "_upstream_get",
        lambda path, params=None: {
            "found": True,
            "slug": "roki",
            "ja": "ロキ",
            "ko": ["bad"],
            "candidate_count": "not-a-number",
        },
    )
    resp = asyncio.run(match_title(BackgroundTasks(), title="ロキ"))
    assert resp.found is True
    assert resp.ko is None
    assert resp.candidate_count == 1


def test_video_id_recovers_identity_after_client_title_miss(monkeypatch):
    _set_url("")
    recovered = SongEntry(slug="polaris", ko="POLARIS", ja="POLARIS")

    def fake_match(title, **kwargs):
        if (kwargs.get("title_candidates") or [None])[0] == "POLARIS":
            return _matched(recovered)
        return MatchDecision(status="not_found", reason="no_exact_title")

    monkeypatch.setattr(vocaro_api, "match_with_evidence", fake_match)
    monkeypatch.setattr(
        vocaro_api,
        "fetch_video_identity",
        lambda video_id: VideoIdentity(
            title="POLARIS - Patterns ft. @rino",
            channel="Patterns",
        ),
    )
    resp = asyncio.run(
        match_title(
            BackgroundTasks(),
            title="Patterns ft. @rino",
            video_id="bEV_tH_yrIc",
        )
    )
    assert resp.found is True
    assert resp.slug == "polaris"
    assert resp.evidence_source == "youtube_oembed"
    assert resp.reason == "video_id_exact_title"
    assert resp.resolved_title == "POLARIS - Patterns ft. @rino"
    assert resp.resolved_channel == "Patterns"


def test_video_id_confirms_client_hit_when_raw_title_is_missing(monkeypatch):
    _set_url("")
    monkeypatch.setattr(
        vocaro_api,
        "match_with_evidence",
        lambda title, **kwargs: _matched(SongEntry(slug="roki", ko="로키", ja="ロキ")),
    )
    monkeypatch.setattr(
        vocaro_api,
        "fetch_video_identity",
        lambda video_id: VideoIdentity(title="ロキ", channel="みきとP"),
    )
    resp = asyncio.run(
        match_title(BackgroundTasks(), title="ロキ", video_id="bEV_tH_yrIc")
    )
    assert resp.found is True
    assert resp.evidence_source == "youtube_oembed"


def test_video_id_lookup_is_skipped_when_raw_parts_corroborate_exact_hit(monkeypatch):
    _set_url("")
    monkeypatch.setattr(
        vocaro_api,
        "match_with_evidence",
        lambda title, **kwargs: _matched(
            SongEntry(slug="blueming", ko="블루밍", ja="Blueming")
        ),
    )
    monkeypatch.setattr(
        vocaro_api,
        "fetch_video_identity",
        lambda video_id: (_ for _ in ()).throw(AssertionError("corroborated raw title")),
    )
    resp = asyncio.run(
        match_title(
            BackgroundTasks(),
            title="Blueming",
            raw_title="IU - Blueming",
            artist="IU",
            channel="IU",
            video_id="bEV_tH_yrIc",
        )
    )
    assert resp.found is True
    assert resp.slug == "blueming"
    assert resp.evidence_source == "client"


def test_video_id_overrides_exact_hit_not_supported_by_raw_title(monkeypatch):
    _set_url("")
    wrong = SongEntry(slug="fate", ko="Fate", ja="Fate")
    recovered = SongEntry(slug="fate-type-l", ko="Fate (.Type.L)", ja="Fate (.Type.L)")

    def fake_match(title, **kwargs):
        if title == "Fate (.Type.L)":
            return _matched(recovered)
        return _matched(wrong)

    monkeypatch.setattr(vocaro_api, "match_with_evidence", fake_match)
    monkeypatch.setattr(
        vocaro_api,
        "fetch_video_identity",
        lambda video_id: VideoIdentity(title="Fate (.Type.L)", channel="Uploader"),
    )
    resp = asyncio.run(
        match_title(
            BackgroundTasks(),
            title="Fate",
            raw_title="Fate (.Type.L)",
            artist="Uploader",
            channel="Uploader",
            video_id="bEV_tH_yrIc",
        )
    )
    assert resp.found is True
    assert resp.slug == "fate-type-l"
    assert resp.evidence_source == "youtube_oembed"


def test_raw_fragment_match_is_promoted_only_after_video_id_confirmation(monkeypatch):
    _set_url("")
    recovered = SongEntry(slug="cyan-blue", ko="시안 블루", ja="シアンブルー")
    monkeypatch.setattr(
        vocaro_api,
        "match_with_evidence",
        lambda title, **kwargs: MatchDecision(
            status="matched",
            reason="raw_evidence_title",
            entry=recovered,
            candidate_count=1,
            matched_query="シアンブルー",
        ),
    )
    monkeypatch.setattr(
        vocaro_api,
        "fetch_video_identity",
        lambda video_id: VideoIdentity(
            title="シアンブルー / ポリスピカデリー feat. 初音ミク",
            channel="Hatsune Miku",
        ),
    )
    raw = "シアンブルー / ポリスピカデリー feat. 初音ミク"
    resp = asyncio.run(
        match_title(
            BackgroundTasks(),
            title=raw,
            candidate=[raw],
            raw_title=raw,
            artist="Hatsune Miku",
            channel="Hatsune Miku",
            video_id="H7PR6K7xff0",
        )
    )
    assert resp.found is True
    assert resp.slug == "cyan-blue"
    assert resp.evidence_source == "youtube_oembed"
    assert resp.reason == "video_id_raw_evidence_title"


def test_video_confirmed_upstream_match_keeps_oembed_evidence_source(monkeypatch):
    _set_url("http://idx.test")
    recovered = SongEntry(slug="cyan-blue", ko="시안 블루", ja="シアンブルー")
    monkeypatch.setattr(
        vocaro_api,
        "match_with_evidence",
        lambda title, **kwargs: MatchDecision(
            status="matched",
            reason="raw_evidence_title",
            entry=recovered,
            candidate_count=1,
            matched_query="シアンブルー",
        ),
    )
    monkeypatch.setattr(
        vocaro_api,
        "_upstream_get",
        lambda path, params=None: {
            "found": True,
            "slug": "cyan-blue",
            "ja": "シアンブルー",
            "matcher_version": "identity-1",
        },
    )
    monkeypatch.setattr(
        vocaro_api,
        "fetch_video_identity",
        lambda video_id: VideoIdentity(
            title="シアンブルー / ポリスピカデリー feat. 初音ミク",
            channel="Hatsune Miku",
        ),
    )
    raw = "シアンブルー / ポリスピカデリー feat. 初音ミク"
    resp = asyncio.run(
        match_title(
            BackgroundTasks(),
            title=raw,
            candidate=[raw],
            raw_title=raw,
            artist="Hatsune Miku",
            channel="Hatsune Miku",
            video_id="H7PR6K7xff0",
        )
    )
    assert resp.found is True
    assert resp.slug == "cyan-blue"
    assert resp.evidence_source == "youtube_oembed"
    assert resp.upstream_matcher_version == "identity-1"


def test_identity_upstream_cannot_override_different_oembed_title(monkeypatch):
    _set_url("http://idx.test")

    def fake_match(title, **kwargs):
        if title == "Fate":
            return _matched(SongEntry(slug="fate", ko="Fate", ja="Fate"))
        return MatchDecision(status="not_found", reason="no_exact_title")

    monkeypatch.setattr(vocaro_api, "match_with_evidence", fake_match)
    monkeypatch.setattr(
        vocaro_api,
        "_upstream_get",
        lambda path, params=None: {
            "found": True,
            "slug": "fate",
            "ja": "Fate",
            "matcher_version": "identity-1",
        },
    )
    monkeypatch.setattr(
        vocaro_api,
        "fetch_video_identity",
        lambda video_id: VideoIdentity(title="Completely Different Song", channel="Other"),
    )
    resp = asyncio.run(
        match_title(BackgroundTasks(), title="Fate", video_id="bEV_tH_yrIc")
    )
    assert resp.found is False
    assert resp.status == "not_found"
    assert resp.evidence_source == "youtube_oembed"
    assert resp.resolved_title == "Completely Different Song"


def test_video_id_queue_wait_is_inside_total_timeout(monkeypatch):
    monkeypatch.setattr(vocaro_api, "_VIDEO_ID_LOOKUP_SEMAPHORE", asyncio.Semaphore(0))
    monkeypatch.setattr(vocaro_api, "_VIDEO_ID_LOOKUP_TIMEOUT_SEC", 0.01)
    assert asyncio.run(vocaro_api._bounded_video_identity("bEV_tH_yrIc")) is None


def test_video_id_recovery_is_visible_even_when_wiki_has_no_entry(monkeypatch):
    _set_url("")
    monkeypatch.setattr(
        vocaro_api,
        "match_with_evidence",
        lambda title, **kwargs: MatchDecision(status="not_found", reason="no_exact_title"),
    )
    monkeypatch.setattr(
        vocaro_api,
        "fetch_video_identity",
        lambda video_id: VideoIdentity(
            title="POLARIS - Patterns ft. @rino",
            channel="Patterns",
        ),
    )
    resp = asyncio.run(
        match_title(
            BackgroundTasks(),
            title="Patterns ft. @rino",
            video_id="bEV_tH_yrIc",
        )
    )
    assert resp.found is False
    assert resp.evidence_source == "youtube_oembed"
    assert resp.resolved_title == "POLARIS - Patterns ft. @rino"
    assert resp.resolved_channel == "Patterns"


def test_manual_search_mode_restores_fuzzy_candidate_without_auto_adoption(monkeypatch):
    _set_url("")
    monkeypatch.setattr(
        vocaro_api,
        "match_with_evidence",
        lambda title, **kwargs: MatchDecision(status="not_found", reason="no_exact_title"),
    )
    monkeypatch.setattr(
        vocaro_api,
        "search_matches",
        lambda title, limit: [
            (SongEntry(slug="long-song", ko="긴 제목의 노래 입니다", ja=None), 0.8),
            (SongEntry(slug="other-song", ko="다른 노래 입니다", ja=None), 0.6),
        ],
    )
    resp = asyncio.run(
        match_title(BackgroundTasks(), title="노래 입니다", mode="search")
    )
    assert resp.found is True
    assert resp.slug == "long-song"
    assert resp.reason == "manual_search_fuzzy"
    assert resp.evidence_source == "manual_search"
    assert [candidate.slug for candidate in resp.candidates] == ["long-song", "other-song"]
    assert resp.candidate_count == 2


def test_manual_search_mode_exposes_ambiguous_producer_candidates(monkeypatch):
    _set_url("")
    monkeypatch.setattr(
        vocaro_api,
        "match_with_evidence",
        lambda title, **kwargs: MatchDecision(
            status="ambiguous",
            reason="duplicate_title",
            candidate_count=2,
            matched_query="CANVAS",
        ),
    )
    monkeypatch.setattr(
        vocaro_api,
        "search_matches",
        lambda title, limit: [
            (SongEntry(slug="canvas-divela", ko="캔버스/DIVELA", ja="CANVAS"), 1.0),
            (SongEntry(slug="canvas-other", ko="캔버스/Other", ja="CANVAS"), 1.0),
        ],
    )
    resp = asyncio.run(match_title(BackgroundTasks(), title="CANVAS", mode="search"))
    assert resp.found is True
    assert [candidate.producer for candidate in resp.candidates] == ["DIVELA", "Other"]


def test_manual_search_preserves_identity_upstream_candidate_list(monkeypatch):
    _set_url("http://idx.test")
    monkeypatch.setattr(
        vocaro_api,
        "match_with_evidence",
        lambda title, **kwargs: MatchDecision(status="not_found", reason="no_exact_title"),
    )
    monkeypatch.setattr(
        vocaro_api,
        "_upstream_get",
        lambda path, params=None: {
            "found": True,
            "slug": "canvas-divela",
            "ja": "CANVAS",
            "matcher_version": "identity-1",
            "candidates": [
                {
                    "slug": "canvas-divela",
                    "ja": "CANVAS",
                    "ko": "캔버스/DIVELA",
                    "score": 1.0,
                },
                {
                    "slug": "canvas-other",
                    "ja": "CANVAS",
                    "ko": "캔버스/Other",
                    "score": 0.9,
                },
            ],
        },
    )
    resp = asyncio.run(match_title(BackgroundTasks(), title="CANVAS", mode="search"))
    assert resp.found is True
    assert resp.candidate_count == 2
    assert [candidate.slug for candidate in resp.candidates] == [
        "canvas-divela",
        "canvas-other",
    ]
    assert [candidate.producer for candidate in resp.candidates] == ["DIVELA", "Other"]


def test_match_upstream_not_found_passthrough(monkeypatch):
    _set_url("http://idx.test")
    monkeypatch.setattr(vocaro_api, "_upstream_get", lambda path, params=None: {"found": False})
    resp = asyncio.run(match_title(BackgroundTasks(), title="없는곡"))
    assert resp.found is False
    assert resp.slug is None


def test_match_upstream_error_falls_back_to_local(monkeypatch):
    """업스트림 오류·미발견은 확정이 아니다 — 로컬 인덱스로 폴백한다(계약 변경 2026-07-29,
    «ホロウ» 무력화 실사고). 로컬도 미발견이면 status 없는 조용한 found=false다."""
    _set_url("http://idx.test")

    def boom(path, params=None):
        raise RuntimeError("timeout")

    monkeypatch.setattr(vocaro_api, "_upstream_get", boom)
    monkeypatch.setattr(
        vocaro_api,
        "match_with_evidence",
        lambda title, **kwargs: MatchDecision(status="not_found", reason="no_exact_title"),
    )
    monkeypatch.setattr(vocaro_api, "index_status", lambda: {"total": 6550})
    resp = asyncio.run(match_title(BackgroundTasks(), title="x"))
    assert resp.found is False
    assert resp.status == "not_found"
    assert resp.reason == "no_exact_title"


def test_reindex_upstream_mode_no_build_kick(monkeypatch):
    _set_url("http://idx.test")
    # 업스트림 모드에선 빌드를 킥하지 않고 status="upstream"만 알린다
    called = {"build": False}
    monkeypatch.setattr(vocaro_api, "build_index", lambda *a, **k: called.__setitem__("build", True))
    bg = BackgroundTasks()
    resp = asyncio.run(reindex(bg, force=True))
    assert resp.status == "upstream"
    assert len(bg.tasks) == 0
    assert called["build"] is False


def test_reindex_local_mode_kicks_build(monkeypatch):
    # url 미설정이면 기존 동작(빌드 킥)
    monkeypatch.setattr(vocaro_api, "is_building", lambda: False)
    bg = BackgroundTasks()
    resp = asyncio.run(reindex(bg, force=False))
    assert resp.status == "building"
    assert len(bg.tasks) == 1


def test_status_upstream_relay(monkeypatch):
    _set_url("http://idx.test")
    monkeypatch.setattr(
        vocaro_api,
        "_upstream_get",
        lambda path, params=None: {
            "built_at": "2026-07-24T00:00:00+00:00",
            "total": 5,
            "with_ja": 3,
            "building": False,
        },
    )
    resp = asyncio.run(status())
    assert resp.total == 5
    assert resp.with_ja == 3
    assert resp.building is False
    assert resp.built_at == "2026-07-24T00:00:00+00:00"


def test_status_upstream_error_returns_empty(monkeypatch):
    _set_url("http://idx.test")

    def boom(path, params=None):
        raise RuntimeError("down")

    monkeypatch.setattr(vocaro_api, "_upstream_get", boom)
    resp = asyncio.run(status())
    assert resp.total == 0
    assert resp.with_ja == 0
    assert resp.building is False
    assert resp.built_at is None
