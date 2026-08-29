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
        return {"found": True, "slug": "polaris", "ja": "POLARIS"}

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


def test_video_id_lookup_is_skipped_when_client_evidence_already_matches(monkeypatch):
    _set_url("")
    monkeypatch.setattr(
        vocaro_api,
        "match_with_evidence",
        lambda title, **kwargs: _matched(SongEntry(slug="roki", ko="로키", ja="ロキ")),
    )
    monkeypatch.setattr(
        vocaro_api,
        "fetch_video_identity",
        lambda video_id: (_ for _ in ()).throw(AssertionError("unneeded metadata fetch")),
    )
    resp = asyncio.run(
        match_title(BackgroundTasks(), title="ロキ", video_id="bEV_tH_yrIc")
    )
    assert resp.found is True
    assert resp.evidence_source == "client"


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
        "search_match",
        lambda title: SongEntry(slug="long-song", ko="긴 제목의 노래 입니다", ja=None),
    )
    resp = asyncio.run(
        match_title(BackgroundTasks(), title="노래 입니다", mode="search")
    )
    assert resp.found is True
    assert resp.slug == "long-song"
    assert resp.reason == "manual_search_fuzzy"
    assert resp.evidence_source == "manual_search"


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
