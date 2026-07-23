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
from everyric2.server.vocaro_index import SongEntry


def _set_url(url: str) -> None:
    object.__setattr__(get_settings().server, "song_index_url", url)


# ── url 미설정: 기존 로컬 인덱스 경로 그대로 ──────────────────────


def test_match_uses_local_index_when_url_unset(monkeypatch):
    # song_index_url 기본값("")이면 로컬 match()로 슬러그를 답하고 page_url은 BASE_URL 기반
    monkeypatch.setattr(vocaro_api, "match", lambda title: SongEntry(slug="roki", ko="로키", ja="ロキ"))
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


def test_match_upstream_not_found_passthrough(monkeypatch):
    _set_url("http://idx.test")
    monkeypatch.setattr(vocaro_api, "_upstream_get", lambda path, params=None: {"found": False})
    resp = asyncio.run(match_title(BackgroundTasks(), title="없는곡"))
    assert resp.found is False
    assert resp.slug is None


def test_match_upstream_error_returns_upstream_error(monkeypatch):
    _set_url("http://idx.test")

    def boom(path, params=None):
        raise RuntimeError("timeout")

    monkeypatch.setattr(vocaro_api, "_upstream_get", boom)
    resp = asyncio.run(match_title(BackgroundTasks(), title="x"))
    assert resp.found is False
    assert resp.status == "upstream_error"


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
