"""X-API-Key 미들웨어의 허용 집합 회귀 — "키를 안 보내면 통과"를 다시는 못 만들게.

실측 사고(엣지 감사 3.1, 2026-08-03): 허용 집합이 `(api_key, admin_api_key or None)`
튜플이라 어드민 키 미설정 배포에서 None이 허용값이 됐고, 헤더를 **아예 안 보낸**
요청(provided=None)이 401 없이 통과했다. 틀린 키는 막히는 형태라 curl로 잘못된 키를
넣어 보는 수동 점검으로는 절대 드러나지 않았다 — 그래서 부재/빈 문자열 케이스를
여기서 고정한다.

TestClient(httpx)가 의존성에 없어(test_cors_on_auth_failure.py와 같은 사정) HTTP
왕복 대신 미들웨어 함수를 직접 호출한다 — 검사 대상이 순수하게 이 함수의 분기라
잃는 것이 없다.
"""

from types import SimpleNamespace

import pytest

from everyric2.server import main as server_main


class _Sentinel:
    pass


def _request(path="/api/sync/x", method="GET", headers=None):
    return SimpleNamespace(
        method=method,
        url=SimpleNamespace(path=path),
        headers=headers or {},
    )


def _server_settings(api_key="secret-key-123", admin_api_key="", worker_key=""):
    return SimpleNamespace(
        server=SimpleNamespace(
            api_key=api_key, admin_api_key=admin_api_key, worker_key=worker_key
        )
    )


async def _call(request, settings, monkeypatch):
    import everyric2.config.settings as settings_mod

    monkeypatch.setattr(settings_mod, "get_settings", lambda: settings)
    sentinel = _Sentinel()

    async def call_next(_req):
        return sentinel

    result = await server_main.require_api_key(request, call_next)
    return result, sentinel


@pytest.mark.asyncio
async def test_missing_header_is_rejected_when_key_is_set(monkeypatch):
    """사고 재현 케이스 — 어드민 키 미설정 + 헤더 부재는 401이어야 한다."""
    result, sentinel = await _call(_request(), _server_settings(), monkeypatch)
    assert result is not sentinel and result.status_code == 401


@pytest.mark.asyncio
async def test_empty_header_is_rejected(monkeypatch):
    result, sentinel = await _call(
        _request(headers={"x-api-key": ""}), _server_settings(), monkeypatch
    )
    assert result is not sentinel and result.status_code == 401


@pytest.mark.asyncio
async def test_wrong_key_is_rejected(monkeypatch):
    result, sentinel = await _call(
        _request(headers={"x-api-key": "wrong"}), _server_settings(), monkeypatch
    )
    assert result is not sentinel and result.status_code == 401


@pytest.mark.asyncio
async def test_correct_key_passes(monkeypatch):
    result, sentinel = await _call(
        _request(headers={"x-api-key": "secret-key-123"}), _server_settings(), monkeypatch
    )
    assert result is sentinel


@pytest.mark.asyncio
async def test_admin_key_passes_when_set(monkeypatch):
    result, sentinel = await _call(
        _request(headers={"x-api-key": "admin-9"}),
        _server_settings(admin_api_key="admin-9"),
        monkeypatch,
    )
    assert result is sentinel


@pytest.mark.asyncio
async def test_no_key_configured_stays_open(monkeypatch):
    """로컬 단일 사용자 기본(키 미설정)은 기존대로 통과한다."""
    result, sentinel = await _call(
        _request(), _server_settings(api_key=""), monkeypatch
    )
    assert result is sentinel


@pytest.mark.asyncio
async def test_worker_key_exempts_worker_path_only(monkeypatch):
    settings = _server_settings(worker_key="wk-1")
    result, sentinel = await _call(
        _request(path="/api/worker/claim", headers={"x-worker-key": "wk-1"}),
        settings,
        monkeypatch,
    )
    assert result is sentinel
    # 같은 워커 키라도 워커 경로 밖이면 면제되지 않는다
    result, sentinel = await _call(
        _request(path="/api/sync/x", headers={"x-worker-key": "wk-1"}),
        settings,
        monkeypatch,
    )
    assert result is not sentinel and result.status_code == 401
