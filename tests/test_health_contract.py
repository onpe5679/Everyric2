"""Public status must report the executable stack, not a response-model default."""

from __future__ import annotations

import asyncio

from everyric2 import __version__
from everyric2.config.settings import get_settings
from everyric2.server import main
from everyric2.server.db.models import ENGINE_VERSION


def test_health_reports_package_and_stack_identity(monkeypatch) -> None:
    monkeypatch.setattr(main, "_gpu_available", lambda: True)

    response = asyncio.run(main.health_check())

    assert response.status == "healthy"
    assert response.version == __version__
    assert response.engine == get_settings().alignment.engine == "adaptive"
    assert response.engine_version == ENGINE_VERSION


def test_root_uses_the_same_identity_contract() -> None:
    response = asyncio.run(main.root())

    assert response["version"] == __version__
    assert response["engine"] == get_settings().alignment.engine
    assert response["engine_version"] == ENGINE_VERSION
