from scripts import verify_deploy as vd


def _health(monkeypatch, *, version: str = "1.0.0", engine: str = "adaptive") -> None:
    monkeypatch.setattr(
        vd,
        "_http",
        lambda server, path: (
            200,
            {
                "status": "healthy",
                "version": version,
                "engine": engine,
                "gpu_available": True,
            },
        ),
    )
    vd._results.clear()


def test_health_requires_the_new_server_identity(monkeypatch):
    _health(monkeypatch)
    vd.check_health("https://example.test", expected_version="1.0.0", expected_engine="adaptive")
    assert vd._results[-1][1] == "PASS"


def test_health_rejects_a_service_that_did_not_restart(monkeypatch):
    _health(monkeypatch, version="0.3.0", engine="ctc")
    vd.check_health("https://example.test", expected_version="1.0.0", expected_engine="adaptive")
    assert vd._results[-1][1] == "FAIL"
    assert "새 코드가 실제 기동하지 않음" in vd._results[-1][2]


def test_baseline_mode_marks_the_old_service_pending(monkeypatch):
    _health(monkeypatch, version="0.3.0", engine="ctc")
    vd.check_health(
        "https://example.test",
        expected_version="1.0.0",
        expected_engine="adaptive",
        baseline=True,
    )
    assert vd._results[-1][1] == "PENDING"
