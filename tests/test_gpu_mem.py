"""VRAM 회수 훅·idle 상주 가드 — GPU 없이 순수 논리만 검증.

실사고(2026-07-24, 동거 호스트에서 idle 예약 18.4GiB) 재발 방지 계약: 잡 경계마다
스크래치를 반환하고, 반환 후에도 임계를 넘으면 웜 캐시를 버린다.
"""

from everyric2 import gpu_mem
from everyric2.config.settings import get_settings


def test_guard_should_drop_boundaries():
    assert gpu_mem.guard_should_drop(9.0, 8.0) is True
    assert gpu_mem.guard_should_drop(7.9, 8.0) is False
    assert gpu_mem.guard_should_drop(None, 8.0) is False
    assert gpu_mem.guard_should_drop(99.0, 0) is False  # 0 = 가드 비활성


def test_reclaim_noop_without_gpu(monkeypatch):
    # torch 없음/CPU 전용이면 release가 None — 조용히 종료 (예외 전파 금지 계약)
    monkeypatch.setattr(gpu_mem, "release_scratch", lambda: None)
    gpu_mem.reclaim_after_job()


def test_reclaim_drops_warm_caches_over_limit(monkeypatch):
    calls = {"drop": 0}
    monkeypatch.setattr(gpu_mem, "release_scratch", lambda: 12.0)
    monkeypatch.setattr(
        gpu_mem, "drop_warm_caches", lambda: calls.__setitem__("drop", calls["drop"] + 1)
    )
    object.__setattr__(get_settings().server, "worker_vram_guard_gb", 8.0)
    gpu_mem.reclaim_after_job()
    assert calls["drop"] == 1


def test_reclaim_keeps_caches_under_limit(monkeypatch):
    calls = {"drop": 0}
    monkeypatch.setattr(gpu_mem, "release_scratch", lambda: 4.0)
    monkeypatch.setattr(
        gpu_mem, "drop_warm_caches", lambda: calls.__setitem__("drop", calls["drop"] + 1)
    )
    object.__setattr__(get_settings().server, "worker_vram_guard_gb", 8.0)
    gpu_mem.reclaim_after_job()
    assert calls["drop"] == 0


def test_drop_warm_caches_survives_partial_failure(monkeypatch):
    called: list[str] = []

    def bad() -> None:
        raise RuntimeError("clear 실패")

    monkeypatch.setattr(
        gpu_mem,
        "_clearers",
        lambda: [
            ("a", lambda: called.append("a")),
            ("bad", bad),
            ("b", lambda: called.append("b")),
        ],
    )
    gpu_mem.drop_warm_caches()
    assert called == ["a", "b"]  # 하나가 죽어도 나머지는 해제된다


def test_clear_shared_separator_resets_singleton():
    from everyric2.audio import separator

    separator._shared_separator = object()  # type: ignore[assignment]
    separator.clear_shared_separator()
    assert separator._shared_separator is None
