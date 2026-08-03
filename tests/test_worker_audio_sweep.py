"""기동 시 워커 전달용 오디오 잔재 정리 — 지울 것만 지우는지 고정.

_WORKER_AUDIO는 인메모리라 재시작하면 dict만 사라지고 파일은 남는다. 그 파일을 지울
주체가 영영 없어져 "잡 터미널 지점에서 삭제"라는 저작권 규약이 조용히 깨진다(엣지 감사
5.1). 여기서 못박는 것은 **정리 범위**다 — temp_dir에는 다른 주인의 파일도 있으므로,
media_cache가 만드는 두 패턴만 지워야 한다.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from everyric2.server.api.worker import sweep_orphan_worker_audio


@pytest.fixture
def temp_audio_dir(tmp_path, monkeypatch):
    """audio.temp_dir을 격리 디렉터리로 돌린다(실제 temp를 건드리지 않게)."""
    import everyric2.config.settings as settings_mod

    monkeypatch.setattr(
        settings_mod,
        "get_settings",
        lambda: SimpleNamespace(audio=SimpleNamespace(temp_dir=tmp_path)),
    )
    return tmp_path


def _touch(d: Path, name: str) -> Path:
    p = d / name
    p.write_bytes(b"x")
    return p


def test_sweeps_worker_and_linkcache_patterns(temp_audio_dir):
    worker = _touch(temp_audio_dir, "b2NTglk9tvI-6037d698.m4a")
    linkcache = _touch(temp_audio_dir, "linkcache-src-b2NTglk9tvI.m4a")

    assert sweep_orphan_worker_audio() == 2
    assert not worker.exists()
    assert not linkcache.exists()


def test_leaves_other_owners_files_alone(temp_audio_dir):
    """temp_dir은 공용이다 — 다른 주인의 파일을 지우면 진행 중 작업을 깨뜨린다."""
    keepers = [
        _touch(temp_audio_dir, "vocals.wav"),          # 분리기 중간 산출물
        _touch(temp_audio_dir, "input.wav"),           # demucs 입력
        _touch(temp_audio_dir, "b2NTglk9tvI.m4a"),     # 잡 접미사 없는 캐시 원본
        _touch(temp_audio_dir, "some-notes.txt"),
        _touch(temp_audio_dir, "b2NTglk9tvI-6037d698.opus"),  # 패턴은 .m4a만
    ]
    assert sweep_orphan_worker_audio() == 0
    assert all(p.exists() for p in keepers)


def test_job_suffix_must_be_eight_hex(temp_audio_dir):
    """job_id[:8]은 16진 8자다 — 그 형태가 아니면 우리 파일이 아니다."""
    not_ours = _touch(temp_audio_dir, "video-notahex1.m4a")
    ours = _touch(temp_audio_dir, "video-0a1b2c3d.m4a")

    assert sweep_orphan_worker_audio() == 1
    assert not_ours.exists()
    assert not ours.exists()


def test_missing_temp_dir_is_not_an_error(tmp_path, monkeypatch):
    import everyric2.config.settings as settings_mod

    monkeypatch.setattr(
        settings_mod,
        "get_settings",
        lambda: SimpleNamespace(audio=SimpleNamespace(temp_dir=tmp_path / "nope")),
    )
    assert sweep_orphan_worker_audio() == 0


def test_subdirectories_are_skipped(temp_audio_dir):
    """이름이 패턴과 겹치는 디렉터리가 있어도 unlink를 시도하지 않는다."""
    (temp_audio_dir / "video-0a1b2c3d.m4a").mkdir()
    assert sweep_orphan_worker_audio() == 0
    assert (temp_audio_dir / "video-0a1b2c3d.m4a").is_dir()
