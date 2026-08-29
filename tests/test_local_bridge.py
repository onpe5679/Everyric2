import json
import sys
import types
from pathlib import Path

from everyric2 import local_bridge
from everyric2.local_runtime import BRIDGE_PROTOCOL, RESULT_SCHEMA


def test_local_bridge_align_streams_v3_and_preserves_audio(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"audio-placeholder")
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "protocol": BRIDGE_PROTOCOL,
                "audioPath": str(audio),
                "lyrics": "hello",
                "language": "en",
                "runtimeRoot": str(tmp_path / "runtime"),
            }
        ),
        encoding="utf-8",
    )
    calls = {}

    def fake_run(audio_path, lyrics, language, **kwargs):
        calls.update(audio_path=audio_path, lyrics=lyrics, language=language, **kwargs)
        kwargs["on_stage"]("adaptive")
        kwargs["on_depth"]("fast")
        return {"timestamps": [{"text": "hello", "start": 0.0, "end": 1.0}]}

    worker = types.ModuleType("everyric2.server.worker")
    worker._run_alignment = fake_run
    monkeypatch.setitem(sys.modules, "everyric2.server.worker", worker)
    monkeypatch.setattr(
        local_bridge,
        "health_report",
        lambda _paths: {"ready": True, "problems": [], "engineVersion": "1.0.0"},
    )

    assert local_bridge.align(request) == 0
    assert calls["delete_audio"] is False
    assert audio.exists()
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [event["event"] for event in events] == [
        "hello",
        "progress",
        "depth",
        "result",
    ]
    assert events[-1]["document"]["schema"] == RESULT_SCHEMA


def test_local_bridge_rejects_unknown_protocol(tmp_path: Path) -> None:
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"x")
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps({"protocol": "unknown/v9", "audioPath": str(audio), "lyrics": "x"}),
        encoding="utf-8",
    )
    try:
        local_bridge._read_request(request)
    except ValueError as exc:
        assert "unsupported bridge protocol" in str(exc)
    else:
        raise AssertionError("unknown protocol was accepted")
