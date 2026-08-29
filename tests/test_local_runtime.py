import json
import os
from pathlib import Path

import pytest

from everyric2.config.settings import AlignmentSettings
from everyric2.local_runtime import (
    BRIDGE_PROTOCOL,
    RESULT_SCHEMA,
    RuntimePaths,
    configure_environment,
)


def test_runtime_manifest_has_pinned_local_contracts() -> None:
    manifest = json.loads(
        (Path(__file__).parents[1] / "everyric2" / "runtime_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["engineVersion"] == "1.0.0"
    assert manifest["bridgeProtocol"] == BRIDGE_PROTOCOL
    assert manifest["resultSchema"] == RESULT_SCHEMA
    assert manifest["platforms"]["win32-x64"]["cuda"] == "12.8"
    assert len(manifest["models"]["omniasr"]["revision"]) == 40
    assert len(manifest["models"]["owsm"]["revision"]) == 40
    assert all(len(item["sha256"]) == 64 for item in manifest["models"]["polarformer"]["files"])
    assert all(
        len(item["sha256"]) == 64 for item in manifest["models"]["polarformer"]["vendor"]["files"]
    )


def test_runtime_paths_configure_all_engine_locations(tmp_path: Path) -> None:
    paths = RuntimePaths.resolve(tmp_path / "runtime", tmp_path / "models")
    configure_environment(paths)
    assert os.environ["EVERYRIC_HOME"] == str(paths.root)
    assert os.environ["HF_HOME"] == str(paths.hf_home)
    assert os.environ["EVERYRIC_AUDIO_SEPARATOR_MODEL_DIR"] == str(paths.models)
    assert os.environ["EVERYRIC_ALIGNMENT_OWSM_PYTHON_PATH"] == str(paths.owsm_python)
    assert os.environ["EVERYRIC_ALIGNMENT_ENGINE"] == "adaptive"


def test_legacy_engine_setting_is_rejected() -> None:
    assert AlignmentSettings().engine == "adaptive"
    with pytest.raises(Exception):
        AlignmentSettings(engine="ctc")
