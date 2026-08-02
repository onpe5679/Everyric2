"""Tests for everyric2.audio.separator — backend selection, htdemucs regression, and the
ported bs-polarformer-fp16 backend (everyric2/audio/polarformer_separator.py).

No GPU/model execution here — heavy paths (real demucs subprocess run, real audio_separator/
torch model forward) are mocked or exercised only up to the point that requires CUDA. See
scripts/bench_adapters/separators_quality.py for the bench harness that DOES run models.
"""

import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from everyric2.audio import polarformer_separator as pf
from everyric2.audio.loader import AudioData
from everyric2.audio.separator import (
    DemucsNotAvailableError,
    SeparationResult,
    SeparatorBackendUnavailableError,
    VocalSeparator,
)
from everyric2.config.settings import AudioSettings


def _silence(seconds: float = 0.1, sr: int = 24000) -> AudioData:
    n = int(seconds * sr)
    return AudioData(waveform=np.zeros(n, dtype=np.float32), sample_rate=sr, duration=seconds)


def _write_polarformer_assets(models_dir: Path) -> None:
    """require_available()가 통과할 만큼의 더미 파일을 심는다(내용은 검사하지 않는다)."""
    checkpoint, config = pf._asset_paths(models_dir)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"x")
    config.write_text("x", encoding="utf-8")
    _, _init_file, attend, model_file = pf._vendor_paths(models_dir)
    attend.parent.mkdir(parents=True, exist_ok=True)
    attend.write_text("x", encoding="utf-8")
    model_file.write_text("x", encoding="utf-8")


class TestAudioSettingsBackend:
    """새 설정 키 — 2026-08-03 배선 전환 이후 기본값은 bs-polarformer-fp16이다
    (alignment.engine=owsm/omniasr 새 스택 기본값과 짝을 이룬다 — Settings의 cross-field
    validator가 어긋나면 기동 시점에 실패시킨다). htdemucs로 되돌리려면 명시적으로 골라야
    한다(아래 TestHtdemucsRegression)."""

    def test_default_backend_is_polarformer(self):
        settings = AudioSettings()
        assert settings.separator_backend == "bs-polarformer-fp16"

    def test_htdemucs_still_selectable_explicitly(self):
        settings = AudioSettings(separator_backend="htdemucs")
        assert settings.separator_backend == "htdemucs"

    def test_default_model_dir(self):
        settings = AudioSettings()
        assert settings.separator_model_dir == Path.home() / ".cache" / "everyric2" / "models"

    def test_backend_overridable_via_env(self, monkeypatch):
        monkeypatch.setenv("EVERYRIC_AUDIO_SEPARATOR_BACKEND", "bs-polarformer-fp16")
        settings = AudioSettings()
        assert settings.separator_backend == "bs-polarformer-fp16"

    def test_invalid_backend_value_rejected(self):
        with pytest.raises(Exception):
            AudioSettings(separator_backend="not-a-real-backend")

    def test_model_dir_overridable_via_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("EVERYRIC_AUDIO_SEPARATOR_MODEL_DIR", str(tmp_path / "models"))
        settings = AudioSettings()
        assert settings.separator_model_dir == tmp_path / "models"


class TestSeparationResultBackendField:
    """반환 계약(SeparationResult)에 추가한 backend 필드 — 기본값이 있어 기존 호출부가
    수정 없이 동작한다는 것을 확인한다."""

    def test_default_backend_field(self):
        result = SeparationResult(vocals=object(), accompaniment=object(), original=object())
        assert result.backend == "htdemucs"

    def test_backend_field_overridable(self):
        result = SeparationResult(
            vocals=object(),
            accompaniment=object(),
            original=object(),
            backend="bs-polarformer-fp16",
        )
        assert result.backend == "bs-polarformer-fp16"

    def test_positional_construction_still_works(self):
        """호출부가 backend 없이 키워드 인자만 넘기는 기존 방식이 안 깨졌는지."""
        result = SeparationResult(
            vocals=object(),
            accompaniment=object(),
            original=object(),
        )
        assert result.backend == "htdemucs"


class TestHtdemucsRegression:
    """demucs 경로는 이 이식 작업으로 절대 바뀌지 않았다는 것을 확인한다.

    기본 백엔드가 bs-polarformer-fp16으로 바뀐 뒤로는(TestAudioSettingsBackend) htdemucs
    경로를 켜려면 명시적으로 골라야 한다 — 이 클래스의 모든 설정이 그렇게 한다."""

    def test_is_available_true_by_default(self):
        # demucs는 이 리포의 필수 의존성(pyproject.toml [separator])이라 개발/CI venv에
        # 실제로 설치돼 있다 — 이 값 자체가 기존 동작(회귀 없음)이다.
        separator = VocalSeparator(AudioSettings(separator_backend="htdemucs"))
        assert separator.is_available() is True

    def test_separate_still_uses_demucs_subprocess_and_tags_backend(self, tmp_path, monkeypatch):
        """subprocess.run을 목으로 대체해 실제 demucs를 돌리지 않고 배선만 확인한다.

        librosa.load()가 첫 호출 시 numpy/scipy를 지연 임포트하면서 CPU 기능탐지용
        subprocess.run을 내부적으로 부른다(numpy.testing._private.utils.check_support_sve) —
        그 호출까지 가로채면 무관한 이유로 깨지므로, demucs 커맨드가 아니면 진짜
        subprocess.run으로 통과시킨다.
        """
        settings = AudioSettings(temp_dir=tmp_path / "work", separator_backend="htdemucs")
        separator = VocalSeparator(settings)
        real_run = subprocess.run

        def fake_run(cmd, **kwargs):
            if not (isinstance(cmd, list) and "-n" in cmd and "demucs" in cmd):
                return real_run(cmd, **kwargs)
            # demucs가 만드는 파일 레이아웃을 재현: output_dir/model/demucs_input/*.wav
            model = cmd[cmd.index("-n") + 1]
            out_dir = Path(cmd[cmd.index("-o") + 1])
            stem_dir = out_dir / model / "demucs_input"
            stem_dir.mkdir(parents=True, exist_ok=True)
            sf.write(stem_dir / "vocals.wav", np.zeros(2400, dtype=np.float32), 24000)
            sf.write(stem_dir / "no_vocals.wav", np.zeros(2400, dtype=np.float32), 24000)
            return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        result = separator.separate(_silence())

        assert isinstance(result, SeparationResult)
        assert result.backend == "htdemucs"

    def test_demucs_not_available_raises(self, monkeypatch):
        separator = VocalSeparator(AudioSettings(separator_backend="htdemucs"))
        monkeypatch.setattr(separator, "_demucs_available", False)
        with pytest.raises(DemucsNotAvailableError):
            separator.separate(_silence())


class TestBackendSelection:
    """설정값에 따라 올바른 백엔드가 선택되는지, 그리고 새 백엔드가 조용히 htdemucs로
    새지 않고 명확한 예외를 던지는지."""

    def test_polarformer_selected_never_touches_demucs_cache(self, tmp_path):
        settings = AudioSettings(
            separator_backend="bs-polarformer-fp16",
            separator_model_dir=tmp_path / "models",
        )
        separator = VocalSeparator(settings)
        assert separator.is_available() is False  # 자산이 없다
        assert separator._demucs_available is None  # demucs 쪽 캐시는 건드리지 않았다

    def test_polarformer_missing_assets_raises_not_silent_fallback(self, tmp_path):
        settings = AudioSettings(
            separator_backend="bs-polarformer-fp16",
            separator_model_dir=tmp_path / "models",
        )
        separator = VocalSeparator(settings)
        with pytest.raises(SeparatorBackendUnavailableError) as excinfo:
            separator.separate(_silence())
        message = str(excinfo.value)
        assert "checkpoint missing" in message
        assert "config missing" in message
        assert str(tmp_path / "models") in message

    def test_polarformer_available_once_dependency_and_assets_present(self, tmp_path, monkeypatch):
        """audio-separator가 실제로 설치돼 있지 않으므로 sys.modules에 더미를 심어 임포트만
        통과시킨다 — 모델은 로드/실행하지 않는다."""
        models_dir = tmp_path / "models"
        _write_polarformer_assets(models_dir)
        monkeypatch.setitem(sys.modules, "audio_separator", types.ModuleType("audio_separator"))

        settings = AudioSettings(separator_backend="bs-polarformer-fp16", separator_model_dir=models_dir)
        separator = VocalSeparator(settings)
        assert separator.is_available() is True

    def test_polarformer_requires_cuda_even_with_assets_present(self, tmp_path):
        """가중치·설정·MSST 소스가 다 있어도 use_gpu=False면 명확한 예외 — fp16 autocast
        forward는 CPU 경로가 없다(scripts/bench_adapters/separators_quality.py와 동일 제약)."""
        models_dir = tmp_path / "models"
        _write_polarformer_assets(models_dir)

        with pytest.raises(pf.PolarFormerUnavailableError, match="CUDA"):
            pf.separate(tmp_path / "in.wav", tmp_path / "out", models_dir, use_gpu=False)

    def test_separate_ignores_model_arg_for_polarformer_backend(self, tmp_path):
        """demucs 전용 model= 인자는 이 백엔드에서 무시된다 — 자산 부재 예외가 그대로 난다
        (model 인자 때문에 다른 경로로 새지 않는다는 것을 확인)."""
        settings = AudioSettings(
            separator_backend="bs-polarformer-fp16",
            separator_model_dir=tmp_path / "models",
        )
        separator = VocalSeparator(settings)
        with pytest.raises(SeparatorBackendUnavailableError):
            separator.separate(_silence(), model="htdemucs_ft")


class TestPolarFormerAssetHelpers:
    """everyric2/audio/polarformer_separator.py 자체의 자산 탐색/이름 정규화 로직."""

    def test_missing_reasons_lists_everything_absent(self, tmp_path, monkeypatch):
        # audio_separator는 2026-08-04부터 pyproject.toml separator extra의 선언된
        # 의존성이라(--extra separator로 설치되는 배포 환경에선) 실제로 깔려 있을 수
        # 있다 — 이 테스트는 "패키지가 없을 때"를 실제 환경 상태와 무관하게 재현해야
        # 한다. sys.modules[name]=None은 그 이름의 import를 무조건 ImportError로 만드는
        # 표준 관용구다(반대 방향은 아래 test_missing_reasons_omits_audio_separator_
        # reason_when_importable, 이미 있던 test_missing_reasons_empty_once_assets_and_
        # dependency_present도 같은 목 패턴으로 "있을 때"를 재현하고 있었다).
        monkeypatch.setitem(sys.modules, "audio_separator", None)
        reasons = pf._missing_reasons(tmp_path / "models")
        joined = "\n".join(reasons)
        assert "checkpoint" in joined
        assert "config" in joined
        assert "attend.py" in joined
        assert "bs_roformer.py" in joined
        assert "audio-separator" in joined

    def test_missing_reasons_omits_audio_separator_reason_when_importable(
        self, tmp_path, monkeypatch
    ):
        # 반대 방향 — 패키지가 있으면(목으로 재현, 실제 설치 여부와 무관) 그 사유만
        # 목록에서 빠져야 한다. 다른 자산은 일부러 안 채운다 — "audio-separator" 사유
        # 하나를 격리해서 보는 것이 목적이다.
        monkeypatch.setitem(sys.modules, "audio_separator", types.ModuleType("audio_separator"))
        reasons = pf._missing_reasons(tmp_path / "models")
        joined = "\n".join(reasons)
        assert "checkpoint" in joined  # 자산은 여전히 안 채웠으니 다른 사유는 남는다
        assert "audio-separator" not in joined

    def test_missing_reasons_empty_once_assets_and_dependency_present(self, tmp_path, monkeypatch):
        models_dir = tmp_path / "models"
        _write_polarformer_assets(models_dir)
        monkeypatch.setitem(sys.modules, "audio_separator", types.ModuleType("audio_separator"))
        assert pf._missing_reasons(models_dir) == []

    def test_require_available_raises_with_all_reasons(self, tmp_path):
        with pytest.raises(pf.PolarFormerUnavailableError):
            pf.require_available(tmp_path / "models")

    def test_require_available_passes_once_provisioned(self, tmp_path, monkeypatch):
        models_dir = tmp_path / "models"
        _write_polarformer_assets(models_dir)
        monkeypatch.setitem(sys.modules, "audio_separator", types.ModuleType("audio_separator"))
        pf.require_available(models_dir)  # raises on failure; no exception == pass

    def test_ensure_vendor_package_marker_only_creates_init_file(self, tmp_path):
        """__init__.py 마커는 자동 생성해도 되지만(순수 패키징 절차), attend.py/bs_roformer.py
        같은 실제 모델 코드는 여전히 사전 조달 대상으로 남아야 한다(다운로드 금지 정책)."""
        models_dir = tmp_path / "models"
        vendor_dir = pf._ensure_vendor_package_marker(models_dir)
        _, init_file, attend, model_file = pf._vendor_paths(models_dir)
        assert init_file.exists()
        assert not attend.exists()
        assert not model_file.exists()
        assert vendor_dir == models_dir / pf._MSST_VENDOR_SUBDIR

    def test_normalize_outputs_renames_by_heuristic(self, tmp_path):
        raw_vocals = tmp_path / "song_(Vocals)_model.wav"
        raw_inst = tmp_path / "song_(Instrumental)_model.wav"
        raw_vocals.write_bytes(b"x")
        raw_inst.write_bytes(b"x")

        vocals_path, inst_path = pf._normalize_outputs([str(raw_vocals), str(raw_inst)], tmp_path)

        assert vocals_path == tmp_path / "vocals.wav"
        assert inst_path == tmp_path / "inst.wav"
        assert vocals_path.exists()
        assert inst_path.exists()

    def test_normalize_outputs_leaves_already_standard_names_alone(self, tmp_path):
        vocals = tmp_path / "vocals.wav"
        inst = tmp_path / "inst.wav"
        vocals.write_bytes(b"x")
        inst.write_bytes(b"x")

        vocals_path, inst_path = pf._normalize_outputs([str(vocals), str(inst)], tmp_path)

        assert vocals_path == vocals
        assert inst_path == inst
