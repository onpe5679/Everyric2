"""Vocal separation using Demucs (default) or the ported bs-polarformer-fp16 backend."""

import logging
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

from everyric2.audio.loader import AudioData, AudioLoader
from everyric2.config.settings import AudioSettings, get_settings

logger = logging.getLogger(__name__)

# 웜 캐시 싱글턴 (WS2-A) — 프로세스 수명 동안 상주. 지연 생성이라 import만으로는 아무것도
# 로드하지 않는다 (API 전용 모드에서 절대 만들어지지 않게 하는 유일한 근거).
_shared_separator: "VocalSeparator | None" = None
_shared_separator_lock = threading.Lock()


def get_shared_separator(config: "AudioSettings | None" = None) -> "VocalSeparator":
    """웜 캐시된 VocalSeparator를 돌려준다 (EVERYRIC_SERVER_WARM_MODELS 기준).

    warm이 켜져 있으면 프로세스 수명 싱글턴을 재사용하고(두 번째 잡부터 재생성 0회), 재사용
    시 "warm model reuse: demucs" 1줄을 남긴다. 꺼져 있으면 매번 새 인스턴스(기존 동작).
    demucs는 서브프로세스로 도는 구조라 인스턴스 재사용이 인프로세스 모델 재로드를 없애지는
    않지만, 스펙의 싱글턴 규약을 동일하게 따른다."""
    if not get_settings().server.warm_models:
        return VocalSeparator(config)
    global _shared_separator
    with _shared_separator_lock:
        if _shared_separator is None:
            _shared_separator = VocalSeparator(config)
        else:
            logger.info("warm model reuse: demucs")
        return _shared_separator


def clear_shared_separator() -> None:
    """웜 캐시 해제 (VRAM 가드용) — 다음 요청에서 지연 재생성된다."""
    global _shared_separator
    with _shared_separator_lock:
        _shared_separator = None


class SeparationError(Exception):
    """Base exception for separation operations."""

    pass


class DemucsNotAvailableError(SeparationError):
    """Raised when Demucs is not installed."""

    pass


class SeparatorBackendUnavailableError(SeparationError):
    """설정된 분리기 백엔드(예: bs-polarformer-fp16)의 의존성/모델 자산/CUDA가 없을 때.

    DemucsNotAvailableError와 대칭인 non-demucs 백엔드용 예외 — 조용히 htdemucs로 새지 않고
    이 예외로 표면화한다(everyric2/audio/polarformer_separator.py의 조용한 폴백 금지 정책).
    """

    pass


@dataclass
class SeparationResult:
    """Result of vocal separation."""

    vocals: AudioData
    accompaniment: AudioData
    original: AudioData
    # 실제로 돌아간 분리기 이름 — 새 필드지만 기본값이 있어 기존 호출부(worker.py 등)는
    # 수정 없이 동작한다. htdemucs 경로는 실제 demucs 모델 이름(예: "htdemucs_ft")을 채워
    # 넣고, bs-polarformer-fp16 경로는 polarformer_separator.BACKEND_NAME을 채워 넣는다.
    backend: str = "htdemucs"


class VocalSeparator:
    """Separate vocals from music using Demucs."""

    AVAILABLE_MODELS = [
        "htdemucs",
        "htdemucs_ft",
        "htdemucs_6s",
        "mdx",
        "mdx_extra",
        "mdx_extra_q",
    ]

    def __init__(self, config: AudioSettings | None = None):
        """Initialize separator.

        Args:
            config: Audio settings. If None, uses global settings.
        """
        self.config = config or get_settings().audio
        self.loader = AudioLoader(config)
        self._demucs_available: bool | None = None

    def is_available(self) -> bool:
        """Check whether the CONFIGURED separator backend is available.

        Returns:
            True if the backend named by ``config.separator_backend`` is installed/provisioned.
        """
        if self.config.separator_backend == "bs-polarformer-fp16":
            # 필요조건은 파일시스템 stat 몇 개 + 임포트 시도뿐이라 캐시하지 않는다 — 캐시하면
            # 서버가 떠 있는 동안 자산을 사후 조달해도 "불가" 판정이 프로세스 재시작 전까지
            # 굳어버린다.
            from everyric2.audio import polarformer_separator as pf

            return pf.dependencies_and_assets_available(self.config.separator_model_dir)

        # 기존 htdemucs 경로 — 아래는 변경하지 않는다.
        if self._demucs_available is not None:
            return self._demucs_available

        try:
            import demucs  # noqa: F401

            self._demucs_available = True
        except ImportError:
            self._demucs_available = False

        return self._demucs_available

    def get_available_models(self) -> list[str]:
        """Get list of available Demucs models.

        Returns:
            List of model names.
        """
        return self.AVAILABLE_MODELS.copy()

    def separate(
        self,
        audio: AudioData,
        model: str | None = None,
        use_gpu: bool = True,
    ) -> SeparationResult:
        """Separate vocals from audio.

        Args:
            audio: Audio data to process.
            model: Demucs model name. Defaults to config setting.
            use_gpu: Whether to use GPU acceleration.

        Returns:
            SeparationResult with vocals and accompaniment.

        Raises:
            DemucsNotAvailableError: If Demucs is not installed (htdemucs backend).
            SeparatorBackendUnavailableError: If the configured non-demucs backend's
                dependencies/model assets/CUDA are missing (e.g. bs-polarformer-fp16).
            SeparationError: If separation fails.
        """
        if self.config.separator_backend == "bs-polarformer-fp16":
            # ``model``은 demucs 모델 선택 인자라 이 백엔드에서는 의미가 없다(단일 스펙) —
            # 조용히 무시한다. 지금 이 값을 넘기는 호출부는 없다(everyric2/cli.py,
            # everyric2/melody/extractor.py, everyric2/server/worker.py 전부 위치인자 없이
            # audio/use_gpu만 넘긴다).
            return self._separate_polarformer(audio, use_gpu=use_gpu)

        # 기존 htdemucs 경로 — 이 시점부터 아래는 변경하지 않는다.
        if not self.is_available():
            raise DemucsNotAvailableError(
                "Demucs is not installed. Install with: pip install demucs"
            )

        model = model or self.config.demucs_model
        temp_dir = self.config.temp_dir
        temp_dir.mkdir(parents=True, exist_ok=True)
        # 잡마다 고유한 하위 디렉터리 — 예전엔 temp_dir 바로 아래 고정 파일명
        # (demucs_input.wav/demucs_output/<model>/demucs_input/...)을 썼는데, temp_dir이
        # 프로세스 전역 공유라 동시 요청은 물론 같은 잡 안에서도(멜로디 f0가 별도
        # ThreadPoolExecutor에서 자기 VocalSeparator로 재분리할 때 — melody/extractor.py의
        # _maybe_separate) 서로의 입력·출력 파일을 덮어쓰거나 지웠다(운영자 지시, 2026-08-04
        # 실곡 검증 — bs-polarformer-fp16 경로에서 먼저 재현됐지만 이 경로도 같은 결함).
        # tempfile.mkdtemp가 원자적으로 고유 이름을 보장한다.
        call_dir = Path(tempfile.mkdtemp(dir=temp_dir, prefix="demucs_"))

        # Save input audio to temp file
        input_path = call_dir / "input.wav"
        audio.to_file(input_path)

        # Output directory for Demucs
        output_dir = call_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Build Demucs command
            cmd = [
                sys.executable,
                "-m",
                "demucs",
                "-n",
                model,
                "--two-stems",
                "vocals",
                # CLI 기본 --shifts 1은 무작위 시프트 «1회»라 평균 이득 없이 무작위성만
                # 넣는다 — 같은 입력의 두 분리가 다른 파형이 되고, 그 차이가 정렬
                # 비결정성의 실제 오염원이었다 (근거는 AudioSettings.demucs_shifts).
                "--shifts",
                str(getattr(self.config, "demucs_shifts", 0)),
                "-o",
                str(output_dir),
            ]

            if not use_gpu:
                cmd.extend(["-d", "cpu"])

            cmd.append(str(input_path))

            # Run Demucs — Windows 기본 콘솔 인코딩(cp949)로는 demucs의 유니코드
            # 진행 표시를 못 읽어 reader thread가 죽으므로 utf-8을 명시한다
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=600,  # 10 minute timeout
            )

            if result.returncode != 0:
                # Try CPU fallback if GPU failed
                if use_gpu and "CUDA" in result.stderr:
                    return self.separate(audio, model, use_gpu=False)
                raise SeparationError(f"Demucs failed: {result.stderr}")

            # Find output files
            # Demucs outputs to: output_dir/model/<input stem>/vocals.wav, no_vocals.wav
            model_output_dir = output_dir / model / input_path.stem

            vocals_path = model_output_dir / "vocals.wav"
            no_vocals_path = model_output_dir / "no_vocals.wav"

            if not vocals_path.exists() or not no_vocals_path.exists():
                raise SeparationError(f"Output files not found. Expected at: {model_output_dir}")

            # Load separated audio
            vocals = self.loader.load(vocals_path)
            accompaniment = self.loader.load(no_vocals_path)

            return SeparationResult(
                vocals=vocals,
                accompaniment=accompaniment,
                original=audio,
                backend=model,
            )

        except subprocess.TimeoutExpired:
            raise SeparationError("Demucs timed out (>10 minutes)")
        except Exception as e:
            if isinstance(e, SeparationError):
                raise
            raise SeparationError(f"Separation failed: {e}") from e
        finally:
            # call_dir 하나만 지운다(입력 wav + demucs 출력 전부 그 안에 있다) — 공유
            # temp_dir 자체는 절대 건드리지 않는다(다른 잡의 call_dir이 같은 부모 밑에
            # 나란히 있을 수 있다). vocals/accompaniment는 이미 loader.load()로 메모리에
            # 읽어 둔 뒤라(AudioData.waveform은 numpy 배열, 파일을 다시 참조하지 않는다)
            # 디렉터리를 지워도 반환값엔 영향이 없다. 예전엔 input_path만 지우고
            # output_dir(=demucs_output/)은 영영 안 지워 잡마다 디스크가 계속 쌓였다 —
            # call_dir 단위 정리가 그 누수도 함께 없앤다.
            shutil.rmtree(call_dir, ignore_errors=True)

    def _separate_polarformer(self, audio: AudioData, use_gpu: bool) -> SeparationResult:
        """bs-polarformer-fp16 경로 — everyric2/audio/polarformer_separator.py로 위임한다.

        모델 조달 실패(자산 부재)와 CUDA 부재는 SeparatorBackendUnavailableError로,
        분리 자체의 실패는 SeparationError로 표면화한다 — 둘 다 htdemucs로 조용히
        폴백하지 않는다(어느 분리기가 실제로 돌았는지 모르면 결과 해석이 불가능해진다).
        """
        from everyric2.audio import polarformer_separator as pf

        models_dir = self.config.separator_model_dir
        try:
            pf.require_available(models_dir)
        except pf.PolarFormerUnavailableError as exc:
            raise SeparatorBackendUnavailableError(str(exc)) from exc

        temp_dir = self.config.temp_dir
        temp_dir.mkdir(parents=True, exist_ok=True)
        # 잡마다 고유한 하위 디렉터리 — temp_dir 바로 아래 고정 파일명(polarformer_input.wav/
        # polarformer_output)을 쓰면 동시 요청은 물론 같은 잡 안에서도 경합한다: 멜로디 f0가
        # 별도 ThreadPoolExecutor에서 자기 VocalSeparator로 재분리할 때(vocals=None으로
        # precompute_f0에 들어가면 melody/extractor.py._maybe_separate가 독립적으로
        # VocalSeparator().separate()를 부른다) 메인 스레드의 분리와 동시에 같은 파일에
        # 쓰고 지운다(운영자 지시, 2026-08-04 실곡 검증 — 熱異常이 이 경합으로 죽었다: 한쪽
        # finally의 unlink가 다른 쪽이 읽던 파일을 지웠다). tempfile.mkdtemp가 원자적으로
        # 고유 이름을 보장한다.
        call_dir = Path(tempfile.mkdtemp(dir=temp_dir, prefix="polarformer_"))
        input_path = call_dir / "input.wav"
        audio.to_file(input_path)
        work_dir = call_dir / "output"

        try:
            try:
                result = pf.separate(input_path, work_dir, models_dir, use_gpu=use_gpu)
            except pf.PolarFormerUnavailableError as exc:
                raise SeparatorBackendUnavailableError(str(exc)) from exc
            except pf.PolarFormerBackendError as exc:
                raise SeparationError(f"{pf.BACKEND_NAME} separation failed: {exc}") from exc

            logger.info(
                "separator backend used: %s (elapsed=%.2fs)", pf.BACKEND_NAME, result.elapsed_sec
            )
            vocals = self.loader.load(result.vocals_path)
            accompaniment = self.loader.load(result.inst_path)

            return SeparationResult(
                vocals=vocals,
                accompaniment=accompaniment,
                original=audio,
                backend=pf.BACKEND_NAME,
            )
        finally:
            # call_dir 하나만 지운다(입력 wav + 분리 출력 전부 그 안에 있다) — 공유
            # temp_dir 자체는 절대 건드리지 않는다(다른 잡의 call_dir이 같은 부모 밑에
            # 나란히 있을 수 있다). vocals/accompaniment는 이미 loader.load()로 메모리에
            # 읽어 둔 뒤라 디렉터리를 지워도 반환값엔 영향이 없다.
            shutil.rmtree(call_dir, ignore_errors=True)

    def separate_file(
        self,
        audio_path: Path | str,
        model: str | None = None,
        use_gpu: bool = True,
    ) -> SeparationResult:
        """Separate vocals from audio file.

        Args:
            audio_path: Path to audio file.
            model: Demucs model name.
            use_gpu: Whether to use GPU.

        Returns:
            SeparationResult with vocals and accompaniment.
        """
        audio = self.loader.load(audio_path)
        return self.separate(audio, model, use_gpu)
