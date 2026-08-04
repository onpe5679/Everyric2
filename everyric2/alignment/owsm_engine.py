"""OWSM-CTC v4 1B 정렬 엔진 — 벤치(``scripts/bench_adapters/owsm_ctc.py``) ja 스택 앵커의
서버 이식.

``espnet/owsm_ctc_v4_1B``는 ESPnet 체크포인트(``config.yaml`` + ``.pth`` + 5만
SentencePiece 모델)이지 Transformers 리포지토리가 아니고, ESPnet 자체가 서버 메인
.venv(torch>=2.0, transformers>=4.40)와 충돌하는 의존성 세트를 고정한다 — 벤치가
``benchmark/.venv-owsm``로 이미 겪은 문제와 동일하다(``pyproject.toml``에 ``espnet``/
``sentencepiece``가 아예 없다는 것으로 확인). 그래서 이 엔진은 **모델을 인프로세스로
로드하지 않는다** — 별도 venv(기본 ``<repo_root>/.venv-owsm``, ``AlignmentSettings.
owsm_python_path``로 override 가능)에 격리된 인터프리터를 서브프로세스로 불러
``_owsm_worker.py``를 돌린다. 이 판단은 VRAM/크래시 격리가 아니라 **의존성 격리가
강제**하는 것이다 — 인프로세스로 두면 서버 .venv에 ESPnet을 얹어야 하는데 그 자체가
torch/transformers 버전 충돌을 부른다. (``OmniASREngine``은 이 문제가 없어 인프로세스로
돈다 — 그 모듈 docstring 참고.)

**핵심 실패 모드** (모델 형태가 강제하는 설계, 실제 추론은 ``_owsm_worker.py``에 있다):
인코더가 ``[<lang>, <asr>]`` 2토큰 프리픽스에 조건화돼 있어 그 상태가 인코더 출력
**앞에 붙는다.** 워커가 프리픽스 프레임 처리를 검증하지 않으면 **모든 타임스탬프가 오류
없이 조용히 ~160ms 밀린다**(값은 정상 반환된다) — ``_owsm_worker._verify_prefix_surplus``가
이 가드다. 이식 과정에서 이 가드가 유실되면 아무도 모르게 전곡이 밀리므로
``tests/test_owsm_engine.py``가 단위 테스트로 못박는다.

vocab: 5만 유니그램 SentencePiece라 토큰 하나가 가사 글자 여러 개를 덮을 수 있다. 워커는
모델의 네이티브 토큰화로 강제정렬하고 각 토큰의 측정된 스팬을 그 토큰이 덮는 글자들에
비례 분배한다 — 줄 경계는 항상 측정된 토큰 경계이고, 다중 글자 토큰 내부만 보간이다.

언어: 다국어. ko/ja/en/zh는 ``<kor>``/``<jpn>``/``<eng>``/``<cmn>``으로 매핑하고, 그 밖은
모델 자신의 "미지정" 심볼 ``<nolang>``으로 떨어진다 — 어떤 계층도 통째로 막히지 않는다.

라이선스: CC-BY-4.0(모델 카드). 상업적 사용에 저작자 표시만 요구한다 — CC-BY-NC-4.0 MMS
기준선과 다르다. 학습 말뭉치 ``espnet/yodas_owsmv4``는 별도 라이선스다.

**emission 노출 없음.** 이 엔진은 ``BaseAlignmentEngine.emission_for``의 기본 구현
(``None``)을 그대로 쓴다 — emission 텐서가 격리 프로세스에 살아 안전하게 못 넘어온다.
2패스 리파이너가 이 엔진에서 실제로 필요로 하는 것은 텐서가 아니라 **라인 창**이고,
``align()``이 돌려주는 ``SyncResult.start_time``/``end_time``이 그 계약이다
(``everyric2/alignment/emission.py`` 모듈 docstring 참고).
"""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from everyric2.alignment.base import (
    AlignmentError,
    BaseAlignmentEngine,
    EngineNotAvailableError,
    TranscriptionResult,
    WordTimestamp,
)
from everyric2.alignment.hf_cache import find_cached_file
from everyric2.audio.loader import AudioData
from everyric2.config.settings import AlignmentSettings
from everyric2.inference.prompt import LyricLine, SyncResult, WordSegment

REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKER_SCRIPT = Path(__file__).resolve().parent / "_owsm_worker.py"
_VENV_DIR_NAME = ".venv-owsm"

MODEL_ID = "espnet/owsm_ctc_v4_1B"
ADAPTER_NAME = "owsm-ctc-v4-1b"
EXP_DIR_NAME = "s2t_train_owsmctc_ebf27_conv2d8_size1024_mel128_bs320_raw_bpe50000"
MODEL_FILE_NAME = "valid.total_count.ave_5best.till70epoch.pth"

ALIGN_CHUNK_OVERLAP_SEC = 5.0
DEFAULT_DTYPE = "bfloat16"  # 벤치 채택 구성(owsm-ctc-v4-1b-bf16)과 동일 — 모듈 docstring 참고

LANGUAGE_SYMBOLS = {"ko": "<kor>", "ja": "<jpn>", "en": "<eng>", "zh": "<cmn>"}
DEFAULT_LANGUAGE_SYMBOL = "<nolang>"

WORKER_TIMEOUT_SEC = 3600

logger = logging.getLogger(__name__)


def _default_owsm_python() -> Path:
    """OWSM 워커 인터프리터의 기본 경로. ``AlignmentSettings.owsm_python_path``로 override.

    운영자가 배포 시 ``<repo_root>/.venv-owsm``에 ESPnet 전용 venv를 미리 만들어 둔다는
    전제다(가중치 조달과 마찬가지로 사전 조달 대상 — 서버가 첫 요청에서 자동으로 만들지
    않는다). 플랫폼별 인터프리터 하위 경로만 다르다.
    """
    root = REPO_ROOT / _VENV_DIR_NAME
    if sys.platform == "win32":
        return root / "Scripts" / "python.exe"
    return root / "bin" / "python3"


def _find_snapshot() -> Path | None:
    """``espnet/owsm_ctc_v4_1B`` 스냅샷 루트를 HF 캐시에서 찾는다. 없으면 ``None``."""
    match = find_cached_file(MODEL_ID, f"exp/{EXP_DIR_NAME}/{MODEL_FILE_NAME}")
    if match is None:
        return None
    # match == .../snapshots/<hash>/exp/<EXP_DIR_NAME>/<MODEL_FILE_NAME>
    # parents[0]=EXP_DIR_NAME, [1]="exp", [2]=스냅샷 루트(<hash>)
    return match.parents[2]


def _base_language(language: str) -> str:
    return (language or "").strip().lower()


def _line_confidence(word_confidences: list[float | None]) -> float | None:
    """줄 전체 신뢰도 — 글자별(``WordSegment``) confidence의 기하평균.

    ``_owsm_worker.py``가 이미 각 세그에 같은 공식(``exp(min(0, log_score))``)으로
    ``confidence``를 채워 보낸다 — 이 함수는 그 값들을 라인 하나로 합칠 뿐이다.
    ``scripts/bench_adapters/hf_ctc.py::_confidence``(원시 로그점수 평균 뒤 exp 한 번)와
    수학적으로 같은 값이고, ``everyric2.server.worker._geomean``도 같은 공식이다 — 라인
    conf가 비어 있으면(과거엔 이 함수가 없어 늘 그랬다) 그쪽이 폴백으로 다시 계산해
    왔는데, 정렬 직후 라인 단위 신호가 필요한 호출부(예: 새 스택 라우팅)는 그 폴백보다
    먼저 실행돼 언제나 ``None``을 봤다 — 실측 버그(2026-08-03, 코디네이터 실곡 검증
    — omniasr_engine.py에서 먼저 발견돼 owsm에도 같은 자리를 채웠다).
    """
    values = [v for v in word_confidences if v is not None and v > 0]
    if not values:
        return None
    return round(math.exp(sum(math.log(v) for v in values) / len(values)), 6)


def _interpolate_line_times(
    times: list[tuple[float, float] | None], audio_length: float
) -> list[tuple[float, float]]:
    """정렬된 글자가 0개인 줄(전부 OOV/모델이 못 들음)을 앞뒤 줄 사이 간격에 균등 배분."""
    result = list(times)
    n = len(result)
    i = 0
    while i < n:
        if result[i] is not None:
            i += 1
            continue
        start = i
        end = i
        while end + 1 < n and result[end + 1] is None:
            end += 1
        prev_end = result[start - 1][1] if start > 0 and result[start - 1] else 0.0
        next_start = result[end + 1][0] if end + 1 < n and result[end + 1] else audio_length
        available = max(0.0, next_start - prev_end)
        count = end - start + 1
        seg = max(available / count if count else 0.0, 0.1)
        for j in range(start, end + 1):
            offset = j - start
            result[j] = (prev_end + offset * seg, prev_end + (offset + 1) * seg)
        i = end + 1
    return [t if t is not None else (0.0, 0.0) for t in result]


def _build_sync_results(
    lyrics: list[LyricLine], lines: list[str], result: dict[str, Any]
) -> tuple[list[SyncResult], list[WordTimestamp]]:
    """워커 응답(줄별 글자 세그 JSON) -> (``SyncResult`` 목록, 평탄화된 ``WordTimestamp`` 목록)."""
    worker_lines = result.get("lines") or []
    if len(worker_lines) != len(lines):
        raise AlignmentError(
            f"owsm worker returned {len(worker_lines)} lines for {len(lines)} inputs"
        )
    audio_length = float(result.get("audio_sec") or 0.0)

    line_times: list[tuple[float, float] | None] = []
    line_segments: list[list[WordSegment]] = []
    for entry in worker_lines:
        segs = entry.get("segs") or []
        word_segments = [
            WordSegment(
                word=seg["t"],
                start=float(seg["start"]),
                end=float(seg["end"]),
                confidence=seg.get("confidence"),
            )
            for seg in segs
        ]
        line_segments.append(word_segments)
        line_times.append(
            (word_segments[0].start, word_segments[-1].end) if word_segments else None
        )

    interpolated = _interpolate_line_times(line_times, audio_length)
    results: list[SyncResult] = []
    all_words: list[WordTimestamp] = []
    for line, (start, end), segs in zip(lyrics, interpolated, line_segments):
        results.append(
            SyncResult(
                line_number=line.line_number,
                text=line.text,
                start_time=start,
                end_time=end,
                word_segments=segs or None,
                confidence=_line_confidence([s.confidence for s in segs]) if segs else None,
            )
        )
        all_words.extend(
            WordTimestamp(word=s.word, start=s.start, end=s.end, confidence=s.confidence)
            for s in segs
        )
    return results, all_words


class OwsmEngine(BaseAlignmentEngine):
    """OWSM-CTC v4 1B 강제 정렬 엔진 — 서브프로세스 격리(ESPnet 전용 venv)."""

    def __init__(self, config: AlignmentSettings | None = None) -> None:
        super().__init__(config)
        self._snapshot: Path | None = None
        self._last_word_timestamps: list[WordTimestamp] = []

    def _worker_python(self) -> Path:
        override = getattr(self.config, "owsm_python_path", None)
        return Path(override) if override else _default_owsm_python()

    def is_available(self) -> bool:
        try:
            python_path = self._worker_python()
        except Exception:
            return False
        if not python_path.is_file():
            return False
        return _find_snapshot() is not None

    def transcribe(
        self,
        audio: AudioData,
        language: str | None = None,
    ) -> TranscriptionResult:
        raise NotImplementedError(
            "OwsmEngine does not support transcription. Use for forced alignment only."
        )

    def align(
        self,
        audio: AudioData,
        lyrics: list[LyricLine],
        language: str | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[SyncResult]:
        if not lyrics:
            raise AlignmentError("no lyric lines to align")

        python_path = self._worker_python()
        if not python_path.is_file():
            raise EngineNotAvailableError(
                f"OWSM worker interpreter not found: {python_path}. Provision a dedicated "
                f"ESPnet venv there (see module docstring) or set "
                "EVERYRIC_ALIGNMENT_OWSM_PYTHON_PATH."
            )
        if self._snapshot is None:
            self._snapshot = _find_snapshot()
        if self._snapshot is None:
            raise EngineNotAvailableError(
                f"{MODEL_ID} was not found under HF_HOME/TRANSFORMERS_CACHE; download it "
                "before using the owsm engine"
            )

        if progress_callback:
            progress_callback(1, 4)

        lines = [line.text for line in lyrics]
        lang_sym = LANGUAGE_SYMBOLS.get(_base_language(language or ""), DEFAULT_LANGUAGE_SYMBOL)
        dtype = str(getattr(self.config, "owsm_dtype", None) or DEFAULT_DTYPE)

        with tempfile.TemporaryDirectory(prefix="owsm_align_") as tmp:
            from everyric2.audio.loader import AudioLoader

            loader = AudioLoader()
            prepared = loader.prepare_for_alignment(audio, target_sr=16000, normalize=True)
            audio_path = Path(tmp) / "audio.wav"
            prepared.to_file(audio_path)

            if progress_callback:
                progress_callback(2, 4)

            payload = {
                "vocals_path": str(audio_path),
                "lines": lines,
                "lang_sym": lang_sym,
                "snapshot": str(self._snapshot),
                "overlap_sec": ALIGN_CHUNK_OVERLAP_SEC,
                "dtype": dtype,
            }
            result = self._run_worker(payload)

        if progress_callback:
            progress_callback(3, 4)

        results, word_timestamps = _build_sync_results(lyrics, lines, result)
        self._last_word_timestamps = word_timestamps

        if progress_callback:
            progress_callback(4, 4)

        return results

    def _run_worker(self, payload: dict[str, Any]) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="owsm_worker_") as tmp:
            request = Path(tmp) / "request.json"
            response = Path(tmp) / "response.json"
            request.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            command = [
                str(self._worker_python()),
                str(_WORKER_SCRIPT),
                "--request",
                str(request),
                "--response",
                str(response),
            ]
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=WORKER_TIMEOUT_SEC,
                env=self._worker_env(),
            )
            if completed.returncode != 0 or not response.is_file():
                raise AlignmentError(
                    f"owsm worker failed ({completed.returncode}):\n"
                    f"stdout:\n{(completed.stdout or '')[-4000:]}\n"
                    f"stderr:\n{(completed.stderr or '')[-4000:]}"
                )
            return json.loads(response.read_text(encoding="utf-8"))

    @staticmethod
    def _worker_env() -> dict[str, str]:
        env = dict(os.environ)
        # 워커가 구조화된 오류와 비ASCII 모델 경로를 찍는다.
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        return env

    def get_last_transcription_data(self) -> tuple[list[WordTimestamp], None, str]:
        return (self._last_word_timestamps, None, "owsm")

    def get_transcription_sets(self) -> list[tuple[list[WordTimestamp], None, str]]:
        data = self.get_last_transcription_data()
        if data[0]:
            return [data]
        return []

    @staticmethod
    def get_engine_type() -> Literal["owsm"]:
        return "owsm"


# 웜 캐시 (2026-08-04, ctc_engine.py의 get_shared_ctc_engine과 같은 관례 — EVERYRIC_SERVER_
# WARM_MODELS로 켜고 끈다). config가 다르면 새로 만든다 — 이 엔진이 self.config에서 실제로
# 읽는 값은 owsm_python_path(어느 인터프리터를 서브프로세스로 부를지)와 owsm_dtype(정밀도)
# 뿐이다(__init__·_worker_python·align 본문 확인) — 그래서 그 둘만 키로 삼는다. 나머지
# AlignmentSettings 필드는 이 엔진의 적재·실행에 영향이 없다.
#
# **중요한 한계**: 이 래퍼 인스턴스를 재사용해도 실측된 무거운 적재 비용(MoRef 실측
# 9.48초)은 줄지 않는다 — align()이 매 호출마다 subprocess.run으로 격리 venv를 새로
# 띄우고(_run_worker), 그 서브프로세스 **안에서** ESPnet·모델 가중치가 매번 새로
# 로드된다(위 모듈 docstring "서브프로세스 격리" 참고). 이 래퍼가 인스턴스 수준에서 들고
# 있는 상태는 self._snapshot(경로 문자열) 하나뿐이라, 캐싱으로 아끼는 비용은 그 경로
# 탐색뿐이다. 그래도 (a) 다른 세 웜 캐시(ctc·separator·melody)와 같은 관례를 지키고,
# (b) 나중에 상주 서브프로세스(요청마다 새로 안 띄우는 워커)로 바뀔 때 이 접근자 이름을
# 그대로 쓸 수 있도록 래퍼 캐싱 자체는 넣어 둔다 — 실제 적재 비용 절감은 이번 변경
# 범위에 없다(보고 참고).
_shared_owsm_engines: dict[tuple[str | None, str | None], "OwsmEngine"] = {}
_shared_owsm_lock = threading.Lock()


def _owsm_cache_key(config: AlignmentSettings | None) -> tuple[str | None, str | None]:
    owsm_python_path = getattr(config, "owsm_python_path", None) if config else None
    owsm_dtype = getattr(config, "owsm_dtype", None) if config else None
    return (owsm_python_path, owsm_dtype)


def get_shared_owsm_engine(config: AlignmentSettings | None = None) -> "OwsmEngine":
    """웜 캐시된 OwsmEngine 래퍼를 돌려준다. warm_models가 꺼져 있으면 매번 새 인스턴스
    (기존 동작). 실제 적재 비용 절감에 대해서는 위 모듈 수준 주석의 한계 설명 참고."""
    from everyric2.config.settings import get_settings

    if not get_settings().server.warm_models:
        return OwsmEngine(config)
    key = _owsm_cache_key(config)
    with _shared_owsm_lock:
        engine = _shared_owsm_engines.get(key)
        if engine is None:
            engine = OwsmEngine(config)
            _shared_owsm_engines[key] = engine
        else:
            logger.info("warm model reuse: owsm")
        return engine


def clear_shared_owsm_engine() -> None:
    """웜 캐시 전부 해제(VRAM 가드용) — 다음 요청에서 지연 재생성된다."""
    global _shared_owsm_engines
    with _shared_owsm_lock:
        _shared_owsm_engines = {}
