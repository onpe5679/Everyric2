"""2모드 라우팅 정렬기 — 기본은 무분리 ASR, 붕괴가 의심될 때만 무거운 구원 경로.

## 왜 라우팅인가

UST 17곡 실측에서 **정상곡은 분리기가 사실상 무의미하다**(무분리 × omniasr@kana 79.9/88.3 vs
polar × omniasr@kana 80.2/89.3 — 라인 0.3pp, 음절 1.0pp 차이). 반면 극한곡에서는 결정적이다
(47.6 → 74.3, +26.7pp). 즉 정상곡에 분리기를 돌리는 14초는 순수 낭비이고, 그 14초는
극한곡에서만 값을 한다. 곡마다 필요한 만큼만 쓰면 평균 비용이 크게 떨어진다.

## 판정 신호

``quality_score``로는 안 된다. 그 값은 ``exp(mean_log_score)``라 지수 압축이 심해서 극한곡
(-12 ~ -13.8)과 정상곡 절반(-11.8 ~ -13)이 똑같이 ``0.0000``으로 뭉개진다. 실측에서 정상곡
0.0000~0.0277 / 극한곡 0.0000~0.0003으로 구간이 겹쳐 임계값을 놓을 자리가 없었다.

**로그 영역으로 되돌리면 갈린다.** 라인별 confidence에 log를 취해 곡 단위 중앙값을 보면:

    극한 5곡   -13.82 ~ -12.02   (熱異常 · 토스트 · 소실 · 시니컬 · 루프더룸)
    정상 12곡  -11.80 ~  -4.36

겹치는 곡이 하나도 없다. 정렬 결과의 다른 자기 일관성 지표(세그 duration·무음비·라인밀도·
커버율)는 전부 구간이 겹쳐 실패했으므로, 지금까지 찾은 유일한 판정 신호다.

## 임계값을 보수적으로 두는 이유

실측 마진은 0.22(-12.02 ↔ -11.80)뿐이고 표본이 17곡이다. 새 곡이 그 사이에 떨어질 여지가
크므로 임계를 정상곡 쪽으로 올려 잡는다 — **극한곡을 놓치는 비용(붕괴 방치)이 정상곡을
구원 경로로 잘못 보내는 비용(몇 초 낭비)보다 훨씬 크기 때문이다.** 기본 -11.5는 실측
정상곡 중 아래 두 곡(みむかゥわ·エイリアン², 둘 다 -11.80)까지 구원으로 넘긴다.

## 구조상 전제

``--separators nosep``으로 돌려야 한다. 그래야 ``vocals_path``가 원곡 믹스(AudioLoader 디코드)
이고, 구원 경로가 그 원본에서 분리를 시작할 수 있다. 분리 스템은 하네스의 기존 캐시
(``benchmark/stems/<분리기>/<video_id>/vocals.wav``)를 그대로 재사용한다.
"""

from __future__ import annotations

import logging
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.benchmark_alignment import AlignerAdapter, AlignOut

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
STEMS_ROOT = REPO_ROOT / "benchmark" / "stems"

# 라인 confidence가 0으로 반올림된 줄(6자리 저장)의 로그 바닥값. 실제로는 이보다 낮지만
# -inf를 중앙값에 섞으면 통계가 무너지므로 바닥으로 잡는다.
LOG_FLOOR = math.log(1e-6)


@dataclass(frozen=True)
class RouteConfig:
    name: str
    fast_aligner: str
    rescue_separator: str
    rescue_aligner: str
    # 라인 logConf 중앙값이 이 값 **미만**이면 구원 경로로 보낸다.
    threshold: float
    note: str = ""
    # 언어별 구원 경로 재정의: (언어 접두사, 정렬기, 분리기 or None).
    # 분리기가 None이면 분리 없이 구원한다.
    rescue_by_language: tuple[tuple[str, str, str | None], ...] = ()
    # **신호를 아예 묻지 않고** 구원으로 보낼 언어. logConf가 그 언어에서 붕괴 탐지기로
    # 작동하지 않을 때만 쓴다 — 지금은 en이 그렇다(아래 EN_FORCED_NOTE 참고).
    force_rescue_languages: tuple[str, ...] = ()


# en을 신호가 아니라 언어로 강제 구원하는 근거(5곡 실측, 오입력 확정 곡 제외):
#   Madeon  무분리 MAE 6.57 / PCO 0.24  →  분리 0.07 / 0.63   ← 분리가 고친다
#   나머지 4곡은 무분리·분리 모두 MAE 0.1 안팎
# Madeon의 logConf가 −6.38로 임계(−11.0)에서 아주 멀다. 라틴 가사는 posterior가 구조적으로
# 높아 **틀려도 확신에 차 있어서**, 이 신호로는 원리상 잡히지 않는다. 그래서 언어로 우회한다.
#
# 구원 경로는 **분리 + asr 자기앵커 2패스**다. owsm 앵커를 쓰던 직전 구성보다 정렬이 12배
# 빠르면서(18.1초 → 1.5초) MAE는 절반이다(0.14 → 0.08). 창 자체는 값을 하지만(창 없는 단독
# 대비 PCO +0.02) 그 창을 owsm이 잡을 이유가 없다 — asr이 1패스로 잡으면 된다.
EN_FORCED_NOTE = "en은 신호 무시 강제 구원 — polar 분리 + asr 자기앵커 창 + IPA 정렬"


CONFIGS: tuple[RouteConfig, ...] = (
    # 사용자 확정 구성 — 기본 무분리 ASR, 구원은 polar 분리 + owsm 앵커 2패스.
    RouteConfig(
        name="routed-2mode",
        fast_aligner="omniasr-ctc",
        rescue_separator="bs-polarformer-fp16",
        rescue_aligner="2pass-owsm-omniasr",
        threshold=-11.5,
        note="무분리 omniASR → (붕괴 의심 시) polar 분리 + owsm 앵커 2패스",
    ),
    # ★임계 상향판. 전곡 74곡 감사에서 임계 ±1.0 구간에 16곡이 몰려 있고, 미검출 후보
    # 상위 셋이 전부 −11.08 ~ −11.25로 −11.5 바로 위였다. −11.0으로 올리면 그 셋이 구원으로
    # 가고 비용은 평균 4.0초 → 4.9초(+0.9초)뿐이다 — 여전히 구 스택 12.3초의 절반 이하다.
    # 오판 비대칭(극한곡 놓침 = 붕괴 방치 ≫ 정상곡 오분류 = 몇 초)을 감안하면 이쪽이 맞다.
    RouteConfig(
        name="routed-2mode-safe",
        fast_aligner="omniasr-ctc",
        rescue_separator="bs-polarformer-fp16",
        rescue_aligner="2pass-owsm-omniasr",
        threshold=-11.0,
        note="무분리 omniASR → (붕괴 의심 시) polar 분리 + owsm 앵커 2패스 · 임계 상향판",
    ),
    # ★언어 배선판 — 임계 −11.0에 **en 강제 구원**을 얹었다. 신호가 en에서 작동하지 않는다는
    # 것이 6곡 실측으로 확인됐으므로(EN_FORCED_NOTE), 그 언어만 신호를 묻지 않는다. en 구원은
    # 분리를 빼고 2패스만 한다 — 위 실측이 이미 무분리에서 나온 값이라 polar 10초의 근거가 없다.
    RouteConfig(
        name="routed-2mode-lang",
        fast_aligner="omniasr-ctc",
        rescue_separator="bs-polarformer-fp16",
        rescue_aligner="2pass-owsm-omniasr",
        threshold=-11.0,
        rescue_by_language=(("en", "2pass-asr-ipa-hangul", "bs-polarformer-fp16"),),
        force_rescue_languages=("en",),
        note="무분리 omniASR → (붕괴 의심) polar+owsm 2패스 · en은 polar 분리 + asr 자기앵커 IPA",
    ),
    # 표시를 가나로 내는 변형. 정렬 결과는 위와 완전히 같고 세그 텍스트만 다르다.
    RouteConfig(
        name="routed-2mode-lang-kana",
        fast_aligner="omniasr-ctc",
        rescue_separator="bs-polarformer-fp16",
        rescue_aligner="2pass-owsm-omniasr",
        threshold=-11.0,
        rescue_by_language=(("en", "2pass-asr-ipa-kana", "bs-polarformer-fp16"),),
        force_rescue_languages=("en",),
        note="routed-2mode-lang과 동일 · en 표시만 가나",
    ),
    # 구원을 앵커 없이 분리만으로 하는 경량 변형. 극한곡 라인 74.3(2패스 80.8)로 6.5pp
    # 낮지만 구원 비용이 26.8초 → 15.6초다. 앵커 값어치를 라우팅 위에서 재보는 대조군.
    RouteConfig(
        name="routed-2mode-nosanchor",
        fast_aligner="omniasr-ctc",
        rescue_separator="bs-polarformer-fp16",
        rescue_aligner="omniasr-ctc",
        threshold=-11.5,
        note="무분리 omniASR → (붕괴 의심 시) polar 분리 + omniASR (앵커 없음)",
    ),
)


def line_log_conf_median(lines: list[dict[str, Any]]) -> float | None:
    """곡 단위 라우팅 점수 — 라인 confidence를 로그로 되돌린 중앙값.

    평균이 아니라 중앙값인 이유는 곡 앞뒤의 몇 줄(인트로 애드립·페이드아웃)이 통째로
    바닥값을 찍는 일이 흔해서다. 그 줄들이 평균을 끌어내리면 멀쩡한 곡이 구원으로 샌다.
    """
    values = [
        math.log(line["confidence"]) if line["confidence"] > 0 else LOG_FLOOR
        for line in lines
        if line.get("confidence") is not None
    ]
    return statistics.median(values) if values else None


def _resolve_aligner(name: str) -> Any:
    from scripts.bench_adapters.two_pass import _resolve_adapter as resolve_component

    try:
        return resolve_component(name)
    except ValueError:
        pass
    # 2패스 조합은 two_pass 자신의 레지스트리에 있다.
    from scripts.bench_adapters.two_pass import register as register_two_pass

    registry: dict[str, Any] = {}
    register_two_pass(registry)
    adapter = registry.get(name)
    if adapter is None:
        raise ValueError(f"라우팅 구성이 가리키는 정렬기가 없다: {name!r}")
    return adapter()


def _resolve_separator(name: str) -> Any:
    """분리기 이름 → 인스턴스. 각 분리기 모듈의 ``register``를 직접 재사용한다.

    ``_register_optional_aligners``를 부르지 않는다 — 이 모듈 자신이 그 함수에서 배선되므로
    재귀가 된다. 분리기 모듈만 골라 직접 등록하면 그 고리가 생기지 않는다.
    """
    import importlib

    from scripts.benchmark_alignment import SEPARATORS

    registry: dict[str, Any] = dict(SEPARATORS)
    for module_name in ("separators_roformer", "separators_quality", "separators_light"):
        try:
            importlib.import_module(f"scripts.bench_adapters.{module_name}").register(registry)
        except Exception as exc:
            logger.debug("분리기 모듈 %s 배선 실패: %r", module_name, exc)
    adapter = registry.get(name)
    if adapter is None:
        raise ValueError(f"라우팅 구성이 가리키는 분리기가 없다: {name!r}")
    return adapter()


class RoutedAligner(AlignerAdapter):
    """무분리 1차 → 붕괴 의심 시 분리+앵커 구원."""

    name: str = ""
    config: RouteConfig

    def __init__(self, config: RouteConfig | None = None) -> None:
        if config is None:
            config = self.config
        self.config = config
        self.name = config.name
        self._fast: Any | None = None
        self._rescue: dict[str, Any] = {}
        self._separator: dict[str, Any] = {}

    def align(self, vocals_path: Path, lyrics: str, language: str) -> AlignOut:
        started = time.perf_counter()
        lang = (language or "").lower()
        aligner_name, separator_name = self._rescue_plan(lang)
        forced = any(lang.startswith(p) for p in self.config.force_rescue_languages)

        fast: AlignOut | None = None
        score: float | None = None
        if not forced:
            fast = self._fast_aligner().align(vocals_path, lyrics, language)
            score = line_log_conf_median(fast.lines)
            if score is not None and score >= self.config.threshold:
                fast.elapsed_sec = round(time.perf_counter() - started, 2)
                fast.meta = {**fast.meta, "routing": self._routing_meta("fast", score, None)}
                return fast

        if separator_name is None:
            rescue_vocals, separation_sec, cached = vocals_path, None, False
        else:
            rescue_vocals, separation_sec, cached = self._rescue_stems(vocals_path, separator_name)
        rescue = self._rescue_aligner(aligner_name).align(rescue_vocals, lyrics, language)
        rescue.elapsed_sec = round(time.perf_counter() - started, 2)
        rescue.meta = {
            **rescue.meta,
            "routing": self._routing_meta(
                "forced" if forced else "rescue", score, separation_sec, cached,
                aligner_name, separator_name,
            ),
        }
        if fast is not None:
            rescue.meta["fast_meta"] = fast.meta
            # 카드가 지는 부담은 두 경로의 최댓값이다(순차 실행).
            rescue.vram_peak_mb = _max_or_none(rescue.vram_peak_mb, fast.vram_peak_mb)
            rescue.vram_device_peak_mb = _max_or_none(
                rescue.vram_device_peak_mb, fast.vram_device_peak_mb
            )
        return rescue

    def _rescue_plan(self, language: str) -> tuple[str, str | None]:
        """이 언어의 구원 경로 — (정렬기, 분리기 or None). 재정의가 없으면 기본 구성."""
        for prefix, aligner, separator in self.config.rescue_by_language:
            if language.startswith(prefix):
                return aligner, separator
        return self.config.rescue_aligner, self.config.rescue_separator

    def _routing_meta(
        self,
        route: str,
        score: float | None,
        separation_sec: float | None,
        cached: bool = False,
        aligner: str | None = None,
        separator: str | None = None,
    ) -> dict[str, Any]:
        return {
            "route": route,
            "line_log_conf_median": None if score is None else round(score, 3),
            "threshold": self.config.threshold,
            "fast_aligner": self.config.fast_aligner,
            "rescue_separator": separator,
            "rescue_aligner": aligner or self.config.rescue_aligner,
            "rescue_separation_sec": separation_sec,
            "rescue_stems_cached": cached,
            "note": self.config.note,
        }

    def _rescue_stems(self, vocals_path: Path, separator_name: str) -> tuple[Path, float | None, bool]:
        """구원 경로용 분리 스템 — 하네스 캐시를 재사용하고, 없으면 그 자리에서 만든다.

        ``vocals_path``는 ``benchmark/stems/nosep/<video_id>/vocals.wav``이므로 부모
        디렉터리 이름이 곧 video_id다. 같은 규약으로 구원 분리기의 캐시 자리를 찾는다.
        """
        video_id = vocals_path.parent.name
        from scripts.benchmark_alignment import fs_name

        stem_dir = STEMS_ROOT / fs_name(separator_name) / video_id
        cached_vocals = stem_dir / "vocals.wav"
        if cached_vocals.is_file():
            return cached_vocals, None, True

        source = _original_audio(video_id) or vocals_path
        separator = self._separator_adapter(separator_name)
        started = time.perf_counter()
        out = separator.separate(source, stem_dir)
        return out.vocals_path, round(time.perf_counter() - started, 2), False

    def _fast_aligner(self) -> Any:
        if self._fast is None:
            self._fast = _resolve_aligner(self.config.fast_aligner)
        return self._fast

    def _rescue_aligner(self, name: str) -> Any:
        # 언어마다 구원 정렬기가 다를 수 있어 이름별로 캐시한다(모델 로딩이 비싸다).
        if name not in self._rescue:
            self._rescue[name] = _resolve_aligner(name)
        return self._rescue[name]

    def _separator_adapter(self, name: str) -> Any:
        if name not in self._separator:
            self._separator[name] = _resolve_separator(name)
        return self._separator[name]


def _original_audio(video_id: str) -> Path | None:
    """벤치 오디오 디렉터리에서 원곡 파일을 찾는다.

    구원 분리는 **원본**에서 시작해야 한다. 무분리 경로가 넘겨준 wav는 AudioLoader가 이미
    24kHz 모노로 떨어뜨린 것이라, 그걸 분리기에 다시 먹이면 기존 polar 스템과 조건이 달라진다.
    """
    audio_dir = REPO_ROOT / "benchmark" / "audio"
    for path in sorted(audio_dir.glob(f"{video_id}.*")):
        if path.suffix.lower() in {".m4a", ".mp3", ".wav", ".opus", ".webm", ".flac"}:
            return path
    return None


def _max_or_none(*values: float | None) -> float | None:
    present = [v for v in values if v is not None]
    return max(present) if present else None


def register(aligner_registry: dict[str, type[AlignerAdapter]]) -> None:
    for spec in CONFIGS:
        aligner_registry[spec.name] = _config_class(spec)


def _config_class(spec: RouteConfig) -> type[RoutedAligner]:
    class ConfiguredRoutedAligner(RoutedAligner):
        name = spec.name
        config = spec

    ConfiguredRoutedAligner.__name__ = "Routed_" + spec.name.replace("-", "_")
    ConfiguredRoutedAligner.__qualname__ = ConfiguredRoutedAligner.__name__
    return ConfiguredRoutedAligner
