from typing import Literal

from everyric2.alignment.base import BaseAlignmentEngine, EngineNotAvailableError
from everyric2.config.settings import AlignmentSettings, get_settings

EngineType = Literal["ctc", "nemo", "gpu-hybrid", "sofa", "owsm", "omniasr"]


class EngineFactory:
    @staticmethod
    def get_engine(
        engine_type: str | None = None,
        config: AlignmentSettings | None = None,
    ) -> BaseAlignmentEngine:
        config = config or get_settings().alignment
        engine_type = engine_type or config.engine

        if engine_type == "ctc":
            from everyric2.alignment.ctc_engine import CTCEngine

            engine = CTCEngine(config)
        elif engine_type == "nemo":
            from everyric2.alignment.nemo_engine import NeMoEngine

            engine = NeMoEngine(config)
        elif engine_type == "gpu-hybrid":
            from everyric2.alignment.gpu_hybrid_engine import GPUHybridEngine

            engine = GPUHybridEngine(config)
        elif engine_type == "sofa":
            from everyric2.alignment.sofa_engine import SOFAEngine

            engine = SOFAEngine(config)
        elif engine_type == "owsm":
            # 웜 캐시(2026-08-04) — get_shared_owsm_engine 자체가 server.warm_models를
            # 보므로(꺼져 있으면 매번 새 인스턴스, 이전과 동일 동작) 여기서 따로 분기할
            # 필요가 없다. 캐시 접근을 **이 메서드 안에** 둔 이유: worker.py의 잡 실행
            # 경로가 이 메서드를 직접 부르므로, 여기 하나만 캐시를 타면 그 경로가 자동으로
            # 혜택을 받는다. 처음엔 worker.py에 별도 디스패처(_get_shared_new_stack_engine)를
            # 두고 이 메서드를 아예 안 거치게 했는데, tests/test_new_stack_wiring.py 등
            # 여러 파일이 `EngineFactory.get_engine` **자체**를 몽키패치해 라우팅을
            # 검증하는 방식이라(routing/refiner 등 18개 테스트) 그 우회가 주입 지점을
            # 완전히 무력화시켰다(실측 — 전체 스위트 18건 실패로 발견, 2026-08-04). 이
            # 메서드를 owsm/omniasr 캐시의 유일한 진입점으로 되돌려 기존 테스트 주입
            # 지점을 그대로 살린다(테스트 수정 0건 — 이 저장소에 이미 있는 "_run_alignment
            # 시그니처를 테스트 대역 3파일이 미러링한다"는 부담을 더 늘리지 않는 쪽 선택).
            from everyric2.alignment.owsm_engine import get_shared_owsm_engine

            engine = get_shared_owsm_engine(config)
        elif engine_type == "omniasr":
            from everyric2.alignment.omniasr_engine import get_shared_omniasr_engine

            engine = get_shared_omniasr_engine(config)
        else:
            raise ValueError(f"Unknown engine type: {engine_type}")

        return engine

    @staticmethod
    def get_available_engines() -> list[dict]:
        engines = []

        try:
            from everyric2.alignment.ctc_engine import CTCEngine

            engine = CTCEngine()
            engines.append(
                {
                    "type": "ctc",
                    "available": engine.is_available(),
                    "description": "CTC forced aligner (GPU, recommended)",
                }
            )
        except Exception:
            engines.append(
                {
                    "type": "ctc",
                    "available": False,
                    "description": "CTC forced aligner (GPU, recommended)",
                }
            )

        try:
            from everyric2.alignment.nemo_engine import NeMoEngine

            engine = NeMoEngine()
            engines.append(
                {
                    "type": "nemo",
                    "available": engine.is_available(),
                    "description": "NeMo NFA (GPU, English only)",
                }
            )
        except Exception:
            engines.append(
                {
                    "type": "nemo",
                    "available": False,
                    "description": "NeMo NFA (GPU, English only)",
                }
            )

        try:
            from everyric2.alignment.sofa_engine import SOFAEngine

            engine = SOFAEngine()
            engines.append(
                {
                    "type": "sofa",
                    "available": engine.is_available(),
                    "description": "SOFA singing-oriented forced aligner (English/Japanese)",
                }
            )
        except Exception:
            engines.append(
                {
                    "type": "sofa",
                    "available": False,
                    "description": "SOFA singing-oriented forced aligner (English/Japanese)",
                }
            )

        try:
            from everyric2.alignment.owsm_engine import OwsmEngine

            engine = OwsmEngine()
            engines.append(
                {
                    "type": "owsm",
                    "available": engine.is_available(),
                    "description": "OWSM-CTC v4 1B (subprocess-isolated, multilingual)",
                }
            )
        except Exception:
            engines.append(
                {
                    "type": "owsm",
                    "available": False,
                    "description": "OWSM-CTC v4 1B (subprocess-isolated, multilingual)",
                }
            )

        try:
            from everyric2.alignment.omniasr_engine import OmniASREngine

            engine = OmniASREngine()
            engines.append(
                {
                    "type": "omniasr",
                    "available": engine.is_available(),
                    "description": "omniASR-CTC-300M (in-process, multilingual)",
                }
            )
        except Exception:
            engines.append(
                {
                    "type": "omniasr",
                    "available": False,
                    "description": "omniASR-CTC-300M (in-process, multilingual)",
                }
            )

        return engines

    @staticmethod
    def get_best_available_engine(
        config: AlignmentSettings | None = None,
    ) -> BaseAlignmentEngine:
        config = config or get_settings().alignment

        preferred_order = ["ctc", "nemo", "sofa"]

        for engine_type in preferred_order:
            try:
                engine = EngineFactory.get_engine(engine_type, config)
                if engine.is_available():
                    return engine
            except Exception:
                continue

        raise EngineNotAvailableError(
            "No alignment engine available. Install: pip install transformers torchaudio"
        )
