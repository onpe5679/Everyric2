from typing import Literal

from everyric2.alignment.base import BaseAlignmentEngine, EngineNotAvailableError
from everyric2.config.settings import AlignmentSettings, get_settings

EngineType = Literal["ctc", "nemo", "gpu-hybrid", "sofa"]


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
