from typing import Literal

from everyric2.alignment.base import BaseAlignmentEngine, EngineNotAvailableError
from everyric2.config.settings import AlignmentSettings, get_settings


EngineType = Literal["whisperx", "mfa", "hybrid", "qwen", "ctc", "nemo", "gpu-hybrid"]


class EngineFactory:
    @staticmethod
    def get_engine(
        engine_type: str | None = None,
        config: AlignmentSettings | None = None,
    ) -> BaseAlignmentEngine:
        config = config or get_settings().alignment
        engine_type = engine_type or config.engine

        if engine_type == "whisperx":
            from everyric2.alignment.whisperx_engine import WhisperXEngine

            engine = WhisperXEngine(config)
        elif engine_type == "mfa":
            from everyric2.alignment.mfa_engine import MFAEngine

            engine = MFAEngine(config)
        elif engine_type == "hybrid":
            from everyric2.alignment.hybrid_engine import HybridEngine

            engine = HybridEngine(config)
        elif engine_type == "qwen":
            from everyric2.alignment.qwen_engine import QwenEngine

            engine = QwenEngine(config)
        elif engine_type == "ctc":
            from everyric2.alignment.ctc_engine import CTCEngine

            engine = CTCEngine(config)
        elif engine_type == "nemo":
            from everyric2.alignment.nemo_engine import NeMoEngine

            engine = NeMoEngine(config)
        elif engine_type == "gpu-hybrid":
            from everyric2.alignment.gpu_hybrid_engine import GPUHybridEngine

            engine = GPUHybridEngine(config)
        else:
            raise ValueError(f"Unknown engine type: {engine_type}")

        return engine

    @staticmethod
    def get_available_engines() -> list[dict]:
        engines = []

        try:
            from everyric2.alignment.whisperx_engine import WhisperXEngine

            engine = WhisperXEngine()
            engines.append(
                {
                    "type": "whisperx",
                    "available": engine.is_available(),
                    "description": "WhisperX with word-level alignment (recommended)",
                }
            )
        except Exception:
            engines.append(
                {
                    "type": "whisperx",
                    "available": False,
                    "description": "WhisperX with word-level alignment (recommended)",
                }
            )

        try:
            from everyric2.alignment.mfa_engine import MFAEngine

            engine = MFAEngine()
            engines.append(
                {
                    "type": "mfa",
                    "available": engine.is_available(),
                    "description": "Montreal Forced Aligner (highest precision)",
                }
            )
        except Exception:
            engines.append(
                {
                    "type": "mfa",
                    "available": False,
                    "description": "Montreal Forced Aligner (highest precision)",
                }
            )

        engines.append(
            {
                "type": "hybrid",
                "available": any(
                    e["available"] for e in engines if e["type"] in ["whisperx", "mfa"]
                ),
                "description": "WhisperX + MFA hybrid (best of both)",
            }
        )

        try:
            from everyric2.alignment.qwen_engine import QwenEngine

            engine = QwenEngine()
            engines.append(
                {
                    "type": "qwen",
                    "available": engine.is_available(),
                    "description": "Qwen-Omni multimodal (legacy)",
                }
            )
        except Exception:
            engines.append(
                {
                    "type": "qwen",
                    "available": False,
                    "description": "Qwen-Omni multimodal (legacy)",
                }
            )

        try:
            from everyric2.alignment.ctc_engine import CTCEngine

            engine = CTCEngine()
            engines.append(
                {
                    "type": "ctc",
                    "available": engine.is_available(),
                    "description": "CTC forced aligner (GPU, fast)",
                }
            )
        except Exception:
            engines.append(
                {
                    "type": "ctc",
                    "available": False,
                    "description": "CTC forced aligner (GPU, fast)",
                }
            )

        try:
            from everyric2.alignment.nemo_engine import NeMoEngine

            engine = NeMoEngine()
            engines.append(
                {
                    "type": "nemo",
                    "available": engine.is_available(),
                    "description": "NeMo NFA (GPU, NVIDIA)",
                }
            )
        except Exception:
            engines.append(
                {
                    "type": "nemo",
                    "available": False,
                    "description": "NeMo NFA (GPU, NVIDIA)",
                }
            )

        engines.append(
            {
                "type": "gpu-hybrid",
                "available": any(e["available"] for e in engines if e["type"] in ["ctc", "nemo"]),
                "description": "CTC + NeMo GPU hybrid (best GPU performance)",
            }
        )

        return engines

    @staticmethod
    def get_best_available_engine(
        config: AlignmentSettings | None = None,
    ) -> BaseAlignmentEngine:
        config = config or get_settings().alignment

        preferred_order = ["gpu-hybrid", "ctc", "nemo", "hybrid", "whisperx", "mfa", "qwen"]

        for engine_type in preferred_order:
            try:
                engine = EngineFactory.get_engine(engine_type, config)
                if engine.is_available():
                    return engine
            except Exception:
                continue

        raise EngineNotAvailableError(
            "No alignment engine available. Install WhisperX: pip install whisperx"
        )
