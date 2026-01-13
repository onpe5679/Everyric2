from typing import Literal

from everyric2.alignment.base import BaseAlignmentEngine, EngineNotAvailableError
from everyric2.config.settings import AlignmentSettings, get_settings


EngineType = Literal["whisperx", "mfa", "hybrid", "qwen"]


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

        return engines

    @staticmethod
    def get_best_available_engine(
        config: AlignmentSettings | None = None,
    ) -> BaseAlignmentEngine:
        config = config or get_settings().alignment

        preferred_order = ["hybrid", "whisperx", "mfa", "qwen"]

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
