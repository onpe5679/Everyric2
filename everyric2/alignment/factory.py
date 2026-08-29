"""Factory for adaptive-stack anchor engines.

Anchors are implementation details selected by the adaptive router.  The public CLI and CEP
panel do not expose this factory as an engine picker.
"""

from typing import Literal

from everyric2.alignment.base import BaseAlignmentEngine
from everyric2.config.settings import AlignmentSettings, get_settings

EngineType = Literal["adaptive", "owsm", "omniasr"]


class EngineFactory:
    @staticmethod
    def get_engine(
        engine_type: str | None = None,
        config: AlignmentSettings | None = None,
    ) -> BaseAlignmentEngine:
        config = config or get_settings().alignment
        if engine_type in (None, "adaptive"):
            from everyric2.alignment.adaptive_engine import AdaptiveEngine

            return AdaptiveEngine(config)
        if engine_type == "owsm":
            from everyric2.alignment.owsm_engine import get_shared_owsm_engine

            return get_shared_owsm_engine(config)
        if engine_type == "omniasr":
            from everyric2.alignment.omniasr_engine import get_shared_omniasr_engine

            return get_shared_omniasr_engine(config)
        raise ValueError(
            "Adaptive routing selects an internal anchor automatically; "
            f"unsupported anchor: {engine_type!r}"
        )

    @staticmethod
    def get_available_engines() -> list[dict[str, object]]:
        try:
            public_ready = EngineFactory.get_engine("adaptive").is_available()
        except Exception:
            public_ready = False
        return [
            {
                "type": "adaptive",
                "available": public_ready,
                "description": "Automatic local fast/medium/heavy alignment stack",
            }
        ]
