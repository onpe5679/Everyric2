"""Optional adapters for the alignment benchmark harness."""

from .hf_ctc import CANDIDATES, HFCTCAligner, register

__all__ = ["CANDIDATES", "HFCTCAligner", "register"]
