"""Inference module for Qwen-Omni models."""

from everyric2.inference.qwen_omni import QwenOmniEngine
from everyric2.inference.prompt import PromptBuilder, LyricLine, SyncResult

__all__ = ["QwenOmniEngine", "PromptBuilder", "LyricLine", "SyncResult"]
