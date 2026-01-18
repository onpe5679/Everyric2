"""Inference module for Qwen-Omni models."""

from everyric2.inference.prompt import LyricLine, PromptBuilder, SyncResult
from everyric2.inference.qwen_omni import QwenOmniEngine

__all__ = ["QwenOmniEngine", "PromptBuilder", "LyricLine", "SyncResult"]
