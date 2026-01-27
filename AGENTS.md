# AGENTS.md - Everyric2 Development Guide

> This file is for AI coding agents working on the everyric2 codebase.

SPEC_md 의 내용을 구현하고 있어 변경된 파일들을 검토하자. 구현이 잘 됐는지 누락된 건 없는지 개선할 부분은 없는지 검토하자. 크리티컬한 이슈는 없는지 검토하자. 너무 과도하게 구현된 부분은 없는지 검토하자. 너무 긴 함수나 파일이 있다면 적절히 나누자. 설계 자체에 문제가 있는지도 검토하자. 전체적으로 꼼꼼히 검토하고 수정해야 할 부분이 있다면 수정해줘. 결정해야 할 부분이 있다면 크리티컬한 부분을 제외하고는 깊게 생각한 후 최적의 방안으로 진행해.

코딩을 시작하기 전에, 문제가 복잡하거나 명확하지 않다면 아래 항목을 포함한 Problem 1-Pager를 작성한 뒤 진행해. 불분명한 항목이 있다면 인터뷰를 요청해서 내용을 명확히 하도록 해.

- 배경(Background): 변경이 필요한 맥락과 동기
- 문제(Problem): 우리가 해결하려는 이슈는 무엇인가?
- 목표(Goal): 성공의 기준(성공한 상태)은 무엇인가?
- 비목표(Non-goals): 명확히 범위 밖(스코프 아웃)인 것은 무엇인가?
- 제약(Constraints): 반드시 준수해야 할 기술적/비즈니스적 제약사항

## Project Overview

**everyric2** is a lyrics synchronization tool that aligns audio with lyrics using various alignment engines (CTC, WhisperX, NeMo, SOFA). It supports multiple languages (English, Japanese, Korean, Chinese) and outputs synchronized subtitles (SRT, ASS, LRC).

## Build & Development Commands

```bash
# Activate virtual environment
source .venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Install with all extras (separator, alignment)
pip install -e ".[all]"
```

## Test Commands

```bash
# Run all tests
pytest

# Run single test file
pytest tests/test_matcher.py

# Run single test function
pytest tests/test_matcher.py::test_normalize_text_english

# Run with verbose output
pytest -v tests/test_matcher.py

# Run with coverage
pytest --cov=everyric2

# Run tests matching pattern
pytest -k "normalize"
```

## Lint & Format Commands

```bash
# Lint with ruff
ruff check everyric2/

# Auto-fix lint issues
ruff check --fix everyric2/

# Format code
ruff format everyric2/

# Type check with mypy
mypy everyric2/
```

## CLI Usage (for testing)

```bash
# Basic sync
everyric2 sync audio.wav lyrics.txt --engine ctc

# With vocal separation and debug
everyric2 sync audio.wav lyrics.txt --engine ctc --separate --debug

# With translation and pronunciation
everyric2 sync audio.wav lyrics.txt --engine ctc --translate --pronunciation

# Different segment modes
everyric2 sync audio.wav lyrics.txt --segment-mode word
everyric2 sync audio.wav lyrics.txt --segment-mode character
```

## Code Style Guidelines

### Python Version
- Target: Python 3.10+
- Use modern syntax: `list[str]` not `List[str]`, `str | None` not `Optional[str]`

### Imports (ruff I)
```python
# Standard library first, then third-party, then local
import logging
from pathlib import Path
from typing import Callable, Literal

import torch
import torchaudio

from everyric2.alignment.base import BaseAlignmentEngine
from everyric2.config.settings import AlignmentSettings
```

### Formatting
- Line length: 100 characters (soft limit, 120 hard)
- Use double quotes for strings
- Trailing commas in multi-line collections

### Type Hints
- Always use type hints for function signatures
- Use `| None` instead of `Optional[]`
- Use `Literal["a", "b"]` for string enums
- Use dataclasses for data structures

```python
def align(
    self,
    audio: AudioData,
    lyrics: list[LyricLine],
    language: str | None = None,
) -> list[SyncResult]:
```

### Naming Conventions
- Classes: `PascalCase` (e.g., `CTCEngine`, `SyncResult`)
- Functions/methods: `snake_case` (e.g., `align_lyrics`, `get_engine`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `LANG_MODEL_MAP`)
- Private methods: `_leading_underscore` (e.g., `_ensure_model_loaded`)

### Error Handling
- Use custom exceptions from `everyric2.alignment.base`:
  - `AlignmentError` - general alignment failures
  - `EngineNotAvailableError` - missing dependencies
- Log errors with context before raising

```python
logger = logging.getLogger(__name__)

try:
    result = self._process(audio)
except Exception as e:
    logger.error(f"Processing failed: {e}")
    raise AlignmentError(f"Failed to process audio: {e}") from e
```

### Engine Pattern

All alignment engines inherit from `BaseAlignmentEngine`:

```python
class SOFAEngine(BaseAlignmentEngine):
    def __init__(self, config: AlignmentSettings | None = None):
        super().__init__(config)
    
    def is_available(self) -> bool:
        """Check if engine dependencies are installed."""
        pass
    
    def align(
        self,
        audio: AudioData,
        lyrics: list[LyricLine],
        language: str | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[SyncResult]:
        """Main alignment method."""
        pass
    
    def transcribe(self, audio: AudioData, language: str | None = None) -> TranscriptionResult:
        """Transcribe audio (optional for forced alignment engines)."""
        pass
    
    @staticmethod
    def get_engine_type() -> Literal["sofa"]:
        return "sofa"
```

Register new engines in `everyric2/alignment/factory.py`.

## Project Structure

```
everyric2/
├── alignment/          # Alignment engines (CTC, WhisperX, NeMo, SOFA)
│   ├── base.py         # BaseAlignmentEngine abstract class
│   ├── factory.py      # EngineFactory for engine selection
│   ├── ctc_engine.py   # CTC forced alignment
│   └── matcher.py      # LyricsMatcher for text matching
├── audio/              # Audio loading and processing
├── config/             # Settings and configuration
├── debug/              # Diagnostics visualization
├── inference/          # LLM inference and prompts
├── output/             # Output formatters (SRT, ASS, LRC)
├── separator/          # Demucs vocal separation
├── translation/        # Translation services
└── cli.py              # Typer CLI entry point
```

## Key Data Structures

```python
@dataclass
class SyncResult:
    text: str
    start_time: float
    end_time: float
    confidence: float | None = None
    word_segments: list[WordSegment] | None = None
    translation: str | None = None
    pronunciation: str | None = None

@dataclass
class WordTimestamp:
    word: str
    start: float
    end: float
    confidence: float | None = None
```

## DO NOT

- Do NOT use `as any`, `@ts-ignore`, `type: ignore` to suppress errors
- Do NOT commit `.env` files or API keys
- Do NOT modify `torchaudio.functional.forced_align` behavior (deprecated in 2.9)
- Do NOT use blocking I/O in async contexts
- Do NOT hardcode paths - use Path objects and config

## Testing Notes

- Tests use `pytest` with `conftest.py` for fixtures
- Settings are auto-reset between tests via `reset_settings_before_test` fixture
- Mock external services (LLM APIs, model downloads) in tests
