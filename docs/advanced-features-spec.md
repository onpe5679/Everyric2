# Advanced Features Specification

## Overview

4가지 고급 기능 구현 스펙:

1. **무성 구간 감지 및 병합** - 짧은 간격 자동 처리
2. **발음 표기 (Pronunciation)** - 일본어 로마자 등 발음 전사
3. **자막 분할 모드 (Segmentation)** - Line/Word/Character 단위 선택
4. **번역 엔진 확장** - 톤 조정 + 로컬 LLM 지원

---

## 1. 무성 구간 감지 및 병합 (Silence Handling)

### 문제
- 자막 간 간격이 너무 짧으면 (예: 0.1초) 깜빡이는 느낌
- 사용자가 원하는 최소 간격을 설정할 수 있어야 함

### 해결책

```
[Before]
Line 1: 0.00 - 1.50  "안녕하세요"
Line 2: 1.55 - 3.00  "반갑습니다"  <- 0.05초 간격 (너무 짧음)

[After: min_gap=0.3s]
Line 1: 0.00 - 1.525 "안녕하세요"  <- 간격 제거, 중간점으로 병합
Line 2: 1.525 - 3.00 "반갑습니다"
```

### 설정

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `min_silence_gap` | float | 0.3 | 최소 무성 구간 (초). 이보다 짧으면 병합 |
| `silence_merge_mode` | enum | `midpoint` | 병합 방식: `midpoint`, `extend_prev`, `extend_next` |

### CLI

```bash
everyric2 sync audio.wav lyrics.txt --min-silence-gap 0.5
everyric2 sync audio.wav lyrics.txt --silence-merge-mode extend_prev
```

---

## 2. 발음 표기 (Pronunciation Transcription)

### 문제
- 일본어 가사를 들으면서 따라 부르고 싶은데 한자/히라가나를 못 읽음
- 번역본에 로마자 발음도 함께 표시하고 싶음

### 해결책
- LLM에게 번역과 발음 전사를 동시에 요청
- JSON 형식으로 구조화된 응답 수신

### LLM 프롬프트 (예시)

```
Translate these Japanese lyrics to Korean.
Also provide romanized pronunciation for each line.

Output JSON format:
[
  {"original": "桜が咲く", "translation": "벚꽃이 핀다", "pronunciation": "sakura ga saku"},
  ...
]

LYRICS:
桜が咲く
風が吹く
```

### 출력 파일 조합

| 플래그 | 출력 파일 | 내용 |
|--------|----------|------|
| 기본 | `output.srt` | 원본 가사 |
| `--translate` | `output_translated.srt` | 번역만 |
| `--pronunciation` | `output_pronunciation.srt` | 원본 + 발음 |
| `--translate --pronunciation` | `output_full.srt` | 원본 + 발음 + 번역 |

### 출력 예시: `output_full.srt`

```srt
1
00:00:05,230 --> 00:00:08,450
桜が咲く
(sakura ga saku)
벚꽃이 핀다

2
00:00:08,900 --> 00:00:12,100
風が吹く
(kaze ga fuku)
바람이 분다
```

### 설정

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `include_pronunciation` | bool | false | 발음 표기 포함 여부 |
| `pronunciation_format` | enum | `parentheses` | 표시 형식: `parentheses`, `brackets`, `newline` |

---

## 3. 자막 분할 모드 (Segmentation Mode)

### 문제
- 긴 가사를 오래 표시할지, 짧게 쪼개서 빠르게 넘길지 선택하고 싶음
- 노래방 스타일로 글자 단위 타이밍도 가능하게 하고 싶음

### 분할 모드

| 모드 | 설명 | 예시 |
|------|------|------|
| `line` | 줄 단위 (기본) | `"전부 전부 아타의 탓이다"` 전체가 하나의 자막 |
| `word` | 단어/구 단위 | `"전부"`, `"전부"`, `"아타의"`, `"탓이다"` 각각 자막 |
| `character` | 글자 단위 | `"전"`, `"부"`, `"전"`, `"부"` ... 각각 자막 |

### 동작 원리

**전제조건**: CTC 엔진이 단어별 타임스탬프(`WordTimestamp`)를 생성함

```python
# CTCEngine 출력
word_timestamps = [
    WordTimestamp("전부", 0.50, 0.80),
    WordTimestamp("전부", 0.85, 1.15),
    WordTimestamp("아타의", 1.20, 1.60),
    WordTimestamp("탓이다", 1.65, 2.10),
]
```

**Line 모드** (기존):
```
0.50 --> 2.10: "전부 전부 아타의 탓이다"
```

**Word 모드**:
```
0.50 --> 0.80: "전부"
0.85 --> 1.15: "전부"
1.20 --> 1.60: "아타의"
1.65 --> 2.10: "탓이다"
```

**Character 모드** (Word 타임스탬프를 글자 수로 균등 분배):
```
# "전부" (0.50-0.80, 2글자, 0.15초/글자)
0.50 --> 0.65: "전"
0.65 --> 0.80: "부"
# "전부" (0.85-1.15, 2글자, 0.15초/글자)
0.85 --> 1.00: "전"
1.00 --> 1.15: "부"
...
```

### 설정

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `segment_mode` | enum | `line` | 분할 모드: `line`, `word`, `character` |
| `min_segment_duration` | float | 0.2 | 최소 세그먼트 지속 시간 (초) |
| `max_chars_per_segment` | int | 50 | 세그먼트당 최대 글자 수 (line 모드에서 자동 분할) |

### CLI

```bash
everyric2 sync audio.wav lyrics.txt --segment-mode word
everyric2 sync audio.wav lyrics.txt --segment-mode character --min-segment-duration 0.15
```

---

## 4. 번역 엔진 확장 (Translation Engine)

### 문제
- 현재 Gemini API만 지원
- 로컬 LLM (Ollama, LM Studio, vLLM 등)도 사용하고 싶음
- 번역 톤/스타일을 조정하고 싶음

### 해결책: OpenAI-Compatible API 지원

대부분의 로컬 LLM 서버는 OpenAI API 형식을 지원:
- Ollama: `http://localhost:11434/v1/chat/completions`
- LM Studio: `http://localhost:1234/v1/chat/completions`
- vLLM: `http://localhost:8000/v1/chat/completions`

### 번역 엔진 종류

| 엔진 | 설명 | 설정 |
|------|------|------|
| `gemini` | Google Gemini API (기본) | `GEMINI_API_KEY` |
| `openai` | OpenAI API | `OPENAI_API_KEY` |
| `local` | OpenAI-compatible 로컬 서버 | `--translate-api-url` |

### 톤 설정

| 톤 | 설명 | 프롬프트 힌트 |
|----|------|---------------|
| `literal` | 직역 | "Translate literally, preserving original meaning" |
| `natural` | 자연스러운 번역 (기본) | "Translate naturally for Korean speakers" |
| `poetic` | 시적/문학적 | "Translate poetically, maintaining rhythm and beauty" |
| `casual` | 구어체 | "Translate in casual, conversational Korean" |
| `formal` | 격식체 | "Translate in formal, polite Korean" |

### 설정

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `translate_engine` | enum | `gemini` | 번역 엔진 |
| `translate_model` | str | `gemini-2.0-flash` | 모델명 |
| `translate_api_url` | str | null | 로컬 LLM API URL |
| `translate_api_key` | str | null | API 키 (환경변수 우선) |
| `translate_tone` | enum | `natural` | 번역 톤 |
| `translate_temperature` | float | 0.3 | 생성 온도 |

### CLI

```bash
# Gemini (기본)
everyric2 sync audio.wav lyrics.txt --translate

# OpenAI
everyric2 sync audio.wav lyrics.txt --translate --translate-engine openai

# 로컬 LLM (Ollama)
everyric2 sync audio.wav lyrics.txt --translate \
  --translate-engine local \
  --translate-api-url http://localhost:11434/v1/chat/completions \
  --translate-model llama3.1

# 톤 설정
everyric2 sync audio.wav lyrics.txt --translate --translate-tone poetic
```

### 환경변수

```bash
# .env
GEMINI_API_KEY=AIza...
OPENAI_API_KEY=sk-...
EVERYRIC_TRANSLATE__ENGINE=gemini
EVERYRIC_TRANSLATE__TONE=natural
EVERYRIC_TRANSLATE__API_URL=http://localhost:11434/v1/chat/completions
```

---

## 데이터 구조 변경

### SyncResult 확장

```python
@dataclass
class WordSegment:
    word: str
    start: float
    end: float
    confidence: float | None = None


@dataclass
class SyncResult:
    text: str
    start_time: float
    end_time: float
    confidence: float | None = None
    line_number: int | None = None
    
    # New fields
    word_segments: list[WordSegment] | None = None
    translation: str | None = None
    pronunciation: str | None = None
```

### TranslationResult (새 구조)

```python
@dataclass
class TranslationLine:
    original: str
    translation: str
    pronunciation: str | None = None


@dataclass
class TranslationResult:
    lines: list[TranslationLine]
    source_lang: str
    target_lang: str
    engine: str
    tone: str
```

---

## Settings 구조

### TranslationSettings

```python
class TranslationSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EVERYRIC_TRANSLATE_")
    
    engine: Literal["gemini", "openai", "local"] = "gemini"
    model: str = "gemini-2.0-flash"
    api_url: str | None = None
    api_key: str | None = None
    tone: Literal["literal", "natural", "poetic", "casual", "formal"] = "natural"
    temperature: float = 0.3
    include_pronunciation: bool = False
    target_language: str = "ko"
```

### SegmentationSettings

```python
class SegmentationSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EVERYRIC_SEGMENT_")
    
    mode: Literal["line", "word", "character"] = "line"
    min_duration: float = 0.2
    max_chars_per_segment: int = 50
    min_silence_gap: float = 0.3
    silence_merge_mode: Literal["midpoint", "extend_prev", "extend_next"] = "midpoint"
```

---

## 파이프라인 흐름

```
Audio + Lyrics
      │
      ▼
┌─────────────────┐
│  CTC Engine     │ → WordTimestamp[] (단어별 타임스탬프)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LyricsMatcher  │ → SyncResult[] (word_segments 포함)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ SilenceHandler  │ → 짧은 간격 병합
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Translator     │ → translation + pronunciation 추가
│  (optional)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Segmentation    │ → Line/Word/Character 모드 적용
│  Processor      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ MultiOutput     │ → 여러 SRT 파일 생성
│  Generator      │
└─────────────────┘
```

---

## 출력 파일 구조

`--translate --pronunciation --debug` 사용 시:

```
output/20260115_231500/
├── output.srt                    # 원본 가사만
├── output_translated.srt         # 번역만
├── output_pronunciation.srt      # 원본 + 발음
├── output_full.srt               # 원본 + 발음 + 번역
├── output.json                   # 전체 데이터 (구조화)
├── lyrics_original.txt
├── lyrics_translated_ko.txt
├── audio_original.wav
├── audio_vocals.wav              # (--separate 시)
├── diagnostics.png               # 시각화
└── debug_info.json
```

---

## Diagnostics 시각화 변경

### 새 컬럼 추가

| 컬럼 | 조건 | 내용 |
|------|------|------|
| Silence Gaps | 항상 | 무성 구간 표시 (빨간 점선) |
| Word Segments | word/char 모드 | 단어별 경계 표시 |
| Pronunciation | `--pronunciation` | 발음 표기 |

### 색상 코드

- 🔴 빨강: 원본/무성구간
- 🟢 초록: 번역
- 🔵 파랑: 최종 출력
- 🟣 보라: 전사 결과
- 🟡 노랑: 발음 표기

---

## 구현 순서

### Phase 1: 데이터 구조 (기반)
1. `SyncResult` 확장 (word_segments, translation, pronunciation)
2. `TranslationSettings`, `SegmentationSettings` 추가
3. `TranslationResult` 데이터 클래스

### Phase 2: 핵심 모듈
4. `SilenceHandler` - 무성 구간 처리
5. `SegmentationProcessor` - 분할 모드 적용
6. `BaseTranslator` 추상화 + `GeminiTranslator` + `LocalTranslator`

### Phase 3: 통합
7. `LyricsMatcher` 수정 - word_segments 보존
8. `MultiOutputGenerator` - 다중 파일 생성
9. `DiagnosticsVisualizer` 업데이트

### Phase 4: CLI & 테스트
10. CLI 옵션 추가
11. 통합 테스트

---

## 테스트 계획

### Unit Tests

```python
# test_silence_handler.py
def test_merge_short_gaps():
    results = [
        SyncResult("A", 0.0, 1.0),
        SyncResult("B", 1.05, 2.0),  # 0.05s gap - should merge
    ]
    handler = SilenceHandler(min_gap=0.3)
    merged = handler.process(results)
    assert merged[0].end_time == merged[1].start_time

# test_segmentation.py
def test_word_mode():
    result = SyncResult(
        "hello world",
        0.0, 2.0,
        word_segments=[
            WordSegment("hello", 0.0, 0.9),
            WordSegment("world", 1.1, 2.0),
        ]
    )
    processor = SegmentationProcessor(mode="word")
    segments = processor.process([result])
    assert len(segments) == 2
```

### Integration Test

```bash
# 실제 오디오로 전체 파이프라인
everyric2 sync ftest1/audio.wav ftest1/lyrics.txt \
  --engine ctc \
  --language ja \
  --translate \
  --pronunciation \
  --segment-mode word \
  --min-silence-gap 0.5 \
  --debug
```

### 검증 항목

- [ ] 무성 구간이 설정값 미만이면 병합됨
- [ ] 발음 표기가 SRT에 포함됨
- [ ] word 모드에서 단어별 자막 생성됨
- [ ] 로컬 LLM으로 번역 가능
- [ ] diagnostics.png에 새 정보 표시됨
