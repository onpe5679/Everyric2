# Everyric2

GPU 가속 가사 싱크 도구. CTC / WhisperX 지원 + 번역/발음 표기/자막 분할.

## 주요 기능

- **CTC 강제 정렬**: GPU 가속, 5초 내 처리 (256초 오디오)
- **번역 + 발음 표기**: 일본어 → 한국어 번역 + 로마자 발음
- **다중 출력**: 원본/번역/발음/통합 SRT 동시 생성
- **자막 분할 모드**: Line/Word/Character 단위 선택
- **무성 구간 처리**: 짧은 간격 자동 병합
- **로컬 LLM 지원**: Ollama, LM Studio 등 연동

## 엔진 비교

| 엔진 | 속도 | 일본어 | 특징 |
|------|------|--------|------|
| **CTC** | **5초** | ✅ | GPU 가속, HuggingFace wav2vec2 |
| **WhisperX** | 10초 | ⚠️ | 전사 기반, ~35% match rate |

## 설치

```bash
git clone https://github.com/onpe5679/Everyric2.git
cd Everyric2
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
```

## 사용법

### 기본 사용

```bash
# CTC 엔진 (권장)
everyric2 sync audio.wav lyrics.txt --engine ctc --language ja --debug

# 보컬 분리 포함
everyric2 sync audio.wav lyrics.txt --engine ctc --separate --language ja
```

### 번역 + 발음 표기

```bash
# 번역만
everyric2 sync audio.wav lyrics.txt --translate --debug

# 번역 + 로마자 발음
everyric2 sync audio.wav lyrics.txt --translate --pronunciation --debug

# 번역 톤 설정 (literal, natural, poetic, casual, formal)
everyric2 sync audio.wav lyrics.txt --translate --translate-tone poetic
```

### 로컬 LLM 사용

```bash
# Ollama
everyric2 sync audio.wav lyrics.txt --translate \
  --translate-engine local \
  --translate-api-url http://localhost:11434/v1/chat/completions \
  --translate-model llama3.1

# LM Studio
everyric2 sync audio.wav lyrics.txt --translate \
  --translate-engine local \
  --translate-api-url http://localhost:1234/v1/chat/completions
```

### 자막 분할 모드

```bash
# 줄 단위 (기본)
everyric2 sync audio.wav lyrics.txt --segment-mode line

# 단어 단위 (영어에서 효과적)
everyric2 sync audio.wav lyrics.txt --segment-mode word

# 글자 단위 (노래방 스타일)
everyric2 sync audio.wav lyrics.txt --segment-mode character
```

### 무성 구간 처리

```bash
# 0.5초 미만 간격 병합
everyric2 sync audio.wav lyrics.txt --min-silence-gap 0.5
```

## 출력 구조

`--translate --pronunciation --debug` 사용 시:

```
output/20260115_234824/
├── output.srt                    # 원본 가사
├── output_translated.srt         # 번역만
├── output_pronunciation.srt      # 원본 + 발음
├── output_full.srt               # 원본 + 발음 + 번역
├── audio_original.wav
├── audio_vocals.wav              # (--separate 시)
├── lyrics_original.txt
├── lyrics_translated_ko.txt
├── diagnostics.png               # 시각화
└── debug_info.json
```

### output_full.srt 예시

```srt
1
00:00:18,181 --> 00:00:20,041
ずっと願っていた奇跡は
(zutto negatteita kiseki wa)
간절히 바라던 기적은

2
00:00:20,401 --> 00:00:24,662
一瞬で散ってなくなった
(isshun de chitte nakunatta)
한순간에 흩어져 사라졌어
```

## CLI 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--engine` | 정렬 엔진 (ctc, whisperx) | ctc |
| `--language` | 언어 (auto, en, ja, ko) | auto |
| `--translate` | 번역 활성화 | false |
| `--pronunciation` | 발음 표기 포함 | false |
| `--segment-mode` | 분할 모드 (line, word, character) | line |
| `--min-silence-gap` | 최소 무성 간격 (초) | 0.3 |
| `--translate-engine` | 번역 엔진 (gemini, openai, local) | gemini |
| `--translate-model` | 번역 모델명 | gemini-2.0-flash |
| `--translate-api-url` | 로컬 LLM API URL | - |
| `--translate-tone` | 번역 톤 | natural |
| `--separate` | 보컬 분리 | false |
| `--debug` | 디버그 파일 저장 | false |

## 아키텍처

> 다이어그램: [diagrams/01_overall_pipeline.mermaid](diagrams/01_overall_pipeline.mermaid)

```
Audio ─→ [Demucs?] ─→ CTC Engine ─→ Word Timestamps
                           │
Lyrics ────────────────────┘
                           ↓
                    LyricsMatcher
                           ↓
                    SilenceHandler ─→ 짧은 간격 병합
                           ↓
                    Translator ─→ 번역 + 발음 (optional)
                           ↓
                    SegmentationProcessor ─→ Line/Word/Char
                           ↓
                    MultiOutputGenerator
                           ↓
        ┌──────────┬───────────┬────────────┐
        ↓          ↓           ↓            ↓
   output.srt  _translated  _pronunciation  _full
```

## 설정

### 환경변수

```bash
# .env 파일
GEMINI_API_KEY=AIza...
OPENAI_API_KEY=sk-...

# 번역 설정
EVERYRIC_TRANSLATE__ENGINE=gemini
EVERYRIC_TRANSLATE__TONE=natural
EVERYRIC_TRANSLATE__API_URL=http://localhost:11434/v1/chat/completions

# 분할 설정
EVERYRIC_SEGMENT__MODE=line
EVERYRIC_SEGMENT__MIN_SILENCE_GAP=0.3
```

## 프로젝트 구조

```
everyric2/
├── cli.py                     # CLI
├── alignment/
│   ├── base.py               # BaseAlignmentEngine
│   ├── ctc_engine.py         # CTC (GPU, HuggingFace wav2vec2)
│   ├── whisperx_engine.py    # WhisperX
│   ├── matcher.py            # LyricsMatcher
│   ├── segmentation.py       # SegmentationProcessor (NEW)
│   └── silence.py            # SilenceHandler (NEW)
├── audio/
│   ├── loader.py             # AudioLoader
│   └── separator.py          # Demucs
├── debug/
│   ├── debug_info.py         # DebugInfo
│   └── visualizer.py         # diagnostics.png
├── config/
│   └── settings.py           # Settings (TranslationSettings, SegmentationSettings)
├── output/
│   ├── formatters.py         # SRT/ASS/LRC/JSON
│   └── multi_output.py       # MultiOutputGenerator (NEW)
└── translation/
    └── translator.py         # BaseTranslator, GeminiTranslator, OpenAICompatibleTranslator
```

## 테스트

```bash
# 기본 테스트
everyric2 sync ftest1/audio.wav ftest1/lyrics.txt --engine ctc --language ja --debug

# 전체 기능 테스트
everyric2 sync ftest1/audio.wav ftest1/lyrics.txt \
  --engine ctc --language ja \
  --translate --pronunciation \
  --min-silence-gap 0.5 \
  --debug
```

## 라이선스

MIT License
