# Everyric2

GPU 가속 가사 싱크 도구. CTC / WhisperX 지원.

## 엔진 비교 (155초 일본어 오디오 기준)

| 엔진 | 속도 | 일본어 | 특징 |
|------|------|--------|------|
| **CTC** | **5초** | ✅ | GPU 가속, HuggingFace wav2vec2 |
| **WhisperX** | 10초 | ⚠️ | 전사 기반, ~35% match rate |

### 언어별 권장 엔진

| 언어 | 권장 엔진 | 이유 |
|------|----------|------|
| 일본어/한국어/중국어 | `ctc` | HuggingFace 모델로 native 문자 지원, 90배 빠름 |
| 영어/유럽어 | `ctc` | MMS_FA 모델 사용 |

## 설치

```bash
# 기본 설치
git clone https://github.com/onpe5679/Everyric2.git
cd Everyric2
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"

# WhisperX (선택)
pip install git+https://github.com/m-bain/whisperx.git
```

## 사용법

```bash
# 권장: CTC 엔진 (가장 빠름, 일본어/한국어/중국어 지원)
everyric2 sync audio.wav lyrics.txt --engine ctc --language ja --debug

# 전체 파이프라인 (보컬분리 + 번역 + 진단)
everyric2 sync audio.wav lyrics.txt --engine ctc --separate --translate --debug --language ja

# WhisperX (전사 기반)
everyric2 sync audio.wav lyrics.txt --engine whisperx --language ja
```

### 출력 구조

```
output/20260113_212716/
├── output.srt              # 메인 출력
├── output_translated.srt   # 번역본 (--translate 시)
├── audio_vocals.wav        # 분리된 보컬 (--separate 시)
├── lyrics_original.txt     # 원본 가사
├── lyrics_translated_ko.txt # 번역된 가사 (--translate 시)
└── diagnostics.png         # 진단 이미지 (--debug 시)
```

### diagnostics.png 컬럼 구성

| 컬럼 | 조건 | 내용 |
|------|------|------|
| Audio (Original) | 항상 | 원본 오디오 파형 |
| Audio (Vocals) | --separate | 분리된 보컬 파형 |
| Original Lyrics | 항상 | 원본 가사 텍스트 |
| Transcription | ctc/whisperx | 전사 결과 + match % |
| Translated | --translate | 번역된 가사 |
| Synced Output | 항상 | 최종 싱크 결과 |

### diagnostics.png 예시

| Line# | Lyric | Start | End | ctc (match: 98%) |
|-------|-------|-------|-----|------------------|
| 1 | 全部 全部 アンタのせいだ | 0.50 | 2.54 | 全部 全部 アンタのせいだ |
| 2 | 反吐が出るくらいにウザったいわ | 2.68 | 4.78 | 反吐が出るくらいにウザったいわ |

## 아키텍처

> 다이어그램: [diagrams/01_overall_pipeline.mermaid](diagrams/01_overall_pipeline.mermaid)

```
Audio ─→ [Demucs?] ─→ ┌─────────────────────────────┐
                      │     Engine Selection        │
Lyrics ─────────────→ │                             │
                      │  --engine ctc (권장, GPU)   │
                      │  --engine whisperx (GPU)    │
                      └──────────────┬──────────────┘
                                     ↓
                      ┌─────────────────────────────┐
                      │    CTCEngine (ja/ko/en)     │
                      │  HuggingFace wav2vec2-xlsr  │
                      │  or torchaudio MMS_FA       │
                      └──────────────┬──────────────┘
                                     ↓
                              Word Timestamps
                                     ↓
                      ┌─────────────────────────────┐
                      │      LyricsMatcher          │
                      │      98.4% match rate       │
                      └──────────────┬──────────────┘
                                     ↓
                           output.srt + diagnostics.png
```

## 핵심 컴포넌트

### CTCEngine

```python
class CTCEngine(BaseAlignmentEngine):
    """CTC 기반 강제 정렬 엔진 (GPU 가속)"""
    
    # 언어별 모델 매핑
    CJK_MODELS = {
        "ja": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",  # 2341 tokens
        "ko": "kresnik/wav2vec2-large-xlsr-korean",
        "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    }
    
    def align(self, audio, lyrics, language, progress_callback):
        # CJK 언어: HuggingFace wav2vec2 사용
        # 기타: torchaudio MMS_FA 사용
        ...
```

## 설정

### 환경변수

```bash
# .env 파일
GEMINI_API_KEY=AIza...                    # 번역용
EVERYRIC_ALIGNMENT__ENGINE=ctc
EVERYRIC_ALIGNMENT__LANGUAGE=ja
```

### AlignmentSettings

```python
class AlignmentSettings(BaseSettings):
    engine: Literal["whisperx", "qwen", "ctc", "nemo", "gpu-hybrid"] = "ctc"
    language: Literal["auto", "en", "ja", "ko"] = "auto"
    
    whisperx_model: str = "large-v3"
    whisperx_batch_size: int = 16
    whisperx_compute_type: str = "float16"
```

## 문제 해결

### WhisperX 낮은 Match Rate

- `--separate` 옵션으로 보컬 분리
- CTC 엔진 사용 권장 (더 높은 정확도)

## 프로젝트 구조

```
everyric2/
├── cli.py                     # CLI
├── alignment/
│   ├── base.py               # BaseAlignmentEngine
│   ├── ctc_engine.py         # CTC (GPU, HuggingFace wav2vec2)
│   ├── whisperx_engine.py    # WhisperX
│   ├── nemo_engine.py        # NeMo NFA (영어만)
│   ├── gpu_hybrid_engine.py  # CTC + NeMo
│   └── matcher.py            # LyricsMatcher
├── audio/
│   ├── loader.py             # AudioLoader
│   └── separator.py          # Demucs
├── debug/
│   ├── debug_info.py         # DebugInfo
│   └── visualizer.py         # diagnostics.png
├── config/
│   └── settings.py           # 설정
├── output/
│   └── formatters.py         # SRT/ASS/LRC/JSON
└── translation/
    └── translator.py         # Gemini
```

## 테스트

```bash
# 테스트 실행
.venv/bin/pytest tests/ -q

# CTC 엔진 테스트 (권장)
.venv/bin/everyric2 sync ftest1/audio.wav ftest1/lyrics.txt \
  --engine ctc --debug --language ja
```

## 라이선스

MIT License
