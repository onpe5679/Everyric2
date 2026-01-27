# GPU Accelerated Alignment Engines Specification

## Overview

MFA(Montreal Forced Aligner)의 성능 한계를 극복하기 위해 GPU 가속 forced alignment 엔진을 추가한다.

### 현재 문제점
- MFA: 155초 오디오 → 400초 처리 (CPU 단일 코어만 사용)
- 32코어 CPU, RTX 5090이 있지만 활용 못함

### 해결 방안
GPU 기반 CTC forced alignment 엔진 2종 추가:
1. **ctc-forced-aligner**: 경량, 빠른 설치
2. **NeMo NFA**: NVIDIA 공식, 높은 정확도

---

## Architecture

> 다이어그램: [../diagrams/09_ctc_engine.mermaid](../diagrams/09_ctc_engine.mermaid)

```
Audio 16kHz ─→ ┌─────────────────────────────────┐
               │        Language Selection       │
               │                                 │
               │  ja → HuggingFace wav2vec2-xlsr │
               │  ko → HuggingFace wav2vec2-xlsr │
               │  en → torchaudio MMS_FA         │
               └────────────────┬────────────────┘
                                ↓
               ┌─────────────────────────────────┐
Lyrics ──────→ │     torchaudio.forced_align     │
               │           (GPU CUDA)            │
               └────────────────┬────────────────┘
                                ↓
                         Token Spans
                                ↓
                      Word Timestamps (96 words)
                                ↓
               ┌─────────────────────────────────┐
               │        LyricsMatcher            │
               │       98.4% match rate          │
               └────────────────┬────────────────┘
                                ↓
                     SyncResult (61 lines)
```

### CTC 언어별 모델 매핑

| Language | Model | Vocabulary |
|----------|-------|------------|
| ja | `jonatasgrosman/wav2vec2-large-xlsr-53-japanese` | 2341 tokens (한자+가나) |
| ko | `kresnik/wav2vec2-large-xlsr-korean` | 한글 자모 |
| en/other | `torchaudio.pipelines.MMS_FA` | a-z 26자 |

---

## Engine Comparison

| Engine | Backend | GPU | Speed | Japanese | Install |
|--------|---------|-----|-------|----------|---------|
| MFA | Kaldi | ❌ | 454s | ✅ | conda |
| CTC | Wav2Vec2 | ✅ | **5s** | ✅ | pip |
| NeMo | Conformer | ✅ | ~40s | ❌ | pip |
| WhisperX | Whisper | ✅ | ~10s | ⚠️ | pip |

### Language Support

| Engine | Japanese | Korean | Chinese | English | Others |
|--------|----------|--------|---------|---------|--------|
| MFA | ✅ | ✅ | ✅ | ✅ | Many |
| CTC | ✅ HuggingFace | ✅ HuggingFace | ✅ HuggingFace | ✅ MMS_FA | MMS_FA (Latin) |
| NeMo | ❌ No model | ❌ No model | ❌ No model | ✅ | 6 langs only |
| WhisperX | ⚠️ Low accuracy | ⚠️ Low accuracy | ⚠️ Low accuracy | ✅ | Many |

**CTC Japanese Support**: Uses `jonatasgrosman/wav2vec2-large-xlsr-53-japanese` with native kanji/hiragana/katakana vocabulary (2341 tokens including 2155 kanji).

**NeMo Limitation**: Only supports en, es, de, fr, it, ru, pl. No CJK models available.

---

## Engines

### 1. CTCEngine (권장)

> 클래스 다이어그램: [../diagrams/08_class_diagram.mermaid](../diagrams/08_class_diagram.mermaid)

**특징**:
- GPU 가속 (CUDA)
- 일본어/한국어: HuggingFace wav2vec2-xlsr 모델 (native 문자 지원)
- 영어/기타: torchaudio MMS_FA (Latin alphabet)
- **속도**: 155초 오디오 → 5초 (MFA 대비 90x)
- **정확도**: 98.4% match rate (일본어)

**모델 매핑**:
```python
LANG_MODEL_MAP = {
    "ja": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",  # 2341 tokens
    "ko": "kresnik/wav2vec2-large-xlsr-korean",
    "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
}
# 그 외: torchaudio.pipelines.MMS_FA (a-z 26자)
```

---

### 2. NeMoEngine (제한적)

**⚠️ 제한사항**: CJK 언어 모델 없음

지원 언어: en, es, de, fr, it, ru, pl (7개만)

```python
NEMO_LANG_MODEL = {
    "en": "stt_en_conformer_ctc_large",
    "es": "stt_es_conformer_ctc_large",
    # ... ja, ko, zh 없음
}
```

---

## Configuration Updates

### settings.py

```python
class AlignmentSettings(BaseSettings):
    # Existing
    engine: Literal["whisperx", "mfa", "hybrid", "ctc", "nemo", "gpu-hybrid"] = "gpu-hybrid"
    
    # New: CTC settings
    ctc_model: str = "MahmoudAshraf/mms-300m-1130-forced-aligner"
    ctc_batch_size: int = 1
    
    # New: NeMo settings  
    nemo_model_ja: str = "nvidia/stt_ja_conformer_ctc_large"
    nemo_model_en: str = "nvidia/stt_en_conformer_ctc_large"
    nemo_model_ko: str = "nvidia/stt_ko_conformer_ctc_large"
```

---

## CLI Updates

```bash
# New engine options
everyric2 sync audio.wav lyrics.txt --engine ctc --language ja
everyric2 sync audio.wav lyrics.txt --engine nemo --language ja
everyric2 sync audio.wav lyrics.txt --engine gpu-hybrid --language ja  # CTC + NeMo

# Backward compatible
everyric2 sync audio.wav lyrics.txt --engine hybrid --language ja      # WhisperX + MFA (legacy)
```

---

## Implementation Status ✅

### Phase 1: CTCEngine ✅
- [x] HuggingFace wav2vec2 모델 통합 (ja/ko/zh)
- [x] torchaudio MMS_FA 폴백 (en/기타)
- [x] EngineFactory 등록
- [x] 155초 일본어 오디오 5초 처리 (90x 향상)

### Phase 2: NeMoEngine ⚠️
- [x] NeMo toolkit 설치 및 테스트
- [x] NeMoEngine 구현
- ⚠️ **제한**: CJK 언어 모델 없음 (en/es/de/fr/it/ru/pl만 지원)

### Phase 3: Integration ✅
- [x] CLI 옵션 업데이트 (`--engine ctc`)
- [x] diagnostics.png에 match rate 표시
- [x] README 업데이트
- [x] 벤치마크 완료

---

## Benchmark Results (155s Japanese audio)

| Engine | Time | Speedup | Japanese Support |
|--------|------|---------|------------------|
| MFA | 454s | 1x (baseline) | ✅ Perfect |
| CTC | **5s** | **90x** | ✅ HuggingFace wav2vec2 |
| WhisperX | 10s | 45x | ⚠️ ~35% match rate |
| NeMo | N/A | N/A | ❌ No Japanese model |

### Recommended Usage

- **Japanese/Korean/Chinese**: Use `--engine ctc` (90x faster than MFA)
- **English/European**: Use `--engine ctc` (MMS_FA model)
- **Maximum accuracy needed**: Use `--engine mfa` (slow but precise)
- **Transcription + alignment**: Use `--engine whisperx`

---

## File Structure

```
everyric2/alignment/
├── base.py              # BaseAlignmentEngine (existing)
├── factory.py           # EngineFactory (update)
├── whisperx_engine.py   # WhisperX (existing)
├── mfa_engine.py        # MFA (existing)
├── hybrid_engine.py     # WhisperX+MFA hybrid (existing)
├── ctc_engine.py        # NEW: CTC forced aligner
├── nemo_engine.py       # NEW: NeMo NFA
└── gpu_hybrid_engine.py # NEW: CTC+NeMo hybrid
```

---

## Testing Strategy

### Unit Tests
- `test_ctc_engine.py`: CTCEngine 단위 테스트
- `test_nemo_engine.py`: NeMoEngine 단위 테스트
- `test_gpu_hybrid_engine.py`: GPUHybridEngine 통합 테스트

### Integration Tests
```bash
# Accuracy comparison
pytest tests/integration/test_accuracy.py -v

# Performance benchmark
pytest tests/integration/test_benchmark.py -v --benchmark
```

### Manual Verification
```bash
# ftest1 테스트
everyric2 sync ftest1/audio.wav ftest1/lyrics.txt --engine ctc --debug
everyric2 sync ftest1/audio.wav ftest1/lyrics.txt --engine nemo --debug
everyric2 sync ftest1/audio.wav ftest1/lyrics.txt --engine gpu-hybrid --debug
```

---

## Dependencies

### New Requirements
```
# requirements-gpu.txt
ctc-forced-aligner>=0.1.0
nemo_toolkit[asr]>=2.0.0
```

### Optional (for diagnostics)
```
torch>=2.0.0  # Already installed
torchaudio>=2.0.0  # Already installed
```
