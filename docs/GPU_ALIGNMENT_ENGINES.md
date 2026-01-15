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

```
                    ┌─────────────────────────────────────┐
                    │         HybridEngine v2             │
                    │                                     │
Audio + Lyrics ────→│  ┌─────────┐    ┌─────────┐       │
                    │  │ CTCEngine│    │NeMoEngine│       │
                    │  │  (GPU)   │    │  (GPU)   │       │
                    │  └────┬────┘    └────┬────┘       │
                    │       │              │             │
                    │       ▼              ▼             │
                    │  ┌─────────────────────────┐      │
                    │  │   Result Selector       │      │
                    │  │ (confidence-based)      │      │
                    │  └───────────┬─────────────┘      │
                    └──────────────┼──────────────────────┘
                                   ▼
                           Word Timestamps
```

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

## New Engines

### 1. CTCEngine (ctc-forced-aligner)

```python
class CTCEngine(BaseAlignmentEngine):
    """
    CTC-based forced alignment using Wav2Vec2/MMS models.
    
    Features:
    - GPU acceleration (CUDA)
    - 1130+ languages via MMS
    - Lightweight installation
    
    Models:
    - Japanese: MMS-based or wav2vec2-large-xlsr-53-japanese
    - Korean: wav2vec2-large-xlsr-korean
    - English: wav2vec2-base-960h
    """
    
    def align(self, audio, lyrics, language, progress_callback):
        # 1. Load audio (16kHz)
        # 2. Tokenize lyrics text
        # 3. Run CTC forced alignment on GPU
        # 4. Extract word timestamps
        # 5. Match with lyrics lines
        pass
```

**Installation**:
```bash
pip install ctc-forced-aligner
```

**Configuration**:
```python
ctc_model: str = "MahmoudAshraf/mms-300m-1130-forced-aligner"
ctc_language: str = "jpn"  # ISO 639-3 code
```

---

### 2. NeMoEngine (NVIDIA NeMo NFA)

```python
class NeMoEngine(BaseAlignmentEngine):
    """
    NVIDIA NeMo Forced Aligner using Conformer-CTC.
    
    Features:
    - Production-grade from NVIDIA
    - High accuracy
    - Long audio support (1hr+)
    
    Models:
    - Japanese: stt_ja_conformer_ctc_large
    - Korean: stt_ko_conformer_ctc_large  
    - English: stt_en_conformer_ctc_large
    """
    
    def align(self, audio, lyrics, language, progress_callback):
        # 1. Load NeMo ASR model
        # 2. Generate CTC emissions
        # 3. Run forced alignment
        # 4. Extract word/token timestamps
        # 5. Match with lyrics lines
        pass
```

**Installation**:
```bash
pip install nemo_toolkit[asr]
```

**Configuration**:
```python
nemo_model: str = "nvidia/stt_ja_conformer_ctc_large"
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

## Implementation Plan

### Phase 1: CTCEngine
1. [ ] Install and test ctc-forced-aligner
2. [ ] Implement CTCEngine class
3. [ ] Add to EngineFactory
4. [ ] Test with ftest1

### Phase 2: NeMoEngine
1. [ ] Install and test NeMo toolkit
2. [ ] Implement NeMoEngine class
3. [ ] Add to EngineFactory
4. [ ] Test with ftest1

### Phase 3: GPU Hybrid
1. [ ] Create GPUHybridEngine (CTC + NeMo)
2. [ ] Implement confidence-based result selection
3. [ ] Parallel execution on GPU
4. [ ] Benchmark vs MFA

### Phase 4: Integration
1. [ ] Update CLI options
2. [ ] Update diagnostics visualization
3. [ ] Update README
4. [ ] Performance comparison table

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
