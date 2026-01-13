# Everyric2

WhisperX + MFA 기반 하이브리드 가사 싱크 도구.

## 엔진 비교

| 엔진 | Match Rate | Speed | VRAM | 특징 |
|------|------------|-------|------|------|
| **WhisperX** | ~35% | ~10x | ~6GB | 전사 기반, 퍼지 매칭 |
| **MFA** | ~100% | ~5x | CPU | 정확한 가사 필요, 초정밀 |
| **Hybrid** | Best | ~3x | ~6GB | WhisperX + MFA 동시 실행, 진단 비교 |

## 설치

```bash
# 기본 설치
git clone https://github.com/onpe5679/Everyric2.git
cd Everyric2
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"

# WhisperX
pip install git+https://github.com/m-bain/whisperx.git
```

### MFA 설치 (Conda)

```bash
conda create -n mfaenv python=3.10
conda activate mfaenv
conda install -c conda-forge montreal-forced-aligner kalpy
pip install fugashi unidic-lite  # 일본어

# 모델 다운로드
mfa model download acoustic japanese_mfa
mfa model download dictionary japanese_mfa
mfa model download acoustic english_mfa
mfa model download dictionary english_us_arpa
mfa model download acoustic korean_mfa
mfa model download dictionary korean_mfa
```

## 사용법

```bash
# 기본 (Hybrid - WhisperX + MFA)
PATH="$HOME/.conda/envs/mfaenv/bin:$PATH" \
everyric2 sync audio.wav lyrics.txt --language ja --debug

# 전체 파이프라인 (보컬분리 + 번역 + 진단)
PATH="$HOME/.conda/envs/mfaenv/bin:$PATH" \
everyric2 sync audio.wav lyrics.txt --separate --translate --debug --language ja

# WhisperX만 사용
everyric2 sync audio.wav lyrics.txt --engine whisperx --language ja

# MFA만 사용
everyric2 sync audio.wav lyrics.txt --engine mfa --language ja
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
| whisperx Transcription | hybrid/whisperx | WhisperX 전사 결과 + match % |
| mfa Transcription | hybrid/mfa | MFA 전사 결과 + match % |
| Translated | --translate | 번역된 가사 |
| Synced Output | 항상 | 최종 싱크 결과 |

### diagnostics.png 예시

| Line# | Lyric | Start | End | whisperx (match: 35%) | mfa (match: 100%) |
|-------|-------|-------|-----|----------------------|-------------------|
| 1 | 歌詞テキスト | 0.50 | 2.30 | かし てきすと | 歌詞 テキスト |

## 아키텍처

```
Audio ─→ [Demucs] ─→ Vocals ─→ ┌─────────────────┐
                               │  HybridEngine   │
Lyrics ──────────────────────→ │                 │
                               │ 1. WhisperX ────┼─→ transcription_sets
                               │ 2. MFA ─────────┼─→ transcription_sets
                               └────────┬────────┘
                                        ↓
                               ┌─────────────────┐
                               │ DiagnosticsViz  │ → diagnostics.png
                               └────────┬────────┘
                                        ↓
                               SRT / ASS / LRC / JSON
```

## 핵심 컴포넌트

### HybridEngine

```python
class HybridEngine(BaseAlignmentEngine):
    def align(self, audio, lyrics, language, progress_callback):
        self._transcription_sets = []
        
        # 1. WhisperX 실행
        wx_results = self.whisperx.align(...)
        self._transcription_sets.extend(self.whisperx.get_transcription_sets())
        
        # 2. MFA 실행 (가능한 경우)
        if self.mfa.is_available():
            mfa_results = self.mfa.align(...)
            self._transcription_sets.extend(self.mfa.get_transcription_sets())
            return mfa_results
        
        return wx_results
    
    def get_transcription_sets(self):
        """모든 엔진의 전사 데이터 반환 (진단용)"""
        return self._transcription_sets
```

### MFA 바이너리 자동 감지

```python
def _resolve_mfa_bin(self):
    # 1. 환경변수: EVERYRIC_ALIGNMENT__MFA_BIN
    # 2. Conda 기본 경로: ~/.conda/envs/mfaenv/bin/mfa
    # 3. PATH: shutil.which("mfa")
```

## 설정

### 환경변수

```bash
# .env 파일
GEMINI_API_KEY=AIza...                    # 번역용
EVERYRIC_ALIGNMENT__ENGINE=hybrid
EVERYRIC_ALIGNMENT__LANGUAGE=ja
EVERYRIC_ALIGNMENT__MFA_BEAM=1000         # 노래용 증가 (기본 100)
EVERYRIC_ALIGNMENT__MFA_RETRY_BEAM=4000   # 노래용 증가 (기본 400)
```

### AlignmentSettings

```python
class AlignmentSettings(BaseSettings):
    engine: Literal["whisperx", "mfa", "hybrid", "qwen"] = "hybrid"  # 기본값
    language: Literal["auto", "en", "ja", "ko"] = "auto"
    
    whisperx_model: str = "large-v3"
    whisperx_batch_size: int = 16
    whisperx_compute_type: str = "float16"
    
    mfa_beam: int = 1000        # 노래용 증가
    mfa_retry_beam: int = 4000  # 노래용 증가
```

## 문제 해결

### MFA NoAlignmentsError

```
NoAlignmentsError: There were no successful alignments
```

**해결**: beam 값 증가 (이미 기본값으로 적용됨)

### MFA 바이너리 못 찾음

```bash
# PATH에 conda 환경 추가
PATH="$HOME/.conda/envs/mfaenv/bin:$PATH" everyric2 sync ...

# 또는 환경변수 설정
export EVERYRIC_ALIGNMENT__MFA_BIN=$HOME/.conda/envs/mfaenv/bin/mfa
```

### WhisperX 낮은 Match Rate

- `--separate` 옵션으로 보컬 분리
- MFA가 더 높은 정확도 (정확한 가사 필요)

## 프로젝트 구조

```
everyric2/
├── cli.py                     # CLI
├── alignment/
│   ├── base.py               # BaseAlignmentEngine
│   ├── whisperx_engine.py    # WhisperX
│   ├── mfa_engine.py         # MFA (conda 자동 감지)
│   ├── hybrid_engine.py      # Hybrid (둘 다 실행)
│   └── matcher.py            # LyricsMatcher, MatchStats
├── audio/
│   ├── loader.py             # AudioLoader
│   └── separator.py          # Demucs
├── debug/
│   ├── debug_info.py         # DebugInfo
│   └── visualizer.py         # diagnostics.png 생성
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

# Hybrid 엔진 테스트
PATH="$HOME/.conda/envs/mfaenv/bin:$PATH" \
.venv/bin/everyric2 sync ftest1/audio.wav ftest1/lyrics.txt \
  --engine hybrid --debug --language ja
```

## 라이선스

MIT License
