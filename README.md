# Everyric2

Qwen3-Omni 멀티모달 LLM을 사용한 가사 싱크 도구입니다.

## 특징

기존 가사 싱크 도구들의 한계를 해결합니다:

| 기존 방식 | Everyric2 |
|-----------|-----------|
| Audio → Whisper → LLM (텍스트만) | Audio + 가사 → Qwen3-Omni |
| LLM이 오디오를 못 들음 | **오디오를 직접 들으면서** 타이밍 정렬 |
| 추측 기반 타이밍 | 실제 음성 기반 정확한 타이밍 |

## 구현 로직

### 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              CLI (cli.py)                                │
│  everyric2 sync audio.mp3 lyrics.txt --separate --translate --debug     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
            │ AudioLoader │ │ LyricLine   │ │ Translator  │
            │ (loader.py) │ │ (prompt.py) │ │(translator) │
            └─────────────┘ └─────────────┘ └─────────────┘
                    │               │               │
                    ▼               │               │
            ┌─────────────┐         │               │
            │VocalSeparator│        │               │
            │(separator.py)│        │               │
            │  (Demucs)   │         │               │
            └─────────────┘         │               │
                    │               │               │
                    └───────────────┼───────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │     QwenOmniGGUFEngine        │
                    │    (qwen_omni_gguf.py)        │
                    │                               │
                    │  ┌─────────────────────────┐  │
                    │  │    Chunk Processing     │  │
                    │  │  60초씩 분할 + 전체 가사  │  │
                    │  └─────────────────────────┘  │
                    │              │                │
                    │              ▼                │
                    │  ┌─────────────────────────┐  │
                    │  │     llama-server        │  │
                    │  │   (localhost:8081)      │  │
                    │  │  Qwen3-Omni GGUF 모델   │  │
                    │  └─────────────────────────┘  │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │         Output                │
                    │  ┌─────────┐ ┌─────────────┐  │
                    │  │Formatter│ │ Visualizer  │  │
                    │  │SRT/ASS/ │ │diagnostics  │  │
                    │  │LRC/JSON │ │   .png      │  │
                    │  └─────────┘ └─────────────┘  │
                    └───────────────────────────────┘
```

### 처리 흐름

```
1. 오디오 로드
   audio.mp3 → AudioLoader → AudioData (waveform, sample_rate, duration)

2. 보컬 분리 (--separate)
   AudioData → Demucs → vocals AudioData (배경음 제거)

3. 가사 로드
   lyrics.txt → LyricLine.from_file() → list[LyricLine]

4. 번역 (--translate)
   list[LyricLine] → LyricsTranslator → translated_text (한국어)

5. 청크 분할 및 LLM 처리
   ┌────────────────────────────────────────────────────────┐
   │ 220초 오디오 예시 (60초 청크)                           │
   │                                                        │
   │ Chunk 0: 0-60초   ──┐                                  │
   │ Chunk 1: 60-120초 ──┼── 각 청크에 전체 가사 포함        │
   │ Chunk 2: 120-180초──┤   "이 오디오는 X초~Y초 구간입니다" │
   │ Chunk 3: 180-220초──┘   → LLM이 해당 구간 가사만 타이밍 │
   └────────────────────────────────────────────────────────┘

6. 결과 병합 및 중복 제거
   chunk_results → deduplicate → final_results

7. 출력 생성
   - output.srt (원본 가사 자막)
   - output_translated.srt (번역 자막, --translate 시)
   - diagnostics.png (시각화, --debug 시)
```

### 프롬프트 구조

```
# 첫 번째 청크 (0-60초)
Listen to this audio and synchronize these lyrics with timestamps.

LYRICS:
1. 名前を握りしめて
2. 振り向かれないように
...

# 이후 청크들 (60초 이후)
This audio segment is from 60.0s to 120.0s of a 220.0s song.

FULL LYRICS OF THE SONG:
1. 名前を握りしめて
2. 振り向かれないように
...

Listen to this audio segment and identify which lyrics are sung.
Provide timestamps relative to the FULL SONG (not this segment).
```

### 디버그 출력 구조 (--debug)

```
output/20260112_211713/
├── audio_original.wav      # 원본 전체 오디오
├── audio_vocals.wav        # 보컬 분리된 오디오 (--separate 시)
├── chunks/                 # 청크별 디버그 파일
│   ├── chunk_000_0.0s-60.0s.wav      # 청크 오디오 (보컬 분리본)
│   ├── chunk_000_prompt.txt          # LLM에 보낸 프롬프트
│   ├── chunk_000_response.txt        # LLM 응답
│   ├── chunk_001_60.0s-120.0s.wav
│   ├── chunk_001_prompt.txt
│   ├── chunk_001_response.txt
│   └── ...
├── lyrics_original.txt     # 원본 가사
├── lyrics_translated_ko.txt # 번역된 가사 (--translate 시)
├── output.srt              # 원본 자막
├── output_translated.srt   # 번역 자막 (--translate 시)
├── debug_info.json         # 전체 디버그 정보 (타이밍, 설정 등)
├── diagnostics.png         # 시각화 차트
└── settings.json           # 실행 설정
```

### Diagnostics 시각화 (6컬럼)

```
┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
│  Audio   │  Audio   │ Original │Translated│  Chunk   │  Synced  │
│(Original)│ (Vocals) │  Lyrics  │  Lyrics  │Processing│  Output  │
├──────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
│          │          │名前を握り│이름을 꼭 │ Chunk 1  │名前を握り│
│ ▓▓▓▓▓▓  │ ▓▓▓▓▓   │しめて    │쥐고      │ 0-60s    │ 0.0-2.5s │
│          │          │          │          │          │          │
│ ▓▓▓     │ ▓▓▓▓    │振り向か  │뒤돌아보지│          │振り向か  │
│          │          │れない    │않도록    │          │ 2.5-5.0s │
│          │          │ように    │          │          │          │
│  ...     │  ...     │  ...     │  ...     │ Chunk 2  │  ...     │
│          │          │          │          │ 60-120s  │          │
└──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
```

## 요구사항

- Python 3.10+
- NVIDIA GPU (24GB+ VRAM 권장)
- ffmpeg
- [llama.cpp with Qwen3-Omni support](https://github.com/user/llama-cpp-qwen3-omni)

### 모델 파일

GGUF 모델 파일이 필요합니다:
- `thinker-q4_k_m.gguf` (~18GB) - 메인 모델
- `mmproj-f16.gguf` (~2.4GB) - 멀티모달 프로젝터

## 설치

```bash
# 저장소 클론
git clone https://github.com/onpe5679/Everyric2.git
cd Everyric2

# 가상환경 생성 (권장)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac

# 전체 설치 (보컬 분리 포함)
pip install -e ".[all]"

# 추가 의존성
pip install demucs torchcodec
```

### llama.cpp 설정

```bash
# llama.cpp 빌드 (Qwen3-Omni 지원 버전)
git clone https://github.com/user/llama-cpp-qwen3-omni.git
cd llama-cpp-qwen3-omni
mkdir build && cd build
cmake .. -DGGML_CUDA=ON
cmake --build . --config Release -j

# 모델 파일 배치
mkdir -p /mnt/d/models/qwen3-omni
# thinker-q4_k_m.gguf와 mmproj-f16.gguf를 해당 경로에 복사
```

## 사용법

### 빠른 시작

```bash
cd /home/at192u/dev/everyric2
source .venv/bin/activate

# 권장: 보컬 분리 + 번역 + 디버그 모드
everyric2 sync song.mp3 lyrics.txt --separate --translate --debug

# 결과: output/YYYYMMDD_HHMMSS/ 폴더에 저장됨
```

### CLI 옵션

```bash
everyric2 sync [OPTIONS] SOURCE LYRICS

Arguments:
  SOURCE    오디오 파일 경로 또는 YouTube URL
  LYRICS    가사 텍스트 파일 경로

Options:
  -o, --output PATH           출력 파일 경로
  -f, --format TEXT           출력 포맷: srt, ass, lrc, json (기본: srt)
  -s, --separate              Demucs로 보컬 분리 (정확도 향상)
  -t, --translate             가사를 한국어로 번역
  -d, --debug                 디버그 파일 저장 (청크, 프롬프트, 진단 이미지)
  -c, --chunk-duration INT    청크 길이 초 (기본: 60)
  -m, --model TEXT            모델 경로 오버라이드
  --help                      도움말
```

### 예시

```bash
# 기본 사용
everyric2 sync song.mp3 lyrics.txt -o output.srt

# 보컬 분리 (배경음 제거로 정확도 향상)
everyric2 sync song.mp3 lyrics.txt -s -o output.srt

# 번역 포함
everyric2 sync song.mp3 lyrics.txt -s -t -o output.srt

# 전체 기능 (디버그 모드)
everyric2 sync song.mp3 lyrics.txt -s -t -d

# 청크 크기 조절 (긴 오디오용)
everyric2 sync long_song.mp3 lyrics.txt -s -t -d -c 90

# YouTube에서 직접
everyric2 sync "https://youtube.com/watch?v=..." lyrics.txt -s -t -d

# LRC 포맷 출력
everyric2 sync song.mp3 lyrics.txt -f lrc -o output.lrc
```

### 배치 모드

여러 곡을 한 번에 처리:

```bash
everyric2 batch batch.yaml
everyric2 batch batch.yaml --resume  # 이어서 실행
```

배치 설정 파일 예시 (`batch.yaml`):

```yaml
output_dir: ./output
formats: [srt, ass, lrc, json]
tests:
  - title: "Song Name"
    source: "./song.mp3"
    lyrics_file: "./lyrics.txt"
  - title: "Another Song"
    source: "https://youtube.com/watch?v=..."
    lyrics: |
      First line
      Second line
```

### API 서버 모드

```bash
everyric2 serve --port 8000
everyric2 serve --port 8000 --reload  # 개발 모드
```

API 엔드포인트:
- `GET /health` - 서버 상태 확인
- `GET /formats` - 지원 포맷 목록
- `POST /sync/upload` - 파일 업로드로 싱크
- `POST /sync/youtube` - YouTube URL로 싱크

### Python 코드에서 사용

```python
from everyric2.inference.qwen_omni_gguf import QwenOmniGGUFEngine
from everyric2.inference.prompt import LyricLine
from everyric2.output.formatters import FormatterFactory

# 엔진 초기화 (청크 크기 60초)
engine = QwenOmniGGUFEngine(chunk_duration=60)
engine.load_model()  # llama-server 자동 시작 (~1분)

# 가사 준비
lyrics = LyricLine.from_file("lyrics.txt")

# 싱크 실행
results = engine.sync_lyrics("audio.mp3", lyrics)

# 결과 포맷팅
formatter = FormatterFactory.get_formatter("srt")
output = formatter.format(results)

# 저장
with open("output.srt", "w") as f:
    f.write(output)

# 서버 종료
engine.unload_model()
```

## 프로젝트 구조

```
everyric2/
├── cli.py                    # CLI 인터페이스
├── server.py                 # FastAPI 서버
├── batch.py                  # 배치 처리
├── inference/
│   ├── qwen_omni_gguf.py    # GGUF 엔진 (llama.cpp)
│   │   ├── QwenOmniGGUFEngine  # 메인 엔진 클래스
│   │   ├── ChunkContext        # 청크 컨텍스트 (dataclass)
│   │   ├── _sync_single()      # 단일 청크 처리
│   │   ├── _sync_with_chunks() # 청크 분할 처리
│   │   └── _build_prompt()     # 프롬프트 생성
│   └── prompt.py             # 프롬프트 빌더, LyricLine, SyncResult
├── audio/
│   ├── downloader.py        # YouTube 다운로더 (yt-dlp)
│   ├── loader.py            # 오디오 로더 (AudioData)
│   └── separator.py         # 보컬 분리 (Demucs)
├── translation/
│   └── translator.py        # 가사 번역 (LLM 기반)
├── debug/
│   ├── output_manager.py    # 디버그 출력 관리
│   ├── debug_info.py        # 디버그 정보 수집
│   └── visualizer.py        # Diagnostics 시각화
├── output/
│   └── formatters.py        # SRT/ASS/LRC/JSON 포매터
└── config/
    └── settings.py          # 설정 관리
```

## 설정

환경변수 또는 `.env` 파일:

```bash
EVERYRIC_MODEL__CACHE_DIR=/mnt/d/huggingface_cache
EVERYRIC_SERVER__PORT=8080
```

현재 설정 확인:
```bash
everyric2 config
```

## 지원 포맷

| 포맷 | 확장자 | 설명 |
|------|--------|------|
| SRT | `.srt` | 가장 널리 지원되는 자막 포맷 |
| ASS | `.ass` | 스타일링 지원 (색상, 폰트 등) |
| LRC | `.lrc` | 음악 플레이어용 가사 포맷 |
| JSON | `.json` | 프로그래밍 용도 |

## GPU 메모리 요구사항

| 모델 | VRAM |
|------|------|
| GGUF Q4_K_M | ~18GB |
| GGUF Q4_K_M + 보컬분리 + 긴 오디오 | ~24GB |

## 문제 해결

### llama-server 시작 실패 / 타임아웃

```bash
# 1. 기존 프로세스 정리
pkill -9 llama-server

# 2. 수동 시작
/home/at192u/dev/llama-cpp-qwen3-omni/build/bin/llama-server \
  -m /mnt/d/models/qwen3-omni/thinker-q4_k_m.gguf \
  --mmproj /mnt/d/models/qwen3-omni/mmproj-f16.gguf \
  --port 8081 -ngl 99 -c 8192

# 3. 상태 확인 (1~2분 대기 후)
curl http://localhost:8081/health
# {"status":"ok"} 나오면 정상
```

### 청크 처리 중 연결 끊김

청크 크기를 줄여보세요:
```bash
everyric2 sync song.mp3 lyrics.txt -s -t -d -c 45  # 45초 청크
```

### YouTube 다운로드 오류

```
ERROR: Sign in to confirm you're not a bot
```

로컬 파일을 직접 사용하세요:
```bash
everyric2 sync local_audio.mp3 lyrics.txt -s -t -d
```

### VRAM 부족

- 더 작은 양자화 모델 사용 (Q3_K_M 등)
- 청크 크기 줄이기: `-c 30`
- 보컬 분리 비활성화: `-s` 옵션 제거

## 라이선스

MIT License

## 크레딧

- [Qwen3-Omni](https://github.com/QwenLM/Qwen2.5-Omni) - 멀티모달 LLM
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - GGUF 추론 엔진
- [Demucs](https://github.com/facebookresearch/demucs) - 보컬 분리
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - YouTube 다운로드
