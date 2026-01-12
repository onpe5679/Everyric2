# Everyric2

Qwen3-Omni 멀티모달 LLM을 사용한 가사 싱크 도구입니다.

## 특징

기존 가사 싱크 도구들의 한계를 해결합니다:

| 기존 방식 | Everyric2 |
|-----------|-----------|
| Audio → Whisper → LLM (텍스트만) | Audio + 가사 → Qwen3-Omni |
| LLM이 오디오를 못 들음 | **오디오를 직접 들으면서** 타이밍 정렬 |
| 추측 기반 타이밍 | 실제 음성 기반 정확한 타이밍 |

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
# or: .venv\Scripts\activate  # Windows

# 기본 설치
pip install -e .

# 개발 의존성 포함 설치
pip install -e ".[dev]"

# 보컬 분리 (Demucs) 포함 설치
pip install -e ".[separator]"

# 전체 설치
pip install -e ".[all]"
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
# 기본 경로: /mnt/d/models/qwen3-omni/
mkdir -p /mnt/d/models/qwen3-omni
# thinker-q4_k_m.gguf와 mmproj-f16.gguf를 해당 경로에 복사
```

## 사용법

### CLI 모드

```bash
# 로컬 오디오 파일과 가사 싱크
everyric2 sync song.mp3 lyrics.txt -o output.srt

# YouTube 영상에서 가사 싱크
# 주의: YouTube 봇 감지로 인해 쿠키가 필요할 수 있습니다
everyric2 sync "https://youtube.com/watch?v=..." lyrics.txt -o output.srt

# 보컬 분리 사용 (정확도 향상)
everyric2 sync song.mp3 lyrics.txt --separate -o output.srt

# 다른 출력 포맷
everyric2 sync song.mp3 lyrics.txt -f lrc -o output.lrc
everyric2 sync song.mp3 lyrics.txt -f ass -o output.ass
everyric2 sync song.mp3 lyrics.txt -f json -o output.json

# 도움말
everyric2 --help
everyric2 sync --help
```

### 배치 모드

여러 곡을 한 번에 처리할 수 있습니다:

```bash
# 배치 테스트 실행
everyric2 batch batch.yaml

# 이전 결과에서 이어서 실행
everyric2 batch batch.yaml --resume
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
      Third line
```

### API 서버 모드

```bash
# 서버 시작
everyric2 serve --port 8000

# 개발 모드 (auto-reload)
everyric2 serve --port 8000 --reload
```

API 엔드포인트:
- `GET /health` - 서버 상태 확인
- `GET /formats` - 지원 포맷 목록
- `POST /sync/upload` - 파일 업로드로 싱크
- `POST /sync/youtube` - YouTube URL로 싱크
- `GET /youtube/info` - YouTube 영상 정보

### Python 코드에서 사용

```python
from everyric2.inference.qwen_omni_gguf import QwenOmniGGUFEngine
from everyric2.inference.prompt import LyricLine
from everyric2.output.formatters import FormatterFactory

# 엔진 초기화
engine = QwenOmniGGUFEngine()
engine.load_model()  # llama-server 시작 (첫 실행 시 ~1분)

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

## 설정

환경변수 또는 `.env` 파일로 설정합니다:

```bash
# 캐시 디렉토리 (WSL에서 D: 드라이브 사용)
EVERYRIC_MODEL__CACHE_DIR=/mnt/d/huggingface_cache

# 서버 설정
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
| GGUF Q4_K_M | ~16GB |
| GGUF Q4_K_M + 긴 오디오 | ~24GB |

## 프로젝트 구조

```
everyric2/
├── cli.py                    # CLI 인터페이스
├── server.py                 # FastAPI 서버
├── batch.py                  # 배치 처리
├── inference/
│   ├── qwen_omni_gguf.py    # GGUF 엔진 (llama.cpp)
│   ├── qwen_omni_vllm.py    # vLLM 엔진 (미사용)
│   └── prompt.py            # 프롬프트 빌더
├── audio/
│   ├── downloader.py        # YouTube 다운로더
│   ├── loader.py            # 오디오 로더
│   └── separator.py         # 보컬 분리 (Demucs)
├── output/
│   └── formatters.py        # SRT/ASS/LRC/JSON 포매터
└── config/
    └── settings.py          # 설정 관리
```

## 개발

```bash
# 테스트 실행
pytest

# 커버리지 포함
pytest --cov=everyric2

# 린트
ruff check everyric2

# 타입 체크
mypy everyric2
```

## 문제 해결

### YouTube 다운로드 오류

YouTube 봇 감지로 인해 다운로드가 실패할 수 있습니다:

```
ERROR: Sign in to confirm you're not a bot
```

해결 방법:
1. 브라우저에서 YouTube에 로그인
2. 쿠키를 추출하여 사용 ([yt-dlp 쿠키 가이드](https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp))

또는 로컬 파일을 직접 사용하세요:
```bash
everyric2 sync local_audio.mp3 lyrics.txt -o output.srt
```

### llama-server 시작 실패

모델 경로를 확인하세요:
```bash
ls -la /mnt/d/models/qwen3-omni/
# thinker-q4_k_m.gguf, mmproj-f16.gguf 파일이 있어야 함
```

llama-server 바이너리 경로 확인:
```bash
ls -la /home/at192u/dev/llama-cpp-qwen3-omni/build/bin/llama-server
```

### VRAM 부족

긴 오디오는 자동으로 30초 청크로 분할됩니다. 그래도 부족하면:
- 더 작은 양자화 모델 사용 (Q3_K_M 등)
- 보컬 분리 비활성화

## 라이선스

MIT License

## 크레딧

- [Qwen3-Omni](https://github.com/QwenLM/Qwen2.5-Omni) - 멀티모달 LLM
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - GGUF 추론 엔진
- [Demucs](https://github.com/facebookresearch/demucs) - 보컬 분리
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - YouTube 다운로드
