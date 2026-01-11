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

## 설치

```bash
# 저장소 클론
git clone https://github.com/onpe5679/Everyric2.git
cd Everyric2

# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# 기본 설치
pip install -e .

# 개발 의존성 포함 설치
pip install -e ".[dev]"

# 보컬 분리 (Demucs) 포함 설치
pip install -e ".[separator]"

# 전체 설치
pip install -e ".[all]"
```

## 모델 캐시 설정 (WSL 사용자)

WSL에서 D: 드라이브에 모델을 저장하여 여러 프로젝트에서 공유할 수 있습니다:

```bash
# .env 파일 생성
cp .env.example .env

# .env 파일 편집
EVERYRIC_MODEL__CACHE_DIR=/mnt/d/huggingface_cache
```

또는 환경변수로 직접 설정:

```bash
export HF_HOME=/mnt/d/huggingface_cache
```

## 사용법

### CLI 모드

```bash
# 로컬 오디오 파일과 가사 싱크
everyric2 sync song.mp3 lyrics.txt -o output.srt

# YouTube 영상에서 가사 싱크
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
from everyric2.inference import QwenOmniEngine, LyricLine
from everyric2.output import FormatterFactory

# 엔진 초기화
engine = QwenOmniEngine()
engine.load_model()

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
```

## 설정

환경변수 또는 `.env` 파일로 설정합니다:

```bash
# 모델 경로 변경
EVERYRIC_MODEL__PATH=your/custom/model

# 캐시 디렉토리 (WSL에서 D: 드라이브 사용)
EVERYRIC_MODEL__CACHE_DIR=/mnt/d/huggingface_cache

# GPU 메모리가 부족한 경우 청크 크기 줄이기
EVERYRIC_MODEL__CHUNK_DURATION=600  # 10분

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
| AWQ-4bit (기본) | ~24GB |
| AWQ-4bit + 긴 오디오 | ~32GB |

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

## 라이선스

MIT License

## 크레딧

- [Qwen-Omni](https://github.com/QwenLM/Qwen2.5-Omni) - 멀티모달 LLM
- [Demucs](https://github.com/facebookresearch/demucs) - 보컬 분리
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - YouTube 다운로드
