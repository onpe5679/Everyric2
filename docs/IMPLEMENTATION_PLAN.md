# Everyric2 - 상세 구현 계획서

## 개요

Qwen3-Omni 멀티모달 LLM을 활용하여 오디오를 직접 듣고 가사 타이밍을 정렬하는 도구입니다. 기존 Everyric의 "LLM이 오디오를 듣지 못하는" 근본적 한계를 해결합니다.

### 기존 문제점
```
Audio → Whisper(전사) → LLM(타이밍)
                          ↑
                    LLM이 오디오를 못 듣고
                    텍스트만 보고 추측
```

### 새로운 접근법
```
Audio + 원본가사 + 인식가사 → Qwen3-Omni → 타이밍된 가사
                                ↑
                          오디오를 직접 들으면서
                          텍스트와 정렬
```

---

## 1. `everyric2/config/settings.py`

### 목적
전역 설정 관리. Pydantic을 사용한 타입 안전한 설정 클래스 제공.

### 핵심 클래스/함수

```python
class ModelConfig(BaseSettings):
    """모델 관련 설정"""
    model_path: str = "cpatonn/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit"
    use_flash_attention: bool = True
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"
    max_audio_duration: int = 2400  # 40분 (초)
    chunk_duration: int = 1800  # 30분 청킹 단위

class AudioConfig(BaseSettings):
    """오디오 처리 설정"""
    target_sample_rate: int = 24000  # Qwen3-Omni 네이티브
    demucs_model: str = "htdemucs"
    temp_dir: Path = Path("/tmp/everyric2")

class OutputConfig(BaseSettings):
    """출력 설정"""
    default_format: str = "srt"
    supported_formats: list[str] = ["srt", "ass", "lrc", "json"]

class Settings(BaseSettings):
    """통합 설정"""
    model: ModelConfig
    audio: AudioConfig
    output: OutputConfig
    
    model_config = SettingsConfigDict(
        env_prefix="EVERYRIC_",
        env_nested_delimiter="__"
    )
```

### 의존성
- `pydantic-settings>=2.0`

### 데이터 흐름
```
환경변수/config.toml → Settings 파싱 → 각 모듈에서 참조
```

### 에러 처리
| 에러 | 처리 |
|------|------|
| 잘못된 모델 경로 | `ValidationError` with helpful message |
| 지원하지 않는 포맷 | `ValueError` |

### 테스트 방법
- 환경변수 오버라이드 테스트
- 기본값 검증
- 타입 검증

---

## 2. `everyric2/audio/loader.py`

### 목적
로컬 오디오 파일 로딩 및 Qwen3-Omni 입력 형식으로 변환 (24kHz 리샘플링).

### 핵심 클래스/함수

```python
@dataclass
class AudioData:
    """로드된 오디오 데이터"""
    waveform: np.ndarray  # shape: (samples,) mono
    sample_rate: int
    duration: float  # seconds
    source_path: Path

class AudioLoader:
    def __init__(self, config: AudioConfig): ...
    
    def load(self, path: Path | str) -> AudioData:
        """오디오 파일 로드 및 리샘플링"""
        
    def resample(self, audio: AudioData, target_sr: int = 24000) -> AudioData:
        """24kHz로 리샘플링 (Qwen3-Omni 요구사항)"""
        
    def to_mono(self, audio: AudioData) -> AudioData:
        """스테레오 → 모노 변환"""
        
    def validate_duration(self, audio: AudioData, max_duration: int) -> bool:
        """최대 길이 검증 (40분)"""

    def chunk_audio(
        self, 
        audio: AudioData, 
        chunk_duration: int = 1800,
        overlap: int = 30
    ) -> list[tuple[AudioData, float, float]]:
        """긴 오디오를 청킹. Returns: [(chunk, start_time, end_time), ...]"""
```

### 의존성
- `librosa>=0.10.0` (로딩, 리샘플링)
- `soundfile>=0.12.0` (파일 I/O)
- `numpy`

### 데이터 흐름
```
파일 경로 → librosa.load() → resample(24kHz) → to_mono() → AudioData
```

### 에러 처리
| 에러 | 처리 |
|------|------|
| 파일 없음 | `FileNotFoundError` with path |
| 지원하지 않는 포맷 | `UnsupportedFormatError` |
| 손상된 파일 | `CorruptedAudioError` |
| 40분 초과 | 경고 + 자동 청킹 제안 |

### 테스트 방법
- 다양한 포맷 로딩 (mp3, wav, flac, m4a)
- 리샘플링 정확도 검증
- 청킹 경계 검증

---

## 3. `everyric2/audio/downloader.py`

### 목적
YouTube URL에서 오디오 추출.

### 핵심 클래스/함수

```python
@dataclass
class DownloadResult:
    """다운로드 결과"""
    audio_path: Path
    title: str
    duration: float
    url: str

class YouTubeDownloader:
    def __init__(self, config: AudioConfig): ...
    
    def download(
        self, 
        url: str, 
        output_dir: Path | None = None
    ) -> DownloadResult:
        """YouTube URL에서 오디오 다운로드"""
        
    def extract_audio(
        self, 
        video_path: Path, 
        output_format: str = "wav"
    ) -> Path:
        """비디오에서 오디오 추출"""
        
    def validate_url(self, url: str) -> bool:
        """YouTube URL 유효성 검사"""
        
    def get_video_info(self, url: str) -> dict:
        """메타데이터 조회 (다운로드 전 길이 확인용)"""
```

### 의존성
- `yt-dlp>=2024.0.0`
- `ffmpeg` (시스템)

### 데이터 흐름
```
YouTube URL → yt-dlp (best audio) → ffmpeg (wav 변환) → DownloadResult
```

### 구현 세부사항
```python
ydl_opts = {
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
        'preferredquality': '192',
    }],
    'outtmpl': str(output_dir / '%(title)s.%(ext)s'),
}
```

### 에러 처리
| 에러 | 처리 |
|------|------|
| 잘못된 URL | `InvalidURLError` |
| 비공개/삭제된 영상 | `VideoUnavailableError` |
| 연령 제한 | `AgeRestrictedError` |
| 네트워크 오류 | 재시도 3회 후 실패 |
| ffmpeg 없음 | `DependencyError` |

### 테스트 방법
- Mock yt-dlp으로 단위 테스트
- 짧은 public 영상으로 통합 테스트

---

## 4. `everyric2/audio/separator.py`

### 목적
Demucs를 사용한 보컬/반주 분리 (선택적).

### 핵심 클래스/함수

```python
@dataclass
class SeparationResult:
    """분리 결과"""
    vocals: AudioData
    accompaniment: AudioData  # no_vocals
    original: AudioData

class VocalSeparator:
    def __init__(self, config: AudioConfig): ...
    
    def separate(
        self, 
        audio: AudioData,
        model: str = "htdemucs"
    ) -> SeparationResult:
        """보컬 분리"""
        
    def is_available(self) -> bool:
        """GPU/CPU 가용성 확인"""
        
    def get_available_models(self) -> list[str]:
        """사용 가능한 Demucs 모델 목록"""
```

### 의존성
- `demucs>=4.0.0`
- `torch>=2.0`

### 데이터 흐름
```
AudioData → demucs.separate.main() → vocals.wav + no_vocals.wav → SeparationResult
```

### 에러 처리
| 에러 | 처리 |
|------|------|
| GPU OOM | CPU 폴백 (경고 표시) |
| 모델 다운로드 실패 | 재시도 + 캐시 확인 |
| 분리 실패 | 원본 오디오 반환 (경고) |

### 테스트 방법
- 짧은 샘플로 분리 품질 검증
- CPU/GPU 모드 전환 테스트

---

## 5. `everyric2/inference/prompt.py`

### 목적
Qwen3-Omni용 프롬프트 템플릿 관리.

### 핵심 클래스/함수

```python
@dataclass
class LyricLine:
    """가사 라인"""
    text: str
    line_number: int

@dataclass
class SyncResult:
    """싱크 결과"""
    text: str
    start_time: float  # seconds
    end_time: float
    confidence: float | None = None

class PromptBuilder:
    def __init__(self): ...
    
    def build_sync_prompt(
        self,
        lyrics: list[LyricLine],
        language: str = "auto"
    ) -> str:
        """가사 싱크용 프롬프트 생성"""
        
    def build_conversation(
        self,
        audio_path: Path,
        lyrics: list[LyricLine],
        system_prompt: str | None = None
    ) -> list[dict]:
        """Qwen3-Omni conversation 형식으로 변환"""
        
    def parse_response(
        self, 
        response: str
    ) -> list[SyncResult]:
        """모델 응답 파싱"""
```

### 프롬프트 템플릿

```python
SYNC_SYSTEM_PROMPT = """You are a professional lyrics synchronization assistant.
Your task is to align lyrics with audio timestamps.

Rules:
1. Listen to the audio carefully and identify when each lyric line is sung
2. Output format: JSON array with start_time, end_time, text for each line
3. Times are in seconds with 2 decimal precision
4. Preserve the original lyric text exactly
5. If a line is not sung, mark it with null timestamps"""

SYNC_USER_PROMPT = """Listen to the following audio and align these lyrics:

{lyrics}

Output the synchronized lyrics as a JSON array:
[{{"text": "line 1", "start": 0.00, "end": 2.50}}, ...]"""
```

### Conversation 구조
```python
def build_conversation(self, audio_path: Path, lyrics: list[LyricLine]) -> list[dict]:
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYNC_SYSTEM_PROMPT}]
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": str(audio_path)},
                {"type": "text", "text": SYNC_USER_PROMPT.format(
                    lyrics="\n".join(f"{l.line_number}. {l.text}" for l in lyrics)
                )}
            ]
        }
    ]
```

### 의존성
- 없음 (순수 Python)

### 에러 처리
| 에러 | 처리 |
|------|------|
| JSON 파싱 실패 | 정규식 폴백 파서 |
| 누락된 필드 | 기본값 또는 경고 |

### 테스트 방법
- 다양한 응답 형식 파싱 테스트
- 프롬프트 토큰 수 검증

---

## 6. `everyric2/inference/qwen_omni.py`

### 목적
Qwen3-Omni 모델 로딩 및 추론.

### 핵심 클래스/함수

```python
class QwenOmniEngine:
    def __init__(
        self, 
        config: ModelConfig,
        backend: Literal["transformers", "vllm"] = "transformers"
    ): ...
    
    def load_model(self) -> None:
        """모델 로드 (lazy loading)"""
        
    def infer(
        self,
        conversation: list[dict],
        max_tokens: int = 4096,
        temperature: float = 0.1
    ) -> str:
        """추론 실행"""
        
    def sync_lyrics(
        self,
        audio: AudioData,
        lyrics: list[LyricLine],
        chunk_callback: Callable[[int, int], None] | None = None
    ) -> list[SyncResult]:
        """가사 싱크 (메인 API)"""
        
    def unload_model(self) -> None:
        """메모리 해제"""
        
    @property
    def is_loaded(self) -> bool: ...
    
    @property
    def memory_usage(self) -> dict: ...
```

### 구현 세부사항 (Transformers 백엔드)

```python
def load_model(self) -> None:
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    
    self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        self.config.model_path,
        torch_dtype=getattr(torch, self.config.torch_dtype),
        device_map=self.config.device_map,
        attn_implementation="flash_attention_2" if self.config.use_flash_attention else "eager",
    )
    # 가사 싱크에는 음성 출력 불필요 → 메모리 절약
    self.model.disable_talker()
    
    self.processor = Qwen2_5OmniProcessor.from_pretrained(self.config.model_path)

def infer(self, conversation: list[dict], **kwargs) -> str:
    from qwen_omni_utils import process_mm_info
    
    text = self.processor.apply_chat_template(
        conversation, 
        add_generation_prompt=True, 
        tokenize=False
    )
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
    
    inputs = self.processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True
    )
    inputs = inputs.to(self.model.device).to(self.model.dtype)
    
    text_ids, _ = self.model.generate(
        **inputs,
        return_audio=False,
        max_new_tokens=kwargs.get("max_tokens", 4096),
        temperature=kwargs.get("temperature", 0.1),
    )
    
    return self.processor.batch_decode(
        text_ids.sequences[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )[0]
```

### 긴 오디오 처리 (청킹)

```python
def sync_lyrics(self, audio: AudioData, lyrics: list[LyricLine], ...) -> list[SyncResult]:
    if audio.duration > self.config.max_audio_duration:
        raise AudioTooLongError(f"Audio exceeds {self.config.max_audio_duration}s limit")
    
    # 30분 이상이면 청킹
    if audio.duration > self.config.chunk_duration:
        return self._sync_with_chunks(audio, lyrics, chunk_callback)
    
    return self._sync_single(audio, lyrics)

def _sync_with_chunks(self, audio: AudioData, lyrics: list[LyricLine], ...) -> list[SyncResult]:
    loader = AudioLoader(self.config)
    chunks = loader.chunk_audio(audio, self.config.chunk_duration, overlap=30)
    
    all_results = []
    for i, (chunk, start_offset, end_offset) in enumerate(chunks):
        if chunk_callback:
            chunk_callback(i + 1, len(chunks))
        
        # 해당 시간대 가사만 추출 (대략적)
        chunk_lyrics = self._estimate_lyrics_for_chunk(lyrics, start_offset, end_offset, len(chunks), i)
        
        results = self._sync_single(chunk, chunk_lyrics)
        
        # 오프셋 적용
        for r in results:
            r.start_time += start_offset
            r.end_time += start_offset
        
        all_results.extend(results)
    
    return self._merge_overlapping_results(all_results)
```

### 의존성
- `transformers` (git install required)
- `qwen-omni-utils`
- `torch>=2.0`
- `flash-attn` (선택적)
- `accelerate`

### 에러 처리
| 에러 | 처리 |
|------|------|
| GPU OOM | 에러 메시지 + 권장 조치 |
| 모델 로드 실패 | 상세 에러 (의존성 체크) |
| 추론 타임아웃 | 설정 가능한 타임아웃 |
| 잘못된 응답 형식 | 재시도 1회 + 폴백 파서 |

### 테스트 방법
- Mock 모델로 단위 테스트
- 짧은 오디오로 통합 테스트
- 메모리 사용량 프로파일링

---

## 7. `everyric2/output/formatters.py`

### 목적
동기화 결과를 다양한 포맷으로 변환.

### 핵심 클래스/함수

```python
class BaseFormatter(ABC):
    @abstractmethod
    def format(self, results: list[SyncResult], metadata: dict | None = None) -> str: ...
    
    @abstractmethod
    def get_extension(self) -> str: ...

class SRTFormatter(BaseFormatter):
    """SubRip 포맷"""
    def format(self, results: list[SyncResult], metadata: dict | None = None) -> str: ...
    def get_extension(self) -> str: return "srt"

class ASSFormatter(BaseFormatter):
    """Advanced SubStation Alpha 포맷"""
    def format(self, results: list[SyncResult], metadata: dict | None = None) -> str: ...
    def get_extension(self) -> str: return "ass"
    
class LRCFormatter(BaseFormatter):
    """LRC 가사 포맷"""
    def format(self, results: list[SyncResult], metadata: dict | None = None) -> str: ...
    def get_extension(self) -> str: return "lrc"

class JSONFormatter(BaseFormatter):
    """JSON 포맷 (재활용 용이)"""
    def format(self, results: list[SyncResult], metadata: dict | None = None) -> str: ...
    def get_extension(self) -> str: return "json"

class FormatterFactory:
    @staticmethod
    def get_formatter(format_type: str) -> BaseFormatter:
        """포맷터 인스턴스 반환"""
        
    @staticmethod
    def get_supported_formats() -> list[str]:
        """지원 포맷 목록"""
```

### 포맷 예시

**SRT:**
```
1
00:00:05,230 --> 00:00:08,450
첫 번째 가사 라인

2
00:00:08,900 --> 00:00:12,100
두 번째 가사 라인
```

**LRC:**
```
[ti:Song Title]
[ar:Artist]
[00:05.23]첫 번째 가사 라인
[00:08.90]두 번째 가사 라인
```

**ASS:**
```
[Script Info]
Title: Song Title
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize, ...
Style: Default,Arial,20,...

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:05.23,0:00:08.45,Default,,0,0,0,,첫 번째 가사 라인
```

### 의존성
- 없음 (순수 Python)

### 테스트 방법
- 각 포맷 출력 검증
- 특수 문자 이스케이프 테스트
- 빈 결과 처리

---

## 8. `everyric2/cli.py`

### 목적
Typer 기반 CLI 인터페이스.

### 핵심 명령어

```python
import typer
from rich.console import Console
from rich.progress import Progress

app = typer.Typer(
    name="everyric2",
    help="Lyrics synchronization using Qwen3-Omni"
)
console = Console()

@app.command()
def sync(
    source: str = typer.Argument(..., help="YouTube URL or local audio file"),
    lyrics: Path = typer.Argument(..., help="Lyrics file (txt)"),
    output: Path = typer.Option(None, "-o", "--output", help="Output file path"),
    format: str = typer.Option("srt", "-f", "--format", help="Output format"),
    separate_vocals: bool = typer.Option(False, "--separate", help="Use Demucs vocal separation"),
    model: str = typer.Option(None, "-m", "--model", help="Model path override"),
):
    """Synchronize lyrics with audio."""
    ...

@app.command()
def formats():
    """List supported output formats."""
    ...

@app.command()
def info(source: str):
    """Show audio/video information."""
    ...

@app.command()  
def config():
    """Show current configuration."""
    ...

if __name__ == "__main__":
    app()
```

### CLI 사용 예시

```bash
# YouTube 영상에서 가사 싱크
everyric2 sync "https://youtube.com/watch?v=..." lyrics.txt -o output.srt

# 로컬 파일 + 보컬 분리
everyric2 sync song.mp3 lyrics.txt --separate -f ass -o output.ass

# LRC 포맷으로 출력
everyric2 sync audio.wav lyrics.txt -f lrc

# 지원 포맷 확인
everyric2 formats

# 영상 정보 확인
everyric2 info "https://youtube.com/watch?v=..."
```

### 의존성
- `typer>=0.9.0`
- `rich>=13.0.0` (progress bar, formatting)

### 에러 처리
- 사용자 친화적 에러 메시지 (rich formatting)
- Ctrl+C graceful shutdown
- 진행률 표시 (청킹 시)

---

## 9. 전체 데이터 흐름

```
┌─────────────────────────────────────────────────────────────────────┐
│                           CLI (cli.py)                              │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
            YouTube URL                       Local Audio
                    │                               │
                    ▼                               │
        ┌─────────────────────┐                     │
        │ downloader.py       │                     │
        │ (yt-dlp + ffmpeg)   │                     │
        └─────────────────────┘                     │
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
                    ┌─────────────────────┐
                    │ loader.py           │
                    │ (24kHz resample)    │
                    └─────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │               (--separate)     │
                    ▼                               ▼
            Raw Audio                    ┌─────────────────────┐
                    │                    │ separator.py        │
                    │                    │ (Demucs vocals)     │
                    │                    └─────────────────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
                    ┌─────────────────────┐
                    │ prompt.py           │
                    │ (build_conversation)│
                    └─────────────────────┘
                                    │
                                    ▼
                    ┌─────────────────────┐
                    │ qwen_omni.py        │
                    │ (Qwen3-Omni infer)  │
                    │ [chunking if >30min]│
                    └─────────────────────┘
                                    │
                                    ▼
                    ┌─────────────────────┐
                    │ formatters.py       │
                    │ (SRT/ASS/LRC/JSON)  │
                    └─────────────────────┘
                                    │
                                    ▼
                              Output File
```

---

## 10. 의존성 총정리

### requirements.txt
```
# Core
pydantic-settings>=2.0
typer>=0.9.0
rich>=13.0.0

# Audio
librosa>=0.10.0
soundfile>=0.12.0
yt-dlp>=2024.0.0

# ML/Inference
torch>=2.0
transformers>=4.40.0
accelerate
qwen-omni-utils

# Optional
demucs>=4.0.0      # vocal separation
flash-attn         # GPU acceleration
```

### System Dependencies
```
ffmpeg  # Required for yt-dlp and audio processing
```

---

## 11. GPU 메모리 요구사항

| 모델 | 정밀도 | 30초 오디오 | 5분 오디오 | 30분 오디오 |
|------|--------|-------------|------------|-------------|
| Qwen3-Omni-30B-A3B-Instruct | BF16 | ~80GB | ~90GB | ~130GB |
| AWQ-4bit (cpatonn) | INT4 | ~24GB | ~28GB | ~35GB |

**권장**: AWQ-4bit 모델 + `disable_talker()` → **~20-24GB VRAM**으로 실행 가능

---

## 12. 구현 우선순위

| 순서 | 모듈 | 이유 |
|------|------|------|
| 1 | `config/settings.py` | 다른 모듈의 기반 |
| 2 | `audio/loader.py` | 핵심 입력 처리 |
| 3 | `inference/prompt.py` | 모델 인터페이스 설계 |
| 4 | `inference/qwen_omni.py` | 핵심 기능 |
| 5 | `output/formatters.py` | 출력 처리 |
| 6 | `cli.py` | 사용자 인터페이스 |
| 7 | `audio/downloader.py` | YouTube 지원 |
| 8 | `audio/separator.py` | 선택적 기능 |

---

## 13. 결정 사항

### 확정된 사항
1. **모델**: `cpatonn/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit` (AWQ 4bit 양자화)
2. **샘플레이트**: 24kHz (Qwen-Omni 네이티브)
3. **청킹**: 30분 단위, 30초 오버랩
4. **백엔드**: transformers (1차), vLLM (추후)

### 미결정 사항
1. Whisper 폴백 여부
2. 테스트 데이터 확보

---

## 14. 폴더 구조

```
everyric2/
├── pyproject.toml
├── README.md
├── docs/
│   └── IMPLEMENTATION_PLAN.md
├── everyric2/
│   ├── __init__.py
│   ├── cli.py
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── downloader.py
│   │   └── separator.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── qwen_omni.py
│   │   └── prompt.py
│   └── output/
│       ├── __init__.py
│       └── formatters.py
└── tests/
    └── ...
```
