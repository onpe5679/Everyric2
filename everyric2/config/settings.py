"""Configuration settings for Everyric2."""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelSettings(BaseSettings):
    """Model configuration."""

    model_config = SettingsConfigDict(env_prefix="EVERYRIC_MODEL_")

    # Model path - can be HuggingFace hub ID or local path
    path: str = Field(
        default="cpatonn/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit",
        description="HuggingFace model ID or local path",
    )

    # HuggingFace cache directory (for D: drive sharing)
    cache_dir: Path | None = Field(
        default=None,
        description="HuggingFace cache directory. Set to /mnt/d/huggingface_cache for WSL.",
    )

    # Inference settings
    device_map: str = Field(default="auto", description="Device mapping strategy")
    torch_dtype: Literal["float16", "bfloat16", "float32", "auto"] = Field(
        default="auto", description="Torch dtype for model weights"
    )
    use_flash_attention: bool = Field(
        default=True, description="Use Flash Attention 2 if available"
    )

    # Generation settings
    max_new_tokens: int = Field(default=4096, description="Maximum tokens to generate")
    temperature: float = Field(default=0.1, description="Sampling temperature")

    # Audio limits
    max_audio_duration: int = Field(
        default=2400, description="Maximum audio duration in seconds (40 min)"
    )
    chunk_duration: int = Field(
        default=1800, description="Chunk duration for long audio in seconds (30 min)"
    )
    chunk_overlap: int = Field(default=30, description="Overlap between chunks in seconds")


class AudioSettings(BaseSettings):
    """Audio processing configuration."""

    model_config = SettingsConfigDict(env_prefix="EVERYRIC_AUDIO_")

    # Sample rate - Qwen-Omni native is 24kHz
    target_sample_rate: int = Field(default=24000, description="Target sample rate for model input")

    # Demucs settings
    demucs_model: str = Field(default="htdemucs", description="Demucs model for vocal separation")

    # Temp directory
    temp_dir: Path = Field(
        default=Path("/tmp/everyric2"), description="Temporary directory for processing"
    )

    # YouTube cookie settings
    cookies_from_browser: str | None = Field(
        default=None,
        description="Browser to extract cookies from (chrome, firefox, edge, brave, opera, chromium)",
    )
    cookie_file: Path | None = Field(
        default=None, description="Path to Netscape format cookie file"
    )

    # Multi-NIC download routing
    source_address: str | None = Field(
        default=None,
        description="Local IP to bind yt-dlp connections to. On multi-NIC machines this "
        "routes downloads through a different public IP when YouTube throttles the "
        "default one with HTTP 403",
    )

    @field_validator("temp_dir", mode="after")
    @classmethod
    def ensure_temp_dir_exists(cls, v: Path) -> Path:
        """Ensure temp directory exists."""
        v.mkdir(parents=True, exist_ok=True)
        return v


class AlignmentSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EVERYRIC_ALIGNMENT_")

    engine: Literal["ctc", "nemo", "gpu-hybrid", "sofa"] = Field(
        default="ctc", description="Alignment engine to use"
    )
    language: Literal["auto", "en", "ja", "ko"] = Field(
        default="auto", description="Language for transcription/alignment"
    )

    nemo_model_en: str = Field(
        default="nvidia/stt_en_conformer_ctc_large",
        description="NeMo model for English",
    )

    alignment_sample_rate: int = Field(
        default=16000, description="Sample rate for alignment engines"
    )

    align_on_vocals: bool = Field(
        default=True,
        description="Run CTC alignment on the demucs-separated vocal stem instead of the full "
        "mix — the original CLI design (--separate swapped the audio before alignment). CTC "
        "emissions are far cleaner without instrumentals, which matters most on dense mixes. "
        "Separation output is reused for VAD clamping and melody f0 either way, so enabling "
        "this adds no extra compute. Falls back to the mix when demucs is unavailable.",
    )

    star_guard_splice: bool = Field(
        default=True,
        description="When the star-swallow guard confirms the ko alignment compressed the "
        "post-interlude block forward, splice instead of discarding the whole ko alignment: "
        "keep ko (syllable-accurate) timings for lines before the interlude and take the "
        "original-text alignment's timings for lines it places after the interlude. Falls back "
        "to the full original-text alignment when the splice boundary is degenerate.",
    )

    star_tokens: bool = Field(
        default=True,
        description="Insert wildcard <star> tokens between lyric lines during CTC alignment "
        "so ad-libs/repeats not present in the lyrics are absorbed instead of "
        "stretching neighboring lines",
    )

    use_pronunciation: bool = Field(
        default=True,
        description="When line-level Korean pronunciation (독음, e.g. from the Vocaloid lyrics "
        "wiki) covers enough of the lyrics, run CTC forced alignment on the pronunciation "
        "text with the MMS 'kor' adapter and map the resulting syllable timings back onto the "
        "original lines. On synthesized/Vocaloid vocals this lifts alignment confidence and "
        "fixes gross post-interlude misplacement that local clamps cannot repair, and yields "
        "syllable-level spans that split multi-mora kanji into separate karaoke notes. "
        "Falls back to original-text alignment when coverage is insufficient or it fails.",
    )

    star_vocal_fallback_sec: float = Field(
        default=8.0,
        description="Cost gate for the pronunciation (ko) alignment guard. When a single wildcard "
        "<star> span absorbs at least this many seconds of real VAD vocal activity, the ko "
        "alignment may have compressed genuine lyric lines out of that region (kor adapter "
        "failing on a heavy-effect section). Swallow magnitude alone cannot tell 'compressed real "
        "lyrics' from 'a genuine lyric-free bridge' (熱異常 swallows ~21s benignly), so exceeding "
        "this only triggers the definitive cross-check (post_interlude_fill_margin_sec) rather "
        "than a fallback. Its purpose is to avoid running a second alignment on songs that "
        "clearly do not need it. Set to 0 to disable the guard entirely.",
    )

    interlude_min_gap_sec: float = Field(
        default=5.0,
        description="Minimum silence gap (seconds) between consecutive VAD vocal regions to count "
        "as a structural interlude. The largest such gap anchors the post-interlude window used by "
        "the ko-alignment fallback cross-check. Songs without a gap this long skip the check.",
    )

    post_interlude_fill_margin_sec: float = Field(
        default=15.0,
        description="Decision threshold for the ko-alignment star-swallow guard. Once the swallow "
        "gate trips and an interlude exists, the original-text (ja) alignment is run and both "
        "alignments are measured by how many seconds of lyric lines they place in the "
        "post-interlude vocal window. If ja fills at least this many seconds MORE than ko, the ko "
        "path compressed the post-interlude block forward (out of the window) → fall back to "
        "original-text alignment for the whole song. Anchoring on the interlude (fixed by the "
        "audio) rather than the star span (which moves between alignments) makes this robust: "
        "初音ミクの消失 shows ja−ko = +46.7 to +79.4s across runs (falls back), 熱異常 shows "
        "−1.3 to +5.5s (keeps ko) — VAD boundaries drift between runs but the separation stays "
        "enormous.",
    )


class TranslationSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EVERYRIC_TRANSLATE_")

    engine: Literal["gemini", "openai", "local", "nvidia"] = Field(
        default="gemini", description="Translation engine"
    )
    model: str = Field(default="gemini-2.0-flash", description="Model name")
    nvidia_model: str = Field(
        default="openai/gpt-oss-120b",
        description="Model name for the NVIDIA NIM engine (separate from `model` so the "
        "gemini default doesn't leak into NIM requests). 2026-07 실측: gpt-oss-120b가 "
        "ja→ko 30줄 기준 오역 0·22s로 최선 (qwen3.5-122b는 2026-07-20 EOL, "
        "qwen3-next-80b는 君→쿤 오독·정반대 오역, deepseek-v4-pro는 장문 120s 타임아웃)",
    )
    api_url: str | None = Field(default=None, description="Custom API URL for local LLM")
    api_key: str | None = Field(default=None, description="API key (env var takes precedence)")
    tone: Literal["literal", "natural", "poetic", "casual", "formal"] = Field(
        default="natural", description="Translation tone/style"
    )
    temperature: float = Field(default=0.3, description="Generation temperature")
    include_pronunciation: bool = Field(
        default=False, description="Include pronunciation transcription"
    )
    pronunciation_format: Literal["parentheses", "brackets", "newline"] = Field(
        default="parentheses", description="Pronunciation display format"
    )
    target_language: str = Field(default="ko", description="Target language for translation")
    timeout: int = Field(default=120, description="API timeout in seconds")
    max_tokens: int = Field(
        default=8192,
        description="Max completion tokens for OpenAI-compatible chat endpoints (openai/local/"
        "nvidia). Without this, some NIM-hosted models default to a small completion budget "
        "and truncate the pronunciation JSON array mid-response for multi-line lyrics. "
        "reasoning 모델(gpt-oss 등)은 사고 토큰이 이 예산을 같이 쓰므로 4096이면 "
        "30줄 곡에서 JSON이 잘렸다 — 8192로 상향.",
    )


class SegmentationSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EVERYRIC_SEGMENT_")

    mode: Literal["line", "word", "character"] = Field(
        default="line", description="Segmentation mode"
    )
    min_duration: float = Field(default=0.2, description="Minimum segment duration in seconds")
    max_chars_per_segment: int = Field(
        default=50, description="Maximum characters per segment (for auto-split in line mode)"
    )
    min_silence_gap: float = Field(
        default=0.3, description="Minimum silence gap between segments. Shorter gaps are merged"
    )
    silence_merge_mode: Literal["midpoint", "extend_prev", "extend_next"] = Field(
        default="midpoint", description="How to merge short silence gaps"
    )
    interlude_gap: float = Field(
        default=5.0, description="Gaps longer than this are treated as interludes (no subtitle)"
    )
    use_mecab: bool = Field(default=True, description="Use MeCab for Japanese word segmentation")


class MelodySettings(BaseSettings):
    """Vocal melody extraction (karaoke pitch bar) configuration."""

    model_config = SettingsConfigDict(env_prefix="EVERYRIC_MELODY_")

    enabled: bool = Field(
        default=True, description="Annotate word timestamps with MIDI notes"
    )
    separate_vocals: bool = Field(
        default=True,
        description="Run demucs vocal separation before f0 extraction "
        "(mix tracks bleed accompaniment pitch into notes; falls back to mix if unavailable)",
    )
    device: str = Field(default="auto", description="Inference device: auto, cpu, cuda")
    f0_model: Literal["fcpe", "rmvpe"] = Field(
        default="rmvpe",
        description="f0 estimation backend. rmvpe (DeepUnet+BiGRU, singing-pitch SOTA) "
        "measured lower subharmonic lock-on (-12 semitone mass ratio 0.44 vs FCPE's 0.69) "
        "and fewer large frame-to-frame jumps on a real karaoke track A/B; falls back to "
        "FCPE automatically if the rmvpe.pt weights are missing or fail to load.",
    )
    rmvpe_model_path: Path = Field(
        default=Path(__file__).resolve().parents[2] / "models" / "rmvpe" / "rmvpe.pt",
        description="Path to RMVPE weights (rmvpe.pt, ~180MB, MIT-licensed inference code "
        "ported from RVC-Project, weights from HuggingFace lj1995/VoiceConversionWebUI). "
        "Not bundled with the repo; download separately.",
    )
    rmvpe_threshold: float = Field(
        default=0.01,
        description="RMVPE unvoiced salience cutoff (0-1 sigmoid). Lower than the RVC "
        "default of 0.03 — measured to raise line-span voiced coverage to ~FCPE parity "
        "(0.90 vs 0.905 mean) without degrading octave-lock-on or jump-rate metrics.",
    )
    threshold: float = Field(
        default=0.006, description="FCPE unvoiced detection threshold"
    )
    f0_min: float = Field(default=65.0, description="Minimum f0 in Hz (~C2)")
    f0_max: float = Field(default=1100.0, description="Maximum f0 in Hz (~C6)")
    octave_snap: bool = Field(
        default=True,
        description="Fold octave/harmonic jumps (>7 semitones vs previous voiced frame) "
        "back toward the melodic trajectory before note quantization "
        "(fixes FCPE octave lock-on; measured 37%→5% large-jump rate)",
    )
    key_detect: bool = Field(
        default=True,
        description="Estimate the song key (Krumhansl-Schmuckler pitch-class correlation) "
        "and store it with the sync for karaoke display",
    )
    key_snap: bool = Field(
        default=True,
        description="Snap out-of-scale notes whose span f0 median sits near the semitone "
        "rounding boundary to the in-scale neighbor (skipped when key confidence is low; "
        "clear chromatic passing notes are preserved)",
    )
    anchor_to_words: bool = Field(
        default=True,
        description="Cut notes at aligned character (syllable) boundaries instead of free "
        "f0-stability runs, so note timing locks to the lyric alignment the user sees",
    )
    min_note_sec: float = Field(
        default=0.08, description="Minimum stable duration for a note segment"
    )
    max_gap_sec: float = Field(
        default=0.12, description="Unvoiced gap shorter than this stays in the same note"
    )
    min_voiced_ratio: float = Field(
        default=0.15, description="Skip spans whose voiced frame ratio is below this"
    )


class OutputSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EVERYRIC_OUTPUT_")

    default_format: Literal["srt", "ass", "lrc", "json"] = Field(
        default="srt", description="Default output format"
    )
    generate_all_variants: bool = Field(
        default=False, description="Generate all output variants (original, translated, etc.)"
    )


class ServerSettings(BaseSettings):
    """API server configuration."""

    model_config = SettingsConfigDict(env_prefix="EVERYRIC_SERVER_")

    host: str = Field(
        default="127.0.0.1",
        description="Server bind host. Default loopback-only; set 0.0.0.0 explicitly "
        "to expose on the LAN (combine with api_key).",
    )
    port: int = Field(default=8000, description="Server port")
    reload: bool = Field(default=False, description="Enable auto-reload for development")
    workers: int = Field(default=1, description="Number of worker processes")
    api_key: str = Field(
        default="",
        description="When set, every /api request must present this value (or the admin "
        "key) in X-API-Key. Empty = no auth (local single-user default).",
    )
    max_job_audio_sec: int = Field(
        default=1800,
        description="Maximum audio duration (seconds) accepted for sync generation. Longer "
        "videos (podcasts, live archives, hours-long loops) would monopolize the single GPU "
        "slot for hours with no way to cancel mid-alignment; the job now fails fast with a "
        "friendly message right after download instead. 0 disables the cap.",
    )

    max_concurrent_jobs: int = Field(
        default=1,
        description="Max sync-generation jobs processed at once. Alignment+separation+"
        "melody hold significant GPU/RAM; excess jobs wait in a queue (status=queued).",
    )
    admin_api_key: str = Field(
        default="",
        description="Admin API key (X-API-Key). When set, destructive actions "
        "(force regenerate, sync reset) from other callers are rate-limited; "
        "requests presenting this key bypass the limit. Empty = no limits (local use).",
    )
    daily_destructive_limit: int = Field(
        default=2,
        description="Max force-regenerations/resets per video per 24h for non-admin "
        "callers (only enforced when admin_api_key is set). 0 disables the limit.",
    )
    worker_key: str = Field(
        default="",
        description="원격 GPU 워커 풀 인증 키 (X-Worker-Key). 설정하면 /api/worker/* "
        "엔드포인트가 켜져 원격 워커가 잡을 클레임·처리한다. 빈 값이면 워커 API 전체가 "
        "403 (기능 비활성). 개인 풀 모델이라 워커들이 이 키 하나를 공유하고 worker_id로 "
        "머신을 구분한다.",
    )
    local_worker: bool = Field(
        default=True,
        description="서버 프로세스가 직접 생성 파이프라인을 돌릴지 여부. True면 기존처럼 "
        "인프로세스로 처리한다. False면 GPU 없는 API 전용 서버로 보고, 생성 잡을 add_task "
        "없이 status=queued로만 마킹해 원격 워커가 클레임하도록 둔다 (queue_position 표시 유지).",
    )
    worker_lease_sec: int = Field(
        default=120,
        description="원격 워커가 클레임한 잡의 리스 만료(초). 진행률 보고(≤2s 간격)가 "
        "하트비트를 겸해 리스를 갱신한다. 만료되면(워커 하트비트 끊김) 다음 claim 처리 "
        "시 잡을 queued로 되돌려 다른 워커가 다시 가져가게 한다.",
    )
    # ── 중립 연동 (외부 곡 인덱스 / 외부 미디어 캐시) ──────────────────────────
    song_index_url: str = Field(
        default="",
        description="외부 곡 인덱스(songindex/1)의 베이스 URL. 설정하면 /api/vocaro/match를 "
        "업스트림 GET {url}/match?title=... 프록시로 전환한다(확장 응답 형태 무변경). 빈 값이면 "
        "기존 로컬 인덱스 경로 그대로.",
    )
    song_index_key: str = Field(
        default="",
        description="외부 곡 인덱스 인증 키 — 프록시 요청에 Authorization: Bearer <key>로 실린다.",
    )
    media_cache_url: str = Field(
        default="",
        description="외부 미디어 캐시(mediacache/1)의 베이스 URL. 설정하면 잡이 처리 주체에게 "
        "넘어가는 순간 GET {url}/lookup?platform=youtube&id=<video_id>로 조회해, 히트 시 "
        "재다운로드 없이 로컬 원본에서 오디오만 추출해 쓴다. 빈 값이면 항상 yt-dlp 경로.",
    )
    media_cache_key: str = Field(
        default="",
        description="외부 미디어 캐시 인증 키 — 조회 요청에 Authorization: Bearer <key>로 실린다.",
    )
    link_match_threshold: float = Field(
        default=0.55,
        description="반주 상관 링크 검증(link-jobs)에서 match로 판정하는 confidence(정규화 "
        "상관 최고 피크 절대높이) 하한. 실측 캘리브레이션(2026-07-24): 동일 인스트 커버 "
        "0.93, 무관 곡 쌍 0.02 — 0.55는 그 사이의 보수적 경계다.",
    )
    worker_vram_guard_gb: float = Field(
        default=8.0,
        description="잡 경계 VRAM 회수(empty_cache) 후에도 예약이 이 값(GiB)을 넘으면 참조 "
        "누수 회귀로 보고 웜 모델 캐시를 버리고 재적재한다. 동거 호스트 실측(2026-07-24): "
        "모델 실중량 3~6GiB, 앨로케이터 사재기 방치 시 18.4GiB까지 부풂. 0 = 가드 비활성.",
    )
    link_min_offset_margin: float = Field(
        default=0.08,
        description="링크 검증에서 오프셋 유일성 게이트 — (최고 피크 − 이차 피크)가 이 값 "
        "미만이면 confidence가 높아도 자동 링크를 보류한다. 루프 구조 곡은 마디 간격의 "
        "이차 피크가 최고 피크에 근접하는데, 그 간극이 너무 작으면 이웃 박자 오프셋을 "
        "잘못 고를 위험이 있다 (틀린 오프셋 링크는 no-link보다 해롭다).",
    )
    warm_models: bool = Field(
        default=True,
        description="생성 파이프라인의 무거운 모델(demucs 분리기·CTC 정렬 엔진·멜로디 f0 "
        "백엔드)을 프로세스 수명 동안 지연 싱글턴으로 상주시켜 두 번째 잡부터 재로드 0회로 "
        "만든다. 상주 주체는 원격 워커(CLI)와 인프로세스 서버뿐 — API 전용 모드(local_worker="
        "false)는 생성을 돌리지 않으므로 어떤 모델도 로드되지 않는다(torch 지연 임포트 불변). "
        "false면 기존처럼 잡마다 인스턴스를 새로 만든다.",
    )


class Settings(BaseSettings):
    """Main settings container."""

    model_config = SettingsConfigDict(
        env_prefix="EVERYRIC_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model: ModelSettings = Field(default_factory=ModelSettings)
    audio: AudioSettings = Field(default_factory=AudioSettings)
    alignment: AlignmentSettings = Field(default_factory=AlignmentSettings)
    translation: TranslationSettings = Field(default_factory=TranslationSettings)
    segmentation: SegmentationSettings = Field(default_factory=SegmentationSettings)
    melody: MelodySettings = Field(default_factory=MelodySettings)
    output: OutputSettings = Field(default_factory=OutputSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)

    # Debug mode
    debug: bool = Field(default=False, description="Enable debug mode")


# Global settings instance (lazy loaded)
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Reset settings (useful for testing)."""
    global _settings
    _settings = None
