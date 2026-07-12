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
        default="qwen/qwen3.5-122b-a10b",
        description="Model name for the NVIDIA NIM engine (separate from `model` so the "
        "gemini default doesn't leak into NIM requests)",
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
        default=4096,
        description="Max completion tokens for OpenAI-compatible chat endpoints (openai/local/"
        "nvidia). Without this, some NIM-hosted models default to a small completion budget "
        "and truncate the pronunciation JSON array mid-response for multi-line lyrics.",
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
