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

    star_tokens: bool = Field(
        default=True,
        description="Insert wildcard <star> tokens between lyric lines during CTC alignment "
        "so ad-libs/repeats not present in the lyrics are absorbed instead of "
        "stretching neighboring lines",
    )


class TranslationSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EVERYRIC_TRANSLATE_")

    engine: Literal["gemini", "openai", "local"] = Field(
        default="gemini", description="Translation engine"
    )
    model: str = Field(default="gemini-2.0-flash", description="Model name")
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

    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    reload: bool = Field(default=False, description="Enable auto-reload for development")
    workers: int = Field(default=1, description="Number of worker processes")


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
