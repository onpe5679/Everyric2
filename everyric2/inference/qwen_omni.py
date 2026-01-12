"""Qwen-Omni inference engine for lyrics synchronization."""

import os
from pathlib import Path
from typing import Callable

import torch

from everyric2.audio.loader import AudioChunk, AudioData, AudioLoader
from everyric2.config.settings import ModelSettings, get_settings
from everyric2.inference.prompt import LyricLine, PromptBuilder, SyncResult


class InferenceError(Exception):
    """Base exception for inference operations."""

    pass


class ModelNotLoadedError(InferenceError):
    """Raised when model is not loaded."""

    pass


class AudioTooLongError(InferenceError):
    """Raised when audio exceeds maximum duration."""

    pass


class QwenOmniEngine:
    """Qwen-Omni inference engine for lyrics synchronization."""

    def __init__(self, config: ModelSettings | None = None):
        """Initialize inference engine.

        Args:
            config: Model settings. If None, uses global settings.
        """
        self.config = config or get_settings().model
        self.prompt_builder = PromptBuilder()
        self.audio_loader = AudioLoader()

        self._model = None
        self._processor = None
        self._device = None

        # Set HuggingFace cache directory if specified
        if self.config.cache_dir:
            os.environ["HF_HOME"] = str(self.config.cache_dir)
            os.environ["TRANSFORMERS_CACHE"] = str(self.config.cache_dir / "hub")

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    def load_model(self) -> None:
        """Load model and processor.

        Raises:
            InferenceError: If loading fails.
        """
        if self.is_loaded:
            return

        try:
            from transformers import AutoConfig

            # Auto-detect model type from config
            config = AutoConfig.from_pretrained(
                self.config.path,
                cache_dir=str(self.config.cache_dir) if self.config.cache_dir else None,
                trust_remote_code=True,
            )
            model_type = config.model_type

            # Import appropriate classes based on model type
            if model_type == "qwen3_omni_moe":
                from transformers import (
                    Qwen3OmniMoeForConditionalGeneration as ModelClass,
                    Qwen3OmniMoeProcessor as ProcessorClass,
                )
            elif model_type == "qwen2_5_omni":
                from transformers import (
                    Qwen2_5OmniForConditionalGeneration as ModelClass,
                    Qwen2_5OmniProcessor as ProcessorClass,
                )
            else:
                raise InferenceError(f"Unsupported model type: {model_type}")

            # Determine torch dtype
            if self.config.torch_dtype == "auto":
                torch_dtype = "auto"
            else:
                torch_dtype = getattr(torch, self.config.torch_dtype)

            attn_impl = "flash_attention_2" if self.config.use_flash_attention else "eager"

            self._model = ModelClass.from_pretrained(
                self.config.path,
                dtype=torch_dtype,
                device_map="auto",
                attn_implementation=attn_impl,
                cache_dir=str(self.config.cache_dir) if self.config.cache_dir else None,
                trust_remote_code=True,
            )

            if hasattr(self._model, "disable_talker"):
                self._model.disable_talker()

            # Load processor
            self._processor = ProcessorClass.from_pretrained(
                self.config.path,
                cache_dir=str(self.config.cache_dir) if self.config.cache_dir else None,
                trust_remote_code=True,
            )

            # Get device
            self._device = next(self._model.parameters()).device

        except ImportError as e:
            raise InferenceError(
                "transformers with Qwen-Omni support not installed. "
                "Install with: pip install transformers>=4.50"
            ) from e
        except Exception as e:
            raise InferenceError(f"Failed to load model: {e}") from e

    def unload_model(self) -> None:
        """Unload model and free memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def infer(
        self,
        conversation: list[dict],
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Run inference on conversation.

        Args:
            conversation: Conversation in Qwen-Omni format.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Generated text response.

        Raises:
            ModelNotLoadedError: If model is not loaded.
            InferenceError: If inference fails.
        """
        if not self.is_loaded:
            raise ModelNotLoadedError("Model not loaded. Call load_model() first.")

        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature or self.config.temperature

        try:
            from qwen_omni_utils import process_mm_info

            # Apply chat template
            text = self._processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False,
            )

            audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)[:3]

            # Prepare inputs
            inputs = self._processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
            )

            # Move to device
            inputs = inputs.to(self._device)
            if hasattr(inputs, "to"):
                inputs = inputs.to(self._model.dtype)

            # Generate
            with torch.no_grad():
                output = self._model.generate(
                    **inputs,
                    return_audio=False,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                )

            if isinstance(output, tuple):
                text_ids = output[0]
            else:
                text_ids = output

            if hasattr(text_ids, "sequences"):
                text_ids = text_ids.sequences

            response = self._processor.batch_decode(
                text_ids[:, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )[0]

            return response

        except Exception as e:
            raise InferenceError(f"Inference failed: {e}") from e

    def sync_lyrics(
        self,
        audio: AudioData | Path | str,
        lyrics: list[LyricLine] | str | Path,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[SyncResult]:
        """Synchronize lyrics with audio.

        Args:
            audio: Audio data, or path to audio file.
            lyrics: List of LyricLine, lyrics text, or path to lyrics file.
            progress_callback: Callback for progress updates (current_chunk, total_chunks).

        Returns:
            List of SyncResult with synchronized timestamps.

        Raises:
            AudioTooLongError: If audio exceeds maximum duration.
            InferenceError: If synchronization fails.
        """
        # Load model if needed
        if not self.is_loaded:
            self.load_model()

        # Load audio if path
        if isinstance(audio, (str, Path)):
            audio = self.audio_loader.load(audio)

        # Check duration
        if audio.duration > self.config.max_audio_duration:
            raise AudioTooLongError(
                f"Audio duration ({audio.duration:.1f}s) exceeds maximum "
                f"({self.config.max_audio_duration}s). Use chunking for longer audio."
            )

        # Load lyrics if needed
        if isinstance(lyrics, (str, Path)):
            if Path(lyrics).exists():
                lyrics = LyricLine.from_file(lyrics)
            else:
                lyrics = LyricLine.from_text(str(lyrics))

        # Check if chunking is needed
        if audio.duration > self.config.chunk_duration:
            return self._sync_with_chunks(audio, lyrics, progress_callback)

        return self._sync_single(audio, lyrics)

    def _sync_single(
        self,
        audio: AudioData,
        lyrics: list[LyricLine],
    ) -> list[SyncResult]:
        """Synchronize single audio segment.

        Args:
            audio: Audio data.
            lyrics: List of lyric lines.

        Returns:
            List of SyncResult.
        """
        # Save audio to temp file (Qwen-Omni needs file path)
        temp_path = self.audio_loader.save_temp(audio, "sync_input.wav")

        try:
            # Build conversation
            conversation = self.prompt_builder.build_conversation(temp_path, lyrics)

            # Run inference
            response = self.infer(conversation)

            # Parse response
            results = self.prompt_builder.parse_response(response, lyrics)

            return results

        finally:
            # Cleanup temp file
            if temp_path.exists():
                temp_path.unlink()

    def _sync_with_chunks(
        self,
        audio: AudioData,
        lyrics: list[LyricLine],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[SyncResult]:
        """Synchronize long audio using chunking.

        Args:
            audio: Audio data.
            lyrics: List of lyric lines.
            progress_callback: Progress callback.

        Returns:
            List of SyncResult with adjusted timestamps.
        """
        all_results: list[SyncResult] = []
        chunks = list(self.audio_loader.chunk_audio(audio))
        total_chunks = len(chunks)

        # Estimate lyrics per chunk
        lyrics_per_chunk = max(1, len(lyrics) // total_chunks)

        for chunk in chunks:
            if progress_callback:
                progress_callback(chunk.chunk_index + 1, total_chunks)

            # Estimate which lyrics belong to this chunk
            start_idx = chunk.chunk_index * lyrics_per_chunk
            end_idx = min(
                start_idx + lyrics_per_chunk + 2,  # +2 for overlap
                len(lyrics),
            )

            # For last chunk, include remaining lyrics
            if chunk.chunk_index == total_chunks - 1:
                end_idx = len(lyrics)

            chunk_lyrics = lyrics[start_idx:end_idx]

            if not chunk_lyrics:
                continue

            # Sync this chunk
            chunk_results = self._sync_single(chunk.audio, chunk_lyrics)

            # Adjust timestamps by chunk offset
            for result in chunk_results:
                result.start_time += chunk.start_time
                result.end_time += chunk.start_time

            all_results.extend(chunk_results)

        # Remove duplicates (from overlapping chunks)
        return self._deduplicate_results(all_results)

    def _deduplicate_results(
        self,
        results: list[SyncResult],
    ) -> list[SyncResult]:
        """Remove duplicate results from overlapping chunks.

        Args:
            results: List of potentially duplicate results.

        Returns:
            Deduplicated list.
        """
        if not results:
            return results

        # Sort by start time
        results.sort(key=lambda r: r.start_time)

        # Remove duplicates (same text within 2 seconds)
        deduplicated = [results[0]]
        for result in results[1:]:
            last = deduplicated[-1]
            # Check if this is a duplicate
            if result.text == last.text and abs(result.start_time - last.start_time) < 2.0:
                # Keep the one with longer duration
                if (result.end_time - result.start_time) > (last.end_time - last.start_time):
                    deduplicated[-1] = result
            else:
                deduplicated.append(result)

        return deduplicated

    def get_memory_usage(self) -> dict:
        """Get current GPU memory usage.

        Returns:
            Dictionary with memory stats.
        """
        if not torch.cuda.is_available():
            return {"available": False}

        return {
            "available": True,
            "allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
            "reserved": torch.cuda.memory_reserved() / 1024**3,  # GB
            "max_allocated": torch.cuda.max_memory_allocated() / 1024**3,  # GB
        }
