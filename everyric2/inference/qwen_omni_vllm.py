"""Qwen-Omni vLLM inference engine for NVFP4 model."""

import base64
import os
from pathlib import Path
from typing import Callable

import torch

from everyric2.audio.loader import AudioData, AudioLoader
from everyric2.config.settings import ModelSettings, get_settings
from everyric2.inference.prompt import LyricLine, PromptBuilder, SyncResult


class InferenceError(Exception):
    pass


class ModelNotLoadedError(InferenceError):
    pass


class AudioTooLongError(InferenceError):
    pass


class QwenOmniVLLMEngine:
    """vLLM-based Qwen-Omni engine for NVFP4 quantized models."""

    def __init__(self, config: ModelSettings | None = None):
        self.config = config or get_settings().model
        self.prompt_builder = PromptBuilder()
        self.audio_loader = AudioLoader()

        self._llm = None
        self._processor = None

        if self.config.cache_dir:
            os.environ["HF_HOME"] = str(self.config.cache_dir)

    @property
    def is_loaded(self) -> bool:
        return self._llm is not None

    def load_model(self) -> None:
        if self.is_loaded:
            return

        try:
            from vllm import LLM

            model_path = self.config.path
            if "NVFP4" in model_path or "nvfp4" in model_path.lower():
                quantization = "modelopt_fp4"
            else:
                quantization = None

            self._llm = LLM(
                model=model_path,
                quantization=quantization,
                trust_remote_code=True,
                kv_cache_dtype="fp8" if quantization else "auto",
                gpu_memory_utilization=0.95,
                max_model_len=8192,
            )

            from transformers import AutoProcessor

            self._processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
            )

        except Exception as e:
            raise InferenceError(f"Failed to load vLLM model: {e}") from e

    def unload_model(self) -> None:
        if self._llm is not None:
            del self._llm
            self._llm = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _encode_audio_base64(self, audio_path: Path) -> str:
        with open(audio_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def infer(
        self,
        conversation: list[dict],
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        if not self.is_loaded:
            raise ModelNotLoadedError("Model not loaded. Call load_model() first.")

        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature or self.config.temperature

        try:
            from vllm import SamplingParams

            text = self._processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False,
            )

            sampling_params = SamplingParams(
                max_tokens=max_new_tokens,
                temperature=temperature,
            )

            outputs = self._llm.generate(
                [{"prompt": text, "multi_modal_data": self._extract_mm_data(conversation)}],
                sampling_params,
            )

            return outputs[0].outputs[0].text

        except Exception as e:
            raise InferenceError(f"Inference failed: {e}") from e

    def _extract_mm_data(self, conversation: list[dict]) -> dict:
        mm_data = {}
        for msg in conversation:
            if isinstance(msg.get("content"), list):
                for item in msg["content"]:
                    if item.get("type") == "audio":
                        audio_path = item.get("audio")
                        if audio_path and Path(audio_path).exists():
                            import librosa

                            audio_array, sr = librosa.load(audio_path, sr=24000, mono=True)
                            mm_data["audio"] = [(audio_array, sr)]
        return mm_data

    def sync_lyrics(
        self,
        audio: AudioData | Path | str,
        lyrics: list[LyricLine] | str | Path,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[SyncResult]:
        if not self.is_loaded:
            self.load_model()

        if isinstance(audio, (str, Path)):
            audio = self.audio_loader.load(audio)

        if audio.duration > self.config.max_audio_duration:
            raise AudioTooLongError(
                f"Audio duration ({audio.duration:.1f}s) exceeds maximum "
                f"({self.config.max_audio_duration}s)."
            )

        if isinstance(lyrics, (str, Path)):
            if Path(lyrics).exists():
                lyrics = LyricLine.from_file(lyrics)
            else:
                lyrics = LyricLine.from_text(str(lyrics))

        if audio.duration > self.config.chunk_duration:
            return self._sync_with_chunks(audio, lyrics, progress_callback)

        return self._sync_single(audio, lyrics)

    def _sync_single(
        self,
        audio: AudioData,
        lyrics: list[LyricLine],
    ) -> list[SyncResult]:
        temp_path = self.audio_loader.save_temp(audio, "sync_input.wav")

        try:
            conversation = self.prompt_builder.build_conversation(temp_path, lyrics)
            response = self.infer(conversation)
            results = self.prompt_builder.parse_response(response, lyrics)
            return results
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def _sync_with_chunks(
        self,
        audio: AudioData,
        lyrics: list[LyricLine],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[SyncResult]:
        all_results: list[SyncResult] = []
        chunks = list(self.audio_loader.chunk_audio(audio))
        total_chunks = len(chunks)
        lyrics_per_chunk = max(1, len(lyrics) // total_chunks)

        for chunk in chunks:
            if progress_callback:
                progress_callback(chunk.chunk_index + 1, total_chunks)

            start_idx = chunk.chunk_index * lyrics_per_chunk
            end_idx = min(start_idx + lyrics_per_chunk + 2, len(lyrics))

            if chunk.chunk_index == total_chunks - 1:
                end_idx = len(lyrics)

            chunk_lyrics = lyrics[start_idx:end_idx]
            if not chunk_lyrics:
                continue

            chunk_results = self._sync_single(chunk.audio, chunk_lyrics)

            for result in chunk_results:
                result.start_time += chunk.start_time
                result.end_time += chunk.start_time

            all_results.extend(chunk_results)

        return self._deduplicate_results(all_results)

    def _deduplicate_results(self, results: list[SyncResult]) -> list[SyncResult]:
        if not results:
            return results

        results.sort(key=lambda r: r.start_time)
        deduplicated = [results[0]]

        for result in results[1:]:
            last = deduplicated[-1]
            if result.text == last.text and abs(result.start_time - last.start_time) < 2.0:
                if (result.end_time - result.start_time) > (last.end_time - last.start_time):
                    deduplicated[-1] = result
            else:
                deduplicated.append(result)

        return deduplicated
