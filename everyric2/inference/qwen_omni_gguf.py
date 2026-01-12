"""GGUF-based Qwen-Omni inference engine using llama.cpp server."""

import base64
import math
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import requests

from everyric2.audio.loader import AudioData, AudioLoader
from everyric2.config.settings import ModelSettings, get_settings
from everyric2.debug.debug_info import DebugInfo
from everyric2.inference.prompt import LyricLine, PromptBuilder, SyncResult


class InferenceError(Exception):
    pass


class ModelNotLoadedError(InferenceError):
    pass


class AudioTooLongError(InferenceError):
    pass


@dataclass
class ChunkContext:
    chunk_idx: int
    audio_offset: float
    total_duration: float
    all_lyrics: list[LyricLine]
    output_dir: Path | None = None


class QwenOmniGGUFEngine:
    """GGUF-based Qwen-Omni engine using llama.cpp server."""

    LLAMA_CPP_PATH = "/home/at192u/dev/llama-cpp-qwen3-omni/build/bin"
    DEFAULT_MODEL_PATH = "/mnt/d/models/qwen3-omni/thinker-q4_k_m.gguf"
    DEFAULT_MMPROJ_PATH = "/mnt/d/models/qwen3-omni/mmproj-f16.gguf"
    SERVER_PORT = 8081
    DEFAULT_CHUNK_DURATION = 90

    def __init__(self, config: ModelSettings | None = None, chunk_duration: int | None = None):
        self.config = config or get_settings().model
        self.chunk_duration = (
            chunk_duration or self.config.chunk_duration or self.DEFAULT_CHUNK_DURATION
        )
        self.prompt_builder = PromptBuilder()
        self.audio_loader = AudioLoader()
        self._server_process = None
        self._base_url = f"http://localhost:{self.SERVER_PORT}"

    @property
    def is_loaded(self) -> bool:
        try:
            resp = requests.get(f"{self._base_url}/health", timeout=2)
            return resp.status_code == 200
        except Exception:
            return False

    def load_model(self) -> None:
        if self.is_loaded:
            return

        model_path = self.DEFAULT_MODEL_PATH
        mmproj_path = self.DEFAULT_MMPROJ_PATH

        if not Path(model_path).exists():
            raise InferenceError(f"Model not found: {model_path}")
        if not Path(mmproj_path).exists():
            raise InferenceError(f"MMProj not found: {mmproj_path}")

        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = f"{self.LLAMA_CPP_PATH}:{env.get('LD_LIBRARY_PATH', '')}"

        cmd = [
            f"{self.LLAMA_CPP_PATH}/llama-server",
            "-m",
            model_path,
            "--mmproj",
            mmproj_path,
            "--port",
            str(self.SERVER_PORT),
            "-ngl",
            "99",
            "-c",
            "8192",
        ]

        self._server_process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        for _ in range(60):
            time.sleep(1)
            if self.is_loaded:
                return
            if self._server_process.poll() is not None:
                raise InferenceError("llama-server failed to start")

        raise InferenceError("llama-server startup timeout")

    def unload_model(self) -> None:
        if self._server_process:
            self._server_process.send_signal(signal.SIGTERM)
            self._server_process.wait(timeout=10)
            self._server_process = None

    def _encode_audio_base64(self, audio_path: Path) -> str:
        with open(audio_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _call_api(self, messages: list[dict], max_tokens: int = 4096) -> str:
        response = requests.post(
            f"{self._base_url}/v1/chat/completions",
            json={
                "model": "qwen3-omni",
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.1,
            },
            timeout=300,
        )

        if not response.ok:
            raise InferenceError(f"API error: {response.status_code} - {response.text}")

        return response.json()["choices"][0]["message"]["content"]

    def sync_lyrics(
        self,
        audio: AudioData | Path | str,
        lyrics: list[LyricLine] | str | Path,
        progress_callback: Callable[[int, int], None] | None = None,
        debug_info: DebugInfo | None = None,
    ) -> list[SyncResult]:
        if not self.is_loaded:
            self.load_model()

        if isinstance(audio, (str, Path)):
            audio = self.audio_loader.load(audio)

        if isinstance(lyrics, (str, Path)):
            if Path(lyrics).exists():
                lyrics = LyricLine.from_file(lyrics)
            else:
                lyrics = LyricLine.from_text(str(lyrics))

        if debug_info:
            debug_info.audio_duration = audio.duration
            debug_info.audio_sample_rate = audio.sample_rate

        if audio.duration > self.chunk_duration:
            return self._sync_with_chunks(audio, lyrics, progress_callback, debug_info)

        import time as time_module

        t0 = time_module.time()
        results, prompt, response = self._sync_single(audio, lyrics)
        processing_time = time_module.time() - t0

        if debug_info:
            debug_info.add_chunk(
                chunk_idx=0,
                audio_start=0.0,
                audio_end=audio.duration,
                lyrics_start_idx=0,
                lyrics_end_idx=len(lyrics),
                prompt=prompt,
                response=response,
                parsed_results=results,
                processing_time=processing_time,
            )

        return results

    def _sync_single(
        self,
        audio: AudioData,
        lyrics: list[LyricLine],
        *,
        ctx: ChunkContext | None = None,
    ) -> tuple[list[SyncResult], str, str]:
        temp_path = self.audio_loader.save_temp(audio, "sync_input.wav")

        try:
            audio_b64 = self._encode_audio_base64(temp_path)
            prompt_text = self._build_prompt(audio, lyrics, ctx)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": audio_b64, "format": "wav"},
                        },
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]

            response = self._call_api(messages)
            parse_lyrics = ctx.all_lyrics if ctx else lyrics
            results = self.prompt_builder.parse_response(response, parse_lyrics)

            return results, prompt_text, response

        finally:
            if temp_path.exists():
                temp_path.unlink()

    def _build_prompt(
        self,
        audio: AudioData,
        lyrics: list[LyricLine],
        ctx: ChunkContext | None = None,
    ) -> str:
        if ctx and ctx.audio_offset > 0:
            full_lyrics_text = "\n".join(
                [f"{i + 1}. {line.text}" for i, line in enumerate(ctx.all_lyrics)]
            )
            chunk_end = ctx.audio_offset + audio.duration
            return f"""This audio segment is from {ctx.audio_offset:.1f}s to {chunk_end:.1f}s of a {ctx.total_duration:.1f}s song.

FULL LYRICS OF THE SONG:
{full_lyrics_text}

Listen to this audio segment and identify which lyrics are sung.
Provide timestamps relative to the FULL SONG (not this segment).
For example, if a lyric starts 5 seconds into this segment, the start time should be {ctx.audio_offset:.1f} + 5 = {ctx.audio_offset + 5:.1f}s.

Output as JSON array:
[{{"line": 1, "text": "...", "start": {ctx.audio_offset:.1f}, "end": ...}}, ...]

Only output the JSON for lyrics heard in this segment. No other text."""

        lyrics_text = "\n".join([f"{i + 1}. {line.text}" for i, line in enumerate(lyrics)])
        return f"""Listen to this audio and synchronize these lyrics with timestamps.

LYRICS:
{lyrics_text}

For each lyric line, provide the start and end time in seconds.
Output as JSON array:
[{{"line": 1, "text": "...", "start": 0.0, "end": 2.5}}, ...]

Only output the JSON, no other text."""

    def _sync_with_chunks(
        self,
        audio: AudioData,
        lyrics: list[LyricLine],
        progress_callback: Callable[[int, int], None] | None = None,
        debug_info: DebugInfo | None = None,
    ) -> list[SyncResult]:
        chunk_duration = self.chunk_duration
        total_duration = audio.duration
        num_chunks = math.ceil(total_duration / chunk_duration)
        output_dir = debug_info.output_dir if debug_info else None

        all_results: list[SyncResult] = []

        for i in range(num_chunks):
            if progress_callback:
                progress_callback(i + 1, num_chunks)

            chunk_start = i * chunk_duration
            chunk_end = min((i + 1) * chunk_duration, total_duration)
            chunk_audio = self._extract_chunk(audio, chunk_start, chunk_end)

            ctx = ChunkContext(
                chunk_idx=i,
                audio_offset=chunk_start,
                total_duration=total_duration,
                all_lyrics=lyrics,
                output_dir=output_dir,
            )

            try:
                import time as time_module

                t0 = time_module.time()
                chunk_results, prompt, response = self._sync_single(chunk_audio, lyrics, ctx=ctx)
                processing_time = time_module.time() - t0

                all_results.extend(chunk_results)
                self._save_chunk_debug(ctx, chunk_audio, chunk_end, prompt, response)

                if debug_info:
                    debug_info.add_chunk(
                        chunk_idx=i,
                        audio_start=chunk_start,
                        audio_end=chunk_end,
                        lyrics_start_idx=0,
                        lyrics_end_idx=len(lyrics),
                        prompt=prompt,
                        response=response,
                        parsed_results=chunk_results,
                        processing_time=processing_time,
                    )

            except Exception as e:
                if debug_info:
                    debug_info.add_error(f"Chunk {i + 1} failed: {e}")
                print(f"Chunk {i + 1} failed: {e}")
                continue

        return self._deduplicate_results(all_results)

    def _extract_chunk(self, audio: AudioData, start: float, end: float) -> AudioData:
        start_sample = int(start * audio.sample_rate)
        end_sample = int(end * audio.sample_rate)
        return AudioData(
            waveform=audio.waveform[start_sample:end_sample],
            sample_rate=audio.sample_rate,
            duration=end - start,
        )

    def _save_chunk_debug(
        self,
        ctx: ChunkContext,
        audio: AudioData,
        chunk_end: float,
        prompt: str,
        response: str,
    ) -> None:
        if not ctx.output_dir:
            return
        chunks_dir = ctx.output_dir / "chunks"
        chunks_dir.mkdir(exist_ok=True)
        prefix = f"chunk_{ctx.chunk_idx:03d}_{ctx.audio_offset:.1f}s-{chunk_end:.1f}s"
        audio.to_file(chunks_dir / f"{prefix}.wav")
        (chunks_dir / f"chunk_{ctx.chunk_idx:03d}_prompt.txt").write_text(prompt, encoding="utf-8")
        (chunks_dir / f"chunk_{ctx.chunk_idx:03d}_response.txt").write_text(
            response, encoding="utf-8"
        )

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
