"""GGUF-based Qwen-Omni inference engine using llama.cpp server."""

import base64
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Callable

import requests

from everyric2.audio.loader import AudioData, AudioLoader
from everyric2.config.settings import ModelSettings, get_settings
from everyric2.inference.prompt import LyricLine, PromptBuilder, SyncResult


class InferenceError(Exception):
    pass


class ModelNotLoadedError(InferenceError):
    pass


class AudioTooLongError(InferenceError):
    pass


class QwenOmniGGUFEngine:
    """GGUF-based Qwen-Omni engine using llama.cpp server."""

    LLAMA_CPP_PATH = "/home/at192u/dev/llama-cpp-qwen3-omni/build/bin"
    DEFAULT_MODEL_PATH = "/mnt/d/models/qwen3-omni/thinker-q4_k_m.gguf"
    DEFAULT_MMPROJ_PATH = "/mnt/d/models/qwen3-omni/mmproj-f16.gguf"
    SERVER_PORT = 8081
    MAX_AUDIO_SECONDS = 30

    def __init__(self, config: ModelSettings | None = None):
        self.config = config or get_settings().model
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

        if audio.duration > self.MAX_AUDIO_SECONDS:
            return self._sync_with_chunks(audio, lyrics, progress_callback)

        return self._sync_single(audio, lyrics)

    def _sync_single(self, audio: AudioData, lyrics: list[LyricLine]) -> list[SyncResult]:
        temp_path = self.audio_loader.save_temp(audio, "sync_input.wav")

        try:
            audio_b64 = self._encode_audio_base64(temp_path)
            lyrics_text = "\n".join([f"{i + 1}. {line.text}" for i, line in enumerate(lyrics)])

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": audio_b64, "format": "wav"},
                        },
                        {
                            "type": "text",
                            "text": f"""Listen to this audio and synchronize these lyrics with timestamps.

LYRICS:
{lyrics_text}

For each lyric line, provide the start and end time in seconds.
Output as JSON array:
[{{"line": 1, "text": "...", "start": 0.0, "end": 2.5}}, ...]

Only output the JSON, no other text.""",
                        },
                    ],
                }
            ]

            response = self._call_api(messages)
            return self.prompt_builder.parse_response(response, lyrics)

        finally:
            if temp_path.exists():
                temp_path.unlink()

    def _sync_with_chunks(
        self,
        audio: AudioData,
        lyrics: list[LyricLine],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[SyncResult]:
        chunk_duration = self.MAX_AUDIO_SECONDS
        total_duration = audio.duration
        num_chunks = int(total_duration / chunk_duration) + 1
        lyrics_per_chunk = max(1, len(lyrics) // num_chunks)

        all_results: list[SyncResult] = []

        for i in range(num_chunks):
            if progress_callback:
                progress_callback(i + 1, num_chunks)

            start_time = i * chunk_duration
            end_time = min((i + 1) * chunk_duration, total_duration)

            start_sample = int(start_time * audio.sample_rate)
            end_sample = int(end_time * audio.sample_rate)
            chunk_waveform = audio.waveform[start_sample:end_sample]

            chunk_audio = AudioData(
                waveform=chunk_waveform,
                sample_rate=audio.sample_rate,
                duration=end_time - start_time,
            )

            start_idx = i * lyrics_per_chunk
            end_idx = min(start_idx + lyrics_per_chunk + 2, len(lyrics))
            if i == num_chunks - 1:
                end_idx = len(lyrics)

            chunk_lyrics = lyrics[start_idx:end_idx]
            if not chunk_lyrics:
                continue

            try:
                chunk_results = self._sync_single(chunk_audio, chunk_lyrics)
                for result in chunk_results:
                    result.start_time += start_time
                    result.end_time += start_time
                all_results.extend(chunk_results)
            except Exception as e:
                print(f"Chunk {i + 1} failed: {e}")
                continue

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
