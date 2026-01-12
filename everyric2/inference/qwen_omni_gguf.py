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
    chunk_end: float
    total_duration: float
    all_lyrics: list[LyricLine]

    last_matched_line_idx: int | None = None
    last_matched_line_text: str | None = None
    last_matched_end_time: float | None = None

    expected_lyrics_start_idx: int = 0
    expected_lyrics_end_idx: int | None = None

    output_dir: Path | None = None

    @property
    def expected_lyrics(self) -> list[LyricLine]:
        end_idx = self.expected_lyrics_end_idx or len(self.all_lyrics)
        return self.all_lyrics[self.expected_lyrics_start_idx : end_idx]

    @property
    def has_anchor(self) -> bool:
        return self.last_matched_line_idx is not None


class QwenOmniGGUFEngine:
    LLAMA_CPP_PATH = "/home/at192u/dev/llama-cpp-qwen3-omni/build/bin"
    DEFAULT_MODEL_PATH = "/mnt/d/models/qwen3-omni/thinker-q4_k_m.gguf"
    DEFAULT_MMPROJ_PATH = "/mnt/d/models/qwen3-omni/mmproj-f16.gguf"
    SERVER_PORT = 8081
    DEFAULT_CHUNK_DURATION = 90
    DEFAULT_CHUNK_OVERLAP = 5.0

    def __init__(
        self,
        config: ModelSettings | None = None,
        chunk_duration: int | None = None,
        chunk_overlap: float | None = None,
    ):
        self.config = config or get_settings().model
        self.chunk_duration = (
            chunk_duration or self.config.chunk_duration or self.DEFAULT_CHUNK_DURATION
        )
        self.chunk_overlap = (
            chunk_overlap if chunk_overlap is not None else self.DEFAULT_CHUNK_OVERLAP
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
            system_prompt = self._get_system_prompt()

            messages = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": audio_b64, "format": "wav"},
                        },
                        {"type": "text", "text": prompt_text},
                    ],
                },
            ]

            response = self._call_api(messages)
            parse_lyrics = ctx.all_lyrics if ctx else lyrics
            results = self.prompt_builder.parse_response(response, parse_lyrics)

            return results, prompt_text, response

        finally:
            if temp_path.exists():
                temp_path.unlink()

    def _get_system_prompt(self) -> str:
        return """You are a precise audio-to-lyrics alignment assistant.

CRITICAL RULES:
1. You MUST actually LISTEN to the audio. Do NOT guess based on text alone.
2. Output timestamps are RELATIVE to this audio segment (0.0s = start of this audio).
3. If you hear silence, instrumental, or no vocals: output empty array [].
4. If you're uncertain whether a lyric is sung: do NOT include it.
5. Do NOT output all expected lyrics - only those you ACTUALLY HEAR in the audio.
6. Timestamps must match when you HEAR the vocals, not when you expect them.

If the audio contains no vocals or you cannot confidently identify lyrics, respond with: []"""

    def _build_prompt(
        self,
        audio: AudioData,
        lyrics: list[LyricLine],
        ctx: ChunkContext | None = None,
    ) -> str:
        audio_duration = audio.duration

        if ctx is None:
            lyrics_text = "\n".join([f"{i + 1}. {line.text}" for i, line in enumerate(lyrics)])
            return f"""Listen to this {audio_duration:.1f}s audio and identify which lyrics are sung.

LYRICS:
{lyrics_text}

For each lyric you HEAR, provide start/end times in seconds (0.0 to {audio_duration:.1f}).
Output as JSON array:
[{{"line": 1, "text": "...", "start": 0.0, "end": 2.5}}, ...]

IMPORTANT: Only include lyrics you actually HEAR. If uncertain, skip that line.
Only output the JSON, no other text."""

        full_lyrics_text = "\n".join(
            [f"{i + 1}. {line.text}" for i, line in enumerate(ctx.all_lyrics)]
        )

        expected_lyrics = ctx.expected_lyrics
        expected_start = ctx.expected_lyrics_start_idx + 1
        expected_end = ctx.expected_lyrics_end_idx or len(ctx.all_lyrics)
        expected_lyrics_text = "\n".join(
            [
                f"{ctx.expected_lyrics_start_idx + i + 1}. {line.text}"
                for i, line in enumerate(expected_lyrics)
            ]
        )

        anchor_info = ""
        if ctx.has_anchor and ctx.last_matched_line_idx is not None:
            anchor_info = f"""
CONTEXT: In the previous segment, line {ctx.last_matched_line_idx + 1} was the last lyric heard.
"""

        return f"""This is a {audio_duration:.1f}s audio segment. Listen carefully and identify sung lyrics.
{anchor_info}
FULL LYRICS (for reference only):
{full_lyrics_text}

LIKELY LYRICS FOR THIS SEGMENT (lines {expected_start}-{expected_end}):
{expected_lyrics_text}

TASK:
1. LISTEN to this audio segment from 0.0s to {audio_duration:.1f}s
2. Identify which lyrics you ACTUALLY HEAR being sung
3. Provide timestamps RELATIVE to this segment (0.0s = start of this audio)
4. Do NOT include lyrics you don't clearly hear

Output format:
[{{"line": <line_number>, "text": "<lyric>", "start": <seconds>, "end": <seconds>}}, ...]

If you hear no vocals or cannot identify lyrics, output: []
Only output the JSON array, nothing else."""

    def _generate_chunk_ranges(
        self,
        total_duration: float,
    ) -> list[tuple[float, float]]:
        chunk_duration = float(self.chunk_duration)
        overlap = self.chunk_overlap
        step = chunk_duration - overlap

        ranges = []
        chunk_start = 0.0

        while chunk_start < total_duration:
            chunk_end = min(chunk_start + chunk_duration, total_duration)
            ranges.append((chunk_start, chunk_end))

            if chunk_end >= total_duration:
                break

            chunk_start += step

        return ranges

    def _sync_with_chunks(
        self,
        audio: AudioData,
        lyrics: list[LyricLine],
        progress_callback: Callable[[int, int], None] | None = None,
        debug_info: DebugInfo | None = None,
    ) -> list[SyncResult]:
        total_duration = audio.duration
        output_dir = debug_info.output_dir if debug_info else None

        chunk_ranges = self._generate_chunk_ranges(total_duration)
        num_chunks = len(chunk_ranges)

        lyrics_per_chunk = max(1, len(lyrics) // num_chunks)
        window_margin = max(3, lyrics_per_chunk // 2)

        all_results: list[SyncResult] = []

        last_matched_line_idx: int | None = None
        last_matched_line_text: str | None = None
        last_matched_end_time: float | None = None

        for i, (chunk_start, chunk_end) in enumerate(chunk_ranges):
            if progress_callback:
                progress_callback(i + 1, num_chunks)

            chunk_audio = self._extract_chunk(audio, chunk_start, chunk_end)

            if last_matched_line_idx is not None:
                expected_start = max(0, last_matched_line_idx - 1)
            else:
                expected_start = max(0, i * lyrics_per_chunk - window_margin)

            expected_end = min(len(lyrics), (i + 1) * lyrics_per_chunk + window_margin)

            if i == num_chunks - 1:
                expected_end = len(lyrics)

            ctx = ChunkContext(
                chunk_idx=i,
                audio_offset=chunk_start,
                chunk_end=chunk_end,
                total_duration=total_duration,
                all_lyrics=lyrics,
                last_matched_line_idx=last_matched_line_idx,
                last_matched_line_text=last_matched_line_text,
                last_matched_end_time=last_matched_end_time,
                expected_lyrics_start_idx=expected_start,
                expected_lyrics_end_idx=expected_end,
                output_dir=output_dir,
            )

            try:
                import time as time_module

                t0 = time_module.time()
                chunk_results_raw, prompt, response = self._sync_single(
                    chunk_audio, lyrics, ctx=ctx
                )
                processing_time = time_module.time() - t0

                results_before_filter = list(chunk_results_raw)

                chunk_results = self._filter_chunk_results(chunk_results_raw, chunk_audio.duration)

                for result in chunk_results:
                    result.start_time += chunk_start
                    result.end_time += chunk_start

                if chunk_results:
                    last_result = max(chunk_results, key=lambda r: r.line_number or 0)
                    if last_result.line_number is not None:
                        last_matched_line_idx = last_result.line_number - 1
                        last_matched_line_text = last_result.text
                        last_matched_end_time = last_result.end_time

                all_results.extend(chunk_results)
                self._save_chunk_debug(
                    ctx, chunk_audio, prompt, response, results_before_filter, chunk_results
                )

                if debug_info:
                    debug_info.add_chunk(
                        chunk_idx=i,
                        audio_start=chunk_start,
                        audio_end=chunk_end,
                        lyrics_start_idx=expected_start,
                        lyrics_end_idx=expected_end,
                        prompt=prompt,
                        response=response,
                        parsed_results=chunk_results,
                        processing_time=processing_time,
                        anchor_line_idx=ctx.last_matched_line_idx,
                        anchor_line_text=ctx.last_matched_line_text,
                        anchor_end_time=ctx.last_matched_end_time,
                        results_before_filter=results_before_filter,
                        results_after_filter=chunk_results,
                    )

            except Exception as e:
                if debug_info:
                    debug_info.add_error(f"Chunk {i + 1} failed: {e}")
                print(f"Chunk {i + 1} failed: {e}")
                continue

        return self._deduplicate_results(all_results, lyrics)

    def _extract_chunk(self, audio: AudioData, start: float, end: float) -> AudioData:
        start_sample = int(start * audio.sample_rate)
        end_sample = int(end * audio.sample_rate)
        return AudioData(
            waveform=audio.waveform[start_sample:end_sample],
            sample_rate=audio.sample_rate,
            duration=end - start,
        )

    def _filter_chunk_results(
        self,
        results: list[SyncResult],
        chunk_duration: float,
        time_margin: float = 2.0,
    ) -> list[SyncResult]:
        if not results:
            return results

        filtered = []
        for result in results:
            if result.start_time < -time_margin:
                continue
            if result.start_time > chunk_duration + time_margin:
                continue
            if result.end_time < 0:
                continue
            if result.end_time > chunk_duration + time_margin * 2:
                continue

            result.start_time = max(0.0, result.start_time)
            result.end_time = min(chunk_duration + time_margin, result.end_time)

            filtered.append(result)

        return filtered

    def _save_chunk_debug(
        self,
        ctx: ChunkContext,
        audio: AudioData,
        prompt: str,
        response: str,
        results_before_filter: list[SyncResult],
        results_after_filter: list[SyncResult],
    ) -> None:
        if not ctx.output_dir:
            return
        chunks_dir = ctx.output_dir / "chunks"
        chunks_dir.mkdir(exist_ok=True)

        prefix = f"chunk_{ctx.chunk_idx:03d}_{ctx.audio_offset:.1f}s-{ctx.chunk_end:.1f}s"
        audio.to_file(chunks_dir / f"{prefix}.wav")

        (chunks_dir / f"chunk_{ctx.chunk_idx:03d}_prompt.txt").write_text(prompt, encoding="utf-8")
        (chunks_dir / f"chunk_{ctx.chunk_idx:03d}_response.txt").write_text(
            response, encoding="utf-8"
        )

        import json

        def result_to_dict(r: SyncResult) -> dict:
            return {
                "line": r.line_number,
                "text": r.text,
                "start": round(r.start_time, 2),
                "end": round(r.end_time, 2),
            }

        debug_data = {
            "chunk_idx": ctx.chunk_idx,
            "audio_range": {
                "start": ctx.audio_offset,
                "end": ctx.chunk_end,
                "duration": ctx.chunk_end - ctx.audio_offset,
            },
            "total_duration": ctx.total_duration,
            "anchor": {
                "has_anchor": ctx.has_anchor,
                "last_matched_line_idx": ctx.last_matched_line_idx,
                "last_matched_line_text": ctx.last_matched_line_text,
                "last_matched_end_time": ctx.last_matched_end_time,
            }
            if ctx.has_anchor
            else None,
            "expected_lyrics_window": {
                "start_idx": ctx.expected_lyrics_start_idx,
                "end_idx": ctx.expected_lyrics_end_idx,
                "count": len(ctx.expected_lyrics),
                "lines": [
                    f"{ctx.expected_lyrics_start_idx + i + 1}. {l.text}"
                    for i, l in enumerate(ctx.expected_lyrics)
                ],
            },
            "results": {
                "before_filter_count": len(results_before_filter),
                "after_filter_count": len(results_after_filter),
                "filtered_out_count": len(results_before_filter) - len(results_after_filter),
                "before_filter": [result_to_dict(r) for r in results_before_filter],
                "after_filter": [result_to_dict(r) for r in results_after_filter],
            },
        }

        debug_json = json.dumps(debug_data, ensure_ascii=False, indent=2)
        (chunks_dir / f"chunk_{ctx.chunk_idx:03d}_debug.json").write_text(
            debug_json, encoding="utf-8"
        )

    def _deduplicate_results(
        self,
        results: list[SyncResult],
        lyrics: list[LyricLine] | None = None,
    ) -> list[SyncResult]:
        if not results:
            return results

        if lyrics:
            return self._deduplicate_by_line_number(results, lyrics)

        return self._deduplicate_by_time(results)

    def _deduplicate_by_line_number(
        self,
        results: list[SyncResult],
        lyrics: list[LyricLine],
    ) -> list[SyncResult]:
        line_to_results: dict[int, list[SyncResult]] = {}
        unmatched_results: list[SyncResult] = []

        for result in results:
            if result.line_number is not None:
                if result.line_number not in line_to_results:
                    line_to_results[result.line_number] = []
                line_to_results[result.line_number].append(result)
            else:
                unmatched_results.append(result)

        deduplicated = []
        for line_num in sorted(line_to_results.keys()):
            candidates = line_to_results[line_num]
            best = min(candidates, key=lambda r: r.start_time)
            deduplicated.append(best)

        if unmatched_results:
            unmatched_deduped = self._deduplicate_by_time(unmatched_results)
            all_results = deduplicated + unmatched_deduped
            all_results.sort(key=lambda r: r.start_time)
            return all_results

        return deduplicated

    def _deduplicate_by_time(self, results: list[SyncResult]) -> list[SyncResult]:
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
