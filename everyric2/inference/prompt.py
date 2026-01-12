"""Prompt templates and response parsing for Qwen-Omni."""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LyricLine:
    text: str
    line_number: int
    translation: str | None = None

    @classmethod
    def from_text(cls, text: str) -> list["LyricLine"]:
        """Parse multiline text into LyricLine list.

        Args:
            text: Multiline lyrics text.

        Returns:
            List of LyricLine instances.
        """
        lines = []
        for i, line in enumerate(text.strip().split("\n"), start=1):
            line = line.strip()
            if line:  # Skip empty lines
                lines.append(cls(text=line, line_number=i))
        return lines

    @classmethod
    def from_file(cls, path: Path | str) -> list["LyricLine"]:
        """Load lyrics from file.

        Args:
            path: Path to lyrics text file.

        Returns:
            List of LyricLine instances.
        """
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        return cls.from_text(text)


@dataclass
class SyncResult:
    """Synchronization result for a single lyric line."""

    text: str
    start_time: float  # seconds
    end_time: float  # seconds
    confidence: float | None = None
    line_number: int | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "start": self.start_time,
            "end": self.end_time,
            "confidence": self.confidence,
            "line_number": self.line_number,
        }


@dataclass
class PromptBuilder:
    """Build prompts and parse responses for Qwen-Omni."""

    # System prompt for lyrics synchronization
    system_prompt: str = field(
        default="""You are a professional lyrics synchronization assistant.
Your task is to listen to the audio and align the provided lyrics with precise timestamps.

CRITICAL RULES:
1. Listen to the audio CAREFULLY and identify EXACTLY when each lyric line is sung
2. Output format: JSON array with objects containing "text", "start", "end"
3. Times are in SECONDS with 2 decimal places (e.g., 12.34)
4. Preserve the original lyric text EXACTLY as provided
5. Timestamps must be monotonically increasing (no overlaps)
6. If a line has instrumental/silence before it, start timestamp should be when vocals BEGIN
7. End timestamp should be when the vocals for that line END

OUTPUT ONLY THE JSON ARRAY. No explanations, no markdown code blocks."""
    )

    # User prompt template
    user_prompt_template: str = field(
        default="""Listen to the audio and synchronize these lyrics:

{lyrics}

Output a JSON array like this:
[{{"text": "first line", "start": 0.00, "end": 2.50}}, {{"text": "second line", "start": 2.80, "end": 5.20}}]"""
    )

    def build_lyrics_text(self, lyrics: list[LyricLine]) -> str:
        """Format lyrics for prompt.

        Args:
            lyrics: List of lyric lines.

        Returns:
            Formatted lyrics string.
        """
        return "\n".join(f"{l.line_number}. {l.text}" for l in lyrics)

    def build_conversation(
        self,
        audio_path: Path | str,
        lyrics: list[LyricLine],
        custom_system_prompt: str | None = None,
    ) -> list[dict]:
        """Build conversation for Qwen-Omni.

        Args:
            audio_path: Path to audio file.
            lyrics: List of lyric lines.
            custom_system_prompt: Optional custom system prompt.

        Returns:
            Conversation list in Qwen-Omni format.
        """
        lyrics_text = self.build_lyrics_text(lyrics)
        user_prompt = self.user_prompt_template.format(lyrics=lyrics_text)

        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": custom_system_prompt or self.system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": str(audio_path)},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

    def parse_response(
        self,
        response: str,
        lyrics: list[LyricLine] | None = None,
    ) -> list[SyncResult]:
        """Parse model response into SyncResult list.

        Args:
            response: Raw model response.
            lyrics: Original lyrics for line number matching.

        Returns:
            List of SyncResult instances.

        Raises:
            ValueError: If response cannot be parsed.
        """
        # Try to extract JSON from response
        response = response.strip()

        # Remove markdown code blocks if present
        if response.startswith("```"):
            # Extract content between code blocks
            match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response, re.DOTALL)
            if match:
                response = match.group(1).strip()

        # Try JSON parsing first
        try:
            data = json.loads(response)
            if isinstance(data, list):
                return self._parse_json_list(data, lyrics)
        except json.JSONDecodeError:
            pass

        # Fallback: try to find JSON array in response
        array_match = re.search(r"\[.*\]", response, re.DOTALL)
        if array_match:
            try:
                data = json.loads(array_match.group())
                if isinstance(data, list):
                    return self._parse_json_list(data, lyrics)
            except json.JSONDecodeError:
                pass

        # Fallback: try to parse line by line with timestamps
        return self._parse_fallback(response, lyrics)

    def _parse_json_list(
        self,
        data: list,
        lyrics: list[LyricLine] | None = None,
    ) -> list[SyncResult]:
        results = []
        lyrics_text_map = {l.text.lower(): l.line_number for l in lyrics} if lyrics else {}
        lyrics_by_number = {l.line_number: l for l in lyrics} if lyrics else {}

        for item in data:
            if not isinstance(item, dict):
                continue

            text = item.get("text", "")
            start = item.get("start", item.get("start_time", 0))
            end = item.get("end", item.get("end_time", 0))
            confidence = item.get("confidence")

            line_num = item.get("line")

            if line_num is not None and lyrics_by_number:
                original_lyric = lyrics_by_number.get(line_num)
                if original_lyric:
                    text = original_lyric.text

            if line_num is None:
                line_num = lyrics_text_map.get(text.lower())

            results.append(
                SyncResult(
                    text=text,
                    start_time=float(start),
                    end_time=float(end),
                    confidence=float(confidence) if confidence else None,
                    line_number=line_num,
                )
            )

        return results

    def _parse_fallback(
        self,
        response: str,
        lyrics: list[LyricLine] | None = None,
    ) -> list[SyncResult]:
        """Fallback parser for non-JSON responses.

        Tries to parse formats like:
        - [00:01.23] lyrics text
        - 1.23 - 4.56: lyrics text
        """
        results = []

        # LRC-style: [MM:SS.xx] text
        lrc_pattern = r"\[(\d+):(\d+\.?\d*)\]\s*(.+)"

        # Time range style: START - END: text
        range_pattern = r"(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)[:\s]+(.+)"

        for line in response.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            # Try LRC format
            lrc_match = re.match(lrc_pattern, line)
            if lrc_match:
                minutes = int(lrc_match.group(1))
                seconds = float(lrc_match.group(2))
                text = lrc_match.group(3).strip()
                start = minutes * 60 + seconds
                # Estimate end time (next line start or +3 seconds)
                results.append(
                    SyncResult(
                        text=text,
                        start_time=start,
                        end_time=start + 3.0,  # Will be adjusted later
                    )
                )
                continue

            # Try range format
            range_match = re.match(range_pattern, line)
            if range_match:
                start = float(range_match.group(1))
                end = float(range_match.group(2))
                text = range_match.group(3).strip()
                results.append(
                    SyncResult(
                        text=text,
                        start_time=start,
                        end_time=end,
                    )
                )

        # Adjust end times for LRC-style results
        for i in range(len(results) - 1):
            if results[i].end_time == results[i].start_time + 3.0:
                # Set end to next line's start (with small gap)
                results[i].end_time = results[i + 1].start_time - 0.1

        if not results:
            raise ValueError(f"Could not parse response: {response[:200]}...")

        return results
