"""Subtitle output formatters for aligned lyrics.

Reconstructed to match the interface expected by everyric2.cli and
everyric2.output.__init__ (BaseFormatter, SRT/ASS/LRC/JSON, FormatterFactory).
Operates on everyric2.inference.prompt.SyncResult.
"""
from __future__ import annotations

import json

from everyric2.inference.prompt import SyncResult


def _srt_ts(t: float) -> str:
    ms = int(round(max(0.0, t) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _lrc_ts(t: float) -> str:
    t = max(0.0, t)
    m = int(t // 60)
    s = t - m * 60
    return f"[{m:02d}:{s:05.2f}]"


def _ass_ts(t: float) -> str:
    cs = int(round(max(0.0, t) * 100))
    h, cs = divmod(cs, 360_000)
    m, cs = divmod(cs, 6_000)
    s, cs = divmod(cs, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


class BaseFormatter:
    extension: str = "txt"
    include_translation: bool = False
    include_pronunciation: bool = False

    def get_extension(self) -> str:
        return self.extension

    def _display(self, r: SyncResult) -> str:
        return r.get_display_text(
            include_translation=self.include_translation,
            include_pronunciation=self.include_pronunciation,
        )

    def format(self, results: list[SyncResult]) -> str:  # pragma: no cover - abstract
        raise NotImplementedError


class SRTFormatter(BaseFormatter):
    extension = "srt"

    def format(self, results: list[SyncResult]) -> str:
        blocks = []
        for i, r in enumerate(results, start=1):
            blocks.append(
                f"{i}\n{_srt_ts(r.start_time)} --> {_srt_ts(r.end_time)}\n{self._display(r)}\n"
            )
        return "\n".join(blocks)


class LRCFormatter(BaseFormatter):
    extension = "lrc"

    def format(self, results: list[SyncResult]) -> str:
        lines = []
        for r in results:
            text = self._display(r).replace("\n", " ")
            lines.append(f"{_lrc_ts(r.start_time)}{text}")
        return "\n".join(lines) + ("\n" if lines else "")


class JSONFormatter(BaseFormatter):
    extension = "json"

    def format(self, results: list[SyncResult]) -> str:
        return json.dumps(
            [r.to_dict() for r in results], ensure_ascii=False, indent=2
        )


class ASSFormatter(BaseFormatter):
    extension = "ass"

    _HEADER = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "PlayResX: 1920\n"
        "PlayResY: 1080\n"
        "WrapStyle: 0\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
        "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding\n"
        "Style: Default,Arial,64,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,"
        "0,0,0,0,100,100,0,0,1,3,1,2,40,40,60,1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
        "Effect, Text\n"
    )

    def format(self, results: list[SyncResult]) -> str:
        lines = [self._HEADER]
        for r in results:
            text = self._display(r).replace("\n", "\\N")
            lines.append(
                f"Dialogue: 0,{_ass_ts(r.start_time)},{_ass_ts(r.end_time)},"
                f"Default,,0,0,0,,{text}"
            )
        return "\n".join(lines) + "\n"


class FormatterFactory:
    _FORMATTERS: dict[str, type[BaseFormatter]] = {
        "srt": SRTFormatter,
        "ass": ASSFormatter,
        "lrc": LRCFormatter,
        "json": JSONFormatter,
    }

    @classmethod
    def get_supported_formats(cls) -> list[str]:
        return list(cls._FORMATTERS.keys())

    @classmethod
    def get_formatter(cls, fmt: str) -> BaseFormatter:
        key = (fmt or "").lower()
        if key not in cls._FORMATTERS:
            raise ValueError(
                f"Unsupported format: {fmt!r}. Supported: {cls.get_supported_formats()}"
            )
        return cls._FORMATTERS[key]()
