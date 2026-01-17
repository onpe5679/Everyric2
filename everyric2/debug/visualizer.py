from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import numpy as np

from everyric2.debug.debug_info import DebugInfo
from everyric2.inference.prompt import SyncResult


def _configure_cjk_fonts() -> None:
    cjk_fonts = [
        "Noto Sans CJK JP",
        "Noto Sans CJK KR",
        "Noto Sans CJK SC",
        "Noto Sans CJK TC",
        "Noto Sans CJK HK",
        "Droid Sans Fallback",
        "DejaVu Sans",
    ]

    available_fonts = {f.name for f in fm.fontManager.ttflist}
    font_family = [font for font in cjk_fonts if font in available_fonts]

    if font_family:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = font_family + plt.rcParams["font.sans-serif"]

    plt.rcParams["figure.max_open_warning"] = 0


_configure_cjk_fonts()


class DiagnosticsVisualizer:
    COLORS = {
        "original": "#FF6B6B",
        "vocals": "#4ECDC4",
        "final": "#45B7D1",
        "transcription": "#DDA0DD",
        "pronunciation": "#FFD93D",
        "silence_gap": "#FF0000",
        "word_segment": "#98D8C8",
    }

    @staticmethod
    def _confidence_to_color(
        confidence: float | None, min_conf: float = -5.0, max_conf: float = 10.0
    ) -> tuple[str, float]:
        """Convert log-probability confidence to color and alpha.

        Returns (color_hex, alpha) where:
        - Red = low confidence
        - Yellow = medium confidence
        - Green = high confidence
        """
        if confidence is None:
            return "#888888", 0.5

        normalized = (confidence - min_conf) / (max_conf - min_conf)
        normalized = max(0.0, min(1.0, normalized))

        if normalized < 0.5:
            r = 255
            g = int(255 * (normalized * 2))
            b = 0
        else:
            r = int(255 * (1 - (normalized - 0.5) * 2))
            g = 255
            b = 0

        color = f"#{r:02x}{g:02x}{b:02x}"
        alpha = 0.4 + 0.5 * normalized
        return color, alpha

    def __init__(self, figsize: tuple[int, int] = (16, 20)):
        self.figsize = figsize

    def create_diagnostics(
        self,
        debug_info: DebugInfo,
        results: list[SyncResult],
        output_path: Path,
        audio_waveform: np.ndarray | None = None,
        vocals_waveform: np.ndarray | None = None,
        translated_results: list[SyncResult] | None = None,
        sample_rate: int = 16000,
        silence_gaps: list[dict] | None = None,
        segment_mode: str = "line",
    ) -> Path:
        transcription_sets = debug_info.transcription_sets or []
        if not transcription_sets and debug_info.transcription_words:
            transcription_sets = [
                {
                    "engine": debug_info.transcription_engine or "Transcription",
                    "words": debug_info.transcription_words,
                    "match_stats": debug_info.match_stats,
                }
            ]

        num_cols = 3
        if vocals_waveform is not None:
            num_cols += 1
        if translated_results:
            num_cols += 1
        num_cols += len(transcription_sets)

        has_word_segments = any(r.word_segments for r in results)
        if has_word_segments and segment_mode != "line":
            num_cols += 1

        has_pronunciation = any(r.pronunciation for r in results)
        if has_pronunciation:
            num_cols += 1

        fig_width = 4 * num_cols
        fig, axes = plt.subplots(1, num_cols, figsize=(fig_width, 20), sharey=True)
        if num_cols == 1:
            axes = [axes]
        fig.suptitle("Everyric2 Diagnostics", fontsize=16, fontweight="bold")

        duration = debug_info.audio_duration or (len(results) * 5 if results else 60)
        col_idx = 0

        self._draw_waveform(
            axes[col_idx],
            audio_waveform,
            duration,
            "Audio (Original)",
            self.COLORS["original"],
            silence_gaps=silence_gaps,
        )
        col_idx += 1

        if vocals_waveform is not None:
            self._draw_waveform(
                axes[col_idx], vocals_waveform, duration, "Audio (Vocals)", self.COLORS["vocals"]
            )
            col_idx += 1

        self._draw_lyrics_column(
            axes[col_idx],
            debug_info.original_lyrics,
            duration,
            "Original Lyrics",
            self.COLORS["original"],
        )
        col_idx += 1

        for tset in transcription_sets:
            self._draw_transcription_column(
                axes[col_idx],
                tset.get("words", []),
                tset.get("match_stats", {}),
                duration,
                tset.get("engine", "Transcription"),
            )
            col_idx += 1

        if has_word_segments and segment_mode != "line":
            self._draw_word_segments_column(
                axes[col_idx], results, duration, f"Word Segments ({segment_mode})"
            )
            col_idx += 1

        if has_pronunciation:
            self._draw_pronunciation_column(axes[col_idx], results, duration)
            col_idx += 1

        if translated_results:
            self._draw_synced_column(
                axes[col_idx], translated_results, duration, "Translated", self.COLORS["vocals"]
            )
            col_idx += 1

        self._draw_synced_column(
            axes[col_idx], results, duration, "Synced Output", self.COLORS["final"]
        )

        axes[0].set_ylabel("Time (seconds)")
        tick_interval = 10 if duration <= 120 else 30 if duration <= 300 else 60
        y_ticks = np.arange(0, duration + 1, tick_interval)
        axes[0].set_yticks(y_ticks)
        axes[0].set_yticklabels([f"{int(t // 60):02d}:{int(t % 60):02d}" for t in y_ticks])

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        return output_path

    def _draw_waveform(
        self,
        ax,
        waveform: np.ndarray | None,
        duration: float,
        title: str,
        color: str,
        silence_gaps: list[dict] | None = None,
    ) -> None:
        ax.set_title(title, fontsize=12)
        if waveform is not None:
            times = np.linspace(0, duration, len(waveform))
            step = max(1, len(waveform) // 10000)
            ax.fill_betweenx(times[::step], 0, np.abs(waveform[::step]), alpha=0.7, color=color)

        if silence_gaps:
            for gap in silence_gaps:
                if gap.get("is_short", False):
                    ax.axhline(
                        y=gap["start"],
                        color=self.COLORS["silence_gap"],
                        linestyle="--",
                        linewidth=1,
                        alpha=0.8,
                    )
                    ax.axhline(
                        y=gap["end"],
                        color=self.COLORS["silence_gap"],
                        linestyle="--",
                        linewidth=1,
                        alpha=0.8,
                    )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, duration)
        ax.invert_yaxis()
        ax.set_xlabel("Amplitude")

    def _draw_lyrics_column(
        self, ax, lyrics_text: str | None, duration: float, title: str, color: str
    ) -> None:
        ax.set_title(title, fontsize=12)
        lines = lyrics_text.strip().split("\n") if lyrics_text else []
        if lines:
            line_duration = duration / len(lines)
            for i, line in enumerate(lines):
                if not line.strip():
                    continue
                y_start = i * line_duration
                rect = mpatches.FancyBboxPatch(
                    (0.05, y_start),
                    0.9,
                    line_duration * 0.9,
                    boxstyle="round,pad=0.02",
                    facecolor=color,
                    alpha=0.6,
                    edgecolor="none",
                )
                ax.add_patch(rect)
                display = line[:20] + "..." if len(line) > 20 else line
                ax.text(
                    0.5,
                    y_start + line_duration * 0.45,
                    display,
                    ha="center",
                    va="center",
                    fontsize=7,
                )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, duration)
        ax.invert_yaxis()
        ax.set_xticks([])

    def _draw_transcription_column(
        self,
        ax,
        words: list[dict],
        match_stats: dict,
        duration: float,
        engine_name: str,
    ) -> None:
        all_confidences = [w.get("confidence") for w in words if w.get("confidence") is not None]
        min_conf = min(all_confidences) if all_confidences else -5.0
        max_conf = max(all_confidences) if all_confidences else 10.0
        avg_conf = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0

        title = f"{engine_name}"
        if match_stats:
            rate = match_stats.get("match_rate", 0) * 100
            title += f"\n(match: {rate:.0f}%, avg: {avg_conf:.1f})"
        else:
            title += f"\n(avg conf: {avg_conf:.1f})"

        ax.set_title(title, fontsize=12)

        for word_data in words:
            start = word_data.get("start", 0)
            end = word_data.get("end", start + 0.1)
            word = word_data.get("word", "")
            confidence = word_data.get("confidence")

            height = max(0.1, end - start)
            color, alpha = self._confidence_to_color(confidence, min_conf, max_conf)
            rect = mpatches.FancyBboxPatch(
                (0.05, start),
                0.9,
                height,
                boxstyle="round,pad=0.01",
                facecolor=color,
                alpha=alpha,
                edgecolor="none",
            )
            ax.add_patch(rect)

            if height > 0.3:
                display = word[:8] if len(word) > 8 else word
                ax.text(
                    0.5,
                    start + height / 2,
                    display,
                    ha="center",
                    va="center",
                    fontsize=5,
                    rotation=0,
                )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, duration)
        ax.invert_yaxis()
        ax.set_xticks([])

    def _draw_synced_column(
        self, ax, results: list[SyncResult], duration: float, title: str, color: str
    ) -> None:
        ax.set_title(title, fontsize=12)
        for result in results:
            height = max(0.5, result.end_time - result.start_time)
            rect = mpatches.FancyBboxPatch(
                (0.05, result.start_time),
                0.9,
                height,
                boxstyle="round,pad=0.02",
                facecolor=color,
                alpha=0.7,
                edgecolor="none",
            )
            ax.add_patch(rect)
            display = result.text[:15] + "..." if len(result.text) > 15 else result.text
            ax.text(
                0.5, result.start_time + height / 2, display, ha="center", va="center", fontsize=7
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, duration)
        ax.invert_yaxis()
        ax.set_xticks([])

    def _draw_word_segments_column(
        self, ax, results: list[SyncResult], duration: float, title: str
    ) -> None:
        all_confidences = [
            seg.confidence
            for r in results
            if r.word_segments
            for seg in r.word_segments
            if seg.confidence is not None
        ]
        min_conf = min(all_confidences) if all_confidences else -5.0
        max_conf = max(all_confidences) if all_confidences else 10.0
        avg_conf = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0

        ax.set_title(f"{title}\n(avg conf: {avg_conf:.1f})", fontsize=12)
        for result in results:
            if not result.word_segments:
                continue
            for seg in result.word_segments:
                height = max(0.1, seg.end - seg.start)
                color, alpha = self._confidence_to_color(seg.confidence, min_conf, max_conf)
                rect = mpatches.FancyBboxPatch(
                    (0.05, seg.start),
                    0.9,
                    height,
                    boxstyle="round,pad=0.01",
                    facecolor=color,
                    alpha=alpha,
                    edgecolor="none",
                )
                ax.add_patch(rect)
                if height > 0.2:
                    display = seg.word[:6] if len(seg.word) > 6 else seg.word
                    ax.text(
                        0.5, seg.start + height / 2, display, ha="center", va="center", fontsize=5
                    )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, duration)
        ax.invert_yaxis()
        ax.set_xticks([])

    def _draw_pronunciation_column(self, ax, results: list[SyncResult], duration: float) -> None:
        ax.set_title("Pronunciation", fontsize=12)
        for result in results:
            if not result.pronunciation:
                continue
            height = max(0.5, result.end_time - result.start_time)
            rect = mpatches.FancyBboxPatch(
                (0.05, result.start_time),
                0.9,
                height,
                boxstyle="round,pad=0.02",
                facecolor=self.COLORS["pronunciation"],
                alpha=0.7,
                edgecolor="none",
            )
            ax.add_patch(rect)
            display = (
                result.pronunciation[:12] + "..."
                if len(result.pronunciation) > 12
                else result.pronunciation
            )
            ax.text(
                0.5, result.start_time + height / 2, display, ha="center", va="center", fontsize=6
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, duration)
        ax.invert_yaxis()
        ax.set_xticks([])

    def create_simple_timeline(
        self,
        results: list[SyncResult],
        duration: float,
        output_path: Path,
    ) -> Path:
        fig, ax = plt.subplots(figsize=(14, 8))
        fig.suptitle("Lyrics Timeline", fontsize=14, fontweight="bold")

        for i, result in enumerate(results):
            y = len(results) - i - 1
            width = result.end_time - result.start_time
            rect = mpatches.Rectangle(
                (result.start_time, y - 0.4),
                width,
                0.8,
                facecolor=self.COLORS["final"],
                alpha=0.7,
                edgecolor="black",
                linewidth=0.5,
            )
            ax.add_patch(rect)
            ax.text(
                result.start_time + width / 2,
                y,
                result.text[:30],
                ha="center",
                va="center",
                fontsize=8,
            )

        ax.set_xlim(0, duration)
        ax.set_ylim(-0.5, len(results) - 0.5)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Lyric Lines")

        tick_interval = 10 if duration <= 120 else 30
        x_ticks = np.arange(0, duration + 1, tick_interval)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f"{int(t // 60):02d}:{int(t % 60):02d}" for t in x_ticks])

        ax.set_yticks(range(len(results)))
        ax.set_yticklabels([f"L{len(results) - i}" for i in range(len(results))])

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        return output_path
