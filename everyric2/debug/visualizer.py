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
        "chunks": "#96CEB4",
        "llm_response": "#FFEAA7",
        "transcription": "#DDA0DD",
    }

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

        num_cols = 4
        if vocals_waveform is not None:
            num_cols += 1
        if translated_results:
            num_cols += 1
        num_cols += len(transcription_sets)

        fig_width = 4 * num_cols
        fig, axes = plt.subplots(1, num_cols, figsize=(fig_width, 20), sharey=True)
        fig.suptitle("Everyric2 Diagnostics", fontsize=16, fontweight="bold")

        duration = debug_info.audio_duration or (len(results) * 5 if results else 60)
        col_idx = 0

        self._draw_waveform(
            axes[col_idx], audio_waveform, duration, "Audio (Original)", self.COLORS["original"]
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

        if translated_results:
            self._draw_synced_column(
                axes[col_idx], translated_results, duration, "Translated", self.COLORS["vocals"]
            )
            col_idx += 1

        self._draw_chunks_column(axes[col_idx], debug_info.chunks, duration)
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
        self, ax, waveform: np.ndarray | None, duration: float, title: str, color: str
    ) -> None:
        ax.set_title(title, fontsize=12)
        if waveform is not None:
            times = np.linspace(0, duration, len(waveform))
            step = max(1, len(waveform) // 10000)
            ax.fill_betweenx(times[::step], 0, np.abs(waveform[::step]), alpha=0.7, color=color)
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
        title = f"{engine_name} Transcription"
        if match_stats:
            rate = match_stats.get("match_rate", 0) * 100
            title += f"\n(match: {rate:.0f}%)"

        ax.set_title(title, fontsize=12)

        for word_data in words:
            start = word_data.get("start", 0)
            end = word_data.get("end", start + 0.1)
            word = word_data.get("word", "")
            confidence = word_data.get("confidence")

            height = max(0.1, end - start)

            alpha = 0.5 + 0.4 * (confidence if confidence else 0.5)
            rect = mpatches.FancyBboxPatch(
                (0.05, start),
                0.9,
                height,
                boxstyle="round,pad=0.01",
                facecolor=self.COLORS["transcription"],
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

    def _draw_chunks_column(self, ax, chunks: list, duration: float) -> None:
        ax.set_title("Chunk Processing", fontsize=12)
        for chunk in chunks:
            rect = mpatches.FancyBboxPatch(
                (0.05, chunk.audio_start),
                0.9,
                chunk.audio_end - chunk.audio_start,
                boxstyle="round,pad=0.02",
                facecolor=self.COLORS["chunks"],
                alpha=0.6,
                edgecolor="black",
                linewidth=0.5,
            )
            ax.add_patch(rect)
            ax.text(
                0.5,
                (chunk.audio_start + chunk.audio_end) / 2,
                f"Chunk {chunk.chunk_idx + 1}\n{chunk.audio_start:.1f}s-{chunk.audio_end:.1f}s",
                ha="center",
                va="center",
                fontsize=8,
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
