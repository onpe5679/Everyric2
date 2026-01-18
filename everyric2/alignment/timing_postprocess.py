"""Post-processing for subtitle timing based on vocal activity detection.

Adjusts CTC alignment results using actual vocal regions from separated audio.
"""

import logging
from dataclasses import dataclass
from typing import Literal

from everyric2.audio.vad import VADResult, VocalActivityDetector, VocalRegion
from everyric2.config.settings import SegmentationSettings, get_settings
from everyric2.inference.prompt import SyncResult

logger = logging.getLogger(__name__)


@dataclass
class TimingAdjustment:
    line_number: int
    original_start: float
    original_end: float
    adjusted_start: float
    adjusted_end: float
    reason: str


@dataclass
class PostProcessResult:
    results: list[SyncResult]
    adjustments: list[TimingAdjustment]
    vocal_regions: list[VocalRegion]
    stats: dict


class TimingPostProcessor:
    def __init__(
        self,
        settings: SegmentationSettings | None = None,
        reading_buffer: float = 1.0,
        extend_to_vocal: bool = True,
        shrink_to_vocal: bool = True,
        min_display_duration: float = 0.5,
    ):
        self.settings = settings or get_settings().segmentation
        self.reading_buffer = reading_buffer
        self.extend_to_vocal = extend_to_vocal
        self.shrink_to_vocal = shrink_to_vocal
        self.min_display_duration = min_display_duration

    def process(
        self,
        results: list[SyncResult],
        vad_result: VADResult,
        segment_mode: Literal["line", "word", "character"] = "line",
    ) -> PostProcessResult:
        if not results:
            return PostProcessResult(
                results=[], adjustments=[], vocal_regions=vad_result.regions, stats={}
            )

        adjustments: list[TimingAdjustment] = []
        processed: list[SyncResult] = []

        extended_count = 0
        shrunk_count = 0
        buffer_added_count = 0

        for i, result in enumerate(results):
            original_start = result.start_time
            original_end = result.end_time
            new_start = original_start
            new_end = original_end
            reasons = []

            if self.extend_to_vocal:
                vocal_end = self._find_vocal_end_for_line(vad_result, original_start, original_end)
                next_line_start = (
                    results[i + 1].start_time if i + 1 < len(results) else float("inf")
                )
                max_extend = min(next_line_start - 0.05, original_end + 3.0)
                if vocal_end and vocal_end > original_end:
                    new_end = min(vocal_end, max_extend)
                    if new_end > original_end:
                        reasons.append(f"extend_to_vocal:{original_end:.2f}→{new_end:.2f}")
                        extended_count += 1

            if self.shrink_to_vocal:
                vocal_start = self._find_vocal_start_for_line(
                    vad_result, original_start, original_end
                )
                if vocal_start and vocal_start > original_start:
                    if vocal_start < new_end - self.min_display_duration:
                        new_start = vocal_start
                        reasons.append(f"shrink_start:{original_start:.2f}→{vocal_start:.2f}")
                        shrunk_count += 1

            if segment_mode == "line" and self.reading_buffer > 0:
                next_start = results[i + 1].start_time if i + 1 < len(results) else float("inf")
                max_end = next_start - 0.05

                if not self._is_vocal_active_in_range(
                    vad_result, new_end, new_end + self.reading_buffer
                ):
                    buffered_end = min(new_end + self.reading_buffer, max_end)
                    if buffered_end > new_end:
                        reasons.append(f"reading_buffer:{new_end:.2f}→{buffered_end:.2f}")
                        new_end = buffered_end
                        buffer_added_count += 1

            if new_end - new_start < self.min_display_duration:
                new_end = new_start + self.min_display_duration
                reasons.append("min_duration_enforced")

            if new_start != original_start or new_end != original_end:
                adjustments.append(
                    TimingAdjustment(
                        line_number=result.line_number or i,
                        original_start=original_start,
                        original_end=original_end,
                        adjusted_start=new_start,
                        adjusted_end=new_end,
                        reason=", ".join(reasons),
                    )
                )

            new_result = SyncResult(
                text=result.text,
                start_time=new_start,
                end_time=new_end,
                confidence=result.confidence,
                line_number=result.line_number,
                word_segments=result.word_segments,
                translation=result.translation,
                pronunciation=result.pronunciation,
            )
            processed.append(new_result)

        processed = self._resolve_overlaps(processed)

        processed = self._merge_short_gaps(processed)

        stats = {
            "total_lines": len(results),
            "extended_to_vocal": extended_count,
            "shrunk_to_vocal": shrunk_count,
            "reading_buffer_added": buffer_added_count,
            "total_adjustments": len(adjustments),
        }

        logger.info(
            f"Timing post-process: {extended_count} extended, {shrunk_count} shrunk, "
            f"{buffer_added_count} buffered"
        )

        return PostProcessResult(
            results=processed,
            adjustments=adjustments,
            vocal_regions=vad_result.regions,
            stats=stats,
        )

    def _find_vocal_end_for_line(
        self, vad_result: VADResult, start: float, end: float
    ) -> float | None:
        for region in vad_result.regions:
            if region.start <= end <= region.end:
                return region.end
            if region.start <= start <= region.end and region.end > end:
                return region.end
        return None

    def _find_vocal_start_for_line(
        self, vad_result: VADResult, start: float, end: float
    ) -> float | None:
        for region in vad_result.regions:
            if region.start <= start <= region.end:
                return None
            if region.start > start and region.start < end:
                return region.start
        return None

    def _is_vocal_active_in_range(self, vad_result: VADResult, start: float, end: float) -> bool:
        for region in vad_result.regions:
            if region.end > start and region.start < end:
                return True
        return False

    def _resolve_overlaps(self, results: list[SyncResult]) -> list[SyncResult]:
        if len(results) < 2:
            return results

        resolved = [results[0]]
        for i in range(1, len(results)):
            prev = resolved[-1]
            curr = results[i]

            if curr.start_time < prev.end_time:
                midpoint = (prev.end_time + curr.start_time) / 2
                resolved[-1] = SyncResult(
                    text=prev.text,
                    start_time=prev.start_time,
                    end_time=midpoint,
                    confidence=prev.confidence,
                    line_number=prev.line_number,
                    word_segments=prev.word_segments,
                    translation=prev.translation,
                    pronunciation=prev.pronunciation,
                )
                curr = SyncResult(
                    text=curr.text,
                    start_time=midpoint,
                    end_time=curr.end_time,
                    confidence=curr.confidence,
                    line_number=curr.line_number,
                    word_segments=curr.word_segments,
                    translation=curr.translation,
                    pronunciation=curr.pronunciation,
                )

            resolved.append(curr)

        return resolved

    def _merge_short_gaps(self, results: list[SyncResult]) -> list[SyncResult]:
        if len(results) < 2:
            return results

        merged = [results[0]]
        for i in range(1, len(results)):
            prev = merged[-1]
            curr = results[i]
            gap = curr.start_time - prev.end_time

            if 0 < gap < self.settings.min_silence_gap:
                midpoint = (prev.end_time + curr.start_time) / 2
                merged[-1] = SyncResult(
                    text=prev.text,
                    start_time=prev.start_time,
                    end_time=midpoint,
                    confidence=prev.confidence,
                    line_number=prev.line_number,
                    word_segments=prev.word_segments,
                    translation=prev.translation,
                    pronunciation=prev.pronunciation,
                )
                curr = SyncResult(
                    text=curr.text,
                    start_time=midpoint,
                    end_time=curr.end_time,
                    confidence=curr.confidence,
                    line_number=curr.line_number,
                    word_segments=curr.word_segments,
                    translation=curr.translation,
                    pronunciation=curr.pronunciation,
                )

            merged.append(curr)

        return merged

    def process_segments(
        self,
        results: list[SyncResult],
        vad_result: VADResult,
    ) -> list[SyncResult]:
        """Process word/character segments to fix timing issues.

        For each segment, if there's a long silence gap inside (CTC misalignment),
        shrink the segment to the actual vocal region.
        """
        if not results or not vad_result.regions:
            return results

        processed = []
        for result in results:
            start = result.start_time
            end = result.end_time
            duration = end - start

            if duration <= 1.0:
                processed.append(result)
                continue

            vocal_in_range = self._find_vocal_regions_in_range(vad_result, start, end)

            if not vocal_in_range:
                processed.append(result)
                continue

            total_vocal_duration = sum(
                min(v.end, end) - max(v.start, start) for v in vocal_in_range
            )
            silence_ratio = 1 - (total_vocal_duration / duration)

            if silence_ratio > 0.5 and duration > 2.0:
                last_vocal = vocal_in_range[-1]
                new_start = max(start, last_vocal.start - 0.1)
                new_end = min(end, last_vocal.end + 0.1)

                if new_end - new_start < 0.3:
                    new_start = last_vocal.start
                    new_end = last_vocal.end + 0.2

                processed.append(
                    SyncResult(
                        text=result.text,
                        start_time=new_start,
                        end_time=new_end,
                        confidence=result.confidence,
                        line_number=result.line_number,
                        word_segments=result.word_segments,
                        translation=result.translation,
                        pronunciation=result.pronunciation,
                    )
                )
            else:
                processed.append(result)

        return self._fix_segment_overlaps(processed)

    def _find_vocal_regions_in_range(
        self, vad_result: VADResult, start: float, end: float
    ) -> list[VocalRegion]:
        regions = []
        for region in vad_result.regions:
            if region.end > start and region.start < end:
                regions.append(region)
        return regions

    def _fix_segment_overlaps(self, results: list[SyncResult]) -> list[SyncResult]:
        if len(results) < 2:
            return results

        fixed = [results[0]]
        for i in range(1, len(results)):
            prev = fixed[-1]
            curr = results[i]

            if curr.start_time < prev.end_time:
                gap = 0.02
                fixed[-1] = SyncResult(
                    text=prev.text,
                    start_time=prev.start_time,
                    end_time=curr.start_time - gap,
                    confidence=prev.confidence,
                    line_number=prev.line_number,
                    word_segments=prev.word_segments,
                    translation=prev.translation,
                    pronunciation=prev.pronunciation,
                )

            fixed.append(curr)

        return fixed


def detect_timing_issues(results: list[SyncResult], vad_result: VADResult) -> list[dict]:
    issues = []

    for i, result in enumerate(results):
        start, end = result.start_time, result.end_time

        has_vocal = False
        for region in vad_result.regions:
            if region.end > start and region.start < end:
                has_vocal = True
                break

        if not has_vocal:
            issues.append(
                {
                    "type": "subtitle_without_vocal",
                    "line": i,
                    "start": start,
                    "end": end,
                    "text": result.text[:30],
                }
            )

    covered_ranges = [(r.start_time, r.end_time) for r in results]

    for region in vad_result.regions:
        is_covered = False
        for start, end in covered_ranges:
            if end > region.start and start < region.end:
                is_covered = True
                break

        if not is_covered and region.end - region.start > 0.5:
            issues.append(
                {
                    "type": "vocal_without_subtitle",
                    "start": region.start,
                    "end": region.end,
                    "duration": region.end - region.start,
                }
            )

    return issues
