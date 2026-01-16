from dataclasses import dataclass

from everyric2.config.settings import SegmentationSettings, get_settings
from everyric2.inference.prompt import SyncResult


@dataclass
class InterludeInfo:
    start: float
    end: float
    duration: float
    after_line: int


class SilenceHandler:
    def __init__(self, settings: SegmentationSettings | None = None):
        self.settings = settings or get_settings().segmentation
        self.detected_interludes: list[InterludeInfo] = []

    def process(self, results: list[SyncResult]) -> list[SyncResult]:
        if not results or len(results) < 2:
            return results

        self.detected_interludes = []
        processed = [results[0]]

        for i in range(1, len(results)):
            prev = processed[-1]
            curr = results[i]
            gap = curr.start_time - prev.end_time

            if gap >= self.settings.interlude_gap:
                self.detected_interludes.append(
                    InterludeInfo(
                        start=prev.end_time,
                        end=curr.start_time,
                        duration=gap,
                        after_line=prev.line_number or i - 1,
                    )
                )
                processed.append(curr)

            elif gap < self.settings.min_silence_gap and gap >= 0:
                merged_prev, merged_curr = self._merge_gap(prev, curr)
                processed[-1] = merged_prev
                processed.append(merged_curr)
            else:
                processed.append(curr)

        return processed

    def _merge_gap(self, prev: SyncResult, curr: SyncResult) -> tuple[SyncResult, SyncResult]:
        mode = self.settings.silence_merge_mode

        if mode == "midpoint":
            midpoint = (prev.end_time + curr.start_time) / 2
            new_prev = SyncResult(
                text=prev.text,
                start_time=prev.start_time,
                end_time=midpoint,
                confidence=prev.confidence,
                line_number=prev.line_number,
                word_segments=prev.word_segments,
                translation=prev.translation,
                pronunciation=prev.pronunciation,
            )
            new_curr = SyncResult(
                text=curr.text,
                start_time=midpoint,
                end_time=curr.end_time,
                confidence=curr.confidence,
                line_number=curr.line_number,
                word_segments=curr.word_segments,
                translation=curr.translation,
                pronunciation=curr.pronunciation,
            )
        elif mode == "extend_prev":
            new_prev = SyncResult(
                text=prev.text,
                start_time=prev.start_time,
                end_time=curr.start_time,
                confidence=prev.confidence,
                line_number=prev.line_number,
                word_segments=prev.word_segments,
                translation=prev.translation,
                pronunciation=prev.pronunciation,
            )
            new_curr = curr
        elif mode == "extend_next":
            new_prev = prev
            new_curr = SyncResult(
                text=curr.text,
                start_time=prev.end_time,
                end_time=curr.end_time,
                confidence=curr.confidence,
                line_number=curr.line_number,
                word_segments=curr.word_segments,
                translation=curr.translation,
                pronunciation=curr.pronunciation,
            )
        else:
            new_prev = prev
            new_curr = curr

        return new_prev, new_curr

    def detect_silence_gaps(self, results: list[SyncResult]) -> list[dict]:
        gaps = []

        for i in range(1, len(results)):
            prev = results[i - 1]
            curr = results[i]
            gap_duration = curr.start_time - prev.end_time

            if gap_duration > 0:
                is_interlude = gap_duration >= self.settings.interlude_gap
                gaps.append(
                    {
                        "index": i,
                        "start": prev.end_time,
                        "end": curr.start_time,
                        "duration": gap_duration,
                        "is_short": gap_duration < self.settings.min_silence_gap,
                        "is_interlude": is_interlude,
                    }
                )

        return gaps

    def get_interludes(self) -> list[InterludeInfo]:
        return self.detected_interludes
