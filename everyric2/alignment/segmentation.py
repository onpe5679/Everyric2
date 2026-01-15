from everyric2.config.settings import SegmentationSettings, get_settings
from everyric2.inference.prompt import SyncResult, WordSegment


class SegmentationProcessor:
    def __init__(self, settings: SegmentationSettings | None = None):
        self.settings = settings or get_settings().segmentation

    def process(self, results: list[SyncResult]) -> list[SyncResult]:
        if self.settings.mode == "line":
            return self._process_line_mode(results)
        elif self.settings.mode == "word":
            return self._process_word_mode(results)
        elif self.settings.mode == "character":
            return self._process_character_mode(results)
        else:
            return results

    def _process_line_mode(self, results: list[SyncResult]) -> list[SyncResult]:
        processed = []
        for result in results:
            if len(result.text) > self.settings.max_chars_per_segment:
                split_results = self._split_long_line(result)
                processed.extend(split_results)
            else:
                processed.append(result)
        return processed

    def _split_long_line(self, result: SyncResult) -> list[SyncResult]:
        text = result.text
        max_chars = self.settings.max_chars_per_segment

        if result.word_segments and len(result.word_segments) > 1:
            return self._split_by_word_segments(result, max_chars)

        chunks = []
        words = text.split()
        current_chunk = []
        current_len = 0

        for word in words:
            word_len = len(word) + (1 if current_chunk else 0)
            if current_len + word_len > max_chars and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_len = len(word)
            else:
                current_chunk.append(word)
                current_len += word_len

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        if len(chunks) <= 1:
            return [result]

        duration = result.end_time - result.start_time
        chunk_duration = duration / len(chunks)

        split_results = []
        for i, chunk in enumerate(chunks):
            start = result.start_time + i * chunk_duration
            end = result.start_time + (i + 1) * chunk_duration
            split_results.append(
                SyncResult(
                    text=chunk,
                    start_time=start,
                    end_time=end,
                    confidence=result.confidence,
                    line_number=result.line_number,
                    translation=result.translation,
                    pronunciation=result.pronunciation,
                )
            )

        return split_results

    def _split_by_word_segments(self, result: SyncResult, max_chars: int) -> list[SyncResult]:
        segments = result.word_segments
        if not segments:
            return [result]

        chunks = []
        current_words = []
        current_len = 0
        chunk_start_idx = 0

        for i, seg in enumerate(segments):
            word_len = len(seg.word) + (1 if current_words else 0)
            if current_len + word_len > max_chars and current_words:
                chunks.append((chunk_start_idx, i - 1, current_words))
                current_words = [seg]
                current_len = len(seg.word)
                chunk_start_idx = i
            else:
                current_words.append(seg)
                current_len += word_len

        if current_words:
            chunks.append((chunk_start_idx, len(segments) - 1, current_words))

        split_results = []
        for start_idx, end_idx, word_list in chunks:
            text = " ".join(w.word for w in word_list)
            start_time = segments[start_idx].start
            end_time = segments[end_idx].end
            word_segs = [WordSegment(w.word, w.start, w.end, w.confidence) for w in word_list]

            split_results.append(
                SyncResult(
                    text=text,
                    start_time=start_time,
                    end_time=end_time,
                    confidence=result.confidence,
                    line_number=result.line_number,
                    word_segments=word_segs,
                    translation=result.translation,
                    pronunciation=result.pronunciation,
                )
            )

        return split_results

    def _process_word_mode(self, results: list[SyncResult]) -> list[SyncResult]:
        processed = []

        for result in results:
            if not result.word_segments:
                processed.append(result)
                continue

            for seg in result.word_segments:
                duration = seg.end - seg.start
                if duration < self.settings.min_duration:
                    continue

                processed.append(
                    SyncResult(
                        text=seg.word,
                        start_time=seg.start,
                        end_time=seg.end,
                        confidence=seg.confidence,
                        line_number=result.line_number,
                        word_segments=[WordSegment(seg.word, seg.start, seg.end, seg.confidence)],
                        translation=None,
                        pronunciation=None,
                    )
                )

        return processed

    def _process_character_mode(self, results: list[SyncResult]) -> list[SyncResult]:
        processed = []

        for result in results:
            if not result.word_segments:
                char_results = self._split_text_to_chars(result)
                processed.extend(char_results)
                continue

            for seg in result.word_segments:
                char_results = self._split_word_segment_to_chars(seg, result.line_number)
                processed.extend(char_results)

        return processed

    def _split_text_to_chars(self, result: SyncResult) -> list[SyncResult]:
        chars = list(result.text.replace(" ", ""))
        if not chars:
            return [result]

        duration = result.end_time - result.start_time
        char_duration = max(self.settings.min_duration, duration / len(chars))

        char_results = []
        current_time = result.start_time

        for char in chars:
            end_time = min(current_time + char_duration, result.end_time)
            char_results.append(
                SyncResult(
                    text=char,
                    start_time=current_time,
                    end_time=end_time,
                    confidence=result.confidence,
                    line_number=result.line_number,
                )
            )
            current_time = end_time

        return char_results

    def _split_word_segment_to_chars(
        self, seg: WordSegment, line_number: int | None
    ) -> list[SyncResult]:
        chars = list(seg.word.replace(" ", ""))
        if not chars:
            return []

        duration = seg.end - seg.start
        char_duration = max(self.settings.min_duration, duration / len(chars))

        char_results = []
        current_time = seg.start

        for char in chars:
            end_time = min(current_time + char_duration, seg.end)
            char_results.append(
                SyncResult(
                    text=char,
                    start_time=current_time,
                    end_time=end_time,
                    confidence=seg.confidence,
                    line_number=line_number,
                )
            )
            current_time = end_time

        return char_results
