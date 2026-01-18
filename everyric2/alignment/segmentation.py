from everyric2.config.settings import SegmentationSettings, get_settings
from everyric2.inference.prompt import SyncResult, WordSegment
from everyric2.text.tokenizer import Tokenizer, get_tokenizer


class SegmentationProcessor:
    def __init__(
        self,
        settings: SegmentationSettings | None = None,
        language: str = "ja",
    ):
        self.settings = settings or get_settings().segmentation
        self.language = language
        self._tokenizer: Tokenizer | None = None

    @property
    def tokenizer(self) -> Tokenizer:
        if self._tokenizer is None:
            self._tokenizer = get_tokenizer(
                language=self.language,
                use_mecab=self.settings.use_mecab,
            )
        return self._tokenizer

    def process(
        self,
        results: list[SyncResult],
        language: str | None = None,
    ) -> list[SyncResult]:
        if language:
            self.language = language
            self._tokenizer = None

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

        words = self.tokenizer.split_into_words(text)
        if len(words) <= 1:
            return [result]

        chunks = []
        current_chunk = []
        current_len = 0
        is_cjk = self.language in ("ja", "zh", "ko")
        separator = "" if is_cjk else " "

        for word in words:
            word_len = len(word) + (len(separator) if current_chunk else 0)
            if current_len + word_len > max_chars and current_chunk:
                chunks.append(separator.join(current_chunk))
                current_chunk = [word]
                current_len = len(word)
            else:
                current_chunk.append(word)
                current_len += word_len

        if current_chunk:
            chunks.append(separator.join(current_chunk))

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
        is_cjk = self.language in ("ja", "zh", "ko")
        sep_len = 0 if is_cjk else 1

        for i, seg in enumerate(segments):
            word_len = len(seg.word) + (sep_len if current_words else 0)
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
        separator = "" if is_cjk else " "
        for start_idx, end_idx, word_list in chunks:
            text = separator.join(w.word for w in word_list)
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
            if result.word_segments:
                is_char_level = self._is_character_level_segments(result.word_segments)
                if is_char_level:
                    word_results = self._combine_chars_to_words(result)
                else:
                    word_results = self._use_word_segments_directly(result)
                processed.extend(word_results)
            else:
                word_results = self._split_text_to_words(result)
                processed.extend(word_results)

        return processed

    def _is_character_level_segments(self, segments: list[WordSegment]) -> bool:
        if not segments:
            return False

        avg_segment_len = sum(len(seg.word) for seg in segments) / len(segments)

        is_cjk = self.language in ("ja", "zh", "ko")
        if is_cjk:
            # CJK char segments: avg ~1, word segments: avg > 2
            return avg_segment_len < 2.0

        # Non-CJK: char segments avg < 1.5, word segments avg > 1.5
        return avg_segment_len < 1.5

    def _use_word_segments_directly(self, result: SyncResult) -> list[SyncResult]:
        if not result.word_segments:
            return [result]

        word_results = []
        for seg in result.word_segments:
            word_results.append(
                SyncResult(
                    text=seg.word,
                    start_time=seg.start,
                    end_time=seg.end,
                    confidence=seg.confidence,
                    line_number=result.line_number,
                    word_segments=[WordSegment(seg.word, seg.start, seg.end, seg.confidence)],
                )
            )

        return word_results if word_results else [result]

    def _combine_chars_to_words(self, result: SyncResult) -> list[SyncResult]:
        words = self.tokenizer.split_into_words(result.text)
        if not words or not result.word_segments:
            return [result]

        char_segments = list(result.word_segments)
        char_idx = 0
        word_results = []

        for word in words:
            word_chars = list(word)
            word_segs = []

            for _ in word_chars:
                if char_idx < len(char_segments):
                    word_segs.append(char_segments[char_idx])
                    char_idx += 1

            if word_segs:
                start_time = word_segs[0].start
                end_time = word_segs[-1].end
                avg_conf = sum(s.confidence for s in word_segs) / len(word_segs)

                word_results.append(
                    SyncResult(
                        text=word,
                        start_time=start_time,
                        end_time=end_time,
                        confidence=avg_conf,
                        line_number=result.line_number,
                        word_segments=[
                            WordSegment(s.word, s.start, s.end, s.confidence) for s in word_segs
                        ],
                    )
                )

        return word_results if word_results else [result]

    def _split_text_to_words(self, result: SyncResult) -> list[SyncResult]:
        words = self.tokenizer.split_into_words(result.text)
        if not words:
            return [result]

        duration = result.end_time - result.start_time
        word_duration = max(self.settings.min_duration, duration / len(words))

        word_results = []
        current_time = result.start_time

        for word in words:
            end_time = min(current_time + word_duration, result.end_time)
            if end_time - current_time >= self.settings.min_duration:
                word_results.append(
                    SyncResult(
                        text=word,
                        start_time=current_time,
                        end_time=end_time,
                        confidence=result.confidence,
                        line_number=result.line_number,
                    )
                )
            current_time = end_time

        return word_results if word_results else [result]

    def _process_character_mode(self, results: list[SyncResult]) -> list[SyncResult]:
        processed = []

        for result in results:
            if result.word_segments:
                char_results = []
                for seg in result.word_segments:
                    char_results.extend(self._split_word_segment_to_chars(seg, result.line_number))
                char_results = self._extend_to_next_start(char_results, result.end_time)
                processed.extend(char_results)
            else:
                char_results = self._split_text_to_chars(result)
                char_results = self._extend_to_next_start(char_results, result.end_time)
                processed.extend(char_results)

        return processed

    def _extend_to_next_start(
        self, char_results: list[SyncResult], line_end_time: float
    ) -> list[SyncResult]:
        if not char_results:
            return char_results

        for i in range(len(char_results) - 1):
            char_results[i].end_time = char_results[i + 1].start_time

        char_results[-1].end_time = line_end_time
        return char_results

    def _split_text_to_chars(self, result: SyncResult) -> list[SyncResult]:
        chars = self.tokenizer.split_into_characters(result.text)
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
        chars = self.tokenizer.split_into_characters(seg.word)
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
