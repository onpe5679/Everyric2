import re
import unicodedata
from dataclasses import dataclass

from everyric2.alignment.base import WordTimestamp
from everyric2.inference.prompt import LyricLine, SyncResult, WordSegment


MIN_LINE_DURATION = 1.5


@dataclass
class MatchedSegment:
    lyric_idx: int
    word_start_idx: int
    word_end_idx: int
    confidence: float


@dataclass
class MatchStats:
    total_lyrics: int
    matched_lyrics: int
    match_rate: float
    avg_confidence: float


class LyricsMatcher:
    def __init__(self):
        self._ja_pattern = re.compile(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]")
        self._ko_pattern = re.compile(r"[\uAC00-\uD7AF\u1100-\u11FF]")
        self.last_match_stats: MatchStats | None = None
        self.last_transcription_words: list[WordTimestamp] = []

    def normalize_text(self, text: str, language: str = "") -> str:
        text = unicodedata.normalize("NFC", text)
        text = text.lower()
        text = re.sub(r"[^\w\s\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF\uAC00-\uD7AF]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text: str, language: str = "") -> list[str]:
        normalized = self.normalize_text(text, language)

        if self._ja_pattern.search(normalized) or self._ko_pattern.search(normalized):
            tokens = []
            current_token = ""
            for char in normalized:
                if char.isspace():
                    if current_token:
                        tokens.append(current_token)
                        current_token = ""
                elif self._ja_pattern.match(char) or self._ko_pattern.match(char):
                    if current_token and not (
                        self._ja_pattern.match(current_token[-1])
                        or self._ko_pattern.match(current_token[-1])
                    ):
                        tokens.append(current_token)
                        current_token = ""
                    current_token += char
                else:
                    if current_token and (
                        self._ja_pattern.match(current_token[-1])
                        or self._ko_pattern.match(current_token[-1])
                    ):
                        tokens.append(current_token)
                        current_token = ""
                    current_token += char

            if current_token:
                tokens.append(current_token)
            return tokens

        return normalized.split()

    def match_lyrics_to_words(
        self,
        lyrics: list[LyricLine],
        words: list[WordTimestamp],
        language: str,
    ) -> list[SyncResult]:
        self.last_transcription_words = words

        if not lyrics:
            return []

        if words:
            total_duration = words[-1].end
        else:
            total_duration = 180.0
            return self._fallback_uniform_distribution(lyrics, 0.0, total_duration)

        lyric_tokens_list = [self.tokenize(lyric.text, language) for lyric in lyrics]
        word_texts = [self.normalize_text(w.word, language) for w in words]

        matches = self._find_best_matches(lyric_tokens_list, word_texts, words)

        matched_count = len(matches)
        total_count = len(lyrics)
        match_rate = matched_count / total_count if total_count > 0 else 0.0
        avg_confidence = (
            sum(m.confidence for m in matches.values()) / matched_count
            if matched_count > 0
            else 0.0
        )

        self.last_match_stats = MatchStats(
            total_lyrics=total_count,
            matched_lyrics=matched_count,
            match_rate=match_rate,
            avg_confidence=avg_confidence,
        )

        # Match rate < 20% indicates poor transcription - fallback to uniform distribution
        # to prevent clustering unmatched lines into tiny gaps between sparse anchors
        if match_rate < 0.2:
            return self._fallback_uniform_distribution(lyrics, 0.0, total_duration)

        # Check for problematic anchor distribution (large unmatched gaps)
        if self._has_problematic_gaps(matches, len(lyrics), total_duration, words):
            return self._hybrid_interpolation(lyrics, matches, words, total_duration)

        results = self._build_results_with_interpolation(lyrics, matches, words, total_duration)

        return self._postprocess_results(results, total_duration)

    def _has_problematic_gaps(
        self,
        matches: dict[int, MatchedSegment],
        total_lyrics: int,
        total_duration: float,
        words: list[WordTimestamp],
    ) -> bool:
        if not matches:
            return True

        sorted_indices = sorted(matches.keys())

        for i in range(len(sorted_indices) - 1):
            curr_idx = sorted_indices[i]
            next_idx = sorted_indices[i + 1]
            gap_lines = next_idx - curr_idx - 1

            if gap_lines >= 3:
                curr_match = matches[curr_idx]
                next_match = matches[next_idx]
                curr_end_time = words[curr_match.word_end_idx].end
                next_start_time = words[next_match.word_start_idx].start
                available_time = next_start_time - curr_end_time

                time_per_gap_line = available_time / gap_lines if gap_lines > 0 else 0
                if time_per_gap_line < MIN_LINE_DURATION:
                    return True

        first_idx = sorted_indices[0]
        if first_idx >= 3:
            return True

        last_idx = sorted_indices[-1]
        if total_lyrics - last_idx - 1 >= 3:
            return True

        return False

    def _hybrid_interpolation(
        self,
        lyrics: list[LyricLine],
        matches: dict[int, MatchedSegment],
        words: list[WordTimestamp],
        total_duration: float,
    ) -> list[SyncResult]:
        n = len(lyrics)
        expected_duration = total_duration / n

        anchor_times = {}
        anchor_word_segments = {}
        for idx, match in matches.items():
            anchor_times[idx] = (
                words[match.word_start_idx].start,
                words[match.word_end_idx].end,
                match.confidence,
            )
            anchor_word_segments[idx] = [
                WordSegment(
                    word=words[j].word,
                    start=words[j].start,
                    end=words[j].end,
                    confidence=words[j].confidence,
                )
                for j in range(match.word_start_idx, match.word_end_idx + 1)
            ]

        results = []
        for i, lyric in enumerate(lyrics):
            if i in anchor_times:
                start_time, end_time, confidence = anchor_times[i]
                word_segments = anchor_word_segments.get(i)
            else:
                start_time = i * expected_duration
                end_time = (i + 1) * expected_duration
                confidence = 0.0
                word_segments = None

            results.append(
                SyncResult(
                    text=lyric.text,
                    start_time=start_time,
                    end_time=end_time,
                    line_number=lyric.line_number,
                    confidence=confidence,
                    word_segments=word_segments,
                )
            )

        for i in range(1, len(results)):
            if results[i].start_time < results[i - 1].start_time:
                results[i].start_time = results[i - 1].end_time
                if results[i].end_time <= results[i].start_time:
                    results[i].end_time = results[i].start_time + expected_duration

        return self._postprocess_results(results, total_duration)

    def _find_best_matches(
        self,
        lyric_tokens_list: list[list[str]],
        word_texts: list[str],
        words: list[WordTimestamp],
    ) -> dict[int, MatchedSegment]:
        matches = {}
        word_idx = 0

        for lyric_idx, lyric_tokens in enumerate(lyric_tokens_list):
            if not lyric_tokens:
                continue

            best_match = None
            best_score = 0.0
            search_start = max(0, word_idx - 10)
            search_end = min(len(word_texts), word_idx + len(lyric_tokens) * 3 + 20)

            for start in range(search_start, search_end):
                for length in range(1, min(len(lyric_tokens) * 2 + 5, search_end - start + 1)):
                    end = start + length - 1
                    if end >= len(word_texts):
                        break

                    score = self._calculate_match_score(
                        lyric_tokens,
                        word_texts[start : end + 1],
                    )

                    if score > best_score:
                        best_score = score
                        best_match = MatchedSegment(
                            lyric_idx=lyric_idx,
                            word_start_idx=start,
                            word_end_idx=end,
                            confidence=score,
                        )

            if best_match and best_score > 0.3:
                matches[lyric_idx] = best_match
                word_idx = best_match.word_end_idx + 1

        return matches

    def _calculate_match_score(
        self,
        lyric_tokens: list[str],
        word_segment: list[str],
    ) -> float:
        if not lyric_tokens or not word_segment:
            return 0.0

        lyric_str = "".join(lyric_tokens)
        word_str = "".join(word_segment)

        if not lyric_str or not word_str:
            return 0.0

        common_chars = 0
        lyric_chars = list(lyric_str)
        word_chars = list(word_str)

        for char in lyric_chars:
            if char in word_chars:
                common_chars += 1
                word_chars.remove(char)

        precision = common_chars / len(lyric_str) if lyric_str else 0
        recall = common_chars / len("".join(word_segment)) if word_segment else 0

        if precision + recall == 0:
            return 0.0

        f1 = 2 * precision * recall / (precision + recall)

        length_ratio = min(len(lyric_str), len(word_str)) / max(len(lyric_str), len(word_str))
        length_penalty = length_ratio**0.5

        return f1 * length_penalty

    def _build_results_with_interpolation(
        self,
        lyrics: list[LyricLine],
        matches: dict[int, MatchedSegment],
        words: list[WordTimestamp],
        total_duration: float,
    ) -> list[SyncResult]:
        results = []

        for i, lyric in enumerate(lyrics):
            if i in matches:
                match = matches[i]
                start_time = words[match.word_start_idx].start
                end_time = words[match.word_end_idx].end

                if end_time - start_time < MIN_LINE_DURATION:
                    if match.word_end_idx + 1 < len(words):
                        next_word = words[match.word_end_idx + 1]
                        if next_word.start - end_time < 0.3:
                            end_time = max(end_time, start_time + MIN_LINE_DURATION)

                confidence = match.confidence

                word_segments = [
                    WordSegment(
                        word=words[j].word,
                        start=words[j].start,
                        end=words[j].end,
                        confidence=words[j].confidence,
                    )
                    for j in range(match.word_start_idx, match.word_end_idx + 1)
                ]
            else:
                start_time = 0.0
                end_time = 0.0
                confidence = 0.0
                word_segments = None

            results.append(
                SyncResult(
                    text=lyric.text,
                    start_time=start_time,
                    end_time=end_time,
                    line_number=lyric.line_number,
                    confidence=confidence,
                    word_segments=word_segments,
                )
            )

        self._interpolate_unmatched(results, matches, total_duration)

        return results

    def _interpolate_unmatched(
        self,
        results: list[SyncResult],
        matches: dict[int, MatchedSegment],
        total_duration: float,
    ) -> None:
        """Two-pass interpolation: find unmatched groups, distribute time evenly within each."""
        n = len(results)
        if n == 0:
            return

        i = 0
        while i < n:
            if i in matches:
                i += 1
                continue

            group_start = i

            group_end = i
            while group_end < n and group_end not in matches:
                group_end += 1
            group_end -= 1

            if group_start > 0:
                prev_end = results[group_start - 1].end_time
            else:
                prev_end = 0.0

            if group_end < n - 1:
                next_start = results[group_end + 1].start_time
            else:
                next_start = total_duration

            available_time = next_start - prev_end
            num_lines = group_end - group_start + 1

            min_required = num_lines * MIN_LINE_DURATION

            if available_time < min_required:
                segment_duration = max(0.1, available_time / num_lines)
            else:
                segment_duration = available_time / num_lines

            for j in range(group_start, group_end + 1):
                offset = j - group_start
                results[j].start_time = prev_end + offset * segment_duration
                results[j].end_time = prev_end + (offset + 1) * segment_duration

            i = group_end + 1

    def _postprocess_results(
        self,
        results: list[SyncResult],
        total_duration: float,
    ) -> list[SyncResult]:
        if not results:
            return results

        for i in range(len(results) - 1):
            curr = results[i]
            next_line = results[i + 1]

            if curr.end_time > next_line.start_time:
                mid = (curr.end_time + next_line.start_time) / 2
                curr.end_time = mid
                next_line.start_time = mid

        if results and results[-1].end_time > total_duration:
            results[-1].end_time = total_duration

        for r in results:
            if r.start_time >= r.end_time:
                r.end_time = r.start_time + 0.5

        return results

    def _fallback_uniform_distribution(
        self,
        lyrics: list[LyricLine],
        start_time: float,
        end_time: float,
    ) -> list[SyncResult]:
        if not lyrics:
            return []

        total_duration = end_time - start_time
        line_duration = total_duration / len(lyrics)

        results = []
        for i, lyric in enumerate(lyrics):
            line_start = start_time + i * line_duration
            line_end = start_time + (i + 1) * line_duration

            results.append(
                SyncResult(
                    text=lyric.text,
                    start_time=line_start,
                    end_time=line_end,
                    line_number=lyric.line_number,
                    confidence=0.0,
                )
            )

        return results
