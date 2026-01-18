"""CTC-based forced alignment engine using wav2vec2 models.

For Japanese/Korean/Chinese: Uses HuggingFace wav2vec2 models with native character support.
For other languages: Uses torchaudio MMS_FA with Latin alphabet.
"""

import logging
from collections.abc import Callable
from typing import Literal

import torch
import torchaudio
import torchaudio.functional as F

from everyric2.alignment.base import (
    AlignmentError,
    BaseAlignmentEngine,
    EngineNotAvailableError,
    TranscriptionResult,
    WordTimestamp,
)
from everyric2.alignment.matcher import MatchStats
from everyric2.audio.loader import AudioData
from everyric2.config.settings import AlignmentSettings
from everyric2.inference.prompt import LyricLine, SyncResult

logger = logging.getLogger(__name__)

LANG_MODEL_MAP = {
    "ja": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
    "ko": "facebook/mms-1b-all",
    "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    "en": "facebook/mms-1b-all",
}

MMS_LANG_CODES = {
    "ko": "kor",
    "ja": "jpn",
    "zh": "cmn-script_simplified",
    "en": "eng",
}


def detect_language_from_text(text: str) -> tuple[str, bool]:
    """Detect language from lyrics text.

    Returns:
        Tuple of (primary_language, is_multilingual)
        - primary_language: 'ja', 'ko', 'zh', or 'en' (dominant language)
        - is_multilingual: True if multiple scripts detected → recommends MMS 1B-all
    """
    ja_count = 0
    ko_count = 0
    zh_count = 0
    en_count = 0

    for char in text:
        code = ord(char)
        if 0x3040 <= code <= 0x309F or 0x30A0 <= code <= 0x30FF:  # Hiragana/Katakana
            ja_count += 1
        elif 0xAC00 <= code <= 0xD7AF or 0x1100 <= code <= 0x11FF:  # Hangul
            ko_count += 1
        elif 0x4E00 <= code <= 0x9FFF:  # CJK Ideographs
            zh_count += 1
        elif 0x0041 <= code <= 0x007A:  # Basic Latin (A-Za-z)
            en_count += 1

    detected = []
    if ja_count > 0:
        detected.append(("ja", ja_count))
    if ko_count > 0:
        detected.append(("ko", ko_count))
    if zh_count > 0 and ja_count == 0:  # CJK without kana = Chinese
        detected.append(("zh", zh_count))
    if en_count > 10:
        detected.append(("en", en_count))

    is_multilingual = len(detected) >= 2

    if is_multilingual:
        dominant = max(detected, key=lambda x: x[1])
        logger.info(
            f"Multiple languages detected: {[d[0] for d in detected]} → primary: {dominant[0]}, using MMS 1B-all"
        )
        return (dominant[0], True)

    if ja_count > 0:
        return ("ja", False)
    if ko_count > 0:
        return ("ko", False)
    if zh_count > 0:
        return ("zh", False)
    return ("en", False)


class CTCEngine(BaseAlignmentEngine):
    def __init__(self, config: AlignmentSettings | None = None):
        super().__init__(config)
        self._model = None
        self._processor = None
        self._current_lang = None
        self._device = None
        self._last_word_timestamps: list[WordTimestamp] = []
        self._last_match_stats = None

    def is_available(self) -> bool:
        try:
            import torchaudio  # noqa: F401
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor  # noqa: F401

            return True
        except ImportError:
            return False

    def _get_device(self) -> torch.device:
        if self._device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self._device

    def _ensure_model_loaded(self, language: str, force_mms: bool = False) -> None:
        cache_key = f"{language}_mms" if force_mms else language
        if self._model is not None and self._current_lang == cache_key:
            return

        if not self.is_available():
            raise EngineNotAvailableError(
                "Required packages not installed. Install with: pip install transformers torchaudio"
            )

        device = self._get_device()

        use_mms = force_mms or (
            language in LANG_MODEL_MAP and LANG_MODEL_MAP[language] == "facebook/mms-1b-all"
        )

        if use_mms:
            from transformers import AutoProcessor, Wav2Vec2ForCTC

            logger.info(f"Loading MMS 1B-all with {language} adapter (force_mms={force_mms})")
            self._processor = AutoProcessor.from_pretrained("facebook/mms-1b-all")
            self._model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all").to(device)
            self._model.eval()

            mms_lang_code = MMS_LANG_CODES.get(language, "eng")
            self._processor.tokenizer.set_target_lang(mms_lang_code)
            self._model.load_adapter(mms_lang_code)
            logger.info(f"MMS adapter loaded: {mms_lang_code}")
        elif language in LANG_MODEL_MAP:
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

            model_name = LANG_MODEL_MAP[language]
            logger.info(f"Loading HuggingFace model: {model_name}")
            self._processor = Wav2Vec2Processor.from_pretrained(model_name)
            self._model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)  # pyright: ignore[reportArgumentType]
            self._model.eval()
        else:
            logger.info("Loading torchaudio MMS_FA model")
            bundle = torchaudio.pipelines.MMS_FA
            self._model = bundle.get_model(with_star=False).to(device)
            self._processor = bundle

        self._current_lang = cache_key

    def transcribe(
        self,
        audio: AudioData,
        language: str | None = None,
    ) -> TranscriptionResult:
        raise NotImplementedError(
            "CTCEngine does not support transcription. Use for forced alignment only."
        )

    def _align_cjk(
        self,
        waveform: torch.Tensor,
        lyrics: list[LyricLine],
        language: str,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[SyncResult]:
        device = self._get_device()

        if progress_callback:
            progress_callback(2, 5)

        with torch.inference_mode():
            inputs = self._processor(  # pyright: ignore[reportCallIssue,reportOptionalCall]
                waveform.numpy(), sampling_rate=16000, return_tensors="pt", padding=True
            )
            input_values = inputs.input_values.to(device)
            logits = self._model(input_values).logits  # pyright: ignore[reportOptionalCall]

        if progress_callback:
            progress_callback(3, 5)

        vocab = self._processor.tokenizer.get_vocab()  # pyright: ignore[reportAttributeAccessIssue,reportOptionalMemberAccess]

        tokens = []
        line_boundaries = []
        char_info = []
        current_pos = 0

        for line_idx, line in enumerate(lyrics):
            line_start = current_pos
            for char in line.text:
                if char in vocab:
                    char_info.append(
                        {
                            "char": char,
                            "line_idx": line_idx,
                            "token_idx": current_pos,
                        }
                    )
                    tokens.append(vocab[char])
                    current_pos += 1

            if "|" in vocab:
                tokens.append(vocab["|"])
                current_pos += 1

            line_boundaries.append((line_start, current_pos - 1))

        if not tokens:
            raise AlignmentError("No valid tokens found in lyrics for this language")

        try:
            targets = torch.tensor([tokens], dtype=torch.int32, device=device)
            blank_id = self._processor.tokenizer.pad_token_id  # pyright: ignore[reportAttributeAccessIssue,reportOptionalMemberAccess]
            aligned_tokens, alignment_scores = F.forced_align(logits, targets, blank=blank_id)
            token_spans = F.merge_tokens(aligned_tokens[0], alignment_scores[0], blank=blank_id)
        except Exception as e:
            raise AlignmentError(f"CTC forced alignment failed: {e}")

        if progress_callback:
            progress_callback(4, 5)

        num_frames = logits.shape[1]
        audio_length = waveform.shape[0] / 16000
        ratio = audio_length / num_frames

        self._last_word_timestamps = []
        line_char_timestamps: dict[int, list[WordTimestamp]] = {}

        for ci in char_info:
            idx = ci["token_idx"]
            line_idx = ci["line_idx"]
            if idx < len(token_spans):
                span = token_spans[idx]
                start_time = span.start * ratio
                end_time = span.end * ratio
                wt = WordTimestamp(
                    word=ci["char"],
                    start=start_time,
                    end=end_time,
                    confidence=span.score,
                )
                self._last_word_timestamps.append(wt)
                if line_idx not in line_char_timestamps:
                    line_char_timestamps[line_idx] = []
                line_char_timestamps[line_idx].append(wt)

        from everyric2.inference.prompt import WordSegment

        results = []
        for line_idx, line in enumerate(lyrics):
            char_ts = line_char_timestamps.get(line_idx, [])

            if char_ts:
                start_time = char_ts[0].start
                end_time = char_ts[-1].end
                word_segments = [
                    WordSegment(word=wt.word, start=wt.start, end=wt.end, confidence=wt.confidence)
                    for wt in char_ts
                ]
            else:
                start_time = line_idx * audio_length / len(lyrics)
                end_time = (line_idx + 1) * audio_length / len(lyrics)
                word_segments = None

            results.append(
                SyncResult(
                    line_number=line.line_number,
                    text=line.text,
                    start_time=start_time,
                    end_time=end_time,
                    word_segments=word_segments,
                )
            )

        if progress_callback:
            progress_callback(5, 5)

        return results

    def _align_mms(
        self,
        waveform: torch.Tensor,
        lyrics: list[LyricLine],
        language: str,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[SyncResult]:
        device = self._get_device()
        bundle = self._processor

        if progress_callback:
            progress_callback(2, 5)

        dictionary = bundle.get_dict(star=None)  # pyright: ignore[reportAttributeAccessIssue]

        all_words = []
        line_word_counts = []

        for line in lyrics:
            words = line.text.lower().split()
            cleaned_words = []
            for word in words:
                cleaned = "".join(c for c in word if c in dictionary and dictionary[c] != 0)
                if cleaned:
                    cleaned_words.append(cleaned)
                    all_words.append(cleaned)
            line_word_counts.append(len(cleaned_words))

        if not all_words:
            raise AlignmentError("No valid characters found in lyrics for MMS_FA model")

        if progress_callback:
            progress_callback(3, 5)

        waveform_2d = waveform.unsqueeze(0).to(device)
        with torch.inference_mode():
            emission, _ = self._model(waveform_2d)  # pyright: ignore[reportOptionalCall]

        tokens = [
            dictionary[c]
            for word in all_words
            for c in word
            if c in dictionary and dictionary[c] != 0
        ]

        try:
            aligned_tokens, alignment_scores = F.forced_align(
                emission, torch.tensor([tokens], dtype=torch.int32, device=device), blank=0
            )
            token_spans = F.merge_tokens(aligned_tokens[0], alignment_scores[0])
        except Exception as e:
            raise AlignmentError(f"MMS_FA forced alignment failed: {e}")

        if progress_callback:
            progress_callback(4, 5)

        word_lengths = [len(word) for word in all_words]
        word_spans = []
        idx = 0
        for length in word_lengths:
            word_spans.append(token_spans[idx : idx + length])
            idx += length

        num_frames = emission.shape[1]
        ratio = waveform.shape[0] / num_frames / 16000

        self._last_word_timestamps = []
        for word, spans in zip(all_words, word_spans):
            if spans:
                start_time = spans[0].start * ratio
                end_time = spans[-1].end * ratio
                avg_score = sum(s.score for s in spans) / len(spans)

                self._last_word_timestamps.append(
                    WordTimestamp(
                        word=word,
                        start=start_time,
                        end=end_time,
                        confidence=avg_score,
                    )
                )

        results = []
        word_idx = 0
        audio_length = waveform.shape[0] / 16000

        for line_idx, line in enumerate(lyrics):
            word_count = line_word_counts[line_idx]

            if word_count > 0 and word_idx < len(word_spans):
                line_spans = word_spans[word_idx : word_idx + word_count]
                if line_spans and line_spans[0] and line_spans[-1]:
                    start_time = line_spans[0][0].start * ratio
                    end_time = line_spans[-1][-1].end * ratio
                else:
                    start_time = line_idx * audio_length / len(lyrics)
                    end_time = (line_idx + 1) * audio_length / len(lyrics)
                word_idx += word_count
            else:
                start_time = line_idx * audio_length / len(lyrics)
                end_time = (line_idx + 1) * audio_length / len(lyrics)

            results.append(
                SyncResult(
                    line_number=line.line_number,
                    text=line.text,
                    start_time=start_time,
                    end_time=end_time,
                )
            )

        if progress_callback:
            progress_callback(5, 5)

        return results

    def align(
        self,
        audio: AudioData,
        lyrics: list[LyricLine],
        language: str | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[SyncResult]:
        force_mms = False
        if language and language != "auto":
            resolved_lang = language
        else:
            lyrics_text = " ".join(line.text for line in lyrics)
            resolved_lang, is_multilingual = detect_language_from_text(lyrics_text)
            if is_multilingual:
                force_mms = True
            logger.info(f"Auto-detected language: {resolved_lang}, multilingual: {is_multilingual}")
        self._ensure_model_loaded(resolved_lang, force_mms=force_mms)

        if progress_callback:
            progress_callback(1, 5)

        from everyric2.audio.loader import AudioLoader

        loader = AudioLoader()
        prepared = loader.prepare_for_alignment(audio, target_sr=16000, normalize=True)

        waveform = torch.from_numpy(prepared.waveform.astype("float32"))
        if waveform.dim() == 2:
            waveform = waveform.mean(dim=0)

        if resolved_lang in LANG_MODEL_MAP:
            results = self._align_cjk(waveform, lyrics, resolved_lang, progress_callback)
            self._last_match_stats = self._calculate_match_stats(results)
            return results
        else:
            results = self._align_mms(waveform, lyrics, resolved_lang, progress_callback)
            from everyric2.alignment.matcher import LyricsMatcher

            matcher = LyricsMatcher()
            matched_results = matcher.match_lyrics_to_words(
                lyrics, self._last_word_timestamps, resolved_lang
            )

            self._last_match_stats = self._calculate_match_stats(matched_results)
            return matched_results

    def _calculate_match_stats(self, results: list) -> "MatchStats":
        """Calculate match stats consistently for both CJK and MMS paths.

        Uses adaptive threshold: > 0 for positive confidence (CJK),
        > -5 for log-probability confidence (MMS/English).
        """
        from everyric2.alignment.matcher import MatchStats

        all_confidences = [
            seg.confidence
            for r in results
            if r.word_segments
            for seg in r.word_segments
            if seg.confidence is not None
        ]

        if not all_confidences:
            return MatchStats(
                total_lyrics=len(results),
                matched_lyrics=0,
                match_rate=0.0,
                avg_confidence=0.0,
            )

        avg_conf = sum(all_confidences) / len(all_confidences)

        # Adaptive threshold: log-probs are negative, CTC scores can be positive
        # For log-probs (avg < 0): use -5 as "good enough" threshold
        # For CTC scores (avg >= 0): use 0 as threshold
        threshold = -5.0 if avg_conf < 0 else 0.0
        good_matches = sum(1 for c in all_confidences if c > threshold)
        match_rate = good_matches / len(all_confidences)

        return MatchStats(
            total_lyrics=len(results),
            matched_lyrics=sum(1 for r in results if r.word_segments),
            match_rate=match_rate,
            avg_confidence=avg_conf,
        )

    def get_last_transcription_data(self) -> tuple[list[WordTimestamp], MatchStats | None, str]:
        return (self._last_word_timestamps, self._last_match_stats, "ctc")

    def get_transcription_sets(self) -> list[tuple[list[WordTimestamp], MatchStats | None, str]]:
        data = self.get_last_transcription_data()
        if data[0]:
            return [data]
        return []

    @staticmethod
    def get_engine_type() -> Literal["ctc"]:
        return "ctc"
