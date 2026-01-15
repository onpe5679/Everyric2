"""CTC-based forced alignment engine using wav2vec2 models.

For Japanese/Korean/Chinese: Uses HuggingFace wav2vec2 models with native character support.
For other languages: Uses torchaudio MMS_FA with Latin alphabet.
"""

import logging
from typing import Callable, Literal

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
    "ko": "kresnik/wav2vec2-large-xlsr-korean",
    "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
}


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
            import torchaudio
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

            return True
        except ImportError:
            return False

    def _get_device(self) -> torch.device:
        if self._device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self._device

    def _ensure_model_loaded(self, language: str) -> None:
        if self._model is not None and self._current_lang == language:
            return

        if not self.is_available():
            raise EngineNotAvailableError(
                "Required packages not installed. Install with: pip install transformers torchaudio"
            )

        device = self._get_device()

        if language in LANG_MODEL_MAP:
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

        self._current_lang = language

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
        id2char = {v: k for k, v in vocab.items()}

        tokens = []
        line_boundaries = []
        word_info = []
        current_pos = 0

        for line_idx, line in enumerate(lyrics):
            line_start = current_pos
            words = line.text.split()

            for word_idx, word in enumerate(words):
                word_start = current_pos
                for char in word:
                    if char in vocab:
                        tokens.append(vocab[char])
                        current_pos += 1
                word_end = current_pos
                if word_start < word_end:
                    word_info.append(
                        {
                            "word": word,
                            "line_idx": line_idx,
                            "token_start": word_start,
                            "token_end": word_end,
                        }
                    )

                if "|" in vocab and word_idx < len(words) - 1:
                    tokens.append(vocab["|"])
                    current_pos += 1

            if "|" in vocab:
                tokens.append(vocab["|"])
                current_pos += 1

            line_boundaries.append((line_start, current_pos - 1))

        if not tokens:
            raise AlignmentError("No valid tokens found in lyrics for this language")

        try:
            targets = torch.tensor([tokens], dtype=torch.int32, device=device)
            aligned_tokens, alignment_scores = F.forced_align(logits, targets, blank=0)
            token_spans = F.merge_tokens(aligned_tokens[0], alignment_scores[0])
        except Exception as e:
            raise AlignmentError(f"CTC forced alignment failed: {e}")

        if progress_callback:
            progress_callback(4, 5)

        num_frames = logits.shape[1]
        audio_length = waveform.shape[0] / 16000
        ratio = audio_length / num_frames

        self._last_word_timestamps = []
        for wi in word_info:
            start_idx = wi["token_start"]
            end_idx = wi["token_end"] - 1

            if start_idx < len(token_spans) and end_idx < len(token_spans):
                start_frame = token_spans[start_idx].start
                end_frame = token_spans[end_idx].end
                start_time = start_frame * ratio
                end_time = end_frame * ratio

                scores = [
                    token_spans[i].score
                    for i in range(start_idx, end_idx + 1)
                    if i < len(token_spans)
                ]
                avg_score = sum(scores) / len(scores) if scores else 0.0

                self._last_word_timestamps.append(
                    WordTimestamp(
                        word=wi["word"],
                        start=start_time,
                        end=end_time,
                        confidence=avg_score,
                    )
                )

        results = []
        for line_idx, line in enumerate(lyrics):
            start_idx, end_idx = line_boundaries[line_idx]

            if start_idx < len(token_spans) and end_idx < len(token_spans):
                start_frame = token_spans[start_idx].start
                end_frame = token_spans[min(end_idx, len(token_spans) - 1)].end
                start_time = start_frame * ratio
                end_time = end_frame * ratio
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
        resolved_lang = self._resolve_language(language)
        self._ensure_model_loaded(resolved_lang)

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
        else:
            results = self._align_mms(waveform, lyrics, resolved_lang, progress_callback)

        from everyric2.alignment.matcher import LyricsMatcher

        matcher = LyricsMatcher()
        _ = matcher.match_lyrics_to_words(lyrics, self._last_word_timestamps, resolved_lang)
        self._last_match_stats = matcher.last_match_stats

        return results

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
