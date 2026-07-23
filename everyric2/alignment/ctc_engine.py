"""CTC-based forced alignment engine using wav2vec2 models.

For Japanese/Korean/Chinese: Uses HuggingFace wav2vec2 models with native character support.
For other languages: Uses torchaudio MMS_FA with Latin alphabet.
"""

import logging
import math
import threading
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
    "ja": "facebook/mms-1b-all",
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


_HANGUL_CHO = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
_HANGUL_JUNG = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
_TENSE_TO_PLAIN = {"ㄲ": "ㄱ", "ㄸ": "ㄷ", "ㅃ": "ㅂ", "ㅆ": "ㅅ", "ㅉ": "ㅈ"}
_GLIDE_TO_PLAIN = {"ㅑ": "ㅏ", "ㅒ": "ㅐ", "ㅕ": "ㅓ", "ㅖ": "ㅔ", "ㅛ": "ㅗ", "ㅠ": "ㅜ"}


def _oov_substitute(char: str, vocab) -> str | None:
    """vocab에 없는 한글 음절을 발음이 가까운 vocab 음절로 치환(정렬 전용).

    된소리→예사소리(ㅃ→ㅂ), 활음 제거(ㅛ→ㅗ), 종성 제거를 점진 적용해 vocab에 있는
    첫 후보를 반환. 한글 음절이 아니거나(괄호·기호) 후보가 없으면 None → 정렬에서 제외.
    예) 뿅(ㅃㅛㅇ)→뽕→봉, 얍(ㅇㅑㅂ)→얌→압. 출력 글자는 원본을 유지하고 타이밍만 차용한다.
    """
    code = ord(char)
    if not (0xAC00 <= code <= 0xD7A3):  # 한글 완성형 음절이 아님
        return None
    s = code - 0xAC00
    cho_i, jung_i, jong_i = s // 588, (s % 588) // 28, s % 28
    cho, jung = _HANGUL_CHO[cho_i], _HANGUL_JUNG[jung_i]
    cho_alts = [cho, _TENSE_TO_PLAIN.get(cho, cho)]
    jung_alts = [jung, _GLIDE_TO_PLAIN.get(jung, jung)]
    seen: set[str] = set()
    for c in cho_alts:
        for j in jung_alts:
            for jo in (jong_i, 0):  # 원래 종성 유지 → 종성 제거 순
                cand = chr(0xAC00 + _HANGUL_CHO.index(c) * 588 + _HANGUL_JUNG.index(j) * 28 + jo)
                if cand == char or cand in seen:
                    continue
                seen.add(cand)
                if cand in vocab:
                    return cand
    return None


class CTCEngine(BaseAlignmentEngine):
    def __init__(self, config: AlignmentSettings | None = None):
        super().__init__(config)
        self._model = None
        self._processor = None
        self._current_lang = None
        self._device = None
        self._last_word_timestamps: list[WordTimestamp] = []
        self._last_match_stats = None
        # 직전 정렬에서 star 토큰이 흡수한 (start, end) 구간들 — 디버그/진단용
        self._last_star_spans: list[tuple[float, float]] = []

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

        # 가사에 없는 가창(추임새/애드립/반복 후렴)을 흡수하는 와일드카드 star 채널.
        # forced_align은 정규화된 log_probs를 기대하며, star의 0.0(=log 1.0) 트릭은
        # log_softmax 정규화 후에만 작동한다 (raw logits에서는 0이 평범한 값이라 무효 —
        # 실측 검증됨). star 없이는 정규화가 Viterbi 경로에 영향을 주지 않으므로(프레임별
        # 상수 상쇄) 기존 결과와 동일하다.
        use_star = getattr(self.config, "star_tokens", False)
        emission = torch.nn.functional.log_softmax(logits.float(), dim=-1)
        star_id = emission.shape[-1]
        if use_star:
            star_col = torch.zeros(
                (emission.shape[0], emission.shape[1], 1),
                dtype=emission.dtype,
                device=emission.device,
            )
            emission = torch.cat([emission, star_col], dim=2)

        tokens = []
        line_boundaries = []
        char_info = []
        star_positions: list[int] = []
        current_pos = 0

        if use_star:
            # 인트로(첫 라인 전) 애드립 흡수용 선행 star
            star_positions.append(current_pos)
            tokens.append(star_id)
            current_pos += 1

        for line_idx, line in enumerate(lyrics):
            line_start = current_pos
            for char in line.text:
                # vocab에 있으면 그대로, 없으면(OOV) 발음이 가까운 음절로 치환해 정렬.
                # 출력 word는 원본 글자를 유지하고, 타이밍만 치환 음절의 정렬에서 가져온다.
                tok_char = char if char in vocab else _oov_substitute(char, vocab)
                if tok_char is not None:
                    char_info.append(
                        {
                            "char": char,
                            "line_idx": line_idx,
                            "token_idx": current_pos,
                        }
                    )
                    tokens.append(vocab[tok_char])
                    current_pos += 1

            if "|" in vocab:
                tokens.append(vocab["|"])
                current_pos += 1

            if use_star:
                # 라인 사이·마지막 라인 뒤의 가사 밖 가창 흡수.
                # 일본어 MMS 어댑터는 vocab에 "|"가 없어 star가 유일한 라인 간 완충이다.
                # star span은 char_info에 없으므로 라인 시각 계산에서 자동 제외된다.
                star_positions.append(current_pos)
                tokens.append(star_id)
                current_pos += 1

            line_boundaries.append((line_start, current_pos - 1))

        if not tokens:
            raise AlignmentError("No valid tokens found in lyrics for this language")

        try:
            targets = torch.tensor([tokens], dtype=torch.int32, device=device)
            blank_id = self._processor.tokenizer.pad_token_id  # pyright: ignore[reportAttributeAccessIssue,reportOptionalMemberAccess]
            aligned_tokens, alignment_scores = F.forced_align(emission, targets, blank=blank_id)
            token_spans = F.merge_tokens(aligned_tokens[0], alignment_scores[0], blank=blank_id)
        except Exception as e:
            raise AlignmentError(f"CTC forced alignment failed: {e}")

        if progress_callback:
            progress_callback(4, 5)

        num_frames = logits.shape[1]
        audio_length = waveform.shape[0] / 16000
        ratio = audio_length / num_frames

        # star가 실제로 흡수한 구간 기록 (1프레임=20ms짜리 형식적 흡수는 제외)
        self._last_star_spans = []
        for idx in star_positions:
            if idx < len(token_spans):
                span = token_spans[idx]
                s, e = span.start * ratio, span.end * ratio
                if e - s >= 0.1:
                    self._last_star_spans.append((round(s, 2), round(e, 2)))

        self._last_word_timestamps = []
        line_char_timestamps: dict[int, list[WordTimestamp]] = {}

        for ci in char_info:
            idx = ci["token_idx"]
            line_idx = ci["line_idx"]
            if idx < len(token_spans):
                span = token_spans[idx]
                start_time = span.start * ratio
                end_time = span.end * ratio
                # emission이 log_softmax라 span.score는 평균 로그확률(음수) — 그대로
                # 내보내면 클라이언트 신뢰도 표시가 전부 '낮음'으로 찍힌다.
                # exp로 기하평균 확률(0~1)로 변환해 저장한다.
                wt = WordTimestamp(
                    word=ci["char"],
                    start=start_time,
                    end=end_time,
                    confidence=round(math.exp(min(0.0, float(span.score))), 6),
                )
                self._last_word_timestamps.append(wt)
                if line_idx not in line_char_timestamps:
                    line_char_timestamps[line_idx] = []
                line_char_timestamps[line_idx].append(wt)

        from everyric2.inference.prompt import WordSegment

        # 1) 줄별 정렬 결과 수집 (정렬 실패 = 시각 None)
        line_times: list[list] = []
        for line_idx, line in enumerate(lyrics):
            char_ts = line_char_timestamps.get(line_idx, [])
            if char_ts:
                word_segments = [
                    WordSegment(word=wt.word, start=wt.start, end=wt.end, confidence=wt.confidence)
                    for wt in char_ts
                ]
                line_times.append([char_ts[0].start, char_ts[-1].end, word_segments])
            else:
                # OOV 등으로 정렬된 글자가 0개 → 아래에서 이웃 사이로 보간
                line_times.append([None, None, None])

        # 2) 정렬 실패 줄 보간. 균등 배치(line_idx*total/N)는 실제 정렬 시각과 뒤섞여
        #    순서가 깨지므로(역순·겹침), 앞뒤 정렬 줄 사이 간격에 끼워넣어 순서를 보존한다.
        self._interpolate_unaligned(line_times, audio_length)

        # 3) SyncResult 생성
        results = [
            SyncResult(
                line_number=line.line_number,
                text=line.text,
                start_time=line_times[i][0],
                end_time=line_times[i][1],
                word_segments=line_times[i][2],
            )
            for i, line in enumerate(lyrics)
        ]

        if progress_callback:
            progress_callback(5, 5)

        return results

    @staticmethod
    def _interpolate_unaligned(
        line_times: list[list],
        total_duration: float,
    ) -> None:
        """정렬 실패(시각 None) 줄을 앞뒤 정렬 줄 사이 간격에 균등 분배(순서 보존).

        line_times: 각 줄 [start, end, word_segments] (정렬 실패는 [None, None, None]).
        제자리에서 start/end를 채운다. 전부 실패면 전체 구간에 균등 분배.
        """
        n = len(line_times)
        i = 0
        while i < n:
            if line_times[i][0] is not None:
                i += 1
                continue
            group_start = i
            group_end = i
            while group_end < n and line_times[group_end][0] is None:
                group_end += 1
            group_end -= 1

            prev_end = line_times[group_start - 1][1] if group_start > 0 else 0.0
            if group_end < n - 1 and line_times[group_end + 1][0] is not None:
                next_start = line_times[group_end + 1][0]
            else:
                next_start = total_duration

            available = max(0.0, next_start - prev_end)
            num = group_end - group_start + 1
            seg = available / num if num else 0.0
            if seg < 0.1:  # 빈틈이 거의 없을 때도 최소 길이 보장(근사치)
                seg = 0.1
            for j in range(group_start, group_end + 1):
                off = j - group_start
                line_times[j][0] = prev_end + off * seg
                line_times[j][1] = prev_end + (off + 1) * seg
            i = group_end + 1

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


# 웜 캐시 싱글턴 (WS2-A) — 프로세스 수명 동안 CTC 엔진(과 그 안에 lazy 로드된 wav2vec2/MMS
# 모델)을 상주시킨다. torch를 최상위에서 import하는 모듈이라, 이 접근자는 반드시 호출부에서
# 지연 import해야 API 전용 모드(local_worker=false)에 torch가 딸려 들어오지 않는다.
_shared_ctc_engine: "CTCEngine | None" = None
_shared_ctc_lock = threading.Lock()


def get_shared_ctc_engine(config: AlignmentSettings | None = None) -> "CTCEngine":
    """웜 캐시된 CTCEngine을 돌려준다 (EVERYRIC_SERVER_WARM_MODELS 기준).

    엔진 인스턴스는 _ensure_model_loaded가 로드한 모델을 _current_lang로 캐시하므로, 같은
    엔진을 재사용하면 같은 언어의 두 번째 잡부터 모델 재로드가 0회다. 재사용 시 "warm model
    reuse: ctc" 1줄. warm이 꺼져 있으면 매번 새 엔진(기존 동작)."""
    from everyric2.config.settings import get_settings

    if not get_settings().server.warm_models:
        return CTCEngine(config)
    global _shared_ctc_engine
    with _shared_ctc_lock:
        if _shared_ctc_engine is None:
            _shared_ctc_engine = CTCEngine(config)
        else:
            logger.info("warm model reuse: ctc")
        return _shared_ctc_engine


def clear_shared_ctc_engine() -> None:
    """웜 캐시 해제 (VRAM 가드용) — 다음 요청에서 지연 재생성된다."""
    global _shared_ctc_engine
    with _shared_ctc_lock:
        _shared_ctc_engine = None
