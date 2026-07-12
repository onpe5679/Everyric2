import json
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import requests
from dotenv import load_dotenv

from everyric2.config.settings import TranslationSettings, get_settings
from everyric2.inference.prompt import LyricLine

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class TranslationLine:
    original: str
    translation: str
    pronunciation: str | None = None


@dataclass
class TranslationResult:
    lines: list[TranslationLine]
    source_lang: str
    target_lang: str
    engine: str
    tone: str


_HANGUL_RE = re.compile(r"[가-힣]")
_ASCII_LETTER_RE = re.compile(r"[A-Za-z]")
_OTHER_LETTER_RE = re.compile(r"[^\x00-\x7F가-힣\s\W]")

TONE_PROMPTS = {
    "literal": "Translate literally, preserving the original meaning as closely as possible.",
    "natural": "Translate naturally so it sounds fluent to native speakers.",
    "poetic": "Translate poetically, maintaining rhythm, beauty, and artistic expression.",
    "casual": "Translate in casual, conversational language.",
    "formal": "Translate in formal, polite language.",
}


class BaseTranslator(ABC):
    def __init__(self, settings: TranslationSettings | None = None):
        self.settings = settings or get_settings().translation

    @abstractmethod
    def translate(
        self,
        lyrics: list[LyricLine] | str,
        source_lang: str = "auto",
        target_lang: str | None = None,
        context: str | None = None,
    ) -> TranslationResult:
        pass

    def _build_prompt(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        include_pronunciation: bool,
        context: str | None = None,
    ) -> str:
        lang_names = {"ko": "Korean", "en": "English", "ja": "Japanese", "zh": "Chinese"}
        target = lang_names.get(target_lang, target_lang)
        tone_instruction = TONE_PROMPTS.get(self.settings.tone, TONE_PROMPTS["natural"])
        context_block = f"\nSong: {context}" if context else ""

        # 가사 맥락 지시 — 줄별 고립 직역(기계번역 톤)을 막고 곡 전체를 하나의 화자로 잇는다
        register_hint = (
            " Korean song lyrics use the plain intimate register (반말/해라체, e.g. ~해, ~야,"
            " ~잖아) — never formal endings like ~습니다/~어요 unless the original is"
            " explicitly formal."
            if target_lang == "ko"
            else ""
        )
        lyrics_guidance = (
            "These lines are the lyrics of ONE song, in order. Read the whole song first,"
            " then translate so the lines flow as a coherent song: keep one consistent"
            " speaker, emotional register and formality throughout, resolve omitted"
            " subjects/pronouns from surrounding lines, and prefer natural lyrical phrasing"
            f" over word-for-word rendering. Never translate a line in isolation.{register_hint}"
        )

        if include_pronunciation:
            if target_lang == "ko":
                # 한국어 UI에서는 로마자가 아니라 한글 독음이 발음표기다 (보카로 위키와 동일 규약)
                pron_rule = (
                    "2. The Korean Hangul reading (한글 독음) of the ORIGINAL line — transcribe"
                    " how the original line SOUNDS using Hangul only"
                    " (e.g. 時計の針が → 도케이노 하리가). This is NOT the translation and NOT"
                    " romanization."
                )
                pron_example = (
                    '[{"original": "時計の針が", "translation": "시곗바늘이",'
                    ' "pronunciation": "도케이노 하리가"}]'
                )
                pron_note = (
                    "- pronunciation must be the Hangul reading of how the ORIGINAL line"
                    " sounds, never romanization and never a translation"
                )
            else:
                pron_rule = "2. Romanized pronunciation of the ORIGINAL text (not the translation)"
                pron_example = '[{"original": "原文", "translation": "translation", "pronunciation": "genbun"}]'
                pron_note = "- pronunciation should be romanization of the ORIGINAL lyrics"
            return f"""Translate these song lyrics to {target}.
{tone_instruction}
{lyrics_guidance}{context_block}

For each line, provide:
1. The translation
{pron_rule}

Output as JSON array:
{pron_example}

IMPORTANT:
- Keep the same number of lines, in the same order
- Output ONLY the JSON array, no explanations
{pron_note}

LYRICS:
{text}"""
        else:
            return f"""Translate these song lyrics to {target}.
{tone_instruction}
{lyrics_guidance}{context_block}

Keep the same line structure (same number of lines).
Only output the translation, no explanations or notes.

LYRICS:
{text}

TRANSLATION:"""

    def _parse_json_response(
        self, response: str, original_lines: list[str]
    ) -> list[TranslationLine]:
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
        response = response.strip()

        if response.startswith("```"):
            match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response, re.DOTALL)
            if match:
                response = match.group(1).strip()

        try:
            data = json.loads(response)
            if isinstance(data, list):
                results = []
                for i, item in enumerate(data):
                    if isinstance(item, dict):
                        orig = item.get(
                            "original", original_lines[i] if i < len(original_lines) else ""
                        )
                        results.append(
                            TranslationLine(
                                original=orig,
                                translation=item.get("translation", ""),
                                pronunciation=item.get("pronunciation"),
                            )
                        )
                return results
        except json.JSONDecodeError:
            pass

        array_match = re.search(r"\[.*\]", response, re.DOTALL)
        if array_match:
            try:
                data = json.loads(array_match.group())
                results = []
                for i, item in enumerate(data):
                    if isinstance(item, dict):
                        orig = item.get(
                            "original", original_lines[i] if i < len(original_lines) else ""
                        )
                        results.append(
                            TranslationLine(
                                original=orig,
                                translation=item.get("translation", ""),
                                pronunciation=item.get("pronunciation"),
                            )
                        )
                return results
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Failed to parse JSON response: {response[:200]}")

    def _detect_lang_heuristic(self, text: str) -> str:
        """한글/ASCII 비율 기반 언어 추정. source_lang="auto"일 때 발음 생략 게이트에만 쓰이는
        거친 휴리스틱이며 실제 번역 언어 감지에는 관여하지 않는다."""
        hangul = len(_HANGUL_RE.findall(text))
        ascii_letters = len(_ASCII_LETTER_RE.findall(text))
        other_letters = len(_OTHER_LETTER_RE.findall(text))
        total = hangul + ascii_letters + other_letters
        if total == 0:
            return "en"
        if hangul / total >= 0.3:
            return "ko"
        if ascii_letters / total >= 0.5:
            return "en"
        return "other"

    def _should_skip_pronunciation(self, text: str, source_lang: str) -> bool:
        """원문이 영어/한국어면 로마자/한글 발음표기가 무의미하므로 생략한다.
        번역 자체는 그대로 수행되고 pronunciation 필드만 비운다."""
        lang = source_lang
        if lang == "auto":
            lang = self._detect_lang_heuristic(text)
        return lang in ("en", "ko")

    def _parse_text_response(
        self, response: str, original_lines: list[str]
    ) -> list[TranslationLine]:
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)

        for prefix in ["TRANSLATION:", "Translation:", "번역:", "Here is", "Here's"]:
            if response.strip().startswith(prefix):
                response = response.strip()[len(prefix) :].strip()

        lines = [line.strip() for line in response.strip().split("\n") if line.strip()]

        results = []
        for i, trans in enumerate(lines):
            orig = original_lines[i] if i < len(original_lines) else ""
            results.append(TranslationLine(original=orig, translation=trans, pronunciation=None))

        return results


class GeminiTranslator(BaseTranslator):
    def __init__(self, settings: TranslationSettings | None = None):
        super().__init__(settings)
        self.api_key = self.settings.api_key or os.getenv("GEMINI_API_KEY")
        self.model = self.settings.model
        self.api_url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        )

    def translate(
        self,
        lyrics: list[LyricLine] | str,
        source_lang: str = "auto",
        target_lang: str | None = None,
        context: str | None = None,
    ) -> TranslationResult:
        target_lang = target_lang or self.settings.target_language

        if isinstance(lyrics, list):
            text = "\n".join(line.text for line in lyrics)
            original_lines = [line.text for line in lyrics]
        else:
            text = lyrics
            original_lines = [line.strip() for line in text.split("\n") if line.strip()]

        if not text.strip():
            return TranslationResult([], source_lang, target_lang, "gemini", self.settings.tone)

        if not self.api_key:
            return self._fallback_result(original_lines, source_lang, target_lang)

        include_pron = self.settings.include_pronunciation and not self._should_skip_pronunciation(
            text, source_lang
        )
        prompt = self._build_prompt(text, source_lang, target_lang, include_pron, context)

        try:
            response = requests.post(
                f"{self.api_url}?key={self.api_key}",
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": self.settings.temperature,
                        "maxOutputTokens": 8192,
                    },
                },
                timeout=self.settings.timeout,
            )

            if not response.ok:
                raise RuntimeError(f"API error: {response.status_code} - {response.text[:200]}")

            result = response.json()
            content = result["candidates"][0]["content"]["parts"][0]["text"]

            if include_pron:
                lines = self._parse_json_response(content, original_lines)
            else:
                lines = self._parse_text_response(content, original_lines)

            return TranslationResult(lines, source_lang, target_lang, "gemini", self.settings.tone)

        except requests.exceptions.ConnectionError:
            return self._fallback_result(original_lines, source_lang, target_lang)
        except Exception as e:
            raise RuntimeError(f"Translation failed: {e}") from e

    def _fallback_result(
        self, original_lines: list[str], source_lang: str, target_lang: str
    ) -> TranslationResult:
        # API 키가 없거나 연결이 안 되면 무료 웹 번역(deep-translator)으로 폴백.
        # 플레이스홀더 텍스트를 번역인 척 반환하면 클라이언트 UI에 그대로 노출되므로 금지 —
        # 여기서도 실패하면 예외를 올려 API가 5xx로 응답하게 한다(확장은 '번역 실패' 표시).
        from deep_translator import GoogleTranslator

        target = {"zh": "zh-CN"}.get(target_lang, target_lang)
        translator = GoogleTranslator(source="auto", target=target)

        translated = translator.translate("\n".join(original_lines)) or ""
        parts = [p.strip() for p in translated.split("\n")]
        if len(parts) != len(original_lines):
            # 웹 번역이 줄 수를 보존하지 못한 경우 — 줄 단위로 재시도 (느리지만 정확)
            parts = [(t or "").strip() for t in translator.translate_batch(original_lines)]

        lines = [
            TranslationLine(original=orig, translation=trans, pronunciation=None)
            for orig, trans in zip(original_lines, parts)
        ]
        return TranslationResult(lines, source_lang, target_lang, "google-web", self.settings.tone)


class OpenAICompatibleTranslator(BaseTranslator):
    def __init__(self, settings: TranslationSettings | None = None):
        super().__init__(settings)
        # 결과에 찍는 실제 백엔드 이름 — 자동 전환(gemini 설정→NIM) 시 settings.engine과 다르다
        self.engine_name = self.settings.engine
        self.api_key = self.settings.api_key or os.getenv("OPENAI_API_KEY") or "local-gen-ai"
        self.model = self.settings.model

        if self.settings.engine == "openai":
            self.api_url = "https://api.openai.com/v1/chat/completions"
        else:
            self.api_url = self.settings.api_url or "http://localhost:11434/v1/chat/completions"

    def translate(
        self,
        lyrics: list[LyricLine] | str,
        source_lang: str = "auto",
        target_lang: str | None = None,
        context: str | None = None,
    ) -> TranslationResult:
        target_lang = target_lang or self.settings.target_language

        if isinstance(lyrics, list):
            text = "\n".join(line.text for line in lyrics)
            original_lines = [line.text for line in lyrics]
        else:
            text = lyrics
            original_lines = [line.strip() for line in text.split("\n") if line.strip()]

        if not text.strip():
            return TranslationResult(
                [], source_lang, target_lang, self.engine_name, self.settings.tone
            )

        include_pron = self.settings.include_pronunciation and not self._should_skip_pronunciation(
            text, source_lang
        )
        prompt = self._build_prompt(text, source_lang, target_lang, include_pron, context)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
            "stream": False,
        }
        payload.update(self._payload_extras())

        try:
            response = requests.post(
                self.api_url,
                json=payload,
                headers=headers,
                timeout=self.settings.timeout,
            )

            if not response.ok:
                raise RuntimeError(f"API error: {response.status_code} - {response.text[:200]}")

            result = response.json()
            message = result["choices"][0]["message"]
            content = message.get("content") or ""
            if not content.strip():
                # reasoning 모델이 사고에 max_tokens를 소진하면 content가 비어서 온다
                raise RuntimeError(
                    "Empty completion content (model may have spent max_tokens on reasoning)"
                )

            if include_pron:
                lines = self._parse_json_response(content, original_lines)
            else:
                lines = self._parse_text_response(content, original_lines)

            return TranslationResult(
                lines, source_lang, target_lang, self.engine_name, self.settings.tone
            )

        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(f"Connection failed to {self.api_url}: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Translation failed: {e}") from e

    def _payload_extras(self) -> dict:
        """엔진별 추가 페이로드 훅 — 기본은 없음."""
        return {}


class NvidiaTranslator(OpenAICompatibleTranslator):
    """NVIDIA NIM (OpenAI 호환 /v1/chat/completions) 백엔드.

    키 해석 순서: settings.api_key -> env NVIDIA_API_KEY -> 루트 nvapi.txt 파일.
    모델은 gemini 기본값(settings.model)과 섞이지 않도록 settings.nvidia_model을 쓴다.
    """

    NIM_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
    _KEY_FILE = Path(__file__).resolve().parents[2] / "nvapi.txt"

    def __init__(self, settings: TranslationSettings | None = None):
        # OpenAICompatibleTranslator.__init__을 건너뛰고 BaseTranslator.__init__만 호출해
        # OPENAI_API_KEY/로컬 기본 URL 등 다른 엔진 전용 로직이 섞이지 않게 한다.
        BaseTranslator.__init__(self, settings)
        # settings.engine이 "gemini"여도(키 부재 자동 전환) 결과에는 실제 백엔드를 찍는다
        self.engine_name = "nvidia"
        self.api_key = (
            self.settings.api_key or os.getenv("NVIDIA_API_KEY") or self._read_key_file()
        )
        self.model = self.settings.nvidia_model
        self.api_url = self.settings.api_url or self.NIM_API_URL

    def _read_key_file(self) -> str | None:
        try:
            return self._KEY_FILE.read_text(encoding="utf-8").strip() or None
        except OSError:
            return None

    def _payload_extras(self) -> dict:
        # qwen3 계열은 reasoning 모델 — 사고 모드를 끄지 않으면 max_tokens를 사고에
        # 소진해 content가 비거나(빈 응답) 타임아웃이 난다. NIM qwen 챗 템플릿 스위치.
        if "qwen" in self.model.lower():
            return {"chat_template_kwargs": {"thinking": False}}
        return {}


class TranslatorFactory:
    @staticmethod
    def get_translator(settings: TranslationSettings | None = None) -> BaseTranslator:
        settings = settings or get_settings().translation

        if settings.engine == "gemini":
            # gemini 키가 없으면 웹 번역 폴백(번역만 가능, 발음표기 불가·기계번역 톤)으로
            # 조용히 격하된다 — NVIDIA 키(env 또는 루트 nvapi.txt)가 있으면 NIM으로 자동
            # 전환한다. env 없이 uvicorn만 띄운 서버에서 발음이 통째로 빠지는 사고 방지.
            if not (settings.api_key or os.getenv("GEMINI_API_KEY")):
                nvidia = NvidiaTranslator(settings)
                if nvidia.api_key:
                    logger.info(
                        "No Gemini API key; auto-switching translation engine to NVIDIA NIM"
                    )
                    return nvidia
            return GeminiTranslator(settings)
        elif settings.engine == "nvidia":
            return NvidiaTranslator(settings)
        elif settings.engine in ("openai", "local"):
            return OpenAICompatibleTranslator(settings)
        else:
            raise ValueError(f"Unknown translation engine: {settings.engine}")


class LyricsTranslator:
    def __init__(self, api_key: str | None = None, settings: TranslationSettings | None = None):
        if settings is None:
            settings = get_settings().translation
        if api_key:
            settings.api_key = api_key
        self._translator = TranslatorFactory.get_translator(settings)
        self.settings = settings

    def translate(
        self,
        lyrics: list[LyricLine] | str,
        source_lang: str = "auto",
        target_lang: str = "ko",
        context: str | None = None,
    ) -> str:
        result = self._translator.translate(lyrics, source_lang, target_lang, context)
        return "\n".join(line.translation for line in result.lines)

    def translate_with_pronunciation(
        self,
        lyrics: list[LyricLine] | str,
        source_lang: str = "auto",
        target_lang: str = "ko",
        context: str | None = None,
    ) -> TranslationResult:
        old_setting = self.settings.include_pronunciation
        self.settings.include_pronunciation = True
        try:
            return self._translator.translate(lyrics, source_lang, target_lang, context)
        finally:
            self.settings.include_pronunciation = old_setting
