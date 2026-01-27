import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

import requests
from dotenv import load_dotenv

from everyric2.config.settings import TranslationSettings, get_settings
from everyric2.inference.prompt import LyricLine

load_dotenv()


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
    ) -> TranslationResult:
        pass

    def _build_prompt(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        include_pronunciation: bool,
    ) -> str:
        lang_names = {"ko": "Korean", "en": "English", "ja": "Japanese", "zh": "Chinese"}
        target = lang_names.get(target_lang, target_lang)
        tone_instruction = TONE_PROMPTS.get(self.settings.tone, TONE_PROMPTS["natural"])

        if include_pronunciation:
            return f"""Translate these song lyrics to {target}.
{tone_instruction}

For each line, provide:
1. The translation
2. Romanized pronunciation of the ORIGINAL text (not the translation)

Output as JSON array:
[{{"original": "原文", "translation": "번역", "pronunciation": "genbun"}}]

IMPORTANT:
- Keep the same number of lines
- Output ONLY the JSON array, no explanations
- pronunciation should be romanization of the ORIGINAL lyrics

LYRICS:
{text}"""
        else:
            return f"""Translate these song lyrics to {target}.
{tone_instruction}

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

        prompt = self._build_prompt(
            text, source_lang, target_lang, self.settings.include_pronunciation
        )

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

            if self.settings.include_pronunciation:
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
        lines = [
            TranslationLine(original=line, translation=f"[NO API KEY] {line}", pronunciation=None)
            for line in original_lines
        ]
        return TranslationResult(lines, source_lang, target_lang, "gemini", self.settings.tone)


class OpenAICompatibleTranslator(BaseTranslator):
    def __init__(self, settings: TranslationSettings | None = None):
        super().__init__(settings)
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
                [], source_lang, target_lang, self.settings.engine, self.settings.tone
            )

        prompt = self._build_prompt(
            text, source_lang, target_lang, self.settings.include_pronunciation
        )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.settings.temperature,
            "stream": False,
        }

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
            content = result["choices"][0]["message"]["content"]

            if self.settings.include_pronunciation:
                lines = self._parse_json_response(content, original_lines)
            else:
                lines = self._parse_text_response(content, original_lines)

            return TranslationResult(
                lines, source_lang, target_lang, self.settings.engine, self.settings.tone
            )

        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(f"Connection failed to {self.api_url}: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Translation failed: {e}") from e


class TranslatorFactory:
    @staticmethod
    def get_translator(settings: TranslationSettings | None = None) -> BaseTranslator:
        settings = settings or get_settings().translation

        if settings.engine == "gemini":
            return GeminiTranslator(settings)
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
    ) -> str:
        result = self._translator.translate(lyrics, source_lang, target_lang)
        return "\n".join(line.translation for line in result.lines)

    def translate_with_pronunciation(
        self,
        lyrics: list[LyricLine] | str,
        source_lang: str = "auto",
        target_lang: str = "ko",
    ) -> TranslationResult:
        old_setting = self.settings.include_pronunciation
        self.settings.include_pronunciation = True
        try:
            return self._translator.translate(lyrics, source_lang, target_lang)
        finally:
            self.settings.include_pronunciation = old_setting
