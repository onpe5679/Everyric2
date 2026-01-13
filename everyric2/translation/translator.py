import os
import re
from pathlib import Path

import requests
from dotenv import load_dotenv

from everyric2.inference.prompt import LyricLine

load_dotenv()


class LyricsTranslator:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self.api_url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        )

    def translate(
        self,
        lyrics: list[LyricLine] | str,
        source_lang: str = "auto",
        target_lang: str = "ko",
    ) -> str:
        if isinstance(lyrics, list):
            text = "\n".join(line.text for line in lyrics)
        else:
            text = lyrics

        if not text.strip():
            return ""

        if not self.api_key:
            return self._fallback_translate(text)

        prompt = self._build_prompt(text, source_lang, target_lang)

        try:
            response = requests.post(
                f"{self.api_url}?key={self.api_key}",
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.3,
                        "maxOutputTokens": 4096,
                    },
                },
                timeout=120,
            )

            if not response.ok:
                return f"[Translation Error: {response.status_code} - {response.text[:200]}]"

            result = response.json()
            content = result["candidates"][0]["content"]["parts"][0]["text"]
            return self._clean_response(content, text)

        except requests.exceptions.ConnectionError:
            return self._fallback_translate(text)
        except Exception as e:
            return f"[Translation Error: {e}]"

    def _build_prompt(self, text: str, source_lang: str, target_lang: str) -> str:
        lang_names = {"ko": "Korean", "en": "English", "ja": "Japanese", "zh": "Chinese"}
        target = lang_names.get(target_lang, target_lang)

        return f"""Translate these song lyrics to {target}. 
Keep the same line structure (same number of lines). Only output the translation, no explanations or notes.

LYRICS:
{text}

TRANSLATION:"""

    def _clean_response(self, response: str, original: str) -> str:
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)

        for prefix in ["TRANSLATION:", "Translation:", "번역:", "Here is", "Here's"]:
            if response.strip().startswith(prefix):
                response = response.strip()[len(prefix) :].strip()

        lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
        return "\n".join(lines)

    def _fallback_translate(self, text: str) -> str:
        return f"[GEMINI_API_KEY not set - original lyrics]\n{text}"

    def translate_file(self, input_path: Path, output_path: Path | None = None) -> str:
        text = input_path.read_text(encoding="utf-8")
        translated = self.translate(text)

        if output_path:
            output_path.write_text(translated, encoding="utf-8")

        return translated
