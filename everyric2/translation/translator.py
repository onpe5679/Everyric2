"""Lyrics translation using LLM."""

import re
from pathlib import Path

import requests

from everyric2.inference.prompt import LyricLine


class LyricsTranslator:
    """Translates lyrics to Korean using local LLM."""

    def __init__(self, api_url: str = "http://localhost:8081/v1/chat/completions"):
        self.api_url = api_url

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

        prompt = self._build_prompt(text, source_lang, target_lang)

        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": "qwen3-omni",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 4096,
                    "temperature": 0.3,
                },
                timeout=120,
            )

            if not response.ok:
                return f"[Translation Error: {response.status_code}]"

            result = response.json()["choices"][0]["message"]["content"]
            return self._clean_response(result, text)

        except requests.exceptions.ConnectionError:
            return self._fallback_translate(text)
        except Exception as e:
            return f"[Translation Error: {e}]"

    def _build_prompt(self, text: str, source_lang: str, target_lang: str) -> str:
        lang_names = {"ko": "Korean", "en": "English", "ja": "Japanese", "zh": "Chinese"}
        target = lang_names.get(target_lang, target_lang)

        return f"""Translate these song lyrics to {target}. 
Keep the same line structure. Only output the translation, no explanations.

LYRICS:
{text}

TRANSLATION:"""

    def _clean_response(self, response: str, original: str) -> str:
        # Remove thinking tags if present
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)

        # Remove common prefixes
        for prefix in ["TRANSLATION:", "Translation:", "번역:", "Here is", "Here's"]:
            if response.strip().startswith(prefix):
                response = response.strip()[len(prefix) :].strip()

        lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
        original_lines = [l.strip() for l in original.strip().split("\n") if l.strip()]

        # If line counts match, return as-is
        if len(lines) == len(original_lines):
            return "\n".join(lines)

        return "\n".join(lines)

    def _fallback_translate(self, text: str) -> str:
        """Fallback when LLM is not available - return original with note."""
        return f"[LLM not available - original lyrics]\n{text}"

    def translate_file(self, input_path: Path, output_path: Path | None = None) -> str:
        text = input_path.read_text(encoding="utf-8")
        translated = self.translate(text)

        if output_path:
            output_path.write_text(translated, encoding="utf-8")

        return translated
