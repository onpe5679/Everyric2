"""Script-based language routing shared by the adaptive alignment stack.

This module intentionally has no model or legacy-engine dependency.  Its output selects an
internal anchor path; it is not a user-selectable engine setting.
"""

from __future__ import annotations

from collections import Counter

_LANGUAGE_SCRIPTS = {
    "ja": {"kana", "han", "latin"},
    "ko": {"hangul", "latin"},
    "zh": {"han", "latin"},
    "en": {"latin"},
}
_NATIVE_SCRIPT = {"ja": "kana", "ko": "hangul", "zh": "han", "en": "latin"}
_NATIVE_SHARE_FLOOR = 0.05


def _script(char: str) -> str | None:
    code = ord(char)
    if 0x3040 <= code <= 0x30FF or 0x31F0 <= code <= 0x31FF:
        return "kana"
    if 0xAC00 <= code <= 0xD7AF or 0x1100 <= code <= 0x11FF:
        return "hangul"
    if 0x3400 <= code <= 0x4DBF or 0x4E00 <= code <= 0x9FFF:
        return "han"
    if (0x41 <= code <= 0x5A) or (0x61 <= code <= 0x7A):
        return "latin"
    return None


def detect_language_from_text(text: str) -> tuple[str, bool]:
    """Return the dominant supported language and whether native scripts are mixed.

    Kana and Hangul are decisive.  Han without Kana routes to Chinese, and Latin-only text
    routes to English.  This keeps routing deterministic without loading a recognition model.
    """
    counts = Counter(script for char in text if (script := _script(char)) is not None)
    detected: list[str] = []
    if counts["kana"]:
        detected.append("ja")
    if counts["hangul"]:
        detected.append("ko")
    if counts["han"] and not counts["kana"]:
        detected.append("zh")
    if counts["latin"] > 10:
        detected.append("en")
    multilingual = len(detected) >= 2

    if multilingual:
        total = sum(counts.values()) or 1
        viable = [
            language
            for language in ("ja", "ko", "zh")
            if sum(
                amount
                for script, amount in counts.items()
                if script in _LANGUAGE_SCRIPTS[language] and script != "latin"
            )
            / total
            >= _NATIVE_SHARE_FLOOR
        ]
        if not viable:
            return "en", True
        primary = max(
            viable,
            key=lambda language: (
                sum(
                    amount
                    for script, amount in counts.items()
                    if script in _LANGUAGE_SCRIPTS[language]
                ),
                counts[_NATIVE_SCRIPT[language]],
            ),
        )
        return primary, True

    if counts["kana"]:
        return "ja", False
    if counts["hangul"]:
        return "ko", False
    if counts["han"]:
        return "zh", False
    return "en", False
