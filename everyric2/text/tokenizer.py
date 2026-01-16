import logging
import re
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

_mecab_tagger = None


@dataclass
class TokenInfo:
    surface: str
    pos: str | None = None
    reading: str | None = None
    is_content_word: bool = True


class Tokenizer:
    def __init__(self, language: str = "ja", use_mecab: bool = True):
        self.language = language
        self.use_mecab = use_mecab
        self._mecab_available: bool | None = None

    def tokenize(self, text: str) -> list[TokenInfo]:
        if self.language == "ja":
            return self._tokenize_japanese(text)
        elif self.language == "zh":
            return self._tokenize_chinese(text)
        elif self.language == "ko":
            return self._tokenize_korean(text)
        else:
            return self._tokenize_whitespace(text)

    def _tokenize_japanese(self, text: str) -> list[TokenInfo]:
        if self.use_mecab and self._is_mecab_available():
            return self._tokenize_japanese_mecab(text)
        return self._tokenize_characters(text)

    def _is_mecab_available(self) -> bool:
        if self._mecab_available is not None:
            return self._mecab_available

        global _mecab_tagger
        try:
            import fugashi

            _mecab_tagger = fugashi.Tagger()
            self._mecab_available = True
            logger.info("MeCab/fugashi available for Japanese tokenization")
        except ImportError:
            logger.warning("fugashi not installed. Install with: pip install fugashi[unidic-lite]")
            self._mecab_available = False
        except Exception as e:
            logger.warning(f"MeCab initialization failed: {e}")
            self._mecab_available = False

        return self._mecab_available

    def _tokenize_japanese_mecab(self, text: str) -> list[TokenInfo]:
        global _mecab_tagger
        if _mecab_tagger is None:
            return self._tokenize_characters(text)

        tokens = []
        for word in _mecab_tagger(text):
            surface = word.surface
            if not surface.strip():
                continue

            pos = word.feature.pos1 if hasattr(word.feature, "pos1") else None
            reading = word.feature.kana if hasattr(word.feature, "kana") else None

            is_content = pos not in ("助詞", "助動詞", "記号") if pos else True

            tokens.append(
                TokenInfo(
                    surface=surface,
                    pos=pos,
                    reading=reading,
                    is_content_word=is_content,
                )
            )

        return tokens

    def _tokenize_chinese(self, text: str) -> list[TokenInfo]:
        try:
            import jieba

            words = jieba.cut(text)
            return [TokenInfo(surface=w) for w in words if w.strip()]
        except ImportError:
            logger.warning("jieba not installed for Chinese. Using character-level.")
            return self._tokenize_characters(text)

    def _tokenize_korean(self, text: str) -> list[TokenInfo]:
        try:
            from konlpy.tag import Okt

            okt = Okt()
            morphs = okt.morphs(text)
            return [TokenInfo(surface=m) for m in morphs if m.strip()]
        except ImportError:
            logger.warning("konlpy not installed for Korean. Using whitespace.")
            return self._tokenize_whitespace(text)

    def _tokenize_whitespace(self, text: str) -> list[TokenInfo]:
        words = text.split()
        return [TokenInfo(surface=w) for w in words if w.strip()]

    def _tokenize_characters(self, text: str) -> list[TokenInfo]:
        chars = list(text.replace(" ", ""))
        return [TokenInfo(surface=c) for c in chars if c.strip()]

    def split_into_words(self, text: str) -> list[str]:
        tokens = self.tokenize(text)
        return [t.surface for t in tokens]

    def split_into_characters(self, text: str) -> list[str]:
        return [c for c in text if c.strip()]


def get_tokenizer(language: str, use_mecab: bool = True) -> Tokenizer:
    return Tokenizer(language=language, use_mecab=use_mecab)
