"""Tests for the NVIDIA NIM translation engine and the en/ko pronunciation gate."""

from dataclasses import dataclass

import pytest

from everyric2.config.settings import TranslationSettings
from everyric2.translation.translator import (
    BaseTranslator,
    NvidiaTranslator,
    OpenAICompatibleTranslator,
    TranslatorFactory,
)


@dataclass
class FakeResponse:
    status_code: int
    _payload: dict
    ok: bool = True
    text: str = ""

    def json(self):
        return self._payload


def chat_response(content: str) -> FakeResponse:
    return FakeResponse(
        status_code=200,
        _payload={"choices": [{"message": {"content": content}}]},
    )


class TestTranslatorFactory:
    def test_nvidia_engine_returns_nvidia_translator(self, monkeypatch):
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        settings = TranslationSettings(engine="nvidia", api_key="dummy-key")
        translator = TranslatorFactory.get_translator(settings)

        assert isinstance(translator, NvidiaTranslator)
        assert isinstance(translator, OpenAICompatibleTranslator)
        assert translator.api_url == NvidiaTranslator.NIM_API_URL

    def test_nvidia_uses_nvidia_model_field_not_generic_model(self, monkeypatch):
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        settings = TranslationSettings(
            engine="nvidia", api_key="dummy-key", model="gemini-2.0-flash"
        )
        translator = TranslatorFactory.get_translator(settings)

        assert translator.model == settings.nvidia_model
        assert translator.model != "gemini-2.0-flash"


class TestApiKeyResolutionOrder:
    def test_settings_api_key_wins(self, monkeypatch, tmp_path):
        key_file = tmp_path / "nvapi.txt"
        key_file.write_text("file-key\n", encoding="utf-8")
        monkeypatch.setattr(NvidiaTranslator, "_KEY_FILE", key_file)
        monkeypatch.setenv("NVIDIA_API_KEY", "env-key")

        settings = TranslationSettings(engine="nvidia", api_key="settings-key")
        translator = NvidiaTranslator(settings)

        assert translator.api_key == "settings-key"

    def test_env_var_wins_over_key_file(self, monkeypatch, tmp_path):
        key_file = tmp_path / "nvapi.txt"
        key_file.write_text("file-key\n", encoding="utf-8")
        monkeypatch.setattr(NvidiaTranslator, "_KEY_FILE", key_file)
        monkeypatch.setenv("NVIDIA_API_KEY", "env-key")

        settings = TranslationSettings(engine="nvidia", api_key=None)
        translator = NvidiaTranslator(settings)

        assert translator.api_key == "env-key"

    def test_falls_back_to_key_file(self, monkeypatch, tmp_path):
        key_file = tmp_path / "nvapi.txt"
        key_file.write_text("  file-key-with-whitespace  \n", encoding="utf-8")
        monkeypatch.setattr(NvidiaTranslator, "_KEY_FILE", key_file)
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

        settings = TranslationSettings(engine="nvidia", api_key=None)
        translator = NvidiaTranslator(settings)

        assert translator.api_key == "file-key-with-whitespace"

    def test_missing_key_file_yields_none(self, monkeypatch, tmp_path):
        missing_file = tmp_path / "does_not_exist.txt"
        monkeypatch.setattr(NvidiaTranslator, "_KEY_FILE", missing_file)
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

        settings = TranslationSettings(engine="nvidia", api_key=None)
        translator = NvidiaTranslator(settings)

        assert translator.api_key is None


class TestPronunciationGateHeuristic:
    """BaseTranslator._should_skip_pronunciation / _detect_lang_heuristic."""

    def setup_method(self):
        class _Probe(BaseTranslator):
            def translate(self, *a, **k):  # pragma: no cover - not exercised
                raise NotImplementedError

        self.probe = _Probe(TranslationSettings())

    @pytest.mark.parametrize(
        "source_lang,text,expected",
        [
            ("en", "I can hear your voice", True),
            ("ko", "오늘 밤 너의 목소리", True),
            ("ja", "きみの声が聴こえる", False),
            ("zh", "我听到你的声音", False),
        ],
    )
    def test_explicit_source_lang(self, source_lang, text, expected):
        assert self.probe._should_skip_pronunciation(text, source_lang) is expected

    def test_auto_detects_english(self):
        text = "Walking down an empty street tonight"
        assert self.probe._detect_lang_heuristic(text) == "en"
        assert self.probe._should_skip_pronunciation(text, "auto") is True

    def test_auto_detects_korean(self):
        text = "오늘 밤 너의 목소리가 들려"
        assert self.probe._detect_lang_heuristic(text) == "ko"
        assert self.probe._should_skip_pronunciation(text, "auto") is True

    def test_auto_detects_japanese_as_other(self):
        text = "夜の街に消えていく光"
        assert self.probe._detect_lang_heuristic(text) == "other"
        assert self.probe._should_skip_pronunciation(text, "auto") is False


class TestTranslateAppliesGate:
    """The gate must apply inside translate(), overriding settings.include_pronunciation,
    and the request payload must always include max_tokens (NIM truncates long
    pronunciation JSON without it)."""

    def _make_translator(self, monkeypatch, tmp_path, include_pronunciation=True):
        key_file = tmp_path / "nvapi.txt"
        key_file.write_text("dummy-key", encoding="utf-8")
        monkeypatch.setattr(NvidiaTranslator, "_KEY_FILE", key_file)
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        settings = TranslationSettings(engine="nvidia", api_key=None)
        settings.include_pronunciation = include_pronunciation
        return NvidiaTranslator(settings)

    def test_english_source_skips_pronunciation_even_if_requested(self, monkeypatch, tmp_path):
        translator = self._make_translator(monkeypatch, tmp_path, include_pronunciation=True)

        captured = {}

        def fake_post(url, json, headers, timeout):
            captured["json"] = json
            return chat_response("Hello there\nGood morning")

        monkeypatch.setattr(
            "everyric2.translation.translator.requests.post", fake_post
        )

        result = translator.translate(
            "안녕하세요\n좋은 아침입니다",
            source_lang="ko",
            target_lang="en",
        )

        assert all(line.pronunciation is None for line in result.lines)
        assert captured["json"]["max_tokens"] == translator.settings.max_tokens
        # plain-text prompt path was used, not the JSON pronunciation format
        assert "pronunciation" not in captured["json"]["messages"][0]["content"].lower() or (
            "romanized" not in captured["json"]["messages"][0]["content"].lower()
        )

    def test_japanese_source_keeps_pronunciation_and_parses_json(self, monkeypatch, tmp_path):
        translator = self._make_translator(monkeypatch, tmp_path, include_pronunciation=True)

        def fake_post(url, json, headers, timeout):
            content = (
                '[{"original": "おはよう", '
                '"translation": "안녕", '
                '"pronunciation": "Ohayou"}]'
            )
            return chat_response(content)

        monkeypatch.setattr(
            "everyric2.translation.translator.requests.post", fake_post
        )

        result = translator.translate("おはよう", source_lang="ja", target_lang="ko")

        assert len(result.lines) == 1
        assert result.lines[0].pronunciation == "Ohayou"
        assert result.lines[0].translation == "안녕"
        assert result.engine == "nvidia"
