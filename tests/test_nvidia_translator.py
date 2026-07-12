"""Tests for the NVIDIA NIM translation engine and the en/ko pronunciation gate."""

from dataclasses import dataclass

import pytest

from everyric2.config.settings import TranslationSettings
from everyric2.translation.translator import (
    BaseTranslator,
    GeminiTranslator,
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

    def test_gemini_engine_without_key_auto_switches_to_nvidia(self, monkeypatch, tmp_path):
        # gemini 키가 없으면 웹 폴백(발음 불가)으로 격하되는 대신 NIM 키가 있으면 NIM으로
        key_file = tmp_path / "nvapi.txt"
        key_file.write_text("nim-key", encoding="utf-8")
        monkeypatch.setattr(NvidiaTranslator, "_KEY_FILE", key_file)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

        translator = TranslatorFactory.get_translator(
            TranslationSettings(engine="gemini", api_key=None)
        )
        assert isinstance(translator, NvidiaTranslator)

    def test_gemini_engine_with_key_stays_gemini(self, monkeypatch, tmp_path):
        key_file = tmp_path / "nvapi.txt"
        key_file.write_text("nim-key", encoding="utf-8")
        monkeypatch.setattr(NvidiaTranslator, "_KEY_FILE", key_file)
        monkeypatch.setenv("GEMINI_API_KEY", "gm-key")

        translator = TranslatorFactory.get_translator(
            TranslationSettings(engine="gemini", api_key=None)
        )
        assert isinstance(translator, GeminiTranslator)

    def test_gemini_engine_without_any_key_keeps_gemini_web_fallback(self, monkeypatch, tmp_path):
        # NIM 키도 없으면 기존 동작(웹 번역 폴백) 유지 — 번역이라도 나가야 한다
        missing = tmp_path / "does_not_exist.txt"
        monkeypatch.setattr(NvidiaTranslator, "_KEY_FILE", missing)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

        translator = TranslatorFactory.get_translator(
            TranslationSettings(engine="gemini", api_key=None)
        )
        assert isinstance(translator, GeminiTranslator)


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


class TestPromptBuilding:
    """_build_prompt — ko 타깃은 한글 독음, 곡 컨텍스트 주입, 가사 맥락 지시."""

    def setup_method(self):
        class _Probe(BaseTranslator):
            def translate(self, *a, **k):  # pragma: no cover - not exercised
                raise NotImplementedError

        self.probe = _Probe(TranslationSettings())

    def test_ko_target_pron_asks_kana_reading_not_romanization(self):
        # 새 계약: LLM은 가나 독음만 쓰고(문맥 한자 읽기), 한글 변환은 서버가 한다
        # (kana_hangul — 촉음/ん/장음의 기계 전사 실수 원천 차단)
        prompt = self.probe._build_prompt("時計の針が", "ja", "ko", include_pronunciation=True)
        assert "kana reading" in prompt
        # 가나 예시가 있어야 LLM이 로마자/한글로 새지 않는다
        assert "とけいの はりが" in prompt
        assert "never Hangul" in prompt
        assert "Romanized pronunciation" not in prompt

    def test_non_ko_target_pron_stays_romanized(self):
        prompt = self.probe._build_prompt("時計の針が", "ja", "en", include_pronunciation=True)
        assert "Romanized pronunciation" in prompt
        assert "kana reading" not in prompt

    def test_song_context_is_injected(self):
        prompt = self.probe._build_prompt(
            "きみの声", "ja", "ko", include_pronunciation=False, context='"熱異常" by かいりきベア'
        )
        assert 'Song: "熱異常" by かいりきベア' in prompt

    def test_lyrics_guidance_present_in_both_paths(self):
        for pron in (True, False):
            prompt = self.probe._build_prompt("きみの声", "ja", "ko", include_pronunciation=pron)
            assert "ONE song" in prompt
