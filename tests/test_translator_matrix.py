"""번역 모델과 결정론 발음 렌더러의 책임 분리 회귀 테스트.

모델은 모든 언어 조합에서 번역만 반환한다. ``include_pronunciation``은 명시적으로
지원되는 ja/ko 결정론 셀을 켜는 요청 토글이며, 지원되지 않거나 auto에서 해석할 수 없는
셀은 모델 응답에 pronunciation이 섞여 있어도 항상 None이다.
"""

from json import dumps

import pytest

from everyric2.config.settings import TranslationSettings
from everyric2.text.ko_reading import hangul_to_kana, hangul_to_romaja
from everyric2.text.pron_style import romaji_line, wiki_pronunciation
from everyric2.translation.translator import BaseTranslator, NvidiaTranslator


class _Probe(BaseTranslator):
    def translate(self, *args, **kwargs):  # pragma: no cover - not exercised
        raise NotImplementedError


@pytest.fixture
def probe() -> _Probe:
    return _Probe(TranslationSettings())


class TestTranslationDiagonalEndToEnd:
    """원문과 대상 언어가 같으면 기존처럼 모델 호출 자체를 생략한다."""

    KO_SONG = "저기요 제가요 가슴이 떨려서\n말도 제대로 못 하고 서 있어요"
    JA_SONG = "きみの声が聴こえる\n夜の街に消えていく光"

    @staticmethod
    def _translator(monkeypatch, tmp_path) -> NvidiaTranslator:
        key_file = tmp_path / "nvapi.txt"
        key_file.write_text("dummy-key", encoding="utf-8")
        monkeypatch.setattr(NvidiaTranslator, "_KEY_FILE", key_file)
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        settings = TranslationSettings(engine="nvidia", api_key=None)
        settings.include_pronunciation = True
        return NvidiaTranslator(settings)

    @pytest.mark.parametrize(
        ("text", "source_lang", "target_lang"),
        [
            (KO_SONG, "ko", "ko"),
            (JA_SONG, "ja", "ja"),
        ],
    )
    def test_diagonal_skips_without_calling_model(
        self, monkeypatch, tmp_path, text, source_lang, target_lang
    ):
        translator = self._translator(monkeypatch, tmp_path)

        def boom(*args, **kwargs):  # pragma: no cover - 호출되면 테스트 실패
            raise AssertionError("model must not be called for a translation diagonal")

        monkeypatch.setattr("everyric2.translation.translator.requests.post", boom)
        result = translator.translate(text, source_lang=source_lang, target_lang=target_lang)

        assert result.translation_skipped is True
        assert all(line.translation == "" for line in result.lines)
        assert all(line.pronunciation is None for line in result.lines)


class TestDeterministicPronunciationMatrix:
    """지원되는 명시 ja/ko 셀만 원문 기반 결정론 발음을 만든다."""

    @staticmethod
    def _nvidia(monkeypatch, tmp_path, *, include_pronunciation=True) -> NvidiaTranslator:
        key_file = tmp_path / "nvapi.txt"
        key_file.write_text("dummy-key", encoding="utf-8")
        monkeypatch.setattr(NvidiaTranslator, "_KEY_FILE", key_file)
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        settings = TranslationSettings(engine="nvidia", api_key=None)
        settings.include_pronunciation = include_pronunciation
        return NvidiaTranslator(settings)

    @staticmethod
    def _fake_post(monkeypatch, content: str) -> list[dict]:
        from tests.test_nvidia_translator import chat_response

        calls: list[dict] = []

        def post(url, json, headers, timeout):
            calls.append(json)
            return chat_response(content)

        monkeypatch.setattr("everyric2.translation.translator.requests.post", post)
        return calls

    @staticmethod
    def _model_response(text: str, translation: str, pronunciation: str) -> str:
        return dumps(
            [
                {
                    "original": text,
                    "translation": translation,
                    "pronunciation": pronunciation,
                }
            ],
            ensure_ascii=False,
        )

    @pytest.mark.parametrize(
        ("text", "source_lang", "target_lang", "translation", "expected_factory"),
        [
            (
                "君は王女",
                "ja",
                "ko",
                "너는 왕녀",
                wiki_pronunciation,
            ),
            (
                "君は王女",
                "ja",
                "en",
                "you are the princess",
                lambda text: romaji_line(text)[0],
            ),
            (
                "사랑해",
                "ko",
                "ja",
                "愛してる",
                hangul_to_kana,
            ),
            (
                "사랑해",
                "ko",
                "en",
                "I love you",
                hangul_to_romaja,
            ),
            (
                "사랑해",
                "auto",
                "en",
                "I love you",
                hangul_to_romaja,
            ),
            (
                "사랑해",
                "auto",
                "ja",
                "愛してる",
                hangul_to_kana,
            ),
        ],
    )
    def test_supported_cell_uses_only_deterministic_renderer(
        self,
        monkeypatch,
        tmp_path,
        text,
        source_lang,
        target_lang,
        translation,
        expected_factory,
    ):
        translator = self._nvidia(monkeypatch, tmp_path)
        calls = self._fake_post(
            monkeypatch,
            self._model_response(text, translation, "model value"),
        )

        result = translator.translate(text, source_lang=source_lang, target_lang=target_lang)

        assert result.lines[0].translation == translation
        assert result.lines[0].pronunciation == expected_factory(text)
        assert result.lines[0].pronunciation != "model value"
        assert "pronunciation" not in calls[0]["messages"][0]["content"].lower()

    def test_auto_japanese_with_kana_remains_resolvable(self, monkeypatch, tmp_path):
        translator = self._nvidia(monkeypatch, tmp_path)
        calls = self._fake_post(
            monkeypatch,
            '[{"original":"きみの声","translation":"너의 목소리",'
            '"pronunciation":"model value"}]',
        )

        result = translator.translate("きみの声", source_lang="auto", target_lang="ko")

        assert result.lines[0].pronunciation == wiki_pronunciation("きみの声")
        assert "pronunciation" not in calls[0]["messages"][0]["content"].lower()

    @pytest.mark.parametrize(
        ("text", "source_lang", "target_lang", "translation"),
        [
            ("我听到你的声音", "zh", "ko", "네 목소리가 들려"),
            ("我爱你かな", "zh", "ko", "사랑해"),
            ("I love you", "en", "ja", "愛してる"),
            ("我爱你", "auto", "ko", "사랑해"),
        ],
    )
    def test_unsupported_or_auto_unresolved_cell_discards_model_pronunciation(
        self, monkeypatch, tmp_path, text, source_lang, target_lang, translation
    ):
        translator = self._nvidia(monkeypatch, tmp_path)
        calls = self._fake_post(
            monkeypatch,
            self._model_response(text, translation, "must disappear"),
        )

        result = translator.translate(text, source_lang=source_lang, target_lang=target_lang)

        assert result.lines[0].translation == translation
        assert result.lines[0].pronunciation is None
        assert "pronunciation" not in calls[0]["messages"][0]["content"].lower()

    def test_missing_morphological_analyzer_returns_none_instead_of_model_pronunciation(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(
            "everyric2.translation.translator.reading_source", lambda: "pykakasi"
        )
        translator = self._nvidia(monkeypatch, tmp_path)
        calls = self._fake_post(
            monkeypatch,
            '[{"original":"君は王女","translation":"you are the princess",'
            '"pronunciation":"kimi wa oujo"}]',
        )

        result = translator.translate("君は王女", source_lang="ja", target_lang="en")

        assert result.lines[0].translation == "you are the princess"
        assert result.lines[0].pronunciation is None
        assert "pronunciation" not in calls[0]["messages"][0]["content"].lower()

    def test_include_false_discards_unexpected_model_field(self, monkeypatch, tmp_path):
        translator = self._nvidia(monkeypatch, tmp_path, include_pronunciation=False)
        calls = self._fake_post(
            monkeypatch,
            '[{"original":"君は王女","translation":"너는 왕녀",'
            '"pronunciation":"unexpected"}]',
        )

        result = translator.translate("君は王女", source_lang="ja", target_lang="ko")

        assert result.lines[0].translation == "너는 왕녀"
        assert result.lines[0].pronunciation is None
        assert "pronunciation" not in calls[0]["messages"][0]["content"].lower()


def test_prompt_is_translation_only_for_every_target(probe):
    for target in ("ko", "en", "ja", "zh"):
        prompt = probe._build_prompt("時計の針が", target, context='"song" by artist')
        assert "pronunciation" not in prompt.lower()
        assert "romanized" not in prompt.lower()
        assert "kana reading" not in prompt.lower()
        assert 'Song: "song" by artist' in prompt
