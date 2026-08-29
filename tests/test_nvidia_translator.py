"""Tests for the NVIDIA NIM translation engine and the en/ko pronunciation gate."""

from dataclasses import dataclass
from json import dumps

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


def chat_response(content: str, finish_reason: str | None = None) -> FakeResponse:
    return FakeResponse(
        status_code=200,
        _payload={
            "choices": [{"message": {"content": content}, "finish_reason": finish_reason}]
        },
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


class TestPayloadExtras:
    """reasoning 모델별 추가 페이로드 — qwen은 thinking off, gpt-oss는 effort low.
    안 보내면 사고가 max_tokens 예산을 소진해 빈 응답/잘린 JSON이 난다."""

    def _make(self, model: str) -> NvidiaTranslator:
        settings = TranslationSettings(
            engine="nvidia", api_key="dummy-key", nvidia_model=model
        )
        return NvidiaTranslator(settings)

    def test_qwen_disables_thinking(self):
        extras = self._make("qwen/qwen3-next-80b-a3b-instruct")._payload_extras()
        assert extras == {"chat_template_kwargs": {"thinking": False}}

    def test_gpt_oss_uses_low_reasoning_effort(self):
        extras = self._make("openai/gpt-oss-120b")._payload_extras()
        assert extras == {"reasoning_effort": "low"}

    def test_default_model_is_covered_by_extras(self):
        # 기본 모델이 reasoning 계열로 바뀌면 extras 분기도 함께 따라와야 한다
        settings = TranslationSettings(engine="nvidia", api_key="dummy-key")
        extras = NvidiaTranslator(settings)._payload_extras()
        assert extras == {"reasoning_effort": "low"}

    def test_other_models_send_no_extras(self):
        assert self._make("mistralai/mistral-large")._payload_extras() == {}


class TestTranslateAppliesGate:
    """모델은 번역만 받고, include_pronunciation은 결정론 렌더러만 제어한다."""

    def _make_translator(self, monkeypatch, tmp_path, include_pronunciation=True):
        key_file = tmp_path / "nvapi.txt"
        key_file.write_text("dummy-key", encoding="utf-8")
        monkeypatch.setattr(NvidiaTranslator, "_KEY_FILE", key_file)
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        settings = TranslationSettings(engine="nvidia", api_key=None)
        settings.include_pronunciation = include_pronunciation
        return NvidiaTranslator(settings)

    def test_english_source_skips_pronunciation_even_if_requested(self, monkeypatch, tmp_path):
        # en→ko는 지원되는 결정론 셀이 아니므로 요청 토글이 켜져 있어도 발음은 없다.
        translator = self._make_translator(monkeypatch, tmp_path, include_pronunciation=True)

        captured = {}

        def fake_post(url, json, headers, timeout):
            captured["json"] = json
            return chat_response("안녕하세요\n좋은 아침입니다")

        monkeypatch.setattr(
            "everyric2.translation.translator.requests.post", fake_post
        )

        result = translator.translate(
            "Hello there\nGood morning",
            source_lang="en",
            target_lang="ko",
        )

        assert all(line.pronunciation is None for line in result.lines)
        assert captured["json"]["max_tokens"] == translator.settings.max_tokens
        assert "pronunciation" not in captured["json"]["messages"][0]["content"].lower()

    def test_japanese_source_gets_deterministic_pronunciation(self, monkeypatch, tmp_path):
        # 새 계약: 일본어 곡의 한글 독음은 서버가 규칙으로 만든다(text.pron_style) —
        # LLM이 발음을 돌려줘도 무시하고, 프롬프트에도 발음을 요구하지 않는다
        translator = self._make_translator(monkeypatch, tmp_path, include_pronunciation=True)

        captured = {}

        def fake_post(url, json, headers, timeout):
            captured["json"] = json
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
        assert result.lines[0].pronunciation == "오하요오"
        assert result.lines[0].translation == "안녕"
        assert result.engine == "nvidia"
        # 발음을 묻지 않으므로 프롬프트는 번역 전용(참조 읽기 블록도 없다)
        prompt = captured["json"]["messages"][0]["content"]
        assert "pronunciation" not in prompt
        assert "REFERENCE READINGS" not in prompt


class TestTruncatedJsonRecovery:
    """NIM 발음 JSON이 max_tokens에서 잘렸을 때의 복구/재분할/폴백.

    ① 잘린 JSON에서 완전한 앞 객체까지 살려낸다
    ② 못 받은 나머지 라인만 재요청한다(진전 없으면 절반 분할)
    ③ 그래도 실패한 라인은 원문만 담고 failed=True로 반환 — 전체 500을 막는다
    실제 NIM API는 호출하지 않는다(requests.post를 mock).
    """

    def _make_translator(self, monkeypatch, tmp_path):
        key_file = tmp_path / "nvapi.txt"
        key_file.write_text("dummy-key", encoding="utf-8")
        monkeypatch.setattr(NvidiaTranslator, "_KEY_FILE", key_file)
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        settings = TranslationSettings(engine="nvidia", api_key=None)
        settings.include_pronunciation = True
        return NvidiaTranslator(settings)

    def _sequence_post(self, monkeypatch, responses):
        """호출 순서대로 미리 정한 FakeResponse를 돌려주는 requests.post 대체.
        보낸 각 요청의 payload를 calls에 기록한다."""
        calls = []
        it = iter(responses)

        def fake_post(url, json, headers, timeout):
            calls.append(json)
            return next(it)

        monkeypatch.setattr("everyric2.translation.translator.requests.post", fake_post)
        return calls

    def _obj(self, orig, trans, pron):
        return f'{{"original":"{orig}","translation":"{trans}","pronunciation":"{pron}"}}'

    def test_salvages_complete_prefix_then_requests_remainder(self, monkeypatch, tmp_path):
        translator = self._make_translator(monkeypatch, tmp_path)

        # 1차: 3줄 요청인데 2번째 객체까지만 완성되고 3번째에서 끊김(length)
        truncated = (
            "["
            + self._obj("ライン1", "번역1", "ぷろんいち")
            + ","
            + self._obj("ライン2", "번역2", "ぷろんに")
            + ',{"original":"ライン3","translation":"번역'  # 잘림
        )
        # 2차(나머지 1줄): 정상 완결
        remainder = "[" + self._obj("ライン3", "번역3", "ぷろんさん") + "]"

        calls = self._sequence_post(
            monkeypatch,
            [chat_response(truncated, finish_reason="length"), chat_response(remainder, "stop")],
        )

        result = translator.translate("ライン1\nライン2\nライン3", source_lang="ja", target_lang="ko")

        assert len(result.lines) == 3
        assert [l.translation for l in result.lines] == ["번역1", "번역2", "번역3"]
        # 발음은 응답이 아니라 원문에서 결정론적으로 만든다 — 응답의 가나는 버려진다
        assert [l.pronunciation for l in result.lines] == ["라인 1", "라인 2", "라인 3"]
        assert all(not l.failed for l in result.lines)
        # 나머지 재요청은 3번째 라인만 담았어야 한다(전체 재요청 아님)
        assert len(calls) == 2
        assert "ライン3" in calls[1]["messages"][0]["content"]
        assert "ライン1" not in calls[1]["messages"][0]["content"]

    def test_splits_in_half_when_no_object_salvageable(self, monkeypatch, tmp_path):
        translator = self._make_translator(monkeypatch, tmp_path)

        # 1차: 첫 객체조차 완성 못한 채 끊김 → 살릴 객체 0 → 절반 분할
        truncated = '[{"original":"アア","translation":"번'
        left = "[" + self._obj("アア", "왼1", "ひだりいち") + "," + self._obj("イイ", "왼2", "ひだりに") + "]"
        right = "[" + self._obj("ウウ", "오1", "みぎいち") + "," + self._obj("エエ", "오2", "みぎに") + "]"

        calls = self._sequence_post(
            monkeypatch,
            [
                chat_response(truncated, finish_reason="length"),
                chat_response(left, "stop"),
                chat_response(right, "stop"),
            ],
        )

        result = translator.translate(
            "アア\nイイ\nウウ\nエエ", source_lang="ja", target_lang="ko"
        )

        assert len(result.lines) == 4
        assert [l.translation for l in result.lines] == ["왼1", "왼2", "오1", "오2"]
        assert all(not l.failed for l in result.lines)
        assert len(calls) == 3  # 원본 1 + 좌/우 2

    def test_unrecoverable_line_falls_back_to_original_only(self, monkeypatch, tmp_path):
        translator = self._make_translator(monkeypatch, tmp_path)

        # 1차: 1번째만 완성, 2번째 잘림 → 나머지(2번째) 재요청
        truncated = "[" + self._obj("サキ", "성공1", "せいこう") + ',{"original":"ダメ","trans'
        # 2차(2번째 라인 단독): 또 잘려서 살릴 객체 0 → 단일 라인이라 폴백
        still_bad = '[{"original":"ダメ","transl'

        calls = self._sequence_post(
            monkeypatch,
            [
                chat_response(truncated, finish_reason="length"),
                chat_response(still_bad, finish_reason="length"),
            ],
        )

        result = translator.translate("サキ\nダメ", source_lang="ja", target_lang="ko")

        # 전체 500이 아니라 부분 성공으로 마감 — 줄 수는 보존
        assert len(result.lines) == 2
        assert result.lines[0].translation == "성공1"
        assert result.lines[0].failed is False
        # 복구 불가 라인: 번역은 비고 failed 표시. 발음은 LLM 응답과 무관하게 만들므로
        # 번역이 실패해도 남는다(사용자에겐 독음이라도 보이는 쪽이 낫다)
        assert result.lines[1].original == "ダメ"
        assert result.lines[1].translation == ""
        assert result.lines[1].pronunciation == "다메"
        assert result.lines[1].failed is True

    def test_empty_completion_does_not_500_and_marks_failed(self, monkeypatch, tmp_path):
        # reasoning이 max_tokens를 다 써 content가 빈 응답 — 재시도 후에도 비면
        # 단일 라인은 폴백(failed)로, 전체 예외는 나지 않는다
        translator = self._make_translator(monkeypatch, tmp_path)
        self._sequence_post(
            monkeypatch,
            [chat_response("", finish_reason="length"), chat_response("", "length")],
        )

        result = translator.translate("ヒトリ", source_lang="ja", target_lang="ko")

        assert len(result.lines) == 1
        assert result.lines[0].original == "ヒトリ"
        assert result.lines[0].failed is True

    def test_long_input_is_batched_up_front(self, monkeypatch, tmp_path):
        # 라인 수가 임계를 넘으면 처음부터 배치로 나눠 요청(잘림 예방).
        # 모델 출력은 번역 전용 한 경로뿐이므로 _TEXT_BATCH_* 임계만 내린다.
        import everyric2.translation.translator as tr

        monkeypatch.setattr(tr, "_TEXT_BATCH_THRESHOLD", 2)
        monkeypatch.setattr(tr, "_TEXT_BATCH_SIZE", 2)
        translator = self._make_translator(monkeypatch, tmp_path)

        def fake_post(url, json, headers, timeout):
            # 요청에 담긴 원문 라인들에 맞춰 정상 JSON을 만들어 돌려준다
            content = json["messages"][0]["content"]
            lines = content.split("LYRICS:\n")[-1].strip().split("\n")
            arr = ",".join(self._obj(ln, f"t-{ln}", f"p-{ln}") for ln in lines)
            return chat_response("[" + arr + "]", "stop")

        calls = []
        orig = fake_post

        def counting_post(url, json, headers, timeout):
            calls.append(json)
            return orig(url, json, headers, timeout)

        monkeypatch.setattr("everyric2.translation.translator.requests.post", counting_post)

        result = translator.translate(
            "ア\nイ\nウ\nエ\nオ", source_lang="ja", target_lang="ko"
        )

        assert len(result.lines) == 5
        assert [l.translation for l in result.lines] == ["t-ア", "t-イ", "t-ウ", "t-エ", "t-オ"]
        # 임계 2, 배치 2 → 5줄은 3번 요청(2+2+1)
        assert len(calls) == 3


class TestSalvageJsonHelper:
    """_extract_json_items / _decode_json_objects / _align_items 단위 검증(요청 없이)."""

    def _probe(self):
        settings = TranslationSettings(engine="nvidia", api_key="dummy-key")
        return NvidiaTranslator(settings)

    def _slots(self, p, text, lines):
        return p._align_items(p._extract_json_items(text), lines)

    def test_full_array_parses_all(self):
        p = self._probe()
        text = (
            '[{"original":"a","translation":"A","pronunciation":"aa"},'
            '{"original":"b","translation":"B","pronunciation":"bb"}]'
        )
        slots = self._slots(p, text, ["a", "b"])
        assert [s.translation for s in slots] == ["A", "B"]

    def test_truncated_array_keeps_complete_prefix(self):
        p = self._probe()
        text = '[{"original":"a","translation":"A","pronunciation":"aa"},{"original":"b","transl'
        slots = self._slots(p, text, ["a", "b"])
        assert slots[0].translation == "A"
        assert slots[1] is None  # 잘린 라인은 빈 칸 — 호출자가 재요청한다

    def test_code_fenced_and_think_wrapped(self):
        p = self._probe()
        text = (
            "<think>reasoning...</think>\n```json\n"
            '[{"original":"x","translation":"X","pronunciation":"xx"}]\n```'
        )
        slots = self._slots(p, text, ["x"])
        assert len(slots) == 1
        assert slots[0].translation == "X"
        assert slots[0].pronunciation is None

    def test_no_array_returns_empty(self):
        p = self._probe()
        assert p._extract_json_items("sorry, I cannot help") == []
        assert p._extract_json_items("") == []
        assert self._slots(p, "sorry, I cannot help", ["a"]) == [None]

    def test_original_falls_back_to_positions_when_model_omits_it(self):
        # original을 아예 안 돌려준 응답 — 대조할 근거가 없으니 순서대로 채운다
        p = self._probe()
        text = '[{"translation":"A","pronunciation":"aa"}]'
        slots = self._slots(p, text, ["原文"])
        assert slots[0].original == "原文"
        assert slots[0].translation == "A"

    def test_align_ignores_whitespace_and_punctuation_differences(self):
        # 모델이 원문을 되돌려주며 문장부호·공백을 정돈해도 같은 라인으로 본다
        p = self._probe()
        text = '[{"original":"Hello  world","translation":"안녕"}]'
        slots = self._slots(p, text, ["Hello, world!"])
        assert slots[0].translation == "안녕"

    def test_align_handles_repeated_lines_in_order(self):
        # 후렴 반복 — 앞에서부터 순서대로 소비해야 한다
        p = self._probe()
        text = (
            '[{"original":"la","translation":"1"},'
            '{"original":"la","translation":"2"},'
            '{"original":"la","translation":"3"}]'
        )
        slots = self._slots(p, text, ["la", "la", "la"])
        assert [s.translation for s in slots] == ["1", "2", "3"]

    def test_align_keeps_input_original_not_model_echo(self):
        # 응답의 original은 대조용일 뿐 — 저장되는 원문은 항상 입력 라인이어야 한다
        p = self._probe()
        text = '[{"original":"hello world","translation":"안녕"}]'
        slots = self._slots(p, text, ["Hello, World!"])
        assert slots[0].original == "Hello, World!"


class TestLineAlignmentRegression:
    """수정 2 회귀: 모델이 한 줄을 빠뜨려도 이후 라인이 밀리면 안 된다.

    실증(video_id VB9cyPJCtok, 실리카겔 'APEX'): 0번 줄이 '실리카겔 - APEX'라는 제목
    텍스트여서 모델이 가사로 취급하지 않고 누락 → 31줄 전체가 정확히 1칸씩 밀려 저장.
    """

    def _probe(self):
        return NvidiaTranslator(TranslationSettings(engine="nvidia", api_key="dummy-key"))

    def test_missing_first_line_leaves_hole_instead_of_shifting(self):
        p = self._probe()
        lines = ["실리카겔 - APEX", "첫 번째 가사", "두 번째 가사", "세 번째 가사"]
        # 모델이 제목 줄을 건너뛰고 나머지 3줄만 돌려줬다
        text = (
            '[{"original":"첫 번째 가사","translation":"first"},'
            '{"original":"두 번째 가사","translation":"second"},'
            '{"original":"세 번째 가사","translation":"third"}]'
        )
        slots = p._align_items(p._extract_json_items(text), lines)

        assert slots[0] is None  # 제목 줄은 빈 칸 — 재요청 대상
        assert [s.translation for s in slots[1:]] == ["first", "second", "third"]
        # 밀림이 발생했다면 slots[0].translation == "first"였을 것이다

    def test_missing_middle_line_leaves_hole_instead_of_shifting(self):
        p = self._probe()
        lines = ["a", "b", "c", "d"]
        text = (
            '[{"original":"a","translation":"A"},'
            '{"original":"c","translation":"C"},'
            '{"original":"d","translation":"D"}]'
        )
        slots = p._align_items(p._extract_json_items(text), lines)

        assert [None if s is None else s.translation for s in slots] == ["A", None, "C", "D"]

    def test_plain_text_response_refuses_positional_match_on_count_mismatch(self):
        # 예전 구현은 빈 줄을 버리고 무조건 위치로 붙여 밀림을 만들었다
        p = self._probe()
        lines = ["제목", "가사1", "가사2"]
        slots = p._plain_text_slots("번역1\n번역2", lines)
        assert slots == [None, None, None]

    def test_plain_text_response_accepts_exact_count(self):
        p = self._probe()
        lines = ["a", "b"]
        slots = p._plain_text_slots("A\nB", lines)
        assert [s.translation for s in slots] == ["A", "B"]

    def test_plain_text_response_tolerates_one_preamble_line(self):
        p = self._probe()
        slots = p._plain_text_slots("Here is the translation:\nA\nB", ["a", "b"])
        assert [s.translation for s in slots] == ["A", "B"]

    def test_dropped_line_is_re_requested_and_lands_in_its_own_slot(self, monkeypatch, tmp_path):
        # 엔드투엔드: 누락 라인만 재요청해 제자리에 채우고, 나머지는 밀리지 않는다
        key_file = tmp_path / "nvapi.txt"
        key_file.write_text("dummy-key", encoding="utf-8")
        monkeypatch.setattr(NvidiaTranslator, "_KEY_FILE", key_file)
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        settings = TranslationSettings(engine="nvidia", api_key=None)
        settings.include_pronunciation = False
        translator = NvidiaTranslator(settings)

        calls = []
        responses = iter(
            [
                # 1차: 제목 줄(0번)을 누락한 3줄 응답
                chat_response(
                    '[{"original":"첫 줄","translation":"first"},'
                    '{"original":"둘째 줄","translation":"second"},'
                    '{"original":"셋째 줄","translation":"third"}]',
                    "stop",
                ),
                # 2차: 누락된 제목 줄만 재요청
                chat_response('[{"original":"실리카겔 - APEX","translation":"APEX"}]', "stop"),
            ]
        )

        def fake_post(url, json, headers, timeout):
            calls.append(json)
            return next(responses)

        monkeypatch.setattr("everyric2.translation.translator.requests.post", fake_post)

        result = translator.translate(
            "실리카겔 - APEX\n첫 줄\n둘째 줄\n셋째 줄", source_lang="ko", target_lang="en"
        )

        assert [line.original for line in result.lines] == [
            "실리카겔 - APEX", "첫 줄", "둘째 줄", "셋째 줄",
        ]
        assert [line.translation for line in result.lines] == [
            "APEX", "first", "second", "third",
        ]
        assert len(calls) == 2
        # 재요청은 누락 라인만 담았어야 한다
        assert "실리카겔 - APEX" in calls[1]["messages"][0]["content"]
        assert "둘째 줄" not in calls[1]["messages"][0]["content"]

    def test_unrecoverable_dropped_line_is_blank_not_shifted(self, monkeypatch, tmp_path):
        # 재요청도 실패하면 그 라인은 비운다 — 밀린 번역보다 빈 라인이 낫다
        key_file = tmp_path / "nvapi.txt"
        key_file.write_text("dummy-key", encoding="utf-8")
        monkeypatch.setattr(NvidiaTranslator, "_KEY_FILE", key_file)
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        settings = TranslationSettings(engine="nvidia", api_key=None)
        settings.include_pronunciation = False
        translator = NvidiaTranslator(settings)

        responses = iter(
            [
                chat_response(
                    '[{"original":"b","translation":"B"},{"original":"c","translation":"C"}]',
                    "stop",
                ),
                chat_response("", "length"),  # 재요청 실패
                chat_response("", "length"),  # 빈 응답 1회 재시도도 실패
            ]
        )
        monkeypatch.setattr(
            "everyric2.translation.translator.requests.post",
            lambda url, json, headers, timeout: next(responses),
        )

        result = translator.translate("title line\nb\nc", source_lang="ko", target_lang="en")

        assert [line.original for line in result.lines] == ["title line", "b", "c"]
        assert result.lines[0].translation == ""
        assert result.lines[0].failed is True
        assert [line.translation for line in result.lines[1:]] == ["B", "C"]


class TestTranslationSkipGate:
    """수정 1: 원문 언어 == 번역 대상 언어면 LLM을 부르지 않는다.

    실증(video_id uI9WfIkybAc): 한국어 곡을 한국어로 '재번역'해
    "저기요 제가요 가슴이 떨려서" → "저기, 나야. 가슴이 너무 떨려"가 저장됐다.
    """

    KO_SONG = (
        "저기요 제가요 가슴이 떨려서\n"
        "말도 제대로 못 하고 서 있어요\n"
        "그대가 웃어주면 하루가 다 지나가\n"
        "오늘 밤도 그렇게 흘러가네요"
    )
    EN_SONG = (
        "Walking down an empty street tonight\n"
        "Every window holds a different light\n"
        "I keep counting all the ways to leave\n"
        "But the morning never lets me be"
    )
    JA_SONG = "きみの声が聴こえる\n夜の街に消えていく光\n時計の針が止まらない"

    def setup_method(self):
        class _Probe(BaseTranslator):
            def translate(self, *a, **k):  # pragma: no cover - not exercised
                raise NotImplementedError

        self.probe = _Probe(TranslationSettings())

    def test_korean_song_to_korean_is_skipped(self):
        assert self.probe._should_skip_translation(self.KO_SONG, "auto", "ko") is True

    def test_korean_song_to_english_is_not_skipped(self):
        assert self.probe._should_skip_translation(self.KO_SONG, "auto", "en") is False

    def test_english_song_to_english_is_skipped(self):
        assert self.probe._should_skip_translation(self.EN_SONG, "auto", "en") is True

    def test_japanese_song_to_korean_is_not_skipped(self):
        assert self.probe._should_skip_translation(self.JA_SONG, "auto", "ko") is False

    def test_explicit_source_lang_is_trusted(self):
        assert self.probe._should_skip_translation("whatever", "ja", "ja") is True
        assert self.probe._should_skip_translation("whatever", "ja", "ko") is False

    def test_language_tags_are_normalized(self):
        assert self.probe._should_skip_translation("whatever", "ko-KR", "ko") is True
        assert self.probe._should_skip_translation("whatever", "KO", "ko-kr") is True

    @pytest.mark.parametrize(
        "text",
        [
            "짧다",  # 글자 수 부족 — 판정 보류
            "오늘 밤 tonight I walk 거리를 걸으며 thinking about you 그대 생각",  # 혼재
            "夜の街を歩く 오늘 밤 그대 생각 거리를 걸으며 웃어요",  # 다른 문자 체계 혼입
        ],
    )
    def test_uncertain_detection_never_skips(self, text):
        # 잘못 스킵하면 번역이 통째로 사라진다 — 확신이 없으면 번역한다
        assert self.probe._should_skip_translation(text, "auto", "ko") is False

    def test_korean_song_with_a_few_english_words_still_skips(self):
        text = self.KO_SONG + "\nOh yeah"
        assert self.probe._should_skip_translation(text, "auto", "ko") is True

    def test_skipped_translation_makes_no_request(self, monkeypatch, tmp_path):
        key_file = tmp_path / "nvapi.txt"
        key_file.write_text("dummy-key", encoding="utf-8")
        monkeypatch.setattr(NvidiaTranslator, "_KEY_FILE", key_file)
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        settings = TranslationSettings(engine="nvidia", api_key=None)
        settings.include_pronunciation = True  # 한국어 원문이면 발음 게이트가 먼저 끈다
        translator = NvidiaTranslator(settings)

        def boom(*a, **k):  # pragma: no cover - 호출되면 테스트 실패
            raise AssertionError("LLM must not be called when source lang == target lang")

        monkeypatch.setattr("everyric2.translation.translator.requests.post", boom)

        result = translator.translate(self.KO_SONG, source_lang="auto", target_lang="ko")

        assert result.translation_skipped is True
        assert len(result.lines) == 4
        assert [line.original for line in result.lines] == self.KO_SONG.split("\n")
        # 번역 필드는 비운다 — 원문을 복사하면 클라이언트가 같은 문장을 두 줄로 표시한다
        assert all(line.translation == "" for line in result.lines)
        assert all(not line.failed for line in result.lines)

    def test_different_language_still_calls_the_model(self, monkeypatch, tmp_path):
        key_file = tmp_path / "nvapi.txt"
        key_file.write_text("dummy-key", encoding="utf-8")
        monkeypatch.setattr(NvidiaTranslator, "_KEY_FILE", key_file)
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        settings = TranslationSettings(engine="nvidia", api_key=None)
        settings.include_pronunciation = False
        translator = NvidiaTranslator(settings)

        called = []

        def fake_post(url, json, headers, timeout):
            called.append(json)
            lyrics = json["messages"][0]["content"].split("LYRICS:\n")[-1].strip().split("\n")
            body = ",".join(
                '{"original":%s,"translation":"t-%d"}' % (dumps(ln), i)
                for i, ln in enumerate(lyrics)
            )
            return chat_response("[" + body + "]", "stop")

        monkeypatch.setattr("everyric2.translation.translator.requests.post", fake_post)

        result = translator.translate(self.KO_SONG, source_lang="auto", target_lang="en")

        assert called
        assert result.translation_skipped is False
        assert [line.translation for line in result.lines] == ["t-0", "t-1", "t-2", "t-3"]


class TestLowQualityBatchRetry:
    """수정 3: 절단이 아닌데 내용만 빈 배치를 재요청한다.

    실증(video_id vg6pnvn1u10): 48줄 중 19~34번의 번역·발음이 전부 빈 값인데
    서버 로그 전 기간에 잘림 경고가 한 건도 없었다 — 문법적으로 완전한 JSON이었다.
    """

    def _make(self, monkeypatch, tmp_path, include_pronunciation=False):
        key_file = tmp_path / "nvapi.txt"
        key_file.write_text("dummy-key", encoding="utf-8")
        monkeypatch.setattr(NvidiaTranslator, "_KEY_FILE", key_file)
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        settings = TranslationSettings(engine="nvidia", api_key=None)
        settings.include_pronunciation = include_pronunciation
        return NvidiaTranslator(settings)

    @staticmethod
    def _array(pairs):
        return "[" + ",".join(
            '{"original":%s,"translation":%s}' % (dumps(o), dumps(t)) for o, t in pairs
        ) + "]"

    def test_blank_middle_section_is_re_requested(self, monkeypatch, tmp_path):
        translator = self._make(monkeypatch, tmp_path)
        lines = [f"line {i}" for i in range(10)]
        # 1차: 중간 4줄(20% 이상)이 완전한 JSON인데 번역만 빈 값
        first = self._array([(ln, "" if 3 <= i <= 6 else f"T{i}") for i, ln in enumerate(lines)])

        calls = []
        responses = iter(
            [
                chat_response(first, "stop"),
                # 2차: 빈 라인만 다시 요청 — 이번엔 제대로 채워온다
                chat_response(
                    self._array([(f"line {i}", f"R{i}") for i in (3, 4, 5, 6)]), "stop"
                ),
            ]
        )

        def fake_post(url, json, headers, timeout):
            calls.append(json)
            return next(responses)

        monkeypatch.setattr("everyric2.translation.translator.requests.post", fake_post)

        result = translator.translate("\n".join(lines), source_lang="ja", target_lang="ko")

        assert len(calls) == 2
        assert [line.translation for line in result.lines] == [
            "T0", "T1", "T2", "R3", "R4", "R5", "R6", "T7", "T8", "T9",
        ]
        # 재요청은 빈 라인만 담았어야 한다
        retry_prompt = calls[1]["messages"][0]["content"]
        assert "line 3" in retry_prompt and "line 0" not in retry_prompt

    def test_model_pronunciation_is_ignored_by_quality_retry(self, monkeypatch, tmp_path):
        # 모델은 번역만 담당한다. 응답의 pronunciation 유무는 저품질 재요청 조건이 아니다.
        translator = self._make(monkeypatch, tmp_path, include_pronunciation=True)
        lines = [f"第{i}行的歌词" for i in range(6)]

        def obj(orig, trans, pron):
            return '{"original":%s,"translation":%s,"pronunciation":%s}' % (
                dumps(orig), dumps(trans), dumps(pron)
            )

        # 번역은 모두 정상이고 pronunciation만 일부 비어 있다.
        first = "[" + ",".join(
            obj(ln, f"T{i}", "" if i < 3 else f"pron{i}") for i, ln in enumerate(lines)
        ) + "]"

        calls = []

        def fake_post(url, json, headers, timeout):
            calls.append(json)
            return chat_response(first, "stop")

        monkeypatch.setattr("everyric2.translation.translator.requests.post", fake_post)

        result = translator.translate("\n".join(lines), source_lang="zh", target_lang="ko")

        assert len(calls) == 1
        assert [line.translation for line in result.lines] == [f"T{i}" for i in range(6)]
        assert all(line.pronunciation is None for line in result.lines)

    def test_few_blank_lines_do_not_trigger_a_retry(self, monkeypatch, tmp_path):
        # 10줄 중 1줄(10%)만 비었으면 임계 미만 — 재요청하지 않는다
        translator = self._make(monkeypatch, tmp_path)
        lines = [f"line {i}" for i in range(10)]
        payload = self._array([(ln, "" if i == 4 else f"T{i}") for i, ln in enumerate(lines)])

        calls = []

        def fake_post(url, json, headers, timeout):
            calls.append(json)
            return chat_response(payload, "stop")

        monkeypatch.setattr("everyric2.translation.translator.requests.post", fake_post)

        result = translator.translate("\n".join(lines), source_lang="ja", target_lang="ko")

        assert len(calls) == 1
        assert result.lines[4].translation == ""

    def test_retry_is_capped_and_keeps_first_result(self, monkeypatch, tmp_path):
        # 재요청도 비어 오면 한 번만 더 시도하고 멈춘다 (무한 재시도 금지)
        translator = self._make(monkeypatch, tmp_path)
        lines = [f"line {i}" for i in range(6)]
        blank_all = self._array([(ln, "") for ln in lines])
        blank_half = self._array([(f"line {i}", "") for i in (0, 1, 2, 3, 4, 5)])

        calls = []
        responses = iter([chat_response(blank_all, "stop"), chat_response(blank_half, "stop")])

        def fake_post(url, json, headers, timeout):
            calls.append(json)
            return next(responses)

        monkeypatch.setattr("everyric2.translation.translator.requests.post", fake_post)

        result = translator.translate("\n".join(lines), source_lang="ja", target_lang="ko")

        # 1차 + 저품질 재요청 1회 = 2회로 끝. 응답이 더 없으므로 3회째면 StopIteration이 난다
        assert len(calls) == 2
        assert len(result.lines) == 6
        assert all(line.translation == "" for line in result.lines)

    def test_short_originals_are_not_treated_as_low_quality(self, monkeypatch, tmp_path):
        # 감탄사·구두점 같은 짧은 원문은 번역이 비어도 품질 문제로 보지 않는다
        translator = self._make(monkeypatch, tmp_path)
        lines = ["아", "…", "가사가 있는 줄", "또 다른 가사 줄"]
        payload = self._array([("아", ""), ("…", ""), ("가사가 있는 줄", "A"), ("또 다른 가사 줄", "B")])

        calls = []

        def fake_post(url, json, headers, timeout):
            calls.append(json)
            return chat_response(payload, "stop")

        monkeypatch.setattr("everyric2.translation.translator.requests.post", fake_post)

        result = translator.translate("\n".join(lines), source_lang="ja", target_lang="ko")

        assert len(calls) == 1
        assert [line.translation for line in result.lines] == ["", "", "A", "B"]


class TestPromptBuilding:
    """_build_prompt는 대상 언어와 무관하게 번역만 요청한다."""

    def setup_method(self):
        class _Probe(BaseTranslator):
            def translate(self, *a, **k):  # pragma: no cover - not exercised
                raise NotImplementedError

        self.probe = _Probe(TranslationSettings())

    @pytest.mark.parametrize("target", ["ko", "en", "ja", "zh"])
    def test_prompt_never_requests_pronunciation(self, target):
        prompt = self.probe._build_prompt("時計の針が", target)
        assert "pronunciation" not in prompt.lower()
        assert "romanized" not in prompt.lower()
        assert "kana reading" not in prompt.lower()

    def test_song_context_is_injected(self):
        prompt = self.probe._build_prompt(
            "きみの声", "ko", context='"熱異常" by かいりきベア'
        )
        assert 'Song: "熱異常" by かいりきベア' in prompt

    def test_lyrics_guidance_present(self):
        prompt = self.probe._build_prompt("きみの声", "ko")
        assert "ONE song" in prompt
