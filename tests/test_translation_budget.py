"""요청당 총예산 (외부 감사 #9) — 누적 NIM 왕복·소요시간 상한.

실측: 요청 789건 중 42%가 재시도를 겪고 p95 25.4s·최대 57.5s였는데 원인은 429가 아니라
재귀적 미스매치 복구(depth 4까지, OpenAICompatibleTranslator._translate_batch)와 저품질
배치 재요청(_retry_low_quality)의 무제한 중첩이었다 — 영상 하나는 depth 1~4 전 구간에서
'matched 0/N lines'가 150줄어치 반복되며 순차 NIM 왕복을 다수 태웠다.

여기서는 그 무제한 중첩을 결정론적으로 재현(전 라인이 항상 원문과 대조되지 않는 응답)하고,
budget_max_round_trips/budget_max_duration_sec가 그 위에서 실제 HTTP 왕복 수를 상한 안에
가두면서도 예외 없이 부분 결과로 정상 반환하는지 검증한다. 기존 개별 재시도 메커니즘
(미스매치 복구·저품질 재요청·429 백오프)은 손대지 않았으므로 그쪽 회귀는 test_nvidia_
translator.py/test_translate_parallel.py가 그대로 담당한다. 실제 NIM API는 호출하지 않는다
(requests.post를 mock).
"""

import time
from dataclasses import dataclass, field

import everyric2.translation.translator as tr
from everyric2.config.settings import TranslationSettings
from everyric2.translation.translator import NvidiaTranslator, TranslationBudget


@dataclass
class FakeResponse:
    status_code: int = 200
    payload: dict = field(default_factory=dict)
    ok: bool = True
    text: str = ""
    headers: dict = field(default_factory=dict)

    def json(self):
        return self.payload


def chat_response(content: str, finish_reason: str = "stop") -> FakeResponse:
    return FakeResponse(
        payload={"choices": [{"message": {"content": content}, "finish_reason": finish_reason}]}
    )


def translated(lines: list[str]) -> FakeResponse:
    body = ",".join('{"original":"%s","translation":"t-%s"}' % (ln, ln) for ln in lines)
    return chat_response("[" + body + "]")


def prompt_lines(payload: dict) -> list[str]:
    return payload["messages"][0]["content"].split("LYRICS:\n")[-1].strip().split("\n")


def make_translator(monkeypatch, tmp_path, **overrides) -> NvidiaTranslator:
    key_file = tmp_path / "nvapi.txt"
    key_file.write_text("dummy-key", encoding="utf-8")
    monkeypatch.setattr(NvidiaTranslator, "_KEY_FILE", key_file)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    settings = TranslationSettings(
        engine="nvidia", api_key=None, include_pronunciation=False, **overrides
    )
    return NvidiaTranslator(settings)


class TestBudgetDefaults:
    def test_default_budget_matches_the_recommended_values(self):
        # 왕복 8회 / 90초 — 확장 타임아웃(120s)보다 낮게, p95(25.4s)보다는 넉넉하게
        settings = TranslationSettings()
        assert settings.budget_max_round_trips == 8
        assert settings.budget_max_duration_sec == 90.0


class TestTranslationBudgetUnit:
    """TranslationBudget 클래스 자체의 계약 — 0 이하는 그 축을 무제한으로 취급한다."""

    def test_exhausts_on_round_trip_cap(self):
        budget = TranslationBudget(max_round_trips=2, max_duration_sec=0)
        assert budget.exhausted() is False
        budget.record_round_trip()
        assert budget.exhausted() is False
        budget.record_round_trip()
        assert budget.exhausted() is True

    def test_exhausts_on_duration_cap(self):
        budget = TranslationBudget(max_round_trips=0, max_duration_sec=0.05)
        assert budget.exhausted() is False
        time.sleep(0.08)
        assert budget.exhausted() is True

    def test_non_positive_limits_mean_unlimited(self):
        budget = TranslationBudget(max_round_trips=0, max_duration_sec=0)
        for _ in range(1000):
            budget.record_round_trip()
        assert budget.exhausted() is False

    def test_warn_once_logs_a_single_time(self, caplog):
        import logging

        budget = TranslationBudget(max_round_trips=1, max_duration_sec=0)
        budget.record_round_trip()
        with caplog.at_level(logging.WARNING, logger="everyric2.translation.translator"):
            budget.warn_once("[vid] ")
            budget.warn_once("[vid] ")
        assert len(caplog.records) == 1
        assert "budget exhausted" in caplog.records[0].message


class TestUnboundedMismatchRecoveryIsCapped:
    """루트 원인 재현: 응답이 원문과 전혀 대조되지 않으면(예: 빈 배열) 미스매치 복구가
    절반 분할로 재귀한다 — 8줄 전 라인이 매번 대조 실패하면 예산 없이는 15회 왕복
    (depth0:1 + depth1:2 + depth2:4 + depth3:8)까지 치솟는다. 예산이 그 재귀 트리 전체에
    걸쳐 있으므로, 상한에 닿는 순간 이후의 모든 하위 호출이 네트워크 없이 즉시
    실패 라인으로 마감돼야 한다.
    """

    def test_round_trips_are_capped_and_the_rest_returns_as_failed_lines(
        self, monkeypatch, tmp_path
    ):
        translator = make_translator(monkeypatch, tmp_path, budget_max_round_trips=5)
        lines = [f"line{i}" for i in range(8)]

        calls = []

        def fake_post(url, json, headers, timeout):
            calls.append(json)
            return chat_response("[]", "stop")  # 원문을 하나도 못 돌려줌 → 전원 미스매치

        monkeypatch.setattr("everyric2.translation.translator.requests.post", fake_post)

        # 예외 없이 정상 반환돼야 한다 (부분 번역이 무번역보다 낫다)
        result = translator.translate("\n".join(lines), source_lang="ja", target_lang="ko")

        # 예산(5)에서 정확히 멈춘다 — 예산이 없었다면 15회까지 치솟았을 재귀 트리
        assert len(calls) == 5
        assert len(result.lines) == 8
        assert [ln.original for ln in result.lines] == lines
        assert all(ln.translation == "" for ln in result.lines)
        assert all(ln.failed for ln in result.lines)

    def test_without_a_budget_the_same_input_makes_far_more_round_trips(
        self, monkeypatch, tmp_path
    ):
        # 대조군 — 예산을 비활성화(0)하면 기존 동작(무제한 재귀) 그대로 15회까지 간다.
        # 이 비교가 없으면 위 테스트의 "5"가 예산 덕분인지 우연인지 알 수 없다.
        translator = make_translator(monkeypatch, tmp_path, budget_max_round_trips=0)
        lines = [f"line{i}" for i in range(8)]

        calls = []

        def fake_post(url, json, headers, timeout):
            calls.append(json)
            return chat_response("[]", "stop")

        monkeypatch.setattr("everyric2.translation.translator.requests.post", fake_post)

        result = translator.translate("\n".join(lines), source_lang="ja", target_lang="ko")

        assert len(calls) == 15
        assert all(ln.failed for ln in result.lines)

    def test_budget_does_not_interfere_when_nothing_needs_recovering(
        self, monkeypatch, tmp_path
    ):
        # 정상 응답(1회 왕복)만 필요한 곡은 예산이 있어도 그대로 성공한다 — 회귀 방지
        translator = make_translator(monkeypatch, tmp_path, budget_max_round_trips=5)
        lines = ["line0", "line1", "line2"]

        calls = []

        def fake_post(url, json, headers, timeout):
            calls.append(json)
            return translated(prompt_lines(json))

        monkeypatch.setattr("everyric2.translation.translator.requests.post", fake_post)

        result = translator.translate("\n".join(lines), source_lang="ja", target_lang="ko")

        assert len(calls) == 1
        assert [ln.translation for ln in result.lines] == ["t-line0", "t-line1", "t-line2"]
        assert all(not ln.failed for ln in result.lines)


class TestIndependentBatchesRespectTheBudget:
    """여러 독립 배치(순차, batch_concurrency=1로 결정론 확보)에서도 예산은 새 배치를 아예
    시작하지 않는 방식으로 총 왕복을 가둔다."""

    def test_later_batches_are_skipped_once_the_budget_is_spent(self, monkeypatch, tmp_path):
        monkeypatch.setattr(tr, "_TEXT_BATCH_THRESHOLD", 2)
        monkeypatch.setattr(tr, "_TEXT_BATCH_SIZE", 2)
        translator = make_translator(
            monkeypatch, tmp_path, budget_max_round_trips=2, batch_concurrency=1
        )
        lines = [f"line{i}" for i in range(6)]  # 2줄씩 3배치

        calls = []

        def fake_post(url, json, headers, timeout):
            calls.append(json)
            return translated(prompt_lines(json))

        monkeypatch.setattr("everyric2.translation.translator.requests.post", fake_post)

        result = translator.translate("\n".join(lines), source_lang="ja", target_lang="ko")

        assert len(calls) == 2  # 세 번째 배치는 네트워크를 아예 타지 않는다
        assert len(result.lines) == 6
        assert [ln.original for ln in result.lines] == lines
        assert [ln.translation for ln in result.lines[:4]] == [
            "t-line0", "t-line1", "t-line2", "t-line3",
        ]
        assert [ln.translation for ln in result.lines[4:]] == ["", ""]
        assert [ln.failed for ln in result.lines[4:]] == [True, True]


class TestDurationBudget:
    def test_duration_cap_stops_later_batches_without_raising(self, monkeypatch, tmp_path):
        monkeypatch.setattr(tr, "_TEXT_BATCH_THRESHOLD", 1)
        monkeypatch.setattr(tr, "_TEXT_BATCH_SIZE", 1)
        translator = make_translator(
            monkeypatch,
            tmp_path,
            budget_max_round_trips=0,  # 왕복 축은 끄고 시간 축만 검증
            budget_max_duration_sec=0.05,
            batch_concurrency=1,
        )
        lines = ["line0", "line1", "line2"]

        calls = []

        def fake_post(url, json, headers, timeout):
            calls.append(json)
            time.sleep(0.08)  # 첫 배치만으로 시간 예산을 넘긴다
            return translated(prompt_lines(json))

        monkeypatch.setattr("everyric2.translation.translator.requests.post", fake_post)

        result = translator.translate("\n".join(lines), source_lang="ja", target_lang="ko")

        assert len(calls) == 1  # 두 번째부터는 시간 예산 소진으로 네트워크를 타지 않는다
        assert result.lines[0].translation == "t-line0"
        assert result.lines[0].failed is False
        assert [ln.failed for ln in result.lines[1:]] == [True, True]
