"""유튜브 자막 json3 파싱 테스트 (네트워크 없이 순수 변환만)."""
from everyric2.server.api.captions import json3_events_to_lines


def _ev(t_ms: int, d_ms: int, text: str) -> dict:
    return {"tStartMs": t_ms, "dDurationMs": d_ms, "segs": [{"utf8": text}]}


class TestJson3ToLines:
    def test_basic_lines_with_timing(self):
        data = {"events": [_ev(10738, 4370, "隠しきれない"), _ev(15108, 3983, "次の行")]}
        lines = json3_events_to_lines(data)
        assert lines == [
            {"start": 10.738, "end": 15.108, "text": "隠しきれない"},
            {"start": 15.108, "end": 19.091, "text": "次の行"},
        ]

    def test_filters_empty_and_music_notes(self):
        data = {"events": [
            _ev(0, 1000, "♪♪"),
            _ev(1000, 1000, "   "),
            {"tStartMs": 2000, "dDurationMs": 500},  # segs 없음 (창 배치 이벤트)
            _ev(3000, 1000, "실제 가사"),
        ]}
        lines = json3_events_to_lines(data)
        assert [line["text"] for line in lines] == ["실제 가사"]

    def test_merges_consecutive_duplicates(self):
        # 롤링 자막: 같은 줄이 연속 이벤트로 반복되면 end만 늘려 병합
        data = {"events": [_ev(0, 1000, "같은 줄"), _ev(1000, 1000, "같은 줄"), _ev(2000, 1000, "다른 줄")]}
        lines = json3_events_to_lines(data)
        assert len(lines) == 2
        assert lines[0] == {"start": 0.0, "end": 2.0, "text": "같은 줄"}

    def test_multi_seg_event_joined(self):
        data = {"events": [{
            "tStartMs": 0, "dDurationMs": 1000,
            "segs": [{"utf8": "앞"}, {"utf8": " 뒤"}],
        }]}
        assert json3_events_to_lines(data)[0]["text"] == "앞 뒤"
