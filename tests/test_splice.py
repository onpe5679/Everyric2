"""star-swallow 가드의 하이브리드 스플라이스(_splice_alignments) 단위 테스트.

시나리오(VWVtIg5cdDU 실측 축약): ko 정렬이 간주 이후 블록을 간주 앞으로 압축,
ja 정렬은 간주 이후에 정상 배치. 스플라이스는 간주 전 ko 유지 + 간주 후 ja 교체.
"""

from everyric2.inference.prompt import SyncResult
from everyric2.server.worker import _splice_alignments


def _r(i: int, start: float, end: float) -> SyncResult:
    return SyncResult(line_number=i + 1, text=f"line{i}", start_time=start, end_time=end)


def _mk(spans: list[tuple[float, float]]) -> list[SyncResult]:
    return [_r(i, s, e) for i, (s, e) in enumerate(spans)]


class TestSpliceAlignments:
    def test_normal_splice_keeps_ko_head_and_takes_ja_tail(self):
        # 간주 갭 [30, 60] → post_win=(60, 120). ko는 후반 3라인을 20~29s로 압축.
        ko = _mk([(0, 5), (6, 12), (14, 20), (21, 24), (25, 27), (28, 29.5)])
        ja = _mk([(0, 6), (7, 13), (15, 22), (61, 70), (72, 80), (82, 90)])
        post_win = (60.0, 120.0)

        k = _splice_alignments(ko, ja, post_win)

        assert k == 3
        # 간주 전: ko 타이밍 그대로
        assert (ko[0].start_time, ko[0].end_time) == (0, 5)
        assert (ko[2].start_time, ko[2].end_time) == (14, 20)
        # 간주 후: ja 타이밍으로 교체
        assert (ko[3].start_time, ko[3].end_time) == (61, 70)
        assert (ko[5].start_time, ko[5].end_time) == (82, 90)
        # 단조성 유지
        starts = [r.start_time for r in ko]
        assert starts == sorted(starts)

    def test_boundary_clamp_when_ko_head_overruns_ja_start(self):
        # ko 마지막 유지 라인(idx 1)의 끝이 ja 첫 교체 라인 시작(61)을 넘음 → 끝 클램프
        ko = _mk([(0, 10), (12, 70), (20, 25)])
        ja = _mk([(0, 11), (13, 30), (61, 75)])
        k = _splice_alignments(ko, ja, (60.0, 100.0))
        assert k == 2
        assert ko[1].end_time == 61  # bound로 클램프
        assert (ko[2].start_time, ko[2].end_time) == (61, 75)

    def test_degenerate_k0_returns_none(self):
        # ja가 전 라인을 간주 이후에 배치 → 전곡 교체 = 전곡 폴백과 동일하므로 None
        ko = _mk([(0, 5), (6, 10)])
        ja = _mk([(61, 65), (66, 70)])
        assert _splice_alignments(ko, ja, (60.0, 100.0)) is None
        assert (ko[0].start_time, ko[0].end_time) == (0, 5)  # 무변경

    def test_no_ja_line_after_interlude_returns_none(self):
        ko = _mk([(0, 5), (6, 10)])
        ja = _mk([(0, 5), (6, 10)])
        assert _splice_alignments(ko, ja, (60.0, 100.0)) is None

    def test_ko_head_crossing_boundary_returns_none_unchanged(self):
        # ko 유지 구간 라인이 경계 이후에서 시작 → 부분 보존 불가, ko 무변경
        ko = _mk([(0, 5), (62, 66), (20, 25)])
        ja = _mk([(0, 5), (7, 12), (61, 75)])
        before = [(r.start_time, r.end_time) for r in ko]
        assert _splice_alignments(ko, ja, (60.0, 100.0)) is None
        assert [(r.start_time, r.end_time) for r in ko] == before

    def test_word_segments_rescaled_on_clamp(self):
        from everyric2.inference.prompt import WordSegment

        ko = _mk([(0, 70)])
        ko[0].word_segments = [
            WordSegment(word="a", start=0, end=35),
            WordSegment(word="b", start=35, end=70),
        ]
        ko.append(_r(1, 10, 20))
        ja = _mk([(0, 30), (61, 75)])
        k = _splice_alignments(ko, ja, (60.0, 100.0))
        assert k == 1
        assert ko[0].end_time == 61
        assert ko[0].word_segments[-1].end == 61  # 선형 리스케일로 경계 안
