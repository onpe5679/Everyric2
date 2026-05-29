"""OOV 음절 치환 + 미정렬 줄 보간 회귀 테스트.

no-mock: vocab은 실제 MMS 한국어 토크나이저 vocab을 그대로 기록한 fixture
(`fixtures/mms_kor_vocab.json`), 보간 테스트의 시각값은 실제 곡(test_song) CTC
출력에서 가져온 인접 줄 시각이다. 손으로 지어낸 합성 데이터 없음.
"""
import json
from pathlib import Path

from everyric2.alignment.ctc_engine import CTCEngine, _oov_substitute

VOCAB = json.loads(
    (Path(__file__).parent / "fixtures" / "mms_kor_vocab.json").read_text(encoding="utf-8")
)


def test_hook_syllables_are_oov():
    # '오빠가 제일 좋아!'의 훅 음절은 MMS 한국어 vocab(1330) 밖 → 이게 정렬 실패의 근본 원인
    assert "뿅" not in VOCAB
    assert "얍" not in VOCAB


def test_oov_substituted_into_vocab():
    # OOV 음절은 발음이 가까운 vocab 음절로 치환되어야 정렬 토큰이 생긴다
    for ch in ("뿅", "얍"):
        sub = _oov_substitute(ch, VOCAB)
        assert sub is not None, f"{ch} 치환 실패"
        assert sub in VOCAB, f"{ch}→{sub} 가 vocab 밖"
        assert sub != ch


def test_substitution_is_phonetically_close():
    # 된소리 풀기(ㅃ→ㅂ)·활음 풀기(ㅛ→ㅗ)·종성 제거만 적용 → 실측 결과 고정
    assert _oov_substitute("뿅", VOCAB) == "뾰"  # ㅃㅛㅇ → 종성 제거 → 뾰
    assert _oov_substitute("얍", VOCAB) == "야"  # ㅇㅑㅂ → 종성 제거 → 야


def test_non_hangul_not_substituted():
    # 괄호·기호·공백은 한글 음절이 아니므로 치환 대상이 아니다(None → 정렬에서 제외)
    for ch in ("(", ")", "!", " ", "?"):
        assert _oov_substitute(ch, VOCAB) is None


def test_in_vocab_syllable_never_returns_itself():
    # vocab에 있는 음절은 파이프라인에서 치환을 거치지 않지만, 직접 호출 시에도
    # 자기 자신을 후보로 돌려주지 않는다(치환은 '다른' 음절을 찾는 동작)
    assert _oov_substitute("안", VOCAB) != "안"


def test_interpolate_places_unaligned_line_between_neighbors():
    # 실측: test_song CTC 출력의 인접 줄 — idx8 '난 오빠랑...' 끝 48.90,
    # idx9 '(뿅뿅뿅뿅뿅)' 정렬 실패(None), idx10 '난 오빠만...' 시작 51.84.
    # 보간은 실패 줄을 앞뒤 사이에 끼워 순서를 보존해야 한다(역순 금지).
    line_times = [
        [44.58, 48.90, []],     # idx8 (aligned)
        [None, None, None],     # idx9 (정렬 실패)
        [51.84, 55.00, []],     # idx10 (aligned)
    ]
    CTCEngine._interpolate_unaligned(line_times, 204.10)

    s, e, _ = line_times[1]
    assert s is not None and e is not None
    assert line_times[0][1] <= s <= e <= line_times[2][0]  # 이웃 사이, 순서 보존
    assert abs(s - 48.90) < 1e-6   # 직전 줄 끝에서 시작
    assert abs(e - 51.84) < 1e-6   # 다음 줄 시작에서 끝(갭 1줄이라 전체 차지)


def test_interpolate_distributes_consecutive_unaligned_lines():
    # 연속 실패 줄 2개는 갭을 균등 분할
    line_times = [
        [10.0, 20.0, []],
        [None, None, None],
        [None, None, None],
        [50.0, 60.0, []],
    ]
    CTCEngine._interpolate_unaligned(line_times, 204.10)
    # 갭 20~50(30초)을 2줄이 15초씩
    assert abs(line_times[1][0] - 20.0) < 1e-6
    assert abs(line_times[1][1] - 35.0) < 1e-6
    assert abs(line_times[2][0] - 35.0) < 1e-6
    assert abs(line_times[2][1] - 50.0) < 1e-6


if __name__ == "__main__":
    # pytest 없이도 검증 가능한 러너 (Everyric2 venv는 런타임 전용)
    _fns = sorted(
        (v for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)),
        key=lambda f: f.__code__.co_firstlineno,
    )
    for _fn in _fns:
        _fn()
        print(f"PASS {_fn.__name__}")
    print(f"\n{len(_fns)} passed")
