"""가나 읽기 분해 + 발음 표기 DP 정렬 회귀 테스트 (everyric2.text.reading).

no-mock: pykakasi 실제 변환 결과를 그대로 사용한다. 발음 표기 예시는 보카로 위키에
실제로 쓰이는 표기 관습(로키 등)을 본떴다.
"""
import pytest

from everyric2.text.reading import (
    align_pron_to_moras,
    pron_segments_for_line,
    text_to_moras,
)


def test_text_to_moras_basic_arubaito():
    # 아루바이토(5) + 와(は, 1) + 네쿠라모오도(ー 장음 포함 6) = 12모라
    moras = text_to_moras("アルバイトはネクラモード")
    assert [m.kana for m in moras] == [
        "あ", "る", "ば", "い", "と", "は", "ね", "く", "ら", "も", "ー", "ど",
    ]
    assert len(moras) == 12
    # 장음 ー는 독립된 1모라
    assert moras[10].kana == "ー"
    assert all(not m.is_ascii for m in moras)


def test_text_to_moras_youon_combines_into_one_mora():
    # テレキャスター: きゃ(キ+ャ)가 결합해 1모라
    moras = text_to_moras("テレキャスター")
    kanas = [m.kana for m in moras]
    assert kanas == ["て", "れ", "きゃ", "す", "た", "ー"]
    assert len(moras) == 6


def test_text_to_moras_sokuon_shares_kanji_char_span():
    # 背負った: pykakasi 읽기 せおった 기준 ッ은 1모라이며, おくりがな 촉음이
    # 한자(背負) 토큰에 붙어 있으므로 그 한자 글자 구간(0,3)을 공유한다.
    moras = text_to_moras("背負った")
    kanas = [m.kana for m in moras]
    assert kanas == ["せ", "お", "っ", "た"]
    assert moras[2].kana == "っ"
    assert (moras[2].char_start, moras[2].char_end) == (0, 3)
    assert (moras[0].char_start, moras[0].char_end) == (0, 3)
    assert (moras[1].char_start, moras[1].char_end) == (0, 3)
    # た는 별도 토큰(원문 3번째 글자)
    assert (moras[3].char_start, moras[3].char_end) == (3, 4)


def test_align_pron_to_moras_full_line_high_quality():
    # 아루바이토와 네쿠라 모오도 - は가 조사로서 "와"로 읽히는 가창 관습까지 포함해
    # 전 음절이 1:1로 resolved 되어야 하고 품질이 높아야 한다.
    moras = text_to_moras("アルバイトはネクラモード")
    syllables, quality = align_pron_to_moras(moras, "아루바이토와 네쿠라 모오도")

    assert quality >= 0.8
    assert len(syllables) == 12
    assert all(s.resolved for s in syllables)
    # 모라 구간은 순서대로 단조 증가하며 1:1 매칭이다
    assert [s.mora_start for s in syllables] == list(range(12))
    assert [s.mora_end for s in syllables] == [i + 1 for i in range(12)]
    # 촉음/장음 규칙: ー(모라 10)가 "오"에 매칭
    assert syllables[10].text == "오"
    assert syllables[10].mora_start == 10


def test_align_pron_to_moras_sokuon_absorbed_as_batchim():
    # ゆーて お坊っちゃんお嬢ちゃん ↔ 유우테 오봇짱 오죠오짱
    # 촉음(っ)·발음(ん)이 직전 음절 받침으로 흡수되어야 하며, 전체적으로 단조·전 커버.
    moras = text_to_moras("ゆーて お坊っちゃんお嬢ちゃん")
    syllables, quality = align_pron_to_moras(moras, "유우테 오봇짱 오죠오짱")

    assert quality >= 0.5
    assert all(s.resolved for s in syllables)
    # 단조 증가(역순 없음)
    starts = [s.mora_start for s in syllables]
    assert starts == sorted(starts)
    # 모든 모라가 어딘가에는 커버된다 (마지막 음절의 mora_end가 전체 모라 수와 일치)
    assert syllables[-1].mora_end == len(moras)
    # っ 뒤에 오는 봇 음절이 っ까지 흡수해 mora_end가 +1 확장되어야 한다
    bot = next(s for s in syllables if s.text == "봇")
    assert bot.mora_end - bot.mora_start == 2
    absorbed_kana = moras[bot.mora_end - 1].kana
    assert absorbed_kana == "っ"


def test_align_pron_to_moras_ascii_unit_allows_one_to_many():
    # Don't Stop！ ↔ 돈 스탑 - ASCII 유닛(Don't, Stop)에 여러 음절이 몰려 배정될 수
    # 있다(1:N). 정확히 어느 유닛에 몰리는지는 비용이 동률이라 결정론적이되 자명하지
    # 않으므로, 여기서는 1:N 패턴 자체와 resolved 유지만 검증한다.
    moras = text_to_moras("Don't Stop！")
    assert len(moras) == 2
    assert all(m.is_ascii for m in moras)

    syllables, quality = align_pron_to_moras(moras, "돈 스탑")

    assert len(syllables) == 3
    assert all(s.resolved for s in syllables)
    assert quality >= 0.5
    # 두 ASCII 모라가 모두 사용되고, 최소 하나는 2개 이상의 음절을 공유한다(1:N)
    ranges = [(s.mora_start, s.mora_end) for s in syllables]
    assert set(ranges) == {(0, 1), (1, 2)}
    assert any(ranges.count(r) >= 2 for r in set(ranges))
    # 순서는 항상 보존된다 (mora_start가 감소하지 않음)
    assert [r[0] for r in ranges] == sorted(r[0] for r in ranges)


def test_align_pron_to_moras_kanji_multi_mora():
    # 長い前髪(ながいまえがみ): 長い 토큰이 3모라(な,が,い)를 갖고 前髪 토큰이
    # 4모라(ま,え,が,み)를 갖는다. pron과 1:1 정렬되어야 한다.
    moras = text_to_moras("長い前髪")
    assert [m.kana for m in moras] == ["な", "が", "い", "ま", "え", "が", "み"]
    assert (moras[0].char_start, moras[0].char_end) == (0, 2)
    assert (moras[2].char_start, moras[2].char_end) == (0, 2)
    assert (moras[3].char_start, moras[3].char_end) == (2, 4)

    syllables, quality = align_pron_to_moras(moras, "나가이 마에가미")
    assert quality >= 0.8
    assert len(syllables) == 7
    assert all(s.resolved for s in syllables)
    assert [s.mora_start for s in syllables] == list(range(7))


def test_pron_segments_for_line_monotonic_and_span_preserving():
    # 등간격 합성 char_spans(0.1초/글자)로 음절 세그먼트가 단조 증가하고
    # 전체 구간(0.0~1.2초)을 보존하는지 확인한다.
    text = "アルバイトはネクラモード"
    pron = "아루바이토와 네쿠라 모오도"
    char_spans = [(ch, i * 0.1, (i + 1) * 0.1) for i, ch in enumerate(text)]

    segments = pron_segments_for_line(char_spans, text, pron)

    assert segments is not None
    assert len(segments) == 12
    assert segments[0]["start"] == pytest.approx(0.0)
    assert segments[-1]["end"] == pytest.approx(1.2)
    for prev, cur in zip(segments, segments[1:]):
        assert cur["start"] >= prev["end"] - 1e-9
        assert cur["end"] >= cur["start"]
    assert all(s["resolved"] for s in segments)


def test_pron_segments_for_line_interpolates_missing_char():
    # CTC가 일부 글자(OOV 드롭)를 건너뛰어도 이웃 글자 시간으로 보간되어 전체
    # 구간을 보존한 채 세그먼트가 나와야 한다.
    text = "アルバイトはネクラモード"
    pron = "아루바이토와 네쿠라 모오도"
    char_spans = [
        (ch, i * 0.1, (i + 1) * 0.1) for i, ch in enumerate(text) if ch != "は"
    ]

    segments = pron_segments_for_line(char_spans, text, pron)

    assert segments is not None
    assert len(segments) == 12
    assert segments[0]["start"] == pytest.approx(0.0)
    assert segments[-1]["end"] == pytest.approx(1.2)
    for prev, cur in zip(segments, segments[1:]):
        assert cur["start"] >= prev["end"] - 1e-9


def test_pron_segments_for_line_returns_none_for_bad_pronunciation():
    # 엉터리 발음(가나 행/모음과 무관한 음절 반복)은 품질 미달로 None을 반환해야
    # 호출부가 그라데이션 폴백을 쓸 수 있게 한다.
    text = "アルバイトはネクラモード"
    char_spans = [(ch, i * 0.1, (i + 1) * 0.1) for i, ch in enumerate(text)]
    bad_pron = "뻐" * len(text_to_moras(text))

    assert pron_segments_for_line(char_spans, text, bad_pron) is None


def test_pron_segments_for_line_empty_inputs_return_none():
    assert pron_segments_for_line([], "アルバイト", "아루바이토") is None
    assert pron_segments_for_line([("あ", 0.0, 0.1)], "", "아") is None


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
