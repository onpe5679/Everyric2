"""형태소 분석 기반 일본어 읽기(everyric2.text.ja_reading) 회귀 테스트.

no-mock: fugashi + unidic-lite 실제 분석 결과를 그대로 쓴다. 케이스는 pykakasi 사전
읽기가 실제로 오독했던 가사 라인들이다(今更止められない→"やめ", 縋って→"つい"って).

폴백 테스트만 예외적으로 태거 생성을 몽키패치한다 — fugashi가 없는 환경을 실제로
만들 수 없기 때문이며, 검증 대상은 읽기 품질이 아니라 계약(형태/오프셋) 유지다.
"""
import pytest

from everyric2.text import ja_reading
from everyric2.text.ja_reading import (
    ReadingToken,
    kana_reading,
    reading_source,
    tokenize_reading,
)
from everyric2.text.reading import text_to_moras


@pytest.fixture
def kakasi_fallback(monkeypatch):
    """태거 생성을 실패시켜 pykakasi 폴백 경로를 강제한다."""

    def boom():
        raise ImportError("no fugashi in this environment")

    monkeypatch.setattr(ja_reading, "_create_tagger", boom)
    monkeypatch.setattr(ja_reading, "_tagger", None)
    monkeypatch.setattr(ja_reading, "_tagger_unavailable", False)
    yield
    # 몽키패치가 풀려도 캐시된 '사용 불가' 플래그가 남으면 이후 테스트가 폴백에 갇힌다
    monkeypatch.setattr(ja_reading, "_tagger", None)
    monkeypatch.setattr(ja_reading, "_tagger_unavailable", False)


# ---------------------------------------------------------------------------
# 1. 실측 오독 케이스 — 문맥 의존 훈독
# ---------------------------------------------------------------------------

# (원문, 읽기에 반드시 들어가야 하는 부분, 들어가면 안 되는 오독)
_READING_CASES = [
    ("今更止められない", "いまさらとめられない", "やめ"),
    ("縋って", "すがって", "つい"),
    ("涙を止める", "なみだをとめる", "やめる"),
    ("風が止む", "かぜがやむ", "とむ"),
    ("目立って", "めだって", None),
    ("叶えたい", "かなえたい", None),
    ("取り計らって", "とりはからって", None),
    ("巷で流行りの", "ちまたではやりの", None),
    ("君にだけ", "きみにだけ", "くん"),
    ("何一つ解らなくても", "なにひとつわからなくても", "なんひとつ"),
    ("その内声も届かなくなって", "そのうちこえもとどかなくなって", "ないせい"),
    (
        "少しずつ錆びついていく身体 君のことを忘れていく",
        "すこしずつさびついていくしんたい きみのことをわすれていく",
        "くんの",
    ),
]


@pytest.mark.parametrize(("text", "expected", "misread"), _READING_CASES)
def test_reading_matches_context_dependent_truth(text, expected, misread):
    reading = kana_reading(text)
    assert reading == expected
    if misread is not None:
        assert misread not in reading


def test_stop_verb_is_disambiguated_by_particle():
    # 같은 止 한자가 조사에 따라 갈린다 — 사전 표제어만으로는 못 하는 판별
    assert "とめる" in kana_reading("涙を止める")
    assert "やむ" in kana_reading("風が止む")


# ---------------------------------------------------------------------------
# 2. 계약: 표면 이어 붙이기 = 원문 복원
# ---------------------------------------------------------------------------

_ROUNDTRIP_LINES = [
    "縋って いつも縋って",
    "ゆーて お坊っちゃんお嬢ちゃん",
    "Don't Stop！ 2024年の夏",
    "君にだけ、そう…「止められない」！",
    "  先頭と末尾に空白  ",
    "アルバイトはネクラモード",
    "",
]


@pytest.mark.parametrize("text", _ROUNDTRIP_LINES)
def test_surfaces_concatenate_back_to_the_original(text):
    tokens = tokenize_reading(text)
    assert "".join(t.surface for t in tokens) == text


@pytest.mark.parametrize("text", _ROUNDTRIP_LINES)
def test_surfaces_concatenate_back_to_the_original_on_fallback(text, kakasi_fallback):
    tokens = tokenize_reading(text)
    assert "".join(t.surface for t in tokens) == text


def test_whitespace_survives_as_a_literal_token():
    # MeCab은 공백을 버린다 — 리터럴 토큰으로 복원되지 않으면 이후 오프셋이 전부 밀린다
    text = "縋って いつも縋って"
    tokens = tokenize_reading(text)
    space = [t for t in tokens if t.surface == " "]
    assert len(space) == 1
    assert space[0].reading == " "
    assert (space[0].start, space[0].end) == (3, 4)


# ---------------------------------------------------------------------------
# 3. 계약: 오프셋 정확성
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", _ROUNDTRIP_LINES)
def test_token_offsets_point_at_their_own_surface(text):
    pos = 0
    for token in tokenize_reading(text):
        assert text[token.start : token.end] == token.surface
        assert token.start == pos  # 구간은 빈틈 없이 이어진다
        pos = token.end
    assert pos == len(text)


# ---------------------------------------------------------------------------
# 4. text_to_moras 연동 — 모라별 원문 오프셋
# ---------------------------------------------------------------------------


def test_moras_keep_original_char_offsets_across_a_space():
    text = "縋って いつも縋って"
    moras = text_to_moras(text)
    assert [m.kana for m in moras] == [
        "す", "が", "っ", "て", "い", "つ", "も", "す", "が", "っ", "て",
    ]
    # 공백은 모라를 만들지 않고 글자 인덱스만 건너뛴다. 정밀 귀속(2026-07-28) 후
    # 촉음·가나는 자기 글자에, 한자는 자기 런에 붙는다.
    assert (moras[0].char_start, moras[0].char_end) == (0, 1)   # 縋
    assert (moras[2].char_start, moras[2].char_end) == (1, 2)   # っ (자기 글자)
    assert (moras[3].char_start, moras[3].char_end) == (2, 3)   # て
    assert (moras[4].char_start, moras[4].char_end) == (4, 5)   # い (공백 뒤)
    assert (moras[7].char_start, moras[7].char_end) == (7, 8)   # 縋 (두 번째)
    assert (moras[-1].char_start, moras[-1].char_end) == (9, 10)
    # 모라 구간은 원문 범위를 넘지 않고 순서대로 증가한다
    assert all(0 <= m.char_start < m.char_end <= len(text) for m in moras)
    assert [m.char_start for m in moras] == sorted(m.char_start for m in moras)


def test_moras_of_a_misread_line_follow_the_correct_reading():
    # 독음이 틀리면 모라 열이 틀어져 발음 타이밍이 어긋난다 — とめ 기준으로 나와야 한다
    moras = text_to_moras("今更止められない")
    assert [m.kana for m in moras] == [
        "い", "ま", "さ", "ら", "と", "め", "ら", "れ", "な", "い",
    ]


def test_ascii_tokens_stay_single_units_with_correct_offsets():
    text = "Don't Stop！ 2024年"
    moras = text_to_moras(text)
    ascii_moras = [m for m in moras if m.is_ascii]
    assert [m.kana for m in ascii_moras] == ["Don't", "Stop", "2024"]
    for m in ascii_moras:
        assert text[m.char_start : m.char_end] == m.kana
    # 年은 일본어 토큰으로 읽힌다
    assert moras[-1].kana == "ん"
    assert [m.kana for m in moras if not m.is_ascii] == ["ね", "ん"]


# ---------------------------------------------------------------------------
# 5. fugashi 없는 환경 폴백 — 계약 동일
# ---------------------------------------------------------------------------


def test_reading_source_reports_the_engine_in_use():
    assert reading_source() == "fugashi"


def test_reading_source_reports_pykakasi_on_fallback(kakasi_fallback):
    assert reading_source() == "pykakasi"


def test_fallback_keeps_the_token_contract(kakasi_fallback):
    text = "縋って いつも縋って"
    tokens = tokenize_reading(text)

    assert tokens and all(isinstance(t, ReadingToken) for t in tokens)
    assert "".join(t.surface for t in tokens) == text
    for token in tokens:
        assert text[token.start : token.end] == token.surface
        assert isinstance(token.reading, str) and token.reading
    # 폴백 경로는 사전 읽기라 오독한다(つい) — 그래도 형태·오프셋 계약은 지킨다
    assert "つい" in kana_reading(text)


def test_fallback_moras_still_map_to_original_chars(kakasi_fallback):
    text = "縋って いつも縋って"
    moras = text_to_moras(text)
    assert moras
    assert all(text[m.char_start : m.char_end] for m in moras)
    assert all(0 <= m.char_start < m.char_end <= len(text) for m in moras)


# ---------------------------------------------------------------------------
# 6. 장음부(ー)·촉음(っ) — 모라 분해에서 각각 1박을 차지해야 한다
# ---------------------------------------------------------------------------


def test_long_vowel_mark_survives_as_its_own_mora():
    # UniDic kana는 가타카나(コーヒー) — 히라가나로 내려가도 ー는 그대로 남고 1모라다
    assert kana_reading("コーヒー") == "こーひー"
    moras = text_to_moras("コーヒー")
    assert [m.kana for m in moras] == ["こ", "ー", "ひ", "ー"]


def test_sokuon_survives_as_its_own_mora():
    assert kana_reading("行った") == "いった"
    moras = text_to_moras("行った")
    assert [m.kana for m in moras] == ["い", "っ", "た"]
    # 정밀 귀속(2026-07-28): 촉음은 어간 토큰 안에서도 **자기 글자**(っ, 원문 1번)에 붙는다
    assert (moras[0].char_start, moras[0].char_end) == (0, 1)  # 行=い
    assert (moras[1].char_start, moras[1].char_end) == (1, 2)  # っ
    assert (moras[2].char_start, moras[2].char_end) == (2, 3)


def test_katakana_reading_is_lowered_to_hiragana():
    # 요음 ャ도 ゃ로 내려가 직전 가나와 결합해 1모라가 된다
    assert kana_reading("テレキャスター") == "てれきゃすたー"
    assert [m.kana for m in text_to_moras("テレキャスター")] == [
        "て", "れ", "きゃ", "す", "た", "ー",
    ]


def test_long_vowel_and_sokuon_in_one_line():
    moras = text_to_moras("コーヒー行った")
    kanas = [m.kana for m in moras]
    assert kanas.count("ー") == 2
    assert kanas.count("っ") == 1


# ---------------------------------------------------------------------------
# 가타카나 표층 읽기 / 홀로 선 접두사 — 위키 사람 발음 288줄 실측으로 찾은 오독 2종
# ---------------------------------------------------------------------------


def test_katakana_surface_beats_the_dictionary():
    """가타카나는 표음 문자라 사전을 조회하면 오히려 틀린다.

    실측(위키 사람 발음 288줄): UniDic이 エグい를 えぎい로 읽어 「에기이요」가 나왔고
    (정답 「에구이요」), レイニー의 pron이 れーにー로 장음을 뭉개 「레에니이」가 됐다
    (정답 「레이니이」). 두 오독이 독음오류 48줄 중 9줄이었다.
    """
    from everyric2.text.pron_style import wiki_pronunciation

    assert wiki_pronunciation("エグいよ") == "에구이요"
    assert wiki_pronunciation("レイニーブーツ") == "레이니이 부우츠"
    # 진짜 장음(표층에 ー가 그대로 있는 것)은 그대로 남아야 한다 — 이 규칙이
    # 장음을 이중모음으로 바꿔 버리면 반대 방향 회귀가 된다
    assert wiki_pronunciation("ゲーム") == "게에무"
    assert wiki_pronunciation("コーヒー") == "코오히이"


def test_orphan_prefix_does_not_use_the_prefix_reading():
    """붙을 내용어가 없는 접두사는 접두사 읽기를 쓰면 안 된다.

    실측: 「さり気ない愛 盛りすぎる愛」의 첫 愛가 接頭辞/まな로 읽혀 「마나」가 됐다
    (정답 「아이」). 같은 줄 끝의 愛는 名詞/あい로 제대로 읽혔다 — 사전이 아니라 자리가
    문제이므로, 뒤가 공백·문장부호·줄끝이면 접두사 읽기를 버린다.
    """
    from everyric2.text.pron_style import wiki_pronunciation

    got = wiki_pronunciation("さり気ない愛 盛りすぎる愛")
    assert "마나" not in got, got
    assert got.count("아이") == 2, got


def test_katakana_particle_keeps_its_phonetic_reading():
    # 조사·조동사는 표기와 음가가 갈리는 유일한 부류라 표층 규칙에서 제외한다.
    # 가타카나로 적힌 조사까지 표층대로 읽으면 は→ワ 대립이 사라진다.
    from everyric2.text.ja_reading import _PARTICLE_POS

    assert "助詞" in _PARTICLE_POS and "助動詞" in _PARTICLE_POS


# ---------------------------------------------------------------------------
# 아라비아 숫자 + 조수사 읽기 — 실측 근거: tests/fixtures/wiki_pron_sample.json
# 115줄 중 아라비아 숫자 뒤에 조수사가 온 사례는 "1秒"(사람 발음 「이치뵤오」) 단 1건이고,
# 코퍼스(everyric2.db 4,662줄)에는 1人·2つ·10文字·100文字·10年·1000回·1秒가 있었다.
#
# 조수사인지는 이제 UniDic 태그(助数詞/助数詞可能 + 語種=漢)가 판정한다 — 예전에는
# ``_MEASURED_ARABIC_COUNTERS``가 손으로 든 목록(秒·人)이어서 3分이 「3 훈」으로 샜다.
# 그 상수는 사전이 조수사로 표시하지 않는 자리(文字)만 메우는 보충으로 남았다.
# 아래 미적용 케이스(공백이 끼거나 조수사가 없는 경우)가 새 경계를 지킨다.
# ---------------------------------------------------------------------------


def test_arabic_digit_before_a_measured_counter_reads_as_a_number():
    # 실측 그 자체(위키 사람 발음): 1秒 → 이치뵤오. UniDic은 아라비아 숫자에 읽기를
    # 주지 않아(feature.kana/pron 둘 다 빈값) 손대지 않으면 표면 "1"이 그대로 샌다.
    assert kana_reading("1秒") == "いちびょう"
    assert kana_reading("スケジュールは1秒先だって書き記していたい") == (
        "すけじゅーるはいちびょうせんだってかきしるしていたい"
    )


@pytest.mark.parametrize(
    ("digits", "expected_prefix"),
    [("0", "ゼロ"), ("3", "さん"), ("10", "じゅう"), ("24", "にじゅうよん")],
)
def test_arabic_digit_reading_scales_with_the_number(digits, expected_prefix):
    assert kana_reading(f"{digits}秒") == expected_prefix + "びょう"


def test_arabic_digit_before_a_sound_change_counter_is_now_read_too():
    # 예전에는 「1ふん」으로 숫자를 남겼다 — 촉음화·반탁음화 규칙이 없어 잘못된 음변화를
    # 만드는 것이 숫자를 남기는 것보다 나빴기 때문이다. 규칙이 생겼으므로(_counter_sandhi)
    # 이제 읽는다. 사용자에게 가장 눈에 띄는 손상이 화면에 남는 아라비아 숫자였다.
    assert kana_reading("1分") == "いっぷん"
    assert kana_reading("3分") == "さんぷん"
    assert kana_reading("10分") == "じゅっぷん"


def test_arabic_digit_needs_the_counter_immediately_adjacent():
    # 실측 형태("1秒", 사이 공백 없음)를 벗어나면 손대지 않는다.
    assert kana_reading("1 秒") == "1 びょう"


def test_arabic_year_is_read_as_a_number():
    # 年은 UniDic이 助数詞可能으로 표시하는 조수사다 — 자릿수 읽기가 그대로 이어진다.
    # (옛 테스트는 이 줄을 "조수사가 없는 경우"로 잘못 분류해 「2024ねん」을 못박고 있었다.)
    assert kana_reading("2024年") == "にせんにじゅうよんねん"
    assert kana_reading("10年後") == "じゅうねんご"


def test_standalone_arabic_digit_without_a_counter_is_unaffected():
    # 뒤에 조수사가 없으면(줄 끝, 조사, 다른 낱말) 숫자 읽기를 시도하지 않는다 —
    # 무엇으로 읽어야 하는지 알 근거가 없다(코퍼스 실측: 「0と1に還元され」).
    assert kana_reading("2024") == "2024"
    assert kana_reading("0と1に還元され") == "0と1にかんげんされ"
    assert kana_reading("1から100 いただきます") == "1から100 いただきます"


# ---------------------------------------------------------------------------
# 人(にん) 조수사 — 사용자가 실제 곡을 듣고 확인한 오류: たった1人 君に →
# 정답 「탓타 히토리 키미니」, 우리 기존 출력 「탓타 1닌 키미니」. 1・2는 자릿수
# 읽기가 아니라 딴 낱말(히토리/후타리)이 되고, 4는 よん이 아니라 よ로 줄어든다
# (十四人 → じゅうよにん). 3・5~10(끝자리 4 제외)은 자릿수 읽기 + にん을 그대로
# 이으면 맞다.
# ---------------------------------------------------------------------------


def test_one_person_and_two_people_are_irregular_words():
    assert kana_reading("たった1人　君に") == "たったひとり　きみに"
    assert kana_reading("1人") == "ひとり"
    assert kana_reading("2人") == "ふたり"


def test_three_and_up_person_counter_uses_regular_digit_plus_nin():
    assert kana_reading("3人") == "さんにん"
    assert kana_reading("5人") == "ごにん"
    assert kana_reading("10人") == "じゅうにん"


def test_four_people_uses_yo_not_yon_for_both_arabic_and_kanji_digits():
    # 四人 자체(실측 근거 데이터에는 없지만 표준 문법 — 4日・4時와 같은 불규칙)
    assert kana_reading("4人") == "よにん"
    assert kana_reading("四人") == "よにん"
    # 끝자리만 4인 복합수도 마찬가지다(十四人 → じゅうよにん, よんにん이 아니다)
    assert kana_reading("14人") == "じゅうよにん"
    assert kana_reading("十四人") == "じゅうよにん"
    assert kana_reading("二十四人") == "にじゅうよにん"


def test_kanji_one_and_two_person_were_already_correct_and_stay_untouched():
    # UniDic 사전에 一人・二人이 이미 통짜 표제어(ひとり/ふたり)로 올라 있다 —
    # 손대지 않아도 맞았다는 기존 실측을 회귀로 고정한다.
    assert kana_reading("一人") == "ひとり"
    assert kana_reading("二人") == "ふたり"
    assert kana_reading("三人") == "さんにん"


def test_person_counter_does_not_leak_into_nan_no_hito():
    # 何人(なんにん)의 何도 人 앞에서 名詞-数詞로 태깅되지만 よん으로 끝나지 않으므로
    # 손대지 않는다 — 何 관련 규칙과 겹치지 않는지 확인하는 회귀다.
    assert kana_reading("何人") == "なんにん"


# ---------------------------------------------------------------------------
# 수사 + 조수사 음변화(촉음화·반탁음화) 대조표 — 一~十 x 조수사 33개 x (한자/아라비아)
# = 660칸. **음변화는 조수사마다 다르므로 이 표가 유일한 안전망이다.** 규칙을 손볼 때
# 여기가 깨지지 않는지 보는 것 말고는 확인할 방법이 없다.
#
# 값은 사전형 표준 읽기이고, 복수 통용형은 '/'로 나열해 하나라도 맞으면 통과한다
# (四年 よねん/よんねん, 七人 しちにん/ななにん처럼 실제로 둘 다 쓰이는 자리다).
# ---------------------------------------------------------------------------

_COUNTER_MATRIX = {
    # 조수사: 一~十 정답 10칸 (같은 값이 아라비아 숫자 1~10에도 적용된다)
    "分": "いっぷん にふん さんぷん よんぷん ごふん ろっぷん ななふん はっぷん きゅうふん じゅっぷん",
    "秒": "いちびょう にびょう さんびょう よんびょう ごびょう ろくびょう ななびょう はちびょう きゅうびょう じゅうびょう",
    "時": "いちじ にじ さんじ よじ ごじ ろくじ しちじ はちじ くじ じゅうじ",
    "時間": (
        "いちじかん にじかん さんじかん よじかん ごじかん ろくじかん"
        " ななじかん/しちじかん はちじかん きゅうじかん/くじかん じゅうじかん"
    ),
    "週間": (
        "いっしゅうかん にしゅうかん さんしゅうかん よんしゅうかん ごしゅうかん"
        " ろくしゅうかん ななしゅうかん はっしゅうかん きゅうしゅうかん じゅっしゅうかん"
    ),
    "年": "いちねん にねん さんねん よねん/よんねん ごねん ろくねん ななねん はちねん きゅうねん じゅうねん",
    "回": "いっかい にかい さんかい よんかい ごかい ろっかい ななかい はっかい きゅうかい じゅっかい",
    "度": "いちど にど さんど よんど ごど ろくど ななど はちど きゅうど じゅうど",
    "個": "いっこ にこ さんこ よんこ ごこ ろっこ ななこ はっこ きゅうこ じゅっこ",
    "本": "いっぽん にほん さんぼん よんほん ごほん ろっぽん ななほん はっぽん きゅうほん じゅっぽん",
    "杯": "いっぱい にはい さんばい よんはい ごはい ろっぱい ななはい はっぱい きゅうはい じゅっぱい",
    "匹": "いっぴき にひき さんびき よんひき ごひき ろっぴき ななひき はっぴき きゅうひき じゅっぴき",
    "歳": "いっさい にさい さんさい よんさい ごさい ろくさい ななさい はっさい きゅうさい じゅっさい",
    "才": "いっさい にさい さんさい よんさい ごさい ろくさい ななさい はっさい きゅうさい じゅっさい",
    "階": "いっかい にかい さんがい/さんかい よんかい ごかい ろっかい ななかい はっかい きゅうかい じゅっかい",
    "冊": "いっさつ にさつ さんさつ よんさつ ごさつ ろくさつ ななさつ はっさつ きゅうさつ じゅっさつ",
    "枚": "いちまい にまい さんまい よんまい ごまい ろくまい ななまい はちまい きゅうまい じゅうまい",
    "人": "ひとり ふたり さんにん よにん ごにん ろくにん しちにん/ななにん はちにん きゅうにん じゅうにん",
    "名": "いちめい にめい さんめい よんめい ごめい ろくめい ななめい はちめい きゅうめい じゅうめい",
    "台": "いちだい にだい さんだい よんだい ごだい ろくだい ななだい はちだい きゅうだい じゅうだい",
    "点": "いってん にてん さんてん よんてん ごてん ろくてん ななてん はってん きゅうてん じゅってん",
    "頭": "いっとう にとう さんとう よんとう ごとう ろくとう ななとう はっとう きゅうとう じゅっとう",
    "通": "いっつう につう さんつう よんつう ごつう ろくつう ななつう はっつう きゅうつう じゅっつう",
    "着": "いっちゃく にちゃく さんちゃく よんちゃく ごちゃく ろくちゃく ななちゃく はっちゃく きゅうちゃく じゅっちゃく",
    "軒": "いっけん にけん さんげん/さんけん よんけん ごけん ろっけん ななけん はっけん きゅうけん じゅっけん",
    "件": "いっけん にけん さんけん よんけん ごけん ろっけん ななけん はっけん きゅうけん じゅっけん",
    "巻": "いっかん にかん さんかん よんかん ごかん ろっかん ななかん はっかん きゅうかん じゅっかん",
    "円": "いちえん にえん さんえん よえん/よんえん ごえん ろくえん ななえん はちえん きゅうえん じゅうえん",
    "泊": "いっぱく にはく さんぱく よんはく ごはく ろっぱく ななはく はっぱく きゅうはく じゅっぱく",
    "発": "いっぱつ にはつ さんぱつ よんはつ ごはつ ろっぱつ ななはつ はっぱつ きゅうはつ じゅっぱつ",
    "歩": "いっぽ にほ さんぽ よんほ ごほ ろっぽ ななほ はっぽ きゅうほ じゅっぽ",
    "曲": "いっきょく にきょく さんきょく よんきょく ごきょく ろっきょく ななきょく はっきょく きゅうきょく じゅっきょく",
    "つ": "ひとつ ふたつ みっつ よっつ いつつ むっつ ななつ やっつ ここのつ とお",
}

# 위 표에서 **아직 맞히지 못하는 35칸**. 전부 변경 전과 같은 값이다(악화 0) — 값을 적어
# 두므로 실수로 달라지면 이 테스트가 알려 준다. 줄이려는 사람이 왜 비워 뒀는지 알 수
# 있도록 이유를 다섯 부류로 묶어 적는다.
#
# (1) UniDic 표제어·품사 선택 — 수사+조수사 짝이 애초에 생기지 않아 어떤 규칙도 걸릴
#     자리가 없다. 八軒은 지명(名詞-固有名詞-地名/ハチケン), 八頭는 やつがしら,
#     七通는 ななとおり가 통짜 표제어로 이긴다. 「あと十分だ」의 十分은 부사 じゅうぶん
#     (충분히)으로 잡히고 — 그 문장에서는 실제로 맞는 해석일 수 있다 — 「十分」 단독은
#     じゅっぷん으로 제대로 나온다. 四分의 四는 シ로 읽혀(四分の一의 しぶん 계열) 우리
#     규칙의 꼬리 조건에 걸리지 않는다.
# (2) 語種=和 배제 — 아라비아 숫자 뒤의 巻을 UniDic이 和/マキ로 태깅한다(한자 一巻은
#     漢/カン이라 맞는다). 和語 조수사 앞에서는 수사도 和語 계열이라 한자어 자릿수 읽기를
#     붙이면 이중으로 틀리므로 語種 조건이 일부러 막는다(``_is_sino_counter``).
# (3) ``_HA_ROW_COUNTERS``에 없는 は행 조수사의 ん 자리 — 泊·発은 ん 뒤 형태(さんぱく·
#     さんぱつ)를 표에 넣지 않았다. 코퍼스에 없어 실측 근거가 없고, 표가 비어 있으면
#     기존 동작(さんはく)이 유지될 뿐 나빠지지 않는다. 一泊→いっぱく는 규칙이 맞힌다.
# (4) 조수사로 표시되지 않는 낱말 — 曲은 名詞-普通名詞-一般이라 조수사 판정에 걸리지
#     않는다. 사전이 조수사라고 말하지 않는 것을 우리가 조수사로 단정하지 않는다.
# (5) つ 계열에 10이 없다 — 현대어에서 十은 「とお」로 끝나고 十つ라는 꼴이 없다.
_COUNTER_MATRIX_MISSES = {
    # (1) 표제어·품사 선택
    ("漢", "八", "軒"): "はちけん",
    ("漢", "八", "頭"): "やつがしら",
    ("漢", "七", "通"): "ななとーり",
    ("漢", "十", "分"): "じゅーぶん",
    ("漢", "四", "分"): "しふん",
    # (2) 語種=和 배제 (아라비아 숫자 + 巻)
    **{("数", str(n), "巻"): f"{n}まき" for n in range(1, 11)},
    # (3) ん 뒤 형태를 표에 넣지 않은 は행 조수사
    ("漢", "三", "泊"): "さんはく",
    ("数", "3", "泊"): "さんはく",
    ("漢", "三", "発"): "さんはつ",
    ("数", "3", "発"): "さんはつ",
    # (4) 사전이 조수사로 표시하지 않는 낱말
    ("漢", "一", "曲"): "いちきょく",
    ("漢", "六", "曲"): "ろくきょく",
    ("漢", "八", "曲"): "はちきょく",
    ("漢", "十", "曲"): "じゅーきょく",
    **{("数", str(n), "曲"): f"{n}きょく" for n in range(1, 11)},
    # (5) つ 계열에 없는 10
    ("漢", "十", "つ"): "とーつ",
    ("数", "10", "つ"): "10つ",
}

_KANJI_SERIES = "一二三四五六七八九十"
_ARABIC_SERIES = [str(n) for n in range(1, 11)]


# UniDic이 문맥에 따라 고르는 수사 표기 변종. 四를 シ로, 七을 シチ로 주는 자리가 있다
# (四台→しだい, 七件→しちけん) — 그건 음변화 규칙이 아니라 사전의 읽기 선택이고 이
# 대조표의 대상이 아니므로 양쪽을 같은 꼴로 접어 비교한다. 수사 계열 자체의 불규칙은
# 접히면 안 되는 것이라 전용 테스트가 정확히 못박는다
# (``test_hour_counter_uses_its_own_numeral_series``: 4時 よじ / 9時 くじ).
_NUMERAL_HEAD_VARIANTS = (("しち", "なな"), ("し", "よん"), ("く", "きゅう"), ("よ", "よん"))


def _canonical_numeral_head(reading: str) -> str:
    """수사 읽기 변종을 한 꼴로 접는다 (しだい → よんだい, くじ → きゅうじ)."""
    for variant, canonical in _NUMERAL_HEAD_VARIANTS:
        if reading.startswith(canonical):
            return reading  # 이미 표준 꼴 (よんかい를 よんんかい로 만들지 않는다)
        rest = reading[len(variant) :]
        if reading.startswith(variant) and rest[:1] not in ("ゃ", "ゅ", "ょ"):
            return canonical + rest
    return reading


def _collapse_long_vowels(reading: str) -> str:
    """장음 표기 차이를 지운다 — ``ー``와 모음 반복을 같은 표시로 접는다.

    음가 경로는 장음을 ー로 적고(名 メー, 頭 トー, 週間 シューカン) 사전형은 모음을 적는다
    (めい・とう・しゅうかん). 어느 쪽이 원래 모음이었는지는 ー에서 복원할 수 없으므로
    (めー가 めい인지 めえ인지는 ー만 보고 알 수 없다 — 그게 ``pron_style._restore_ei``가
    표층 읽기를 따로 들고 다니는 이유다) **펴지 말고 접어서** 비교한다. 대조표가 보려는
    것은 음변화이고 장음 표기는 그 대상이 아니다.
    """
    out: list[str] = []
    for ch in reading:
        long_vowel = ch == "ー" or (ch in "うい" and out and out[-1] not in "ういー")
        out.append("ː" if long_vowel else ch)
    return "".join(out)


@pytest.mark.parametrize("counter", sorted(_COUNTER_MATRIX))
@pytest.mark.parametrize(("series", "label"), [(_KANJI_SERIES, "漢"), (_ARABIC_SERIES, "数")])
def test_counter_sound_changes_match_the_dictionary(counter, series, label):
    """一~十 x 조수사 = 대조표. 캐리어(あと…だ)에 넣어 표제어 충돌을 줄인다.

    음가 경로(``phonetic=True``)로 본다 — 발음 표기가 쓰는 경로가 그쪽이고, 실제로
    거기서만 나던 버그가 있었다(九時가 표층 경로에서는 くじ, 음가 경로에서는 きゅーじ).
    표층 경로는 이 대조에 쓸 수 없다: UniDic ``kana``가 四·七을 문맥에 따라 シ·シチ로
    주는 자리가 있어(四件→しけん) 음변화와 무관한 차이가 섞인다.
    """
    for numeral, want in zip(series, _COUNTER_MATRIX[counter].split()):
        text = f"あと{numeral}{counter}だ"
        got = kana_reading(text, phonetic=True)[len("あと") : -len("だ")]
        miss = _COUNTER_MATRIX_MISSES.get((label, numeral, counter))
        if miss is not None:
            assert got == miss, f"{numeral}{counter}: 알려진 미해결 칸의 값이 달라졌다"
            continue
        def norm(reading: str) -> str:
            return _collapse_long_vowels(_canonical_numeral_head(reading))

        oks = {norm(w) for w in want.split("/")}
        assert norm(got) in oks, f"{numeral}{counter}: {got} (정답 {want})"


def test_counter_sound_changes_preserve_the_mora_count():
    """음변화 치환은 글자 수를 보존해야 한다 — 모라 글자 오프셋이 토큰 길이에 걸려 있다.

    いち→いっ, ふん→ぷん, ほん→ぽん, じゅう→じゅっ 전부 같은 길이다(모라 수도 같다 —
    っ가 1박을 차지한다). 늘거나 줄면 ``reading.py``의 모라 구간이 원문 글자와 어긋나
    가라오케 음절 타이밍이 그대로 밀린다.
    """
    for text, moras in (
        ("一分", ["い", "っ", "ぷ", "ん"]),
        ("十本", ["じゅ", "っ", "ぽ", "ん"]),
        ("六匹", ["ろ", "っ", "ぴ", "き"]),
        ("三分", ["さ", "ん", "ぷ", "ん"]),
    ):
        got = text_to_moras(text)
        assert [m.kana for m in got] == moras, text
        # 모라 구간이 원문 글자 범위를 넘지 않고 순서대로 증가한다
        assert all(0 <= m.char_start < m.char_end <= len(text) for m in got), text
        assert [m.char_start for m in got] == sorted(m.char_start for m in got), text


def test_non_sino_counters_are_left_to_the_dictionary():
    """和語·외래어 조수사는 음변화 규칙 밖이다 (語種 조건).

    이 조수사들 앞에서는 수사 자체가 和語 계열로 갈리므로(三日 みっか, 一羽 ひとわ)
    한자어 촉음화를 얹으면 이중으로 틀린다. UniDic 語種이 그 갈림을 이미 적어 둔다.
    """
    assert kana_reading("三日") == "みっか"  # 촉음이 이미 수사 쪽에 있다(ミッ+カ)
    assert kana_reading("一羽") == "ひとわ"
    assert kana_reading("一組") == "いちくみ"  # いっくみ라는 꼴을 만들지 않는다
    assert kana_reading("一キロ") == "いちきろ"  # 외래어 조수사엔 촉음화가 없다
    assert kana_reading("一ページ") == "いちぺーじ"


def test_hour_counter_uses_its_own_numeral_series():
    """時는 4·7·9에서 수사가 딴 계열로 갈린다 — 아라비아 숫자를 읽기 시작하면 필요해진다.

    이 표가 없으면 4時가 「욘지」, 9時가 「큐우지」로 나온다(둘 다 오류) — 즉 숫자를
    읽는 것 자체가 새 오류를 만들게 된다.
    """
    for text, expected in (
        ("4時", "よじ"), ("7時", "しちじ"), ("9時", "くじ"),
        ("四時", "よじ"), ("七時", "しちじ"), ("九時", "くじ"),
        ("十九時", "じゅうくじ"), ("4時間", "よじかん"), ("24時間", "にじゅうよじかん"),
    ):
        assert kana_reading(text) == expected, text
    # 음가 경로에서도 같아야 한다 — 九의 음가는 キュー라 꼬리 표기가 갈린다
    assert kana_reading("九時", phonetic=True) == "くじ"
    assert kana_reading("十九時", phonetic=True) == "じゅーくじ"


def test_wago_series_counter_reads_arabic_digits_too():
    """つ는 和語 계열이라 자릿수 읽기를 못 받는다 — 계열 표가 그 자리를 메운다.

    실측: 코퍼스 4줄(「どうしようもなく2つに裂けた心内環境を」)에서 「2츠」로 숫자가
    화면에 남았다. 한자 표기 쪽 오독(四つ→よんつ 등)도 같이 사라진다.
    """
    assert kana_reading("2つに裂けた") == "ふたつにさけた"
    for text, expected in (
        ("1つ", "ひとつ"), ("四つ", "よっつ"), ("六つ", "むっつ"), ("八つ", "やっつ"),
        ("一つ", "ひとつ"), ("三つ", "みっつ"),
    ):
        assert kana_reading(text) == expected, text


# ---------------------------------------------------------------------------
# 何(なに/なん) — 사용자가 실제 곡을 듣고 확인한 오류: 何を含んでたって →
# 정답 「나니오 후쿤데탓테」, 우리 기존 출력 「난오 후쿤데탓테」. 뒤에 격조사
# (が・を・に)가 바로 오면 なに, 그 외(관용구·조수사 앞)는 UniDic 기본값 なん을
# 그대로 둔다.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("何を含んでたって", "なにをふくんでたって"),
        ("何が", "なにが"),
        ("何を", "なにを"),
        ("何に", "なにに"),
    ],
)
def test_nani_before_a_case_particle(text, expected):
    assert kana_reading(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("何で", "なんで"),
        ("何て", "なんて"),
        ("何人", "なんにん"),
        ("何度", "なんど"),
        ("何回", "なんかい"),
        ("何時", "なんじ"),
        ("何とか", "なんとか"),
        ("何の", "なんの"),  # 何が만큼 확고한 なに 대립이 없다 — なんの가 표준이다
        ("何と", "なんと"),  # 何とか・何と言った와 태그가 같아 なに/なん을 못 가른다
    ],
)
def test_nan_stays_before_fixed_idioms_and_counters(text, expected):
    assert kana_reading(text) == expected


# ---------------------------------------------------------------------------
# 私(わたし/わたくし) — 사용자가 실제 곡을 듣고 확인한 오류: 私は → 정답
# 「와타시와」, 우리 기존 출력 「와타쿠시와」. UniDic 사전은 わたくし를 1순위로
# 주지만 가사에서는 わたし가 압도적이다.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("私は", "わたしは"),
        ("私たちは", "わたしたちは"),
        ("私の願いは", "わたしのねがいは"),
    ],
)
def test_watashi_is_the_default_reading_of_shi(text, expected):
    assert kana_reading(text) == expected


def test_shi_reading_compounds_are_not_affected_by_the_watashi_override():
    # 私事・私見・私立・私鉄처럼 し로 읽는 복합어는 UniDic이 통째로 한 표제어로
    # 묶어 내려주므로(surface가 "私" 한 글자가 아니다) わたし로 잘못 바뀌지 않는다.
    assert kana_reading("私鉄") == "してつ"
    assert kana_reading("私立") == "しりつ"
    assert "わたし" not in kana_reading("私鉄")
    assert "わたし" not in kana_reading("私立")
