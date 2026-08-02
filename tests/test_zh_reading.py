"""중국어 한자 → 병음·한글·가나 3표기 변환(everyric2.text.zh_reading) 회귀 테스트.

기대값의 근거는 표기 관례다: 한글은 국립국어원 외래어 표기법의 실제 용례(베이징·상하이·
구이린·류더화), 병음은 ``pypinyin``의 문맥 판정, 가나는 일본 매체 다수 표기다. 읽기(다음자)
와 표기(관례)를 가르는 것이 이 모듈의 설계라 테스트도 둘을 따로 못박는다.
"""
import pytest

from everyric2.text.align_target import join_display
from everyric2.text.zh_reading import (
    derive_zh_display_units,
    read_line,
    split_syllable,
    syllable_to_hangul,
    syllable_to_kana,
    zh_pron_variants,
    zh_to_pinyin,
)


# ---------------------------------------------------------------------------
# 읽기 — pypinyin이 맡는 몫(다음자·성조 변화)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,expected", [
    # 같은 글자가 낱말에 따라 갈린다 — 글자 단위로 읽으면 전부 한쪽으로 무너진다
    ("重要", "zhòng yào"), ("重复", "chóng fù"),
    ("长江", "cháng jiāng"), ("长大", "zhǎng dà"),
    ("银行", "yín háng"), ("行走", "xíng zǒu"),
    ("爱好", "ài hào"), ("好人", "hǎo rén"),
    ("还是", "hái shì"), ("归还", "guī huán"),
    # 不·一의 성조 변화도 pypinyin이 적용한다(bù→bú, yī→yí)
    ("不会", "bú huì"), ("一个", "yí gè"),
])
def test_pinyin_reads_heteronyms_in_context(text, expected):
    assert zh_to_pinyin(text) == expected


def test_heteronyms_split_within_one_line():
    # 한 줄 안에서 같은 글자가 두 번 다르게 읽히는 경우 — 런 전체를 한 번에 넘기는 근거
    assert zh_to_pinyin("重要的重复") == "zhòng yào de chóng fù"


def test_tone_option_strips_marks_only_from_pinyin():
    marked = zh_pron_variants("中国人")
    plain = zh_pron_variants("中国人", tone="none")
    assert marked["romaji"] == "zhōng guó rén"
    assert plain["romaji"] == "zhong guo ren"
    # 성조는 병음에만 적을 자리가 있다 — 나머지 두 표기는 옵션에 흔들리지 않는다
    assert marked["hangul"] == plain["hangul"] == "중궈런"
    assert marked["kana"] == plain["kana"]


# ---------------------------------------------------------------------------
# 한글 — 국립국어원 외래어 표기법
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,expected", [
    ("北京", "베이징"), ("上海", "상하이"), ("中国", "중궈"),
    ("天安门", "톈안먼"), ("毛泽东", "마오쩌둥"), ("重庆", "충칭"),
    ("长江", "창장"), ("刘德华", "류더화"), ("贵州", "구이저우"),
])
def test_hangul_matches_standard_transcriptions(text, expected):
    assert zh_pron_variants(text)["hangul"] == expected


@pytest.mark.parametrize("syllable,expected", [
    # 'ㅈ·ㅉ·ㅊ' 뒤 이중모음은 단모음으로 — 江은 쟝이 아니라 장
    ("jiāng", "장"), ("jiā", "자"), ("jiè", "제"), ("qián", "첸"), ("jiǔ", "주"),
    ("jiǒng", "중"), ("zhāo", "자오"),
    # 초성이 세칙 대상이 아니면 이중모음이 그대로 산다
    ("tiān", "톈"), ("xià", "샤"), ("xiè", "셰"), ("biǎo", "뱌오"), ("liàng", "량"),
])
def test_hangul_collapses_glide_only_after_j_q_ch(syllable, expected):
    assert syllable_to_hangul(syllable) == expected


@pytest.mark.parametrize("syllable,expected", [
    # 성모가 있으면 uei·uen의 가운데 모음이 죽는다(桂 구이, 昆 쿤)
    ("guì", "구이"), ("cuī", "추이"), ("huì", "후이"), ("shuǐ", "수이"),
    ("kūn", "쿤"), ("chūn", "춘"),
    # 성모가 없으면 기본형(魏 웨이, 温 원)
    ("wèi", "웨이"), ("wēn", "원"),
])
def test_hangul_uei_uen_depend_on_initial(syllable, expected):
    assert syllable_to_hangul(syllable) == expected


@pytest.mark.parametrize("syllable,expected", [
    # 권설음·설치음 뒤의 i는 [i]가 아니다 — 한글은 둘 다 ㅡ
    ("zhī", "즈"), ("chī", "츠"), ("shí", "스"), ("rì", "르"),
    ("zì", "쯔"), ("cì", "츠"), ("sì", "쓰"),
])
def test_hangul_apical_vowel(syllable, expected):
    assert syllable_to_hangul(syllable) == expected


@pytest.mark.parametrize("syllable,expected", [
    ("lǜ", "뤼"), ("nǚ", "뉘"), ("xué", "쉐"), ("quán", "취안"), ("jūn", "쥔"),
    ("yú", "위"), ("yuàn", "위안"),
])
def test_hangul_u_umlaut_finals(syllable, expected):
    assert syllable_to_hangul(syllable) == expected


# ---------------------------------------------------------------------------
# 가나 — 일본 매체 다수 표기의 근사
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,expected", [
    ("上海", "シャンハイ"), ("北京", "ベイジン"), ("中国", "ジョングオ"),
    ("天安门", "ティエンアンメン"), ("重庆", "チョンチン"), ("桂林", "グイリン"),
    ("昆明", "クンミン"),
])
def test_kana_matches_common_japanese_transcriptions(text, expected):
    assert zh_pron_variants(text)["kana"] == expected


@pytest.mark.parametrize("syllable,expected", [
    # 권설 -i는 イ단 장음, 설치 -i는 ウ단 장음 — 한글에선 합류하지만 가나는 갈린다
    ("zhī", "ジー"), ("shí", "シー"), ("sì", "スー"), ("cì", "ツー"),
    # ü계는 拗音 ュ로 근사
    ("xú", "シュイ"), ("lǜ", "リュイ"), ("xué", "シュエ"),
    # 성모 없는 y·w 철자는 활음을 살린 관례 표기(イア가 아니라 ヤ)
    ("yǎng", "ヤン"), ("wǒ", "ウオ"), ("yuè", "ユエ"),
])
def test_kana_special_finals(syllable, expected):
    assert syllable_to_kana(syllable) == expected


def test_kana_merges_n_and_ng():
    # 일본어 음운으로는 가를 수단이 없다 — 한계를 못박아 둔다(모듈 docstring)
    assert syllable_to_kana("shān") == syllable_to_kana("shāng") == "シャン"
    # 같은 자리에서 한글은 갈린다
    assert syllable_to_hangul("shān") == "산"
    assert syllable_to_hangul("shāng") == "상"


# ---------------------------------------------------------------------------
# 음절 분해
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("syllable,expected", [
    ("zhōng", ("zh", "ong")),      # 긴 성모부터 봐야 z로 잘리지 않는다
    ("jiǔ", ("j", "iou")),          # 축약 철자 iu → iou
    ("guī", ("g", "uei")),          # ui → uei
    ("kūn", ("k", "uen")),          # un → uen
    ("jūn", ("j", "ün")),           # j 뒤의 u는 실제로 ü라 un → ün(uen 아님)
    ("xué", ("x", "üe")),
    ("wǒ", ("", "uo")),             # 성모 없는 w 철자 → 원래 운모
    ("yīng", ("", "ing")),
    ("ér", ("", "er")),
    ("shì", ("sh", "-i")),          # 권설 뒤 i
    ("zì", ("z", "-iz")),           # 설치 뒤 i
])
def test_split_syllable(syllable, expected):
    assert split_syllable(syllable) == expected


def test_split_syllable_accepts_v_spelling():
    # lazy_pinyin 기본 출력(lv)도 받는다
    assert split_syllable("lv") == ("l", "ü")


# ---------------------------------------------------------------------------
# 줄 단위 — 혼합 표기와 표시 조립
# ---------------------------------------------------------------------------


def test_representative_lyric_lines():
    # 邓丽君「月亮代表我的心」 — 경성(的 de)과 üe(月 yuè)가 함께 있는 줄
    assert zh_pron_variants("月亮代表我的心") == {
        "hangul": "웨량다이뱌오워더신",
        "kana": "ユエリアンダイビアオウオドゥーシン",
        "romaji": "yuè liàng dài biǎo wǒ de xīn",
    }
    # 周杰伦「稻香」 — 还(hái/huán)이 문맥으로 갈리는 줄
    assert zh_pron_variants("还记得你说家是唯一的城堡")["romaji"] == (
        "hái jì de nǐ shuō jiā shì wéi yī de chéng bǎo"
    )
    assert zh_pron_variants("还记得你说家是唯一的城堡")["hangul"] == "하이지더니숴자스웨이이더청바오"


def test_pinyin_separates_syllables_but_hangul_and_kana_do_not():
    # 병음은 붙이면 못 읽는다(wǒàinǐ) — 표기마다 조립 규칙이 다르다는 계약
    variants = zh_pron_variants("我爱你")
    assert variants["romaji"] == "wǒ ài nǐ"
    assert variants["hangul"] == "워아이니"
    assert variants["kana"] == "ウオアイニー"


def test_mixed_line_keeps_latin_and_passes_through_other_scripts():
    variants = zh_pron_variants("我 love you")
    assert variants["romaji"] == "wǒ love you"
    # 라틴 낱말은 en 곡과 같은 도구로 음차하고, 낱말 둘레는 띄운다
    assert variants["hangul"] == "워 럽 유"
    # 한자가 없는 조각(숫자·구두점)은 원형 그대로 실린다
    assert zh_pron_variants("2024年")["hangul"] == "2024녠"
    assert zh_pron_variants("还好吗？")["hangul"] == "하이하오마？"


def test_erhua_stays_a_separate_syllable():
    # 儿을 앞 음절에 합치면 소유자 배열이 원문 글자와 어긋난다(모듈 docstring)
    pieces = read_line("花儿")
    assert [p.source for p in pieces] == ["花", "儿"]
    assert zh_pron_variants("花儿")["romaji"] == "huā ér"
    assert zh_pron_variants("花儿")["hangul"] == "화얼"


@pytest.mark.parametrize("text", ["", "   ", "hello", "안녕하세요", "！？", "我爱你", "我 love you 啊"])
def test_display_units_own_every_source_char(text):
    units = derive_zh_display_units(text)
    # 정렬 타깃은 원문 그대로(zh CTC 모델의 vocab이 한자다)
    assert units.target == text
    for key, owners in units.owners.items():
        assert len(owners) == len(text), key
    assert len(units.word_end) == len(text)


def test_display_units_owners_are_one_per_han_char():
    units = derive_zh_display_units("我爱你")
    assert units.owners["hangul"] == ["워", "아이", "니"]
    assert units.owners["romaji"] == ["wǒ", "ài", "nǐ"]
    # 한자만 있는 줄에는 낱말 경계가 없다
    assert units.word_end == [False, False, False]


def test_display_units_mark_word_boundaries_for_latin_and_spaces():
    units = derive_zh_display_units("我 love you")
    # join_display로 이어도 낱말이 붙지 않는다(한글·가나 표기 기준)
    assert join_display(units.owners["hangul"], units.word_end) == "워 럽 유"
    # 라틴 낱말의 표시는 철자 길이에 비례해 나눈다 — 통째로 첫 글자에 몰면 그 낱말
    # 구간에서만 카라오케가 멈춘다(align_target._distribute_by_length의 실측 근거)
    units = derive_zh_display_units("我 beautiful 啊")
    word = units.owners["hangul"][2:11]
    assert "".join(word) == "비어티펄"
    assert len([slot for slot in word if slot]) == 4
