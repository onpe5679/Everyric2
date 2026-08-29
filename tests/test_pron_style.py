"""결정론적 한글 발음 표기(everyric2.text.pron_style) 회귀 테스트.

no-mock: fugashi + unidic-lite 실제 분석 결과를 그대로 쓴다. 기준 데이터는
``tests/fixtures/wiki_pron_sample.json`` — 보카로 가사 위키의 **사람이 쓴** 발음 115줄이며
조사·장음·촉음·루비·긴 줄·한자 많은 줄이 골고루 섞이도록 뽑은 표본이라 코퍼스 평균보다
어렵다(코퍼스 2,207줄 전체 완전일치율은 81.9%였다).

회귀 기준은 목표치를 박아 두는 대신 **실측값에서 3%p 낮은 값**을 하한으로 쓴다 —
읽기 엔진(UniDic)이나 규칙이 바뀌어 눈에 띄게 나빠지는 것만 잡고, 위키 표기 자체의
비일관성(같은 곡에서 ように를 띄우기도 붙이기도 한다)까지 테스트로 못 박지 않는다.
"""
import json
from pathlib import Path

import pytest

from everyric2.text.ja_reading import ReadingToken, kana_reading, tokenize_reading
from everyric2.text.pron_style import wiki_pronunciation
from everyric2.text.reading import align_pron_to_moras, text_to_moras

_FIXTURE = Path(__file__).parent / "fixtures" / "wiki_pron_sample.json"

# 실측(2026-07, unidic-lite): 완전일치 63/115 = 54.8%, 공백무시 83/115 = 72.2%.
# 하한은 그보다 3%p 낮게 잡는다. 완전일치와 공백무시의 격차(17.4%p)는 전부 띄어쓰기
# 불일치이고, 위키 쪽이 서로 모순되는 줄들이라(〜ている을 붙이는 곡과 띄우는 곡이 섞여
# 있다) 어떤 규칙으로도 동시에 맞출 수 없다.
#
# 참고: 같은 구현을 코퍼스 전량(라틴 제외 2,207줄)으로 재측정하면 공백·부호 무시 78.6%,
# 장음 정규화 80.0%, えい·ふぃ 정규화 82.7%다. 이 표본은 조사·장음·촉음·루비·긴 줄이
# 골고루 섞이도록 일부러 어렵게 뽑은 115줄이라 코퍼스 평균보다 낮게 나온다.
_MIN_EXACT_RATIO = 0.517
_MIN_LOOSE_RATIO = 0.691

# 파이프라인이 독음 정렬을 채택하는 품질 임계 (reading._QUALITY_THRESHOLD와 같은 값)
_QUALITY_THRESHOLD = 0.6


@pytest.mark.parametrize(
    ("text", "expected", "forbidden"),
    [
        ("何一つ解らなくても", "나니 히토츠", "난 히토츠"),
        ("その内声も届かなくなって", "소노 우치코에", "나이세이"),
        ("身体 君のこと", "신타이 키미노", "쿤노"),
    ],
)
def test_live_report_context_readings(text: str, expected: str, forbidden: str) -> None:
    rendered = wiki_pronunciation(text)
    assert expected in rendered
    assert forbidden not in rendered


@pytest.fixture(scope="module")
def wiki_pairs() -> list[dict]:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))["pairs"]


# ---------------------------------------------------------------------------
# 1. 회귀 기준 — 표본 전체 일치율
# ---------------------------------------------------------------------------


def test_matches_human_wiki_pronunciation_at_the_measured_rate(wiki_pairs, capsys):
    exact = loose = 0
    for pair in wiki_pairs:
        got = wiki_pronunciation(pair["text"])
        want = pair["pron"]
        exact += got == want
        loose += got.replace(" ", "") == want.replace(" ", "")

    total = len(wiki_pairs)
    with capsys.disabled():
        print(
            f"\nwiki_pronunciation vs human wiki ({total} lines): "
            f"exact {exact}/{total} = {exact / total * 100:.1f}%, "
            f"ignoring spaces {loose}/{total} = {loose / total * 100:.1f}%"
        )

    assert exact / total >= _MIN_EXACT_RATIO
    assert loose / total >= _MIN_LOOSE_RATIO


def test_every_japanese_line_produces_a_pronunciation(wiki_pairs):
    # 한 줄이라도 빈 발음이 나오면 그 줄은 클라이언트에서 독음이 통째로 빠진다
    assert all(wiki_pronunciation(pair["text"]) for pair in wiki_pairs)


def test_output_carries_no_kana_or_kanji(wiki_pairs):
    # 가나가 새면 서버의 독음 품질 가드(bad_pron_indices)가 라인을 버린다
    leaked = [
        pair["text"]
        for pair in wiki_pairs
        if any("぀" <= ch <= "鿿" for ch in wiki_pronunciation(pair["text"]))
    ]
    assert leaked == []


def test_only_text_with_nothing_to_read_yields_an_empty_string():
    # 이 테스트의 옛 이름은 test_non_japanese_text_yields_empty_string이었고 "hello world"가
    # ""라고 단언했다. 그건 "라틴은 음차하지 않는다"는 옛 방침의 기록이다 — 그 방침이 실측으로
    # 뒤집혀(라틴 줄의 정렬 conf가 1/10) 이제 라틴은 조밀 음차된다. 라틴 100% 줄까지 음차하는
    # 이유는 발음이 비면 그 줄이 독음(ko) 정렬에 아예 들어가지 못하기 때문이다.
    assert wiki_pronunciation("hello world") == "헬로 월"
    assert wiki_pronunciation("") == ""
    assert wiki_pronunciation("오늘 밤") == ""  # 한글은 읽을 것이 없다
    assert wiki_pronunciation("1234 (56)") == ""  # 숫자·부호만 있는 줄도 마찬가지


def test_output_is_reproducible(wiki_pairs):
    # LLM 경로의 핵심 문제였던 '같은 줄을 실행마다 다르게 읽는' 현상이 없어야 한다
    for pair in wiki_pairs[:20]:
        assert wiki_pronunciation(pair["text"]) == wiki_pronunciation(pair["text"])


# ---------------------------------------------------------------------------
# 2. 규칙별 단위 테스트
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("君は王女", "키미와 오오조"),  # は는 음가(ワ) — LLM은 표층대로 "하"라고 쓴다
        ("こっちは暗い", "콧치와 쿠라이"),
        ("波が音を立てる", "나미가 오토오 타테루"),  # を도 음가(オ)
    ],
)
def test_particles_use_their_phonetic_reading(text, expected):
    assert wiki_pronunciation(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("鮮明", "센메이"),  # kana=センメイ / pron=センメー → 3번째가 원래 イ
        ("平穏", "헤이온"),  # kana=ヘイオン / pron=ヘーオン
        ("休憩", "큐우케이"),  # kana=キュウケイ — ウ는 그대로, イ만 되살린다
        ("不安定", "후안테이"),
        ("平和", "헤이와"),
        ("発生", "핫세이"),
        ("綺麗", "키레이"),
        ("星間", "세이칸"),
    ],
)
def test_ei_is_written_as_ei_not_ee(text, expected):
    assert wiki_pronunciation(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ねーすきすき？", "네에 스키 스키?"),  # kana=ネー — えい가 아니라 진짜 장음
        ("かっけー", "캇케에"),  # ー가 별개 토큰(補助記号)이라 되살릴 근거가 없다
        ("誕生日(バースデー)", "(바아스데에)"),  # kana=バースデー에 이미 ー가 있다
        # クレイジー(crazy)의 レイ는 진짜 ei다 — 한글 관습도 「크레이지」다. 이 기댓값은
        # 원래 「쿠레에지이」였는데, 그건 언어적 사실이 아니라 옛 가드의 한계를 적어 둔 것이었다
        # ("표층 어디에든 ー가 있으면 손대지 않는다" → 끝의 ジー 때문에 앞의 レイ까지 포기).
        # 이제 가타카나 토큰은 사전을 조회하지 않고 표층을 그대로 읽으므로(ja_reading의
        # _katakana_as_hiragana) レイ가 살아난다. 같은 목록의 진짜 장음(ゲーム·コーヒー·
        # バースデー)은 표층에 ー가 그대로 있어 영향을 받지 않는다 — 아래 항목들이 그것을 지킨다.
        ("クレイジー", "쿠레이지이"),
        ("ゲーム", "게에무"),
        ("コーヒー", "코오히이"),
    ],
)
def test_real_long_vowels_are_not_turned_into_ei(text, expected):
    """진짜 장음을 "이"로 바꾸면 안 된다.

    한글 단계에서 "ㅔ/ㅖ + 받침 없는 음절 뒤의 에"를 바꾸던 앞선 구현의 오작동이다
    (네이/캇케이/바아스데이). 이제 표층 읽기와 맞춰 보고 판정하므로 여기 걸리지 않는다.
    """
    assert wiki_pronunciation(text) == expected


@pytest.mark.parametrize(
    ("reading", "surface_reading", "expected"),
    [
        ("せんめー", "せんめい", "せんめい"),  # 되살린다
        ("きゅーけー", "きゅうけい", "きゅーけい"),  # ウ 자리는 건드리지 않는다
        ("ばーすでー", "ばーすでー", "ばーすでー"),  # 표층에도 ー가 있으면 손대지 않는다
        ("ねー", "ねー", "ねー"),
        ("めー", "めえ", "めー"),  # 표층이 え면 원래 えい가 아니다
        ("さけー", "さけいる", "さけー"),  # 길이 불일치 — 자리를 못 믿는다
        ("かっけ", "", "かっけ"),  # 표층 읽기 없음(리터럴·폴백)
    ],
)
def test_restore_ei_guards(reading, surface_reading, expected):
    from everyric2.text.pron_style import _restore_ei

    assert _restore_ei(reading, surface_reading) == expected


def test_ei_restoration_is_scoped_to_one_token():
    # 판정은 토큰 안에서만 한다 — 앞 토큰의 ㅔ가 뒷 토큰 첫 "에"를 끌어당기지 않는다
    assert wiki_pronunciation("捨てて えらい") == "스테테 에라이"
    assert wiki_pronunciation("捨てて 描く") == "스테테 에가쿠"
    assert wiki_pronunciation("どうですか？そうだ！ねえ、") == "도오데스카? 소오다! 네에,"


def test_fi_and_fe_follow_the_wiki_spelling():
    # 위키 관례: ふぃ/ふぇ는 휘/훼 (kana_hangul의 표준 표기는 피/페다)
    assert wiki_pronunciation("カタストロフィー") == "카타스토로휘이"
    assert "휘" in wiki_pronunciation("フィルム")


def test_ruby_reading_wins_over_the_kanji_reading():
    # 涙（シル）는 "나미다"가 아니라 작사가가 지정한 "시루"로 불린다.
    # 괄호는 위치를 유지한 채 반자로 적는다(위키도 그렇게 쓴다).
    assert wiki_pronunciation("涙（シル）をこぼす") == "(시루)오 코보스"
    assert wiki_pronunciation("帰る動画（トコ）は既に廃墟") == "카에루 (토코)와 스데니 하이쿄"


# ---------------------------------------------------------------------------
# 사용자가 실제 곡을 듣고 확인한 오류 3건 (everyric2.text.ja_reading의 人・何・私
# 낱말 기본값 수정 회귀) — 원문 그대로 위키 표기 출력을 검증한다.
# ---------------------------------------------------------------------------


def test_person_counter_word_reading_matches_the_song():
    # たった1人 君に → 「탓타 히토리 키미니」(기존 오류: 「탓타 1닌 키미니」)
    assert wiki_pronunciation("たった1人　君に") == "탓타 히토리 키미니"


def test_nani_before_case_particle_matches_the_song():
    # 何を含んでたって → 「나니오 후쿤데탓테」(기존 오류: 「난오 후쿤데탓테」)
    assert wiki_pronunciation("何を含んでたって") == "나니오 후쿤데탓테"


def test_nanika_before_case_particle_matches_the_song():
    # 何かを攫う → 「나니카오 사라우」(기존 오류: 「난카오」 — 사용자 청취 -tKVN2mAKRI,
    # 2026-07-28). 何+か+격조사는 부정칭 대명사 なにか임이 표기로 확정되는 무결한 부분집합이다.
    assert wiki_pronunciation("何かを攫う") == "나니카오 사라우"
    assert wiki_pronunciation("何かが違う") == "나니카가 치가우"


def test_nanika_without_case_particle_keeps_nan():
    # 격조사가 안 따라오면 부사적 용법(회화체 なんか)과 못 가르므로 기본값 なん을 지킨다.
    # 관용구 축(何とか)도 か 뒤에 격조사가 없어 불변이다 — 규칙 확장의 안전 경계.
    assert wiki_pronunciation("何か違う") == "난카 치가우"
    assert wiki_pronunciation("何とかなるさ") == "난토카 나루사"


def test_watashi_default_matches_the_song():
    # 私は → 「와타시와」(기존 오류: 「와타쿠시와」). 같은 곡의 私たちは・私の願いは도.
    assert wiki_pronunciation("私は") == "와타시와"
    assert wiki_pronunciation("私たちは") == "와타시타치와"
    assert wiki_pronunciation("私の願いは") == "와타시노 네가이와"


def test_phrase_spacing_follows_bunsetsu():
    # 조사·조동사는 앞 내용어에 붙고, 내용어에서 새 문절이 시작한다
    assert (
        wiki_pronunciation("叫んだ音は既に列を成さないで")
        == "사켄다 오토와 스데니 레츠오 나사나이데"
    )


def test_auxiliary_verbs_stay_in_the_same_phrase():
    # 補助動詞(〜ている)와 サ変동사(명사+する)는 부속어 — 문절을 가르지 않는다
    assert wiki_pronunciation("ここで君を待っている") == "코코데 키미오 맛테이루"
    assert wiki_pronunciation("かわいい顔で告白するよ") == "카와이이 카오데 코쿠하쿠스루요"


def test_prefix_binds_to_the_following_word():
    assert wiki_pronunciation("お母さん").startswith("오")
    assert " " not in wiki_pronunciation("お馬さん")


def test_numeral_and_counter_are_one_phrase():
    """수사 + 조수사는 한 문절이다 — 사이를 띄우면 안 된다.

    사용자 제보: 1秒 → 「이치 뵤오」(목표 「이치뵤오」). 위키 사람 발음도 붙여 쓴다
    (fixtures/wiki_pron_sample.json: 「1秒先だって」→「이치뵤오 사키닷테」,
    「ひとつふたつ白く」→「히토츠 후타츠 시로쿠」).

    조수사의 절반이 UniDic에서 접미사가 아니라 名詞-普通名詞-助数詞可能이라 그대로 두면
    문절이 갈라진다(``pron_style._COUNTER_POS3``). 코퍼스 실측으로 나온 줄들이다.
    """
    assert wiki_pronunciation("1秒ごとに 崩れていく世界") == "이치뵤오고토니 쿠즈레테이쿠 세카이"
    assert wiki_pronunciation("一度まっさらにしてほしいから") == "이치도 맛사라니 시테 호시이카라"
    assert wiki_pronunciation("何度でも送るよ") == "난도데모 오쿠루요"
    assert wiki_pronunciation("何回だってさ") == "난카이닷테사"
    assert wiki_pronunciation("これからも この先何年も") == "코레카라모 코노 사키 난넨모"
    assert wiki_pronunciation("百点満点は今だけしか") == "햐쿠텐 만텐와 이마다케시카"
    # 접미사로 태깅되는 조수사는 원래 붙어 있었다 — 회귀 감시
    assert wiki_pronunciation("たった1人　君に") == "탓타 히토리 키미니"
    assert wiki_pronunciation("『これ一つ いくらでしょう』") == "『코레 히토츠 이쿠라데쇼오』"


def test_counter_nouns_only_bind_right_after_a_numeral():
    """助数詞可能은 "조수사로 쓰일 **수도** 있는 명사"라는 뜻이다 — 수사 뒤가 아니면 안 붙인다.

    이 조건을 빼고 品詞만 보면 度·回·年·時間처럼 홀로도 쓰는 명사가 앞말에 붙어 정상 줄의
    띄어쓰기가 무너진다. 앞 토큰이 数詞일 때만 붙이는 것이 그 방어다.
    """
    from everyric2.text.pron_style import _starts_phrase

    counter = ReadingToken("回", "かい", 0, 1, pos="名詞", pos2="普通名詞", pos3="助数詞可能")
    assert _starts_phrase(counter, "名詞", "数詞") is False  # 수사 뒤 — 한 문절
    assert _starts_phrase(counter, "名詞", "普通名詞") is True  # 보통명사 뒤 — 갈라진다
    assert _starts_phrase(counter, "動詞", "一般") is True
    # 조수사가 아닌 명사(助数詞可能이 아닌 것)는 수사 뒤에서도 갈라진다
    plain = ReadingToken("世界", "せかい", 0, 2, pos="名詞", pos2="普通名詞", pos3="一般")
    assert _starts_phrase(plain, "名詞", "数詞") is True
    assert wiki_pronunciation("今度は僕が行く") == "콘도와 보쿠가 이쿠"
    assert wiki_pronunciation("言葉の意味 何回も") == "코토바노 이미 난카이모"


def test_original_whitespace_becomes_a_phrase_boundary():
    assert wiki_pronunciation("痛み 痛み 見えない") == "이타미 이타미 미에나이"
    assert wiki_pronunciation("乾いた心臓の音　淡いうわ言") == (
        "카와이타 신조오노 오토 아와이 우와고토"
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("だ！よ！ね！チュ！", "다! 요! 네! 추!"),
        ("どんな味ですか？", "돈나 아지데스카?"),
        ("体、蝕む", "카라다, 무시바무"),
    ],
)
def test_fullwidth_punctuation_is_normalized_to_halfwidth(text, expected):
    assert wiki_pronunciation(text) == expected


def test_opening_brackets_attach_to_the_next_phrase():
    assert wiki_pronunciation("(ぴょんぴょん)") == "(푠푠)"
    assert wiki_pronunciation("「きっとだれにもわからないさ」") == (
        "「킷토 다레니모 와카라나이사」"
    )


def test_trailing_punctuation_is_not_dropped():
    # 라인 끝의 닫는 괄호를 흘리면 원문 문자가 사라진다
    assert wiki_pronunciation("「だめだ！」").endswith("!」")


def test_latin_is_transliterated_and_digits_before_a_counter_are_read_aloud():
    """라틴은 음차하고, 조수사가 뒤따르는 숫자는 소리내어 읽는다.

    옛 기댓값은 ``"numb" in wiki_pronunciation("numb な僕")``였다 — "위키는 음차하지만
    (numb→넘) 규칙화가 불가능하다"는 방침의 기록이다. 실측이 그 방침을 뒤집었다: 라틴을
    남기면 그 줄이 정렬되지 않고(라틴 글자 conf<0.01이 90~99%), 사람 자막 대조에서 잔차
    p90이 0.629s→0.170s로 줄었다. 자세한 수치는 ``everyric2.text.latin_hangul``에 있다.

    숫자도 같은 방향으로 정리됐다. 예전에는 실측된 조수사(秒·人) 뒤에서만 자릿수로 읽고
    ``1分先``은 「1분 사키」처럼 숫자를 남겼다 — 촉음화·반탁음화 규칙이 없어 「いちふん」을
    만드는 것보다 숫자를 남기는 편이 나았기 때문이다. 이제 그 규칙이 있으므로
    (``ja_reading._counter_sandhi``) 조수사 뒤 숫자는 모두 읽는다.
    """
    got = wiki_pronunciation("---深刻なエラーが発生しました---")
    assert got == "---신코쿠나 에라아가 핫세이시마시타---"  # 부호만 있는 구간은 그대로
    assert wiki_pronunciation("numb な僕") == "넘 나 보쿠"
    assert "이치" in wiki_pronunciation("1秒先")
    assert wiki_pronunciation("1分先") == "잇푼 사키"  # 숫자가 화면에 남지 않는다
    # 조수사가 없는 숫자는 여전히 그대로다 — 읽을 근거가 없다
    assert wiki_pronunciation("0と1に還元され") == "0토 1니 칸겐사레"


def test_ampersand_is_the_latin_word_and_not_a_silent_mark():
    """&(앤드)는 부호 묵음이 아니라 라틴 낱말 "and"로 렌더된다(사용자 실청취 피드백, 2026-07).

    &를 「あんど」/「と」 심판 후보로 만드는 첫 시도는 걷어냈다 — 실가창을 들어 보면
    안도/토가 아니라 앤이었다. 그래서 이제 기존 라틴 경로(latin_hangul)를 그대로 탄다:
    hangul은 조밀 음차(앤), romaji·kana는 라틴을 원문 그대로 통과시키는 기존 관례를
    따라 "and"가 그대로 남는다. 여러 &가 있어도 각자 자기 자리에서 렌더된다(치환 위치).
    """
    assert wiki_pronunciation("君&僕") == "키미 앤 보쿠"
    assert wiki_pronunciation("君＆僕") == "키미 앤 보쿠"  # 전각도 반각과 동일하게 렌더된다
    assert wiki_pronunciation("君&僕&友") == "키미 앤 보쿠 앤 토모"  # 여럿이면 각자 자기 자리에
    assert wiki_pronunciation("君&僕", script="romaji") == "kimi and boku"
    assert wiki_pronunciation("君&僕", script="kana") == "きみ and ぼく"


def test_ampersand_romaji_line_gives_and_its_own_mora_and_timing():
    """romaji_line도 wiki_pronunciation(script="romaji")과 같은 값을 내야 한다.

    reading.text_to_moras가 &/＆를 ASCII 유닛으로 인식하도록 고치기 전에는(reading.py
    수정, 2026-07) &가 모라를 아예 못 만들어 romaji_line이 "kimi boku"처럼 and를
    빠뜨렸다 — moras/spaces 목록과 표시 문자열이 단일 소스라는 불변식은 지켜졌지만
    "and" 자체가 증발했다. 이제 &가 자기 모라를 갖고, kana_romaji._MORA_ROMAJI가
    그 모라를 "and"로 환전한다.
    """
    from everyric2.text.pron_style import romaji_line

    display, moras, spaces = romaji_line("君&僕")
    assert display == "kimi and boku"
    assert "and" in moras
    assert "".join(m + (" " if s else "") for m, s in zip(moras, spaces)).strip() == display

    # 전각도 동일, 여럿이면 각자 자기 자리에서 "and"로 나온다
    assert romaji_line("君＆僕")[0] == "kimi and boku"
    assert romaji_line("君&僕&友")[0] == "kimi and boku and tomo"


def test_sokuon_and_n_cross_token_boundaries():
    # 토큰별로 변환하면 토큰 머리의 っ·ん이 앞 음절을 못 찾아 촉음이 사라지고 ん이 "응"이 된다
    assert wiki_pronunciation("ただ一緒にいたいんだ").endswith("이타인다")
    assert wiki_pronunciation("だってやだ").startswith("닷테")


# ---------------------------------------------------------------------------
# 3. ja_reading 옵션 — 기존 호출부의 기본 동작은 그대로다
# ---------------------------------------------------------------------------


def test_default_reading_is_unchanged_by_the_new_options():
    # text_to_moras / kana_hangul.kanji_to_kana가 쓰는 기본값
    assert kana_reading("帰る動画（トコ）は既に廃墟") == "かえるどうが（とこ）はすでにはいきょ"
    assert kana_reading("今更止められない") == "いまさらとめられない"


def test_phonetic_option_uses_the_pron_field():
    # 음가는 조사 は를 わ로, 王女를 おーじょ로 준다 (표층 읽기는 はおうじょ)
    assert kana_reading("君は王女", phonetic=True) == "きみわおーじょ"
    assert kana_reading("鮮明", phonetic=True) == "せんめー"


def test_ruby_option_drops_the_kanji_and_the_parentheses_from_the_reading():
    assert kana_reading("涙（シル）をこぼす", adopt_ruby=True) == "しるをこぼす"


@pytest.mark.parametrize(
    "text",
    [
        "帰る動画（トコ）は既に廃墟",
        "縋って いつも縋って",
        "Don't Stop！ 2024年の夏",
        "楽しかった時間（トキ）に",
    ],
)
def test_options_keep_the_surface_and_offset_contract(text):
    for phonetic in (False, True):
        for adopt_ruby in (False, True):
            tokens = tokenize_reading(text, phonetic=phonetic, adopt_ruby=adopt_ruby)
            assert "".join(t.surface for t in tokens) == text
            pos = 0
            for token in tokens:
                assert text[token.start : token.end] == token.surface
                assert token.start == pos
                pos = token.end
            assert pos == len(text)


# ---------------------------------------------------------------------------
# 4. 공백이 모라 정렬을 깨지 않는지
# ---------------------------------------------------------------------------


def test_generated_pronunciation_aligns_as_well_as_the_human_one(wiki_pairs):
    """생성한 발음의 DP 정렬 품질이 사람이 쓴 발음과 비슷해야 한다.

    독음 정렬(map_pron_alignment_to_line)은 품질이 임계 미만이면 라인을 통째로 버린다 —
    새로 만든 문자열이 위키 발음만큼 정렬되지 않으면 가라오케 타이밍이 사라진다.
    """
    gen_ok = wiki_ok = 0
    for pair in wiki_pairs:
        moras = text_to_moras(pair["text"])
        assert moras
        _, gen_quality = align_pron_to_moras(moras, wiki_pronunciation(pair["text"]))
        _, wiki_quality = align_pron_to_moras(moras, pair["pron"])
        gen_ok += gen_quality >= _QUALITY_THRESHOLD
        wiki_ok += wiki_quality >= _QUALITY_THRESHOLD
    # 사람이 쓴 발음이 통과하는 줄 수보다 적게 통과하면 회귀다
    assert gen_ok >= wiki_ok


def test_alignment_is_indifferent_to_the_phrase_spacing(wiki_pairs):
    # DP는 공백을 음절로 세지 않는다 — 우리가 새로 넣는 문절 공백도 점수를 바꾸지 않는다
    for pair in wiki_pairs[:40]:
        moras = text_to_moras(pair["text"])
        pron = wiki_pronunciation(pair["text"])
        _, spaced = align_pron_to_moras(moras, pron)
        _, packed = align_pron_to_moras(moras, pron.replace(" ", ""))
        assert spaced == pytest.approx(packed)


def test_spaced_pronunciation_maps_back_to_the_original_line():
    from everyric2.text.reading import map_pron_alignment_to_line

    text = "痛み 痛み 見えない"
    pron = wiki_pronunciation(text)
    assert " " in pron
    syllables = [c for c in pron if not c.isspace()]
    spans = [(c, 1.0 + i * 0.2, 1.2 + i * 0.2) for i, c in enumerate(syllables)]
    words, pron_segments = map_pron_alignment_to_line(text, pron, spans)
    assert words is not None  # 품질 임계를 넘어 원문 글자 매핑이 살아 있다
    assert [p["text"] for p in pron_segments] == syllables


# ---------------------------------------------------------------------------
# 5. translator 연동 — 모델 발음은 항상 버리고 지원 셀만 결정론 값을 쓴다
# ---------------------------------------------------------------------------


def _nvidia(monkeypatch, tmp_path):
    from everyric2.config.settings import TranslationSettings
    from everyric2.translation.translator import NvidiaTranslator

    key_file = tmp_path / "nvapi.txt"
    key_file.write_text("dummy-key", encoding="utf-8")
    monkeypatch.setattr(NvidiaTranslator, "_KEY_FILE", key_file)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    settings = TranslationSettings(engine="nvidia", api_key=None)
    settings.include_pronunciation = True
    return NvidiaTranslator(settings)


def _fake_post(monkeypatch, content: str) -> list[dict]:
    """requests.post를 고정 응답으로 대체하고 보낸 payload들을 돌려준다."""
    calls: list[dict] = []

    class _Response:
        status_code = 200
        ok = True
        text = ""

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}

    def post(url, json, headers, timeout):
        calls.append(json)
        return _Response()

    monkeypatch.setattr("everyric2.translation.translator.requests.post", post)
    return calls


_JA_RESPONSE = (
    '[{"original":"君は王女","translation":"너는 왕녀","pronunciation":"키미하 오우조"}]'
)


def test_japanese_song_ignores_the_llm_pronunciation(monkeypatch, tmp_path):
    translator = _nvidia(monkeypatch, tmp_path)
    calls = _fake_post(monkeypatch, _JA_RESPONSE)

    result = translator.translate("君は王女", source_lang="ja", target_lang="ko")

    # LLM이 준 "키미하 오우조"(조사 は를 표층대로 쓴 실수)가 아니라 결정론 값이 들어간다
    assert result.lines[0].pronunciation == "키미와 오오조"
    assert result.lines[0].translation == "너는 왕녀"
    # 발음을 묻지 않으므로 프롬프트에 발음 지시·참조 읽기 블록이 없다
    assert "pronunciation" not in calls[0]["messages"][0]["content"]


def test_kanji_only_japanese_line_is_still_detected_by_the_song_text(monkeypatch, tmp_path):
    # 라인 단위로 판정하면 가나 없는 줄(限界)이 중국어로 오판된다 — 곡 전체를 본다
    translator = _nvidia(monkeypatch, tmp_path)
    _fake_post(
        monkeypatch,
        '[{"original":"限界","translation":"한계","pronunciation":"x"},'
        '{"original":"まで愛して","translation":"까지 사랑해","pronunciation":"y"}]',
    )

    result = translator.translate("限界\nまで愛して", source_lang="auto", target_lang="ko")

    assert [line.pronunciation for line in result.lines] == ["겐카이", "마데 아이시테"]


def test_chinese_song_discards_the_model_pronunciation(monkeypatch, tmp_path):
    # 중국어는 지원되는 결정론 셀이 아니며 모델 자유서술로 폴백하지 않는다.
    translator = _nvidia(monkeypatch, tmp_path)
    calls = _fake_post(
        monkeypatch,
        '[{"original":"我听到你的声音","translation":"네 목소리가 들려",'
        '"pronunciation":"워 팅 다오 니 더 성인"}]',
    )

    result = translator.translate("我听到你的声音", source_lang="zh", target_lang="ko")

    assert result.lines[0].translation == "네 목소리가 들려"
    assert result.lines[0].pronunciation is None
    assert "pronunciation" not in calls[0]["messages"][0]["content"]


def test_pykakasi_fallback_returns_none_instead_of_model_pronunciation(monkeypatch, tmp_path):
    # 폴백 독음은 신뢰도가 낮다(縋って→ついって) — 모델에 넘기지 않고 발음을 비운다.
    monkeypatch.setattr(
        "everyric2.translation.translator.reading_source", lambda: "pykakasi"
    )
    translator = _nvidia(monkeypatch, tmp_path)
    calls = _fake_post(monkeypatch, _JA_RESPONSE)

    result = translator.translate("君は王女", source_lang="ja", target_lang="ko")

    assert result.lines[0].translation == "너는 왕녀"
    assert result.lines[0].pronunciation is None
    assert "pronunciation" not in calls[0]["messages"][0]["content"]


def test_ja_to_en_uses_deterministic_romaji_not_the_llm_value(monkeypatch, tmp_path):
    # 발음 매트릭스(ja×en 셀)가 결정론화된 뒤: 비ko 타깃도 로마자 결정론 경로가 있다
    # (`pron_style.romaji_line`) — 모델이 돌려준 자유서술 로마자는 버려지고 프롬프트도
    # 발음을 묻지 않는다.
    translator = _nvidia(monkeypatch, tmp_path)
    calls = _fake_post(
        monkeypatch,
        '[{"original":"君は王女","translation":"you are the princess",'
        '"pronunciation":"kimi wa oujo"}]',
    )

    result = translator.translate("君は王女", source_lang="ja", target_lang="en")

    # romaji_line("君は王女")[0]의 wapuro 표기(장음 おう→oo) — LLM의 "oujo"가 아니다
    assert result.lines[0].pronunciation == "kimi wa oojo"
    assert "pronunciation" not in calls[0]["messages"][0]["content"]


def test_deterministic_pronunciation_survives_finalize(monkeypatch, tmp_path):
    # LyricsTranslator.translate_with_pronunciation의 마감(finalize_pronunciation)은
    # 한글뿐인 문자열을 건드리지 않아야 한다 (공백만 정돈)
    from everyric2.text.kana_hangul import finalize_pronunciation

    pron = wiki_pronunciation("叫んだ音は既に列を成さないで")
    assert finalize_pronunciation(pron) == pron


# ---------------------------------------------------------------------------
# 6. romaji 라인 렌더 + 심판 후보 토큰 노출 (다국어화, 2026-07-28, Task 4)
# ---------------------------------------------------------------------------


def test_romaji_line_basic():
    from everyric2.text.pron_style import romaji_line

    display, moras, spaces = romaji_line("アルバイトはネクラモード")
    assert display == "arubaito wa nekura moodo"
    assert len(moras) == 12  # text_to_moras와 같은 모라 수
    assert "".join(m + (" " if s else "") for m, s in zip(moras, spaces)).strip() == display


def test_romaji_line_latin_passthrough():
    from everyric2.text.pron_style import romaji_line

    display, _, _ = romaji_line("Take it easy なんて言葉じゃ")
    assert display.startswith("Take it easy")  # 라틴 원형 유지


def test_candidate_token_sets_parallel():
    from everyric2.text.pron_style import candidate_token_sets, pronunciation_candidates

    text = "ずっと見てたよ"
    rendered, tokens = candidate_token_sets(text)
    assert rendered == pronunciation_candidates(text)
    assert len(rendered) == len(tokens)


def test_romaji_line_none_for_non_japanese_non_latin_text():
    from everyric2.text.pron_style import romaji_line

    assert romaji_line("") is None
    assert romaji_line("오늘 밤") is None
    assert romaji_line("1234 (56)") is None


def test_romaji_line_display_matches_moras_invariant_across_lines():
    # 표시=세그 단일 소스 불변식은 특정 라인 하나만의 우연이 아니어야 한다
    from everyric2.text.pron_style import romaji_line

    for text in (
        "フラッシュバック・蝉の声・二度とは帰らぬ君",
        "背負った",
        "ずっと見 てたよ",
        "縋って 縋って",
        "左から右へと",
    ):
        result = romaji_line(text)
        assert result is not None
        display, moras, spaces = result
        assert len(moras) == len(spaces)
        assert "".join(m + (" " if s else "") for m, s in zip(moras, spaces)).strip() == display


def test_candidate_token_sets_tokens_feed_text_to_moras():
    # 심판이 고른 대안 읽기(대체 토큰 열)를 text_to_moras에 넘기면 모라 수가 따라온다
    from everyric2.text.pron_style import candidate_token_sets
    from everyric2.text.reading import text_to_moras

    text = "ずっと観てたよ"
    rendered, tokens = candidate_token_sets(text)
    assert len(rendered) >= 2  # 観て(みて/みえて) 애매어휘 후보가 있다
    for cand_tokens in tokens:
        moras = text_to_moras(text, tokens=cand_tokens)
        assert moras  # 재토크나이즈 없이도 모라를 만든다


def test_text_to_moras_default_unchanged_by_new_tokens_param():
    # 기존 호출자(인자 없이 호출)는 동작이 그대로다
    text = "叫んだ音は既に列を成さないで"
    assert text_to_moras(text) == text_to_moras(text, tokens=None)
