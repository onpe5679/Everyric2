"""``everyric2.text.pron_backcheck`` — 한글 발음 표기의 원문 역검사.

이 검사의 유일한 치명적 실패는 **오탐**이다: 정상 표기를 「불가능」이라 부르면 리포트가
정상 줄로 뒤덮여 도구를 못 쓰게 된다. 그래서 테스트의 무게가 적발보다 오탐 쪽에 실려
있다 — 사람이 만든 관습 표기와 기존 변환기 출력이 **하나도** 걸리지 않아야 한다.
"""

import pytest

from everyric2.text.en_g2p import transliterate_cmu
from everyric2.text.latin_hangul import transliterate_latin
from everyric2.text.pron_backcheck import check_line, word_inventory

# ── 오탐 방지 ──────────────────────────────────────────────────────────────────

# 사람이 만든 관습 외래어 표기. 규칙 엔진 출력과 다른 것이 정상이고(coffee는 AO를 ㅗ가
# 아니라 ㅓ로, shirt는 T를 ㅌ이 아니라 ㅊ로 적는다) 그 차이는 결함이 아니다.
HUMAN_CONVENTIONAL = [
    ("coffee", "커피"), ("shirt", "셔츠"), ("juice", "주스"),
    ("television", "텔레비전"), ("computer", "컴퓨터"), ("chocolate", "초콜릿"),
    ("sandwich", "샌드위치"), ("orange", "오렌지"), ("message", "메시지"),
    ("vision", "비전"), ("nation", "네이션"), ("sure", "슈어"),
    ("picture", "픽처"), ("nature", "네이처"), ("future", "퓨처"),
    ("adventure", "어드벤처"), ("question", "퀘스천"), ("education", "에듀케이션"),
    ("measure", "메저"), ("casual", "캐주얼"), ("schedule", "스케줄"),
    ("gentleman", "젠틀맨"), ("journey", "저니"), ("danger", "데인저"),
    ("beautiful", "뷰티풀"), ("strawberry", "스트로베리"), ("yellow", "옐로"),
    ("window", "윈도"), ("water", "워터"), ("world", "월드"), ("work", "워크"),
    ("want", "원트"), ("would", "우드"), ("woman", "우먼"), ("rhythm", "리듬"),
    ("button", "버튼"), ("little", "리틀"), ("young", "영"), ("thank", "생크"),
    ("birthday", "버스데이"), ("brother", "브라더"), ("mother", "마더"),
    ("special", "스페셜"), ("social", "소셜"), ("official", "오피셜"),
    ("christmas", "크리스마스"), ("chance", "찬스"), ("change", "체인지"),
    ("church", "처치"), ("shadow", "섀도"), ("sugar", "슈가"),
    ("pleasure", "플레저"), ("treasure", "트레저"), ("heart", "하트"),
    ("smile", "스마일"), ("fire", "파이어"),
    # /aʊər/를 「아워」로 적는 관습 — 원문에 W 음소가 없는데 활음이 생기는 자리다.
    # 활음을 판정하면 여기서 전부 오탐이 난다(그래서 핵모음만 본다).
    ("hour", "아워"), ("flower", "플라워"), ("power", "파워"), ("tower", "타워"),
    ("love", "러브"), ("night", "나이트"), ("dream", "드림"), ("summer", "서머"),
    ("always", "올웨이즈"), ("never", "네버"), ("together", "투게더"),
    ("forever", "포에버"), ("yesterday", "예스터데이"), ("everything", "에브리싱"),
    ("nothing", "낫싱"), ("something", "썸싱"), ("girl", "걸"), ("boy", "보이"),
    ("friend", "프렌드"), ("family", "패밀리"), ("happy", "해피"), ("sorry", "쏘리"),
    # 실사용에서 경음으로 굳은 표기 — 표기법은 금하지만 가사·자막에는 흔하다
    ("thank you", "땡큐"), ("goodbye", "굿바이"),
    # 여러 낱말 — 낱말 단위 짝맞춤 경로를 태운다
    ("ice cream", "아이스크림"), ("i love you", "아이 러브 유"),
    ("let it go", "렛 잇 고"), ("all right", "올 라이트"),
]

CONVERTER_LINES = [
    "I never wanted to be the one",
    "Weather girl she brings the rain",
    "Every time I close my eyes",
    "You said you would always stay",
    "Beautiful strangers in the crowd",
    "I would rather laugh than cry about it",
    "Take my hand and never let me go",
    "Nothing really matters anymore",
    "We were dancing in the summer light",
    "Tell me something I don't know",
]


@pytest.mark.parametrize(("source", "pron"), HUMAN_CONVENTIONAL)
def test_human_conventional_spelling_is_not_flagged(source, pron):
    """사람이 만든 관습 표기는 규칙 엔진 출력과 달라도 불가능이 아니다."""
    verdict = check_line(source, pron)
    assert verdict.scope != "skipped", f"판정에서 빠졌다: {verdict.skip_reason}"
    assert verdict.ok, [(f.syllable, f.part, f.reason) for f in verdict.impossible]


@pytest.mark.parametrize("line", CONVERTER_LINES)
def test_converter_output_is_never_impossible(line):
    """기존 두 음차 경로가 실제로 내는 표기는 정의상 통과해야 한다 — 오탐 0의 하한.

    이것이 깨지면 역검사가 **우리 자신의 출력**을 불가능이라 부른다는 뜻이라, 판정 기준이
    아니라 역검사 쪽이 틀린 것이다.
    """
    for pron in (transliterate_latin(line), transliterate_cmu(line)):
        verdict = check_line(line, pron)
        assert verdict.scope != "skipped", f"판정에서 빠졌다: {verdict.skip_reason}"
        assert verdict.ok, (pron, [(f.syllable, f.reason) for f in verdict.impossible])


def test_proper_noun_is_not_judged_word_by_word():
    """CMU에 없는 낱말(고유명사)은 낱말 단위 판정에서 빠진다 — 규칙이 가장 약한 자리다."""
    verdict = check_line("Kaguya smiled", "카구야 스마일드")
    assert verdict.scope == "word"
    assert "Kaguya" in verdict.skipped_words
    assert verdict.ok


# ── 적발 ───────────────────────────────────────────────────────────────────────

# 원문에 그 소리가 아예 없는 음절 — 「물리적으로 있을 수 없는 표기」의 정의 그대로다.
HALLUCINATED = [
    ("Weather girl", "웨더 걸쟈", "쟈"),
    ("I love you", "아이 러브 쮸", "쮸"),
    ("take my hand", "테익 마이 핸쿄", "쿄"),
    ("summer rain", "서머 레인비", "비"),
    ("close my eyes", "클로즈 마이 아이푸", "푸"),
    ("hold me now", "홀드 미 나흐", "흐"),
    ("blue sky", "블루 스카치", "치"),
    # 가나 경유 독음이 한국어 발음 칸에 새어 든 꼴 (ガール→가루)
    ("Weather girl", "웨더 규루", "루"),
    ("I miss you", "아이 미스 유키", "키"),
]


@pytest.mark.parametrize(("source", "pron", "culprit"), HALLUCINATED)
def test_impossible_syllable_is_caught(source, pron, culprit):
    verdict = check_line(source, pron)
    assert not verdict.ok, f"환각을 놓쳤다: {pron}"
    assert culprit in {f.syllable for f in verdict.impossible}


def test_verdict_points_at_the_offending_word_and_part():
    """리포트가 쓸모 있으려면 어느 낱말의 어느 성분인지가 나와야 한다."""
    verdict = check_line("Weather girl", "웨더 걸쟈")
    assert verdict.scope == "word"
    (finding,) = verdict.impossible
    assert finding.word == "girl"
    assert finding.part == "초성"
    assert finding.value == "ㅈ"
    assert verdict.pron[finding.position] == "쟈"


# ── 판정 범위와 제외 ───────────────────────────────────────────────────────────


def test_token_count_mismatch_falls_back_to_line_scope():
    """연음으로 낱말이 합쳐지면(want it → 워닛) 낱말 짝맞춤이 깨진다 — 줄 단위로 내려간다."""
    verdict = check_line("want it", "워닛")
    assert verdict.scope == "line"
    assert verdict.ok
    assert all(f.word is None for f in verdict.impossible)


def test_line_scope_still_catches_syllables_alien_to_the_whole_line():
    verdict = check_line("want it", "워닛쟈")
    assert verdict.scope == "line"
    assert "쟈" in {f.syllable for f in verdict.impossible}


@pytest.mark.parametrize(
    ("text", "pron"),
    [
        ("せめて此処で祈らせてよ flower", "세메테 코코데 이노라세테요 플라워"),  # 일본어 혼합
        ("I&P&V4/6", "아이앤피앤브이 포오브식스"),  # 숫자·기호
        ("君にだけ", "쿤니다케"),  # 라틴 낱말 없음
    ],
)
def test_unjudgeable_lines_are_skipped_not_flagged(text, pron):
    """귀속시킬 원문이 없는 음절은 판정하지 않는다 — 침묵이 오탐보다 낫다."""
    verdict = check_line(text, pron)
    assert verdict.scope == "skipped"
    assert verdict.skip_reason
    assert not verdict.impossible


def test_skipped_line_is_not_reported_as_ok():
    """건너뛴 줄을 ``ok``로 세면 「판정했고 문제없다」와 구별이 사라진다."""
    verdict = check_line("君にだけ", "쿤니다케")
    assert not verdict.ok


def test_line_without_hangul_pron_is_skipped():
    assert check_line("hello world", "hello world").scope == "skipped"


# ── 범위 생성 ──────────────────────────────────────────────────────────────────


def test_inventory_unions_every_dictionary_pronunciation():
    """CMU가 발음을 둘 이상 들면 전부 합집합이다 — 어느 쪽으로 불렀는지는 오디오만 안다."""
    inv = word_inventory("the")
    assert inv.known
    # ðə(더)의 ㅓ와 ði(디)의 ㅣ가 **둘 다** 열려 있어야 한다
    assert {"eo", "i"} <= inv.vowel_cores


def test_inventory_covers_the_letter_name_reading():
    """글자 이름 읽기(ATM → 에이티엠)도 기존 변환기가 내는 값이라 허용 범위에 든다."""
    verdict = check_line("ATM", "에이티엠")
    assert verdict.ok


def test_oov_word_inventory_is_marked_unknown():
    inv = word_inventory("Kaguya")
    assert not inv.known
    assert inv.onsets  # 철자 경로가 재료를 주므로 비어 있지는 않다
