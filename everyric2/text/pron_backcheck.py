"""한글 발음 표기 → 원문 **역검사**: 이 표기가 원문 IPA에서 나올 수 있는가.

## 왜 필요한가

en 가사의 발음 표기는 **규칙 엔진이 만들지 않는다.** ``translation.translator.
_deterministic_pron_fn``의 결정론 매트릭스는 ja×ko · ja×en · ko×ja · ko×en 네 칸뿐이고
**en 원문 칸이 없다** — en 곡은 그 함수가 None을 돌려주므로 발음이 LLM 자유서술 경로로
간다. 즉 표시되는 한글은 ``en_g2p``·``latin_hangul``의 산출물이 아니라 LLM이 쓴 문자열이고,
그래서 원문에 근거가 없는 음절("물리적으로 있을 수 없는 표기")이 섞일 수 있다. 실사용
제보(웨더걸/M7VSEZOQIlg)가 그 사건이다.

이 모듈은 그 표기를 **원문 쪽에서 되짚어** 검사한다. 낱말의 CMU 발음(``en_g2p.
pronunciations``)에서 나올 수 있는 자모의 범위를 만들고, 실제 표기의 각 음절이 그 범위를
벗어나는지만 본다.

## 정밀도 우선 — 「다르다」와 「불가능하다」를 가른다

이 검사의 실패 양상은 하나뿐이어야 한다: **정상 표기를 불가능으로 부르면 도구가 못 쓰게
된다.** 사람·위키·LLM의 표기는 규칙 엔진 출력과 얼마든지 다를 수 있고(관습형 대 조밀형,
고유명사 읽기), 그 차이는 결함이 아니다. 그래서 판정은 「우리 출력과 같은가」가 아니라
「원문 음소로 설명되는가」로만 한다. 구체적으로 다음을 모두 **허용 범위에 넣는다**:

1. 그 낱말의 **모든** CMU 발음(``pronunciations``)에서 파생되는 초성·중성·종성.
   사전이 발음을 둘 이상 들고 있으면 전부 합집합이다 — 어느 발음으로 불렀는지는 오디오만
   안다(``en_g2p.pronunciations`` 참고).
2. 기존 변환기가 실제로 내는 글자(``phones_to_hangul``·``latin_word_to_hangul``의
   tight/loose 두 값)의 자모. 이것이 **오탐 0의 하한**이다 — 우리 자신이 내는 표기를
   불가능이라 부르는 일은 정의상 생기지 않는다. 글자 이름 읽기(ATM→에이티엠)나 못박은
   표(VOCALOID→보오카로이도)도 이 경로로 함께 들어온다.
3. 자음군을 적으려고 끼워 넣는 모음 ㅡ·ㅣ(스·시·즈·지·츠·치)와 무초성 ㅇ, 무종성.
   한국어가 자음을 홀로 적을 때 쓰는 유일한 수단이라 위치 무관으로 늘 허용한다.

또한 **낱말 안에서는 위치를 보지 않는다.** 어느 음절의 초성인지가 아니라 「그 낱말의
어느 음소에서든 나올 수 있는가」만 묻는다. 음절 경계는 조밀화·연음·가창에 따라 얼마든지
움직이므로(``en_g2p.phones_to_units``의 최대 초성 원칙 참고) 위치까지 따지면 정상 표기를
잡는다.

## 판정하지 않는 것 — 침묵이 오탐보다 낫다

- **CMU에 없는 낱말(OOV)은 낱말 단위 판정에서 뺀다.** 고유명사·조어는 규칙 엔진이 가장
  약하고 LLM이 오히려 맞는 자리다(Chloe → 클로이 대 규칙 클로). 여기서 잡으면 리포트가
  고유명사로 뒤덮인다. 라인 단위 판정에서는 그 낱말의 범위가 합집합에 그대로 들어가
  **범위를 넓히는 쪽으로만** 작용한다(오탐을 만들지 않는다).
- **숫자·``&``·비라틴 문자가 섞인 줄은 통째로 건너뛴다.** 「I&P&V4/6 → 아이앤피앤브이
  포오브식스」처럼 낱말이 아닌 것에서 나온 음절은 귀속시킬 원문이 없어 전부 불가능으로
  보인다. 일본어가 섞인 줄도 같다 — 가나에서 온 한글을 라틴 낱말로 설명할 수 없다.
- **음절 수는 세지 않는다.** 조밀 표기가 관습 표기보다 짧은 것이 정상이고(``latin_hangul``
  문서의 실측), 가창은 음절을 더 붙이기도 뺀기도 한다. 수로 판정하면 정상 표기를 잡는다.

## 검사가 못 잡는 것

허용 범위를 **기존 변환 표에서** 만들므로 **표 자체의 결함은 이 검사를 통과한다.** 실측
예: ``want``(W AO1 N T)가 「온」이 되어 /w/가 사라지는데(``en_g2p._W_GLIDE``가 ㅗ→ㅗ,
ㅜ→ㅜ로 활음을 흡수하지 못한다) ㅗ는 AO의 정당한 상이라 여기서는 통과한다. 이 검사는
「원문에 없는 소리가 표기에 들어왔는가」(환각)를 보지 「원문에 있는 소리가 표기에서
빠졌는가」(누락)를 보지 않는다. 누락은 별개의 검사이고, 위 사례는 검사가 아니라 표를
고쳐야 하는 결함이다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from everyric2.text import en_g2p, latin_hangul
from everyric2.text.latin_hangul import _decompose

# 초성이 없는 음절의 자리채움. 모음으로 시작하는 음절은 언제나 이 글자를 쓴다.
_NULL_ONSET = "ㅇ"

# ── 자모 등가류 ────────────────────────────────────────────────────────────────
#
# ``en_g2p``의 ``_ONSETS``·``_CODAS``·``_VOWELS``는 음소마다 **상 하나씩만** 든다 —
# 변환기는 값을 하나 골라야 하기 때문이다. 역검사는 그 반대다: 사람·LLM이 쓸 수 있는
# 표기를 **전부** 통과시켜야 하므로 그 표를 출발점으로 삼되 관습이 실제로 쓰는 다른
# 상까지 열어야 한다. 아래 표들이 그 확장분이고, 근거는 전부 관습 외래어 표기다.
# 확장 전에는 사람이 만든 관습 표기에서 오탐이 13%대로 났고(coffee 커피·mother 마더·
# shirt 셔츠·아워 계열) 확장 후 0이 됐다 — ``tests/test_pron_backcheck.py``가 그 집합을
# 못박는다.
#
# 초성 확장 — 음소별로 ``_ONSETS``에 **더해** 허용할 초성. 여기 없는 음소는 확장분이
# 없다는 뜻이다(``_ONSETS``의 상 하나만 허용).
_ONSET_EQUIV: dict[str, str] = {
    # 무성 파열음은 격음(파이)과 평음(바이)·경음(빠이)으로 두루 적힌다. 한국어 표기법은
    # 경음을 금하지만 실사용 표기와 가창 표기에는 흔하다(써니·빠이·쨈).
    "P": "ㅂㅃ", "T": "ㄷㄸ", "K": "ㄱㄲ",
    "B": "ㅃㅍ", "D": "ㄸㅌ", "G": "ㄲㅋ",
    # /f/의 한국어 상은 ㅍ(``_ONSETS``)이고 ㅂ은 그 평음이다. ㅎ은 **가나 경유** 표기에서
    # 온다(ライフ→라이후) — ja 곡에 섞인 라틴 낱말이 가타카나 관습으로 불리는 자리라
    # 실제 코퍼스에 있다. 그 표기를 불가능으로 부르지 않으려고 열어 둔다. 순수 en 곡만
    # 조사한다면 여기서 ㅎ을 빼는 쪽이 검출력이 높다.
    "F": "ㅎㅂ",
    "V": "ㅍㅃ",
    "S": "ㅆ", "Z": "ㅅㅆㅉ", "SH": "ㅆ", "ZH": "ㅅㅆ",
    # /θ/는 ㅅ이 표준이지만 실사용 표기는 경음으로도 굳었다(땡큐·땡스) — 측정에서
    # 「thank you → 땡큐」가 유일한 오탐이었고 ㄸ을 열어 0으로 떨어졌다.
    "TH": "ㅆㅌㄷㄸ", "DH": "ㅈㅅㄸ",
    "CH": "ㅈㅉ", "JH": "ㅊㅉ",
    # /ŋ/은 초성 자리에서 ㅇ(무음)이지만 어중에서 앞 음절 ㄴ으로도 적힌다(bank 뱅크/반크).
    "NG": "ㄴ",
}
# 위 표의 파찰음화는 음소 키가 아니라 별도 항목이라 표에 섞지 않고 여기서 더한다.
_AFFRICATED: dict[str, str] = {"T": "ㅊ", "D": "ㅈ", "TH": "ㅊ", "DH": "ㅈ"}

# 종성 확장 — ``_CODAS``에 더해 허용할 종성. 대표음이 같은 글자끼리 묶는다
# (표준 발음법의 음절말 중화: ㅅㅆㅈㅊㅌㄷ은 모두 [t̚]).
_CODA_EQUIV: dict[str, str] = {
    "P": "ㅍ", "B": "ㅍ", "F": "ㅍ", "V": "ㅍ",
    "T": "ㄷㅆㅈㅊㅌ", "D": "ㄷㅆㅈㅊㅌ", "TH": "ㄷㅆㅈㅊㅌ", "DH": "ㄷㅆㅈㅊㅌ",
    "S": "ㄷㅆㅈㅊㅌ", "Z": "ㄷㅆㅈㅊㅌ", "SH": "ㄷㅆㅈㅊㅌ", "ZH": "ㄷㅆㅈㅊㅌ",
    "CH": "ㄷㅆㅈㅊㅌ", "JH": "ㄷㅆㅈㅊㅌ",
    "K": "ㅋㄲ", "G": "ㅋㄲ",
    # 비음·유음(M·N·NG·L·R)은 ``_CODAS``의 ㅁ·ㄴ·ㅇ·ㄹ 하나뿐이라 확장분이 없다.
}

# ── 모음 대역 ──────────────────────────────────────────────────────────────────
#
# 모음은 자음보다 훨씬 넓게 갈린다. 같은 /ɔ/를 coffee는 「커」(ㅓ)로 caught은 「콧」(ㅗ)으로
# 적고, /ʌ/는 mother 「마」(ㅏ) · but 「벗」(ㅓ) 둘 다 관습이다. 그래서 중성을 글자 그대로
# 비교하지 않고 **활음을 뗀 핵모음(대역)**으로만 본다 — 정밀도 우선의 핵심 완화다.
# 활음(ㅑ·ㅘ·ㅝ…)은 아예 판정하지 않는다: /aʊər/를 「아워」로 적는 관습(hour·power·flower)
# 처럼 원문에 W 음소가 없어도 활음이 정당하게 생기는 자리가 많다.
_JUNG_CORE: dict[str, str] = {
    "ㅏ": "a", "ㅑ": "a", "ㅘ": "a",
    "ㅐ": "e", "ㅒ": "e", "ㅔ": "e", "ㅖ": "e", "ㅙ": "e", "ㅚ": "e", "ㅞ": "e",
    "ㅓ": "eo", "ㅕ": "eo", "ㅝ": "eo",
    "ㅗ": "o", "ㅛ": "o",
    "ㅜ": "u", "ㅠ": "u", "ㅟ": "u",
    "ㅡ": "eu", "ㅢ": "eu",
    "ㅣ": "i",
}
# ARPABET 모음 → 허용 핵모음 대역. 「그 소리를 한국어로 적을 때 실제로 쓰이는 글자」가
# 기준이고, 음성학적 최근접 하나로 좁히지 않는다.
_VOWEL_BANDS: dict[str, str] = {
    "AA": "a eo o", "AE": "a e", "AH": "a e eo o u", "AO": "a o eo",
    # ER(/ɜr·ər/)에 ㅜ 계열은 넣지 않는다 — 한국어는 이 소리를 어(걸·턴·머더)나 아
    # (슈가·달러)로만 적는다. ㅜ로 적는 것은 가나 경유 표기(ガール→가루)이지 영어 음차가
    # 아니다. 측정에서 이 한 칸을 빼야 「girl → 규루」류가 잡히고, 오탐은 늘지 않았다.
    "EH": "e eo a", "ER": "eo a", "IH": "i e eo", "IY": "i e",
    "UH": "u o eo", "UW": "u o", "OW": "o u eo",
    "AW": "a o u eo", "AY": "a i e", "EY": "e i a", "OY": "o i eo",
}
# 삽입 모음의 대역 — 자음군을 적으려고 끼워 넣는 ㅡ(스트·브)와 ㅣ(시·지·치)는 어느
# 낱말에서나 열려 있다. ``latin_hangul._CONS``의 「홀로 설 때의 중성」이 정확히 이 둘이고,
# 어느 자음에 어느 쪽이 붙는지까지는 따지지 않는다 — 관습이 갈리는 자리라(wish 위시/위쉬)
# 좁히면 정상 표기를 잡는다.
_EPENTHETIC_CORES = frozenset({"eu", "i"})

# 원문 낱말 — 라틴 글자 덩어리와 그 사이의 아포스트로피(don't·I'm·goin').
# ``en_g2p._WORD_RE``(``[A-Za-z']+``)와 달리 글자를 최소 하나 요구한다: 아포스트로피만
# 남은 조각이 낱말로 잡히면 그 자리에 발음 토큰이 하나 있는 것처럼 세어져 낱말↔토큰
# 짝맞춤이 어긋난다.
_WORD_RE = re.compile(r"[A-Za-z]+(?:['’ʼ][A-Za-z]+)*")
# 판정 대상 밖 문자 — 라틴 글자·아포스트로피·공백·문장부호가 아닌 것. 숫자와 ``&``가
# 여기 걸리는 것이 요점이다(모듈 문서의 「판정하지 않는 것」 참고).
_OUT_OF_SCOPE_RE = re.compile(r"[^A-Za-z'’ʼ\s.,!?;:()\[\]/\"“”‘…\-–—]")
_HANGUL_SYLLABLE_RE = re.compile(r"[가-힣]")


@dataclass(frozen=True)
class WordInventory:
    """한 낱말이 낼 수 있는 자모의 범위. 위치는 보지 않는다(모듈 문서 참고)."""

    word: str
    onsets: frozenset[str]
    #: 허용 **핵모음 대역**(``_JUNG_CORE``의 값)이다 — 중성 글자 자체가 아니다.
    vowel_cores: frozenset[str]
    codas: frozenset[str]
    #: CMU 사전에 있었는가. False면 낱말 단위 판정에서 제외한다(고유명사 오탐 방지).
    known: bool


@dataclass
class SyllableVerdict:
    """불가능으로 판정된 음절 하나."""

    syllable: str
    #: 발음 문자열에서의 글자 위치 (0부터)
    position: int
    #: 귀속된 원문 낱말. 라인 단위 판정이면 None(줄 전체를 상대로 봤다는 뜻)
    word: str | None
    #: "초성" | "중성" | "종성" — 어느 성분이 범위를 벗어났는가
    part: str
    #: 그 성분의 실제 값
    value: str
    reason: str


@dataclass
class LineVerdict:
    """한 줄의 판정 결과."""

    text: str
    pron: str
    #: "word"(낱말 단위 짝맞춤 성공) | "line"(줄 단위 합집합) | "skipped"(판정 안 함)
    scope: str
    #: scope="skipped"일 때 그 이유
    skip_reason: str | None = None
    #: 실제로 판정한 한글 음절 수
    checked: int = 0
    #: OOV라서 낱말 단위 판정에서 뺀 낱말들
    skipped_words: tuple[str, ...] = ()
    impossible: tuple[SyllableVerdict, ...] = ()

    @property
    def ok(self) -> bool:
        """판정했고 불가능 음절이 없었는가. 건너뛴 줄은 ok가 아니다(판정 자체가 없다)."""
        return self.scope != "skipped" and not self.impossible


def _add_syllable_jamo(text: str, onsets: set[str], cores: set[str], codas: set[str]) -> None:
    """완성형 한글 문자열의 자모를 범위에 더한다 — 기존 변환기 출력을 그대로 허용하는 통로.

    이 통로가 **오탐 0의 하한**이다: 변환기가 실제로 내는 글자는 정의상 허용된다.
    """
    for char in text:
        parts = _decompose(char)
        if parts is None:
            continue
        cho, jung, jong = parts
        onsets.add(cho)
        core = _JUNG_CORE.get(jung)
        if core:
            cores.add(core)
        if jong:
            codas.add(jong)


def _add_phone_images(
    phones: list[str], onsets: set[str], cores: set[str], codas: set[str]
) -> None:
    """ARPABET 음소열이 낼 수 있는 자모를 범위에 더한다.

    변환기 출력(``_add_syllable_jamo``)만으로는 **그 변환기가 고른 한 갈래**밖에 못 담는다.
    사람·LLM은 같은 음소를 다른 관습으로 적으므로(coffee 커피 대 ㅗ, shirt 셔츠 대 ㅌ)
    음소마다 등가류(``_ONSET_EQUIV``·``_CODA_EQUIV``·``_VOWEL_BANDS``)를 펼쳐 범위를 넓힌다.
    """
    for raw in phones:
        base = en_g2p._strip_stress(raw)

        onset = en_g2p._ONSETS.get(base)
        if onset:
            onsets.add(onset)
        onsets.update(_ONSET_EQUIV.get(base, ""))
        onsets.update(_AFFRICATED.get(base, ""))

        coda = en_g2p._CODAS.get(base)
        if coda:
            codas.add(coda)
        codas.update(_CODA_EQUIV.get(base, ""))
        # 파찰음화된 삽입 음절(츠·즈)은 종성이 아니라 초성이므로 코다에는 더하지 않는다.

        cores.update(_VOWEL_BANDS.get(base, "").split())


def word_inventory(word: str) -> WordInventory:
    """낱말 하나가 낼 수 있는 초성·핵모음 대역·종성의 범위.

    재료는 세 갈래이고 전부 **기존 모듈에서** 온다(모듈 문서의 1·2·3):
    CMU 모든 발음의 음소 등가류 · 기존 변환기 두 경로의 실제 출력 · 삽입 모음과 무초성.
    """
    onsets: set[str] = {_NULL_ONSET}
    cores: set[str] = set(_EPENTHETIC_CORES)
    codas: set[str] = set()

    entries = en_g2p.pronunciations(word)
    for phones in entries:
        _add_phone_images(phones, onsets, cores, codas)
        _add_syllable_jamo(en_g2p.phones_to_hangul(phones), onsets, cores, codas)

    # 철자 경로 — OOV의 유일한 재료이자, 사전에 있는 낱말에서도 관습형 갈래를 더한다.
    # tight/loose 두 값을 다 넣는다(테익·테이크가 둘 다 정당한 표기다).
    for tight in (True, False):
        _add_syllable_jamo(
            latin_hangul.latin_word_to_hangul(word, tight=tight), onsets, cores, codas
        )

    return WordInventory(
        word=word,
        onsets=frozenset(onsets),
        vowel_cores=frozenset(cores),
        codas=frozenset(codas),
        known=bool(entries),
    )


def _merge(inventories: list[WordInventory]) -> WordInventory:
    """줄 전체의 합집합 범위 — 낱말↔토큰 짝맞춤이 안 될 때 쓴다."""
    return WordInventory(
        word="",
        onsets=frozenset().union(*(inv.onsets for inv in inventories)),
        vowel_cores=frozenset().union(*(inv.vowel_cores for inv in inventories)),
        codas=frozenset().union(*(inv.codas for inv in inventories)),
        known=any(inv.known for inv in inventories),
    )


def _check_syllable(
    char: str, position: int, inv: WordInventory, *, attribute: bool
) -> SyllableVerdict | None:
    """음절 하나를 범위와 대조. 통과하면 None.

    초성 → 중성 → 종성 순으로 보고 **처음 벗어난 성분 하나만** 보고한다. 한 음절에서
    둘 이상이 어긋나도 원인은 대개 하나라(그 자리에 없는 낱말을 읽었다) 나열하면
    리포트만 길어진다.
    """
    parts = _decompose(char)
    if parts is None:
        return None
    cho, jung, jong = parts
    owner = inv.word if attribute else None
    if cho not in inv.onsets:
        return SyllableVerdict(
            syllable=char,
            position=position,
            word=owner,
            part="초성",
            value=cho,
            reason=f"원문 음소로 {cho} 초성이 나오지 않는다",
        )
    core = _JUNG_CORE.get(jung)
    if core is not None and core not in inv.vowel_cores:
        return SyllableVerdict(
            syllable=char,
            position=position,
            word=owner,
            part="중성",
            value=jung,
            reason=f"원문 음소로 {jung} 계열({core}) 모음이 나오지 않는다",
        )
    if jong and jong not in inv.codas:
        return SyllableVerdict(
            syllable=char,
            position=position,
            word=owner,
            part="종성",
            value=jong,
            reason=f"원문 음소로 {jong} 종성이 나오지 않는다",
        )
    return None


def check_line(text: str, pron: str) -> LineVerdict:
    """영어 원문 한 줄과 그 한글 발음 표기를 대조해 불가능한 음절을 찾는다.

    ``scope``가 판정의 강도를 말한다:

    - ``"word"`` — 발음의 공백 토큰 수가 원문 낱말 수와 같아 **낱말마다** 그 낱말의 범위로
      본 경우. 가장 강하다(다른 낱말의 음소로는 변명이 안 된다).
    - ``"line"`` — 토큰 수가 어긋나 줄 전체의 합집합으로만 본 경우. 연음이나 낱말 병합
      (want it → 워닛)에서 흔하다. 줄 어디에도 근거가 없는 음절만 잡힌다.
    - ``"skipped"`` — 판정하지 않았다. ``skip_reason``에 이유가 있다.
    """
    words = _WORD_RE.findall(text)
    if not words:
        return LineVerdict(text, pron, "skipped", skip_reason="원문에 라틴 낱말이 없다")
    out_of_scope = _OUT_OF_SCOPE_RE.search(text)
    if out_of_scope:
        return LineVerdict(
            text,
            pron,
            "skipped",
            skip_reason=f"원문에 판정 대상 밖 문자가 있다: {out_of_scope.group(0)!r}",
        )
    if not _HANGUL_SYLLABLE_RE.search(pron or ""):
        return LineVerdict(text, pron, "skipped", skip_reason="발음 표기에 한글이 없다")

    inventories = [word_inventory(word) for word in words]
    tokens = [t for t in (pron or "").split() if _HANGUL_SYLLABLE_RE.search(t)]

    findings: list[SyllableVerdict] = []
    checked = 0
    skipped_words: list[str] = []

    if len(tokens) == len(words):
        scope = "word"
        # 발음 토큰의 글자 위치를 원문 그대로 되짚어야 position이 쓸모 있다 — 토큰을
        # 다시 찾는 대신 원본에서의 시작 offset을 순차로 계산한다.
        cursor = 0
        for token, inv in zip(tokens, inventories):
            start = pron.index(token, cursor)
            cursor = start + len(token)
            if not inv.known:
                # OOV는 낱말 단위로 판정하지 않는다(모듈 문서의 「판정하지 않는 것」)
                skipped_words.append(inv.word)
                continue
            for offset, char in enumerate(token):
                if not _HANGUL_SYLLABLE_RE.match(char):
                    continue
                checked += 1
                verdict = _check_syllable(char, start + offset, inv, attribute=True)
                if verdict:
                    findings.append(verdict)
    else:
        scope = "line"
        merged = _merge(inventories)
        for position, char in enumerate(pron):
            if not _HANGUL_SYLLABLE_RE.match(char):
                continue
            checked += 1
            verdict = _check_syllable(char, position, merged, attribute=False)
            if verdict:
                findings.append(verdict)

    return LineVerdict(
        text=text,
        pron=pron,
        scope=scope,
        checked=checked,
        skipped_words=tuple(skipped_words),
        impossible=tuple(findings),
    )
