"""CMU 발음 사전 기반 영어 → 한글 **조밀 음차** (정렬 전용).

기존 ``everyric2.text.latin_hangul.transliterate_latin``의 한계를 메우는 실험 경로다.
그쪽은 **철자**에서 출발하는 규칙 음차라 영어의 철자-발음 괴리를 못 넘는다 —
``beautiful``(/ˈbjuːtɪfʊl/)이 b-e-a를 글자대로 읽어 ``비어티펄``이 된다. 조밀화 이전에
출발점이 틀려 있어서, 조밀 규칙을 아무리 잘 걸어도 ``뷰티풀`` 계열이 안 나온다.

여기서는 CMU Pronouncing Dictionary(약 13만 표제어, BSD)로 **음소**를 먼저 얻고 거기서
한글을 만든다. ``beautiful`` → ``B Y UW1 T AH0 F AH0 L`` → ``뷰터펄``.

``latin_hangul``이 실측으로 세운 **조밀(tight) 원칙은 그대로 계승한다**: 한글 표기 관습은
자음군에 노래에 없는 모음을 끼워 넣는데(approved → 어프루브드), 그 모음이 CTC 정렬을
망친다(관습 대비 조밀이 7/7 우세, SRT ±0.3s 80%→95%). 그래서 여기서도 어말·음절말 자음은
**최대한 받침으로 흡수**하고, 받침이 불가능할 때만 ``ㅡ``를 끼운다.

음소가 출발점이라 조밀화가 오히려 쉬워진다 — 철자 경로에서는 묵음(``make``의 e)에도 규칙이
반응하지만, 음소열에는 묵음이 애초에 없다.

OOV(사전에 없는 낱말)는 ``transliterate_latin``으로 떨어진다. 가사에는 고유명사·조어가
흔하므로 폴백은 필수다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_CMU: dict[str, list[list[str]]] | None = None

# ARPABET 모음 → (중성, 활음 결합형). 강세 숫자는 떼고 본다.
# 두 번째 값은 앞에 Y가 붙었을 때 쓰는 이중모음이다(Y UW → ㅠ).
_VOWELS: dict[str, tuple[str, str | None]] = {
    "AA": ("ㅏ", "ㅑ"), "AE": ("ㅐ", "ㅒ"), "AH": ("ㅓ", "ㅕ"), "AO": ("ㅗ", "ㅛ"),
    "EH": ("ㅔ", "ㅖ"), "ER": ("ㅓ", "ㅕ"), "IH": ("ㅣ", "ㅣ"), "IY": ("ㅣ", "ㅣ"),
    "UH": ("ㅜ", "ㅠ"), "UW": ("ㅜ", "ㅠ"),
    # OW(/oʊ/)는 음소상 이중모음이지만 한 모라로 부른다 — 편면 revoke가 「리보욱」이 된다.
    # 외래어 표기법도 단모음으로 옮긴다(boat → 보트, 보우트 아님).
    "OW": ("ㅗ", "ㅛ"),
}
# 진짜 두 모라짜리 이중모음만 한글 두 글자로 편다. 노래에서도 둘로 나뉘는 것들이다.
_DIPHTHONGS: dict[str, tuple[str, str]] = {
    "AW": ("ㅏ", "ㅜ"), "AY": ("ㅏ", "ㅣ"), "EY": ("ㅔ", "ㅣ"), "OY": ("ㅗ", "ㅣ"),
}
# 영어의 «합법 초성 자음군» — 이것들은 통째로 뒤 음절에 붙으므로 앞 음절 받침으로 넘기면
# 안 된다(approved /əˈpruːvd/는 a-pproved라 P가 어절 코다가 아니다). 한글에는 자음군 초성이
# 없어 앞 자음이 ``ㅡ`` 음절로 떨어지는데, 그게 관습 표기 「어프루브드」의 「프」다.
_ONSET_CLUSTERS = frozenset(
    {("P", "R"), ("P", "L"), ("B", "R"), ("B", "L"), ("T", "R"), ("D", "R"),
     ("K", "R"), ("K", "L"), ("G", "R"), ("G", "L"), ("F", "R"), ("F", "L"),
     ("TH", "R"), ("SH", "R"), ("S", "P"), ("S", "T"), ("S", "K"), ("S", "M"),
     ("S", "N"), ("S", "L"), ("S", "W"), ("K", "W"), ("G", "W"), ("HH", "W")}
)
# W 활음 — 앞 자음 없이 모음과 합쳐지는 합성 중성.
_W_GLIDE = {"ㅏ": "ㅘ", "ㅐ": "ㅙ", "ㅓ": "ㅝ", "ㅔ": "ㅞ", "ㅣ": "ㅟ", "ㅗ": "ㅗ", "ㅜ": "ㅜ"}

_ONSETS: dict[str, str] = {
    "B": "ㅂ", "CH": "ㅊ", "D": "ㄷ", "DH": "ㄷ", "F": "ㅍ", "G": "ㄱ", "HH": "ㅎ",
    "JH": "ㅈ", "K": "ㅋ", "L": "ㄹ", "M": "ㅁ", "N": "ㄴ", "NG": "ㅇ", "P": "ㅍ",
    "R": "ㄹ", "S": "ㅅ", "SH": "ㅅ", "T": "ㅌ", "TH": "ㅅ", "V": "ㅂ", "Z": "ㅈ",
    "ZH": "ㅈ", "Y": "ㅇ", "W": "ㅇ",
}
# 받침으로 흡수 가능한 자음 → 종성. 표준 발음법 대표음에 맞춘다.
# F/V도 ㅂ으로 받는다 — 음성학적으로는 근사지만, 대안이 ``ㅡ`` 삽입(러브)뿐이고 그 모음이
# 노래에 없다는 것이 ``latin_hangul``의 실측 결론이다(love → 럽, approved → 어프룹).
# HH만 빠진다(종성 대응음이 아예 없다).
_CODAS: dict[str, str] = {
    "P": "ㅂ", "B": "ㅂ", "F": "ㅂ", "V": "ㅂ", "T": "ㅅ", "D": "ㅅ", "TH": "ㅅ",
    "DH": "ㅅ", "S": "ㅅ", "Z": "ㅅ", "SH": "ㅅ", "ZH": "ㅅ", "CH": "ㅅ", "JH": "ㅅ",
    "K": "ㄱ", "G": "ㄱ", "M": "ㅁ", "N": "ㄴ", "NG": "ㅇ", "L": "ㄹ", "R": "ㄹ",
}

_CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_JONG = "ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"
_WORD_RE = re.compile(r"[A-Za-z']+")


def _compose(onset: str, vowel: str, coda: str = "") -> str:
    """초·중·종성 낱자를 한글 완성형 한 글자로 조립한다."""
    try:
        cho = _CHO.index(onset or "ㅇ")
        jung = _JUNG.index(vowel)
    except ValueError:
        return ""
    jong = _JONG.index(coda) + 1 if coda else 0
    return chr(0xAC00 + (cho * 21 + jung) * 28 + jong)


def _load() -> dict[str, list[list[str]]]:
    global _CMU
    if _CMU is None:
        import cmudict

        _CMU = cmudict.dict()
    return _CMU


def _strip_stress(phone: str) -> str:
    return phone.rstrip("012")


@dataclass
class Unit:
    """표시 한 글자 = 정렬 스팬 하나.

    한글 글자와 **그 글자를 만든 음소·IPA를 같이** 든다. 이 대응이 이 모듈의 핵심 산출물이다 —
    IPA로 정렬해 스팬을 얻고 한글(·가나)로 표시하려면 «어느 IPA 조각이 어느 표시 글자에
    속하는가»를 알아야 하는데, 그걸 **음소 단계에서 기록**해두면 나중에 IPA 문자열을 거꾸로
    파싱할 필요가 없다.

    IPA를 역파싱하면 안 되는 이유가 있다: ``tʃ``·``eɪ``·``oʊ``는 2문자 1음소라 문자열만 보고는
    ``t``+``ʃ``인지 ``tʃ``인지 가를 근거가 없다. ARPABET에서는 ``CH``·``EY``·``OW``가 각각 한
    토큰이라 그 모호함이 아예 없다. 그래서 ARPABET을 허브로 두고 양쪽을 **동시에** 만든다.
    """

    hangul: str
    ipa: str
    phones: list[str] = field(default_factory=list)
    kana: str = ""


# 이중모음의 IPA를 두 조각으로 나눈 표. ``_DIPHTHONGS``가 한글을 두 글자로 펴므로(실제로 두
# 모라로 부른다) IPA도 같은 자리에서 갈라야 글자와 스팬이 1:1로 맞는다.
_DIPHTHONG_IPA: dict[str, tuple[str, str]] = {
    "AW": ("a", "u"), "AY": ("a", "i"), "EY": ("e", "i"), "OY": ("o", "i"),
}

# ── 가나 표시 ───────────────────────────────────────────────────────────────────
# 가나는 **표시 전용**이다. 정렬 타깃으로는 쓰지 않는다 — 실측에서 가나가 6/6 곡 모두 한글보다
# 나빴다(무분리 span score −15.81 vs −15.01, 분리 −17.59 vs −16.70). CV 음절 구조라 자음군마다
# 노래에 없는 모라가 생기기 때문이다(strength /strɛŋθ/ 1음절 → ストレングス 6모라). 정렬은 IPA가
# 하고 스팬은 유닛이 들고 있으므로, 가나가 몇 모라든 표시 품질만 좌우한다.
_KANA_ROWS: dict[str, tuple[str, str, str, str, str]] = {
    "": ("ア", "イ", "ウ", "エ", "オ"),
    "K": ("カ", "キ", "ク", "ケ", "コ"), "G": ("ガ", "ギ", "グ", "ゲ", "ゴ"),
    "S": ("サ", "スィ", "ス", "セ", "ソ"), "Z": ("ザ", "ズィ", "ズ", "ゼ", "ゾ"),
    "T": ("タ", "ティ", "トゥ", "テ", "ト"), "D": ("ダ", "ディ", "ドゥ", "デ", "ド"),
    "N": ("ナ", "ニ", "ヌ", "ネ", "ノ"), "HH": ("ハ", "ヒ", "フ", "ヘ", "ホ"),
    "B": ("バ", "ビ", "ブ", "ベ", "ボ"), "P": ("パ", "ピ", "プ", "ペ", "ポ"),
    "M": ("マ", "ミ", "ム", "メ", "モ"),
    "R": ("ラ", "リ", "ル", "レ", "ロ"), "L": ("ラ", "リ", "ル", "レ", "ロ"),
    "F": ("ファ", "フィ", "フ", "フェ", "フォ"), "V": ("ヴァ", "ヴィ", "ヴ", "ヴェ", "ヴォ"),
    "TH": ("サ", "シ", "ス", "セ", "ソ"), "DH": ("ザ", "ジ", "ズ", "ゼ", "ゾ"),
    "SH": ("シャ", "シ", "シュ", "シェ", "ショ"), "ZH": ("ジャ", "ジ", "ジュ", "ジェ", "ジョ"),
    "CH": ("チャ", "チ", "チュ", "チェ", "チョ"), "JH": ("ジャ", "ジ", "ジュ", "ジェ", "ジョ"),
    "Y": ("ヤ", "イ", "ユ", "イェ", "ヨ"), "W": ("ワ", "ウィ", "ウ", "ウェ", "ウォ"),
    "NG": ("ンガ", "ンギ", "ング", "ンゲ", "ンゴ"),
}
# ARPABET 모음 → (가나 단 인덱스, 장음 여부). 이중모음은 유닛이 이미 둘로 갈라져 있으므로
# 여기에 없고 ``_KANA_DIPHTHONG``이 조각별로 준다.
_KANA_VOWEL: dict[str, tuple[int, bool]] = {
    "AA": (0, True), "AE": (0, False), "AH": (0, False), "AO": (4, True),
    "EH": (3, False), "ER": (0, True), "IH": (1, False), "IY": (1, True),
    "UH": (2, False), "UW": (2, True), "OW": (4, True),
}
_KANA_DIPHTHONG: dict[str, tuple[tuple[int, bool], tuple[int, bool]]] = {
    "AW": ((0, False), (2, False)), "AY": ((0, False), (1, False)),
    "EY": ((3, False), (1, False)), "OY": ((4, False), (1, False)),
}
# 받침으로 흡수된 자음 → 뒤에 붙는 모라. 한글은 받침 하나로 삼키지만 가나에는 종성이 없다.
_KANA_CODA: dict[str, str] = {
    "P": "プ", "B": "ブ", "T": "ト", "D": "ド", "K": "ク", "G": "グ", "M": "ム",
    "N": "ン", "NG": "ング", "S": "ス", "Z": "ズ", "SH": "シュ", "ZH": "ジュ",
    "CH": "チ", "JH": "ジ", "F": "フ", "V": "ヴ", "TH": "ス", "DH": "ズ",
    "L": "ル", "R": "ル",
}
# 어중 받침만 다르다: ŋ 뒤에 자음이 이어지면 ``ン``이 자연스럽다(bank → バンク). 어말은 ``ング``
# 그대로다(amazing → アメイズィング). 이 하나 때문에 표를 나눈다.
_KANA_CODA_MEDIAL: dict[str, str] = {**_KANA_CODA, "NG": "ン"}
# 요음 — 자음의 イ단 + 작은 가나. ``B Y UW``(beautiful) → ビ + ュ = ビュ.
_KANA_SMALL_Y: dict[int, str] = {0: "ャ", 2: "ュ", 4: "ョ"}


def _phone_ipa(base: str, unstressed: bool = False) -> str:
    """ARPABET 한 음소 → 정렬 타깃 문자. 무강세 ``AH``만 슈와로 가른다(``word_to_ipa`` 참고)."""
    if base == "AH" and unstressed:
        return _SCHWA
    return _ARPABET_TO_IPA.get(base, "")


def _kana_for(phones: list[str], vowel: tuple[int, bool] | None) -> str:
    """한 글자 몫의 자음·활음 + 모음 → 가나.

    ``vowel``이 ``None``이면 모음 없는 글자(한글의 ``ㅡ`` 음절)라 자음을 ウ단으로 편다.
    """
    cons = [p for p in phones if p not in ("Y", "W")]
    glide_y = "Y" in phones
    glide_w = "W" in phones
    empty = ("", "", "", "", "")
    if vowel is None:
        # 모음이 없는 글자(한글 ``ㅡ`` 음절)는 ウ단이 아니라 **받침 표기**를 쓴다. ウ단을 쓰면
        # ``T``가 ``トゥ``가 되어 ``strength``가 ストゥレン…이 되는데, 외래어 관습은 ト다
        # (ストレングス·ストリート). ``_KANA_CODA``가 정확히 그 «모음 없는 자음» 표를 이미 든다.
        body = "".join(_KANA_CODA.get(c) or _KANA_ROWS.get(c, empty)[2] for c in cons)
        if not body:
            # 활음만 남은 글자 — 고유명사에 드물게 나온다(``mouw``·``lavigne``).
            body = "ウ" if glide_w else ("イ" if glide_y else "")
        return body

    index, long_vowel = vowel
    lead = "".join(_KANA_ROWS.get(c, empty)[2] for c in cons[:-1])
    main = cons[-1] if cons else ""
    if glide_w:
        # 자음 + W는 ウ단 + ワ행으로 편다(sweet → スウィ). 자음이 없으면 ワ행 그대로.
        body = (_KANA_ROWS.get(main, empty)[2] if main else "") + _KANA_ROWS["W"][index]
    elif glide_y:
        base = _KANA_ROWS.get(main, empty)[1] if main else ""
        if len(base) == 1 and index in _KANA_SMALL_Y:
            body = base + _KANA_SMALL_Y[index]
        else:
            body = _KANA_ROWS.get(main or "Y", empty)[index]
    else:
        body = _KANA_ROWS.get(main, _KANA_ROWS[""])[index]
    return lead + body + ("ー" if long_vowel else "")


def phones_to_units(phones: list[str]) -> list[Unit]:
    """ARPABET 음소열 한 낱말 → 표시 글자 단위 ``Unit`` 열.

    음절 나누기 규칙은 ``phones_to_hangul``과 **완전히 같다**(그쪽이 이 함수의 얇은 래퍼다).
    달라지는 것은 결과를 문자열로 이어붙이는 대신 글자마다 기여 음소를 남긴다는 점뿐이다.
    """
    raw = list(phones)
    unstressed = [p.endswith("0") for p in raw]
    phones = [_strip_stress(p) for p in raw]
    stress_of = {i: unstressed[i] for i in range(len(phones))}

    # 모음 위치로 자른다. 각 조각은 (앞 자음군, 모음, ...) 형태가 된다.
    nuclei = [i for i, p in enumerate(phones) if p in _VOWELS or p in _DIPHTHONGS]
    if not nuclei:
        # 모음이 없는 낱말(약어 등) — 자음마다 ㅡ 음절
        return [
            Unit(_compose(_ONSETS.get(p, ""), "ㅡ"), _phone_ipa(p), [p], _kana_for([p], None))
            for p in phones
        ]

    out: list[Unit] = []
    # ER은 r-color 모음이라 자음 R이 따로 오지 않는다. 뒤에 모음이 이어지면 그 r이 **다음
    # 음절의 초성**으로 들린다(forever /fə-rɛ-vər/ → 퍼레버). 받침 ㄹ로 처리하면 「펄에벌」이
    # 되어 음절 경계가 실제 노래와 어긋난다. 어말 ER의 r은 한국어에 안 들리므로 버린다(네버).
    pending_r = False
    for index, pos in enumerate(nuclei):
        prev = nuclei[index - 1] if index else -1
        cluster = phones[prev + 1 : pos]  # 앞 음절과 이 음절 사이 자음군
        # 최대 초성 원칙(maximal onset): 모음 사이 자음이 **하나뿐이면 뒤 음절의 초성**이다.
        # ``today``(T AH D EY)의 D는 EY의 초성이지 AH의 받침이 아니다. 자음군이 둘 이상일
        # 때만 맨 앞 하나를 앞 음절 받침으로 넘겨 삽입 모음을 줄인다.
        if index and len(cluster) >= 2 and out:
            # 뒤쪽 두 자음이 합법 초성 자음군이면 그대로 뒤 음절 몫이다 — 넘기지 않는다.
            onset_cluster = tuple(p for p in cluster[-2:] if p not in ("Y", "W"))
            if len(onset_cluster) < 2 or onset_cluster not in _ONSET_CLUSTERS:
                coda = _CODAS.get(cluster[0])
                if coda and (ord(out[-1].hangul) - 0xAC00) % 28 == 0:
                    # 받침으로 흡수된 자음은 **앞 글자의 소유**다 — IPA도 그 글자에 붙는다.
                    out[-1].hangul = chr(ord(out[-1].hangul) + _JONG.index(coda) + 1)
                    out[-1].phones.append(cluster[0])
                    out[-1].ipa += _phone_ipa(cluster[0])
                    out[-1].kana += _KANA_CODA_MEDIAL.get(cluster[0], "")
                    cluster = cluster[1:]
        # 활음 Y/W는 초성이 아니라 중성에 녹는다. 앞에 자음이 있어도 마찬가지다 — 한글은
        # 자음+활음 결합을 한 글자로 담을 수 있고(뷰·퀘·쉬), 그게 삽입 모음을 없애는
        # 조밀화 그 자체다. ``sweet``이 「스윗」(2음절)이 아니라 「쉿」(1음절)이 되는 것이
        # 노린 결과다 — 원 발음 /swiːt/도 1음절이다.
        glide_y = "Y" in cluster
        glide_w = "W" in cluster
        # 남은 자음 중 마지막 하나만 초성, 앞의 것들은 ㅡ 음절로 떨어진다. 활음은 어느
        # 쪽으로도 안 가고 모음 글자에 녹으므로 **위치로** 갈라야 IPA 순서가 보존된다.
        cons_at = [i for i, p in enumerate(cluster) if p not in ("Y", "W")]
        extra_at = set(cons_at[:-1])
        onset = _ONSETS.get(cluster[cons_at[-1]], "") if cons_at else ""
        r_linked = False
        if pending_r and not onset and not cons_at:
            onset = "ㄹ"
            r_linked = True
        pending_r = False
        # 자음군을 **위치 순서대로** 훑으며 유닛에 배분한다. 활음은 자기 글자를 못 가지므로
        # 다음에 나오는 글자에 얹히는데, 순서를 지키려면 «앞에서부터 쌓아 두었다가 글자가
        # 생길 때 함께 넘기는» 방식이어야 한다. 활음이 자음군 뒤에 오는 흔한 경우(``B Y``,
        # ``S W``)만 보고 「활음은 늘 자음 뒤」로 가정하면 활음이 **앞**에 오는 낱말에서
        # IPA 순서가 뒤집힌다(``warshawsky``의 ``W S K`` → wski가 swki로).
        carry_phones: list[str] = []
        carry_ipa = ""
        for i, p in enumerate(cluster):
            if i in extra_at:
                out.append(
                    Unit(
                        _compose(_ONSETS.get(p, ""), "ㅡ"),
                        carry_ipa + _phone_ipa(p),
                        [*carry_phones, p],
                        _kana_for([*carry_phones, p], None),
                    )
                )
                carry_phones, carry_ipa = [], ""
            else:
                carry_phones.append(p)
                carry_ipa += _phone_ipa(p)
        # 남은 것(마지막 자음 + 활음)이 이 모음 글자의 몫이다.
        head, head_ipa = carry_phones, carry_ipa
        # 앞 음절 ``ER``의 r을 이 글자의 초성으로 넘긴 경우(forever → 퍼**레**버), 가나에도
        # 같은 ラ행을 줘야 한다(ファーレヴァー). IPA에는 넣지 않는다 — 그 r은 앞 글자의 ``ɚ``에
        # 이미 들어 있어서, 여기 또 넣으면 정렬 타깃에 없는 음소가 하나 생긴다.
        kana_head = ["R", *head] if r_linked else head

        phone = phones[pos]
        pending_r = phone == "ER"
        if phone in _DIPHTHONGS:
            first, second = _DIPHTHONGS[phone]
            ipa_first, ipa_second = _DIPHTHONG_IPA[phone]
            kana_first, kana_second = _KANA_DIPHTHONG[phone]
            vowel = _W_GLIDE.get(first, first) if glide_w else first
            out.append(
                Unit(_compose(onset, vowel), head_ipa + ipa_first, [*head, phone],
                     _kana_for(kana_head, kana_first))
            )
            # 뒷 모라는 같은 음소의 후반부라 소유 음소를 다시 세지 않는다(중복 방지).
            out.append(Unit(_compose("", second), ipa_second, [], _kana_for([], kana_second)))
        else:
            plain, glided = _VOWELS[phone]
            vowel = glided if (glide_y and glided) else plain
            if glide_w:
                vowel = _W_GLIDE.get(vowel, vowel)
            out.append(
                Unit(_compose(onset, vowel), head_ipa + _phone_ipa(phone, stress_of[pos]),
                     [*head, phone], _kana_for(kana_head, _KANA_VOWEL.get(phone)))
            )

    # 어말 자음군 — 첫 하나는 받침, 나머지는 ㅡ 음절
    tail = phones[nuclei[-1] + 1 :]
    if tail and out:
        coda = _CODAS.get(tail[0])
        if coda and (ord(out[-1].hangul) - 0xAC00) % 28 == 0:
            out[-1].hangul = chr(ord(out[-1].hangul) + _JONG.index(coda) + 1)
            out[-1].phones.append(tail[0])
            out[-1].ipa += _phone_ipa(tail[0])
            # 어말 여부는 «tail이냐»가 아니라 «뒤에 음소가 더 있느냐»로 갈린다.
            # strength(… NG K TH)의 NG는 tail에 있어도 어중이다.
            table = _KANA_CODA if len(tail) == 1 else _KANA_CODA_MEDIAL
            out[-1].kana += table.get(tail[0], "")
            tail = tail[1:]
    for extra in tail:
        out.append(
            Unit(_compose(_ONSETS.get(extra, ""), "ㅡ"), _phone_ipa(extra), [extra],
                 _kana_for([extra], None))
        )
    return [u for u in out if u.hangul]


def phones_to_hangul(phones: list[str]) -> str:
    """ARPABET 음소열 한 낱말 → 한글 조밀 음차.

    모음마다 음절을 하나 세우고, 모음 사이 자음군은 **앞 음절 종성 하나 + 뒤 음절 초성 하나**로
    나눈다. 그러고도 남는 자음은 어쩔 수 없이 ``ㅡ`` 음절이 되는데(한글에 자음군 표기가 없다),
    받침을 먼저 채우므로 관습 음차보다 삽입 모음이 적다 — 그게 「조밀」의 실체다.

    규칙 자체는 ``phones_to_units``에 있다. 이 함수는 글자만 이어붙이는 얇은 래퍼이므로
    **소유권을 추가해도 기존 음차 출력은 정의상 바뀌지 않는다**.
    """
    return "".join(u.hangul for u in phones_to_units(phones))


# ARPABET → **모델이 실제로 내는 문자**로 쓴 음소 전사.
#
# 처음에는 진짜 IPA 기호를 썼다 — vocab(9,812)에 실려 있으니 음차 없이 음소를 그대로 정렬
# 타깃으로 쓸 수 있다고 봤고, 「37/39 보유」까지 확인했다. **그게 틀린 검사였다.** vocab에
# 있는 것과 모델이 그 토큰을 내는 것은 다르다. omniASR은 철자로 전사하지 IPA로 전사하지
# 않으므로, IPA 전용 기호는 훈련 목표로 한 번도 나온 적이 없어 방출이 0에 붙어 있다.
#
# 실측(Get your Wish 전곡 10,887 프레임, 2026-08-02): 비-ASCII 기호는 **전부 죽어 있었다**.
#   ɚ 최대확률 0.0000(logp −10.97) · ə 0.0012 · ɛ 0.0002 · ɪ 0.0000 · ŋ 0.0002 · ʃ 0.0000
#   반면 살아있는 ASCII: r 0.939 · a 0.938 · e 0.937 · i 0.908 · t 0.927
# 타깃 문자의 **30.2%가 모델이 절대 내지 않는 기호**였고, 정렬은 살아있는 ASCII 뼈대만으로
# 굴러가고 있었다.
#
# 더 나쁜 것은 심판이다. ``our``의 두 후보 aʊɚ/aʊr는 ɚ(죽음) vs r(살아있음)의 대결이라
# **오디오와 무관하게 항상 r이 이긴다**. en 심판이 예외 없이 「ɚ를 r로」 바꿔 온 것은
# 청각적 판단이 아니라 이 결함이었다(2026-08-02 발견).
#
# 그래서 살아있는 ASCII로 옮긴다. 영어 철자에서 그 소리를 흔히 적는 글자를 골랐다 —
# 모델이 내는 것이 철자이므로 철자에 가까울수록 방출과 맞는다.
_ARPABET_TO_IPA: dict[str, str] = {
    "P": "p", "B": "b", "T": "t", "D": "d", "K": "k", "G": "g", "F": "f", "V": "v",
    "TH": "th", "DH": "th", "S": "s", "Z": "z", "SH": "sh", "ZH": "zh", "HH": "h",
    "CH": "ch", "JH": "j", "M": "m", "N": "n", "NG": "ng", "L": "l", "R": "r",
    # ``Y``(/j/)는 ``y``다. IPA에서는 /j/를 ``j``로 쓰지만 영어 철자에서 ``j``는 /dʒ/라,
    # 철자를 내는 모델에게 ``ju``(you)를 주면 judge 쪽으로 읽힌다.
    "Y": "y", "W": "w", "IY": "i", "IH": "i", "EY": "ei", "EH": "e", "AE": "a",
    "AA": "a", "AO": "o", "OW": "ou", "UH": "u", "UW": "u", "AH": "u", "ER": "er",
    "AY": "ai", "AW": "au", "OY": "oi",
}
# 무강세 ``AH0``(슈와)가 쓸 글자. 영어에서 슈와는 ``a``로 적히는 일이 가장 많다
# (about·sofa·america). 강세 있는 ``AH``(``u``: cup·but)와 갈라 두면 ``today``가
# ``tudei``가 아니라 ``tadei``가 되어 철자에 더 가깝다.
_SCHWA = "a"


def word_to_ipa(word: str) -> str | None:
    """낱말 하나를 CMU 사전으로 IPA 전사. 사전에 없으면 None.

    무강세 ``AH0``만 슈와 ``ə``로 갈라 쓴다. ARPABET은 강세를 숫자로 따로 들고 다녀서
    ``AH``가 강세 있는 /ʌ/(cup)와 없는 /ə/(about)를 겸하는데, IPA에서는 다른 기호다.
    ``today``(T AH0 D EY1)가 ``tʌdeɪ``가 아니라 ``tədeɪ``가 되는 차이이고, 영어 가사에는
    무강세 음절이 압도적으로 많아 이 한 갈래가 전사 품질을 크게 좌우한다.
    omniASR vocab에 ``ə``가 실려 있음을 확인하고 넣었다.
    """
    entries = _load().get(word.lower().strip("'"))
    if not entries:
        return None
    pieces = []
    for phone in entries[0]:
        base = _strip_stress(phone)
        unstressed = phone.endswith("0")
        pieces.append(_SCHWA if (base == "AH" and unstressed) else _ARPABET_TO_IPA.get(base, ""))
    return "".join(pieces) or None


def transliterate_ipa(text: str) -> str:
    """한 줄 전체를 IPA로. 사전에 없는 낱말은 **그대로 둔다**.

    한글 경로와 달리 철자 음차로 폴백하지 않는다 — IPA 열에 한글이 섞이면 그 줄의 타깃이
    두 문자 체계를 오가게 되고, 그러면 무엇이 정렬을 깎았는지 읽을 수 없다. 원문 라틴은
    vocab에 있으므로 정렬 자체는 계속된다(약할 뿐이다).
    """
    return _WORD_RE.sub(lambda m: word_to_ipa(m.group(0)) or m.group(0), text)


def pronunciations(word: str) -> list[list[str]]:
    """낱말의 **모든** 사전 발음. CMU는 한 낱말에 여러 발음을 들고 있다(the ðə/ði, our 1·2음절).

    지금까지는 늘 첫 번째만 썼는데, 어느 발음으로 불렀는지는 곡마다 가수마다 다르다 —
    사전이 못 정하는 것을 오디오는 안다. 그 심판(``two_pass`` 후보 재정렬)이 이 목록을 쓴다.
    실측(2026-08-01): en 가사 출현의 36.35%가 발음이 둘 이상인 낱말이다.
    """
    return _load().get(word.lower().strip("'")) or []


def units_for_word(word: str, entry: int = 0) -> list[Unit] | None:
    """낱말 하나 → 표시 글자 단위 ``Unit`` 열. 사전에 없거나 그 후보가 없으면 None.

    ``word_to_ipa``·``word_to_hangul``이 각각 문자열만 주는 것과 달리 **대응**을 준다.
    IPA로 정렬하고 한글·가나로 표시하려면 이 대응이 있어야 한다.
    """
    entries = pronunciations(word)
    if entry >= len(entries):
        return None
    return phones_to_units(entries[entry])


_VOWEL_LETTERS = re.compile(r"[aeiouy]+")
# ARPABET 모음(강세 숫자를 뗀 형태). ER은 r-colored 모음이라 제 음절을 이룬다.
_ARPABET_VOWELS = frozenset(
    "AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".split()
)
# 두 모음이 각각 제 음절을 갖는 조합(hiatus) — abbreviated의 ia. 이중모음(ai·ea·ou)과 달리
# 갈라 읽으므로, 모음 그룹 수가 음절 수보다 모자랄 때 여기서 쪼갠다.
_HIATUS = ("ia", "io", "ua", "uo", "eo", "ie", "ii", "ao", "oi", "yi", "ui", "ae", "oe")
# 두 글자가 한 소리인 자음 — 음절 경계가 그 사이를 지나가면 안 된다. 다만 **어느 쪽 음절에
# 붙는지**가 갈린다. 통째로 뒤로 넘기면 sing이 si|nging이 되고, 통째로 앞에 붙이면 weather가
# weath|er가 된다. 그래서 방향을 따로 둔다.
#   온셋형 — 다음 음절의 첫소리로 간다(wea|ther, wa|shing, tea|cher).
_ONSET_DIGRAPHS = ("th", "sh", "ch", "wh", "ph", "qu")
#   코다형 — 앞 음절의 받침으로 남는다. ng·ck는 영어에서 어두에 설 수 없는 소리라
#   구조적으로 앞에 붙고(sing|ing, king|dom, pick|ing), gh도 같다(laugh|ing).
_CODA_DIGRAPHS = ("ng", "ck", "gh")
_DIGRAPHS = _ONSET_DIGRAPHS + _CODA_DIGRAPHS


def _avoid_digraph(lowered: str, cut: int) -> int:
    """경계 ``cut``이 digraph 한가운데면 그 소리가 속한 음절 쪽으로 비킨다."""
    pair = lowered[max(cut - 1, 0):cut + 1]
    if pair in _ONSET_DIGRAPHS:
        return cut - 1
    if pair in _CODA_DIGRAPHS:
        return cut + 1
    return cut


def syllable_count(phones: list[str]) -> int:
    """ARPABET 음소열의 음절 수 — 모음에 붙은 강세 숫자(0/1/2)를 센다.

    ``Unit`` 개수를 쓰면 안 된다: 한글 표시는 CV 구조라 자음군마다 ㅡ를 끼워 넣어
    (strength → 스트렝쓰) 실제 음절보다 길어진다. 사전 전체에서 1.32배였다.
    """
    return sum(1 for phone in phones if phone and phone[-1].isdigit())


def syllabify_spelling(word: str, want: int) -> list[str] | None:
    """영어 **철자**를 ``want``개 음절 조각으로 가른다. 못 맞추면 None.

    음절 수는 CMU가 이미 준다(``syllable_count``). 없는 것은 그 수만큼 철자를 가르는
    지점뿐인데, 영어 철자에는 음절 경계 표시가 없다 — 사전도 음소열만 주고 음소가 어느
    철자에서 왔는지는 안 알려준다. 그래서 모음 글자를 세어 맞춘다.

    실측(2026-08-01): CMU 표제어 117,493개에서 94.26%, **실제 en 가사 출현 기준 97.46%**.
    남는 몫은 낱말을 통째로 두는 폴백이라 «덜 쪼갠» 것이지 틀린 것이 아니다.

    맞춰지지 않는 대표 유형은 복합어(sidewalk=side+walk)와 음절성 자음(rhythm의 m),
    그리고 실제 발음이 축약되는 경우(everything → ev'rything)다. 마지막 유형은 규칙으로
    못 고친다 — 곡마다 다르므로 오디오만 답할 수 있다.
    """
    lowered = word.lower()
    groups = [[m.start(), m.end()] for m in _VOWEL_LETTERS.finditer(lowered)]
    if not groups:
        return None

    # ① 묵음 어미 — make·abandoned·abates. 끝의 e(+d/s)는 제 음절이 아니라 앞 음절 소속이다.
    while len(groups) > want and len(groups) >= 2 and lowered[groups[-1][0]:] in ("e", "es", "ed"):
        groups[-2][1] = groups[-1][1]
        groups.pop()

    # ①-2 **중간** 묵음 e — side|walk, some|day, eve|ning. 복합어 앞부분의 e가 대표적이다.
    # 수가 남을 때만 손대므로(이미 맞으면 아예 안 건드린다) 살아 있는 e를 잘못 먹지 않는다.
    # 첫 그룹은 흡수시킬 앞이 없어 건너뛴다.
    index = 1
    while len(groups) > want and index < len(groups):
        if lowered[groups[index][0]:groups[index][1]] == "e":
            groups[index - 1][1] = groups[index][1]
            groups.pop(index)
        else:
            index += 1

    # ② hiatus — 한 모음 그룹이 실은 두 음절인 경우를 갈라 수를 채운다.
    index = 0
    while len(groups) < want and index < len(groups):
        start, end = groups[index]
        chunk = lowered[start:end]
        cut = next((k for k in range(1, len(chunk)) if chunk[k - 1:k + 1] in _HIATUS), 0)
        if cut:
            groups[index] = [start, start + cut]
            groups.insert(index + 1, [start + cut, end])
            index += 2
        else:
            index += 1

    if len(groups) != want:
        return None

    # 모음 그룹 사이 자음군을 앞뒤로 나눈다. 자음 1개면 통째로 뒤 음절 온셋(be-gin),
    # 2개 이상이면 반씩 갈라 앞 음절이 받침을 갖는다(ban-dit).
    cuts = [0]
    for left, right in zip(groups, groups[1:]):
        gap = right[0] - left[1]
        cut = _avoid_digraph(lowered, left[1] + (0 if gap <= 1 else gap // 2))
        # 비킨 결과가 자음군 밖으로 나가면 안 된다 — 모음을 삼키거나 빈 조각이 생긴다.
        cuts.append(min(max(cut, left[1]), right[0]))
    cuts.append(len(word))
    return [word[cuts[k]:cuts[k + 1]] for k in range(want)]


def _split_evenly(word: str, want: int) -> list[str]:
    """규칙이 경계를 못 찾았을 때 — 철자를 길이 비례로 ``want``조각. **음절 수는 지킨다.**

    통째로 두면 안 되는 이유가 있다. 음절 수는 오디오(심판)가 고른 발음이 정하는데, 표시가
    한 조각이면 그 수가 버려져 **그 낱말 구간에서 카라오케가 멈춘다**. 경계가 한 글자
    어긋나는 것과 구간이 통째로 멈추는 것은 손상의 크기가 다르다 — 후자가 훨씬 크다.

    경계가 digraph를 가르지 않게만 비킨다(``rhythm`` → ``rhy|thm``, ``rh|ythm`` 아님).
    """
    if want <= 1 or len(word) <= want:
        return [word]
    lowered = word.lower()
    step = len(word) / want
    cuts = [0]
    for k in range(1, want):
        cut = _avoid_digraph(lowered, round(k * step))
        cuts.append(min(max(cut, cuts[-1] + 1), len(word) - (want - k)))
    cuts.append(len(word))
    return [word[cuts[k]:cuts[k + 1]] for k in range(want)]


def syllabify_unknown(word: str) -> list[str]:
    """**사전에 없는** 낱말을 음절로 가른다 — 음절 수까지 모음 글자로 추정한다.

    사전에 있으면 CMU가 음절 수를 주지만(``syllable_count``) OOV는 그마저 없다. 그렇다고
    통째로 두면 그 낱말만 세그 하나가 되어 **그 구간에서 카라오케가 멈춘다** — 한글 경로가
    OOV를 철자 길이에 비례 배분하는 것도 같은 이유다(``_ipa_display_units``).

    모음 그룹 수를 음절 수로 보되 끝의 묵음 e는 뺀다: ``weathervane`` → ea·e·a·(e) → 3음절
    → ``wea|ther|vane``. 가사의 고유명사·조어가 대개 복합어라 이 추정이 잘 맞는다.
    """
    lowered = word.lower()
    groups = [[m.start(), m.end()] for m in _VOWEL_LETTERS.finditer(lowered)]
    if not groups:
        return [word]
    if len(groups) >= 2 and lowered[groups[-1][0]:] == "e":
        groups.pop()
    want = len(groups)
    return syllabify_spelling(word, want) or _split_evenly(word, want)


def syllable_units_for_word(word: str, entry: int = 0) -> list[tuple[str, list[Unit]]] | None:
    """낱말 → [(철자 음절 조각, 그 음절의 ``Unit`` 열)]. 못 가르면 None.

    ``units_for_word``가 «표시 글자 : IPA» 대응을 주는 데 비해 이쪽은 **원문 철자 : IPA**
    대응을 준다. 같은 정렬에서 원문 영어도 음절로 점등시키기 위한 것이다 — 영어 성악 악보가
    ``beau-ti-ful``로 음표마다 음절을 배치하는 그 단위이고, 낱말로 묶으면 한 낱말 안에서
    음높이·박자가 바뀌는 구조를 통째로 버리게 된다.
    """
    entries = pronunciations(word)
    if entry >= len(entries):
        return None
    phones = entries[entry]
    want = syllable_count(phones)
    if not want:
        return None
    pieces = syllabify_spelling(word, want) or _split_evenly(word, want)

    # 원본 phones 인덱스 → 음절 번호. 첫 모음 앞의 자음은 첫 음절 온셋으로 둔다.
    syllable_of: list[int] = []
    slot = -1
    for phone in phones:
        if phone and phone[-1].isdigit():
            slot = min(slot + 1, want - 1)
        syllable_of.append(max(slot, 0))

    # Unit을 음절에 배정. **그 Unit이 든 모음의 위치**로 정하는 것이 요점이다 — Unit 시작
    # 위치로 정하면 온셋 자음이 앞 음절로 끌려간다(pointing의 ``팅``이 첫 음절이 돼 버린다).
    # 모음이 없는 Unit(코다 자음, 이중모음 뒷조각)은 앞 Unit의 음절을 따른다.
    # ``Unit.phones``는 강세가 벗겨져 있으므로(UW·AH) 숫자가 아니라 모음 집합으로 가른다.
    groups: list[list[Unit]] = [[] for _ in range(want)]
    cursor = last = 0
    for unit in phones_to_units(phones):
        vowel_at = next(
            (cursor + k for k, phone in enumerate(unit.phones)
             if _strip_stress(phone) in _ARPABET_VOWELS),
            None,
        )
        if vowel_at is not None and vowel_at < len(syllable_of):
            last = syllable_of[vowel_at]
        groups[last].append(unit)
        cursor += len(unit.phones)
    return list(zip(pieces, groups))


def word_to_hangul(word: str) -> str | None:
    """낱말 하나를 CMU 사전으로 한글 조밀 음차. 사전에 없으면 None."""
    entries = _load().get(word.lower().strip("'"))
    if not entries:
        return None
    return phones_to_hangul(entries[0]) or None


def transliterate_cmu(text: str) -> str:
    """한 줄 전체 — 사전에 있는 낱말은 CMU 경로, 없으면 철자 음차로 폴백한다.

    가사에는 고유명사·조어·의성어가 흔해 OOV가 상시 발생한다. 폴백이 없으면 그 낱말만
    정렬 타깃에서 사라져 라인 전체 타이밍이 밀린다.
    """
    from everyric2.text.latin_hangul import transliterate_latin

    def replace(match: Any) -> str:
        word = match.group(0)
        return word_to_hangul(word) or transliterate_latin(word)

    return _WORD_RE.sub(replace, text)
