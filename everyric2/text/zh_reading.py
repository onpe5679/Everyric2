"""중국어(한자) 가사 → 병음·한글 음차·가나 근사 3표기 결정적 변환.

``kana_hangul``(가나→한글)·``kana_romaji``(가나→로마자)·``ko_reading``(한글→가나/로마자)가
ja·ko 곡에서 맡는 자리를 zh 곡에서 맡는다. 역할 분담도 같다 — **무엇을 읽는가**(다음자
판정: 长江 cháng / 长大 zhǎng, 重要 zhòng / 重复 chóng)는 사전과 구(句) 정보를 가진
``pypinyin``이 하고, **어떻게 적는가**(표기 관례)만 이 모듈이 한다. 반대로 하면 ja에서 이미
겪은 실수가 재발한다 — 읽기를 표에 박으면 문맥 의존 독음이 통째로 틀리고, 표기까지 외부
라이브러리에 맡기면 표기 관례를 고칠 손잡이가 없어진다.

## 세 표기

* ``romaji`` = **병음(성조 부호 포함)**. 라틴 표기를 보는 사용자에게 zh 곡의 로마자는
  병음이다. 성조는 같은 철자의 다른 낱말을 가르는 정보라 기본으로 남긴다(``tone="none"``
  으로 뺄 수 있다). 새 키("pinyin")를 만들지 않고 ``romaji``에 싣는 이유는 클라이언트
  표기 키가 hangul/romaji/kana 셋뿐이기 때문이다(everyric2-chrome ``src/lib/lang.ts``의
  ``PronScript``) — 새 키는 현재 확장도 구버전도 못 읽는다.
* ``hangul`` = **국립국어원 외래어 표기법(한어 병음 자모와 한글 대조표)**. 중국어
  인명·지명 표기의 표준이고(北京→베이징, 上海→상하이, 毛泽东→마오쩌둥) 한국어권
  사용자가 이미 그렇게 읽는다. 성모표와 운모표를 조합하고, 'ㅈ·ㅉ·ㅊ' 뒤에서 이중모음을
  단모음으로 적는 세칙만 따로 적용한다(江 jiāng → 쟝이 아니라 장. 天 tiān은 초성이 ㅌ이라
  세칙 대상이 아니어서 톈 그대로).
* ``kana`` = 일본어권 사용자용 **근사**다. 일본어에 [ɤ]·[ʐ]도 성조도 없어 "정답"이
  없으므로, 일본 매체의 다수 표기(上海→シャンハイ, 北京→ベイジン, 中国→ジョングオ,
  天安門→ティエンアンメン)를 재현하는 표를 쓴다. -n과 -ng는 둘 다 ン으로 합류한다 —
  일본어 음운으로는 가를 수단이 없다.

## 한자 한 글자 = 한 음절

그래서 ``derive_zh_display_units``의 소유자 배열이 원문 글자와 1:1이고, ja 경로
(``align_target.derive_ja_display_units``)가 한자 한 글자를 여러 모라로 펼치며 겪는 타이밍
문제가 여기엔 없다. 儿化(花儿 huā-ér)도 ``pypinyin``이 주는 대로 두 음절 두 칸에 둔다 —
한 칸으로 합치면 소유자 배열이 원문과 어긋난다.

## 정렬 타깃은 원문 한자 그대로

``everyric2.alignment.ctc_engine``의 zh 경로는 한자 vocab을 가진 모델을 쓴다
(``LANG_MODEL_MAP["zh"]``, MMS ``cmn-script_simplified`` 어댑터 vocab의 한자 4,419자 —
같은 파일의 어댑터 커버리지 표). 그래서 ``LineUnits.target``은 **원문 그대로**이고
(``align_target.native_units``와 같은 꼴) 표기는 그 위에 얹기만 한다. 병음을 타깃으로
삼으면 정렬 단위가 글자에서 라틴 철자로 흩어지는데 그럴 이유가 없다.

## 표시 문자열은 ``zh_pron_variants``로 만든다

``align_target.join_display``를 쓰면 안 된다 — 그 함수는 소유자를 **붙여** 잇는데, 병음은
음절 사이에 공백이 없으면 못 읽는다(wǒàinǐ). 한글·가나는 붙여 쓰고 병음만 띄우는 규칙은
표기마다 다르므로 소유자 배열이 아니라 이 모듈의 조립 함수가 정한다.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from everyric2.text.align_target import LineUnits, _distribute_by_length

# 클라이언트가 아는 표기 키와 같은 집합(``align_target.JA_DISPLAY_KEYS``와 동일한 계약).
ZH_DISPLAY_KEYS: tuple[str, ...] = ("hangul", "kana", "romaji")

# 한자 — worker/refine_window의 _JA_CHAR_RE 중 한자 구역만 떼어낸 것(가나는 뺀다).
# CJK 통합 한자 + 확장 A. 확장 B 이상(U+20000~)은 pypinyin 사전에도 거의 없어 넣지 않는다.
_HAN_RE = re.compile("[㐀-鿿]")
# 라틴 낱말 — ``align_target._LATIN_WORD_RE``와 같은 규칙(아포스트로피는 낱말 중간만).
_LATIN_WORD_RE = re.compile(r"[A-Za-z][A-Za-z']*")
_SPACE_RE = re.compile(r"\s+")

# 성조 결합 문자 — 마크론(1성)·양음(2성)·캐런(3성)·억음(4성). ü의 분음부(U+0308)는
# 여기 없으므로 성조만 벗기고 ü는 살아남는다(lǜ → lü, nǚ → nü).
_TONE_MARKS = frozenset("̄́̌̀")


# ---------------------------------------------------------------------------
# 병음 음절 분해
# ---------------------------------------------------------------------------

# 성모 — 긴 것부터 봐야 zh/ch/sh가 z/c/s로 잘리지 않는다.
_INITIALS: tuple[str, ...] = (
    "zh", "ch", "sh",
    "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h",
    "j", "q", "x", "r", "z", "c", "s",
)

# 성모 없는 철자(y·w로 시작) → 운모 이름. 병음 정서법이 i·u·ü로 시작하는 운모를 성모가
# 없을 때 y·w로 고쳐 쓰기 때문에 되돌려야 표를 한 벌만 유지할 수 있다.
_ZERO_INITIAL_FINALS: dict[str, str] = {
    "yi": "i", "ya": "ia", "ye": "ie", "yao": "iao", "you": "iou",
    "yan": "ian", "yin": "in", "yang": "iang", "ying": "ing", "yong": "iong",
    "yo": "io",
    "wu": "u", "wa": "ua", "wo": "uo", "wai": "uai", "wei": "uei",
    "wan": "uan", "wen": "uen", "wang": "uang", "weng": "ueng",
    "yu": "ü", "yue": "üe", "yuan": "üan", "yun": "ün",
}

# 축약 철자 → 정식 운모 이름. 병음은 성모가 있을 때 iou/uei/uen의 가운데 모음을 적지
# 않는다(jiu=jiou, gui=guei, gun=guen).
_CONTRACTED_FINALS: dict[str, str] = {"iu": "iou", "ui": "uei", "un": "uen"}

# 권설음(zh·ch·sh·r)과 설치음(z·c·s) 뒤의 i는 [i]가 아니라 성모의 자리를 그대로 끄는
# 무성 모음이다. 한글은 둘 다 ㅡ지만(즈·쯔) 가나는 행이 갈려(ジー·ズー) 운모를 나눈다.
_RETROFLEX = frozenset({"zh", "ch", "sh", "r"})
_DENTAL_SIBILANT = frozenset({"z", "c", "s"})

# 음절성 비음 감탄사(嗯 ń·呣 ḿ·哼 hng) — 성모/운모 구조 밖이라 통째로 표에 둔다.
_INTERJECTIONS: dict[str, tuple[str, str]] = {
    "n": ("은", "ン"), "ng": ("응", "ン"), "m": ("음", "ム"),
    "hm": ("흠", "フム"), "hng": ("흥", "フン"), "ê": ("에", "エ"),
}


def _toneless(syllable: str) -> str:
    """성조 부호만 벗긴 병음. ü의 분음부는 살린다(lǜ → lü)."""
    decomposed = unicodedata.normalize("NFD", syllable)
    stripped = "".join(ch for ch in decomposed if ch not in _TONE_MARKS)
    return unicodedata.normalize("NFC", stripped)


def split_syllable(syllable: str) -> tuple[str, str] | None:
    """병음 음절 → (성모, 운모). 성모가 없으면 성모는 빈 문자열. 못 가르면 None.

    입력은 성조가 있어도 되고(벗겨서 본다) ü를 v로 적은 형태여도 된다(``lazy_pinyin``
    기본 출력이 lv다 — 이 모듈은 성조 표기 출력에서 성조만 벗겨 쓰므로 실제로는 ü로
    들어오지만, 다른 호출자를 위해 v도 받는다).
    """
    plain = _toneless(syllable).strip().lower().replace("v", "ü")
    if not plain:
        return None
    if plain in _INTERJECTIONS:
        return "", plain
    if plain[0] in "yw":
        final = _ZERO_INITIAL_FINALS.get(plain)
        return ("", final) if final else None

    initial = ""
    for candidate in _INITIALS:
        if plain.startswith(candidate):
            initial = candidate
            break
    final = plain[len(initial) :]
    if not final:
        return None

    # j·q·x는 ü하고만 결합하므로 병음 정서법이 분음부를 생략한다(ju=jü, juan=jüan).
    if initial in ("j", "q", "x") and final.startswith("u"):
        final = "ü" + final[1:]
    final = _CONTRACTED_FINALS.get(final, final)
    if final == "i":
        if initial in _RETROFLEX:
            final = "-i"
        elif initial in _DENTAL_SIBILANT:
            final = "-iz"
    return initial, final


# ---------------------------------------------------------------------------
# 한글 — 국립국어원 외래어 표기법(한어 병음 자모와 한글 대조표)
# ---------------------------------------------------------------------------

_CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_HANGUL_BASE = 0xAC00

_HANGUL_INITIALS: dict[str, str] = {
    "": "ㅇ",
    "b": "ㅂ", "p": "ㅍ", "m": "ㅁ", "f": "ㅍ",
    "d": "ㄷ", "t": "ㅌ", "n": "ㄴ", "l": "ㄹ",
    "g": "ㄱ", "k": "ㅋ", "h": "ㅎ",
    "j": "ㅈ", "q": "ㅊ", "x": "ㅅ",
    "zh": "ㅈ", "ch": "ㅊ", "sh": "ㅅ", "r": "ㄹ",
    "z": "ㅉ", "c": "ㅊ", "s": "ㅆ",
}

# 운모의 **성모 없는 형태**를 그대로 적는다 — 성모가 붙으면 첫 음절의 초성 자리(ㅇ)만
# 갈아 끼우면 되므로(_with_onset) 표를 한 벌만 둔다: 아오→마오, 웅→중, 워→궈, 옌→톈.
_HANGUL_FINALS: dict[str, str] = {
    "a": "아", "o": "오", "e": "어", "ê": "에", "er": "얼",
    "ai": "아이", "ei": "에이", "ao": "아오", "ou": "어우",
    "an": "안", "en": "언", "ang": "앙", "eng": "엉", "ong": "웅",
    "i": "이", "ia": "야", "ie": "예", "iao": "야오", "iou": "유",
    "ian": "옌", "in": "인", "iang": "양", "ing": "잉", "iong": "융", "io": "요",
    "u": "우", "ua": "와", "uo": "워", "uai": "와이", "uei": "웨이",
    "uan": "완", "uen": "원", "uang": "왕", "ueng": "웡",
    "ü": "위", "üe": "웨", "üan": "위안", "ün": "윈",
    "-i": "으", "-iz": "으",
}

# 성모가 붙으면 표기가 갈리는 운모. 병음 정서법이 uei·uen의 가운데 e를 생략하는 것
# (guei→gui, kuen→kun)과 짝이 맞는 한글 관례다 — 桂林은 궤이린이 아니라 구이린,
# 昆明은 퀀밍이 아니라 쿤밍이다(回族 후이족, 崔健 추이젠, 春 춘도 같은 자리).
# 성모가 없을 때는 기본형이 맞는다(魏 웨이, 温州 원저우).
_HANGUL_FINALS_WITH_INITIAL: dict[str, str] = {"uei": "우이", "uen": "운"}

# 'ㅈ·ㅉ·ㅊ' 뒤에서 이중모음을 단모음으로 적는 세칙(외래어 표기법 제3장 중국어 표기).
# ㅕ→ㅔ는 운모 ian(옌) 한 자리에만 걸린다: 錢 qián은 천이 아니라 첸이다.
_JQ_COLLAPSE: dict[str, str] = {"ㅑ": "ㅏ", "ㅕ": "ㅔ", "ㅖ": "ㅔ", "ㅛ": "ㅗ", "ㅠ": "ㅜ"}
_JQ_INITIALS = frozenset({"j", "q", "zh", "ch", "z", "c"})


def _with_onset(syllable: str, cho: str, *, collapse: bool) -> str:
    """한글 음절의 초성을 ``cho``로 갈아 끼운다. ``collapse``면 세칙(이중모음→단모음)도 적용."""
    code = ord(syllable) - _HANGUL_BASE
    jung_index = (code // 28) % 21
    jong_index = code % 28
    if collapse:
        replacement = _JQ_COLLAPSE.get(_JUNG[jung_index])
        if replacement:
            jung_index = _JUNG.index(replacement)
    return chr(_HANGUL_BASE + (_CHO.index(cho) * 21 + jung_index) * 28 + jong_index)


def syllable_to_hangul(syllable: str) -> str:
    """병음 음절 하나 → 한글 음차. 못 읽으면 빈 문자열."""
    split = split_syllable(syllable)
    if not split:
        return ""
    initial, final = split
    if final in _INTERJECTIONS:
        return _INTERJECTIONS[final][0]
    base = (_HANGUL_FINALS_WITH_INITIAL if initial else _HANGUL_FINALS).get(final)
    base = base or _HANGUL_FINALS.get(final)
    cho = _HANGUL_INITIALS.get(initial)
    if not base or not cho:
        return ""
    head = _with_onset(base[0], cho, collapse=initial in _JQ_INITIALS)
    return head + base[1:]


# ---------------------------------------------------------------------------
# 가나 — 일본 매체 다수 표기의 재현(근사)
# ---------------------------------------------------------------------------

# 성모 × 운모 핵모음 → 가나. ko_reading._CHO_ROWS와 같은 모양이다.
_KANA_ROWS: dict[str, dict[str, str]] = {
    "": {"a": "ア", "i": "イ", "u": "ウ", "e": "エ", "o": "オ"},
    "b": {"a": "バ", "i": "ビ", "u": "ブ", "e": "ベ", "o": "ボ"},
    "p": {"a": "パ", "i": "ピ", "u": "プ", "e": "ペ", "o": "ポ"},
    "m": {"a": "マ", "i": "ミ", "u": "ム", "e": "メ", "o": "モ"},
    "f": {"a": "ファ", "i": "フィ", "u": "フ", "e": "フェ", "o": "フォ"},
    "d": {"a": "ダ", "i": "ディ", "u": "ドゥ", "e": "デ", "o": "ド"},
    "t": {"a": "タ", "i": "ティ", "u": "トゥ", "e": "テ", "o": "ト"},
    "n": {"a": "ナ", "i": "ニ", "u": "ヌ", "e": "ネ", "o": "ノ"},
    "l": {"a": "ラ", "i": "リ", "u": "ル", "e": "レ", "o": "ロ"},
    "g": {"a": "ガ", "i": "ギ", "u": "グ", "e": "ゲ", "o": "ゴ"},
    "k": {"a": "カ", "i": "キ", "u": "ク", "e": "ケ", "o": "コ"},
    "h": {"a": "ハ", "i": "ヒ", "u": "フ", "e": "ヘ", "o": "ホ"},
    # j·zh는 유성 파찰음이 아니지만(둘 다 무기음) 일본어 표기 관례가 ジ행이다
    # (蒋介石 → ジアン・ジエシー). q·ch는 유기음이라 チ행, x·sh는 마찰음이라 シ행.
    "j": {"a": "ジャ", "i": "ジ", "u": "ジュ", "e": "ジェ", "o": "ジョ"},
    "q": {"a": "チャ", "i": "チ", "u": "チュ", "e": "チェ", "o": "チョ"},
    "x": {"a": "シャ", "i": "シ", "u": "シュ", "e": "シェ", "o": "ショ"},
    "zh": {"a": "ジャ", "i": "ジ", "u": "ジュ", "e": "ジェ", "o": "ジョ"},
    "ch": {"a": "チャ", "i": "チ", "u": "チュ", "e": "チェ", "o": "チョ"},
    "sh": {"a": "シャ", "i": "シ", "u": "シュ", "e": "シェ", "o": "ショ"},
    "r": {"a": "ラ", "i": "リ", "u": "ル", "e": "レ", "o": "ロ"},
    "z": {"a": "ザ", "i": "ジ", "u": "ズ", "e": "ゼ", "o": "ゾ"},
    "c": {"a": "ツァ", "i": "ツィ", "u": "ツ", "e": "ツェ", "o": "ツォ"},
    "s": {"a": "サ", "i": "シ", "u": "ス", "e": "セ", "o": "ソ"},
}

# 운모 → (성모와 결합할 핵모음, 뒤에 그대로 붙일 가나).
# -n/-ng가 둘 다 ン인 것은 근사가 아니라 한계다(모듈 docstring).
_KANA_FINALS: dict[str, tuple[str, str]] = {
    "a": ("a", ""), "o": ("o", ""), "e": ("u", "ー"), "ê": ("e", ""), "er": ("a", "ル"),
    "ai": ("a", "イ"), "ei": ("e", "イ"), "ao": ("a", "オ"), "ou": ("o", "ウ"),
    "an": ("a", "ン"), "en": ("e", "ン"), "ang": ("a", "ン"), "eng": ("e", "ン"),
    "ong": ("o", "ン"),
    "i": ("i", "ー"), "ia": ("i", "ア"), "ie": ("i", "エ"), "iao": ("i", "アオ"),
    "iou": ("i", "ウ"), "ian": ("i", "エン"), "in": ("i", "ン"), "iang": ("i", "アン"),
    "ing": ("i", "ン"), "iong": ("i", "オン"), "io": ("i", "オ"),
    "u": ("u", "ー"), "ua": ("u", "ア"), "uo": ("u", "オ"), "uai": ("u", "アイ"),
    # uei·uen은 성모가 붙으면 가운데 e가 죽는다(桂林 クイリン, 昆明 クンミン) — 한글의
    # 구이린·쿤밍과 같은 자리다. 성모가 없을 때(wei ウェイ, wen ウェン)는 아래 zero 표가 덮는다.
    "uei": ("u", "イ"), "uan": ("u", "アン"), "uen": ("u", "ン"),
    "uang": ("u", "アン"), "ueng": ("u", "オン"),
    # ü계는 성모의 イ단에 拗音 ュ를 얹어 [y]를 근사한다(徐 xú → シュイ, 绿 lǜ → リュイ).
    "ü": ("i", "ュイ"), "üe": ("i", "ュエ"), "üan": ("i", "ュエン"), "ün": ("i", "ュン"),
    # 권설 -i는 イ단 장음(知 zhī → ジー), 설치 -i는 ウ단 장음(四 sì → スー).
    "-i": ("i", "ー"), "-iz": ("u", "ー"),
}

# 성모가 없는 y·w 철자는 표 조합(イ+ア=イア)이 아니라 활음을 살린 관례 표기를 쓴다
# (ヤ·ワ). 한글 쪽은 운모표의 기본형이 이미 성모 없는 형태라 이런 표가 필요 없다.
_KANA_ZERO_INITIAL: dict[str, str] = {
    "i": "イー", "ia": "ヤ", "ie": "イエ", "iao": "ヤオ", "iou": "ヨウ",
    "ian": "イエン", "in": "イン", "iang": "ヤン", "ing": "イン", "iong": "ヨン",
    "io": "ヨ",
    "u": "ウー", "ua": "ワ", "uo": "ウオ", "uai": "ワイ", "uei": "ウェイ",
    "uan": "ワン", "uen": "ウェン", "uang": "ワン", "ueng": "ウォン",
    "ü": "ユイ", "üe": "ユエ", "üan": "ユエン", "ün": "ユン",
}


def syllable_to_kana(syllable: str) -> str:
    """병음 음절 하나 → 가타카나 근사. 못 읽으면 빈 문자열."""
    split = split_syllable(syllable)
    if not split:
        return ""
    initial, final = split
    if final in _INTERJECTIONS:
        return _INTERJECTIONS[final][1]
    if not initial and final in _KANA_ZERO_INITIAL:
        return _KANA_ZERO_INITIAL[final]
    entry = _KANA_FINALS.get(final)
    row = _KANA_ROWS.get(initial)
    if not entry or not row:
        return ""
    nucleus, tail = entry
    return row[nucleus] + tail


# ---------------------------------------------------------------------------
# 줄 단위 읽기
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ZhReading:
    """원문 한 조각의 3표기 읽기.

    ``kind``는 ``han``(한자 한 글자)·``latin``(라틴 낱말)·``space``(공백)·``other``
    (구두점·숫자 등 그대로 통과하는 것)다. ``source``를 이으면 원문이 그대로 복원된다 —
    소유자 배열이 원문 글자와 1:1이라는 계약(모듈 docstring)이 여기서 나온다.
    """

    source: str
    kind: str
    hangul: str
    kana: str
    pinyin: str


def has_han(text: str) -> bool:
    """한자가 하나라도 있는가. 곡 언어 판정에 쓰는 호출부용(worker 참고)."""
    return bool(_HAN_RE.search(text))


def _pinyin_run(run: str) -> list[str]:
    """한자 런 → 글자당 병음(성조 부호 포함). 읽기를 못 얻은 자리는 빈 문자열.

    **런 전체를 한 번에** 넘겨야 한다 — ``pypinyin``은 구(句) 사전으로 다음자를 가르므로
    (长江 cháng / 长大 zhǎng) 글자 단위로 부르면 그 판단이 통째로 죽는다. 결과 길이가
    입력과 다르면(사전에 없는 확장 한자 등으로 pypinyin이 조각을 묶어 낸 경우) 글자
    단위로 다시 불러 1:1을 지킨다 — 표시 소유자 배열의 전제다.
    """
    from pypinyin import Style, pinyin

    def _flat(text: str) -> list[str]:
        groups = pinyin(text, style=Style.TONE, heteronym=False)
        return [(group[0] if group else "") for group in groups]

    result = _flat(run)
    if len(result) == len(run):
        return result
    per_char: list[str] = []
    for char in run:
        single = _flat(char)
        per_char.append(single[0] if single else "")
    return per_char


def _latin_readings(word: str) -> tuple[str, str]:
    """라틴 낱말 → (한글 음차, 가나 근사). ko_reading의 혼합 줄 처리와 같은 도구를 쓴다."""
    from everyric2.text.ko_reading import latin_to_kana
    from everyric2.text.latin_hangul import transliterate_latin

    try:
        hangul = transliterate_latin(word) or word
    except Exception:
        hangul = word
    try:
        kana = latin_to_kana(word) or word
    except Exception:
        kana = word
    return hangul, kana


def read_line(text: str, *, tone: str = "mark") -> list[ZhReading]:
    """중국어 한 줄 → 조각별 3표기 읽기. 이 모듈의 다른 산출물은 전부 여기서 파생된다.

    ``tone``: ``"mark"``(기본, mā)·``"none"``(ma). 성조는 병음 표기에만 영향을 준다 —
    한글·가나 표기에는 성조를 적을 자리가 애초에 없다.

    한자가 아닌 것은 계열대로 갈라 통과시킨다: 라틴 낱말은 en 곡과 같은 도구로 음차하고
    (혼합 줄이 흔하다), 공백·구두점·숫자는 원형 그대로 세 표기에 같이 실린다.
    """
    pieces: list[ZhReading] = []
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if _HAN_RE.match(char):
            end = index
            while end < length and _HAN_RE.match(text[end]):
                end += 1
            run = text[index:end]
            for offset, syllable in enumerate(_pinyin_run(run)):
                source = run[offset]
                plain = _toneless(syllable)
                shown = syllable if tone == "mark" else plain
                pieces.append(
                    ZhReading(
                        source=source,
                        kind="han",
                        hangul=syllable_to_hangul(plain) or source,
                        kana=syllable_to_kana(plain) or source,
                        pinyin=shown or source,
                    )
                )
            index = end
            continue

        match = _LATIN_WORD_RE.match(text, index)
        if match:
            word = match.group()
            hangul, kana = _latin_readings(word)
            pieces.append(
                ZhReading(source=word, kind="latin", hangul=hangul, kana=kana, pinyin=word)
            )
            index = match.end()
            continue

        match = _SPACE_RE.match(text, index)
        if match:
            run = match.group()
            pieces.append(ZhReading(source=run, kind="space", hangul=" ", kana=" ", pinyin=" "))
            index = match.end()
            continue

        pieces.append(ZhReading(source=char, kind="other", hangul=char, kana=char, pinyin=char))
        index += 1
    return pieces


def _join(pieces: list[ZhReading], key: str, *, space_between_syllables: bool) -> str:
    """조각별 읽기를 표시 문자열로 잇는다.

    병음만 음절 사이를 띄운다(``space_between_syllables``) — 붙이면 못 읽는다(wǒàinǐ).
    한글·가나는 ja·ko 표기와 같이 붙여 쓰고, 라틴 낱말 둘레와 원문 공백 자리에서만
    띄운다. 구두점은 앞 조각에 그대로 붙인다.
    """
    out: list[str] = []
    pending_space = False
    for piece in pieces:
        if piece.kind == "space":
            pending_space = bool(out)
            continue
        value = getattr(piece, key)
        if not value:
            continue
        # 구두점(other)은 언제나 앞 조각에 붙인다 — 라틴 낱말 뒤라고 띄우면 "배비 ~"가 된다.
        if out and piece.kind != "other" and (
            pending_space or piece.kind == "latin" or space_between_syllables
        ):
            out.append(" ")
        out.append(value)
        # 라틴 낱말 뒤에는 다음 조각과 반드시 띄어야 한다(붙이면 한 낱말로 보인다).
        pending_space = piece.kind == "latin"
    return "".join(out).strip()


def zh_to_pinyin(text: str, *, tone: str = "mark") -> str:
    """중국어 한 줄 → 병음 표시 문자열(음절 사이 공백). ``romaji`` 표기 키의 값."""
    return _join(read_line(text, tone=tone), "pinyin", space_between_syllables=True)


def zh_to_hangul(text: str) -> str:
    """중국어 한 줄 → 한글 음차 표시 문자열. ``hangul`` 표기 키의 값."""
    return _join(read_line(text), "hangul", space_between_syllables=False)


def zh_to_kana(text: str) -> str:
    """중국어 한 줄 → 가타카나 근사 표시 문자열. ``kana`` 표기 키의 값."""
    return _join(read_line(text), "kana", space_between_syllables=False)


def zh_pron_variants(text: str, *, tone: str = "mark") -> dict[str, str]:
    """중국어 한 줄 → ``{"hangul", "kana", "romaji"}`` 표시 문자열 — 세그 ``pron``에 그대로 실을 값.

    한 번의 ``read_line``으로 셋을 다 만든다(세 번 부르면 pypinyin 조회가 세 배가 된다).
    값이 빈 표기는 넣지 않는다 — 클라이언트는 키가 없으면 그 표기 줄을 생략한다
    (everyric2-chrome ``lang.ts``의 ``resolvedPronunciation``).
    """
    pieces = read_line(text, tone=tone)
    variants = {
        "hangul": _join(pieces, "hangul", space_between_syllables=False),
        "kana": _join(pieces, "kana", space_between_syllables=False),
        "romaji": _join(pieces, "pinyin", space_between_syllables=True),
    }
    return {key: value for key, value in variants.items() if value}


def derive_zh_display_units(text: str, *, tone: str = "mark") -> LineUnits:
    """중국어 한 줄 → (원문 그대로인 정렬 타깃, {hangul, kana, romaji} 표시 소유자).

    ``target``이 원문과 같은 이유는 모듈 docstring 참조(zh CTC 모델의 vocab이 한자다).
    한자는 한 글자가 한 음절이라 소유자가 1:1로 떨어지고, 라틴 낱말만 철자 길이에 비례해
    나눈다(``align_target._distribute_by_length`` — 통째로 첫 글자에 몰면 그 낱말 구간에서
    카라오케가 멈춘다는 실측이 붙어 있는 로직이라 복사하지 않고 그대로 쓴다).

    ``word_end``는 라틴 낱말의 끝과 원문 공백 앞에 선다 — ``align_target.join_display``로
    이었을 때 낱말이 붙어 버리지 않게 하는 표시다. 다만 **병음 표기를 조립할 때는 그
    함수를 쓰면 안 된다**(음절 사이가 붙는다) — 표시 문자열은 ``zh_pron_variants``가 낸다.
    """
    pieces = read_line(text, tone=tone)
    owners: dict[str, list[str]] = {key: [] for key in ZH_DISPLAY_KEYS}
    word_end: list[bool] = []
    for piece in pieces:
        span = len(piece.source)
        if piece.kind == "latin" and span > 1:
            owners["hangul"].extend(_distribute_by_length(piece.source, piece.hangul))
            owners["kana"].extend(_distribute_by_length(piece.source, piece.kana))
            owners["romaji"].extend(_distribute_by_length(piece.source, piece.pinyin))
        else:
            owners["hangul"].extend([piece.hangul] + [""] * (span - 1))
            owners["kana"].extend([piece.kana] + [""] * (span - 1))
            owners["romaji"].extend([piece.pinyin] + [""] * (span - 1))
        word_end.extend([False] * span)
        if piece.kind == "latin":
            word_end[-1] = True
        elif piece.kind == "space" and len(word_end) > span:
            word_end[-span - 1] = True
    return LineUnits(text, owners, word_end)
