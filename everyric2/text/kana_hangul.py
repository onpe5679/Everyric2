"""가나 → 한글 독음 결정적 변환 (K-보카로 팬 표기 관례).

LLM에게 한글 전사까지 맡기면 기계적 실수가 재발한다 (실측: ずっと→즈토 촉음 소실,
じぶんが→지부가 ん 소실, ずっと→쯔우토 억지 장음). 그래서 역할을 나눈다:
한자→가나 읽기(문맥 판단이 필요한 부분)는 LLM이, 가나→한글(순수 기계 변환)은 이
모듈이 한다. 표기 관례: か/た행은 위치 무관 격음(카·타·치·츠), 촉음(っ)=앞 음절
ㅅ받침(ずっと→즛토), ん=앞 음절 ㄴ받침(じぶんが→지분가), 장음(ー)=직전 모음 반복.
가나가 아닌 문자(공백·문장부호·한글 등)는 그대로 통과한다.
"""

import re

_KANA_RE = re.compile(r"[぀-ゟ゠-ヿ]")
_KANJI_RE = re.compile(r"[㐀-鿿]")

# 단일 가나 (히라가나 기준 — 가타카나는 변환 전 히라가나로 정규화)
_SINGLES = {
    "あ": "아", "い": "이", "う": "우", "え": "에", "お": "오",
    "か": "카", "き": "키", "く": "쿠", "け": "케", "こ": "코",
    "が": "가", "ぎ": "기", "ぐ": "구", "げ": "게", "ご": "고",
    "さ": "사", "し": "시", "す": "스", "せ": "세", "そ": "소",
    "ざ": "자", "じ": "지", "ず": "즈", "ぜ": "제", "ぞ": "조",
    "た": "타", "ち": "치", "つ": "츠", "て": "테", "と": "토",
    "だ": "다", "ぢ": "지", "づ": "즈", "で": "데", "ど": "도",
    "な": "나", "に": "니", "ぬ": "누", "ね": "네", "の": "노",
    "は": "하", "ひ": "히", "ふ": "후", "へ": "헤", "ほ": "호",
    "ば": "바", "び": "비", "ぶ": "부", "べ": "베", "ぼ": "보",
    "ぱ": "파", "ぴ": "피", "ぷ": "푸", "ぺ": "페", "ぽ": "포",
    "ま": "마", "み": "미", "む": "무", "め": "메", "も": "모",
    "や": "야", "ゆ": "유", "よ": "요",
    "ら": "라", "り": "리", "る": "루", "れ": "레", "ろ": "로",
    "わ": "와", "ゐ": "이", "ゑ": "에", "を": "오",
    "ゔ": "부", "ゎ": "와", "ゕ": "카", "ゖ": "케",
}

# 소형 모음 단독 출현 (외래어 표기 잔재) — 완모음으로 근사
_SMALL_VOWELS = {"ぁ": "아", "ぃ": "이", "ぅ": "우", "ぇ": "에", "ぉ": "오",
                 "ゃ": "야", "ゅ": "유", "ょ": "요"}

# 2글자 조합 (요음·외래어 조합) — 단일 매핑보다 먼저 검사
_DIGRAPHS = {
    "きゃ": "캬", "きゅ": "큐", "きょ": "쿄",
    "ぎゃ": "갸", "ぎゅ": "규", "ぎょ": "교",
    "しゃ": "샤", "しゅ": "슈", "しょ": "쇼",
    "じゃ": "자", "じゅ": "주", "じょ": "조",
    "ちゃ": "차", "ちゅ": "추", "ちょ": "초",
    "ぢゃ": "자", "ぢゅ": "주", "ぢょ": "조",
    "にゃ": "냐", "にゅ": "뉴", "にょ": "뇨",
    "ひゃ": "햐", "ひゅ": "휴", "ひょ": "효",
    "びゃ": "뱌", "びゅ": "뷰", "びょ": "뵤",
    "ぴゃ": "퍄", "ぴゅ": "퓨", "ぴょ": "표",
    "みゃ": "먀", "みゅ": "뮤", "みょ": "묘",
    "りゃ": "랴", "りゅ": "류", "りょ": "료",
    "ふぁ": "파", "ふぃ": "피", "ふぇ": "페", "ふぉ": "포", "ふゅ": "퓨",
    "うぃ": "위", "うぇ": "웨", "うぉ": "워",
    "ゔぁ": "바", "ゔぃ": "비", "ゔぇ": "베", "ゔぉ": "보",
    "てぃ": "티", "でぃ": "디", "とぅ": "투", "どぅ": "두",
    "しぇ": "셰", "じぇ": "제", "ちぇ": "체", "いぇ": "예",
}

_JONG_S = 19  # 종성 ㅅ (촉음)
_JONG_N = 4   # 종성 ㄴ (ん)

# 중성 인덱스 → 장음(ー)에서 반복할 모음 음절 (음가의 말단 모음 기준: ㅑ→아, ㅘ→아).
# ㅡ(18)→우: ㅡ 음절은 す/ず/つ 행뿐이고 일본어 음가는 u라 장음도 우가 맞다 (スー→스우)
_JUNG_LONG = {
    0: "아", 1: "애", 2: "아", 3: "애", 4: "어", 5: "에", 6: "어", 7: "에",
    8: "오", 9: "아", 10: "애", 11: "에", 12: "오", 13: "우", 14: "어",
    15: "에", 16: "이", 17: "우", 18: "우", 19: "이", 20: "이",
}


def has_kana(text: str) -> bool:
    return bool(_KANA_RE.search(text))


def has_kanji(text: str) -> bool:
    return bool(_KANJI_RE.search(text))


def _to_hiragana(text: str) -> str:
    """가타카나(ァ 0x30A1 ~ ヶ 0x30F6)를 히라가나로 정규화 — 장음 ー는 그대로 둔다."""
    return "".join(
        chr(ord(ch) - 0x60) if "ァ" <= ch <= "ヶ" else ch for ch in text
    )


def _attach_jong(out: list[str], jong: int) -> bool:
    """직전 출력 음절(받침 없는 한글)에 종성을 붙인다. 성공 여부 반환."""
    if not out:
        return False
    prev = out[-1]
    code = ord(prev[-1]) - 0xAC00
    if 0 <= code <= 11171 and code % 28 == 0:
        out[-1] = prev[:-1] + chr(ord(prev[-1]) + jong)
        return True
    return False


def _long_vowel(out: list[str]) -> str | None:
    """직전 출력 음절의 모음을 장음 반복용 음절로 돌려준다 (없으면 None)."""
    if not out:
        return None
    code = ord(out[-1][-1]) - 0xAC00
    if not (0 <= code <= 11171):
        return None
    return _JUNG_LONG.get((code // 28) % 21)


def kana_to_hangul(text: str) -> str:
    """가나 문자열을 한글 독음으로 변환한다. 가나 외 문자는 그대로 통과."""
    s = _to_hiragana(text)
    out: list[str] = []
    i = 0
    while i < len(s):
        pair = s[i : i + 2]
        if len(pair) == 2 and pair in _DIGRAPHS:
            out.append(_DIGRAPHS[pair])
            i += 2
            continue
        ch = s[i]
        if ch == "っ":
            _attach_jong(out, _JONG_S)  # 붙일 음절이 없으면(라인 첫머리) 무시
        elif ch == "ん":
            if not _attach_jong(out, _JONG_N):
                out.append("응")
        elif ch == "ー":
            rep = _long_vowel(out)
            if rep:
                out.append(rep)
        elif ch in _SINGLES:
            out.append(_SINGLES[ch])
        elif ch in _SMALL_VOWELS:
            out.append(_SMALL_VOWELS[ch])
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def kanji_to_kana(text: str) -> str:
    """잔존 한자를 pykakasi로 가나화한다 (LLM이 일부 한자를 남긴 경우의 폴백).

    LLM이 이미 가나로 쓴 부분은 pykakasi를 통과해도 그대로이므로, LLM의 문맥 읽기를
    보존하면서 빠뜨린 한자만 사전 읽기로 메꾼다. 실패 시 원문 그대로 반환.
    """
    try:
        import pykakasi

        reader = pykakasi.kakasi()
        return "".join(item.get("hira", "") for item in reader.convert(text))
    except Exception:
        return text


def finalize_pronunciation(pron: str | None) -> str | None:
    """LLM 발음 필드(가나 기대)를 최종 한글 독음으로 마감한다.

    - 한자 잔존 → 가나화(kanji_to_kana) 후 변환
    - 가나 포함 → kana_to_hangul (촉음/ん/장음 정확 처리)
    - 이미 한글뿐(가나·한자 없음) → 그대로 (구형 프롬프트 응답·비일본어 호환)
    """
    if not pron:
        return pron
    if has_kanji(pron):
        pron = kanji_to_kana(pron)
    if has_kana(pron):
        pron = kana_to_hangul(pron)
    return " ".join(pron.split()) or None
