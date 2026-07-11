"""일본어 가사 라인의 발음(모라) 분해 + 한국어 발음 표기 DP 정렬.

오디오/모델 코드와 무관한 순수 텍스트 모듈이다. 흐름은 다음과 같다.

1. ``text_to_moras``: 원문 라인을 pykakasi로 읽어 모라(가나 1음) 시퀀스로 분해한다.
   각 모라는 원문의 어느 글자 구간에서 왔는지(char_start/char_end)를 들고 있다.
2. ``align_pron_to_moras``: 위키 등에 사람이 적어 둔 한국어 발음 표기를 모라 시퀀스에
   DP(가중 편집거리)로 정렬해 음절별 모라 구간을 찾는다.
3. ``pron_segments_for_line``: 위 둘을 CTC 글자 타이밍과 합쳐 발음 음절별 타임스탬프를
   만든다. 호출부(오디오 파이프라인)는 이 함수만 쓰면 된다.

한국어 발음 표기는 가창 관습을 따르므로 사전 읽기와 어긋나는 경우(예: 조사 は를
"와"로 적는 등)가 흔하다. DP 비용 함수는 그런 불일치도 허용하되 비용을 높게 매겨
품질 점수에 반영한다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import pykakasi

# ---------------------------------------------------------------------------
# 데이터 모델
# ---------------------------------------------------------------------------


@dataclass
class Mora:
    """원문 텍스트에서 파생된 모라(발음의 최소 단위) 1개."""

    kana: str
    char_start: int
    char_end: int
    is_ascii: bool = False


@dataclass
class PronSyllable:
    """위키 발음 표기의 한글 음절 1개와, 그것이 매칭된 모라 구간."""

    text: str
    mora_start: int
    mora_end: int
    resolved: bool


# ---------------------------------------------------------------------------
# 모라 분해 (text_to_moras)
# ---------------------------------------------------------------------------

# 직전 가나와 결합해 1모라를 이루는 소문자(요음/외래어 확장 모음)
_SMALL_COMBINING = set("ゃゅょぁぃぅぇぉゎ")

# 연속 ASCII(영단어/숫자, 아포스트로피·하이픈 포함)를 1 유닛으로 묶는 패턴
_ASCII_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*")

_kakasi_instance: pykakasi.kakasi | None = None


def _get_kakasi() -> pykakasi.kakasi:
    global _kakasi_instance
    if _kakasi_instance is None:
        _kakasi_instance = pykakasi.kakasi()
    return _kakasi_instance


def _is_japanese_char(ch: str) -> bool:
    """히라가나/가타카나/한자 범위인지 (일본어 유닛 판별용)."""
    cp = ord(ch)
    return 0x3040 <= cp <= 0x30FF or 0x3400 <= cp <= 0x9FFF


def _hira_to_kana_moras(hira: str) -> list[str]:
    """pykakasi 'hira' 문자열을 모라(가나) 리스트로 분해.

    요음(ゃゅょ 등 소문자)은 직전 가나와 결합해 1모라. 촉음(っ)·발음(ん)·장음(ー)은
    각각 독립된 1모라(고전 일본어 박자 규칙과 동일).
    """
    moras: list[str] = []
    for ch in hira:
        if ch in _SMALL_COMBINING and moras:
            moras[-1] += ch
        else:
            moras.append(ch)
    return moras


def text_to_moras(text: str) -> list[Mora]:
    """원문 라인을 모라 시퀀스로 분해.

    pykakasi로 토큰화 + 읽기(가나)를 얻은 뒤, 일본어 토큰은 히라가나 읽기를 모라로
    쪼개고, 비일본어(ASCII 단어/숫자) 토큰은 공백/비ASCII 경계로 묶어 1 Mora로 만든다.
    공백·문장부호는 모라를 만들지 않고 글자 인덱스만 건너뛴다.

    pykakasi의 토큰(orig)은 원문을 그대로 이어 붙이면 복원되므로, 누적 오프셋으로
    char_start/char_end를 계산한다. 한 토큰에서 나온 여러 모라는 그 토큰의 글자
    구간을 공유한다(한자 낱글자 단위로는 세분화할 수 없기 때문).
    """
    items = _get_kakasi().convert(text)
    moras: list[Mora] = []
    pos = 0
    for item in items:
        chunk = item["orig"]
        chunk_start = pos
        chunk_end = pos + len(chunk)
        pos = chunk_end

        if any(_is_japanese_char(c) for c in chunk):
            for kana in _hira_to_kana_moras(item["hira"]):
                moras.append(Mora(kana=kana, char_start=chunk_start, char_end=chunk_end))
        else:
            for match in _ASCII_WORD_RE.finditer(chunk):
                moras.append(
                    Mora(
                        kana=match.group(),
                        char_start=chunk_start + match.start(),
                        char_end=chunk_start + match.end(),
                        is_ascii=True,
                    )
                )
    return moras


# ---------------------------------------------------------------------------
# 가나 -> 기대 한글 초성/모음 표 (DP 비용 함수용)
# ---------------------------------------------------------------------------

# (행, 모음) — 행은 ROW_ONSETS의 키와 대응
_KANA_TABLE: dict[str, tuple[str, str]] = {
    "あ": ("あ", "a"), "い": ("あ", "i"), "う": ("あ", "u"), "え": ("あ", "e"), "お": ("あ", "o"),
    "か": ("か", "a"), "き": ("か", "i"), "く": ("か", "u"), "け": ("か", "e"), "こ": ("か", "o"),
    "が": ("が", "a"), "ぎ": ("が", "i"), "ぐ": ("が", "u"), "げ": ("が", "e"), "ご": ("が", "o"),
    "さ": ("さ", "a"), "し": ("さ", "i"), "す": ("さ", "u"), "せ": ("さ", "e"), "そ": ("さ", "o"),
    "ざ": ("ざ", "a"), "じ": ("ざ", "i"), "ず": ("ざ", "u"), "ぜ": ("ざ", "e"), "ぞ": ("ざ", "o"),
    "た": ("た", "a"), "ち": ("た", "i"), "つ": ("た", "u"), "て": ("た", "e"), "と": ("た", "o"),
    "だ": ("だ", "a"), "ぢ": ("だ", "i"), "づ": ("だ", "u"), "で": ("だ", "e"), "ど": ("だ", "o"),
    "な": ("な", "a"), "に": ("な", "i"), "ぬ": ("な", "u"), "ね": ("な", "e"), "の": ("な", "o"),
    "は": ("は", "a"), "ひ": ("は", "i"), "ふ": ("は", "u"), "へ": ("は", "e"), "ほ": ("は", "o"),
    "ば": ("ば", "a"), "び": ("ば", "i"), "ぶ": ("ば", "u"), "べ": ("ば", "e"), "ぼ": ("ば", "o"),
    "ぱ": ("ぱ", "a"), "ぴ": ("ぱ", "i"), "ぷ": ("ぱ", "u"), "ぺ": ("ぱ", "e"), "ぽ": ("ぱ", "o"),
    "ま": ("ま", "a"), "み": ("ま", "i"), "む": ("ま", "u"), "め": ("ま", "e"), "も": ("ま", "o"),
    "や": ("や", "a"), "ゆ": ("や", "u"), "よ": ("や", "o"),
    "ら": ("ら", "a"), "り": ("ら", "i"), "る": ("ら", "u"), "れ": ("ら", "e"), "ろ": ("ら", "o"),
    "わ": ("わ", "a"), "ゐ": ("わ", "i"), "ゑ": ("わ", "e"), "を": ("わ", "o"),
}

# 요음/외래어 확장 소문자가 부여하는 모음
_SMALL_VOWEL: dict[str, str] = {
    "ゃ": "a", "ゅ": "u", "ょ": "o", "ぁ": "a", "ぃ": "i", "ぅ": "u", "ぇ": "e", "ぉ": "o", "ゎ": "a",
}

# 행 -> 허용 가능한 한글 초성 집합 (고쥬온 + 탁음/반탁음)
_ROW_ONSETS: dict[str, set[str]] = {
    "あ": {"ㅇ"},
    "か": {"ㄱ", "ㅋ", "ㄲ"},
    "が": {"ㄱ", "ㄲ"},
    "さ": {"ㅅ", "ㅆ"},
    "ざ": {"ㅈ", "ㅉ"},
    "た": {"ㄷ", "ㅌ", "ㄸ", "ㅊ", "ㅉ"},
    "だ": {"ㄷ", "ㄸ"},
    "な": {"ㄴ"},
    "は": {"ㅎ"},
    "ば": {"ㅂ", "ㅃ"},
    "ぱ": {"ㅍ", "ㅃ"},
    "ま": {"ㅁ"},
    "や": {"ㅇ"},
    "ら": {"ㄹ"},
    "わ": {"ㅇ"},
}

# 한글 중성(모음) -> 일본어 5모음 그룹으로 단순화
_VOWEL_GROUP: dict[str, str] = {
    "ㅏ": "a", "ㅑ": "a", "ㅘ": "a",
    "ㅐ": "e", "ㅒ": "e", "ㅔ": "e", "ㅖ": "e", "ㅙ": "e", "ㅞ": "e",
    "ㅓ": "eo", "ㅕ": "eo", "ㅝ": "eo",
    "ㅗ": "o", "ㅛ": "o",
    "ㅚ": "e",
    "ㅜ": "u", "ㅠ": "u",
    "ㅟ": "i",
    "ㅡ": "eu",
    "ㅢ": "i",
    "ㅣ": "i",
}

_HANGUL_BASE = 0xAC00
_HANGUL_END = 0xD7A3
_CHO = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
_JUNG = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
_JONG = [""] + list("ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")

_LOW_COST = 0.05
_PARTIAL_COST = 0.35
_MID_COST = 0.5
_GAP_COST = 0.55
_INSERT_COST = 0.6
_HIGH_COST = 0.9
_ASCII_COST = 0.1


def _decompose_hangul(ch: str) -> tuple[str, str, str] | None:
    if len(ch) != 1 or not (_HANGUL_BASE <= ord(ch) <= _HANGUL_END):
        return None
    code = ord(ch) - _HANGUL_BASE
    cho = code // (21 * 28)
    jung = (code % (21 * 28)) // 28
    jong = code % 28
    return _CHO[cho], _JUNG[jung], _JONG[jong]


def _kana_row_vowel(kana: str) -> tuple[str, str] | None:
    """가나(1~2자)에서 (행, 모음)을 얻는다. 요음/외래어 확장 소문자는 직전 가나의
    행 + 소문자의 모음을 조합한다."""
    if len(kana) == 1:
        return _KANA_TABLE.get(kana)
    if len(kana) == 2 and kana[1] in _SMALL_VOWEL:
        base = _KANA_TABLE.get(kana[0])
        if base is None:
            return None
        return base[0], _SMALL_VOWEL[kana[1]]
    return None


def _match_cost(moras: list[Mora], i: int, syllable: str) -> float:
    """모라 moras[i]와 한글 음절 syllable의 매칭 비용 (0=완전 일치, 1에 가까울수록 불일치)."""
    mora = moras[i]
    if mora.is_ascii:
        return _ASCII_COST

    decomposed = _decompose_hangul(syllable)
    if decomposed is None:
        return _HIGH_COST
    cho, jung, _jong = decomposed
    vowel_actual = _VOWEL_GROUP.get(jung)

    if mora.kana == "ん":
        return _LOW_COST if cho in {"ㅇ", "ㄴ", "ㅁ"} else _HIGH_COST

    if mora.kana == "っ":
        # 촉음은 보통 직전 음절 받침으로 흡수되므로 직접 매칭은 드물다.
        return _MID_COST

    if mora.kana == "ー":
        expected_vowel = None
        if i > 0 and not moras[i - 1].is_ascii and moras[i - 1].kana not in ("ん", "っ", "ー"):
            prev_info = _kana_row_vowel(moras[i - 1].kana)
            if prev_info:
                expected_vowel = prev_info[1]
        if expected_vowel is not None and vowel_actual == expected_vowel:
            return _LOW_COST
        if cho == "ㅇ":
            return _PARTIAL_COST
        return _HIGH_COST

    info = _kana_row_vowel(mora.kana)
    if info is None:
        return _MID_COST
    row, vowel_expected = info
    onset_ok = cho in _ROW_ONSETS.get(row, set())
    vowel_ok = vowel_actual == vowel_expected
    if onset_ok and vowel_ok:
        return 0.0
    if onset_ok or vowel_ok:
        return _PARTIAL_COST
    return _HIGH_COST


def _mora_skip_cost(mora: Mora) -> float:
    if mora.kana in ("っ", "ん"):
        return _LOW_COST
    return _GAP_COST


def _syll_extra(moras: list[Mora], i: int) -> tuple[float, bool]:
    """대기 중인(아직 소비되지 않은) 모라 i에 음절 하나를 추가 배정.

    반환: (비용, 실제로 모라 i에 배정됐는지). ASCII 유닛에만 허용되며, 그 외에는
    매칭될 모라가 없는 순수 삽입(미해결)으로 처리한다."""
    if i < len(moras) and moras[i].is_ascii:
        return _ASCII_COST, True
    return _INSERT_COST, False


# ---------------------------------------------------------------------------
# DP 정렬 (align_pron_to_moras)
# ---------------------------------------------------------------------------


def align_pron_to_moras(moras: list[Mora], pron: str) -> tuple[list[PronSyllable], float]:
    """위키 발음 문자열(공백 포함 한글)을 모라 시퀀스에 DP 정렬.

    허용 연산: 모라:음절 1:1 매칭(대체 비용 포함), 모라 단독 소실(촉음/발음은 저비용
    = 직전 음절 받침 흡수, 그 외는 중간비용), ASCII 유닛에 여러 음절을 몰아 배정
    (저비용), 매칭될 모라가 없는 음절 순수 삽입(미해결, 중간비용).

    반환: (음절 리스트, 품질 점수 0~1). 품질 = 매칭된 음절 비율 x 평균 유사도.
    """
    syllables = [c for c in pron if not c.isspace()]
    m, s = len(moras), len(syllables)
    if m == 0 or s == 0:
        return [], 0.0

    cost = [[0.0] * (s + 1) for _ in range(m + 1)]
    back: list[list[tuple]] = [[() for _ in range(s + 1)] for _ in range(m + 1)]

    for i in range(1, m + 1):
        cost[i][0] = cost[i - 1][0] + _mora_skip_cost(moras[i - 1])
        back[i][0] = ("skip",)
    for j in range(1, s + 1):
        extra_cost, claims = _syll_extra(moras, 0)
        cost[0][j] = cost[0][j - 1] + extra_cost
        back[0][j] = ("extra", 0, claims)

    for i in range(1, m + 1):
        for j in range(1, s + 1):
            best_cost = cost[i - 1][j - 1] + _match_cost(moras, i - 1, syllables[j - 1])
            best_back: tuple = ("match",)

            skip_cost = cost[i - 1][j] + _mora_skip_cost(moras[i - 1])
            if skip_cost < best_cost:
                best_cost, best_back = skip_cost, ("skip",)

            extra_cost, claims = _syll_extra(moras, i)
            ex_cost = cost[i][j - 1] + extra_cost
            if ex_cost < best_cost:
                best_cost, best_back = ex_cost, ("extra", i, claims)

            cost[i][j] = best_cost
            back[i][j] = best_back

    # 역추적 (역순으로 모은 뒤 뒤집어서 시간순으로 정리)
    events: list[tuple] = []
    i, j = m, s
    while i > 0 or j > 0:
        op = back[i][j]
        if op[0] == "match":
            events.append(("match", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif op[0] == "skip":
            events.append(("skip", i - 1))
            i -= 1
        else:
            _, mora_idx, claims = op
            events.append(("extra", mora_idx, j - 1, claims))
            j -= 1
    events.reverse()

    result: list[PronSyllable] = []
    costs: list[float] = []
    last_end: int | None = None
    for ev in events:
        if ev[0] == "match":
            _, mora_idx, syl_idx = ev
            c = _match_cost(moras, mora_idx, syllables[syl_idx])
            result.append(
                PronSyllable(
                    text=syllables[syl_idx], mora_start=mora_idx, mora_end=mora_idx + 1, resolved=True
                )
            )
            costs.append(c)
            last_end = mora_idx + 1
        elif ev[0] == "extra":
            _, mora_idx, syl_idx, claims = ev
            if claims:
                result.append(
                    PronSyllable(
                        text=syllables[syl_idx],
                        mora_start=mora_idx,
                        mora_end=mora_idx + 1,
                        resolved=True,
                    )
                )
                costs.append(_ASCII_COST)
                last_end = mora_idx + 1
            else:
                result.append(
                    PronSyllable(text=syllables[syl_idx], mora_start=-1, mora_end=-1, resolved=False)
                )
                costs.append(_INSERT_COST)
        else:  # skip
            _, mora_idx = ev
            kana = moras[mora_idx].kana
            if (
                kana in ("っ", "ん")
                and result
                and result[-1].resolved
                and last_end == mora_idx
            ):
                prev = result[-1]
                result[-1] = PronSyllable(
                    text=prev.text, mora_start=prev.mora_start, mora_end=mora_idx + 1, resolved=True
                )
                last_end = mora_idx + 1
            # else: 매칭되는 음절이 없는 모라는 조용히 버려진다 (커버리지에 영향 없음)

    resolved_costs = [c for syl, c in zip(result, costs) if syl.resolved]
    if not result:
        quality = 0.0
    else:
        matched_ratio = len(resolved_costs) / len(result)
        avg_similarity = (
            sum(1.0 - c for c in resolved_costs) / len(resolved_costs) if resolved_costs else 0.0
        )
        quality = matched_ratio * avg_similarity

    return result, quality


# ---------------------------------------------------------------------------
# CTC 글자 타이밍과의 통합 (pron_segments_for_line)
# ---------------------------------------------------------------------------

_QUALITY_THRESHOLD = 0.6


def _build_char_time(
    text: str, char_spans: list[tuple[str, float, float]]
) -> list[tuple[float, float] | None]:
    """원문 글자별 (start, end) 시간. char_spans는 원문 순서의 부분열이며 일부 글자가
    누락될 수 있다. 누락된 글자는 이웃 글자 시간으로 선형 보간한다.
    스팬은 (글자, start, end) 또는 뒤에 confidence가 붙은 (글자, start, end, conf)도 허용한다."""
    times: list[tuple[float, float] | None] = [None] * len(text)
    p = 0
    for span in char_spans:
        ch, start, end = span[0], span[1], span[2]
        q = p
        while q < len(text) and text[q] != ch:
            q += 1
        if q >= len(text):
            continue
        times[q] = (start, end)
        p = q + 1

    known = [idx for idx, v in enumerate(times) if v is not None]
    if not known:
        return times

    for idx in range(known[0]):
        times[idx] = times[known[0]]
    for idx in range(known[-1] + 1, len(text)):
        times[idx] = times[known[-1]]

    for a, b in zip(known, known[1:]):
        gap = b - a
        if gap <= 1:
            continue
        gap_start = times[a][1]
        gap_end = times[b][0]
        step = (gap_end - gap_start) / gap
        for k in range(1, gap):
            times[a + k] = (gap_start + step * (k - 1), gap_start + step * k)

    return times


def _build_mora_time(
    moras: list[Mora], char_time: list[tuple[float, float] | None]
) -> list[tuple[float, float]]:
    """모라별 (start, end). 같은 글자 구간을 공유하는 모라들은 그 구간의 시간을
    모라 개수만큼 균등 분할한다."""
    result: list[tuple[float, float]] = [(0.0, 0.0)] * len(moras)
    i, n = 0, len(moras)
    while i < n:
        j = i
        span = (moras[i].char_start, moras[i].char_end)
        while j + 1 < n and (moras[j + 1].char_start, moras[j + 1].char_end) == span:
            j += 1

        cs, ce = span
        sub_times = [t for t in char_time[cs:ce] if t is not None]
        if sub_times:
            token_start, token_end = sub_times[0][0], sub_times[-1][1]
        else:
            token_start = token_end = 0.0

        count = j - i + 1
        total = max(token_end - token_start, 0.0)
        for k in range(count):
            result[i + k] = (
                token_start + total * k / count,
                token_start + total * (k + 1) / count,
            )
        i = j + 1
    return result


def pron_segments_for_line(
    char_spans: list[tuple[str, float, float]],
    text: str,
    pron: str,
) -> list[dict] | None:
    """CTC 글자 타이밍 + 위키 발음을 합쳐 발음 음절별 타임스탬프를 만든다.

    글자 타이밍 -> 모라 수로 비례 분할해 모라별 시간을 만들고, DP 정렬 결과로
    음절별 시간을 배정한다. 인접 음절 시간은 단조 증가하도록 클램프한다.
    품질 점수가 낮으면(정렬을 신뢰할 수 없으면) None을 반환해 호출부가 그라데이션
    폴백을 쓰도록 한다.
    """
    if not char_spans or not text.strip():
        return None

    moras = text_to_moras(text)
    if not moras:
        return None

    pron_syllables, quality = align_pron_to_moras(moras, pron)
    if quality < _QUALITY_THRESHOLD or not pron_syllables:
        return None

    char_time = _build_char_time(text, char_spans)
    mora_time = _build_mora_time(moras, char_time)

    segments: list[dict] = []
    i, n = 0, len(pron_syllables)
    while i < n:
        syl = pron_syllables[i]
        if not syl.resolved or syl.mora_start < 0:
            i += 1
            continue

        j = i
        while (
            j + 1 < n
            and pron_syllables[j + 1].resolved
            and pron_syllables[j + 1].mora_start == syl.mora_start
            and pron_syllables[j + 1].mora_end == syl.mora_end
        ):
            j += 1
        group = pron_syllables[i : j + 1]

        start, _ = mora_time[syl.mora_start]
        if syl.mora_end - syl.mora_start > 1:
            _, end = mora_time[syl.mora_end - 1]
        else:
            _, end = mora_time[syl.mora_start]
        total = max(end - start, 0.0)

        count = len(group)
        for idx, g in enumerate(group):
            segments.append(
                {
                    "text": g.text,
                    "start": start + total * idx / count,
                    "end": start + total * (idx + 1) / count,
                    "resolved": True,
                }
            )
        i = j + 1

    for idx in range(1, len(segments)):
        if segments[idx]["start"] < segments[idx - 1]["end"]:
            segments[idx]["start"] = segments[idx - 1]["end"]
        if segments[idx]["end"] < segments[idx]["start"]:
            segments[idx]["end"] = segments[idx]["start"]

    return segments if segments else None


# ---------------------------------------------------------------------------
# 독음(ko) 정렬 결과의 역매핑 (map_pron_alignment_to_line)
# ---------------------------------------------------------------------------


def _clamp_monotonic(spans: list[dict]) -> None:
    """인접 span 시간을 단조 증가하도록 제자리 클램프."""
    for idx in range(1, len(spans)):
        if spans[idx]["start"] < spans[idx - 1]["end"]:
            spans[idx]["start"] = spans[idx - 1]["end"]
        if spans[idx]["end"] < spans[idx]["start"]:
            spans[idx]["end"] = spans[idx]["start"]


def _geomean(values: list[float]) -> float | None:
    """양수 값들의 기하평균 (0/None 제외). 없으면 None. CTC conf는 확률/지수라 기하평균이 자연스럽다."""
    import math

    xs = [v for v in values if v is not None and v > 0]
    if not xs:
        return None
    return math.exp(sum(math.log(v) for v in xs) / len(xs))


def _syllable_confidences(
    syllables: list[str], pron_char_spans: list
) -> list[float | None]:
    """ko CTC 정렬 스팬의 음절별 confidence를 ``syllables`` 위치에 매핑.

    ``pron_char_spans``는 (음절, start, end[, confidence])의 부분열이며 일부 음절(OOV)이
    누락될 수 있다 — ``_build_char_time``의 글자 매칭과 동일 로직으로 순서대로 맞춘다.
    conf가 없는(3-튜플) 스팬이나 매칭 안 된 음절은 None(호출부가 라인 기하평균으로 폴백)."""
    confs: list[float | None] = [None] * len(syllables)
    p = 0
    for span in pron_char_spans:
        ch = span[0]
        conf = span[3] if len(span) > 3 else None
        q = p
        while q < len(syllables) and syllables[q] != ch:
            q += 1
        if q >= len(syllables):
            continue
        confs[q] = conf
        p = q + 1
    return confs


def map_pron_alignment_to_line(
    text: str,
    pron: str,
    pron_char_spans: list[tuple[str, float, float]],
) -> tuple[list[dict] | None, list[dict] | None]:
    """독음(한국어 발음)으로 정렬한 CTC 결과를 원문 라인에 역매핑한다.

    ``pron_segments_for_line``의 역방향이다. 저기서는 원문 글자 타이밍 → 모라 →
    발음 음절 순으로 시간을 만들었다면, 여기서는 발음 음절이 이미 오디오에 정렬돼
    있고(ko CTC), 그 음절 시간을 모라를 거쳐 원문 글자에 되돌린다.

    체인: 발음 음절 k ─(DP 정렬)→ 모라 구간 ─(pykakasi char_start/end)→ 원문 글자.

    Args:
        text: 원문 라인 (표시용, 한자 포함).
        pron: 위키 발음 표기(공백 포함 한글).
        pron_char_spans: ko CTC 정렬의 한글 음절별 (음절, start, end). 공백은 없고
            일부 음절(OOV)이 누락될 수 있다 — 누락 음절은 이웃으로 보간한다.

    Returns:
        (word_segments, pron_segments)
        - pron_segments: [{text, start, end, resolved}] 발음 음절별 스팬 — 노트 앵커·
          발음 표시용. 음절 정렬이 하나라도 있으면 항상 만든다.
        - word_segments: [{word, start, end}] 원문 글자별 스팬 — 모라 역매핑이
          성공하고 품질이 임계값 이상일 때만. 매핑 불가 라인은 None(라인 타이밍만).
        음절 정렬 자체가 비었으면 (None, None).
    """
    syllables = [c for c in pron if not c.isspace()]
    if not syllables or not pron_char_spans:
        return None, None

    # 누락 음절을 이웃으로 보간해 음절별 (start, end)를 만든다 (원문 글자 보간과 동일 로직)
    syl_time = _build_char_time("".join(syllables), pron_char_spans)
    if all(t is None for t in syl_time):
        return None, None
    # 음절별 confidence(ko CTC) — pron_segments 표시·글자 conf 역매핑용
    syl_conf = _syllable_confidences(syllables, pron_char_spans)

    pron_segments: list[dict] = []
    for k, syl in enumerate(syllables):
        t = syl_time[k]
        if t is None:
            continue
        seg = {"text": syl, "start": t[0], "end": t[1], "resolved": True}
        if syl_conf[k] is not None:
            seg["confidence"] = round(syl_conf[k], 6)
        pron_segments.append(seg)
    _clamp_monotonic(pron_segments)

    words = _map_syllable_times_to_chars(text, pron, syllables, syl_time, syl_conf)
    return words, (pron_segments or None)


def _map_syllable_times_to_chars(
    text: str,
    pron: str,
    syllables: list[str],
    syl_time: list[tuple[float, float] | None],
    syl_conf: list[float | None] | None = None,
) -> list[dict] | None:
    """발음 음절 시간을 모라 역매핑으로 원문 글자에 분배. 실패 시 None.

    ``syl_conf``를 주면 각 글자에 매핑된 음절(들)의 conf 기하평균을 글자 conf로 실어
    반환한다 (한 라인 안에서 글자별 conf가 달라지도록). 매핑 불가/보간 글자는 conf 없음
    → 호출부(worker)가 라인 기하평균으로 폴백한다."""
    try:
        moras = text_to_moras(text)
        if not moras:
            return None
        result_syls, quality = align_pron_to_moras(moras, pron)
        # DP 결과는 음절 1개당 1 엔트리 → syl_time과 인덱스가 1:1이어야 한다
        if quality < _QUALITY_THRESHOLD or len(result_syls) != len(syllables):
            return None

        char_s: list[float | None] = [None] * len(text)
        char_e: list[float | None] = [None] * len(text)
        # 글자별로 그 글자를 덮은 음절들의 conf를 모아 둔다 (뒤에서 기하평균)
        char_confs: list[list[float]] = [[] for _ in range(len(text))]
        for k, ps in enumerate(result_syls):
            if not ps.resolved or ps.mora_start < 0:
                continue
            t = syl_time[k]
            if t is None:
                continue
            c_conf = syl_conf[k] if syl_conf is not None and k < len(syl_conf) else None
            for mi in range(ps.mora_start, ps.mora_end):
                if not 0 <= mi < len(moras):
                    continue
                for c in range(moras[mi].char_start, moras[mi].char_end):
                    if not 0 <= c < len(text):
                        continue
                    if char_s[c] is None or t[0] < char_s[c]:
                        char_s[c] = t[0]
                    if char_e[c] is None or t[1] > char_e[c]:
                        char_e[c] = t[1]
                    if c_conf is not None:
                        char_confs[c].append(c_conf)

        known = [i for i in range(len(text)) if char_s[i] is not None]
        if not known:
            return None
        for i in range(known[0]):
            char_s[i], char_e[i] = char_s[known[0]], char_e[known[0]]
        for i in range(known[-1] + 1, len(text)):
            char_s[i], char_e[i] = char_s[known[-1]], char_e[known[-1]]
        for a, b in zip(known, known[1:]):
            if b - a <= 1:
                continue
            gs, ge = char_e[a], char_s[b]
            step = (ge - gs) / (b - a)
            for kk in range(1, b - a):
                char_s[a + kk] = gs + step * (kk - 1)
                char_e[a + kk] = gs + step * kk

        words: list[dict] = []
        for c, ch in enumerate(text):
            if ch.isspace() or char_s[c] is None:
                continue
            w = {"word": ch, "start": char_s[c], "end": char_e[c]}
            conf = _geomean(char_confs[c])
            if conf is not None:
                w["confidence"] = round(conf, 6)
            words.append(w)
        _clamp_monotonic(words)
        return words or None
    except Exception:
        return None
