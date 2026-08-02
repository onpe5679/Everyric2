"""2패스 정렬 타깃 파생 — 화면 표시(표기별 전부)와 CTC 정렬 타깃을 가른다.

``everyric2.alignment.refine_window``(2패스 음절 리파이너)가 라인 텍스트를 경량 CTC 모델에
먹이기 전에 거치는 변환들이다. **정렬은 한 번만 하고, 표시는 표기마다 전부 낸다** —
카라오케 노트에 무엇을 띄울지(한글/로마자/가나/원문)는 재생 시점 사용자 설정이 정하므로,
서버가 생성 시점에 하나만 골라 구우면 표기를 바꿀 때마다 재생성해야 한다. 추가 정렬 비용은
없다 — ``en_g2p.Unit``이 이미 (한글, IPA, 가나, 기여 음소)를 **동시에** 들고 다니므로,
한 번의 스팬 위에 표기별 문자열만 얹으면 된다.

* en: 정렬은 IPA(CMU 발음 사전 유래, ``everyric2.text.en_g2p``)로 하고 표시는 **네 표기**를
  함께 낸다 — ``hangul``·``kana``·``romaji``(가나를 로마자화)·``en``(원문 영어 음절분리,
  ``beau-ti-ful``). 철자 CTC는 음절 경계도 묵음도 못 가르지만(``make``의 e에도 스팬을
  준다), IPA는 소리에만 스팬을 준다(벤치 실측: 한 줄에 철자 세그 2,018개, 같은 구간 IPA
  음절 749개).
* ja: 정렬은 가나 독음으로 하고(``everyric2.text.ja_reading.tokenize_reading``), 표시는
  ``hangul``·``kana``·``romaji`` 세 표기를 낸다. 한자 한 글자를 표시 한 칸으로 켜면
  (見た → 見|た) 그 칸이 2~3모라를 통째로 덮어 타이밍이 뭉개지므로(벤치 실측: 深海少女
  세그 634 → 429, UST 음절 97.7% → 97.3%) 모라 단위로 낸다.

``LineUnits(target, owners, word_end)``가 산출물이다. ``owners[key][i]``는 ``target[i]``가
발성될 때 표기 ``key``로 화면에 켤 글자다. **빈 문자열은 "표시 없음"** — 직전에 표시를 낸
타깃 문자에 딸린 연속(이중모음의 둘째 음소, 요음의 작은 가나 등)이라는 뜻이며, 정제 단계가
그 구간을 앞 세그의 끝으로 흡수한다(``refine_window``의 세그 구성 로직).

**표기별 세그 수는 다를 수 있다** — 한글 글자 수와 가나 모라 수는 같지 않다. 인덱스는
``target``에 대해서만 맞춰져 있고 표기 사이의 대응은 보장하지 않는다(요구사항).

``word_end[i]``는 ``target[i]``가 **라틴 낱말의 마지막 타깃 문자**인지를 표시한다(en 표기
전용 — ja 표기는 원문 공백이 이미 소유자에 실려 있어 이 표시가 필요 없다). 라틴 발음
표기(한글·가나·로마자)는 원문과 달리 낱말 사이 공백이 CTC 어휘에 없어 정렬 후 세그에서
사라지므로, 세그 자체가 아니라 **이 플래그**로 낱말 경계를 옮긴다 — 표시(``pron`` 문자열
조립)에서만 쓰고 정렬에는 영향이 없다. ``everyric2.alignment.refine_window``가 세그를 지을
때 이 플래그가 걸린 타깃 위치를 포함하는 세그에 ``PronSegmentSpan.word_end``를 얹는다.

표시 파생이 **정렬을 건드리지 않는다**는 성질(``target``이 표시 선택과 무관하게 항상
같다)은 ``tests/test_align_target.py``가 못박는다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class LineUnits:
    """한 줄의 (CTC 정렬 타깃, 표기별 표시 소유권, 낱말 경계) 묶음. 모듈 docstring 참조."""

    target: str
    owners: dict[str, list[str]]
    # 라틴 낱말의 마지막 타깃 인덱스에서 True. 비워 두면(빈 리스트) target 길이만큼 False로
    # 채운다 — ja 전용 파생 함수처럼 낱말 경계가 없는 경로가 매번 채울 필요는 없게 한다.
    word_end: list[bool] = field(default_factory=list)

    def __post_init__(self) -> None:
        for key, arr in self.owners.items():
            if len(arr) != len(self.target):
                raise ValueError(
                    f"owners[{key!r}] 길이({len(arr)})가 target 길이({len(self.target)})와 다르다"
                )
        if not self.word_end:
            object.__setattr__(self, "word_end", [False] * len(self.target))
        elif len(self.word_end) != len(self.target):
            raise ValueError(
                f"word_end 길이({len(self.word_end)})가 target 길이({len(self.target)})와 다르다"
            )


EN_DISPLAY_KEYS: tuple[str, ...] = ("hangul", "kana", "romaji", "en")
JA_DISPLAY_KEYS: tuple[str, ...] = ("hangul", "kana", "romaji")


def native_units(source: str) -> LineUnits:
    """입력 표기를 그대로 쓰는 경로 — 정렬 텍스트와 표시가 같은 문자(ko 원문 등)."""
    return LineUnits(source, {"native": list(source)})


# ---------------------------------------------------------------------------
# en: IPA 정렬 + {hangul, kana, romaji, en} 동시 표시
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z']+")
# 앞 글자와 한 모라를 이루는 가나 — 요음·촉음·장음·소문자 모음. OOV 표시를 철자 길이에
# 비례 배분할 때 이 글자들이 앞 모라와 갈라지면 안 된다(``ウィ``가 ``ウ``+``ィ``로 쪼개진다).
_COMBINING_KANA = frozenset("ャュョァィゥェォッーゃゅょぁぃぅぇぉっ")


def _distribute_by_length(word: str, shown: str) -> list[str]:
    """``shown``(음차/음절 결과)을 ``word``(원문 철자) 길이에 비례해 슬롯으로 배분한다.

    통째로 첫 글자에 몰면 그 낱말이 세그 하나가 되어 그 구간에서만 카라오케가 멈춘다
    (weathergirl의 weathervane이 실제로 그랬다) — OOV 처리의 핵심 요구사항.
    """
    slots = [""] * len(word)
    if not shown:
        return slots
    cursor = -1
    for k, char in enumerate(shown):
        if char in _COMBINING_KANA and cursor >= 0:
            slot = cursor
        else:
            slot = max(min(k * len(word) // len(shown), len(word) - 1), cursor)
        slots[slot] += char
        cursor = slot
    return slots


def _slots_from_pieces(word: str, pieces: list[str]) -> list[str]:
    """음절 조각 목록을 낱말 글자 슬롯에 배치 — 조각의 첫 글자 위치에 조각 전체를 싣는다."""
    slots = [""] * len(word)
    cursor = 0
    for piece in pieces:
        if cursor < len(slots):
            slots[cursor] = piece
        cursor += len(piece)
    return slots


def derive_en_display_units(source: str) -> LineUnits:
    """영어 한 줄 → (IPA 정렬 타깃, {hangul, kana, romaji, en} 표시).

    낱말마다 ``en_g2p.units_for_word``로 IPA 유닛 열을 얻는다. 유닛 하나가 타깃 문자
    ``len(unit.ipa)``개를 차지하고, 그 유닛의 **첫 타깃 문자**에만 표시를 싣는다(나머지는
    빈 소유자 — 이중모음 둘째 음소 등). ``en`` 표시는 낱말을 음절 조각으로 갈라
    (``syllable_units_for_word``) 그 음절의 **첫 유닛**에만 싣는다 — 한 음절이 유닛
    여럿을 덮을 수 있어(예: 이중모음) hangul/kana보다 성긴 소유자 배열이 된다. 세 표기가
    같은 인덱스를 공유하지만(같은 ``target``) 실제 비어있지 않은 자리는 서로 다르다 —
    그래서 표기별 세그 개수가 다르다(요구사항).

    ``romaji``는 별도 유닛 데이터가 없다 — ``kana`` 소유자를 그대로 로마자화한다
    (``kana_romaji.kana_to_romaji``는 유닛 하나의 가나 조각이 자기 안에서 문맥을 완결하므로
    — 요음·장음이 있어도 그 유닛 문자열 하나로 충분하다 — 격리 변환이 안전하다).

    OOV(사전에 없는 낱말)는 타깃에 원문 철자를 그대로 두고, 표기별 표시를 철자 길이에
    비례 배분한다(``_distribute_by_length``/``_slots_from_pieces``) — 통째로 몰지 않는다.
    """
    from everyric2.text.en_g2p import syllabify_unknown, syllable_units_for_word, units_for_word
    from everyric2.text.kana_romaji import kana_to_romaji
    from everyric2.text.ko_reading import latin_to_kana
    from everyric2.text.latin_hangul import transliterate_latin

    chars: list[str] = []
    hangul_owners: list[str] = []
    kana_owners: list[str] = []
    en_owners: list[str] = []
    word_end_at: list[int] = []  # 낱말이 끝난 자리의 owners 인덱스 (아래 word_end로 환원)
    shown_at = -1  # 표시를 마지막으로 얹은 owners 위치

    def emit(target: str, hangul: str, kana: str, en_piece: str) -> None:
        nonlocal shown_at
        if not target:
            if shown_at >= 0:
                hangul_owners[shown_at] += hangul
                kana_owners[shown_at] += kana
                en_owners[shown_at] += en_piece
            return
        chars.append(target)
        shown_at = len(hangul_owners)
        hangul_owners.append(hangul)
        kana_owners.append(kana)
        en_owners.append(en_piece)
        pad = len(target) - 1
        if pad:
            hangul_owners.extend([""] * pad)
            kana_owners.extend([""] * pad)
            en_owners.extend([""] * pad)

    pos = 0
    for match in _WORD_RE.finditer(source):
        for char in source[pos : match.start()]:  # 낱말 사이 공백·구두점
            emit(char, char, char, char)
        word = match.group(0)
        syllables = syllable_units_for_word(word, 0)
        units = units_for_word(word, 0)
        if syllables:
            for piece, syl_units in syllables:
                pending_en = piece
                for unit in syl_units:
                    en_piece = pending_en if unit.ipa else ""
                    emit(unit.ipa, unit.hangul, unit.kana, en_piece)
                    if unit.ipa:
                        pending_en = ""
        elif units:
            # 사전엔 있는데 철자를 못 갈랐다(가사 출현 2.5%) — en 표시는 낱말 통째로
            # 첫 유닛에 싣는다.
            pending_en = word
            for unit in units:
                en_piece = pending_en if unit.ipa else ""
                emit(unit.ipa, unit.hangul, unit.kana, en_piece)
                if unit.ipa:
                    pending_en = ""
        else:
            # OOV — 타깃은 원문 철자, 표기별 표시는 철자 길이에 비례 배분한다.
            hangul_shown = transliterate_latin(word) or word
            kana_shown = latin_to_kana(word) or word
            h_slots = _distribute_by_length(word, hangul_shown)
            k_slots = _distribute_by_length(word, kana_shown)
            e_slots = _slots_from_pieces(word, syllabify_unknown(word))
            for index, char in enumerate(word):
                emit(char, h_slots[index], k_slots[index], e_slots[index])
        if hangul_owners:  # 낱말은 항상 최소 한 글자를 emit하므로 이 자리는 그 낱말의 끝이다
            word_end_at.append(len(hangul_owners) - 1)
        pos = match.end()
    for char in source[pos:]:
        emit(char, char, char, char)

    romaji_owners = [kana_to_romaji(k) if k else "" for k in kana_owners]
    # 문자 단위 인덱스로 채운다 — ``chars``는 유닛 단위(IPA가 여러 문자일 수 있다)라
    # 그 길이로 채우면 target(문자 단위)과 어긋난다. owners 배열은 이미 문자 단위로
    # 패딩돼 있으므로 그 길이를 쓴다.
    word_end = [False] * len(hangul_owners)
    for index in word_end_at:
        if 0 <= index < len(word_end):
            word_end[index] = True
    return LineUnits(
        "".join(chars),
        word_end=word_end,
        owners={
            "hangul": hangul_owners,
            "kana": kana_owners,
            "romaji": romaji_owners,
            "en": en_owners,
        },
    )


# ---------------------------------------------------------------------------
# ja: 가나 독음 정렬 + {hangul, kana, romaji} 동시 표시
# ---------------------------------------------------------------------------


def _prefix_growth_owners(source: str, convert) -> list[str]:
    """``source``를 한 글자씩 늘려가며 ``convert``의 출력이 언제 자라는지로 글자별 소유
    텍스트를 정한다.

    ``convert``가 전체 문자열을 보고 문맥을 반영하는 결정론 변환일 때만 유효하다 —
    ``kana_to_hangul``이 그렇다: 촉음(っ)·요음(きゃ)·발음(ん)이 이웃 글자와 함께
    처리되어 출력 길이가 안 늘어나는 자리(흡수된 글자)와 한 번에 느는 자리(합쳐진
    글자)가 생긴다. 늘어난 만큼을 **직전에 늘게 만든 글자**의 소유로 돌린다.
    """
    if not source:
        return []
    starts: list[int] = []
    buffer = ""
    for char in source:
        starts.append(len(convert(buffer)))
        buffer += char
    final = convert(buffer)
    owners: list[str] = []
    for index, start in enumerate(starts):
        stop = starts[index + 1] if index + 1 < len(starts) else len(final)
        owners.append(final[start:stop] if stop > start else "")
    return owners


# 장음부(ー)가 이어받는 모음 — omniASR류 어휘에 ー가 없어 그대로 두면 그 모라가 정렬
# 타깃에서 통째로 빠진다(ja 커버리지 95.6% → 92.3% 실측, two_pass.py). 앞 가나의 모음으로 편다.
_CHOONPU = ("ー", "ｰ")
_KANA_VOWEL: dict[str, str] = {}
for _vowel, _row in (
    ("あ", "あかさたなはまやらわがざだばぱゃゎ"),
    ("い", "いきしちにひみりぎじぢびぴ"),
    ("う", "うくすつぬふむゆるぐずづぶぷゅ"),
    ("え", "えけせてねへめれげぜでべぺ"),
    ("お", "おこそとのほもよろをごぞどぼぽょ"),
):
    for _kana in _row:
        _KANA_VOWEL[_kana] = _vowel
        _KANA_VOWEL[chr(ord(_kana) + 0x60)] = chr(ord(_vowel) + 0x60)  # 가타카나도 같이


def _expand_choonpu(text: str) -> tuple[str, frozenset[int]]:
    """장음부(ー)를 앞 가나의 모음으로 편다. 편 자리의 (편 뒤) 인덱스 집합을 함께 돌려준다
    — 표시 파생 뒤 그 자리 소유자를 앞 글자로 흡수시키는 데 쓴다(``_absorb_expanded``).

    앞이 가나가 아니면 그 ー는 버린다(정렬 타깃에서도 빠진다) — 원본과 동일한 정책이다.
    """
    if not any(char in _CHOONPU for char in text):
        return text, frozenset()
    chars: list[str] = []
    expanded: set[int] = set()
    for char in text:
        if char in _CHOONPU:
            vowel = _KANA_VOWEL.get(chars[-1]) if chars else None
            if vowel is None:
                continue
            expanded.add(len(chars))
            chars.append(vowel)
            continue
        chars.append(char)
    return "".join(chars), frozenset(expanded)


def _absorb_expanded(owners: dict[str, list[str]], expanded: frozenset[int]) -> None:
    """장음부에서 편 자리의 표시를 앞 자리로 합치고 그 자리를 비운다(모든 표기 키 공통)."""
    if not expanded:
        return
    for arr in owners.values():
        for index in sorted(expanded):
            if index < len(arr) and arr[index]:
                if index > 0:
                    arr[index - 1] += arr[index]
                arr[index] = ""


def _kana_run_units(run: str) -> tuple[str, list[str], list[str], list[str]]:
    """가나(+한자 독음 등, 라틴이 아닌) 한 덩이 → (그대로인 타깃, hangul·kana·romaji 소유자).

    ``hangul``은 덩이 전체에 ``_prefix_growth_owners``를 걸어 얻는다 — ``kana_to_hangul``이
    촉음·요음·장음을 문맥과 함께 처리하므로 격리 변환보다 정확하다.

    ``kana``/``romaji``는 **모라 그룹** 단위다: 작은 요음(ゃゅょ 등)만 직전 모라에
    합치고(``kana_romaji._SMALL_COMBINING``과 같은 집합), 촉음·발음·장음부는 독립 그룹으로
    남겨 ``moras_to_romaji``가 이웃 문맥으로 처리하게 한다(っか→kka 같은 중복은 다음
    모라를, ー는 직전 모라의 마지막 모음을 본다 — 격리 호출로는 못 낸다). 그래서 ``kana``
    표시도 요음이 앞 글자와 한 칸으로 뜬다(きゃ → 한 칸) — 문자 단위였던 벤치 원본과
    달리, 실제로 한 모라인 것을 한 노트로 보여주는 쪽이 더 낫다는 판단이다(표시 파생은
    새 요구사항이라 이 선택에는 실측 문턱값이 걸려있지 않다 — 정렬 자체의 문자 단위 타깃과
    세그 늘이기 문턱은 손대지 않았다).
    """
    from everyric2.text.kana_hangul import kana_to_hangul
    from everyric2.text.kana_romaji import _SMALL_COMBINING, moras_to_romaji

    if not run:
        return "", [], [], []
    hangul_owners = _prefix_growth_owners(run, kana_to_hangul)

    group_spans: list[list[int]] = []
    for index, char in enumerate(run):
        if char in _SMALL_COMBINING and group_spans:
            group_spans[-1][1] = index + 1
        else:
            group_spans.append([index, index + 1])
    group_texts = [run[a:b] for a, b in group_spans]
    romaji_pieces = moras_to_romaji(group_texts)

    kana_owners = [""] * len(run)
    romaji_owners = [""] * len(run)
    for (start, _end), group_text, romaji in zip(group_spans, group_texts, romaji_pieces):
        kana_owners[start] = group_text
        romaji_owners[start] = romaji
    return run, hangul_owners, kana_owners, romaji_owners


# 라틴(영어) 낱말 — 아포스트로피는 낱말 중간에만 허용한다(don't → 통째로, 't → 't 로는
# 안 가른다). ``_WORD_RE``(en 경로)와 달리 첫 글자가 반드시 알파벳이어야 한다.
_LATIN_WORD_RE = re.compile(r"[A-Za-z][A-Za-z']*")


def derive_ja_display_units(
    source: str, *, phonetic: bool = True, expand_long: bool = True
) -> LineUnits:
    """일본어 한 줄 → (가나 독음 정렬 타깃, {hangul, kana, romaji} 표시).

    **혼재 표기**: 라틴(영어) 낱말이 섞여 있으면(numb numb 라틴 비율 53.7%, 벤치 실측)
    그 구간만 en 경로(``derive_en_display_units``)로 갈아 끼운다 — 통째로 글자 단위
    (``n|u|m|b``)로 쪼개지는 것을 막는다(``two_pass.py``의 ``_mixed_units``/``_route_latin``
    이식). 라틴이 아닌 나머지(가나·한자 독음·구두점)는 ja 경로 그대로다. 결과 ``target``은
    두 표기의 타깃(가나 문자·IPA 문자)이 **원문 등장 순서대로 이어붙은 하나의 문자열**이라
    — 정렬은 한 번의 forced-align 호출로 끝나고, 라인 안에서 계열이 바뀌는 경계에서도
    시간축이 끊기지 않는다(연속성은 ``refine_window``의 세그 구성이 보장한다 —
    ``tests/test_refine_window.py`` 참조).

    한자(중국어 등 일본어 형태소 분석이 못 읽는 한자)는 ``tokenize_reading``이 읽어내지
    못하면 원문 그대로 남는다 — 이 저장소에는 한자→한글 독음 사전이나 병음(로마자) 엔진이
    없어 그 이상은 못 한다(가능한 범위까지, 못 하면 원문 통과). 실제 일본어 한자(見た 류)는
    ``tokenize_reading``이 정상적으로 읽어 가나 독음이 나오므로 이 제약과 무관하다.

    ``phonetic=True``(기본값)는 ``everyric2.text.pron_style``이 실제로 쓰는 값과 같다 —
    조사(は→わ 등)·장음까지 반영된 발음형 독음이다. ``expand_long=True``(기본값, 채택
    구성과 동일)는 장음부 ー를 앞 모음으로 펴 정렬 타깃 커버리지를 지킨다
    (``_expand_choonpu``) — 표시는 원래대로 앞 글자에 흡수된다.
    """
    from everyric2.text.ja_reading import tokenize_reading

    tokens = tokenize_reading(source, phonetic=phonetic)
    pieces: list[str] = []
    for token in tokens:
        reading = token.reading or token.surface
        if reading:
            pieces.append(reading)
    stage1 = "".join(pieces)
    if not stage1:
        return LineUnits("", {key: [] for key in JA_DISPLAY_KEYS})

    target_parts: list[str] = []
    hangul_owners: list[str] = []
    kana_owners: list[str] = []
    romaji_owners: list[str] = []
    word_end: list[bool] = []

    def append_ja_run(run: str) -> None:
        """가나(비라틴) 덩이 하나를 이어붙인다. 장음부 펴기는 **이 덩이 안에서** 한다 —
        라틴 라우팅 뒤에 전역으로 걸면 라틴 치환으로 문자열 길이가 바뀌어 편 자리 인덱스가
        어긋난다(IPA 유닛 길이는 원문 낱말 길이와 다르다)."""
        if not run:
            return
        expanded: frozenset[int] = frozenset()
        if expand_long:
            run, expanded = _expand_choonpu(run)
        if not run:
            return
        run_target, run_hangul, run_kana, run_romaji = _kana_run_units(run)
        run_owners = {"hangul": run_hangul, "kana": run_kana, "romaji": run_romaji}
        _absorb_expanded(run_owners, expanded)
        target_parts.append(run_target)
        hangul_owners.extend(run_owners["hangul"])
        kana_owners.extend(run_owners["kana"])
        romaji_owners.extend(run_owners["romaji"])
        word_end.extend([False] * len(run_target))

    def append_latin_run(word: str) -> None:
        # 라틴 구간만 낱말 경계를 옮긴다(요구사항 5 — ja 경로는 원문 공백이 이미 소유자에
        # 실려 있어 이 표시가 필요 없다). ``derive_en_display_units``가 이 낱말 하나에
        # 대해 낸 word_end를 그대로 이어 붙인다.
        en_units = derive_en_display_units(word)
        target_parts.append(en_units.target)
        hangul_owners.extend(en_units.owners["hangul"])
        kana_owners.extend(en_units.owners["kana"])
        romaji_owners.extend(en_units.owners["romaji"])
        word_end.extend(en_units.word_end)

    pos = 0
    for match in _LATIN_WORD_RE.finditer(stage1):
        append_ja_run(stage1[pos : match.start()])
        append_latin_run(match.group(0))
        pos = match.end()
    append_ja_run(stage1[pos:])

    return LineUnits(
        "".join(target_parts),
        word_end=word_end,
        owners={"hangul": hangul_owners, "kana": kana_owners, "romaji": romaji_owners},
    )
