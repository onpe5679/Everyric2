"""일본어 텍스트 → 토큰별 (원문 표면, 히라가나 읽기, 원문 오프셋).

"일본어를 어떻게 읽는가"의 단일 소유 모듈이다. 읽기는 표시용 발음만의 문제가 아니다 —
``text.reading``이 이 읽기로 모라를 쪼개고, 그 모라 열이 발음 음절 타이밍 DP 정렬의
입력이다. 독음이 틀리면 모라 열이 틀어져 가라오케 음절 타이밍이 그대로 어긋난다.

그래서 사전 표제어를 문맥 없이 긁는 pykakasi 대신 형태소 분석(fugashi + unidic-lite)을
1순위로 쓴다. 실측: 今更止められない → pykakasi는 いまさら"やめ"られない (정답 とめ),
縋って → "つい"って (정답 すがって). 형태소 분석은 둘 다 맞히고 涙を止める(とめる) /
風が止む(やむ)의 문맥 구분까지 한다. fugashi/unidic-lite를 못 쓰는 환경에서는 pykakasi로
폴백하며(어느 쪽을 썼는지는 ``reading_source``로 노출), 어느 경로든 계약은 같다.

계약: **토큰 표면을 순서대로 이어 붙이면 원문이 정확히 복원된다** (공백·기호·라틴·숫자
포함). ``reading.py``의 모라 글자 오프셋과 ``map_pron_alignment_to_line``의 원문 역매핑,
worker의 ``_full_coverage_words``가 모두 이 불변식에 걸려 있다.
"""
from __future__ import annotations

import logging
import re
import threading
import unicodedata
from dataclasses import dataclass

from everyric2.text.ja_numbers import digits_to_reading

logger = logging.getLogger(__name__)

_KANJI_RE = re.compile(r"[㐀-鿿]")

# UniDic의 kana/pron은 가타카나다 — 모라 표(reading.py)와 한글 변환표(kana_hangul)는
# 히라가나 기준이라 여기서 내려준다. 장음부 ー(0x30FC)는 ァ~ヶ 범위 밖이라 그대로 남고
# 촉음 ッ은 っ로 내려간다 — 둘 다 1박을 차지하므로 읽기에 남겨야 모라 수가 맞는다.
_KATAKANA_START, _KATAKANA_END = "ァ", "ヶ"

# 표층에서 읽기를 직접 뽑을 때 제외하는 품사. 조사·조동사는 표기와 음가가 갈리는 유일한
# 부류다(は→ワ, へ→エ) — 가타카나로 적힌 조사(ハ)까지 표층대로 읽으면 그 대립이 사라진다.
_PARTICLE_POS = frozenset({"助詞", "助動詞"})

# 가타카나 표층에서 읽기를 뽑을 때 통과시키는 글자. 가나·장음부·촉음만 허용하고
# 라틴·숫자·기호가 섞이면 규칙을 쓰지 않는다(그건 사전/폴백이 다룰 몫이다).
_KANA_ONLY_RE = re.compile(r"^[ぁ-ゖァ-ヺー゛゜ｦ-ﾟ]+$")
_HAS_KATAKANA_RE = re.compile(r"[ァ-ヺｦ-ﾟ]")

# 접두사 뒤에 이것이 오면 붙을 내용어가 없다 — 공백·문장부호·줄끝 (_is_orphan_prefix 참조)
_ORPHAN_TAIL_RE = re.compile(r"^[\s、。，．,\.！？!?…‥・「」『』（）\(\)\[\]〜~ー\-—/／]")

# UniDic이 数詞로 묶는 아라비아 숫자 토큰의 표면 형태 (_numeral_override 참조)
_ARABIC_DIGITS_RE = re.compile(r"^[0-9]+$")

# ---------------------------------------------------------------------------
# 수사 + 조수사(助数詞)
# ---------------------------------------------------------------------------

# 조수사 판정은 **UniDic 태그로만** 한다 — 조수사 목록을 손으로 들지 않는다. 두 태그가
# 조수사를 표시하고(실측), 어느 쪽이냐는 조수사마다 갈린다:
#   접미사형 …… 接尾辞-名詞的 (本・歳・個・枚・匹・人・冊・軒・発)
#   명사형   …… 名詞-普通名詞-助数詞可能 (分・回・階・年・度・秒・時・台・点・歩)
_COUNTER_SUFFIX_POS = ("接尾辞", "名詞的")
_COUNTER_NOUN_POS3 = "助数詞可能"
_NUMERAL_POS2 = "数詞"

# 語種(goshu)=漢 조건이 이 규칙 전체의 안전장치다. 촉음화·반탁음화는 **한자어 조수사에서만**
# 일어나고, 和語 조수사(つ・日=カ・組・羽)와 외래어 조수사(キロ・ページ)는 규칙 밖이다.
# 게다가 和語 조수사 앞에서는 수사 자체가 和語 계열로 갈리므로(1日 ついたち, 2つ ふたつ)
# 한자어 자릿수 읽기를 붙이면 이중으로 틀린다. UniDic이 그 갈림을 이미 語種에 적어 둔다
# — 실측: 같은 日이 一日에서는 漢/ニチ, 1日・3日에서는 和/カ로 태그가 다르고, 1月의 月은
# 和/ツキ, 1巻의 巻은 和/マキ, 1通의 通은 和/トーリ다. 즉 "수사가 和語로 읽히는 자리"를
# 사전이 스스로 표시해 준다 — 우리가 조수사 목록을 짐작할 필요가 없다.
_SINO_GOSHU = "漢"

# UniDic이 조수사로 표시하지 않지만 코퍼스에 아라비아 숫자와 붙어 나온 조수사.
# 이 상수의 이름이 "MEASURED"인 이유는 그대로다 — **실측된 것만** 넣는다. 다만 역할이
# 바뀌었다: 예전에는 이 집합이 판정의 전부였고(秒・人 둘) 그래서 3分이 「3 훈」으로 샜다.
# 지금은 위 UniDic 태그가 판정을 하고, 이 집합은 사전이 놓친 자리만 메운다 — 秒(名詞-
# 助数詞可能)와 人(接尾辞-名詞的)은 사전이 이미 표시하므로 여기서 빠졌다(회귀 테스트가
# 그 둘의 기존 동작을 그대로 지킨다).
# 文字: 코퍼스 4줄(「10文字以内で 答エヨ」「100文字以内で」 gsGjcLVI6X4)에서 아라비아
# 숫자 뒤에 나왔고 UniDic 태그는 名詞-普通名詞-一般이다. ま행이라 음변화가 없어
# 자릿수 읽기를 그대로 이으면 된다(じゅうもじ・ひゃくもじ).
_MEASURED_ARABIC_COUNTERS = frozenset({"文字"})


def _is_sino_counter(pos1: str, pos2: str, pos3: str, goshu: str) -> bool:
    """이 토큰이 **한자어 조수사**인가 (음변화 규칙과 자릿수 읽기의 공통 판정)."""
    if goshu != _SINO_GOSHU:
        return False
    if (pos1, pos2) == _COUNTER_SUFFIX_POS:
        return True
    return pos1 == "名詞" and pos3 == _COUNTER_NOUN_POS3


def _reads_arabic_digits(word) -> bool:
    """아라비아 숫자 뒤에서 자릿수 읽기를 붙일 조수사인가 (사전 판정 + 실측 보충)."""
    feature = word.feature
    goshu = getattr(feature, "goshu", "") or ""
    if goshu != _SINO_GOSHU:
        return False
    if word.surface in _MEASURED_ARABIC_COUNTERS:
        return True
    return _is_sino_counter(
        getattr(feature, "pos1", "") or "",
        getattr(feature, "pos2", "") or "",
        getattr(feature, "pos3", "") or "",
        goshu,
    )


# ---------------------------------------------------------------------------
# 촉음화(促音化)·반탁음화(半濁音化) — 규칙
# ---------------------------------------------------------------------------
#
# 촉음이 붙는지는 **수사 읽기의 꼬리**가 정한다. 꼬리로 키를 잡으면 합성수가 자동으로
# 따라온다 — 二十分→にじゅっぷん, 十一回→じゅういっかい, 三百回→さんびゃっかい를 따로
# 적지 않아도 된다.
#   いち(一)·はち(八)             → か·さ·た·は행 조수사 앞에서 촉음
#                                    (いっかい・いっさい・いってん・いっぷん / はっかい・はっさい)
#   く (六 ろく・百 ひゃく…)        → **か·は행에서만** (ろっかい・ろっぽん).
#                                    さ·た행은 그대로다 — ろくさい(六歳)·ろくてん(六点)이
#                                    ろっさい·ろってん이 아니다. 이 제약을 빼면 새 오류가 된다.
#   じゅう/じゅー (十)             → か·さ·た·は행 (じゅっかい・じゅっさい・じゅってん・じゅっぷん)
#   ん (三 さん・四 よん・何 なん…) → 촉음 없음. は행의 유·무성은 조수사마다 갈린다
#                                    (아래 ``_HA_ROW_COUNTERS``)
# 그 밖(に·ご·なな·きゅう)은 아무 일도 일어나지 않는다.
#
# **꼬리를 "ち"로 잡으면 안 된다** — しち(七)도 ち로 끝나는데 촉음이 없다(七件 しちけん,
# 七回 しちかい/ななかい). 대조표가 실제로 그 오류를 잡아냈다(しっけん·しっぱく·しっぽ가
# 나왔다). 그래서 いち·はち를 낱개로 적는다.
#
# 음가(pron)는 十을 ジュー로 주므로 ``じゅー``까지 함께 받는다 — ``phonetic=True`` 경로가
# 그 값을 쓴다(``_token_readings`` 참조).
_SOKUON_ALL_ROWS = frozenset("かさたは")
_SOKUON_K_H_ROWS = frozenset("かは")

# 조수사 읽기의 첫 글자 → 행(行). 청음 か·さ·た·は행만 담는다 — 탁음(が·ざ·だ·ば)·な·ま·
# や·ら·わ행 조수사는 음변화가 없다(いちだい・いちど・いちねん・いちまい・いちびょう).
_KANA_ROW = {
    ch: row
    for row, chars in (
        ("か", "かきくけこ"),
        ("さ", "さしすせそ"),
        ("た", "たちつてと"),
        ("は", "はひふへほ"),
    )
    for ch in chars
}

_HANDAKU = {"は": "ぱ", "ひ": "ぴ", "ふ": "ぷ", "へ": "ぺ", "ほ": "ぽ"}

# は행 조수사의 **어휘적** 정보. 규칙으로 만들 수 없는 두 가지만 담는다:
#
#  (1) UniDic이 음변화형을 표제 읽기로 주는 조수사의 청음 원형. 本의 읽기는 문맥과
#      무관하게 항상 ポン이고(실측: 二本도 ニ+ポン → 니폰) 杯는 항상 バイ다. 원형을
#      모르면 二本(にほん)도, 一本의 촉음(いっぽん)도 만들 수 없다 — っ 뒤에 ば가 올 수는
#      없으니 원형 없이 촉음만 얹으면 いっばい 같은 없는 꼴이 나온다.
#
#  (2) ん으로 끝나는 수사(三·何·千·万) 뒤에서 は행이 어느 쪽으로 가는가. **이건 조수사마다
#      갈리고 규칙이 없다**: 分은 반탁음(さんぷん), 本·杯·匹는 탁음(さんぼん·さんばい·
#      さんびき)이다. 같은 자리에서 갈리므로 음운만으로 유추할 수 없다. 예외의 예외로
#      よん은 連濁을 일으키지 않는다(よんほん·よんはい·よんひき — 和語 수사라서다).
#      分만 よん 뒤에서도 반탁음이 된다(よんぷん) — 세 번째 값이 그것이다.
#
# 표에 없는 は행 조수사(泊·発 등)는 촉음화 규칙만 받고 ん 뒤에서는 손대지 않는다
# (一泊→いっぱく는 규칙이 맞히고, 三泊은 기존 동작 さんはく 그대로) — 즉 이 표가
# 비어 있어도 기존보다 나빠지지 않는다. 넓히려면 사전 대조가 먼저다.
_HA_ROW_COUNTERS: dict[str, tuple[str, str, bool]] = {
    # 조수사 표면: (청음 원형 읽기, ん 뒤 형태, よん 뒤에도 ん 형태를 쓰는가)
    "分": ("ふん", "ぷん", True),
    "本": ("ほん", "ぼん", False),
    "杯": ("はい", "ばい", False),
    "匹": ("ひき", "びき", False),
    "歩": ("ほ", "ぽ", False),
}


def _long_vowel_variants(reading: str) -> tuple[str, ...]:
    """``う`` 장음을 ``ー``로도 적은 변종까지 (음가 경로 대응).

    ``phonetic=True`` 경로의 읽기는 음가(``feature.pron``)라 장음을 ー로 적는다 —
    十은 ジュウ/ジュー, 九는 キュウ/キュー로 표기가 갈린다. 수사 읽기의 꼬리로 규칙을
    찾는 이상 두 표기를 모두 받아야 한다. 실측으로 찾은 함정이다: 九時가 표층 읽기
    경로에서는 くじ로 맞고 음가 경로에서는 きゅーじ로 틀렸다(발음 표기가 쓰는 쪽이
    음가 경로다).
    """
    return (reading, reading[:-1] + "ー") if reading.endswith("う") else (reading,)


_SOKUON_ALL_ROW_TAILS = (*_long_vowel_variants("じゅう"), "いち", "はち")


def _sokuon_rows(numeral: str) -> frozenset[str]:
    """이 수사 읽기 뒤에서 촉음이 붙는 조수사의 행 집합 (없으면 빈 집합).

    한 글자 읽기는 아예 제외한다 — 촉음화는 끝 글자를 っ로 바꾸는 것이라 한 글자짜리에
    걸리면 수사가 「っ」하나로 남는다. 여기 걸리는 수사(いち·はち·ろく·ひゃく·じゅう)는
    모두 두 글자 이상이고, 한 글자 읽기 く는 九를 時 앞에서 줄인 꼴(9時 くじ)뿐이다.
    """
    if len(numeral) < 2:
        return frozenset()
    if numeral.endswith(_SOKUON_ALL_ROW_TAILS):
        return _SOKUON_ALL_ROWS
    if numeral.endswith("く"):  # ろく·ひゃく·びゃく·ぴゃく
        return _SOKUON_K_H_ROWS
    return frozenset()


def _handaku(reading: str) -> str:
    """첫 글자만 반탁음으로 (ふん→ぷん, ほ→ぽ)."""
    return _HANDAKU.get(reading[:1], reading[:1]) + reading[1:]


def _counter_base(surface: str, reading: str) -> tuple[str, tuple[str, str, bool] | None] | None:
    """조수사 읽기를 청음 원형으로 되돌린다 → (원형, 표 항목). 표를 못 믿으면 ``None``.

    표에 있는 조수사인데 관측된 읽기가 표의 어느 꼴도 아니면(복합어·미등록 활용 등
    자리가 안 맞으면) 손대지 않는다 — 짐작으로 원형을 만들지 않는다.
    """
    entry = _HA_ROW_COUNTERS.get(surface)
    if entry is None:
        return reading, None
    base, after_n, _ = entry
    if reading in (base, after_n, _handaku(base)):
        return base, entry
    return None


def _counter_sandhi(numeral: str, counter_surface: str, counter: str) -> tuple[str, str] | None:
    """(수사 읽기, 조수사 읽기) → 음변화를 적용한 새 짝. 바뀔 것이 없으면 ``None``.

    치환은 모두 **글자 수를 보존한다**(いち→いっ, ふん→ぷん, ほん→ぽん, じゅう→じゅっ) —
    ``reading.py``의 모라 글자 오프셋이 토큰 길이에 걸려 있어서 늘거나 줄면 안 된다.
    """
    if not numeral or not counter:
        return None
    based = _counter_base(counter_surface, counter)
    if based is None:
        return None
    base, entry = based
    row = _KANA_ROW.get(base[:1])
    if row is None:
        # 탁음·な·ま행… 음변화가 없는 조수사. 원형 복원만 남았을 수 있다(현재 표엔 없다).
        return (numeral, base) if base != counter else None
    if row in _sokuon_rows(numeral):
        return numeral[:-1] + "っ", (_handaku(base) if row == "は" else base)
    if entry is not None and numeral.endswith("ん"):
        _, after_n, yon_too = entry
        if yon_too or not numeral.endswith("よん"):
            return numeral, after_n
    return (numeral, base) if base != counter else None


def _katakana_as_hiragana(surface: str) -> str | None:
    """가타카나 표층을 히라가나 읽기로 바꾼다. 규칙을 쓸 수 없으면 ``None``.

    가타카나는 표음 문자라 표층이 곧 읽기다. 사전 조회는 이 경우 이득이 없고 손해만 있다 —
    실측 오독 2종(エグい→えぎい, レイニー→れーにー)이 그 예다. 반각 가나도 함께 받아
    전각으로 정규화한다(``ｱｲｳ``류가 옛 자막에 남아 있다).

    한자가 섞이면 쓰지 않는다 — 그건 사전이 필요한 진짜 조회 대상이다.
    """
    if not surface or _KANJI_RE.search(surface):
        return None
    norm = unicodedata.normalize("NFKC", surface)
    if not _HAS_KATAKANA_RE.search(norm) or not _KANA_ONLY_RE.match(norm):
        return None
    out: list[str] = []
    for ch in norm:
        # ァ~ヶ만 내린다. ー(장음)·ヷ~ヺ는 대응 히라가나가 없거나 범위 밖이라 그대로 둔다.
        out.append(
            chr(ord(ch) - 0x60) if _KATAKANA_START <= ch <= _KATAKANA_END else ch
        )
    return "".join(out)


@dataclass
class ReadingToken:
    """형태소 토큰 1개: 원문 표면 + 히라가나 읽기 + 원문 글자 구간 [start, end).

    ``pos``/``pos2``는 UniDic ``feature.pos1``/``pos2``(名詞-普通名詞, 動詞-非自立可能…).
    위키식 발음 표기의 문절(文節) 띄어쓰기가 품사 경계를 봐야 해서 실어 보낸다
    (``text.pron_style``). pos2까지 필요한 이유는 補助動詞다 — ``〜ている``/``〜てしまう``의
    いる·しまう는 pos1이 動詞라 그대로 두면 문절이 갈라지는데(``맛테 이루``), 위키는
    앞 문절에 붙여 쓴다(``맛테이루``). 그 판정이 pos2=非自立可能에만 걸려 있다.
    폴백(pykakasi) 경로와 공백 등 리터럴 토큰은 품사가 없어 빈 문자열이다.

    ``surface_reading``은 같은 토큰의 **표층 읽기**(``feature.kana``)다. ``phonetic=True``로
    받은 ``reading``(음가)과 나란히 두고 비교하는 용도로만 있다 — 음가는 エイ를 장음 ー로
    뭉개므로(鮮明 kana=センメイ / pron=センメー) 두 읽기를 맞춰 보면 "여기가 원래 い였다"를
    알 수 있다(``pron_style._restore_ei``). 표층 읽기를 따로 알 수 없는 토큰(리터럴·폴백)은
    빈 문자열이며, 그 경우 비교하는 쪽이 아무것도 하지 않는다.

    ``pos3``/``goshu``는 조수사(助数詞) 판정에만 쓴다. 조수사는 UniDic에서 pos3
    (助数詞·助数詞可能)로 표시되고, 그 조수사가 한자어냐 和語냐가 語種(goshu)에 적혀 있다
    — 수사와의 음변화(一分→いっぷん)도, 수사+조수사를 한 문절로 붙이는 판정도
    (``pron_style._starts_phrase``) 그 두 필드에 걸려 있다. 품사가 없는 토큰(리터럴·폴백)은
    빈 문자열이며, 그 경우 어느 규칙도 걸리지 않는다.
    """

    surface: str
    reading: str
    start: int
    end: int
    pos: str = ""
    pos2: str = ""
    surface_reading: str = ""
    pos3: str = ""
    goshu: str = ""


# ---------------------------------------------------------------------------
# 엔진 (지연 생성 싱글턴)
# ---------------------------------------------------------------------------

_tagger = None
_tagger_unavailable = False
_kakasi = None
# fugashi(MeCab) Tagger도 pykakasi 변환기도 내부 상태를 들고 있어 스레드 안전하지 않다.
# 번역 배치와 정렬 워커가 여러 스레드에서 동시에 부르므로(translator.py가 _kakasi_lock으로
# 막던 것과 같은 사정) 지연 생성 경합과 변환 구간을 한 락으로 함께 감싼다 — 라인당
# 밀리초 수준이라 8초짜리 HTTP 호출 옆에서는 직렬화 비용이 사실상 없다.
# 폴백 사다리가 락 안에서 다시 엔진을 잡으므로 재진입 가능한 RLock이어야 한다.
_lock = threading.RLock()


def _create_tagger():
    """fugashi Tagger 생성. 임포트 실패도 여기서 터지게 모아 둔다(폴백 판정 지점)."""
    import fugashi

    return fugashi.Tagger()


def _get_tagger():
    """형태소 분석기 지연 생성 싱글턴. 사용 불가 환경이면 None (pykakasi 폴백)."""
    global _tagger, _tagger_unavailable
    with _lock:
        if _tagger is None and not _tagger_unavailable:
            try:
                _tagger = _create_tagger()
            except Exception:
                # 의존성 미설치·사전 로드 실패 — 라인마다 재시도하지 않게 못 박는다
                _tagger_unavailable = True
                logger.warning(
                    "fugashi/unidic-lite unavailable; falling back to pykakasi readings",
                    exc_info=True,
                )
        return _tagger


def _get_kakasi():
    global _kakasi
    with _lock:
        if _kakasi is None:
            import pykakasi

            _kakasi = pykakasi.kakasi()
        return _kakasi


def _katakana_to_hiragana(text: str) -> str:
    return "".join(
        chr(ord(ch) - 0x60) if _KATAKANA_START <= ch <= _KATAKANA_END else ch for ch in text
    )


def _pykakasi_reading(surface: str) -> str:
    """표면 하나를 pykakasi로 읽는다. 실패 시 표면 그대로."""
    try:
        return "".join(item.get("hira", "") for item in _get_kakasi().convert(surface)) or surface
    except Exception:
        logger.warning("pykakasi reading failed for %r", surface, exc_info=True)
        return surface


# ---------------------------------------------------------------------------
# 토큰화
# ---------------------------------------------------------------------------


def _feature_reading(feature, attrs: tuple[str, ...]) -> str | None:
    """``attrs`` 순서로 UniDic 읽기 필드를 훑어 첫 유효값을 히라가나로 내린다."""
    for attr in attrs:
        value = getattr(feature, attr, None)
        if value and value != "*":
            return _katakana_to_hiragana(value)
    return None


def _is_orphan_prefix(words, i: int, text: str, idx: int) -> bool:
    """이 토큰이 **붙을 말이 없는 접두사**인가.

    접두사 읽기는 뒤에 오는 내용어와 한 낱말을 이룰 때만 성립한다. UniDic은 홀로 선 한자를
    접두사로 잡을 때가 있고, 그러면 접두사 전용 읽기가 나와 틀린다 — 실측(위키 사람 발음):
    「さり気ない愛 盛りすぎる愛」의 첫 愛가 接頭辞/まな로 읽혀 「마나」가 됐다(정답 「아이」).
    같은 줄의 두 번째 愛는 名詞/あい로 제대로 읽혔다. 즉 사전이 아니라 **자리**가 문제다.

    "붙을 말이 없다"의 판정: 다음 토큰이 없거나, 원문에서 이 토큰 바로 뒤가 공백·문장부호다
    (그 경우 접두사가 될 수 없다). 참이면 접두사 읽기를 버리고 폴백 사다리로 내려보낸다.
    """
    word = words[i]
    if (getattr(word.feature, "pos1", "") or "") != "接頭辞":
        return False
    tail = text[idx + len(word.surface) :]
    if not tail:
        return True
    return bool(_ORPHAN_TAIL_RE.match(tail))


def _numeral_override(words, i: int, text: str, idx: int) -> str | None:
    """아라비아 숫자 토큰 ``words[i]``를 자릿수 읽기로 바꿀지 판정.

    UniDic은 아라비아 숫자에 읽기를 주지 않아(``feature.kana``/``pron`` 둘 다 빈값) 사다리가
    표면 그대로("1")로 떨어진다(``_token_readings`` 3·4단 참조) — 한자 숫자(一/二/三…)는
    사전에 읽기가 있어 이 문제가 없다. 여기서 그 빈 자리만 메운다. **화면에 아라비아 숫자가
    그대로 남는 것**이 이 함수가 없을 때의 손상이다(실측: 「3分」→「3 훈」, 「1000回」→
    「1000 카이」, 「10年後」→「10 넨고」).

    바로 뒤 토큰이 한자어 조수사이고(``_reads_arabic_digits`` — UniDic 태그 판정) 원문에서
    숫자와 공백·기호 없이 붙어 있을 때만 적용한다. 사이가 뜨면("1 秒") 손대지 않는다.
    음변화가 걸리는 조수사(分・回…)도 이제 함께 다룬다 — ``_adopt_counter_sandhi``가 규칙으로
    촉음화·반탁음화를 얹기 때문이다. 그 규칙이 없던 시절에는 여기서 막는 것이 맞았다
    (잘못된 음변화를 만드는 것이 숫자를 남기는 것보다 나빴다).
    """
    surface = words[i].surface
    if not _ARABIC_DIGITS_RE.match(surface):
        return None
    feature = words[i].feature
    if (getattr(feature, "pos1", "") or "") != "名詞" or (getattr(feature, "pos2", "") or "") != "数詞":
        return None
    if i + 1 >= len(words):
        return None
    nxt = words[i + 1].surface
    if not _reads_arabic_digits(words[i + 1]):
        return None
    tail_start = idx + len(surface)
    if text[tail_start : tail_start + len(nxt)] != nxt:
        return None
    return digits_to_reading(surface)


# 특정 조수사 앞에서 **수사 자체가** 불규칙해지는 자리. 위의 촉음화·반탁음화는 조수사
# 쪽 음운 규칙이라 이 부류를 못 메운다 — 여기서는 수사 읽기가 딴 낱말로 바뀌거나
# (1人 ひとり) 다른 계열에서 끌어오기 때문이다(4時 よじ, 9時 くじ). 그래서 규칙이 아니라
# 표다. 두 갈래로 적는다:
#
#   series …… 수사의 **값**(1~9)마다 읽기가 정해진 계열. 和語 수사 계열이 붙는 조수사가
#             이 꼴이다(2つ→ふたつ). ``absorbs``가 참이면 그 읽기가 조수사까지 삼킨다
#             (1人→ひとり 하나로 끝. 안 지우면 "히토리" + "닌"으로 두 번 읽힌다).
#   tail  …… **끝자리 읽기**만 갈아 끼우는 것. 끝자리만 갈리므로 앞자리는 그대로 둔다
#             — 十四人→じゅうよにん, 19時→じゅうくじ가 자동으로 따라온다.
#
# 실측 근거:
#   人 — 사용자가 노래를 듣고 확인했다(たった1人 → 「탓타 히토리」, 우리 「탓타 1닌」).
#        4는 よん이 아니라 よ다(十四人→주우요닌). 3·5~10은 자릿수 읽기+にん이 맞다.
#   つ — 코퍼스 4줄(「どうしようもなく2つに裂けた心内環境を」 b_cuMcDWwsI)에서 아라비아
#        숫자와 붙어 나왔다. 和語 조수사라 語種 조건에 걸려 자릿수 읽기를 못 받고
#        「2츠」로 샜다. 계열이 1~9로 닫혀 있어 표가 완결된다(10 이상은 つ를 안 쓴다 —
#        とお로 끝난다 — 그래서 표에 없고, 없으면 손대지 않는다).
#        한자 표기 쪽 오독도 같이 사라진다(UniDic 실측: 四つ→よんつ, 六つ→むいつ,
#        八つ→ようつ. 정답 よっつ・むっつ・やっつ).
#   時 — 아라비아 숫자를 읽기 시작하면서 필요해졌다. 4時를 「욘지」로, 9時를 「큐우지」로
#        읽는 것은 오류다(정답 よじ・くじ) — 즉 이 표가 없으면 ②를 고치는 것이 새 오류를
#        만든다. 7은 しち가 표준이다(ななじ는 회화체).
#   時間 — 4만 넣는다. よじかん이 표준이고(24時間→にじゅうよじかん) 7·9는 ななじかん・
#        きゅうじかん도 통용되므로 건드리지 않는다.
#
# 一人・二人・一つ은 UniDic에 통짜 표제어로 올라 있어(ひとり/ふたり/ひと+つ) 이 표와
# 무관하게 이미 맞다 — 아래 함수는 pos2=数詞인 토큰만 보므로 그 표제어를 건드리지 않는다.
_IRREGULAR_COUNTER_NUMERALS: dict[str, tuple[dict[str, str], bool, dict[str, str]]] = {
    # 조수사: (값→수사 읽기 계열, 그 읽기가 조수사까지 삼키는가, 끝자리 읽기 치환)
    "人": ({"1": "ひとり", "2": "ふたり"}, True, {"よん": "よ"}),
    "つ": (
        {
            "1": "ひと", "2": "ふた", "3": "みっ", "4": "よっ", "5": "いつ",
            "6": "むっ", "7": "なな", "8": "やっ", "9": "ここの",
        },
        False,
        {},
    ),
    "時": ({}, False, {"よん": "よ", "なな": "しち", "きゅう": "く"}),
    "時間": ({}, False, {"よん": "よ"}),
}

# 수사 표면 → 값 문자열. 계열 표(``_IRREGULAR_COUNTER_NUMERALS``의 series)를 아라비아
# 숫자와 한자 숫자 어느 표기로도 같이 찾기 위한 정규화다.
_KANJI_DIGIT_VALUES = {
    "一": "1", "二": "2", "三": "3", "四": "4", "五": "5",
    "六": "6", "七": "7", "八": "8", "九": "9",
}


def _numeral_value(surface: str) -> str:
    """수사 표면의 값 문자열("1"~"9"). 계열 표에서 찾을 수 없는 표면이면 빈 문자열."""
    if _ARABIC_DIGITS_RE.match(surface):
        return surface
    return _KANJI_DIGIT_VALUES.get(surface, "")


def _replace_numeral_tail(reading: str, tail: dict[str, str]) -> str | None:
    """수사 읽기의 **끝자리**를 표대로 갈아 끼운다. 걸리는 것이 없으면 ``None``.

    앞자리는 건드리지 않는다 — 十四人→じゅうよにん, 十九時→じゅうくじ처럼 합성수의
    끝자리만 불규칙해지기 때문이다.
    """
    for suffix, replacement in tail.items():
        for variant in _long_vowel_variants(suffix):
            if reading.endswith(variant):
                return reading[: -len(variant)] + replacement
    return None


def _adopt_irregular_counter_numerals(tokens: list[ReadingToken]) -> None:
    """조수사 앞에서 불규칙해지는 수사 읽기를 표대로 고친다(제자리 수정).

    대상은 品詞가 名詞-数詞인 토큰뿐이다(아라비아 숫자는 ``_numeral_override``가 이미
    자릿수 읽기를 매겨 둔 상태, 한자 숫자는 사전 읽기 그대로) — 何人의 何도 人 앞에서
    名詞-数詞로 태깅되지만 값도 끝자리 읽기도 표에 없어 그대로 남는다(실측 확인).
    """
    for i, token in enumerate(tokens):
        if i + 1 >= len(tokens) or token.pos != "名詞" or token.pos2 != _NUMERAL_POS2:
            continue
        entry = _IRREGULAR_COUNTER_NUMERALS.get(tokens[i + 1].surface)
        if entry is None:
            continue
        series, absorbs, tail = entry
        whole = series.get(_numeral_value(token.surface))
        if whole is not None:
            token.reading = token.surface_reading = whole
            if absorbs:
                tokens[i + 1].reading = tokens[i + 1].surface_reading = ""
            continue
        # 읽기와 표층 읽기는 장음 표기가 갈릴 수 있어(きゅー / きゅう) 따로 치환한다
        replaced = _replace_numeral_tail(token.reading, tail)
        if replaced is not None:
            token.reading = replaced
        replaced = _replace_numeral_tail(token.surface_reading, tail)
        if replaced is not None:
            token.surface_reading = replaced


def _adopt_counter_sandhi(tokens: list[ReadingToken]) -> None:
    """수사 + 한자어 조수사 짝마다 촉음화·반탁음화를 적용한다 (제자리 수정).

    토큰 열은 빈틈 없이 이어지므로(``_tokens_from_words`` 계약) 바로 다음 토큰이 곧
    "원문에서 붙어 있는 조수사"다 — 사이에 공백·기호가 있으면 리터럴 토큰이 끼어
    자동으로 걸리지 않는다(「1 秒」이 안 바뀌는 이유와 같다).

    ``_adopt_irregular_counter_numerals`` 뒤에 돌려야 한다: 수사 읽기가 먼저 확정돼야
    촉음화 판정(끝 모라)이 맞는다. 실제로 두 표가 겹치는 자리는 없다 — 불규칙 표의
    조수사는 にん·じ·じかん·つ로 모두 음변화가 없는 쪽이다(탁음·和語).
    """
    for i, token in enumerate(tokens):
        if token.pos != "名詞" or token.pos2 != _NUMERAL_POS2 or i + 1 >= len(tokens):
            continue
        nxt = tokens[i + 1]
        if not _is_sino_counter(nxt.pos, nxt.pos2, nxt.pos3, nxt.goshu):
            continue
        changed = _counter_sandhi(token.reading, nxt.surface, nxt.reading)
        if changed is not None:
            token.reading, nxt.reading = changed
        changed = _counter_sandhi(token.surface_reading, nxt.surface, nxt.surface_reading)
        if changed is not None:
            token.surface_reading, nxt.surface_reading = changed


# 何(なに/なん) — UniDic 사전 표제어(代名詞)의 kana는 문맥과 무관하게 항상 なん으로
# 고정돼 있다(何を・何で 어느 쪽이든 동일). 그런데 실제 발음은 갈린다: 격조사(が・を・に)가
# 바로 뒤에 오면 표준 발음은 なに다(실측: 何を含んでたって → 정답 「나니오 후쿤데탓테」,
# 기존 출력 「난오 후쿤데탓테」).
#
# で・と・の・て는 일부러 뺐다 — UniDic 품사만으로는 관용구(なん 고정)와 진짜 격조사
# 용법(なに)을 가를 수 없다(실측: 何とか・何となく・何と言った가 何と戦う・何と一緒に와
# 品詞・pos2 태그가 완전히 같다 — 全部 代名詞/ナン + 助詞/格助詞). 何の도 뺐다: 何の花・
# 何のため・何の意味もない 전부 UniDic kana가 なん이고, 실제 표준 발음도 なんの다(何が
# 만큼 확고한 なに가 아니다 — 넣으면 새 오류가 된다).
#
# 何か는 한때 「か가 副助詞라 자동 제외되고, 하나로 굳어 있다」고 봤는데 실측이 반증했다
# (-tKVN2mAKRI 「何かを攫う」: 우리 「난카오」, 실제 가창 「나니카오」). 표준 발음도
# 부정칭 대명사 何か는 なにか다(なんか는 회화체 축약). 다만 부사적 용법(なんか君って…)과
# 표기만으로 못 가르는 경우가 있으므로 **何+か 뒤에 격조사(が・を・に)가 또 붙는 꼴만**
# なに로 돌린다 — 何かを・何かが・何かに는 부정칭 대명사임이 표기로 확정되는 무결한
# 부분집합이고, 위 관용구 축(何とか류)은 か 뒤에 격조사가 오지 않아 걸리지 않는다.
_NANI_CASE_PARTICLES = frozenset({"が", "を", "に"})


def _nani_override(words, i: int, text: str, idx: int) -> str | None:
    """격조사(が・を・に)가 바로, 또는 副助詞 か를 사이에 두고 붙는 何를 なに로 읽을지 판정.

    아니면 None(なん 유지)."""
    word = words[i]
    if word.surface != "何" or (getattr(word.feature, "pos1", "") or "") != "代名詞":
        return None
    if i + 1 >= len(words):
        return None
    nxt = words[i + 1]
    tail_start = idx + len(word.surface)
    # ``何一つ`` is the fixed pronoun phrase なにひとつ.  UniDic returns なん for
    # bare 何 and splits 一/つ into separate tokens, so the ordinary case-particle
    # rule below never sees it (live report 6ObvFPLk8eI).
    if (
        nxt.surface == "一"
        and i + 2 < len(words)
        and words[i + 2].surface == "つ"
        and text[tail_start : tail_start + 2] == "一つ"
    ):
        return "なに"
    # 何+か+격조사 (부정칭 대명사 なにか) — か 다음 토큰이 격조사인지까지 본다
    if nxt.surface == "か" and (getattr(nxt.feature, "pos1", "") or "") == "助詞":
        if text[tail_start : tail_start + 1] != "か" or i + 2 >= len(words):
            return None
        nxt2 = words[i + 2]
        if (
            nxt2.surface in _NANI_CASE_PARTICLES
            and (getattr(nxt2.feature, "pos2", "") or "") == "格助詞"
            and text[tail_start + 1 : tail_start + 1 + len(nxt2.surface)] == nxt2.surface
        ):
            return "なに"
        return None
    nxt_surface = nxt.surface
    if nxt_surface not in _NANI_CASE_PARTICLES:
        return None
    if (getattr(nxt.feature, "pos2", "") or "") != "格助詞":
        return None
    if text[tail_start : tail_start + len(nxt_surface)] != nxt_surface:
        return None
    return "なに"


def _sonouchi_override(word, text: str, idx: int, reading: str) -> str | None:
    """Split the lexical phrase ``その内`` before a following noun.

    UniDic can merge ``内`` with that noun (``内声`` -> ないせい, ``内空`` ->
    ないくう).  After the demonstrative その, however, 内 is the adverbial うち.
    Replace only the merged reading prefix and preserve the following noun's reading.
    """
    if idx < 2 or text[idx - 2 : idx] != "その" or not word.surface.startswith("内"):
        return None
    if not reading.startswith("ない"):
        return None
    # The merged token's suffix can itself use the wrong on-reading (内声=ないせい,
    # while this lyric means 内 + 声=うち + こえ).  Read the remaining surface as
    # a standalone noun instead of preserving that contaminated suffix.
    tail = _pykakasi_reading(word.surface[1:]) if len(word.surface) > 1 else ""
    return "うち" + (tail or reading[2:])


_KIMI_FOLLOWING_PARTICLES = frozenset({"が", "の", "を", "に", "は", "へ", "と", "も"})


def _kimi_override(words, i: int, text: str, idx: int) -> str | None:
    """Read standalone ``君`` followed by a particle as the pronoun きみ.

    Honorific ``-kun`` remains untouched when it follows a name without a textual
    boundary.  The live failure had an explicit phrase space (``身体 君の``) but
    UniDic still tagged 君 as a suffix.
    """
    word = words[i]
    if word.surface != "君" or i + 1 >= len(words):
        return None
    if idx > 0 and not text[idx - 1].isspace() and text[idx - 1] not in "「『（([、。！？…":
        return None
    nxt = words[i + 1]
    if nxt.surface not in _KIMI_FOLLOWING_PARTICLES:
        return None
    if (getattr(nxt.feature, "pos1", "") or "") != "助詞":
        return None
    return "きみ"


# 私(わたし/わたくし) — UniDic 사전은 わたくし(격식체)를 1순위로 준다. 가사에서는
# わたし가 압도적으로 우세하다(실측: 私は → 정답 「와타시와」, 우리 기존 출력
# 「와타쿠시와」 — 같은 곡의 私たちは・私の願いは도 마찬가지로 틀렸다). 私事・私見・
# 私立・私鉄・私大・私利私欲・私語처럼 し로 읽는 복합어는 UniDic이 통째로 한 표제어로
# 묶어 내려주므로(surface가 애초에 "私" 한 글자가 아니다) 이 함수가 건드릴 일이 없다
# (실측 확인 — 私自身만 私+自身으로 쪼개지고 私 자체는 여전히 대명사다).
def _watashi_override(word) -> str | None:
    """대명사 私의 기본 읽기를 わたくし에서 わたし로 낮출지 판정. 아니면 None."""
    if word.surface != "私" or (getattr(word.feature, "pos1", "") or "") != "代名詞":
        return None
    return "わたし"


def _token_readings(
    word, *, phonetic: bool = False, orphan_prefix: bool = False
) -> tuple[str, str]:
    """토큰 1개의 (읽기, 표층 읽기). 폴백 사다리(위에서부터):

    1. ``feature.kana`` — UniDic의 표층 읽기(활용형 그대로). 1순위.
    2. ``feature.pron`` — 발음형. kana가 비고 pron만 채워진 항목이 있다. 순서가 중요한데,
       조사 は는 kana=ハ / pron=ワ라 pron을 먼저 보면 모라가 は가 아니라 わ가 되어
       ``reading.py``의 가나 표·DP 비용이 기대하는 표층 읽기와 어긋난다.
    3. 표면에 한자가 있으면 **그 토큰만** pykakasi로 읽는다 — UniDic 미등록 한자어
       (가사에 흔한 조어·이체자)를 표면 그대로 흘리면 모라가 통째로 비어 타이밍이 무너진다.
    4. 그 외(라틴·숫자·기호)는 표면 그대로 — reading.py가 ASCII 유닛으로 따로 센다.

    위 "kana를 먼저 보는 이유"는 **모라 정렬용 기본값**에만 해당한다. ``phonetic=True``면
    1·2의 순서를 뒤집어 ``pron``(음가)을 먼저 본다 — 사람이 쓴 발음 표기는 표층이 아니라
    음가를 적기 때문이다(조사 は→ワ, 王女→オージョ). 실측(보카로 위키 사람 발음 2,207줄
    완전일치): kana 우선 72.0% vs pron 우선 75.5%. 반대로 모라 열은 표층 읽기를 기대하므로
    ``text_to_moras``는 기본값(kana 우선)을 계속 쓴다.

    두 번째 반환값은 항상 **표층 읽기(kana 우선)** 다 — 음가가 뭉갠 자리를 되살리려는
    호출부가 두 읽기를 나란히 비교할 수 있게 함께 내려보낸다(``ReadingToken`` 참조).
    """
    feature = word.feature
    # 0단: 가타카나 표기는 **이미 표음 문자**라 사전을 조회할 이유가 없다. 조회하면 오히려
    # 틀린다 — 실측(위키 사람 발음): エグい를 UniDic이 えぎい로 읽고(에기이요, 정답 에구이요),
    # レイニー의 pron이 れーにー로 장음을 뭉개 레에니이가 된다(정답 레이니이). 두 사례가
    # 독음오류 48줄 중 9줄이었다. 표층을 그대로 가나로 바꾸면 둘 다 사라진다.
    surface_kana = _katakana_as_hiragana(word.surface)
    if surface_kana is not None and (getattr(feature, "pos1", "") or "") not in _PARTICLE_POS:
        return surface_kana, surface_kana
    order = ("pron", "kana") if phonetic else ("kana", "pron")
    reading = None if orphan_prefix else _feature_reading(feature, order)
    if reading is None:
        # 사다리 3·4단: UniDic에 읽기가 없는 토큰 — 이 경우 표층/음가 구분 자체가 없다
        fallback = (
            _pykakasi_reading(word.surface)
            if _KANJI_RE.search(word.surface)
            else word.surface
        )
        return fallback, fallback
    surface_reading = None if orphan_prefix else _feature_reading(feature, ("kana", "pron"))
    return reading, surface_reading or reading


def _tokens_from_words(words, text: str, *, phonetic: bool = False) -> list[ReadingToken]:
    """형태소 토큰을 원문 오프셋에 다시 앉힌다.

    MeCab은 공백을 토큰으로 내주지 않는다 — 표면을 그냥 이어 붙이면 원문보다 짧아져
    이후 오프셋이 전부 밀리고, 그 밀림은 예외 없이 조용히 발음 타이밍을 망가뜨린다.
    그래서 표면을 원문에서 앞으로 검색해 위치를 다시 잡고, 건너뛴 구간(공백 등)은
    읽기=표면인 리터럴 토큰으로 내보내 '표면 이어 붙이기 = 원문' 계약을 지킨다.

    ``words``는 1순위 파스(``tagger(text)``)든 N-best 대안 파스(``nbestToNodeList``)든
    같은 노드 열이라 같은 규칙으로 앉힌다 — 대안 파스마다 규칙을 복제하면 오프셋 계약이
    한쪽에서만 깨진다.
    """
    tokens: list[ReadingToken] = []
    pos = 0
    for i, word in enumerate(words):
        surface = word.surface
        if not surface:
            continue
        idx = text.find(surface, pos)
        if idx < 0:
            # 표면이 원문에서 안 잡히는 이례적 경우(정규화 등) — 읽기 하나를 잃더라도
            # 오프셋 계약은 지킨다. 아래 리터럴 보충이 이 구간을 원문 그대로 덮는다.
            continue
        if idx > pos:
            tokens.append(ReadingToken(text[pos:idx], text[pos:idx], pos, idx))
        reading, surface_reading = _token_readings(
            word, phonetic=phonetic, orphan_prefix=_is_orphan_prefix(words, i, text, idx)
        )
        sonouchi = _sonouchi_override(word, text, idx, reading)
        if sonouchi is not None:
            reading = sonouchi
        sonouchi_surface = _sonouchi_override(word, text, idx, surface_reading)
        if sonouchi_surface is not None:
            surface_reading = sonouchi_surface
        numeral = _numeral_override(words, i, text, idx)
        if numeral is not None:
            reading = surface_reading = numeral
        else:
            nani = _nani_override(words, i, text, idx)
            if nani is not None:
                reading = surface_reading = nani
            else:
                kimi = _kimi_override(words, i, text, idx)
                if kimi is not None:
                    reading = surface_reading = kimi
                else:
                    watashi = _watashi_override(word)
                    if watashi is not None:
                        reading = surface_reading = watashi
        tokens.append(
            ReadingToken(
                surface,
                reading,
                idx,
                idx + len(surface),
                pos=getattr(word.feature, "pos1", "") or "",
                pos2=getattr(word.feature, "pos2", "") or "",
                surface_reading=surface_reading,
                pos3=getattr(word.feature, "pos3", "") or "",
                goshu=getattr(word.feature, "goshu", "") or "",
            )
        )
        pos = idx + len(surface)
    if pos < len(text):
        tokens.append(ReadingToken(text[pos:], text[pos:], pos, len(text)))
    _adopt_irregular_counter_numerals(tokens)
    _adopt_counter_sandhi(tokens)
    return tokens


def _tokenize_with_tagger(tagger, text: str, *, phonetic: bool = False) -> list[ReadingToken]:
    return _tokens_from_words(tagger(text), text, phonetic=phonetic)


def _tokenize_with_kakasi(text: str) -> list[ReadingToken]:
    """pykakasi 폴백. convert의 'orig'는 이어 붙이면 원문이 복원되므로 누적 오프셋으로
    구간을 매긴다 (형태소 경로와 계약 동일)."""
    try:
        items = _get_kakasi().convert(text)
    except Exception:
        logger.warning("pykakasi unavailable; emitting the line as one literal token", exc_info=True)
        return [ReadingToken(text, text, 0, len(text))]

    tokens: list[ReadingToken] = []
    pos = 0
    for item in items:
        surface = item.get("orig", "")
        if not surface:
            continue
        end = pos + len(surface)
        tokens.append(ReadingToken(surface, item.get("hira") or surface, pos, end))
        pos = end
    if pos < len(text):
        tokens.append(ReadingToken(text[pos:], text[pos:], pos, len(text)))
    return tokens


# 루비(후리가나) 표기: 한자 뭉치 바로 뒤에 괄호로 읽기를 적어 둔 가사 관례.
# 실측 예: 涙（シル）, 動画（トコ）, 時間（トキ）, 誕生日(バースデー) — 반자 괄호도 쓰인다.
_RUBY_RE = re.compile(r"[㐀-鿿々]+[（(]([぀-ヿ]+)[）)]")


def _adopt_ruby_readings(text: str, tokens: list[ReadingToken]) -> None:
    """``한자런（가나）`` 패턴에서 괄호 안 가나만 읽고 한자·괄호의 읽기는 비운다(제자리 수정).

    가사가 루비를 달아 둔 곳은 작사가가 지정한 독음이므로 사전 독음보다 정확하다
    (涙（シル）는 "나미다"가 아니라 "시루"로 불린다). 실측 순이득 +0.7p(개선 14줄,
    악화 6줄) — 위키가 한자 독음과 루비를 **둘 다** 적는 경우가 있어 손실이 섞인다.
    그래서 기본값이 아니라 옵션으로 둔다.

    괄호 자체는 읽기에서 사라지지만(읽기는 가나 열이다) 표면·오프셋 계약은 건드리지
    않으므로, 괄호를 표기에 남기고 싶은 호출부는 ``token.surface``를 보면 된다
    (``pron_style``이 그렇게 한다 — 위키도 ``(토코)``처럼 괄호를 남긴다).
    """
    for match in _RUBY_RE.finditer(text):
        kana_start, kana_end = match.start(1), match.end(1)
        for token in tokens:
            if token.end <= match.start() or token.start >= match.end():
                continue
            if kana_start <= token.start and token.end <= kana_end:
                continue  # 괄호 안 가나 — 읽기 유지
            if token.start <= kana_start and kana_end <= token.end:
                # 형태소 분석기가 루비 전체를 한 토큰으로 삼킨 경우 — 가나만 남긴다
                token.reading = _katakana_to_hiragana(text[kana_start:kana_end])
            else:
                token.reading = ""


def tokenize_reading(
    text: str, *, phonetic: bool = False, adopt_ruby: bool = False
) -> list[ReadingToken]:
    """일본어 텍스트를 (표면, 히라가나 읽기, 원문 오프셋) 토큰 열로 쪼갠다.

    표면을 순서대로 이어 붙이면 항상 원문이 복원되고, 각 토큰은
    ``text[token.start:token.end] == token.surface``를 만족한다. 두 옵션은 **읽기만**
    바꾸며 이 계약에는 손대지 않는다.

    Args:
        phonetic: 읽기 사다리를 ``feature.pron``(음가) 우선으로 바꾼다 — 사람이 쓴 발음
            표기를 재현할 때 쓴다(``_token_readings`` 참조). 기본값은 표층 읽기(kana)
            우선이며 모라 정렬(``reading.py``)이 그 값을 전제한다.
        adopt_ruby: ``한자런（가나）``의 괄호 안 가나만 읽는다(``_adopt_ruby_readings``).
    """
    if not text:
        return []
    with _lock:
        tagger = _get_tagger()
        if tagger is None:
            tokens = _tokenize_with_kakasi(text)
        else:
            try:
                tokens = _tokenize_with_tagger(tagger, text, phonetic=phonetic)
            except Exception:
                logger.warning(
                    "morphological tokenization failed; falling back to pykakasi", exc_info=True
                )
                tokens = _tokenize_with_kakasi(text)
    if adopt_ruby:
        _adopt_ruby_readings(text, tokens)
    return tokens


def has_ruby(text: str) -> bool:
    """``한자런（가나）`` 루비가 있는 라인인가 (``adopt_ruby``가 실제로 뭔가 바꿀 라인인가)."""
    return bool(text) and _RUBY_RE.search(text) is not None


def tokenize_reading_nbest(
    text: str, *, n: int = 8, phonetic: bool = False, adopt_ruby: bool = False
) -> list[list[ReadingToken]]:
    """같은 라인의 **대안 파스**(MeCab N-best) 토큰 열들. 1순위 파스는 제외한다.

    남은 오독은 대부분 "어느 독음이 맞나" 하나로 수렴하고(사람 발음 2,207줄 대비 82.1%),
    그 갈림이 정확히 MeCab의 후순위 파스에 들어 있다 — ``私は三日月を見た``의 nbest는
    ワタクシ/ワタシ와 ミッカツキ/ミッカズキ/ミカツキ를 모두 내놓는다(실측). 사전은 어느
    쪽이 맞는지 모르지만 오디오는 알기 때문에(``ctc_engine`` 심판) 후보만 만들어 준다.

    실패(폴백 엔진·미설치·nbest 예외)는 빈 목록이다 — 후보가 없으면 심판이 안 돌고
    기존 동작 그대로다. 오프셋·표면 계약은 1순위 파스와 동일하다(``_tokens_from_words``).
    """
    if not text or n <= 1:
        return []
    parses: list[list[ReadingToken]] = []
    with _lock:
        tagger = _get_tagger()
        if tagger is None:
            return []
        try:
            # 1순위 파스는 호출부가 이미 갖고 있으므로 건너뛴다 (첫 열 = tagger(text))
            for i, nodes in enumerate(tagger.nbestToNodeList(text, n)):
                if i == 0:
                    continue
                parses.append(_tokens_from_words(nodes, text, phonetic=phonetic))
        except Exception:
            logger.warning("MeCab n-best parse failed for %r", text, exc_info=True)
            return []
    if adopt_ruby:
        for tokens in parses:
            _adopt_ruby_readings(text, tokens)
    return parses


def tokenize_reading_pykakasi(text: str) -> list[ReadingToken]:
    """사전 표제어 기반(pykakasi) 토큰 열 — 형태소 분석과 **한자 독음이 갈리는** 변종.

    폴백 경로를 후보 생성에 재사용한다: 今更止められない를 pykakasi는 いまさら"やめ"られない로
    읽고 형태소 분석은 とめ로 읽는다(실측). 어느 쪽이 맞는지는 오디오가 고른다.
    """
    if not text:
        return []
    with _lock:
        return _tokenize_with_kakasi(text)


def kana_reading(text: str, *, phonetic: bool = False, adopt_ruby: bool = False) -> str:
    """텍스트 전체의 히라가나 읽기 (비일본어 구간은 원문 그대로 통과).

    옵션 의미는 ``tokenize_reading``과 같다.
    """
    return "".join(
        token.reading
        for token in tokenize_reading(text, phonetic=phonetic, adopt_ruby=adopt_ruby)
    )


def reading_source() -> str:
    """실제로 쓰이는 읽기 엔진 이름: ``"fugashi"`` 또는 ``"pykakasi"``.

    호출부가 엔진에 따라 프롬프트를 달리 짜야 하기 때문에 노출한다 (translator의
    다의어 후보 주석은 사전 읽기 폴백일 때만 의미가 있다).
    """
    return "fugashi" if _get_tagger() is not None else "pykakasi"
