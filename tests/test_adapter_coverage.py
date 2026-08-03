"""어댑터 선택은 글자 수가 아니라 vocab 커버리지로 정해진다 — 그 규칙의 회귀 테스트.

## 무엇이 깨졌었나 (실측)

`detect_language_from_text`가 낱글자 수만 비교해 언어를 정했다. 한글 234자 + 라틴 406자인
KPOP 곡(`ItSKahBISg0`, YENA '캐치 캐치')은 라틴이 더 많다는 이유로 `en`으로 판정됐고,
`eng` 어댑터 vocab에는 한글이 0개라 순수 한글 15줄이 정렬에서 통째로 빠져 균등 보간만
됐다. 구간 오차가 −2.5초에서 −11.4초로 단조 악화했다.

같은 오디오·같은 가사(유튜브 수동 작성 ko 자막 61줄)를 어댑터만 바꿔 정렬한 결과:

| 어댑터 | 글자 커버리지 | 정렬된 줄 | \\|잔차\\| 중앙 | \\|잔차\\| 평균 | \\|잔차\\| p90 |
|---|---|---|---|---|---|
| eng | 0.632 | 46/61 | 2.002 | 3.718 |  9.556 |
| kor | 0.994 | 61/61 | 0.397 | 0.880 |  1.868 |
| jpn | 0.628 | 44/61 | 3.673 | 5.716 | 13.699 |

커버리지가 결과를 그대로 예측한다 — 그래서 판정 기준을 커버리지로 바꿨다.

## eng 어댑터를 왜 아예 안 쓰는가 (실측)

`kor`/`jpn`/`cmn` 모두 ASCII 소문자 26개를 갖고 있어 라틴을 eng와 똑같이 덮는다. 순수
영어 4곡을 같은 오디오·같은 가사로 세 어댑터에 돌려 유튜브 **수동 작성** 영어 자막의 cue
시작 시각과 비교했다 (가사 입력 자체를 자막 줄로 써서 라인 매칭 오차를 없앴다).
어댑터 간 confidence는 vocab 크기가 달라 posterior 스케일이 다르므로 쓰지 않았다.

|잔차| 중앙 / 평균 / p90, 단위 초:

| 곡 | eng | kor | jpn |
|---|---|---|---|
| dQw4w9WgXcQ Never Gonna Give You Up (59줄) | 0.059 / 0.195 / 0.559 | 0.064 / 0.252 / 0.225 | 0.064 / 0.256 / 0.315 |
| JGwWNGJdvx8 Shape of You (81줄)            | 0.080 / 0.301 / 0.165 | 0.074 / 0.135 / 0.165 | 0.109 / 0.321 / 0.987 |
| 2Vv-BfVoq4g Perfect (54줄)                 | 0.984 / 0.994 / 1.143 | 0.998 / 1.004 / 1.133 | 0.990 / 0.995 / 1.143 |
| lp-EO5I60KA Thinking Out Loud (54줄)       | 0.069 / 0.074 / 0.134 | 0.069 / 0.097 / 0.149 | 0.069 / 0.094 / 0.145 |

(Perfect의 ~1.0초는 세 어댑터에 공통인 자막 자체의 선행 바이어스다. 네 곡 모두 248줄
전부 word_segments가 잡혔고 시작 시각은 단조였다.)

중앙값 차는 최대 0.035초 = CTC 프레임(20ms) 2칸 이하이고 부호가 곡마다 뒤집힌다. eng가
평균에서 이긴 곡이 둘, 진 곡이 하나, 같은 곡이 하나다. 정확도 우위가 없는 반면 eng는
vocab 154개에 한글·가나가 0개라 CJK가 조금이라도 섞이면 그 글자를 통째로 놓친다. 그래서
`MMS_LANG_CODES["en"]`을 `kor`로 바꿔 eng를 선택 경로에서 뺐다. 남은 셋 중 kor을 고른
근거는 네 곡의 최악 p90이 가장 낮았다는 것이다 (kor 1.133 / eng 1.143 / jpn 1.143,
그리고 Shape of You에서 kor 0.165 vs jpn 0.987).

eng만 덮는 글자가 91개 있다 (악센트 라틴 ß à é ñ ø 등 + 그리스·키릴·히브리 조각 + 일부
구두점). 이 코퍼스(일본어/한국어 가창)에는 사실상 나타나지 않고, 나타나도 그 글자 하나만
토큰에서 빠질 뿐 줄 전체는 이웃 글자로 정렬된다 — 반대로 kor만 덮는 글자는 1267개다.
"""

import json
from pathlib import Path

import pytest

from everyric2.alignment.ctc_engine import (
    MMS_LANG_CODES,
    _ADAPTER_SCRIPTS,
    _char_script,
    adapter_coverage,
    detect_language_from_text,
    script_census,
)

CENSUS = json.loads(
    (Path(__file__).parent / "fixtures" / "mms_adapter_script_census.json").read_text(
        encoding="utf-8"
    )
)
KOR_VOCAB = json.loads(
    (Path(__file__).parent / "fixtures" / "mms_kor_vocab.json").read_text(encoding="utf-8")
)

# 실측 가사 (짧게 자른 대표 줄). 판정은 스크립트 구성만 보므로 전문이 필요 없다.
KPOP_MIXED = "\n".join(  # ItSKahBISg0 — 한글 + 라틴, 라틴이 더 많다
    [
        "Hey",
        "심장이 Up down Up down",
        "가까이 와 봐 내 눈을 봐",
        "Catch me catch me catch me now",
        "너의 맘을 훔쳐 갈 거야",
        "DA DA RA DA DA",
    ]
)
JA_PURE = "本当は分かってる\nこんなことだったって\n弾いて歌ってずっとそうだ"
JA_WITH_LATIN = "Overdose 君とふたり\n解像度の悪い夢を見た\nDon't stop it music, darling"
KO_PURE = "그대 없는 밤은 길고\n달빛만 창가에 스미네\n말하지 못한 마음"
EN_PURE = "We're no strangers to love\nYou know the rules and so do I\nNever gonna give you up"
ZH_PURE = "月亮代表我的心\n你问我爱你有多深\n我爱你有几分"


# --------------------------------------------------------------------------
# 하드코딩한 스크립트 표가 실제 vocab과 어긋나지 않는가
# --------------------------------------------------------------------------


def test_census_fixture_matches_measured_vocab_sizes():
    # 이 픽스처가 실제 어댑터를 센 것이라는 전제 — 다른 테스트가 고정한 크기와 맞아야 한다
    assert CENSUS["eng"]["vocab_size"] == 154
    assert CENSUS["kor"]["vocab_size"] == 1330
    assert CENSUS["jpn"]["vocab_size"] == 2268
    assert CENSUS["cmn-script_simplified"]["vocab_size"] == 4495
    for adapter in ("eng", "kor", "jpn", "cmn-script_simplified"):
        assert CENSUS[adapter]["pad_token_id"] == 0


def test_kor_row_of_census_is_reproducible_from_the_real_vocab_fixture():
    """census의 kor 행을 실제 kor vocab 픽스처(1330개)로 독립 재계산한다."""
    counts = {"latin": 0, "hangul": 0, "kana": 0, "han": 0}
    for token in KOR_VOCAB:
        if len(token) != 1:
            continue
        script = _char_script(token)
        if script:
            counts[script] += 1
    assert counts == CENSUS["kor"]["scripts"]


@pytest.mark.parametrize("adapter", sorted(_ADAPTER_SCRIPTS))
def test_declared_scripts_are_substantially_present_in_the_real_vocab(adapter):
    # 표에 넣은 스크립트는 vocab에 실제로 넉넉히 있어야 한다 (라틴 26자가 하한)
    for script in _ADAPTER_SCRIPTS[adapter]:
        assert CENSUS[adapter]["scripts"][script] >= 26, (
            f"{adapter}가 {script}를 덮는다고 선언했지만 vocab에 "
            f"{CENSUS[adapter]['scripts'][script]}개뿐이다"
        )


@pytest.mark.parametrize("adapter", sorted(_ADAPTER_SCRIPTS))
def test_undeclared_scripts_are_at_most_a_trace_in_the_real_vocab(adapter):
    """표에서 뺀 스크립트는 흔적 수준(한 자리 수)이어야 한다.

    eng의 한자 8개, cmn의 가나 4개가 이 흔적이다 — 수천 자 규모와 같은 'True'로 접으면
    커버리지 순위가 뒤집힌다. 어느 어댑터든 새로 수십 자를 갖게 되면 이 테스트가 터져
    표를 갱신하게 만든다.
    """
    for script, n in CENSUS[adapter]["scripts"].items():
        if script not in _ADAPTER_SCRIPTS[adapter]:
            assert n <= 8, f"{adapter}에 {script}가 {n}개 — 표에서 빠져 있다"


def test_no_adapter_vocab_has_ascii_uppercase():
    # 커버리지 계산이 대문자를 'latin'으로 세는 근거 — vocab에는 대문자가 없고
    # _resolve_token_char의 소문자화 폴백이 이를 메운다
    for adapter in CENSUS:
        if adapter.startswith("_"):
            continue
        assert CENSUS[adapter]["latin_ascii_upper"] == 0
        assert CENSUS[adapter]["latin_ascii_lower"] == 26


def test_every_selectable_adapter_has_a_script_row():
    # MMS_LANG_CODES가 가리키는 어댑터는 전부 커버리지 판정이 가능해야 한다
    assert set(MMS_LANG_CODES.values()) <= set(_ADAPTER_SCRIPTS)


# --------------------------------------------------------------------------
# 스크립트 분류
# --------------------------------------------------------------------------


def test_uppercase_latin_counts_as_covered():
    # 회귀 방지: 대문자를 커버 불가로 세면 판정이 소문자화 폴백과 어긋난다
    assert _char_script("A") == "latin"
    assert _char_script("Z") == "latin"
    assert adapter_coverage(script_census("DA DA RA DA DA"), "kor") == 1.0
    assert adapter_coverage(script_census("DA DA RA DA DA"), "jpn") == 1.0


def test_non_sung_chars_are_excluded_from_the_denominator():
    # 숫자·구두점·공백이 분모에 들어가면 어떤 어댑터도 1.0을 못 받는다
    for ch in " 0123456789!?,.-()♪":
        assert _char_script(ch) is None
    assert script_census("a, b! 1 2 ?") == {"latin": 2, "hangul": 0, "kana": 0, "han": 0}
    assert adapter_coverage(script_census("Up, down! 1 2"), "kor") == 1.0


def test_census_counts_each_script():
    counts = script_census("안녕 hello こんにちは 漢字")
    assert counts == {"hangul": 2, "latin": 5, "kana": 5, "han": 2}


def test_coverage_of_text_with_no_alignable_chars_is_zero():
    assert script_census("123 !!! ...") == {"latin": 0, "hangul": 0, "kana": 0, "han": 0}
    for adapter in _ADAPTER_SCRIPTS:
        assert adapter_coverage(script_census("123 !!!"), adapter) == 0.0


# --------------------------------------------------------------------------
# 커버리지 값
# --------------------------------------------------------------------------


def test_kor_covers_hangul_latin_mix_completely():
    counts = script_census(KPOP_MIXED)
    assert counts["hangul"] > 0 and counts["latin"] > counts["hangul"], (
        "이 곡의 회귀 조건은 '라틴이 한글보다 많다'다"
    )
    assert adapter_coverage(counts, "kor") == 1.0
    # eng는 표에서 빠졌으므로 kor 이외 후보로 대조한다 — 라틴만 덮으니 1.0이 될 수 없다
    assert adapter_coverage(counts, "jpn") < 1.0
    assert adapter_coverage(counts, "cmn-script_simplified") < 1.0


def test_pure_latin_ties_across_every_adapter():
    """순수 영어는 세 어댑터가 모두 1.0 — 실측 잔차가 사실상 같았던 이유다."""
    counts = script_census(EN_PURE)
    for adapter in _ADAPTER_SCRIPTS:
        assert adapter_coverage(counts, adapter) == 1.0


def test_jpn_covers_kana_han_latin_mix():
    counts = script_census(JA_WITH_LATIN)
    assert counts["kana"] > 0 and counts["han"] > 0 and counts["latin"] > 10
    assert adapter_coverage(counts, "jpn") == 1.0
    assert adapter_coverage(counts, "kor") < 1.0


# --------------------------------------------------------------------------
# 언어 판정 — 단일 언어 곡의 기존 판정이 바뀌지 않는가
# --------------------------------------------------------------------------


def test_pure_japanese_still_resolves_to_ja():
    assert detect_language_from_text(JA_PURE) == ("ja", False)
    assert MMS_LANG_CODES["ja"] == "jpn"


def test_pure_korean_still_resolves_to_ko():
    assert detect_language_from_text(KO_PURE) == ("ko", False)
    assert MMS_LANG_CODES["ko"] == "kor"


def test_pure_chinese_still_resolves_to_zh():
    # zh는 MMS가 아닌 전용 베이스 모델을 쓴다 — 단일 언어 판정이 바뀌면 모델이 바뀐다
    assert detect_language_from_text(ZH_PURE) == ("zh", False)


def test_pure_english_still_resolves_to_en():
    # 반환 언어 태그는 그대로 'en'이다 (로그·디버그 가독성). 바뀐 것은 어댑터 매핑뿐.
    assert detect_language_from_text(EN_PURE) == ("en", False)
    assert MMS_LANG_CODES["en"] == "kor"


def test_japanese_with_english_hook_stays_ja():
    # 'Overdose ... Don't stop it music' — 라틴이 섞여도 jpn이 전부 덮으므로 ja
    lang, multi = detect_language_from_text(JA_WITH_LATIN)
    assert (lang, multi) == ("ja", True)


def test_hangul_heavy_song_with_few_latin_chars_stays_ko():
    # 라틴이 10자 이하면 애초에 다중 언어로 세지 않는다 (기존 동작)
    assert detect_language_from_text(KO_PURE + "\nOh") == ("ko", False)


# --------------------------------------------------------------------------
# 언어 판정 — 고쳐진 회귀
# --------------------------------------------------------------------------


def test_latin_majority_kpop_song_resolves_to_ko_not_en():
    """이 파일이 존재하는 이유. 라틴이 더 많아도 한글을 덮는 kor을 골라야 한다."""
    lang, multi = detect_language_from_text(KPOP_MIXED)
    assert (lang, multi) == ("ko", True)
    assert MMS_LANG_CODES[lang] == "kor"


def test_char_count_majority_never_decides_on_its_own():
    """라틴을 아무리 늘려도 한글이 남아 있으면 kor을 버리지 않는다."""
    for padding in (0, 50, 500, 5000):
        text = "가나다라마" + ("english " * (padding // 8))
        lang, _ = detect_language_from_text(text)
        assert MMS_LANG_CODES[lang] == "kor", f"라틴 {padding}자에서 판정이 흔들렸다"


def test_chinese_with_latin_majority_still_covers_han():
    # 한자 + 라틴 다수 → cmn이 둘 다 덮는다 (kor은 한자를 못 덮는다)
    text = ZH_PURE + "\n" + ("only you " * 40)
    lang, multi = detect_language_from_text(text)
    assert (lang, multi) == ("zh", True)
    assert MMS_LANG_CODES[lang] == "cmn-script_simplified"


def test_english_song_with_trace_ja_bridge_resolves_to_en():
    """영어 곡의 일본어 브리지 4줄(가나 2%)이 ja 판정을 뒤집으면 안 된다.

    UgK6n1KKUxY([English GUMI] About Me, 영어 56줄 + 일본어 4줄) 실측 회귀 —
    jpn vocab이 라틴까지 덮는다는 이유로 ja가 되면 fast 라우팅으로 새고, en 강제
    medium이 무산돼 첫 줄이 35초 앞으로 끌려갔다. 지목 스크립트(가나)가 흔적
    수준이면 후보에서 빠져야 한다(_NATIVE_SHARE_FLOOR).
    """
    text = (EN_PURE + "\n") * 20 + "気付いた時には終わりを告げ\n君を思い出すよ"
    counts = script_census(text)
    assert (counts["kana"] + counts["han"]) / sum(counts.values()) < 0.05
    lang, multi = detect_language_from_text(text)
    assert (lang, multi) == ("en", True)


# --------------------------------------------------------------------------
# 동점 처리
# --------------------------------------------------------------------------


def test_tie_is_broken_by_the_identifying_script_count():
    """한글 100 + 한자 100은 ko/ja/zh가 모두 0.5 — 가나가 0인 ja가 먼저 떨어진다."""
    text = "가" * 100 + "漢" * 100
    counts = script_census(text)
    assert adapter_coverage(counts, "kor") == adapter_coverage(counts, "jpn") == 0.5
    lang, multi = detect_language_from_text(text)
    assert multi is True
    assert lang != "ja", "가나가 0자인데 ja를 골랐다"
    assert lang == "ko"  # ko와 zh가 완전 동점 → 후보 순서(ja→ko→zh)로 ko


def test_full_tie_is_deterministic():
    # 같은 입력이 항상 같은 답을 줘야 한다 (dict 순회 순서에 기대지 않는다)
    text = "가" * 30 + "漢" * 30
    assert len({detect_language_from_text(text)[0] for _ in range(20)}) == 1


def test_kana_presence_wins_over_a_larger_hangul_block():
    """가나 + 한자가 한글보다 적어도, jpn이 더 많이 덮으면 ja다."""
    # 한글 30 / 가나 20 + 한자 20 → kor 0.43, jpn 0.57
    text = "가" * 30 + "あ" * 20 + "漢" * 20
    counts = script_census(text)
    assert adapter_coverage(counts, "jpn") > adapter_coverage(counts, "kor")
    assert detect_language_from_text(text)[0] == "ja"


def test_empty_and_symbol_only_text_do_not_crash():
    assert detect_language_from_text("") == ("en", False)
    assert detect_language_from_text("♪♪♪ 123 !!!") == ("en", False)


# --------------------------------------------------------------------------
# 어댑터 하나로 안 덮이는 입력 — 실측된 한계와 그 처리
# --------------------------------------------------------------------------
#
# 코퍼스 73곡(정렬 대상 4881라인 / 55486글자)을 실제 vocab에 걸어 센 결과:
#   · 한글과 가나가 **같은 줄에** 섞인 라인: 0개. 단일 어댑터로 못 덮는 라인은 없다.
#   · 라인 단위로 가나 라인과 한글 라인이 공존하는 곡: 2개. 둘 다 노래가 아니라
#     **번역 병기 가사 시트**였다 —
#       FxOfDVyITak: (가나, 한글, 한글) 3줄 블록이 74/74회 완벽히 반복(일본어 원문 /
#         한국어 발음 / 한국어 번역). 2/3가 가창이 아니다. 이 시트는 정렬된 적도 없다
#         (다운로드 403 실패). 사용자가 곧바로 깨끗한 일본어 가사로 다시 넣었고 그건
#         ja/jpn으로 판정된다.
#       ba7YbGO2aq4: 한글 8줄이 각각 직전 일본어 줄의 번역. 같은 곡의 다른 가사
#         버전(120줄)은 한글 0줄로 깨끗하고 jpn 커버리지 1.0이다.
#   · 그 두 곡을 빼면 손실은 390글자/약 53000글자 = 0.7%이고, 전손실 라인은 0개다.
#     잃는 글자는 어느 MMS 어댑터에도 없는 희귀 한자(孕 瞳 潰 籠 檎 …)와 작은
#     가나(ぃ ぉ)뿐이라 어댑터를 바꿔도 못 건진다 — 확장할 곳은 `_oov_substitute`다.
#
# 그래서 라인별 어댑터 선택(라인마다 덮는 어댑터로 갈아 정렬)은 넣지 않았다. 필요 조건인
# '같은 줄 혼용'이 0건이고, 위 2곡에 대해서는 **역효과**다 — 번역 라인까지 자신 있게
# 정렬해 한 보컬에 3배 길이의 토큰 열을 맞추려 들기 때문이다. 지금은 번역 라인이 OOV로
# 탈락해 보간되므로 최소한 가창 라인의 타이밍을 망치지 않는다. 실제로 필요한 것은
# 병기 시트를 정렬 입력에서 걸러내는 전처리이고, 그건 이 파일이 아니라 워커 소관이다.


def test_translation_annotated_sheet_picks_the_majority_script():
    """번역 병기 시트는 다수 스크립트를 고른다 — 어댑터로 고칠 문제가 아니라는 기록.

    소수 스크립트 라인은 OOV로 탈락해 보간된다(정렬 실패 라인 보간 경로). 이 판정을
    뒤집어도 반대쪽 라인이 탈락할 뿐이므로, 여기서 할 수 있는 최선은 결정론적 기록이다.
    """
    sheet = "\n".join(
        ["未来がどうとか", "미라이가 도-토카", "미래가 어떻다든가"] * 8
    )  # FxOfDVyITak과 같은 (가나, 한글, 한글) 3줄 블록
    counts = script_census(sheet)
    assert counts["kana"] > 0 and counts["hangul"] > counts["kana"]
    lang, multi = detect_language_from_text(sheet)
    assert (lang, multi) == ("ko", True)
    assert adapter_coverage(counts, "kor") > adapter_coverage(counts, "jpn")


def test_clean_japanese_version_of_the_same_song_resolves_to_ja():
    # 병기 시트를 걷어낸 가사(사용자가 실제로 다시 넣은 형태)는 ja/jpn으로 정확히 간다
    clean = "未来がどうとか 理想がどうとか\nブランコに揺られふと考えてた\nまぶたの裏 浮かんだハテナ"
    assert detect_language_from_text(clean) == ("ja", False)
    assert adapter_coverage(script_census(clean), "jpn") == 1.0


def test_hangul_and_kana_in_one_line_degrades_without_crashing():
    """같은 줄 혼용은 코퍼스에 0건이지만, 들어와도 죽지 않고 다수쪽을 덮어야 한다."""
    line = "안녕 こんにちは 안녕하세요"
    counts = script_census(line)
    assert counts["hangul"] > counts["kana"]
    lang, multi = detect_language_from_text(line)
    assert (lang, multi) == ("ko", True)
    # 어느 어댑터도 1.0이 아니다 — 이것이 '단일 어댑터로 못 덮는' 유일한 형태다
    assert max(adapter_coverage(counts, a) for a in _ADAPTER_SCRIPTS) < 1.0
