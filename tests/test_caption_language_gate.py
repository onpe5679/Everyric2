"""자막 원어 판정 오류로 인한 가사 오염 게이트 회귀 테스트.

실측 버그(2026-07-26): `zyRt-nBM3dY`는 일본어 보카로 곡인데 `video_language='vi'`이고 유일한
`-orig` ASR 트랙이 `vi-orig`다. 그래서 `detect_original_language`가 `('vi', 'asr_orig')`을
돌려주고, 자막 경로가 **일본어 오디오에 베트남어 ASR을 돌린 전사를 곡 가사로 저장한다**
(야간 배치 예행에서 `th-orig` 태국어로도 재현).

여기서 못박는 것: 라틴 전용 자막이 포기되는가, 가나·한글·한자가 하나라도 있으면 통과하는가,
사유가 호출부에 전달되는가(4xx = 확정 판정), 설정으로 끌 수 있는가, 판정 규칙
(`select_original_track`)은 그대로인가.
"""

import contextlib

import pytest

from everyric2.alignment.caption_anchors import script_counts
from everyric2.config.settings import get_settings
from everyric2.server.services import youtube_captions as yc

VIDEO = "zyRt-nBM3dY"

# 실측 트랙 구성 — 수동 12종, 자동 vi-orig. 유튜브 신호는 전부 vi를 가리킨다.
REAL_INFO = {
    "subtitles": {
        lang: [{"name": lang}]
        for lang in "ar zh-TW en fil id ja ko ms es th tr vi".split()
    },
    "automatic_captions": {"vi-orig": [{"name": "Vietnamese"}]},
    "language": "vi",
}

# REAL_INFO에서 asr '-orig' 신호까지 뺀 변형 — "제목도 asr도 아무 힌트가 없을 때"의
# 성질을 보는 테스트 전용이다. 2026-08-03부로 asr_lang_hint가 title_script_hint의
# 폴백으로 쓰이므로(c_9UTrrqcLI 수정), REAL_INFO 그대로는 제목이 없어도 asr('vi')이
# 힌트가 되어 이 성질을 더는 관찰할 수 없다.
NO_HINT_INFO = {**REAL_INFO, "automatic_captions": {}}


def _lines(*texts):
    return [{"start": i * 2.0, "end": i * 2.0 + 1.5, "text": t} for i, t in enumerate(texts)]


@contextlib.contextmanager
def _ytdlp(info, lines):
    """extract_caption_info / download_track_lines만 갈아끼운다 — 판정·정리·게이트는 실코드."""
    calls: dict = {"tracks": []}

    def fake_extract(video_id):
        calls["video_id"] = video_id
        return info

    def fake_download(video_id, lang, auto):
        calls["track"] = (lang, auto)
        calls["tracks"].append(lang)
        return lines

    orig = (yc.extract_caption_info, yc.download_track_lines)
    yc.extract_caption_info = fake_extract
    yc.download_track_lines = fake_download
    try:
        yield calls
    finally:
        yc.extract_caption_info, yc.download_track_lines = orig


@contextlib.contextmanager
def _gate(enabled: bool):
    settings = get_settings()
    saved = settings.server.caption_require_cjk
    object.__setattr__(settings.server, "caption_require_cjk", enabled)
    try:
        yield
    finally:
        object.__setattr__(settings.server, "caption_require_cjk", saved)


# --------------------------------------------------------------------------
# 1) 버그가 실제로 존재한다 — 게이트가 필요한 이유
# --------------------------------------------------------------------------


def test_youtube_signals_really_do_point_at_the_wrong_language():
    """유튜브 신호가 이 영상에서 vi를 가리킨다는 **사실**을 못박는다.

    이것이 판정에서 유튜브 신호를 빼기로 한 근거다. 신호 자체는 그대로 있으므로 (실측
    데이터가 바뀐 것이 아니다) 그 사실을 여기서 확인하고, 아래에서 우리 판정이 그것을
    따르지 않는 것을 확인한다.
    """
    assert REAL_INFO["language"] == "vi"
    assert [k for k in REAL_INFO["automatic_captions"] if k.endswith("-orig")] == ["vi-orig"]


def test_title_always_outranks_the_asr_signal():
    """옛 판정은 vi-orig를 1순위로 봐서 vi 트랙을 골랐다 — 제목이 있으면 그 신호를 안 쓴다.

    실제 zyRt-nBM3dY 영상에는 제목이 있다(일본어 문자) — REAL_INFO가 제목을 뺀 것은
    이 테스트에서 "제목이 없을 때" 성질을 별도로 보기 위해서일 뿐이다. 제목이 있으면
    asr('vi-orig')는 title_script_hint('ja')에 항상 밀린다(order_manual_tracks 우선순위
    ①lang_hint ②title_script_hint ③asr_lang_hint ④알파벳순).
    """
    titled = yc.order_manual_tracks({**REAL_INFO, "title": "シニカルナイトプラン"}, None, 12)
    assert titled[0] == "ja", f"제목의 문자 체계가 asr 신호를 이겨야 한다: {titled}"


def test_asr_signal_is_only_a_fallback_when_the_title_gives_no_hint():
    """제목도 채널명도 힌트가 없을 때만 asr('-orig') 언어가 순서에 관여한다(c_9UTrrqcLI 수정).

    이것이 실사용 결함을 고친 변경이다 — 라틴 제목뿐인 곡은 예전엔 알파벳순 폴백으로
    떨어져 원어 트랙을 아예 못 받아 봤다. 잔여 위험(자동 더빙 오사고, zyRt-nBM3dY류)은
    본문 CJK 게이트(verify_track_body)와 "힌트 언어에 수동 트랙이 없으면 포기" 규칙이
    받는다 — 이 테스트는 순서 하나만 고정한다.
    """
    order = yc.order_manual_tracks(REAL_INFO, None, 12)
    assert order[0] == "vi", f"제목이 없으면 asr 신호가 순서를 정해야 한다: {order}"

    # 제목도 asr도 없으면(NO_HINT_INFO) 재현 가능한 알파벳순으로 떨어진다 — 예전 동작 그대로.
    no_hint = yc.order_manual_tracks(NO_HINT_INFO, None, 12)
    assert no_hint[0] == "ar", f"신호가 전혀 없는데도 뭔가를 따라갔다: {no_hint}"


# --------------------------------------------------------------------------
# 2) 게이트
# --------------------------------------------------------------------------


def test_latin_only_caption_is_refused_with_a_reason():
    vietnamese_asr = _lines(
        "Anh muốn chạm vào điều bí mật",
        "Mơ hồ về em",
        "Không có ý nghĩa gì cả",
    )
    with _gate(True), _ytdlp(REAL_INFO, vietnamese_asr) as calls:
        with pytest.raises(yc.CaptionUnavailable) as e:
            yc.fetch_lyrics_from_captions(VIDEO)
    # 후보를 여러 개 받아 봤지만 어느 것도 CJK가 아니라 전부 떨어졌다 (이 목은 모든 트랙에
    # 같은 베트남어 본문을 준다 — 자동 더빙 업로드에서 팬 번역 트랙까지 오염된 상황이다)
    assert calls["tracks"] and all(not k.endswith("-orig") for k in calls["tracks"])
    assert e.value.code == "non_cjk_caption"
    # 4xx = 이 영상은 자막으로 안 된다는 확정 판정 → 클라이언트가 붙여넣기로 안내한다
    assert e.value.http_status == 404
    assert e.value.terminal is True
    assert "가사로 쓰면" in e.value.message


def test_a_single_kana_is_enough_to_pass():
    """비율이 아니라 «하나도 없음»이 기준이다 — 진짜 CJK 곡을 잘못 버리지 않는다.

    NO_HINT_INFO를 쓴다(REAL_INFO 그대로면 asr('vi') 힌트가 이 본문("君" 하나만
    CJK)과 어긋나 기각된다 — 이 테스트가 보는 성질은 힌트 유무와 무관하다)."""
    mostly_latin = _lines("Wow oh yeah", "Come on baby", "君")
    with _gate(True), _ytdlp(NO_HINT_INFO, mostly_latin):
        found = yc.fetch_lyrics_from_captions(VIDEO)
    assert found.lines == ["Wow oh yeah", "Come on baby", "君"]


@pytest.mark.parametrize(
    "text", ["ゆらゆら", "너를 만나고", "我想見你", "ｱｲｳｴｵ", "ㅋㅋㅋ그래"]
)
def test_any_cjk_script_passes(text):
    # NO_HINT_INFO — asr 힌트('vi')가 있으면 이 본문들(ja/ko/zh)과 어긋나 기각된다
    with _gate(True), _ytdlp(NO_HINT_INFO, _lines(text, "second", "third")):
        assert yc.fetch_lyrics_from_captions(VIDEO).lines[0] == text


def test_romaji_glossed_japanese_captions_pass():
    """이 곡 자막의 실제 형태 — 로마자를 괄호로 병기한다. 라틴이 더 많아도 통과해야 한다."""
    glossed = _lines(
        "(Furetemitai himitsu to) 触れてみたい秘密と",
        "(Aimai na anata no koto) 曖昧なあなたのこと",
        "(Shiyou yo) しようよ",
    )
    with _gate(True), _ytdlp(NO_HINT_INFO, glossed):
        found = yc.fetch_lyrics_from_captions(VIDEO)
    assert len(found.lines) == 3
    counts = yc.caption_script_counts(found.lines)
    assert counts["latin"] > counts["kana"] + counts["han"], "라틴이 더 많은 구성이어야 한다"


def test_gate_can_be_switched_off():
    latin = _lines("Anh muốn chạm vào", "Mơ hồ về em", "Không có gì")
    with _gate(False), _ytdlp(REAL_INFO, latin):
        found = yc.fetch_lyrics_from_captions(VIDEO)
    assert len(found.lines) == 3, "설정으로 껐는데도 게이트가 동작했다"


def test_gate_stays_on_when_the_setting_cannot_be_read(monkeypatch):
    """설정 로드 실패는 게이트를 끄는 사유가 아니다 — 오염이 더 큰 손해다."""
    import everyric2.config.settings as settings_mod

    def boom():
        raise RuntimeError("no settings")

    monkeypatch.setattr(settings_mod, "get_settings", boom)
    assert yc._require_cjk_enabled() is True


# --------------------------------------------------------------------------
# 3) 문자 구성 판정 자체 (앵커 경로와 공유하는 유틸리티)
# --------------------------------------------------------------------------


def test_has_cjk_script():
    assert yc.has_cjk_script(script_counts("ゆらゆら"))
    assert yc.has_cjk_script(script_counts("너를"))
    assert yc.has_cjk_script(script_counts("影"))
    assert not yc.has_cjk_script(script_counts("Anh muốn chạm vào điều bí mật"))
    assert not yc.has_cjk_script(script_counts("just latin 123 !?"))
    assert not yc.has_cjk_script(script_counts(""))


def test_script_counts_ignores_whitespace_and_counts_by_family():
    c = script_counts("君は ABC 한글 漢")
    assert c == {"kana": 1, "hangul": 2, "han": 2, "latin": 3, "total": 8}


# --------------------------------------------------------------------------
# 4) 실사용 사고 재현 (2026-08-03)
# --------------------------------------------------------------------------


def test_latin_titled_song_now_finds_the_original_via_asr_hint():
    """c_9UTrrqcLI 재현: 라틴 제목뿐인 한국 곡. 예전엔 title_script_hint=None → 알파벳순
    폴백(en→ja→ko→zh)이 en 탈락 후 ja(번역 자막)를 원문으로 채택해 ko(진짜 가사)를
    받아 보지도 못했다. asr_lang_hint('ko', 실제 오디오 언어)가 이제 순서를 정한다."""
    info = {
        "title": "[MV] TAK - 'Lucky Doki' feat. xei",
        "uploader": "TAK",
        "subtitles": {k: [{"name": k}] for k in ("zh", "en", "ja", "ko")},
        "automatic_captions": {"ko-orig": [{"name": "Korean"}]},
    }
    tracks = {
        "en": ["hi"],  # 너무 짧아 too_short로 탈락
        "ja": ["これは日本語の翻訳です", "二行目です", "三行目です"],  # 번역
        "ko": ["이것이 진짜 가사예요", "둘째 줄이에요", "셋째 줄이에요"],  # 원문
        "zh": ["这是中文翻译", "第二行", "第三行"],  # 번역
    }
    order = yc.order_manual_tracks(info, None, 4)
    assert order[0] == "ko", f"asr 힌트(ko)가 ko를 첫 후보로 올려야 한다: {order}"

    downloaded: list[str] = []

    def fake_extract(video_id):
        return info

    def fake_download(video_id, lang, auto):
        downloaded.append(lang)
        return [
            {"start": i * 2.0, "end": i * 2.0 + 1.5, "text": t}
            for i, t in enumerate(tracks[lang])
        ]

    orig = (yc.extract_caption_info, yc.download_track_lines)
    yc.extract_caption_info, yc.download_track_lines = fake_extract, fake_download
    try:
        with _gate(True):
            found = yc.fetch_lyrics_from_captions(VIDEO)
    finally:
        yc.extract_caption_info, yc.download_track_lines = orig

    assert downloaded[0] == "ko", f"ko를 첫 후보로 받아 봐야 한다: {downloaded}"
    assert found.track.lang == "ko"
    assert found.lines == tracks["ko"]
    assert found.track.language == "ko"


def test_no_ja_manual_track_gives_up_instead_of_picking_a_random_translation():
    """Atvsg_zogxo 재현: 힌트(제목)는 ja인데 수동 트랙 18종이 전부 번역이고 ja 트랙이
    없다. 예전엔 전원 동순위로 알파벳순 폴백이 de(독일어) 등 엉뚱한 번역을 원문으로
    채택했다. 이제는 ja 수동 트랙이 없다는 것을 확인한 순간 확정 포기한다 — 어느 트랙도
    받아 보지 않는다."""
    langs = ["ar", "de", "en", "es", "fr", "id", "it", "ko", "ms", "pl",
             "pt", "ru", "th", "tr", "uk", "vi", "zh-Hans", "zh-Hant"]
    info = {
        "title": "ずっと真夜中でいいのに。「勘冴えて悔しいわ」",
        "uploader": "ZUTOMAYO",
        "subtitles": {lang: [{"name": lang}] for lang in langs},
        "automatic_captions": {},
    }
    with _gate(True), _ytdlp(info, _lines("Das ist die deutsche Übersetzung")) as calls:
        with pytest.raises(yc.CaptionUnavailable) as e:
            yc.fetch_lyrics_from_captions(VIDEO)
    assert calls["tracks"] == [], "ja 트랙이 없다는 것을 확인한 순간 포기해야 한다 — 아무것도 받아 보면 안 된다"
    assert e.value.code == "no_original_track"
    assert e.value.http_status == 404


def test_body_language_rejects_a_few_credit_line_cjk_chars_in_a_latin_body():
    """Atvsg_zogxo의 실제 오염 형태 — 콜론 없는 크레딧 줄(``ACAね``·``矢野達也``류)이
    ``_is_credit_line``을 통과해 독일어 본문에 섞여 남는다. 가나·한자 몇 글자만으로
    ``kana >= hangul``이 성립해 ja로 오판되던 것을, CJK 비중 하한(``_MIN_CJK_PROPORTION``)
    이 막는다. 순수 CJK 본문(라틴 오염이 전혀 없는 한자 위주 줄)은 절대 개수가 적어도
    영향받지 않는다(기존 계약 — ``kana=1, han=100`` 여전히 ja)."""
    german_body = " ".join(["Das ist ein deutscher Satz ohne CJK Schriftzeichen"] * 30)
    credit_lines = "ACAね 矢野達也 100回嘔吐"  # kana 몇 자 + han 몇 자, 콜론 없음
    counts = yc.caption_script_counts([german_body, credit_lines])
    assert counts["kana"] > 0 or counts["han"] > 0, "테스트 전제(약간의 CJK가 섞여 있어야 함)가 깨졌다"
    assert yc.body_language(counts) is None, "라틴에 파묻힌 크레딧 CJK 몇 글자로 ja/ko 판정이 나오면 안 된다"

    # 대조군 — 라틴 오염이 전혀 없으면 가나 1자뿐이어도 그대로 ja다(기존 계약 불변,
    # test_body_language_needs_han_only_for_chinese의 kana=1·han=100과 동형)
    assert yc.body_language(yc.caption_script_counts(["ぁ" + "国" * 100])) == "ja"
