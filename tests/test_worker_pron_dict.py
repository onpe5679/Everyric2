"""worker의 표기별 발음 부착(`attach_pron_variants`)과 생성 번역의 언어 분리 — 순수 함수 단위.

실오디오도 DB도 쓰지 않는다. 세그먼트는 정렬 파이프라인이 만드는 모양(text + words 글자
스팬 + pronunciation)을 손으로 합성하고, 시각은 글자당 0.5초로 깔아 단조성을 눈으로 검산할
수 있게 했다.
"""

from everyric2.server.worker import (
    attach_pron_variants,
    job_target_lang,
    translation_layer_lines,
)
from everyric2.text.pron_style import candidate_token_sets
from everyric2.text.reading import mora_segments_for_line

# 골든 스냅샷(tests/test_pron_golden.py)에 있는 줄 — 한글 독음과 romaji가 둘 다 실측값이다
NEKURA = "アルバイトはネクラモード"
NEKURA_HANGUL = "아루바이토와 네쿠라 모오도"
NEKURA_ROMAJI = "arubaito wa nekura moodo"

# 애매 낱말 刃(は/やいば) — 심판이 뒤집으면 모라 수가 4에서 6으로 늘어난다
YAIBA = "刃を研ぐ"
YAIBA_DEFAULT_HANGUL = "하오 토구"
YAIBA_CHOSEN_HANGUL = "야이바오 토구"


def _words(text: str, step: float = 0.5) -> list[dict]:
    """비공백 글자별 (start, end) 스팬 — 정렬 word_segments(공백을 만들지 않는 CTC 토큰)와
    같은 모양이다. ``_full_coverage_words``가 직렬화에서 실제로 만드는 ``seg["words"]``는
    이것과 **다르다**(라인 전체 글자를 공백까지 포함해 1:1로 덮는다) — 그 모양은
    ``_full_words``가 낸다."""
    return [
        {"word": ch, "start": i * step, "end": (i + 1) * step}
        for i, ch in enumerate(text)
        if not ch.isspace()
    ]


def _full_words(text: str, step: float = 0.5) -> list[dict]:
    """글자별(공백 포함) (start, end) 스팬 — ``_full_coverage_words``가 직렬화에서 실제로
    만드는 ``seg["words"]``와 같은 모양(실측: "옛날 머나먼 그 어느 마을엔" → words 15개,
    그중 공백 4개). ko 분기의 ``_ko_char_time``이 공백 항목을 걸러내지 않으면 원문의
    비공백 글자와 개수가 어긋나 kana/romaji 시각이 전멸했던 실사용 버그(N_vYUNEktsA)의
    재현 픽스처다."""
    return [{"word": ch, "start": i * step, "end": (i + 1) * step} for i, ch in enumerate(text)]


def _seg(text: str, pronunciation: str, *, words: bool = True) -> dict:
    seg: dict = {"text": text, "start": 0.0, "end": len(text) * 0.5, "pronunciation": pronunciation}
    if words:
        seg["words"] = _words(text)
    return seg


def _rebuild(segments: list[dict]) -> str:
    """모라 스팬을 표시 문자열로 되돌린다 — «표시=세그» 단일 소스 불변식 검산용."""
    return "".join(s["text"] + (" " if s.get("space") else "") for s in segments).strip()


def _chosen_tokens(text: str, chosen: str) -> list:
    rendered, token_sets = candidate_token_sets(text)
    return token_sets[rendered.index(chosen)]


def test_attaches_hangul_and_romaji():
    seg = _seg(NEKURA, NEKURA_HANGUL)
    attach_pron_variants(seg)

    assert seg["pron"]["hangul"] == seg["pronunciation"] == NEKURA_HANGUL
    assert seg["pron"]["romaji"] == NEKURA_ROMAJI


def test_romaji_segments_are_monotonic_and_rebuild_the_display():
    seg = _seg(NEKURA, NEKURA_HANGUL)
    attach_pron_variants(seg)

    segments = seg["pron_segs"]["romaji"]
    assert len(segments) == 12  # 모라 수
    assert _rebuild(segments) == seg["pron"]["romaji"]
    for prev, cur in zip(segments, segments[1:]):
        assert cur["start"] >= prev["end"]
        assert cur["end"] >= cur["start"]
    # 라인 구간을 벗어나지 않는다 (글자 스팬에서 파생됐으므로)
    assert segments[0]["start"] >= 0.0
    assert segments[-1]["end"] <= seg["end"]


# ---------------------------------------------------------------------------
# 감사 2차(O1 실측 4케이스) — 혼합 줄(ja/ko/라틴이 섞인 줄)의 분기 판정(M2)·소실
# 방지(M1)·라틴 환전+공백(M3)·kana 표시=세그 단일 소스(M4)를 한 번에 검증한다.
# ---------------------------------------------------------------------------


def test_mixed_line_branch_decision_by_character_majority():
    # M2: 문자 수 우세로 분기한다("일본어 글자가 하나라도 있으면 ja"였던 예전 규칙은
    # 한글이 더 많은 혼합 줄까지 ja로 새게 했다). 감사 2차 R2로 ko 분기도 혼합 줄이면
    # hangul 키를 갖게 됐으니(ja 런만 독음 환전) 키 유무로는 더 이상 분기를 못 가른다
    # — 대신 romaji 렌더 스타일로 가른다: ja 분기는 한글 구간을 가나 경유(카타카나
    # 근사, M1)로 읽어 "choahe"류가 나오고, ko 분기는 RR을 직접 써 "saranghae"가
    # 정확히 나온다(R1).
    tie_favors_ja = _seg("좋아해 きみが", "", words=True)  # 한글3=일본어3 → 동률은 ja
    tie_favors_ja["words"] = _full_words("좋아해 きみが")
    attach_pron_variants(tie_favors_ja)
    assert tie_favors_ja["pron"]["romaji"] == "choahe kimi ga"  # ja 분기의 가나 경유 근사

    ko_majority = _seg("사랑해 デス", "", words=True)  # 한글3 > 일본어2 → ko
    ko_majority["words"] = _full_words("사랑해 デス")
    attach_pron_variants(ko_majority)
    assert ko_majority["pron"]["romaji"] == "saranghae desu"  # ko 분기의 정확한 RR
    assert ko_majority["pron"]["hangul"] == "사랑해 데스"  # R2: ja 런만 독음 환전

    ko_no_ja = _seg("사랑해 baby", "", words=True)  # 일본어 0 → ko(원래도 ko였다)
    ko_no_ja["words"] = _full_words("사랑해 baby")
    attach_pron_variants(ko_no_ja)
    assert "hangul" not in ko_no_ja["pron"]  # ja 글자가 없으니 R2도 발동 안 함(순한글 취급)


def test_mixed_line_no_deletion_and_display_equals_segments():
    # M1(로마자 삭제 금지)·M3(라틴 환전+공백)·M4(kana 단일 소스)를 한 번에: 모든
    # 표기(pron 값)와 그 세그를 재구성한 문자열이 완전히 같아야 한다(소실 0).
    fixtures = ["좋아해 きみが", "사랑해 デス", "사랑해 baby", NEKURA]
    for text in fixtures:
        seg = {"text": text, "start": 0.0, "end": len(text) * 0.5, "words": _full_words(text)}
        attach_pron_variants(seg)
        assert seg.get("pron"), text
        for script, display in seg["pron"].items():
            segments = (seg.get("pron_segs") or {}).get(script)
            if segments is None:
                continue  # 표시만 있고 세그가 없는 표기(예: 라틴 곡)는 이 라운드 범위 밖
            assert _rebuild(segments) == display, (text, script)


def test_mixed_ja_line_hangul_substring_is_not_deleted_from_romaji_and_kana():
    # M1 정공법 검증: 「좋아해 きみが」의 「좋아해」가 romaji/kana에서 통째로 사라지던
    # 것(text_to_moras가 한글을 모라로 못 만들어서)이 이제 살아 있어야 한다.
    text = "좋아해 きみが"
    seg = {"text": text, "start": 0.0, "end": len(text) * 0.5, "words": _full_words(text)}
    attach_pron_variants(seg)

    assert seg["pron"]["hangul"] == "좋아해 키미가"
    assert seg["pron"]["romaji"] == "choahe kimi ga"
    assert seg["pron"]["kana"] == "チョアヘ キミ ガ"
    assert seg["pron"]["romaji"].startswith("choahe")  # 삭제됐다면 "kimi ga"로 시작했을 것
    assert seg["pron"]["kana"].startswith("チョアヘ")


def test_mixed_ko_line_latin_run_is_transliterated_and_space_preserved():
    # M3: ko 분기의 라틴 런이 latin_to_kana로 환전되고(«サランヘ baby» 사고 재현 방지),
    # 세그를 재구성하면 원문 공백이 그대로 살아 있어야 한다.
    text = "사랑해 baby"
    seg = {"text": text, "start": 0.0, "end": len(text) * 0.5, "words": _full_words(text)}
    attach_pron_variants(seg)

    assert seg["pron"]["kana"] == "サランヘ ペビ"  # baby가 raw 라틴으로 안 남는다
    kana_segments = seg["pron_segs"]["kana"]
    assert kana_segments[-1]["text"] == "ペビ"  # 낱말 하나로 묶인 모라
    assert any(s.get("space") for s in kana_segments)  # 공백 플래그가 살아 있다
    assert _rebuild(kana_segments) == seg["pron"]["kana"]

    romaji_segments = seg["pron_segs"]["romaji"]
    assert _rebuild(romaji_segments) == seg["pron"]["romaji"] == "saranghae baby"


def test_mixed_ko_line_kana_run_is_transliterated_to_romaji_and_hangul():
    # 감사 2차 R1·R2(배포 전 마지막 관문): O1 실측 케이스 — «사랑해 デス»가 ko 분기로
    # 온다(한글3>일본어2, M2). R1: romaji의 デス가 raw로 새지 않고 "desu"로 환전된다
    # (M1의 정확한 역방향 — kana_to_romaji). R2: ko 곡은 원래 hangul 키를 안 만들지만
    # 혼합 줄은 예외 — ja 런만 독음으로 환전한 hangul 표시가 붙는다.
    text = "사랑해 デス"
    seg = {"text": text, "start": 0.0, "end": len(text) * 0.5, "words": _full_words(text)}
    attach_pron_variants(seg)

    assert seg["pron"]["romaji"] == "saranghae desu"
    assert seg["pron"]["hangul"] == "사랑해 데스"
    assert seg["pron"]["kana"] == "サランヘ デス"

    # 세그 정합(표시=재구성) — romaji 세그도 "デス" raw가 아니라 "desu"로 묶여야 한다
    romaji_segments = seg["pron_segs"]["romaji"]
    assert _rebuild(romaji_segments) == "saranghae desu"
    assert romaji_segments[-1]["text"] == "desu"  # 가나 런 하나가 토큰 하나로 묶인다


def test_pure_hangul_ko_line_still_has_no_hangul_key():
    # R2는 혼합 줄(ja 글자가 있는 줄)에만 발동한다 — 순한글 줄에 원문과 똑같은 값을
    # 또 저장하는 낭비를 만들면 안 된다(기존 설계 유지).
    seg = _seg("사랑해", "", words=True)
    attach_pron_variants(seg)

    assert "hangul" not in seg["pron"]
    assert seg["pron"]["kana"] == "サランヘ"
    assert seg["pron"]["romaji"] == "saranghae"


def test_ja_segment_gets_kana_display_and_shares_timing_with_romaji():
    # 사용자 버그 보고: ja 세그의 pron dict에 kana가 없어 script=kana 설정에서 발음 줄이
    # 통째로 사라졌다. 감사 2차 M4로 표시는 세그와 같은 카타카나 모라 열에서 합성된다
    # (romaji와 같은 모라/토큰 경계 띄어쓰기 — 문절 띄어쓰기이던 예전 값과 다르다).
    seg = _seg(NEKURA, NEKURA_HANGUL)
    attach_pron_variants(seg)

    assert seg["pron"]["kana"] == "アルバイト ワ ネクラ モード"
    kana_segments = seg["pron_segs"]["kana"]
    romaji_segments = seg["pron_segs"]["romaji"]
    assert len(kana_segments) == len(romaji_segments) == 12
    assert [s["text"] for s in kana_segments] == [
        "ア", "ル", "バ", "イ", "ト", "ワ", "ネ", "ク", "ラ", "モ", "ー", "ド",
    ]
    # 같은 모라 목록에서 나온 시각이라 start/end/space가 romaji와 글자 하나 안 어긋난다
    for k, r in zip(kana_segments, romaji_segments):
        assert k["start"] == r["start"]
        assert k["end"] == r["end"]
        assert k.get("space") == r.get("space")


def test_ja_referee_switch_kana_follows_the_winning_reading():
    # 심판이 刃를 やいば(6모라)로 뒤집은 줄 — kana도 は(4모라, 하오 토구)가 아니라
    # 이긴 읽기(야이바오 토구)를 따라야 한다. text_to_moras(text, tokens=referee_tokens)가
    # romaji·kana 둘의 재료라 심판 판정이 저절로 반영된다.
    seg = _seg(YAIBA, YAIBA_CHOSEN_HANGUL)
    attach_pron_variants(seg, referee_tokens=_chosen_tokens(YAIBA, YAIBA_CHOSEN_HANGUL))

    assert seg["pron"]["romaji"] == "yaiba o togu"
    kana_segments = seg["pron_segs"]["kana"]
    romaji_segments = seg["pron_segs"]["romaji"]
    assert len(kana_segments) == len(romaji_segments) == 6
    assert [s["text"] for s in kana_segments] == ["ヤ", "イ", "バ", "オ", "ト", "グ"]
    for k, r in zip(kana_segments, romaji_segments):
        assert k["start"] == r["start"]
        assert k["end"] == r["end"]


def test_legacy_hangul_fields_are_untouched():
    seg = _seg(NEKURA, NEKURA_HANGUL)
    seg["pron_segments"] = [{"text": "아", "start": 0.0, "end": 0.5, "resolved": True}]
    attach_pron_variants(seg)

    assert seg["pronunciation"] == NEKURA_HANGUL
    assert seg["pron_segments"] == [{"text": "아", "start": 0.0, "end": 0.5, "resolved": True}]


def test_referee_tokens_switch_the_reading():
    default_seg = _seg(YAIBA, YAIBA_DEFAULT_HANGUL)
    attach_pron_variants(default_seg)
    assert default_seg["pron"]["romaji"] == "ha o togu"
    assert len(default_seg["pron_segs"]["romaji"]) == 4

    # 심판이 やいば를 골랐다 — romaji도 그 읽기를 따라야 한다(모라 4 → 6)
    chosen_seg = _seg(YAIBA, YAIBA_CHOSEN_HANGUL)
    attach_pron_variants(chosen_seg, referee_tokens=_chosen_tokens(YAIBA, YAIBA_CHOSEN_HANGUL))

    assert chosen_seg["pron"]["hangul"] == YAIBA_CHOSEN_HANGUL
    assert chosen_seg["pron"]["romaji"] == "yaiba o togu"
    segments = chosen_seg["pron_segs"]["romaji"]
    assert len(segments) == 6
    assert _rebuild(segments) == "yaiba o togu"


def test_referee_switched_segment_gets_no_romaji_without_tokens():
    # 심판이 바꾼 줄인데 이긴 읽기의 토큰 열이 없다(캐시 재사용·늦은 병합 경로).
    # 기본 읽기로 렌더하면 화면의 한글 독음과 다른 낱말이 찍히므로 표기를 붙이지 않는다.
    seg = _seg(YAIBA, YAIBA_CHOSEN_HANGUL)
    seg["debug"] = {"referee": {"default": YAIBA_DEFAULT_HANGUL, "chosen": YAIBA_CHOSEN_HANGUL}}
    attach_pron_variants(seg)

    assert seg["pron"] == {"hangul": YAIBA_CHOSEN_HANGUL}
    assert "pron_segs" not in seg


def test_referee_untouched_segment_still_gets_romaji():
    # 심판이 돌긴 했지만 기본값을 그대로 유지한 줄은 기본 읽기가 곧 정답이다
    seg = _seg(YAIBA, YAIBA_DEFAULT_HANGUL)
    seg["debug"] = {"referee": {"default": YAIBA_DEFAULT_HANGUL, "chosen": YAIBA_DEFAULT_HANGUL}}
    attach_pron_variants(seg)

    assert seg["pron"]["romaji"] == "ha o togu"


def test_is_idempotent():
    seg = _seg(YAIBA, YAIBA_CHOSEN_HANGUL)
    attach_pron_variants(seg, referee_tokens=_chosen_tokens(YAIBA, YAIBA_CHOSEN_HANGUL))
    before = {"pron": dict(seg["pron"]), "pron_segs": {k: list(v) for k, v in seg["pron_segs"].items()}}

    # 두 번째 호출은 다른 읽기를 들고 와도 이미 붙은 값을 덮지 않는다
    attach_pron_variants(seg)

    assert seg["pron"] == before["pron"]
    assert seg["pron_segs"] == before["pron_segs"]


def test_display_survives_when_timing_is_unavailable():
    # 글자 스팬이 없으면(라인 타이밍만 있는 줄) 표기 문자열만 남고 확장이 그라데이션으로 폴백한다
    seg = _seg(NEKURA, NEKURA_HANGUL, words=False)
    attach_pron_variants(seg)

    assert seg["pron"]["romaji"] == NEKURA_ROMAJI
    assert "pron_segs" not in seg


def test_display_survives_when_char_spans_do_not_match_the_text():
    # words가 이 줄의 글자가 아니면 시각 환산이 성립하지 않는다 — 표시만 남긴다
    seg = _seg(NEKURA, NEKURA_HANGUL)
    seg["words"] = _words("전혀 다른 줄")
    attach_pron_variants(seg)

    assert seg["pron"]["romaji"] == NEKURA_ROMAJI
    assert "pron_segs" not in seg


def test_ja_segment_self_generates_hangul_pronunciation_when_missing():
    # 감사 2차 E4: 비ko 사용자의 생성 요청은 line_meta에 한글 발음이 없다(번역 API가
    # 그 사용자 언어로 번역만 만든다) — 예전엔 pron dict가 통째로 안 생겼다. 이제는
    # wiki_pronunciation(text)로 서버가 직접 만들어 legacy 슬롯과 pron.hangul에 싣고,
    # romaji·kana도 평소처럼 파생된다.
    no_pron = _seg(NEKURA, "")
    attach_pron_variants(no_pron)

    assert no_pron["pronunciation"] == NEKURA_HANGUL  # legacy 슬롯도 채워진다
    assert no_pron["pron"]["hangul"] == NEKURA_HANGUL
    assert no_pron["pron"]["romaji"] == NEKURA_ROMAJI
    assert "kana" in no_pron["pron"]


def test_skips_segment_without_ja_ko_or_latin_text():
    # 숫자·기호뿐인 줄은 세 분기(ja/ko/라틴) 어디에도 안 걸린다.
    symbols_only = {"text": "…！", "start": 0.0, "end": 0.5}
    attach_pron_variants(symbols_only)
    assert "pron" not in symbols_only


def test_ko_segment_gets_kana_and_romaja():
    # ko 곡 세그는 ``pronunciation`` 필드가 없어도(원문 한글 자체가 독음) kana/romaji가 붙는다.
    seg = _seg("사랑해", "", words=True)
    attach_pron_variants(seg)

    assert seg["pron"] == {"kana": "サランヘ", "romaji": "saranghae"}
    assert "hangul" not in seg["pron"]  # 원문이 이미 표시라 hangul 키는 만들지 않는다


def test_ko_segment_kana_segs_are_monotonic_and_bisect_the_coda():
    seg = _seg("사랑해", "", words=True)
    attach_pron_variants(seg)

    segments = seg["pron_segs"]["kana"]
    # 사(1모라) + 랑(받침 ㅇ→independent ン, 2모라) + 해(1모라) = 4모라
    assert [s["text"] for s in segments] == ["サ", "ラ", "ン", "ヘ"]
    assert "".join(s["text"] for s in segments) == seg["pron"]["kana"]
    for prev, cur in zip(segments, segments[1:]):
        assert cur["start"] >= prev["end"]
        assert cur["end"] >= cur["start"]
    # 받침 이등분: 랑(글자 스팬 0.5~1.0)의 두 모라(ラ/ン)가 그 구간을 균등 분할한다
    lang_span = _words("사랑해")[1]
    ra, n = segments[1], segments[2]
    assert ra["start"] == lang_span["start"]
    assert ra["end"] == n["start"] == (lang_span["start"] + lang_span["end"]) / 2
    assert n["end"] == lang_span["end"]


def test_ko_segment_romaja_segs_are_monotonic_and_rebuild_the_display():
    seg = _seg("사랑해", "", words=True)
    attach_pron_variants(seg)

    segments = seg["pron_segs"]["romaji"]
    # 한 글자 = 로마자 한 덩이(받침이 갈라지지 않는다) — kana처럼 이등분이 없다
    assert [s["text"] for s in segments] == ["sa", "rang", "hae"]
    assert "".join(s["text"] for s in segments) == seg["pron"]["romaji"]
    for prev, cur in zip(segments, segments[1:]):
        assert cur["start"] >= prev["end"]
        assert cur["end"] >= cur["start"]
    # 글자 스팬을 그대로 옮겨 붙인다 — 랑의 스팬과 정확히 같아야 한다(균등분할 없음)
    lang_span = _words("사랑해")[1]
    assert segments[1]["start"] == lang_span["start"]
    assert segments[1]["end"] == lang_span["end"]


def test_ko_segment_kana_and_romaja_segs_survive_words_with_blank_entries():
    # 실사용 버그 재현: _full_coverage_words가 만드는 words는 공백도 항목으로 포함한다.
    # 필터링 없이 원문 비공백 글자와 zip하면 전 줄에서 개수 불일치 → kana/romaji segs 전멸.
    from everyric2.text.ko_reading import hangul_line_moras, hangul_line_romaja_syllables

    text = "사랑해 진짜"
    seg = _seg(text, "", words=False)
    seg["words"] = _full_words(text)  # 공백 포함 — 실제 _full_coverage_words 모양
    attach_pron_variants(seg)

    kana_segments = seg["pron_segs"]["kana"]
    romaja_segments = seg["pron_segs"]["romaji"]

    assert len(kana_segments) == len(hangul_line_moras(text)) > 0
    assert len(romaja_segments) == len(hangul_line_romaja_syllables(text)) > 0

    for segments in (kana_segments, romaja_segments):
        for prev, cur in zip(segments, segments[1:]):
            assert cur["start"] >= prev["end"]
            assert cur["end"] >= cur["start"]


def test_ko_segment_display_survives_when_timing_is_unavailable():
    seg = _seg("좋아해 그대를", "", words=False)
    attach_pron_variants(seg)

    assert seg["pron"]["kana"] and seg["pron"]["romaji"]
    assert "pron_segs" not in seg


def test_latin_segment_gets_all_four_display_scripts():
    # 결함 수정(2026-08-03, 운영자 지시): 라틴 곡도 표기 4종(hangul/kana/romaji/en)을
    # 전부 받는다 — 예전엔 가나 근사 하나뿐이라 2패스가 안 닿은 en 곡(고속 라우팅으로
    # 끝난 곡 등)의 한국어 사용자가 기본 표기(hangul)를 아예 못 받았다. CTC 정렬이 라틴
    # 위에서 약해서(latin_hangul 모듈 실측) pron_segs(타이밍)는 여전히 안 만든다 — 표기
    # 문자열만 결정론 근사다.
    #
    # romaji==en(2026-08-03 추가 수정): en 곡의 romaji 정답은 원문 철자다 — 이전엔
    # "teiku ito iiずぃ"처럼 가나 음차를 다시 로마자로 되돌린 근사가 나갔다(za wezaa
    # poreketusu류 오염). en 곡에서는 romaji가 en과 같아진다.
    seg = _seg("Take it easy", "", words=True)
    attach_pron_variants(seg)

    assert seg["pron"] == {
        "hangul": "테익 잇 이지",
        "kana": "テイク イト イーズィー",
        "romaji": "Take it easy",
        "en": "Take it easy",
        # ipa는 정렬 타깃 자체(IPA 표시 옵션, 2026-08-03) — 파생이 아니라 타깃 문자열
        "ipa": "teik it izi",
    }
    assert "pron_segs" not in seg


def test_latin_segment_display_has_no_doubled_word_gaps():
    # derive_en_display_units가 낱말 사이 원문 공백도 owners에 그 글자 그대로 얹으므로
    # (align_target 모듈의 "낱말 사이 공백·구두점" 패스스루), word_end 플래그가 넣는
    # 공백과 겹쳐 두 칸으로 벌어지지 않아야 한다(_join_display_units 회귀 방지).
    seg = _seg("Take it easy", "", words=True)
    attach_pron_variants(seg)

    for script, display in seg["pron"].items():
        assert "  " not in display, (script, display)


def test_mora_segments_follow_the_given_tokens():
    # attach가 기대는 계약: 같은 글자 스팬이라도 토큰 열을 주면 모라 수가 그 읽기를 따른다
    char_spans = [(w["word"], w["start"], w["end"]) for w in _words(YAIBA)]

    assert len(mora_segments_for_line(char_spans, YAIBA)) == 4
    chosen = _chosen_tokens(YAIBA, YAIBA_CHOSEN_HANGUL)
    assert len(mora_segments_for_line(char_spans, YAIBA, tokens=chosen)) == 6


def test_mora_segments_return_none_without_spans():
    assert mora_segments_for_line([], YAIBA) is None
    assert mora_segments_for_line([("刃", 0.0, 0.5)], "   ") is None


def test_referee_token_set_finds_the_winning_reading():
    from everyric2.server.worker import _referee_token_set

    tokens = _referee_token_set(YAIBA, YAIBA_CHOSEN_HANGUL)
    seg = _seg(YAIBA, YAIBA_CHOSEN_HANGUL)
    attach_pron_variants(seg, referee_tokens=tokens)
    assert seg["pron"]["romaji"] == "yaiba o togu"

    # 후보에 없는 문자열(사람이 손으로 쓴 발음 등)은 None — 기본 읽기로 조용히 떨어진다
    assert _referee_token_set(YAIBA, "엉뚱한 독음") is None


class _FakeEngine:
    """심판이 이긴 후보를 text로 돌려주는 엔진 대역 (tests/test_pron_candidates.py와 같은 모양).

    여기서 보는 것은 채점이 아니라 배선이다: 이긴 읽기의 토큰 열이 pron_data를 거쳐
    직렬화의 attach_pron_variants까지 흘러가는가.
    """

    def __init__(self, winner: str):
        self.winner = winner
        self._last_referee: list[dict] = []
        self._last_heard: dict = {}
        self._last_heard_spans: dict = {}

    def align(self, audio, lyrics, language=None, **kwargs):
        from everyric2.inference.prompt import SyncResult, WordSegment

        syllables = [ch for ch in self.winner if ch != " "]
        step = 0.2
        self._last_referee = [
            {
                "line": 0,
                "default": YAIBA_DEFAULT_HANGUL,
                "chosen": self.winner,
                "margin": 0.15,
                "gain": 0.42,
                "frames": 60,
                "scores": [[YAIBA_DEFAULT_HANGUL, -3.1], [self.winner, -2.68]],
            }
        ]
        return [
            SyncResult(
                line_number=lyrics[0].line_number,
                text=self.winner,
                start_time=0.0,
                end_time=step * len(syllables),
                word_segments=[
                    WordSegment(word=ch, start=step * k, end=step * (k + 1), confidence=0.5)
                    for k, ch in enumerate(syllables)
                ],
            )
        ]


def test_referee_reading_reaches_the_serialized_segment():
    from everyric2.config.settings import AlignmentSettings
    from everyric2.inference.prompt import LyricLine
    from everyric2.server.worker import _align_with_pronunciation, _pron_by_text

    engine = _FakeEngine(YAIBA_CHOSEN_HANGUL)
    lines = [LyricLine(text=YAIBA, line_number=1)]
    by_text = _pron_by_text([{"text": YAIBA, "pronunciation": YAIBA_DEFAULT_HANGUL}])
    results, pron_data = _align_with_pronunciation(
        engine, object(), lines, by_text, AlignmentSettings(pron_referee=True)
    )

    pd = pron_data[0]
    assert pd["pronunciation"] == YAIBA_CHOSEN_HANGUL
    assert pd["tokens"] is not None  # 이긴 읽기의 토큰 열이 실려 나왔다

    # 직렬화 루프가 하는 일 그대로 — 세그를 세우고 그 토큰 열로 표기를 얹는다
    seg = {
        "text": results[0].text,
        "start": results[0].start_time,
        "end": results[0].end_time,
        "pronunciation": pd["pronunciation"],
        "words": [
            {"word": w.word, "start": w.start, "end": w.end} for w in results[0].word_segments
        ],
        "debug": {"referee": pd["referee"]},
    }
    attach_pron_variants(seg, referee_tokens=pd["tokens"])

    assert seg["pron"]["hangul"] == YAIBA_CHOSEN_HANGUL
    assert seg["pron"]["romaji"] == "yaiba o togu"
    assert _rebuild(seg["pron_segs"]["romaji"]) == "yaiba o togu"


def test_translation_layer_lines_keeps_only_translated_pairs():
    items = [
        {"text": "アルバイトは", "translation": "아르바이트는"},
        {"text": "  ", "translation": "공백 줄"},
        {"text": "간주", "translation": "   "},
        {"text": "ネクラモード", "translation": "네쿠라 모드"},
    ]
    assert translation_layer_lines(items) == [
        {"text": "アルバイトは", "translation": "아르바이트는"},
        {"text": "ネクラモード", "translation": "네쿠라 모드"},
    ]
    assert translation_layer_lines(None) == []


def test_job_target_lang_defaults_to_ko():
    class _Job:
        def __init__(self, target_lang=None):
            if target_lang is not None:
                self.target_lang = target_lang

    assert job_target_lang(_Job()) == "ko"  # 컬럼이 없던 시절의 잡 행
    assert job_target_lang(_Job("")) == "ko"
    assert job_target_lang(_Job(" en ")) == "en"


# ---------------------------------------------------------------------------
# ja 채택 곡의 hangul 음절 카라오케 파생 (실측 xvH0hNzMjhg — 독음 정렬이 저신뢰로
# 밀려 ja 원문 정렬이 채택되면 pron_segs가 romaji·kana뿐이었다)
# ---------------------------------------------------------------------------


def test_ja_hangul_segments_derive_from_kana_moras():
    seg = _seg(NEKURA, NEKURA_HANGUL)
    attach_pron_variants(seg)

    segs = seg["pron_segs"]["hangul"]
    kana = seg["pron_segs"]["kana"]
    # 음절 텍스트를 이으면 표기와 같다 (공백 위치는 kana 모라 공백 규칙을 따르므로 제외)
    assert "".join(s["text"] for s in segs) == NEKURA_HANGUL.replace(" ", "")
    # 시간축은 kana 모라 세그의 첫/끝과 일치하고 단조롭다
    assert segs[0]["start"] == kana[0]["start"]
    assert segs[-1]["end"] == kana[-1]["end"]
    for prev, cur in zip(segs, segs[1:]):
        assert cur["start"] >= prev["end"]


def test_ja_hangul_segments_merge_final_consonant_moras():
    # 받침으로 실현되는 모라(칸→カ+ン)는 그 음절 하나의 스팬으로 병합된다
    seg = _seg("乾杯", "칸파이")
    attach_pron_variants(seg)

    kana = seg["pron_segs"]["kana"]
    segs = seg["pron_segs"]["hangul"]
    assert [s["text"] for s in segs] == ["칸", "파", "이"]
    assert segs[0]["start"] == kana[0]["start"]
    assert segs[0]["end"] == kana[1]["end"]


def test_ja_hangul_segments_mora_mismatch_now_yields_flagged_approximation():
    # 계약 변경(2026-08-03): 모라 수 불일치를 통째로 포기하면 하필 기본 표기(hangul)만
    # 카라오케 타이밍이 죽는다(실측 6_toWwEFXyA 세그 2·3). kana 시간축 비례 근사를
    # resolved: False로 정직하게 내린다 — "틀린 카라오케보다 없는 쪽"에서 "근사임을
    # 표시한 카라오케"로.
    seg = _seg(NEKURA, "아루")
    attach_pron_variants(seg)

    hangul_segs = (seg.get("pron_segs") or {}).get("hangul")
    assert hangul_segs, "근사 폴백이 hangul 세그를 내야 한다"
    assert all(s.get("resolved") is False for s in hangul_segs)


def test_ja_hangul_segments_not_derived_over_alignment_output():
    # 독음 정렬의 실측 스팬(legacy pron_segments)이 있으면 파생하지 않는다 —
    # 확장 표시 우선순위상 파생본이 실측을 가리기 때문
    seg = _seg(NEKURA, NEKURA_HANGUL)
    seg["pron_segments"] = [{"text": "아", "start": 0.0, "end": 0.5}]
    attach_pron_variants(seg)

    assert "hangul" not in (seg.get("pron_segs") or {})


# ---------------------------------------------------------------------------
# zh 곡 게이트 — 순한자 라인은 곡 언어로만 ja와 갈린다 (2026-08-03)
# ---------------------------------------------------------------------------


def test_zh_song_gate_attaches_chinese_readings():
    # 순한자 라인은 문자만으로 ja와 구별할 수 없다(한자는 두 언어 공용) — 곡 언어(zh)로
    # 게이트한다. 게이트가 없으면 중국어 가사에 일본어 한자 독음이 붙는다(오표기).
    seg = _seg("月亮代表我的心", "", words=False)
    attach_pron_variants(seg, language="zh")
    assert set(seg["pron"]) >= {"hangul", "kana", "romaji"}
    # 병음 성조 문자가 실렸다는 것 자체가 zh 분기의 증거다 — ja 분기는 병음을 못 만든다
    assert "yuè" in seg["pron"]["romaji"]


def test_zh_gate_leaves_kana_mixed_lines_to_ja():
    # zh 곡이어도 가나가 섞인 라인(일본어 인용 등)은 ja 파생이 맞다
    seg = _seg("愛してる", "", words=False)
    attach_pron_variants(seg, language="zh")
    assert seg["pron"]["kana"]
    assert "ì" not in seg["pron"]["romaji"]  # 헵번 로마자 — 병음 성조가 아니다


def test_no_language_keeps_existing_ja_behavior_for_pure_han():
    # language를 모르는 호출부(캐시 병합 등)는 기존 동작 그대로 — 게이트 미작동
    gated = _seg("月亮代表我的心", "", words=False)
    attach_pron_variants(gated, language="zh")
    ungated = _seg("月亮代表我的心", "", words=False)
    attach_pron_variants(ungated)
    assert ungated.get("pron") != gated.get("pron")


# ---------------------------------------------------------------------------
# F3 — zh 병음 음절 공백 (2026-08-04 감사)
#
# align_target.join_display(owners를 공백 없이 붙이는 범용 조립기)를 쓰면 병음 음절
# 사이 공백이 사라져 "wǒbùxiǎngshuōzàijiàn"처럼 못 읽는 문자열이 된다.
# ---------------------------------------------------------------------------


def test_zh_pron_romaji_keeps_spaces_between_pinyin_syllables():
    from everyric2.text.zh_reading import zh_to_pinyin

    text = "月亮代表我的心"
    seg = _seg(text, "", words=False)
    attach_pron_variants(seg, language="zh")
    # zh_reading.zh_pron_variants(text)와 정확히 같은 값이어야 한다 — join_display를 쓰면
    # 공백이 사라져 이 등식이 깨진다(F3 결함의 회귀 표지).
    assert seg["pron"]["romaji"] == zh_to_pinyin(text)
    assert seg["pron"]["romaji"].count(" ") == len(text) - 1, seg["pron"]["romaji"]


# ---------------------------------------------------------------------------
# 구세대 kana 단독 근사 보완 — 멱등 가드는 동결이 아니라 보존이다 (2026-08-03)
# ---------------------------------------------------------------------------


def test_legacy_kana_only_latin_pron_is_augmented_not_frozen():
    # 구세대 라틴 곡은 옛 경로가 kana 1형만 저장했다 — 표시값 E2E 실측(weathergirl):
    # 이 모양이 완결로 취급돼 한국어 사용자가 hangul 표기를 영영 못 받았다.
    seg = _seg("Take it easy", "", words=False)
    seg["pron"] = {"kana": "テイクイットイージー"}
    attach_pron_variants(seg)
    # 기존 키는 덮지 않는다 (저장된 값이 이긴다)
    assert seg["pron"]["kana"] == "テイクイットイージー"
    # 빠진 표기가 전부 보완된다
    for key in ("hangul", "romaji", "en", "ipa"):
        assert seg["pron"].get(key), key


def test_complete_pron_dict_is_still_frozen():
    # 멱등 가드의 본래 목적(직렬화·심판 판정 반영값 보존)은 그대로다 — kana 단독이
    # 아닌 pron은 어떤 키도 추가·변경되지 않는다.
    seg = _seg(NEKURA, NEKURA_HANGUL)
    custom = {"hangul": "커스텀", "kana": "カスタム", "romaji": "custom"}
    seg["pron"] = dict(custom)
    attach_pron_variants(seg)
    assert seg["pron"] == custom


def test_ja_text_with_kana_only_pron_is_not_latin_augmented():
    # kana 단독이라도 원문이 일본어면 옛 라틴 근사가 아니다 — 라틴 파생을 덧대면
    # 엉뚱한 표기가 생기므로 그대로 둔다.
    seg = _seg(NEKURA, "", words=False)
    seg["pron"] = {"kana": "アルバイトハネクラモード"}
    attach_pron_variants(seg)
    assert seg["pron"] == {"kana": "アルバイトハネクラモード"}


# ---------------------------------------------------------------------------
# en 곡 romaji 오염 — "영어→가타카나 음차→로마자 재변환" 근사 제거 (2026-08-03)
# ---------------------------------------------------------------------------


def test_en_song_romaji_matches_the_original_spelling_not_a_katakana_roundtrip():
    """en 곡의 romaji 정답은 원문 철자다 — 가타카나 음차를 거친 재변환(za wezaa
    poreketusu류)이 아니다. en 곡은 derive_en_display_units가 두 표기(romaji/en)를
    동시에 내므로, romaji가 그냥 en과 같아지는지로 오염 여부를 검산한다."""
    seg = _seg("weather vane", "", words=False)
    attach_pron_variants(seg)
    assert seg["pron"]["romaji"] == seg["pron"]["en"]
    # en 표시 자체가 원문 철자 기반이라 원문 낱말이 그대로(음절 구분 하이픈 정도만) 보여야 한다
    assert "weather" in seg["pron"]["en"].lower().replace("-", "")


def test_ja_song_latin_run_keeps_kana_derived_romaji():
    """ja 곡(라틴 리퍼리 경로가 아니다)에서는 이 수정이 영향을 주면 안 된다 — 가나·로마자
    변환이 여전히 정답이다. attach_pron_variants의 ja 분기(_attach_ja_pron_variants)는
    애초에 _attach_latin_pron_variants를 타지 않으므로 romaji가 en과 같아질 이유가 없다."""
    seg = _seg(NEKURA, NEKURA_HANGUL)
    attach_pron_variants(seg)
    assert seg["pron"]["romaji"] == NEKURA_ROMAJI
    assert "en" not in seg["pron"]  # ja 곡 표기에는 애초에 en 키가 없다


def test_legacy_contaminated_en_romaji_is_corrected_by_lazy_attach():
    """구세대 en 곡 구제 — romaji가 예전 버그로 en과 다르게 저장돼 있으면(가타카나
    재변환 근사) lazy 보완이 en 값으로 정정한다."""
    seg = _seg("Take it easy", "", words=False)
    seg["pron"] = {
        "hangul": "테이크 잇 이지",
        "kana": "テイクイットイージー",
        "romaji": "teikuittoiizii",  # 옛 근사(가타카나 재변환) — 오염된 값
        "en": "take it ea-sy",
    }
    attach_pron_variants(seg)
    assert seg["pron"]["romaji"] == "take it ea-sy"


def test_legacy_correction_is_idempotent():
    seg = _seg("Take it easy", "", words=False)
    seg["pron"] = {
        "hangul": "테이크 잇 이지",
        "kana": "テイクイットイージー",
        "romaji": "teikuittoiizii",
        "en": "take it ea-sy",
    }
    attach_pron_variants(seg)
    first = dict(seg["pron"])
    attach_pron_variants(seg)
    assert seg["pron"] == first


def test_legacy_correction_skips_when_romaji_already_matches_en():
    """이미 romaji==en이면 손댈 것이 없다 — 조건 자체가 거짓이라 아무 일도 안 한다."""
    seg = _seg("Take it easy", "", words=False)
    seg["pron"] = {"hangul": "테이크 잇 이지", "romaji": "take it ea-sy", "en": "take it ea-sy"}
    before = dict(seg["pron"])
    attach_pron_variants(seg)
    assert seg["pron"] == before


# ---------------------------------------------------------------------------
# F1 lazy 치유 — refine_window가 en 갈래로 잘못 보내 저장한 ko/zh 파손 pron 복구
# (2026-08-04 감사). 파손 지문: hangul 값이 원문에서 공백만 뺀 것과 완전히 같다.
# ---------------------------------------------------------------------------


def test_broken_ko_route_pron_is_healed_by_lazy_attach():
    """F1 실측 재현 — «사랑해 너를 위해»가 예전엔 en 갈래(라틴 전용 _WORD_RE)로 새
    hangul이 «사랑해너를위해»(공백만 소실된 원문)로 저장됐다. attach_pron_variants가
    그 파손 지문을 알아보고 버린 뒤 올바른 ko 파생(가타카나/RR 로마자)으로 재생성한다."""
    text = "사랑해 너를 위해"
    seg = _seg(text, "", words=False)
    seg["pron"] = {"hangul": "".join(text.split())}  # 파손 지문
    attach_pron_variants(seg, language="ko")
    # 순한글 줄은 hangul 표기 키를 새로 만들지 않는다(원문 자체가 표시라는 공유 계약,
    # _attach_ko_pron_variants) — 파손된 값이 남아 있지 않은 것이 핵심이다.
    assert "hangul" not in seg["pron"]
    assert seg["pron"]["kana"]
    assert seg["pron"]["romaji"]


def test_broken_ko_route_healing_is_idempotent():
    text = "사랑해 너를 위해"
    seg = _seg(text, "", words=False)
    seg["pron"] = {"hangul": "".join(text.split())}
    attach_pron_variants(seg, language="ko")
    first = dict(seg["pron"])
    attach_pron_variants(seg, language="ko")
    assert seg["pron"] == first


def test_broken_zh_route_pron_is_healed_by_lazy_attach():
    """zh 곡의 순한자 줄이 en 갈래를 거쳐 저장된 파손 지문을 치유한다 — 병음(공백
    포함, F3)이 실제로 생성된다."""
    text = "我不想说再见"
    seg = _seg(text, "", words=False)
    seg["pron"] = {"hangul": "".join(text.split())}
    attach_pron_variants(seg, language="zh")
    assert seg["pron"]["hangul"] != "".join(text.split())
    assert seg["pron"]["romaji"]


def test_broken_zh_route_healing_is_idempotent():
    text = "我不想说再见"
    seg = _seg(text, "", words=False)
    seg["pron"] = {"hangul": "".join(text.split())}
    attach_pron_variants(seg, language="zh")
    first = dict(seg["pron"])
    attach_pron_variants(seg, language="zh")
    assert seg["pron"] == first


def test_pron_that_is_not_a_broken_fingerprint_is_left_alone():
    # 우연히 hangul 키가 없어도(파손 지문이 아니면) 멱등 가드가 그대로 지킨다 — 이미
    # 올바르게 파생된 ko pron(원문과 다른 실제 변환값)은 건드리지 않는다.
    text = "사랑해 너를 위해"
    seg = _seg(text, "", words=False)
    correct = {"kana": "サランヘ ノルル ウィヘ", "romaji": "saranghae neoreul wihae"}
    seg["pron"] = dict(correct)
    attach_pron_variants(seg, language="ko")
    assert seg["pron"] == correct


def test_broken_fingerprint_check_ignores_lines_that_should_not_have_skipped():
    # 한자가 한글보다 많은 mixed 줄(_should_skip_derivation이 False를 내는 자리) 등
    # 정말로 en/ja 갈래가 정답인 원문은 우연히 fingerprint 모양이어도 건드리지 않는다
    # — 여기서는 순수 en 곡이라 애초에 should_have_skipped_en_route 자체가 거짓이다.
    seg = _seg("hi", "", words=False)
    seg["pron"] = {"hangul": "hi"}  # "hi".split()으로도 "hi" — 우연히 같은 모양
    attach_pron_variants(seg)
    assert seg["pron"] == {"hangul": "hi"}


# ---------------------------------------------------------------------------
# hangul 세그 근사 폴백 — 모라 수 불일치를 통째 포기하지 않는다 (2026-08-03)
# ---------------------------------------------------------------------------


def _kana_segs(n: int, step: float = 0.5) -> list[dict]:
    return [{"text": "カ", "start": i * step, "end": (i + 1) * step} for i in range(n)]


def test_hangul_segs_exact_mora_match_stays_resolved():
    from everyric2.server.worker import _ja_hangul_segments_from_kana

    seg = {"pron": {"hangul": "카카카"}, "pron_segs": {"kana": _kana_segs(3)}}
    out = _ja_hangul_segments_from_kana(seg)
    assert out and len(out) == 3
    assert all("resolved" not in s for s in out)  # 정합 경로는 기존 그대로(신뢰 표시)


def test_hangul_segs_mismatch_falls_back_to_proportional_approximation():
    # 실측 6_toWwEFXyA 세그 2·3: 모라 수 불일치(장음 축약·라틴 혼입)면 예전엔 None —
    # 기본 표기만 카라오케 타이밍이 죽었다. 이제 kana 시간축을 비례 배분하되
    # resolved: False로 근사임을 표시한다.
    from everyric2.server.worker import _ja_hangul_segments_from_kana

    seg = {"pron": {"hangul": "카카카카"}, "pron_segs": {"kana": _kana_segs(3)}}
    out = _ja_hangul_segments_from_kana(seg)
    assert out and len(out) == 4
    assert all(s.get("resolved") is False for s in out)
    # 시간축은 단조 — 비례 매핑(바닥 나눗셈)은 역행하지 않는다
    starts = [s["start"] for s in out]
    assert starts == sorted(starts)
    assert out[-1]["end"] == _kana_segs(3)[-1]["end"]
