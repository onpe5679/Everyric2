"""자막 앵커 → 금지 구간 도출 회귀 테스트 (순수 로직 — 네트워크도 torch도 쓰지 않는다).

픽스처가 둘이고 둘 다 실측이다.

**표적** `zyRt-nBM3dY`(시니컬 나이트 플랜): 로마자를 괄호로 병기한 수동 자막이고 8.7~24.9초
간주 자리에 크레디트 3줄이 들어 있다. 정렬은 그 구간에 가사 8줄을 밀어 넣고 25~58초를 33초
비웠다. → 금지 구간이 나와야 한다.

**오폭 대조군** `ba7YbGO2aq4`(numb numb, 정상 곡): 자막이 우리 줄보다 짧다
(`우리 '網膜に焼き付く影 numb numb'` vs `자막 '網膜に焼き付く影'`). 단방향 포함 매칭이 실패해
매칭률이 76%로 떨어지고, **가사가 가득한 구간 3개가 「가사 없는 공백」으로 잘못 잡혔다**(그 안에
우리 줄이 5·5·2개). → 금지 구간이 하나도 나오지 않아야 한다.

emission 마스킹 쪽 기전은 `test_emission_mask.py`가 맡는다.
"""

import pytest

from everyric2.alignment.caption_anchors import (
    anchor_key,
    derive_anchor_plan,
    forbidden_spans,
    keys_match,
    lyric_like_events,
    match_anchors,
    script_lang_hint,
    span_candidates,
)
from everyric2.server.services.youtube_captions import manual_track_keys, order_manual_tracks

# 우리 가사 (위키에서 온 것 — 크레디트가 없다). idx 8~10은 자막에 없어 매칭되지 않는다.
LYRICS = [
    "触れてみたい秘密と",
    "曖昧なあなたのこと",
    "知りたいけど知りたくない",
    "しようよ",
    "別に意味とか無いけどさ",
    "眠い目を擦る",
    "夜が明けるまで",
    "話し続けようか",
    "くだらない話で笑って",
    "何も無かったみたいに",
    "朝を待っているだけ",
    "目が合う度に名前を呼ばないで",
]


def _ev(start, end, text):
    return {"start": start, "end": end, "text": text}


# 이 곡 자막의 실제 형태 — 로마자 병기 + 간주 자리의 크레디트 3줄.
# 한 이벤트가 우리 두 줄을 담는 경우(4·5번 줄, 6·7번 줄)도 그대로 재현한다.
CAPTIONS = [
    _ev(0.54, 2.30, "(Furetemitai himitsu to) 触れてみたい秘密と"),
    _ev(2.40, 4.60, "(Aimai na anata no koto) 曖昧なあなたのこと"),
    _ev(4.70, 6.90, "(Shiritai kedo shiritakunai) 知りたいけど知りたくない"),
    _ev(7.98, 9.40, "(Shiyou yo) しようよ"),
    _ev(10.20, 13.80, "･Vocal:初音ミク"),
    _ev(14.20, 18.60, "・Music＆Words : Ayase"),
    _ev(19.00, 23.50, "Ayase/シニカルナイトプラン"),
    _ev(24.94, 28.40, "(Betsu ni imi toka) 別に意味とか無いけどさ、眠い目を擦る"),
    _ev(28.60, 32.00, "(Yo ga akeru made) 夜が明けるまで話し続けようか"),
    _ev(44.18, 47.00, "(Me ga au tabi ni) 目が合う度に名前を呼ばないで"),
]

CREDIT_TEXTS = {
    "･Vocal:初音ミク",
    "・Music＆Words : Ayase",
    "Ayase/シニカルナイトプラン",
}

# ba7YbGO2aq4 (numb numb) — 정상 곡. 자막이 우리 줄보다 짧다(후렴 「numb numb」를 생략).
# 단방향 포함 매칭에서는 0·1·5·6번 줄이 탈락해 12.74~24.98이 「가사 없는 공백」으로 보였다.
NUMB_LYRICS = [
    "ゆらゆら numb numb",
    "網膜に焼き付く影 numb numb",
    "0.1mmの距離で",
    "好き好きすぎて",
    "息が止まりそうだ",
    "ゆらゆら numb numb",
    "網膜に焼き付く影 numb numb",
    "何が本当か分からない",
]
NUMB_CAPTIONS = [
    _ev(10.00, 12.50, "ゆらゆら"),
    _ev(12.74, 14.20, "網膜に焼き付く影"),
    _ev(14.50, 16.00, "0.1mmの距離で"),
    _ev(17.00, 19.00, "好き好きすぎて"),
    _ev(19.50, 22.00, "息が止まりそうだ"),
    _ev(24.98, 26.50, "ゆらゆら"),
    _ev(26.80, 28.30, "網膜に焼き付く影"),
    _ev(29.00, 31.00, "何が本当か分からない"),
]


def _spans(lyrics, events, min_gap_sec=8.0, margin_sec=1.0):
    return forbidden_spans(
        match_anchors(lyrics, events), events, lyrics, min_gap_sec, margin_sec
    )


# --------------------------------------------------------------------------
# 1) 정규화 없이는 매칭이 안 되고, 정규화하면 된다
# --------------------------------------------------------------------------


def test_romaji_gloss_must_be_stripped_before_matching():
    # 원문 그대로는 자막이 우리 줄을 담고 있지 않다 (실측: 이 곡 57줄 중 6줄만 매칭됐다)
    assert LYRICS[0] not in CAPTIONS[0]["text"].replace(" ", "")[:10]
    # 괄호 병기와 구두점을 걷어내면 담고 있다
    assert anchor_key(LYRICS[0]) in anchor_key(CAPTIONS[0]["text"])
    # 읽점으로 이어 붙인 자막 한 줄이 우리 두 줄을 담는다
    joined = anchor_key(CAPTIONS[7]["text"])
    assert anchor_key(LYRICS[4]) in joined and anchor_key(LYRICS[5]) in joined


# --------------------------------------------------------------------------
# 2) 크레디트는 매칭되지 않는다 — 따로 걸러내지 않는데도
# --------------------------------------------------------------------------


def test_credit_lines_never_become_anchors():
    anchors = match_anchors(LYRICS, CAPTIONS)
    assert anchors, "앵커가 하나도 안 잡혔다"
    for a in anchors:
        assert a.text not in CREDIT_TEXTS, f"크레디트가 앵커가 됐다: {a.text}"
    # 크레디트 3줄은 자막 이벤트로는 남아 있다 — 걸러낸 것이 아니라 매칭이 안 된 것이다
    assert sum(1 for e in CAPTIONS if e["text"] in CREDIT_TEXTS) == 3


def test_anchors_follow_our_line_order():
    anchors = match_anchors(LYRICS, CAPTIONS)
    assert [a.line_idx for a in anchors] == [0, 1, 2, 3, 4, 5, 6, 7, 11]
    # 한 이벤트가 우리 두 줄을 담을 수 있으므로 포인터는 그 자리에 머문다
    assert anchors[4].event_idx == anchors[5].event_idx == 7
    assert anchors[6].event_idx == anchors[7].event_idx == 8


# --------------------------------------------------------------------------
# 3) 금지 구간이 간주 자리에 나온다
# --------------------------------------------------------------------------


def test_forbidden_span_lands_on_the_interlude():
    spans = _spans(LYRICS, CAPTIONS)
    assert len(spans) == 1, f"금지 구간이 하나여야 한다: {spans}"
    lo, hi = spans[0]
    # 「しようよ」(7.98~9.40)와 「別に意味とか…」(24.94) 사이 — 실측 사고 구간 8.7~24.9다
    assert 7.98 < lo < 11.0 and 23.0 < hi < 24.94
    assert hi - lo > 12.0, "간주 대부분이 금지되어야 한다"


def test_gap_is_not_forbidden_when_our_unmatched_lines_could_live_there():
    # 8번 앵커(28.60~32.00)와 11번 앵커(44.18) 사이 간격은 12.2초로 게이트를 넘지만,
    # 그 사이에 매칭 안 된 우리 줄 3개(idx 8·9·10)가 있다 — 거기서 불릴 수도 있으므로
    # 금지하지 않는다. 이 규칙이 이 기능의 보수성 전체를 지탱한다.
    assert all(not (32.0 <= lo < 44.18) for lo, _ in _spans(LYRICS, CAPTIONS))


def test_same_event_pair_yields_no_span():
    # 우리 두 줄이 같은 이벤트에 붙으면 간격이 음수라 금지 구간이 생기지 않는다
    anchors = match_anchors(LYRICS[4:6], CAPTIONS[7:8])
    assert len(anchors) == 2
    assert _spans(LYRICS[4:6], CAPTIONS[7:8], min_gap_sec=0.1, margin_sec=0.0) == []


def test_short_gap_is_left_alone():
    assert _spans(LYRICS, CAPTIONS, min_gap_sec=30.0) == [], (
        "간주보다 긴 게이트에서는 아무것도 금지하지 않아야 한다"
    )


def test_left_edge_respects_a_caption_that_vanishes_before_the_line_is_sung():
    """자막이 가창보다 먼저 사라지는 트랙 — 표시 종료를 그대로 믿으면 앞 줄의 꼬리를 금지한다."""
    lyrics = ["長い一行をゆっくり歌う", "次の一行です"]
    # 11자짜리 줄인데 자막은 0.12초만 떠 있다
    events = [_ev(8.0, 8.12, "長い一行をゆっくり歌う"), _ev(40.0, 42.0, "次の一行です")]
    (lo, hi), = _spans(lyrics, events)
    # 글자 수로 본 가창 길이(11자 × 0.4s)가 표시 종료보다 늦으므로 그쪽이 앞 경계가 된다
    assert lo > 8.12 + 1.0 and lo == round(8.0 + 11 * 0.4 + 1.0, 3)
    assert hi == 39.0


def test_contiguous_caption_display_disables_the_span_conservatively():
    """다음 줄까지 계속 띄워 두는 트랙에서는 «간격»이 없다 — 아무것도 금지하지 않는다.

    이것이 이 기능이 조용히 아무 일도 하지 않는 가장 흔한 경로일 수 있다. 판정 근거는
    debug의 `anchors`(줄별 start/end)와 `skipped: no_gap`으로 그대로 드러난다.
    """
    lyrics = ["最初の一行です", "次の一行です"]
    events = [_ev(8.0, 40.0, "最初の一行です"), _ev(40.0, 42.0, "次の一行です")]
    assert _spans(lyrics, events) == []
    plan = derive_anchor_plan(lyrics, [("ja", events)], audio_sec=120.0)
    assert plan.debug["skipped"] == "no_gap"


def test_margin_only_shrinks_the_span():
    wide = _spans(LYRICS, CAPTIONS, margin_sec=0.0)[0]
    narrow = _spans(LYRICS, CAPTIONS, margin_sec=2.0)[0]
    assert narrow[0] > wide[0] and narrow[1] < wide[1]
    # 여유가 간격보다 크면 구간이 사라진다 (보수적 실패)
    assert _spans(LYRICS, CAPTIONS, margin_sec=20.0) == []


# --------------------------------------------------------------------------
# 3-B) 오폭 회귀 — 자막이 우리보다 짧은 정상 곡 (ba7YbGO2aq4)
# --------------------------------------------------------------------------


def test_bidirectional_matching_is_required_when_captions_are_shorter_than_our_lines():
    """단방향 포함(`우리 ⊂ 자막`)만 보면 이 곡의 절반이 탈락한다 — 그것이 오폭의 1차 원인이었다."""
    keys = [anchor_key(t) for t in NUMB_LYRICS]
    ev_keys = [anchor_key(e["text"]) for e in NUMB_CAPTIONS]
    one_way = sum(1 for k in keys if any(k in ek for ek in ev_keys))
    assert one_way == 4, f"단방향 매칭이 {one_way}/8 — 실측 76%로 떨어진 그 상황이다"
    assert len(match_anchors(NUMB_LYRICS, NUMB_CAPTIONS)) == 8, "양방향이면 전부 매칭돼야 한다"
    # 실측 정당 사례 둘이 모두 통과해야 한다 (_MIN_FRAGMENT_RATIO의 근거)
    assert keys_match(anchor_key("網膜に焼き付く影 numb numb"), anchor_key("網膜に焼き付く影"))
    assert keys_match(anchor_key("ゆらゆら numb numb"), anchor_key("ゆらゆら"))


def test_healthy_song_gets_no_forbidden_span():
    assert _spans(NUMB_LYRICS, NUMB_CAPTIONS) == [], "정상 곡에 금지 구간이 생겼다 — 오폭이다"
    plan = derive_anchor_plan(NUMB_LYRICS, [("ja", NUMB_CAPTIONS)], audio_sec=200.0)
    assert plan.debug["rate"] == 1.0 and plan.debug["skipped"] == "no_gap"


def test_the_apparent_void_is_full_of_lyric_events():
    """단방향 매칭이 「공백」으로 본 12.74~24.98 구간의 이벤트는 전부 우리 가사다.

    설령 다른 게이트가 그 구간을 후보로 올렸더라도 이 검사가 기각한다 — 「매칭이 없다」와
    「가사가 없다」를 가르는 것이 이 검사다.
    """
    lyricish = lyric_like_events(NUMB_LYRICS, NUMB_CAPTIONS)
    assert {2, 3, 4} <= lyricish, f"공백으로 오인된 구간의 이벤트가 가사로 인정되지 않았다: {lyricish}"
    assert lyricish == set(range(len(NUMB_CAPTIONS)))


def test_credit_events_are_not_lyric_like():
    """표적 곡의 금지 구간에는 이벤트가 3개 있다 — 개수가 조건이 될 수 없는 이유다."""
    lyricish = lyric_like_events(LYRICS, CAPTIONS)
    credits = [j for j, e in enumerate(CAPTIONS) if e["text"] in CREDIT_TEXTS]
    assert credits == [4, 5, 6]
    assert not (set(credits) & lyricish), "크레디트가 가사로 인정됐다"
    (cand,) = span_candidates(match_anchors(LYRICS, CAPTIONS), CAPTIONS, LYRICS, 8.0, 1.0)
    assert cand.events_between == 3 and cand.lyric_like == 0 and cand.accepted


def test_lyric_events_inside_a_gap_reject_the_span():
    """인접성·간격 게이트를 통과했는데도 구간 안이 가사면 기각한다 (게이트 ③ 단독 검증).

    우리 목록과 자막이 반복 후렴의 위치를 다르게 적은 상황이다 — 우리 1·2번 줄이 가사에서
    이웃이고 자막 간격도 14초라 후보가 되지만, 그 사이 이벤트가 우리 가사에 있는 말이다.
    """
    lyrics = ["前の行です", "サビの歌詞です", "次の行です", "サビの歌詞です"]
    events = [
        _ev(10.0, 12.0, "前の行です"),
        _ev(13.0, 15.0, "サビの歌詞です"),
        _ev(20.0, 22.0, "サビの歌詞です"),  # 우리 3번 줄과 같은 말 — 가사다
        _ev(30.0, 32.0, "次の行です"),
    ]
    (cand,) = span_candidates(match_anchors(lyrics, events), events, lyrics, 8.0, 1.0)
    assert cand.events_between == 1 and cand.lyric_like == 1
    assert not cand.accepted and cand.reason == "lyric_events"
    assert _spans(lyrics, events) == []
    # 그 이벤트가 우리 가사에 없는 말이면 같은 후보가 채택된다 — 기각의 원인이 내용임을 못박는다
    events[2] = _ev(20.0, 22.0, "Music＆Words : だれか")
    (ok,) = span_candidates(match_anchors(lyrics, events), events, lyrics, 8.0, 1.0)
    assert ok.events_between == 1 and ok.lyric_like == 0 and ok.accepted


# --------------------------------------------------------------------------
# 4) 반복 후렴 오매칭 방지 — 앞으로만 진행한다
# --------------------------------------------------------------------------


def test_repeated_chorus_matches_the_earliest_occurrence_after_the_pointer():
    lyrics = ["ラララ歌おう", "君の名前を", "ラララ歌おう"]
    events = [
        _ev(1.0, 3.0, "ラララ歌おう"),
        _ev(3.5, 5.0, "君の名前を"),
        _ev(60.0, 62.0, "ラララ歌おう"),
    ]
    anchors = match_anchors(lyrics, events)
    assert [(a.line_idx, a.event_idx) for a in anchors] == [(0, 0), (1, 1), (2, 2)]
    # 자유 탐색이면 첫 줄이 60초 이벤트에 붙어 앞 구간 전체가 금지될 수 있었다
    assert anchors[0].start == 1.0


def test_too_short_lines_are_not_anchors():
    # 「ああ」처럼 짧은 줄은 아무 이벤트에나 들어간다 — 앵커로 쓰지 않는다
    assert match_anchors(["ああ"], [_ev(1.0, 2.0, "ああ、君のああという声")]) == []
    # 「ラララ」(3자)도 후렴 조각이라 앵커가 아니다. 하한 4자는 실측 앵커 「しようよ」에서 왔다
    assert match_anchors(["ラララ"], [_ev(1.0, 2.0, "ラララと歌う声が")]) == []
    assert len(match_anchors(["しようよ"], [_ev(1.0, 2.0, "(Shiyou yo) しようよ")])) == 1


def test_a_short_refrain_fragment_does_not_anchor_a_long_line():
    # 자막 이벤트가 우리 줄의 아주 작은 조각이면 근거가 약하다 — 비율 하한이 막는다
    long_line = "とても長い一行がここにあって続きます"
    assert not keys_match(anchor_key(long_line), anchor_key("ahah"))
    assert match_anchors([long_line], [_ev(1.0, 2.0, "ah ah")]) == []


# --------------------------------------------------------------------------
# 5) 계획 도출 + 안전장치
# --------------------------------------------------------------------------


def test_plan_skips_tracks_that_do_not_match_and_takes_the_one_that_does():
    # 번역 트랙(vi)이 먼저 와도 기준을 못 넘으므로 넘어가고, 우리 가사와 맞는 ja가 채택된다
    vi = [_ev(s, e, f"dòng số {i}") for i, (s, e) in enumerate([(1.0, 2.0), (3.0, 4.0)])]
    plan = derive_anchor_plan(LYRICS, [("vi", vi), ("ja", CAPTIONS)], audio_sec=240.0)
    assert plan.debug["track"] == "ja"
    assert plan.debug["matched"] == 9
    assert plan.spans and len(plan.spans) == 1


def test_plan_stops_at_the_first_track_clearing_the_bar():
    """최적 탐색이 아니다 — 기준을 넘는 첫 트랙에서 멈춘다 (트랙당 yt-dlp 호출 1회다)."""
    seen = []

    def tracks():
        for lang, events in [("ja", CAPTIONS), ("ko", CAPTIONS), ("vi", [])]:
            seen.append(lang)
            yield lang, events

    plan = derive_anchor_plan(LYRICS, tracks(), audio_sec=240.0)
    assert seen == ["ja"], "기준을 넘은 뒤에도 트랙을 더 받았다"
    assert plan.debug["track"] == "ja" and plan.debug["rate"] == 0.75


def test_plan_gives_up_after_the_candidate_bound():
    """상한은 «최적 탐색 예산»이 아니라 «포기 지점»이다 — 넘겨도 기준 미달이면 앵커를 쓰지 않는다.

    실측 근거: zyRt-nBM3dY의 수동 트랙은 알파벳순으로 ja가 6번째다. 상한 5개로 자르고 최적을
    고르는 방식이면 zh-TW(11%)가 뽑혀 앵커가 버려졌다. 순서를 문자 체계로 정하는 것이
    상한보다 중요하다 — 그 순서는 `order_manual_tracks`가 만든다.
    """
    junk = [(f"t{i}", [_ev(1.0, 3.0, "全然違う歌詞です")]) for i in range(6)]
    plan = derive_anchor_plan(LYRICS, junk, audio_sec=240.0)
    assert plan.spans == [] and plan.debug["skipped"] == "low_match"
    assert len(plan.debug["tracks"]) == 6, "후보를 다 본 근거가 남아야 한다"


def test_low_match_rate_drops_the_anchors():
    # 다른 곡의 자막 — 시각은 그럴싸하지만 우리 가사와 맞지 않는다
    other = [_ev(1.0, 3.0, "全然違う歌詞です"), _ev(30.0, 32.0, "これも違う行です")]
    plan = derive_anchor_plan(LYRICS, [("ja", other)], audio_sec=240.0)
    assert plan.spans == []
    assert plan.debug["skipped"] == "low_match"
    assert not plan


def test_absurd_total_forbidden_length_drops_the_anchors():
    # 앵커가 크게 밀려 곡의 대부분을 금지하게 되면 매칭을 의심한다
    lyrics = ["最初の行です", "最後の行です"]
    events = [_ev(1.0, 3.0, "最初の行です"), _ev(200.0, 205.0, "最後の行です")]
    plan = derive_anchor_plan(lyrics, [("ja", events)], audio_sec=240.0)
    assert plan.spans == []
    assert plan.debug["skipped"] == "too_much_forbidden"
    # 상한을 풀면 같은 입력이 금지 구간을 낸다 — 걸러낸 것이 이 게이트임을 못박는다
    loose = derive_anchor_plan(
        lyrics, [("ja", events)], audio_sec=240.0, max_forbidden_ratio=1.0
    )
    assert loose.spans


def test_plan_records_its_reasoning_for_audit():
    plan = derive_anchor_plan(LYRICS, [("ja", CAPTIONS)], audio_sec=240.0)
    d = plan.debug
    assert d["track"] == "ja" and d["rate"] == 0.75
    assert d["tracks"] == [["ja", 10, 9, 0.75]]
    assert d["spans"] == [list(plan.spans[0])]
    # 어느 자막 줄이 경계를 만들었는지 되짚을 수 있어야 한다
    assert any("しようよ" in a[3] for a in d["anchors"])
    # 후보별 판정(사이 이벤트 수 / 그 중 가사로 보인 수 / 사유)도 남는다
    assert d["candidates"] == [[3, plan.spans[0][0], plan.spans[0][1], 3, 0, "ok"]]


def test_plan_records_rejected_candidates_too():
    """기각된 후보와 사유가 남아야 한다 — 그것이 오폭을 막았다는 증거다."""
    lyrics = ["前の行です", "サビの歌詞です", "次の行です", "サビの歌詞です"]
    events = [
        _ev(10.0, 12.0, "前の行です"),
        _ev(13.0, 15.0, "サビの歌詞です"),
        _ev(20.0, 22.0, "サビの歌詞です"),
        _ev(30.0, 32.0, "次の行です"),
    ]
    plan = derive_anchor_plan(lyrics, [("ja", events)], audio_sec=120.0)
    assert plan.spans == [] and plan.debug["skipped"] == "no_gap"
    assert [c[-1] for c in plan.debug["candidates"]] == ["lyric_events"]


def test_no_track_means_no_anchors():
    plan = derive_anchor_plan(LYRICS, [], audio_sec=240.0)
    assert plan.spans == [] and plan.debug["skipped"] == "no_manual_track"


# --------------------------------------------------------------------------
# 6) 트랙 선택 — ASR 배제, 우리 가사의 문자 체계가 순서를 정한다
# --------------------------------------------------------------------------

# zyRt-nBM3dY의 실측 트랙 구성. 원어는 ja인데 유튜브 신호는 vi를 가리키고,
# ja는 **알파벳순으로 6번째**다 — 상한으로 자르면 놓치는 그 상황이다.
MANY_TRACKS = {
    "subtitles": {
        lang: [{"name": lang}]
        for lang in "ar zh-TW en fil id ja ko ms es th tr vi".split()
    },
    "automatic_captions": {"vi-orig": [{"name": "Vietnamese"}], "en": [{"name": "English"}]},
    "language": "vi",
}


def test_asr_tracks_are_never_anchor_candidates():
    keys = manual_track_keys(MANY_TRACKS)
    assert "vi-orig" not in keys and "ja" in keys
    assert set(keys) == set(MANY_TRACKS["subtitles"])
    # 후보 순서에도 자동 생성은 끼지 않는다
    assert all(not k.endswith("-orig") for k in order_manual_tracks(MANY_TRACKS, "ja", 5))


def test_live_chat_is_not_a_caption_track():
    info = {"subtitles": {"live_chat": [{}], "ja": [{}]}}
    assert manual_track_keys(info) == ["ja"]


def test_our_lyrics_script_orders_the_candidates_over_youtube_signals():
    """실측 오폭 ②의 수정 — 상한이 아니라 **순서**가 원어를 맞힌다.

    zyRt-nBM3dY는 유튜브 신호가 vi를 가리키고(vi-orig + language=vi) ja가 알파벳 6번째다.
    상한 5개로 자르면 ja가 후보에서 아예 빠진다 — 그것이 표적곡에서 앵커를 놓친 원인이었다.
    """
    alphabetical = sorted(MANY_TRACKS["subtitles"])
    assert alphabetical.index("ja") == 5, "실측 트랙 구성이 아니다 (ja가 6번째여야 한다)"
    assert "ja" not in alphabetical[:5], "상한 5개로 자르면 ja가 빠지는 구성이어야 한다"

    order = order_manual_tracks(MANY_TRACKS, script_lang_hint("\n".join(LYRICS)), 5)
    assert order[0] == "ja", "우리 가사의 문자 체계가 ja를 첫 후보로 올려야 한다"
    assert len(order) == 5, "후보 트랙 수에 상한이 있어야 한다 (트랙당 다운로드 1회)"
    # 유튜브 신호(vi-orig · language=vi)는 순서에 **쓰지 않는다**. 자동 더빙 업로드에서
    # 일본어 곡에 vi-orig가 붙는 것이 실측으로 확인됐고, 그 신호를 순서에 넣으면 틀린
    # 트랙을 먼저 받아 본다. 힌트도 제목도 없으면 재현 가능한 알파벳순으로 떨어진다.
    no_hint = order_manual_tracks(MANY_TRACKS, None, 5)
    assert no_hint[0] == "ar", f"유튜브 신호를 따라갔다: {no_hint}"
    assert "vi" not in no_hint
    assert "ja" not in no_hint, "힌트가 없으면 상한 안에 ja가 들어오지 않는다"

    # 실제 영상에는 제목이 있고, 그것이 가사 힌트를 대신한다
    with_title = order_manual_tracks({**MANY_TRACKS, "title": "シニカルナイトプラン"}, None, 5)
    assert with_title[0] == "ja", f"제목의 문자 체계가 ja를 올려야 한다: {with_title}"


def test_no_manual_track_yields_no_candidates():
    assert order_manual_tracks({"automatic_captions": {"ja-orig": [{}]}}, "ja", 5) == []


@pytest.mark.parametrize(
    "text,expected",
    [
        ("触れてみたい秘密と", "ja"),
        ("너를 만나고 싶어", "ko"),
        ("我想見你", "zh"),
        ("just a latin line", None),
    ],
)
def test_script_lang_hint(text, expected):
    assert script_lang_hint(text) == expected


# --------------------------------------------------------------------------
# 7) 워커 배선 — 스위치가 꺼져 있으면 호출 자체가 예전과 같아야 한다
# --------------------------------------------------------------------------


class _RecordingEngine:
    """CTC 엔진 대역 — align이 받은 키워드를 그대로 기록한다."""

    def __init__(self):
        self._current_adapter = "jpn"
        self._current_lang = "ja"
        self._last_star_spans: list = []
        self._last_caption_anchor = None
        self.calls: list[dict] = []

    def is_available(self) -> bool:
        return True

    def align(self, audio, lyrics, language=None, progress_callback=None, **kwargs):
        from everyric2.inference.prompt import SyncResult, WordSegment

        self.calls.append(kwargs)
        self._last_caption_anchor = {"adopted": True, "loss": 0.0}
        return [
            SyncResult(
                text=ln.text,
                start_time=float(k),
                end_time=float(k) + 1.0,
                confidence=0.5,
                line_number=ln.line_number,
                word_segments=[WordSegment(word=ln.text[0], start=float(k), end=float(k) + 1.0)],
            )
            for k, ln in enumerate(lyrics)
        ]


def _run_worker_alignment(monkeypatch, tmp_path, *, anchors_on: bool, tracks, scaffold_on=None):
    """`_run_alignment`을 실제로 돌린다 — 오디오·보컬 분리·멜로디·자막 IO만 대역으로."""
    import numpy as np

    from everyric2.alignment import ctc_engine as ctc_mod
    from everyric2.audio import loader as loader_mod
    from everyric2.config.settings import get_settings
    from everyric2.server import worker as worker_mod
    from everyric2.server.services import youtube_captions as yc

    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"fake")
    fake_audio = loader_mod.AudioData(
        waveform=np.zeros(16000, dtype="float32"), sample_rate=16000, duration=240.0
    )

    class _FakeLoader:
        def load(self, path):
            return fake_audio

    engine = _RecordingEngine()
    monkeypatch.setattr(loader_mod, "AudioLoader", _FakeLoader)
    monkeypatch.setattr(ctc_mod, "get_shared_ctc_engine", lambda _s: engine)
    monkeypatch.setattr(worker_mod, "_separate_vocals", lambda _a: None)
    monkeypatch.setattr(worker_mod, "_estimate_tempo", lambda _a: None)
    monkeypatch.setattr(
        yc, "iter_manual_caption_events", lambda vid, hint, limit: iter(tracks)  # noqa: ARG005
    )

    settings = get_settings()
    saved = {
        "melody": settings.melody.enabled,
        "anchors": settings.alignment.caption_anchors,
        "scaffold": settings.alignment.caption_scaffold,
        # 캡션 앵커는 레거시 CTC 엔진의 forbidden_spans 계약을 시험한다(위 get_shared_ctc_
        # engine 대역) — 새 스택(기본값 owsm/omniasr)은 이 경로를 안 쓰므로 이 테스트가
        # 실제로 exercising하는 레거시 경로로 강제 고정한다.
        "engine": settings.alignment.engine,
    }
    object.__setattr__(settings.melody, "enabled", False)
    object.__setattr__(settings.alignment, "caption_anchors", anchors_on)
    object.__setattr__(settings.alignment, "engine", "ctc")
    if scaffold_on is not None:
        object.__setattr__(settings.alignment, "caption_scaffold", scaffold_on)
    try:
        result = worker_mod._run_alignment(
            str(audio_file), "\n".join(LYRICS), "ja", video_id="zyRt-nBM3dY"
        )
    finally:
        object.__setattr__(settings.melody, "enabled", saved["melody"])
        object.__setattr__(settings.alignment, "caption_anchors", saved["anchors"])
        object.__setattr__(settings.alignment, "caption_scaffold", saved["scaffold"])
        object.__setattr__(settings.alignment, "engine", saved["engine"])
    return result, engine


def test_worker_feeds_the_forbidden_spans_and_records_the_decision(monkeypatch, tmp_path):
    result, engine = _run_worker_alignment(
        monkeypatch, tmp_path, anchors_on=True, tracks=[("ja", CAPTIONS)]
    )
    assert engine.calls and "forbidden_spans" in engine.calls[0]
    lo, hi = engine.calls[0]["forbidden_spans"][0]
    assert 7.98 < lo < 11.0 and 23.0 < hi < 24.94
    # 사후 감사용 근거가 곡 단위 debug에 남는다 (어느 트랙·몇 줄 매칭·어느 구간·채택 여부)
    anchors = result["debug"]["caption_anchors"]
    assert anchors["track"] == "ja" and anchors["matched"] == 9
    assert anchors["spans"] == [[lo, hi]]
    assert anchors["decision"] == {"adopted": True, "loss": 0.0}


def test_worker_records_why_anchors_were_not_used(monkeypatch, tmp_path):
    other = [_ev(1.0, 3.0, "全然違う歌詞です"), _ev(30.0, 32.0, "これも違う行です")]
    result, engine = _run_worker_alignment(
        monkeypatch, tmp_path, anchors_on=True, tracks=[("ja", other)]
    )
    assert engine.calls[0] == {}, "앵커를 안 쓸 때는 키워드 자체를 넘기지 않아야 한다"
    assert result["debug"]["caption_anchors"]["skipped"] == "low_match"


def test_constraint_off_still_records_the_plan_for_the_scaffold(monkeypatch, tmp_path):
    """제약(caption_anchors)이 꺼져도 스캐폴드(기본 ON)가 같은 조달·매칭을 쓴다 — 계획과
    판정은 debug에 남되, **정렬 호출에는 아무 제약도 들어가지 않는다** (제약/골격 분리)."""
    result, engine = _run_worker_alignment(
        monkeypatch, tmp_path, anchors_on=False, tracks=[("ja", CAPTIONS)]
    )
    assert engine.calls == [{}], "제약 스위치가 꺼졌는데 정렬 호출이 달라졌다"
    assert "caption_anchors" in result["debug"]
    assert "caption_scaffold" in result["debug"]


def test_both_switches_off_keep_the_call_and_the_debug_exactly_as_before(monkeypatch, tmp_path):
    result, engine = _run_worker_alignment(
        monkeypatch, tmp_path, anchors_on=False, scaffold_on=False, tracks=[("ja", CAPTIONS)]
    )
    assert engine.calls == [{}], "스위치가 꺼졌는데 정렬 호출이 달라졌다"
    assert "caption_anchors" not in result["debug"]
    assert "caption_scaffold" not in result["debug"]
