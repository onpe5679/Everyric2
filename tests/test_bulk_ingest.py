"""scripts/bulk_ingest.py — 영상 후보 검증, 자막 원문 게이트, 상태 파일 멱등성.

네트워크는 쓰지 않는다: yt-dlp는 가짜 JSON, 자막은 youtube_captions의 두 IO 함수만
갈아끼우고(판정·정리·게이트는 실코드), 위키는 스텁 조회기, API는 가짜 클라이언트다.

여기서 고정하는 것이 사용자의 절대 조건이다 — **ASR 자막은 가사로 쓰지 않고**, 원문
언어가 아닌 수동 트랙도 원문으로 쓰지 않는다. 그리고 «오매칭 1곡이 스킵 10곡보다
해롭다»는 판단을 점수 규칙으로 못박는다.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from everyric2.server.services import youtube_captions as yc

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"


def _load_bulk_ingest():
    """scripts/는 패키지가 아니라 경로로 불러온다 (verify_regen.py와 같은 위치의 스크립트).

    ``sys.modules``에 먼저 꽂아야 한다 — ``@dataclass``가 클래스의 ``__module__``을
    ``sys.modules``에서 되찾아 보므로, 등록 전에 실행하면 수집 단계에서 터진다.
    """
    spec = importlib.util.spec_from_file_location(
        "bulk_ingest", REPO_ROOT / "scripts" / "bulk_ingest.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bi = _load_bulk_ingest()


class _NullFetcher:
    """어떤 위키 요청에도 «못 받았다»로 답하는 조회기."""

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def get_text(self, url):
        return None

    def get_json(self, url):
        return None


@pytest.fixture(autouse=True)
def no_wiki_network(monkeypatch):
    """진입점 테스트가 실제 위키를 두드리지 못하게 막는다.

    ``main()``은 자기 손으로 ``WikiFetcher``를 만든다 — 스텁을 넣어 주지 않으면 곡마다
    vocaro·miraheze에 진짜 HTTP를 보내고 백오프까지 기다린다(실측: 청크당 수 초).
    협력자를 직접 넣는 테스트(_runtime)는 이 패치와 무관하게 자기 스텁을 쓴다.
    """
    monkeypatch.setattr(bi, "WikiFetcher", _NullFetcher)


# ── 영상 후보 (가짜 yt-dlp JSON) ───────────────────────────────────


def _entry(
    video_id: str = "abcdefghijk",
    title: str = "ダミー曲 / ダミーP feat. 初音ミク",
    channel: str = "ダミーP",
    duration: float = 210.0,
    view_count: int | None = 500_000,
) -> dict:
    return {
        "id": video_id,
        "title": title,
        "channel": channel,
        "duration": duration,
        "view_count": view_count,
    }


def test_accepts_a_producer_upload_with_the_song_name_in_the_title_head():
    verdict = bi.score_candidate("ダミー曲", bi.candidate_from_json(_entry()), producer="ダミーP")

    assert verdict.rejected is None
    assert "title_exact_head" in verdict.signals
    assert "producer_channel" in verdict.signals
    assert verdict.score >= bi.ACCEPT_SCORE


def test_accepts_a_topic_channel_upload():
    """『- Topic』은 배급 음원에 유튜브가 자동 생성한 채널 — 공식 음원 그 자체다."""
    entry = _entry(title="ダミー曲", channel="ダミーP - Topic", view_count=None)

    verdict = bi.score_candidate("ダミー曲", bi.candidate_from_json(entry))

    assert verdict.rejected is None
    assert "topic_channel" in verdict.signals


@pytest.mark.parametrize(
    "title",
    [
        "ダミー曲 歌ってみた",
        "【カラオケ】ダミー曲",
        "ダミー曲 (Cover)",
        "ダミー曲 off vocal",
        "ダミー曲 instrumental",
        "ダミー曲 inst",
        "ダミー曲 替え歌",
        "ダミー曲 空耳",
        "ダミー曲 ピアノアレンジ",
        "ダミー曲 リミックス",
        "ダミー曲 メドレー",
        "ダミー曲 弾き語り",
        "ダミー曲 ライブ",
        "ダミー曲 nightcore",
        "ダミー曲 커버",
        "ダミー曲 불러봤다",
        "ダミー曲 MMD",
    ],
)
def test_rejects_uploads_that_are_not_the_original(title):
    verdict = bi.score_candidate("ダミー曲", bi.candidate_from_json(_entry(title=title)))

    assert verdict.rejected == "not_original_upload"


def test_does_not_reject_a_word_that_merely_contains_a_marker_substring():
    """「inst」를 낱말 경계 없이 잡으면 「Instant」 같은 제목까지 날아간다."""
    entry = _entry(title="Instant ダミー曲 / ダミーP", channel="ダミーP - Topic")

    verdict = bi.score_candidate("ダミー曲", bi.candidate_from_json(entry))

    assert verdict.rejected is None


@pytest.mark.parametrize("duration", [30.0, 59.0, 601.0, 3600.0])
def test_rejects_durations_outside_the_song_range(duration):
    verdict = bi.score_candidate(
        "ダミー曲", bi.candidate_from_json(_entry(duration=duration)), producer="ダミーP"
    )

    assert verdict.rejected == "duration_out_of_range"


def test_rejects_a_candidate_with_no_duration_at_all():
    """길이를 모르면 곡인지 가릴 수 없다 — 확신 없으면 스킵이 이 파이프라인의 기본값이다."""
    entry = _entry()
    del entry["duration"]

    verdict = bi.score_candidate("ダミー曲", bi.candidate_from_json(entry), producer="ダミーP")

    assert verdict.rejected == "no_duration"


def test_rejects_a_title_that_does_not_contain_the_song_name():
    entry = _entry(title="ぜんぜんちがうきょく / ダミーP")

    verdict = bi.score_candidate("ダミー曲", bi.candidate_from_json(entry), producer="ダミーP")

    assert verdict.rejected == "title_mismatch"


def test_short_song_names_require_an_exact_head_match():
    """「ロキ」처럼 짧은 제목은 포함 매칭이 우연에 약하다 — 곡명 자리와 정확히 같아야 한다."""
    loose = bi.candidate_from_json(_entry(title="ロキロキのロキ / ダミーP", channel="ダミーP"))
    exact = bi.candidate_from_json(_entry(title="ロキ / ダミーP", channel="ダミーP - Topic"))

    assert bi.score_candidate("ロキ", loose).rejected == "short_title_needs_exact_match"
    assert bi.score_candidate("ロキ", exact).rejected is None


def test_title_containment_alone_is_not_enough_to_be_accepted():
    """제목 하나로 채택하면 오매칭이 들어온다 — 독립된 근거가 하나는 더 있어야 한다."""
    entry = _entry(
        title="なにかのどうが ダミー曲 まとめ",
        channel="랜덤 채널",
        view_count=None,
    )

    verdict = bi.score_candidate("ダミー曲", bi.candidate_from_json(entry))

    assert verdict.rejected == "below_accept_score"
    assert verdict.signals == ["title_contained"]


def test_blocked_video_ids_are_rejected_even_when_everything_else_matches():
    entry = _entry(video_id="b2NTglk9tvI", channel="ダミーP - Topic")

    assert bi.score_candidate("ダミー曲", bi.candidate_from_json(entry)).rejected == "blocklisted"


def test_candidate_from_json_needs_an_id_and_a_title():
    assert bi.candidate_from_json({"title": "ダミー曲"}) is None
    assert bi.candidate_from_json({"id": "abcdefghijk"}) is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ダミー曲 / ダミーP feat. 初音ミク", "ダミー曲"),
        ("【MV】ダミー曲 / ダミーP", "ダミー曲"),
        ("ダミー曲", "ダミー曲"),
        ("ダミー曲 - ダミーP", "ダミー曲"),
    ],
)
def test_video_title_head_finds_the_song_name_slot(raw, expected):
    assert bi.video_title_head(raw) == expected


# ── 후보 고르기 ────────────────────────────────────────────────────


def test_pick_video_takes_the_highest_scoring_candidate_and_reports_every_verdict():
    entries = [
        _entry(video_id="cover000000", title="ダミー曲 歌ってみた"),
        _entry(video_id="plain111111", title="なにか ダミー曲", channel="랜덤", view_count=None),
        _entry(video_id="topic222222", title="ダミー曲", channel="ダミーP - Topic"),
    ]

    chosen, evidence = bi.pick_video("ダミー曲", entries)

    assert chosen.candidate.video_id == "topic222222"
    assert len(evidence) == 3  # 떨어진 후보의 사유까지 상태 파일에 남는다
    assert {e["video_id"]: e["rejected"] for e in evidence}["cover000000"] == "not_original_upload"


def test_pick_video_breaks_score_ties_by_view_count():
    entries = [
        _entry(video_id="fewviews000", title="ダミー曲", channel="ダミーP - Topic", view_count=200_000),
        _entry(video_id="manyviews00", title="ダミー曲", channel="ダミーP - Topic", view_count=9_000_000),
    ]

    chosen, _ = bi.pick_video("ダミー曲", entries)

    assert chosen.candidate.video_id == "manyviews00"


def test_pick_video_returns_none_when_no_candidate_is_convincing():
    entries = [_entry(video_id="cover000000", title="ダミー曲 cover")]

    chosen, evidence = bi.pick_video("ダミー曲", entries)

    assert chosen is None
    assert evidence[0]["rejected"] == "not_original_upload"


# ── yt-dlp 호출 ────────────────────────────────────────────────────


class _FakeProc:
    def __init__(self, stdout: str = "", stderr: str = "") -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = 0


def test_search_builds_a_metadata_only_ytsearch_command():
    seen: list[list[str]] = []

    def runner(cmd, timeout):
        seen.append(cmd)
        return _FakeProc(stdout=json.dumps(_entry()))

    search = bi.YtdlpSearch(count=5, runner=runner, cookie_file=Path("cookies.txt"))
    entries = search.search("ダミー曲")

    assert entries[0]["id"] == "abcdefghijk"
    cmd = seen[0]
    assert "ytsearch5:ダミー曲" in cmd
    assert "--dump-json" in cmd and "--no-download" in cmd
    assert "--cookies" in cmd  # 검색도 봇 확인에 걸린다 — 배포 쿠키를 물려 준다


def test_search_parses_one_json_per_line_and_skips_garbage():
    stdout = "\n".join([json.dumps(_entry(video_id="aaaaaaaaaaa")), "not json", json.dumps(_entry(video_id="bbbbbbbbbbb"))])
    search = bi.YtdlpSearch(runner=lambda cmd, timeout: _FakeProc(stdout=stdout))

    entries = search.search("ダミー曲")

    assert [e["id"] for e in entries] == ["aaaaaaaaaaa", "bbbbbbbbbbb"]


def test_search_flattens_a_playlist_shaped_dump():
    stdout = json.dumps({"_type": "playlist", "entries": [_entry(video_id="ccccccccccc")]})
    search = bi.YtdlpSearch(runner=lambda cmd, timeout: _FakeProc(stdout=stdout))

    assert [e["id"] for e in search.search("ダミー曲")] == ["ccccccccccc"]


@pytest.mark.parametrize(
    "stderr",
    [
        "ERROR: unable to download video data: HTTP Error 403: Forbidden",
        "ERROR: HTTP Error 429: Too Many Requests",
        "ERROR: Sign in to confirm you're not a bot",
    ],
)
def test_search_raises_on_youtube_blocking_us(stderr):
    """403은 곡 탓이 아니라 이 서버 출구·쿠키 탓이다 — 곡 스킵으로 접으면 안 된다."""
    search = bi.YtdlpSearch(runner=lambda cmd, timeout: _FakeProc(stderr=stderr))

    with pytest.raises(bi.YtdlpBlockedError):
        search.search("ダミー曲")
    assert search.block_streak == 1


def test_block_streak_resets_after_a_good_search():
    responses = [
        _FakeProc(stderr="ERROR: HTTP Error 403: Forbidden"),
        _FakeProc(stdout=json.dumps(_entry())),
    ]
    search = bi.YtdlpSearch(runner=lambda cmd, timeout: responses.pop(0))

    with pytest.raises(bi.YtdlpBlockedError):
        search.search("ダミー曲")
    search.search("ダミー曲")

    assert search.block_streak == 0


def test_search_timeout_yields_no_candidates_rather_than_dying():
    import subprocess

    def runner(cmd, timeout):
        raise subprocess.TimeoutExpired(cmd, timeout)

    assert bi.YtdlpSearch(runner=runner).search("ダミー曲") == []


def test_fetch_video_requests_the_single_video_url_with_the_same_flags():
    seen = []

    def runner(cmd, timeout):
        seen.append(cmd)
        return _FakeProc(stdout=json.dumps(_entry(video_id="pinnedvid11")))

    search = bi.YtdlpSearch(runner=runner, cookie_file=Path("cookies.txt"))
    entries = search.fetch_video("pinnedvid11")

    assert entries[0]["id"] == "pinnedvid11"
    cmd = seen[0]
    assert "https://www.youtube.com/watch?v=pinnedvid11" in cmd
    assert "--dump-json" in cmd and "--no-download" in cmd and "--cookies" in cmd


# ── 자막 원문 게이트 ───────────────────────────────────────────────

_JA_LINES = ["だみーいちぎょうめ", "だみーにぎょうめ", "だみーさんぎょうめ", "だみーよんぎょうめ"]
_KO_LINES = ["더미 첫째 줄", "더미 둘째 줄", "더미 셋째 줄", "더미 넷째 줄"]
_MIXED_LINES = ["だみー 더미 하나", "にぎょうめ 둘째 줄", "さんぎょうめ 셋째 줄"]


def _info(manual: dict[str, list[str]], automatic: dict | None = None) -> dict:
    return {
        "title": "ダミー曲 / ダミーP",
        "uploader": "ダミーP",
        "subtitles": {k: [{"name": k}] for k in manual},
        "automatic_captions": automatic or {},
    }


@contextlib.contextmanager
def _captions(info: dict, tracks: dict[str, list[str]]):
    """extract_caption_info / download_track_lines만 갈아끼운다 — 게이트는 실코드가 돈다."""
    calls: dict[str, list] = {"downloaded": []}

    def fake_extract(video_id):
        return info

    def fake_download(video_id, lang, auto):
        assert auto is False, "ASR 자막을 받으려 했다 — 가사 경로에서 절대 금지다"
        calls["downloaded"].append(lang)
        lines = tracks.get(lang)
        if lines is None:
            raise yc.CaptionUnavailable("download_failed", "트랙을 못 받았어요")
        return [
            {"start": i * 3.0, "end": i * 3.0 + 2.5, "text": t} for i, t in enumerate(lines)
        ]

    original = (yc.extract_caption_info, yc.download_track_lines)
    yc.extract_caption_info, yc.download_track_lines = fake_extract, fake_download
    try:
        yield calls
    finally:
        yc.extract_caption_info, yc.download_track_lines = original


def test_picks_the_original_language_manual_track():
    info = _info({"ja": _JA_LINES, "ko": _KO_LINES})

    with _captions(info, {"ja": _JA_LINES, "ko": _KO_LINES}):
        lyrics, evidence = bi.caption_lyrics("abcdefghijk", "ja")

    assert lyrics is not None
    assert lyrics.track_key == "ja"
    assert lyrics.lines == _JA_LINES
    assert evidence["tracks"][0]["accepted"] is True


def test_refuses_a_korean_fan_translation_track_as_the_original():
    """일본어 곡에 한국어 자막을 원문으로 쓰면 정렬이 파국적으로 무너진다."""
    info = _info({"ko": _KO_LINES})

    with _captions(info, {"ko": _KO_LINES}):
        lyrics, evidence = bi.caption_lyrics("abcdefghijk", "ja")

    assert lyrics is None
    assert evidence["error"] == "no_original_track"
    assert evidence["tracks"][0]["rejected"] == "lang_code_mismatch"


def test_never_touches_automatic_captions_even_when_they_are_the_only_track():
    """사용자 절대 조건: 자동 생성(ASR) 자막은 가사로 쓰지 않는다."""
    info = _info({}, automatic={"ja": [{"name": "일본어 (자동 생성)"}], "vi-orig": [{}]})

    with _captions(info, {"ja": _JA_LINES}) as calls:
        lyrics, evidence = bi.caption_lyrics("abcdefghijk", "ja")

    assert lyrics is None
    assert evidence["error"] == "no_manual_captions"
    assert calls["downloaded"] == []  # 내려받기 자체를 시도하지 않았다


def test_rejects_a_dual_language_track_that_passes_the_body_language_check():
    """``verify_track_body``는 «가나 >= 한글»이면 ja로 인정한다 — 반반 섞인 자막이 통과한다.

    원문만 필요한 이 경로에는 그 여유가 없다: 원문 아닌 문자 비율로 한 겹 더 거른다.
    """
    info = _info({"ja": _MIXED_LINES})

    with _captions(info, {"ja": _MIXED_LINES}):
        lyrics, evidence = bi.caption_lyrics("abcdefghijk", "ja")

    assert lyrics is None
    assert evidence["tracks"][0]["rejected"] == "mixed_script"


def test_rejects_a_track_with_too_few_lines_to_be_lyrics():
    info = _info({"ja": ["だみー"]})

    with _captions(info, {"ja": ["だみー"]}):
        lyrics, evidence = bi.caption_lyrics("abcdefghijk", "ja")

    assert lyrics is None
    assert evidence["tracks"][0]["rejected"] == "too_short"


def test_rejects_a_latin_only_track_as_not_being_the_japanese_original():
    """로마자 트랙(romaji)은 원문이 아니다 — CJK 문자가 없으면 본문 검사에서 떨어진다."""
    romaji = ["damii ichigyoume", "damii nigyoume", "damii sangyoume"]
    info = _info({"ja": romaji})

    with _captions(info, {"ja": romaji}):
        lyrics, evidence = bi.caption_lyrics("abcdefghijk", "ja")

    assert lyrics is None
    assert evidence["tracks"][0]["rejected"] == "body_mismatch"


def test_caption_listing_failure_is_reported_not_raised():
    def boom(video_id):
        raise yc.CaptionUnavailable("listing_failed", "자막 목록을 못 받았어요")

    original = yc.extract_caption_info
    yc.extract_caption_info = boom
    try:
        lyrics, evidence = bi.caption_lyrics("abcdefghijk", "ja")
    finally:
        yc.extract_caption_info = original

    assert lyrics is None
    assert evidence["error"] == "listing_failed"


# ── 원문 언어 판정 ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("ダミー曲", "ja"),  # 가나 우세
        ("仮死化", "ja"),  # 한자만 — body_language는 'zh'로 읽지만 이 코퍼스는 일본 보카로다
        ("!mperfection", "ja"),  # 라틴만 — 판정 근거가 없다는 뜻이지 일본어가 아니란 뜻이 아니다
        ("더미 곡", "ko"),  # 한글 우세
    ],
)
def test_original_language_uses_corpus_knowledge_where_script_is_ambiguous(title, expected):
    assert bi.original_language(title) == expected


@pytest.mark.parametrize(
    ("lang", "counts", "expected"),
    [
        ("ja", {"kana": 10, "hangul": 0, "han": 5}, 0.0),
        ("ja", {"kana": 0, "hangul": 0, "han": 12}, 0.0),  # 한자는 일·중이 공유한다
        ("ja", {"kana": 6, "hangul": 4, "han": 0}, 0.4),
        ("ko", {"kana": 4, "hangul": 6, "han": 0}, 0.4),
        ("zh", {"kana": 0, "hangul": 5, "han": 5}, 0.0),  # 대응 표가 없는 언어는 0
        ("ja", {"kana": 0, "hangul": 0, "han": 0, "latin": 9}, 0.0),
    ],
)
def test_foreign_script_ratio(lang, counts, expected):
    assert bi.foreign_script_ratio(lang, counts) == pytest.approx(expected)


# ── 상태 파일 (재실행 멱등) ────────────────────────────────────────


def test_state_survives_a_reload_and_skips_finished_songs(tmp_path):
    path = tmp_path / "bulk_state.json"
    state = bi.BulkState(path)
    state.record(bi.SongRecord(slug="done-song", status="ok", video_id="abcdefghijk"))
    state.save()

    reloaded = bi.BulkState(path)

    assert reloaded.done_reason("done-song") == "already_done"
    assert reloaded.done_reason("never-seen") is None


def test_failures_are_retried_but_skips_are_not(tmp_path):
    path = tmp_path / "bulk_state.json"
    state = bi.BulkState(path)
    state.record(bi.SongRecord(slug="flaky", status="failed", reason="job_timeout"))
    state.record(bi.SongRecord(slug="passed-over", status="skipped", reason="no_confident_video"))
    state.save()

    reloaded = bi.BulkState(path)

    assert reloaded.done_reason("flaky") is None  # 잡 타임아웃은 대개 일시적이다
    assert reloaded.done_reason("passed-over") == "already_skipped:no_confident_video"


def test_retry_skipped_reopens_non_terminal_skips_only(tmp_path):
    path = tmp_path / "bulk_state.json"
    state = bi.BulkState(path)
    state.record(bi.SongRecord(slug="maybe-later", status="skipped", reason="no_confident_video"))
    state.record(bi.SongRecord(slug="never-again", status="skipped", reason="already_ingested"))
    state.save()

    reloaded = bi.BulkState(path, retry_skipped=True)

    assert reloaded.done_reason("maybe-later") is None
    assert reloaded.done_reason("never-again") == "already_skipped:already_ingested"


def test_dry_run_records_do_not_block_the_real_run(tmp_path):
    """총괄이 첫 25곡을 dry-run으로 검수한 뒤 같은 구간을 실제로 돌린다 — 막히면 안 된다."""
    path = tmp_path / "bulk_state.json"
    state = bi.BulkState(path)
    state.record_dry_run(bi.SongRecord(slug="inspected", status="skipped", reason="dry_run"))
    state.save()

    reloaded = bi.BulkState(path)

    assert reloaded.done_reason("inspected") is None
    assert "inspected" in reloaded.dry_runs  # 검수 근거는 남아 있다


def test_corrupt_state_file_starts_empty_instead_of_crashing(tmp_path):
    path = tmp_path / "bulk_state.json"
    path.write_text("{ 깨진 JSON", encoding="utf-8")

    state = bi.BulkState(path)

    assert state.songs == {}


def test_state_save_is_atomic_and_leaves_no_temp_file(tmp_path):
    path = tmp_path / "out" / "bulk_state.json"
    state = bi.BulkState(path)
    state.record(bi.SongRecord(slug="x", status="ok"))
    state.save()

    assert json.loads(path.read_text(encoding="utf-8"))["songs"]["x"]["status"] == "ok"
    assert list(path.parent.iterdir()) == [path]


# ── 코퍼스·스킵 목록 ───────────────────────────────────────────────


def test_load_corpus_keeps_file_order_so_offset_chunks_stay_stable(tmp_path):
    path = tmp_path / "vocaro_index.json"
    path.write_text(
        json.dumps(
            {
                "entries": [
                    {"slug": "b-song", "ko": "나", "ja": "ダミーニ"},
                    {"slug": "a-song", "ko": "가", "ja": "ダミーイチ"},
                    {"slug": "no-ja", "ko": "다"},
                    {"ko": "슬러그 없음"},
                ]
            }
        ),
        encoding="utf-8",
    )

    songs = bi.load_corpus(path)

    assert [s.slug for s in songs] == ["b-song", "a-song", "no-ja"]
    assert songs[2].ja is None


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (["a", "b"], ({"a", "b"}, set())),
        ({"slugs": ["a"], "video_ids": ["abcdefghijk"]}, ({"a"}, {"abcdefghijk"})),
        ({}, (set(), set())),
    ],
)
def test_load_skip_file_accepts_a_bare_list_or_an_object(tmp_path, payload, expected):
    path = tmp_path / "skip.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert bi.load_skip_file(path) == expected


# ── 기존 싱크 대조 ─────────────────────────────────────────────────


def test_existing_corpus_matches_a_stored_youtube_title_by_its_song_name_slot():
    corpus = bi.ExistingCorpus(
        video_ids={"abcdefghijk"}, titled=[("abcdefghijk", "【MV】ダミー曲 / ダミーP")]
    )

    assert corpus.title_hit("ダミー曲") == "abcdefghijk"
    assert corpus.title_hit("ぜんぜんちがうきょく") is None
    assert corpus.has_video("abcdefghijk") is True


def test_title_matching_misses_when_the_stored_title_has_no_separator():
    """실측 한계: 구분자 없이 공백으로만 이어진 제목은 길이비 0.44로 떨어진다.

    이 누락은 영상 id 대조와 서버의 ``(video_id, lyrics_hash)`` 판정이 받아 내므로
    재생성 사고로 번지지 않는다 — 대신 그 사실을 여기 못박아 둔다.
    """
    corpus = bi.ExistingCorpus(titled=[("abcdefghijk", "ダミー曲 ダミーP 초회한정")])

    assert corpus.title_hit("ダミー曲") is None


def test_missing_db_degrades_to_no_title_matching(tmp_path):
    corpus = bi.load_existing_corpus(tmp_path / "nope.db")

    assert corpus.available is False
    assert corpus.video_ids == set()


# ── 요약 ───────────────────────────────────────────────────────────


# ── 곡 하나 전체 (협력자 전부 스텁) ───────────────────────────────


class _FakeApi:
    def __init__(
        self,
        generate_response: dict | None = None,
        job_response: dict | None = None,
        layer_response: dict | None = None,
        layer_error: Exception | None = None,
    ) -> None:
        self.generate_response = generate_response or {"job_id": "job-1", "status": "processing"}
        self.job_response = job_response or {"status": "completed"}
        self.layer_response = layer_response or {"saved": True, "matched": 9, "total": 10}
        self.layer_error = layer_error
        self.generated: list[dict] = []
        self.waited: list[str] = []
        self.layers: list[tuple[str, dict]] = []

    def generate(self, payload):
        self.generated.append(payload)
        return self.generate_response

    def wait_job(self, job_id, timeout_sec, poll_sec, sleep):
        self.waited.append(job_id)
        return self.job_response

    def save_translation_layer(self, video_id, payload):
        self.layers.append((video_id, payload))
        if self.layer_error:
            raise self.layer_error
        return self.layer_response


class _FakeSearch:
    def __init__(self, entries: list[dict] | None = None, error: Exception | None = None) -> None:
        self.entries = entries if entries is not None else [_entry(video_id="topic222222", title="ダミー曲", channel="ダミーP - Topic")]
        self.error = error
        self.block_streak = 0
        self.queries: list[str] = []
        self.fetched: list[str] = []

    def search(self, query):
        self.queries.append(query)
        if self.error:
            self.block_streak += 1
            raise self.error
        return self.entries

    def fetch_video(self, video_id):
        self.fetched.append(video_id)
        if self.error:
            self.block_streak += 1
            raise self.error
        return self.entries


class _TextFetcher:
    def __init__(self, text: str | None) -> None:
        self.text = text

    def get_text(self, url):
        return self.text


class _JsonFetcher:
    def __init__(self, search_hits=None, pages=None) -> None:
        self.search_hits = search_hits or {}
        self.pages = pages or {}

    def get_json(self, url):
        from urllib.parse import parse_qs, urlparse

        params = parse_qs(urlparse(url).query)
        if params.get("action") == ["query"]:
            return {"query": {"search": self.search_hits.get(params["srsearch"][0], [])}}
        pageid = int(params["pageid"][0])
        html = self.pages.get(pageid)
        page_title = next(
            (
                hit["title"]
                for hits in self.search_hits.values()
                for hit in hits
                if hit["pageid"] == pageid
            ),
            None,
        )
        return {"parse": {"title": page_title, "text": {"*": html}}} if html else {}


def _runtime(tmp_path, **overrides):
    defaults = dict(
        api=_FakeApi(),
        search=_FakeSearch(),
        state=bi.BulkState(tmp_path / "state.json"),
        existing=bi.ExistingCorpus(),
        vocaro_fetcher=_TextFetcher((FIXTURES / "vocaro_song_3row.html").read_text(encoding="utf-8")),
        miraheze_fetcher=_JsonFetcher(
            search_hits={"ダミー曲": [{"pageid": 1, "title": "ダミー曲 (Dummy)"}]},
            pages={1: (FIXTURES / "miraheze_lyrics_3col.html").read_text(encoding="utf-8")},
        ),
        sleep=lambda _s: None,
    )
    defaults.update(overrides)
    return bi.Runtime(**defaults)


_SONG = bi.Song(slug="dummy-slug", ko="더미 곡", ja="ダミー曲")


def test_process_song_submits_caption_lyrics_with_wiki_pronunciation_and_translation(tmp_path):
    """행복 경로 — 가사는 원문 자막, 발음·한국어 번역은 vocaro, 영어는 miraheze."""
    rt = _runtime(tmp_path)
    info = _info({"ja": _JA_LINES, "ko": _KO_LINES})

    with _captions(info, {"ja": _JA_LINES, "ko": _KO_LINES}):
        rec = bi.process_song(_SONG, rt)

    assert rec.status == "ok"
    assert rec.lyrics_source == "caption"
    payload = rt.api.generated[0]
    assert payload["video_id"] == "topic222222"
    assert payload["lyrics"] == "\n".join(_JA_LINES)
    assert payload["language"] == "ja"
    assert payload["target_lang"] == "ko" and payload["line_meta_lang"] == "ko"
    assert payload["attribution"]["source_id"] == "vocaro"
    assert len(payload["line_meta"]) == 3  # vocaro 3세트의 발음+번역
    assert rt.api.waited == ["job-1"]
    assert rec.en_layer == "saved:9/10"


def test_process_song_sends_the_english_layer_as_a_wiki_origin_layer(tmp_path):
    rt = _runtime(tmp_path)
    info = _info({"ja": _JA_LINES})

    with _captions(info, {"ja": _JA_LINES}):
        bi.process_song(_SONG, rt)

    video_id, payload = rt.api.layers[0]
    assert video_id == "topic222222"
    assert payload["target_lang"] == "en"
    assert payload["origin"] == "wiki"  # 서버가 받는 origin은 caption|wiki|manual뿐이다
    assert payload["attribution"]["license"] == "CC BY-SA 4.0"
    assert all(set(ln) == {"text", "translation"} for ln in payload["lines"])


def test_process_song_falls_back_to_wiki_original_when_there_is_no_original_track(tmp_path):
    """수동 원문 자막이 없으면 위키 원문으로 가사를 구성한다 — 곡을 버리지 않는다."""
    rt = _runtime(tmp_path)
    info = _info({"ko": _KO_LINES})  # 한국어 팬 번역 트랙뿐

    with _captions(info, {"ko": _KO_LINES}):
        rec = bi.process_song(_SONG, rt)

    assert rec.status == "ok"
    assert rec.lyrics_source == "vocaro"
    assert rt.api.generated[0]["lyrics"].startswith("仮のいちぎょうめ")
    assert rec.evidence["caption"]["error"] == "no_original_track"


def test_process_song_skips_when_neither_captions_nor_wiki_have_lyrics(tmp_path):
    rt = _runtime(
        tmp_path,
        vocaro_fetcher=_TextFetcher(None),
        miraheze_fetcher=_JsonFetcher(),
    )
    info = _info({})

    with _captions(info, {}):
        rec = bi.process_song(_SONG, rt)

    assert rec.status == "skipped"
    assert rec.reason == "no_lyrics"
    assert rt.api.generated == []


def test_dry_run_resolves_everything_but_submits_nothing(tmp_path):
    """총괄이 첫 25곡을 검수하는 경로 — 영상 해석·가사 확보 근거는 남고 제출은 없다."""
    rt = _runtime(tmp_path, dry_run=True)
    info = _info({"ja": _JA_LINES})

    with _captions(info, {"ja": _JA_LINES}):
        rec = bi.process_song(_SONG, rt)

    assert rec.reason == "dry_run"
    assert rec.video_id == "topic222222"
    assert rec.lyrics_source == "caption"
    assert rec.evidence["lyrics_lines"] == len(_JA_LINES)
    assert rt.api.generated == [] and rt.api.layers == []


def test_process_song_skips_a_song_with_no_convincing_video(tmp_path):
    rt = _runtime(tmp_path, search=_FakeSearch(entries=[_entry(title="ダミー曲 歌ってみた")]))

    rec = bi.process_song(_SONG, rt)

    assert rec.reason == "no_confident_video"
    assert rt.api.generated == []
    assert rec.evidence["candidates"][0]["rejected"] == "not_original_upload"


def test_process_song_skips_a_song_with_no_search_result_at_all(tmp_path):
    rt = _runtime(tmp_path, search=_FakeSearch(entries=[]))

    assert bi.process_song(_SONG, rt).reason == "no_search_result"


def test_process_song_skips_a_video_that_already_has_a_sync(tmp_path):
    rt = _runtime(tmp_path, existing=bi.ExistingCorpus(video_ids={"topic222222"}))

    rec = bi.process_song(_SONG, rt)

    assert rec.reason == "already_ingested"
    assert rec.evidence["matched_by"] == "db_video_id"
    assert rt.api.generated == []


def test_process_song_skips_a_blocklisted_video_before_submitting(tmp_path):
    rt = _runtime(
        tmp_path,
        search=_FakeSearch(
            entries=[_entry(video_id="b2NTglk9tvI", title="ダミー曲", channel="ダミーP - Topic")]
        ),
    )

    rec = bi.process_song(_SONG, rt)

    assert rt.api.generated == []
    assert rec.reason in ("blocklisted", "no_confident_video")


def test_song_without_an_original_title_is_skipped_before_any_network_call(tmp_path):
    rt = _runtime(tmp_path)

    rec = bi.process_song(bi.Song(slug="s", ko="더미 곡", ja=None), rt)

    assert rec.reason == "no_original_title"
    assert rt.search.queries == []


# ── 인계(handoff) 모드 — 영상 고정 곡 ─────────────────────────────


_PINNED_SONG = bi.Song(
    slug="dummy-slug", ko="더미 곡", ja="ダミー曲", pinned_video_id="pinnedvid11"
)


def test_pinned_song_skips_search_and_resolves_the_given_video(tmp_path):
    search = _FakeSearch(
        entries=[
            _entry(video_id="pinnedvid11", title="ダミー曲 / ダミーP feat. 初音ミク", channel="ダミーP - Topic")
        ]
    )
    rt = _runtime(tmp_path, search=search, dry_run=True)
    info = _info({"ja": _JA_LINES})

    with _captions(info, {"ja": _JA_LINES}):
        rec = bi.process_song(_PINNED_SONG, rt)

    assert rec.reason == "dry_run"
    assert rec.video_id == "pinnedvid11"
    assert rec.evidence["pinned_video"] == "pinnedvid11"
    assert search.queries == []  # 검색은 한 번도 부르지 않는다
    assert search.fetched == ["pinnedvid11"]


def test_pinned_song_still_rejects_a_title_mismatch(tmp_path):
    """슬러그 부분일치 오매칭 방어 — 영상이 정해져 있어도 채점은 그대로 통과해야 한다."""
    search = _FakeSearch(entries=[_entry(video_id="pinnedvid11", title="全然違う曲", channel="誰か")])
    rt = _runtime(tmp_path, search=search)

    rec = bi.process_song(_PINNED_SONG, rt)

    assert rec.reason == "no_confident_video"
    assert rec.evidence["candidates"][0]["rejected"] == "title_mismatch"
    assert rt.api.generated == []


def test_load_handoff_songs_keeps_matched_rows_and_dedupes(tmp_path):
    corpus = [
        bi.Song(slug="dummy-slug", ko="더미 곡", ja="ダミー曲"),
        bi.Song(slug="other-slug", ko="다른 곡", ja="別曲"),
    ]
    rows = [
        {"video_id": "aaaaaaaaaaa", "slug": "dummy-slug", "match": "exact"},
        {"video_id": "bbbbbbbbbbb", "slug": "dummy-slug", "match": "substring"},  # 슬러그 중복
        {"video_id": "aaaaaaaaaaa", "slug": "other-slug", "match": "exact"},  # 영상 중복
        {"video_id": "ccccccccccc", "slug": "other-slug", "match": "substring"},
        {"video_id": "ddddddddddd", "match": "no_match"},
        {"video_id": "eeeeeeeeeee", "match": "unavailable", "error": "http_404"},
        {"video_id": "fffffffffff", "slug": "not-in-corpus", "match": "exact"},
    ]
    path = tmp_path / "handoff.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")

    songs = bi.load_handoff_songs(path, corpus)

    assert [(s.slug, s.pinned_video_id) for s in songs] == [
        ("dummy-slug", "aaaaaaaaaaa"),
        ("other-slug", "ccccccccccc"),
    ]
    assert songs[0].ja == "ダミー曲"  # 제목은 코퍼스 것을 쓴다 — 검색·대조 규칙과 같은 기준


def test_an_existing_sync_response_is_not_polled_as_a_job(tmp_path):
    """캐시 히트면 job_id 자리에 싱크 id가 온다 — 그것을 폴링하면 404다."""
    rt = _runtime(
        tmp_path,
        api=_FakeApi(generate_response={"job_id": "sync-abc", "status": "completed"}),
    )
    info = _info({"ja": _JA_LINES})

    with _captions(info, {"ja": _JA_LINES}):
        rec = bi.process_song(_SONG, rt)

    assert rec.status == "ok"
    assert rec.reason == "existing_sync"
    assert rec.sync_id == "sync-abc"
    assert rt.api.waited == []


@pytest.mark.parametrize(
    ("job_status", "expected"),
    [("failed", "job_failed"), ("timeout", "job_timeout")],
)
def test_job_failure_is_recorded_as_a_failure_not_a_skip(tmp_path, job_status, expected):
    rt = _runtime(tmp_path, api=_FakeApi(job_response={"status": job_status, "error": "그냥 실패"}))
    info = _info({"ja": _JA_LINES})

    with _captions(info, {"ja": _JA_LINES}):
        rec = bi.process_song(_SONG, rt)

    assert rec.status == "failed"
    assert rec.reason == expected


def test_generate_error_is_recorded_with_its_status(tmp_path):
    rt = _runtime(tmp_path, api=_FakeApi())
    rt.api.generate = lambda payload: (_ for _ in ()).throw(bi.ApiError(429, "일일 한도"))
    info = _info({"ja": _JA_LINES})

    with _captions(info, {"ja": _JA_LINES}):
        rec = bi.process_song(_SONG, rt)

    assert rec.status == "failed"
    assert rec.reason == "generate_error:429"
    assert rec.evidence["error"] == "일일 한도"


def test_a_rejected_english_layer_does_not_undo_a_successful_song(tmp_path):
    """번역 레이어는 덧layer다 — 422(일치율 미달)로 ko 싱크를 실패로 만들지 않는다."""
    rt = _runtime(tmp_path, api=_FakeApi(layer_error=bi.ApiError(422, "거의 안 맞아요")))
    info = _info({"ja": _JA_LINES})

    with _captions(info, {"ja": _JA_LINES}):
        rec = bi.process_song(_SONG, rt)

    assert rec.status == "ok"
    assert rec.en_layer == "low_match"


def test_english_layer_is_capped_at_the_server_line_limit():
    """서버는 한 번에 400줄까지 받는다 — 넘겨 보내면 422로 전부 버려진다."""
    from everyric2.sources.base import SourceLine

    api = _FakeApi()
    song = bi.miraheze.MirahezeSong(
        page_title="ダミー曲",
        url="https://x.test",
        lines=[SourceLine(text=f"だみー{i}", translation=f"dummy {i}") for i in range(500)],
        has_translation=True,
    )

    bi.save_en_layer(api, "abcdefghijk", song)

    assert len(api.layers[0][1]["lines"]) == bi.MAX_TRANSLATION_LAYER_LINES


@pytest.mark.parametrize(
    ("song", "expected"),
    [
        (None, "no_source"),
        (
            bi.miraheze.MirahezeSong(page_title="t", url="u", lines=[], has_translation=False),
            "no_source",
        ),
        (
            bi.miraheze.MirahezeSong(page_title="t", url="u", lines=[], has_translation=True),
            "no_translation_lines",
        ),
    ],
)
def test_save_en_layer_short_circuits_without_a_usable_source(song, expected):
    api = _FakeApi()

    assert bi.save_en_layer(api, "abcdefghijk", song) == expected
    assert api.layers == []


# ── 청크 실행 ──────────────────────────────────────────────────────


def test_run_chunk_does_not_reprocess_songs_recorded_in_a_previous_run(tmp_path):
    rt = _runtime(tmp_path)
    rt.state.record(bi.SongRecord(slug="dummy-slug", status="ok", video_id="topic222222"))

    summary = bi.run_chunk([_SONG], rt, sleep_between_sec=0)

    assert summary["skipped"] == {"already_done": 1}
    assert rt.search.queries == []  # yt-dlp를 아예 부르지 않았다


def test_run_chunk_persists_each_song_immediately(tmp_path):
    """밤새 돌다 중간에 죽어도 앞서 한 일이 남아야 한다."""
    rt = _runtime(tmp_path)
    info = _info({"ja": _JA_LINES})

    with _captions(info, {"ja": _JA_LINES}):
        bi.run_chunk([_SONG], rt, sleep_between_sec=0)

    saved = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert saved["songs"]["dummy-slug"]["status"] == "ok"
    assert saved["songs"]["dummy-slug"]["video_id"] == "topic222222"


def test_run_chunk_aborts_the_chain_after_three_consecutive_blocks(tmp_path):
    """403은 곡 탓이 아니다 — 계속 돌리면 코퍼스 전체가 실패 기록으로 오염된다."""
    rt = _runtime(tmp_path, search=_FakeSearch(error=bi.YtdlpBlockedError("403")))
    songs = [bi.Song(slug=f"s{i}", ko="더미", ja="ダミー曲") for i in range(6)]

    summary = bi.run_chunk(songs, rt, sleep_between_sec=0)

    assert summary["aborted"] == "ytdlp_blocked"
    assert summary["processed"] == bi.YTDLP_BLOCK_ABORT_STREAK  # 나머지는 건드리지 않았다
    assert summary["failed"] == {"ytdlp_blocked": 3}


def test_run_chunk_folds_an_unexpected_exception_into_that_songs_record(tmp_path):
    """한 곡의 사고로 밤샘 청크 전체를 잃지 않는다."""
    rt = _runtime(tmp_path)
    rt.search.search = lambda q: (_ for _ in ()).throw(RuntimeError("예상 못 한 사고"))

    summary = bi.run_chunk([_SONG], rt, sleep_between_sec=0)

    assert summary["failed"] == {"exception": 1}
    assert "RuntimeError" in summary["songs"][0]["evidence"]["error"]


def test_run_chunk_sleeps_between_songs_but_not_after_the_last(tmp_path):
    slept: list[float] = []
    rt = _runtime(tmp_path, sleep=slept.append, search=_FakeSearch(entries=[]))
    songs = [bi.Song(slug=f"s{i}", ko="더미", ja="ダミー曲") for i in range(3)]

    bi.run_chunk(songs, rt, sleep_between_sec=6.0)

    assert slept == [6.0, 6.0]


def test_dry_run_chunk_reprocesses_songs_already_dry_run(tmp_path):
    rt = _runtime(tmp_path, dry_run=True, search=_FakeSearch(entries=[]))

    bi.run_chunk([_SONG], rt, sleep_between_sec=0)
    summary = bi.run_chunk([_SONG], rt, sleep_between_sec=0)

    assert summary["processed"] == 1
    assert summary["skipped"] == {"no_search_result": 1}
    assert rt.state.songs == {}  # 실제 실행의 판정에는 손대지 않았다


# ── 진입점 (argparse → 실행 → 요약 → 종료 코드) ──────────────────


def _tiny_corpus(tmp_path, count: int = 2) -> Path:
    path = tmp_path / "vocaro_index.json"
    path.write_text(
        json.dumps(
            {"entries": [{"slug": f"s{i}", "ko": "더미 곡", "ja": "ダミー曲"} for i in range(count)]}
        ),
        encoding="utf-8",
    )
    return path


def _main(tmp_path, corpus: Path, *extra: str) -> int:
    return bi.main(
        [
            "--corpus",
            str(corpus),
            "--state-file",
            str(tmp_path / "state.json"),
            "--db",
            str(tmp_path / "absent.db"),
            "--sleep",
            "0",
            *extra,
        ]
    )


def test_missing_corpus_is_a_usage_error(tmp_path, capsys):
    assert _main(tmp_path, tmp_path / "nope.json", "--dry-run") == 2


def test_missing_admin_key_is_a_usage_error_for_a_real_run(tmp_path, monkeypatch):
    """dry-run은 키 없이도 되지만, 제출하는 실행은 키 없이 시작하면 전부 401이다."""
    monkeypatch.delenv("EVERYRIC_SERVER_ADMIN_API_KEY", raising=False)

    assert _main(tmp_path, _tiny_corpus(tmp_path)) == 2


def test_dry_run_chunk_exits_zero_and_prints_a_summary(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bi, "_run_command", lambda cmd, timeout: _FakeProc(stdout=""))

    code = _main(tmp_path, _tiny_corpus(tmp_path), "--dry-run", "--limit", "2")

    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["chunk"] == {"offset": 0, "limit": 2, "corpus": 2}
    assert summary["skipped"] == {"no_search_result": 2}


def test_a_blocked_chain_exits_three_so_the_caller_stops_looping(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        bi,
        "_run_command",
        lambda cmd, timeout: _FakeProc(stderr="ERROR: HTTP Error 403: Forbidden"),
    )

    code = _main(tmp_path, _tiny_corpus(tmp_path, count=5), "--dry-run", "--limit", "5")

    assert code == 3
    assert json.loads(capsys.readouterr().out)["aborted"] == "ytdlp_blocked"


def test_skip_file_removes_songs_from_the_chunk(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bi, "_run_command", lambda cmd, timeout: _FakeProc(stdout=""))
    skip = tmp_path / "skip.json"
    skip.write_text(json.dumps({"slugs": ["s0"]}), encoding="utf-8")

    _main(tmp_path, _tiny_corpus(tmp_path), "--dry-run", "--skip-file", str(skip))

    summary = json.loads(capsys.readouterr().out)
    assert summary["processed"] == 1
    assert [s["slug"] for s in summary["songs"]] == ["s1"]


def test_summary_out_writes_the_same_json_to_a_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bi, "_run_command", lambda cmd, timeout: _FakeProc(stdout=""))
    out = tmp_path / "nested" / "summary.json"

    _main(tmp_path, _tiny_corpus(tmp_path), "--dry-run", "--summary-out", str(out))

    assert json.loads(out.read_text(encoding="utf-8")) == json.loads(capsys.readouterr().out)


def test_summary_counts_by_reason():
    records = [
        bi.SongRecord(slug="a", status="ok"),
        bi.SongRecord(slug="b", status="skipped", reason="no_confident_video"),
        bi.SongRecord(slug="c", status="skipped", reason="no_confident_video"),
        bi.SongRecord(slug="d", status="skipped", reason="no_lyrics"),
        bi.SongRecord(slug="e", status="failed", reason="job_timeout"),
    ]

    summary = bi.summarize(records, aborted=None, elapsed_sec=12.34)

    assert summary["ok"] == 1
    assert summary["skipped"] == {"no_confident_video": 2, "no_lyrics": 1}
    assert summary["skipped_total"] == 3
    assert summary["failed"] == {"job_timeout": 1}
    assert summary["aborted"] is None
    assert len(summary["songs"]) == 5


def test_process_song_rejects_wiki_lyrics_when_parser_fallback_yields_tri_line_mix(tmp_path):
    """행수가 어긋난 가사 표(3의 배수도 2의 배수도 아님)는 파서가 «전부 원문»으로
    살린다 — 원문·독음·번역 3줄이 통째로 가사가 되는 실사고(2026-07-28,
    DAr03V5IIeQ 149줄)의 재현. 혼합 스크립트 검사가 인제스트 직전에 걸러야 한다."""
    rows = [
        "ショーケース", "쇼케스", "쇼케이스",
        "固定された姿のままで", "코테이사레타 스가타노 마마데", "고정된 모습 그대로",
        "ガラスの向こう", "가라스노 무코오", "유리 너머",
        "追加のはぐれ行",  # 11행 — 3의 배수도 2의 배수도 아니게 만드는 잉여 행
        "하나 더 남은 한글 행",
    ]
    html = (
        '<table class="wiki-content-table">'
        + "".join(f"<tr><td>{r}</td></tr>" for r in rows)
        + "</table>"
    )
    rt = _runtime(tmp_path, vocaro_fetcher=_TextFetcher(html))
    info = _info({})  # 자막 없음 — 위키 원문 폴백 강제

    with _captions(info, {}):
        rec = bi.process_song(_SONG, rt)

    assert rec.status == "skipped"
    assert rec.reason == "mixed_script_lyrics"
    assert rec.evidence["lyrics_foreign_ratio"] > bi.MAX_FOREIGN_SCRIPT_RATIO
    assert rt.api.generated == []
