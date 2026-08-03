"""everyric2/sources/vocaro.py — 곡 페이지 파서 (네트워크 없이 픽스처로).

확장 everyric2-chrome/src/lib/vocaro.ts의 parseSongPage를 포트한 것이므로, 여기서
고정하는 것은 «그 규칙과 같게 동작한다»는 계약이다: 3행 1세트(원문/발음/번역),
2행 1세트(원문/번역), 루비 읽기는 원문에서 제외, <br>은 공백으로 접기.
"""

from pathlib import Path

import pytest

from everyric2.sources import vocaro
from everyric2.sources.base import WikiFetcher

FIXTURES = Path(__file__).parent / "fixtures"


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


class _StubFetcher:
    """WikiFetcher 자리에 끼우는 스텁 — 요청한 URL을 기록하고 준 본문을 돌려준다."""

    def __init__(self, text: str | None) -> None:
        self.text = text
        self.urls: list[str] = []

    def get_text(self, url: str) -> str | None:
        self.urls.append(url)
        return self.text


# ── 3행 1세트 ──────────────────────────────────────────────────────


def test_parses_three_row_sets_into_original_pron_translation():
    title, lines = vocaro.parse_song_page(_fixture("vocaro_song_3row.html"))

    assert title == "テスト曲"
    assert len(lines) == 3
    assert lines[0].pronunciation == "카리노 이치교오메"
    assert lines[0].translation == "가짜 첫째 줄"
    assert lines[2].translation == "셋째 줄 계속"


def test_ruby_reading_is_stripped_from_the_original_column():
    """루비(<span class="rt">)를 남기면 원문이 "仮かりの…"가 되어 정렬이 글자 수부터 틀린다."""
    _title, lines = vocaro.parse_song_page(_fixture("vocaro_song_3row.html"))

    assert lines[0].text == "仮のいちぎょうめ"
    assert "かり" not in lines[0].text


def test_entities_are_decoded_and_br_folds_to_space():
    _title, lines = vocaro.parse_song_page(_fixture("vocaro_song_3row.html"))

    assert lines[1].text == "にぎょうめ&ダミー"
    assert lines[2].text == "さんぎょうめ つづき"  # <br /> → 공백 (한 칸 = 한 줄)


# ── 2행 1세트 ──────────────────────────────────────────────────────


def test_parses_two_row_sets_as_original_and_translation_without_pronunciation():
    title, lines = vocaro.parse_song_page(_fixture("vocaro_song_2row.html"))

    assert title == "발음なし테스트"
    assert len(lines) == 2
    assert lines[0].text == "だみーいちぎょうめ"
    assert lines[0].translation == "더미 첫째 줄"
    assert all(ln.pronunciation is None for ln in lines)


# ── 어긋난 표·빠진 표 ──────────────────────────────────────────────


def test_returns_none_when_there_is_no_lyrics_table():
    assert vocaro.parse_song_page("<div>가사 표가 없는 페이지</div>") is None


def test_odd_row_count_falls_back_to_original_only_lines():
    """3도 2도 아닌 행 수 — 조용히 버리지 않고 전부 원문으로 싣는다 (vocaro.ts와 같다)."""
    html = (
        '<table class="wiki-content-table"><tbody>'
        "<tr><td>いち</td></tr><tr><td>に</td></tr><tr><td>さん</td></tr>"
        "<tr><td>よん</td></tr><tr><td>ご</td></tr>"
        "</tbody></table>"
    )
    _title, lines = vocaro.parse_song_page(html)

    assert [ln.text for ln in lines] == ["いち", "に", "さん", "よん", "ご"]
    assert all(ln.translation is None for ln in lines)


def test_empty_original_lines_are_dropped():
    html = (
        '<table class="wiki-content-table"><tbody>'
        "<tr><td>いち</td></tr><tr><td>ichi</td></tr><tr><td>하나</td></tr>"
        "<tr><td></td></tr><tr><td></td></tr><tr><td></td></tr>"
        "</tbody></table>"
    )
    _title, lines = vocaro.parse_song_page(html)

    assert [ln.text for ln in lines] == ["いち"]


# ── 다중 버전 페이지 (원곡/리믹스가 한 페이지에) ────────────────────

# /monitoring 실측 구조: h1 "가사" 아래 h2 버전 헤딩("오리지널"/"Best Friend Remix")이
# 각 표 앞에 붙는다. 힌트 없으면(구버전 확장) 첫 표 = 기존 동작.
_MULTI_VERSION_HTML = (
    "<h1>가사</h1>"
    "<h2>오리지널</h2>"
    '<table class="wiki-content-table"><tbody>'
    "<tr><td>げんきょく</td></tr><tr><td>genkyoku</td></tr><tr><td>원곡</td></tr>"
    "</tbody></table>"
    "<h2>Best Friend Remix</h2>"
    '<table class="wiki-content-table"><tbody>'
    "<tr><td>りみっくす</td></tr><tr><td>rimikkusu</td></tr><tr><td>리믹스</td></tr>"
    "</tbody></table>"
)


def test_variant_hint_picks_the_matching_version_table():
    """리믹스 영상 제목을 힌트로 주면 리믹스 표를 고른다 (실측: /monitoring 오리지널 오집)."""
    _title, lines = vocaro.parse_song_page(
        _MULTI_VERSION_HTML, "DECO*27 - モニタリング (Best Friend Remix) feat. 初音ミク"
    )

    assert [ln.text for ln in lines] == ["りみっくす"]
    assert lines[0].translation == "리믹스"


def test_no_hint_keeps_first_table_for_old_callers():
    _title, lines = vocaro.parse_song_page(_MULTI_VERSION_HTML)

    assert [ln.text for ln in lines] == ["げんきょく"]


def test_unmatched_hint_falls_back_to_first_table():
    """어느 버전 헤딩도 힌트에 없으면 첫 표 — 오리지널 영상 제목이 바로 이 경우다."""
    _title, lines = vocaro.parse_song_page(
        _MULTI_VERSION_HTML, "DECO*27 - モニタリング feat. 初音ミク"
    )

    assert [ln.text for ln in lines] == ["げんきょく"]


def test_single_table_page_ignores_hint():
    _title, lines = vocaro.parse_song_page(_fixture("vocaro_song_3row.html"), "무슨 힌트든")

    assert len(lines) == 3


def test_fetch_song_threads_variant_hint():
    fetcher = _StubFetcher(_MULTI_VERSION_HTML)

    song = vocaro.fetch_song("monitoring", fetcher, "モニタリング (Best Friend Remix)")

    assert [ln.text for ln in song.lines] == ["りみっくす"]


# ── 조회 + 파생 계약 ───────────────────────────────────────────────


def test_fetch_song_uses_plain_http_and_one_request():
    """사이트가 HTTPS를 지원하지 않는다 — https로 가면 301로 튕긴다."""
    fetcher = _StubFetcher(_fixture("vocaro_song_3row.html"))

    song = vocaro.fetch_song("dummy-slug", fetcher)

    assert fetcher.urls == ["http://vocaro.wikidot.com/dummy-slug"]
    assert song is not None
    assert song.slug == "dummy-slug"
    assert song.page_url == "http://vocaro.wikidot.com/dummy-slug"


@pytest.mark.parametrize("body", [None, "", "<div>표 없음</div>"])
def test_fetch_song_returns_none_when_page_is_unusable(body):
    assert vocaro.fetch_song("dummy-slug", _StubFetcher(body)) is None


def test_fetch_song_page_returns_empty_list_instead_of_none():
    assert vocaro.fetch_song_page("dummy-slug", _StubFetcher(None)) == []


def test_line_meta_keeps_only_lines_that_actually_have_pron_or_translation():
    fetcher = _StubFetcher(_fixture("vocaro_song_3row.html"))
    song = vocaro.fetch_song("dummy-slug", fetcher)

    meta = song.line_meta()

    assert len(meta) == 3
    assert set(meta[0]) == {"text", "pronunciation", "translation"}


def test_original_text_joins_only_the_original_column():
    song = vocaro.fetch_song("dummy-slug", _StubFetcher(_fixture("vocaro_song_2row.html")))

    assert song.original_text() == "だみーいちぎょうめ\nだみーにぎょうめ"


def test_attribution_carries_license_and_source_id():
    song = vocaro.fetch_song("dummy-slug", _StubFetcher(_fixture("vocaro_song_2row.html")))

    assert song.attribution() == {
        "name": "보카로 가사 위키",
        "url": "http://vocaro.wikidot.com/dummy-slug",
        "license": "CC BY 4.0",
        "source_id": "vocaro",
    }


# ── 조회기 (간격·백오프) ───────────────────────────────────────────


class _FakeResponse:
    def __init__(self, status_code: int, text: str = "", encoding: str | None = "utf-8") -> None:
        self.status_code = status_code
        self.text = text
        self.encoding = encoding
        self.apparent_encoding = "utf-8"


class _FakeSession:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls = 0
        self.headers: dict[str, str] = {}

    def get(self, url: str, timeout: float):
        self.calls += 1
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_fetcher_retries_on_429_then_succeeds():
    session = _FakeSession([_FakeResponse(429), _FakeResponse(200, "본문")])
    slept: list[float] = []
    fetcher = WikiFetcher(
        min_interval_sec=0, backoff_sec=2.0, sleep=slept.append, session=session
    )

    assert fetcher.get_text("http://example.test/x") == "본문"
    assert session.calls == 2
    assert slept == [2.0]  # 첫 시도 실패 → 2초 백오프


def test_fetcher_gives_up_immediately_on_404():
    """다시 물어도 같은 답이 오는 상태 코드에는 백오프를 쓰지 않는다."""
    session = _FakeSession([_FakeResponse(404)])
    slept: list[float] = []
    fetcher = WikiFetcher(min_interval_sec=0, sleep=slept.append, session=session)

    assert fetcher.get_text("http://example.test/x") is None
    assert session.calls == 1
    assert slept == []


def test_fetcher_returns_none_after_exhausting_attempts():
    session = _FakeSession([_FakeResponse(503), _FakeResponse(503), _FakeResponse(503)])
    fetcher = WikiFetcher(
        min_interval_sec=0, max_attempts=3, sleep=lambda _s: None, session=session
    )

    assert fetcher.get_text("http://example.test/x") is None
    assert session.calls == 3


def test_fetcher_fixes_latin1_guess_so_cjk_does_not_break():
    """wikidot은 charset을 안 실어 requests가 ISO-8859-1로 추측한다 — 그대로 두면 깨진다."""
    resp = _FakeResponse(200, "テスト", encoding="ISO-8859-1")
    fetcher = WikiFetcher(
        min_interval_sec=0, sleep=lambda _s: None, session=_FakeSession([resp])
    )

    assert fetcher.get_text("http://example.test/x") == "テスト"
    assert resp.encoding == "utf-8"  # apparent_encoding으로 교체됐다
