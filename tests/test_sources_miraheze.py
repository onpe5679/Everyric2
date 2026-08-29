"""everyric2/sources/miraheze.py — 검색어 후보 생성 + 가사 표 파서 (네트워크 없이).

확장 everyric2-chrome/src/lib/miraheze.ts의 포트다. 여기서 고정하는 계약:
  · 검색어는 유튜브 원제 그대로가 아니라 «구분자 앞 조각 → 장식 제거판 → 원문» 순서
  · 접두 일치 히트를 우선하고, **마지막 후보에서만** 최상위 결과로 물러난다
  · 한 페이지에 표가 여럿이면 헤더에 lyrics-jp가 있는 표만 고른다
  · 빈 연 구분 행은 버리고, colspan 삽입구는 원문 한 줄로만 싣는다
"""

from pathlib import Path

import pytest

from everyric2.sources import miraheze

FIXTURES = Path(__file__).parent / "fixtures"


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


# ── 검색어 후보 ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # 실사용 사고(2026-07): 원제 전체를 검색하면 진짜 곡 페이지가 상위 10위 밖으로
        # 밀리고 프로듀서 페이지가 채택됐다 — 구분자 앞 조각이 첫 후보여야 한다
        (
            "シアンブルー / ポリスピカデリー feat. 初音ミク",
            ["シアンブルー", "シアンブルー / ポリスピカデリー feat. 初音ミク"],
        ),
        # 장식만 있는 제목 — ① 조각 없음, ② 장식 제거판, ③ 원문
        (
            "【MV】ダミー曲",
            ["ダミー曲", "【MV】ダミー曲"],
        ),
        # 구분자도 장식도 없으면 원문 하나뿐
        ("ダミー曲", ["ダミー曲"]),
    ],
)
def test_title_candidates_order(raw, expected):
    assert miraheze.title_candidates(raw) == expected


def test_title_candidates_are_capped_at_three():
    """장식과 구분자가 함께 있으면 후보는 셋 — 조각은 **원문에서** 뜬다(장식이 붙어 있다).

    확장 원본과 같은 순서다. 두 처리를 겹쳐 「ダミー曲」을 만들지는 않는다 — 대량 인제스트는
    유튜브 원제가 아니라 위키 원제로 조회하므로 이 경우가 사실상 오지 않아 그대로 뒀다.
    """
    got = miraheze.title_candidates("【MV】ダミー曲 / プロデューサー feat. 初音ミク")

    assert got == [
        "【MV】ダミー曲",
        "ダミー曲 / プロデューサー feat. 初音ミク",
        "【MV】ダミー曲 / プロデューサー feat. 初音ミク",
    ]


def test_feat_after_cjk_is_recognized_as_a_separator():
    """포팅 함정: 파이썬 \\b는 유니코드 기준이라 CJK 뒤 feat을 낱말 시작으로 안 본다.

    JS 원본과 같이 동작해야 한다 — 안 그러면 구분자를 못 찾아 원제 전체가 첫 후보가 되고,
    위 실사용 사고가 그대로 재현된다.
    """
    assert miraheze.strip_before_separator("ダミー曲feat.初音ミク") == "ダミー曲"


def test_ascii_word_boundary_does_not_split_inside_a_word():
    """「soft」 속 「ft」로 자르면 안 된다 — 낱말 경계를 요구하는 이유."""
    assert miraheze.strip_before_separator("soft dummy title") is None


@pytest.mark.parametrize("raw", ["/ダミー", "ダミー曲"])
def test_strip_before_separator_returns_none_without_a_usable_head(raw):
    assert miraheze.strip_before_separator(raw) is None


def test_strip_decorations_returns_none_when_nothing_changed():
    assert miraheze.strip_decorations("ダミー曲") is None


# ── 표 선택 + 3열 파싱 ─────────────────────────────────────────────


def test_picks_the_japanese_table_not_the_mandarin_cover_table():
    """실측: Vampire 페이지의 lyrics-2는 만다린 커버(중국어/병음/영어) 표였다."""
    lines, has_translation = miraheze.parse_lyrics_table(_fixture("miraheze_lyrics_3col.html"))

    assert has_translation is True
    assert all("假的" not in ln.text for ln in lines)
    assert lines[0].text == "だみーいちぎょうめ"
    assert lines[0].pronunciation == "damii ichigyoume"
    assert lines[0].translation == "dummy first line"


def test_singer_color_style_on_the_row_does_not_break_parsing():
    lines, _ = miraheze.parse_lyrics_table(_fixture("miraheze_lyrics_3col.html"))

    assert lines[1].text == "だみーにぎょうめ"
    assert lines[1].translation == "dummy second line"


def test_blank_stanza_row_is_dropped_and_merged_interjection_becomes_text_only():
    lines, _ = miraheze.parse_lyrics_table(_fixture("miraheze_lyrics_3col.html"))

    # <tr><td><br /></td></tr>(칸 1개 + 내용 없음)는 가사가 아니다
    assert all(ln.text for ln in lines)
    # colspan 삽입구는 원문/로마자 구분이 없으므로 text 하나로만 싣는다
    interjection = next(ln for ln in lines if ln.text == "dummy interjection")
    assert interjection.pronunciation is None
    assert interjection.translation is None


def test_br_inside_a_cell_folds_to_one_line():
    lines, _ = miraheze.parse_lyrics_table(_fixture("miraheze_lyrics_3col.html"))

    assert any(ln.text == "だみーさんぎょうめ つづき" for ln in lines)


def test_empty_translation_cell_becomes_none_not_empty_string():
    lines, _ = miraheze.parse_lyrics_table(_fixture("miraheze_lyrics_3col.html"))

    last = lines[-1]
    assert last.text == "だみーよんぎょうめ"
    assert last.translation is None


# ── 2열(번역 없음) ─────────────────────────────────────────────────


def test_two_column_table_reports_no_translation():
    lines, has_translation = miraheze.parse_lyrics_table(_fixture("miraheze_lyrics_2col.html"))

    assert has_translation is False
    assert len(lines) == 2
    assert lines[0].pronunciation == "damii ichigyoume"
    assert all(ln.translation is None for ln in lines)


def test_returns_none_when_no_japanese_lyrics_table_exists():
    html = '<table class="lyrics-table" id="lyrics-1"><tbody>' \
        '<tr class="lyrics-table-header"><th class="lyrics-zh">Chinese</th>' \
        '<th class="lyrics-en">English</th></tr>' \
        "<tr><td>假的</td><td>dummy</td></tr></tbody></table>"

    assert miraheze.parse_lyrics_table(html) is None


def test_returns_none_when_header_row_is_missing():
    html = '<table class="lyrics-table"><tbody><tr><td class="lyrics-jp">x</td></tr></tbody></table>'

    assert miraheze.parse_lyrics_table(html) is None


# ── URL·프로듀서 ───────────────────────────────────────────────────


def test_wiki_url_keeps_slashes_and_parens_but_swaps_spaces():
    """encodeURIComponent를 쓰면 '/'가 %2F로 깨져 문서를 못 찾는다."""
    assert miraheze.wiki_url("Dummy (Song)/prod") == (
        "https://vocaloidlyrics.miraheze.org/wiki/Dummy_(Song)/prod"
    )


def test_wiki_url_percent_encodes_non_ascii():
    assert miraheze.wiki_url("ダミー").startswith("https://vocaloidlyrics.miraheze.org/wiki/%")


@pytest.mark.parametrize(
    ("page_title", "expected"),
    [
        ("フラジール (Fragile)/nulut", "nulut"),
        ("ダミー曲 (Dummy)", None),
        ("ダミー曲/", None),
    ],
)
def test_producer_from_page_title(page_title, expected):
    assert miraheze.producer_from_page_title(page_title) == expected


# ── lookup (검색 → 파싱) ───────────────────────────────────────────


class _StubFetcher:
    """search/parse 응답을 미리 정해 두는 스텁. 요청한 URL을 순서대로 기록한다."""

    def __init__(
        self,
        search_hits: dict[str, list[dict]] | None = None,
        pages: dict[int, str] | None = None,
        page_titles: dict[int, str] | None = None,
    ) -> None:
        self.search_hits = search_hits or {}
        self.pages = pages or {}
        self.page_titles = page_titles or {}
        self.urls: list[str] = []

    def get_json(self, url: str):
        self.urls.append(url)
        from urllib.parse import parse_qs, urlparse

        params = parse_qs(urlparse(url).query)
        if params.get("action") == ["query"]:
            query = params["srsearch"][0]
            return {"query": {"search": self.search_hits.get(query, [])}}
        pageid = int(params["pageid"][0])
        html = self.pages.get(pageid)
        page_title = self.page_titles.get(pageid)
        if page_title is None:
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


def _hit(pageid: int, title: str) -> dict:
    return {"pageid": pageid, "title": title}


@pytest.mark.parametrize(
    ("candidate", "page_title", "expected"),
    [
        ("Ｒｏｋｉ", "roki (song)", True),  # NFKC + case-insensitive
        ("ロキ", "ロキ (Roki)", True),
        ("フラジール", "フラジール/nulut", True),
        ("Wonder", "Wonderland", False),  # 일반 문자가 이어진 부분열은 접미가 아님
        ("Wonder", "Tenshi", False),
        ("Whole Blue World", "トコヨトキヨ (Tokoyo Tokiyo)", False),
    ],
)
def test_page_title_match_is_normalized_and_boundary_aware(candidate, page_title, expected):
    assert miraheze.page_title_matches_candidate(candidate, page_title) is expected


def test_lookup_prefers_a_prefix_matching_hit_over_the_top_result():
    """실측: "ロキ" 검색 1위는 수록 앨범 페이지, 진짜 곡 페이지는 2위였다."""
    fetcher = _StubFetcher(
        search_hits={"ダミー曲": [_hit(1, "Dummy Album"), _hit(2, "ダミー曲 (Dummy)")]},
        pages={2: _fixture("miraheze_lyrics_3col.html")},
    )

    song = miraheze.lookup("ダミー曲", fetcher)

    assert song is not None
    assert song.page_title == "ダミー曲 (Dummy)"
    assert song.has_translation is True


@pytest.mark.parametrize(
    ("query", "wrong_page_title"),
    [
        pytest.param("Wonder", "Tenshi", id="wonder-must-not-adopt-tenshi"),
        pytest.param(
            "Whole Blue World",
            "トコヨトキヨ (Tokoyo Tokiyo)",
            id="whole-blue-world-must-not-adopt-tokoyo",
        ),
    ],
)
def test_lookup_rejects_real_incident_top_hit_even_when_it_has_a_japanese_lyrics_table(
    query, wrong_page_title
):
    """실사용 오채택 회귀: 검색 1위의 다른 곡에 정상 가사 표가 있어도 채택하지 않는다."""
    fetcher = _StubFetcher(
        search_hits={query: [_hit(1, wrong_page_title)]},
        pages={1: _fixture("miraheze_lyrics_3col.html")},
    )

    assert miraheze.lookup(query, fetcher) is None
    assert not any("action=parse" in url for url in fetcher.urls)


@pytest.mark.parametrize(
    ("query", "page_title"),
    [
        pytest.param("ロキ", "ロキ (Roki)", id="roki"),
        pytest.param("シアンブルー", "シアンブルー (Cyan Blue)", id="cyan-blue"),
    ],
)
def test_lookup_preserves_verified_real_prefix_matches(query, page_title):
    fetcher = _StubFetcher(
        search_hits={query: [_hit(1, "Unrelated album"), _hit(2, page_title)]},
        pages={2: _fixture("miraheze_lyrics_3col.html")},
    )

    song = miraheze.lookup(query, fetcher)

    assert song is not None
    assert song.page_title == page_title


def test_lookup_revalidates_the_canonical_title_returned_by_parse():
    """검색 히트가 맞아도 pageid가 다른 정규 제목으로 해석되면 채택하지 않는다."""
    fetcher = _StubFetcher(
        search_hits={"Wonder": [_hit(1, "Wonder (Song)")]},
        pages={1: _fixture("miraheze_lyrics_3col.html")},
        page_titles={1: "Tenshi"},
    )

    assert miraheze.lookup("Wonder", fetcher) is None


def test_lookup_does_not_fall_back_to_the_top_hit_on_a_non_final_candidate():
    """실사용 사고의 핵심: 첫 후보에서 물러나면 프로듀서 페이지를 오채택한다.

    첫 후보("ダミー曲")는 접두 일치가 없으므로 최상위 결과(프로듀서 페이지)를 채택하지
    않고 다음 후보로 넘어가야 한다. 마지막 후보에서 접두 일치가 잡혀 진짜 곡을 얻는다.
    """
    full_title = "ダミー曲 / プロデューサー"
    fetcher = _StubFetcher(
        search_hits={
            "ダミー曲": [_hit(9, "プロデューサー")],  # 접두 일치 없음
            full_title: [_hit(2, f"{full_title} feat")],  # 원문 후보에서 접두 일치
        },
        pages={
            9: '<table class="lyrics-table"><tbody>'
            '<tr class="lyrics-table-header"><th class="lyrics-jp">J</th>'
            '<th class="lyrics-romaji">R</th></tr>'
            "<tr><td>프로듀서 페이지의 표</td><td>x</td></tr></tbody></table>",
            2: _fixture("miraheze_lyrics_3col.html"),
        },
    )

    song = miraheze.lookup(full_title, fetcher)

    assert song is not None
    assert song.page_title.startswith(full_title)


def test_lookup_moves_to_the_next_candidate_when_the_page_has_no_lyrics_table():
    fetcher = _StubFetcher(
        search_hits={
            "ダミー曲": [_hit(1, "ダミー曲 (Dummy)")],
            "ダミー曲 / プロデューサー": [_hit(2, "ダミー曲 / プロデューサー")],
        },
        pages={1: "<div>가사 표가 없는 문서</div>", 2: _fixture("miraheze_lyrics_3col.html")},
    )

    song = miraheze.lookup("ダミー曲 / プロデューサー", fetcher)

    assert song is not None
    assert song.page_title == "ダミー曲 / プロデューサー"


def test_lookup_returns_none_when_nothing_matches():
    assert miraheze.lookup("ダミー曲", _StubFetcher()) is None


@pytest.mark.parametrize("title", ["", "   "])
def test_lookup_short_circuits_on_an_empty_title(title):
    fetcher = _StubFetcher()

    assert miraheze.lookup(title, fetcher) is None
    assert fetcher.urls == []  # 네트워크를 아예 안 쓴다


def test_lookup_uses_pageid_not_page_title_for_parsing():
    """실측: action=parse&page=<제목>은 공백+괄호가 섞이면 missingtitle로 실패했다."""
    fetcher = _StubFetcher(
        search_hits={"ダミー曲": [_hit(7, "ダミー曲 (Dummy)")]},
        pages={7: _fixture("miraheze_lyrics_3col.html")},
    )

    miraheze.lookup("ダミー曲", fetcher)

    parse_url = next(u for u in fetcher.urls if "action=parse" in u)
    assert "pageid=7" in parse_url
    assert "&page=" not in parse_url


def test_translation_lines_drop_rows_without_a_translation():
    fetcher = _StubFetcher(
        search_hits={"ダミー曲": [_hit(1, "ダミー曲")]},
        pages={1: _fixture("miraheze_lyrics_3col.html")},
    )
    song = miraheze.lookup("ダミー曲", fetcher)

    rows = song.translation_lines()

    assert all(r["text"] and r["translation"] for r in rows)
    assert all(set(r) == {"text", "translation"} for r in rows)
    assert "だみーよんぎょうめ" not in [r["text"] for r in rows]  # 번역 칸이 비어 있었다


def test_attribution_carries_share_alike_license():
    fetcher = _StubFetcher(
        search_hits={"ダミー曲": [_hit(1, "ダミー曲 (Dummy)")]},
        pages={1: _fixture("miraheze_lyrics_2col.html")},
    )
    song = miraheze.lookup("ダミー曲", fetcher)

    attribution = song.attribution()

    assert attribution["license"] == "CC BY-SA 4.0"
    assert attribution["source_id"] == "miraheze"
    assert attribution["name"] == "ダミー曲 (Dummy) — VocaloidLyrics Wiki"
