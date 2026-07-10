"""vocaro_index 단위 테스트 — 네트워크 접근 없이 매칭/파서 로직만 검증한다.

match() 계열 테스트는 실제 크롤 결과(models/vocaro_index.json)에 의존하지 않도록
모듈 전역 캐시(_cache/_built_at)를 직접 주입하고 테스트 종료 후 원복한다.
"""

import pytest

from everyric2.server import vocaro_index as vi
from everyric2.server.vocaro_index import SongEntry


@pytest.fixture(autouse=True)
def _isolate_module_state():
    saved = (vi._cache, vi._built_at, vi._building)
    yield
    vi._cache, vi._built_at, vi._building = saved


def _set_entries(entries: list[SongEntry]) -> None:
    vi._cache = entries
    vi._built_at = "2026-07-10T00:00:00+00:00"


# ── match() ──────────────────────────────────────────────────────

def test_match_returns_none_when_index_empty():
    _set_entries([])
    assert vi.match("ロキ") is None


def test_match_exact_japanese_title():
    _set_entries([SongEntry(slug="roki", ko="로키", ja="ロキ")])
    result = vi.match("ロキ")
    assert result is not None
    assert result.slug == "roki"


def test_match_exact_korean_title_ignores_case_and_spaces():
    _set_entries([SongEntry(slug="isolation-ward", ko="격리병동", ja="隔離病棟")])
    result = vi.match("격리 병동")
    assert result is not None
    assert result.slug == "isolation-ward"


def test_match_partial_inclusion_within_length_ratio():
    _set_entries([SongEntry(slug="song", ko="긴 제목의 노래 입니다", ja=None)])
    # 쿼리가 등록된 제목에 포함되고 길이 비율이 0.5 이상이면 매칭된다
    result = vi.match("노래 입니다")
    assert result is not None
    assert result.slug == "song"


def test_match_rejects_partial_overlap_below_length_ratio():
    _set_entries([SongEntry(slug="song", ko="아주 긴 제목의 노래 가사 전문 입니다요", ja=None)])
    # 짧은 우연한 포함은 길이 비율 조건으로 걸러진다
    assert vi.match("노래") is None


def test_match_query_too_short_returns_none():
    _set_entries([SongEntry(slug="a", ko="a", ja=None)])
    assert vi.match("a") is None


def test_match_returns_none_when_nothing_matches():
    _set_entries([SongEntry(slug="roki", ko="로키", ja="ロキ")])
    assert vi.match("전혀 다른 제목") is None


def test_match_checks_both_ja_and_ko_fields():
    _set_entries(
        [
            SongEntry(slug="a", ko="가나다", ja=None),
            SongEntry(slug="b", ko="라마바", ja="日本語原題"),
        ]
    )
    assert vi.match("日本語原題").slug == "b"
    assert vi.match("가나다").slug == "a"


# ── 파서: 인덱스 페이지 (<li><a href="/slug">제목</a></li>) ──────

def test_parse_index_entries_extracts_slug_and_title():
    html_fixture = """
    <ul>
      <li><a href="/roki">로키</a></li>
      <li><a href="/isolation-ward">격리병동</a></li>
    </ul>
    """
    entries = vi._parse_index_entries(html_fixture)
    assert entries == [("roki", "로키"), ("isolation-ward", "격리병동")]


def test_parse_index_entries_excludes_allsongs_system_guide_slugs():
    html_fixture = """
    <li><a href="/roki">로키</a></li>
    <li><a href="/allsongs-h1">수록곡 일람</a></li>
    <li><a href="/guide-edit">편집 가이드</a></li>
    """
    entries = vi._parse_index_entries(html_fixture)
    assert entries == [("roki", "로키")]


def test_parse_index_entries_decodes_html_entities():
    html_fixture = '<li><a href="/song">Q&amp;A</a></li>'
    entries = vi._parse_index_entries(html_fixture)
    assert entries == [("song", "Q&A")]


# ── 파서: 곡 페이지 title-cell (원제) ──────────────────────────

def test_parse_title_cell_extracts_original_title():
    html_fixture = """
    <table class="wiki-content-table">
      <tr><th class="title-cell" colspan="2">ロキ</th></tr>
    </table>
    """
    assert vi._parse_title_cell(html_fixture) == "ロキ"


def test_parse_title_cell_strips_furigana_and_br_tags():
    html_fixture = (
        '<th class="title-cell">'
        '<span class="rt">かくり</span>隔離<br/>病棟'
        "</th>"
    )
    assert vi._parse_title_cell(html_fixture) == "隔離 病棟"


def test_parse_title_cell_returns_none_when_missing():
    assert vi._parse_title_cell("<table><tr><td>no title cell here</td></tr></table>") is None


# ── _normalize_title ───────────────────────────────────────────

def test_normalize_title_strips_spaces_and_symbols_case_insensitively():
    assert vi._normalize_title("Roki ロキ!") == "rokiロキ"
