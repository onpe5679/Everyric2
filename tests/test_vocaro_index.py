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


# ── 아티스트 토큰 오매핑 가드 (2026-08-03 실측: ダミーロマンス → cinderella-deco-27) ──

def test_artist_token_does_not_match_a_different_song_by_the_same_artist():
    # 위키에 없는 신곡의 풀 제목 — 아티스트 후보 "DECO*27"이 같은 아티스트의 다른 곡
    # 항목(ko 필드에 아티스트가 붙는 표기)에 포함 매칭돼 엉뚱한 가사가 붙었다.
    _set_entries([SongEntry(slug="cinderella-deco-27", ko="신데렐라/DECO*27", ja="シンデレラ")])
    assert vi.match("DECO*27 - ダミーロマンス feat. 初音ミク") is None


def test_full_title_of_the_actual_song_still_matches():
    # 같은 인덱스 항목이라도 진짜 그 곡의 풀 제목은 여전히 붙는다(ja 정확 일치 경로)
    _set_entries([SongEntry(slug="cinderella-deco-27", ko="신데렐라/DECO*27", ja="シンデレラ")])
    result = vi.match("DECO*27 - シンデレラ feat. 初音ミク")
    assert result is not None
    assert result.slug == "cinderella-deco-27"


def test_whole_query_as_partial_title_is_still_a_legitimate_match():
    # 가드는 부분 후보(q != 풀 쿼리)에만 적용된다 — 쿼리 전체가 위키 제목의 부분
    # 문자열인 기존 정당 케이스는 그대로 살아 있어야 한다.
    _set_entries([SongEntry(slug="song", ko="긴 제목의 노래 입니다", ja=None)])
    result = vi.match("노래 입니다")
    assert result is not None
    assert result.slug == "song"


def test_short_entry_title_inside_artist_token_is_rejected():
    # 실측 2호: 3글자 곡 "Dec."이 아티스트 후보 "deco27" 안에 포함(길이비 정확히 0.5)돼
    # 붙었다 — 역방향(n ⊂ q) 포함은 비율 엄격 초과만 허용한다.
    _set_entries([SongEntry(slug="dec", ko="Dec.", ja="Dec.")])
    assert vi.match("DECO*27 - ダミーロマンス feat. 初音ミク") is None


def test_entry_title_covering_most_of_a_segment_still_matches():
    # 정당한 역방향 포함(세그 후보 "シンデレラmv" 안의 "シンデレラ", 비율 5/7 > 0.5)은 유지
    _set_entries([SongEntry(slug="cinderella-deco-27", ko="신데렐라/DECO*27", ja="シンデレラ")])
    result = vi.match("DECO*27 - シンデレラ MV")
    assert result is not None
    assert result.slug == "cinderella-deco-27"


def test_same_title_different_artist_is_disambiguated_by_query_artist_token():
    # 동명이곡(실측: シンデレラ ZIG판 vs DECO*27판) — 정확 일치가 여럿이면 쿼리의
    # 아티스트 토큰이 항목 ko/슬러그에 나타나는 쪽을 고른다.
    _set_entries([
        SongEntry(slug="cinderella-zig", ko="신데렐라/ZIG", ja="シンデレラ"),
        SongEntry(slug="cinderella-deco-27", ko="신데렐라/DECO*27", ja="シンデレラ"),
    ])
    result = vi.match("DECO*27 - シンデレラ feat. 初音ミク")
    assert result is not None
    assert result.slug == "cinderella-deco-27"


def test_same_title_without_artist_hint_keeps_deterministic_first_entry():
    _set_entries([
        SongEntry(slug="cinderella-zig", ko="신데렐라/ZIG", ja="シンデレラ"),
        SongEntry(slug="cinderella-deco-27", ko="신데렐라/DECO*27", ja="シンデレラ"),
    ])
    result = vi.match("シンデレラ")
    assert result is not None
    assert result.slug == "cinderella-zig"  # 힌트 없으면 기존 순서 유지(결정론)


# ── 슬러그 영문 별칭 (2026-08-03 실측: candy-cookie-chocolate) ──


def test_english_transliterated_title_matches_via_slug_alias():
    # 인덱스는 ko/ja만 갖는데 영상 제목이 영문 전사인 곡 — 슬러그가 그 전사다.
    _set_entries([
        SongEntry(slug="candy-cookie-chocolate", ko="캔디 쿠키 초콜릿", ja="キャンディークッキーチョコレート"),
    ])
    result = vi.match("Candy Cookie Chocolate / Hatsune Miku")
    assert result is not None
    assert result.slug == "candy-cookie-chocolate"


def test_slug_alias_participates_in_containment_pass():
    # 영문 전사 제목에 장식이 붙어 정확 일치가 깨져도 포함 매칭으로 잡힌다
    _set_entries([
        SongEntry(slug="candy-cookie-chocolate", ko="캔디 쿠키 초콜릿", ja=None),
    ])
    result = vi.match("Candy Cookie Chocolate (Official MV)")
    assert result is not None
    assert result.slug == "candy-cookie-chocolate"


def test_slug_alias_does_not_bypass_artist_token_guard():
    # 슬러그에 아티스트 구분자가 붙은 항목(cinderella-deco-27)의 별칭이 아티스트
    # 토큰("deco27")만으로 엉뚱한 곡에 붙으면 안 된다 — 기존 3중 가드가 필드
    # 무관하게 적용되는지 고정.
    _set_entries([SongEntry(slug="cinderella-deco-27", ko="신데렐라/DECO*27", ja="シンデレラ")])
    assert vi.match("DECO*27 - ダミーロマンス feat. 初音ミク") is None
