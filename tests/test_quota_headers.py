"""앞단(게이트웨이) 쿼터 헤더 수신부 단위 테스트 — everyric2/server/api/quota_headers.py.

SSOT: docs/user-quota-spec.md §6(한도값은 앞단이 알려준다) · §8(운영 작업 식별자).
집행·표시에 실제로 반영되는지는 test_user_quota.py가 라우트 경로로 못박는다. 여기서는
**해석 자체**를 못박는다 — 게이트웨이가 확정되면 이 파일이 "무엇이 바뀌어도 되는가"의
경계선이 된다.

여기서 못박는 계약:
  ① 버킷별 명시(`k=v`)와 맨 숫자 하나를 모두 받는다.
  ② 무제한 표현은 전부 None 하나로 접힌다 — 0·음수·문자 토큰 모두.
  ③ 안 준 버킷은 서버 기본값으로 내려앉고, 서버 기본값의 0도 None(무제한)이다.
  ④ 못 받거나 못 알아들으면 **거절하지 않고** 기본값으로 내려앉되 **경고가 남는다**
     (조용한 티어 무력화 금지). 경고는 첫 1회 + 이후 샘플링이라 로그를 덮지 않는다.
  ⑤ all_unlimited는 **전부** 무제한일 때만 참이다 — 일부만 무제한인데 참이면 확장이
     한도가 살아 있는 버킷까지 무제한으로 그린다.
  ⑥ 운영 작업 식별자는 접두사로 판별한다(§8).
"""

import logging

import pytest

from everyric2.server.api.quota_headers import (
    ACTOR_HEADER,
    BUCKETS,
    LIMIT_HEADER,
    OPS_ACTOR_BULK_INGEST,
    is_ops_actor,
    parse_limit_header,
    reset_limit_warning_state,
    resolve_limits,
)

# 이 서버의 기본값 자리 — 실제 값은 sync._default_limits가 댄다.
DEFAULTS = {"generate": 20, "link": 20, "upgrade": 10, "destructive": 4}


@pytest.fixture(autouse=True)
def _fresh_warn_state():
    """경고 샘플링 상태는 프로세스 수명 동안 누적된다 — 테스트끼리 간섭하지 않게 비운다."""
    reset_limit_warning_state()
    yield
    reset_limit_warning_state()


# ── ① 형식 ─────────────────────────────────────────────────────────


def test_bucket_pairs_are_parsed():
    parsed, problems = parse_limit_header("generate=60, destructive=8; upgrade:30 | link=25")
    assert parsed == {"generate": 60, "destructive": 8, "upgrade": 30, "link": 25}
    assert problems == []


def test_a_bare_number_is_refused_instead_of_guessed():
    """버킷 이름이 없는 값은 **어느 예산인지 알 수 없다** — 지어내지 않는다.

    앞단 확정 전에는 맨 숫자를 generate로 해석하는 갈래가 있었으나 지웠다. 그 형태를
    보내는 앞단이 없어 도달하지 않는 코드였고, 남겨 두면 나중에 누가 숫자 하나를 보냈을 때
    경고 없이 generate에만 붙는다. 지금은 전 버킷이 기본값으로 내려앉고 사유가 남는다.
    """
    parsed, problems = parse_limit_header("60")
    assert parsed == {}
    assert problems and "버킷 이름이 없는" in problems[0]

    limits = resolve_limits("60", DEFAULTS)
    assert (limits.generate, limits.destructive, limits.upgrade, limits.link) == (
        DEFAULTS["generate"],
        DEFAULTS["destructive"],
        DEFAULTS["upgrade"],
        DEFAULTS["link"],
    )
    assert limits.from_gateway == frozenset()


def test_alias_and_case_are_absorbed():
    parsed, problems = parse_limit_header("GEN=5,Manual-Link=7,RESET=2")
    assert parsed == {"generate": 5, "link": 7, "destructive": 2}
    assert problems == []


def test_unknown_bucket_is_reported_but_does_not_kill_the_request():
    parsed, problems = parse_limit_header("generate=5,quantum=9")
    assert parsed == {"generate": 5}
    assert problems and "quantum" in problems[0]


# ── ② 무제한 표현 ───────────────────────────────────────────────────


@pytest.mark.parametrize("token", ["unlimited", "none", "inf", "*", "-1", "0", "-5"])
def test_unlimited_tokens_all_fold_to_none(token):
    """무제한의 표현을 None 하나로 못박는다 — 0/-1/문자열이 코드 안을 돌아다니면
    비교(`used >= limit`)가 조용히 뒤집힌다."""
    parsed, problems = parse_limit_header(f"generate={token}")
    assert parsed == {"generate": None}
    assert problems == []


def test_zero_from_the_server_default_is_also_unlimited():
    """설정의 `0 disables the limit`과 같은 뜻으로 접는다 — 같은 숫자가 두 뜻을 가지면 안 된다."""
    limits = resolve_limits(None, {**DEFAULTS, "upgrade": 0})
    assert limits.upgrade is None
    assert limits.generate == 20


# ── ③ 내려앉기 ──────────────────────────────────────────────────────


def test_missing_header_falls_back_to_every_server_default():
    limits = resolve_limits(None, DEFAULTS)
    assert [limits.get(b) for b in BUCKETS] == [DEFAULTS[b] for b in BUCKETS]
    assert limits.from_gateway == frozenset()


def test_partial_header_only_overrides_what_it_carries():
    limits = resolve_limits("destructive=9", DEFAULTS)
    assert limits.destructive == 9
    assert limits.generate == 20
    assert limits.from_gateway == frozenset({"destructive"})


# ── ④ 경고 (조용한 티어 무력화 금지) ─────────────────────────────────


def test_missing_header_logs_a_warning(caplog):
    with caplog.at_level(logging.WARNING):
        resolve_limits(None, DEFAULTS)
    assert any(LIMIT_HEADER in r.getMessage() for r in caplog.records), caplog.text


def test_unparseable_header_logs_a_warning_and_still_falls_back(caplog):
    with caplog.at_level(logging.WARNING):
        limits = resolve_limits("완전히 이상한 값", DEFAULTS)
    assert limits.generate == 20  # 거절하지 않는다 — 배포 순서를 유연하게 두기 위해
    assert caplog.records, "해석 실패가 조용히 넘어갔다"


def test_partial_header_logs_a_warning_about_the_missing_buckets(caplog):
    with caplog.at_level(logging.WARNING):
        resolve_limits("generate=60", DEFAULTS)
    text = caplog.text
    assert "누락" in text and "destructive" in text, text


def test_warning_is_sampled_not_emitted_every_request(caplog):
    """요청마다 뱉으면 로그가 그 한 줄로 덮여 다른 사고가 안 보인다 — 첫 1회 + 샘플링."""
    with caplog.at_level(logging.WARNING):
        for _ in range(50):
            resolve_limits(None, DEFAULTS)
    assert len(caplog.records) == 1, [r.getMessage() for r in caplog.records]


def test_complete_header_logs_nothing(caplog):
    with caplog.at_level(logging.WARNING):
        resolve_limits("generate=60,link=25,upgrade=30,destructive=8", DEFAULTS)
    assert caplog.records == []


def test_warn_false_stays_silent(caplog):
    """한도를 아예 강제하지 않는 배포(로컬 단일 사용자)에는 무력화될 티어가 없다."""
    with caplog.at_level(logging.WARNING):
        resolve_limits(None, DEFAULTS, warn=False)
    assert caplog.records == []


# ── ⑤ all_unlimited ────────────────────────────────────────────────


def test_all_unlimited_requires_every_bucket():
    partial = resolve_limits("generate=unlimited", DEFAULTS, warn=False)
    assert partial.all_unlimited is False  # 나머지는 유한하다
    full = resolve_limits(
        "generate=unlimited,link=unlimited,upgrade=unlimited,destructive=unlimited",
        DEFAULTS,
        warn=False,
    )
    assert full.all_unlimited is True


# ── ⑥ 운영 작업 식별자 ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "actor,expected",
    [
        (OPS_ACTOR_BULK_INGEST, True),
        ("ops:whatever", True),
        ("  ops:spaced  ", True),
        ("user-abc", False),
        ("", False),
        (None, False),
        # 사람 식별자가 우연히 ops로 시작할 수는 있어도 접두사 구분자(:)까지 맞을 수는 없다
        ("opsuser-123", False),
    ],
)
def test_ops_actor_detection(actor, expected):
    assert is_ops_actor(actor) is expected


def test_header_names_are_lowercase_for_alias_comparison():
    """라우트 별칭 비교(test_user_quota)가 소문자로 이뤄지므로 상수도 소문자여야 한다."""
    assert ACTOR_HEADER == ACTOR_HEADER.lower()
    assert LIMIT_HEADER == LIMIT_HEADER.lower()
