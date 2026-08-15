"""앞단(게이트웨이)이 붙이는 쿼터 헤더의 **유일한 수신부**.

이 서버는 한도를 집행하지만 한도**값**의 권위는 아니다 — 앞단이 이용자를 티어로 나눠
서로 다른 값을 준다(docs/user-quota-spec.md §6). 두 값 중 작은 쪽이 항상 먼저 걸리므로,
여기서 평평한 상수 하나로 집행하면 티어에 준 값이 그보다 클 때 **영원히 도달할 수 없다**
(앞단 trusted 60 · 여기 전원 20 → 실효 20).

🔴 **헤더 이름·형식은 아직 확정 전이다**(게이트웨이 쪽을 다른 레인이 동시에 만들고 있다).
그래서 이름·구분자·무제한 표현을 전부 이 파일 하나에 가둔다. 확정되면 **여기만** 고친다:

  ① `LIMIT_HEADER` / `ACTOR_HEADER` — 헤더 이름
  ② `_BUCKET_ALIASES` — 앞단이 쓰는 버킷 이름 → 이 서버의 버킷 이름
  ③ `_UNLIMITED_TOKENS` — 무제한을 뜻하는 표현
  ④ `parse_limit_header` — 값의 문법(현재: `k=v` 목록, 맨숫자 하나도 허용)

라우트 쪽에는 **파라미터 이름**이라는 짝이 하나 더 있다(FastAPI가 `x_lyric_limits` →
`x-lyric-limits`로 변환한다). 상수만 바꾸고 파라미터 이름을 안 바꾸면 서버는 조용히
"헤더 못 받음"으로 내려앉는다 — tests/test_user_quota.py가 라우트의 헤더 별칭과 이 상수의
일치를 못박아 그 조용한 실패를 막는다.

**의존성 없음이 계약이다**: fastapi도 settings도 임포트하지 않는다. 앞단을 우회하는 운영
스크립트(§8)가 이 파일에서 식별자 상수를 가져다 쓰기 때문이다 — 서버 스택을 끌고 들어오면
그 스크립트들이 무거워진다.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ── 헤더 이름 (게이트웨이 확정 전 잠정값) ──────────────────────────────────
#
# 이용자 식별자 — 발급 키 이용자는 불변 키 ID, 익명 이용자는 게이트웨이가 만든 솔트 해시
# (docs/user-quota-spec.md §1). 이 상수는 문서·테스트·운영 스크립트가 참조하는 SSOT다.
ACTOR_HEADER = "x-lyric-user"
# 그 이용자에게 허용된 한도값 — 위 식별자와 **같이** 온다(§6).
LIMIT_HEADER = "x-lyric-limits"


# ── 버킷 어휘 ─────────────────────────────────────────────────────────────
#
# 이 서버가 집행하는 예산 4종. GET /api/limits의 응답 필드 이름과 같다(확장이 그대로
# 그리는 이름이라 여기서 바꾸면 화면이 깨진다).
BUCKETS = ("generate", "link", "upgrade", "destructive")

# 앞단이 다른 이름을 쓸 수 있다 — 매핑을 여기서 흡수한다. 키는 소문자·비교 전 정규화된다.
_BUCKET_ALIASES = {
    "generate": "generate",
    "gen": "generate",
    "sync": "generate",
    "link": "link",
    "links": "link",
    "manual_link": "link",
    "upgrade": "upgrade",
    "depth": "upgrade",
    "destructive": "destructive",
    "reset": "destructive",
    "regenerate": "destructive",
}

# 무제한을 뜻하는 표현 — 앞단이 무엇으로 보낼지 아직 모른다. 전부 None(무제한)으로 흡수한다.
#
# **0과 음수도 무제한이다**: 이 서버는 이미 `limit <= 0`을 "이 행위의 한도 비활성"으로 쓴다
# (ServerSettings.daily_* 설명의 "0 disables the limit"). 입력 쪽에서만 0을 "0회 허용"으로
# 읽으면 같은 숫자가 두 뜻을 갖는다. **응답 쪽은 반대다** — 응답의 `limit=0`은 확장이
# `0 / 0` 소진으로 그리므로 절대 싣지 않는다(무제한은 enforced=false로 표현한다, §응답 계약).
_UNLIMITED_TOKENS = frozenset(
    {"unlimited", "unlimit", "none", "null", "inf", "infinite", "infinity", "*", "-"}
)

# `k=v` 목록의 구분자 — 앞단이 무엇을 쓸지 몰라 흔한 것을 모두 받는다.
_PAIR_SEPARATORS = ",;|"
_KEY_VALUE_SEPARATORS = "=:"


@dataclass(frozen=True)
class QuotaLimits:
    """이 요청에 적용할 버킷별 상한. 값이 ``None``이면 **무제한**이다.

    무제한의 표현을 None 하나로 못박는 이유: 0/-1/"unlimited"가 코드 안을 돌아다니면
    비교(`used >= limit`)가 조용히 뒤집힌다. 파서가 입구에서 전부 None으로 접고, 응답을
    만드는 쪽(limits.py)이 출구에서 다시 표시용 숫자로 편다.
    """

    generate: int | None
    link: int | None
    upgrade: int | None
    destructive: int | None
    # 앞단 헤더에서 **실제로 값이 온** 버킷 이름들. 나머지는 이 서버의 기본값이다 —
    # 그 사실이 로그·테스트에서 보여야 "티어를 줬는데 왜 안 먹지"를 찾을 수 있다(§6).
    from_gateway: frozenset[str] = field(default_factory=frozenset)

    def get(self, bucket: str) -> int | None:
        return getattr(self, bucket)

    @property
    def any_from_gateway(self) -> bool:
        return bool(self.from_gateway)

    @property
    def all_unlimited(self) -> bool:
        """모든 버킷이 무제한 — 이 경우에만 응답을 enforced=False로 낼 수 있다.

        일부만 무제한인 경우까지 enforced=False로 내면 **한도가 살아 있는 버킷까지**
        확장이 무제한으로 그려 429가 예고 없이 튀어나온다(확장에는 버킷별 무제한 표현이
        아예 없다 — 응답 계약).
        """
        return all(self.get(b) is None for b in BUCKETS)


def parse_limit_value(raw: object) -> int | None | str:
    """한 버킷의 값 하나를 해석한다. 정수 / ``None``(무제한) / ``str``(해석 실패 사유).

    실패를 예외가 아니라 값으로 돌려주는 이유: 앞단이 이상한 값을 보냈다고 요청을 거절하면
    §6의 "못 받으면 기본값으로 내려앉되 통과시킨다"를 어긴다. 호출부는 사유를 로그로 남기고
    기본값을 쓴다.
    """
    text = str(raw).strip().lower()
    if not text:
        return "빈 값"
    if text in _UNLIMITED_TOKENS:
        return None
    try:
        value = int(text)
    except ValueError:
        return f"숫자가 아님({text[:32]!r})"
    # 0·음수도 무제한이다 — 위 _UNLIMITED_TOKENS 주석 참고.
    return None if value <= 0 else value


def parse_limit_header(raw: str | None) -> tuple[dict[str, int | None], list[str]]:
    """앞단이 준 한도 헤더를 버킷→값으로 푼다. 반환은 (해석된 값들, 문제 사유들).

    받는 형태는 **버킷별 명시 하나**다:

      ``generate=60, destructive=8, upgrade=30, link=20``
      구분자는 `,` `;` `|`, 키/값 구분은 `=` `:` 모두 받는다.

    앞단 확정(2026-08-10, 게이트웨이 `lyricQuotaHeaders`)으로 형식이 위 하나로 굳었다.
    그전에는 "맨 숫자 하나(``60``)를 generate에만 적용"하는 추정 갈래가 있었는데 지웠다 —
    앞단이 그 형태를 보내지 않으므로 도달하지 않는 코드였고, 남겨 두면 나중에 누가 숫자
    하나를 보냈을 때 **아무 경고 없이 generate에만 붙는다**. 형식을 모르면 값을 지어내지
    않고 사유로 남기는 편이 낫다(추측 금지) — 그러면 기본값으로 내려앉고 §6의 경고가
    "티어를 줬는데 왜 안 먹지"를 찾게 해 준다.

    모르는 키는 사유로 남기고 무시한다 — 앞단이 이 서버가 모르는 버킷을 추가해도 요청이
    죽지는 않아야 한다.

    문자열이 아닌 값은 **없는 것으로 본다**: 이 리포의 서버 테스트는 라우트 코루틴을 직접
    await하므로(httpx 불사용) 안 넘긴 헤더 인자에는 FastAPI의 ``Header(None)`` 기본값
    객체가 그대로 들어온다 — HTTP로는 절대 안 생기는 값이라 방어가 아니라 하네스 정합이다.
    """
    if not isinstance(raw, str):
        return {}, []
    text = raw.strip()
    if not text:
        return {}, ["빈 헤더"]

    parsed: dict[str, int | None] = {}
    problems: list[str] = []

    has_kv = any(sep in text for sep in _KEY_VALUE_SEPARATORS)
    if not has_kv:
        # 버킷 이름이 없으면 어느 예산의 값인지 알 수 없다. 지어내지 않고 사유로 남긴다 —
        # 호출부는 기본값으로 내려앉고 §6 경고가 뜬다(조용한 오적용보다 낫다).
        return {}, [f"버킷 이름이 없는 한도 헤더: {text!r}"]

    # ① 버킷별 명시
    for chunk in _split(text, _PAIR_SEPARATORS):
        key, sep, value_text = _partition(chunk, _KEY_VALUE_SEPARATORS)
        if not sep:
            problems.append(f"키=값 형태가 아님({chunk[:32]!r})")
            continue
        bucket = _BUCKET_ALIASES.get(key.strip().lower().replace("-", "_"))
        if bucket is None:
            problems.append(f"모르는 버킷({key.strip()[:32]!r})")
            continue
        value = parse_limit_value(value_text)
        if isinstance(value, str):
            problems.append(f"{bucket}: {value}")
            continue
        parsed[bucket] = value
    return parsed, problems


def _split(text: str, separators: str) -> list[str]:
    parts = [text]
    for sep in separators:
        parts = [piece for part in parts for piece in part.split(sep)]
    return [p for p in (part.strip() for part in parts) if p]


def _partition(text: str, separators: str) -> tuple[str, str, str]:
    """가장 먼저 나오는 구분자 하나로 나눈다 — `=`와 `:`이 섞여 와도 앞선 쪽이 이긴다."""
    best: tuple[int, str] | None = None
    for sep in separators:
        idx = text.find(sep)
        if idx >= 0 and (best is None or idx < best[0]):
            best = (idx, sep)
    if best is None:
        return text, "", ""
    idx, sep = best
    return text[:idx], sep, text[idx + 1 :]


def resolve_limits(
    raw: str | None, defaults: Mapping[str, int | None], *, warn: bool = True
) -> QuotaLimits:
    """앞단 값 + 서버 기본값 → 이 요청에 집행할 상한.

    **못 받으면 기본값으로 내려앉되 요청은 거절하지 않는다**(§6) — 배포 순서(앞단 먼저 /
    여기 나중)를 유연하게 두기 위해서다. 🔴 다만 그 상태는 **티어 무력화와 같으므로** 경고를
    남긴다. 조용히 넘어가면 "티어를 줬는데 왜 안 먹지"를 아무도 못 찾는다.

    warn=False: 한도를 아예 강제하지 않는 배포(admin_api_key 미설정, 로컬 단일 사용자)에서
    쓴다 — 거기서는 앞단도 없고 무력화될 티어도 없으므로 경고가 잡음이다.
    """
    parsed, problems = parse_limit_header(raw)
    if warn:
        _warn_gaps(raw, parsed, problems)
    values = {b: _normalize_default(defaults.get(b)) for b in BUCKETS}
    values.update({b: v for b, v in parsed.items() if b in values})
    return QuotaLimits(
        generate=values["generate"],
        link=values["link"],
        upgrade=values["upgrade"],
        destructive=values["destructive"],
        from_gateway=frozenset(parsed),
    )


def _warn_gaps(raw: object, parsed: Mapping[str, int | None], problems: list[str]) -> None:
    """내려앉은 **모든** 형태를 남긴다 — 전무·일부 누락·해석 실패는 사유가 다르므로
    카운터도 따로 센다(하나가 샘플링에 걸려도 다른 하나가 묻히지 않게)."""
    if problems:
        warn_limits_unavailable("값 해석 실패", detail="; ".join(problems[:4]))
    missing = [b for b in BUCKETS if b not in parsed]
    if not missing:
        return
    if not isinstance(raw, str):
        warn_limits_unavailable("헤더 없음")
    elif not parsed:
        warn_limits_unavailable("헤더에서 쓸 값을 못 얻음", detail=f"raw={raw.strip()[:64]!r}")
    else:
        warn_limits_unavailable("일부 버킷 누락", detail=",".join(missing))


def _normalize_default(value: int | None) -> int | None:
    """서버 기본값도 같은 규칙으로 접는다 — `0`(한도 비활성)은 None(무제한)이다."""
    if value is None:
        return None
    return None if value <= 0 else int(value)


# ── 내려앉음 경고 (§6) ────────────────────────────────────────────────────
#
# 요청마다 warning을 뱉으면 로그가 그 한 줄로 덮여 다른 사고가 안 보인다. 그렇다고 완전
# 1회성으로 두면 로테이션 뒤에는 증거가 사라진다. 처음 1회 + 이후 샘플링으로 둘 다 피한다.
_WARN_EVERY = 500
_warn_counts: dict[str, int] = {}


def warn_limits_unavailable(reason: str, *, detail: str = "") -> None:
    """앞단 한도값 없이 기본값으로 내려앉았다는 사실을 남긴다(처음 1회 + 이후 샘플링)."""
    count = _warn_counts.get(reason, 0) + 1
    _warn_counts[reason] = count
    if count != 1 and count % _WARN_EVERY:
        return
    logger.warning(
        "앞단 한도값(%s)을 쓰지 못해 서버 기본값으로 집행합니다 — 티어별 차등이 무력화된 "
        "상태입니다. 사유=%s%s (누적 %d회, 이 경고는 %d회마다 반복)",
        LIMIT_HEADER,
        reason,
        f" 상세={detail}" if detail else "",
        count,
        _WARN_EVERY,
    )


def reset_limit_warning_state() -> None:
    """테스트 전용 — 경고 샘플링 상태를 비운다(프로세스 수명 동안 누적되는 값이라
    테스트끼리 간섭한다)."""
    _warn_counts.clear()


# ── 운영 작업 식별자 (§8) ─────────────────────────────────────────────────
#
# 앞단을 우회해 이 서버를 직접 부르는 운영 스크립트(대량 인제스트·재생성 검증·벤치)는
# 이용자 식별자가 없어 §1의 fail-closed에 걸린다. 스크립트마다 **사람과 구분되는** 고유
# 식별자를 쓴다 — 사람의 예산에 섞이면 통계가 오염된다.
#
# 🔴 "어드민 키면 식별자 없이 통과"로 풀지 않는다(§8) — 그것은 §5가 고치는 "면제라서
# 기록이 없다"를 운영 작업에 대해 되살리는 것이다. **면제하더라도 기록은 남는다.**
OPS_ACTOR_PREFIX = "ops:"

OPS_ACTOR_BULK_INGEST = f"{OPS_ACTOR_PREFIX}bulk_ingest"
OPS_ACTOR_VERIFY_REGEN = f"{OPS_ACTOR_PREFIX}verify_regen"
OPS_ACTOR_VERIFY_DEPLOY = f"{OPS_ACTOR_PREFIX}verify_deploy"
OPS_ACTOR_BENCH_REGEN = f"{OPS_ACTOR_PREFIX}bench_vg_regen"


def is_ops_actor(actor: str | None) -> bool:
    """운영 작업 식별자인가 — 상한은 면제하되 기록은 남기는 대상이다.

    **위조 가능성에 대하여**: 이 헤더는 게이트웨이가 붙이는 값이고, 서버는 게이트웨이
    뒤에만 있어야 한다(§1). 게이트웨이는 이 헤더를 이용자 요청에서 그대로 통과시키지 않고
    항상 자기가 계산한 값으로 덮어써야 한다 — 그렇지 않으면 `ops:` 를 스스로 붙인 요청이
    상한을 사는다. 직접 도달 경로는 배포의 전역 X-API-Key 미들웨어(main.require_api_key)가
    한 겹 더 막는다. 이 층에서 더 조일 수단은 없으므로 전제를 여기 적어 둔다.
    """
    return bool(actor) and actor.strip().lower().startswith(OPS_ACTOR_PREFIX)
