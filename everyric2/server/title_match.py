"""제목 정규화·후보 매칭 유틸 (곡 인덱스 매칭과 커버 링크 후보 탐색이 공유).

원래 vocaro_index에만 있던 정규화/후보 생성 규칙을 여기로 옮겼다. 규칙 자체는
everyric2-chrome/src/lib/vocaro.ts의 findMatch와 동일 기준을 유지한다:
정규화 정확 일치 → 상호 포함 + 길이비 0.5 이상.

**이 모듈의 결과는 "후보 발견"에만 쓴다.** 커버/원곡이 실제로 같은 곡인지의 최종 판정은
반주 상관 검증(link-jobs)이 담당한다 — 제목만으로 SyncLink를 만들지 않는다. 그래서
매칭 규칙이 다소 헐거워도 안전하다(오탐의 대가는 검증 잡 한 번).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from typing import TypeVar

# 유튜브 곡 제목 관례("곡명 / 아티스트", "곡명 - 아티스트 feat.가수", "【가수】곡명" 등)의
# 구분자 — 하이픈은 양옆 공백이 있을 때만 분리해 합성어를 보존한다.
# ㅣ(U+3163, 한글 자모 "이")는 한국어 커버 제목이 세로줄(｜) 대신 흔히 쓰는 장식 구분자다
# (실측: 「【MV】 로키 (ROKI) ㅣ한국어 Coverㅣ【레볼루션 하트】」) — 반각/전각 세로줄과 코드
# 포인트가 아예 달라 예전 문자 집합에 없었다. 이게 빠지면 "로키"가 뒤따르는 장식(한국어
# Cover 등)과 통째로 붙어 다른 커버 제목("로키" 단독 후보를 내는)과 매칭이 안 된다
# (실측: 두 실제 한국어 커버 제목끼리 score=None → ㅣ 추가 후 1.0).
_TITLE_SPLIT_RE = re.compile(r"\s*[/／|｜ㅣ–—―~〜]\s*|\s+-\s+|\s*[「」『』【】\[\]]\s*")
_FEAT_RE = re.compile(r"(?:^|\s)(?:feat|ft)\.?\s*\S.*$", re.IGNORECASE)

# 「곡명 cover by 이름」/「곡명 covered by 이름」— 구분자 없이 곡명 뒤에 바로 붙는 커버
# 업로더 병기(실측, 2026-08). 단순 토큰 제거(_NOISE_TOKEN_RE의 cover(?:ed)?\s*by)만으로는
# "이름"이 남아 곡명에 알파벳으로 융합된다(둘 다 alnum이라 normalize_title이 공백만
# 지워도 안 떨어진다 — «첫사랑 cover by 홍길동» → «첫사랑홍길동», 길이비 0.5로 기본
# 임계값 0.6에 미달해 실패). feat. 접미(_FEAT_RE)와 같은 "표시부터 끝까지 통째로 버린다"
# 전략을 따른다 — cover by 뒤에 곡명이 다시 오는 관례는 사실상 없다(feat.와 같은 전제).
_COVER_BY_RE = re.compile(r"\bcover(?:ed)?\s*by\s+\S.*$", re.IGNORECASE)

# 괄호로 묶인 가수/독음 병기 («【初音ミク】곡명», «곡명 (아쿠노)») — 통째로 걷어낸 변형도 후보에 넣는다
_BRACKETED_RE = re.compile(r"【[^】]*】|「[^」]*」|『[^』]*』|\[[^\]]*\]|\([^)]*\)|（[^）]*）")

# 업로드 관례 잡토큰 — 커버/인스트/공식MV가 같은 곡을 서로 다른 제목으로 올리는 주범.
# 곡명 자체에 들어갈 일이 거의 없는 표현만 넣는다(제거가 곡명을 깎으면 후보를 놓친다).
#
# 커버 표기 확장(실측, 2026-07): カバー(가타카나 cover)가 빠져 있었다 — 일본어 업로드
# 관례에서 「歌ってみた」만큼 흔한 표기다. 한국어 「불러봤다」는 축약 활용형 하나만 잡아
# 「불러보았다」(문어체 과거, 비축약)·「불러봤음」(축약+명사형)·「불러본」(관형형)·
# 「불러봄」(명사형)·「불러보는」(현재 관형형) 등을 놓쳤다 — 인스타그램 해시태그
# (#불러보았다)·니코니코 문화("歌ってみた"의 한국어 번역어) 실측으로 확인한 실제 활용형들이다.
# 부르다+아/어보다(시도) 활용은 축약(보다→봤다) 여부로 어간 표면형이 갈린다("보"/"봤"는
# 서로 다른 코드포인트라 접두 매칭 하나로 못 묶는다) — 그래서 두 어간 계열(보-/봤-)을
# 각각 열거한다. 곡명 자체에 이 어간이 들어갈 일은 사실상 없다(기존 방침과 같은 근거).
#
# cover(?:ed)?\s*by (실측, 2026-08): 「covered by」만 잡고 「cover by」(과거형 -ed 없는
# 표기, 유튜브에 「제목 Cover by 업로더」류로 흔하다)는 못 잡았다 — \bcover\b가 "cover"
# 단어만 지우고 뒤따르는 "by 업로더"가 곡명에 그대로 들러붙어(둘 다 알파벳이라
# normalize_title이 공백만 지우면 서로 붙는다) 원제와의 정확 매칭이 깨졌다. by를
# cover(?:ed)?의 선택적 접미로 묶어 두 표기를 한 알터네이션으로 흡수한다.
_NOISE_TOKEN_RE = re.compile(
    r"official(?:\s*(?:music|lyric)s?)?(?:\s*video|\s*audio|\s*mv)?"
    r"|music\s*video|lyrics?\s*video|audio\s*only"
    r"|\bmv\b|\bpv\b|\bhd\b|\bhq\b|\b4k\b|\b1080p\b|\b720p\b"
    r"|full\s*ver(?:sion)?\.?|short\s*ver(?:sion)?\.?|tv\s*size"
    r"|off\s*vocal|instrumental|\binst\b|karaoke|cover(?:ed)?\s*by|\bcover\b"
    r"|カラオケ|カバー|歌ってみた|唄ってみた|弾いてみた|叩いてみた|踊ってみた"
    r"|オリジナル曲?|本家|ボカロ"
    r"|커버|불러(?:보다|봄|본|보는|보았\S*|봤\S*)|한글\s*자막|한국어\s*자막|번역\s*자막",
    re.IGNORECASE,
)

T = TypeVar("T")


def normalize_title(title: str) -> str:
    """NFKC 정규화 + 소문자 + 영숫자/한글/가나/한자만 남기기 (공백·기호 전부 제거)."""
    t = unicodedata.normalize("NFKC", title.lower())
    return "".join(ch for ch in t if ch.isalnum())


def strip_noise_tokens(title: str) -> str:
    """MV/Official/Cover/歌ってみた 류 업로드 잡토큰 제거 (곡명 자체는 건드리지 않는다).

    제거 전에 NFKC로 정규화한다 — 장식 서체(수학 산세리프 «𝖢𝖮𝖵𝖤𝖱» 류)는 NFKC를 거쳐야
    ASCII cover가 되어 패턴에 걸린다. 실측(unite 2026-07-29): 정규화 전 제거 순서 탓에
    «𝖢𝖮𝖵𝖤𝖱» 잡토큰이 살아남아 헛 후보를 만들었다. 공백은 보존하므로 \\b 경계는 유효하다.

    cover(ed) by는 뒤따르는 업로더 이름까지 먼저 통째로 지운다(_COVER_BY_RE) — 그래야
    「곡명 cover by 이름」처럼 구분자가 없는 표기에서 이름이 곡명에 융합되지 않는다.
    이 뒤에 _NOISE_TOKEN_RE가 나머지 잡토큰(및 이름 없이 남은 bare cover by)을 처리한다.
    """
    t = unicodedata.normalize("NFKC", title)
    return _NOISE_TOKEN_RE.sub(" ", _COVER_BY_RE.sub(" ", t))


def _is_pure_noise(raw: str) -> bool:
    """조각이 잡토큰만으로 이루어졌는가 — «MV»·«Official MV»·«𝖢𝖮𝖵𝖤𝖱» 류.

    이런 조각이 후보로 살아남으면 «[MV] A곡»과 «[MV] B곡»이 mv==mv로 만점 매칭된다
    (실측 unite 2026-07-29: 매칭 조각 mv·officialmv). normalize_title이 공백을 지운 뒤에는
    \\b 경계가 무력해 정규식으로 못 거르므로, 공백이 살아 있는 원시 조각 단계에서 거른다."""
    return not normalize_title(strip_noise_tokens(raw))


def candidate_queries(title: str, drop_noise: bool = False) -> list[str]:
    """풀 제목에서 곡명 후보를 정규화 형태로 생성.

    순서: 원문 → feat 제거 → 괄호 세그먼트 제거 → (drop_noise면 잡토큰 제거 변형) →
    각 변형의 구분자 조각(왼쪽 우선). 괄호 제거 변형을 조각보다 먼저 두어
    «【가수】곡명» 류에서 가수명이 곡명보다 먼저 매칭되는 오탐을 막는다.

    반환 순서가 곧 우선순위다 — 호출부는 앞쪽 후보의 매칭을 더 신뢰한다.
    drop_noise 기본값 False는 곡 인덱스 매칭의 기존 동작을 그대로 보존한다.
    """
    seen: set[str] = set()
    out: list[str] = []

    def add(raw: str) -> None:
        q = normalize_title(raw)
        if len(q) >= 2 and q not in seen and not (drop_noise and _is_pure_noise(raw)):
            seen.add(q)
            out.append(q)

    stripped = _BRACKETED_RE.sub(" ", title)
    variants = [title, _FEAT_RE.sub("", title), stripped, _FEAT_RE.sub("", stripped)]
    if drop_noise:
        variants = variants + [strip_noise_tokens(v) for v in variants]
    for v in variants:
        add(v)
    for v in variants:
        for part in _TITLE_SPLIT_RE.split(v):
            add(part)
            add(_FEAT_RE.sub("", part))
    return out


def _score_query_lists(qa: list[str], qb: list[str]) -> tuple[float, int] | None:
    """후보 리스트 두 개의 유사도 (score, priority). match_score/rank_matches가 공유한다."""
    best: tuple[float, int] | None = None
    for i, x in enumerate(qa):
        for j, y in enumerate(qb):
            if x == y:
                cand = (1.0, i + j)
            elif x in y or y in x:
                ratio = min(len(x), len(y)) / max(len(x), len(y))
                if ratio < 0.5:
                    continue
                cand = (ratio, i + j)
            else:
                continue
            if best is None or (-cand[0], cand[1]) < (-best[0], best[1]):
                best = cand
    return best


def match_score(a: str, b: str, drop_noise: bool = True) -> tuple[float, int] | None:
    """두 제목의 유사도 (score, priority). 매칭이 없으면 None.

    score: 1.0 = 어떤 후보 변형끼리 정규화 정확 일치, 0.5~1.0 = 상호 포함 시 길이비.
    priority: 매칭된 두 후보의 우선순위 인덱스 합 (작을수록 원제에 가까운 변형끼리 맞음).
    동점 정렬의 tie-break용 — 가수명 조각끼리 우연히 겹친 매칭을 뒤로 민다.

    주의: «ARTIST - 곡명» 관례에서는 아티스트 조각이 제목 앞머리라 priority가 오히려 곡명
    조각보다 낮다(실측 unite 2026-07-29: deco27 매칭 priority 4 < 진짜 곡명 12). 이 함수
    단독으로는 그 오탐을 못 거른다 — 코퍼스가 있는 rank_matches의 문서빈도 억제가 담당한다.
    """
    return _score_query_lists(
        candidate_queries(a, drop_noise=drop_noise), candidate_queries(b, drop_noise=drop_noise)
    )


def rank_matches(
    query_title: str,
    entries: Sequence[tuple[T, str]],
    min_score: float = 0.6,
    limit: int = 5,
    max_fragment_df: int | None = 4,
) -> list[tuple[T, float]]:
    """(키, 제목) 목록을 질의 제목과의 유사도로 정렬해 상위 limit개 반환.

    코퍼스가 작아(수십~수백 곡) 전수 스캔으로 충분하다 — 인덱스/근사 검색은 넣지 않았다.

    max_fragment_df: 문서빈도(DF) 억제 — 이보다 많은 항목이 공유하는 조각은 매칭에서 뺀다.
    실패 조각 세 부류(프로듀서명 deco27·稲葉曇, 가수명 初音ミク, 잡토큰 잔재)는 성질이
    같다: **코퍼스에서 여러 곡이 공유하는 조각**이다. 사전 관리 없이 한 규칙으로 덮는다.
    실측(unite 2026-07-29, 코퍼스 634곡): DF≤4에서 헛 후보 20→13건·정탐 손실 0.
    각 제목의 첫 후보(전체 정규화)는 DF와 무관하게 남겨 정확 일치 경로는 잃지 않는다.
    None이면 끈다(기존 동작).
    """
    prepared: list[tuple[int, T, list[str]]] = []
    for order, (key, title) in enumerate(entries):
        if not title:
            continue
        prepared.append((order, key, candidate_queries(title, drop_noise=True)))

    qa = candidate_queries(query_title, drop_noise=True)
    if max_fragment_df is not None:
        df: dict[str, int] = {}
        for _, _, qs in prepared:
            for q in set(qs):
                df[q] = df.get(q, 0) + 1

        def prune(qs: list[str]) -> list[str]:
            return [q for i, q in enumerate(qs) if i == 0 or df.get(q, 0) <= max_fragment_df]

        prepared = [(order, key, prune(qs)) for order, key, qs in prepared]
        qa = prune(qa)

    scored: list[tuple[float, int, int, T]] = []
    for order, key, qb in prepared:
        hit = _score_query_lists(qa, qb)
        if hit is None or hit[0] < min_score:
            continue
        score, priority = hit
        # 정렬 키: 점수 내림차순 → 우선순위 오름차순 → 입력 순서(최신 우선) 유지
        scored.append((-score, priority, order, key))
    scored.sort(key=lambda row: row[:3])
    return [(key, round(-neg_score, 4)) for neg_score, _, _, key in scored[:limit]]
