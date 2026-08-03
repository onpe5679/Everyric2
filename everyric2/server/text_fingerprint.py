"""번역 레이어의 지문(fingerprint) — 가사 원문이 같으면 재생성에도 번역이 살아남게 하는 키.

`normalize_line`은 worker.py의 동명 내부 함수(`_normalize_line`)를 그대로 복사한 것이다
(worker는 아직 이 모듈을 임포트하지 않는다 — 순환 임포트 없이 worker를 이 모듈로 전환하는
것은 후속 작업). 두 구현이 갈라지면 TranslationLayer.fingerprint가 worker의 line_meta
매칭 키와 어긋나 저장된 번역을 못 찾는 사고가 난다 — 고칠 때는 반드시 양쪽을 함께 고친다.
"""

import hashlib
import re
import unicodedata


def normalize_line(s: str) -> str:
    """라인 매칭용 정규화 키 — 유니코드 호환 정규화(NFKC) + 서식문자 제거 + 공백 전면 제거.

    라인 메타(발음/번역)는 가사 원문과 별도 경로로 들어와 표기가 미세하게 어긋난다.
    공백만 접던 예전 규칙은 구두점 앞뒤 띄어쓰기 차이(``Are you ready ?`` vs
    ``Are you ready?``)나 전각/반각 차이(``！`` vs ``!``)를 다른 라인으로 취급해
    실측 6줄이 메타를 못 받았다. NFKC가 전각/반각·호환 문자를 접고, 공백을 전부
    지워 띄어쓰기 위치 차이를 흡수한다.
    **구두점 자체는 지우지 않는다** — 지우면 ``行く。``와 ``行く？``처럼 부호만 다른
    별개 라인이 같은 키로 뭉쳐 엉뚱한 메타가 붙는다(과잉 정규화 위험).
    """
    t = unicodedata.normalize("NFKC", s)
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Cf")  # ZWSP·BOM 등 서식문자
    return re.sub(r"\s+", "", t)


def lines_fingerprint(texts: list[str]) -> str:
    """세그먼트 원문 텍스트 목록의 md5 32hex — 정규화 라인들을 "\\n"으로 이어 해시한다.

    normalize_line과 같은 정규화를 쓰므로, worker의 line_meta 매칭에서 같은 라인으로
    취급되는 표기 차이(공백·전각/반각)는 지문도 바꾸지 않는다.
    """
    joined = "\n".join(normalize_line(t) for t in texts)
    return hashlib.md5(joined.encode("utf-8")).hexdigest()


# ── 느슨한 매칭 키 — 엄격 키(normalize_line)가 못 찾을 때만 쓰는 폴백 ────────────
#
# 실측(2026-08 OVwCr2MESfo·vg6pnvn1u10): miraheze/vocaro 두 출처가 같은 줄을 구두점
# 유무만 다르게 적는다 — "I love you." vs "I love you", 줄바꿈 위치가 달라 이어붙인
# 결과가 "…なあなたのこと" vs "…な あなたのこと"(공백은 normalize_line이 이미 지운다)
# 처럼 되는 경우. 엄격 키는 구두점을 보존한다(설계 근거: "行く。"≠"行く？"처럼 뜻이
# 갈리는 구두점 쌍을 지키기 위해서다 — normalize_line 문서 참고). 그래서 구두점만
# 다른 줄은 엄격 키로 영원히 못 붙는다.
#
# 이 폴백에서 값이 다른 두 줄이 우연히 같아져도 실사용 피해가 없다 — **엄격 키가 항상
# 먼저 시도되고, 정확히 갈리는 구두점 쌍은 둘 다 값이 있어 엄격 키가 이미 각각 정확히
# 집어내므로 이 폴백까지 오지 않는다.** 폴백은 "한쪽에만 있어서 엄격 매칭이 원천적으로
# 실패하는" 줄에만 실제로 쓰인다.
_LOOSE_PUNCT_RE = re.compile(
    "[" + ".,!?" + "、。！？…‥・" + "「」『』（）()" + "‘’“”" + "'\"" + "]+"
)
# 정규화 후 이 길이 미만이면 매칭에서 제외한다 — 감탄사·기호뿐인 짧은 줄이 무관한 다른
# 짧은 줄과 우연히 같은 키가 되는 것을 막는다.
_MIN_LOOSE_KEY_LEN = 2


def loose_normalize_line(s: str) -> str | None:
    """엄격 키(``normalize_line``)가 못 찾을 때만 시도하는 폴백 키 — 구두점까지 지운다.

    정규화 후 ``_MIN_LOOSE_KEY_LEN`` 미만이면 None(매칭 불가 — 우연 일치 방지).
    """
    t = _LOOSE_PUNCT_RE.sub("", normalize_line(s))
    return t if len(t) >= _MIN_LOOSE_KEY_LEN else None


def _lines_match(strict_a: str, loose_a: str | None, strict_b: str, loose_b: str | None) -> bool:
    """엄격 키가 같거나, 둘 다 유효한 느슨한 키가 있고 그것이 같으면 매칭."""
    return strict_a == strict_b or (loose_a is not None and loose_a == loose_b)


# ── 순서 정렬 매칭 — 줄 분할이 다른 두 원문 시퀀스에서 번역을 옮겨 온다 ─────────
#
# 실증(H7PR): miraheze 기반 새 싱크(36줄)와 vocaro 위키 페이지(42줄)의 줄 분할이 달라
# normalize_line 기준 줄 단위 동등 매칭이 50%를 못 넘겼다 — 위키에 번역이 있는데도 LLM이
# 다시 불려 저품질 번역이 고착됐다. 두 소스가 **같은 곡을 같은 순서로** 담고 있다는 전제
# 아래, 문장 결합(합침/쪼갬)만 감안하면 되므로 완전한 편집거리 DP는 필요 없다.

# 합침/쪼갬 결합 탐색 상한 — 한 줄이 이보다 많은 조각으로 쪼개지는 정상 가사는 드물다
# (제목·크레딧 블록이 우연히 이어붙는 오탐을 여기서 자른다).
_ALIGN_MAX_COMBINE = 6

# 재동기화 앵커 탐색 상한(양쪽 각각) — 그 이상 떨어진 완전 일치는 우연의 일치일 위험이
# 커 신뢰하지 않는다. 곡 하나가 이 창보다 크게 밀리는 경우는 애초에 다른 곡이라고 본다.
_ALIGN_RESYNC_WINDOW = 12


def _find_combine(
    norm: list[str],
    loose: list[str | None],
    start: int,
    target: str,
    target_loose: str | None,
    length: int,
) -> int | None:
    """``norm[start:start+k]``를 이어붙인 것이 ``target``과 같아지는 최소 k(>=2)를 찾는다.

    엄격 결합(``norm``)이 같아지면 즉시 채택한다. 아니면 구두점을 지운 결합(``loose``)도
    함께 시도한다 — 조각 중 느슨한 키가 없는(``None``, 너무 짧은) 것이 섞이면 그 k의
    느슨한 비교는 건너뛴다. 못 찾으면 None. ``_ALIGN_MAX_COMBINE``까지만 시도하고,
    엄격 결합 길이가 이미 target을 넘어서면 더 붙여도 같아질 수 없으므로 조기 종료한다."""
    acc = norm[start]
    acc_loose: str | None = loose[start]
    limit = min(_ALIGN_MAX_COMBINE, length - start)
    for k in range(2, limit + 1):
        acc += norm[start + k - 1]
        piece_loose = loose[start + k - 1]
        acc_loose = None if (acc_loose is None or piece_loose is None) else acc_loose + piece_loose
        if acc == target:
            return k
        if target_loose is not None and acc_loose is not None and acc_loose == target_loose:
            return k
        if len(acc) > len(target):
            break
    return None


def _find_resync_anchor(
    seg_norm: list[str],
    wiki_norm: list[str],
    seg_loose: list[str | None],
    wiki_loose: list[str | None],
    i: int,
    j: int,
) -> tuple[int, int] | None:
    """``i``·``j`` 이후 제한된 창 안에서 다음 일치 지점(엄격 또는 느슨)을 찾는다.

    총 이동 거리(양쪽에서 건너뛰는 줄 수의 합)가 가장 작은 지점을 고른다 — 국소적으로
    가장 그럴듯한 재동기화 지점이다. 창 안에 없으면 None(포기 — 남은 세그는 전부
    매칭 없이 끝난다)."""
    n, m = len(seg_norm), len(wiki_norm)
    best: tuple[int, int] | None = None
    best_cost: int | None = None
    for ni in range(i, min(i + _ALIGN_RESYNC_WINDOW, n)):
        for nj in range(j, min(j + _ALIGN_RESYNC_WINDOW, m)):
            if _lines_match(seg_norm[ni], seg_loose[ni], wiki_norm[nj], wiki_loose[nj]):
                cost = (ni - i) + (nj - j)
                if best_cost is None or cost < best_cost:
                    best_cost, best = cost, (ni, nj)
    return best


def align_translation_lines(
    seg_texts: list[str], wiki_lines: list[dict[str, str]]
) -> list[str | None]:
    """세그 원문 순서를 기준으로, 다른 줄 분할(위키 등)의 원문·번역 쌍에서 정렬된 번역을
    뽑아낸다. 반환은 ``seg_texts``와 길이가 같다 — 매칭된 번역 문자열, 매칭 실패는 None.

    같은 곡·같은 순서라는 전제 아래 DP 없이 투 포인터로 정렬한다:

    1. 정규화(``normalize_line``) 후 정확히 같으면 그대로 대응.
    2. **위키가 합침** — 위키 한 줄이 세그 여러 줄의 연결과 같으면, 그 번역을 **첫
       세그에만** 배정하고 나머지는 None(같은 번역을 여러 줄에 중복 배정하지 않는다).
    3. **위키가 쪼갬** — 세그 한 줄이 위키 여러 연속 줄의 연결과 같으면, 그 줄들의
       번역을 이어 붙여(구분자 없이 연결) 그 세그 하나에 배정한다.
    4. 어느 쪽도 아니면(한쪽에만 있는 여분의 줄 등) **재동기화**한다 — 두 포인터를
       각각 제한된 창(``_ALIGN_RESYNC_WINDOW``) 안에서 훑어 다음 완전 일치 지점을
       찾고, 그 사이 세그는 매칭 없이 건너뛴다(위키에만 있던 여분 줄도 버려진다).
       앵커를 못 찾으면 남은 세그는 전부 None으로 포기한다.

    ``wiki_lines``는 ``TranslationLayer.lines``와 같은 모양 — ``[{"text": 원문,
    "translation": 번역}, ...]``. 이 함수는 원문 텍스트끼리만 정렬을 판단하고 번역
    내용 자체는 결과를 그대로 옮기는 데만 쓴다.

    커버리지(반환값 중 None이 아닌 비율)로 이 정렬이 쓸 만한지 판정하는 것은 호출부의
    몫이다(저장 문턱·이관 문턱이 서로 다를 수 있어 이 함수는 임계값을 모른다).
    """
    n = len(seg_texts)
    if n == 0:
        return []
    m = len(wiki_lines)
    result: list[str | None] = [None] * n
    if m == 0:
        return result

    seg_norm = [normalize_line(t) for t in seg_texts]
    wiki_norm = [normalize_line(w.get("text", "") or "") for w in wiki_lines]
    # 느슨한 키(구두점 제거)는 엄격 키가 못 찾을 때만 폴백으로 쓴다 — 아래 각 비교
    # 지점(정확 일치·합침·쪼갬·재동기화)이 전부 이 폴백을 공유한다.
    seg_loose = [loose_normalize_line(t) for t in seg_texts]
    wiki_loose = [loose_normalize_line(w.get("text", "") or "") for w in wiki_lines]
    wiki_trans = [w.get("translation", "") or "" for w in wiki_lines]

    i = j = 0
    while i < n and j < m:
        if _lines_match(seg_norm[i], seg_loose[i], wiki_norm[j], wiki_loose[j]):
            result[i] = wiki_trans[j]
            i += 1
            j += 1
            continue

        merged_k = _find_combine(seg_norm, seg_loose, i, wiki_norm[j], wiki_loose[j], n)
        if merged_k:
            result[i] = wiki_trans[j]
            i += merged_k
            j += 1
            continue

        split_k = _find_combine(wiki_norm, wiki_loose, j, seg_norm[i], seg_loose[i], m)
        if split_k:
            result[i] = "".join(wiki_trans[j : j + split_k])
            i += 1
            j += split_k
            continue

        anchor = _find_resync_anchor(seg_norm, wiki_norm, seg_loose, wiki_loose, i, j)
        if anchor is None:
            break
        i, j = anchor

    return result
