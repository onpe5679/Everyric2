"""CTC-based forced alignment engine using wav2vec2 models.

For Japanese/Korean/Chinese: Uses HuggingFace wav2vec2 models with native character support.
For other languages: Uses torchaudio MMS_FA with Latin alphabet.
"""

import logging
import math
import threading
from collections.abc import Callable
from typing import Any, Literal, NamedTuple

import torch
import torchaudio
import torchaudio.functional as F

from everyric2.alignment.base import (
    AlignmentError,
    BaseAlignmentEngine,
    EngineNotAvailableError,
    TranscriptionResult,
    WordTimestamp,
)
from everyric2.alignment.matcher import MatchStats
from everyric2.audio.loader import AudioData
from everyric2.config.settings import AlignmentSettings
from everyric2.inference.prompt import LyricLine, SyncResult

logger = logging.getLogger(__name__)

MMS_BASE_MODEL = "facebook/mms-1b-all"

LANG_MODEL_MAP = {
    "ja": MMS_BASE_MODEL,
    "ko": MMS_BASE_MODEL,
    "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    "en": MMS_BASE_MODEL,
}

# torchaudio MMS_FA 폴백 경로의 베이스 식별자 (HF 모델 id와 겹치지 않는 이름)
_TORCHAUDIO_MMS_FA = "torchaudio:MMS_FA"

MMS_LANG_CODES = {
    "ko": "kor",
    "ja": "jpn",
    "zh": "cmn-script_simplified",
    # 영어는 'eng'가 아니라 'kor' 어댑터로 정렬한다. eng vocab은 154개뿐이고 한글·가나가
    # 0개라 한 글자라도 CJK가 섞이면 그 글자가 통째로 정렬에서 빠지는데, kor/jpn/cmn은
    # 전부 ASCII 소문자 26개를 갖고 있어 라틴을 똑같이 덮는다. 순수 영어 4곡을 같은
    # 오디오·같은 가사로 세 어댑터에 돌려 유튜브 수동 자막 기준 잔차를 비교한 결과
    # eng의 우위가 없었다 (중앙값 차 ≤ 0.035초 = CTC 프레임 20ms의 2칸 이하이고,
    # 부호도 곡마다 뒤집힌다). 상세 수치는 tests/test_adapter_coverage.py 참조.
    "en": "kor",
}

# 어댑터 vocab이 실제로 덮는 문자 스크립트. facebook/mms-1b-all의 tokenizer vocab.json을
# 직접 센 값이다(2026-07-25, 단일 글자 토큰만):
#     eng 154 — latin_lower 26, latin_ext 51, han 8,    hangul    0, kana  0
#     kor 1330 — latin_lower 26, latin_ext  5, han 2,    hangul 1261, kana  0
#     jpn 2268 — latin_lower 26, latin_ext  1, han 2048, hangul    0, kana 158
#     cmn 4495 — latin_lower 26, latin_ext  5, han 4419, hangul    0, kana  4
# 한 자리 수의 흔적(eng의 han 8개, cmn의 kana 4개)은 그 스크립트를 덮는다고 보지 않는다 —
# 수천 자 규모와 한 자리 수를 같은 'True'로 접으면 커버리지 판정이 뒤집힌다.
#
# 왜 하드코딩인가: 어느 어댑터를 로드할지 **로드 전에** 정해야 하므로 vocab을 조회할 수
# 없다(vocab.json은 HF 캐시가 이미 있을 때만 읽히고, 첫 실행에는 없다). 대신 글자 단위가
# 아니라 스크립트 단위로만 판정한다 — kor vocab의 한글은 1261자로 완성형 11172자를 다
# 담지 못하지만, 빠진 음절은 `_oov_substitute`가 발음이 가까운 in-vocab 음절로 치환하므로
# "kor은 한글을 덮는다"가 실효적으로 성립한다. 실측 대조: 한글 234자 + 라틴 406자인 곡의
# 글자 단위 커버리지는 kor 0.994 / eng 0.632 / jpn 0.628이었고, 아래 스크립트 표가 주는
# 예측(1.000 / 0.634 / 0.634)과 순위가 일치한다.
# 이 표가 실제 vocab과 어긋나지 않는지는 tests/test_adapter_coverage.py가 고정한다.
_ADAPTER_SCRIPTS: dict[str, frozenset[str]] = {
    "kor": frozenset({"latin", "hangul"}),
    "jpn": frozenset({"latin", "kana", "han"}),
    "cmn-script_simplified": frozenset({"latin", "han"}),
}

# 여러 스크립트가 섞였을 때 후보로 두는 언어 → 그 언어를 지목하는 스크립트.
# 삽입 순서가 곧 완전 동점 시의 우선순위다 (_pick_by_coverage의 max가 첫 최대를 남긴다) —
# 기존 단일 스크립트 분기의 우선순위(ja → ko → zh)와 같게 두어 판정이 예측 가능하게 한다.
# 'en'이 없는 것이 의도다: en은 kor 어댑터를 쓰므로 ko 후보와 커버리지가 항상 같고,
# 스크립트가 둘 이상이면 라틴만 있는 후보는 어차피 이길 수 없다.
_MULTILINGUAL_CANDIDATES: dict[str, str] = {
    "ja": "kana",
    "ko": "hangul",
    "zh": "han",
}


def _char_script(char: str) -> str | None:
    """정렬 토큰이 될 수 있는 글자를 스크립트로 분류한다. 그 밖이면 None.

    대문자 라틴도 'latin'이다 — `_resolve_token_char`가 소문자로 조회하므로 어댑터
    vocab에 대문자가 0개여도 실제로 정렬된다. 여기서 커버 불가로 세면 커버리지 판정이
    소문자화 폴백과 어긋난다.
    숫자·구두점·공백은 어느 어댑터로도 노래로 정렬되지 않으므로 분모에서 아예 뺀다.
    """
    code = ord(char)
    if 0x3040 <= code <= 0x30FF:  # Hiragana + Katakana
        return "kana"
    if 0xAC00 <= code <= 0xD7AF or 0x1100 <= code <= 0x11FF:  # Hangul
        return "hangul"
    if 0x4E00 <= code <= 0x9FFF:  # CJK Ideographs
        return "han"
    if 0x41 <= code <= 0x5A or 0x61 <= code <= 0x7A:  # A-Z a-z
        return "latin"
    return None


def script_census(text: str) -> dict[str, int]:
    """가사 텍스트의 스크립트별 글자 수 (정렬 대상이 아닌 글자는 세지 않는다)."""
    counts = {"kana": 0, "hangul": 0, "han": 0, "latin": 0}
    for char in text:
        script = _char_script(char)
        if script is not None:
            counts[script] += 1
    return counts


def adapter_coverage(counts: dict[str, int], adapter: str) -> float:
    """이 어댑터 vocab이 덮는 글자 비율 (0~1). 정렬 대상 글자가 없으면 0.0."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    scripts = _ADAPTER_SCRIPTS[adapter]
    return sum(n for script, n in counts.items() if script in scripts) / total


# 후보가 되려면 그 어댑터가 "라틴 말고" 덮는 글자가 이만큼은 있어야 한다(전체 대비 비율).
# "덮는 쪽" 판정의 이면: jpn vocab은 라틴도 덮으므로, 영어 곡에 일본어 브리지 4줄만
# 있어도(가나 27/1354자 = 2.0%) ja가 en을 커버리지로 이겨 버린다 — UgK6n1KKUxY 실측,
# ja 판정→fast 라우팅→첫 줄이 35초 앞으로 끌려갔다. 커버리지 우세가 라틴을 덮는 데서만
# 오는 후보는 "덮어서 얻는 것"이 흔적뿐이므로 뺀다. 기준이 지목 스크립트(가나)가 아니라
# **비라틴 커버리지**인 이유: 한자 위주 일본어 가사(가나 5% 미만 + 한자 다수)는 가나만
# 재면 탈락해 zh/en으로 새는데, jpn이 라틴 밖에서 덮는 양(가나+한자)으로 재면 당당히
# 남는다(엣지 감사 #9, 2026-08-03). 0.05인 이유: 진짜 혼용곡의 비라틴 비중은 실측
# 26~46%로 여유가 크고, 흔적 혼입(크레딧 줄·브리지 한 절)은 2~3%다.
_NATIVE_SHARE_FLOOR = 0.05


def _nonlatin_gain(counts: dict[str, int], lang: str) -> float:
    """이 언어 어댑터가 라틴 밖에서 덮는 글자 비율 — 후보 자격(_NATIVE_SHARE_FLOOR) 판정용."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    scripts = _ADAPTER_SCRIPTS[MMS_LANG_CODES[lang]]
    return sum(n for script, n in counts.items() if script in scripts and script != "latin") / total


def _pick_by_coverage(counts: dict[str, int]) -> str:
    """스크립트가 섞인 가사에서 쓸 언어를 어댑터 vocab 커버리지로 고른다.

    "어느 문자가 더 많나"로 고르면 안 된다 — 한글 234자 + 라틴 406자인 KPOP 곡이 라틴
    다수라는 이유로 en으로 판정됐고, eng vocab에는 한글이 0개라 순수 한글 15줄이 정렬에서
    통째로 빠져 균등 보간만 됐다(구간 오차 −2.5초 → −11.4초로 단조 악화, 실측). 많은 쪽이
    아니라 **덮는 쪽**을 골라야 한다: kor은 한글과 라틴을 모두 덮으므로 이 곡을 100% 덮는다.

    단, 비라틴 커버리지(_nonlatin_gain)가 _NATIVE_SHARE_FLOOR 미만인 후보는 애초에
    빠진다 — 커버리지 우세가 라틴을 덮는 데서만 오는 후보를 걸러, 사실상 라틴 단일
    가사가 ja/ko/zh로 새지 않게 한다. 전 후보가 빠지면 en이다.

    동점 처리:
      ① 커버리지가 같으면 그 언어를 지목하는 스크립트가 실제로 더 많은 쪽. 한글 100자 +
         한자 100자처럼 ko/ja/zh가 모두 0.5로 묶이는 경우, 가나가 0자인 ja를 이 기준이
         떨어뜨린다.
      ② 그래도 같으면 _MULTILINGUAL_CANDIDATES의 순서(ja → ko → zh). 남은 후보가
         정말로 구별 불가능한 경우이므로 결정론만 확보하면 된다.
    """
    viable = [
        lang
        for lang in _MULTILINGUAL_CANDIDATES
        if _nonlatin_gain(counts, lang) >= _NATIVE_SHARE_FLOOR
    ]
    if not viable:
        return "en"
    return max(
        viable,
        key=lambda lang: (
            adapter_coverage(counts, MMS_LANG_CODES[lang]),
            counts[_MULTILINGUAL_CANDIDATES[lang]],
        ),
    )


def detect_language_from_text(text: str) -> tuple[str, bool]:
    """Detect language from lyrics text.

    Returns:
        Tuple of (primary_language, is_multilingual)
        - primary_language: 'ja', 'ko', 'zh', or 'en' (dominant language)
        - is_multilingual: True if multiple scripts detected → recommends MMS 1B-all
    """
    counts = script_census(text)
    ja_count, ko_count, zh_count, en_count = (
        counts["kana"],
        counts["hangul"],
        counts["han"],
        counts["latin"],
    )

    detected = []
    if ja_count > 0:
        detected.append("ja")
    if ko_count > 0:
        detected.append("ko")
    if zh_count > 0 and ja_count == 0:  # CJK without kana = Chinese
        detected.append("zh")
    if en_count > 10:
        detected.append("en")

    is_multilingual = len(detected) >= 2

    if is_multilingual:
        primary = _pick_by_coverage(counts)
        coverage = ", ".join(
            f"{lang}/{MMS_LANG_CODES[lang]}={adapter_coverage(counts, MMS_LANG_CODES[lang]):.3f}"
            for lang in _MULTILINGUAL_CANDIDATES
        )
        logger.info(
            f"Multiple languages detected: {detected} → primary: {primary} "
            f"(chars {counts}, vocab coverage {coverage}), using MMS 1B-all"
        )
        return (primary, True)

    if ja_count > 0:
        return ("ja", False)
    if ko_count > 0:
        return ("ko", False)
    if zh_count > 0:
        return ("zh", False)
    return ("en", False)


_HANGUL_CHO = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
_HANGUL_JUNG = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
_TENSE_TO_PLAIN = {"ㄲ": "ㄱ", "ㄸ": "ㄷ", "ㅃ": "ㅂ", "ㅆ": "ㅅ", "ㅉ": "ㅈ"}
_GLIDE_TO_PLAIN = {"ㅑ": "ㅏ", "ㅒ": "ㅐ", "ㅕ": "ㅓ", "ㅖ": "ㅔ", "ㅛ": "ㅗ", "ㅠ": "ㅜ"}


def _oov_substitute(char: str, vocab) -> str | None:
    """vocab에 없는 한글 음절을 발음이 가까운 vocab 음절로 치환(정렬 전용).

    된소리→예사소리(ㅃ→ㅂ), 활음 제거(ㅛ→ㅗ), 종성 제거를 점진 적용해 vocab에 있는
    첫 후보를 반환. 한글 음절이 아니거나(괄호·기호) 후보가 없으면 None → 정렬에서 제외.
    예) 뿅(ㅃㅛㅇ)→뽕→봉, 얍(ㅇㅑㅂ)→얌→압. 출력 글자는 원본을 유지하고 타이밍만 차용한다.
    """
    code = ord(char)
    if not (0xAC00 <= code <= 0xD7A3):  # 한글 완성형 음절이 아님
        return None
    s = code - 0xAC00
    cho_i, jung_i, jong_i = s // 588, (s % 588) // 28, s % 28
    cho, jung = _HANGUL_CHO[cho_i], _HANGUL_JUNG[jung_i]
    cho_alts = [cho, _TENSE_TO_PLAIN.get(cho, cho)]
    jung_alts = [jung, _GLIDE_TO_PLAIN.get(jung, jung)]
    seen: set[str] = set()
    for c in cho_alts:
        for j in jung_alts:
            for jo in (jong_i, 0):  # 원래 종성 유지 → 종성 제거 순
                cand = chr(0xAC00 + _HANGUL_CHO.index(c) * 588 + _HANGUL_JUNG.index(j) * 28 + jo)
                if cand == char or cand in seen:
                    continue
                seen.add(cand)
                if cand in vocab:
                    return cand
    return None


def _resolve_token_char(char: str, vocab) -> str | None:
    """정렬 토큰으로 조회할 글자를 고른다. 원문 글자는 호출부가 따로 보존한다.

    1) vocab에 그대로 있으면 그것 — 대문자가 vocab에 있는 모델이면 원형이 우선한다.
    2) 없으면 소문자로 한 번 더 조회. MMS 어댑터 vocab은 라틴 문자가 전부 소문자라
       (eng/kor/jpn 실측: ASCII 대문자 0개, 소문자 26개) 대문자를 그대로 조회하면 미스가
       나 그 글자가 정렬에서 통째로 빠졌다 — "DA DA RA DA DA" 같은 줄은 word_segments가
       0개였고 한영 혼용 줄은 전 글자가 0.00초에 박혔다.
    3) 그래도 없으면 한글 OOV 음절 치환(_oov_substitute).
    한글은 lower()가 항등이라 2)를 그냥 통과하므로 기존 한글 동작은 바뀌지 않는다.
    """
    if char in vocab:
        return char
    lowered = char.lower()
    if lowered != char and lowered in vocab:
        return lowered
    return _oov_substitute(char, vocab)


# ---------------------------------------------------------------------------
# 오디오 심판 (라인별 후보 독음 채점) + 「들은 것」 greedy 디코딩
#
# 왜 여기인가: emission은 곡당 한 번 계산되는 [1, T, V] log-softmax 텐서이고 `align()`
# 안에서만 살아 있다(80MB급이라 밖으로 내보내면 수명·메모리 문제가 생긴다). 후보 채점은
# 그 위의 DP(`F.forced_align`)뿐이라 **모델 forward가 없다** — 같은 emission에 다른 토큰열을
# 넣어 점수만 비교하면 되므로 비용이 사실상 0이다.
# ---------------------------------------------------------------------------

# greedy 디코딩에서 글자로 내보내지 않는 토크나이저 특수 토큰. blank(pad)는 id로 따로 걸러진다.
_SPECIAL_TOKENS = frozenset({"<pad>", "<s>", "</s>", "<unk>", "[PAD]", "[UNK]"})

# 설정을 못 읽을 때 쓰는 심판 마진 (AlignmentSettings.pron_referee_margin와 같은 값 —
# 근거는 그 필드 description에 있다). 0으로 떨어지면 노이즈로도 독음이 뒤집힌다.
_DEFAULT_REFEREE_MARGIN = 0.15


def _candidate_tokens(text: str, vocab) -> tuple[list[str], list[int]]:
    """후보 문자열 → (정렬에 실제로 들어가는 원문 글자 목록, 토큰 id 열).

    글자 선택 규칙은 1패스와 **똑같이** `_resolve_token_char`를 쓴다 — 규칙이 갈라지면
    기본값과 후보가 서로 다른 글자 집합으로 채점돼 점수 비교가 성립하지 않는다.
    """
    chars: list[str] = []
    ids: list[int] = []
    for ch in text:
        tok = _resolve_token_char(ch, vocab)
        if tok is not None:
            chars.append(ch)
            ids.append(vocab[tok])
    return chars, ids


def _min_ctc_frames(ids: list[int]) -> int:
    """이 토큰열을 CTC로 강제정렬하는 데 필요한 최소 프레임 수.

    같은 토큰이 연달아 오면 사이에 blank가 반드시 하나 들어가야 하므로 그만큼 더 필요하다.
    창이 이보다 짧으면 forced_align이 실패하거나 무의미한 경로를 낸다 → 그 후보는 버린다.
    """
    return len(ids) + sum(1 for a, b in zip(ids, ids[1:]) if a == b)


# 금지 프레임에서 실제 토큰 열에 덮어쓰는 값. **-inf가 아니라 유한값**인 이유는 실측이다
# (torchaudio 2.8.0, CPU): 실행 가능한 마스크에서는 두 값이 완전히 같은 경로를 내고 NaN도
# 없었지만, 마스크가 실행 불가능해질 때 갈렸다 — 전 구간을 -inf로 막으면 forced_align이
# **토큰 순서를 지키지 않는 무효 경로**([0, 3, 3, 3, ...] — 토큰 1·2가 아예 배출되지 않음)를
# 내놓고 score에 -inf를 실어 보낸다. -1e4는 같은 상황에서도 순서가 살아 있는 정상 경로를
# 내고(그 경로는 아래 손실 판정에서 압도적으로 탈락한다), -inf가 char_info 역매핑을 조용히
# 망가뜨릴 여지를 아예 없앤다. 크기 근거: 어떤 실제 log 확률보다 4자리 낮고(어댑터 글자당
# 약 -3 nats), T=5만 프레임을 전부 곱해도 float32 범위(~3.4e38)에 한참 못 미친다.
_FORBIDDEN_LOGP = -1.0e4

# 설정을 못 읽을 때 쓰는 앵커 채택 상한 (AlignmentSettings.caption_anchor_max_token_loss와
# 같은 값 — 근거는 그 필드 description에 있다).
_DEFAULT_ANCHOR_MAX_TOKEN_LOSS = 0.3


def _forbidden_frames(
    spans: list[tuple[float, float]], num_frames: int, ratio: float
) -> torch.Tensor:
    """금지 구간(초) → 그 구간에 **완전히** 들어가는 프레임 인덱스 (1-D long 텐서).

    경계 프레임은 ceil/floor로 잘라 버린다 — 반쯤 걸친 프레임까지 막으면 앞 앵커의 마지막
    글자나 다음 앵커의 첫 글자를 밀어낼 수 있고, 이 기능의 기본자세는 «덜 막는다»다.
    """
    mask = torch.zeros(num_frames, dtype=torch.bool)
    for start, end in spans:
        f0 = max(0, math.ceil(start / ratio))
        f1 = min(num_frames, math.floor(end / ratio))
        if f1 > f0:
            mask[f0:f1] = True
    return mask.nonzero(as_tuple=True)[0]


class _Span(NamedTuple):
    """``F.merge_tokens``의 TokenSpan과 같은 인터페이스 (start/end/score).

    창별 정렬 결과를 프레임 오프셋만큼 옮겨 전곡 위치 인덱스로 다시 늘어놓을 때 쓴다.
    downstream(`char_info` 역매핑·라인 창·star 기록)은 이 세 필드만 본다.
    """

    start: int
    end: int
    score: float


def _anchor_blocks(
    line_starts: dict[int, float],
    n_lines: int,
    line_boundaries: list[tuple[int, int]],
    n_tokens: int,
    num_frames: int,
    ratio: float,
    window_sec: float,
) -> list[tuple[int, int, int, int]]:
    """앵커 줄을 칸막이로 삼아 (토큰 시작, 토큰 끝, 프레임 시작, 프레임 끝) 블록을 만든다.

    각 블록은 «앵커 줄 하나 + 그 뒤의 매칭 안 된 줄들»이고, 프레임 창은 그 앵커의 자막 시각
    ± ``window_sec``에서 다음 앵커의 자막 시각 + ``window_sec``까지다. 앵커 줄은 자기 창 밖에서
    시작할 수 없고, 매칭 안 된 줄은 창 안에서 자유롭다.

    프레임 시작은 «앞 블록이 실제로 끝난 프레임»과 «자막 시각 - 창» 중 늦은 쪽으로 잡히는데
    그 순차화는 호출부가 한다(앞 블록의 정렬 결과를 알아야 한다). 여기서는 자막 기준값만 낸다.
    """
    anchored = sorted(line_starts)
    if not anchored:
        return []

    def frame_of(t: float) -> int:
        return max(0, min(num_frames, int(round(t / ratio))))

    # 줄 구간 경계: [0, a_0), [a_0, a_1), ..., [a_m, n_lines)
    edges = ([0] if anchored[0] > 0 else []) + anchored + [n_lines]
    blocks: list[tuple[int, int, int, int]] = []
    for k in range(len(edges) - 1):
        lo_line, hi_line = edges[k], edges[k + 1]
        if hi_line <= lo_line:
            continue
        p_lo = 0 if k == 0 and lo_line == 0 else line_boundaries[lo_line][0]
        p_hi = n_tokens if hi_line >= n_lines else line_boundaries[hi_line - 1][1] + 1
        # 이 블록의 왼쪽 시각 근거 — 선행 블록(앵커 없는 첫 줄들)은 0부터다
        t_lo = line_starts.get(lo_line)
        f_lo = 0 if t_lo is None else frame_of(t_lo - window_sec)
        # 오른쪽은 다음 앵커의 자막 시각 + 창 (마지막 블록은 곡 끝)
        t_hi = line_starts.get(hi_line)
        f_hi = num_frames if t_hi is None else frame_of(t_hi + window_sec)
        blocks.append((p_lo, p_hi, f_lo, f_hi))
    return blocks


def _span_bounds(path: torch.Tensor, blank_id: int) -> list[tuple[int, int]]:
    """정렬 경로 → 타깃 토큰별 (시작, 끝) 프레임. k번째 스팬이 k번째 타깃 토큰이다.

    ``F.merge_tokens``와 같은 경계를 내지만 점수를 요구하지 않는다. 마스킹된 emission 위에서
    구한 경로의 **경계만** 먼저 알아야 하고(다음 블록의 하한 계산), 점수는 마스크를 되돌린 뒤
    원본에서 매겨야 하기 때문이다. 같은 토큰이 연달아 오는 경우 CTC가 사이에 blank를 넣으므로
    스팬은 타깃과 1:1로 대응한다 — 이 파일의 기존 코드가 이미 그 대응에 의존한다.
    """
    labels = path.tolist()
    bounds: list[tuple[int, int]] = []
    start = None
    prev = blank_id
    for t, lab in enumerate(labels):
        if lab != prev:
            if start is not None:
                bounds.append((start, t))
                start = None
            if lab != blank_id:
                start = t
            prev = lab
    if start is not None:
        bounds.append((start, len(labels)))
    return bounds


def _spans_by_position(
    emission: torch.Tensor,
    tokens: list[int],
    blank_id: int,
    paths: list[tuple[int, int, torch.Tensor]],
    n_tokens: int,
) -> list:
    """블록별 경로들을 전곡 토큰 위치 인덱스의 스팬 목록으로 되돌린다.

    점수는 **원본 emission**에서 매긴다 — 마스크 바닥값이 섞이면 신뢰도가 오염되고 그 값이
    quality_score까지 흘러간다. 스팬 안에서는 라벨이 상수이므로 그 열의 평균이 곧
    ``merge_tokens``가 내는 점수다.
    """
    out: list = [None] * n_tokens
    for p_lo, f_lo, path in paths:
        for k, (s, e) in enumerate(_span_bounds(path, blank_id)):
            pos = p_lo + k
            if pos >= n_tokens:
                break
            gs, ge = f_lo + s, f_lo + e
            out[pos] = _Span(gs, ge, float(emission[0, gs:ge, tokens[pos]].mean()))
    return out


def _token_peak_support(
    emission: torch.Tensor,
    spans_by_pos: list,
    tokens: list[int],
    char_positions: list[int],
) -> float | None:
    """글자 토큰이 자기 슬롯 안에서 얻은 **최고 로그확률**의 평균 (원본 emission 기준).

    두 정렬 경로를 비교하는 척도다. 경로 전체 로그우도로 비교할 수 없는 이유:

    ① star 열이 ``log(1.0)=0``이라 **star가 덮는 프레임 수가 점수를 지배한다.** 양성 제약은
       star를 실제 가창 구간에서 걷어내므로 경로 총점이 크게 떨어지는데, 그 하락은 「배치가
       나빠졌다」가 아니라 「공짜 채널을 덜 썼다」다. 실측(zyRt-nBM3dY): 금지 구간만으로도
       총점 -4777 → -4831이었고, 그 차이는 음향 근거와 무관했다.
    ② 스팬 길이에 흔들리지 않아야 한다. 1패스는 토큰 프레임 수를 최소화해 각 토큰이 1프레임만
       갖는 경향이 있고, 제약이 걸린 2패스는 더 넓은 슬롯을 갖는다. 평균은 넓은 슬롯에
       불리하지만 **최댓값은 슬롯 길이에 중립**이다.

    묻는 것은 하나다: **각 글자가 자기 자리에서 음향 근거를 찾았는가.** 균일 바닥 posterior
    (합성보컬)에서는 두 경로가 같은 값을 내고(= 제약이 아무것도 잃지 않는다), 봉우리가 있는
    곡에서 글자를 봉우리 밖으로 끌어내면 즉시 벌어진다 — 정상 곡을 지키는 것이 이 척도다.
    """
    vals: list[float] = []
    for pos in char_positions:
        if pos >= len(spans_by_pos):
            continue
        span = spans_by_pos[pos]
        if span is None:
            continue
        s, e = int(span.start), int(span.end)
        if e <= s:
            continue
        vals.append(float(emission[0, s:e, tokens[pos]].max()))
    return sum(vals) / len(vals) if vals else None


def _path_frame_logprobs(emission: torch.Tensor, path: torch.Tensor) -> torch.Tensor:
    """경로(프레임별 라벨)를 주어진 emission에서 재채점한 프레임별 로그확률.

    두 정렬 경로를 **같은 목적함수**로 비교하는 유일한 방법이다. 마스킹된 emission의 경로
    점수는 목적함수가 달라(금지 열이 -1e4) 서로 비교할 수 없다.
    """
    idx = torch.arange(path.shape[-1], device=emission.device)
    return emission[0, idx, path.to(torch.long)]


def _score_tokens(window: torch.Tensor, ids: list[int], blank_id: int):
    """창(emission 슬라이스)에 토큰열을 강제정렬하고 (프레임당 평균 로그확률, 스팬)을 낸다.

    **창의 모든 프레임으로 정규화한다** — blank 프레임까지 포함한 경로 전체의 로그우도를
    창 길이로 나눈 값이다. 후보들이 **같은 창**에서 채점되므로 분모가 상수이고, 따라서
    이 비교는 사실상 경로 총 로그우도 비교다.

    처음에는 `merge_tokens`로 병합한 **토큰 스팬만** 평균했다. 그것이 실오디오에서 파국을
    만들었다(s5Rkv_5Sbbo 134줄 중 53줄 교체, 그중 11줄이 글자가 사라진 것: 私 와타쿠시오→
    시오, 彼らを 카레라오→카라오, 苦しみ 쿠루시미→쿠 시미). 이유는 비대칭이다 — 토큰 스팬만
    보면 짧은 후보는 원치 않는 프레임을 blank로 넘기고 **그 프레임 비용을 전혀 내지 않는다.**
    삽입은 벌점을 받고 삭제는 공짜였다. 창 전체로 정규화하면 blank 프레임의 로그확률도
    합에 들어가므로, 실제로 발성된 프레임을 blank로 버리는 후보가 그만큼 벌점을 받는다.

    당시 정규화를 못박은 테스트는 균일 emission에서 합 vs 평균만 비교해 이 편향을 볼 수
    없었다. 봉우리 emission에서 삭제 후보를 세우는 테스트가 함께 있어야 한다.
    """
    targets = torch.tensor([ids], dtype=torch.int32, device=window.device)
    aligned, scores = F.forced_align(window, targets, blank=blank_id)
    spans = F.merge_tokens(aligned[0], scores[0], blank=blank_id)
    if not spans:
        return None, None
    n_frames = int(scores.shape[-1])
    if n_frames <= 0:
        return None, None
    return float(scores[0].sum()) / n_frames, spans


def _line_frame_windows(char_info: list[dict], token_spans) -> dict[int, tuple[int, int]]:
    """라인 인덱스 → 그 라인이 차지한 [첫 글자 시작, 마지막 글자 끝) 프레임 창.

    1패스가 잡은 창이므로 심판·greedy 디코딩이 라인 밖 오디오를 보지 않는다. 라인 구분자
    "|"와 star는 char_info에 없으므로 자연히 창에서 빠진다.
    """
    windows: dict[int, tuple[int, int]] = {}
    n = len(token_spans)
    for ci in char_info:
        idx = ci["token_idx"]
        if idx >= n:
            continue
        line_idx = ci["line_idx"]
        span = token_spans[idx]
        # 창 정렬 경로는 위치별 목록이라 빈 자리가 있을 수 있다 (앵커 블록이 못 덮은 위치)
        if span is None:
            continue
        cur = windows.get(line_idx)
        windows[line_idx] = (
            int(span.start) if cur is None else cur[0],
            int(span.end),
        )
    return windows


def _greedy_text(ids: list[int], blank_id: int, id_to_token: dict[int, str]) -> str:
    """CTC greedy 디코딩: 반복 축약 → blank 제거 → 토큰 문자열 연결.

    "모델이 실제로 들은 것"이다. 정렬 텍스트와의 불일치는 **posterior 크기에 의존하지 않는**
    라인 단위 품질 지표다(현재 conf는 어댑터 vocab 크기에 스케일 의존해 곡 단위로만 유효하다).
    노래 ASR은 약하므로 이 텍스트로 발음을 만들면 안 된다 — 후보 중 고르기와 진단 전용이다.
    """
    out: list[str] = []
    prev: int | None = None
    for tid in ids:
        if tid != prev:
            prev = tid
            if tid == blank_id:
                continue
            tok = id_to_token.get(tid)
            if tok is None or tok in _SPECIAL_TOKENS:
                continue
            out.append(" " if tok == "|" else tok)
    return " ".join("".join(out).split())


def _greedy_spans(
    frames: list[int],
    blank_id: int,
    id_to_token: dict[int, str],
    frame_sec: float,
    t0: float,
) -> list[tuple[str, float]]:
    """CTC greedy 열 → (토큰, 첫 프레임 시각) 목록. `_greedy_text`와 같은 반복 축약·blank
    제거 규칙이지만 문자열로 뭉치지 않고 토큰별 시각을 남긴다 — heard 텍스트의 타임라인
    버전이다(디버그 표시 전용, 정렬 결과와 무관).
    """
    spans: list[tuple[str, float]] = []
    prev: int | None = None
    for idx, tid in enumerate(frames):
        if tid != prev:
            prev = tid
            if tid == blank_id:
                continue
            tok = id_to_token.get(tid)
            if tok is None or tok in _SPECIAL_TOKENS:
                continue
            spans.append((" " if tok == "|" else tok, t0 + idx * frame_sec))
    return spans


class CTCEngine(BaseAlignmentEngine):
    def __init__(self, config: AlignmentSettings | None = None):
        super().__init__(config)
        self._model = None
        self._processor = None
        # 캐시 키 — force_mms면 "{language}_mms", 아니면 순수 언어 그대로. 어댑터 재사용
        # 여부 판정(_ensure_model_loaded 655행)에만 쓰는 내부 값이라 그대로 둔다.
        self._current_lang = None
        # DB로 흘러가는 값은 이 둘로 갈라서 노출한다(결함 #5) — worker.py의 _run_alignment가
        # 여기서 순수 언어와 변형을 따로 읽어 SyncResult.language/engine_variant에 각각
        # 저장한다. _current_lang(위)은 "{language}_mms" 캐시 키를 그대로 유지하므로 절대
        # DB에 직접 쓰지 마라 — 결함 #5가 바로 그 실수였다.
        self._current_language: str | None = None
        self._current_engine_variant: str | None = None
        # 현재 상주 중인 베이스 모델 식별자와 MMS 어댑터 코드 — 언어 전환 시 전체 재로드가
        # 필요한지(베이스가 다름) 어댑터 교체로 충분한지(베이스가 같음) 판단하는 근거다.
        self._current_base: str | None = None
        self._current_adapter: str | None = None
        # 상주 MMS 프로세서와 그것이 현재 겨누고 있는 어댑터 코드. 인스턴스 필드라
        # 엔진을 버리면(clear_shared_ctc_engine) 같이 사라진다.
        self._mms_processor_obj = None
        self._mms_processor_lang: str | None = None
        # 모델 가중치는 인스턴스 하나를 공유하므로, 어댑터 교체가 다른 스레드의 forward
        # 중간에 끼어들면 lm_head/어댑터가 뒤바뀐 채 정렬된다. align 전체를 이 락으로 감싼다.
        self._model_lock = threading.RLock()
        self._device = None
        self._last_word_timestamps: list[WordTimestamp] = []
        self._last_match_stats = None
        # 직전 정렬에서 star 토큰이 흡수한 (start, end) 구간들 — 디버그/진단용
        self._last_star_spans: list[tuple[float, float]] = []
        # 직전 정렬의 자막 앵커 판정 기록 (금지 구간·실행가능성·손실·채택) — 사후 감사용.
        # 앵커를 주지 않은 정렬에서는 None이다.
        self._last_caption_anchor: dict[str, Any] | None = None
        # 직전 정렬의 star 성형 기록 (weight·가격이 매겨진 프레임 비율) — presence를 주지
        # 않은 정렬에서는 None이다.
        self._last_star_prior: dict[str, Any] | None = None
        # 직전 정렬의 오디오 심판 판정 기록 (line_idx 오름차순). 워커가 로그·디버그 메타에
        # 실어 실오디오에서 판정 근거를 되짚을 수 있게 한다 — 조용히 바꾸면 사후 추적 불가.
        self._last_referee: list[dict] = []
        # 직전 정렬에서 모델이 라인 구간에서 "들은" greedy 디코딩 텍스트 (line_idx → 텍스트)
        self._last_heard: dict[int, str] = {}
        # 위 텍스트의 글자별 시각 (line_idx → [(글자, 초)]) — heard_spans 디버그 타임라인 재료
        self._last_heard_spans: dict[int, list[tuple[str, float]]] = {}

    def is_available(self) -> bool:
        try:
            import torchaudio  # noqa: F401
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor  # noqa: F401

            return True
        except ImportError:
            return False

    def _get_device(self) -> torch.device:
        if self._device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self._device

    def _mms_processor(self, adapter_code: str):
        """상주 MMS 프로세서를 요청한 어댑터 코드로 겨눠서 돌려준다.

        언어마다 프로세서를 새로 만들면 최적화가 첫 전환에서 통째로 날아간다 — 3090 실측:
        AutoProcessor.from_pretrained 3.65초 vs load_adapter 0.23초 vs set_target_lang
        0.000초. 그래서 인스턴스 하나를 두고 다시 겨누기만 한다.

        set_target_lang 반복이 이전 언어의 토큰을 남기지 않는지는 실측으로 확인했다:
        eng/kor/jpn/cmn 4개 코드의 24개 전환 순서 전부 + 같은 코드 반복 설정까지 101회를
        매번 새로 만든 인스턴스와 대조해 vocab 해시·pad_token_id·added_tokens가 전부
        일치했다(불일치 0회). 정렬 결과 자체의 동등성도 별도로 확인했다 — tests/
        test_adapter_swap.py 참조.
        """
        if self._mms_processor_obj is None:
            from transformers import AutoProcessor

            self._mms_processor_obj = AutoProcessor.from_pretrained(MMS_BASE_MODEL)
            self._mms_processor_lang = None
        if self._mms_processor_lang != adapter_code:
            self._mms_processor_obj.tokenizer.set_target_lang(adapter_code)
            self._mms_processor_lang = adapter_code
        return self._mms_processor_obj

    def _ensure_model_loaded(self, language: str, force_mms: bool = False) -> None:
        cache_key = f"{language}_mms" if force_mms else language
        if self._model is not None and self._current_lang == cache_key:
            return

        if not self.is_available():
            raise EngineNotAvailableError(
                "Required packages not installed. Install with: pip install transformers torchaudio"
            )

        device = self._get_device()

        use_mms = force_mms or (
            language in LANG_MODEL_MAP and LANG_MODEL_MAP[language] == MMS_BASE_MODEL
        )

        if use_mms:
            # 매핑에 없는 언어까지 오면 라틴+한글을 덮는 kor로 떨어진다. 예전 기본값
            # 'eng'는 vocab 154개에 한글·가나가 0개라, 정체를 모르는 언어에 대해 가장
            # 잃을 게 많은 선택이었다.
            mms_lang_code = MMS_LANG_CODES.get(language, "kor")

            # 베이스가 이미 MMS면 인코더 가중치는 언어와 무관하게 동일하다 — 언어별로 다른
            # 것은 어댑터 레이어와 lm_head뿐이고 load_adapter가 그 둘을 통째로 덮어쓴다
            # (transformers modeling_wav2vec2.py: _get_adapters + missing_keys 검사로
            # 어댑터 파라미터 전부가 교체됨이 보장된다). 전체 재로드(약 5초) 대신 어댑터만
            # 교체하면 0.2초대이고 결과 emission은 동일하다.
            if self._model is not None and self._current_base == MMS_BASE_MODEL:
                self._processor = self._mms_processor(mms_lang_code)
                if self._current_adapter != mms_lang_code:
                    self._model.load_adapter(mms_lang_code)
                    self._model.eval()
                    logger.info(
                        f"MMS adapter swapped: {self._current_adapter} → {mms_lang_code} "
                        f"(base model kept resident)"
                    )
                self._current_adapter = mms_lang_code
                self._current_lang = cache_key
                self._current_language = language
                self._current_engine_variant = "mms" if force_mms else None
                return

            from transformers import Wav2Vec2ForCTC

            logger.info(f"Loading MMS 1B-all with {language} adapter (force_mms={force_mms})")
            self._processor = self._mms_processor(mms_lang_code)
            self._model = Wav2Vec2ForCTC.from_pretrained(MMS_BASE_MODEL).to(device)
            self._model.load_adapter(mms_lang_code)
            self._model.eval()
            self._current_base = MMS_BASE_MODEL
            self._current_adapter = mms_lang_code
            logger.info(f"MMS adapter loaded: {mms_lang_code}")
        elif language in LANG_MODEL_MAP:
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

            model_name = LANG_MODEL_MAP[language]
            logger.info(f"Loading HuggingFace model: {model_name}")
            self._processor = Wav2Vec2Processor.from_pretrained(model_name)
            self._model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)  # pyright: ignore[reportArgumentType]
            self._model.eval()
            self._current_base = model_name
            self._current_adapter = None
        else:
            logger.info("Loading torchaudio MMS_FA model")
            bundle = torchaudio.pipelines.MMS_FA
            self._model = bundle.get_model(with_star=False).to(device)
            self._processor = bundle
            self._current_base = _TORCHAUDIO_MMS_FA
            self._current_adapter = None

        self._current_lang = cache_key
        self._current_language = language
        self._current_engine_variant = "mms" if force_mms else None

    def transcribe(
        self,
        audio: AudioData,
        language: str | None = None,
    ) -> TranscriptionResult:
        raise NotImplementedError(
            "CTCEngine does not support transcription. Use for forced alignment only."
        )

    def _chunk_windows(self, n_samples: int) -> list[tuple[int, int]]:
        """align_chunk_sec/overlap로 겹침 청크 윈도를 계획한다 (16kHz 기준).

        chunk_sec=0이거나 오디오가 한 청크에 들어가면 윈도 1개 → 통짜 경로(정렬 결과 불변)."""
        from everyric2.audio.chunking import plan_chunk_windows

        sr = 16000
        chunk_sec = getattr(self.config, "align_chunk_sec", 0.0) or 0.0
        overlap_sec = getattr(self.config, "align_chunk_overlap_sec", 5.0) or 0.0
        return plan_chunk_windows(n_samples, int(chunk_sec * sr), int(overlap_sec * sr))

    def _model_logits(self, waveform_1d: torch.Tensor, device) -> torch.Tensor:
        """단일 파형 청크 → wav2vec2/MMS 로짓 [1, t, V] (device 위, no-grad)."""
        with torch.inference_mode():
            inputs = self._processor(  # pyright: ignore[reportCallIssue,reportOptionalCall]
                waveform_1d.numpy(), sampling_rate=16000, return_tensors="pt", padding=True
            )
            input_values = inputs.input_values.to(device)
            return self._model(input_values).logits  # pyright: ignore[reportOptionalCall]

    def _ctc_log_emission(self, waveform: torch.Tensor, device) -> torch.Tensor:
        """CJK 경로 log_softmax emission [1, T, V].

        긴 오디오는 겹침 청크로 나눠 청크별 forward 후 CPU에서 스티칭해 피크 VRAM을 청크
        길이로 제한한다. 단일 청크면 통짜와 완전히 동일한 device 텐서를 돌려준다
        (log_softmax는 기존과 동일하게 inference_mode 밖에서 수행)."""
        n = int(waveform.shape[0])
        windows = self._chunk_windows(n)
        if len(windows) == 1:
            logits = self._model_logits(waveform, device)
            return torch.nn.functional.log_softmax(logits.float(), dim=-1)

        from everyric2.audio.chunking import stitch_chunk_outputs

        pieces = []
        for s, e in windows:
            logits = self._model_logits(waveform[s:e].contiguous(), device)
            pieces.append(torch.nn.functional.log_softmax(logits.float(), dim=-1).cpu())
            del logits
        return stitch_chunk_outputs(pieces, windows, n, frame_axis=1)

    def _mms_emission(self, waveform: torch.Tensor, device) -> torch.Tensor:
        """MMS_FA 경로 emission [1, T, V] (torchaudio 모델이 이미 log-probs 출력).

        긴 오디오는 겹침 청크로 나눠 CPU에서 스티칭. 단일 청크면 통짜와 동일한 device 텐서."""
        n = int(waveform.shape[0])
        windows = self._chunk_windows(n)
        if len(windows) == 1:
            with torch.inference_mode():
                emission, _ = self._model(waveform.unsqueeze(0).to(device))  # pyright: ignore[reportOptionalCall]
            return emission

        from everyric2.audio.chunking import stitch_chunk_outputs

        pieces = []
        for s, e in windows:
            with torch.inference_mode():
                emis, _ = self._model(  # pyright: ignore[reportOptionalCall]
                    waveform[s:e].contiguous().unsqueeze(0).to(device)
                )
            pieces.append(emis.cpu())
            del emis
        return stitch_chunk_outputs(pieces, windows, n, frame_axis=1)

    def _heard_lines(
        self,
        emission: torch.Tensor,
        base_dim: int,
        blank_id: int,
        vocab,
        line_window: dict[int, tuple[int, int]],
    ) -> dict[int, str]:
        """라인별 「들은 것」 greedy 디코딩. 추가 연산이 사실상 없다 (argmax 1회).

        정렬 결과를 **바꾸지 않는다** — 진단·디버그 표시 전용이다. 실패는 빈 dict로 삼킨다
        (진단 기능이 정렬을 죽이면 안 된다).
        """
        if not line_window:
            return {}
        try:
            id_to_token = {tid: tok for tok, tid in vocab.items()}
            greedy = emission[0, :, :base_dim].argmax(dim=-1).tolist()
            return {
                line_idx: _greedy_text(greedy[f0:f1], blank_id, id_to_token)
                for line_idx, (f0, f1) in sorted(line_window.items())
            }
        except Exception:
            logger.warning("Greedy decode of heard text failed; continuing", exc_info=True)
            return {}

    def _heard_span_lines(
        self,
        emission: torch.Tensor,
        base_dim: int,
        blank_id: int,
        vocab,
        line_window: dict[int, tuple[int, int]],
        frame_sec: float,
    ) -> dict[int, list[tuple[str, float]]]:
        """`_heard_lines`와 같은 emission·창을 훑어 라인별 「들은 것」의 글자별 시각을 낸다.

        `_heard_lines`와 별개의 argmax·try/except다 — 한쪽이 실패해도 다른 쪽(heard 텍스트)은
        살아야 한다. 정렬 결과를 **바꾸지 않는다**. 실패는 빈 dict로 삼킨다.
        """
        if not line_window:
            return {}
        try:
            id_to_token = {tid: tok for tok, tid in vocab.items()}
            greedy = emission[0, :, :base_dim].argmax(dim=-1).tolist()
            return {
                line_idx: _greedy_spans(
                    greedy[f0:f1], blank_id, id_to_token, frame_sec, f0 * frame_sec
                )
                for line_idx, (f0, f1) in sorted(line_window.items())
            }
        except Exception:
            logger.warning("Greedy span decode of heard text failed; continuing", exc_info=True)
            return {}

    def _referee_lines(
        self,
        emission: torch.Tensor,
        base_dim: int,
        blank_id: int,
        vocab,
        line_window: dict[int, tuple[int, int]],
        line_candidates: dict[int, list[str]] | None,
        referee_margins: dict[int, float] | None,
    ) -> tuple[dict[int, list[tuple[str, int, int, float]]], dict[int, str]]:
        """라인별 후보 독음을 같은 emission에 강제정렬해 오디오가 고르게 한다.

        판정 철학은 곡 단위 이중정렬(worker의 ``_dual_align_*``)과 같다 — 두 후보를 정렬해
        점수로 고르고, 바닥 근처 노이즈로 뒤집히지 않게 **마진**을 요구한다. 다른 것은
        단위(곡 → 라인)와 비용(정렬 패스 1회 추가 → 0회, 같은 emission 위 DP뿐)이다.

        반환: (line_idx → [(글자, 시작프레임, 끝프레임, 로그확률)], line_idx → 이긴 후보 텍스트).
        교체하지 않은 라인은 두 dict에 아예 들어가지 않는다(= 1패스 결과 그대로).
        """
        self._last_referee = []
        overrides: dict[int, list[tuple[str, int, int, float]]] = {}
        chosen: dict[int, str] = {}
        if not line_candidates:
            return overrides, chosen

        # 설정에 마진이 없는 구성(직접 호출·구버전 config)에서도 0으로 떨어지지 않게 한다 —
        # 마진 0은 노이즈로도 독음이 뒤집히는 값이다. 기본값은 AlignmentSettings와 같다.
        default_margin = float(
            getattr(self.config, "pron_referee_margin", None) or _DEFAULT_REFEREE_MARGIN
        )
        margins = referee_margins or {}
        for line_idx, raw in sorted(line_candidates.items()):
            window_range = line_window.get(line_idx)
            if window_range is None:
                continue  # 정렬된 글자가 0개인 라인 — 창이 없어 심판할 수 없다
            # 중복 후보는 제거하고 순서를 보존한다. 1개만 남으면 심판을 돌리지 않는다 → 비용 0.
            cands = list(dict.fromkeys(c for c in (raw or []) if c))
            if len(cands) < 2:
                continue
            f0, f1 = window_range
            n_frames = f1 - f0
            window = emission[:, f0:f1, :base_dim].contiguous()

            scored: list[tuple[str, float, list[str], object]] = []
            # 토큰열이 같은 후보(문절 띄어쓰기만 다른 변종 등)는 점수도 같다 — 한 번만 채점해
            # 낭비와 다중비교 부풀림을 함께 줄인다. 동점이면 먼저 나온 후보(기본값)가 남는다.
            by_tokens: dict[tuple[int, ...], str] = {}
            for text in cands:
                chars, ids = _candidate_tokens(text, vocab)
                if not ids:
                    continue
                key = tuple(ids)
                if key in by_tokens:
                    continue
                if _min_ctc_frames(ids) > n_frames:
                    continue  # 창이 후보 토큰열보다 짧다 — 이 후보는 건너뛴다
                try:
                    score, spans = _score_tokens(window, ids, blank_id)
                except Exception:
                    logger.warning(
                        "Referee scoring failed on line %d candidate %r", line_idx, text,
                        exc_info=True,
                    )
                    continue
                if score is None:
                    continue
                by_tokens[key] = text
                scored.append((text, score, chars, spans))

            # 기본값(첫 후보)을 못 채점했으면 비교 기준이 없다 — 라인을 그대로 둔다.
            if not scored or scored[0][0] != cands[0]:
                continue
            base_text, base_score, _, _ = scored[0]
            margin = float(margins.get(line_idx, default_margin))
            best = max(scored[1:], key=lambda s: s[1], default=None)
            adopt = best is not None and (best[1] - base_score) >= margin

            record = {
                "line": line_idx,
                "default": base_text,
                "chosen": best[0] if adopt else base_text,
                "margin": round(margin, 4),
                "gain": round(best[1] - base_score, 4) if best is not None else 0.0,
                "frames": n_frames,
                # 후보별 토큰당 평균 로그확률 — 실오디오에서 판정 근거를 되짚는 유일한 자료다.
                "scores": [[t, round(s, 4)] for t, s, _, _ in scored],
            }
            self._last_referee.append(record)
            if not adopt:
                continue
            _, _, best_chars, best_spans = best  # pyright: ignore[reportOptionalSubscript]
            chosen[line_idx] = best[0]
            overrides[line_idx] = [
                (ch, f0 + int(sp.start), f0 + int(sp.end), float(sp.score))
                for ch, sp in zip(best_chars, best_spans)  # pyright: ignore[reportArgumentType]
            ]
        return overrides, chosen

    def _caption_anchor_pass(
        self,
        emission: torch.Tensor,
        targets: torch.Tensor,
        tokens: list[int],
        blank_id: int,
        star_id: int | None,
        base_path: torch.Tensor,
        base_spans: list,
        forbidden_spans: list[tuple[float, float]],
        ratio: float,
        char_positions: list[int],
        blocks: list[tuple[int, int, int, int]] | None = None,
    ):
        """금지 구간 프레임을 마스킹해 2패스 정렬을 돌리고, 이득이 있으면 그 결과를 쓴다.

        **DP를 고치지 않는다.** ``F.forced_align``에는 구간 제약 API가 없으므로, 금지 프레임에서
        blank와 star를 제외한 모든 열을 바닥값으로 눌러 DP가 그 프레임에서 실제 토큰을 배출할
        수 없게 만든다 — 경로는 blank/star로 그 구간을 통과한다. 열을 붙여 DP를 조종하는 star
        트릭, 같은 emission을 슬라이스해 DP만 다시 도는 심판과 같은 관용구다.

        **양성 제약(``blocks``)은 마스킹으로 표현할 수 없다.** 「이 줄은 여기서 시작해야 한다」는
        줄마다 다른 제약인데 emission의 축은 vocab이다 — 「の」의 열은 그 글자를 쓰는 모든 줄이
        공유하므로 열을 눌러서 특정 줄만 막을 방법이 없다. 그래서 양성 제약은 **창을 쪼개** 건다:
        앵커 줄을 칸막이로 삼아 전곡을 블록으로 나누고, 각 블록의 토큰열을 그 블록의 프레임
        창(emission 슬라이스)에만 정렬한다. 창 안 정렬은 이 파일이 이미 하는 일이다(심판의
        ``_score_tokens``). 두 제약은 함께 걸린다 — 마스크를 씌운 emission 위에서 창 정렬을 한다.

        **왜 음성 제약만으로는 부족한가 (실측 zyRt-nBM3dY 2026-07-26).** 금지 구간은 지켜졌는데
        (간주에 줄이 하나도 안 들어갔다) 앵커 줄 52개 중 **4개만** 자막 시각 ±5초 안에 들어왔고
        중앙값이 **+29.6초** 밀렸다. 「여기 있을 수 없다」는 「여기 있어야 한다」를 함의하지 않고,
        posterior가 균일 바닥이면 그 차이가 전부다.

        **채택 판정의 방향이 심판과 반대다.** 1패스는 제약 없는 Viterbi 최적해이고 2패스는 같은
        토큰열의 다른 경로이므로, 2패스가 이길 수는 없다. 물어야 할 것은 **제약을 지키느라
        얼마를 포기했는가**이고, 척도는 ``_token_peak_support``(글자별 최고 로그확률의 평균)다 —
        경로 총점은 star 커버리지에 지배되어 쓸 수 없다(그 함수 주석에 실측 근거가 있다).

        반환: (채택한 위치별 스팬, 판정 기록). 포기했으면 기록의 ``skipped``에 이유가 들어가고
        1패스 스팬이 그대로 나간다.
        """
        record: dict[str, Any] = {
            "spans": [[round(s, 2), round(e, 2)] for s, e in forbidden_spans],
            "tokens": len(char_positions),
            "blocks": len(blocks or []),
        }
        num_frames = int(emission.shape[1])
        frames = _forbidden_frames(forbidden_spans, num_frames, ratio).to(emission.device)
        record["frames"] = int(frames.numel())
        if frames.numel() == 0 and not blocks:
            record["skipped"] = "no_constraint"
            logger.info("Caption anchors constrain nothing here; alignment left untouched")
            return base_spans, record

        if frames.numel():
            # 실행가능성: 금지되지 않은 프레임이 토큰열이 필요로 하는 최소 프레임 수보다 적으면
            # 정렬 자체가 무의미해진다. star는 금지 프레임도 지나갈 수 있으므로 이 검사는 필요보다
            # 크게 잡은 값이다(= 보수적으로 더 자주 포기한다).
            free = num_frames - int(frames.numel())
            need = _min_ctc_frames(tokens)
            record["free_frames"] = free
            record["need_frames"] = need
            if free < need:
                record["skipped"] = "infeasible"
                logger.warning(
                    f"Caption anchors would leave {free} free frame(s) for {need} token frame(s); "
                    f"dropping the constraint and keeping the unconstrained alignment"
                )
                return base_spans, record

        # 금지 프레임 블록만 저장해 두고 제자리에서 누른다 — emission 전체 복제는 긴 곡에서
        # 수백 MB다(17분 곡 T≈5만 × vocab 1330 × 4B ≈ 271MB). 재채점은 원본에서 해야 하므로
        # finally로 반드시 되돌린다.
        saved = emission[0, frames, :].clone() if frames.numel() else None
        # (위치, 프레임오프셋, 경로) — 마스크를 되돌린 뒤 원본 emission으로 점수를 매긴다
        paths: list[tuple[int, int, torch.Tensor]] = []
        try:
            if saved is not None:
                emission[0, frames, :] = _FORBIDDEN_LOGP
                emission[0, frames, blank_id] = saved[:, blank_id]
                if star_id is not None:
                    emission[0, frames, star_id] = saved[:, star_id]
            if blocks:
                paths = self._align_in_blocks(
                    emission, tokens, blank_id, blocks, num_frames, char_positions, record
                )
                if not paths:
                    return base_spans, record
            else:
                path, _ = F.forced_align(emission, targets, blank=blank_id)
                paths = [(0, 0, path[0])]
        except Exception:
            record["skipped"] = "align_failed"
            logger.warning(
                "Constrained (caption-anchored) alignment failed; keeping the unconstrained "
                "alignment", exc_info=True,
            )
            return base_spans, record
        finally:
            if saved is not None:
                emission[0, frames, :] = saved

        # 여기서부터 emission은 원본이다 — 스팬 점수와 채택 판정 모두 원본으로 낸다.
        new_spans = _spans_by_position(emission, tokens, blank_id, paths, len(tokens))
        base_support = _token_peak_support(emission, base_spans, tokens, char_positions)
        new_support = _token_peak_support(emission, new_spans, tokens, char_positions)
        max_loss = float(
            getattr(self.config, "caption_anchor_max_token_loss", None)
            or _DEFAULT_ANCHOR_MAX_TOKEN_LOSS
        )
        loss = (
            base_support - new_support
            if base_support is not None and new_support is not None
            else float("inf")
        )
        adopt = math.isfinite(loss) and loss <= max_loss
        record.update(
            {
                "support": [
                    None if base_support is None else round(base_support, 4),
                    None if new_support is None else round(new_support, 4),
                ],
                "loss": round(loss, 4) if math.isfinite(loss) else None,
                "max_loss": max_loss,
                "adopted": adopt,
            }
        )
        # 경로 총점도 함께 남긴다 — 채택 판정에는 쓰지 않지만(star 커버리지에 지배된다) 이전
        # 실측치와 비교할 수 있어야 한다. 창 정렬은 전곡 경로가 아니므로 1패스만 낸다.
        if not blocks:
            record["path_score"] = [
                round(float(_path_frame_logprobs(emission, base_path).sum()), 2),
                round(float(_path_frame_logprobs(emission, paths[0][2]).sum()), 2),
            ]
        if not adopt:
            logger.warning(
                f"Caption-anchored alignment gives up {loss:.4f} nats/char of peak support "
                f"(> {max_loss}); keeping the unconstrained alignment — the constraint "
                f"(spans {record['spans']}, {record['blocks']} block(s)) probably disagrees "
                f"with where this song is actually sung"
            )
            return base_spans, record

        logger.info(
            f"Caption-anchored alignment adopted: {record['frames']} frame(s) forbidden across "
            f"{len(forbidden_spans)} span(s), {record['blocks']} anchored block(s); peak support "
            f"{base_support:.4f} → {new_support:.4f} (loss {loss:.4f} <= {max_loss})"
        )
        return new_spans, record

    def _align_in_blocks(
        self,
        emission: torch.Tensor,
        tokens: list[int],
        blank_id: int,
        blocks: list[tuple[int, int, int, int]],
        num_frames: int,
        char_positions: list[int],
        record: dict[str, Any],
    ) -> list[tuple[int, int, torch.Tensor]]:
        """블록별로 창 안 강제정렬. (토큰위치, 프레임오프셋, 경로) 목록 또는 빈 목록(포기).

        **순차화**: 블록 k+1의 창 시작은 「자막 시각 - 창」과 「블록 k가 실제로 끝난 프레임」 중
        늦은 쪽이다. 창이 겹치도록 잡혀 있어(양쪽 ±window) 그대로 두면 블록 k의 마지막 줄이
        블록 k+1의 첫 줄보다 늦게 끝날 수 있고, 그러면 라인 시각의 순서가 깨진다.

        「실제로 끝난 프레임」은 **마지막 글자 토큰의 끝**으로 잡는다. 블록 끝의 star는 점수가
        0.0이라 남은 창 전부를 삼키므로(그것이 star의 일이다) 그것을 기준으로 삼으면 다음 블록의
        창이 통째로 사라진다.

        창이 토큰열에 필요한 최소 프레임보다 짧으면 창을 뒤로 늘린다. 늘림이 연쇄되면 제약이
        점점 약해져 결국 제약 없는 배치로 수렴하는데, 그것이 안전한 실패 방향이다. 늘려도
        모자라면 전체를 포기한다.
        """
        char_set = set(char_positions)
        out: list[tuple[int, int, torch.Tensor]] = []
        cursor = 0
        widened = 0
        for p_lo, p_hi, f_lo0, f_hi0 in blocks:
            block_tokens = tokens[p_lo:p_hi]
            if not block_tokens:
                continue
            need = _min_ctc_frames(block_tokens)
            f_lo = max(int(f_lo0), cursor)
            f_hi = min(num_frames, max(int(f_hi0), f_lo + need))
            if f_hi > f_hi0:
                widened += 1
            if f_hi - f_lo < need:
                record["skipped"] = "infeasible_block"
                record["block_fail"] = [p_lo, p_hi, f_lo, f_hi, need]
                logger.warning(
                    f"Caption-anchored block [tokens {p_lo}:{p_hi}] needs {need} frame(s) but only "
                    f"{f_hi - f_lo} are available at frames {f_lo}:{f_hi}; dropping the whole "
                    f"anchored-block constraint and keeping the unconstrained alignment"
                )
                return []
            targets = torch.tensor([block_tokens], dtype=torch.int32, device=emission.device)
            path, _ = F.forced_align(
                emission[:, f_lo:f_hi, :].contiguous(), targets, blank=blank_id
            )
            out.append((p_lo, f_lo, path[0]))
            # 이 블록의 마지막 **글자** 토큰이 끝난 프레임 → 다음 블록의 하한
            bounds = _span_bounds(path[0], blank_id)
            last = max(
                (e for k, (_s, e) in enumerate(bounds) if (p_lo + k) in char_set),
                default=0,
            )
            cursor = f_lo + last
        record["widened_blocks"] = widened
        return out

    def _align_cjk(
        self,
        waveform: torch.Tensor,
        lyrics: list[LyricLine],
        language: str,
        progress_callback: Callable[[int, int], None] | None = None,
        line_candidates: dict[int, list[str]] | None = None,
        referee_margins: dict[int, float] | None = None,
        forbidden_spans: list[tuple[float, float]] | None = None,
        line_starts: dict[int, float] | None = None,
        vocal_presence: tuple | None = None,
    ) -> list[SyncResult]:
        device = self._get_device()

        if progress_callback:
            progress_callback(2, 5)

        # 오디오를 겹침 청크로 나눠 청크별 forward 후 CPU에서 log_softmax emission을
        # 스티칭한다 — 통짜 forward는 인코더 활성값이 길이 비례라 긴 곡에서 OOM
        # (실사고 2026-07-24: 17분 곡). 짧은 곡·비활성은 단일 청크라 통짜와 완전히
        # 동일한 device 텐서를 돌려받는다 (align_chunk_sec).
        emission = self._ctc_log_emission(waveform, device)

        if progress_callback:
            progress_callback(3, 5)

        vocab = self._processor.tokenizer.get_vocab()  # pyright: ignore[reportAttributeAccessIssue,reportOptionalMemberAccess]

        # 가사에 없는 가창(추임새/애드립/반복 후렴)을 흡수하는 와일드카드 star 채널.
        # forced_align은 정규화된 log_probs를 기대하며, star의 0.0(=log 1.0) 트릭은
        # log_softmax 정규화 후에만 작동한다 (raw logits에서는 0이 평범한 값이라 무효 —
        # 실측 검증됨). star 없이는 정규화가 Viterbi 경로에 영향을 주지 않으므로(프레임별
        # 상수 상쇄) 기존 결과와 동일하다.
        use_star = getattr(self.config, "star_tokens", False)
        star_id = emission.shape[-1]
        # 심판 채점과 greedy 디코딩은 star 채널을 **제외한** 원래 vocab 폭만 본다: star는
        # log(1.0)=0.0이라 argmax가 항상 star를 고르고, 후보 비교도 실제 토큰 확률로만
        # 해야 한다. 슬라이스로 보므로 텐서 복사가 없다(star를 붙인 뒤엔 뷰가 된다).
        base_dim = star_id
        num_frames = int(emission.shape[1])
        audio_length = waveform.shape[0] / 16000
        ratio = audio_length / num_frames
        if use_star:
            star_col = torch.zeros(
                (emission.shape[0], emission.shape[1], 1),
                dtype=emission.dtype,
                device=emission.device,
            )
            # star의 상수 0(=log 1)은 «어디서 흡수할지»에 선호가 없다는 뜻이고, 균일 바닥
            # posterior에서는 그 동점이 간주 오배치·재실행 비결정성의 뿌리다. f0 유성
            # presence가 주어지면 프레임별로 가격을 매긴다 — 유성 프레임의 star가
            # -weight×presence로 비싸져 star는 간주로, 글자는 가창 구간으로 간다.
            # 금지(-1e4)가 아니라 가격이라 남은 자리가 동점으로 남지 않는다
            # (전체 근거는 alignment/star_prior.py 모듈 주석).
            prior_weight = float(getattr(self.config, "star_prior_weight", 0.0) or 0.0)
            if vocal_presence is not None and prior_weight > 0:
                from everyric2.alignment.star_prior import star_frame_scores

                vals = star_frame_scores(
                    vocal_presence[0], vocal_presence[1], num_frames, ratio, prior_weight
                )
                star_col[0, :, 0] = torch.from_numpy(vals).to(
                    dtype=star_col.dtype, device=star_col.device
                )
                priced = float((vals < -0.5 * prior_weight).mean()) if num_frames else 0.0
                self._last_star_prior = {
                    "weight": prior_weight,
                    "priced_frac": round(priced, 3),
                }
                logger.info(
                    f"Star prior shaped: {priced:.0%} of {num_frames} frame(s) cost more than "
                    f"half weight (full presence = {prior_weight} nats/frame)"
                )
            emission = torch.cat([emission, star_col], dim=2)

        tokens = []
        line_boundaries = []
        char_info = []
        star_positions: list[int] = []
        current_pos = 0

        if use_star:
            # 인트로(첫 라인 전) 애드립 흡수용 선행 star
            star_positions.append(current_pos)
            tokens.append(star_id)
            current_pos += 1

        for line_idx, line in enumerate(lyrics):
            line_start = current_pos
            for char in line.text:
                # 조회용 글자는 소문자화·OOV 치환을 거칠 수 있지만, char_info의 "char"는
                # 원본을 그대로 넣는다 — 표시 문자열과 역매핑 인덱스가 원문 기준이어야
                # 하고, 타이밍만 치환 글자의 정렬에서 빌려 온다.
                tok_char = _resolve_token_char(char, vocab)
                if tok_char is not None:
                    char_info.append(
                        {
                            "char": char,
                            "line_idx": line_idx,
                            "token_idx": current_pos,
                        }
                    )
                    tokens.append(vocab[tok_char])
                    current_pos += 1

            if "|" in vocab:
                tokens.append(vocab["|"])
                current_pos += 1

            if use_star:
                # 라인 사이·마지막 라인 뒤의 가사 밖 가창 흡수.
                # 일본어 MMS 어댑터는 vocab에 "|"가 없어 star가 유일한 라인 간 완충이다.
                # star span은 char_info에 없으므로 라인 시각 계산에서 자동 제외된다.
                star_positions.append(current_pos)
                tokens.append(star_id)
                current_pos += 1

            line_boundaries.append((line_start, current_pos - 1))

        if not tokens:
            raise AlignmentError("No valid tokens found in lyrics for this language")

        blank_id = self._processor.tokenizer.pad_token_id  # pyright: ignore[reportAttributeAccessIssue,reportOptionalMemberAccess]
        try:
            # emission이 청킹으로 CPU에 있으면 targets도 같은 디바이스여야 한다
            targets = torch.tensor([tokens], dtype=torch.int32, device=emission.device)
            aligned_tokens, alignment_scores = F.forced_align(emission, targets, blank=blank_id)
            token_spans = F.merge_tokens(aligned_tokens[0], alignment_scores[0], blank=blank_id)
        except Exception as e:
            raise AlignmentError(f"CTC forced alignment failed: {e}")

        # 자막 앵커 2패스는 **이 안에** 둔다 — `_ctc_log_emission`은 `_align_cjk` 진입마다
        # 모델 forward를 다시 돌리므로(캐시 없음, 4.7분 곡 ~9s) 바깥에서 align()을 다시 부르면
        # 그 비용을 또 낸다. 안에서는 같은 emission을 재사용해 forward가 0이다.
        if forbidden_spans or line_starts:
            window_sec = float(getattr(self.config, "caption_anchor_window_sec", None) or 5.0)
            blocks = (
                _anchor_blocks(
                    line_starts,
                    len(lyrics),
                    line_boundaries,
                    len(tokens),
                    num_frames,
                    ratio,
                    window_sec,
                )
                if line_starts
                else None
            )
            token_spans, self._last_caption_anchor = self._caption_anchor_pass(
                emission,
                targets,
                tokens,
                blank_id,
                star_id if use_star else None,
                aligned_tokens[0],
                token_spans,
                forbidden_spans or [],
                ratio,
                [ci["token_idx"] for ci in char_info],
                blocks=blocks,
            )

        if progress_callback:
            progress_callback(4, 5)

        # 1패스 결과로 라인별 프레임 창이 정해진다 — 그 창 안에서만 후보를 심판하고
        # greedy 디코딩을 뜬다. 정렬된 글자가 0개인 라인(전부 OOV)은 창이 없어 제외된다.
        line_window = _line_frame_windows(char_info, token_spans)
        self._last_heard = self._heard_lines(emission, base_dim, blank_id, vocab, line_window)
        self._last_heard_spans = self._heard_span_lines(
            emission, base_dim, blank_id, vocab, line_window, ratio
        )
        # 이긴 후보의 슬라이스 정렬 결과를 그대로 쓴다 → 3패스가 필요 없다.
        referee_overrides, chosen_text = self._referee_lines(
            emission,
            base_dim,
            blank_id,
            vocab,
            line_window,
            line_candidates,
            referee_margins,
        )

        # star가 실제로 흡수한 구간 기록 (1프레임=20ms짜리 형식적 흡수는 제외)
        self._last_star_spans = []
        for idx in star_positions:
            if idx < len(token_spans) and token_spans[idx] is not None:
                span = token_spans[idx]
                s, e = span.start * ratio, span.end * ratio
                if e - s >= 0.1:
                    self._last_star_spans.append((round(s, 2), round(e, 2)))

        self._last_word_timestamps = []
        line_char_timestamps: dict[int, list[WordTimestamp]] = {}

        # 심판이 교체한 라인은 1패스 스팬 대신 이긴 후보의 슬라이스 정렬 스팬을 쓴다.
        # 그 밖의 라인은 예전과 완전히 같은 순서·값으로 채워진다 (후보가 없으면 항등).
        line_chars: dict[int, list[dict]] = {}
        for ci in char_info:
            line_chars.setdefault(ci["line_idx"], []).append(ci)

        for line_idx in range(len(lyrics)):
            items = referee_overrides.get(line_idx)
            if items is None:
                items = [
                    (ci["char"], span.start, span.end, float(span.score))
                    for ci in line_chars.get(line_idx, [])
                    if ci["token_idx"] < len(token_spans)
                    for span in (token_spans[ci["token_idx"]],)
                    if span is not None
                ]
            for char, start_frame, end_frame, score in items:
                # emission이 log_softmax라 score는 평균 로그확률(음수) — 그대로
                # 내보내면 클라이언트 신뢰도 표시가 전부 '낮음'으로 찍힌다.
                # exp로 기하평균 확률(0~1)로 변환해 저장한다.
                wt = WordTimestamp(
                    word=char,
                    start=start_frame * ratio,
                    end=end_frame * ratio,
                    confidence=round(math.exp(min(0.0, score)), 6),
                )
                self._last_word_timestamps.append(wt)
                if line_idx not in line_char_timestamps:
                    line_char_timestamps[line_idx] = []
                line_char_timestamps[line_idx].append(wt)

        from everyric2.inference.prompt import WordSegment

        # 1) 줄별 정렬 결과 수집 (정렬 실패 = 시각 None)
        line_times: list[list] = []
        for line_idx, line in enumerate(lyrics):
            char_ts = line_char_timestamps.get(line_idx, [])
            if char_ts:
                word_segments = [
                    WordSegment(word=wt.word, start=wt.start, end=wt.end, confidence=wt.confidence)
                    for wt in char_ts
                ]
                line_times.append([char_ts[0].start, char_ts[-1].end, word_segments])
            else:
                # OOV 등으로 정렬된 글자가 0개 → 아래에서 이웃 사이로 보간
                line_times.append([None, None, None])

        # 2) 정렬 실패 줄 보간. 균등 배치(line_idx*total/N)는 실제 정렬 시각과 뒤섞여
        #    순서가 깨지므로(역순·겹침), 앞뒤 정렬 줄 사이 간격에 끼워넣어 순서를 보존한다.
        self._interpolate_unaligned(line_times, audio_length)

        # 3) SyncResult 생성. 심판이 독음을 바꾼 라인은 **이긴 후보를 text로 내보낸다** —
        #    호출부(worker의 독음 역매핑)가 이 텍스트로 모라를 쪼개야 음절 스팬과 표시
        #    발음이 실제로 정렬된 독음과 일치한다.
        results = [
            SyncResult(
                line_number=line.line_number,
                text=chosen_text.get(i, line.text),
                start_time=line_times[i][0],
                end_time=line_times[i][1],
                word_segments=line_times[i][2],
            )
            for i, line in enumerate(lyrics)
        ]

        if progress_callback:
            progress_callback(5, 5)

        return results

    @staticmethod
    def _interpolate_unaligned(
        line_times: list[list],
        total_duration: float,
    ) -> None:
        """정렬 실패(시각 None) 줄을 앞뒤 정렬 줄 사이 간격에 균등 분배(순서 보존).

        line_times: 각 줄 [start, end, word_segments] (정렬 실패는 [None, None, None]).
        제자리에서 start/end를 채운다. 전부 실패면 전체 구간에 균등 분배.
        """
        n = len(line_times)
        i = 0
        while i < n:
            if line_times[i][0] is not None:
                i += 1
                continue
            group_start = i
            group_end = i
            while group_end < n and line_times[group_end][0] is None:
                group_end += 1
            group_end -= 1

            prev_end = line_times[group_start - 1][1] if group_start > 0 else 0.0
            if group_end < n - 1 and line_times[group_end + 1][0] is not None:
                next_start = line_times[group_end + 1][0]
            else:
                next_start = total_duration

            available = max(0.0, next_start - prev_end)
            num = group_end - group_start + 1
            seg = available / num if num else 0.0
            if seg < 0.1:  # 빈틈이 거의 없을 때도 최소 길이 보장(근사치)
                seg = 0.1
            for j in range(group_start, group_end + 1):
                off = j - group_start
                line_times[j][0] = prev_end + off * seg
                line_times[j][1] = prev_end + (off + 1) * seg
            i = group_end + 1

    def _align_mms(
        self,
        waveform: torch.Tensor,
        lyrics: list[LyricLine],
        language: str,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[SyncResult]:
        device = self._get_device()
        bundle = self._processor

        if progress_callback:
            progress_callback(2, 5)

        dictionary = bundle.get_dict(star=None)  # pyright: ignore[reportAttributeAccessIssue]

        all_words = []
        line_word_counts = []

        for line in lyrics:
            words = line.text.lower().split()
            cleaned_words = []
            for word in words:
                cleaned = "".join(c for c in word if c in dictionary and dictionary[c] != 0)
                if cleaned:
                    cleaned_words.append(cleaned)
                    all_words.append(cleaned)
            line_word_counts.append(len(cleaned_words))

        if not all_words:
            raise AlignmentError("No valid characters found in lyrics for MMS_FA model")

        if progress_callback:
            progress_callback(3, 5)

        # 통짜 forward는 활성값이 길이 비례라 긴 곡에서 OOM — 겹침 청크로 나눠 추론 후
        # CPU에서 emission을 스티칭한다. 짧은 곡·비활성은 단일 청크=통짜와 동일 (align_chunk_sec).
        emission = self._mms_emission(waveform, device)

        tokens = [
            dictionary[c]
            for word in all_words
            for c in word
            if c in dictionary and dictionary[c] != 0
        ]

        try:
            aligned_tokens, alignment_scores = F.forced_align(
                emission,
                torch.tensor([tokens], dtype=torch.int32, device=emission.device),
                blank=0,
            )
            token_spans = F.merge_tokens(aligned_tokens[0], alignment_scores[0])
        except Exception as e:
            raise AlignmentError(f"MMS_FA forced alignment failed: {e}")

        if progress_callback:
            progress_callback(4, 5)

        word_lengths = [len(word) for word in all_words]
        word_spans = []
        idx = 0
        for length in word_lengths:
            word_spans.append(token_spans[idx : idx + length])
            idx += length

        num_frames = emission.shape[1]
        ratio = waveform.shape[0] / num_frames / 16000

        self._last_word_timestamps = []
        for word, spans in zip(all_words, word_spans):
            if spans:
                start_time = spans[0].start * ratio
                end_time = spans[-1].end * ratio
                avg_score = sum(s.score for s in spans) / len(spans)

                self._last_word_timestamps.append(
                    WordTimestamp(
                        word=word,
                        start=start_time,
                        end=end_time,
                        confidence=avg_score,
                    )
                )

        results = []
        word_idx = 0
        audio_length = waveform.shape[0] / 16000

        for line_idx, line in enumerate(lyrics):
            word_count = line_word_counts[line_idx]

            if word_count > 0 and word_idx < len(word_spans):
                line_spans = word_spans[word_idx : word_idx + word_count]
                if line_spans and line_spans[0] and line_spans[-1]:
                    start_time = line_spans[0][0].start * ratio
                    end_time = line_spans[-1][-1].end * ratio
                else:
                    start_time = line_idx * audio_length / len(lyrics)
                    end_time = (line_idx + 1) * audio_length / len(lyrics)
                word_idx += word_count
            else:
                start_time = line_idx * audio_length / len(lyrics)
                end_time = (line_idx + 1) * audio_length / len(lyrics)

            results.append(
                SyncResult(
                    line_number=line.line_number,
                    text=line.text,
                    start_time=start_time,
                    end_time=end_time,
                )
            )

        if progress_callback:
            progress_callback(5, 5)

        return results

    def align(
        self,
        audio: AudioData,
        lyrics: list[LyricLine],
        language: str | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        line_candidates: dict[int, list[str]] | None = None,
        referee_margins: dict[int, float] | None = None,
        forbidden_spans: list[tuple[float, float]] | None = None,
        line_starts: dict[int, float] | None = None,
        vocal_presence: tuple | None = None,
    ) -> list[SyncResult]:
        """가사를 오디오에 강제정렬한다.

        Args:
            forbidden_spans: **가사 줄이 놓일 수 없는 (start, end) 구간(초)** 목록. 주면 1패스
                정렬 뒤 그 구간 프레임에서 blank·star 외 모든 열을 눌러 한 번 더 정렬한다.
            line_starts: **line_idx → 그 줄이 시작해야 하는 시각(초)**. 주면 그 줄들을 칸막이로
                전곡을 블록으로 나눠 각 블록을 자기 프레임 창 안에서만 정렬한다 — 「여기 있을 수
                없다」(forbidden_spans)가 함의하지 못하는 「여기 있어야 한다」를 건다.
                둘 다 주면 함께 걸린다. 두 경우 모두 제약을 지키느라 잃은 글자별 음향 근거가
                ``caption_anchor_max_token_loss`` 이하일 때만 결과를 채택한다
                (``_caption_anchor_pass``). 출처는 사람이 만든 유튜브 자막의 타임스탬프다
                (``alignment.caption_anchors``). **둘 다 선택적 인자다** — 주지 않으면 기존
                동작과 완전히 동일하다.
            line_candidates: 라인 인덱스 → 그 라인의 **대체 독음 후보 목록**(``[0]``이 기본값).
                주면 1패스 정렬 뒤 라인 프레임 창 안에서 후보들을 같은 emission에 다시
                강제정렬해, 기본값을 마진 이상 이기는 후보가 있을 때만 그 라인을 교체한다
                (``_referee_lines``). 교체된 라인의 ``SyncResult.text``는 이긴 후보다.
                **선택적 인자다** — 주지 않으면(또는 후보가 1개면) 기존 동작과 완전히 동일하다.
            referee_margins: 라인별 마진(토큰당 평균 로그확률 차). 없는 라인은 설정
                ``pron_referee_margin``을 쓴다. 사람이 쓴 발음이 기본값인 라인처럼 더 큰
                마진을 요구해야 하는 경우에 쓴다.
            vocal_presence: ``(times, presence)`` — f0 유성 지시자에서 만든 보컬 존재
                신호 (``star_prior.vocal_presence_from_f0``). 주면 star 채널을 상수 0
                대신 프레임별 ``-star_prior_weight × presence``로 성형한다 — 균일 바닥
                posterior에서 «어디서 흡수할지» 동점을 없애는 가격 신호다. **선택적
                인자다** — 주지 않으면(또는 star_tokens가 꺼져 있으면) 기존 동작과
                완전히 동일하다.
        """
        # 심판·「들은 것」 기록은 정렬마다 새로 쓴다 — 이전 정렬(예: ko)의 기록이 다음
        # 정렬(ja) 뒤에 남아 있으면 호출부가 엉뚱한 라인 창의 텍스트를 보고한다.
        self._last_referee = []
        self._last_heard = {}
        self._last_heard_spans = {}
        self._last_caption_anchor = None
        self._last_star_prior = None
        force_mms = False
        if language and language != "auto":
            resolved_lang = language
        else:
            lyrics_text = " ".join(line.text for line in lyrics)
            resolved_lang, is_multilingual = detect_language_from_text(lyrics_text)
            if is_multilingual:
                force_mms = True
            logger.info(f"Auto-detected language: {resolved_lang}, multilingual: {is_multilingual}")

        # 모델 로드부터 추론까지를 한 락으로 묶는다. 어댑터 교체는 상주 모델을 제자리에서
        # 바꾸므로, 다른 스레드가 forward 중일 때 끼어들면 어댑터와 가사 토큰의 언어가
        # 어긋난 채 정렬된다(max_concurrent_jobs>1 설정에서 발생 가능). 기본값 1에서는
        # 경합이 없어 비용 0이고, RLock이라 같은 스레드의 중첩 호출은 그대로 통과한다.
        with self._model_lock:
            self._ensure_model_loaded(resolved_lang, force_mms=force_mms)

            if progress_callback:
                progress_callback(1, 5)

            from everyric2.audio.loader import AudioLoader

            loader = AudioLoader()
            prepared = loader.prepare_for_alignment(audio, target_sr=16000, normalize=True)

            waveform = torch.from_numpy(prepared.waveform.astype("float32"))
            if waveform.dim() == 2:
                waveform = waveform.mean(dim=0)

            if resolved_lang in LANG_MODEL_MAP:
                results = self._align_cjk(
                    waveform,
                    lyrics,
                    resolved_lang,
                    progress_callback,
                    line_candidates=line_candidates,
                    referee_margins=referee_margins,
                    forbidden_spans=forbidden_spans,
                    line_starts=line_starts,
                    vocal_presence=vocal_presence,
                )
                self._last_match_stats = self._calculate_match_stats(results)
                return results
            else:
                # MMS_FA(라틴) 경로는 글자 단위 vocab이 아니라 단어 사전을 쓰고 라인 창도
                # 다르게 잡힌다 — 심판은 CJK 경로 전용이므로 후보는 조용히 무시한다.
                if line_candidates:
                    logger.info(
                        f"Pronunciation referee skipped: MMS_FA path ({resolved_lang}) "
                        f"does not score per-line candidates"
                    )
                if forbidden_spans or line_starts:
                    # 같은 사정으로 앵커 제약도 CJK 경로 전용이다 — 라틴 경로는 emission을
                    # 이 함수 밖(bundle)에서 만들고 토큰 단위도 달라 마스킹 지점이 없다.
                    logger.info(
                        f"Caption anchors skipped: MMS_FA path ({resolved_lang}) has no "
                        f"character-level emission to mask"
                    )
                if vocal_presence is not None:
                    # star 성형도 같은 사정 — 이 경로는 star 채널 자체가 없다(with_star=False).
                    logger.info(
                        f"Star prior skipped: MMS_FA path ({resolved_lang}) has no star channel"
                    )
                results = self._align_mms(waveform, lyrics, resolved_lang, progress_callback)
                from everyric2.alignment.matcher import LyricsMatcher

                matcher = LyricsMatcher()
                matched_results = matcher.match_lyrics_to_words(
                    lyrics, self._last_word_timestamps, resolved_lang
                )

                self._last_match_stats = self._calculate_match_stats(matched_results)
                return matched_results

    def _calculate_match_stats(self, results: list) -> "MatchStats":
        """Calculate match stats consistently for both CJK and MMS paths.

        Uses adaptive threshold: > 0 for positive confidence (CJK),
        > -5 for log-probability confidence (MMS/English).
        """
        from everyric2.alignment.matcher import MatchStats

        all_confidences = [
            seg.confidence
            for r in results
            if r.word_segments
            for seg in r.word_segments
            if seg.confidence is not None
        ]

        if not all_confidences:
            return MatchStats(
                total_lyrics=len(results),
                matched_lyrics=0,
                match_rate=0.0,
                avg_confidence=0.0,
            )

        avg_conf = sum(all_confidences) / len(all_confidences)

        # Adaptive threshold: log-probs are negative, CTC scores can be positive
        # For log-probs (avg < 0): use -5 as "good enough" threshold
        # For CTC scores (avg >= 0): use 0 as threshold
        threshold = -5.0 if avg_conf < 0 else 0.0
        good_matches = sum(1 for c in all_confidences if c > threshold)
        match_rate = good_matches / len(all_confidences)

        return MatchStats(
            total_lyrics=len(results),
            matched_lyrics=sum(1 for r in results if r.word_segments),
            match_rate=match_rate,
            avg_confidence=avg_conf,
        )

    def get_last_referee_decisions(self) -> list[dict]:
        """직전 정렬에서 심판이 채점한 라인들의 판정 기록 (line_idx 오름차순).

        각 항목: ``line``, ``default``, ``chosen``, ``margin``, ``gain``, ``frames``,
        ``scores``(후보별 토큰당 평균 로그확률). ``chosen != default``인 항목이 교체된 라인이다.
        """
        return list(self._last_referee)

    def get_last_caption_anchor(self) -> dict[str, Any] | None:
        """직전 정렬의 자막 앵커 판정 기록 — 앵커를 주지 않았으면 None.

        항목: ``spans``(금지 구간), ``frames``/``free_frames``/``need_frames``(실행가능성),
        ``score``([1패스, 2패스] 원본 emission 재채점 총점), ``loss``(글자당 포기액),
        ``max_loss``, ``adopted``, 포기했으면 ``skipped``(no_frames|infeasible|align_failed).
        """
        return dict(self._last_caption_anchor) if self._last_caption_anchor else None

    def get_last_heard_lines(self) -> dict[int, str]:
        """직전 정렬에서 모델이 각 라인 프레임 창에서 「들은」 greedy 디코딩 텍스트."""
        return dict(self._last_heard)

    def get_last_heard_spans(self) -> dict[int, list[tuple[str, float]]]:
        """직전 정렬에서 모델이 각 라인 프레임 창에서 「들은」 greedy 토큰의 글자별 시각."""
        return dict(self._last_heard_spans)

    def get_last_transcription_data(self) -> tuple[list[WordTimestamp], MatchStats | None, str]:
        return (self._last_word_timestamps, self._last_match_stats, "ctc")

    def get_transcription_sets(self) -> list[tuple[list[WordTimestamp], MatchStats | None, str]]:
        data = self.get_last_transcription_data()
        if data[0]:
            return [data]
        return []

    @staticmethod
    def get_engine_type() -> Literal["ctc"]:
        return "ctc"


# 웜 캐시 싱글턴 (WS2-A) — 프로세스 수명 동안 CTC 엔진(과 그 안에 lazy 로드된 wav2vec2/MMS
# 모델)을 상주시킨다. torch를 최상위에서 import하는 모듈이라, 이 접근자는 반드시 호출부에서
# 지연 import해야 API 전용 모드(local_worker=false)에 torch가 딸려 들어오지 않는다.
_shared_ctc_engine: "CTCEngine | None" = None
_shared_ctc_lock = threading.Lock()


def get_shared_ctc_engine(config: AlignmentSettings | None = None) -> "CTCEngine":
    """웜 캐시된 CTCEngine을 돌려준다 (EVERYRIC_SERVER_WARM_MODELS 기준).

    엔진 인스턴스는 _ensure_model_loaded가 로드한 모델을 _current_lang로 캐시하므로, 같은
    엔진을 재사용하면 같은 언어의 두 번째 잡부터 모델 재로드가 0회다. 재사용 시 "warm model
    reuse: ctc" 1줄. warm이 꺼져 있으면 매번 새 엔진(기존 동작)."""
    from everyric2.config.settings import get_settings

    if not get_settings().server.warm_models:
        return CTCEngine(config)
    global _shared_ctc_engine
    with _shared_ctc_lock:
        if _shared_ctc_engine is None:
            _shared_ctc_engine = CTCEngine(config)
        else:
            logger.info("warm model reuse: ctc")
        return _shared_ctc_engine


def clear_shared_ctc_engine() -> None:
    """웜 캐시 해제 (VRAM 가드용) — 다음 요청에서 지연 재생성된다."""
    global _shared_ctc_engine
    with _shared_ctc_lock:
        _shared_ctc_engine = None
