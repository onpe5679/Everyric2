"""YouTube audio downloader using yt-dlp."""

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from everyric2.config.settings import AudioSettings, get_settings

logger = logging.getLogger(__name__)

# 다운로드 산출물로 인정하는 확장자. postprocessor를 "best"(스트림카피)로 돌리면 소스 코덱에
# 따라 결과가 opus/m4a/webm 등으로 갈린다 — wav 하나만 찾던 예전 글롭은 이제 못 찾는다.
# 순서가 판정이다: opus를 가장 먼저 봐 스트림카피 산출물을 우선 채택하고, wav는 로컬 트랜스코드
# 구제 경로(_ensure_decodable)의 산출물이라 맨 뒤에 둔다.
_DOWNLOAD_EXTS: tuple[str, ...] = (".opus", ".m4a", ".webm", ".ogg", ".mp3", ".wav")


class DownloadError(Exception):
    """Base exception for download operations.

    분류된 실패는 ``code``/``terminal``/``cause_text``를 채워 올라온다
    (`classify_download_error`). 이때 ``str(e)``는 **사용자에게 그대로 보여줄 한국어 문구**다 —
    잡 실패 문구가 ``error=str(e)``로 저장되기 때문이다(server/worker.py의 잡 마감 경로).
    분류하지 못한 실패는 세 필드가 기본값으로 남고 ``str(e)``는 yt-dlp 영문 원문이다(기존 동작).
    """

    #: 실패 분류 코드 (분류 못 하면 "unknown") — 로그·상위 분기용
    code: str = "unknown"
    #: 재시도해도 결과가 같은가. False면 재시도할 가치가 있다(403·네트워크).
    terminal: bool = False
    #: yt-dlp 원문 (로그·디버그용, 사용자 노출용이 아님)
    cause_text: str | None = None


class InvalidURLError(DownloadError):
    """Raised when URL is invalid."""

    pass


class VideoUnavailableError(DownloadError):
    """Raised when video is unavailable."""

    pass


class DependencyError(DownloadError):
    """Raised when required dependency is missing."""

    pass


# ── yt-dlp 실패 분류 ─────────────────────────────────────────────────────
#
# 예전 구현은 모든 실패를 ``DownloadError(f"Download failed: {e}")``로 접었다. 403 / JS 런타임
# 미설치 / 지역 차단 / 나이 제한 / 삭제가 전부 같은 영문 덤프가 되고, 그것이 잡 에러
# (``update_status(..., error=str(e))``)로 올라가 확장이 사용자에게 ``ERROR: [youtube] ...``
# 원문을 그대로 보여줬다. 사용자가 할 수 있는 일이 있는 경우조차 알려주지 못했다.
#
# **근거의 한계를 먼저 적는다**: yt-dlp는 실패 사유를 구조화된 코드로 주지 않는다. 전용 예외가
# 있는 것은 지역 차단(``yt_dlp.utils.GeoRestrictedError``, utils/_utils.py:1037)뿐이고, 삭제·
# 비공개·봇 확인 같은 사유는 유튜브의 ``playabilityStatus.reason``을 yt-dlp가 문자열로 그대로
# 전달하는 것이라(extractor/youtube/_video.py:4040-4063) yt-dlp 소스에 상수로도 없다. 그래서
# 문자열 패턴이 유일한 수단이고, 패턴마다 근거를 아래에 남긴다(yt-dlp 2026.07.04 소스 실측).
#
# **못 가리는 실패는 분류하지 않고 기존 동작(영문 원문 노출)으로 폴백한다** — 엉뚱한 한국어
# 안내가 원문보다 해롭다. 라이브/예약 방송과 멤버십 전용은 유튜브 reason 원문에만 나타나고
# 설치된 yt-dlp 소스에서 문자열을 확인할 수 없어 **의도적으로 분류하지 않았다**(확인 못 한
# 추측 패턴은 넣지 않는다). 그 둘은 지금도 원문이 그대로 노출된다.


@dataclass(frozen=True)
class DownloadFailure:
    """분류된 다운로드 실패. ``message``가 사용자에게 보일 한국어 문구다."""

    code: str
    message: str
    #: 재시도해도 결과가 같은가 (False = 재시도 가치 있음)
    terminal: bool


# 코드 → 사용자 문구. 문구는 **무엇을 하면 되는지**를 말한다. 쿠키 등록은 실재하는 경로다
# (POST /api/cookies — server/api/cookies.py). 확장에는 쿠키 UI가 없으므로 "확장에서"라고
# 쓰지 않는다.
_FAILURES: dict[str, DownloadFailure] = {
    "js_runtime": DownloadFailure(
        "js_runtime",
        "서버 설정 문제로 유튜브 오디오를 받지 못했어요 (JavaScript 런타임 미설치). "
        "사용자가 할 수 있는 조치는 없어요 — 서버 관리자에게 알려 주세요.",
        terminal=True,
    ),
    "age_restricted": DownloadFailure(
        "age_restricted",
        "나이 제한이 걸린 영상이에요. 서버에 유튜브 로그인 쿠키를 등록하면 받을 수 있어요.",
        terminal=True,
    ),
    "login_required": DownloadFailure(
        "login_required",
        "유튜브가 로그인 확인을 요구했어요. 서버에 유튜브 로그인 쿠키를 등록하면 받을 수 있어요.",
        terminal=True,
    ),
    # 봇 확인은 login_required와 갈라 둔다 — 문구는 같은 계열이지만 **주소를 바꿔 재시도할
    # 값어치가 있는 유일한 인증 계열 실패**다 (_ADDRESS_RETRYABLE_CODES 근거 참고).
    "bot_check": DownloadFailure(
        "bot_check",
        "유튜브가 이 서버를 자동 접속으로 의심해 막았어요. 조금 뒤에 다시 시도하거나, "
        "서버에 유튜브 로그인 쿠키를 등록하면 풀려요.",
        terminal=False,
    ),
    "geo_blocked": DownloadFailure(
        "geo_blocked",
        "이 영상은 서버가 있는 지역에서 막혀 있어요. 다른 업로드본을 골라 보거나, "
        "해당 지역 쿠키·회선을 등록해야 받을 수 있어요.",
        terminal=True,
    ),
    "unavailable": DownloadFailure(
        "unavailable",
        "영상을 받을 수 없어요 (삭제됐거나 비공개예요). 다른 영상을 골라 주세요.",
        terminal=True,
    ),
    "throttled": DownloadFailure(
        "throttled",
        "유튜브가 이 서버의 다운로드를 잠시 막았어요 (403). 조금 뒤에 다시 시도해 주세요.",
        terminal=False,
    ),
    "network": DownloadFailure(
        "network",
        "유튜브에 연결하지 못했어요 (네트워크 오류). 조금 뒤에 다시 시도해 주세요.",
        terminal=False,
    ),
}

# 순서가 판정이다 — 위에서 아래로 처음 걸리는 것을 채택하므로 **더 구체적인 것을 먼저** 둔다.
# (예: "Sign in to confirm your age"는 나이 제한 패턴과 로그인 패턴에 모두 걸리는데, 나이
#  제한이 먼저라 더 정확한 안내가 나간다. 지역 차단 문구에는 "not available"이 들어 있어
#  삭제·비공개보다 먼저 판정해야 한다.)
_FAILURE_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # (d) 서버 구성 문제 — 사용자가 할 수 있는 게 없다.
    # 근거: extractor/youtube/_video.py:2985 "No supported JavaScript runtime could be found."
    #       extractor/youtube/_video.py:3333 "Ensure you have a supported JavaScript runtime and
    #       challenge solver script distribution installed."
    ("js_runtime", ("supported javascript runtime",)),
    # (a) 사용자 조치 가능 — 나이 제한.
    # 근거: extractor/youtube/_video.py:2894 AGE_GATE_REASONS
    #       = ('confirm your age', 'age-restricted', 'inappropriate',
    #          'age_verification_required', 'age_check_required')
    #       + :3149 "This video is age-restricted; some formats may be missing"
    #       + :3157 "This video is age-restricted and YouTube is requiring account age-verification"
    (
        "age_restricted",
        (
            "confirm your age",
            "age-restricted",
            "age_verification_required",
            "age_check_required",
            "age-verification",
        ),
    ),
    # (a/c) 봇 확인 — 계정이 아니라 **접속 IP 평판**에 걸리는 게이트라 주소를 바꾸면 풀릴 수
    # 있다. 그래서 login_required와 갈라 재시도 대상으로 둔다. "Sign in to confirm you're not a
    # bot"은 유튜브 reason 원문으로 그대로 전달되고(_video.py:4040-4063) 이 프로젝트에서 실제로
    # 관측된 문구다. login_required보다 먼저 둬야 한다 — 이 문구에는 로그인 힌트가 함께 붙어
    # 나와서 순서가 뒤면 로그인으로 잡힌다.
    ("bot_check", ("not a bot", "confirm you're not")),
    # (a) 사용자 조치 가능 — 로그인/쿠키 요구.
    # 근거: extractor/common.py:1246 raise_login_required 기본 문구
    #       "This video is only available for registered users"
    #       extractor/youtube/_base.py:670 "Login details are needed to download this content."
    #       extractor/common.py:601-604 로그인 힌트 "Use --cookies-from-browser or --cookies ..."
    (
        "login_required",
        (
            "sign in to confirm",
            "only available for registered users",
            "login details are needed",
            "--cookies-from-browser",
        ),
    ),
    # (a) 사용자 조치 가능 — 지역 차단. 전용 예외(GeoRestrictedError)로도 먼저 잡지만,
    # yt-dlp가 문구만 남기고 예외를 접는 경로가 있어 패턴도 함께 둔다.
    # 근거: extractor/common.py:1257 "This video is not available from your location due to geo
    #       restriction" / extractor/youtube/_video.py:4043 subreason.startswith(
    #       'The uploader has not made this video available in your country') → raise_geo_restricted
    (
        "geo_blocked",
        (
            "geo restriction",
            "not available from your location",
            # 근거 문자열은 "...not made this video available in your country"지만 패턴은
            # "in your country"까지 넓힌다 — 유튜브 reason에는 "Video unavailable. ... has
            # blocked it in your country" 변형이 있고, 좁게 두면 그 변형이 아래 unavailable
            # (삭제·비공개)로 먼저 걸려 "다른 영상을 고르라"는 잘못된 안내가 나간다.
            # 지역 차단이 아닌데 이 구절이 나오는 실패는 없으므로 오탐 위험이 낮다.
            "in your country",
        ),
    ),
    # (b) 영구 실패 — 삭제·비공개. 예전 구현의 판정("unavailable"/"private" 부분일치)을 그대로
    # 유지해 기존에 VideoUnavailableError가 되던 실패가 계속 같은 분류로 남게 한다.
    # 근거: extractor/youtube/_video.py:197 'Private video', :770 'This video is not available.',
    #       :817 'This video has been removed for violating YouTube's policy ...'
    ("unavailable", ("unavailable", "private", "has been removed")),
    # (c) 일시 실패 — 403 스로틀. 재시도 가치가 있다. 이 프로젝트에서 실제로 겪은 실패이고
    # 회선 우회 설정이 그 대응책으로 존재한다(README의 EVERYRIC_AUDIO_SOURCE_ADDRESS =
    # "다운로드 회선 바인딩 (403 스로틀 우회)"). 관리자 대응책은 yt-dlp 업데이트/회선 교체이고
    # 사용자가 할 수 있는 것은 재시도뿐이라 문구는 재시도만 안내한다.
    # 429(Too Many Requests)도 같은 계열이다 — 유튜브가 이 **접속 IP**의 요청 속도를 막은
    # 것이므로 주소를 바꾸면 풀릴 수 있다(운영자 요청 ②의 "403 / 429 / blocked 계열").
    (
        "throttled",
        (
            "http error 403",
            "403: forbidden",
            "403 forbidden",
            "http error 429",
            "429: too many requests",
            "too many requests",
        ),
    ),
    # (c) 일시 실패 — 네트워크.
    ("network", ("timed out", "timeout", "connection reset", "connection refused",
                 "temporary failure in name resolution", "network is unreachable",
                 "unable to download webpage")),
)


# ── 다운로드 egress 순회 ────────────────────────────────────────────────
#
# 운영자가 공인 IP를 추가 확보해 공개 서비스 트래픽을 전용 대역으로 격리했고, 그 결과 워커 env에
# 출구가 **하나** 박혀 있다(EVERYRIC_AUDIO_SOURCE_ADDRESS). 예전 구현은 그 값을 한 번 박고
# 끝이라 그 출구가 막히는 순간 다운로드가 통째로 멈췄다 — 야간 배치 실패 원장의
# ``Download failed: ERROR: unable to download video data: HTTP Error 403``이 이 경로다.
# 그래서 egress를 목록으로 받아 **한 바퀴만** 순회한다.
#
# **계층 분할**: "재시도할까"는 앱만 알 수 있고(에러의 의미는 HTTPS 안에 있어서 네트워크
# 계층은 CONNECT 터널의 바이트만 본다), "어느 출구로 나갈까"는 호스트만 안다(토폴로지).
# 그래서 이 파일은 egress 값을 **불투명 문자열**로만 다룬다 — IP 형태 검증을 하지 않는다.
# 오늘은 그 값이 로컬 IP(source_address)이고, 장기적으로는 회선별 로컬 프록시 URL
# (http://127.0.0.1:3128)이 된다. 전환 지점은 _apply_egress 하나뿐이다.
#
# 어떤 실패에 출구를 바꿔 볼 것인가 — 이 비대칭이 이 기능의 핵심이고 **앱만 할 수 있는 부분**
# 이다. 영상 탓인 실패에 출구를 돌리면 회수는 0인데 유튜브 접촉만 출구 수만큼 배수로 늘어
# **차단을 피하려다 차단을 부른다.** 그래서 "그 실패가 우리가 어느 출구로 나갔는지에 달려
# 있는가"만 기준으로 가른다. (아래 근거에 나오는 "공인 IP"는 **유튜브가 보는 쪽**을 말한다 —
# 그 IP를 source_address로 고르든 프록시로 고르든 판정은 그대로 성립한다.)
#
#   재시도 O
#     throttled  : 403/429는 유튜브가 이 공인 IP의 요청을 막은 것이다. 출구 우회가 정확히 그
#                  대응책이고 README가 EVERYRIC_AUDIO_SOURCE_ADDRESS를 "403 스로틀 우회"로
#                  문서화한다.
#     network    : 연결 실패는 그 출구 자체가 죽은 경우다 — 다른 출구가 살아 있을 수 있다.
#     bot_check  : 봇 확인은 계정이 아니라 **IP 평판**에 걸리는 게이트다(같은 쿠키·같은 영상이
#                  공인 IP만 바꿔 통과하는 것이 이 게이트의 성질). 인증 계열 중 유일한 예외다.
#   재시도 X
#     unavailable: 삭제·비공개는 영상 탓이다. 어느 출구로 봐도 없다 — 즉시 중단(운영자 요청 ②).
#     login_required / age_restricted
#                : 쿠키·계정이 없어서 막힌 것이다. 출구를 바꿔도 로그인이 생기지 않는다.
#     geo_blocked: 지역 판정은 공인 IP의 위치로 결정되는데, 지금 출구들은 **같은 구내의 다른
#                  회선**이라 위치가 같다(동일 ISP/국가). 바뀔 가망이 없는데 접촉만 배수로
#                  늘어난다. 다른 국가로 나가는 출구를 넣는 배포가 생기면 그때 재검토.
#     js_runtime : 서버에 런타임이 없는 것이다. 출구와 무관하다.
#     unknown    : 분류하지 못한 실패는 재시도하지 않는다. 원인을 모르는 채로 접촉을 배수로
#                  늘리는 쪽이 위험하고, 실패 문구(원문)가 그대로 올라가므로 운영자가 패턴을
#                  추가해 판정에 편입시킬 수 있다. 실제 사고 문구(403)는 이미 throttled다.
_EGRESS_RETRYABLE_CODES = frozenset({"throttled", "network", "bot_check"})

# 목록 설정용 환경변수 (쉼표 구분, 값은 불투명 문자열). settings.py는 이 작업의 소유 범위
# 밖이라 **여기서 직접 읽는다** — 정식 필드(AudioSettings.egress)가 생기면 이 상수들과
# _egress_targets의 env 분기를 지우고 config를 읽으면 된다.
EGRESS_ENV = "EVERYRIC_AUDIO_EGRESS"
# 같은 목록의 옛 이름. 이 세션에 운영자에게 이 이름으로 먼저 알려서 env에 이미 들어갔을 수
# 있어 별칭으로 받는다 (읽기 2줄이 이름 불일치로 폴백이 조용히 죽는 것보다 싸다).
LEGACY_ADDRESSES_ENV = "EVERYRIC_AUDIO_SOURCE_ADDRESSES"


def _apply_egress(ydl_opts: dict[str, Any], egress: str) -> None:
    """**egress 값을 yt-dlp 옵션에 적용하는 유일한 지점** — 전환은 이 함수만 바꾼다.

    오늘: 값은 로컬 바인딩 IP라 ``source_address``로 넣는다.
    나중: 회선별 로컬 프록시로 옮기면 ``ydl_opts["proxy"] = egress`` 한 줄로 끝난다
          (yt-dlp가 ``--proxy``를 그대로 지원한다). 목록 순회·재시도 판정·로깅은 egress를
          불투명 문자열로만 다루므로 그때 손댈 곳이 없다.

    **값의 형태를 검증하지 않는다** — IPv4 정규식 같은 것을 넣으면 프록시 URL이 들어올 때
    깨진다. 주소인지 URL인지는 호스트가 아는 지식이고 앱이 알 필요가 없다(계층 분할).
    """
    ydl_opts["source_address"] = egress


def parse_egress_targets(raw: str | None) -> list[str]:
    """쉼표 구분 egress 목록을 파싱 — 공백 제거, 빈 항목 제거, **순서 유지 중복 제거**.

    값을 해석하지 않는다(불투명 문자열) — IP든 프록시 URL이든 그대로 통과시킨다.
    중복을 지우는 이유: 같은 출구를 두 번 시도하는 것은 회수 0인 유튜브 접촉을 한 번 더
    늘리는 것뿐이다(단일 변수와 목록에 같은 값이 겹치는 배포에서 실제로 생긴다).
    """
    out: list[str] = []
    for part in (raw or "").split(","):
        target = part.strip()
        if target and target not in out:
            out.append(target)
    return out


def _is_geo_restricted(cause: BaseException | None) -> bool:
    """원인 예외 사슬에 yt-dlp의 GeoRestrictedError가 있는지 — 문자열보다 강한 근거다.

    yt_dlp.utils.DownloadError는 원인을 ``exc_info``(sys.exc_info() 튜플)로 보관하므로
    (utils/_utils.py:1058-1069) 그 안까지 따라간다. yt-dlp 미설치·구버전이면 조용히 False.
    """
    try:
        from yt_dlp.utils import GeoRestrictedError
    except Exception:
        return False
    seen: set[int] = set()
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        if isinstance(cause, GeoRestrictedError):
            return True
        info = getattr(cause, "exc_info", None)
        nxt = info[1] if isinstance(info, tuple) and len(info) > 2 else None
        cause = nxt or cause.__cause__
    return False


def classify_download_error(
    error_text: str, cause: BaseException | None = None
) -> DownloadFailure | None:
    """yt-dlp 실패를 사용자 조치 기준으로 분류. **못 가리면 None**(호출부가 원문으로 폴백)."""
    if _is_geo_restricted(cause):
        return _FAILURES["geo_blocked"]
    text = (error_text or "").lower()
    for code, patterns in _FAILURE_PATTERNS:
        if any(p in text for p in patterns):
            return _FAILURES[code]
    return None


def _classified_error(cause: BaseException, url: str, fallback_prefix: str) -> DownloadError:
    """실패를 분류해 사용자 문구를 담은 예외를 만든다 — 분류 실패 시 기존 영문 폴백.

    분류된 실패는 ``str(e)``가 한국어 문구가 되므로 잡 에러가 그대로 사용자에게 쓸 수 있다.
    삭제·비공개는 예전처럼 VideoUnavailableError로 올린다(상위에서 형으로 구분할 수 있게).
    """
    # 우리가 try 안에서 이미 올린 예외는 다시 감싸지 않는다 — 형(VideoUnavailableError 등)과
    # 문구를 보존한다. 감싸면 "Download failed: Failed to download: ..."처럼 접두사가 겹친다.
    if isinstance(cause, DownloadError):
        return cause
    text = str(cause)
    failure = classify_download_error(text, cause)
    if failure is None:
        # 분류 불가 — 기존 동작 유지(영문 원문 노출). 추측 안내보다 원문이 낫다.
        return DownloadError(f"{fallback_prefix}: {cause}")
    exc_cls = VideoUnavailableError if failure.code == "unavailable" else DownloadError
    err = exc_cls(failure.message)
    err.code = failure.code
    err.terminal = failure.terminal
    err.cause_text = text
    logger.info(
        "다운로드 실패 분류: code=%s terminal=%s url=%s (원문: %s)",
        failure.code,
        failure.terminal,
        url,
        text[:300],
    )
    return err


def egress_retryable(error: BaseException) -> bool:
    """이 실패에 **다른 출구로 재시도할 값어치**가 있는가 (_EGRESS_RETRYABLE_CODES 근거)."""
    return getattr(error, "code", "unknown") in _EGRESS_RETRYABLE_CODES


def _probe_decodable(path: Path) -> bool:
    """이 파일을 디코더가 열 수 있는가 — soundfile 먼저, 실패하면 librosa(ffmpeg 경유) 폴백.

    ``loader.AudioLoader.load``가 실제로 쓰는 두 경로와 순서까지 같다(soundfile은 m4a를
    못 읽는다는 게 loader.py의 실측이라 opus/m4a는 librosa 폴백이 정상 경로다 — 거기서
    실패가 아니다). 그래서 둘 다 실패해야 진짜로 못 읽는 파일이고, import 자체가 없는
    환경도 조용히 "못 읽음"으로 처리한다.
    """
    try:
        import soundfile as sf

        sf.info(str(path))
        return True
    except Exception:
        pass
    try:
        import librosa

        librosa.get_duration(path=str(path))
        return True
    except Exception:
        return False


@dataclass
class VideoInfo:
    """YouTube video information."""

    title: str
    duration: float  # seconds
    url: str
    channel: str | None = None
    upload_date: str | None = None


@dataclass
class DownloadResult:
    """Result of audio download."""

    audio_path: Path
    title: str
    duration: float
    url: str


class YouTubeDownloader:
    """Download audio from YouTube using yt-dlp."""

    # YouTube URL patterns
    YOUTUBE_PATTERNS = [
        r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})",
        r"(?:https?://)?(?:www\.)?youtu\.be/([a-zA-Z0-9_-]{11})",
        r"(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})",
        r"(?:https?://)?(?:www\.)?youtube\.com/v/([a-zA-Z0-9_-]{11})",
        r"(?:https?://)?(?:music\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})",
    ]

    def __init__(self, config: AudioSettings | None = None):
        """Initialize downloader.

        Args:
            config: Audio settings. If None, uses global settings.
        """
        self.config = config or get_settings().audio
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """Check if required dependencies are available.

        문구는 한국어다 — 이 예외도 잡 에러(``error=str(e)``)로 사용자 화면까지 올라가고,
        서버 구성 문제라 사용자가 할 수 있는 게 없다는 것을 알려야 한다(분류 (d)와 같은 성격).
        """
        if not shutil.which("ffmpeg"):
            raise DependencyError(
                "서버 설정 문제로 오디오를 처리할 수 없어요 (ffmpeg 미설치). "
                "사용자가 할 수 있는 조치는 없어요 — 서버 관리자에게 알려 주세요."
            )

    def _add_cookie_options(self, ydl_opts: dict[str, Any]) -> None:
        if self.config.cookies_from_browser:
            ydl_opts["cookiesfrombrowser"] = (self.config.cookies_from_browser,)
        elif self.config.cookie_file and self.config.cookie_file.exists():
            ydl_opts["cookiefile"] = str(self.config.cookie_file)
        else:
            from everyric2.config.paths import cookies_read_path

            default_cookie_file = cookies_read_path()
            if default_cookie_file.exists():
                ydl_opts["cookiefile"] = str(default_cookie_file)

    def _add_ejs_options(self, ydl_opts: dict[str, Any]) -> None:
        import os

        deno_path = Path.home() / ".deno" / "bin"
        if deno_path.exists():
            os.environ["PATH"] = f"{deno_path}:{os.environ.get('PATH', '')}"
        ydl_opts["allowed_remote_components"] = ["ejs:github"]

    def _add_network_options(
        self, ydl_opts: dict[str, Any], egress: str | None = None
    ) -> None:
        """다운로드가 나갈 출구(egress)를 옵션에 적용한다 — 적용 자체는 _apply_egress가 한다.

        ``egress``를 주면 그 값으로, 안 주면 설정된 **첫 출구**로 적용한다. 인자를 생략하는
        호출부(server/services/youtube_captions.ydl_opts)의 기존 동작을 그대로 유지하기 위한
        기본값이다 — 그 경로는 순회를 하지 않는다.
        """
        target = egress or next(iter(self._egress_targets()), None)
        if target:
            _apply_egress(ydl_opts, target)

    def _egress_targets(self) -> list[str]:
        """시도할 출구 목록 (빈 목록 = 출구 지정 없이 기본 경로 1회 시도).

        우선순위: ``EVERYRIC_AUDIO_EGRESS``(쉼표 구분 목록) > 옛 이름
        ``EVERYRIC_AUDIO_SOURCE_ADDRESSES`` > ``config.source_address``(기존 단일 변수, 현재
        배포가 쓰는 값). 목록을 준 경우에도 단일 값이 목록에 없으면 **뒤에 붙인다** — 운영자가
        목록만 새로 넣고 기존 변수를 지우지 않았을 때 그 출구를 잃지 않게. 값은 해석하지 않는다
        (불투명 문자열 — EGRESS_ENV/_apply_egress 주석 참고).

        **기본 경로(출구 미지정)를 암묵적 폴백으로 넣지 않는다** — 운영자가 공개 서비스
        트래픽을 전용 대역으로 격리한 것이 이 설정의 목적이므로, 몰래 모뎀 NAT로 되돌아가면
        그 격리가 깨진다. 기본 경로도 쓰고 싶으면 그 출구를 목록에 명시해야 한다.
        """
        import os

        raw = os.environ.get(EGRESS_ENV) or os.environ.get(LEGACY_ADDRESSES_ENV)
        targets = parse_egress_targets(raw)
        single = (self.config.source_address or "").strip()
        if single and single not in targets:
            targets.append(single)
        return targets

    def _with_egress_fallback(self, what: str, url: str, attempt):
        """출구를 **한 바퀴만** 순회하며 ``attempt(egress)``를 시도한다.

        출구 탓일 수 있는 실패(_EGRESS_RETRYABLE_CODES)만 다음 출구로 넘어가고, 영상 탓인
        실패는 즉시 올린다. 성공/실패한 출구를 로그에 남긴다 — 관측이 없으면 폴백이 실제로
        도는지 알 수 없다(운영자 요청 ③). 전부 실패하면 마지막 실패를 올리되 "egress N개 전부
        실패"를 문구에 덧붙여 운영자가 출구 문제임을 알 수 있게 한다(요청 ④).
        """
        targets: list[str | None] = list(self._egress_targets()) or [None]
        last: BaseException | None = None
        for i, target in enumerate(targets):
            try:
                result = attempt(target)
            except DownloadError as e:
                last = e
                remaining = len(targets) - i - 1
                logger.warning(
                    "%s 실패 (egress=%s, code=%s, 남은 egress %d개): %s",
                    what,
                    target or "기본 경로",
                    getattr(e, "code", "unknown"),
                    remaining,
                    getattr(e, "cause_text", None) or str(e),
                )
                if not egress_retryable(e) or remaining == 0:
                    break
                continue
            if i > 0 or target:
                logger.info("%s 성공 (egress=%s, %d번째 시도)", what, target or "기본 경로", i + 1)
            return result
        raise self._exhausted(last, targets, what)

    @staticmethod
    def _exhausted(last: BaseException | None, targets: list, what: str) -> BaseException:
        """마지막 실패에 "N개 전부 실패"를 덧붙인다 — 출구가 실제로 여러 개일 때만.

        출구를 아예 안 쓰는 배포(목록 비어 있음 → [None])에서는 덧붙이지 않는다: 그 문구가
        붙으면 출구 설정이 있는 것처럼 읽혀 운영자를 엉뚱한 곳으로 보낸다.
        덧붙이는 문구는 **사용자 화면에도 뜨는 잡 에러**라 "egress"·"IP" 같은 운영 용어를 쓰지
        않는다("다운로드 경로"). 운영자용 식별은 로그의 ``egress=`` 줄이 담당한다.
        """
        if last is None:  # 방어적 — 루프는 성공 아니면 예외로만 빠져나온다
            return DownloadError(f"{what} 실패 (원인 미기록)")
        configured = [t for t in targets if t]
        if len(configured) < 2 or not egress_retryable(last):
            return last
        note = f" (다운로드 경로 {len(configured)}개 전부 실패 — 회선 쪽 차단으로 보여요)"
        wrapped = type(last)(f"{last}{note}")
        wrapped.code = getattr(last, "code", "unknown")
        wrapped.terminal = getattr(last, "terminal", False)
        wrapped.cause_text = getattr(last, "cause_text", None)
        return wrapped

    def validate_url(self, url: str) -> bool:
        """Check if URL is a valid YouTube URL.

        Args:
            url: URL to validate.

        Returns:
            True if valid YouTube URL.
        """
        return any(re.match(pattern, url) for pattern in self.YOUTUBE_PATTERNS)

    def extract_video_id(self, url: str) -> str | None:
        """Extract video ID from YouTube URL.

        Args:
            url: YouTube URL.

        Returns:
            Video ID or None if not found.
        """
        for pattern in self.YOUTUBE_PATTERNS:
            match = re.match(pattern, url)
            if match:
                return match.group(1)
        return None

    def get_video_info(self, url: str) -> VideoInfo:
        """Get video information without downloading.

        Args:
            url: YouTube URL.

        Returns:
            VideoInfo instance.

        Raises:
            InvalidURLError: If URL is invalid.
            VideoUnavailableError: If video is unavailable.
        """
        if not self.validate_url(url):
            raise InvalidURLError(f"Invalid YouTube URL: {url}")

        # 정보 조회도 403을 맞는다(링크 검증 경로) — 다운로드와 같은 egress 순회를 쓴다
        return self._with_egress_fallback(
            "영상 정보 조회", url, lambda egress: self._extract_info_once(url, egress)
        )

    def _extract_info_once(self, url: str, egress: str | None) -> VideoInfo:
        """정보 조회 1회 시도 (출구 1개) — 실패는 분류된 예외로 올린다."""
        try:
            import yt_dlp

            ydl_opts: dict[str, Any] = {
                "quiet": True,
                "no_warnings": True,
                "extract_flat": False,
            }

            self._add_cookie_options(ydl_opts)
            self._add_ejs_options(ydl_opts)
            self._add_network_options(ydl_opts, egress)

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                if info is None:
                    raise VideoUnavailableError(f"Could not extract info: {url}")

                return VideoInfo(
                    title=info.get("title", "Unknown"),
                    duration=float(info.get("duration", 0)),
                    url=url,
                    channel=info.get("channel"),
                    upload_date=info.get("upload_date"),
                )

        except Exception as e:
            # 정보 조회도 같은 사유로 실패한다(쿠키·지역·삭제·403) — 다운로드와 같은 분류를
            # 쓰지 않으면 링크 검증 경로만 영문 원문을 노출한다
            raise _classified_error(e, url, "Failed to get video info") from e

    def download(
        self,
        url: str,
        output_dir: Path | None = None,
        filename: str | None = None,
    ) -> DownloadResult:
        """Download audio from YouTube URL.

        Args:
            url: YouTube URL.
            output_dir: Output directory. Defaults to temp dir.
            filename: Output filename (without extension). Defaults to video title.

        Returns:
            DownloadResult with path to downloaded audio.

        Raises:
            InvalidURLError: If URL is invalid.
            VideoUnavailableError: If video is unavailable.
            DownloadError: If download fails.
        """
        if not self.validate_url(url):
            raise InvalidURLError(f"Invalid YouTube URL: {url}")

        output_dir = output_dir or self.config.temp_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # 출구(egress)를 한 바퀴 순회한다 — 403/429처럼 출구 탓일 수 있는 실패만 다음 출구로
        # 넘어가고, 삭제·비공개는 즉시 중단한다(_with_egress_fallback).
        return self._with_egress_fallback(
            "오디오 다운로드",
            url,
            lambda egress: self._download_once(url, output_dir, filename, egress),
        )

    def _download_once(
        self,
        url: str,
        output_dir: Path,
        filename: str | None,
        egress: str | None,
    ) -> DownloadResult:
        """다운로드 1회 시도 (출구 1개) — 실패는 분류된 예외로 올린다.

        코덱은 opus 우선 + 스트림카피다(전곡 재인코딩 아님) — 예전엔 전곡을 wav로
        트랜스코드해 곡당 수십 MB가 캐시에 쌓였다(운영자 요청: 전곡 wav 보존 용량 과다).
        "opus 소스는 .opus, aac 소스는 .m4a"로 원본 컨테이너를 그대로 살리고, 디코드
        가능성만 값싸게 확인해 극소수의 안 풀리는 파일만 로컬 트랜스코드로 구제한다
        (_ensure_decodable) — 유튜브를 다시 접촉하지 않는다.
        """
        try:
            import yt_dlp

            # Template for output filename
            if filename:
                outtmpl = str(output_dir / f"{filename}.%(ext)s")
            else:
                outtmpl = str(output_dir / "%(title)s.%(ext)s")

            ydl_opts: dict[str, Any] = {
                # opus 오디오 트랙이 있으면 그것부터 받는다 — 유튜브 업로드 대부분이
                # opus-in-webm이라 재인코딩 없이 그대로 받을 수 있다. 없으면 최선의
                # 오디오로 폴백.
                "format": "bestaudio[acodec=opus]/bestaudio/best",
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        # "best" = 스트림카피 우선(재인코딩 안 함) — 컨테이너만 오디오
                        # 전용으로 바꾼다(yt-dlp FFmpegExtractAudioPP 실측: 이미 공용
                        # 오디오 컨테이너면 후처리 자체를 건너뛴다). preferredquality는
                        # 재인코딩 비트레이트 옵션이라 스트림카피엔 의미가 없어 뺐다.
                        "preferredcodec": "best",
                    }
                ],
                "outtmpl": outtmpl,
                "quiet": True,
                "no_warnings": True,
                "compat_opts": ["allow-unsafe-ext"],
            }

            self._add_cookie_options(ydl_opts)
            self._add_ejs_options(ydl_opts)
            self._add_network_options(ydl_opts, egress)

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                if info is None:
                    raise DownloadError(f"Failed to download: {url}")

                title = info.get("title", "Unknown")
                duration = float(info.get("duration", 0))

                audio_path = self._locate_downloaded_file(info, output_dir, filename, title)
                audio_path = self._ensure_decodable(audio_path)
                logger.info("오디오 확보 형식 유지: %s (url=%s)", audio_path.suffix, url)

                return DownloadResult(
                    audio_path=audio_path,
                    title=title,
                    duration=duration,
                    url=url,
                )

        except yt_dlp.utils.DownloadError as e:
            raise _classified_error(e, url, "Download failed") from e
        except Exception as e:
            raise _classified_error(e, url, "Download failed") from e

    def _locate_downloaded_file(
        self,
        info: dict[str, Any],
        output_dir: Path,
        filename: str | None,
        title: str,
    ) -> Path:
        """postprocessor가 실제로 만든 파일을 찾는다.

        최우선은 ``requested_downloads[0]["filepath"]`` — 최신 yt-dlp는 다운로드용으로
        복사한 info dict를 postprocessor가 그대로 물고 있다가 완료 후 ``filepath``/``ext``를
        갱신하고(FFmpegExtractAudioPP.run), 그 dict가 참조로 ``requested_downloads``에
        실려 있어(YoutubeDL.process_video_result) 여기서 최종 경로가 그대로 보인다. 이
        필드가 없거나 가리키는 파일이 없으면(구버전 yt-dlp) 예상 경로 → 확장자 제한 글롭
        순으로 내려간다. 명시적 filename이 있으면 글롭도 그 접두사로 좁혀 동시 잡의 파일을
        가로채지 않는다(기존 불변식 유지).
        """
        requested = info.get("requested_downloads") or []
        if requested:
            filepath = requested[0].get("filepath")
            if filepath:
                p = Path(filepath)
                if p.exists():
                    return p

        import yt_dlp

        expected_stem = filename or yt_dlp.utils.sanitize_filename(title)

        for ext in _DOWNLOAD_EXTS:
            candidate = output_dir / f"{expected_stem}{ext}"
            if candidate.exists():
                return candidate

        # 예상 경로에 없다 — 확장자별로 글롭한다. filename이 있으면 그 접두사로 좁혀
        # 동시에 도는 다른 잡의 파일을 절대 집지 않는다(기존 불변식).
        glob_prefix = f"{filename}*" if filename else "*"
        for ext in _DOWNLOAD_EXTS:
            matches = list(output_dir.glob(f"{glob_prefix}{ext}"))
            if matches:
                return matches[0]

        raise DownloadError(f"Downloaded file not found: {output_dir / expected_stem}")

    def _ensure_decodable(self, path: Path) -> Path:
        """받은 파일이 디코드되는지 값싸게 확인하고, 안 되면 로컬 트랜스코드로 구제한다.

        스트림카피는 컨테이너만 바꾸므로 아주 드물게 손상된 헤더 등으로 디코더가 못 여는
        조합이 나올 수 있다. ``_probe_decodable``이 쓰는 두 경로(soundfile → librosa)는
        loader.AudioLoader.load가 실제로 쓰는 경로와 같다 — 여기서 통과하면 정렬 단계에서도
        통과한다는 뜻이다. 둘 다 실패하면 **유튜브를 다시 접촉하지 않고** 로컬 ffmpeg로 wav
        트랜스코드한다. 원본은 지운다 — 디코드 안 되는 사본을 캐시에 남기면 다음 캐시
        히트도 똑같이 깨진다.
        """
        if _probe_decodable(path):
            return path

        logger.warning("다운로드 파일이 디코드되지 않아 로컬 wav로 트랜스코드해요: %s", path)
        wav_path = path.with_suffix(".wav")
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", str(path),
                    "-vn", "-acodec", "pcm_s16le", "-ar", "44100",
                    str(wav_path),
                ],
                check=True,
                capture_output=True,
            )
        except (subprocess.CalledProcessError, OSError) as exc:
            raise DownloadError(
                f"Downloaded file is not decodable and local transcode failed: {exc}"
            ) from exc
        finally:
            # 트랜스코드 성패와 무관하게 원본은 지운다 — 실패해도 디코드 안 되는 원본을
            # 캐시/작업 디렉터리에 남겨 둘 이유가 없다.
            path.unlink(missing_ok=True)

        if not wav_path.exists() or wav_path.stat().st_size == 0:
            raise DownloadError(f"Local transcode produced no output: {wav_path}")

        logger.info("로컬 트랜스코드로 구제했어요: %s -> %s", path.name, wav_path.name)
        return wav_path

    def cleanup(self, result: DownloadResult) -> None:
        """Clean up downloaded file.

        Args:
            result: Download result to clean up.
        """
        if result.audio_path.exists():
            result.audio_path.unlink()
