"""유튜브 자막 조달 서비스 — 라우터와 무관하게 재사용되는 IO/변환 로직.

워치 페이지에서 긁은 timedtext URL은 POT(proof-of-origin) 강제로 브라우저 플레이어
밖에서는 200 + 빈 본문을 돌려준다 (2026-07 실측). yt-dlp는 클라이언트 선택/POT을
내부에서 처리하고 계속 업데이트되므로, 자막은 **반드시 서버가 yt-dlp로 받아야 한다**.
확장이 timedtext를 직접 치는 경로로 대체할 수 없다.

소비자는 둘이다:
  ① `/api/captions` GET 라우트 — 사용자가 트랙을 고르던 옛 경로 (폐기 예정)
  ② `POST /api/sync/generate-from-caption` — video_id만으로 원어 트랙을 자동 판정해
     싱크 생성 잡을 만드는 현 경로

여기 함수는 전부 블로킹이다(yt-dlp). async 라우트에서 부를 때는 threadpool로 넘긴다.
"""

import json
import logging
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 경로 파라미터를 URL·파일 glob에 그대로 쓰므로 형식을 강제한다 (쿼리 인젝션 차단)
VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
LANG_RE = re.compile(r"^[A-Za-z0-9-]{1,16}$")

# 자동 생성 자막은 번역 대상 언어 ~150개가 전부 나열된다 — 원어 트랙('-orig')과
# 사용자가 실제로 쓸 번역만 노출한다 (공개 트랙 목록 전용)
AUTO_LANG_ALLOW = {"ja", "ko", "en"}

# CTC 엔진이 전용 모델/어댑터를 가진 언어 (alignment.ctc_engine.LANG_MODEL_MAP과 동일).
# 자막에서 판정한 언어가 여기 없으면 job.language를 비워 엔진의 텍스트 기반 자동 판정에
# 맡긴다 — 'gl'/'fil' 같은 코드를 그대로 넘기면 엉뚱한 어댑터가 잡힌다.
ALIGNABLE_LANGS = {"ja", "ko", "zh", "en"}

# 정리 후 이 줄 수를 못 넘기면 가사로 쓰지 않는다. 자막이 «[음악]» 같은 효과음 표기나
# 안내 문구뿐인 영상에서 GPU를 태우고 쓰레기 싱크를 남기는 것을 막는 하한선이다.
MIN_LYRIC_LINES = 3


# 실패 코드 → HTTP 상태. 5xx는 조달 자체가 실패한 일시적 사건(재시도 가치 있음),
# 4xx는 "이 영상은 자막으로 안 된다"는 확정 판정이다 — 클라이언트는 4xx를 받으면
# 가사 직접 붙여넣기로 안내한다.
HTTP_STATUS_BY_CODE = {
    "listing_failed": 502,
    "download_failed": 502,
    "no_captions": 404,
    "no_manual_captions": 404,
    "no_original_track": 404,
    "language_undetermined": 404,
    "empty_caption": 404,
    "too_short": 404,
    "non_cjk_caption": 404,
}


class CaptionUnavailable(Exception):
    """자막으로 가사를 만들 수 없을 때 — code로 실패 사유를 구분한다.

    code: listing_failed | no_captions | no_manual_captions | no_original_track |
          language_undetermined | download_failed | empty_caption | too_short |
          non_cjk_caption
    """

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message

    @property
    def http_status(self) -> int:
        return HTTP_STATUS_BY_CODE.get(self.code, 502)

    @property
    def terminal(self) -> bool:
        """재시도해도 결과가 같은가 — True면 사용자에게 다른 경로를 안내해야 한다."""
        return self.http_status < 500


@dataclass(frozen=True)
class TrackChoice:
    """자동 판정으로 고른 트랙. lang은 yt-dlp 트랙 키 그대로다 (예: 'ja', 'ja-orig')."""

    lang: str
    auto: bool
    label: str
    # 판정 근거 (asr_orig | asr_only | video_language | sole_manual) — 응답·로그에 실어
    # 어떤 신호로 이 언어를 골랐는지 사람이 확인할 수 있게 한다
    reason: str
    # 트랙 키에서 벗겨낸 순수 언어 코드 ('ja-orig' → 'ja')
    language: str


@dataclass(frozen=True)
class CaptionLyrics:
    """자막에서 뽑아낸 가사 후보."""

    track: TrackChoice
    lines: list[str]
    # 같은 영상의 한국어 수동 자막에서 가져온 번역. `lines`와 같은 길이이고, 붙일 번역이
    # 없는 자리는 빈 문자열이다. 없으면 None — 그때는 기존대로 서버가 기계 번역을 만든다.
    translations: list[str] | None = None

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    @property
    def align_language(self) -> str | None:
        """CTC 엔진에 넘길 언어 — 지원 목록 밖이면 None(엔진 자동 판정)."""
        return self.track.language if self.track.language in ALIGNABLE_LANGS else None


# ── yt-dlp 조달 ───────────────────────────────────────────────────


def ydl_opts(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """다운로더와 동일한 쿠키/EJS/회선 옵션을 공유하는 yt-dlp 옵션."""
    from everyric2.audio.downloader import YouTubeDownloader

    opts: dict[str, Any] = {"quiet": True, "no_warnings": True, "skip_download": True}
    dl = YouTubeDownloader()
    dl._add_cookie_options(opts)
    dl._add_ejs_options(opts)
    dl._add_network_options(opts)
    if extra:
        opts.update(extra)
    return opts


def extract_caption_info(video_id: str) -> dict[str, Any]:
    """영상 메타(자막 트랙 목록 포함)를 yt-dlp로 가져온다."""
    import yt_dlp

    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        with yt_dlp.YoutubeDL(ydl_opts()) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as e:
        logger.exception(f"caption listing failed for {video_id}")
        raise CaptionUnavailable("listing_failed", "자막 목록을 가져오지 못했어요") from e
    if not info:
        raise CaptionUnavailable("listing_failed", "자막 목록을 가져오지 못했어요")
    return info


def download_track_events(video_id: str, lang: str, auto: bool) -> list[dict[str, Any]]:
    """선택한 트랙을 json3로 받아 [{start, end, text}]로 파싱한다 — **정리 전 원본**.

    가사 정리(`_clean_line`)를 거치지 않은 이벤트 그대로다. 정렬 앵커 경로가 이것을 쓴다:
    앵커는 «우리 가사와 매칭되는 이벤트»만 채택하므로 크레딧·효과음·화자 표기는 매칭에서
    자연히 탈락하고, 그 판정을 정리 규칙에 앞세울 이유가 없다(정리 규칙이 텍스트를 바꾸면
    매칭 키가 흔들린다). 표시·가사용 경로는 `download_track_lines`를 그대로 쓴다.
    """
    import yt_dlp

    url = f"https://www.youtube.com/watch?v={video_id}"
    tmp = Path(tempfile.mkdtemp(prefix="eycap-"))
    try:
        opts = ydl_opts(
            {
                "writesubtitles": not auto,
                "writeautomaticsub": auto,
                "subtitleslangs": [lang],
                "subtitlesformat": "json3",
                "outtmpl": str(tmp / "cap.%(ext)s"),
            }
        )
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.extract_info(url, download=True)
        except Exception as e:
            logger.exception(f"caption download failed for {video_id}/{lang}")
            raise CaptionUnavailable("download_failed", "자막을 내려받지 못했어요") from e

        files = sorted(tmp.glob("*.json3"))
        if not files:
            raise CaptionUnavailable("empty_caption", f"이 영상에 {lang} 자막이 없어요")
        data = json.loads(files[0].read_text(encoding="utf-8"))
        return json3_events_to_lines(data)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def download_track_lines(video_id: str, lang: str, auto: bool) -> list[dict[str, Any]]:
    """선택한 트랙을 json3로 받아 정리까지 마친 [{start, end, text}]로 돌려준다."""
    lines = download_track_events(video_id, lang, auto)
    # 표시용 경로도 가사 생성 경로와 **같은 규칙**으로 정리한다 — 크레딧·효과음 표기가
    # 확장 화면에는 가사인 양 보이는데 생성된 싱크에는 없으면 그 자체로 혼란이다.
    # 타이밍은 표시에 필요하므로 줄을 버리거나 텍스트만 바꾸고 start/end는 보존한다.
    cleaned: list[dict[str, Any]] = []
    for line in lines:
        text = _clean_line(line.get("text", ""))
        if text:
            cleaned.append({**line, "text": text})
    if not cleaned:
        raise CaptionUnavailable("empty_caption", "자막 트랙이 비어 있어요")
    return cleaned


# ── json3 → 라인 ──────────────────────────────────────────────────


def json3_events_to_lines(data: dict[str, Any]) -> list[dict[str, Any]]:
    """timedtext json3 → [{start, end, text}] — 빈 줄/음표만 있는 줄 제거,
    연속 중복(롤링 자막)은 첫 줄의 end를 늘려 병합한다."""
    lines: list[dict[str, Any]] = []
    for ev in data.get("events") or []:
        segs = ev.get("segs")
        if not segs:
            continue
        text = " ".join("".join(s.get("utf8", "") for s in segs).split())
        if not text or all(ch in "♪♫♬ " for ch in text):
            continue
        start = float(ev.get("tStartMs", 0)) / 1000.0
        end = start + float(ev.get("dDurationMs", 0)) / 1000.0
        if lines and lines[-1]["text"] == text:
            lines[-1]["end"] = max(lines[-1]["end"], round(end, 3))
            continue
        lines.append({"start": round(start, 3), "end": round(end, 3), "text": text})
    return lines


def track_label(entries: list[dict[str, Any]] | None, lang: str, auto: bool) -> str:
    name = next((e.get("name") for e in (entries or []) if e.get("name")), None) or lang
    return f"{name} (자동 생성)" if auto else str(name)


def list_public_tracks(info: dict[str, Any]) -> list[dict[str, Any]]:
    """확장에 노출하던 트랙 목록 (업로더 자막 전체 + 자동 생성은 원어/ja·ko·en만).

    **폐기 예정** — 사용자가 트랙을 고르는 UI가 사라지면 함께 지운다.
    """
    tracks: list[dict[str, Any]] = []
    for lang, entries in (info.get("subtitles") or {}).items():
        if lang == "live_chat":
            continue  # 라이브 채팅 리플레이 — 자막이 아니다
        tracks.append({"lang": lang, "label": track_label(entries, lang, False), "auto": False})
    for lang, entries in (info.get("automatic_captions") or {}).items():
        if not (lang.endswith("-orig") or lang in AUTO_LANG_ALLOW):
            continue
        tracks.append({"lang": lang, "label": track_label(entries, lang, True), "auto": True})
    # 업로더 자막 우선, 그 다음 자동 원어 — 지나치게 길어지지 않게 상한
    tracks.sort(key=lambda t: (t["auto"], not t["lang"].endswith("-orig")))
    return tracks[:12]


# ── 원어 판정 ─────────────────────────────────────────────────────


def base_lang(lang: str) -> str:
    """트랙 키에서 순수 언어 코드만 남긴다 — 'ja-orig'/'ja-JP'/'zh-Hans' → 'ja'/'ja'/'zh'."""
    return (lang or "").split("-", 1)[0].lower()


def _usable_keys(mapping: dict[str, Any] | None) -> list[str]:
    return [k for k in (mapping or {}) if k != "live_chat"]


def body_language(counts: dict[str, int]) -> str | None:
    """가사 **본문의 문자 구성**에서 언어를 정한다. CJK 문자가 없으면 None.

    `script_lang_hint`와 왜 다른 함수인가: 그쪽은 «틀려도 트랙 순서만 바뀐다»는 전제로 한
    글자만 있어도 판정한다. 이 함수가 내는 값은 **저장되는 language**이고 그것이 정렬
    어댑터와 한글 독음 표기 여부를 가르므로, 한 글자로 뒤집히면 안 된다.

    규칙은 저장된 가사 283건을 실측해 정했다 (2026-07-26):

        language=ja 196건 — kana/CJK 비율 최소 0.500 · 중앙 0.755 · hangul 비율 최대 0.088
        language=ko  83건 — hangul 비율 중앙 1.000
        language=zh   1건 — han 1.000, kana 0

    그래서 **가나와 한글을 맞대 놓고 많은 쪽**을 고르고, 둘 다 없을 때만 한자를 보고 zh로
    간다. 한자는 ja/ko 판정에 쓰지 않는다 — 일본어 가사의 한자 비율은 정상적으로 최대
    0.500까지 올라가므로 «한자가 많으면 중국어»는 일본어 곡을 중국어로 오판한다.

    「가나가 한 글자라도 있으면 ja」(= `script_lang_hint`의 규칙)를 그대로 쓰면 실측에서
    한 건이 깨진다: `2zilNT7hgFc`는 hangul 485자(98%)에 kana가 8자 섞인 한국 곡인데 ja로
    판정된다. 비교로 바꾸면 맞는다.

    **크레딧 오염 방어(2026-08-03 Atvsg_zogxo 실측)**: 독일어 번역 자막 69줄 중 크레딧
    줄(`ACAね`처럼 콜론 없이 이름만 적혀 `_is_credit_line`을 통과한 줄)의 가나·한자
    몇 글자만으로 `kana >= hangul`이 성립해 통째로 독일어인 본문이 ja로 오판됐다. 그래서
    **CJK 문자 전체(가나+한글+한자)가 본문에서 차지하는 비중**이 `_MIN_CJK_PROPORTION`
    미만이면 어느 분기든 None으로 돌린다 — 절대 개수(`kana=1, han=100`처럼 라틴 오염이
    전혀 없는 순수 CJK 본문)가 아니라 **라틴에 파묻힌 정도**를 본다. zh 분기(한자만 있고
    가나·한글이 전혀 없는 경우)도 2026-08-04 감사(F2-b)로 이 게이트를 함께 받는다 —
    한자 몇 글자 섞인 크레딧 줄이 비CJK 본문에서 zh로 새는 경로가 실측됐다(독일어 340자
    + 한자 4자류). 순정 zh 본문은 이 게이트를 걸어도 안전하다(비중 1.0 실측, 병음 병기가
    섞여도 0.231로 문턱을 크게 웃돈다).
    """
    kana, hangul, han = counts.get("kana", 0), counts.get("hangul", 0), counts.get("han", 0)
    if not (kana or hangul or han):
        return None
    total = counts.get("total") or (kana + hangul + han + counts.get("latin", 0))
    if total and (kana + hangul + han) / total < _MIN_CJK_PROPORTION:
        return None
    if kana or hangul:
        return "ja" if kana >= hangul else "ko"
    return "zh"


# body_language의 ja/ko 판정 최소 CJK 비중 — Atvsg_zogxo(독일어 69줄 + 크레딧 줄의 가나
# 2~3자) 실측에서 CJK 비중은 1% 미만이었다. 진짜 ja/ko 가사는 로마자 음차·절반 영어
# K-pop이 섞여도 이보다 훨씬 높다(로마자 병기 자막 실측 비중 27%대) — 5%는 그 사이에서
# 여유를 크게 두면서도 크레딧 오염(1% 미만)은 확실히 걸러낸다.
_MIN_CJK_PROPORTION = 0.05


def title_script_hint(info: dict[str, Any]) -> str | None:
    """제목·채널명의 문자 체계에서 얻는 원어 힌트. 판정할 수 없으면 None.

    **이것이 유튜브 신호를 대신하는 근거다.** 원래 원어 판정은 `-orig` ASR 트랙과
    `info['language']`에 의존했는데, 자동 더빙·다중 오디오 업로드가 퍼지면서 일본어 곡에
    `vi-orig`·`th-orig`가 붙는 일이 생겨 전제가 깨졌다(실측). 제목과 채널명은 업로더가 쓴
    것이고 오디오 트랙 구성과 무관하므로 그 오염을 받지 않는다.

    한계도 분명하다 — 제목이 로마자뿐인 곡(영어 제목을 붙인 보카로 곡 등)에는 힌트가
    없다. 그때는 None을 돌려주고, 호출부가 «본문이 CJK이기만 하면 받는다»로 느슨하게
    판정한다. 실측(폐기 44건 전수 조사)에서 제목 문자가 ja/ko인 곡이 38건, 로마자인 곡이
    6건이었다.

    **한자만(가나·한글 0) 있는 제목은 None(2026-08-04 감사 F2-a)**: `body_language`는
    이 경우 무조건 zh를 낸다(한자는 ja·zh 공용 문자라 제목만으로는 못 가른다). 실측
    (廻廻奇譚): 제목이 한자 4자(가나 없음)인 실제 ja 곡이 zh로 오판돼, `order_manual_
    tracks`의 하드 리젝트(원어 트랙 없음 사유)가 실재하는 ja 수동 트랙을 "원어(zh) 자막
    없음"이라는 틀린 사유로 포기시켰다. 여기서 None을 돌리면 `asr_lang_hint`(-orig 트랙
    언어 코드) 폴백이 자리를 얻는다 — 순정 zh 곡은 대개 zh-orig가 있어 그쪽에서 정답을
    받고, ja 곡은 ja-orig에서 정답을 받는다. 자막 **본문**(`body_language`) 판정의 zh
    분기는 그대로다 — 본문은 실제로 그 트랙을 받아 본 뒤라 한자만으로 zh 단정해도
    `verify_track_body`의 대칭 예외(아래)가 있어 ja 곡을 오탈락시키지 않는다.
    """
    from everyric2.alignment.caption_anchors import script_counts

    text = f"{info.get('title') or ''} {info.get('uploader') or ''}"
    counts = script_counts(text)
    if counts.get("han") and not counts.get("kana") and not counts.get("hangul"):
        return None
    return body_language(counts)


def asr_lang_hint(info: dict[str, Any]) -> str | None:
    """자동 생성(ASR) '-orig' 트랙의 언어 코드 — 오디오 언어의 메타데이터 신호로만 쓴다.

    ASR **본문**은 정책상 가사로 쓰지 않는다(`manual_track_keys`가 자동 생성 트랙을 아예
    후보에서 뺀다) — 하지만 그 트랙의 **언어 코드**는 여전히 메타데이터다. `title_script_
    hint`가 None일 때(제목·채널명이 로마자뿐인 곡)만 `order_manual_tracks`의 폴백 힌트로
    쓴다.

    실측(2026-08-03 c_9UTrrqcLI): 라틴 제목 곡("[MV] TAK - 'Lucky Doki' feat. xei", 실제
    원어는 한국어)은 제목에 원어 힌트가 전혀 없어 알파벳순 폴백으로 떨어졌다 — en 트랙이
    먼저 걸려 탈락한 뒤 ja(번역 자막)가 힌트 없이 통과해 원어(ko) 트랙은 받아 보지도
    못했다. `-orig` 트랙은 유튜브가 원본으로 인식한 오디오에 대해서만 만들어지므로, 제목이
    아무 신호도 못 주는 곡에서는 알파벳순보다 나은 순서 힌트다.

    자동 더빙·다중 오디오 업로드에서 이 신호가 실제 원어와 다를 수 있다는 것은 이미
    알려져 있다(`zyRt-nBM3dY`, `caption_require_cjk` 설정 문서 참고) — 그래서 이 힌트는
    `order_manual_tracks`에서 `title_script_hint`보다 항상 밀리고(제목이 있으면 절대 안
    쓰인다), `fetch_lyrics_from_captions`의 "힌트 언어에 수동 트랙이 없으면 포기" 규칙과
    본문 CJK 게이트(`verify_track_body`/`has_cjk_script`)가 여전히 최종 방어선으로
    남는다 — 제목도 asr도 신호가 없거나 본문이 어긋나면 그 방어선 하나로 버틴다(예전과
    동일).
    """
    for key in info.get("automatic_captions") or {}:
        if key.endswith("-orig"):
            return base_lang(key)
    return None


def order_manual_tracks(
    info: dict[str, Any], lang_hint: str | None = None, limit: int = 4
) -> list[str]:
    """수동 자막 트랙을 «원어일 가능성이 높은 순»으로 정렬해 상한까지 자른다.

    두 소비자가 같은 순서를 쓴다 — 자막 가사 조달(`fetch_lyrics_from_captions`)과 정렬
    앵커(`iter_manual_caption_events`). 순서 규칙이 갈리면 한쪽에서 옳은 트랙이 다른 쪽에서
    밀린다.

    우선순위:
      ① lang_hint — 이미 가진 가사의 문자 체계에서 얻은 언어. 앵커 경로에만 있는 신호이고
         (가사가 있어야 나온다) 자막 목록 구성에 흔들리지 않아 가장 강하다.
      ② title_script_hint — 제목·채널명의 문자 체계. 가사가 없는 조달 경로의 주 신호다.
      ③ asr_lang_hint — title_script_hint가 없을 때만 쓰는 약한 폴백(`asr_lang_hint`
         문서 참고). 제목이 로마자뿐인 곡에서 알파벳순보다 나은 순서를 준다.
      ④ 나머지는 키 알파벳순 (재현 가능한 순서)

    ③은 ②보다 약하고 신뢰도가 낮다 — 자동 더빙·다중 오디오 업로드에서 일본어 곡에
    `vi-orig`·`th-orig`가 붙는 것이 실측으로 확인됐다(`zyRt-nBM3dY`). 그 사고의 실제
    영상에는 제목이 있어(일본어 문자) ②가 먼저 이겨 ③은 쓰이지 않는다 — ③이 실제로
    관여하는 것은 제목도 채널명도 원어 힌트를 못 주는 소수 사례뿐이고, 그 잔여 위험은
    `fetch_lyrics_from_captions`의 "힌트에 맞는 수동 트랙이 없으면 포기" 규칙과 본문 CJK
    게이트가 받는다.

    트랙 하나가 yt-dlp 호출 1회이므로 상한을 둔다.
    """
    keys = manual_track_keys(info)
    if not keys:
        return []
    hint = title_script_hint(info) or asr_lang_hint(info)
    priors = [base_lang(lang_hint) if lang_hint else None, hint]

    def rank(key: str) -> tuple[int, str]:
        b = base_lang(key)
        for i, p in enumerate(priors):
            if p and b == p:
                return (i, key)
        return (len(priors), key)

    return sorted(keys, key=rank)[: max(0, limit)]


def verify_track_body(hint: str | None, counts: dict[str, int]) -> str | None:
    """받아 본 트랙 본문이 원어 가사로 쓸 만한가 — 쓸 만하면 그 언어를, 아니면 None.

    두 관문이다:

    ① **CJK 문자가 하나라도 있어야 한다.** 우리가 다루는 곡은 압도적으로 일본어·한국어다.
       라틴 전용 본문은 진짜 비CJK 곡보다 «엉뚱한 언어의 ASR 전사거나 팬 번역»일 확률이
       훨씬 높다(실측: 폐기 44건 중 진짜 비CJK 곡은 2건뿐이었다).
    ② **제목 힌트가 있으면 본문이 그것과 같아야 한다.** 제목이 일본어인데 본문이 한글이면
       그 트랙은 원어가 아니라 한국어 팬 번역이다. 이 관문이 없으면 «수동 자막이 하나뿐이면
       원어»라는 옛 규칙이 남긴 오염이 그대로 통과한다(MoRef 집계: `sole_manual` 근거로
       고른 트랙의 62.5%가 오염이었다).

    힌트가 없으면(제목이 로마자뿐인 곡) ①만 본다 — 느슨하지만 그 경우 반박할 근거가 없다.

    ②에는 예외가 하나 있다. **한자는 일본어와 중국어가 공유하는 문자다.** 본문에 가나도
    한글도 없이 한자만 있으면 `body_language`는 zh를 내지만, 그것만으로 두 언어를 가릴 수는
    없다 — 짧은 자막이나 한자 위주 줄에서 실제로 일어난다(실측: language=ja 196건의 한자
    비율이 최대 0.500까지 올라간다). 그래서 그 경우는 제목 힌트를 따른다. 반대 방향은
    거른다 — 가나가 있는 본문은 중국어 곡의 가사가 아니다(`test_verify_does_not_accept_
    kana_for_a_chinese_title`이 이 방향을 명시적으로 못박는다 — 2026-08-04 감사가 "한자
    예외 대칭화"를 제안했으나 이 기존 계약과 직접 충돌해 채택하지 않았다. 廻廻奇譚류의
    실제 결함은 `title_script_hint`가 한자만 있는 제목에 더는 zh를 단정하지 않는 F2-a
    하나로 이미 해소된다 — asr_lang_hint 폴백이 정답을 준다).
    """
    lang = body_language(counts)
    if lang is None:
        return None
    if hint and lang != hint:
        if lang == "zh" and hint in ("ja", "zh"):
            return hint
        return None
    return lang


# ── 자막 → 가사 텍스트 정리 ───────────────────────────────────────

# «[음악]» «[Applause]» «【拍手】» — 자동 생성 자막의 효과음/상황 표기. 대괄호류는
# 가사 표기에 거의 쓰이지 않아 줄 안에 있어도 통째로 걷어낸다.
_BRACKET_RE = re.compile(r"[\[［【][^\]］】]*[\]］】]")
# 소괄호는 코러스/애드리브 표기로 실제 가사에 흔하다 — 줄 전체가 괄호이고 내용이 알려진
# 효과음일 때만 버린다
_PAREN_LINE_RE = re.compile(r"^[(（]\s*([^)）]*?)\s*[)）]$")
_EFFECT_WORDS = {
    "음악", "박수", "웃음", "환호", "노래", "연주", "무음",
    "music", "applause", "laughter", "laughs", "cheering", "cheers", "singing",
    "instrumental", "sound", "noise", "silence", "no audio",
    "音楽", "拍手", "笑", "笑い", "歓声", "演奏", "音楽が流れる",
}
# 화자 표기: '>> ', '- ', '이름: ' — 이름은 공백 없는 짧은 토큰만 인정해 가사 안의
# 콜론(«3:00» 등)을 잘라내지 않게 한다
_LEAD_MARKER_RE = re.compile(r"^(?:>>+|♪+|-(?=\s))\s*")
_SPEAKER_RE = re.compile(r"^[^\s:：]{1,12}[:：]\s+")
_TRAIL_NOTE_RE = re.compile(r"[♪♫♬\s]+$")

# 업로더 자막의 크레딧 줄 — «Music & Lyrics : 40mP», «Vocal : 初音ミク», «作詞：〇〇».
# 가사가 아닌데 정렬에 들어가면 첫 줄부터 「뮤우짓쿠 안도 레릿쿠스」 같은 독음이 박힌다
# (실측: TXzfQ0cP1P0의 자막 첫 세 줄이 전부 크레딧이었다).
#
# 판정은 **콜론 앞이 전부 역할어일 때만** 참이다. «3:00»이나 «君: 僕»처럼 콜론이 든 진짜
# 가사를 잘라내지 않기 위한 조건이고, 목록에 없는 말은 아예 건드리지 않는다. 역할어는
# 소문자로 비교하므로 목록도 소문자로 둔다(일본어·한국어는 대소문자가 없다).
_CREDIT_ROLES = frozenset({
    # 영어
    "music", "lyric", "lyrics", "composer", "compose", "composed", "composition",
    "arrange", "arranged", "arrangement", "arranger", "vocal", "vocals", "voice",
    "sung", "singer", "chorus", "mix", "mixed", "mixing", "mastering", "mastered",
    "master", "illust", "illustration", "illustrations", "illustrator", "art",
    "artwork", "movie", "video", "mv", "pv", "animation", "anime", "director",
    "direction", "produce", "produced", "producer", "production", "guitar", "bass",
    "drum", "drums", "piano", "keyboard", "strings", "violin", "synth", "translation",
    "translated", "translator", "subtitle", "subtitles", "encode", "encoded",
    "thanks", "staff", "credit", "credits", "tuning", "editing", "editor", "cast",
    # 일본어
    "作詞", "作曲", "編曲", "補作", "詞", "曲", "歌", "唄", "歌唱", "歌手", "ボーカル",
    "ボカロ", "コーラス", "調声", "調整", "ミックス", "ミキシング", "マスタリング",
    "イラスト", "絵", "画", "動画", "映像", "撮影", "監督", "演出", "制作", "製作",
    "企画", "演奏", "ギター", "ベース", "ドラム", "ドラムス", "ピアノ", "キーボード",
    "翻訳", "字幕", "協力", "出演", "原曲", "本家", "音源", "スタッフ",
    # 한국어
    "작사", "작곡", "편곡", "노래", "보컬", "코러스", "믹스", "믹싱", "마스터링",
    "일러스트", "그림", "영상", "편집", "제작", "기획", "연주", "기타", "베이스",
    "드럼", "피아노", "번역", "자막", "원곡", "출연", "스태프",
})
# 역할어를 잇는 구분자 — «Music & Lyrics», «作詞・作曲», «Vocal/Mix»
_CREDIT_SEP_RE = re.compile(r"[\s&/,、・·+＋]+")
# 콜론 앞이 이보다 길면 크레딧 표기로 보지 않는다 (역할어 4개까지 이어 붙는 정도)
_CREDIT_HEAD_MAX = 28
_COLON_RE = re.compile(r"[:：]")
_URL_LINE_RE = re.compile(r"^(?:https?://|www\.)\S+$", re.IGNORECASE)


def _is_credit_line(text: str) -> bool:
    """«역할어 : 이름» 형태의 크레딧 줄인가."""
    m = _COLON_RE.search(text)
    if not m:
        return False
    head, tail = text[: m.start()], text[m.end() :]
    if not tail.strip() or len(head) > _CREDIT_HEAD_MAX:
        return False
    roles = [t for t in _CREDIT_SEP_RE.split(head.strip().lower()) if t]
    return bool(roles) and len(roles) <= 4 and all(r in _CREDIT_ROLES for r in roles)


def _clean_line(raw: str) -> str:
    text = _BRACKET_RE.sub(" ", raw or "")
    text = _LEAD_MARKER_RE.sub("", text.strip())
    # 크레딧·URL 판정은 화자 표기 제거보다 **먼저** 한다 — «作詞: 〇〇»는 _SPEAKER_RE에
    # 걸려 역할어만 떨어져 나가고 이름이 가사 줄로 남는다
    text = " ".join(text.split())
    if _is_credit_line(text) or _URL_LINE_RE.match(text):
        return ""
    text = _SPEAKER_RE.sub("", text)
    text = _TRAIL_NOTE_RE.sub("", text)
    text = " ".join(text.split())
    if not text:
        return ""
    paren = _PAREN_LINE_RE.match(text)
    if paren and paren.group(1).strip().lower() in _EFFECT_WORDS:
        return ""
    return text


def clean_caption_events(
    lines: list[dict[str, Any]] | list[str], merge_rolling: bool = False
) -> list[dict[str, Any]]:
    """자막 라인 → 정리된 `[{start, end, text}]`. 정리 규칙의 **정본**이다.

    걸러내는 것: 효과음/상황 표기(`[음악]`), 화자 표기(`>>`, `이름:`), 장식 음표,
    빈 줄, **연속 중복 줄**. 같은 가사가 후렴으로 다시 나오는 것은 진짜 반복이므로
    떨어져 있는 중복은 건드리지 않는다.

    merge_rolling=True(자동 생성 자막)면 롤링 자막이 만드는 «앞부분 → 앞부분+뒷부분»
    누적 이벤트를 마지막 형태 하나로 합친다. 업로더 자막은 롤링하지 않고, 반대로
    앞 줄이 뒷 줄의 접두사인 진짜 가사(«君は» → «君は僕を»)가 존재하므로 기본은 끈다.

    시각을 남기는 이유: 같은 영상의 한국어 자막을 번역으로 붙일 때 **원문 줄과 번역 줄을
    시간으로 맞춰야** 한다. 줄 순서로 맞출 수 없다 — 여기서 버려지는 줄(효과음·중복)의
    개수가 두 트랙에서 다르므로 인덱스가 어긋난다. 정렬용 가사에는 시각이 필요 없고
    (`clean_caption_lines`가 텍스트만 꺼낸다) 서버가 오디오에서 새로 잡는다.
    """
    out: list[dict[str, Any]] = []
    for item in lines:
        if isinstance(item, dict):
            raw = item.get("text", "")
            start, end = item.get("start"), item.get("end")
        else:
            raw, start, end = str(item), None, None
        text = _clean_line(raw)
        if not text:
            continue
        if out:
            prev = out[-1]
            if text == prev["text"]:
                # 연속 중복은 같은 줄이 이어진 것 — 뒤엣것의 끝까지 한 줄로 본다
                if end is not None and prev.get("end") is not None:
                    prev["end"] = max(prev["end"], end)
                continue
            if merge_rolling and text.startswith(prev["text"]):
                prev["text"] = text
                if end is not None and prev.get("end") is not None:
                    prev["end"] = max(prev["end"], end)
                continue
            if merge_rolling and prev["text"].startswith(text):
                continue
        out.append({"start": start, "end": end, "text": text})
    return out


def clean_caption_lines(
    lines: list[dict[str, Any]] | list[str], merge_rolling: bool = False
) -> list[str]:
    """자막 라인 → 가사 텍스트 줄. 타임스탬프는 버린다 (정렬은 서버가 새로 한다)."""
    return [e["text"] for e in clean_caption_events(lines, merge_rolling)]


# ── 같은 영상의 한국어 자막을 번역으로 ────────────────────────────

# 원문 줄 하나에 번역을 붙이려면 겹침이 이만큼은 되어야 한다. 인접 줄의 꼬리에 스친 것을
# 번역으로 오인하지 않기 위한 하한이고, 둘 중 **하나만** 넘으면 된다 — 짧은 원문 줄
# («ねぇ» 한 마디)은 절대 시간으로는 작고, 긴 원문 줄은 비율로는 작기 때문이다.
_TRANSLATION_MIN_OVERLAP_SEC = 0.5
_TRANSLATION_MIN_OVERLAP_RATIO = 0.3


def match_translation_lines(
    originals: list[dict[str, Any]], translations: list[dict[str, Any]]
) -> list[str]:
    """원문 줄마다 시간이 겹치는 번역 자막을 골라 원문과 같은 길이의 목록으로 돌려준다.

    같은 영상의 자막이라 두 트랙은 **같은 시간축**을 쓴다. 그래서 시간 겹침이 가장 큰
    번역 이벤트를 고르는 것으로 대응이 된다. 줄 순서로 맞출 수 없는 이유는
    `clean_caption_events`에 적어 두었다(버려지는 줄 수가 트랙마다 다르다).

    한 번역 줄이 여러 원문 줄에 걸리는 것은 허용한다 — 번역자가 두 줄을 한 줄로 합쳐 적는
    일이 흔하고, 그때 두 원문 줄에 같은 번역이 붙는 것이 빈칸보다 낫다.

    붙일 번역이 없는 자리는 빈 문자열이다. 시각이 없는 입력(문자열 목록)은 전부 빈칸이 된다
    — 맞출 근거가 없으면 아무것도 붙이지 않는다.
    """
    out: list[str] = []
    for orig in originals:
        os_, oe = orig.get("start"), orig.get("end")
        if os_ is None or oe is None or oe <= os_:
            out.append("")
            continue
        span = oe - os_
        best, best_overlap = "", 0.0
        for tr in translations:
            ts, te = tr.get("start"), tr.get("end")
            if ts is None or te is None:
                continue
            overlap = min(oe, te) - max(os_, ts)
            if overlap <= best_overlap:
                continue
            if overlap < _TRANSLATION_MIN_OVERLAP_SEC and (
                overlap / span < _TRANSLATION_MIN_OVERLAP_RATIO
            ):
                continue
            best, best_overlap = tr.get("text") or "", overlap
        out.append(best)
    return out


def _fetch_translation_lines(
    video_id: str,
    info: dict[str, Any],
    original_lang: str,
    originals: list[dict[str, Any]],
) -> list[str] | None:
    """같은 영상의 한국어 수동 자막을 받아 원문 줄에 맞춰 돌려준다. 없거나 실패하면 None.

    번역은 «있으면 좋은» 것이다 — 여기서 나는 어떤 실패도 가사 조달을 깨선 안 된다. 실패하면
    None을 돌려주고, 기존대로 서버가 기계 번역을 만든다.

    왜 이걸 하는가: 원문을 유튜브 자막에서 가져오는 곡이라면 같은 영상에 한국 팬이 붙인
    자막이 있을 확률이 높고(어젯밤 300곡 실측: 원문이 한국어가 아닌데 한국어 수동 자막이
    있는 곡이 93곡, 31%), 사람이 옮긴 번역이 기계 번역보다 낫다. 시간축도 같으므로 줄
    대응이 자연스럽다.
    """
    key = _translation_track_key(info, original_lang)
    if not key or not LANG_RE.match(key):
        return None
    try:
        raw = download_track_lines(video_id, key, auto=False)
    except Exception:
        logger.info("번역용 한국어 자막을 받지 못했어요 — 기계 번역으로 갑니다 (video %s)", video_id)
        return None
    matched = match_translation_lines(originals, clean_caption_events(raw))
    filled = sum(1 for m in matched if m)
    logger.info(
        f"caption translation for {video_id}: track={key} matched={filled}/{len(matched)}"
    )
    # 한 줄도 못 붙였으면 없는 것과 같다 — 빈 번역을 line_meta에 실어 보내지 않는다
    return matched if filled else None


def _translation_track_key(info: dict[str, Any], original_lang: str) -> str | None:
    """번역으로 쓸 한국어 수동 자막 트랙. 없거나 원문이 한국어면 None.

    왜 한국어만인가: 지금 싱크 스키마에 번역의 언어를 적을 자리가 없어서 표시 쪽이 한국어를
    가정한다. 다른 언어를 넣으면 한국어 자리에 그 언어가 나온다.
    """
    if base_lang(original_lang) == "ko":
        return None
    keys = [k for k in manual_track_keys(info) if base_lang(k) == "ko"]
    return sorted(keys)[0] if keys else None


# ── 원어 판정 오류로 인한 가사 오염 게이트 ────────────────────────

# 이 게이트가 막는 것 (실측 2026-07-26):
#
#   zyRt-nBM3dY (시니컬 나이트 플랜 / 하츠네 미쿠 — 일본어 보카로 곡)
#     video_language = 'vi'      ← 유튜브가 기본 오디오를 베트남어로 보고한다
#     -orig 트랙     = ['vi-orig'] ← ASR도 베트남어
#     detect_original_language → ('vi', 'asr_orig')
#
# `detect_original_language`의 규칙 ①(asr_orig)과 ③(video_language)이 함께 서 있는 전제 —
# 「유튜브는 원어 오디오에 대해서만 ASR을 만든다」 — 가 자동 더빙·다중 오디오 트랙의 확산으로
# 깨졌다. 그 결과 **일본어 오디오에 베트남어 ASR을 돌린 전사가 곡 가사로 저장된다**
# (MoRef 야간 배치 예행에서 th-orig 태국어로도 재현됐다).
#
# 근본 수정은 오디오로 언어를 판정하는 것이고 별도 작업이다. 여기서는 **오염만 막는다.**
# 판정 규칙(`select_original_track`)은 건드리지 않는다.
#
# 왜 앵커 쪽 판정(매칭률로 트랙 선택)을 쓸 수 없나: 그것은 **가사가 이미 있을 때만** 성립한다.
# 이 경로는 가사가 없어서 자막으로 가사를 만드는 경로다 — 맞춰 볼 대상이 없다.
# 문자 구성으로 「판정 언어와 자막 문자가 일치하는가」를 물어도 통과한다: vi-orig 자막은 실제로
# 베트남어 문자를 낸다(ASR이 베트남어로 돌았으니 당연하다). 유튜브가 오디오를 베트남어라고
# 믿는 것을 반박할 텍스트 근거는 없다.
#
# 그래서 반박이 아니라 **적용 범위**로 막는다: 우리가 다루는 곡은 압도적으로 일본어·한국어다.
# 자막에 가나·한글·한자가 **하나도 없으면** 자막 경로를 포기한다.
_CJK_SCRIPTS = ("kana", "hangul", "han")


def caption_script_counts(lines: list[str]) -> dict[str, int]:
    """자막 줄들의 문자 체계 구성. 판정 근거를 로그에 싣기 위해 개수까지 돌려준다."""
    from everyric2.alignment.caption_anchors import script_counts

    return script_counts("".join(lines))


def has_cjk_script(counts: dict[str, int]) -> bool:
    """가나·한글·한자가 하나라도 있는가.

    비율 기준(«CJK가 몇 % 미만»)이 아니라 **하나도 없음**으로 둔 이유: 후자가 더 보수적이라
    진짜 CJK 곡을 잘못 버릴 여지가 없다. 로마자 음차가 섞인 일본어 가사, 영어 후렴이 절반인
    K-pop, 제목만 라틴인 자막이 모두 통과한다.
    """
    return any(counts.get(k, 0) for k in _CJK_SCRIPTS)


def _require_cjk_enabled() -> bool:
    """게이트 스위치 (기본 켜짐). 설정을 못 읽는 구성에서도 켜진 것으로 본다."""
    try:
        from everyric2.config.settings import get_settings

        return bool(get_settings().server.caption_require_cjk)
    except Exception:  # pragma: no cover - 설정 로드 실패는 게이트를 끄는 사유가 아니다
        logger.warning("caption_require_cjk setting unreadable; keeping the gate ON")
        return True


# 본문을 받아 볼 수동 트랙 상한 — 트랙 하나가 yt-dlp 호출 1회다. 설정을 못 읽으면
# 기본값으로 계속한다(조달을 멈출 이유가 아니다).
_PROBE_LIMIT_FALLBACK = 4


def _probe_limit() -> int:
    try:
        from everyric2.config.settings import get_settings

        return max(1, int(get_settings().server.caption_track_probe_limit))
    except Exception:  # pragma: no cover - 설정 로드 실패는 조달을 멈추는 사유가 아니다
        return _PROBE_LIMIT_FALLBACK


# ── 한 번에: video_id → 가사 ──────────────────────────────────────


def fetch_lyrics_from_captions(video_id: str) -> CaptionLyrics:
    """video_id만으로 원어 자막을 찾아 가사 텍스트를 만든다 (블로킹).

    실패는 전부 CaptionUnavailable로 나가며, 호출부가 code를 보고 클라이언트에게
    "직접 붙여넣기"를 안내할 수 있다.
    """
    if not VIDEO_ID_RE.match(video_id or ""):
        raise CaptionUnavailable("listing_failed", "invalid video_id")

    info = extract_caption_info(video_id)
    manual = manual_track_keys(info)
    if not manual:
        # 자동 생성(ASR)만 있는 영상이 여기로 온다. ASR 전사는 가사로 쓰지 않는다 —
        # 사용자 확인(2026-07-26): 표시는 되지만 인식 품질이 가사로 쓸 수준이 아니다.
        # 코드에도 처음부터 «자동 생성은 가사 오인식이 많다»고 적혀 있었고, 언어 라벨까지
        # 못 믿는다는 것이 실측으로 확인됐다(일본어 곡에 vi-orig·th-orig).
        raise CaptionUnavailable(
            "no_manual_captions",
            "이 영상에는 사람이 쓴 자막이 없어요 (자동 생성 자막은 가사로 쓰지 않아요)",
        )

    hint = title_script_hint(info) or asr_lang_hint(info)
    require_cjk = _require_cjk_enabled()

    if hint and len(manual) >= 2 and not any(base_lang(k) == hint for k in manual):
        # 힌트(제목 스크립트든 asr이든)가 가리키는 언어의 수동 트랙이 아예 없다 — 남은
        # 트랙은 전부 번역일 가능성이 높다. 실측(2026-08-03 Atvsg_zogxo): 힌트 ja인데 ja
        # 수동 트랙이 없어(18종 전부 번역) 전원 동순위로 알파벳순 폴백이 de(독일어)를
        # 골랐다. 엉뚱한 언어를 받아 보는 대신 여기서 포기해 다음 소스(가사 직접 붙여넣기
        # 등)로 넘긴다.
        #
        # **len(manual) >= 2 조건이 필요한 이유**(회귀 실측: test_fetch_lyrics_takes_the_
        # language_from_the_body_not_the_track_code): 수동 트랙이 **하나뿐**이면 그 하나가
        # 힌트와 다른 언어 코드를 달고 있어도 "남은 트랙이 전부 번역"이라고 말할 근거가
        # 없다 — 오히려 한국 팬이 일본어 원문을 ko 트랙에 잘못 올리는 것처럼 **그 하나가
        # 원문 자체가 잘못 라벨된 경우**가 실측돼 있다(언어 코드는 틀렸지만 트랙 코드 33건
        # 오염 사례). 트랙이 하나뿐이면 포기하지 않고 기존 본문 검증(`verify_track_body`)
        # 이 그 트랙의 실제 내용으로 판정하게 둔다 — 힌트와 본문이 맞으면 받아들이고
        # (트랙 코드가 아니라 본문에서 유도한 언어로 저장한다), 아니면 그 검증이 알아서
        # 기각한다. 후보가 여럿인데 전부 힌트와 다르면(이 분기) 그 다수가 우연히 전부
        # 잘못 라벨됐을 확률보다 "전부 번역"일 확률이 훨씬 높다.
        logger.warning(
            f"caption lyrics rejected for {video_id}: hint={hint} has no matching manual "
            f"track among {manual} — the rest are translations, not the original"
        )
        raise CaptionUnavailable(
            "no_original_track",
            f"사람이 쓴 자막은 있는데 이 곡의 원어({hint}) 자막 트랙이 없어요 — "
            f"있는 자막은 전부 번역일 가능성이 높아요",
        )

    # (트랙, 사유코드) — 전부 같은 사유로 떨어졌으면 그 사유를 그대로 전한다. 「자막이
    # 효과음뿐이라 줄이 부족하다」를 「원어 트랙이 없다」로 뭉개면 안내가 틀린다.
    rejected: list[tuple[str, str]] = []

    # 후보를 순서대로 **받아 보고 본문으로 판정한다.** 목록만 보고 고를 수 없다는 것이
    # 이 경로의 핵심 교훈이다 — 트랙의 언어 코드는 «그 트랙이 무슨 언어인가»만 말하고
    # «이 곡의 원어가 무엇인가»는 말하지 않는다. 팬 번역 트랙도 자기 언어와 일치한다.
    for key in order_manual_tracks(info, hint, _probe_limit()):
        if not LANG_RE.match(key):
            continue
        try:
            raw_lines = download_track_lines(video_id, key, auto=False)
        except CaptionUnavailable as e:
            rejected.append((key, e.code))
            continue
        # 시각을 남긴 채로 정리한다 — 한국어 자막을 번역으로 붙일 때 필요하다
        events = clean_caption_events(raw_lines, merge_rolling=False)
        lines = [e["text"] for e in events]
        if len(lines) < MIN_LYRIC_LINES:
            rejected.append((key, "too_short"))
            continue
        counts = caption_script_counts(lines)
        lang = verify_track_body(hint, counts)
        # 판정 근거를 항상 남긴다 — 오염 규모를 사후에 세는 유일한 자료다 (MoRef soak 집계용)
        logger.info(
            f"caption lyrics probe {video_id}: track={key} hint={hint} body={lang} "
            f"lines={len(lines)} script={counts}"
        )
        if lang is None and require_cjk:
            rejected.append((key, "body_mismatch" if has_cjk_script(counts) else "non_cjk"))
            continue
        return CaptionLyrics(
            track=TrackChoice(
                lang=key,
                auto=False,
                label=track_label((info.get("subtitles") or {}).get(key), key, False),
                # 어느 근거로 이 트랙을 받았는지 — 사후 집계가 이 문자열로 갈린다
                reason="title_script" if hint else "body_script",
                # **트랙 코드가 아니라 본문에서 유도한 언어를 쓴다.** 이것이 이 변경의
                # 핵심이다: 한국 팬이 일본어 원문을 ko 트랙에 올리는 일이 흔해서, 트랙
                # 코드를 그대로 쓰면 일본어 가사가 language=ko로 저장된다(실측 33건).
                # 그러면 한국어 어댑터로 정렬되고 한글 독음도 붙지 않는다.
                language=lang or base_lang(key),
            ),
            lines=lines,
            translations=_fetch_translation_lines(
                video_id, info, lang or base_lang(key), events
            ),
        )

    logger.warning(
        f"caption lyrics rejected for {video_id}: no usable original track among {manual} "
        f"(hint={hint}, probed={rejected}). A manual track whose body script disagrees with the "
        f"title is a fan translation, not the original; a body with no kana/hangul/han is far more "
        f"likely to be a wrong-language transcript than a genuine non-CJK song "
        f"(measured: only 2 of 44 purged songs were genuinely non-CJK)"
    )
    detail = ", ".join(f"{k}:{c}" for k, c in rejected) or "받아 볼 트랙이 없었어요"
    codes = {c for _, c in rejected}

    # 후보 전부가 같은 사유로 떨어졌고 그 사유가 그대로 전할 만한 것이면 그것을 쓴다.
    # 「자막이 효과음뿐이다」와 「원어 트랙이 없다」는 사용자가 할 일이 다르다.
    if codes == {"too_short"}:
        raise CaptionUnavailable(
            "too_short", "자막에서 쓸 만한 가사 줄이 너무 적어 가사로 쓸 수 없어요"
        )
    if codes == {"empty_caption"} or codes == {"download_failed"}:
        raise CaptionUnavailable(next(iter(codes)), f"자막을 받지 못했어요 ({detail})")

    if hint and codes and codes != {"non_cjk"}:
        raise CaptionUnavailable(
            "no_original_track",
            f"사람이 쓴 자막은 있는데 이 곡의 원어({hint}) 가사인 트랙이 없어요 — "
            f"팬 번역 자막만 있는 경우예요 ({detail})",
        )
    raise CaptionUnavailable(
        "non_cjk_caption",
        f"사람이 쓴 자막에 일본어·한국어 문자가 전혀 없어요 — 유튜브가 원어를 잘못 알려 "
        f"주는 경우라 가사로 쓰면 엉뚱한 내용이 저장돼요 ({detail})",
    )


# ── 정렬 앵커용: 타임스탬프를 살린 수동 자막 트랙 ─────────────────


def manual_track_keys(info: dict[str, Any]) -> list[str]:
    """업로더가 직접 쓴(수동) 자막 트랙 키 목록. 자동 생성(ASR)은 절대 포함하지 않는다.

    ASR 트랙은 롤링 병합으로 이벤트 시각이 «누적 표시» 시점이 되어 발성 시점과 구조적으로
    어긋난다 — 앵커로 쓰면 없는 구조를 만들어낸다. 앵커는 «사람이 찍은 시각»만 쓴다.
    """
    return _usable_keys(info.get("subtitles"))


def iter_manual_caption_events(video_id: str, lang_hint: str | None = None, limit: int = 5):
    """앵커 후보 트랙을 (lang, events) 로 **게으르게** 내놓는다 (블로킹 IO, 트랙당 1회 다운로드).

    제너레이터인 이유: 호출부가 매칭률을 보고 충분히 좋은 트랙에서 즉시 멈출 수 있어야
    한다(첫 후보가 맞는 경우가 대부분이고, 트랙 하나가 yt-dlp 호출 1회다).

    앵커는 «있으면 좋은» 신호라 실패는 전부 삼킨다 — 자막을 못 받는다고 정렬이 죽어선 안 된다.
    """
    if not VIDEO_ID_RE.match(video_id or ""):
        return
    try:
        info = extract_caption_info(video_id)
    except Exception:
        logger.info(f"caption anchors: listing failed for {video_id}; continuing without anchors")
        return
    for lang in order_manual_tracks(info, lang_hint, limit):
        if not LANG_RE.match(lang):
            continue
        try:
            events = download_track_events(video_id, lang, auto=False)
        except Exception:
            logger.info(f"caption anchors: track {lang} unavailable for {video_id}; skipping")
            continue
        if events:
            yield lang, events
