import logging
import re

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from everyric2.config.settings import get_settings
from everyric2.translation.translator import LyricsTranslator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/translate", tags=["translate"])

# 한글 독음에 가나가 섞이면 LLM 출력 실수 ("카에데노 타네가 오츠 에코오 체ン바" 실측)
_KANA_RE = re.compile(r"[぀-ゟ゠-ヿ]")


def bad_pron_indices(lines) -> list[int]:
    """발음표기에 가나 문자가 섞인 라인 인덱스 (LLM 전사 실수 감지)."""
    return [
        i for i, line in enumerate(lines)
        if line.pronunciation and _KANA_RE.search(line.pronunciation)
    ]


def merge_pron_retry(lines, retry_lines, bad: list[int]) -> int:
    """오염 라인의 발음만 재시도 결과로 교체 (번역은 1차 결과 유지 — 이미 정상).

    재시도도 오염이면 가나 문자만 제거하고, 제거로 내용이 크게 유실되면(>20%)
    발음을 비워 클라이언트가 발음 없이 표시하게 한다. 교체/정리한 라인 수 반환.
    """
    fixed = 0
    for i in bad:
        retry_pron = (
            retry_lines[i].pronunciation
            if retry_lines is not None and i < len(retry_lines)
            else None
        )
        if retry_pron and not _KANA_RE.search(retry_pron):
            lines[i].pronunciation = retry_pron
            fixed += 1
            continue
        original = lines[i].pronunciation or ""
        stripped = " ".join(_KANA_RE.sub("", original).split())
        lines[i].pronunciation = (
            stripped if len(stripped) >= len(original) * 0.8 else None
        )
        fixed += 1
    return fixed


class TranslateRequest(BaseModel):
    text: str = Field(..., description="Text to translate (newline-separated lines)")
    source_lang: str = Field(default="auto", description="Source language code")
    target_lang: str = Field(default="ko", description="Target language code")
    tone: str = Field(default="natural", description="Translation tone")
    include_pronunciation: bool = Field(
        default=False, description="Include pronunciation of the original text"
    )
    # 곡 컨텍스트 (선택) — LLM이 가사 맥락(제목/아티스트)을 알고 번역하게 한다
    title: str | None = Field(default=None, description="Song title for context")
    artist: str | None = Field(default=None, description="Artist name for context")


class TranslationLineResponse(BaseModel):
    original: str
    translation: str
    pronunciation: str | None = None


class TranslateResponse(BaseModel):
    lines: list[TranslationLineResponse]
    source_lang: str
    target_lang: str
    engine: str


@router.post("", response_model=TranslateResponse)
def translate_lyrics(request: TranslateRequest):
    # async def가 아닌 plain def — 내부의 동기 LLM 호출(requests.post, 수십 초)이
    # 이벤트 루프를 세우면 /health까지 밀려 확장이 서버가 죽은 줄 알게 된다.
    # FastAPI는 plain def 엔드포인트를 스레드풀에서 돌린다.

    # 입력 상한 — 대형(122B) 모델 특성상 초장문은 분 단위로 걸려(5000자 실측 90초+)
    # 확장 UI가 죽은 것처럼 보인다. 가사 규모를 한참 넘는 입력은 422로 즉시 거절한다.
    # 주의: 아래 try 안에서 올리면 except Exception이 500으로 삼킨다 — 반드시 밖에서.
    lines_in = request.text.split("\n")
    if len(request.text) > 15000 or len(lines_in) > 400 or any(len(ln) > 1000 for ln in lines_in):
        raise HTTPException(
            status_code=422,
            detail="가사가 너무 길어요 — 최대 400줄, 줄당 1000자, 전체 15000자까지 지원해요.",
        )
    valid_tones = ("literal", "natural", "poetic", "casual", "formal")
    if request.tone not in valid_tones:
        raise HTTPException(
            status_code=422, detail=f"unknown tone '{request.tone}' — one of {valid_tones}"
        )

    try:
        # 전역 싱글턴을 직접 변조하면 스레드풀 동시 요청끼리 tone/발음 플래그가 새어
        # 나간다 — 요청마다 깊은 복사본을 만들어 격리한다
        settings = get_settings().translation.model_copy(deep=True)
        object.__setattr__(settings, "tone", request.tone)
        settings.include_pronunciation = request.include_pronunciation

        translator = LyricsTranslator(settings=settings)

        context = None
        if request.title:
            context = f'"{request.title.strip()}"'
            if request.artist:
                context += f" by {request.artist.strip()}"

        if request.include_pronunciation:
            result = translator.translate_with_pronunciation(
                request.text,
                source_lang=request.source_lang,
                target_lang=request.target_lang,
                context=context,
            )
            # 독음에 가나가 섞였으면(LLM 실수) 한 번 재시도해 오염 라인만 교체한다
            bad = bad_pron_indices(result.lines)
            if bad:
                logger.warning(
                    f"Kana leaked into {len(bad)} pronunciation lines; retrying once"
                )
                retry_lines = None
                try:
                    retry = translator.translate_with_pronunciation(
                        request.text,
                        source_lang=request.source_lang,
                        target_lang=request.target_lang,
                        context=context,
                    )
                    retry_lines = retry.lines
                except Exception:
                    logger.exception("Pronunciation retry failed; sanitizing in place")
                merge_pron_retry(result.lines, retry_lines, bad)
        else:
            result = translator._translator.translate(
                request.text,
                source_lang=request.source_lang,
                target_lang=request.target_lang,
                context=context,
            )

        return TranslateResponse(
            lines=[
                TranslationLineResponse(
                    original=line.original,
                    translation=line.translation,
                    pronunciation=line.pronunciation,
                )
                for line in result.lines
            ],
            source_lang=result.source_lang,
            target_lang=result.target_lang,
            engine=result.engine,
        )
    except Exception:
        # 내부 예외 문자열엔 API URL(키 포함 가능)·경로가 실릴 수 있다 — 클라엔 일반
        # 메시지만, 상세는 서버 로그로
        logger.exception("Translation request failed")
        raise HTTPException(status_code=500, detail="translation failed (see server log)")
