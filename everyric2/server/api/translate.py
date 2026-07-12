
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from everyric2.config.settings import get_settings
from everyric2.translation.translator import LyricsTranslator

router = APIRouter(prefix="/api/translate", tags=["translate"])


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
    try:
        settings = get_settings().translation
        valid_tones = ("literal", "natural", "poetic", "casual", "formal")
        if request.tone in valid_tones:
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
