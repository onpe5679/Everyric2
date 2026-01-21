from typing import Literal, cast

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
        default=False, description="Include romanized pronunciation"
    )


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
async def translate_lyrics(request: TranslateRequest):
    try:
        settings = get_settings().translation
        valid_tones = ("literal", "natural", "poetic", "casual", "formal")
        if request.tone in valid_tones:
            object.__setattr__(settings, "tone", request.tone)
        settings.include_pronunciation = request.include_pronunciation

        translator = LyricsTranslator(settings=settings)

        if request.include_pronunciation:
            result = translator.translate_with_pronunciation(
                request.text,
                source_lang=request.source_lang,
                target_lang=request.target_lang,
            )
        else:
            result = translator._translator.translate(
                request.text,
                source_lang=request.source_lang,
                target_lang=request.target_lang,
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
