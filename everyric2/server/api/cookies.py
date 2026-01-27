from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel

from everyric2.config.settings import get_settings

router = APIRouter(prefix="/api/cookies", tags=["cookies"])

COOKIES_FILE_PATH = Path("/tmp/everyric2/youtube_cookies.txt")


class CookiesStatus(BaseModel):
    configured: bool
    method: str | None = None
    path: str | None = None


class CookiesTextRequest(BaseModel):
    content: str


@router.get("", response_model=CookiesStatus)
async def get_cookies_status():
    settings = get_settings().audio

    if settings.cookies_from_browser:
        return CookiesStatus(
            configured=True,
            method="browser",
            path=settings.cookies_from_browser,
        )
    elif settings.cookie_file and settings.cookie_file.exists():
        return CookiesStatus(
            configured=True,
            method="file",
            path=str(settings.cookie_file),
        )
    elif COOKIES_FILE_PATH.exists():
        return CookiesStatus(
            configured=True,
            method="file",
            path=str(COOKIES_FILE_PATH),
        )

    return CookiesStatus(configured=False)


@router.post("/upload")
async def upload_cookies_file(file: UploadFile):
    if not file.filename or not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="File must be a .txt file")

    content = await file.read()

    COOKIES_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    COOKIES_FILE_PATH.write_bytes(content)

    settings = get_settings().audio
    settings.cookie_file = COOKIES_FILE_PATH
    settings.cookies_from_browser = None

    return {"status": "ok", "path": str(COOKIES_FILE_PATH)}


@router.post("/text")
async def set_cookies_text(request: CookiesTextRequest):
    if not request.content.strip():
        raise HTTPException(status_code=400, detail="Content cannot be empty")

    COOKIES_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    COOKIES_FILE_PATH.write_text(request.content)

    settings = get_settings().audio
    settings.cookie_file = COOKIES_FILE_PATH
    settings.cookies_from_browser = None

    return {"status": "ok", "path": str(COOKIES_FILE_PATH)}


@router.delete("")
async def clear_cookies():
    if COOKIES_FILE_PATH.exists():
        COOKIES_FILE_PATH.unlink()

    settings = get_settings().audio
    settings.cookie_file = None
    settings.cookies_from_browser = None

    return {"status": "ok"}
