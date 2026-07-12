from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel

from everyric2.config.paths import LEGACY_COOKIES_PATH, cookies_read_path, cookies_write_path
from everyric2.config.settings import get_settings

router = APIRouter(prefix="/api/cookies", tags=["cookies"])


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
    elif cookies_read_path().exists():
        return CookiesStatus(
            configured=True,
            method="file",
            path=str(cookies_read_path()),
        )

    return CookiesStatus(configured=False)


@router.post("/upload")
async def upload_cookies_file(file: UploadFile):
    if not file.filename or not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="File must be a .txt file")

    content = await file.read()

    target = cookies_write_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)

    settings = get_settings().audio
    settings.cookie_file = target
    settings.cookies_from_browser = None

    return {"status": "ok", "path": str(target)}


@router.post("/text")
async def set_cookies_text(request: CookiesTextRequest):
    if not request.content.strip():
        raise HTTPException(status_code=400, detail="Content cannot be empty")

    target = cookies_write_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(request.content)

    settings = get_settings().audio
    settings.cookie_file = target
    settings.cookies_from_browser = None

    return {"status": "ok", "path": str(target)}


@router.delete("")
async def clear_cookies():
    # 새 위치와 레거시(/tmp → Windows에선 C:\tmp) 둘 다 지워야 완전히 초기화된다
    for path in (cookies_write_path(), LEGACY_COOKIES_PATH):
        path.unlink(missing_ok=True)

    settings = get_settings().audio
    settings.cookie_file = None
    settings.cookies_from_browser = None

    return {"status": "ok"}
