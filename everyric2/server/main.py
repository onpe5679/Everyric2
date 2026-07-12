from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from everyric2 import __version__
from everyric2.server.api.captions import router as captions_router
from everyric2.server.api.cookies import router as cookies_router
from everyric2.server.api.job import router as job_router
from everyric2.server.api.sync import router as sync_router
from everyric2.server.api.translate import router as translate_router
from everyric2.server.api.vocaro import router as vocaro_router
from everyric2.server.db.connection import close_db, init_db

# torch.cuda.is_available()이 GPU 사용 상태에 따라 호출당 ~2초까지 걸린다(드라이버 질의) —
# /health가 요청마다 부르면 확장 헬스체크 타임아웃(1.5s)을 넘겨 생성 버튼이 잠긴다.
# 런타임에 바뀌는 값이 아니므로 기동 시 1회만 확인해 캐시한다.
_GPU_AVAILABLE: bool | None = None


def _gpu_available() -> bool:
    global _GPU_AVAILABLE
    if _GPU_AVAILABLE is None:
        _GPU_AVAILABLE = torch.cuda.is_available()
    return _GPU_AVAILABLE


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    _gpu_available()  # 기동 시 프리웜 — 첫 /health가 2초 페널티를 물지 않게
    yield
    await close_db()


app = FastAPI(
    title="Everyric2 API",
    description="Lyrics synchronization API using CTC forced alignment",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sync_router)
app.include_router(job_router)
app.include_router(translate_router)
app.include_router(cookies_router)
app.include_router(vocaro_router)
app.include_router(captions_router)


class HealthResponse(BaseModel):
    status: str
    version: str
    gpu_available: bool
    engine: str = "ctc"


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        version=__version__,
        gpu_available=_gpu_available(),
    )


@app.get("/")
async def root():
    return {
        "name": "Everyric2 API",
        "version": __version__,
        "docs": "/docs",
    }
