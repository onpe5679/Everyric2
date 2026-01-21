from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from everyric2 import __version__
from everyric2.server.api.cookies import router as cookies_router
from everyric2.server.api.job import router as job_router
from everyric2.server.api.sync import router as sync_router
from everyric2.server.api.translate import router as translate_router
from everyric2.server.db.connection import close_db, init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
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
        gpu_available=torch.cuda.is_available(),
    )


@app.get("/")
async def root():
    return {
        "name": "Everyric2 API",
        "version": __version__,
        "docs": "/docs",
    }
