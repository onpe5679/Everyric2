from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from everyric2 import __version__
from everyric2.server.api.captions import router as captions_router
from everyric2.server.api.cookies import router as cookies_router
from everyric2.server.api.job import router as job_router
from everyric2.server.api.link_jobs import router as link_jobs_router
from everyric2.server.api.sync import router as sync_router
from everyric2.server.api.translate import router as translate_router
from everyric2.server.api.vocaro import router as vocaro_router
from everyric2.server.api.worker import router as worker_router
from everyric2.server.db.connection import close_db, init_db

# torch.cuda.is_available()이 GPU 사용 상태에 따라 호출당 ~2초까지 걸린다(드라이버 질의) —
# /health가 요청마다 부르면 확장 헬스체크 타임아웃(1.5s)을 넘겨 생성 버튼이 잠긴다.
# 런타임에 바뀌는 값이 아니므로 기동 시 1회만 확인해 캐시한다.
# torch 임포트는 이 함수 안에서만 한다 — API 전용 모드(local_worker=false)는 처리를
# 원격 워커 풀이 맡아 이 프로세스에 torch가 필요 없고, 최상위 임포트는 상주 RAM만
# 수 GB 늘린다 (동거 호스트 RAM 예산 조건).
_GPU_AVAILABLE: bool | None = None


def _gpu_available() -> bool:
    global _GPU_AVAILABLE
    if _GPU_AVAILABLE is None:
        from everyric2.config.settings import get_settings

        if not get_settings().server.local_worker:
            # 생성은 원격 워커 풀이 수행 — 이 서버의 GPU 유무와 무관하게 가용
            _GPU_AVAILABLE = True
        else:
            import torch

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

# 확장의 모든 API 호출은 background service worker(chrome-extension:// 오리진)에서
# 나간다 — 일반 웹사이트 오리진은 반영하지 않아, 방문한 악성 페이지가 브라우저를
# 통해 파괴적 엔드포인트(DELETE/PUT)를 호출·응답 열람하는 것을 막는다.
# (curl·서버 간 호출은 Origin이 없어 CORS와 무관하게 동작)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"chrome-extension://.*",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def require_api_key(request, call_next):
    """EVERYRIC_SERVER_API_KEY가 설정된 배포에서만 /api 전체에 키를 요구한다.

    미설정(로컬 단일 사용자 기본)이면 통과. 어드민 키도 유효한 키로 인정.
    /health는 상태 점검용이라 항상 열어 둔다.
    """
    from fastapi.responses import JSONResponse

    from everyric2.config.settings import get_settings

    server = get_settings().server
    # OPTIONS(CORS 프리플라이트)는 브라우저가 커스텀 헤더 없이 보낸다 — 키 검사 제외
    if server.api_key and request.method != "OPTIONS" and request.url.path.startswith("/api"):
        # 원격 워커 엔드포인트는 클라이언트 API 키를 모른다 — 유효한 X-Worker-Key가 있으면
        # X-API-Key 검사를 면제한다 (워커 라우터가 X-Worker-Key를 다시 검증한다)
        worker_key = server.worker_key
        worker_authed = (
            request.url.path.startswith("/api/worker")
            and bool(worker_key)
            and request.headers.get("x-worker-key") == worker_key
        )
        if not worker_authed:
            provided = request.headers.get("x-api-key")
            if provided not in (server.api_key, server.admin_api_key or None):
                return JSONResponse(
                    status_code=401,
                    content={"detail": "API 키가 필요해요 (확장 설정의 API 키 칸에 입력)"},
                )
    return await call_next(request)

app.include_router(sync_router)
app.include_router(job_router)
app.include_router(link_jobs_router)
app.include_router(translate_router)
app.include_router(cookies_router)
app.include_router(vocaro_router)
app.include_router(captions_router)
app.include_router(worker_router)


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
