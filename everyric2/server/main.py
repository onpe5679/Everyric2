from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from everyric2 import __version__
from everyric2.server.api.captions import router as captions_router
from everyric2.server.api.cookies import router as cookies_router
from everyric2.server.api.job import router as job_router
from everyric2.server.api.limits import router as limits_router
from everyric2.server.api.link_jobs import router as link_jobs_router
from everyric2.server.api.notices import router as notices_router
from everyric2.server.api.stats import router as stats_router
from everyric2.server.api.sync import router as sync_router
from everyric2.server.api.translate import router as translate_router
from everyric2.server.api.vocaro import router as vocaro_router
from everyric2.server.api.worker import router as worker_router
from everyric2.server.db.connection import close_db, init_db
from everyric2.server.privacy_page import PRIVACY_HTML

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
    # 만료 리스 주기 스윕을 여기서만 띄운다 — 임포트 시점에 뜨면 실행 중인 루프가 없고,
    # 앱을 띄우지 않는 테스트에 태스크가 남는다 (api/worker.start_lease_sweeper 주석 참고).
    from everyric2.server.api.worker import (
        start_lease_sweeper,
        stop_lease_sweeper,
        sweep_orphan_worker_audio,
    )
    # 고아 잡 TTL 리퍼도 같은 이유로 lifespan에서만 띄운다 — 리스 스위퍼는 원격 워커
    # 리스가 있는 잡만 커버하고, 인프로세스 워커·번역 대기 구간은 별도 안전망이 필요하다
    # (db/orphan_reaper.py 모듈 docstring 참고).
    from everyric2.server.db.orphan_reaper import start_orphan_sweeper, stop_orphan_sweeper

    await init_db()
    _gpu_available()  # 기동 시 프리웜 — 첫 /health가 2초 페널티를 물지 않게
    # 읽기 엔진(fugashi+UniDic) 프리웜 — lazy attach가 요청 스레드에서 첫 사전 로드(수 초)를
    # 지불하지 않게 기동 시 한 번 데운다. 스레드에서 돌려 기동 자체도 안 막는다. 실패해도
    # 기능은 lazy 경로가 폴백 판정하므로 무해(로그만).
    import anyio

    def _warm_reading_engine() -> None:
        try:
            from everyric2.text.ja_reading import reading_source, tokenize_reading

            if reading_source() == "fugashi":
                tokenize_reading("予熱", phonetic=True, adopt_ruby=True)
        except Exception:
            import logging

            logging.getLogger(__name__).warning(
                "Reading engine warm-up failed; lazy attach will pay the cold load"
            )

    await anyio.to_thread.run_sync(_warm_reading_engine)
    # 이전 생의 워커 전달용 오디오 잔재 정리 — 인메모리 레지스트리가 재시작으로 비면
    # 그 파일을 지울 주체가 사라진다(저작권 규약상 남기면 안 되는 파일이다).
    swept = await anyio.to_thread.run_sync(sweep_orphan_worker_audio)
    if swept:
        import logging

        logging.getLogger(__name__).info(
            f"Swept {swept} orphaned worker-audio file(s) from a previous run"
        )
    start_lease_sweeper()
    start_orphan_sweeper()
    try:
        yield
    finally:
        # 예외로 끝나는 종료에서도 태스크를 반드시 회수한다 (누수 금지)
        await stop_orphan_sweeper()
        await stop_lease_sweeper()
        await close_db()


app = FastAPI(
    title="Everyric2 API",
    description="Lyrics synchronization API using CTC forced alignment",
    version=__version__,
    lifespan=lifespan,
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
            # 허용 집합에 falsy 값이 스미면 안 된다 — 예전 `(api_key, admin or None)`
            # 튜플은 어드민 키 미설정 시 None을 허용해, 헤더를 **아예 안 보낸** 요청
            # (provided=None)이 통과했다(엣지 감사 3.1, 런타임 재현됨). 틀린 키는
            # 막히는 형태라 수동 점검으로는 안 드러나는 우회였다.
            allowed = {server.api_key} | ({server.admin_api_key} if server.admin_api_key else set())
            provided = request.headers.get("x-api-key")
            if not provided or provided not in allowed:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "API 키가 필요해요 (확장 설정의 API 키 칸에 입력)"},
                )
    return await call_next(request)


# 확장의 모든 API 호출은 background service worker(chrome-extension:// 오리진)에서
# 나간다 — 일반 웹사이트 오리진은 반영하지 않아, 방문한 악성 페이지가 브라우저를
# 통해 파괴적 엔드포인트(DELETE/PUT)를 호출·응답 열람하는 것을 막는다.
# (curl·서버 간 호출은 Origin이 없어 CORS와 무관하게 동작)
#
# **등록 순서 주의**: Starlette은 나중에 등록한 미들웨어가 바깥이다. CORS를 위 인증
# 미들웨어보다 **뒤에** 등록해야 인증 거절(401)에도 CORS 헤더가 붙는다. 반대로 두면
# 401이 CORS 미들웨어를 거치지 않아 헤더가 빠지고, 브라우저는 그 응답을 통째로
# 차단해 확장이 상태 코드조차 못 본다 — 실측으로 키가 틀렸을 때 확장이 401이 아니라
# "Failed to fetch"를 받아 "서버 연결 실패"로 오표시하던 원인이었다.
# 싱크 조회 응답이 다국어화로 세그당 3표기 발음+모라 타이밍을 실어 210~240KB에 달한다
# (실측 2026-07-28). 압축 없이 게이트웨이·사용자 회선을 타면 확장의 조회 타임아웃을
# 넘긴다 — JSON이라 gzip이 크게 줄인다(실측 4.5배). **CORS보다 먼저 등록**한다 —
# 나중 등록이 바깥이라는 Starlette 규칙에서 CORS가 최외곽이어야 한다는 기존 불변식
# (아래 주석·tests/test_cors_on_auth_failure)을 지키기 위함이고, 압축은 안쪽에서 돌아도
# Content-Encoding이 그대로 실려 기능 차이가 없다.
from fastapi.middleware.gzip import GZipMiddleware  # noqa: E402

app.add_middleware(GZipMiddleware, minimum_size=1024)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"chrome-extension://.*",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sync_router)
app.include_router(job_router)
app.include_router(link_jobs_router)
app.include_router(translate_router)
app.include_router(cookies_router)
app.include_router(vocaro_router)
app.include_router(captions_router)
app.include_router(worker_router)
app.include_router(notices_router)
app.include_router(limits_router)
app.include_router(stats_router)


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
        "privacy": "/privacy",
    }


@app.get("/privacy", response_class=HTMLResponse, include_in_schema=False)
async def privacy_policy():
    """개인정보처리방침 — 크롬 웹스토어에 제출하는 **공개 URL**이다.

    스토어는 방침을 URL로 요구하고 그 링크를 리스팅(설치 *전* 화면)에 표시하므로, 확장 내장
    페이지로는 대신할 수 없다(``chrome-extension://``는 확장 ID에 묶여 외부에서 열 수 없고
    심사자도 못 본다). 인증 없이 열려 있어야 한다 — 이 라우트에 API 키 검사를 붙이면 심사가
    막힌다.

    ``include_in_schema=False``: /docs의 API 목록에 낄 문서가 아니다.
    """
    return HTMLResponse(PRIVACY_HTML)
