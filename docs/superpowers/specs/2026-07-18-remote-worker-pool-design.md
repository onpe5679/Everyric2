# 원격 GPU 워커 풀 (개인 풀) 설계

2026-07-18. 배경: 중앙서버를 플랫폼 서버(리눅스, GPU가 본 서비스로 바쁨)로 이전하면
생성 잡을 돌릴 GPU가 없다. 해법: 서버는 API+DB+잡 큐만 맡고, 생성 파이프라인은
원격 워커(집 PC RTX 5090 등)가 아웃바운드 폴링으로 잡을 클레임해 처리하고 결과
JSON만 제출한다. 크롬 확장은 수정 0 (서버 API만 보고, `/health`의 `gpu_available`도
확장이 사용하지 않음을 확인).

```
[확장] ──> 서버: API + DB + 큐 (GPU 불필요, EVERYRIC_SERVER_LOCAL_WORKER=false)
                  ▲ HTTPS 아웃바운드 폴링 (X-Worker-Key)
[워커 N대] everyric2 worker ── claim → yt-dlp/분리/정렬/멜로디 → 결과 JSON 제출
```

신뢰 모델: **개인 풀** (내 머신 + 신뢰 워커). 워커 키 하나를 공유하고 worker_id로
머신을 구분한다. 불특정 다수 자원봉사 그리드(결과 검증 필요)는 명시적 비범위.

## 현재 구조의 제약 (조사 완료, file:line은 2026-07-18 기준)

- 잡 실행: FastAPI `BackgroundTasks.add_task(process_job, job_id)` (`api/sync.py:452,581`)
  → 인프로세스 세마포어 `_JOB_SEMAPHORE` 대기 (`worker.py:76-85`). 별도 큐 스캐너 없음.
- 잡 입력 중 `line_meta`/`attribution`/`force`는 DB가 아니라 **서버 인메모리 스태시**
  (`_PENDING_LINE_META`/`_PENDING_ATTRIBUTION`/`_PENDING_FORCE`, `worker.py:14,16,24`).
  Job 행에는 video_id/lyrics/lyrics_hash/language만 저장 (`db/models.py:28-45`).
- 취소: 인메모리 `_CANCEL_REQUESTED: set[str]` (`worker.py:38`) + 경계 취소
  (`_consume_cancel`, `worker.py:45-56`). cancel API가 DB도 즉시 failed 마킹
  (`api/job.py:34-52`).
- 진행률: `_set_progress` (`worker.py:370-381`, 취소 가드 포함) + `_tick_progress`
  (다운로드 틱) + `_stage_monitor` (STAGE_WINDOWS 근사, `worker.py:61-69`).
- 결과 persist: `SyncRepository.create(...)` + `JobRepository.update_status(completed)`
  (`worker.py:339-357`). 오디오 해시 캐시 재사용은 `_try_complete_from_cache`
  (`worker.py:190-246`).
- `_download_and_hash`/`_run_alignment`는 DB 무접촉 순수 함수 — 원격화 용이.
  문제는 앞뒤(진행률·취소·캐시 판정·저장)가 로컬 DB 세션+인메모리 결합.

## 설계 결정

### D1. 클레임/리스 모델 — DB 스키마 변경 없음

리스(어느 워커가 어떤 잡을 물고 있는지 + 만료 시각)는 **서버 인메모리 레지스트리**
`{job_id: (worker_id, expires_at)}`로 관리한다. 근거: 서버는 단일 프로세스이고, 서버
재시작 시 잡 유실은 기존 좀비 잡 정리(`db/connection.py:41-57`)가 이미 커버하는
동작이라 일관적이다. Job 테이블에 컬럼을 추가하지 않아 마이그레이션이 불필요.

리스 만료(하트비트 끊김)는 클레임 요청 처리 시 lazy 스윕: 만료 리스의 잡을
`status=queued`로 되돌리고 레지스트리에서 제거. 스태시(line_meta 등)는 peek 방식이라
재클레임 시 다시 전달 가능하다.

### D2. 워커 API 5개 — `/api/worker/*`, 새 라우터 `server/api/worker.py`

인증: 헤더 `X-Worker-Key` == `EVERYRIC_SERVER_WORKER_KEY`(신규 설정, 기본 "").
키 미설정이면 워커 API 전체 403 (기능 비활성). 기존 api_key 미들웨어(`main.py:59-78`)는
`/api` 전체에 X-API-Key를 요구하므로, 워커 엔드포인트는 X-Worker-Key가 유효하면
X-API-Key 검사를 면제한다 (미들웨어에 예외 추가 — 워커는 클라이언트 키를 모름).

1. `POST /api/worker/claim` — body `{worker_id: str, version: str}`.
   - `version != everyric2.__version__`이면 409 + 안내 메시지 (스키마 어긋남 방지).
   - 만료 리스 스윕 → 가장 오래된 `queued` 잡 선택 → `status=processing, progress=0,
     stage="워커 할당"` 마킹, 리스 등록.
   - 응답: `{job: {job_id, video_id, lyrics, language, line_meta, attribution, force,
     max_audio_sec}, lease_seconds}` 또는 `{job: null}`. line_meta 등은 인메모리
     스태시를 **peek**(제거하지 않음)해서 싣는다.
2. `POST /api/worker/jobs/{id}/progress` — body `{progress: int, stage: str}`.
   - 리스 검증(해당 worker_id 소유) → `_set_progress` 경유(취소 가드 재사용) →
     리스 갱신. 응답 `{cancel_requested: bool}` (`_CANCEL_REQUESTED` 조회; true 반환
     시 레지스트리·스태시 정리 — 잡은 cancel API가 이미 failed 마킹했음).
3. `POST /api/worker/jobs/{id}/cache-check` — body `{audio_hash: str}`.
   - `_try_complete_from_cache` 로직 재사용. 응답 `{completed: bool}`. true면 워커는
     잡을 종료(정렬 생략) — S1 교차 영상 캐시 기능이 원격에서도 유지된다.
4. `POST /api/worker/jobs/{id}/result` — body `{timestamps, language, quality_score,
   audio_hash, extra}` (인프로세스 저장 경로 `worker.py:339-357`와 동일 데이터).
   - **status가 processing이고 리스 소유자일 때만 수락** — 취소된 잡(failed)·좀비
     정리된 잡의 뒤늦은 결과를 거부한다. 수락 시 `SyncRepository.create` +
     `update_status(completed, result_id)` + 리스·스태시 정리.
5. `POST /api/worker/jobs/{id}/fail` — body `{error: str}`. 리스 검증 후 failed 마킹
   + 정리. (워커 쪽 다운로드 실패·파이프라인 예외 보고용. 메시지는 사용자에게
   보이므로 기존 워커의 한국어 실패 문구 톤을 따른다.)

### D3. 파이프라인 코어 추출 (핵심 리팩터)

`_process_job_inner`의 파이프라인 본체를 콜백 주입형 함수로 추출한다:

```python
async def run_pipeline(job: JobInput, hooks: PipelineHooks) -> PipelineResult | None
# JobInput: job_id, video_id, lyrics, language, line_meta, attribution, force, max_audio_sec
# PipelineHooks:
#   progress(progress: int, stage: str) -> Awaitable[bool]  # False = 취소 요청됨
#   cache_check(audio_hash: str) -> Awaitable[bool]         # True = 캐시로 완결, 중단
# PipelineResult: timestamps, language, quality_score, audio_hash, extra
```

- 인프로세스 워커: hooks가 기존 `_set_progress`/`_consume_cancel`/
  `_try_complete_from_cache`를 감싼다. **관찰 가능한 동작(단계 문구·진행률 값·취소
  경계·캐시 동작·실패 문구)은 리팩터 전과 동일해야 한다.**
- 원격 워커: hooks가 워커 API 호출을 감싼다. `_tick_progress`/`_stage_monitor`의
  진행률 근사도 코어에 남겨 원격에서도 같은 UX가 나오게 한다 (진행률 보고 주기는
  하트비트를 겸하므로 최소 리스 갱신 주기보다 짧아야 함 — 2초 틱이면 충분).

### D4. 서버 로컬 처리 토글

`EVERYRIC_SERVER_LOCAL_WORKER: bool = True` (신규 설정). False면 generate/regenerate가
`add_task`를 생략하고 잡을 직접 `status=queued`로 마킹만 한다 (queue_position 표시
유지 — `count_queued_before` 활용). 스태시 적재는 동일하게 수행 (클레임 응답이 읽음).

### D5. 워커 CLI

`everyric2 worker` (cli.py의 기존 `@app.command()` 패턴, `cli.py:872-911` serve 참조):

```
everyric2 worker --server https://everyric.example.com --key <워커키>
                 [--worker-id <이름=hostname>] [--poll 5.0] [--once]
```

- 루프: claim → 없으면 poll초 대기 → 있으면 run_pipeline(원격 hooks) → result/fail 제출.
- 한 번에 한 잡 (개인 풀 GPU 1장 전제. 병렬은 워커 프로세스를 더 띄우는 것으로 해결).
- 네트워크 오류: 지수 백오프(최대 60s), 잡 처리 중 progress 실패는 재시도하되
  연속 실패 시 잡 포기(fail 제출 시도 후 다음 루프).
- cancel_requested 수신 시 경계에서 포기, 제출 없음.
- 시작 시 서버 `/health` 확인 + 버전 로그. httpx 대신 기존 의존성 `requests` 사용
  (파이프라인 스레드와 별개로 동기 호출을 `asyncio.to_thread`로 감싸거나, 워커 루프
  자체를 동기로 구성 — 구현 단순한 쪽 선택).

### D6. 리스 파라미터

`EVERYRIC_SERVER_WORKER_LEASE_SEC: int = 120` (신규 설정). 진행률 보고(≤2s 간격)가
하트비트를 겸하므로 여유가 크다. CTC/demucs 스레드 실행 중에도 stage_monitor 틱이
돌므로 장시간 무보고 구간은 없다.

## 비범위 (이번 배치에서 안 함)

- Job 테이블 스키마 변경·알렘빅 도입 (D1로 회피)
- 결과 검증/평판 (개인 풀 신뢰 모델)
- 워커별 우선순위·능력 매칭, 멀티 GPU 스케줄링
- 확장(everyric2-chrome) 변경 — 0이어야 한다

## 테스트

기존 패턴(`tests/test_sync_link.py`: 라우트 코루틴 직접 await + in-memory sqlite
StaticPool + `db_conn.async_session` 몽키패치 + `asyncio.run`) 준수. httpx/TestClient
금지 (기존 주석의 이유 유지).

- claim: 키 없음/오류 403 · 버전 불일치 409 · queued 클레임 → processing+페이로드에
  lyrics/line_meta 포함 · 빈 큐 → job null · 만료 리스 스윕 후 재클레임 가능
- progress: 리스 갱신 · 취소 후 cancel_requested=true + 스태시 정리
- result: 정상 수락 → SyncResult 생성+completed · 취소된 잡 거부 · 타 worker_id 거부
- fail: failed 마킹 + error 문구
- LOCAL_WORKER=false: generate가 add_task 없이 queued 마킹 (BackgroundTasks에 태스크
  미등록 확인)
- 리팩터 회귀: 기존 테스트 전부 통과 (인프로세스 경로 동작 보존)

## 검증 (구현 후, 메인 세션이 직접)

pytest + ruff check 전체 통과 → 로컬 E2E: 127.0.0.1:8001에 임시 DB·
LOCAL_WORKER=false·워커 키로 서버 기동, `everyric2 worker --server
http://127.0.0.1:8001`로 실제 짧은 영상 생성 잡을 원격 경로로 완주시켜 timestamps
확인 (반드시 127.0.0.1, localhost 금지).

## 배포 반영 (구현 후 문서 갱신)

- `deploy/.env.example`: `EVERYRIC_SERVER_WORKER_KEY`, `EVERYRIC_SERVER_LOCAL_WORKER=false` 항목
- `deploy/DEPLOY.md`: "API 전용 서버 + 원격 워커" 구성 절 추가 (집 PC에서
  `everyric2 worker` 상시 실행, Windows 작업 스케줄러 예시)
