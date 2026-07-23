# 중립 연동 3종 + 파이프라인 성능 — 설계 스펙 (2026-07-24)

배경: 중앙 서버가 외부 플랫폼과 동거하며 3가지 중립 계약으로 연동한다(계약 상세는 외부 문서,
리포에는 계약명·필드만). 더불어 처리 파이프라인의 속도 개선 2종. 원칙: **이 리포에는 특정
외부 서비스를 추론할 수 있는 이름·경로·스키마를 남기지 않는다** — "외부 미디어 캐시",
"외부 곡 인덱스" 같은 중립 명칭만 사용.

작업 분할: WS1(통합 C+D+E) → WS2(성능 A+B) 순차. 마일스톤마다 메인이 커밋.

---

## WS1-C. 곡 인덱스 프록시 전환 (songindex/1 consumer)

- `ServerSettings`에 `song_index_url: str = ""`, `song_index_key: str = ""` 추가
  (env: `EVERYRIC_SERVER_SONG_INDEX_URL/KEY`).
- `api/vocaro.py`:
  - `match_title`: url 설정 시 업스트림 `GET {url}/match?title=<urlencoded>`
    (`Authorization: Bearer <key>`, timeout 3s, `asyncio.to_thread(requests.get,...)`) →
    응답 {found, slug, page_url, ko, ja} → 기존 `VocaroMatchResponse`로 1:1 매핑
    (확장 쪽 형태 무변경). 업스트림 오류/타임아웃 → `found=false, status="upstream_error"`.
  - url 미설정 시 기존 로컬 인덱스 경로 그대로 (이관 검증 후 별도 커밋으로 제거 예정 —
    이번 WS에서 크롤러를 지우지 않는다).
  - `/reindex`: 업스트림 모드에선 빌드 킥 없이 `{"status":"upstream"}`.
  - `/status`: 업스트림 모드에선 업스트림 `/status` 중계(오류 시 building=false, total=0).
- 의존성 추가 금지 (requests 기존 사용).

## WS1-D. 외부 미디어 캐시 조회 + 워커 오디오 전달 (mediacache/1 consumer)

- `ServerSettings`: `media_cache_url: str = ""`, `media_cache_key: str = ""`.
- 조회 시점 = 잡이 처리 주체에게 넘어가는 순간:
  - 원격: `/api/worker/claim`에서 sync 잡 확정 직후.
  - 인프로세스: `process_job` 슬롯 획득 직후.
- 흐름: `GET {url}/lookup?platform=youtube&id=<video_id>` (Bearer, timeout 3s) →
  `{found, path, ext, duration_sec}`.
  - `found=false`/오류/`path` 비존재·비가독 → 조용히 기존 yt-dlp 경로 (INFO 로그 1줄).
  - `duration_sec`이 있고 `max_audio_sec` 초과 → 기존 과길이 실패 문구로 잡 즉시 실패
    (다운로드 없이 프리플라이트).
  - 히트 → ffmpeg 추출: `nice -n 19 ionice -c 3 ffmpeg -y -i <path> -vn -acodec copy
    <EVERYRIC_AUDIO_TEMP_DIR>/<video_id>-<job8>.m4a`. **전역 asyncio.Semaphore(1)**
    (동거 호스트 CPU/NAS I/O 예산 — 합의 조건). 스트림카피 실패(코덱 비호환) → yt-dlp 폴백.
    Windows에는 nice/ionice 없음 — POSIX에서만 프리픽스, 아니면 ffmpeg 단독.
- 워커 전달:
  - claim 응답에 `audio_url: str | None` 추가 (예: `/api/worker/jobs/{job_id}/audio`).
  - `GET /api/worker/jobs/{job_id}/audio`: X-Worker-Key + **리스 소유 워커만**(409 규약 재사용),
    FileResponse. 파일 정리는 잡 터미널 지점(스태시 pop과 동일 위치들) + 서버 재시작 좀비
    정리에 편승.
  - `JobInput`에 `audio_url: str | None = None` 추가. `run_pipeline`의 다운로드 블록:
    audio_url 있으면 yt-dlp 대신 HTTP 다운로드(워커는 서버 base URL + X-Worker-Key로,
    인프로세스는 로컬 파일 직사용 — JobInput에 `audio_path: str | None`도 허용해 두 경로 통일).
    audio_hash는 받은 파일로 동일 계산. 실패 시 yt-dlp 폴백.
- 저작권 규약: 추출 오디오는 워커 전달용 임시 파일 — 외부 재서빙 엔드포인트 금지
  (워커 인증 뒤에만 존재), 터미널 시 삭제.

## WS1-E. 링크 후보 검증 (link-jobs) + 반주 상관 판정

목적: "커버 영상이 원곡과 같은 반주(인스트)를 쓰는가"를 자동 판정해 SyncLink를 안전하게
자동 생성. 외부 오케스트레이터(무엇이든)가 X-API-Key로 호출하는 범용 API.

- DB: 신규 테이블 `link_jobs` (create_all 추가 — 기존 스키마 무변경):
  id(uuid pk), video_id, source_video_id, status(queued/processing/done/failed),
  match(bool|null), offset_sec(float|null), confidence(float|null), error(str|null),
  created_at(server_default now — count 비교 시 기존 off-by-one 교훈 주의).
- API (`api/link_jobs.py`):
  - `POST /api/link-jobs` {video_id, source_video_id} (X-API-Key) → {id}. 동일 쌍
    진행 중(queued/processing) 존재 시 그 id 반환(중복 방지).
  - `GET /api/link-jobs/{id}` → {status, match, offset_sec, confidence, error}.
- 워커 프로토콜 확장 (`api/worker.py` + `cli.py`):
  - claim 응답에 `kind: "sync" | "link_validate"` (기본 "sync" — 구버전 워커는 버전 게이트로
    이미 차단되므로 호환 부담 없음). sync 잡 우선, 없으면 link 잡 FIFO.
  - 리스 레지스트리 공유하되 키는 `link:{id}`로 네임스페이스 분리. 리스 300s(처리가 짧아
    v1은 진행 하트비트 생략).
  - `POST /api/worker/link-jobs/{id}/result` {match, offset_sec, confidence} /
    `POST /api/worker/link-jobs/{id}/fail` {error} — status·리스 소유 검증은 sync 잡 규약 준용.
  - result 수신 시 match=true면 `SyncLinkRepository.upsert(video_id, source_video_id,
    offset_sec, rate=1.0)` 자동 실행. **offset 부호 규약: 기존 GET /api/sync/{video_id}의
    SyncLink 소비부가 적용하는 방향과 일치**시키고 테스트로 못 박을 것 (코드 읽고 규약을
    docstring에 명시).
- 워커 태스크 (`cli.py` 또는 전용 모듈):
  1. 두 video_id 오디오 확보 — audio_url(서버 캐시) 우선, 없으면 yt-dlp.
  2. 각각 demucs 분리 → **반주 스템** 확보 (separator가 vocals만 반환하면 반주 = 믹스−보컬
     파형 차분; separator 구현을 읽고 실제 스템 출력에 맞출 것).
  3. 반주 2개를 동일 sr(22050 모노)로 리샘플 → `librosa.onset.onset_strength` 엔벨로프
     (hop 512) → 정규화 크로스 코릴레이션 → 최고 피크 위치=offset_sec(탐색 범위 ±90s),
     confidence = 피크값 / (이차피크값+eps) 기반 정규화 점수.
  4. `confidence >= settings.server.link_match_threshold`(신규, 기본 0.35, env 조정) →
     match. offset은 초 단위 float.
  - GPU 실행이 로컬에서 불가하므로(개발기 GPU 사용 금지) 상관 로직은 **순수 함수로 분리**
    (`everyric2/audio/correlate.py` 등): 파형 2개 입력 → (offset, confidence). 합성 신호
    (클릭 트랙 시프트 + 노이즈)로 단위 테스트 — 시프트 검출 ±1 hop 오차 내, 무관 신호는
    저신뢰 판정.

## WS2-A. 모델 웜 캐시

- 대상: VocalSeparator(demucs), 정렬 엔진(CTC 모델), MelodyExtractor(FCPE/RMVPE).
- 방식: 각 소유 모듈에 지연 싱글턴 + threading.Lock (프로세스 수명 상주). 두 번째 잡부터
  모델 로드 0회가 요구사항. `EVERYRIC_SERVER_WARM_MODELS`(기본 true)로 끄면 기존처럼
  잡마다 생성·해제.
- 상주 주체는 워커 프로세스(CLI)와 인프로세스 서버 — API 전용 모드 프로세스에는 어떤
  경로로도 모델이 로드되면 안 된다(기존 torch 지연 임포트 불변).
- 로그: 웜 히트 시 "warm model reuse: <name>" 1줄 (실측 비교용).

## WS2-B. 멜로디 f0를 정렬과 병렬 실행

- `_run_alignment` 내부에서 f0 전곡 계산을 CTC 정렬과 동시에 스레드로 수행,
  annotate(노트 부착)는 정렬 결과가 나온 뒤. MelodyExtractor에 f0 사전 계산/주입 API가
  없으면 분리(계산·부착 단계 분해)한다.
- stage 표시 순서(보컬 분리→전사 정렬→타이밍 보정→멜로디 분석)는 기존 그대로 —
  진행 UX 계약 불변. 멜로디 실패는 지금처럼 노트 없이 계속(치명 아님).

## 공통 제약

- 테스트: 기존 패턴(라우트 코루틴 직접 await + `sqlite+aiosqlite:///:memory:` StaticPool +
  async_session 몽키패치, httpx/TestClient 금지). 신규 커버: 프록시 분기(설정 유/무/오류),
  claim kind 분기, audio_url 인가(리스 비소유 409), link result→SyncLink 생성·부호,
  상관 함수 합성 신호, 중복 link-job 병합.
- uvx ruff 신규 위반 0. 의존성 추가 금지(librosa/scipy/soundfile 기존 존재 확인 후 사용).
- 검증 계획: 로컬 pytest 전체 green → 메인이 커밋 → 서버(3090 워커)에서 실전 통합 테스트.
