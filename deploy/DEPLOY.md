# everyric2 리눅스 서버 배포 가이드

everyric2 중앙서버를 NVIDIA GPU가 있는 리눅스 서버(플랫폼 서버 동거)에 올리는 절차.
플랫폼 코드는 건드리지 않는다 — everyric2는 별도 systemd 서비스로 돌고, 기존 nginx가
서브도메인으로 프록시만 해준다.

```
[크롬 확장] ──HTTPS──> nginx ──> 127.0.0.1:8000 everyric2 (systemd)
                         └─────> 기존 플랫폼 (그대로)
```

전제: systemd 운영 리눅스, NVIDIA 드라이버 설치됨(`nvidia-smi` 동작), nginx 운영 중,
서브도메인 하나 확보(예: `everyric.example.com` → 서버 IP A레코드).

> **참고(2026-08-04 추가)**: 아래 1~7절은 새 서버에 처음 설치하는 절차(루트 권한,
> `/opt/everyric2`, 루트 systemd 유닛 `everyric2.service`) 기준이다. 이미 운영 중인
> 서버에 **동거 배포**하거나 sudo 없이 유저 systemd로 운영하려면
> `deploy/everyric2-user.service`(API) + `deploy/everyric2-worker-user.service`(GPU
> 워커)를 대신 쓴다 — 두 파일 머리말에 심링크·활성화 명령이 있다. 이 경우
> `WorkingDirectory`가 `%h/everyric2`(리포 위치에 맞게 유닛을 drop-in으로 덮어 조정
> 가능, 유닛 파일 주석 참고)이고 포트도 `.env`의 `EVERYRIC_PORT`로 자유롭게 바꿀 수
> 있다 — 8000을 다른 서비스가 이미 쓰는 호스트라면 이쪽이다. **이 문서의 §3(systemd
> 등록) 외 나머지 절차(설치·설정·업데이트·마이그레이션·롤백)는 경로·유닛 이름만 바꾸면
> 유저 유닛 배포에도 그대로 적용된다** — 별도 문서는 없다.

## 0. 사전 확인

```bash
nvidia-smi                       # 드라이버·GPU 인식 확인
df -h /                          # 여유 디스크: 모델 캐시 ~5GB + 오디오 임시 파일
```

## 1. 코드·런타임 설치

```bash
sudo mkdir -p /opt/everyric2 && sudo chown "$USER" /opt/everyric2
git clone <리포-URL> /opt/everyric2
cd /opt/everyric2

# uv 미설치 시
curl -LsSf https://astral.sh/uv/install.sh | sh

# demucs(보컬 분리, 구스택용) + audio-separator/onnxruntime(bs-polarformer-fp16, 새
# 스택 medium/heavy 깊이용) 포함 설치. 새 스택이 기본값이므로(아래 "새 정렬 스택 준비물"
# 참고) 이 extra는 사실상 필수다 — 없으면 정상곡(fast 깊이)만 동작하고 극한곡(medium/
# heavy로 승급하는 곡)은 전부 명시적으로 실패한다.
uv sync --extra separator
```

### GPU torch 확인

```bash
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

`True`가 아니거나, GPU가 RTX 50xx(Blackwell, sm_120)인데 CUDA가 12.8 미만이면:

```bash
uv pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu128
```

주의: `uv sync`를 다시 돌리면 락파일 버전으로 되돌아가므로, sync 후에는 위 명령을
재실행해야 한다 (영구 고정하려면 pyproject `[tool.uv.sources]`에 인덱스를 박는다).

### 새 정렬 스택(owsm/omniasr 앵커 + 2패스, 기본값) 준비물

`alignment.engine` 기본값이 `owsm`이라(요청마다 fast/medium/heavy 3단계 라우팅이 실제
앵커를 고른다 — `everyric2/server/worker.py::_run_new_stack_alignment` 참고) 아래 셋이
**전부** 있어야 새 스택이 완전히 동작한다. 하나라도 없으면 해당 요청이 조용히 저하되지
않고 명시적으로 실패한다(운영자 지시 — `jobs.failure_kind='system'`으로 기록됨. fast
깊이만으로 끝나는 정상곡은 분리·OWSM이 없어도 동작한다 — 아래가 필요해지는 건
medium/heavy로 승급하는 곡부터다).

1. **bs-polarformer-fp16 분리기 자산** — `everyric2/audio/polarformer_separator.py`가
   `EVERYRIC_AUDIO_SEPARATOR_MODEL_DIR`(기본 `~/.cache/everyric2/models`) 아래에서 찾는다:
   - `model_bs_polarformer_float16.ckpt` / `.yaml` — 체크포인트·설정
   - `msst_src_<커밋8자리>/models/bs_roformer/{attend.py,bs_roformer.py}` — MSST 벤더
     소스. 핀 커밋은 그 모듈의 `_MSST_COMMIT` 참고 — 정확히 맞아야 한다(main을 따라가면
     모델 정의가 바뀌어 체크포인트가 안 붙는다).
   - 자산 URL·조달 절차는 `scripts/bench_adapters/separators_quality.py`의
     `BS_POLARFORMER` 정의를 참고한다. **서버는 이 자산을 절대 자동으로 내려받지
     않는다** — 없으면 기동이 아니라 그 백엔드를 실제로 쓰는 요청 시점에 실패한다.
2. **`audio-separator` + `PoPE-pytorch` 파이썬 패키지** — 둘 다 `uv sync --extra
   separator`(위 명령)로 설치. `audio-separator`는 `onnxruntime`(CPU 빌드)을 같이
   끌고 온다 — `audio_separator.separator` 패키지의 임포트 체인이 무조건 요구하지만,
   이 코드 경로는 그걸 실제 추론에는 안 쓴다(로더를 몽키패치로 갈아끼운다 —
   `polarformer_separator.py` 모듈 docstring 참고). `PoPE-pytorch`는 MSST 벤더 소스
   (`bs_roformer.py`)가 극좌표 위치 임베딩(PoPE, arXiv 2509.10534) 사용 여부를 조건부
   import로 가르는데, 없으면 자산 검사는 통과해도 첫 요청의 forward 시점에
   `AssertionError`로 죽는다(실곡 검증 2026-08-04 — 그래서 `polarformer_separator.
   _missing_reasons`가 이 패키지도 사전 검사한다). 기존 torch/numpy/scipy/transformers를
   재설치·강등하지 않는다(2026-08-04 `uv lock` 실측 확인, `pyproject.toml`의
   `separator` extra 주석 참고).
3. **OWSM 전용 격리 venv** — ESPnet이 메인 venv의 torch/transformers 버전과 충돌해
   별도 venv가 필요하다(`everyric2/alignment/owsm_engine.py` 모듈 docstring). 기본
   경로는 `<repo_root>/.venv-owsm`, 다른 경로면 `.env`에
   `EVERYRIC_ALIGNMENT_OWSM_PYTHON_PATH`를 채운다. 이 venv를 만드는 절차 자체는 이
   문서 범위 밖이다 — `scripts/bench_adapters/owsm_ctc.py`/`benchmark/.venv-owsm`과
   같은 구성(ESPnet + SentencePiece + `espnet/owsm_ctc_v4_1B` HF 캐시 스냅샷)으로
   맞춘다.

**자산 조달 전까지 임시로 구스택으로 되돌리려면** `.env`에 아래 둘을 **함께** 채운다
(하나만 바꾸면 기동이 거부된다 — `Settings`의 cross-field validator가 새 앵커+htdemucs
조합을 조용한 오조합으로 보고 막는다):

```bash
EVERYRIC_ALIGNMENT_ENGINE=ctc
EVERYRIC_AUDIO_SEPARATOR_BACKEND=htdemucs
```

### deno — yt-dlp JS 런타임

yt-dlp가 유튜브 서명 해독에 JS 런타임을 쓴다. 서비스 계정 홈에 설치하면
서버 코드가 `~/.deno/bin`을 자동으로 PATH에 얹는다:

```bash
curl -fsSL https://deno.land/install.sh | sh
```

## 2. 설정 (.env)

```bash
cp deploy/.env.example .env
openssl rand -hex 32             # → .env의 EVERYRIC_SERVER_API_KEY에
nano .env                        # 키·경로 채우기 (파일 안 주석 참고)
```

- `.env`와 `nvapi.txt`는 gitignore에 있어 커밋되지 않는다.
- NIM 번역을 쓰면 `NVIDIA_API_KEY`를 채우거나 리포 루트에 `nvapi.txt`를 둔다.

## 3. systemd 등록

```bash
sudo cp deploy/everyric2.service /etc/systemd/system/
sudo nano /etc/systemd/system/everyric2.service   # User=CHANGE_ME 를 실제 계정으로
sudo systemctl daemon-reload
sudo systemctl enable --now everyric2
curl -s http://127.0.0.1:8000/health              # {"status":"healthy",...} 확인
```

함정: `WorkingDirectory=/opt/everyric2`가 핵심이다 — SQLite 상대 경로, `nvapi.txt`,
pydantic `.env` 해석이 전부 이 디렉터리 기준. 지우거나 바꾸면 DB가 엉뚱한 곳에 생긴다.

## 4. nginx + HTTPS

```bash
sudo cp deploy/nginx-everyric.conf /etc/nginx/sites-available/everyric
sudo nano /etc/nginx/sites-available/everyric     # server_name을 실제 서브도메인으로
sudo ln -s /etc/nginx/sites-available/everyric /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
sudo certbot --nginx -d everyric.example.com      # 플랫폼에서 쓰는 방식 그대로
curl -s https://everyric.example.com/health
```

- 8000 포트는 루프백 바인드라 방화벽을 열 필요 없다. 공개는 80/443만.
- 경로 방식(`/everyric/`)으로 붙이려면 uvicorn `--root-path` 설정이 추가로 필요하다
  — 서브도메인이면 무수정이라 이쪽을 권장.

## 5. 크롬 확장 전환

확장 옵션 → 서버 주소 `https://everyric.example.com` + API 키 입력.
패널 상태 표시가 healthy(초록)면 끝.

## 6. 첫 가동 시 알아둘 것

- 첫 싱크 생성 때 HuggingFace 모델 다운로드(수 GB)가 있어 첫 잡만 오래 걸린다
  (`~/.cache/huggingface`에 캐시됨).
- `EVERYRIC_SERVER_MAX_CONCURRENT_JOBS=1`이 GPU 1장 기준 안전값 — 초과분은
  status=queued로 대기한다.

## 7. 운영

```bash
journalctl -u everyric2 -f                        # 로그
git pull && uv sync --extra separator && sudo systemctl restart everyric2   # 업데이트
```

- yt-dlp 403이 뜨면(유튜브 쪽 변경이 잦다):
  `uv lock --upgrade-package yt-dlp && uv sync --extra separator && sudo systemctl restart everyric2`
  그래도 안 되면 `.env`의 `EVERYRIC_AUDIO_COOKIE_FILE` 또는
  `EVERYRIC_AUDIO_SOURCE_ADDRESS`(멀티 회선) 사용.
- DB 백업: 서비스 잠깐 멈추고 `everyric2.db` 파일 복사.

### 업데이트가 DB 스키마 변경을 포함할 때

alembic을 안 쓴다 — 새 컬럼은 `init_db()`(`everyric2/server/db/connection.py`)가 **서버
기동 시점에** `PRAGMA table_info`로 존재를 확인하고 없으면 `ALTER TABLE ... ADD COLUMN`을
직접 실행한다(멱등, nullable 컬럼만 추가하는 additive 계약 — NOT NULL 추가·컬럼 rename은
하지 않는다). 즉 **재기동이 곧 마이그레이션 실행**이다 — 아래 순서를 지킨다:

1. **먼저 DB를 백업한다**(`systemctl stop everyric2` → `everyric2.db` 복사 → 다시
   start, 또는 띄운 채로 `sqlite3 everyric2.db ".backup backup.db"`). 컬럼 추가만이면
   되돌리기 쉽지만, 같은 배치에 데이터 백필(UPDATE)이 섞이면 되돌리기 번거로울 수 있다 —
   커밋 로그·릴리스 노트에 "마이그레이션"·"백필" 언급이 있으면 반드시 백업한다.
2. `git pull && uv sync --extra separator && sudo systemctl restart everyric2`.
3. 워커를 별도 프로세스(§8) 또는 별도 유닛으로 운영 중이면, **서버가 완전히 뜬 뒤에**
   워커를 재기동한다 — 서버·워커 버전이 다르면 워커가 잡을 못 받는 안전장치가 있다(§8
   참고, 순서를 반대로 해도 위험하진 않고 그 안전장치에 걸려 잠깐 잡을 못 받을 뿐이다).
4. 재기동 후 `.venv/bin/python scripts/verify_deploy.py <서버URL>`을 돌려 확인한다
   (신규 라우터·스키마 반영·의존성 상태를 한 번에 점검 — 읽기 전용, 기본 실행은 아무것도
   쓰지 않는다). FAIL이 있으면 그 결과로 롤백 여부를 판단한다(§10).

## 8. API 전용 서버 + 원격 GPU 워커 (서버 GPU가 바쁠 때 권장)

서버의 GPU가 본 서비스로 바쁘면, 서버는 API+DB+잡 큐만 맡기고 생성 파이프라인은
원격 워커(집 GPU PC 등)에 맡길 수 있다. 확장은 잡이 어디서 처리되는지 모른다 —
조회는 24시간 되고, 생성은 워커가 켜져 있을 때 처리된다(꺼져 있으면 큐 대기).

서버 쪽 `.env`:

```bash
EVERYRIC_SERVER_WORKER_KEY=<openssl rand -hex 32>
EVERYRIC_SERVER_LOCAL_WORKER=false      # 이 서버는 GPU 처리 안 함
sudo systemctl restart everyric2
```

워커 쪽 (집 PC 등 — 리포 clone + `uv sync --extra separator` + GPU torch는 1절과 동일):

```bash
uv run everyric2 worker --server https://everyric.example.com --key <워커 키>
```

- 워커는 아웃바운드 폴링이라 워커 쪽 공유기 포트포워딩·방화벽 설정이 필요 없다.
- 워커 여러 대를 같은 키로 띄우면 풀이 된다 (`--worker-id`로 머신 구분).
- 워커가 잡을 물고 죽으면 리스 만료(기본 120초) 후 자동으로 큐에 반환된다.
- 서버·워커의 everyric2 버전이 다르면 워커가 잡을 받지 못한다(의도된 안전장치) —
  업데이트는 서버 먼저, 워커들 순서로.
- Windows 워커 자동 시작: 작업 스케줄러에 "로그온 시" 트리거로
  `uv run everyric2 worker ...`를 등록하면 된다 (작업 디렉터리 = 리포 루트).

## 9. 나중 확장 훅 (이번 배포에선 안 함)

이 구조 그대로 두고 얹을 수 있게 접점만 정리해 둔다:

- ~~**yt-dlp 캐시 공유(2단계)**~~ → **구현됨**. `video_id` 오디오 캐시가 확보 경로 맨 앞에
  서고(`everyric2/audio/cache.py`), 같은 영상의 동시 요청은 락으로 한 번으로 병합된다.
  설정은 `EVERYRIC_AUDIO_CACHE_*`(위 `.env.example`) — **`EVERYRIC_AUDIO_TEMP_DIR`과 같은
  곳으로 지정하지 말 것**(`/tmp`가 tmpfs면 RAM을 먹고 재부팅에 사라진다).
  왜 필요했는지: 기존 캐시는 `(audio_hash, lyrics_hash)` 키라 해시를 구하려면 파일이,
  파일을 구하려면 다운로드가 필요했다. 2026-07-26 밤샘 배치 실측으로 **싱크 생성 182건에
  유튜브 다운로드 275회**가 확인됐다 — GPU만 아끼고 유튜브 접촉은 하나도 아끼지 못했다.
- **원곡 참조 연동(3단계)**: 플랫폼이 `X-API-Key` 헤더로 everyric2 REST를 호출.
  `GET /api/sync/{video_id}`(싱크·linked 조회), SyncLink(원곡 video_id+offset+rate)가
  이미 있어 everyric2 쪽 수정 없이 연계 기능을 만들 수 있다.

## 10. 롤백

기본 원칙: 이 프로젝트는 **API 응답·DB 컬럼을 additive로만 바꾼다**(신규 필드는 옵셔널,
컬럼은 nullable, 기존 필드 rename·삭제 없음) — 그래서 구버전 코드로 돌아가도 DB 자체를
되돌릴 필요가 없다. 구코드는 새 컬럼·새 응답 필드를 그냥 모른 채 무시한다.

1. `git log --oneline`으로 되돌릴 커밋(또는 이전 릴리스 태그)을 확인 → `git checkout
   <커밋/태그>`.
2. **`uv sync --extra separator`를 다시 돈다** — 되돌아간 시점의 `uv.lock`을 기준으로
   의존성도 함께 되돌아간다(코드만 되돌리고 sync를 생략하면 새 코드가 기대하던 패키지
   버전이 그대로 남아 오히려 불일치가 난다).
3. 워커가 있으면(§8 또는 유저 유닛) 여기서도 순서를 지킨다 — 서버 먼저, 워커는 그 다음.
4. `sudo systemctl restart everyric2` (+ 워커 유닛).
5. `.venv/bin/python scripts/verify_deploy.py <서버URL>`로 롤백 후 상태를 확인한다 —
   되돌린 배치가 추가한 라우터(예: `/api/notices`)가 다시 404로 나오는 것은 **정상**이다
   (구코드에는 그 라우터가 없다).
6. DB는 되돌리지 않는다(1단계에서 뜬 백업은 정말 필요할 때만 복원용으로 쓴다) — 새
   컬럼이 테이블에 남아 있어도 구코드가 그냥 무시하므로 무해하다.

### 이번 배치(1.6.0)에서 롤백이 안전한 근거

- `notices.translations`(nullable, 기존 `title`/`body` 의미 불변 — 폴백 언어로 유지).
- `/api/limits` 응답의 `next_reset_at`·`link`·`upgrade` 분리는 기존 필드를 그대로 두고
  필드만 **추가**한다.
- 위 두 가지와 §7의 발음 의존성 수리(cmudict·pypinyin 락)는 이번 점검(verify_deploy.py
  실측)으로 직접 확인했다.
- vocaro 버전 선택·동의어 변경은 **API 응답 스키마를 한 줄도 안 바꾼다**(확인: 두 커밋
  b940e12·71fb4f0의 변경 범위가 `everyric2/sources/vocaro.py` 파서 내부 로직과 테스트뿐,
  `everyric2/server/api/vocaro.py` diff 없음). 롤백하면 표 선택이 예전 동작으로 돌아갈
  뿐이라 계약 파손이 없다.
- 롤백의 **의도된 부작용**: cmudict·pypinyin 락 수리도 함께 빠지므로, 롤백 후에는 en/zh
  발음이 다시 kana 단독 근사로 저하된다. 새로운 결함이 아니라 롤백 자체의 결과다 — 롤백
  후 원인 조사 시 착각하지 않도록 남겨 둔다.
