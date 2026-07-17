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

# demucs(보컬 분리) 포함 설치 — 없어도 돌지만 VAD 클램프/멜로디 품질이 떨어진다
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

## 8. 나중 확장 훅 (이번 배포에선 안 함)

이 구조 그대로 두고 얹을 수 있게 접점만 정리해 둔다:

- **yt-dlp 캐시 공유(2단계)**: `EVERYRIC_AUDIO_TEMP_DIR`을 플랫폼 캐시 디렉터리로
  바꾸고, 다운로더에 video_id 기준 "받기 전 캐시 조회"를 추가(소규모 수정 — 현재는
  잡마다 `{video_id}-{job_id}` 파일명으로 새로 받는다).
- **원곡 참조 연동(3단계)**: 플랫폼이 `X-API-Key` 헤더로 everyric2 REST를 호출.
  `GET /api/sync/{video_id}`(싱크·linked 조회), SyncLink(원곡 video_id+offset+rate)가
  이미 있어 everyric2 쪽 수정 없이 연계 기능을 만들 수 있다.
