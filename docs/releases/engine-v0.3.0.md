# everyric2 엔진 0.3.0

이 문서는 **자체 호스팅 운영자·기여자**를 대상으로 한다(크롬 확장 사용자 노트는
`docs/releases/chrome-v1.6.0.md` 참고 — 그쪽은 "서버 업데이트 필요"로만 적혀 있는
항목들의 실제 내용이 여기 있다). 이번 배치는 **정렬기 자체가 통째로 바뀌었고
라이선스 구성도 달라졌다** — 나중에 "그때 뭐가 어떻게 좋아졌나"를 되짚을 사람을
위해 근거를 남긴다.

기재 원칙: 리포에 실제로 남아 있는 근거(코드 docstring·커밋·조사 문서)만 인용한다.
추정치는 적지 않고, 확인 못 한 것은 "미측정"이라고 그대로 적는다.

---

## 1. 정렬 스택 전환

### 무엇에서 무엇으로

| | 구(舊) 스택 | 신(新) 스택(이번 배치, 기본값) |
|---|---|---|
| 정렬 음향모델 | `facebook/mms-1b-all`(CTC, wav2vec2 계열) | **OWSM-CTC v4 1B**(다국어 앵커) + **omniASR-CTC-300M**(다국어 앵커 겸 2패스 리파이너) |
| 보컬 분리 | htdemucs(demucs CLI) | **bs-polarformer-fp16**(BS-RoFormer + PoPE 극좌표 위치 임베딩) |
| 정렬 방식 | ko/ja 이중정렬 + star 토큰 + pron_data DP 근사 | 앵커 정렬 → (medium/heavy만) 2패스 리파이너로 음절 재정렬, 표기(hangul/kana/romaji/en/ipa) 실측 산출 |
| 라우팅 | 없음(단일 경로) | fast/medium/heavy 3단계 자동 승급 |

`everyric2/alignment/{owsm_engine,omniasr_engine,refine_window}.py`, `everyric2/audio/
polarformer_separator.py`, `everyric2/alignment/display_fixes.py`가 이식된 부품이고,
`everyric2/server/worker.py`의 `_run_new_stack_alignment`(및 그 하위 `_run_fast_stage`/
`_run_deep_stage`)가 서버 배선이다. 레거시 ko/ja 이중정렬·star 토큰·pron_data DP 근사·
캡션 앵커/스캐폴드는 새 경로에 배선하지 않았다 — 구스택 CTC 엔진의 특정 실패 모드에
맞춰진 장치라 새 앵커에는 전제가 안 맞는다. `EVERYRIC_ALIGNMENT_ENGINE=ctc` +
`EVERYRIC_AUDIO_SEPARATOR_BACKEND=htdemucs`를 **함께** 지정하면 구스택으로 되돌릴 수
있다(하나만 바꾸면 기동이 거부된다 — cross-field validator가 오조합을 막는다).

### 왜 바꿨나

**직접적 동기는 라이선스다.** `docs/research/2026-07-30-model-replacement/README.md`의
"현행 스택의 라이선스 실태" 표가 명시한 대로, 구 스택의 정렬 음향모델
`facebook/mms-1b-all`은 **CC-BY-NC-4.0(비상업)** — 상업 서비스에 그대로 쓸 수 없는
모델이었다("❌ 상업 부적합 — 교체 필요의 진원"이라고 그 문서가 직접 적었다). 구 분리기
htdemucs도 코드는 MIT지만 **가중치 라이선스가 불명확**했다(원저자 Alexandre Défossez가
GitHub 이슈 #327에서 "모델 가중치는 MIT가 아니고 과학적 목적으로만 제공한다"고 직접
답변 — §2 라이선스 표 참고).

### 라우팅 구조 (fast → medium/heavy 자동 승급)

`_run_new_stack_alignment`(worker.py) docstring 원문 기준:

1. **fast**(무분리 omniASR, `_run_fast_stage`) — 대부분의 곡이 여기서 끝난다(전곡 평균
   시간을 지키는 근거). **언어가 en이면 이 신호가 원리상 무력해 fast를 건너뛰고 곧장
   medium으로 진입한다.**
2. **medium/heavy**(분리 + 앵커 2패스, `_run_deep_stage`) — fast 결과의 라인 신뢰도
   중앙값이 임계 미만이면 승급한다. **en 외 언어는 fast에서 곧장 heavy로**(medium을
   건너뛰고 owsm 앵커) 가고, **en은 medium에서 시작**한다(omniasr 자기앵커 — 2패스
   리파이너가 이미 음절 단위라 owsm이 잡을 이유가 없다).
3. **en 전용 사후 heavy 승급** — medium 결과에 "stranded"(놓친 부분) 시그니처가 남으면
   heavy(owsm 앵커)로 다시 돌리되, 그 시그니처가 **줄어드는 경우에만** 채택한다(악화
   방향으로는 안 간다). 이때 medium이 이미 분리해 둔 스템을 재사용한다(분리를 두 번
   안 한다).

ja는 fast→heavy, en은 medium→heavy로 **진입 지점만 다르고 같은 사다리**를 오른다.
분리는 medium/heavy의 **필수 구성요소**다 — 없거나 실패하면 조용히 무분리로 물러서지
않고 명시적으로 실패한다(운영자 지시 — "조용한 구스택 폴백 금지"). 앵커 모델·분리기
자산·(2패스가 켜져 있을 때의) 리파이너가 없으면 해당 요청은 예외를 그대로 올린다.

---

## 2. 라이선스 표 (가장 중요)

원문 라이선스명·모델명은 번역하지 않고 그대로 적는다. 출처는
`docs/research/2026-07-30-model-replacement/final-weights-licenses.md`(2026-07-31
재검증판)과 `docs/research/2026-07-30-model-replacement/README.md`.

### 신 스택(이번 배치)

| 구성요소 | 모델 | 라이선스(표기) | 서버 상업 사용 | 클라이언트 배포 | 비고 |
|---|---|---|---|---|---|
| 정렬 앵커(다국어) | `facebook/omniASR-CTC-300M` | **Apache-2.0**(HF 태그 + GitHub LICENSE 파일 일치) | **가능** | **가능** | 6개 후보 중 **가장 깨끗한 판정**. 다만 훈련 코퍼스 전체 목록은 미공개(데이터 provenance 자체는 "미확인"으로 남김) |
| 정렬 앵커(다국어, heavy 전용) | `espnet/owsm_ctc_v4_1B` | **CC-BY-4.0**(모델 카드) | **불확실**(경영 판단 영역) | **불확실**(서버보다 위험) | 가중치 자체는 CC-BY-4.0으로 명시 허여됐으나, 훈련 데이터에 **NC/ND/LDC 코퍼스가 다수 포함**됨이 논문 부록(arXiv:2402.12654 App. A.1)에 원문으로 명시돼 있다 — "가중치가 훈련 데이터 제약을 물려받는가"는 법적 미확정 영역. 전작 OWSM v3.1은 이를 의식해 별도 저제약(LR) 모델을 냈으나 **v4-1B에는 그 계열이 없다**(2026-08-01 재검증 확인) |
| 보컬 분리 | bs-polarformer-fp16(ZFTurbo MSST v1.0.20 릴리스 자산, BS-RoFormer+PoPE) | **MIT**(저장소 라이선스 상속 — 2026-07-31 개정 확정) | **가능**(저작권 고지 + MIT 사본 포함) | **가능**(v1.0.20 자산 pin 권장) | 자가 학습·자가 배포 자산이라 저장소 LICENSE(MIT)를 상속한다고 판정(개정 근거는 문서 §1 참고). 훈련 데이터 층위 저작권은 통상 회색지대로 남음 |
| 2패스 리파이너 | omniASR-CTC 재사용(별도 가중치 없음) | omniASR과 동일(Apache-2.0) | 가능 | 가능 | 신규 가중치 조달 없음 |
| en 발음(CMU Pronouncing Dictionary) | 원본 데이터 리포 동봉(`everyric2/text/data/cmudict/cmudict.dict` + 원문 `LICENSE`) | **BSD 계열 허가 라이선스**(Carnegie Mellon University, 동봉 `LICENSE` 원문 직접 확인 — "Redistribution and use in source and binary forms, with or without modification, are permitted...") | **가능** | **가능**(저작권 고지 유지 조건) | **2026-08-04 해소.** 최초 이번 배치(3b990f3)에서 PyPI `cmudict` 패키지(GPL-3.0-or-later, `importlib.metadata`로 확인)를 신규 선언했었는데, 원 CMU 데이터 자체는 BSD 계열이고 GPL은 그 PyPI 패키지의 **래퍼 코드**만의 선택이었다 — 원본 데이터 파일을 리포에 직접 동봉하고 파서를 자체 구현해(`en_g2p.py::_load()`) GPL 의존 자체를 제거했다(엔진 0.3.0 태그 전, 운영자 지시). 대안(`pronouncing`은 결국 cmudict를 끌어옴, `nltk`/`g2p-en`은 런타임 데이터 다운로드가 필요해 오프라인 배포가 깨짐)을 검토한 뒤 데이터 동봉이 채택됐다. 교체 전후 CMU 사전 전체(126,052개 표제어) 대조 결과 **값 불일치 0건** — 기능 회귀 없음 |
| zh 발음(pypinyin) | `pypinyin`(PyPI) | **MIT**(설치 패키지 메타데이터 확인) | 가능 | 가능 | 이번 배치(3b990f3)에서 신규 선언 |

### 구 스택(비교 기준선, 참고용)

`docs/research/2026-07-30-model-replacement/README.md` "현행 스택의 라이선스 실태" 표
원문:

| 구성요소 | 현행 | 라이선스 | 상업 판정 |
|---|---|---|---|
| 보컬 분리 | htdemucs(demucs CLI) | 코드 MIT / **가중치 불명확**(이슈#327 무응답, 저장소 2025-01 아카이브) | ⚠️ 리스크 — 승인받지 않은 상태로 봐야 함 |
| 정렬 음향모델 | `facebook/mms-1b-all` | **CC-BY-NC-4.0**(모델카드 재확인) | ❌ 상업 부적합 — 교체 필요의 진원 |
| 일본어 정렬 | jonatasgrosman xlsr-53-japanese | Apache-2.0 | ✅ |
| f0(멜로디) | RMVPE(기본)/FCPE(폴백) | 추론 코드 MIT / RMVPE 가중치 출처 확인 필요 | △ 가중치 provenance 점검 필요(미해결) |

**요약**: 구 스택의 핵심 정렬 모델(MMS)이 명백한 비상업 라이선스였던 것이 이번 교체의
직접 동기이고, 신 스택은 그 문제를 omniASR(Apache-2.0, 가장 깨끗)로 해소했다. en 발음
경로에서 신규 도입됐던 GPL-3.0-or-later(cmudict 패키지)도 원본 데이터 동봉으로 제거해
엔진 0.3.0 태그 전에 해소했다(위 표 참고). 다만 heavy 깊이 전용인 OWSM은 완전한
청신호가 아니라 "경영 판단 영역"으로 남아 있다 — **"완전히 라이선스 프리해졌다"는
과장이고, "가장 위험했던 구성요소 하나(MMS)를 확실히 해소하고, 뒤늦게 섞여 들어온
GPL 하나도 잡았고, 나머지(OWSM)는 리스크를 낮췄다"가 정확한 서술이다.**

f0(멜로디, RMVPE/FCPE)는 이번 배치의 교체 대상이 아니다 — 가중치 provenance 확인이
여전히 미해결로 남아 있다는 사실만 참고로 옮겨 적는다.

---

## 3. 성능·품질 실측

리포에 남아 있는 인용 가능한 실측치만 적는다. 벤치 산출물(`benchmark/`)은 커밋 대상이
아니라 파일 자체는 참조하지 않는다.

- **분리 포함 조합의 효과**: `_run_deep_stage`(worker.py) docstring 원문 — "벤치가 이
  조합(bs-polarformer-fp16 분리 + owsm/omniasr 앵커)으로만 **+26.7pp**를 실측했다"
  (분리 없이 같은 앵커만 썼을 때 대비). 이 수치가 medium/heavy 깊이에서 분리를 **필수
  구성요소**로 강제한(생략 시 명시적 실패) 직접 근거다.
- **라우팅 구성 자체**: `scripts/bench_adapters/routed.py`의 `routed-2mode-lang` 구성이
  이번 3단계 라우팅(fast/medium/heavy)의 원 실험이다 — 그 스크립트가 산출한 상세 지표
  (SDR, 정렬 정확도 등급 등)는 이 문서에 옮기지 않는다(스크립트이지 결과 문서가
  아니라서 인용 근거가 약함) — **벤치 상세는 별도 문서 없음, 필요하면 그 스크립트를
  직접 재실행해 확인**.
- 그 밖의 세부 지표(전체 곡 평균 처리 시간, 붕괴 곡 비율 등)는 이번 조사에서 리포에
  커밋된 인용 가능 수치를 찾지 못했다 — **미측정**으로 남긴다.

---

## 4. 자체 호스팅 준비물 변화

`deploy/DEPLOY.md` 1절 "새 정렬 스택(owsm/omniasr 앵커 + 2패스, 기본값) 준비물"에 이미
정리돼 있다(요약):

1. **bs-polarformer-fp16 분리기 자산** — 체크포인트(`model_bs_polarformer_float16.ckpt`
   + `.yaml`) + MSST 벤더 소스(핀 커밋 `e247dfe4abc1f17c69dff719207fe045dc04413a`,
   정확히 일치해야 함). 서버가 자동으로 내려받지 않는다.
2. **`audio-separator` + `PoPE-pytorch` 파이썬 패키지** — `uv sync --extra separator`로
   설치.
3. **OWSM 전용 격리 venv**(`.venv-owsm`, 기본 경로 `<repo_root>/.venv-owsm`) — ESPnet이
   메인 venv의 torch/transformers 버전과 충돌해 별도 venv가 필요하다.

셋 다 없으면 fast 깊이(정상곡 대다수)는 그대로 동작하고, **medium/heavy로 승급하는
곡만 명시적으로 실패한다**(조용한 저하 없음, 운영자 지시). zh 발음(pypinyin)은
`uv sync`(extra 여부 무관, base 의존성)로 들어온다 — en 발음(CMU Pronouncing
Dictionary)은 이제 **의존성이 아니라 리포에 동봉된 데이터 파일**(`everyric2/text/data/
cmudict/`)이라 `git pull`만으로 함께 온다(별도 설치 단계 없음, §2 참고).

---

## 5. 이번 배치의 서버 변경 전부

`chrome-v1.5.5..HEAD` 범위(2026-07-28 이후 서버가 재배포 안 된 것으로 보이는 전체
누적분, 이 문서 작성 시점 기준)를 정리한다. 커밋 해시는 `git log`로 재확인 가능.

| 항목 | 무엇이 바뀌었나 |
|---|---|
| **정렬 스택 전환**(§1) | OWSM/omniASR 앵커 + bs-polarformer 분리 + 2패스 리파이너, 기본값 전환. 라우팅(fast/medium/heavy) 신설 |
| **진행 표시 정직화** | 실제로 하지 않는 작업(예: fast 경로에서 "보컬 분리")을 진행 단계로 잘못 표시하던 문제 수정, 깊이 승급 통지 선행 |
| **vocaro 매칭 개선** | 동명이곡 아티스트 판별(정확 일치 다건 시 쿼리 아티스트 토큰으로 분리, 역방향 포함·아티스트 토큰 포함 오매칭 기각), 롱/숏 버전 오채택 방지(힌트·헤딩 언어 교차 시 동의어 매칭), 부분열 매칭 오탐(커버 감지·버전 동의어를 토큰 경계로) |
| **쿼터(limits) API** | 요청별 다음 리셋 시각(`next_reset_at`), 커버 잇기(`link`)·정렬 업그레이드(`upgrade`) 한도를 생성(`generate`)에서 완전 분리(자체 카운터·한도) |
| **공지사항 다국어화** | `notices.translations` JSON 컬럼(additive) — 언어별 title/body, 기존 `title`/`body`는 기본/폴백 언어로 의미 유지 |
| **발음 의존성 수리** | `cmudict`(en)·`pypinyin`(zh) 선언 누락 — 서버에 우연히 안 깔려 있으면 en/zh 발음이 **아무 안내 없이 kana 단독 근사로 저하**되던 경로 봉합(3b990f3). 이후 en 쪽은 cmudict(GPL) 대신 CMU 원본 데이터 리포 동봉으로 재교체(§2) |
| **엔진 웜 캐시** | owsm/omniasr 앵커를 `EngineFactory.get_engine()` 안에서 웜 캐시(설정 키+락+`EVERYRIC_SERVER_WARM_MODELS` 게이트) — 상세는 §6 "알려진 한계" 참고 |
| **`sync_results.engine` 오기록 수리** | 신 스택으로 정렬해도 이 컬럼이 항상 `'ctc'`로 적히던 결함(`engine_version`과 모순) — 실제 사용 엔진(`get_engine_type()`)을 적도록 수정. **기존 행은 소급 수정 안 함** |
| **번역/발음 정확도 일괄** | en 곡 romaji=원문 철자 문제, 한글 카라오케 타이밍 소실(모라 불일치 근사 폴백), 구세대 kana 단독 표기 영구 동결 해제(빠진 표기만 보완), zh 곡 표기 게이트(순한자 라인의 ja 오표기 방지), IPA 역검사, language/engine_variant 분리(결함 #5) |
| **안정성/견고성** | 고아 잡 TTL 리퍼(하트비트 끊긴 잡 회수), 번역 요청당 총예산(무한 재시도 방지), 워커 전달용 오디오 잔재 정리(저작권 규약), 싱크 초기화 시 번역 레이어·오프셋 정리, 분리기 임시 파일 경합 제거 |
| **캡션 트랙 선택** | ASR 언어 힌트 폴백, 번역뿐이면 포기, 크레딧 오염 방어 |
| **버저닝/스냅샷** | 재처리 시 직전 1세대 스냅샷 보관 + 조회 API(`GET /api/sync/{video_id}/previous`) |
| **링크/코퍼스 성능** | `list_titled` 컬럼 프로젝션으로 이벤트루프 5~7초 블로킹 제거(실사용자 체감 지연) |

**별도 취급**: X-API-Key 인증 우회 차단(`be5f155`)은 release notes 자체가 "이전 배치"로
구분해 둔 건이다 — 이번 배치 목록에 안 넣는다. `scripts/verify_deploy.py`로 실측한 결과
지금 프로드는 api_key 자체를 안 켠 공개 배포라 이 결함 경로는 애초에 해당 없음(안전).

---

## 6. 알려진 한계

- **OWSM은 웜 캐시로 적재 시간이 안 줄어든다.** `OwsmEngine.align()`은 서브프로세스
  격리(별도 venv, ESPnet 의존성 충돌 회피)라 호출마다 `subprocess.run`으로 새 프로세스를
  띄우고, 무거운 모델 적재는 **그 서브프로세스 안에서 매번 새로** 일어난다. 웜 캐시는
  Python 래퍼 인스턴스(경로 문자열 하나)만 재사용할 뿐이라, 실측된 적재 비용(9.48초,
  MoRef 실측)은 이번 배치로 줄지 않는다. **omniASR은 인프로세스라 웜 캐시가 실제로
  유효**하다(재사용 시 4.25초가 0회로 줄어듦). OWSM의 적재 비용을 실제로 줄이려면
  상주 서브프로세스(요청마다 새로 안 띄우는 워커) 설계가 필요한데, 이번 배치 범위에는
  없다.
- **owsm-ctc-v4-1b 라이선스는 "경영 판단 영역"** — CC-BY-4.0 태그와 훈련 데이터
  NC/ND/LDC 혼입 사이 긴장관계가 법적으로 확정되지 않았다(§2).
- **zh 발음 검증 커버리지 미확보** — `scripts/verify_deploy.py`의 발음 의존성 확인
  항목(⑤)이 zh는 이 코퍼스에 실측된 실곡 video_id가 없어 SKIP으로 남아 있다.
- **새 정렬 스택 자산이 프로드에 실제로 준비됐는지 미확인** — §4의 자산 3종(체크포인트·
  파이썬 패키지·격리 venv)이 배포 대상 서버에 있는지는 이 문서 작성 시점에 직접 확인하지
  못했다 — 배포 전 반드시 서버에서 확인할 것(없으면 medium/heavy 곡이 전부 명시적으로
  실패한다).
- 그 밖에 리포에서 "미해결"로 명시된 항목(RMVPE 가중치 provenance 등)은 이번 배치의
  대상이 아니라 이 문서에서는 언급만 남긴다(§2 구 스택 표 참고).

---

## 7. 업그레이드 방법

기존 배포(구 스택 실행 중)에서 이 버전으로 올라올 때:

1. **DB 백업** — `notices.translations` 등 스키마 변경이 서버 **기동 시점**에 자동
   적용된다(alembic 없음, `init_db()`의 `PRAGMA table_info` 확인 후 `ALTER TABLE`,
   additive만·NOT NULL 추가 없음). 재기동 전에 `everyric2.db`를 복사해 둘 것
   (`deploy/DEPLOY.md` §7 "업데이트가 DB 스키마 변경을 포함할 때" 참고).
2. **자산 준비 확인** — §4의 자산 3종이 이미 있는지 먼저 확인한다. 없으면 medium/heavy
   승급 곡이 전부 실패하므로, 부족하면 `.env`에 `EVERYRIC_ALIGNMENT_ENGINE=ctc` +
   `EVERYRIC_AUDIO_SEPARATOR_BACKEND=htdemucs`를 함께 설정해 구스택으로 임시 운영 가능.
3. `git pull && uv sync --extra separator && sudo systemctl restart everyric2`
   (또는 유저 유닛 동등 명령).
4. **워커를 별도 프로세스/유닛으로 운영 중이면 서버 재기동 직후 워커도 반드시 함께
   올린다 — 이번 릴리스부터는 "먼저/나중" 순서 문제가 아니라 "둘 다 안 하면 잡이 아예
   안 돈다".** 원격 워커 claim 엔드포인트(`api/worker.py`의 `POST /claim`)는 원래부터
   `request.version != __version__`이면 409로 거부하는 버전 게이트를 갖고 있었지만,
   `__version__`이 그동안 릴리스마다 안 올라가고 계속 `"0.1.0"`으로 고정돼 있어서
   (코드 자체 주석이 그렇게 명시한다) **사실상 한 번도 작동한 적이 없었다.** 이번에
   버전을 0.1.0 → 0.3.0으로 올리면서(`pyproject.toml` + `everyric2/__init__.py` 동기)
   **이 게이트가 처음으로 실제로 작동하기 시작한다.** 즉 서버만 재기동하고 워커(구버전
   `__version__="0.1.0"` 그대로)를 안 올리면, 워커가 claim 요청마다 409로 거부당해
   **그 순간부터 잡이 전혀 처리되지 않는다** — 조용한 품질 저하가 아니라 명시적 정지라
   원인 파악은 쉽지만("워커 로그에 409/버전 불일치"), 모르고 있으면 "재기동했는데 갑자기
   생성이 멈췄다"로 보인다. 이건 결함이 아니라 **의도된 안전장치**다 — 이번 배치처럼
   정렬 스택 자체가 바뀐 릴리스에서 구버전 워커가 신버전 서버의 잡을 집어 조용히 다른
   (구스택) 결과를 내는 사고를 막는다.
5. `.venv/bin/python scripts/verify_deploy.py <서버URL>`로 확인 — 신규 라우터(`/api/
   limits`·`/api/notices`)·발음 의존성(en/zh 표기 채움)·vocaro 생존을 읽기 전용으로 점검한다.
   웜 캐시 재사용(`warm model reuse: omniasr`/`warm model reuse: owsm` 로그 문구)과
   `sync_results.engine` 컬럼 정확성은 이 스크립트가 못 잡는다 — 재기동 후 곡을 연속
   두 번 생성해 로그를 보거나 `SELECT id, engine, engine_version, created_at FROM
   sync_results ORDER BY created_at DESC LIMIT 5;`로 직접 확인한다.

**롤백 안전성**: additive-only 계약(신규 필드는 옵셔널, 컬럼은 nullable, 기존 필드
rename·삭제 없음)이라 구버전 코드로 되돌려도 DB 자체를 되돌릴 필요가 없다 — 구코드는
새 컬럼·새 응답 필드를 그냥 무시한다. `git checkout <이전 커밋/태그>` → **`uv sync
--extra separator`를 다시 돈다**(되돌아간 시점의 `uv.lock` 기준으로 의존성도 같이
되돌아간다). 되돌리는 지점에 따라 두 가지가 갈린다:
- **cmudict 데이터 동봉 커밋 이전, pypinyin 선언 이후**(3b990f3~해소 커밋 사이)로
  롤백하면, 그 시점의 `uv.lock`이 아직 PyPI `cmudict`(GPL-3.0-or-later)를 참조하므로
  **`uv sync`가 그 GPL 의존성을 다시 설치한다** — en 발음 자체는 정상 동작하지만
  §2에서 해소했던 GPL 의존이 되돌아온다는 뜻이다. 이 상태로 오래 운영하지 말 것.
- **3b990f3 이전**으로 더 되돌리면 pypinyin도 함께 빠져 en/zh 발음이 다시 kana 단독
  근사로 저하된다.
둘 다 롤백의 **의도된 부작용**이지 새 결함이 아니다 — 워커 순서 지켜 재기동 →
`verify_deploy.py`로 확인(신규 라우터가 다시 404 나오는 것은 정상). 상세
절차는 `deploy/DEPLOY.md` §10 "롤백" 참고.
