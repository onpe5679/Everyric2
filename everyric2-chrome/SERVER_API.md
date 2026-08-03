# Everyric 확장 ↔ 서버 API 계약

확장 프로그램(v1.1)이 호출하는 서버 인터페이스. 이 계약만 충족하면
어떤 서버든(모션그래픽 플랫폼 서버 포함) 확장의 싱크/생성/번역 백엔드로 동작한다.

## 공통

- 인증: 확장 설정에 API 키가 입력돼 있으면 모든 요청에 `X-API-Key: <key>` 헤더 첨부.
  빈 값이면 헤더 생략(익명). 401/403 등 비-2xx는 확장에서 "실패"로 처리된다.
- CORS: 확장(service worker) 호출은 호스트 권한으로 CORS를 우회하므로 불필요.
  단, 그 origin에 대한 호스트 권한이 **실제로 부여돼 있어야** 한다 — manifest의
  `host_permissions`(설치 시 부여)이거나 `optional_host_permissions` + 사용자 허용이어야
  한다. 로컬 서버는 후자다 (아래 "manifest 준비사항").
- 서버 URL은 확장 설정에서 변경 가능 (기본값은 `src/lib/settings.ts`의 `DEFAULT_SETTINGS.serverUrl`).

## 엔드포인트

### `GET /health` — 연결 확인 (타임아웃 1.5s)
2xx + JSON이면 OK. 응답 본문은 무엇이든 상관없음.
"싱크 생성"/"번역" 버튼 활성화 여부가 이걸로 결정된다.

### `GET /api/sync/{videoId}` — 싱크 조회 (타임아웃 2.5s)
확장의 **최우선 가사 소스**. 없으면 LRCLIB로 폴백한다.

```jsonc
// 있음
{
  "found": true,
  "timestamps": [
    {
      "text": "가사 한 줄",
      "start": 12.34,            // 초 (float)
      "end": 15.6,               // 선택 — 없으면 다음 줄 start 사용
      "words": [                 // 선택 — 있으면 단어 단위 카라오케 하이라이트
        { "word": "가사", "start": 12.34, "end": 12.9 }
      ],
      "notes": [                 // 선택 — 있으면 PiP에 가라오케 음정 바 표시
        { "midi": 62, "start": 12.34, "end": 13.1, "confidence": 0.9 }
      ],
      "pronunciation": "가사 한 줄의 한국어 발음",  // 선택 — 원문 밑에 발음 표기
      "pron_segments": [                            // 선택 — 발음 음절별 타이밍 (카라오케 스텝 필)
        { "text": "아", "start": 12.34, "end": 12.5, "resolved": true }
      ],
      "translation": "사람이 단 번역"               // 선택 — 있으면 기계번역 대신 사용
    }
  ]
}
// 없음
{ "found": false }
```
- 줄 순서는 무관 (확장이 start 기준 정렬)
- 빈 text 줄은 무시됨
- `notes`: 라인 구간의 멜로디를 반음(MIDI 번호) 단위로 양자화한 것. 이 저장소의
  FastAPI 서버는 FCPE 기반으로 자동 생성한다 (`EVERYRIC_MELODY_ENABLED`, 기본 켜짐).
  word 레벨에 `notes`를 넣어도 확장이 동일하게 표시한다.
- `pronunciation`/`translation`: 생성 요청의 `line_meta`로 들어온 값이 저장된 것.
  확장은 translation이 있으면 해당 곡에서 서버 기계번역을 호출하지 않는다.
- `pron_segments`: 이 저장소의 서버는 정렬된 글자 타이밍 + 한자 읽기(pykakasi) 모라 분해
  + 발음 문자열 DP 매칭으로 자동 산출한다 (전사 모델 추가 실행 없음). 매칭 품질이 낮은
  라인은 필드를 생략하며, 확장은 그 경우 라인 진행률 그라데이션으로 폴백한다.

### `POST /api/sync/generate` — 생성 요청 (타임아웃 15s)
**큐에 등록만 하고 즉시 반환**할 것. 처리 시간과 무관하게 이 응답은 빨라야 한다.

```jsonc
// 요청
{
  "video_id": "dQw4w9WgXcQ",
  "lyrics": "줄1\n줄2\n...",
  "language": null,
  "line_meta": [               // 선택 — 라인별 발음/사람 번역 (보카로 위키 가사로 생성 시)
    { "text": "줄1", "pronunciation": "발음1", "translation": "번역1" }
  ]
}
// 응답
{ "job_id": "uuid", "status": "queued" }        // 또는 "processing"
// 동일 (video_id, lyrics_hash)가 이미 있으면:
{ "job_id": "기존id", "status": "completed" }    // 확장이 즉시 재조회
```
- `line_meta`는 세그먼트에 **라인 텍스트 매칭**(공백 정규화)으로 병합돼 저장된다.
  이미 완성된 싱크에 대해 다시 호출해도 메타만 병합된다 (정렬 재사용).

### `GET /api/job/{jobId}` — 진행 상태 (타임아웃 4s, 확장이 2초 간격 폴링)
```jsonc
{
  "job_id": "uuid",
  "status": "queued" | "pending" | "processing" | "completed" | "failed",
  "progress": 0,               // 0~100
  "queue_position": 3,         // 선택 — 있으면 "대기열 3번째" 표시 (1 = 다음 차례)
  "queue_size": 7,             // 선택 — 있으면 "(총 7개)" 표시
  "error": null                // failed일 때 사용자에게 그대로 표시됨
}
```
- `completed` → 확장이 `GET /api/sync/{videoId}` 재조회
- `failed` → error 메시지 표시
- 그 외 → progress/대기열 표시하며 계속 폴링

### `POST /api/translate` — 번역 (타임아웃 120s)
```jsonc
// 요청
{ "text": "줄1\n줄2\n...", "source_lang": "auto", "target_lang": "ko" }
// 응답 — lines는 입력 줄과 1:1 인덱스 매핑이어야 한다
{ "lines": [ { "original": "줄1", "translation": "번역1", "pronunciation": null } ] }
```

## manifest 준비사항

새 서버 도메인 사용 시 `manifest.json`에 추가 후 리빌드:

```json
"host_permissions": [ "...", "https://your-server.example/*" ]
```

현재 `host_permissions`(설치 시 전원에게 부여)에 등록된 서버 호스트와 근거:
- `https://everyric.moref.co/*` — 확장 기본 서버(`src/lib/settings.ts`의 `DEFAULT_SETTINGS.serverUrl`).
  설정을 바꾸지 않은 모든 사용자가 첫 실행부터 이 호스트로 싱크 조회/생성/번역을 호출한다.

`optional_host_permissions`(필요한 사용자만 런타임에 허용)에 등록된 호스트:
- `http://localhost:8000/*`, `http://127.0.0.1:8000/*` — 이 저장소의 서버(`everyric2/server`)를
  직접 구동해 붙이는 개발/자체 호스팅용 진입점. 자체 호스팅하는 소수만 쓰므로 설치 시
  전원에게 부여하지 않고, 옵션 페이지(`src/options.html`)에서 사용자가 직접 허용한다.
  허용 전에 이 주소로 요청하면 `everyric2-chrome/src/lib/host-permissions.ts`의 가드가
  요청을 막고 `ApiFailureKind: 'permission'`으로 분류한다 — 화면은 "서버가 꺼져 있다"가
  아니라 권한 안내와 '권한 설정 열기' 버튼을 보여 준다.
  실제로 요청·허용되는 패턴은 `http://127.0.0.1:8000/*` 하나다(루프백은 `localhost` →
  `127.0.0.1`로 정규화해 보낸다 — Windows IPv6 선시도 지연 회피).

제3의 서버로 바꾸려면 위 예시처럼 `manifest.json`에 도메인을 추가하고 리빌드해야 한다.
로컬 서버라도 **포트가 8000이 아니면** 마찬가지다(선언되지 않은 패턴은 런타임에 허용할 수
없다). 옵션 페이지는 "선언된 패턴 목록의 허용 상태를 보여 주고 허용/철회한다"는 구조라,
나중에 사용자 지정 서버까지 다루려면 `LOCAL_SERVER_ORIGINS`를 사용자가 추가한 origin
목록으로 넓히면 된다.

## 2026-08 추가분 (신 정렬 스택 + 차기 확장)

전부 additive — 구버전 확장(1.5.5)은 모르는 필드/엔드포인트를 무시하면 그만이다.

### 조회 응답 추가 필드 (`GET /api/sync/{videoId}`)
- `engine_version` — 이 싱크를 만든 정렬 스택 식별자. **null/부재 = 스탬프 도입 전
  구세대**(확장은 이 경우 노란 "새 엔진으로 업그레이드" 버튼을 재생성 버튼 자리에 띄운다).
- `engine_variant` — 엔진 변형(MMS 강제 폴백 등, 없으면 null).
- `debug.routing` — `{route: "fast"|"medium"|"heavy", language, language_source:
  "label"|"script_census", ...}`. route가 확장 깊이 배지(1/2/3)의 재료다.
- `debug.alignment_text` — `"fast"`/`"medium"`/`"heavy-2pass"` 등 실제 깊이+2패스 여부.
- `adlib` — `[[start,end],...]` 가사가 주장하지 않은 가창 구간 후보(신 스택 전용).
- 세그 `pron`/`pron_segs`의 표기 키에 `en`(원문 음절)·`ipa`(정렬 타깃 IPA)가 추가될 수
  있다 — 확장은 아는 키만 골라 쓴다.

### `GET /api/sync/{videoId}/previous` — 직전 세대 스냅샷 (A/B 고스트 비교)
재처리로 덮어써지기 전 세대. 이력이 없으면 `{"found": false}`(404 아님). 응답:
`timestamps/language/quality_score/created_at/replaced_at/lyrics_hash/engine_variant/
engine_version`. 확장 디버그 패널의 「이전 세대와 비교」가 소비한다.

### `POST /api/sync/regenerate` 추가 필드
- `min_depth`: `"medium" | "heavy"`(선택) — 분석 깊이 하한. 서버가 라우팅 판정을
  건너뛰고 이 깊이에서 시작한다(확장 깊이 버튼). 같은 가사의 기존 싱크 조기 반환과
  캐시 재사용을 우회한다. 한도는 비force generate 한도와 동일.

### `POST /api/sync/feedback` — 정렬 품질 별점·오류 제보 (수집 전용)
```jsonc
{ "video_id": "...", "rating": 4,             // 1~5 필수
  "category": "timing",                        // 선택: timing|pronunciation|lyrics|other
  "comment": "후렴이 밀려요" }                  // 선택, ≤1000자
```
응답 `{"ok": true}`. 제출 시점의 최신 싱크 sync_id·engine_version이 서버에 함께 남는다.

## 참고 구현

이 저장소의 FastAPI 서버가 참고 구현이다 (큐 순번 필드만 미구현):
- `everyric2/server/api/sync.py` — 조회/생성 (video_id + lyrics_hash 멱등 처리 포함)
- `everyric2/server/api/job.py` — 진행 상태
- `everyric2/server/api/translate.py` — 번역
- UI만 확인하려면 `scripts/mock-server.mjs` (목업, :8000)
