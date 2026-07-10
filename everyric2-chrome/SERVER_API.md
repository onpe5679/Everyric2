# Everyric 확장 ↔ 서버 API 계약

확장 프로그램(v1.1)이 호출하는 서버 인터페이스. 이 계약만 충족하면
어떤 서버든(모션그래픽 플랫폼 서버 포함) 확장의 싱크/생성/번역 백엔드로 동작한다.

## 공통

- 인증: 확장 설정에 API 키가 입력돼 있으면 모든 요청에 `X-API-Key: <key>` 헤더 첨부.
  빈 값이면 헤더 생략(익명). 401/403 등 비-2xx는 확장에서 "실패"로 처리된다.
- CORS: 확장(service worker) 호출은 host_permissions로 CORS를 우회하므로 불필요.
  단, 확장 manifest의 `host_permissions`에 서버 origin이 등록돼 있어야 한다.
- 서버 URL은 확장 설정에서 변경 가능 (기본 `http://localhost:8000`).

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

## 참고 구현

이 저장소의 FastAPI 서버가 참고 구현이다 (큐 순번 필드만 미구현):
- `everyric2/server/api/sync.py` — 조회/생성 (video_id + lyrics_hash 멱등 처리 포함)
- `everyric2/server/api/job.py` — 진행 상태
- `everyric2/server/api/translate.py` — 번역
- UI만 확인하려면 `scripts/mock-server.mjs` (목업, :8000)
