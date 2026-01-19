# Everyric2 Chrome Extension PRD

> Product Requirements Document v1.0
> Last Updated: 2025-01-19

---

## 1. 개요

### 1.1 제품 비전
YouTube/YouTube Music에서 **어떤 노래든** 싱크된 가사를 표시하는 Chrome 확장 프로그램.
기존 서비스가 커버하지 못하는 신곡, 인디, 커버곡도 AI 기반 싱크 생성으로 지원.

### 1.2 킬링 포인트
| 기존 서비스 | Everyric2 |
|------------|-----------|
| DB에 있으면 표시, 없으면 "없음" | DB에 없으면 **AI가 싱크 생성** |
| 신곡/인디/커버 = 포기 | 신곡/인디/커버 = **가능** |
| Plain 가사만 표시 | Plain → **싱크 가사로 변환** |

### 1.3 타겟 사용자
- YouTube Music 헤비 유저
- K-pop/J-pop 팬 (비영어권 가사)
- 언어 학습자
- 커버/인디 음악 청취자
- 영상 편집자 (AE 연동)

---

## 2. 아키텍처

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Chrome 확장 프로그램                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  [1단계] 곡 감지                                                     │
│  ├─ navigator.mediaSession.metadata (우선)                          │
│  ├─ YouTube Music DOM 파싱 (폴백)                                   │
│  └─ 비디오 제목 파싱 (최후)                                          │
│                                                                     │
│  [2단계] 싱크 가사 검색                                              │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  P1: Everyric 서버 (기존 타임스탬프 조회)                      │   │
│  │  P2: LRCLIB API (무료, 오픈)                                  │   │
│  │  P3: YouTube 자막 매핑                                        │   │
│  │  P4: Musixmatch 역공학 API (설정 on/off)                      │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  [3단계] 결과 처리                                                   │
│  ├─ 싱크 가사 있음 → 바로 표시                                       │
│  ├─ Plain 가사만 있음 → "싱크 생성" 버튼 표시                        │
│  └─ 아무것도 없음 → 사용자 입력 유도 + "싱크 생성" 버튼               │
│                                                                     │
│  [4단계] 싱크 생성 요청 (버튼 클릭 시)                                │
│  └─ 외부 웹사이트로 이동 (새 탭)                                     │
│      https://everyric.com/sync?v={videoId}&lyrics={encoded}         │
│                                                                     │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ (externally_connectable)
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Everyric 웹사이트                                 │
├─────────────────────────────────────────────────────────────────────┤
│  - YouTube URL + 가사 자동 입력됨                                    │
│  - "생성" 버튼 클릭 → 서버에서 처리                                   │
│  - 완료 시 확장으로 메시지 전송 (chrome.runtime.sendMessage)          │
│  - 결과: 타임스탬프만 반환 (오디오 다운로드 버튼 없음)                 │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Everyric 서버                                     │
├─────────────────────────────────────────────────────────────────────┤
│  처리:                                                               │
│  1. yt-dlp로 오디오 추출 (내부, 노출 X)                               │
│  2. MMS + Forced Alignment                                          │
│  3. 타임스탬프 생성 및 저장                                          │
│  4. 오디오 즉시 삭제                                                 │
│                                                                     │
│  저장:                                                               │
│  ✅ 타임스탬프: { song_id, timestamps[], lyrics_hash, created_at }   │
│  ❌ 가사 텍스트: 저장 안 함                                          │
│  ❌ 오디오 파일: 저장 안 함                                          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. 기능 명세

### 3.1 가사 표시 (Core)

| 기능 | 설명 | 우선순위 |
|------|------|----------|
| 싱크 가사 표시 | 현재 재생 위치에 맞춰 가사 하이라이트 | P0 |
| 자동 스크롤 | 현재 라인으로 자동 스크롤 | P0 |
| 클릭-투-시크 | 가사 클릭 시 해당 시점으로 이동 | P1 |
| 다크/라이트 테마 | YouTube 테마에 맞춤 | P1 |
| 폰트 크기 조절 | 사용자 설정 | P2 |
| 번역 표시 | 원문 + 번역 동시 표시 | P2 |
| 발음 표시 | 로마자/한글 발음 | P2 |

### 3.2 가사 소스

| 소스 | 타입 | 우선순위 | 법적 상태 |
|------|------|----------|----------|
| Everyric 서버 | Synced | P0 | ✅ 자체 |
| LRCLIB | Synced/Plain | P0 | ✅ 오픈 |
| YouTube 자막 매핑 | Synced | P1 | ✅ 공식 |
| Musixmatch | Synced | P1 | ⚠️ Gray area |
| 사용자 입력 | Plain | P0 | ✅ 사용자 책임 |

### 3.3 싱크 생성 요청

| 기능 | 설명 | 우선순위 |
|------|------|----------|
| "싱크 생성" 버튼 | Plain만 있을 때 표시 | P0 |
| 웹사이트 이동 | 새 탭에서 everyric.com 열기 | P0 |
| 자동 정보 전달 | video_id, lyrics, artist, title | P0 |
| 완료 알림 | 웹사이트 → 확장 메시지 | P1 |
| 자동 새로고침 | 완료 시 가사 자동 업데이트 | P1 |

### 3.4 설정

| 설정 | 기본값 | 설명 |
|------|--------|------|
| Musixmatch 사용 | OFF | Gray area라서 기본 꺼짐 |
| 자동 가사 검색 | ON | 페이지 로드 시 자동 검색 |
| 오버레이 위치 | 오른쪽 | 왼쪽/오른쪽/하단 |
| 폰트 크기 | 중간 | 작게/중간/크게 |
| 번역 언어 | 한국어 | 사용자 선택 |

---

## 4. UI/UX

### 4.1 오버레이 레이아웃

```
┌──────────────────────────────────────┐
│  🎵 [곡 제목] - [아티스트]           │
│  ────────────────────────────────── │
│                                      │
│  이전 가사 라인 (흐림)                │
│                                      │
│  ▶ 현재 가사 라인 (강조)             │
│                                      │
│  다음 가사 라인 (흐림)                │
│                                      │
│  ────────────────────────────────── │
│  [⚙️] [📋 복사] [🔗 LRCLIB]          │
└──────────────────────────────────────┘
```

### 4.2 싱크 없음 상태

```
┌──────────────────────────────────────┐
│  🎵 [곡 제목] - [아티스트]           │
│  ────────────────────────────────── │
│                                      │
│  😢 싱크된 가사가 없습니다            │
│                                      │
│  [✨ 싱크 가사 생성하기]              │
│                                      │
│  ~15초 소요, 다른 사용자도            │
│  이 노래의 싱크 가사를 볼 수 있게 됩니다│
│                                      │
│  ────────────────────────────────── │
│  [가사 직접 입력] [왜 없나요?]        │
└──────────────────────────────────────┘
```

### 4.3 생성 중 상태

```
┌──────────────────────────────────────┐
│  🎵 [곡 제목] - [아티스트]           │
│  ────────────────────────────────── │
│                                      │
│  ⏳ 싱크 가사 생성 중...              │
│                                      │
│  [■■■■■□□□□□] 50%                    │
│                                      │
│  YouTube 계속 시청하세요.             │
│  완료되면 자동으로 표시됩니다.         │
│                                      │
│  ────────────────────────────────── │
│  [웹사이트에서 확인]                  │
└──────────────────────────────────────┘
```

---

## 5. 기술 스펙

### 5.1 Manifest V3

```json
{
  "manifest_version": 3,
  "name": "Everyric - Synced Lyrics for YouTube",
  "version": "1.0.0",
  "description": "Display time-synced lyrics for any song on YouTube",
  
  "permissions": ["storage"],
  
  "host_permissions": [
    "https://lrclib.net/*",
    "https://api.everyric.com/*"
  ],
  
  "content_scripts": [{
    "matches": [
      "https://www.youtube.com/*",
      "https://music.youtube.com/*"
    ],
    "js": ["content.js"],
    "css": ["content.css"]
  }],
  
  "background": {
    "service_worker": "background.js",
    "type": "module"
  },
  
  "externally_connectable": {
    "matches": ["https://everyric.com/*"]
  }
}
```

### 5.2 곡 감지 로직

```typescript
interface SongInfo {
  title: string;
  artist: string | null;
  videoId: string;
  duration: number;
}

function detectSong(): SongInfo | null {
  // 1. Media Session API (가장 안정적)
  const metadata = navigator.mediaSession?.metadata;
  if (metadata?.title) {
    return {
      title: metadata.title,
      artist: metadata.artist || null,
      videoId: getVideoIdFromUrl(),
      duration: getVideoDuration()
    };
  }
  
  // 2. YouTube Music DOM
  if (location.host === 'music.youtube.com') {
    return parseYouTubeMusicDOM();
  }
  
  // 3. YouTube DOM / Title parsing
  return parseYouTubeDOM();
}
```

### 5.3 가사 싱크 렌더링

```typescript
function startLyricsSync(lyrics: LyricLine[]) {
  const video = document.querySelector('video');
  if (!video) return;
  
  let lastIndex = -1;
  
  const tick = () => {
    const currentTime = video.currentTime;
    const index = findCurrentLineIndex(lyrics, currentTime);
    
    if (index !== lastIndex) {
      lastIndex = index;
      highlightLine(index);
      scrollToLine(index);
    }
    
    requestAnimationFrame(tick);
  };
  
  requestAnimationFrame(tick);
}

function findCurrentLineIndex(lyrics: LyricLine[], time: number): number {
  // Binary search for efficiency
  let lo = 0, hi = lyrics.length - 1, result = -1;
  
  while (lo <= hi) {
    const mid = Math.floor((lo + hi) / 2);
    if (lyrics[mid].startTime <= time) {
      result = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  
  return result;
}
```

---

## 6. API 명세

### 6.1 Everyric Server API

#### GET /api/sync/{videoId}
기존 타임스탬프 조회

**Request:**
```
GET /api/sync/dQw4w9WgXcQ?lyrics_hash=abc123
```

**Response (있음):**
```json
{
  "found": true,
  "sync_id": "uuid-xxx",
  "timestamps": [
    { "start": 0.0, "end": 2.5 },
    { "start": 2.5, "end": 5.1 }
  ],
  "lyrics_source": "lrclib",
  "quality_score": 0.95,
  "created_at": "2025-01-19T..."
}
```

**Response (없음):**
```json
{
  "found": false,
  "plain_lyrics_available": true,
  "suggested_source": "lrclib"
}
```

#### POST /api/sync/generate
싱크 생성 요청 (웹사이트에서 호출)

**Request:**
```json
{
  "video_id": "dQw4w9WgXcQ",
  "lyrics": "Never gonna give you up\nNever gonna let you down",
  "lyrics_source": "user_input",
  "language": "en"
}
```

**Response:**
```json
{
  "job_id": "job-xxx",
  "status": "processing",
  "estimated_time": 15
}
```

#### GET /api/sync/job/{jobId}
생성 작업 상태 조회

**Response:**
```json
{
  "job_id": "job-xxx",
  "status": "completed",
  "sync_id": "uuid-xxx",
  "timestamps": [...]
}
```

---

## 7. Chrome Web Store 전략

### 7.1 리스팅 가이드라인

**✅ 사용할 문구:**
- "Display time-synced lyrics for YouTube"
- "Karaoke-style lyric highlighting"
- "Community-powered lyrics timing"
- "Returns timing metadata only"

**❌ 피할 문구:**
- "Download", "Extract", "Rip"
- "YouTube audio", "MP3"
- "yt-dlp", "youtube-dl"
- "Bypass", "Circumvent"

### 7.2 권한 최소화

| 권한 | 필요 여부 | 이유 |
|------|----------|------|
| `storage` | ✅ 필수 | 설정, 캐시 |
| `host_permissions` (lrclib, everyric) | ✅ 필수 | API 호출 |
| `tabs` | ❌ 불필요 | 사용 안 함 |
| `<all_urls>` | ❌ 불필요 | YouTube만 |
| `webRequest` | ❌ 불필요 | 사용 안 함 |

### 7.3 Privacy Policy 필수 항목

- 수집 데이터: video_id, 사용자 제공 가사 (처리 목적)
- 저장 데이터: 타임스탬프만 (가사/오디오 저장 안 함)
- 제3자 공유: 없음
- 데이터 보존: 타임스탬프 무기한, 처리용 데이터 즉시 삭제

---

## 8. 로드맵

### Phase 1: MVP (4주)
- [ ] 기본 가사 표시 (LRCLIB)
- [ ] YouTube/YouTube Music 지원
- [ ] 싱크 생성 버튼 + 웹사이트 연동
- [ ] 기본 오버레이 UI
- [ ] Chrome Web Store 제출

### Phase 2: 개선 (4주)
- [ ] Musixmatch 연동 (설정)
- [ ] 웹사이트 → 확장 완료 알림
- [ ] 번역/발음 표시
- [ ] 테마 커스터마이징
- [ ] 키보드 단축키

### Phase 3: 확장 (4주)
- [ ] Firefox/Edge 지원
- [ ] LRCLIB 기여 기능
- [ ] 품질 투표 시스템
- [ ] AE 플러그인 연동

---

## 9. 성공 지표

| 지표 | 목표 (3개월) | 목표 (6개월) |
|------|-------------|-------------|
| 설치 수 | 1,000 | 10,000 |
| DAU | 100 | 1,000 |
| 싱크 생성 수 | 500곡 | 5,000곡 |
| 서버 타임스탬프 DB | 1,000곡 | 10,000곡 |
| Chrome Store 평점 | 4.0+ | 4.5+ |

---

## 10. 리스크 및 대응

| 리스크 | 확률 | 영향 | 대응 |
|--------|------|------|------|
| Chrome Store 거부 | 중 | 높음 | 문구 수정 후 재제출, 필요시 웹사이트 분리 |
| YouTube DOM 변경 | 높음 | 중 | 다중 감지 전략, 빠른 업데이트 |
| LRCLIB 서비스 중단 | 낮음 | 중 | 자체 DB로 폴백 |
| 서버 비용 증가 | 중 | 중 | Rate limit, 후원 모델, 캐싱 강화 |
| 저작권 클레임 | 낮음 | 높음 | 타임스탬프만 저장, DMCA 대응 프로세스 |

---

## 부록: 참고 확장 프로그램

| 확장 | 특징 | 배울 점 |
|------|------|---------|
| Better Lyrics | 외부 링크, 다중 소스 | Provider 체인, UI |
| SponsorBlock | 타임스탬프 DB, 커뮤니티 | 서버 구조, 후원 모델 |
| Return YouTube Dislike | 외부 API 연동 | Chrome Store 승인 |
