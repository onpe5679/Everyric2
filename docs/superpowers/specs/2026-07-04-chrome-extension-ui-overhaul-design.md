# Everyric2 Chrome 확장 — UI 전면 개편 + PiP 수리 설계

> 2026-07-04 / 대상: `everyric2-chrome/`

## Problem 1-Pager

### 배경 (Background)
YouTube/YouTube Music 위에 싱크 가사를 표시하는 MV3 확장. 사용자 피드백:
UI 상태가 나쁘고("개차반"), PiP로 가사를 따로 띄우는 기능이 동작하지 않음.
"UI를 싹 갈아엎어도 좋으니 실제로 사용자가 편하게 쓸 수 있게" 재구축 요청.

### 문제 (Problem)
1. **소스가 빌드 불가 상태.** `content.ts`/`background.ts`가 import하는 `src/lib/*` 6개 모듈
   (song-detector, lyrics-parser, sync-renderer, overlay, lrclib, settings)이 디스크/git 어디에도 없음.
   현재 사용 중인 확장은 과거 빌드 산출물 → 유지보수 불능.
2. **UI 결함.** 페이지 CSS와 충돌(Shadow DOM 미사용), 두 벌로 갈라진 CSS
   (`public/content.css` 765줄 vs `src/styles/content.css` 1391줄), 아이콘 플레이스홀더 □ 노출,
   추천 영상 위를 덮는 배치, 닫으면 되살릴 방법 없음.
3. **PiP 미동작.** manifest로 주입된 CSS는 Document PiP 창에 전파되지 않으므로
   PiP 창 내부가 무스타일로 렌더링됨(`.pip-mode` 클래스만 존재).
4. **로컬 서버 연동 버그.** `background.ts`가 `http://localhost:8000`을 fetch하지만
   `host_permissions`에 localhost가 없어 항상 실패 → 자체 싱크 DB를 한 번도 사용 못 함.

### 목표 (Goal)
- `npm run build`가 통과하고 dist를 로드하면 YouTube/YT Music에서 즉시 동작.
- 새 오버레이 UI: 상태(로딩/싱크/플레인/없음/생성 중/오류)가 명확하고,
  드래그·리사이즈·접기·닫기·재열기(툴바 아이콘)가 전부 동작.
- PiP 버튼 → 스타일이 적용된 가사 전용 창이 뜨고, 재생과 싱크되며, 닫으면 패널 복원.
- 로컬 Everyric 서버(기본 `http://localhost:8000`)에서 싱크 조회 + "싱크 생성" + job 진행률 표시.

### 비목표 (Non-goals)
- 번역/발음 표시(P2), Musixmatch 연동, 미니 자막 pill, Firefox 지원, Chrome Web Store 제출 준비.
- 백엔드(Python) 변경. 기존 API를 그대로 사용한다.

### 제약 (Constraints)
- MV3 + Vite + CRXJS 유지, 신규 런타임 의존성 없음(vanilla TS).
- Document PiP는 Chrome 116+ 전용 → 미지원 브라우저에서는 버튼 숨김.
- MV3 SW는 수시로 언로드됨 → 장기 폴링은 content script가 주도.

## 아키텍처

```
src/
  content.ts            엔트리: 내비게이션 감지, 상태 머신, 메시지 수신
  background.ts         메시지 허브(단발 fetch), action 클릭 토글, 외부 사이트 알림
  types.ts              공유 타입
  lib/
    song-detector.ts    mediaSession → YTM DOM → YT DOM/제목 파싱 폴백
    lyrics-parser.ts    LRC/plain 파서 (+ enhanced LRC 단어 타이밍)
    sync-engine.ts      rAF + timeupdate 하이브리드 tick, 이진 탐색, 전역 오프셋
    lrclib.ts           LRCLIB get/search 클라이언트 (duration 근접도 매칭)
    everyric-api.ts     로컬/원격 서버 클라이언트 (lookup/generate/job)
    settings.ts         chrome.storage.local 래퍼 + 변경 브로드캐스트
  ui/
    dom.ts              h() 엘리먼트 헬퍼
    overlay.ts          메인 패널 (Shadow DOM 안에 렌더)
    pip.ts              Document PiP 컨트롤러 (스타일 텍스트 주입)
public/
  overlay.css           단일 CSS 소스 (web_accessible_resources; 패널/PiP 공용)
```

- **Shadow DOM**: `#everyric-root` 호스트에 open shadow root. YouTube CSS와 완전 격리.
  CSS는 `fetch(chrome.runtime.getURL('overlay.css'))`로 텍스트를 받아 `<style>`로 주입
  → 같은 텍스트를 PiP 창 document에도 주입 (PiP 무스타일 문제의 근본 해결).
  manifest의 `content_scripts.css` 항목은 제거.
- **가사 데이터 흐름**: Everyric 서버(segments, 단어 타이밍 보존) → LRCLIB synced → LRCLIB plain.
  기존 코드처럼 segments를 LRC 문자열로 변환했다 재파싱하지 않고 `LyricLine[]`으로 직접 매핑.
- **싱크 생성**: 패널에서 가사 확보(LRCLIB plain 또는 사용자 붙여넣기) 후
  `POST /api/sync/generate` → content script가 2초 간격 `GET /api/job/{id}` 폴링(진행률 표시)
  → 완료 시 가사 재조회. SW 수명 문제를 피하기 위해 폴링 루프는 content 쪽.
- **재열기**: `chrome.action.onClicked` → 해당 탭에 `TOGGLE_OVERLAY` 메시지.

## UI 설계

패널(기본: 우측 상단, 마스트헤드 아래, 340×min(70vh) 다크 글래스):
- **헤더(드래그 핸들)**: ♪ 곡명 — 아티스트(말줄임) | [PiP] [⚙] [—접기] [×]
- **본문 상태**
  - 로딩: 스켈레톤 3줄
  - 싱크 가사: 지난 줄 dim, 현재 줄 강조(스케일+색), 단어 타이밍 있으면 카라오케 필,
    클릭-투-시크, 자동 중앙 스크롤. 수동 스크롤 시 4초 일시정지 + "↓ 현재 가사로" 칩.
  - plain 가사: 상단 배너 "타임싱크 없음" + [✨ 싱크 생성] 버튼
  - 없음: 검색어(제목/아티스트) 수정 후 재검색 + [가사 붙여넣기…] → textarea → 싱크 생성
  - 생성 중: 진행률 바(%) + "완료되면 자동으로 표시됩니다"
  - 오류: 메시지 + [다시 시도]
- **푸터**: 소스 배지(everyric/lrclib) | 오프셋 −0.1s/+0.1s/리셋 | 현재 오프셋 표시
- **설정(⚙ 토글 시트)**: 자동 검색, 폰트 크기 S/M/L, 테마 auto/dark/light, 서버 URL
- 위치/크기는 드래그·`resize: both`+ResizeObserver로 조정, `chrome.storage.local`에
  호스트별(youtube/music) 저장. 접기 = 헤더만 남김.

PiP 창(400×280 기본):
- 이전/현재/다음 3라인 뷰. 현재 줄 크게 중앙, 단어 카라오케 필 동일 적용.
- 이전/다음 줄 클릭 시 시크. 하단에 진행 바 + 곡명.
- 창 크기에 따라 폰트 스케일(cqw 단위). 열리면 패널은 "PiP에서 표시 중" 플레이스홀더,
  닫히면(pagehide) 패널 복원. 백그라운드 탭 rAF 스로틀 대응으로 timeupdate 이벤트 병행 구동.

## 설정 스키마 (신규)

```ts
interface Settings {
  autoSearch: boolean;      // true
  fontSize: 'small'|'medium'|'large';  // medium
  theme: 'auto'|'dark'|'light';        // auto
  serverUrl: string;        // 'http://localhost:8000'
  offsetSec: number;        // 0 (전역)
}
```
구 스키마의 useMusixmatch/debugMode/showWordTiming/showCharTiming/showMiniSubtitle/
overlayPosition/showTranslation/translationLanguage는 폐기(단어 하이라이트는 데이터가 있으면 항상 켬,
위치는 자유 드래그로 대체).

## manifest 변경

- `host_permissions` += `http://localhost:8000/*`, `http://127.0.0.1:8000/*`
- `web_accessible_resources` += `overlay.css`
- `content_scripts.css` 제거 (Shadow DOM 주입으로 대체)

## 오류 처리

- 서버 미기동: 2초 타임아웃 후 조용히 LRCLIB 폴백(현행 유지). 싱크 생성 버튼은
  서버 도달 불가 시 비활성 + 툴팁.
- 곡 감지 실패: 5회(1초 간격) 재시도 후 "노래를 인식하지 못했어요" + 수동 검색 UI.
- SPA 내비게이션: `yt-navigate-finish` + URL 폴링 병행, videoId 변경 시 전체 리셋.

## 검증

- `tsc --noEmit` + `vite build` 통과, dist에 content/background/overlay.css 존재 확인.
- 별도 code-reviewer 패스로 로직 결함 검토.
- 수동 확인 가이드(README)에 unpacked 로드 절차 기재.
