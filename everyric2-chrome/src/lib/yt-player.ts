/**
 * 유튜브 재생 컨트롤 (다음/이전 곡 + 재생목록) — **유튜브 비공식 마크업 의존**.
 *
 * 유튜브는 페이지 스크립트에 공개 API를 노출하지 않는다(iframe API는 임베드 전용이고,
 * 워치 페이지의 `#movie_player`는 격리된 월드에 있어 콘텐츠 스크립트에서 잡을 수 없다).
 * 그래서 플레이어 컨트롤 버튼과 재생목록 패널 DOM을 직접 조작하는 것이 유일한 경로다 —
 * lib/song-detector.ts가 제목·영상 엘리먼트를 찾는 것과 같은 리스크 프로파일이다.
 *
 * 유튜브가 마크업을 개편하면 여기가 먼저 깨진다. 그래서:
 * - 셀렉터는 전부 아래 SELECTORS 상수 한곳에 모아 둔다 (개편 시 여기만 고치면 된다).
 * - 못 찾으면 **예외를 던지지 않고 조용히 false / 빈 배열**을 돌려준다.
 *   호출부(PiP 컨트롤)는 그 값으로 버튼을 비활성화하거나 숨긴다 — 창이 깨지지 않는다.
 */

/** 유튜브 비공식 마크업 의존 지점 — 개편에 대비해 후보를 여러 개 둔다 (앞에서부터 시도) */
const SELECTORS = {
  /** 워치 페이지 플레이어 컨트롤 + 유튜브 뮤직 플레이어 바 */
  next: ['.ytp-next-button', 'ytmusic-player-bar .next-button', 'ytmusic-player-bar tp-yt-paper-icon-button.next-button'],
  prev: ['.ytp-prev-button', 'ytmusic-player-bar .previous-button', 'ytmusic-player-bar tp-yt-paper-icon-button.previous-button'],
  /** 워치 페이지 우측 재생목록 패널 항목 + 뮤직 대기열 항목 */
  playlistItem: ['ytd-playlist-panel-video-renderer', 'ytmusic-player-queue-item'],
  /** 항목 안 제목 (워치 / 뮤직) */
  itemTitle: ['#video-title', '.song-title', 'yt-formatted-string.title'],
  /** 항목 안 부제(채널·아티스트) */
  itemByline: ['#byline', '.byline', '#video-info'],
  /** 항목을 실제로 여는 링크 — 없으면 항목 자체를 클릭한다 */
  itemLink: ['a#wc-endpoint', 'a#thumbnail', 'a'],
} as const;

export interface PlaylistEntry {
  /** getPlaylist()가 돌려준 배열에서의 위치 — playPlaylistItem에 그대로 넘긴다 */
  index: number;
  title: string;
  byline: string;
  /** 지금 재생 중인 항목 (유튜브가 selected 속성으로 표시) */
  selected: boolean;
  /** 항목 링크 href에서 뽑은 videoId — 서버 싱크 존재 조회(exists 배지)에 쓴다.
   *  셀렉터가 안 맞는 레이아웃(뮤직 대기열 등)이면 null(그 항목만 배지를 건너뛴다) */
  videoId: string | null;
}

/** 클릭해도 아무 일이 없을 버튼(비활성·숨김)을 걸러낸다 */
function isUsable(el: HTMLElement): boolean {
  if (el.getAttribute('aria-disabled') === 'true') return false;
  if (el.hasAttribute('disabled')) return false;
  // 유튜브는 다음 곡이 없으면 컨트롤을 숨긴다 — 렌더 박스가 없으면 눌러도 무의미
  return el.getClientRects().length > 0;
}

/** 후보 셀렉터를 앞에서부터 시도해 쓸 수 있는 첫 엘리먼트 (없으면 null) */
function findUsable(selectors: readonly string[]): HTMLElement | null {
  try {
    for (const sel of selectors) {
      for (const el of document.querySelectorAll<HTMLElement>(sel)) {
        if (isUsable(el)) return el;
      }
    }
  } catch {
    /* 셀렉터 자체가 무효한 환경 — 기능 없음으로 처리 */
  }
  return null;
}

/** 항목 안에서 첫 번째로 잡히는 텍스트 */
function textIn(root: Element, selectors: readonly string[]): string {
  for (const sel of selectors) {
    const t = root.querySelector(sel)?.textContent?.replace(/\s+/g, ' ').trim();
    if (t) return t;
  }
  return '';
}

export function hasNext(): boolean {
  return findUsable(SELECTORS.next) !== null;
}

export function hasPrevious(): boolean {
  return findUsable(SELECTORS.prev) !== null;
}

/** 다음 곡으로 넘기기 — 버튼을 못 찾으면 false (예외 없음) */
export function playNext(): boolean {
  const btn = findUsable(SELECTORS.next);
  if (!btn) return false;
  btn.click();
  return true;
}

/** 이전 곡으로 — 버튼을 못 찾으면 false (예외 없음) */
export function playPrevious(): boolean {
  const btn = findUsable(SELECTORS.prev);
  if (!btn) return false;
  btn.click();
  return true;
}

/** 항목 링크 href에서 videoId 추출 — itemLink 후보를 앞에서부터 시도한다(itemLink의
 *  마지막 후보 'a'는 링크가 아예 없는 구조에서도 잡힐 수 있어, href가 없거나 워치
 *  URL이 아니면 다음 후보로 넘어간다). */
function itemVideoId(node: Element): string | null {
  for (const sel of SELECTORS.itemLink) {
    const href = node.querySelector<HTMLAnchorElement>(sel)?.getAttribute('href');
    const m = href?.match(/[?&]v=([\w-]{11})/);
    if (m) return m[1];
  }
  return null;
}

/** 현재 페이지의 재생목록 항목들 — 재생목록이 없으면 빈 배열 */
export function getPlaylist(): PlaylistEntry[] {
  const nodes = playlistNodes();
  return nodes.map((node, index) => ({
    index,
    title: textIn(node, SELECTORS.itemTitle) || `${index + 1}번째 항목`,
    byline: textIn(node, SELECTORS.itemByline),
    selected: node.hasAttribute('selected'),
    videoId: itemVideoId(node),
  }));
}

/** 재생목록 N번째 항목으로 이동 — 항목이나 링크를 못 찾으면 false (예외 없음) */
export function playPlaylistItem(index: number): boolean {
  const node = playlistNodes()[index];
  if (!node) return false;
  for (const sel of SELECTORS.itemLink) {
    const link = node.querySelector<HTMLElement>(sel);
    if (link) {
      link.click();
      return true;
    }
  }
  // 링크가 없는 구조(뮤직 대기열 등)는 항목 자체가 클릭 타깃이다
  (node as HTMLElement).click();
  return true;
}

/** 재생목록 항목 노드 — 후보 셀렉터 중 실제로 항목이 잡히는 첫 종류를 쓴다 */
function playlistNodes(): HTMLElement[] {
  try {
    for (const sel of SELECTORS.playlistItem) {
      const found = Array.from(document.querySelectorAll<HTMLElement>(sel));
      if (found.length > 0) return found;
    }
  } catch {
    /* 셀렉터 무효 — 재생목록 없음으로 처리 */
  }
  return [];
}
