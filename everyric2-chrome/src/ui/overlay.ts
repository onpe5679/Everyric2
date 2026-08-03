import type { DebugInfo, LyricLine, LyricsSource, PanelGeometry, SearchCandidate, ServerLogEntry, ServerStatus, Settings, SongInfo, SongKey, SongTempo, SourceAttribution, SyncDebugMeta, SyncListItem, SyncPreviousVersion } from '../types';
import type { MicSample } from '../lib/mic-pitch';
import { resolveScript } from '../lib/lang';
import { t } from '../lib/i18n';
import { needsHostPermission, serverUsable, statusLine, unknownStatus } from '../lib/server-status';
import { resolveTheme } from '../lib/theme';
import { buildDebugPanel } from './debug-panel';
import { h, icon, ICONS } from './dom';
import { buildLineEl, buildPronEl, collectFillTargets, setElFilled, updateFillTargets, type FillTarget } from './line-render';
import { PitchLaneRenderer } from './pitch-lane';
import {
  applyServerGate,
  buildContributionSheet,
  buildEmptyState,
  buildErrorState,
  buildGeneratingState,
  buildLoadingState,
  buildNoticesSheet,
  buildPlainLines,
  buildRatingPop,
  buildSearchSheet,
  buildServerStatusSlot,
  buildSettingsSheet,
  buildWrongLyricsConfirm,
  createGenerateButton,
  probeNotices,
  renderCandidateList,
  setListStatus,
  type PanelContext,
  type SettingRow,
  type SettingsSection,
} from './panels';

/**
 * 분석 깊이 아이콘 — 위로 올라가는 화살표의 샤프트를 가로 나눔선 count개가 가로지른다
 * (깊이 1~3 시각화, 운영자 지시 도안). count=0은 나눔선 없는 순수 화살표(구세대 업그레이드).
 */
function depthArrowIcon(count: number): string {
  const ys = [17, 13, 9].slice(0, count);
  const bars = ys.map(y => `<line x1="8" y1="${y}" x2="16" y2="${y}"/>`).join('');
  return '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" '
    + 'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    + `<path d="M12 21V5"/><polyline points="6 11 12 5 18 11"/>${bars}</svg>`;
}

/** 퀵 토글 줄 아이콘 — PiP 좌상단 미니 버튼(.ey-pip-mini)과 같은 13px 도안 계열 */
const MINI_LANE_SVG = '<svg viewBox="0 0 24 24" width="13" height="13" fill="currentColor"><path d="M3 5h18v2H3zm0 4h12v2H3zm0 4h18v2H3zm0 4h9v2H3z"/></svg>';
const MINI_CAPTION_SVG = '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"><rect x="2.5" y="5" width="19" height="14" rx="2"/><line x1="6" y1="14" x2="11" y2="14" stroke-linecap="round"/><line x1="13" y1="14" x2="18" y2="14" stroke-linecap="round"/></svg>';
/** 재생목록 패널 토글 — 목록 줄 세 개 + 재생 삼각형(대기열을 표시하는 관용 도안) */
const MINI_PLAYLIST_SVG = '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h12"/><path d="M3 12h12"/><path d="M3 18h8"/><path d="M17 14.5v5l4.5-2.5z" fill="currentColor" stroke="none"/></svg>';
/** 재생목록 패널 헤더의 이전/다음 버튼 — PiP 재생 컨트롤과 같은 삼각형 도안 계열 */
const PL_PREV_SVG = '<svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor"><path d="M19 5l-9 7 9 7V5zm-11 0h-3v14h3V5z"/></svg>';
const PL_NEXT_SVG = '<svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor"><path d="M5 5l9 7-9 7V5zm11 0h3v14h-3V5z"/></svg>';
/** 레인 배치 토글 — 왼쪽 열(좁은 세로 막대 + 본문) / 아래 띠(본문 + 가로 막대) */
const MINI_POS_LEFT_SVG = '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="4" width="18" height="16" rx="2"/><rect x="3" y="4" width="6" height="16" fill="currentColor" stroke="none" opacity="0.75"/></svg>';
const MINI_POS_BOTTOM_SVG = '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="4" width="18" height="16" rx="2"/><rect x="3" y="14" width="18" height="6" fill="currentColor" stroke="none" opacity="0.75"/></svg>';
/** 레인 배치 토글 — 부착(패널 밖 왼쪽에 따로 붙는 작은 상자 + 간격 + 본문 상자) */
const MINI_POS_ATTACHED_SVG = '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2"><rect x="1.5" y="5" width="5" height="14" rx="1" fill="currentColor" stroke="none" opacity="0.75"/><rect x="9" y="4" width="13.5" height="16" rx="2"/></svg>';
/** PiP 열 스왑 — 강조가 오른쪽에 있어 «레인이 오른쪽으로 갔다»를 그린다(LEFT의 거울) */
/** 레인 머리 — 멜로디(음표) / 메트로놈(추) */
const LANE_NOTE_SVG = '<svg viewBox="0 0 24 24" width="13" height="13" fill="currentColor"><path d="M12 3v10.55A4 4 0 1 0 14 17V7h4V3h-6z"/></svg>';
const LANE_METRO_SVG = '<svg viewBox="0 0 24 24" width="13" height="13" fill="currentColor"><path d="M9 2h6l4 19H5L9 2zm1.6 2L7.4 19h9.2l-1.2-6.1-2.7 2.7-1.4-1.4L15 10.4 13.9 4h-3.3z"/></svg>';
const MINI_POS_RIGHT_SVG = '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="4" width="18" height="16" rx="2"/><rect x="15" y="4" width="6" height="16" fill="currentColor" stroke="none" opacity="0.75"/></svg>';
/**
 * 헤더 공지·기여 버튼 도안 — ICONS(dom.ts)에 넣지 않은 이유는 그 집합이 패널·PiP가
 * 함께 쓰는 최소 공용 아이콘이기 때문이다. 이 둘은 메인 패널 헤더에만 있다.
 */
const HEADER_BELL_SVG = '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 8a6 6 0 1 0-12 0c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.7 21a2 2 0 0 1-3.4 0"/></svg>';
const HEADER_CONTRIB_SVG = '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 20V10"/><path d="M9 20V4"/><path d="M15 20v-7"/><path d="M21 20v-11"/></svg>';

/**
 * 다음 재생 영상 정보 — 제목만 있던 문자열 API의 확대판.
 * videoId가 있으면 썸네일 URL을 유도할 수 있어(thumbnail 생략 가능) content는 보통
 * id만 넘기면 된다. 어느 필드든 없으면 카드가 그 조각만 생략한다.
 */
export interface NextUpInfo {
  title: string;
  videoId?: string;
  /** 직접 지정할 썸네일 URL — 생략하면 videoId에서 유도한다 */
  thumbnail?: string;
  channel?: string;
}

/** 이어질 재생목록 한 줄 — content가 lib/yt-player.ts로 스크랩해 채운다 */
export interface PlaylistItem {
  title: string;
  videoId?: string;
  channel?: string;
  /** 재생목록 안 번호(0-base) — playPlaylistItem에 그대로 넘기는 인덱스라 렌더 순서와
   *  달라질 수 없다(항상 채워져 온다, content가 yt-player.PlaylistEntry.index를 그대로 싣는다) */
  index?: number;
  /** 지금 재생 중인 항목 (유튜브가 selected로 표시) */
  current?: boolean;
  /** 서버 싱크(링크 포함) 존재 여부 — undefined면 아직 모름(조회 전/실패), 배지를 안 그린다 */
  syncExists?: boolean;
}

/** 유튜브 썸네일 URL — mqdefault(320×180)는 카드 크기에 비해 넉넉하고 항상 존재한다 */
function thumbUrl(videoId: string): string {
  return `https://i.ytimg.com/vi/${encodeURIComponent(videoId)}/mqdefault.jpg`;
}

/**
 * 이 패널 인스턴스가 창을 쓰는 방식.
 *
 * floating = 유튜브 페이지 위에 떠 있는 기존 가사창(좌표·드래그·엣지 클램프·접기).
 * filled   = PiP 창을 통째로 채우는 인스턴스. 창이 곧 패널이라 기하 층이 통째로 죽는다.
 *
 * PiP를 «반쪽 패널로 다시 구현»하는 대신 같은 클래스를 두 번 세우기 위한 유일한 분기다 —
 * 화면 구현이 하나뿐이어야 두 창이 영원히 같은 모습을 유지한다.
 */
export type OverlayChrome = 'floating' | 'filled';

/**
 * 부착 패널(레인·재생목록)을 **패널 바깥의 남의 열**에 꽂아 달라는 요청 — filled 전용.
 *
 * floating에서는 이 둘이 `.ey-panel`의 Shadow DOM 형제로 살면서 화면 좌표로 떠 있다.
 * PiP 창에서는 그럴 수 없다: 창을 가로로 [레인][영상 열][가사창][재생목록]으로 나누는데,
 * host 하나의 Shadow DOM은 그 사이사이로 쪼개질 수 없기 때문이다(영상 열이 가운데 끼어
 * 있다). 그래서 filled에서만 이 두 조각을 PiP 문서의 light DOM 열에 직접 붙인다 —
 * 같은 CSS 전문이 PiP 문서 <head>에도 주입돼 있어 `.ey-attach-lane` 규칙과 `:root`
 * 변수가 그대로 먹고, 레인 캔버스의 색 소스는 여전히 `this.panel`이라 테마도 안 갈라진다.
 */
export interface OverlaySlots {
  /** 가라오케 레인 부착 패널이 들어갈 자리 (영상 왼쪽 열) */
  laneSlot?: Element;
  /** 재생목록 부착 패널이 들어갈 자리 (가사창 오른쪽 열) */
  playlistSlot?: Element;
  /** 열 구성(레인 표시 여부 등)이 바뀌었을 때 — 창 주인이 폭·접힘을 다시 계산한다.
   *  노트 없는 곡으로 넘어가면 레인이 스스로 꺼지는데, 그때 열이 공간을 반납해야 한다. */
  onColumnsChanged?: () => void;
}

export interface OverlayCallbacks {
  onSeek: (time: number) => void;
  /** attribution은 붙여넣기 경로에서 사용자가 적어 넣은 출처(선택) */
  onGenerate: (lyrics: string, attribution?: string) => void;
  onRetrySearch: (query?: { title: string; artist: string }) => void;
  onOffsetChange: (offsetSec: number) => void;
  onSettingsChange: (patch: Partial<Settings>) => void;
  onPipToggle: () => void;
  onGeometryChange: (geometry: PanelGeometry) => void;
  /** 수동 검색: 후보 리스트 요청 — 결과는 showSearchResults로 되돌아온다 */
  onCandidateSearch: (query: { title: string; artist: string }) => void;
  /** 후보 리스트에서 사용자가 직접 선택 */
  onPickCandidate: (candidate: SearchCandidate) => void;
  /** 다른 영상의 싱크에 연결 (inst·커버) — rate는 원곡 대비 배속(nightcore≈1.25) */
  onLinkSync: (sourceVideoId: string, offsetSec: number, rate: number) => void;
  /** 진행 중인 전사 잡 취소 (진행 칩 클릭) */
  onCancelGenerate: () => void;
  /** 현재 영상의 싱크 링크 해제 */
  onUnlinkSync: () => void;
  /** 서버 저장 싱크 목록 요청 — 결과는 showSyncList로 되돌아온다 */
  onRequestSyncList: () => void;
  /** 이 영상 싱크의 직전 세대 조회 — 디버그 패널 A/B 고스트 비교용. 이력 없으면 found=false */
  onLoadPreviousSync: () => Promise<SyncPreviousVersion | null>;
  /** 디버그 패널 "깊이·버전 비교" 버튼이 SYNC_VERSIONS를 직접 호출할 때 필요 — 콜백이
   *  아니라 값 자체가 필요해서 게터로 둔다 */
  getVideoId: () => string | null;
  /** 분석 깊이 올리기/구세대 업그레이드 — minDepth 없으면 일반 재생성(=신 스택 자동 라우팅) */
  onDepthUpgrade: (minDepth?: 'medium' | 'heavy') => void;
  /** 정렬 품질 별점(1~5) + 선택 오류 제보 전송 — 성공 여부를 돌려준다 */
  onSubmitFeedback: (rating: number, category?: string, comment?: string) => Promise<boolean>;
  /** 매칭 표시줄의 "이 가사가 아니에요" — 오매칭 제보 후 검색 시트를 연다 */
  onWrongLyrics: () => void;
  /** 재생목록 패널 — 이전/다음 곡 이동 (lib/yt-player.ts DOM 조작을 content가 대신한다) */
  onPlaylistPrev: () => void;
  onPlaylistNext: () => void;
  /** 재생목록 항목 클릭 — index는 content가 스크랩한 순서(yt-player.playPlaylistItem과
   *  같은 인덱스 체계)를 그대로 되돌려준다 */
  onPlaylistSelect: (index: number) => void;
  /** 다음 영상 카드 클릭(재생목록 부착 패널의 폴백 카드) — videoId가 있으면 그 영상으로,
   *  없으면(문자열만 받은 구버전 setNextUp 경로) content가 유튜브 다음 버튼을 눌러 이동한다 */
  onNextUpClick: (videoId?: string) => void;
  /** 저신뢰 경고 바의 "이 곡에서 다시 보지 않기" — 이 영상에서는 영구히 억제한다 */
  onWarnDismissSong: () => void;
  /** 설정 시트 ♻️ 범주 — 곡별로 끈 저신뢰 경고 억제를 전부 되살린다 */
  onResetWarnDismiss: () => void;
  /** 이 영상의 서버 싱크 전부 삭제(초기화) — 잘못 붙여넣은 가사에서 새로 시작 */
  onResetSync: () => void;
  /** 검색 시트에서 원래 보던 가사 화면으로 복귀 (실수로 검색을 연 경우 탈출구) */
  onCloseSearch: () => void;
  /** 서버 상태 다시 확인 (서버 오류 배너의 '다시 확인') */
  onRecheckServer: () => void;
  /** 로컬 서버 호스트 권한을 허용하는 확장 페이지 열기 (배너·설정의 '권한 설정 열기') */
  onOpenPermissions: () => void;
  /** 최근 서버 요청 로그 — 접이식 섹션을 펼칠 때만 호출된다 */
  loadServerLog: () => Promise<ServerLogEntry[]>;
  /** 확장 전체 초기화 — 설정 시트 ♻️ 범주의 2단계 확인(같은 버튼 재클릭) 뒤에만 불린다 */
  onFullReset: () => void;
  /** 마이크 음정 궤적 공급자 — 레인이 매 프레임 최근 샘플을 끌어온다. 검출이 꺼져 있으면
   *  빈 배열을 돌려준다. 예전엔 PiP 창에만 배선돼 있어 메인 패널 레인에는 궤적이 아예
   *  뜨지 않았는데, 두 창이 같은 인스턴스를 쓰는 지금은 배선도 한 곳이면 된다. */
  getMicSamples: () => MicSample[];
}

/** 'sheet' = 공지·기여처럼 본문 자리를 통째로 빌린 화면 — 가사가 없으므로 하이라이트·
 *  레인·오프셋 계열이 전부 스스로 꺼진다(각 메서드의 stateKind 가드) */
type StateKind = 'loading' | 'synced' | 'plain' | 'empty' | 'generating' | 'error' | 'pip' | 'search' | 'sheet';

/** confidence(CTC 확률 기하평균, 0~1)를 e표기 없이 10진수로 — 아주 작은 값도 첫 유효숫자까지 */
function fmtConf(v: number): string {
  if (!(v > 0)) return '0';
  if (v >= 0.001) return v.toFixed(3);
  const digits = Math.min(10, 1 - Math.floor(Math.log10(v)));
  return v.toFixed(digits);
}

/** 유튜브 URL 또는 순수 11자리 ID에서 videoId 추출 */
function parseVideoId(input: string): string | null {
  if (/^[\w-]{11}$/.test(input)) return input;
  const m = input.match(/(?:v=|youtu\.be\/|\/shorts\/|\/embed\/)([\w-]{11})/);
  return m ? m[1] : null;
}

const DEFAULT_WIDTH = 340;
const DEFAULT_HEIGHT = 480;
const EDGE_MARGIN = 8;
const USER_SCROLL_HOLD_MS = 4000;
/** 줄을 클릭했을 때 그 줄 안쪽으로 밀어 넣는 양 — 브라우저 시크가 요청 지점 이하로
 *  스냅해 한 줄 위가 활성화되는 것을 막는다. 사람이 못 느낄 만큼 작아야 한다. */
const SEEK_INTO_LINE_SEC = 0.05;
/**
 * 좌측 레인 열 폭의 하한 — 압축 렌더로도 못 읽는 폭이다.
 *
 * 상한(예전 LANE_WIDTH_MAX=480)은 없앴다. "레인이 가사창을 잡아먹는다"는 것은 사용자가
 * 손잡이를 끝까지 끌었을 때의 **선택**이지 막을 일이 아니고(운영자 지시: "이런 건 왜
 * 제한이 있는지 모르겠어 없애버려"), 남는 진짜 제약은 «가사 목록에 LANE_BODY_MIN은
 * 남겨야 한다»는 물리 한계 하나뿐이다 — 그건 clampLaneWidth의 roomCap이 이미 지킨다.
 */
const LANE_WIDTH_MIN = 140;
/** 레인이 아무리 넓어져도 가사 목록에 남겨 두는 최소 폭 */
const LANE_BODY_MIN = 160;
/** 디바이더 폭 (CSS .ey-lane-divider와 같은 값) — 2단 성립 여부 계산에 필요하다 */
const LANE_DIVIDER_W = 8;
/**
 * 2단(레인 | 가사)이 성립하는 최소 패널 폭 — 레인 하한 + 가사 하한 + 디바이더.
 * 이보다 좁으면 두 하한을 동시에 만족할 수 없어, 한쪽이 반드시 뭉개진다(실측: 280px
 * 패널에서 레인 140을 지키면 가사가 132px로 떨어져 일본어 한 줄이 넉 줄로 접힌다).
 * 그때는 사용자가 'left'를 골랐더라도 가로 띠로 되돌린다 — 둘 다 못 읽는 2단보다
 * 읽히는 1단이 낫고, 패널을 넓히면 즉시 2단으로 복귀한다(handlePanelResize).
 */
const LANE_TWO_COL_MIN = LANE_WIDTH_MIN + LANE_BODY_MIN + LANE_DIVIDER_W;
/** 부착(attached) 레인 패널의 폭 하한 — 상한은 «화면 밖으로 나가지 않는다»는 물리
 *  클램프뿐이다(예전 고정 상한 560px 제거, LANE_WIDTH_MIN 주석과 같은 근거) */
const ATTACH_WIDTH_MIN = 140;
/** 부착 패널과 메인 패널 사이 간격(px) */
const ATTACH_GAP = 8;
/** 재생목록 부착 패널의 고정 폭(px) — 레인 부착 패널과 달리 사용자 조절 손잡이를 두지
 *  않는다(운영자 요청 범위 밖 — 목록 폭은 제목 말줄임이 감당하므로 굳이 필요하지 않다) */
const PLAYLIST_PANEL_WIDTH = 260;
/** 퀵 토글 레인 배치 버튼의 순환 순서 — left → bottom → attached → left */
const LANE_POS_CYCLE: Record<Settings['mainLanePos'], Settings['mainLanePos']> = {
  left: 'bottom', bottom: 'attached', attached: 'left',
};

export class LyricsOverlay {
  private host: HTMLDivElement;
  private panel: HTMLDivElement;
  private header: HTMLDivElement;
  private songTitleEl: HTMLDivElement;
  private songArtistEl: HTMLDivElement;
  private body: HTMLDivElement;
  private footer: HTMLDivElement;
  private debugEl: HTMLDivElement;
  /** debugEl(텍스트) + «전체» 토글 버튼을 담는 줄 — debugEl 단독이 아니라 이 줄 전체가
   *  settings.debugInfo를 따라 켜지고 꺼진다 */
  private debugStrip: HTMLDivElement;
  private debugToggleBtn: HTMLButtonElement;
  /** 곡 전체 디버그 패널(원문 vs heard 전수 대비) — 토글로 열고 닫는다 */
  private debugPanelEl: HTMLDivElement;
  private debugPanelOpen = false;
  /** 곡 단위 정렬 진단(자막 스캐폴드 등) — content가 setDebugMeta로 밀어넣는다.
   *  아직 아무도 안 채우면 null로 남고, 패널은 곡 전체 요약 줄만 생략한 채 정상 동작한다 */
  private debugMeta: SyncDebugMeta | null = null;
  private banner: HTMLDivElement;
  private resumeChip: HTMLButtonElement;
  private genChip: HTMLDivElement;
  private genList: HTMLDivElement;
  private genListOpen = false;
  private genListItems: { title: string; state: string; isCurrent: boolean }[] = [];
  /** 알림 칩 — 커버 자동 연결 진행/결과, 붙여넣기 표기 필터 결과 등 한 줄 알림 */
  private noticeChip: HTMLDivElement;
  private noticeTimer = 0;
  private warnBar: HTMLDivElement;
  /** 경고 바가 지금 그리는 점수 — renderWarnBar가 접힘 토글마다 다시 읽는다(null=숨김) */
  private warnScore: number | null = null;
  /** "자세히" 펼침 상태 — 곡이 바뀌면 setQualityWarning이 접힘으로 되돌린다 */
  private warnExpanded = false;
  /** ×로 세션 한정 닫음 — setDebugMeta가 뒤늦게 도착한 깊이로 renderWarnBar를 다시
   *  불러도(경고→중립 전환) 사용자가 이미 닫은 배너를 되살리지 않기 위한 가드.
   *  setQualityWarning(새 곡·설정 변경)이 새 컨텍스트를 열 때만 되돌린다. */
  private warnClosedByUser = false;
  /** 미보유 언어 칩을 눌러 번역을 기다리는 동안의 표시(U3-b) — 칩 펄스만으론 눈에 잘
   *  안 띈다는 실보고로 추가. 라인 목록 바로 위(.ey-warn-bar와 같은 자리 규칙)에 둬서
   *  "빈 번역 줄"이 아니라 "준비 중"임을 알린다. .ey-tr-status 클래스를 재사용한다
   *  (이번 작업은 overlay.css 수정 권한이 없어 기존 클래스만 쓴다). */
  private translationPendingBar: HTMLDivElement;
  /** 서버 오류 배너 — body 밖에 있어 resetBody()로 지워지지 않는다.
   *  덕분에 어떤 화면(가사·검색·생성 중·오류)에서도 사유 한 줄이 반드시 보인다. */
  private serverBar: HTMLDivElement;
  private pipBtn: HTMLButtonElement;
  /** 이 영상의 서버 싱크를 통째로 지우고 새로 시작 — UI명은 "초기화"다(운영자 결정,
   *  2026-08-04). 와이어(SYNC_RESET/handleResetSync)는 원래부터 "reset" 계약이라 이름이
   *  이미 일치한다 — 바뀐 것은 이 버튼이 예전에 onRegenerate(다시 정렬만, 가사 보존)를
   *  불렀다가 지금은 onResetSync(삭제 후 재검색)를 부른다는 점이다. "재생성"이라는
   *  말 자체가 남아있는 진짜 재생성 경로는 depthBtn(깊이 올리기, onDepthUpgrade →
   *  handleRegenerate → REGENERATE_SYNC)뿐이다 — 그쪽은 가사를 지우지 않고 다시
   *  분석만 하므로 이름을 그대로 둔다. */
  private resetSyncBtn: HTMLButtonElement;
  private depthBtn: HTMLButtonElement;
  /** depthBtn 클릭 시 동작 — 상태(깊이/구세대/최대)에 따라 updateDepthButton이 바꾼다 */
  private depthAction: (() => void) | null = null;
  private feedbackBtn: HTMLButtonElement;
  private feedbackPop: HTMLDivElement;
  /** 공지 진입점 — probeNotices가 available=false를 주면 통째로 숨는다(구서버엔 없는 기능) */
  private noticesBtn: HTMLButtonElement;
  /** 공지 버튼 우상단 안 읽음 점 — 목록을 한 번 열면(onSeen) 꺼진다 */
  private noticesDot: HTMLSpanElement;
  private contribBtn: HTMLButtonElement;
  /** 보컬 글로우 현재 상태 — 매 tick classList 쓰기를 피하기 위한 캐시 */
  private vocalGlowOn = false;
  /** 마지막으로 받은 재생 시각·정지 여부 — 레인을 tick 밖에서 즉시 다시 그릴 때 쓴다 */
  private lastTime = 0;
  private lanePaused = false;
  /** [모듈] 레인이 지금 화면에 있는가 — 꺼져 있을 때 매 tick 캔버스를 만지지 않기 위한 게이트 */
  private laneShown = false;
  /** 다음 영상 카드의 현재 내용 — 재생목록 부착 패널의 폴백 카드(목록 없는 단일 영상
   *  페이지)가 쓴다. 메인 패널 하단 전용 카드는 2026-08-04 제거됐다(운영자 지시 —
   *  재생목록 모듈과 정보가 겹친다). */
  private nextUpInfo: NextUpInfo | null = null;
  /** 이어질 재생목록 (content가 setPlaylist로 밀어넣는다) — 비면 목록 자체를 안 그린다 */
  private playlistItems: PlaylistItem[] = [];
  private matchedBar: HTMLDivElement;
  private matchedTitleEl: HTMLSpanElement;
  /** "이 가사가 아니에요" 확인 자리 — 매칭 표시줄 바로 아래에 접혀 있다가 펼쳐진다.
   *  절대 배치 팝오버로 띄우지 않는 이유: 패널이 좁아 오른쪽으로 넘치면 확인 버튼이
   *  화면 밖으로 나간다(확인을 못 하면 취소도 못 한다). */
  private wrongLyricsPop: HTMLDivElement;
  /**
   * [모듈] 가라오케 레인 (설정 modMainLane) — PiP 창의 음정 레인을 메인 패널에도 띄운다.
   * 그리는 코드는 PiP와 **완전히 같은** PitchLaneRenderer 하나뿐이라 둘이 갈라질 수 없다.
   * 캔버스는 body 밖(푸터 위)에 두어 resetBody()의 화면 전환에 쓸려 나가지 않는다.
   */
  private laneCanvas: HTMLCanvasElement;
  private lane = new PitchLaneRenderer();
  /**
   * 레인 열 컨테이너 — 캔버스와 음절 타이밍 안내 배너를 함께 들고 다닌다.
   * mainLanePos에 따라 **부모가 바뀐다**: 'left'면 mainRow 안(가사 왼쪽 세로 열),
   * 'bottom'이면 패널 직속(가사 아래 가로 띠, 1.5.5까지의 배치). 엘리먼트를 두 벌
   * 만들지 않고 옮기기만 하는 이유는 캔버스가 하나여야 렌더러 attach가 유지되기 때문이다.
   */
  private laneWrap: HTMLDivElement;
  /**
   * 레인 열 머리 — **가라오케 전용 컨트롤이 사는 곳.**
   *
   * 운영자 원칙: «컨트롤은 자기가 제어하는 열에 붙어야 한다». 예전에는 멜로디·메트로놈이
   * PiP 중앙 열 푸터(재생 컨트롤 자리)에 섞여 있었고, 마디 창·계이름·카운트다운은 설정
   * 시트 깊숙이 있어 가라오케 중에 닿기 어려웠다. 레인 위에 붙이면 «지금 보고 있는 것»과
   * «그것을 조절하는 것»이 한자리에 온다.
   *
   * laneWrap 안에 두므로 배치(left/bottom/attached·PiP 열)를 따라 통째로 움직이고,
   * **두 표면이 같은 코드로 같은 컨트롤을 얻는다**(구현 단일화).
   */
  private laneHead: HTMLDivElement;
  private laneWindowLabel!: HTMLButtonElement; // buildLaneHead가 채운다(생성자에서 호출)
  private laneModeBtn!: HTMLButtonElement; // buildLaneHead가 채운다(생성자에서 호출)
  private laneMelodyBtn!: HTMLButtonElement; // buildLaneHead가 채운다(생성자에서 호출)
  private laneMetroBtn!: HTMLButtonElement; // buildLaneHead가 채운다(생성자에서 호출)
  private laneMetroRateBtn!: HTMLButtonElement; // buildLaneHead가 채운다(생성자에서 호출)
  private laneMetroBeatBtn!: HTMLButtonElement; // buildLaneHead가 채운다(생성자에서 호출)
  private laneSolfegeBtn!: HTMLButtonElement; // buildLaneHead가 채운다(생성자에서 호출)
  private laneCountBtn!: HTMLButtonElement; // buildLaneHead가 채운다(생성자에서 호출)
  /** 레인/가사 경계 드래그 손잡이 — 'left' 배치에서만 DOM에 붙는다 */
  private laneDivider: HTMLDivElement;
  /**
   * 부착(attached) 레인 패널 — mainLanePos==='attached'일 때 laneWrap이 여기로 옮겨
   * 온다. `this.panel`의 **형제**로 Shadow DOM에 직접 붙는다(패널 내부 분할이 아니라
   * 패널 밖에 독립적으로 뜨는 패널 — 운영자 요청 2026-08-03, "메인 가사창 왼쪽에
   * 따로 붙는 뷰"). position:fixed라 패널과 같은 좌표계를 쓴다.
   */
  private attachPanel: HTMLDivElement;
  /** 부착 패널 자체의 폭 조절 손잡이(왼쪽 모서리) — 내부 분할용 laneDivider와는 별개다 */
  private attachDivider: HTMLDivElement;
  /**
   * 재생목록 부착 패널 — 설정 modPlaylist가 켜지면 뜬다. attachPanel(레인)과 형제로
   * Shadow DOM에 직접 붙는 독립 패널이지만 **반대쪽**을 기본으로 삼는다: 레인이 왼쪽
   * 우선(폴백 오른쪽)인 것과 대칭으로, 이쪽은 오른쪽 우선(폴백 왼쪽) — updatePlaylistPlacement.
   */
  private attachPlaylistPanel: HTMLDivElement;
  private playlistHeaderEl: HTMLDivElement;
  private playlistStatusEl: HTMLSpanElement;
  private playlistPrevBtn: HTMLButtonElement;
  private playlistNextBtn: HTMLButtonElement;
  private playlistListEl: HTMLDivElement;
  /** 가사 목록 + (좌측 배치의) 레인 열을 담는 가로/세로 전환 컨테이너 */
  private mainRow: HTMLDivElement;
  /** 음절 타이밍 안내 배너 (fast/medium 깊이에서만) — 레인 영역 안에 산다 */
  private laneNotice: HTMLDivElement;
  /** '닫기'로 이번 세션만 접은 상태 — 설정(karaokeTimingNoticeDismissed)과 별개다 */
  private timingNoticeHidden = false;
  /** 모듈 퀵 토글 줄 — 설정 시트를 열지 않고 켜고 끄는 작은 버튼들 */
  private quickRow: HTMLDivElement;
  private quickLaneBtn: HTMLButtonElement;
  private quickLanePosBtn: HTMLButtonElement;
  private quickCaptionBtn: HTMLButtonElement;
  private quickPlaylistBtn: HTMLButtonElement;
  private collapseBtn: HTMLButtonElement;
  private settingsSheet: HTMLDivElement | null = null;
  private settingsDot: HTMLSpanElement | null = null;
  /** 설정 시트의 '권한 설정 열기' — 로컬 서버 권한이 없을 때만 보인다 */
  private settingsPermBtn: HTMLButtonElement | null = null;
  private sourceBadge: HTMLSpanElement;
  private offsetLabel: HTMLSpanElement;
  private progressBar: HTMLDivElement | null = null;
  private progressText: HTMLDivElement | null = null;

  private settings: Settings;
  private readonly callbacks: OverlayCallbacks;

  private stateKind: StateKind = 'loading';
  private lines: LyricLine[] = [];
  private lineEls: HTMLElement[] = [];
  private trStatusEl: HTMLSpanElement;
  private activeWordEls: FillTarget[] = [];
  private currentIndex = -1;
  /** 앞선 이 개수만큼의 줄이 '전부 채워진' 상태다 — fillUpTo가 유지하는 경계 */
  private filledUpTo = 0;
  private userScrollUntil = 0;
  private offsetSec: number;
  private visible = true;
  private fullscreenHidden = false;
  /**
   * 전체화면 동안 도착해 사용자가 볼 기회가 없던 알림 — 해제되는 순간 다시 띄운다.
   *
   * 전체화면에서는 브라우저가 전체화면 요소를 top layer에 그리므로, documentElement에 붙은
   * 이 호스트는 z-index와 무관하게 영상 뒤로 가려진다 — 그래서 handleFullscreenChange가
   * 아예 감춘다. 문제는 칩(진행·알림·서버 배너)도 그 안에 있다는 것이고, 특히 전사 완료
   * 경고는 20초 타이머로 떴다가 아무것도 안 보인 채 만료됐다.
   *
   * 칩만 전체화면 위로 올리려면 호스트를 전체화면 요소 서브트리로 옮겨야 하는데, 유튜브가
   * 자기 플레이어 DOM을 다시 그릴 때 우리 노드가 쓸려 나가는 위험을 안는다. 그 값을 치를
   * 만한 이득이 아니므로 **타이머를 미뤄** 해제 직후 처음부터 보여 주는 쪽을 택했다.
   * (PiP 창은 별도 최상위 창이라 페이지 전체화면과 무관하게 보인다 — content.showNotice가
   *  같은 알림을 그쪽에도 밀어넣으므로 PiP를 켠 사용자는 전체화면 중에도 바로 본다.)
   */
  private pendingNotice: { text: string; autoHideMs: number } | null = null;
  private serverStatus: ServerStatus = unknownStatus();
  private generateButtons: HTMLButtonElement[] = [];
  private plainTextForGenerate = '';
  private pipEnabled = false;
  private sourceUrl: string | null = null;
  private attributionName: string | null = null;
  /** attribution.sourceId — 가사 원문 출처가 vocaro인지 miraheze인지 구분한다(U2).
   *  source==='vocaro'는 두 위키를 모두 대표하는 값이라(설계 메모: adoptSourceResult
   *  참고) 이 값 없이는 배지가 항상 "보카로 가사 위키"로만 뜬다. */
  private attributionSourceId: string | null = null;
  /** setSourceBadge(source, synced)가 마지막으로 받은 인자 — setTranslationSource가
   *  독립적으로 배지를 다시 그려야 할 때(번역 출처가 나중에 붙을 때) 재사용한다. */
  private badgeSource: LyricsSource = 'lrclib';
  private badgeSynced = false;
  /** 사후 채택 번역(자막·위키·LLM)의 출처 — 가사 원문 출처(attribution)와 별개다(U2).
   *  null이면 배지에 병기하지 않는다. 곡이 바뀌면 resetBody가 지운다. */
  private translationSourceKind: 'caption' | 'wiki' | 'llm' | null = null;
  /** kind==='wiki'일 때만 의미가 있다 — 실제로 히트한 위키의 짧은 이름(미라헤즈/보카로),
   *  content가 sourceId를 이미 알고 있으므로 문자열로 확정해 넘긴다. */
  private translationSourceWikiName: string | null = null;
  private lastSong: SongInfo | null = null;
  private searchResultsEl: HTMLDivElement | null = null;
  private linkListEl: HTMLDivElement | null = null;
  private linkSrcInput: HTMLInputElement | null = null;
  private linkFilterInput: HTMLInputElement | null = null;
  private syncListItems: SyncListItem[] | null = null;
  /** 현재 표시 중인 싱크의 링크 상태 (없으면 null) — content가 setLinked로 밀어넣는다.
   *  verified는 반주 대조로 검증된 자동 링크 여부 (undefined = 구버전 서버로 알 수 없음) */
  private linkedInfo: {
    sourceVideoId: string; offsetSec: number; rate?: number; verified?: boolean;
  } | null = null;

  /** 제목바 언어 칩 줄 — 가사가 로드돼 있으면 항상 뜬다(setAvailableLangs로 채움). 가사
   *  자체가 없는 상태(로딩 중·빈 패널)에서만 숨긴다(setAvailableLangs(null)) */
  private langChipsRow: HTMLDivElement;
  private langChipButtons: { code: string; btn: HTMLButtonElement }[] = [];
  /** 이 곡의 "보유" 언어 — content가 setAvailableLangs로 밀어넣는다. content 쪽에서 이미
   *  서버 신호·세션 내 성공·곡 자신의 언어를 합쳐 계산해 넘기므로, 가사가 있는 한 항상
   *  최소 1개 이상(곡 언어)을 담은 배열이 온다. null은 "가사 자체가 없다"는 뜻으로만
   *  쓰인다 — "서버 신호 없음"과 혼동하지 않는다(그건 그냥 빈 배열에 가까운 폴백일 뿐
   *  숨김 사유가 아니다). */
  private availableLangs: string[] | null = null;
  /** 방금 클릭해 요청을 보낸 언어 — 응답 오기 전까지 그 칩만 로딩 표시(펄스) */
  private pendingLang: string | null = null;

  private geometry: PanelGeometry;
  private applyingGeometry = false;
  /**
   * applySettings가 유발한 재배치가 «사용자의 창 크기 조절»로 오인되지 않게 하는 빗장.
   *
   * 실제 사고: PiP에서 모듈을 토글하면 content가 applySettings를 **양쪽에** 방송하고,
   * 그 결과 메인 패널이 내부 재배치(레인 열 폭·재생목록 열)를 한다. 그 리플로우를
   * ResizeObserver가 잡아 handlePanelResize가 «패널 크기가 바뀌었다»로 읽고
   * geometry를 덮어쓴 뒤 **저장까지** 했다 — 사용자가 손대지도 않은 메인 창 크기가
   * PiP 토글 한 번에 초기화되는 정체다.
   *
   * 불변식: 모듈 on/off는 자기 표면의 열 공간만 바꾸고 **다른 표면의 기하는 건드리지
   * 않는다.** applyingGeometry와 같은 규약(다음 프레임에 스스로 풀림)으로 막는다.
   */
  private applyingSettings = false;
  private saveGeomTimer = 0;
  /** mountInto에서 «대상 문서의 창»으로 만든다 — unbindWindow가 끊으면 null */
  private resizeObserver: ResizeObserver | null = null;
  /**
   * 이 패널이 창을 쓰는 방식. floating은 유튜브 페이지 위에 떠 있어 기하·드래그·엣지
   * 클램프가 살아 있고, filled는 PiP 창을 통째로 채운다 — 창이 곧 패널이므로 기하 층
   * 전체가 의미를 잃는다(이식할 게 아니라 죽여야 하는 층이다).
   */
  private readonly chrome: OverlayChrome;
  /**
   * 마운트된 문서와 그 창 — mountInto 전에는 null이다. 전역 document/window를 그대로
   * 쓰면 PiP 인스턴스가 «유튜브 탭의» 창에 리스너·옵저버를 걸어 버리고, 그 탭이 숨겨져
   * 스로틀되는 순간 PiP 쪽 콜백이 멎는다(pitch-lane.ts의 setupResizeObserver가 같은
   * 이유로 canvas.ownerDocument.defaultView를 쓴다 — 그 규약을 패널에도 맞춘다).
   */
  private doc: Document | null = null;
  private win: Window | null = null;
  /**
   * 곡명을 뽑아 오는 «유튜브 페이지» 문서. 인스턴스가 둘이 되면 this.doc은 PiP 문서일
   * 수 있는데, PiP 문서의 title은 pip.ts가 t('pip.docTitle')로 덮어쓴 전혀 다른 값이라
   * 거기서 제목을 읽으면 조용히 엉뚱한 곡명이 나온다. 생성 시점의 것을 붙들어 둔다.
   */
  private readonly pageDoc = document;
  /** 2단계 확인 버튼의 되돌리기 타이머 — 예전엔 지역 클로저 변수라 destroy()가 원리적으로
   *  건드릴 수 없었다(누수 실측에서 코드상 지적으로 남긴 항목) */
  private confirmTimer = 0;
  /** applyGeometry가 거는 «다음 프레임에 플래그 되돌리기» 핸들. 콜백이 자기 필드만
   *  만져서 실해는 없지만, teardown 잔재가 0이어야 다음 회귀를 하네스가 잡아낸다 */
  private geomRaf = 0;
  /** applyingSettings를 «다음 프레임에» 되돌리는 핸들 — destroy가 반드시 걷는다 */
  private settingsRaf = 0;
  /** 두 번 눌러 확인이 «무장»된 버튼과 원래 툴팁 (confirmTwice) */
  private armedBtn: HTMLElement | null = null;
  private armedTitle = '';
  /** 메인 레인이 폭 부족으로 가로 띠로 접혔음을 이미 알렸는가 — 전이에서만 한 번 알린다 */
  private laneFoldedNotified = false;
  /** 폭 유도 글자 배율(widthFontScale)을 마지막으로 반영한 패널 폭 — 같은 폭이면 다시 걸지 않는다 */
  private lastFontWidth = -1;
  /** 레인에 **지금 실제로 적용된** 표시 구간(마디). 설정 왕복이 늦어도 휠 연속 조작이
   *  한 단계씩 움직이도록 화면 값을 직접 들고 있는다(attachLaneWheel 주석) */
  private laneWindow = 4;
  /** 부착 패널을 내보낸 바깥 열 (filled 전용, OverlaySlots) */
  private laneSlot: Element | null = null;
  private playlistSlot: Element | null = null;
  /** 열 구성이 바뀌었음을 창 주인에게 알린다 — 폭·접힘 재계산은 그쪽이 한다 */
  private onColumnsChanged: (() => void) | null = null;
  /** 가사 단축 표시(현재 줄 한 줄) — PiP 중앙 열이 빌려 가는 뷰. attachShortView가 만든다 */
  private shortEl: HTMLDivElement | null = null;
  private shortPrevEl: HTMLDivElement | null = null;
  private shortCurrentEl: HTMLDivElement | null = null;
  private shortNextEl: HTMLDivElement | null = null;
  /** 단축 표시의 현재 줄 안 채움 대상 — 매 프레임 updateFillTargets가 갱신한다 */
  private shortFillTargets: FillTarget[] = [];

  constructor(
    cssText: string,
    settings: Settings,
    callbacks: OverlayCallbacks,
    geometry: PanelGeometry | null,
    opts: { chrome?: OverlayChrome } = {},
  ) {
    this.chrome = opts.chrome ?? 'floating';
    this.settings = settings;
    this.callbacks = callbacks;
    this.offsetSec = settings.offsetSec;

    this.host = h('div', { attrs: { id: 'everyric-root' } });
    // floating은 «좌표를 가진 패널»을 띄우는 0크기 앵커고, filled는 창 전체가 패널이라
    // 호스트부터 창을 덮는다 — 이 한 줄이 기하 층의 존재 여부를 가른다
    this.host.style.cssText = this.chrome === 'filled'
      // PiP 창 안에서는 가로 행의 **한 열**로 산다(영상 열 오른쪽). fixed로 창을 덮으면
      // 레인·영상·재생목록을 전부 가려 버린다. 실제 폭은 pip.ts applyColumnLayout이
      // flex-basis로 매기고(열 디바이더로 조절), 높이는 행의 stretch가 채운다.
      // 패널 자신은 이 호스트 안에서 position:absolute라 크기를 한 픽셀도 보태지 않으므로
      // basis를 auto로 두면 «내용 0»이 되어 계산만 헷갈린다 — 0 + grow로 시작한다.
      ? 'all:initial;position:relative;display:block;flex:1 1 0;min-width:0;'
      : 'all:initial;position:fixed;top:0;left:0;width:0;height:0;z-index:2147483647;';
    const shadow = this.host.attachShadow({ mode: 'open' });

    const style = document.createElement('style');
    style.textContent = cssText;
    shadow.append(style);

    this.songTitleEl = h('div', { className: 'ey-song-title', text: t('overlay.detecting') });
    this.songArtistEl = h('div', { className: 'ey-song-artist' });

    this.pipBtn = this.headerButton(ICONS.pip, t('overlay.header.pip'), () => this.callbacks.onPipToggle());
    this.pipBtn.style.display = 'none';
    // 헤더가 아니라 풋터(별점 아이콘 오른쪽)에 붙는다 — 아래 this.footer 조립부에서 넣는다.
    this.resetSyncBtn = this.headerButton(ICONS.refresh, t('overlay.header.resetSync'), () => {
      if (this.confirmTwice(this.resetSyncBtn, t('overlay.header.resetSyncConfirm'))) {
        this.callbacks.onResetSync();
      }
    });
    // headerButton()은 헤더의 28px 아이콘 버튼 스타일(.ey-btn)을 준다 — 풋터의 작은
    // 아이콘 줄(★ 옆)에 맞게 .ey-reset-sync-btn으로 크기·색을 다시 정의한다(overlay.css).
    this.resetSyncBtn.classList.add('ey-reset-sync-btn');
    this.resetSyncBtn.style.display = 'none';
    // 분석 깊이 버튼 — 내용(아이콘·배지·툴팁·동작)은 updateDepthButton이 상태에 따라 채운다
    this.depthBtn = h('button', {
      className: 'ey-btn ey-depth-btn',
      attrs: { type: 'button' },
      on: { click: () => this.depthAction?.() },
    });
    this.depthBtn.style.display = 'none';
    const searchBtn = this.headerButton(ICONS.search, t('overlay.header.search'), () => this.openSearch());
    // 공지는 **응답 전까지 없는 셈 친다** — 구서버에서 잠깐 떴다 사라지는 버튼은
    // "고장난 기능"으로 읽힌다(probeNotices 규약: 실패=기능 없음).
    this.noticesBtn = this.headerButton(HEADER_BELL_SVG, t('overlay.header.notices'), () => this.openNotices());
    this.noticesBtn.classList.add('ey-notices-btn');
    this.noticesBtn.style.display = 'none';
    this.noticesDot = h('span', { className: 'ey-unread-dot' });
    this.noticesDot.style.display = 'none';
    this.noticesBtn.append(this.noticesDot);
    this.contribBtn = this.headerButton(HEADER_CONTRIB_SVG, t('overlay.header.contrib'), () => this.openContribution());
    const gearBtn = this.headerButton(ICONS.gear, t('overlay.header.settings'), () => this.toggleSettings());
    this.collapseBtn = this.headerButton(ICONS.collapse, t('overlay.header.collapse'), () => this.setCollapsed(!this.geometry.collapsed));
    const closeBtn = this.headerButton(ICONS.close, t('overlay.header.close'), () => this.setVisible(false));
    // 접기·닫기·PiP 열기는 «떠 있는 패널»에만 있는 개념이다. 창을 통째로 채우는 인스턴스에서
    // 접으면 빈 창만 남고, 닫으면 아무것도 없는 창이 남는다(창을 닫는 것은 창 자신의 몫이다).
    // 그래서 세 버튼은 filled에서 아예 내리고, 그 상태가 뒤집히지 않도록 setPipEnabled도
    // 아래에서 한 번 더 막는다.
    if (this.chrome === 'filled') {
      this.collapseBtn.style.display = 'none';
      closeBtn.style.display = 'none';
    }

    this.header = h('div', { className: 'ey-header' },
      h('div', { className: 'ey-header-left' },
        icon(ICONS.note),
        h('div', { className: 'ey-song' }, this.songTitleEl, this.songArtistEl),
      ),
      h('div', { className: 'ey-actions' },
        this.pipBtn, this.depthBtn, searchBtn,
        this.noticesBtn, this.contribBtn, gearBtn, this.collapseBtn, closeBtn),
    );

    // 제목바 언어 칩 — 이 곡에 어떤 언어가 준비돼 있는지 한눈에 보여주고 클릭 한 번으로
    // 전환한다. 라벨은 언어 자신의 이름으로 고정(한국어/ENG/日本語) — langSelect의 언어명과
    // 같은 관례로 uiLanguage로 번역하지 않는다(어느 표시 언어에서도 자기 언어를 바로 찾아야 함).
    // availableLangs가 없으면(구서버·아직 싱크 없음) 줄 전체를 숨긴다 — setAvailableLangs 참고.
    const LANG_CHIP_DEFS: [string, string][] = [['ko', '한국어'], ['en', 'ENG'], ['ja', '日本語']];
    this.langChipButtons = LANG_CHIP_DEFS.map(([code, label]) => {
      const btn = h('button', {
        className: 'ey-lang-chip',
        text: label,
        attrs: { type: 'button' },
        on: {
          click: () => {
            if (code === this.settings.translationLanguage) return;
            // showTranslation을 함께 켠다 — 꺼진 채로는 언어를 바꿔도 아무것도 안 보인다
            // (설정 시트의 langSelect는 이걸 안 해도 됐다 — 거기는 이미 번역 화면을 보는 중)
            this.callbacks.onSettingsChange({ translationLanguage: code, showTranslation: true });
          },
        },
      });
      return { code, btn };
    });
    this.langChipsRow = h('div', { className: 'ey-lang-chips' }, ...this.langChipButtons.map(c => c.btn));
    this.langChipsRow.style.display = 'none';

    this.banner = h('div', { className: 'ey-banner' });
    this.banner.style.display = 'none';

    // 전사 진행 칩 — 패널을 점유하지 않고 헤더 밑에 작게 진행률만 보여준다
    this.genChip = h('div', { className: 'ey-gen-chip' }, icon(ICONS.sparkle), '');
    this.genChip.style.display = 'none';

    // 칩 클릭 시 펼쳐지는 내 생성 대기열 목록 — 이 브라우저에서 시킨 잡만 저장돼
    // 있으므로(activeJobs) 다른 사용자의 큐는 구조적으로 보이지 않는다
    this.genList = h('div', { className: 'ey-gen-list' });
    this.genList.style.display = 'none';
    this.genChip.style.cursor = 'pointer';
    this.genChip.title = t('overlay.genChip.title');
    this.genChip.addEventListener('click', () => {
      this.genListOpen = !this.genListOpen;
      this.renderGenList();
    });

    // 알림 칩 — 전사 진행 칩과 같은 모양·같은 자리 규약을 쓰되 **별개 엘리먼트**다.
    // (전사 진행과 자동 연결 확인은 동시에 일어날 수 있어 한 칩을 공유하면 서로를 지운다)
    this.noticeChip = h('div', { className: 'ey-gen-chip ey-notice-chip' });
    this.noticeChip.style.display = 'none';

    // 낮은 정렬 신뢰도 경고 바 — X로 닫을 수 있고 설정에서 아예 끌 수 있다
    this.warnBar = h('div', { className: 'ey-warn-bar' });
    this.warnBar.style.display = 'none';

    // 번역 대기 표시(U3-b) — .ey-tr-status를 재사용하되 라인 목록 바로 위에 단독으로
    // 둔다(원래는 푸터 안 flex row 전용 클래스라 여백을 직접 보정한다)
    this.translationPendingBar = h('div', { className: 'ey-tr-status' });
    this.translationPendingBar.style.display = 'none';
    this.translationPendingBar.style.padding = '4px 14px 6px';
    this.translationPendingBar.style.whiteSpace = 'normal';
    this.translationPendingBar.style.flex = 'none';
    this.translationPendingBar.style.margin = '0';

    // 서버 오류 배너 — 상태가 정상/미확인이면 비어 있고, 아니면 사유+복구 동작이 들어간다
    this.serverBar = h('div', { className: 'ey-server-bar-slot' });
    this.serverBar.style.display = 'none';

    this.body = h('div', {
      className: 'ey-body',
      on: {
        wheel: () => this.markUserScroll(),
        touchmove: () => this.markUserScroll(),
        pointerdown: () => this.markUserScroll(),
      },
    });

    this.resumeChip = h('button', {
      className: 'ey-resume-chip',
      on: { click: () => this.resumeAutoScroll() },
    }, icon(ICONS.down), t('overlay.resumeChip'));
    this.resumeChip.style.display = 'none';

    this.sourceBadge = h('span', {
      className: 'ey-source',
      on: {
        click: () => {
          if (this.sourceUrl) window.open(this.sourceUrl, '_blank', 'noopener');
        },
      },
    });
    this.trStatusEl = h('span', { className: 'ey-tr-status' });
    this.offsetLabel = h('span', { className: 'ey-offset-value', text: '0.0s' });
    // 별점·오류 제보 — everyric 싱크에서만 보인다 (showSyncedLyrics에서 표시 결정)
    this.feedbackBtn = h('button', {
      className: 'ey-feedback-btn',
      text: '★',
      title: t('overlay.feedback.title'),
      attrs: { type: 'button' },
      on: { click: () => this.toggleFeedbackPop() },
    });
    this.feedbackBtn.style.display = 'none';
    this.feedbackPop = h('div', { className: 'ey-feedback-pop' });
    this.feedbackPop.style.display = 'none';
    this.footer = h('div', { className: 'ey-footer' },
      this.sourceBadge,
      this.feedbackBtn,
      // 별점(feedbackBtn) 바로 오른쪽 — 운영자 지시(2026-08-04), 헤더에서 옮겨왔다
      this.resetSyncBtn,
      this.feedbackPop,
      this.trStatusEl,
      h('div', { className: 'ey-offset' },
        h('span', { className: 'ey-offset-caption', text: t('overlay.footer.syncCaption') }),
        this.footerButton('−0.1', t('overlay.footer.pullEarlier'), () => this.changeOffset(-0.1)),
        this.offsetLabel,
        this.footerButton('+0.1', t('overlay.footer.pushLater'), () => this.changeOffset(0.1)),
        this.footerButton(t('overlay.footer.resetLabel'), t('overlay.footer.resetTitle'), () => this.changeOffset(null)),
      ),
    );
    this.footer.style.display = 'none';

    this.debugEl = h('div', { className: 'ey-debug', text: t('overlay.debug.waiting') });

    // «전체» — 곡 전체 디버그 패널(원문 vs heard 전수 대비) 토글. debugStrip과 함께
    // settings.debugInfo에 따라서만 보이고 숨는다(라인 유무는 패널 안에서 안내)
    this.debugToggleBtn = h('button', {
      className: 'ey-debug-toggle',
      text: t('overlay.debug.toggleAll'),
      title: t('overlay.debug.toggleAllTitle'),
      attrs: { type: 'button' },
      on: { click: () => this.toggleDebugPanel() },
    });
    this.debugStrip = h('div', { className: 'ey-debug-strip' }, this.debugEl, this.debugToggleBtn);
    this.debugStrip.style.display = 'none';

    this.debugPanelEl = h('div', { className: 'ey-debug-panel-wrap' });
    this.debugPanelEl.style.display = 'none';

    // 자동 매칭 표시줄 — 위키가 고른 곡 제목을 가사 위에 명시하고, 오매칭이면 그 자리에서
    // 제보(기존 피드백 시스템)할 수 있게 한다(운영자 요청 2026-08-03: ダミーロマンス가
    // 다른 곡에 붙었는데 화면만으로는 무엇에 매칭됐는지 알 수 없었다).
    this.matchedTitleEl = h('span', { className: 'ey-matched-title' });
    this.matchedBar = h('div', { className: 'ey-matched-bar' },
      h('span', { className: 'ey-matched-label', text: t('overlay.matched.label') }),
      this.matchedTitleEl,
      h('button', {
        className: 'ey-matched-report',
        text: t('overlay.matched.notThis'),
        title: t('overlay.matched.notThisTitle'),
        attrs: { type: 'button' },
        // 제보는 되돌릴 수 없는데 이 버튼은 가사 바로 위에 상시 떠 있어 오클릭 거리가
        // 가장 짧다 — 확인을 한 단계 세운다(buildWrongLyricsConfirm 주석 참고)
        on: { click: () => this.toggleWrongLyricsConfirm() },
      }),
    );
    this.matchedBar.style.display = 'none';
    this.wrongLyricsPop = h('div', { className: 'ey-confirm-slot' });
    this.wrongLyricsPop.style.display = 'none';

    // [모듈] 가라오케 레인 (설정 modMainLane) — 기본 꺼짐, applySettings가 표시를 정한다
    this.laneCanvas = h('canvas', {
      className: 'ey-main-lane',
      title: t('overlay.mainLane.seekTitle'),
      on: { click: e => this.seekFromLane(e) },
    });
    // 휠 확대축소·팬 — h()의 on:이 아니라 직접 건다(preventDefault를 하려면 passive:false가
    // 필요하고, 휠은 크롬에서 기본이 passive다)
    this.attachLaneWheel();
    this.laneNotice = h('div', { className: 'ey-lane-notice' });
    this.laneNotice.style.display = 'none';
    this.laneHead = this.buildLaneHead();
    this.laneWrap = h('div', { className: 'ey-lane-wrap' },
      this.laneHead, this.laneCanvas, this.laneNotice);
    this.laneWrap.style.display = 'none';
    this.laneDivider = this.buildLaneDivider();
    this.attachDivider = this.buildAttachDivider();
    // 부착 패널 — 시작할 때는 attachDivider만 자식으로 둔다. laneWrap은 applyLanePlacement가
    // mainLanePos==='attached'일 때만 reparent해 넣는다(항상 만들어 두되 필요할 때만 채운다).
    this.attachPanel = h('div', { className: 'ey-attach-lane' }, this.attachDivider);
    this.attachPanel.style.display = 'none';

    // 재생목록 부착 패널 (설정 modPlaylist) — 헤더(이전/다음 + 상태 문구) + 스크롤 목록.
    // 레인 부착 패널과 마찬가지로 처음부터 만들어 두고 표시만 토글한다(캔버스는 없으므로
    // 재생성 비용 문제는 없지만, 같은 구조를 유지해 두 부착 패널의 코드 계열을 맞춘다).
    this.playlistPrevBtn = h('button', {
      className: 'ey-pl-nav-btn', title: t('overlay.playlist.prev'), attrs: { type: 'button' },
      on: { click: () => this.callbacks.onPlaylistPrev() },
    }, icon(PL_PREV_SVG));
    this.playlistNextBtn = h('button', {
      className: 'ey-pl-nav-btn', title: t('overlay.playlist.next'), attrs: { type: 'button' },
      on: { click: () => this.callbacks.onPlaylistNext() },
    }, icon(PL_NEXT_SVG));
    this.playlistStatusEl = h('span', { className: 'ey-pl-status' });
    this.playlistHeaderEl = h('div', { className: 'ey-pl-header' },
      this.playlistPrevBtn, this.playlistStatusEl, this.playlistNextBtn);
    this.playlistListEl = h('div', { className: 'ey-pl-list' });
    this.attachPlaylistPanel = h('div', { className: 'ey-attach-playlist' }, this.playlistHeaderEl, this.playlistListEl);
    this.attachPlaylistPanel.style.display = 'none';

    // 가사 목록 컨테이너 — 'left' 배치에서 레인 열과 가로로 나란히 서는 자리다.
    // 'bottom'이면 세로 컨테이너 하나에 body만 들어 있어 예전 레이아웃과 같다.
    this.mainRow = h('div', { className: 'ey-main-row' }, this.body);

    // 모듈 퀵 토글 — 설정 시트를 열어야만 모듈을 켜고 끌 수 있던 불편(운영자 실보고)에
    // 대한 답이다. 헤더(.ey-actions)는 이미 버튼 7개로 꽉 차서 제목을 밀어내므로,
    // 제목 바로 아래 언어 칩 줄과 같은 계열의 얇은 줄을 따로 둔다.
    this.quickLaneBtn = this.miniButton(MINI_LANE_SVG, t('overlay.quick.lane'),
      () => this.callbacks.onSettingsChange(this.chrome === 'filled'
        ? { pitchGuide: !this.settings.pitchGuide }
        : { modMainLane: !this.settings.modMainLane }));
    // 배치 토글은 레인이 켜져 있을 때만 의미가 있다 — syncQuickRow가 표시를 정한다.
    // 3단 순환: left → bottom → attached → left (LANE_POS_CYCLE 참고)
    // 레인 «위치» 버튼은 표면에 따라 다른 것을 옮긴다.
    //   floating: 패널 안 배치 순환(left → bottom → attached)
    //   filled  : 부착 개념이 없어 순환이 유명무실하다 → **레인 열과 중앙 열을 맞바꾼다**.
    //             기본 배치에서는 레인과 가사창이 창 양 끝이라 둘을 함께 보기 어렵다
    //             (운영자 용례: 따라 부르며 가사도 같이 보고 싶다).
    this.quickLanePosBtn = this.miniButton(MINI_POS_LEFT_SVG, t('overlay.quick.lanePos'),
      () => this.callbacks.onSettingsChange(this.chrome === 'filled'
        ? { pipLaneSwapped: !this.settings.pipLaneSwapped }
        : { mainLanePos: LANE_POS_CYCLE[this.settings.mainLanePos] }));
    this.quickCaptionBtn = this.miniButton(MINI_CAPTION_SVG, t('overlay.quick.caption'),
      () => this.callbacks.onSettingsChange({ videoCaptions: !this.settings.videoCaptions }));
    this.quickPlaylistBtn = this.miniButton(MINI_PLAYLIST_SVG, t('overlay.quick.playlist'),
      () => this.callbacks.onSettingsChange(this.chrome === 'filled'
        ? { pipPlaylist: !this.settings.pipPlaylist }
        : { modPlaylist: !this.settings.modPlaylist }));
    // 번역 언어 칩을 같은 줄 오른쪽 끝에 함께 싣는다 — 별도 줄로 두면 얇은 줄이 두 개
    // 쌓여 세로 공간만 먹는다(실사용 제보). 칩 줄 표시/숨김(renderLangChips)은 그대로
    // 자기 display로 하고, 접힘 상태는 퀵 줄과 함께 사라진다.
    this.quickRow = h('div', { className: 'ey-quick-row' },
      this.quickLaneBtn, this.quickLanePosBtn, this.quickCaptionBtn,
      this.quickPlaylistBtn, this.langChipsRow);

    this.panel = h('div', { className: 'ey-panel' },
      this.header, this.quickRow, this.matchedBar, this.wrongLyricsPop, this.serverBar, this.banner, this.genChip, this.genList, this.noticeChip,
      this.warnBar, this.translationPendingBar, this.mainRow, this.resumeChip, this.laneWrap,
      this.footer, this.debugStrip, this.debugPanelEl,
    );
    // 레인 attach는 mountInto로 옮겼다 — 캔버스가 «어느 문서에 붙었는지»가 정해진 뒤에
    // 해야 ResizeObserver·dpr·CSS 변수가 그 창 것으로 잡힌다
    // 패널 안 타이핑(검색창·가사 붙여넣기)이 유튜브 전역 단축키(스페이스=재생/정지,
    // 방향키=시킹 등)로 새지 않도록 키 이벤트를 패널에서 끊는다
    for (const type of ['keydown', 'keyup', 'keypress'] as const) {
      this.panel.addEventListener(type, e => e.stopPropagation());
    }
    shadow.append(this.panel);
    // 부착 레인 패널 — this.panel의 형제. 패널 뒤에 붙여서 우연히 겹치는 경우(폴백)
    // 스택 순서상 패널 위에 그려지게 한다.
    shadow.append(this.attachPanel);
    // 재생목록 부착 패널도 같은 규약(this.panel의 형제, 겹침 폴백 시 패널 위)
    shadow.append(this.attachPlaylistPanel);

    this.geometry = geometry ?? this.defaultGeometry();
    this.applySettings(settings);
    this.updateOffsetLabel();
    // 드래그는 좌표가 있는 패널에만 의미가 있다 — filled는 창이 곧 패널이라 옮길 곳이 없다
    if (this.chrome === 'floating') this.setupDrag();
    void this.refreshNoticesButton();
  }

  /**
   * 이 패널을 문서에 붙이고 «그 문서의 창»에 결합한다.
   *
   * 생성자에서 떼어낸 이유는 인스턴스가 둘이기 때문이다 — 메인 문서에 하나, PiP 문서에
   * 하나. 옵저버·리스너를 대상 문서의 창에서 만들지 않으면 PiP 인스턴스가 유튜브 탭의
   * window에 매달려, 그 탭이 숨겨져 스로틀되는 순간 조용히 멎는다.
   *
   * 레인 캔버스도 여기서 다시 attach한다: PitchLaneRenderer는 attach 시점의
   * canvas.ownerDocument로 ResizeObserver·devicePixelRatio·CSS 변수를 잡는데,
   * h()가 만든 캔버스는 문서에 붙기 전까지 «메인 문서» 소속이기 때문이다.
   */
  mountInto(doc: Document, container?: Element, slots?: OverlaySlots): void {
    if (this.doc === doc) return;
    if (this.doc) this.unbindWindow();
    this.doc = doc;
    this.win = doc.defaultView;
    // 유튜브 페이지에서는 documentElement 직속(페이지 레이아웃에 끼어들지 않는 0크기 앵커),
    // PiP에서는 창을 가로로 나눈 가운데 열 하나로 들어간다
    (container ?? doc.documentElement).append(this.host);
    // 부착 패널을 «패널 바깥의 남의 열»로 내보낸다 — filled 전용. 근거는 OverlaySlots 주석.
    if (slots?.laneSlot) {
      this.laneSlot = slots.laneSlot;
      slots.laneSlot.append(this.attachPanel);
    }
    if (slots?.playlistSlot) {
      this.playlistSlot = slots.playlistSlot;
      slots.playlistSlot.append(this.attachPlaylistPanel);
    }
    this.onColumnsChanged = slots?.onColumnsChanged ?? null;
    this.lane.attach(this.laneCanvas, this.panel);
    this.applyGeometry();
    const RO = (this.win as unknown as { ResizeObserver?: typeof ResizeObserver })?.ResizeObserver
      ?? ResizeObserver;
    this.resizeObserver = new RO(() => this.handlePanelResize());
    this.resizeObserver.observe(this.panel);
    // 창 크기 추종과 전체화면 회피는 유튜브 페이지 위에 떠 있는 패널만의 문제다.
    // PiP 창에는 전체화면 개념이 없고, 크기 변화는 ResizeObserver가 이미 잡는다.
    if (this.chrome === 'floating') {
      this.win?.addEventListener('resize', this.handleWindowResize);
      doc.addEventListener('fullscreenchange', this.handleFullscreenChange);
    }
    // 배치를 **여기서 한 번 더** 돌린다. 생성자의 applySettings가 이미 한 번 돌았지만
    // 그때는 아직 slots가 없어(mountInto가 방금 넣었다) 레인이 패널 안쪽 열로 갔고,
    // 바깥 열의 표시 여부도 정해지지 않았다. 슬롯을 아는 지금이 진짜 배치 시점이다.
    if (slots) {
      this.applyLaneVisibility();
      this.updatePlaylistPlacement();
    }
  }

  /**
   * «같은 버튼을 두 번 눌러 확인» — window.confirm을 대신한다.
   *
   * window.confirm은 그것을 부른 창이 아니라 최상위 탭에 모달을 띄운다. 인스턴스가 둘이
   * 되면서(유튜브 페이지 + PiP 문서) PiP에서 부르면 사용자가 보고 있는 창에는 아무 일도
   * 일어나지 않고 유튜브 탭이 응답을 기다려, 화면이 멈춘 것처럼 보인다. 설정 시트의
   * 전체 초기화가 이미 쓰던 방식을 공용으로 올려 되돌릴 수 없는 동작 전부에 적용한다.
   *
   * 무장은 4초 뒤 스스로 풀린다 — 남아 있는 무장이 다음 클릭을 삼키면 사용자는 "한 번
   * 눌렀는데 실행됐다"고 느낀다. 다른 버튼을 무장하면 이전 무장은 즉시 풀린다.
   *
   * @returns 이번 클릭이 «확정»이면 true — 호출부는 이때만 실제 동작을 실행한다
   */
  private confirmTwice(btn: HTMLElement, prompt: string): boolean {
    if (this.armedBtn === btn) {
      this.disarmConfirm();
      return true;
    }
    this.disarmConfirm();
    this.armedBtn = btn;
    this.armedTitle = btn.title;
    btn.title = prompt;
    btn.classList.add('ey-confirm-armed');
    this.confirmTimer = window.setTimeout(() => this.disarmConfirm(), 4000);
    return false;
  }

  private disarmConfirm(): void {
    clearTimeout(this.confirmTimer);
    const b = this.armedBtn;
    if (!b) return;
    b.classList.remove('ey-confirm-armed');
    b.title = this.armedTitle;
    this.armedBtn = null;
  }

  /** 창에 건 것만 끊는다 — DOM·상태는 그대로 두므로 다른 문서로 다시 mountInto할 수 있다 */
  private unbindWindow(): void {
    this.resizeObserver?.disconnect();
    this.resizeObserver = null;
    (this.win ?? window).cancelAnimationFrame(this.geomRaf);
    this.geomRaf = 0;
    (this.win ?? window).cancelAnimationFrame(this.settingsRaf);
    this.settingsRaf = 0;
    // 등록과 «같은 조건»으로만 해제한다 — filled는 애초에 걸지 않았다. removeEventListener
    // 자체는 무해하지만, 조건이 어긋나 있으면 읽는 사람이 "filled에도 리스너가 있나?"로
    // 오해하고, 등록/해제 대칭을 보는 누수 계측도 음수로 어긋난다(반복 측정에서 실제로 발생).
    if (this.chrome === 'floating') {
      this.win?.removeEventListener('resize', this.handleWindowResize);
      this.doc?.removeEventListener('fullscreenchange', this.handleFullscreenChange);
    }
    // 레인은 자기 ResizeObserver를 따로 들고 있다 — 이걸 빼먹은 것이 실측에서 잡힌
    // 유일한 «치명» 누수였다(PiP를 여닫을 때마다 옵저버와 캔버스가 통째로 남았다)
    this.lane.detach();
  }

  /**
   * 창 결합을 전부 끊고 DOM에서 뗀다.
   *
   * 예전엔 호출처가 없는 «한 번도 실행된 적 없는» 코드였는데, PiP 인스턴스가 닫힐 때마다
   * 도는 임계 경로가 되면서 실검증했다. 그때 드러난 누수 3종을 여기서 막는다:
   * 레인 ResizeObserver(lane.detach 누락), 알림 칩 타이머, 2단계 확인 타이머.
   */
  destroy(): void {
    this.unbindWindow();
    // 아래 host.remove()는 «host 안»만 걷는다 — filled에서 바깥 열로 내보낸 세 조각
    // (단축 표시·부착 레인·부착 재생목록)은 남의 부모에 붙어 있으므로 따로 떼어낸다.
    this.detachShortView();
    if (this.laneSlot) this.attachPanel.remove();
    if (this.playlistSlot) this.attachPlaylistPanel.remove();
    clearTimeout(this.saveGeomTimer);
    clearTimeout(this.noticeTimer);
    clearTimeout(this.confirmTimer);
    this.host.remove();
    this.doc = null;
    this.win = null;
  }

  // ── 상태 렌더링 ────────────────────────────────────────────────

  /** 패널 조각(panels.ts)에 넘기는 호스트 컨텍스트 — 콜백 + 서버 상태 연동 생성 버튼 */
  private panelContext(): PanelContext {
    return {
      callbacks: {
        onGenerate: (lyrics, attribution) => this.callbacks.onGenerate(lyrics, attribution),
        onRetrySearch: query => this.callbacks.onRetrySearch(query),
        onCandidateSearch: query => this.callbacks.onCandidateSearch(query),
        onPickCandidate: candidate => this.callbacks.onPickCandidate(candidate),
        onOpenSearch: () => this.openSearch(),
        onOpenSettings: () => this.openSettings(),
        onRecheckServer: () => this.callbacks.onRecheckServer(),
        onOpenPermissions: () => this.callbacks.onOpenPermissions(),
      },
      makeGenerateButton: (label, onClick) => this.makeGenerateButton(label, onClick),
      server: this.serverStatus,
      debug: this.settings.debugInfo,
      loadServerLog: () => this.callbacks.loadServerLog(),
    };
  }

  showLoading(message = t('overlay.loading.default')): void {
    this.stateKind = 'loading';
    this.resetBody();
    this.body.append(buildLoadingState(this.panelContext(), message));
  }

  /**
   * @param generateBlocked 이 가사로는 싱크를 만들 수 없는 사유. 주면 배너에 버튼 없이
   *   그 사유만 띄운다 — 누르면 늘 거절하는 버튼을 내주지 않기 위해서다.
   */
  showSyncedLyrics(
    lines: LyricLine[], source: LyricsSource, plainText?: string, generateBlocked?: string,
  ): void {
    this.stateKind = 'synced';
    this.resetBody();
    this.lines = lines;
    this.lineEls = [];
    this.filledUpTo = 0;
    this.currentIndex = -1;

    // LRCLIB 등 외부 싱크 가사도 서버 전사를 만들면 음정 노트·발음 정렬·가라오케를 쓸 수 있다
    if (source !== 'everyric') {
      const text = plainText ?? lines.map(l => l.text).join('\n');
      if (generateBlocked) this.showBanner(generateBlocked);
      else {
        // 유튜브 자막 표시 상태는 문구를 따로 쓴다 — "지금 보는 건 업로더 자막이지
        // Everyric 싱크가 아니다"를 명시(운영자 지시, 2026-08-03: 자막 타이밍 오차를
        // 확장의 싱크 품질로 오해하는 사용자 보고). 배지(.ey-source.caption)와 함께
        // 이 상태를 한눈에 구분되게 한다.
        const bannerText = source === 'caption'
          ? t('overlay.banner.captionPreSync') : t('overlay.banner.aiKaraoke');
        this.showBanner(bannerText,
          this.makeGenerateButton(t('overlay.banner.aiTranscribe'), () => this.callbacks.onGenerate(text)));
      }
    }

    // 발음 표기 방식은 라인마다 다시 해석할 이유가 없다 — 곡 하나에 한 번만 계산
    const pronScript = resolveScript(this.settings);
    const list = h('div', { className: 'ey-lines' });
    lines.forEach((line, index) => {
      const { el } = buildLineEl(line, pronScript, this.settings);
      el.title = t('overlay.line.seekTitle');
      el.addEventListener('click', () => {
        // 줄 시작 시각으로 정확히 시크하면 브라우저가 그 지점 **이하**의 디코딩
        // 가능한 위치로 스냅해서, 곡 시간이 줄 시작보다 살짝 앞에 떨어진다.
        // 그러면 활성 줄 판정이 한 줄 위로 가서 "누른 줄의 윗칸이 눌린" 것처럼
        // 보인다. 줄 안쪽으로 아주 조금 밀어 넣어 의도한 줄에서 시작하게 한다.
        if (line.time !== null) this.callbacks.onSeek(line.time + SEEK_INTO_LINE_SEC);
      });
      el.dataset.index = String(index);
      this.lineEls.push(el);
      list.append(el);
    });
    this.body.append(list);

    this.setSourceBadge(source, true);
    this.footer.classList.remove('no-offset');
    this.footer.style.display = '';
    this.pipBtn.style.display = this.pipEnabled ? '' : 'none';
    // 싱크 초기화는 서버(everyric) 싱크에서만 의미가 있다
    this.resetSyncBtn.style.display = source === 'everyric' ? '' : 'none';
    // 깊이 버튼도 여기서 갱신 — 구세대 싱크면 초기화 버튼을 업그레이드 버튼이 대신한다
    this.updateDepthButton();
    // 별점·오류 제보도 everyric 싱크에서만 — 평가 대상이 서버 정렬이다
    this.feedbackBtn.style.display = source === 'everyric' ? '' : 'none';
    // [모듈] 레인도 같은 가사를 받는다 — 노트가 없으면 applyLaneVisibility가 알아서 숨긴다
    this.lane.setLines(lines);
    this.applyLaneVisibility();
  }

  showPlainLyrics(lines: LyricLine[], source: LyricsSource, plainText: string): void {
    this.stateKind = 'plain';
    this.resetBody();
    this.plainTextForGenerate = plainText;

    const generateBtn = this.makeGenerateButton(t('overlay.plain.generateSync'), () => this.callbacks.onGenerate(this.plainTextForGenerate));
    this.showBanner(t('overlay.plain.noTimesync'), generateBtn);

    this.lines = lines;
    const plain = buildPlainLines(lines);
    this.lineEls.push(...plain.lineEls);
    this.body.append(plain.el);

    this.setSourceBadge(source, false);
    this.footer.classList.add('no-offset');
    this.footer.style.display = '';
  }

  showEmpty(song: SongInfo | null): void {
    this.stateKind = 'empty';
    this.resetBody();
    this.body.append(buildEmptyState(this.panelContext(), song));
  }

  /** 상시 재검색: 현재 곡 정보를 초기값으로 검색 폼 + 소스별 후보 리스트를 연다 */
  openSearch(): void {
    this.stateKind = 'search';
    this.resetBody();

    const sheet = buildSearchSheet(
      this.panelContext(),
      {
        title: this.lastSong?.title ?? '',
        artist: this.lastSong?.artist ?? '',
        rawTitle: this.rawVideoTitle(),
      },
      {
        onBack: () => this.callbacks.onCloseSearch(),
        // 메인 패널에만 있는 고급 섹션 — 다른 영상 싱크 연결과 서버 저장 삭제는
        // 실수 여파가 커서 PiP의 축약 검색 시트에는 넣지 않는다
        extras: [
          h('div', { className: 'ey-divider' }),
          this.buildLinkSection(),
          h('div', { className: 'ey-divider' }),
          h('button', {
            className: 'ey-secondary-btn',
            text: t('overlay.search.backToAuto'),
            on: { click: () => this.callbacks.onRetrySearch() },
          }),
          h('button', {
            className: 'ey-secondary-btn',
            text: t('overlay.search.resetSync'),
            attrs: { title: t('overlay.search.resetSyncTitle') },
            on: {
              click: (e: MouseEvent) => {
                const btn = e.currentTarget as HTMLElement;
                if (this.confirmTwice(btn, t('overlay.search.resetSyncConfirm'))) {
                  this.callbacks.onResetSync();
                }
              },
            },
          }),
        ],
      },
    );
    this.searchResultsEl = sheet.results;
    this.body.append(sheet.el);
    sheet.runSearch();
  }

  /**
   * 영상 제목 원본 — SongInfo는 **정리된** 제목만 들고 다닌다(song-detector가 【】·MV·
   * 업로더 접두를 걷어낸 뒤 원본을 버린다). 검색 폼의 '영상 제목 그대로' 탈출구는 걷기
   * 전 값이 있어야 나타나므로, 여기서 페이지 제목에서 유튜브 접미만 떼어 되살린다.
   *
   * 이것은 근사치다 — song-detector가 실제로 정리 대상으로 삼은 h1(또는 mediaSession)
   * 제목과 다를 수 있다. 정확한 원본은 content가 SongInfo에 실어 보내야 하고, 그때
   * 이 폴백은 지워진다. 정리본과 같은 값이면 buildSearchForm이 버튼을 내지 않으므로
   * 근사가 빗나가도 화면에 남는 피해는 없다.
   */
  private rawVideoTitle(): string | undefined {
    // 반드시 «유튜브 페이지» 문서에서 읽는다 — PiP 문서의 title은 pip.ts가 덮어쓴 값이라
    // filled 인스턴스가 전역 document를 보면 곡명 대신 PiP 창 제목이 잡힌다
    const raw = this.pageDoc.title.replace(/ - YouTube$/, '').trim();
    return raw && raw !== 'YouTube' ? raw : undefined;
  }

  /**
   * 공지 진입점 표시 판정 — 서버에 공지 기능이 없으면 버튼을 **통째로 숨긴다**.
   * 오류 문구를 띄우지 않는 이유는 probeNotices 주석과 같다: 구버전 서버에서 404는
   * 비정상이 아니라 정상 경로이고, 없는 기능을 고장난 기능처럼 보이게 하면 안 된다.
   */
  private async refreshNoticesButton(): Promise<void> {
    const probe = await probeNotices();
    this.noticesBtn.style.display = probe.available ? '' : 'none';
    this.noticesDot.style.display = probe.unread ? '' : 'none';
  }

  /**
   * 공지함 — 본문 자리를 빌린다(패널 안에 또 다른 창을 띄우지 않는다).
   * '뒤로'는 검색 시트의 탈출구와 **같은 경로**(onCloseSearch)를 쓴다: content가 지금
   * 곡 데이터를 그대로 다시 그리므로, 시트를 여닫아도 보던 화면이 정확히 돌아온다.
   */
  openNotices(): void {
    if (!this.visible) this.setVisible(true);
    if (this.geometry.collapsed) this.setCollapsed(false);
    this.stateKind = 'sheet';
    this.resetBody();
    const { el } = buildNoticesSheet({
      onBack: () => this.callbacks.onCloseSearch(),
      // 목록을 그린 순간이 '본' 시점이다 — 점을 끄는 자리도 여기 하나뿐이다
      onSeen: () => { this.noticesDot.style.display = 'none'; },
    });
    this.body.append(el);
  }

  /** 내 기여·남은 한도 시트 — 한도는 영상 기준 조회라 지금 영상 id가 필요하다 */
  openContribution(): void {
    if (!this.visible) this.setVisible(true);
    if (this.geometry.collapsed) this.setCollapsed(false);
    this.stateKind = 'sheet';
    this.resetBody();
    const { el } = buildContributionSheet({
      videoId: this.callbacks.getVideoId(),
      // 지금 보던 곡을 잃지 않도록 새 탭으로 — 같은 탭 이동이면 여기 목록을 다시 열
      // 방법이 뒤로가기뿐이다
      onOpenVideo: videoId => window.open(
        `https://www.youtube.com/watch?v=${encodeURIComponent(videoId)}`, '_blank', 'noopener'),
      onBack: () => this.callbacks.onCloseSearch(),
    });
    this.body.append(el);
  }

  /** 다른 영상 싱크 연결 섹션 (inst·커버 영상용) — 검색 시트 하단 */
  private buildLinkSection(): HTMLDivElement {
    const srcInput = h('input', {
      className: 'ey-input',
      attrs: { placeholder: t('overlay.link.srcPlaceholder') },
    });
    this.linkSrcInput = srcInput;
    const offsetInput = h('input', {
      className: 'ey-input ey-input-narrow',
      attrs: { type: 'number', step: '0.1', placeholder: t('overlay.link.offsetPlaceholder'), title: t('overlay.link.offsetTitle') },
    });
    offsetInput.value = this.linkedInfo ? String(this.linkedInfo.offsetSec) : '0';
    // 배속이 다른 커버(nightcore 등)는 고정 오프셋만으론 뒤로 갈수록 밀린다 — 서버가
    // t/배속+오프셋으로 시간축을 사상한다
    const rateInput = h('input', {
      className: 'ey-input ey-input-narrow',
      attrs: {
        type: 'number', step: '0.01', min: '0.5', max: '2', placeholder: t('overlay.link.ratePlaceholder'),
        title: t('overlay.link.rateTitle'),
      },
    });
    rateInput.value = this.linkedInfo?.rate ? String(this.linkedInfo.rate) : '1';
    this.linkListEl = h('div', { className: 'ey-result-list' });
    const filterInput = h('input', {
      className: 'ey-input',
      attrs: { placeholder: t('overlay.link.filterPlaceholder') },
      on: { input: () => this.renderSyncList() },
    });
    filterInput.style.display = 'none'; // 목록을 불러온 뒤에만 노출
    this.linkFilterInput = filterInput;

    const doLink = () => {
      const src = parseVideoId(srcInput.value.trim());
      if (!src) {
        this.setLinkStatus(t('overlay.link.needVideoId'));
        return;
      }
      const offset = Number(offsetInput.value) || 0;
      const rate = Math.min(2, Math.max(0.5, Number(rateInput.value) || 1));
      this.setLinkStatus(t('overlay.link.connecting'));
      this.callbacks.onLinkSync(src, offset, rate);
    };

    const section = h('div', { className: 'ey-link-section' },
      h('div', { className: 'ey-state-text', text: t('overlay.link.sectionTitle') }),
    );
    if (this.linkedInfo) {
      const rateBadge = this.linkedInfo.rate && this.linkedInfo.rate !== 1
        ? ` ×${this.linkedInfo.rate}` : '';
      // 검증된 자동 연결과 검증 없는 수동 연결을 구분해 말한다 — 코퍼스에 검증 없는
      // 링크가 섞여 있어서, 가사가 어긋날 때 이 표시가 원인 판단의 첫 단서가 된다
      const verifyBadge = this.linkedInfo.verified === true ? t('overlay.link.verifiedBadge')
        : this.linkedInfo.verified === false ? t('overlay.link.unverifiedBadge') : '';
      section.append(h('div', { className: 'ey-link-current' },
        h('span', {
          text: t('overlay.link.currentStatus', [
            this.linkedInfo.sourceVideoId,
            `${this.linkedInfo.offsetSec >= 0 ? '+' : ''}${this.linkedInfo.offsetSec}s${rateBadge}`,
            verifyBadge,
          ]),
          attrs: { title: this.describeLink(this.linkedInfo) },
        }),
        h('button', {
          className: 'ey-secondary-btn',
          text: t('overlay.link.unlink'),
          on: { click: () => this.callbacks.onUnlinkSync() },
        }),
      ));
    }
    section.append(
      h('div', { className: 'ey-search-form' },
        srcInput,
        offsetInput,
        rateInput,
        h('button', { className: 'ey-primary-btn', text: t('overlay.link.connect'), on: { click: doLink } }),
      ),
      h('button', {
        className: 'ey-secondary-btn',
        text: t('overlay.link.pickFromList'),
        on: {
          click: () => {
            this.setLinkStatus(t('overlay.link.loadingList'));
            this.callbacks.onRequestSyncList();
          },
        },
      }),
      filterInput,
      this.linkListEl,
    );
    return section;
  }

  /** SYNC_LIST 응답 반영 — 목록을 캐시하고 검색 필터와 함께 렌더 */
  showSyncList(items: SyncListItem[]): void {
    if (this.stateKind !== 'search' || !this.linkListEl) return;
    this.syncListItems = items;
    if (this.linkFilterInput) {
      this.linkFilterInput.style.display = items.length === 0 ? 'none' : '';
      this.linkFilterInput.value = '';
    }
    if (items.length === 0) {
      this.setLinkStatus(t('overlay.link.noSavedSyncs'));
      return;
    }
    this.renderSyncList();
  }

  /** 저장 싱크 목록 렌더 — 필터(가사 첫 줄·영상 ID·출처·정렬문) 적용 */
  private renderSyncList(): void {
    if (!this.linkListEl || !this.syncListItems) return;
    const q = (this.linkFilterInput?.value ?? '').trim().toLowerCase();
    const items = q
      ? this.syncListItems.filter(it =>
        it.video_id.toLowerCase().includes(q)
        || (it.first_line ?? '').toLowerCase().includes(q)
        || (it.attribution_name ?? '').toLowerCase().includes(q)
        || (it.alignment_text ?? '').toLowerCase().includes(q))
      : this.syncListItems;
    if (items.length === 0) {
      this.linkListEl.replaceChildren(
        h('div', { className: 'ey-state-sub', text: t('overlay.link.noFilterMatch') }));
      return;
    }
    const hint = h('div', {
      className: 'ey-state-sub',
      text: t('overlay.link.pickHint'),
    });
    this.linkListEl.replaceChildren(hint, ...items.map(it => {
      const btn = h('button', { className: 'ey-result-item' },
        h('span', { className: 'ey-result-src', text: it.video_id }),
        h('span', { className: 'ey-result-title', text: it.first_line || t('overlay.link.noFirstLine') }),
        h('span', { className: 'ey-result-meta', text: t('overlay.link.lineCountMeta', [String(it.line_count)]) + (it.attribution_name ? ' · ' + it.attribution_name : '') }),
      );
      btn.addEventListener('click', () => {
        if (this.linkSrcInput) this.linkSrcInput.value = it.video_id;
        this.linkListEl?.querySelectorAll('.ey-selected').forEach(el => el.classList.remove('ey-selected'));
        btn.classList.add('ey-selected');
      });
      return btn;
    }));
  }

  /** 링크 섹션 상태 메시지 (검색 상태가 아니면 무시) */
  setLinkStatus(message: string): void {
    setListStatus(this.linkListEl, message);
  }

  /** 현재 싱크의 링크 상태 — 검색 시트의 해제 UI와 출처 배지에 반영 */
  setLinked(
    info: { sourceVideoId: string; offsetSec: number; rate?: number; verified?: boolean } | null,
  ): void {
    this.linkedInfo = info;
  }

  /** 제목바 언어 칩의 "보유" 목록 — null이면(가사 자체가 없음: 로딩 중·빈 패널) 칩 줄을
   *  숨긴다. 가사가 있으면 content가 항상 비어 있지 않은 배열을 넘긴다(서버 신호가 없어도
   *  최소 곡 자신의 언어는 담아서). 곡이 바뀔 때(applyLyricsData)와, 번역이 새로 성공
   *  적용될 때(다른 언어가 막 생겼을 때) 둘 다 호출된다. */
  setAvailableLangs(langs: string[] | null): void {
    this.availableLangs = langs;
    this.renderLangChips();
  }

  /**
   * 방금 클릭해 요청을 보낸 언어 — 응답이 올 때까지(성공이든 실패든) 그 칩만 로딩 표시.
   * null로 부르면(항상 요청 직후 한 번) 로딩을 끈다.
   *
   * U3-b: 칩 펄스만으론 "빈 번역 줄"이 로딩 중이라는 게 눈에 잘 안 띈다는 실보고 —
   * 라인 목록 바로 위(translationPendingBar)에도 같은 사실을 알린다. 칩과 같은 언어
   * 라벨을 재사용해 어느 언어를 기다리는지도 함께 보여준다.
   */
  setLangPending(lang: string | null): void {
    this.pendingLang = lang;
    this.renderLangChips();
    if (lang) {
      const label = this.langChipButtons.find(c => c.code === lang)?.btn.textContent ?? lang;
      this.translationPendingBar.textContent = t('overlay.translationPending', [label]);
      this.translationPendingBar.style.display = '';
    } else {
      this.translationPendingBar.style.display = 'none';
    }
  }

  private renderLangChips(): void {
    if (!this.availableLangs) {
      this.langChipsRow.style.display = 'none';
      return;
    }
    this.langChipsRow.style.display = '';
    for (const { code, btn } of this.langChipButtons) {
      const isCurrent = code === this.settings.translationLanguage;
      const isPending = code === this.pendingLang;
      const isAvailable = this.availableLangs.includes(code);
      btn.classList.toggle('current', isCurrent && !isPending);
      btn.classList.toggle('pending', isPending);
      btn.classList.toggle('available', !isCurrent && !isPending && isAvailable);
      btn.classList.toggle('unavailable', !isCurrent && !isPending && !isAvailable);
      btn.title = isPending
        ? t('overlay.langChip.generating')
        : isCurrent
          ? t('overlay.langChip.current')
          : isAvailable
            ? t('overlay.langChip.switchTo')
            : t('overlay.langChip.generateFor');
    }
  }

  /** SEARCH_CANDIDATES 응답 반영 — 검색 상태가 아니면 무시 (stale 응답 방지) */
  showSearchResults(candidates: SearchCandidate[]): void {
    if (this.stateKind !== 'search' || !this.searchResultsEl) return;
    renderCandidateList(this.searchResultsEl, candidates, c => this.callbacks.onPickCandidate(c));
  }

  showGenerating(progress: number, label?: string): void {
    const pct = Math.max(0, Math.min(100, Math.round(progress)));
    const text = label ?? t('overlay.generating.default', [String(pct)]);
    if (this.stateKind === 'generating' && this.progressBar && this.progressText) {
      this.progressBar.style.width = `${pct}%`;
      this.progressText.textContent = text;
      return;
    }
    this.stateKind = 'generating';
    this.resetBody();
    const refs = buildGeneratingState(pct, text);
    this.progressBar = refs.bar;
    this.progressText = refs.text;
    this.body.append(refs.el);
  }

  /** detail은 서버가 준 힌트 등 추가 사유 — 있으면 문구 아래 한 줄로 함께 보여 준다 */
  showError(message: string, detail?: string): void {
    this.stateKind = 'error';
    this.resetBody();
    this.body.append(buildErrorState(this.panelContext(), message, detail));
  }

  /**
   * 이 화면을 오류 화면으로 덮으면 사용자가 잃는 것이 있는가.
   *
   * showError는 resetBody()를 타서 본문을 통째로 버린다 — 보고 있던 가사·검색 시트·**방금
   * 붙여넣던 본문**까지. 잡 실패·요청 실패·500줄 초과가 전부 그 경로였고, 특히 500줄 초과는
   * 실패 문구 한 줄을 주면서 사용자가 옮겨 적은 가사를 날렸다. 실패는 알리되 화면 상태는
   * 보존해야 하므로, 호출부(content.reportFailure)는 이 판정이 참이면 알림 칩만 쓴다.
   *
   * 반대로 가사도 입력도 없는 화면(검색 중·조회 실패 직후)에서는 잃을 것이 없고, 그때는
   * 본문 오류 화면이 '다시 시도' 버튼까지 함께 줄 수 있어 더 낫다.
   */
  hasPreservableContent(): boolean {
    if (this.stateKind === 'synced' || this.stateKind === 'plain' || this.stateKind === 'search') return true;
    // 빈 상태·검색 시트의 붙여넣기 칸은 이제 항상 열려 있다(buildPasteSection이 접힘
    // 토글 없이 상시 노출한다, 2026-07) — 타이핑한 본문이 있으면 그것이 이 화면에서
    // 가장 값진 것이다
    return [...this.body.querySelectorAll<HTMLTextAreaElement>('textarea')]
      .some(ta => ta.value.trim().length > 0);
  }

  showPipPlaceholder(): void {
    this.stateKind = 'pip';
    this.resetBody();
    this.body.append(
      h('div', { className: 'ey-state' },
        h('div', { className: 'ey-state-emoji', text: '🪟' }),
        h('div', { className: 'ey-state-text', text: t('overlay.pip.placeholder') }),
        h('button', { className: 'ey-primary-btn', text: t('overlay.pip.backToPanel'), on: { click: () => this.callbacks.onPipToggle() } }),
      ),
    );
  }

  // ── 싱크 업데이트 ──────────────────────────────────────────────

  highlightLine(index: number): void {
    if (this.stateKind !== 'synced') return;
    const prevIndex = this.currentIndex;
    this.currentIndex = index;
    this.lane.setIndex(index); // 레인의 현재 라인 번역·발음 폴백 줄이 이 값을 따른다
    this.activeWordEls = [];
    this.lineEls.forEach((el, i) => {
      el.classList.toggle('active', i === index);
      el.classList.toggle('past', index >= 0 && i < index);
    });
    // 채움(sung) 상태는 **현재 위치의 함수**다 — 활성 줄 앞은 전부 채우고 뒤는 전부
    // 비운다. 재생으로 지나왔는지 클릭으로 건너뛰었는지와 무관하게 같은 화면이 나와야
    // 한다. 예전에는 updateTime이 활성 줄에만 sung을 붙여서, 앞으로 건너뛰면 지나친
    // 줄들이 안 채워지고 되감으면 미래 줄에 채움이 남았다.
    //
    // 직전 활성 줄은 부분적으로만 채워져 있을 수 있으니 먼저 비우고, 아래 fillUpTo가
    // 규칙대로 다시 칠한다.
    if (prevIndex >= 0 && prevIndex !== index) this.setLineFilled(prevIndex, false);
    this.fillUpTo(Math.max(0, index));
    const active = index >= 0 ? this.lineEls[index] : undefined;
    if (active) {
      // 발음 음절(.ey-pron-syl)도 단어와 같은 sung 토글 메커니즘에 합류
      this.activeWordEls = collectFillTargets(active);
      if (Date.now() >= this.userScrollUntil) {
        this.scrollToCurrent();
      } else {
        this.resumeChip.style.display = '';
      }
    }
    // 단축 표시(PiP 중앙 열)도 같은 지점에서 따라온다 — 별도 갱신 경로를 만들지 않는다
    this.renderShortView();
  }

  updateTime(time: number, paused = false): void {
    updateFillTargets(this.activeWordEls, time);
    // 단축 표시의 현재 줄도 같은 함수로 채운다(목록과 채움 규칙이 갈라질 수 없다)
    if (this.shortFillTargets.length > 0) updateFillTargets(this.shortFillTargets, time);
    this.updateVocalGlow(time);
    this.lastTime = time;
    this.lanePaused = paused;
    // 모듈이 꺼져 있으면(기본값) 렌더러를 아예 부르지 않는다 — 숨긴 캔버스라도 매 프레임
    // clientWidth를 읽으면 유튜브 페이지에 강제 리플로우를 60Hz로 얹게 된다
    if (this.laneShown) this.lane.render(time, paused);
  }

  /**
   * 레인 클릭 → 그 시각으로 시크 (노트 위면 노트 시작으로 스냅) — PiP 레인의 클릭 시크와
   * 같은 규칙이다. 드래그 팬·휠 줌은 PiP에만 두었다: 메인 패널은 휠이 가사 목록 스크롤과
   * 겹치고, 여기서 창 설정을 바꾸면 저장 경로가 갈라진다.
   */
  private seekFromLane(e: MouseEvent): void {
    const view = this.lane.viewport();
    if (!view) return;
    const rect = this.laneCanvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    if (px < view.plotX || py > view.staffBottom) return; // 사이드바·가사 줄 클릭은 무시
    const t = view.t0 + ((px - view.plotX) / view.plotW) * view.W;
    const hit = this.lane.noteAt(t);
    this.callbacks.onSeek(hit ? hit.start : Math.max(0, t));
  }

  /** 레인을 마지막 시각으로 즉시 다시 그린다 — 일시정지·설정 변경 때 다음 tick을 안 기다린다 */
  private renderLane(): void {
    if (this.laneShown) this.lane.render(this.lastTime, this.lanePaused);
  }

  // ── 가사 단축 표시 (PiP 중앙 열이 빌려 가는 뷰) ──────────────────
  //
  // 「영상 바로 아래 현재 줄 한 줄」은 PiP의 오래된 화면이고 운영자가 유지를 지시했다.
  // 그런데 그것을 pip.ts가 다시 그리면 가사 렌더 구현이 **세 번째**로 늘어난다(메인 목록,
  // 레인, 그리고 이것) — 이번 재작업이 없애려던 바로 그 구조다. 그래서 소유자를 이 클래스로
  // 둔다: 줄 DOM은 목록과 **완전히 같은** buildLineEl이 만들고, 채움은 같은
  // collectFillTargets/updateFillTargets가 굴리며, 갱신 시점도 이미 도는 highlightLine·
  // updateTime·refreshTranslations에 얹힌다. pip.ts는 «어디에 놓을지»만 정한다.

  /**
   * 단축 표시 뷰를 만들어 주어진 컨테이너에 붙인다 (PiP 문서의 light DOM).
   * 같은 CSS 전문이 PiP 문서 <head>에 주입돼 있어 `.ey-pip-stage`·`.ey-line` 규칙이 그대로 먹는다.
   */
  attachShortView(container: Element): void {
    if (!this.shortEl) {
      this.shortPrevEl = h('div', { className: 'ey-pip-line prev', attrs: { dir: 'auto' } });
      this.shortCurrentEl = h('div', { className: 'ey-pip-line current', attrs: { dir: 'auto' } });
      this.shortNextEl = h('div', { className: 'ey-pip-line next', attrs: { dir: 'auto' } });
      // 앞뒤 줄 클릭 = 그 줄로 시크 (구 pip.ts seekRelative와 같은 규약)
      this.shortPrevEl.addEventListener('click', () => this.seekToLine(this.currentIndex - 1));
      this.shortNextEl.addEventListener('click', () => this.seekToLine(this.currentIndex + 1));
      this.shortCurrentEl.addEventListener('click', () => this.seekToLine(this.currentIndex));
      this.shortEl = h('div', { className: 'ey-pip-stage' },
        this.shortPrevEl, this.shortCurrentEl, this.shortNextEl);
    }
    container.append(this.shortEl);
    this.renderShortView();
  }

  /** 단축 표시를 떼어 낸다 — 창이 닫힐 때(destroy가 host만 걷으므로 이건 따로 걷어야 한다) */
  detachShortView(): void {
    this.shortEl?.remove();
    this.shortFillTargets = [];
  }

  private seekToLine(index: number): void {
    const line = this.lines[index];
    if (line && line.time !== null) this.callbacks.onSeek(line.time + SEEK_INTO_LINE_SEC);
  }

  /**
   * 단축 표시 다시 그리기 — 현재 줄만 «진짜 줄»로, 앞뒤는 흐린 한 줄 텍스트로.
   * 현재 줄의 원문/발음/번역과 카라오케 채움 구조가 가사 목록과 한 글자도 다르지 않다.
   */
  private renderShortView(): void {
    if (!this.shortEl || !this.shortCurrentEl) return;
    const cur = this.currentIndex >= 0 ? this.lines[this.currentIndex] : undefined;
    if (this.shortPrevEl) this.shortPrevEl.textContent = this.lines[this.currentIndex - 1]?.text ?? '';
    if (this.shortNextEl) this.shortNextEl.textContent = this.lines[this.currentIndex + 1]?.text ?? '';
    if (!cur) {
      this.shortCurrentEl.replaceChildren('♪');
      this.shortFillTargets = [];
      return;
    }
    const { el } = buildLineEl(cur, resolveScript(this.settings), this.settings);
    el.classList.add('active'); // 목록의 «현재 줄»과 같은 강조 규칙을 그대로 받는다
    this.shortCurrentEl.replaceChildren(el);
    this.shortFillTargets = collectFillTargets(el);
  }

  // ── 레인 다리 (창 주인이 부르는 세 가지) ────────────────────────
  //
  // PiP 창은 «창»의 일(OS 테마 변경 알림·재생 배속·재생 재개)을 알고 패널은 모른다.
  // 레인 렌더러를 통째로 노출하는 대신 필요한 세 동작만 이름 붙여 내준다.

  /**
   * 이 패널이 «레인 열을 원하는가» / «재생목록 열을 원하는가».
   *
   * 창 주인(pip.ts)이 열 폭을 계산할 때 쓴다. **자기가 쓴 style.display를 되읽으면 안
   * 되기 때문에** 따로 내준다 — 한 번 접은 열은 영원히 «원하지 않음»으로 굳어 창을 다시
   * 넓혀도 돌아오지 않는다(실측으로 잡힌 자기참조 래치).
   */
  laneVisible(): boolean {
    return this.laneShown;
  }

  playlistVisible(): boolean {
    // 표면별 키 — 메인에서 재생목록을 끄더라도 PiP의 열은 그대로다(그 반대도)
    return this.chrome === 'filled' ? this.settings.pipPlaylist : this.settings.modPlaylist;
  }

  /** OS/앱 테마가 바뀌어 CSS 변수가 낡았다 — 캐시한 레인 색을 버리고 즉시 한 번 다시 그린다 */
  refreshLaneColors(): void {
    this.lane.refreshColors();
    this.renderLane();
  }

  /** 원본 video 배속 — 마이크 궤적의 벽시계→곡 시간 환산에 필요하다 */
  setLanePlaybackRate(rate: number): void {
    this.lane.setOptions({ playbackRate: rate });
  }

  /** 재생이 재개되면 수동 스크롤(드래그·휠 팬)을 풀고 오토스크롤로 되돌린다 */
  clearLaneManualScroll(): void {
    this.lane.setManualT0(null);
  }

  /**
   * 레인 캔버스 위 휠 — 순수 세로 휠은 표시 구간(마디) 확대축소, 가로 휠·Shift+세로휠은
   * 일시정지 중 좌우 탐색(팬).
   *
   * 예전에는 PiP 창의 캔버스에만 있었다(구 pip.ts attachPitchPointer). 두 창이 같은
   * 인스턴스를 쓰는 지금은 «이식»이 아니라 **한 곳에 두기**다 — 여기에 한 번 달면
   * 메인 패널 레인과 PiP 레인이 함께 얻는다.
   *
   * Ctrl+휠은 브라우저 확대/축소와 충돌하니 절대 건드리지 않는다. ±값·클램프는 설정
   * 시트의 마디 창 행과 정확히 같은 축을 쓴다(새 배율 축을 만들지 않는 것이 버그 표면을
   * 최소화한다는 지시) — 저장은 onSettingsChange 한 경로로만 나간다.
   */
  private attachLaneWheel(): void {
    this.laneCanvas.addEventListener('wheel', (e: WheelEvent) => {
      if (!this.laneShown) return;
      if (!e.ctrlKey && !e.shiftKey && e.deltaX === 0 && e.deltaY !== 0) {
        e.preventDefault(); // 레인 위에서만 막는다 — 가사 목록 스크롤 침범 금지
        // **저장된 설정이 아니라 지금 화면의 값**에서 출발한다. onSettingsChange는
        // 저장소를 한 바퀴 돌아 applySettings로 되돌아오므로, 빠르게 두 번 굴리면 두
        // 번째가 아직 낡은 값을 읽어 4→2→8처럼 튄다(실측). 화면에 실제로 적용한 값을
        // 들고 있으면 연속 조작이 언제나 한 단계씩 움직인다.
        const cur = this.laneWindow;
        const next = e.deltaY < 0
          ? Math.max(0.5, cur / 2)   // 휠 위로 = 확대(표시 구간 축소)
          : Math.min(16, cur * 2);   // 휠 아래로 = 축소(표시 구간 확장)
        if (next === cur) return;
        // 저장 왕복을 기다리지 않고 그 자리에서 먼저 반영한다 — 일시정지 중엔 tick이
        // 없어 다음 프레임을 기다리면 «안 먹는» 것처럼 보인다
        this.laneWindow = next;
        this.lane.setOptions({ windowMeasures: next });
        this.syncLaneHead(); // 라벨이 저장 왕복을 기다리지 않게
        this.renderLane();
        this.callbacks.onSettingsChange({ pitchWindowMeasures: next });
        return;
      }
      if (!this.lanePaused) return;
      const view = this.lane.viewport();
      if (!view) return;
      const delta = e.deltaX !== 0 ? e.deltaX : e.shiftKey ? e.deltaY : 0;
      if (delta === 0) return;
      e.preventDefault();
      this.lane.setManualT0((this.lane.getManualT0() ?? view.t0) + delta * (view.W / view.plotW));
      this.renderLane();
    }, { passive: false });
  }

  /**
   * 이 창에서 «레인을 보고 싶다»는 설정이 켜져 있는가 — 두 크롬이 서로 다른 스위치를 본다.
   *
   * floating(유튜브 페이지 위 패널): `modMainLane`. 좁은 패널에서 레인이 가사 목록을
   *   밀어내므로 기본 꺼짐인 **옵트인 모듈**이다.
   * filled(PiP 창): `pitchGuide`(가라오케 기능 자체, 기본 켜짐). PiP는 애초에 가라오케를
   *   보려고 여는 창이라 «모듈을 켜야 나오는» 것이 아니다 — 재작업 전 PiP도 정확히 이
   *   설정으로 레인을 띄웠고, 여기서 modMainLane을 쓰면 기본값이 꺼짐이라 **PiP를 열어도
   *   가라오케가 안 나오는** 회귀가 된다(실브라우저 검증에서 실제로 잡혔다).
   *
   * 「같은 UI」에 어긋나지 않는다: 두 창 모두 같은 캔버스·같은 렌더러·같은 표시 설정을
   * 쓰고, 갈리는 것은 «이 창에 띄울지»라는 창별 취향 하나뿐이다(퀵 줄 토글도 각 창에서
   * 자기 스위치를 뒤집는다).
   */
  private laneWanted(): boolean {
    return this.chrome === 'filled' ? this.settings.pitchGuide : this.settings.modMainLane;
  }

  /**
   * 레인 표시 조건 = 위 laneWanted + 싱크 가사 화면 + 노트 데이터 있음.
   * 노트가 없는 곡(자막·LRCLIB 싱크)에서 빈 오선지만 남기지 않으려는 것으로, PiP의
   * applyPitchVisibility와 같은 규칙이다.
   */
  private applyLaneVisibility(): void {
    const show = this.laneWanted() && this.stateKind === 'synced' && this.lane.hasNotes();
    this.laneShown = show;
    this.laneWrap.style.display = show ? '' : 'none';
    this.applyLanePlacement();
    this.renderLaneNotice();
    this.syncQuickRow();
    if (show) this.renderLane(); // 켠 즉시 한 프레임 — 정지 상태에서도 빈 칸으로 남지 않게
  }

  /**
   * 레인 배치 적용 — 'left'면 가사 왼쪽 세로 열(폭 조절 가능), 'bottom'이면 가사 아래
   * 가로 띠(1.5.5까지의 배치, 무회귀 경로), 'attached'면 패널 **밖** 왼쪽에 독립적으로
   * 뜨는 부착 패널(attachPanel — this.panel의 형제, 내부 분할이 아니다).
   *
   * 엘리먼트를 옮기기만 하고 새로 만들지 않는다 — 캔버스가 바뀌면 렌더러 attach와
   * 백버퍼가 함께 날아가므로, 배치 전환마다 한 프레임이 빈 칸으로 깜빡였을 것이다.
   */
  private applyLanePlacement(): void {
    const panelW = this.panel.clientWidth || this.geometry.width;
    // filled에서는 레인이 **언제나** 부착 패널로 간다 — PiP 창의 레인 자리는 영상 왼쪽
    // 열이고(운영자 지시), 그 열을 pip.ts가 laneSlot으로 내주기 때문이다. 패널 안쪽
    // 좌측 열(mainLanePos='left')로 두면 레인이 가사창 안에 갇혀 영상 오른쪽으로 밀린다.
    // 좌표 계산(updateAttachPlacement)은 아래에서 filled를 그냥 통과시킨다 — 자리는
    // flex 열이 정하지 절대좌표가 정하지 않는다.
    const attached = this.chrome === 'filled'
      ? this.laneSlot !== null
      : this.settings.mainLanePos === 'attached';
    // 패널이 2단을 못 담을 만큼 좁으면 설정과 무관하게 가로 띠로 — LANE_TWO_COL_MIN 주석 참고.
    // attached는 패널 폭과 무관(부착 패널은 화면에 독립적으로 뜬다)하므로 이 좁음 폴백과 무관하다.
    const left = !attached && this.settings.mainLanePos !== 'bottom' && panelW >= LANE_TWO_COL_MIN;
    // 메인 표면의 «자동 접힘» — 사용자가 'left'를 골랐는데 패널이 2단을 못 담아 가로 띠로
    // 되돌아간 경우다(LANE_TWO_COL_MIN). 지금까지 조용히 일어나서 «배치 버튼이 안 먹는다»로
    // 읽혔다. PiP의 열 접힘과 같은 규약으로 한 줄 알린다 — 전이에서만 한 번.
    const laneFolded = this.chrome === 'floating' && this.settings.mainLanePos === 'left'
      && this.laneShown && panelW < LANE_TWO_COL_MIN;
    if (laneFolded && !this.laneFoldedNotified) this.notifyAutoCollapsed();
    this.laneFoldedNotified = laneFolded;
    this.mainRow.classList.toggle('lane-left', left);
    if (attached) {
      if (this.laneWrap.parentElement !== this.attachPanel) this.attachPanel.append(this.laneWrap);
      this.laneDivider.remove(); // 내부 분할 손잡이는 부착 모드에서 쓰지 않는다 — attachDivider가 대신한다
      this.laneWrap.style.width = ''; // 부착 패널 자신이 폭을 정한다(CSS flex:1)
      this.updateAttachPlacement();
    } else if (left) {
      if (this.laneWrap.parentElement !== this.mainRow || this.laneDivider.parentElement !== this.mainRow) {
        this.mainRow.prepend(this.laneWrap, this.laneDivider);
      }
      this.laneWrap.style.width = `${this.clampLaneWidth(this.settings.mainLaneWidth)}px`;
      this.laneDivider.style.display = this.laneShown ? '' : 'none';
      this.attachPanel.style.display = 'none';
    } else {
      if (this.laneWrap.parentElement !== this.panel) this.panel.insertBefore(this.laneWrap, this.footer);
      this.laneDivider.remove();
      this.laneWrap.style.width = '';
      this.attachPanel.style.display = 'none';
    }
    // 좁은 세로 열(left)이거나 부착 패널(attached, 폭이 140~560px로 역시 좁은 세로 열)일
    // 때만 압축 렌더 — 가로 띠(bottom)는 폭이 넉넉하므로 예전과 완전히 같은 그림을 그린다
    this.lane.setOptions({ compact: left || attached });
  }

  /**
   * 레인 열 폭 클램프 — 하한과 **패널이 실제로 내줄 수 있는 폭**만 본다.
   *
   * 고정 상한은 없앴다(LANE_WIDTH_MIN 주석). 남은 roomCap은 취향이 아니라 물리다:
   * 패널 최소 폭이 280px이라 레인이 그 전부를 가져가면 가사 목록이 통째로 사라진다.
   */
  private clampLaneWidth(px: number): number {
    const panelW = this.panel.clientWidth || this.geometry.width;
    const roomCap = Math.max(LANE_WIDTH_MIN, panelW - LANE_BODY_MIN);
    return Math.round(Math.min(Math.max(px, LANE_WIDTH_MIN), roomCap));
  }

  /** 부착 패널 폭 클램프 — 하한 + «화면 밖 금지» 물리 상한. 패널에서 깎이는 내부 열이
   *  아니라는 것이 애초에 부착 모드를 만든 이유이므로 clampLaneWidth의 roomCap은 여기
   *  적용하지 않는다. */
  private clampAttachedWidth(px: number): number {
    const winW = this.win?.innerWidth ?? window.innerWidth;
    const screenCap = Math.max(ATTACH_WIDTH_MIN, winW - EDGE_MARGIN * 2);
    return Math.round(Math.min(screenCap, Math.max(ATTACH_WIDTH_MIN, px)));
  }

  /**
   * 부착 패널의 위치·크기를 지금의 geometry(메인 패널 좌표)에 맞춰 다시 계산한다.
   * mainLanePos !== 'attached'거나 레인이 안 보이거나 패널이 접혀 있으면 숨긴다
   * (visible/fullscreenHidden은 별도로 보지 않는다 — attachPanel은 this.host 안 this.panel의
   * 형제라 host 자체가 숨으면 함께 숨는다, updateHostVisibility 참고).
   *
   * 왼쪽에 놓을 공간이 없으면(화면 밖으로 나감) 패널 **오른쪽**으로 폴백한다. 그마저도
   * 공간이 없으면(창이 아주 좁음) 화면 왼쪽 끝에 맞춰 살짝 겹치더라도 보이게 둔다 —
   * 완전히 숨기면 "레인이 꺼졌다"로 오인하기 쉽고, 겹침은 사용자가 패널을 옮기면 바로
   * 풀리는 반면 숨김은 부착 모드를 껐다 켜기 전까지 원인을 알 수 없다(판단 근거).
   */
  private updateAttachPlacement(): void {
    // filled: 자리는 pip.ts가 내준 **flex 열**이 정한다. 절대좌표 로직(아래)을 태우면
    // 메인 창 geometry 기준으로 계산돼 PiP 창 구석에 뭉친다 — 그래서 여기서 갈라선다.
    // 이 창에서 할 일은 «열을 보이게 할지»뿐이고, 꺼지면 열이 공간을 반납한다.
    if (this.chrome === 'filled') {
      this.attachPanel.style.display = this.laneShown ? '' : 'none';
      this.attachPanel.style.left = '';
      this.attachPanel.style.top = '';
      this.attachPanel.style.width = '';
      this.attachPanel.style.height = '';
      // **열 자체의 표시는 창 주인이 정한다** — 좁은 창 자동 접힘과 같은 판단이라
      // 두 곳에서 쓰면 서로 덮어쓴다. 바뀐 사실만 알리고 결정은 넘긴다.
      this.onColumnsChanged?.();
      return;
    }
    const show = this.settings.mainLanePos === 'attached'
      && this.laneShown && !this.geometry.collapsed;
    if (!show) {
      this.attachPanel.style.display = 'none';
      return;
    }
    const width = this.clampAttachedWidth(this.settings.attachedLaneWidth);
    let left = this.geometry.x - width - ATTACH_GAP;
    let onRight = false;
    if (left < EDGE_MARGIN) {
      const rightLeft = this.geometry.x + this.geometry.width + ATTACH_GAP;
      if (rightLeft + width + EDGE_MARGIN <= window.innerWidth) {
        left = rightLeft;
        onRight = true;
      } else {
        left = Math.max(EDGE_MARGIN, Math.min(left, window.innerWidth - width - EDGE_MARGIN));
      }
    }
    this.attachPanel.classList.toggle('ey-attach-right', onRight);
    this.attachPanel.style.display = '';
    this.attachPanel.style.width = `${width}px`;
    this.attachPanel.style.height = `${this.geometry.height}px`;
    this.attachPanel.style.left = `${left}px`;
    this.attachPanel.style.top = `${this.geometry.y}px`;
  }

  /**
   * 재생목록 부착 패널의 위치·크기 — updateAttachPlacement와 **대칭** 규약이지만
   * 우선순위가 반대다: 레인은 왼쪽 우선(폴백 오른쪽)인데, 이쪽은 패널 **오른쪽**을
   * 우선하고 공간이 없으면 왼쪽으로 폴백한다(운영자 요청 2026-08-03 — 레인·재생목록을
   * 동시에 켜도 서로 반대편에 서게 해 겹칠 확률을 줄인다). 그마저 안 되면 화면 오른쪽
   * 끝에 맞춰 클램프한다(레인 쪽과 같은 판단 근거 — 완전히 숨기는 것보다 겹침이 낫다).
   */
  private updatePlaylistPlacement(): void {
    // filled(PiP 창): «패널 밖»이 없으므로 같은 패널을 본문 오른쪽 열로 눕힌다 —
    // 부착 레인이 좌측 열로 눕는 것(applyLanePlacement)과 대칭이고, DOM·조각은 그대로라
    // 두 창에 뜨는 재생목록이 한 픽셀도 갈라지지 않는다.
    if (this.chrome === 'filled') {
      // filled: 가사창 오른쪽 **열**이 자리다(pip.ts가 playlistSlot으로 내준다).
      // 절대좌표는 여기서도 금물 — 좌표를 갖던 시절의 인라인 스타일을 반드시 걷어낸다.
      // 열 표시 여부는 위 laneSlot과 같은 이유로 창 주인이 정한다.
      this.attachPlaylistPanel.style.left = '';
      this.attachPlaylistPanel.style.top = '';
      this.attachPlaylistPanel.style.width = '';
      this.attachPlaylistPanel.style.height = '';
      // playlistVisible()이어야 한다 — modPlaylist를 직접 읽으면 브로드캐스트 때
      // filled 인스턴스가 메인 키를 따라가 "메인에서 껐다 켜면 PiP도 껐다 켜지는"
      // 표면 동기화 버그가 된다(운영자 실확인 P1, 2026-08-04).
      this.attachPlaylistPanel.style.display = this.playlistVisible() ? '' : 'none';
      this.onColumnsChanged?.();
      return;
    }
    const show = this.playlistVisible() && !this.geometry.collapsed;
    if (!show) {
      this.attachPlaylistPanel.style.display = 'none';
      return;
    }
    const width = PLAYLIST_PANEL_WIDTH;
    let left = this.geometry.x + this.geometry.width + ATTACH_GAP;
    let onLeft = false;
    if (left + width + EDGE_MARGIN > window.innerWidth) {
      const leftLeft = this.geometry.x - width - ATTACH_GAP;
      if (leftLeft >= EDGE_MARGIN) {
        left = leftLeft;
        onLeft = true;
      } else {
        left = Math.max(EDGE_MARGIN, Math.min(left, window.innerWidth - width - EDGE_MARGIN));
      }
    }
    this.attachPlaylistPanel.classList.toggle('ey-attach-left', onLeft);
    this.attachPlaylistPanel.style.display = '';
    this.attachPlaylistPanel.style.width = `${width}px`;
    this.attachPlaylistPanel.style.height = `${this.geometry.height}px`;
    this.attachPlaylistPanel.style.left = `${left}px`;
    this.attachPlaylistPanel.style.top = `${this.geometry.y}px`;
  }

  /**
   * 레인/가사 경계 드래그 손잡이 — PiP의 buildDivider(영상 비율)와 같은 규약이다:
   * 드래그 중에는 화면만 즉시 따라가고, **떼는 순간 한 번만** 설정에 저장한다
   * (매 pointermove마다 저장하면 chrome.storage 쓰기가 초당 수십 번 발생한다).
   */
  private buildLaneDivider(): HTMLDivElement {
    const divider = h('div', {
      className: 'ey-lane-divider',
      title: t('overlay.mainLane.resizeTitle'),
    }, h('div', { className: 'ey-lane-divider-grip' }));
    let dragging = false;
    divider.addEventListener('pointerdown', (e: PointerEvent) => {
      dragging = true;
      divider.setPointerCapture(e.pointerId);
      e.preventDefault();
    });
    divider.addEventListener('pointermove', (e: PointerEvent) => {
      if (!dragging) return;
      const width = this.clampLaneWidth(e.clientX - this.panel.getBoundingClientRect().left);
      this.laneWrap.style.width = `${width}px`;
      // 캔버스 백버퍼는 렌더 앞머리에서 clientWidth×devicePixelRatio로 다시 잡힌다
      // (pitch-lane.render) — 폭을 바꾼 직후 한 프레임을 더 그려야 그 자리에서 선명해진다
      this.renderLane();
    });
    divider.addEventListener('pointerup', (e: PointerEvent) => {
      if (!dragging) return;
      dragging = false;
      divider.releasePointerCapture(e.pointerId);
      const width = parseInt(this.laneWrap.style.width, 10);
      if (Number.isFinite(width)) this.callbacks.onSettingsChange({ mainLaneWidth: width });
    });
    return divider;
  }

  /**
   * 부착 패널 자신의 폭 조절 손잡이 — **왼쪽** 모서리에 붙는다(오른쪽 모서리는 메인
   * 패널과의 간격에 고정돼 있어야 하므로, 왼쪽으로 끌수록 넓어진다). buildLaneDivider와
   * 같은 규약: 드래그 중엔 화면만, 떼는 순간에만 저장.
   */
  private buildAttachDivider(): HTMLDivElement {
    const divider = h('div', {
      className: 'ey-attach-divider',
      title: t('overlay.mainLane.resizeTitle'),
    }, h('div', { className: 'ey-lane-divider-grip' }));
    let dragging = false;
    let startX = 0;
    let startWidth = 0;
    divider.addEventListener('pointerdown', (e: PointerEvent) => {
      dragging = true;
      startX = e.clientX;
      startWidth = this.clampAttachedWidth(this.settings.attachedLaneWidth);
      divider.setPointerCapture(e.pointerId);
      e.preventDefault();
    });
    divider.addEventListener('pointermove', (e: PointerEvent) => {
      if (!dragging) return;
      const width = this.clampAttachedWidth(startWidth + (startX - e.clientX));
      this.attachPanel.style.width = `${width}px`;
      // 오른쪽 모서리가 메인 패널에 계속 붙어 있으려면 폭이 바뀔 때마다 left도 다시 계산해야 한다
      const onRight = this.attachPanel.classList.contains('ey-attach-right');
      this.attachPanel.style.left = onRight
        ? `${this.geometry.x + this.geometry.width + ATTACH_GAP}px`
        : `${Math.max(EDGE_MARGIN, this.geometry.x - width - ATTACH_GAP)}px`;
      this.renderLane();
    });
    divider.addEventListener('pointerup', (e: PointerEvent) => {
      if (!dragging) return;
      dragging = false;
      divider.releasePointerCapture(e.pointerId);
      const width = parseInt(this.attachPanel.style.width, 10);
      if (Number.isFinite(width)) this.callbacks.onSettingsChange({ attachedLaneWidth: width });
    });
    return divider;
  }

  /**
   * 레인 머리 컨트롤 — 가라오케 중에 손이 가는 것들만. 나머지(옥타브·밝기 등)는
   * 설정 시트에 남긴다: 곡마다 바꾸는 값이 아니라 한 번 정하고 잊는 값이다.
   *
   * 저장은 전부 onSettingsChange 한 경로다(새 축을 만들지 않는다는 규약) — 설정 시트의
   * 같은 항목을 만지는 것과 완전히 같은 결과가 된다.
   */
  private buildLaneHead(): HTMLDivElement {
    const set = (patch: Partial<Settings>): void => this.callbacks.onSettingsChange(patch);
    const btn = (cls: string, title: string, onClick: () => void): HTMLButtonElement =>
      h('button', {
        className: `ey-lane-head-btn ${cls}`, title, attrs: { type: 'button' },
        on: { click: onClick },
      });

    this.laneMelodyBtn = btn('', t('pip.controls.melody'),
      () => set({ melodyPlayback: !this.settings.melodyPlayback }));
    this.laneMelodyBtn.append(icon(LANE_NOTE_SVG));
    this.laneMetroBtn = btn('', t('pip.controls.metronome'),
      () => set({ metronome: !this.settings.metronome }));
    this.laneMetroBtn.append(icon(LANE_METRO_SVG));
    // 메트로놈 세부(배속·시작 박)는 메트로놈이 켜져 있을 때만 뜻이 있다 — syncLaneHead가 가린다
    this.laneMetroRateBtn = btn('ey-lane-head-text', t('pip.controls.metronomeRate'), () => {
      const r = this.settings.metronomeRate;
      set({ metronomeRate: r === 1 ? 2 : r === 2 ? 0.5 : 1 });
    });
    this.laneMetroBeatBtn = btn('ey-lane-head-text', t('pip.controls.metronomeBeat'),
      () => set({ metronomeBeat: (this.settings.metronomeBeat + 1) % 4 }));

    // 마디 창 ± — 휠 확대축소와 **같은 축·같은 클램프**(laneWindow가 화면 정본)
    const minus = btn('ey-lane-head-text', t('pip.controls.windowMinus'), () => {
      const next = Math.max(0.5, this.laneWindow / 2);
      if (next === this.laneWindow) return;
      this.laneWindow = next;
      this.lane.setOptions({ windowMeasures: next });
      this.renderLane();
      set({ pitchWindowMeasures: next });
    });
    minus.textContent = '−';
    const plus = btn('ey-lane-head-text', t('pip.controls.windowPlus'), () => {
      const next = Math.min(16, this.laneWindow * 2);
      if (next === this.laneWindow) return;
      this.laneWindow = next;
      this.lane.setOptions({ windowMeasures: next });
      this.renderLane();
      set({ pitchWindowMeasures: next });
    });
    plus.textContent = '+';
    this.laneWindowLabel = btn('ey-lane-head-text', t('pip.controls.windowLabel'), () => {});
    this.laneWindowLabel.disabled = true;

    this.laneModeBtn = btn('ey-lane-head-text', t('pip.controls.modeToggle'),
      () => set({ pitchScrollMode: this.settings.pitchScrollMode === 'page' ? 'scroll' : 'page' }));
    this.laneSolfegeBtn = btn('ey-lane-head-text', t('overlay.settings.row.solfegeNotation'),
      () => set({ solfegeNotation: this.settings.solfegeNotation === 'korean' ? 'english' : 'korean' }));
    this.laneCountBtn = btn('ey-lane-head-text', t('overlay.settings.row.pitchCountdown'),
      () => set({ pitchCountdown: !this.settings.pitchCountdown }));

    return h('div', { className: 'ey-lane-head' },
      this.laneMelodyBtn, this.laneMetroBtn, this.laneMetroRateBtn, this.laneMetroBeatBtn,
      minus, this.laneWindowLabel, plus, this.laneModeBtn,
      this.laneSolfegeBtn, this.laneCountBtn);
  }

  /** 레인 머리 컨트롤의 라벨·on 상태를 현재 설정에 맞춘다 */
  private syncLaneHead(): void {
    const s = this.settings;
    this.laneMelodyBtn.classList.toggle('on', s.melodyPlayback);
    this.laneMetroBtn.classList.toggle('on', s.metronome);
    // 배속·시작 박은 메트로놈이 꺼져 있으면 조절할 대상이 없다
    this.laneMetroRateBtn.style.display = s.metronome ? '' : 'none';
    this.laneMetroBeatBtn.style.display = s.metronome ? '' : 'none';
    this.laneMetroRateBtn.textContent = s.metronomeRate === 0.5 ? '½×' : `${s.metronomeRate}×`;
    this.laneMetroBeatBtn.textContent = t('overlay.settings.metronomeBeat.n', [String(s.metronomeBeat + 1)]);
    const m = this.laneWindow;
    this.laneWindowLabel.textContent = m === 0.5
      ? t('overlay.settings.pitchWindow.half')
      : t('overlay.settings.pitchWindow.bars', [String(m)]);
    this.laneModeBtn.textContent = s.pitchScrollMode === 'page'
      ? t('pip.controls.modeFixed') : t('pip.controls.modeScroll');
    this.laneSolfegeBtn.textContent = s.solfegeNotation === 'korean' ? '도레미' : 'CDE';
    this.laneCountBtn.textContent = '4·3·2·1';
    this.laneCountBtn.classList.toggle('on', s.pitchCountdown);
  }

  private miniButton(svg: string, title: string, onClick: () => void): HTMLButtonElement {
    return h('button', {
      className: 'ey-mini', title, attrs: { type: 'button' }, on: { click: onClick },
    }, icon(svg));
  }

  /** 퀵 토글 줄의 on/off 상태·툴팁을 현재 설정에 맞춘다 (표시 언어 변경도 여기서 흡수) */
  private syncQuickRow(): void {
    this.quickLaneBtn.classList.toggle('on', this.laneWanted());
    this.quickLaneBtn.title = t('overlay.quick.lane');
    this.quickCaptionBtn.classList.toggle('on', this.settings.videoCaptions);
    this.quickCaptionBtn.title = t('overlay.quick.caption');
    this.quickPlaylistBtn.classList.toggle('on', this.playlistVisible());
    this.quickPlaylistBtn.title = t('overlay.quick.playlist');
    // 배치 버튼은 "지금 어디에 있는가"가 아니라 "누르면 어디로 가는가"를 말한다 —
    // 아이콘은 현재 배치를 그리고 툴팁이 목적지를 밝힌다.
    if (this.chrome === 'filled') {
      // PiP: 부착 개념이 없어 순환이 무의미하다 → 레인 열 ⇄ 중앙 열 맞바꾸기.
      // 아이콘도 «지금 레인이 어느 쪽인가»를 그려 눌렀을 때의 결과가 읽히게 한다.
      const swapped = this.settings.pipLaneSwapped;
      this.quickLanePosBtn.replaceChildren(icon(swapped ? MINI_POS_RIGHT_SVG : MINI_POS_LEFT_SVG));
      this.quickLanePosBtn.title = t('overlay.quick.laneSwap');
      this.quickLanePosBtn.classList.toggle('on', swapped);
    } else {
      const pos = this.settings.mainLanePos;
      const POS_ICON: Record<Settings['mainLanePos'], string> = {
        left: MINI_POS_LEFT_SVG, bottom: MINI_POS_BOTTOM_SVG, attached: MINI_POS_ATTACHED_SVG,
      };
      const POS_NEXT_TITLE: Record<Settings['mainLanePos'], string> = {
        left: t('overlay.quick.lanePosToBottom'),
        bottom: t('overlay.quick.lanePosToAttached'),
        attached: t('overlay.quick.lanePosToLeft'),
      };
      this.quickLanePosBtn.replaceChildren(icon(POS_ICON[pos]));
      this.quickLanePosBtn.title = POS_NEXT_TITLE[pos];
    }
    this.quickLanePosBtn.style.display = this.laneWanted() ? '' : 'none';
  }

  /**
   * 「폭이 모자라 자동으로 접혔다」를 사용자에게 알린다 — **이 표면의 알림 칩**으로.
   *
   * 사용자가 펼침 버튼을 눌렀는데 아무 일도 안 일어난 것처럼 보이는 상황이 이 지시의
   * 출발점이다(운영자). 새 UI를 만들지 않고 이미 있는 한 줄 칩을 그대로 쓴다 —
   * 어떤 화면 위에도 뜨고, 자동으로 사라지며, 두 창이 각자 자기 것을 갖고 있다.
   */
  notifyAutoCollapsed(): void {
    this.setNoticeChip(t('overlay.notice.autoCollapsed'), 7000);
  }

  /**
   * 음절 타이밍 안내 배너 — fast/medium 깊이에서 음절이 뭉치는 것은 정렬 실패가 아니라
   * 그 깊이의 한계다. 사용자가 "레인이 고장났다"로 읽지 않도록 레인 안에서 원인과
   * 해결책(분석 깊이 올리기)을 한 줄로 알린다.
   *
   * heavy에서는 띄우지 않는다 — 더 올릴 곳이 없는데 올리라고 말하면 그건 그냥 소음이다.
   * 깊이는 헤더의 깊이 버튼과 **같은 출처**(debugMeta.routing.route)를 읽는다.
   */
  private renderLaneNotice(): void {
    const route = this.debugMeta?.routing?.route;
    const label = route === 'fast' ? t('overlay.depthLabel.fast')
      : route === 'medium' ? t('overlay.depthLabel.medium')
      : null;
    // 구싱크(라우팅 메타 없음)는 route가 undefined라 label도 null이 된다 — 예전엔 그 경우
    // 배너가 아예 안 떴다(fast/medium 곡의 실사용 미표시 원인 후보). heavy는 계속 억제하되,
    // "깊이를 모른다"와 "heavy라 올릴 곳이 없다"는 구분해야 한다 — meta 자체가 없거나
    // route가 fast/medium/heavy 어디에도 속하지 않을 때만 깊이 표기 없는 문구로 띄운다.
    const knownNonUpgradable = route === 'heavy';
    const show = this.laneShown && !knownNonUpgradable
      && !this.settings.karaokeTimingNoticeDismissed && !this.timingNoticeHidden;
    if (!show) {
      this.laneNotice.style.display = 'none';
      this.laneNotice.replaceChildren();
      return;
    }
    const text = label !== null
      ? t('overlay.laneNotice.text', [label])
      : t('overlay.laneNotice.textNoDepth');
    this.laneNotice.replaceChildren(
      h('span', { className: 'ey-lane-notice-text', text }),
      h('div', { className: 'ey-lane-notice-actions' },
        h('button', {
          className: 'ey-lane-notice-btn',
          text: t('overlay.laneNotice.close'),
          attrs: { type: 'button' },
          on: {
            click: () => {
              this.timingNoticeHidden = true;
              this.renderLaneNotice();
            },
          },
        }),
        h('button', {
          className: 'ey-lane-notice-btn',
          text: t('overlay.laneNotice.never'),
          title: t('overlay.laneNotice.neverTitle'),
          attrs: { type: 'button' },
          on: {
            click: () => {
              this.timingNoticeHidden = true;
              this.callbacks.onSettingsChange({ karaokeTimingNoticeDismissed: true });
              this.renderLaneNotice();
            },
          },
        }),
      ),
    );
    this.laneNotice.style.display = '';
  }

  /**
   * 보컬 존재 구간 글로우 — 서버가 내려준 발성 구간(debugMeta.vad_regions) 안에서 패널
   * 테두리가 은은하게 밝아진다(웅웅거리는 소프트 펄스는 CSS 애니메이션이 담당, 여기는
   * 켜고 끄기만). 구간 데이터가 없으면(구서버·자막 싱크) 아무 일도 하지 않는다 — 효과가
   * 없는 것이지 오류가 아니다.
   */
  private updateVocalGlow(time: number): void {
    const want = this.settings.vocalGlow
      && this.stateKind === 'synced'
      && (this.debugMeta?.vad_regions?.some(([s, e]) => time >= s && time < e) ?? false);
    if (want !== this.vocalGlowOn) {
      this.vocalGlowOn = want;
      this.panel.classList.toggle('ey-vocal-glow', want);
    }
  }

  /** 한 줄의 글자·음절을 전부 채우거나 전부 비운다 */
  private setLineFilled(i: number, filled: boolean): void {
    const el = this.lineEls[i];
    if (!el) return;
    setElFilled(el, filled);
  }

  /**
   * 앞선 `target`개 줄이 전부 채워진 상태로 맞춘다 (그 뒤는 비운다).
   *
   * `filledUpTo`로 현재 경계를 들고 있어 매번 전 줄을 훑지 않는다 — 위치가 한 줄
   * 움직이면 한 줄만 칠하거나 지운다. 되감기·건너뛰기처럼 여러 줄을 뛰면 그 구간만
   * 처리한다.
   */
  private fillUpTo(target: number): void {
    while (this.filledUpTo < target) {
      this.setLineFilled(this.filledUpTo, true);
      this.filledUpTo++;
    }
    while (this.filledUpTo > target) {
      this.filledUpTo--;
      this.setLineFilled(this.filledUpTo, false);
    }
  }

  // ── 외부 상태 주입 ─────────────────────────────────────────────

  setSong(song: SongInfo | null): void {
    this.lastSong = song;
    if (song) {
      this.songTitleEl.textContent = song.title;
      this.songTitleEl.title = song.title;
      this.songArtistEl.textContent = song.artist ?? '';
    } else {
      this.songTitleEl.textContent = t('overlay.detecting');
      this.songTitleEl.title = '';
      this.songArtistEl.textContent = '';
    }
  }

  /**
   * 패널 표시/숨김 — «떠 있는 패널»에만 뜻이 있다.
   *
   * filled는 창이 곧 패널이라 숨기면 빈 창만 남는다. content가 곡 없는 페이지로 이동할 때
   * 거는 setVisible(false)가 PiP 창까지 비워서는 안 되므로(그 창은 사용자가 닫기 전까지
   * 살아 있다는 것이 PiP의 설계다) 여기서 스스로 막는다 — 호출부는 방송만 하고 «어느
   * 인스턴스인지»를 따지지 않아도 된다.
   */
  setVisible(visible: boolean): void {
    if (this.chrome === 'filled') return;
    this.visible = visible;
    this.updateHostVisibility();
  }

  isVisible(): boolean {
    return this.visible;
  }

  /**
   * 서버 상태 주입 — 사유까지 함께 받는다.
   *
   * 서버가 필요한 컨트롤(생성·재생성)을 잠그고 사유를 툴팁으로 붙이며, 배너를 갱신한다.
   * 지금 화면이 "가사를 찾지 못했어요"라면 그것도 서버 문제 화면으로 바꿔 준다 —
   * 상태 확인이 검색보다 늦게 끝나 잘못된 문구가 먼저 떠 있을 수 있기 때문이다.
   */
  setServerStatus(status: ServerStatus): void {
    const prevKind = this.serverStatus.kind;
    this.serverStatus = status;
    this.generateButtons = this.generateButtons.filter(btn => btn.isConnected);
    for (const btn of this.generateButtons) applyServerGate(btn, status);
    this.applyResetSyncGate();
    this.renderServerBar();

    if (this.settingsDot) {
      this.applyDotClasses(this.settingsDot, status);
      this.settingsDot.title = t('overlay.settings.serverStatusTitle', [statusLine(status)]);
    }
    // 설정 시트가 열린 채로 상태가 바뀔 수 있다 (URL을 고치면 곧바로 재확인이 돌아온다)
    if (this.settingsPermBtn) {
      this.settingsPermBtn.style.display = needsHostPermission(status) ? '' : 'none';
    }
    if (prevKind !== status.kind && this.stateKind === 'empty') {
      // 설정 시트에서 키를 고치던 중일 수 있다 — 화면은 다시 그리되 시트는 되살린다
      // (resetBody가 시트를 닫는다). 시트는 저장된 설정으로 새로 만들어진다.
      const settingsWasOpen = this.settingsSheet !== null;
      this.showEmpty(this.lastSong);
      if (settingsWasOpen) this.openSettings();
    }
  }

  /** 설정 시트의 상태 점 색 — ok(초록) / auth(주황) / permission(파랑) / 그 밖(빨강) */
  private applyDotClasses(dot: HTMLSpanElement, status: ServerStatus): void {
    dot.classList.toggle('ok', status.kind === 'ok');
    dot.classList.toggle('auth', status.kind === 'auth');
    dot.classList.toggle('perm', status.kind === 'permission');
  }

  /** 서버가 필요한 헤더 버튼(재생성) 잠금 — 표시 여부는 기존 로직 그대로 */
  private applyResetSyncGate(): void {
    applyServerGate(this.resetSyncBtn, this.serverStatus, t('overlay.header.resetSync'));
  }

  private renderServerBar(): void {
    const bar = buildServerStatusSlot(this.panelContext());
    if (!bar) {
      this.serverBar.replaceChildren();
      this.serverBar.style.display = 'none';
      return;
    }
    this.serverBar.replaceChildren(bar);
    this.serverBar.style.display = '';
  }

  /** PiP 열기 버튼 노출 — filled 인스턴스는 «이미 그 PiP 창 안»이라 언제나 숨는다 */
  setPipEnabled(enabled: boolean): void {
    this.pipEnabled = enabled && this.chrome === 'floating';
    this.pipBtn.style.display = this.pipEnabled && this.stateKind === 'synced' ? '' : 'none';
  }

  setPipActive(active: boolean): void {
    this.pipBtn.classList.toggle('active', active);
  }

  isShowingPipPlaceholder(): boolean {
    return this.stateKind === 'pip';
  }

  /**
   * lines[].translation을 다시 읽어 각 라인 아래 번역을 갱신/제거하고, 발음 표기도
   * 항상 새로 그린다(무조건 지우고 buildPronEl로 재구성) — 번역 API가 발음을 늦게
   * 채워주는 경우뿐 아니라, 발음 표기 방식(pronunciationScript/translationLanguage)이
   * 바뀌었을 때도 이미 그려진 줄이 새 표기를 반영해야 한다(감사 #8 — 예전엔 이미 그린
   * 줄의 .ey-line-pron이 있으면 건드리지 않아서, 서버가 표기별 발음(pron dict)을 아직
   * 안 주던 시절의 전제("화면상 차이 없음")가 다국어 배포 이후 거짓이 됐는데도 메인
   * 패널이 pip.setPronScript만큼 따라가지 못했다).
   */
  refreshTranslations(): void {
    const pronScript = resolveScript(this.settings);
    this.lineEls.forEach((el, i) => {
      el.querySelector('.ey-line-tr')?.remove();
      el.querySelector('.ey-line-pron')?.remove();
      const line = this.lines[i];
      if (line) {
        const pronEl = buildPronEl(line, pronScript, this.settings);
        if (pronEl) el.append(pronEl);
      }
      if (line?.translation) el.append(h('div', { className: 'ey-line-tr', text: line.translation, attrs: { dir: 'auto' } }));
    });
    // 레인의 노트 부착 발음은 setLines 시점에 계산된다 — 번역·발음이 뒤늦게 붙은 라인을
    // 반영하려면 같은 배열로 다시 한 번 태워야 한다(라인 객체는 그대로라 비용은 평탄화뿐)
    this.lane.setLines(this.lines);
    this.lane.setIndex(this.currentIndex);
    this.applyLaneVisibility();
    // 활성 줄의 .ey-line-pron을 방금 통째로 새로 지었다 — highlightLine이 만들어 둔
    // activeWordEls는 이제 DOM에서 떨어져 나간 옛 스팬을 가리킨다. 재수집하지 않으면
    // 언어 전환 직후 활성 줄의 카라오케 채움(sung)이 다음 줄 전환까지 한 줄 비어
    // 보인다(감사 C8c).
    if (this.currentIndex >= 0 && this.lineEls[this.currentIndex]) {
      this.activeWordEls = collectFillTargets(this.lineEls[this.currentIndex]);
    }
    this.renderShortView(); // 단축 표시도 같은 줄 객체를 보므로 함께 다시 짓는다
  }

  setTranslationStatus(text: string | null): void {
    this.trStatusEl.textContent = text ?? '';
  }

  /** 낮은 정렬 신뢰도 경고 바 — score가 null이면 숨김. X로 닫을 수 있다. */
  setQualityWarning(score: number | null): void {
    this.warnScore = score;
    this.warnExpanded = false; // 곡이 바뀌면 접힘 상태로 되돌린다
    this.warnClosedByUser = false; // 새 컨텍스트(곡·설정 변경)는 이전 × 닫음을 승계하지 않는다
    this.renderWarnBar();
  }

  /**
   * 경고 바 — 짧은 줄(항상 보임) + "자세히"로 펼치는 안내문. 예전엔 안내(가사 원문 확인·
   * 분석 깊이 올리기 제안)가 툴팁에만 있어 실제로 읽히지 않았다(운영자 지시 2026-08-03:
   * 본문 노출로 승격). ×는 이번 세션만 닫고, "이 곡에서 다시 보지 않기"는 영상별로
   * 영구히 억제한다(content가 chrome.storage에 적는다) — 둘의 되돌릴 수 있는 정도가
   * 달라 같은 버튼으로 합치지 않는다.
   *
   * **깊이별 톤 분기(운영자 실사용 제보 2026-08-04)**: fast로 **저장된** 싱크는 이미
   * 라우터를 통과한 것이다 — fast 결과의 라인 신뢰도 중앙값이 임계 미달이면 서버가
   * 자동으로 heavy까지 승급시키므로, fast 저장본은 "라우터가 품질을 확인한 곡"이다.
   * 반면 이 배너가 읽는 절대 신뢰도 수치(qualityScore, 0.00026류)는 정상-어려운 곡과
   * 사고 곡이 겹치는 대리 지표라 그 수치만으로 매번 느낌표+노란색 경고를 내면 정상
   * 곡까지 과잉 경보가 된다. fast로 확인되면 같은 정보(신뢰도 수치)를 느낌표·노란
   * 배경 없이 중립 톤으로만 보여주고, medium·heavy(라우터가 fast로 못 끝낸 곡)는
   * 기존 경고 스타일을 그대로 유지한다 — 그 경우엔 절대 수치가 진짜 위험 신호다.
   * 깊이는 currentDepth()(헤더 깊이 버튼·레인 안내와 같은 출처)로 읽는다.
   */
  private renderWarnBar(): void {
    const score = this.warnScore;
    if (score === null || this.warnClosedByUser) {
      this.warnBar.style.display = 'none';
      this.warnBar.replaceChildren();
      return;
    }
    // depth===null(메타 미도착·구세대)은 라우터 통과를 확인할 수 없으므로 안전하게
    // 경고 톤을 유지한다 — setDebugMeta가 뒤늦게 'fast'를 확정하면 이 함수를 다시 불러
    // 그때 중립으로 내려간다(경고→중립 방향만 있고, 반대는 없다 — score 자체가 fast
    // 승급 로직을 통과한 뒤에야 정해지므로 medium/heavy가 뒤늦게 fast로 내려갈 일은 없다).
    const isFastVerified = this.currentDepth() === 'fast';
    this.warnBar.classList.toggle('ey-warn-neutral', isFastVerified);
    const headText = isFastVerified
      ? t('overlay.warn.textNeutral', [fmtConf(score)])
      : `⚠️ ${t('overlay.warn.text', [fmtConf(score)])}`;
    const head = h('div', { className: 'ey-warn-head' },
      h('span', { className: 'ey-warn-text', text: headText }),
      h('button', {
        className: 'ey-warn-expand',
        text: this.warnExpanded ? t('overlay.warn.collapse') : t('overlay.warn.expand'),
        title: isFastVerified ? t('overlay.warn.titleNeutral') : t('overlay.warn.title'),
        attrs: { type: 'button' },
        on: { click: () => { this.warnExpanded = !this.warnExpanded; this.renderWarnBar(); } },
      }),
      h('button', {
        className: 'ey-warn-close',
        text: '×',
        title: t('overlay.warn.close'),
        attrs: { type: 'button' },
        on: {
          click: () => {
            this.warnClosedByUser = true;
            this.warnBar.style.display = 'none';
          },
        },
      }),
    );
    const children: HTMLElement[] = [head];
    if (this.warnExpanded) {
      children.push(h('div', { className: 'ey-warn-detail' },
        h('span', {
          className: 'ey-warn-detail-text',
          text: isFastVerified ? t('overlay.warn.detailNeutral') : t('overlay.warn.detail'),
        }),
        h('button', {
          className: 'ey-warn-dismiss-song',
          text: t('overlay.warn.dismissSong'),
          title: t('overlay.warn.dismissSongTitle'),
          attrs: { type: 'button' },
          on: {
            click: () => {
              this.warnClosedByUser = true;
              this.warnBar.style.display = 'none';
              this.callbacks.onWarnDismissSong();
            },
          },
        }),
      ));
    }
    this.warnBar.replaceChildren(...children);
    this.warnBar.style.display = '';
  }

  /** 영상별 저장 오프셋을 UI에 반영 (설정 전역값과 분리된 per-video 상태) */
  setOffsetValue(offsetSec: number): void {
    this.offsetSec = offsetSec;
    this.updateOffsetLabel();
  }

  /** 전사 진행 칩 — null이면 숨김. 패널 본문을 점유하지 않는 작은 표시.
   *  cancellable이면 ✕ 버튼으로 진행 중인 전사를 취소할 수 있다 (현재 영상 잡만). */
  setGenerationChip(text: string | null, cancellable = false): void {
    if (!text) {
      this.genChip.style.display = 'none';
      this.genList.style.display = 'none';
      return;
    }
    this.genChip.replaceChildren(icon(ICONS.sparkle), text);
    if (cancellable) {
      this.genChip.append(h('button', {
        className: 'ey-gen-chip-cancel',
        text: '×',
        title: t('overlay.genChip.cancel'),
        on: {
          click: (e) => {
            e.stopPropagation(); // 칩의 대기열 목록 토글로 새지 않게
            const btn = e.currentTarget as HTMLElement;
            if (this.confirmTwice(btn, t('overlay.genChip.cancelConfirm'))) {
              this.callbacks.onCancelGenerate();
            }
          },
        },
      }));
    }
    this.genChip.style.display = '';
    this.renderGenList();
  }

  /**
   * 한 줄 알림 칩 — null이면 숨김.
   *
   * 쓰는 곳: 커버 자동 연결("동일 곡 추정 — 자동 연결 확인 중" → "자동 연결됨"),
   * 붙여넣기 표기 필터 결과. 패널 본문을 점유하지 않아 어떤 화면 위에서도 뜬다.
   * autoHideMs를 주면 그 뒤 스스로 사라진다 (마지막 호출이 이긴다 — 알림은 상태가
   * 아니라 사건이므로 겹치면 새 소식을 보여주는 편이 맞다).
   */
  setNoticeChip(text: string | null, autoHideMs?: number): void {
    clearTimeout(this.noticeTimer);
    this.pendingNotice = null;
    if (!text) {
      this.hideNoticeChip();
      return;
    }
    this.noticeChip.replaceChildren(icon(ICONS.sparkle), text);
    this.noticeChip.title = text; // 칩이 좁아 잘려도 전문을 볼 수 있게
    this.noticeChip.style.display = '';
    if (autoHideMs !== undefined) {
      // 전체화면 중에는 이 칩이 화면에 나갈 수 없다(pendingNotice 주석의 top layer 근거) —
      // 타이머를 걸어 두면 사용자가 아무것도 못 본 채 만료된다. 해제될 때 처음부터 센다.
      if (this.fullscreenHidden) this.pendingNotice = { text, autoHideMs };
      else this.noticeTimer = window.setTimeout(() => this.hideNoticeChip(), autoHideMs);
    }
  }

  private hideNoticeChip(): void {
    this.noticeChip.style.display = 'none';
    this.noticeChip.replaceChildren();
  }

  /** 내 생성 대기열 목록 데이터 갱신 — 진행 칩 클릭으로 펼친다.
   *  이 브라우저(activeJobs)가 시킨 잡만 들어오므로 타인 큐는 노출되지 않는다. */
  setGenerationList(items: { title: string; state: string; isCurrent: boolean }[]): void {
    this.genListItems = items;
    this.renderGenList();
  }

  private renderGenList(): void {
    const chipHidden = this.genChip.style.display === 'none';
    if (!this.genListOpen || chipHidden || this.genListItems.length === 0) {
      this.genList.style.display = 'none';
      return;
    }
    this.genList.replaceChildren(...this.genListItems.map((it) =>
      h('div', { className: `ey-gen-list-row${it.isCurrent ? ' current' : ''}` },
        h('span', {
          className: 'ey-gen-list-title',
          text: it.isCurrent ? t('overlay.genList.currentVideo', [it.title]) : it.title,
          title: it.title,
        }),
        h('span', { className: 'ey-gen-list-state', text: it.state }),
      )));
    this.genList.style.display = '';
  }

  updateDebug(info: DebugInfo): void {
    if (!this.settings.debugInfo) return;
    const timeStr = info.time === null ? '-' : `${info.time.toFixed(2)}s`;
    const off = `${info.offsetSec > 0 ? '+' : ''}${info.offsetSec.toFixed(1)}`;
    const line = info.lineCount > 0 ? `${info.lineIndex + 1}/${info.lineCount}` : '-';
    const video = info.videoInfo === 'none' ? 'none' : `${info.videoBound ? 'OK' : 'MISMATCH'}(${info.videoInfo})`;
    const g = info.confGrades;
    const diag = [
      // 이 싱크가 언제 만들어졌는가 — 수정 배포 후에도 옛 싱크는 그대로라 판단 기준이 된다
      info.syncCreated ? t('overlay.debug.created', [info.syncCreated]) : null,
      // 사람이 읽는 등급 분포 (글자 색과 동일 기준: 좋음=초록, 보통=노랑, 낮음=빨강)
      g ? t('overlay.debug.grades', [
        String(Math.round(g.ok * 100)), String(Math.round(g.mid * 100)), String(Math.round(g.low * 100)),
      ]) : null,
      info.quality != null ? `conf=${fmtConf(info.quality)}` : null,
      info.qualityMed != null ? `med=${fmtConf(info.qualityMed)}` : null,
      info.alignmentText
        ? t('overlay.debug.alignmentText', [
          info.alignmentText === 'pronunciation' ? t('overlay.debug.alignmentPronunciation') : t('overlay.debug.alignmentOriginal'),
        ])
        : null,
      info.zone ? `zone=${info.zone}` : null,
      info.lineDebug,
    ].filter(Boolean).join(' ');
    this.debugEl.textContent =
      `vid=${info.videoId ?? '-'} src=${info.source}${info.synced ? '/sync' : '/plain'} line=${line} pip=${info.pipOpen ? 'Y' : 'N'}\n`
      + `t=${timeStr} off=${off} video=${video} eng=${info.engineRunning ? 'Y' : 'N'}${info.jobStatus ? ` ${info.jobStatus}` : ''}`
      + (diag ? `\n${diag}` : '');
  }

  /** 곡 단위 정렬 진단(자막 스캐폴드 등) — 디버그 패널 머리 요약줄이 이걸 읽는다.
   *  content가 아직 이 메서드를 안 부르면 null로 남고, 패널은 요약줄만 생략한 채 동작한다. */
  /** 자동 매칭 표시줄 — 위키가 고른 곡 제목. null이면 숨김(서버 싱크·매칭 없음) */
  setMatchedSource(title: string | null): void {
    // 곡이 바뀌면 열려 있던 확인도 함께 닫는다 — 이전 곡을 겨냥한 확인이 새 곡 위에
    // 남으면, 누른 사람은 지금 곡을 제보한 줄 알지만 실제로는 다른 곡이 나간다
    this.hideWrongLyricsConfirm();
    if (title) {
      this.matchedTitleEl.textContent = title;
      this.matchedTitleEl.title = title;
      this.matchedBar.style.display = '';
    } else {
      this.matchedBar.style.display = 'none';
    }
  }

  /**
   * 다음 영상 정보 모듈 — null이면 숨김 (설정 modNextUp이 꺼져 있어도 content가 null을 준다).
   *
   * 문자열 한 줄만 받던 옛 API도 그대로 받는다 — content가 유튜브 다음 버튼 툴팁에서
   * 제목만 긁어 오는 경로가 아직 살아 있고, 그 경로에서도 카드는(썸네일 없이) 정상 렌더된다.
   * videoId를 함께 주면 썸네일이 붙는다.
   */
  setNextUp(info: string | NextUpInfo | null): void {
    this.nextUpInfo = typeof info === 'string' ? { title: info } : info;
    // 메인 패널 하단 전용 카드는 제거됐다(2026-08-04, 운영자 지시) — 재생목록이 없는
    // 페이지(단일 영상)에서는 부착 패널의 폴백 카드가 이 값을 재사용해 보여준다.
    this.renderPlaylistPanel();
  }

  /** 다음 영상 카드 본체(썸네일 + 제목 + 채널) — 재생목록 부착 패널의 폴백(목록이 없는
   *  단일 영상 페이지)에서 쓴다. 클릭하면 그 영상으로 이동한다(카드 전체가 타깃) —
   *  videoId가 있으면 content가 그 영상으로 직접, 없으면 유튜브 다음 버튼을 눌러
   *  이동한다(onNextUpClick, 실사용 제보: 카드를 눌러도 아무 일도 안 일어났다). */
  private buildNextUpCard(info: NextUpInfo): HTMLDivElement {
    const meta = h('div', { className: 'ey-nextup-meta' },
      h('div', { className: 'ey-nextup-title', text: info.title, title: info.title }),
    );
    if (info.channel) meta.append(h('div', { className: 'ey-nextup-channel', text: info.channel }));
    const card = h('div', {
      className: 'ey-nextup-card',
      attrs: { title: info.title },
      on: { click: () => this.callbacks.onNextUpClick(info.videoId) },
    });
    const thumb = info.thumbnail ?? (info.videoId ? thumbUrl(info.videoId) : null);
    if (thumb) {
      // 썸네일은 있으면 좋은 장식이다 — 404·CSP·오프라인 어느 이유로 실패하든 깨진
      // 이미지 아이콘을 남기지 말고 조용히 자리를 접는다(제목·채널은 그대로 읽힌다)
      const img = h('img', {
        className: 'ey-nextup-thumb',
        attrs: { src: thumb, alt: '', loading: 'lazy', referrerpolicy: 'no-referrer' },
      });
      img.addEventListener('error', () => img.remove());
      card.append(img);
    }
    card.append(meta);
    return card;
  }

  /**
   * 이어질 재생목록 — content가 lib/yt-player.ts로 스크랩한 항목을 그대로 밀어넣는다.
   * null/빈 배열이면 "재생목록에 속하지 않은 영상"으로 보고 다음 영상 카드로 대체한다
   * (renderPlaylistPanel). 표시 자체는 표면별 판정(playlistVisible — 메인 modPlaylist /
   * PiP pipPlaylist)을 따른다.
   */
  setPlaylist(items: PlaylistItem[] | null): void {
    this.playlistItems = items ?? [];
    this.renderPlaylistPanel();
  }

  /** 재생목록 부착 패널 — 목록이 있으면 스크롤 리스트, 없으면 다음 영상 카드 하나만 */
  private renderPlaylistPanel(): void {
    if (!this.playlistVisible() || this.geometry.collapsed) {
      this.attachPlaylistPanel.style.display = 'none';
      return;
    }
    const list = this.playlistItems;
    if (list.length > 0) {
      this.playlistStatusEl.textContent = t('overlay.playlist.status', [String(list.length)]);
      this.playlistStatusEl.title = '';
      this.playlistListEl.replaceChildren(...list.map((it, i) => this.buildPlaylistRow(it, i)));
    } else if (this.nextUpInfo) {
      this.playlistStatusEl.textContent = t('overlay.playlist.noPlaylist');
      this.playlistStatusEl.title = t('overlay.playlist.noPlaylistTitle');
      this.playlistListEl.replaceChildren(
        h('div', { className: 'ey-pl-nextup-wrap' },
          h('div', { className: 'ey-nextup-label', text: t('overlay.nextUp.label') }),
          this.buildNextUpCard(this.nextUpInfo),
        ),
      );
    } else {
      this.playlistStatusEl.textContent = t('overlay.playlist.empty');
      this.playlistStatusEl.title = '';
      this.playlistListEl.replaceChildren();
    }
    this.attachPlaylistPanel.style.display = '';
    this.updatePlaylistPlacement();
  }

  /** 재생목록 항목 한 줄 — 클릭하면 그 영상으로 이동. 서버 싱크가 있으면 작은 점 배지 */
  private buildPlaylistRow(it: PlaylistItem, i: number): HTMLDivElement {
    const row = h('div', {
      className: `ey-pl-row${it.current ? ' current' : ''}`,
      attrs: { title: it.title },
      on: { click: () => this.callbacks.onPlaylistSelect(it.index ?? i) },
    },
      h('span', { className: 'ey-pl-row-index', text: String((it.index ?? i) + 1) }),
      h('span', { className: 'ey-pl-row-title', text: it.title }),
    );
    if (it.syncExists) {
      row.append(h('span', { className: 'ey-pl-row-dot', title: t('overlay.playlist.existsTitle') }));
    }
    return row;
  }

  setDebugMeta(meta: SyncDebugMeta | null): void {
    this.debugMeta = meta;
    this.lane.setDebugMeta(meta); // 레인의 f0 곡선·VAD 스트립이 같은 메타를 쓴다
    if (this.debugPanelOpen) this.renderDebugPanel(); // 열려 있으면 요약줄도 즉시 갱신
    this.updateDepthButton(); // 깊이·세대 정보의 출처가 이 메타다
    // 음절 타이밍 안내도 같은 깊이를 읽는다 — 메타는 가사보다 늦게 도착할 수 있어,
    // 여기서 다시 판정하지 않으면 fast 싱크인데 배너가 끝내 안 뜨는 곡이 생긴다
    this.renderLaneNotice();
    // 저신뢰 경고 바도 같은 이유로 다시 그린다 — setQualityWarning은 applyLyricsData에서
    // 메타 도착보다 먼저 불려 depth가 아직 null(경고 톤 기본값)인 채로 그려질 수 있다.
    // 여기서 다시 부르면 fast가 뒤늦게 확정된 순간 경고→중립으로 자연스럽게 내려간다.
    this.renderWarnBar();
  }

  /**
   * [모듈] 레인이 쓰는 곡 단위 값 — 마디 격자(템포)와 좌상단 키·BPM 라벨.
   * PiP는 pip.setTempo/setKey로 같은 값을 받는다. 없으면 레인은 120BPM 가정으로 폴백한다.
   */
  setLaneMeta(tempo: SongTempo | null, key: SongKey | null): void {
    this.lane.setTempo(tempo);
    this.lane.setKey(key);
  }

  private toggleDebugPanel(): void {
    this.debugPanelOpen = !this.debugPanelOpen;
    this.renderDebugPanel();
  }

  private closeDebugPanel(): void {
    if (!this.debugPanelOpen) return;
    this.debugPanelOpen = false;
    this.renderDebugPanel();
  }

  private renderDebugPanel(): void {
    this.debugToggleBtn.classList.toggle('active', this.debugPanelOpen);
    if (!this.debugPanelOpen) {
      this.debugPanelEl.style.display = 'none';
      this.debugPanelEl.replaceChildren();
      return;
    }
    // SEEK_INTO_LINE_SEC 보정은 여기서 적용 — debug-panel.ts는 line.time을 그대로 받는
    // "UI만" 모듈이라 이 보정을 모른다(라인 목록의 클릭 시크와 같은 이유·같은 값)
    const { el } = buildDebugPanel(this.lines, this.debugMeta,
      time => this.callbacks.onSeek(time + SEEK_INTO_LINE_SEC),
      () => this.callbacks.onLoadPreviousSync(),
      this.callbacks.getVideoId());
    this.debugPanelEl.replaceChildren(el);
    this.debugPanelEl.style.display = '';
  }

  applySettings(settings: Settings): void {
    // 이 안에서 일어나는 리플로우는 «사용자의 창 조절»이 아니다 — 기하 갱신·저장 금지
    this.applyingSettings = true;
    const w = this.win ?? window;
    w.cancelAnimationFrame(this.settingsRaf);
    this.settingsRaf = w.requestAnimationFrame(() => { this.applyingSettings = false; });
    this.settings = settings;
    this.laneWindow = settings.pitchWindowMeasures; // 설정이 정본 — 휠이 앞서 간 값도 여기서 맞춰진다
    // 번역 언어가 설정 시트·다른 경로로 바뀌어도 제목바 칩의 "현재 선택"이 따라가야 한다
    this.renderLangChips();
    this.panel.classList.remove('ey-fs-small', 'ey-fs-medium', 'ey-fs-large');
    this.panel.classList.add(`ey-fs-${settings.fontSize}`);
    this.applyFontScale();
    // 스트리밍용 글자 외곽선 — 크로마키/영상 위에 얹힐 때 얇은 글자가 배경에 먹히지
    // 않도록 CSS가 text-shadow 링을 두른다(설정 streamTextOutline)
    this.panel.classList.toggle('ey-text-outline', settings.streamTextOutline);
    this.attachPanel.classList.toggle('ey-text-outline', settings.streamTextOutline);
    // 테마 판정은 lib/theme.ts 한 곳에서만 — PiP도 content가 같은 값을 받아 칠한다.
    // attachPanel은 this.panel의 형제라 CSS 변수를 :host에서 상속받지만, 라이트 테마
    // 오버라이드(.ey-panel.ey-light)는 클래스 스코프라 자신도 같은 클래스를 받아야 한다.
    const isLight = resolveTheme(settings) === 'light';
    this.panel.classList.toggle('ey-light', isLight);
    this.attachPanel.classList.toggle('ey-light', isLight);
    this.attachPlaylistPanel.classList.toggle('ey-light', isLight);
    // 오프셋은 영상별 상태(setOffsetValue로 주입) — 전역 설정으로 되돌리지 않는다
    this.debugStrip.style.display = settings.debugInfo ? '' : 'none';
    if (!settings.debugInfo) this.closeDebugPanel(); // 버튼이 숨는데 패널만 열려 남으면 안 된다
    this.panel.classList.toggle('ey-hide-pron', !settings.showPronunciation);
    // 디버그 모드에서 글자별 CTC 신뢰도를 색으로 표시
    this.panel.classList.toggle('ey-show-conf', settings.debugInfo);
    // 디버그 토글은 서버 요청 로그의 노출 조건이기도 하다 — 배너를 다시 그려 반영
    this.renderServerBar();
    // [모듈] 레인은 PiP와 **같은 설정 값**을 그대로 따른다 — 창마다 다른 축을 만들지
    // 않는다(마디 창·글자 크기·계이름·밝기는 사용자가 한 번만 정한다). content가 설정이
    // 바뀔 때마다 이 함수를 부르므로 별도 배선 없이 여기가 유일한 반영 지점이다.
    this.lane.setOptions({
      windowMeasures: settings.pitchWindowMeasures,
      scrollMode: settings.pitchScrollMode,
      fontScale: settings.pitchFontScale * this.widthFontScale(),
      countdown: settings.pitchCountdown,
      solfege: settings.solfegeNotation,
      lineOpacity: settings.pitchLineOpacity,
      f0Opacity: settings.pitchF0Opacity,
      pronPosition: settings.pitchPronPosition,
      pronScript: resolveScript(settings),
      showF0: settings.pitchF0Curve,
      showConfidence: settings.debugInfo,
      metronomeBeat: settings.metronomeBeat,
      // 마이크 궤적 — 예전엔 PiP 창에만 배선돼 있었다. 인스턴스가 하나의 클래스가 된
      // 지금은 여기 한 줄이 두 창을 모두 덮는다(검출이 꺼져 있으면 빈 배열이 온다).
      micOctave: settings.micOctave,
      getMicSamples: this.callbacks.getMicSamples,
    });
    this.lane.refreshColors(); // 테마가 바뀌었을 수 있다 — CSS 변수를 다시 읽게 한다
    this.syncLaneHead();
    this.applyLaneVisibility();
    this.renderPlaylistPanel();
  }

  /**
   * 패널 폭에서 유도하는 **완만한** 글자 배율 — 사용자가 정한 배율(mainFontScale·
   * pitchFontScale)에 곱해진다.
   *
   * PiP 창을 넓히면 글자도 어느 정도 따라 커져야 한다는 운영자 요구인데, 폭에 정비례로
   * 걸면 창을 조금만 늘려도 글자가 튀어 «급변»한다. 0.35제곱이면 폭이 두 배가 돼도 글자는
   * 1.27배에 그치고(360→720px), 절반이 돼도 0.78배까지만 준다 — 눈에는 «따라오는» 정도로
   * 읽히고 줄바꿈 위치가 통째로 뒤집히지는 않는다. 상·하한은 그 위의 안전장치다.
   *
   * 기준 폭 360px = 기본 패널 폭(340)과 저장 없는 PiP 기본 폭(440)의 사이 — 어느 쪽에서도
   * 처음 열었을 때 배율이 1 근처라 «갑자기 글자가 달라졌다»가 생기지 않는다.
   */
  private widthFontScale(): number {
    const w = this.panel.clientWidth || this.geometry.width;
    if (!(w > 0)) return 1;
    return Math.min(1.5, Math.max(0.8, Math.pow(w / 360, 0.35)));
  }

  /** 3단 프리셋(--ey-line-size) 위에 얹는 미세 배율 — Shadow DOM 안이라 document.body가
   *  아니라 패널 엘리먼트 자신에 심어야 overlay.css의 calc()가 이 값을 본다 */
  private applyFontScale(): void {
    this.panel.style.setProperty('--ey-main-fs', String(this.settings.mainFontScale * this.widthFontScale()));
  }

  // ── 내부 헬퍼 ─────────────────────────────────────────────────

  private headerButton(svg: string, title: string, onClick: () => void): HTMLButtonElement {
    return h('button', { className: 'ey-btn', title, on: { click: onClick } }, icon(svg));
  }

  private toggleFeedbackPop(): void {
    if (this.feedbackPop.style.display !== 'none') {
      this.feedbackPop.style.display = 'none';
      return;
    }
    this.renderFeedbackPop();
    this.feedbackPop.style.display = '';
  }

  /**
   * 지금 보고 있는 싱크의 분석 깊이 — 헤더 깊이 버튼(updateDepthButton)·레인 안내
   * 배너(renderLaneNotice)와 **같은 출처**를 읽는다. 세 곳이 다른 값을 말하면
   * 사용자는 어느 쪽을 믿어야 할지 알 수 없다.
   */
  private currentDepth(): 'fast' | 'medium' | 'heavy' | null {
    const route = this.debugMeta?.routing?.route;
    return route === 'fast' || route === 'medium' || route === 'heavy' ? route : null;
  }

  /** 별점 팝오버 — 별은 클릭 한 번으로 전송되고, 자세한 제보는 그 안에서 갈라진다 */
  private renderFeedbackPop(): void {
    const depth = this.currentDepth();
    const { el } = buildRatingPop({
      depth,
      onSubmitRating: rating => this.callbacks.onSubmitFeedback(rating),
      // 제보에도 별점 값이 필요하다 — 서버 계약이 rating 1~5(ge=1)라 "별점 없음"을
      // 보낼 방법이 없다. 오매칭 제보(content.onWrongLyrics)가 이미 쓰고 있는
      // "가사 오류 = 최저 별점" 관례를 그대로 따른다.
      onSubmitReport: (category, comment) => this.callbacks.onSubmitFeedback(1, category, comment),
      // 깊이를 올릴 곳이 남아 있을 때만 안내 버튼을 낸다 — heavy에서 '올리기'는 소음이다
      onDepthUpgrade: depth === 'fast' || depth === 'medium'
        ? () => this.callbacks.onDepthUpgrade(depth === 'fast' ? 'medium' : 'heavy')
        : undefined,
      onClose: () => { this.feedbackPop.style.display = 'none'; },
    });
    this.feedbackPop.replaceChildren(el);
  }

  /** "이 가사가 아니에요" 확인 — 한 번 더 누르면 접힌다(잘못 연 사람의 탈출구) */
  private toggleWrongLyricsConfirm(): void {
    if (this.wrongLyricsPop.style.display !== 'none') {
      this.hideWrongLyricsConfirm();
      return;
    }
    const { el } = buildWrongLyricsConfirm({
      // 무엇을 제보하는지 눈으로 확인시킨다 — 매칭 표시줄이 접히면 값도 사라지므로
      // 표시줄이 들고 있는 지금 제목을 그대로 넘긴다
      matchedTitle: this.matchedTitleEl.textContent,
      onConfirm: () => {
        this.hideWrongLyricsConfirm();
        this.callbacks.onWrongLyrics();
      },
      onCancel: () => this.hideWrongLyricsConfirm(),
    });
    this.wrongLyricsPop.replaceChildren(el);
    this.wrongLyricsPop.style.display = '';
  }

  private hideWrongLyricsConfirm(): void {
    this.wrongLyricsPop.style.display = 'none';
    this.wrongLyricsPop.replaceChildren();
  }

  /**
   * 분석 깊이 버튼 — 헤더에서 현재 싱크의 분석 깊이(1=무분리 ASR, 2=분리+ASR,
   * 3=분리+ASR+OWSM 앵커)를 화살표 나눔선·배지 숫자로 보여주고, 클릭하면 한 단계
   * 깊은 재분석(regenerate min_depth)을 요청한다. 최대 깊이(3)는 빨간 배지 + 비활성 +
   * "가사 입력 상태를 확인하세요" 툴팁. 구세대 싱크(engine_version 스탬프 없음/구서버)는
   * 노란 업그레이드 버튼이 되고 **초기화 버튼(resetSyncBtn)을 대신한다**(운영자 지시 —
   * 그 경우 일반 재분석 자체가 곧 새 엔진 업그레이드다).
   */
  private updateDepthButton(): void {
    const meta = this.debugMeta;
    const isEveryric = this.badgeSource === 'everyric' && this.badgeSynced;
    this.depthBtn.classList.remove('ey-depth-upgrade', 'ey-depth-max');
    if (!isEveryric || !meta) {
      this.depthBtn.style.display = 'none';
      this.depthAction = null;
      return;
    }
    // null=스탬프 도입 전 세대, undefined=engine_version을 모르는 구서버 응답 — 둘 다
    // 구세대로 취급한다(신 스택 서버는 항상 스탬프를 내려준다).
    if (meta.engine_version == null) {
      this.depthBtn.replaceChildren(icon(depthArrowIcon(0)));
      this.depthBtn.classList.add('ey-depth-upgrade');
      this.depthBtn.title = t('overlay.depth.upgradeTitle');
      this.depthBtn.disabled = false;
      this.depthAction = () => {
        if (this.confirmTwice(this.depthBtn, t('overlay.depth.upgradeConfirm'))) {
          this.callbacks.onDepthUpgrade();
        }
      };
      this.depthBtn.style.display = '';
      this.resetSyncBtn.style.display = 'none'; // 업그레이드 버튼이 초기화 버튼을 대신한다
      return;
    }
    const route = meta.routing?.route;
    const depth = route === 'fast' ? 1 : route === 'medium' ? 2 : route === 'heavy' ? 3 : null;
    if (depth === null) {
      // 신세대인데 라우팅 메타가 없다 — 깊이를 모르니 버튼을 띄우지 않는다
      this.depthBtn.style.display = 'none';
      this.depthAction = null;
      return;
    }
    const badge = h('span', {
      className: `ey-depth-badge${depth === 3 ? ' max' : ''}`, text: String(depth),
    });
    this.depthBtn.replaceChildren(icon(depthArrowIcon(depth)), badge);
    if (depth === 3) {
      this.depthBtn.classList.add('ey-depth-max');
      this.depthBtn.title = t('overlay.depth.t3');
      this.depthBtn.disabled = true;
      this.depthAction = null;
    } else {
      this.depthBtn.title = depth === 1 ? t('overlay.depth.t1') : t('overlay.depth.t2');
      this.depthBtn.disabled = false;
      const next = depth === 1 ? ('medium' as const) : ('heavy' as const);
      this.depthAction = () => {
        if (this.confirmTwice(this.depthBtn, t('overlay.depth.confirm'))) {
          this.callbacks.onDepthUpgrade(next);
        }
      };
    }
    this.depthBtn.style.display = '';
  }

  private footerButton(text: string, title: string, onClick: () => void): HTMLButtonElement {
    return h('button', { className: 'ey-offset-btn', text, title, on: { click: onClick } });
  }

  private makeGenerateButton(label: string, onClick: () => void): HTMLButtonElement {
    const btn = createGenerateButton(label, this.serverStatus, onClick);
    this.generateButtons.push(btn);
    return btn;
  }

  private resetBody(): void {
    this.body.replaceChildren();
    // 스크롤 위치도 되돌린다 — 안 그러면 카라오케 자동 스크롤로 내려간 위치를 새 화면이
    // 그대로 물려받아, 검색 시트를 열었는데 검색창 대신 «싱크 초기화(서버 저장 삭제)»가
    // 첫 화면에 오는 사고가 난다(감사 A1-D1 실측: scrollTop 2599 → 시트 머리 y=-480).
    this.body.scrollTop = 0;
    // 화면 전환마다 도는 지점이라 분리된(더 이상 DOM에 없는) 생성 버튼 참조를 여기서도
    // 걸러낸다 — 그 버튼들이 들고 있던 가사 전문 클로저가 다음 setServerStatus까지
    // 기다리지 않고 곧바로 해제된다(5fps 감사 #3, 메모리 누적 방지).
    this.generateButtons = this.generateButtons.filter(btn => btn.isConnected);
    this.banner.style.display = 'none';
    this.footer.style.display = 'none';
    this.resumeChip.style.display = 'none';
    this.pipBtn.style.display = 'none';
    this.resetSyncBtn.style.display = 'none';
    this.depthBtn.style.display = 'none';
    this.depthAction = null;
    this.feedbackBtn.style.display = 'none';
    this.feedbackPop.style.display = 'none';
    this.hideWrongLyricsConfirm();
    // 번역 출처 병기(U2)·번역 대기 표시(U3-b)는 곡 단위 상태다 — 이전 곡 것이 새 곡
    // 화면에 남으면 안 된다. content가 setLangPending(null)/setAvailableLangs를 곡
    // 전환마다 다시 부르긴 하지만, 여기서도 방어적으로 지운다(resetBody는 모든 show*
    // 경로가 공유하는 단일 지점이라 새는 경로가 생기기 어렵다).
    this.translationSourceKind = null;
    this.translationSourceWikiName = null;
    this.translationPendingBar.style.display = 'none';
    this.lines = [];
    this.lineEls = [];
    this.filledUpTo = 0;
    this.activeWordEls = [];
    this.currentIndex = -1;
    this.userScrollUntil = 0;
    this.progressBar = null;
    this.progressText = null;
    this.searchResultsEl = null;
    this.closeSettings();
    this.closeDebugPanel(); // 곡이 바뀌면 이전 곡의 디버그 패널(원문·heard·시크 대상)이 남으면 안 된다
    // 레인도 같은 지점에서 비운다 — 모든 화면 전환이 이 함수를 지나므로 이전 곡 노트가
    // 새 화면에 남는 경로가 생기지 않는다(가사가 있는 경로는 곧바로 setLines로 다시 채운다)
    this.lane.setLines([]);
    this.laneShown = false;
    this.laneWrap.style.display = 'none';
    this.laneNotice.style.display = 'none';
    // 곡이 바뀌면 "이번 세션만 닫기"도 초기화한다 — 새 곡의 깊이는 다른 판단이므로
    // (영구 차단은 설정 karaokeTimingNoticeDismissed 쪽이 맡는다)
    this.timingNoticeHidden = false;
  }

  private showBanner(text: string, action?: HTMLElement): void {
    this.banner.replaceChildren(h('span', { className: 'ey-banner-text', text }));
    if (action) this.banner.append(action);
    this.banner.style.display = '';
  }

  private setSourceBadge(source: LyricsSource, synced: boolean): void {
    this.badgeSource = source;
    this.badgeSynced = synced;
    this.renderSourceBadge();
  }

  /**
   * 배지를 실제로 그린다 — setSourceBadge(곡 전환)와 setTranslationSource(번역 출처가
   * 나중에 붙을 때) 둘 다 여기로 모인다. source·synced는 badgeSource/badgeSynced에
   * 저장된 마지막 값을 쓴다.
   *
   * base 라벨(U2 배지 절충 수정): source==='vocaro'는 vocaro·miraheze 두 위키를 모두
   * 대표하는 값이라(adoptSourceResult 설계 메모) attributionSourceId로 실제 출처를
   * 갈라야 한다 — 이게 없으면 miraheze 채택분도 항상 "보카로 가사 위키"로 떴다.
   */
  private renderSourceBadge(): void {
    const source = this.badgeSource;
    const synced = this.badgeSynced;
    // source==='everyric'는 base로 'Everyric'을 더 이상 안 쓴다(운영자 지시 2026-08-04:
    // "Everyric · 보카로 가사 위키 · 번역: 보카로 가사 위키"처럼 이 확장 자신을 가리키는
    // 첫 조각이 늘 같은 값이라 소음이었다 — 이 배지는 이미 이 확장 안에서만 보이므로
    // "Everyric"이라는 정보 자체가 무의미하다). 실제 가사 출처(attributionName)를 그
    // 자리에 바로 쓴다 — 수동 붙여넣기 생성 등 출처가 없으면 빈 문자열로 접힌다(아래
    // parts 필터가 걸러낸다).
    const base = source === 'everyric' ? (this.attributionName ?? '')
      : source === 'vocaro'
        ? (this.attributionSourceId === 'miraheze' ? t('overlay.source.miraheze') : t('overlay.source.vocaro'))
      : source === 'caption' ? t('overlay.source.caption')
      : 'LRCLIB';
    // 가사 원출처(위키 등)를 병기 — everyric 분기는 이미 attributionName을 base로 썼으므로
    // 중복 병기하지 않는다. 그 밖의 소스(vocaro 직접 채택 등)만 부가 정보로 덧붙인다.
    const extra = source !== 'everyric' && this.attributionName && this.attributionName !== base
      ? this.attributionName : '';
    // 다른 영상의 싱크를 빌려온 경우 링크 표시 (해제는 검색 시트에서).
    // 검증(반주 대조)을 통과한 자동 링크와 검증 없는 수동 링크는 신뢰도가 다르다 —
    // 어긋난 가사를 보고 있을 때 원인을 짚을 수 있도록 ✓/? 로 구분해 표시한다
    const link = this.linkedInfo
      ? `🔗${this.linkedInfo.verified ? '✓' : '?'}${this.linkedInfo.offsetSec !== 0 ? `${this.linkedInfo.offsetSec > 0 ? '+' : ''}${this.linkedInfo.offsetSec}s` : ''}`
      : '';
    // 번역 출처 병기(U2) — 가사 원출처(extra)와 별개로, 사후 채택 번역이 어디서 왔는지.
    // kind==='wiki'면 실제로 히트한 위키 이름(translationSourceWikiName)을 그대로 쓴다.
    const trSource = this.translationSourceKind === 'caption' ? t('overlay.translationSource.caption')
      : this.translationSourceKind === 'wiki'
        ? t('overlay.translationSource.wiki', [this.translationSourceWikiName ?? t('overlay.source.vocaro')])
      : this.translationSourceKind === 'llm' ? t('overlay.translationSource.llm')
      : '';
    // base가 비어도(everyric×출처 없음) 뒤 조각들이 선두 " · "를 달고 뜨지 않도록 join으로
    // 조립한다 — 예전 raw concat은 base가 항상 비어 있지 않다는 전제였다.
    this.sourceBadge.textContent = [base, extra, link, trSource].filter(Boolean).join(' · ');
    // 출처 상세: 무엇을 어디서 가져왔는지 — 클릭 전에 툴팁으로도 확인 가능
    const kind = synced ? t('overlay.source.syncedLyrics') : t('overlay.source.plainLyrics');
    this.sourceBadge.title = this.sourceUrl ? `${kind} · ${t('overlay.source.openPage')}\n${this.sourceUrl}` : kind;
    if (this.linkedInfo) this.sourceBadge.title += `\n${this.describeLink(this.linkedInfo)}`;
    this.sourceBadge.classList.toggle('everyric', source === 'everyric');
    // 유튜브 자막 표시 상태를 색으로도 구분 — showSyncedLyrics의 captionPreSync 배너와 짝
    this.sourceBadge.classList.toggle('caption', source === 'caption');
  }

  /**
   * 사후 채택 번역(자막·위키·LLM)의 출처 — 가사 원문 출처(setAttribution)와 별개로
   * 배지에 병기한다(U2). null이면 표시 안 함. content가 tryCaptionTranslationLayer·
   * tryWikiTranslationLayer·applyTranslations(LLM 적용) 각 채택 시점에 부른다.
   * 곡이 바뀌면 resetBody가 지운다.
   */
  setTranslationSource(kind: 'caption' | 'wiki' | 'llm' | null, wikiName?: string | null): void {
    this.translationSourceKind = kind;
    this.translationSourceWikiName = wikiName ?? null;
    this.renderSourceBadge();
  }

  /**
   * 링크 한 건을 사람이 읽는 한 줄로 — 배지 툴팁과 검색 시트가 같은 문구를 쓴다.
   *
   * verified가 undefined면 서버가 검증 여부를 안 내려준 구버전이다 — 단정하지 않는다
   * (검증됐다고 잘못 말하면 어긋난 싱크를 신뢰하게 된다).
   */
  private describeLink(info: { sourceVideoId: string; verified?: boolean }): string {
    if (info.verified === true) return t('overlay.link.describeVerified', [info.sourceVideoId]);
    if (info.verified === false) {
      return t('overlay.link.describeUnverified', [info.sourceVideoId]);
    }
    return t('overlay.link.describeUnknown', [info.sourceVideoId]);
  }

  /**
   * 가사 원출처 표기 (이름+링크+출처 구분). show* 호출 전에 설정해야 배지에 반영된다.
   * SourceAttribution을 그대로 받는다 — sourceId까지 받아야 vocaro/miraheze 배지를
   * 가른다(U2, renderSourceBadge 참고). 예전엔 {name,url}만 받아 sourceId가 호출부
   * 객체에 실려 있어도 버려졌다.
   */
  setAttribution(attr: SourceAttribution | null): void {
    this.attributionName = attr?.name ?? null;
    this.attributionSourceId = attr?.sourceId ?? null;
    this.setSourceUrl(attr?.url ?? null);
  }

  /** 출처 페이지 링크 (보카로 위키 등 CC BY 출처 표기) — null이면 배지는 단순 라벨 */
  setSourceUrl(url: string | null): void {
    this.sourceUrl = url;
    this.sourceBadge.classList.toggle('link', url !== null);
    if (url) this.sourceBadge.title = t('overlay.source.openPage');
  }

  private changeOffset(delta: number | null): void {
    const next = delta === null ? 0 : Math.round((this.offsetSec + delta) * 10) / 10;
    this.offsetSec = next;
    this.updateOffsetLabel();
    this.callbacks.onOffsetChange(next);
  }

  private updateOffsetLabel(): void {
    const v = this.offsetSec;
    this.offsetLabel.textContent = `${v > 0 ? '+' : ''}${v.toFixed(1)}s`;
    this.offsetLabel.classList.toggle('nonzero', v !== 0);
  }

  private markUserScroll(): void {
    if (this.stateKind !== 'synced') return;
    this.userScrollUntil = Date.now() + USER_SCROLL_HOLD_MS;
    if (this.currentIndex >= 0) this.resumeChip.style.display = '';
  }

  private resumeAutoScroll(): void {
    this.userScrollUntil = 0;
    this.resumeChip.style.display = 'none';
    this.scrollToCurrent();
  }

  private scrollToCurrent(): void {
    const el = this.currentIndex >= 0 ? this.lineEls[this.currentIndex] : undefined;
    if (!el) return;
    const top = el.offsetTop - this.body.clientHeight / 2 + el.offsetHeight / 2;
    this.body.scrollTo({ top, behavior: 'smooth' });
    this.resumeChip.style.display = 'none';
  }

  // ── 설정 시트 ─────────────────────────────────────────────────

  private toggleSettings(): void {
    if (this.settingsSheet) {
      this.closeSettings();
      return;
    }
    this.openSettings();
  }

  /** 설정 시트 열기 — 서버 오류 배너의 '설정 열기'가 여기로 온다 (패널이 숨어 있으면 함께 띄운다) */
  openSettings(): void {
    if (!this.visible) this.setVisible(true);
    if (this.geometry.collapsed) this.setCollapsed(false);
    if (this.settingsSheet) return;
    const sheet = buildSettingsSheet({
      sections: this.settingsSections(),
      onClose: () => this.closeSettings(),
    });
    this.settingsSheet = sheet.el;
    this.panel.append(sheet.el);
    // 검색칸으로 바로 — 설정을 여는 사람 대부분은 이미 무엇을 고칠지 알고 있고,
    // 그때 필요한 건 스크롤이 아니라 이름을 치는 것이다. 패널이 키 이벤트를
    // 끊으므로(생성자) 여기 타이핑이 유튜브 단축키로 새지 않는다.
    sheet.focusFilter();
  }

  private closeSettings(): void {
    this.settingsSheet?.remove();
    this.settingsSheet = null;
    this.settingsDot = null;
    this.settingsPermBtn = null;
  }

  /**
   * 설정 시트 명세 — 범주와 각 줄의 값·저장 동작만 여기서 정하고, 접힘·검색·그리기는
   * panels.buildSettingsSheet가 맡는다(그쪽 주석에 분류+검색이 둘 다 필요한 근거).
   *
   * **여기 없는 설정이 넷 있다**(offsetSec·pipVideoRatio·pipWidth·pipHeight). 전부 화면
   * 조작으로 정해지는 값이라 숫자로 고를 자리를 만드는 편이 더 나쁘다: 오프셋은 푸터의
   * ±0.1 버튼이 **영상별로** 들고 있어 전역 설정으로 되돌리면 안 되고, 나머지 셋은 PiP
   * 창을 끌어 놓은 크기·비율을 그대로 기억한 값이다.
   *
   * 각 줄의 key는 DOM id가 아니라 **검색어**다 — 라벨을 모른 채 'serverUrl'·'pitchF0'로
   * 찾는 사람이 실제로 있어서(문서·제보 따라 하기) Settings의 실제 키 이름을 그대로 쓴다.
   */
  private settingsSections(): SettingsSection[] {
    const set = (patch: Partial<Settings>): void => this.callbacks.onSettingsChange(patch);
    const pct = (v: number): string => `${Math.round(v * 100)}%`;
    const px = (v: number): string => `${Math.round(v)}px`;

    // 아래 다섯은 **호스트가 참조를 계속 들고 있어야 하는** 컨트롤이라 custom 줄로 꽂는다:
    // 상태 점·권한 버튼은 setServerStatus가 시트가 열린 채로 갱신하고, 기기 목록 select는
    // enumerateDevices가 늦게 돌아와 비동기로 채워진다.
    const dot = h('span', {
      className: 'ey-dot',
      title: t('overlay.settings.serverStatusTitle', [statusLine(this.serverStatus)]),
    });
    this.applyDotClasses(dot, this.serverStatus);
    this.settingsDot = dot;

    // 서버가 정상이 아니면 설정 안에서도 사유를 글자로 남긴다 (색맹·툴팁 미표시 환경 대비)
    const serverNote = h('div', { className: 'ey-settings-note ey-settings-server-note' });
    if (!serverUsable(this.serverStatus)) {
      serverNote.textContent = statusLine(this.serverStatus)
        + (this.serverStatus.detail ? ` — ${this.serverStatus.detail}` : '');
      serverNote.classList.add('bad');
    } else {
      serverNote.style.display = 'none';
    }

    /**
     * 로컬 서버 URL을 입력한 직후 사용자가 보고 있는 자리 — 여기에 허용 버튼을 둔다.
     *
     * 배너(buildServerStatusBar)에도 같은 버튼이 있지만 그건 패널 상단이고, URL을 막 고친
     * 사람의 눈은 이 입력칸에 있다. 두 버튼은 **같은 콜백 하나**를 부르므로 갈라질 여지가
     * 없다 (URL 입력을 두 곳에 두지 않은 것과 같은 이유다).
     */
    const permBtn = h('button', {
      className: 'ey-secondary-btn ey-settings-perm-btn',
      text: t('panels.serverBar.openPermissions'),
      attrs: { type: 'button', title: t('overlay.settings.permBtnTitle') },
      on: { click: () => this.callbacks.onOpenPermissions() },
    });
    this.settingsPermBtn = permBtn;
    permBtn.style.display = needsHostPermission(this.serverStatus) ? '' : 'none';

    const audioOut = h('select', { className: 'ey-select' });
    audioOut.addEventListener('change', () => set({ audioOutputId: audioOut.value }));
    const micDevice = h('select', { className: 'ey-select' });
    micDevice.addEventListener('change', () => set({ micDeviceId: micDevice.value }));
    void this.populateAudioDevices(audioOut, micDevice);

    const searchRows: SettingRow[] = [
      {
        kind: 'checkbox', key: 'autoSearch', label: t('overlay.settings.row.autoSearch'),
        title: t('overlay.settings.row.autoSearchTitle'), value: this.settings.autoSearch,
        onChange: v => set({ autoSearch: v }),
      },
      {
        kind: 'checkbox', key: 'autoSearchShorts', label: t('overlay.settings.row.autoSearchShorts'),
        title: t('overlay.settings.row.autoSearchShortsTitle'), value: this.settings.autoSearchShorts,
        onChange: v => set({ autoSearchShorts: v }),
      },
      {
        kind: 'select', key: 'lyricsSourcePriority', label: t('overlay.settings.row.sourcePriority'),
        value: this.settings.lyricsSourcePriority,
        options: [
          ['vocaro', t('overlay.settings.sourcePriority.vocaro')],
          ['lrclib', t('overlay.settings.sourcePriority.lrclib')],
        ],
        onChange: v => set({ lyricsSourcePriority: v as Settings['lyricsSourcePriority'] }),
      },
    ];

    const displayRows: SettingRow[] = [
      {
        kind: 'select', key: 'fontSize', label: t('overlay.settings.row.fontSize'),
        keywords: '폰트 글꼴 글씨 크기 font',
        value: this.settings.fontSize,
        options: [
          ['small', t('overlay.settings.fontSize.small')],
          ['medium', t('overlay.settings.fontSize.medium')],
          ['large', t('overlay.settings.fontSize.large')],
        ],
        onChange: v => set({ fontSize: v as Settings['fontSize'] }),
      },
      {
        // fontSize(3단 프리셋) 위에 얹는 미세 배율 — 기존 select는 그대로 두고 바로
        // 아래에 슬라이더를 추가한다(선례: laneRows의 pitchFontScale 행)
        kind: 'range', key: 'mainFontScale', label: t('overlay.settings.row.mainFontScale'),
        keywords: '가사창 폰트 글씨 크기 배율',
        value: this.settings.mainFontScale, min: 0.7, max: 1.6, step: 0.05,
        format: v => `${v.toFixed(2)}×`,
        onChange: v => set({ mainFontScale: v }),
      },
      {
        kind: 'select', key: 'theme', label: t('overlay.settings.row.theme'),
        keywords: '다크 라이트 dark light',
        value: this.settings.theme,
        options: [
          ['auto', t('overlay.settings.optAuto')],
          ['dark', t('overlay.settings.theme.dark')],
          ['light', t('overlay.settings.theme.light')],
        ],
        onChange: v => set({ theme: v as Settings['theme'] }),
      },
      {
        kind: 'select', key: 'uiLanguage', label: t('overlay.settings.row.uiLanguage'),
        value: this.settings.uiLanguage,
        // 언어 이름은 각 언어 자신의 표기로 고정 — uiLanguage가 바뀌어도 번역하지 않는다
        // (사용자가 어느 표시 언어에서도 자기 언어를 바로 찾을 수 있어야 하는 표준 관례)
        options: [['auto', t('overlay.settings.optAuto')], ['ko', '한국어'], ['en', 'English'], ['ja', '日本語']],
        onChange: v => set({ uiLanguage: v as Settings['uiLanguage'] }),
      },
      {
        kind: 'checkbox', key: 'showTranslation', label: t('overlay.settings.row.showTranslation'),
        value: this.settings.showTranslation, onChange: v => set({ showTranslation: v }),
      },
      {
        kind: 'select', key: 'translationLanguage', label: t('overlay.settings.row.translationLanguage'),
        value: this.settings.translationLanguage,
        options: [['ko', '한국어'], ['en', 'English'], ['ja', '日本語'], ['zh', '中文']],
        onChange: v => set({ translationLanguage: v }),
      },
      {
        // '자동'이면 번역 언어 기준으로 hangul/romaji/kana 중 골라진다(lib/lang.ts resolveScript)
        kind: 'select', key: 'pronunciationScript', label: t('overlay.settings.row.pronScript'),
        keywords: '발음 독음 로마자 가나 ipa',
        value: this.settings.pronunciationScript,
        options: [
          ['auto', t('overlay.settings.optAuto')],
          ['hangul', t('overlay.settings.pronScript.hangul')],
          ['romaji', t('overlay.settings.pronScript.romaji')],
          ['kana', t('overlay.settings.pronScript.kana')],
          ['ipa', t('overlay.settings.pronScript.ipa')],
        ],
        onChange: v => set({ pronunciationScript: v as Settings['pronunciationScript'] }),
      },
      {
        kind: 'checkbox', key: 'showPronunciation', label: t('overlay.settings.row.showPronunciation'),
        value: this.settings.showPronunciation, onChange: v => set({ showPronunciation: v }),
      },
      {
        // showPronunciation이 꺼져 있으면 이 설정은 어차피 소음이므로 켜져 있을 때만 의미가
        // 있다 — 그래도 항상 노출해 둔다(비어 있는 조건부 표시는 "설정이 사라졌다"로 읽힌다)
        kind: 'checkbox', key: 'hidePronForEnglish', label: t('overlay.settings.row.hidePronForEnglish'),
        title: t('overlay.settings.row.hidePronForEnglishTitle'),
        value: this.settings.hidePronForEnglish, onChange: v => set({ hidePronForEnglish: v }),
      },
      {
        kind: 'checkbox', key: 'vocalGlow', label: t('overlay.settings.row.vocalGlow'),
        title: t('overlay.settings.row.vocalGlowTitle'), value: this.settings.vocalGlow,
        onChange: v => set({ vocalGlow: v }),
      },
      {
        kind: 'checkbox', key: 'lowConfWarning', label: t('overlay.settings.row.lowConfWarning'),
        title: t('overlay.settings.row.lowConfWarningTitle'), value: this.settings.lowConfWarning,
        onChange: v => set({ lowConfWarning: v }),
      },
      {
        kind: 'checkbox', key: 'notifyOnComplete', label: t('overlay.settings.row.notifyOnComplete'),
        title: t('overlay.settings.row.notifyOnCompleteTitle'), value: this.settings.notifyOnComplete,
        onChange: v => set({ notifyOnComplete: v }),
      },
    ];

    const laneRows: SettingRow[] = [
      {
        kind: 'checkbox', key: 'pitchGuide', label: t('overlay.settings.row.pitchGuide'),
        value: this.settings.pitchGuide, onChange: v => set({ pitchGuide: v }),
      },
      {
        kind: 'select', key: 'pitchWindowMeasures', label: t('overlay.settings.row.pitchWindow'),
        keywords: '마디 창 window',
        value: String(this.settings.pitchWindowMeasures),
        options: [
          ['0.5', t('overlay.settings.pitchWindow.half')],
          ['1', t('overlay.settings.pitchWindow.bars', ['1'])],
          ['2', t('overlay.settings.pitchWindow.bars', ['2'])],
          ['4', t('overlay.settings.pitchWindow.bars', ['4'])],
          ['8', t('overlay.settings.pitchWindow.bars', ['8'])],
        ],
        onChange: v => set({ pitchWindowMeasures: Number(v) }),
      },
      {
        kind: 'select', key: 'pitchScrollMode', label: t('overlay.settings.row.pitchMode'),
        value: this.settings.pitchScrollMode,
        options: [
          ['page', t('overlay.settings.pitchMode.page')],
          ['scroll', t('overlay.settings.pitchMode.scroll')],
        ],
        onChange: v => set({ pitchScrollMode: v as Settings['pitchScrollMode'] }),
      },
      {
        kind: 'select', key: 'pitchFontScale', label: t('overlay.settings.row.pitchFont'),
        keywords: '레인 글씨 크기',
        value: String(this.settings.pitchFontScale),
        options: [
          ['0.85', t('overlay.settings.pitchFont.small')],
          ['1', t('overlay.settings.pitchFont.normal')],
          ['1.2', t('overlay.settings.pitchFont.large')],
          ['1.45', t('overlay.settings.pitchFont.xlarge')],
        ],
        onChange: v => set({ pitchFontScale: Number(v) }),
      },
      {
        kind: 'select', key: 'solfegeNotation', label: t('overlay.settings.row.solfegeNotation'),
        keywords: '계이름 도레미 solfege',
        value: this.settings.solfegeNotation,
        options: [
          ['korean', t('overlay.settings.solfegeNotation.korean')],
          ['english', t('overlay.settings.solfegeNotation.english')],
        ],
        onChange: v => set({ solfegeNotation: v as Settings['solfegeNotation'] }),
      },
      {
        kind: 'checkbox', key: 'pitchCountdown', label: t('overlay.settings.row.pitchCountdown'),
        value: this.settings.pitchCountdown, onChange: v => set({ pitchCountdown: v }),
      },
      {
        kind: 'checkbox', key: 'pitchF0Curve', label: t('overlay.settings.row.pitchF0Curve'),
        title: t('overlay.settings.row.pitchF0CurveTitle'), value: this.settings.pitchF0Curve,
        onChange: v => set({ pitchF0Curve: v }),
      },
      {
        kind: 'range', key: 'pitchLineOpacity', label: t('overlay.settings.row.pitchLineOpacity'),
        title: t('overlay.settings.row.pitchLineOpacityTitle'),
        value: this.settings.pitchLineOpacity, min: 0.2, max: 1, step: 0.05, format: pct,
        onChange: v => set({ pitchLineOpacity: v }),
      },
      {
        kind: 'range', key: 'pitchF0Opacity', label: t('overlay.settings.row.pitchF0Opacity'),
        title: t('overlay.settings.row.pitchF0OpacityTitle'),
        value: this.settings.pitchF0Opacity, min: 0.2, max: 1.5, step: 0.05, format: pct,
        onChange: v => set({ pitchF0Opacity: v }),
      },
      {
        kind: 'select', key: 'pitchPronPosition', label: t('overlay.settings.row.pronPosition'),
        title: t('overlay.settings.row.pronPositionTitle'),
        value: this.settings.pitchPronPosition,
        // 'note'는 목록에서 빠졌다 — 노트 위 음절은 이제 설정 밖에서 항상 표시된다.
        // 남은 값은 «이중표시 줄»의 자리다.
        options: [
          ['off', t('overlay.settings.pronPosition.off')],
          ['bottom', t('overlay.settings.pronPosition.bottom')],
          ['center', t('overlay.settings.pronPosition.center')],
          ['both', t('overlay.settings.pronPosition.both')],
        ],
        onChange: v => set({ pitchPronPosition: v as Settings['pitchPronPosition'] }),
      },
      {
        // 디바이더 드래그와 **같은 값**을 만진다 — 드래그가 안 되는 좁은 패널에서도
        // 폭을 정할 수 있어야 해서 숫자 경로를 함께 둔다(clampLaneWidth가 실제 상한을 다시 깎는다)
        kind: 'range', key: 'mainLaneWidth', label: t('panels.settings.row.mainLaneWidth'),
        keywords: '가사창 레인 너비 width',
        value: this.settings.mainLaneWidth,
        // 슬라이더 최대값은 «고를 수 있는 범위»일 뿐 상한이 아니다 — 실제 제한은
        // clampLaneWidth의 roomCap(가사 목록에 남겨야 하는 최소 폭)뿐이다. 저장값이
        // 이보다 크면 슬라이더가 그 값을 표시하지 못하므로 함께 늘린다.
        min: LANE_WIDTH_MIN, max: Math.max(720, this.settings.mainLaneWidth), step: 10, format: px,
        onChange: v => set({ mainLaneWidth: Math.round(v) }),
      },
      {
        kind: 'select', key: 'mainLanePos', label: t('panels.settings.row.mainLanePos'),
        keywords: '가사창 레인 위치 왼쪽 아래 부착',
        value: this.settings.mainLanePos,
        options: [
          ['left', t('panels.settings.mainLanePos.left')],
          ['bottom', t('panels.settings.mainLanePos.bottom')],
          ['attached', t('panels.settings.mainLanePos.attached')],
        ],
        onChange: v => set({ mainLanePos: v as Settings['mainLanePos'] }),
      },
      {
        // mainLaneWidth와 별개 값 — 부착 패널은 패널 폭에서 깎이지 않으므로 화면 폭까지 넓힐 수 있다
        kind: 'range', key: 'attachedLaneWidth', label: t('panels.settings.row.attachedLaneWidth'),
        keywords: '가사창 레인 부착 너비 width attached',
        value: this.settings.attachedLaneWidth,
        min: ATTACH_WIDTH_MIN, max: Math.max(720, this.settings.attachedLaneWidth), step: 10, format: px,
        onChange: v => set({ attachedLaneWidth: Math.round(v) }),
      },
    ];

    const moduleRows: SettingRow[] = [
      {
        kind: 'checkbox', key: 'videoCaptions', label: t('overlay.settings.row.videoCaptions'),
        title: t('overlay.settings.row.videoCaptionsTitle'), value: this.settings.videoCaptions,
        onChange: v => set({ videoCaptions: v }),
      },
      {
        kind: 'range', key: 'captionFontScale', label: t('overlay.settings.row.captionFontScale'),
        keywords: '자막 글자 크기 caption',
        value: this.settings.captionFontScale, min: 0.7, max: 1.6, step: 0.05,
        format: v => `${v.toFixed(2)}×`,
        onChange: v => set({ captionFontScale: v }),
      },
      {
        kind: 'range', key: 'captionBgOpacity', label: t('overlay.settings.row.captionBgOpacity'),
        keywords: '자막 배경 caption',
        value: this.settings.captionBgOpacity, min: 0, max: 1, step: 0.05, format: pct,
        onChange: v => set({ captionBgOpacity: v }),
      },
      {
        // 표면별 키 — PiP 창의 설정 시트에서 이 행을 만지면 **그 창의** 재생목록이
        // 바뀐다(메인 것이 아니라). 퀵 줄·코너 버튼과 같은 규칙이라 «이 창에서 보이는
        // 토글은 전부 이 창의 것»으로 일관된다.
        kind: 'checkbox', key: 'modPlaylist', label: t('overlay.settings.row.modPlaylist'),
        title: t('overlay.settings.row.modPlaylistTitle'), value: this.playlistVisible(),
        onChange: v => set(this.chrome === 'filled' ? { pipPlaylist: v } : { modPlaylist: v }),
      },
      {
        // 표면별 키 — floating은 modMainLane(옵트인 모듈), filled는 pitchGuide.
        // laneWanted()가 그 판정을 쥐고 있으므로 표시값도 거기서 읽는다.
        kind: 'checkbox', key: 'modMainLane', label: t('overlay.settings.row.modMainLane'),
        title: t('overlay.settings.row.modMainLaneTitle'), value: this.laneWanted(),
        onChange: v => set(this.chrome === 'filled' ? { pitchGuide: v } : { modMainLane: v }),
      },
    ];

    const pipRows: SettingRow[] = [
      {
        kind: 'checkbox', key: 'pipKeepPanel', label: t('overlay.settings.row.pipKeepPanel'),
        value: this.settings.pipKeepPanel, onChange: v => set({ pipKeepPanel: v }),
      },
      {
        kind: 'checkbox', key: 'pipShowVideo', label: t('overlay.settings.row.pipShowVideo'),
        value: this.settings.pipShowVideo, onChange: v => set({ pipShowVideo: v }),
      },
      {
        // 중앙 열(영상 폭)의 «현재 줄 한 줄» — 끄면 그 자리를 영상·컨트롤이 나눠 갖는다.
        // 제거된 pipLyricsList(오른쪽 가사 목록 컬럼)의 자리를 이어받은 토글이라 총량은 ±0이다.
        kind: 'checkbox', key: 'pipShortLyrics', label: t('overlay.settings.row.pipShortLyrics'),
        title: t('overlay.settings.row.pipShortLyricsTitle'),
        value: this.settings.pipShortLyrics, onChange: v => set({ pipShortLyrics: v }),
      },
      {
        // 코너 미니 버튼과 같은 값 — 시트에도 두는 이유는 «어디서 껐는지» 잊었을 때
        // 찾을 곳이 필요해서다(코너 버튼은 빠른 길이지 유일한 길이 아니다)
        kind: 'checkbox', key: 'pipShowPanel', label: t('overlay.settings.row.pipShowPanel'),
        title: t('overlay.settings.row.pipShowPanelTitle'),
        value: this.settings.pipShowPanel, onChange: v => set({ pipShowPanel: v }),
      },
      {
        kind: 'select', key: 'pipChromaKey', label: t('overlay.settings.row.pipChromaKey'),
        title: t('overlay.settings.row.pipChromaKeyTitle'),
        keywords: '방송 obs 크로마키',
        value: this.settings.pipChromaKey,
        options: [
          ['off', t('overlay.settings.chroma.off')],
          ['green', t('overlay.settings.chroma.green')],
          ['blue', t('overlay.settings.chroma.blue')],
          ['magenta', t('overlay.settings.chroma.magenta')],
        ],
        onChange: v => set({ pipChromaKey: v as Settings['pipChromaKey'] }),
      },
      {
        // 크로마키 바로 밑 — 둘 다 «방송 화면에서 읽히게» 하는 설정이라 함께 찾게 된다
        kind: 'checkbox', key: 'streamTextOutline', label: t('overlay.settings.row.streamTextOutline'),
        title: t('overlay.settings.row.streamTextOutlineTitle'),
        keywords: '방송 obs 외곽선 테두리 outline stroke',
        value: this.settings.streamTextOutline, onChange: v => set({ streamTextOutline: v }),
      },
      {
        // PiP 창 안 디바이더로도 바꾸는 값 — 다음에 여는 창부터 이 높이로 열린다
        kind: 'range', key: 'pitchLaneHeight', label: t('panels.settings.row.pitchLaneHeight'),
        keywords: 'pip 레인 높이 lane height',
        value: this.settings.pitchLaneHeight, min: 90, max: 420, step: 10, format: px,
        onChange: v => set({ pitchLaneHeight: Math.round(v) }),
      },
    ];

    const audioRows: SettingRow[] = [
      {
        // 멜로디·메트로놈은 예전부터 '볼륨 슬라이더 + 켜기' 한 줄이었다 — 켜고 나서
        // 볼륨을 찾으러 다른 줄로 가지 않아도 된다
        kind: 'checkbox', key: 'melodyPlayback', label: t('overlay.settings.row.melodyPlayback'),
        title: t('overlay.settings.row.melodyPlaybackTitle'), value: this.settings.melodyPlayback,
        onChange: v => set({ melodyPlayback: v }),
        range: { value: this.settings.melodyVolume, onChange: v => set({ melodyVolume: v }) },
        keywords: 'melodyVolume 볼륨',
      },
      {
        kind: 'checkbox', key: 'metronome', label: t('overlay.settings.row.metronome'),
        title: t('overlay.settings.row.metronomeTitle'), value: this.settings.metronome,
        onChange: v => set({ metronome: v }),
        range: { value: this.settings.metronomeVolume, onChange: v => set({ metronomeVolume: v }) },
        keywords: 'metronomeVolume 볼륨 박자',
      },
      {
        kind: 'select', key: 'metronomeRate', label: t('overlay.settings.row.metronomeRate'),
        title: t('overlay.settings.row.metronomeRateTitle'),
        value: String(this.settings.metronomeRate),
        options: [
          ['0.5', t('overlay.settings.metronomeRate.half')],
          ['1', t('overlay.settings.metronomeRate.one')],
          ['2', t('overlay.settings.metronomeRate.two')],
        ],
        onChange: v => set({ metronomeRate: Number(v) }),
      },
      {
        kind: 'select', key: 'metronomeBeat', label: t('overlay.settings.row.metronomeBeat'),
        title: t('overlay.settings.row.metronomeBeatTitle'),
        value: String(this.settings.metronomeBeat),
        options: [
          ['0', t('overlay.settings.metronomeBeat.n', ['1'])],
          ['1', t('overlay.settings.metronomeBeat.n', ['2'])],
          ['2', t('overlay.settings.metronomeBeat.n', ['3'])],
          ['3', t('overlay.settings.metronomeBeat.n', ['4'])],
        ],
        onChange: v => set({ metronomeBeat: Number(v) }),
      },
      {
        kind: 'custom', key: 'audioOutputId', label: t('overlay.settings.row.audioOut'),
        title: t('overlay.settings.row.audioOutTitle'), keywords: '출력 스피커 기기',
        control: audioOut,
      },
      {
        kind: 'checkbox', key: 'micPitch', label: t('overlay.settings.row.micPitch'),
        title: t('overlay.settings.row.micPitchTitle'), value: this.settings.micPitch,
        onChange: v => set({ micPitch: v }),
      },
      {
        kind: 'custom', key: 'micDeviceId', label: t('overlay.settings.row.micDevice'),
        keywords: '마이크 입력 기기', control: micDevice,
      },
      {
        kind: 'select', key: 'micOctave', label: t('overlay.settings.row.micOctave'),
        title: t('overlay.settings.row.micOctaveTitle'),
        value: String(this.settings.micOctave),
        options: [
          ['-2', t('overlay.settings.micOctave.n', ['-2'])],
          ['-1', t('overlay.settings.micOctave.n', ['-1'])],
          ['0', t('overlay.settings.micOctave.none')],
          ['1', t('overlay.settings.micOctave.n', ['+1'])],
          ['2', t('overlay.settings.micOctave.n', ['+2'])],
        ],
        onChange: v => set({ micOctave: Number(v) }),
      },
      { kind: 'note', key: 'deviceNote', text: t('overlay.settings.deviceNote') },
    ];

    const serverRows: SettingRow[] = [
      {
        kind: 'text', key: 'serverUrl', label: t('overlay.settings.serverUrlLabel'),
        layout: 'col', labelSuffix: dot, value: this.settings.serverUrl,
        // 빈 값이면 buildSettingControl이 저장하지 않고 되돌린다 — 주소를 통째로 지운
        // 상태로 저장되면 그 뒤 모든 요청이 갈 곳을 잃는다
        sanitize: v => v.trim().replace(/\/+$/, ''),
        onChange: v => set({ serverUrl: v }),
      },
      {
        kind: 'text', key: 'apiKey', label: t('overlay.settings.apiKeyLabel'),
        layout: 'col', password: true, placeholder: t('overlay.settings.apiKeyPlaceholder'),
        value: this.settings.apiKey, onChange: v => set({ apiKey: v }),
      },
      { kind: 'custom', key: 'serverStatusNote', control: serverNote },
      { kind: 'custom', key: 'serverPermission', keywords: '권한 permission', control: permBtn },
      {
        kind: 'checkbox', key: 'debugInfo', label: t('overlay.settings.row.debugInfo'),
        value: this.settings.debugInfo, onChange: v => set({ debugInfo: v }),
      },
      { kind: 'note', key: 'serverRequiredNote', text: t('overlay.settings.serverRequiredNote') },
    ];

    const resetRows: SettingRow[] = [
      {
        // "다시 보지 않기"로 끈 음절 타이밍 안내를 되살리는 **유일한** 경로 — 안내 자체가
        // 사라진 뒤에는 여기 말고 되돌릴 자리가 없다. 이미 켜져 있을 때 눌러도 무해하므로
        // 숨기지 않는다(비어 있는 범주는 고장난 것처럼 보인다).
        kind: 'button', key: 'karaokeTimingNoticeDismissed',
        label: t('panels.settings.reset.karaokeNotice'),
        title: t('panels.settings.reset.karaokeNoticeTitle'),
        keywords: '가라오케 타이밍 안내 배너 karaoke timing notice',
        onClick: btn => {
          this.timingNoticeHidden = false;
          set({ karaokeTimingNoticeDismissed: false });
          btn.textContent = t('panels.settings.reset.done');
        },
      },
      {
        // 곡별로 끈 저신뢰 경고를 전부 되살리는 유일한 경로 — 카라오케 안내 되살리기
        // 버튼과 같은 자리 규칙(비어 있어도 숨기지 않는다, 눌러도 무해).
        kind: 'button', key: 'warnDismissReset',
        label: t('panels.settings.reset.warnDismiss'),
        title: t('panels.settings.reset.warnDismissTitle'),
        keywords: '저신뢰 경고 신뢰도 낮음 정렬 다시 보기 warning confidence',
        onClick: btn => {
          this.callbacks.onResetWarnDismiss();
          btn.textContent = t('panels.settings.reset.done');
        },
      },
      {
        // 2단계 확인 — 같은 버튼을 두 번 눌러야 실행된다(window.confirm 대신 이 방식을
        // 쓰는 이유: 되돌릴 수 없는 동작이고, 4초 뒤 자동으로 원래 라벨로 돌아가 실수로
        // 남겨둔 무장 상태가 다음 클릭을 삼키지 않는다). 시트를 열 때마다 이 행이 새로
        // 만들어지므로(settingsSections는 openSettings마다 새로 호출된다) armed 상태를
        // 인스턴스 필드가 아니라 클로저 지역변수로 둬도 다음에 여는 시트엔 영향이 없다.
        // 단 «타이머»만은 인스턴스 필드(confirmTimer)에 둔다 — 지역변수로 두면 destroy()가
        // 원리적으로 걷을 수 없어서, 창을 닫아도 4초짜리 타이머가 죽은 버튼을 만진다.
        kind: 'button', key: 'fullReset',
        label: t('panels.settings.reset.fullReset'),
        title: t('panels.settings.reset.fullResetTitle'),
        keywords: '전체 초기화 리셋 기여 이력 설정 reset all',
        onClick: (() => {
          let armed = false;
          return (btn: HTMLButtonElement) => {
            if (!armed) {
              armed = true;
              btn.textContent = t('panels.settings.reset.fullResetConfirm');
              window.clearTimeout(this.confirmTimer);
              this.confirmTimer = window.setTimeout(() => {
                armed = false;
                btn.textContent = t('panels.settings.reset.fullReset');
              }, 4000);
              return;
            }
            window.clearTimeout(this.confirmTimer);
            armed = false;
            btn.textContent = t('panels.settings.reset.done'); // 카라오케 안내 되살리기 버튼과 같은 완료 문구 재사용
            this.callbacks.onFullReset();
          };
        })(),
      },
    ];

    return [
      { id: 'search', title: t('panels.settings.section.search'), icon: '🔎', rows: searchRows },
      { id: 'display', title: t('panels.settings.section.display'), icon: '🎨', rows: displayRows },
      { id: 'lane', title: t('panels.settings.section.lane'), icon: '🎹', rows: laneRows },
      { id: 'modules', title: t('panels.settings.section.modules'), icon: '🧩', rows: moduleRows },
      { id: 'pip', title: t('panels.settings.section.pip'), icon: '🪟', rows: pipRows },
      { id: 'audio', title: t('panels.settings.section.audio'), icon: '🔊', rows: audioRows },
      { id: 'server', title: t('panels.settings.section.server'), icon: '🖥️', rows: serverRows },
      { id: 'reset', title: t('panels.settings.section.reset'), icon: '♻️', rows: resetRows },
    ];
  }

  /** 오디오 입출력 기기 목록 채우기 — 라벨은 마이크 권한을 허용해야 브라우저가 내려준다 */
  private async populateAudioDevices(outSel: HTMLSelectElement, inSel: HTMLSelectElement): Promise<void> {
    const fill = (sel: HTMLSelectElement, devices: MediaDeviceInfo[], defLabel: string, cur: string) => {
      sel.replaceChildren(h('option', { text: defLabel, attrs: { value: '' } }));
      devices.forEach((d, i) => {
        if (!d.deviceId || d.deviceId === 'default' || d.deviceId === 'communications') return;
        sel.append(h('option', { text: d.label || t('overlay.settings.deviceN', [String(i + 1)]), attrs: { value: d.deviceId } }));
      });
      sel.value = Array.from(sel.options).some(o => o.value === cur) ? cur : '';
    };
    let devices: MediaDeviceInfo[] = [];
    try {
      devices = await navigator.mediaDevices.enumerateDevices();
    } catch {
      /* 권한 API 불가 환경 — 기본 항목만 표시 */
    }
    fill(outSel, devices.filter(d => d.kind === 'audiooutput'), t('overlay.settings.defaultOutput'), this.settings.audioOutputId);
    fill(inSel, devices.filter(d => d.kind === 'audioinput'), t('overlay.settings.defaultMic'), this.settings.micDeviceId);
  }

  // ── 위치/크기 ─────────────────────────────────────────────────

  private defaultGeometry(): PanelGeometry {
    // 생성 시점엔 아직 마운트 전이라 창을 모른다 — 유튜브 페이지 창을 기준으로 잡고,
    // filled면 어차피 applyGeometry가 이 값을 쓰지 않는다
    return {
      x: Math.max(EDGE_MARGIN, window.innerWidth - DEFAULT_WIDTH - 24),
      y: 72,
      width: DEFAULT_WIDTH,
      height: Math.min(DEFAULT_HEIGHT, Math.round(window.innerHeight * 0.7)),
      collapsed: false,
    };
  }

  private applyGeometry(): void {
    // filled: 창이 곧 패널이다. 좌표·크기·접기를 창에서 빼앗아 오는 대신 창을 그대로
    // 채운다 — 이 인스턴스에는 "화면 밖으로 나가는" 상태가 존재할 수 없다.
    if (this.chrome === 'filled') {
      this.panel.classList.add('ey-panel-filled');
      this.panel.classList.remove('collapsed');
      this.panel.style.left = '';
      this.panel.style.top = '';
      this.panel.style.width = '';
      this.panel.style.height = '';
      return;
    }
    this.applyingGeometry = true;
    const g = this.geometry;
    this.panel.style.left = `${g.x}px`;
    this.panel.style.top = `${g.y}px`;
    this.panel.style.width = `${g.width}px`;
    this.panel.classList.toggle('collapsed', g.collapsed);
    this.panel.style.height = g.collapsed ? 'auto' : `${g.height}px`;
    this.collapseBtn.replaceChildren(icon(g.collapsed ? ICONS.expand : ICONS.collapse));
    this.collapseBtn.title = g.collapsed ? t('overlay.header.expand') : t('overlay.header.collapse');
    this.updateAttachPlacement(); // 부착 모드면 패널 좌표·접힘 상태가 바뀔 때마다 따라간다
    this.updatePlaylistPlacement();
    const w = this.win ?? window;
    w.cancelAnimationFrame(this.geomRaf);
    this.geomRaf = w.requestAnimationFrame(() => {
      this.applyingGeometry = false;
    });
  }

  private setCollapsed(collapsed: boolean): void {
    this.geometry.collapsed = collapsed;
    this.applyGeometry();
    this.scheduleGeometrySave();
  }

  private setupDrag(): void {
    let startX = 0;
    let startY = 0;
    let origX = 0;
    let origY = 0;
    let dragging = false;

    this.header.addEventListener('pointerdown', (e: PointerEvent) => {
      if ((e.target as HTMLElement).closest('button')) return;
      dragging = true;
      startX = e.clientX;
      startY = e.clientY;
      origX = this.geometry.x;
      origY = this.geometry.y;
      this.header.setPointerCapture(e.pointerId);
    });
    this.header.addEventListener('pointermove', (e: PointerEvent) => {
      if (!dragging) return;
      this.geometry.x = this.clampX(origX + e.clientX - startX);
      this.geometry.y = this.clampY(origY + e.clientY - startY);
      this.panel.style.left = `${this.geometry.x}px`;
      this.panel.style.top = `${this.geometry.y}px`;
      this.updateAttachPlacement();
      this.updatePlaylistPlacement();
    });
    this.header.addEventListener('pointerup', (e: PointerEvent) => {
      if (!dragging) return;
      dragging = false;
      this.header.releasePointerCapture(e.pointerId);
      this.scheduleGeometrySave();
    });
    this.header.addEventListener('dblclick', () => this.setCollapsed(!this.geometry.collapsed));
  }

  private handlePanelResize(): void {
    // 폭이 실제로 바뀌었을 때만 글자 배율을 다시 건다 — 매 옵저버 콜백마다 CSS 변수를
    // 쓰면 (변수가 레이아웃을 바꾸지 않더라도) 옵저버가 자기 자신을 다시 부를 여지를
    // 남긴다. 값이 그대로면 아무 것도 하지 않는 것이 그 여지를 원천 차단한다.
    const w = this.panel.clientWidth;
    if (w !== this.lastFontWidth) {
      this.lastFontWidth = w;
      this.applyFontScale();
      this.lane.setOptions({ fontScale: this.settings.pitchFontScale * this.widthFontScale() });
      this.renderLane();
    }
    // 레인 열 폭은 패널 폭에 대해 상대적으로 클램프되고, 패널이 아주 좁아지면 2단 자체가
    // 가로 띠로 접힌다 — 그 판정을 모두 쥔 applyLanePlacement를 그대로 다시 태운다.
    // 저장값(mainLaneWidth)은 건드리지 않으므로 패널을 도로 넓히면 원래 폭으로 돌아온다.
    // attached도 함께 본다 — 부착 패널의 높이·폴백 판정이 패널 크기(geometry.width/height)에
    // 종속되므로, 아래에서 geometry를 갱신한 **뒤에** 다시 그려야 한 프레임 지연 없이 맞는다.
    if (this.laneShown && this.settings.mainLanePos === 'left') {
      this.applyLanePlacement();
      this.renderLane();
    }
    // filled는 창이 곧 패널이라 저장할 «기하»가 없다 — 값을 붙들고 있을 이유도 없다
    if (this.applyingGeometry || this.applyingSettings || this.geometry.collapsed
      || this.chrome === 'filled') {
      this.updateAttachPlacement();
      this.updatePlaylistPlacement();
      return;
    }
    const { offsetWidth, offsetHeight } = this.panel;
    if (offsetWidth !== this.geometry.width || offsetHeight !== this.geometry.height) {
      this.geometry.width = offsetWidth;
      this.geometry.height = offsetHeight;
      this.scheduleGeometrySave();
    }
    this.updateAttachPlacement();
    this.updatePlaylistPlacement();
  }

  private handleWindowResize = (): void => {
    this.geometry.x = this.clampX(this.geometry.x);
    this.geometry.y = this.clampY(this.geometry.y);
    this.panel.style.left = `${this.geometry.x}px`;
    this.panel.style.top = `${this.geometry.y}px`;
    this.updateAttachPlacement();
    this.updatePlaylistPlacement();
  };

  private handleFullscreenChange = (): void => {
    const wasHidden = this.fullscreenHidden;
    this.fullscreenHidden = (this.doc ?? document).fullscreenElement !== null;
    this.updateHostVisibility();
    // 전체화면 동안 도착해 못 본 알림을 해제 직후 다시 띄운다 — 타이머도 여기서 처음부터.
    // 칩은 이미 DOM에 그려져 있으므로 문구가 바뀌지 않고 표시 시간만 새로 주어진다.
    if (wasHidden && !this.fullscreenHidden && this.pendingNotice) {
      const { text, autoHideMs } = this.pendingNotice;
      this.setNoticeChip(text, autoHideMs);
    }
  };

  // 클램프는 «패널이 창보다 작다»는 전제에서만 뜻이 있다 — 마운트된 창을 기준으로 재고,
  // filled면 애초에 호출되지 않는다(applyGeometry·handleWindowResize·드래그가 전부 죽는다).
  private clampX(x: number): number {
    const w = this.win?.innerWidth ?? window.innerWidth;
    return Math.min(Math.max(x, EDGE_MARGIN), Math.max(EDGE_MARGIN, w - this.geometry.width - EDGE_MARGIN));
  }

  private clampY(y: number): number {
    const h = this.win?.innerHeight ?? window.innerHeight;
    return Math.min(Math.max(y, EDGE_MARGIN), Math.max(EDGE_MARGIN, h - 48));
  }

  private updateHostVisibility(): void {
    this.host.style.display = this.visible && !this.fullscreenHidden ? '' : 'none';
  }

  private scheduleGeometrySave(): void {
    // filled 인스턴스의 "기하"는 창이 정하는 것이라 저장할 값이 없다 — 저장하면 PiP 창
    // 크기가 유튜브 페이지 패널의 저장된 좌표를 덮어쓴다
    if (this.chrome === 'filled') return;
    clearTimeout(this.saveGeomTimer);
    this.saveGeomTimer = window.setTimeout(() => {
      this.callbacks.onGeometryChange({ ...this.geometry });
    }, 400);
  }
}
