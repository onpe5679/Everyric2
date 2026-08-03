import { t } from '../lib/i18n';
import type { Settings } from '../types';
import type { ThemeName } from '../lib/theme';
import { h, icon } from './dom';
import { LyricsOverlay, type OverlayCallbacks } from './overlay';
import { hasNext, hasPrevious, playNext, playPrevious } from '../lib/yt-player';

interface DocumentPictureInPictureApi {
  requestWindow(options?: { width?: number; height?: number }): Promise<Window>;
}

export interface PipOptions {
  /**
   * PiP 문서에 세울 «두 번째 가사 패널»의 재료.
   *
   * 이 창은 가사 UI를 스스로 그리지 않는다 — 메인 가사창과 **완전히 같은 클래스**
   * (LyricsOverlay)를 chrome:'filled'로 한 번 더 세우고, 화면 구현은 그쪽 하나만 남긴다.
   * 예전에는 이 파일이 스테이지·패널·레인·칩을 반쪽씩 다시 구현해서, 메인 창에만 있는
   * 검색·오프셋·전사·생성·설정이 PiP에는 통째로 없었다(운영자 질책의 근원).
   */
  settings: Settings;
  callbacks: OverlayCallbacks;
  /** 열 때 영상 미러 영역을 포함한 크기로 열지 여부 */
  showVideo: boolean;
  /** 창 너비(px) — 호출부가 저장값/기존 기본값을 미리 계산해서 넘긴다 */
  width: number;
  /** 창 높이(px) — 호출부가 저장값/기존 기본값을 미리 계산해서 넘긴다 */
  height: number;
  /** 창 크기가 바뀐 채로 닫힐 때(pagehide) 마지막 크기를 알려준다 — 설정에 저장용 */
  onSizeChange: (width: number, height: number) => void;
  /** 중앙 열 안에서 영상이 차지하는 세로 비율 (0 = 자동 16:9) — 나머지는 가사 단축 표시 몫 */
  initialVideoRatio: number;
  /** 가로 디바이더 드래그로 영상 비율 변경 완료 시 */
  onVideoRatioChange: (ratio: number) => void;
  /** 세로 디바이더로 레인 열 폭이 바뀌었을 때 (설정 attachedLaneWidth를 이어 쓴다) */
  onLaneColWidthChange: (px: number) => void;
  /** 세로 디바이더로 가사창 열 폭이 바뀌었을 때 (설정 pipPanelWidth) */
  onPanelColWidthChange: (px: number) => void;
  /** 열 때의 테마 — content가 lib/theme.resolveTheme로 판정한 값. 이후 갱신은 setTheme */
  theme: ThemeName;
  /** 크로마키 스트리밍 모드 — 'off'가 아니면 PIP 문서 배경을 단색 키 컬러로 (OBS 키잉용) */
  pipChromaKey: 'off' | 'green' | 'blue' | 'magenta';
  /** 가사 라인 클릭·화살표 시크 — 가사 타임라인(초) 기준 */
  onSeek: (time: number) => void;
  /** 진행 바 클릭 — 영상 길이 대비 비율(0..1) */
  onSeekRatio: (ratio: number) => void;
  onPlayPause: () => void;
  /** 디버그 표시 토글 — PiP 창에 포커스가 있을 때의 Alt+Shift+D 진입점.
   *
   *  메인 패널에서는 이 키를 브라우저가 `chrome.commands`로 처리하지만, Document PiP는
   *  별도 최상위 창이라 그 경로가 닿지 않는다(키 이벤트가 이 document에서 발생한다).
   *  같은 키가 창에 따라 되기도 안 되기도 하면 안 되므로 여기서 직접 받는다. */
  onToggleDebug: () => void;
  /** 볼륨 슬라이더 (0..1) — 원본 video에 적용 */
  onVolumeChange: (volume: number) => void;
  onMuteToggle: () => void;
  /** 멜로디 재생 초기 상태 + 토글 (footer 버튼) */
  melodyOn: boolean;
  onMelodyToggle: () => void;
  /** 메트로놈 초기 상태 + 토글 (footer 버튼) */
  metronomeOn: boolean;
  onMetronomeToggle: () => void;
  /** 좌상단 미니 버튼 — PiP 영상 표시 토글 (설정 pipShowVideo) */
  onVideoToggle: (on: boolean) => void;
  /** 좌상단 미니 버튼 — 가사창 열 접기/펴기 (설정 pipShowPanel) */
  onPanelToggle: (on: boolean) => void;
  /** 좌상단 미니 버튼 — 발음 이중표시 줄 위치 순환 (설정 pitchPronPosition) */
  onDualPositionChange: (position: Settings['pitchPronPosition']) => void;
  /** 좌상단 미니 버튼 — 중앙 열(영상·단축 표시·재생 컨트롤) 접기/펴기 */
  onCenterToggle: (on: boolean) => void;
  /** 좌상단 미니 버튼 — 열 토글 3종. 패널 퀵 줄과 **같은 설정**을 뒤집지만, 그 줄은
   *  가사창 열 안이라 접히면 사라진다 — 코너는 어떤 열이 접혀도 살아 있다(고아 방지). */
  onLaneToggle: (on: boolean) => void;
  onPlaylistToggle: (on: boolean) => void;
  onShortLyricsToggle: (on: boolean) => void;
  onClosed: () => void;
}

const PLAY_SVG = '<svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>';
const PAUSE_SVG = '<svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M6 5h4v14H6zM14 5h4v14h-4z"/></svg>';
const VOLUME_SVG = '<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3a4.5 4.5 0 0 0-2.5-4v8a4.5 4.5 0 0 0 2.5-4zM14 3.2v2.1a7 7 0 0 1 0 13.4v2.1a9 9 0 0 0 0-17.6z"/></svg>';
const MUTED_SVG = '<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M3 9v6h4l5 5V4L7 9H3zm13.6 3 2.7-2.7-1.4-1.4-2.7 2.7-2.7-2.7-1.4 1.4 2.7 2.7-2.7 2.7 1.4 1.4 2.7-2.7 2.7 2.7 1.4-1.4-2.7-2.7z"/></svg>';
const SCREEN_SVG = '<svg viewBox="0 0 24 24" width="13" height="13" fill="currentColor"><path d="M3 4h18v13H3V4zm2 2v9h14V6H5zm3 13h8v2H8v-2z"/></svg>';
/** 코너 — 가라오케 레인 열 (피아노롤 막대) */
const LANE_SVG = '<svg viewBox="0 0 24 24" width="13" height="13" fill="currentColor"><path d="M3 5h18v2H3zm0 4h12v2H3zm0 4h18v2H3zm0 4h9v2H3z"/></svg>';
/** 코너 — 재생목록 열 (목록 줄 + 재생 삼각형) */
const PLAYLIST_SVG = '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h12"/><path d="M3 12h12"/><path d="M3 18h8"/><path d="M17 14.5v5l4.5-2.5z" fill="currentColor" stroke="none"/></svg>';
/** 코너 — 영상 아래 «가사 한 줄»(상자 + 가운데 한 줄) */
/** 코너 — 중앙 열(영상 + 한 줄 + 컨트롤): 상자 안 가운데 세로 강조 구역 */
const CENTER_SVG = '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="4" width="18" height="16" rx="2"/><rect x="9" y="4" width="6" height="16" fill="currentColor" stroke="none" opacity="0.75"/></svg>';
const SHORT_SVG = '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="4" width="18" height="16" rx="2"/><line x1="6.5" y1="12" x2="17.5" y2="12" stroke-linecap="round"/></svg>';
/** 코너 — 발음 이중표시 줄 위치 순환. 상자 + 강조 띠 위치로 «어디에 뜨는가»를 그린다 */
const DUAL_OFF_SVG = '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="4" width="18" height="16" rx="2"/><line x1="4.5" y1="19.5" x2="19.5" y2="4.5"/></svg>';
const DUAL_BOTTOM_SVG = '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="4" width="18" height="16" rx="2"/><rect x="3" y="15" width="18" height="5" fill="currentColor" stroke="none" opacity="0.75"/></svg>';
const DUAL_CENTER_SVG = '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="4" width="18" height="16" rx="2"/><rect x="3" y="10" width="18" height="5" fill="currentColor" stroke="none" opacity="0.75"/></svg>';
const DUAL_BOTH_SVG = '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="4" width="18" height="16" rx="2"/><rect x="3" y="9" width="18" height="4" fill="currentColor" stroke="none" opacity="0.75"/><rect x="3" y="15.5" width="18" height="4" fill="currentColor" stroke="none" opacity="0.75"/></svg>';

/** 이중표시 줄 순환 — 끄기는 한 바퀴 끝에 둔다(한 번 눌렀을 때 사라지는 게 아니라 늘어나야 한다) */
const DUAL_CYCLE: Record<Settings['pitchPronPosition'], Settings['pitchPronPosition']> = {
  off: 'bottom', bottom: 'center', center: 'both', both: 'off',
};
const DUAL_ICON: Record<Settings['pitchPronPosition'], string> = {
  off: DUAL_OFF_SVG, bottom: DUAL_BOTTOM_SVG, center: DUAL_CENTER_SVG, both: DUAL_BOTH_SVG,
};

const PREV_SVG = '<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M6 6h2.5v12H6zm12 12-9-6 9-6z"/></svg>';
/** 코너 미니 버튼 — 가사창 열 접기/펴기. 상자 + 오른쪽 강조 구역(그 열이 오른쪽에
 *  붙는다는 암시) — overlay.ts의 MINI_POS_LEFT_SVG와 같은 도안 계열이다. */
const PANEL_SVG = '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="4" width="18" height="16" rx="2"/><rect x="14" y="4" width="7" height="16" fill="currentColor" stroke="none" opacity="0.75"/></svg>';
const NEXT_SVG = '<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M15.5 6H18v12h-2.5zM6 18l9-6-9-6z"/></svg>';

const MIN_VIDEO_RATIO = 0.15;
const MAX_VIDEO_RATIO = 0.75;

/** 열 폭 하한 — 이보다 좁으면 그 열은 읽을 수 없어 «있으나 마나»가 된다 */
const LANE_COL_MIN = 140;
const PANEL_COL_MIN = 280;
const CENTER_COL_MIN = 200;
const PLAYLIST_COL_W = 220;
/** 세로 디바이더 폭 (CSS .ey-pip-vdivider와 같은 값) — 접힘 계산에 필요하다 */
const VDIVIDER_W = 8;
/** 열 폭 저장값 클램프 — 화면 밖으로 나가는 것만 막고 상한은 두지 않는다 */
function clampColWidth(px: number, min: number): number {
  return Math.round(Math.max(min, Number.isFinite(px) && px > 0 ? px : min));
}

/**
 * 재생 상태가 이 시간보다 오래 갱신되지 않으면 시간 보간을 멈춘다(마지막 시각에 정지).
 *
 * 메인 창이 tick을 끊는 상황은 정상적으로 존재한다 — 내비게이션, 싱크 없는 곡, engine.stop().
 * 그때도 paused=false인 마지막 상태로 계속 보간하면 레인이 혼자 끝없이 흘러간다.
 * 숨은 탭의 timeupdate는 ~250ms 간격이라 이 상한에 걸리지 않는다.
 */
const STATE_STALE_MS = 1000;

/**
 * PiP 창의 rAF 콜백이 이보다 오래 돌지 않았으면 **메인 창 tick이 대신 그린다** (자기치유 안전망).
 *
 * 왜 있나: 이 클래스는 "PiP 창은 숨은 탭에서도 자기 rAF가 계속 돈다"는 전제 위에 렌더를
 * 옮겼는데, **그 전제를 실측으로 확인하지 못했다.** 보이는 상태에서 PiP rAF가 120/s로 돌고
 * 레인이 그것으로 갱신되는 것은 확인됐지만(실측), 진짜로 숨겨진 탭은 자동화로 만들 수
 * 없었다 — Playwright가 백그라운드 스로틀링을 끄는 플래그를 기본으로 넣고, 그 플래그를
 * 제거하고 다른 탭을 앞으로 가져와도 `document.hidden`이 계속 false였다.
 *
 * 전제가 거짓이면 숨은 탭에서 렌더가 **0**이 되어 예전(tick 4Hz)보다 나빠진다. 확인할 수
 * 없는 전제 위에 하한을 두지 않으려고 이 안전망을 둔다. 판단 근거는 "칠했나"가 아니라
 * **"rAF가 살아 있나"**(lastFrameAt)다 — 그래서 결과가 두 갈래로 깔끔하게 갈린다:
 *   - rAF가 살아 있으면 lastFrameAt이 매 프레임(~8~16ms) 갱신돼 이 조건이 절대 성립하지
 *     않는다 → **이중 렌더 없음** (tick이 60Hz로 와도 마찬가지)
 *   - rAF가 죽어 있으면 lastFrameAt이 영원히 낡아 **매 tick** 발동한다 → 렌더 주기가 정확히
 *     예전으로 되돌아간다(보이는 탭 60Hz, 숨은 탭 ~4Hz). 즉 **최악이 예전과 같다.**
 *
 * **지우지 마라.** 중복처럼 보이지만 전제가 확인되기 전까지 이것이 렌더 주기의 유일한
 * 하한이다. 숨은 탭에서 PiP rAF가 유지된다는 것이 실측되면 그때 함께 지워도 된다.
 */
const RENDER_FALLBACK_MS = 200;

/**
 * 곡 시간이 멈춰 있을 때(일시정지·내비게이션 후 상태 정지)의 최소 재그리기 간격(ms).
 *
 * 시각이 그대로면 레인 내용이 바뀔 이유가 거의 없는데 60~120Hz로 캔버스를 다시 칠하는 것은
 * 낭비다. 다만 **아예 건너뛰지는 않고 늦추기만** 한다: 멈춘 동안에도 설정(마디 ±·글자
 * 크기·발음 위치·신뢰도 색)은 바뀔 수 있고 그것들이 화면에 반영되는 유일한 경로가 다음
 * 렌더다. 100ms면 설정을 바꾼 사람 눈에는 즉시이고, 재그리기 비용은 6분의 1 이하로 준다.
 */
const IDLE_REDRAW_MS = 100;

// PiP 창 크기 기억 — 비정상적으로 작은 값이 저장/전달돼 창이 못 쓰게 되지 않도록 클램프.
// 상한은 «화면 밖으로 나가지 않는다»는 물리 한계만 본다: 예전의 960/1280 고정 상한은
// 가사 목록 컬럼 유무에 따라 갈리던 값인데, 창 안이 통째로 메인 패널과 같은 UI가 된
// 지금은 넓게 쓸 이유가 얼마든지 있다(운영자 지시: "이런 건 왜 제한이 있는지 모르겠어").
const MIN_PIP_WIDTH = 280;
const MIN_PIP_HEIGHT = 200;
/** 저장값 오염 방어용 절대 상한 — 실제 상한은 아래 화면 크기 클램프가 정한다 */
const ABS_PIP_MAX = 4096;
function clampPipSize(width: number, height: number): { width: number; height: number } {
  // screen이 없는 환경(테스트 하네스)에서는 절대 상한만 본다
  const maxW = Math.min(ABS_PIP_MAX, window.screen?.availWidth || ABS_PIP_MAX);
  const maxH = Math.min(ABS_PIP_MAX, window.screen?.availHeight || ABS_PIP_MAX);
  return {
    width: Math.round(Math.min(maxW, Math.max(MIN_PIP_WIDTH, width))),
    height: Math.round(Math.min(maxH, Math.max(MIN_PIP_HEIGHT, height))),
  };
}

/** 스페이스바/화살표 단축키를 가로챌지 판단 — 입력 요소에 포커스가 있으면 무시 */
function isTypingTarget(el: EventTarget | null): boolean {
  if (!(el instanceof HTMLElement)) return false;
  const tag = el.tagName;
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || el.isContentEditable;
}

/**
 * Document Picture-in-Picture 가사 창.
 *
 * **이 클래스가 하는 일은 창 하나를 열고 그 안에 «영상 미러 + 가사 패널 + 재생 컨트롤»을
 * 세로로 쌓는 것뿐이다.** 가운데의 가사 패널은 메인 가사창과 같은 LyricsOverlay 인스턴스라,
 * 검색·오프셋·번역·발음·레인·설정·공지·기여·별점이 배선 없이 전부 따라온다. 두 창의 모습이
 * 갈라질 수 없는 이유가 바로 이것이다 — 화면 구현이 애초에 하나뿐이다.
 *
 * - manifest 주입 CSS는 PiP 문서에 적용되지 않으므로 CSS 텍스트를 직접 주입한다
 *   (문서 <style> 하나 + 패널 인스턴스의 Shadow DOM 안에 하나).
 * - 브라우저는 PiP 창을 하나만 허용하므로, 유튜브 네이티브 PiP 대신
 *   video.captureStream() 미러로 영상을 함께 표시한다 (원본 재생/오디오는 탭에 유지).
 */
export class PipController {
  private win: Window | null = null;
  /**
   * 이 창의 가사 패널 — 메인 창과 같은 클래스의 **두 번째 인스턴스**.
   * 창이 닫힐 때 반드시 destroy한다(안 하면 닫힌 문서의 옵저버·타이머가 통째로 남는다 —
   * overlay.ts destroy 주석의 누수 3종이 정확히 이 경로에서 드러났다).
   */
  private panel: LyricsOverlay | null = null;
  private videoWrapEl: HTMLDivElement | null = null;
  private dividerEl: HTMLDivElement | null = null;
  private mirrorStream: MediaStream | null = null;
  private videoRatio = 0;
  private footerEl: HTMLDivElement | null = null;
  private progressEl: HTMLDivElement | null = null;
  private playBtn: HTMLButtonElement | null = null;
  private muteBtn: HTMLButtonElement | null = null;
  private volumeSlider: HTMLInputElement | null = null;
  private volumeDragging = false;
  private timeEl: HTMLSpanElement | null = null;
  private lastPaused: boolean | null = null;
  private lastMuted: boolean | null = null;
  private paused = false;
  private lastTime = 0;
  /**
   * 메인 창이 밀어넣는 최신 재생 상태 — PiP는 이것을 **보간해 자기 rAF로** 그린다.
   *
   * 왜: 메인 창의 rAF(sync-engine)는 탭이 숨으면 멈추고 timeupdate(~4Hz)만 남는다.
   * PiP 렌더가 그 tick에 실려 있으면, **탭을 떠나 보는 것이 존재 이유인 이 창이** 초당
   * 4프레임으로 끊긴다(레인·가사 채움 전부). PiP는 별도 최상위 창이라 자기 rAF는 계속
   * 돌므로, 메인 창은 상태(시각·재생 여부·배속)만 공급하고 그리기는 이 창이 한다.
   *
   * 이 규약이 이중 인스턴스에서도 그대로 살아 있어야 한다 — 그래서 content는 PiP 인스턴스에
   * `updateTime`을 **방송하지 않는다**(그러면 4Hz로 떨어진다). 대신 아래 renderFrame이
   * 이 창의 rAF에서 `panel.updateTime()`을 부른다. 그리는 곳이 한 곳뿐이라 이중 렌더도 없다.
   *
   * at은 상태를 받은 시각(메인 창 performance.now() — 이 클래스는 content script에서
   * 돌므로 rAF 콜백에서 읽는 값과 같은 시계다).
   */
  private state = { time: 0, duration: 0, paused: true, at: 0 };
  /** PiP 창의 rAF 핸들 — pagehide에서 반드시 취소한다(창이 닫힌 뒤 도는 루프 = 누수) */
  private rafId = 0;
  /**
   * PiP 창의 rAF 콜백이 마지막으로 **돈** 시각 — 안전망의 판단 근거는 "rAF가 살아 있나"다.
   * 실제로 칠했는지가 아니라 콜백이 돌았는지를 기록한다(일시정지 간격에 걸려 안 칠한
   * 프레임도 rAF가 산 증거다) — 두 기준을 한 필드로 겸하면 임계값이 서로를 흔든다.
   */
  private lastFrameAt = 0;
  /** renderFrame이 마지막으로 실제로 칠한 시각 — 일시정지 재그리기 간격(IDLE_REDRAW_MS) 기준 */
  private lastPaintAt = 0;
  /** 크로마키 스트리밍 모드 — applyChroma가 문서 루트 클래스·CSS 변수로 반영 */
  private chromaKey: 'off' | 'green' | 'blue' | 'magenta' = 'off';
  /**
   * 원본 video의 재생 배속 — 숨은 탭에서 상태 사이를 보간할 때(sampleTime) 쓴다.
   * 레인의 마이크 궤적 보정에도 같은 값이 필요해 패널 인스턴스에도 함께 밀어넣는다.
   */
  private playbackRate = 1;
  /** 창을 가로로 나눈 열들 — [레인][중앙(영상·단축가사·컨트롤)][가사창][재생목록].
   *  자리는 전부 flex가 정한다: 절대좌표를 쓰면 좁은 창에서 구석에 뭉친다(운영자 우려). */
  private rowEl: HTMLDivElement | null = null;
  private laneColEl: HTMLDivElement | null = null;
  private centerColEl: HTMLDivElement | null = null;
  private playlistColEl: HTMLDivElement | null = null;
  private stageEl: HTMLDivElement | null = null;
  /** 열 사이 세로 디바이더 — 레인↔중앙, 중앙↔가사창 */
  private laneDividerEl: HTMLDivElement | null = null;
  private panelDividerEl: HTMLDivElement | null = null;
  /** 열 폭(px) — 레인은 attachedLaneWidth, 가사창은 pipPanelWidth를 이어받는다 */
  private laneColW = 300;
  private panelColW = 360;
  /** 사용자가 «가사창 열을 펼쳐 두겠다»고 정한 값(설정 pipShowPanel) — 좁아서 자동으로
   *  접히는 것과는 별개다. 자동 접힘은 설정을 건드리지 않아 창을 넓히면 그대로 돌아온다. */
  private panelWanted = true;
  /** 좁은 창에서 자동으로 접힌 열 — 사용자 설정은 건드리지 않아 넓히면 그대로 돌아온다 */
  private autoCollapsed = { lane: false, panel: false, playlist: false };
  /** 창 크기 저장 디바운스 — 드래그 중 매 픽셀 저장하면 storage 쓰기가 폭주한다 */
  private sizeSaveTimer = 0;
  private sizeObserver: ResizeObserver | null = null;
  /** 좌상단 미니 버튼 — PiP 영상 표시 토글. 나머지 토글은 패널 자신의 퀵 줄에 있다 */
  private cornerVideoBtn: HTMLButtonElement | null = null;
  /** 코너 미니 버튼 — 가사창 열 접기/펴기 (설정 pipShowPanel) */
  private cornerPanelBtn: HTMLButtonElement | null = null;
  /** 코너 미니 버튼 — 중앙 열(영상·단축·컨트롤) 접기/펴기 (설정 pipShowCenter).
   *  이걸 끄고 가사창·재생목록도 끄면 «가라오케 단독 모드»가 된다. */
  private cornerCenterBtn: HTMLButtonElement | null = null;
  /** 코너 미니 버튼 — 발음 이중표시 줄 순환 (설정 pitchPronPosition).
   *  "이중표시 기능 어디 갔지" 제보가 있어 **발견 가능성이 인수 기준**이다 */
  private cornerDualBtn: HTMLButtonElement | null = null;
  private dualPos: Settings['pitchPronPosition'] = 'off';
  /** 레인 열과 중앙 열을 맞바꿨는가 (설정 pipLaneSwapped) */
  private laneSwapped = false;
  /**
   * 코너의 «열 토글» 3종 — 레인·재생목록·단축 표시.
   *
   * **불변식(운영자 지시 2026-08-04): 어떤 열의 제어 수단이 다른 접힘 가능한 열 안에만
   * 있으면 안 된다.** 이 셋은 패널의 퀵 줄에도 있지만 그 퀵 줄은 «가사창 열» 안이라,
   * 가사창을 접는 순간 레인을 켜 놓고도 끌 방법이 사라졌다(실사용 제보). 코너는
   * 열 행 **바깥**의 툴바 줄이라 어떤 열이 접혀도 살아남으므로, 여기 두면 그 고아 상태가 원리적으로
   * 생기지 않는다 — 퀵 줄은 «가까운 길», 코너는 «항상 있는 길»이다.
   */
  private cornerLaneBtn: HTMLButtonElement | null = null;
  private cornerPlaylistBtn: HTMLButtonElement | null = null;
  private cornerShortBtn: HTMLButtonElement | null = null;
  private laneOn = true;
  /** 사용자가 중앙 열을 펼쳐 두겠다고 정한 값 (설정 pipShowCenter) */
  private centerWanted = true;
  private playlistOn = true;
  private shortOn = true;
  /** PiP 영상 표시 '설정' 상태 — 미러 성공 여부(DRM 실패 등)와 무관한 사용자 의도 */
  private videoOn = false;
  /** 현재 창에 칠해진 테마 — 판정은 content(lib/theme)가 하고 여기선 받아 쓰기만 한다 */
  private theme: ThemeName = 'dark';

  // ── 재생 컨트롤 (유튜브 DOM 조작 — lib/yt-player.ts) ─────────────
  private prevBtn: HTMLButtonElement | null = null;
  private nextBtn: HTMLButtonElement | null = null;
  // open()의 requestWindow await 동안 두 번째 open()을 차단 — 고아 PiP 창 방지(아래 주석)
  private opening = false;

  static isSupported(): boolean {
    return 'documentPictureInPicture' in window;
  }

  isOpen(): boolean {
    return this.win !== null;
  }

  /**
   * 이 창의 가사 패널 인스턴스 — 창이 닫혀 있으면 null.
   *
   * content는 «어느 창에 그릴지»를 알 필요가 없다: 살아 있는 패널 목록(panels())에만 대고
   * 말하고, 그 목록이 이 함수로 만들어진다.
   */
  panelInstance(): LyricsOverlay | null {
    return this.panel;
  }

  async open(cssText: string, opts: PipOptions): Promise<boolean> {
    if (this.win) return true;
    // 재진입 가드 — `if (this.win)` 검사와 `this.win = win` 대입 사이에 await가 끼어
    // 있어, 토글을 빠르게 두 번 누르면 requestWindow가 두 번 나가고 먼저 열린 창이
    // 고아가 됐다. 고아의 rAF는 스스로 멈추지만 captureStream 영상 미러는 브라우저
    // 미디어 파이프라인이 계속 프레임을 밀어 넣어, SPA 세션 내내 누적되면 메인 탭이
    // 5fps대로 주저앉는다(새로고침으로만 해소 — fps 추적 감사, 2026-08-03).
    if (this.opening) return false;
    this.opening = true;
    try {
      const api = (window as unknown as { documentPictureInPicture?: DocumentPictureInPictureApi })
        .documentPictureInPicture;
      if (!api) return false;

      const { width, height } = clampPipSize(opts.width, opts.height);
      let win: Window;
      try {
        win = await api.requestWindow({ width, height });
      } catch {
        return false;
      }
      if (this.win) {
        // 가드가 있어도 도달할 수 있는 마지막 창구(예: 외부에서 open을 직접 두 번
        // 호출) — 늦게 온 창을 즉시 닫아 고아를 만들지 않는다.
        try { win.close(); } catch { /* 이미 닫힌 창 */ }
        return true;
      }
      return this.finishOpen(win, cssText, opts);
    } finally {
      this.opening = false;
    }
  }

  private finishOpen(win: Window, cssText: string, opts: PipOptions): boolean {
    this.win = win;
    this.videoRatio = opts.initialVideoRatio;
    this.theme = opts.theme;
    this.chromaKey = opts.pipChromaKey;

    const doc = win.document;
    doc.title = t('pip.docTitle');
    const style = doc.createElement('style');
    style.textContent = cssText;
    doc.head.append(style);
    doc.body.className = 'ey-pip';
    // 라이트 테마는 :root에 걸어야 한다 — 레인 캔버스 색을 readPitchColors가
    // documentElement의 계산된 CSS 변수에서 읽기 때문이다 (body에만 걸면 캔버스가 다크로 남는다)
    this.applyTheme();
    this.applyChroma();

    // 영상 미러 영역 + 비율 조절 디바이더 (attachVideo 전까지 숨김)
    this.videoWrapEl = h('div', { className: 'ey-pip-video' });
    this.videoWrapEl.style.display = 'none';
    this.dividerEl = this.buildDivider(win, opts.onVideoRatioChange);
    this.dividerEl.style.display = 'none';

    this.playBtn = h('button', {
      className: 'ey-pip-play',
      title: t('pip.controls.playPause'),
      on: { click: () => opts.onPlayPause() },
    }, icon(PLAY_SVG));

    // 이전/다음 곡 — 유튜브 비공식 마크업 의존(lib/yt-player.ts).
    // 셀렉터가 안 맞으면 클릭이 false를 돌려주므로 그 자리에서 비활성으로 내린다.
    // 재생목록 «목록»은 패널 자신의 부착 재생목록 패널(설정 modPlaylist)이 담당한다 —
    // 예전의 창 안 반쪽 목록은 메인과 다른 UI였고, 지금은 같은 조각이 창 안에 들어온다.
    this.prevBtn = h('button', {
      className: 'ey-pip-play ey-pip-mute',
      title: t('pip.controls.prevTrack'),
      on: { click: () => { if (!playPrevious()) this.refreshPlayerControls(); } },
    }, icon(PREV_SVG));
    this.nextBtn = h('button', {
      className: 'ey-pip-play ey-pip-mute',
      title: t('pip.controls.nextTrack'),
      on: { click: () => { if (!playNext()) this.refreshPlayerControls(); } },
    }, icon(NEXT_SVG));

    this.muteBtn = h('button', {
      className: 'ey-pip-play ey-pip-mute',
      title: t('pip.controls.mute'),
      on: { click: () => opts.onMuteToggle() },
    }, icon(VOLUME_SVG));

    // **가라오케 컨트롤은 여기 없다.** 멜로디·메트로놈·마디 창·진행 방식·계이름·
    // 카운트다운은 전부 레인 열 머리(.ey-lane-head, overlay.ts buildLaneHead)로 갔다 —
    // «컨트롤은 자기가 제어하는 열에 붙는다»는 운영자 원칙. 이 푸터에는 영상/재생
    // 계열(이전·재생·다음·음소거·볼륨·진행바·시간)만 남는다.

    this.volumeSlider = h('input', {
      className: 'ey-pip-volume',
      title: t('pip.controls.volume'),
      attrs: { type: 'range', min: '0', max: '100', step: '1', value: '100' },
    });
    this.volumeSlider.addEventListener('input', () => {
      opts.onVolumeChange(Number(this.volumeSlider?.value ?? 100) / 100);
    });
    this.volumeSlider.addEventListener('pointerdown', () => { this.volumeDragging = true; });
    this.volumeSlider.addEventListener('pointerup', () => { this.volumeDragging = false; });

    this.progressEl = h('div', { className: 'ey-pip-progress-bar' });
    const progressWrap = h('div', {
      className: 'ey-pip-progress-wrap',
      title: t('pip.controls.progressSeek'),
      on: {
        click: (e: MouseEvent) => {
          const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
          if (rect.width > 0) {
            opts.onSeekRatio(Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width)));
          }
        },
      },
    }, h('div', { className: 'ey-pip-progress' }, this.progressEl));

    this.timeEl = h('span', { className: 'ey-pip-time', text: '0:00 / 0:00' });

    // 좌상단 미니 설정 — 영상 표시만 남는다. 가라오케 레인·자막·재생목록 토글은 패널
    // 자신의 퀵 줄(.ey-quick-row)에 이미 있고, 그 줄이 이 창 안에도 그대로 들어온다.
    this.videoOn = opts.showVideo;
    this.cornerVideoBtn = h('button', {
      className: 'ey-pip-mini',
      title: t('pip.controls.videoToggle'),
      on: { click: () => opts.onVideoToggle(!this.videoOn) },
    }, icon(SCREEN_SVG));
    // 가사창 열 접기/펴기 — **여기 말고는 되돌릴 곳이 없다.** 설정 시트는 그 패널 안에
    // 있어서, 패널을 접고 나면 시트로 다시 펼 수 없다(운영자 원칙: 되돌아올 수 없는
    // 축약 컨트롤 금지 — 코너 발음 버튼이 같은 이유로 5값 순환이 됐다).
    this.panelWanted = opts.settings.pipShowPanel;
    this.centerWanted = opts.settings.pipShowCenter;
    this.cornerCenterBtn = h('button', {
      className: 'ey-pip-mini',
      title: t('pip.controls.centerColToggle'),
      on: { click: () => opts.onCenterToggle(!this.centerWanted) },
    }, icon(CENTER_SVG));
    this.cornerPanelBtn = h('button', {
      className: 'ey-pip-mini',
      title: t('pip.controls.panelColToggle'),
      on: { click: () => opts.onPanelToggle(!this.panelWanted) },
    }, icon(PANEL_SVG));
    // 발음 이중표시 줄 — 코너의 «이 창에 무엇을 띄울까» 묶음 중 하나.
    // 예전에는 코너에 영상 토글 하나만 덩그러니 있어 겉돌았고("영상 표시 버튼이 혼자
    // 따로 놀아"), 이중표시는 설정 시트 깊숙이 있어 아예 못 찾았다("어디 갔지").
    // 셋을 같은 도안 언어(상자 + 강조 구역)로 나란히 두어 «창 표시 묶음»으로 읽히게 한다.
    this.dualPos = opts.settings.pitchPronPosition;
    this.cornerDualBtn = h('button', {
      className: 'ey-pip-mini',
      on: { click: () => opts.onDualPositionChange(DUAL_CYCLE[this.dualPos]) },
    });
    // 열 토글 3종 — 위 필드 주석의 «고아 방지» 불변식을 실현하는 자리다
    this.laneOn = opts.settings.pitchGuide;
    this.playlistOn = opts.settings.pipPlaylist;
    this.shortOn = opts.settings.pipShortLyrics;
    this.cornerLaneBtn = h('button', {
      className: 'ey-pip-mini',
      on: { click: () => opts.onLaneToggle(!this.laneOn) },
    }, icon(LANE_SVG));
    this.cornerPlaylistBtn = h('button', {
      className: 'ey-pip-mini',
      on: { click: () => opts.onPlaylistToggle(!this.playlistOn) },
    }, icon(PLAYLIST_SVG));
    this.cornerShortBtn = h('button', {
      className: 'ey-pip-mini',
      on: { click: () => opts.onShortLyricsToggle(!this.shortOn) },
    }, icon(SHORT_SVG));
    this.laneSwapped = opts.settings.pipLaneSwapped;
    this.syncCornerButtons();

    this.footerEl = h('div', { className: 'ey-pip-footer' },
      h('div', { className: 'ey-pip-controls' },
        this.prevBtn, this.playBtn, this.nextBtn,
        this.muteBtn, this.volumeSlider, progressWrap, this.timeEl),
    );

    // ── 창 구조 (운영자 지시 2026-08-04) ───────────────────────────
    //
    //   [레인 열] │ [중앙 열: 영상 ↑ / 가사 단축 표시 / 재생 컨트롤 ↓] │ [가사창] [재생목록]
    //
    // 전부 **flex 행 하나**다. 예전처럼 세로로 쌓지 않는다: 메인 가사창이 영상 «아래»가
    // 아니라 «오른쪽»에 통째로 서야 하고, 가라오케 레인은 영상 왼쪽에 선다.
    // 중앙 열만 기존 PiP의 세로 구성(영상→단축 가사→컨트롤)을 그대로 유지한다.
    //
    // 부착 패널(레인·재생목록)은 절대좌표 배치 로직을 절대 타지 않는다 — 그 로직은 메인
    // 창 geometry 기준이라 PiP에서 돌면 구석에 뭉친다. overlay.mountInto에 열을 넘겨
    // 그쪽이 flex 자식으로 들어가게 한다(OverlaySlots).
    this.laneColW = clampColWidth(opts.settings.pipLaneWidth, LANE_COL_MIN);
    this.panelColW = clampColWidth(opts.settings.pipPanelWidth, PANEL_COL_MIN);

    this.laneColEl = h('div', { className: 'ey-pip-col ey-pip-lane-col' });
    this.playlistColEl = h('div', { className: 'ey-pip-col ey-pip-playlist-col' });
    // 가사 단축 표시는 패널 인스턴스가 만들어 준다(attachShortView) — 여기서 다시 그리면
    // 가사 렌더 구현이 세 번째로 늘어난다. 지금은 자리만 잡아 둔다.
    this.stageEl = h('div', { className: 'ey-pip-stage-slot' });
    this.centerColEl = h('div', { className: 'ey-pip-col ey-pip-center' },
      this.videoWrapEl, this.dividerEl, this.stageEl, this.footerEl);
    this.laneDividerEl = this.buildColDivider('lane', px => opts.onLaneColWidthChange(px));
    this.panelDividerEl = this.buildColDivider('panel', px => opts.onPanelColWidthChange(px));

    this.rowEl = h('div', { className: 'ey-pip-row' },
      this.laneColEl, this.laneDividerEl, this.centerColEl, this.panelDividerEl);
    // 코너 묶음 = «이 창에 무엇을 띄울까». **모든 열의 토글이 여기 있다**(고아 방지
    // 불변식). 순서는 화면에 놓인 열 순서대로 — 레인 → 중앙(영상·단축) → 가사창 →
    // 재생목록, 그리고 마지막이 레인 내용 옵션(이중표시).
    // 툴바가 **먼저** — body가 세로 흐름이라 append 순서가 곧 위아래다.
    // (예전엔 position:fixed로 떠 있어 순서가 무의미했다.)
    doc.body.append(h('div', { className: 'ey-pip-corner' },
      this.cornerLaneBtn, this.cornerCenterBtn, this.cornerVideoBtn, this.cornerShortBtn,
      this.cornerPanelBtn, this.cornerPlaylistBtn, this.cornerDualBtn), this.rowEl);

    // 패널을 **문서에 붙은 뒤에** 마운트한다 — mountInto가 그 시점의 문서로 레인 캔버스를
    // attach하고 ResizeObserver를 «그 창»에서 만든다(h()가 만든 노드는 append 전까지
    // ownerDocument가 메인 문서라, 순서가 뒤집히면 옵저버가 유튜브 탭에 매달린다).
    this.panel = new LyricsOverlay(cssText, opts.settings, opts.callbacks, null, { chrome: 'filled' });
    this.panel.mountInto(doc, this.rowEl, {
      laneSlot: this.laneColEl,
      playlistSlot: this.playlistColEl,
      // 레인이 스스로 꺼지거나(노트 없는 곡) 재생목록 모듈이 토글되면 열 폭을 다시 나눈다
      onColumnsChanged: () => this.applyColumnLayout(),
    });
    // 재생목록 열은 가사창 **뒤**에 와야 오른쪽에 선다 (mountInto가 host를 먼저 넣는다)
    this.rowEl.append(this.playlistColEl);
    // 단축 표시를 중앙 열에 꽂는다 — 설정이 꺼져 있으면 자리만 비워 둔다
    if (opts.settings.pipShortLyrics) this.panel.attachShortView(this.stageEl);

    this.refreshPlayerControls();
    this.applyColumnLayout();
    win.addEventListener('resize', () => this.applyColumnLayout());
    // 창 크기 영속 — pagehide에만 기대면 창이 비정상 종료될 때 마지막 크기를 잃는다.
    // 크기 변화의 «확정 시점»이 없으므로(사용자가 모서리를 계속 끌 수 있다) 500ms
    // 디바운스로 잦아든 뒤 한 번만 저장한다(chrome.storage 폭주 금지 규약).
    try {
      const RO = (win as unknown as { ResizeObserver?: typeof ResizeObserver }).ResizeObserver
        ?? ResizeObserver;
      this.sizeObserver = new RO(() => this.scheduleSizeSave(opts.onSizeChange));
      this.sizeObserver.observe(doc.documentElement);
    } catch { /* ResizeObserver 미지원 환경 — pagehide 저장으로 폴백 */ }
    // 창이 막 열린 시점에는 innerWidth가 아직 요청 크기로 정착하지 않을 수 있다 —
    // 첫 프레임 뒤에 한 번 더 재서 접힘 판정을 확정한다(실측: 개창 직후 계산이
    // 낡은 폭으로 돌아 열이 접히지 않고 창 밖으로 밀려났다).
    win.requestAnimationFrame(() => this.applyColumnLayout());

    // OS 테마가 바뀌면 캐시된 레인 색이 낡는다 — 다음 렌더에서 CSS 변수 재판독
    try {
      win.matchMedia('(prefers-color-scheme: dark)')
        .addEventListener('change', () => this.panel?.refreshLaneColors());
    } catch { /* matchMedia 미지원 환경은 무시 */ }

    // Document PiP는 별도 최상위 창이라, 포커스가 PiP에 있으면 키 이벤트가
    // 원본 탭이 아니라 이 document에서 발생한다 — 스페이스바 재생/일시정지,
    // 좌우 화살표로 5초 시크. 입력 요소에 포커스가 있으면 가로채지 않는다.
    win.addEventListener('keydown', (e: KeyboardEvent) => {
      // 디버그 토글은 **입력 중에도** 받는다. 메인 패널에서는 브라우저가 chrome.commands로
      // 처리해 포커스와 무관하게 동작하므로, PiP만 입력 필드에서 죽으면 같은 키가 창에 따라
      // 되기도 안 되기도 한다. Alt+Shift 조합은 타이핑을 방해하지 않으니 앞에서 받는다.
      //
      // e.key가 아니라 e.code로 본다 — Alt 조합에서 e.key는 OS·레이아웃에 따라 다른 문자가
      // 되지만(macOS의 Option+Shift+D 등) code는 물리 키라 안정적이다.
      //
      // 한계: 사용자가 chrome://extensions/shortcuts에서 단축키를 재지정하면 메인 패널만
      // 따라가고 이 조합은 그대로다(여기서는 manifest의 기본값을 하드코딩한다).
      if (e.altKey && e.shiftKey && e.code === 'KeyD') {
        e.preventDefault();
        opts.onToggleDebug();
        return;
      }
      if (isTypingTarget(e.target)) return;
      if (e.code === 'Space') {
        e.preventDefault();
        opts.onPlayPause();
      } else if (e.code === 'ArrowRight') {
        e.preventDefault();
        opts.onSeek(this.lastTime + 5);
      } else if (e.code === 'ArrowLeft') {
        e.preventDefault();
        opts.onSeek(Math.max(0, this.lastTime - 5));
      } else if (e.code === 'ArrowUp' || e.code === 'ArrowDown') {
        // 볼륨 ±0.05 — preventDefault로 페이지 스크롤(이 창엔 스크롤할 것도 없지만)을 막는다.
        // onVolumeChange가 볼륨>0일 때 mute도 함께 풀어 준다(content.ts 콜백, 슬라이더와 동일 경로)
        e.preventDefault();
        const cur = Number(this.volumeSlider?.value ?? 100) / 100;
        const next = Math.min(1, Math.max(0, cur + (e.code === 'ArrowUp' ? 0.05 : -0.05)));
        if (this.volumeSlider) this.volumeSlider.value = String(Math.round(next * 100));
        opts.onVolumeChange(next);
      }
    });

    win.addEventListener('pagehide', () => {
      // 닫히기 직전 창 크기를 기억해 두었다가 다음에 열 때 복원 (위치는 브라우저가 자체 재사용)
      if (win.innerWidth > 0 && win.innerHeight > 0) {
        const size = clampPipSize(win.innerWidth, win.innerHeight);
        opts.onSizeChange(size.width, size.height);
      }
      // 프레임 루프를 먼저 끊는다 — 닫힌 창의 rAF가 남으면 누수이고, 아래에서 null로
      // 비우는 엘리먼트를 그 루프가 계속 만지려 한다
      win.cancelAnimationFrame(this.rafId);
      this.rafId = 0;
      this.sizeObserver?.disconnect();
      this.sizeObserver = null;
      clearTimeout(this.sizeSaveTimer);
      this.lastFrameAt = 0;
      this.lastPaintAt = 0;
      this.state = { time: 0, duration: 0, paused: true, at: 0 };
      this.stopMirror();
      this.win = null;
      // 패널 인스턴스를 **반드시** 걷는다 — 닫힌 문서의 ResizeObserver·타이머·rAF가
      // 남으면 여닫을 때마다 누적된다(overlay.ts destroy 주석의 누수 3종).
      // onClosed(아래)가 메인 패널 복원을 판단하므로 그 **전에** 걷어야, content가
      // 이미 사라진 창의 인스턴스에 대고 말하지 않는다.
      this.panel?.destroy();
      this.panel = null;
      this.lastPaused = null;
      this.lastMuted = null;
      this.volumeDragging = false;
      this.videoWrapEl = null;
      this.dividerEl = null;
      this.footerEl = null;
      this.rowEl = null;
      this.laneColEl = null;
      this.centerColEl = null;
      this.playlistColEl = null;
      this.stageEl = null;
      this.laneDividerEl = null;
      this.panelDividerEl = null;
      this.progressEl = null;
      this.playBtn = null;
      this.muteBtn = null;
      this.volumeSlider = null;
      this.timeEl = null;
      this.cornerVideoBtn = null;
      this.cornerPanelBtn = null;
      this.cornerDualBtn = null;
      this.cornerLaneBtn = null;
      this.cornerPlaylistBtn = null;
      this.cornerShortBtn = null;
      this.cornerCenterBtn = null;
      this.videoOn = false;
      this.prevBtn = null;
      this.nextBtn = null;
      this.playbackRate = 1;
      opts.onClosed();
    });

    // 첫 tick이 오기 전에는 0초 상태 — 정지로 두어 보간이 앞서 나가지 않게 한다
    this.state = { time: 0, duration: 0, paused: true, at: performance.now() };
    // 첫 프레임은 지연 없이 그린다(일시정지 간격 가드 통과). lastFrameAt도 0이라 첫 tick이
    // rAF보다 먼저 오면 안전망이 한 번 그리는데, 그것이 이 창의 첫 그림이라 옳다.
    this.lastFrameAt = 0;
    this.lastPaintAt = 0;
    this.startFrameLoop(win); // 이제부터 렌더는 이 창의 rAF가 맡는다 (state 주석 참조)
    return true;
  }

  close(): void {
    this.win?.close();
  }

  /**
   * 페이지의 video를 captureStream으로 미러링해 PiP 상단에 표시.
   * DRM 등으로 캡처가 불가하면 조용히 영역을 숨긴다.
   */
  attachVideo(source: HTMLVideoElement): void {
    if (!this.win || !this.videoWrapEl) return;
    this.stopMirror();
    // 미디어가 아직 없으면(SPA 이동 직후 readyState 0) 캡처를 시도하지 않는다 —
    // bindMirrorRefresh의 loadeddata가 다시 부른다. 여기서 만들어지는 0트랙 스트림도
    // 살아 있는 캡처 sink라, 버리기만 하면 원본 video에 매달린 채 프레임 복사를 계속해
    // 페이지 전체가 계단식으로 느려진다(고아 PIP 미러 5fps와 같은 기전).
    if (source.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      this.hideVideoArea();
      return;
    }
    try {
      const capturable = source as HTMLVideoElement & { captureStream?: () => MediaStream };
      const stream = capturable.captureStream?.();
      if (!stream || stream.getVideoTracks().length === 0) {
        // 참조만 버려선 안 된다 — sink를 명시적으로 끊는다 (위 주석과 같은 이유)
        stream?.getTracks().forEach(track => track.stop());
        this.hideVideoArea();
        return;
      }
      this.mirrorStream = stream;
      const mirror = this.win.document.createElement('video');
      mirror.muted = true;
      mirror.autoplay = true;
      mirror.playsInline = true;
      mirror.srcObject = stream;
      this.videoWrapEl.replaceChildren(mirror);
      this.videoWrapEl.style.display = '';
      this.syncVideoLayout();
      void mirror.play().catch(() => { /* autoplay 실패 시 프레임은 srcObject로도 갱신됨 */ });
    } catch {
      this.hideVideoArea();
    }
  }

  setVideoEnabled(enabled: boolean, source: HTMLVideoElement | null): void {
    if (!this.win) return;
    this.videoOn = enabled;
    this.syncCornerButtons();
    if (enabled && source) {
      this.attachVideo(source);
    } else {
      this.stopMirror();
      this.hideVideoArea();
    }
  }

  /**
   * 창 «레이아웃»에 걸리는 설정 반영 — 가사 단축 표시 on/off와 열 폭.
   *
   * 패널 내용은 content가 applySettings를 방송해 알아서 따라가지만, 이 셋은 패널 바깥
   * (중앙 열·열 폭)의 일이라 창 주인이 직접 받아야 한다.
   */
  applyLayoutSettings(settings: Settings): void {
    if (!this.win || !this.panel || !this.stageEl) return;
    const wantShort = settings.pipShortLyrics;
    const hasShort = this.stageEl.childElementCount > 0;
    if (wantShort && !hasShort) this.panel.attachShortView(this.stageEl);
    else if (!wantShort && hasShort) this.panel.detachShortView();
    this.panelWanted = settings.pipShowPanel;
    this.centerWanted = settings.pipShowCenter;
    this.dualPos = settings.pitchPronPosition;
    this.laneSwapped = settings.pipLaneSwapped;
    this.laneOn = settings.pitchGuide;
    this.playlistOn = settings.pipPlaylist;
    this.shortOn = settings.pipShortLyrics;
    // 레인 열 폭은 **PiP 전용 키**다 — 메인 부착 레인(attachedLaneWidth)과 별개여야
    // 한쪽을 정리해도 다른 쪽이 안 무너진다(표면별 상태 원칙)
    this.laneColW = clampColWidth(settings.pipLaneWidth, LANE_COL_MIN);
    this.panelColW = clampColWidth(settings.pipPanelWidth, PANEL_COL_MIN);
    this.applyColumnLayout();
  }

  /** 크로마키 스트리밍 모드 즉시 반영 — 창이 열려 있으면 배경이 바로 바뀐다 */
  setChromaKey(mode: 'off' | 'green' | 'blue' | 'magenta'): void {
    this.chromaKey = mode;
    this.applyChroma();
  }

  /**
   * 크로마키 배경 적용 — PIP 문서 루트에 클래스·키 컬러 CSS 변수를 건다(overlay.css의
   * :root.ey-chroma 규칙이 배경을 단색으로 통일한다). PIP는 반투명을 지원하지 않아
   * 방송(OBS 등)에서 창을 겹칠 수 없다 — 대신 배경을 표준 키 컬러(green #00b140 등)로
   * 바꿔 스트리머가 크로마키 필터로 배경만 떼어낼 수 있게 한다(운영자 요청, 2026-08-03).
   */
  private applyChroma(): void {
    const doc = this.win?.document;
    if (!doc) return;
    const CHROMA_COLORS: Record<string, string> = {
      green: '#00b140', blue: '#0047bb', magenta: '#ff00ff',
    };
    const color = CHROMA_COLORS[this.chromaKey] ?? '';
    doc.documentElement.classList.toggle('ey-chroma', Boolean(color));
    if (color) doc.documentElement.style.setProperty('--ey-chroma-bg', color);
    else doc.documentElement.style.removeProperty('--ey-chroma-bg');
  }

  /**
   * PiP **문서**의 테마 즉시 반영 — content가 lib/theme.resolveTheme로 판정한 값을 받는다.
   *
   * 패널 자신의 색은 applySettings(모든 인스턴스에 방송된다)가 맡는다. 여기서 남는 몫은
   * 문서 루트다: body 배경·스크롤바·기본 폼 컨트롤이 :root.ey-light 변수를 본다.
   */
  setTheme(theme: ThemeName): void {
    if (this.theme === theme) return;
    this.theme = theme;
    this.applyTheme();
  }

  private applyTheme(): void {
    this.win?.document.documentElement.classList.toggle('ey-light', this.theme === 'light');
  }

  /** 원본 video 배속 — 시간 보간(sampleTime)과 마이크 궤적 시간축 보정용 */
  setPlaybackRate(rate: number): void {
    if (!Number.isFinite(rate) || rate <= 0) return;
    this.playbackRate = rate;
    this.panel?.setLanePlaybackRate(rate);
  }

  /** 좌상단 미니 버튼의 on/off 시각 상태를 현재 설정에 맞춘다 */
  private syncCornerButtons(): void {
    this.cornerVideoBtn?.classList.toggle('on', this.videoOn);
    // 자동 접힘 중에는 «켜져 있지만 지금은 자리가 없다»를 구분해 보여준다 — 툴팁도
    // 사유를 말한다(버튼을 눌렀는데 아무 일도 없는 것처럼 보이지 않게)
    if (this.cornerPanelBtn) {
      this.cornerPanelBtn.classList.toggle('on', this.panelWanted && !this.autoCollapsed.panel);
      this.cornerPanelBtn.classList.toggle('ey-pip-mini-auto', this.autoCollapsed.panel);
      this.cornerPanelBtn.title = this.autoCollapsed.panel
        ? t('pip.controls.autoCollapsed')
        : t('pip.controls.panelColToggle');
    }
    if (this.cornerDualBtn) {
      this.cornerDualBtn.replaceChildren(icon(DUAL_ICON[this.dualPos]));
      this.cornerDualBtn.classList.toggle('on', this.dualPos !== 'off');
      this.cornerDualBtn.title = t('pip.controls.dualPosition');
    }
    // 열 토글 3종 — 자동 접힘 중이면 「켜져 있지만 자리가 없다」를 구분해 말한다
    const colBtn = (
      btn: HTMLButtonElement | null, on: boolean, auto: boolean, titleKey: string,
    ): void => {
      if (!btn) return;
      btn.classList.toggle('on', on && !auto);
      btn.classList.toggle('ey-pip-mini-auto', auto);
      btn.title = auto ? t('pip.controls.autoCollapsed') : t(titleKey);
    };
    colBtn(this.cornerLaneBtn, this.laneOn, this.autoCollapsed.lane, 'pip.controls.laneColToggle');
    colBtn(this.cornerCenterBtn, this.centerWanted, false, 'pip.controls.centerColToggle');
    colBtn(this.cornerPlaylistBtn, this.playlistOn, this.autoCollapsed.playlist,
      'pip.controls.playlistColToggle');
    // 단축 표시는 중앙 열 안이라 자동 접힘 대상이 아니다(중앙 열은 최후 생존)
    colBtn(this.cornerShortBtn, this.shortOn, false, 'pip.controls.shortLyricsToggle');
  }

  // ── 재생 컨트롤 (유튜브 DOM 조작) ───────────────────────────────
  //
  // 유튜브 공식 페이지 API가 없어 lib/yt-player.ts가 플레이어 버튼 DOM을 직접 만진다.
  // 유튜브가 마크업을 개편하면 그쪽 셀렉터가 전부 빗나가는데, 그때는 hasNext/hasPrevious가
  // false를 돌려주므로 여기서 **버튼이 조용히 비활성이 될 뿐** 창은 그대로 동작한다.

  /** 이전/다음 버튼의 사용 가능 여부를 다시 판정한다 */
  refreshPlayerControls(): void {
    if (!this.win) return;
    if (this.prevBtn) this.prevBtn.disabled = !hasPrevious();
    if (this.nextBtn) this.nextBtn.disabled = !hasNext();
  }

  /**
   * 멜로디/메트로놈 상태 반영 — 버튼은 레인 열 머리로 옮겨갔으므로 여기서는 할 일이 없다.
   * content가 applyAudioSettings에서 무조건 부르는 자리라 계약만 남긴다(패널의
   * applySettings가 레인 머리 라벨을 갱신한다).
   */
  setAudioState(_melody: boolean, _metronome: boolean): void {
    /* 레인 머리(overlay.syncLaneHead)가 표시를 맡는다 */
  }

  /**
   * 메인 창이 주는 재생 상태를 받아 둔다 — **평소에는 그리지 않는다.**
   *
   * 시간에 따라 변하는 렌더(가사 채움·진행 바·시간 라벨·레인)는 전부 이 창의 rAF
   * (startFrameLoop → renderFrame)가 맡는다. 근거는 state 필드 주석 참조: 이 함수는
   * 숨은 탭에서 ~4Hz로만 불리므로, 여기서 그리면 PiP가 초당 4프레임이 된다.
   * 상태 변화로만 바뀌는 것(재생/정지 아이콘)은 값이 실제로 바뀔 때 여기서 처리한다.
   *
   * 예외는 아래 안전망 하나뿐이다 — PiP rAF가 돌지 않는 환경에서도 최악이 예전과
   * 같도록(4Hz) 보장한다. 근거는 RENDER_FALLBACK_MS 주석.
   */
  tick(time: number, duration: number, paused: boolean): void {
    if (!this.win) return;
    this.paused = paused;
    // 상태를 **먼저** 갱신한다 — 아래 안전망이 그릴 때 sampleTime()이 방금 받은 시각을
    // 쓰도록(갱신 전에 그리면 STATE_STALE_MS에 걸린 낡은 시각으로 그린다)
    this.state = { time, duration, paused, at: performance.now() };
    if (!paused) this.panel?.clearLaneManualScroll(); // 재생 재개 → 오토스크롤 복귀
    if (this.playBtn && paused !== this.lastPaused) {
      this.lastPaused = paused;
      this.playBtn.replaceChildren(icon(paused ? PLAY_SVG : PAUSE_SVG));
    }
    // 자기치유 안전망: PiP rAF가 살아 있으면 이 조건은 성립하지 않아 아무 일도 없다.
    // 죽어 있으면 lastFrameAt이 갱신되지 않아 **매 tick** 여기서 그린다 — 즉 렌더 주기가
    // 정확히 예전(tick 주기: 보이면 60Hz, 숨으면 ~4Hz)으로 되돌아간다.
    if (performance.now() - this.lastFrameAt > RENDER_FALLBACK_MS) {
      this.renderFrame(this.sampleTime());
    }
  }

  /**
   * PiP 창 자신의 rAF 루프 — 이 창이 열려 있는 동안만 돈다.
   *
   * win을 클로저로 잡고 매 프레임 `this.win !== win`을 확인한다: 창이 닫히거나(pagehide가
   * win을 null로 만든다) 새 창으로 교체되면 그 프레임에서 스스로 끝난다. pagehide도
   * cancelAnimationFrame으로 즉시 끊는다 — 닫힌 창의 루프가 남으면 그게 누수다.
   */
  private startFrameLoop(win: Window): void {
    const loop = (): void => {
      if (this.win !== win) return;
      this.rafId = win.requestAnimationFrame(loop);
      // 콜백이 돌았다는 사실을 먼저 남긴다 — 이것이 안전망의 "rAF 살아 있음" 근거다.
      // renderFrame이 일시정지 간격에 걸려 안 칠하고 돌아와도 rAF는 산 것이다.
      this.lastFrameAt = performance.now();
      this.renderFrame(this.sampleTime());
    };
    this.rafId = win.requestAnimationFrame(loop);
  }

  /**
   * 마지막으로 받은 상태를 벽시계로 보간한 현재 곡 시간.
   * 숨은 탭에서는 상태가 ~4Hz로만 오므로 그 사이를 배속만큼 이어 붙인다 —
   * 250ms마다 실측값으로 교정되므로 어긋남이 누적되지 않는다.
   */
  private sampleTime(): number {
    const s = this.state;
    const age = performance.now() - s.at;
    // 일시정지 중이거나 상태가 낡았으면(메인 창이 tick을 끊었다) 마지막 시각에 멈춘다
    if (s.paused || age > STATE_STALE_MS) return s.time;
    return s.time + (age / 1000) * this.playbackRate;
  }

  /**
   * 한 프레임의 시간 의존 렌더 — 진행 바 + 시간 라벨 + **패널 인스턴스의 가사 채움·레인**.
   * 이 창의 rAF가 매 프레임 부르고, rAF가 죽은 환경에서는 메인 창 tick이 대신 부른다.
   */
  private renderFrame(time: number): void {
    if (!this.win) return;
    const at = performance.now();
    // 곡 시간이 멈춰 있으면 재그리기를 늦춘다 — 건너뛰지는 않는다(IDLE_REDRAW_MS 주석).
    // `paused`를 보지 않고 **시각이 같은지**만 보는 이유: 멈추는 경우가 일시정지 하나가
    // 아니다 — 메인 창이 tick을 끊으면(내비게이션) sampleTime이 STATE_STALE_MS에 걸려
    // 같은 시각을 계속 돌려주는데, 그때도 같은 그림을 60~120Hz로 다시 칠할 이유가 없다.
    if (time === this.lastTime && at - this.lastPaintAt < IDLE_REDRAW_MS) return;
    this.lastPaintAt = at;
    this.lastTime = time;
    const duration = this.state.duration;
    if (this.progressEl && duration > 0) {
      this.progressEl.style.width = `${Math.min(100, (time / duration) * 100)}%`;
    }
    if (this.timeEl && duration > 0) {
      // 초 단위 문구는 프레임마다 같다 — 바뀔 때만 써서 불필요한 리플로우를 피한다
      const label = `${formatTime(time)} / ${formatTime(duration)}`;
      if (this.timeEl.textContent !== label) this.timeEl.textContent = label;
    }
    // 가사 채움·보컬 글로우·레인은 패널이 그린다 — content의 tick이 아니라 **이 창의
    // rAF**가 부르는 것이 핵심이다(state 주석: 숨은 탭에서 4Hz로 떨어지지 않게).
    this.panel?.updateTime(time, this.paused);
  }

  /** 원본 video의 볼륨 상태를 컨트롤에 반영 (tick과 함께 주기 호출) */
  updateVolume(volume: number, muted: boolean): void {
    if (!this.win) return;
    if (this.volumeSlider && !this.volumeDragging) {
      const v = String(Math.round(volume * 100));
      if (this.volumeSlider.value !== v) this.volumeSlider.value = v;
      this.volumeSlider.classList.toggle('muted', muted);
    }
    if (this.muteBtn && muted !== this.lastMuted) {
      this.lastMuted = muted;
      this.muteBtn.replaceChildren(icon(muted ? MUTED_SVG : VOLUME_SVG));
      this.muteBtn.title = muted ? t('pip.controls.unmute') : t('pip.controls.mute');
    }
  }

  /**
   * 열 사이 **세로** 디바이더 — 끌면 그 열의 폭이 바뀐다.
   *
   * 규약은 기존 가로 디바이더(영상 비율)와 같다: 드래그 중에는 화면만 즉시 따라가고,
   * **떼는 순간 한 번만** 설정에 저장한다(매 pointermove 저장 = 초당 수십 번 storage 쓰기).
   *
   * 방향이 반대인 두 경우를 한 함수로 다룬다 — 레인 열은 디바이더의 **왼쪽**에 있어
   * 오른쪽으로 끌수록 넓어지고, 가사창 열은 **오른쪽**에 있어 왼쪽으로 끌수록 넓어진다.
   */
  private buildColDivider(which: 'lane' | 'panel', onDone: (px: number) => void): HTMLDivElement {
    const divider = h('div', {
      className: 'ey-pip-vdivider',
      title: t('pip.controls.colWidthDrag'),
    }, h('div', { className: 'ey-pip-divider-grip' }));
    let dragging = false;
    let startX = 0;
    let startW = 0;
    divider.addEventListener('pointerdown', (e: PointerEvent) => {
      dragging = true;
      startX = e.clientX;
      startW = which === 'lane' ? this.laneColW : this.panelColW;
      divider.setPointerCapture(e.pointerId);
      e.preventDefault();
    });
    divider.addEventListener('pointermove', (e: PointerEvent) => {
      if (!dragging) return;
      const dx = e.clientX - startX;
      // 레인은 오른쪽으로 끌수록(+dx) 넓어지고, 가사창은 왼쪽으로 끌수록(-dx) 넓어진다
      // 스왑하면 레인이 중앙 열 오른쪽으로 가므로, 오른쪽으로 끌 때 레인이 «줄어든다».
      // 부호를 뒤집지 않으면 손이 가는 방향과 열이 자라는 방향이 반대가 된다.
      const laneSign = this.laneSwapped ? -1 : 1;
      const raw = which === 'lane' ? startW + dx * laneSign : startW - dx;
      const min = which === 'lane' ? LANE_COL_MIN : PANEL_COL_MIN;
      // 중앙 열이 CENTER_COL_MIN 아래로 짓눌리지 않는 선까지만 — 물리 상한 하나뿐이다
      const other = which === 'lane' ? this.panelColW : this.laneColW;
      const room = (this.win?.innerWidth ?? 0) - other - CENTER_COL_MIN - VDIVIDER_W * 2
        - (this.playlistVisible() ? PLAYLIST_COL_W : 0);
      const next = Math.round(Math.min(Math.max(raw, min), Math.max(min, room)));
      if (which === 'lane') this.laneColW = next;
      else this.panelColW = next;
      this.applyColumnLayout();
    });
    divider.addEventListener('pointerup', (e: PointerEvent) => {
      if (!dragging) return;
      dragging = false;
      divider.releasePointerCapture(e.pointerId);
      onDone(which === 'lane' ? this.laneColW : this.panelColW);
    });
    return divider;
  }

  /**
   * 창 크기를 잦아든 뒤 한 번만 저장한다 — 모서리를 끄는 동안 매 프레임 저장하면
   * chrome.storage 쓰기가 초당 수십 번 나간다(디바이더 드래그와 같은 규약).
   */
  private scheduleSizeSave(onSizeChange: (w: number, h: number) => void): void {
    const win = this.win;
    if (!win) return;
    clearTimeout(this.sizeSaveTimer);
    this.sizeSaveTimer = window.setTimeout(() => {
      const w = this.win?.innerWidth ?? 0;
      const h = this.win?.innerHeight ?? 0;
      if (w > 0 && h > 0) {
        const size = clampPipSize(w, h);
        onSizeChange(size.width, size.height);
      }
    }, 500);
  }

  /**
   * 「방금 자리가 없어 접혔다」를 사용자에게 알린다 — 전이(false→true)에서만 한 번.
   *
   * 매 resize마다 띄우면 창을 끄는 동안 칩이 도배된다. 새 UI는 만들지 않고 패널의
   * 알림 칩을 그대로 쓴다(운영자 지시: 기존 알림 칩 재사용).
   *
   * 가사창 열 자신이 접힌 경우에는 이 창에 칩을 띄울 자리가 없다 — 그때는 코너 버튼의
   * 점선 테두리와 툴팁(syncCornerButtons)이 사유를 말한다. 칩과 버튼이 서로를 보완한다.
   */
  private notifyIfNewlyCollapsed(prev: { lane: boolean; panel: boolean; playlist: boolean }): void {
    const now = this.autoCollapsed;
    const newly = (!prev.lane && now.lane) || (!prev.playlist && now.playlist)
      || (!prev.panel && now.panel);
    if (!newly) return;
    // 패널이 살아 있어야 칩을 볼 수 있다 — 접힌 경우는 코너 버튼이 대신 말한다
    if (!now.panel) this.panel?.notifyAutoCollapsed();
  }

  private playlistVisible(): boolean {
    return this.playlistColEl?.style.display !== 'none' && !this.autoCollapsed.playlist;
  }

  /**
   * 열 폭 적용 + **좁은 창 자동 접힘**.
   *
   * 운영자 우려("구석에 몰려 있는 개차반")의 실체는 «다 넣으려다 아무것도 못 읽게 되는 것»
   * 이다. 그래서 겹쳐 그리거나 짓누르지 않고, 폭이 모자라면 우선순위가 낮은 열부터
   * 통째로 접어 공간을 반납한다. 접힘은 **화면에만** 있고 설정은 건드리지 않으므로,
   * 창을 다시 넓히면 사용자가 정한 폭 그대로 되돌아온다(메인 패널의 LANE_TWO_COL_MIN
   * 폴백과 같은 규약).
   *
   * 우선순위(뒤부터 접힌다): 가사창 > 중앙 열(영상·컨트롤) > 레인 > 재생목록.
   * 가사창이 1순위인 이유는 단순하다 — 검색·설정·오프셋이 전부 거기 있어서, 그것이
   * 사라지면 이 창으로 할 수 있는 일이 없어진다.
   */
  private applyColumnLayout(): void {
    const win = this.win;
    if (!win || !this.rowEl) return;
    const avail = win.innerWidth;
    // **패널에 묻는다.** 예전에는 열의 style.display를 되읽었는데, 그 값을 쓰는 것도
    // 이 함수라 한 번 접힌 열이 영원히 «원하지 않음»으로 굳었다(실측으로 잡힌 래치).
    const laneWanted = this.panel?.laneVisible() ?? false;
    const playlistWanted = this.panel?.playlistVisible() ?? false;

    // 접힘 판정 — 우선순위가 낮은 열부터 떨어뜨린다.
    //
    // **최후까지 남는 것은 중앙 열(영상 + 지금 부르는 줄 + 재생 컨트롤)이다.** 좁은 PiP
    // 창의 고유 가치가 정확히 그것이고, 그 모습이 재작업 전 PiP와 같아서 좁아졌을 때
    // «기존처럼» 보인다(운영자 확인). 가사창은 전 기능 패널이지만 좁은 창에서는 어차피
    // 다 못 보여주고, 필요하면 유튜브 탭의 메인 패널을 열면 된다.
    //
    // 순서: 재생목록(최초) → 레인 → 가사창 → 중앙 열(항상 유지).
    let lane = laneWanted;
    let playlist = playlistWanted;
    let panel = this.panelWanted;
    // **사용자가 끈 중앙 열은 켜지 않는다.** 자동 접힘 우선순위(중앙 열 최후 생존)는
    // «폭이 모자랄 때 무엇부터 버릴까»의 규칙이지 사용자 선택의 제한이 아니다 —
    // 레인만 남기는 «가라오케 단독 모드»가 이 한 줄로 성립한다(운영자 요청).
    let center = this.centerWanted;
    const need = () => (center ? CENTER_COL_MIN : 0)
      + (panel ? PANEL_COL_MIN + VDIVIDER_W : 0)
      + (lane ? this.laneColW + VDIVIDER_W : 0)
      + (playlist ? PLAYLIST_COL_W : 0);
    if (need() > avail && playlist) playlist = false;
    if (need() > avail && lane) lane = false;
    if (need() > avail && panel) panel = false;
    // 마지막 안전망: 무엇 하나는 반드시 남아야 한다(빈 창은 «고장»으로 읽힌다).
    // 사용자가 전부 껐다면 중앙 열을 되살린다 — 그것이 PiP의 정체성이다.
    if (!center && !panel && !lane && !playlist) center = true;
    const prevCollapsed = this.autoCollapsed;
    this.autoCollapsed = {
      lane: laneWanted && !lane,
      panel: this.panelWanted && !panel,
      playlist: playlistWanted && !playlist,
    };
    void center;

    if (this.laneColEl) {
      this.laneColEl.style.flex = `0 0 ${this.laneColW}px`;
      this.laneColEl.style.display = lane ? '' : 'none';
    }
    // 레인 열 ⇄ 중앙 열 맞바꾸기 — DOM 순서를 바꾼다(운영자 용례: 스왑하면 레인이
    // 가사창 바로 옆에 와서 따라 부르며 가사도 같이 보기 쉽다). 디바이더는 언제나
    // 두 열 «사이»에 있으므로 함께 옮긴다.
    if (this.rowEl && this.laneColEl && this.centerColEl && this.laneDividerEl) {
      const first = this.laneSwapped ? this.centerColEl : this.laneColEl;
      const second = this.laneSwapped ? this.laneColEl : this.centerColEl;
      if (this.rowEl.firstElementChild !== first) {
        this.rowEl.prepend(first, this.laneDividerEl, second);
      }
    }
    // 중앙 열 — 남는 폭을 가져가되, 사용자가 접었으면 자리를 통째로 반납한다
    if (this.centerColEl) {
      this.centerColEl.style.display = center ? '' : 'none';
      this.centerColEl.style.flex = '1 1 auto';
      this.centerColEl.style.minWidth = `${CENTER_COL_MIN}px`;
    }
    if (this.laneDividerEl) {
      // 레인·중앙 사이 디바이더는 둘 다 있을 때만 뜻이 있다
      this.laneDividerEl.style.display = lane && center ? '' : 'none';
    }
    // 중앙 열이 없으면 레인이 남는 폭을 가져간다(가라오케 단독 모드)
    if (this.laneColEl && lane) {
      this.laneColEl.style.flex = center ? `0 0 ${this.laneColW}px` : '1 1 auto';
    }
    if (this.playlistColEl) {
      this.playlistColEl.style.flex = `0 0 ${PLAYLIST_COL_W}px`;
      this.playlistColEl.style.display = playlist ? '' : 'none';
    }
    const host = this.panel ? win.document.getElementById('everyric-root') : null;
    if (host) {
      host.style.flex = `0 0 ${this.panelColW}px`;
      host.style.minWidth = `${PANEL_COL_MIN}px`;
      host.style.display = panel ? '' : 'none';
    }
    if (this.panelDividerEl) this.panelDividerEl.style.display = panel ? '' : 'none';
    this.syncCornerButtons();
    this.notifyIfNewlyCollapsed(prevCollapsed);
  }

  private buildDivider(win: Window, onRatioChange: (ratio: number) => void): HTMLDivElement {
    const divider = h('div', { className: 'ey-pip-divider', title: t('pip.controls.videoRatioDrag') },
      h('div', { className: 'ey-pip-divider-grip' }));
    let dragging = false;
    divider.addEventListener('pointerdown', (e: PointerEvent) => {
      dragging = true;
      divider.setPointerCapture(e.pointerId);
      e.preventDefault();
    });
    divider.addEventListener('pointermove', (e: PointerEvent) => {
      if (!dragging || win.innerHeight === 0) return;
      const ratio = Math.min(MAX_VIDEO_RATIO, Math.max(MIN_VIDEO_RATIO, e.clientY / win.innerHeight));
      this.applyVideoRatio(ratio);
    });
    divider.addEventListener('pointerup', (e: PointerEvent) => {
      if (!dragging) return;
      dragging = false;
      divider.releasePointerCapture(e.pointerId);
      onRatioChange(this.videoRatio);
    });
    return divider;
  }

  private applyVideoRatio(ratio: number): void {
    this.videoRatio = ratio;
    this.syncVideoLayout();
  }

  /**
   * 영상 영역의 세로 비율 적용 — 사용자가 정한 비율이 있으면 그만큼 차지하고, 없으면
   * CSS 기본(16:9, 남는 세로는 패널이 가져간다)으로 되돌린다.
   */
  private syncVideoLayout(): void {
    if (!this.videoWrapEl) return;
    const videoShown = this.videoWrapEl.style.display !== 'none';
    if (this.dividerEl) this.dividerEl.style.display = videoShown ? '' : 'none';
    if (!videoShown) return;
    if (this.videoRatio > 0) {
      this.videoWrapEl.style.aspectRatio = 'auto';
      this.videoWrapEl.style.maxHeight = 'none';
      this.videoWrapEl.style.flex = `0 1 ${(this.videoRatio * 100).toFixed(1)}%`;
    } else {
      this.videoWrapEl.style.flex = '';
      this.videoWrapEl.style.aspectRatio = '';
      this.videoWrapEl.style.maxHeight = '';
    }
  }

  private hideVideoArea(): void {
    if (this.videoWrapEl) {
      this.videoWrapEl.style.display = 'none';
      this.videoWrapEl.replaceChildren();
    }
    if (this.dividerEl) this.dividerEl.style.display = 'none';
  }

  private stopMirror(): void {
    this.mirrorStream?.getTracks().forEach(track => track.stop());
    this.mirrorStream = null;
  }
}

function formatTime(sec: number): string {
  const total = Math.max(0, Math.floor(sec));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${s.toString().padStart(2, '0')}`;
}
