import type { LyricLine, SearchCandidate, ServerLogEntry, ServerStatus, SongInfo, SongKey, SongTempo, SyncDebugMeta } from '../types';
import type { MicSample } from '../lib/mic-pitch';
import { resolvedPronSegments, resolvedPronunciation, type PronScript } from '../lib/lang';
import { t } from '../lib/i18n';
import { unknownStatus } from '../lib/server-status';
import type { ThemeName } from '../lib/theme';
import { h, icon } from './dom';
import { appendKaraokeSpans, appendTimedSpans } from './karaoke';
import { PitchLaneRenderer } from './pitch-lane';
import {
  applyServerGate,
  buildEmptyState,
  buildErrorState,
  buildLoadingState,
  buildPlainLines,
  buildSearchSheet,
  buildServerStatusSlot,
  createGenerateButton,
  renderCandidateList,
  type PanelCallbacks,
  type PanelContext,
} from './panels';
import { getPlaylist, hasNext, hasPrevious, playNext, playPlaylistItem, playPrevious } from '../lib/yt-player';

interface DocumentPictureInPictureApi {
  requestWindow(options?: { width?: number; height?: number }): Promise<Window>;
}

export interface PipOptions {
  /** 열 때 영상 미러 영역을 포함한 크기로 열지 여부 */
  showVideo: boolean;
  /** 창 너비(px) — 호출부가 저장값/기존 기본값을 미리 계산해서 넘긴다 */
  width: number;
  /** 창 높이(px) — 호출부가 저장값/기존 기본값을 미리 계산해서 넘긴다 */
  height: number;
  /** 창 크기가 바뀐 채로 닫힐 때(pagehide) 마지막 크기를 알려준다 — 설정에 저장용 */
  onSizeChange: (width: number, height: number) => void;
  /** 영상 영역 세로 비율 (0 = 자동 16:9) */
  initialVideoRatio: number;
  /** 현재 라인 밑에 한국어 발음 표기 표시 여부 */
  showPronunciation: boolean;
  /** 발음 표기 방식(자동 해석 완료 — hangul/romaji/kana). 호출부(content.ts)가
   *  lib/lang.ts의 resolveScript(settings)로 미리 해석해 넘긴다 */
  pronScript: PronScript;
  /** 가라오케 음정 바 표시 여부 (노트 데이터가 있는 곡에서만 실제 표시) */
  pitchEnabled: boolean;
  /** 가라오케 레인 높이(px) — 디바이더 드래그로 조절, 설정에 저장 */
  pitchLaneHeight: number;
  /** 레인 표시 구간(마디 수) — 서버 BPM 기준, 템포 없으면 120BPM 가정 폴백 */
  pitchWindowMeasures: number;
  /** 레인 진행 방식: page = 화면 고정 + 플레이헤드 이동, scroll = 플레이헤드 고정 + 횡스크롤 */
  pitchScrollMode: 'page' | 'scroll';
  /** 레인 글자 크기 배율 */
  pitchFontScale: number;
  /** 긴 묵음 뒤 가사 시작 전 4·3·2·1 카운트다운 */
  pitchCountdown: boolean;
  /** 계이름 표기: korean(도레미)·english(C4·D#5 등, 옥타브 포함) */
  solfegeNotation: 'korean' | 'english';
  /** 음정선(노트 바) 밝기 배율 0.2~1.0 — 기존 알파값에 곱해진다 */
  pitchLineOpacity: number;
  /** f0 곡선(음정 보는 선) 밝기 배율 0.2~1.5 — 노트 바와 별개 페이더 (운영자 요청) */
  pitchF0Opacity: number;
  /** 크로마키 스트리밍 모드 — 'off'가 아니면 PIP 문서 배경을 단색 키 컬러로 (OBS 키잉용) */
  pipChromaKey: 'off' | 'green' | 'blue' | 'magenta';
  /** 디버그: 글자별 CTC 신뢰도를 색으로 표시 */
  showConfidence: boolean;
  /** 발음 표기 위치: note = 노트마다 위에 부착, bottom = 화면 하단 중앙(진행률 그라데이션) */
  pitchPronPosition: 'note' | 'bottom' | 'both';
  /** 레인 높이 드래그 조절 완료 시 */
  onPitchHeightChange: (px: number) => void;
  /** 가사 라인 클릭 — 가사 타임라인(초) 기준 */
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
  /** 디바이더 드래그로 영상 비율 변경 완료 시 */
  onVideoRatioChange: (ratio: number) => void;
  /** 멜로디 재생 초기 상태 + 토글 (footer 버튼) */
  melodyOn: boolean;
  onMelodyToggle: () => void;
  /** 메트로놈 초기 상태 + 토글 (footer 버튼) */
  metronomeOn: boolean;
  onMetronomeToggle: () => void;
  /** 메트로놈 배속 (0.5|1|2) — footer 버튼으로 순환 */
  metronomeRate: number;
  onMetronomeRateChange: (rate: number) => void;
  /** 마디 시작 박(0~3) — 메트로놈 강세와 레인 마디선이 함께 이동 */
  metronomeBeat: number;
  onMetronomeBeatChange: (beat: number) => void;
  /** 마이크 음정 옥타브 보정 (옥타브 단위, -2~+2) */
  micOctave: number;
  /** 레인 표시 구간(마디) 변경 — 창 안 ± 버튼 */
  onPitchWindowChange: (measures: number) => void;
  /** 레인 진행 방식 변경 — 창 안 토글 버튼 */
  onPitchScrollModeChange: (mode: 'page' | 'scroll') => void;
  /** 마이크 음정 표시가 켜져 있으면 최근 피치 샘플을 돌려준다 (없으면 null) */
  getMicSamples: (() => MicSample[]) | null;
  /** 좌상단 미니 버튼 — 가라오케 음정 바 기능 자체 토글 (설정 pitchGuide) */
  onKaraokeToggle: (on: boolean) => void;
  /** 좌상단 미니 버튼 — PiP 영상 표시 토글 (설정 pipShowVideo) */
  onVideoToggle: (on: boolean) => void;
  /** 가사 패널(가사 없음·검색·붙여넣기·싱크 생성) 콜백 — 메인 창과 같은 핸들러로 배선한다 */
  panel: PanelCallbacks;
  /** 열 때의 테마 — content가 lib/theme.resolveTheme로 판정한 값. 이후 갱신은 setTheme */
  theme: ThemeName;
  /** 열 때의 서버 상태(사유 포함) — 이후 갱신은 setServerStatus */
  serverStatus: ServerStatus;
  /** 디버그 모드 — 서버가 정상일 때도 요청 로그를 노출할지 */
  debug: boolean;
  /** 최근 서버 요청 로그 — 접이식 섹션을 펼칠 때만 호출된다 */
  loadServerLog: () => Promise<ServerLogEntry[]>;
  onClosed: () => void;
}

const PLAY_SVG = '<svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>';
const PAUSE_SVG = '<svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M6 5h4v14H6zM14 5h4v14h-4z"/></svg>';
const VOLUME_SVG = '<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3a4.5 4.5 0 0 0-2.5-4v8a4.5 4.5 0 0 0 2.5-4zM14 3.2v2.1a7 7 0 0 1 0 13.4v2.1a9 9 0 0 0 0-17.6z"/></svg>';
const MUTED_SVG = '<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M3 9v6h4l5 5V4L7 9H3zm13.6 3 2.7-2.7-1.4-1.4-2.7 2.7-2.7-2.7-1.4 1.4 2.7 2.7-2.7 2.7 1.4 1.4 2.7-2.7 2.7 2.7 1.4-1.4-2.7-2.7z"/></svg>';
const NOTE_SVG = '<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M12 3v10.55A4 4 0 1 0 14 17V7h4V3h-6z"/></svg>';
const LANE_SVG = '<svg viewBox="0 0 24 24" width="13" height="13" fill="currentColor"><path d="M3 5h18v2H3zm0 4h12v2H3zm0 4h18v2H3zm0 4h9v2H3z"/></svg>';
const SCREEN_SVG = '<svg viewBox="0 0 24 24" width="13" height="13" fill="currentColor"><path d="M3 4h18v13H3V4zm2 2v9h14V6H5zm3 13h8v2H8v-2z"/></svg>';
const METRO_SVG = '<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M9 2h6l4 19H5L9 2zm1.6 2L7.4 19h9.2l-1.2-6.1-2.7 2.7-1.4-1.4L15 10.4 13.9 4h-3.3z"/></svg>';
const PREV_SVG = '<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M6 6h2.5v12H6zm12 12-9-6 9-6z"/></svg>';
const NEXT_SVG = '<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M15.5 6H18v12h-2.5zM6 18l9-6-9-6z"/></svg>';
const LIST_SVG = '<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M3 6h11v2H3zm0 5h11v2H3zm0 5h8v2H3zm16-11v8.1a2.8 2.8 0 1 0 2 2.7V7h2V5h-4z"/></svg>';
const FIND_SVG = '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="11" cy="11" r="7"/><line x1="21" y1="21" x2="16.2" y2="16.2"/></svg>';

const MIN_VIDEO_RATIO = 0.15;
const MAX_VIDEO_RATIO = 0.75;

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
 * 200ms의 근거: 숨은 탭 timeupdate 간격(실측 3.7/s ≈ 270ms)보다 작아 rAF가 죽었을 때 매
 * tick 발동하고, 살아 있을 때의 프레임 간격(~8~16ms, 30fps로 떨어져도 33ms)보다 한참 커서
 * 정상 동작 중 오발동이 없다.
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
 * 렌더다. 완전히 건너뛰려면 남은 setter 스무 개마다 dirty 표시를 달아야 하고, 하나만
 * 빠뜨리면 레인이 낡은 채로 영구히 멈춘다 — 눈에 잘 안 띄는 대신 치명적인 실패다.
 * 100ms면 설정을 바꾼 사람 눈에는 즉시이고, 재그리기 비용은 6분의 1 이하로 준다.
 */
const IDLE_REDRAW_MS = 100;

// PiP 창 크기 기억 — 비정상적으로 작거나 큰 값이 저장/전달돼 창이 못 쓰게 되지 않도록 클램프
const MIN_PIP_WIDTH = 280;
const MAX_PIP_WIDTH = 960;
const MIN_PIP_HEIGHT = 200;
const MAX_PIP_HEIGHT = 1000;
function clampPipSize(width: number, height: number): { width: number; height: number } {
  return {
    width: Math.round(Math.min(MAX_PIP_WIDTH, Math.max(MIN_PIP_WIDTH, width))),
    height: Math.round(Math.min(MAX_PIP_HEIGHT, Math.max(MIN_PIP_HEIGHT, height))),
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
 * - manifest 주입 CSS는 PiP 문서에 적용되지 않으므로 CSS 텍스트를 직접 주입한다.
 * - 브라우저는 PiP 창을 하나만 허용하므로, 유튜브 네이티브 PiP 대신
 *   video.captureStream() 미러로 영상을 함께 표시한다 (원본 재생/오디오는 탭에 유지).
 */
export class PipController {
  private win: Window | null = null;
  private lines: LyricLine[] = [];
  private index = -1;
  private videoWrapEl: HTMLDivElement | null = null;
  private dividerEl: HTMLDivElement | null = null;
  private mirrorStream: MediaStream | null = null;
  private videoRatio = 0;
  private prevEl: HTMLDivElement | null = null;
  private currentEl: HTMLDivElement | null = null;
  private pronEl: HTMLDivElement | null = null;
  private trEl: HTMLDivElement | null = null;
  private nextEl: HTMLDivElement | null = null;
  private titleEl: HTMLDivElement | null = null;
  /** 다음 영상 정보 모듈 (설정 modNextUp) — content가 setNextUp으로 채운다 */
  private nextUpEl: HTMLDivElement | null = null;
  private footerEl: HTMLDivElement | null = null;
  private progressEl: HTMLDivElement | null = null;
  private playBtn: HTMLButtonElement | null = null;
  private muteBtn: HTMLButtonElement | null = null;
  private melodyBtn: HTMLButtonElement | null = null;
  private metroBtn: HTMLButtonElement | null = null;
  private volumeSlider: HTMLInputElement | null = null;
  private volumeDragging = false;
  private timeEl: HTMLSpanElement | null = null;
  private lastPaused: boolean | null = null;
  private lastMuted: boolean | null = null;
  private wordEls: { start: number; el: HTMLElement }[] = [];
  private onSeek: (time: number) => void = () => {};
  private pitchCanvas: HTMLCanvasElement | null = null;
  /**
   * 레인 렌더러 — 오선·노트·가사/발음/번역·f0를 그리는 몸통은 전부 여기에 있다(pitch-lane.ts).
   * 메인 가사창의 [모듈] 레인과 **같은 클래스**라 두 창이 절대 갈라지지 않는다.
   * 이 클래스는 캔버스 만들기·크기 조절·포인터 상호작용·설정 반영만 담당한다.
   */
  private lane = new PitchLaneRenderer();
  private pitchPointer: { id: number; startX: number; startT0: number; moved: boolean } | null = null;
  private paused = false;
  private lastTime = 0;
  /**
   * 메인 창이 밀어넣는 최신 재생 상태 — PiP는 이것을 **보간해 자기 rAF로** 그린다.
   *
   * 왜: 메인 창의 rAF(sync-engine)는 탭이 숨으면 멈추고 timeupdate(~4Hz)만 남는다.
   * PiP 렌더가 그 tick에 실려 있으면, **탭을 떠나 보는 것이 존재 이유인 이 창이** 초당
   * 4프레임으로 끊긴다(레인·가사 채움 전부). PiP는 별도 최상위 창이라 자기 rAF는 계속
   * 돌므로, 메인 창은 상태(시각·재생 여부·배속)만 공급하고 그리기는 이 창이 한다.
   * 그리는 곳이 rAF 한 곳뿐이라 메인 창이 보일 때도 같은 프레임을 두 번 그리지 않는다.
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
  private pitchDividerEl: HTMLDivElement | null = null;
  private pitchEnabled = true;
  private pitchLaneHeight = 170;
  private pitchWindowMeasures = 4;
  private pitchScrollMode: 'page' | 'scroll' = 'page';
  private pitchFontScale = 1.2;
  /** K1: 레인 위 마우스 휠 줌이 ±버튼과 같은 값을 저장하려면 그 콜백을 들고 있어야 한다
   *  (attachPitchPointer는 별도 메서드라 open()의 opts 클로저에 안 잡힌다 — onSeek와 같은
   *  패턴). open()에서 opts.onPitchWindowChange로 채워진다. */
  private onPitchWindowChange: (measures: number) => void = () => {};
  /** 크로마키 스트리밍 모드 — applyChroma가 문서 루트 클래스·CSS 변수로 반영 */
  private chromaKey: 'off' | 'green' | 'blue' | 'magenta' = 'off';
  /** 발음 표기 방식 — 스테이지 발음 줄(renderLines)이 쓴다. 레인 쪽 반영은 lane이 맡는다 */
  private pronScript: PronScript = 'hangul';
  private metronomeRate = 1;
  private metronomeBeat = 0;
  private metroRateBtn: HTMLButtonElement | null = null;
  private metroBeatBtn: HTMLButtonElement | null = null;
  private windowLabelBtn: HTMLButtonElement | null = null;
  private windowMinusBtn: HTMLButtonElement | null = null;
  private windowPlusBtn: HTMLButtonElement | null = null;
  private modeBtn: HTMLButtonElement | null = null;
  /** 레인(가라오케 모드)이 실제로 표시 중인지 — 가라오케 관련 버튼 노출 게이트 */
  private laneShown = false;
  private metroOn = false;
  /** 원본 video의 재생 배속 — 숨은 탭에서 상태 사이를 보간할 때(sampleTime) 쓴다.
   *  레인의 마이크 궤적 보정에도 같은 값이 필요해 lane에도 함께 밀어넣는다 */
  private playbackRate = 1;
  /** 좌상단 미니 버튼 — 가라오케 기능·PiP 영상 표시 토글 */
  private cornerKaraokeBtn: HTMLButtonElement | null = null;
  private cornerVideoBtn: HTMLButtonElement | null = null;
  private cornerPanelBtn: HTMLButtonElement | null = null;
  /** PiP 영상 표시 '설정' 상태 — 미러 성공 여부(DRM 실패 등)와 무관한 사용자 의도 */
  private videoOn = false;

  // ── 가사 패널 모드 (가사가 없어도 창이 살아남게) ─────────────────
  /** 패널 조각(panels.ts)이 그려지는 컨테이너 */
  private panelEl: HTMLDivElement | null = null;
  /** 패널이 지금 화면을 점유 중인가 (가사뷰/피치뷰와 배타) */
  private panelActive = false;
  /** 보여줄 패널 내용이 있는가 — 없으면 토글 버튼을 숨긴다 */
  private panelAvailable = false;
  private panelCallbacks: PanelCallbacks | null = null;
  private panelResultsEl: HTMLDivElement | null = null;
  /** 검색 시트를 열기 전 곡 정보 프리필용 */
  private lastSong: { title: string; artist: string } | null = null;
  private serverStatus: ServerStatus = unknownStatus();
  /** 현재 창에 칠해진 테마 — 판정은 content(lib/theme)가 하고 여기선 받아 쓰기만 한다 */
  private theme: ThemeName = 'dark';
  private debug = false;
  private loadServerLog: () => Promise<ServerLogEntry[]> = () => Promise.resolve([]);
  private generateButtons: HTMLButtonElement[] = [];
  /** 창 안 서버 오류 배너 — 패널이 닫혀 있어도 보이도록 body 흐름에 둔다 */
  private serverBarEl: HTMLDivElement | null = null;
  /** 전사 진행 칩 — 메인 패널의 .ey-gen-chip과 같은 정보를 창 안에 작게 */
  private chipEl: HTMLDivElement | null = null;
  /** 한 줄 알림 칩 — 메인 패널의 .ey-notice-chip과 같은 소식을 창 안에도.
   *  pipKeepPanel=false로 PiP만 보고 있으면 메인 패널의 칩은 볼 기회가 아예 없다 */
  private noticeEl: HTMLDivElement | null = null;
  private noticeTimer = 0;

  // ── 재생 컨트롤 (유튜브 DOM 조작 — lib/yt-player.ts) ─────────────
  private prevBtn: HTMLButtonElement | null = null;
  private nextBtn: HTMLButtonElement | null = null;
  private playlistBtn: HTMLButtonElement | null = null;
  private playlistEl: HTMLDivElement | null = null;
  private playlistOpen = false;

  static isSupported(): boolean {
    return 'documentPictureInPicture' in window;
  }

  isOpen(): boolean {
    return this.win !== null;
  }

  async open(cssText: string, opts: PipOptions): Promise<boolean> {
    if (this.win) return true;
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
    this.win = win;
    this.onSeek = opts.onSeek;
    this.panelCallbacks = opts.panel;
    this.serverStatus = opts.serverStatus;
    this.theme = opts.theme;
    this.debug = opts.debug;
    this.loadServerLog = opts.loadServerLog;
    this.generateButtons = [];
    this.videoRatio = opts.initialVideoRatio;
    this.pitchEnabled = opts.pitchEnabled;
    this.pitchLaneHeight = opts.pitchLaneHeight;
    this.pitchWindowMeasures = opts.pitchWindowMeasures;
    this.pitchScrollMode = opts.pitchScrollMode;
    this.pitchFontScale = opts.pitchFontScale;
    this.onPitchWindowChange = opts.onPitchWindowChange;
    this.chromaKey = opts.pipChromaKey;
    this.pronScript = opts.pronScript;
    this.metronomeRate = opts.metronomeRate;
    this.metronomeBeat = opts.metronomeBeat;
    // 레인 표시 취향은 렌더러가 들고 있다 — 창을 열 때 한 번에 밀어넣고, 이후에는
    // 개별 setter가 같은 경로(lane.setOptions)로 갱신한다
    this.lane.setOptions({
      enabled: opts.pitchEnabled,
      windowMeasures: opts.pitchWindowMeasures,
      scrollMode: opts.pitchScrollMode,
      fontScale: opts.pitchFontScale,
      countdown: opts.pitchCountdown,
      solfege: opts.solfegeNotation,
      lineOpacity: opts.pitchLineOpacity,
      f0Opacity: opts.pitchF0Opacity,
      pronPosition: opts.pitchPronPosition,
      pronScript: opts.pronScript,
      showConfidence: opts.showConfidence,
      metronomeBeat: opts.metronomeBeat,
      micOctave: opts.micOctave,
      getMicSamples: opts.getMicSamples,
    });

    const doc = win.document;
    doc.title = t('pip.docTitle');
    const style = doc.createElement('style');
    style.textContent = cssText;
    doc.head.append(style);
    doc.body.className = 'ey-pip';
    doc.body.classList.toggle('ey-hide-pron', !opts.showPronunciation);
    // 라이트 테마는 :root에 걸어야 한다 — 레인 캔버스 색을 readPitchColors가
    // documentElement의 계산된 CSS 변수에서 읽기 때문이다 (body에만 걸면 캔버스가 다크로 남는다)
    this.applyTheme();
    this.applyChroma();
    // 글자 크기 배율 — 레인(캔버스)뿐 아니라 스테이지 가사/발음/번역(CSS)에도 적용
    doc.body.style.setProperty('--ey-pip-fs', String(this.pitchFontScale));

    // 영상 미러 영역 + 비율 조절 디바이더 (attachVideo 전까지 숨김)
    this.videoWrapEl = h('div', { className: 'ey-pip-video' });
    this.videoWrapEl.style.display = 'none';
    this.dividerEl = this.buildDivider(win, opts.onVideoRatioChange);
    this.dividerEl.style.display = 'none';

    // dir=auto — 아랍어·히브리어 등 RTL 가사가 좌측 정렬로 어색하게 붙는 것 방지
    this.prevEl = h('div', { className: 'ey-pip-line prev', attrs: { dir: 'auto' }, on: { click: () => this.seekRelative(-1) } });
    this.currentEl = h('div', { className: 'ey-pip-line current', attrs: { dir: 'auto' }, on: { click: () => this.seekRelative(0) } });
    this.pronEl = h('div', { className: 'ey-pip-pron', attrs: { dir: 'auto' } });
    this.trEl = h('div', { className: 'ey-pip-tr', attrs: { dir: 'auto' } });
    this.nextEl = h('div', { className: 'ey-pip-line next', attrs: { dir: 'auto' }, on: { click: () => this.seekRelative(1) } });
    this.titleEl = h('div', { className: 'ey-pip-title' });

    this.playBtn = h('button', {
      className: 'ey-pip-play',
      title: t('pip.controls.playPause'),
      on: { click: () => opts.onPlayPause() },
    }, icon(PLAY_SVG));

    // 이전/다음 곡 + 재생목록 — 유튜브 비공식 마크업 의존(lib/yt-player.ts).
    // 셀렉터가 안 맞으면 클릭이 false를 돌려주므로 그 자리에서 비활성으로 내린다.
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
    this.playlistBtn = h('button', {
      className: 'ey-pip-play ey-pip-mute',
      title: t('pip.controls.playlist'),
      on: {
        click: () => {
          this.playlistOpen = !this.playlistOpen;
          this.renderPlaylist();
        },
      },
    }, icon(LIST_SVG));
    this.playlistEl = h('div', { className: 'ey-pip-playlist' });
    this.playlistEl.style.display = 'none';

    this.muteBtn = h('button', {
      className: 'ey-pip-play ey-pip-mute',
      title: t('pip.controls.mute'),
      on: { click: () => opts.onMuteToggle() },
    }, icon(VOLUME_SVG));

    this.melodyBtn = h('button', {
      className: 'ey-pip-play ey-pip-mute',
      title: t('pip.controls.melody'),
      on: { click: () => opts.onMelodyToggle() },
    }, icon(NOTE_SVG));
    this.metroBtn = h('button', {
      className: 'ey-pip-play ey-pip-mute',
      title: t('pip.controls.metronome'),
      on: { click: () => opts.onMetronomeToggle() },
    }, icon(METRO_SVG));
    // 메트로놈 세부 조절 — 배속(½×/1×/2×)과 마디 시작 박(1~4박)을 창 안에서 바로 순환
    this.metroRateBtn = h('button', {
      className: 'ey-pip-play ey-pip-mute ey-pip-metro-opt',
      title: t('pip.controls.metronomeRate'),
      on: {
        click: () => {
          const next = this.metronomeRate === 1 ? 2 : this.metronomeRate === 2 ? 0.5 : 1;
          this.setMetronomeConfig(next, this.metronomeBeat);
          opts.onMetronomeRateChange(next);
        },
      },
    });
    this.metroBeatBtn = h('button', {
      className: 'ey-pip-play ey-pip-mute ey-pip-metro-opt',
      title: t('pip.controls.metronomeBeat'),
      on: {
        click: () => {
          const next = (this.metronomeBeat + 1) % 4;
          this.setMetronomeConfig(this.metronomeRate, next);
          opts.onMetronomeBeatChange(next);
        },
      },
    });
    // 레인 표시 구간(마디) ±와 진행 방식 토글 — 설정 시트까지 안 가도 창 안에서 조절
    this.windowMinusBtn = h('button', {
      className: 'ey-pip-play ey-pip-metro-opt',
      title: t('pip.controls.windowMinus'),
      on: {
        click: () => {
          const next = Math.max(0.5, this.pitchWindowMeasures / 2);
          this.setPitchWindow(next);
          opts.onPitchWindowChange(next);
        },
      },
    }, '−');
    this.windowLabelBtn = h('button', {
      className: 'ey-pip-play ey-pip-metro-opt ey-pip-window-label',
      title: t('pip.controls.windowLabel'),
    });
    this.windowLabelBtn.disabled = true;
    this.windowPlusBtn = h('button', {
      className: 'ey-pip-play ey-pip-metro-opt',
      title: t('pip.controls.windowPlus'),
      on: {
        click: () => {
          const next = Math.min(16, this.pitchWindowMeasures * 2);
          this.setPitchWindow(next);
          opts.onPitchWindowChange(next);
        },
      },
    }, '+');
    this.modeBtn = h('button', {
      className: 'ey-pip-play ey-pip-metro-opt',
      title: t('pip.controls.modeToggle'),
      on: {
        click: () => {
          const next = this.pitchScrollMode === 'page' ? 'scroll' : 'page';
          this.setPitchScrollMode(next);
          opts.onPitchScrollModeChange(next);
        },
      },
    });
    this.updateWindowControls();
    this.setMetronomeConfig(this.metronomeRate, this.metronomeBeat);
    this.setAudioState(opts.melodyOn, opts.metronomeOn);

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

    // 가라오케 음정 레인 — 설정이 켜져 있고 노트 데이터가 있는 곡에서만 표시.
    // 위쪽 디바이더로 높이 조절 가능, 레인이 켜지면 스테이지(중복 표시)는 숨긴다.
    this.pitchCanvas = doc.createElement('canvas');
    this.pitchCanvas.className = 'ey-pip-pitch';
    this.pitchCanvas.style.height = `${this.pitchLaneHeight}px`;
    // 색은 PiP 문서 루트에서 읽는다 — 라이트 테마 변수(:root.ey-light)가 걸리는 자리다
    this.lane.attach(this.pitchCanvas, doc.documentElement);
    this.attachPitchPointer(this.pitchCanvas);
    this.pitchDividerEl = this.buildPitchDivider(opts.onPitchHeightChange);
    this.applyPitchVisibility();

    // 좌상단 미니 설정 — 가라오케 기능 자체·영상 표시를 창 안에서 바로 토글.
    // 레인이 창 상단에 붙을 때 캔버스 키 라벨은 x를 밀어 겹치지 않는다(renderPitch).
    this.videoOn = opts.showVideo;
    this.cornerKaraokeBtn = h('button', {
      className: 'ey-pip-mini',
      title: t('pip.controls.karaokeToggle'),
      on: { click: () => opts.onKaraokeToggle(!this.pitchEnabled) },
    }, icon(LANE_SVG));
    this.cornerVideoBtn = h('button', {
      className: 'ey-pip-mini',
      title: t('pip.controls.videoToggle'),
      on: { click: () => opts.onVideoToggle(!this.videoOn) },
    }, icon(SCREEN_SVG));
    // 가사뷰/피치뷰 ↔ 가사 패널(검색·붙여넣기) 전환 — 패널에 보여줄 내용이 있고
    // 되돌아갈 가사도 있을 때만 노출된다 (syncCornerButtons)
    this.cornerPanelBtn = h('button', {
      className: 'ey-pip-mini',
      title: t('pip.controls.panelToggle'),
      on: { click: () => this.togglePanel() },
    }, icon(FIND_SVG));
    this.syncCornerButtons();

    // 가사 패널 컨테이너 — 내용이 채워지기 전까지 숨김
    this.panelEl = h('div', { className: 'ey-pip-panel' });
    this.panelEl.style.display = 'none';
    // 전사 진행 칩 — 창을 점유하지 않고 진행률만
    this.chipEl = h('div', { className: 'ey-pip-chip' });
    this.chipEl.style.display = 'none';
    // 알림 칩 — 진행 칩과 같은 자리·다른 색(메인 패널의 두 칩과 같은 규약).
    // 전사 진행과 알림은 동시에 일어날 수 있어 한 엘리먼트를 공유하면 서로를 지운다
    this.noticeEl = h('div', { className: 'ey-pip-chip ey-pip-notice' });
    this.noticeEl.style.display = 'none';

    // 서버 오류 배너 — 메인 패널과 같은 조각을 쓴다. 패널(가사 찾기)이 닫혀 있어도
    // 보이도록 body 흐름에 두어, 가라오케 화면에서도 사유 한 줄이 사라지지 않는다
    this.serverBarEl = h('div', { className: 'ey-server-bar-slot ey-pip-server-bar' });
    this.serverBarEl.style.display = 'none';

    this.nextUpEl = h('div', { className: 'ey-pip-nextup' });
    this.nextUpEl.style.display = 'none';

    this.footerEl = h('div', { className: 'ey-pip-footer' },
      this.titleEl,
      this.nextUpEl,
      h('div', { className: 'ey-pip-controls' },
        this.prevBtn, this.playBtn, this.nextBtn, this.playlistBtn,
        this.muteBtn, this.melodyBtn, this.metroBtn,
        this.metroRateBtn, this.metroBeatBtn,
        this.windowMinusBtn, this.windowLabelBtn, this.windowPlusBtn, this.modeBtn,
        this.volumeSlider, progressWrap, this.timeEl),
    );
    doc.body.append(
      this.videoWrapEl,
      this.dividerEl,
      h('div', { className: 'ey-pip-stage' }, this.prevEl, this.currentEl, this.pronEl, this.trEl, this.nextEl),
      this.panelEl,
      this.pitchDividerEl,
      this.pitchCanvas,
      this.serverBarEl,
      this.chipEl,
      this.noticeEl,
      this.playlistEl,
      this.footerEl,
      h('div', { className: 'ey-pip-corner' }, this.cornerKaraokeBtn, this.cornerVideoBtn, this.cornerPanelBtn),
    );
    this.refreshPlayerControls();
    this.renderServerBar(); // 서버가 이미 죽어 있는 채로 열렸다면 열자마자 사유를 보여 준다

    // 창 크기가 줄면 레인이 푸터를 밀어내지 않도록 즉시 재클램프
    win.addEventListener('resize', () => this.clampLaneHeight());

    // OS 테마가 바뀌면 캐시된 레인 색이 낡는다 — 다음 렌더에서 CSS 변수 재판독
    try {
      win.matchMedia('(prefers-color-scheme: dark)')
        .addEventListener('change', () => this.refreshColors());
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
      // 따라가고 이 조합은 그대로다(여기서는 manifest의 기본값을 하드코딩한다). 실제 단축키를
      // 읽어 오려면 chrome.commands.getAll()의 문자열을 파싱해 내려보내야 해서, 재지정이 드문
      // 개발자용 토글에는 과하다고 판단했다.
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
      this.lastFrameAt = 0;
      this.lastPaintAt = 0;
      this.state = { time: 0, duration: 0, paused: true, at: 0 };
      this.stopMirror();
      this.win = null;
      this.index = -1;
      this.lane.setIndex(-1);
      this.wordEls = [];
      this.lastPaused = null;
      this.lastMuted = null;
      this.volumeDragging = false;
      this.pitchCanvas = null;
      this.pitchDividerEl = null;
      this.lane.detach(); // 닫힌 창의 캔버스를 계속 들고 있으면 누수다 (곡 데이터는 남긴다)
      this.videoWrapEl = null;
      this.dividerEl = null;
      this.footerEl = null;
      this.melodyBtn = null;
      this.metroBtn = null;
      this.metroRateBtn = null;
      this.metroBeatBtn = null;
      this.windowLabelBtn = null;
      this.windowMinusBtn = null;
      this.windowPlusBtn = null;
      this.modeBtn = null;
      this.cornerKaraokeBtn = null;
      this.cornerVideoBtn = null;
      this.cornerPanelBtn = null;
      this.videoOn = false;
      this.laneShown = false;
      this.panelEl = null;
      this.panelActive = false;
      this.panelAvailable = false;
      this.panelCallbacks = null;
      this.panelResultsEl = null;
      this.generateButtons = [];
      this.chipEl = null;
      clearTimeout(this.noticeTimer);
      this.noticeEl = null;
      this.serverBarEl = null;
      this.prevBtn = null;
      this.nextBtn = null;
      this.playlistBtn = null;
      this.playlistEl = null;
      this.playlistOpen = false;
      this.playbackRate = 1;
      // 마이크 공급자는 창과 함께 끊는다 — 닫힌 창의 궤적을 계속 끌어오면 누수다
      this.lane.setOptions({ getMicSamples: null, playbackRate: 1 });
      opts.onClosed();
    });
    this.renderLines();
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
    try {
      const capturable = source as HTMLVideoElement & { captureStream?: () => MediaStream };
      const stream = capturable.captureStream?.();
      if (!stream || stream.getVideoTracks().length === 0) {
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

  setSong(title: string, artist: string): void {
    // 패널 검색 폼 프리필용으로도 보관 — 창이 열린 채 곡이 바뀌어도 따라간다
    this.lastSong = title || artist ? { title, artist } : null;
    if (this.titleEl) {
      this.titleEl.textContent = artist ? `${title} — ${artist}` : title;
      this.setNextUp(null); // 곡이 바뀌면 이전 곡 기준 "다음 영상"은 낡은 정보다
    }
  }

  /** 다음 영상 정보 모듈 — null이면 숨김 (설정 modNextUp이 꺼져 있으면 content가 null을 준다) */
  setNextUp(title: string | null): void {
    if (!this.nextUpEl) return;
    if (title) {
      this.nextUpEl.textContent = t('pip.nextUp', [title]);
      this.nextUpEl.style.display = '';
    } else {
      this.nextUpEl.style.display = 'none';
    }
  }

  setLines(lines: LyricLine[]): void {
    this.lines = lines;
    this.index = -1;
    this.lane.setLines(lines);
    this.applyPitchVisibility();
    this.syncCornerButtons(); // 가사 유무가 바뀌면 패널 토글 버튼 노출 조건도 바뀐다
    this.renderLines();
  }

  /** 레인 표시 구간(마디 수) 설정 즉시 반영 — 0.5마디까지 허용 */
  setPitchWindow(measures: number): void {
    this.pitchWindowMeasures = Math.min(16, Math.max(0.25, measures));
    this.lane.setOptions({ windowMeasures: this.pitchWindowMeasures });
    this.updateWindowControls();
  }

  /** 레인 진행 방식(페이지/스크롤) 즉시 반영 */
  setPitchScrollMode(mode: 'page' | 'scroll'): void {
    this.pitchScrollMode = mode;
    this.lane.setOptions({ scrollMode: mode });
    this.updateWindowControls();
  }

  /** K2: 계이름 표기(korean/english) 즉시 반영 — 다음 프레임까지 안 기다린다(일시정지 중
   *  설정을 바꿔도 바로 보이게, pointermove 팬과 같은 이유). */
  setSolfegeNotation(notation: 'korean' | 'english'): void {
    this.lane.setOptions({ solfege: notation });
    this.renderLane();
  }

  /** K3: 음정선(노트 바) 밝기 배율 즉시 반영 — [0.2, 1] 클램프. */
  setPitchLineOpacity(opacity: number): void {
    this.lane.setOptions({ lineOpacity: Math.min(1, Math.max(0.2, opacity)) });
    this.renderLane();
  }

  /** f0 곡선(음정 보는 선) 밝기 배율 즉시 반영 — [0.2, 1.5] 클램프(1 초과 = 기본보다 밝게). */
  setPitchF0Opacity(opacity: number): void {
    this.lane.setOptions({ f0Opacity: Math.min(1.5, Math.max(0.2, opacity)) });
    this.renderLane();
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

  /** 창 안 레인 조절 버튼(마디 ±·진행 방식)의 라벨을 현재 값에 맞춘다 */
  private updateWindowControls(): void {
    if (this.windowLabelBtn) {
      const m = this.pitchWindowMeasures;
      this.windowLabelBtn.textContent = m === 0.5
        ? t('overlay.settings.pitchWindow.half')
        : t('overlay.settings.pitchWindow.bars', [String(m)]);
    }
    if (this.modeBtn) {
      this.modeBtn.textContent = this.pitchScrollMode === 'page' ? t('pip.controls.modeFixed') : t('pip.controls.modeScroll');
    }
    if (this.windowMinusBtn) this.windowMinusBtn.disabled = this.pitchWindowMeasures <= 0.5;
    if (this.windowPlusBtn) this.windowPlusBtn.disabled = this.pitchWindowMeasures >= 16;
  }

  /** 가라오케 글자 크기 배율 즉시 반영 — 레인(캔버스) + 스테이지(CSS 변수) 공통 */
  setPitchFontScale(scale: number): void {
    this.pitchFontScale = Math.min(2, Math.max(0.6, scale));
    this.lane.setOptions({ fontScale: this.pitchFontScale });
    this.win?.document.body.style.setProperty('--ey-pip-fs', String(this.pitchFontScale));
  }

  /** 서버가 추정한 곡 키 — 레인 좌상단 표시용 */
  setKey(key: SongKey | null): void {
    this.lane.setKey(key);
  }

  /** 메트로놈 배속/시작 박 — footer 버튼 라벨과 레인 마디선 표시에 반영 */
  setMetronomeConfig(rate: number, beat: number): void {
    this.metronomeRate = rate;
    this.metronomeBeat = beat;
    this.lane.setOptions({ metronomeBeat: beat });
    if (this.metroRateBtn) {
      this.metroRateBtn.textContent = rate === 0.5 ? '½×' : `${rate}×`;
    }
    if (this.metroBeatBtn) this.metroBeatBtn.textContent = t('overlay.settings.metronomeBeat.n', [String(beat + 1)]);
  }

  /** 마이크 음정 옥타브 보정 즉시 반영 */
  setMicOctave(octave: number): void {
    this.lane.setOptions({ micOctave: octave });
  }

  /** 서버가 추정한 곡 템포 — 마디 창 폭·비트 격자에 사용, null이면 초 단위 폴백 */
  setTempo(tempo: SongTempo | null): void {
    this.lane.setTempo(tempo);
  }

  /** 곡 단위 정렬 진단 메타 — 디버그 모드에서 VAD/간주 스트립·RAW f0 곡선을 그린다 */
  setDebugMeta(meta: SyncDebugMeta | null): void {
    this.lane.setDebugMeta(meta);
  }

  /** 카운트다운 설정 즉시 반영 */
  setPitchCountdown(enabled: boolean): void {
    this.lane.setOptions({ countdown: enabled });
  }

  /** 발음 표기 위치(노트 위/화면 하단) 즉시 반영 — 다음 레인 렌더에서 바로 적용됨 */
  setPitchPronPosition(position: 'note' | 'bottom' | 'both'): void {
    this.lane.setOptions({ pronPosition: position });
  }

  /** 발음 표기 방식 변경 즉시 반영 — 음정 노트에 부착된 발음(collectPitchData)도 다시 계산한다
   *  (재계산은 lane.setOptions가 pronScript 변화를 보고 스스로 한다) */
  setPronScript(script: PronScript): void {
    if (this.pronScript === script) return;
    this.pronScript = script;
    this.lane.setOptions({ pronScript: script });
    this.renderLines();
  }

  /** 현재 라인 인덱스 변경 시 호출 */
  update(index: number): void {
    if (!this.win || index === this.index) return;
    this.index = index;
    this.lane.setIndex(index);
    this.renderLines();
  }

  /** 번역 등 라인 데이터가 바뀐 뒤 강제 재렌더 */
  refresh(): void {
    if (this.win) this.renderLines();
  }

  /**
   * 테마 변경 등으로 CSS 변수가 바뀌었을 때 — 캐시된 레인 색을 버리고 재판독.
   * 일시정지 중엔 tick이 오지 않아 캔버스가 낡은 색으로 남으므로 즉시 한 번 다시 그린다.
   */
  refreshColors(): void {
    this.lane.refreshColors();
    this.renderLane();
  }

  /** 레인을 마지막 시각으로 즉시 한 번 다시 그린다 — 일시정지 중엔 tick이 오지 않아
   *  설정 변경이 다음 프레임을 기다리면 반영이 멈춘 것처럼 보인다 */
  private renderLane(): void {
    if (this.win) this.lane.render(this.lastTime, this.paused);
  }

  /**
   * 테마 즉시 반영 — content가 lib/theme.resolveTheme로 판정한 값을 밀어넣는다.
   * PiP 창은 스스로 판정하지 않는다(유튜브 페이지 컨텍스트가 없어 반드시 어긋난다).
   */
  setTheme(theme: ThemeName): void {
    if (this.theme === theme) return;
    this.theme = theme;
    this.applyTheme();
    this.refreshColors(); // 레인 색은 CSS 변수 캐시라 테마와 함께 다시 읽어야 한다
  }

  private applyTheme(): void {
    // CSS 변수 재정의(:root.ey-light)를 문서 루트에 건다 — body가 아니라 루트여야
    // getComputedStyle(documentElement)로 색을 읽는 레인 캔버스까지 함께 밝아진다
    this.win?.document.documentElement.classList.toggle('ey-light', this.theme === 'light');
  }

  /** 원본 video 배속 — 마이크 궤적 시간축 보정용 (content가 tick마다 밀어넣는다) */
  setPlaybackRate(rate: number): void {
    if (!Number.isFinite(rate) || rate <= 0) return;
    this.playbackRate = rate;
    this.lane.setOptions({ playbackRate: rate });
  }

  /** 발음 표기 설정 토글 즉시 반영 */
  setShowPronunciation(visible: boolean): void {
    this.win?.document.body.classList.toggle('ey-hide-pron', !visible);
  }

  /** 가라오케 음정 바 설정 토글 즉시 반영 */
  setPitchEnabled(enabled: boolean): void {
    this.pitchEnabled = enabled;
    this.lane.setOptions({ enabled });
    this.applyPitchVisibility();
    this.syncCornerButtons();
  }

  /** 좌상단 미니 버튼의 on/off 시각 상태를 현재 설정에 맞춘다 */
  private syncCornerButtons(): void {
    this.cornerKaraokeBtn?.classList.toggle('on', this.pitchEnabled);
    this.cornerVideoBtn?.classList.toggle('on', this.videoOn);
    if (this.cornerPanelBtn) {
      // 되돌아갈 가사가 있을 때만 토글 의미가 있다 — 가사가 없으면 패널이 유일한 화면
      const toggleable = this.panelAvailable && this.lines.length > 0;
      this.cornerPanelBtn.style.display = toggleable ? '' : 'none';
      this.cornerPanelBtn.classList.toggle('on', this.panelActive);
    }
  }

  // ── 가사 패널 모드 ──────────────────────────────────────────────
  //
  // 예전에는 가사를 못 찾거나(=applyLyricsData(null)) 싱크가 없으면 content.ts가
  // pip.close()를 불러 창이 통째로 사라졌다. 재생목록을 돌리다 싱크 없는 곡 하나만
  // 만나도 PiP가 증발하는 원인이 정확히 그것이었다. 이제는 닫는 대신 이 패널을 띄운다 —
  // 창은 사용자가 직접 닫기 전까지 살아 있다.

  /** 패널 조각(panels.ts)에 넘길 호스트 컨텍스트 */
  private panelContext(): PanelContext {
    const cb = this.panelCallbacks;
    return {
      callbacks: {
        onGenerate: (lyrics, attribution) => cb?.onGenerate(lyrics, attribution),
        onRetrySearch: query => cb?.onRetrySearch(query),
        onCandidateSearch: query => cb?.onCandidateSearch(query),
        onPickCandidate: candidate => cb?.onPickCandidate(candidate),
        onOpenSearch: () => this.openPanelSearch(),
        onOpenSettings: () => cb?.onOpenSettings(),
        onRecheckServer: () => cb?.onRecheckServer(),
        onOpenPermissions: () => cb?.onOpenPermissions(),
      },
      makeGenerateButton: (label, onClick) => {
        const btn = createGenerateButton(label, this.serverStatus, onClick);
        this.generateButtons.push(btn);
        return btn;
      },
      server: this.serverStatus,
      debug: this.debug,
      loadServerLog: () => this.loadServerLog(),
    };
  }

  /** 서버 상태 — 창 안 생성 버튼과 오류 배너를 일괄 갱신 (메인 패널과 동일 규칙·동일 조각) */
  setServerStatus(status: ServerStatus): void {
    this.serverStatus = status;
    this.generateButtons = this.generateButtons.filter(btn => btn.isConnected);
    for (const btn of this.generateButtons) applyServerGate(btn, status);
    this.renderServerBar();
  }

  /** 디버그 토글 — 서버 요청 로그의 노출 조건이라 배너를 다시 그린다 */
  setDebug(debug: boolean): void {
    this.debug = debug;
    this.renderServerBar();
  }

  private renderServerBar(): void {
    const slot = this.serverBarEl;
    if (!slot) return;
    const bar = buildServerStatusSlot(this.panelContext());
    if (!bar) {
      slot.replaceChildren();
      slot.style.display = 'none';
      return;
    }
    slot.replaceChildren(bar);
    slot.style.display = '';
  }

  /** 패널 내용 교체 공통 경로 — 이전 조각의 참조를 확실히 끊고 새로 그린다 */
  private setPanelContent(node: Node): void {
    if (!this.panelEl) return;
    this.panelResultsEl = null;
    this.generateButtons = [];
    this.panelEl.replaceChildren(node);
    this.panelAvailable = true;
    this.panelActive = true;
    this.applyPanelVisibility();
  }

  /** 가사를 못 찾았을 때 — 재검색 폼 + 외부 검색 링크 + 붙여넣기 */
  showPanelEmpty(song: SongInfo | null): void {
    if (!this.win) return;
    if (song) this.setSong(song.title, song.artist ?? '');
    this.setPanelContent(buildEmptyState(this.panelContext(), song));
  }

  /** 가사 검색 중 */
  showPanelLoading(message = t('overlay.loading.default')): void {
    if (!this.win) return;
    this.setPanelContent(buildLoadingState(this.panelContext(), message));
  }

  /** 오류 — detail은 서버가 준 힌트 등 추가 사유 */
  showPanelError(message: string, detail?: string): void {
    if (!this.win) return;
    this.setPanelContent(buildErrorState(this.panelContext(), message, detail));
  }

  /** 타임싱크가 없는 가사 — 목록을 보여주면서 싱크 생성 버튼을 함께 준다 */
  showPanelPlain(lines: LyricLine[], plainText: string): void {
    if (!this.win) return;
    const ctx = this.panelContext();
    const plain = buildPlainLines(lines);
    this.setPanelContent(h('div', { className: 'ey-pip-panel-plain' },
      h('div', { className: 'ey-banner' },
        h('span', { className: 'ey-banner-text', text: t('overlay.plain.noTimesync') }),
        ctx.makeGenerateButton(t('overlay.plain.generateSync'), () => ctx.callbacks.onGenerate(plainText)),
      ),
      plain.el,
    ));
  }

  /** 상세 검색 시트 — 메인 패널의 '다른 영상 싱크 연결'·'싱크 초기화'는 제외한 축약판 */
  openPanelSearch(): void {
    if (!this.win) return;
    const ctx = this.panelContext();
    const sheet = buildSearchSheet(
      ctx,
      { title: this.lastSong?.title ?? '', artist: this.lastSong?.artist ?? '' },
      {
        onBack: () => {
          // 되돌아갈 가사가 있으면 가사뷰로, 없으면 빈 상태로
          if (this.lines.length > 0) this.clearPanel();
          else this.showPanelEmpty(this.lastSong ? { title: this.lastSong.title, artist: this.lastSong.artist || null, videoId: '', duration: 0 } : null);
        },
        extras: [
          h('div', { className: 'ey-divider' }),
          h('button', {
            className: 'ey-secondary-btn',
            text: t('overlay.search.backToAuto'),
            on: { click: () => ctx.callbacks.onRetrySearch() },
          }),
        ],
      },
    );
    this.setPanelContent(sheet.el);
    this.panelResultsEl = sheet.results;
    sheet.runSearch();
  }

  /** 패널을 접고 가사뷰(스테이지/레인)로 복귀 — 싱크 가사가 도착했을 때 */
  clearPanel(): void {
    if (!this.win) return;
    this.panelAvailable = false;
    this.panelActive = false;
    this.panelResultsEl = null;
    this.generateButtons = [];
    this.panelEl?.replaceChildren();
    this.applyPanelVisibility();
  }

  /** 전사 진행 칩 — null이면 숨김. 창이 닫혀 있으면(chipEl=null) 아무 일도 하지 않는다 */
  setGenerationChip(text: string | null): void {
    if (!this.chipEl) return;
    this.chipEl.textContent = text ?? '';
    this.chipEl.style.display = text ? '' : 'none';
  }

  /**
   * 한 줄 알림 칩 — null이면 숨김. 메인 패널 setNoticeChip과 같은 규약(마지막 호출이 이긴다).
   *
   * 왜 PiP에도 있어야 하나: pipKeepPanel=false면 메인 패널은 placeholder로 접혀 있어,
   * 거절 사유("자동 생성 자막은…")·완료 경고·표기 필터 결과가 전부 사용자가 보지 않는
   * 창으로만 갔다. PiP만 보며 '싱크 생성'을 누르면 완전한 무반응이었던 원인의 절반이다.
   */
  setNoticeChip(text: string | null, autoHideMs?: number): void {
    clearTimeout(this.noticeTimer);
    const el = this.noticeEl;
    if (!el) return; // 창이 닫혀 있으면 알릴 자리가 없다
    if (!text) {
      el.textContent = '';
      el.style.display = 'none';
      return;
    }
    el.textContent = text;
    el.title = text; // 창이 좁아 잘려도 전문을 볼 수 있게
    el.style.display = '';
    if (autoHideMs !== undefined) {
      this.noticeTimer = window.setTimeout(() => {
        // 그 사이 창이 닫혔거나 다른 칩으로 교체됐을 수 있다 — 지금 엘리먼트만 건드린다
        if (this.noticeEl !== el) return;
        el.textContent = '';
        el.style.display = 'none';
      }, autoHideMs);
    }
  }

  /**
   * 이 창을 오류 화면으로 덮으면 사용자가 잃는 것이 있는가 (overlay.hasPreservableContent와 같은 규약).
   *
   * showPanelError는 setPanelContent를 타서 패널 내용을 통째로 교체한다 — 창 안에서
   * 붙여넣던 가사나 열어 둔 검색 시트가 그대로 사라진다.
   */
  hasPreservableContent(): boolean {
    if (this.lines.length > 0) return true; // 스테이지·레인에 이 곡 가사가 떠 있다
    if (this.panelResultsEl) return true; // 검색 시트가 살아 있다
    const panel = this.panelEl;
    if (!panel) return false;
    // 타임싱크 없는 가사 목록(showPanelPlain)도 지금 읽고 있는 가사다 — 이때 this.lines는
    // 비어 있으므로(스테이지에 그릴 타이밍이 없다) 패널 안을 직접 확인해야 한다
    if (panel.querySelector('.ey-lines-plain')) return true;
    return [...panel.querySelectorAll<HTMLTextAreaElement>('textarea')]
      .some(t => t.value.trim().length > 0);
  }

  /** SEARCH_CANDIDATES 응답 — 패널의 검색 시트가 살아 있을 때만 반영 (stale 방지) */
  showSearchResults(candidates: SearchCandidate[]): void {
    if (!this.panelResultsEl) return;
    renderCandidateList(this.panelResultsEl, candidates, c => this.panelCallbacks?.onPickCandidate(c));
  }

  /** 가사뷰 ↔ 패널 전환 (좌상단 미니 버튼) */
  private togglePanel(): void {
    if (!this.panelAvailable) return;
    this.panelActive = !this.panelActive;
    this.applyPanelVisibility();
  }

  /** 패널/가사뷰 배타 표시 — 레인·스테이지는 패널이 떠 있는 동안 숨는다 */
  private applyPanelVisibility(): void {
    if (this.panelEl) this.panelEl.style.display = this.panelActive ? '' : 'none';
    this.win?.document.body.classList.toggle('ey-panel-mode', this.panelActive);
    this.applyPitchVisibility(); // 레인 게이트가 panelActive를 함께 본다
    this.syncCornerButtons();
  }

  /** 레인 표시 조건 = 설정 on + 노트 데이터 있음 + 패널이 화면을 점유하지 않음.
   *  레인이 켜지면 스테이지는 숨긴다(중복 표시). */
  private applyPitchVisibility(): void {
    const show = this.pitchEnabled && this.lane.hasNotes() && !this.panelActive;
    this.laneShown = show;
    if (this.pitchCanvas) this.pitchCanvas.style.display = show ? '' : 'none';
    if (this.pitchDividerEl) this.pitchDividerEl.style.display = show ? '' : 'none';
    this.win?.document.body.classList.toggle('ey-lane-active', show);
    this.syncKaraokeControls();
    this.syncVideoLayout();
    this.clampLaneHeight();
  }

  /** 가라오케 관련 버튼(멜로디·메트로놈·마디·진행 방식)은 레인이 실제로 보일 때만 노출 */
  private syncKaraokeControls(): void {
    const lane = this.laneShown;
    const display = (el: HTMLElement | null, show: boolean) => {
      if (el) el.style.display = show ? '' : 'none';
    };
    display(this.melodyBtn, lane);
    display(this.metroBtn, lane);
    display(this.metroRateBtn, lane && this.metroOn);
    display(this.metroBeatBtn, lane && this.metroOn);
    display(this.windowMinusBtn, lane);
    display(this.windowLabelBtn, lane);
    display(this.windowPlusBtn, lane);
    display(this.modeBtn, lane);
  }

  /**
   * 레인 모드에 따라 영상 flex·비율 디바이더 노출을 한곳에서 일관 유지.
   * 레인 모드에선 스테이지가 숨어 영상이 유일한 성장 요소여야 하는데, 비율 드래그가 남긴
   * 인라인 `flex: 0 1 X%`(grow=0)가 CSS(flex:1)를 덮어쓰면 성장 요소가 사라져
   * 창 크기 조절 시 빈 공간·찌그러짐이 생긴다 — 레인 모드에선 인라인을 걷어낸다.
   */
  private syncVideoLayout(): void {
    if (!this.videoWrapEl) return;
    const laneShown = this.win?.document.body.classList.contains('ey-lane-active') ?? false;
    const videoShown = this.videoWrapEl.style.display !== 'none';
    if (this.dividerEl && videoShown) this.dividerEl.style.display = laneShown ? 'none' : '';
    if (laneShown) {
      this.videoWrapEl.style.flex = '';
      this.videoWrapEl.style.aspectRatio = '';
      this.videoWrapEl.style.maxHeight = '';
    } else if (this.videoRatio > 0) {
      this.videoWrapEl.style.aspectRatio = 'auto';
      this.videoWrapEl.style.maxHeight = 'none';
      this.videoWrapEl.style.flex = `0 1 ${(this.videoRatio * 100).toFixed(1)}%`;
    }
    this.syncLaneSizing();
  }

  /**
   * 레인 높이 배분 — 영상이 함께 표시될 때만 드래그 높이(px)가 의미 있다.
   * 영상이 없으면 레인이 유일한 성장 요소라 창을 꽉 채우고(높이 조절로 아래가 비는
   * 버그 방지), 높이 디바이더도 숨긴다 (조절할 대상이 없음).
   */
  private syncLaneSizing(): void {
    if (!this.pitchCanvas || !this.laneShown) return;
    const videoShown = this.videoWrapEl ? this.videoWrapEl.style.display !== 'none' : false;
    if (videoShown) {
      this.pitchCanvas.style.flex = '';
      this.pitchCanvas.style.height = `${this.pitchLaneHeight}px`;
      this.pitchCanvas.style.minHeight = '';
      if (this.pitchDividerEl) this.pitchDividerEl.style.display = '';
    } else {
      this.pitchCanvas.style.flex = '1 1 0';
      this.pitchCanvas.style.height = 'auto';
      this.pitchCanvas.style.minHeight = '90px';
      if (this.pitchDividerEl) this.pitchDividerEl.style.display = 'none';
    }
  }

  /** 레인 높이 상한 — 푸터·영상 최소 영역만 남기고 창 전체까지 키울 수 있다 (고정 캡 없음) */
  private maxLaneHeight(): number {
    const win = this.win;
    if (!win || win.innerHeight === 0) return 320;
    const footerH = this.footerEl?.offsetHeight || 60;
    const videoShown = this.videoWrapEl ? this.videoWrapEl.style.display !== 'none' : false;
    const reserved = footerH + 8 + (videoShown ? 48 : 0);
    return Math.max(90, win.innerHeight - reserved);
  }

  /** 창이 줄어 현재 레인 높이가 상한을 넘으면 맞춰서 줄인다 (창 resize·레인 토글 시) */
  private clampLaneHeight(): void {
    if (!this.pitchCanvas || this.pitchCanvas.style.display === 'none') return;
    // 영상 미표시 = flex 채움 모드 — 고정 높이가 없으니 클램프 대상이 아니다
    if (!this.videoWrapEl || this.videoWrapEl.style.display === 'none') return;
    const max = this.maxLaneHeight();
    if (this.pitchCanvas.clientHeight > max) {
      this.pitchLaneHeight = max;
      this.pitchCanvas.style.height = `${max}px`;
    }
  }

  /** 디버그 신뢰도 색상 토글 즉시 반영 */
  setShowConfidence(enabled: boolean): void {
    this.lane.setOptions({ showConfidence: enabled });
  }

  /** RAW f0 곡선 상시 표시 토글 (설정 pitchF0Curve) */
  setShowF0(enabled: boolean): void {
    this.lane.setOptions({ showF0: enabled });
  }

  // ── 재생 컨트롤 (유튜브 DOM 조작) ───────────────────────────────
  //
  // 유튜브 공식 페이지 API가 없어 lib/yt-player.ts가 플레이어 버튼·재생목록 DOM을
  // 직접 만진다. 유튜브가 마크업을 개편하면 그쪽 셀렉터가 전부 빗나가는데, 그때는
  // hasNext/hasPrevious가 false, getPlaylist()가 빈 배열을 돌려주므로 여기서
  // **버튼이 조용히 비활성/숨김이 될 뿐** 창은 그대로 동작한다.

  /**
   * 이전/다음·재생목록 버튼의 사용 가능 여부를 다시 판정한다.
   * 내비게이션으로 재생목록이 생기거나 사라질 수 있어 주기적으로도 호출된다.
   */
  refreshPlayerControls(): void {
    if (!this.win) return;
    if (this.prevBtn) this.prevBtn.disabled = !hasPrevious();
    if (this.nextBtn) this.nextBtn.disabled = !hasNext();
    const entries = getPlaylist();
    if (this.playlistBtn) this.playlistBtn.style.display = entries.length > 0 ? '' : 'none';
    if (entries.length === 0 && this.playlistOpen) this.playlistOpen = false;
    if (this.playlistOpen) this.renderPlaylist(entries);
    else if (this.playlistEl) this.playlistEl.style.display = 'none';
  }

  /** 재생목록 목록 렌더 — 항목 클릭 시 그 곡으로 이동 */
  private renderPlaylist(entries = getPlaylist()): void {
    const listEl = this.playlistEl;
    if (!listEl) return;
    if (!this.playlistOpen || entries.length === 0) {
      listEl.style.display = 'none';
      return;
    }
    listEl.replaceChildren(...entries.map(entry => {
      const btn = h('button', {
        className: `ey-pip-playlist-item${entry.selected ? ' current' : ''}`,
        title: entry.byline ? `${entry.title} — ${entry.byline}` : entry.title,
        on: {
          click: () => {
            // 이동에 실패하면(셀렉터 불일치) 목록을 다시 판정해 사라지게 둔다
            if (!playPlaylistItem(entry.index)) this.refreshPlayerControls();
          },
        },
      },
        h('span', { className: 'ey-pip-playlist-title', text: entry.title }),
        h('span', { className: 'ey-pip-playlist-by', text: entry.byline }),
      );
      return btn;
    }));
    listEl.style.display = '';
  }

  /** 멜로디/메트로놈 토글 버튼 활성 상태 반영 — 세부 버튼은 메트로놈+레인이 켜졌을 때만 */
  setAudioState(melody: boolean, metronome: boolean): void {
    this.metroOn = metronome;
    this.melodyBtn?.classList.toggle('on', melody);
    this.metroBtn?.classList.toggle('on', metronome);
    this.syncKaraokeControls();
  }

  /** 레인 위 디바이더 — 위로 끌면 레인이 커진다. 놓으면 설정에 저장. */
  /** 레인 상호작용 — 클릭=그 위치로 시크(노트 위면 노트 시작으로 스냅),
   *  일시정지 중 드래그·Shift+휠=좌우 탐색 (멜로다인식) */
  private attachPitchPointer(canvas: HTMLCanvasElement): void {
    canvas.addEventListener('pointerdown', e => {
      const view = this.lane.viewport();
      if (!view) return;
      const manual = this.lane.getManualT0();
      this.pitchPointer = {
        id: e.pointerId, startX: e.clientX,
        startT0: this.paused && manual != null ? manual : view.t0,
        moved: false,
      };
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener('pointermove', e => {
      const p = this.pitchPointer;
      const view = this.lane.viewport();
      if (!p || p.id !== e.pointerId || !view) return;
      const dx = e.clientX - p.startX;
      if (Math.abs(dx) > 4) p.moved = true;
      if (p.moved && this.paused) {
        this.lane.setManualT0(p.startT0 - dx * (view.W / view.plotW));
        this.renderLane(); // 일시정지 중엔 tick이 없어도 즉시 반영
      }
    });
    canvas.addEventListener('pointerup', e => {
      const p = this.pitchPointer;
      this.pitchPointer = null;
      const view = this.lane.viewport();
      if (!p || p.id !== e.pointerId || !view || p.moved) return;
      const rect = canvas.getBoundingClientRect();
      const px = e.clientX - rect.left;
      const py = e.clientY - rect.top;
      if (px < view.plotX || py > view.staffBottom) return; // 사이드바·가사 줄 클릭은 무시
      const t = view.t0 + ((px - view.plotX) / view.plotW) * view.W;
      const hit = this.lane.noteAt(t);
      this.onSeek(hit ? hit.start : Math.max(0, t));
    });
    canvas.addEventListener('wheel', e => {
      // K1: 순수 세로 휠 = 줌(레인 표시 구간 조절). Ctrl+휠은 브라우저 확대/축소와
      // 충돌하니 절대 안 건드리고, 가로 휠·Shift+세로휠은 기존 팬(아래)에 그대로 맡긴다.
      // ±버튼(windowMinusBtn/windowPlusBtn)과 정확히 같은 값·같은 클램프(setPitchWindow가
      // 이미 [0.25,16]으로 한 번 더 clamp)를 재사용해 저장·복원·렌더 경로가 전부 기존
      // 그대로다 — 새 배율 축을 만들지 않는 것이 버그 표면을 최소화한다는 지시.
      // 재생 중에도 동작한다(아래 팬과 달리 paused 조건 없음 — ±버튼도 마찬가지다).
      if (!e.ctrlKey && !e.shiftKey && e.deltaX === 0 && e.deltaY !== 0) {
        e.preventDefault(); // 레인 위에서만 막는다 — 페이지 스크롤 침범 금지
        const next = e.deltaY < 0
          ? Math.max(0.5, this.pitchWindowMeasures / 2)   // 휠 위로 = 확대(표시 구간 축소)
          : Math.min(16, this.pitchWindowMeasures * 2);   // 휠 아래로 = 축소(표시 구간 확장)
        this.setPitchWindow(next);
        this.onPitchWindowChange(next);
        this.renderLane(); // 일시정지 중엔 tick이 없어도 즉시 반영(아래 팬과 같은 이유)
        return;
      }
      if (!this.paused) return;
      const view = this.lane.viewport();
      if (!view) return;
      const delta = e.deltaX !== 0 ? e.deltaX : e.shiftKey ? e.deltaY : 0;
      if (delta === 0) return;
      e.preventDefault();
      this.lane.setManualT0((this.lane.getManualT0() ?? view.t0) + delta * (view.W / view.plotW));
      this.renderLane();
    }, { passive: false });
  }

  private buildPitchDivider(onHeightChange: (px: number) => void): HTMLDivElement {
    const divider = h('div', {
      className: 'ey-pip-divider ey-pip-pitch-divider',
      title: t('pip.controls.laneHeightDrag'),
    }, h('div', { className: 'ey-pip-divider-grip' }));
    let dragging = false;
    let startY = 0;
    let startH = 0;
    divider.addEventListener('pointerdown', (e: PointerEvent) => {
      if (!this.pitchCanvas) return;
      dragging = true;
      startY = e.clientY;
      startH = this.pitchCanvas.clientHeight;
      divider.setPointerCapture(e.pointerId);
      e.preventDefault();
    });
    divider.addEventListener('pointermove', (e: PointerEvent) => {
      if (!dragging || !this.pitchCanvas) return;
      const px = Math.min(this.maxLaneHeight(), Math.max(90, Math.round(startH + (startY - e.clientY))));
      this.pitchLaneHeight = px;
      this.pitchCanvas.style.height = `${px}px`;
    });
    divider.addEventListener('pointerup', (e: PointerEvent) => {
      if (!dragging) return;
      dragging = false;
      divider.releasePointerCapture(e.pointerId);
      onHeightChange(this.pitchLaneHeight);
    });
    return divider;
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
    if (!paused) this.lane.setManualT0(null); // 재생 재개 → 수동 스크롤 해제, 오토스크롤 복귀
    if (this.playBtn && paused !== this.lastPaused) {
      this.lastPaused = paused;
      this.playBtn.replaceChildren(icon(paused ? PLAY_SVG : PAUSE_SVG));
    }
    // 자기치유 안전망: PiP rAF가 살아 있으면 이 조건은 성립하지 않아 아무 일도 없다.
    // 죽어 있으면 lastFrameAt이 갱신되지 않아 **매 tick** 여기서 그린다 — 즉 렌더 주기가
    // 정확히 예전(tick 주기: 보이면 60Hz, 숨으면 ~4Hz)으로 되돌아간다.
    // (tick 자체가 끊긴 경우엔 이 함수가 안 불리므로 발동하지 않고 마지막 프레임에서
    //  정지한다 — sampleTime의 STATE_STALE_MS와 같은 결론이다.)
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
   * 한 프레임의 시간 의존 렌더 — 가사 채움 + 진행 바 + 시간 라벨 + 레인.
   * 이 창의 rAF가 매 프레임 부르고, rAF가 죽은 환경에서는 메인 창 tick이 대신 부른다.
   */
  private renderFrame(time: number): void {
    if (!this.win) return;
    const at = performance.now();
    // 곡 시간이 멈춰 있으면 재그리기를 늦춘다 — 건너뛰지는 않는다(IDLE_REDRAW_MS 주석).
    // `paused`를 보지 않고 **시각이 같은지**만 보는 이유: 멈추는 경우가 일시정지 하나가
    // 아니다 — 메인 창이 tick을 끊으면(내비게이션) sampleTime이 STATE_STALE_MS에 걸려
    // 같은 시각을 계속 돌려주는데, 그때도 같은 그림을 60~120Hz로 다시 칠할 이유가 없다.
    // 재생 중에는 프레임마다 보간값이 달라 이 조건이 성립하지 않는다.
    if (time === this.lastTime && at - this.lastPaintAt < IDLE_REDRAW_MS) return;
    this.lastPaintAt = at;
    this.lastTime = time;
    const duration = this.state.duration;
    for (const { start, el } of this.wordEls) {
      el.classList.toggle('sung', start <= time);
    }
    if (this.progressEl && duration > 0) {
      this.progressEl.style.width = `${Math.min(100, (time / duration) * 100)}%`;
    }
    if (this.timeEl && duration > 0) {
      // 초 단위 문구는 프레임마다 같다 — 바뀔 때만 써서 불필요한 리플로우를 피한다
      const label = `${formatTime(time)} / ${formatTime(duration)}`;
      if (this.timeEl.textContent !== label) this.timeEl.textContent = label;
    }
    this.lane.render(time, this.paused);
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
    // shrink 1: 사용자가 지정한 비율이라도 가사 스테이지 min-height는 침범하지 못한다.
    // 레인 모드 여부에 따른 실제 스타일 적용은 syncVideoLayout이 담당.
    this.videoRatio = ratio;
    this.syncVideoLayout();
  }

  private hideVideoArea(): void {
    if (this.videoWrapEl) {
      this.videoWrapEl.style.display = 'none';
      this.videoWrapEl.replaceChildren();
    }
    if (this.dividerEl) this.dividerEl.style.display = 'none';
    this.syncLaneSizing(); // 영상이 사라지면 레인이 창을 채운다
  }

  private stopMirror(): void {
    this.mirrorStream?.getTracks().forEach(track => track.stop());
    this.mirrorStream = null;
  }

  private seekRelative(offset: number): void {
    const line = this.lines[this.index + offset];
    if (line && line.time !== null) this.onSeek(line.time);
  }

  private renderLines(): void {
    if (!this.prevEl || !this.currentEl || !this.nextEl) return;
    const prev = this.lines[this.index - 1];
    const current = this.index >= 0 ? this.lines[this.index] : undefined;
    const next = this.lines[this.index + 1];

    this.prevEl.textContent = prev?.text ?? '';
    this.nextEl.textContent = next?.text ?? '';
    if (this.trEl) this.trEl.textContent = current?.translation ?? '';

    this.wordEls = [];
    // 발음도 음절 타이밍(pronSegments)이 있으면 패널처럼 부른 만큼 색이 차오르게 —
    // 음절 span을 wordEls에 합류시켜 프레임 렌더(renderFrame)의 sung 토글을 그대로 태운다
    if (this.pronEl) {
      this.pronEl.replaceChildren();
      const pron = (current && resolvedPronunciation(current, this.pronScript)) ?? '';
      const segs = current ? resolvedPronSegments(current, this.pronScript) : undefined;
      const mapped = current && pron && segs && segs.length > 0
        ? appendTimedSpans(this.pronEl, pron, segs, s => s.text, seg => {
            const el = h('span', { className: 'ey-pron-syl', text: seg.text });
            this.wordEls.push({ start: seg.start, el });
            return el;
          })
        : 0;
      if (mapped === 0) this.pronEl.textContent = pron;
    }
    this.currentEl.replaceChildren();
    if (!current) {
      this.currentEl.textContent = '♪';
      return;
    }
    // words가 없어도 호출한다 — appendKaraokeSpans가 음절 타이밍/라인 구간으로
    // 비례 배분 폴백을 만들어 라인이 통째로 켜지는 것을 피한다
    appendKaraokeSpans(this.currentEl, current, word => {
      const el = h('span', { className: 'ey-word', text: word.word });
      this.wordEls.push({ start: word.start, el });
      return el;
    });
  }
}

function formatTime(sec: number): string {
  const total = Math.max(0, Math.floor(sec));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${s.toString().padStart(2, '0')}`;
}
