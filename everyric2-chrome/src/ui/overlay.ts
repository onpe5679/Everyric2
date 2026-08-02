import type { DebugInfo, LyricLine, LyricsSource, PanelGeometry, SearchCandidate, ServerLogEntry, ServerStatus, Settings, SongInfo, SourceAttribution, SyncDebugMeta, SyncListItem, SyncPreviousVersion } from '../types';
import { resolveScript, resolvedPronSegments, resolvedPronunciation, type PronScript } from '../lib/lang';
import { t } from '../lib/i18n';
import { needsHostPermission, serverUsable, statusLine, unknownStatus } from '../lib/server-status';
import { resolveTheme } from '../lib/theme';
import { buildDebugPanel } from './debug-panel';
import { h, icon, ICONS } from './dom';
import { appendKaraokeSpans, appendTimedSpans } from './karaoke';
import {
  applyServerGate,
  buildEmptyState,
  buildErrorState,
  buildGeneratingState,
  buildLoadingState,
  buildPlainLines,
  buildSearchSheet,
  buildServerStatusSlot,
  createGenerateButton,
  renderCandidateList,
  setListStatus,
  type PanelContext,
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

export interface OverlayCallbacks {
  onSeek: (time: number) => void;
  /** attribution은 붙여넣기 경로에서 사용자가 적어 넣은 출처(선택) */
  onGenerate: (lyrics: string, attribution?: string) => void;
  onRetrySearch: (query?: { title: string; artist: string }) => void;
  onOffsetChange: (offsetSec: number) => void;
  onSettingsChange: (patch: Partial<Settings>) => void;
  /** 현재 everyric 싱크의 강제 재생성 (서버 캐시 무시) */
  onRegenerate: () => void;
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
  /** 분석 깊이 올리기/구세대 업그레이드 — minDepth 없으면 일반 재생성(=신 스택 자동 라우팅) */
  onDepthUpgrade: (minDepth?: 'medium' | 'heavy') => void;
  /** 정렬 품질 별점(1~5) + 선택 오류 제보 전송 — 성공 여부를 돌려준다 */
  onSubmitFeedback: (rating: number, category?: string, comment?: string) => Promise<boolean>;
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
}

type StateKind = 'loading' | 'synced' | 'plain' | 'empty' | 'generating' | 'error' | 'pip' | 'search';

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
  /** 미보유 언어 칩을 눌러 번역을 기다리는 동안의 표시(U3-b) — 칩 펄스만으론 눈에 잘
   *  안 띈다는 실보고로 추가. 라인 목록 바로 위(.ey-warn-bar와 같은 자리 규칙)에 둬서
   *  "빈 번역 줄"이 아니라 "준비 중"임을 알린다. .ey-tr-status 클래스를 재사용한다
   *  (이번 작업은 overlay.css 수정 권한이 없어 기존 클래스만 쓴다). */
  private translationPendingBar: HTMLDivElement;
  /** 서버 오류 배너 — body 밖에 있어 resetBody()로 지워지지 않는다.
   *  덕분에 어떤 화면(가사·검색·생성 중·오류)에서도 사유 한 줄이 반드시 보인다. */
  private serverBar: HTMLDivElement;
  private pipBtn: HTMLButtonElement;
  private regenBtn: HTMLButtonElement;
  private depthBtn: HTMLButtonElement;
  /** depthBtn 클릭 시 동작 — 상태(깊이/구세대/최대)에 따라 updateDepthButton이 바꾼다 */
  private depthAction: (() => void) | null = null;
  private feedbackBtn: HTMLButtonElement;
  private feedbackPop: HTMLDivElement;
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
  private activeWordEls: { start: number; el: HTMLElement }[] = [];
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
  private saveGeomTimer = 0;
  private resizeObserver: ResizeObserver;

  constructor(cssText: string, settings: Settings, callbacks: OverlayCallbacks, geometry: PanelGeometry | null) {
    this.settings = settings;
    this.callbacks = callbacks;
    this.offsetSec = settings.offsetSec;

    this.host = h('div', { attrs: { id: 'everyric-root' } });
    this.host.style.cssText = 'all:initial;position:fixed;top:0;left:0;width:0;height:0;z-index:2147483647;';
    const shadow = this.host.attachShadow({ mode: 'open' });

    const style = document.createElement('style');
    style.textContent = cssText;
    shadow.append(style);

    this.songTitleEl = h('div', { className: 'ey-song-title', text: t('overlay.detecting') });
    this.songArtistEl = h('div', { className: 'ey-song-artist' });

    this.pipBtn = this.headerButton(ICONS.pip, t('overlay.header.pip'), () => this.callbacks.onPipToggle());
    this.pipBtn.style.display = 'none';
    this.regenBtn = this.headerButton(ICONS.refresh, t('overlay.header.regen'), () => {
      if (window.confirm(t('overlay.header.regenConfirm'))) {
        this.callbacks.onRegenerate();
      }
    });
    this.regenBtn.style.display = 'none';
    // 분석 깊이 버튼 — 내용(아이콘·배지·툴팁·동작)은 updateDepthButton이 상태에 따라 채운다
    this.depthBtn = h('button', {
      className: 'ey-btn ey-depth-btn',
      attrs: { type: 'button' },
      on: { click: () => this.depthAction?.() },
    });
    this.depthBtn.style.display = 'none';
    const searchBtn = this.headerButton(ICONS.search, t('overlay.header.search'), () => this.openSearch());
    const gearBtn = this.headerButton(ICONS.gear, t('overlay.header.settings'), () => this.toggleSettings());
    this.collapseBtn = this.headerButton(ICONS.collapse, t('overlay.header.collapse'), () => this.setCollapsed(!this.geometry.collapsed));
    const closeBtn = this.headerButton(ICONS.close, t('overlay.header.close'), () => this.setVisible(false));

    this.header = h('div', { className: 'ey-header' },
      h('div', { className: 'ey-header-left' },
        icon(ICONS.note),
        h('div', { className: 'ey-song' }, this.songTitleEl, this.songArtistEl),
      ),
      h('div', { className: 'ey-actions' }, this.pipBtn, this.depthBtn, this.regenBtn, searchBtn, gearBtn, this.collapseBtn, closeBtn),
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

    this.panel = h('div', { className: 'ey-panel' },
      this.header, this.langChipsRow, this.serverBar, this.banner, this.genChip, this.genList, this.noticeChip,
      this.warnBar, this.translationPendingBar, this.body, this.resumeChip, this.footer, this.debugStrip, this.debugPanelEl,
    );
    // 패널 안 타이핑(검색창·가사 붙여넣기)이 유튜브 전역 단축키(스페이스=재생/정지,
    // 방향키=시킹 등)로 새지 않도록 키 이벤트를 패널에서 끊는다
    for (const type of ['keydown', 'keyup', 'keypress'] as const) {
      this.panel.addEventListener(type, e => e.stopPropagation());
    }
    shadow.append(this.panel);

    this.geometry = geometry ?? this.defaultGeometry();
    this.applyGeometry();
    this.applySettings(settings);
    this.updateOffsetLabel();

    this.setupDrag();
    this.resizeObserver = new ResizeObserver(() => this.handlePanelResize());
    this.resizeObserver.observe(this.panel);
    window.addEventListener('resize', this.handleWindowResize);
    document.addEventListener('fullscreenchange', this.handleFullscreenChange);

    document.documentElement.append(this.host);
  }

  /** 현재 오버레이는 페이지 수명 싱글턴이라 호출처가 없다 — 향후 하드 teardown 경로용 */
  destroy(): void {
    this.resizeObserver.disconnect();
    window.removeEventListener('resize', this.handleWindowResize);
    document.removeEventListener('fullscreenchange', this.handleFullscreenChange);
    clearTimeout(this.saveGeomTimer);
    this.host.remove();
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
      const el = h('div', {
        className: 'ey-line',
        title: t('overlay.line.seekTitle'),
        // dir=auto — RTL(아랍어·히브리어) 가사가 문장 방향대로 정렬되게
        attrs: { dir: 'auto' },
        on: {
          click: () => {
            // 줄 시작 시각으로 정확히 시크하면 브라우저가 그 지점 **이하**의 디코딩
            // 가능한 위치로 스냅해서, 곡 시간이 줄 시작보다 살짝 앞에 떨어진다.
            // 그러면 활성 줄 판정이 한 줄 위로 가서 "누른 줄의 윗칸이 눌린" 것처럼
            // 보인다. 줄 안쪽으로 아주 조금 밀어 넣어 의도한 줄에서 시작하게 한다.
            if (line.time !== null) this.callbacks.onSeek(line.time + SEEK_INTO_LINE_SEC);
          },
        },
      });
      // words가 없어도 호출 — appendKaraokeSpans가 음절 타이밍/라인 구간 비례
      // 배분으로 폴백해, 라인이 한 번에 통째로 켜지는 표시를 피한다
      appendKaraokeSpans(el, line, word => {
        // 신뢰도 등급 클래스 — .ey-show-conf(디버그 모드)에서만 색이 입혀진다.
        // 값은 CTC 프레임 로그확률의 기하평균(0~1) — 절대값이 작아 로그 스케일로 버킷:
        // <1e-4(로그 -9 이하)=낮음, <2e-2(로그 -4 이하)=중간
        const conf = word.confidence;
        // 버킷 색은 레인(pip.ts confBucketColor)과 동일: 빨강<1e-4, 노랑<2e-2, 초록=양호
        const confClass = conf == null ? '' : conf < 1e-4 ? ' ey-conf-low' : conf < 2e-2 ? ' ey-conf-mid' : ' ey-conf-ok';
        return h('span', { className: `ey-word${confClass}`, text: word.word, attrs: { 'data-start': String(word.start) } });
      });
      const pronEl = this.buildPronEl(line, pronScript);
      if (pronEl) el.append(pronEl);
      if (line.translation) el.append(h('div', { className: 'ey-line-tr', text: line.translation, attrs: { dir: 'auto' } }));
      el.dataset.index = String(index);
      this.lineEls.push(el);
      list.append(el);
    });
    this.body.append(list);

    this.setSourceBadge(source, true);
    this.footer.classList.remove('no-offset');
    this.footer.style.display = '';
    this.pipBtn.style.display = this.pipEnabled ? '' : 'none';
    // 재생성은 서버(everyric) 싱크에서만 의미가 있다
    this.regenBtn.style.display = source === 'everyric' ? '' : 'none';
    // 깊이 버튼도 여기서 갱신 — 구세대 싱크면 재생성 버튼을 업그레이드 버튼이 대신한다
    this.updateDepthButton();
    // 별점·오류 제보도 everyric 싱크에서만 — 평가 대상이 서버 정렬이다
    this.feedbackBtn.style.display = source === 'everyric' ? '' : 'none';
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
      { title: this.lastSong?.title ?? '', artist: this.lastSong?.artist ?? '' },
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
              click: () => {
                if (window.confirm(t('overlay.search.resetSyncConfirm'))) {
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
      for (const wordEl of active.querySelectorAll<HTMLElement>('.ey-word, .ey-pron-syl')) {
        this.activeWordEls.push({ start: Number(wordEl.dataset.start), el: wordEl });
      }
      if (Date.now() >= this.userScrollUntil) {
        this.scrollToCurrent();
      } else {
        this.resumeChip.style.display = '';
      }
    }
  }

  updateTime(time: number): void {
    for (const { start, el } of this.activeWordEls) {
      el.classList.toggle('sung', start <= time);
    }
  }

  /** 한 줄의 글자·음절을 전부 채우거나 전부 비운다 */
  private setLineFilled(i: number, filled: boolean): void {
    const el = this.lineEls[i];
    if (!el) return;
    for (const w of el.querySelectorAll<HTMLElement>('.ey-word, .ey-pron-syl')) {
      w.classList.toggle('sung', filled);
    }
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

  setVisible(visible: boolean): void {
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
    this.applyRegenGate();
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
  private applyRegenGate(): void {
    applyServerGate(this.regenBtn, this.serverStatus, t('overlay.header.regen'));
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

  setPipEnabled(enabled: boolean): void {
    this.pipEnabled = enabled;
    this.pipBtn.style.display = enabled && this.stateKind === 'synced' ? '' : 'none';
  }

  setPipActive(active: boolean): void {
    this.pipBtn.classList.toggle('active', active);
  }

  isShowingPipPlaceholder(): boolean {
    return this.stateKind === 'pip';
  }

  /**
   * 라인 하나의 발음 표기 엘리먼트를 만든다 — showSyncedLyrics(최초 렌더)와
   * refreshTranslations(재렌더: 번역 API가 늦게 채워줄 때·발음 표기 전환 시) 둘 다
   * 이걸 거친다. 음절 타이밍(pronSegments)이 있으면 단어처럼 부른 만큼 색이 차오르게
   * 스팬으로(사이 텍스트는 appendTimedSpans가 인접 span에 끼워 넣어 흰 글자 없이
   * 칠해진다), 없으면 통짜 텍스트로. 발음이 없으면 null(둘 다 append를 생략하게).
   */
  private buildPronEl(line: LyricLine, pronScript: PronScript): HTMLDivElement | null {
    const pron = resolvedPronunciation(line, pronScript);
    if (!pron) return null;
    const segs = resolvedPronSegments(line, pronScript);
    const pronEl = h('div', { className: 'ey-line-pron', attrs: { dir: 'auto' } });
    const mapped = segs && segs.length > 0
      ? appendTimedSpans(pronEl, pron, segs, s => s.text, seg =>
          h('span', {
            className: 'ey-pron-syl',
            text: seg.text,
            attrs: { 'data-start': String(seg.start) },
          }))
      : 0;
    if (mapped === 0) pronEl.replaceChildren(pron);
    return pronEl;
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
        const pronEl = this.buildPronEl(line, pronScript);
        if (pronEl) el.append(pronEl);
      }
      if (line?.translation) el.append(h('div', { className: 'ey-line-tr', text: line.translation, attrs: { dir: 'auto' } }));
    });
  }

  setTranslationStatus(text: string | null): void {
    this.trStatusEl.textContent = text ?? '';
  }

  /** 낮은 정렬 신뢰도 경고 바 — score가 null이면 숨김. X로 닫을 수 있다. */
  setQualityWarning(score: number | null): void {
    if (score === null) {
      this.warnBar.style.display = 'none';
      return;
    }
    this.warnBar.replaceChildren(
      h('span', {
        className: 'ey-warn-text',
        text: `⚠️ ${t('overlay.warn.text', [fmtConf(score)])}`,
        attrs: { title: t('overlay.warn.title') },
      }),
      h('button', {
        className: 'ey-warn-close',
        text: '×',
        title: t('overlay.warn.close'),
        on: { click: () => { this.warnBar.style.display = 'none'; } },
      }),
    );
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
            if (window.confirm(t('overlay.genChip.cancelConfirm'))) this.callbacks.onCancelGenerate();
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
  setDebugMeta(meta: SyncDebugMeta | null): void {
    this.debugMeta = meta;
    if (this.debugPanelOpen) this.renderDebugPanel(); // 열려 있으면 요약줄도 즉시 갱신
    this.updateDepthButton(); // 깊이·세대 정보의 출처가 이 메타다
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
      () => this.callbacks.onLoadPreviousSync());
    this.debugPanelEl.replaceChildren(el);
    this.debugPanelEl.style.display = '';
  }

  applySettings(settings: Settings): void {
    this.settings = settings;
    // 번역 언어가 설정 시트·다른 경로로 바뀌어도 제목바 칩의 "현재 선택"이 따라가야 한다
    this.renderLangChips();
    this.panel.classList.remove('ey-fs-small', 'ey-fs-medium', 'ey-fs-large');
    this.panel.classList.add(`ey-fs-${settings.fontSize}`);
    // 테마 판정은 lib/theme.ts 한 곳에서만 — PiP도 content가 같은 값을 받아 칠한다
    this.panel.classList.toggle('ey-light', resolveTheme(settings) === 'light');
    // 오프셋은 영상별 상태(setOffsetValue로 주입) — 전역 설정으로 되돌리지 않는다
    this.debugStrip.style.display = settings.debugInfo ? '' : 'none';
    if (!settings.debugInfo) this.closeDebugPanel(); // 버튼이 숨는데 패널만 열려 남으면 안 된다
    this.panel.classList.toggle('ey-hide-pron', !settings.showPronunciation);
    // 디버그 모드에서 글자별 CTC 신뢰도를 색으로 표시
    this.panel.classList.toggle('ey-show-conf', settings.debugInfo);
    // 디버그 토글은 서버 요청 로그의 노출 조건이기도 하다 — 배너를 다시 그려 반영
    this.renderServerBar();
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

  /** 별점(1~5) + 오류 유형·코멘트 팝오버 — 열 때마다 초기 상태로 다시 그린다 */
  private renderFeedbackPop(): void {
    let rating = 0;
    const stars: HTMLButtonElement[] = [];
    const paint = () => stars.forEach((s, i) => s.classList.toggle('on', i < rating));
    const starRow = h('div', { className: 'ey-feedback-stars' });
    for (let i = 1; i <= 5; i++) {
      const s = h('button', {
        className: 'ey-feedback-star', text: '★', attrs: { type: 'button' },
        on: { click: () => { rating = i; paint(); } },
      });
      stars.push(s);
      starRow.append(s);
    }
    const category = h('select', { className: 'ey-select' }, ...([
      ['', t('overlay.feedback.catNone')],
      ['timing', t('overlay.feedback.catTiming')],
      ['pronunciation', t('overlay.feedback.catPron')],
      ['lyrics', t('overlay.feedback.catLyrics')],
      ['other', t('overlay.feedback.catOther')],
    ] as [string, string][]).map(([v, label]) => h('option', { text: label, attrs: { value: v } })));
    const comment = h('input', {
      className: 'ey-input',
      attrs: { placeholder: t('overlay.feedback.commentPh'), maxlength: '500' },
    });
    const status = h('span', { className: 'ey-feedback-status' });
    const send = h('button', {
      className: 'ey-btn', text: t('overlay.feedback.send'), attrs: { type: 'button' },
      on: {
        click: () => {
          if (rating === 0) {
            status.textContent = t('overlay.feedback.needRating');
            return;
          }
          send.disabled = true;
          void this.callbacks
            .onSubmitFeedback(rating, category.value || undefined, comment.value.trim() || undefined)
            .then(ok => {
              status.textContent = ok ? t('overlay.feedback.thanks') : t('overlay.feedback.failed');
              if (ok) {
                window.setTimeout(() => { this.feedbackPop.style.display = 'none'; }, 1200);
              } else {
                send.disabled = false; // 실패 — 입력을 남긴 채 재시도 가능
              }
            });
        },
      },
    }) as HTMLButtonElement;
    this.feedbackPop.replaceChildren(
      h('div', { className: 'ey-feedback-row' }, starRow),
      h('div', { className: 'ey-feedback-row' }, category),
      h('div', { className: 'ey-feedback-row' }, comment),
      h('div', { className: 'ey-feedback-row' }, send, status),
    );
  }

  /**
   * 분석 깊이 버튼 — 헤더에서 현재 싱크의 분석 깊이(1=무분리 ASR, 2=분리+ASR,
   * 3=분리+ASR+OWSM 앵커)를 화살표 나눔선·배지 숫자로 보여주고, 클릭하면 한 단계
   * 깊은 재분석(regenerate min_depth)을 요청한다. 최대 깊이(3)는 빨간 배지 + 비활성 +
   * "가사 입력 상태를 확인하세요" 툴팁. 구세대 싱크(engine_version 스탬프 없음/구서버)는
   * 노란 업그레이드 버튼이 되고 **재생성 버튼을 대신한다**(운영자 지시 — 그 경우 일반
   * 재생성 자체가 곧 새 엔진 업그레이드다).
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
        if (window.confirm(t('overlay.depth.upgradeConfirm'))) this.callbacks.onDepthUpgrade();
      };
      this.depthBtn.style.display = '';
      this.regenBtn.style.display = 'none'; // 업그레이드 버튼이 재생성 버튼을 대신한다
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
        if (window.confirm(t('overlay.depth.confirm'))) this.callbacks.onDepthUpgrade(next);
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
    this.banner.style.display = 'none';
    this.footer.style.display = 'none';
    this.resumeChip.style.display = 'none';
    this.pipBtn.style.display = 'none';
    this.regenBtn.style.display = 'none';
    this.depthBtn.style.display = 'none';
    this.depthAction = null;
    this.feedbackBtn.style.display = 'none';
    this.feedbackPop.style.display = 'none';
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
    const base = source === 'everyric' ? 'Everyric'
      : source === 'vocaro'
        ? (this.attributionSourceId === 'miraheze' ? t('overlay.source.miraheze') : t('overlay.source.vocaro'))
      : source === 'caption' ? t('overlay.source.caption')
      : 'LRCLIB';
    // 가사 원출처(위키 등)를 병기 — 전사는 서버가 했어도 가사의 출처는 따로 표기
    const extra = this.attributionName && this.attributionName !== base ? ` · ${this.attributionName}` : '';
    // 다른 영상의 싱크를 빌려온 경우 링크 표시 (해제는 검색 시트에서).
    // 검증(반주 대조)을 통과한 자동 링크와 검증 없는 수동 링크는 신뢰도가 다르다 —
    // 어긋난 가사를 보고 있을 때 원인을 짚을 수 있도록 ✓/? 로 구분해 표시한다
    const link = this.linkedInfo
      ? ` · 🔗${this.linkedInfo.verified ? '✓' : '?'}${this.linkedInfo.offsetSec !== 0 ? `${this.linkedInfo.offsetSec > 0 ? '+' : ''}${this.linkedInfo.offsetSec}s` : ''}`
      : '';
    // 번역 출처 병기(U2) — 가사 원출처(extra)와 별개로, 사후 채택 번역이 어디서 왔는지.
    // kind==='wiki'면 실제로 히트한 위키 이름(translationSourceWikiName)을 그대로 쓴다.
    const trSource = this.translationSourceKind === 'caption' ? ` · ${t('overlay.translationSource.caption')}`
      : this.translationSourceKind === 'wiki'
        ? ` · ${t('overlay.translationSource.wiki', [this.translationSourceWikiName ?? t('overlay.source.vocaro')])}`
      : this.translationSourceKind === 'llm' ? ` · ${t('overlay.translationSource.llm')}`
      : '';
    this.sourceBadge.textContent = base + extra + link + trSource;
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
    const sheet = this.buildSettingsSheet();
    this.settingsSheet = sheet;
    this.panel.append(sheet);
  }

  private closeSettings(): void {
    this.settingsSheet?.remove();
    this.settingsSheet = null;
    this.settingsDot = null;
    this.settingsPermBtn = null;
  }

  private buildSettingsSheet(): HTMLDivElement {
    const autoSearch = h('input', { attrs: { type: 'checkbox' } });
    autoSearch.checked = this.settings.autoSearch;
    autoSearch.addEventListener('change', () => this.callbacks.onSettingsChange({ autoSearch: autoSearch.checked }));

    const autoSearchShorts = h('input', { attrs: { type: 'checkbox' } });
    autoSearchShorts.checked = this.settings.autoSearchShorts;
    autoSearchShorts.addEventListener('change', () =>
      this.callbacks.onSettingsChange({ autoSearchShorts: autoSearchShorts.checked }));

    const fontSelect = this.buildSelect(
      [['small', t('overlay.settings.fontSize.small')], ['medium', t('overlay.settings.fontSize.medium')], ['large', t('overlay.settings.fontSize.large')]],
      this.settings.fontSize,
      v => this.callbacks.onSettingsChange({ fontSize: v as Settings['fontSize'] }),
    );
    const themeSelect = this.buildSelect(
      [['auto', t('overlay.settings.optAuto')], ['dark', t('overlay.settings.theme.dark')], ['light', t('overlay.settings.theme.light')]],
      this.settings.theme,
      v => this.callbacks.onSettingsChange({ theme: v as Settings['theme'] }),
    );

    const showTranslation = h('input', { attrs: { type: 'checkbox' } });
    showTranslation.checked = this.settings.showTranslation;
    showTranslation.addEventListener('change', () =>
      this.callbacks.onSettingsChange({ showTranslation: showTranslation.checked }));

    const langSelect = this.buildSelect(
      // 언어 이름은 각 언어 자신의 표기로 고정 — uiLanguage가 바뀌어도 번역하지 않는다
      // (사용자가 어느 표시 언어에서도 자기 언어를 바로 찾을 수 있어야 하는 표준 관례)
      [['ko', '한국어'], ['en', 'English'], ['ja', '日本語'], ['zh', '中文']],
      this.settings.translationLanguage,
      v => this.callbacks.onSettingsChange({ translationLanguage: v }),
    );

    // 확장 UI 표시 언어 — 지금은 값만 저장한다(실제 반영은 다음 i18n 태스크에서: chrome.i18n은
    // 브라우저 로케일에 고정되므로 이 값이 있어야 사용자가 직접 오버라이드할 수 있다)
    const uiLangSelect = this.buildSelect(
      [['auto', t('overlay.settings.optAuto')], ['ko', '한국어'], ['en', 'English'], ['ja', '日本語']],
      this.settings.uiLanguage,
      v => this.callbacks.onSettingsChange({ uiLanguage: v as Settings['uiLanguage'] }),
    );

    // 발음 표기 방식 — '자동'이면 번역 언어 기준으로 hangul/romaji/kana 중 골라진다
    // (lib/lang.ts resolveScript). 서버가 표기별 발음(pron dict)을 아직 안 주므로 지금은
    // hangul(한글 독음) 외의 선택은 화면상 차이가 없다 — 표기가 배포되면 그대로 반영된다.
    const pronScriptSelect = this.buildSelect(
      [['auto', t('overlay.settings.optAuto')], ['hangul', t('overlay.settings.pronScript.hangul')], ['romaji', t('overlay.settings.pronScript.romaji')], ['kana', t('overlay.settings.pronScript.kana')], ['ipa', t('overlay.settings.pronScript.ipa')]],
      this.settings.pronunciationScript,
      v => this.callbacks.onSettingsChange({ pronunciationScript: v as Settings['pronunciationScript'] }),
    );

    const showPronunciation = h('input', { attrs: { type: 'checkbox' } });
    showPronunciation.checked = this.settings.showPronunciation;
    showPronunciation.addEventListener('change', () =>
      this.callbacks.onSettingsChange({ showPronunciation: showPronunciation.checked }));

    const sourcePriority = this.buildSelect(
      [['vocaro', t('overlay.settings.sourcePriority.vocaro')], ['lrclib', t('overlay.settings.sourcePriority.lrclib')]],
      this.settings.lyricsSourcePriority,
      v => this.callbacks.onSettingsChange({ lyricsSourcePriority: v as Settings['lyricsSourcePriority'] }),
    );

    const pipKeepPanel = h('input', { attrs: { type: 'checkbox' } });
    pipKeepPanel.checked = this.settings.pipKeepPanel;
    pipKeepPanel.addEventListener('change', () =>
      this.callbacks.onSettingsChange({ pipKeepPanel: pipKeepPanel.checked }));

    const pipShowVideo = h('input', { attrs: { type: 'checkbox' } });
    pipShowVideo.checked = this.settings.pipShowVideo;
    pipShowVideo.addEventListener('change', () =>
      this.callbacks.onSettingsChange({ pipShowVideo: pipShowVideo.checked }));

    const pitchGuide = h('input', { attrs: { type: 'checkbox' } });
    pitchGuide.checked = this.settings.pitchGuide;
    pitchGuide.addEventListener('change', () =>
      this.callbacks.onSettingsChange({ pitchGuide: pitchGuide.checked }));

    const pitchWindow = this.buildSelect(
      [
        ['0.5', t('overlay.settings.pitchWindow.half')], ['1', t('overlay.settings.pitchWindow.bars', ['1'])],
        ['2', t('overlay.settings.pitchWindow.bars', ['2'])], ['4', t('overlay.settings.pitchWindow.bars', ['4'])],
        ['8', t('overlay.settings.pitchWindow.bars', ['8'])],
      ],
      String(this.settings.pitchWindowMeasures),
      v => this.callbacks.onSettingsChange({ pitchWindowMeasures: Number(v) }),
    );

    const pitchMode = this.buildSelect(
      [['page', t('overlay.settings.pitchMode.page')], ['scroll', t('overlay.settings.pitchMode.scroll')]],
      this.settings.pitchScrollMode,
      v => this.callbacks.onSettingsChange({ pitchScrollMode: v as Settings['pitchScrollMode'] }),
    );

    const pitchFont = this.buildSelect(
      [['1', t('overlay.settings.pitchFont.normal')], ['1.2', t('overlay.settings.pitchFont.large')], ['1.45', t('overlay.settings.pitchFont.xlarge')], ['0.85', t('overlay.settings.pitchFont.small')]],
      String(this.settings.pitchFontScale),
      v => this.callbacks.onSettingsChange({ pitchFontScale: Number(v) }),
    );

    // K2: 계이름 표기 — 한국어(도레미)/영어(C4·D#5)
    const solfegeNotation = this.buildSelect(
      [
        ['korean', t('overlay.settings.solfegeNotation.korean')],
        ['english', t('overlay.settings.solfegeNotation.english')],
      ],
      this.settings.solfegeNotation,
      v => this.callbacks.onSettingsChange({ solfegeNotation: v as Settings['solfegeNotation'] }),
    );

    const pitchCountdown = h('input', { attrs: { type: 'checkbox' } });
    pitchCountdown.checked = this.settings.pitchCountdown;
    pitchCountdown.addEventListener('change', () =>
      this.callbacks.onSettingsChange({ pitchCountdown: pitchCountdown.checked }));

    const pitchF0Curve = h('input', { attrs: { type: 'checkbox' } });
    pitchF0Curve.checked = this.settings.pitchF0Curve;
    pitchF0Curve.addEventListener('change', () =>
      this.callbacks.onSettingsChange({ pitchF0Curve: pitchF0Curve.checked }));

    // K3: 음정선(f0 곡선·노트 바) 밝기 — 0.2~1.0(기존 볼륨 슬라이더와 같은 buildRange,
    // 범위만 좁힌다)
    const pitchF0Opacity = this.buildRange(
      this.settings.pitchF0Opacity, v => this.callbacks.onSettingsChange({ pitchF0Opacity: v }), 0.2, 1.5,
    );
    const pitchLineOpacity = this.buildRange(
      this.settings.pitchLineOpacity, v => this.callbacks.onSettingsChange({ pitchLineOpacity: v }), 0.2, 1,
    );

    const pitchPronPosition = this.buildSelect(
      [['note', t('overlay.settings.pronPosition.note')], ['bottom', t('overlay.settings.pronPosition.bottom')]],
      this.settings.pitchPronPosition,
      v => this.callbacks.onSettingsChange({ pitchPronPosition: v as Settings['pitchPronPosition'] }),
    );

    const melodyPlayback = h('input', { attrs: { type: 'checkbox' } });
    melodyPlayback.checked = this.settings.melodyPlayback;
    melodyPlayback.addEventListener('change', () =>
      this.callbacks.onSettingsChange({ melodyPlayback: melodyPlayback.checked }));
    const melodyVolume = this.buildRange(this.settings.melodyVolume, v =>
      this.callbacks.onSettingsChange({ melodyVolume: v }));

    const metronome = h('input', { attrs: { type: 'checkbox' } });
    metronome.checked = this.settings.metronome;
    metronome.addEventListener('change', () =>
      this.callbacks.onSettingsChange({ metronome: metronome.checked }));
    const metronomeVolume = this.buildRange(this.settings.metronomeVolume, v =>
      this.callbacks.onSettingsChange({ metronomeVolume: v }));
    const metronomeRate = this.buildSelect(
      [
        ['0.5', t('overlay.settings.metronomeRate.half')], ['1', t('overlay.settings.metronomeRate.one')],
        ['2', t('overlay.settings.metronomeRate.two')],
      ],
      String(this.settings.metronomeRate),
      v => this.callbacks.onSettingsChange({ metronomeRate: Number(v) }),
    );
    const metronomeBeat = this.buildSelect(
      [
        ['0', t('overlay.settings.metronomeBeat.n', ['1'])], ['1', t('overlay.settings.metronomeBeat.n', ['2'])],
        ['2', t('overlay.settings.metronomeBeat.n', ['3'])], ['3', t('overlay.settings.metronomeBeat.n', ['4'])],
      ],
      String(this.settings.metronomeBeat),
      v => this.callbacks.onSettingsChange({ metronomeBeat: Number(v) }),
    );

    const micPitch = h('input', { attrs: { type: 'checkbox' } });
    micPitch.checked = this.settings.micPitch;
    micPitch.addEventListener('change', () =>
      this.callbacks.onSettingsChange({ micPitch: micPitch.checked }));
    const micOctave = this.buildSelect(
      [
        ['-2', t('overlay.settings.micOctave.n', ['-2'])], ['-1', t('overlay.settings.micOctave.n', ['-1'])],
        ['0', t('overlay.settings.micOctave.none')], ['1', t('overlay.settings.micOctave.n', ['+1'])],
        ['2', t('overlay.settings.micOctave.n', ['+2'])],
      ],
      String(this.settings.micOctave),
      v => this.callbacks.onSettingsChange({ micOctave: Number(v) }),
    );

    const audioOut = h('select', { className: 'ey-select' });
    audioOut.addEventListener('change', () =>
      this.callbacks.onSettingsChange({ audioOutputId: audioOut.value }));
    const micDevice = h('select', { className: 'ey-select' });
    micDevice.addEventListener('change', () =>
      this.callbacks.onSettingsChange({ micDeviceId: micDevice.value }));
    void this.populateAudioDevices(audioOut, micDevice);

    const lowConfWarning = h('input', { attrs: { type: 'checkbox' } });
    lowConfWarning.checked = this.settings.lowConfWarning;
    lowConfWarning.addEventListener('change', () =>
      this.callbacks.onSettingsChange({ lowConfWarning: lowConfWarning.checked }));

    const notifyOnComplete = h('input', { attrs: { type: 'checkbox' } });
    notifyOnComplete.checked = this.settings.notifyOnComplete;
    notifyOnComplete.addEventListener('change', () =>
      this.callbacks.onSettingsChange({ notifyOnComplete: notifyOnComplete.checked }));

    const debugInfo = h('input', { attrs: { type: 'checkbox' } });
    debugInfo.checked = this.settings.debugInfo;
    debugInfo.addEventListener('change', () =>
      this.callbacks.onSettingsChange({ debugInfo: debugInfo.checked }));

    const serverInput = h('input', { className: 'ey-input' });
    serverInput.value = this.settings.serverUrl;
    serverInput.addEventListener('change', () => {
      const url = serverInput.value.trim().replace(/\/+$/, '');
      if (url) this.callbacks.onSettingsChange({ serverUrl: url });
    });
    // 점 색만으론 "왜 빨간지"를 알 수 없다 — 사유를 툴팁으로 붙이고, 인증 실패는 따로 표시
    const dot = h('span', { className: 'ey-dot', title: t('overlay.settings.serverStatusTitle', [statusLine(this.serverStatus)]) });
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
      attrs: { title: t('overlay.settings.permBtnTitle') },
      on: { click: () => this.callbacks.onOpenPermissions() },
    });
    this.settingsPermBtn = permBtn;
    permBtn.style.display = needsHostPermission(this.serverStatus) ? '' : 'none';

    const apiKeyInput = h('input', { className: 'ey-input', attrs: { type: 'password', placeholder: t('overlay.settings.apiKeyPlaceholder') } });
    apiKeyInput.value = this.settings.apiKey;
    apiKeyInput.addEventListener('change', () =>
      this.callbacks.onSettingsChange({ apiKey: apiKeyInput.value.trim() }));

    return h('div', { className: 'ey-settings' },
      h('div', { className: 'ey-settings-row' }, h('label', { text: t('overlay.settings.row.autoSearch'), attrs: { title: t('overlay.settings.row.autoSearchTitle') } }), autoSearch),
      h('div', { className: 'ey-settings-row' }, h('label', { text: t('overlay.settings.row.autoSearchShorts'), attrs: { title: t('overlay.settings.row.autoSearchShortsTitle') } }), autoSearchShorts),
      h('div', { className: 'ey-settings-row' }, h('label', { text: t('overlay.settings.row.fontSize') }), fontSelect),
      h('div', { className: 'ey-settings-row' }, h('label', { text: t('overlay.settings.row.theme') }), themeSelect),
      h('div', { className: 'ey-settings-row' }, h('label', { text: t('overlay.settings.row.showTranslation') }), showTranslation),
      h('div', { className: 'ey-settings-row' }, h('label', { text: t('overlay.settings.row.translationLanguage') }), langSelect),
      h('div', { className: 'ey-settings-row' }, h('label', { text: t('overlay.settings.row.uiLanguage') }), uiLangSelect),
      h('div', { className: 'ey-settings-row' }, h('label', { text: t('overlay.settings.row.pronScript') }), pronScriptSelect),
      h('div', { className: 'ey-settings-row' }, h('label', { text: t('overlay.settings.row.showPronunciation') }), showPronunciation),
      h('div', { className: 'ey-settings-row' }, h('label', { text: t('overlay.settings.row.sourcePriority') }), sourcePriority),
      h('div', { className: 'ey-settings-row' }, h('label', { text: t('overlay.settings.row.pipKeepPanel') }), pipKeepPanel),
      h('div', { className: 'ey-settings-row' }, h('label', { text: t('overlay.settings.row.pipShowVideo') }), pipShowVideo),
      h('div', { className: 'ey-settings-row' }, h('label', { text: t('overlay.settings.row.pitchGuide') }), pitchGuide),
      h('div', { className: 'ey-settings-row' }, h('label', { text: t('overlay.settings.row.pitchWindow') }), pitchWindow),
      h('div', { className: 'ey-settings-row' }, h('label', { text: t('overlay.settings.row.pitchMode') }), pitchMode),
      h('div', { className: 'ey-settings-row' }, h('label', { text: t('overlay.settings.row.pitchFont') }), pitchFont),
      h('div', { className: 'ey-settings-row' }, h('label', { text: t('overlay.settings.row.solfegeNotation') }), solfegeNotation),
      h('div', { className: 'ey-settings-row' }, h('label', { text: t('overlay.settings.row.pitchCountdown') }), pitchCountdown),
      h('div', { className: 'ey-settings-row' }, h('label', { text: t('overlay.settings.row.pitchF0Curve'), attrs: { title: t('overlay.settings.row.pitchF0CurveTitle') } }), pitchF0Curve),
      h('div', { className: 'ey-settings-row' }, h('label', { text: t('overlay.settings.row.pitchLineOpacity'), attrs: { title: t('overlay.settings.row.pitchLineOpacityTitle') } }), pitchLineOpacity),
      h('div', { className: 'ey-settings-row' }, h('label', { text: t('overlay.settings.row.pitchF0Opacity'), attrs: { title: t('overlay.settings.row.pitchF0OpacityTitle') } }), pitchF0Opacity),
      h('div', { className: 'ey-settings-row' }, h('label', { text: t('overlay.settings.row.pronPosition'), attrs: { title: t('overlay.settings.row.pronPositionTitle') } }), pitchPronPosition),
      h('div', { className: 'ey-settings-row' },
        h('label', { text: t('overlay.settings.row.melodyPlayback'), attrs: { title: t('overlay.settings.row.melodyPlaybackTitle') } }),
        h('span', { className: 'ey-settings-inline' }, melodyVolume, melodyPlayback)),
      h('div', { className: 'ey-settings-row' },
        h('label', { text: t('overlay.settings.row.metronome'), attrs: { title: t('overlay.settings.row.metronomeTitle') } }),
        h('span', { className: 'ey-settings-inline' }, metronomeVolume, metronome)),
      h('div', { className: 'ey-settings-row' },
        h('label', { text: t('overlay.settings.row.metronomeRate'), attrs: { title: t('overlay.settings.row.metronomeRateTitle') } }), metronomeRate),
      h('div', { className: 'ey-settings-row' },
        h('label', { text: t('overlay.settings.row.metronomeBeat'), attrs: { title: t('overlay.settings.row.metronomeBeatTitle') } }), metronomeBeat),
      h('div', { className: 'ey-settings-row' },
        h('label', { text: t('overlay.settings.row.audioOut'), attrs: { title: t('overlay.settings.row.audioOutTitle') } }), audioOut),
      h('div', { className: 'ey-settings-row' },
        h('label', { text: t('overlay.settings.row.micPitch'), attrs: { title: t('overlay.settings.row.micPitchTitle') } }), micPitch),
      h('div', { className: 'ey-settings-row' }, h('label', { text: t('overlay.settings.row.micDevice') }), micDevice),
      h('div', { className: 'ey-settings-row' },
        h('label', { text: t('overlay.settings.row.micOctave'), attrs: { title: t('overlay.settings.row.micOctaveTitle') } }), micOctave),
      h('div', { className: 'ey-settings-note', text: t('overlay.settings.deviceNote') }),
      h('div', { className: 'ey-settings-row ey-settings-col' },
        h('label', {}, t('overlay.settings.serverUrlLabel'), dot),
        serverInput,
      ),
      h('div', { className: 'ey-settings-row ey-settings-col' },
        h('label', { text: t('overlay.settings.apiKeyLabel') }),
        apiKeyInput,
      ),
      serverNote,
      permBtn,
      h('div', { className: 'ey-settings-row' },
        h('label', { text: t('overlay.settings.row.lowConfWarning'), attrs: { title: t('overlay.settings.row.lowConfWarningTitle') } }), lowConfWarning),
      h('div', { className: 'ey-settings-row' },
        h('label', { text: t('overlay.settings.row.notifyOnComplete'), attrs: { title: t('overlay.settings.row.notifyOnCompleteTitle') } }), notifyOnComplete),
      h('div', { className: 'ey-settings-row' }, h('label', { text: t('overlay.settings.row.debugInfo') }), debugInfo),
      h('div', { className: 'ey-settings-note', text: t('overlay.settings.serverRequiredNote') }),
      h('button', { className: 'ey-secondary-btn', text: t('overlay.settings.closeButton'), on: { click: () => this.closeSettings() } }),
    );
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

  /** min/max는 buildRange의 출력 스케일 기준 소수(예: 0.2~1) — 생략하면 기존 호출부와
   *  똑같이 0~1 전체 범위(K3 이전부터 있던 볼륨류 슬라이더는 인자를 안 넘기므로 그대로다). */
  private buildRange(
    value: number, onChange: (v: number) => void, min = 0, max = 1,
  ): HTMLInputElement {
    const range = h('input', {
      className: 'ey-settings-range',
      attrs: {
        type: 'range', min: String(Math.round(min * 100)), max: String(Math.round(max * 100)), step: '1',
        value: String(Math.round(value * 100)),
      },
    });
    range.addEventListener('change', () => onChange(Number(range.value) / 100));
    return range;
  }

  private buildSelect(options: [string, string][], value: string, onChange: (v: string) => void): HTMLSelectElement {
    const select = h('select', { className: 'ey-select' });
    for (const [v, label] of options) {
      const opt = h('option', { text: label, attrs: { value: v } });
      select.append(opt);
    }
    select.value = value;
    select.addEventListener('change', () => onChange(select.value));
    return select;
  }

  // ── 위치/크기 ─────────────────────────────────────────────────

  private defaultGeometry(): PanelGeometry {
    return {
      x: Math.max(EDGE_MARGIN, window.innerWidth - DEFAULT_WIDTH - 24),
      y: 72,
      width: DEFAULT_WIDTH,
      height: Math.min(DEFAULT_HEIGHT, Math.round(window.innerHeight * 0.7)),
      collapsed: false,
    };
  }

  private applyGeometry(): void {
    this.applyingGeometry = true;
    const g = this.geometry;
    this.panel.style.left = `${g.x}px`;
    this.panel.style.top = `${g.y}px`;
    this.panel.style.width = `${g.width}px`;
    this.panel.classList.toggle('collapsed', g.collapsed);
    this.panel.style.height = g.collapsed ? 'auto' : `${g.height}px`;
    this.collapseBtn.replaceChildren(icon(g.collapsed ? ICONS.expand : ICONS.collapse));
    this.collapseBtn.title = g.collapsed ? t('overlay.header.expand') : t('overlay.header.collapse');
    requestAnimationFrame(() => {
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
    if (this.applyingGeometry || this.geometry.collapsed) return;
    const { offsetWidth, offsetHeight } = this.panel;
    if (offsetWidth === this.geometry.width && offsetHeight === this.geometry.height) return;
    this.geometry.width = offsetWidth;
    this.geometry.height = offsetHeight;
    this.scheduleGeometrySave();
  }

  private handleWindowResize = (): void => {
    this.geometry.x = this.clampX(this.geometry.x);
    this.geometry.y = this.clampY(this.geometry.y);
    this.panel.style.left = `${this.geometry.x}px`;
    this.panel.style.top = `${this.geometry.y}px`;
  };

  private handleFullscreenChange = (): void => {
    const wasHidden = this.fullscreenHidden;
    this.fullscreenHidden = document.fullscreenElement !== null;
    this.updateHostVisibility();
    // 전체화면 동안 도착해 못 본 알림을 해제 직후 다시 띄운다 — 타이머도 여기서 처음부터.
    // 칩은 이미 DOM에 그려져 있으므로 문구가 바뀌지 않고 표시 시간만 새로 주어진다.
    if (wasHidden && !this.fullscreenHidden && this.pendingNotice) {
      const { text, autoHideMs } = this.pendingNotice;
      this.setNoticeChip(text, autoHideMs);
    }
  };

  private clampX(x: number): number {
    return Math.min(Math.max(x, EDGE_MARGIN), Math.max(EDGE_MARGIN, window.innerWidth - this.geometry.width - EDGE_MARGIN));
  }

  private clampY(y: number): number {
    return Math.min(Math.max(y, EDGE_MARGIN), Math.max(EDGE_MARGIN, window.innerHeight - 48));
  }

  private updateHostVisibility(): void {
    this.host.style.display = this.visible && !this.fullscreenHidden ? '' : 'none';
  }

  private scheduleGeometrySave(): void {
    clearTimeout(this.saveGeomTimer);
    this.saveGeomTimer = window.setTimeout(() => {
      this.callbacks.onGeometryChange({ ...this.geometry });
    }, 400);
  }
}
