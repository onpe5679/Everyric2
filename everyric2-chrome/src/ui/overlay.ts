import type { DebugInfo, LyricLine, LyricsSource, PanelGeometry, SearchCandidate, Settings, SongInfo } from '../types';
import { h, icon, ICONS } from './dom';
import { appendKaraokeSpans, appendTimedSpans } from './karaoke';

export interface OverlayCallbacks {
  onSeek: (time: number) => void;
  onGenerate: (lyrics: string) => void;
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
  /** 다른 영상의 싱크에 연결 (inst·커버) — 성공 시 content가 재조회한다 */
  onLinkSync: (sourceVideoId: string, offsetSec: number) => void;
  /** 현재 영상의 싱크 링크 해제 */
  onUnlinkSync: () => void;
  /** 서버 저장 싱크 목록 요청 — 결과는 showSyncList로 되돌아온다 */
  onRequestSyncList: () => void;
}

type StateKind = 'loading' | 'synced' | 'plain' | 'empty' | 'generating' | 'error' | 'pip' | 'search';

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

export class LyricsOverlay {
  private host: HTMLDivElement;
  private panel: HTMLDivElement;
  private header: HTMLDivElement;
  private songTitleEl: HTMLDivElement;
  private songArtistEl: HTMLDivElement;
  private body: HTMLDivElement;
  private footer: HTMLDivElement;
  private debugEl: HTMLDivElement;
  private banner: HTMLDivElement;
  private resumeChip: HTMLButtonElement;
  private genChip: HTMLDivElement;
  private pipBtn: HTMLButtonElement;
  private regenBtn: HTMLButtonElement;
  private collapseBtn: HTMLButtonElement;
  private settingsSheet: HTMLDivElement | null = null;
  private settingsDot: HTMLSpanElement | null = null;
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
  private userScrollUntil = 0;
  private offsetSec: number;
  private visible = true;
  private fullscreenHidden = false;
  private serverAvailable = false;
  private generateButtons: HTMLButtonElement[] = [];
  private plainTextForGenerate = '';
  private pipEnabled = false;
  private sourceUrl: string | null = null;
  private attributionName: string | null = null;
  private lastSong: SongInfo | null = null;
  private searchResultsEl: HTMLDivElement | null = null;
  private linkListEl: HTMLDivElement | null = null;
  private linkSrcInput: HTMLInputElement | null = null;
  /** 현재 표시 중인 싱크의 링크 상태 (없으면 null) — content가 setLinked로 밀어넣는다 */
  private linkedInfo: { sourceVideoId: string; offsetSec: number } | null = null;

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

    this.songTitleEl = h('div', { className: 'ey-song-title', text: '노래 인식 중…' });
    this.songArtistEl = h('div', { className: 'ey-song-artist' });

    this.pipBtn = this.headerButton(ICONS.pip, 'PiP 창으로 보기', () => this.callbacks.onPipToggle());
    this.pipBtn.style.display = 'none';
    this.regenBtn = this.headerButton(ICONS.refresh, '싱크 다시 생성 (서버 캐시 무시)', () => {
      if (window.confirm('서버 싱크를 다시 생성할까요?\n1분 정도 걸리고, 완료될 때까지 기존 가사는 계속 표시됩니다.')) {
        this.callbacks.onRegenerate();
      }
    });
    this.regenBtn.style.display = 'none';
    const searchBtn = this.headerButton(ICONS.search, '가사 다시 검색 (다른 결과 선택)', () => this.openSearch());
    const gearBtn = this.headerButton(ICONS.gear, '설정', () => this.toggleSettings());
    this.collapseBtn = this.headerButton(ICONS.collapse, '접기', () => this.setCollapsed(!this.geometry.collapsed));
    const closeBtn = this.headerButton(ICONS.close, '닫기 (툴바 아이콘으로 다시 열기)', () => this.setVisible(false));

    this.header = h('div', { className: 'ey-header' },
      h('div', { className: 'ey-header-left' },
        icon(ICONS.note),
        h('div', { className: 'ey-song' }, this.songTitleEl, this.songArtistEl),
      ),
      h('div', { className: 'ey-actions' }, this.pipBtn, this.regenBtn, searchBtn, gearBtn, this.collapseBtn, closeBtn),
    );

    this.banner = h('div', { className: 'ey-banner' });
    this.banner.style.display = 'none';

    // 전사 진행 칩 — 패널을 점유하지 않고 헤더 밑에 작게 진행률만 보여준다
    this.genChip = h('div', { className: 'ey-gen-chip' }, icon(ICONS.sparkle), '');
    this.genChip.style.display = 'none';

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
    }, icon(ICONS.down), '현재 가사로');
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
    this.footer = h('div', { className: 'ey-footer' },
      this.sourceBadge,
      this.trStatusEl,
      h('div', { className: 'ey-offset' },
        h('span', { className: 'ey-offset-caption', text: '싱크' }),
        this.footerButton('−0.1', '가사를 0.1초 당기기', () => this.changeOffset(-0.1)),
        this.offsetLabel,
        this.footerButton('+0.1', '가사를 0.1초 늦추기', () => this.changeOffset(0.1)),
        this.footerButton('리셋', '오프셋 초기화', () => this.changeOffset(null)),
      ),
    );
    this.footer.style.display = 'none';

    this.debugEl = h('div', { className: 'ey-debug', text: 'debug: 대기 중…' });
    this.debugEl.style.display = 'none';

    this.panel = h('div', { className: 'ey-panel' },
      this.header, this.banner, this.genChip, this.body, this.resumeChip, this.footer, this.debugEl,
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

  showLoading(message = '가사 검색 중…'): void {
    this.stateKind = 'loading';
    this.resetBody();
    const skeleton = h('div', { className: 'ey-skeleton' });
    for (let i = 0; i < 3; i++) skeleton.append(h('div', { className: 'ey-skeleton-bar' }));
    this.body.append(
      h('div', { className: 'ey-state' },
        skeleton,
        h('div', { className: 'ey-state-text', text: message }),
        // 자동 검색을 기다릴 필요 없이 바로 수동 검색으로 전환
        h('button', {
          className: 'ey-secondary-btn',
          text: '기다리지 않고 수동 검색',
          on: { click: () => this.openSearch() },
        }),
      ),
    );
  }

  showSyncedLyrics(lines: LyricLine[], source: LyricsSource): void {
    this.stateKind = 'synced';
    this.resetBody();
    this.lines = lines;
    this.lineEls = [];
    this.currentIndex = -1;

    const list = h('div', { className: 'ey-lines' });
    lines.forEach((line, index) => {
      const el = h('div', {
        className: 'ey-line',
        title: '클릭해서 이 부분으로 이동',
        on: {
          click: () => {
            if (line.time !== null) this.callbacks.onSeek(line.time);
          },
        },
      });
      if (line.words && line.words.length > 0) {
        appendKaraokeSpans(el, line, word => {
          // 신뢰도 등급 클래스 — .ey-show-conf(디버그 모드)에서만 색이 입혀진다.
          // 값은 CTC 프레임 로그확률의 기하평균(0~1) — 절대값이 작아 로그 스케일로 버킷:
          // <1e-4(로그 -9 이하)=낮음, <2e-2(로그 -4 이하)=중간
          const conf = word.confidence;
          // 버킷 색은 레인(pip.ts confBucketColor)과 동일: 빨강<1e-4, 노랑<2e-2, 초록=양호
          const confClass = conf == null ? '' : conf < 1e-4 ? ' ey-conf-low' : conf < 2e-2 ? ' ey-conf-mid' : ' ey-conf-ok';
          return h('span', { className: `ey-word${confClass}`, text: word.word, attrs: { 'data-start': String(word.start) } });
        });
      } else {
        el.textContent = line.text;
      }
      if (line.pronunciation) {
        // 음절 타이밍(pronSegments)이 있으면 단어처럼 부른 만큼 색이 차오르게 스팬으로
        // (사이 텍스트는 appendTimedSpans가 인접 span에 끼워 넣어 흰 글자 없이 칠해진다)
        const segs = line.pronSegments;
        const pronEl = h('div', { className: 'ey-line-pron' });
        const mapped = segs && segs.length > 0
          ? appendTimedSpans(pronEl, line.pronunciation, segs, s => s.text, seg =>
              h('span', {
                className: 'ey-pron-syl',
                text: seg.text,
                attrs: { 'data-start': String(seg.start) },
              }))
          : 0;
        if (mapped === 0) pronEl.replaceChildren(line.pronunciation);
        el.append(pronEl);
      }
      if (line.translation) el.append(h('div', { className: 'ey-line-tr', text: line.translation }));
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
  }

  showPlainLyrics(lines: LyricLine[], source: LyricsSource, plainText: string): void {
    this.stateKind = 'plain';
    this.resetBody();
    this.plainTextForGenerate = plainText;

    const generateBtn = this.makeGenerateButton('싱크 생성', () => this.callbacks.onGenerate(this.plainTextForGenerate));
    this.showBanner('타임싱크가 없는 가사예요', generateBtn);

    this.lines = lines;
    const list = h('div', { className: 'ey-lines ey-lines-plain' });
    for (const line of lines) {
      const el = h('div', { className: 'ey-line ey-line-plain', text: line.text });
      if (line.pronunciation) el.append(h('div', { className: 'ey-line-pron', text: line.pronunciation }));
      if (line.translation) el.append(h('div', { className: 'ey-line-tr', text: line.translation }));
      this.lineEls.push(el);
      list.append(el);
    }
    this.body.append(list);

    this.setSourceBadge(source, false);
    this.footer.classList.add('no-offset');
    this.footer.style.display = '';
  }

  showEmpty(song: SongInfo | null): void {
    this.stateKind = 'empty';
    this.resetBody();

    const titleInput = h('input', { className: 'ey-input', attrs: { placeholder: '곡 제목' } });
    titleInput.value = song?.title ?? '';
    const artistInput = h('input', { className: 'ey-input', attrs: { placeholder: '아티스트 (선택)' } });
    artistInput.value = song?.artist ?? '';

    this.body.append(
      h('div', { className: 'ey-state' },
        h('div', { className: 'ey-state-emoji', text: '🎵' }),
        h('div', { className: 'ey-state-text', text: '가사를 찾지 못했어요' }),
        h('div', { className: 'ey-search-form' },
          titleInput,
          artistInput,
          h('button', {
            className: 'ey-primary-btn',
            text: '다시 검색',
            on: {
              click: () => {
                const title = titleInput.value.trim();
                if (title) this.callbacks.onRetrySearch({ title, artist: artistInput.value.trim() });
              },
            },
          }),
        ),
        h('button', {
          className: 'ey-secondary-btn',
          text: '상세 검색 (후보 선택·싱크 연결)',
          on: { click: () => this.openSearch() },
        }),
        h('div', { className: 'ey-divider' }),
        this.buildPasteSection(true),
      ),
    );
  }

  /** 가사 직접 붙여넣기 섹션 (토글 접힘) — 빈 상태·검색 시트 공용 */
  private buildPasteSection(startOpen = false): HTMLDivElement {
    const lyricsArea = h('textarea', {
      className: 'ey-textarea',
      attrs: { placeholder: '여기에 가사를 붙여넣으면 AI가 타이밍을 맞춰줘요', rows: '6' },
    });
    const pasteSection = h('div', { className: 'ey-paste-section' },
      lyricsArea,
      this.makeGenerateButton('붙여넣은 가사로 싱크 생성', () => {
        const text = lyricsArea.value.trim();
        if (text) this.callbacks.onGenerate(text);
      }),
    );
    pasteSection.style.display = startOpen ? '' : 'none';

    const pasteToggle = h('button', {
      className: 'ey-secondary-btn',
      text: startOpen ? '붙여넣기 닫기' : '가사 직접 붙여넣기',
      on: {
        click: () => {
          const hidden = pasteSection.style.display === 'none';
          pasteSection.style.display = hidden ? '' : 'none';
          pasteToggle.textContent = hidden ? '붙여넣기 닫기' : '가사 직접 붙여넣기';
        },
      },
    });
    return h('div', { className: 'ey-paste-wrap' }, pasteToggle, pasteSection);
  }

  /** 상시 재검색: 현재 곡 정보를 초기값으로 검색 폼 + 소스별 후보 리스트를 연다 */
  openSearch(): void {
    this.stateKind = 'search';
    this.resetBody();

    const titleInput = h('input', { className: 'ey-input', attrs: { placeholder: '곡 제목' } });
    titleInput.value = this.lastSong?.title ?? '';
    const artistInput = h('input', { className: 'ey-input', attrs: { placeholder: '아티스트 (선택)' } });
    artistInput.value = this.lastSong?.artist ?? '';

    this.searchResultsEl = h('div', { className: 'ey-result-list' });
    const doSearch = () => {
      const title = titleInput.value.trim();
      if (!title) return;
      this.setSearchStatus('검색 중…');
      this.callbacks.onCandidateSearch({ title, artist: artistInput.value.trim() });
    };
    titleInput.addEventListener('keydown', e => { if (e.key === 'Enter') doSearch(); });
    artistInput.addEventListener('keydown', e => { if (e.key === 'Enter') doSearch(); });

    this.body.append(
      h('div', { className: 'ey-state ey-search-state' },
        h('div', { className: 'ey-state-text', text: '가사 검색 — 결과에서 직접 선택할 수 있어요' }),
        h('div', { className: 'ey-state-sub', text: '보카로 위키(발음·번역 포함) + LRCLIB(싱크 가사)를 함께 검색합니다' }),
        h('div', { className: 'ey-search-form' },
          titleInput,
          artistInput,
          h('button', { className: 'ey-primary-btn', text: '검색', on: { click: doSearch } }),
        ),
        this.searchResultsEl,
        h('div', { className: 'ey-divider' }),
        this.buildPasteSection(),
        h('div', { className: 'ey-divider' }),
        this.buildLinkSection(),
        h('div', { className: 'ey-divider' }),
        h('button', {
          className: 'ey-secondary-btn',
          text: '자동 검색으로 되돌리기',
          on: { click: () => this.callbacks.onRetrySearch() },
        }),
      ),
    );
    if (titleInput.value) doSearch();
  }

  /** 다른 영상 싱크 연결 섹션 (inst·커버 영상용) — 검색 시트 하단 */
  private buildLinkSection(): HTMLDivElement {
    const srcInput = h('input', {
      className: 'ey-input',
      attrs: { placeholder: '원본 영상 URL 또는 ID (전사가 이미 있는 영상)' },
    });
    this.linkSrcInput = srcInput;
    const offsetInput = h('input', {
      className: 'ey-input ey-input-narrow',
      attrs: { type: 'number', step: '0.1', placeholder: '오프셋(초)', title: '이 영상이 원본보다 늦게 시작하면 +, 빠르면 -' },
    });
    offsetInput.value = this.linkedInfo ? String(this.linkedInfo.offsetSec) : '0';
    this.linkListEl = h('div', { className: 'ey-result-list' });

    const doLink = () => {
      const src = parseVideoId(srcInput.value.trim());
      if (!src) {
        this.setLinkStatus('영상 URL 또는 11자리 ID를 입력해 주세요');
        return;
      }
      const offset = Number(offsetInput.value) || 0;
      this.setLinkStatus('연결 중…');
      this.callbacks.onLinkSync(src, offset);
    };

    const section = h('div', { className: 'ey-link-section' },
      h('div', { className: 'ey-state-text', text: '다른 영상의 싱크 연결 (inst·커버용)' }),
    );
    if (this.linkedInfo) {
      section.append(h('div', { className: 'ey-link-current' },
        h('span', {
          text: `현재 ${this.linkedInfo.sourceVideoId}에 연결됨 (${this.linkedInfo.offsetSec >= 0 ? '+' : ''}${this.linkedInfo.offsetSec}s)`,
        }),
        h('button', {
          className: 'ey-secondary-btn',
          text: '링크 해제',
          on: { click: () => this.callbacks.onUnlinkSync() },
        }),
      ));
    }
    section.append(
      h('div', { className: 'ey-search-form' },
        srcInput,
        offsetInput,
        h('button', { className: 'ey-primary-btn', text: '연결', on: { click: doLink } }),
      ),
      h('button', {
        className: 'ey-secondary-btn',
        text: '서버에 저장된 싱크 목록에서 고르기',
        on: {
          click: () => {
            this.setLinkStatus('목록 불러오는 중…');
            this.callbacks.onRequestSyncList();
          },
        },
      }),
      this.linkListEl,
    );
    return section;
  }

  /** SYNC_LIST 응답 반영 — 항목 클릭 시 원본 입력칸에 채워 넣는다 */
  showSyncList(items: { video_id: string; first_line: string; line_count: number; attribution_name?: string | null; alignment_text?: string | null }[]): void {
    if (this.stateKind !== 'search' || !this.linkListEl) return;
    if (items.length === 0) {
      this.setLinkStatus('서버에 저장된 싱크가 없어요');
      return;
    }
    this.linkListEl.replaceChildren(...items.map(it =>
      h('button', {
        className: 'ey-result-item',
        on: {
          click: () => {
            if (this.linkSrcInput) this.linkSrcInput.value = it.video_id;
          },
        },
      },
        h('span', { className: 'ey-result-src', text: it.video_id }),
        h('span', { className: 'ey-result-title', text: it.first_line || '(첫 줄 없음)' }),
        h('span', { className: 'ey-result-meta', text: `${it.line_count}줄${it.attribution_name ? ' · ' + it.attribution_name : ''}` }),
      )));
  }

  /** 링크 섹션 상태 메시지 (검색 상태가 아니면 무시) */
  setLinkStatus(message: string): void {
    this.linkListEl?.replaceChildren(h('div', { className: 'ey-state-sub', text: message }));
  }

  /** 현재 싱크의 링크 상태 — 검색 시트의 해제 UI와 출처 배지에 반영 */
  setLinked(info: { sourceVideoId: string; offsetSec: number } | null): void {
    this.linkedInfo = info;
  }

  /** SEARCH_CANDIDATES 응답 반영 — 검색 상태가 아니면 무시 (stale 응답 방지) */
  showSearchResults(candidates: SearchCandidate[]): void {
    if (this.stateKind !== 'search' || !this.searchResultsEl) return;
    if (candidates.length === 0) {
      this.setSearchStatus('결과가 없어요 — 제목을 줄이거나 원제(일본어)로 시도해 보세요');
      return;
    }
    const fmt = (sec: number) => `${Math.floor(sec / 60)}:${String(Math.round(sec % 60)).padStart(2, '0')}`;
    this.searchResultsEl.replaceChildren(...candidates.map(c => {
      const isWiki = c.source === 'vocaro';
      const label = isWiki ? c.title : `${c.title}${c.artist ? ' — ' + c.artist : ''}`;
      const meta = isWiki
        ? '발음·번역'
        : `${c.synced ? '싱크' : '일반'}${c.duration > 0 ? ` · ${fmt(c.duration)}` : ''}`;
      const btn = h('button', {
        className: 'ey-result-item',
        on: { click: () => this.callbacks.onPickCandidate(c) },
      },
        h('span', { className: `ey-result-src${isWiki ? ' vocaro' : ''}`, text: isWiki ? '보카로 위키' : 'LRCLIB' }),
        h('span', { className: 'ey-result-title', text: label }),
        h('span', { className: 'ey-result-meta', text: meta }),
      );
      btn.title = isWiki ? c.url : label;
      return btn;
    }));
  }

  private setSearchStatus(message: string): void {
    this.searchResultsEl?.replaceChildren(h('div', { className: 'ey-state-sub', text: message }));
  }

  showGenerating(progress: number, label?: string): void {
    const pct = Math.max(0, Math.min(100, Math.round(progress)));
    const text = label ?? `싱크 생성 중… ${pct}%`;
    if (this.stateKind === 'generating' && this.progressBar && this.progressText) {
      this.progressBar.style.width = `${pct}%`;
      this.progressText.textContent = text;
      return;
    }
    this.stateKind = 'generating';
    this.resetBody();
    this.progressBar = h('div', { className: 'ey-progress-bar' });
    this.progressBar.style.width = `${pct}%`;
    this.progressText = h('div', { className: 'ey-state-text', text });
    this.body.append(
      h('div', { className: 'ey-state' },
        h('div', { className: 'ey-state-emoji', text: '✨' }),
        this.progressText,
        h('div', { className: 'ey-progress' }, this.progressBar),
        h('div', { className: 'ey-state-sub', text: '계속 시청하셔도 돼요. 완료되면 자동으로 표시됩니다.' }),
      ),
    );
  }

  showError(message: string): void {
    this.stateKind = 'error';
    this.resetBody();
    this.body.append(
      h('div', { className: 'ey-state' },
        h('div', { className: 'ey-state-emoji', text: '⚠️' }),
        h('div', { className: 'ey-state-text', text: message }),
        h('button', { className: 'ey-primary-btn', text: '다시 시도', on: { click: () => this.callbacks.onRetrySearch() } }),
      ),
    );
  }

  showPipPlaceholder(): void {
    this.stateKind = 'pip';
    this.resetBody();
    this.body.append(
      h('div', { className: 'ey-state' },
        h('div', { className: 'ey-state-emoji', text: '🪟' }),
        h('div', { className: 'ey-state-text', text: 'PiP 창에서 가사를 표시하고 있어요' }),
        h('button', { className: 'ey-primary-btn', text: '패널로 되돌리기', on: { click: () => this.callbacks.onPipToggle() } }),
      ),
    );
  }

  // ── 싱크 업데이트 ──────────────────────────────────────────────

  highlightLine(index: number): void {
    if (this.stateKind !== 'synced') return;
    this.currentIndex = index;
    this.activeWordEls = [];
    this.lineEls.forEach((el, i) => {
      el.classList.toggle('active', i === index);
      el.classList.toggle('past', index >= 0 && i < index);
    });
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

  // ── 외부 상태 주입 ─────────────────────────────────────────────

  setSong(song: SongInfo | null): void {
    this.lastSong = song;
    if (song) {
      this.songTitleEl.textContent = song.title;
      this.songTitleEl.title = song.title;
      this.songArtistEl.textContent = song.artist ?? '';
    } else {
      this.songTitleEl.textContent = '노래 인식 중…';
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

  setServerAvailable(available: boolean): void {
    this.serverAvailable = available;
    this.generateButtons = this.generateButtons.filter(btn => btn.isConnected);
    for (const btn of this.generateButtons) {
      btn.disabled = !available;
      btn.title = available ? '' : 'Everyric 서버에 연결할 수 없어요 (설정에서 서버 URL 확인)';
    }
    this.settingsDot?.classList.toggle('ok', available);
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

  /** lines[].translation을 다시 읽어 각 라인 아래 번역을 갱신/제거한다 */
  refreshTranslations(): void {
    this.lineEls.forEach((el, i) => {
      el.querySelector('.ey-line-tr')?.remove();
      const translation = this.lines[i]?.translation;
      if (translation) el.append(h('div', { className: 'ey-line-tr', text: translation }));
    });
  }

  setTranslationStatus(text: string | null): void {
    this.trStatusEl.textContent = text ?? '';
  }

  /** 전사 진행 칩 — null이면 숨김. 패널 본문을 점유하지 않는 작은 표시. */
  setGenerationChip(text: string | null): void {
    if (!text) {
      this.genChip.style.display = 'none';
      return;
    }
    this.genChip.replaceChildren(icon(ICONS.sparkle), text);
    this.genChip.style.display = '';
  }

  updateDebug(info: DebugInfo): void {
    if (!this.settings.debugInfo) return;
    const t = info.time === null ? '-' : `${info.time.toFixed(2)}s`;
    const off = `${info.offsetSec > 0 ? '+' : ''}${info.offsetSec.toFixed(1)}`;
    const line = info.lineCount > 0 ? `${info.lineIndex + 1}/${info.lineCount}` : '-';
    const video = info.videoInfo === 'none' ? 'none' : `${info.videoBound ? 'OK' : 'MISMATCH'}(${info.videoInfo})`;
    const g = info.confGrades;
    const diag = [
      // 사람이 읽는 등급 분포 (글자 색과 동일 기준: 좋음=초록, 보통=노랑, 낮음=빨강)
      g ? `정렬 좋음${Math.round(g.ok * 100)}%·보통${Math.round(g.mid * 100)}%·낮음${Math.round(g.low * 100)}%` : null,
      info.quality != null ? `conf=${info.quality.toExponential(1)}` : null,
      info.qualityMed != null ? `med=${info.qualityMed.toExponential(1)}` : null,
      info.alignmentText ? `전사텍스트=${info.alignmentText === 'pronunciation' ? '독음(한국어 발음)' : '원문(원어)'}` : null,
      info.zone ? `zone=${info.zone}` : null,
      info.lineDebug,
    ].filter(Boolean).join(' ');
    this.debugEl.textContent =
      `vid=${info.videoId ?? '-'} src=${info.source}${info.synced ? '/sync' : '/plain'} line=${line} pip=${info.pipOpen ? 'Y' : 'N'}\n`
      + `t=${t} off=${off} video=${video} eng=${info.engineRunning ? 'Y' : 'N'}${info.jobStatus ? ` ${info.jobStatus}` : ''}`
      + (diag ? `\n${diag}` : '');
  }

  applySettings(settings: Settings): void {
    this.settings = settings;
    this.panel.classList.remove('ey-fs-small', 'ey-fs-medium', 'ey-fs-large');
    this.panel.classList.add(`ey-fs-${settings.fontSize}`);
    const light = this.resolveTheme(settings) === 'light';
    this.panel.classList.toggle('ey-light', light);
    if (this.offsetSec !== settings.offsetSec) {
      this.offsetSec = settings.offsetSec;
      this.updateOffsetLabel();
    }
    this.debugEl.style.display = settings.debugInfo ? '' : 'none';
    this.panel.classList.toggle('ey-hide-pron', !settings.showPronunciation);
    // 디버그 모드에서 글자별 CTC 신뢰도를 색으로 표시
    this.panel.classList.toggle('ey-show-conf', settings.debugInfo);
  }

  // ── 내부 헬퍼 ─────────────────────────────────────────────────

  private resolveTheme(settings: Settings): 'dark' | 'light' {
    if (settings.theme !== 'auto') return settings.theme;
    if (location.host === 'music.youtube.com') return 'dark';
    return document.documentElement.hasAttribute('dark') ? 'dark' : 'light';
  }

  private headerButton(svg: string, title: string, onClick: () => void): HTMLButtonElement {
    return h('button', { className: 'ey-btn', title, on: { click: onClick } }, icon(svg));
  }

  private footerButton(text: string, title: string, onClick: () => void): HTMLButtonElement {
    return h('button', { className: 'ey-offset-btn', text, title, on: { click: onClick } });
  }

  private makeGenerateButton(label: string, onClick: () => void): HTMLButtonElement {
    const btn = h('button', { className: 'ey-primary-btn ey-generate-btn', on: { click: onClick } },
      icon(ICONS.sparkle), label);
    btn.disabled = !this.serverAvailable;
    if (!this.serverAvailable) btn.title = 'Everyric 서버에 연결할 수 없어요 (설정에서 서버 URL 확인)';
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
    this.lines = [];
    this.lineEls = [];
    this.activeWordEls = [];
    this.currentIndex = -1;
    this.userScrollUntil = 0;
    this.progressBar = null;
    this.progressText = null;
    this.searchResultsEl = null;
    this.closeSettings();
  }

  private showBanner(text: string, action?: HTMLElement): void {
    this.banner.replaceChildren(h('span', { className: 'ey-banner-text', text }));
    if (action) this.banner.append(action);
    this.banner.style.display = '';
  }

  private setSourceBadge(source: LyricsSource, synced: boolean): void {
    const base = source === 'everyric' ? 'Everyric' : source === 'vocaro' ? '보카로 가사 위키' : 'LRCLIB';
    // 가사 원출처(위키 등)를 병기 — 전사는 서버가 했어도 가사의 출처는 따로 표기
    const extra = this.attributionName && this.attributionName !== base ? ` · ${this.attributionName}` : '';
    // 다른 영상의 싱크를 빌려온 경우 링크 표시 (해제는 검색 시트에서)
    const link = this.linkedInfo
      ? ` · 🔗${this.linkedInfo.offsetSec !== 0 ? `${this.linkedInfo.offsetSec > 0 ? '+' : ''}${this.linkedInfo.offsetSec}s` : ''}`
      : '';
    this.sourceBadge.textContent = base + extra + link;
    // 출처 상세: 무엇을 어디서 가져왔는지 — 클릭 전에 툴팁으로도 확인 가능
    const kind = synced ? '싱크 가사' : '일반 가사';
    this.sourceBadge.title = this.sourceUrl ? `${kind} · 출처 페이지 열기\n${this.sourceUrl}` : kind;
    this.sourceBadge.classList.toggle('everyric', source === 'everyric');
  }

  /** 가사 원출처 표기 (이름+링크). show* 호출 전에 설정해야 배지에 반영된다. */
  setAttribution(attr: { name: string; url?: string | null } | null): void {
    this.attributionName = attr?.name ?? null;
    this.setSourceUrl(attr?.url ?? null);
  }

  /** 출처 페이지 링크 (보카로 위키 등 CC BY 출처 표기) — null이면 배지는 단순 라벨 */
  setSourceUrl(url: string | null): void {
    this.sourceUrl = url;
    this.sourceBadge.classList.toggle('link', url !== null);
    if (url) this.sourceBadge.title = '출처 페이지 열기';
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
    const sheet = this.buildSettingsSheet();
    this.settingsSheet = sheet;
    this.panel.append(sheet);
  }

  private closeSettings(): void {
    this.settingsSheet?.remove();
    this.settingsSheet = null;
    this.settingsDot = null;
  }

  private buildSettingsSheet(): HTMLDivElement {
    const autoSearch = h('input', { attrs: { type: 'checkbox' } });
    autoSearch.checked = this.settings.autoSearch;
    autoSearch.addEventListener('change', () => this.callbacks.onSettingsChange({ autoSearch: autoSearch.checked }));

    const fontSelect = this.buildSelect(
      [['small', '작게'], ['medium', '보통'], ['large', '크게']],
      this.settings.fontSize,
      v => this.callbacks.onSettingsChange({ fontSize: v as Settings['fontSize'] }),
    );
    const themeSelect = this.buildSelect(
      [['auto', '자동'], ['dark', '다크'], ['light', '라이트']],
      this.settings.theme,
      v => this.callbacks.onSettingsChange({ theme: v as Settings['theme'] }),
    );

    const showTranslation = h('input', { attrs: { type: 'checkbox' } });
    showTranslation.checked = this.settings.showTranslation;
    showTranslation.addEventListener('change', () =>
      this.callbacks.onSettingsChange({ showTranslation: showTranslation.checked }));

    const langSelect = this.buildSelect(
      [['ko', '한국어'], ['en', 'English'], ['ja', '日本語'], ['zh', '中文']],
      this.settings.translationLanguage,
      v => this.callbacks.onSettingsChange({ translationLanguage: v }),
    );

    const showPronunciation = h('input', { attrs: { type: 'checkbox' } });
    showPronunciation.checked = this.settings.showPronunciation;
    showPronunciation.addEventListener('change', () =>
      this.callbacks.onSettingsChange({ showPronunciation: showPronunciation.checked }));

    const sourcePriority = this.buildSelect(
      [['vocaro', '보카로 위키 우선'], ['lrclib', 'LRCLIB 우선']],
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
      [['0.5', '½마디'], ['1', '1마디'], ['2', '2마디'], ['4', '4마디'], ['8', '8마디']],
      String(this.settings.pitchWindowMeasures),
      v => this.callbacks.onSettingsChange({ pitchWindowMeasures: Number(v) }),
    );

    const pitchMode = this.buildSelect(
      [['page', '고정 화면·헤드 이동'], ['scroll', '스크롤·헤드 고정']],
      this.settings.pitchScrollMode,
      v => this.callbacks.onSettingsChange({ pitchScrollMode: v as Settings['pitchScrollMode'] }),
    );

    const pitchFont = this.buildSelect(
      [['1', '보통'], ['1.2', '크게'], ['1.45', '아주 크게'], ['0.85', '작게']],
      String(this.settings.pitchFontScale),
      v => this.callbacks.onSettingsChange({ pitchFontScale: Number(v) }),
    );

    const pitchCountdown = h('input', { attrs: { type: 'checkbox' } });
    pitchCountdown.checked = this.settings.pitchCountdown;
    pitchCountdown.addEventListener('change', () =>
      this.callbacks.onSettingsChange({ pitchCountdown: pitchCountdown.checked }));

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

    const micPitch = h('input', { attrs: { type: 'checkbox' } });
    micPitch.checked = this.settings.micPitch;
    micPitch.addEventListener('change', () =>
      this.callbacks.onSettingsChange({ micPitch: micPitch.checked }));

    const audioOut = h('select', { className: 'ey-select' });
    audioOut.addEventListener('change', () =>
      this.callbacks.onSettingsChange({ audioOutputId: audioOut.value }));
    const micDevice = h('select', { className: 'ey-select' });
    micDevice.addEventListener('change', () =>
      this.callbacks.onSettingsChange({ micDeviceId: micDevice.value }));
    void this.populateAudioDevices(audioOut, micDevice);

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
    const dot = h('span', { className: 'ey-dot', title: '서버 연결 상태' });
    dot.classList.toggle('ok', this.serverAvailable);
    this.settingsDot = dot;

    const apiKeyInput = h('input', { className: 'ey-input', attrs: { type: 'password', placeholder: '(선택) 서버 API 키' } });
    apiKeyInput.value = this.settings.apiKey;
    apiKeyInput.addEventListener('change', () =>
      this.callbacks.onSettingsChange({ apiKey: apiKeyInput.value.trim() }));

    return h('div', { className: 'ey-settings' },
      h('div', { className: 'ey-settings-row' }, h('label', { text: '자동 가사 검색 (음악 영상만)', attrs: { title: '유튜브 음악 메타·채널·제목으로 음악 영상을 판별해 자동으로 가사창을 엽니다. 꺼도 툴바 아이콘으로 수동으로 열 수 있어요.' } }), autoSearch),
      h('div', { className: 'ey-settings-row' }, h('label', { text: '폰트 크기' }), fontSelect),
      h('div', { className: 'ey-settings-row' }, h('label', { text: '테마' }), themeSelect),
      h('div', { className: 'ey-settings-row' }, h('label', { text: '가사 번역 표시' }), showTranslation),
      h('div', { className: 'ey-settings-row' }, h('label', { text: '번역 언어' }), langSelect),
      h('div', { className: 'ey-settings-row' }, h('label', { text: '발음 표기 표시 (있을 때)' }), showPronunciation),
      h('div', { className: 'ey-settings-row' }, h('label', { text: '가사 소스 우선순위' }), sourcePriority),
      h('div', { className: 'ey-settings-row' }, h('label', { text: 'PiP 중에도 패널 가사 유지' }), pipKeepPanel),
      h('div', { className: 'ey-settings-row' }, h('label', { text: 'PiP에 영상 함께 표시' }), pipShowVideo),
      h('div', { className: 'ey-settings-row' }, h('label', { text: '가라오케 음정 바 (PiP)' }), pitchGuide),
      h('div', { className: 'ey-settings-row' }, h('label', { text: '음정 바 표시 구간' }), pitchWindow),
      h('div', { className: 'ey-settings-row' }, h('label', { text: '음정 바 진행 방식' }), pitchMode),
      h('div', { className: 'ey-settings-row' }, h('label', { text: '음정 바 글자 크기' }), pitchFont),
      h('div', { className: 'ey-settings-row' }, h('label', { text: '가사 시작 카운트다운' }), pitchCountdown),
      h('div', { className: 'ey-settings-row' },
        h('label', { text: '멜로디 재생 (가라오케 창)', attrs: { title: '전사된 노트를 신디사이즈로 재생합니다. 가라오케 창이 열려 있을 때만 소리가 나요.' } }),
        h('span', { className: 'ey-settings-inline' }, melodyVolume, melodyPlayback)),
      h('div', { className: 'ey-settings-row' },
        h('label', { text: '메트로놈', attrs: { title: '서버가 추정한 BPM에 맞춰 박자를 칩니다 (4/4 가정, 4박마다 강세).' } }),
        h('span', { className: 'ey-settings-inline' }, metronomeVolume, metronome)),
      h('div', { className: 'ey-settings-row' },
        h('label', { text: '가라오케 오디오 출력 기기', attrs: { title: '멜로디·메트로놈만 이 기기로 나갑니다. 영상 소리는 기존 출력 그대로.' } }), audioOut),
      h('div', { className: 'ey-settings-row' },
        h('label', { text: '마이크 음정 표시 (레인)', attrs: { title: '마이크로 부른 음정을 가라오케 레인에 청록 궤적으로 표시합니다. 켜면 마이크 권한을 요청해요.' } }), micPitch),
      h('div', { className: 'ey-settings-row' }, h('label', { text: '마이크 기기' }), micDevice),
      h('div', { className: 'ey-settings-note', text: '기기 이름은 마이크 권한을 한 번 허용해야 표시돼요' }),
      h('div', { className: 'ey-settings-row ey-settings-col' },
        h('label', {}, '싱크 서버 URL ', dot),
        serverInput,
      ),
      h('div', { className: 'ey-settings-row ey-settings-col' },
        h('label', { text: 'API 키' }),
        apiKeyInput,
      ),
      h('div', { className: 'ey-settings-row' }, h('label', { text: '디버그 정보 표시' }), debugInfo),
      h('div', { className: 'ey-settings-note', text: '싱크 생성·번역은 Everyric 서버가 필요해요' }),
      h('button', { className: 'ey-secondary-btn', text: '닫기', on: { click: () => this.closeSettings() } }),
    );
  }

  private buildRange(value: number, onChange: (v: number) => void): HTMLInputElement {
    const range = h('input', {
      className: 'ey-settings-range',
      attrs: { type: 'range', min: '0', max: '100', step: '1', value: String(Math.round(value * 100)) },
    });
    range.addEventListener('change', () => onChange(Number(range.value) / 100));
    return range;
  }

  /** 오디오 입출력 기기 목록 채우기 — 라벨은 마이크 권한을 허용해야 브라우저가 내려준다 */
  private async populateAudioDevices(outSel: HTMLSelectElement, inSel: HTMLSelectElement): Promise<void> {
    const fill = (sel: HTMLSelectElement, devices: MediaDeviceInfo[], defLabel: string, cur: string) => {
      sel.replaceChildren(h('option', { text: defLabel, attrs: { value: '' } }));
      devices.forEach((d, i) => {
        if (!d.deviceId || d.deviceId === 'default' || d.deviceId === 'communications') return;
        sel.append(h('option', { text: d.label || `기기 ${i + 1}`, attrs: { value: d.deviceId } }));
      });
      sel.value = Array.from(sel.options).some(o => o.value === cur) ? cur : '';
    };
    let devices: MediaDeviceInfo[] = [];
    try {
      devices = await navigator.mediaDevices.enumerateDevices();
    } catch {
      /* 권한 API 불가 환경 — 기본 항목만 표시 */
    }
    fill(outSel, devices.filter(d => d.kind === 'audiooutput'), '기본 출력', this.settings.audioOutputId);
    fill(inSel, devices.filter(d => d.kind === 'audioinput'), '기본 마이크', this.settings.micDeviceId);
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
    this.collapseBtn.title = g.collapsed ? '펼치기' : '접기';
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
    this.fullscreenHidden = document.fullscreenElement !== null;
    this.updateHostVisibility();
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
