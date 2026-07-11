import type { LyricLine, SongTempo, SyncDebugMeta } from '../types';
import type { MicSample } from '../lib/mic-pitch';
import { h, icon } from './dom';
import { appendKaraokeSpans, appendTimedSpans } from './karaoke';

interface DocumentPictureInPictureApi {
  requestWindow(options?: { width?: number; height?: number }): Promise<Window>;
}

export interface PipOptions {
  /** 열 때 영상 미러 영역을 포함한 크기로 열지 여부 */
  showVideo: boolean;
  /** 영상 영역 세로 비율 (0 = 자동 16:9) */
  initialVideoRatio: number;
  /** 현재 라인 밑에 한국어 발음 표기 표시 여부 */
  showPronunciation: boolean;
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
  /** 디버그: 글자별 CTC 신뢰도를 색으로 표시 */
  showConfidence: boolean;
  /** 레인 높이 드래그 조절 완료 시 */
  onPitchHeightChange: (px: number) => void;
  /** 가사 라인 클릭 — 가사 타임라인(초) 기준 */
  onSeek: (time: number) => void;
  /** 진행 바 클릭 — 영상 길이 대비 비율(0..1) */
  onSeekRatio: (ratio: number) => void;
  onPlayPause: () => void;
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
  /** 마이크 음정 표시가 켜져 있으면 최근 피치 샘플을 돌려준다 (없으면 null) */
  getMicSamples: (() => MicSample[]) | null;
  onClosed: () => void;
}

const PLAY_SVG = '<svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>';
const PAUSE_SVG = '<svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M6 5h4v14H6zM14 5h4v14h-4z"/></svg>';
const VOLUME_SVG = '<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3a4.5 4.5 0 0 0-2.5-4v8a4.5 4.5 0 0 0 2.5-4zM14 3.2v2.1a7 7 0 0 1 0 13.4v2.1a9 9 0 0 0 0-17.6z"/></svg>';
const MUTED_SVG = '<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M3 9v6h4l5 5V4L7 9H3zm13.6 3 2.7-2.7-1.4-1.4-2.7 2.7-2.7-2.7-1.4 1.4 2.7 2.7-2.7 2.7 1.4 1.4 2.7-2.7 2.7 2.7 1.4-1.4-2.7-2.7z"/></svg>';
const NOTE_SVG = '<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M12 3v10.55A4 4 0 1 0 14 17V7h4V3h-6z"/></svg>';
const METRO_SVG = '<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M9 2h6l4 19H5L9 2zm1.6 2L7.4 19h9.2l-1.2-6.1-2.7 2.7-1.4-1.4L15 10.4 13.9 4h-3.3z"/></svg>';

const MIN_VIDEO_RATIO = 0.15;
const MAX_VIDEO_RATIO = 0.75;

// 계이름 (pitch class 0 = 도)
const PITCH_NAMES_KO = ['도', '도#', '레', '레#', '미', '파', '파#', '솔', '솔#', '라', '라#', '시'];

/** CTC 신뢰도 로그 버킷 색 — 패널(overlay.css의 .ey-conf-*)과 반드시 같은 값 유지 */
const CONF_COLOR_LOW = '#ff6b6b';
const CONF_COLOR_MID = '#ffd166';
const CONF_COLOR_OK = '#51cf66';
function confBucketColor(conf: number): string {
  return conf < 1e-4 ? CONF_COLOR_LOW : conf < 2e-2 ? CONF_COLOR_MID : CONF_COLOR_OK;
}

/** 디버그 오버레이: 세이프가드 규칙별 마커 색 */
const FIX_COLORS: Record<string, string> = {
  stretch: '#ff6b6b', // 8s+ 늘어짐 클램프
  repeat: '#ff8787',  // 반복행 outlier 클램프
  pull: '#4dabf7',    // 간주 후 시작 당김
  tail: '#ffa94d',    // 끝음 연장
  snap: '#da77f2',    // 무음 온셋 스냅
};

interface PitchNote {
  midi: number;
  start: number;
  end: number;
  /** 이 노트(음절)에 대응하는 발음 표기 — 계이름처럼 노트에 직접 붙여 그린다 */
  pron?: string;
}

interface PitchWord {
  word: string;
  start: number;
  end: number;
  confidence?: number;
}

/** 라인 메타 (카운트다운·현재 라인 번역/발음 폴백용) — lines 배열과 인덱스 1:1 */
interface PitchLine {
  line: LyricLine;
  start: number;
  end: number;
  hasNotes: boolean;
}

/** setLines에서 미리 평탄화해 두는 레인 데이터 (렌더는 시간 창으로 필터만) */
interface PitchData {
  pages: PitchLine[];
  notes: PitchNote[];
  words: PitchWord[];
  /** 곡 전체 고정 세로 스케일 (라인마다 출렁이지 않게) */
  lo: number;
  hi: number;
}

interface PitchColors {
  /** 오선·번역 — 배경 요소 */
  faint: string;
  /** 아직 안 부른 노트·계이름 라벨·발음·스위프 선 */
  dim: string;
  /** 부른 부분·현재 노트 글로우·스위프 마커 */
  accent: string;
  /** 가사·현재 노트 테두리 */
  text: string;
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
  private footerEl: HTMLDivElement | null = null;
  private progressEl: HTMLDivElement | null = null;
  private playBtn: HTMLButtonElement | null = null;
  private muteBtn: HTMLButtonElement | null = null;
  private melodyBtn: HTMLButtonElement | null = null;
  private metroBtn: HTMLButtonElement | null = null;
  private getMicSamples: (() => MicSample[]) | null = null;
  private volumeSlider: HTMLInputElement | null = null;
  private volumeDragging = false;
  private timeEl: HTMLSpanElement | null = null;
  private lastPaused: boolean | null = null;
  private lastMuted: boolean | null = null;
  private wordEls: { start: number; el: HTMLElement }[] = [];
  private onSeek: (time: number) => void = () => {};
  private pitchCanvas: HTMLCanvasElement | null = null;
  private pitchDividerEl: HTMLDivElement | null = null;
  private pitch: PitchData = { pages: [], notes: [], words: [], lo: 57, hi: 71 };
  private pitchEnabled = true;
  private pitchLaneHeight = 170;
  private pitchWindowMeasures = 4;
  private pitchScrollMode: 'page' | 'scroll' = 'page';
  private pitchFontScale = 1.2;
  private pitchCountdown = true;
  private tempo: SongTempo | null = null;
  private showConfidence = false;
  private pitchColors: PitchColors | null = null;
  /** 곡 단위 정렬 진단 (디버그 모드 레인 오버레이: VAD/간주 스트립·RAW f0·정렬 텍스트) */
  private debugMeta: SyncDebugMeta | null = null;
  /** 라인 confidence 통계 — setLines에서 1회 계산 (디버그 헤더 표시용) */
  private confStats: { med: number; ok: number; mid: number; low: number } | null = null;

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

    let win: Window;
    try {
      win = await api.requestWindow({ width: 440, height: opts.showVideo ? 500 : 260 });
    } catch {
      return false;
    }
    this.win = win;
    this.onSeek = opts.onSeek;
    this.videoRatio = opts.initialVideoRatio;
    this.pitchEnabled = opts.pitchEnabled;
    this.pitchLaneHeight = opts.pitchLaneHeight;
    this.pitchWindowMeasures = opts.pitchWindowMeasures;
    this.pitchScrollMode = opts.pitchScrollMode;
    this.pitchFontScale = opts.pitchFontScale;
    this.pitchCountdown = opts.pitchCountdown;
    this.showConfidence = opts.showConfidence;

    const doc = win.document;
    doc.title = 'Everyric 가사';
    const style = doc.createElement('style');
    style.textContent = cssText;
    doc.head.append(style);
    doc.body.className = 'ey-pip';
    doc.body.classList.toggle('ey-hide-pron', !opts.showPronunciation);

    // 영상 미러 영역 + 비율 조절 디바이더 (attachVideo 전까지 숨김)
    this.videoWrapEl = h('div', { className: 'ey-pip-video' });
    this.videoWrapEl.style.display = 'none';
    this.dividerEl = this.buildDivider(win, opts.onVideoRatioChange);
    this.dividerEl.style.display = 'none';

    this.prevEl = h('div', { className: 'ey-pip-line prev', on: { click: () => this.seekRelative(-1) } });
    this.currentEl = h('div', { className: 'ey-pip-line current', on: { click: () => this.seekRelative(0) } });
    this.pronEl = h('div', { className: 'ey-pip-pron' });
    this.trEl = h('div', { className: 'ey-pip-tr' });
    this.nextEl = h('div', { className: 'ey-pip-line next', on: { click: () => this.seekRelative(1) } });
    this.titleEl = h('div', { className: 'ey-pip-title' });

    this.playBtn = h('button', {
      className: 'ey-pip-play',
      title: '재생/일시정지',
      on: { click: () => opts.onPlayPause() },
    }, icon(PLAY_SVG));

    this.muteBtn = h('button', {
      className: 'ey-pip-play ey-pip-mute',
      title: '음소거',
      on: { click: () => opts.onMuteToggle() },
    }, icon(VOLUME_SVG));

    this.melodyBtn = h('button', {
      className: 'ey-pip-play ey-pip-mute',
      title: '멜로디 재생 (노트 신디사이즈)',
      on: { click: () => opts.onMelodyToggle() },
    }, icon(NOTE_SVG));
    this.metroBtn = h('button', {
      className: 'ey-pip-play ey-pip-mute',
      title: '메트로놈',
      on: { click: () => opts.onMetronomeToggle() },
    }, icon(METRO_SVG));
    this.setAudioState(opts.melodyOn, opts.metronomeOn);
    this.getMicSamples = opts.getMicSamples;

    this.volumeSlider = h('input', {
      className: 'ey-pip-volume',
      title: '볼륨',
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
      title: '클릭해서 이동',
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
    this.pitchDividerEl = this.buildPitchDivider(opts.onPitchHeightChange);
    this.applyPitchVisibility();

    this.footerEl = h('div', { className: 'ey-pip-footer' },
      this.titleEl,
      h('div', { className: 'ey-pip-controls' },
        this.playBtn, this.muteBtn, this.melodyBtn, this.metroBtn, this.volumeSlider, progressWrap, this.timeEl),
    );
    doc.body.append(
      this.videoWrapEl,
      this.dividerEl,
      h('div', { className: 'ey-pip-stage' }, this.prevEl, this.currentEl, this.pronEl, this.trEl, this.nextEl),
      this.pitchDividerEl,
      this.pitchCanvas,
      this.footerEl,
    );

    // 창 크기가 줄면 레인이 푸터를 밀어내지 않도록 즉시 재클램프
    win.addEventListener('resize', () => this.clampLaneHeight());

    win.addEventListener('pagehide', () => {
      this.stopMirror();
      this.win = null;
      this.index = -1;
      this.wordEls = [];
      this.lastPaused = null;
      this.lastMuted = null;
      this.volumeDragging = false;
      this.pitchCanvas = null;
      this.pitchDividerEl = null;
      this.pitchColors = null;
      this.videoWrapEl = null;
      this.dividerEl = null;
      this.footerEl = null;
      this.melodyBtn = null;
      this.metroBtn = null;
      this.getMicSamples = null;
      opts.onClosed();
    });
    this.renderLines();
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
    if (enabled && source) {
      this.attachVideo(source);
    } else {
      this.stopMirror();
      this.hideVideoArea();
    }
  }

  setSong(title: string, artist: string): void {
    if (this.titleEl) {
      this.titleEl.textContent = artist ? `${title} — ${artist}` : title;
    }
  }

  setLines(lines: LyricLine[]): void {
    this.lines = lines;
    this.index = -1;
    this.pitch = collectPitchData(lines);
    const confs = lines.map(l => l.confidence).filter((v): v is number => v != null).sort((a, b) => a - b);
    const low = confs.filter(v => v < 1e-4).length / Math.max(1, confs.length);
    const mid = confs.filter(v => v >= 1e-4 && v < 2e-2).length / Math.max(1, confs.length);
    this.confStats = confs.length === 0 ? null : {
      med: confs[Math.floor(confs.length / 2)],
      ok: 1 - low - mid,
      mid,
      low,
    };
    this.applyPitchVisibility();
    this.renderLines();
  }

  /** 레인 표시 구간(마디 수) 설정 즉시 반영 — 0.5마디까지 허용 */
  setPitchWindow(measures: number): void {
    this.pitchWindowMeasures = Math.min(16, Math.max(0.25, measures));
  }

  /** 레인 진행 방식(페이지/스크롤) 즉시 반영 */
  setPitchScrollMode(mode: 'page' | 'scroll'): void {
    this.pitchScrollMode = mode;
  }

  /** 레인 글자 크기 배율 즉시 반영 */
  setPitchFontScale(scale: number): void {
    this.pitchFontScale = Math.min(2, Math.max(0.6, scale));
  }

  /** 서버가 추정한 곡 템포 — 마디 창 폭·비트 격자에 사용, null이면 초 단위 폴백 */
  setTempo(tempo: SongTempo | null): void {
    this.tempo = tempo && tempo.bpm > 0 ? tempo : null;
  }

  /** 곡 단위 정렬 진단 메타 — 디버그 모드에서 VAD/간주 스트립·RAW f0 곡선을 그린다 */
  setDebugMeta(meta: SyncDebugMeta | null): void {
    this.debugMeta = meta;
  }

  /** 카운트다운 설정 즉시 반영 */
  setPitchCountdown(enabled: boolean): void {
    this.pitchCountdown = enabled;
  }

  /** 현재 라인 인덱스 변경 시 호출 */
  update(index: number): void {
    if (!this.win || index === this.index) return;
    this.index = index;
    this.renderLines();
  }

  /** 번역 등 라인 데이터가 바뀐 뒤 강제 재렌더 */
  refresh(): void {
    if (this.win) this.renderLines();
  }

  /** 발음 표기 설정 토글 즉시 반영 */
  setShowPronunciation(visible: boolean): void {
    this.win?.document.body.classList.toggle('ey-hide-pron', !visible);
  }

  /** 가라오케 음정 바 설정 토글 즉시 반영 */
  setPitchEnabled(enabled: boolean): void {
    this.pitchEnabled = enabled;
    this.applyPitchVisibility();
  }

  /** 레인 표시 조건 = 설정 on + 노트 데이터 있음. 레인이 켜지면 스테이지는 숨긴다(중복 표시). */
  private applyPitchVisibility(): void {
    const show = this.pitchEnabled && this.pitch.notes.length > 0;
    if (this.pitchCanvas) this.pitchCanvas.style.display = show ? '' : 'none';
    if (this.pitchDividerEl) this.pitchDividerEl.style.display = show ? '' : 'none';
    this.win?.document.body.classList.toggle('ey-lane-active', show);
    this.syncVideoLayout();
    this.clampLaneHeight();
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
  }

  /** 레인 높이 상한 — 푸터·영상 최소 영역을 침범해 창 밖으로 밀어내지 않는 한도 */
  private maxLaneHeight(): number {
    const win = this.win;
    if (!win || win.innerHeight === 0) return 320;
    const footerH = this.footerEl?.offsetHeight || 60;
    const videoShown = this.videoWrapEl ? this.videoWrapEl.style.display !== 'none' : false;
    const reserved = footerH + 8 + (videoShown ? 48 : 0);
    return Math.max(90, Math.min(320, win.innerHeight - reserved));
  }

  /** 창이 줄어 현재 레인 높이가 상한을 넘으면 맞춰서 줄인다 (창 resize·레인 토글 시) */
  private clampLaneHeight(): void {
    if (!this.pitchCanvas || this.pitchCanvas.style.display === 'none') return;
    const max = this.maxLaneHeight();
    if (this.pitchCanvas.clientHeight > max) {
      this.pitchLaneHeight = max;
      this.pitchCanvas.style.height = `${max}px`;
    }
  }

  /** 디버그 신뢰도 색상 토글 즉시 반영 */
  setShowConfidence(enabled: boolean): void {
    this.showConfidence = enabled;
  }

  /** 멜로디/메트로놈 토글 버튼 활성 상태 반영 */
  setAudioState(melody: boolean, metronome: boolean): void {
    this.melodyBtn?.classList.toggle('on', melody);
    this.metroBtn?.classList.toggle('on', metronome);
  }

  /** 레인 위 디바이더 — 위로 끌면 레인이 커진다. 놓으면 설정에 저장. */
  private buildPitchDivider(onHeightChange: (px: number) => void): HTMLDivElement {
    const divider = h('div', {
      className: 'ey-pip-divider ey-pip-pitch-divider',
      title: '드래그해서 가라오케 레인 높이 조절',
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

  /** 매 tick 호출: 카라오케 필 + 진행 바 + 시간/재생 상태 */
  tick(time: number, duration: number, paused: boolean): void {
    if (!this.win) return;
    for (const { start, el } of this.wordEls) {
      el.classList.toggle('sung', start <= time);
    }
    if (this.progressEl && duration > 0) {
      this.progressEl.style.width = `${Math.min(100, (time / duration) * 100)}%`;
    }
    if (this.timeEl && duration > 0) {
      this.timeEl.textContent = `${formatTime(time)} / ${formatTime(duration)}`;
    }
    if (this.playBtn && paused !== this.lastPaused) {
      this.lastPaused = paused;
      this.playBtn.replaceChildren(icon(paused ? PLAY_SVG : PAUSE_SVG));
    }
    this.renderPitch(time);
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
      this.muteBtn.title = muted ? '음소거 해제' : '음소거';
    }
  }

  /**
   * 음정 레인 렌더 — 창 폭이 N마디(서버 추정 BPM 기준)로 일정한 오선지.
   * page 모드(기본)는 마디 창이 고정된 채 플레이헤드가 왼→오로 쓸고 지나가고,
   * scroll 모드는 플레이헤드가 좌측 28%에 고정된 채 오선이 횡스크롤된다.
   * 간주는 빈 오선으로 지나가고, 발음은 계이름처럼 각 노트 앞머리에 붙는다.
   */
  private renderPitch(now: number): void {
    const canvas = this.pitchCanvas;
    const win = this.win;
    if (!canvas || !win || !this.pitchEnabled || this.pitch.notes.length === 0) return;

    const dpr = win.devicePixelRatio || 1;
    const cw = canvas.clientWidth;
    const ch = canvas.clientHeight;
    if (cw === 0 || ch === 0) return;
    const bw = Math.round(cw * dpr);
    const bh = Math.round(ch * dpr);
    if (canvas.width !== bw || canvas.height !== bh) {
      canvas.width = bw;
      canvas.height = bh;
    }
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cw, ch);

    const colors = this.pitchColors ?? (this.pitchColors = readPitchColors(win));
    const { pages, notes, words, lo, hi } = this.pitch;

    // ── 세로 레이아웃: 오선 영역 + 가사 줄 + (발음 폴백 줄) + 번역 줄
    const hasSegs = pages.some(p => p.line.pronSegments && p.line.pronSegments.length > 0);
    const hasPronRow = !hasSegs && pages.some(p => p.line.pronunciation);
    const hasTr = pages.some(p => p.line.translation);
    const fs = this.pitchFontScale;
    const lyricH = Math.max(16, Math.min(34, Math.round(ch * 0.15 * fs)));
    const lyricPx = Math.max(13, Math.round(lyricH * 0.72));
    const pronPx = Math.max(10, Math.round(lyricPx * 0.8));
    const pronRowH = hasPronRow ? pronPx + 6 : 0;
    const trPx = Math.max(12, Math.min(22, Math.round(ch * 0.085 * fs)));
    const trH = hasTr ? trPx + 7 : 0;
    const namePx = Math.max(10, Math.round(11 * fs));
    const padTop = 2;
    const staffH = Math.max(30, ch - padTop - lyricH - pronRowH - trH - 2);

    // ── 고정 시간 스케일: 창 폭 = N마디(4/4 가정) — 템포 없으면 120BPM 가정 폴백
    const secPerBeat = this.tempo ? 60 / this.tempo.bpm : 0.5;
    const W = this.pitchWindowMeasures * 4 * secPerBeat;
    let t0: number;
    let playheadX: number;
    if (this.pitchScrollMode === 'page') {
      // 페이지 모드: 창을 마디 경계(beat_offset 기준)에 고정하고 플레이헤드가 이동
      const offset = this.tempo?.beat_offset ?? 0;
      t0 = offset + Math.floor((now - offset) / W) * W;
      playheadX = ((now - t0) / W) * cw;
    } else {
      // 스크롤 모드: 플레이헤드 좌측 28% 고정, 오선이 흐른다
      t0 = now - W * 0.28;
      playheadX = cw * 0.28;
    }
    const x = (t: number) => ((t - t0) / W) * cw;

    // ── 곡 전체 고정 세로 스케일 (위아래 덧줄 여백 포함)
    const marginY = staffH * 0.16;
    const usable = staffH - marginY * 2;
    const semiPx = usable / Math.max(1, hi - lo);
    const y = (midi: number) => padTop + marginY + usable - (midi - lo) * semiPx;

    // 오선 5줄 — 음역을 4등분한 시각 기준선
    ctx.fillStyle = colors.faint;
    for (let i = 0; i < 5; i++) {
      ctx.fillRect(0, padTop + marginY + (usable / 4) * i - 0.5, cw, 1);
    }
    // 세로 눈금: 템포가 있으면 첫 비트에 정렬된 비트(옅게)/마디(진하게) 격자,
    // 없으면 1초 간격 폴백
    if (this.tempo) {
      const offset = this.tempo.beat_offset ?? 0;
      for (let b = Math.ceil((t0 - offset) / secPerBeat); ; b++) {
        const t = offset + b * secPerBeat;
        if (t >= t0 + W) break;
        const isMeasure = ((b % 4) + 4) % 4 === 0;
        ctx.globalAlpha = isMeasure ? 0.55 : 0.18;
        ctx.fillRect(x(t) - (isMeasure ? 0.75 : 0.5), padTop, isMeasure ? 1.5 : 1, staffH);
      }
    } else {
      ctx.globalAlpha = 0.3;
      for (let t = Math.ceil(t0); t < t0 + W; t++) {
        ctx.fillRect(x(t) - 0.5, padTop, 1, staffH);
      }
    }
    ctx.globalAlpha = 1;

    // ── 디버그 배경 레이어: VAD 가창/간주 스트립 + star 흡수 밴드 + RAW f0 곡선
    if (this.showConfidence) {
      this.renderDebugUnderlay(ctx, t0, W, x, y, lo, hi, cw, padTop, staffH);
    }

    // ── 노트 막대 + 계이름(위) + 발음(아래) — 시간 창 안의 노트만
    // 페이지 모드는 창 밖 요소를 그리면 가장자리에 다음 페이지 글자가 뭉치므로 여유 0
    const edgePad = this.pitchScrollMode === 'page' ? 0 : 0.5;
    const vis = notes.filter(n => n.end > t0 - edgePad && n.start < t0 + W + edgePad);
    const noteH = Math.max(5, Math.min(13, semiPx * 1.6));
    const noteR = Math.min(noteH / 2, 4);
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (const n of vis) {
      const x1 = x(n.start);
      const x2 = x(n.end);
      const w = Math.max(3, x2 - x1 - 1);
      const top = y(n.midi) - noteH / 2;
      const isCurrent = n.start <= now && now < n.end;

      ctx.fillStyle = colors.dim;
      ctx.beginPath();
      ctx.roundRect(x1, top, w, noteH, noteR);
      ctx.fill();
      if (now > n.start) {
        const sungW = Math.max(2, Math.min(w, x(now) - x1));
        ctx.fillStyle = colors.accent;
        ctx.beginPath();
        ctx.roundRect(x1, top, sungW, noteH, noteR);
        ctx.fill();
      }
      if (isCurrent) {
        // 지금 불러야 하는 노트: accent 글로우 + 밝은 테두리로 강조
        ctx.save();
        ctx.shadowColor = colors.accent;
        ctx.shadowBlur = 8;
        ctx.strokeStyle = colors.text;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.roundRect(x1, top, w, noteH, noteR);
        ctx.stroke();
        ctx.restore();
      }

      // 글자는 노트 중앙이 아니라 노트 앞머리(시작 지점)에 붙인다
      const lx = x1 + 1;
      ctx.textAlign = 'left';
      // 계이름 — 항상 노트 위
      if (w >= 14) {
        ctx.font = `bold ${namePx}px system-ui, sans-serif`;
        ctx.fillStyle = isCurrent ? colors.text : colors.dim;
        ctx.fillText(PITCH_NAMES_KO[((n.midi % 12) + 12) % 12], lx, Math.max(namePx * 0.7, top - namePx * 0.7));
      }
      // 발음 — 계이름처럼 노트 바로 아래에 부착 (사용자 요구: 노트마다 붙일 것)
      if (n.pron) {
        ctx.font = `${pronPx}px system-ui, sans-serif`;
        ctx.fillStyle = n.start <= now ? colors.accent : colors.dim;
        ctx.fillText(n.pron, lx, Math.min(padTop + staffH - pronPx / 2, top + noteH + 2 + pronPx / 2));
      }
      ctx.textAlign = 'center';
    }

    // ── 마이크 음정 궤적 — 사용자가 부른 피치를 청록 점 트레일로 (노트 accent와 구분되는 색)
    const mic = this.getMicSamples?.();
    if (mic && mic.length > 0) {
      const wallNow = performance.now() / 1000;
      for (const s of mic) {
        const age = wallNow - s.at;
        if (age > 2.5) continue;
        const t = now - age; // 표시용 근사 — 배속 중에는 궤적 간격만 살짝 달라진다
        if (t < t0 || t > t0 + W) continue;
        // 검출 옥타브 오차·성별 음역 차이를 흡수: 레인 음역으로 옥타브 폴딩
        let m = s.midi;
        while (m < lo && m + 12 <= hi + 6) m += 12;
        while (m > hi && m - 12 >= lo - 6) m -= 12;
        if (m < lo - 1 || m > hi + 1) continue;
        ctx.globalAlpha = Math.max(0.08, 1 - age / 2.5) * 0.9;
        ctx.fillStyle = '#4dd0e1';
        ctx.beginPath();
        ctx.arc(x(t), y(m), age < 0.12 ? 3.4 : 2.2, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalAlpha = 1;
    }

    // ── 카운트다운: 긴 묵음(5s+) 뒤 라인 시작 4초 전부터 4·3·2·1
    if (this.pitchCountdown) {
      const next = pages.find(p => p.start > now + 0.05 && p.hasNotes);
      if (next) {
        const prevEnd = pages.reduce(
          (acc, p) => (p.end <= next.start + 0.01 && p.end > acc ? p.end : acc), 0);
        const remain = next.start - now;
        if (remain <= 4 && next.start - prevEnd >= 5 && now >= prevEnd) {
          const num = Math.max(1, Math.ceil(remain));
          // 숫자를 라인 시작 시각 위치에 그린다 (창 밖이면 가장자리에 고정)
          const nx = Math.max(Math.min(playheadX + 30, cw - 30), Math.min(cw - 30, x(next.start)));
          ctx.font = `bold ${Math.round(staffH * 0.5)}px system-ui, sans-serif`;
          ctx.fillStyle = colors.accent;
          ctx.globalAlpha = 0.9;
          ctx.fillText(String(num), nx, padTop + staffH / 2);
          ctx.globalAlpha = 1;
        }
      }
    }

    // ── 라벨 충돌 회피 (좌→우 최소 간격 + 우측 되밀기)
    const placeRow = (items: { cx: number; w: number }[]): number[] => {
      const gap = 3;
      const xs = items.map(it => it.cx);
      for (let i = 0; i < xs.length; i++) {
        const minCx = i === 0
          ? items[i].w / 2 + 2
          : xs[i - 1] + items[i - 1].w / 2 + gap + items[i].w / 2;
        if (xs[i] < minCx) xs[i] = minCx;
      }
      for (let i = xs.length - 1; i >= 0; i--) {
        const maxCx = i === xs.length - 1
          ? cw - items[i].w / 2 - 2
          : xs[i + 1] - items[i + 1].w / 2 - gap - items[i].w / 2;
        if (xs[i] > maxCx) xs[i] = maxCx;
      }
      return xs;
    };

    // ── 노트 아래 가사: 창 안의 글자 토큰을 시간 위치 + 충돌 회피로
    let ty = padTop + staffH;
    ctx.font = `bold ${lyricPx}px system-ui, sans-serif`;
    const visWords = words.filter(w => w.end > t0 - edgePad && w.start < t0 + W + edgePad);
    if (visWords.length > 0) {
      // 글자를 노트 시작 지점에서 시작하도록 배치 (충돌 회피는 중심 좌표 기준)
      const items = visWords.map(w => {
        const width = ctx.measureText(w.word).width;
        return { w, cx: x(w.start) + width / 2, width };
      });
      const xs = placeRow(items.map(it => ({ cx: it.cx, w: it.width })));
      items.forEach((it, i) => {
        let color = it.w.start <= now ? colors.accent : colors.text;
        // 디버그: 정렬 신뢰도(기하평균 확률, 로그 버킷) — 패널과 동일 색 (confBucketColor)
        if (this.showConfidence && it.w.confidence != null) {
          color = confBucketColor(it.w.confidence);
        }
        ctx.fillStyle = color;
        ctx.fillText(it.w.word, xs[i], ty + lyricH * 0.55);
      });
    }
    ty += lyricH;

    // ── 발음 폴백 줄 (음절 타이밍이 아예 없는 곡): 현재 라인 발음 그라데이션
    const page = this.index >= 0 ? pages[Math.min(this.index, pages.length - 1)] : undefined;
    if (hasPronRow) {
      if (page?.line.pronunciation) {
        this.renderPronFallback(ctx, page, now, cw, ty + pronRowH * 0.5, colors, pronPx);
      }
      ty += pronRowH;
    }

    // ── 번역: 현재 라인 번역을 하단 중앙에
    if (hasTr && page?.line.translation) {
      ctx.font = `${trPx}px system-ui, sans-serif`;
      ctx.fillStyle = colors.dim;
      ctx.fillText(page.line.translation, cw / 2, ty + trH * 0.5, cw - 16);
    }

    // ── 디버그 전경 레이어: 보정된 라인의 원본 타이밍 고스트 + 곡 전체 confidence 헤더
    if (this.showConfidence) {
      this.renderDebugOverlay(ctx, pages, t0, W, x, cw, padTop);
    }

    // ── 플레이헤드 (page 모드: 왼→오 이동, scroll 모드: 좌측 28% 고정)
    ctx.fillStyle = colors.dim;
    ctx.fillRect(playheadX - 0.75, padTop, 1.5, staffH);
    ctx.fillStyle = colors.accent;
    ctx.beginPath();
    ctx.moveTo(playheadX - 4, padTop);
    ctx.lineTo(playheadX + 4, padTop);
    ctx.lineTo(playheadX, padTop + 6);
    ctx.closePath();
    ctx.fill();
  }

  /**
   * 디버그 배경 레이어 — 정렬 파이프라인의 "재료"를 오선 뒤에 깐다:
   * 하단 스트립 = VAD 가창(초록)/간주·무성(회색), 그 위 보라 밴드 = star 흡수(가사 밖 가창),
   * 파란 곡선 = 음정 모델(RMVPE/FCPE) RAW f0 (노트 양자화 전 원본, null=무성에서 끊김).
   */
  private renderDebugUnderlay(
    ctx: CanvasRenderingContext2D,
    t0: number, W: number,
    x: (t: number) => number, y: (midi: number) => number,
    lo: number, hi: number, cw: number, padTop: number, staffH: number,
  ): void {
    const meta = this.debugMeta;
    if (!meta) return;
    const clampX = (v: number) => Math.max(0, Math.min(cw, v));
    const stripY = padTop + staffH - 4;
    const regions = meta.vad_regions ?? [];
    // 간주(리전 사이 갭 2s+)를 회색으로 — interpolation이 간주를 예약하지 않는 문제를 눈으로 본다
    for (let i = 0; i <= regions.length; i++) {
      const gapS = i === 0 ? 0 : regions[i - 1][1];
      const gapE = i === regions.length ? t0 + W : regions[i][0];
      if (gapE - gapS >= 2.0 && gapE > t0 && gapS < t0 + W) {
        ctx.fillStyle = 'rgba(134, 142, 150, 0.30)';
        ctx.fillRect(clampX(x(gapS)), stripY, clampX(x(gapE)) - clampX(x(gapS)), 4);
      }
    }
    for (const [s, e] of regions) {
      if (e <= t0 || s >= t0 + W) continue;
      ctx.fillStyle = 'rgba(81, 207, 102, 0.35)';
      ctx.fillRect(clampX(x(s)), stripY, Math.max(1, clampX(x(e)) - clampX(x(s))), 4);
    }
    for (const [s, e] of meta.star_spans ?? []) {
      if (e <= t0 || s >= t0 + W) continue;
      ctx.fillStyle = 'rgba(218, 119, 242, 0.5)';
      ctx.fillRect(clampX(x(s)), stripY - 3, Math.max(1, clampX(x(e)) - clampX(x(s))), 2);
    }
    const f0 = meta.f0_curve;
    if (f0 && f0.dt > 0 && f0.midi.length > 0) {
      ctx.strokeStyle = 'rgba(77, 171, 247, 0.65)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      let pen = false;
      const i0 = Math.max(0, Math.floor((t0 - f0.t0) / f0.dt));
      const i1 = Math.min(f0.midi.length - 1, Math.ceil((t0 + W - f0.t0) / f0.dt));
      for (let i = i0; i <= i1; i++) {
        const m = f0.midi[i];
        if (m == null) { pen = false; continue; }
        const px = x(f0.t0 + i * f0.dt);
        const py = y(Math.max(lo, Math.min(hi, m)));
        if (pen) ctx.lineTo(px, py);
        else { ctx.moveTo(px, py); pen = true; }
      }
      ctx.stroke();
    }
  }

  /**
   * 디버그 전경 레이어 — 세이프가드가 고친 라인의 보정 전 원본 타이밍을 점선 고스트로
   * (규칙별 색: stretch/repeat=빨강, pull=파랑, tail=주황, snap=보라, 세로 틱 = 원본·보정 후 시작),
   * 우상단에 곡 전체 confidence(median/avg/저신뢰 비율)와 정렬 텍스트(독음/원문) 헤더.
   */
  private renderDebugOverlay(
    ctx: CanvasRenderingContext2D,
    pages: PitchLine[],
    t0: number, W: number,
    x: (t: number) => number,
    cw: number, padTop: number,
  ): void {
    ctx.save();
    ctx.textBaseline = 'middle';
    let row = 0;
    for (const p of pages) {
      const dbg = p.line.debug;
      if (!dbg?.orig) continue;
      const [os, oe] = dbg.orig;
      if (Math.max(oe, p.end) < t0 || Math.min(os, p.start) > t0 + W) continue;
      const color = FIX_COLORS[dbg.fixes?.[0] ?? ''] ?? '#868e96';
      const gy = padTop + 6 + (row % 3) * 8; // 인접 라인 겹침 방지용 3행 로테이션
      row++;
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(x(os), gy);
      ctx.lineTo(x(oe), gy);
      ctx.stroke();
      ctx.setLineDash([]);
      // 세로 틱: 보정 전 시작(os)과 보정 후 시작 — 얼마나 옮겨졌는지 한눈에
      const ns = p.line.time ?? p.start;
      ctx.beginPath();
      ctx.moveTo(x(os), gy - 3);
      ctx.lineTo(x(os), gy + 3);
      ctx.moveTo(x(ns), gy - 3);
      ctx.lineTo(x(ns), gy + 3);
      ctx.stroke();
      if (dbg.fixes && dbg.fixes.length > 0) {
        ctx.font = '9px ui-monospace, monospace';
        ctx.fillStyle = color;
        ctx.textAlign = 'left';
        ctx.fillText(dbg.fixes.join('·'), Math.max(2, x(Math.max(os, t0)) + 2), gy - 6);
      }
    }
    if (this.confStats) {
      // 사람이 읽는 등급 분포 — 글자 색과 같은 3색으로 "정렬 좋음 87% 보통 10% 낮음 3%"
      const s = this.confStats;
      const at = this.debugMeta?.alignment_text;
      const parts: { text: string; color: string }[] = [
        { text: '정렬 ', color: '#868e96' },
        { text: `좋음${Math.round(s.ok * 100)}%`, color: CONF_COLOR_OK },
        { text: ` 보통${Math.round(s.mid * 100)}%`, color: CONF_COLOR_MID },
        { text: ` 낮음${Math.round(s.low * 100)}%`, color: CONF_COLOR_LOW },
      ];
      if (at) parts.push({ text: at === 'pronunciation' ? ' · 독음' : ' · 원문', color: '#868e96' });
      ctx.font = '10px ui-monospace, monospace';
      ctx.textAlign = 'left';
      ctx.globalAlpha = 0.9;
      let px = cw - 4 - parts.reduce((a, p) => a + ctx.measureText(p.text).width, 0);
      for (const p of parts) {
        ctx.fillStyle = p.color;
        ctx.fillText(p.text, px, padTop + 7);
        px += ctx.measureText(p.text).width;
      }
      ctx.globalAlpha = 1;
    }
    ctx.restore();
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
  }

  /** 발음 폴백 (음절 타이밍 없는 곡) — 현재 라인 발음을 가운데 정렬 + 진행률 그라데이션 */
  private renderPronFallback(
    ctx: CanvasRenderingContext2D,
    page: PitchLine,
    now: number,
    cw: number,
    py: number,
    colors: PitchColors,
    fontPx: number,
  ): void {
    const pron = page.line.pronunciation ?? '';
    if (!pron) return;
    ctx.font = `${fontPx}px system-ui, sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    const maxW = cw - 16;
    ctx.fillStyle = colors.dim;
    ctx.fillText(pron, cw / 2, py, maxW);
    const span = Math.max(0.001, page.end - page.start);
    const sungRatio = Math.max(0, Math.min(1, (now - page.start) / span));
    if (sungRatio <= 0) return;
    const textW = Math.min(ctx.measureText(pron).width, maxW);
    const x0 = cw / 2 - textW / 2;
    ctx.save();
    ctx.beginPath();
    ctx.rect(x0, py - fontPx, textW * sungRatio, fontPx * 2);
    ctx.clip();
    ctx.fillStyle = colors.accent;
    ctx.fillText(pron, cw / 2, py, maxW);
    ctx.restore();
  }

  private buildDivider(win: Window, onRatioChange: (ratio: number) => void): HTMLDivElement {
    const divider = h('div', { className: 'ey-pip-divider', title: '드래그해서 영상/가사 비율 조절' },
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
    // 음절 span을 wordEls에 합류시켜 tick의 sung 토글을 그대로 태운다
    if (this.pronEl) {
      this.pronEl.replaceChildren();
      const pron = current?.pronunciation ?? '';
      const segs = current?.pronSegments;
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
    if (current.words && current.words.length > 0) {
      appendKaraokeSpans(this.currentEl, current, word => {
        const el = h('span', { className: 'ey-word', text: word.word });
        this.wordEls.push({ start: word.start, el });
        return el;
      });
    } else {
      this.currentEl.textContent = current.text;
    }
  }
}

/**
 * 라인 배열에서 레인 데이터를 평탄화한다.
 * - pages: 라인 인덱스와 1:1 (카운트다운·현재 라인 번역/발음 폴백용)
 * - notes: 전 곡 노트, 각 노트에 대응 발음 음절(pron)을 최대 겹침 기준으로 부착
 * - words: 전 곡 글자 토큰
 * - lo/hi: 곡 전체 고정 세로 스케일 (최소 14반음)
 */
function collectPitchData(lines: LyricLine[]): PitchData {
  const pages: PitchLine[] = [];
  const notes: PitchNote[] = [];
  const words: PitchWord[] = [];
  let lo = Infinity;
  let hi = -Infinity;

  lines.forEach((line, i) => {
    const lineNotes: PitchNote[] = [];
    if (line.words) {
      for (const word of line.words) {
        words.push({ word: word.word, start: word.start, end: word.end, confidence: word.confidence });
        if (!word.notes) continue;
        for (const n of word.notes) lineNotes.push({ midi: n.midi, start: n.start, end: n.end });
      }
    }
    if (line.notes) {
      for (const n of line.notes) lineNotes.push({ midi: n.midi, start: n.start, end: n.end });
    }
    lineNotes.sort((a, b) => a.start - b.start);

    // 발음 음절을 최대 겹침 노트에 부착 — 계이름처럼 노트에 직접 그린다
    if (line.pronSegments) {
      for (const seg of line.pronSegments) {
        let best: PitchNote | null = null;
        let bestOv = 0;
        for (const n of lineNotes) {
          const ov = Math.min(n.end, seg.end) - Math.max(n.start, seg.start);
          if (ov > bestOv) {
            bestOv = ov;
            best = n;
          }
        }
        if (best) best.pron = best.pron ? best.pron + seg.text : seg.text;
      }
    }

    let start = line.time ?? lineNotes[0]?.start ?? 0;
    let end = line.endTime ?? lines[i + 1]?.time ?? start + 4;
    for (const n of lineNotes) {
      start = Math.min(start, n.start);
      end = Math.max(end, n.end);
      lo = Math.min(lo, n.midi);
      hi = Math.max(hi, n.midi);
    }
    pages.push({ line, start, end: Math.max(end, start + 0.5), hasNotes: lineNotes.length > 0 });
    notes.push(...lineNotes);
  });

  notes.sort((a, b) => a.start - b.start);
  words.sort((a, b) => a.start - b.start);
  if (!Number.isFinite(lo)) {
    lo = 57;
    hi = 71;
  }
  lo -= 1;
  hi += 1;
  while (hi - lo < 14) {
    lo -= 1;
    hi += 1;
  }
  return { pages, notes, words, lo, hi };
}

function readPitchColors(win: Window): PitchColors {
  const style = win.getComputedStyle(win.document.documentElement);
  const pick = (name: string, fallback: string) => style.getPropertyValue(name).trim() || fallback;
  return {
    faint: pick('--ey-text-faint', 'rgba(241, 241, 242, 0.34)'),
    dim: pick('--ey-text-dim', 'rgba(241, 241, 242, 0.58)'),
    accent: pick('--ey-accent', '#ffb02e'),
    text: pick('--ey-text', '#f1f1f2'),
  };
}

function formatTime(sec: number): string {
  const total = Math.max(0, Math.floor(sec));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${s.toString().padStart(2, '0')}`;
}
