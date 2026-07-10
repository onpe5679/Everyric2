import type { LyricLine } from '../types';
import { h, icon } from './dom';
import { appendKaraokeSpans } from './karaoke';

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
  onClosed: () => void;
}

const PLAY_SVG = '<svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>';
const PAUSE_SVG = '<svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M6 5h4v14H6zM14 5h4v14h-4z"/></svg>';
const VOLUME_SVG = '<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3a4.5 4.5 0 0 0-2.5-4v8a4.5 4.5 0 0 0 2.5-4zM14 3.2v2.1a7 7 0 0 1 0 13.4v2.1a9 9 0 0 0 0-17.6z"/></svg>';
const MUTED_SVG = '<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M3 9v6h4l5 5V4L7 9H3zm13.6 3 2.7-2.7-1.4-1.4-2.7 2.7-2.7-2.7-1.4 1.4 2.7 2.7-2.7 2.7 1.4 1.4 2.7-2.7 2.7 2.7 1.4-1.4-2.7-2.7z"/></svg>';

const MIN_VIDEO_RATIO = 0.15;
const MAX_VIDEO_RATIO = 0.75;

// 계이름 (pitch class 0 = 도)
const PITCH_NAMES_KO = ['도', '도#', '레', '레#', '미', '파', '파#', '솔', '솔#', '라', '라#', '시'];

interface PitchNote {
  midi: number;
  start: number;
  end: number;
}

/** 오선지 레인 한 페이지 = 가사 한 라인 (노트에 가사/발음/번역을 정렬해 그린다) */
interface PitchLine {
  line: LyricLine;
  notes: PitchNote[];
  /** 페이지 시간 범위 — 노트·가사 토큰·라인 타임스탬프를 모두 포함 */
  start: number;
  end: number;
  /** 페이지 내 고정 세로 매핑값 — 노트 midi 중앙과 범위(반음) */
  centerMidi: number;
  span: number;
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
  private progressEl: HTMLDivElement | null = null;
  private playBtn: HTMLButtonElement | null = null;
  private muteBtn: HTMLButtonElement | null = null;
  private volumeSlider: HTMLInputElement | null = null;
  private volumeDragging = false;
  private timeEl: HTMLSpanElement | null = null;
  private lastPaused: boolean | null = null;
  private lastMuted: boolean | null = null;
  private wordEls: { start: number; el: HTMLElement }[] = [];
  private onSeek: (time: number) => void = () => {};
  private pitchCanvas: HTMLCanvasElement | null = null;
  private pitchDividerEl: HTMLDivElement | null = null;
  private pitchLines: PitchLine[] = [];
  private pitchEnabled = true;
  private pitchLaneHeight = 170;
  private showConfidence = false;
  private pitchColors: PitchColors | null = null;

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

    doc.body.append(
      this.videoWrapEl,
      this.dividerEl,
      h('div', { className: 'ey-pip-stage' }, this.prevEl, this.currentEl, this.pronEl, this.trEl, this.nextEl),
      this.pitchDividerEl,
      this.pitchCanvas,
      h('div', { className: 'ey-pip-footer' },
        this.titleEl,
        h('div', { className: 'ey-pip-controls' },
          this.playBtn, this.muteBtn, this.volumeSlider, progressWrap, this.timeEl),
      ),
    );

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
      if (this.dividerEl) this.dividerEl.style.display = '';
      if (this.videoRatio > 0) this.applyVideoRatio(this.videoRatio);
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
    this.pitchLines = collectPitchLines(lines);
    this.applyPitchVisibility();
    this.renderLines();
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
    const show = this.pitchEnabled && this.pitchLines.some(p => p.notes.length > 0);
    if (this.pitchCanvas) this.pitchCanvas.style.display = show ? '' : 'none';
    if (this.pitchDividerEl) this.pitchDividerEl.style.display = show ? '' : 'none';
    this.win?.document.body.classList.toggle('ey-lane-active', show);
  }

  /** 디버그 신뢰도 색상 토글 즉시 반영 */
  setShowConfidence(enabled: boolean): void {
    this.showConfidence = enabled;
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
      const px = Math.min(320, Math.max(90, Math.round(startH + (startY - e.clientY))));
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

  /** 음정 레인 렌더 — TJ노래방식 오선지: 현재 가사 라인을 한 페이지로 펼쳐 그린다 */
  private renderPitch(now: number): void {
    const canvas = this.pitchCanvas;
    const win = this.win;
    if (!canvas || !win || !this.pitchEnabled || this.pitchLines.length === 0) return;

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

    // 현재 페이지 — 패널과 같은 엔진 라인 인덱스로 구동해 두 표시가 절대 어긋나지 않게 한다.
    // 현재 라인이 끝난 간주 구간에서는 다음 라인을 미리 보여준다.
    let pageIdx = this.index >= 0 ? Math.min(this.index, this.pitchLines.length - 1) : 0;
    if (pageIdx < this.pitchLines.length - 1 && now > this.pitchLines[pageIdx].end + 0.25) {
      pageIdx += 1;
    }
    const page = this.pitchLines[pageIdx];
    if (!page) return;

    const colors = this.pitchColors ?? (this.pitchColors = readPitchColors(win));

    // ── 세로 레이아웃: 위 오선지 영역 + 아래 가사/발음/번역 줄 (줄 수는 곡 단위 고정)
    const hasPron = this.pitchLines.some(p => p.line.pronunciation);
    const hasTr = this.pitchLines.some(p => p.line.translation);
    // 레인이 클수록 가사도 크게 — 참조(TJ)처럼 음절이 주인공급 크기
    const lyricH = Math.max(16, Math.min(26, Math.round(ch * 0.16)));
    const lyricPx = Math.max(13, Math.round(lyricH * 0.72));
    // 발음은 가사 음절 바로 아래 같은 시간축에 붙는다 — 가사와 크기 균형
    const pronPx = Math.max(11, Math.round(lyricPx * 0.8));
    const pronH = hasPron ? pronPx + 7 : 0;
    const trPx = Math.max(12, Math.min(16, Math.round(ch * 0.09)));
    const trH = hasTr ? trPx + 7 : 0;
    const padTop = 3;
    const staffH = Math.max(24, ch - padTop - lyricH - pronH - trH - 3);
    const g = staffH / 8; // 오선 간격 — 5선 4칸 + 위아래 덧줄 여백 2칸씩
    const staffTop = padTop + 2 * g;

    // 오선 5줄
    ctx.fillStyle = colors.faint;
    for (let i = 0; i < 5; i++) {
      ctx.fillRect(0, staffTop + i * g - 0.5, cw, 1);
    }

    // 시간→x: 페이지(현재 라인)를 레인 전체 폭에 펼친다
    const padX = 10;
    const pageSpan = Math.max(0.001, page.end - page.start);
    const x = (t: number) => padX + ((t - page.start) / pageSpan) * (cw - padX * 2);

    // midi→y: 반음당 g/2 (온음 ≈ 오선 한 칸), 라인 음역이 오선을 넘으면 영역에 맞게 압축
    const semiPx = Math.min(g / 2, (staffH - g) / Math.max(1, page.span));
    const staffMidY = staffTop + 2 * g;
    const y = (midi: number) => staffMidY - (midi - page.centerMidi) * semiPx;

    // ── 노트 막대: 듀레이션 반영 가로 길이 + 부른 부분 accent 필
    const noteH = Math.max(5, Math.min(12, g * 0.8));
    const noteR = Math.min(noteH / 2, 4);
    for (const n of page.notes) {
      const x1 = x(n.start);
      const x2 = x(n.end);
      const w = Math.max(3, x2 - x1 - 1); // 인접 노트와 1px 간격
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

      // 계이름 라벨 — 막대 위 (위 공간이 없으면 아래)
      if (x2 - x1 >= 18) {
        let labelY = top - 7;
        if (labelY < 6) labelY = top + noteH + 8;
        ctx.font = 'bold 10px system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = isCurrent ? colors.text : colors.dim;
        ctx.fillText(PITCH_NAMES_KO[((n.midi % 12) + 12) % 12], (x1 + x2) / 2, labelY);
      }
    }

    // ── 시간 위치 기반 라벨 배치 + 충돌 회피: 좌→우로 최소 간격을 보장하고,
    //    우측으로 밀린 만큼은 뒤에서부터 되민다 — 타이밍이 몰린 음절이 뭉치지 않게
    const placeRow = (items: { cx: number; w: number }[]): number[] => {
      const gap = 3;
      const xs = items.map(it => it.cx);
      for (let i = 0; i < xs.length; i++) {
        const minCx = i === 0
          ? padX + items[i].w / 2
          : xs[i - 1] + items[i - 1].w / 2 + gap + items[i].w / 2;
        if (xs[i] < minCx) xs[i] = minCx;
      }
      for (let i = xs.length - 1; i >= 0; i--) {
        const maxCx = i === xs.length - 1
          ? cw - padX - items[i].w / 2
          : xs[i + 1] - items[i + 1].w / 2 - gap - items[i].w / 2;
        if (xs[i] > maxCx) xs[i] = maxCx;
      }
      return xs;
    };
    const idealCx = (s: number, e: number) =>
      Math.max(padX, Math.min(cw - padX, (x(s) + x(e)) / 2));

    // ── 노트 아래 가사: word 토큰을 시간 위치에 정렬(충돌 회피), 부른 토큰은 accent
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    let ty = padTop + staffH;
    ctx.font = `bold ${lyricPx}px system-ui, sans-serif`;
    const words = page.line.words;
    if (words && words.length > 0) {
      const items = words.map(w => ({ w, cx: idealCx(w.start, w.end), width: ctx.measureText(w.word).width }));
      const xs = placeRow(items.map(it => ({ cx: it.cx, w: it.width })));
      items.forEach((it, i) => {
        let color = it.w.start <= now ? colors.accent : colors.text;
        // 디버그: 정렬 신뢰도(기하평균 확률, 로그 스케일 버킷) — 빨강<1e-4, 노랑<2e-2
        if (this.showConfidence && it.w.confidence != null && it.w.confidence < 2e-2) {
          color = it.w.confidence < 1e-4 ? '#ff6b6b' : '#ffd166';
        }
        ctx.fillStyle = color;
        ctx.fillText(it.w.word, xs[i], ty + lyricH * 0.55);
      });
    } else {
      // word 타이밍이 없는 라인 — 라인 텍스트 통째 중앙 폴백
      ctx.fillStyle = page.start <= now ? colors.accent : colors.text;
      ctx.fillText(page.line.text, cw / 2, ty + lyricH * 0.55, cw - padX * 2);
    }
    ty += lyricH;
    if (hasPron) {
      const segs = page.line.pronSegments;
      if (segs && segs.length > 0) {
        // 발음 음절도 같은 시간축 + 충돌 회피 — 노트/원문 바로 아래에서 대응이 읽힌다
        ctx.font = `${pronPx}px system-ui, sans-serif`;
        const items = segs.map(seg => ({ seg, cx: idealCx(seg.start, seg.end), width: ctx.measureText(seg.text).width }));
        const xs = placeRow(items.map(it => ({ cx: it.cx, w: it.width })));
        items.forEach((it, i) => {
          ctx.fillStyle = it.seg.start <= now ? colors.accent : colors.dim;
          ctx.fillText(it.seg.text, xs[i], ty + pronH * 0.5);
        });
      } else if (page.line.pronunciation) {
        this.renderPron(ctx, page, now, cw, padX, ty + pronH * 0.5, colors, x, pronPx);
      }
      ty += pronH;
    }
    if (hasTr && page.line.translation) {
      ctx.font = `${trPx}px system-ui, sans-serif`;
      ctx.fillStyle = colors.dim;
      ctx.fillText(page.line.translation, cw / 2, ty + trH * 0.5, cw - padX * 2);
    }

    // ── 현재 위치 수직 스위프 — 페이지 재생 중일 때만 (간주 미리보기에선 숨김)
    if (now >= page.start && now <= page.end) {
      const px = x(now);
      ctx.fillStyle = colors.dim;
      ctx.fillRect(px - 0.75, 0, 1.5, padTop + staffH);
      ctx.fillStyle = colors.accent;
      ctx.beginPath();
      ctx.moveTo(px - 4, 0);
      ctx.lineTo(px + 4, 0);
      ctx.lineTo(px, 6);
      ctx.closePath();
      ctx.fill();
    }
  }

  /**
   * 발음 줄 렌더 — 부른 만큼 accent로 칠한다.
   * 음절 타이밍(pronSegments)이 있으면 각 음절을 가사 음절과 **같은 시간축 위치**에
   * 배치해 노트·원문 바로 아래에서 대응이 읽히게 하고, 없으면 가운데 정렬
   * 문자열에 라인 진행률 그라데이션으로 폴백한다.
   */
  private renderPron(
    ctx: CanvasRenderingContext2D,
    page: PitchLine,
    now: number,
    cw: number,
    padX: number,
    py: number,
    colors: PitchColors,
    x: (t: number) => number,
    fontPx: number,
  ): void {
    const pron = page.line.pronunciation ?? '';
    if (!pron) return;
    ctx.font = `${fontPx}px system-ui, sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    const maxW = cw - padX * 2;

    const segs = page.line.pronSegments;
    if (segs && segs.length > 0) {
      for (const seg of segs) {
        const sx = Math.max(padX, Math.min(cw - padX, (x(seg.start) + x(seg.end)) / 2));
        ctx.fillStyle = seg.start <= now ? colors.accent : colors.dim;
        ctx.fillText(seg.text, sx, py);
      }
      return;
    }

    // 폴백: 가운데 정렬 + 진행률 비례 그라데이션 필
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
    if (!this.videoWrapEl) return;
    this.videoRatio = ratio;
    this.videoWrapEl.style.aspectRatio = 'auto';
    this.videoWrapEl.style.maxHeight = 'none';
    // shrink 1: 사용자가 지정한 비율이라도 가사 스테이지 min-height는 침범하지 못한다
    this.videoWrapEl.style.flex = `0 1 ${(ratio * 100).toFixed(1)}%`;
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
    if (this.pronEl) this.pronEl.textContent = current?.pronunciation ?? '';
    if (this.trEl) this.trEl.textContent = current?.translation ?? '';

    this.wordEls = [];
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
 * 라인별 페이지 구조 수집 — 라인 배열과 인덱스가 1:1로 정렬된다.
 * (renderPitch가 엔진 라인 인덱스로 페이지를 고르므로, 노트가 없는 라인도
 * 빈 오선 페이지로 포함해야 패널 하이라이트와 어긋나지 않는다.)
 */
function collectPitchLines(lines: LyricLine[]): PitchLine[] {
  return lines.map((line, i) => {
    const notes: PitchNote[] = [];
    if (line.words) {
      for (const word of line.words) {
        if (!word.notes) continue;
        for (const n of word.notes) notes.push({ midi: n.midi, start: n.start, end: n.end });
      }
    }
    if (line.notes) {
      for (const n of line.notes) notes.push({ midi: n.midi, start: n.start, end: n.end });
    }
    notes.sort((a, b) => a.start - b.start);

    let start = line.time ?? notes[0]?.start ?? 0;
    let end = line.endTime ?? lines[i + 1]?.time ?? (notes.length > 0 ? notes[notes.length - 1].end : start + 4);
    let lo = Infinity;
    let hi = -Infinity;
    for (const n of notes) {
      start = Math.min(start, n.start);
      end = Math.max(end, n.end);
      lo = Math.min(lo, n.midi);
      hi = Math.max(hi, n.midi);
    }
    if (line.words) {
      for (const w of line.words) {
        start = Math.min(start, w.start);
        end = Math.max(end, w.end);
      }
    }
    if (notes.length === 0) {
      lo = 62;
      hi = 66;
    }
    return {
      line,
      notes,
      start,
      end: Math.max(end, start + 0.5),
      centerMidi: (lo + hi) / 2,
      span: hi - lo,
    };
  });
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
