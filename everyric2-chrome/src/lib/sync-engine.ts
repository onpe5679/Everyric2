import type { LyricLine } from '../types';

export interface SyncHandlers {
  onLineChange: (index: number) => void;
  onTick?: (time: number) => void;
}

/**
 * video.currentTime을 가사 타임라인 인덱스로 변환하는 엔진.
 * rAF는 백그라운드 탭에서 스로틀되므로 timeupdate 이벤트를 병행해
 * PiP 창만 보이는 상황에서도 줄 단위 싱크가 유지되게 한다.
 */
export class SyncEngine {
  private video: HTMLVideoElement | null = null;
  private times: number[] = [];
  private offset = 0;
  // findIndex()가 "첫 라인 이전"을 -1로 반환하므로, 최초 tick에서도
  // onLineChange(-1)가 발화되도록 초기값은 -2를 쓴다
  private lastIndex = -2;
  private rafId = 0;
  private running = false;
  private handlers: SyncHandlers = { onLineChange: () => {} };

  start(video: HTMLVideoElement, lines: LyricLine[], handlers: SyncHandlers): void {
    this.stop();
    this.video = video;
    this.times = lines.map(l => l.time ?? 0);
    this.handlers = handlers;
    this.lastIndex = -2;
    this.running = true;

    video.addEventListener('timeupdate', this.handleTimeEvent);
    video.addEventListener('seeking', this.handleTimeEvent);
    document.addEventListener('visibilitychange', this.handleVisibility);
    this.scheduleLoop();
  }

  stop(): void {
    this.running = false;
    cancelAnimationFrame(this.rafId);
    if (this.video) {
      this.video.removeEventListener('timeupdate', this.handleTimeEvent);
      this.video.removeEventListener('seeking', this.handleTimeEvent);
    }
    document.removeEventListener('visibilitychange', this.handleVisibility);
    this.video = null;
    this.times = [];
  }

  isRunning(): boolean {
    return this.running;
  }

  /** 현재 바인딩된 video — 워치독이 엘리먼트 교체를 감지하는 데 사용 */
  getVideo(): HTMLVideoElement | null {
    return this.video;
  }

  isPaused(): boolean {
    return this.video?.paused ?? true;
  }

  setOffset(sec: number): void {
    this.offset = sec;
    this.resync();
  }

  /** onLineChange를 강제로 재발화 (PiP 오픈 직후 등 뷰가 새로 붙었을 때) */
  resync(): void {
    this.lastIndex = -2;
    this.tick();
  }

  getOffset(): number {
    return this.offset;
  }

  /** time은 가사 타임라인 기준. 실제 비디오 시각으로 역변환해 시크한다. */
  seekTo(time: number): void {
    if (this.video) this.video.currentTime = Math.max(0, time - this.offset);
  }

  getCurrentTime(): number {
    return (this.video?.currentTime ?? 0) + this.offset;
  }

  getDuration(): number {
    const d = this.video?.duration ?? 0;
    return Number.isFinite(d) ? d : 0;
  }

  private handleTimeEvent = (): void => {
    this.tick();
  };

  private handleVisibility = (): void => {
    if (!document.hidden) this.scheduleLoop();
  };

  private scheduleLoop(): void {
    cancelAnimationFrame(this.rafId);
    const loop = (): void => {
      if (!this.running || document.hidden) return;
      this.tick();
      this.rafId = requestAnimationFrame(loop);
    };
    this.rafId = requestAnimationFrame(loop);
  }

  private tick(): void {
    if (!this.video || !this.running) return;
    const t = this.video.currentTime + this.offset;
    this.handlers.onTick?.(t);
    const index = this.findIndex(t);
    if (index !== this.lastIndex) {
      this.lastIndex = index;
      this.handlers.onLineChange(index);
    }
  }

  private findIndex(t: number): number {
    let lo = 0;
    let hi = this.times.length - 1;
    let result = -1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (this.times[mid] <= t) {
        result = mid;
        lo = mid + 1;
      } else {
        hi = mid - 1;
      }
    }
    return result;
  }
}
