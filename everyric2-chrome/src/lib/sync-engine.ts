import type { LyricLine } from '../types';

export interface SyncHandlers {
  onLineChange: (index: number) => void;
  onTick?: (time: number) => void;
}

/**
 * video.currentTime을 가사 타임라인 인덱스로 변환하는 엔진.
 * rAF는 백그라운드 탭에서 스로틀되므로 timeupdate 이벤트를 병행해
 * PiP 창만 보이는 상황에서도 줄 단위 싱크가 유지되게 한다.
 *
 * 숨은 탭에서 남는 것은 timeupdate(~4Hz)뿐이라 이 엔진의 tick도 그 주기가 된다 —
 * **프레임 단위 렌더(PiP 레인·가사 채움)를 여기에 얹으면 안 된다.** PiP는 별도 최상위
 * 창이라 자기 rAF가 계속 돌므로, 이 tick은 상태 공급자로만 쓰고 그리기는 PiP가 한다
 * (ui/pip.ts의 state 필드 주석 참조).
 */
/** 시크 되돌림 가드 판정 창(ms) — 근거는 seekToVideoTime 주석 */
const SEEK_GUARD_MS = 1200;

export class SyncEngine {
  private video: HTMLVideoElement | null = null;
  private times: number[] = [];
  private offset = 0;
  /** 직전 시크의 되돌림 감시 상태 — seekToVideoTime 주석 참조 */
  private seekGuard: { target: number; prev: number; until: number } | null = null;
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
    this.seekGuard = null;
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
    this.seekToVideoTime(Math.max(0, time - this.offset));
  }

  /**
   * 비디오 시간 기준 시크 (진행바 등 비율 UI용) — 되돌림 가드가 함께 걸린다.
   *
   * 실사용 보고(2026-07-27): 일시정지↔재생을 오간 뒤 가사 줄 클릭으로 시크하면 그
   * 위치로 가지 않고 «시크 직전 위치»로 되돌아가 재생됐다. 확장 코드에서 currentTime을
   * 쓰는 곳은 이 클래스와 진행바 클릭 둘뿐이고 둘 다 시간을 저장·복원하지 않으므로,
   * 되돌리는 주체는 유튜브 플레이어 자신이다(#movie_player는 격리 월드라 공식 API로
   * 시크할 수 없어 raw currentTime 대입이 유일한 경로다 — yt-player.ts 주석).
   *
   * 가드는 «시크 직전 위치 근처(±2s)로의 복귀»만 되돌림으로 판정한다 — 목표 도달·
   * 사용자의 새 수동 시크(임의 위치)는 건드리지 않고, 재적용은 1회뿐이라 최악의 경우에도
   * 유튜브 진행바 조작과 한 번 겨루고 끝난다. 판정 창은 시크 후 1.2초지만 일시정지
   * 동안에는 계속 연장된다 — 「멈춘 채 시크 → 한참 뒤 재생」 흐름에서 되돌림은 재생
   * 시점에 일어나기 때문이다. 발동 시 console.debug로 남겨 원인 추적 증거를 만든다.
   */
  seekToVideoTime(sec: number): void {
    if (!this.video) return;
    this.seekGuard = {
      target: sec,
      prev: this.video.currentTime,
      until: performance.now() + SEEK_GUARD_MS,
    };
    this.video.currentTime = sec;
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
      // 다음 프레임을 먼저 예약한 뒤 tick()을 부른다 (pip.ts의 startFrameLoop와 같은 순서).
      // tick()은 onLineChange/onTick 핸들러를 통해 overlay·pip의 DOM 렌더 코드를 부르므로,
      // 그쪽에서 예외가 나면 예약을 tick() 뒤에 뒀을 때 이 루프가 그 프레임에서 영구히
      // 끊긴다 — running은 true로 남고 video도 그대로라 watchVideoBinding의 재바인딩
      // 조건(video !== engine.getVideo())도 걸리지 않아 같은 영상에서는 새로고침 전까지
      // 아무도 다시 살리지 못한다. 예약을 먼저 해 두면 예외가 나도 다음 프레임에서 이어진다.
      this.rafId = requestAnimationFrame(loop);
      try {
        this.tick();
      } catch (error) {
        console.error('[everyric] sync-engine tick 예외 — 이번 프레임만 건너뛰고 루프는 계속 돈다', error);
      }
    };
    this.rafId = requestAnimationFrame(loop);
  }

  private checkSeekGuard(videoTime: number): void {
    const g = this.seekGuard;
    if (!g || !this.video) return;
    if (this.video.paused) {
      // 일시정지 중에는 판정을 미룬다 — 되돌림은 재생 재개 시점에 일어난다
      g.until = performance.now() + SEEK_GUARD_MS;
      return;
    }
    if (performance.now() > g.until) {
      this.seekGuard = null;
      return;
    }
    if (Math.abs(videoTime - g.target) < 1.5) {
      this.seekGuard = null; // 시크 안착 — 정상
      return;
    }
    // 목표엔 못 갔는데 시크 «직전» 위치 근처로 돌아와 있다 — 플레이어가 시크를 삼켰다.
    // 짧은 시크(<3s)는 위 안착 판정과 겹쳐 오판 여지가 있어 제외한다.
    if (Math.abs(videoTime - g.prev) < 2.0 && Math.abs(g.target - g.prev) > 3.0) {
      const target = g.target;
      this.seekGuard = null; // 재적용은 1회뿐 — 유튜브 진행바 조작과 무한히 겨루지 않는다
      console.debug(
        `[everyric] seek reverted by player (target ${target.toFixed(1)}s, `
        + `back at ${videoTime.toFixed(1)}s near pre-seek ${g.prev.toFixed(1)}s); reapplying once`,
      );
      this.video.currentTime = target;
    }
  }

  private tick(): void {
    if (!this.video || !this.running) return;
    this.checkSeekGuard(this.video.currentTime);
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
