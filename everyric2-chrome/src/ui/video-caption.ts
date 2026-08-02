import type { LyricLine } from '../types';
import { resolvedPronunciation, type PronScript } from '../lib/lang';

/**
 * 유튜브 플레이어 화면 자체에 가사를 자막처럼 띄우는 모듈 (Language Reactor식,
 * 운영자 요청 2026-08-03). 가사창(패널)과 독립적으로 켜고 끌 수 있는 표시 모듈이다.
 *
 * - 호스트는 #movie_player **안**에 붙인다 — 전체화면 전환 시 플레이어 서브트리만
 *   전체화면이 되므로, 밖에 붙이면 전체화면에서 자막이 사라진다.
 * - 켜져 있는 동안 유튜브 자체 자막(.ytp-caption-window-container)은 숨긴다 —
 *   같은 자리에 둘이 겹치면 어느 쪽도 못 읽는다. 끄면 즉시 되살린다(클래스 토글).
 * - 유튜브 페이지 DOM에 직접 붙으므로 스타일은 전부 인라인 — 페이지 CSS 개편의
 *   영향을 받지 않고, 우리 스타일이 페이지에 새지도 않는다(주입 <style>은 자막
 *   숨김 셀렉터 한 줄뿐).
 * - SPA 이동에서 유튜브가 플레이어 DOM을 재구성하면 호스트가 떨어져 나갈 수 있다 —
 *   updateTime마다 isConnected를 확인해 재부착한다(watchVideoBinding과 같은 이유).
 */
export class VideoCaption {
  private host: HTMLDivElement | null = null;
  private lineEl: HTMLDivElement | null = null;
  private pronEl: HTMLDivElement | null = null;
  private trEl: HTMLDivElement | null = null;
  private styleEl: HTMLStyleElement | null = null;
  private lines: LyricLine[] = [];
  private currentIndex = -1;
  private enabled = false;
  private pronScript: PronScript = 'hangul';
  private showPron = true;
  private showTr = true;

  setEnabled(on: boolean): void {
    this.enabled = on;
    if (on) {
      this.mount();
    } else {
      this.unmount();
    }
  }

  setLines(lines: LyricLine[]): void {
    this.lines = lines;
    this.currentIndex = -1;
    this.render(null);
  }

  applyDisplay(pronScript: PronScript, showPron: boolean, showTr: boolean): void {
    this.pronScript = pronScript;
    this.showPron = showPron;
    this.showTr = showTr;
    // 표시 방식이 바뀌면 현재 줄을 다시 그린다 (다음 tick을 기다리지 않는다)
    const index = this.currentIndex;
    this.currentIndex = -1;
    this.render(index >= 0 ? this.lines[index] ?? null : null, index);
  }

  updateTime(time: number): void {
    if (!this.enabled || this.lines.length === 0) return;
    // 유튜브가 SPA 이동으로 플레이어 DOM을 재구성하면 호스트가 조용히 떨어져 나간다
    if (!this.host?.isConnected) this.mount();
    let index = -1;
    for (let i = 0; i < this.lines.length; i++) {
      const line = this.lines[i];
      if (line.time !== null && line.time <= time) index = i;
      else if (line.time !== null && line.time > time) break;
    }
    if (index === this.currentIndex) return;
    this.render(index >= 0 ? this.lines[index] : null, index);
  }

  destroy(): void {
    this.unmount();
    this.lines = [];
  }

  private mount(): void {
    const player = document.querySelector<HTMLElement>('#movie_player');
    if (!player) return; // 플레이어가 아직 없다 — 다음 updateTime에서 재시도된다
    if (!this.styleEl || !this.styleEl.isConnected) {
      // 유일한 페이지 주입 스타일 — 우리 자막이 켜진 동안만 유튜브 자막을 숨긴다
      this.styleEl = document.createElement('style');
      this.styleEl.textContent =
        '.ey-video-caption-on .ytp-caption-window-container { display: none !important; }';
      document.head.append(this.styleEl);
    }
    player.classList.add('ey-video-caption-on');
    if (this.host?.isConnected) return;
    this.host = document.createElement('div');
    // 자막 관례 위치: 하단 중앙, 조작을 막지 않게 pointer-events 없음
    this.host.style.cssText = [
      'position:absolute', 'left:50%', 'bottom:8%', 'transform:translateX(-50%)',
      'max-width:86%', 'z-index:60', 'pointer-events:none', 'text-align:center',
      "font-family:'YouTube Sans',Roboto,'Noto Sans KR',sans-serif",
    ].join(';');
    const shared = 'color:#fff;background:rgba(8,8,8,0.75);border-radius:4px;'
      + 'padding:2px 10px;display:inline-block;white-space:pre-wrap;'
      + 'text-shadow:0 0 4px rgba(0,0,0,0.9)';
    this.lineEl = document.createElement('div');
    this.lineEl.style.cssText = `${shared};font-size:2.1vmin;font-weight:600;line-height:1.4`;
    this.pronEl = document.createElement('div');
    // 발음 줄은 패널과 같은 앰버 계열로 원문과 구분한다
    this.pronEl.style.cssText = `${shared};font-size:1.6vmin;opacity:0.92;margin-top:2px;color:#ffd98e`;
    this.trEl = document.createElement('div');
    this.trEl.style.cssText = `${shared};font-size:1.7vmin;opacity:0.95;margin-top:2px`;
    for (const el of [this.lineEl, this.pronEl, this.trEl]) {
      const wrap = document.createElement('div');
      wrap.append(el);
      this.host.append(wrap);
    }
    player.append(this.host);
    this.render(this.currentIndex >= 0 ? this.lines[this.currentIndex] ?? null : null, this.currentIndex);
  }

  private unmount(): void {
    document.querySelector('#movie_player')?.classList.remove('ey-video-caption-on');
    this.styleEl?.remove();
    this.styleEl = null;
    this.host?.remove();
    this.host = null;
    this.lineEl = this.pronEl = this.trEl = null;
  }

  private render(line: LyricLine | null, index = -1): void {
    this.currentIndex = index;
    if (!this.lineEl || !this.pronEl || !this.trEl) return;
    const show = (el: HTMLDivElement, text: string | undefined) => {
      el.textContent = text ?? '';
      el.style.display = text ? 'inline-block' : 'none';
    };
    show(this.lineEl, line?.text);
    show(this.pronEl, this.showPron && line ? resolvedPronunciation(line, this.pronScript) : undefined);
    show(this.trEl, this.showTr ? line?.translation : undefined);
  }
}
