import type { LyricLine } from '../types';
import { resolvedPronunciation, resolvedPronSegments, shouldShowPron, type PronScript } from '../lib/lang';
import { appendKaraokeSpans, appendTimedSpans } from './karaoke';

// 스케일 배율의 기준값(기존 하드코딩 값과 동일 — fontScale 1일 때 지금과 완전히 같다)
const BASE_LINE_FONT_VMIN = 2.1;
const BASE_PRON_FONT_VMIN = 1.6;
const BASE_TR_FONT_VMIN = 1.7;
// 부른 음절/단어 강조색 — 패널(overlay.css --ey-accent)과 같은 색이지만, 이 모듈은
// 유튜브 페이지 DOM에 직접 붙어 그 CSS 변수 스코프 밖이라 리터럴로 고정한다
const SUNG_COLOR = '#ffb02e';

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
 *   숨김 셀렉터 + 카라오케 sung 색상 두 줄뿐).
 * - SPA 이동에서 유튜브가 플레이어 DOM을 재구성하면 호스트가 떨어져 나갈 수 있다 —
 *   updateTime마다 isConnected를 확인해 재부착한다(watchVideoBinding과 같은 이유).
 * - 카라오케 채움: 패널(overlay.ts)·PIP(pip.ts)와 같은 공유 헬퍼(karaoke.ts의
 *   appendKaraokeSpans/appendTimedSpans)로 단어/음절 span을 만들고, activeWordEls에
 *   {start, el}로 모아 매 updateTime마다 start<=time이면 'sung' 클래스를 토글한다.
 *   타이밍 데이터가 없는 줄은 그 헬퍼들이 알아서 통짜 텍스트로 폴백한다.
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
  private hidePronForEnglish = false;
  private showTr = true;
  private fontScale = 1;
  private bgOpacity = 0.75;
  private lastTime = 0;
  private activeWordEls: { start: number; el: HTMLElement }[] = [];
  /** 마지막 mount() 재시도 시각 — updateTime은 최대 60Hz로 도는데 호스트가 안 붙어
   *  있으면 매 tick 재시도해 DOM 삽입 시도가 프레임마다 반복됐다(5fps 감사 #4).
   *  500ms 미만이면 건너뛴다. */
  private lastMountAttempt = 0;

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
    // 코덱스 감사 Med(2026-08-03): 싱크 없는 영상으로 넘어가면 자체 자막은 비는데
    // 유튜브 자막 숨김 클래스는 unmount에서만 지워져 **둘 다 안 보이는** 상태로
    // 방치됐다 — 표시할 라인이 있을 때만 유튜브 자막을 숨긴다.
    this.syncYtSuppression();
  }

  applyDisplay(pronScript: PronScript, showPron: boolean, showTr: boolean, hidePronForEnglish: boolean): void {
    this.pronScript = pronScript;
    this.showPron = showPron;
    this.hidePronForEnglish = hidePronForEnglish;
    this.showTr = showTr;
    // 표시 방식이 바뀌면 현재 줄을 다시 그린다 (다음 tick을 기다리지 않는다)
    const index = this.currentIndex;
    this.currentIndex = -1;
    this.render(index >= 0 ? this.lines[index] ?? null : null, index);
    // 재구성된 span은 전부 unsung 상태로 시작한다 — 마지막 시각으로 즉시 채움을 다시 입힌다
    this.applyFill(this.lastTime);
  }

  /** 자막 글자 크기·배경 불투명도 — 설정값이 바뀔 때마다 호출, 다음 render 이후에도 유지된다 */
  applyStyle(style: { fontScale: number; bgOpacity: number }): void {
    this.fontScale = style.fontScale;
    this.bgOpacity = style.bgOpacity;
    this.applyStyleToEls();
  }

  updateTime(time: number): void {
    if (!this.enabled || this.lines.length === 0) return;
    // 유튜브가 SPA 이동으로 플레이어 DOM을 재구성하면 호스트가 조용히 떨어져 나간다.
    // updateTime은 최대 60Hz(rAF)로 도는데, 재시도할 곳(#movie_player)이 계속 없으면
    // 매 tick DOM 삽입을 시도해 프레임을 깎는다 — 500ms 백오프로 재시도 빈도를 낮춘다.
    if (!this.host?.isConnected) {
      const now = performance.now();
      if (now - this.lastMountAttempt >= 500) {
        this.lastMountAttempt = now;
        this.mount();
      }
    }
    this.lastTime = time;
    let index = -1;
    for (let i = 0; i < this.lines.length; i++) {
      const line = this.lines[i];
      if (line.time !== null && line.time <= time) index = i;
      else if (line.time !== null && line.time > time) break;
    }
    if (index !== this.currentIndex) {
      this.render(index >= 0 ? this.lines[index] : null, index);
    }
    // 값싼 경로: 줄이 그대로여도 매 tick 실행 — 부른 만큼 활성 줄 안에서 색이 차오른다
    this.applyFill(time);
  }

  /** activeWordEls(현재 줄의 단어/음절 span)에 부른 만큼 'sung' 클래스를 입힌다 */
  private applyFill(time: number): void {
    for (const { start, el } of this.activeWordEls) {
      el.classList.toggle('sung', start <= time);
    }
  }

  destroy(): void {
    this.unmount();
    this.lines = [];
  }

  private mount(): void {
    const player = document.querySelector<HTMLElement>('#movie_player');
    if (!player) return; // 플레이어가 아직 없다 — 다음 updateTime에서 재시도된다
    // 이전 시도의 잔재(끊어진 호스트/스타일)를 먼저 정리한다 — 그대로 둔 채 새로 만들면
    // 참조 없는 detached 엘리먼트가 재삽입 시도마다 쌓인다(5fps 감사 #4).
    if (this.host && !this.host.isConnected) {
      this.host.remove();
      this.host = null;
      this.lineEl = this.pronEl = this.trEl = null;
    }
    if (this.styleEl && !this.styleEl.isConnected) {
      this.styleEl.remove();
      this.styleEl = null;
    }
    if (!this.styleEl || !this.styleEl.isConnected) {
      // 유일한 페이지 주입 스타일 — 유튜브 자막 숨김 + 카라오케 sung 강조색 두 규칙뿐
      this.styleEl = document.createElement('style');
      this.styleEl.textContent =
        '.ey-video-caption-on .ytp-caption-window-container { display: none !important; }'
        + '.ey-vc-word, .ey-vc-syl { transition: color 0.12s; }'
        + `.ey-vc-word.sung, .ey-vc-syl.sung { color: ${SUNG_COLOR}; }`;
      document.head.append(this.styleEl);
    }
    this.syncYtSuppression();
    if (this.host?.isConnected) return;
    this.host = document.createElement('div');
    // 식별용 클래스 — 스타일은 인라인이라 CSS로 쓰이진 않지만, 이게 없으면 이 호스트가
    // 유튜브 DOM 안에서 «이름 없는 div»라 디버깅으로도 E2E 선택자로도 못 짚는다(실측:
    // 자막 검사가 요소를 못 찾아 검증 자체가 건너뛰어졌다).
    this.host.className = 'ey-video-caption';
    // 자막 관례 위치: 하단 중앙, 조작을 막지 않게 pointer-events 없음.
    // z-index는 유튜브가 일시정지 때 띄우는 "More videos" 오버레이(.ytp-pause-overlay)보다
    // 높아야 한다 — 안 그러면 그 오버레이가 자막 호스트를 덮어 일시정지 중 자막이 사라진
    // 것처럼 보인다(실사용 제보). 유튜브 자체 오버레이 스택보다 확실히 위로.
    this.host.style.cssText = [
      'position:absolute', 'left:50%', 'bottom:8%', 'transform:translateX(-50%)',
      'max-width:86%', 'z-index:2000000', 'pointer-events:none', 'text-align:center',
      "font-family:'YouTube Sans',Roboto,'Noto Sans KR',sans-serif",
    ].join(';');
    // font-size·background는 applyStyleToEls가 fontScale·bgOpacity에 맞춰 별도로 채운다
    // (display는 render가 매번 토글하므로 여기 cssText에 넣지 않는다 — 통짜로 덮어쓰면 지워진다)
    const shared = 'border-radius:4px;padding:2px 10px;display:inline-block;'
      + 'white-space:pre-wrap;text-shadow:0 0 4px rgba(0,0,0,0.9)';
    this.lineEl = document.createElement('div');
    this.lineEl.style.cssText = `${shared};color:#fff;font-weight:600;line-height:1.4`;
    this.pronEl = document.createElement('div');
    // 발음 줄 기본색은 차분한 청회색 — 예전 앰버(#ffd98e)는 sung 강조색(#ffb02e)과
    // 동계열이라 카라오케 진행이 안 보였다(실사용 제보 "노란 자막에 노란 하이라이트").
    // 원문(흰색)과도, 진행(노랑)과도 갈리는 제3색이어야 한다.
    this.pronEl.style.cssText = `${shared};color:#a9c4e6;opacity:0.95;margin-top:2px`;
    this.trEl = document.createElement('div');
    this.trEl.style.cssText = `${shared};color:#fff;opacity:0.95;margin-top:2px`;
    for (const el of [this.lineEl, this.pronEl, this.trEl]) {
      const wrap = document.createElement('div');
      wrap.append(el);
      this.host.append(wrap);
    }
    player.append(this.host);
    this.applyStyleToEls();
    this.render(this.currentIndex >= 0 ? this.lines[this.currentIndex] ?? null : null, this.currentIndex);
    this.applyFill(this.lastTime);
  }

  /** fontScale·bgOpacity를 세 줄에 적용 — display는 건드리지 않아 render의 토글과 충돌하지 않는다 */
  private applyStyleToEls(): void {
    if (!this.lineEl || !this.pronEl || !this.trEl) return;
    const bg = `rgba(8,8,8,${this.bgOpacity})`;
    this.lineEl.style.background = bg;
    this.pronEl.style.background = bg;
    this.trEl.style.background = bg;
    this.lineEl.style.fontSize = `${BASE_LINE_FONT_VMIN * this.fontScale}vmin`;
    this.pronEl.style.fontSize = `${BASE_PRON_FONT_VMIN * this.fontScale}vmin`;
    this.trEl.style.fontSize = `${BASE_TR_FONT_VMIN * this.fontScale}vmin`;
  }

  /** 유튜브 자막 숨김은 "우리 자막이 실제로 나올 때"만 — 모듈 on + 표시할 라인 존재 */
  private syncYtSuppression(): void {
    const on = this.enabled && this.lines.length > 0;
    document.querySelector('#movie_player')?.classList.toggle('ey-video-caption-on', on);
  }

  private unmount(): void {
    document.querySelector('#movie_player')?.classList.remove('ey-video-caption-on');
    this.styleEl?.remove();
    this.styleEl = null;
    this.host?.remove();
    this.host = null;
    this.lineEl = this.pronEl = this.trEl = null;
    this.activeWordEls = [];
  }

  private render(line: LyricLine | null, index = -1): void {
    this.currentIndex = index;
    if (!this.lineEl || !this.pronEl || !this.trEl) return;
    this.activeWordEls = [];

    // 원문 줄 — appendKaraokeSpans가 실 단어 타이밍 → 합성 글자 타이밍(pronSegments/라인
    // 구간 비례 배분) → 통짜 텍스트 순으로 폴백을 알아서 고른다(패널·PIP와 동일 헬퍼)
    this.lineEl.replaceChildren();
    if (line && line.text) {
      this.lineEl.style.display = 'inline-block';
      appendKaraokeSpans(this.lineEl, line, word => {
        const el = document.createElement('span');
        el.className = 'ey-vc-word';
        el.textContent = word.word;
        this.activeWordEls.push({ start: word.start, el });
        return el;
      });
    } else {
      this.lineEl.style.display = 'none';
    }

    // 발음 줄 — 음절 세그(script별)가 있으면 카라오케 span, 없으면 통짜 텍스트로 폴백
    this.pronEl.replaceChildren();
    const pronVisible = line
      ? shouldShowPron(line.text, { showPronunciation: this.showPron, hidePronForEnglish: this.hidePronForEnglish })
      : false;
    const pronText = pronVisible && line ? resolvedPronunciation(line, this.pronScript) : undefined;
    if (pronText) {
      this.pronEl.style.display = 'inline-block';
      const segs = line ? resolvedPronSegments(line, this.pronScript) : undefined;
      const mapped = segs && segs.length > 0
        ? appendTimedSpans(this.pronEl, pronText, segs, s => s.text, seg => {
            const el = document.createElement('span');
            el.className = 'ey-vc-syl';
            el.textContent = seg.text;
            this.activeWordEls.push({ start: seg.start, el });
            return el;
          })
        : 0;
      if (mapped === 0) this.pronEl.textContent = pronText;
    } else {
      this.pronEl.style.display = 'none';
    }

    // 번역 줄 — 타이밍 데이터가 없어 기존과 동일하게 통짜 텍스트
    const trText = this.showTr ? line?.translation : undefined;
    this.trEl.textContent = trText ?? '';
    this.trEl.style.display = trText ? 'inline-block' : 'none';
  }
}
