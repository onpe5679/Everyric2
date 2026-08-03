import type { LyricLine } from '../types';
import { resolvedPronSegments, resolvedPronunciation, shouldShowPron, type PronScript } from '../lib/lang';
import { h } from './dom';
import { appendKaraokeSpans, appendTimedSpans } from './karaoke';

/**
 * 가사 줄 하나의 DOM(.ey-line 구조)을 만드는 공용 렌더러.
 *
 * 메인 가사창(overlay.ts)과 PIP 가사 목록 컬럼(pip.ts)이 이 모듈을 공유해 같은 클래스
 * 구조(.ey-line/.ey-word/.ey-line-pron/.ey-line-tr/.ey-pron-syl)를 만든다 — PIP는 메인과
 * 같은 overlay.css 전문을 주입받으므로(pip.open(cssText)), 같은 DOM이면 같은 모양이 난다.
 * 리스트 스크롤·시트 같은 패널 로직, 클릭 시크·title·dataset 등 문맥별 장식은 이 모듈의
 * 책임이 아니다 — 호출부가 반환된 엘리먼트에 덧붙인다.
 */

/** buildLineEl/buildPronEl이 발음 표시 여부 판정에 쓰는 설정 조각 — lib/lang.ts
 *  shouldShowPron과 계약이 같다(전체 끔 + 영어만 끔). */
export interface LineRenderSettings {
  showPronunciation: boolean;
  hidePronForEnglish: boolean;
}

/** 카라오케 채움(sung) 갱신 대상 — 단어/발음 음절 스팬과 시작 시각.
 *  overlay.ts activeWordEls, pip.ts wordEls와 같은 규약이다. */
export interface FillTarget {
  start: number;
  el: HTMLElement;
}

/**
 * 라인의 발음 표기 엘리먼트(.ey-line-pron)를 만든다 — 음절 타이밍(pronSegments)이 있으면
 * 부른 만큼 색이 차오르게 스팬으로(사이 텍스트는 appendTimedSpans가 인접 span에 끼워
 * 넣어 흰 글자 없이 칠해진다), 없으면 통짜 텍스트로. 발음이 없으면 null(호출부가 append를
 * 생략하게).
 *
 * fillTargets를 주면 만들어진 음절 스팬을 그 배열에도 밀어넣는다 — 호출부가 카라오케
 * 채움 갱신 때 원문 단어 스팬과 함께 순회할 수 있게 한다.
 */
export function buildPronEl(
  line: LyricLine,
  pronScript: PronScript,
  settings: LineRenderSettings,
  fillTargets?: FillTarget[],
): HTMLDivElement | null {
  const pron = resolvedPronunciation(line, pronScript);
  if (!pron) return null;
  if (!shouldShowPron(line.text, settings, pron)) return null;
  const segs = resolvedPronSegments(line, pronScript);
  const pronEl = h('div', { className: 'ey-line-pron', attrs: { dir: 'auto' } });
  const mapped = segs && segs.length > 0
    ? appendTimedSpans(pronEl, pron, segs, s => s.text, seg => {
        const span = h('span', {
          className: 'ey-pron-syl',
          text: seg.text,
          attrs: { 'data-start': String(seg.start) },
        });
        fillTargets?.push({ start: seg.start, el: span });
        return span;
      })
    : 0;
  if (mapped === 0) pronEl.replaceChildren(pron);
  return pronEl;
}

/**
 * 가사 한 줄의 DOM(원문 카라오케 스팬 + 발음 줄 + 번역 줄)을 만든다.
 * 반환된 fillTargets는 원문 단어 + 발음 음절 스팬 전체 — 호출부가 카라오케 채움 갱신에 쓴다.
 */
export function buildLineEl(
  line: LyricLine,
  pronScript: PronScript,
  settings: LineRenderSettings,
): { el: HTMLDivElement; fillTargets: FillTarget[] } {
  const el = h('div', { className: 'ey-line', attrs: { dir: 'auto' } });
  const fillTargets: FillTarget[] = [];

  // words가 없어도 호출 — appendKaraokeSpans가 음절 타이밍/라인 구간 비례
  // 배분으로 폴백해, 라인이 한 번에 통째로 켜지는 표시를 피한다
  appendKaraokeSpans(el, line, word => {
    // 신뢰도 등급 클래스 — .ey-show-conf(디버그 모드)에서만 색이 입혀진다.
    // 값은 CTC 프레임 로그확률의 기하평균(0~1) — 절대값이 작아 로그 스케일로 버킷:
    // <1e-4(로그 -9 이하)=낮음, <2e-2(로그 -4 이하)=중간
    const conf = word.confidence;
    // 버킷 색은 레인(pip.ts confBucketColor)과 동일: 빨강<1e-4, 노랑<2e-2, 초록=양호
    const confClass = conf == null ? '' : conf < 1e-4 ? ' ey-conf-low' : conf < 2e-2 ? ' ey-conf-mid' : ' ey-conf-ok';
    const span = h('span', { className: `ey-word${confClass}`, text: word.word, attrs: { 'data-start': String(word.start) } });
    fillTargets.push({ start: word.start, el: span });
    return span;
  });

  const pronEl = buildPronEl(line, pronScript, settings, fillTargets);
  if (pronEl) el.append(pronEl);
  if (line.translation) el.append(h('div', { className: 'ey-line-tr', text: line.translation, attrs: { dir: 'auto' } }));

  return { el, fillTargets };
}

/** 채움 대상 배열을 현재 시각 기준으로 갱신 — start<=time인 것만 칠한다(sung).
 *  overlay.ts updateTime의 activeWordEls 순회, PIP 목록 컬럼의 활성 줄 순회가 공유한다. */
export function updateFillTargets(targets: readonly FillTarget[], time: number): void {
  for (const { start, el } of targets) el.classList.toggle('sung', start <= time);
}

/** 엘리먼트 안의 .ey-word/.ey-pron-syl 전체를 채움 대상 배열로 수집한다 — 활성 줄이
 *  바뀔 때(overlay.ts highlightLine) activeWordEls를 다시 만드는 데 쓴다. */
export function collectFillTargets(el: HTMLElement): FillTarget[] {
  const targets: FillTarget[] = [];
  for (const w of el.querySelectorAll<HTMLElement>('.ey-word, .ey-pron-syl')) {
    targets.push({ start: Number(w.dataset.start), el: w });
  }
  return targets;
}

/** 줄 하나의 채움 상태를 전부 켜거나 끈다 — 과거/미래 줄 일괄 처리(overlay.ts fillUpTo)에 쓴다. */
export function setElFilled(el: HTMLElement, filled: boolean): void {
  for (const w of el.querySelectorAll<HTMLElement>('.ey-word, .ey-pron-syl')) {
    w.classList.toggle('sung', filled);
  }
}
