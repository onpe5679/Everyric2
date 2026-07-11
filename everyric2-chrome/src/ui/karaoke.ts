import type { LyricLine, WordSegment } from '../types';

/**
 * 본문 텍스트 위에 타이밍 토큰을 위치 매핑해 카라오케 span을 구성하는 공통 헬퍼.
 *
 * 토큰을 본문에서 찾아 span으로 감싸되, **토큰 사이/앞뒤의 나머지 텍스트(공백·
 * 문장부호·매핑 실패 글자)는 인접한 토큰 span 안에 끼워 넣는다** — span 밖의
 * 텍스트 노드는 sung 색이 영원히 입혀지지 않아 흰 글자로 남는 버그가 있었다.
 * 사이 텍스트는 직전 토큰과 함께, 첫 토큰 앞 텍스트는 첫 토큰과 함께 칠해진다.
 *
 * 반환: 매핑된 토큰 수 (0이면 호출부가 폴백 표시).
 */
export function appendTimedSpans<T>(
  el: HTMLElement,
  text: string,
  tokens: readonly T[],
  tokenText: (t: T) => string,
  makeEl: (t: T) => HTMLElement,
): number {
  let pos = 0;
  let mapped = 0;
  let prevEl: HTMLElement | null = null;
  let pendingLead = '';
  for (const token of tokens) {
    const tt = tokenText(token);
    if (!tt) continue;
    const idx = text.indexOf(tt, pos);
    if (idx === -1) continue; // 표기 차이로 본문에서 못 찾는 토큰은 건너뜀
    if (idx > pos) {
      const inter = text.slice(pos, idx);
      if (prevEl) prevEl.append(inter);
      else pendingLead += inter;
    }
    const spanEl = makeEl(token);
    if (pendingLead) {
      spanEl.prepend(pendingLead);
      pendingLead = '';
    }
    el.append(spanEl);
    prevEl = spanEl;
    pos = idx + tt.length;
    mapped++;
  }
  if (mapped > 0 && pos < text.length && prevEl) {
    prevEl.append(text.slice(pos));
  }
  return mapped;
}

/**
 * line.text 본문 위에 word 토큰을 위치 매핑해 카라오케 span을 구성한다.
 *
 * 서버(CTC) 싱크의 words는 글자 단위 토큰이라, 토큰을 공백으로 이어 붙이면
 * "N e v e r"처럼 깨진다. 대신 본문에서 각 토큰의 위치를 찾아 span으로 감싼다.
 * (LRCLIB 단어 타이밍처럼 진짜 단어 토큰에도 동일하게 동작)
 */
export function appendKaraokeSpans(
  el: HTMLElement,
  line: LyricLine,
  makeWordEl: (word: WordSegment) => HTMLElement,
): void {
  const mapped = appendTimedSpans(el, line.text, line.words ?? [], w => w.word, makeWordEl);
  if (mapped === 0) {
    // 아무 토큰도 매핑 못 함 — 본문만 표시 (카라오케 필 없이)
    el.replaceChildren(line.text);
  }
}
