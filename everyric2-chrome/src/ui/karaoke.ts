import type { LyricLine, WordSegment } from '../types';

/**
 * line.text 본문 위에 word 토큰을 위치 매핑해 카라오케 span을 구성한다.
 *
 * 서버(CTC) 싱크의 words는 글자 단위 토큰이라, 토큰을 공백으로 이어 붙이면
 * "N e v e r"처럼 깨진다. 대신 본문에서 각 토큰의 위치를 찾아 span으로 감싸고
 * 사이의 공백/문장부호는 본문 그대로 텍스트 노드로 흘려보낸다.
 * (LRCLIB 단어 타이밍처럼 진짜 단어 토큰에도 동일하게 동작)
 */
export function appendKaraokeSpans(
  el: HTMLElement,
  line: LyricLine,
  makeWordEl: (word: WordSegment) => HTMLElement,
): void {
  const words = line.words ?? [];
  const text = line.text;
  let pos = 0;
  let mapped = 0;
  for (const word of words) {
    const token = word.word;
    if (!token) continue;
    const idx = text.indexOf(token, pos);
    if (idx === -1) continue; // 표기 차이로 본문에서 못 찾는 토큰은 건너뜀
    if (idx > pos) el.append(text.slice(pos, idx));
    el.append(makeWordEl(word));
    pos = idx + token.length;
    mapped++;
  }
  if (mapped === 0) {
    // 아무 토큰도 매핑 못 함 — 본문만 표시 (카라오케 필 없이)
    el.replaceChildren(text);
    return;
  }
  if (pos < text.length) el.append(text.slice(pos));
}
