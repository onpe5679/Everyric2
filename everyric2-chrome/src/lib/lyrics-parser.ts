import type { EveryricSegment, LyricLine, WordSegment } from '../types';

const LINE_TIME_RE = /\[(\d{1,3}):(\d{1,2})(?:[.:](\d{1,3}))?\]/g;
const WORD_TOKEN_RE = /<(\d{1,3}):(\d{1,2})(?:[.:](\d{1,3}))?>([^<]*)/g;

function toSeconds(min: string, sec: string, frac?: string): number {
  const fracSec = frac ? Number(`0.${frac}`) : 0;
  return Number(min) * 60 + Number(sec) + fracSec;
}

export function parseLRC(lrc: string): LyricLine[] {
  const lines: LyricLine[] = [];

  for (const raw of lrc.split('\n')) {
    LINE_TIME_RE.lastIndex = 0;
    const times: number[] = [];
    let bodyStart = 0;
    let m: RegExpExecArray | null;
    while ((m = LINE_TIME_RE.exec(raw)) !== null) {
      if (m.index !== bodyStart) break;
      times.push(toSeconds(m[1], m[2], m[3]));
      bodyStart = LINE_TIME_RE.lastIndex;
    }
    if (times.length === 0) continue;

    const { text, words } = parseWordTimings(raw.slice(bodyStart).trim());
    if (!text) continue;
    for (const time of times) {
      // 반복 타임스탬프 라인끼리 words 배열을 공유하지 않도록 복제
      lines.push({ time, endTime: null, text, words: words?.map(w => ({ ...w })) });
    }
  }

  lines.sort((a, b) => (a.time ?? 0) - (b.time ?? 0));
  for (let i = 0; i < lines.length; i++) {
    lines[i].endTime = i + 1 < lines.length ? lines[i + 1].time : null;
  }
  return lines;
}

function parseWordTimings(body: string): { text: string; words?: WordSegment[] } {
  WORD_TOKEN_RE.lastIndex = 0;
  const words: WordSegment[] = [];
  let m: RegExpExecArray | null;
  while ((m = WORD_TOKEN_RE.exec(body)) !== null) {
    const word = m[4].trim();
    if (!word) continue;
    words.push({ word, start: toSeconds(m[1], m[2], m[3]), end: 0 });
  }
  if (words.length === 0) {
    return { text: body.replace(/<[^>]*>/g, ' ').replace(/\s{2,}/g, ' ').trim() };
  }
  for (let i = 0; i < words.length; i++) {
    words[i].end = i + 1 < words.length ? words[i + 1].start : words[i].start + 1;
  }
  return { text: words.map(w => w.word).join(' '), words };
}

export function parsePlainLyrics(text: string): LyricLine[] {
  return text
    .split('\n')
    .map(line => line.trim())
    .filter(line => line.length > 0)
    .map(line => ({ time: null, endTime: null, text: line }));
}

export function segmentsToLines(segments: EveryricSegment[]): LyricLine[] {
  // 이진 탐색(SyncEngine)이 시간 오름차순을 전제하므로 서버 순서를 믿지 않고 정렬
  const valid = segments
    .filter(s => typeof s.start === 'number' && (s.text ?? '').trim().length > 0)
    .sort((a, b) => a.start - b.start);
  return valid.map((s, i) => ({
    time: s.start,
    endTime: s.end ?? valid[i + 1]?.start ?? null,
    text: s.text.trim(),
    words: s.words && s.words.length > 0 ? s.words : undefined,
    notes: s.notes && s.notes.length > 0 ? s.notes : undefined,
    pronunciation: s.pronunciation || undefined,
    pronSegments: s.pron_segments && s.pron_segments.length > 0 ? s.pron_segments : undefined,
    translation: s.translation || undefined,
    confidence: s.confidence,
    debug: s.debug
      ? {
          activeRatio: s.debug.active_ratio,
          clamped: s.debug.clamped,
          orig: s.debug.orig,
          fixes: s.debug.fixes,
        }
      : undefined,
  }));
}
