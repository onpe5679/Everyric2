// 보카로 가사 위키 (vocaro.wikidot.com) 클라이언트.
// - 사이트가 HTTPS를 지원하지 않고 https → http로 301 리다이렉트하므로 평문 http로 접근한다
//   (manifest host_permissions에 http://vocaro.wikidot.com/* 필요).
// - service worker에는 DOMParser가 없어 파싱은 정규식으로 처리한다. 위키 전체가
//   고정 템플릿(가사 표 = 원문/발음/번역 3행 1세트)이라 실용적으로 안전하다.
// - 라이선스: 위키 편집 콘텐츠는 CC BY 4.0(출처 표기 필요), 인용된 원문 가사의
//   저작권은 원저작자에게 있음 — UI에서 출처 페이지 링크를 항상 노출한다.

const BASE = 'http://vocaro.wikidot.com';
const FETCH_TIMEOUT_MS = 4000;
const INDEX_TTL_MS = 24 * 60 * 60 * 1000;

export interface VocaroLine {
  text: string;
  pronunciation?: string;
  translation?: string;
}

export interface VocaroResult {
  pageUrl: string;
  pageTitle: string;
  /** 위키 페이지 슬러그 — videoId별로 저장해두면 재방문 시 발음/번역을 다시 입힐 수 있다 */
  slug: string;
  lines: VocaroLine[];
}

interface IndexEntry {
  title: string;
  slug: string;
}

/** 제목으로 곡 페이지를 찾아 가사(원문+발음+번역)를 반환. 못 찾으면 null */
export async function vocaroLookup(title: string): Promise<VocaroResult | null> {
  const trimmed = title.trim();
  if (!trimmed) return null;

  // 1) ASCII 위주 제목이면 슬러그를 직접 추측 — 요청 1회로 끝나는 경우가 많다
  const guessed = guessSlug(trimmed);
  if (guessed) {
    const page = await fetchSongPage(guessed);
    if (page) return page;
  }

  // 2) 제목 첫 글자에 해당하는 '수록곡 일람' 인덱스에서 제목 매칭
  //    (곡 슬러그는 번역자가 수동으로 지어 규칙이 없으므로 인덱스가 유일한 안정 경로)
  const entries = await getIndexEntries(indexPageFor(trimmed));
  const match = entries ? findMatch(entries, trimmed) : null;
  if (match && match.slug !== guessed) {
    return fetchSongPage(match.slug);
  }
  return null;
}

// ── 슬러그/인덱스 결정 ─────────────────────────────────────────

function guessSlug(title: string): string | null {
  const compact = title.replace(/\s+/g, '');
  if (!compact) return null;
  // 비 ASCII(일본어 등) 제목은 위키 슬러그를 유추할 수 없다 — 일부만 남으면 오탐이므로 포기
  const ascii = compact.replace(/[^\x21-\x7e]/g, '');
  if (ascii.length / compact.length < 0.7) return null;
  const slug = title
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
  return slug.length >= 2 ? slug : null;
}

// 한글 초성 ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ → allsongs-h1~h14 (쌍자음은 기본 자음에 합침)
const CHOSEONG_TO_INDEX = [1, 1, 2, 3, 3, 4, 5, 6, 6, 7, 7, 8, 9, 9, 10, 11, 12, 13, 14];

function indexPageFor(title: string): string {
  const ch = title.trim().charAt(0);
  const code = ch.charCodeAt(0);
  if (code >= 0xac00 && code <= 0xd7a3) {
    return `allsongs-h${CHOSEONG_TO_INDEX[Math.floor((code - 0xac00) / 588)]}`;
  }
  const lower = ch.toLowerCase();
  if (lower >= 'a' && lower <= 'z') return `allsongs-${lower}`;
  if (ch >= '0' && ch <= '9') return 'allsongs-num';
  return 'allsongs-symbols';
}

// ── 인덱스 조회 (24시간 캐시) ──────────────────────────────────

async function getIndexEntries(page: string): Promise<IndexEntry[] | null> {
  const key = `vocaroIdx:${page}`;
  let cached: { at: number; entries: IndexEntry[] } | undefined;
  try {
    const stored = await chrome.storage.local.get(key);
    cached = stored[key] as typeof cached;
    if (cached && Date.now() - cached.at < INDEX_TTL_MS) return cached.entries;
  } catch {
    /* storage 실패는 무시하고 네트워크로 */
  }

  const html = await fetchText(`/${page}`);
  if (!html) return cached?.entries ?? null; // 네트워크 실패 시 만료된 캐시라도 사용

  const entries = parseIndexEntries(html);
  if (entries.length > 0) {
    try {
      await chrome.storage.local.set({ [key]: { at: Date.now(), entries } });
    } catch {
      /* 캐시 저장 실패는 무시 */
    }
  }
  return entries;
}

/** 수록곡 일람 페이지에서 <li><a href="/slug">제목</a></li> 쌍을 추출 */
export function parseIndexEntries(html: string): IndexEntry[] {
  const entries: IndexEntry[] = [];
  const re = /<li>\s*<a\s+href="\/([^"#:]+)"[^>]*>([^<]+)<\/a>\s*<\/li>/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(html)) !== null) {
    const slug = m[1];
    if (slug.startsWith('allsongs') || slug.startsWith('system') || slug.startsWith('guide')) continue;
    const title = decodeEntities(m[2]).trim();
    if (title) entries.push({ title, slug });
  }
  return entries;
}

function normalizeTitle(t: string): string {
  return t.toLowerCase().normalize('NFKC').replace(/[^\p{L}\p{N}]+/gu, '');
}

function findMatch(entries: IndexEntry[], title: string): IndexEntry | null {
  const q = normalizeTitle(title);
  if (q.length < 2) return null;
  const exact = entries.find(e => normalizeTitle(e.title) === q);
  if (exact) return exact;
  // 포함 매칭 — 길이 비율 조건으로 짧은 제목의 우연한 포함을 걸러낸다
  const partial = entries
    .map(e => ({ e, n: normalizeTitle(e.title) }))
    .filter(({ n }) =>
      n.length >= 2
      && (q.includes(n) || n.includes(q))
      && Math.min(q.length, n.length) / Math.max(q.length, n.length) >= 0.5)
    .sort((a, b) => b.n.length - a.n.length)[0];
  return partial?.e ?? null;
}

// ── 곡 페이지 파싱 ─────────────────────────────────────────────

export async function fetchSongPage(slug: string): Promise<VocaroResult | null> {
  const html = await fetchText(`/${slug}`);
  if (!html) return null;
  const parsed = parseSongPage(html);
  if (!parsed || parsed.lines.length === 0) return null;
  return { pageUrl: `${BASE}/${slug}`, pageTitle: parsed.title || slug, slug, lines: parsed.lines };
}

/**
 * 곡 페이지 HTML에서 제목(info-table)과 가사(wiki-content-table)를 파싱.
 * 가사 표는 공식 가이드상 원문/발음/번역 3행 1세트 — 어긋난 표는 방어적으로 처리한다.
 */
export function parseSongPage(html: string): { title: string; lines: VocaroLine[] } | null {
  const table = /<table class="wiki-content-table">([\s\S]*?)<\/table>/.exec(html);
  if (!table) return null;

  const rows: string[] = [];
  const rowRe = /<tr[^>]*>([\s\S]*?)<\/tr>/g;
  let m: RegExpExecArray | null;
  while ((m = rowRe.exec(table[1])) !== null) rows.push(cellText(m[1]));

  let lines: VocaroLine[];
  if (rows.length % 3 === 0) {
    lines = [];
    for (let i = 0; i < rows.length; i += 3) {
      lines.push({ text: rows[i], pronunciation: rows[i + 1] || undefined, translation: rows[i + 2] || undefined });
    }
  } else if (rows.length % 2 === 0) {
    lines = [];
    for (let i = 0; i < rows.length; i += 2) {
      lines.push({ text: rows[i], translation: rows[i + 1] || undefined });
    }
  } else {
    lines = rows.map(text => ({ text }));
  }
  lines = lines.filter(l => l.text.length > 0);

  const titleMatch = /<th[^>]*class="[^"]*title-cell[^"]*"[^>]*>([\s\S]*?)<\/th>/.exec(html);
  const title = titleMatch ? cellText(titleMatch[1]) : '';
  return { title, lines };
}

function cellText(cellHtml: string): string {
  return decodeEntities(
    cellHtml
      .replace(/<span class="rt">[\s\S]*?<\/span>/g, '') // 후리가나 읽기는 원문에서 제외
      .replace(/<br\s*\/?>/g, ' ')
      .replace(/<[^>]+>/g, ''),
  ).replace(/\s+/g, ' ').trim();
}

function decodeEntities(s: string): string {
  return s
    .replace(/&#(\d+);/g, (_, code: string) => String.fromCodePoint(Number(code)))
    .replace(/&#x([0-9a-fA-F]+);/g, (_, code: string) => String.fromCodePoint(parseInt(code, 16)))
    .replace(/&nbsp;/g, ' ')
    .replace(/&quot;/g, '"')
    .replace(/&#039;/g, "'")
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&amp;/g, '&');
}

// ── fetch 유틸 ────────────────────────────────────────────────

async function fetchText(path: string): Promise<string | null> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);
  try {
    const res = await fetch(`${BASE}${path}`, { signal: controller.signal });
    if (!res.ok) return null;
    return await res.text();
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}
