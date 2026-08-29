// vocaloidlyrics.miraheze.org (VocaloidLyrics Wiki) 클라이언트.
// - MediaWiki API로 검색 → 제목 검증을 통과한 후보 문서만 파싱한다. 발음은 로마자,
//   번역은 영어다
//   (vocaro.wikidot.com의 한글 발음/한국어 번역과 대칭축 — SourceResult.pronLang/translationLang
//   로 구분해 소비처가 script를 고른다).
// - service worker에는 DOMParser가 없어 vocaro.ts와 동일하게 정규식으로 파싱한다.
// - 라이선스: 위키 편집 콘텐츠는 CC BY-SA 4.0(출처 표기 필요) — UI에서 출처 페이지 링크를
//   항상 노출한다.
//
// 실곡 조사(2026-07, 「ロキ (Roki)」「ヴァンパイア (Vampire)/DECO*27」「!mperfection」
// 「フラジール (Fragile)/nulut」 4곡 실제 HTML 대조):
// - 가사 표는 `<table class="lyrics-table" id="lyrics-N">`. **한 페이지에 언어가 다른
//   표가 여러 개 있을 수 있다** — Vampire 페이지의 lyrics-2는 만다린 커버(중국어/병음/영어)
//   였다. 그래서 헤더에 `<th class="lyrics-jp">`가 있는 표만 고른다.
// - 헤더 행 `<tr class="lyrics-table-header">`의 `<th>` 개수(2 또는 3)가 실제 열 수다.
//   Japanese+Romaji만 있으면(2열) 번역이 없다(하꼬곡·romaji-only, 실측: "!mperfection").
//   Japanese+Romaji+English(3열)가 일반적이다.
// - 데이터 행은 `<tr>`(가수별 `style="color:..."`는 무시) 안 `<td>` N개가 열 순서대로
//   원문/로마자/[번역]이다.
// - 연 사이 빈 줄은 `<tr><td><br /></td></tr>` — **칸 1개, 내용 없음** 단독행으로 나타나며
//   건너뛴다.
// - 영어만 있는 삽입구(간투사 등)는 `<td colspan="N" class="merged ...">text</td>` **칸
//   1개, 내용 있음** 단독행 — 원문/로마자 구분이 없으므로 text 하나로만 싣는다(발음·번역
//   없음). 실측(4곡)으로는 한 칸 안에서 `<br>`가 텍스트 두 줄을 가르는 경우(칸 하나에
//   여러 물리 줄)는 없었다 — `cellText`가 `<br>`를 공백으로 접어 한 줄로 합치므로, 그런
//   칸을 만나도 깨지지 않고 한 줄로 근사한다(조용한 실패보다 안전한 폴백).
// - `action=parse&page=<제목>`은 제목에 공백+괄호가 섞이면 "missingtitle"로 실패하는
//   경우가 실측됐다(원인 미상 — 인코딩 정규화 이슈로 추정). search가 이미 pageid를 주므로
//   `action=parse&pageid=<id>`로 우회한다(제목 인코딩 문제 자체가 사라진다).
//
// 실사용 사고(2026-07, H7PR6K7xff0): 유튜브 원제 그대로("シアンブルー / ポリスピカデリー
// feat. 初音ミク")를 검색어로 넣으면 진짜 곡 페이지 「シアンブルー (Cyan Blue)」가 상위
// 10위 안에 아예 안 잡히고(전문검색이 전체 토큰으로 좁혀버린다) 프로듀서 페이지 「Police
// Piccadilly」 1건만 나온다 — startsWith 접두 일치도 실패해 그 프로듀서 페이지를 그대로
// 채택했다. 유튜브 보카로 제목 관례상 곡명은 구분자(/·feat. 등) **앞**에 온다 — 그 조각만
// 검색하면(실측: "シアンブルー") 진짜 곡 페이지가 접두 일치로 잡힌다. `titleCandidates`가
// 이 조각·장식 제거판·원문을 순서대로 시도하게 한다(아래 mirahezeLookup).

import type { SourceLine, SourceResult } from './sources';

const BASE = 'https://vocaloidlyrics.miraheze.org';
const API = `${BASE}/w/api.php`;
const FETCH_TIMEOUT_MS = 4000;
const LICENSE = 'CC BY-SA 4.0';

interface SearchHit {
  pageid: number;
  title: string;
}

interface MirahezeEvidence {
  artist?: string;
  channel?: string;
  rawTitle?: string;
}

/**
 * 제목으로 곡 페이지를 찾아 가사(원문+로마자+[영어 번역])를 반환. 못 찾으면 null.
 *
 * `titleCandidates`가 낸 후보를 순서대로 시도한다. 검색 결과와 실제 parse한 정규
 * 페이지 제목이 모두 후보와 정규화 접두 관계이고, 그 페이지에 일본어 가사 표가
 * 있을 때만 채택한다. 하나라도 어긋나면 다음 후보로 넘어가며, 검증되지 않은 검색
 * 1위로는 절대 물러나지 않는다.
 */
export async function mirahezeLookup(
  title: string,
  preferredTitles?: string[],
  evidence?: MirahezeEvidence,
): Promise<SourceResult | null> {
  const trimmed = title.trim();
  if (!trimmed) return null;

  const roots = [...(preferredTitles ?? []), trimmed]
    .map(value => value.trim())
    .filter((value, index, all) => value && all.indexOf(value) === index)
    .slice(0, 4);
  const candidates = roots
    .flatMap(root => titleCandidates(root, evidence))
    .filter((value, index, all) => all.indexOf(value) === index)
    .slice(0, 6);
  for (const candidate of candidates) {
    const hits = await searchTitleHits(candidate);
    const valid: SourceResult[] = [];
    for (const hit of hits) {
      if (!producerMatchesEvidence(hit.title, evidence, candidate)) continue;
      const page = await fetchParsedPage(hit.pageid);
      if (!page || !titleMatchesCandidate(candidate, page.title)) continue;
      if (!producerMatchesEvidence(page.title, evidence, candidate)) continue;
      const parsed = parseLyricsTable(page.html);
      if (!parsed || parsed.lines.length === 0) continue;
      valid.push({
        sourceId: 'miraheze',
        // MediaWiki 문서 URL은 공백→'_'만 치환하고 나머지는 그대로 남긴다.
        pageUrl: `${BASE}/wiki/${encodeURI(page.title.replace(/ /g, '_'))}`,
        pageTitle: page.title,
        lines: parsed.lines,
        pronLang: 'romaji',
        translationLang: parsed.hasTranslation ? 'en' : undefined,
        license: LICENSE,
      });
      if (valid.length > 1) break; // 실제 가사 페이지가 둘이면 제목만으로는 못 가른다
    }
    if (valid.length === 1) return valid[0];
  }
  return null;
}

function producerMatchesEvidence(
  pageTitle: string,
  evidence?: MirahezeEvidence,
  candidate?: string,
): boolean {
  if (!pageTitle.includes('/')) return true;
  const suffix = pageTitle.slice(pageTitle.lastIndexOf('/') + 1).trim();
  const producerKey = normalizeIdentityToken(suffix);
  const rawParts = (evidence?.rawTitle ?? '')
    .split(/\s*[/／|｜ㅣ]\s*|\s[-–—]\s/)
    .flatMap(part => [part, part.replace(SAFE_FEAT_SUFFIX, '').trim()]);
  const candidateKey = normalizeIdentityToken(candidate ?? '');
  const hints = [evidence?.artist, evidence?.channel, ...rawParts]
    .map(value => normalizeIdentityToken(value ?? ''))
    .filter(value => (
      value.length >= 3
      && value !== candidateKey
      && !_KNOWN_VOCALS.has(value)
    ));
  if (!producerKey || hints.length === 0) return false;
  return hints.includes(producerKey);
}

function normalizeIdentityToken(value: string): string {
  return value.normalize('NFKC').toLowerCase().replace(/[^\p{L}\p{N}]+/gu, '');
}

// ── 검색어 후보 생성 ─────────────────────────────────────────

// 대표 구분자 — 유튜브 보카로 관례상 곡명 다음에 아티스트·가수 표기가 붙는 자리.
// feat./ft.는 대소문자·마침표 유무를 가리지 않는다(feat, Feat., FT 등). \b로 낱말
// 경계를 요구해 "soft" 같은 낱말 속 "ft"를 오매칭하지 않는다.
const _TITLE_SEPARATOR_RE = /\/|｜|\||\s-\s|〜|\bfeat\.?|\bft\.?/i;
const SAFE_FEAT_SUFFIX = /(?:^|\s)(?:feat|ft)\.?\s*\S.*$/i;

const _BRACKET_RE = /【([^】]*)】|\[([^\]]*)\]|（([^）]*)）|\(([^)]*)\)/g;
const _SAFE_NOISE_RE = /^(?:official(?:\s+(?:music|lyric)\s*video|\s+audio|\s+mv)?|music\s*video|lyrics?|audio|mv|pv|hd|hq|4k|1080p|720p|cover|カバー|커버)$/i;
const _KNOWN_VOCALS = new Set([
  '初音ミク', 'hatsunemiku', '鏡音リン', 'kagaminerin', '鏡音レン', 'kagaminelen',
  '巡音ルカ', 'megurineluka', 'gumi', 'ia', 'kaito', 'meiko', '重音テト', 'kasaneteto',
  '可不', 'kafu', 'flower', 'vflower', '歌愛ユキ', 'kaaiyuki',
].map(normalizeIdentityToken));

/** 구분자 **앞** 조각(=곡명) — 구분자가 없거나 맨 앞에 있으면(짐작할 곡명이 없으면) null. */
function _stripBeforeSeparator(
  raw: string,
  evidence?: MirahezeEvidence,
): string | null {
  const m = _TITLE_SEPARATOR_RE.exec(raw);
  if (!m || m.index === 0) return null;
  const head = raw.slice(0, m.index).trim();
  const tail = raw.slice(m.index + m[0].length).trim();
  if (!head || !tail) return null;
  const tailKey = normalizeIdentityToken(tail);
  const tailKeys = new Set([
    tailKey,
    normalizeIdentityToken(tail.replace(SAFE_FEAT_SUFFIX, '').trim()),
  ]);
  const hints = [evidence?.artist, evidence?.channel]
    .map(value => normalizeIdentityToken(value ?? ''))
    .filter(value => value.length >= 3);
  const featVocal = tail.match(/(?:^|\s)(?:feat|ft)\.?\s*(.+)$/i)?.[1];
  const featVocalKey = normalizeIdentityToken(featVocal ?? '');
  const corroborated = hints.some(hint => tailKeys.has(hint))
    || _KNOWN_VOCALS.has(tailKey)
    || _KNOWN_VOCALS.has(featVocalKey);
  return corroborated ? head : null;
}

/** 장식 구간을 제거한 판 — 장식이 없어 원문과 같으면(바뀐 게 없으면) null. */
function _stripDecorations(raw: string): string | null {
  const cleaned = raw.replace(
    _BRACKET_RE,
    (whole, corner?: string, square?: string, wide?: string, round?: string) => {
      const content = [corner, square, wide, round]
        .find(value => value !== undefined)?.trim() ?? '';
      return _SAFE_NOISE_RE.test(content) ? ' ' : whole;
    },
  ).replace(/\s+/g, ' ').trim();
  return cleaned && cleaned !== raw ? cleaned : null;
}

/**
 * 검색 전 제목 후보를 순서대로 낸다: ① 구분자 앞 조각(곡명), ② 장식 제거판, ③ 원문
 * 그대로(최후 폴백). 중복·빈 문자열은 제외하고 최대 3개다.
 *
 * 실사용 사고(위 파일 머리말 참조): 유튜브 원제를 그대로 검색하면 부가 정보(아티스트·
 * feat.·장식)가 전문검색의 관련도를 흐려 진짜 곡 페이지가 상위 10위 밖으로 밀린다.
 */
export function titleCandidates(
  raw: string,
  evidence?: MirahezeEvidence,
): string[] {
  const trimmed = raw.trim();
  const out: string[] = [];
  const add = (v: string | null): void => {
    if (v && !out.includes(v)) out.push(v);
  };
  add(_stripBeforeSeparator(trimmed, evidence));
  add(_stripDecorations(trimmed));
  add(trimmed);
  return out.slice(0, 3);
}

/** MediaWiki 제목 비교용 정규화 — NFKC + 소문자 + 공백 접기. */
function normalizeMatchTitle(value: string): string {
  return value.normalize('NFKC').toLowerCase().replace(/\s+/g, ' ').trim();
}

/**
 * 정확 일치 또는 `ロキ (Roki)`·`フラジール/nulut`처럼 인정된 접미가 붙은 접두만
 * 자동 채택한다. 접두 뒤가 일반 문자인 `Wonderland`는 `Wonder`의 일치가 아니다.
 */
export function titleMatchesCandidate(candidate: string, pageTitle: string): boolean {
  const candidateNorm = normalizeMatchTitle(candidate);
  const pageNorm = normalizeMatchTitle(pageTitle);
  if (!candidateNorm || !pageNorm) return false;
  if (pageNorm === candidateNorm) return true;
  if (!pageNorm.startsWith(candidateNorm)) return false;
  const suffix = pageNorm.slice(candidateNorm.length);
  if (suffix.startsWith('/')) return true; // `/producer`는 별도 exact evidence로 다시 검증한다
  const romanAlias = suffix.match(/^ \(([^()]*)\)(?:\/.+)?$/);
  if (!romanAlias || !/[^\x00-\x7F]/.test(candidateNorm)) return false;
  // 일본어 원제 뒤 로마자/영문 별칭만 허용한다. 버전·리믹스 표기는 다른 곡 정체성이다.
  const alias = normalizeIdentityToken(romanAlias[1]);
  const aliasLabel = romanAlias[1].normalize('NFKC').toLowerCase();
  const versionToken = /(?:^|[\s._-])(?:remix|mix|live|cover|acoustic|version|ver|edit|type|long|short|20\d{2})(?:$|[\s._-])/i;
  return Boolean(
    alias
    && !versionToken.test(aliasLabel),
  );
}

// ── 검색 ──────────────────────────────────────────────────────

// MediaWiki 전문검색(list=search)은 제목이 아니라 본문 관련도로 정렬한다 — 짧고 흔한
// 제목일수록 진짜 곡 페이지가 밀려난다(실측: "ロキ" 검색 1위는 그 곡을 수록한 앨범
// 페이지고 진짜 곡 페이지 「ロキ (Roki)」는 2위, "フラジール"는 프로듀서 본인 페이지
// "Nulut"가 1위이고 곡 페이지 「フラジール (Fragile)/nulut」는 5위였다). 곡 페이지 제목은
// 예외 없이 원어 제목으로 **시작한다**(뒤에 "(로마자)"·"/프로듀서"가 붙는 식) — 그래서
// 상위 10개 중 검색어로 시작하는 첫 제목을 우선한다.
const SEARCH_LIMIT = 10;

/**
 * 정규화 접두 일치 히트가 없으면 null(다음 후보로 넘어가라는 신호). 검색 순위는
 * 같은 곡임을 증명하지 않으므로 마지막 후보에서도 최상위 결과로 폴백하지 않는다.
 */
async function searchTitleHits(title: string): Promise<SearchHit[]> {
  const params = new URLSearchParams({
    action: 'query',
    list: 'search',
    srsearch: title,
    srlimit: String(SEARCH_LIMIT),
    format: 'json',
    origin: '*',
  });
  const data = await getJSON<{ query?: { search?: SearchHit[] } }>(`${API}?${params}`);
  const hits = data?.query?.search ?? [];
  return hits.filter(h => titleMatchesCandidate(title, h.title));
}

async function fetchParsedPage(pageid: number): Promise<{ title: string; html: string } | null> {
  const params = new URLSearchParams({
    action: 'parse',
    pageid: String(pageid),
    prop: 'text',
    format: 'json',
    origin: '*',
  });
  const data = await getJSON<{ parse?: { title?: string; text?: { '*': string } } }>(`${API}?${params}`);
  const title = data?.parse?.title;
  const html = data?.parse?.text?.['*'];
  return title && html ? { title, html } : null;
}

async function getJSON<T>(url: string): Promise<T | null> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);
  try {
    const res = await fetch(url, { signal: controller.signal });
    if (!res.ok) return null;
    return (await res.json()) as T;
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

// ── 가사 표 파싱 ─────────────────────────────────────────────

/** `<table class="lyrics-table" ...>`들 중 헤더에 lyrics-jp가 있는 표(원문이 일본어인 표) */
function findJapaneseLyricsTable(html: string): string | null {
  const re = /<table class="lyrics-table"[^>]*>([\s\S]*?)<\/table>/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(html)) !== null) {
    if (m[1].includes('class="lyrics-jp"')) return m[1];
  }
  return null;
}

function parseLyricsTable(html: string): { lines: SourceLine[]; hasTranslation: boolean } | null {
  const table = findJapaneseLyricsTable(html);
  if (!table) return null;

  const headerMatch = /<tr class="lyrics-table-header">([\s\S]*?)<\/tr>/.exec(table);
  if (!headerMatch) return null;
  const headerCount = (headerMatch[1].match(/<th/g) ?? []).length;
  if (headerCount < 2) return null; // 최소 원문+로마자

  const body = table.slice(headerMatch.index + headerMatch[0].length);
  const lines: SourceLine[] = [];
  const rowRe = /<tr(?:\s[^>]*)?>([\s\S]*?)<\/tr>/g;
  let m: RegExpExecArray | null;
  while ((m = rowRe.exec(body)) !== null) {
    for (const line of rowToLines(extractCells(m[1]))) lines.push(line);
  }
  return { lines, hasTranslation: headerCount >= 3 };
}

/** 행 하나(칸 목록)를 SourceLine 0~n개로. 빈 연 구분·삽입구·정상 행을 여기서 가른다 */
function rowToLines(cells: string[]): SourceLine[] {
  if (cells.length === 0) return [];
  if (cells.length === 1) {
    // <td><br /></td> 단독(내용 없음) — 연 사이 공백 줄, 가사가 아니다
    if (!cells[0]) return [];
    // <td colspan="N" class="merged ...">text</td> — 원문/로마자 구분 없는 삽입구
    return [{ text: cells[0] }];
  }
  const [text, pronunciation, translation] = cells;
  if (!text) return [];
  return [{ text, pronunciation: pronunciation || undefined, translation: translation || undefined }];
}

function extractCells(rowHtml: string): string[] {
  const cells: string[] = [];
  const re = /<td[^>]*>([\s\S]*?)<\/td>/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(rowHtml)) !== null) cells.push(cellText(m[1]));
  return cells;
}

function cellText(cellHtml: string): string {
  return decodeEntities(
    cellHtml
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
