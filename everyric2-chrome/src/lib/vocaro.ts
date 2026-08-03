// 보카로 가사 위키 (vocaro.wikidot.com) 클라이언트 — **서버 프록시 경유** (1.5.5+).
//
// 1.5.4까지는 이 파일이 위키를 직접 fetch했고 그래서 manifest에
// http://vocaro.wikidot.com/* host 권한이 필요했다(위키가 CORS 헤더를 안 보낸다 —
// 실측 2026-07-28, access-control-* 전무). 스토어 심사 부담을 줄이려고(사용자 결정)
// 위키 조회를 서버(/api/vocaro/page·/index)로 옮기고 권한을 제거했다. 파싱도 서버가
// 한다(everyric2/sources/vocaro.py — 이 파일의 구 파서를 포팅한 것이라 줄 나눔이 같다).
// 이 엔드포인트가 없는 구버전 자체 호스팅 서버에서는 vocaro 폴백만 조용히 꺼진다.
//
// - 라이선스: 위키 편집 콘텐츠는 CC BY 4.0(출처 표기 필요), 인용된 원문 가사의
//   저작권은 원저작자에게 있음 — UI에서 출처 페이지 링크를 항상 노출한다.

import type { SourceResult } from './sources';
import { vocaroIndex, vocaroPage, type ServerConfig } from './everyric-api';

const INDEX_TTL_MS = 24 * 60 * 60 * 1000;
const LICENSE = 'CC BY 4.0'; // 위키 편집 콘텐츠 라이선스 — 파일 상단 주석 참고

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

/**
 * ``VocaroResult``를 소스 공통 계약(``SourceResult``)으로 어댑트한다.
 *
 * 기존 반환 타입·호출부(``vocaroLookup``/``fetchSongPage``)는 그대로 둔다 — slug 같은
 * vocaro 전용 필드(재방문 시 재조회용)를 잃지 않기 위해서다. 이 함수는 miraheze 등
 * 다른 소스와 같은 모양으로 다뤄야 하는 자리(소스 체인·attribution 생성)에서만 쓰는
 * 점진 전환용 어댑터다.
 */
export function vocaroToSourceResult(r: VocaroResult): SourceResult {
  return {
    sourceId: 'vocaro',
    pageUrl: r.pageUrl,
    pageTitle: r.pageTitle,
    lines: r.lines,
    pronLang: 'hangul',
    translationLang: 'ko',
    license: LICENSE,
  };
}

interface IndexEntry {
  title: string;
  slug: string;
}

/** 제목으로 곡 페이지를 찾아 가사(원문+발음+번역)를 반환. 못 찾으면 null */
export async function vocaroLookup(server: ServerConfig, title: string): Promise<VocaroResult | null> {
  const trimmed = title.trim();
  if (!trimmed) return null;

  // 1) ASCII 위주 제목이면 슬러그를 직접 추측 — 요청 1회로 끝나는 경우가 많다
  const guessed = guessSlug(trimmed);
  if (guessed) {
    const page = await fetchSongPage(server, guessed);
    if (page) return page;
  }

  // 2) 제목 첫 글자에 해당하는 '수록곡 일람' 인덱스에서 제목 매칭
  //    (곡 슬러그는 번역자가 수동으로 지어 규칙이 없으므로 인덱스가 유일한 안정 경로)
  //    인덱스가 지원하지 않는 제목(일본어 원제 등)은 여기서 끝낸다 — indexPageFor 주석 참고.
  const indexPage = indexPageFor(trimmed);
  if (!indexPage) return null;
  const entries = await getIndexEntries(server, indexPage);
  const match = entries ? findMatch(entries, trimmed) : null;
  if (match && match.slug !== guessed) {
    return fetchSongPage(server, match.slug);
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

/**
 * 이 제목을 찾아볼 '수록곡 일람' 인덱스 페이지 — **인덱스가 지원하지 않는 제목이면 null**.
 *
 * 이 인덱스는 위키 등재 제목(한국어 독음 또는 원어 로마자/숫자) 기준으로 나뉜다. 일본어
 * 원제(「我ら！ゴミ分別団」)를 넣으면 어느 초성·알파벳에도 안 걸려 예전에는 잡동사니
 * 페이지(`allsongs-symbols`)로 떨어졌고, 거기서 findMatch의 포함 매칭이 **아무 상관 없는
 * 곡**을 집어 그 가사를 원곡 가사로 띄웠다(실측 오매칭 사고). 후보가 실제로 있을 리 없는
 * 자리에서 포함 매칭을 돌리는 것은 오답을 만드는 일밖에 하지 않는다.
 *
 * 원제 매칭은 서버 인덱스(`/api/vocaro/match`)의 몫이다 — 거기엔 원제·한국어 표기가 함께
 * 있고 다중 후보 가드도 서버에 있다. 이 클라이언트 경로는 **초성 인덱스가 실제로 답할 수
 * 있는 제목**(한글·라틴·숫자 시작)으로만 남기고, 나머지는 못 찾았다고 말한다.
 */
function indexPageFor(title: string): string | null {
  const ch = title.trim().charAt(0);
  const code = ch.charCodeAt(0);
  if (code >= 0xac00 && code <= 0xd7a3) {
    return `allsongs-h${CHOSEONG_TO_INDEX[Math.floor((code - 0xac00) / 588)]}`;
  }
  const lower = ch.toLowerCase();
  if (lower >= 'a' && lower <= 'z') return `allsongs-${lower}`;
  if (ch >= '0' && ch <= '9') return 'allsongs-num';
  return null;
}

// ── 인덱스 조회 (24시간 캐시) ──────────────────────────────────

async function getIndexEntries(server: ServerConfig, page: string): Promise<IndexEntry[] | null> {
  const key = `vocaroIdx:${page}`;
  let cached: { at: number; entries: IndexEntry[] } | undefined;
  try {
    const stored = await chrome.storage.local.get(key);
    cached = stored[key] as typeof cached;
    if (cached && Date.now() - cached.at < INDEX_TTL_MS) return cached.entries;
  } catch {
    /* storage 실패는 무시하고 네트워크로 */
  }

  const res = await vocaroIndex(server, page);
  // found=false(서버가 위키 조회 실패·페이지명 거절)와 null(서버 미도달·구버전 404)을
  // 구분하지 않는다 — 어느 쪽이든 만료된 캐시라도 있으면 그것으로 매칭을 시도한다
  if (!res?.found || !res.entries) return cached?.entries ?? null;

  const entries: IndexEntry[] = res.entries.map(e => ({ title: e.title, slug: e.slug }));
  if (entries.length > 0) {
    try {
      await chrome.storage.local.set({ [key]: { at: Date.now(), entries } });
    } catch {
      /* 캐시 저장 실패는 무시 */
    }
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

// ── 곡 페이지 조회 ─────────────────────────────────────────────

export async function fetchSongPage(server: ServerConfig, slug: string): Promise<VocaroResult | null> {
  const res = await vocaroPage(server, slug);
  if (!res?.found || !res.slug || !res.lines || res.lines.length === 0) return null;
  return {
    pageUrl: res.page_url ?? `http://vocaro.wikidot.com/${res.slug}`,
    pageTitle: res.page_title || res.slug,
    slug: res.slug,
    lines: res.lines.map(l => ({
      text: l.text,
      pronunciation: l.pronunciation ?? undefined,
      translation: l.translation ?? undefined,
    })),
  };
}
