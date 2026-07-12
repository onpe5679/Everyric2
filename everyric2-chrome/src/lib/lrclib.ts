import type { LRCLibTrack, SongInfo } from '../types';

const BASE = 'https://lrclib.net/api';

async function getJSON<T>(url: string): Promise<T | null> {
  try {
    const res = await fetch(url, {
      headers: { 'Lrclib-Client': 'everyric2-chrome/1.1.0' },
      // 미스가 흔한 경로라 짧게 — 느린 응답 하나가 "가사를 찾지 못했어요"까지의
      // 체감 시간을 지배한다 (수동 검색 진입이 늦어짐)
      signal: AbortSignal.timeout(3500),
    });
    if (!res.ok) return null;
    return await res.json() as T;
  } catch {
    return null;
  }
}

export async function fetchFromLrclib(song: SongInfo): Promise<LRCLibTrack | null> {
  if (song.artist) {
    const params = new URLSearchParams({
      track_name: song.title,
      artist_name: song.artist,
    });
    if (song.duration > 0) params.set('duration', String(song.duration));
    const exact = await getJSON<LRCLibTrack>(`${BASE}/get?${params}`);
    if (exact && !exact.instrumental) return exact;
  }
  return searchLrclib(song);
}

async function searchLrclib(song: SongInfo): Promise<LRCLibTrack | null> {
  // [attempt, 제목 검증 필요 여부] — q= 자유검색은 LRCLIB이 유사도 무관 fuzzy 매칭을
  // 하므로(전혀 무관한 곡이 duration만 비슷해도 반환됨) 제목 검증을 강제한다.
  const attempts: [URLSearchParams, boolean][] = [];
  if (song.artist) {
    attempts.push([new URLSearchParams({ track_name: song.title, artist_name: song.artist }), false]);
  }
  // duration도 아티스트도 없으면 자유검색을 걸러낼 신호가 제목뿐 — 흔한 제목(“花” 등)은
  // 전혀 다른 곡이 통과하므로 자유검색 자체를 생략한다 (수동 검색으로 유도)
  if (song.artist || song.duration > 0) {
    attempts.push([new URLSearchParams({ q: song.artist ? `${song.artist} ${song.title}` : song.title }), true]);
  }
  if (attempts.length === 0) return null;

  // 두 시도를 병렬 발사하되 우선순위(정확 검색 먼저)대로 채택 — 대기 시간 절반
  const settled = await Promise.all(
    attempts.map(async ([params, requireTitle]) => {
      const results = await getJSON<LRCLibTrack[]>(`${BASE}/search?${params}`);
      return pickBest(results ?? [], song.duration, requireTitle ? song.title : null);
    }),
  );
  return settled.find(t => t !== null) ?? null;
}

/** 수동 검색용 후보 리스트 — 두 검색을 병렬로 모아 dedup 후 점수순 상위만 반환 */
export async function searchTracksLrclib(
  query: { title: string; artist: string; duration: number },
  limit = 8,
): Promise<LRCLibTrack[]> {
  const attempts: URLSearchParams[] = [];
  if (query.artist) {
    attempts.push(new URLSearchParams({ track_name: query.title, artist_name: query.artist }));
  }
  attempts.push(new URLSearchParams({ q: query.artist ? `${query.artist} ${query.title}` : query.title }));

  const settled = await Promise.all(
    attempts.map(params => getJSON<LRCLibTrack[]>(`${BASE}/search?${params}`)),
  );
  const seen = new Set<number>();
  const merged: LRCLibTrack[] = [];
  for (const list of settled) {
    for (const t of list ?? []) {
      if (t.instrumental || (!t.syncedLyrics && !t.plainLyrics) || seen.has(t.id)) continue;
      seen.add(t.id);
      merged.push(t);
    }
  }
  // 수동 선택 리스트는 사용자가 직접 보고 고르므로 제목 검증 없이 duration·싱크 점수로만 정렬
  return merged
    .map(t => {
      const diff = query.duration > 0 && t.duration > 0 ? Math.abs(t.duration - query.duration) : 999;
      return { t, score: Math.min(diff, 999) + (t.syncedLyrics ? 0 : 1000) };
    })
    .sort((a, b) => a.score - b.score)
    .slice(0, limit)
    .map(({ t }) => t);
}

/** 후보 선택 후 해당 트랙만 다시 가져온다 */
export function getLrclibById(id: number): Promise<LRCLibTrack | null> {
  return getJSON<LRCLibTrack>(`${BASE}/get/${id}`);
}

function normalizeTitle(t: string): string {
  return t.toLowerCase().normalize('NFKC').replace(/[^\p{L}\p{N}]+/gu, '');
}

/** 자유검색 결과가 실제로 이 곡인지 — 정규화 제목의 상호 포함으로 판정 */
function titleMatches(trackName: string | undefined, query: string): boolean {
  const a = normalizeTitle(trackName ?? '');
  const b = normalizeTitle(query);
  if (a.length < 2 || b.length < 2) return false;
  return a.includes(b) || b.includes(a);
}

function pickBest(tracks: LRCLibTrack[], duration: number, requireTitle: string | null = null): LRCLibTrack | null {
  const scored = tracks
    .filter(t => !t.instrumental && (t.syncedLyrics || t.plainLyrics))
    .filter(t => requireTitle === null || titleMatches(t.trackName, requireTitle))
    .map(t => {
      const diff = duration > 0 && t.duration > 0 ? Math.abs(t.duration - duration) : 999;
      let score = Math.min(diff, 999);
      if (!t.syncedLyrics) score += 1000;
      // duration이 크게 어긋난 싱크 가사는 하이라이트가 어긋나므로 뒤로 민다
      else if (diff > 20) score += 500;
      return { t, score };
    })
    .sort((a, b) => a.score - b.score);
  return scored[0]?.t ?? null;
}
