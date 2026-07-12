import type { EveryricSyncResponse, GenerateResponse, JobStatusResponse, LineMeta, SourceAttribution, SyncListItem, TranslateResult } from '../types';

export interface ServerConfig {
  serverUrl: string;
  /** 빈 문자열이면 인증 헤더를 보내지 않는다 */
  apiKey?: string;
}

function baseUrl(server: ServerConfig): string {
  return server.serverUrl.replace(/\/+$/, '');
}

function buildHeaders(server: ServerConfig, extra?: Record<string, string>): Record<string, string> {
  const headers: Record<string, string> = { ...extra };
  if (server.apiKey) headers['X-API-Key'] = server.apiKey;
  return headers;
}

async function request<T>(server: ServerConfig, path: string, init?: RequestInit, timeoutMs = 4000): Promise<T | null> {
  try {
    const res = await fetch(`${baseUrl(server)}${path}`, {
      ...init,
      headers: buildHeaders(server, init?.headers as Record<string, string> | undefined),
      signal: AbortSignal.timeout(timeoutMs),
    });
    if (!res.ok) return null;
    return await res.json() as T;
  } catch {
    return null;
  }
}

export function lookupSync(server: ServerConfig, videoId: string): Promise<EveryricSyncResponse | null> {
  return request<EveryricSyncResponse>(server, `/api/sync/${encodeURIComponent(videoId)}`, undefined, 2500);
}

/** 서버는 여기서 큐 등록만 하고 즉시 job_id를 반환해야 한다 (처리 대기와 무관) */
export function generateSync(
  server: ServerConfig,
  payload: { video_id: string; lyrics: string; language?: string; line_meta?: LineMeta[]; attribution?: SourceAttribution },
): Promise<GenerateResponse | null> {
  return request<GenerateResponse>(server, '/api/sync/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  }, 15000);
}

/** 캐시를 무시하고 싱크를 강제 재생성한다 — 큐 등록 후 즉시 job_id 반환 */
export function regenerateSync(
  server: ServerConfig,
  payload: { video_id: string; lyrics: string; line_meta?: LineMeta[]; attribution?: SourceAttribution },
): Promise<GenerateResponse | null> {
  return request<GenerateResponse>(server, '/api/sync/regenerate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...payload, force: true }),
  }, 15000);
}

export function getJobStatus(server: ServerConfig, jobId: string): Promise<JobStatusResponse | null> {
  return request<JobStatusResponse>(server, `/api/job/${encodeURIComponent(jobId)}`);
}

/** LLM 번역은 곡 전체 기준 수십 초가 걸릴 수 있어 타임아웃을 길게 잡는다.
 *  발음표기(target=ko면 한글 독음)도 항상 함께 요청한다 — 원문이 en/ko면 서버 게이트가
 *  발음만 생략하고 번역은 정상 반환. 곡 제목/아티스트는 LLM 번역 맥락으로 쓰인다. */
export function translateLyrics(
  server: ServerConfig,
  text: string,
  targetLang: string,
  song?: { title?: string | null; artist?: string | null },
): Promise<TranslateResult | null> {
  return request<TranslateResult>(server, '/api/translate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      text,
      source_lang: 'auto',
      target_lang: targetLang,
      include_pronunciation: true,
      title: song?.title || undefined,
      artist: song?.artist || undefined,
    }),
  }, 120000);
}

/** inst·커버 영상을 다른 영상의 싱크에 오프셋과 함께 연결 (재등록은 upsert) */
export function linkSync(
  server: ServerConfig,
  payload: { video_id: string; source_video_id: string; offset_sec: number },
): Promise<Record<string, unknown> | null> {
  return request<Record<string, unknown>>(server, '/api/sync/link', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

export function unlinkSync(server: ServerConfig, videoId: string): Promise<{ removed: boolean } | null> {
  return request<{ removed: boolean }>(server, `/api/sync/link/${encodeURIComponent(videoId)}`, {
    method: 'DELETE',
  });
}

/** 서버에 저장된 싱크 목록 — 링크 후보 선택용 (최신순) */
export function listSyncs(server: ServerConfig, limit = 50): Promise<SyncListItem[] | null> {
  return request<SyncListItem[]>(server, `/api/sync/list?limit=${limit}`);
}

export async function checkHealth(server: ServerConfig): Promise<boolean> {
  const res = await request<{ status: string }>(server, '/health', undefined, 1500);
  return res !== null;
}

export interface VocaroMatchResponse {
  found: boolean;
  slug?: string | null;
  page_url?: string | null;
  ko?: string | null;
  ja?: string | null;
}

/** 일본어 원제 등 클라이언트 독음 인덱스로 못 찾는 제목을 서버 원제 인덱스에 묻는다 */
export function vocaroMatch(server: ServerConfig, title: string): Promise<VocaroMatchResponse | null> {
  return request<VocaroMatchResponse>(server, `/api/vocaro/match?title=${encodeURIComponent(title)}`, undefined, 2500);
}
