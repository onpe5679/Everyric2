import type { ApiFailure, EveryricSyncResponse, GenerateResponse, JobStatusResponse, LineMeta, LinkCandidatesResponse, LinkJobStatusResponse, SaveTranslationLayerResponse, ServerLogEntry, ServerStatus, SourceAttribution, SyncListItem, SyncPreviousVersion, TranslateResult } from '../types';
import { affectsServerStatus, failureKindFromStatus, failureToStatus, maskPath, maskSecret, okStatus } from './server-status';
import { localPermissionBlock, normalizeLoopbackUrl } from './host-permissions';

export interface ServerConfig {
  serverUrl: string;
  /** 빈 문자열이면 인증 헤더를 보내지 않는다 */
  apiKey?: string;
}

/**
 * 실패 사유를 회수하기 위해 **호출부가 직접 만들어 넘기는 그릇**.
 *
 * 왜 이 모양인가: 예전 `request()`는 모든 실패를 `null` 하나로 뭉갰다. 사유를 살리려면
 * 반환 타입을 결과 유니온으로 바꾸는 게 정석이지만, 그러면 이 모듈의 모든 export와
 * background의 모든 핸들러를 동시에 고쳐야 한다. 대신 **마지막 인자에 선택적 싱크**를
 * 두었다 — 기존 시그니처(`T | null`)는 그대로라 안 넘기는 호출부는 아무 변화가 없고,
 * 사유가 필요한 호출부만 `const sink: FailureSink = {}`를 만들어 넘긴 뒤 읽는다.
 *
 * 모듈 전역에 "마지막 실패"를 담아 두는 방식은 쓰지 않았다 — 서비스워커에서 여러 탭의
 * 요청이 겹치면 남의 실패를 주워 오기 때문이다. 싱크는 호출 한 건이 소유하므로 안전하다.
 */
export interface FailureSink {
  failure?: ApiFailure;
}

// ── 최근 요청 로그 ───────────────────────────────────────────────
// 화면의 접이식 로그에 그대로 나간다. 서비스워커가 잠들면 사라지는 휘발성 버퍼로 충분하다
// (지금 뭐가 잘못됐는지 보려는 용도이지, 감사 로그가 아니다).

const LOG_MAX = 20;
const serverLog: ServerLogEntry[] = [];

function pushLog(entry: ServerLogEntry): void {
  serverLog.push(entry);
  if (serverLog.length > LOG_MAX) serverLog.splice(0, serverLog.length - LOG_MAX);
}

/** 최근 요청부터 (최신순) */
export function getServerLog(): ServerLogEntry[] {
  return serverLog.slice().reverse();
}

function baseUrl(server: ServerConfig): string {
  // Windows에서 localhost 해석이 ::1(IPv6)을 먼저 시도해 요청마다 ~2s를 태울 수 있다
  // (서버는 IPv4 0.0.0.0 바인딩) — 루프백은 127.0.0.1로 정규화해 스톨을 없앤다.
  // 정규화 규칙 자체는 host-permissions에 있다: 권한 확인이 **실제로 요청이 나가는**
  // origin을 봐야 하므로, 두 곳이 각자 정규화하면 반드시 어긋난다.
  return normalizeLoopbackUrl(server.serverUrl).replace(/\/+$/, '');
}

/**
 * 서버 스키마 상한(title 256·artist 128자)에 맞춰 자른다.
 *
 * 넘기면 FastAPI가 422를 낸다 — 그게 하필 **싱크 조회**에서 터지면 제목을 실어 보낸 대가로
 * 가사가 통째로 사라진다(백필은 부수 효과일 뿐인데 본 기능을 깨뜨린다). 서버 쪽 백필도
 * 자른 제목으로 충분히 매칭되므로 잘라 보내는 편이 언제나 낫다.
 */
const TITLE_MAX = 256;
const ARTIST_MAX = 128;

function clip(value: string | null | undefined, max: number): string | undefined {
  const text = value?.trim();
  return text ? text.slice(0, max) : undefined;
}

function buildHeaders(server: ServerConfig, extra?: Record<string, string>): Record<string, string> {
  const headers: Record<string, string> = { ...extra };
  if (server.apiKey) headers['X-API-Key'] = server.apiKey;
  return headers;
}

/** 오류 본문에서 사람이 읽을 문구를 뽑는다 — 서버는 error·hint·detail·message를 쓴다 */
const DETAIL_MAX = 240;

async function readErrorDetail(res: Response, secret?: string): Promise<string | undefined> {
  let body: string;
  try {
    body = (await res.text()).slice(0, 2000);
  } catch {
    return undefined;
  }
  if (!body.trim()) return undefined;

  let text = body;
  try {
    const parsed = JSON.parse(body) as unknown;
    if (parsed && typeof parsed === 'object') {
      const obj = parsed as Record<string, unknown>;
      // FastAPI는 detail, 우리 핸들러는 error+hint를 쓴다 — 있는 것을 순서대로 잇는다
      const parts = ['error', 'detail', 'message', 'hint']
        .map(k => obj[k])
        .filter(v => v !== undefined && v !== null)
        .map(v => (typeof v === 'string' ? v : JSON.stringify(v)));
      if (parts.length > 0) text = parts.join(' — ');
    }
  } catch {
    /* JSON이 아니면 본문 앞부분을 그대로 (HTML 오류 페이지 등) */
  }
  const cleaned = maskSecret(text.replace(/\s+/g, ' ').trim(), secret);
  return cleaned ? cleaned.slice(0, DETAIL_MAX) : undefined;
}

/**
 * 서버 요청 한 건.
 *
 * 반환은 예전과 같은 `T | null`이지만, 실패한 이유는 더 이상 버리지 않는다 —
 * sink를 넘기면 거기에 담기고, 넘기지 않아도 최근 요청 로그에는 항상 남는다.
 */
async function request<T>(
  server: ServerConfig, path: string, init?: RequestInit, timeoutMs = 4000, sink?: FailureSink,
): Promise<T | null> {
  const started = Date.now();
  const method = init?.method ?? 'GET';
  const safePath = maskPath(path, server.apiKey);

  const fail = (kind: ApiFailure['kind'], status?: number, detail?: string): null => {
    const failure: ApiFailure = {
      kind,
      status,
      detail: detail ? maskSecret(detail, server.apiKey).slice(0, DETAIL_MAX) : undefined,
      path: safePath,
      elapsedMs: Date.now() - started,
    };
    if (sink) sink.failure = failure;
    pushLog({
      at: started, method, path: safePath, ok: false,
      status, kind, detail: failure.detail, elapsedMs: failure.elapsedMs,
    });
    return null;
  };

  // 자체 호스팅 서버는 optional 권한이다 — 없이 부르면 fetch가 그냥 실패해 'offline'으로
  // 분류되고, 화면은 "서버가 꺼져 있거나 주소가 잘못됐어요"라고 말한다. 서버는 멀쩡히 돌고
  // 있는데 사용자는 서버를 의심하며 로그를 뒤진다. 그래서 **부르기 전에** 막고 권한 문제로
  // 정확히 분류한다. 원격 주소는 문자열 판정 한 번으로 빠져나가 크롬 API를 부르지 않는다.
  const url = baseUrl(server);
  const blocked = await localPermissionBlock(url);
  if (blocked !== null) return fail('permission', undefined, blocked);

  let res: Response;
  try {
    res = await fetch(`${url}${path}`, {
      ...init,
      headers: buildHeaders(server, init?.headers as Record<string, string> | undefined),
      signal: AbortSignal.timeout(timeoutMs),
    });
  } catch (error) {
    // AbortSignal.timeout은 TimeoutError를 던진다 — 그 밖은 서버에 닿지 못한 것.
    //
    // **Chrome 103~123은 TimeoutError 대신 AbortError를 던진다**(124에서 고쳐졌다).
    // 우리 하한이 103이라 그 구간 사용자는 "2500ms 초과"가 아니라 "오프라인"으로 오분류된
    // 문구를 봤다 — 실패한다는 결과는 같지만 원인이 틀리면 서버를 의심할 자리에서 네트워크를
    // 의심하게 된다.
    //
    // AbortError도 타임아웃으로 단정해도 되는 이유: **이 fetch에 붙는 signal은 timeout
    // 하나뿐이고 수동 취소 경로가 없다.** 나중에 사용자 취소를 붙이면 그때는 둘을 갈라야
    // 한다(그 경우 signal.reason으로 판정해야 한다 — name만으로는 구별되지 않는다).
    const timedOut = error instanceof DOMException
      && (error.name === 'TimeoutError' || error.name === 'AbortError');
    const message = error instanceof Error ? error.message : String(error);
    return fail(timedOut ? 'timeout' : 'offline', undefined, timedOut ? `${timeoutMs}ms 초과` : message);
  }

  if (!res.ok) {
    return fail(failureKindFromStatus(res.status), res.status, await readErrorDetail(res, server.apiKey));
  }
  try {
    const value = await res.json() as T;
    pushLog({
      at: started, method, path: safePath, ok: true,
      status: res.status, elapsedMs: Date.now() - started,
    });
    return value;
  } catch {
    return fail('malformed', res.status, '응답 본문이 JSON이 아니에요');
  }
}

/**
 * 이 영상의 싱크 조회.
 *
 * song(제목·아티스트)을 함께 보내면 서버가 **비어 있을 때만** 기존 싱크의 제목을 채운다
 * (기존 값은 덮어쓰지 않는다). 재생성 없이 코퍼스에 제목이 쌓이게 하는 유일한 경로라서
 * 조회할 때마다 싣는다 — 제목이 없으면 커버 링크 후보 탐색이 영원히 빈손이 된다.
 * 구버전 서버는 모르는 쿼리 파라미터를 그냥 무시하므로 붙여도 안전하다.
 */
export function lookupSync(
  server: ServerConfig,
  videoId: string,
  song?: { title?: string | null; artist?: string | null; lang?: string },
  sink?: FailureSink,
): Promise<EveryricSyncResponse | null> {
  const params = new URLSearchParams();
  const title = clip(song?.title, TITLE_MAX);
  const artist = clip(song?.artist, ARTIST_MAX);
  if (title) params.set('title', title);
  if (artist) params.set('artist', artist);
  // lang 없이 조회하면 서버 응답은 기존과 필드 단위로 동일해야 한다 — 값이 없을 때만 생략
  if (song?.lang) params.set('lang', song.lang);
  const query = params.size > 0 ? `?${params.toString()}` : '';
  // 8000ms — 2500ms는 응답이 수십 KB이던 시절의 보정치다. 다국어화로 조회 응답이
  // 210~240KB(무압축 실측)가 되면서 게이트웨이·일반 회선에서 2.5초를 넘나들어
  // «됐다가 안 됐다가» 타임아웃이 났다(사용자 실보고). 서버 gzip과 함께 양쪽에서 잡는다.
  return request<EveryricSyncResponse>(
    server, `/api/sync/${encodeURIComponent(videoId)}${query}`, undefined, 8000, sink,
  );
}

/** 정렬 품질 별점(1~5) + 선택 오류 제보 — 수집 전용, 응답은 {ok: true} */
export function submitFeedback(
  server: ServerConfig,
  payload: { video_id: string; rating: number; category?: string; comment?: string },
  sink?: FailureSink,
): Promise<{ ok: boolean } | null> {
  return request<{ ok: boolean }>(server, '/api/sync/feedback', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  }, 8000, sink);
}

/**
 * 이 영상 자기 싱크의 직전(재처리로 덮어써지기 전) 세대 — 디버그 패널의 A/B 고스트
 * 비교용. 이력이 없으면 서버가 found=false를 준다(404가 아니다 — 조회 관례 동일).
 */
export async function fetchPreviousSync(
  server: ServerConfig, videoId: string, sink?: FailureSink,
): Promise<SyncPreviousVersion | null> {
  return request<SyncPreviousVersion>(
    server, `/api/sync/${encodeURIComponent(videoId)}/previous`, undefined, 8000, sink,
  );
}

/**
 * 같은 곡의 다른 영상(원곡·다른 업로드) 후보 탐색.
 *
 * 후보가 있으면 **서버가** 반주 상관 검증 잡을 자동 제출하고 그 id를 돌려준다. 쿨다운·
 * 중복 제출 억제도 서버가 한다 — 클라이언트는 잡 id만 받아 폴링하면 된다.
 * 이 엔드포인트가 없는 구버전 서버에서는 404 → null이므로 호출부는 조용히 포기한다.
 */
export function findLinkCandidates(
  server: ServerConfig,
  videoId: string,
  song: { title: string; artist?: string | null },
  sink?: FailureSink,
): Promise<LinkCandidatesResponse | null> {
  const title = clip(song.title, TITLE_MAX);
  if (!title) return Promise.resolve(null); // 서버가 title을 필수로 요구한다 (min_length=1)
  const params = new URLSearchParams({ title });
  const artist = clip(song.artist, ARTIST_MAX);
  if (artist) params.set('artist', artist);
  return request<LinkCandidatesResponse>(
    server,
    `/api/sync/${encodeURIComponent(videoId)}/link-candidates?${params.toString()}`,
    undefined,
    6000,
    sink,
  );
}

/** 반주 상관 검증 잡의 상태. 404(기록 소실)는 gone 마커로 돌려 폴링을 마감시킨다 —
 *  null로 뭉개면 사라진 잡을 영원히 폴링한다 (getJobStatus와 같은 규약). */
export async function getLinkJobStatus(
  server: ServerConfig, linkJobId: string, sink?: FailureSink,
): Promise<LinkJobStatusResponse | null> {
  const probe: FailureSink = {};
  const res = await request<LinkJobStatusResponse>(
    server, `/api/link-jobs/${encodeURIComponent(linkJobId)}`, undefined, 4000, probe,
  );
  if (res !== null) return res;
  if (probe.failure?.status === 404) {
    return { status: 'failed', gone: true };
  }
  if (sink) sink.failure = probe.failure;
  return null;
}

/** 서버는 여기서 큐 등록만 하고 즉시 job_id를 반환해야 한다 (처리 대기와 무관).
 *  title·artist는 완성된 싱크에 함께 저장돼 나중에 커버 링크 후보 탐색의 단서가 된다. */
export function generateSync(
  server: ServerConfig,
  payload: {
    video_id: string; lyrics: string; language?: string; line_meta?: LineMeta[];
    line_meta_pending?: boolean;
    attribution?: SourceAttribution; title?: string; artist?: string;
    /** 생성 요청자의 번역 언어(기본 서버 "ko") — 없으면 서버가 기존과 동일하게 동작 */
    target_lang?: string;
    /** line_meta로 실은 번역의 언어(기본 서버 "ko") */
    line_meta_lang?: string;
  },
  sink?: FailureSink,
): Promise<GenerateResponse | null> {
  return request<GenerateResponse>(server, '/api/sync/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    // 제목이 서버 상한을 넘으면 422로 생성 자체가 실패한다 — 자르고 보낸다
    body: JSON.stringify({
      ...payload,
      title: clip(payload.title, TITLE_MAX),
      artist: clip(payload.artist, ARTIST_MAX),
    }),
  }, 15000, sink);
}

/** 캐시를 무시하고 싱크를 강제 재생성한다 — 큐 등록 후 즉시 job_id 반환 */
export function regenerateSync(
  server: ServerConfig,
  payload: {
    video_id: string; lyrics: string; line_meta?: LineMeta[];
    attribution?: SourceAttribution; title?: string; artist?: string;
    target_lang?: string; line_meta_lang?: string;
    /** 분석 깊이 하한 — 서버가 라우팅을 건너뛰고 이 깊이에서 시작한다 (깊이 버튼) */
    min_depth?: 'medium' | 'heavy';
  },
  sink?: FailureSink,
): Promise<GenerateResponse | null> {
  return request<GenerateResponse>(server, '/api/sync/regenerate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      ...payload,
      force: true,
      title: clip(payload.title, TITLE_MAX),
      artist: clip(payload.artist, ARTIST_MAX),
    }),
  }, 15000, sink);
}

/** 404(잡 기록 소실 — 서버 재시작 등)는 null(일시 실패)과 구분해 gone 마커로 돌려준다.
 *  null이면 폴링이 무한 재시도하므로 사라진 잡은 명시적으로 마감시켜야 한다. */
export async function getJobStatus(
  server: ServerConfig, jobId: string, sink?: FailureSink,
): Promise<JobStatusResponse | null> {
  const probe: FailureSink = {};
  const res = await request<JobStatusResponse>(
    server, `/api/job/${encodeURIComponent(jobId)}`, undefined, 4000, probe,
  );
  if (res !== null) return res;
  // 404 = 잡 기록 소실. 폴링을 명시적으로 마감시키되, 서버가 죽은 것은 아니므로
  // 실패 사유는 호출부에 넘기지 않는다 (전역 서버 상태를 내리면 안 된다)
  if (probe.failure?.status === 404) {
    return { job_id: jobId, status: 'failed', progress: 0, gone: true };
  }
  if (sink) sink.failure = probe.failure;
  return null;
}

/** 진행 중인 전사 잡 취소 요청 — 이미 끝난 잡이면 cancelled=false와 현재 상태가 온다 */
export function cancelJob(
  server: ServerConfig, jobId: string, sink?: FailureSink,
): Promise<{ job_id: string; cancelled: boolean; status?: string } | null> {
  return request<{ job_id: string; cancelled: boolean; status?: string }>(
    server, `/api/job/${encodeURIComponent(jobId)}/cancel`, { method: 'POST' }, 4000, sink,
  );
}

/** LLM 번역은 곡 전체 기준 수십 초가 걸릴 수 있어 타임아웃을 길게 잡는다.
 *  발음표기(target=ko면 한글 독음)도 항상 함께 요청한다 — 원문이 en/ko면 서버 게이트가
 *  발음만 생략하고 번역은 정상 반환. 곡 제목/아티스트는 LLM 번역 맥락으로 쓰인다. */
export function translateLyrics(
  server: ServerConfig,
  text: string,
  targetLang: string,
  song?: { title?: string | null; artist?: string | null; persist?: boolean; videoId?: string },
  sink?: FailureSink,
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
      // persist+video_id를 함께 주면 서버가 이 번역을 언어별 레이어로 저장한다(둘 다 optional
      // — 안 주면 오늘과 동일하게 저장 없이 결과만 반환)
      persist: song?.persist || undefined,
      video_id: song?.videoId || undefined,
    }),
  }, 120000, sink);
}

/** inst·커버 영상을 다른 영상의 싱크에 오프셋·배속과 함께 연결 (재등록은 upsert).
 *  rate는 원곡 대비 재생 배속(nightcore≈1.25) — 서버가 t/rate+offset으로 시간축을 사상한다. */
export function linkSync(
  server: ServerConfig,
  payload: { video_id: string; source_video_id: string; offset_sec: number; rate: number },
  sink?: FailureSink,
): Promise<Record<string, unknown> | null> {
  return request<Record<string, unknown>>(server, '/api/sync/link', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  }, 4000, sink);
}

/** 이 영상의 서버 싱크 전부 삭제(초기화) — 잘못 붙여넣은 가사로 만든 싱크를 걷어내고 새로 시작 */
export function resetSync(
  server: ServerConfig, videoId: string, sink?: FailureSink,
): Promise<{ removed_syncs: number; removed_links: number } | null> {
  return request<{ removed_syncs: number; removed_links: number }>(
    server, `/api/sync/${encodeURIComponent(videoId)}`, { method: 'DELETE' }, 4000, sink,
  );
}

/**
 * 자막·위키 채택 번역을 서버 레이어로 저장 — POST /api/sync/{video_id}/translations.
 * 거절도 200으로 온다(saved=false, request()가 그대로 통과시킨다) — 404(싱크 없음)·
 * 422(매칭률<50%·origin 위반·상한)만 request()의 !res.ok 경로로 빠져 null이 된다.
 * 호출부(content.ts)는 fire-and-forget으로 쓴다 — 실패해도 표시엔 영향 없다.
 */
export function saveTranslationLayer(
  server: ServerConfig,
  videoId: string,
  payload: {
    target_lang: string;
    lines: { text: string; translation: string }[];
    origin: 'caption' | 'wiki' | 'manual';
    attribution?: SourceAttribution;
  },
  sink?: FailureSink,
): Promise<SaveTranslationLayerResponse | null> {
  return request<SaveTranslationLayerResponse>(
    server, `/api/sync/${encodeURIComponent(videoId)}/translations`,
    { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) },
    8000, sink,
  );
}

/** 영상별 사용자 싱크 오프셋 저장 — 다음 조회부터 user_offset으로 내려온다 */
export function saveUserOffset(
  server: ServerConfig, videoId: string, offsetSec: number, sink?: FailureSink,
): Promise<{ offset_sec: number } | null> {
  return request<{ offset_sec: number }>(server, `/api/sync/offset/${encodeURIComponent(videoId)}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ offset_sec: offsetSec }),
  }, 4000, sink);
}

export function unlinkSync(
  server: ServerConfig, videoId: string, sink?: FailureSink,
): Promise<{ removed: boolean } | null> {
  return request<{ removed: boolean }>(server, `/api/sync/link/${encodeURIComponent(videoId)}`, {
    method: 'DELETE',
  }, 4000, sink);
}

/** 서버에 저장된 싱크 목록 — 링크 후보 선택용 (최신순) */
export function listSyncs(
  server: ServerConfig, limit = 50, sink?: FailureSink,
): Promise<SyncListItem[] | null> {
  return request<SyncListItem[]>(server, `/api/sync/list?limit=${limit}`, undefined, 4000, sink);
}

/**
 * 서버를 쓸 수 있는지 + 못 쓴다면 왜인지.
 *
 * **`/health`만 보면 안 된다.** 서버 미들웨어는 `/api` 전체에만 X-API-Key를 요구하고
 * `/health`는 상태 점검용으로 항상 열어 둔다 — 키가 틀린 배포에서도 health는 200이다.
 * 실제로 이것 때문에 모든 /api 호출이 401인데 화면은 "가사를 찾지 못했어요"를 띄웠다.
 * 그래서 도달성(/health)과 인증(/api)을 **따로** 확인한다.
 */
export async function checkServerStatus(server: ServerConfig): Promise<ServerStatus> {
  const reach: FailureSink = {};
  let health = await request<{ status: string }>(server, '/health', undefined, 1500, reach);
  if (health === null && (reach.failure?.kind === 'offline' || reach.failure?.kind === 'timeout')) {
    // 일시적 지연(콜드 스타트 등)으로 서버를 죽었다고 오판하면 생성 버튼이 잠긴다 — 한 번 더.
    // 401·5xx는 다시 물어도 답이 같으므로 재시도하지 않는다.
    reach.failure = undefined;
    health = await request<{ status: string }>(server, '/health', undefined, 4000, reach);
  }
  if (health === null) return failureToStatus(reach.failure);

  // 서버는 살아 있다 — 이제 이 키로 /api를 쓸 수 있는지 본다 (가장 가벼운 인증 엔드포인트)
  const auth: FailureSink = {};
  const listed = await listSyncs(server, 1, auth);
  if (listed !== null) return okStatus();
  // 인증·도달 문제가 아니면(구버전 서버의 404 등) 서버 자체는 쓸 수 있다고 본다
  if (auth.failure && !affectsServerStatus(auth.failure.kind)) return okStatus();
  return failureToStatus(auth.failure);
}

/** 자막 트랙의 라인(타이밍 포함) — 서버가 yt-dlp json3를 파싱해 준다.
 *  트랙 **목록**은 서버를 부르지 않는다 — 클라이언트가 워치 페이지에서 직접 읽는다
 *  (lib/yt-captions.ts). 본문만 POT 강제로 서버 경유가 유일 경로. */
export function fetchCaptionLines(
  server: ServerConfig, videoId: string, lang: string, auto: boolean, sink?: FailureSink,
): Promise<{ lines: { start: number; end: number; text: string }[] } | null> {
  return request(
    server,
    `/api/captions/${encodeURIComponent(videoId)}/${encodeURIComponent(lang)}?auto=${auto}`,
    undefined,
    35000,
    sink,
  );
}

/**
 * 자막으로 보고 있던 곡의 싱크 생성 — 서버가 자막을 직접 읽어 가사로 쓰는 전용 경로.
 *
 * **이 엔드포인트는 아직 없을 수 있다.** 없으면(404·미배포 서버) request()가 null을
 * 돌려주므로 호출부는 기존 /api/sync/generate 경로로 폴백한다 — 그래서 여기서 404와
 * 네트워크 실패를 구분하지 않는다(둘 다 "이 경로는 못 쓴다"로 같게 취급하면 충분).
 * sink에는 사유가 담기지만, 404는 affectsServerStatus()가 걸러 서버 상태를 내리지 않는다.
 */
/**
 * 진행 중인 생성 잡에 번역·독음을 나중에 붙인다 — 번역을 다운로드·보컬 분리와 겹치는 경로.
 *
 * **번역이 실패했어도 빈 배열로 반드시 한 번 호출해야 한다.** 서버는 정렬 직전에 이 메타를
 * 유한 시간 기다리는데, 아무것도 안 보내면 그 상한까지 잡이 헛되게 서 있는다(빈 배열은
 * "붙일 것이 없음"이 확정된 신호라 서버가 즉시 원문 정렬로 넘어간다).
 *
 * 잡이 이미 끝났어도(캐시 히트 등) 호출은 성공한다 — 서버가 완성된 싱크에 발음·번역만
 * 병합하고 applied로 무엇이 일어났는지 알려준다. 클라이언트가 분기할 필요는 없다.
 */
export function attachLineMeta(
  server: ServerConfig,
  jobId: string,
  payload: {
    line_meta: LineMeta[]; attribution?: SourceAttribution; title?: string; artist?: string;
    line_meta_lang?: string;
  },
  sink?: FailureSink,
): Promise<{ job_id: string; status: string; applied: string; merged_segments?: number } | null> {
  return request(server, `/api/sync/jobs/${encodeURIComponent(jobId)}/line-meta`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      ...payload,
      title: clip(payload.title, TITLE_MAX),
      artist: clip(payload.artist, ARTIST_MAX),
    }),
  }, 15000, sink);
}

export function generateSyncFromCaption(
  server: ServerConfig, videoId: string, sink?: FailureSink,
): Promise<GenerateResponse | null> {
  return request<GenerateResponse>(server, '/api/sync/generate-from-caption', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ video_id: videoId }),
  }, 15000, sink);
}

export interface VocaroMatchResponse {
  found: boolean;
  slug?: string | null;
  page_url?: string | null;
  ko?: string | null;
  ja?: string | null;
}

/** 일본어 원제 등 클라이언트 독음 인덱스로 못 찾는 제목을 서버 원제 인덱스에 묻는다 */
export function vocaroMatch(
  server: ServerConfig, title: string, sink?: FailureSink,
): Promise<VocaroMatchResponse | null> {
  return request<VocaroMatchResponse>(
    server, `/api/vocaro/match?title=${encodeURIComponent(title)}`, undefined, 2500, sink,
  );
}

export interface VocaroPageLineDto {
  text: string;
  pronunciation?: string | null;
  translation?: string | null;
}

export interface VocaroPageResponse {
  found: boolean;
  slug?: string | null;
  page_url?: string | null;
  page_title?: string | null;
  lines?: VocaroPageLineDto[];
}

export interface VocaroIndexEntryDto {
  slug: string;
  title: string;
}

export interface VocaroIndexResponse {
  found: boolean;
  entries?: VocaroIndexEntryDto[];
}

/**
 * 위키 곡 페이지 프록시 — 1.5.5부터 vocaro host 권한 없이 서버가 대신 조회한다
 * (위키는 CORS 헤더가 없어 확장이 직접 읽을 수 없다). 이 엔드포인트가 없는 구버전
 * 자체 호스팅 서버는 404 → null이 되고, vocaro 폴백만 조용히 꺼진다.
 * 타임아웃은 서버의 위키 조회(예의 간격+백오프)까지 감싼 값이다.
 */
export function vocaroPage(
  server: ServerConfig, slug: string, sink?: FailureSink,
): Promise<VocaroPageResponse | null> {
  return request<VocaroPageResponse>(
    server, `/api/vocaro/page?slug=${encodeURIComponent(slug)}`, undefined, 10000, sink,
  );
}

/** 수록곡 일람 프록시 — 한국어 독음 제목 매칭용 (slug, 제목) 쌍. 위 vocaroPage와 같은 사정. */
export function vocaroIndex(
  server: ServerConfig, page: string, sink?: FailureSink,
): Promise<VocaroIndexResponse | null> {
  return request<VocaroIndexResponse>(
    server, `/api/vocaro/index?page=${encodeURIComponent(page)}`, undefined, 10000, sink,
  );
}
