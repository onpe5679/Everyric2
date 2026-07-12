import { fetchFromLrclib, getLrclibById, searchTracksLrclib } from './lib/lrclib';
import { checkHealth, generateSync, getJobStatus, linkSync, listSyncs, lookupSync, regenerateSync, resetSync, translateLyrics, unlinkSync, vocaroMatch, type ServerConfig } from './lib/everyric-api';
import { parseLRC, parsePlainLyrics, segmentsToLines } from './lib/lyrics-parser';
import { fetchSongPage, vocaroLookup } from './lib/vocaro';
import { getSettings } from './lib/settings';
import type { BgRequest, CaptionTrack, LRCLibTrack, LyricsData, MessageResponse, SearchCandidate, SongInfo } from './types';

async function getServerConfig(): Promise<ServerConfig> {
  const { serverUrl, apiKey } = await getSettings();
  return { serverUrl, apiKey };
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  handleMessage(message as BgRequest)
    .then(sendResponse)
    .catch((error: unknown) => {
      sendResponse({ error: error instanceof Error ? error.message : String(error) });
    });
  return true;
});

// everyric.com 웹사이트에서 싱크 생성 완료 알림 (기존 플로우 유지)
chrome.runtime.onMessageExternal.addListener((message, sender, sendResponse) => {
  const origin = sender.origin ?? '';
  const isEveryric = origin === 'https://everyric.com' || origin.endsWith('.everyric.com');
  const videoId = (message?.payload as { videoId?: unknown } | undefined)?.videoId;
  if (isEveryric && message?.type === 'SYNC_COMPLETE' && typeof videoId === 'string') {
    void broadcastToYouTubeTabs({ videoId });
    sendResponse({ success: true });
  }
  return true;
});

// 툴바 아이콘 클릭 → 해당 탭의 오버레이 토글
chrome.action.onClicked.addListener(tab => {
  if (tab.id !== undefined) {
    chrome.tabs.sendMessage(tab.id, { type: 'TOGGLE_OVERLAY' }).catch(() => {
      /* content script가 없는 탭(비 YouTube)은 무시 */
    });
  }
});

async function handleMessage(message: BgRequest): Promise<MessageResponse> {
  switch (message.type) {
    case 'FETCH_LYRICS':
      return { data: await fetchLyricsChain(message.payload, message.payload.skipLrclib === true) };

    case 'FETCH_LRCLIB': {
      const track = await fetchFromLrclib(message.payload);
      return { data: track ? lrclibToLyricsData(track) : null };
    }

    case 'SEARCH_CANDIDATES':
      return { data: await searchCandidates(message.payload) };

    case 'PICK_LRCLIB': {
      const track = await getLrclibById(message.payload.id);
      return { data: track ? lrclibToLyricsData(track) : null };
    }

    case 'GENERATE_SYNC': {
      const res = await generateSync(await getServerConfig(), {
        video_id: message.payload.videoId,
        lyrics: message.payload.lyrics,
        language: message.payload.language,
        line_meta: message.payload.lineMeta,
        attribution: message.payload.attribution,
      });
      return res ? { data: res } : { error: 'generate_request_failed' };
    }

    case 'REGENERATE_SYNC': {
      const res = await regenerateSync(await getServerConfig(), {
        video_id: message.payload.videoId,
        lyrics: message.payload.lyrics,
        line_meta: message.payload.lineMeta,
        attribution: message.payload.attribution,
      });
      return res ? { data: res } : { error: 'regenerate_request_failed' };
    }

    case 'JOB_STATUS': {
      const res = await getJobStatus(await getServerConfig(), message.payload.jobId);
      return res ? { data: res } : { error: 'job_status_failed' };
    }

    case 'TRANSLATE': {
      const res = await translateLyrics(
        await getServerConfig(), message.payload.text, message.payload.targetLang,
        { title: message.payload.title, artist: message.payload.artist },
      );
      return res ? { data: res } : { error: 'translate_failed' };
    }

    case 'SERVER_HEALTH': {
      return { data: { ok: await checkHealth(await getServerConfig()) } };
    }

    case 'VOCARO_LOOKUP': {
      const direct = await vocaroLookup(message.payload.title);
      if (direct) return { data: direct };
      // 일본어 원제는 클라이언트의 한국어 독음 인덱스로 못 찾는다 — 서버 원제 인덱스 폴백
      const matched = await vocaroMatch(await getServerConfig(), message.payload.title);
      return { data: matched?.found && matched.slug ? await fetchSongPage(matched.slug) : null };
    }

    case 'SYNC_LINK': {
      const res = await linkSync(await getServerConfig(), {
        video_id: message.payload.videoId,
        source_video_id: message.payload.sourceVideoId,
        offset_sec: message.payload.offsetSec,
      });
      return res ? { data: res } : { error: 'link_failed' };
    }

    case 'SYNC_UNLINK': {
      const res = await unlinkSync(await getServerConfig(), message.payload.videoId);
      return res ? { data: res } : { error: 'unlink_failed' };
    }

    case 'SYNC_RESET': {
      const res = await resetSync(await getServerConfig(), message.payload.videoId);
      return res ? { data: res } : { error: 'sync_reset_failed' };
    }

    case 'SYNC_LIST': {
      const res = await listSyncs(await getServerConfig());
      return { data: res ?? [] };
    }

    case 'VOCARO_PAGE':
      return { data: await fetchSongPage(message.payload.slug) };

    case 'YT_CAPTION_TRACKS':
      return { data: await fetchCaptionTracks(message.payload.videoId) };

    case 'YT_CAPTION_TEXT':
      return { data: await fetchCaptionText(message.payload.baseUrl) };

    default:
      return { error: 'unknown_message_type' };
  }
}

// ── 유튜브 자막 (가사 소스) ─────────────────────────────────────
// 영상에 올라간 자막(예: 일본어 가사 자막)을 가사 붙여넣기 칸으로 가져온다.
// 자막이 가사가 아닌 영상(해설 등)도 있으므로 자동 적용하지 않고, 항상 사용자가
// 붙여넣기 칸에서 내용을 확인·수정한 뒤 직접 생성 버튼을 누르는 흐름을 유지한다.

/** 워치 페이지 HTML에서 captionTracks JSON 배열을 추출 */
async function fetchCaptionTracks(videoId: string): Promise<CaptionTrack[]> {
  const res = await fetch(`https://www.youtube.com/watch?v=${encodeURIComponent(videoId)}`, {
    credentials: 'omit',
  });
  if (!res.ok) return [];
  const html = await res.text();
  const anchor = html.indexOf('"captionTracks":');
  if (anchor === -1) return [];
  const start = html.indexOf('[', anchor);
  if (start === -1) return [];
  // 대괄호 짝을 맞춰 배열 리터럴 끝을 찾는다 (문자열 내부의 괄호는 이스케이프 처리 감안)
  let depth = 0;
  let end = -1;
  let inStr = false;
  for (let i = start; i < html.length; i++) {
    const ch = html[i];
    if (inStr) {
      if (ch === '\\') i++;
      else if (ch === '"') inStr = false;
      continue;
    }
    if (ch === '"') inStr = true;
    else if (ch === '[') depth++;
    else if (ch === ']' && --depth === 0) {
      end = i + 1;
      break;
    }
  }
  if (end === -1) return [];
  try {
    const raw = JSON.parse(html.slice(start, end)) as {
      baseUrl?: string;
      name?: { simpleText?: string; runs?: { text?: string }[] };
      languageCode?: string;
      kind?: string;
    }[];
    return raw
      .filter(t => t.baseUrl)
      .map(t => ({
        baseUrl: t.baseUrl as string,
        label: t.name?.simpleText
          ?? t.name?.runs?.map(r => r.text ?? '').join('')
          ?? t.languageCode ?? '?',
        languageCode: t.languageCode ?? '',
        auto: t.kind === 'asr',
      }));
  } catch {
    return [];
  }
}

/** timedtext(json3)에서 자막 텍스트 라인을 뽑는다 — 연속 중복 제거, 음표/빈 줄 필터 */
async function fetchCaptionText(baseUrl: string): Promise<string | null> {
  const url = baseUrl.includes('fmt=') ? baseUrl : `${baseUrl}&fmt=json3`;
  const res = await fetch(url, { credentials: 'omit' });
  if (!res.ok) return null;
  try {
    const data = await res.json() as { events?: { segs?: { utf8?: string }[] }[] };
    const lines: string[] = [];
    for (const ev of data.events ?? []) {
      const text = (ev.segs ?? []).map(s => s.utf8 ?? '').join('')
        .replace(/\s+/g, ' ').trim();
      // 음표 기호만 있는 줄·빈 줄은 가사가 아니다
      if (!text || /^[♪♫♬\s]+$/.test(text)) continue;
      if (lines[lines.length - 1] !== text) lines.push(text);
    }
    return lines.length > 0 ? lines.join('\n') : null;
  } catch {
    return null;
  }
}

// E2E 스모크 테스트가 SW 컨텍스트에서 직접 호출하기 위한 노출 — 프로덕션 동작에는 영향 없음
(globalThis as { __vocaroLookup?: typeof vocaroLookup }).__vocaroLookup = vocaroLookup;

/** 우선순위: Everyric 서버(단어 타이밍 보존) → (skipLrclib가 아니면) LRCLIB 싱크 → LRCLIB 일반 */
async function fetchLyricsChain(song: SongInfo, skipLrclib = false): Promise<LyricsData | null> {
  const sync = await lookupSync(await getServerConfig(), song.videoId);
  if (sync?.found && sync.timestamps && sync.timestamps.length > 0) {
    const lines = segmentsToLines(sync.timestamps);
    if (lines.length > 0) {
      return {
        source: 'everyric',
        synced: true,
        lines,
        plainText: lines.map(l => l.text).join('\n'),
        // 서버가 사람 번역(위키 병합분)을 내려줬으면 기계번역으로 덮어쓰지 않는다
        humanTranslated: lines.some(l => l.translation),
        debugMeta: sync.debug ?? undefined,
        attribution: sync.attribution ?? undefined,
        tempo: sync.tempo ?? undefined,
        key: sync.key ?? undefined,
        qualityScore: sync.quality_score ?? undefined,
        linked: sync.linked
          ? { sourceVideoId: sync.linked.source_video_id, offsetSec: sync.linked.offset_sec }
          : undefined,
      };
    }
  }

  // 보카로 위키 우선 설정이면 LRCLIB은 content 쪽에서 위키 미스 이후에 별도로 시도한다
  if (skipLrclib) return null;
  const track = await fetchFromLrclib(song);
  return track ? lrclibToLyricsData(track) : null;
}

function lrclibToLyricsData(track: LRCLibTrack): LyricsData | null {
  if (track.syncedLyrics) {
    const lines = parseLRC(track.syncedLyrics);
    if (lines.length > 0) {
      return {
        source: 'lrclib',
        synced: true,
        lines,
        plainText: track.plainLyrics ?? lines.map(l => l.text).join('\n'),
      };
    }
  }
  if (track.plainLyrics) {
    const lines = parsePlainLyrics(track.plainLyrics);
    if (lines.length > 0) {
      return { source: 'lrclib', synced: false, lines, plainText: track.plainLyrics };
    }
  }
  return null;
}

/** 수동 검색 후보: LRCLIB 트랙들 + 보카로 위키 매칭(서버 원제 인덱스 → 클라 독음 인덱스) */
async function searchCandidates(query: { title: string; artist: string; duration: number }): Promise<SearchCandidate[]> {
  const [tracks, wikiMatch] = await Promise.all([
    searchTracksLrclib(query),
    vocaroMatch(await getServerConfig(), query.title),
  ]);

  const candidates: SearchCandidate[] = [];
  if (wikiMatch?.found && wikiMatch.slug) {
    candidates.push({
      source: 'vocaro',
      slug: wikiMatch.slug,
      title: wikiMatch.ja ?? wikiMatch.ko ?? query.title,
      url: wikiMatch.page_url ?? `http://vocaro.wikidot.com/${wikiMatch.slug}`,
    });
  } else {
    // 서버가 없거나 미스 — 클라이언트 독음 인덱스로 한 번 더 (페이지까지 확보되면 그 제목 사용)
    const direct = await vocaroLookup(query.title);
    if (direct) {
      candidates.push({ source: 'vocaro', slug: direct.slug, title: direct.pageTitle, url: direct.pageUrl });
    }
  }
  for (const t of tracks) {
    candidates.push({
      source: 'lrclib',
      id: t.id,
      title: t.trackName ?? '(제목 없음)',
      artist: t.artistName ?? '',
      duration: t.duration ?? 0,
      synced: Boolean(t.syncedLyrics),
    });
  }
  return candidates;
}

async function broadcastToYouTubeTabs(payload: { videoId: string }): Promise<void> {
  const tabs = await chrome.tabs.query({
    url: ['*://www.youtube.com/*', '*://music.youtube.com/*'],
  });
  for (const tab of tabs) {
    if (tab.id !== undefined) {
      chrome.tabs.sendMessage(tab.id, { type: 'SYNC_GENERATED', payload }).catch(() => {
        /* content script 미주입 탭은 무시 */
      });
    }
  }
}
