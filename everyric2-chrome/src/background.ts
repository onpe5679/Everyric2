import { fetchFromLrclib, getLrclibById, searchTracksLrclib } from './lib/lrclib';
import { checkHealth, generateSync, getJobStatus, lookupSync, translateLyrics, vocaroMatch, type ServerConfig } from './lib/everyric-api';
import { parseLRC, parsePlainLyrics, segmentsToLines } from './lib/lyrics-parser';
import { fetchSongPage, vocaroLookup } from './lib/vocaro';
import { getSettings } from './lib/settings';
import type { BgRequest, LRCLibTrack, LyricsData, MessageResponse, SearchCandidate, SongInfo } from './types';

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

    case 'JOB_STATUS': {
      const res = await getJobStatus(await getServerConfig(), message.payload.jobId);
      return res ? { data: res } : { error: 'job_status_failed' };
    }

    case 'TRANSLATE': {
      const res = await translateLyrics(await getServerConfig(), message.payload.text, message.payload.targetLang);
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

    case 'VOCARO_PAGE':
      return { data: await fetchSongPage(message.payload.slug) };

    default:
      return { error: 'unknown_message_type' };
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
