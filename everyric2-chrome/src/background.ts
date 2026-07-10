import { fetchFromLrclib } from './lib/lrclib';
import { checkHealth, generateSync, getJobStatus, lookupSync, translateLyrics, type ServerConfig } from './lib/everyric-api';
import { parseLRC, parsePlainLyrics, segmentsToLines } from './lib/lyrics-parser';
import { fetchSongPage, vocaroLookup } from './lib/vocaro';
import { getSettings } from './lib/settings';
import type { BgRequest, LyricsData, MessageResponse, SongInfo } from './types';

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
      return { data: await fetchLyricsChain(message.payload) };

    case 'GENERATE_SYNC': {
      const res = await generateSync(await getServerConfig(), {
        video_id: message.payload.videoId,
        lyrics: message.payload.lyrics,
        language: message.payload.language,
        line_meta: message.payload.lineMeta,
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

    case 'VOCARO_LOOKUP':
      return { data: await vocaroLookup(message.payload.title) };

    case 'VOCARO_PAGE':
      return { data: await fetchSongPage(message.payload.slug) };

    default:
      return { error: 'unknown_message_type' };
  }
}

// E2E 스모크 테스트가 SW 컨텍스트에서 직접 호출하기 위한 노출 — 프로덕션 동작에는 영향 없음
(globalThis as { __vocaroLookup?: typeof vocaroLookup }).__vocaroLookup = vocaroLookup;

/** 우선순위: Everyric 서버(단어 타이밍 보존) → LRCLIB 싱크 → LRCLIB 일반 */
async function fetchLyricsChain(song: SongInfo): Promise<LyricsData | null> {
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
      };
    }
  }

  const track = await fetchFromLrclib(song);
  if (track?.syncedLyrics) {
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
  if (track?.plainLyrics) {
    const lines = parsePlainLyrics(track.plainLyrics);
    if (lines.length > 0) {
      return { source: 'lrclib', synced: false, lines, plainText: track.plainLyrics };
    }
  }
  return null;
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
