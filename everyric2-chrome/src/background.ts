import { fetchLyrics } from './lib/lrclib';
import { getSettings, saveSettings } from './lib/settings';
import type { SongInfo, Settings, EveryricSyncResponse, LyricsResult, MessageResponse } from './types';

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  handleMessage(message)
    .then(sendResponse)
    .catch(error => sendResponse({ error: error.message }));
  return true;
});

chrome.runtime.onMessageExternal.addListener((message, sender, sendResponse) => {
  const origin = sender.origin || '';
  const isEveryric = origin.includes('everyric.com');
  
  if (isEveryric && message.type === 'SYNC_COMPLETE') {
    broadcastToYouTubeTabs(message.payload);
    sendResponse({ success: true });
  }
  
  return true;
});

async function handleMessage(message: { type: string; payload?: unknown }): Promise<MessageResponse> {
  switch (message.type) {
    case 'FETCH_LYRICS':
      return handleFetchLyrics(message.payload as SongInfo);
    
    case 'FETCH_EVERYRIC_SYNC':
      return handleFetchEveryricSync(message.payload as { videoId: string; lyricsHash?: string });
    
    case 'GET_SETTINGS':
      return { data: await getSettings() };
    
    case 'SAVE_SETTINGS':
      await saveSettings(message.payload as Settings);
      return { success: true };
    
    default:
      return { error: 'Unknown message type' };
  }
}

async function handleFetchLyrics(songInfo: SongInfo): Promise<MessageResponse<LyricsResult[]>> {
  const results: LyricsResult[] = [];
  
  try {
    const everyricResult = await fetchFromEveryricServer(songInfo.videoId);
    if (everyricResult) {
      results.push(everyricResult);
      return { data: results };
    }
  } catch (error) {
    console.log('[Everyric] Local server not available, falling back to LRCLIB');
  }
  
  try {
    const lrclibResult = await fetchLyrics(songInfo);
    if (lrclibResult) {
      results.push(lrclibResult);
    }
  } catch (error) {
    console.error('[Everyric] Error fetching lyrics:', error);
  }
  
  return { data: results };
}

async function fetchFromEveryricServer(videoId: string): Promise<LyricsResult | null> {
  const response = await fetch(`http://localhost:8000/api/sync/${videoId}`, {
    signal: AbortSignal.timeout(2000)
  });
  
  if (!response.ok) return null;
  
  const data = await response.json();
  if (!data.found || !data.timestamps) return null;
  
  const syncedLyrics = data.timestamps
    .map((t: { text: string; start: number }) => `[${formatTime(t.start)}]${t.text}`)
    .join('\n');
  
  return {
    type: 'synced',
    source: 'everyric',
    syncedLyrics,
    plainLyrics: data.timestamps.map((t: { text: string }) => t.text).join('\n')
  };
}

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  const ms = Math.floor((seconds % 1) * 100);
  return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}.${ms.toString().padStart(2, '0')}`;
}

async function handleFetchEveryricSync(params: { videoId: string; lyricsHash?: string }): Promise<MessageResponse<EveryricSyncResponse>> {
  try {
    const url = new URL(`https://api.everyric.com/api/sync/${params.videoId}`);
    if (params.lyricsHash) {
      url.searchParams.set('lyrics_hash', params.lyricsHash);
    }
    
    const response = await fetch(url.toString());
    
    if (response.ok) {
      return { data: await response.json() };
    }
    
    return { data: { found: false } };
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    return { data: { found: false, error: message } };
  }
}

async function broadcastToYouTubeTabs(payload: { videoId: string }): Promise<void> {
  const tabs = await chrome.tabs.query({
    url: ['*://www.youtube.com/*', '*://music.youtube.com/*']
  });
  
  for (const tab of tabs) {
    if (tab.id) {
      chrome.tabs.sendMessage(tab.id, {
        type: 'SYNC_GENERATED',
        payload
      }).catch(() => {});
    }
  }
}

console.log('[Everyric] Background service worker initialized');
