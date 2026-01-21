import { detectSong, getCurrentVideoId, getVideoElement } from './lib/song-detector';
import { parseLRC, parsePlainLyrics } from './lib/lyrics-parser';
import { syncRenderer } from './lib/sync-renderer';
import { LyricsOverlay } from './lib/overlay';
import type { Settings, LyricsResult, LyricLine } from './types';

let currentVideoId: string | null = null;
let lyrics: LyricLine[] = [];
let overlay: LyricsOverlay | null = null;
let settings: Settings | null = null;
let hasSyncedLyrics = false;

async function init(): Promise<void> {
  settings = await fetchSettings();
  
  if (!settings.autoSearch) return;
  
  observeNavigation();
  checkCurrentPage();
}

function observeNavigation(): void {
  let lastUrl = location.href;
  
  const observer = new MutationObserver(() => {
    if (location.href !== lastUrl) {
      lastUrl = location.href;
      setTimeout(checkCurrentPage, 1000);
    }
  });
  
  observer.observe(document.body, { childList: true, subtree: true });
  
  document.addEventListener('yt-navigate-finish', () => {
    setTimeout(checkCurrentPage, 500);
  });
}

function checkCurrentPage(): void {
  const videoId = getCurrentVideoId();
  
  if (!videoId) {
    cleanup();
    return;
  }
  
  if (videoId !== currentVideoId) {
    currentVideoId = videoId;
    lyrics = [];
    hasSyncedLyrics = false;
    searchLyrics();
  }
}

function cleanup(): void {
  syncRenderer.stop();
  overlay?.hide();
  overlay = null;
  currentVideoId = null;
  lyrics = [];
  hasSyncedLyrics = false;
}

async function waitForSongInfo(maxRetries = 5, delay = 1000): Promise<ReturnType<typeof detectSong>> {
  for (let i = 0; i < maxRetries; i++) {
    const info = detectSong();
    if (info?.title && info.title !== 'YouTube' && info.title !== '') {
      return info;
    }
    await new Promise(r => setTimeout(r, delay));
  }
  return detectSong();
}

async function searchLyrics(): Promise<void> {
  if (!settings) return;
  
  overlay?.hide();
  
  overlay = new LyricsOverlay(settings);
  overlay.setOnSyncGenerated(() => {
    if (!hasSyncedLyrics) {
      setTimeout(() => searchLyrics(), 1000);
    }
  });
  overlay.showLoading();
  
  const songInfo = await waitForSongInfo();
  if (!songInfo?.title || songInfo.title === 'YouTube') {
    overlay.showNoLyrics('Unable to detect song');
    return;
  }
  
  overlay.updateSongInfo(songInfo);
  
  try {
    const response = await chrome.runtime.sendMessage({
      type: 'FETCH_LYRICS',
      payload: songInfo
    });
    
    if (response.error) {
      overlay.showNoLyrics('Error fetching lyrics');
      return;
    }
    
    const results: LyricsResult[] = response.data || [];
    
    if (results.length > 0) {
      const result = results[0];
      
      if (result.type === 'synced' && result.syncedLyrics) {
        lyrics = parseLRC(result.syncedLyrics);
        hasSyncedLyrics = true;
        overlay.showLyrics(lyrics, result.source, handleSeek);
        startSync();
      } else if (result.plainLyrics) {
        lyrics = parsePlainLyrics(result.plainLyrics);
        overlay.showPlainLyrics(lyrics, result.source);
      } else {
        overlay.showNoLyrics();
      }
    } else {
      overlay.showNoLyrics();
    }
  } catch (error) {
    console.error('[Everyric] Error searching lyrics:', error);
    overlay.showNoLyrics('Error searching lyrics');
  }
}

function startSync(): void {
  const video = getVideoElement();
  if (!video || lyrics.length === 0 || !overlay) return;
  
  syncRenderer.start(video, lyrics, (index) => {
    overlay?.highlightLine(index);
    overlay?.scrollToLine(index);
  });
}

function handleSeek(time: number): void {
  syncRenderer.seekTo(time);
}

async function fetchSettings(): Promise<Settings> {
  try {
    const response = await chrome.runtime.sendMessage({ type: 'GET_SETTINGS' });
    return response.data || getDefaultSettings();
  } catch {
    return getDefaultSettings();
  }
}

function getDefaultSettings(): Settings {
  return {
    autoSearch: true,
    overlayPosition: 'right',
    fontSize: 'medium',
    showTranslation: false,
    translationLanguage: 'ko',
    useMusixmatch: false,
    theme: 'auto',
    debugMode: false,
    showWordTiming: false,
    showCharTiming: false,
    showMiniSubtitle: false
  };
}

chrome.runtime.onMessage.addListener((message) => {
  if (message.type === 'SYNC_GENERATED' && message.payload.videoId === currentVideoId) {
    searchLyrics();
  }
});

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
