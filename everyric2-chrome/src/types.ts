export interface SongInfo {
  title: string;
  artist: string | null;
  videoId: string;
  duration: number;
}

export interface LyricLine {
  time: number | null;
  text: string;
}

export interface LRCLibResponse {
  id: number;
  trackName: string;
  artistName: string;
  albumName: string;
  duration: number;
  instrumental: boolean;
  plainLyrics: string | null;
  syncedLyrics: string | null;
  lang?: string;
}

export interface LyricsResult {
  source: 'lrclib' | 'everyric' | 'user';
  type: 'synced' | 'plain';
  syncedLyrics?: string;
  plainLyrics?: string;
  trackName?: string;
  artistName?: string;
  duration?: number;
}

export interface EveryricSyncResponse {
  found: boolean;
  sync_id?: string;
  timestamps?: Array<{ start: number; end: number }>;
  lyrics_source?: string;
  quality_score?: number;
  created_at?: string;
  plain_lyrics_available?: boolean;
  suggested_source?: string;
  error?: string;
}

export interface Settings {
  autoSearch: boolean;
  overlayPosition: 'left' | 'right' | 'bottom';
  fontSize: 'small' | 'medium' | 'large';
  showTranslation: boolean;
  translationLanguage: string;
  useMusixmatch: boolean;
  theme: 'auto' | 'dark' | 'light';
}

export type MessageType =
  | { type: 'FETCH_LYRICS'; payload: SongInfo }
  | { type: 'FETCH_EVERYRIC_SYNC'; payload: { videoId: string; lyricsHash?: string } }
  | { type: 'GET_SETTINGS' }
  | { type: 'SAVE_SETTINGS'; payload: Settings }
  | { type: 'SYNC_GENERATED'; payload: { videoId: string } };

export interface MessageResponse<T = unknown> {
  success?: boolean;
  data?: T;
  error?: string;
}
