import type { PanelGeometry, Settings } from '../types';

export const DEFAULT_SETTINGS: Settings = {
  autoSearch: true,
  fontSize: 'medium',
  theme: 'auto',
  serverUrl: 'http://127.0.0.1:8000',
  offsetSec: 0,
  showTranslation: false,
  translationLanguage: 'ko',
  showPronunciation: true,
  lyricsSourcePriority: 'vocaro',
  pipKeepPanel: true,
  pipShowVideo: true,
  apiKey: '',
  pipVideoRatio: 0,
  pitchGuide: true,
  melodyPlayback: false,
  melodyVolume: 0.5,
  metronome: false,
  metronomeVolume: 0.5,
  metronomeRate: 1,
  metronomeBeat: 0,
  audioOutputId: '',
  micPitch: false,
  micDeviceId: '',
  micOctave: 0,
  pitchLaneHeight: 170,
  pitchWindowMeasures: 4,
  pitchScrollMode: 'page',
  pitchFontScale: 1.2,
  pitchCountdown: true,
  debugInfo: false,
};

const SETTINGS_KEY = 'settings';

export async function getSettings(): Promise<Settings> {
  try {
    const stored = await chrome.storage.local.get(SETTINGS_KEY);
    return { ...DEFAULT_SETTINGS, ...(stored[SETTINGS_KEY] as Partial<Settings> | undefined) };
  } catch {
    return { ...DEFAULT_SETTINGS };
  }
}

export async function saveSettings(patch: Partial<Settings>): Promise<Settings> {
  const merged = { ...(await getSettings()), ...patch };
  try {
    await chrome.storage.local.set({ [SETTINGS_KEY]: merged });
  } catch {
    /* storage 실패 시에도 메모리 값은 유지 */
  }
  return merged;
}

function geometryKey(): string {
  return `geometry:${location.host}`;
}

export async function getGeometry(): Promise<PanelGeometry | null> {
  try {
    const key = geometryKey();
    const stored = await chrome.storage.local.get(key);
    return (stored[key] as PanelGeometry | undefined) ?? null;
  } catch {
    return null;
  }
}

export async function saveGeometry(geometry: PanelGeometry): Promise<void> {
  try {
    await chrome.storage.local.set({ [geometryKey()]: geometry });
  } catch {
    /* 저장 실패는 무시 — 다음 세션에 기본 위치 사용 */
  }
}
