import type { PanelGeometry, Settings } from '../types';

export const DEFAULT_SETTINGS: Settings = {
  autoSearch: true,
  autoSearchShorts: false,
  fontSize: 'medium',
  theme: 'auto',
  serverUrl: 'https://everyric.moref.co',
  offsetSec: 0,
  showTranslation: false,
  translationLanguage: 'ko',
  showPronunciation: true,
  lyricsSourcePriority: 'vocaro',
  pipKeepPanel: true,
  pipShowVideo: true,
  apiKey: '',
  pipVideoRatio: 0,
  pipWidth: 0,
  pipHeight: 0,
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
  pitchF0Curve: true,
  solfegeNotation: 'korean',
  pitchLineOpacity: 1,
  pitchF0Opacity: 1,
  vocalGlow: true,
  pipChromaKey: 'off',
  videoCaptions: false,
  captionFontScale: 1,
  captionBgOpacity: 0.75,
  modNextUp: false,
  modMainLane: false,
  pitchPronPosition: 'note',
  lowConfWarning: true,
  notifyOnComplete: true,
  debugInfo: false,
  pronunciationScript: 'auto',
  uiLanguage: 'auto',
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

// 저장 직렬화 체인(코덱스 감사 Med, 2026-08-03): read-modify-write가 비원자라 토글을
// 빠르게 연타하면 겹친 두 저장 중 나중 것이 먼저 것의 변경을 덮어썼다 — 저장을 한
// 줄로 세워 각 patch가 직전 저장 결과 위에 병합되게 한다. 실패해도 체인은 끊기지
// 않는다(catch 후 다음 저장 진행).
let saveChain: Promise<unknown> = Promise.resolve();

export function saveSettings(patch: Partial<Settings>): Promise<Settings> {
  const next = saveChain
    .catch(() => undefined)
    .then(async () => {
      const merged = { ...(await getSettings()), ...patch };
      try {
        await chrome.storage.local.set({ [SETTINGS_KEY]: merged });
      } catch {
        /* storage 실패 시에도 메모리 값은 유지 */
      }
      return merged;
    });
  saveChain = next;
  return next;
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
