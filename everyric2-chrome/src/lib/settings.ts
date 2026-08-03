import type { PanelGeometry, Settings } from '../types';

export const DEFAULT_SETTINGS: Settings = {
  autoSearch: true,
  autoSearchShorts: false,
  fontSize: 'medium',
  mainFontScale: 1,
  theme: 'auto',
  serverUrl: 'https://everyric.moref.co',
  offsetSec: 0,
  showTranslation: false,
  translationLanguage: 'ko',
  showPronunciation: true,
  hidePronForEnglish: false,
  lyricsSourcePriority: 'vocaro',
  pipKeepPanel: true,
  pipShowVideo: true,
  pipLyricsList: false,
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
  modPlaylist: false,
  modMainLane: false,
  pitchPronPosition: 'note',
  lowConfWarning: true,
  notifyOnComplete: true,
  debugInfo: false,
  pronunciationScript: 'auto',
  uiLanguage: 'auto',
  mainLaneWidth: 220,
  mainLanePos: 'left',
  attachedLaneWidth: 300,
  karaokeTimingNoticeDismissed: false,
};

const SETTINGS_KEY = 'settings';

/**
 * 마지막으로 **실제로 읽거나 쓴** 설정 — 저장소 조회가 실패했을 때 기본값 대신 돌려준다.
 *
 * 기본값으로 떨어지는 것 자체는 무해해 보이지만, 그 값이 saveSettings의 병합 바탕이 되면
 * 사용자의 모든 설정이 기본값으로 덮여 **영구히** 사라진다(아래 saveSettings 주석 참고).
 * 세션 안에서 한 번이라도 성공한 조회가 있으면 그 값이 언제나 기본값보다 진실에 가깝다.
 */
let lastKnown: Settings | null = null;

/** 저장소 실패를 삼키지 않는 내부 조회 — 실패는 throw로 올라간다("비어 있음"과 구분). */
async function readSettings(): Promise<Settings> {
  const stored = await chrome.storage.local.get(SETTINGS_KEY);
  const merged = { ...DEFAULT_SETTINGS, ...(stored[SETTINGS_KEY] as Partial<Settings> | undefined) };
  lastKnown = merged;
  return merged;
}

export async function getSettings(): Promise<Settings> {
  try {
    return await readSettings();
  } catch {
    // 읽기 실패 — 이 세션에서 성공한 적이 있으면 그 값이 기본값보다 정확하다
    return { ...(lastKnown ?? DEFAULT_SETTINGS) };
  }
}

// 저장 직렬화 체인(코덱스 감사 Med, 2026-08-03): read-modify-write가 비원자라 토글을
// 빠르게 연타하면 겹친 두 저장 중 나중 것이 먼저 것의 변경을 덮어썼다 — 저장을 한
// 줄로 세워 각 patch가 직전 저장 결과 위에 병합되게 한다. 실패해도 체인은 끊기지
// 않는다(catch 후 다음 저장 진행).
let saveChain: Promise<unknown> = Promise.resolve();

/**
 * 설정 한 조각을 저장한다 — 반환값은 병합된 전체 설정(호출부가 그대로 자기 상태로 쓴다).
 *
 * **읽기가 실패하면 쓰지 않는다.** 예전에는 getSettings()가 조회 실패도 기본값으로 뭉갰고
 * 그 기본값 위에 patch를 얹어 **그대로 저장**했다 — 저장소가 한 번 흔들린 순간(확장 갱신
 * 직후·프로필 잠김·용량 압박) 사용자의 서버 주소·API 키·표시 설정 전부가 기본값으로 덮여
 * 영구히 사라진다. 토글 하나 눌렀을 뿐인 사용자에게 복구 경로가 없는 손실이다. 그래서
 * 읽기 실패는 "빈 저장소"가 아니라 **쓰기를 포기할 사유**로 다룬다: 화면은 메모리 병합값을
 * 받아 그 세션 동안 정상 동작하고, 다음 저장이 성공하면 그때 디스크에 반영된다.
 */
export function saveSettings(patch: Partial<Settings>): Promise<Settings> {
  const next = saveChain
    .catch(() => undefined)
    .then(async () => {
      let base: Settings;
      try {
        base = await readSettings();
      } catch {
        // 읽지 못한 것을 바탕으로 쓰지 않는다 — 메모리 최선값만 돌려주고 저장은 건너뛴다
        const memory = { ...(lastKnown ?? DEFAULT_SETTINGS), ...patch };
        lastKnown = memory;
        return memory;
      }
      const merged = { ...base, ...patch };
      try {
        await chrome.storage.local.set({ [SETTINGS_KEY]: merged });
      } catch {
        /* 쓰기만 실패 — 읽은 값은 유효하므로 메모리 값은 그대로 진행한다 */
      }
      lastKnown = merged;
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
