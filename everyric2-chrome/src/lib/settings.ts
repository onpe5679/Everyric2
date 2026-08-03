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
  // 기본은 «PiP를 열면 메인 패널은 물러난다» — PiP의 정체성이 «영상 + 지금 부르는 줄»이라
  // 운영자가 정했다. 동시 표시는 설정으로 남아 있다(기능이 사라진 것이 아니다).
  pipKeepPanel: false,
  pipShowVideo: true,
  apiKey: '',
  pipVideoRatio: 0,
  pipShortLyrics: true,
  pipShowPanel: true,
  pipShowCenter: true,
  settingsMigrations: [],
  pipPlaylist: true,
  pipLaneWidth: 300,
  pipLaneSwapped: false,
  pipPanelWidth: 360,
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
  streamTextOutline: false,
  videoCaptions: false,
  captionFontScale: 1,
  captionBgOpacity: 0.75,
  modNextUp: false,
  modPlaylist: false,
  modMainLane: false,
  // 노트 위 음절은 이제 설정과 무관하게 항상 나온다 — 이 값은 «이중표시 줄»만 정한다.
  // 기본 off = 예전 기본('note')과 화면이 같다(노트만, 아래 줄 없음).
  pitchPronPosition: 'off',
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
/**
 * 저장된 값 중 **의미가 바뀐 것**을 새 어휘로 옮긴다 — 읽는 길목 한 곳에서만.
 *
 * pitchPronPosition: 예전엔 «노트 부착»까지 이 값이 정했다('note'). 노트 텍스트가 설정에서
 * 분리돼 항상 표시되도록 바뀌면서(운영자 지시 2026-08-04) 이 값은 이중표시 줄만 정한다.
 *   'note' → 'off'  (노트만 나오던 화면 = 이제 줄 없음. 화면이 그대로다)
 *   'off'  → 'off'  (**함정 자동 구제**: 노트가 비어 있던 사용자는 노트가 되살아난다)
 *   'both' → 'bottom' (노트+하단이었으니 이제 하단만 남기면 같은 화면)
 *   bottom/center는 그대로. 새 'both'는 하단+중앙이라는 새 뜻이다.
 *
 * 마이그레이션은 파괴적이지 않다 — 다음 저장 때 새 값이 디스크에 남고, 그 전까지는
 * 메모리에서만 옮겨진 채 동작한다.
 */
/**
 * 1회성 마이그레이션 id — 실행하면 settingsMigrations에 남아 두 번 돌지 않는다.
 *
 * pipKeepPanel: 기본값을 true→false로 뒤집었지만(운영자: "그게 기본 동작이면 안 된다"),
 * saveSettings가 **전체 객체**를 저장하는 구조라 설정을 한 번이라도 바꾼 사용자는 옛
 * 기본값 true를 그대로 물고 있다. 그 true는 «흡수한 기본값»이지 의도 선택이 아니고,
 * 값만 봐서는 둘을 구분할 수 없다 — 그래서 딱 한 번 내리고 마커를 남긴다.
 * 이후 사용자가 설정에서 다시 켜면 마커가 있으니 다시 내리지 않는다.
 */
const MIGRATION_PIP_KEEP_PANEL = 'pipKeepPanel-default-false-2026-08';

function migrateSettings(s: Settings): { settings: Settings; changed: boolean } {
  let out = s;
  let changed = false;

  // 레인 발음 위치 어휘 변경 — 값 자체가 새 어휘에 없으므로 마커 없이 값으로 판정한다
  // (노트 위 음절이 설정에서 분리되면서 'note'·구 'both'가 뜻을 잃었다)
  const pos = out.pitchPronPosition as string;
  if (pos === 'note') { out = { ...out, pitchPronPosition: 'off' }; changed = true; }
  else if (pos === 'both') { out = { ...out, pitchPronPosition: 'bottom' }; changed = true; }

  // pipKeepPanel 1회성 하향 — 마커가 없을 때만
  const done = Array.isArray(out.settingsMigrations) ? out.settingsMigrations : [];
  if (!done.includes(MIGRATION_PIP_KEEP_PANEL)) {
    out = {
      ...out,
      // 이미 false면 값은 그대로 두고 마커만 남긴다(다음부터 아예 안 본다)
      pipKeepPanel: false,
      settingsMigrations: [...done, MIGRATION_PIP_KEEP_PANEL],
    };
    changed = true;
  }
  return { settings: out, changed };
}

/** 마지막 읽기에서 마이그레이션이 값을 실제로 바꿨는가 — getSettings가 이때만 되쓴다 */
let lastReadMigrated = false;

async function readSettings(): Promise<Settings> {
  const stored = await chrome.storage.local.get(SETTINGS_KEY);
  const { settings: merged, changed } = migrateSettings(
    { ...DEFAULT_SETTINGS, ...(stored[SETTINGS_KEY] as Partial<Settings> | undefined) });
  lastReadMigrated = changed;
  lastKnown = merged;
  return merged;
}

export async function getSettings(): Promise<Settings> {
  try {
    const merged = await readSettings();
    // 마이그레이션 결과를 **되쓴다**. 안 쓰면 마커가 디스크에 안 남아 매 로드마다 다시
    // 돌고, 사용자가 pipKeepPanel을 다시 켜도 다음 로드에서 도로 꺼진다(되돌린 선택을
    // 존중하지 못한다). 쓰기는 saveSettings 체인에 실어 다른 저장과 뒤엉키지 않게 한다.
    if (lastReadMigrated) {
      lastReadMigrated = false;
      await saveSettings({});
    }
    return lastKnown ?? merged;
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
