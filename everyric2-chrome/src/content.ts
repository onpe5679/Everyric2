import { detectSong, getCurrentVideoId, getVideoElement } from './lib/song-detector';
import { SyncEngine, type SyncHandlers } from './lib/sync-engine';
import { KaraokeAudio, collectMelodyNotes } from './lib/karaoke-audio';
import { MicPitch } from './lib/mic-pitch';
import { getGeometry, getSettings, saveGeometry, saveSettings } from './lib/settings';
import { LyricsOverlay } from './ui/overlay';
import { PipController } from './ui/pip';
import type {
  BgRequest,
  CaptionLine,
  CaptionTrack,
  ContentMessage,
  GenerateResponse,
  JobStatusResponse,
  LyricLine,
  LyricsData,
  MessageResponse,
  PanelGeometry,
  SearchCandidate,
  Settings,
  SongInfo,
  SyncListItem,
  TranslateResult,
  TranslatedLine,
} from './types';
import type { VocaroLine, VocaroResult } from './lib/vocaro';

let settings: Settings;
let cssText = '';
let initialGeometry: PanelGeometry | null = null;
let overlay: LyricsOverlay | null = null;
const pip = new PipController();
const engine = new SyncEngine();
const karaokeAudio = new KaraokeAudio(() => engine.getVideo() ?? getVideoElement());
const micPitch = new MicPitch();

let currentVideoId: string | null = null;
let currentSong: SongInfo | null = null;
let currentData: LyricsData | null = null;
let currentSourceUrl: string | null = null; // 보카로 위키 출처 페이지 (CC BY 출처 표기용)
let lastVocaro: { videoId: string; lines: VocaroLine[] } | null = null; // 싱크 생성 후 발음/번역 재병합용
/** 진행 중인 전사 잡 — videoId 키. 영상을 이동해도 백그라운드로 계속 추적한다 */
const generatingJobs = new Map<string, {
  jobId: string; progress: number; queueLabel?: string;
  stage?: string; stageProgress?: number; title?: string;
}>();
// 생성 요청 준비 단계(LLM 번역·독음 대기, 수십 초) 중인 영상 — 잡 등록 전이라
// generatingJobs가 비어 있어, 이 가드가 없으면 버튼 연타가 전부 서버로 나간다
const preparingGenerate = new Set<string>();
// 진행 중 잡을 탭 간 공유하는 storage 키 — 다른 탭/새 탭에서도 진행 칩이 이어진다
const JOBS_STORAGE_KEY = 'activeJobs';
// 현재 영상의 사용자 싱크 오프셋(초) — 영상마다 서버에 저장·복원된다 (전역 설정 아님)
let videoOffset = 0;
let offsetSaveTimer: number | undefined;
let pollTimer: number | undefined;
let searchSeq = 0;
let lastLineIndex = -1;
let lastDebugPush = 0;
const translationCache = new Map<string, TranslatedLine[]>(); // `${videoId}:${lang}` → 라인별 번역+발음
// 같은 곡의 번역 요청이 동시에 여러 갈래(표시 경로·생성 경로)에서 뜨면 하나로 합친다 —
// LLM 호출은 수십 초짜리라 중복이 곧 서버 스레드 낭비 + 진행 지연이다
const pendingTranslate = new Map<string, Promise<TranslatedLine[] | undefined>>();

async function init(): Promise<void> {
  settings = await getSettings();
  [cssText, initialGeometry] = await Promise.all([loadCss(), getGeometry()]);
  chrome.runtime.onMessage.addListener(handleRuntimeMessage);
  await restoreActiveJobs();
  observeNavigation();
  checkCurrentPage();
}

/** 다른 탭(또는 이전 세션)이 시작한 전사 잡을 이어받아 진행 칩·폴링을 계속한다 */
async function restoreActiveJobs(): Promise<void> {
  try {
    const stored = await chrome.storage.local.get(JOBS_STORAGE_KEY);
    const jobs = stored[JOBS_STORAGE_KEY] as
      Record<string, { jobId: string; title?: string }> | undefined;
    if (!jobs) return;
    for (const [videoId, job] of Object.entries(jobs)) {
      if (job?.jobId && !generatingJobs.has(videoId)) {
        generatingJobs.set(videoId, { jobId: job.jobId, progress: 0, title: job.title });
      }
    }
    if (generatingJobs.size > 0) ensurePolling();
  } catch { /* storage 실패 — 이 탭에서 시작한 잡만 추적 */ }
}

/** 진행 중 잡 목록을 storage에 반영 — 다른 탭이 이어받을 수 있게 */
function persistActiveJobs(): void {
  try {
    void chrome.storage.local.set({
      [JOBS_STORAGE_KEY]: Object.fromEntries(
        [...generatingJobs].map(([v, j]) => [v, { jobId: j.jobId, title: j.title }]),
      ),
    });
  } catch { /* storage 실패는 치명적이지 않다 */ }
}

async function loadCss(): Promise<string> {
  try {
    const res = await fetch(chrome.runtime.getURL('overlay.css'));
    return await res.text();
  } catch {
    return '';
  }
}

function handleRuntimeMessage(message: ContentMessage): void {
  if (message.type === 'TOGGLE_OVERLAY') {
    void toggleOverlay();
  } else if (message.type === 'SYNC_GENERATED' && message.payload.videoId === currentVideoId) {
    void searchLyrics();
  }
}

function observeNavigation(): void {
  document.addEventListener('yt-navigate-finish', () => window.setTimeout(checkCurrentPage, 300));
  window.setInterval(checkCurrentPage, 1500);
  window.setInterval(watchVideoBinding, 3000);
}

/**
 * YouTube가 광고/프리뷰 등으로 video 엘리먼트를 교체하거나 엔진이 재생 중이
 * 아닌 video에 붙은 경우, 실제 재생 중인 video로 자동 재바인딩한다.
 * (증상: 가사는 뜨는데 하이라이트가 재생 시간을 따라오지 않음)
 */
function watchVideoBinding(): void {
  if (!currentData?.synced || !engine.isRunning()) return;
  const video = getVideoElement();
  if (video && video !== engine.getVideo()) {
    engine.start(video, currentData.lines, makeEngineHandlers());
    engine.setOffset(videoOffset);
    // 미러 스트림도 새 video 기준으로 갱신
    if (pip.isOpen() && settings.pipShowVideo) pip.attachVideo(video);
  }
}

/** 자동 검색이 꺼져 있으면 사용자가 패널을 열어둔 경우에만 따라간다.
 * 자동 검색이 켜져 있어도 음악 영상으로 판별될 때만 자동으로 뜬다 —
 * 브이로그/게임 영상에서 노래를 찾겠다고 패널이 뜨는 것을 막는다. */
function shouldFollow(): boolean {
  if (overlay?.isVisible()) return true; // 사용자가 열어둔 패널은 항상 따라간다
  return settings.autoSearch && isLikelyMusicVideo();
}

/** 음악 영상 판별 — 유튜브 자체 신호 우선, 없으면 채널/제목 휴리스틱 */
function isLikelyMusicVideo(): boolean {
  // 1) 설명란 '음악' 섹션 (콘텐츠 ID로 곡이 식별된 영상) — 가장 신뢰
  if (document.querySelector('ytd-video-description-music-section-renderer')) return true;
  // 2) 워치 페이지 microdata 장르 — 있으면 그대로 믿는다 (Music이 아니면 차단)
  const genre = document.querySelector<HTMLMetaElement>('meta[itemprop="genre"]');
  if (genre?.content) {
    const g = genre.content.trim().toLowerCase();
    return g === 'music' || g === '음악';
  }
  // 3) 자동 생성 음악 채널(" - Topic")
  const channel = document.querySelector('ytd-watch-metadata ytd-channel-name a')?.textContent?.trim() ?? '';
  if (/ - Topic$/.test(channel)) return true;
  // 4) 제목 휴리스틱 — MV/가사/커버/보컬로이드 계열 표기
  const title = document.title;
  return /(M\/?V|Official\s*(Music\s*)?Video|뮤직\s*비디오|가사|lyrics?|\bcover(ed)?\b|커버|불러보았다|歌ってみた|feat\.|ft\.|【[^】]*(MV|PV|오리지널|Original)[^】]*】)/i.test(title);
}

function checkCurrentPage(): void {
  const videoId = getCurrentVideoId();
  if (!videoId) {
    cleanupForPage();
    return;
  }
  if (videoId === currentVideoId || !shouldFollow()) return;
  currentVideoId = videoId;
  void searchLyrics();
}

function cleanupForPage(): void {
  if (currentVideoId === null) return;
  currentVideoId = null;
  currentSong = null;
  currentData = null;
  currentSourceUrl = null;
  videoOffset = 0;
  clearTimeout(offsetSaveTimer);
  // 전사 잡은 서버에서 계속 돌므로 추적을 유지한다 (완료 시 해당 영상으로 돌아오면 반영)
  engine.stop();
  pip.close();
  karaokeAudio.setNotes([]);
  karaokeAudio.setTempo(null);
  overlay?.setVisible(false);
}

async function toggleOverlay(): Promise<void> {
  if (overlay?.isVisible()) {
    overlay.setVisible(false);
    return;
  }
  const videoId = getCurrentVideoId();
  if (!videoId) return;
  ensureOverlay().setVisible(true);
  if (videoId !== currentVideoId || !currentData) {
    currentVideoId = videoId;
    await searchLyrics();
  }
}

function ensureOverlay(): LyricsOverlay {
  if (overlay) return overlay;
  overlay = new LyricsOverlay(cssText, settings, {
    onSeek: time => engine.seekTo(time),
    onGenerate: text => void handleGenerate(text),
    onRetrySearch: query => void searchLyrics(query),
    onOffsetChange: offsetSec => {
      // 오프셋은 영상별 상태 — 서버에 저장해 다음 시청·다른 기기에서도 복원된다.
      // 링크로 빌려온 싱크(inst/커버)도 보는 영상 기준이라 영상마다 따로 저장된다.
      engine.setOffset(offsetSec);
      videoOffset = offsetSec;
      scheduleOffsetSave();
    },
    onCloseSearch: () => {
      applyLyricsData(currentData);
      updateGenChip();
    },
    onSettingsChange: patch => void handleSettingsChange(patch),
    onRegenerate: () => void handleRegenerate(),
    onPipToggle: () => void handlePipToggle(),
    onGeometryChange: geometry => void saveGeometry(geometry),
    onCandidateSearch: query => void handleCandidateSearch(query),
    onPickCandidate: candidate => void handlePickCandidate(candidate),
    onLinkSync: (sourceVideoId, offsetSec) => void handleLinkSync(sourceVideoId, offsetSec),
    onUnlinkSync: () => void handleUnlinkSync(),
    onRequestSyncList: () => void handleRequestSyncList(),
    onResetSync: () => void handleResetSync(),
    onCaptionTracks: () => void handleCaptionTracks(),
    onCaptionPick: track => void handleCaptionPick(track),
  }, initialGeometry);
  return overlay;
}

/** 오프셋 변경을 디바운스해 서버에 저장 (연타 중 매 클릭 요청 방지) */
function scheduleOffsetSave(): void {
  const videoId = currentVideoId;
  if (!videoId) return;
  clearTimeout(offsetSaveTimer);
  offsetSaveTimer = window.setTimeout(() => {
    void sendToBackground({ type: 'SYNC_OFFSET', payload: { videoId, offsetSec: videoOffset } });
  }, 800);
}

// ── 유튜브 자막 → 가사 붙여넣기 칸 ─────────────────────────────

async function handleCaptionTracks(): Promise<void> {
  const videoId = currentVideoId;
  if (!videoId) return;
  const res = await sendToBackground<CaptionTrack[]>({
    type: 'YT_CAPTION_TRACKS', payload: { videoId },
  });
  if (videoId !== currentVideoId) return;
  overlay?.showCaptionTracks(res.data ?? []);
}

async function handleCaptionPick(track: CaptionTrack): Promise<void> {
  const videoId = currentVideoId;
  if (!videoId) return;
  const res = await sendToBackground<CaptionLine[]>({
    type: 'YT_CAPTION_TEXT', payload: { videoId, lang: track.lang, auto: track.auto },
  });
  if (videoId !== currentVideoId) return;
  if (!res.data || res.data.length === 0) {
    overlay?.setCaptionStatus('자막을 불러오지 못했어요 — 서버 상태를 확인하거나 다른 트랙을 시도해 보세요');
    return;
  }
  // 자막 타이밍을 그대로 싱크 가사로 표시한다 — 자막이 가사가 아닌 영상이면 눈으로
  // 바로 확인되고, 그때는 재검색으로 되돌리면 된다. AI 전사(음정·발음)는 배너로 이어진다.
  const lines: LyricLine[] = res.data.map(l => ({
    time: l.start,
    endTime: l.end,
    text: l.text,
  }));
  applyLyricsData({
    source: 'caption',
    synced: true,
    lines,
    plainText: lines.map(l => l.text).join('\n'),
    attribution: { name: track.label },
  });
}

// ── 싱크 링크 (inst·커버 영상이 다른 영상의 전사를 재사용) ──────────

async function handleLinkSync(sourceVideoId: string, offsetSec: number): Promise<void> {
  const videoId = currentVideoId;
  if (!videoId) return;
  if (sourceVideoId === videoId) {
    ensureOverlay().setLinkStatus('자기 자신에게는 연결할 수 없어요');
    return;
  }
  // 자체 전사가 있으면 조회가 링크보다 자체 전사를 우선해 연결이 무시된다 —
  // 사용자가 명시적으로 연결을 원했으니 확인 후 자체 전사를 지우고 연결한다
  if (currentData?.synced && currentData.source === 'everyric' && !currentData.linked) {
    const ok = window.confirm(
      '이 영상에는 자체 전사가 이미 있어요.\n연결하면 자체 전사를 삭제하고 원본 영상의 싱크를 대신 사용합니다. 계속할까요?',
    );
    if (!ok) {
      ensureOverlay().setLinkStatus('연결 취소됨 — 자체 전사를 유지합니다');
      return;
    }
    const reset = await sendToBackground<{ removed_syncs: number }>({
      type: 'SYNC_RESET', payload: { videoId },
    });
    if (reset.error) {
      ensureOverlay().setLinkStatus('자체 전사 삭제에 실패했어요 — 서버 상태를 확인해 주세요');
      return;
    }
    for (const key of [...translationCache.keys()]) {
      if (key.startsWith(`${videoId}:`)) translationCache.delete(key);
    }
  }
  const res = await sendToBackground<Record<string, unknown>>({
    type: 'SYNC_LINK',
    payload: { videoId, sourceVideoId, offsetSec },
  });
  if (videoId !== currentVideoId) return;
  if (res.error || !res.data) {
    ensureOverlay().setLinkStatus('연결 실패 — 원본 영상에 전사(싱크)가 있는지 확인해 주세요');
    return;
  }
  void searchLyrics(); // 링크된 싱크를 즉시 불러온다
}

async function handleUnlinkSync(): Promise<void> {
  const videoId = currentVideoId;
  if (!videoId) return;
  const res = await sendToBackground<{ removed: boolean }>({ type: 'SYNC_UNLINK', payload: { videoId } });
  if (videoId !== currentVideoId) return;
  if (res.error) {
    ensureOverlay().setLinkStatus('해제 실패 — 서버 상태를 확인해 주세요');
    return;
  }
  ensureOverlay().setLinked(null);
  void searchLyrics();
}

async function handleRequestSyncList(): Promise<void> {
  const res = await sendToBackground<SyncListItem[]>({ type: 'SYNC_LIST' });
  overlay?.showSyncList(res.data ?? []);
}

async function handleSettingsChange(patch: Partial<Settings>): Promise<void> {
  settings = await saveSettings(patch);
  overlay?.applySettings(settings);
  if (patch.serverUrl !== undefined) void refreshServerStatus();

  if (patch.showTranslation === true || (patch.translationLanguage && settings.showTranslation)) {
    void loadTranslations();
  } else if (patch.showTranslation === false) {
    clearTranslations();
  }

  // PiP 사용 중 패널 유지 설정을 토글하면 즉시 반영
  if (patch.pipKeepPanel !== undefined && pip.isOpen() && currentData?.synced) {
    if (patch.pipKeepPanel) {
      applyLyricsData(currentData);
    } else {
      overlay?.showPipPlaceholder();
    }
  }

  // PiP 영상 표시 토글 즉시 반영
  if (patch.pipShowVideo !== undefined && pip.isOpen()) {
    pip.setVideoEnabled(patch.pipShowVideo, engine.getVideo() ?? getVideoElement());
  }

  // 발음 표기 토글 즉시 반영 (패널은 applySettings에서 처리됨)
  if (patch.showPronunciation !== undefined) {
    pip.setShowPronunciation(patch.showPronunciation);
  }

  // 디버그 토글 → 레인 신뢰도 색상도 함께
  if (patch.debugInfo !== undefined) {
    pip.setShowConfidence(patch.debugInfo);
  }

  // 레인 표시 구간/진행 방식/글자 크기/카운트다운 즉시 반영
  if (patch.pitchWindowMeasures !== undefined) {
    pip.setPitchWindow(patch.pitchWindowMeasures);
  }
  if (patch.pitchScrollMode !== undefined) {
    pip.setPitchScrollMode(patch.pitchScrollMode);
  }
  if (patch.pitchFontScale !== undefined) {
    pip.setPitchFontScale(patch.pitchFontScale);
  }
  if (patch.pitchCountdown !== undefined) {
    pip.setPitchCountdown(patch.pitchCountdown);
  }

  // 가라오케 음정 바 토글 즉시 반영
  if (patch.pitchGuide !== undefined) {
    pip.setPitchEnabled(patch.pitchGuide);
  }

  // 멜로디/메트로놈/마이크 — 토글·볼륨·배속·시작박·기기 변경 즉시 반영
  if (
    patch.melodyPlayback !== undefined || patch.melodyVolume !== undefined ||
    patch.metronome !== undefined || patch.metronomeVolume !== undefined ||
    patch.metronomeRate !== undefined || patch.metronomeBeat !== undefined ||
    patch.audioOutputId !== undefined || patch.micPitch !== undefined ||
    patch.micDeviceId !== undefined
  ) {
    applyAudioSettings();
  }
  if (patch.metronomeRate !== undefined || patch.metronomeBeat !== undefined) {
    pip.setMetronomeConfig(settings.metronomeRate, settings.metronomeBeat);
  }
  if (patch.micOctave !== undefined) {
    pip.setMicOctave(settings.micOctave);
  }

  // 저신뢰 경고 토글 즉시 반영
  if (patch.lowConfWarning !== undefined) {
    overlay?.setQualityWarning(
      settings.lowConfWarning && currentData?.synced && currentData.source === 'everyric'
        && currentData.qualityScore != null && currentData.qualityScore < 0.001
        ? currentData.qualityScore
        : null,
    );
  }

  if (patch.debugInfo === true) pushDebug(null);
}

/** 멜로디/메트로놈/마이크 상태를 설정에 맞춰 동기화 — 가라오케 창(PiP)이 열려 있을 때만 소리·검출 */
function applyAudioSettings(): void {
  karaokeAudio.configure({
    melody: settings.melodyPlayback,
    melodyVolume: settings.melodyVolume,
    metronome: settings.metronome,
    metronomeVolume: settings.metronomeVolume,
    metronomeRate: settings.metronomeRate,
    metronomeBeat: settings.metronomeBeat,
    sinkId: settings.audioOutputId,
  });
  pip.setAudioState(settings.melodyPlayback, settings.metronome);
  if (pip.isOpen() && settings.micPitch) {
    if (micPitch.isRunning() && micPitch.currentDeviceId() !== settings.micDeviceId) micPitch.stop();
    if (!micPitch.isRunning()) void micPitch.start(settings.micDeviceId || undefined);
  } else if (micPitch.isRunning()) {
    micPitch.stop();
  }
}

/** 현재 시각이 어떤 구간으로 판정됐는지 (star 흡수/가창/간주) — everyric 소스만 */
function debugZoneAt(time: number | null): string | null {
  const meta = currentData?.debugMeta;
  if (!meta || time === null) return null;
  const t = time - videoOffset;
  if (meta.star_spans?.some(([s, e]) => t >= s && t < e)) return '추임새★';
  if (meta.vad_regions == null) return null;
  return meta.vad_regions.some(([s, e]) => t >= s && t < e) ? '가창' : '간주·무성';
}

/** 디버그 스트립에 현재 내부 상태를 밀어넣는다 (설정 꺼져 있으면 no-op) */
function pushDebug(time: number | null): void {
  if (!settings.debugInfo || !overlay) return;
  const bound = engine.getVideo();
  const dom = getVideoElement();
  const video = bound ?? dom;
  const line = lastLineIndex >= 0 ? currentData?.lines[lastLineIndex] : undefined;
  const lineDebug = line?.debug
    ? `act=${Math.round((line.debug.activeRatio ?? 0) * 100)}%${line.debug.clamped ? ' CLAMP' : ''}`
    : null;
  overlay.updateDebug({
    zone: debugZoneAt(time ?? (video ? video.currentTime : null)),
    lineDebug,
    videoId: currentVideoId,
    source: currentData?.source ?? '-',
    synced: currentData?.synced ?? false,
    time: time ?? (video ? video.currentTime : null),
    offsetSec: videoOffset,
    lineIndex: lastLineIndex,
    lineCount: currentData?.lines.length ?? 0,
    videoBound: bound !== null && (dom === null || bound === dom),
    videoInfo: video ? `rs${video.readyState},${video.paused ? 'pause' : 'play'}` : 'none',
    engineRunning: engine.isRunning(),
    pipOpen: pip.isOpen(),
    jobStatus: currentJobStatus(),
    quality: currentData?.qualityScore ?? null,
    ...lineConfSummary(),
    alignmentText: currentData?.debugMeta?.alignment_text ?? null,
  });
}

function currentJobStatus(): string | null {
  const cur = currentVideoId ? generatingJobs.get(currentVideoId) : undefined;
  if (cur) return `job=${cur.jobId.slice(0, 8)}(${cur.progress}%)`;
  return generatingJobs.size > 0 ? `bg-jobs=${generatingJobs.size}` : null;
}

/** 라인 confidence의 median·등급 분포 — 곡 전체 정렬 품질 요약 (디버그 표시용) */
function lineConfSummary(): {
  qualityMed: number | null;
  lowConfRatio: number | null;
  confGrades: { ok: number; mid: number; low: number } | null;
} {
  const vals = (currentData?.lines ?? [])
    .map(l => l.confidence)
    .filter((v): v is number => v != null)
    .sort((a, b) => a - b);
  if (vals.length === 0) return { qualityMed: null, lowConfRatio: null, confGrades: null };
  const low = vals.filter(v => v < 1e-4).length / vals.length;
  const mid = vals.filter(v => v >= 1e-4 && v < 2e-2).length / vals.length;
  return {
    qualityMed: vals[Math.floor(vals.length / 2)],
    lowConfRatio: low,
    confGrades: { ok: 1 - low - mid, mid, low },
  };
}

/**
 * 서버 싱크(everyric) 라인에 보카로 위키의 발음/사람 번역을 텍스트 매칭으로 입힌다.
 * 싱크가 위키 가사로 생성됐다면 라인 텍스트가 그대로 보존되므로 대부분 1:1로 매칭된다.
 */
async function enrichFromVocaro(videoId: string, data: LyricsData): Promise<void> {
  let lines: VocaroLine[] | null = lastVocaro?.videoId === videoId ? lastVocaro.lines : null;
  if (!lines) {
    let slug: string | null = null;
    try {
      const stored = await chrome.storage.local.get(`vocaroRef:${videoId}`);
      slug = (stored[`vocaroRef:${videoId}`] as string | undefined) ?? null;
    } catch { /* storage 실패 → 병합 생략 */ }
    if (!slug) return;
    const res = await sendToBackground<VocaroResult | null>({ type: 'VOCARO_PAGE', payload: { slug } });
    lines = res.data?.lines ?? null;
    if (lines) lastVocaro = { videoId, lines };
  }
  if (!lines) return;

  const norm = (s: string) => s.replace(/\s+/g, ' ').trim();
  const byText = new Map<string, VocaroLine>();
  for (const l of lines) {
    if (l.text && !byText.has(norm(l.text))) byText.set(norm(l.text), l);
  }
  for (const line of data.lines) {
    const v = byText.get(norm(line.text));
    if (!v) continue;
    if (v.pronunciation && !line.pronunciation) line.pronunciation = v.pronunciation;
    if (v.translation && !line.translation) {
      line.translation = v.translation;
      data.humanTranslated = true;
    }
  }
}

function clearTranslations(): void {
  overlay?.setTranslationStatus(null);
  if (!currentData) return;
  if (currentData.source === 'vocaro' || currentData.humanTranslated) return; // 사람 번역은 가사 자체의 일부 — 지우지 않는다
  for (const line of currentData.lines) delete line.translation;
  overlay?.refreshTranslations();
  pip.refresh();
}

async function loadTranslations(): Promise<void> {
  const data = currentData;
  const videoId = currentVideoId;
  if (!data || !videoId || !settings.showTranslation) return;
  if (data.source === 'vocaro' || data.humanTranslated) return; // 위키가 이미 사람 번역을 제공
  // 서버 싱크에 번역·발음이 이미 저장돼 있으면(생성 시 LLM 메타 병합) LLM 재호출 생략.
  // 단, 발음이 기대되는 원문(일본어 등 CJK)인데 발음이 하나도 없으면 — 번역만 저장된
  // 낡은 싱크 — 발음까지 다시 받아온다 (그냥 반환하면 발음이 영영 채워지지 않는다)
  const expectsPron = expectsPronunciation(data.lines.map(l => l.text));
  if (
    data.lines.every(l => l.translation)
    && (data.lines.some(l => l.pronunciation) || !expectsPron)
  ) return;

  const lang = settings.translationLanguage;
  const cached = translationCache.get(`${videoId}:${lang}`);
  // 캐시도 같은 기준으로 검증 — 발음 빠진 캐시(구버전 응답)는 다시 받아온다
  if (cached && (!expectsPron || cached.some(l => l.pronunciation))) {
    applyTranslations(data, cached);
    return;
  }

  overlay?.setTranslationStatus('번역·발음 생성 중…');
  const lines = await requestTranslation(videoId, data.lines.map(l => l.text));
  if (currentData !== data || currentVideoId !== videoId) return; // 곡이 바뀜
  if (!settings.showTranslation || settings.translationLanguage !== lang) return;

  if (!lines || lines.length === 0) {
    overlay?.setTranslationStatus('번역 실패 — 서버 확인');
    return;
  }
  applyTranslations(data, lines);
}

function applyTranslations(data: LyricsData, translated: TranslatedLine[]): void {
  let pronApplied = false;
  data.lines.forEach((line, i) => {
    const t = translated[i]?.translation?.trim();
    // '[NO API KEY]'는 구버전 서버의 키 미설정 플레이스홀더 — 번역으로 표시하지 않는다
    if (t && t !== line.text && !t.startsWith('[NO API KEY]')) line.translation = t;
    // 발음표기(target=ko면 한글 독음) — 사람이 단 발음(보카로 위키)이 있으면 건드리지 않는다
    const p = translated[i]?.pronunciation?.trim();
    if (p && !line.pronunciation) {
      line.pronunciation = p;
      pronApplied = true;
    }
  });
  overlay?.setTranslationStatus(null);
  overlay?.refreshTranslations();
  // 발음이 새로 붙었으면 PiP 내부 변환 캐시(setLines 시점 복사)도 다시 채운다
  if (pronApplied && currentData === data) pip.setLines(data.lines);
  pip.refresh();
}

/** 원문에 CJK(가나·한자 등)가 실질적으로 있으면 발음표기(한글 독음)가 기대되는 곡 */
function expectsPronunciation(texts: string[]): boolean {
  const cjk = texts.join('').match(/[぀-ヿ㐀-鿿]/g);
  return (cjk?.length ?? 0) >= 5;
}

/** 서버 번역(발음 포함) 요청 — video+언어 기준으로 동시 요청을 하나로 합치고 캐시에 저장 */
function requestTranslation(
  videoId: string, srcLines: string[],
): Promise<TranslatedLine[] | undefined> {
  const key = `${videoId}:${settings.translationLanguage}`;
  const inFlight = pendingTranslate.get(key);
  if (inFlight) return inFlight;
  const p = (async () => {
    const res = await sendToBackground<TranslateResult>({
      type: 'TRANSLATE',
      payload: {
        text: srcLines.join('\n'),
        targetLang: settings.translationLanguage,
        title: currentSong?.title,
        artist: currentSong?.artist ?? undefined,
      },
    });
    const lines = res.data?.lines;
    if (lines && lines.length > 0) {
      translationCache.set(key, lines);
      // 장시간 세션 메모리 상한 — 가장 오래된 항목부터 축출 (Map은 삽입 순서 유지)
      while (translationCache.size > 24) {
        const oldest = translationCache.keys().next().value;
        if (oldest === undefined) break;
        translationCache.delete(oldest);
      }
    }
    return lines && lines.length > 0 ? lines : undefined;
  })().finally(() => pendingTranslate.delete(key));
  pendingTranslate.set(key, p);
  return p;
}

/** LLM 번역·한글 독음을 받아 line_meta로 변환 — 캐시 우선, 실패 시 undefined(원문 정렬 폴백).
 *  LLM이 echo한 original 대신 넘겨받은 원문으로 인덱스 매핑한다 (서버 병합은 텍스트 매칭이라
 *  원문이 정확해야 하고, 서버 번역도 같은 규칙으로 줄을 나누므로 인덱스가 일치). */
async function fetchLlmLineMeta(
  videoId: string, srcLines: string[],
): Promise<{ text: string; pronunciation?: string; translation?: string }[] | undefined> {
  const lang = settings.translationLanguage;
  try {
    overlay?.setTranslationStatus('AI 번역·독음 생성 중…');
    let translated = translationCache.get(`${videoId}:${lang}`);
    // 발음이 빠진 캐시(구버전 응답 등)는 다시 받아온다
    if (
      !translated || translated.length !== srcLines.length
      || (expectsPronunciation(srcLines) && !translated.some(l => l.pronunciation))
    ) {
      translated = await requestTranslation(videoId, srcLines);
    }
    if (translated && translated.length > 0) {
      return srcLines
        .map((t, i) => ({
          text: t,
          pronunciation: translated![i]?.pronunciation?.trim() || undefined,
          translation: translated![i]?.translation?.trim() || undefined,
        }))
        .filter(m => m.pronunciation || m.translation);
    }
  } catch { /* 번역 실패 — 메타 없이 진행 */ } finally {
    if (videoId === currentVideoId) overlay?.setTranslationStatus(null);
  }
  return undefined;
}

async function searchLyrics(queryOverride?: { title: string; artist: string }): Promise<void> {
  const videoId = currentVideoId;
  if (!videoId) return;
  const seq = ++searchSeq;
  const panel = ensureOverlay();
  panel.setVisible(true);
  panel.showLoading();
  updateGenChip(); // 이 영상(또는 다른 영상)의 전사 진행 칩은 검색과 무관하게 유지
  engine.stop();

  void refreshServerStatus();

  let song: SongInfo | null;
  if (queryOverride) {
    song = {
      title: queryOverride.title,
      artist: queryOverride.artist || null,
      videoId,
      duration: currentSong?.duration ?? Math.round(getVideoElement()?.duration ?? 0),
    };
  } else {
    song = await waitForSongInfo(seq);
  }
  if (seq !== searchSeq || videoId !== currentVideoId) return;

  // 곡 인식 시점엔 video 메타데이터가 아직 없을 수 있음 — duration 없이 LRCLIB에
  // 조회하면 길이가 다른 버전이 매칭될 수 있으므로 한 번 더 읽는다
  if (song && song.duration === 0) {
    const d = getVideoElement()?.duration;
    song.duration = d && Number.isFinite(d) ? Math.round(d) : 0;
  }

  if (!song) {
    panel.setSong(null);
    panel.showEmpty(null);
    return;
  }
  currentSong = song;
  panel.setSong(song);

  // 소스 우선순위: 서버 싱크는 항상 최우선, 그 다음은 설정에 따라
  // 보카로 위키(발음·사람 번역) → LRCLIB 순서 또는 그 반대
  const wikiFirst = settings.lyricsSourcePriority === 'vocaro';
  const res = await sendToBackground<LyricsData | null>({
    type: 'FETCH_LYRICS',
    payload: { ...song, skipLrclib: wikiFirst },
  });
  if (seq !== searchSeq || videoId !== currentVideoId) return;
  if (res.error) {
    panel.showError('가사를 불러오지 못했어요');
    return;
  }

  let data = res.data ?? null;
  currentSourceUrl = null;
  if (!data) {
    const vocaro = await sendToBackground<VocaroResult | null>({
      type: 'VOCARO_LOOKUP',
      payload: { title: song.title },
    });
    if (seq !== searchSeq || videoId !== currentVideoId) return;
    if (vocaro.data && vocaro.data.lines.length > 0) {
      data = adoptVocaroResult(videoId, vocaro.data);
    }
  }
  // 위키 우선 모드에서 위키까지 미스면 후순위 LRCLIB 시도
  if (!data && wikiFirst) {
    const lr = await sendToBackground<LyricsData | null>({ type: 'FETCH_LRCLIB', payload: song });
    if (seq !== searchSeq || videoId !== currentVideoId) return;
    data = lr.data ?? null;
  }
  // 서버 싱크(위키 가사로 생성된 것)에 위키의 발음/사람 번역을 텍스트 매칭으로 병합
  if (data && data.source === 'everyric' && data.synced) {
    await enrichFromVocaro(videoId, data);
    if (seq !== searchSeq || videoId !== currentVideoId) return;
  }
  applyLyricsData(data);
}

/** 위키 조회 결과를 LyricsData로 변환하고 출처·재병합 캐시를 채운다 */
function adoptVocaroResult(videoId: string, vocaro: VocaroResult): LyricsData {
  const lines: LyricLine[] = vocaro.lines.map(l => ({
    time: null,
    endTime: null,
    text: l.text,
    translation: l.translation,
    pronunciation: l.pronunciation,
  }));
  currentSourceUrl = vocaro.pageUrl;
  // 이 곡의 위키 페이지를 기억 — 싱크 생성 뒤에도 발음/번역을 다시 입힐 수 있게
  lastVocaro = { videoId, lines: vocaro.lines };
  try {
    void chrome.storage.local.set({ [`vocaroRef:${videoId}`]: vocaro.slug });
  } catch { /* 저장 실패는 무시 — 세션 내 캐시로도 동작 */ }
  return { source: 'vocaro', synced: false, lines, plainText: lines.map(l => l.text).join('\n') };
}

/** 수동 검색: 소스별 후보 리스트를 모아 패널에 전달 */
async function handleCandidateSearch(query: { title: string; artist: string }): Promise<void> {
  const videoId = currentVideoId;
  if (!videoId) return;
  const res = await sendToBackground<SearchCandidate[]>({
    type: 'SEARCH_CANDIDATES',
    payload: { ...query, duration: currentSong?.duration ?? 0 },
  });
  if (videoId !== currentVideoId) return;
  ensureOverlay().showSearchResults(res.data ?? []);
}

/** 후보 선택: 해당 소스에서 가사를 받아 현재 가사를 교체한다 (잘못 가져온 가사 롤백 경로) */
async function handlePickCandidate(candidate: SearchCandidate): Promise<void> {
  const videoId = currentVideoId;
  if (!videoId) return;
  const seq = ++searchSeq; // 진행 중이던 자동 검색/생성 흐름은 폐기
  generatingJobs.delete(videoId); // 다른 가사를 고르면 이 영상의 기존 전사 추적은 버린다
  persistActiveJobs();
  updateGenChip();
  engine.stop();
  const panel = ensureOverlay();
  panel.showLoading('선택한 가사를 불러오는 중…');

  let data: LyricsData | null = null;
  currentSourceUrl = null;
  if (candidate.source === 'vocaro') {
    const page = await sendToBackground<VocaroResult | null>({
      type: 'VOCARO_PAGE',
      payload: { slug: candidate.slug },
    });
    if (seq !== searchSeq || videoId !== currentVideoId) return;
    if (page.data && page.data.lines.length > 0) data = adoptVocaroResult(videoId, page.data);
  } else {
    const res = await sendToBackground<LyricsData | null>({
      type: 'PICK_LRCLIB',
      payload: { id: candidate.id },
    });
    if (seq !== searchSeq || videoId !== currentVideoId) return;
    data = res.data ?? null;
  }

  if (!data) {
    panel.showError('선택한 가사를 불러오지 못했어요');
    return;
  }
  applyLyricsData(data);
}

function applyLyricsData(data: LyricsData | null): void {
  const panel = ensureOverlay();
  currentData = data;
  lastLineIndex = -1;
  engine.stop();
  // 영상별 저장 오프셋 복원 (서버에 저장된 값, 없으면 0) — UI 라벨도 함께
  videoOffset = data?.userOffset ?? 0;
  panel.setOffsetValue(videoOffset);
  // 곡 전체 정렬 신뢰도가 매우 낮으면 경고 바 (설정으로 끌 수 있음)
  panel.setQualityWarning(
    settings.lowConfWarning && data?.synced && data.source === 'everyric'
      && data.qualityScore != null && data.qualityScore < 0.001
      ? data.qualityScore
      : null,
  );
  const attribution = data?.attribution
    ?? (data?.source === 'vocaro' ? { name: '보카로 가사 위키', url: currentSourceUrl } : null);
  panel.setAttribution(attribution ?? null);
  // 다른 영상 싱크를 빌려온 상태면 출처 배지·검색 시트 해제 UI에 반영
  panel.setLinked(data?.source === 'everyric' ? data.linked ?? null : null);

  if (!data) {
    if (pip.isOpen()) pip.close();
    panel.showEmpty(currentSong);
    return;
  }
  if (data.synced) {
    if (pip.isOpen()) {
      pip.setTempo(data.tempo ?? null);
      pip.setKey(data.key ?? null);
      pip.setDebugMeta(data.debugMeta ?? null);
      pip.setLines(data.lines);
      karaokeAudio.setNotes(collectMelodyNotes(data.lines));
      karaokeAudio.setTempo(data.tempo ?? null);
      if (settings.pipKeepPanel) {
        panel.showSyncedLyrics(data.lines, data.source, data.plainText);
        panel.setPipEnabled(PipController.isSupported());
      } else {
        panel.showPipPlaceholder();
      }
      panel.setPipActive(true);
    } else {
      panel.showSyncedLyrics(data.lines, data.source, data.plainText);
      panel.setPipEnabled(PipController.isSupported());
    }
    void startEngine(data.lines);
  } else {
    if (pip.isOpen()) pip.close();
    panel.showPlainLyrics(data.lines, data.source, data.plainText);
  }
  if (settings.showTranslation) void loadTranslations();
  pushDebug(null);
}

function makeEngineHandlers(): SyncHandlers {
  return {
    onLineChange: index => {
      lastLineIndex = index;
      overlay?.highlightLine(index);
      pip.update(index);
    },
    onTick: time => {
      overlay?.updateTime(time);
      pip.tick(time, engine.getDuration(), engine.isPaused());
      const video = engine.getVideo();
      if (video) pip.updateVolume(video.volume, video.muted);
      if (settings.debugInfo && Date.now() - lastDebugPush >= 500) {
        lastDebugPush = Date.now();
        pushDebug(time);
      }
    },
  };
}

async function startEngine(lines: LyricLine[]): Promise<void> {
  const video = await waitForVideo();
  if (!video || !currentData?.synced) return;
  engine.start(video, lines, makeEngineHandlers());
  engine.setOffset(videoOffset);
}

async function waitForVideo(maxRetries = 10, delayMs = 500): Promise<HTMLVideoElement | null> {
  for (let i = 0; i < maxRetries; i++) {
    const video = getVideoElement();
    if (video) return video;
    await sleep(delayMs);
  }
  return null;
}

async function waitForSongInfo(seq: number, maxRetries = 6, delayMs = 700): Promise<SongInfo | null> {
  for (let i = 0; i < maxRetries; i++) {
    if (seq !== searchSeq) return null; // 새 검색이 시작됨 — 즉시 중단
    const info = detectSong();
    if (info?.title) return info;
    await sleep(delayMs);
  }
  return detectSong();
}

// ── 싱크 생성 ───────────────────────────────────────────────────

async function handleGenerate(lyricsText: string): Promise<void> {
  const videoId = currentVideoId;
  const seq = searchSeq;
  const text = lyricsText.trim();
  if (!videoId || !text) return;

  // 이미 전사 중이거나 요청 준비 중(LLM 번역·독음 대기) — 연타를 서버로 내보내지 않는다.
  // 같은 영상의 중복 잡은 임시 오디오 파일을 두고 경합해 다운로드 실패(WinError 32)까지 냈다.
  if (generatingJobs.has(videoId) || preparingGenerate.has(videoId)) return;
  preparingGenerate.add(videoId);
  updateGenChip(); // 버튼을 누르자마자 "준비 중" 칩으로 즉시 반응을 보여준다

  try {
    // 보카로 위키 가사로 생성할 때는 발음/사람 번역도 서버에 함께 저장한다
    // (서버 싱크에 병합돼 다른 프로필·사용자에게도 그대로 표시됨)
    let lineMeta: { text: string; pronunciation?: string; translation?: string }[] | undefined =
      currentData?.source === 'vocaro'
        ? currentData.lines
          .filter(l => l.pronunciation || l.translation)
          .map(l => ({ text: l.text, pronunciation: l.pronunciation, translation: l.translation }))
        : undefined;

    // 위키 출처는 싱크에 영구 저장돼 조회 시 푸터에 병기된다 (CC BY 표기)
    const attribution = currentData?.source === 'vocaro'
      ? { name: '보카로 가사 위키', url: currentSourceUrl }
      : undefined;

    // 위키 발음이 없으면(수동 붙여넣기·LRCLIB 등) LLM 번역·한글 독음을 먼저 받아
    // line_meta로 넘긴다 — 서버가 독음(ko) 정렬 경로를 타고 발음/번역도 싱크에 저장된다.
    // 실패해도 싱크 생성 자체는 계속한다 (원문 정렬 폴백).
    if (!lineMeta || lineMeta.length === 0) {
      const srcLines = text.split('\n').map(s => s.trim()).filter(Boolean);
      lineMeta = await fetchLlmLineMeta(videoId, srcLines);
    }

    const panel = ensureOverlay();
    const res = await sendToBackground<GenerateResponse>({
      type: 'GENERATE_SYNC',
      payload: {
        videoId,
        lyrics: text,
        lineMeta: lineMeta && lineMeta.length > 0 ? lineMeta : undefined,
        attribution,
      },
    });
    if (res.error || !res.data) {
      if (videoId === currentVideoId && seq === searchSeq) {
        panel.showError('싱크 생성 요청에 실패했어요. 서버 상태를 확인해 주세요.');
      }
      return;
    }
    if (res.data.status === 'completed') {
      if (videoId === currentVideoId) void searchLyrics();
      return;
    }
    // 패널을 점유하지 않는다 — 현재 화면(가사/검색)은 그대로 두고 작은 칩으로 진행률만 표시.
    // 다른 영상으로 이동해도 잡은 계속 추적되고, 완료 후 돌아오면 조회 시 자동 반영된다.
    generatingJobs.set(videoId, { jobId: res.data.job_id, progress: 0, title: currentSong?.title });
    persistActiveJobs();
    ensurePolling();
  } finally {
    preparingGenerate.delete(videoId);
    updateGenChip();
  }
}

/** 재생성: 현재 everyric 싱크의 가사·발음·출처 그대로 서버 캐시를 무시하고 다시 정렬 */
async function handleRegenerate(): Promise<void> {
  const videoId = currentVideoId;
  const data = currentData;
  if (!videoId || !data?.synced || data.source !== 'everyric') return;
  if (generatingJobs.has(videoId) || preparingGenerate.has(videoId)) return;
  preparingGenerate.add(videoId);
  updateGenChip();

  try {
    const lyrics = data.lines.map(l => l.text).join('\n').trim();
    if (!lyrics) return;
    let lineMeta: { text: string; pronunciation?: string; translation?: string }[] = data.lines
      .filter(l => l.pronunciation || l.translation)
      .map(l => ({ text: l.text, pronunciation: l.pronunciation, translation: l.translation }));

    // 발음이 기대되는 원문인데 발음이 하나도 없으면(번역만 저장된 낡은 싱크) LLM 독음을
    // 새로 받아 재생성이 독음 정렬 경로를 타게 한다 — 안 그러면 발음 없는 싱크가 재생산된다
    const texts = data.lines.map(l => l.text);
    if (!data.lines.some(l => l.pronunciation) && expectsPronunciation(texts)) {
      const fetched = await fetchLlmLineMeta(videoId, texts);
      if (fetched && fetched.length > 0) lineMeta = fetched;
    }

    const res = await sendToBackground<GenerateResponse>({
      type: 'REGENERATE_SYNC',
      payload: {
        videoId,
        lyrics,
        lineMeta: lineMeta.length > 0 ? lineMeta : undefined,
        attribution: data.attribution,
      },
    });
    if (res.error || !res.data) {
      if (videoId === currentVideoId) ensureOverlay().showError('재생성 요청에 실패했어요. 서버 상태를 확인해 주세요.');
      return;
    }
    generatingJobs.set(videoId, { jobId: res.data.job_id, progress: 0, title: currentSong?.title });
    persistActiveJobs();
    ensurePolling();
  } finally {
    preparingGenerate.delete(videoId);
    updateGenChip();
  }
}

/** 이 영상의 서버 싱크 전부 삭제(초기화) 후 처음부터 다시 검색 — 잘못 붙여넣은 가사 복구용 */
async function handleResetSync(): Promise<void> {
  const videoId = currentVideoId;
  if (!videoId) return;
  const res = await sendToBackground<{ removed_syncs: number }>({
    type: 'SYNC_RESET', payload: { videoId },
  });
  if (res.error) {
    ensureOverlay().showError('싱크 초기화에 실패했어요. 서버 상태를 확인해 주세요.');
    return;
  }
  // 세션 캐시(언어별 번역·발음)와 진행 중 잡 추적도 함께 비워 완전히 처음부터
  for (const key of [...translationCache.keys()]) {
    if (key.startsWith(`${videoId}:`)) translationCache.delete(key);
  }
  generatingJobs.delete(videoId);
  persistActiveJobs();
  updateGenChip();
  void searchLyrics();
}

async function pollJobs(): Promise<void> {
  if (generatingJobs.size === 0) {
    stopPolling();
    updateGenChip();
    return;
  }
  for (const [videoId, job] of [...generatingJobs]) {
    const res = await sendToBackground<JobStatusResponse>({ type: 'JOB_STATUS', payload: { jobId: job.jobId } });
    if (generatingJobs.get(videoId)?.jobId !== job.jobId) continue; // 그 사이 교체/취소됨
    const status = res.data;
    if (!status) continue; // 일시적 실패 — 다음 폴링에서 재시도

    if (status.status === 'completed') {
      generatingJobs.delete(videoId);
      persistActiveJobs();
      notifyJobDone(job.jobId, '전사 완료', `${job.title ?? videoId} — 가사 싱크가 준비됐어요`);
      if (videoId === currentVideoId) void searchLyrics();
    } else if (status.status === 'failed') {
      generatingJobs.delete(videoId);
      persistActiveJobs();
      notifyJobDone(
        job.jobId, '전사 실패',
        `${job.title ?? videoId} — ${status.error || '싱크 생성에 실패했어요'}`,
      );
      if (videoId === currentVideoId) {
        ensureOverlay().showError(status.error || '싱크 생성에 실패했어요');
      }
    } else {
      job.progress = status.progress ?? job.progress;
      job.stage = status.stage ?? undefined;
      job.stageProgress = status.stage_progress ?? undefined;
      job.queueLabel = status.queue_position != null && status.queue_position > 0
        ? `대기열 ${status.queue_position}번째`
        : (status.status === 'queued' || status.status === 'pending' ? '대기열' : undefined);
    }
  }
  updateGenChip();
}

/** 전사 잡 종료 OS 알림 — 다른 탭/창에 있어도 결과를 알 수 있다.
 *  잡 id를 알림 id로 써서 여러 탭이 같은 잡을 폴링해도 중복되지 않는다. */
function notifyJobDone(jobId: string, title: string, message: string): void {
  if (!settings.notifyOnComplete) return;
  void sendToBackground({ type: 'NOTIFY', payload: { id: `ey-job-${jobId}`, title, message } });
}

/** 진행 칩 갱신 — 현재 영상 잡의 진행률, 그 외 영상 잡은 건수로 요약 */
function updateGenChip(): void {
  if (!overlay) return;
  const cur = currentVideoId ? generatingJobs.get(currentVideoId) : undefined;
  const others = generatingJobs.size - (cur ? 1 : 0);
  let text: string | null = null;
  if (!cur && currentVideoId && preparingGenerate.has(currentVideoId)) {
    // 잡 등록 전 준비 단계 — 버튼이 무반응처럼 보이지 않게 즉시 표시
    text = '싱크 생성 준비 중 — AI 번역·독음 요청…';
  } else if (cur) {
    // 단계명이 오면 "보컬 분리 60% · 전체 68%"처럼 무슨 과정인지 함께 보여준다
    const state = cur.queueLabel
      ?? (cur.stage
        ? `${cur.stage} ${cur.stageProgress ?? 0}% · 전체 ${cur.progress}%`
        : `${cur.progress}%`);
    text = `전사 중 ${state}${others > 0 ? ` · 외 ${others}건` : ''}`;
  } else if (others > 0) {
    text = `다른 영상 전사 중 ${others}건`;
  }
  overlay.setGenerationChip(text);
}

function ensurePolling(): void {
  if (pollTimer === undefined) {
    pollTimer = window.setInterval(() => void pollJobs(), 2000);
  }
}

function stopPolling(): void {
  if (pollTimer !== undefined) {
    clearInterval(pollTimer);
    pollTimer = undefined;
  }
}

// ── PiP ────────────────────────────────────────────────────────

async function handlePipToggle(): Promise<void> {
  if (pip.isOpen()) {
    pip.close(); // pagehide → onClosed에서 패널 복원
    return;
  }
  if (!currentData?.synced) return;
  const videoId = currentVideoId;
  const panel = ensureOverlay();
  const opened = await pip.open(cssText, {
    showVideo: settings.pipShowVideo,
    initialVideoRatio: settings.pipVideoRatio,
    showPronunciation: settings.showPronunciation,
    pitchEnabled: settings.pitchGuide,
    pitchLaneHeight: settings.pitchLaneHeight,
    pitchWindowMeasures: settings.pitchWindowMeasures,
    pitchScrollMode: settings.pitchScrollMode,
    pitchFontScale: settings.pitchFontScale,
    pitchCountdown: settings.pitchCountdown,
    showConfidence: settings.debugInfo,
    onPitchHeightChange: px => {
      settings = { ...settings, pitchLaneHeight: px };
      void saveSettings({ pitchLaneHeight: px });
    },
    onSeek: time => engine.seekTo(time),
    onSeekRatio: ratio => {
      const video = engine.getVideo() ?? getVideoElement();
      if (video && Number.isFinite(video.duration) && video.duration > 0) {
        video.currentTime = ratio * video.duration;
      }
    },
    onPlayPause: () => {
      const video = engine.getVideo() ?? getVideoElement();
      if (!video) return;
      if (video.paused) void video.play().catch(() => { /* 사용자 제스처 필요 시 무시 */ });
      else video.pause();
      engine.resync(); // 재생 상태 아이콘 즉시 갱신
    },
    onVolumeChange: volume => {
      const video = engine.getVideo() ?? getVideoElement();
      if (!video) return;
      video.volume = Math.min(1, Math.max(0, volume));
      if (volume > 0 && video.muted) video.muted = false;
    },
    onMuteToggle: () => {
      const video = engine.getVideo() ?? getVideoElement();
      if (video) video.muted = !video.muted;
    },
    onVideoRatioChange: ratio => {
      settings = { ...settings, pipVideoRatio: ratio };
      void saveSettings({ pipVideoRatio: ratio });
    },
    melodyOn: settings.melodyPlayback,
    onMelodyToggle: () => void handleSettingsChange({ melodyPlayback: !settings.melodyPlayback }),
    metronomeOn: settings.metronome,
    onMetronomeToggle: () => void handleSettingsChange({ metronome: !settings.metronome }),
    metronomeRate: settings.metronomeRate,
    onMetronomeRateChange: rate => void handleSettingsChange({ metronomeRate: rate }),
    metronomeBeat: settings.metronomeBeat,
    onMetronomeBeatChange: beat => void handleSettingsChange({ metronomeBeat: beat }),
    micOctave: settings.micOctave,
    onPitchWindowChange: measures => void handleSettingsChange({ pitchWindowMeasures: measures }),
    onPitchScrollModeChange: mode => void handleSettingsChange({ pitchScrollMode: mode }),
    onKaraokeToggle: on => void handleSettingsChange({ pitchGuide: on }),
    onVideoToggle: on => void handleSettingsChange({ pipShowVideo: on }),
    getMicSamples: () => micPitch.samples(),
    onClosed: () => {
      karaokeAudio.setActive(false);
      micPitch.stop();
      overlay?.setPipActive(false);
      // 패널이 placeholder 상태일 때만 복원 (동시 표시 모드면 이미 가사가 떠 있음)
      if (overlay?.isShowingPipPlaceholder()) restoreOverlayState();
    },
  });
  if (!opened) return;
  // requestWindow 대기 중 내비게이션이 일어났으면 stale한 PiP는 닫는다
  if (videoId !== currentVideoId || !currentData?.synced) {
    pip.close();
    return;
  }
  pip.setSong(currentSong?.title ?? '', currentSong?.artist ?? '');
  pip.setTempo(currentData.tempo ?? null);
  pip.setKey(currentData.key ?? null);
  pip.setDebugMeta(currentData.debugMeta ?? null);
  pip.setLines(currentData.lines);
  karaokeAudio.setNotes(collectMelodyNotes(currentData.lines));
  karaokeAudio.setTempo(currentData.tempo ?? null);
  karaokeAudio.setActive(true);
  applyAudioSettings();
  if (settings.pipShowVideo) {
    const video = engine.getVideo() ?? getVideoElement();
    if (video) pip.attachVideo(video);
  }
  engine.resync(); // PiP에 현재 라인을 즉시 반영
  panel.setPipActive(true);
  if (!settings.pipKeepPanel) panel.showPipPlaceholder();
}

function restoreOverlayState(): void {
  if (!overlay || currentVideoId === null) return;
  applyLyricsData(currentData);
  updateGenChip(); // 전사 중이면 칩으로 표시 (패널 점유 없음)
}

// ── 서버 상태/유틸 ─────────────────────────────────────────────

async function refreshServerStatus(): Promise<void> {
  const res = await sendToBackground<{ ok: boolean }>({ type: 'SERVER_HEALTH' });
  overlay?.setServerAvailable(res.data?.ok ?? false);
}

async function sendToBackground<T>(message: BgRequest): Promise<MessageResponse<T>> {
  try {
    return await chrome.runtime.sendMessage(message) as MessageResponse<T>;
  } catch (error) {
    return { error: error instanceof Error ? error.message : String(error) };
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => void init());
} else {
  void init();
}
