import { detectSong, getCurrentVideoId, getVideoElement } from './lib/song-detector';
import { SyncEngine, type SyncHandlers } from './lib/sync-engine';
import { getGeometry, getSettings, saveGeometry, saveSettings } from './lib/settings';
import { LyricsOverlay } from './ui/overlay';
import { PipController } from './ui/pip';
import type {
  BgRequest,
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
  TranslateResult,
} from './types';
import type { VocaroLine, VocaroResult } from './lib/vocaro';

let settings: Settings;
let cssText = '';
let initialGeometry: PanelGeometry | null = null;
let overlay: LyricsOverlay | null = null;
const pip = new PipController();
const engine = new SyncEngine();

let currentVideoId: string | null = null;
let currentSong: SongInfo | null = null;
let currentData: LyricsData | null = null;
let currentSourceUrl: string | null = null; // 보카로 위키 출처 페이지 (CC BY 출처 표기용)
let lastVocaro: { videoId: string; lines: VocaroLine[] } | null = null; // 싱크 생성 후 발음/번역 재병합용
let generatingJob: { jobId: string; videoId: string; seq: number; progress: number } | null = null;
let pollTimer: number | undefined;
let searchSeq = 0;
let lastLineIndex = -1;
let lastDebugPush = 0;
const translationCache = new Map<string, string[]>(); // `${videoId}:${lang}` → 라인별 번역

async function init(): Promise<void> {
  settings = await getSettings();
  [cssText, initialGeometry] = await Promise.all([loadCss(), getGeometry()]);
  chrome.runtime.onMessage.addListener(handleRuntimeMessage);
  observeNavigation();
  checkCurrentPage();
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
    engine.setOffset(settings.offsetSec);
    // 미러 스트림도 새 video 기준으로 갱신
    if (pip.isOpen() && settings.pipShowVideo) pip.attachVideo(video);
  }
}

/** 자동 검색이 꺼져 있으면 사용자가 패널을 열어둔 경우에만 따라간다 */
function shouldFollow(): boolean {
  return settings.autoSearch || (overlay?.isVisible() ?? false);
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
  generatingJob = null;
  stopPolling();
  engine.stop();
  pip.close();
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
      engine.setOffset(offsetSec);
      settings = { ...settings, offsetSec };
      void saveSettings({ offsetSec });
    },
    onSettingsChange: patch => void handleSettingsChange(patch),
    onPipToggle: () => void handlePipToggle(),
    onGeometryChange: geometry => void saveGeometry(geometry),
    onCandidateSearch: query => void handleCandidateSearch(query),
    onPickCandidate: candidate => void handlePickCandidate(candidate),
  }, initialGeometry);
  return overlay;
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

  // 레인 표시 구간/카운트다운 즉시 반영
  if (patch.pitchWindowMeasures !== undefined) {
    pip.setPitchWindow(patch.pitchWindowMeasures);
  }
  if (patch.pitchCountdown !== undefined) {
    pip.setPitchCountdown(patch.pitchCountdown);
  }

  // 가라오케 음정 바 토글 즉시 반영
  if (patch.pitchGuide !== undefined) {
    pip.setPitchEnabled(patch.pitchGuide);
  }

  if (patch.debugInfo === true) pushDebug(null);
}

/** 현재 시각이 어떤 구간으로 판정됐는지 (star 흡수/가창/간주) — everyric 소스만 */
function debugZoneAt(time: number | null): string | null {
  const meta = currentData?.debugMeta;
  if (!meta || time === null) return null;
  const t = time - settings.offsetSec;
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
    offsetSec: settings.offsetSec,
    lineIndex: lastLineIndex,
    lineCount: currentData?.lines.length ?? 0,
    videoBound: bound !== null && (dom === null || bound === dom),
    videoInfo: video ? `rs${video.readyState},${video.paused ? 'pause' : 'play'}` : 'none',
    engineRunning: engine.isRunning(),
    pipOpen: pip.isOpen(),
    jobStatus: generatingJob ? `job=${generatingJob.jobId.slice(0, 8)}(${generatingJob.progress}%)` : null,
  });
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

  const lang = settings.translationLanguage;
  const cached = translationCache.get(`${videoId}:${lang}`);
  if (cached) {
    applyTranslations(data, cached);
    return;
  }

  overlay?.setTranslationStatus('번역 중…');
  const res = await sendToBackground<TranslateResult>({
    type: 'TRANSLATE',
    payload: { text: data.lines.map(l => l.text).join('\n'), targetLang: lang },
  });
  if (currentData !== data || currentVideoId !== videoId) return; // 곡이 바뀜
  if (!settings.showTranslation || settings.translationLanguage !== lang) return;

  const translations = res.data?.lines?.map(l => l.translation);
  if (!translations || translations.length === 0) {
    overlay?.setTranslationStatus('번역 실패 — 서버 확인');
    return;
  }
  translationCache.set(`${videoId}:${lang}`, translations);
  applyTranslations(data, translations);
}

function applyTranslations(data: LyricsData, translations: string[]): void {
  data.lines.forEach((line, i) => {
    const t = translations[i]?.trim();
    // '[NO API KEY]'는 구버전 서버의 키 미설정 플레이스홀더 — 번역으로 표시하지 않는다
    if (t && t !== line.text && !t.startsWith('[NO API KEY]')) line.translation = t;
  });
  overlay?.setTranslationStatus(null);
  overlay?.refreshTranslations();
  pip.refresh();
}

async function searchLyrics(queryOverride?: { title: string; artist: string }): Promise<void> {
  const videoId = currentVideoId;
  if (!videoId) return;
  const seq = ++searchSeq;
  const panel = ensureOverlay();
  panel.setVisible(true);
  panel.showLoading();
  stopPolling();
  generatingJob = null;
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
  stopPolling();
  generatingJob = null;
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
  const attribution = data?.attribution
    ?? (data?.source === 'vocaro' ? { name: '보카로 가사 위키', url: currentSourceUrl } : null);
  panel.setAttribution(attribution ?? null);

  if (!data) {
    if (pip.isOpen()) pip.close();
    panel.showEmpty(currentSong);
    return;
  }
  if (data.synced) {
    if (pip.isOpen()) {
      pip.setTempo(data.tempo ?? null);
      pip.setLines(data.lines);
      if (settings.pipKeepPanel) {
        panel.showSyncedLyrics(data.lines, data.source);
        panel.setPipEnabled(PipController.isSupported());
      } else {
        panel.showPipPlaceholder();
      }
      panel.setPipActive(true);
    } else {
      panel.showSyncedLyrics(data.lines, data.source);
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
  engine.setOffset(settings.offsetSec);
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

  // 보카로 위키 가사로 생성할 때는 발음/사람 번역도 서버에 함께 저장한다
  // (서버 싱크에 병합돼 다른 프로필·사용자에게도 그대로 표시됨)
  const lineMeta = currentData?.source === 'vocaro'
    ? currentData.lines
      .filter(l => l.pronunciation || l.translation)
      .map(l => ({ text: l.text, pronunciation: l.pronunciation, translation: l.translation }))
    : undefined;

  // 위키 출처는 싱크에 영구 저장돼 조회 시 푸터에 병기된다 (CC BY 표기)
  const attribution = currentData?.source === 'vocaro'
    ? { name: '보카로 가사 위키', url: currentSourceUrl }
    : undefined;

  const panel = ensureOverlay();
  panel.showGenerating(0);
  const res = await sendToBackground<GenerateResponse>({
    type: 'GENERATE_SYNC',
    payload: {
      videoId,
      lyrics: text,
      lineMeta: lineMeta && lineMeta.length > 0 ? lineMeta : undefined,
      attribution,
    },
  });
  // 요청 중 내비게이션/재검색이 일어났으면 이 생성 흐름은 폐기
  if (videoId !== currentVideoId || seq !== searchSeq) return;
  if (res.error || !res.data) {
    panel.showError('싱크 생성 요청에 실패했어요. 서버 상태를 확인해 주세요.');
    return;
  }
  if (res.data.status === 'completed') {
    void searchLyrics();
    return;
  }
  generatingJob = { jobId: res.data.job_id, videoId, seq, progress: 0 };
  stopPolling();
  pollTimer = window.setInterval(() => void pollJob(), 2000);
}

async function pollJob(): Promise<void> {
  const job = generatingJob;
  if (!job) {
    stopPolling();
    return;
  }
  if (job.videoId !== currentVideoId || job.seq !== searchSeq) {
    stopPolling();
    generatingJob = null;
    return;
  }
  const res = await sendToBackground<JobStatusResponse>({ type: 'JOB_STATUS', payload: { jobId: job.jobId } });
  if (generatingJob?.jobId !== job.jobId) return;
  const status = res.data;
  if (!status) return; // 일시적 실패 — 다음 폴링에서 재시도

  if (status.status === 'completed') {
    stopPolling();
    generatingJob = null;
    void searchLyrics();
  } else if (status.status === 'failed') {
    stopPolling();
    generatingJob = null;
    ensureOverlay().showError(status.error || '싱크 생성에 실패했어요');
  } else {
    job.progress = status.progress ?? job.progress;
    let label: string | undefined;
    if (status.queue_position != null && status.queue_position > 0) {
      label = status.queue_size != null
        ? `대기열 ${status.queue_position}번째 (총 ${status.queue_size}개) — 곧 시작해요`
        : `대기열 ${status.queue_position}번째 — 곧 시작해요`;
    } else if (status.status === 'queued' || status.status === 'pending') {
      label = '대기열에 등록됐어요 — 곧 시작해요';
    }
    ensureOverlay().showGenerating(job.progress, label);
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
    onClosed: () => {
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
  pip.setLines(currentData.lines);
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
  if (generatingJob) {
    overlay.showGenerating(generatingJob.progress);
    return;
  }
  applyLyricsData(currentData);
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
