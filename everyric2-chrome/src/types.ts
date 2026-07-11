export interface SongInfo {
  title: string;
  artist: string | null;
  videoId: string;
  duration: number;
}

/** 가라오케 음정 바용 노트 — 서버(FCPE)가 음절 구간을 반음 양자화한 결과 */
export interface NoteSegment {
  midi: number;
  start: number;
  end: number;
  confidence?: number;
}

export interface WordSegment {
  word: string;
  start: number;
  end: number;
  notes?: NoteSegment[];
  /** CTC 정렬 신뢰도 (0~1) — 디버그 모드에서 글자 색으로 표시 */
  confidence?: number;
}

export interface LyricLine {
  time: number | null;
  endTime: number | null;
  text: string;
  words?: WordSegment[];
  /** 단어 분해가 없는 라인의 라인 단위 노트 */
  notes?: NoteSegment[];
  translation?: string;
  /** 원문 가사의 한국어 발음 표기 (보카로 가사 위키 등 사람이 단 것) */
  pronunciation?: string;
  /** 발음 음절별 타이밍 (서버가 모라 분해+DP로 산출) — 없으면 시간 비례 그라데이션 폴백 */
  pronSegments?: PronSegment[];
  /** 라인 단위 CTC 정렬 신뢰도 (0~1) — 곡 전체 통계·디버그 표시용 */
  confidence?: number;
  /** 서버 정렬 진단 (디버그 스트립·레인 디버그 오버레이용) */
  debug?: LineDebug;
}

/** 라인 정렬 진단 — 세이프가드가 고친 라인은 보정 전 원본 타이밍과 규칙 라벨을 담는다 */
export interface LineDebug {
  activeRatio?: number;
  clamped?: boolean;
  /** 세이프가드 적용 전 원본 [start, end] (raw CTC) — 유의미하게 바뀐 라인만 */
  orig?: [number, number];
  /** 적용된 보정 규칙: stretch(8s+클램프)/repeat(반복행)/pull(간주 후 당김)/tail(끝음 연장)/snap(무음 온셋 스냅) */
  fixes?: string[];
}

/** 발음 표기 한 음절의 타임스탬프 */
export interface PronSegment {
  text: string;
  start: number;
  end: number;
  /** DP 매칭 신뢰 가능 여부 — false면 근사 배치 */
  resolved?: boolean;
}

export type LyricsSource = 'everyric' | 'lrclib' | 'vocaro';

export interface LyricsData {
  source: LyricsSource;
  synced: boolean;
  lines: LyricLine[];
  plainText: string;
  /** 사람이 단 번역(위키 등)이 병합돼 있음 — 기계번역으로 덮어쓰지 않는다 */
  humanTranslated?: boolean;
  /** 곡 단위 정렬 진단 (everyric 소스만) */
  debugMeta?: SyncDebugMeta;
  /** 가사 원출처 (서버 저장분 또는 vocaro 직접 조회) — 푸터에 병기 */
  attribution?: SourceAttribution;
  /** 곡 템포 (everyric 소스만) — 레인 마디 창/비트 격자 */
  tempo?: SongTempo;
  /** 곡 전체 평균 정렬 신뢰도 (기하평균 확률 평균) — 디버그 표시용 */
  qualityScore?: number;
  /** 다른 영상의 싱크에 링크된 상태 (해제 UI 표시용) */
  linked?: { sourceVideoId: string; offsetSec: number };
}

export interface LRCLibTrack {
  id: number;
  trackName: string;
  artistName: string;
  albumName: string;
  duration: number;
  instrumental: boolean;
  plainLyrics: string | null;
  syncedLyrics: string | null;
}

export interface EveryricSegment {
  text: string;
  start: number;
  end: number;
  /** 라인 단위 CTC 정렬 신뢰도 (0~1) */
  confidence?: number;
  words?: WordSegment[];
  notes?: NoteSegment[];
  /** 서버에 저장된 발음 표기/사람 번역 (생성 시 line_meta로 전달된 것) */
  pronunciation?: string;
  translation?: string;
  /** 발음 음절별 타이밍 (서버 계산) */
  pron_segments?: PronSegment[];
  /** 라인 진단: 발성 비율/클램프 여부/보정 전 원본 타이밍/적용 규칙 */
  debug?: { active_ratio?: number; clamped?: boolean; orig?: [number, number]; fixes?: string[] };
}

/** 가사 출처 표기 (예: 보카로 가사 위키 CC BY) */
export interface SourceAttribution {
  name: string;
  url?: string | null;
}

/** 서버(librosa)가 추정한 곡 템포 — 레인의 마디 단위 고정 창과 비트/마디 격자용 */
export interface SongTempo {
  bpm: number;
  /** 첫 비트 시각(초) — 격자를 실제 박에 맞춰 정렬 */
  beat_offset?: number | null;
}

/** 곡 단위 정렬 진단 메타 (서버 debug 필드) */
export interface SyncDebugMeta {
  /** star 토큰이 흡수한 가사 밖 가창 구간들 */
  star_spans?: [number, number][] | null;
  /** VAD가 발성으로 판정한 구간들 */
  vad_regions?: [number, number][] | null;
  /** 음정 인식 모델(RMVPE/FCPE) RAW f0 곡선 — 균일 샘플, null = 무성 프레임 */
  f0_curve?: F0Curve | null;
  /** 정렬에 쓴 텍스트: "pronunciation"(독음) | "original"(원문) */
  alignment_text?: string | null;
}

/** RAW f0 곡선 (다운샘플) — midi[i]의 시각 = t0 + i*dt */
export interface F0Curve {
  t0: number;
  dt: number;
  midi: (number | null)[];
}

/** 싱크 생성 시 서버에 함께 저장할 라인별 발음/번역 */
export interface LineMeta {
  text: string;
  pronunciation?: string;
  translation?: string;
}

export interface EveryricSyncResponse {
  found: boolean;
  sync_id?: string;
  timestamps?: EveryricSegment[];
  lyrics_source?: string;
  quality_score?: number;
  language?: string;
  created_at?: string;
  error?: string;
  debug?: SyncDebugMeta | null;
  attribution?: SourceAttribution | null;
  tempo?: SongTempo | null;
  /** 다른 영상의 싱크를 빌려온 경우 (inst·커버 링크) — 타이밍은 이미 오프셋 적용됨 */
  linked?: { source_video_id: string; offset_sec: number } | null;
}

/** GET /api/sync/list 항목 — 링크 후보 선택용 */
export interface SyncListItem {
  video_id: string;
  first_line: string;
  line_count: number;
  attribution_name?: string | null;
  created_at?: string | null;
  alignment_text?: string | null;
}

export interface GenerateResponse {
  job_id: string;
  status: string;
  estimated_time?: number;
}

export interface JobStatusResponse {
  job_id: string;
  status: 'pending' | 'queued' | 'processing' | 'completed' | 'failed' | string;
  progress: number;
  timestamps?: EveryricSegment[] | null;
  error?: string | null;
  /** 서버가 큐잉을 지원하면 대기 순번(1 = 다음 차례)을 내려줄 수 있다 */
  queue_position?: number | null;
  queue_size?: number | null;
}

export interface Settings {
  autoSearch: boolean;
  fontSize: 'small' | 'medium' | 'large';
  theme: 'auto' | 'dark' | 'light';
  serverUrl: string;
  offsetSec: number;
  showTranslation: boolean;
  translationLanguage: string;
  /** 원문 밑에 한국어 발음 표기(있을 때만) 표시 — 패널·PiP 공통 */
  showPronunciation: boolean;
  /** 서버 싱크가 없을 때 어느 가사 소스를 먼저 찾을지 — 보카로 위키는 발음·사람 번역 제공 */
  lyricsSourcePriority: 'vocaro' | 'lrclib';
  pipKeepPanel: boolean;
  pipShowVideo: boolean;
  /** 빈 문자열이면 헤더 생략 */
  apiKey: string;
  /** PiP에서 영상 영역이 차지하는 세로 비율 (0 = 자동 16:9) */
  pipVideoRatio: number;
  /** 가라오케 레인 높이(px) — 레인 위 디바이더 드래그로 조절 */
  pitchLaneHeight: number;
  /** 가라오케 레인 표시 구간(마디 수) — 서버 BPM 기준, 템포 없으면 120BPM 가정 폴백 */
  pitchWindowMeasures: number;
  /** 레인 진행 방식: page = 화면 고정 + 플레이헤드 이동, scroll = 플레이헤드 고정 + 횡스크롤 */
  pitchScrollMode: 'page' | 'scroll';
  /** 레인 글자 크기 배율 (계이름·발음·가사·번역 공통) */
  pitchFontScale: number;
  /** 긴 묵음 뒤 가사 시작 전 4·3·2·1 카운트다운 표시 */
  pitchCountdown: boolean;
  /** PiP 하단 가라오케 음정 바 표시 (노트 데이터가 있는 곡에서만) */
  pitchGuide: boolean;
  /** 가라오케 창에서 노트를 신디사이즈로 재생 */
  melodyPlayback: boolean;
  /** 멜로디 볼륨 (0..1) */
  melodyVolume: number;
  /** 가라오케 창 메트로놈 — 서버 추정 BPM 기준, 4/4 가정 */
  metronome: boolean;
  /** 메트로놈 볼륨 (0..1) */
  metronomeVolume: number;
  /** 멜로디·메트로놈 출력 기기 id (AudioContext.setSinkId) — '' = 기본 출력 */
  audioOutputId: string;
  /** 마이크로 부른 음정을 가라오케 레인에 표시 */
  micPitch: boolean;
  /** 마이크 입력 기기 id — '' = 기본 마이크 */
  micDeviceId: string;
  /** 패널 하단에 내부 상태(비디오 바인딩, 싱크 소스 등) 표시 */
  debugInfo: boolean;
}

/** 디버그 스트립에 표시할 내부 상태 스냅샷 */
export interface DebugInfo {
  videoId: string | null;
  source: string;
  synced: boolean;
  /** 비디오 currentTime — 비디오가 없으면 null */
  time: number | null;
  offsetSec: number;
  lineIndex: number;
  lineCount: number;
  /** 엔진이 붙잡은 video가 지금 DOM에서 재생 중인 video와 같은가 */
  videoBound: boolean;
  videoInfo: string;
  engineRunning: boolean;
  pipOpen: boolean;
  jobStatus: string | null;
  /** 현재 시각의 구간 판정: 가창 / 간주(VAD무성) / 추임새(star흡수) */
  zone: string | null;
  /** 현재 라인 진단 (발성 비율, 클램프 여부) */
  lineDebug: string | null;
  /** 곡 전체 평균 정렬 신뢰도 */
  quality: number | null;
  /** 곡 전체 median 정렬 신뢰도 (라인 confidence 기준) */
  qualityMed: number | null;
  /** 저신뢰(<1e-4) 라인 비율 (0~1) */
  lowConfRatio: number | null;
  /** 라인 신뢰도 등급 분포 (좋음/보통/낮음, 0~1) — 사람이 읽는 요약 */
  confGrades: { ok: number; mid: number; low: number } | null;
  /** 정렬에 쓴 텍스트 (독음/원문) — 서버 debug 메타 */
  alignmentText: string | null;
}

export interface TranslatedLine {
  original: string;
  translation: string;
  pronunciation?: string | null;
}

export interface TranslateResult {
  lines: TranslatedLine[];
  source_lang?: string;
  target_lang?: string;
  engine?: string;
}

export interface PanelGeometry {
  x: number;
  y: number;
  width: number;
  height: number;
  collapsed: boolean;
}

/** 수동 검색에서 사용자가 직접 고를 수 있는 후보 (소스별) */
export type SearchCandidate =
  | { source: 'lrclib'; id: number; title: string; artist: string; duration: number; synced: boolean }
  | { source: 'vocaro'; slug: string; title: string; url: string };

export type BgRequest =
  | { type: 'FETCH_LYRICS'; payload: SongInfo & { skipLrclib?: boolean } }
  | { type: 'FETCH_LRCLIB'; payload: SongInfo }
  | { type: 'SEARCH_CANDIDATES'; payload: { title: string; artist: string; duration: number } }
  | { type: 'PICK_LRCLIB'; payload: { id: number } }
  | { type: 'GENERATE_SYNC'; payload: { videoId: string; lyrics: string; language?: string; lineMeta?: LineMeta[]; attribution?: SourceAttribution } }
  | { type: 'REGENERATE_SYNC'; payload: { videoId: string; lyrics: string; lineMeta?: LineMeta[]; attribution?: SourceAttribution } }
  | { type: 'SYNC_LINK'; payload: { videoId: string; sourceVideoId: string; offsetSec: number } }
  | { type: 'SYNC_UNLINK'; payload: { videoId: string } }
  | { type: 'SYNC_LIST' }
  | { type: 'JOB_STATUS'; payload: { jobId: string } }
  | { type: 'TRANSLATE'; payload: { text: string; targetLang: string } }
  | { type: 'SERVER_HEALTH' }
  | { type: 'VOCARO_LOOKUP'; payload: { title: string } }
  | { type: 'VOCARO_PAGE'; payload: { slug: string } };

export type ContentMessage =
  | { type: 'TOGGLE_OVERLAY' }
  | { type: 'SYNC_GENERATED'; payload: { videoId: string } };

export interface MessageResponse<T = unknown> {
  data?: T;
  error?: string;
}
