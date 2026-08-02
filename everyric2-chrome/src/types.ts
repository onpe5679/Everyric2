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
  /** 위키(vocaro·miraheze) 채택 시 원본 번역 — 언어가 내 translationLanguage와 다르면
   *  표시 격리를 위해 translation에는 안 싣지만(adoptVocaroResult·adoptSourceResult가
   *  gate), 이 필드에는 언어 무관하게 항상 남는다. AI 생성(handleGenerate) line_meta는
   *  이 필드를 읽는다 — 화면엔 안 보여도 그 언어 레이어로 서버에 저장돼 그 언어
   *  사용자에게는 이득이 되게 하기 위함(사용자 요청: "내 화면만 격리"). */
  wikiTranslation?: string;
  /** 원문 가사의 한국어 발음 표기 (보카로 가사 위키 등 사람이 단 것) */
  pronunciation?: string;
  /** 발음 음절별 타이밍 (서버가 모라 분해+DP로 산출) — 없으면 시간 비례 그라데이션 폴백 */
  pronSegments?: PronSegment[];
  /** 표기별 발음 문자열 — {"hangul": "...", "romaji": "...", "kana": "..."}.
   *  값이 없는 표기는 레거시 pronunciation(한글)으로 폴백한다 — 항상 lib/lang.ts의
   *  resolvedPronunciation을 거쳐 읽는다 (직접 인덱싱 금지). */
  pron?: Record<string, string>;
  /** 표기별 발음 음절 타이밍 — {"hangul": PronSegment[], ...}. 폴백 규칙은 pron과 동일하며
   *  lib/lang.ts의 resolvedPronSegments를 거쳐 읽는다. */
  pronSegsByScript?: Record<string, PronSegment[]>;
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
  /** CTC가 이 라인 구간에서 실제로 전사한 문자열(독음 정렬 곡은 한글 독음) */
  heard?: string;
  /** heard의 글자별 시각(초) — [글자, 시각] 쌍, 시간 오름차순 */
  heard_spans?: [string, number][];
  /** 심판이 기본 발음 후보 대신 다른 후보를 골랐을 때의 판정 근거 */
  referee?: {
    default?: string;
    chosen?: string;
    margin?: number;
    gain?: number;
    frames?: number;
    scores?: [string, number][];
  };
}

/** 발음 표기 한 음절의 타임스탬프 */
export interface PronSegment {
  text: string;
  start: number;
  end: number;
  /** DP 매칭 신뢰 가능 여부 — false면 근사 배치 */
  resolved?: boolean;
  /** 음절 CTC 정렬 신뢰도 (0~1) — 서버가 음절별로 실어 보낸다. 디버그 레인 색에 쓴다 */
  confidence?: number;
}

export type LyricsSource = 'everyric' | 'lrclib' | 'vocaro' | 'caption';

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
  /** 자동 생성(ASR) 자막인가 — source가 'caption'일 때만 의미가 있다.
   *  노래를 ASR로 받아 적으면 원문과 딴 텍스트가 나오므로(실측) 화면 표시까지만
   *  허용하고 싱크 생성의 원문으로는 승격하지 않는다 (content.handleGenerate). */
  captionAuto?: boolean;
  /** 곡 템포 (everyric 소스만) — 레인 마디 창/비트 격자 */
  tempo?: SongTempo;
  /** 곡 키 (everyric 소스만) — 레인 좌상단 표시 */
  key?: SongKey;
  /** 곡 전체 평균 정렬 신뢰도 (기하평균 확률 평균) — 디버그 표시용 */
  qualityScore?: number;
  /** 서버 싱크 생성 시각 (서버가 준 원본 문자열, 타임존 표기 없는 UTC) — 디버그 표시용 */
  createdAt?: string;
  /** 다른 영상의 싱크에 링크된 상태 (해제 UI 표시용) — rate는 원곡 대비 배속.
   *  verified는 반주 크로스 코릴레이션 검증을 통과한 자동 링크임을 뜻한다 (수동 링크는 false) */
  linked?: { sourceVideoId: string; offsetSec: number; rate?: number; verified?: boolean };
  /** 이 영상에 서버 저장된 사용자 싱크 오프셋(초) — 로드 시 적용 */
  userOffset?: number;
  /** 지금 실린 translation이 어느 언어인지 — 서버 EveryricSyncResponse.translation_lang을
   *  그대로 옮긴 값이거나(구서버는 undefined), content.applyTranslations가 새 번역을 적용한
   *  뒤 세션 내에서 직접 채운 값. undefined면 "모름"(구서버 또는 아직 안 채워짐) — 그 경우
   *  기존 규칙(모든 줄에 translation이 있으면 충분)으로 폴백한다. */
  translationLang?: string | null;
  /** 제목바 언어 칩의 "보유" 판정 기준 — everyric 서버 sync면 EveryricSyncResponse.
   *  available_langs로 시작하지만, 소스 무관하게 content.applyTranslations가 새 언어
   *  번역을 성공 적용할 때마다 그 언어를 이 배열에 직접 추가한다(서버 재조회 없이 즉시
   *  "보유"로 반영). undefined면 "서버 신호도 세션 내 성공도 없음" — 칩은 그래도 뜬다
   *  (곡 자신의 언어만이라도 폴백 표시, content.availableLangsForChip 참고). */
  availableLangs?: string[];
  /** EveryricSyncResponse.translations_by_lang을 lib/lyrics-parser.segmentsToLines가
   *  lines와 같은 순서로 재정렬한 값(V2 확장). content가 첫 로딩에서 이 필드를 훑어
   *  세션 언어별 캐시(translationCache)를 선채움한다 — 그러면 서버가 이미 갖고 있는
   *  언어로의 전환은 **첫 클릭부터** 네트워크 0 즉시 스왑이다(기존 V2는 그 언어를
   *  로컬에서 한 번 직접 받아온 뒤에야 캐시가 생겨 두 번째 전환부터만 그랬다).
   *  값이 없는 인덱스(그 세그에 번역 없음)는 undefined. */
  translationsByLang?: Record<string, (string | undefined)[]>;
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
  /** 표기별 발음 문자열 — {"hangul": ..., "romaji": ..., "kana": ...} (서버 wire 포맷) */
  pron?: Record<string, string>;
  /** 표기별 발음 음절 타이밍 (서버 wire 포맷) */
  pron_segs?: Record<string, PronSegment[]>;
  /** 라인 진단: 발성 비율/클램프 여부/보정 전 원본 타이밍/적용 규칙 */
  debug?: {
    active_ratio?: number;
    clamped?: boolean;
    orig?: [number, number];
    fixes?: string[];
    heard?: string;
    heard_spans?: [string, number][];
    referee?: {
      default?: string;
      chosen?: string;
      margin?: number;
      gain?: number;
      frames?: number;
      scores?: [string, number][];
    };
  };
}

/** 가사 출처 표기 (예: 보카로 가사 위키 CC BY) */
export interface SourceAttribution {
  name: string;
  url?: string | null;
  /** "CC BY-SA 4.0" 등 — 라이선스별 표기 문구 분기·서버 왕복(sync.py Attribution)에 쓴다 */
  license?: string;
  /** 'vocaro' | 'miraheze' 등 — data.source 문자열 하드코딩 판정을 이 필드로 대체해 간다
   *  (구버전 attribution엔 없다 — 그 경우는 폴백 판정을 유지해야 한다) */
  sourceId?: string;
}

/** 서버(librosa)가 추정한 곡 템포 — 레인의 마디 단위 고정 창과 비트/마디 격자용 */
export interface SongTempo {
  bpm: number;
  /** 첫 비트 시각(초) — 격자를 실제 박에 맞춰 정렬 */
  beat_offset?: number | null;
}

/** 서버(멜로디 분석)가 추정한 곡 키 — 레인 표시 + 노트 반음 보정에 사용됨 */
export interface SongKey {
  /** 으뜸음 pitch class (0=C … 11=B) */
  tonic: number;
  mode: 'major' | 'minor';
  /** 표시용 이름 (예: "G#m", "A") */
  name: string;
  /** K-S 프로파일 상관 (0~1) — 낮으면 서버가 보정을 건너뛴다 */
  confidence?: number | null;
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
  /** 자막 스캐폴드 판정 — 붕괴 곡의 줄 타이밍을 자막 시각으로 교체했는지와 그 근거.
   *  applied=false면 skipped에 사유. sources = {caption: 자막 고정, interp: 보간, kept: CTC 유지} */
  caption_scaffold?: {
    applied?: boolean;
    skipped?: string | null;
    moved?: number;
    sources?: { caption?: number; interp?: number; kept?: number };
    rate?: number | null;
    track?: string | null;
    drift_median?: number | null;
  } | null;
  /** 이 싱크를 만든 정렬 스택 식별자(서버 models.ENGINE_VERSION). 응답 최상위 필드를
   *  background이 디버그 표시용으로 여기에 접어 넣는다. null/없음 = 스탬프 도입 전 구세대 */
  engine_version?: string | null;
  /** 엔진 변형 (MMS 강제 폴백 등) — engine_version과 같은 경로로 접힌다 */
  engine_variant?: string | null;
  /** 새 스택 라우팅 판정 근거 — route는 채택된 깊이(fast/medium/heavy), language_source가
   *  "script_census"면 라벨이 비어 가사 문자 계열로 판정했다는 뜻 */
  routing?: {
    route?: string;
    language?: string;
    language_source?: string;
    line_log_conf_median?: number | null;
    threshold?: number;
    stranded_before?: number;
    stranded_after?: number;
  } | null;
}

/** GET /api/sync/{video_id}/previous — 재처리로 덮어써지기 전 세대 (A/B 고스트 비교용) */
export interface SyncPreviousVersion {
  found: boolean;
  timestamps?: EveryricSegment[];
  language?: string | null;
  quality_score?: number | null;
  created_at?: string | null;
  replaced_at?: string | null;
  lyrics_hash?: string | null;
  engine_variant?: string | null;
  engine_version?: string | null;
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
  /** 이 싱크를 만든 정렬 스택 식별자 — 없으면 스탬프 도입 전 구세대 (additive 필드) */
  engine_version?: string | null;
  engine_variant?: string | null;
  created_at?: string;
  error?: string;
  debug?: SyncDebugMeta | null;
  attribution?: SourceAttribution | null;
  tempo?: SongTempo | null;
  key?: SongKey | null;
  /** 다른 영상의 싱크를 빌려온 경우 (inst·커버 링크) — 타이밍은 이미 오프셋·배속 적용됨.
   *  verified=true면 서버가 반주 상관으로 같은 곡임을 확인한 자동 링크 (수동 링크는 false) */
  linked?: {
    source_video_id: string; offset_sec: number; rate?: number | null; verified?: boolean | null;
  } | null;
  /** 이 영상에 저장된 사용자 싱크 오프셋(초) */
  user_offset?: number | null;
  /** 지금 실린 세그먼트 translation의 언어 — lang 쿼리 파라미터를 준 요청에만 의미가 있다.
   *  lang 없이 조회하면 항상 undefined(구버전 서버와 응답 필드 구성이 완전히 동일해야 함) */
  translation_lang?: string | null;
  /** 이 곡에 이미 저장된 번역 레이어의 언어 목록 — 제목바 언어 칩이 "보유"/"미보유"를
   *  가르는 데 쓴다. 구버전 서버는 이 필드 자체가 없으므로 undefined → 칩을 숨긴다. */
  available_langs?: string[] | null;
  /** 이 곡에 저장된 모든 번역 레이어를 통째로 동봉(V2 확장) — 언어 코드 → **timestamps와
   *  같은 세그 순서**의 번역 배열(그 세그에 번역이 없으면 null). lyrics-parser.
   *  segmentsToLines가 timestamps를 정렬·필터링해 LyricLine[]으로 바꿀 때 인덱스가
   *  바뀌므로, 이 필드도 반드시 segmentsToLines에 함께 넘겨 같은 재정렬을 거쳐야 한다
   *  (그냥 배열 인덱스로 lines[i]에 직접 대응시키면 필터·정렬로 어긋난다). 구서버는
   *  필드 자체가 없다 — 그 경우 세션 내 첫 전환까지는 기존 로컬 체인(자막→위키→LLM)이
   *  그대로 동작한다(폴백, 회귀 아님). */
  translations_by_lang?: Record<string, (string | null)[]> | null;
  /** 곡 단위 추임새 후보 [(start, end), ...] — 가사가 주장하지 않은 가창 구간. 새 정렬
   *  스택(owsm/omniasr 앵커) 전용 additive 필드다 — 판정이 아니라 후보이고, 레거시
   *  스택으로 만든 싱크에는 이 필드가 없다(undefined). 구버전 확장은 필드 자체를 모르니
   *  무시하면 그만이다. */
  adlib?: [number, number][] | null;
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

// ── 커버 자동 연결 (GET /api/sync/{video_id}/link-candidates) ─────
// 제목·아티스트로 코퍼스에서 같은 곡 후보를 찾고, 후보가 있으면 **서버가** 반주 상관
// 검증 잡을 자동 제출한다. 클라이언트는 그 잡만 추적하면 된다 — 링크를 만드는 판단은
// 전부 서버에 있다(제목이 맞았다는 이유만으로는 링크가 생기지 않는다).

export interface LinkCandidate {
  video_id: string;
  title?: string | null;
  artist?: string | null;
  /** 제목 유사도 (1.0 = 정규화 정확 일치) — 후보 순위일 뿐, 같은 곡인지의 판정값이 아니다 */
  score: number;
}

export interface LinkCandidatesResponse {
  video_id: string;
  /** has_sync·linked = 연결이 불필요, none·disabled = 후보 없음/기능 off,
   *  submitted·pending = 검증 잡 진행 중, cooldown = 최근에 이미 시도함 */
  status: 'has_sync' | 'linked' | 'disabled' | 'none' | 'submitted' | 'pending' | 'cooldown' | string;
  candidates?: LinkCandidate[];
  /** 낸 후속 작업의 종류 — 오늘은 'link_validate'(반주 상관 검증)뿐 */
  followup?: string | null;
  /** submitted·pending·cooldown일 때의 후속 작업 id (GET /api/link-jobs/{id}로 폴링) */
  job_id?: string | null;
}

/** GET /api/link-jobs/{id} — 반주 상관 검증 잡의 상태 */
export interface LinkJobStatusResponse {
  /** queued | processing | done | failed */
  status: string;
  /** done일 때만 의미 있음 — true면 서버가 이미 SyncLink를 만들었다 */
  match?: boolean | null;
  offset_sec?: number | null;
  confidence?: number | null;
  error?: string | null;
  /** 서버에 잡 기록이 없음(404) — 폴링을 마감시키는 마커 (서버 필드가 아니다) */
  gone?: boolean;
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
  /** 현재 진행 단계명 (다운로드/전사 정렬/보컬 분리/…) + 단계 내 진행률(%) */
  stage?: string | null;
  stage_progress?: number | null;
  /** 서버가 404를 반환 — 잡 기록이 사라짐(서버 재시작 등). 폴링은 실패로 마감한다 */
  gone?: boolean;
}

// ── 서버 오류 표면 ──────────────────────────────────────────────
// 예전에는 서버 요청 실패가 전부 `null` 하나로 뭉개져서, 화면이 "이 곡엔 가사가 없다"와
// "서버가 인증을 거부했다"와 "서버가 꺼져 있다"를 구분할 수 없었다. 아래 타입들은 그
// 구분을 백그라운드 → 콘텐츠 스크립트 → 화면까지 잃지 않고 나르기 위한 것이다.

/** 서버 요청이 실패한 이유의 종류 */
export type ApiFailureKind =
  | 'offline' // 서버에 닿지 못함 (연결 거부·DNS·CORS 등 fetch 자체가 실패)
  | 'timeout' // 제한 시간 안에 응답이 오지 않음
  // 확장에 이 호스트 권한이 없어 **부르기 전에** 막았다 (로컬 서버는 optional 권한이다).
  // 실제로 불러 보면 fetch가 실패해 'offline'로 보이는데, 서버는 멀쩡한 경우다 —
  // 그 오해를 막기 위해 별도 종류로 둔다 (lib/host-permissions.localPermissionBlock).
  | 'permission'
  | 'auth' // 401/403 — API 키가 없거나 틀림
  | 'notfound' // 404 — 엔드포인트나 리소스 없음 (구버전 서버일 수도)
  | 'client' // 그 밖의 4xx
  | 'server' // 5xx
  | 'malformed'; // 2xx인데 본문이 JSON이 아님

export interface ApiFailure {
  kind: ApiFailureKind;
  /** HTTP 상태 코드 — 응답을 받은 경우에만 있다 */
  status?: number;
  /** 서버가 준 error·hint·detail·message를 합친 문구 (API 키는 마스킹된 상태) */
  detail?: string;
  /** 요청 경로 — 쿼리의 키·토큰류 값은 마스킹된 상태 */
  path: string;
  elapsedMs: number;
}

/** 최근 서버 요청 한 건 — 패널의 접이식 로그에 그대로 표시된다 */
export interface ServerLogEntry {
  /** 요청을 보낸 시각 (epoch ms) */
  at: number;
  method: string;
  /** 마스킹된 경로 */
  path: string;
  ok: boolean;
  status?: number;
  kind?: ApiFailureKind;
  detail?: string;
  elapsedMs: number;
}

/** 서버를 쓸 수 있는가 — 못 쓴다면 왜인지까지.
 *  'permission' = 로컬(자체 호스팅) 서버 호스트 권한이 없다. 서버 장애가 아니라 한 번의
 *  허용으로 풀리는 상태라, 화면은 이 경우에만 '권한 설정 열기'를 내놓는다. */
export type ServerStatusKind = 'unknown' | 'ok' | 'offline' | 'auth' | 'error' | 'permission';

export interface ServerStatus {
  kind: ServerStatusKind;
  /** 사용자에게 보여줄 한 줄 사유 ('ok'·'unknown'이면 빈 문자열) */
  reason: string;
  /** 원인 코드 한 조각 — 'HTTP 401', '연결 실패', '응답 없음(타임아웃)' */
  code?: string;
  /** 서버가 준 원문 힌트 (있을 때만) */
  detail?: string;
  /** 이 판정을 만든 시각 (epoch ms) */
  at: number;
}

export interface Settings {
  autoSearch: boolean;
  /** 쇼츠(/shorts/)에서도 가사창 자동 열기 허용 — 기본 꺼짐 */
  autoSearchShorts: boolean;
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
  /** PiP 창 너비(px) — 닫을 때 기억, 0 = 미설정(기존 기본값 440 사용) */
  pipWidth: number;
  /** PiP 창 높이(px) — 닫을 때 기억, 0 = 미설정(showVideo에 따라 500/260 사용) */
  pipHeight: number;
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
  /** 음정 모델 RAW f0 곡선을 디버그 모드와 무관하게 레인에 상시 표시 */
  pitchF0Curve: boolean;
  /** 계이름 표기: korean(도레미)·english(멜로다인식 C4·D#5, 옥타브 포함). 기본 korean(무회귀) */
  solfegeNotation: 'korean' | 'english';
  /** 음정선(f0 곡선·노트 바) 밝기 배율 — 0.2~1.0, 기존 알파값에 곱해진다. 기본 1(현행과
   *  동일 — f0 곡선 0.65·노트 채움 0.55/0.65가 그대로 유지돼 무회귀) */
  pitchLineOpacity: number;
  /** f0 곡선(음정 보는 선) 밝기 배율 0.2~1.5 — 노트 바(pitchLineOpacity)와 별개 */
  pitchF0Opacity: number;
  /** 보컬 발성 구간에서 가사창 패널이 은은하게 밝아지는 글로우 효과 */
  vocalGlow: boolean;
  /** 발음 표기 위치: note = 노트마다 위에 부착, bottom = 화면 하단 중앙(진행률 그라데이션) */
  pitchPronPosition: 'note' | 'bottom';
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
  /** 메트로놈 배속 (0.5|1|2) — 느린 곡은 2배로 세분, 빠른 곡은 절반으로 */
  metronomeRate: number;
  /** 마디 시작 박 (0~3) — 강세·레인 마디선 위치를 함께 이동 */
  metronomeBeat: number;
  /** 멜로디·메트로놈 출력 기기 id (AudioContext.setSinkId) — '' = 기본 출력 */
  audioOutputId: string;
  /** 마이크로 부른 음정을 가라오케 레인에 표시 */
  micPitch: boolean;
  /** 마이크 입력 기기 id — '' = 기본 마이크 */
  micDeviceId: string;
  /** 마이크 음정 옥타브 보정 (옥타브 단위, -2~+2) — 자동 폴딩 전에 적용 */
  micOctave: number;
  /** 전사 신뢰도가 매우 낮은 곡(<0.001)에서 가사창 상단 경고 바 표시 */
  lowConfWarning: boolean;
  /** 전사 잡 완료/실패 시 브라우저 알림 — 다른 탭에 있어도 확인 가능 */
  notifyOnComplete: boolean;
  /** 패널 하단에 내부 상태(비디오 바인딩, 싱크 소스 등) 표시 */
  debugInfo: boolean;
  /** 발음 표기 방식 — 'auto'면 translationLanguage 기준 자동 결정(lib/lang.ts resolveScript) */
  pronunciationScript: 'auto' | 'hangul' | 'romaji' | 'kana' | 'ipa';
  /** 확장 UI 언어 — 'auto'면 브라우저 로케일. 지금은 값만 저장(i18n 태스크에서 소비) */
  uiLanguage: 'auto' | 'ko' | 'en' | 'ja';
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
  /** 이 싱크가 언제 만들어졌는가 — 현지 시각 + 경과 시간으로 미리 다듬은 문구.
   *  "지금 보는 싱크가 어느 시점 파이프라인 산물인가"를 화면에서 바로 알기 위한 것이다
   *  (수정 배포 후에도 옛 싱크는 그대로라, 이걸 모르면 고쳐졌는지 판단할 수 없다). */
  syncCreated: string | null;
}

export interface TranslatedLine {
  original: string;
  translation: string;
  pronunciation?: string | null;
  /** 서버가 이 줄만 복구하지 못했다는 표시(응답 잘림 등) — 원문만 채워져 온다.
   *  서버는 처음부터 이 필드를 보냈지만 타입에 없어 아무도 읽지 않았고, 그래서 일부 줄이
   *  빈 채로 와도 완료 알림이 "번역이 준비됐어요"라고 말했다. 부분 실패를 말하려면 필요하다. */
  failed?: boolean;
}

export interface TranslateResult {
  lines: TranslatedLine[];
  source_lang?: string;
  target_lang?: string;
  engine?: string;
}

/** POST /api/sync/{video_id}/translations 응답 — 거절도 200으로 온다(saved=false).
 *  404(싱크 없음)·422(매칭률<50%·origin 위반·상한)는 request()의 !res.ok 경로로 빠져
 *  null이 된다. saveTranslationLayer는 fire-and-forget이라 호출부는 실패를 로그만 한다. */
export interface SaveTranslationLayerResponse {
  saved: boolean;
  matched: number;
  total: number;
  target_lang: string;
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

/** 자막 한 줄 (타이밍 포함) — 싱크 가사로 바로 표시하는 데 쓴다.
 *  트랙 **목록**은 클라이언트가 워치 페이지에서 직접 읽는다(lib/yt-captions.ts).
 *  **본문**만 서버 경유다 — timedtext URL은 POT 강제로 브라우저 플레이어 밖에선 빈 응답. */
export interface CaptionLine {
  start: number;
  end: number;
  text: string;
}

export type BgRequest =
  // lang은 번역 레이어 언어별 서빙 요청용(옵션) — 없으면 서버는 기존 응답 그대로 준다
  | { type: 'FETCH_LYRICS'; payload: SongInfo & { skipLrclib?: boolean; lang?: string } }
  | { type: 'FETCH_LRCLIB'; payload: SongInfo }
  | { type: 'SEARCH_CANDIDATES'; payload: { title: string; artist: string; duration: number } }
  | { type: 'PICK_LRCLIB'; payload: { id: number } }
  // title·artist는 완성된 싱크에 함께 저장돼 커버 링크 후보 탐색의 단서가 된다 —
  // 이게 없으면 코퍼스에 제목이 쌓이지 않아 후보 탐색이 영원히 빈손이다
  // targetLang·lineMetaLang은 생성 요청자의 번역 언어(옵션, 기본 서버는 "ko") — background가
  // 아직 서버 호출에 전달하지 않으면(구버전 배선) 서버 기본값 "ko"로 생성된다
  | { type: 'GENERATE_SYNC'; payload: { videoId: string; lyrics: string; language?: string; lineMeta?: LineMeta[]; lineMetaPending?: boolean; attribution?: SourceAttribution; title?: string; artist?: string; targetLang?: string; lineMetaLang?: string } }
  /** 진행 중인 잡에 번역·독음을 나중에 붙인다 (다운로드와 번역을 겹치는 경로).
   *  번역이 실패했어도 **빈 배열로 반드시 한 번 보내야** 잡이 대기 상한까지 서 있지 않는다. */
  | { type: 'ATTACH_LINE_META'; payload: { jobId: string; lineMeta: LineMeta[]; attribution?: SourceAttribution; title?: string; artist?: string; lineMetaLang?: string } }
  | { type: 'REGENERATE_SYNC'; payload: { videoId: string; lyrics: string; lineMeta?: LineMeta[]; attribution?: SourceAttribution; title?: string; artist?: string; targetLang?: string; lineMetaLang?: string; minDepth?: 'medium' | 'heavy' } }
  | { type: 'SYNC_LINK'; payload: { videoId: string; sourceVideoId: string; offsetSec: number; rate: number } }
  /** 같은 곡의 다른 영상 후보 탐색 — 후보가 있으면 서버가 검증 잡까지 자동 제출한다 */
  | { type: 'LINK_CANDIDATES'; payload: { videoId: string; title: string; artist?: string } }
  | { type: 'LINK_JOB_STATUS'; payload: { linkJobId: string } }
  | { type: 'SYNC_UNLINK'; payload: { videoId: string } }
  | { type: 'SYNC_RESET'; payload: { videoId: string } }
  | { type: 'SYNC_OFFSET'; payload: { videoId: string; offsetSec: number } }
  | { type: 'SYNC_LIST' }
  /** 이 영상 싱크의 직전 세대 조회 — 디버그 패널의 A/B 고스트 비교용 */
  | { type: 'SYNC_PREVIOUS'; payload: { videoId: string } }
  /** 정렬 품질 별점 + 오류 제보 (수집 전용) */
  | { type: 'SYNC_FEEDBACK'; payload: { videoId: string; rating: number; category?: string; comment?: string } }
  | { type: 'JOB_STATUS'; payload: { jobId: string } }
  | { type: 'JOB_CANCEL'; payload: { jobId: string } }
  | { type: 'NOTIFY'; payload: { id?: string; title: string; message: string } }
  // persist+videoId를 함께 주면 서버가 이 번역을 (video_id, fingerprint, target_lang) 레이어로 저장한다(옵션)
  | { type: 'TRANSLATE'; payload: { text: string; targetLang: string; title?: string; artist?: string; persist?: boolean; videoId?: string } }
  | { type: 'SERVER_HEALTH' }
  | { type: 'SERVER_LOG' }
  /** 권한 관리 페이지(options_ui) 열기 — `chrome.permissions.request()`는 사용자 제스처가
   *  있는 **확장 페이지**에서만 되므로 content script는 여기까지만 할 수 있다.
   *  (service worker에서 request()를 부르면 제스처 컨텍스트가 없어 실패한다.) */
  | { type: 'OPEN_OPTIONS' }
  | { type: 'VOCARO_LOOKUP'; payload: { title: string } }
  | { type: 'VOCARO_PAGE'; payload: { slug: string } }
  | { type: 'MIRAHEZE_LOOKUP'; payload: { title: string } }
  | { type: 'YT_CAPTION_TEXT'; payload: { videoId: string; lang: string; auto: boolean } }
  | { type: 'GENERATE_FROM_CAPTION'; payload: { videoId: string } }
  /** 자막·위키 채택 번역을 서버 레이어로 저장 — fire-and-forget(호출부가 실패를 무시한다).
   *  origin은 그 번역이 어디서 왔는지: 사람 origin(caption·wiki·manual)은 다른 사람
   *  origin이 못 덮지만 llm 위 승격은 허용된다(서버 규칙, 클라이언트는 몰라도 된다). */
  | {
    type: 'SAVE_TRANSLATION_LAYER';
    payload: {
      videoId: string;
      targetLang: string;
      lines: { text: string; translation: string }[];
      origin: 'caption' | 'wiki' | 'manual';
      attribution?: SourceAttribution;
    };
  };

export type ContentMessage =
  | { type: 'TOGGLE_OVERLAY' }
  | { type: 'TOGGLE_DEBUG' }
  | { type: 'SYNC_GENERATED'; payload: { videoId: string } }
  /** 호스트 권한이 허용·철회됐다 (옵션 페이지나 chrome://extensions에서) — 서버 상태를
   *  다시 판정해야 한다. 허용은 다른 탭에서 일어나므로 이 알림이 없으면 허용한 뒤에도
   *  가사창에는 "권한이 없어요" 배너가 그대로 남는다. */
  | { type: 'PERMISSIONS_CHANGED' };

export interface MessageResponse<T = unknown> {
  data?: T;
  error?: string;
  /** 이 요청이 Everyric 서버 호출에서 실패했다면 그 구조화된 사유.
   *  data가 null이어도 이게 있으면 "결과가 없다"가 아니라 "서버가 못 줬다"는 뜻이다. */
  failure?: ApiFailure;
}
