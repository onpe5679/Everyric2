export interface TimingAtom {
  text: string;
  start: number;
  end: number;
  confidence?: number;
}

/** 발음 표기 종류. 서버가 한 줄에 여러 표기를 함께 실어 보낸다. */
export type PronScript = "hangul" | "romaji" | "kana";

/** 번역 대상 언어. 서버의 translation_layers가 다루는 언어들. */
export type TranslationLanguage = "ko" | "en" | "ja";

export type PronVariants = Partial<Record<PronScript, string>>;

export interface SyncLine {
  text: string;
  start: number;
  end: number;
  confidence?: number;
  translation?: string;
  /**
   * 레거시 발음 슬롯 — **항상 한글 값**이다. 표시할 때 직접 읽지 말고
   * `resolvedPronunciation()`을 거쳐야 한다(romaji·kana 사용자에게 한글이 새지 않도록).
   */
  pronunciation?: string;
  /** 표기별 발음. 서버 신버전이 채운다. */
  pron?: PronVariants;
  atoms: TimingAtom[];
}

export interface SyncDocument {
  lines: SyncLine[];
  language: string;
  sourceLabel: string;
  duration: number;
  /** 이 싱크로 서빙 가능한 번역 언어 — 서버가 알려준 목록. */
  availableLangs?: string[];
  /** 지금 lines[].translation이 어느 언어인지. */
  translationLang?: string;
  /** 언어별 번역 배열({lang: [줄 순서대로]}) — 재조회 없이 언어를 갈아끼우는 데 쓴다. */
  translationsByLang?: Record<string, Array<string | null>>;
}

export type MatchQuality = "exact" | "substring" | "time" | "none";

export interface CharTiming {
  char: string;
  start: number;
  end: number;
  /** 공백·개행이 아니라 실제로 그려지는 글자인가. */
  visible: boolean;
  /** 시각이 atom 측정치가 아니라 추정치인가(단어 atom 분할, 공백 메움, 폴백 배분). */
  interpolated: boolean;
}

export interface CutSession {
  layerIndex: number;
  layerName: string;
  text: string;
  inPoint: number;
  outPoint: number;
  chars: CharTiming[];
  matchQuality: MatchQuality;
  lineText?: string;
  pronunciation?: string;
  translation?: string;
  /** 값이 있으면 이 레이어는 자를 수 없고, 문자열이 그 사유다. */
  blocked?: string;
}

export interface CutPoint {
  /** chars 배열에서 이 인덱스 앞을 자른다. 1..chars.length-1 */
  index: number;
  time: number;
  /** 기본 계산값 그대로인가(드래그로 옮기면 false). */
  auto: boolean;
}

/**
 * 자른 조각이 화면에 남는 방식.
 * - `cumulative`: 제 시각에 나타나 줄이 끝날 때까지 남는다(한 줄이 차례로 채워진다).
 * - `sequential`: 다음 조각이 나오면 사라진다.
 */
export type CutReveal = "cumulative" | "sequential";

export interface CutPiece {
  text: string;
  start: number;
  end: number;
  charStart: number;
  charEnd: number;
  /**
   * 원본 텍스트의 처음부터 이 조각의 끝까지. host가 "접두사+조각"에서 "조각"을 빼는 방식으로
   * 앞선 글자들의 폭을 재는 데 쓴다. ExtendScript는 코드포인트 단위 slice를 못 하므로
   * 패널이 만들어서 넘긴다.
   */
  head: string;
}

export interface TextLayerInfo {
  index: number;
  name: string;
  inPoint: number;
  outPoint: number;
  text: string;
  sourceTextKeys: number;
  locked: boolean;
}

export interface CompInfo {
  hasComp: boolean;
  /** 컴포지션 고유 id. 프로젝트 경로와 함께 싱크 데이터를 매어 두는 열쇠가 된다. */
  compId?: number;
  projectPath?: string;
  name?: string;
  width?: number;
  height?: number;
  duration?: number;
  frameRate?: number;
  time?: number;
  selectedTextLayers?: TextLayerInfo[];
  generatedLayerCount?: number;
  everyricMarkerCount?: number;
  audioLayers?: Array<{
    index: number;
    name: string;
    inPoint: number;
    outPoint: number;
    filePath?: string;
  }>;
  error?: string;
}

export type Density = "readable" | "balanced" | "rhythmic";
export type LayoutPreset = "auto" | "center" | "editorial" | "split" | "diagonal";
export type RevealMode = "cumulative" | "simultaneous";
export type LayerOrder = "bottom-to-top" | "top-to-bottom";
export type UiLocale = "ko" | "ja" | "en";
export type TypographyMode = "designed" | "line";

export interface PlannerOptions {
  density: Density;
  layout: LayoutPreset;
  width: number;
  height: number;
  frameRate: number;
  fontSize: number;
  preRollFrames: number;
  postRollFrames: number;
  pauseThreshold: number;
  maxBlocksPerCard: number;
  phraseTargetChars: number;
  maxTokensPerBlock: number;
  revealMode: RevealMode;
}

export interface TypographyBlock {
  id: string;
  cardId: string;
  text: string;
  start: number;
  end: number;
  position: [number, number];
  fontSize: number;
  rotation: number;
  justification: "left" | "center" | "right";
  color: [number, number, number];
  emphasis: number;
}

export interface TypographyCard {
  id: string;
  start: number;
  end: number;
  sourceText: string;
  blocks: TypographyBlock[];
}

export interface TypographyPlan {
  groupId: string;
  cards: TypographyCard[];
  blocks: TypographyBlock[];
  warnings: string[];
}

export interface FillAssignment {
  layerIndex: number;
  layerName: string;
  text: string;
  inPoint: number;
  outPoint: number;
  skippedReason?: string;
}

export interface LocalSyncOptions {
  pythonPath: string;
  language: string;
  audioPath: string;
  lyrics: string;
  runtimeRoot?: string;
  modelDir?: string;
  minDepth?: "fast" | "medium" | "heavy";
}

export interface AppSettings {
  uiLocale: UiLocale;
  pythonPath: string;
  runtimeRoot: string;
  modelDir: string;
  minDepth: "fast" | "medium" | "heavy";
  language: string;
  density: Density;
  typographyMode: TypographyMode;
  layout: LayoutPreset;
  fontSize: number;
  preRollFrames: number;
  postRollFrames: number;
  pauseThreshold: number;
  maxBlocksPerCard: number;
  phraseTargetChars: number;
  maxTokensPerBlock: number;
  revealMode: RevealMode;
  layerOrder: LayerOrder;
  replacePrevious: boolean;
  autoLabelColors: boolean;
  /** 커팅으로 나뉜 조각을 원본 위치에 그대로 둘지. 끄면 각 글자가 있던 자리로 옮긴다. */
  keepCutPosition: boolean;
  /** 자른 조각이 줄 끝까지 남을지(cumulative), 다음 조각이 나오면 사라질지(sequential). */
  cutReveal: CutReveal;
  /** 이미 만들어진 싱크를 영상 ID로 조회할 서버. */
  /** 서버에서 받아올 번역 언어. */
  translationLanguage: TranslationLanguage;
  /** 발음 표기. 'auto'면 번역 언어를 따른다(en→romaji, ja→kana, 그 밖→hangul). */
  pronunciationScript: PronScript | "auto";
}

export interface HostResult {
  ok: boolean;
  created?: number;
  updated?: number;
  skipped?: number;
  removed?: number;
  generatedLayerCount?: number;
  markerCount?: number;
  error?: string;
  warnings?: string[];
}

export interface EnvironmentReport {
  ready: boolean;
  everyricVersion: string;
  nodeVersion: string;
  platform: string;
  cpu: string;
  systemMemoryGb: number;
  gpuName?: string;
  vramTotalMb?: number;
  vramFreeMb?: number;
  cudaVersion?: string;
  recommended: {
    minimumVramGb: number;
    comfortableVramGb: number;
    systemMemoryGb: number;
  };
  notes: string[];
}
