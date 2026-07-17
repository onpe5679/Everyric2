export interface TimingAtom {
  text: string;
  start: number;
  end: number;
  confidence?: number;
}

export interface SyncLine {
  text: string;
  start: number;
  end: number;
  confidence?: number;
  translation?: string;
  pronunciation?: string;
  atoms: TimingAtom[];
}

export interface SyncDocument {
  lines: SyncLine[];
  language: string;
  sourceLabel: string;
  duration: number;
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
  engine: string;
  language: string;
  audioPath: string;
  lyrics: string;
}

export interface AppSettings {
  uiLocale: UiLocale;
  pythonPath: string;
  engine: string;
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
