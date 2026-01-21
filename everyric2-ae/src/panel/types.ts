export interface TimestampData {
  text: string;
  start: number;
  end?: number;
  translation?: string;
  pronunciation?: string;
  confidence?: number;
}

export interface SyncResult {
  segments: TimestampData[];
  metadata: {
    duration: number;
    language: string;
    engine: string;
  };
}

export interface CompInfo {
  hasComp: boolean;
  name?: string;
  duration?: number;
  frameRate?: number;
  width?: number;
  height?: number;
  numLayers?: number;
  audioLayers?: AudioLayerInfo[];
  existingMarkers?: number;
  error?: string;
}

export interface AudioLayerInfo {
  index: number;
  name: string;
  inPoint: number;
  outPoint: number;
  duration: number;
  filePath?: string;
}

export interface Settings {
  cliPath: string;
  apiUrl: string;
  markerColor: string;
  fontSize: number;
  processMode: "local" | "cloud";
  outputType: "markers" | "textLayers" | "both";
  translate: boolean;
  pronunciation: boolean;
  language: string;
  segmentMode: "line" | "word" | "character";
}

export interface LrcLibResult {
  id: number;
  name: string;
  trackName: string;
  artistName: string;
  albumName?: string;
  duration: number;
  syncedLyrics?: string;
  plainLyrics?: string;
}

export type ProcessingStatus = "idle" | "processing" | "success" | "error";
