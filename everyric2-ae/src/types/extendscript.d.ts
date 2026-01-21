declare const app: Application;
declare const $: Dollar;

interface Dollar {
  write(text: string): void;
  writeln(text: string): void;
  sleep(milliseconds: number): void;
  os: string;
}

interface Application {
  project: Project;
  beginUndoGroup(undoGroupName: string): void;
  endUndoGroup(): void;
  version: string;
}

interface Project {
  activeItem: CompItem | null;
  file: File | null;
  renderQueue: RenderQueue;
}

interface RenderQueue {
  items: RenderQueueItemCollection;
}

interface RenderQueueItemCollection {
  add(comp: CompItem): RenderQueueItem;
}

interface RenderQueueItem {
  outputModule(index: number): OutputModule;
}

interface OutputModule {
  file: File;
  applyTemplate(templateName: string): void;
}

interface Item {
  name: string;
  id: number;
}

interface CompItem extends Item {
  duration: number;
  frameRate: number;
  width: number;
  height: number;
  numLayers: number;
  layers: LayerCollection;
  markerProperty: MarkerPropertyGroup;
  selectedLayers: Layer[];
  layer(index: number): Layer;
  layer(name: string): Layer;
}

interface LayerCollection {
  addText(text?: string): TextLayer;
  length: number;
}

interface Layer {
  name: string;
  index: number;
  hasAudio: boolean;
  inPoint: number;
  outPoint: number;
  startTime: number;
  source: FootageItem | null;
  property(name: string): Property | PropertyGroup;
}

interface TextLayer extends Layer {
  property(name: "Source Text"): Property<TextDocument>;
  property(name: "Position"): Property<[number, number]>;
  property(name: "Anchor Point"): Property<[number, number]>;
  sourceRectAtTime(time: number, extents: boolean): SourceRect;
}

interface SourceRect {
  left: number;
  top: number;
  width: number;
  height: number;
}

interface FootageItem extends Item {
  file: File | null;
}

interface Property<T = unknown> {
  value: T;
  setValue(value: T): void;
  setValueAtTime(time: number, value: T): void;
  numKeys: number;
  removeKey(index: number): void;
}

interface PropertyGroup {
  numKeys: number;
  setValueAtTime(time: number, value: MarkerValue): void;
  removeKey(index: number): void;
}

interface MarkerPropertyGroup extends PropertyGroup {}

declare class MarkerValue {
  constructor(comment: string);
  comment: string;
  duration: number;
  label: number;
}

interface TextDocument {
  text: string;
  fontSize: number;
  font: string;
  fillColor: [number, number, number];
  strokeColor: [number, number, number];
  strokeWidth: number;
  strokeOverFill: boolean;
  applyStroke: boolean;
  applyFill: boolean;
  justification: ParagraphJustification;
}

declare enum ParagraphJustification {
  LEFT_JUSTIFY = 7413,
  CENTER_JUSTIFY = 7415,
  RIGHT_JUSTIFY = 7414,
}

interface File {
  fsName: string;
  name: string;
}

declare class File {
  constructor(path: string);
  fsName: string;
  name: string;
}
