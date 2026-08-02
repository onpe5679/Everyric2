import type { LyricLine, SongKey, SongTempo, SyncDebugMeta } from '../types';
import type { MicSample } from '../lib/mic-pitch';
import { resolvedPronSegments, resolvedPronunciation, type PronScript } from '../lib/lang';
import { t } from '../lib/i18n';

/**
 * 가라오케 음정 레인(피아노롤) 렌더러 — 캔버스 하나에 오선·노트·가사·발음·번역·f0를 그린다.
 *
 * PiP 창과 메인 가사창(오버레이 패널)이 **같은 코드로** 레인을 그리도록 pip.ts의
 * renderPitch를 통째로 옮겨 온 순수 렌더러다. 창 관리·설정 저장·포인터 상호작용은
 * 호출부(pip.ts / overlay.ts)에 남고, 여기는 "주어진 캔버스에 지금 시각을 그린다"만 한다.
 *
 * 캔버스가 어느 문서에 있든 상관하지 않는다(PiP 문서 / 메인 문서의 Shadow DOM) —
 * 필요한 창은 canvas.ownerDocument.defaultView에서 얻고, 색은 호출부가 지정한
 * 색 소스 엘리먼트의 CSS 변수에서 읽는다.
 */

// 계이름(고정도법 솔페지) — 한국어 UI의 익숙한 값을 폴백으로 두고, 실제로는 매 렌더마다
// solfegeNames()가 t()로 다시 읽는다(언어별로 값이 다르므로 모듈 상수로 굳히지 않는다)
const PITCH_NAMES_KO = ['도', '도#', '레', '레#', '미', '파', '파#', '솔', '솔#', '라', '라#', '시'];
/** 카탈로그값(쉼표 12개 토큰)을 매 호출마다 분해 — t()가 배열을 못 돌려주는 messages.json
 *  제약을 우회한다. 실패하면(분해 결과가 12개가 아니면) 한국어 폴백으로 안전하게 되돌아간다 */
function solfegeNames(): string[] {
  const names = t('pip.solfege.names').split(',');
  return names.length === 12 ? names : PITCH_NAMES_KO;
}
// 사이드바 음정 라벨 (영문, 멜로다인식) — C4 = MIDI 60
const PITCH_NAMES_EN = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const BLACK_KEYS = new Set([1, 3, 6, 8, 10]);

/** K2: 노트 바에 붙는 음정 라벨 — korean은 기존 계이름(solfegeNames, i18n 카탈로그),
 *  english는 사이드바와 같은 멜로다인식 음이름+옥타브(C4 = MIDI 60, 같은 계산식 재사용). */
function noteLabel(midi: number, notation: 'korean' | 'english'): string {
  const pc = ((midi % 12) + 12) % 12;
  if (notation === 'english') return `${PITCH_NAMES_EN[pc]}${Math.floor(midi / 12) - 1}`;
  return solfegeNames()[pc];
}

/** CTC 신뢰도 로그 버킷 색 — 패널(overlay.css의 .ey-conf-*)과 반드시 같은 값 유지 */
const CONF_COLOR_LOW = '#ff6b6b';
const CONF_COLOR_MID = '#ffd166';
const CONF_COLOR_OK = '#51cf66';
export function confBucketColor(conf: number): string {
  return conf < 1e-4 ? CONF_COLOR_LOW : conf < 2e-2 ? CONF_COLOR_MID : CONF_COLOR_OK;
}

/** 디버그 오버레이: 세이프가드 규칙별 마커 색 */
const FIX_COLORS: Record<string, string> = {
  stretch: '#ff6b6b', // 8s+ 늘어짐 클램프
  repeat: '#ff8787',  // 반복행 outlier 클램프
  pull: '#4dabf7',    // 간주 후 시작 당김
  tail: '#ffa94d',    // 끝음 연장
  snap: '#da77f2',    // 무음 온셋 스냅
  leak: '#20c997',    // 간주 역방향 누출 — ja 타이밍 스플라이스
  // 원문 글자 융합 — 라인 경계는 그대로 두고 라인 **안쪽** 글자 분포만 다시 만든다.
  // 채도 높은 핑크: 빨강 계열(stretch/repeat)·주황(tail)·보라(snap)와 색상이 뚜렷이 갈린다
  fuse: '#f06595',
  // 자막 스캐폴드 — 붕괴 곡의 줄 시작을 사람이 찍은 자막 시각으로 교체 (CTC 결과 폐기).
  // 노랑: 다른 보정과 달리 «음향이 아니라 자막이 낸 타이밍»이라 한눈에 구분돼야 한다
  scaffold: '#fcc419',
};

// 디버그 타이밍 레인 — CTC가 재생 중인 라인의 원문 글자·발음 음절에 실제로 준 시각을
// 시간축에 찍는다("라인 집중 뷰" — 여러 줄을 한꺼번에 보여주던 예전 방식은 고스트와
// 뭉쳐 헷갈림의 주범이었다).
// 독음(ko) 정렬 곡은 원문 글자 시간이 "발음 음절 → 모라 → 원문 글자" 3단 역매핑
// 산물이라, 여러 글자가 같은 시각에 뭉치거나 길이 0으로 찍힌다(실측: 독음 정렬 곡은
// 3글자 이상 동시 시작이 38~59%인데 원문 정렬 곡은 2%). 원문·heard·발음 세 레인을
// 나란히 그리면 어디서 어긋났는지 눈으로 바로 구분된다.
// 틱·글자 색은 정렬 신뢰도(confBucketColor)로 칠한다 — 레인이 위아래로 나뉘어 있어
// 색으로 원문/발음을 구분할 필요가 없으므로 그 채널을 신뢰도에 쓴다. 아래 색은
// 신뢰도가 없는 세그먼트의 폴백이자 레인 이름표 색이다.
const TIMING_WORD = '#74c0fc';
const TIMING_PRON = '#63e6be';
/** heard 레인(CTC가 실제로 들은 글자) — 원문 레인과 뚜렷이 구분되도록 흐린 회색 */
const TIMING_HEARD = '#868e96';

/** 글자별 시각(heard_spans)이 없을 때 heard 문자열을 라인 구간에 균등 배치하는 폴백 */
function evenlyPlaceHeard(heard: string, start: number, end: number): [string, number][] {
  const chars = Array.from(heard.trim());
  if (chars.length === 0) return [];
  const span = Math.max(0.01, end - start);
  const out: [string, number][] = [];
  for (let i = 0; i < chars.length; i++) {
    out.push([chars[i], start + ((i + 0.5) / chars.length) * span]);
  }
  return out;
}

export interface PitchNote {
  midi: number;
  start: number;
  end: number;
  /** 이 노트(음절)에 대응하는 발음 표기 — 계이름처럼 노트에 직접 붙여 그린다 */
  pron?: string;
}

interface PitchWord {
  word: string;
  start: number;
  end: number;
  confidence?: number;
}

/** 라인 메타 (카운트다운·현재 라인 번역/발음 폴백용) — lines 배열과 인덱스 1:1 */
interface PitchLine {
  line: LyricLine;
  start: number;
  end: number;
  hasNotes: boolean;
}

/** setLines에서 미리 평탄화해 두는 레인 데이터 (렌더는 시간 창으로 필터만) */
interface PitchData {
  pages: PitchLine[];
  notes: PitchNote[];
  words: PitchWord[];
  /** 곡 전체 고정 세로 스케일 (라인마다 출렁이지 않게) */
  lo: number;
  hi: number;
}

interface PitchColors {
  /** 오선·번역 — 배경 요소 */
  faint: string;
  /** 아직 안 부른 노트·계이름 라벨·발음·스위프 선 */
  dim: string;
  /** 부른 부분·현재 노트 글로우·스위프 마커 */
  accent: string;
  /** 가사·현재 노트 테두리 */
  text: string;
}

/** 마지막 렌더 뷰포트 — 클릭·드래그의 좌표→시간 역변환용 */
export interface PitchLaneView {
  t0: number;
  W: number;
  plotX: number;
  plotW: number;
  staffBottom: number;
}

/**
 * 렌더 옵션 — 전부 "설정에서 온 표시 취향"이라 언제든 통째로 갈아끼울 수 있다(setOptions).
 * 곡 데이터(라인·템포·키·디버그 메타)는 옵션이 아니라 전용 setter로 들어온다.
 */
export interface PitchLaneOptions {
  /** 레인 자체를 그릴지 — 꺼져 있으면 render가 아무것도 하지 않는다 (PiP의 pitchGuide 게이트) */
  enabled: boolean;
  /** 표시 구간(마디 수) — 서버 BPM 기준, 템포 없으면 120BPM 가정 폴백 */
  windowMeasures: number;
  /** 진행 방식: page = 화면 고정 + 플레이헤드 이동, scroll = 플레이헤드 고정 + 횡스크롤 */
  scrollMode: 'page' | 'scroll';
  /** 글자 크기 배율 */
  fontScale: number;
  /** 계이름 표기: korean(도레미)·english(C4·D#5 등, 옥타브 포함) */
  solfege: 'korean' | 'english';
  /** 음정선(노트 바) 밝기 배율 0.2~1.0 — 기존 알파값에 곱해진다 */
  lineOpacity: number;
  /** f0 곡선(음정 보는 선) 밝기 배율 0.2~1.5 — 노트 바와 별개 페이더 */
  f0Opacity: number;
  /** 발음 표기 위치: note = 노트마다 위에 부착, bottom = 화면 하단 중앙, both = 동시 */
  pronPosition: 'note' | 'bottom' | 'both';
  /** 발음 표기 방식(hangul/romaji/kana/…) — 호출부가 lib/lang.resolveScript로 해석해 넘긴다 */
  pronScript: PronScript;
  /** RAW f0 곡선 상시 표시 (설정 pitchF0Curve) — 디버그 언더레이와 독립 */
  showF0: boolean;
  /** 디버그: 글자별 CTC 신뢰도 색 + 진단 레이어(VAD 스트립·타이밍 레인·헤더) */
  showConfidence: boolean;
  /** 긴 묵음 뒤 가사 시작 전 4·3·2·1 카운트다운 */
  countdown: boolean;
  /** 마디 시작 박(0~3) — 메트로놈 강세와 마디선이 함께 이동 */
  metronomeBeat: number;
  /** 마이크 음정 옥타브 보정 (옥타브 단위) */
  micOctave: number;
  /** 원본 video 배속 — 마이크 궤적의 벽시계 나이를 곡 시간축으로 환산할 때 쓴다 */
  playbackRate: number;
  /** 마이크 음정 궤적 공급자 — null이면 궤적을 그리지 않는다(메인 패널 기본) */
  getMicSamples: (() => MicSample[]) | null;
}

const DEFAULT_OPTIONS: PitchLaneOptions = {
  enabled: true,
  windowMeasures: 4,
  scrollMode: 'page',
  fontScale: 1.2,
  solfege: 'korean',
  lineOpacity: 1,
  f0Opacity: 1,
  pronPosition: 'note',
  pronScript: 'hangul',
  showF0: true,
  showConfidence: false,
  countdown: true,
  metronomeBeat: 0,
  micOctave: 0,
  playbackRate: 1,
  getMicSamples: null,
};

export class PitchLaneRenderer {
  private canvas: HTMLCanvasElement | null = null;
  /** CSS 변수(--ey-accent 등)를 읽어 갈 대상 — PiP는 문서 루트, 오버레이는 패널 엘리먼트 */
  private colorSource: Element | null = null;
  private opts: PitchLaneOptions;
  private lines: LyricLine[] = [];
  private data: PitchData = { pages: [], notes: [], words: [], lo: 57, hi: 71 };
  private index = -1;
  private tempo: SongTempo | null = null;
  private songKey: SongKey | null = null;
  /** 곡 단위 정렬 진단 (디버그 모드 레인 오버레이: VAD/간주 스트립·RAW f0·정렬 텍스트) */
  private debugMeta: SyncDebugMeta | null = null;
  /** 라인 confidence 통계 — setLines에서 1회 계산 (디버그 헤더 표시용) */
  private confStats: { med: number; ok: number; mid: number; low: number } | null = null;
  private colors: PitchColors | null = null;
  private view: PitchLaneView | null = null;
  /** 일시정지 중 수동 스크롤 창 시작 시각 — 재생 재개 시 호출부가 해제한다 */
  private manualT0: number | null = null;
  private paused = false;

  constructor(opts: Partial<PitchLaneOptions> = {}) {
    this.opts = { ...DEFAULT_OPTIONS, ...opts };
  }

  /** 그릴 캔버스와 색 소스를 지정 — 창을 새로 열 때마다 갈아끼운다(색 캐시는 버린다) */
  attach(canvas: HTMLCanvasElement, colorSource: Element): void {
    this.canvas = canvas;
    this.colorSource = colorSource;
    this.colors = null;
  }

  /** 캔버스가 사라졌을 때(창 닫힘) — 곡 데이터는 남기고 그리기 대상만 끊는다 */
  detach(): void {
    this.canvas = null;
    this.colorSource = null;
    this.colors = null;
    this.view = null;
  }

  setOptions(patch: Partial<PitchLaneOptions>): void {
    const scriptChanged = patch.pronScript !== undefined && patch.pronScript !== this.opts.pronScript;
    this.opts = { ...this.opts, ...patch };
    // 노트에 부착된 발음은 collectPitchData가 만든다 — 표기 방식이 바뀌면 다시 계산해야 한다
    if (scriptChanged) this.data = collectPitchData(this.lines, this.opts.pronScript);
  }

  options(): Readonly<PitchLaneOptions> {
    return this.opts;
  }

  setLines(lines: LyricLine[]): void {
    this.lines = lines;
    this.index = -1;
    this.data = collectPitchData(lines, this.opts.pronScript);
    const confs = lines.map(l => l.confidence).filter((v): v is number => v != null).sort((a, b) => a - b);
    const low = confs.filter(v => v < 1e-4).length / Math.max(1, confs.length);
    const mid = confs.filter(v => v >= 1e-4 && v < 2e-2).length / Math.max(1, confs.length);
    this.confStats = confs.length === 0 ? null : {
      med: confs[Math.floor(confs.length / 2)],
      ok: 1 - low - mid,
      mid,
      low,
    };
  }

  setIndex(index: number): void {
    this.index = index;
  }

  /** 서버가 추정한 곡 템포 — 마디 창 폭·비트 격자에 사용, null이면 초 단위 폴백 */
  setTempo(tempo: SongTempo | null): void {
    this.tempo = tempo && tempo.bpm > 0 ? tempo : null;
  }

  /** 서버가 추정한 곡 키 — 레인 좌상단 표시용 */
  setKey(key: SongKey | null): void {
    this.songKey = key && key.name ? key : null;
  }

  setDebugMeta(meta: SyncDebugMeta | null): void {
    this.debugMeta = meta;
  }

  /** 테마 변경 등으로 CSS 변수가 바뀌었을 때 — 캐시된 색을 버린다(다음 렌더에서 재판독) */
  refreshColors(): void {
    this.colors = null;
  }

  /** 노트가 하나라도 있는가 — 레인 표시 여부 판단(호출부의 게이트) */
  hasNotes(): boolean {
    return this.data.notes.length > 0;
  }

  /** 마지막 렌더 뷰포트 — 클릭 시크·드래그 팬의 좌표↔시간 변환용 */
  viewport(): PitchLaneView | null {
    return this.view;
  }

  /** 이 시각을 품는 노트 — 클릭 시크의 노트 시작 스냅용 */
  noteAt(time: number): PitchNote | undefined {
    return this.data.notes.find(n => n.start <= time && time <= n.end);
  }

  /** 일시정지 중 수동 탐색 뷰포트 — null이면 오토스크롤로 복귀 */
  setManualT0(t0: number | null): void {
    this.manualT0 = t0;
  }

  getManualT0(): number | null {
    return this.manualT0;
  }

  /**
   * 음정 레인 렌더 — 창 폭이 N마디(서버 추정 BPM 기준)로 일정한 오선지.
   * page 모드(기본)는 마디 창이 고정된 채 플레이헤드가 왼→오로 쓸고 지나가고,
   * scroll 모드는 플레이헤드가 좌측 28%에 고정된 채 오선이 횡스크롤된다.
   * 간주는 빈 오선으로 지나가고, 발음은 계이름처럼 각 노트 앞머리에 붙는다.
   */
  render(now: number, paused: boolean): void {
    this.paused = paused;
    const canvas = this.canvas;
    const win = canvas?.ownerDocument.defaultView ?? null;
    if (!canvas || !win || !this.opts.enabled || this.data.notes.length === 0) return;

    const dpr = win.devicePixelRatio || 1;
    const cw = canvas.clientWidth;
    const ch = canvas.clientHeight;
    if (cw === 0 || ch === 0) return;
    const bw = Math.round(cw * dpr);
    const bh = Math.round(ch * dpr);
    if (canvas.width !== bw || canvas.height !== bh) {
      canvas.width = bw;
      canvas.height = bh;
    }
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cw, ch);

    const colors = this.colors ?? (this.colors = readPitchColors(win, this.colorSource ?? win.document.documentElement));
    const { pages, notes, words, lo, hi } = this.data;

    // ── 세로 레이아웃: 오선 영역 + 가사 줄 + (발음 폴백 줄) + 번역 줄
    // pronPosition: 'note'=노트 부착(음절 타이밍이 없을 때만 하단 폴백),
    // 'bottom'=항상 하단 줄(진행률 그라데이션), 'both'=노트 부착 + 하단 줄 동시
    // (운영자 요청 2026-08-03: "노트 말고도 밑에도").
    const noteAttach = this.opts.pronPosition !== 'bottom';
    const hasSegs = noteAttach
      && pages.some(p => (resolvedPronSegments(p.line, this.opts.pronScript)?.length ?? 0) > 0);
    const hasPronRow = (this.opts.pronPosition !== 'note' || !hasSegs)
      && pages.some(p => resolvedPronunciation(p.line, this.opts.pronScript));
    const hasTr = pages.some(p => p.line.translation);
    const fs = this.opts.fontScale;
    // 배율은 반드시 클램프 **이후에** 곱한다 — 예전엔 상한(34px 등) 안에서 곱해서,
    // 레인이 조금만 커져도 상한에 걸려 배율이 계이름(무상한)에만 먹히는 것처럼 보였다
    const lyricH = Math.round(Math.max(16, Math.min(34, ch * 0.15)) * fs);
    const lyricPx = Math.max(13, Math.round(lyricH * 0.72));
    const pronPx = Math.max(10, Math.round(lyricPx * 0.8));
    const pronRowH = hasPronRow ? pronPx + 6 : 0;
    const trPx = Math.round(Math.max(12, Math.min(22, ch * 0.085)) * fs);
    const trH = hasTr ? trPx + 7 : 0;
    const namePx = Math.max(10, Math.round(11 * fs));
    const padTop = 2;
    const staffH = Math.max(30, ch - padTop - lyricH - pronRowH - trH - 2);

    // ── 고정 시간 스케일: 창 폭 = N마디(4/4 가정) — 템포 없으면 120BPM 가정 폴백
    const secPerBeat = this.tempo ? 60 / this.tempo.bpm : 0.5;
    const W = this.opts.windowMeasures * 4 * secPerBeat;
    // 좌측 음정 라벨 사이드바(멜로다인식) — 플롯은 그만큼 오른쪽에서 시작
    const sidebarW = Math.max(24, Math.round(27 * fs));
    const plotX = sidebarW;
    const plotW = Math.max(10, cw - sidebarW);
    let t0: number;
    if (this.opts.scrollMode === 'page') {
      // 페이지 모드: 창은 고정한 채 플레이헤드가 이동하되, 페이지를 창 폭의 75%씩만
      // 전진시킨다 — 플레이헤드가 75% 지점에 닿으면 다음 페이지로 넘어가고, 아직
      // 안 부른 마지막 25%가 새 페이지 왼쪽에 그대로 보여 다음 가사를 미리 읽을 수 있다
      const offset = this.tempo?.beat_offset ?? 0;
      const step = W * 0.75;
      t0 = offset + Math.floor((now - offset) / step) * step;
    } else {
      // 스크롤 모드: 플레이헤드 좌측 28% 고정, 오선이 흐른다
      t0 = now - W * 0.28;
    }
    // 일시정지 중 드래그·Shift+휠로 옮긴 수동 뷰포트가 있으면 우선한다
    if (this.paused && this.manualT0 != null) t0 = this.manualT0;
    const x = (t: number) => plotX + ((t - t0) / W) * plotW;
    const playheadX = x(now);
    this.view = { t0, W, plotX, plotW, staffBottom: padTop + staffH };

    // ── 곡 전체 고정 세로 스케일 (위아래 덧줄 여백 포함)
    const marginY = staffH * 0.16;
    const usable = staffH - marginY * 2;
    const semiPx = usable / Math.max(1, hi - lo);
    const y = (midi: number) => padTop + marginY + usable - (midi - lo) * semiPx;

    // ── 반음 격자 (멜로다인식): 검은건반 칸은 옅은 밴드, C 경계는 진한 선.
    // 음역이 아주 넓어 칸이 뭉개지면(semiPx<3) 예전 오선 5줄로 폴백.
    ctx.fillStyle = colors.faint;
    if (semiPx >= 3) {
      for (let m = lo; m <= hi; m++) {
        if (BLACK_KEYS.has(((m % 12) + 12) % 12)) {
          ctx.globalAlpha = 0.1;
          ctx.fillRect(plotX, y(m) - semiPx / 2, plotW, semiPx);
        }
      }
      for (let m = lo; m <= hi + 1; m++) {
        const pc = ((m % 12) + 12) % 12;
        ctx.globalAlpha = pc === 0 ? 0.55 : 0.16; // C 아래 경계(B와의 사이)는 진하게
        ctx.fillRect(plotX, y(m) + semiPx / 2 - 0.5, plotW, 1);
      }
      ctx.globalAlpha = 1;
    } else {
      for (let i = 0; i < 5; i++) {
        ctx.fillRect(plotX, padTop + marginY + (usable / 4) * i - 0.5, plotW, 1);
      }
    }
    // ── 사이드바: 배경 + 영문 음정 라벨 — 칸 밀도에 따라 전부/흰건반/C만
    ctx.fillStyle = colors.faint;
    ctx.globalAlpha = 0.22;
    ctx.fillRect(0, padTop, sidebarW, staffH);
    ctx.globalAlpha = 1;
    ctx.font = `${Math.max(7, Math.min(10, Math.floor(semiPx + 2)))}px ui-monospace, monospace`;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let m = lo; m <= hi; m++) {
      const pc = ((m % 12) + 12) % 12;
      if (semiPx < 9 && (semiPx >= 5 ? BLACK_KEYS.has(pc) : pc !== 0)) continue;
      ctx.fillStyle = pc === 0 ? colors.text : colors.dim;
      ctx.fillText(`${PITCH_NAMES_EN[pc]}${Math.floor(m / 12) - 1}`, sidebarW - 3, y(m));
    }
    ctx.textAlign = 'center';
    // 세로 눈금: 템포가 있으면 첫 비트에 정렬된 비트(옅게)/마디(진하게) 격자,
    // 없으면 1초 간격 폴백
    if (this.tempo) {
      const offset = this.tempo.beat_offset ?? 0;
      for (let b = Math.ceil((t0 - offset) / secPerBeat); ; b++) {
        const t = offset + b * secPerBeat;
        if (t >= t0 + W) break;
        // 사용자가 고른 마디 시작 박(metronomeBeat)에 맞춰 마디선을 이동 — 메트로놈 강세와 일치
        const isMeasure = (((b - this.opts.metronomeBeat) % 4) + 4) % 4 === 0;
        ctx.globalAlpha = isMeasure ? 0.55 : 0.18;
        ctx.fillRect(x(t) - (isMeasure ? 0.75 : 0.5), padTop, isMeasure ? 1.5 : 1, staffH);
      }
    } else {
      ctx.globalAlpha = 0.3;
      for (let t = Math.ceil(t0); t < t0 + W; t++) {
        ctx.fillRect(x(t) - 0.5, padTop, 1, staffH);
      }
    }
    ctx.globalAlpha = 1;

    // ── 디버그 배경 레이어: VAD 가창/간주 스트립 + star 흡수 밴드 + RAW f0 곡선.
    // f0 곡선만은 설정(pitchF0Curve)으로 디버그 없이도 상시 표시 가능
    if (this.opts.showConfidence) {
      this.renderDebugUnderlay(ctx, t0, W, x, y, lo, hi, cw, padTop, staffH);
    } else if (this.opts.showF0) {
      this.renderF0Curve(ctx, t0, W, x, y, lo, hi);
    }

    // ── 노트 막대 + 계이름(위) + 발음(아래) — 시간 창 안의 노트만
    // 페이지 모드는 창 밖 요소를 그리면 가장자리에 다음 페이지 글자가 뭉치므로 여유 0
    const edgePad = this.opts.scrollMode === 'page' ? 0 : 0.5;
    const vis = notes.filter(n => n.end > t0 - edgePad && n.start < t0 + W + edgePad);
    const noteH = Math.max(5, Math.min(13, semiPx * 1.6));
    const noteR = Math.min(noteH / 2, 4);
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (const n of vis) {
      const x1 = x(n.start);
      const x2 = x(n.end);
      const w = Math.max(3, x2 - x1 - 1);
      const top = y(n.midi) - noteH / 2;
      const isCurrent = n.start <= now && now < n.end;

      // 노트 채움은 반투명 — 채워진 뒤에도 격자·f0 곡선이 비쳐 보이게. K3: 기존 알파값에
      // lineOpacity를 곱한다(기본 1 = 현행과 동일, 무회귀).
      ctx.fillStyle = colors.dim;
      ctx.globalAlpha = 0.55 * this.opts.lineOpacity;
      ctx.beginPath();
      ctx.roundRect(x1, top, w, noteH, noteR);
      ctx.fill();
      if (now > n.start) {
        const sungW = Math.max(2, Math.min(w, x(now) - x1));
        ctx.fillStyle = colors.accent;
        ctx.globalAlpha = 0.65 * this.opts.lineOpacity;
        ctx.beginPath();
        ctx.roundRect(x1, top, sungW, noteH, noteR);
        ctx.fill();
      }
      ctx.globalAlpha = 1;
      if (isCurrent) {
        // 지금 불러야 하는 노트: accent 글로우 + 밝은 테두리로 강조
        ctx.save();
        ctx.shadowColor = colors.accent;
        ctx.shadowBlur = 8;
        ctx.strokeStyle = colors.text;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.roundRect(x1, top, w, noteH, noteR);
        ctx.stroke();
        ctx.restore();
      }

      // 글자는 노트 중앙이 아니라 노트 앞머리(시작 지점)에 붙인다
      const lx = x1 + 1;
      ctx.textAlign = 'left';
      // 계이름 — 항상 노트 위
      if (w >= 14) {
        ctx.font = `bold ${namePx}px system-ui, sans-serif`;
        ctx.fillStyle = isCurrent ? colors.text : colors.dim;
        ctx.fillText(noteLabel(n.midi, this.opts.solfege), lx, Math.max(namePx * 0.7, top - namePx * 0.7));
      }
      // 발음 — 계이름처럼 노트 바로 아래에 부착 (설정 pronPosition === 'note'일 때만;
      // 'bottom'이면 collectPitchData가 채워둔 n.pron은 무시하고 하단 폴백 줄만 사용)
      if (n.pron && noteAttach) {
        ctx.font = `${pronPx}px system-ui, sans-serif`;
        ctx.fillStyle = n.start <= now ? colors.accent : colors.dim;
        ctx.fillText(n.pron, lx, Math.min(padTop + staffH - pronPx / 2, top + noteH + 2 + pronPx / 2));
      }
      ctx.textAlign = 'center';
    }

    // ── 마이크 음정 궤적 — 사용자가 부른 피치를 청록 점 트레일로 (노트 accent와 구분되는 색)
    const mic = this.opts.getMicSamples?.();
    if (mic && mic.length > 0) {
      const wallNow = performance.now() / 1000;
      for (const s of mic) {
        const age = wallNow - s.at;
        if (age > 2.5) continue;
        // 곡 시간은 벽시계보다 배속만큼 빨리/느리게 흐른다 — 배속 재생 중 궤적이
        // 뒤로 밀리지 않게 나이를 배속으로 환산해 사상한다
        const t = now - age * this.opts.playbackRate;
        if (t < t0 || t > t0 + W) continue;
        // 사용자 지정 옥타브 보정(설정) 먼저 적용한 뒤, 남는 오차는 레인 음역으로 폴딩
        let m = s.midi + this.opts.micOctave * 12;
        while (m < lo && m + 12 <= hi + 6) m += 12;
        while (m > hi && m - 12 >= lo - 6) m -= 12;
        if (m < lo - 1 || m > hi + 1) continue;
        ctx.globalAlpha = Math.max(0.08, 1 - age / 2.5) * 0.9;
        ctx.fillStyle = '#4dd0e1';
        ctx.beginPath();
        ctx.arc(x(t), y(m), age < 0.12 ? 3.4 : 2.2, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalAlpha = 1;
    }

    // ── 카운트다운: 긴 묵음(5s+) 뒤 라인 시작 4초 전부터 4·3·2·1
    if (this.opts.countdown) {
      const next = pages.find(p => p.start > now + 0.05 && p.hasNotes);
      if (next) {
        const prevEnd = pages.reduce(
          (acc, p) => (p.end <= next.start + 0.01 && p.end > acc ? p.end : acc), 0);
        const remain = next.start - now;
        if (remain <= 4 && next.start - prevEnd >= 5 && now >= prevEnd) {
          const num = Math.max(1, Math.ceil(remain));
          // 숫자를 라인 시작 시각 위치에 그린다 (창 밖이면 가장자리에 고정)
          const nx = Math.max(Math.min(playheadX + 30, cw - 30), Math.min(cw - 30, x(next.start)));
          ctx.font = `bold ${Math.round(staffH * 0.5)}px system-ui, sans-serif`;
          ctx.fillStyle = colors.accent;
          ctx.globalAlpha = 0.9;
          ctx.fillText(String(num), nx, padTop + staffH / 2);
          ctx.globalAlpha = 1;
        }
      }
    }

    // ── 라벨 충돌 회피 — 이웃 간 최소 간격만 유지한다.
    // 예전엔 화면 밖 라벨을 뷰포트 가장자리로 되밀어서, 스크롤 모드에서 아직 안 들어온/
    // 이미 지나간 글자들이 좌우 구석에 멈춰 쌓이는 잔상이 생겼다 — 화면 밖 라벨은
    // 자연 위치 그대로 두고 캔버스 클리핑에 맡긴다.
    const placeRow = (items: { cx: number; w: number }[]): number[] => {
      const gap = 3;
      const xs = items.map(it => it.cx);
      for (let i = 1; i < xs.length; i++) {
        const minCx = xs[i - 1] + items[i - 1].w / 2 + gap + items[i].w / 2;
        if (xs[i] < minCx) xs[i] = minCx;
      }
      return xs;
    };

    // ── 노트 아래 가사: 창 안의 글자 토큰을 시간 위치 + 충돌 회피로
    let ty = padTop + staffH;
    ctx.font = `bold ${lyricPx}px system-ui, sans-serif`;
    const visWords = words.filter(w => w.end > t0 - edgePad && w.start < t0 + W + edgePad);
    if (visWords.length > 0) {
      // 글자를 노트 시작 지점에서 시작하도록 배치 (충돌 회피는 중심 좌표 기준)
      const items = visWords.map(w => {
        const width = ctx.measureText(w.word).width;
        return { w, cx: x(w.start) + width / 2, width };
      });
      const xs = placeRow(items.map(it => ({ cx: it.cx, w: it.width })));
      items.forEach((it, i) => {
        let color = it.w.start <= now ? colors.accent : colors.text;
        // 디버그: 정렬 신뢰도(기하평균 확률, 로그 버킷) — 패널과 동일 색 (confBucketColor)
        if (this.opts.showConfidence && it.w.confidence != null) {
          color = confBucketColor(it.w.confidence);
        }
        ctx.fillStyle = color;
        ctx.fillText(it.w.word, xs[i], ty + lyricH * 0.55);
      });
    }
    ty += lyricH;

    // ── 발음 폴백 줄 (음절 타이밍이 아예 없는 곡): 현재 라인 발음 그라데이션
    const page = this.index >= 0 ? pages[Math.min(this.index, pages.length - 1)] : undefined;
    if (hasPronRow) {
      if (page && resolvedPronunciation(page.line, this.opts.pronScript)) {
        this.renderPronFallback(ctx, page, now, cw, ty + pronRowH * 0.5, colors, pronPx);
      }
      ty += pronRowH;
    }

    // ── 번역: 현재 라인 번역을 하단 중앙에 — 길면 가로 압축(찌그러짐) 대신 폰트 축소
    if (hasTr && page?.line.translation) {
      const tr = page.line.translation;
      const maxW = cw - 16;
      ctx.font = `${trPx}px system-ui, sans-serif`;
      const tw = ctx.measureText(tr).width;
      if (tw > maxW) {
        ctx.font = `${Math.max(9, Math.floor(trPx * maxW / tw))}px system-ui, sans-serif`;
      }
      ctx.fillStyle = colors.dim;
      ctx.fillText(tr, cw / 2, ty + trH * 0.5, maxW);
    }

    // ── 곡 키·BPM 라벨 (좌상단) — 서버 멜로디 분석이 추정한 키
    if (this.songKey || this.tempo) {
      const parts: string[] = [];
      if (this.songKey) parts.push(t('pip.songKeyLabel', [this.songKey.name]));
      if (this.tempo) parts.push(`${Math.round(this.tempo.bpm)}BPM`);
      ctx.font = `${Math.max(9, Math.round(10 * fs))}px ui-monospace, monospace`;
      ctx.textAlign = 'left';
      ctx.fillStyle = colors.dim;
      ctx.globalAlpha = 0.85;
      // 레인이 창 상단에 붙어 있으면(영상·스테이지 없음) 좌상단 미니 버튼과 겹치지
      // 않게 라벨을 오른쪽으로 민다
      const labelX = canvas.getBoundingClientRect().top < 34 ? 62 : 4;
      ctx.fillText(parts.join(' · '), labelX, padTop + 7);
      ctx.globalAlpha = 1;
      ctx.textAlign = 'center';
    }

    // ── 디버그 전경 레이어: 보정된 라인의 원본 타이밍 고스트 + 곡 전체 confidence 헤더
    if (this.opts.showConfidence) {
      this.renderDebugOverlay(ctx, pages, t0, W, x, cw, padTop);
    }

    // ── 플레이헤드 (page 모드: 왼→오 이동, scroll 모드: 좌측 28% 고정)
    ctx.fillStyle = colors.dim;
    ctx.fillRect(playheadX - 0.75, padTop, 1.5, staffH);
    ctx.fillStyle = colors.accent;
    ctx.beginPath();
    ctx.moveTo(playheadX - 4, padTop);
    ctx.lineTo(playheadX + 4, padTop);
    ctx.lineTo(playheadX, padTop + 6);
    ctx.closePath();
    ctx.fill();
  }

  /**
   * 디버그 배경 레이어 — 정렬 파이프라인의 "재료"를 오선 뒤에 깐다:
   * 하단 스트립 = VAD 가창(초록)/간주·무성(회색), 그 위 보라 밴드 = star 흡수(가사 밖 가창),
   * 파란 곡선 = 음정 모델(RMVPE/FCPE) RAW f0 (노트 양자화 전 원본, null=무성에서 끊김).
   */
  private renderDebugUnderlay(
    ctx: CanvasRenderingContext2D,
    t0: number, W: number,
    x: (t: number) => number, y: (midi: number) => number,
    lo: number, hi: number, cw: number, padTop: number, staffH: number,
  ): void {
    const meta = this.debugMeta;
    if (!meta) return;
    const clampX = (v: number) => Math.max(0, Math.min(cw, v));
    const stripY = padTop + staffH - 4;
    const regions = meta.vad_regions ?? [];
    // 간주(리전 사이 갭 2s+)를 회색으로 — interpolation이 간주를 예약하지 않는 문제를 눈으로 본다
    for (let i = 0; i <= regions.length; i++) {
      const gapS = i === 0 ? 0 : regions[i - 1][1];
      const gapE = i === regions.length ? t0 + W : regions[i][0];
      if (gapE - gapS >= 2.0 && gapE > t0 && gapS < t0 + W) {
        ctx.fillStyle = 'rgba(134, 142, 150, 0.30)';
        ctx.fillRect(clampX(x(gapS)), stripY, clampX(x(gapE)) - clampX(x(gapS)), 4);
      }
    }
    for (const [s, e] of regions) {
      if (e <= t0 || s >= t0 + W) continue;
      ctx.fillStyle = 'rgba(81, 207, 102, 0.35)';
      ctx.fillRect(clampX(x(s)), stripY, Math.max(1, clampX(x(e)) - clampX(x(s))), 4);
    }
    for (const [s, e] of meta.star_spans ?? []) {
      if (e <= t0 || s >= t0 + W) continue;
      ctx.fillStyle = 'rgba(218, 119, 242, 0.5)';
      ctx.fillRect(clampX(x(s)), stripY - 3, Math.max(1, clampX(x(e)) - clampX(x(s))), 2);
    }
    this.renderF0Curve(ctx, t0, W, x, y, lo, hi);
  }

  /** 음정 모델 RAW f0 곡선 — 디버그 언더레이의 일부지만, 설정(pitchF0Curve)으로
   *  디버그 모드 없이도 단독 표시된다 */
  private renderF0Curve(
    ctx: CanvasRenderingContext2D,
    t0: number, W: number,
    x: (t: number) => number, y: (midi: number) => number,
    lo: number, hi: number,
  ): void {
    const f0 = this.debugMeta?.f0_curve;
    if (!f0 || f0.dt <= 0 || f0.midi.length === 0) return;
    // f0 곡선 전용 페이더(f0Opacity) — 노트 바(lineOpacity)와 분리(운영자 요청:
    // "음정 보는 선"만 따로 밝게). 1 초과면 알파 상한(0.98)에 두께로 마저 보탠다.
    ctx.strokeStyle = `rgba(77, 171, 247, ${Math.min(0.98, 0.65 * this.opts.f0Opacity)})`;
    ctx.lineWidth = this.opts.f0Opacity > 1 ? 1.6 : 1;
    ctx.beginPath();
    let pen = false;
    const i0 = Math.max(0, Math.floor((t0 - f0.t0) / f0.dt));
    const i1 = Math.min(f0.midi.length - 1, Math.ceil((t0 + W - f0.t0) / f0.dt));
    for (let i = i0; i <= i1; i++) {
      const m = f0.midi[i];
      if (m == null) { pen = false; continue; }
      const px = x(f0.t0 + i * f0.dt);
      const py = y(Math.max(lo, Math.min(hi, m)));
      if (pen) ctx.lineTo(px, py);
      else { ctx.moveTo(px, py); pen = true; }
    }
    ctx.stroke();
  }

  /**
   * "라인 집중 뷰" — 지금 재생 중인 라인 하나만 확대해, CTC가 원문 글자·heard(실제로
   * 들은 것)·발음 음절에 준 시각을 시간축 위 세 줄로 찍는다. 여러 줄을 한꺼번에 겹쳐
   * 보여주던 예전 방식은 고스트와 뭉쳐 헷갈림의 주범이었다(사용자 확인 후 폐기).
   *
   * 위 줄이 원문 글자, 가운데 줄이 heard(CTC 전사 — 정답이 아니라 모델이 실제로 들은
   * 것), 아래 줄이 발음 음절(있는 곡만)이다. 각 세그먼트는 시작에 세로 틱 + 지속만큼
   * 가로 막대로 그리고, 틱 위에 그 글자를 적는다.
   *
   * 원문·발음 두 채널은 **모양**이 타이밍 구조를(길이가 0이면 막대 없이 틱만 남아 여러
   * 글자가 한 지점에 겹친 "뭉침"이 즉시 보인다), **색**이 정렬 신뢰도를 나타낸다(패널·
   * 노트 가사와 같은 3색 규칙). heard 줄은 신뢰도가 없어 항상 흐린 회색 고정색 —
   * 원문 줄과 나란히 놓고 "CTC가 실제로 이렇게 들었다"를 대조하는 용도다. 심판이 기본
   * 발음 후보를 갈아치운 라인은 맨 아래에 ⚖ 뱃지로 판정 근거(기본→선택, gain)를 적는다.
   */
  private renderTimingLanes(
    ctx: CanvasRenderingContext2D,
    pages: PitchLine[],
    t0: number, W: number,
    x: (t: number) => number,
    padTop: number,
  ): void {
    const current = this.index >= 0 ? pages[Math.min(this.index, pages.length - 1)] : undefined;
    if (!current) return;
    const t1 = t0 + W;
    if (current.end < t0 || current.start > t1) return; // 재생 중 라인이 창 밖이면 그릴 게 없다

    const wordY = padTop + 40;  // 보정 라벨 3행 로테이션(padTop+6 ~ +22) 아래
    const heardY = wordY + 13;
    const pronY = heardY + 13;
    // 글자 라벨은 촘촘하면 서로 겹쳐 못 읽는다 — 마지막으로 그린 x에서 이만큼 떨어져야
    // 새로 그린다. 뭉친 구간에서는 라벨이 하나만 남아 그 자체로 뭉침 신호가 된다.
    const LABEL_GAP = 9;
    ctx.save();
    ctx.lineWidth = 1;
    ctx.setLineDash([]);
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    let drewWord = false;
    let drewHeard = false;
    let drewPron = false;
    let lastWordLabelX = -Infinity;
    let lastHeardLabelX = -Infinity;
    let lastPronLabelX = -Infinity;

    // ── 원문 글자 타이밍 바 (기존 conf 색 등급 그대로 유지)
    ctx.font = '10px ui-monospace, monospace';
    for (const w of current.line.words ?? []) {
      if (w.end < t0 || w.start > t1) continue;
      const xs = x(w.start);
      const xe = x(w.end);
      // 1px 미만은 화면상 점 — 길이 0과 구분해도 의미가 없으므로 같이 취급한다
      const zero = xe - xs < 1;
      const color = w.confidence != null ? confBucketColor(w.confidence) : TIMING_WORD;
      ctx.strokeStyle = color;
      ctx.beginPath();
      ctx.moveTo(xs, wordY - 3);
      ctx.lineTo(xs, wordY + 3);
      if (!zero) {
        ctx.moveTo(xs, wordY);
        ctx.lineTo(xe, wordY);
      }
      ctx.stroke();
      // 이 시각에 CTC가 붙인 실제 원문 글자 — 레인 바로 위에 작게
      if (w.word && xs - lastWordLabelX >= LABEL_GAP) {
        ctx.fillStyle = color;
        ctx.fillText(w.word, xs + 1, wordY - 5);
        lastWordLabelX = xs;
      }
      drewWord = true;
    }

    // ── heard 레인: CTC가 실제로 들은 글자. heard_spans(글자별 시각)가 정본이고,
    // 없으면 heard 문자열을 라인 구간에 균등 배치해 대략 위치만이라도 보여준다
    const dbg = current.line.debug;
    const spans: [string, number][] | undefined =
      dbg?.heard_spans && dbg.heard_spans.length > 0 ? dbg.heard_spans
        : dbg?.heard ? evenlyPlaceHeard(dbg.heard, current.start, current.end)
        : undefined;
    if (spans) {
      ctx.font = '9px ui-monospace, monospace';
      ctx.strokeStyle = TIMING_HEARD;
      ctx.fillStyle = TIMING_HEARD;
      ctx.globalAlpha = 0.85;
      for (const [ch, t] of spans) {
        if (t < t0 || t > t1) continue;
        const hx = x(t);
        ctx.beginPath();
        ctx.moveTo(hx, heardY - 2.5);
        ctx.lineTo(hx, heardY + 2.5);
        ctx.stroke();
        if (ch && hx - lastHeardLabelX >= LABEL_GAP) {
          ctx.fillText(ch, hx + 1, heardY - 4);
          lastHeardLabelX = hx;
        }
        drewHeard = true;
      }
      ctx.globalAlpha = 1;
    }

    // ── 발음 음절 레인 (기존 그대로) — 색은 음절 신뢰도, 지속은 막대
    ctx.font = '10px ui-monospace, monospace';
    for (const s of resolvedPronSegments(current.line, this.opts.pronScript) ?? []) {
      if (s.end < t0 || s.start > t1) continue;
      const sx = x(s.start);
      const se = x(s.end);
      const pcolor = s.confidence != null ? confBucketColor(s.confidence) : TIMING_PRON;
      const pzero = se - sx < 1;
      ctx.strokeStyle = pcolor;
      ctx.beginPath();
      ctx.moveTo(sx, pronY - 3);
      ctx.lineTo(sx, pronY + 3);
      if (!pzero) {
        ctx.moveTo(sx, pronY);
        ctx.lineTo(se, pronY);
      }
      ctx.stroke();
      // 같은 시각의 전사 발음 음절
      if (s.text && sx - lastPronLabelX >= LABEL_GAP) {
        ctx.fillStyle = pcolor;
        ctx.fillText(s.text, sx + 1, pronY - 5);
        lastPronLabelX = sx;
      }
      drewPron = true;
    }

    if (drewWord || drewHeard || drewPron) {
      ctx.font = '9px ui-monospace, monospace';
      ctx.textBaseline = 'middle';
      ctx.globalAlpha = 0.9;
      if (drewWord) { ctx.fillStyle = TIMING_WORD; ctx.fillText(t('pip.debug.laneOriginal'), 4, wordY); }
      if (drewHeard) { ctx.fillStyle = TIMING_HEARD; ctx.fillText(t('pip.debug.laneHeard'), 4, heardY); }
      if (drewPron) { ctx.fillStyle = TIMING_PRON; ctx.fillText(t('pip.debug.lanePron'), 4, pronY); }
      ctx.globalAlpha = 1;
    }

    // ── 심판 뱃지: 기본 발음 후보 대신 다른 후보를 골랐을 때만 표시
    const ref = dbg?.referee;
    if (ref?.chosen && ref.chosen !== ref.default) {
      const gain = ref.gain ?? 0;
      const sign = gain >= 0 ? '+' : '';
      const badge = `⚖ ${ref.default ?? '?'}→${ref.chosen} (${sign}${gain.toFixed(2)})`;
      ctx.font = 'bold 10px ui-monospace, monospace';
      ctx.textBaseline = 'alphabetic';
      ctx.fillStyle = '#da77f2';
      ctx.fillText(badge, Math.max(2, x(Math.max(current.start, t0)) + 2), pronY + 14);
    }

    ctx.restore();
  }

  /**
   * 디버그 전경 레이어 — 세이프가드가 고친 라인의 보정 규칙 라벨
   * (규칙별 색: stretch/repeat=빨강, pull=파랑, tail=주황, snap=보라),
   * 재생 중 라인의 타이밍 레인, 우상단에 곡 전체 confidence와 정렬 텍스트 헤더.
   */
  private renderDebugOverlay(
    ctx: CanvasRenderingContext2D,
    pages: PitchLine[],
    t0: number, W: number,
    x: (t: number) => number,
    cw: number, padTop: number,
  ): void {
    ctx.save();
    ctx.textBaseline = 'middle';
    // 세이프가드가 고친 라인의 보정 규칙 라벨만 남긴다 — 보정 전 원본 타이밍을 점선
    // 고스트로 겹쳐 그리던 예전 방식은 실제(보정 후) 위치와 헷갈려 혼란의 주범이었다
    // (사용자 확인 후 제거). "무엇이 바뀌었는지"는 이 라벨로 충분하고, "어떻게
    // 바뀌었는지"는 재생 중 라인에 한해 renderTimingLanes의 원문/heard/발음 레인이 보여준다.
    let row = 0;
    for (const p of pages) {
      const dbg = p.line.debug;
      if (!dbg?.fixes || dbg.fixes.length === 0) continue;
      const ns = p.line.time ?? p.start;
      const ne = p.line.endTime ?? p.end;
      if (ne < t0 || ns > t0 + W) continue;
      const color = FIX_COLORS[dbg.fixes[0]] ?? '#868e96';
      const gy = padTop + 6 + (row % 3) * 8; // 인접 라인 겹침 방지용 3행 로테이션
      row++;
      ctx.font = '9px ui-monospace, monospace';
      ctx.fillStyle = color;
      ctx.textAlign = 'left';
      ctx.fillText(dbg.fixes.join('·'), Math.max(2, x(Math.max(ns, t0)) + 2), gy);
    }
    this.renderTimingLanes(ctx, pages, t0, W, x, padTop);
    if (this.confStats) {
      // 사람이 읽는 등급 분포 — 글자 색과 같은 3색으로 "정렬 좋음 87% 보통 10% 낮음 3%"
      const s = this.confStats;
      const at = this.debugMeta?.alignment_text;
      const parts: { text: string; color: string }[] = [
        { text: t('pip.debug.alignPrefix'), color: '#868e96' },
        { text: t('pip.debug.gradeOk', [String(Math.round(s.ok * 100))]), color: CONF_COLOR_OK },
        { text: t('pip.debug.gradeMid', [String(Math.round(s.mid * 100))]), color: CONF_COLOR_MID },
        { text: t('pip.debug.gradeLow', [String(Math.round(s.low * 100))]), color: CONF_COLOR_LOW },
      ];
      if (at) {
        // 어떤 텍스트로 CTC 전사했는지: 독음 = 한국어 발음표기, 원문 = 일/영 등 원어 그대로
        parts.push({
          text: t('pip.debug.alignmentTextPrefix', [
            at === 'pronunciation' ? t('overlay.debug.alignmentPronunciation') : t('overlay.debug.alignmentOriginal'),
          ]),
          color: '#868e96',
        });
      }
      const sc = this.debugMeta?.caption_scaffold;
      if (sc?.applied) {
        // 자막 골격이 타이밍을 교체했다 — 고스트 라벨(scaffold)과 같은 노랑으로 묶는다
        const src = sc.sources ?? {};
        parts.push({
          text: t('pip.debug.scaffoldApplied', [
            String(sc.moved ?? 0), String(src.caption ?? 0), String(src.interp ?? 0), String(src.kept ?? 0),
          ]),
          color: FIX_COLORS.scaffold,
        });
      } else if (sc?.skipped && sc.skipped !== 'not_collapsed') {
        // 붕괴인데 골격을 못 쓴 이유(자막 없음·매칭 미달)는 진단 가치가 있어 표시한다.
        // not_collapsed(정상 곡)는 소음이라 숨긴다.
        parts.push({ text: t('pip.debug.scaffoldSkipped', [sc.skipped]), color: '#868e96' });
      }
      ctx.font = '10px ui-monospace, monospace';
      ctx.textAlign = 'left';
      ctx.globalAlpha = 0.9;
      let px = cw - 4 - parts.reduce((a, p) => a + ctx.measureText(p.text).width, 0);
      for (const p of parts) {
        ctx.fillStyle = p.color;
        ctx.fillText(p.text, px, padTop + 7);
        px += ctx.measureText(p.text).width;
      }
      ctx.globalAlpha = 1;
    }
    ctx.restore();
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
  }

  /** 발음 폴백 (음절 타이밍 없는 곡) — 현재 라인 발음을 가운데 정렬 + 진행률 그라데이션 */
  private renderPronFallback(
    ctx: CanvasRenderingContext2D,
    page: PitchLine,
    now: number,
    cw: number,
    py: number,
    colors: PitchColors,
    fontPx: number,
  ): void {
    const pron = resolvedPronunciation(page.line, this.opts.pronScript) ?? '';
    if (!pron) return;
    ctx.font = `${fontPx}px system-ui, sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    const maxW = cw - 16;
    ctx.fillStyle = colors.dim;
    ctx.fillText(pron, cw / 2, py, maxW);
    const span = Math.max(0.001, page.end - page.start);
    const sungRatio = Math.max(0, Math.min(1, (now - page.start) / span));
    if (sungRatio <= 0) return;
    const textW = Math.min(ctx.measureText(pron).width, maxW);
    const x0 = cw / 2 - textW / 2;
    ctx.save();
    ctx.beginPath();
    ctx.rect(x0, py - fontPx, textW * sungRatio, fontPx * 2);
    ctx.clip();
    ctx.fillStyle = colors.accent;
    ctx.fillText(pron, cw / 2, py, maxW);
    ctx.restore();
  }
}

/**
 * 라인 배열에서 레인 데이터를 평탄화한다.
 * - pages: 라인 인덱스와 1:1 (카운트다운·현재 라인 번역/발음 폴백용)
 * - notes: 전 곡 노트, 각 노트에 대응 발음 음절(pron)을 최대 겹침 기준으로 부착
 * - words: 전 곡 글자 토큰
 * - lo/hi: 곡 전체 고정 세로 스케일 (최소 14반음)
 */
function collectPitchData(lines: LyricLine[], script: PronScript): PitchData {
  const pages: PitchLine[] = [];
  const notes: PitchNote[] = [];
  const words: PitchWord[] = [];
  let lo = Infinity;
  let hi = -Infinity;

  lines.forEach((line, i) => {
    const lineNotes: PitchNote[] = [];
    if (line.words) {
      for (const word of line.words) {
        words.push({ word: word.word, start: word.start, end: word.end, confidence: word.confidence });
        if (!word.notes) continue;
        for (const n of word.notes) lineNotes.push({ midi: n.midi, start: n.start, end: n.end });
      }
    }
    if (line.notes) {
      for (const n of line.notes) lineNotes.push({ midi: n.midi, start: n.start, end: n.end });
    }
    lineNotes.sort((a, b) => a.start - b.start);

    // 발음 음절을 최대 겹침 노트에 부착 — 계이름처럼 노트에 직접 그린다
    const pronSegs = resolvedPronSegments(line, script);
    if (pronSegs) {
      for (const seg of pronSegs) {
        let best: PitchNote | null = null;
        let bestOv = 0;
        for (const n of lineNotes) {
          const ov = Math.min(n.end, seg.end) - Math.max(n.start, seg.start);
          if (ov > bestOv) {
            bestOv = ov;
            best = n;
          }
        }
        if (best) best.pron = best.pron ? best.pron + seg.text : seg.text;
      }
    }

    let start = line.time ?? lineNotes[0]?.start ?? 0;
    let end = line.endTime ?? lines[i + 1]?.time ?? start + 4;
    for (const n of lineNotes) {
      start = Math.min(start, n.start);
      end = Math.max(end, n.end);
      lo = Math.min(lo, n.midi);
      hi = Math.max(hi, n.midi);
    }
    pages.push({ line, start, end: Math.max(end, start + 0.5), hasNotes: lineNotes.length > 0 });
    notes.push(...lineNotes);
  });

  notes.sort((a, b) => a.start - b.start);
  words.sort((a, b) => a.start - b.start);
  if (!Number.isFinite(lo)) {
    lo = 57;
    hi = 71;
  }
  lo -= 1;
  hi += 1;
  while (hi - lo < 14) {
    lo -= 1;
    hi += 1;
  }
  return { pages, notes, words, lo, hi };
}

/**
 * 레인 색은 CSS 변수에서 읽는다 — 테마(다크/라이트)를 캔버스가 따라가는 유일한 경로다.
 * 대상 엘리먼트는 호출부가 정한다: PiP는 문서 루트(:root.ey-light), 메인 패널은
 * 패널 엘리먼트(.ey-panel.ey-light) — 변수가 걸리는 자리가 서로 다르기 때문이다.
 */
function readPitchColors(win: Window, source: Element): PitchColors {
  const style = win.getComputedStyle(source);
  const pick = (name: string, fallback: string) => style.getPropertyValue(name).trim() || fallback;
  return {
    faint: pick('--ey-text-faint', 'rgba(241, 241, 242, 0.34)'),
    dim: pick('--ey-text-dim', 'rgba(241, 241, 242, 0.58)'),
    accent: pick('--ey-accent', '#ffb02e'),
    text: pick('--ey-text', '#f1f1f2'),
  };
}
