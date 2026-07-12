import type { LyricLine, SongTempo } from '../types';

export interface MelodyNote {
  midi: number;
  start: number;
  end: number;
}

/** PiP 레인과 동일한 소스(word.notes + line.notes)에서 멜로디 노트를 평탄화 */
export function collectMelodyNotes(lines: LyricLine[]): MelodyNote[] {
  const notes: MelodyNote[] = [];
  for (const line of lines) {
    if (line.words) {
      for (const w of line.words) {
        if (!w.notes) continue;
        for (const n of w.notes) notes.push({ midi: n.midi, start: n.start, end: n.end });
      }
    }
    if (line.notes) {
      for (const n of line.notes) notes.push({ midi: n.midi, start: n.start, end: n.end });
    }
  }
  notes.sort((a, b) => a.start - b.start);
  return notes;
}

export interface KaraokeAudioConfig {
  melody: boolean;
  melodyVolume: number; // 0..1
  metronome: boolean;
  metronomeVolume: number; // 0..1
  /** 메트로놈 배속 — 0.5(2분음표)·1(4분음표)·2(8분음표) */
  metronomeRate: number;
  /** 마디 시작 박(0~3) — 강세 위치를 이동 (곡 다운비트가 첫 비트와 다를 때) */
  metronomeBeat: number;
  /** AudioContext.setSinkId용 출력 기기 id — '' = 기본 출력 */
  sinkId: string;
}

/** 스케줄 선행 창(초, 벽시계 기준) — 틱 간격보다 넉넉해야 이음새 없이 이어진다 */
const LOOKAHEAD = 0.45;
const TICK_MS = 120;
/** 삼각파 원음이 크므로 슬라이더 100%가 이 값이 되도록 낮춘다 */
const MELODY_GAIN_SCALE = 0.35;

function clamp01(v: number): number {
  return Math.min(1, Math.max(0, v));
}

/**
 * 가라오케 멜로디 재생 + 메트로놈 — WebAudio 신디사이즈.
 * 비디오 currentTime을 기준 타임라인으로 삼아 선행 창(LOOKAHEAD)만큼 미리 스케줄하고,
 * 시크/일시정지가 감지되면 스케줄된 소리를 전부 끊고 현재 위치부터 다시 잇는다.
 * 별도 AudioContext를 쓰므로 setSinkId로 영상과 다른 출력 기기로 보낼 수 있다.
 */
export class KaraokeAudio {
  private ctx: AudioContext | null = null;
  private melodyGain: GainNode | null = null;
  private metroGain: GainNode | null = null;
  private notes: MelodyNote[] = [];
  private tempo: SongTempo | null = null;
  private melodyOn = false;
  private metroOn = false;
  private melodyVol = 0.5;
  private metroVol = 0.5;
  private metroRate = 1;
  private metroBeat = 0;
  private sinkId = '';
  /** 가라오케 창(PiP)이 열려 있을 때만 소리 낸다 */
  private active = false;
  private timer: number | undefined;
  private getVideo: () => HTMLVideoElement | null;
  /** 비디오 타임라인 기준으로 여기까지 스케줄 완료 */
  private scheduledUntil = 0;
  private lastT = -1;
  private lastWall = 0;
  private live: { node: AudioScheduledSourceNode; gain: GainNode; until: number }[] = [];

  constructor(getVideo: () => HTMLVideoElement | null) {
    this.getVideo = getVideo;
  }

  setNotes(notes: MelodyNote[]): void {
    this.notes = notes;
    this.resync();
  }

  setTempo(tempo: SongTempo | null): void {
    this.tempo = tempo && tempo.bpm > 0 ? tempo : null;
    this.resync();
  }

  configure(cfg: KaraokeAudioConfig): void {
    this.melodyOn = cfg.melody;
    this.metroOn = cfg.metronome;
    this.melodyVol = clamp01(cfg.melodyVolume);
    this.metroVol = clamp01(cfg.metronomeVolume);
    if (cfg.metronomeRate !== this.metroRate || cfg.metronomeBeat !== this.metroBeat) {
      this.metroRate = cfg.metronomeRate === 0.5 || cfg.metronomeRate === 2 ? cfg.metronomeRate : 1;
      this.metroBeat = Math.min(3, Math.max(0, Math.round(cfg.metronomeBeat)));
      this.resync(); // 이미 스케줄된 틱을 끊고 새 배속/강세로 다시 잇는다
    }
    if (this.melodyGain) this.melodyGain.gain.value = this.melodyVol * MELODY_GAIN_SCALE;
    if (this.metroGain) this.metroGain.gain.value = this.metroVol;
    if (cfg.sinkId !== this.sinkId) {
      this.sinkId = cfg.sinkId;
      void this.applySink();
    }
    this.syncRunning();
  }

  setActive(active: boolean): void {
    this.active = active;
    this.syncRunning();
  }

  /** 시크 등으로 타임라인이 끊겼을 때 스케줄을 현재 위치부터 다시 잇는다 */
  resync(): void {
    this.cancelScheduled();
    this.lastT = -1;
  }

  dispose(): void {
    this.setActive(false);
    void this.ctx?.close().catch(() => { /* 이미 닫힘 */ });
    this.ctx = null;
    this.melodyGain = null;
    this.metroGain = null;
  }

  private syncRunning(): void {
    const shouldRun = this.active && (this.melodyOn || this.metroOn);
    if (shouldRun && this.timer === undefined) {
      const ctx = this.ensureCtx();
      void ctx.resume().catch(() => { /* 사용자 제스처 전 — 다음 상호작용에서 재개됨 */ });
      this.resync();
      this.timer = window.setInterval(() => this.pump(), TICK_MS);
    } else if (!shouldRun && this.timer !== undefined) {
      clearInterval(this.timer);
      this.timer = undefined;
      this.cancelScheduled();
      void this.ctx?.suspend().catch(() => { /* 이미 정지 */ });
    }
  }

  private ensureCtx(): AudioContext {
    if (this.ctx) return this.ctx;
    this.ctx = new AudioContext();
    this.melodyGain = this.ctx.createGain();
    this.melodyGain.gain.value = this.melodyVol * MELODY_GAIN_SCALE;
    this.melodyGain.connect(this.ctx.destination);
    this.metroGain = this.ctx.createGain();
    this.metroGain.gain.value = this.metroVol;
    this.metroGain.connect(this.ctx.destination);
    void this.applySink();
    return this.ctx;
  }

  private async applySink(): Promise<void> {
    const ctx = this.ctx as (AudioContext & { setSinkId?: (id: string) => Promise<void> }) | null;
    if (!ctx?.setSinkId) return;
    try {
      await ctx.setSinkId(this.sinkId);
    } catch {
      /* 기기가 뽑혔거나 권한 없음 — 기본 출력 유지 */
    }
  }

  private pump(): void {
    const ctx = this.ctx;
    const video = this.getVideo();
    if (!ctx || !video) return;
    if (video.paused || video.seeking || video.ended) {
      this.resync();
      return;
    }
    const rate = video.playbackRate || 1;
    const t = video.currentTime;
    const wall = performance.now() / 1000;
    if (this.lastT >= 0) {
      // 시크·프레임 드랍 등 타임라인 불연속 감지
      const expected = this.lastT + (wall - this.lastWall) * rate;
      if (Math.abs(t - expected) > 0.35) {
        this.cancelScheduled();
        this.scheduledUntil = t;
      }
    } else {
      this.scheduledUntil = t;
    }
    this.lastT = t;
    this.lastWall = wall;

    const from = Math.max(this.scheduledUntil, t);
    const to = t + LOOKAHEAD * rate;
    if (to <= from) return;
    const toCtx = (vt: number) => ctx.currentTime + Math.max(0, (vt - t) / rate);
    if (this.melodyOn) this.scheduleMelody(ctx, from, to, toCtx, rate);
    if (this.metroOn) this.scheduleMetronome(ctx, from, to, toCtx);
    this.scheduledUntil = to;
    this.prune(ctx.currentTime);
  }

  private scheduleMelody(
    ctx: AudioContext,
    from: number,
    to: number,
    toCtx: (vt: number) => number,
    rate: number,
  ): void {
    if (!this.melodyGain) return;
    for (const n of this.notes) {
      if (n.start >= to) break;
      if (n.start < from) continue;
      const start = toCtx(n.start);
      const dur = Math.max(0.06, (n.end - n.start) / rate);
      const osc = ctx.createOscillator();
      osc.type = 'triangle';
      osc.frequency.value = 440 * Math.pow(2, (n.midi - 69) / 12);
      const g = ctx.createGain();
      const attack = Math.min(0.02, dur * 0.3);
      const release = Math.min(0.05, dur * 0.3);
      g.gain.setValueAtTime(0, start);
      g.gain.linearRampToValueAtTime(1, start + attack);
      g.gain.setValueAtTime(1, start + dur - release);
      g.gain.linearRampToValueAtTime(0.0001, start + dur);
      osc.connect(g).connect(this.melodyGain);
      osc.start(start);
      osc.stop(start + dur + 0.02);
      this.live.push({ node: osc, gain: g, until: start + dur + 0.02 });
    }
  }

  private scheduleMetronome(
    ctx: AudioContext,
    from: number,
    to: number,
    toCtx: (vt: number) => number,
  ): void {
    if (!this.metroGain || !this.tempo) return;
    // 배속: 0.5=2분음표(느린 곡 확인용), 2=8분음표(빠른 세분) — 틱 간격을 나눈다
    const tick = 60 / this.tempo.bpm / this.metroRate;
    const offset = this.tempo.beat_offset ?? 0;
    // 강세 주기는 배속과 무관하게 음악적 한 마디(4박)를 유지, 시작 박 선택만큼 이동
    const per = Math.max(1, Math.round(4 * this.metroRate));
    const shift = Math.round(this.metroBeat * this.metroRate);
    for (let k = Math.ceil((from - offset) / tick); ; k++) {
      const bt = offset + k * tick;
      if (bt >= to) break;
      if (bt < from) continue;
      const accent = (((k - shift) % per) + per) % per === 0;
      const start = toCtx(bt);
      const osc = ctx.createOscillator();
      osc.type = 'square';
      osc.frequency.value = accent ? 1760 : 1175;
      const g = ctx.createGain();
      g.gain.setValueAtTime(accent ? 0.5 : 0.28, start);
      g.gain.exponentialRampToValueAtTime(0.001, start + 0.06);
      osc.connect(g).connect(this.metroGain);
      osc.start(start);
      osc.stop(start + 0.08);
      this.live.push({ node: osc, gain: g, until: start + 0.08 });
    }
  }

  private cancelScheduled(): void {
    for (const { node, gain } of this.live) {
      try {
        gain.gain.cancelScheduledValues(0);
        gain.gain.value = 0;
        node.stop();
      } catch {
        /* 이미 정지된 노드 */
      }
    }
    this.live = [];
    if (this.ctx) this.scheduledUntil = 0;
  }

  private prune(ctxNow: number): void {
    if (this.live.length > 64) {
      this.live = this.live.filter(s => s.until > ctxNow);
    }
  }
}
