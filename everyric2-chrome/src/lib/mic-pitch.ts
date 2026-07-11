/** 마이크 피치 샘플 — at은 performance.now()/1000 (벽시계 초) */
export interface MicSample {
  at: number;
  midi: number;
}

const HISTORY_SEC = 4;
const SAMPLE_MS = 45;

/**
 * 마이크 입력 실시간 피치 검출 — 자기상관(ACF) 기반.
 * 가라오케 레인이 samples()를 읽어 사용자 음정 궤적을 그린다.
 * echo/noise 억제를 꺼서 노래 원음에 가까운 신호를 받는다.
 */
export class MicPitch {
  private ctx: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private analyser: AnalyserNode | null = null;
  private buf: Float32Array<ArrayBuffer> | null = null;
  private timer: number | undefined;
  private history: MicSample[] = [];
  private starting = false;
  private deviceId = '';

  isRunning(): boolean {
    return this.timer !== undefined || this.starting;
  }

  currentDeviceId(): string {
    return this.deviceId;
  }

  /** 마이크 권한을 요청하고 검출을 시작. 거부/실패 시 false. */
  async start(deviceId?: string): Promise<boolean> {
    if (this.isRunning()) return true;
    this.starting = true;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          deviceId: deviceId ? { exact: deviceId } : undefined,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
      });
      this.stream = stream;
      this.deviceId = deviceId ?? '';
      this.ctx = new AudioContext();
      const src = this.ctx.createMediaStreamSource(stream);
      this.analyser = this.ctx.createAnalyser();
      this.analyser.fftSize = 2048;
      src.connect(this.analyser);
      this.buf = new Float32Array(this.analyser.fftSize);
      this.timer = window.setInterval(() => this.sample(), SAMPLE_MS);
      return true;
    } catch {
      this.stop();
      return false;
    } finally {
      this.starting = false;
    }
  }

  stop(): void {
    if (this.timer !== undefined) {
      clearInterval(this.timer);
      this.timer = undefined;
    }
    this.stream?.getTracks().forEach(t => t.stop());
    this.stream = null;
    void this.ctx?.close().catch(() => { /* 이미 닫힘 */ });
    this.ctx = null;
    this.analyser = null;
    this.buf = null;
    this.history = [];
    this.deviceId = '';
  }

  samples(): MicSample[] {
    return this.history;
  }

  private sample(): void {
    if (!this.analyser || !this.buf || !this.ctx) return;
    this.analyser.getFloatTimeDomainData(this.buf);
    const freq = autoCorrelate(this.buf, this.ctx.sampleRate);
    const now = performance.now() / 1000;
    if (this.history.length > 0 && now - this.history[0].at > HISTORY_SEC) {
      this.history = this.history.filter(s => now - s.at < HISTORY_SEC);
    }
    if (freq > 0) {
      this.history.push({ at: now, midi: 69 + 12 * Math.log2(freq / 440) });
    }
  }
}

/**
 * 자기상관 피치 검출 — 사람 목소리 대역(70Hz~1kHz)만 탐색.
 * 검출 실패(무음·비주기 신호)면 -1.
 */
function autoCorrelate(buf: Float32Array, sampleRate: number): number {
  const size = buf.length;
  let energy = 0;
  for (let i = 0; i < size; i++) energy += buf[i] * buf[i];
  const rms = Math.sqrt(energy / size);
  if (rms < 0.015 || energy === 0) return -1;

  const minLag = Math.max(2, Math.floor(sampleRate / 1000));
  const maxLag = Math.min(size - 2, Math.floor(sampleRate / 70));
  if (maxLag <= minLag) return -1;

  const corr = new Float32Array(maxLag + 1);
  let bestLag = -1;
  let bestVal = 0;
  for (let lag = minLag; lag <= maxLag; lag++) {
    let sum = 0;
    for (let i = 0; i < size - lag; i++) sum += buf[i] * buf[i + lag];
    const val = sum / energy;
    corr[lag] = val;
    if (val > bestVal) {
      bestVal = val;
      bestLag = lag;
    }
  }
  if (bestLag < 0 || bestVal < 0.5) return -1;

  // 포물선 보간으로 서브샘플 정밀도 확보 (반음 미만 오차용)
  let lag = bestLag;
  if (bestLag > minLag && bestLag < maxLag) {
    const a = corr[bestLag - 1];
    const b = corr[bestLag];
    const c = corr[bestLag + 1];
    const denom = a - 2 * b + c;
    if (Math.abs(denom) > 1e-9) lag = bestLag + 0.5 * (a - c) / denom;
  }
  return sampleRate / lag;
}
