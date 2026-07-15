// Phát audio (MP3 base64) qua Web Audio API và trích biên độ âm thanh mỗi frame
// để điều khiển độ mở miệng (0..1) cho avatar — dùng cho lip-sync.

export type LevelListener = (level: number) => void;

function base64ToArrayBuffer(b64: string): ArrayBuffer {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes.buffer;
}

export class AudioLipSync {
  private ctx: AudioContext | null = null;
  private analyser: AnalyserNode | null = null;
  private data: Uint8Array | null = null;
  private raf = 0;
  private listeners = new Set<LevelListener>();
  private current: AudioBufferSourceNode | null = null;

  onLevel(cb: LevelListener): () => void {
    this.listeners.add(cb);
    return () => this.listeners.delete(cb);
  }

  private emit(level: number) {
    for (const cb of this.listeners) cb(level);
  }

  private ensureContext() {
    if (!this.ctx) {
      this.ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
      this.analyser = this.ctx.createAnalyser();
      this.analyser.fftSize = 512;
      this.data = new Uint8Array(this.analyser.frequencyBinCount);
    }
  }

  /** Gọi khi có tương tác người dùng để mở khoá AudioContext (chính sách autoplay). */
  async resume() {
    this.ensureContext();
    if (this.ctx && this.ctx.state === "suspended") await this.ctx.resume();
  }

  /** Phát 1 đoạn audio base64 MP3, trả về Promise hoàn tất khi phát xong. */
  async play(base64: string): Promise<void> {
    if (!base64) return;
    this.ensureContext();
    const ctx = this.ctx!;
    if (ctx.state === "suspended") await ctx.resume();

    const buffer = await ctx.decodeAudioData(base64ToArrayBuffer(base64));
    this.stop();

    const src = ctx.createBufferSource();
    src.buffer = buffer;
    src.connect(this.analyser!);
    this.analyser!.connect(ctx.destination);
    this.current = src;

    this.startLoop();

    return new Promise<void>((resolve) => {
      src.onended = () => {
        if (this.current === src) {
          this.stopLoop();
          this.emit(0);
          this.current = null;
        }
        resolve();
      };
      src.start();
    });
  }

  private startLoop() {
    cancelAnimationFrame(this.raf);
    const tick = () => {
      if (!this.analyser || !this.data) return;
      this.analyser.getByteFrequencyData(this.data as any);
      // RMS trên phổ tần → chuẩn hoá về 0..1, nhấn mạnh dải giọng nói.
      let sum = 0;
      const n = this.data.length;
      for (let i = 0; i < n; i++) sum += this.data[i] * this.data[i];
      const rms = Math.sqrt(sum / n) / 255;
      const level = Math.min(1, Math.max(0, rms * 2.2));
      this.emit(level);
      this.raf = requestAnimationFrame(tick);
    };
    this.raf = requestAnimationFrame(tick);
  }

  private stopLoop() {
    cancelAnimationFrame(this.raf);
    this.raf = 0;
  }

  stop() {
    if (this.current) {
      try {
        this.current.onended = null;
        this.current.stop();
      } catch {
        /* ignore */
      }
      this.current = null;
    }
    this.stopLoop();
    this.emit(0);
  }
}
