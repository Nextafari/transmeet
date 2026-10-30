/* AudioWorklet processor: downsamples tab audio to 16 kHz mono int16 PCM
 * and posts fixed-duration frames to the offscreen document.
 *
 * Runs in AudioWorkletGlobalScope (no DOM, no chrome APIs). The graph must
 * stay connected to the destination (via a zero-gain node) so process()
 * keeps being pulled.
 */

class PcmTapProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._targetRate = 16000;
    this._frameSamples = 1280; // 80 ms at 16 kHz
    this._carry = new Float32Array(0);
    this._carryPos = 0;
    this._pending = [];
    this._pendingCount = 0;
    this._started = false;

    this.port.onmessage = (event) => {
      const data = event.data || {};
      if (data.type === "config") {
        if (Number.isFinite(data.targetRate) && data.targetRate > 0) {
          this._targetRate = Math.floor(data.targetRate);
        }
        if (Number.isFinite(data.frameSamples) && data.frameSamples > 0) {
          this._frameSamples = Math.floor(data.frameSamples);
        }
        this._started = true;
      } else if (data.type === "stop") {
        this._started = false;
        this._carry = new Float32Array(0);
        this._carryPos = 0;
        this._pending = [];
        this._pendingCount = 0;
      }
    };
  }

  _pushResampled(samples) {
    for (let i = 0; i < samples.length; i += 1) {
      this._pending.push(samples[i]);
    }
    this._pendingCount += samples.length;

    while (this._pendingCount >= this._frameSamples) {
      const frame = this._pending.splice(0, this._frameSamples);
      this._pendingCount -= this._frameSamples;
      const pcm = new Int16Array(frame.length);
      for (let i = 0; i < frame.length; i += 1) {
        const clamped = Math.max(-1, Math.min(1, frame[i]));
        pcm[i] = Math.round(clamped * 32767);
      }
      this.port.postMessage({ type: "pcm", buffer: pcm.buffer }, [pcm.buffer]);
    }
  }

  _resample(block) {
    const step = sampleRate / this._targetRate;
    let input = block;
    if (this._carry.length > 0) {
      input = new Float32Array(this._carry.length + block.length);
      input.set(this._carry, 0);
      input.set(block, this._carry.length);
    }

    let pos = this._carryPos;
    const out = [];
    for (;;) {
      const index = Math.floor(pos);
      const frac = pos - index;
      if (index + 1 >= input.length) {
        this._carry = input.slice(index);
        this._carryPos = frac;
        break;
      }
      out.push(input[index] * (1 - frac) + input[index + 1] * frac);
      pos += step;
    }
    if (out.length > 0) {
      this._pushResampled(out);
    }
  }

  process(inputs) {
    const input = inputs && inputs[0];
    if (this._started && input && input.length > 0 && input[0].length > 0) {
      const frames = input[0].length;
      const mono = new Float32Array(frames);
      for (let ch = 0; ch < input.length; ch += 1) {
        const data = input[ch];
        for (let i = 0; i < frames; i += 1) {
          mono[i] += data[i];
        }
      }
      const scale = 1 / input.length;
      for (let i = 0; i < frames; i += 1) {
        mono[i] *= scale;
      }
      this._resample(mono);
    }
    return true;
  }
}

registerProcessor("pcm-tap", PcmTapProcessor);
