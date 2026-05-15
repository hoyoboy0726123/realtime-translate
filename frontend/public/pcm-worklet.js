// AudioWorklet processor: downsamples the microphone stream to 16 kHz mono
// PCM16 and posts ~80 ms chunks back to the main thread for WebSocket upload.
class PCMWorklet extends AudioWorkletProcessor {
  constructor() {
    super();
    this._ratio = sampleRate / 16000; // sampleRate is the AudioContext rate
    this._frac = 0;
    this._buf = [];
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel) return true;

    for (let i = 0; i < channel.length; i++) {
      this._frac -= 1;
      if (this._frac < 0) {
        this._frac += this._ratio;
        let s = channel[i];
        if (s > 1) s = 1;
        else if (s < -1) s = -1;
        this._buf.push(s < 0 ? s * 0x8000 : s * 0x7fff);
      }
    }

    if (this._buf.length >= 1280) {
      const out = new Int16Array(this._buf);
      this.port.postMessage(out.buffer, [out.buffer]);
      this._buf = [];
    }
    return true;
  }
}

registerProcessor("pcm-worklet", PCMWorklet);
