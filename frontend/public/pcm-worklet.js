class PCMProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super()
    const opts = (options && options.processorOptions) || {}
    this.targetRate = opts.targetRate || 22050
    this.frameSize = opts.frameSize || 2048
    this._buffer = []
    this._bufferLength = 0
  }

  process(inputs) {
    const input = inputs[0]
    if (!input || input.length === 0) return true
    const channel = input[0]
    if (!channel) return true

    const ratio = sampleRate / this.targetRate
    const outLen = Math.floor(channel.length / ratio)
    const out = new Float32Array(outLen)
    for (let i = 0; i < outLen; i++) {
      const idx = Math.floor(i * ratio)
      out[i] = channel[idx]
    }

    this._buffer.push(out)
    this._bufferLength += out.length

    while (this._bufferLength >= this.frameSize) {
      const merged = new Float32Array(this.frameSize)
      let offset = 0
      while (offset < this.frameSize && this._buffer.length > 0) {
        const head = this._buffer[0]
        const remaining = this.frameSize - offset
        if (head.length <= remaining) {
          merged.set(head, offset)
          offset += head.length
          this._buffer.shift()
        } else {
          merged.set(head.subarray(0, remaining), offset)
          this._buffer[0] = head.subarray(remaining)
          offset += remaining
        }
      }
      this._bufferLength -= this.frameSize
      this.port.postMessage(merged.buffer, [merged.buffer])
    }
    return true
  }
}

registerProcessor('pcm-worklet', PCMProcessor)
