import { useEffect, useRef } from 'react'
import styles from '../styles/components.module.css'

const BAR_COUNT = 20
const VOICE_COLORS = {
  confident: 'var(--confidence-high)',
  reactive: 'var(--confidence-mid)',
  nervous: 'var(--confidence-low)',
  tense: 'var(--confidence-low)',
}

export default function VoiceWave({ stream, voiceSignal = 'confident' }) {
  const containerRef = useRef(null)
  const animationRef = useRef(null)
  const analyserRef = useRef(null)
  const audioContextRef = useRef(null)
  const sourceRef = useRef(null)

  useEffect(() => {
    if (!stream) return undefined
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)()
    audioContextRef.current = audioCtx
    const source = audioCtx.createMediaStreamSource(stream)
    sourceRef.current = source
    const analyser = audioCtx.createAnalyser()
    analyser.fftSize = 256
    source.connect(analyser)
    analyserRef.current = analyser

    const buffer = new Uint8Array(analyser.frequencyBinCount)
    const bars = containerRef.current?.querySelectorAll('span') || []

    const tick = () => {
      analyser.getByteFrequencyData(buffer)
      const step = Math.floor(buffer.length / BAR_COUNT)
      for (let i = 0; i < BAR_COUNT && i < bars.length; i++) {
        const slice = buffer.slice(i * step, (i + 1) * step)
        const avg = slice.reduce((a, b) => a + b, 0) / Math.max(slice.length, 1)
        const norm = Math.min(1, avg / 200)
        bars[i].style.transform = `scaleY(${0.15 + 0.85 * norm})`
      }
      animationRef.current = requestAnimationFrame(tick)
    }
    tick()

    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current)
      try { source.disconnect() } catch {}
      try { audioCtx.close() } catch {}
    }
  }, [stream])

  const barColor = VOICE_COLORS[voiceSignal] || 'var(--text-tertiary)'

  return (
    <div className={styles.voiceWaveWrap} ref={containerRef}>
      {Array.from({ length: BAR_COUNT }).map((_, i) => (
        <span key={i} className={styles.voiceBar} style={{ background: barColor }} />
      ))}
    </div>
  )
}
