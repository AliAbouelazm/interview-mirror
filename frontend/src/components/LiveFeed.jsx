import { useEffect, useRef } from 'react'
import styles from '../styles/components.module.css'

const RING_COLORS = {
  engaged: 'var(--confidence-high)',
  reactive: 'var(--confidence-mid)',
  nervous: 'var(--confidence-low)',
  tense: 'var(--confidence-low)',
}

const STATUS_DOT = {
  connected: 'var(--confidence-high)',
  connecting: 'var(--confidence-mid)',
  disconnected: 'var(--confidence-low)',
  error: 'var(--confidence-low)',
  idle: 'var(--text-tertiary)',
}

const STATUS_LABEL = {
  connected: 'Connected',
  connecting: 'Connecting',
  disconnected: 'Reconnecting',
  error: 'Connection lost',
  idle: 'Idle',
}

export default function LiveFeed({ stream, faceSignal = 'engaged', wsStatus = 'idle' }) {
  const videoRef = useRef(null)

  useEffect(() => {
    const v = videoRef.current
    if (!v) return
    if (stream) {
      v.srcObject = stream
      v.play().catch(() => {})
    } else {
      v.srcObject = null
    }
  }, [stream])

  const ringColor = RING_COLORS[faceSignal] || 'var(--border)'

  return (
    <div className={styles.liveFeedContainer}>
      <div className={styles.liveFeedWrap} style={{ boxShadow: `0 0 0 2px ${ringColor}, var(--shadow-1)` }}>
        <video ref={videoRef} muted playsInline className={styles.liveFeedVideo} />
        {!stream && <div className={styles.liveFeedPlaceholder}>Waiting for camera</div>}
      </div>
      <div className={styles.connectionRow}>
        <span className={styles.dot} style={{ background: STATUS_DOT[wsStatus] }} />
        <span className={styles.connectionLabel}>{STATUS_LABEL[wsStatus] || wsStatus}</span>
      </div>
    </div>
  )
}
