import styles from '../styles/components.module.css'

function fmtTime(s) {
  const total = Math.max(0, Math.floor(s))
  const m = Math.floor(total / 60)
  const sec = total % 60
  return `${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`
}

export default function MomentCard({ moment, kind = 'high' }) {
  if (!moment) return null
  const accent = kind === 'high' ? 'var(--confidence-high)' : 'var(--confidence-low)'
  return (
    <div className={styles.momentCard} style={{ borderLeft: `2px solid ${accent}` }}>
      <div className={styles.momentHeader}>
        <span className={styles.momentTime}>
          {fmtTime(moment.start_seconds)} - {fmtTime(moment.end_seconds)}
        </span>
        <span className={styles.momentScore}>Score {moment.avg_score}</span>
      </div>
      <div className={styles.momentDriver}>
        Dominant signal: <strong>{moment.dominant_driver}</strong>
      </div>
      {moment.transcript_excerpt ? (
        <div className={styles.momentTranscript}>{moment.transcript_excerpt}</div>
      ) : (
        <div className={styles.momentEmpty}>No transcript captured for this window.</div>
      )}
    </div>
  )
}
