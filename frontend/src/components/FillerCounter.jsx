import { useEffect, useState } from 'react'
import styles from '../styles/components.module.css'

export default function FillerCounter({ count = 0, latestPhrase = '' }) {
  const [shownPhrase, setShownPhrase] = useState('')
  const [fade, setFade] = useState(false)

  useEffect(() => {
    if (!latestPhrase) return undefined
    setShownPhrase(latestPhrase)
    setFade(false)
    const t = setTimeout(() => setFade(true), 100)
    return () => clearTimeout(t)
  }, [latestPhrase])

  return (
    <div className={styles.fillerWrap}>
      <div className={styles.fillerHeader}>
        <span className={styles.fillerNumber}>{count}</span>
        <span className={styles.fillerLabel}>fillers</span>
      </div>
      <div className={`${styles.fillerLatest} ${fade ? styles.fillerLatestFade : ''}`}>
        {shownPhrase ? (
          <span style={{ color: 'var(--filler)' }}>{shownPhrase}</span>
        ) : (
          <span style={{ color: 'var(--text-tertiary)' }}>None detected</span>
        )}
      </div>
    </div>
  )
}
