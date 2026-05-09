import { useEffect, useState } from 'react'
import styles from '../styles/components.module.css'

function fmt(seconds) {
  const s = Math.max(0, Math.floor(seconds))
  const m = Math.floor(s / 60)
  const sec = s % 60
  return `${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`
}

export default function QuestionPanel({
  questions,
  index,
  startedAt,
  onNext,
  onPrev,
  onSkip,
}) {
  const [now, setNow] = useState(Date.now())
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(id)
  }, [])

  if (!questions || questions.length === 0) {
    return (
      <div className={styles.questionCard}>
        <div className={styles.questionMeta}>
          <span>No question loaded</span>
        </div>
        <div className={styles.questionText}>Open practice. Speak about anything you want to rehearse.</div>
      </div>
    )
  }

  const q = questions[Math.min(index, questions.length - 1)]
  const elapsed = startedAt ? Math.floor((now - startedAt) / 1000) : 0
  const overTarget = elapsed > q.target_seconds
  const last = index >= questions.length - 1

  return (
    <div className={styles.questionCard}>
      <div className={styles.questionMeta}>
        <span>
          <span className={styles.questionCategory}>{q.category}</span>
          {' '}&middot;{' '}{q.difficulty}{' '}&middot;{' '}target {q.target_seconds}s
        </span>
        <span className={styles.questionTimer} style={overTarget ? { color: 'var(--filler)' } : null}>
          {fmt(elapsed)} / {fmt(q.target_seconds)}
        </span>
      </div>
      <div className={styles.questionText}>{q.text}</div>
      <div className={styles.questionMeta}>
        <span>Question {index + 1} of {questions.length}</span>
      </div>
      <div className={styles.questionActions}>
        <button type="button" className={styles.questionButton} onClick={onPrev} disabled={index === 0}>Prev</button>
        <button type="button" className={styles.questionButton} onClick={onSkip}>Skip</button>
        <button type="button" className={styles.questionButton} onClick={onNext} disabled={last}>
          {last ? 'Last' : 'Next'}
        </button>
      </div>
    </div>
  )
}
