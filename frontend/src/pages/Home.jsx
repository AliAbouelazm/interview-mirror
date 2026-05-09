import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { getQuestionSet, listQuestionCategories, listSessions } from '../api/client'
import { useMediaDevices } from '../hooks/useMediaDevices'
import { SkeletonCard } from '../components/Skeleton'
import styles from '../styles/home.module.css'

const FEATURES = [
  ['01', 'Real-time signal fusion', 'Face, voice, and language scored every 500ms and combined into a single confidence and engagement gauge.'],
  ['02', 'Curated interview prompts', 'Behavioural, technical, leadership, situational, and personal questions cycle through the session with a per-question timer.'],
  ['03', 'Live facial tracking', 'MediaPipe face mesh runs locally and overlays 468 landmarks plus head pose on the live feed.'],
  ['04', 'Filler and hedge detection', 'Whisper transcribes you in 5-second windows; rule-based parsers flag filler words and hedging language as you speak.'],
  ['05', 'Per-question breakdown', 'After the session, every question gets its own confidence, engagement, word count, and filler line.'],
  ['06', 'Actionable insights', 'Each insight cites a real timestamp or measurement. No template advice.'],
]

function formatDate(ts) {
  if (!ts) return ''
  const d = new Date(ts * 1000)
  return d.toLocaleString(undefined, {
    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
  })
}

function formatDuration(s) {
  if (!s || s < 1) return '0:00'
  const total = Math.floor(s)
  const m = Math.floor(total / 60)
  const sec = total % 60
  return `${m}:${String(sec).padStart(2, '0')}`
}

export default function Home() {
  const navigate = useNavigate()
  const { request, status, error, stop } = useMediaDevices()
  const [sessions, setSessions] = useState(null)
  const [categories, setCategories] = useState([])
  const [selectedCategory, setSelectedCategory] = useState(null)
  const [questionCount, setQuestionCount] = useState(5)
  const [loadError, setLoadError] = useState(null)

  useEffect(() => {
    listSessions(20).then(setSessions).catch((e) => setLoadError(e.message))
    listQuestionCategories().then((r) => setCategories(r.categories || [])).catch(() => {})
  }, [])

  const handleStart = async () => {
    try {
      const stream = await request()
      if (!stream) return
      stop()
      const set = await getQuestionSet(questionCount, selectedCategory, Math.floor(Math.random() * 1e9))
      navigate('/session', { state: { questions: set, mode: selectedCategory || 'mixed' } })
    } catch {
      // permission denied path is rendered below
    }
  }

  return (
    <div className={styles.page}>
      <section className={styles.hero}>
        <span className={styles.eyebrow}>v1.0 / interview practice</span>
        <h1 className={styles.heading}>
          Read the room.<br />
          <span className={styles.headingHighlight}>Especially when the room is you.</span>
        </h1>
        <p className={styles.subheading}>
          Three independent emotion models score your face, your voice, and your words while you answer real interview
          questions. After the session you get the precise moments you slipped, what drove each one, and one specific
          thing to work on per insight.
        </p>

        <div className={styles.modeSelector}>
          <span className={styles.modeSelectorLabel}>Question pack</span>
          <div className={styles.modeSelectorRow}>
            <button
              className={`${styles.modeButton} ${selectedCategory === null ? styles.modeButtonActive : ''}`}
              onClick={() => setSelectedCategory(null)}
            >
              Mixed
            </button>
            {categories.map((c) => (
              <button
                key={c}
                className={`${styles.modeButton} ${selectedCategory === c ? styles.modeButtonActive : ''}`}
                onClick={() => setSelectedCategory(c)}
              >
                {c}
              </button>
            ))}
          </div>
        </div>

        <div className={styles.modeSelector}>
          <span className={styles.modeSelectorLabel}>Number of questions</span>
          <div className={styles.modeSelectorRow}>
            {[3, 5, 8, 12].map((n) => (
              <button
                key={n}
                className={`${styles.modeButton} ${questionCount === n ? styles.modeButtonActive : ''}`}
                onClick={() => setQuestionCount(n)}
              >
                {n}
              </button>
            ))}
          </div>
        </div>

        <div className={styles.actionRow}>
          <button
            type="button"
            className={styles.startButton}
            onClick={handleStart}
            disabled={status === 'requesting'}
          >
            Start session
            <span style={{ opacity: 0.6 }}>&rarr;</span>
          </button>
          {status === 'requesting' && (
            <span className={styles.permissionStatus}>Requesting camera and microphone</span>
          )}
        </div>

        {status === 'denied' && (
          <div className={styles.permissionDenied}>
            Camera or microphone access was denied. Enable both in your browser site settings, then try again. The
            stream stays in your browser; nothing is recorded server-side.
            {error ? ` (${error})` : ''}
          </div>
        )}
      </section>

      <section className={styles.section}>
        <span className={styles.sectionLabel}>What you get</span>
        <div className={styles.featureGrid}>
          {FEATURES.map(([n, title, desc]) => (
            <div key={n} className={styles.featureItem}>
              <span className={styles.featureNumber}>{n}</span>
              <div className={styles.featureTitle}>{title}</div>
              <div className={styles.featureDesc}>{desc}</div>
            </div>
          ))}
        </div>
      </section>

      <section className={styles.section}>
        <span className={styles.sectionLabel}>Past sessions</span>
        {sessions === null && !loadError && (
          <div>
            <SkeletonCard height={48} />
            <div style={{ height: 1 }} />
            <SkeletonCard height={48} />
          </div>
        )}
        {loadError && <div className={styles.empty}>{loadError}</div>}
        {sessions && sessions.length === 0 && (
          <div className={styles.empty}>No sessions yet. Start your first mock interview above.</div>
        )}
        {sessions && sessions.length > 0 && (
          <div>
            {sessions.map((s) => (
              <div
                key={s.id}
                className={styles.sessionRow}
                onClick={() => navigate(`/analysis/${s.id}`)}
              >
                <div className={styles.sessionMeta}>
                  <span className={styles.sessionDate}>{formatDate(s.ended_at || s.started_at)}</span>
                  <span className={styles.sessionDuration}>
                    {formatDuration(s.duration_seconds)} session
                  </span>
                </div>
                <span className={styles.sessionScoreLabel}>confidence</span>
                <span className={styles.sessionScore}>{Math.round(s.avg_confidence)}</span>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  )
}
