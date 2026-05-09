import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { LineChart, Line, ResponsiveContainer, YAxis } from 'recharts'
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
  ['07', 'Eye contact and head pose', 'Yaw, pitch, smile and looking-at-camera percentage tracked across the session and charted over time.'],
  ['08', 'Persistent past sessions', 'Every session is stored in SQLite. Walk back through old runs, compare moments, and watch your trend.'],
]

function TrendCard({ sessions }) {
  const chartData = useMemo(() => {
    return [...sessions]
      .filter((s) => s.ended_at)
      .sort((a, b) => a.ended_at - b.ended_at)
      .map((s, i) => ({
        i,
        confidence: Math.round(s.avg_confidence),
        engagement: Math.round(s.avg_engagement),
      }))
  }, [sessions])

  const last = chartData[chartData.length - 1]
  const first = chartData[0]
  const delta = last && first ? last.confidence - first.confidence : 0
  const arrow = delta > 0 ? '↑' : delta < 0 ? '↓' : '→'

  return (
    <div className={styles.trendCard}>
      <div className={styles.trendStats}>
        <div>
          <div className={styles.trendValue}>{last?.confidence ?? 0}</div>
          <div className={styles.trendLabel}>Last session</div>
        </div>
        <div>
          <div className={styles.trendValue}>
            {arrow} {Math.abs(delta)}
          </div>
          <div className={styles.trendLabel}>Vs. first session</div>
        </div>
        <div>
          <div className={styles.trendValue}>{chartData.length}</div>
          <div className={styles.trendLabel}>Sessions logged</div>
        </div>
      </div>
      <div className={styles.trendChart}>
        <ResponsiveContainer>
          <LineChart data={chartData} margin={{ top: 4, right: 4, bottom: 4, left: 4 }}>
            <YAxis hide domain={[0, 100]} />
            <Line type="monotone" dataKey="confidence" stroke="var(--confidence-high)" strokeWidth={1.5} dot={{ r: 2 }} isAnimationActive={false} />
            <Line type="monotone" dataKey="engagement" stroke="var(--engagement)" strokeWidth={1.5} dot={false} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

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
          <div className={styles.sliderHeader}>
            <span className={styles.modeSelectorLabel}>Number of questions</span>
            <span className={styles.sliderValue}>{questionCount}</span>
          </div>
          <input
            type="range"
            min={1}
            max={15}
            step={1}
            value={questionCount}
            onChange={(e) => setQuestionCount(Number(e.target.value))}
            className={styles.slider}
          />
          <div className={styles.sliderTicks}>
            <span>1</span>
            <span>5</span>
            <span>10</span>
            <span>15</span>
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

      {sessions && sessions.length >= 2 && (
        <section className={styles.section}>
          <span className={styles.sectionLabel}>Confidence trend</span>
          <TrendCard sessions={sessions} />
        </section>
      )}

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
