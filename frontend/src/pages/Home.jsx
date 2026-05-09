import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { listSessions } from '../api/client'
import { useMediaDevices } from '../hooks/useMediaDevices'
import { SkeletonCard } from '../components/Skeleton'
import styles from '../styles/home.module.css'

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
  const [loadError, setLoadError] = useState(null)

  useEffect(() => {
    listSessions(20)
      .then(setSessions)
      .catch((e) => setLoadError(e.message))
  }, [])

  const handleStart = async () => {
    try {
      const stream = await request()
      if (stream) {
        stop()
        navigate('/session')
      }
    } catch {
      // status is already 'denied'
    }
  }

  return (
    <div className={styles.page}>
      <h1 className={styles.heading}>Interview Mirror</h1>
      <p className={styles.subheading}>Real-time feedback on how you come across in interviews.</p>

      <div className={styles.cardRow}>
        <button
          type="button"
          className={styles.card}
          onClick={handleStart}
          disabled={status === 'requesting'}
        >
          <div className={styles.cardTitle}>Start Session</div>
          <div className={styles.cardBody}>
            Use your camera and microphone for a live mock interview with continuous feedback.
          </div>
          {status === 'requesting' && (
            <div className={styles.permissionStatus}>Requesting camera and microphone access...</div>
          )}
          {status === 'denied' && (
            <div className={styles.permissionDenied}>
              Camera or microphone access was denied. Enable both in your browser site settings,
              then click again. We do not record video; the stream stays in your browser.
              {error ? ` (${error})` : ''}
            </div>
          )}
        </button>

        <div className={styles.card} style={{ cursor: 'default' }}>
          <div className={styles.cardTitle}>Past Sessions</div>
          <div className={styles.cardBody}>
            Review completed interviews with timeline data, moments, and insights.
          </div>
        </div>
      </div>

      <div className={styles.pastList}>
        <div className={styles.sectionLabel}>Past Sessions</div>
        {sessions === null && !loadError && (
          <>
            <SkeletonCard height={48} />
            <div style={{ height: 8 }} />
            <SkeletonCard height={48} />
          </>
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
                <span className={styles.sessionScore}>{Math.round(s.avg_confidence)}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
