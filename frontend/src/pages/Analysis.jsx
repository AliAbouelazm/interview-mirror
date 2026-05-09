import { useEffect, useMemo, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import {
  AreaChart, Area, BarChart, Bar, CartesianGrid, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from 'recharts'
import { getAnalysis, getTimeline } from '../api/client'
import { SkeletonCard, SkeletonChart } from '../components/Skeleton'
import MomentCard from '../components/MomentCard'
import InsightCard from '../components/InsightCard'
import styles from '../styles/analysis.module.css'

const TREND_GLYPH = { improving: '↑', declining: '↓', stable: '→' }

function fmtTime(s) {
  const total = Math.max(0, Math.floor(s))
  const m = Math.floor(total / 60)
  const sec = total % 60
  return `${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`
}

export default function Analysis() {
  const { sessionId } = useParams()
  const navigate = useNavigate()
  const [analysis, setAnalysis] = useState(null)
  const [timeline, setTimeline] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!sessionId) return
    setAnalysis(null)
    setTimeline(null)
    setError(null)
    Promise.all([getAnalysis(sessionId), getTimeline(sessionId)])
      .then(([a, t]) => {
        setAnalysis(a)
        setTimeline(t)
      })
      .catch((e) => setError(e.message || 'Failed to load analysis'))
  }, [sessionId])

  const chartData = useMemo(() => {
    if (!timeline?.frames?.length) return []
    const base = timeline.frames[0].timestamp
    return timeline.frames.map((f) => ({
      t: Math.round(f.timestamp - base),
      confidence: f.confidence_score,
      engagement: f.engagement_score,
    }))
  }, [timeline])

  const fillerChart = useMemo(() => {
    if (!analysis?.filler?.chart_data) return []
    return analysis.filler.chart_data.map((c) => ({ name: c.label, count: c.count }))
  }, [analysis])

  if (error) {
    return (
      <div className={styles.page}>
        <Link to="/" className={styles.backLink}>← Back</Link>
        <div className={styles.errorCard}>
          <h3 style={{ marginBottom: 8 }}>Could not load analysis</h3>
          <p style={{ color: 'var(--text-secondary)', fontSize: 13 }}>{error}</p>
        </div>
      </div>
    )
  }

  if (!analysis) {
    return (
      <div className={styles.page}>
        <Link to="/" className={styles.backLink}>← Back</Link>
        <div className={styles.scoreRow}>
          <SkeletonCard height={120} />
          <SkeletonCard height={120} />
        </div>
        <SkeletonChart height={280} />
        <SkeletonChart height={200} />
      </div>
    )
  }

  const overall = analysis.overall
  return (
    <div className={styles.page}>
      <Link to="/" className={styles.backLink}>← Back</Link>

      <div>
        <h1 className={styles.heading}>Session analysis</h1>
        <div className={styles.headingMeta}>
          {fmtTime(analysis.duration_seconds)} duration · {analysis.frame_count} frames analysed
        </div>
      </div>

      <div className={styles.scoreRow}>
        <div className={styles.scoreCard}>
          <div className={styles.scoreCardLeft}>
            <div className={styles.scoreLabel}>Confidence</div>
            <div className={styles.scoreValue}>{Math.round(overall.avg_confidence)}</div>
            <div className={styles.scoreTrend}>
              <span aria-hidden="true">{TREND_GLYPH[overall.trend] || '→'}</span>
              {overall.trend} ({overall.trend_slope >= 0 ? '+' : ''}{overall.trend_slope})
            </div>
          </div>
        </div>
        <div className={styles.scoreCard}>
          <div className={styles.scoreCardLeft}>
            <div className={styles.scoreLabel}>Engagement</div>
            <div className={styles.scoreValue}>{Math.round(overall.avg_engagement)}</div>
            <div className={styles.scoreTrend}>
              {analysis.face.dominant} face most often
            </div>
          </div>
        </div>
      </div>

      <div className={styles.section}>
        <div className={styles.sectionTitle}>Confidence and engagement over time</div>
        <div className={styles.chartCard}>
          <div style={{ width: '100%', height: 280 }}>
            <ResponsiveContainer>
              <AreaChart data={chartData} margin={{ top: 10, right: 10, bottom: 8, left: 0 }}>
                <defs>
                  <linearGradient id="confArea" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="var(--confidence-high)" stopOpacity="0.25" />
                    <stop offset="100%" stopColor="var(--confidence-high)" stopOpacity="0" />
                  </linearGradient>
                  <linearGradient id="engArea" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="var(--engagement)" stopOpacity="0.25" />
                    <stop offset="100%" stopColor="var(--engagement)" stopOpacity="0" />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="var(--border-subtle)" />
                <XAxis dataKey="t" stroke="var(--text-tertiary)" fontSize={11}
                       tickFormatter={(v) => fmtTime(v)} />
                <YAxis domain={[0, 100]} stroke="var(--text-tertiary)" fontSize={11} />
                <Tooltip
                  contentStyle={{
                    background: 'var(--surface-raised)', border: '1px solid var(--border)',
                    borderRadius: 6, fontSize: 12, color: 'var(--text-primary)',
                  }}
                  labelFormatter={(v) => `At ${fmtTime(v)}`}
                />
                <Area type="monotone" dataKey="confidence" stroke="var(--confidence-high)"
                      strokeWidth={2} fill="url(#confArea)" isAnimationActive={false} />
                <Area type="monotone" dataKey="engagement" stroke="var(--engagement)"
                      strokeWidth={2} fill="url(#engArea)" isAnimationActive={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className={styles.momentsRow}>
        <div className={styles.momentColumn}>
          <div className={styles.momentColumnHeader}>Strongest moments</div>
          {(analysis.moments.strongest || []).map((m, i) => (
            <MomentCard key={`high-${i}`} moment={m} kind="high" />
          ))}
          {(!analysis.moments.strongest || analysis.moments.strongest.length === 0) && (
            <div className={styles.errorCard}>Not enough data for moment detection.</div>
          )}
        </div>
        <div className={styles.momentColumn}>
          <div className={styles.momentColumnHeader}>Weakest moments</div>
          {(analysis.moments.weakest || []).map((m, i) => (
            <MomentCard key={`low-${i}`} moment={m} kind="low" />
          ))}
          {(!analysis.moments.weakest || analysis.moments.weakest.length === 0) && (
            <div className={styles.errorCard}>Not enough data for moment detection.</div>
          )}
        </div>
      </div>

      <div className={styles.section}>
        <div className={styles.sectionTitle}>Filler usage</div>
        <div className={styles.fillerStats}>
          <span><span className={styles.fillerStatNum}>{analysis.filler.total_count}</span> total</span>
          <span><span className={styles.fillerStatNum}>{analysis.filler.rate_per_minute}</span> per minute</span>
        </div>
        <div className={styles.chartCard}>
          {fillerChart.length === 0 ? (
            <div className={styles.sectionSubtitle}>No filler words detected. Strong delivery.</div>
          ) : (
            <div style={{ width: '100%', height: 200 }}>
              <ResponsiveContainer>
                <BarChart data={fillerChart} layout="vertical" margin={{ top: 10, right: 16, bottom: 8, left: 16 }}>
                  <CartesianGrid stroke="var(--border-subtle)" />
                  <XAxis type="number" stroke="var(--text-tertiary)" fontSize={11} allowDecimals={false} />
                  <YAxis dataKey="name" type="category" stroke="var(--text-tertiary)" fontSize={12} width={80} />
                  <Tooltip
                    cursor={{ fill: 'var(--border-subtle)' }}
                    contentStyle={{
                      background: 'var(--surface-raised)', border: '1px solid var(--border)',
                      borderRadius: 6, fontSize: 12,
                    }}
                  />
                  <Bar dataKey="count" fill="var(--filler)" radius={[0, 3, 3, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      </div>

      <div className={styles.section}>
        <div className={styles.sectionTitle}>Insights</div>
        <div className={styles.insightsRow}>
          {analysis.insights.map((insight, i) => (
            <InsightCard key={i} insight={insight} />
          ))}
        </div>
      </div>

      <button type="button" className={styles.startNew} onClick={() => navigate('/')}>
        Start new session
      </button>
    </div>
  )
}
