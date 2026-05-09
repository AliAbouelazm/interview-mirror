import { LineChart, Line, YAxis, ResponsiveContainer, Tooltip } from 'recharts'
import styles from '../styles/components.module.css'

const MAX_POINTS = 120

export default function SignalTimeline({ history = [] }) {
  const trimmed = history.slice(-MAX_POINTS).map((h, i) => ({
    i,
    confidence: h.confidence_score,
    engagement: h.engagement_score,
  }))

  return (
    <div className={styles.timelineWrap}>
      <div className={styles.timelineLabel}>Last 60 seconds</div>
      <div style={{ width: '100%', height: 160 }}>
        <ResponsiveContainer>
          <LineChart data={trimmed} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
            <YAxis hide domain={[0, 100]} />
            <Tooltip
              contentStyle={{
                background: 'var(--surface-raised)',
                border: '1px solid var(--border)',
                borderRadius: 6,
                fontSize: 12,
              }}
              labelFormatter={() => ''}
              formatter={(v, name) => [Math.round(v), name === 'confidence' ? 'Confidence' : 'Engagement']}
            />
            <Line type="monotone" dataKey="confidence" stroke="var(--confidence-high)" strokeWidth={2} dot={false} isAnimationActive={false} />
            <Line type="monotone" dataKey="engagement" stroke="var(--engagement)" strokeWidth={2} dot={false} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className={styles.timelineLegend}>
        <span><span className={styles.legendDot} style={{ background: 'var(--confidence-high)' }} /> Confidence</span>
        <span><span className={styles.legendDot} style={{ background: 'var(--engagement)' }} /> Engagement</span>
      </div>
    </div>
  )
}
