import styles from '../styles/components.module.css'

const METRIC_LABELS = {
  filler: 'Filler usage',
  hedge: 'Hedging',
  low_moment: 'Weakest moment',
  high_moment: 'Strongest moment',
  energy_drop: 'Voice energy',
  face: 'Face signal',
  trend: 'Session trend',
  empty: 'Session length',
}

export default function InsightCard({ insight }) {
  return (
    <article className={styles.insightCard}>
      <div className={styles.insightTag}>{METRIC_LABELS[insight.metric] || 'Insight'}</div>
      <h4 className={styles.insightTitle}>{insight.title}</h4>
      <p className={styles.insightBody}>{insight.body}</p>
    </article>
  )
}
