import styles from '../styles/components.module.css'

export function SkeletonCard({ height = 120 }) {
  return <div className={styles.skeleton} style={{ height }} />
}

export function SkeletonGauge({ size = 200 }) {
  return (
    <div
      className={styles.skeleton}
      style={{ width: size, height: size, borderRadius: '50%' }}
    />
  )
}

export function SkeletonChart({ height = 200 }) {
  return <div className={styles.skeleton} style={{ height, borderRadius: 'var(--radius-md)' }} />
}

export function SkeletonText({ width = '70%', height = 14 }) {
  return <div className={styles.skeleton} style={{ width, height, borderRadius: 4 }} />
}

export default { SkeletonCard, SkeletonGauge, SkeletonChart, SkeletonText }
