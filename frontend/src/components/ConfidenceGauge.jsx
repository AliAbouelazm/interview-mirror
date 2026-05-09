import { useEffect, useRef, useState } from 'react'
import styles from '../styles/components.module.css'

const SIZE = 200
const STROKE = 12
const RADIUS = SIZE / 2 - STROKE / 2 - 4
const ARC_START_DEG = 135
const ARC_END_DEG = 405
const ARC_RANGE = ARC_END_DEG - ARC_START_DEG

function colorFor(score) {
  if (score >= 70) return 'var(--confidence-high)'
  if (score >= 45) return 'var(--confidence-mid)'
  return 'var(--confidence-low)'
}

function describeArc(cx, cy, r, startDeg, endDeg) {
  const start = polarToCartesian(cx, cy, r, endDeg)
  const end = polarToCartesian(cx, cy, r, startDeg)
  const largeArc = endDeg - startDeg <= 180 ? 0 : 1
  return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArc} 0 ${end.x} ${end.y}`
}

function polarToCartesian(cx, cy, r, deg) {
  const rad = ((deg - 90) * Math.PI) / 180
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) }
}

export default function ConfidenceGauge({ score = 0, label = 'Confidence' }) {
  const [displayed, setDisplayed] = useState(score)
  const targetRef = useRef(score)
  const rafRef = useRef(null)

  useEffect(() => {
    targetRef.current = score
    const animate = () => {
      setDisplayed((cur) => {
        const next = cur + (targetRef.current - cur) * 0.15
        if (Math.abs(next - targetRef.current) < 0.1) {
          rafRef.current = null
          return targetRef.current
        }
        rafRef.current = requestAnimationFrame(animate)
        return next
      })
    }
    if (!rafRef.current) {
      rafRef.current = requestAnimationFrame(animate)
    }
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
      rafRef.current = null
    }
  }, [score])

  const cx = SIZE / 2
  const cy = SIZE / 2
  const trackPath = describeArc(cx, cy, RADIUS, ARC_START_DEG, ARC_END_DEG)
  const valueDeg = ARC_START_DEG + ARC_RANGE * (Math.max(0, Math.min(100, displayed)) / 100)
  const valuePath = describeArc(cx, cy, RADIUS, ARC_START_DEG, valueDeg)

  return (
    <div className={styles.gaugeWrap}>
      <svg width={SIZE} height={SIZE} aria-label={`${label}: ${Math.round(displayed)}`}>
        <path d={trackPath} stroke="var(--border)" strokeWidth={STROKE} strokeLinecap="round" fill="none" />
        <path d={valuePath} stroke={colorFor(displayed)} strokeWidth={STROKE} strokeLinecap="round" fill="none"
              style={{ transition: 'stroke 200ms ease' }} />
      </svg>
      <div className={styles.gaugeInner}>
        <div className={styles.gaugeValue}>{Math.round(displayed)}</div>
        <div className={styles.gaugeLabel}>{label}</div>
      </div>
    </div>
  )
}
