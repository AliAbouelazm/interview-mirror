import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import LiveFeed from '../components/LiveFeed'
import VoiceWave from '../components/VoiceWave'
import ConfidenceGauge from '../components/ConfidenceGauge'
import SignalTimeline from '../components/SignalTimeline'
import FillerCounter from '../components/FillerCounter'
import QuestionPanel from '../components/QuestionPanel'
import { useMediaDevices } from '../hooks/useMediaDevices'
import { useSession } from '../hooks/useSession'
import { useWebSocket } from '../hooks/useWebSocket'
import { useFaceMesh } from '../hooks/useFaceMesh'
import { recordFaceTicks, recordQuestionEvent } from '../api/client'
import styles from '../styles/session.module.css'

const FRAME_INTERVAL_MS = 500
const FRAME_TARGET_W = 320
const FRAME_TARGET_H = 240
const JPEG_QUALITY = 0.7
const TICK_FLUSH_MS = 5000

function arrayBufferToBase64(buffer) {
  const bytes = new Uint8Array(buffer)
  let binary = ''
  const chunk = 0x8000
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk))
  }
  return btoa(binary)
}

function HighlightedTranscript({ text, flagged }) {
  if (!text) return <span className={styles.transcriptEmpty}>Transcript will appear as you speak.</span>
  if (!flagged || flagged.length === 0) return text
  const lower = text.toLowerCase()
  const parts = []
  let cursor = 0
  const sortedFlags = [...new Set(flagged.map((p) => p.toLowerCase()))]
    .sort((a, b) => b.length - a.length)
  while (cursor < text.length) {
    let matched = null
    let matchIdx = -1
    for (const phrase of sortedFlags) {
      const idx = lower.indexOf(phrase, cursor)
      if (idx >= 0 && (matchIdx < 0 || idx < matchIdx)) {
        matchIdx = idx
        matched = phrase
      }
    }
    if (matched && matchIdx >= 0) {
      if (matchIdx > cursor) parts.push(text.slice(cursor, matchIdx))
      parts.push(
        <mark key={parts.length} className={styles.transcriptFlagged}>
          {text.slice(matchIdx, matchIdx + matched.length)}
        </mark>,
      )
      cursor = matchIdx + matched.length
    } else {
      parts.push(text.slice(cursor))
      cursor = text.length
    }
  }
  return parts
}

export default function Session() {
  const navigate = useNavigate()
  const location = useLocation()
  const questions = location.state?.questions || []

  const { stream, status: mediaStatus, request, stop, error: mediaError } = useMediaDevices()
  const { sessionId, status: sessionStatus, error: sessionError, start, finish } = useSession()
  const [latest, setLatest] = useState(null)
  const [history, setHistory] = useState([])
  const [elapsedMs, setElapsedMs] = useState(0)
  const [questionIdx, setQuestionIdx] = useState(0)
  const [questionStartedAt, setQuestionStartedAt] = useState(null)
  const startTsRef = useRef(null)
  const frameCanvasRef = useRef(null)
  const audioCtxRef = useRef(null)
  const audioNodeRef = useRef(null)
  const sourceNodeRef = useRef(null)

  const [videoEl, setVideoEl] = useState(null)
  const [latestLandmarks, setLatestLandmarks] = useState(null)
  const tickBufferRef = useRef([])

  useEffect(() => {
    if (!stream && mediaStatus === 'idle') {
      request().catch(() => {})
    }
  }, [stream, mediaStatus, request])

  useEffect(() => {
    if (mediaStatus === 'granted' && !sessionId && sessionStatus === 'idle') {
      start().catch(() => {})
    }
  }, [mediaStatus, sessionId, sessionStatus, start])

  useEffect(() => {
    if (sessionStatus === 'active') startTsRef.current = Date.now()
  }, [sessionStatus])

  useEffect(() => {
    if (sessionStatus !== 'active') return undefined
    const id = setInterval(() => {
      if (startTsRef.current) setElapsedMs(Date.now() - startTsRef.current)
    }, 1000)
    return () => clearInterval(id)
  }, [sessionStatus])

  // Record question events when index advances
  useEffect(() => {
    if (sessionStatus !== 'active' || !sessionId || questions.length === 0) return
    const q = questions[questionIdx]
    if (!q) return
    const startedAt = Date.now()
    setQuestionStartedAt(startedAt)
    recordQuestionEvent(sessionId, {
      question_id: q.id,
      started_at: startedAt / 1000,
      ended_at: null,
    }).catch(() => {})
  }, [sessionId, sessionStatus, questionIdx, questions])

  const handleMessage = useCallback((data) => {
    if (data.type !== 'realtime') return
    setLatest(data)
    setHistory((prev) => {
      const next = prev.length >= 200 ? prev.slice(-199) : prev.slice()
      next.push(data)
      return next
    })
  }, [])

  const { status: wsStatus, send } = useWebSocket(sessionId, {
    enabled: !!sessionId && sessionStatus === 'active',
    onMessage: handleMessage,
  })

  // FaceMesh on the live video element
  const handleFaceTick = useCallback((m) => {
    setLatestLandmarks(m.landmarks)
    tickBufferRef.current.push({
      timestamp: Date.now() / 1000,
      head_yaw: m.head_yaw,
      head_pitch: m.head_pitch,
      head_roll: m.head_roll,
      eye_openness: m.eye_openness,
      smile: m.smile,
      looking_at_camera: m.looking_at_camera,
    })
  }, [])

  useFaceMesh({ video: videoEl, enabled: !!videoEl && sessionStatus === 'active', onTick: handleFaceTick })

  // Periodically flush face ticks to the server
  useEffect(() => {
    if (!sessionId || sessionStatus !== 'active') return undefined
    const id = setInterval(() => {
      if (tickBufferRef.current.length === 0) return
      const ticks = tickBufferRef.current.splice(0, tickBufferRef.current.length)
      recordFaceTicks(sessionId, ticks).catch(() => {})
    }, TICK_FLUSH_MS)
    return () => clearInterval(id)
  }, [sessionId, sessionStatus])

  // Send video frames at FRAME_INTERVAL_MS
  useEffect(() => {
    if (!stream || !sessionId || wsStatus !== 'connected') return undefined
    if (!videoEl) return undefined
    if (!frameCanvasRef.current) {
      frameCanvasRef.current = document.createElement('canvas')
      frameCanvasRef.current.width = FRAME_TARGET_W
      frameCanvasRef.current.height = FRAME_TARGET_H
    }
    const canvas = frameCanvasRef.current
    const ctx = canvas.getContext('2d')

    const tick = () => {
      try {
        if (videoEl.videoWidth > 0) {
          ctx.drawImage(videoEl, 0, 0, FRAME_TARGET_W, FRAME_TARGET_H)
          const dataUrl = canvas.toDataURL('image/jpeg', JPEG_QUALITY)
          send({ type: 'video', frame: dataUrl })
        }
      } catch {}
    }
    const id = setInterval(tick, FRAME_INTERVAL_MS)
    return () => clearInterval(id)
  }, [stream, sessionId, wsStatus, videoEl, send])

  // Stream PCM via AudioWorklet
  useEffect(() => {
    if (!stream || !sessionId || wsStatus !== 'connected') return undefined
    let cancelled = false
    const setup = async () => {
      try {
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 48000 })
        await audioCtx.audioWorklet.addModule('/pcm-worklet.js')
        if (cancelled) { audioCtx.close(); return }
        const source = audioCtx.createMediaStreamSource(stream)
        const node = new AudioWorkletNode(audioCtx, 'pcm-worklet', {
          processorOptions: { targetRate: 22050, frameSize: 4096 },
        })
        node.port.onmessage = (ev) => {
          const buf = ev.data
          if (buf && buf.byteLength > 0) {
            send({ type: 'audio', samples: arrayBufferToBase64(buf) })
          }
        }
        source.connect(node)
        const silentGain = audioCtx.createGain()
        silentGain.gain.value = 0
        node.connect(silentGain)
        silentGain.connect(audioCtx.destination)
        audioCtxRef.current = audioCtx
        audioNodeRef.current = node
        sourceNodeRef.current = source
      } catch {}
    }
    setup()
    return () => {
      cancelled = true
      try { audioNodeRef.current?.disconnect() } catch {}
      try { sourceNodeRef.current?.disconnect() } catch {}
      try { audioCtxRef.current?.close() } catch {}
      audioCtxRef.current = null
      audioNodeRef.current = null
      sourceNodeRef.current = null
    }
  }, [stream, sessionId, wsStatus, send])

  const elapsedLabel = useMemo(() => {
    const s = Math.floor(elapsedMs / 1000)
    const m = Math.floor(s / 60)
    const sec = s % 60
    return `${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`
  }, [elapsedMs])

  const handleEnd = async () => {
    try {
      // Mark current question ended
      if (sessionId && questions[questionIdx]) {
        recordQuestionEvent(sessionId, {
          question_id: questions[questionIdx].id,
          started_at: (questionStartedAt || Date.now()) / 1000,
          ended_at: Date.now() / 1000,
        }).catch(() => {})
      }
      // Flush remaining ticks
      if (sessionId && tickBufferRef.current.length > 0) {
        const ticks = tickBufferRef.current.splice(0, tickBufferRef.current.length)
        await recordFaceTicks(sessionId, ticks).catch(() => {})
      }
      stop()
      const r = await finish()
      if (r) navigate(`/analysis/${r.session_id}`)
    } catch {}
  }

  const handleQuestionAdvance = async (nextIdx) => {
    if (!sessionId) return
    const cur = questions[questionIdx]
    const start = questionStartedAt || Date.now()
    if (cur) {
      try {
        await recordQuestionEvent(sessionId, {
          question_id: cur.id,
          started_at: start / 1000,
          ended_at: Date.now() / 1000,
        })
      } catch {}
    }
    setQuestionIdx(nextIdx)
  }

  const ending = sessionStatus === 'ending'
  const error = mediaError || sessionError
  const faceSignal = latest?.face_signal || 'engaged'
  const voiceSignal = latest?.voice_signal || 'confident'
  const confidence = latest?.confidence_score ?? 0
  const engagement = latest?.engagement_score ?? 0
  const flagged = latest?.flagged_phrases || []
  const transcript = latest?.latest_transcript || ''
  const fillerCount = latest?.filler_count ?? 0
  const latestPhrase = flagged[flagged.length - 1] || ''
  const driver = latest?.dominant_driver || 'voice'
  const cycleMs = latest?.actual_cycle_ms ?? 500
  const totalLatency = latest?.latency_ms?.total ?? 0

  return (
    <div className={styles.page}>
      <div className={styles.column}>
        <LiveFeed
          stream={stream}
          faceSignal={faceSignal}
          wsStatus={wsStatus}
          landmarks={latestLandmarks}
          onVideoReady={setVideoEl}
        />
        <VoiceWave stream={stream} voiceSignal={voiceSignal} />
        <QuestionPanel
          questions={questions}
          index={questionIdx}
          startedAt={questionStartedAt}
          onPrev={() => handleQuestionAdvance(Math.max(0, questionIdx - 1))}
          onNext={() => handleQuestionAdvance(Math.min(questions.length - 1, questionIdx + 1))}
          onSkip={() => handleQuestionAdvance(Math.min(questions.length - 1, questionIdx + 1))}
        />
        {error && <div className={styles.errorBanner}>{error}</div>}
      </div>

      <div className={styles.centerColumn}>
        <div className={styles.gaugeBlock}>
          <ConfidenceGauge score={confidence} label="Confidence" />
          <div className={styles.engagementWrap}>
            <div className={styles.engagementHeader}>
              <span>Engagement</span>
              <span className="tabular">{Math.round(engagement)}</span>
            </div>
            <div className={styles.engagementBar}>
              <div className={styles.engagementFill} style={{ width: `${Math.max(2, engagement)}%` }} />
            </div>
          </div>
        </div>

        <div className={styles.signalGrid}>
          <div className={styles.signalCell}>
            <span className={styles.signalCellLabel}>Face</span>
            <span className={styles.signalCellValue}>{faceSignal}</span>
          </div>
          <div className={styles.signalCell}>
            <span className={styles.signalCellLabel}>Voice</span>
            <span className={styles.signalCellValue}>{voiceSignal}</span>
          </div>
          <div className={styles.signalCell}>
            <span className={styles.signalCellLabel}>Driver</span>
            <span className={styles.signalCellValue}>{driver}</span>
          </div>
          <div className={styles.signalCell}>
            <span className={styles.signalCellLabel}>Cycle</span>
            <span className={styles.signalCellValue}>{Math.round(cycleMs)}ms</span>
          </div>
        </div>

        <FillerCounter count={fillerCount} latestPhrase={latestPhrase} />

        <div className={styles.transcriptCard}>
          <div className={styles.transcriptHeader}>Transcript</div>
          <HighlightedTranscript text={transcript} flagged={flagged} />
        </div>
      </div>

      <div className={styles.column}>
        <SignalTimeline history={history} />
        <div className={styles.timerBlock}>
          <div className={styles.timerLabel}>Elapsed</div>
          <div className={styles.timer}>{elapsedLabel}</div>
        </div>
        <div className={styles.cycleIndicator}>
          <span>Cycle {Math.round(cycleMs)}ms</span>
          <span>Frame {Math.round(totalLatency)}ms</span>
        </div>
        <button
          type="button"
          className={styles.endButton}
          onClick={handleEnd}
          disabled={ending || !sessionId}
        >
          {ending && <span className={styles.endButtonSpinner} />}
          {ending ? 'Ending...' : 'End session'}
        </button>
      </div>
    </div>
  )
}
