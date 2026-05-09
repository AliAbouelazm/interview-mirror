import { useCallback, useEffect, useRef, useState } from 'react'

const VIDEO_CONSTRAINTS = { width: { ideal: 320 }, height: { ideal: 240 }, frameRate: { ideal: 15 } }
const AUDIO_CONSTRAINTS = { echoCancellation: true, noiseSuppression: true, channelCount: 1 }

export function useMediaDevices() {
  const [stream, setStream] = useState(null)
  const [status, setStatus] = useState('idle')
  const [error, setError] = useState(null)
  const streamRef = useRef(null)

  const stop = useCallback(() => {
    const s = streamRef.current
    if (s) {
      s.getTracks().forEach(t => t.stop())
    }
    streamRef.current = null
    setStream(null)
    setStatus('idle')
  }, [])

  const request = useCallback(async () => {
    setStatus('requesting')
    setError(null)
    try {
      const s = await navigator.mediaDevices.getUserMedia({
        video: VIDEO_CONSTRAINTS,
        audio: AUDIO_CONSTRAINTS,
      })
      streamRef.current = s
      setStream(s)
      setStatus('granted')
      return s
    } catch (e) {
      setError(e.message || String(e))
      setStatus('denied')
      throw e
    }
  }, [])

  useEffect(() => () => stop(), [stop])

  return { stream, status, error, request, stop }
}
