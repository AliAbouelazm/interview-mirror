import { useCallback, useEffect, useRef, useState } from 'react'
import { buildWsUrl } from '../api/websocket'

const RECONNECT_BASE_MS = 500
const RECONNECT_MAX_MS = 15000
const MAX_ATTEMPTS = 6
const PING_INTERVAL_MS = 25000

export function useWebSocket(sessionId, { onMessage, enabled = true } = {}) {
  const [status, setStatus] = useState('idle')
  const wsRef = useRef(null)
  const reconnectAttemptsRef = useRef(0)
  const closingRef = useRef(false)
  const pingTimerRef = useRef(null)

  const send = useCallback((msg) => {
    const ws = wsRef.current
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(typeof msg === 'string' ? msg : JSON.stringify(msg))
      return true
    }
    return false
  }, [])

  useEffect(() => {
    if (!enabled || !sessionId) return undefined
    let cancelled = false
    closingRef.current = false

    const clearPing = () => {
      if (pingTimerRef.current) {
        clearInterval(pingTimerRef.current)
        pingTimerRef.current = null
      }
    }

    const connect = () => {
      if (cancelled) return
      setStatus('connecting')
      const ws = new WebSocket(buildWsUrl(sessionId))
      wsRef.current = ws

      ws.onopen = () => {
        reconnectAttemptsRef.current = 0
        setStatus('connected')
        clearPing()
        pingTimerRef.current = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            try { ws.send(JSON.stringify({ type: 'ping' })) } catch {}
          }
        }, PING_INTERVAL_MS)
      }
      ws.onmessage = (ev) => {
        try {
          const data = JSON.parse(ev.data)
          if (data.type === 'pong') return
          onMessage && onMessage(data)
        } catch {}
      }
      ws.onerror = () => {
        setStatus('error')
      }
      ws.onclose = (ev) => {
        clearPing()
        wsRef.current = null
        if (closingRef.current || cancelled) {
          setStatus('idle')
          return
        }
        if (ev.code === 4404) {
          setStatus('lost')
          return
        }
        setStatus('disconnected')
        const attempt = reconnectAttemptsRef.current + 1
        reconnectAttemptsRef.current = attempt
        if (attempt > MAX_ATTEMPTS) {
          setStatus('lost')
          return
        }
        const delay = Math.min(RECONNECT_BASE_MS * 2 ** (attempt - 1), RECONNECT_MAX_MS)
        setTimeout(connect, delay)
      }
    }

    connect()

    return () => {
      cancelled = true
      closingRef.current = true
      clearPing()
      const ws = wsRef.current
      if (ws) {
        try { ws.close() } catch {}
      }
      wsRef.current = null
    }
  }, [sessionId, enabled, onMessage])

  return { status, send }
}
