import { useCallback, useState } from 'react'
import { startSession, endSession } from '../api/client'

export function useSession() {
  const [sessionId, setSessionId] = useState(null)
  const [startedAt, setStartedAt] = useState(null)
  const [status, setStatus] = useState('idle')
  const [error, setError] = useState(null)

  const start = useCallback(async (context = '') => {
    setStatus('starting')
    setError(null)
    try {
      const r = await startSession(context)
      setSessionId(r.session_id)
      setStartedAt(r.started_at)
      setStatus('active')
      return r
    } catch (e) {
      setError(e.message || String(e))
      setStatus('error')
      throw e
    }
  }, [])

  const finish = useCallback(async () => {
    if (!sessionId) return null
    setStatus('ending')
    try {
      const r = await endSession(sessionId)
      setStatus('ended')
      return r
    } catch (e) {
      setError(e.message || String(e))
      setStatus('error')
      throw e
    }
  }, [sessionId])

  const reset = useCallback(() => {
    setSessionId(null)
    setStartedAt(null)
    setStatus('idle')
    setError(null)
  }, [])

  return { sessionId, startedAt, status, error, start, finish, reset }
}
