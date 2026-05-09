const WS_BASE = import.meta.env.VITE_WS_URL || ''

export function buildWsUrl(sessionId) {
  if (WS_BASE) return `${WS_BASE.replace(/\/$/, '')}/ws/${sessionId}`
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
  return `${proto}://${window.location.host}/ws/${sessionId}`
}
