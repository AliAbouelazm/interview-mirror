const BASE = import.meta.env.VITE_API_URL || ''

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  })
  if (!res.ok) {
    let detail = ''
    try { detail = (await res.json()).detail || '' } catch {}
    throw new Error(detail || `${res.status} ${res.statusText}`)
  }
  return res.json()
}

export function startSession(context = '') {
  return request('/api/session/start', {
    method: 'POST',
    body: JSON.stringify({ context }),
  })
}

export function endSession(sessionId) {
  return request(`/api/session/${sessionId}/end`, { method: 'POST' })
}

export function getAnalysis(sessionId) {
  return request(`/api/session/${sessionId}/analysis`)
}

export function getTimeline(sessionId) {
  return request(`/api/session/${sessionId}/timeline`)
}

export function listSessions(limit = 50) {
  return request(`/api/sessions?limit=${limit}`)
}

export function getModelMetrics() {
  return request('/api/model/metrics')
}

export function getHealth() {
  return request('/api/health')
}

export function listQuestionCategories() {
  return request('/api/questions/categories')
}

export function getQuestionSet(n = 5, category = null, seed = null) {
  return request('/api/questions/set', {
    method: 'POST',
    body: JSON.stringify({ n, category, seed }),
  })
}

export function recordQuestionEvent(sessionId, payload) {
  return request(`/api/session/${sessionId}/question`, {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export function recordFaceTicks(sessionId, ticks) {
  return request(`/api/session/${sessionId}/face-ticks`, {
    method: 'POST',
    body: JSON.stringify({ ticks }),
  })
}

export function addBookmark(sessionId, payload) {
  return request(`/api/session/${sessionId}/bookmark`, {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}
