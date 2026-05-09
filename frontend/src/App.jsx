import { Routes, Route, Navigate } from 'react-router-dom'
import Nav from './components/Nav'
import ErrorBoundary from './components/ErrorBoundary'
import Home from './pages/Home'
import Session from './pages/Session'
import Analysis from './pages/Analysis'

export default function App() {
  return (
    <ErrorBoundary>
      <Nav />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/session" element={<Session />} />
        <Route path="/analysis/:sessionId" element={<Analysis />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </ErrorBoundary>
  )
}
