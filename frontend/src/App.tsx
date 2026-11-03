import { Navigate, Route, Routes } from 'react-router-dom'

import NavBar from './components/NavBar'
import LiveCamera from './pages/LiveCamera'
import PhotoAnalysis from './pages/PhotoAnalysis'
import VideoAnalysis from './pages/VideoAnalysis'

export default function App() {
  return (
    <div className="flex min-h-screen flex-col bg-paper">
      <NavBar />
      <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-10 sm:px-6 lg:px-8">
        <Routes>
          <Route path="/" element={<PhotoAnalysis />} />
          <Route path="/video" element={<VideoAnalysis />} />
          <Route path="/live-camera" element={<LiveCamera />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
      <footer className="mx-auto w-full max-w-6xl border-t border-line px-4 py-6 text-xs leading-relaxed text-faint sm:px-6 lg:px-8">
        Quiet / Busy / Crowded is a simple rule on the number of people the detector
        finds — not a head count, and not a trained classifier. This app does not score
        anomalies.
      </footer>
    </div>
  )
}
