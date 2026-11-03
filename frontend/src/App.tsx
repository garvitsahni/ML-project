import { Navigate, Route, Routes } from 'react-router-dom'

import NavBar from './components/NavBar'
import LiveCamera from './pages/LiveCamera'
import PhotoAnalysis from './pages/PhotoAnalysis'
import VideoAnalysis from './pages/VideoAnalysis'

export default function App() {
  return (
    <div className="min-h-screen bg-slate-50">
      <NavBar />
      <main className="mx-auto w-full max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        <Routes>
          <Route path="/" element={<PhotoAnalysis />} />
          <Route path="/video" element={<VideoAnalysis />} />
          <Route path="/live-camera" element={<LiveCamera />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
      <footer className="mx-auto w-full max-w-7xl px-4 pb-8 text-xs text-slate-500 sm:px-6 lg:px-8">
        Detector counts are estimates; no counting ground truth exists. Speeds are
        px/frame — never m/s. Anomaly scores come from a modest sequence-held-out model.
      </footer>
    </div>
  )
}
