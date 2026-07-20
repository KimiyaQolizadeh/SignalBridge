import { Route, Routes } from 'react-router-dom'

import { AppLayout } from './components/AppLayout'
import { Dashboard } from './pages/Dashboard'
import { UploadTranscript } from './pages/UploadTranscript'
import { TranscriptDetailPage } from './pages/TranscriptDetailPage'

export default function App() {
  return (
    <AppLayout>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/upload" element={<UploadTranscript />} />
        <Route path="/transcripts/:id" element={<TranscriptDetailPage />} />
      </Routes>
    </AppLayout>
  )
}
