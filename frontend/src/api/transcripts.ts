import { API_BASE_URL, ApiError, apiRequest } from './client'
import type {
  Transcript,
  TranscriptDetail,
  PipelineRunResponse,
  FinalSignal,
  DiagnosticCandidate,
  TranscriptTurn,
  TranscriptUploadResponse,
  ProcessingStatus,
  PipelineDiagnostics,
  AnalysisRun,
} from '../types/transcript'

export type TranscriptExportFormat = 'csv' | 'jsonl'

export interface TranscriptDownload {
  filename: string
  size: number
}

function getDownloadFilename(
  contentDisposition: string | null,
  fallback: string,
): string {
  if (!contentDisposition) return fallback

  const encodedMatch = contentDisposition.match(/filename\*=UTF-8''([^;]+)/i)
  if (encodedMatch) {
    try {
      return decodeURIComponent(encodedMatch[1].trim())
    } catch {
      // Fall through to a plain filename or the safe fallback.
    }
  }

  const plainMatch = contentDisposition.match(/filename="?([^";]+)"?/i)
  return plainMatch?.[1].trim() || fallback
}

export function getTranscripts(): Promise<Transcript[]> {
  return apiRequest<Transcript[]>('/api/transcripts')
}

export function getTranscript(id: number): Promise<TranscriptDetail> {
  return apiRequest<TranscriptDetail>(`/api/transcripts/${id}`)
}

export function getTranscriptTurns(id: number): Promise<TranscriptTurn[]> {
  return apiRequest<TranscriptTurn[]>(`/api/transcripts/${id}/turns`)
}

export function processTranscript(id: number): Promise<PipelineRunResponse> {
  return apiRequest<PipelineRunResponse>(`/api/transcripts/${id}/process-all`, {
    method: 'POST',
  })
}

export function getProcessingStatus(id: number): Promise<ProcessingStatus> {
  return apiRequest<ProcessingStatus>(`/api/transcripts/${id}/processing-status`)
}

export function getPipelineDiagnostics(id: number): Promise<PipelineDiagnostics> {
  return apiRequest<PipelineDiagnostics>(`/api/transcripts/${id}/diagnostics`)
}

export function getFinalSignals(id: number): Promise<FinalSignal[]> {
  return apiRequest<FinalSignal[]>(`/api/transcripts/${id}/final-signals`)
}

export function getAnalysisRuns(id: number): Promise<AnalysisRun[]> {
  return apiRequest<AnalysisRun[]>(`/api/transcripts/${id}/runs`)
}

export function getRunFinalSignals(runId: string): Promise<FinalSignal[]> {
  return apiRequest<FinalSignal[]>(`/api/runs/${runId}/final-signals`)
}

export function replayValidation(runId: string): Promise<AnalysisRun> {
  return apiRequest<AnalysisRun>(`/api/runs/${runId}/replay-validation`, { method: 'POST' })
}

export function getCandidates(id: number): Promise<DiagnosticCandidate[]> {
  return apiRequest<DiagnosticCandidate[]>(`/api/transcripts/${id}/candidates`)
}

export async function downloadTranscriptExport(
  id: number,
  format: TranscriptExportFormat,
): Promise<TranscriptDownload> {
  let response: Response
  try {
    response = await fetch(
      `${API_BASE_URL}/api/transcripts/${id}/export.${format}?debug=false`,
      { headers: { Accept: format === 'csv' ? 'text/csv' : 'application/jsonl' } },
    )
  } catch {
    throw new ApiError(
      'Could not reach the SignalBridge API. Confirm the backend is running.',
      0,
    )
  }

  if (!response.ok) {
    let message = `The export request returned status ${response.status}.`
    try {
      const body = await response.json() as { detail?: string }
      if (body.detail) message = body.detail
    } catch {
      // Keep the safe status-based message when the response is not JSON.
    }
    throw new ApiError(message, response.status)
  }

  const blob = await response.blob()
  const fallback = `transcript-${id}-signals.${format}`
  const filename = getDownloadFilename(
    response.headers.get('Content-Disposition'),
    fallback,
  )
  const objectUrl = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = objectUrl
  link.download = filename
  document.body.appendChild(link)
  link.click()
  link.remove()
  window.setTimeout(() => URL.revokeObjectURL(objectUrl), 0)
  return { filename, size: blob.size }
}

export function uploadTranscript(
  file: File,
): Promise<TranscriptUploadResponse> {
  const formData = new FormData()
  formData.append('file', file)

  return apiRequest<TranscriptUploadResponse>('/api/transcripts/upload', {
    method: 'POST',
    body: formData,
  })
}
