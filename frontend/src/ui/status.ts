export type StatusTone =
  | 'neutral'
  | 'info'
  | 'success'
  | 'warning'
  | 'danger'

export interface StatusPresentation {
  label: string
  tone: StatusTone
}

const statusPresentations: Record<string, StatusPresentation> = {
  pass: { label: 'Validated', tone: 'success' },
  needs_review: { label: 'Needs review', tone: 'warning' },
  reject: { label: 'Excluded', tone: 'danger' },
  rejected: { label: 'Excluded', tone: 'danger' },
  uploaded: { label: 'Ready to analyze', tone: 'info' },
  parsed: { label: 'Ready to continue', tone: 'info' },
  speakers_classified: { label: 'Ready to continue', tone: 'info' },
  candidates_extracted: { label: 'Ready to continue', tone: 'info' },
  evidence_validated: { label: 'Ready to continue', tone: 'info' },
  signals_scored: { label: 'Ready to continue', tone: 'info' },
  signals_deduplicated: { label: 'Ready to continue', tone: 'info' },
  queued: { label: 'Queued', tone: 'info' },
  processing: { label: 'Processing', tone: 'info' },
  running: { label: 'Processing', tone: 'info' },
  completed: { label: 'Complete', tone: 'success' },
  completed_without_results: { label: 'Complete — no signals', tone: 'success' },
  finalized: { label: 'Analysis complete', tone: 'success' },
  failed: { label: 'Processing failed', tone: 'danger' },
  error: { label: 'Processing failed', tone: 'danger' },
  pending: { label: 'Pending', tone: 'neutral' },
}

export function humanizeStatus(value: string | null | undefined): string {
  if (!value) return 'Not available'
  return value
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (letter) => letter.toUpperCase())
}

export function getStatusPresentation(
  value: string | null | undefined,
): StatusPresentation {
  const normalized = value?.trim().toLowerCase().replace(/[\s-]+/g, '_') ?? ''
  if (statusPresentations[normalized]) return statusPresentations[normalized]
  if (normalized.includes('fail') || normalized.includes('error')) {
    return statusPresentations.failed
  }
  if (
    normalized.includes('processing') ||
    normalized.includes('running') ||
    normalized.includes('progress')
  ) {
    return statusPresentations.processing
  }
  return { label: humanizeStatus(value), tone: 'neutral' }
}
