import type { SpeakerRole, TranscriptTurn } from '../types/transcript'
import { EvidenceText } from './EvidenceText'

const speakerLabels: Record<SpeakerRole, string> = {
  advisor: 'Advisor',
  optimize_rep: 'Optimize representative',
  mixed: 'Mixed',
  unknown: 'Unknown',
}

interface TranscriptWorkspaceProps {
  turns: TranscriptTurn[]
  selectedTurnId: number | null
  selectedQuote?: string | null
  loading: boolean
  errorMessage?: string | null
  evidenceLocateFailed?: boolean
  onRetry?: () => void
  compact?: boolean
  idPrefix?: string
}

export function TranscriptWorkspace({
  turns,
  selectedTurnId,
  selectedQuote,
  loading,
  errorMessage,
  evidenceLocateFailed = false,
  onRetry,
  compact = false,
  idPrefix = 'turn',
}: TranscriptWorkspaceProps) {
  const selectedIndex = turns.findIndex((turn) => turn.id === selectedTurnId)
  const visibleTurns = compact && selectedIndex >= 0
    ? turns.slice(Math.max(0, selectedIndex - 1), selectedIndex + 2)
    : turns

  return (
    <section className={`transcript-workspace${compact ? ' transcript-workspace--compact' : ''}`} aria-label="Transcript evidence">
      {evidenceLocateFailed ? <p className="transcript-workspace__notice">Evidence turn could not be located.</p> : null}
      {loading ? <div className="inline-state" role="status"><span className="spinner" aria-hidden="true" /><span>Loading transcript turns...</span></div> : null}
      {errorMessage ? (
        <div className="inline-error" role="alert">
          <div><strong>Unable to load transcript turns</strong><p>{errorMessage}</p></div>
          {onRetry ? <button type="button" onClick={onRetry}>Try again</button> : null}
        </div>
      ) : null}
      {!loading && !errorMessage && turns.length === 0 ? <div className="inline-state empty-state"><span>No parsed transcript turns are available.</span></div> : null}
      {visibleTurns.length > 0 ? (
        <ol className="turn-list">
          {visibleTurns.map((turn) => {
            const role: SpeakerRole = turn.inferred_role ?? 'unknown'
            const selected = selectedTurnId === turn.id
            const relation = selected
              ? 'source'
              : selectedIndex >= 0 && turn.id === turns[selectedIndex - 1]?.id
                ? 'previous'
                : selectedIndex >= 0 && turn.id === turns[selectedIndex + 1]?.id
                  ? 'next'
                  : null
            return (
              <li id={`${idPrefix}-${turn.id}`} className={`turn-item turn-item--${role}${selected ? ' turn-item--selected' : ''}${relation ? ` turn-item--context-${relation}` : ''}`} key={turn.id}>
                <div className="turn-meta">
                  <span className="turn-speaker">{turn.raw_speaker_label ?? 'Speaker not identified'}</span>
                  <time>{turn.timestamp ?? 'No timestamp'}</time>
                  <span className={`speaker-badge speaker-${role}`}>{speakerLabels[role]}</span>
                  {relation ? <span className="source-turn-label">{relation === 'source' ? 'Source evidence' : `${relation} context`}</span> : null}
                </div>
                <p><EvidenceText text={turn.text} quote={selected ? selectedQuote : null} /></p>
              </li>
            )
          })}
        </ol>
      ) : null}
    </section>
  )
}
