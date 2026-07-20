import { useState } from 'react'

import type { FinalSignal, TranscriptTurn } from '../types/transcript'
import { TranscriptWorkspace } from './TranscriptWorkspace'

interface SignalReviewWorkspaceProps {
  drivers: FinalSignal[]
  blockers: FinalSignal[]
  selectedSignal: FinalSignal | null
  selectedTurnId: number | null
  turns: TranscriptTurn[]
  turnsLoading: boolean
  turnsError?: string | null
  evidenceLocateFailed: boolean
  onSelect: (signal: FinalSignal) => void
  onViewTranscript: (signal: FinalSignal) => void
  onRetryTurns: () => void
}

function formatLabel(value: string | null): string {
  return value ? value.replace(/_/g, ' ') : 'Not available'
}

function SignalSection({
  title,
  signals,
  expandedSignal,
  selectedTurnId,
  turns,
  turnsLoading,
  turnsError,
  evidenceLocateFailed,
  onToggle,
  onViewTranscript,
  onRetryTurns,
}: {
  title: 'Drivers' | 'Blockers'
  signals: FinalSignal[]
  expandedSignal: FinalSignal | null
  selectedTurnId: number | null
  turns: TranscriptTurn[]
  turnsLoading: boolean
  turnsError?: string | null
  evidenceLocateFailed: boolean
  onToggle: (signal: FinalSignal) => void
  onViewTranscript: (signal: FinalSignal) => void
  onRetryTurns: () => void
}) {
  const emptyCopy = title === 'Drivers'
    ? 'No supported Drivers were identified.'
    : 'No supported Blockers were identified.'

  return (
    <section className={`simple-signal-section simple-signal-section--${title.toLowerCase()}`}>
      <header className="simple-signal-section__header">
        <div>
          <h4>{title}</h4>
          <p>{title === 'Drivers'
            ? 'What may motivate the advisor to move forward.'
            : 'What may delay or prevent movement.'}</p>
        </div>
        <span>{signals.length}</span>
      </header>

      {signals.length === 0 ? <p className="simple-signal-empty">{emptyCopy}</p> : (
        <div className="simple-signal-list">
          {signals.map((signal) => {
            const expanded = expandedSignal === signal
            return (
              <article className={`simple-signal-row simple-signal-row--${signal.item_type}`} key={`${signal.item_type}-${signal.rank}-${signal.category}`}>
                <div className="simple-signal-row__body">
                  <p className="simple-signal-row__eyebrow">{signal.item_type} · Rank {signal.rank}</p>
                  <div className="simple-signal-row__title">
                    <h5>{signal.category}</h5>
                    {signal.validator_verdict === 'needs_review' ? <span>Needs review</span> : null}
                  </div>
                  <p className="simple-signal-row__rationale">{signal.rationale}</p>
                  <blockquote>“{signal.advisor_quote}”</blockquote>
                  <footer>
                    <time>{signal.timestamp ?? 'No timestamp'}</time>
                    <button type="button" aria-expanded={expanded} onClick={() => onToggle(signal)}>
                      {expanded ? 'Close evidence' : 'View evidence'}
                    </button>
                  </footer>
                </div>

                {expanded ? (
                  <div className="simple-evidence" id={`signal-evidence-${signal.item_type}-${signal.rank}`}>
                    {signal.validator_verdict === 'needs_review' ? (
                      <p className="simple-evidence__review">Grounded in the transcript, but the interpretation should be confirmed.</p>
                    ) : null}
                    <TranscriptWorkspace
                      turns={turns}
                      selectedTurnId={selectedTurnId}
                      selectedQuote={signal.advisor_quote}
                      loading={turnsLoading}
                      errorMessage={turnsError}
                      evidenceLocateFailed={evidenceLocateFailed}
                      onRetry={onRetryTurns}
                      compact
                      idPrefix={`evidence-${signal.item_type}-${signal.rank}`}
                    />
                    {signal.supporting_evidence.length ? (
                      <div className="simple-evidence__supporting">
                        <strong>Supporting evidence</strong>
                        <ul>{signal.supporting_evidence.map((quote) => <li key={quote}>“{quote}”</li>)}</ul>
                      </div>
                    ) : null}
                    <div className="simple-evidence__actions">
                      <button type="button" onClick={() => navigator.clipboard.writeText(signal.advisor_quote)}>Copy quote</button>
                      <button type="button" onClick={() => onViewTranscript(signal)}>View in full transcript →</button>
                    </div>
                    <details className="simple-evidence__details">
                      <summary>Analysis details</summary>
                      <dl>
                        <div><dt>Business score</dt><dd>{signal.final_score?.toFixed(2) ?? 'Not available'}</dd></div>
                        <div><dt>Validation</dt><dd>{formatLabel(signal.validator_verdict)}</dd></div>
                        <div><dt>Evidence strength</dt><dd>{formatLabel(signal.evidence_strength)}</dd></div>
                      </dl>
                    </details>
                  </div>
                ) : null}
              </article>
            )
          })}
        </div>
      )}
    </section>
  )
}

export function SignalReviewWorkspace({
  drivers,
  blockers,
  selectedSignal,
  selectedTurnId,
  turns,
  turnsLoading,
  turnsError,
  evidenceLocateFailed,
  onSelect,
  onViewTranscript,
  onRetryTurns,
}: SignalReviewWorkspaceProps) {
  const [expandedSignal, setExpandedSignal] = useState<FinalSignal | null>(null)

  function toggleEvidence(signal: FinalSignal) {
    if (expandedSignal === signal) {
      setExpandedSignal(null)
      return
    }
    setExpandedSignal(signal)
    onSelect(signal)
  }

  const activeExpandedSignal = expandedSignal && selectedSignal === expandedSignal
    ? expandedSignal
    : expandedSignal

  return (
    <div className="simple-results">
      {drivers.length === 0 && blockers.length === 0 ? (
        <div className="simple-results__empty">
          <h4>No decision-relevant advisor signals were identified in this transcript.</h4>
          <p>An empty result means the available evidence did not meet the standard for a supported Driver or Blocker.</p>
        </div>
      ) : null}
      <SignalSection title="Drivers" signals={drivers} expandedSignal={activeExpandedSignal} selectedTurnId={selectedTurnId} turns={turns} turnsLoading={turnsLoading} turnsError={turnsError} evidenceLocateFailed={evidenceLocateFailed} onToggle={toggleEvidence} onViewTranscript={onViewTranscript} onRetryTurns={onRetryTurns} />
      <SignalSection title="Blockers" signals={blockers} expandedSignal={activeExpandedSignal} selectedTurnId={selectedTurnId} turns={turns} turnsLoading={turnsLoading} turnsError={turnsError} evidenceLocateFailed={evidenceLocateFailed} onToggle={toggleEvidence} onViewTranscript={onViewTranscript} onRetryTurns={onRetryTurns} />
    </div>
  )
}
