import { useState } from 'react'

import type { FinalSignal } from '../types/transcript'
import { StatusBadge } from './StatusBadge'

interface SignalCardProps {
  signal: FinalSignal
  onViewEvidence?: (signal: FinalSignal) => void
}

function formatLabel(value: string | null): string {
  return value ? value.replace(/_/g, ' ') : 'Not available'
}

export function SignalCard({ signal, onViewEvidence }: SignalCardProps) {
  const [copyState, setCopyState] = useState<'idle' | 'copied' | 'failed'>('idle')

  async function copyEvidence() {
    try {
      await navigator.clipboard.writeText(signal.advisor_quote)
      setCopyState('copied')
      window.setTimeout(() => setCopyState('idle'), 1800)
    } catch {
      setCopyState('failed')
      window.setTimeout(() => setCopyState('idle'), 2200)
    }
  }

  return (
    <article className={"signal-row signal-row--" + signal.item_type}>
      <div className="signal-row__rank" aria-label={`Rank ${signal.rank}`}>
        {signal.rank.toString().padStart(2, '0')}
      </div>
      <div className="signal-row__content">
        <header className="signal-row__header">
          <div>
            <div className="signal-row__badges">
              <span className={"signal-type-badge signal-type-badge--" + signal.item_type}>
                {signal.item_type}
              </span>
              <StatusBadge value={signal.validator_verdict ?? 'pass'} />
            </div>
            <h4>{signal.category}</h4>
          </div>
          <div className="signal-score">
            <span>Business score</span>
            <strong>{signal.final_score === null ? '—' : signal.final_score.toFixed(2)}</strong>
          </div>
        </header>
        <p className="signal-row__rationale-label">Decision rationale</p>
        <p className="signal-row__rationale">{signal.rationale}</p>
        <blockquote className="evidence-panel">
          <span>Advisor evidence</span>
          <p>{signal.advisor_quote}</p>
        </blockquote>
        {signal.supporting_evidence.length > 0 ? (
          <div className="evidence-supporting">
            <span>Supporting evidence</span>
            <ul>{signal.supporting_evidence.map((evidence) => <li key={evidence}>{evidence}</li>)}</ul>
          </div>
        ) : null}
        {signal.evidence_context.length > 0 ? (
          <div className="evidence-context">
            <span>Adjacent transcript context</span>
            {signal.evidence_context.map((context) => (
              <p key={context.turn_id}><strong>{context.speaker ?? 'Speaker'}:</strong> {context.text}</p>
            ))}
          </div>
        ) : null}
        <footer className="signal-row__footer">
          <time>{signal.timestamp ?? 'No timestamp'}</time>
          <span>{formatLabel(signal.evidence_strength)} evidence</span>
          <button className="evidence-copy" type="button" onClick={copyEvidence}>
            {copyState === 'copied'
              ? 'Copied'
              : copyState === 'failed'
                ? 'Copy unavailable'
                : 'Copy evidence'}
          </button>
          {onViewEvidence ? (
            <button className="evidence-link" type="button" onClick={() => onViewEvidence(signal)}>
              View in transcript <span aria-hidden="true">&rarr;</span>
            </button>
          ) : null}
        </footer>
      </div>
    </article>
  )
}
