import { useState } from 'react'

import type { FinalSignal } from '../types/transcript'
import { EvidenceText } from './EvidenceText'
import { StatusBadge } from './StatusBadge'

interface SignalCardProps {
  signal: FinalSignal
  onViewEvidence?: (signal: FinalSignal) => void
  showAdjacentContext?: boolean
}

function formatLabel(value: string | null): string {
  return value ? value.replace(/_/g, ' ') : 'Not available'
}

export function SignalCard({ signal, onViewEvidence, showAdjacentContext = true }: SignalCardProps) {
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
      <div className="signal-row__content">
        <header className="signal-row__header">
          <div className="signal-row__identity">
            <span className="signal-row__eyebrow">
              <span className={"signal-row__kind signal-row__kind--" + signal.item_type}>
                {signal.item_type}
              </span>
              <span aria-hidden="true">&middot;</span>
              <span aria-label={`Rank ${signal.rank}`}>Rank {signal.rank.toString().padStart(2, '0')}</span>
            </span>
            <div className="signal-row__title">
              <h4>{signal.category}</h4>
              <StatusBadge value={signal.validator_verdict ?? 'pass'} />
            </div>
          </div>
          <div className="signal-score">
            <span>Business score</span>
            <strong>{signal.final_score === null ? '—' : signal.final_score.toFixed(2)}</strong>
          </div>
        </header>

        <div className="signal-row__rationale-block">
          <p className="signal-row__rationale-label">Why it matters</p>
          <p className="signal-row__rationale">{signal.rationale}</p>
        </div>

        <blockquote className="evidence-panel">
          <span className="visually-hidden">Advisor evidence</span>
          <span className="evidence-panel__mark" aria-hidden="true">“</span>
          <p>{signal.advisor_quote}</p>
        </blockquote>

        {signal.supporting_evidence.length > 0 ? (
          <div className="evidence-supporting">
            <span>Supporting evidence</span>
            <ul>{signal.supporting_evidence.map((evidence) => <li key={evidence}>“{evidence}”</li>)}</ul>
          </div>
        ) : null}

        {showAdjacentContext && signal.evidence_context.length > 0 ? (
          <details className="evidence-context">
            <summary>
              <span>Adjacent transcript context</span>
              <svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true"><path d="m4 6 4 4 4-4" /></svg>
            </summary>
            <div className="evidence-context__content">
              {signal.evidence_context.map((context) => (
                <div className={`evidence-context__turn${context.text.includes(signal.advisor_quote) ? ' evidence-context__turn--source' : ''}`} key={context.turn_id}>
                  <div>
                    <strong>{context.speaker ?? 'Speaker'}</strong>
                    {context.timestamp ? <time>{context.timestamp}</time> : null}
                  </div>
                  <p>{context.text}</p>
                </div>
              ))}
            </div>
          </details>
        ) : null}

        <footer className="signal-row__footer">
          <div className="signal-row__metadata">
            <span>
              <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><circle cx="8" cy="8" r="5.5" /><path d="M8 4.8v3.5l2.4 1.4" /></svg>
              <time>{signal.timestamp ?? 'No timestamp'}</time>
            </span>
            <span>{formatLabel(signal.evidence_strength)} evidence</span>
          </div>
          <div className="signal-row__actions">
            <button className="evidence-copy" type="button" onClick={copyEvidence}>
              <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><rect x="5.5" y="5.5" width="7" height="7" rx="1" /><path d="M3.5 10.5h-1v-7h7v1" /></svg>
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
          </div>
        </footer>
      </div>
    </article>
  )
}
