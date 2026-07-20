import type { DiagnosticCandidate } from '../types/transcript'
import { StatusBadge } from './StatusBadge'

interface DiagnosticCandidateCardProps {
  candidate: DiagnosticCandidate
}

function formatScore(value: number | null): string {
  return value === null ? 'N/A' : value.toFixed(2)
}

export function DiagnosticCandidateCard({ candidate }: DiagnosticCandidateCardProps) {
  return (
    <details className="candidate-row">
      <summary>
        <span className={`candidate-row__type candidate-row__type--${candidate.item_type}`}>{candidate.item_type}</span>
        <strong>{candidate.category}</strong>
        <StatusBadge value={candidate.validator_verdict} />
        <span>{candidate.is_canonical ? 'Canonical' : 'Duplicate'}</span>
        <span aria-hidden="true">⌄</span>
      </summary>
      <div className="candidate-row__details">
        <blockquote>“{candidate.advisor_quote}”</blockquote>
        <p>{candidate.rationale}</p>
        <dl>
          <div><dt>Candidate</dt><dd>{candidate.id}</dd></div>
          <div><dt>Timestamp</dt><dd>{candidate.timestamp ?? 'None'}</dd></div>
          <div><dt>Rejection</dt><dd>{candidate.rejection_reason ?? 'None'}</dd></div>
          <div><dt>Support</dt><dd>{formatScore(candidate.support_score)}</dd></div>
          <div><dt>Advisor-side</dt><dd>{formatScore(candidate.advisor_side_score)}</dd></div>
          <div><dt>False-positive risk</dt><dd>{formatScore(candidate.false_positive_risk)}</dd></div>
          <div><dt>Final score</dt><dd>{formatScore(candidate.final_score)}</dd></div>
          <div><dt>Duplicate group</dt><dd>{candidate.duplicate_group_id ?? 'None'}</dd></div>
          <div><dt>Source turns</dt><dd>{candidate.source_turn_ids?.length ? candidate.source_turn_ids.join(', ') : 'None'}</dd></div>
        </dl>
      </div>
    </details>
  )
}
