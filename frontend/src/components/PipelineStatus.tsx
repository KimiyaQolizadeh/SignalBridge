import type { ProcessingStatus } from '../types/transcript'

interface PipelineStatusProps {
  status: ProcessingStatus | undefined
  analysisComplete: boolean
  failed: boolean
}

const stages = [
  { key: 'parsing', label: 'Transcript parsing' },
  { key: 'classifying_speakers', label: 'Speaker classification' },
  { key: 'extracting_candidates', label: 'Signal extraction' },
  { key: 'validating_evidence', label: 'Evidence validation' },
  { key: 'scoring_signals', label: 'Business scoring' },
  { key: 'deduplicating', label: 'Deduplication' },
  { key: 'reranking', label: 'Final ranking' },
] as const

export function PipelineStatus({
  status,
  analysisComplete,
  failed,
}: PipelineStatusProps) {
  const activeIndex = stages.findIndex((stage) => stage.key === status?.current_stage)
  const steps = (
    <ol className="pipeline-steps">
      {stages.map((stage, index) => {
        const state = analysisComplete
          ? 'completed'
          : failed && index === activeIndex
            ? 'failed'
            : index < activeIndex
              ? 'completed'
              : index === activeIndex
                ? 'active'
                : 'pending'
        return (
          <li className={`pipeline-step pipeline-step--${state}`} key={stage.key}>
            <span className="pipeline-step__marker" aria-hidden="true">
              {state === 'completed' ? '✓' : index + 1}
            </span>
            <span>
              <strong>{stage.label}</strong>
              <small>
                {state === 'completed'
                  ? 'Complete'
                  : state === 'active'
                    ? 'In progress'
                    : state === 'failed'
                      ? 'Failed'
                      : 'Pending'}
              </small>
            </span>
          </li>
        )
      })}
    </ol>
  )

  if (analysisComplete && !failed) {
    return (
      <details className="pipeline-disclosure">
        <summary>
          <span><strong>Analysis complete</strong><small>7 processing stages</small></span>
          <span aria-hidden="true">+</span>
        </summary>
        <div className="pipeline-disclosure__content">{steps}</div>
      </details>
    )
  }

  return (
    <section className="pipeline-panel" aria-labelledby="pipeline-status-title">
      <div className="pipeline-panel__header">
        <div>
          <p className="eyebrow">Processing control</p>
          <h3 id="pipeline-status-title">Analysis pipeline</h3>
        </div>
        <span className="pipeline-panel__summary">
          {failed
            ? 'Needs attention'
            : status?.status === 'queued'
              ? 'Starting'
              : activeIndex >= 0
                ? `Stage ${activeIndex + 1} of 7`
                : 'Not started'}
        </span>
      </div>
      {steps}
    </section>
  )
}