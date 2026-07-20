import { useEffect, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Link, useParams } from 'react-router-dom'

import { ApiError } from '../api/client'
import {
  downloadTranscriptExport,
  getCandidates,
  getFinalSignals,
  getAnalysisRuns,
  getRunFinalSignals,
  getPipelineDiagnostics,
  getProcessingStatus,
  getTranscript,
  getTranscriptTurns,
  processTranscript,
  replayValidation,
} from '../api/transcripts'
import { Breadcrumbs } from '../components/Breadcrumbs'
import { DiagnosticCandidateCard } from '../components/DiagnosticCandidateCard'
import { PipelineStatus } from '../components/PipelineStatus'
import { SignalReviewWorkspace } from '../components/SignalReviewWorkspace'
import { TranscriptWorkspace } from '../components/TranscriptWorkspace'
import type {
  FinalSignal,
  ProcessingStatus,
  ProcessingState,
  TranscriptTurn,
} from '../types/transcript'

const numberFormatter = new Intl.NumberFormat('en-US')
const dateFormatter = new Intl.DateTimeFormat('en-US', {
  dateStyle: 'medium',
  timeStyle: 'short',
})

const processingStageLabels: Record<string, string> = {
  queued: 'Preparing transcript',
  parsing: 'Preparing transcript',
  classifying_speakers: 'Classifying speakers',
  extracting_candidates: 'Extracting candidate signals',
  validating_evidence: 'Validating evidence',
  scoring_signals: 'Scoring signals',
  deduplicating: 'Removing duplicates',
  reranking: 'Ranking final results',
}

const nonTerminalProcessingStatuses = new Set<ProcessingState>([
  'queued',
  'processing',
  'parsing',
  'classifying_speakers',
  'extracting_candidates',
  'validating_evidence',
  'scoring_signals',
  'deduplicating',
  'reranking',
])

export function isNonTerminalProcessingStatus(
  status: ProcessingState | undefined,
): boolean {
  return status !== undefined && nonTerminalProcessingStatuses.has(status)
}

function normalizeEvidenceText(value: string): string {
  return value
    .replace(/[“”]/g, '"')
    .replace(/[‘’]/g, "'")
    .replace(/\s+/g, ' ')
    .trim()
}

export function resolveEvidenceTurn({
  turns,
  advisorQuote,
  timestamp,
}: {
  turns: TranscriptTurn[]
  advisorQuote: string
  timestamp: string | null
}): TranscriptTurn | null {
  const quote = normalizeEvidenceText(advisorQuote)
  if (!quote) return null
  const advisorTurns = turns.filter((turn) => turn.inferred_role === 'advisor')
  const exactMatches = advisorTurns.filter(
    (turn) => normalizeEvidenceText(turn.text) === quote,
  )
  const matches = exactMatches.length > 0
    ? exactMatches
    : advisorTurns.filter((turn) => normalizeEvidenceText(turn.text).includes(quote))
  if (matches.length === 1) return matches[0]
  if (timestamp) {
    const timestampMatches = matches.filter((turn) => turn.timestamp === timestamp)
    if (timestampMatches.length === 1) return timestampMatches[0]
  }
  return null
}

function formatOptionalNumber(value: number | null): string {
  return value === null ? '—' : numberFormatter.format(value)
}

function formatCost(value: string | null): string {
  return value === null ? '—' : `$${Number(value).toFixed(4)}`
}

function formatDate(value: string): string {
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? 'Unknown date' : dateFormatter.format(date)
}

function getErrorMessage(error: unknown, fallback: string): string {
  return error instanceof ApiError ? error.message : fallback
}

function formatElapsed(seconds: number): string {
  const minutes = Math.floor(seconds / 60)
  const remainder = seconds % 60
  return minutes ? `${minutes}m ${remainder.toString().padStart(2, '0')}s` : `${remainder}s`
}

function useElapsedTime(active: boolean): number {
  const [seconds, setSeconds] = useState(0)

  useEffect(() => {
    if (!active) {
      setSeconds(0)
      return
    }
    const startedAt = Date.now()
    const timer = window.setInterval(() => {
      setSeconds(Math.floor((Date.now() - startedAt) / 1000))
    }, 1000)
    return () => window.clearInterval(timer)
  }, [active])

  return seconds
}

export function TranscriptDetailPage() {
  const { id } = useParams()
  const transcriptId = Number(id)
  const validId = Number.isInteger(transcriptId) && transcriptId > 0
  const queryClient = useQueryClient()
  const [activeTab, setActiveTab] = useState<'signals' | 'transcript' | 'diagnostics'>('signals')
  const [selectedTurnId, setSelectedTurnId] = useState<number | null>(null)
  const [selectedSignal, setSelectedSignal] = useState<FinalSignal | null>(null)
  const [evidenceLocateFailed, setEvidenceLocateFailed] = useState(false)
  const [diagnosticsOpened, setDiagnosticsOpened] = useState(false)
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null)
  const [diagnosticFilter, setDiagnosticFilter] =
    useState<'all' | 'driver' | 'blocker'>('all')

  const transcriptQuery = useQuery({
    queryKey: ['transcript', transcriptId],
    queryFn: () => getTranscript(transcriptId),
    enabled: validId,
  })
  const runsQuery = useQuery({
    queryKey: ['analysis-runs', transcriptId],
    queryFn: () => getAnalysisRuns(transcriptId),
    enabled: validId,
  })
  const turnsQuery = useQuery({
    queryKey: ['transcript-turns', transcriptId],
    queryFn: () => getTranscriptTurns(transcriptId),
    enabled: validId,
  })
  const pipelineMutation = useMutation({
    mutationFn: () => processTranscript(transcriptId),
    onMutate: async () => {
      setSelectedRunId(null)
      await Promise.all([
        queryClient.cancelQueries({ queryKey: ['processing-status', transcriptId] }),
        queryClient.cancelQueries({ queryKey: ['final-signals', transcriptId] }),
        queryClient.cancelQueries({ queryKey: ['pipeline-diagnostics', transcriptId] }),
      ])
      const previousStatus = queryClient.getQueryData<ProcessingStatus>(
        ['processing-status', transcriptId],
      )
      const previousSignals = queryClient.getQueryData<FinalSignal[]>(
        ['final-signals', transcriptId],
      )
      queryClient.setQueryData<ProcessingStatus>(
        ['processing-status', transcriptId],
        {
          transcript_id: transcriptId,
          run_id: null,
          current_stage: 'queued',
          status: 'queued',
          started_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
          completed_at: null,
          elapsed_seconds: 0,
          error_category: null,
        },
      )
      queryClient.setQueryData<FinalSignal[]>(['final-signals', transcriptId], [])
      queryClient.removeQueries({ queryKey: ['pipeline-diagnostics', transcriptId] })
      return { previousStatus, previousSignals }
    },
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['transcript', transcriptId] }),
        queryClient.invalidateQueries({ queryKey: ['final-signals', transcriptId] }),
        queryClient.invalidateQueries({ queryKey: ['processing-status', transcriptId] }),
        queryClient.invalidateQueries({ queryKey: ['pipeline-diagnostics', transcriptId] }),
        queryClient.invalidateQueries({ queryKey: ['analysis-runs', transcriptId] }),
      ])
    },
    onError: async (_error, _variables, context) => {
      if (context?.previousStatus) {
        queryClient.setQueryData(
          ['processing-status', transcriptId],
          context.previousStatus,
        )
      } else {
        queryClient.removeQueries({ queryKey: ['processing-status', transcriptId] })
      }
      if (context?.previousSignals) {
        queryClient.setQueryData(
          ['final-signals', transcriptId],
          context.previousSignals,
        )
      }
      await queryClient.invalidateQueries({ queryKey: ['transcript', transcriptId] })
    },
  })
  const replayMutation = useMutation({
    mutationFn: (runId: string) => replayValidation(runId),
    onMutate: async () => {
      setSelectedRunId(null)
      queryClient.setQueryData<FinalSignal[]>(['final-signals', transcriptId], [])
    },
    onSuccess: async (run) => {
      setSelectedRunId(run.run_id)
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['analysis-runs', transcriptId] }),
        queryClient.invalidateQueries({ queryKey: ['final-signals', transcriptId] }),
      ])
    },
  })
  const processingStatusQuery = useQuery({
    queryKey: ['processing-status', transcriptId],
    queryFn: () => getProcessingStatus(transcriptId),
    enabled: validId,
    refetchInterval: (data) =>
      isNonTerminalProcessingStatus(data?.status) ? 1000 : false,
    onSuccess: async (data) => {
      if (!['completed', 'completed_without_results', 'failed'].includes(data.status)) {
        return
      }
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['transcript', transcriptId] }),
        queryClient.invalidateQueries({ queryKey: ['final-signals', transcriptId] }),
        queryClient.invalidateQueries({ queryKey: ['pipeline-diagnostics', transcriptId] }),
      ])
    },
  })
  const finalSignalsQuery = useQuery({
    queryKey: ['final-signals', transcriptId],
    queryFn: () => getFinalSignals(transcriptId),
    enabled:
      validId &&
      (
        transcriptQuery.data?.status === 'finalized' ||
        processingStatusQuery.data?.status === 'completed' ||
        processingStatusQuery.data?.status === 'completed_without_results' ||
        pipelineMutation.isSuccess
      ),
  })
  const historicalSignalsQuery = useQuery({
    queryKey: ['run-final-signals', selectedRunId],
    queryFn: () => getRunFinalSignals(selectedRunId as string),
    enabled: Boolean(selectedRunId),
  })
  const csvDownload = useMutation({
    mutationFn: () => downloadTranscriptExport(transcriptId, 'csv'),
  })
  const jsonlDownload = useMutation({
    mutationFn: () => downloadTranscriptExport(transcriptId, 'jsonl'),
  })
  const candidatesQuery = useQuery({
    queryKey: ['candidates', transcriptId],
    queryFn: () => getCandidates(transcriptId),
    enabled: validId && (diagnosticsOpened || activeTab === 'diagnostics'),
  })
  const pipelineDiagnosticsQuery = useQuery({
    queryKey: ['pipeline-diagnostics', transcriptId],
    queryFn: () => getPipelineDiagnostics(transcriptId),
    enabled: validId && (
      transcriptQuery.data?.status === 'finalized' ||
      processingStatusQuery.data?.status === 'completed' ||
      processingStatusQuery.data?.status === 'completed_without_results' ||
      processingStatusQuery.data?.status === 'failed'
    ),
    retry: false,
  })
  const localElapsedSeconds = useElapsedTime(
    pipelineMutation.isLoading ||
    isNonTerminalProcessingStatus(processingStatusQuery.data?.status),
  )

  if (!validId) {
    return (
      <div className="state-card error-state alert alert--error" role="alert">
        <div><h3>Invalid transcript</h3><p>The transcript ID in this address is not valid.</p></div>
        <Link className="button button--secondary" to="/">Back to transcripts</Link>
      </div>
    )
  }

  if (transcriptQuery.isLoading) {
    return (
      <div className="analysis-skeleton" role="status" aria-label="Loading transcript analysis">
        <span className="visually-hidden">Loading transcript analysis</span>
        <div className="skeleton skeleton--title" />
        <div className="metric-grid">
          {[0, 1, 2, 3].map((item) => <div className="skeleton skeleton--metric" key={item} />)}
        </div>
        <div className="skeleton skeleton--panel" />
      </div>
    )
  }

  if (transcriptQuery.isError) {
    return (
      <div className="state-card error-state alert alert--error" role="alert">
        <div>
          <h3>Unable to load transcript</h3>
          <p>{getErrorMessage(transcriptQuery.error, 'An unexpected error occurred while loading the transcript.')}</p>
        </div>
        <button type="button" onClick={() => transcriptQuery.refetch()}>Try again</button>
      </div>
    )
  }

  const transcript = transcriptQuery.data
  if (!transcript) return null

  const progressStatus = processingStatusQuery.data
  const isProcessing = pipelineMutation.isLoading || replayMutation.isLoading ||
    isNonTerminalProcessingStatus(progressStatus?.status)
  const analysisActionDisabled = isProcessing || processingStatusQuery.isLoading
  const completedWithoutResults =
    progressStatus?.status === 'completed_without_results'
  const analysisAvailable = !isProcessing && (
    transcript.status === 'finalized' || pipelineMutation.isSuccess ||
    progressStatus?.status === 'completed' || completedWithoutResults
  )
  const transcriptFailed =
    transcript.status.toLowerCase().includes('fail') ||
    transcript.status.toLowerCase().includes('error')
  const runFailed = progressStatus?.status === 'failed'
  const status = isProcessing
    ? { label: 'Processing', modifier: 'processing', description: 'Analysis is currently running.' }
    : pipelineMutation.isError || transcriptFailed || runFailed
      ? { label: 'Failed', modifier: 'failed', description: 'Analysis needs attention before results can be reviewed.' }
      : analysisAvailable
        ? { label: 'Completed', modifier: 'finalized', description: 'Analysis complete.' }
        : { label: 'Ready to process', modifier: 'ready', description: 'Review the parsed transcript, then run analysis.' }

  const actionLabel = isProcessing
    ? 'Processing...'
    : pipelineMutation.isError || transcriptFailed || runFailed
      ? 'Retry analysis'
      : analysisAvailable
        ? 'Re-run analysis'
        : 'Run analysis'

  const visibleFinalSignals = isProcessing ? [] : selectedRunId
    ? (historicalSignalsQuery.data ?? [])
    : (finalSignalsQuery.data ?? [])
  const drivers = visibleFinalSignals
    .filter((signal) => signal.item_type === 'driver')
    .sort((left, right) => left.rank - right.rank)
  const blockers = visibleFinalSignals
    .filter((signal) => signal.item_type === 'blocker')
    .sort((left, right) => left.rank - right.rank)
  const totalSignals = drivers.length + blockers.length
  const activeSignal = selectedSignal ?? drivers[0] ?? blockers[0] ?? null
  const activeEvidenceTurn = activeSignal ? resolveEvidenceTurn({ turns: turnsQuery.data ?? [], advisorQuote: activeSignal.advisor_quote, timestamp: activeSignal.timestamp }) : null
  const activeTurnId = selectedSignal ? selectedTurnId : activeEvidenceTurn?.id ?? null
  const validatedSignalCount = visibleFinalSignals.filter(
    (signal) => signal.validator_verdict === 'pass',
  ).length
  const needsReviewSignalCount = visibleFinalSignals.filter(
    (signal) => signal.validator_verdict === 'needs_review',
  ).length
  const speakerTurnCount = turnsQuery.isSuccess ? turnsQuery.data.length : null
  const elapsedSeconds = Math.floor(
    progressStatus?.elapsed_seconds ?? localElapsedSeconds
  )

  const diagnosticCandidates = (candidatesQuery.data ?? [])
    .filter((candidate) => diagnosticFilter === 'all' || candidate.item_type === diagnosticFilter)
    .sort((left, right) =>
      (right.final_score ?? -Infinity) - (left.final_score ?? -Infinity) ||
      left.id - right.id
    )
  const rejectedCandidates = diagnosticCandidates.filter(
    (candidate) => candidate.validator_verdict === 'reject',
  )
  const needsReviewCandidates = diagnosticCandidates.filter(
    (candidate) => candidate.validator_verdict === 'needs_review',
  )
  const duplicateCandidates = diagnosticCandidates.filter(
    (candidate) => !candidate.is_canonical,
  )

  function runAnalysis() {
    if (!analysisActionDisabled) {
      pipelineMutation.mutate()
      window.setTimeout(() => processingStatusQuery.refetch(), 100)
    }
  }

  function locateSignal(signal: FinalSignal) {
    const turn = resolveEvidenceTurn({
      turns: turnsQuery.data ?? [],
      advisorQuote: signal.advisor_quote,
      timestamp: signal.timestamp,
    })
    setSelectedTurnId(turn?.id ?? null)
    setSelectedSignal(signal)
    setEvidenceLocateFailed(turn === null)
    return turn
  }

  function selectInsight(signal: FinalSignal) {
    const turn = locateSignal(signal)
    if (turn) {
      window.setTimeout(() => {
        document.getElementById(`insight-turn-${turn.id}`)?.scrollIntoView({
          behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth',
          block: 'center',
        })
      }, 0)
    }
  }

  function viewEvidence(signal: FinalSignal) {
    const turn = locateSignal(signal)
    setActiveTab('transcript')
    if (turn) {
      window.setTimeout(() => {
        document.getElementById(`turn-${turn.id}`)?.scrollIntoView({
          behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth',
          block: 'center',
        })
      }, 0)
    }
  }
  return (
    <section className="transcript-detail" aria-labelledby="transcript-title">
      <Breadcrumbs
        items={[
          { label: 'Transcripts', to: '/' },
          { label: transcript.file_name },
        ]}
      />

      <header className="analysis-header" id="overview">
        <div className="analysis-header__identity">
          <p className="eyebrow">Transcript analysis</p>
          <div className="analysis-header__title-row">
            <h2 className="page-title" id="transcript-title">{transcript.file_name}</h2>
            <span className={`status-badge status-badge--${status.modifier}`}>{status.label}</span>
          </div>
          <p className="analysis-header__summary">{status.description} <span>Updated {formatDate(transcript.updated_at)}</span></p>
        </div>
        <div className="analysis-header__actions">
          <button className="button button--primary" type="button" disabled={analysisActionDisabled} onClick={runAnalysis}>{actionLabel}</button>
          {analysisAvailable ? (
            <details className="export-menu">
              <summary className="button button--secondary">Export</summary>
              <div>
                <button type="button" disabled={csvDownload.isLoading} onClick={() => csvDownload.mutate()}>CSV</button>
                <button type="button" disabled={jsonlDownload.isLoading} onClick={() => jsonlDownload.mutate()}>JSONL</button>
              </div>
            </details>
          ) : <button className="button button--secondary" type="button" disabled>Export</button>}
        </div>
      </header>

      <nav className="analysis-tabs" aria-label="Transcript analysis" role="tablist">
        {(['signals', 'transcript'] as const).map((tab) => (
          <button
            key={tab}
            type="button"
            role="tab"
            aria-selected={activeTab === tab}
            aria-controls={tab + '-panel'}
            className={activeTab === tab ? 'active' : ''}
            onClick={() => setActiveTab(tab)}
          >
            {tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </nav>




      <section className="transcript-review section tab-panel" id="transcript-panel" role="tabpanel" hidden={activeTab !== 'transcript'} aria-labelledby="transcript-review-title">
        <div className="section-header">
          <div>
            <h3 id="transcript-review-title">Transcript</h3>
            <p>{speakerTurnCount === null ? 'Loading speaker turns...' : `${numberFormatter.format(speakerTurnCount)} total speaker turns`}</p>
          </div>
        </div>
        <TranscriptWorkspace
          turns={turnsQuery.data ?? []}
          selectedTurnId={selectedTurnId}
          selectedQuote={selectedSignal?.advisor_quote}
          loading={turnsQuery.isLoading}
          errorMessage={turnsQuery.isError ? getErrorMessage(turnsQuery.error, 'An unexpected error occurred while loading speaker turns.') : null}
          evidenceLocateFailed={evidenceLocateFailed}
          onRetry={() => turnsQuery.refetch()}
        />
      </section>
      {isProcessing ? (
        <div className="analysis-processing alert" role="status" aria-live="polite">
          <span className="spinner" aria-hidden="true" />
          <div>
            <strong>{processingStageLabels[progressStatus?.current_stage ?? 'queued'] ?? 'Processing transcript'}</strong>
            <p>SignalBridge is reviewing the transcript evidence.</p>
          </div>
          <time aria-hidden="true">Elapsed: {formatElapsed(elapsedSeconds)}</time>
        </div>
      ) : null}

      {pipelineMutation.isError || runFailed ? (
        <div className="alert alert--error analysis-error" role="alert">
          <div>
            <strong>Analysis could not be completed</strong>
            <p>Stopped during {processingStageLabels[progressStatus?.current_stage ?? ''] ?? progressStatus?.current_stage ?? 'processing'}.</p>
            <small>Error category: {progressStatus?.error_category ?? getErrorMessage(pipelineMutation.error, 'Pipeline failure')}</small>
          </div>
          <div className="analysis-error__actions"><button className="button button--danger" type="button" onClick={runAnalysis}>Retry analysis</button><button className="button button--secondary" type="button" onClick={() => { setDiagnosticsOpened(true); setActiveTab('diagnostics') }}>View diagnostics</button></div>
        </div>
      ) : null}


      {analysisAvailable ? (
        <section className="results-section section tab-panel" id="signals-panel" role="tabpanel" hidden={activeTab !== 'signals'} aria-labelledby="results-title">
          <div className="section-header results-heading">
            <div><p className="eyebrow">AI analysis</p><h3 id="results-title">Business insights</h3><p>{totalSignals ? `${totalSignals} final signals` : 'No final signals identified'}</p></div>
          </div>

          {finalSignalsQuery.isLoading ? (
            <div className="inline-state results-state" role="status"><span className="spinner" aria-hidden="true" /><span>Loading final drivers and blockers...</span></div>
          ) : null}
          {finalSignalsQuery.isError ? (
            <div className="inline-error results-state" role="alert">
              <div><strong>Unable to load final results</strong><p>{getErrorMessage(finalSignalsQuery.error, 'An unexpected error occurred while loading final signals.')}</p></div>
              <button type="button" onClick={() => finalSignalsQuery.refetch()}>Try again</button>
            </div>
          ) : null}
          {finalSignalsQuery.isSuccess ? (
            <SignalReviewWorkspace
              drivers={drivers}
              blockers={blockers}
              selectedSignal={activeSignal}
              selectedTurnId={activeTurnId}
              turns={turnsQuery.data ?? []}
              turnsLoading={turnsQuery.isLoading}
              turnsError={turnsQuery.isError ? getErrorMessage(turnsQuery.error, 'An unexpected error occurred while loading speaker turns.') : null}
              evidenceLocateFailed={evidenceLocateFailed}
              onSelect={selectInsight}
              onViewTranscript={viewEvidence}
              onRetryTurns={() => turnsQuery.refetch()}
            />
          ) : null}
        </section>
      ) : activeTab === 'signals' ? <div className="workspace-empty"><h3>Signals</h3><p>Run analysis to generate validated drivers and blockers.</p></div> : null}

      <details className="analysis-details">
        <summary><span id="analysis-details-title">Analysis details</span><small>History, processing, and diagnostics</small></summary>
        <div className="analysis-details__content">
        <button className="button button--secondary analysis-details__diagnostics" type="button" onClick={() => { setDiagnosticsOpened(true); setActiveTab('diagnostics') }}>Open diagnostics</button>
        <details className="transcript-metadata"><summary>Transcript metadata</summary><dl className="analysis-header__metadata">
            <div><dt>Updated</dt><dd>{formatDate(transcript.updated_at)}</dd></div>
            <div><dt>Tokens</dt><dd>{numberFormatter.format(transcript.token_count)}</dd></div>
            <div><dt>Speaker turns</dt><dd>{speakerTurnCount === null ? 'Loading...' : numberFormatter.format(speakerTurnCount)}</dd></div>
          </dl>
        </details>
      {runsQuery.data?.length ? (
        <details className="run-history-disclosure">
          <summary><span>Analysis history</span><small>{runsQuery.data.length} analysis {runsQuery.data.length === 1 ? 'run' : 'runs'}</small></summary>
          <div className="run-history__content">
          <div className="section-header">
            <div>
              <h3 id="run-history-title">Analysis runs</h3>
              <p>Full analysis calls extraction again. Replay validation reuses saved candidates and costs less.</p>
            </div>
          </div>
          <div className="run-history__controls">
            <label>
              <span>Inspect run</span>
              <select value={selectedRunId ?? ''} disabled={isProcessing} onChange={(event) => setSelectedRunId(event.target.value || null)}>
                <option value="">Latest completed run</option>
                {runsQuery.data.map((run) => (
                  <option key={run.run_id} value={run.run_id}>
                    {formatDate(run.started_at)} · {run.run_type} · {run.status}
                  </option>
                ))}
              </select>
            </label>
            <button className="button button--secondary" type="button" disabled={isProcessing || !selectedRunId} onClick={() => selectedRunId && replayMutation.mutate(selectedRunId)}>
              {replayMutation.isLoading ? 'Replaying…' : 'Replay validation'}
            </button>
          </div>
          {selectedRunId ? <p className="run-history__selection">Selected run <code>{selectedRunId}</code></p> : <p className="run-history__selection">Latest run <code>{runsQuery.data[0].run_id}</code> · {runsQuery.data[0].run_type}</p>}
          </div>
        </details>
      ) : null}

      <PipelineStatus
        status={progressStatus}
        analysisComplete={analysisAvailable && !isProcessing}
        failed={pipelineMutation.isError || transcriptFailed || runFailed}
      />
        </div>
      </details>

      {csvDownload.isError || jsonlDownload.isError ? (
        <div className="export-error" role="alert">{getErrorMessage(csvDownload.error ?? jsonlDownload.error, 'The export could not be downloaded. Please try again.')}</div>
      ) : null}

      {analysisAvailable || runFailed ? (
        <section
          className="diagnostics-panel section"
          id="diagnostics-panel"
          role="tabpanel"
          hidden={activeTab !== 'diagnostics'}
          aria-labelledby="diagnostics-title"
        >
          <div className="section-header"><div><h3 id="diagnostics-title">Diagnostics</h3><p>Run performance and candidate review details.</p></div></div>
          <div className="diagnostics-content">
            {pipelineDiagnosticsQuery.isLoading ? (
              <div className="inline-state" role="status"><span className="spinner" aria-hidden="true" /><span>Loading run observability...</span></div>
            ) : null}
            {pipelineDiagnosticsQuery.data ? (
              <div className="run-diagnostics">
                <dl className="diagnostics-summary">
                  <div><dt>Status</dt><dd>{pipelineDiagnosticsQuery.data.status}</dd></div>
                  <div><dt>Total duration</dt><dd>{(pipelineDiagnosticsQuery.data.total_duration_ms / 1000).toFixed(2)}s</dd></div>
                  <div><dt>LLM calls</dt><dd>{numberFormatter.format(pipelineDiagnosticsQuery.data.total_call_count)}</dd></div>
                  <div><dt>Total tokens</dt><dd>{formatOptionalNumber(pipelineDiagnosticsQuery.data.total_tokens)}</dd></div>
                  <div><dt>Estimated cost</dt><dd>{formatCost(pipelineDiagnosticsQuery.data.total_estimated_cost)}</dd></div>
                </dl>
                <div className="diagnostics-table-wrap">
                  <table className="diagnostics-table">
                    <thead><tr><th scope="col">Stage</th><th scope="col">Status</th><th scope="col">Model</th><th scope="col">Duration</th><th scope="col">Calls</th><th scope="col">Tokens</th><th scope="col">Cost</th></tr></thead>
                    <tbody>{pipelineDiagnosticsQuery.data.stages.map((stage) => (
                      <tr key={stage.stage}><td>{stage.stage}</td><td>{stage.status}</td><td>{stage.model ?? '—'}</td><td>{(stage.duration_ms / 1000).toFixed(2)}s</td><td>{numberFormatter.format(stage.call_count)}</td><td>{formatOptionalNumber(stage.total_tokens)}</td><td>{formatCost(stage.estimated_cost)}</td></tr>
                    ))}</tbody>
                  </table>
                </div>
                <details className="diagnostics-technical">
                  <summary>Technical details</summary>
                  <dl>
                    <div><dt>Run ID</dt><dd><code>{pipelineDiagnosticsQuery.data.run_id}</code></dd></div>
                    <div><dt>Retries</dt><dd>{numberFormatter.format(pipelineDiagnosticsQuery.data.total_retry_count)}</dd></div>
                    <div><dt>Completed</dt><dd>{pipelineDiagnosticsQuery.data.completed_at ? formatDate(pipelineDiagnosticsQuery.data.completed_at) : '—'}</dd></div>
                    <div><dt>Embedding model</dt><dd>{pipelineDiagnosticsQuery.data.embedding_model}</dd></div>
                    <div><dt>Deduplication threshold</dt><dd>{pipelineDiagnosticsQuery.data.deduplication_threshold}</dd></div>
                    <div><dt>Reranker fallback</dt><dd>{pipelineDiagnosticsQuery.data.reranker_fallback === null ? '—' : pipelineDiagnosticsQuery.data.reranker_fallback ? 'Yes' : 'No'}</dd></div>
                    <div><dt>Failure stage</dt><dd>{pipelineDiagnosticsQuery.data.failed_stage ?? '—'}</dd></div>
                    <div><dt>Error category</dt><dd>{pipelineDiagnosticsQuery.data.error_category ?? '—'}</dd></div>
                  </dl>
                  <h5>Prompt hashes</h5>
                  <ul>{pipelineDiagnosticsQuery.data.prompt_provenance.map((prompt) => <li key={prompt.prompt_file_name}><span>{prompt.prompt_file_name}</span><code>{prompt.sha256}</code></li>)}</ul>
                  <h5>Scoring policy</h5>
                  <code>{JSON.stringify(pipelineDiagnosticsQuery.data.scoring_policy)}</code>
                </details>
              </div>
            ) : null}
            {candidatesQuery.isLoading ? (
              <div className="inline-state" role="status"><span className="spinner" aria-hidden="true" /><span>Loading analysis diagnostics...</span></div>
            ) : null}
            {candidatesQuery.isError ? (
              <div className="inline-error" role="alert">
                <div><strong>Unable to load analysis diagnostics</strong><p>{getErrorMessage(candidatesQuery.error, 'An unexpected error occurred while loading candidates.')}</p></div>
                <button type="button" onClick={() => candidatesQuery.refetch()}>Try again</button>
              </div>
            ) : null}
            {candidatesQuery.isSuccess ? (
              <>
                <div className="diagnostic-toolbar" aria-label="Filter analysis diagnostics">
                  {(['all', 'driver', 'blocker'] as const).map((filter) => (
                    <button key={filter} type="button" aria-pressed={diagnosticFilter === filter} className={diagnosticFilter === filter ? 'active' : ''} onClick={() => setDiagnosticFilter(filter)}>
                      {filter === 'all' ? 'All' : `${filter.charAt(0).toUpperCase()}${filter.slice(1)}s`}
                    </button>
                  ))}
                </div>
                <div className="diagnostic-groups">
                  {[
                    { key: 'rejected', title: 'Rejected', items: rejectedCandidates },
                    { key: 'needs-review', title: 'Needs Review', items: needsReviewCandidates },
                    { key: 'duplicates', title: 'Duplicates', items: duplicateCandidates },
                  ].map((group) => (
                    <section key={group.key} className="diagnostic-group" aria-labelledby={`diagnostic-${group.key}`}>
                      <div className="diagnostic-group-title"><h4 id={`diagnostic-${group.key}`}>{group.title}</h4><span>{group.items.length}</span></div>
                      {group.items.length ? <div className="diagnostic-list">{group.items.map((candidate) => <DiagnosticCandidateCard key={`${group.key}-${candidate.id}`} candidate={candidate} />)}</div> : <div className="diagnostic-empty">No {group.title.toLowerCase()} {diagnosticFilter === 'all' ? 'candidates' : `${diagnosticFilter}s`}.</div>}
                    </section>
                  ))}
                </div>
              </>
            ) : null}
          </div>
        </section>
      ) : (
        <div className="diagnostics-unavailable surface tab-panel" hidden={activeTab !== 'diagnostics'}>
          <strong>Analysis diagnostics</strong>
          <p>Diagnostics will be available after analysis produces candidate signals.</p>
        </div>
      )}
    </section>
  )
}
