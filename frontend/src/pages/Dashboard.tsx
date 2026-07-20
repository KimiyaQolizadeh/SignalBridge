import { useMemo, useState } from 'react'
import { useQueries, useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'

import { ApiError } from '../api/client'
import { getFinalSignals, getTranscripts } from '../api/transcripts'
import { StatusBadge } from '../components/StatusBadge'
import type { FinalSignal, Transcript } from '../types/transcript'

const dateFormatter = new Intl.DateTimeFormat('en-US', {
  dateStyle: 'medium',
  timeStyle: 'short',
})

type WorkflowGroup = 'needs-processing' | 'ready-for-review' | 'failed'
type StatusFilter = 'all' | WorkflowGroup
type SortOption = 'newest' | 'oldest' | 'filename-asc' | 'filename-desc' | 'status'

interface WorkflowStatus {
  group: WorkflowGroup
  label: string
  modifier: string
  action: string
}

const filterOptions: Array<{ value: StatusFilter; label: string }> = [
  { value: 'all', label: 'All' },
  { value: 'needs-processing', label: 'Needs processing' },
  { value: 'ready-for-review', label: 'Ready for review' },
  { value: 'failed', label: 'Failed' },
]

function formatDate(value: string): string {
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? 'Unknown date' : dateFormatter.format(date)
}

export function getWorkflowStatus(status: string): WorkflowStatus {
  const normalized = status.toLowerCase().replace(/[s-]+/g, '_')

  if (normalized.includes('fail') || normalized.includes('error')) {
    return {
      group: 'failed',
      label: 'Failed',
      modifier: 'failed',
      action: 'Review issue',
    }
  }

  if (normalized === 'finalized') {
    return {
      group: 'ready-for-review',
      label: 'Ready for review',
      modifier: 'finalized',
      action: 'Review results',
    }
  }

  if (
    normalized.includes('processing') ||
    normalized.includes('running') ||
    normalized.includes('in_progress')
  ) {
    return {
      group: 'needs-processing',
      label: 'Processing',
      modifier: 'processing',
      action: 'View progress',
    }
  }

  return {
    group: 'needs-processing',
    label: 'Needs processing',
    modifier: 'ready',
    action: 'Review & process',
  }
}

function dateValue(value: string): number {
  const parsed = new Date(value).getTime()
  return Number.isNaN(parsed) ? 0 : parsed
}

export function sortTranscripts(
  transcripts: Transcript[],
  sort: SortOption,
): Transcript[] {
  const items = [...transcripts]
  const filename = (left: Transcript, right: Transcript) =>
    left.file_name.localeCompare(right.file_name, undefined, {
      sensitivity: 'base',
      numeric: true,
    })

  return items.sort((left, right) => {
    if (sort === 'oldest') return dateValue(left.created_at) - dateValue(right.created_at)
    if (sort === 'filename-asc') return filename(left, right)
    if (sort === 'filename-desc') return filename(right, left)
    if (sort === 'status') {
      const priority: Record<WorkflowGroup, number> = {
        failed: 0,
        'needs-processing': 1,
        'ready-for-review': 2,
      }
      const difference =
        priority[getWorkflowStatus(left.status).group] -
        priority[getWorkflowStatus(right.status).group]
      return difference || dateValue(right.created_at) - dateValue(left.created_at)
    }
    return dateValue(right.created_at) - dateValue(left.created_at)
  })
}

function SignalCount({
  value,
  loading,
  available,
}: {
  value: number
  loading: boolean
  available: boolean
}) {
  if (loading) return <span className="queue-count queue-count--loading">…</span>
  if (!available) return <span className="queue-count">—</span>
  return <span className="queue-count">{value}</span>
}

export function Dashboard() {
  const [search, setSearch] = useState('')
  const [statusFilter, setStatusFilter] = useState<StatusFilter>('all')
  const [sort, setSort] = useState<SortOption>('newest')
  const transcriptsQuery = useQuery({
    queryKey: ['transcripts'],
    queryFn: getTranscripts,
  })

  const transcripts = transcriptsQuery.data ?? []
  const signalQueries = useQueries({
    queries: transcripts.map((transcript) => ({
      queryKey: ['final-signals', transcript.id],
      queryFn: () => getFinalSignals(transcript.id),
      enabled: transcript.status === 'finalized',
      staleTime: 30_000,
    })),
  })
  const signalCounts = new Map(transcripts.map((transcript, index) => {
    const query = signalQueries[index]
    const signals = query?.data ?? []
    return [transcript.id, {
      drivers: signals.filter((signal) => signal.item_type === 'driver').length,
      blockers: signals.filter((signal) => signal.item_type === 'blocker').length,
      loading: transcript.status === 'finalized' && Boolean(query?.isLoading),
      available: transcript.status === 'finalized' && Boolean(query?.isSuccess),
      strongest: signals.filter((signal) => signal.validator_verdict === 'pass').reduce<FinalSignal | null>((best, signal) => !best || (signal.final_score ?? -Infinity) > (best.final_score ?? -Infinity) ? signal : best, null),
    }]
  }))
  const counts = useMemo(() => {
    const summary = {
      all: transcripts.length,
      'needs-processing': 0,
      'ready-for-review': 0,
      failed: 0,
    }
    transcripts.forEach((transcript) => {
      summary[getWorkflowStatus(transcript.status).group] += 1
    })
    return summary
  }, [transcripts])

  const visibleTranscripts = useMemo(() => {
    const query = search.trim().toLocaleLowerCase()
    const filtered = transcripts.filter((transcript) => {
      const matchesSearch =
        !query || transcript.file_name.toLocaleLowerCase().includes(query)
      const matchesStatus =
        statusFilter === 'all' ||
        getWorkflowStatus(transcript.status).group === statusFilter
      return matchesSearch && matchesStatus
    })
    return sortTranscripts(filtered, sort)
  }, [search, sort, statusFilter, transcripts])

  const hasActiveFilters = search.trim().length > 0 || statusFilter !== 'all'

  function clearFilters() {
    setSearch('')
    setStatusFilter('all')
  }

  return (
    <section aria-labelledby="dashboard-title">
      <div className="page-header">
        <div className="page-header__content">
          <p className="eyebrow">AI meeting intelligence</p>
          <h2 className="page-title" id="dashboard-title">Transcript insights</h2>
          <p className="page-description">
            Review the business insights AI discovered across your transcripts.
          </p>
        </div>
      </div>

      {transcriptsQuery.isLoading ? (
        <div className="skeleton-table" role="status" aria-label="Loading transcript workspace">
          <span className="visually-hidden">Loading transcripts</span>
          <div className="skeleton skeleton--toolbar" />
          {[0, 1, 2, 3].map((row) => <div className="skeleton skeleton--row" key={row} />)}
        </div>
      ) : null}

      {transcriptsQuery.isError ? (
        <div className="state-card error-state alert alert--error" role="alert">
          <div>
            <h3>Unable to load transcripts</h3>
            <p>
              {transcriptsQuery.error instanceof ApiError
                ? transcriptsQuery.error.message
                : 'An unexpected error occurred while loading transcripts.'}
            </p>
          </div>
          <button type="button" onClick={() => transcriptsQuery.refetch()}>
            Try again
          </button>
        </div>
      ) : null}

      {transcriptsQuery.isSuccess && transcripts.length === 0 ? (
        <div className="dashboard-empty empty-state surface">
          <p className="eyebrow">Start with a transcript</p>
          <h3>No transcripts yet</h3>
          <p>
            Upload a UTF-8 .txt transcript, review its speaker turns, and run the
            pipeline to generate advisor drivers and blockers.
          </p>
          <Link className="button button--primary" to="/upload">
            Upload transcript
          </Link>
        </div>
      ) : null}

      {transcriptsQuery.isSuccess && transcripts.length > 0 ? (
        <>
          <section className="dashboard-work-queue" aria-labelledby="work-queue-title">
            <div className="section-header dashboard-list-header">
              <div>
                <h3 id="work-queue-title">Transcripts</h3>
                <p className="metadata-text">
                  {visibleTranscripts.length} of {counts.all} transcripts
                </p>
              </div>
            </div>

            <div className="dashboard-controls surface">
              <div className="search-field">
                <label htmlFor="transcript-search">Search</label>
                <div className="search-field__control">
                  <input
                    id="transcript-search"
                    type="search"
                    value={search}
                    placeholder="Search transcripts"
                    onChange={(event) => setSearch(event.target.value)}
                  />
                  {search ? (
                    <button
                      className="button button--ghost search-clear"
                      type="button"
                      onClick={() => setSearch('')}
                    >
                      Clear
                    </button>
                  ) : null}
                </div>
              </div>

              <fieldset className="status-filters">
                <legend>Status</legend>
                <div className="status-filters__options">
                  {filterOptions.map((option) => (
                    <button
                      key={option.value}
                      className="filter-button"
                      type="button"
                      aria-pressed={statusFilter === option.value}
                      onClick={() => setStatusFilter(option.value)}
                    >
                      {option.label} <span>{counts[option.value]}</span>
                    </button>
                  ))}
                </div>
              </fieldset>

              <div className="sort-field">
                <label htmlFor="transcript-sort">Sort by</label>
                <select
                  id="transcript-sort"
                  value={sort}
                  onChange={(event) => setSort(event.target.value as SortOption)}
                >
                  <option value="newest">Newest first</option>
                  <option value="oldest">Oldest first</option>
                  <option value="filename-asc">Filename A-Z</option>
                  <option value="filename-desc">Filename Z-A</option>
                  <option value="status">Status priority</option>
                </select>
              </div>
            </div>

            {visibleTranscripts.length === 0 ? (
              <div className="dashboard-empty empty-state">
                <h3>No transcripts match</h3>
                <p>Try a different search or clear the current filters.</p>
                <button className="button button--secondary" type="button" onClick={clearFilters}>
                  Clear filters
                </button>
              </div>
            ) : (
              <>
                <div className="table-card dashboard-table">
                  <div className="table-scroll">
                    <table>
                      <thead>
                        <tr>
                          <th scope="col">Transcript</th>
                          <th scope="col">Status</th>
                          <th scope="col">Drivers</th>
                          <th scope="col">Blockers</th>
                          <th scope="col">Updated</th>
                          <th scope="col"><span className="visually-hidden">Open</span></th>
                        </tr>
                      </thead>
                      <tbody>
                        {visibleTranscripts.map((transcript) => {
                          const workflow = getWorkflowStatus(transcript.status)
                          return (
                            <tr key={transcript.id}>
                              <td>
                                <Link className="file-name file-link" to={`/transcripts/${transcript.id}`}>
                                  {transcript.file_name}
                                </Link>
                                {signalCounts.get(transcript.id)?.strongest ? (
                                  <p className="dashboard-strongest-signal">
                                    <span>{signalCounts.get(transcript.id)?.strongest?.item_type}</span>
                                    {signalCounts.get(transcript.id)?.strongest?.category}
                                  </p>
                                ) : null}
                              </td>
                              <td><StatusBadge value={transcript.status} label={workflow.label} /></td>
                              <td><SignalCount value={signalCounts.get(transcript.id)?.drivers ?? 0} loading={signalCounts.get(transcript.id)?.loading ?? false} available={signalCounts.get(transcript.id)?.available ?? false} /></td>
                              <td><SignalCount value={signalCounts.get(transcript.id)?.blockers ?? 0} loading={signalCounts.get(transcript.id)?.loading ?? false} available={signalCounts.get(transcript.id)?.available ?? false} /></td>
                              <td>{formatDate(transcript.updated_at)}</td>
                              <td>
                                <Link className="queue-open-link" aria-label={`${workflow.action}: ${transcript.file_name}`} to={`/transcripts/${transcript.id}`}>
                                  Open <span aria-hidden="true">→</span>
                                </Link>
                              </td>
                            </tr>
                          )
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>

              </>
            )}
          </section>
        </>
      ) : null}
    </section>
  )
}
