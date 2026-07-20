export interface Transcript {
  id: number
  file_name: string
  status: string
  token_count: number
  created_at: string
  updated_at: string
}

export interface TranscriptUploadResponse {
  id: number
  file_name: string
  status: string
  token_count: number
}

export interface TranscriptDetail extends Transcript {
  raw_text: string
}

export type SpeakerRole = 'advisor' | 'optimize_rep' | 'mixed' | 'unknown'

export interface TranscriptTurn {
  id: number
  turn_index: number
  timestamp: string | null
  raw_speaker_label: string | null
  inferred_role: SpeakerRole | null
  role_confidence: number | null
  text: string
}

export interface PipelineStep {
  name: string
  status: string
  details: Record<string, unknown>
}

export interface PipelineRunResponse {
  transcript_id: number
  status: string
  steps: PipelineStep[]
  final_driver_count: number
  final_blocker_count: number
}

export interface AnalysisRun {
  run_id: string
  transcript_id: number
  status: string
  run_type: 'full' | 'replay_validation' | 'replay_downstream'
  source_run_id: string | null
  started_at: string
  completed_at: string | null
  failed_stage: string | null
  error_category: string | null
  summary: Record<string, number>
}

export type ProcessingState =
  | 'idle' | 'queued' | 'processing' | 'completed'
  | 'completed_without_results' | 'failed'
  | 'parsing' | 'classifying_speakers' | 'extracting_candidates'
  | 'validating_evidence' | 'scoring_signals' | 'deduplicating' | 'reranking'

export interface ProcessingStatus {
  transcript_id: number
  run_id: string | null
  current_stage: string
  status: ProcessingState
  started_at: string | null
  updated_at: string | null
  completed_at: string | null
  elapsed_seconds: number
  error_category: string | null
}

export interface DiagnosticsStage {
  stage: string
  model: string | null
  call_count: number
  retry_count: number
  input_tokens: number | null
  output_tokens: number | null
  reasoning_tokens: number | null
  cached_input_tokens: number | null
  total_tokens: number | null
  estimated_cost: string | null
  duration_ms: number
  status: string
}

export interface PipelineDiagnostics {
  transcript_id: number
  run_id: string
  status: string
  started_at: string
  completed_at: string | null
  total_duration_ms: number
  total_call_count: number
  total_retry_count: number
  total_tokens: number | null
  total_estimated_cost: string | null
  stages: DiagnosticsStage[]
  prompt_provenance: Array<{ prompt_file_name: string; sha256: string }>
  embedding_model: string
  scoring_policy: Record<string, number>
  deduplication_threshold: number
  reranker_fallback: boolean | null
  failed_stage: string | null
  error_category: string | null
}

export type SignalType = 'driver' | 'blocker'

export interface EvidenceContext {
  turn_id: number
  timestamp: string | null
  speaker: string | null
  text: string
}

export interface FinalSignal {
  transcript_id: number
  item_type: SignalType
  rank: number
  category: string
  advisor_quote: string
  timestamp: string | null
  evidence_strength: string | null
  rationale: string
  final_score: number | null
  validator_verdict: string | null
  supporting_evidence: string[]
  evidence_context: EvidenceContext[]
}

export interface DiagnosticCandidate {
  id: number
  item_type: SignalType
  category: string
  advisor_quote: string
  timestamp: string | null
  evidence_strength: string | null
  rationale: string
  extraction_confidence: number | null
  source_turn_ids: number[] | null
  validator_verdict: string | null
  support_score: number | null
  advisor_side_score: number | null
  false_positive_risk: number | null
  rejection_reason: string | null
  advisor_ownership: number | null
  decision_impact: number | null
  explicitness: number | null
  urgency: number | null
  evidence_quality: number | null
  final_score: number | null
  duplicate_group_id: string | null
  is_canonical: boolean
}
