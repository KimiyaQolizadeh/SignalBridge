from datetime import datetime

from pydantic import BaseModel, ConfigDict


class TranscriptUploadResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    file_name: str
    status: str
    token_count: int | None


class TranscriptListItem(TranscriptUploadResponse):
    created_at: datetime
    updated_at: datetime


class TranscriptDetail(TranscriptListItem):
    raw_text: str


class DeleteResponse(BaseModel):
    status: str
    message: str


class ParseTranscriptResponse(BaseModel):
    transcript_id: int
    status: str
    turn_count: int


class TranscriptTurnResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    turn_index: int
    timestamp: str | None
    raw_speaker_label: str | None
    inferred_role: str | None
    role_confidence: float | None
    text: str


class SpeakerClassificationResponse(BaseModel):
    transcript_id: int
    status: str
    turn_count: int
    advisor_turns: int
    optimize_rep_turns: int
    unknown_turns: int
    mixed_turns: int


class CandidateExtractionResponse(BaseModel):
    transcript_id: int
    status: str
    candidate_count: int
    driver_candidates: int
    blocker_candidates: int


class CandidateSignalResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    item_type: str
    category: str
    advisor_quote: str
    timestamp: str | None
    evidence_strength: str | None
    rationale: str
    extraction_confidence: float | None
    source_turn_ids: list[int]


class EvidenceValidationResponse(BaseModel):
    transcript_id: int
    status: str
    candidate_count: int
    passed: int
    rejected: int
    needs_review: int


class CandidateSignalWithScoreResponse(CandidateSignalResponse):
    validator_verdict: str
    support_score: float | None
    advisor_side_score: float | None
    false_positive_risk: float | None
    rejection_reason: str | None
    advisor_ownership: int | None
    decision_impact: int | None
    explicitness: int | None
    urgency: int | None
    evidence_quality: int | None
    final_score: float | None
    duplicate_group_id: str | None
    is_canonical: bool


class SignalScoringResponse(BaseModel):
    transcript_id: int
    status: str
    candidate_count: int
    eligible_count: int
    scored_count: int
    rejected_skipped: int


class SignalDeduplicationResponse(BaseModel):
    transcript_id: int
    status: str
    candidate_count: int
    eligible_count: int
    canonical_count: int
    duplicate_count: int
    rejected_excluded: int


class FinalRerankingResponse(BaseModel):
    transcript_id: int
    status: str
    eligible_count: int
    final_driver_count: int
    final_blocker_count: int
    used_fallback: bool


class EvidenceContextResponse(BaseModel):
    turn_id: int
    timestamp: str | None
    speaker: str | None
    text: str


class FinalSignalResponse(BaseModel):
    transcript_id: int
    item_type: str
    rank: int
    category: str
    advisor_quote: str
    timestamp: str | None
    evidence_strength: str | None
    rationale: str
    final_score: float
    validator_verdict: str
    supporting_evidence: list[str]
    evidence_context: list[EvidenceContextResponse]


class PipelineStepResponse(BaseModel):
    name: str
    status: str
    details: dict


class PipelineRunResponse(BaseModel):
    transcript_id: int
    status: str
    steps: list[PipelineStepResponse]
    final_driver_count: int
    final_blocker_count: int


class ProcessingStatusResponse(BaseModel):
    transcript_id: int
    run_id: str | None
    current_stage: str
    status: str
    started_at: datetime | None
    updated_at: datetime | None
    completed_at: datetime | None
    elapsed_seconds: float
    error_category: str | None


class DiagnosticsStageResponse(BaseModel):
    stage: str
    model: str | None
    call_count: int
    retry_count: int
    input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    cached_input_tokens: int | None
    total_tokens: int | None
    estimated_cost: str | None
    duration_ms: float
    status: str


class DiagnosticsPromptResponse(BaseModel):
    prompt_file_name: str
    sha256: str


class PipelineDiagnosticsResponse(BaseModel):
    transcript_id: int
    run_id: str
    status: str
    started_at: datetime
    completed_at: datetime | None
    total_duration_ms: float
    total_call_count: int
    total_retry_count: int
    total_tokens: int | None
    total_estimated_cost: str | None
    stages: list[DiagnosticsStageResponse]
    prompt_provenance: list[DiagnosticsPromptResponse]
    embedding_model: str
    scoring_policy: dict[str, float]
    deduplication_threshold: float
    reranker_fallback: bool | None
    failed_stage: str | None
    error_category: str | None
