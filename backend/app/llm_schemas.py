from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


Probability = Annotated[float, Field(ge=0.0, le=1.0)]
BusinessComponentScore = Annotated[int, Field(ge=1, le=5)]
BusinessFinalScore = Annotated[float, Field(ge=1.0, le=5.0)]
SignalRank = Annotated[int, Field(ge=1, le=3)]


def reject_blank_text(value: str) -> str:
    if not value.strip():
        raise ValueError("text must not be empty")
    return value


class SpeakerRoleItem(BaseModel):
    turn_id: int
    inferred_role: Literal["advisor", "optimize_rep", "unknown", "mixed"]
    confidence: Probability
    reason: str


class SpeakerRoleBatchOutput(BaseModel):
    items: list[SpeakerRoleItem]


class CandidateSignalItem(BaseModel):
    factor_kind: Literal[
        "explicit_commitment",
        "current_pain",
        "desired_outcome",
        "future_condition",
        "future_risk",
        "incompatibility",
        "dependency",
        "procedural_diligence",
        "information_request",
        "factual_background",
        "preference",
        "unclear",
    ]
    decision_direction: Literal[
        "supports_move", "opposes_move", "affects_timing", "neutral"
    ]
    decision_relation: str
    item_type: Literal["driver", "blocker", "no_signal"]
    category: str | None
    advisor_quote: str
    timestamp: str | None
    evidence_strength: Literal["explicit", "implied"]
    rationale: str | None
    source_turn_ids: list[int]
    extraction_confidence: Probability

    _validate_required_text = field_validator(
        "advisor_quote", "decision_relation", mode="after"
    )(reject_blank_text)

    @model_validator(mode="after")
    def validate_signal_fields(self) -> "CandidateSignalItem":
        if self.item_type != "no_signal":
            if self.category is None or not self.category.strip():
                raise ValueError("signal category must not be empty")
            if self.rationale is None or not self.rationale.strip():
                raise ValueError("signal rationale must not be empty")
        return self


class CandidateSignalBatchOutput(BaseModel):
    items: list[CandidateSignalItem]


class EvidenceValidationOutput(BaseModel):
    """Internal semantic findings; application code owns the final verdict."""

    quote_traceability: Literal["exact", "normalized_exact", "partial", "absent"] | None = None
    source_turn_match: Literal["exact", "ambiguous", "missing"] | None = None
    advisor_ownership: Literal["reliable_advisor", "mixed", "representative", "unknown", "conflicting"] | None = None
    ownership_basis: str | None = None
    context_sufficiency: Literal["sufficient", "incomplete", "contradictory", "irrelevant"] | None = None
    context_scope: Literal["quote_only", "local_advisor_thought", "multi_turn_advisor_thought"] | None = None
    decision_relevance: Literal["material", "weak", "none"] | None = None
    supported_decision_effect: Literal["increases_move_likelihood", "decreases_move_likelihood", "creates_timing_dependency", "neutral", "indeterminate"] | None = None
    direction_support: Literal["supports_driver", "supports_blocker", "supports_timing_blocker", "contradicts_candidate_type", "unsupported"] | None = None
    validated_evidence_strength: Literal["explicit", "tightly_implied", "weakly_implied", "unsupported"] | None = None
    rationale_grounding: Literal["fully_grounded", "partially_grounded", "unsupported", "contradicts_evidence"] | None = None
    unsupported_rationale_claims: list[str] = Field(default_factory=list)
    representative_dependency: Literal["independent", "partially_dependent", "fully_dependent"] | None = None
    procedural_status: Literal["substantive_factor", "procedural_only", "mixed_procedural_and_substantive", "not_applicable"] | None = None
    unsupported_escalations: list[Literal["pain_to_move", "preference_to_requirement", "question_to_blocker", "possibility_to_condition", "representative_claim_to_advisor_claim", "scheduling_to_urgency", "interest_to_commitment", "discussion_to_approval_dependency", "current_state_to_future_condition", "other"]] = Field(default_factory=list)
    contradiction_status: Literal["none", "unresolved", "direct_contradiction"] | None = None
    contradiction_summary: str | None = None
    semantic_consistency: Literal["consistent", "internally_conflicting"] | None = None
    consistency_issues: list[str] = Field(default_factory=list)

    verdict: Literal["pass", "reject", "needs_review"] | None = None
    support_score: Probability = 0.5
    advisor_side_score: Probability = 0.5
    false_positive_risk: Probability = 0.5
    rejection_reason: str | None = None
    explanation: str = "Structured evidence validation findings."

    _validate_explanation = field_validator("explanation", mode="after")(
        reject_blank_text
    )


class BusinessScoreOutput(BaseModel):
    advisor_ownership: BusinessComponentScore
    decision_impact: BusinessComponentScore
    explicitness: BusinessComponentScore
    urgency: BusinessComponentScore
    evidence_quality: BusinessComponentScore
    final_score: BusinessFinalScore
    explanation: str

    _validate_explanation = field_validator("explanation", mode="after")(
        reject_blank_text
    )


class FinalSelectedSignal(BaseModel):
    signal_id: int
    item_type: Literal["driver", "blocker"]
    rank: SignalRank


class FinalRerankOutput(BaseModel):
    selected_signals: list[FinalSelectedSignal]
    explanation: str

    _validate_explanation = field_validator("explanation", mode="after")(
        reject_blank_text
    )


class ImportanceItemOutput(BaseModel):
    candidate_id: str
    importance_tier: Literal["critical", "high", "medium", "low", "peripheral", "indeterminate"]
    importance_score: Probability
    relative_rank: int = Field(ge=1)
    decision_centrality: Literal["decisive", "major", "contributing", "minor", "contextual", "indeterminate"]
    priority_expression: Literal["explicitly_ranked", "explicitly_required", "explicitly_emphasized", "repeatedly_emphasized", "implicitly_central", "mentioned_once", "unclear"]
    dependency_role: Literal["gating_condition", "timing_condition", "outcome_driver", "comparative_preference", "supporting_detail", "no_dependency", "indeterminate"]
    consequence_strength: Literal["decision_stopping", "materially_changes_decision", "affects_timing_or_terms", "influences_preference", "limited_effect", "no_stated_effect", "indeterminate"]
    attention_strength: Literal["dominant", "repeated", "developed", "brief", "incidental", "indeterminate"]
    importance_confidence: Probability
    importance_basis: str
    comparison_basis: str
    supporting_candidate_ids: list[str] = Field(default_factory=list)
    conflicting_candidate_ids: list[str] = Field(default_factory=list)
    importance_warnings: list[Literal["insufficient_comparison_set", "unresolved_validation", "duplicate_or_overlapping_factor", "possible_broad_narrow_overlap", "contradictory_priority", "importance_inferred_from_frequency_only", "importance_inferred_from_category_only", "representative_context_present", "incomplete_transcript_context", "order_sensitivity_detected", "score_tie", "other"]] = Field(default_factory=list)

    _validate_text = field_validator("candidate_id", "importance_basis", "comparison_basis", mode="after")(reject_blank_text)


class ImportanceListwiseOutput(BaseModel):
    items: list[ImportanceItemOutput]


class ImportancePairwiseItemOutput(BaseModel):
    candidate_a_id: str
    candidate_b_id: str
    relative_importance: Literal["a_more_important", "b_more_important", "approximately_equal", "indeterminate"]
    comparison_confidence: Probability
    comparison_basis: Literal["explicit_priority", "gating_dependency", "stronger_consequence", "greater_emphasis", "closer_to_final_decision", "broader_business_effect", "timing_criticality", "insufficient_difference", "conflicting_evidence", "other"]
    reason: str

    _validate_text = field_validator("candidate_a_id", "candidate_b_id", "reason", mode="after")(reject_blank_text)


class ImportancePairwiseOutput(BaseModel):
    items: list[ImportancePairwiseItemOutput]
