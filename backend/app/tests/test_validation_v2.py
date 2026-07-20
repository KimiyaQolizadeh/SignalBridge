from dataclasses import dataclass

import pytest

from backend.app.llm_schemas import EvidenceValidationOutput
from backend.app.services.evidence_validator import derive_validation_decision


@dataclass
class Candidate:
    item_type: str
    advisor_quote: str = "Synthetic advisor evidence."
    rationale: str = "The advisor states a material recruiting decision effect."


def findings(item_type: str = "driver", **overrides: object) -> EvidenceValidationOutput:
    values: dict[str, object] = {
        "quote_traceability": "exact", "source_turn_match": "exact",
        "advisor_ownership": "reliable_advisor", "ownership_basis": "turn_id=1",
        "context_sufficiency": "sufficient", "context_scope": "quote_only",
        "decision_relevance": "material",
        "supported_decision_effect": "increases_move_likelihood" if item_type == "driver" else "decreases_move_likelihood",
        "direction_support": "supports_driver" if item_type == "driver" else "supports_blocker",
        "validated_evidence_strength": "explicit", "rationale_grounding": "fully_grounded",
        "unsupported_rationale_claims": [], "representative_dependency": "independent",
        "procedural_status": "substantive_factor", "unsupported_escalations": [],
        "contradiction_status": "none", "semantic_consistency": "consistent",
        "consistency_issues": [], "support_score": 0.9, "advisor_side_score": 0.95,
        "false_positive_risk": 0.1, "explanation": "Synthetic structured finding.",
    }
    values.update(overrides)
    return EvidenceValidationOutput(**values)


# De-identified policy regression set. Defaults supply factor_kind=desired_outcome,
# decision_direction matching item_type, decision_relation=material effect, bounded
# quote-only advisor context, and the grounded rationale above.
CASES = [
    ("strong_product_value", "driver", {}, "pass"),
    ("long_term_alignment", "driver", {"validated_evidence_strength": "tightly_implied", "context_scope": "local_advisor_thought"}, "pass"),
    ("procedural_weekend", "driver", {"procedural_status": "procedural_only", "decision_relevance": "none", "supported_decision_effect": "neutral"}, "reject"),
    ("pain_without_transition", "driver", {"validated_evidence_strength": "weakly_implied", "rationale_grounding": "unsupported", "unsupported_escalations": ["pain_to_move"]}, "reject"),
    ("pain_with_transition", "driver", {}, "pass"),
    ("digital_preference_as_requirement", "blocker", {"unsupported_escalations": ["preference_to_requirement"]}, "reject"),
    ("digital_requirement", "blocker", {}, "pass"),
    ("salesforce_question", "blocker", {"unsupported_escalations": ["question_to_blocker"], "supported_decision_effect": "indeterminate"}, "reject"),
    ("salesforce_dependency", "blocker", {}, "pass"),
    ("representative_value", "driver", {"advisor_ownership": "representative", "representative_dependency": "fully_dependent"}, "reject"),
    ("explicit_commitment", "driver", {}, "pass"),
    ("interest_as_commitment", "driver", {"unsupported_escalations": ["interest_to_commitment"]}, "reject"),
    ("partner_discussion", "blocker", {"procedural_status": "procedural_only", "unsupported_escalations": ["discussion_to_approval_dependency"]}, "reject"),
    ("partner_approval", "blocker", {}, "pass"),
    ("scheduling_urgency", "driver", {"unsupported_escalations": ["scheduling_to_urgency"]}, "reject"),
    ("timing_dependency", "blocker", {"supported_decision_effect": "creates_timing_dependency", "direction_support": "supports_timing_blocker"}, "pass"),
    ("direct_contradiction", "driver", {"contradiction_status": "direct_contradiction"}, "reject"),
    ("weak_contradiction", "driver", {"contradiction_status": "unresolved"}, "needs_review"),
    ("acknowledgement", "driver", {"decision_relevance": "none", "supported_decision_effect": "neutral", "procedural_status": "not_applicable"}, "reject"),
    ("mixed_ownership", "driver", {"advisor_ownership": "mixed"}, "reject"),
    ("context_boundary", "driver", {"context_scope": "quote_only"}, "pass"),
    ("partially_grounded", "blocker", {"rationale_grounding": "partially_grounded"}, "needs_review"),
    ("structured_inconsistency", "driver", {"supported_decision_effect": "neutral", "unsupported_escalations": ["pain_to_move"]}, "reject"),
    ("missing_source", "driver", {"source_turn_match": "missing"}, "reject"),
    ("normalized_quote", "driver", {"quote_traceability": "normalized_exact"}, "pass"),
    ("partial_quote", "driver", {"quote_traceability": "partial"}, "needs_review"),
    ("absent_quote", "driver", {"quote_traceability": "absent"}, "reject"),
    ("ambiguous_source", "blocker", {"source_turn_match": "ambiguous"}, "needs_review"),
    ("unknown_owner", "driver", {"advisor_ownership": "unknown"}, "reject"),
    ("conflicting_owner", "blocker", {"advisor_ownership": "conflicting"}, "reject"),
    ("incomplete_context", "driver", {"context_sufficiency": "incomplete"}, "needs_review"),
    ("irrelevant_context", "driver", {"context_sufficiency": "irrelevant"}, "reject"),
    ("weak_relevance", "driver", {"decision_relevance": "weak"}, "needs_review"),
    ("direct_preference_uncertain_impact", "driver", {
        "decision_relevance": "weak",
        "supported_decision_effect": "indeterminate",
        "direction_support": "supports_driver",
        "validated_evidence_strength": "weakly_implied",
    }, "needs_review"),
    ("direct_constraint_uncertain_impact", "blocker", {
        "decision_relevance": "weak",
        "supported_decision_effect": "indeterminate",
        "direction_support": "supports_blocker",
        "validated_evidence_strength": "weakly_implied",
    }, "needs_review"),
    ("direct_preference_with_escalated_rationale", "driver", {
        "decision_relevance": "weak",
        "supported_decision_effect": "indeterminate",
        "direction_support": "supports_driver",
        "validated_evidence_strength": "weakly_implied",
        "rationale_grounding": "unsupported",
        "unsupported_escalations": ["interest_to_commitment"],
    }, "reject"),
    ("neutral_effect", "blocker", {"supported_decision_effect": "neutral"}, "reject"),
    ("wrong_driver_direction", "driver", {"direction_support": "supports_blocker", "supported_decision_effect": "decreases_move_likelihood"}, "reject"),
    ("wrong_blocker_direction", "blocker", {"direction_support": "supports_driver", "supported_decision_effect": "increases_move_likelihood"}, "reject"),
    ("unsupported_rationale", "driver", {"rationale_grounding": "unsupported", "unsupported_rationale_claims": ["move motivation"]}, "reject"),
    ("partial_representative", "driver", {"representative_dependency": "partially_dependent"}, "needs_review"),
    ("mixed_procedural", "blocker", {"procedural_status": "mixed_procedural_and_substantive"}, "needs_review"),
    ("current_state_as_condition", "blocker", {"unsupported_escalations": ["current_state_to_future_condition"]}, "reject"),
]


@pytest.mark.parametrize(("case_id", "item_type", "overrides", "expected"), CASES, ids=[case[0] for case in CASES])
def test_validation_v2_regression_policy(case_id: str, item_type: str, overrides: dict, expected: str) -> None:
    del case_id
    output = findings(item_type, **overrides)
    assert derive_validation_decision(output, Candidate(item_type)).verdict == expected


def test_model_verdict_cannot_override_structured_reject() -> None:
    output = findings("driver", verdict="pass", unsupported_escalations=["pain_to_move"])
    assert derive_validation_decision(output, Candidate("driver")).verdict == "reject"


def test_hard_reject_precedes_review() -> None:
    output = findings("driver", context_sufficiency="incomplete", rationale_grounding="unsupported")
    assert derive_validation_decision(output, Candidate("driver")).verdict == "reject"


def test_fully_grounded_with_unsupported_claim_is_inconsistent() -> None:
    output = findings("driver", unsupported_rationale_claims=["invented requirement"])
    decision = derive_validation_decision(output, Candidate("driver"))
    assert decision.verdict == "reject"
    assert "fully_grounded_with_unsupported_claims" in decision.consistency_issues
