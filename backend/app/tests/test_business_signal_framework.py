import pytest

from backend.app.llm_schemas import EvidenceValidationOutput
from backend.app.models import CandidateSignal
from backend.app.services.evidence_validator import (
    correct_candidate_direction,
    derive_validation_decision,
)


def candidate(item_type: str, quote: str = "Synthetic advisor evidence.") -> CandidateSignal:
    return CandidateSignal(
        item_type=item_type,
        category="Synthetic factor",
        advisor_quote=quote,
        rationale="The advisor states a grounded effect on the movement decision.",
        evidence_strength="explicit",
        source_turn_ids=[1],
    )


def findings(item_type: str, **overrides: object) -> EvidenceValidationOutput:
    values: dict[str, object] = {
        "quote_traceability": "exact",
        "source_turn_match": "exact",
        "advisor_ownership": "reliable_advisor",
        "context_sufficiency": "sufficient",
        "context_scope": "local_advisor_thought",
        "decision_relevance": "material",
        "supported_decision_effect": (
            "increases_move_likelihood"
            if item_type == "driver"
            else "decreases_move_likelihood"
        ),
        "direction_support": (
            "supports_driver" if item_type == "driver" else "supports_blocker"
        ),
        "validated_evidence_strength": "explicit",
        "rationale_grounding": "fully_grounded",
        "unsupported_rationale_claims": [],
        "representative_dependency": "independent",
        "procedural_status": "substantive_factor",
        "unsupported_escalations": [],
        "contradiction_status": "none",
        "semantic_consistency": "consistent",
        "consistency_issues": [],
    }
    values.update(overrides)
    return EvidenceValidationOutput(**values)


@pytest.mark.parametrize(
    ("name", "item_type"),
    [
        ("current-firm administration reduces client time", "driver"),
        ("values misalignment motivates a search", "driver"),
        ("improved client offering motivates movement", "driver"),
        ("technology frees the advisor team", "driver"),
        ("current-firm restrictions motivate communication flexibility", "driver"),
        ("eligibility threshold concern", "blocker"),
        ("partner approval dependency", "blocker"),
        ("transition complexity concern", "blocker"),
        ("client attrition concern", "blocker"),
        ("contractual restriction", "blocker"),
        ("performance proof required before proceeding", "blocker"),
        ("Optimize conflicts with the core business model", "blocker"),
    ],
)
def test_grounded_directional_business_factors_pass(
    name: str, item_type: str
) -> None:
    result = derive_validation_decision(findings(item_type), candidate(item_type))
    assert result.verdict == "pass", name


@pytest.mark.parametrize(
    "name",
    [
        "returns are important",
        "technology should be compared",
        "compliance is discussed",
        "general industry observation",
        "existing-practice description",
        "polite interest",
        "clarification question",
        "scheduling",
        "representative claim with minimal acknowledgement",
        "hypothetical product benefit without a decision effect",
    ],
)
@pytest.mark.parametrize("item_type", ["driver", "blocker"])
def test_neutral_evaluation_topics_cannot_become_signals(
    name: str, item_type: str
) -> None:
    output = findings(
        item_type,
        decision_relevance="none",
        supported_decision_effect="neutral",
        direction_support="unsupported",
        validated_evidence_strength="unsupported",
        rationale_grounding="unsupported",
    )
    result = derive_validation_decision(output, candidate(item_type))
    assert result.verdict == "reject", name


@pytest.mark.parametrize(
    "resolution",
    [
        "dedicated support already handles the apparent burden",
        "the advisor says the operating model works well",
        "later same-turn language contradicts the concern",
    ],
)
def test_resolved_or_contradicted_concern_rejects(resolution: str) -> None:
    output = findings(
        "blocker",
        context_sufficiency="contradictory",
        supported_decision_effect="neutral",
        direction_support="unsupported",
        rationale_grounding="contradicts_evidence",
        contradiction_status="direct_contradiction",
    )
    result = derive_validation_decision(output, candidate("blocker"))
    assert result.verdict == "reject", resolution


def test_explicit_conditional_gate_is_a_blocker() -> None:
    result = derive_validation_decision(
        findings("blocker"),
        candidate("blocker", "I can proceed only if my partner approves."),
    )
    assert result.verdict == "pass"


def test_credible_conditional_benefit_can_remain_needs_review() -> None:
    output = findings(
        "driver",
        decision_relevance="weak",
        supported_decision_effect="indeterminate",
        validated_evidence_strength="weakly_implied",
        rationale_grounding="partially_grounded",
    )
    result = derive_validation_decision(
        output,
        candidate("driver", "If the platform saves my team time, that could help a move."),
    )
    assert result.verdict == "needs_review"


@pytest.mark.parametrize(
    ("original", "effect", "expected"),
    [
        ("blocker", "increases_move_likelihood", "driver"),
        ("driver", "decreases_move_likelihood", "blocker"),
    ],
)
def test_structured_direction_correction(
    original: str, effect: str, expected: str
) -> None:
    item = candidate(original)
    output = findings(
        original,
        supported_decision_effect=effect,
        direction_support="contradicts_candidate_type",
    )
    assert correct_candidate_direction(output, item) == original
    assert item.item_type == expected
    assert derive_validation_decision(output, item).verdict == "pass"


def test_representative_led_claim_is_never_advisor_adoption() -> None:
    output = findings(
        "driver",
        advisor_ownership="representative",
        representative_dependency="fully_dependent",
        supported_decision_effect="neutral",
        direction_support="unsupported",
        rationale_grounding="unsupported",
        unsupported_escalations=["representative_claim_to_advisor_claim"],
    )
    assert derive_validation_decision(output, candidate("driver")).verdict == "reject"
