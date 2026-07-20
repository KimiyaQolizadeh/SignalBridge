"""Hard evidence failures that cannot enter the business-signal pipeline."""

SCORING_ELIGIBLE_VERDICTS = frozenset({"pass", "needs_review"})
DEDUP_ELIGIBLE_VERDICTS = frozenset({"pass", "needs_review"})
FINAL_RANKING_ELIGIBLE_VERDICTS = frozenset({"pass", "needs_review"})

HARD_VALIDATION_FAILURE_REASONS = frozenset({
    "malformed_candidate",
    "quote_not_found",
    "quote_absent",
    "source_turn_missing",
    "not_advisor_side",
    "ownership_failure",
    "context_failure",
    "rationale_type_contradiction",
    "direct_contradiction",
})


def validation_allows_business_pipeline(
    verdict: str | None, reason: str | None
) -> bool:
    """Validation annotates unless it found a hard evidence integrity failure."""
    return not (verdict == "reject" and reason in HARD_VALIDATION_FAILURE_REASONS)
