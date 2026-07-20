import pytest
from pydantic import ValidationError

from backend.app.llm_schemas import (
    BusinessScoreOutput,
    CandidateSignalItem,
    FinalSelectedSignal,
)


def valid_candidate_data() -> dict:
    return {
        "factor_kind": "desired_outcome",
        "decision_direction": "supports_move",
        "decision_relation": "The desired support increases willingness to move.",
        "item_type": "driver",
        "category": "platform_support",
        "advisor_quote": "The support would save me time.",
        "timestamp": "00:12:34",
        "evidence_strength": "explicit",
        "rationale": "The advisor identifies support as a reason to move forward.",
        "source_turn_ids": [12],
        "extraction_confidence": 0.9,
    }


def test_valid_candidate_signal_item() -> None:
    candidate = CandidateSignalItem(**valid_candidate_data())

    assert candidate.item_type == "driver"
    assert candidate.extraction_confidence == 0.9


def test_empty_advisor_quote_fails() -> None:
    data = valid_candidate_data()
    data["advisor_quote"] = "   "

    with pytest.raises(ValidationError):
        CandidateSignalItem(**data)


def test_invalid_item_type_fails() -> None:
    data = valid_candidate_data()
    data["item_type"] = "interest"

    with pytest.raises(ValidationError):
        CandidateSignalItem(**data)


def test_no_signal_allows_null_category_and_rationale() -> None:
    data = valid_candidate_data()
    data.update(
        factor_kind="procedural_diligence",
        decision_direction="neutral",
        decision_relation="Ordinary follow-up does not establish a decision factor.",
        item_type="no_signal",
        category=None,
        rationale=None,
    )

    candidate = CandidateSignalItem(**data)

    assert candidate.item_type == "no_signal"
    assert candidate.category is None


def test_signal_requires_category_and_rationale() -> None:
    data = valid_candidate_data()
    data["category"] = None

    with pytest.raises(ValidationError):
        CandidateSignalItem(**data)


def test_confidence_over_one_fails() -> None:
    data = valid_candidate_data()
    data["extraction_confidence"] = 1.01

    with pytest.raises(ValidationError):
        CandidateSignalItem(**data)


def test_rank_four_fails() -> None:
    with pytest.raises(ValidationError):
        FinalSelectedSignal(signal_id=1, item_type="driver", rank=4)


def valid_business_score_data() -> dict:
    return {
        "advisor_ownership": 5,
        "decision_impact": 4,
        "explicitness": 4,
        "urgency": 3,
        "evidence_quality": 5,
        "final_score": 4.2,
        "explanation": "Strong advisor ownership and high-quality evidence.",
    }


def test_valid_business_final_score_passes() -> None:
    score = BusinessScoreOutput(**valid_business_score_data())

    assert score.final_score == 4.2


def test_business_final_score_above_five_fails() -> None:
    data = valid_business_score_data()
    data["final_score"] = 5.5

    with pytest.raises(ValidationError):
        BusinessScoreOutput(**data)


def test_business_final_score_below_one_fails() -> None:
    data = valid_business_score_data()
    data["final_score"] = 0.5

    with pytest.raises(ValidationError):
        BusinessScoreOutput(**data)
