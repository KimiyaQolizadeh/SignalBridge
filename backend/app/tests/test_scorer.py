from collections.abc import Generator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.app.database import Base
from backend.app.llm_schemas import BusinessScoreOutput
from backend.app.models import CandidateSignal, SignalScore, Transcript
from backend.app.services import scorer
from backend.app.services.scorer import score_signals_for_transcript


@pytest.fixture
def db() -> Generator[Session, None, None]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    Base.metadata.drop_all(engine)


def create_transcript(db: Session) -> Transcript:
    transcript = Transcript(file_name="synthetic.txt", raw_text="Synthetic content")
    db.add(transcript)
    db.flush()
    return transcript


def add_candidate(
    db: Session, transcript: Transcript, *, verdict: str | None
) -> CandidateSignal:
    candidate = CandidateSignal(
        transcript_id=transcript.id,
        item_type="driver",
        category="synthetic_category",
        advisor_quote="Synthetic advisor evidence.",
        timestamp="00:00:00",
        evidence_strength="explicit",
        rationale="Synthetic decision-relevant rationale.",
        extraction_confidence=0.9,
        source_turn_ids=[1],
    )
    db.add(candidate)
    db.flush()
    candidate.score = SignalScore(
        signal_id=candidate.id,
        validator_verdict=verdict,
        support_score=0.91,
        advisor_side_score=0.92,
        false_positive_risk=0.08,
        rejection_reason=None,
    )
    db.commit()
    return candidate


def business_output() -> BusinessScoreOutput:
    return BusinessScoreOutput(
        advisor_ownership=5,
        decision_impact=4,
        explicitness=3,
        urgency=2,
        evidence_quality=1,
        final_score=4.9,
        explanation="Synthetic component-score explanation.",
    )


def test_soft_rejected_candidates_are_scored_as_annotations(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    candidate = add_candidate(db, transcript, verdict="reject")
    monkeypatch.setattr(
        scorer,
        "call_llm_json",
        lambda **_: business_output(),
    )

    summary = score_signals_for_transcript(transcript.id, db)
    db.refresh(candidate.score)

    assert summary["scored_count"] == 1
    assert summary["rejected_skipped"] == 0
    assert candidate.score.final_score == 3.5


def test_hard_evidence_failure_is_skipped(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    candidate = add_candidate(db, transcript, verdict="reject")
    candidate.score.rejection_reason = "quote_not_found"
    db.commit()
    monkeypatch.setattr(scorer, "call_llm_json", lambda **_: pytest.fail())

    summary = score_signals_for_transcript(transcript.id, db)

    assert summary["scored_count"] == 0
    assert summary["rejected_skipped"] == 1


def test_pass_candidate_uses_deterministic_final_score(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    candidate = add_candidate(db, transcript, verdict="pass")
    monkeypatch.setattr(scorer, "call_llm_json", lambda **_: business_output())

    score_signals_for_transcript(transcript.id, db)
    db.refresh(candidate.score)

    assert candidate.score.final_score == 3.5
    assert candidate.score.final_score != business_output().final_score


def test_needs_review_candidate_is_scored(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    candidate = add_candidate(db, transcript, verdict="needs_review")
    monkeypatch.setattr(scorer, "call_llm_json", lambda **_: business_output())

    summary = score_signals_for_transcript(transcript.id, db)
    db.refresh(candidate.score)

    assert summary["eligible_count"] == 1
    assert candidate.score.advisor_ownership == 5


def test_unannotated_candidate_is_scored(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    add_candidate(db, transcript, verdict=None)
    monkeypatch.setattr(
        scorer,
        "call_llm_json",
        lambda **_: business_output(),
    )

    summary = score_signals_for_transcript(transcript.id, db)

    assert summary["eligible_count"] == 1
    assert summary["scored_count"] == 1


def test_existing_validator_fields_are_preserved(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    candidate = add_candidate(db, transcript, verdict="pass")
    original_fields = (
        candidate.score.validator_verdict,
        candidate.score.support_score,
        candidate.score.advisor_side_score,
        candidate.score.false_positive_risk,
    )
    monkeypatch.setattr(scorer, "call_llm_json", lambda **_: business_output())

    score_signals_for_transcript(transcript.id, db)
    db.refresh(candidate.score)

    assert (
        candidate.score.validator_verdict,
        candidate.score.support_score,
        candidate.score.advisor_side_score,
        candidate.score.false_positive_risk,
    ) == original_fields


def test_summary_counts_are_correct(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    add_candidate(db, transcript, verdict="pass")
    add_candidate(db, transcript, verdict="needs_review")
    add_candidate(db, transcript, verdict="reject")
    monkeypatch.setattr(scorer, "call_llm_json", lambda **_: business_output())

    summary = score_signals_for_transcript(transcript.id, db)

    assert summary == {
        "transcript_id": transcript.id,
        "status": "signals_scored",
        "candidate_count": 3,
        "eligible_count": 3,
        "scored_count": 3,
        "rejected_skipped": 0,
    }
