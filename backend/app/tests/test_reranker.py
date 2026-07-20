from collections.abc import Generator

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.app.database import Base
from backend.app.llm_schemas import FinalRerankOutput, FinalSelectedSignal
from backend.app.models import CandidateSignal, FinalSignal, SignalScore, Transcript, TranscriptTurn
from backend.app.services import reranker
from backend.app.services.reranker import rerank_final_signals_for_transcript


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
    db: Session,
    transcript: Transcript,
    *,
    item_type: str = "driver",
    verdict: str = "pass",
    is_canonical: bool = True,
    evidence_strength: str = "explicit",
    final_score: float = 4.0,
    support_score: float = 0.9,
    advisor_side_score: float = 0.9,
    category: str | None = None,
    quote: str | None = None,
    rationale: str = "Synthetic decision-relevant rationale.",
) -> CandidateSignal:
    candidate = CandidateSignal(
        transcript_id=transcript.id,
        item_type=item_type,
        category=category or f"category_{item_type}_{final_score}",
        advisor_quote=quote or f"I need this {item_type} factor to move forward.",
        timestamp="00:00:00",
        evidence_strength=evidence_strength,
        rationale=rationale,
        extraction_confidence=0.9,
        source_turn_ids=[1],
        duplicate_group_id="g_synthetic",
        is_canonical=is_canonical,
    )
    db.add(candidate)
    db.flush()
    candidate.score = SignalScore(
        signal_id=candidate.id,
        validator_verdict=verdict,
        support_score=support_score,
        advisor_side_score=advisor_side_score,
        false_positive_risk=0.1,
        advisor_ownership=4,
        decision_impact=4,
        explicitness=4,
        urgency=3,
        evidence_quality=4,
        final_score=final_score,
    )
    db.commit()
    return candidate


def saved_final_signals(db: Session, transcript_id: int) -> list[FinalSignal]:
    return list(
        db.scalars(
            select(FinalSignal)
            .where(FinalSignal.transcript_id == transcript_id)
            .order_by(FinalSignal.item_type, FinalSignal.rank)
        ).all()
    )


def test_zero_eligible_finalizes_without_llm(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    add_candidate(db, transcript, verdict="pass", final_score=2.99)
    monkeypatch.setattr(
        reranker,
        "call_llm_json",
        lambda **_: pytest.fail("LLM must not be called"),
    )

    summary = rerank_final_signals_for_transcript(transcript.id, db)

    assert summary["eligible_count"] == 0
    assert summary["used_fallback"] is False
    assert saved_final_signals(db, transcript.id) == []


def test_rejected_candidate_is_not_eligible(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    add_candidate(db, transcript, verdict="reject", final_score=5.0)
    monkeypatch.setattr(reranker, "call_llm_json", lambda **_: pytest.fail())

    summary = rerank_final_signals_for_transcript(transcript.id, db)

    assert summary["eligible_count"] == 0


def test_hard_evidence_failure_is_not_final_eligible(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    candidate = add_candidate(db, transcript, verdict="reject", final_score=5.0)
    candidate.score.rejection_reason = "not_advisor_side"
    db.commit()
    monkeypatch.setattr(reranker, "call_llm_json", lambda **_: pytest.fail())

    assert rerank_final_signals_for_transcript(transcript.id, db)["eligible_count"] == 0


def test_noncanonical_candidate_is_not_eligible(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    add_candidate(db, transcript, is_canonical=False, final_score=5.0)
    monkeypatch.setattr(reranker, "call_llm_json", lambda **_: pytest.fail())

    summary = rerank_final_signals_for_transcript(transcript.id, db)

    assert summary["eligible_count"] == 0


def test_explicit_candidate_requires_three_point_five(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    add_candidate(db, transcript, evidence_strength="explicit", final_score=3.49)
    monkeypatch.setattr(reranker, "call_llm_json", lambda **_: pytest.fail())

    assert rerank_final_signals_for_transcript(transcript.id, db)["eligible_count"] == 0


def test_implied_candidate_requires_four(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    add_candidate(db, transcript, evidence_strength="implied", final_score=3.99)
    monkeypatch.setattr(reranker, "call_llm_json", lambda **_: pytest.fail())

    assert rerank_final_signals_for_transcript(transcript.id, db)["eligible_count"] == 0


def test_valid_llm_selection_saves_final_signals(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    driver = add_candidate(db, transcript, item_type="driver", final_score=4.5)
    blocker = add_candidate(db, transcript, item_type="blocker", final_score=4.2)
    monkeypatch.setattr(
        reranker,
        "call_llm_json",
        lambda **_: FinalRerankOutput(
            selected_signals=[
                FinalSelectedSignal(signal_id=driver.id, item_type="driver", rank=1),
                FinalSelectedSignal(signal_id=blocker.id, item_type="blocker", rank=1),
            ],
            explanation="Synthetic final selection.",
        ),
    )

    summary = rerank_final_signals_for_transcript(transcript.id, db)
    saved = saved_final_signals(db, transcript.id)

    assert summary["used_fallback"] is False
    assert {item.signal_id for item in saved} == {driver.id, blocker.id}


def test_invalid_selected_id_uses_deterministic_fallback(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    stronger = add_candidate(db, transcript, final_score=4.5, category="stronger")
    add_candidate(db, transcript, final_score=4.0, category="weaker")
    monkeypatch.setattr(
        reranker,
        "call_llm_json",
        lambda **_: FinalRerankOutput(
            selected_signals=[
                FinalSelectedSignal(signal_id=999999, item_type="driver", rank=1)
            ],
            explanation="Synthetic invalid selection.",
        ),
    )

    summary = rerank_final_signals_for_transcript(transcript.id, db)
    saved = saved_final_signals(db, transcript.id)

    assert summary["used_fallback"] is True
    assert saved[0].signal_id == stronger.id


def test_max_three_drivers_and_blockers_are_saved(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    for index in range(4):
        add_candidate(
            db,
            transcript,
            item_type="driver",
            final_score=4.9 - index * 0.1,
            category=f"driver_{index}",
        )
        add_candidate(
            db,
            transcript,
            item_type="blocker",
            final_score=4.9 - index * 0.1,
            category=f"blocker_{index}",
        )
    monkeypatch.setattr(
        reranker,
        "call_llm_json",
        lambda **_: FinalRerankOutput(
            selected_signals=[
                FinalSelectedSignal(signal_id=999999, item_type="driver", rank=1)
            ],
            explanation="Force deterministic fallback.",
        ),
    )

    summary = rerank_final_signals_for_transcript(transcript.id, db)

    assert summary["final_driver_count"] == 3
    assert summary["final_blocker_count"] == 3


def test_final_signals_have_consecutive_ranks(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    for index in range(3):
        add_candidate(
            db,
            transcript,
            final_score=4.8 - index * 0.1,
            category=f"driver_{index}",
        )
    monkeypatch.setattr(
        reranker,
        "call_llm_json",
        lambda **_: FinalRerankOutput(
            selected_signals=[
                FinalSelectedSignal(signal_id=999999, item_type="driver", rank=1)
            ],
            explanation="Force deterministic fallback.",
        ),
    )

    rerank_final_signals_for_transcript(transcript.id, db)
    saved = saved_final_signals(db, transcript.id)

    assert [item.rank for item in saved] == [1, 2, 3]


def test_needs_review_candidate_can_become_final(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    add_candidate(
        db, transcript, item_type="driver", category="Generic Interest",
        final_score=5.0, verdict="needs_review",
    )
    monkeypatch.setattr(
        reranker, "call_llm_json",
        lambda **_: FinalRerankOutput(
            selected_signals=[FinalSelectedSignal(signal_id=1, item_type="driver", rank=1)],
            explanation="Select the strongest scored signal.",
        ),
    )
    summary = rerank_final_signals_for_transcript(transcript.id, db)
    assert summary["final_driver_count"] == 1
    assert len(saved_final_signals(db, transcript.id)) == 1


def test_below_threshold_grounded_driver_uses_review_fallback(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    candidate = add_candidate(
        db, transcript, verdict="needs_review", final_score=3.45,
        evidence_strength="explicit",
    )
    monkeypatch.setattr(reranker, "call_llm_json", lambda **_: pytest.fail())

    summary = rerank_final_signals_for_transcript(transcript.id, db)

    assert summary["final_driver_count"] == 1
    assert summary["selection_reasons"] == {candidate.id: "best_grounded_fallback"}
    assert saved_final_signals(db, transcript.id)[0].signal.score.validator_verdict == "needs_review"


def test_multiple_below_threshold_drivers_select_only_highest(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    highest = add_candidate(db, transcript, verdict="needs_review", final_score=3.45)
    add_candidate(db, transcript, verdict="needs_review", final_score=3.2)
    monkeypatch.setattr(reranker, "call_llm_json", lambda **_: pytest.fail())

    summary = rerank_final_signals_for_transcript(transcript.id, db)

    saved = saved_final_signals(db, transcript.id)
    assert [item.signal_id for item in saved] == [highest.id]
    assert summary["selection_reasons"] == {highest.id: "best_grounded_fallback"}


def test_rejected_below_threshold_candidate_does_not_fallback(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    add_candidate(db, transcript, verdict="reject", final_score=3.45)
    monkeypatch.setattr(reranker, "call_llm_json", lambda **_: pytest.fail())

    summary = rerank_final_signals_for_transcript(transcript.id, db)

    assert summary["final_driver_count"] == 0
    assert summary["selection_reasons"] == {}


def test_no_grounded_candidates_remains_empty(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    add_candidate(db, transcript, verdict="reject", final_score=2.0)
    monkeypatch.setattr(reranker, "call_llm_json", lambda **_: pytest.fail())

    assert rerank_final_signals_for_transcript(transcript.id, db)["final_driver_count"] == 0


def test_above_threshold_selection_reason_is_unchanged(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    candidate = add_candidate(db, transcript, verdict="pass", final_score=4.2)
    monkeypatch.setattr(
        reranker,
        "call_llm_json",
        lambda **_: FinalRerankOutput(
            selected_signals=[FinalSelectedSignal(signal_id=candidate.id, item_type="driver", rank=1)],
            explanation="Select the preferred candidate.",
        ),
    )

    summary = rerank_final_signals_for_transcript(transcript.id, db)

    assert summary["selection_reasons"] == {candidate.id: "above_threshold"}
@pytest.mark.parametrize(
    "quote",
    [
        "Everyone is looking for a way to take back-end work off their plate.",
        "Every advisor wants more time with clients.",
    ],
)
def test_general_observation_needs_review_does_not_fallback(
    db: Session, monkeypatch: pytest.MonkeyPatch, quote: str
) -> None:
    transcript = create_transcript(db)
    add_candidate(db, transcript, verdict="needs_review", final_score=3.45, quote=quote)
    monkeypatch.setattr(reranker, "call_llm_json", lambda **_: pytest.fail())

    summary = rerank_final_signals_for_transcript(transcript.id, db)

    assert summary["final_driver_count"] == 0
    assert saved_final_signals(db, transcript.id) == []


def test_conditional_hypothetical_benefit_does_not_fallback(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    add_candidate(
        db, transcript, verdict="needs_review", final_score=3.45,
        quote="If it works, it could free up time for me.",
    )
    monkeypatch.setattr(reranker, "call_llm_json", lambda **_: pytest.fail())

    assert rerank_final_signals_for_transcript(transcript.id, db)["final_driver_count"] == 0


def test_driver_and_blocker_fallback_gates_are_independent(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    blocker = add_candidate(
        db, transcript, item_type="blocker", verdict="needs_review", final_score=3.4,
        quote="I cannot proceed unless my partner approves.",
    )
    add_candidate(
        db, transcript, item_type="driver", verdict="needs_review", final_score=3.45,
        quote="Everyone wants more support.",
    )
    monkeypatch.setattr(reranker, "call_llm_json", lambda **_: pytest.fail())

    summary = rerank_final_signals_for_transcript(transcript.id, db)

    assert summary["final_driver_count"] == 0
    assert summary["final_blocker_count"] == 1
    assert summary["selection_reasons"] == {blocker.id: "best_grounded_fallback"}


def test_transcript_three_general_candidate_is_not_fallback() -> None:
    candidate = CandidateSignal(
        item_type="driver", category="Client-Facing Time",
        advisor_quote="Everyone is looking for a way to take some of the back end off of their plate, so they can get out there and get in front of more clients and prospects.",
        rationale="The advisor states a general goal of reducing back-end work.",
        evidence_strength="explicit", is_canonical=True, source_turn_ids=[1],
    )
    candidate.score = SignalScore(
        validator_verdict="needs_review", rejection_reason="weak_relevance",
        advisor_ownership=4, decision_impact=3, explicitness=4, urgency=2,
        evidence_quality=4, final_score=3.45,
    )

    assert reranker._is_fallback_eligible(candidate) is False

def context_candidate(item_type: str, quote: str, category: str = "Context") -> CandidateSignal:
    candidate = CandidateSignal(
        item_type=item_type, category=category, advisor_quote=quote,
        rationale="The evidence affects movement.", evidence_strength="explicit",
        is_canonical=True, source_turn_ids=[2],
    )
    candidate.score = SignalScore(
        validator_verdict="pass", advisor_ownership=4, decision_impact=3,
        evidence_quality=4, explicitness=4, urgency=2, final_score=3.0,
    )
    return candidate


def context_turn(turn_id: int, index: int, text: str) -> TranscriptTurn:
    return TranscriptTurn(
        id=turn_id, transcript_id=1, turn_index=index, timestamp=f"00:00:0{index}",
        raw_speaker_label="Advisor", inferred_role="advisor", text=text,
    )


def test_previous_concern_completes_threshold_blocker() -> None:
    candidate = context_candidate("blocker", "I'm not anywhere near that threshold for you guys.", "Threshold Status")
    turns = [
        context_turn(1, 0, "Yeah, I'm worried about that."),
        context_turn(2, 1, candidate.advisor_quote),
    ]
    context = reranker._candidate_context(candidate, turns)

    reranker._apply_contextual_direction(candidate, context)

    assert candidate.category == "Minimum Book Threshold"
    assert "concerned" in candidate.rationale
    assert reranker._is_fallback_eligible(candidate, context) is True


def test_current_firm_compliance_friction_becomes_driver() -> None:
    candidate = context_candidate("blocker", "Then it is hard to put something out that's timely.")
    context = "With all the rules that every firm I've worked with has put on, it is hard to be timely."

    reranker._apply_contextual_direction(candidate, context)

    assert candidate.item_type == "driver"
    assert candidate.category == "Communication Flexibility"


def test_optimize_proof_requirement_remains_blocker() -> None:
    candidate = context_candidate("blocker", "It is hard to put something out that's timely.")
    context = "Every firm has rules, and I need proof that Optimize is different before proceeding."

    reranker._apply_contextual_direction(candidate, context)

    assert candidate.item_type == "blocker"


def test_same_turn_dedicated_support_resolves_time_constraint() -> None:
    candidate = context_candidate("blocker", "I manage money. I don't have time for financial planning.")
    source = "I manage money. I don't have time for financial planning. So, fortunately, we have dedicated financial planning support whose job is to support us."

    assert reranker._passes_contextual_final_gate(candidate, source, source) is False


def test_positive_continuation_resolves_accountability_blocker() -> None:
    candidate = context_candidate("blocker", "We have to tell clients the rationale for every decision.")
    context = candidate.advisor_quote + " It has worked very well."

    assert reranker._passes_contextual_final_gate(candidate, context, candidate.advisor_quote) is False


def test_existing_practice_outcome_is_not_standalone_driver() -> None:
    candidate = context_candidate("driver", "We use this strategy for income generation and partial portfolio protection.")

    assert reranker._passes_contextual_final_gate(candidate, candidate.advisor_quote, candidate.advisor_quote) is False
