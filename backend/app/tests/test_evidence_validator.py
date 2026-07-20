from collections.abc import Generator

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.app.database import Base
from backend.app.llm_schemas import EvidenceValidationOutput
from backend.app.models import CandidateSignal, SignalScore, Transcript, TranscriptTurn
from backend.app.services import evidence_validator
from backend.app.services.evidence_validator import validate_evidence_for_transcript


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


def add_turn(
    db: Session,
    transcript: Transcript,
    *,
    text: str,
    role: str,
    index: int = 0,
    confidence: float = 0.9,
) -> TranscriptTurn:
    turn = TranscriptTurn(
        transcript_id=transcript.id,
        turn_index=index,
        timestamp=f"00:00:0{index}",
        raw_speaker_label=f"Speaker {index}",
        inferred_role=role,
        role_confidence=confidence,
        text=text,
    )
    db.add(turn)
    db.flush()
    return turn


def add_candidate(
    db: Session,
    transcript: Transcript,
    *,
    quote: str,
    source_turn_ids: list[int],
    item_type: str = "driver",
    rationale: str = "Synthetic decision-relevant rationale.",
) -> CandidateSignal:
    candidate = CandidateSignal(
        transcript_id=transcript.id,
        item_type=item_type,
        category="synthetic_category",
        advisor_quote=quote,
        timestamp="00:00:00",
        evidence_strength="explicit",
        rationale=rationale,
        extraction_confidence=0.9,
        source_turn_ids=source_turn_ids,
    )
    db.add(candidate)
    db.commit()
    return candidate


def llm_output(verdict: str = "pass") -> EvidenceValidationOutput:
    return EvidenceValidationOutput(
        verdict=verdict,
        support_score=0.9,
        advisor_side_score=0.95,
        false_positive_risk=0.1,
        rejection_reason=None,
        explanation="Synthetic validation explanation.",
    )


def structured_neutral_commitment_output(**overrides: object) -> EvidenceValidationOutput:
    findings: dict[str, object] = {
        "quote_traceability": "exact",
        "source_turn_match": "exact",
        "advisor_ownership": "reliable_advisor",
        "ownership_basis": "The persisted source turn is reliably advisor-owned.",
        "context_sufficiency": "sufficient",
        "context_scope": "local_advisor_thought",
        "decision_relevance": "material",
        "supported_decision_effect": "indeterminate",
        "direction_support": "supports_driver",
        "validated_evidence_strength": "explicit",
        "rationale_grounding": "fully_grounded",
        "unsupported_rationale_claims": [],
        "representative_dependency": "independent",
        "procedural_status": "substantive_factor",
        "unsupported_escalations": [],
        "contradiction_status": "none",
        "semantic_consistency": "consistent",
        "consistency_issues": [],
        "support_score": 0.9,
        "advisor_side_score": 0.95,
        "false_positive_risk": 0.1,
        "explanation": "The evidence is explicit and advisor-owned, but the effect was marked indeterminate.",
    }
    findings.update(overrides)
    return EvidenceValidationOutput(**findings)


def test_quote_not_found_rejects_without_llm_call(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    turn = add_turn(db, transcript, text="Different synthetic text.", role="advisor")
    candidate = add_candidate(
        db, transcript, quote="Missing quote.", source_turn_ids=[turn.id]
    )
    monkeypatch.setattr(
        evidence_validator,
        "call_llm_json",
        lambda **_: pytest.fail("LLM must not be called"),
    )

    validate_evidence_for_transcript(transcript.id, db)
    db.refresh(candidate)

    assert candidate.score.validator_verdict == "reject"
    assert candidate.score.rejection_reason == "quote_not_found"


def test_optimize_rep_only_quote_rejects_without_llm_call(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    turn = add_turn(db, transcript, text="Representative statement.", role="optimize_rep")
    candidate = add_candidate(
        db, transcript, quote=turn.text, source_turn_ids=[turn.id]
    )
    monkeypatch.setattr(
        evidence_validator,
        "call_llm_json",
        lambda **_: pytest.fail("LLM must not be called"),
    )

    validate_evidence_for_transcript(transcript.id, db)
    db.refresh(candidate)

    assert candidate.score.validator_verdict == "reject"
    assert candidate.score.rejection_reason == "not_advisor_side"


def test_advisor_quote_calls_llm_and_saves_pass(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    turn = add_turn(db, transcript, text="Advisor evidence.", role="advisor")
    candidate = add_candidate(
        db, transcript, quote=turn.text, source_turn_ids=[turn.id]
    )
    calls: list[dict] = []

    def fake_call(**kwargs: object) -> EvidenceValidationOutput:
        calls.append(kwargs)
        return llm_output("pass")

    monkeypatch.setattr(evidence_validator, "call_llm_json", fake_call)

    validate_evidence_for_transcript(transcript.id, db)
    db.refresh(candidate)

    assert len(calls) == 1
    assert candidate.score.validator_verdict == "pass"
    assert candidate.score.support_score == 0.9


def test_mixed_turn_cannot_pass_ownership_check(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    turn = add_turn(db, transcript, text="Ambiguous advisor evidence.", role="mixed")
    candidate = add_candidate(
        db, transcript, quote=turn.text, source_turn_ids=[turn.id]
    )
    monkeypatch.setattr(
        evidence_validator,
        "call_llm_json",
        lambda **_: pytest.fail("Mixed evidence must not reach the LLM"),
    )

    validate_evidence_for_transcript(transcript.id, db)
    db.refresh(candidate)

    assert candidate.score.validator_verdict == "reject"
    assert candidate.score.rejection_reason == "not_advisor_side"


def test_existing_score_is_updated_not_duplicated(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    turn = add_turn(db, transcript, text="Advisor evidence.", role="advisor")
    candidate = add_candidate(
        db, transcript, quote=turn.text, source_turn_ids=[turn.id]
    )
    existing_score = SignalScore(
        signal_id=candidate.id,
        validator_verdict="needs_review",
        support_score=0.4,
    )
    db.add(existing_score)
    db.commit()
    existing_id = existing_score.id
    monkeypatch.setattr(
        evidence_validator, "call_llm_json", lambda **_: llm_output("pass")
    )

    validate_evidence_for_transcript(transcript.id, db)

    score_count = db.scalar(
        select(func.count(SignalScore.id)).where(SignalScore.signal_id == candidate.id)
    )
    db.refresh(existing_score)
    assert score_count == 1
    assert existing_score.id == existing_id
    assert existing_score.validator_verdict == "pass"


def test_summary_counts_are_correct(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    reject_turn = add_turn(
        db, transcript, text="Unrelated advisor text.", role="advisor", index=0
    )
    pass_turn = add_turn(
        db, transcript, text="Strong advisor evidence.", role="advisor", index=1
    )
    review_turn = add_turn(
        db, transcript, text="Ambiguous advisor evidence.", role="advisor", index=2
    )
    add_candidate(
        db, transcript, quote="Missing quote.", source_turn_ids=[reject_turn.id]
    )
    pass_candidate = add_candidate(
        db, transcript, quote=pass_turn.text, source_turn_ids=[pass_turn.id]
    )
    review_candidate = add_candidate(
        db,
        transcript,
        quote=review_turn.text,
        source_turn_ids=[review_turn.id],
        item_type="blocker",
    )

    def fake_call(**kwargs: object) -> EvidenceValidationOutput:
        payload = kwargs["input_payload"]
        signal_id = payload["candidate"]["signal_id"]
        return llm_output("pass" if signal_id == pass_candidate.id else "needs_review")

    monkeypatch.setattr(evidence_validator, "call_llm_json", fake_call)

    summary = validate_evidence_for_transcript(transcript.id, db)

    assert review_candidate.id != pass_candidate.id
    assert summary == {
        "transcript_id": transcript.id,
        "status": "evidence_validated",
        "candidate_count": 3,
        "passed": 1,
        "rejected": 1,
        "needs_review": 1,
    }


def test_rationale_type_contradiction_is_rejected_without_llm(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    turn = add_turn(
        db,
        transcript,
        text="Does that mean I pay the fee?",
        role="advisor",
    )
    candidate = add_candidate(
        db,
        transcript,
        quote=turn.text,
        source_turn_ids=[turn.id],
        item_type="driver",
        rationale="The advisor expresses concern and uncertainty about cost.",
    )
    monkeypatch.setattr(
        evidence_validator,
        "call_llm_json",
        lambda **_: pytest.fail("Contradictory driver must not reach the LLM"),
    )

    validate_evidence_for_transcript(transcript.id, db)
    db.refresh(candidate)

    assert candidate.score.validator_verdict == "reject"
    assert candidate.score.rejection_reason == "insufficient_evidence"


def test_ambiguous_short_quote_includes_neighboring_context(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    question = add_turn(
        db,
        transcript,
        text="What happens to my insurance book?",
        role="advisor",
        index=0,
    )
    add_turn(
        db,
        transcript,
        text="Your insurance book stays as is.",
        role="optimize_rep",
        index=1,
    )
    short_turn = add_turn(
        db,
        transcript,
        text="[CURRENT_DEALER] would can me.",
        role="advisor",
        index=2,
    )
    add_turn(
        db,
        transcript,
        text="Not necessarily.",
        role="optimize_rep",
        index=3,
    )
    ownership = add_turn(
        db,
        transcript,
        text="I don't own it. [CURRENT_DEALER] does.",
        role="advisor",
        index=4,
    )
    add_candidate(
        db,
        transcript,
        quote=short_turn.text,
        source_turn_ids=[short_turn.id],
        item_type="blocker",
        rationale="The advisor identifies a risk to the current insurance book.",
    )
    calls: list[dict] = []

    def fake_call(**kwargs: object) -> EvidenceValidationOutput:
        calls.append(kwargs)
        return llm_output("pass")

    monkeypatch.setattr(evidence_validator, "call_llm_json", fake_call)

    validate_evidence_for_transcript(transcript.id, db)

    supporting_turns = calls[0]["input_payload"]["supporting_turns"]
    assert [turn["turn_id"] for turn in supporting_turns] == [short_turn.id]
    assert supporting_turns[0]["text"] == short_turn.text


def test_explicit_moving_forward_driver_can_pass(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    turn = add_turn(
        db, transcript, text="We're moving forward.", role="advisor"
    )
    candidate = add_candidate(
        db,
        transcript,
        quote=turn.text,
        source_turn_ids=[turn.id],
        item_type="driver",
        rationale="The advisor makes an explicit positive commitment.",
    )
    monkeypatch.setattr(
        evidence_validator, "call_llm_json", lambda **_: llm_output("pass")
    )

    validate_evidence_for_transcript(transcript.id, db)
    db.refresh(candidate)

    assert candidate.score.validator_verdict == "pass"


@pytest.mark.parametrize(
    "quote",
    [
        "We're moving forward.",
        "I've decided to proceed.",
        "We're ready to proceed.",
    ],
)
def test_structured_neutral_effect_does_not_reject_explicit_advisor_commitment(
    db: Session, monkeypatch: pytest.MonkeyPatch, quote: str
) -> None:
    transcript = create_transcript(db)
    turn = add_turn(db, transcript, text=quote, role="advisor")
    candidate = add_candidate(
        db,
        transcript,
        quote=quote,
        source_turn_ids=[turn.id],
        rationale="The advisor makes an explicit positive transition commitment.",
    )
    monkeypatch.setattr(
        evidence_validator,
        "call_llm_json",
        lambda **_: structured_neutral_commitment_output(),
    )

    validate_evidence_for_transcript(transcript.id, db)
    db.refresh(candidate)

    assert candidate.score.validator_verdict == "pass"


@pytest.mark.parametrize(
    ("quote", "output_overrides"),
    [
        ("If we move forward, we will need approval.", {}),
        ("Before moving forward, send the documents.", {}),
        ("We could move forward later.", {}),
        ('The client said, "We\'re moving forward."', {}),
        ("Moving forward, use this form.", {"procedural_status": "procedural_only"}),
        ("We're moving forward.", {"quote_traceability": "absent"}),
        ("We're moving forward.", {"unsupported_escalations": ["interest_to_commitment"]}),
        ("We're moving forward.", {"contradiction_status": "direct_contradiction"}),
    ],
)
def test_explicit_commitment_safeguard_preserves_hard_rejections(
    db: Session,
    monkeypatch: pytest.MonkeyPatch,
    quote: str,
    output_overrides: dict[str, object],
) -> None:
    transcript = create_transcript(db)
    turn = add_turn(db, transcript, text=quote, role="advisor")
    candidate = add_candidate(
        db,
        transcript,
        quote=quote,
        source_turn_ids=[turn.id],
        rationale="The advisor makes an explicit positive transition commitment.",
    )
    monkeypatch.setattr(
        evidence_validator,
        "call_llm_json",
        lambda **_: structured_neutral_commitment_output(**output_overrides),
    )

    validate_evidence_for_transcript(transcript.id, db)
    db.refresh(candidate)

    assert candidate.score.validator_verdict == "reject"


def test_representative_commitment_does_not_reach_semantic_safeguard(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    turn = add_turn(db, transcript, text="We're moving forward.", role="optimize_rep")
    candidate = add_candidate(
        db,
        transcript,
        quote=turn.text,
        source_turn_ids=[turn.id],
        rationale="The representative states that the process is moving forward.",
    )
    monkeypatch.setattr(
        evidence_validator,
        "call_llm_json",
        lambda **_: pytest.fail("Representative evidence must be rejected first"),
    )

    validate_evidence_for_transcript(transcript.id, db)
    db.refresh(candidate)

    assert candidate.score.validator_verdict == "reject"
    assert candidate.score.rejection_reason == "not_advisor_side"


@pytest.mark.parametrize(
    ("quote", "reason"),
    [
        ("Absolutely. Absolutely.", "polite_or_procedural"),
        ("Happy to connect next week.", "polite_or_procedural"),
        ("I can design seminars.", "contextual_only"),
        ("I've been in the business for 20 years.", "contextual_only"),
        ("Interesting.", "polite_or_procedural"),
    ],
)
def test_deterministic_gate_rejects_non_signals_without_llm(
    db: Session, monkeypatch: pytest.MonkeyPatch, quote: str, reason: str
) -> None:
    transcript = create_transcript(db)
    turn = add_turn(db, transcript, text=quote, role="advisor")
    candidate = add_candidate(
        db, transcript, quote=quote, source_turn_ids=[turn.id],
        rationale="The advisor expresses general interest.",
    )
    monkeypatch.setattr(
        evidence_validator, "call_llm_json",
        lambda **_: pytest.fail("Deterministic non-signal must not reach the LLM"),
    )

    validate_evidence_for_transcript(transcript.id, db)
    db.refresh(candidate)

    assert candidate.score.validator_verdict == "reject"
    assert candidate.score.rejection_reason == reason


def test_short_explicit_blocker_can_pass(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    turn = add_turn(db, transcript, text="Fees are too high.", role="advisor")
    candidate = add_candidate(
        db, transcript, quote=turn.text, source_turn_ids=[turn.id],
        item_type="blocker",
        rationale="The advisor states that fees could prevent proceeding.",
    )
    monkeypatch.setattr(
        evidence_validator, "call_llm_json", lambda **_: llm_output("pass")
    )

    validate_evidence_for_transcript(transcript.id, db)
    db.refresh(candidate)

    assert candidate.score.validator_verdict == "pass"


def test_driver_question_is_insufficient_evidence_without_llm(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    turn = add_turn(db, transcript, text="Can you send details?", role="advisor")
    candidate = add_candidate(
        db, transcript, quote=turn.text, source_turn_ids=[turn.id],
        item_type="driver", rationale="The advisor wants information.",
    )
    monkeypatch.setattr(
        evidence_validator, "call_llm_json",
        lambda **_: pytest.fail("A question cannot pass as a driver"),
    )

    validate_evidence_for_transcript(transcript.id, db)
    db.refresh(candidate)

    assert candidate.score.rejection_reason == "insufficient_evidence"


@pytest.mark.parametrize(
    ("quote", "rationale"),
    [
        (
            "I need better operational support.",
            "The advisor requires stronger operational support to scale the business.",
        ),
        (
            "We require better technology before we can grow.",
            "The advisor requires a stronger technology platform to support growth.",
        ),
    ],
)
def test_need_language_does_not_deterministically_reject_driver(
    db: Session,
    monkeypatch: pytest.MonkeyPatch,
    quote: str,
    rationale: str,
) -> None:
    transcript = create_transcript(db)
    turn = add_turn(db, transcript, text=quote, role="advisor")
    candidate = add_candidate(
        db,
        transcript,
        quote=quote,
        source_turn_ids=[turn.id],
        rationale=rationale,
    )
    monkeypatch.setattr(
        evidence_validator, "call_llm_json", lambda **_: llm_output("pass")
    )

    validate_evidence_for_transcript(transcript.id, db)
    db.refresh(candidate)

    assert candidate.score.validator_verdict == "pass"


def test_structurally_obvious_driver_contradiction_is_rejected(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    turn = add_turn(db, transcript, text="The restriction remains.", role="advisor")
    candidate = add_candidate(
        db,
        transcript,
        quote=turn.text,
        source_turn_ids=[turn.id],
        rationale="The restriction prevents the advisor from proceeding.",
    )
    monkeypatch.setattr(
        evidence_validator,
        "call_llm_json",
        lambda **_: pytest.fail("Structural contradiction must not reach the LLM"),
    )

    validate_evidence_for_transcript(transcript.id, db)
    db.refresh(candidate)

    assert candidate.score.rejection_reason == "rationale_type_contradiction"


def test_legitimate_blocker_is_unaffected(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    turn = add_turn(db, transcript, text="The contract blocks the transition.", role="advisor")
    candidate = add_candidate(
        db,
        transcript,
        quote=turn.text,
        source_turn_ids=[turn.id],
        item_type="blocker",
        rationale="The contract restriction prevents proceeding.",
    )
    monkeypatch.setattr(
        evidence_validator, "call_llm_json", lambda **_: llm_output("pass")
    )

    validate_evidence_for_transcript(transcript.id, db)
    db.refresh(candidate)

    assert candidate.score.validator_verdict == "pass"


@pytest.mark.parametrize(
    "quote",
    [
        "Let's connect next week after my partner approves it.",
        "We can meet Friday because I want to review the transition plan.",
        "Call me next week once the contract restriction is resolved.",
    ],
)
def test_procedural_phrase_with_decision_content_reaches_semantic_validation(
    db: Session, monkeypatch: pytest.MonkeyPatch, quote: str
) -> None:
    transcript = create_transcript(db)
    turn = add_turn(db, transcript, text=quote, role="advisor")
    candidate = add_candidate(
        db,
        transcript,
        quote=quote,
        source_turn_ids=[turn.id],
        item_type="blocker",
        rationale="The advisor states a decision-relevant condition.",
    )
    monkeypatch.setattr(
        evidence_validator, "call_llm_json", lambda **_: llm_output("pass")
    )

    validate_evidence_for_transcript(transcript.id, db)
    db.refresh(candidate)

    assert candidate.score.validator_verdict == "pass"


def test_schedule_another_call_is_purely_procedural(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    turn = add_turn(db, transcript, text="Let's schedule another call.", role="advisor")
    candidate = add_candidate(
        db, transcript, quote=turn.text, source_turn_ids=[turn.id]
    )
    monkeypatch.setattr(
        evidence_validator,
        "call_llm_json",
        lambda **_: pytest.fail("Purely procedural evidence must not reach the LLM"),
    )

    validate_evidence_for_transcript(transcript.id, db)
    db.refresh(candidate)

    assert candidate.score.rejection_reason == "polite_or_procedural"


def test_low_confidence_advisor_cannot_pass(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    turn = add_turn(
        db,
        transcript,
        text="I want to move forward.",
        role="advisor",
        confidence=0.64,
    )
    candidate = add_candidate(
        db, transcript, quote=turn.text, source_turn_ids=[turn.id]
    )
    monkeypatch.setattr(
        evidence_validator,
        "call_llm_json",
        lambda **_: pytest.fail("Low-confidence ownership must not reach the LLM"),
    )

    validate_evidence_for_transcript(transcript.id, db)
    db.refresh(candidate)

    assert candidate.score.rejection_reason == "not_advisor_side"


def test_source_turn_resolves_repeated_quote_and_exact_timestamp(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    first = add_turn(db, transcript, text="I need support.", role="advisor", index=0)
    add_turn(db, transcript, text="Representative context.", role="optimize_rep", index=1)
    second = add_turn(db, transcript, text="I need support.", role="advisor", index=2)
    candidate = add_candidate(
        db,
        transcript,
        quote=second.text,
        source_turn_ids=[second.id],
        rationale="The advisor values additional support.",
    )
    calls: list[dict] = []

    def fake_call(**kwargs: object) -> EvidenceValidationOutput:
        calls.append(kwargs)
        return llm_output("pass")

    monkeypatch.setattr(evidence_validator, "call_llm_json", fake_call)

    validate_evidence_for_transcript(transcript.id, db)
    db.refresh(candidate)

    assert candidate.timestamp == second.timestamp
    supporting = calls[0]["input_payload"]["supporting_turns"]
    assert [turn["turn_id"] for turn in supporting] == [second.id]


def test_invalid_source_falls_back_to_unique_global_advisor_match(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    wrong = add_turn(db, transcript, text="Different source text.", role="advisor", index=0)
    match = add_turn(db, transcript, text="I value the support.", role="advisor", index=1)
    candidate = add_candidate(
        db,
        transcript,
        quote=match.text,
        source_turn_ids=[wrong.id],
        rationale="The advisor values the support.",
    )
    monkeypatch.setattr(
        evidence_validator, "call_llm_json", lambda **_: llm_output("pass")
    )

    validate_evidence_for_transcript(transcript.id, db)
    db.refresh(candidate)

    assert candidate.timestamp == match.timestamp
    assert candidate.score.validator_verdict == "pass"


def test_multiple_global_advisor_matches_are_reviewable_ambiguity(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    first = add_turn(db, transcript, text="I value the support.", role="advisor", index=0)
    add_turn(db, transcript, text=first.text, role="advisor", index=1)
    candidate = add_candidate(
        db,
        transcript,
        quote=first.text,
        source_turn_ids=[999999],
        rationale="The advisor values the support.",
    )
    candidate.timestamp = None
    db.commit()
    monkeypatch.setattr(
        evidence_validator,
        "call_llm_json",
        lambda **_: pytest.fail("Ambiguous evidence must not reach the LLM"),
    )

    validate_evidence_for_transcript(transcript.id, db)
    db.refresh(candidate)

    assert candidate.score.validator_verdict == "needs_review"
    assert candidate.score.rejection_reason == "ambiguous_evidence"


def test_grounded_indeterminate_effect_with_supplementary_uncertainty_needs_review() -> None:
    candidate = CandidateSignal(
        item_type="blocker", category="Operational constraint",
        advisor_quote="We would need to work through client file transfer.",
        rationale="File transfer may affect transition timing.",
        evidence_strength="implicit", source_turn_ids=[1],
    )
    output = structured_neutral_commitment_output(
        decision_relevance="material",
        supported_decision_effect="indeterminate",
        direction_support="supports_blocker",
        validated_evidence_strength="weakly_implied",
        rationale_grounding="partially_grounded",
        unsupported_rationale_claims=["The exact timing impact is not stated."],
    )

    decision = evidence_validator.derive_validation_decision(output, candidate)

    assert decision.verdict == "needs_review"
    assert "weak_evidence" in decision.review_reasons


def test_fully_grounded_direction_with_incomplete_effect_needs_review() -> None:
    candidate = CandidateSignal(
        item_type="driver", category="Positive reaction",
        advisor_quote="That support sounds useful.",
        rationale="The advisor reacts positively to the support.",
        evidence_strength="explicit", source_turn_ids=[1],
    )
    output = structured_neutral_commitment_output()

    decision = evidence_validator.derive_validation_decision(output, candidate)

    assert decision.verdict == "needs_review"
    assert "indeterminate_effect" in decision.review_reasons


def test_conditional_language_rewritten_as_commitment_rejects_without_llm(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    turn = add_turn(db, transcript, text="I could consider moving later.", role="advisor")
    candidate = add_candidate(
        db, transcript, quote=turn.text, source_turn_ids=[turn.id],
        rationale="The advisor is committed and will move.",
    )
    monkeypatch.setattr(
        evidence_validator, "call_llm_json",
        lambda **_: pytest.fail("Invented commitment must reject deterministically"),
    )

    validate_evidence_for_transcript(transcript.id, db)
    db.refresh(candidate)

    assert candidate.score.validator_verdict == "reject"
    assert candidate.score.rejection_reason == "interest_to_commitment"


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"unsupported_escalations": ["other"]}, "unsupported_escalation"),
        ({"advisor_ownership": "representative"}, "ownership_failure"),
        ({"contradiction_status": "direct_contradiction"}, "direct_contradiction"),
        ({"supported_decision_effect": "neutral", "direction_support": "unsupported"}, "neutral_or_indeterminate_effect"),
    ],
)
def test_recalibration_preserves_hard_factual_rejections(
    overrides: dict[str, object], reason: str
) -> None:
    candidate = CandidateSignal(
        item_type="driver", category="Support", advisor_quote="That support sounds useful.",
        rationale="The advisor reacts positively to support.", evidence_strength="implicit",
        source_turn_ids=[1],
    )
    output = structured_neutral_commitment_output(**overrides)

    decision = evidence_validator.derive_validation_decision(output, candidate)

    assert decision.verdict == "reject"
    assert reason in decision.hard_failures


def test_representative_match_is_ignored_when_unique_advisor_match_exists(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    quote = "The transition support matters."
    add_turn(db, transcript, text=quote, role="optimize_rep", index=0)
    advisor = add_turn(db, transcript, text=quote, role="advisor", index=1)
    candidate = add_candidate(
        db,
        transcript,
        quote=quote,
        source_turn_ids=[999999],
        rationale="The advisor values transition support.",
    )
    monkeypatch.setattr(
        evidence_validator, "call_llm_json", lambda **_: llm_output("pass")
    )

    validate_evidence_for_transcript(transcript.id, db)
    db.refresh(candidate)

    assert candidate.timestamp == advisor.timestamp
    assert candidate.score.validator_verdict == "pass"
