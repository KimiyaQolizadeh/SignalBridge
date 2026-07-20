from collections.abc import Generator

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.app.database import Base
from backend.app.llm_schemas import (
    CandidateSignalBatchOutput,
    CandidateSignalItem,
    SpeakerRoleBatchOutput,
    SpeakerRoleItem,
)
from backend.app.models import CandidateSignal, Transcript, TranscriptTurn
from backend.app.services import signal_extractor
from backend.app.services import speaker_classifier
from backend.app.services.signal_extractor import (
    SpeakersNotClassifiedError,
    extract_candidate_signals_for_transcript,
)


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


def create_transcript(db: Session, roles: list[str | None]) -> tuple[Transcript, list[TranscriptTurn]]:
    transcript = Transcript(file_name="synthetic.txt", raw_text="Synthetic content")
    db.add(transcript)
    db.flush()
    turns = [
        TranscriptTurn(
            transcript_id=transcript.id,
            turn_index=index,
            timestamp=f"00:00:0{index}",
            raw_speaker_label=f"Speaker {index}",
            inferred_role=role,
            role_confidence=0.9 if role is not None else None,
            text=f"Synthetic evidence statement {index}",
        )
        for index, role in enumerate(roles)
    ]
    db.add_all(turns)
    db.commit()
    return transcript, turns


def candidate_item(
    *,
    item_type: str,
    quote: str,
    source_turn_ids: list[int],
    category: str = "platform_support",
    rationale: str = "Synthetic decision-relevant rationale.",
    factor_kind: str | None = None,
    decision_direction: str | None = None,
    decision_relation: str = "The advisor connects this factor to the decision.",
    confidence: float = 0.9,
) -> CandidateSignalItem:
    normalized_factor_kind = factor_kind or (
        "desired_outcome" if item_type == "driver" else "future_condition"
    )
    normalized_direction = decision_direction or (
        "supports_move" if item_type == "driver" else "opposes_move"
    )
    return CandidateSignalItem(
        factor_kind=normalized_factor_kind,
        decision_direction=normalized_direction,
        decision_relation=decision_relation,
        item_type=item_type,
        category=None if item_type == "no_signal" else category,
        advisor_quote=quote,
        timestamp="00:00:01",
        evidence_strength="explicit",
        rationale=None if item_type == "no_signal" else rationale,
        source_turn_ids=source_turn_ids,
        extraction_confidence=confidence,
    )


def mock_output(
    monkeypatch: pytest.MonkeyPatch, items: list[CandidateSignalItem]
) -> None:
    monkeypatch.setattr(
        signal_extractor,
        "call_llm_json",
        lambda **_: CandidateSignalBatchOutput(items=items),
    )


def saved_candidates(db: Session, transcript_id: int) -> list[CandidateSignal]:
    return list(
        db.scalars(
            select(CandidateSignal)
            .where(CandidateSignal.transcript_id == transcript_id)
            .order_by(CandidateSignal.id)
        ).all()
    )


def test_requires_speaker_classification(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript, _ = create_transcript(db, [None])
    mock_output(monkeypatch, [])

    with pytest.raises(SpeakersNotClassifiedError):
        extract_candidate_signals_for_transcript(transcript.id, db)


def test_extracts_and_saves_driver_but_not_mixed_turn_evidence(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript, turns = create_transcript(db, ["advisor", "mixed"])
    mock_output(
        monkeypatch,
        [
            candidate_item(
                item_type="driver",
                quote=turns[0].text,
                source_turn_ids=[turns[0].id],
            ),
            candidate_item(
                item_type="blocker",
                quote=turns[1].text,
                source_turn_ids=[turns[1].id],
                category="transition_timing",
            ),
        ],
    )

    summary = extract_candidate_signals_for_transcript(transcript.id, db)
    candidates = saved_candidates(db, transcript.id)

    assert [candidate.item_type for candidate in candidates] == ["driver"]
    assert summary["candidate_count"] == 1


def test_skips_invalid_sources_when_quote_is_not_found(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript, _ = create_transcript(db, ["advisor"])
    mock_output(
        monkeypatch,
        [
            candidate_item(
                item_type="driver",
                quote="Quote absent from every evidence turn.",
                source_turn_ids=[999999],
            )
        ],
    )

    summary = extract_candidate_signals_for_transcript(transcript.id, db)

    assert summary["candidate_count"] == 0
    assert saved_candidates(db, transcript.id) == []


def test_removes_invalid_source_ids_when_quote_is_found(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript, turns = create_transcript(db, ["advisor"])
    mock_output(
        monkeypatch,
        [
            candidate_item(
                item_type="driver",
                quote=turns[0].text,
                source_turn_ids=[turns[0].id, 999999],
            )
        ],
    )

    extract_candidate_signals_for_transcript(transcript.id, db)
    candidate = saved_candidates(db, transcript.id)[0]

    assert candidate.source_turn_ids == [turns[0].id]


def test_exact_duplicate_candidates_are_saved_once(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript, turns = create_transcript(db, ["advisor"])
    mock_output(
        monkeypatch,
        [
            candidate_item(
                item_type="driver",
                quote=turns[0].text,
                source_turn_ids=[turns[0].id],
                category="Platform Support",
            ),
            candidate_item(
                item_type="driver",
                quote=turns[0].text,
                source_turn_ids=[turns[0].id],
                category=" platform support ",
            ),
        ],
    )

    summary = extract_candidate_signals_for_transcript(transcript.id, db)

    assert summary["candidate_count"] == 1
    assert len(saved_candidates(db, transcript.id)) == 1


def test_summary_counts_are_correct(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript, turns = create_transcript(db, ["advisor", "advisor", "mixed"])
    mock_output(
        monkeypatch,
        [
            candidate_item(
                item_type="driver",
                quote=turns[0].text,
                source_turn_ids=[turns[0].id],
            ),
            candidate_item(
                item_type="driver",
                quote=turns[1].text,
                source_turn_ids=[turns[1].id],
                category="growth",
            ),
            candidate_item(
                item_type="blocker",
                quote=turns[2].text,
                source_turn_ids=[turns[2].id],
                category="timing",
            ),
        ],
    )

    summary = extract_candidate_signals_for_transcript(transcript.id, db)

    assert summary == {
        "transcript_id": transcript.id,
        "status": "candidates_extracted",
        "candidate_count": 2,
        "driver_candidates": 2,
        "blocker_candidates": 0,
    }


def test_business_relevant_fee_question_is_persisted(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript, turns = create_transcript(db, ["advisor"])
    turns[0].text = "Does that mean I don't pay them, or the client doesn't pay them?"
    db.commit()
    mock_output(
        monkeypatch,
        [
            candidate_item(
                item_type="driver",
                quote=turns[0].text,
                source_turn_ids=[turns[0].id],
                category="Fee Structure Clarity",
                rationale="The advisor expresses concern about fee costs.",
            )
        ],
    )

    summary = extract_candidate_signals_for_transcript(transcript.id, db)
    candidates = saved_candidates(db, transcript.id)

    assert len(candidates) == 1
    assert summary["driver_candidates"] == 1
    assert summary["blocker_candidates"] == 0


def test_conditional_interest_can_split_into_driver_and_blocker(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript, turns = create_transcript(db, ["advisor"])
    turns[0].text = (
        "I like the family office service, but I need my dealer to participate."
    )
    db.commit()
    mock_output(
        monkeypatch,
        [
            candidate_item(
                item_type="driver",
                quote=turns[0].text,
                source_turn_ids=[turns[0].id],
                category="Family Office Value",
                rationale="The advisor explicitly values the family office service.",
                factor_kind="desired_outcome",
                decision_direction="supports_move",
            ),
            candidate_item(
                item_type="blocker",
                quote=turns[0].text,
                source_turn_ids=[turns[0].id],
                category="Dealer Participation",
                rationale="Dealer participation is an explicit dependency.",
                factor_kind="dependency",
                decision_direction="opposes_move",
            ),
        ],
    )

    extract_candidate_signals_for_transcript(transcript.id, db)
    candidates = saved_candidates(db, transcript.id)

    assert [(candidate.item_type, candidate.category) for candidate in candidates] == [
        ("driver", "Family Office Value"),
        ("blocker", "Dealer Participation"),
    ]


def test_explicit_moving_forward_remains_driver(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript, turns = create_transcript(db, ["advisor"])
    turns[0].text = "We're moving forward."
    db.commit()
    mock_output(
        monkeypatch,
        [
            candidate_item(
                item_type="driver",
                quote=turns[0].text,
                source_turn_ids=[turns[0].id],
                category="Willingness to Proceed",
                rationale="The advisor makes an explicit positive commitment.",
                factor_kind="explicit_commitment",
                decision_direction="supports_move",
                confidence=0.99,
            )
        ],
    )

    extract_candidate_signals_for_transcript(transcript.id, db)

    assert saved_candidates(db, transcript.id)[0].item_type == "driver"


def test_classified_advisor_turns_create_extraction_batch(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript, turns = create_transcript(db, [None, None])
    turns[0].raw_speaker_label = "ADVISOR_1"
    turns[1].raw_speaker_label = "OPTIMIZE_REP"
    db.commit()

    def classify_call(**kwargs: object) -> SpeakerRoleBatchOutput:
        batch_turns = kwargs["input_payload"]["turns"]
        return SpeakerRoleBatchOutput(
            items=[
                SpeakerRoleItem(
                    turn_id=turn["turn_id"],
                    inferred_role=(
                        "advisor"
                        if turn["raw_speaker_label"].startswith("ADVISOR")
                        else "optimize_rep"
                    ),
                    confidence=0.99,
                    reason="Explicit synthetic role label.",
                )
                for turn in batch_turns
            ]
        )

    monkeypatch.setattr(speaker_classifier, "call_llm_json", classify_call)
    speaker_classifier.classify_speakers_for_transcript(transcript.id, db)
    db.expire_all()

    extraction_calls: list[dict] = []

    def extract_call(**kwargs: object) -> CandidateSignalBatchOutput:
        extraction_calls.append(kwargs)
        payload_turns = kwargs["input_payload"]["turns"]
        advisor_turn = next(
            turn for turn in payload_turns if turn["inferred_role"] == "advisor"
        )
        return CandidateSignalBatchOutput(
            items=[
                candidate_item(
                    item_type="driver",
                    quote=advisor_turn["text"],
                    source_turn_ids=[advisor_turn["turn_id"]],
                    rationale="The advisor makes an explicit positive commitment.",
                )
            ]
        )

    monkeypatch.setattr(signal_extractor, "call_llm_json", extract_call)
    summary = extract_candidate_signals_for_transcript(transcript.id, db)

    assert len(extraction_calls) == 1
    assert summary["candidate_count"] == 1
    assert saved_candidates(db, transcript.id)[0].item_type == "driver"


def test_zero_eligible_advisor_turns_are_safe(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript, _ = create_transcript(db, ["optimize_rep", "unknown"])
    monkeypatch.setattr(
        signal_extractor,
        "call_llm_json",
        lambda **_: pytest.fail("No extraction batch should start"),
    )

    summary = extract_candidate_signals_for_transcript(transcript.id, db)

    assert summary["candidate_count"] == 0
    assert summary["driver_candidates"] == 0
    assert summary["blocker_candidates"] == 0


@pytest.mark.parametrize(
    ("quote", "factor_kind", "item_type", "decision_direction"),
    [
        (
            "I want to review the materials again and take the weekend to think.",
            "procedural_diligence",
            "driver",
            "supports_move",
        ),
        (
            "I'm used to having endless digital files so my compliance is right.",
            "preference",
            "blocker",
            "opposes_move",
        ),
        (
            "I'm working all the time and I'm exhausted.",
            "current_pain",
            "driver",
            "supports_move",
        ),
        (
            "Do you integrate with Salesforce?",
            "future_condition",
            "blocker",
            "opposes_move",
        ),
        (
            "That sounds good.",
            "desired_outcome",
            "driver",
            "supports_move",
        ),
        (
            "I'll discuss it with my partner.",
            "procedural_diligence",
            "blocker",
            "affects_timing",
        ),
        (
            "I prefer having all documents digitally.",
            "preference",
            "blocker",
            "opposes_move",
        ),
    ],
)
def test_business_relevant_items_are_left_for_downstream_scoring(
    db: Session,
    monkeypatch: pytest.MonkeyPatch,
    quote: str,
    factor_kind: str,
    item_type: str,
    decision_direction: str,
) -> None:
    transcript, turns = create_transcript(db, ["advisor"])
    turns[0].text = quote
    db.commit()
    mock_output(
        monkeypatch,
        [
            candidate_item(
                item_type=item_type,
                quote=quote,
                source_turn_ids=[turns[0].id],
                factor_kind=factor_kind,
                decision_direction=decision_direction,
                rationale="The advisor's statement is a mandatory decision factor.",
            )
        ],
    )

    summary = extract_candidate_signals_for_transcript(transcript.id, db)

    assert summary["candidate_count"] == 1
    assert len(saved_candidates(db, transcript.id)) == 1


@pytest.mark.parametrize(
    (
        "quote",
        "factor_kind",
        "item_type",
        "decision_direction",
        "category",
    ),
    [
        (
            "I would need confirmation that all compliance records remain "
            "accessible before I could move.",
            "future_condition",
            "blocker",
            "opposes_move",
            "Compliance Record Access Requirement",
        ),
        (
            "I'm working all the time, and that's why I'm looking for a firm "
            "that gives me more support.",
            "current_pain",
            "driver",
            "supports_move",
            "Operational Support Need",
        ),
        (
            "We cannot move unless your platform integrates with Salesforce.",
            "dependency",
            "blocker",
            "opposes_move",
            "Salesforce Integration Dependency",
        ),
        (
            "My partner has to approve before we can proceed.",
            "dependency",
            "blocker",
            "affects_timing",
            "Partner Approval Dependency",
        ),
        (
            "I cannot proceed unless all documents remain digitally accessible.",
            "future_condition",
            "blocker",
            "opposes_move",
            "Digital Document Access Requirement",
        ),
    ],
)
def test_grounded_decision_factors_are_persisted(
    db: Session,
    monkeypatch: pytest.MonkeyPatch,
    quote: str,
    factor_kind: str,
    item_type: str,
    decision_direction: str,
    category: str,
) -> None:
    transcript, turns = create_transcript(db, ["advisor"])
    turns[0].text = quote
    db.commit()
    mock_output(
        monkeypatch,
        [
            candidate_item(
                item_type=item_type,
                quote=quote,
                source_turn_ids=[turns[0].id],
                factor_kind=factor_kind,
                decision_direction=decision_direction,
                category=category,
                rationale="The advisor explicitly connects this factor to proceeding.",
            )
        ],
    )

    summary = extract_candidate_signals_for_transcript(transcript.id, db)
    candidates = saved_candidates(db, transcript.id)

    assert summary["candidate_count"] == 1
    assert [(candidate.item_type, candidate.category) for candidate in candidates] == [
        (item_type, category)
    ]


def test_current_pain_can_use_contiguous_advisor_context_for_transition_link(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript, turns = create_transcript(db, ["advisor", "advisor"])
    turns[0].text = "I'm working all the time and I'm exhausted."
    turns[1].text = "That's why I'm looking for a firm that gives me more support."
    db.commit()
    mock_output(
        monkeypatch,
        [
            candidate_item(
                item_type="driver",
                quote=turns[0].text,
                source_turn_ids=[turns[0].id],
                factor_kind="current_pain",
                decision_direction="supports_move",
                category="Operational Support Need",
            )
        ],
    )

    extract_candidate_signals_for_transcript(transcript.id, db)

    assert saved_candidates(db, transcript.id)[0].item_type == "driver"


def test_no_signal_interpretation_is_not_persisted(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript, turns = create_transcript(db, ["advisor"])
    turns[0].text = "I'll think about it."
    db.commit()
    mock_output(
        monkeypatch,
        [
            candidate_item(
                item_type="no_signal",
                quote=turns[0].text,
                source_turn_ids=[turns[0].id],
                factor_kind="procedural_diligence",
                decision_direction="neutral",
                decision_relation="Ordinary follow-up does not establish a decision factor.",
            )
        ],
    )

    summary = extract_candidate_signals_for_transcript(transcript.id, db)

    assert summary["candidate_count"] == 0


def test_representative_turn_cannot_be_authoritative_evidence(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript, turns = create_transcript(db, ["optimize_rep", "advisor"])
    turns[0].text = "Our support model gives advisors much more client time."
    turns[1].text = "That sounds good."
    db.commit()
    mock_output(
        monkeypatch,
        [
            candidate_item(
                item_type="driver",
                quote=turns[0].text,
                source_turn_ids=[turns[0].id],
                factor_kind="desired_outcome",
                decision_direction="supports_move",
            )
        ],
    )

    extract_candidate_signals_for_transcript(transcript.id, db)

    assert saved_candidates(db, transcript.id) == []
