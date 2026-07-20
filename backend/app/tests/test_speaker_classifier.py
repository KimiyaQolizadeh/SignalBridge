from collections.abc import Generator

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.app.database import Base
from backend.app.llm_schemas import SpeakerRoleBatchOutput, SpeakerRoleItem
from backend.app.models import Transcript, TranscriptTurn
from backend.app.services import speaker_classifier
from backend.app.services.speaker_classifier import classify_speakers_for_transcript


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


def create_transcript_with_turns(db: Session, count: int = 4) -> Transcript:
    transcript = Transcript(file_name="synthetic.txt", raw_text="Synthetic text")
    db.add(transcript)
    db.flush()
    db.add_all(
        TranscriptTurn(
            transcript_id=transcript.id,
            turn_index=index,
            timestamp=f"00:00:0{index}",
            raw_speaker_label=f"Speaker {index}",
            text=f"Synthetic turn {index}",
        )
        for index in range(count)
    )
    db.commit()
    db.refresh(transcript)
    return transcript


def get_turns(db: Session, transcript_id: int) -> list[TranscriptTurn]:
    return list(
        db.scalars(
            select(TranscriptTurn)
            .where(TranscriptTurn.transcript_id == transcript_id)
            .order_by(TranscriptTurn.turn_index)
        ).all()
    )


def test_low_confidence_role_becomes_unknown(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript_with_turns(db, count=1)
    turn = get_turns(db, transcript.id)[0]

    monkeypatch.setattr(
        speaker_classifier,
        "call_llm_json",
        lambda **_: SpeakerRoleBatchOutput(
            items=[
                SpeakerRoleItem(
                    turn_id=turn.id,
                    inferred_role="advisor",
                    confidence=0.64,
                    reason="Synthetic test reason",
                )
            ]
        ),
    )

    summary = classify_speakers_for_transcript(transcript.id, db)
    db.refresh(turn)

    assert turn.inferred_role == "unknown"
    assert turn.role_confidence == 0.64
    assert summary["unknown_turns"] == 1


def test_invalid_returned_turn_id_is_ignored(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript_with_turns(db, count=1)
    turn = get_turns(db, transcript.id)[0]
    monkeypatch.setattr(
        speaker_classifier,
        "call_llm_json",
        lambda **_: SpeakerRoleBatchOutput(
            items=[
                SpeakerRoleItem(
                    turn_id=999999,
                    inferred_role="optimize_rep",
                    confidence=0.99,
                    reason="Synthetic test reason",
                )
            ]
        ),
    )

    classify_speakers_for_transcript(transcript.id, db)
    db.refresh(turn)

    assert turn.inferred_role == "unknown"
    assert turn.role_confidence == 0.0


def test_missing_returned_turn_remains_unknown(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript_with_turns(db, count=2)
    first_turn, missing_turn = get_turns(db, transcript.id)
    monkeypatch.setattr(
        speaker_classifier,
        "call_llm_json",
        lambda **_: SpeakerRoleBatchOutput(
            items=[
                SpeakerRoleItem(
                    turn_id=first_turn.id,
                    inferred_role="advisor",
                    confidence=0.9,
                    reason="Synthetic test reason",
                )
            ]
        ),
    )

    classify_speakers_for_transcript(transcript.id, db)
    db.refresh(missing_turn)

    assert missing_turn.inferred_role == "unknown"
    assert missing_turn.role_confidence == 0.0


def test_summary_counts_are_correct(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript_with_turns(db, count=4)
    turns = get_turns(db, transcript.id)
    monkeypatch.setattr(
        speaker_classifier,
        "call_llm_json",
        lambda **_: SpeakerRoleBatchOutput(
            items=[
                SpeakerRoleItem(
                    turn_id=turns[0].id,
                    inferred_role="advisor",
                    confidence=0.95,
                    reason="Synthetic test reason",
                ),
                SpeakerRoleItem(
                    turn_id=turns[1].id,
                    inferred_role="optimize_rep",
                    confidence=0.9,
                    reason="Synthetic test reason",
                ),
                SpeakerRoleItem(
                    turn_id=turns[2].id,
                    inferred_role="mixed",
                    confidence=0.8,
                    reason="Synthetic test reason",
                ),
            ]
        ),
    )

    summary = classify_speakers_for_transcript(transcript.id, db)

    assert summary == {
        "transcript_id": transcript.id,
        "status": "speakers_classified",
        "turn_count": 4,
        "advisor_turns": 1,
        "optimize_rep_turns": 1,
        "unknown_turns": 1,
        "mixed_turns": 1,
    }
