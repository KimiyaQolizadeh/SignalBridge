import csv
import io
import json
from collections.abc import Generator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.app.database import Base
from backend.app.models import CandidateSignal, FinalSignal, SignalScore, Transcript
from backend.app.services.exporter import (
    DEBUG_COLUMNS,
    FINAL_COLUMNS,
    export_all_transcripts_csv,
    export_transcript_csv,
    export_transcript_jsonl,
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


def create_final_signal(
    db: Session, *, file_name: str, item_type: str = "driver", rank: int = 1
) -> tuple[Transcript, CandidateSignal]:
    transcript = Transcript(file_name=file_name, raw_text="Synthetic content")
    db.add(transcript)
    db.flush()
    candidate = CandidateSignal(
        transcript_id=transcript.id,
        item_type=item_type,
        category="synthetic_category",
        advisor_quote="Synthetic advisor evidence.",
        timestamp="00:00:01",
        evidence_strength="explicit",
        rationale="Synthetic decision-relevant rationale.",
        extraction_confidence=0.9,
        source_turn_ids=[1],
        duplicate_group_id="g_synthetic",
        is_canonical=True,
    )
    db.add(candidate)
    db.flush()
    candidate.score = SignalScore(
        signal_id=candidate.id,
        validator_verdict="pass",
        support_score=0.95,
        advisor_side_score=0.96,
        false_positive_risk=0.05,
        advisor_ownership=5,
        decision_impact=4,
        explicitness=5,
        urgency=3,
        evidence_quality=5,
        final_score=4.55,
    )
    db.add(
        FinalSignal(
            transcript_id=transcript.id,
            signal_id=candidate.id,
            item_type=item_type,
            rank=rank,
        )
    )
    db.commit()
    return transcript, candidate


def csv_rows(content: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(content)))


def test_final_export_has_exact_assignment_columns(db: Session) -> None:
    transcript, _ = create_final_signal(db, file_name="one.txt")

    content = export_transcript_csv(transcript.id, db)
    reader = csv.DictReader(io.StringIO(content))

    assert reader.fieldnames == FINAL_COLUMNS
    assert len(list(reader)) == 1


def test_debug_export_includes_validation_and_scoring_columns(db: Session) -> None:
    transcript, _ = create_final_signal(db, file_name="one.txt")

    content = export_transcript_csv(transcript.id, db, debug=True)
    reader = csv.DictReader(io.StringIO(content))

    assert reader.fieldnames == DEBUG_COLUMNS
    assert "validator_verdict" in reader.fieldnames
    assert "final_score" in reader.fieldnames
    assert "duplicate_group_id" in reader.fieldnames


def test_empty_final_export_returns_headers_only(db: Session) -> None:
    transcript = Transcript(file_name="empty.txt", raw_text="Synthetic content")
    db.add(transcript)
    db.commit()

    content = export_transcript_csv(transcript.id, db)

    assert content.strip() == ",".join(FINAL_COLUMNS)
    assert csv_rows(content) == []


def test_jsonl_exports_one_object_per_row(db: Session) -> None:
    transcript, _ = create_final_signal(db, file_name="one.txt")

    content = export_transcript_jsonl(transcript.id, db)
    lines = content.splitlines()

    assert len(lines) == 1
    assert set(json.loads(lines[0])) == set(FINAL_COLUMNS)


def test_batch_export_includes_multiple_transcripts(db: Session) -> None:
    first, _ = create_final_signal(db, file_name="one.txt")
    second, _ = create_final_signal(db, file_name="two.txt", item_type="blocker")

    rows = csv_rows(export_all_transcripts_csv(db))

    assert {int(row["transcript_id"]) for row in rows} == {first.id, second.id}
    assert len(rows) == 2
