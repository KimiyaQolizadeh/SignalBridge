import csv
import io
import json

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ..models import CandidateSignal, FinalSignal, SignalScore, Transcript


FINAL_COLUMNS = [
    "transcript_id",
    "item_type",
    "rank",
    "category",
    "advisor_quote",
    "timestamp",
    "evidence_strength",
    "rationale",
]

DEBUG_COLUMNS = FINAL_COLUMNS + [
    "final_score",
    "validator_verdict",
    "support_score",
    "advisor_side_score",
    "false_positive_risk",
    "advisor_ownership",
    "decision_impact",
    "explicitness",
    "urgency",
    "evidence_quality",
    "duplicate_group_id",
    "is_canonical",
]


class TranscriptNotFoundError(Exception):
    """Raised when an export is requested for an unknown transcript."""


class ExportError(Exception):
    """A safe error for export query or serialization failures."""


def _query_rows(
    db: Session, transcript_id: int | None = None
) -> list[tuple[FinalSignal, CandidateSignal, SignalScore]]:
    statement = (
        select(FinalSignal, CandidateSignal, SignalScore)
        .join(CandidateSignal, FinalSignal.signal_id == CandidateSignal.id)
        .join(SignalScore, SignalScore.signal_id == CandidateSignal.id)
    )
    if transcript_id is not None:
        statement = statement.where(FinalSignal.transcript_id == transcript_id)
    return list(
        db.execute(
            statement.order_by(
                FinalSignal.transcript_id,
                FinalSignal.item_type,
                FinalSignal.rank,
            )
        ).all()
    )


def _final_row(final: FinalSignal, candidate: CandidateSignal) -> dict:
    return {
        "transcript_id": final.transcript_id,
        "item_type": final.item_type,
        "rank": final.rank,
        "category": candidate.category,
        "advisor_quote": candidate.advisor_quote,
        "timestamp": candidate.timestamp,
        "evidence_strength": candidate.evidence_strength,
        "rationale": candidate.rationale,
    }


def _debug_row(
    final: FinalSignal, candidate: CandidateSignal, score: SignalScore
) -> dict:
    return {
        **_final_row(final, candidate),
        "final_score": score.final_score,
        "validator_verdict": score.validator_verdict,
        "support_score": score.support_score,
        "advisor_side_score": score.advisor_side_score,
        "false_positive_risk": score.false_positive_risk,
        "advisor_ownership": score.advisor_ownership,
        "decision_impact": score.decision_impact,
        "explicitness": score.explicitness,
        "urgency": score.urgency,
        "evidence_quality": score.evidence_quality,
        "duplicate_group_id": candidate.duplicate_group_id,
        "is_canonical": candidate.is_canonical,
    }


def _require_transcript(transcript_id: int, db: Session) -> None:
    if db.get(Transcript, transcript_id) is None:
        raise TranscriptNotFoundError("Transcript not found")


def get_final_results_rows(transcript_id: int, db: Session) -> list[dict]:
    try:
        _require_transcript(transcript_id, db)
        rows = _query_rows(db, transcript_id)
    except TranscriptNotFoundError:
        raise
    except SQLAlchemyError:
        db.rollback()
        raise ExportError("Final results could not be loaded") from None

    return [_final_row(final, candidate) for final, candidate, _score in rows]


def get_debug_results_rows(transcript_id: int, db: Session) -> list[dict]:
    try:
        _require_transcript(transcript_id, db)
        rows = _query_rows(db, transcript_id)
    except TranscriptNotFoundError:
        raise
    except SQLAlchemyError:
        db.rollback()
        raise ExportError("Debug results could not be loaded") from None

    return [_debug_row(final, candidate, score) for final, candidate, score in rows]


def _rows_to_csv(rows: list[dict], *, debug: bool) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=DEBUG_COLUMNS if debug else FINAL_COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _rows_to_jsonl(rows: list[dict]) -> str:
    if not rows:
        return ""
    return "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"


def export_transcript_csv(
    transcript_id: int, db: Session, debug: bool = False
) -> str:
    try:
        rows = (
            get_debug_results_rows(transcript_id, db)
            if debug
            else get_final_results_rows(transcript_id, db)
        )
        return _rows_to_csv(rows, debug=debug)
    except (TranscriptNotFoundError, ExportError):
        raise
    except (csv.Error, TypeError, ValueError):
        raise ExportError("CSV export could not be generated") from None


def export_transcript_jsonl(
    transcript_id: int, db: Session, debug: bool = False
) -> str:
    try:
        rows = (
            get_debug_results_rows(transcript_id, db)
            if debug
            else get_final_results_rows(transcript_id, db)
        )
        return _rows_to_jsonl(rows)
    except (TranscriptNotFoundError, ExportError):
        raise
    except (TypeError, ValueError):
        raise ExportError("JSONL export could not be generated") from None


def export_all_transcripts_csv(db: Session, debug: bool = False) -> str:
    try:
        queried_rows = _query_rows(db)
        rows = [
            _debug_row(final, candidate, score)
            if debug
            else _final_row(final, candidate)
            for final, candidate, score in queried_rows
        ]
        return _rows_to_csv(rows, debug=debug)
    except SQLAlchemyError:
        db.rollback()
        raise ExportError("Batch export could not be loaded") from None
    except (csv.Error, TypeError, ValueError):
        raise ExportError("Batch CSV export could not be generated") from None
