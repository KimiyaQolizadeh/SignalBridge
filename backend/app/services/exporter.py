import csv
import io
import json
import re

from sqlalchemy import case, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ..models import (
    AnalysisRun,
    CandidateSignal,
    FinalSignal,
    SignalScore,
    Transcript,
    TranscriptTurn,
)


FINAL_COLUMNS = [
    # Preserve the original eight columns in their established order.
    "transcript_id",
    "item_type",
    "rank",
    "category",
    "advisor_quote",
    "timestamp",
    "evidence_strength",
    "rationale",
    # Additive final-signal metadata.
    "analysis_run_id",
    "validation_verdict",
    "business_score",
    "selection_reason",
    "supporting_evidence",
    "adjacent_context",
    "final_signal_id",
    "canonical",
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

_REFERENTIAL_EVIDENCE_PATTERN = re.compile(
    r"\b(it|this|that|there|those|these|they|them)\b", re.IGNORECASE
)


class TranscriptNotFoundError(Exception):
    """Raised when an export is requested for an unknown transcript."""


class ExportError(Exception):
    """A safe error for export query or serialization failures."""


def _latest_completed_runs(db: Session) -> dict[int, str]:
    runs = db.scalars(
        select(AnalysisRun)
        .where(AnalysisRun.status == "completed")
        .order_by(
            AnalysisRun.transcript_id,
            AnalysisRun.started_at.desc(),
            AnalysisRun.id.desc(),
        )
    ).all()
    latest: dict[int, str] = {}
    for run in runs:
        latest.setdefault(run.transcript_id, run.id)
    return latest


def _query_rows(
    db: Session, transcript_id: int | None = None
) -> list[tuple[FinalSignal, CandidateSignal, SignalScore]]:
    """Return only canonical finals from the run represented by the UI."""
    statement = (
        select(FinalSignal, CandidateSignal, SignalScore)
        .join(CandidateSignal, FinalSignal.signal_id == CandidateSignal.id)
        .join(SignalScore, SignalScore.signal_id == CandidateSignal.id)
        .where(CandidateSignal.is_canonical.is_(True))
    )
    if transcript_id is not None:
        statement = statement.where(FinalSignal.transcript_id == transcript_id)
    queried = list(
        db.execute(
            statement.order_by(
                FinalSignal.transcript_id,
                case(
                    (FinalSignal.item_type == "driver", 0),
                    (FinalSignal.item_type == "blocker", 1),
                    else_=2,
                ),
                FinalSignal.rank,
                FinalSignal.id,
            )
        ).all()
    )
    latest_by_transcript = _latest_completed_runs(db)
    return [
        row
        for row in queried
        if (
            (
                row[0].transcript_id in latest_by_transcript
                and row[0].analysis_run_id == latest_by_transcript[row[0].transcript_id]
            )
            or (
                row[0].transcript_id not in latest_by_transcript
                and row[0].analysis_run_id is None
            )
        )
    ]


def _same_run(statement, run_id: str | None):
    return (
        statement.where(CandidateSignal.analysis_run_id == run_id)
        if run_id is not None
        else statement.where(CandidateSignal.analysis_run_id.is_(None))
    )


def _supporting_evidence(
    db: Session, final: FinalSignal, candidate: CandidateSignal
) -> list[dict]:
    if not candidate.duplicate_group_id:
        return []
    statement = select(CandidateSignal).where(
        CandidateSignal.transcript_id == candidate.transcript_id,
        CandidateSignal.duplicate_group_id == candidate.duplicate_group_id,
        CandidateSignal.id != candidate.id,
        CandidateSignal.is_canonical.is_(False),
    )
    statement = _same_run(statement, final.analysis_run_id)
    duplicates = db.scalars(statement.order_by(CandidateSignal.id)).all()
    return [
        {
            "quote": duplicate.advisor_quote,
            "timestamp": duplicate.timestamp,
            "relationship": "supporting_reason",
        }
        for duplicate in duplicates
    ]


def _adjacent_context(db: Session, candidate: CandidateSignal) -> list[dict]:
    if not _REFERENTIAL_EVIDENCE_PATTERN.search(candidate.advisor_quote):
        return []
    source_ids = (
        {item for item in candidate.source_turn_ids if isinstance(item, int)}
        if isinstance(candidate.source_turn_ids, list)
        else set()
    )
    if not source_ids:
        return []
    turns = list(
        db.scalars(
            select(TranscriptTurn)
            .where(TranscriptTurn.transcript_id == candidate.transcript_id)
            .order_by(TranscriptTurn.turn_index, TranscriptTurn.id)
        ).all()
    )
    positions = [index for index, turn in enumerate(turns) if turn.id in source_ids]
    adjacent_positions = {
        position + offset
        for position in positions
        for offset in (-1, 1)
        if 0 <= position + offset < len(turns)
        and turns[position + offset].id not in source_ids
    }
    return [
        {
            "turn_id": turns[position].id,
            "speaker": turns[position].raw_speaker_label,
            "text": turns[position].text,
            "timestamp": turns[position].timestamp,
        }
        for position in sorted(adjacent_positions)
    ]


def _selection_reason(candidate: CandidateSignal, score: SignalScore) -> str:
    threshold = 3.5 if candidate.evidence_strength == "explicit" else 4.0
    return (
        "above_threshold"
        if score.final_score is not None and score.final_score >= threshold
        else "best_grounded_fallback"
    )


def _final_row(
    db: Session,
    final: FinalSignal,
    candidate: CandidateSignal,
    score: SignalScore,
) -> dict:
    return {
        "export_version": "1.0",
        "transcript_id": final.transcript_id,
        "analysis_run_id": final.analysis_run_id,
        "final_signal_id": final.id,
        "item_type": final.item_type,
        "rank": final.rank,
        "category": candidate.category,
        "validation_verdict": score.validator_verdict,
        "business_score": score.final_score,
        "selection_reason": _selection_reason(candidate, score),
        "advisor_quote": candidate.advisor_quote,
        "timestamp": candidate.timestamp,
        "evidence_strength": candidate.evidence_strength,
        "rationale": candidate.rationale,
        "supporting_evidence": _supporting_evidence(db, final, candidate),
        "adjacent_context": _adjacent_context(db, candidate),
        "canonical": True,
    }


def _debug_row(
    db: Session,
    final: FinalSignal,
    candidate: CandidateSignal,
    score: SignalScore,
) -> dict:
    return {
        **_final_row(db, final, candidate, score),
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
        return [
            _final_row(db, final, candidate, score)
            for final, candidate, score in rows
        ]
    except TranscriptNotFoundError:
        raise
    except SQLAlchemyError:
        db.rollback()
        raise ExportError("Final results could not be loaded") from None


def get_debug_results_rows(transcript_id: int, db: Session) -> list[dict]:
    try:
        _require_transcript(transcript_id, db)
        rows = _query_rows(db, transcript_id)
        return [
            _debug_row(db, final, candidate, score)
            for final, candidate, score in rows
        ]
    except TranscriptNotFoundError:
        raise
    except SQLAlchemyError:
        db.rollback()
        raise ExportError("Debug results could not be loaded") from None


def _supporting_columns(rows: list[dict]) -> list[str]:
    count = max((len(row.get("supporting_evidence", [])) for row in rows), default=0)
    return [
        column
        for index in range(1, max(1, count) + 1)
        for column in (f"supporting_quote_{index}", f"supporting_timestamp_{index}")
    ]


def _csv_row(row: dict, supporting_columns: list[str]) -> dict:
    converted = {key: value for key, value in row.items() if key != "export_version"}
    converted["supporting_evidence"] = json.dumps(
        converted.get("supporting_evidence", []), ensure_ascii=False
    )
    converted["adjacent_context"] = json.dumps(
        converted.get("adjacent_context", []), ensure_ascii=False
    )
    for column in supporting_columns:
        converted[column] = ""
    for index, evidence in enumerate(row.get("supporting_evidence", []), start=1):
        converted[f"supporting_quote_{index}"] = evidence["quote"]
        converted[f"supporting_timestamp_{index}"] = evidence["timestamp"] or ""
    if converted.get("business_score") is not None:
        converted["business_score"] = f"{converted['business_score']:.2f}"
    if converted.get("final_score") is not None:
        converted["final_score"] = f"{converted['final_score']:.2f}"
    return converted


def _rows_to_csv(rows: list[dict], *, debug: bool) -> str:
    output = io.StringIO(newline="")
    supporting_columns = _supporting_columns(rows)
    columns = [*(DEBUG_COLUMNS if debug else FINAL_COLUMNS), *supporting_columns]
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    writer.writerows(_csv_row(row, supporting_columns) for row in rows)
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
            _debug_row(db, final, candidate, score)
            if debug
            else _final_row(db, final, candidate, score)
            for final, candidate, score in queried_rows
        ]
        return _rows_to_csv(rows, debug=debug)
    except SQLAlchemyError:
        db.rollback()
        raise ExportError("Batch export could not be loaded") from None
    except (csv.Error, TypeError, ValueError):
        raise ExportError("Batch CSV export could not be generated") from None
