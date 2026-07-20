from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import (
    AnalysisRun,
    CandidateSnapshot,
    ExtractionBatch,
    FinalRankingSnapshot,
    ScoringSnapshot,
    ValidationSnapshot,
)
from ..services.run_replay import RunNotFoundError, RunNotReplayableError, replay_validation


router = APIRouter(tags=["analysis-runs"])


def _run_payload(run: AnalysisRun) -> dict:
    return {
        "run_id": run.id,
        "transcript_id": run.transcript_id,
        "status": run.status,
        "run_type": run.run_type,
        "source_run_id": run.source_run_id,
        "started_at": run.started_at,
        "completed_at": run.completed_at,
        "failed_stage": run.failed_stage,
        "error_category": run.error_category,
        "summary": run.summary or {},
    }


@router.get("/api/transcripts/{transcript_id}/runs")
def list_runs(transcript_id: int, db: Session = Depends(get_db)) -> list[dict]:
    runs = db.scalars(select(AnalysisRun).where(AnalysisRun.transcript_id == transcript_id).order_by(AnalysisRun.started_at.desc())).all()
    return [_run_payload(run) for run in runs]


@router.get("/api/runs/{run_id}")
def get_run(run_id: str, db: Session = Depends(get_db)) -> dict:
    run = db.get(AnalysisRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Analysis run not found")
    return _run_payload(run) | {"configuration_snapshot": run.configuration_snapshot}


@router.get("/api/runs/{run_id}/diagnostics")
def get_run_diagnostics(run_id: str, db: Session = Depends(get_db)) -> dict:
    run = db.get(AnalysisRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Analysis run not found")
    validations = list(db.scalars(select(ValidationSnapshot).where(ValidationSnapshot.analysis_run_id == run_id)))
    verdicts = {key: sum(item.derived_verdict == key for item in validations) for key in ("pass", "needs_review", "reject")}
    return _run_payload(run) | {"validation_verdicts": verdicts}


@router.get("/api/runs/{run_id}/extraction-batches")
def get_extraction_batches(run_id: str, db: Session = Depends(get_db)) -> list[dict]:
    batches = db.scalars(select(ExtractionBatch).where(ExtractionBatch.analysis_run_id == run_id).order_by(ExtractionBatch.batch_index)).all()
    return [{"id": batch.id, "batch_index": batch.batch_index, "status": batch.status, "input_turn_ids": batch.input_turn_ids, "raw_response": batch.raw_response, "raw_item_count": batch.raw_item_count, "parsed_item_count": batch.parsed_item_count, "post_filter_item_count": batch.post_filter_item_count, "token_usage": batch.token_usage, "retry_count": batch.retry_count} for batch in batches]


@router.get("/api/runs/{run_id}/candidates")
def get_run_candidates(run_id: str, db: Session = Depends(get_db)) -> list[dict]:
    items = db.scalars(select(CandidateSnapshot).where(CandidateSnapshot.analysis_run_id == run_id).order_by(CandidateSnapshot.id)).all()
    return [{"id": item.id, "item_type": item.item_type, "category": item.category, "advisor_quote": item.advisor_quote, "timestamp": item.timestamp, "evidence_strength": item.evidence_strength, "rationale": item.rationale, "source_turn_ids": item.source_turn_ids, "ownership": item.ownership} for item in items]


@router.get("/api/runs/{run_id}/final-signals")
def get_run_finals(run_id: str, db: Session = Depends(get_db)) -> list[dict]:
    finals = db.scalars(select(FinalRankingSnapshot).where(FinalRankingSnapshot.analysis_run_id == run_id, FinalRankingSnapshot.selected.is_(True)).order_by(FinalRankingSnapshot.item_type, FinalRankingSnapshot.rank)).all()
    payload = []
    for item in finals:
        candidate = db.get(CandidateSnapshot, item.candidate_snapshot_id)
        score = db.scalar(select(ScoringSnapshot).where(ScoringSnapshot.analysis_run_id == run_id, ScoringSnapshot.candidate_snapshot_id == item.candidate_snapshot_id))
        if candidate is None:
            continue
        score_output = score.output if score and score.output else {}
        payload.append({
            "transcript_id": candidate.transcript_id,
            "item_type": item.item_type,
            "rank": item.rank,
            "category": candidate.category,
            "advisor_quote": candidate.advisor_quote,
            "timestamp": candidate.timestamp,
            "evidence_strength": candidate.evidence_strength,
            "rationale": candidate.rationale,
            "final_score": score_output.get("final_score"),
            "validator_verdict": score_output.get("validator_verdict"),
        })
    return payload


@router.post("/api/runs/{run_id}/replay-validation", status_code=status.HTTP_201_CREATED)
def replay_run_validation(run_id: str, db: Session = Depends(get_db)) -> dict:
    try:
        return _run_payload(replay_validation(run_id, db))
    except RunNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from None
    except RunNotReplayableError as error:
        raise HTTPException(status_code=409, detail=str(error)) from None
