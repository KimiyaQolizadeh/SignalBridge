"""Durable, run-scoped pipeline artifacts and replay helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from typing import Any
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import settings
from ..models import (
    AnalysisRun,
    CandidateSignal,
    CandidateSnapshot,
    DeduplicationSnapshot,
    FinalRankingSnapshot,
    FinalSignal,
    ScoringSnapshot,
    SpeakerClassificationSnapshot,
    Transcript,
    TranscriptTurn,
    ValidationSnapshot,
)
from .prompt_loader import prompt_sha256
from .eligibility_policy import validation_allows_business_pipeline


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def configuration_snapshot() -> dict[str, Any]:
    return {
        "models": {
            "speaker_classifier": settings.speaker_classifier_model,
            "candidate_extractor": settings.candidate_extractor_model,
            "evidence_validator": settings.evidence_validator_model,
            "business_scorer": settings.business_scorer_model,
            "final_reranker": settings.final_reranker_model,
            "embedding": settings.embedding_model,
        },
        "prompts": {
            name: prompt_sha256(name)
            for name in (
                "01_speaker_role_classifier.txt",
                "02_candidate_signal_extractor.txt",
                "03_evidence_validator.txt",
                "04_business_scorer.txt",
                "05_final_reranker.txt",
            )
        },
        "deduplication_threshold": settings.dedup_similarity_threshold,
        "deduplication_mode": settings.dedup_experiment_mode,
    }


def create_analysis_run(
    db: Session,
    transcript: Transcript,
    *,
    run_type: str = "full",
    source_run_id: str | None = None,
    run_id: str | None = None,
) -> AnalysisRun:
    """Create a run record; source runs and their snapshots are never overwritten."""
    run = AnalysisRun(
        id=run_id or str(uuid4()),
        transcript_id=transcript.id,
        status="running",
        run_type=run_type,
        source_run_id=source_run_id,
        input_sha256=sha256(transcript.raw_text.encode("utf-8")).hexdigest(),
        started_at=utcnow(),
        retry_count=0,
        code_version=None,
        configuration_snapshot=configuration_snapshot(),
    )
    db.add(run)
    db.commit()
    return run


def mark_run_completed(db: Session, run_id: str, summary: dict[str, Any]) -> None:
    run = db.get(AnalysisRun, run_id)
    if run is None:
        return
    run.status = "completed"
    run.completed_at = utcnow()
    run.summary = summary
    db.commit()


def mark_run_failed(
    db: Session,
    run_id: str,
    *,
    failed_stage: str,
    error_category: str,
    error_message: str,
) -> None:
    run = db.get(AnalysisRun, run_id)
    if run is None:
        return
    run.status = "failed"
    run.completed_at = utcnow()
    run.failed_stage = failed_stage
    run.error_category = error_category
    run.error_message = error_message
    db.commit()


def latest_run_for_transcript(
    db: Session, transcript_id: int, *, completed_only: bool = False
) -> AnalysisRun | None:
    statement = select(AnalysisRun).where(AnalysisRun.transcript_id == transcript_id)
    if completed_only:
        statement = statement.where(AnalysisRun.status == "completed")
    return db.scalar(statement.order_by(AnalysisRun.started_at.desc(), AnalysisRun.id.desc()))


def snapshot_speaker_classifications(
    db: Session,
    run_id: str,
    turns: list[TranscriptTurn],
    rationales: dict[int, str] | None = None,
    raw_responses: dict[int, dict[str, Any] | list[Any]] | None = None,
) -> None:
    rationale_by_id = rationales or {}
    raw_by_id = raw_responses or {}
    db.add_all(
        SpeakerClassificationSnapshot(
            analysis_run_id=run_id,
            turn_id=turn.id,
            turn_index=turn.turn_index,
            original_speaker_label=turn.raw_speaker_label,
            classified_role=turn.inferred_role or "unknown",
            confidence=turn.role_confidence or 0.0,
            rationale=rationale_by_id.get(turn.id),
            model_identifier=settings.speaker_classifier_model,
            prompt_hash=prompt_sha256("01_speaker_role_classifier.txt"),
            raw_response=raw_by_id.get(turn.id),
        )
        for turn in turns
    )
    db.commit()


def candidate_snapshot_for(
    db: Session, run_id: str, candidate_id: int
) -> CandidateSnapshot | None:
    return db.scalar(
        select(CandidateSnapshot).where(
            CandidateSnapshot.analysis_run_id == run_id,
            CandidateSnapshot.legacy_candidate_id == candidate_id,
        )
    )


def snapshot_validation(
    db: Session,
    *,
    run_id: str,
    candidate_id: int,
    structured_findings: dict[str, Any] | None,
    deterministic_findings: dict[str, Any],
    verdict: str,
    rejection_reasons: list[str],
    review_reasons: list[str],
    primary_reason: str | None,
    raw_response: str | None,
    token_usage: dict[str, Any] | None,
    retry_count: int,
    started_at: datetime,
    completed_at: datetime,
) -> None:
    candidate_snapshot = candidate_snapshot_for(db, run_id, candidate_id)
    if candidate_snapshot is None:
        return
    existing = db.scalar(
        select(ValidationSnapshot).where(
            ValidationSnapshot.analysis_run_id == run_id,
            ValidationSnapshot.candidate_snapshot_id == candidate_snapshot.id,
        )
    )
    if existing is not None:
        return
    db.add(ValidationSnapshot(
        analysis_run_id=run_id,
        candidate_snapshot_id=candidate_snapshot.id,
        structured_findings=structured_findings,
        deterministic_findings=deterministic_findings,
        derived_verdict=verdict,
        rejection_reasons=rejection_reasons,
        review_reasons=review_reasons,
        primary_reason=primary_reason,
        prompt_hash=prompt_sha256("03_evidence_validator.txt"),
        model_identifier=settings.evidence_validator_model,
        raw_response=raw_response,
        token_usage=token_usage,
        retry_count=retry_count,
        started_at=started_at,
        completed_at=completed_at,
    ))


def snapshot_downstream(db: Session, run_id: str) -> None:
    """Persist scoring, canonical/duplicate, and ranking audit state for one run."""
    candidates = list(db.scalars(
        select(CandidateSignal).where(CandidateSignal.analysis_run_id == run_id)
    ))
    groups: dict[str, list[CandidateSignal]] = {}
    for candidate in candidates:
        if candidate.duplicate_group_id:
            groups.setdefault(candidate.duplicate_group_id, []).append(candidate)
    canonical_by_group = {
        group_id: next((item for item in members if item.is_canonical), None)
        for group_id, members in groups.items()
    }
    finals = {
        item.signal_id: item
        for item in db.scalars(select(FinalSignal).where(FinalSignal.analysis_run_id == run_id))
    }
    for candidate in candidates:
        snapshot = candidate_snapshot_for(db, run_id, candidate.id)
        if snapshot is None:
            continue
        score = candidate.score
        db.add(ScoringSnapshot(
            analysis_run_id=run_id,
            candidate_snapshot_id=snapshot.id,
            eligible=score is not None and score.final_score is not None,
            output=None if score is None else {
                "validator_verdict": score.validator_verdict,
                "advisor_ownership": score.advisor_ownership,
                "decision_impact": score.decision_impact,
                "explicitness": score.explicitness,
                "urgency": score.urgency,
                "evidence_quality": score.evidence_quality,
                "final_score": score.final_score,
                "support_score": score.support_score,
                "advisor_side_score": score.advisor_side_score,
                "false_positive_risk": score.false_positive_risk,
            },
        ))
        db.add(DeduplicationSnapshot(
            analysis_run_id=run_id,
            candidate_snapshot_id=snapshot.id,
            eligible=score is not None and score.final_score is not None,
            is_canonical=candidate.is_canonical,
            duplicate_group_id=candidate.duplicate_group_id,
            rationale={
                "canonical_candidate_id": (
                    canonical_by_group.get(candidate.duplicate_group_id).id
                    if candidate.duplicate_group_id
                    and canonical_by_group.get(candidate.duplicate_group_id) is not None
                    else None
                ),
                "relationship": "canonical" if candidate.is_canonical else "duplicate_of_canonical",
                "group_member_candidate_ids": [
                    item.id for item in groups.get(candidate.duplicate_group_id, [])
                ],
            },
        ))
        final = finals.get(candidate.id)
        db.add(FinalRankingSnapshot(
            analysis_run_id=run_id,
            candidate_snapshot_id=snapshot.id,
            eligible=(
                score is not None
                and validation_allows_business_pipeline(
                    score.validator_verdict, score.rejection_reason
                )
                and score.final_score is not None
                and candidate.is_canonical
            ),
            selected=final is not None,
            item_type=candidate.item_type,
            rank=final.rank if final else None,
            output=None if final is None else {"item_type": final.item_type, "rank": final.rank},
        ))
    db.commit()


def persist_validation_diagnostics(
    db: Session, run_id: str, diagnostics: list[dict[str, Any]]
) -> None:
    now = utcnow()
    for diagnostic in diagnostics:
        candidate_id = diagnostic.get("candidate_id")
        if not isinstance(candidate_id, int):
            continue
        verdict = str(diagnostic.get("derived_verdict") or "reject")
        hard_reason = diagnostic.get("hard_failure_reason")
        review_reason = diagnostic.get("needs_review_reason")
        snapshot_validation(
            db,
            run_id=run_id,
            candidate_id=candidate_id,
            structured_findings=diagnostic.get("structured_findings"),
            deterministic_findings=diagnostic.get("deterministic_prechecks") or {
                "status": "rejected_before_model" if hard_reason else "passed"
            },
            verdict=verdict,
            rejection_reasons=[str(hard_reason)] if hard_reason else [],
            review_reasons=[str(review_reason)] if review_reason else [],
            primary_reason=str(hard_reason or review_reason) if hard_reason or review_reason else None,
            raw_response=diagnostic.get("raw_response"),
            token_usage=diagnostic.get("token_usage"),
            retry_count=int(diagnostic.get("response_retry_count") or diagnostic.get("retry_count") or 0),
            started_at=now,
            completed_at=now,
        )
    db.commit()
