"""Safe replay workflows over immutable run artifacts."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import (
    AnalysisRun,
    CandidateSignal,
    CandidateSnapshot,
    SpeakerClassificationSnapshot,
    Transcript,
)
from .evidence_validator import validate_evidence_for_transcript
from .reranker import rerank_final_signals_for_transcript
from .run_persistence import (
    create_analysis_run,
    mark_run_completed,
    mark_run_failed,
    snapshot_downstream,
)
from .scorer import score_signals_for_transcript
from .signal_deduplicator import deduplicate_signals_for_transcript


class RunNotFoundError(Exception):
    pass


class RunNotReplayableError(Exception):
    pass


def replay_validation(source_run_id: str, db: Session) -> AnalysisRun:
    """Create a new immutable run from saved candidates, skipping upstream model calls."""
    source = db.get(AnalysisRun, source_run_id)
    if source is None:
        raise RunNotFoundError("Analysis run not found")
    if source.status != "completed":
        raise RunNotReplayableError("Only completed runs can be replayed")
    transcript = db.get(Transcript, source.transcript_id)
    if transcript is None:
        raise RunNotFoundError("Transcript not found")

    replay = create_analysis_run(
        db,
        transcript,
        run_type="replay_validation",
        source_run_id=source.id,
    )
    try:
        source_speakers = list(db.scalars(select(SpeakerClassificationSnapshot).where(
            SpeakerClassificationSnapshot.analysis_run_id == source.id
        )))
        db.add_all(SpeakerClassificationSnapshot(
            analysis_run_id=replay.id,
            turn_id=item.turn_id,
            turn_index=item.turn_index,
            original_speaker_label=item.original_speaker_label,
            classified_role=item.classified_role,
            confidence=item.confidence,
            rationale=item.rationale,
            model_identifier=item.model_identifier,
            prompt_hash=item.prompt_hash,
            raw_response=item.raw_response,
        ) for item in source_speakers)

        source_candidates = list(db.scalars(select(CandidateSnapshot).where(
            CandidateSnapshot.analysis_run_id == source.id
        ).order_by(CandidateSnapshot.id)))
        for item in source_candidates:
            candidate = CandidateSignal(
                transcript_id=transcript.id,
                analysis_run_id=replay.id,
                item_type=item.item_type,
                category=item.category,
                advisor_quote=item.advisor_quote,
                timestamp=item.timestamp,
                evidence_strength=item.evidence_strength,
                rationale=item.rationale,
                extraction_confidence=item.extraction_confidence,
                source_turn_ids=item.source_turn_ids,
            )
            db.add(candidate)
            db.flush()
            db.add(CandidateSnapshot(
                analysis_run_id=replay.id,
                transcript_id=transcript.id,
                extraction_batch_item_id=None,
                legacy_candidate_id=candidate.id,
                item_type=item.item_type,
                category=item.category,
                advisor_quote=item.advisor_quote,
                normalized_evidence=item.normalized_evidence,
                timestamp=item.timestamp,
                evidence_strength=item.evidence_strength,
                rationale=item.rationale,
                extraction_confidence=item.extraction_confidence,
                source_turn_ids=item.source_turn_ids,
                ownership=item.ownership,
            ))
        db.commit()

        if source_candidates:
            validate_evidence_for_transcript(transcript.id, db, run_id=replay.id)
            score_signals_for_transcript(transcript.id, db, run_id=replay.id)
            deduplicate_signals_for_transcript(transcript.id, db, run_id=replay.id)
            result = rerank_final_signals_for_transcript(transcript.id, db, run_id=replay.id)
            snapshot_downstream(db, replay.id)
        else:
            result = {"final_driver_count": 0, "final_blocker_count": 0}
        summary = {
            "extracted_candidates": len(source_candidates),
            "final_driver_count": int(result.get("final_driver_count", 0)),
            "final_blocker_count": int(result.get("final_blocker_count", 0)),
        }
        mark_run_completed(db, replay.id, summary)
        db.refresh(replay)
        return replay
    except Exception as error:
        db.rollback()
        mark_run_failed(
            db,
            replay.id,
            failed_stage="replay_validation",
            error_category=type(error).__name__,
            error_message="Validation replay failed",
        )
        raise
