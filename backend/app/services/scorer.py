from typing import cast

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, selectinload

from ..config import settings
from ..llm_schemas import BusinessScoreOutput
from ..logging_config import get_logger
from ..models import CandidateSignal, Transcript
from .eligibility_policy import validation_allows_business_pipeline
from .llm_client import LLMClientError, call_llm_json


logger = get_logger(__name__)


class TranscriptNotFoundError(Exception):
    """Raised when a requested transcript does not exist."""


class NoCandidateSignalsError(Exception):
    """Raised when scoring is requested before candidate extraction."""


class ScoringError(Exception):
    """A safe error for scoring or persistence failures."""


def calculate_final_score(output: BusinessScoreOutput) -> float:
    """Apply the fixed business policy after the model suggests components."""
    return round(
        0.30 * output.advisor_ownership
        + 0.25 * output.decision_impact
        + 0.20 * output.explicitness
        + 0.15 * output.urgency
        + 0.10 * output.evidence_quality,
        2,
    )


def score_signals_for_transcript(transcript_id: int, db: Session, *, run_id: str | None = None) -> dict:
    try:
        transcript = db.get(Transcript, transcript_id)
        if transcript is None:
            logger.warning(
                "action=score_signals transcript_id=%s candidate_count=0 "
                "eligible_count=0 scored_count=0 success=false",
                transcript_id,
            )
            raise TranscriptNotFoundError("Transcript not found")

        candidate_query = select(CandidateSignal).options(selectinload(CandidateSignal.score)).where(CandidateSignal.transcript_id == transcript_id)
        if run_id is not None:
            candidate_query = candidate_query.where(CandidateSignal.analysis_run_id == run_id)
        candidates = list(
            db.scalars(
                candidate_query.order_by(CandidateSignal.id)
            ).all()
        )
    except SQLAlchemyError:
        db.rollback()
        logger.error(
            "action=score_signals transcript_id=%s candidate_count=0 "
            "eligible_count=0 scored_count=0 success=false",
            transcript_id,
        )
        raise ScoringError("Signal scoring could not be completed") from None

    if not candidates:
        logger.warning(
            "action=score_signals transcript_id=%s candidate_count=0 "
            "eligible_count=0 scored_count=0 success=false",
            transcript_id,
        )
        raise NoCandidateSignalsError("Transcript has no candidate signals")

    eligible_candidates = [
        candidate
        for candidate in candidates
        if candidate.score is not None
        and validation_allows_business_pipeline(
            candidate.score.validator_verdict, candidate.score.rejection_reason
        )
    ]
    rejected_count = sum(
        candidate.score is not None
        and not validation_allows_business_pipeline(
            candidate.score.validator_verdict, candidate.score.rejection_reason
        )
        for candidate in candidates
    )

    for candidate in eligible_candidates:
        score = candidate.score
        try:
            output = cast(
                BusinessScoreOutput,
                call_llm_json(
                    prompt_file_name="04_business_scorer.txt",
                    input_payload={
                        "transcript_id": transcript_id,
                        "candidate": {
                            "signal_id": candidate.id,
                            "item_type": candidate.item_type,
                            "category": candidate.category,
                            "advisor_quote": candidate.advisor_quote,
                            "timestamp": candidate.timestamp,
                            "evidence_strength": candidate.evidence_strength,
                            "rationale": candidate.rationale,
                        },
                    },
                    response_model=BusinessScoreOutput,
                    model=settings.business_scorer_model,
                    temperature=0,
                    transcript_id=transcript_id,
                ),
            )
        except LLMClientError:
            db.rollback()
            logger.error(
                "action=score_signals transcript_id=%s candidate_count=%s "
                "eligible_count=%s scored_count=0 success=false",
                transcript_id,
                len(candidates),
                len(eligible_candidates),
            )
            raise ScoringError("Signal scoring service is unavailable") from None

        score.advisor_ownership = output.advisor_ownership
        score.decision_impact = output.decision_impact
        score.explicitness = output.explicitness
        score.urgency = output.urgency
        score.evidence_quality = output.evidence_quality
        score.final_score = calculate_final_score(output)

    transcript.status = "signals_scored"
    try:
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        logger.error(
            "action=score_signals transcript_id=%s candidate_count=%s "
            "eligible_count=%s scored_count=0 success=false",
            transcript_id,
            len(candidates),
            len(eligible_candidates),
        )
        raise ScoringError("Signal scores could not be saved") from None

    scored_count = len(eligible_candidates)
    logger.info(
        "action=score_signals transcript_id=%s candidate_count=%s "
        "eligible_count=%s scored_count=%s success=true",
        transcript_id,
        len(candidates),
        len(eligible_candidates),
        scored_count,
    )
    return {
        "transcript_id": transcript_id,
        "status": "signals_scored",
        "candidate_count": len(candidates),
        "eligible_count": len(eligible_candidates),
        "scored_count": scored_count,
        "rejected_skipped": rejected_count,
    }
