from typing import cast

from sqlalchemy import delete, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, selectinload

from ..config import settings
from ..llm_schemas import FinalRerankOutput, FinalSelectedSignal
from ..logging_config import get_logger
from ..models import CandidateSignal, FinalSignal, Transcript
from .llm_client import LLMClientError, call_llm_json


MAX_PER_TYPE = 3
RERANK_INPUT_LIMIT = 8
logger = get_logger(__name__)


class TranscriptNotFoundError(Exception):
    """Raised when a requested transcript does not exist."""


class NoCandidateSignalsError(Exception):
    """Raised when final reranking is requested before extraction."""


class RerankingError(Exception):
    """A safe error for final-signal persistence failures."""


def _is_grounded(candidate: CandidateSignal) -> bool:
    score = candidate.score
    if (
        not candidate.is_canonical
        or score is None
        or score.validator_verdict not in {"pass", "needs_review"}
        or score.final_score is None
    ):
        return False
    return True


def _is_eligible(candidate: CandidateSignal) -> bool:
    """Apply preferred thresholds: 3.5 explicit and 4.0 implied evidence."""
    if not _is_grounded(candidate):
        return False
    score = candidate.score
    if candidate.evidence_strength == "explicit":
        return score.final_score >= 3.5
    if candidate.evidence_strength == "implied":
        return score.final_score >= 4.0
    return False


def _sort_key(candidate: CandidateSignal) -> tuple:
    score = candidate.score
    return (
        -score.final_score,
        candidate.evidence_strength != "explicit",
        candidate.id,
    )


def _candidate_payload(candidate: CandidateSignal) -> dict:
    score = candidate.score
    return {
        "signal_id": candidate.id,
        "item_type": candidate.item_type,
        "category": candidate.category,
        "advisor_quote": candidate.advisor_quote,
        "timestamp": candidate.timestamp,
        "evidence_strength": candidate.evidence_strength,
        "rationale": candidate.rationale,
        "validator_verdict": score.validator_verdict,
        "support_score": score.support_score,
        "advisor_side_score": score.advisor_side_score,
        "false_positive_risk": score.false_positive_risk,
        "advisor_ownership": score.advisor_ownership,
        "decision_impact": score.decision_impact,
        "explicitness": score.explicitness,
        "urgency": score.urgency,
        "evidence_quality": score.evidence_quality,
        "final_score": score.final_score,
        "duplicate_group_id": candidate.duplicate_group_id,
    }


def _deterministic_selection(
    drivers: list[CandidateSignal], blockers: list[CandidateSignal]
) -> list[FinalSelectedSignal]:
    selections: list[FinalSelectedSignal] = []
    for item_type, candidates in (("driver", drivers), ("blocker", blockers)):
        selected_categories: set[str] = set()
        rank = 1
        for candidate in candidates:
            category_key = candidate.category.strip().lower()
            if category_key in selected_categories:
                continue
            selected_categories.add(category_key)
            selections.append(
                FinalSelectedSignal(
                    signal_id=candidate.id,
                    item_type=item_type,
                    rank=rank,
                )
            )
            rank += 1
            if rank > MAX_PER_TYPE:
                break
    return selections


def _valid_llm_selection(
    selections: list[FinalSelectedSignal], candidates: list[CandidateSignal]
) -> bool:
    candidates_by_id = {candidate.id: candidate for candidate in candidates}
    seen_ids: set[int] = set()
    categories_by_type: dict[str, set[str]] = {"driver": set(), "blocker": set()}
    ranks_by_type: dict[str, list[int]] = {"driver": [], "blocker": []}

    for selection in selections:
        candidate = candidates_by_id.get(selection.signal_id)
        if (
            candidate is None
            or selection.signal_id in seen_ids
            or selection.item_type != candidate.item_type
            or selection.rank not in {1, 2, 3}
        ):
            return False
        seen_ids.add(selection.signal_id)

        category_key = candidate.category.strip().lower()
        if category_key in categories_by_type[selection.item_type]:
            return False
        categories_by_type[selection.item_type].add(category_key)
        ranks_by_type[selection.item_type].append(selection.rank)

    for ranks in ranks_by_type.values():
        if len(ranks) > MAX_PER_TYPE or sorted(ranks) != list(range(1, len(ranks) + 1)):
            return False
    return True


def rerank_final_signals_for_transcript(transcript_id: int, db: Session, *, run_id: str | None = None) -> dict:
    try:
        transcript = db.get(Transcript, transcript_id)
        if transcript is None:
            logger.warning(
                "action=rerank_final transcript_id=%s eligible_count=0 "
                "final_driver_count=0 final_blocker_count=0 "
                "used_fallback=false success=false",
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
            "action=rerank_final transcript_id=%s eligible_count=0 "
            "final_driver_count=0 final_blocker_count=0 "
            "used_fallback=false success=false",
            transcript_id,
        )
        raise RerankingError("Final reranking could not be completed") from None

    if not candidates:
        logger.warning(
            "action=rerank_final transcript_id=%s eligible_count=0 "
            "final_driver_count=0 final_blocker_count=0 "
            "used_fallback=false success=false",
            transcript_id,
        )
        raise NoCandidateSignalsError("Transcript has no candidate signals")

    eligible = [candidate for candidate in candidates if _is_eligible(candidate)]
    grounded = [candidate for candidate in candidates if _is_grounded(candidate)]
    drivers = sorted(
        (candidate for candidate in eligible if candidate.item_type == "driver"),
        key=_sort_key,
    )
    blockers = sorted(
        (candidate for candidate in eligible if candidate.item_type == "blocker"),
        key=_sort_key,
    )
    rerank_candidates = drivers[:RERANK_INPUT_LIMIT] + blockers[:RERANK_INPUT_LIMIT]
    used_fallback = False

    if not eligible:
        selections: list[FinalSelectedSignal] = []
    else:
        try:
            output = cast(
                FinalRerankOutput,
                call_llm_json(
                    prompt_file_name="05_final_reranker.txt",
                    input_payload={
                        "transcript_id": transcript_id,
                        "eligible_candidates": [
                            _candidate_payload(candidate)
                            for candidate in rerank_candidates
                        ],
                        "selection_rules": {
                            "max_drivers": MAX_PER_TYPE,
                            "max_blockers": MAX_PER_TYPE,
                            "allow_empty": True,
                        },
                    },
                    response_model=FinalRerankOutput,
                    model=settings.final_reranker_model,
                    temperature=0,
                    transcript_id=transcript_id,
                ),
            )
            selections = output.selected_signals
            if not _valid_llm_selection(selections, rerank_candidates):
                selections = _deterministic_selection(drivers, blockers)
                used_fallback = True
        except LLMClientError:
            selections = _deterministic_selection(drivers, blockers)
            used_fallback = True

    # A direction with no threshold-qualified result may expose one grounded
    # needs-review candidate; rejected candidates never enter this fallback.
    selection_reasons = {
        selection.signal_id: "above_threshold" for selection in selections
    }
    preferred_types = {candidate.item_type for candidate in eligible}
    for item_type in ("driver", "blocker"):
        if item_type in preferred_types:
            continue
        fallback_candidates = sorted(
            (
                candidate
                for candidate in grounded
                if candidate.item_type == item_type
                and candidate.score.validator_verdict == "needs_review"
            ),
            key=_sort_key,
        )
        if not fallback_candidates:
            continue
        fallback = fallback_candidates[0]
        selections.append(
            FinalSelectedSignal(
                signal_id=fallback.id,
                item_type=item_type,
                rank=1,
            )
        )
        selection_reasons[fallback.id] = "best_grounded_fallback"

    try:
        final_delete = delete(FinalSignal).where(FinalSignal.transcript_id == transcript_id)
        if run_id is not None:
            final_delete = final_delete.where(FinalSignal.analysis_run_id == run_id)
        db.execute(final_delete)
        db.add_all(
            FinalSignal(
                transcript_id=transcript_id,
                analysis_run_id=run_id,
                signal_id=selection.signal_id,
                item_type=selection.item_type,
                rank=selection.rank,
            )
            for selection in selections
        )
        transcript.status = "finalized"
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        logger.error(
            "action=rerank_final transcript_id=%s eligible_count=%s "
            "final_driver_count=0 final_blocker_count=0 used_fallback=%s "
            "success=false",
            transcript_id,
            len(eligible),
            str(used_fallback).lower(),
        )
        raise RerankingError("Final signals could not be saved") from None

    driver_count = sum(selection.item_type == "driver" for selection in selections)
    blocker_count = sum(selection.item_type == "blocker" for selection in selections)
    logger.info(
        "action=rerank_final transcript_id=%s eligible_count=%s "
        "final_driver_count=%s final_blocker_count=%s used_fallback=%s "
        "success=true",
        transcript_id,
        len(eligible),
        driver_count,
        blocker_count,
        str(used_fallback).lower(),
    )
    return {
        "transcript_id": transcript_id,
        "status": "finalized",
        "eligible_count": len(eligible),
        "final_driver_count": driver_count,
        "final_blocker_count": blocker_count,
        "used_fallback": used_fallback,
        "selection_reasons": selection_reasons,
    }
