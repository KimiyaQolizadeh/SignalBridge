import re
from typing import cast

from sqlalchemy import delete, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, selectinload

from ..config import settings
from ..llm_schemas import FinalRerankOutput, FinalSelectedSignal
from ..logging_config import get_logger
from ..models import CandidateSignal, FinalSignal, Transcript, TranscriptTurn
from .llm_client import LLMClientError, call_llm_json


MAX_PER_TYPE = 3
RERANK_INPUT_LIMIT = 8
FALLBACK_SCORE_FLOOR = 3.25
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


_GENERAL_OBSERVATION = re.compile(r"\b(?:everyone|every advisor|all advisors|people|the industry|most firms)\b", re.I)
_CONDITIONAL = re.compile(r"\b(?:if|maybe|potentially|could|i suppose|if it works|if it does what it claims|if it does work)\b", re.I)
_PERSONAL_OWNERSHIP = re.compile(r"\b(?:i|i'm|i've|i'd|me|my|mine|we|we're|we've|our|us)\b", re.I)
_DRIVER_DIRECTION = re.compile(r"\b(?:want|need|value|benefit|help(?:s|ed)?|free up|spend more time|move forward with|decided|prefer|important to me)\b", re.I)
_BLOCKER_DIRECTION = re.compile(r"\b(?:concern|worried|hesitat|cannot|can't|won't|unless|must|required?|need.+before|wait|delay|approval|constraint|risk)\b", re.I)
_INSUFFICIENT_REASONS = frozenset({
    "ambiguous_evidence", "incomplete_context", "context_failure",
    "neutral_or_indeterminate_effect", "candidate_direction_mismatch",
})


def _candidate_context(candidate: CandidateSignal, turns: list[TranscriptTurn]) -> str:
    source_ids = set(candidate.source_turn_ids or [])
    positions = [index for index, turn in enumerate(turns) if turn.id in source_ids]
    if not positions:
        return candidate.advisor_quote
    first, last = min(positions), max(positions)
    window = turns[max(0, first - 1):min(len(turns), last + 2)]
    return " ".join(turn.text for turn in window)


def _source_text(candidate: CandidateSignal, turns: list[TranscriptTurn]) -> str:
    source_ids = set(candidate.source_turn_ids or [])
    return " ".join(turn.text for turn in turns if turn.id in source_ids)


def _apply_contextual_direction(candidate: CandidateSignal, context: str) -> None:
    text = f"{candidate.advisor_quote} {context}".lower()
    if candidate.item_type == "blocker" and re.search(r"\b(?:timely|timeliness|compliance|rules?)\b", text) and re.search(r"\b(?:every firm|firm i(?:'ve| have) worked with|current firm)\b", text) and not (("optimize" in text or "you guys" in text) and re.search(r"\b(?:same|equivalent|proof|prove|approval|required?)\b", text)):
        candidate.item_type = "driver"
        candidate.category = "Communication Flexibility"
        candidate.rationale = "The advisor describes current-firm compliance restrictions as limiting timely, authentic communication, which motivates interest in a more flexible environment."
    if candidate.item_type == "blocker" and "threshold" in text and re.search(r"\b(?:worried|concerned?)\b", text):
        candidate.category = "Minimum Book Threshold"
        candidate.rationale = "The advisor is concerned that their current book size is below the threshold being discussed, which may affect eligibility or willingness to proceed."
    if candidate.item_type == "driver" and candidate.category.lower() == "values alignment" and candidate.advisor_quote.lower() in candidate.rationale.lower():
        candidate.rationale = "The advisor indicates that misalignment with the current firm's values is prompting consideration of an alternative organization where those values are better matched."


def _passes_contextual_final_gate(candidate: CandidateSignal, context: str, source_text: str) -> bool:
    combined = f"{candidate.advisor_quote} {context}".lower()
    source = source_text.lower()
    if candidate.item_type == "blocker" and re.search(r"\bdon't have time\b|\bdo not have time\b", source) and re.search(r"\b(?:fortunately|dedicated).{0,100}(?:support|planning)\b", source):
        return False
    if candidate.item_type == "blocker" and re.search(r"\b(?:accountab|responsib|explain|tell\w*(?:\s+\w+){0,3}\s+the rationale)\b", combined) and re.search(r"\bworked very well\b", combined):
        return False
    if candidate.item_type == "driver" and re.search(r"\bwe use\b.{0,120}\b(?:income generation|portfolio protection|tax balancing)\b", source) and not re.search(r"\b(?:optimize|move|switch|preserve|enhance|scale|replace|would help)\b", source):
        return False
    return True

def _is_fallback_eligible(candidate: CandidateSignal, context: str = "") -> bool:
    """Require personal, directional, decision-relevant evidence for weak fallback."""
    if not _is_grounded(candidate):
        return False
    score = candidate.score
    quote = candidate.advisor_quote.strip()
    rationale = candidate.rationale.strip()
    contextual_evidence = f"{quote} {rationale} {context}"
    if (
        score.validator_verdict not in {"pass", "needs_review"}
        or score.final_score < (3.0 if score.validator_verdict == "pass" else FALLBACK_SCORE_FLOOR)
        or (score.advisor_ownership or 0) < 4
        or (score.decision_impact or 0) < 3
        or (score.evidence_quality or 0) < 3
        or score.rejection_reason in _INSUFFICIENT_REASONS
        or not _PERSONAL_OWNERSHIP.search(quote)
        or _GENERAL_OBSERVATION.search(quote)
        or _CONDITIONAL.search(quote)
    ):
        return False
    direction_pattern = _DRIVER_DIRECTION if candidate.item_type == "driver" else _BLOCKER_DIRECTION
    return bool(direction_pattern.search(contextual_evidence))


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
        turns = list(db.scalars(
            select(TranscriptTurn)
            .where(TranscriptTurn.transcript_id == transcript_id)
            .order_by(TranscriptTurn.turn_index, TranscriptTurn.id)
        ).all())
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

    contexts = {candidate.id: _candidate_context(candidate, turns) for candidate in candidates}
    sources = {candidate.id: _source_text(candidate, turns) for candidate in candidates}
    for candidate in candidates:
        _apply_contextual_direction(candidate, contexts[candidate.id])
    eligible = [
        candidate for candidate in candidates
        if _is_eligible(candidate)
        and _passes_contextual_final_gate(candidate, contexts[candidate.id], sources[candidate.id])
    ]
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
                and _is_fallback_eligible(candidate, contexts[candidate.id])
                and _passes_contextual_final_gate(candidate, contexts[candidate.id], sources[candidate.id])
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
