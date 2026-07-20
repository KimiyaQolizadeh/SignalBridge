import math
import re
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from typing import cast

from sqlalchemy import delete, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ..config import settings
from ..llm_schemas import CandidateSignalBatchOutput, CandidateSignalItem
from ..logging_config import get_logger
from ..models import (
    CandidateSignal,
    CandidateSnapshot,
    ExtractionBatch,
    ExtractionBatchItem,
    FinalSignal,
    HumanReview,
    SignalScore,
    Transcript,
    TranscriptTurn,
)
from .llm_client import LLMClientError, call_llm_json
from .prompt_loader import prompt_sha256
from .speaker_classifier import MIN_ROLE_CONFIDENCE


BATCH_SIZE = 60
CONTEXT_ROLES = {"advisor"}
NON_SIGNAL_FACTOR_KINDS = {
    "procedural_diligence",
    "information_request",
    "factual_background",
    "unclear",
}
POLITE_ACKNOWLEDGEMENTS = {
    "absolutely",
    "that sounds good",
    "sounds good",
    "makes sense",
    "okay",
    "ok",
    "got it",
}
PROCEDURAL_PATTERNS = (
    r"\b(?:review|look over) (?:the |those )?(?:materials?|information)\b",
    r"\btake the weekend\b",
    r"\bthink about it\b",
    r"\bsend me (?:the |more )?(?:materials?|information)\b",
    r"\b(?:schedule|set up) (?:a|another) call\b",
    r"\b(?:discuss|speak about|talk about) it with (?:my|the) "
    r"(?:partner|team)\b",
)
DEPENDENCY_LANGUAGE = re.compile(
    r"\b(?:cannot|can't|could not|couldn't|won't|would not)\s+"
    r"(?:move|proceed)|\bunless\b|\bbefore (?:i|we) (?:can |could )?"
    r"(?:move|proceed)|\b(?:has|have|need|needs) to approve\b|"
    r"\brequires? approval\b",
    re.IGNORECASE,
)
QUESTION_MATERIALITY_LANGUAGE = re.compile(
    r"\b(?:cannot|can't|could not|couldn't|won't|would not)\s+"
    r"(?:move|proceed)|\bunless\b|\bbefore (?:i|we) (?:can |could )?"
    r"(?:move|proceed)|\b(?:need|required?|requirement|must)\b",
    re.IGNORECASE,
)
TRANSITION_LINK_LANGUAGE = re.compile(
    r"\b(?:that's why|that is why|looking for|considering|consider a change|"
    r"want(?:ing)? (?:a|to) (?:different|change|leave|move)|"
    r"need(?:ing)? more support|move forward|moving forward|"
    r"ready to (?:move|proceed)|leave (?:my|the|this)|change firms?)\b",
    re.IGNORECASE,
)
EXPLICIT_COMMITMENT_LANGUAGE = re.compile(
    r"\b(?:we(?:'re| are) moving forward|i want to proceed|"
    r"this is the direction i want to take|i(?:'m| am) ready to "
    r"(?:make the move|move|proceed))\b",
    re.IGNORECASE,
)
logger = get_logger(__name__)


class TranscriptNotFoundError(Exception):
    """Raised when a requested transcript does not exist."""


class NoTranscriptTurnsError(Exception):
    """Raised when candidate extraction is requested before parsing."""


class SpeakersNotClassifiedError(Exception):
    """Raised when candidate extraction is requested before role classification."""


class CandidateExtractionError(Exception):
    """A safe error for extraction or persistence failures."""


def _build_extraction_turns(turns: list[TranscriptTurn]) -> list[TranscriptTurn]:
    selected_indexes: set[int] = set()
    for index, turn in enumerate(turns):
        if turn.inferred_role not in CONTEXT_ROLES:
            continue
        selected_indexes.add(index)
        if index > 0:
            selected_indexes.add(index - 1)
        if index + 1 < len(turns):
            selected_indexes.add(index + 1)

    return [turns[index] for index in sorted(selected_indexes)]


def _is_reliably_advisor_owned(turn: TranscriptTurn) -> bool:
    return (
        turn.inferred_role == "advisor"
        and turn.role_confidence is not None
        and turn.role_confidence >= MIN_ROLE_CONFIDENCE
    )


def _is_candidate_evidence_turn(turn: TranscriptTurn) -> bool:
    return (
        turn.inferred_role == "advisor"
        and turn.role_confidence is not None
        and turn.role_confidence >= MIN_ROLE_CONFIDENCE
    )


def _turn_payload(turn: TranscriptTurn) -> dict:
    return {
        "turn_id": turn.id,
        "turn_index": turn.turn_index,
        "timestamp": turn.timestamp,
        "raw_speaker_label": turn.raw_speaker_label,
        "inferred_role": turn.inferred_role,
        "role_confidence": turn.role_confidence,
        "text": turn.text,
    }


def _bounded_advisor_context(
    item: CandidateSignalItem,
    turns: list[TranscriptTurn],
    turns_by_id: dict[int, TranscriptTurn],
) -> str:
    index_by_id = {turn.id: index for index, turn in enumerate(turns)}
    selected_indexes: set[int] = set()
    for turn_id in item.source_turn_ids:
        source = turns_by_id.get(turn_id)
        if source is None or not _is_reliably_advisor_owned(source):
            continue
        source_index = index_by_id[turn_id]
        selected_indexes.add(source_index)
        for direction in (-1, 1):
            for distance in (1, 2):
                context_index = source_index + direction * distance
                if not 0 <= context_index < len(turns):
                    break
                context_turn = turns[context_index]
                if not _is_reliably_advisor_owned(context_turn):
                    break
                selected_indexes.add(context_index)
    return " ".join(turns[index].text for index in sorted(selected_indexes))


def _is_purely_procedural(quote: str) -> bool:
    return any(
        re.search(pattern, quote, re.IGNORECASE) for pattern in PROCEDURAL_PATTERNS
    )


def _has_supported_semantics(
    item: CandidateSignalItem, advisor_context: str
) -> bool:
    """Keep business semantics for downstream scoring; reject only no-signal."""
    quote = item.advisor_quote.strip()
    if item.item_type == "no_signal":
        return False
    if item.factor_kind == "explicit_commitment":
        return (
            item.item_type == "driver"
            and item.decision_direction == "supports_move"
            and EXPLICIT_COMMITMENT_LANGUAGE.search(quote) is not None
        )
    return True


def extract_candidate_signals_for_transcript(
    transcript_id: int, db: Session, *, run_id: str | None = None
) -> dict:
    """Extract broadly; evidence validation separately enforces final grounding."""
    try:
        transcript = db.get(Transcript, transcript_id)
        if transcript is None:
            raise TranscriptNotFoundError("Transcript not found")

        turns = list(
            db.scalars(
                select(TranscriptTurn)
                .where(TranscriptTurn.transcript_id == transcript_id)
                .order_by(TranscriptTurn.turn_index, TranscriptTurn.id)
            ).all()
        )
    except SQLAlchemyError:
        db.rollback()
        logger.error(
            "action=extract_candidates transcript_id=%s batch_number=0 "
            "candidate_count=0 success=false",
            transcript_id,
        )
        raise CandidateExtractionError(
            "Candidate extraction could not be completed"
        ) from None

    if not turns:
        raise NoTranscriptTurnsError("Transcript has no parsed turns")
    if not any(turn.inferred_role is not None for turn in turns):
        raise SpeakersNotClassifiedError("Transcript speakers are not classified")

    extraction_turns = _build_extraction_turns(turns)
    role_counts = Counter(turn.inferred_role or "null" for turn in turns)
    eligible_turn_count = sum(
        _is_reliably_advisor_owned(turn) for turn in turns
    )
    extraction_batch_count = math.ceil(len(extraction_turns) / BATCH_SIZE)
    logger.info(
        "action=extract_candidates_input transcript_id=%s total_turn_count=%s "
        "advisor_count=%s mixed_count=%s optimize_rep_count=%s "
        "unknown_count=%s null_count=%s eligible_advisor_turn_count=%s "
        "extraction_turn_count=%s extraction_batch_count=%s",
        transcript_id,
        len(turns),
        role_counts["advisor"],
        role_counts["mixed"],
        role_counts["optimize_rep"],
        role_counts["unknown"],
        role_counts["null"],
        eligible_turn_count,
        len(extraction_turns),
        extraction_batch_count,
    )
    extracted_items: list[tuple[CandidateSignalItem, ExtractionBatchItem | None]] = []

    for batch_offset in range(0, len(extraction_turns), BATCH_SIZE):
        batch = extraction_turns[batch_offset : batch_offset + BATCH_SIZE]
        batch_number = (batch_offset // BATCH_SIZE) + 1
        logger.info(
            "action=extract_candidates_batch_start transcript_id=%s "
            "batch_number=%s candidate_count=0 success=pending",
            transcript_id,
            batch_number,
        )
        batch_payload = [_turn_payload(turn) for turn in batch]
        batch_record = None
        if run_id is not None:
            batch_record = ExtractionBatch(
                analysis_run_id=run_id,
                batch_index=batch_number,
                input_turn_ids=[turn.id for turn in batch],
                input_hash=sha256(str(batch_payload).encode("utf-8")).hexdigest(),
                input_snapshot=batch_payload,
                prompt_hash=prompt_sha256("02_candidate_signal_extractor.txt"),
                model_identifier=settings.candidate_extractor_model,
                model_configuration={"temperature": 0},
                started_at=datetime.now(timezone.utc),
                status="running",
            )
            db.add(batch_record)
            db.commit()
        response_metadata: dict = {}
        try:
            output = cast(
                CandidateSignalBatchOutput,
                call_llm_json(
                    prompt_file_name="02_candidate_signal_extractor.txt",
                    input_payload={
                        "transcript_id": transcript_id,
                        "turns": batch_payload,
                    },
                    response_model=CandidateSignalBatchOutput,
                    model=settings.candidate_extractor_model,
                    temperature=0,
                    transcript_id=transcript_id,
                    response_observer=response_metadata.update,
                ),
            )
        except LLMClientError:
            if batch_record is not None:
                batch_record.status = "failed"
                batch_record.completed_at = datetime.now(timezone.utc)
                batch_record.error_category = "LLMClientError"
                batch_record.error_message = "Candidate extraction service is unavailable"
                db.commit()
            else:
                db.rollback()
            logger.error(
                "action=extract_candidates_batch transcript_id=%s batch_number=%s "
                "candidate_count=0 success=false",
                transcript_id,
                batch_number,
            )
            raise CandidateExtractionError(
                "Candidate extraction service is unavailable"
            ) from None

        if batch_record is not None:
            batch_record.status = "completed"
            batch_record.completed_at = datetime.now(timezone.utc)
            batch_record.raw_response = response_metadata.get("raw_response")
            batch_record.token_usage = response_metadata.get("token_usage")
            batch_record.retry_count = response_metadata.get("retry_count", 0) or 0
            batch_record.finish_reason = str(response_metadata.get("finish_reason") or "") or None
            batch_record.raw_item_count = len(output.items)
            batch_record.parsed_item_count = len(output.items)
        for item_index, item in enumerate(output.items):
            item_record = None
            if batch_record is not None:
                item_record = ExtractionBatchItem(
                    extraction_batch_id=batch_record.id,
                    item_index=item_index,
                    raw_structured_item=item.model_dump(mode="json"),
                    parsed_successfully=True,
                    classification=item.item_type,
                    accepted_after_filter=False,
                    source_turn_ids=item.source_turn_ids,
                    evidence_text=item.advisor_quote,
                    normalized_evidence=" ".join(item.advisor_quote.split()).lower(),
                    item_type=item.item_type,
                    category=item.category,
                    rationale=item.rationale,
                    confidence=item.extraction_confidence,
                )
                db.add(item_record)
            extracted_items.append((item, item_record))
        if batch_record is not None:
            db.commit()
        logger.info(
            "action=extract_candidates_batch transcript_id=%s batch_number=%s "
            "candidate_count=%s success=true",
            transcript_id,
            batch_number,
            len(output.items),
        )

    valid_turn_ids = {turn.id for turn in turns}
    turns_by_id = {turn.id: turn for turn in turns}
    evidence_turns = [turn for turn in turns if _is_candidate_evidence_turn(turn)]
    seen_keys: dict[tuple[str, str, str], ExtractionBatchItem | None] = {}
    candidates_to_save: list[CandidateSignal] = []

    accepted_records: list[ExtractionBatchItem] = []
    for item, item_record in extracted_items:
        def reject(reason: str) -> None:
            if item_record is not None:
                item_record.filter_reason = reason

        if item.item_type == "no_signal":
            reject("no_signal")
            continue
        advisor_context = _bounded_advisor_context(item, turns, turns_by_id)
        if not _has_supported_semantics(item, advisor_context):
            reject("unsupported_semantics")
            continue
        category = item.category
        rationale = item.rationale
        if category is None or rationale is None:
            reject("invalid_schema")
            continue
        item_type = item.item_type
        source_turn_ids = list(
            dict.fromkeys(
                turn_id
                for turn_id in item.source_turn_ids
                if turn_id in valid_turn_ids
                and _is_candidate_evidence_turn(turns_by_id[turn_id])
                and item.advisor_quote in turns_by_id[turn_id].text
            )
        )
        if len(source_turn_ids) > 1:
            timestamp_matches = [
                turn_id
                for turn_id in source_turn_ids
                if turns_by_id[turn_id].timestamp == item.timestamp
            ]
            if len(timestamp_matches) != 1:
                reject("ambiguous_evidence_match")
                continue
            source_turn_ids = timestamp_matches
        if not source_turn_ids:
            matching_turn_ids = [
                turn.id for turn in evidence_turns if item.advisor_quote in turn.text
            ]
            if len(matching_turn_ids) != 1:
                any_turn_match = any(item.advisor_quote in turn.text for turn in turns)
                reject("invalid_ownership" if any_turn_match and not matching_turn_ids else "invalid_source_turn" if not matching_turn_ids else "ambiguous_evidence_match")
                continue
            source_turn_ids = matching_turn_ids

        authoritative_turn = turns_by_id[source_turn_ids[0]]

        duplicate_key = (
            item_type,
            category.lower().strip(),
            item.advisor_quote.strip(),
        )
        if duplicate_key in seen_keys:
            previous_record = seen_keys[duplicate_key]
            same_batch = item_record is not None and previous_record is not None and item_record.extraction_batch_id == previous_record.extraction_batch_id
            reject("duplicate_within_batch" if same_batch else "duplicate_across_batches")
            continue
        seen_keys[duplicate_key] = item_record

        candidates_to_save.append(
            CandidateSignal(
                transcript_id=transcript_id,
                analysis_run_id=run_id,
                item_type=item_type,
                category=category,
                advisor_quote=item.advisor_quote,
                timestamp=authoritative_turn.timestamp,
                evidence_strength=item.evidence_strength,
                rationale=rationale,
                extraction_confidence=item.extraction_confidence,
                source_turn_ids=source_turn_ids,
            )
        )
        if item_record is not None:
            item_record.accepted_after_filter = True
            item_record.filter_reason = None
            accepted_records.append(item_record)

    try:
        if run_id is None:
            existing_candidate_ids = select(CandidateSignal.id).where(CandidateSignal.transcript_id == transcript_id)
            db.execute(delete(FinalSignal).where(FinalSignal.transcript_id == transcript_id))
            db.execute(delete(HumanReview).where(HumanReview.signal_id.in_(existing_candidate_ids)))
            db.execute(delete(SignalScore).where(SignalScore.signal_id.in_(existing_candidate_ids)))
            db.execute(delete(CandidateSignal).where(CandidateSignal.transcript_id == transcript_id))
        db.add_all(candidates_to_save)
        db.flush()
        if run_id is not None:
            for candidate, item_record in zip(candidates_to_save, accepted_records, strict=True):
                db.add(CandidateSnapshot(
                    analysis_run_id=run_id,
                    transcript_id=transcript_id,
                    extraction_batch_item_id=item_record.id,
                    legacy_candidate_id=candidate.id,
                    item_type=candidate.item_type,
                    category=candidate.category,
                    advisor_quote=candidate.advisor_quote,
                    normalized_evidence=" ".join(candidate.advisor_quote.split()).lower(),
                    timestamp=candidate.timestamp,
                    evidence_strength=candidate.evidence_strength,
                    rationale=candidate.rationale,
                    extraction_confidence=candidate.extraction_confidence,
                    source_turn_ids=candidate.source_turn_ids or [],
                    ownership="advisor",
                ))
            for batch_record in db.scalars(select(ExtractionBatch).where(ExtractionBatch.analysis_run_id == run_id)):
                batch_record.post_filter_item_count = sum(record.accepted_after_filter for record in db.scalars(select(ExtractionBatchItem).where(ExtractionBatchItem.extraction_batch_id == batch_record.id)))
        transcript.status = "candidates_extracted"
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        logger.error(
            "action=extract_candidates_save transcript_id=%s batch_number=0 "
            "candidate_count=0 success=false",
            transcript_id,
        )
        raise CandidateExtractionError(
            "Candidate signals could not be saved"
        ) from None

    driver_count = sum(
        candidate.item_type == "driver" for candidate in candidates_to_save
    )
    blocker_count = sum(
        candidate.item_type == "blocker" for candidate in candidates_to_save
    )
    logger.info(
        "action=extract_candidates transcript_id=%s batch_number=0 "
        "candidate_count=%s success=true",
        transcript_id,
        len(candidates_to_save),
    )
    return {
        "transcript_id": transcript_id,
        "status": "candidates_extracted",
        "candidate_count": len(candidates_to_save),
        "driver_candidates": driver_count,
        "blocker_candidates": blocker_count,
    }
