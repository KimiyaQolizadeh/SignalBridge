from typing import cast

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ..config import settings
from ..llm_schemas import SpeakerRoleBatchOutput
from ..logging_config import get_logger
from ..models import Transcript, TranscriptTurn
from .llm_client import LLMClientError, call_llm_json
from .run_persistence import snapshot_speaker_classifications


BATCH_SIZE = 40
MIN_ROLE_CONFIDENCE = 0.65
logger = get_logger(__name__)


class TranscriptNotFoundError(Exception):
    """Raised when a requested transcript does not exist."""


class NoTranscriptTurnsError(Exception):
    """Raised when speaker classification is requested before parsing."""


class SpeakerClassificationError(Exception):
    """A safe error for classification or persistence failures."""


def classify_speakers_for_transcript(
    transcript_id: int, db: Session, *, run_id: str | None = None
) -> dict:
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
            "action=classify_speakers transcript_id=%s batch_number=0 "
            "turn_count=0 success=false",
            transcript_id,
        )
        raise SpeakerClassificationError(
            "Speaker classification could not be completed"
        ) from None

    if not turns:
        raise NoTranscriptTurnsError("Transcript has no parsed turns")

    # Establish conservative defaults before applying any model response. Missing
    # items therefore remain unknown without retaining stale classifications.
    for turn in turns:
        turn.inferred_role = "unknown"
        turn.role_confidence = 0.0

    rationales: dict[int, str] = {}
    raw_responses: dict[int, dict] = {}

    for batch_offset in range(0, len(turns), BATCH_SIZE):
        batch = turns[batch_offset : batch_offset + BATCH_SIZE]
        batch_number = (batch_offset // BATCH_SIZE) + 1
        turn_count = len(batch)
        logger.info(
            "action=classify_speakers_batch_start transcript_id=%s "
            "batch_number=%s turn_count=%s success=pending",
            transcript_id,
            batch_number,
            turn_count,
        )

        input_payload = {
            "transcript_id": transcript_id,
            "turns": [
                {
                    "turn_id": turn.id,
                    "turn_index": turn.turn_index,
                    "timestamp": turn.timestamp,
                    "raw_speaker_label": turn.raw_speaker_label,
                    "text": turn.text,
                }
                for turn in batch
            ],
        }

        try:
            output = cast(
                SpeakerRoleBatchOutput,
                call_llm_json(
                    prompt_file_name="01_speaker_role_classifier.txt",
                    input_payload=input_payload,
                    response_model=SpeakerRoleBatchOutput,
                    model=settings.speaker_classifier_model,
                    temperature=0,
                    transcript_id=transcript_id,
                ),
            )
        except LLMClientError:
            db.rollback()
            logger.error(
                "action=classify_speakers_batch transcript_id=%s "
                "batch_number=%s turn_count=%s success=false",
                transcript_id,
                batch_number,
                turn_count,
            )
            raise SpeakerClassificationError(
                "Speaker classification service is unavailable"
            ) from None

        turns_by_id = {turn.id: turn for turn in batch}
        for item in output.items:
            turn = turns_by_id.get(item.turn_id)
            if turn is None:
                continue

            turn.role_confidence = item.confidence
            turn.inferred_role = (
                item.inferred_role
                if item.confidence >= MIN_ROLE_CONFIDENCE
                else "unknown"
            )
            rationales[turn.id] = item.reason
            raw_responses[turn.id] = item.model_dump(mode="json")

        logger.info(
            "action=classify_speakers_batch transcript_id=%s batch_number=%s "
            "turn_count=%s success=true",
            transcript_id,
            batch_number,
            turn_count,
        )

    transcript.status = "speakers_classified"
    try:
        db.commit()
        if run_id is not None:
            snapshot_speaker_classifications(
                db, run_id, turns, rationales, raw_responses
            )
    except SQLAlchemyError:
        db.rollback()
        logger.error(
            "action=classify_speakers_commit transcript_id=%s batch_number=0 "
            "turn_count=%s success=false",
            transcript_id,
            len(turns),
        )
        raise SpeakerClassificationError(
            "Speaker classifications could not be saved"
        ) from None

    counts = {role: 0 for role in ("advisor", "optimize_rep", "unknown", "mixed")}
    for turn in turns:
        counts[turn.inferred_role or "unknown"] += 1

    return {
        "transcript_id": transcript_id,
        "status": "speakers_classified",
        "turn_count": len(turns),
        "advisor_turns": counts["advisor"],
        "optimize_rep_turns": counts["optimize_rep"],
        "unknown_turns": counts["unknown"],
        "mixed_turns": counts["mixed"],
    }
