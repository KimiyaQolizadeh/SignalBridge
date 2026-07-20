from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile, status
from sqlalchemy import delete, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, selectinload

from ..config import settings
from ..database import get_db
from ..logging_config import get_logger
from ..models import CandidateSignal, FinalSignal, Transcript, TranscriptTurn
from ..schemas import (
    CandidateExtractionResponse,
    CandidateSignalResponse,
    CandidateSignalWithScoreResponse,
    DeleteResponse,
    EvidenceValidationResponse,
    FinalRerankingResponse,
    FinalSignalResponse,
    ParseTranscriptResponse,
    PipelineRunResponse,
    ProcessingStatusResponse,
    PipelineDiagnosticsResponse,
    SpeakerClassificationResponse,
    SignalScoringResponse,
    SignalDeduplicationResponse,
    TranscriptDetail,
    TranscriptListItem,
    TranscriptTurnResponse,
    TranscriptUploadResponse,
)
from ..services import progress
from ..services.run_persistence import latest_run_for_transcript
from ..services.pipeline_runner import (
    PipelineRunError,
    TranscriptNotFoundError as PipelineTranscriptNotFoundError,
    run_full_pipeline_for_transcript,
)
from ..services.reranker import (
    NoCandidateSignalsError as NoRerankingCandidateSignalsError,
    RerankingError,
    TranscriptNotFoundError as RerankingTranscriptNotFoundError,
    rerank_final_signals_for_transcript,
)
from ..services.signal_deduplicator import (
    NoCandidateSignalsError as NoDeduplicationCandidateSignalsError,
    SignalDeduplicationError,
    TranscriptNotFoundError as DeduplicationTranscriptNotFoundError,
    deduplicate_signals_for_transcript,
)
from ..services.scorer import (
    NoCandidateSignalsError as NoScoringCandidateSignalsError,
    ScoringError,
    TranscriptNotFoundError as ScoringTranscriptNotFoundError,
    score_signals_for_transcript,
)
from ..services.evidence_validator import (
    EvidenceValidationError,
    NoCandidateSignalsError,
    TranscriptNotFoundError as EvidenceTranscriptNotFoundError,
    validate_evidence_for_transcript,
)
from ..services.evidence_context import context_payload, context_turns
from ..services.exporter import (
    ExportError,
    TranscriptNotFoundError as ExportTranscriptNotFoundError,
    export_all_transcripts_csv,
    export_transcript_csv,
    export_transcript_jsonl,
)
from ..services.signal_extractor import (
    CandidateExtractionError,
    NoTranscriptTurnsError as NoCandidateTranscriptTurnsError,
    SpeakersNotClassifiedError,
    TranscriptNotFoundError as CandidateTranscriptNotFoundError,
    extract_candidate_signals_for_transcript,
)
from ..services.speaker_classifier import (
    NoTranscriptTurnsError,
    SpeakerClassificationError,
    TranscriptNotFoundError,
    classify_speakers_for_transcript,
)
from ..services.transcript_parser import parse_transcript_text
from ..services.text_encoding import repair_common_utf8_mojibake


router = APIRouter(prefix="/api/transcripts", tags=["transcripts"])
logger = get_logger(__name__)


def _download_response(content: str, *, media_type: str, file_name: str) -> Response:
    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{file_name}"'},
    )


@router.post(
    "/upload",
    response_model=TranscriptUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_transcript(
    file: UploadFile = File(...), db: Session = Depends(get_db)
) -> Transcript:
    file_name = "".join(
        character for character in Path(file.filename or "").name
        if character.isprintable() and character not in "\r\n"
    ).strip()[:255]
    if not file_name or Path(file_name).suffix.lower() != ".txt":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .txt transcript files are supported",
        )

    max_bytes = settings.max_upload_mb * 1024 * 1024
    try:
        file_bytes = await file.read(max_bytes + 1)
    finally:
        await file.close()

    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the {settings.max_upload_mb} MB upload limit",
        )

    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Transcript file is empty",
        )

    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Transcript must be valid UTF-8 text",
        ) from None

    text = repair_common_utf8_mojibake(text)

    if not text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Transcript file is empty",
        )

    transcript = Transcript(
        file_name=file_name,
        raw_text=text,
        token_count=max(1, len(text) // 4),
        status="uploaded",
    )
    db.add(transcript)

    try:
        db.commit()
        db.refresh(transcript)
    except SQLAlchemyError:
        db.rollback()
        logger.error(
            "action=upload status=failed transcript_id=none file_name=%s", file_name
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to save transcript",
        ) from None

    logger.info(
        "action=upload status=uploaded transcript_id=%s file_name=%s",
        transcript.id,
        transcript.file_name,
    )
    return transcript


@router.get("", response_model=list[TranscriptListItem])
def list_transcripts(db: Session = Depends(get_db)) -> list[Transcript]:
    transcripts = db.scalars(
        select(Transcript).order_by(Transcript.created_at.desc(), Transcript.id.desc())
    ).all()
    logger.info("action=list status=ok transcript_id=none file_name=none")
    return list(transcripts)


@router.get("/export/all.csv", response_class=Response)
def export_all_transcripts(
    debug: bool = False, db: Session = Depends(get_db)
) -> Response:
    try:
        content = export_all_transcripts_csv(db, debug=debug)
    except ExportError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch export could not be generated",
        ) from None

    suffix = "debug" if debug else "final"
    return _download_response(
        content,
        media_type="text/csv",
        file_name=f"signalbridge_all_{suffix}.csv",
    )


@router.get("/{transcript_id}", response_model=TranscriptDetail)
def get_transcript(transcript_id: int, db: Session = Depends(get_db)) -> Transcript:
    transcript = db.get(Transcript, transcript_id)
    if transcript is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Transcript not found"
        )

    # Returning raw text is acceptable for this local prototype. Production
    # access to confidential transcript content must be role-controlled.
    logger.info(
        "action=detail status=ok transcript_id=%s file_name=%s",
        transcript.id,
        transcript.file_name,
    )
    return transcript


@router.post("/{transcript_id}/parse", response_model=ParseTranscriptResponse)
def parse_transcript(
    transcript_id: int, db: Session = Depends(get_db)
) -> ParseTranscriptResponse:
    transcript = db.get(Transcript, transcript_id)
    if transcript is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Transcript not found"
        )

    parsed_turns = parse_transcript_text(transcript.raw_text)

    try:
        db.execute(
            delete(TranscriptTurn).where(
                TranscriptTurn.transcript_id == transcript_id
            )
        )
        db.add_all(
            TranscriptTurn(transcript_id=transcript_id, **turn)
            for turn in parsed_turns
        )
        transcript.status = "parsed"
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        logger.error(
            "action=parse transcript_id=%s turn_count=0",
            transcript_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to parse transcript",
        ) from None

    turn_count = len(parsed_turns)
    logger.info(
        "action=parse transcript_id=%s turn_count=%s", transcript_id, turn_count
    )
    return ParseTranscriptResponse(
        transcript_id=transcript_id, status="parsed", turn_count=turn_count
    )


@router.get("/{transcript_id}/turns", response_model=list[TranscriptTurnResponse])
def list_transcript_turns(
    transcript_id: int, db: Session = Depends(get_db)
) -> list[TranscriptTurn]:
    if db.get(Transcript, transcript_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Transcript not found"
        )

    turns = db.scalars(
        select(TranscriptTurn)
        .where(TranscriptTurn.transcript_id == transcript_id)
        .order_by(TranscriptTurn.turn_index)
    ).all()
    logger.info(
        "action=list_turns transcript_id=%s turn_count=%s",
        transcript_id,
        len(turns),
    )
    return list(turns)


@router.post(
    "/{transcript_id}/classify-speakers",
    response_model=SpeakerClassificationResponse,
)
def classify_transcript_speakers(
    transcript_id: int, db: Session = Depends(get_db)
) -> SpeakerClassificationResponse:
    try:
        summary = classify_speakers_for_transcript(transcript_id, db)
    except TranscriptNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Transcript not found"
        ) from None
    except NoTranscriptTurnsError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Transcript must be parsed before speaker classification",
        ) from None
    except SpeakerClassificationError:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Speaker classification could not be completed",
        ) from None

    return SpeakerClassificationResponse.model_validate(summary)


@router.post(
    "/{transcript_id}/extract-candidates",
    response_model=CandidateExtractionResponse,
)
def extract_transcript_candidates(
    transcript_id: int, db: Session = Depends(get_db)
) -> CandidateExtractionResponse:
    try:
        summary = extract_candidate_signals_for_transcript(transcript_id, db)
    except CandidateTranscriptNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Transcript not found"
        ) from None
    except NoCandidateTranscriptTurnsError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Transcript must be parsed before candidate extraction",
        ) from None
    except SpeakersNotClassifiedError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Speakers must be classified before candidate extraction",
        ) from None
    except CandidateExtractionError:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Candidate extraction could not be completed",
        ) from None

    return CandidateExtractionResponse.model_validate(summary)


@router.get(
    "/{transcript_id}/candidates",
    response_model=list[CandidateSignalWithScoreResponse],
)
def list_transcript_candidates(
    transcript_id: int, db: Session = Depends(get_db)
) -> list[CandidateSignalWithScoreResponse]:
    if db.get(Transcript, transcript_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Transcript not found"
        )

    # Candidate quotes are available for local review. Production access to this
    # confidential advisor evidence must be authenticated and role-controlled.
    latest_run = latest_run_for_transcript(db, transcript_id, completed_only=True)
    candidate_query = select(CandidateSignal).options(selectinload(CandidateSignal.score)).where(CandidateSignal.transcript_id == transcript_id)
    if latest_run is not None:
        candidate_query = candidate_query.where(CandidateSignal.analysis_run_id == latest_run.id)
    candidates = db.scalars(candidate_query.order_by(CandidateSignal.item_type, CandidateSignal.id)).all()
    logger.info(
        "action=list_candidates transcript_id=%s batch_number=0 "
        "candidate_count=%s success=true",
        transcript_id,
        len(candidates),
    )
    return [
        CandidateSignalWithScoreResponse(
            **CandidateSignalResponse.model_validate(candidate).model_dump(),
            validator_verdict=(
                candidate.score.validator_verdict if candidate.score else None
            ),
            support_score=candidate.score.support_score if candidate.score else None,
            advisor_side_score=(
                candidate.score.advisor_side_score if candidate.score else None
            ),
            false_positive_risk=(
                candidate.score.false_positive_risk if candidate.score else None
            ),
            rejection_reason=(
                candidate.score.rejection_reason if candidate.score else None
            ),
            advisor_ownership=(
                candidate.score.advisor_ownership if candidate.score else None
            ),
            decision_impact=(
                candidate.score.decision_impact if candidate.score else None
            ),
            explicitness=candidate.score.explicitness if candidate.score else None,
            urgency=candidate.score.urgency if candidate.score else None,
            evidence_quality=(
                candidate.score.evidence_quality if candidate.score else None
            ),
            final_score=candidate.score.final_score if candidate.score else None,
            duplicate_group_id=candidate.duplicate_group_id,
            is_canonical=candidate.is_canonical,
        )
        for candidate in candidates
    ]


@router.post(
    "/{transcript_id}/validate-evidence",
    response_model=EvidenceValidationResponse,
)
def validate_transcript_evidence(
    transcript_id: int, db: Session = Depends(get_db)
) -> EvidenceValidationResponse:
    try:
        summary = validate_evidence_for_transcript(transcript_id, db)
    except EvidenceTranscriptNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Transcript not found"
        ) from None
    except NoCandidateSignalsError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Candidate signals must be extracted before evidence validation",
        ) from None
    except EvidenceValidationError:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Evidence validation could not be completed",
        ) from None

    return EvidenceValidationResponse.model_validate(summary)


@router.post(
    "/{transcript_id}/score-signals",
    response_model=SignalScoringResponse,
)
def score_transcript_signals(
    transcript_id: int, db: Session = Depends(get_db)
) -> SignalScoringResponse:
    try:
        summary = score_signals_for_transcript(transcript_id, db)
    except ScoringTranscriptNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Transcript not found"
        ) from None
    except NoScoringCandidateSignalsError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Candidate signals must exist before scoring",
        ) from None
    except ScoringError:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Signal scoring could not be completed",
        ) from None

    return SignalScoringResponse.model_validate(summary)


@router.post(
    "/{transcript_id}/deduplicate-signals",
    response_model=SignalDeduplicationResponse,
)
def deduplicate_transcript_signals(
    transcript_id: int, db: Session = Depends(get_db)
) -> SignalDeduplicationResponse:
    try:
        summary = deduplicate_signals_for_transcript(transcript_id, db)
    except DeduplicationTranscriptNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Transcript not found"
        ) from None
    except NoDeduplicationCandidateSignalsError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Candidate signals must exist before deduplication",
        ) from None
    except SignalDeduplicationError:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Signal deduplication could not be completed",
        ) from None

    return SignalDeduplicationResponse.model_validate(summary)


@router.post(
    "/{transcript_id}/rerank-final",
    response_model=FinalRerankingResponse,
)
def rerank_transcript_final_signals(
    transcript_id: int, db: Session = Depends(get_db)
) -> FinalRerankingResponse:
    try:
        summary = rerank_final_signals_for_transcript(transcript_id, db)
    except RerankingTranscriptNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Transcript not found"
        ) from None
    except NoRerankingCandidateSignalsError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Candidate signals must exist before final reranking",
        ) from None
    except RerankingError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Final reranking could not be completed",
        ) from None

    return FinalRerankingResponse.model_validate(summary)


@router.post(
    "/{transcript_id}/process-all",
    response_model=PipelineRunResponse,
)
def process_all_transcript_stages(
    transcript_id: int, db: Session = Depends(get_db)
) -> PipelineRunResponse:
    try:
        summary = run_full_pipeline_for_transcript(transcript_id, db)
    except PipelineTranscriptNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transcript not found",
        ) from None
    except PipelineRunError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline failed at step: {exc.step_name}",
        ) from None

    return PipelineRunResponse.model_validate(summary)


@router.get(
    "/{transcript_id}/processing-status",
    response_model=ProcessingStatusResponse,
)
def get_processing_status(
    transcript_id: int, db: Session = Depends(get_db)
) -> ProcessingStatusResponse:
    if db.get(Transcript, transcript_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Transcript not found"
        )
    return ProcessingStatusResponse.model_validate(progress.get(transcript_id))


@router.get(
    "/{transcript_id}/diagnostics",
    response_model=PipelineDiagnosticsResponse,
)
def get_pipeline_diagnostics(
    transcript_id: int, db: Session = Depends(get_db)
) -> PipelineDiagnosticsResponse:
    if db.get(Transcript, transcript_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Transcript not found"
        )
    manifest = progress.get_diagnostics(transcript_id)
    if manifest is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Run diagnostics are not available",
        )
    payload = {
        "transcript_id": manifest["transcript_id"],
        "run_id": manifest["run_id"],
        "status": manifest["status"],
        "started_at": manifest["started_at"],
        "completed_at": manifest["completed_at"],
        "total_duration_ms": round(
            sum(item["duration_ms"] for item in manifest["stage_timings"]), 2
        ),
        "total_call_count": manifest["total_call_count"],
        "total_retry_count": manifest["total_retry_count"],
        "total_tokens": manifest["total_tokens"],
        "total_estimated_cost": manifest["total_estimated_cost"],
        "stages": manifest["stage_usage"],
        "prompt_provenance": manifest["prompt_provenance"],
        "embedding_model": manifest["embedding_model"],
        "scoring_policy": manifest["scoring_policy"],
        "deduplication_threshold": manifest["deduplication_threshold"],
        "reranker_fallback": manifest["reranker_fallback"],
        "failed_stage": manifest["failed_stage"],
        "error_category": manifest["error_type"],
    }
    return PipelineDiagnosticsResponse.model_validate(payload)


def _adjacent_evidence_context(
    candidate: CandidateSignal, turns: list[TranscriptTurn]
) -> list[dict]:
    """Return neighboring turns separately; never rewrite the exact evidence quote."""
    return [
        context_payload(turn)
        for turn in context_turns(candidate, turns, include_source=False)
    ]
@router.get(
    "/{transcript_id}/final-signals",
    response_model=list[FinalSignalResponse],
)
def list_final_signals(
    transcript_id: int, db: Session = Depends(get_db)
) -> list[FinalSignalResponse]:
    if db.get(Transcript, transcript_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Transcript not found"
        )

    # Final evidence contains confidential advisor quotes. Production access
    # must be authenticated and role-controlled.
    latest_run = latest_run_for_transcript(db, transcript_id, completed_only=True)
    final_query = select(FinalSignal).options(selectinload(FinalSignal.signal).selectinload(CandidateSignal.score)).where(FinalSignal.transcript_id == transcript_id)
    if latest_run is not None:
        final_query = final_query.where(FinalSignal.analysis_run_id == latest_run.id)
    final_signals = db.scalars(final_query.order_by(FinalSignal.item_type, FinalSignal.rank)).all()
    candidate_query = select(CandidateSignal).where(CandidateSignal.transcript_id == transcript_id)
    if latest_run is not None:
        candidate_query = candidate_query.where(CandidateSignal.analysis_run_id == latest_run.id)
    candidates = list(db.scalars(candidate_query.order_by(CandidateSignal.id)).all())
    candidates_by_group: dict[str, list[CandidateSignal]] = {}
    for candidate in candidates:
        if candidate.duplicate_group_id:
            candidates_by_group.setdefault(candidate.duplicate_group_id, []).append(candidate)
    turns = list(db.scalars(
        select(TranscriptTurn)
        .where(TranscriptTurn.transcript_id == transcript_id)
        .order_by(TranscriptTurn.turn_index, TranscriptTurn.id)
    ).all())

    responses = [
        FinalSignalResponse(
            transcript_id=final_signal.transcript_id,
            item_type=final_signal.item_type,
            rank=final_signal.rank,
            category=final_signal.signal.category,
            advisor_quote=final_signal.signal.advisor_quote,
            timestamp=final_signal.signal.timestamp,
            evidence_strength=final_signal.signal.evidence_strength,
            rationale=final_signal.signal.rationale,
            final_score=final_signal.signal.score.final_score,
            validator_verdict=final_signal.signal.score.validator_verdict,
            supporting_evidence=[
                candidate.advisor_quote
                for candidate in candidates_by_group.get(
                    final_signal.signal.duplicate_group_id, []
                )
                if candidate.id != final_signal.signal.id
            ],
            evidence_context=_adjacent_evidence_context(final_signal.signal, turns),
        )
        for final_signal in final_signals
    ]
    logger.info(
        "action=list_final_signals transcript_id=%s eligible_count=0 "
        "final_driver_count=%s final_blocker_count=%s used_fallback=false "
        "success=true",
        transcript_id,
        sum(item.item_type == "driver" for item in responses),
        sum(item.item_type == "blocker" for item in responses),
    )
    return responses


@router.get("/{transcript_id}/export.csv", response_class=Response)
def export_transcript_as_csv(
    transcript_id: int,
    debug: bool = False,
    db: Session = Depends(get_db),
) -> Response:
    try:
        content = export_transcript_csv(transcript_id, db, debug=debug)
    except ExportTranscriptNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Transcript not found"
        ) from None
    except ExportError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="CSV export could not be generated",
        ) from None

    suffix = "debug" if debug else "final"
    return _download_response(
        content,
        media_type="text/csv",
        file_name=f"signalbridge_transcript_{transcript_id}_{suffix}.csv",
    )


@router.get("/{transcript_id}/export.jsonl", response_class=Response)
def export_transcript_as_jsonl(
    transcript_id: int,
    debug: bool = False,
    db: Session = Depends(get_db),
) -> Response:
    try:
        content = export_transcript_jsonl(transcript_id, db, debug=debug)
    except ExportTranscriptNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Transcript not found"
        ) from None
    except ExportError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="JSONL export could not be generated",
        ) from None

    suffix = "debug" if debug else "final"
    return _download_response(
        content,
        media_type="application/jsonl",
        file_name=f"signalbridge_transcript_{transcript_id}_{suffix}.jsonl",
    )


@router.delete("/{transcript_id}", response_model=DeleteResponse)
def delete_transcript(transcript_id: int, db: Session = Depends(get_db)) -> DeleteResponse:
    transcript = db.get(Transcript, transcript_id)
    if transcript is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Transcript not found"
        )

    file_name = transcript.file_name
    db.delete(transcript)
    try:
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        logger.error(
            "action=delete status=failed transcript_id=%s file_name=%s",
            transcript_id,
            file_name,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to delete transcript",
        ) from None

    logger.info(
        "action=delete status=ok transcript_id=%s file_name=%s",
        transcript_id,
        file_name,
    )
    return DeleteResponse(status="ok", message="Transcript deleted")
