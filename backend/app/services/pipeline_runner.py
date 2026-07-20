import json
import inspect
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import uuid4
from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy import delete
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ..config import settings
from ..logging_config import get_logger
from ..models import Transcript, TranscriptTurn
from .evidence_validator import validate_evidence_for_transcript
from .pipeline_context import pipeline_run_id
from . import progress
from .prompt_loader import prompt_sha256
from .reranker import rerank_final_signals_for_transcript
from .run_persistence import (
    create_analysis_run,
    mark_run_completed,
    mark_run_failed,
    snapshot_downstream,
)
from .scorer import score_signals_for_transcript
from .signal_deduplicator import (
    deduplicate_signals_for_transcript,
    get_last_deduplication_diagnostics,
)
from .signal_extractor import extract_candidate_signals_for_transcript
from .speaker_classifier import classify_speakers_for_transcript
from .transcript_parser import parse_transcript_text
from .telemetry import finish_collection, get_calls, pipeline_stage, start_collection


logger = get_logger(__name__)

PROMPT_FILES = (
    "01_speaker_role_classifier.txt",
    "02_candidate_signal_extractor.txt",
    "03_evidence_validator.txt",
    "04_business_scorer.txt",
    "05_final_reranker.txt",
)
SCORING_POLICY = {
    "advisor_ownership": 0.30,
    "decision_impact": 0.25,
    "explicitness": 0.20,
    "urgency": 0.15,
    "evidence_quality": 0.10,
}
PROGRESS_STAGE_NAMES = {
    "parse": "parsing",
    "classify_speakers": "classifying_speakers",
    "extract_candidates": "extracting_candidates",
    "validate_evidence": "validating_evidence",
    "score_signals": "scoring_signals",
    "deduplicate_signals": "deduplicating",
    "rerank_final": "reranking",
}


@dataclass(frozen=True)
class PipelinePromptProvenance:
    prompt_file_name: str
    sha256: str


@dataclass(frozen=True)
class PipelineStageTiming:
    name: str
    duration_ms: float


@dataclass(frozen=True)
class PipelineModelProvenance:
    stage: str
    model: str


@dataclass(frozen=True)
class PipelineStageUsage:
    stage: str
    model: str | None
    call_count: int
    retry_count: int
    input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    cached_input_tokens: int | None
    total_tokens: int | None
    estimated_cost: str | None
    duration_ms: float
    status: str


@dataclass
class PipelineExecutionManifest:
    """Internal-only provenance and observability for one pipeline execution."""
    run_id: str
    transcript_id: int
    status: str = "running"
    stage_timings: list[PipelineStageTiming] = field(default_factory=list)
    prompt_provenance: list[PipelinePromptProvenance] = field(default_factory=list)
    model_provenance: list[PipelineModelProvenance] = field(default_factory=list)
    embedding_model: str = ""
    scoring_policy: dict[str, float] = field(default_factory=dict)
    deduplication_threshold: float = 0.0
    reranker_fallback: bool | None = None
    deduplication_diagnostics: dict[str, Any] | None = None
    failed_stage: str | None = None
    error_type: str | None = None
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    completed_at: str | None = None
    stage_usage: list[PipelineStageUsage] = field(default_factory=list)
    total_call_count: int = 0
    total_retry_count: int = 0
    total_tokens: int | None = None
    total_estimated_cost: str | None = None


def _model_provenance() -> list[PipelineModelProvenance]:
    return [
        PipelineModelProvenance("classify_speakers", settings.speaker_classifier_model),
        PipelineModelProvenance("extract_candidates", settings.candidate_extractor_model),
        PipelineModelProvenance("validate_evidence", settings.evidence_validator_model),
        PipelineModelProvenance("score_signals", settings.business_scorer_model),
        PipelineModelProvenance("rerank_final", settings.final_reranker_model),
    ]


def _log_manifest(manifest: PipelineExecutionManifest) -> None:
    logger.info(
        "action=pipeline_manifest manifest=%s",
        json.dumps(asdict(manifest), sort_keys=True, separators=(",", ":")),
    )


def _safe_progress(action: Callable[[], None]) -> None:
    try:
        action()
    except Exception:
        logger.warning("action=progress_observability success=false error_type=ProgressError")


def _aggregate_usage(manifest: PipelineExecutionManifest) -> None:
    calls = get_calls()
    stages = [
        timing.name for timing in manifest.stage_timings
    ]
    if any(call.stage == "embeddings" for call in calls):
        stages.append("embeddings")
    model_lookup = {item.stage: item.model for item in manifest.model_provenance}
    model_lookup["embeddings"] = manifest.embedding_model
    usages: list[PipelineStageUsage] = []
    for stage in stages:
        stage_calls = [call for call in calls if call.stage == stage]
        timing = next((item.duration_ms for item in manifest.stage_timings if item.name == stage), None)
        duration = timing if timing is not None else sum(call.duration_ms for call in stage_calls)
        known_costs = [call.estimated_cost for call in stage_calls if call.estimated_cost is not None]
        token_values = [call.total_tokens for call in stage_calls if call.total_tokens is not None]
        def optional_sum(name: str) -> int | None:
            values = [getattr(call, name) for call in stage_calls if getattr(call, name) is not None]
            return sum(values) if values else None
        usages.append(PipelineStageUsage(
            stage=stage,
            model=stage_calls[0].model if stage_calls else model_lookup.get(stage),
            call_count=len(stage_calls),
            retry_count=sum(call.retry_count for call in stage_calls),
            input_tokens=optional_sum("input_tokens"),
            output_tokens=optional_sum("output_tokens"),
            reasoning_tokens=optional_sum("reasoning_tokens"),
            cached_input_tokens=optional_sum("cached_input_tokens"),
            total_tokens=sum(token_values) if token_values else None,
            estimated_cost=format(sum(known_costs, Decimal("0")), ".8f") if stage_calls and len(known_costs) == len(stage_calls) else None,
            duration_ms=round(max(0.0, duration), 2),
            status="failed" if manifest.failed_stage == stage else "completed",
        ))
    manifest.stage_usage = usages
    manifest.total_call_count = sum(item.call_count for item in usages)
    manifest.total_retry_count = sum(item.retry_count for item in usages)
    totals = [item.total_tokens for item in usages if item.total_tokens is not None]
    manifest.total_tokens = sum(totals) if totals else None
    costs = [Decimal(item.estimated_cost) for item in usages if item.estimated_cost is not None]
    manifest.total_estimated_cost = (
        format(sum(costs, Decimal("0")), ".8f")
        if usages and len(costs) == sum(item.call_count > 0 for item in usages)
        else None
    )


def _publish_manifest(manifest: PipelineExecutionManifest) -> None:
    try:
        manifest.completed_at = datetime.now(timezone.utc).isoformat()
        _aggregate_usage(manifest)
        progress.store_diagnostics(manifest.transcript_id, asdict(manifest))
    except Exception:
        logger.warning("action=manifest_observability success=false error_type=TelemetryError")
    try:
        _log_manifest(manifest)
    except Exception:
        logger.warning("action=manifest_logging success=false error_type=TelemetryError")


def _record_failure(
    manifest: PipelineExecutionManifest, step_name: str, error: Exception
) -> None:
    manifest.status = "failed"
    manifest.failed_stage = step_name
    manifest.error_type = type(error).__name__
    _safe_progress(
        lambda: progress.fail(manifest.transcript_id, step_name, type(error).__name__)
    )
    _publish_manifest(manifest)


class TranscriptNotFoundError(Exception):
    """Raised when the requested transcript does not exist."""


class PipelineRunError(Exception):
    """A safe pipeline failure that identifies the failed stage."""

    def __init__(self, step_name: str) -> None:
        self.step_name = step_name
        super().__init__(f"Pipeline failed at step: {step_name}")


def _call_run_scoped(function: Callable[..., dict], transcript_id: int, db: Session, run_id: str) -> dict:
    """Keep monkeypatched/legacy two-argument stage callables compatible."""
    if "run_id" in inspect.signature(function).parameters:
        return function(transcript_id, db, run_id=run_id)
    return function(transcript_id, db)


def _parse_transcript(transcript: Transcript, db: Session) -> dict:
    existing_turns = list(db.query(TranscriptTurn).filter(TranscriptTurn.transcript_id == transcript.id).all())
    if existing_turns:
        transcript.status = "parsed"
        db.commit()
        return {"transcript_id": transcript.id, "status": "parsed", "turn_count": len(existing_turns), "reused": True}
    parsed_turns = parse_transcript_text(transcript.raw_text)
    try:
        db.execute(
            delete(TranscriptTurn).where(
                TranscriptTurn.transcript_id == transcript.id
            )
        )
        db.add_all(
            TranscriptTurn(transcript_id=transcript.id, **turn)
            for turn in parsed_turns
        )
        transcript.status = "parsed"
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        raise PipelineRunError("parse") from None

    return {
        "transcript_id": transcript.id,
        "status": "parsed",
        "turn_count": len(parsed_turns),
    }


def run_full_pipeline_for_transcript(transcript_id: int, db: Session) -> dict:
    """Run parse → classify → extract → validate → score → deduplicate → rank."""
    """Run every persisted processing stage for one uploaded transcript.

    Existing services commit internally, which is acceptable for this local
    prototype. Production orchestration should track durable job-step state and
    retry from the failed step instead of wrapping the pipeline in one giant
    transaction.
    """
    manifest = PipelineExecutionManifest(
        run_id=str(uuid4()),
        transcript_id=transcript_id,
        model_provenance=_model_provenance(),
        embedding_model=settings.embedding_model,
        scoring_policy=dict(SCORING_POLICY),
        deduplication_threshold=settings.dedup_similarity_threshold,
    )
    context_token = pipeline_run_id.set(manifest.run_id)
    telemetry_token = start_collection()
    _safe_progress(lambda: progress.start(transcript_id, manifest.run_id))
    try:
        try:
            manifest.prompt_provenance = [
                PipelinePromptProvenance(name, prompt_sha256(name))
                for name in PROMPT_FILES
            ]
        except Exception as error:
            _record_failure(manifest, "prompt_provenance", error)
            raise PipelineRunError("prompt_provenance") from None

        return _run_full_pipeline(transcript_id, db, manifest)
    finally:
        finish_collection(telemetry_token)
        pipeline_run_id.reset(context_token)


def _run_full_pipeline(
    transcript_id: int, db: Session, manifest: PipelineExecutionManifest
) -> dict:
    try:
        transcript = db.get(Transcript, transcript_id)
    except SQLAlchemyError as error:
        db.rollback()
        logger.error(
            "action=process_all transcript_id=%s step=load_transcript "
            "status=failed success=false",
            transcript_id,
        )
        _record_failure(manifest, "load_transcript", error)
        raise PipelineRunError("load_transcript") from None

    if transcript is None:
        logger.warning(
            "action=process_all transcript_id=%s step=load_transcript "
            "status=failed success=false",
            transcript_id,
        )
        error = TranscriptNotFoundError("Transcript not found")
        _record_failure(manifest, "load_transcript", error)
        raise error

    try:
        create_analysis_run(db, transcript, run_id=manifest.run_id)
    except SQLAlchemyError as error:
        db.rollback()
        _record_failure(manifest, "create_run", error)
        raise PipelineRunError("create_run") from None

    steps: list[dict[str, Any]] = []
    pipeline_steps: list[tuple[str, Callable[[], dict]]] = [
        ("parse", lambda: _parse_transcript(transcript, db)),
        (
            "classify_speakers",
            lambda: _call_run_scoped(classify_speakers_for_transcript, transcript_id, db, manifest.run_id),
        ),
        (
            "extract_candidates",
            lambda: _call_run_scoped(extract_candidate_signals_for_transcript, transcript_id, db, manifest.run_id),
        ),
        (
            "validate_evidence",
            lambda: _call_run_scoped(validate_evidence_for_transcript, transcript_id, db, manifest.run_id),
        ),
        ("score_signals", lambda: _call_run_scoped(score_signals_for_transcript, transcript_id, db, manifest.run_id)),
        (
            "deduplicate_signals",
            lambda: _call_run_scoped(deduplicate_signals_for_transcript, transcript_id, db, manifest.run_id),
        ),
        (
            "rerank_final",
            lambda: _call_run_scoped(rerank_final_signals_for_transcript, transcript_id, db, manifest.run_id),
        ),
    ]

    for step_name, run_step in pipeline_steps:
        public_stage = PROGRESS_STAGE_NAMES[step_name]
        _safe_progress(lambda stage=public_stage: progress.transition(transcript_id, stage))
        logger.info(
            "action=process_all transcript_id=%s step=%s "
            "status=started success=pending",
            transcript_id,
            step_name,
        )
        started_at = time.monotonic()
        stage_token = pipeline_stage.set(step_name)
        try:
            details = run_step()
        except PipelineRunError as error:
            manifest.stage_timings.append(
                PipelineStageTiming(
                    step_name,
                    round(max(0.0, (time.monotonic() - started_at) * 1000), 2),
                )
            )
            logger.error(
                "action=process_all transcript_id=%s step=%s "
                "status=failed success=false",
                transcript_id,
                step_name,
            )
            _record_failure(manifest, step_name, error)
            mark_run_failed(db, manifest.run_id, failed_stage=step_name, error_category=type(error).__name__, error_message=f"Pipeline failed at {step_name}")
            raise
        except Exception as error:
            manifest.stage_timings.append(
                PipelineStageTiming(
                    step_name,
                    round(max(0.0, (time.monotonic() - started_at) * 1000), 2),
                )
            )
            # Downstream errors are deliberately replaced with a stage-only error.
            # Raw exception details could contain confidential model/provider data.
            logger.error(
                "action=process_all transcript_id=%s step=%s "
                "status=failed success=false",
                transcript_id,
                step_name,
            )
            _record_failure(manifest, step_name, error)
            mark_run_failed(db, manifest.run_id, failed_stage=step_name, error_category=type(error).__name__, error_message=f"Pipeline failed at {step_name}")
            raise PipelineRunError(step_name) from None
        finally:
            pipeline_stage.reset(stage_token)

        manifest.stage_timings.append(
            PipelineStageTiming(
                step_name,
                round(max(0.0, (time.monotonic() - started_at) * 1000), 2),
            )
        )
        steps.append({"name": step_name, "status": "ok", "details": details})
        if step_name == "deduplicate_signals":
            manifest.deduplication_diagnostics = (
                get_last_deduplication_diagnostics(transcript_id)
            )
        logger.info(
            "action=process_all transcript_id=%s step=%s "
            "status=ok success=true",
            transcript_id,
            step_name,
        )

        if (
            step_name == "extract_candidates"
            and "candidate_count" in details
            and int(details["candidate_count"]) == 0
        ):
            transcript.status = "finalized"
            try:
                db.commit()
            except SQLAlchemyError as error:
                db.rollback()
                _record_failure(manifest, "finalize_empty_result", error)
                raise PipelineRunError("finalize_empty_result") from None

            manifest.reranker_fallback = False
            manifest.status = "succeeded"
            mark_run_completed(db, manifest.run_id, {"extracted_candidates": 0, "final_driver_count": 0, "final_blocker_count": 0})
            _safe_progress(lambda: progress.complete(transcript_id, without_results=True))
            _publish_manifest(manifest)
            return {
                "transcript_id": transcript_id,
                "status": "finalized",
                "steps": steps,
                "final_driver_count": 0,
                "final_blocker_count": 0,
            }

    final_details = steps[-1]["details"]
    manifest.reranker_fallback = bool(final_details.get("used_fallback", False))
    manifest.status = "succeeded"
    snapshot_downstream(db, manifest.run_id)
    mark_run_completed(db, manifest.run_id, {"extracted_candidates": next((int(step["details"].get("candidate_count", 0)) for step in steps if step["name"] == "extract_candidates"), 0), "final_driver_count": int(final_details.get("final_driver_count", 0)), "final_blocker_count": int(final_details.get("final_blocker_count", 0))})
    _safe_progress(lambda: progress.complete(transcript_id))
    _publish_manifest(manifest)
    return {
        "transcript_id": transcript_id,
        "status": "finalized",
        "steps": steps,
        "final_driver_count": int(final_details.get("final_driver_count", 0)),
        "final_blocker_count": int(final_details.get("final_blocker_count", 0)),
    }
