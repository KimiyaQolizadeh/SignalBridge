from collections.abc import Generator
from dataclasses import asdict
from uuid import UUID

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.app.database import Base
from backend.app.models import Transcript, TranscriptTurn
from backend.app.schemas import PipelineRunResponse
from backend.app.services import pipeline_runner
from backend.app.services.pipeline_runner import PipelineRunError


@pytest.fixture
def db() -> Generator[Session, None, None]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    Base.metadata.drop_all(engine)


def create_transcript(db: Session) -> Transcript:
    transcript = Transcript(
        file_name="synthetic.txt",
        raw_text="Speaker 1: Synthetic transcript content.",
    )
    db.add(transcript)
    db.commit()
    db.refresh(transcript)
    return transcript


def install_successful_stage_mocks(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[str],
) -> None:
    monkeypatch.setattr(
        pipeline_runner,
        "parse_transcript_text",
        lambda _text: [
            {
                "turn_index": 0,
                "timestamp": None,
                "raw_speaker_label": "Speaker 1",
                "text": "Synthetic transcript content.",
            }
        ],
    )

    stage_names = [
        "classify_speakers",
        "extract_candidates",
        "validate_evidence",
        "score_signals",
        "deduplicate_signals",
    ]
    function_names = [
        "classify_speakers_for_transcript",
        "extract_candidate_signals_for_transcript",
        "validate_evidence_for_transcript",
        "score_signals_for_transcript",
        "deduplicate_signals_for_transcript",
    ]
    for stage_name, function_name in zip(stage_names, function_names, strict=True):
        monkeypatch.setattr(
            pipeline_runner,
            function_name,
            lambda _transcript_id, _db, name=stage_name: (
                calls.append(name) or {"status": "ok"}
            ),
        )

    monkeypatch.setattr(
        pipeline_runner,
        "rerank_final_signals_for_transcript",
        lambda _transcript_id, _db: (
            calls.append("rerank_final")
            or {
                "status": "finalized",
                "final_driver_count": 2,
                "final_blocker_count": 1,
            }
        ),
    )


def test_full_pipeline_calls_steps_in_order(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    calls: list[str] = []
    install_successful_stage_mocks(monkeypatch, calls)

    result = pipeline_runner.run_full_pipeline_for_transcript(transcript.id, db)

    assert [step["name"] for step in result["steps"]] == [
        "parse",
        "classify_speakers",
        "extract_candidates",
        "validate_evidence",
        "score_signals",
        "deduplicate_signals",
        "rerank_final",
    ]
    assert calls == [
        "classify_speakers",
        "extract_candidates",
        "validate_evidence",
        "score_signals",
        "deduplicate_signals",
        "rerank_final",
    ]
    assert db.scalar(
        select(TranscriptTurn).where(TranscriptTurn.transcript_id == transcript.id)
    ) is not None


def test_failed_step_raises_safe_error_with_step_name(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    calls: list[str] = []
    install_successful_stage_mocks(monkeypatch, calls)
    monkeypatch.setattr(
        pipeline_runner,
        "validate_evidence_for_transcript",
        lambda _transcript_id, _db: (_ for _ in ()).throw(RuntimeError("private")),
    )

    with pytest.raises(PipelineRunError) as error:
        pipeline_runner.run_full_pipeline_for_transcript(transcript.id, db)

    assert error.value.step_name == "validate_evidence"
    assert str(error.value) == "Pipeline failed at step: validate_evidence"
    assert db.scalar(
        select(TranscriptTurn).where(TranscriptTurn.transcript_id == transcript.id)
    ) is not None


def test_summary_includes_final_counts_without_real_llm_calls(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    calls: list[str] = []
    install_successful_stage_mocks(monkeypatch, calls)

    result = pipeline_runner.run_full_pipeline_for_transcript(transcript.id, db)

    assert result["status"] == "finalized"
    assert result["final_driver_count"] == 2
    assert result["final_blocker_count"] == 1
    assert all(step["status"] == "ok" for step in result["steps"])


def capture_manifests(
    monkeypatch: pytest.MonkeyPatch,
) -> list[pipeline_runner.PipelineExecutionManifest]:
    manifests: list[pipeline_runner.PipelineExecutionManifest] = []
    monkeypatch.setattr(pipeline_runner, "_log_manifest", manifests.append)
    return manifests


def test_success_manifest_is_internal_and_complete(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    calls: list[str] = []
    install_successful_stage_mocks(monkeypatch, calls)
    manifests = capture_manifests(monkeypatch)

    result = pipeline_runner.run_full_pipeline_for_transcript(transcript.id, db)

    assert "pipeline_manifest" not in result
    assert set(result) == {
        "transcript_id",
        "status",
        "steps",
        "final_driver_count",
        "final_blocker_count",
    }
    assert len(manifests) == 1
    manifest = manifests[0]
    assert UUID(manifest.run_id).version == 4
    assert manifest.status == "succeeded"
    assert [timing.name for timing in manifest.stage_timings] == [
        "parse",
        "classify_speakers",
        "extract_candidates",
        "validate_evidence",
        "score_signals",
        "deduplicate_signals",
        "rerank_final",
    ]
    assert all(timing.duration_ms >= 0 for timing in manifest.stage_timings)
    assert all(
        timing.duration_ms == round(timing.duration_ms, 2)
        for timing in manifest.stage_timings
    )
    assert [(item.stage, item.model) for item in manifest.model_provenance] == [
        ("classify_speakers", pipeline_runner.settings.speaker_classifier_model),
        ("extract_candidates", pipeline_runner.settings.candidate_extractor_model),
        ("validate_evidence", pipeline_runner.settings.evidence_validator_model),
        ("score_signals", pipeline_runner.settings.business_scorer_model),
        ("rerank_final", pipeline_runner.settings.final_reranker_model),
    ]
    assert manifest.embedding_model == pipeline_runner.settings.embedding_model
    assert [item.prompt_file_name for item in manifest.prompt_provenance] == list(
        pipeline_runner.PROMPT_FILES
    )
    assert all(
        len(item.sha256) == 64
        and set(item.sha256) <= set("0123456789abcdef")
        for item in manifest.prompt_provenance
    )
    assert manifest.scoring_policy == pipeline_runner.SCORING_POLICY
    assert (
        manifest.deduplication_threshold
        == pipeline_runner.settings.dedup_similarity_threshold
    )
    assert manifest.reranker_fallback is False


def test_run_ids_are_unique(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    calls: list[str] = []
    install_successful_stage_mocks(monkeypatch, calls)
    manifests = capture_manifests(monkeypatch)

    pipeline_runner.run_full_pipeline_for_transcript(transcript.id, db)
    pipeline_runner.run_full_pipeline_for_transcript(transcript.id, db)

    assert len({manifest.run_id for manifest in manifests}) == 2


def test_reranker_fallback_is_captured(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    calls: list[str] = []
    install_successful_stage_mocks(monkeypatch, calls)
    monkeypatch.setattr(
        pipeline_runner,
        "rerank_final_signals_for_transcript",
        lambda _transcript_id, _db: {
            "status": "finalized",
            "final_driver_count": 1,
            "final_blocker_count": 0,
            "used_fallback": True,
        },
    )
    manifests = capture_manifests(monkeypatch)

    pipeline_runner.run_full_pipeline_for_transcript(transcript.id, db)

    assert manifests[0].reranker_fallback is True


def test_failure_manifest_redacts_exception_and_preserves_pipeline_error(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    calls: list[str] = []
    install_successful_stage_mocks(monkeypatch, calls)
    private_message = "provider-secret-payload"
    monkeypatch.setattr(
        pipeline_runner,
        "validate_evidence_for_transcript",
        lambda _transcript_id, _db: (_ for _ in ()).throw(
            RuntimeError(private_message)
        ),
    )
    manifests = capture_manifests(monkeypatch)

    with pytest.raises(PipelineRunError) as error:
        pipeline_runner.run_full_pipeline_for_transcript(transcript.id, db)

    assert error.value.step_name == "validate_evidence"
    manifest = manifests[0]
    assert manifest.status == "failed"
    assert manifest.failed_stage == "validate_evidence"
    assert manifest.error_type == "RuntimeError"
    assert private_message not in str(asdict(manifest))

    preserved = PipelineRunError("safe_inner_stage")
    monkeypatch.setattr(
        pipeline_runner,
        "validate_evidence_for_transcript",
        lambda _transcript_id, _db: (_ for _ in ()).throw(preserved),
    )
    with pytest.raises(PipelineRunError) as preserved_error:
        pipeline_runner.run_full_pipeline_for_transcript(transcript.id, db)
    assert preserved_error.value is preserved


def test_prompt_provenance_failure_emits_failure_manifest(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    manifests = capture_manifests(monkeypatch)
    monkeypatch.setattr(
        pipeline_runner,
        "prompt_sha256",
        lambda _name: (_ for _ in ()).throw(OSError("private prompt path")),
    )

    with pytest.raises(PipelineRunError) as error:
        pipeline_runner.run_full_pipeline_for_transcript(transcript.id, db)

    assert error.value.step_name == "prompt_provenance"
    assert manifests[0].status == "failed"
    assert manifests[0].failed_stage == "prompt_provenance"
    assert manifests[0].error_type == "OSError"


def test_pipeline_response_schema_has_exact_public_contract() -> None:
    assert set(PipelineRunResponse.model_fields) == {
        "transcript_id",
        "status",
        "steps",
        "final_driver_count",
        "final_blocker_count",
    }


def test_zero_candidate_extraction_completes_with_zero_results(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    calls: list[str] = []
    install_successful_stage_mocks(monkeypatch, calls)
    monkeypatch.setattr(
        pipeline_runner,
        "extract_candidate_signals_for_transcript",
        lambda _transcript_id, _db: {
            "transcript_id": transcript.id,
            "status": "candidates_extracted",
            "candidate_count": 0,
            "driver_candidates": 0,
            "blocker_candidates": 0,
        },
    )
    monkeypatch.setattr(
        pipeline_runner,
        "validate_evidence_for_transcript",
        lambda *_: pytest.fail("Validation must be skipped for zero candidates"),
    )
    manifests = capture_manifests(monkeypatch)

    result = pipeline_runner.run_full_pipeline_for_transcript(transcript.id, db)
    db.refresh(transcript)

    assert result["status"] == "finalized"
    assert result["final_driver_count"] == 0
    assert result["final_blocker_count"] == 0
    assert [step["name"] for step in result["steps"]] == [
        "parse",
        "classify_speakers",
        "extract_candidates",
    ]
    assert transcript.status == "finalized"
    assert manifests[0].status == "succeeded"


def test_progress_transitions_follow_pipeline_order(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    install_successful_stage_mocks(monkeypatch, [])
    transitions: list[str] = []
    monkeypatch.setattr(
        pipeline_runner.progress, "transition",
        lambda _transcript_id, stage, status="processing": transitions.append(stage),
    )

    pipeline_runner.run_full_pipeline_for_transcript(transcript.id, db)

    assert transitions == [
        "parsing", "classifying_speakers", "extracting_candidates",
        "validating_evidence", "scoring_signals", "deduplicating", "reranking",
    ]


def test_observability_storage_failure_does_not_fail_analysis(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    install_successful_stage_mocks(monkeypatch, [])
    manifests = capture_manifests(monkeypatch)
    monkeypatch.setattr(
        pipeline_runner.progress, "store_diagnostics",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("observability only")),
    )

    result = pipeline_runner.run_full_pipeline_for_transcript(transcript.id, db)

    assert result["status"] == "finalized"
    assert manifests[0].status == "succeeded"
