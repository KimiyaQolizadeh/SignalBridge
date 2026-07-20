import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.app.database import Base
from backend.app.models import CandidateSignal, FinalSignal, SignalScore, Transcript, TranscriptTurn


from backend.app.api import transcripts
from backend.app.services import progress


def test_process_all_returns_zero_result_without_server_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        transcripts,
        "run_full_pipeline_for_transcript",
        lambda _transcript_id, _db: {
            "transcript_id": 4,
            "status": "finalized",
            "steps": [
                {
                    "name": "extract_candidates",
                    "status": "ok",
                    "details": {"candidate_count": 0},
                }
            ],
            "final_driver_count": 0,
            "final_blocker_count": 0,
        },
    )

    response = transcripts.process_all_transcript_stages(4, db=object())

    assert response.status == "finalized"
    assert response.final_driver_count == 0
    assert response.final_blocker_count == 0


class ExistingTranscriptDB:
    def get(self, _model: object, _identifier: int) -> object:
        return object()


def test_processing_status_endpoint_before_during_and_after() -> None:
    db = ExistingTranscriptDB()
    idle = transcripts.get_processing_status(8801, db=db)
    assert idle.status == "idle"
    progress.start(8801, "safe-run-id")
    progress.transition(8801, "extracting_candidates")
    running = transcripts.get_processing_status(8801, db=db)
    assert running.status == "processing"
    assert running.current_stage == "extracting_candidates"
    progress.complete(8801)
    completed = transcripts.get_processing_status(8801, db=db)
    assert completed.status == "completed"
    assert completed.completed_at is not None


def test_processing_status_endpoint_failure_is_redacted() -> None:
    progress.start(8802, "safe-run-id")
    progress.fail(8802, "validating_evidence", "EvidenceValidationError")
    response = transcripts.get_processing_status(8802, db=ExistingTranscriptDB())
    assert response.status == "failed"
    assert response.error_category == "EvidenceValidationError"
    assert set(response.model_dump()) == {
        "transcript_id", "run_id", "current_stage", "status", "started_at",
        "updated_at", "completed_at", "elapsed_seconds", "error_category",
    }


def test_diagnostics_endpoint_is_safe_and_additive() -> None:
    progress.store_diagnostics(8803, {
        "run_id": "run-id", "transcript_id": 8803, "status": "succeeded",
        "started_at": "2026-01-01T00:00:00+00:00",
        "completed_at": "2026-01-01T00:00:01+00:00",
        "stage_timings": [{"name": "parse", "duration_ms": 10.0}],
        "prompt_provenance": [{"prompt_file_name": "safe.txt", "sha256": "a" * 64}],
        "stage_usage": [{"stage": "parse", "model": None, "call_count": 0,
            "retry_count": 0, "input_tokens": None, "output_tokens": None,
            "reasoning_tokens": None, "cached_input_tokens": None,
            "total_tokens": None, "estimated_cost": None,
            "duration_ms": 10.0, "status": "completed"}],
        "total_call_count": 0, "total_retry_count": 0, "total_tokens": None,
        "total_estimated_cost": None, "embedding_model": "text-embedding-3-small",
        "scoring_policy": {"safe": 1.0}, "deduplication_threshold": 0.86,
        "reranker_fallback": False, "failed_stage": None, "error_type": None,
    })
    response = transcripts.get_pipeline_diagnostics(8803, db=ExistingTranscriptDB())
    serialized = response.model_dump_json()
    assert response.run_id == "run-id"
    assert response.total_duration_ms == 10.0
    assert "transcript text" not in serialized
    assert "prompt content" not in serialized


def test_final_signals_include_support_context_and_distinct_verdict_counts() -> None:
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        transcript = Transcript(
            file_name="context.txt", raw_text="Synthetic", status="finalized"
        )
        db.add(transcript)
        db.flush()
        turns = [
            TranscriptTurn(
                transcript_id=transcript.id, turn_index=index,
                raw_speaker_label=speaker, inferred_role=role, text=text,
            )
            for index, speaker, role, text in [
                (0, "Recruiter", "representative", "We need nationwide service coverage."),
                (1, "Advisor", "advisor", "I need to know that you guys are already there."),
                (2, "Recruiter", "representative", "We currently cover all fifty states."),
            ]
        ]
        db.add_all(turns)
        db.flush()
        canonical = CandidateSignal(
            transcript_id=transcript.id, item_type="driver", category="Coverage",
            advisor_quote="I need to know that you guys are already there.",
            evidence_strength="explicit", rationale="Coverage availability matters.",
            source_turn_ids=[turns[1].id], duplicate_group_id="g_context",
            is_canonical=True,
        )
        duplicate = CandidateSignal(
            transcript_id=transcript.id, item_type="driver", category="Nationwide Access",
            advisor_quote="We need nationwide service coverage.",
            evidence_strength="explicit", rationale="Nationwide access matters.",
            source_turn_ids=[turns[0].id], duplicate_group_id="g_context",
            is_canonical=False,
        )
        review = CandidateSignal(
            transcript_id=transcript.id, item_type="blocker", category="Timing",
            advisor_quote="The timing may be difficult.", evidence_strength="implied",
            rationale="Timing remains uncertain.", source_turn_ids=[turns[1].id],
            duplicate_group_id="g_review", is_canonical=True,
        )
        db.add_all([canonical, duplicate, review])
        db.flush()
        canonical.score = SignalScore(signal_id=canonical.id, validator_verdict="pass", final_score=4.2)
        duplicate.score = SignalScore(signal_id=duplicate.id, validator_verdict="pass", final_score=4.0)
        review.score = SignalScore(signal_id=review.id, validator_verdict="needs_review", final_score=3.4)
        db.add_all([
            FinalSignal(transcript_id=transcript.id, signal_id=canonical.id, item_type="driver", rank=1),
            FinalSignal(transcript_id=transcript.id, signal_id=review.id, item_type="blocker", rank=1),
        ])
        db.commit()

        response = transcripts.list_final_signals(transcript.id, db=db)

        coverage = next(item for item in response if item.category == "Coverage")
        assert coverage.advisor_quote == "I need to know that you guys are already there."
        assert coverage.supporting_evidence == ["We need nationwide service coverage."]
        assert [item.text for item in coverage.evidence_context] == [
            "We need nationwide service coverage.",
            "We currently cover all fifty states.",
        ]
        assert {verdict: sum(item.validator_verdict == verdict for item in response)
                for verdict in ("pass", "needs_review")} == {
            "pass": 1, "needs_review": 1,
        }
    Base.metadata.drop_all(engine)
