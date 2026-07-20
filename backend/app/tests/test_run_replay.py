from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.app.database import Base
from backend.app.models import AnalysisRun, CandidateSignal, CandidateSnapshot, Transcript
from backend.app.services import run_replay
from backend.app.services.run_persistence import create_analysis_run, mark_run_completed


def test_replay_validation_clones_candidates_without_upstream_calls(monkeypatch) -> None:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        transcript = Transcript(file_name="replay.txt", raw_text="Advisor: We are moving forward.")
        db.add(transcript)
        db.commit()
        source = create_analysis_run(db, transcript)
        candidate = CandidateSignal(
            transcript_id=transcript.id, analysis_run_id=source.id, item_type="driver",
            category="Commitment", advisor_quote="We are moving forward.",
            rationale="The advisor explicitly commits to proceed.", source_turn_ids=[1],
        )
        db.add(candidate)
        db.flush()
        db.add(CandidateSnapshot(
            analysis_run_id=source.id, transcript_id=transcript.id,
            legacy_candidate_id=candidate.id, item_type=candidate.item_type,
            category=candidate.category, advisor_quote=candidate.advisor_quote,
            normalized_evidence="we are moving forward.", rationale=candidate.rationale,
            source_turn_ids=[1], ownership="advisor",
        ))
        db.commit()
        mark_run_completed(db, source.id, {"extracted_candidates": 1})

        calls: list[str] = []
        monkeypatch.setattr(run_replay, "validate_evidence_for_transcript", lambda *args, **kwargs: calls.append("validation") or {})
        monkeypatch.setattr(run_replay, "score_signals_for_transcript", lambda *args, **kwargs: calls.append("scoring") or {})
        monkeypatch.setattr(run_replay, "deduplicate_signals_for_transcript", lambda *args, **kwargs: calls.append("dedup") or {})
        monkeypatch.setattr(run_replay, "rerank_final_signals_for_transcript", lambda *args, **kwargs: calls.append("ranking") or {"final_driver_count": 0, "final_blocker_count": 0})
        monkeypatch.setattr(run_replay, "snapshot_downstream", lambda *args: None)

        replay = run_replay.replay_validation(source.id, db)
        assert calls == ["validation", "scoring", "dedup", "ranking"]
        assert replay.id != source.id
        assert replay.run_type == "replay_validation"
        assert replay.source_run_id == source.id
        assert replay.status == "completed"
        cloned = list(db.scalars(select(CandidateSnapshot).where(CandidateSnapshot.analysis_run_id == replay.id)))
        assert [(item.item_type, item.advisor_quote, item.source_turn_ids) for item in cloned] == [("driver", "We are moving forward.", [1])]
        assert db.get(AnalysisRun, source.id).status == "completed"
