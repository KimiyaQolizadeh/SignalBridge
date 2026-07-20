"""Small deterministic end-to-end precision guard at final-selection boundaries."""

from __future__ import annotations

from collections.abc import Generator

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.app.database import Base
from backend.app.models import CandidateSignal, FinalSignal, SignalScore, Transcript
from backend.app.services import reranker, signal_deduplicator
from backend.app.services.reranker import rerank_final_signals_for_transcript
from backend.app.services.signal_deduplicator import (
    _embedding_only_groups,
    deduplicate_signals_for_transcript,
)


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


def _candidate(
    db: Session,
    transcript: Transcript,
    *,
    category: str,
    quote: str,
    verdict: str,
    item_type: str = "driver",
    embedding: list[float] | None = None,
) -> CandidateSignal:
    candidate = CandidateSignal(
        transcript_id=transcript.id,
        item_type=item_type,
        category=category,
        advisor_quote=quote,
        timestamp="00:01:00",
        evidence_strength="explicit",
        rationale="Grounded synthetic rationale.",
        extraction_confidence=0.9,
        source_turn_ids=[1],
        embedding=embedding,
        duplicate_group_id=f"g_{category}",
        is_canonical=True,
    )
    db.add(candidate)
    db.flush()
    candidate.score = SignalScore(
        signal_id=candidate.id,
        validator_verdict=verdict,
        support_score=0.95,
        advisor_side_score=0.95,
        false_positive_risk=0.05,
        advisor_ownership=5,
        decision_impact=5,
        explicitness=5,
        urgency=4,
        evidence_quality=5,
        final_score=4.8,
    )
    db.commit()
    return candidate


def _transcript(db: Session) -> Transcript:
    transcript = Transcript(file_name="precision-fixture.txt", raw_text="Synthetic")
    db.add(transcript)
    db.flush()
    return transcript


def _run_without_model(db: Session, monkeypatch: pytest.MonkeyPatch, transcript: Transcript) -> list[FinalSignal]:
    monkeypatch.setattr(reranker, "call_llm_json", lambda **_: reranker.LLMClientError("offline"))
    # Raise the safe client error to exercise the deterministic production fallback.
    def unavailable(**_kwargs: object) -> object:
        raise reranker.LLMClientError("offline")
    monkeypatch.setattr(reranker, "call_llm_json", unavailable)
    rerank_final_signals_for_transcript(transcript.id, db)
    return list(db.scalars(select(FinalSignal).where(FinalSignal.transcript_id == transcript.id)).all())


@pytest.mark.parametrize(
    ("category", "quote", "item_type"),
    [
        ("Representative claim", "We provide complete transition support.", "driver"),
        ("Scheduling", "Let's reconnect next Tuesday.", "driver"),
        ("Unsupported dependency", "What are the fees?", "blocker"),
    ],
)
def test_rejected_precision_risks_cannot_become_final(
    db: Session,
    monkeypatch: pytest.MonkeyPatch,
    category: str,
    quote: str,
    item_type: str,
) -> None:
    transcript = _transcript(db)
    _candidate(
        db,
        transcript,
        category=category,
        quote=quote,
        item_type=item_type,
        verdict="reject",
    )
    assert _run_without_model(db, monkeypatch, transcript) == []


def test_valid_explicit_advisor_driver_can_become_final(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = _transcript(db)
    driver = _candidate(
        db,
        transcript,
        category="Commitment",
        quote="We're moving forward with Optimize.",
        verdict="pass",
    )
    saved = _run_without_model(db, monkeypatch, transcript)
    assert [item.signal_id for item in saved] == [driver.id]


def test_substantively_distinct_broad_and_narrow_factors_do_not_auto_merge(
    db: Session,
) -> None:
    transcript = _transcript(db)
    broad = _candidate(
        db,
        transcript,
        category="Operational support",
        quote="I need broader operational support.",
        verdict="pass",
        embedding=[1.0] + [0.0] * 1535,
    )
    narrow = _candidate(
        db,
        transcript,
        category="Technology migration",
        quote="The CRM migration must be resolved first.",
        verdict="pass",
        embedding=[0.0, 1.0] + [0.0] * 1534,
    )
    groups = _embedding_only_groups([broad, narrow], threshold=0.86)
    assert {frozenset(item.id for item in group) for group in groups} == {
        frozenset({broad.id}),
        frozenset({narrow.id}),
    }


def test_grounded_needs_review_candidate_can_reach_final_signal(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = _transcript(db)
    _candidate(
        db,
        transcript,
        category="Ambiguous interest",
        quote="That might be interesting.",
        verdict="needs_review",
    )
    saved = _run_without_model(db, monkeypatch, transcript)
    assert len(saved) == 1
    assert saved[0].signal.score.validator_verdict == "needs_review"


def test_default_deduplication_does_not_load_cross_encoder(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = _transcript(db)
    _candidate(
        db,
        transcript,
        category="Supported factor",
        quote="I need transition support.",
        verdict="pass",
        embedding=[1.0] + [0.0] * 1535,
    )
    monkeypatch.setattr(
        signal_deduplicator,
        "load_cross_encoder",
        lambda *_args, **_kwargs: pytest.fail(
            "default deduplication loaded the cross-encoder"
        ),
    )
    monkeypatch.setattr(
        signal_deduplicator.settings,
        "dedup_experiment_mode",
        "existing_embedding_only",
    )

    summary = deduplicate_signals_for_transcript(transcript.id, db)

    assert summary["canonical_count"] == 1
