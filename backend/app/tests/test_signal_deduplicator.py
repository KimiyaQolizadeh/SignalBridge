from collections.abc import Generator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.app.database import Base
from backend.app.models import CandidateSignal, SignalScore, Transcript
from backend.app.services import signal_deduplicator
from backend.app.services.cross_encoder import CrossEncoderError, normalize_score
from backend.app.services.signal_deduplicator import (
    deduplicate_signals_for_transcript,
    get_last_deduplication_diagnostics,
)


VECTOR_SIZE = 1536


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


def unit_vector(index: int) -> list[float]:
    vector = [0.0] * VECTOR_SIZE
    vector[index] = 1.0
    return vector


def create_transcript(db: Session) -> Transcript:
    transcript = Transcript(file_name="synthetic.txt", raw_text="Synthetic content")
    db.add(transcript)
    db.flush()
    return transcript


def add_candidate(
    db: Session,
    transcript: Transcript,
    *,
    verdict: str,
    final_score: float | None,
    quote: str,
    item_type: str = "driver",
    evidence_strength: str = "explicit",
    support_score: float = 0.9,
    false_positive_risk: float = 0.1,
    category: str = "synthetic_category",
    rationale: str = "Synthetic rationale.",
    source_turn_ids: list[int] | None = None,
) -> CandidateSignal:
    candidate = CandidateSignal(
        transcript_id=transcript.id,
        item_type=item_type,
        category=category,
        advisor_quote=quote,
        timestamp="00:00:00",
        evidence_strength=evidence_strength,
        rationale=rationale,
        extraction_confidence=0.9,
        source_turn_ids=source_turn_ids or [1],
    )
    db.add(candidate)
    db.flush()
    candidate.score = SignalScore(
        signal_id=candidate.id,
        validator_verdict=verdict,
        support_score=support_score,
        advisor_side_score=0.9,
        false_positive_risk=false_positive_risk,
        final_score=final_score,
    )
    db.commit()
    return candidate


class StaticScorer:
    model_id = "synthetic-cross-encoder"
    revision = "test"

    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.calls: list[list[tuple[str, str]]] = []

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.calls.append(pairs)
        return self.scores


def enable_mode(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    scorer: StaticScorer | None = None,
) -> None:
    monkeypatch.setattr(signal_deduplicator.settings, "dedup_experiment_mode", mode)
    monkeypatch.setattr(signal_deduplicator.settings, "dedup_shortlist_threshold", 0.7)
    monkeypatch.setattr(
        signal_deduplicator.settings, "dedup_cross_encoder_merge_threshold", 0.8
    )
    monkeypatch.setattr(
        signal_deduplicator.settings,
        "dedup_cross_encoder_representation",
        "decision_factor_evidence",
    )
    if scorer is not None:
        monkeypatch.setattr(
            signal_deduplicator, "load_cross_encoder", lambda _config: scorer
        )


def mock_embeddings(
    monkeypatch: pytest.MonkeyPatch, vectors_by_quote: dict[str, list[float]]
) -> None:
    def fake_embed(text: str, **_: object) -> list[float]:
        for quote, vector in vectors_by_quote.items():
            if quote in text:
                return vector
        raise AssertionError("Unexpected synthetic embedding input")

    monkeypatch.setattr(signal_deduplicator, "embed_text", fake_embed)


def test_rejected_candidate_is_excluded_and_noncanonical(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    candidate = add_candidate(
        db, transcript, verdict="reject", final_score=None, quote="Rejected quote"
    )
    candidate.score.rejection_reason = "quote_not_found"
    db.commit()
    monkeypatch.setattr(
        signal_deduplicator,
        "embed_text",
        lambda *_args, **_kwargs: pytest.fail("Rejected candidate was embedded"),
    )

    summary = deduplicate_signals_for_transcript(transcript.id, db)
    db.refresh(candidate)

    assert candidate.is_canonical is False
    assert candidate.duplicate_group_id is None
    assert summary["rejected_excluded"] == 1


def test_unique_eligible_candidates_become_canonical(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    first = add_candidate(db, transcript, verdict="pass", final_score=4.0, quote="First")
    second = add_candidate(db, transcript, verdict="pass", final_score=3.8, quote="Second")
    mock_embeddings(monkeypatch, {"First": unit_vector(0), "Second": unit_vector(1)})

    deduplicate_signals_for_transcript(transcript.id, db)
    db.refresh(first)
    db.refresh(second)

    assert first.is_canonical is True
    assert second.is_canonical is True
    assert first.duplicate_group_id == f"g_{first.id}"
    assert second.duplicate_group_id == f"g_{second.id}"


def test_similar_candidates_are_grouped(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    first = add_candidate(db, transcript, verdict="pass", final_score=4.0, quote="First")
    second = add_candidate(db, transcript, verdict="pass", final_score=3.0, quote="Second")
    shared_vector = unit_vector(0)
    mock_embeddings(monkeypatch, {"First": shared_vector, "Second": shared_vector})

    deduplicate_signals_for_transcript(transcript.id, db)
    db.refresh(first)
    db.refresh(second)

    assert first.duplicate_group_id == second.duplicate_group_id
    assert sum((first.is_canonical, second.is_canonical)) == 1


def test_canonical_selection_ignores_validation_annotation(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    passed = add_candidate(db, transcript, verdict="pass", final_score=3.0, quote="Pass")
    review = add_candidate(
        db, transcript, verdict="needs_review", final_score=5.0, quote="Review"
    )
    shared_vector = unit_vector(0)
    mock_embeddings(monkeypatch, {"Pass": shared_vector, "Review": shared_vector})

    deduplicate_signals_for_transcript(transcript.id, db)
    db.refresh(passed)
    db.refresh(review)

    assert passed.is_canonical is False
    assert review.is_canonical is True


def test_canonical_selection_prefers_higher_final_score(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    lower = add_candidate(db, transcript, verdict="pass", final_score=3.0, quote="Lower")
    higher = add_candidate(db, transcript, verdict="pass", final_score=4.5, quote="Higher")
    shared_vector = unit_vector(0)
    mock_embeddings(monkeypatch, {"Lower": shared_vector, "Higher": shared_vector})

    deduplicate_signals_for_transcript(transcript.id, db)
    db.refresh(lower)
    db.refresh(higher)

    assert higher.is_canonical is True
    assert lower.is_canonical is False
    assert higher.duplicate_group_id == f"g_{higher.id}"


def test_summary_counts_are_correct(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    first = add_candidate(db, transcript, verdict="pass", final_score=4.5, quote="First")
    second = add_candidate(db, transcript, verdict="pass", final_score=3.5, quote="Second")
    third = add_candidate(
        db,
        transcript,
        verdict="needs_review",
        final_score=4.0,
        quote="Third",
        item_type="blocker",
    )
    rejected = add_candidate(db, transcript, verdict="reject", final_score=None, quote="Rejected")
    rejected.score.rejection_reason = "quote_not_found"
    db.commit()
    mock_embeddings(
        monkeypatch,
        {
            "First": unit_vector(0),
            "Second": unit_vector(0),
            "Third": unit_vector(1),
        },
    )

    summary = deduplicate_signals_for_transcript(transcript.id, db)

    assert first.id != second.id != third.id
    assert summary == {
        "transcript_id": transcript.id,
        "status": "signals_deduplicated",
        "candidate_count": 4,
        "eligible_count": 3,
        "canonical_count": 2,
        "duplicate_count": 1,
        "rejected_excluded": 1,
    }


def test_driver_blocker_pairs_are_never_shortlisted(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    driver = add_candidate(db, transcript, verdict="pass", final_score=4.0, quote="Driver")
    blocker = add_candidate(
        db, transcript, verdict="pass", final_score=4.0, quote="Blocker", item_type="blocker"
    )
    mock_embeddings(monkeypatch, {"Driver": unit_vector(0), "Blocker": unit_vector(0)})
    scorer = StaticScorer([])
    enable_mode(monkeypatch, "cross_encoder_active", scorer)

    deduplicate_signals_for_transcript(transcript.id, db)
    diagnostics = get_last_deduplication_diagnostics(transcript.id)

    assert diagnostics["compatible_pair_count"] == 0
    assert scorer.calls == []
    assert driver.duplicate_group_id != blocker.duplicate_group_id


def test_low_bi_encoder_score_skips_cross_encoder(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    add_candidate(db, transcript, verdict="pass", final_score=4.0, quote="First")
    add_candidate(db, transcript, verdict="pass", final_score=4.0, quote="Second")
    mock_embeddings(monkeypatch, {"First": unit_vector(0), "Second": unit_vector(1)})
    scorer = StaticScorer([])
    enable_mode(monkeypatch, "cross_encoder_active", scorer)

    deduplicate_signals_for_transcript(transcript.id, db)
    diagnostics = get_last_deduplication_diagnostics(transcript.id)

    assert diagnostics["compatible_pair_count"] == 1
    assert diagnostics["shortlisted_pair_count"] == 0
    assert diagnostics["cross_encoder_scored_pair_count"] == 0
    assert scorer.calls == []


@pytest.mark.parametrize(
    ("score", "expected_duplicates"),
    [(0.95, 1), (0.25, 0)],
)
def test_active_mode_uses_cross_encoder_pair_decision(
    db: Session,
    monkeypatch: pytest.MonkeyPatch,
    score: float,
    expected_duplicates: int,
) -> None:
    transcript = create_transcript(db)
    add_candidate(
        db, transcript, verdict="pass", final_score=4.0,
        quote="My partner must approve before we proceed.",
        category="Partner Approval",
    )
    add_candidate(
        db, transcript, verdict="pass", final_score=3.8,
        quote="We cannot proceed until my partner agrees.",
        category="Decision Maker Dependency",
    )
    shared = unit_vector(0)
    mock_embeddings(
        monkeypatch,
        {"My partner": shared, "We cannot": shared},
    )
    scorer = StaticScorer([score])
    enable_mode(monkeypatch, "cross_encoder_active", scorer)

    summary = deduplicate_signals_for_transcript(transcript.id, db)
    diagnostics = get_last_deduplication_diagnostics(transcript.id)

    assert summary["duplicate_count"] == expected_duplicates
    assert diagnostics["cross_encoder_scored_pair_count"] == 1
    assert diagnostics["pair_diagnostics"][0]["cross_encoder_score"] == score


def test_same_concept_family_distinct_business_responses_stay_separate(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    add_candidate(
        db, transcript, verdict="pass", final_score=4.5,
        quote="We will not use third-party products.",
        category="Operating Model Compatibility",
    )
    add_candidate(
        db, transcript, verdict="pass", final_score=4.4,
        quote="We do not delegate portfolio decisions.",
        category="Operating Model Compatibility",
    )
    shared = unit_vector(0)
    mock_embeddings(monkeypatch, {"third-party": shared, "delegate": shared})
    enable_mode(monkeypatch, "cross_encoder_active", StaticScorer([0.2]))

    summary = deduplicate_signals_for_transcript(transcript.id, db)

    assert summary["duplicate_count"] == 0


def test_generated_rationale_is_not_in_default_pair_representation(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    add_candidate(
        db, transcript, verdict="pass", final_score=4.0, quote="Same factor A",
        rationale="Generated rationale alpha.",
    )
    add_candidate(
        db, transcript, verdict="pass", final_score=4.0, quote="Same factor B",
        rationale="Completely different generated rationale beta.",
    )
    shared = unit_vector(0)
    mock_embeddings(monkeypatch, {"Same factor A": shared, "Same factor B": shared})
    scorer = StaticScorer([0.9])
    enable_mode(monkeypatch, "cross_encoder_active", scorer)

    deduplicate_signals_for_transcript(transcript.id, db)
    pair_text = " ".join(scorer.calls[0][0])

    assert "Generated rationale" not in pair_text
    assert "alpha" not in pair_text
    assert "beta" not in pair_text


def test_nontransitive_scores_do_not_bridge_clusters(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    candidates = [
        add_candidate(db, transcript, verdict="pass", final_score=4.0, quote=quote)
        for quote in ("A factor", "B factor", "C factor")
    ]
    shared = unit_vector(0)
    mock_embeddings(monkeypatch, {candidate.advisor_quote: shared for candidate in candidates})
    enable_mode(monkeypatch, "cross_encoder_active", StaticScorer([0.9, 0.1, 0.9]))

    summary = deduplicate_signals_for_transcript(transcript.id, db)
    diagnostics = get_last_deduplication_diagnostics(transcript.id)

    assert summary["canonical_count"] == 2
    assert sorted(len(cluster) for cluster in diagnostics["clusters"]) == [1, 2]


def test_cross_encoder_load_failure_preserves_candidates_in_active_mode(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    first = add_candidate(db, transcript, verdict="pass", final_score=4.0, quote="First")
    second = add_candidate(db, transcript, verdict="pass", final_score=3.0, quote="Second")
    shared = unit_vector(0)
    mock_embeddings(monkeypatch, {"First": shared, "Second": shared})
    enable_mode(monkeypatch, "cross_encoder_active")
    monkeypatch.setattr(
        signal_deduplicator,
        "load_cross_encoder",
        lambda _config: (_ for _ in ()).throw(CrossEncoderError("unavailable")),
    )

    summary = deduplicate_signals_for_transcript(transcript.id, db)
    diagnostics = get_last_deduplication_diagnostics(transcript.id)

    assert summary["duplicate_count"] == 0
    assert first.is_canonical is True and second.is_canonical is True
    assert diagnostics["fallback_used"] is True


def test_cross_encoder_batch_failure_preserves_candidates(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FailingScorer(StaticScorer):
        def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
            raise CrossEncoderError("batch failed")

    transcript = create_transcript(db)
    add_candidate(db, transcript, verdict="pass", final_score=4.0, quote="First")
    add_candidate(db, transcript, verdict="pass", final_score=3.0, quote="Second")
    shared = unit_vector(0)
    mock_embeddings(monkeypatch, {"First": shared, "Second": shared})
    enable_mode(monkeypatch, "cross_encoder_active", FailingScorer([]))

    summary = deduplicate_signals_for_transcript(transcript.id, db)

    assert summary["duplicate_count"] == 0


def test_shadow_mode_preserves_embedding_only_clusters(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    add_candidate(db, transcript, verdict="pass", final_score=4.0, quote="First")
    add_candidate(db, transcript, verdict="pass", final_score=3.0, quote="Second")
    shared = unit_vector(0)
    mock_embeddings(monkeypatch, {"First": shared, "Second": shared})
    enable_mode(monkeypatch, "cross_encoder_shadow", StaticScorer([0.1]))

    summary = deduplicate_signals_for_transcript(transcript.id, db)
    diagnostics = get_last_deduplication_diagnostics(transcript.id)

    assert summary["duplicate_count"] == 1
    assert diagnostics["pair_diagnostics"][0]["proposed_duplicate"] is False


def test_single_candidate_is_safe_in_active_mode(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    add_candidate(db, transcript, verdict="pass", final_score=4.0, quote="Only")
    mock_embeddings(monkeypatch, {"Only": unit_vector(0)})
    scorer = StaticScorer([])
    enable_mode(monkeypatch, "cross_encoder_active", scorer)

    summary = deduplicate_signals_for_transcript(transcript.id, db)

    assert summary["canonical_count"] == 1
    assert scorer.calls == []


def test_score_normalization_is_stable() -> None:
    assert normalize_score(0.0, "sigmoid") == 0.5
    assert normalize_score(1000.0, "sigmoid") == pytest.approx(1.0)
    assert normalize_score(-1000.0, "sigmoid") == pytest.approx(0.0)
    assert normalize_score(0.75, "identity") == 0.75


def test_nearby_reasons_consolidate_under_explicit_misfit_conclusion(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    third_party = add_candidate(
        db, transcript, verdict="pass", final_score=4.5,
        quote="We are not doing any third party, period.",
        item_type="blocker", category="Third-Party Avoidance",
        rationale="The advisor rejects third-party products.", source_turn_ids=[10],
    )
    delegation = add_candidate(
        db, transcript, verdict="pass", final_score=4.4,
        quote="We don't delegate anything to anyone. We don't sell third-party products.",
        item_type="blocker", category="No Outsourcing Preference",
        rationale="The advisor rejects delegation.", source_turn_ids=[11],
    )
    conclusion = add_candidate(
        db, transcript, verdict="needs_review", final_score=3.9,
        quote="So I think, unfortunately, we are a misfit for the models that you have.",
        item_type="blocker", category="Business Model Fit",
        rationale="The advisor concludes the business models do not fit.", source_turn_ids=[12],
    )
    mock_embeddings(monkeypatch, {
        third_party.advisor_quote: unit_vector(0),
        delegation.advisor_quote: unit_vector(1),
        conclusion.advisor_quote: unit_vector(2),
    })

    summary = deduplicate_signals_for_transcript(transcript.id, db)
    diagnostics = get_last_deduplication_diagnostics(transcript.id)
    db.refresh(third_party)
    db.refresh(delegation)
    db.refresh(conclusion)

    assert summary["canonical_count"] == 1
    assert conclusion.is_canonical is True
    assert third_party.is_canonical is False
    assert delegation.is_canonical is False
    assert conclusion.score.validator_verdict == "needs_review"
    assert conclusion.score.final_score == 4.5
    assert "Third-Party Avoidance" in conclusion.rationale
    assert "No Outsourcing Preference" in conclusion.rationale
    assert {row["duplicate_candidate_id"] for row in diagnostics["duplicate_relationships"]} == {
        third_party.id, delegation.id,
    }
    assert all(
        row["canonical_candidate_id"] == conclusion.id
        for row in diagnostics["duplicate_relationships"]
    )


def test_same_topic_independent_blockers_without_conclusion_link_stay_separate(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = create_transcript(db)
    products = add_candidate(
        db, transcript, verdict="pass", final_score=4.2,
        quote="We will not sell third-party products.",
        item_type="blocker", category="Operating Model",
        source_turn_ids=[20],
    )
    control = add_candidate(
        db, transcript, verdict="pass", final_score=4.1,
        quote="I need sole control over client pricing.",
        item_type="blocker", category="Operating Model",
        source_turn_ids=[21],
    )
    mock_embeddings(monkeypatch, {
        products.advisor_quote: unit_vector(0),
        control.advisor_quote: unit_vector(1),
    })

    summary = deduplicate_signals_for_transcript(transcript.id, db)

    assert summary["canonical_count"] == 2
    assert products.duplicate_group_id != control.duplicate_group_id
