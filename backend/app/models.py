from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector

from .database import Base


class Transcript(Base):
    __tablename__ = "transcripts"

    id: Mapped[int] = mapped_column(primary_key=True)
    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="uploaded")
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    turns: Mapped[list[TranscriptTurn]] = relationship(
        back_populates="transcript", cascade="all, delete-orphan"
    )
    candidate_signals: Mapped[list[CandidateSignal]] = relationship(
        back_populates="transcript", cascade="all, delete-orphan"
    )
    final_signals: Mapped[list[FinalSignal]] = relationship(
        back_populates="transcript", cascade="all, delete-orphan"
    )
    analysis_runs: Mapped[list[AnalysisRun]] = relationship(
        back_populates="transcript", cascade="all, delete-orphan"
    )


class AnalysisRun(Base):
    __tablename__ = "analysis_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    transcript_id: Mapped[int] = mapped_column(
        ForeignKey("transcripts.id", ondelete="CASCADE"), nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="running")
    run_type: Mapped[str] = mapped_column(String(30), nullable=False, default="full")
    source_run_id: Mapped[str | None] = mapped_column(
        ForeignKey("analysis_runs.id", ondelete="SET NULL"), nullable=True, index=True
    )
    input_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    failed_stage: Mapped[str | None] = mapped_column(String(100))
    error_category: Mapped[str | None] = mapped_column(String(100))
    error_message: Mapped[str | None] = mapped_column(Text)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    code_version: Mapped[str | None] = mapped_column(String(100))
    configuration_snapshot: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    summary: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    transcript: Mapped[Transcript] = relationship(back_populates="analysis_runs")


class SpeakerClassificationSnapshot(Base):
    __tablename__ = "speaker_classification_snapshots"
    __table_args__ = (UniqueConstraint("analysis_run_id", "turn_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    analysis_run_id: Mapped[str] = mapped_column(
        ForeignKey("analysis_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    turn_id: Mapped[int] = mapped_column(Integer, nullable=False)
    turn_index: Mapped[int] = mapped_column(Integer, nullable=False)
    original_speaker_label: Mapped[str | None] = mapped_column(String(255))
    classified_role: Mapped[str] = mapped_column(String(50), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    rationale: Mapped[str | None] = mapped_column(Text)
    model_identifier: Mapped[str] = mapped_column(String(100), nullable=False)
    prompt_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    raw_response: Mapped[dict[str, Any] | list[Any] | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class ExtractionBatch(Base):
    __tablename__ = "extraction_batches"
    __table_args__ = (UniqueConstraint("analysis_run_id", "batch_index"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    analysis_run_id: Mapped[str] = mapped_column(ForeignKey("analysis_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    batch_index: Mapped[int] = mapped_column(Integer, nullable=False)
    input_turn_ids: Mapped[list[int]] = mapped_column(JSON, nullable=False)
    input_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    input_snapshot: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False)
    prompt_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    model_identifier: Mapped[str] = mapped_column(String(100), nullable=False)
    model_configuration: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(30), nullable=False)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    token_usage: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    estimated_cost: Mapped[str | None] = mapped_column(String(40))
    finish_reason: Mapped[str | None] = mapped_column(String(100))
    raw_response: Mapped[str | None] = mapped_column(Text)
    raw_item_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    parsed_item_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    post_filter_item_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_category: Mapped[str | None] = mapped_column(String(100))
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class ExtractionBatchItem(Base):
    __tablename__ = "extraction_batch_items"
    __table_args__ = (UniqueConstraint("extraction_batch_id", "item_index"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    extraction_batch_id: Mapped[int] = mapped_column(ForeignKey("extraction_batches.id", ondelete="CASCADE"), nullable=False, index=True)
    item_index: Mapped[int] = mapped_column(Integer, nullable=False)
    raw_structured_item: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    parsed_successfully: Mapped[bool] = mapped_column(Boolean, nullable=False)
    classification: Mapped[str | None] = mapped_column(String(30))
    accepted_after_filter: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    filter_reason: Mapped[str | None] = mapped_column(String(100))
    source_turn_ids: Mapped[list[int] | None] = mapped_column(JSON)
    evidence_text: Mapped[str | None] = mapped_column(Text)
    normalized_evidence: Mapped[str | None] = mapped_column(Text)
    item_type: Mapped[str | None] = mapped_column(String(20))
    category: Mapped[str | None] = mapped_column(String(100))
    rationale: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class CandidateSnapshot(Base):
    __tablename__ = "candidate_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True)
    analysis_run_id: Mapped[str] = mapped_column(ForeignKey("analysis_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    transcript_id: Mapped[int] = mapped_column(ForeignKey("transcripts.id", ondelete="CASCADE"), nullable=False, index=True)
    extraction_batch_item_id: Mapped[int | None] = mapped_column(ForeignKey("extraction_batch_items.id", ondelete="SET NULL"))
    legacy_candidate_id: Mapped[int | None] = mapped_column(Integer, index=True)
    item_type: Mapped[str] = mapped_column(String(20), nullable=False)
    category: Mapped[str] = mapped_column(String(100), nullable=False)
    advisor_quote: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_evidence: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[str | None] = mapped_column(String(50))
    evidence_strength: Mapped[str | None] = mapped_column(String(20))
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    extraction_confidence: Mapped[float | None] = mapped_column(Float)
    source_turn_ids: Mapped[list[int]] = mapped_column(JSON, nullable=False)
    ownership: Mapped[str] = mapped_column(String(50), nullable=False, default="advisor")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class ValidationSnapshot(Base):
    __tablename__ = "validation_snapshots"
    __table_args__ = (UniqueConstraint("analysis_run_id", "candidate_snapshot_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    analysis_run_id: Mapped[str] = mapped_column(ForeignKey("analysis_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    candidate_snapshot_id: Mapped[int] = mapped_column(ForeignKey("candidate_snapshots.id", ondelete="CASCADE"), nullable=False, index=True)
    structured_findings: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    deterministic_findings: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    derived_verdict: Mapped[str] = mapped_column(String(30), nullable=False)
    rejection_reasons: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    review_reasons: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    primary_reason: Mapped[str | None] = mapped_column(Text)
    prompt_hash: Mapped[str | None] = mapped_column(String(64))
    model_identifier: Mapped[str | None] = mapped_column(String(100))
    raw_response: Mapped[str | None] = mapped_column(Text)
    token_usage: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class ScoringSnapshot(Base):
    __tablename__ = "scoring_snapshots"
    id: Mapped[int] = mapped_column(primary_key=True)
    analysis_run_id: Mapped[str] = mapped_column(ForeignKey("analysis_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    candidate_snapshot_id: Mapped[int] = mapped_column(ForeignKey("candidate_snapshots.id", ondelete="CASCADE"), nullable=False)
    eligible: Mapped[bool] = mapped_column(Boolean, nullable=False)
    output: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    rationale: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class DeduplicationSnapshot(Base):
    __tablename__ = "deduplication_snapshots"
    id: Mapped[int] = mapped_column(primary_key=True)
    analysis_run_id: Mapped[str] = mapped_column(ForeignKey("analysis_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    candidate_snapshot_id: Mapped[int] = mapped_column(ForeignKey("candidate_snapshots.id", ondelete="CASCADE"), nullable=False)
    eligible: Mapped[bool] = mapped_column(Boolean, nullable=False)
    is_canonical: Mapped[bool] = mapped_column(Boolean, nullable=False)
    duplicate_group_id: Mapped[str | None] = mapped_column(String(100))
    rationale: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class FinalRankingSnapshot(Base):
    __tablename__ = "final_ranking_snapshots"
    id: Mapped[int] = mapped_column(primary_key=True)
    analysis_run_id: Mapped[str] = mapped_column(ForeignKey("analysis_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    candidate_snapshot_id: Mapped[int] = mapped_column(ForeignKey("candidate_snapshots.id", ondelete="CASCADE"), nullable=False)
    eligible: Mapped[bool] = mapped_column(Boolean, nullable=False)
    selected: Mapped[bool] = mapped_column(Boolean, nullable=False)
    item_type: Mapped[str] = mapped_column(String(20), nullable=False)
    rank: Mapped[int | None] = mapped_column(Integer)
    output: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class TranscriptTurn(Base):
    __tablename__ = "transcript_turns"

    id: Mapped[int] = mapped_column(primary_key=True)
    transcript_id: Mapped[int] = mapped_column(
        ForeignKey("transcripts.id", ondelete="CASCADE"), nullable=False, index=True
    )
    turn_index: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp: Mapped[str | None] = mapped_column(String(50), nullable=True)
    raw_speaker_label: Mapped[str | None] = mapped_column(String(255), nullable=True)
    inferred_role: Mapped[str | None] = mapped_column(String(50), nullable=True)
    role_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)

    transcript: Mapped[Transcript] = relationship(back_populates="turns")


class CandidateSignal(Base):
    __tablename__ = "candidate_signals"

    id: Mapped[int] = mapped_column(primary_key=True)
    transcript_id: Mapped[int] = mapped_column(
        ForeignKey("transcripts.id", ondelete="CASCADE"), nullable=False, index=True
    )
    analysis_run_id: Mapped[str | None] = mapped_column(
        ForeignKey("analysis_runs.id", ondelete="CASCADE"), nullable=True, index=True
    )
    item_type: Mapped[str] = mapped_column(String(20), nullable=False)
    category: Mapped[str] = mapped_column(String(100), nullable=False)
    advisor_quote: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[str | None] = mapped_column(String(50), nullable=True)
    evidence_strength: Mapped[str | None] = mapped_column(String(20), nullable=True)
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    extraction_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    source_turn_ids: Mapped[list[int] | dict[str, Any] | None] = mapped_column(
        JSON, nullable=True
    )
    duplicate_group_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    is_canonical: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    # Embedding supports semantic deduplication and review/search. It is not the
    # source of truth for final decision-making.
    embedding: Mapped[list[float] | None] = mapped_column(Vector(1536), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    transcript: Mapped[Transcript] = relationship(back_populates="candidate_signals")
    score: Mapped[SignalScore | None] = relationship(
        back_populates="signal", cascade="all, delete-orphan", uselist=False
    )
    reviews: Mapped[list[HumanReview]] = relationship(
        back_populates="signal", cascade="all, delete-orphan"
    )


class SignalScore(Base):
    __tablename__ = "signal_scores"

    id: Mapped[int] = mapped_column(primary_key=True)
    signal_id: Mapped[int] = mapped_column(
        ForeignKey("candidate_signals.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    advisor_ownership: Mapped[int | None] = mapped_column(Integer, nullable=True)
    decision_impact: Mapped[int | None] = mapped_column(Integer, nullable=True)
    explicitness: Mapped[int | None] = mapped_column(Integer, nullable=True)
    urgency: Mapped[int | None] = mapped_column(Integer, nullable=True)
    evidence_quality: Mapped[int | None] = mapped_column(Integer, nullable=True)
    final_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    validator_verdict: Mapped[str | None] = mapped_column(String(30), nullable=True)
    support_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    advisor_side_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    false_positive_risk: Mapped[float | None] = mapped_column(Float, nullable=True)
    rejection_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    signal: Mapped[CandidateSignal] = relationship(back_populates="score")


class FinalSignal(Base):
    __tablename__ = "final_signals"

    id: Mapped[int] = mapped_column(primary_key=True)
    transcript_id: Mapped[int] = mapped_column(
        ForeignKey("transcripts.id", ondelete="CASCADE"), nullable=False, index=True
    )
    analysis_run_id: Mapped[str | None] = mapped_column(
        ForeignKey("analysis_runs.id", ondelete="CASCADE"), nullable=True, index=True
    )
    signal_id: Mapped[int] = mapped_column(
        ForeignKey("candidate_signals.id", ondelete="CASCADE"), nullable=False
    )
    item_type: Mapped[str] = mapped_column(String(20), nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    transcript: Mapped[Transcript] = relationship(back_populates="final_signals")
    signal: Mapped[CandidateSignal] = relationship()


class HumanReview(Base):
    __tablename__ = "human_reviews"

    id: Mapped[int] = mapped_column(primary_key=True)
    signal_id: Mapped[int] = mapped_column(
        ForeignKey("candidate_signals.id", ondelete="CASCADE"), nullable=False, index=True
    )
    reviewer_decision: Mapped[str] = mapped_column(String(30), nullable=False)
    reviewer_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    corrected_category: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    signal: Mapped[CandidateSignal] = relationship(back_populates="reviews")
