"""Consolidate same-direction evidence without merging independent decision effects."""

from __future__ import annotations

import json
import math
import re
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from threading import RLock

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, selectinload

from ..config import settings
from ..logging_config import get_logger
from ..models import CandidateSignal, Transcript
from .cross_encoder import CrossEncoderConfig, CrossEncoderError, CrossEncoderScorer, load_cross_encoder
from .embedding_client import EmbeddingClientError, embed_text
from .eligibility_policy import validation_allows_business_pipeline


EMBEDDING_DIMENSION = 1536
EXPERIMENT_MODES = {"existing_embedding_only", "cross_encoder_shadow", "cross_encoder_active"}
REPRESENTATIONS = {"type_category_evidence", "decision_factor_evidence", "decision_factor_evidence_rationale", "evidence_only"}
PAIR_QUESTION = (
    "Do these candidates represent the same underlying recruiting decision "
    "factor, require substantially the same business response, and deserve one "
    "final signal slot?"
)
logger = get_logger(__name__)
_diagnostics_lock = RLock()
_last_diagnostics: dict[int, dict] = {}


@dataclass(frozen=True)
class PairDiagnostic:
    left_id: int
    right_id: int
    bi_encoder_score: float
    cross_encoder_score: float | None
    shortlisted: bool
    proposed_duplicate: bool | None


@dataclass
class DeduplicationDiagnostics:
    transcript_id: int
    mode: str
    model_name: str | None
    model_revision: str | None
    representation: str
    compatible_pair_count: int = 0
    shortlisted_pair_count: int = 0
    cross_encoder_scored_pair_count: int = 0
    shortlist_threshold: float = 0.0
    merge_threshold: float = 0.0
    fallback_used: bool = False
    failure_type: str | None = None
    inference_duration_ms: float = 0.0
    batch_count: int = 0
    pair_diagnostics: list[PairDiagnostic] = field(default_factory=list)
    clusters: list[list[int]] = field(default_factory=list)
    representatives: list[dict] = field(default_factory=list)
    duplicate_relationships: list[dict] = field(default_factory=list)


class TranscriptNotFoundError(Exception):
    """Raised when a requested transcript does not exist."""


class NoCandidateSignalsError(Exception):
    """Raised when deduplication is requested before candidate extraction."""


class SignalDeduplicationError(Exception):
    """A safe error for embedding, grouping, or persistence failures."""


def get_last_deduplication_diagnostics(transcript_id: int) -> dict | None:
    with _diagnostics_lock:
        return deepcopy(_last_diagnostics.get(transcript_id))


def _store_diagnostics(diagnostics: DeduplicationDiagnostics) -> None:
    data = asdict(diagnostics)
    with _diagnostics_lock:
        _last_diagnostics[diagnostics.transcript_id] = deepcopy(data)
    logger.info("action=dedup_experiment diagnostics=%s", json.dumps(data, sort_keys=True, separators=(",", ":")))


def _embedding_text(candidate: CandidateSignal) -> str:
    """Legacy production fingerprint retained for embedding-only comparison."""
    return "\n".join((candidate.item_type, candidate.category, candidate.rationale, candidate.advisor_quote))


def _normalized_text(value: str) -> str:
    return " ".join(value.lower().strip().split())


def _pair_representation(candidate: CandidateSignal, representation: str) -> str:
    evidence = candidate.advisor_quote.strip()
    category = _normalized_text(candidate.category)
    direction = "supports move" if candidate.item_type == "driver" else "opposes move"
    if representation == "type_category_evidence":
        return f"Type: {candidate.item_type}\nCategory: {category}\nAdvisor evidence: {evidence}"
    if representation == "decision_factor_evidence":
        return f"Type: {candidate.item_type}\nDecision direction: {direction}\nDecision factor: {category}\nAdvisor evidence: {evidence}"
    if representation == "decision_factor_evidence_rationale":
        return f"Type: {candidate.item_type}\nDecision direction: {direction}\nDecision factor: {category}\nAdvisor evidence: {evidence}\nGenerated rationale: {candidate.rationale.strip()}"
    if representation == "evidence_only":
        return f"Advisor evidence: {evidence}"
    raise CrossEncoderError("Unsupported pair representation")


def _as_float_list(embedding: object) -> list[float]:
    if embedding is None:
        raise ValueError("missing embedding")
    values = [float(value) for value in embedding]  # type: ignore[union-attr]
    if len(values) != EMBEDDING_DIMENSION:
        raise ValueError("unexpected embedding dimension")
    return values


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def _legacy_canonical_key(candidate: CandidateSignal) -> tuple:
    score = candidate.score
    return (
        score.final_score if score.final_score is not None else float("-inf"),
        candidate.evidence_strength == "explicit",
        score.support_score if score.support_score is not None else float("-inf"),
        -(score.false_positive_risk if score.false_positive_risk is not None else 1.0),
        -candidate.id,
    )


def _experimental_canonical_key(candidate: CandidateSignal) -> tuple:
    score = candidate.score
    return (
        candidate.evidence_strength == "explicit",
        score.advisor_side_score if score.advisor_side_score is not None else -1.0,
        score.final_score if score.final_score is not None else float("-inf"),
        score.support_score if score.support_score is not None else -1.0,
        -(score.false_positive_risk if score.false_positive_risk is not None else 1.0),
        len(candidate.advisor_quote.split()),
        candidate.timestamp or "",
        -candidate.id,
    )


def _compatible_pairs(candidates: list[CandidateSignal]) -> list[tuple[CandidateSignal, CandidateSignal]]:
    return [(left, right) for index, left in enumerate(candidates) for right in candidates[index + 1:] if left.item_type == right.item_type]


def _pair_key(left: CandidateSignal, right: CandidateSignal) -> tuple[int, int]:
    return (min(left.id, right.id), max(left.id, right.id))


_CONCLUSION_PATTERN = re.compile(
    r"\b(misfit|not (?:a )?fit|incompatib(?:le|ility)|does(?:n't| not) fit|"
    r"won't work|cannot work|can't work)\b",
    re.IGNORECASE,
)
_CONCLUSION_LINK_PATTERN = re.compile(
    r"\b(so|therefore|thus|consequently|for that reason|because of that|unfortunately)\b",
    re.IGNORECASE,
)


def _turn_ids(candidate: CandidateSignal) -> list[int]:
    value = candidate.source_turn_ids
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, int)]


def _is_decision_conclusion(candidate: CandidateSignal) -> bool:
    text = " ".join((candidate.category, candidate.advisor_quote, candidate.rationale))
    return bool(_CONCLUSION_PATTERN.search(text))


def _is_nearby_reason_conclusion(
    left: CandidateSignal, right: CandidateSignal, *, max_turn_distance: int = 4
) -> bool:
    if left.item_type != right.item_type:
        return False
    conclusion, reason = (
        (left, right) if _is_decision_conclusion(left) else (right, left)
    )
    if not _is_decision_conclusion(conclusion):
        return False
    if not _CONCLUSION_LINK_PATTERN.search(conclusion.advisor_quote):
        return False
    conclusion_turns, reason_turns = _turn_ids(conclusion), _turn_ids(reason)
    if not conclusion_turns or not reason_turns:
        return False
    return 0 <= min(conclusion_turns) - max(reason_turns) <= max_turn_distance


def _merge_reason_conclusion_groups(
    groups: list[list[CandidateSignal]],
) -> list[list[CandidateSignal]]:
    candidates = [candidate for group in groups for candidate in group]
    parents = {candidate.id: candidate.id for candidate in candidates}

    def find(candidate_id: int) -> int:
        while parents[candidate_id] != candidate_id:
            parents[candidate_id] = parents[parents[candidate_id]]
            candidate_id = parents[candidate_id]
        return candidate_id

    def union(left_id: int, right_id: int) -> None:
        left_root, right_root = find(left_id), find(right_id)
        if left_root != right_root:
            parents[right_root] = left_root

    for group in groups:
        for member in group[1:]:
            union(group[0].id, member.id)
    for index, left in enumerate(candidates):
        for right in candidates[index + 1:]:
            if _is_nearby_reason_conclusion(left, right):
                union(left.id, right.id)

    merged: dict[int, list[CandidateSignal]] = {}
    for candidate in candidates:
        merged.setdefault(find(candidate.id), []).append(candidate)
    return list(merged.values())


def _embedding_only_groups(candidates: list[CandidateSignal], threshold: float) -> list[list[CandidateSignal]]:
    parents = list(range(len(candidates)))
    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index
    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root
    for left_index, left in enumerate(candidates):
        for right_index in range(left_index + 1, len(candidates)):
            right = candidates[right_index]
            if left.item_type == right.item_type and cosine_similarity(_as_float_list(left.embedding), _as_float_list(right.embedding)) >= threshold:
                union(left_index, right_index)
    grouped: dict[int, list[CandidateSignal]] = {}
    for index, candidate in enumerate(candidates):
        grouped.setdefault(find(index), []).append(candidate)
    return list(grouped.values())


def _complete_link_groups(candidates: list[CandidateSignal], pair_decisions: dict[tuple[int, int], bool]) -> list[list[CandidateSignal]]:
    """Greedy complete-link clustering prevents weak-bridge transitive merges."""
    groups: list[list[CandidateSignal]] = []
    for candidate in sorted(candidates, key=lambda item: item.id):
        for group in groups:
            if candidate.item_type == group[0].item_type and all(pair_decisions.get(_pair_key(candidate, member), False) for member in group):
                group.append(candidate)
                break
        else:
            groups.append([candidate])
    return groups


def _cross_encoder_config() -> CrossEncoderConfig:
    return CrossEncoderConfig(
        model_id=settings.dedup_cross_encoder_model,
        revision=settings.dedup_cross_encoder_revision,
        device=settings.dedup_cross_encoder_device,
        batch_size=settings.dedup_cross_encoder_batch_size,
        max_length=settings.dedup_cross_encoder_max_length,
        normalization=settings.dedup_cross_encoder_normalization,
        cache_dir=settings.dedup_cross_encoder_cache_dir,
        local_files_only=settings.dedup_cross_encoder_local_files_only,
        trust_remote_code=settings.dedup_cross_encoder_trust_remote_code,
    )


def _experimental_groups(
    candidates: list[CandidateSignal],
    diagnostics: DeduplicationDiagnostics,
    experimental_embeddings: dict[int, list[float]] | None = None,
    scorer: CrossEncoderScorer | None = None,
) -> list[list[CandidateSignal]]:
    pairs = _compatible_pairs(candidates)
    diagnostics.compatible_pair_count = len(pairs)
    shortlisted: list[tuple[CandidateSignal, CandidateSignal, float]] = []
    pair_rows: list[PairDiagnostic] = []
    for left, right in pairs:
        left_embedding = (
            experimental_embeddings[left.id]
            if experimental_embeddings is not None
            else _as_float_list(left.embedding)
        )
        right_embedding = (
            experimental_embeddings[right.id]
            if experimental_embeddings is not None
            else _as_float_list(right.embedding)
        )
        similarity = cosine_similarity(left_embedding, right_embedding)
        is_shortlisted = similarity >= settings.dedup_shortlist_threshold
        if is_shortlisted:
            shortlisted.append((left, right, similarity))
        pair_rows.append(PairDiagnostic(left.id, right.id, round(similarity, 6), None, is_shortlisted, None))
    diagnostics.shortlisted_pair_count = len(shortlisted)
    diagnostics.batch_count = math.ceil(len(shortlisted) / settings.dedup_cross_encoder_batch_size)
    if not shortlisted:
        diagnostics.pair_diagnostics = pair_rows
        return [[candidate] for candidate in candidates]
    started_at = time.monotonic()
    scorer = scorer or load_cross_encoder(_cross_encoder_config())
    representations = [
        (
            f"Candidate A:\n{_pair_representation(left, diagnostics.representation)}"
            f"\n\nQuestion: {PAIR_QUESTION}",
            f"Candidate B:\n{_pair_representation(right, diagnostics.representation)}",
        )
        for left, right, _ in shortlisted
    ]
    scores = scorer.score_pairs(representations)
    diagnostics.inference_duration_ms = round(max(0.0, (time.monotonic() - started_at) * 1000), 2)
    if len(scores) != len(shortlisted):
        raise CrossEncoderError("Cross-encoder returned an unexpected score count")
    score_by_pair: dict[tuple[int, int], float] = {}
    decisions: dict[tuple[int, int], bool] = {}
    for (left, right, _), score in zip(shortlisted, scores, strict=True):
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise CrossEncoderError("Cross-encoder returned a malformed score")
        key = _pair_key(left, right)
        score_by_pair[key] = score
        decisions[key] = score >= settings.dedup_cross_encoder_merge_threshold
    diagnostics.cross_encoder_scored_pair_count = len(shortlisted)
    diagnostics.pair_diagnostics = [PairDiagnostic(row.left_id, row.right_id, row.bi_encoder_score, round(score_by_pair[(row.left_id, row.right_id)], 6) if (row.left_id, row.right_id) in score_by_pair else None, row.shortlisted, decisions.get((row.left_id, row.right_id))) for row in pair_rows]
    return _complete_link_groups(candidates, decisions)


def _apply_groups(
    groups: list[list[CandidateSignal]],
    diagnostics: DeduplicationDiagnostics,
    *,
    experimental_representative: bool,
) -> None:
    for group in groups:
        has_reason_conclusion = any(
            _is_nearby_reason_conclusion(left, right)
            for index, left in enumerate(group)
            for right in group[index + 1:]
        )
        canonical = max(
            group,
            key=lambda candidate: (
                _is_decision_conclusion(candidate) if has_reason_conclusion else False,
                *(
                    _experimental_canonical_key(candidate)
                    if experimental_representative
                    else _legacy_canonical_key(candidate)
                ),
            ),
        )
        group_id = f"g_{canonical.id}"
        score_source = max(group, key=_legacy_canonical_key)
        if has_reason_conclusion and canonical.score and score_source.score:
            canonical.score.final_score = score_source.score.final_score
            supporting_categories = [
                candidate.category for candidate in group if candidate.id != canonical.id
            ]
            if supporting_categories:
                canonical.rationale = (
                    f"{canonical.rationale.rstrip('.')} Supporting reasons: "
                    f"{'; '.join(dict.fromkeys(supporting_categories))}."
                )
        for candidate in group:
            candidate.duplicate_group_id = group_id
            candidate.is_canonical = candidate.id == canonical.id
            if candidate.id != canonical.id:
                diagnostics.duplicate_relationships.append({
                    "duplicate_candidate_id": candidate.id,
                    "canonical_candidate_id": canonical.id,
                    "group_id": group_id,
                    "reason": "nearby_reason_conclusion" if has_reason_conclusion else "semantic_similarity",
                })
        diagnostics.representatives.append({
            "group_id": group_id,
            "candidate_id": canonical.id,
            "member_candidate_ids": [candidate.id for candidate in group],
            "score_source_candidate_id": score_source.id,
            "reason": "explicit_decision_conclusion" if has_reason_conclusion else "score,evidence_strength,support,false_positive_risk,id",
        })
    diagnostics.clusters = [[candidate.id for candidate in group] for group in groups]


def deduplicate_signals_for_transcript(transcript_id: int, db: Session, *, run_id: str | None = None) -> dict:
    try:
        transcript = db.get(Transcript, transcript_id)
        if transcript is None:
            raise TranscriptNotFoundError("Transcript not found")
        candidate_query = select(CandidateSignal).options(selectinload(CandidateSignal.score)).where(CandidateSignal.transcript_id == transcript_id)
        if run_id is not None:
            candidate_query = candidate_query.where(CandidateSignal.analysis_run_id == run_id)
        candidates = list(db.scalars(candidate_query.order_by(CandidateSignal.id)).all())
    except SQLAlchemyError:
        db.rollback()
        raise SignalDeduplicationError("Signal deduplication could not be completed") from None
    if not candidates:
        raise NoCandidateSignalsError("Transcript has no candidate signals")
    eligible = [candidate for candidate in candidates if candidate.score is not None and validation_allows_business_pipeline(candidate.score.validator_verdict, candidate.score.rejection_reason) and candidate.score.final_score is not None]
    rejected = [candidate for candidate in candidates if candidate.score is not None and not validation_allows_business_pipeline(candidate.score.validator_verdict, candidate.score.rejection_reason)]
    for candidate in rejected:
        candidate.is_canonical = False
        candidate.duplicate_group_id = None
    mode = settings.dedup_experiment_mode
    representation = settings.dedup_cross_encoder_representation
    diagnostics = DeduplicationDiagnostics(transcript_id=transcript_id, mode=mode, model_name=settings.dedup_cross_encoder_model if mode != "existing_embedding_only" else None, model_revision=settings.dedup_cross_encoder_revision, representation=representation, shortlist_threshold=settings.dedup_shortlist_threshold, merge_threshold=settings.dedup_cross_encoder_merge_threshold)
    try:
        for candidate in eligible:
            if candidate.embedding is None:
                candidate.embedding = embed_text(_embedding_text(candidate), model=settings.embedding_model)
        baseline_groups = _embedding_only_groups(eligible, settings.dedup_similarity_threshold)
        if mode == "existing_embedding_only":
            groups = baseline_groups
        elif mode not in EXPERIMENT_MODES or representation not in REPRESENTATIONS:
            diagnostics.failure_type = "InvalidExperimentConfiguration"
            diagnostics.fallback_used = True
            groups = [[candidate] for candidate in eligible]
        else:
            try:
                experimental_embeddings = {
                    candidate.id: embed_text(
                        _pair_representation(candidate, representation),
                        model=settings.embedding_model,
                    )
                    for candidate in eligible
                }
                proposed_groups = _experimental_groups(
                    eligible, diagnostics, experimental_embeddings
                )
                groups = baseline_groups if mode == "cross_encoder_shadow" else proposed_groups
            except (CrossEncoderError, EmbeddingClientError) as error:
                diagnostics.failure_type = type(error).__name__
                diagnostics.fallback_used = True
                if mode == "cross_encoder_shadow" or settings.dedup_cross_encoder_fallback_to_embedding:
                    groups = baseline_groups
                else:
                    groups = [[candidate] for candidate in eligible]
        groups = _merge_reason_conclusion_groups(groups)
        _apply_groups(
            groups,
            diagnostics,
            experimental_representative=(
                mode == "cross_encoder_active" and not diagnostics.fallback_used
            ),
        )
        transcript.status = "signals_deduplicated"
        db.commit()
    except EmbeddingClientError:
        db.rollback()
        raise SignalDeduplicationError("Embedding service is unavailable") from None
    except SQLAlchemyError:
        db.rollback()
        raise SignalDeduplicationError("Signal deduplication could not be saved") from None
    except (TypeError, ValueError):
        db.rollback()
        raise SignalDeduplicationError("Candidate embeddings could not be processed") from None
    finally:
        _store_diagnostics(diagnostics)
    canonical_count = sum(candidate.is_canonical for candidate in eligible)
    duplicate_count = len(eligible) - canonical_count
    logger.info("action=deduplicate_signals transcript_id=%s candidate_count=%s eligible_count=%s canonical_count=%s duplicate_count=%s mode=%s fallback_used=%s success=true", transcript_id, len(candidates), len(eligible), canonical_count, duplicate_count, mode, str(diagnostics.fallback_used).lower())
    return {"transcript_id": transcript_id, "status": "signals_deduplicated", "candidate_count": len(candidates), "eligible_count": len(eligible), "canonical_count": canonical_count, "duplicate_count": duplicate_count, "rejected_excluded": len(rejected)}
