"""Experimental transcript-relative importance estimation; production-disabled."""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from itertools import combinations
from typing import Any, Iterable, cast

from ..config import settings
from ..llm_schemas import (
    ImportanceItemOutput, ImportanceListwiseOutput,
    ImportancePairwiseItemOutput, ImportancePairwiseOutput,
)
from .llm_client import LLMClientError, call_llm_json
from .prompt_loader import prompt_sha256


IMPORTANCE_MODES = {
    "importance_disabled", "importance_shadow_listwise",
    "importance_shadow_pairwise", "importance_shadow_hybrid",
}
TIER_ORDER = {"critical": 5, "high": 4, "medium": 3, "low": 2, "peripheral": 1, "indeterminate": 0}
CENTRALITY = {"decisive": 4, "major": 3, "contributing": 2, "minor": 1, "contextual": 0, "indeterminate": 0}
DEPENDENCY = {"gating_condition": 4, "timing_condition": 3, "outcome_driver": 3, "comparative_preference": 2, "supporting_detail": 1, "no_dependency": 0, "indeterminate": 0}
CONSEQUENCE = {"decision_stopping": 4, "materially_changes_decision": 3, "affects_timing_or_terms": 2, "influences_preference": 1, "limited_effect": 0.5, "no_stated_effect": 0, "indeterminate": 0}
PRIORITY = {"explicitly_ranked": 4, "explicitly_required": 4, "explicitly_emphasized": 3, "repeatedly_emphasized": 3, "implicitly_central": 2, "mentioned_once": 1, "unclear": 0}
ATTENTION = {"dominant": 3, "repeated": 2.5, "developed": 2, "brief": 1, "incidental": 0, "indeterminate": 0}
MAX_STRUCTURED_SCORE = 19.0
IMPORTANCE_DIAGNOSTICS: dict[str, dict[str, Any]] = {}


@dataclass(frozen=True)
class ImportanceCandidate:
    candidate_id: str
    transcript_id: str
    item_type: str
    category: str
    advisor_evidence: str
    rationale: str
    validation_verdict: str
    evidence_strength: str
    transcript_order: int
    traceable: bool = True
    current: bool = True
    decision_direction: str | None = None
    bounded_advisor_context: str | None = None
    business_score: float | None = None
    support_score: float | None = None
    mention_count: int = 1
    representative_context_present: bool = False


@dataclass(frozen=True)
class ExperimentalImportanceResult:
    candidate_id: str
    validation_verdict: str
    importance_tier: str
    direct_model_score: float
    structured_score: float
    score_difference: float
    relative_rank: int
    decision_centrality: str
    priority_expression: str
    dependency_role: str
    consequence_strength: str
    attention_strength: str
    importance_confidence: float
    importance_basis: str
    comparison_basis: str
    supporting_candidate_ids: tuple[str, ...]
    conflicting_candidate_ids: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass
class ImportanceRunResult:
    mode: str
    transcript_id: str
    results: list[ExperimentalImportanceResult] = field(default_factory=list)
    pairwise: list[ImportancePairwiseItemOutput] = field(default_factory=list)
    pairwise_cycles: list[tuple[str, str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    used_fallback: bool = False
    malformed_output: bool = False


class ImportanceOutputError(ValueError):
    pass


def eligible_candidates(candidates: Iterable[ImportanceCandidate]) -> list[ImportanceCandidate]:
    items = list(candidates)
    transcript_ids = {item.transcript_id for item in items}
    if len(transcript_ids) > 1:
        raise ValueError("Importance candidates must belong to one transcript")
    return [item for item in items if item.current and item.traceable and item.item_type in {"driver", "blocker"} and item.validation_verdict in {"pass", "needs_review"}]


def structured_importance_score(item: ImportanceItemOutput) -> float:
    value = CENTRALITY[item.decision_centrality] + DEPENDENCY[item.dependency_role] + CONSEQUENCE[item.consequence_strength] + PRIORITY[item.priority_expression] + ATTENTION[item.attention_strength]
    return round(value / MAX_STRUCTURED_SCORE, 4)


def tier_for_score(score: float) -> str:
    if score >= 0.80: return "critical"
    if score >= 0.65: return "high"
    if score >= 0.45: return "medium"
    if score >= 0.25: return "low"
    return "peripheral"


def _consistency_warnings(item: ImportanceItemOutput, candidate: ImportanceCandidate) -> list[str]:
    warnings = list(item.importance_warnings)
    structured = structured_importance_score(item)
    if item.importance_tier == "critical" and (item.consequence_strength == "no_stated_effect" or item.attention_strength == "incidental" or item.decision_centrality == "contextual"):
        warnings.append("critical_dimension_conflict")
    if item.importance_tier == "peripheral" and (item.dependency_role == "gating_condition" or item.consequence_strength == "decision_stopping" or item.decision_centrality == "decisive"):
        warnings.append("peripheral_dimension_conflict")
    if item.importance_tier != "indeterminate" and abs(TIER_ORDER[item.importance_tier] - TIER_ORDER[tier_for_score(structured)]) > 1:
        warnings.append("tier_score_inconsistency")
    if candidate.validation_verdict == "needs_review": warnings.append("unresolved_validation")
    if candidate.representative_context_present: warnings.append("representative_context_present")
    return list(dict.fromkeys(warnings))


def validate_listwise_output(output: ImportanceListwiseOutput, candidates: list[ImportanceCandidate]) -> list[ExperimentalImportanceResult]:
    expected = {item.candidate_id for item in candidates}
    ids = [item.candidate_id for item in output.items]
    if len(ids) != len(set(ids)): raise ImportanceOutputError("duplicate candidate result")
    if set(ids) != expected: raise ImportanceOutputError("missing or unknown candidate ID")
    ranks = [item.relative_rank for item in output.items]
    if len(ranks) != len(set(ranks)) or sorted(ranks) != list(range(1, len(ranks) + 1)):
        raise ImportanceOutputError("relative ranks must be unique and contiguous")
    by_id = {item.candidate_id: item for item in candidates}
    results = []
    for item in output.items:
        if not math.isfinite(item.importance_score): raise ImportanceOutputError("non-finite score")
        related = set(item.supporting_candidate_ids + item.conflicting_candidate_ids)
        if not related <= expected or item.candidate_id in related: raise ImportanceOutputError("invalid related candidate ID")
        candidate = by_id[item.candidate_id]
        structured = structured_importance_score(item)
        warnings = _consistency_warnings(item, candidate)
        results.append(ExperimentalImportanceResult(
            candidate_id=item.candidate_id, validation_verdict=candidate.validation_verdict,
            importance_tier=item.importance_tier, direct_model_score=item.importance_score,
            structured_score=structured, score_difference=round(abs(item.importance_score - structured), 4),
            relative_rank=item.relative_rank, decision_centrality=item.decision_centrality,
            priority_expression=item.priority_expression, dependency_role=item.dependency_role,
            consequence_strength=item.consequence_strength, attention_strength=item.attention_strength,
            importance_confidence=item.importance_confidence, importance_basis=item.importance_basis,
            comparison_basis=item.comparison_basis, supporting_candidate_ids=tuple(item.supporting_candidate_ids),
            conflicting_candidate_ids=tuple(item.conflicting_candidate_ids), warnings=tuple(warnings),
        ))
    return sorted(results, key=lambda item: item.relative_rank)


def validate_pairwise_output(output: ImportancePairwiseOutput, candidates: list[ImportanceCandidate]) -> list[ImportancePairwiseItemOutput]:
    ids = {item.candidate_id for item in candidates}
    expected = {tuple(sorted((a.candidate_id, b.candidate_id))) for a, b in combinations(candidates, 2)}
    seen: set[tuple[str, str]] = set()
    for item in output.items:
        pair = tuple(sorted((item.candidate_a_id, item.candidate_b_id)))
        if item.candidate_a_id not in ids or item.candidate_b_id not in ids or item.candidate_a_id == item.candidate_b_id: raise ImportanceOutputError("unknown pairwise candidate ID")
        if pair in seen: raise ImportanceOutputError("duplicate pairwise result")
        seen.add(pair)
    if seen != expected: raise ImportanceOutputError("missing pairwise result")
    return output.items


def pairwise_cycles(items: list[ImportancePairwiseItemOutput]) -> list[tuple[str, str, str]]:
    edges: set[tuple[str, str]] = set()
    nodes: set[str] = set()
    for item in items:
        nodes.update((item.candidate_a_id, item.candidate_b_id))
        if item.relative_importance == "a_more_important": edges.add((item.candidate_a_id, item.candidate_b_id))
        elif item.relative_importance == "b_more_important": edges.add((item.candidate_b_id, item.candidate_a_id))
    cycles = set()
    for a, b, c in combinations(sorted(nodes), 3):
        if {(a, b), (b, c), (c, a)} <= edges or {(b, a), (c, b), (a, c)} <= edges:
            cycles.add((a, b, c))
    return sorted(cycles)


def aggregate_pairwise(items: list[ImportancePairwiseItemOutput], candidate_ids: list[str]) -> list[tuple[str, int, float]]:
    wins = Counter({candidate_id: 0.0 for candidate_id in candidate_ids})
    for item in items:
        if item.relative_importance == "a_more_important": wins[item.candidate_a_id] += 1
        elif item.relative_importance == "b_more_important": wins[item.candidate_b_id] += 1
        elif item.relative_importance == "approximately_equal":
            wins[item.candidate_a_id] += 0.5; wins[item.candidate_b_id] += 0.5
    ordered = sorted(candidate_ids, key=lambda candidate_id: (-wins[candidate_id], candidate_id))
    return [(candidate_id, rank, wins[candidate_id]) for rank, candidate_id in enumerate(ordered, 1)]


def _candidate_payload(item: ImportanceCandidate) -> dict[str, Any]:
    return {"candidate_id": item.candidate_id, "item_type": item.item_type, "category": item.category,
            "decision_direction": item.decision_direction, "validation_verdict": item.validation_verdict,
            "validated_evidence_strength": item.evidence_strength, "advisor_evidence": item.advisor_evidence,
            "bounded_advisor_context": item.bounded_advisor_context, "validated_rationale": item.rationale,
            "transcript_order": item.transcript_order, "mention_count": item.mention_count,
            "representative_context_present": item.representative_context_present}


def run_importance(candidates: list[ImportanceCandidate], mode: str | None = None) -> ImportanceRunResult:
    selected_mode = mode or settings.importance_mode
    if selected_mode not in IMPORTANCE_MODES: raise ValueError("Unknown importance mode")
    transcript_id = candidates[0].transcript_id if candidates else "unknown"
    run = ImportanceRunResult(selected_mode, transcript_id)
    if selected_mode == "importance_disabled": return run
    eligible = eligible_candidates(candidates)
    if len(eligible) < 2:
        run.warnings.append("insufficient_comparison_set"); return run
    started = time.monotonic()
    try:
        if selected_mode in {"importance_shadow_listwise", "importance_shadow_hybrid"}:
            output = cast(ImportanceListwiseOutput, call_llm_json(
                prompt_file_name="06_importance_listwise_experiment.txt",
                input_payload={"transcript_id": transcript_id, "eligible_candidates": [_candidate_payload(item) for item in eligible]},
                response_model=ImportanceListwiseOutput, model=settings.importance_model,
                temperature=0, transcript_id=None))
            run.results = validate_listwise_output(output, eligible)
        if selected_mode in {"importance_shadow_pairwise", "importance_shadow_hybrid"}:
            pairs = [{"candidate_a": _candidate_payload(a), "candidate_b": _candidate_payload(b)} for a, b in combinations(eligible, 2)]
            output = cast(ImportancePairwiseOutput, call_llm_json(
                prompt_file_name="07_importance_pairwise_experiment.txt",
                input_payload={"transcript_id": transcript_id, "pairs": pairs},
                response_model=ImportancePairwiseOutput, model=settings.importance_model,
                temperature=0, transcript_id=None))
            run.pairwise = validate_pairwise_output(output, eligible)
            run.pairwise_cycles = pairwise_cycles(run.pairwise)
            if run.pairwise_cycles: run.warnings.append("pairwise_cycle")
            if selected_mode == "importance_shadow_pairwise":
                aggregate = aggregate_pairwise(run.pairwise, [item.candidate_id for item in eligible])
                run.results = [ExperimentalImportanceResult(candidate_id, next(x.validation_verdict for x in eligible if x.candidate_id == candidate_id), "indeterminate", 0.0, 0.0, 0.0, rank, "indeterminate", "unclear", "indeterminate", "indeterminate", "indeterminate", 0.0, "Pairwise aggregate only.", "Copeland-style pairwise wins.", (), (), ("pairwise_only",)) for candidate_id, rank, _ in aggregate]
            elif run.results:
                pair_ranks = {candidate_id: rank for candidate_id, rank, _ in aggregate_pairwise(run.pairwise, [item.candidate_id for item in eligible])}
                if any(pair_ranks[item.candidate_id] != item.relative_rank for item in run.results): run.warnings.append("listwise_pairwise_disagreement")
    except (LLMClientError, ImportanceOutputError, ValueError):
        run.results = []; run.pairwise = []; run.used_fallback = True; run.malformed_output = True
    IMPORTANCE_DIAGNOSTICS[transcript_id] = {
        "mode": selected_mode, "transcript_id": transcript_id, "eligible_candidate_count": len(eligible),
        "candidate_ids": [item.candidate_id for item in eligible], "model": settings.importance_model,
        "duration_ms": round((time.monotonic() - started) * 1000, 2),
        "prompt_hashes": {"listwise": prompt_sha256("06_importance_listwise_experiment.txt"), "pairwise": prompt_sha256("07_importance_pairwise_experiment.txt")},
        "results": [asdict(item) for item in run.results], "pairwise": [item.model_dump() for item in run.pairwise],
        "cycles": run.pairwise_cycles, "warnings": run.warnings, "fallback": run.used_fallback,
        "malformed_output": run.malformed_output,
    }
    return run
