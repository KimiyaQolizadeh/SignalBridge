from dataclasses import replace

import pytest

from backend.app.config import settings
from backend.app.llm_schemas import (
    ImportanceItemOutput, ImportanceListwiseOutput,
    ImportancePairwiseItemOutput, ImportancePairwiseOutput,
)
from backend.app.services import importance_estimator as estimator
from backend.app.services.importance_evaluation import (
    baseline_rankings, build_importance_dataset, kendall, ndcg,
    ranking_metrics, run_baselines, spearman, validate_dataset, write_report,
)


def candidate(candidate_id: str, *, transcript: str = "t1", verdict: str = "pass", item_type: str = "driver", traceable: bool = True) -> estimator.ImportanceCandidate:
    return estimator.ImportanceCandidate(candidate_id, transcript, item_type, "category", "Synthetic advisor evidence.", "Synthetic rationale.", verdict, "explicit", int(candidate_id[-1]), traceable=traceable, business_score=4.0, support_score=0.9)


def output(candidate_id: str, rank: int, *, tier: str = "high", dependency: str = "outcome_driver", attention: str = "developed") -> ImportanceItemOutput:
    return ImportanceItemOutput(candidate_id=candidate_id, importance_tier=tier, importance_score=0.7, relative_rank=rank, decision_centrality="major", priority_expression="explicitly_emphasized", dependency_role=dependency, consequence_strength="materially_changes_decision", attention_strength=attention, importance_confidence=0.9, importance_basis="Advisor-grounded importance.", comparison_basis="Stronger consequence than peers.")


def pair(a: str, b: str, relation: str = "a_more_important") -> ImportancePairwiseItemOutput:
    return ImportancePairwiseItemOutput(candidate_a_id=a, candidate_b_id=b, relative_importance=relation, comparison_confidence=0.9, comparison_basis="stronger_consequence", reason="Greater advisor decision consequence.")


def test_default_mode_is_disabled() -> None:
    assert settings.importance_mode == "importance_disabled"
    assert estimator.run_importance([candidate("c1")]).results == []


def test_eligibility_and_transcript_isolation() -> None:
    items = [candidate("c1"), candidate("c2", verdict="needs_review"), candidate("c3", verdict="reject"), candidate("c4", traceable=False)]
    assert [item.candidate_id for item in estimator.eligible_candidates(items)] == ["c1", "c2"]
    with pytest.raises(ValueError, match="one transcript"):
        estimator.eligible_candidates([candidate("c1"), candidate("c2", transcript="t2")])


def test_structured_score_is_deterministic_and_ignores_frequency_category_type() -> None:
    item = output("c1", 1)
    assert estimator.structured_importance_score(item) == estimator.structured_importance_score(item)
    assert estimator.structured_importance_score(item) == estimator.structured_importance_score(item.model_copy(update={"importance_basis": "Different prose."}))


def test_brief_gate_scores_above_verbose_preference() -> None:
    gate = output("c1", 1, tier="critical", dependency="gating_condition", attention="brief").model_copy(update={"decision_centrality":"decisive","priority_expression":"explicitly_required","consequence_strength":"decision_stopping"})
    preference = output("c2", 2, tier="low", dependency="comparative_preference", attention="dominant").model_copy(update={"decision_centrality":"minor","priority_expression":"mentioned_once","consequence_strength":"limited_effect"})
    assert estimator.structured_importance_score(gate) > estimator.structured_importance_score(preference)


def test_listwise_exactly_one_output_per_candidate() -> None:
    items = [candidate("c1"), candidate("c2")]
    result = estimator.validate_listwise_output(ImportanceListwiseOutput(items=[output("c1",1), output("c2",2)]), items)
    assert [item.candidate_id for item in result] == ["c1", "c2"]


@pytest.mark.parametrize("items", [[output("c1",1)], [output("c1",1), output("c3",2)], [output("c1",1), output("c1",2)]])
def test_missing_unknown_and_duplicate_ids_fail(items) -> None:
    with pytest.raises(estimator.ImportanceOutputError):
        estimator.validate_listwise_output(ImportanceListwiseOutput(items=items), [candidate("c1"), candidate("c2")])


def test_duplicate_or_noncontiguous_ranks_fail() -> None:
    with pytest.raises(estimator.ImportanceOutputError, match="ranks"):
        estimator.validate_listwise_output(ImportanceListwiseOutput(items=[output("c1",1), output("c2",1)]), [candidate("c1"),candidate("c2")])


def test_tier_dimension_conflicts_are_visible() -> None:
    bad = output("c1",1,tier="critical",attention="incidental").model_copy(update={"decision_centrality":"contextual","consequence_strength":"no_stated_effect"})
    result=estimator.validate_listwise_output(ImportanceListwiseOutput(items=[bad]),[candidate("c1")])[0]
    assert "critical_dimension_conflict" in result.warnings


def test_needs_review_status_remains_visible() -> None:
    result=estimator.validate_listwise_output(ImportanceListwiseOutput(items=[output("c1",1)]),[candidate("c1",verdict="needs_review")])[0]
    assert result.validation_verdict == "needs_review"
    assert "unresolved_validation" in result.warnings


def test_related_ids_preserve_broad_and_narrow_candidates() -> None:
    item=output("c1",1).model_copy(update={"supporting_candidate_ids":["c2"],"importance_warnings":["possible_broad_narrow_overlap"]})
    results=estimator.validate_listwise_output(ImportanceListwiseOutput(items=[item,output("c2",2)]),[candidate("c1"),candidate("c2")])
    assert len(results)==2 and "possible_broad_narrow_overlap" in results[0].warnings


def test_pairwise_requires_every_pair() -> None:
    candidates=[candidate("c1"),candidate("c2"),candidate("c3")]
    with pytest.raises(estimator.ImportanceOutputError,match="missing"):
        estimator.validate_pairwise_output(ImportancePairwiseOutput(items=[pair("c1","c2")]),candidates)


def test_pairwise_aggregation_and_ties_are_deterministic() -> None:
    items=[pair("c1","c2"),pair("c1","c3"),pair("c2","c3","approximately_equal")]
    assert estimator.aggregate_pairwise(items,["c1","c2","c3"])[0][0]=="c1"
    assert estimator.aggregate_pairwise(items,["c1","c2","c3"])[1][0]=="c2"


def test_pairwise_cycle_detection() -> None:
    items=[pair("c1","c2"),pair("c2","c3"),pair("c3","c1")]
    assert estimator.pairwise_cycles(items)==[("c1","c2","c3")]


def test_disabled_mode_does_not_call_model(monkeypatch) -> None:
    monkeypatch.setattr(estimator,"call_llm_json",lambda **_:pytest.fail("disabled mode called model"))
    assert estimator.run_importance([candidate("c1"),candidate("c2")],"importance_disabled").results==[]


def test_shadow_failure_preserves_candidates(monkeypatch) -> None:
    original=[candidate("c1"),candidate("c2")]
    monkeypatch.setattr(estimator,"call_llm_json",lambda **_:(_ for _ in ()).throw(estimator.LLMClientError("safe")))
    result=estimator.run_importance(original,"importance_shadow_listwise")
    assert result.used_fallback and result.results==[] and original==[candidate("c1"),candidate("c2")]


def test_representative_context_only_adds_warning() -> None:
    result=estimator.validate_listwise_output(ImportanceListwiseOutput(items=[output("c1",1)]),[replace(candidate("c1"),representative_context_present=True)])[0]
    assert "representative_context_present" in result.warnings


def test_dataset_split_size_and_stable_ids() -> None:
    groups=build_importance_dataset(); summary=validate_dataset(groups)
    assert summary=={"groups":40,"candidates":240,"eligible":200,"development":28,"holdout":12}
    assert [g.group_id for g in groups]==[g.group_id for g in build_importance_dataset()]


def test_baselines_and_metrics_are_reproducible() -> None:
    groups=build_importance_dataset(); assert run_baselines(groups)==run_baselines(groups)
    assert spearman([1,2,3],[1,2,3])==pytest.approx(1)
    assert kendall([1,2,3],[1,2,3])==pytest.approx(1)
    assert ndcg([1,2,3],[0,1,2])==pytest.approx(1)


def test_frequency_and_evidence_length_baselines_do_not_win_by_construction() -> None:
    metrics=run_baselines(build_importance_dataset())
    assert metrics["mention_frequency"]["top1"] < metrics["structured_rule"]["top1"]
    assert metrics["evidence_length"]["top1"] < metrics["structured_rule"]["top1"]


def test_no_public_or_persisted_fields_are_added() -> None:
    from backend.app.schemas import PipelineRunResponse
    assert set(PipelineRunResponse.model_fields)=={"transcript_id","status","steps","final_driver_count","final_blocker_count"}


def test_report_generation_is_reproducible_and_contains_no_evidence(tmp_path) -> None:
    result={"mode":"baselines","baselines":run_baselines(build_importance_dataset())}
    first=tmp_path/"first.md"; second=tmp_path/"second.md"
    write_report(result,first); write_report(result,second)
    assert first.read_bytes()==second.read_bytes()
    assert "We cannot proceed unless" not in first.read_text(encoding="utf-8")


def test_disabled_mode_logs_no_confidential_evidence(caplog) -> None:
    secret="PRIVATE_SYNTHETIC_EVIDENCE_MARKER"
    item=replace(candidate("c1"),advisor_evidence=secret)
    estimator.run_importance([item],"importance_disabled")
    assert secret not in caplog.text
