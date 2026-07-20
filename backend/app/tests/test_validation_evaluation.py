import json

import pytest

from backend.app.llm_schemas import EvidenceValidationOutput
from backend.app.services import validation_evaluation as evaluation
from backend.app.services.prompt_loader import load_prompt, prompt_sha256


def test_dataset_schema_size_distribution_and_stable_ids() -> None:
    dataset = evaluation.build_dataset()
    summary = evaluation.validate_dataset(dataset)
    assert summary["count"] == 120
    assert dict(summary["split"]) == {"development": 80, "holdout": 40}
    assert dict(summary["verdict"]) == {"pass": 35, "needs_review": 30, "reject": 55}
    assert [item.example_id for item in dataset] == [item.example_id for item in evaluation.build_dataset()]


def test_development_and_holdout_do_not_overlap() -> None:
    dataset = evaluation.build_dataset()
    development = {item.example_id for item in dataset if item.split == "development"}
    holdout = {item.example_id for item in dataset if item.split == "holdout"}
    assert development.isdisjoint(holdout)


def test_every_expert_label_has_all_structured_fields() -> None:
    required = set(evaluation.FIELDS + evaluation.LIST_FIELDS + ("expected_verdict",))
    assert all(required <= set(item.expert) for item in evaluation.build_dataset())


def test_oracle_verdict_derivation_is_exact() -> None:
    result = evaluation.oracle_report(evaluation.build_dataset())
    assert result["accuracy"] == 1
    assert result["macro_f1"] == 1


def test_categorical_metrics_include_confusion_and_missing_output() -> None:
    result = evaluation.categorical_metrics(["pass", "reject", "reject"], ["pass", "pass", None])
    assert result["accuracy"] == 1 / 3
    assert result["missing_rate"] == 1 / 3
    assert result["confusion_matrix"]["reject"]["__missing__"] == 1


def test_list_metrics_count_false_and_missed_detections() -> None:
    result = evaluation.list_metrics([["pain_to_move"], []], [["question_to_blocker"], ["pain_to_move"]])
    assert result["exact_match"] == 0
    assert result["false_detections"] == 2
    assert result["missed_detections"] == 1


def test_precheck_attribution_has_no_false_rejects() -> None:
    result = evaluation.precheck_report(evaluation.build_dataset())
    assert result["resolved_count"] == 40
    assert result["hard_reject_precision"] == 1
    assert result["false_rejects"] == []


def test_live_runner_is_opt_in_mockable_and_does_not_mutate_candidates(tmp_path, monkeypatch) -> None:
    dataset = evaluation.build_dataset()
    before = dataset
    expert = dataset[0].expert
    monkeypatch.setattr(evaluation, "call_llm_json", lambda **_: EvidenceValidationOutput(**expert))
    document = evaluation.run_live(dataset, "holdout", tmp_path / "live.json")
    assert len(document["results"]) == 40
    assert dataset == before


def test_malformed_output_is_accounted_for_conservatively(tmp_path, monkeypatch) -> None:
    dataset = evaluation.build_dataset()
    monkeypatch.setattr(evaluation, "call_llm_json", lambda **_: (_ for _ in ()).throw(evaluation.LLMClientError("safe failure")))
    document = evaluation.run_live(dataset, "holdout", tmp_path / "live.json")
    report = evaluation.analyze_live(dataset, document)
    assert report["malformed_count"] == 40
    assert all(row["attempts"][0]["derived_verdict"] == "reject" for row in document["results"])


def test_clamping_difference_analysis_counts_adjustments(tmp_path, monkeypatch) -> None:
    dataset = evaluation.build_dataset()
    expert = dict(dataset[0].expert, support_score=0.1, advisor_side_score=0.1, false_positive_risk=0.9)
    monkeypatch.setattr(evaluation, "call_llm_json", lambda **_: EvidenceValidationOutput(**expert))
    report = evaluation.analyze_live(dataset, evaluation.run_live(dataset, "holdout", tmp_path / "live.json"))
    assert report["clamping"]["values_changed"] > 0
    assert report["clamping"]["max_adjustment"] > 0


def test_report_generation_is_reproducible(tmp_path) -> None:
    report = {"count": 1, "malformed_count": 0, "verdict": {"accuracy": 1, "macro_f1": 1}, "fields": {}, "errors": []}
    first, second = tmp_path / "first.md", tmp_path / "second.md"
    evaluation.write_markdown(report, first); evaluation.write_markdown(report, second)
    assert first.read_bytes() == second.read_bytes()


def test_live_result_contains_no_candidate_evidence(tmp_path, monkeypatch) -> None:
    dataset = evaluation.build_dataset()
    secret = dataset[-1].advisor_quote
    monkeypatch.setattr(evaluation, "call_llm_json", lambda **_: EvidenceValidationOutput(**dataset[-1].expert))
    evaluation.run_live(dataset, "holdout", tmp_path / "live.json")
    assert secret not in (tmp_path / "live.json").read_text(encoding="utf-8")


def test_experimental_prompt_loads_and_production_prompt_is_versioned() -> None:
    experiment = load_prompt("03_evidence_validator_review_experiment.txt")
    assert "MANDATORY EVIDENCE-STRENGTH LADDER" in experiment
    assert "Weak evidence alone is not escalation" in experiment
    assert "partially_grounded" in experiment
    assert "incomplete" in experiment and "irrelevant" in experiment
    assert prompt_sha256("03_evidence_validator.txt") == "9d61219e638e48d6495abda5ffdcd3294f884239271fe258ef2910cb47d0adb9"
    assert prompt_sha256("03_evidence_validator_review_experiment.txt") != prompt_sha256("03_evidence_validator.txt")


def test_experimental_prompt_is_development_only(tmp_path) -> None:
    with pytest.raises(ValueError, match="development split"):
        evaluation.run_live(
            evaluation.build_dataset(), "holdout", tmp_path / "forbidden.json",
            prompt_file_name="03_evidence_validator_review_experiment.txt",
        )
    assert not (tmp_path / "forbidden.json").exists()


def test_experimental_prompt_metadata_is_reported(tmp_path, monkeypatch) -> None:
    dataset = evaluation.build_dataset()
    monkeypatch.setattr(evaluation, "call_llm_json", lambda **_: EvidenceValidationOutput(**dataset[0].expert))
    document = evaluation.run_live(
        dataset, "development", tmp_path / "experiment.json", limit=1,
        prompt_file_name="03_evidence_validator_review_experiment.txt",
    )
    assert document["prompt_version"] == "validation_2.0_review_experiment_v1"
    assert document["prompt_file_name"] == "03_evidence_validator_review_experiment.txt"
    assert len(document["prompt_sha256"]) == 64


def test_stability_limit_selects_four_examples_per_verdict(tmp_path, monkeypatch) -> None:
    dataset = evaluation.build_dataset()
    labels = {item.example_id: item.expert["expected_verdict"] for item in dataset}
    monkeypatch.setattr(evaluation, "call_llm_json", lambda **_: EvidenceValidationOutput(**dataset[0].expert))
    document = evaluation.run_live(
        dataset, "development", tmp_path / "stability.json", limit=12,
        prompt_file_name="03_evidence_validator_review_experiment.txt",
    )
    selected = [labels[row["example_id"]] for row in document["results"]]
    assert selected.count("pass") == 4
    assert selected.count("needs_review") == 4
    assert selected.count("reject") == 4
