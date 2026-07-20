"""Deterministic guards for the production/research boundary."""

from __future__ import annotations

import ast
import builtins
import importlib
import inspect
from pathlib import Path

import pytest

from backend.app import config
from backend.app.services import evidence_validator, importance_estimator
from backend.app.services.eligibility_policy import (
    DEDUP_ELIGIBLE_VERDICTS,
    FINAL_RANKING_ELIGIBLE_VERDICTS,
    SCORING_ELIGIBLE_VERDICTS,
)


APP_ROOT = Path(__file__).resolve().parents[1]
NON_PRODUCTION_MODULES = {
    "validation_evaluation.py",
    "importance_evaluation.py",
    "importance_estimator.py",
    "dedup_evaluation.py",
    "cross_encoder.py",
}
FORBIDDEN_PARTS = {
    "experiments",
    "validation_evaluation",
    "importance_evaluation",
    "importance_estimator",
    "dedup_evaluation",
    "evaluate_dedup_cross_encoder",
    "benchmark",
}
# Remove this one exact exception when cross-encoder research is extracted from
# the production deduplicator during the architectural-separation phase.
APPROVED_IMPORT_EXCEPTIONS = {
    (
        "backend.app.services.signal_deduplicator",
        "backend.app.services.cross_encoder",
    )
}


def _module_name(path: Path) -> str:
    return ".".join(("backend", "app", *path.relative_to(APP_ROOT).with_suffix("").parts))


def _resolved_imports(path: Path) -> set[str]:
    source_module = _module_name(path)
    package = source_module.rpartition(".")[0]
    imports: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                relative = "." * node.level + (node.module or "")
                imports.add(importlib.util.resolve_name(relative, package))
            elif node.module:
                imports.add(node.module)
    return imports


def test_production_modules_do_not_import_research_modules() -> None:
    violations: list[tuple[str, str]] = []
    for path in APP_ROOT.rglob("*.py"):
        if "tests" in path.parts or path.name in NON_PRODUCTION_MODULES:
            continue
        source = _module_name(path)
        for target in _resolved_imports(path):
            pair = (source, target)
            target_parts = set(target.split("."))
            if pair not in APPROVED_IMPORT_EXCEPTIONS and target_parts & FORBIDDEN_PARTS:
                violations.append(pair)
    assert violations == []


def test_cross_encoder_exception_is_exact_and_still_present() -> None:
    deduplicator = APP_ROOT / "services" / "signal_deduplicator.py"
    imports = _resolved_imports(deduplicator)
    assert APPROVED_IMPORT_EXCEPTIONS == {
        ("backend.app.services.signal_deduplicator", "backend.app.services.cross_encoder")
    }
    assert "backend.app.services.cross_encoder" in imports


def test_pipeline_runner_does_not_import_importance_estimation() -> None:
    imports = _resolved_imports(APP_ROOT / "services" / "pipeline_runner.py")
    assert not any("importance" in imported for imported in imports)


def test_startup_does_not_invoke_importance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        importance_estimator,
        "run_importance",
        lambda *_args, **_kwargs: pytest.fail("startup invoked importance estimation"),
    )
    import backend.app.main as main

    importlib.reload(main)


def test_default_experiments_are_inactive() -> None:
    settings = config.Settings(_env_file=None)
    assert settings.experiments.importance_mode == "importance_disabled"
    assert settings.experiments.dedup_mode == "existing_embedding_only"


def test_constructing_settings_does_not_load_optional_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith(("sentence_transformers", "torch")):
            pytest.fail(f"configuration loaded optional model dependency: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    settings = config.Settings(_env_file=None)
    assert settings.dedup_experiment_mode == "existing_embedding_only"


def test_production_validator_uses_only_production_prompt() -> None:
    source = inspect.getsource(evidence_validator)
    assert 'prompt_file_name="03_evidence_validator.txt"' in source
    assert "03_evidence_validator_review_experiment.txt" not in source


def test_verdict_policies_are_immutable_and_intentionally_different() -> None:
    assert SCORING_ELIGIBLE_VERDICTS == frozenset({"pass", "needs_review"})
    assert DEDUP_ELIGIBLE_VERDICTS == frozenset({"pass", "needs_review"})
    assert FINAL_RANKING_ELIGIBLE_VERDICTS == frozenset({"pass", "needs_review"})
    assert "reject" not in FINAL_RANKING_ELIGIBLE_VERDICTS
