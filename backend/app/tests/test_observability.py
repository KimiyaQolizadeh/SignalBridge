from decimal import Decimal
from types import SimpleNamespace

import pytest

from backend.app.services import pipeline_runner, progress
from backend.app.services.llm_client import extract_response_usage
from backend.app.services.model_pricing import estimate_cost
from backend.app.services.telemetry import (
    finish_collection,
    get_calls,
    record_call,
    start_collection,
)


def test_response_usage_extracts_reasoning_and_cached_tokens() -> None:
    response = SimpleNamespace(usage=SimpleNamespace(
        input_tokens=100,
        output_tokens=30,
        total_tokens=130,
        input_tokens_details=SimpleNamespace(cached_tokens=40),
        output_tokens_details=SimpleNamespace(reasoning_tokens=12),
    ))
    assert extract_response_usage(response) == {
        "input_tokens": 100,
        "output_tokens": 30,
        "reasoning_tokens": 12,
        "cached_input_tokens": 40,
        "total_tokens": 130,
    }


def test_response_usage_handles_missing_fields() -> None:
    assert all(value is None for value in extract_response_usage(object()).values())


def test_known_and_unknown_model_costs() -> None:
    assert estimate_cost(
        "gpt-4.1-mini", input_tokens=1000,
        cached_input_tokens=500, output_tokens=100,
    ) == Decimal("0.00041")
    assert estimate_cost("unknown", input_tokens=1, output_tokens=1) is None


def test_stage_and_run_aggregation_across_calls() -> None:
    token = start_collection()
    try:
        for retries in (0, 1):
            record_call(
                stage="validate_evidence", model="gpt-4.1",
                input_tokens=100, output_tokens=20, reasoning_tokens=5,
                cached_input_tokens=10, total_tokens=120, duration_ms=10,
                success=True, retry_count=retries, error_type=None,
            )
        manifest = pipeline_runner.PipelineExecutionManifest("run", 1)
        manifest.stage_timings = [
            pipeline_runner.PipelineStageTiming("validate_evidence", 25)
        ]
        pipeline_runner._aggregate_usage(manifest)
        stage = manifest.stage_usage[0]
        assert stage.call_count == 2
        assert stage.retry_count == 1
        assert stage.input_tokens == 200
        assert stage.total_tokens == 240
        assert manifest.total_call_count == 2
        assert manifest.total_tokens == 240
        assert manifest.total_estimated_cost == stage.estimated_cost
        assert len(get_calls()) == 2
    finally:
        finish_collection(token)


def test_progress_states_are_safe_and_support_no_results() -> None:
    progress.start(901, "run-901")
    progress.transition(901, "validating_evidence")
    running = progress.get(901)
    assert running["status"] == "processing"
    assert running["current_stage"] == "validating_evidence"
    progress.complete(901, without_results=True)
    assert progress.get(901)["status"] == "completed_without_results"


def test_failed_progress_has_category_without_exception_message() -> None:
    progress.start(902, "run-902")
    progress.fail(902, "scoring_signals", "ScoringError")
    state = progress.get(902)
    assert state["status"] == "failed"
    assert state["error_category"] == "ScoringError"
    assert "traceback" not in str(state).lower()


def test_idle_progress_before_run() -> None:
    state = progress.get(987654)
    assert state["status"] == "idle"
    assert state["current_stage"] == "not_started"
    assert state["run_id"] is None
