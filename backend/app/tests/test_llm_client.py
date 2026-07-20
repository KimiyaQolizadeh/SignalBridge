import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from backend.app.services import llm_client
from backend.app.services.llm_client import (
    LLMClientError,
    call_llm_json,
    safe_json_dumps,
)
from backend.app.services.pipeline_context import pipeline_run_id


class MockResponseModel(BaseModel):
    value: int


class FakeResponses:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text
        self.calls: list[dict] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=self.output_text)


def test_missing_api_key_raises_safe_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_client.settings, "openai_api_key", None)

    with pytest.raises(LLMClientError, match="OPENAI_API_KEY is not configured"):
        call_llm_json(
            prompt_file_name="test.txt",
            input_payload={"value": 1},
            response_model=MockResponseModel,
            model="test-model",
        )


def test_safe_json_dumps_serializes_simple_dict() -> None:
    serialized = safe_json_dumps({"name": "SignalBridge", "count": 2})

    assert json.loads(serialized) == {"name": "SignalBridge", "count": 2}


def test_schema_validation_failure_is_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_responses = FakeResponses('{"value": "not-an-integer"}')
    fake_client = SimpleNamespace(responses=fake_responses)
    monkeypatch.setattr(llm_client.settings, "openai_api_key", "test-key")
    monkeypatch.setattr(llm_client, "load_prompt", lambda _: "Return JSON.")
    monkeypatch.setattr(llm_client, "_create_client", lambda _: fake_client)

    with pytest.raises(LLMClientError, match="schema validation"):
        call_llm_json(
            prompt_file_name="test.txt",
            input_payload={"private": "payload"},
            response_model=MockResponseModel,
            model="test-model",
        )

    assert len(fake_responses.calls) == 1


def test_provided_prompt_file_name_is_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded_names: list[str] = []
    fake_responses = FakeResponses('{"value": 7}')
    fake_client = SimpleNamespace(responses=fake_responses)

    def fake_load_prompt(prompt_file_name: str) -> str:
        loaded_names.append(prompt_file_name)
        return "Return JSON."

    monkeypatch.setattr(llm_client.settings, "openai_api_key", "test-key")
    monkeypatch.setattr(llm_client, "load_prompt", fake_load_prompt)
    monkeypatch.setattr(llm_client, "_create_client", lambda _: fake_client)

    result = call_llm_json(
        prompt_file_name="02_candidate_signal_extractor.txt",
        input_payload={"turns": []},
        response_model=MockResponseModel,
        model="test-model",
    )

    assert loaded_names == ["02_candidate_signal_extractor.txt"]
    assert result == MockResponseModel(value=7)
    assert fake_responses.calls[0]["store"] is False


def test_llm_logs_pipeline_run_correlation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log_messages: list[str] = []
    fake_client = SimpleNamespace(responses=FakeResponses('{"value": 7}'))
    monkeypatch.setattr(llm_client.settings, "openai_api_key", "test-key")
    monkeypatch.setattr(llm_client, "load_prompt", lambda _: "Return JSON.")
    monkeypatch.setattr(llm_client, "_create_client", lambda _: fake_client)
    monkeypatch.setattr(
        llm_client.logger,
        "info",
        lambda message, *args: log_messages.append(message % args),
    )
    token = pipeline_run_id.set("run-correlation-id")
    try:
        call_llm_json(
            prompt_file_name="test.txt",
            input_payload={"value": 1},
            response_model=MockResponseModel,
            model="test-model",
        )
    finally:
        pipeline_run_id.reset(token)

    assert any("run_id=run-correlation-id" in message for message in log_messages)
