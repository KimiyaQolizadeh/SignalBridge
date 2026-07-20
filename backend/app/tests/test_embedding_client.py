from types import SimpleNamespace

import pytest

from backend.app.services import embedding_client
from backend.app.services.pipeline_context import pipeline_run_id


def test_embedding_logs_pipeline_run_correlation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log_messages: list[str] = []
    fake_embeddings = SimpleNamespace(
        create=lambda **_kwargs: SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.1, 0.2])]
        )
    )
    monkeypatch.setattr(embedding_client.settings, "openai_api_key", "test-key")
    monkeypatch.setattr(
        embedding_client,
        "OpenAI",
        lambda **_kwargs: SimpleNamespace(embeddings=fake_embeddings),
    )
    monkeypatch.setattr(
        embedding_client.logger,
        "info",
        lambda message, *args: log_messages.append(message % args),
    )
    token = pipeline_run_id.set("run-correlation-id")
    try:
        assert embedding_client.embed_text("safe input") == [0.1, 0.2]
    finally:
        pipeline_run_id.reset(token)

    assert any("run_id=run-correlation-id" in message for message in log_messages)
