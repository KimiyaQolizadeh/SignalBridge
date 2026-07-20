import time

from openai import OpenAI

from ..config import settings
from ..logging_config import get_logger
from .pipeline_context import pipeline_run_id
from .telemetry import record_call


logger = get_logger(__name__)


class EmbeddingClientError(Exception):
    """A safe application-level error raised by the embedding client."""


def embed_text(text: str, *, model: str | None = None) -> list[float]:
    selected_model = model or settings.embedding_model
    started_at = time.monotonic()

    if not settings.openai_api_key:
        record_call(
            model=selected_model, stage="embeddings", input_tokens=None,
            output_tokens=None, reasoning_tokens=None, cached_input_tokens=None,
            total_tokens=None, duration_ms=(time.monotonic() - started_at) * 1000,
            success=False, retry_count=0, error_type="MissingAPIKey",
        )
        logger.info(
            "action=embed_text model=%s duration_seconds=%.3f success=false "
            "error_type=MissingAPIKey run_id=%s",
            selected_model,
            time.monotonic() - started_at,
            pipeline_run_id.get(),
        )
        raise EmbeddingClientError("OPENAI_API_KEY is not configured")

    try:
        client = OpenAI(api_key=settings.openai_api_key)
        response = client.embeddings.create(
            model=selected_model,
            input=text,
            encoding_format="float",
        )
        embedding = response.data[0].embedding
        if not embedding:
            raise ValueError("empty embedding")
    except Exception as error:
        record_call(
            model=selected_model, stage="embeddings", input_tokens=None,
            output_tokens=None, reasoning_tokens=None, cached_input_tokens=None,
            total_tokens=None, duration_ms=(time.monotonic() - started_at) * 1000,
            success=False, retry_count=0, error_type=type(error).__name__,
        )
        logger.info(
            "action=embed_text model=%s duration_seconds=%.3f success=false "
            "error_type=%s run_id=%s",
            selected_model,
            time.monotonic() - started_at,
            type(error).__name__,
            pipeline_run_id.get(),
        )
        raise EmbeddingClientError("Embedding request failed") from None

    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "prompt_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)
    record_call(
        model=selected_model, stage="embeddings",
        input_tokens=input_tokens if isinstance(input_tokens, int) else None,
        output_tokens=0 if isinstance(input_tokens, int) else None,
        reasoning_tokens=None, cached_input_tokens=None,
        total_tokens=total_tokens if isinstance(total_tokens, int) else None,
        duration_ms=(time.monotonic() - started_at) * 1000,
        success=True, retry_count=0, error_type=None,
    )
    logger.info(
        "action=embed_text model=%s duration_seconds=%.3f success=true "
        "error_type=none run_id=%s",
        selected_model,
        time.monotonic() - started_at,
        pipeline_run_id.get(),
    )
    return list(embedding)
