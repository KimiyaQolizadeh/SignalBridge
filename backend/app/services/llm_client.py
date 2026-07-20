import json
import time
from typing import Any, Callable

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel, ValidationError

from ..config import settings
from ..logging_config import get_logger
from .prompt_loader import load_prompt
from .pipeline_context import pipeline_run_id
from .telemetry import record_call


logger = get_logger(__name__)


class LLMClientError(Exception):
    """A safe application-level error raised by the LLM client wrapper."""


def _optional_int(value: object) -> int | None:
    return int(value) if isinstance(value, int) else None


def extract_response_usage(response: object) -> dict[str, int | None]:
    """Safely normalize optional Responses API usage fields."""
    usage = getattr(response, "usage", None)
    input_details = getattr(usage, "input_tokens_details", None)
    output_details = getattr(usage, "output_tokens_details", None)
    return {
        "input_tokens": _optional_int(getattr(usage, "input_tokens", None)),
        "output_tokens": _optional_int(getattr(usage, "output_tokens", None)),
        "reasoning_tokens": _optional_int(
            getattr(output_details, "reasoning_tokens", None)
        ),
        "cached_input_tokens": _optional_int(
            getattr(input_details, "cached_tokens", None)
        ),
        "total_tokens": _optional_int(getattr(usage, "total_tokens", None)),
    }


def _record_telemetry(
    *,
    model: str,
    started_at: float,
    success: bool,
    retry_count: int,
    error_type: str | None,
    response: object | None = None,
) -> None:
    usage = (
        extract_response_usage(response)
        if response is not None
        else {
            "input_tokens": None,
            "output_tokens": None,
            "reasoning_tokens": None,
            "cached_input_tokens": None,
            "total_tokens": None,
        }
    )
    record_call(
        model=model,
        duration_ms=(time.monotonic() - started_at) * 1000,
        success=success,
        retry_count=retry_count,
        error_type=error_type,
        **usage,
    )


def safe_json_dumps(payload: dict) -> str:
    """Serialize model input without logging it."""

    return json.dumps(payload, ensure_ascii=False)


def _create_client(api_key: str) -> OpenAI:
    # Disable SDK retries so this module owns and observes the retry policy.
    return OpenAI(api_key=api_key, max_retries=0)


def _is_transient_error(error: Exception) -> bool:
    if isinstance(error, (APIConnectionError, APITimeoutError, RateLimitError)):
        return True
    return isinstance(error, APIStatusError) and error.status_code >= 500


def _log_result(
    *,
    prompt_file_name: str,
    model: str,
    transcript_id: int | None,
    started_at: float,
    success: bool,
    error_type: str,
) -> None:
    logger.info(
        "action=llm_call prompt_file_name=%s model=%s transcript_id=%s "
        "run_id=%s duration_seconds=%.3f success=%s error_type=%s",
        prompt_file_name,
        model,
        transcript_id,
        pipeline_run_id.get(),
        time.monotonic() - started_at,
        success,
        error_type,
    )


def call_llm_json(
    *,
    prompt_file_name: str,
    input_payload: dict,
    response_model: type[BaseModel],
    model: str,
    temperature: float = 0.0,
    max_retries: int = 2,
    transcript_id: int | None = None,
    response_observer: Callable[[dict[str, Any]], None] | None = None,
) -> BaseModel:
    """Call the Responses API and validate its structured JSON output."""

    started_at = time.monotonic()
    logger.info(
        "action=llm_call_start prompt_file_name=%s model=%s transcript_id=%s run_id=%s",
        prompt_file_name,
        model,
        transcript_id,
        pipeline_run_id.get(),
    )

    if not settings.openai_api_key:
        _record_telemetry(
            model=model, started_at=started_at, success=False,
            retry_count=0, error_type="MissingAPIKey",
        )
        _log_result(
            prompt_file_name=prompt_file_name,
            model=model,
            transcript_id=transcript_id,
            started_at=started_at,
            success=False,
            error_type="MissingAPIKey",
        )
        raise LLMClientError("OPENAI_API_KEY is not configured")

    if max_retries < 0:
        _record_telemetry(
            model=model, started_at=started_at, success=False,
            retry_count=0, error_type="InvalidRetryConfiguration",
        )
        _log_result(
            prompt_file_name=prompt_file_name,
            model=model,
            transcript_id=transcript_id,
            started_at=started_at,
            success=False,
            error_type="InvalidRetryConfiguration",
        )
        raise LLMClientError("max_retries must be zero or greater")

    try:
        prompt = load_prompt(prompt_file_name)
    except (FileNotFoundError, OSError):
        _record_telemetry(
            model=model, started_at=started_at, success=False,
            retry_count=0, error_type="PromptLoadError",
        )
        _log_result(
            prompt_file_name=prompt_file_name,
            model=model,
            transcript_id=transcript_id,
            started_at=started_at,
            success=False,
            error_type="PromptLoadError",
        )
        raise LLMClientError("LLM prompt could not be loaded") from None

    try:
        serialized_payload = safe_json_dumps(input_payload)
    except (TypeError, ValueError):
        _record_telemetry(
            model=model, started_at=started_at, success=False,
            retry_count=0, error_type="InputSerializationError",
        )
        _log_result(
            prompt_file_name=prompt_file_name,
            model=model,
            transcript_id=transcript_id,
            started_at=started_at,
            success=False,
            error_type="InputSerializationError",
        )
        raise LLMClientError("LLM input payload could not be serialized") from None

    try:
        client = _create_client(settings.openai_api_key)
    except Exception as error:
        _record_telemetry(
            model=model, started_at=started_at, success=False,
            retry_count=0, error_type=type(error).__name__,
        )
        _log_result(
            prompt_file_name=prompt_file_name,
            model=model,
            transcript_id=transcript_id,
            started_at=started_at,
            success=False,
            error_type=type(error).__name__,
        )
        raise LLMClientError("LLM client could not be initialized") from None

    messages: list[dict[str, Any]] = [
        {"role": "developer", "content": prompt},
        {"role": "user", "content": serialized_payload},
    ]

    for attempt in range(max_retries + 1):
        try:
            response = client.responses.create(
                model=model,
                input=messages,
                temperature=temperature,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": response_model.__name__,
                        "schema": response_model.model_json_schema(),
                        "strict": False,
                    }
                },
                store=False,
            )
            parsed_json = json.loads(response.output_text)
            validated_response = response_model.model_validate(parsed_json)
        except (json.JSONDecodeError, TypeError, ValidationError):
            _record_telemetry(
                model=model, started_at=started_at, success=False,
                retry_count=attempt, error_type="ResponseValidationError",
                response=locals().get("response"),
            )
            _log_result(
                prompt_file_name=prompt_file_name,
                model=model,
                transcript_id=transcript_id,
                started_at=started_at,
                success=False,
                error_type="ResponseValidationError",
            )
            raise LLMClientError("LLM response failed schema validation") from None
        except Exception as error:
            if _is_transient_error(error) and attempt < max_retries:
                time.sleep(0.5 * (2**attempt))
                continue

            _log_result(
                prompt_file_name=prompt_file_name,
                model=model,
                transcript_id=transcript_id,
                started_at=started_at,
                success=False,
                error_type=type(error).__name__,
            )
            _record_telemetry(
                model=model, started_at=started_at, success=False,
                retry_count=attempt, error_type=type(error).__name__,
            )
            raise LLMClientError("LLM request failed") from None

        _log_result(
            prompt_file_name=prompt_file_name,
            model=model,
            transcript_id=transcript_id,
            started_at=started_at,
            success=True,
            error_type="none",
        )
        _record_telemetry(
            model=model, started_at=started_at, success=True,
            retry_count=attempt, error_type=None, response=response,
        )
        if response_observer is not None:
            usage = getattr(response, "usage", None)
            response_observer({
                "raw_response": response.output_text,
                "response_id": getattr(response, "id", None),
                "status": getattr(response, "status", None),
                "finish_reason": getattr(response, "incomplete_details", None),
                "retry_count": attempt,
                "token_usage": {
                    "input_tokens": getattr(usage, "input_tokens", None),
                    "output_tokens": getattr(usage, "output_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                },
            })
        return validated_response

    raise LLMClientError("LLM request failed")
