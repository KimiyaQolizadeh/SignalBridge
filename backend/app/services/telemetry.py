"""Failure-isolated request telemetry for pipeline model operations."""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass
from decimal import Decimal

from ..logging_config import get_logger
from .model_pricing import estimate_cost


logger = get_logger(__name__)


@dataclass(frozen=True)
class ModelCallTelemetry:
    stage: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    cached_input_tokens: int | None
    total_tokens: int | None
    duration_ms: float
    success: bool
    retry_count: int
    error_type: str | None
    estimated_cost: Decimal | None


_telemetry: ContextVar[list[ModelCallTelemetry] | None] = ContextVar(
    "pipeline_telemetry", default=None
)
pipeline_stage: ContextVar[str | None] = ContextVar("pipeline_stage", default=None)


def start_collection() -> Token[list[ModelCallTelemetry] | None]:
    return _telemetry.set([])


def finish_collection(token: Token[list[ModelCallTelemetry] | None]) -> None:
    _telemetry.reset(token)


def get_calls() -> list[ModelCallTelemetry]:
    return list(_telemetry.get() or [])


def record_call(
    *,
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
    reasoning_tokens: int | None,
    cached_input_tokens: int | None,
    total_tokens: int | None,
    duration_ms: float,
    success: bool,
    retry_count: int,
    error_type: str | None,
    stage: str | None = None,
) -> None:
    """Record telemetry without allowing observability to affect a model call."""
    try:
        calls = _telemetry.get()
        if calls is None:
            return
        cost = None
        if input_tokens is not None and output_tokens is not None:
            cost = estimate_cost(
                model,
                input_tokens=input_tokens,
                cached_input_tokens=cached_input_tokens or 0,
                output_tokens=output_tokens,
            )
        calls.append(
            ModelCallTelemetry(
                stage=stage or pipeline_stage.get() or "unknown",
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                reasoning_tokens=reasoning_tokens,
                cached_input_tokens=cached_input_tokens,
                total_tokens=total_tokens,
                duration_ms=round(max(0.0, duration_ms), 2),
                success=success,
                retry_count=max(0, retry_count),
                error_type=error_type,
                estimated_cost=cost,
            )
        )
    except Exception:
        logger.warning("action=record_telemetry success=false error_type=TelemetryError")
