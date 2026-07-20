from contextvars import ContextVar


pipeline_run_id: ContextVar[str | None] = ContextVar(
    "pipeline_run_id", default=None
)
