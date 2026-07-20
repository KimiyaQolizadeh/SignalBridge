"""Thread-safe in-memory processing progress and safe run diagnostics.

State is process-local and is lost on restart. The abstraction can be replaced
with durable storage later without changing API handlers.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from threading import RLock
from typing import Any


_lock = RLock()
_progress: dict[int, dict[str, Any]] = {}
_diagnostics: dict[int, dict[str, Any]] = {}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def start(transcript_id: int, run_id: str) -> None:
    now = _now()
    with _lock:
        _progress[transcript_id] = {
            "transcript_id": transcript_id,
            "run_id": run_id,
            "current_stage": "queued",
            "status": "queued",
            "started_at": now,
            "updated_at": now,
            "completed_at": None,
            "elapsed_seconds": 0.0,
            "error_category": None,
        }


def transition(transcript_id: int, stage: str, status: str = "processing") -> None:
    with _lock:
        state = _progress.get(transcript_id)
        if state is None:
            return
        now = _now()
        state.update(current_stage=stage, status=status, updated_at=now)
        state["elapsed_seconds"] = round((now - state["started_at"]).total_seconds(), 2)


def complete(transcript_id: int, *, without_results: bool = False) -> None:
    terminal = "completed_without_results" if without_results else "completed"
    with _lock:
        state = _progress.get(transcript_id)
        if state is None:
            return
        now = _now()
        state.update(current_stage=terminal, status=terminal, updated_at=now, completed_at=now)
        state["elapsed_seconds"] = round((now - state["started_at"]).total_seconds(), 2)


def fail(transcript_id: int, stage: str, error_category: str) -> None:
    with _lock:
        state = _progress.get(transcript_id)
        if state is None:
            return
        now = _now()
        state.update(current_stage=stage, status="failed", updated_at=now, completed_at=now, error_category=error_category)
        state["elapsed_seconds"] = round((now - state["started_at"]).total_seconds(), 2)


def get(transcript_id: int) -> dict[str, Any]:
    with _lock:
        state = deepcopy(_progress.get(transcript_id))
    if state is None:
        return {"transcript_id": transcript_id, "run_id": None, "current_stage": "not_started", "status": "idle", "started_at": None, "updated_at": None, "completed_at": None, "elapsed_seconds": 0.0, "error_category": None}
    if state["completed_at"] is None:
        state["elapsed_seconds"] = round((_now() - state["started_at"]).total_seconds(), 2)
    return state


def store_diagnostics(transcript_id: int, data: dict[str, Any]) -> None:
    with _lock:
        _diagnostics[transcript_id] = deepcopy(data)


def get_diagnostics(transcript_id: int) -> dict[str, Any] | None:
    with _lock:
        return deepcopy(_diagnostics.get(transcript_id))
