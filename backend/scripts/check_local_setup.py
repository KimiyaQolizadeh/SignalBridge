"""Safely diagnose the local SignalBridge development environment."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REQUIRED_MODULES = (
    "backend.app.main",
    "backend.app.models",
    "backend.app.database",
)
REQUIRED_TABLES = (
    "transcripts",
    "transcript_turns",
    "candidate_signals",
    "signal_scores",
    "final_signals",
    "human_reviews",
)


def redact_database_url(database_url: str) -> str:
    """Preserve the database location while removing credentials and query data."""
    scheme, separator, remainder = database_url.partition("://")
    if not separator:
        return "<redacted>"

    authority_and_path = remainder.split("?", 1)[0].split("#", 1)[0]
    if "@" in authority_and_path:
        _credentials, location = authority_and_path.rsplit("@", 1)
        authority_and_path = f"***:***@{location}"
    elif scheme.startswith("sqlite"):
        authority_and_path = "/[redacted]"

    return f"{scheme}://{authority_and_path}"


def _failure(error: Exception | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"status": "fail"}
    if error is not None:
        result["error_type"] = type(error).__name__
    return result


def _load_modules() -> tuple[dict[str, Any], dict[str, ModuleType]]:
    results: dict[str, Any] = {}
    modules: dict[str, ModuleType] = {}
    for module_name in REQUIRED_MODULES:
        try:
            modules[module_name] = importlib.import_module(module_name)
            results[module_name] = {"status": "ok"}
        except Exception as exc:
            results[module_name] = _failure(exc)

    status = "ok" if all(item["status"] == "ok" for item in results.values()) else "fail"
    return {"status": status, "modules": results}, modules


def _environment_result() -> tuple[dict[str, Any], Any | None]:
    try:
        config = importlib.import_module("backend.app.config")
        settings = config.settings
        return (
            {
                "status": "ok",
                "app_env": settings.app_env,
                "database_url": redact_database_url(settings.database_url),
                "openai_api_key_set": "yes" if settings.openai_api_key else "no",
                "models": {
                    "speaker_classifier": settings.speaker_classifier_model,
                    "candidate_extractor": settings.candidate_extractor_model,
                    "evidence_validator": settings.evidence_validator_model,
                    "business_scorer": settings.business_scorer_model,
                    "final_reranker": settings.final_reranker_model,
                },
                "embedding_model": settings.embedding_model,
                "dedup_similarity_threshold": settings.dedup_similarity_threshold,
            },
            settings,
        )
    except Exception as exc:
        return _failure(exc), None


def _database_check(database_module: ModuleType) -> dict[str, Any]:
    try:
        from sqlalchemy import text

        with database_module.engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return {"status": "ok"}
    except Exception as exc:
        return _failure(exc)


def _pgvector_check(database_module: ModuleType) -> dict[str, Any]:
    try:
        from sqlalchemy import text

        with database_module.engine.connect() as connection:
            installed = bool(
                connection.execute(
                    text(
                        "SELECT EXISTS ("
                        "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
                        ")"
                    )
                ).scalar()
            )
        return {
            "status": "ok" if installed else "not_installed",
            "installed": installed,
        }
    except Exception as exc:
        return _failure(exc)


def _schema_checks(database_module: ModuleType) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        from sqlalchemy import inspect

        inspector = inspect(database_module.engine)
        tables = {
            table_name: inspector.has_table(table_name)
            for table_name in REQUIRED_TABLES
        }
        tables_result = {
            "status": "ok" if all(tables.values()) else "fail",
            "tables": tables,
        }

        if not tables["candidate_signals"]:
            embedding_result = {"status": "fail", "exists": False}
        else:
            columns = {
                column["name"] for column in inspector.get_columns("candidate_signals")
            }
            embedding_exists = "embedding" in columns
            embedding_result = {
                "status": "ok" if embedding_exists else "fail",
                "exists": embedding_exists,
            }
        return tables_result, embedding_result
    except Exception as exc:
        return _failure(exc), _failure(exc)


def _critical_checks_pass(summary: dict[str, Any], init_requested: bool) -> bool:
    keys = ("imports", "environment", "database", "pgvector", "tables", "embedding_column")
    passed = all(summary[key]["status"] == "ok" for key in keys)
    if init_requested:
        passed = passed and summary.get("init_db", {}).get("status") == "ok"
    return passed


def run_diagnostics(*, init_db_requested: bool = False) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    summary["imports"], modules = _load_modules()
    summary["environment"], settings = _environment_result()

    database_module = modules.get("backend.app.database")
    if database_module is None or settings is None:
        summary["database"] = {"status": "fail", "reason": "prerequisite_failed"}
        summary["pgvector"] = {"status": "fail", "reason": "prerequisite_failed"}
        summary["tables"] = {"status": "fail", "reason": "prerequisite_failed"}
        summary["embedding_column"] = {
            "status": "fail",
            "reason": "prerequisite_failed",
        }
    else:
        summary["database"] = _database_check(database_module)

        if init_db_requested:
            try:
                database_module.init_db()
                summary["init_db"] = {"status": "ok"}
            except Exception as exc:
                summary["init_db"] = _failure(exc)

        summary["pgvector"] = _pgvector_check(database_module)
        summary["tables"], summary["embedding_column"] = _schema_checks(
            database_module
        )

    summary["critical_ok"] = _critical_checks_pass(summary, init_db_requested)
    return summary


def _print_human(summary: dict[str, Any], *, init_requested: bool) -> None:
    if init_requested:
        print("NOTICE: --init-db may create or alter local prototype tables.")

    imports = summary["imports"]
    print(f"Imports: {imports['status']}")
    for module_name, result in imports["modules"].items():
        suffix = f" ({result['error_type']})" if "error_type" in result else ""
        print(f"  {module_name}: {result['status']}{suffix}")

    environment = summary["environment"]
    print(f"Environment: {environment['status']}")
    if environment["status"] == "ok":
        print(f"  APP_ENV: {environment['app_env']}")
        print(f"  DATABASE_URL: {environment['database_url']}")
        print(f"  OPENAI_API_KEY set: {environment['openai_api_key_set']}")
        for name, model in environment["models"].items():
            print(f"  {name} model: {model}")
        print(f"  embedding model: {environment['embedding_model']}")
        print(f"  dedup threshold: {environment['dedup_similarity_threshold']}")

    if init_requested:
        print(f"Database initialization: {summary.get('init_db', {}).get('status', 'fail')}")
    print(f"Database SELECT 1: {summary['database']['status']}")
    print(f"pgvector extension: {summary['pgvector']['status']}")
    print(f"Required tables: {summary['tables']['status']}")
    for table_name, exists in summary["tables"].get("tables", {}).items():
        print(f"  {table_name}: {'ok' if exists else 'missing'}")
    print(f"candidate_signals.embedding: {summary['embedding_column']['status']}")
    print(f"Overall: {'ok' if summary['critical_ok'] else 'fail'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check the local SignalBridge setup.")
    parser.add_argument(
        "--init-db",
        action="store_true",
        help="Initialize or alter local prototype tables before schema checks.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a machine-readable JSON summary.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_diagnostics(init_db_requested=args.init_db)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        _print_human(summary, init_requested=args.init_db)
    return 0 if summary["critical_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
