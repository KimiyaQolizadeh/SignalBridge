from importlib import import_module

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session

from backend.app.database import Base, schema_issues
from backend.app.models import AnalysisRun, Transcript
from backend.app.services.run_persistence import create_analysis_run


migration = import_module(
    "backend.migrations.versions.20260719_01_run_scoped_observability"
)


def _create_run(engine) -> str:
    with Session(engine) as db:
        transcript = db.query(Transcript).first()
        if transcript is None:
            transcript = Transcript(file_name="migration-test.txt", raw_text="No LLM call")
            db.add(transcript)
            db.commit()
        return create_analysis_run(db, transcript).status


def test_create_run_succeeds_on_fresh_database() -> None:
    test_engine = create_engine("sqlite://")
    with test_engine.begin() as connection:
        migration.upgrade(connection)

    assert schema_issues(test_engine) == []
    assert _create_run(test_engine) == "running"


def test_create_run_succeeds_after_legacy_database_upgrade() -> None:
    test_engine = create_engine("sqlite://")
    legacy_tables = [
        table for table in Base.metadata.sorted_tables
        if table.name in {"transcripts", "candidate_signals", "final_signals"}
    ]
    Base.metadata.create_all(test_engine, tables=legacy_tables)
    with test_engine.begin() as connection:
        connection.execute(text(
            "INSERT INTO transcripts (file_name, status, raw_text) "
            "VALUES ('legacy.txt', 'uploaded', 'preserve me')"
        ))
        migration.upgrade(connection)

    columns = {item["name"] for item in inspect(test_engine).get_columns("candidate_signals")}
    assert "analysis_run_id" in columns
    with test_engine.connect() as connection:
        assert connection.scalar(text("SELECT raw_text FROM transcripts")) == "preserve me"
    assert _create_run(test_engine) == "running"
