from collections.abc import Generator

from importlib import import_module

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import settings


engine = create_engine(settings.database_url, pool_pre_ping=True)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_database_connection() -> bool:
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
    return True


def init_db() -> None:
    # Import models here so their table metadata is registered before creation.
    from . import models  # noqa: F401

    with engine.begin() as connection:
        if connection.dialect.name == "postgresql":
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        migration = import_module(
            "backend.migrations.versions.20260719_01_run_scoped_observability"
        )
        migration.upgrade(connection)
    verify_schema()


class SchemaVerificationError(RuntimeError):
    """The configured database lacks tables or columns required by the app."""


def schema_issues(bind=engine) -> list[str]:
    from . import models  # noqa: F401

    db_inspector = inspect(bind)
    existing_tables = set(db_inspector.get_table_names())
    issues: list[str] = []
    for table in Base.metadata.sorted_tables:
        if table.name not in existing_tables:
            issues.append(f"missing table: {table.name}")
            continue
        existing_columns = {
            column["name"] for column in db_inspector.get_columns(table.name)
        }
        for column in table.columns:
            if column.name not in existing_columns:
                issues.append(f"missing column: {table.name}.{column.name}")
    return issues


def verify_schema(bind=engine) -> None:
    issues = schema_issues(bind)
    if issues:
        raise SchemaVerificationError(
            "Configured database schema is incomplete: " + "; ".join(issues)
        )
