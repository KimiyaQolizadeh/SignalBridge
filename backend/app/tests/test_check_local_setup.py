from types import SimpleNamespace

from backend.scripts import check_local_setup


class FakeResult:
    def scalar(self) -> bool:
        return True


class FakeConnection:
    def __enter__(self) -> "FakeConnection":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, _statement: object) -> FakeResult:
        return FakeResult()


class FakeEngine:
    def connect(self) -> FakeConnection:
        return FakeConnection()


def test_database_url_redacts_credentials_and_query() -> None:
    value = check_local_setup.redact_database_url(
        "postgresql+psycopg://user:secret@localhost:5432/db?token=private"
    )

    assert value == "postgresql+psycopg://***:***@localhost:5432/db"
    assert "secret" not in value
    assert "private" not in value


def test_database_and_pgvector_checks_can_use_mock_engine() -> None:
    database_module = SimpleNamespace(engine=FakeEngine())

    assert check_local_setup._database_check(database_module)["status"] == "ok"
    assert check_local_setup._pgvector_check(database_module) == {
        "status": "ok",
        "installed": True,
    }


def test_missing_pgvector_is_critical() -> None:
    summary = {
        "imports": {"status": "ok"},
        "environment": {"status": "ok"},
        "database": {"status": "ok"},
        "pgvector": {"status": "not_installed"},
        "tables": {"status": "ok"},
        "embedding_column": {"status": "ok"},
    }

    assert not check_local_setup._critical_checks_pass(summary, False)
