from collections.abc import Generator
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.app.database import Base
from backend.app.models import Transcript
from backend.scripts.import_local_transcripts import import_local_transcripts


@pytest.fixture
def db() -> Generator[Session, None, None]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    Base.metadata.drop_all(engine)


def test_imports_utf8_and_skips_duplicate_and_empty(
    tmp_path: Path, db: Session
) -> None:
    (tmp_path / "one.txt").write_text("Synthetic transcript.", encoding="utf-8")
    (tmp_path / "empty.txt").write_text("", encoding="utf-8")
    messages: list[str] = []

    import_local_transcripts(tmp_path, db, output=messages.append)
    import_local_transcripts(tmp_path, db, output=messages.append)

    transcripts = list(db.scalars(select(Transcript)).all())
    assert len(transcripts) == 1
    assert transcripts[0].file_name == "one.txt"
    assert transcripts[0].status == "uploaded"
    assert any(message.startswith("Imported: one.txt") for message in messages)
    assert "Skipped duplicate: one.txt" in messages
    assert messages.count("Skipped empty: empty.txt") == 2


def test_force_replaces_existing_transcript(tmp_path: Path, db: Session) -> None:
    path = tmp_path / "one.txt"
    path.write_text("First synthetic version.", encoding="utf-8")
    import_local_transcripts(tmp_path, db, output=lambda _message: None)

    path.write_text("Replacement synthetic version.", encoding="utf-8")
    import_local_transcripts(
        tmp_path,
        db,
        force=True,
        output=lambda _message: None,
    )

    transcripts = list(db.scalars(select(Transcript)).all())
    assert len(transcripts) == 1
    assert transcripts[0].raw_text == "Replacement synthetic version."


def test_limit_imports_at_most_requested_files(tmp_path: Path, db: Session) -> None:
    (tmp_path / "a.txt").write_text("Synthetic A", encoding="utf-8")
    (tmp_path / "b.txt").write_text("Synthetic B", encoding="utf-8")

    import_local_transcripts(
        tmp_path,
        db,
        limit=1,
        output=lambda _message: None,
    )

    assert db.scalar(select(Transcript.file_name)) == "a.txt"
