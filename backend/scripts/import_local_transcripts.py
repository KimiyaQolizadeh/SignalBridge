"""Import gitignored local transcript files into the prototype database."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.database import SessionLocal  # noqa: E402
from backend.app.models import Transcript  # noqa: E402
from backend.app.services.text_encoding import (  # noqa: E402
    repair_common_utf8_mojibake,
)


TRANSCRIPT_DIRECTORY = PROJECT_ROOT / "data" / "transcripts" / "real"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("limit must be at least 1")
    return parsed


def _safe_file_name(path: Path) -> str:
    return path.name.replace("\r", "?").replace("\n", "?")


def import_local_transcripts(
    directory: Path,
    db: Session,
    *,
    force: bool = False,
    limit: int | None = None,
    output: Callable[[str], None] = print,
) -> None:
    """Import local UTF-8 .txt files without printing their contents."""
    try:
        files = sorted(
            (
                path
                for path in directory.iterdir()
                if path.is_file() and path.suffix.lower() == ".txt"
            ),
            key=lambda path: path.name.lower(),
        )
    except OSError:
        output("Failed: transcript_directory reason=directory_unavailable")
        return

    if limit is not None:
        files = files[:limit]

    for path in files:
        file_name = path.name
        safe_name = _safe_file_name(path)

        try:
            existing = list(
                db.scalars(
                    select(Transcript).where(Transcript.file_name == file_name)
                ).all()
            )
        except SQLAlchemyError:
            db.rollback()
            output(f"Failed: {safe_name} reason=database_error")
            continue

        # Avoid opening confidential files that have already been imported.
        if existing and not force:
            output(f"Skipped duplicate: {safe_name}")
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            output(f"Failed: {safe_name} reason=invalid_utf8")
            continue
        except OSError:
            output(f"Failed: {safe_name} reason=read_error")
            continue

        text = repair_common_utf8_mojibake(text)

        if not text:
            output(f"Skipped empty: {safe_name}")
            continue

        try:
            if force:
                for transcript in existing:
                    db.delete(transcript)
                db.flush()

            transcript = Transcript(
                file_name=file_name,
                raw_text=text,
                token_count=max(1, len(text) // 4),
                status="uploaded",
            )
            db.add(transcript)
            db.commit()
            db.refresh(transcript)
        except SQLAlchemyError:
            db.rollback()
            output(f"Failed: {safe_name} reason=database_error")
            continue

        output(
            f"Imported: {safe_name} -> transcript_id={transcript.id} "
            f"token_count={transcript.token_count} status={transcript.status}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import local confidential transcripts into SignalBridge."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete rows with matching file names before importing.",
    )
    parser.add_argument(
        "--limit",
        type=_positive_int,
        metavar="N",
        help="Import at most N transcript files.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with SessionLocal() as db:
        import_local_transcripts(
            TRANSCRIPT_DIRECTORY,
            db,
            force=args.force,
            limit=args.limit,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
