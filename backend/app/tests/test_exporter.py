import csv
import io
import json
from collections.abc import Generator
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.app.api import transcripts
from backend.app.database import Base
from backend.app.models import (
    AnalysisRun,
    CandidateSignal,
    FinalSignal,
    SignalScore,
    Transcript,
    TranscriptTurn,
)
from backend.app.services.exporter import (
    DEBUG_COLUMNS,
    FINAL_COLUMNS,
    export_all_transcripts_csv,
    export_transcript_csv,
    export_transcript_jsonl,
)


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


def csv_rows(content: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(content)))


def jsonl_rows(content: str) -> list[dict]:
    return [json.loads(line) for line in content.splitlines()]


def create_legacy_final(
    db: Session, *, file_name: str, item_type: str = "driver", rank: int = 1
) -> tuple[Transcript, CandidateSignal]:
    transcript = Transcript(file_name=file_name, raw_text="Synthetic content")
    db.add(transcript)
    db.flush()
    candidate = CandidateSignal(
        transcript_id=transcript.id,
        item_type=item_type,
        category="synthetic_category",
        advisor_quote="Synthetic advisor evidence.",
        timestamp="00:00:01",
        evidence_strength="explicit",
        rationale="Synthetic decision-relevant rationale.",
        extraction_confidence=0.9,
        source_turn_ids=[1],
        duplicate_group_id="g_synthetic",
        is_canonical=True,
    )
    db.add(candidate)
    db.flush()
    candidate.score = SignalScore(
        signal_id=candidate.id,
        validator_verdict="pass",
        support_score=0.95,
        advisor_side_score=0.96,
        false_positive_risk=0.05,
        advisor_ownership=5,
        decision_impact=4,
        explicitness=5,
        urgency=3,
        evidence_quality=5,
        final_score=4.55,
    )
    db.add(
        FinalSignal(
            transcript_id=transcript.id,
            signal_id=candidate.id,
            item_type=item_type,
            rank=rank,
        )
    )
    db.commit()
    return transcript, candidate


def create_three_signal_run(db: Session) -> dict:
    transcript = Transcript(file_name="transcript-4.txt", raw_text="Synthetic")
    db.add(transcript)
    db.flush()
    runs = [
        AnalysisRun(
            id=run_id,
            transcript_id=transcript.id,
            status="completed",
            run_type=run_type,
            source_run_id=source_run_id,
            input_sha256=str(index) * 64,
            started_at=datetime(2026, 1, index, tzinfo=timezone.utc),
            completed_at=datetime(2026, 1, index, 0, 1, tzinfo=timezone.utc),
            retry_count=0,
            configuration_snapshot={},
        )
        for index, run_id, run_type, source_run_id in (
            (1, "old-full-run", "full", None),
            (2, "replay-run", "replay_validation", "old-full-run"),
            (3, "current-full-run", "full", None),
        )
    ]
    db.add_all(runs)
    db.flush()

    turns = [
        TranscriptTurn(
            transcript_id=transcript.id,
            turn_index=index,
            timestamp=timestamp,
            raw_speaker_label=speaker,
            inferred_role=role,
            text=text,
        )
        for index, timestamp, speaker, role, text in (
            (0, "00:01:00", "OPTIMIZE_REP", "representative", "Tell me about your objectives."),
            (
                1,
                "00:01:05",
                "ADVISOR_1",
                "advisor",
                "I need that objective—growth, \"income\", and O'Brien's plan.\nIt matters.",
            ),
            (2, None, "OPTIMIZE_REP", "representative", "That makes sense, and we can help."),
        )
    ]
    db.add_all(turns)
    db.flush()

    def candidate(
        *,
        quote: str,
        category: str,
        item_type: str,
        group: str,
        canonical: bool,
        timestamp: str | None,
        run_id: str = "current-full-run",
        source_turn_ids: list[int] | None = None,
        evidence_strength: str = "explicit",
    ) -> CandidateSignal:
        item = CandidateSignal(
            transcript_id=transcript.id,
            analysis_run_id=run_id,
            item_type=item_type,
            category=category,
            advisor_quote=quote,
            timestamp=timestamp,
            evidence_strength=evidence_strength,
            rationale=f"{category} rationale, with \"quoted\" detail.",
            extraction_confidence=0.9,
            source_turn_ids=source_turn_ids or [turns[1].id],
            duplicate_group_id=group,
            is_canonical=canonical,
        )
        db.add(item)
        db.flush()
        return item

    investment = candidate(
        quote=turns[1].text,
        category="Investment Objectives",
        item_type="driver",
        group="g_investment",
        canonical=True,
        timestamp=turns[1].timestamp,
        source_turn_ids=[turns[1].id],
    )
    time_constraint = candidate(
        quote="I simply don't have time to make a change.",
        category="Time Constraint",
        item_type="blocker",
        group="g_time",
        canonical=True,
        timestamp="00:17:50.000",
    )
    business_fit = candidate(
        quote="So I think, unfortunately, we are a misfit for the models that you have.",
        category="Business Model Fit",
        item_type="blocker",
        group="g_fit",
        canonical=True,
        timestamp="00:18:13.950",
    )
    support_one = candidate(
        quote="We are not doing any third party, period.",
        category="Third-Party Avoidance",
        item_type="blocker",
        group="g_fit",
        canonical=False,
        timestamp="00:18:00.000",
    )
    support_two = candidate(
        quote="We don't delegate anything to anyone. We don't sell third-party products.",
        category="No Outsourcing Preference",
        item_type="blocker",
        group="g_fit",
        canonical=False,
        timestamp="00:18:08.500",
    )
    old_signal = candidate(
        quote="Old full-run signal.",
        category="Historical Signal",
        item_type="driver",
        group="g_old",
        canonical=True,
        timestamp="00:00:10",
        run_id="old-full-run",
    )
    replay_signal = candidate(
        quote="Replay diagnostic signal.",
        category="Replay Signal",
        item_type="driver",
        group="g_replay",
        canonical=True,
        timestamp="00:00:20",
        run_id="replay-run",
    )
    nonselected = candidate(
        quote="A scored candidate that was not selected.",
        category="Nonselected Candidate",
        item_type="driver",
        group="g_nonselected",
        canonical=True,
        timestamp="00:02:00",
    )

    for item, verdict, score in (
        (investment, "needs_review", 3.75),
        (time_constraint, "needs_review", 4.50),
        (business_fit, "pass", 4.45),
        (old_signal, "pass", 4.10),
        (replay_signal, "pass", 4.20),
        (nonselected, "pass", 4.30),
    ):
        item.score = SignalScore(
            signal_id=item.id,
            validator_verdict=verdict,
            final_score=score,
        )

    # Insert out of presentation order; the exporter must sort deterministically.
    finals = [
        FinalSignal(
            transcript_id=transcript.id,
            analysis_run_id="current-full-run",
            signal_id=business_fit.id,
            item_type="blocker",
            rank=2,
        ),
        FinalSignal(
            transcript_id=transcript.id,
            analysis_run_id="current-full-run",
            signal_id=investment.id,
            item_type="driver",
            rank=1,
        ),
        FinalSignal(
            transcript_id=transcript.id,
            analysis_run_id="current-full-run",
            signal_id=time_constraint.id,
            item_type="blocker",
            rank=1,
        ),
        FinalSignal(
            transcript_id=transcript.id,
            analysis_run_id="old-full-run",
            signal_id=old_signal.id,
            item_type="driver",
            rank=1,
        ),
        FinalSignal(
            transcript_id=transcript.id,
            analysis_run_id="replay-run",
            signal_id=replay_signal.id,
            item_type="driver",
            rank=1,
        ),
    ]
    db.add_all(finals)
    db.commit()

    return {
        "transcript": transcript,
        "finals": {
            "Investment Objectives": finals[1],
            "Time Constraint": finals[2],
            "Business Model Fit": finals[0],
        },
        "signals": {
            "Investment Objectives": investment,
            "Time Constraint": time_constraint,
            "Business Model Fit": business_fit,
        },
        "support": [support_one, support_two],
    }


def test_exact_count_matches_three_final_signals(db: Session) -> None:
    fixture = create_three_signal_run(db)
    transcript_id = fixture["transcript"].id

    assert len(csv_rows(export_transcript_csv(transcript_id, db))) == 3
    assert len(jsonl_rows(export_transcript_jsonl(transcript_id, db))) == 3
    assert len(transcripts.list_final_signals(transcript_id, db=db)) == 3


def test_exported_identities_are_exact(db: Session) -> None:
    fixture = create_three_signal_run(db)
    transcript_id = fixture["transcript"].id
    expected = ["Investment Objectives", "Time Constraint", "Business Model Fit"]

    assert [row["category"] for row in csv_rows(export_transcript_csv(transcript_id, db))] == expected
    assert [row["category"] for row in jsonl_rows(export_transcript_jsonl(transcript_id, db))] == expected


def test_duplicates_and_nonselected_candidates_never_export_standalone(
    db: Session,
) -> None:
    fixture = create_three_signal_run(db)
    rows = jsonl_rows(export_transcript_jsonl(fixture["transcript"].id, db))

    assert len({row["final_signal_id"] for row in rows}) == 3
    assert sum(row["category"] == "Business Model Fit" for row in rows) == 1
    exported_quotes = {row["advisor_quote"] for row in rows}
    assert all(item.advisor_quote not in exported_quotes for item in fixture["support"])
    assert "A scored candidate that was not selected." not in exported_quotes


def test_current_run_isolated_from_older_successful_run(db: Session) -> None:
    fixture = create_three_signal_run(db)
    rows = jsonl_rows(export_transcript_jsonl(fixture["transcript"].id, db))

    assert {row["analysis_run_id"] for row in rows} == {"current-full-run"}
    assert "Historical Signal" not in {row["category"] for row in rows}


def test_replay_and_diagnostics_do_not_add_export_rows(db: Session) -> None:
    fixture = create_three_signal_run(db)
    rows = jsonl_rows(export_transcript_jsonl(fixture["transcript"].id, db))

    assert "Replay Signal" not in {row["category"] for row in rows}
    assert "Nonselected Candidate" not in {row["category"] for row in rows}


def test_csv_jsonl_and_final_api_field_parity(db: Session) -> None:
    fixture = create_three_signal_run(db)
    transcript_id = fixture["transcript"].id
    csv_by_category = {
        row["category"]: row for row in csv_rows(export_transcript_csv(transcript_id, db))
    }
    json_by_category = {
        row["category"]: row for row in jsonl_rows(export_transcript_jsonl(transcript_id, db))
    }
    api_by_category = {
        row.category: row for row in transcripts.list_final_signals(transcript_id, db=db)
    }

    for category, api_row in api_by_category.items():
        csv_row, json_row = csv_by_category[category], json_by_category[category]
        assert int(csv_row["transcript_id"]) == json_row["transcript_id"] == api_row.transcript_id
        assert csv_row["item_type"] == json_row["item_type"] == api_row.item_type
        assert int(csv_row["rank"]) == json_row["rank"] == api_row.rank
        assert float(csv_row["business_score"]) == json_row["business_score"] == api_row.final_score
        assert csv_row["validation_verdict"] == json_row["validation_verdict"] == api_row.validator_verdict
        assert csv_row["advisor_quote"] == json_row["advisor_quote"] == api_row.advisor_quote
        assert csv_row["timestamp"] == (json_row["timestamp"] or "") == (api_row.timestamp or "")
        assert csv_row["evidence_strength"] == json_row["evidence_strength"] == api_row.evidence_strength
        assert csv_row["rationale"] == json_row["rationale"] == api_row.rationale


def test_structured_fields_are_json_arrays_and_empty_arrays(db: Session) -> None:
    fixture = create_three_signal_run(db)
    transcript_id = fixture["transcript"].id
    csv_by_category = {
        row["category"]: row for row in csv_rows(export_transcript_csv(transcript_id, db))
    }
    json_by_category = {
        row["category"]: row for row in jsonl_rows(export_transcript_jsonl(transcript_id, db))
    }

    fit_csv = csv_by_category["Business Model Fit"]
    fit_json = json_by_category["Business Model Fit"]
    assert json.loads(fit_csv["supporting_evidence"]) == fit_json["supporting_evidence"]
    assert [item["quote"] for item in fit_json["supporting_evidence"]] == [
        item.advisor_quote for item in fixture["support"]
    ]
    assert fit_csv["supporting_quote_1"] == fixture["support"][0].advisor_quote
    assert fit_csv["supporting_timestamp_1"] == (fixture["support"][0].timestamp or "")
    assert fit_csv["supporting_quote_2"] == fixture["support"][1].advisor_quote
    assert fit_csv["supporting_timestamp_2"] == (fixture["support"][1].timestamp or "")
    assert isinstance(json.loads(fit_csv["adjacent_context"]), list)
    assert isinstance(fit_json["adjacent_context"], list)

    time_csv = csv_by_category["Time Constraint"]
    time_json = json_by_category["Time Constraint"]
    assert json.loads(time_csv["supporting_evidence"]) == []
    assert json.loads(time_csv["adjacent_context"]) == []
    assert time_json["supporting_evidence"] == []
    assert time_json["adjacent_context"] == []


def test_quotes_commas_unicode_and_line_breaks_remain_parseable(db: Session) -> None:
    fixture = create_three_signal_run(db)
    transcript_id = fixture["transcript"].id
    expected = fixture["signals"]["Investment Objectives"].advisor_quote

    csv_row = next(
        row for row in csv_rows(export_transcript_csv(transcript_id, db))
        if row["category"] == "Investment Objectives"
    )
    json_row = next(
        row for row in jsonl_rows(export_transcript_jsonl(transcript_id, db))
        if row["category"] == "Investment Objectives"
    )

    assert csv_row["advisor_quote"] == expected
    assert json_row["advisor_quote"] == expected
    assert "—" in csv_row["advisor_quote"]
    assert "\n" in csv_row["advisor_quote"]


def test_repeated_exports_have_deterministic_ui_order(db: Session) -> None:
    fixture = create_three_signal_run(db)
    transcript_id = fixture["transcript"].id

    first_csv = export_transcript_csv(transcript_id, db)
    second_csv = export_transcript_csv(transcript_id, db)
    first_jsonl = export_transcript_jsonl(transcript_id, db)
    second_jsonl = export_transcript_jsonl(transcript_id, db)

    assert first_csv == second_csv
    assert first_jsonl == second_jsonl
    assert [(row["item_type"], int(row["rank"])) for row in csv_rows(first_csv)] == [
        ("driver", 1), ("blocker", 1), ("blocker", 2)
    ]


def test_empty_result_has_csv_header_and_empty_jsonl(db: Session) -> None:
    transcript = Transcript(file_name="empty.txt", raw_text="Synthetic")
    db.add(transcript)
    db.commit()

    assert export_transcript_csv(transcript.id, db).strip() == ",".join(
        [*FINAL_COLUMNS, "supporting_quote_1", "supporting_timestamp_1"]
    )
    assert csv_rows(export_transcript_csv(transcript.id, db)) == []
    assert export_transcript_jsonl(transcript.id, db) == ""


def test_original_csv_columns_are_preserved_first(db: Session) -> None:
    transcript, _ = create_legacy_final(db, file_name="compatible.txt")
    reader = csv.DictReader(io.StringIO(export_transcript_csv(transcript.id, db)))

    assert reader.fieldnames[:len(FINAL_COLUMNS)] == FINAL_COLUMNS
    assert reader.fieldnames[-2:] == ["supporting_quote_1", "supporting_timestamp_1"]
    assert reader.fieldnames[:8] == [
        "transcript_id", "item_type", "rank", "category", "advisor_quote",
        "timestamp", "evidence_strength", "rationale",
    ]


def test_debug_export_retains_existing_diagnostic_columns(db: Session) -> None:
    transcript, _ = create_legacy_final(db, file_name="debug.txt")
    reader = csv.DictReader(io.StringIO(export_transcript_csv(transcript.id, db, debug=True)))

    assert reader.fieldnames[:len(DEBUG_COLUMNS)] == DEBUG_COLUMNS
    assert reader.fieldnames[-2:] == ["supporting_quote_1", "supporting_timestamp_1"]
    assert {"validator_verdict", "final_score", "duplicate_group_id"}.issubset(
        reader.fieldnames or []
    )


def test_batch_export_includes_each_transcript_once(db: Session) -> None:
    first, _ = create_legacy_final(db, file_name="one.txt")
    second, _ = create_legacy_final(db, file_name="two.txt", item_type="blocker")

    rows = csv_rows(export_all_transcripts_csv(db))

    assert {int(row["transcript_id"]) for row in rows} == {first.id, second.id}
    assert len(rows) == 2
