"""Add durable run-scoped pipeline artifacts and backfill attributable data."""

from sqlalchemy import inspect, text

from backend.app.database import Base
from backend.app import models  # noqa: F401


revision = "20260719_01"
down_revision = None


def upgrade(connection) -> None:
    Base.metadata.create_all(bind=connection)
    db_inspector = inspect(connection)
    for table_name in ("candidate_signals", "final_signals"):
        columns = {column["name"] for column in db_inspector.get_columns(table_name)}
        if "analysis_run_id" not in columns:
            connection.execute(text(f"ALTER TABLE {table_name} ADD COLUMN analysis_run_id VARCHAR(36) REFERENCES analysis_runs(id) ON DELETE CASCADE"))
    connection.execute(text("CREATE INDEX IF NOT EXISTS ix_candidate_signals_analysis_run_id ON candidate_signals (analysis_run_id)"))
    connection.execute(text("CREATE INDEX IF NOT EXISTS ix_final_signals_analysis_run_id ON final_signals (analysis_run_id)"))
    if connection.dialect.name != "postgresql":
        return
    connection.execute(text("""
        INSERT INTO analysis_runs (id, transcript_id, status, run_type, input_sha256, started_at, completed_at, retry_count, configuration_snapshot, summary, created_at)
        SELECT 'legacy-' || t.id, t.id, 'completed', 'full', repeat('0', 64), t.updated_at, t.updated_at, 0,
               json_build_object('backfilled', true, 'provenance', 'unavailable'), json_build_object('backfilled', true), t.updated_at
        FROM transcripts t
        WHERE EXISTS (SELECT 1 FROM candidate_signals c WHERE c.transcript_id = t.id)
           OR EXISTS (SELECT 1 FROM final_signals f WHERE f.transcript_id = t.id)
        ON CONFLICT (id) DO NOTHING
    """))
    connection.execute(text("UPDATE candidate_signals SET analysis_run_id = 'legacy-' || transcript_id WHERE analysis_run_id IS NULL AND EXISTS (SELECT 1 FROM analysis_runs r WHERE r.id = 'legacy-' || candidate_signals.transcript_id)"))
    connection.execute(text("UPDATE final_signals SET analysis_run_id = 'legacy-' || transcript_id WHERE analysis_run_id IS NULL AND EXISTS (SELECT 1 FROM analysis_runs r WHERE r.id = 'legacy-' || final_signals.transcript_id)"))
    connection.execute(text("""
        INSERT INTO candidate_snapshots (analysis_run_id, transcript_id, legacy_candidate_id, item_type, category, advisor_quote, normalized_evidence, timestamp, evidence_strength, rationale, extraction_confidence, source_turn_ids, ownership, created_at)
        SELECT c.analysis_run_id, c.transcript_id, c.id, c.item_type, c.category, c.advisor_quote,
               lower(trim(c.advisor_quote)), c.timestamp, c.evidence_strength, c.rationale,
               c.extraction_confidence, COALESCE(c.source_turn_ids, '[]'::json), 'advisor', c.created_at
        FROM candidate_signals c
        WHERE c.analysis_run_id IS NOT NULL
          AND NOT EXISTS (SELECT 1 FROM candidate_snapshots s WHERE s.analysis_run_id = c.analysis_run_id AND s.legacy_candidate_id = c.id)
    """))
    connection.execute(text("""
        INSERT INTO validation_snapshots (analysis_run_id, candidate_snapshot_id, structured_findings, deterministic_findings, derived_verdict, rejection_reasons, review_reasons, primary_reason, retry_count, created_at)
        SELECT s.analysis_run_id, s.id, NULL, json_build_object('backfilled', true, 'details', 'unavailable'),
               COALESCE(sc.validator_verdict, 'unavailable'),
               CASE WHEN sc.validator_verdict = 'reject' AND sc.rejection_reason IS NOT NULL THEN json_build_array(sc.rejection_reason) ELSE '[]'::json END,
               CASE WHEN sc.validator_verdict = 'needs_review' AND sc.rejection_reason IS NOT NULL THEN json_build_array(sc.rejection_reason) ELSE '[]'::json END,
               sc.rejection_reason, 0, COALESCE(sc.created_at, s.created_at)
        FROM candidate_snapshots s LEFT JOIN signal_scores sc ON sc.signal_id = s.legacy_candidate_id
        WHERE NOT EXISTS (SELECT 1 FROM validation_snapshots v WHERE v.analysis_run_id = s.analysis_run_id AND v.candidate_snapshot_id = s.id)
    """))


def downgrade(connection) -> None:
    for table in ("final_ranking_snapshots", "deduplication_snapshots", "scoring_snapshots", "validation_snapshots", "candidate_snapshots", "extraction_batch_items", "extraction_batches", "speaker_classification_snapshots"):
        connection.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
    connection.execute(text("ALTER TABLE final_signals DROP COLUMN IF EXISTS analysis_run_id"))
    connection.execute(text("ALTER TABLE candidate_signals DROP COLUMN IF EXISTS analysis_run_id"))
    connection.execute(text("DROP TABLE IF EXISTS analysis_runs CASCADE"))
