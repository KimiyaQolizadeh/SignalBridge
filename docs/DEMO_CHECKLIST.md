# SignalBridge Demo and Review Checklist

## Before demo

- [ ] PostgreSQL/pgvector is running: `docker compose ps`.
- [ ] Backend is running and `/health`, `/health/db`, and `/docs` respond.
- [ ] Startup reports migrations applied and schema verification passed.
- [ ] Frontend is running at the configured CORS origin.
- [ ] `OPENAI_API_KEY` and database credentials are configured without displaying them.
- [ ] A known, approved non-private transcript is available.
- [ ] The transcript's latest successful analysis run has been checked.
- [ ] Final drivers/blockers, evidence links, and run history load before the presentation.
- [ ] No unrelated external model experiment is running.

## During demo

- [ ] Upload a UTF-8 `.txt` transcript.
- [ ] Show the seven computational stages:
  1. parsing,
  2. speaker classification,
  3. candidate extraction,
  4. evidence validation,
  5. business scoring,
  6. semantic deduplication,
  7. final ranking.
- [ ] Show at least one driver when the approved run contains one.
- [ ] Show at least one blocker when the approved run contains one.
- [ ] Open the exact advisor evidence in the transcript.
- [ ] Show supporting evidence for a consolidated signal.
- [ ] Show adjacent context for a referential quote.
- [ ] Point out separate **Validated** and **Needs review** counts.
- [ ] Open analysis run history and diagnostics.
- [ ] Explain that full analysis calls upstream models and creates a new immutable run.
- [ ] Explain that replay validation reuses saved candidates, skips extraction/speaker classification, and leaves the source run unchanged.

## Failure fallback

- [ ] Do not repeatedly rerun external analysis during a time-limited presentation.
- [ ] Select an existing completed run from history.
- [ ] Inspect the failed stage and safe error category in diagnostics.
- [ ] Confirm database health before retrying.
- [ ] Use replay validation only when demonstrating downstream re-evaluation is necessary.
- [ ] If the provider is unavailable, continue with persisted evidence, final signals, and audit snapshots.

## Post-demo review

- [ ] Confirm no private transcript was exported or copied unintentionally.
- [ ] Remove temporary presentation files from shared locations.
- [ ] Record any failed stage/run ID without copying transcript content into tickets or chat.
- [ ] Confirm no source run was modified.
