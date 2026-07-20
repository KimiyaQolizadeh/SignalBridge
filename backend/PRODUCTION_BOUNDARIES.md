# Production boundaries

This Phase 1 hardening changes no production behavior.

SignalBridge's production path has seven ordered stages: transcript parsing,
speaker classification, candidate extraction, evidence validation, business
scoring, semantic deduplication, and final ranking.

Validation verdicts are intentionally eligible by purpose:

- scoring: `pass`, `needs_review`;
- deduplication: `pass`, `needs_review`;
- final ranking: `pass` only.

Validation evaluation, synthetic datasets, benchmark runners, experimental
prompts, and importance estimation are not production dependencies. Importance
and cross-encoder experiments are disabled by default. Production code must not
import evaluation modules.

The one temporary exception is the exact import of `cross_encoder` by
`signal_deduplicator`. The default `existing_embedding_only` mode does not load
the optional model. This exception is guarded by a test and should be removed
when deduplication research moves under `backend/experiments`.
