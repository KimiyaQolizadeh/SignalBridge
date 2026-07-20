# SignalBridge presentation code guide

## 1. System purpose

SignalBridge turns recruiting-call transcripts into up to three evidence-grounded
advisor drivers and three blockers. Every final signal retains a verbatim
advisor quote, beginning timestamp, evidence strength, rationale, type, category,
and rank.

## 2. End-to-end request flow

The React frontend calls `POST /api/transcripts/{id}/process-all` through
`processTranscript()` in `frontend/src/api/transcripts.ts`. The FastAPI handler
in `backend/app/api/transcripts.py` calls
`run_full_pipeline_for_transcript()` in
`backend/app/services/pipeline_runner.py`. Seven ordered stages persist their
results, after which the frontend loads `/final-signals`.

## 3. Seven production stages

| Stage | Main entry point | Input | Output |
|---|---|---|---|
| Parsing | `transcript_parser.parse_transcript_text()` | Uploaded raw text | Ordered `TranscriptTurn` rows |
| Speaker classification | `speaker_classifier.classify_speakers_for_transcript()` | Parsed turns | Inferred role and confidence per turn |
| Candidate extraction | `signal_extractor.extract_candidate_signals_for_transcript()` | Reliably advisor-owned turns and bounded context | Recall-oriented `CandidateSignal` rows |
| Evidence validation | `evidence_validator.validate_evidence_for_transcript()` | Candidates and transcript turns | Verdict, support, ownership, risk, and validation diagnostics |
| Business scoring | `scorer.score_signals_for_transcript()` | Pass or needs-review candidates | Five component scores and deterministic final score |
| Deduplication | `signal_deduplicator.deduplicate_signals_for_transcript()` | Scored pass or needs-review candidates | Embeddings, duplicate groups, canonical candidates |
| Final ranking | `reranker.rerank_final_signals_for_transcript()` | Canonical pass candidates meeting final gates | Up to three persisted drivers and blockers |

The ordered orchestration is visible in `_run_full_pipeline()` in
`backend/app/services/pipeline_runner.py`.

## 4. Main models and contracts

- `Transcript`: uploaded file metadata, raw text, and processing status.
- `TranscriptTurn`: ordered text, timestamp, raw label, inferred role, confidence.
- `CandidateSignal`: type, category, verbatim evidence, provenance, rationale,
  embedding, and deduplication state.
- `SignalScore`: validation verdict and diagnostics plus business-score fields.
- `FinalSignal`: candidate reference, type, and final per-type rank.
- `backend/app/llm_schemas.py`: Pydantic contracts for structured model outputs.
- `backend/app/schemas.py`: public FastAPI request/response contracts.

The ORM definitions are in `backend/app/models.py`. The pipeline uses ORM
objects internally and converts to Pydantic models at provider and API boundaries.

## 5. Validation verdicts

- `pass`: sufficiently grounded, advisor-owned, decision-relevant evidence.
- `needs_review`: potentially useful but unresolved or incomplete evidence.
- `reject`: unsupported, incorrectly owned, procedural, contradictory, or unsafe.

Purpose-specific immutable sets live in
`backend/app/services/eligibility_policy.py`:

- scoring: `pass`, `needs_review`;
- deduplication: `pass`, `needs_review`;
- final ranking: `pass` only.

Needs-review candidates are scored and deduplicated so diagnostics remain useful
and comparable. They are excluded from final ranking because unresolved evidence
must not become user-facing intelligence.

## 6. Evidence and precision controls

Extraction is recall-oriented but applies deterministic semantic gates before
persistence. Validation first checks exact quote traceability, source-turn
ownership, structural candidate validity, obvious procedural content, and
rationale/type contradictions. Only candidates surviving those checks reach the
validation model. Final ranking adds stricter support, advisor ownership,
evidence-strength, score, canonical, and verdict gates.

Representative statements may supply context but cannot establish advisor
ownership. Raw transcript text, prompts, payloads, quotes, and provider output
are not written to application logs.

## 7. Scoring

`BusinessScoreOutput` supplies five integer dimensions: advisor ownership,
decision impact, explicitness, urgency, and evidence quality. The service then
recalculates the weighted score deterministically in `calculate_final_score()`.
The model cannot directly choose the stored final number.

The scoring formula and thresholds are production policy and are not changed by
the boundary-hardening work.

## 8. Deduplication

The production default is `existing_embedding_only`. Candidates are embedded,
compared only within the same transcript and signal type, and joined when cosine
similarity reaches the configured threshold. Each group receives one canonical
candidate selected by validation and score quality.

Cross-encoder shadow/active code exists as an inactive experiment. The exact
`signal_deduplicator -> cross_encoder` import is a temporary guarded exception;
the default path does not load the optional model.

## 9. Final ranking

`reranker._is_eligible()` admits only canonical `pass` candidates with the
required support, advisor ownership, evidence strength, and score. The reranker
receives at most eight candidates per type and may select up to three distinct
categories per type. IDs, types, ranks, and category uniqueness are validated.
Invalid model output or an LLM client failure uses deterministic score ordering.
Empty output is valid.

## 10. Persistence flow

Each stage commits its own database changes. A later failure therefore preserves
completed intermediate artifacts for diagnosis. Rerunning parsing replaces
turns; extraction replaces candidates; final ranking replaces `FinalSignal`
rows. This is practical for the local application but is not one atomic job.

PostgreSQL stores application data and pgvector stores 1,536-dimensional
candidate embeddings. `backend/app/database.py` owns the engine and sessions.

## 11. Main API endpoints

- `POST /api/transcripts/upload`: upload a UTF-8 text transcript.
- `GET /api/transcripts`: list transcript metadata.
- `GET /api/transcripts/{id}`: retrieve transcript detail.
- `POST /api/transcripts/{id}/process-all`: execute all seven stages.
- `GET /api/transcripts/{id}/processing-status`: process-local progress.
- `GET /api/transcripts/{id}/turns`: parsed and classified turns.
- `GET /api/transcripts/{id}/final-signals`: user-facing final results.
- `GET /api/transcripts/{id}/candidates`: diagnostic candidates.
- `GET /api/transcripts/{id}/diagnostics`: internal run diagnostics.
- `GET /api/transcripts/{id}/export.csv` and `.jsonl`: final exports.

Standalone endpoints also expose individual stages for local diagnosis.

## 12. Frontend flow

`frontend/src/App.tsx` defines dashboard, upload, and transcript-detail routes.
`Dashboard.tsx` lists and filters work. `UploadTranscript.tsx` validates and
uploads a text file. `TranscriptDetailPage.tsx` owns processing, polling, result,
transcript, evidence-navigation, diagnostics, and export state. `SignalCard.tsx`
presents each final signal; `DiagnosticCandidateCard.tsx` presents internal
candidate details.

TanStack Query owns server state. The process request is synchronous while the
page polls the process-local status endpoint for visible stage progress.

## 13. Important configuration

`backend/app/config.py` loads `.env` through Pydantic settings. Important fields
are the database URL, upload limit, five stage model names, embedding model, and
production deduplication threshold. `ExperimentSettings` provides a clearly
named view of inactive importance and cross-encoder mode defaults while legacy
environment names remain compatible.

Production prompts are `backend/app/prompts/01_...` through `05_...`.
Importance and review-validation prompts are experimental and are not imported
by the production runner.

## 14. Error-handling strategy

Stage services translate database/provider failures into safe domain exceptions.
The runner records the failed stage, redacts internal exception text, associates
model and embedding calls with a UUID run ID, publishes an internal manifest,
and raises `PipelineRunError`. Routes translate these errors into safe HTTP
responses. Observability failures are isolated so logging cannot break business
processing.

Legitimate zero-candidate transcripts finalize successfully with zero drivers
and blockers. Invalid final model selections use the deterministic fallback;
the system never fabricates evidence.

## 15. Current architectural trade-offs and limitations

- The pipeline is synchronous and may hold an HTTP request for several minutes.
- Progress and diagnostics are process-local and disappear on restart.
- Every stage commits separately rather than as one atomic transaction.
- Production and research files are still colocated under `backend/app`.
- Cross-encoder code remains imported by the deduplicator but inactive by default.
- The transcript detail page contains several responsibilities in one component.
- There is no persisted approve, dismiss, or correction workflow.
- Frontend TypeScript/build validation exists, but no frontend interaction suite.

These are incremental cleanup opportunities, not reasons to rewrite the system.

## 16. Likely interviewer questions

**How do you prevent representative statements from becoming advisor signals?**

Speaker attribution is persisted per turn. Extraction uses reliably advisor-owned
evidence, validation deterministically resolves the exact quote to an eligible
source turn, and final ranking requires a high advisor-side score and `pass`.

**Why use multiple model stages?**

Each model has a narrow responsibility and a validated schema. Deterministic
boundaries between stages reduce unsupported output and make failures observable.

**Why score needs-review evidence?**

It preserves useful diagnostics and comparison through deduplication, but the
final policy admits only `pass`, so unresolved evidence never becomes final.

**Can the model choose the final business score?**

No. It returns bounded component values; application code applies the fixed
weighted formula.

**What happens if reranking fails?**

The service uses deterministic ordering of already validated and gated
candidates. It never falls back to unvalidated content.

**How is evidence traceable?**

Each candidate stores a verbatim quote, timestamp, and source-turn IDs. Validation
checks the exact quote against persisted transcript turns.

**How do you control cost and privacy?**

Calls are batched, model-specific, retried only when transient, made with
`store=False`, and recorded with token/cost metadata. Sensitive payloads and
outputs are not logged.

**Why not rewrite the architecture?**

The production call graph is clear and well tested. Experiments can be extracted
incrementally without replacing stable API or persistence contracts.

## 17. Five-minute walkthrough

1. `backend/app/api/transcripts.py`: show the process-all endpoint.
2. `backend/app/services/pipeline_runner.py`: show `_run_full_pipeline()` and the
   seven ordered callables.
3. `speaker_classifier.py` and `signal_extractor.py`: explain advisor ownership
   and recall-oriented extraction.
4. `evidence_validator.py`: show deterministic checks before model validation.
5. `eligibility_policy.py`, `scorer.py`, `signal_deduplicator.py`, `reranker.py`:
   explain the deliberate narrowing from reviewable candidates to final pass.
6. `models.py`: show persisted provenance and `FinalSignal`.
7. `TranscriptDetailPage.tsx` and `SignalCard.tsx`: show how final evidence is
   reviewed and traced back to a turn.

## 18. Fifteen-minute deep dive

1. Start with the five-minute path.
2. Open `llm_client.py` and `llm_schemas.py` to explain structured Responses API
   calls, schema validation, retries, redaction, and correlation.
3. In `evidence_validator.py`, trace deterministic resolution, hard failures,
   LLM validation, derived verdicts, and stored diagnostics.
4. In `scorer.py`, trace component output to deterministic score.
5. In `signal_deduplicator.py`, trace embedding creation, cosine grouping, and
   canonical selection; mention the guarded inactive cross-encoder branch.
6. In `reranker.py`, trace `_is_eligible()`, input limits, response validation,
   deterministic fallback, and final persistence.
7. Return to `pipeline_runner.py` for manifests, timing, usage, errors, and the
   zero-result path.
8. Finish in `TranscriptDetailPage.tsx` with processing status, final results,
   evidence navigation, diagnostics, and export.
