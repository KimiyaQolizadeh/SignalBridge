# SignalBridge Pipeline Reference

This document is the implementation map for the accepted production pipeline. Stage order and business policy are intentionally stable.

## 1. Transcript parsing

- **Purpose:** Convert raw UTF-8 transcript text into ordered turns.
- **Primary file:** `backend/app/services/transcript_parser.py`
- **Input:** Persisted `Transcript.raw_text`.
- **Output:** Ordered `TranscriptTurn` records with speaker labels, timestamps, and text.
- **Persisted artifacts:** `transcript_turns`.
- **Important rules:** Preserve source text; parsing does not infer business meaning.
- **Common failures:** Empty text, unsupported formatting, or database persistence errors.
- **Relevant tests:** `test_transcript_parser.py`, `test_pipeline_runner.py`.

## 2. Speaker classification

- **Purpose:** Establish advisor ownership before extraction.
- **Primary file:** `backend/app/services/speaker_classifier.py`
- **Input:** Parsed turns.
- **Output:** Role and confidence per turn.
- **Persisted artifacts:** Updated turns and `speaker_classification_snapshots`, including model response metadata.
- **Important rules:** Extraction evidence must be advisor-owned with sufficient confidence.
- **Common failures:** Missing turns, malformed structured output, ambiguous roles, provider errors.
- **Relevant tests:** `test_speaker_classifier.py`, `test_pipeline_runner.py`.

## 3. Candidate extraction

- **Purpose:** Maximize grounded recall for statements that could help explain a business decision.
- **Primary file:** `backend/app/services/signal_extractor.py`
- **Input:** Classified turns with bounded adjacent context.
- **Output:** Driver/blocker candidate records with exact evidence, category, rationale, confidence, and source turn IDs.
- **Persisted artifacts:** `extraction_batches`, `extraction_batch_items`, `candidate_signals`, and immutable `candidate_snapshots`. Raw structured model responses are stored on extraction batches.
- **Important rules:** Business-relevant preferences, constraints, dependencies, concerns, reactions, and conditional intent are candidates. No-signal is reserved for greetings, scheduling, filler, jokes, small talk, acknowledgements, and repeats. Exact evidence and ownership checks remain mandatory.
- **Common failures:** Missing classified turns, provider failure, malformed schema, quote/source mismatch, duplicate exact evidence.
- **Relevant tests:** `test_signal_extractor.py`, `test_pipeline_runner.py`.

## 4. Evidence validation

- **Purpose:** Assess factual grounding and attach confidence metadata.
- **Primary file:** `backend/app/services/evidence_validator.py`
- **Input:** Persisted candidates and bounded transcript context.
- **Output:** `pass`, `needs_review`, or `reject`, plus structured findings and reasons.
- **Persisted artifacts:** `signal_scores` validation fields and `validation_snapshots`, including intended raw response fields.
- **Important rules:** Ambiguous but grounded interpretations normally become `needs_review`. Hard fabrication, missing evidence, impossible ownership, unrelated evidence, and direct contradiction are excluded. Validation is not a general importance gate.
- **Common failures:** Missing candidates/turns, untraceable quote, wrong ownership, malformed model output, provider error.
- **Relevant tests:** `test_evidence_validator.py`, `test_validation_v2.py`, `test_validation_evaluation.py`.

## 5. Business scoring

- **Purpose:** Estimate decision importance without changing validation.
- **Primary file:** `backend/app/services/scorer.py`
- **Input:** Candidates allowed by the evidence-integrity policy.
- **Output:** Component scores and fixed weighted `final_score`.
- **Persisted artifacts:** `signal_scores` and `scoring_snapshots`.
- **Important rules:** Weights are ownership 30%, decision impact 25%, explicitness 20%, urgency 15%, evidence quality 10%. Both `pass` and grounded `needs_review` can be scored.
- **Common failures:** Missing validation state, malformed score output, provider or persistence error.
- **Relevant tests:** `test_scorer.py`, `test_precision_regression.py`.

## 6. Semantic deduplication

- **Purpose:** Give one final slot to one underlying decision factor while preserving supporting evidence.
- **Primary file:** `backend/app/services/signal_deduplicator.py`
- **Input:** Scored same-run candidates and embeddings.
- **Output:** Canonical flags, duplicate group IDs, consolidated rationale, and duplicate relationships.
- **Persisted artifacts:** Candidate canonical/group fields and `deduplication_snapshots` with canonical relationship rationale.
- **Important rules:** Directions never merge. Existing embedding similarity remains the baseline. A nearby same-direction reason/conclusion relationship can consolidate when the later statement explicitly concludes misfit/incompatibility. The conclusion is canonical, supporting excerpts remain exact, validation metadata stays visible, and the highest defensible group score is retained. Broad topic overlap alone is insufficient.
- **Common failures:** Embedding service errors, invalid vector dimensions, database errors, or over-broad similarity (guarded by direction, proximity, and independent-effect tests).
- **Relevant tests:** `test_signal_deduplicator.py`, `test_dedup_evaluation.py`, `test_precision_regression.py`.

## 7. Final ranking

- **Purpose:** Select the strongest canonical drivers and blockers.
- **Primary file:** `backend/app/services/reranker.py`
- **Input:** Canonical grounded candidates and business scores.
- **Output:** Ordered final signals and a selection reason.
- **Persisted artifacts:** `final_signals` and `final_ranking_snapshots`.
- **Important rules:** Preferred thresholds are 3.5 for explicit evidence and 4.0 for implied evidence. Above-threshold selection is `above_threshold`. If one direction has no threshold-qualified candidate, its single highest grounded `needs_review` candidate may be selected as `best_grounded_fallback`. Rejects are never promoted; neither direction is forced; no minimum count exists.
- **Common failures:** Missing scores, invalid reranker IDs/ranks, provider errors, or persistence failure. Invalid/provider output uses the existing deterministic fallback for threshold-qualified candidates.
- **Relevant tests:** `test_reranker.py`, `test_pipeline_runner.py`, `test_precision_regression.py`.

## 8. Persisted final signals and presentation

- **Purpose:** Serve auditable final results with exact evidence and context.
- **Primary files:** `backend/app/api/transcripts.py`, `frontend/src/components/SignalCard.tsx`.
- **Input:** Latest completed run's final and candidate records plus transcript turns.
- **Output:** Drivers/blockers with verdict, score, exact quote, supporting evidence, and adjacent context.
- **Persisted artifacts:** `final_signals`; supporting evidence derives from persisted duplicate groups, and context derives from persisted turns.
- **Important rules:** Exact evidence is never rewritten. Supporting evidence and adjacent context are separate fields. UI counts `pass` as Validated and `needs_review` separately.
- **Common failures:** Missing latest run relationships or missing source turns; final evidence remains available even when optional context is empty.
- **Relevant tests:** `test_transcripts_api.py`; frontend TypeScript check and production build.

## Full analysis and replay orchestration

- **Full analysis:** `backend/app/services/pipeline_runner.py` executes all seven computational stages and creates a new immutable run.
- **Replay validation:** `backend/app/services/run_replay.py` clones saved speaker/candidate snapshots into a new run, then runs validation, scoring, deduplication, and ranking only. It makes zero extraction and speaker-classification calls.
- **Persistence:** `backend/app/services/run_persistence.py` stores run configuration, prompt hashes, batches, raw response metadata, and downstream snapshots. Source runs are never overwritten.

## Field reference

| Field | Meaning |
|---|---|
| `validation_verdict` | Evidence result: `pass`, `needs_review`, or `reject`. Runtime/API model naming may use `validator_verdict`; snapshots use `derived_verdict`. |
| `validation_reason` | Primary explanation for rejection or review; snapshots also retain structured reason lists. |
| `business_score` | Fixed-weight decision-importance score, exposed as `final_score` in current APIs. |
| `selection_reason` | `above_threshold` or `best_grounded_fallback` for selected signals. |
| `canonical_candidate_id` | Candidate representing a duplicate group; duplicate snapshots point to it. |
| `supporting_evidence` | Exact evidence excerpts from noncanonical candidates in the canonical group. |
| `adjacent_context` | Neighboring transcript turns that resolve a referential quote; current API field is `evidence_context`. |
| `analysis_run_id` | Immutable UUID tying candidates, responses, snapshots, and finals to one execution. |

## Focused verification

```powershell
.\.venv\Scripts\python.exe -m pytest backend\app\tests\test_signal_extractor.py -q
.\.venv\Scripts\python.exe -m pytest backend\app\tests\test_evidence_validator.py backend\app\tests\test_validation_v2.py -q
.\.venv\Scripts\python.exe -m pytest backend\app\tests\test_scorer.py -q
.\.venv\Scripts\python.exe -m pytest backend\app\tests\test_signal_deduplicator.py -q
.\.venv\Scripts\python.exe -m pytest backend\app\tests\test_reranker.py -q
.\.venv\Scripts\python.exe -m pytest backend\app\tests\test_run_replay.py -q
.\.venv\Scripts\python.exe -m pytest backend\app\tests\test_transcripts_api.py -q
.\.venv\Scripts\python.exe -m pytest backend\app\tests\test_database_migration.py -q
```
