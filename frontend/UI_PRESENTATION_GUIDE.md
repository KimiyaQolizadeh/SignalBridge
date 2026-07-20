# SignalBridge UI presentation guide

## Product direction and users

SignalBridge is presented as a restrained internal financial-services workspace,
not a consumer AI dashboard. The visual language emphasizes evidence,
operational status, and compact decision information for financial advisors,
analysts, relationship managers, operations teams, and business leadership.

The palette uses deep navy, slate, cool gray, white, restrained green, muted
amber, and restrained red. System fonts, tabular numerals, subtle borders, and
limited shadows keep the product credible and information-dense.

## Main workflow

1. Open `/` and review the transcript work queue.
2. Search, filter, or sort existing records.
3. Open `/upload`, select one UTF-8 text transcript, and upload it.
4. Open `/transcripts/:id` and start analysis.
5. Follow the seven-stage processing panel.
6. Review ranked drivers and blockers.
7. Copy evidence or navigate to its transcript turn.
8. Review transcript context or secondary diagnostics.
9. Export final results as CSV or JSONL.

## Information architecture

- `AppLayout`: stable institutional shell, product identity, current navigation,
  primary upload action, and internal-use guidance.
- `Dashboard`: operational metrics, search/filter controls, and responsive table.
- `UploadTranscript`: safe file-selection, validation, progress, and success flow.
- `TranscriptDetailPage`: identity/header, metrics, navigation, pipeline state,
  final signals, transcript context, diagnostics, and export.
- `SignalCard`: evidence-first final decision factor.
- `DiagnosticCandidateCard`: collapsed secondary candidate details.

Final signals receive primary prominence. Rejected, needs-review, and duplicate
candidates remain in the separate Diagnostics tab.

## Design-system conventions

Tokens live in `src/styles.css`:

- spacing follows 4, 8, 12, 16, 24, 32, 40, and 48 pixels;
- controls use consistent 38-pixel minimum heights;
- radii are 4 or 8 pixels, with pills reserved for compact status badges;
- main text is navy, secondary text is slate, and surfaces are white;
- focus uses a visible professional-blue outline;
- scores, counts, duration, and cost use tabular numerals;
- shadows are limited to surfaces that require hierarchy.

Shared primitives include `StatusBadge`, `MetricCard`, existing button classes,
page and section headers, table surfaces, loading skeletons, inline error states,
empty states, and the processing panel.

## Status and verdict mapping

`src/ui/status.ts` preserves raw API values and maps them for presentation:

- `pass` → Validated;
- `needs_review` → Needs review;
- `reject` → Excluded;
- `finalized` → Analysis complete;
- active processing states → Processing;
- `failed` or `error` → Processing failed;
- pending or unknown states retain a neutral readable label.

Text accompanies every color treatment.

## Responsive behavior

- At desktop widths, dashboard metrics use four equal columns, final drivers and
  blockers use two equal columns, and all seven pipeline stages appear in one row.
- At laptop/tablet widths, metrics use two columns and pipeline stages wrap.
- Below 768 pixels, signal sections stack and transcript rows collapse from two
  columns to a readable single-column conversation layout.
- At approximately 390 pixels, controls stack, metrics remain compact, pipeline
  stages become a vertical list, and evidence actions remain usable.
- Wide operational and diagnostics tables retain controlled horizontal scrolling.

## Accessibility improvements

- A skip link reaches the focusable main region.
- Tabs expose `tablist`, `tab`, `aria-selected`, `aria-controls`, and `tabpanel`.
- Pipeline states use labels and markers in addition to color.
- Loading skeletons have status labels hidden visually but available to assistive
  technology.
- Inputs retain explicit labels; upload errors use alert semantics.
- Buttons remain native keyboard controls with visible focus and disabled states.
- Reduced-motion preferences disable nonessential animation and smooth scrolling.
- Status colors meet a text-supported, non-color-only communication model.

## Major components changed

- `components/AppLayout.tsx`: institutional shell, subtitle, internal indicator,
  clearer navigation, and restrained review disclaimer.
- `pages/Dashboard.tsx`: derived operational metrics, table hierarchy, and skeleton.
- `pages/TranscriptDetailPage.tsx`: compact analysis metrics, truthful pipeline
  status, accessible tabs, and layout-matched loading state.
- `components/SignalCard.tsx`: type, validation, business score, rationale,
  evidence panel, copy action, and transcript navigation.
- `components/DiagnosticCandidateCard.tsx`: centralized verdict presentation.
- `styles.css`: consolidated palette, surface hierarchy, responsive layouts,
  focus behavior, state styling, and data density.

New focused components are `MetricCard`, `PipelineStatus`, and `StatusBadge`.
Data fetching and state ownership remain in the existing pages.

## Intentional UX decisions

- The dashboard uses a table because transcripts are operational records users
  compare across stable fields; decorative cards would reduce scan efficiency.
- Final signals use evidence-first cards because each result requires deeper
  reading, traceability, and rationale.
- Business score is labeled accurately and is not called confidence.
- The pipeline shows stage-level completion only when supported by current status.
- Candidate diagnostics remain secondary so rejected output cannot be confused
  with final intelligence.
- Re-running remains explicit and disabled while processing, preventing duplicate
  submissions.
- Upload and analysis remain separate because that is the real application flow.

## Existing limitations

- Processing is synchronous and progress is process-local.
- The frontend has no persisted approve, dismiss, or correction action.
- Evidence navigation must match quote text and timestamp because final signals do
  not expose source-turn IDs.
- Diagnostics expose technical detail suitable for internal users but are not
  role-gated by the current application.
- There is no configured frontend interaction-test or lint script.

## Three-minute demo

1. Start at `/`: identify the institutional shell, summary metrics, filters, and
   operational transcript table.
2. Open `/upload`: explain format guidance, validation, and the separation between
   upload and analysis.
3. Open a completed `/transcripts/:id`: point out status, compact metrics, and the
   seven-stage pipeline.
4. In Signals, compare drivers and blockers, identify the business score, copy an
   advisor quote, and choose View in transcript.
5. Briefly show Transcript context, Diagnostics separation, and Export.

## Seven-minute demo

1. Explain the shell and internal-use positioning.
2. On `/`, demonstrate search, status filtering, sorting, derived signal counts,
   empty-filter recovery, and row navigation.
3. On `/upload`, show keyboard-accessible selection, accepted-file guidance,
   validation, upload progress, and success actions.
4. On `/transcripts/:id`, explain the header metadata and why technical cost is
   kept in Diagnostics rather than the primary business summary.
5. Start or describe analysis and trace the truthful active/completed/pending
   pipeline states.
6. Review final drivers and blockers: rank, type, validation, business score,
   rationale, verbatim evidence, evidence strength, timestamp, copy, and context.
7. Use View in transcript to show advisor attribution and highlighted context.
8. Open Diagnostics to demonstrate that excluded output is visibly secondary.
9. Finish with export and the internal-review disclaimer.

## Suggested answers

### Why did you choose this layout?

The hierarchy follows the user decision: identify the record, understand run
state, review final signals, verify evidence, then inspect technical detail.
Compact alignment supports frequent internal use without hiding information.

### Why use a table instead of cards?

Transcripts are comparable operational records with repeated fields. A table
provides better scan speed, alignment, sorting context, and density than a card
grid. Cards are reserved for final signals where evidence requires deeper reading.

### How do users distinguish final and filtered signals?

Final validated signals occupy the primary Signals tab. Rejected, needs-review,
and duplicate candidates are grouped separately in Diagnostics with explicit
verdict labels. Color reinforces the distinction but never carries it alone.

### How is evidence traceability presented?

Each final signal shows the verbatim advisor quote, timestamp, evidence strength,
and rationale. Users can copy the quote or navigate directly to the matched
advisor transcript turn.

### How does the interface handle long-running analysis?

The run action disables during processing, existing polling continues, elapsed
time remains visible, and the pipeline panel distinguishes completed, active,
pending, and failed stages without inventing a completion estimate.

### How did you approach accessibility?

I retained native controls, explicit labels, semantic tables and headings, visible
focus, alert/status regions, non-color status text, accessible tab semantics,
responsive text wrapping, and reduced-motion handling.

### How would this scale to more transcripts?

The table, search, status filters, and sort controls already scale conceptually.
The next step would be server-backed pagination and aggregated signal counts once
the API supports them, without changing the information hierarchy.

### What would you improve next?

I would add frontend interaction tests, durable processing jobs, URL-backed tab
state, persisted review actions, role-gated diagnostics, and source-turn IDs in
final results. Those require deliberate product or API work and were not invented
in this UI-only phase.
