# Reference Review Promotion And Notes Lifecycle

## Overview

This document defines durable review notes, failed-review semantics, and
dirty-to-gold promotion for the Reference Review Workbench.

The workbench must make promotion explicit and auditable. Notes without
promotion mean the fixture was reviewed and not accepted.

## Parent Architecture

- [Reference Review Workbench Architecture](reference-review-workbench-architecture.md)
- [Reference Review Fixture Source Contract](reference-review-fixture-source-contract.md)
- [Reference Review Async Concurrency](reference-review-async-concurrency.md)

## Review States

- `unreviewed`: dirty artifacts exist, no review note exists.
- `needs-work`: reviewer wrote notes and did not promote.
- `blocked`: reviewer could not judge because source, preview, artifact, or
  context is incomplete.
- `approved-source`: live source model has been accepted, but artifacts have
  not yet been promoted.
- `promoted`: source model and derived artifacts were accepted and gold
  artifacts were written.

## Notes Store

Preferred storage:

```text
tests/reference_review_notes/<fixture-id>.md
```

Each note includes:

- fixture id
- review state
- reviewer notes
- reviewed source model identity
- source revision or checksum when available
- reviewed artifact paths
- promotion provenance
- candidate model links
- timestamp

## Promotion Gate

Promotion requires:

1. Complete fixture source record.
2. Source model loads into interactive preview.
3. Reviewer confirms model behavior from relevant angles.
4. Derived dirty artifact set is complete.
5. Reviewer confirms artifacts match accepted model output.
6. Gold writes are atomic and serialized.
7. Promotion provenance records source and artifact identities.

Codex may recommend promotion but cannot perform it.

## Release Gate Meaning

Release reporting treats states as:

- `promoted`: passes promoted evidence gate.
- `unreviewed`: fails review completion.
- `needs-work`: fails review completion with notes.
- `blocked`: fails review completion with blocking reason.
- `approved-source`: fails promotion completion until artifacts are promoted.

## Specification Manifest For Discovery

## Manifest Review History

- 2026-05-31 loop 1: Critical review found the original candidate mixed note
  storage, promotion, provenance, release gates, and UI presentation.
- 2026-05-31 loop 2: Rescored after moving UI fields to the UI manifest.
- 2026-05-31 loop 3: Split provenance from promotion execution because
  provenance also feeds `.impress`-style evidence and audit reports.
- 2026-05-31 loop 4: Added cross-process locking to the atomic promotion leaf.
- 2026-05-31 loop 5: Final review confirmed all candidates are implementation
  leaves below the split threshold.

### Candidate Spec: Review Note Store

Discovery purpose:
- Persist fixture-scoped review notes without promoting dirty artifacts.

Responsibilities:
- Functions/methods:
  - note loader
  - note writer
- Data structures/models:
  - review note record
  - note write result
- Dependencies/services:
  - source record resolver
  - async durable write queue
- Returns/outputs/signals:
  - loaded note
  - saved note
- UI surfaces/components:
  - none; notes UI belongs to the UI manifest
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: current reference artifact state concepts
  - Additions to existing reusable library/module: reference lifecycle helpers
  - New reusable library/module to create: review note store
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - note writes are serialized through durable write lane
- Destructive/write behavior:
  - writes note files
- Security/privacy-sensitive behavior:
  - note files do not persist full chat logs or local secrets by default
- Performance-sensitive behavior:
  - bounded note size
- Cross-screen reusable behavior:
  - notes feed queue status, notes panel, and release reports

Project readiness fields:
- Implementation owner/module:
  - future `reference_review/notes`
- Chosen defaults / parameters:
  - notes without promotion fail review completion
- Test strategy:
  - note read/write, redaction/default content, stale write completion tests
- Data ownership:
  - note store owns durable review notes
- Routes:
  - fixture id to note path to note record
- Open questions / nuance discovered:
  - none
- Readiness blockers:
  - none

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 1 x 3 = 3
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 0 x 2 = 0
- Total: 24.5

Split decision:
- No split needed. This is high but cohesive around durable note persistence.

### Candidate Spec: Review State Classifier

Discovery purpose:
- Classify each fixture as unreviewed, needs-work, blocked, approved-source,
  promoted, or release-gate failing.

Responsibilities:
- Functions/methods:
  - review state classifier
  - state reason builder
- Data structures/models:
  - review state enum
  - state reason record
- Dependencies/services:
  - note store
  - promotion provenance store
- Returns/outputs/signals:
  - review state
  - state reason
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: reference artifact state concepts
  - Additions to existing reusable library/module: reference lifecycle helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - classifier can run in discovery/report worker
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - state reasons avoid note body leakage in summary reports
- Performance-sensitive behavior:
  - bounded per-fixture reads
- Cross-screen reusable behavior:
  - review state feeds queue, action bar, and release reports

Project readiness fields:
- Implementation owner/module:
  - future `reference_review/review_state`
- Chosen defaults / parameters:
  - notes without promotion classify as needs-work
- Test strategy:
  - state matrix tests for every review status
- Data ownership:
  - classifier owns derived state, not durable records
- Routes:
  - notes/provenance/artifact presence to review state
- Open questions / nuance discovered:
  - none
- Readiness blockers:
  - none

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 0 x 2 = 0
- Total: 21.5

Split decision:
- No split needed. State classification and reason building are one derived
  review-state boundary over existing note, source, and promotion records.

### Candidate Spec: Promotion Validator

Discovery purpose:
- Validate that a fixture can be promoted only after source model and derived
  artifact evidence are reviewable.

Responsibilities:
- Functions/methods:
  - promotion validator
  - blocked promotion diagnostic builder
- Data structures/models:
  - promotion validation result
  - blocked promotion diagnostic
- Dependencies/services:
  - source record resolver
  - dirty/clean reference path helpers
- Returns/outputs/signals:
  - promotion allowed
  - blocked promotion diagnostic
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: reference path helpers
  - Additions to existing reusable library/module: reference lifecycle helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - validator can run in worker before durable write
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - diagnostics omit unrelated local paths
- Performance-sensitive behavior:
  - checksum validation is bounded to fixture artifacts
- Cross-screen reusable behavior:
  - validation result feeds confirmation, queue, and release report

Project readiness fields:
- Implementation owner/module:
  - future `reference_review/promotion_validation`
- Chosen defaults / parameters:
  - no source record means promotion refused
- Test strategy:
  - missing source, missing dirty artifact, checksum mismatch, and allowed cases
- Data ownership:
  - validator owns promotion eligibility
- Routes:
  - fixture evidence to validation result
- Open questions / nuance discovered:
  - none
- Readiness blockers:
  - none

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 0 x 2 = 0
- Total: 21.5

Split decision:
- No split needed. Validation and blocked-promotion diagnostics are cohesive
  because diagnostics explain the same eligibility decision.

### Candidate Spec: Atomic Promotion Executor

Discovery purpose:
- Promote dirty artifacts to gold references atomically with cross-process
  locking and rollback diagnostics.

Responsibilities:
- Functions/methods:
  - promotion executor
  - rollback diagnostic builder
- Data structures/models:
  - promotion request
  - promotion result
- Dependencies/services:
  - async durable write queue
  - file lock wrapper
- Returns/outputs/signals:
  - promotion success
  - promotion failure diagnostic
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: reference path helpers
  - Additions to existing reusable library/module: reference lifecycle helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - promotion writes run through serialized durable write lane
- Destructive/write behavior:
  - writes gold artifacts and replaces previous promoted evidence
- Security/privacy-sensitive behavior:
  - writes only inside configured reference roots
- Performance-sensitive behavior:
  - copies bounded fixture artifacts and verifies checksums
- Cross-screen reusable behavior:
  - promotion result feeds queue, notes panel, and release reports

Project readiness fields:
- Implementation owner/module:
  - future `reference_review/promotion_executor`
- Chosen defaults / parameters:
  - bounded lock wait; failure leaves dirty artifacts untouched
- Test strategy:
  - atomic write, lock conflict, checksum failure, rollback, and stale
    completion tests
- Data ownership:
  - promotion executor owns gold artifact mutation
- Routes:
  - validated promotion request to durable write lane to result
- Open questions / nuance discovered:
  - none
- Readiness blockers:
  - none

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 0 x 2 = 0
- Total: 24.5

Split decision:
- No split needed. Score is high but cohesive because atomicity and locking are
  the promotion execution boundary.

### Candidate Spec: Promotion Provenance And Release Gate Report

Discovery purpose:
- Record source provenance for promotions and report fixtures that remain
  unreviewed, noted-only, blocked, or unpromoted.

Responsibilities:
- Functions/methods:
  - provenance writer
  - release gate reporter
- Data structures/models:
  - promotion provenance record
  - release gate report
- Dependencies/services:
  - source context payload
  - review state classifier
- Returns/outputs/signals:
  - provenance record
  - release gate failure report
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: source context payload
  - Additions to existing reusable library/module: reference lifecycle helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - provenance writes use durable write lane; reports can run in worker
- Destructive/write behavior:
  - writes provenance files
- Security/privacy-sensitive behavior:
  - provenance excludes local secrets and full chat logs
- Performance-sensitive behavior:
  - release report scans bounded reference roots
- Cross-screen reusable behavior:
  - provenance/report data feeds queue, release summaries, and audit trails

Project readiness fields:
- Implementation owner/module:
  - future `reference_review/provenance`
- Chosen defaults / parameters:
  - promotion provenance includes source identity and artifact checksums
- Test strategy:
  - provenance shape, redaction, release failure matrix, and report ordering
- Data ownership:
  - provenance owns promotion evidence metadata
- Routes:
  - promotion result plus source context to provenance and release report
- Open questions / nuance discovered:
  - UI may say `gold` while legacy code paths may say `clean`
- Readiness blockers:
  - none

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 0 x 2 = 0
- Total: 24.5

Split decision:
- No split needed. Provenance and release report share the same promoted
  evidence state and source context.

## Change History

- 2026-05-31: Ran five critical review, rescore, and split passes over the
  specification manifest. Split note persistence, review-state classification,
  promotion validation, atomic promotion, and provenance/release reporting into
  separate leaves.
- 2026-05-30: Split notes and promotion lifecycle out of the parent workbench
  architecture and made notes-without-promotion a release-gate failure state.
