# Reference Review Codex Sandbox

## Overview

This document defines the constrained Codex sidecar for the Reference Review
Workbench.

Codex can help diagnose fixture failures, suggest candidate model changes, and
write candidate source files through workbench-provided tools. Codex cannot
promote references, mutate gold artifacts, run arbitrary shell commands, or
write outside approved roots.

## Parent Architecture

- [Reference Review Workbench Architecture](reference-review-workbench-architecture.md)
- [Reference Review Fixture Source Contract](reference-review-fixture-source-contract.md)
- [Reference Review Async Concurrency](reference-review-async-concurrency.md)
- [Reference Review Promotion And Notes Lifecycle](reference-review-promotion-and-notes-lifecycle.md)

## Sandbox Principle

The sidecar is sandboxed by capability. It does not receive general developer
agent tools. It receives selected fixture context and can only call named
workbench tools through a broker.

## Allowed Tools

- `read_fixture_context`
- `read_allowed_source`
- `write_candidate_model`
- `write_candidate_note_patch`
- `request_candidate_regeneration`
- `list_candidate_outputs`
- `explain_blocking_diagnostic`

## Forbidden Actions

- arbitrary shell execution
- arbitrary filesystem read or write
- network access
- direct mutation of tests outside candidate roots
- direct mutation of product code
- writing or deleting gold artifacts
- promotion
- git operations

## Candidate Store

Candidate source files are written under:

```text
tests/reference_model_candidates/<fixture-id>/<candidate-name>.py
```

Candidate note patches are written through the notes API and marked as
assistant-suggested until a human accepts them.

## Context Injection

Codex receives:

- fixture id
- source model record
- feature description
- expected output
- owning test/spec links
- current diagnostics
- artifact summary
- allowed read/write roots
- tool policy

Codex does not receive full local environment dumps, full chat logs, or
unrelated repository files.

## Specification Manifest For Discovery

## Manifest Review History

- 2026-05-31 loop 1: Critical review found the original candidate mixed
  context, tool policy, candidate storage, notes, regeneration, process
  boundaries, audit, and UI work.
- 2026-05-31 loop 2: Rescored after moving chat/candidate UI to the UI
  manifest.
- 2026-05-31 loop 3: Split process boundary from tool policy because in-process
  Python restrictions are not a security boundary.
- 2026-05-31 loop 4: Added structured audit events as a first-class leaf.
- 2026-05-31 loop 5: Final review confirmed all candidates are below the split
  threshold and have explicit authority boundaries.

### Candidate Spec: Codex Fixture Context Builder

Discovery purpose:
- Build the smallest useful fixture context payload for Codex without exposing
  unrelated local files, environment state, or chat history.

Responsibilities:
- Functions/methods:
  - context payload builder
  - context redactor
- Data structures/models:
  - Codex context payload
  - context omission diagnostic
- Dependencies/services:
  - source context payload
  - review note store
- Returns/outputs/signals:
  - sanitized Codex context
  - omission diagnostic
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: deterministic review context payload
  - Additions to existing reusable library/module: Codex context adapter
  - New reusable library/module to create: Codex sidecar broker
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - context build is fixture-scoped and stale-guarded
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - no full local environment, full chat log, or unrelated repo files
- Performance-sensitive behavior:
  - bounded payload size
- Cross-screen reusable behavior:
  - context feeds sidecar stream, candidate generation, and refusal diagnostics

Project readiness fields:
- Implementation owner/module:
  - future `codex_sidecar/context`
- Chosen defaults / parameters:
  - deny extra context unless explicitly supplied by fixture metadata
- Test strategy:
  - context minimization, redaction, and stale fixture context tests
- Data ownership:
  - broker owns Codex-facing context; source registry owns source context
- Routes:
  - selected fixture context to Codex context payload
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
- Destructive/write behavior: 0 x 3 = 0
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
- No split needed. This is high but cohesive around the Codex-facing context
  boundary.

### Candidate Spec: Tool Policy Validator And Broker

Discovery purpose:
- Validate every sidecar tool request against an explicit deny-by-default
  allowlist.

Responsibilities:
- Functions/methods:
  - tool policy validator
  - broker request router
- Data structures/models:
  - tool policy record
  - tool request record
- Dependencies/services:
  - candidate store protocol
  - regeneration protocol
- Returns/outputs/signals:
  - accepted tool request
  - refused tool call diagnostic
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: none
  - Additions to existing reusable library/module: Codex sidecar broker
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - tool calls are cancellable and stale-guarded per fixture
- Destructive/write behavior:
  - only routes to explicitly allowed write APIs
- Security/privacy-sensitive behavior:
  - no shell, no git, no gold writes, no promotion, strict root validation
- Performance-sensitive behavior:
  - per-fixture tool-call queue is bounded
- Cross-screen reusable behavior:
  - broker authority applies to chat, candidate generation, notes, and regen

Project readiness fields:
- Implementation owner/module:
  - future `codex_sidecar/tool_broker`
- Chosen defaults / parameters:
  - deny by default; every call requires explicit policy match
- Test strategy:
  - allowed candidate write, refused shell, refused git, refused promote, and
    outside-root write refusal
- Data ownership:
  - broker owns sidecar authority
- Routes:
  - sidecar tool request to validator to allowed service
- Open questions / nuance discovered:
  - exact Codex runtime API remains an integration detail
- Readiness blockers:
  - none

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 0 x 0.5 = 0
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
- Total: 24

Split decision:
- No split needed. Policy validation and broker routing are one capability
  boundary; splitting them would make authority decisions harder to audit.

### Candidate Spec: Candidate Model Store

Discovery purpose:
- Write generated candidate model files only under approved candidate roots and
  expose them as non-promoted review sources.

Responsibilities:
- Functions/methods:
  - candidate model writer
  - candidate root verifier
- Data structures/models:
  - candidate model record
  - candidate write result
- Dependencies/services:
  - generated source resolver
  - async durable write lane
- Returns/outputs/signals:
  - candidate file path
  - refused candidate write diagnostic
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: generated source contract
  - Additions to existing reusable library/module: Codex sidecar broker
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - candidate writes use serialized write lane
- Destructive/write behavior:
  - writes candidate files only
- Security/privacy-sensitive behavior:
  - strict root validation; candidates cannot overwrite gold references
- Performance-sensitive behavior:
  - bounded candidate file size
- Cross-screen reusable behavior:
  - candidates feed preview, candidate list, and adoption workflow

Project readiness fields:
- Implementation owner/module:
  - future `codex_sidecar/candidates`
- Chosen defaults / parameters:
  - generated candidates are never promoted directly
- Test strategy:
  - allowed candidate write, outside-root refusal, overwrite policy, and source
    resolver handoff tests
- Data ownership:
  - candidate store owns candidate files; human reviewer owns adoption
- Routes:
  - tool request to candidate writer to generated source resolver
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
- No split needed. Candidate writes and candidate-source exposure are one
  bounded store boundary.

### Candidate Spec: Candidate Note Patch Route

Discovery purpose:
- Let Codex propose note patches without writing durable notes directly.

Responsibilities:
- Functions/methods:
  - candidate note patch writer
  - note patch validator
- Data structures/models:
  - candidate note patch
  - patch validation result
- Dependencies/services:
  - note store protocol
  - tool broker
- Returns/outputs/signals:
  - candidate note patch
  - refused patch diagnostic
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: note record shape
  - Additions to existing reusable library/module: Codex sidecar broker
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - patch proposals are stale-guarded per fixture
- Destructive/write behavior:
  - no durable note write until human adoption
- Security/privacy-sensitive behavior:
  - patch cannot include full chat logs by default
- Performance-sensitive behavior:
  - bounded patch size
- Cross-screen reusable behavior:
  - patch feeds Codex panel and notes panel adoption route

Project readiness fields:
- Implementation owner/module:
  - future `codex_sidecar/note_patches`
- Chosen defaults / parameters:
  - human adoption required for note changes
- Test strategy:
  - patch creation, oversized patch refusal, stale patch refusal, adoption handoff
- Data ownership:
  - broker owns proposed patch; note store owns durable notes
- Routes:
  - sidecar patch request to candidate patch to human adoption
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
- No split needed. Note patch validation and routing are cohesive because the
  broker must validate the patch before any candidate note state is exposed.

### Candidate Spec: Regeneration Request Route

Discovery purpose:
- Allow the sidecar to request artifact regeneration for a selected source or
  candidate without executing promotion.

Responsibilities:
- Functions/methods:
  - regeneration request router
  - regeneration eligibility validator
- Data structures/models:
  - regeneration request
  - regeneration request result
- Dependencies/services:
  - artifact regeneration service
  - async dispatcher
- Returns/outputs/signals:
  - regeneration request id
  - refused regeneration diagnostic
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: artifact regeneration command shape
  - Additions to existing reusable library/module: Codex sidecar broker
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - routed through dispatcher and stale-guarded by fixture
- Destructive/write behavior:
  - writes dirty artifacts only through regeneration service
- Security/privacy-sensitive behavior:
  - cannot write gold artifacts or call promotion APIs
- Performance-sensitive behavior:
  - per-fixture regeneration queue is bounded
- Cross-screen reusable behavior:
  - route feeds artifacts panel and candidate preview

Project readiness fields:
- Implementation owner/module:
  - future `codex_sidecar/regeneration`
- Chosen defaults / parameters:
  - regeneration is allowed only for current selected fixture/candidate
- Test strategy:
  - allowed regeneration, stale fixture refusal, candidate regeneration, promote
    refusal
- Data ownership:
  - regeneration service owns dirty artifacts
- Routes:
  - sidecar request to dispatcher to regeneration service
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
- No split needed. Regeneration request validation and dispatch are one brokered
  route from sidecar intent to controlled workbench execution.

### Candidate Spec: Sidecar Process Boundary And Audit

Discovery purpose:
- Keep Codex execution/tooling out of trusted UI process authority and emit
  structured audit events for every sidecar action and refusal.

Responsibilities:
- Functions/methods:
  - sidecar process launcher
  - sidecar audit emitter
- Data structures/models:
  - sidecar session record
  - sidecar audit event
- Dependencies/services:
  - async dispatcher
  - structured logging
- Returns/outputs/signals:
  - sidecar session started
  - sidecar audit event
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: structured task audit event shape
  - Additions to existing reusable library/module: Codex sidecar broker
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - sidecar session is cancellable and fixture-scoped
- Destructive/write behavior:
  - writes audit events only
- Security/privacy-sensitive behavior:
  - in-process Python restrictions are not treated as a security boundary
- Performance-sensitive behavior:
  - audit event size is bounded
- Cross-screen reusable behavior:
  - audit events support Codex UI, release reports, and troubleshooting

Project readiness fields:
- Implementation owner/module:
  - future `codex_sidecar/process_boundary`
- Chosen defaults / parameters:
  - sidecar may request tools but cannot directly mutate project state
- Test strategy:
  - sidecar cancellation, audit event, refused direct write, and process failure
    tests
- Data ownership:
  - broker owns sidecar authority and audit records
- Routes:
  - UI sidecar request to sidecar process to tool broker to audit stream
- Open questions / nuance discovered:
  - exact Codex runtime embedding remains an implementation choice
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
- No split needed. Process boundary and audit are coupled by the authority
  model.

## Change History

- 2026-05-31: Ran five critical review, rescore, and split passes over the
  specification manifest. Split Codex context, tool policy, candidate store,
  note patches, regeneration, and process/audit boundaries into separate
  leaves.
- 2026-05-30: Split Codex sidecar sandbox into its own architecture document
  with allowlisted tools, candidate store, forbidden actions, and human
  promotion boundary.
