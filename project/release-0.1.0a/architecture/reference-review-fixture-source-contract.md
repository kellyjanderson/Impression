# Reference Review Fixture Source Contract

## Overview

This document defines how reference tests expose the model under test to the
Reference Review Workbench.

The workbench tests modeling features, not PNG or STL generation. A reviewable
fixture must therefore provide a loadable source model or deterministic fixture
entrypoint that can be opened in the real-time preview.

## Parent Architecture

- [Reference Review Workbench Architecture](reference-review-workbench-architecture.md)

## Source Model Rule

Every reference fixture must expose one of these source forms:

- standalone Python model module with a preview-compatible `build()` entrypoint
- deterministic fixture producer callable with declared parameters
- generated review model module produced from a fixture contract before review

Generated PNG, STL, slice, and diagnostic files are derived evidence. They do
not make a fixture reviewable unless the source model can also be loaded.

## Loadable Source Record

Each fixture has a `ReviewSourceModelRecord`:

- `fixture_id`
- `feature_name`
- `description`
- `expected_output`
- `load_mode`: `module`, `callable`, or `generated-review-module`
- `model_path`
- `entrypoint`
- `parameters`
- `determinism_inputs`
- `owning_test_path`
- `owning_spec_paths`
- `generation_command`
- `artifact_paths`
- `preview_interaction_contract`

The record is the source of truth for both the UI context panel and the Codex
sidecar context payload.

## Fixture Authoring Requirements

- Inline test-only model construction must be extracted into a reusable fixture
  module before the fixture can enter review.
- Randomness, temporary data, environment variables, and generated inputs must
  be captured in `determinism_inputs`.
- Callable fixtures must declare argument defaults rather than relying on test
  globals.
- The same source record must drive artifact regeneration and review preview.
- Missing or non-loadable source records are blocking review diagnostics.

## Data Flow

```text
test fixture or reference manifest
-> ReviewSourceModelRecord
-> source resolver
-> model loader
-> preview bridge
-> artifact regeneration
```

## Specification Manifest For Discovery

## Manifest Review History

- 2026-05-31 loop 1: Critical review found the original single candidate hid
  schema, validation, discovery, determinism, and generated-module work.
- 2026-05-31 loop 2: Rescored the split candidates and added privacy,
  performance, and reusable registry responsibilities where undercounted.
- 2026-05-31 loop 3: Split deterministic context from source schema because it
  feeds Codex, promotion, and diagnostics differently than the loader contract.
- 2026-05-31 loop 4: Added generated review module support as its own leaf
  because candidate modules need different roots and lifecycle rules.
- 2026-05-31 loop 5: Final review confirmed every candidate is below the split
  threshold and 16-24 candidates have cohesion rationale.

### Candidate Spec: Review Source Model Record Schema

Discovery purpose:
- Define the typed record that maps a reference fixture to the loadable model
  source used by the review workbench.

Responsibilities:
- Functions/methods:
  - source record parser
  - source record normalizer
- Data structures/models:
  - `ReviewSourceModelRecord`
  - source identity value object
  - entrypoint parameter record
- Dependencies/services:
  - reference fixture metadata
- Returns/outputs/signals:
  - normalized source record
  - schema diagnostic
- UI surfaces/components:
  - none; UI display belongs to the context-panel spec
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: preview-compatible model entrypoint convention
  - Additions to existing reusable library/module: reference fixture helpers
  - New reusable library/module to create: review source registry types
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none; callable from discovery workers
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - source paths are normalized before display or Codex context use
- Performance-sensitive behavior:
  - bounded parsing with no model import
- Cross-screen reusable behavior:
  - same record feeds queue, preview, notes, promotion, and Codex panes

Project readiness fields:
- Implementation owner/module:
  - future `src/impression/devtools/reference_review/source_registry`
- Chosen defaults / parameters:
  - every reviewable fixture must define a loadable source record
- Test strategy:
  - schema acceptance and rejection tests for path, module, callable, and
    parameter forms
- Data ownership:
  - fixture source contract owns source identity
- Routes:
  - fixture id to source record
- Open questions / nuance discovered:
  - none
- Readiness blockers:
  - none

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 1 x 1 = 1
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
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 0 x 2 = 0
- Total: 21.5

Split decision:
- No split needed. This is a cohesive schema leaf after moving presentation and
  execution work to separate candidates.

### Candidate Spec: Source Record Validation And Diagnostics

Discovery purpose:
- Validate source records without executing the model and report every blocking
  fixture problem in one pass.

Responsibilities:
- Functions/methods:
  - source record validator
  - aggregate diagnostic reporter
- Data structures/models:
  - validation diagnostic
  - validation result envelope
- Dependencies/services:
  - source record schema
  - filesystem path policy
- Returns/outputs/signals:
  - valid result
  - source validation diagnostics
- UI surfaces/components:
  - none; presentation belongs to UI panel specs
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: path helper conventions
  - Additions to existing reusable library/module: reference fixture helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - runs in discovery worker and returns typed diagnostics
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - diagnostics redact unrelated absolute paths
- Performance-sensitive behavior:
  - validation never imports or tessellates the model
- Cross-screen reusable behavior:
  - diagnostic objects are reusable by queue, context, and Codex context summary

Project readiness fields:
- Implementation owner/module:
  - future `source_registry.validation`
- Chosen defaults / parameters:
  - invalid source records block review execution but do not stop full scan
- Test strategy:
  - missing path, missing callable, bad parameters, and multi-error aggregation
- Data ownership:
  - validator owns source-record diagnostics
- Routes:
  - source record to validation result to queue/context UI
- Open questions / nuance discovered:
  - none
- Readiness blockers:
  - none

Score:
- Functions/methods: 3 x 2 = 6
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
- Total: 23.5

Split decision:
- No split needed. Diagnostic presentation is intentionally not included in
  this service leaf.

### Candidate Spec: Fixture Discovery Integration

Discovery purpose:
- Connect active reference fixture roots to source-record discovery without
  loading models or relying on derived PNG/STL artifacts.

Responsibilities:
- Functions/methods:
  - fixture root scanner
  - fixture-to-source resolver
- Data structures/models:
  - discovery item
  - discovery summary
- Dependencies/services:
  - source record schema
  - reference artifact root helpers
- Returns/outputs/signals:
  - discovered fixture list
  - skipped fixture diagnostic
- UI surfaces/components:
  - none; queue rendering belongs to UI specs
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: reference artifact root helpers
  - Additions to existing reusable library/module: fixture discovery helper
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - discovery runs in bounded worker task
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - scan is restricted to configured reference roots
- Performance-sensitive behavior:
  - incremental scan and bounded stat calls
- Cross-screen reusable behavior:
  - discovery list feeds queue, release gate, and review reports

Project readiness fields:
- Implementation owner/module:
  - future `source_registry.discovery`
- Chosen defaults / parameters:
  - derived artifacts without source records are skipped with diagnostics
- Test strategy:
  - discovery tests for clean, dirty, missing source, duplicate fixture id, and
    non-reference-root fixtures
- Data ownership:
  - discovery owns fixture list, not review decisions
- Routes:
  - reference roots to discovery items to queue model
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
- No split needed. Queue display mapping is owned by the UI manifest.

### Candidate Spec: Deterministic Review Context Payload

Discovery purpose:
- Define the fixture context payload used by preview, notes, promotion, and
  Codex without exposing unrelated local environment state.

Responsibilities:
- Functions/methods:
  - context payload builder
  - context sanitizer
- Data structures/models:
  - review context payload
  - expected output summary
  - determinism input record
- Dependencies/services:
  - source record schema
  - fixture metadata
- Returns/outputs/signals:
  - sanitized context payload
  - context omission diagnostic
- UI surfaces/components:
  - none; Markdown/context rendering belongs to UI specs
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: fixture metadata conventions
  - Additions to existing reusable library/module: source registry context API
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - context build can run in discovery worker
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - omits secrets, unrelated paths, and full environment dumps
- Performance-sensitive behavior:
  - no model execution during context build
- Cross-screen reusable behavior:
  - payload feeds context UI, Codex, promotion provenance, and notes

Project readiness fields:
- Implementation owner/module:
  - future `source_registry.context`
- Chosen defaults / parameters:
  - context is fixture-scoped and source-derived
- Test strategy:
  - context payload snapshot tests and secret/path redaction tests
- Data ownership:
  - source registry owns context payload construction
- Routes:
  - source record plus fixture metadata to context payload
- Open questions / nuance discovered:
  - none
- Readiness blockers:
  - none

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 3 x 1 = 3
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
- Total: 22.5

Split decision:
- No split needed. The payload and sanitizer are cohesive because they define
  the same safe context boundary.

### Candidate Spec: Generated Review Module Contract

Discovery purpose:
- Define how candidate or generated review modules can be loaded by the
  workbench without confusing them with committed source fixtures.

Responsibilities:
- Functions/methods:
  - generated module resolver
  - candidate root verifier
- Data structures/models:
  - generated source reference
  - candidate source lifecycle state
- Dependencies/services:
  - source registry
  - Codex candidate store
- Returns/outputs/signals:
  - generated source record
  - refused generated source diagnostic
- UI surfaces/components:
  - none; candidate list presentation belongs to Codex/UI specs
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: source record schema
  - Additions to existing reusable library/module: generated-source adapter
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - candidate resolution runs through source discovery worker
- Destructive/write behavior:
  - none in resolver; writes belong to Codex candidate store
- Security/privacy-sensitive behavior:
  - candidate modules must live under allowed generated roots
- Performance-sensitive behavior:
  - no broad import scanning
- Cross-screen reusable behavior:
  - candidate source records feed preview, Codex, and adoption UI

Project readiness fields:
- Implementation owner/module:
  - future `source_registry.generated`
- Chosen defaults / parameters:
  - generated modules are never promoted directly as gold evidence
- Test strategy:
  - generated-root acceptance, outside-root refusal, stale candidate rejection
- Data ownership:
  - source registry identifies candidates; Codex store owns candidate files
- Routes:
  - candidate file to generated source record to preview
- Open questions / nuance discovered:
  - none
- Readiness blockers:
  - none; integrate against a candidate-store protocol stub in tests

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
- No split needed. This leaf only resolves candidate source records; candidate
  writes and adoption remain in the Codex sandbox manifest.

## Change History

- 2026-05-31: Ran five critical review, rescore, and split passes over the
  specification manifest. Split the original broad source-record candidate into
  schema, validation, discovery, deterministic context, and generated-module
  leaves.
- 2026-05-30: Split fixture source responsibilities out of the parent
  workbench architecture and made loadable source models mandatory.
