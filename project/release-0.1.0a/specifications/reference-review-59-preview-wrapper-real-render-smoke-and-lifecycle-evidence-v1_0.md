# Reference Review Spec 59: Preview Wrapper Real-Render Smoke And Lifecycle Evidence (v1.0)

## Overview

Define the verification evidence needed to prove the embedded wrapper uses a stable live renderer with real `.impress` fixtures.

## Backlink

- [Architecture: Reference Review Preview Qt Wrapper Architecture](../architecture/reference-review-preview-qt-wrapper-architecture.md)

## Source Manifest

- This leaf is derived from the manifest candidate `Preview Wrapper Real-Render Smoke And Lifecycle Evidence`.
- Manifest score: 20.5

## Scope

This specification covers:

- Define the verification evidence needed to prove the embedded wrapper uses a stable live renderer with real `.impress` fixtures.
- the data structures, helpers, diagnostics, and verification gates named by the manifest
- the exact implementation boundary needed by the owning architecture document

## Responsibilities

- Functions/methods:
  - manual smoke command
  - lifecycle evidence capture
- Data structures/models:
  - smoke result record
- Dependencies/services:
  - test fixtures
  - workbench launcher
- Returns/outputs/signals:
  - smoke pass/fail note
- UI surfaces/components:
  - preview pane
- UI fields/elements:
  - preview viewport
- Reusable code plan:
  - Existing code reused as-is:
    - workbench launcher and fixtures
  - Additions to existing reusable library/module:
    - none
  - New reusable library/module to create:
    - none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - verifies launch, interaction, fixture switch, and shutdown ordering
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - verifies renderer is not recreated per interaction
- Cross-screen reusable behavior:
  - smoke evidence supports review readiness

## Implementation Boundary

Owner/module:

- future paired test specification

Routes:

- launcher to fixture to preview pane to widget

Reuse/extraction decision:

- Existing code reused as-is:
  - workbench launcher and fixtures
- Additions to existing reusable library/module:
  - none
- New reusable library/module to create:
  - none

UI field/control inventory:

- preview viewport

## Data And Defaults

Chosen defaults / parameters:

- use dirty `.impress` fixtures, not demo PNG/STL snapshot smoke

Data ownership:

- evidence belongs to test/review artifacts

Open questions and resolved assumptions:

- none

Implementation prerequisites:

- preview widget lifecycle

## Behavior

The implementation must:

- Define the verification evidence needed to prove the embedded wrapper uses a stable live renderer with real `.impress` fixtures.
- preserve the shared-preview boundary so the workbench does not grow a separate renderer path
- preserve renderer, payload, and UI-thread ownership described by the architecture
- emit deterministic diagnostics for invalid, unsupported, unsafe, or stale preview states

## Verification

Test strategy:

- manual real-render smoke plus focused lifecycle tests

Additional verification requirements:

- add focused unit coverage for the records, helpers, or routes introduced by this specification
- add regression coverage for stale-result, cancellation, diagnostic, or renderer-lifetime behavior when this leaf touches those paths
- ensure real `.impress` fixtures remain the preferred manual smoke path for interactive preview behavior

## Manifest Assessment

Score:

- Functions/methods: 2 x 2 = 4
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 1 x 1 = 1
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 0 x 1 = 0
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 1 x 1 = 1
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 1 x 2 = 2
- Total: 20.5

Split decision:

- No split needed. Cohesive verification leaf for the wrapper's renderer
  lifetime and real-fixture behavior.

## Refinement Status

Final, with prerequisite dependencies listed below before implementation starts.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- the Preview Wrapper Real-Render Smoke And Lifecycle Evidence boundary is implemented as described
- the specified ownership, routing, and reuse boundaries are preserved
- the listed test strategy passes for this leaf
- listed implementation prerequisites are satisfied or carried as explicit blocking failures before execution
