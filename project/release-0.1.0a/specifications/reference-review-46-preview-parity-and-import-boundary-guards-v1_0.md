# Reference Review Spec 46: Preview Parity And Import-Boundary Guards (v1.0)

## Overview

Add regression guards that prevent the workbench from regrowing a separate preview renderer or importing UI modules into shared preview code.

## Backlink

- [Architecture: Reference Review Preview Engine Sharing Architecture](../architecture/reference-review-preview-engine-sharing-architecture.md)

## Source Manifest

- This leaf is derived from the manifest candidate `Preview Parity And Import-Boundary Guards`.
- Manifest score: 15.5

## Scope

This specification covers:

- Add regression guards that prevent the workbench from regrowing a separate preview renderer or importing UI modules into shared preview code.
- the data structures, helpers, diagnostics, and verification gates named by the manifest
- the exact implementation boundary needed by the owning architecture document

## Responsibilities

- Functions/methods:
  - import-boundary check
  - duplicate-renderer scan
- Data structures/models:
  - guard report
- Dependencies/services:
  - test suite
  - shared preview controller
- Returns/outputs/signals:
  - test pass/fail diagnostic
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is:
    - current test harness
  - Additions to existing reusable library/module:
    - preview import-boundary tests
  - New reusable library/module to create:
    - none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - guards against duplicate heavy render paths
- Cross-screen reusable behavior:
  - protects CLI and workbench parity

## Implementation Boundary

Owner/module:

- future focused preview regression tests

Routes:

- test suite to preview modules

Reuse/extraction decision:

- Existing code reused as-is:
  - current test harness
- Additions to existing reusable library/module:
  - preview import-boundary tests
- New reusable library/module to create:
  - none

UI field/control inventory:

- none

## Data And Defaults

Chosen defaults / parameters:

- workbench UI must call shared preview controller for scene application

Data ownership:

- tests own boundary enforcement only

Open questions and resolved assumptions:

- none

Implementation prerequisites:

- shared controller extraction

## Behavior

The implementation must:

- Add regression guards that prevent the workbench from regrowing a separate preview renderer or importing UI modules into shared preview code.
- preserve the shared-preview boundary so the workbench does not grow a separate renderer path
- preserve renderer, payload, and UI-thread ownership described by the architecture
- emit deterministic diagnostics for invalid, unsupported, unsafe, or stale preview states

## Verification

Test strategy:

- import-boundary and duplicate-renderer guard tests

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
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 1 x 2 = 2
- Total: 15.5

Split decision:

- No split needed. Below split-review threshold and cohesive as a verification
  guard leaf.

## Refinement Status

Final, with prerequisite dependencies listed below before implementation starts.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- the Preview Parity And Import-Boundary Guards boundary is implemented as described
- the specified ownership, routing, and reuse boundaries are preserved
- the listed test strategy passes for this leaf
- listed implementation prerequisites are satisfied or carried as explicit blocking failures before execution
