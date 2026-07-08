# Reference Review Spec 60: Preview Stale Success And Failure Rejection Tests (v1.0)

## Overview

Verify that stale payload successes and stale payload failures cannot mutate a newer selected fixture's preview state.

## Backlink

- [Architecture: Reference Review Preview Payload Boundary Architecture](../architecture/reference-review-preview-payload-boundary-architecture.md)

## Source Manifest

- This leaf is derived from the manifest candidate `Preview Stale Success And Failure Rejection Tests`.
- Manifest score: 21.5

## Scope

This specification covers:

- Verify that stale payload successes and stale payload failures cannot mutate a newer selected fixture's preview state.
- the data structures, helpers, diagnostics, and verification gates named by the manifest
- the exact implementation boundary needed by the owning architecture document

## Responsibilities

- Functions/methods:
  - stale-result test harness
- Data structures/models:
  - fixture switch scenario record
- Dependencies/services:
  - preview payload controller
  - preview pane
- Returns/outputs/signals:
  - stale-result test pass/fail
- UI surfaces/components:
  - preview pane
- UI fields/elements:
  - loading and failure diagnostics
- Reusable code plan:
  - Existing code reused as-is:
    - Qt test harness
  - Additions to existing reusable library/module:
    - preview async tests
  - New reusable library/module to create:
    - none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - validates request id rejection and stale error handling
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - diagnostics redaction remains intact in failure paths
- Performance-sensitive behavior:
  - verifies stale results do not trigger extra scene application
- Cross-screen reusable behavior:
  - protects preview, artifact comparison, and promotion readiness

## Implementation Boundary

Owner/module:

- future preview async tests

Routes:

- test selection changes to payload controller to pane state

Reuse/extraction decision:

- Existing code reused as-is:
  - Qt test harness
- Additions to existing reusable library/module:
  - preview async tests
- New reusable library/module to create:
  - none

UI field/control inventory:

- loading and failure diagnostics

## Data And Defaults

Chosen defaults / parameters:

- fixture B selection wins over fixture A completion

Data ownership:

- tests own verification scenarios only

Open questions and resolved assumptions:

- none

Implementation prerequisites:

- preview payload UI handoff route

## Behavior

The implementation must:

- Verify that stale payload successes and stale payload failures cannot mutate a newer selected fixture's preview state.
- preserve the shared-preview boundary so the workbench does not grow a separate renderer path
- preserve renderer, payload, and UI-thread ownership described by the architecture
- emit deterministic diagnostics for invalid, unsupported, unsafe, or stale preview states

## Verification

Test strategy:

- controlled rapid fixture switch, stale success, and stale failure tests

Additional verification requirements:

- add focused unit coverage for the records, helpers, or routes introduced by this specification
- add regression coverage for stale-result, cancellation, diagnostic, or renderer-lifetime behavior when this leaf touches those paths
- ensure real `.impress` fixtures remain the preferred manual smoke path for interactive preview behavior

## Manifest Assessment

Score:

- Functions/methods: 1 x 2 = 2
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 1 x 1 = 1
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 2 x 1 = 2
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 1 x 2 = 2
- Total: 21.5

Split decision:

- No split needed. Cohesive stale-result verification leaf.

## Refinement Status

Final, with prerequisite dependencies listed below before implementation starts.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- the Preview Stale Success And Failure Rejection Tests boundary is implemented as described
- the specified ownership, routing, and reuse boundaries are preserved
- the listed test strategy passes for this leaf
- listed implementation prerequisites are satisfied or carried as explicit blocking failures before execution
