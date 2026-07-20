# Reference Review Spec 51: Preview Payload Process Controller (v1.0)

## Overview

Add the Qt-side controller that starts payload work, tracks active request identity, and captures process diagnostics.

## Backlink

- [Architecture: Reference Review Preview Payload Boundary Architecture](../architecture/reference-review-preview-payload-boundary-architecture.md)

## Source Manifest

- This leaf is derived from the manifest candidate `Preview Payload Process Controller`.
- Manifest score: 22.5

## Scope

This specification covers:

- Add the Qt-side controller that starts payload work, tracks active request identity, and captures process diagnostics.
- the data structures, helpers, diagnostics, and verification gates named by the manifest
- the exact implementation boundary needed by the owning architecture document

## Responsibilities

- Functions/methods:
  - preview payload request launcher
  - process diagnostic collector
- Data structures/models:
  - controller diagnostic record
- Dependencies/services:
  - Qt process or worker controller
  - preview payload records
- Returns/outputs/signals:
  - controller diagnostic
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is:
    - async envelope pattern
  - Additions to existing reusable library/module:
    - async controller preview route
  - New reusable library/module to create:
    - none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - owner/request id tracking and stdout/stderr capture
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - diagnostics redact unsafe environment details
- Performance-sensitive behavior:
  - controller does not decode or tessellate payloads on UI thread
- Cross-screen reusable behavior:
  - controller events feed preview handoff route

## Implementation Boundary

Owner/module:

- future preview route in async controller

Routes:

- selection to controller to builder to pane handoff

Reuse/extraction decision:

- Existing code reused as-is:
  - async envelope pattern
- Additions to existing reusable library/module:
  - async controller preview route
- New reusable library/module to create:
  - none

UI field/control inventory:

- none

## Data And Defaults

Chosen defaults / parameters:

- one active selected-fixture preview payload build at a time

Data ownership:

- controller owns active request identity and diagnostics

Open questions and resolved assumptions:

- exact worker technology remains behind controller policy

Implementation prerequisites:

- payload records

## Behavior

The implementation must:

- Add the Qt-side controller that starts payload work, tracks active request identity, and captures process diagnostics.
- preserve the shared-preview boundary so the workbench does not grow a separate renderer path
- preserve renderer, payload, and UI-thread ownership described by the architecture
- emit deterministic diagnostics for invalid, unsupported, unsafe, or stale preview states

## Verification

Test strategy:

- success, failure, stdout/stderr, and process-state tests

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
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 1 x 2 = 2
- Total: 22.5

Split decision:

- No split needed. Cohesive controller/process-supervision leaf; handoff and
  cleanup are separate candidates.

## Refinement Status

Final, with prerequisite dependencies listed below before implementation starts.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- the Preview Payload Process Controller boundary is implemented as described
- the specified ownership, routing, and reuse boundaries are preserved
- the listed test strategy passes for this leaf
- listed implementation prerequisites are satisfied or carried as explicit blocking failures before execution
