# Reference Review Spec 56: Preview Widget Payload Application (v1.0)

## Overview

Let the preview widget accept a prepared payload and apply it through the shared preview controller without recreating the renderer.

## Backlink

- [Architecture: Reference Review Preview Qt Wrapper Architecture](../architecture/reference-review-preview-qt-wrapper-architecture.md)

## Source Manifest

- This leaf is derived from the manifest candidate `Preview Widget Payload Application`.
- Manifest score: 24.5

## Scope

This specification covers:

- Let the preview widget accept a prepared payload and apply it through the shared preview controller without recreating the renderer.
- the data structures, helpers, diagnostics, and verification gates named by the manifest
- the exact implementation boundary needed by the owning architecture document

## Responsibilities

- Functions/methods:
  - `set_preview_payload`
  - `clear_preview`
  - payload generation check
- Data structures/models:
  - payload generation id
  - widget payload state
- Dependencies/services:
  - shared preview controller
  - preview payload record
- Returns/outputs/signals:
  - preview ready signal
  - preview failed signal
- UI surfaces/components:
  - preview widget
- UI fields/elements:
  - preview viewport
- Reusable code plan:
  - Existing code reused as-is:
    - shared preview controller
  - Additions to existing reusable library/module:
    - none
  - New reusable library/module to create:
    - none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - payload application happens on Qt UI thread
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - scene replacement reuses the existing renderer
- Cross-screen reusable behavior:
  - preview payload state feeds preview readiness

## Implementation Boundary

Owner/module:

- future `ui/preview_widget.py`

Routes:

- preview pane to widget to shared preview controller

Reuse/extraction decision:

- Existing code reused as-is:
  - shared preview controller
- Additions to existing reusable library/module:
  - none
- New reusable library/module to create:
  - none

UI field/control inventory:

- preview viewport

## Data And Defaults

Chosen defaults / parameters:

- stale payloads are rejected before widget application

Data ownership:

- widget owns current payload generation after pane validation

Open questions and resolved assumptions:

- payload shape depends on preview payload boundary spec

Implementation prerequisites:

- shared preview controller extraction

## Behavior

The implementation must:

- Let the preview widget accept a prepared payload and apply it through the shared preview controller without recreating the renderer.
- preserve the shared-preview boundary so the workbench does not grow a separate renderer path
- preserve renderer, payload, and UI-thread ownership described by the architecture
- emit deterministic diagnostics for invalid, unsupported, unsafe, or stale preview states

## Verification

Test strategy:

- mocked payload handoff and scene replacement tests

Additional verification requirements:

- add focused unit coverage for the records, helpers, or routes introduced by this specification
- add regression coverage for stale-result, cancellation, diagnostic, or renderer-lifetime behavior when this leaf touches those paths
- ensure real `.impress` fixtures remain the preferred manual smoke path for interactive preview behavior

## Manifest Assessment

Score:

- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
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
- Total: 24.5

Split decision:

- No split needed. Cohesion reason: widget payload acceptance, clear, ready,
  and failed signals are one scene-replacement boundary; stale-result ownership
  belongs to the payload-boundary architecture.

## Refinement Status

Final, with prerequisite dependencies listed below before implementation starts.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- the Preview Widget Payload Application boundary is implemented as described
- the specified ownership, routing, and reuse boundaries are preserved
- the listed test strategy passes for this leaf
- listed implementation prerequisites are satisfied or carried as explicit blocking failures before execution
