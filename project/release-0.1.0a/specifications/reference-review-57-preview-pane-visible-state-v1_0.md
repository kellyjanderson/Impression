# Reference Review Spec 57: Preview Pane Visible State (v1.0)

## Overview

Keep workbench-visible preview state outside the render widget while still embedding the widget in the preview pane.

## Backlink

- [Architecture: Reference Review Preview Qt Wrapper Architecture](../architecture/reference-review-preview-qt-wrapper-architecture.md)

## Source Manifest

- This leaf is derived from the manifest candidate `Preview Pane Visible State`.
- Manifest score: 24.5

## Scope

This specification covers:

- Keep workbench-visible preview state outside the render widget while still embedding the widget in the preview pane.
- the data structures, helpers, diagnostics, and verification gates named by the manifest
- the exact implementation boundary needed by the owning architecture document

## Responsibilities

- Functions/methods:
  - preview pane state reducer
  - diagnostic display update
- Data structures/models:
  - preview pane state record
- Dependencies/services:
  - `ImpressionPreviewWidget`
  - workbench selection model
- Returns/outputs/signals:
  - toolbar enabled state
  - pane diagnostic state
- UI surfaces/components:
  - preview pane
- UI fields/elements:
  - placeholder
  - loading indicator
  - diagnostic banner
- Reusable code plan:
  - Existing code reused as-is:
    - workbench panel patterns
  - Additions to existing reusable library/module:
    - preview pane state model
  - New reusable library/module to create:
    - none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - pane state mutates only on Qt UI thread after owner/request checks
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - diagnostics avoid unsafe environment dumps
- Performance-sensitive behavior:
  - pane state changes do not recreate the renderer
- Cross-screen reusable behavior:
  - preview state feeds review readiness, artifact panels, and promotion state

## Implementation Boundary

Owner/module:

- future preview pane module or shell preview section

Routes:

- selection and payload controller events to pane state

Reuse/extraction decision:

- Existing code reused as-is:
  - workbench panel patterns
- Additions to existing reusable library/module:
  - preview pane state model
- New reusable library/module to create:
  - none

UI field/control inventory:

- placeholder
- loading indicator
- diagnostic banner

## Data And Defaults

Chosen defaults / parameters:

- toolbar disabled until widget is interactive

Data ownership:

- pane owns visible state; widget owns renderer state

Open questions and resolved assumptions:

- none

Implementation prerequisites:

- preview widget lifecycle

## Behavior

The implementation must:

- Keep workbench-visible preview state outside the render widget while still embedding the widget in the preview pane.
- preserve the shared-preview boundary so the workbench does not grow a separate renderer path
- preserve renderer, payload, and UI-thread ownership described by the architecture
- emit deterministic diagnostics for invalid, unsupported, unsafe, or stale preview states

## Verification

Test strategy:

- pane state transition tests

Additional verification requirements:

- add focused unit coverage for the records, helpers, or routes introduced by this specification
- add regression coverage for stale-result, cancellation, diagnostic, or renderer-lifetime behavior when this leaf touches those paths
- ensure real `.impress` fixtures remain the preferred manual smoke path for interactive preview behavior

## Manifest Assessment

Score:

- Functions/methods: 2 x 2 = 4
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 3 x 1 = 3
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 1 x 2 = 2
- Total: 24.5

Split decision:

- No split needed. Cohesion reason: placeholder, loading, and diagnostic states
  are one visible preview-pane state boundary; toolbar routing is separate.

## Refinement Status

Final, with prerequisite dependencies listed below before implementation starts.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- the Preview Pane Visible State boundary is implemented as described
- the specified ownership, routing, and reuse boundaries are preserved
- the listed test strategy passes for this leaf
- listed implementation prerequisites are satisfied or carried as explicit blocking failures before execution
