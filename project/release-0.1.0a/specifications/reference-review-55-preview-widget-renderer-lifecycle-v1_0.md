# Reference Review Spec 55: Preview Widget Renderer Lifecycle (v1.0)

## Overview

Add the Qt widget host that owns one long-lived embedded render surface and disposes it only with the widget lifecycle.

## Backlink

- [Architecture: Reference Review Preview Qt Wrapper Architecture](../architecture/reference-review-preview-qt-wrapper-architecture.md)

## Source Manifest

- This leaf is derived from the manifest candidate `Preview Widget Renderer Lifecycle`.
- Manifest score: 23.5

## Scope

This specification covers:

- Add the Qt widget host that owns one long-lived embedded render surface and disposes it only with the widget lifecycle.
- the data structures, helpers, diagnostics, and verification gates named by the manifest
- the exact implementation boundary needed by the owning architecture document

## Responsibilities

- Functions/methods:
  - widget initialization
  - render surface creation
  - clear/dispose methods
- Data structures/models:
  - renderer lifecycle state
- Dependencies/services:
  - PySide6 widgets
  - PyVistaQt render surface
- Returns/outputs/signals:
  - lifecycle diagnostic
- UI surfaces/components:
  - preview widget
- UI fields/elements:
  - preview viewport
- Reusable code plan:
  - Existing code reused as-is:
    - shared preview controller from
      `impression.preview`
  - Additions to existing reusable library/module:
    - none
  - New reusable library/module to create:
    - workbench preview widget module
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - renderer mutation happens only on the Qt UI thread
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - renderer is long-lived and not recreated per frame or fixture
- Cross-screen reusable behavior:
  - none

## Implementation Boundary

Owner/module:

- future `src/impression/devtools/reference_review/ui/preview_widget.py`

Routes:

- preview pane creates widget; widget creates renderer

Reuse/extraction decision:

- Existing code reused as-is:
  - shared preview controller from
    `impression.preview`
- Additions to existing reusable library/module:
  - none
- New reusable library/module to create:
  - workbench preview widget module

UI field/control inventory:

- preview viewport

## Data And Defaults

Chosen defaults / parameters:

- dark blue background and light orange object color supplied through shared
  preview style

Data ownership:

- widget owns renderer lifecycle only

Open questions and resolved assumptions:

- offscreen Qt tests may need a fake render surface because VTK interactor
  is unstable on offscreen platforms

Implementation prerequisites:

- shared preview controller extraction

## Behavior

The implementation must:

- Add the Qt widget host that owns one long-lived embedded render surface and disposes it only with the widget lifecycle.
- preserve the shared-preview boundary so the workbench does not grow a separate renderer path
- preserve renderer, payload, and UI-thread ownership described by the architecture
- emit deterministic diagnostics for invalid, unsupported, unsafe, or stale preview states

## Verification

Test strategy:

- Qt widget lifecycle tests and mocked render-surface tests

Additional verification requirements:

- add focused unit coverage for the records, helpers, or routes introduced by this specification
- add regression coverage for stale-result, cancellation, diagnostic, or renderer-lifetime behavior when this leaf touches those paths
- ensure real `.impress` fixtures remain the preferred manual smoke path for interactive preview behavior

## Manifest Assessment

Score:

- Functions/methods: 3 x 2 = 6
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 1 x 1 = 1
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 0 x 1 = 0
- Creating a new reusable library/module: 1 x 3 = 3
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 1 x 1 = 1
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 1 x 2 = 2
- Total: 23.5

Split decision:

- No split needed. Cohesion reason: renderer creation, stable ownership, and
  disposal are one lifecycle boundary.

## Refinement Status

Final, with prerequisite dependencies listed below before implementation starts.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- the Preview Widget Renderer Lifecycle boundary is implemented as described
- the specified ownership, routing, and reuse boundaries are preserved
- the listed test strategy passes for this leaf
- listed implementation prerequisites are satisfied or carried as explicit blocking failures before execution
