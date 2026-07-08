# Reference Review Spec 58: Preview Toolbar Command Routing (v1.0)

## Overview

Route preview toolbar commands to the widget without letting the toolbar own camera or interaction semantics.

## Backlink

- [Architecture: Reference Review Preview Qt Wrapper Architecture](../architecture/reference-review-preview-qt-wrapper-architecture.md)

## Source Manifest

- This leaf is derived from the manifest candidate `Preview Toolbar Command Routing`.
- Manifest score: 21.5

## Scope

This specification covers:

- Route preview toolbar commands to the widget without letting the toolbar own camera or interaction semantics.
- the data structures, helpers, diagnostics, and verification gates named by the manifest
- the exact implementation boundary needed by the owning architecture document

## Responsibilities

- Functions/methods:
  - toolbar command router
  - command enablement resolver
- Data structures/models:
  - camera command record
- Dependencies/services:
  - `ImpressionPreviewWidget`
  - preview pane state
- Returns/outputs/signals:
  - toolbar enabled state
- UI surfaces/components:
  - preview pane
- UI fields/elements:
  - reset control
  - camera preset controls
- Reusable code plan:
  - Existing code reused as-is:
    - workbench toolbar patterns
  - Additions to existing reusable library/module:
    - none
  - New reusable library/module to create:
    - none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - commands execute on Qt UI thread
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - commands do not recreate renderer
- Cross-screen reusable behavior:
  - toolbar command state supports preview and promotion readiness

## Implementation Boundary

Owner/module:

- future preview pane module

Routes:

- toolbar action to pane router to widget method

Reuse/extraction decision:

- Existing code reused as-is:
  - workbench toolbar patterns
- Additions to existing reusable library/module:
  - none
- New reusable library/module to create:
  - none

UI field/control inventory:

- reset control
- camera preset controls

## Data And Defaults

Chosen defaults / parameters:

- toolbar disabled until widget is interactive

Data ownership:

- pane owns command state; shared preview controller owns command semantics

Open questions and resolved assumptions:

- exact preset list inherits current UI definition

Implementation prerequisites:

- preview pane visible state

## Behavior

The implementation must:

- Route preview toolbar commands to the widget without letting the toolbar own camera or interaction semantics.
- preserve the shared-preview boundary so the workbench does not grow a separate renderer path
- preserve renderer, payload, and UI-thread ownership described by the architecture
- emit deterministic diagnostics for invalid, unsupported, unsafe, or stale preview states

## Verification

Test strategy:

- command enablement and routing tests with a fake widget

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
- UI fields/elements: 2 x 1 = 2
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 1 x 2 = 2
- Total: 21.5

Split decision:

- No split needed. Cohesive command-routing leaf.

## Refinement Status

Final, with prerequisite dependencies listed below before implementation starts.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- the Preview Toolbar Command Routing boundary is implemented as described
- the specified ownership, routing, and reuse boundaries are preserved
- the listed test strategy passes for this leaf
- listed implementation prerequisites are satisfied or carried as explicit blocking failures before execution
