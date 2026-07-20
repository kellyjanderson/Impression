# Reference Review Spec 43: Shared Preview Controller API And Style Records (v1.0)

## Overview

Define the reusable preview-controller API and style/interaction records that CLI preview and the workbench wrapper will both call.

## Backlink

- [Architecture: Reference Review Preview Engine Sharing Architecture](../architecture/reference-review-preview-engine-sharing-architecture.md)

## Source Manifest

- This leaf is derived from the manifest candidate `Shared Preview Controller API And Style Records`.
- Manifest score: 22.5

## Scope

This specification covers:

- Define the reusable preview-controller API and style/interaction records that CLI preview and the workbench wrapper will both call.
- the data structures, helpers, diagnostics, and verification gates named by the manifest
- the exact implementation boundary needed by the owning architecture document

## Responsibilities

- Functions/methods:
  - preview controller constructor
  - plotter configuration method
  - style/default resolver
- Data structures/models:
  - preview style record
  - interaction policy record
  - controller options record
- Dependencies/services:
  - `impression.preview`
  - PyVista plotter protocol
- Returns/outputs/signals:
  - configured controller
  - style resolution diagnostic
- UI surfaces/components:
  - CLI preview host
  - workbench preview widget host
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is:
    - CLI preview defaults as source behavior
  - Additions to existing reusable library/module:
    - controller API in
      `impression.preview`
  - New reusable library/module to create:
    - none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none; caller owns thread affinity when using the controller
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - controller initialization is lightweight and does not create renderers
- Cross-screen reusable behavior:
  - shared by CLI preview and workbench preview

## Implementation Boundary

Owner/module:

- `src/impression/preview.py`

Routes:

- CLI host or workbench widget to shared preview controller

Reuse/extraction decision:

- Existing code reused as-is:
  - CLI preview defaults as source behavior
- Additions to existing reusable library/module:
  - controller API in
    `impression.preview`
- New reusable library/module to create:
  - none

UI field/control inventory:

- none

## Data And Defaults

Chosen defaults / parameters:

- workbench style uses dark blue background and light orange object color via
  style configuration

Data ownership:

- preview controller owns preview semantic configuration, not renderer
  lifecycle

Open questions and resolved assumptions:

- exact controller class names may change during implementation

Implementation prerequisites:

- none

## Behavior

The implementation must:

- Define the reusable preview-controller API and style/interaction records that CLI preview and the workbench wrapper will both call.
- preserve the shared-preview boundary so the workbench does not grow a separate renderer path
- preserve renderer, payload, and UI-thread ownership described by the architecture
- emit deterministic diagnostics for invalid, unsupported, unsafe, or stale preview states

## Verification

Test strategy:

- controller API construction tests and style default tests

Additional verification requirements:

- add focused unit coverage for the records, helpers, or routes introduced by this specification
- add regression coverage for stale-result, cancellation, diagnostic, or renderer-lifetime behavior when this leaf touches those paths
- ensure real `.impress` fixtures remain the preferred manual smoke path for interactive preview behavior

## Manifest Assessment

Score:

- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 2 x 2 = 4
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 0 x 2 = 0
- Total: 22.5

Split decision:

- No split needed. Cohesive API/configuration leaf; scene mutation, CLI
  migration, and guard tests are separate candidates.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- the Shared Preview Controller API And Style Records boundary is implemented as described
- the specified ownership, routing, and reuse boundaries are preserved
- the listed test strategy passes for this leaf
- listed implementation prerequisites are satisfied or carried as explicit blocking failures before execution
