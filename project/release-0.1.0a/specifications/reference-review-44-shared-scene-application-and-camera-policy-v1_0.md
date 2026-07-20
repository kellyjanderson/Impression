# Reference Review Spec 44: Shared Scene Application And Camera Policy (v1.0)

## Overview

Move scene application, edge policy, and camera reset behavior into the shared preview controller without changing host lifecycle ownership.

## Backlink

- [Architecture: Reference Review Preview Engine Sharing Architecture](../architecture/reference-review-preview-engine-sharing-architecture.md)

## Source Manifest

- This leaf is derived from the manifest candidate `Shared Scene Application And Camera Policy`.
- Manifest score: 24.5

## Scope

This specification covers:

- Move scene application, edge policy, and camera reset behavior into the shared preview controller without changing host lifecycle ownership.
- the data structures, helpers, diagnostics, and verification gates named by the manifest
- the exact implementation boundary needed by the owning architecture document

## Responsibilities

- Functions/methods:
  - scene application method
  - edge policy helper
  - camera reset method
- Data structures/models:
  - scene application options
  - camera reset diagnostic
- Dependencies/services:
  - `impression.preview`
  - PyVista plotter protocol
- Returns/outputs/signals:
  - applied scene
  - camera reset diagnostic
- UI surfaces/components:
  - CLI preview host
  - workbench preview widget host
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is:
    - CLI private scene, edge, and camera behavior
  - Additions to existing reusable library/module:
    - shared controller methods
  - New reusable library/module to create:
    - none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - caller must invoke scene mutation on the caller-owned render thread
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - avoids duplicate mesh conversion and edge extraction paths
- Cross-screen reusable behavior:
  - shared by CLI preview and workbench preview

## Implementation Boundary

Owner/module:

- `src/impression/preview.py`

Routes:

- host render surface to shared controller scene methods

Reuse/extraction decision:

- Existing code reused as-is:
  - CLI private scene, edge, and camera behavior
- Additions to existing reusable library/module:
  - shared controller methods
- New reusable library/module to create:
  - none

UI field/control inventory:

- none

## Data And Defaults

Chosen defaults / parameters:

- object edge policy matches CLI preview unless overridden by style

Data ownership:

- controller owns scene semantics; hosts own renderer lifecycle

Open questions and resolved assumptions:

- mock plotter coverage may need a small plotter protocol shim

Implementation prerequisites:

- shared preview controller API

## Behavior

The implementation must:

- Move scene application, edge policy, and camera reset behavior into the shared preview controller without changing host lifecycle ownership.
- preserve the shared-preview boundary so the workbench does not grow a separate renderer path
- preserve renderer, payload, and UI-thread ownership described by the architecture
- emit deterministic diagnostics for invalid, unsupported, unsafe, or stale preview states

## Verification

Test strategy:

- controller scene-application tests with a mock plotter and camera reset
  tests

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
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 2 x 2 = 4
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 1 x 2 = 2
- Total: 24.5

Split decision:

- No split needed. Cohesion reason: scene application, edge policy, and camera
  reset are one render-semantic boundary; host migration is separate.

## Refinement Status

Final, with prerequisite dependencies listed below before implementation starts.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- the Shared Scene Application And Camera Policy boundary is implemented as described
- the specified ownership, routing, and reuse boundaries are preserved
- the listed test strategy passes for this leaf
- listed implementation prerequisites are satisfied or carried as explicit blocking failures before execution
