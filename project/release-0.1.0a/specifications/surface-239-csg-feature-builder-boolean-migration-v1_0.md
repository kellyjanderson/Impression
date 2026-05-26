# Surface Spec 239: CSG Feature Builder Boolean Migration (v1.0)

## Overview

Migrate feature builders such as threading and hinges to the shared surface CSG
gate and remove hidden mesh boolean fallback from authored feature paths.

## Backlink

- [Architecture: SurfaceBody CSG Architecture](../architecture/surfacebody-csg-architecture.md)

## Scope

This specification extracts non-primitive feature migration hidden inside
Surface Spec 222.

## Responsibilities

- Functions/methods:
  - feature boolean route updates
  - feature no-fallback tests
- Data structures/models:
  - feature CSG dependency record
  - feature diagnostic assertion
- Dependencies/services:
  - threading feature builders
  - hinge feature builders
  - shared CSG caller gate
- Returns/outputs/signals:
  - surface feature result
  - explicit unsupported diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: surface feature builders
  - Additions to existing reusable library/module: feature route tests
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes feature route behavior
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded per feature operation
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- threading, hinge, and other feature builder modules that call CSG

Routes:

- feature authoring API to shared CSG gate

Reuse/extraction decision:

- reuse Surface Spec 237 gate helper

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- feature builders refuse explicitly when required surface CSG is unavailable

Data ownership:

- feature modules own caller policy; CSG owns support truth

## Behavior

The implementation must:

- migrate feature boolean call sites found by the inventory
- preserve explicit mesh compatibility routes only when named as mesh
  compatibility
- add tests proving authored feature paths never choose mesh booleans as hidden
  fallback
- return deterministic diagnostics for unsupported CSG-dependent features

## Verification

Test strategy:

- feature route tests for threading, hinges, and any additional inventoried
  feature builders

## Manifest Assessment

Score:

- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 17.5

Split decision:

- Review for split. Cohesion reason: this is feature-builder caller migration.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

