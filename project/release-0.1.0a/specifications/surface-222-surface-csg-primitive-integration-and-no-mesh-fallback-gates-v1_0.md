# Surface Spec 222: Surface CSG Primitive Integration And No-Mesh Fallback Gates (v1.0)

## Overview

Ensure primitives and modeled features that require booleans consume surface CSG
or refuse explicitly rather than invoking mesh booleans.

## Backlink

- [Architecture: SurfaceBody CSG Architecture](../architecture/surfacebody-csg-architecture.md)

## Scope

This specification promotes the manifest candidate `Surface CSG Primitive
Integration And No-Mesh Fallback Gates` into a final implementation leaf.

## Responsibilities

- Functions/methods:
  - primitive CSG route audit
  - boolean-dependent feature gate
- Data structures/models:
  - primitive CSG dependency record
  - no-mesh-fallback assertion fixture
- Dependencies/services:
  - primitives
  - threading and hinge feature builders
  - CSG support gate
- Returns/outputs/signals:
  - surface primitive result or diagnostic
  - test failure on mesh boolean route
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: primitive surface defaults
  - Additions to existing reusable library/module: feature CSG gates
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes primitive/feature routing where needed
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded per feature operation
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- primitive and feature builder modules that call CSG

Routes:

- feature builder to CSG support gate to surface result/diagnostic

Reuse/extraction decision:

- share one gate helper instead of per-feature ad hoc checks

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- boolean-dependent surface features refuse when surface CSG cannot execute

Data ownership:

- feature modules own caller policy; CSG owns operation support truth

## Behavior

The implementation must:

- audit primitive and feature routes that invoke boolean behavior
- route surface features through the surface CSG support gate
- retain explicit mesh compatibility only where the API name and docs make the
  mesh boundary obvious
- fail tests when a surface feature chooses a mesh boolean as fallback

## Constraints

- Surface primitives cannot call mesh booleans to manufacture modeled truth.
- The implementation must not remove explicit mesh tools that are already
  quarantined and named as mesh tools.

## Verification

Test strategy:

- audit tests proving no primitive falls back to mesh boolean

Automated or review verification must prove CSG-dependent features either
produce `SurfaceBody` output or explicit diagnostics.

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

- Review for split. Cohesion reason: every caller uses the same CSG support
  gate and no-fallback assertion.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when boolean-dependent primitives and features no
longer use mesh booleans as hidden fallback paths.
