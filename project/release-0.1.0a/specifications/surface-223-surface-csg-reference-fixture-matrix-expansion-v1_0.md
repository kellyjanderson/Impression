# Surface Spec 223: Surface CSG Reference Fixture Matrix Expansion (v1.0)

## Overview

Add reference fixtures that prove the completed CSG scope works as
surface-native modeling and refuses unsupported cases without mesh fallback.

## Backlink

- [Architecture: SurfaceBody CSG Architecture](../architecture/surfacebody-csg-architecture.md)

## Scope

This specification promotes the manifest candidate `Surface CSG Reference
Fixture Matrix Expansion` into a final implementation leaf.

## Responsibilities

- Functions/methods:
  - CSG fixture builder
  - no-mesh assertion helper
  - tessellation-boundary verifier
- Data structures/models:
  - CSG fixture matrix
  - expected result/diagnostic record
- Dependencies/services:
  - CSG public API
  - tessellation boundary
  - reference artifact harness
- Returns/outputs/signals:
  - passing reference fixtures
  - explicit failure reports
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: reference artifact harness
  - Additions to existing reusable library/module: CSG fixture matrix
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - writes reference fixtures/artifacts as needed
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - fixtures must be bounded and deterministic
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- tests and reference fixture modules

Routes:

- fixture builder to public CSG API to tessellation verifier when needed

Reuse/extraction decision:

- reuse no-hidden-mesh helper from mesh boundary specs

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- every CSG fixture declares whether surface execution or refusal is expected

Data ownership:

- tests own fixture expectations; CSG owns operation behavior

## Behavior

The implementation must:

- cover executable analytic CSG pairs and explicit unsupported family pairs
- verify modeled results as `SurfaceBody` before any tessellation checks
- verify tessellated views only at the tessellation boundary
- fail when a fixture result is a mesh rewrap or hidden mesh boolean output

## Constraints

- Fixture outputs must not bless tessellated meshes as modeled truth.
- Reference artifacts must be deterministic and bounded.

## Verification

Test strategy:

- automated acceptance for exact results, diagnostics, and tessellation-only
  mesh

Automated or review verification must prove fixture expectations are role-aware
and no-hidden-mesh assertions run for CSG cases.

## Manifest Assessment

Score:

- Functions/methods: 3 x 2 = 6
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
- Total: 19.5

Split decision:

- Review for split. Cohesion reason: this is one fixture matrix that verifies
  the CSG completion pass end to end.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when CSG fixtures prove surface-native execution,
explicit refusal, and tessellation-only mesh output.
