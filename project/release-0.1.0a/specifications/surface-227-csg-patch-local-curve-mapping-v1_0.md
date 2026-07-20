# Surface Spec 227: CSG Patch-Local Curve Mapping (v1.0)

## Overview

Map 3D intersection curve records into deterministic patch-local UV curve
records for every participating patch.

## Backlink

- [Architecture: SurfaceBody CSG Architecture](../architecture/surfacebody-csg-architecture.md)

## Scope

This specification extracts UV mapping work hidden inside analytic intersection
and trim fragment generation.

## Responsibilities

- Functions/methods:
  - 3D curve to patch-local mapper
  - patch-domain validation helper
  - mapping diagnostic builder
- Data structures/models:
  - patch-local curve record
  - mapping diagnostic record
- Dependencies/services:
  - CSG curve primitives
  - patch evaluators and domains
- Returns/outputs/signals:
  - UV curve record
  - explicit mapping failure diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: patch domain and evaluation APIs
  - Additions to existing reusable library/module: CSG patch-local mapping helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes CSG internal mapping
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded mapping by curve and patch count
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- CSG-private curve mapping helpers

Routes:

- intersection curve record to trim/fragment graph

Reuse/extraction decision:

- reuse curve/tolerance records from Surface Spec 226

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- mapping refuses when no deterministic patch-local representation exists

Data ownership:

- CSG owns mapping records; patch families own parameterization

## Behavior

The implementation must:

- produce patch-local curve records with stable orientation and domain bounds
- preserve participating patch references and source curve identity
- refuse outside-domain, singular, periodic-ambiguity, or non-invertible cases
  explicitly
- avoid dense mesh samples as the authoritative mapping

## Verification

Test strategy:

- planar and revolution patch mapping fixtures, including outside-domain and
  singular cases

## Manifest Assessment

Score:

- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
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
- Total: 18.5

Split decision:

- Review for split. Cohesion reason: this is one mapping boundary.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

