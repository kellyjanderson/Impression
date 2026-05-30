# Surface Spec 238: CSG Primitive Migration (v1.0)

## Overview

Migrate primitive CSG consumers to the shared surface CSG gate and remove hidden
mesh boolean fallback from primitive authored paths.

## Backlink

- [Architecture: SurfaceBody CSG Architecture](../architecture/surfacebody-csg-architecture.md)

## Scope

This specification extracts primitive caller migration hidden inside Surface
Spec 222.

## Responsibilities

- Functions/methods:
  - primitive boolean route updates
  - primitive no-fallback tests
- Data structures/models:
  - primitive CSG route record
  - primitive diagnostic assertion
- Dependencies/services:
  - primitive modules
  - shared CSG caller gate
- Returns/outputs/signals:
  - surface primitive result
  - explicit unsupported diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: primitive surface defaults
  - Additions to existing reusable library/module: primitive route tests
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes primitive route behavior
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded per primitive operation
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- primitive modules and primitive tests

Routes:

- primitive authoring API to shared CSG gate

Reuse/extraction decision:

- reuse Surface Spec 237 gate helper

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- primitive authored paths return surface results or explicit diagnostics

Data ownership:

- primitive modules own caller policy; CSG owns support truth

## Behavior

The implementation must:

- migrate primitive boolean call sites found by the inventory
- keep explicit mesh compatibility APIs named as such
- add tests proving primitive authoring paths never choose mesh booleans as
  hidden fallback
- preserve public surface result contracts

## Verification

Test strategy:

- primitive route tests for successful surface CSG and unsupported diagnostics

## Manifest Assessment

Score:

- Functions/methods: 2 x 2 = 4
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
- Total: 16.5

Split decision:

- Review for split. Cohesion reason: this is primitive-only caller migration.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

