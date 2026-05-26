# Surface Spec 237: CSG Primitive Caller Inventory And Gate Helper (v1.0)

## Overview

Inventory every primitive or feature caller that needs CSG and provide one
shared helper for refusing unsupported surface CSG before mesh fallback can run.

## Backlink

- [Architecture: SurfaceBody CSG Architecture](../architecture/surfacebody-csg-architecture.md)

## Scope

This specification extracts the audit and shared gate hidden inside Surface
Spec 222.

## Responsibilities

- Functions/methods:
  - CSG caller scanner
  - shared boolean readiness helper
  - no-mesh fallback assertion helper
- Data structures/models:
  - CSG caller inventory record
  - feature gate diagnostic record
- Dependencies/services:
  - primitive modules
  - feature builder modules
  - CSG support gate
- Returns/outputs/signals:
  - durable caller inventory
  - shared diagnostic result
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: CSG family gate
  - Additions to existing reusable library/module: shared caller gate helper
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - writes tests/docs and gate code
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded gate lookup per caller
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- CSG caller inventory documentation/tests and shared gate helper

Routes:

- primitive/feature builder to CSG support gate

Reuse/extraction decision:

- one helper shared by primitives, threads, and hinges

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- unsupported CSG in authored surface paths returns explicit diagnostics

Data ownership:

- feature modules own caller intent; CSG owns support truth

## Behavior

The implementation must:

- identify all CSG-dependent primitive and feature call sites
- route every caller through one shared support helper
- fail tests when callers invoke mesh booleans as hidden fallback
- preserve explicitly named mesh compatibility routes

## Verification

Test strategy:

- source inventory plus unit tests for the shared gate helper

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

- Review for split. Cohesion reason: inventory and helper are one migration
  preparation boundary.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

