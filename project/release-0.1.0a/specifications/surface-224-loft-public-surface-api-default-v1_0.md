# Surface Spec 224: Loft Public Surface API Default (v1.0)

## Overview

Make authored loft public APIs return `SurfaceBody` by default and reserve mesh
output for explicitly named tessellation, debug, or compatibility APIs.

## Backlink

- [Architecture: Mesh Execution Tessellation Boundary Architecture](../architecture/mesh-execution-tessellation-boundary-architecture.md)

## Scope

This specification promotes the split manifest candidate `Loft Public Surface
API Default` into a final implementation leaf.

## Responsibilities

- Functions/methods:
  - `loft_sections`
  - `loft_execute_plan`
  - `loft_execute_plan_debug_mesh`
- Data structures/models:
  - `LoftPlan`
  - `SurfaceBody` loft result
- Dependencies/services:
  - `src/impression/modeling/loft.py`
  - tessellation boundary
- Returns/outputs/signals:
  - canonical `SurfaceBody`
  - explicit debug mesh
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: loft planning and surface executor
  - Additions to existing reusable library/module: public route migration tests
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes public loft return defaults
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - avoids implicit tessellation on default path
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- `src/impression/modeling/loft.py`

Routes:

- public loft API to surface executor

Reuse/extraction decision:

- reuse existing surface executor and debug mesh executor

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- `loft_sections` returns `SurfaceBody` unless the caller chooses an explicitly
  named mesh/debug route

Data ownership:

- loft owns modeled surface result; tessellation/debug owns mesh output

## Behavior

The implementation must:

- change authored loft defaults to `SurfaceBody`
- preserve explicit debug mesh behavior only through names that clearly mark it
  as debug, tessellated, or compatibility output
- update tests and docs that still assume `loft_sections` is a mesh-returning
  authored route
- refuse unsupported surface loft cases explicitly instead of producing mesh
  fallback geometry

## Constraints

- Backward compatibility must be explicit; compatibility mesh helpers cannot
  remain the default authored path.
- The default path must not tessellate.

## Verification

Test strategy:

- public API tests for `SurfaceBody` default and explicit mesh route

Automated or review verification must prove authored loft output is surface
native and mesh output is reachable only through explicit boundary APIs.

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

- Review for split. Cohesion reason: this is one public return-contract change.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when `loft_sections` and authored loft routes
return `SurfaceBody` by default and mesh paths are explicitly named boundaries.
