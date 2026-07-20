# Surface Spec 231: CSG Curve Arrangement And Trim Loop Splitting (v1.0)

## Overview

Arrange patch-local intersection curves against existing trim loops and split
them into deterministic trim-loop fragments.

## Backlink

- [Architecture: SurfaceBody CSG Architecture](../architecture/surfacebody-csg-architecture.md)

## Scope

This specification extracts curve arrangement and trim splitting hidden inside
Surface Spec 220.

## Responsibilities

- Functions/methods:
  - curve arrangement builder
  - trim-loop splitter
  - overlap/snap diagnostic helper
- Data structures/models:
  - patch-local arrangement graph
  - split trim loop record
- Dependencies/services:
  - patch-local curve mapping
  - trim loop validation
- Returns/outputs/signals:
  - split trim fragments
  - invalid arrangement diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: `TrimLoop` validation
  - Additions to existing reusable library/module: CSG arrangement helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes CSG trim processing
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded by curve and loop count
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- CSG trim/arrangement helpers

Routes:

- patch-local curves to fragment graph

Reuse/extraction decision:

- reuse Specs 226 and 227

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- arrangement order is deterministic; ambiguous overlaps refuse

Data ownership:

- CSG owns temporary arrangement graphs

## Behavior

The implementation must:

- split existing outer and inner trim loops at intersection events
- snap only within the shared CSG tolerance policy
- preserve loop category, orientation, and patch-local coordinates
- refuse ambiguous overlap, self-intersection, or zero-length fragments

## Verification

Test strategy:

- trim arrangement fixtures for outer-only, holes, tangent contacts, overlap,
  and zero-length fragments

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

- Review for split. Cohesion reason: arrangement and trim splitting are one
  patch-local preprocessing boundary.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

