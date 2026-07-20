# Surface Spec 249: Seam Continuity Promotion And Diagnostics (v1.0)

## Overview

Promote seam and continuity handling beyond C0/G0 storage by defining request
records, supported continuity classes, validation, and diagnostics.

## Backlink

- [Architecture: Surface Body Completion Architecture](../architecture/surface-body-completion-architecture.md)

## Scope

This specification promotes the manifest candidate `Seam Continuity Promotion
And Diagnostics` into a final implementation leaf.

## Responsibilities

- Functions/methods:
  - continuity request validator
  - seam participation validator
  - unsupported continuity diagnostic builder
- Data structures/models:
  - continuity request record
  - continuity support record
  - seam participation validation result
- Dependencies/services:
  - seam/adjacency architecture
  - patch family promotion audit
  - tessellation watertightness checks
- Returns/outputs/signals:
  - continuity support verdict
  - unsupported continuity diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: seam and adjacency records
  - Additions to existing reusable library/module: continuity validation helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes validation and diagnostics for seam requests
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - validation bounded by seam and boundary-use count
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- `src/impression/modeling/surface.py` and seam/adjacency helpers

Routes:

- patch boundary records to seam validator to tessellation/CSG consumers

Reuse/extraction decision:

- extend seam/adjacency module; do not duplicate continuity rules in CSG or
  loft

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- unsupported G1/G2 or curvature requests refuse with diagnostics rather than
  silently downgrading

Data ownership:

- seam records own boundary identity
- continuity records own requested class

## Behavior

The implementation must:

- record continuity requests separately from observed continuity
- validate seam participation for every promoted family
- classify supported, unsupported, and not-yet-implemented continuity classes
- report unsupported continuity without silently downgrading requests
- expose results to tessellation, CSG, and loft consumers

## Verification

Test strategy:

- seam validation tests
- continuity request tests
- watertightness interaction tests
- unsupported-class diagnostic tests across promoted families

## Manifest Assessment

Score:

- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
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
- Total: 20.5

Split decision:

- Review for split. Cohesion reason: request, validation, and diagnostics are
  one continuity contract; family-specific math remains in family specs.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- continuity requests are explicit records
- unsupported classes refuse with diagnostics
- seam validation is shared by tessellation, CSG, and loft consumers
