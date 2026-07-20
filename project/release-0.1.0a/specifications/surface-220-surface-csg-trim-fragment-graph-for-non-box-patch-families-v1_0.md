# Surface Spec 220: Surface CSG Trim Fragment Graph For Non-Box Patch Families (v1.0)

## Overview

Generalize boolean fragment records beyond the initial box slice so analytic and
trimmed patch families can be split without tessellating first.

## Backlink

- [Architecture: SurfaceBody CSG Architecture](../architecture/surfacebody-csg-architecture.md)

## Scope

This specification promotes the manifest candidate `Surface CSG Trim Fragment
Graph For Non-Box Patch Families` into a final implementation leaf.

## Responsibilities

- Functions/methods:
  - intersection curve to trim-fragment mapper
  - fragment classification builder
- Data structures/models:
  - trim fragment graph
  - fragment provenance record
- Dependencies/services:
  - intersection curve records
  - seam and trim validation
- Returns/outputs/signals:
  - classified patch fragments
  - invalid fragment diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: initial boolean fragment concepts
  - Additions to existing reusable library/module: generalized fragment records
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes CSG internal records
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded by patch/curve count
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- CSG internal fragment/classification code

Routes:

- intersection discovery to result reconstruction

Reuse/extraction decision:

- reuse seam/trim validation helpers

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- trim fragments retain patch-local curve references and provenance

Data ownership:

- CSG owns temporary fragment graph

## Behavior

The implementation must:

- map intersection curves into trim fragments without tessellating patches
- classify fragments with source provenance and patch-local references
- reject invalid fragment graphs explicitly
- keep fragment graph records executor-internal and out of persisted `.impress`
  truth

## Constraints

- The fragment graph is temporary CSG execution state.
- The implementation must remain deterministic for equivalent inputs.
- Mesh vertices cannot be used as the authoritative fragment boundary.

## Verification

Test strategy:

- fragment fixtures for planar, cylinder, sphere, and trimmed cut cases

Automated or review verification must prove fragments preserve patch-local
truth and reject invalid mappings without mesh fallback.

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

- Review for split. Cohesion reason: mapping and classification share the same
  fragment graph contract.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when non-box CSG intersections become classified
surface fragments with provenance and no mesh-derived boundary truth.
