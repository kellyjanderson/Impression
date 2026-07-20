# Surface Spec 221: Surface CSG Multi-Family Result Reconstruction (v1.0)

## Overview

Reconstruct boolean results from mixed analytic and trimmed fragments as
`SurfaceBody` shells with durable seams, trims, and provenance.

## Backlink

- [Architecture: SurfaceBody CSG Architecture](../architecture/surfacebody-csg-architecture.md)

## Scope

This specification promotes the manifest candidate `Surface CSG Multi-Family
Result Reconstruction` into a final implementation leaf.

## Responsibilities

- Functions/methods:
  - fragment-to-shell assembler
  - seam/use rebuilder
  - result validity gate
- Data structures/models:
  - reconstructed shell record
  - boolean provenance metadata
- Dependencies/services:
  - fragment graph
  - seam/adjacency model
  - `.impress` surface payload constraints
- Returns/outputs/signals:
  - `SurfaceBody` boolean result
  - invalid reconstruction diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: seam and adjacency primitives
  - Additions to existing reusable library/module: result assembly helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes CSG result construction
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded reconstruction validation
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- CSG result reconstruction code

Routes:

- fragment graph to validity gate to public result

Reuse/extraction decision:

- reuse existing `SurfaceBody` and seam containers

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- reconstructed truth is surface-native; no mesh rewrap is permitted

Data ownership:

- result `SurfaceBody` owns durable topology; CSG owns provenance metadata

## Behavior

The implementation must:

- assemble surviving fragments into one or more valid `SurfaceBody` shells
- rebuild shared seams, open boundaries, trim loops, and adjacency truth
- preserve operation provenance deterministically
- return empty or multi-shell results as valid outcomes when geometry dictates
- reject invalid reconstruction rather than replacing the result with a mesh

## Constraints

- The result cannot serialize tessellated triangles as native surface truth.
- The implementation must remain deterministic for equivalent inputs.
- Bounded healing may normalize topology but cannot materially alter geometry.

## Verification

Test strategy:

- mixed-family union, difference, and intersection reconstruction fixtures

Automated or review verification must prove the result is a surface-native
`SurfaceBody` and mesh appears only after explicit tessellation.

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

- Review for split. Cohesion reason: shell assembly, seam rebuilding, and
  validity are one result-boundary operation.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when CSG reconstruction emits durable
multi-family `SurfaceBody` results without mesh rewrapping.
