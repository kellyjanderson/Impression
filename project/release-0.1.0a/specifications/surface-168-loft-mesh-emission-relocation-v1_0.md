# Surface Spec 168: Loft Mesh Emission Relocation (v1.0)

## Overview

Move loft mesh face emission into tessellation/debug/compatibility boundaries instead of canonical plan execution.

## Backlink

- [Architecture: Mesh Execution Tessellation Boundary Architecture](../architecture/mesh-execution-tessellation-boundary-architecture.md)

## Scope

This specification promotes the manifest candidate `Loft Mesh Emission Relocation` into a final implementation leaf. It covers the behavior, data ownership, reuse boundary, and verification expectations needed to implement this leaf without returning to the architecture document for hidden decisions.

## Responsibilities

- Functions/methods:
  - `loft_execute_plan`
  - `emit_mesh_faces_from_sample_correspondence`
- Data structures/models:
  - `LoftPlan`
  - debug mesh result
- Dependencies/services:
  - `loft.py`
  - `tessellation.py`
- Returns/outputs/signals:
  - canonical surface result
  - explicit debug/tessellated mesh
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: loft plan validation and correspondence records
  - Additions to existing reusable library/module: `loft.py`, `tessellation.py`
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes only the scoped API, spec, or migration boundary described here
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded to the scoped operation and existing quality/request controls
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- `src/impression/modeling/loft.py` and `tessellation.py`

Routes:

- planner to surface executor to tessellation/debug route

Reuse/extraction decision:

- add to existing loft/tessellation modules

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- `LoftPlan -> SurfaceBody` is canonical; mesh emission is explicit only

Data ownership:

- `LoftPlan` owns correspondence; tessellation/debug owns mesh

## Behavior

The implementation must:

- satisfy every function, data-structure, dependency, and output responsibility listed above
- preserve the architecture boundary named in the backlink
- reject unsupported or ambiguous states with explicit diagnostics rather than silent fallback behavior
- keep mesh data outside canonical authored-surface state unless this spec explicitly names a tessellation, compatibility, or mesh-utility boundary
- expose only the public API surface needed by downstream specs and tests

## Constraints

- The implementation must remain deterministic for equivalent inputs.
- The implementation must keep metadata and stable identity behavior explicit when the leaf touches persisted or reusable surface state.
- The implementation must not introduce hidden mesh execution in authored modeling paths.
- The implementation must not broaden industry interchange, patch-family, or mesh compatibility scope beyond what this leaf names.

## Verification

Test strategy:

- tests proving no surface loft fallback to mesh and explicit mesh route behavior

Automated or review verification must prove that the implementation satisfies the manifest responsibilities and that failure modes are explicit. Where this spec touches serialization, verification must include round-trip or refusal coverage. Where this spec touches modeling output, verification must include surface-native output checks and no-hidden-mesh-fallback checks.

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
- Total: 21.5

Split decision:

- Review for split. Cohesion reason: this candidate now owns one discovered boundary.

Open questions / nuance resolved for implementation:

- This should be implemented after or alongside surface executor correspondence consumption.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- all manifest responsibilities are implemented or explicitly refused by the leaf
- owner/module, routes, data ownership, reuse, UI inventory, defaults, and test strategy are represented in code or verification artifacts
- related progression items can be checked without relying on unstated architecture assumptions
- downstream specs can cite this leaf instead of re-reading the manifest candidate
