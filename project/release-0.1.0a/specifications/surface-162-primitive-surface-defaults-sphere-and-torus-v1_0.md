# Surface Spec 162: Primitive Surface Defaults: Sphere And Torus (v1.0)

## Overview

Promote curved analytic primitives to surface-default output and make sampling density a tessellation concern.

## Backlink

- [Architecture: Mesh Execution Tessellation Boundary Architecture](../architecture/mesh-execution-tessellation-boundary-architecture.md)

## Scope

This specification promotes the manifest candidate `Primitive Surface Defaults: Sphere And Torus` into a final implementation leaf. It covers the behavior, data ownership, reuse boundary, and verification expectations needed to implement this leaf without returning to the architecture document for hidden decisions.

## Responsibilities

- Functions/methods:
  - `make_sphere`
  - `make_torus`
- Data structures/models:
  - `SurfaceBody`
  - analytic curved primitive metadata payload
- Dependencies/services:
  - `primitives.py`
  - `_surface_primitives.py`
  - `tessellation.py`
- Returns/outputs/signals:
  - surface-default primitive return
  - explicit tessellated mesh output
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: `make_surface_sphere`, `make_surface_torus`
  - Additions to existing reusable library/module: `primitives.py`
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes public defaults for listed primitives
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - tessellation should preserve quality and periodic sampling bounds
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- `src/impression/modeling/primitives.py`

Routes:

- primitive API to surface primitive constructor to tessellation

Reuse/extraction decision:

- add to existing primitive modules; no new reusable module

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- listed constructors return `SurfaceBody` by default; angular sample counts become tessellation guidance

Data ownership:

- authored truth lives in analytic surface bodies

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

- unit tests for default return type, periodic seam tessellation, color metadata, and mesh compatibility

Automated or review verification must prove that the implementation satisfies the manifest responsibilities and that failure modes are explicit. Where this spec touches serialization, verification must include round-trip or refusal coverage. Where this spec touches modeling output, verification must include surface-native output checks and no-hidden-mesh-fallback checks.

## Manifest Assessment

Score:

- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 2 x 0.5 = 1
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
- Total: 18

Split decision:

- Review for split. Cohesion reason: these are the two current analytic curved primitives with the same tessellation-density issue.

Open questions / nuance resolved for implementation:

- Periodic seam behavior should be tested explicitly so surface-default output does not regress watertight tessellation.

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
