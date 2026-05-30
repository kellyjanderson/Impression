# Surface Spec 176: Heightmap Mesh Compatibility And Serialization Guard (v1.0)

## Overview

Preserve explicit mesh compatibility while preventing mesh-derived triangle wrappers from being serialized as native surface truth.

## Backlink

- [Architecture: Mesh Execution Tessellation Boundary Architecture](../architecture/mesh-execution-tessellation-boundary-architecture.md)

## Scope

This specification promotes the manifest candidate `Heightmap Mesh Compatibility And Serialization Guard` into a final implementation leaf. It covers the behavior, data ownership, reuse boundary, and verification expectations needed to implement this leaf without returning to the architecture document for hidden decisions.

## Responsibilities

- Functions/methods:
  - mesh compatibility heightmap helper
  - serialization guard
- Data structures/models:
  - mesh compatibility result
  - invalid surface-wrapper diagnostic
- Dependencies/services:
  - `heightmap.py`
  - `.impress` codec
  - tests
- Returns/outputs/signals:
  - explicit mesh result
  - refusal diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current mesh implementation
  - Additions to existing reusable library/module: `heightmap.py`, `.impress` guard tests
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes compatibility and persistence behavior
- Security/privacy-sensitive behavior:
  - local image path input
- Performance-sensitive behavior:
  - compatibility mesh generation keeps existing LOD bounds
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- `src/impression/modeling/heightmap.py` and future `.impress` codec

Routes:

- explicit mesh helper or serializer guard route

Reuse/extraction decision:

- add guards to existing modules

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- mesh heightfields are explicit mesh data only; triangle wrappers are rejected as native surface truth

Data ownership:

- mesh compatibility owns mesh-only data; `.impress` owns persisted surface truth

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

- tests for explicit mesh route and serialization refusal

Automated or review verification must prove that the implementation satisfies the manifest responsibilities and that failure modes are explicit. Where this spec touches serialization, verification must include round-trip or refusal coverage. Where this spec touches modeling output, verification must include surface-native output checks and no-hidden-mesh-fallback checks.

## Manifest Assessment

Score:

- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 2 x 1 = 2
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 21.5

Split decision:

- Review for split. Cohesion reason: compatibility and persistence guard are one anti-wrapper policy.

Open questions / nuance resolved for implementation:

- This spec should land after the inventory names all heightmap mesh wrapper paths.

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
