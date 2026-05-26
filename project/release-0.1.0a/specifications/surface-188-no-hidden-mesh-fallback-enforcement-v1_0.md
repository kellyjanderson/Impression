# Surface Spec 188: No Hidden Mesh Fallback Enforcement (v1.0)

## Overview

Ensure unsupported surface operations fail with diagnostics rather than silently falling back to mesh.

## Backlink

- [Architecture: Mesh Execution Tessellation Boundary Architecture](../architecture/mesh-execution-tessellation-boundary-architecture.md)

## Scope

This specification promotes the manifest candidate `No Hidden Mesh Fallback Enforcement` into a final implementation leaf. It covers the behavior, data ownership, reuse boundary, and verification expectations needed to implement this leaf without returning to the architecture document for hidden decisions.

## Responsibilities

- Functions/methods:
  - fallback detection tests
  - diagnostic assertion helpers
- Data structures/models:
  - unsupported operation diagnostic
  - no-fallback test fixture matrix
- Dependencies/services:
  - loft
  - CSG
  - primitives
  - text/drafting
  - heightmap/threading/hinges
  - `.impress` serializer
- Returns/outputs/signals:
  - explicit unsupported result
  - failing test on mesh fallback
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: existing warnings and diagnostics
  - Additions to existing reusable library/module: test helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - test/spec only; implementation changes happen in dependent specs
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - fallback tests should use bounded fixtures
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- test suite under `tests/`

Routes:

- test helper to subsystem API

Reuse/extraction decision:

- add reusable test helpers

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- hidden mesh fallback is always test failure for authored surface APIs

Data ownership:

- surface subsystem owns fallback policy; tests own evidence

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

- automated acceptance tests across each affected subsystem

Automated or review verification must prove that the implementation satisfies the manifest responsibilities and that failure modes are explicit. Where this spec touches serialization, verification must include round-trip or refusal coverage. Where this spec touches modeling output, verification must include surface-native output checks and no-hidden-mesh-fallback checks.

## Manifest Assessment

Score:

- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 6 x 1 = 6
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 20.5

Split decision:

- Review for split.
- Cohesion reason: the fixture matrix spans subsystems, but the actual behavior
  is one policy: no hidden mesh fallback.

Open questions / nuance resolved for implementation:

- This spec should be implemented after at least the initial inventory so it
  knows the full subsystem list.

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
