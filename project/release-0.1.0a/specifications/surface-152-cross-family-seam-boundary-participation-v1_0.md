# Surface Spec 152: Cross-Family Seam Boundary Participation (v1.0)

## Overview

Define how non-planar and advanced patch families participate in seams, boundary references, adjacency, and continuity metadata.

## Backlink

- [Architecture: Full Surface Patch Family Architecture](../architecture/full-surface-patch-family-architecture.md)

## Scope

This specification promotes the manifest candidate `Cross-Family Seam Boundary Participation` into a final implementation leaf. It covers the behavior, data ownership, reuse boundary, and verification expectations needed to implement this leaf without returning to the architecture document for hidden decisions.

## Responsibilities

- Functions/methods:
  - boundary extraction
  - seam compatibility check
  - continuity classification
- Data structures/models:
  - family boundary descriptor
  - seam participation record
  - continuity metadata
- Dependencies/services:
  - seam/adjacency architecture
  - patch family evaluators
- Returns/outputs/signals:
  - seam validation result
  - adjacency update
  - unsupported continuity diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: seam and boundary records
  - Additions to existing reusable library/module: surface seam/adjacency module
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes seam validation for new families
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - boundary comparison must avoid relying only on dense mesh sampling
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- surface seam/adjacency module selected by implementation spec

Routes:

- patch boundary descriptor to seam validation to tessellation

Reuse/extraction decision:

- add to existing seam/adjacency logic

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- compare analytic/parametric boundaries first; approximation metadata required
    when exact comparison is unavailable

Data ownership:

- seams own shared boundary truth; patches own boundary evaluation

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

- tests for compatible/incompatible family seams, continuity metadata, and
    tessellation watertightness

Automated or review verification must prove that the implementation satisfies the manifest responsibilities and that failure modes are explicit. Where this spec touches serialization, verification must include round-trip or refusal coverage. Where this spec touches modeling output, verification must include surface-native output checks and no-hidden-mesh-fallback checks.

## Manifest Assessment

Score:

- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 3 x 1 = 3
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
- Cohesion reason: all family seam behavior uses the same seam contract; per-
  family fixtures can be test cases.

Open questions / nuance resolved for implementation:

- Exactness policy may need per-family approximation metadata.

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
