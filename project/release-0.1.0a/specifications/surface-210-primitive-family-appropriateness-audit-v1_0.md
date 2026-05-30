# Surface Spec 210: Primitive Family Appropriateness Audit (v1.0)

## Overview

Confirm primitives use the simplest exact patch family and identify only natural advanced-family primitive additions.

## Backlink

- [Architecture: Patch Family Integration Architecture](../architecture/patch-family-integration-architecture.md)

## Scope

This specification promotes the manifest candidate `Primitive Family Appropriateness Audit` into a final implementation leaf. It covers the behavior, data ownership, reuse boundary, and verification expectations needed to implement this leaf without returning to the architecture document for hidden decisions.

## Responsibilities

- Functions/methods:
  - primitive family audit tests
  - missing advanced producer notes
- Data structures/models:
  - primitive-to-family matrix
  - producer gap record
- Dependencies/services:
  - `src/impression/modeling/primitives.py`
  - `src/impression/modeling/_surface_primitives.py`
  - heightmap/displacement builders
- Returns/outputs/signals:
  - primitive family assertions
  - unsupported/missing producer diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current primitive builders
  - Additions to existing reusable library/module: tests and fixture matrix
  - New reusable library/module to create: none by default
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - test/spec only unless stale primitive output is found
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded primitive smoke fixtures
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

  - tests covering `primitives.py`, `_surface_primitives.py`, and heightmap

Routes:

  - public primitive APIs to surface primitive builders

Reuse/extraction decision:

  - add audit tests; only add new primitive builders in later specs

UI field/control inventory:

  - not applicable

## Data And Defaults

Chosen defaults / parameters:

  - primitives use simplest exact family; no advanced family is used merely for
    coverage

Data ownership:

  - primitive builders own producer family choice

## Behavior

The implementation must:

- satisfy every function, data-structure, dependency, and output responsibility listed above
- preserve the architecture boundary named in the backlink
- reject unsupported or ambiguous states with explicit diagnostics rather than silent fallback behavior
- keep mesh data outside canonical authored-surface state unless this spec explicitly names a tessellation, compatibility, or mesh-utility boundary
- preserve family-native payloads, stable identity, and capability metadata when the leaf touches surface storage, traversal, persistence, or tessellation
- expose only the public API surface needed by downstream specs and tests

## Constraints

- The implementation must remain deterministic for equivalent inputs.
- The implementation must keep metadata and stable identity behavior explicit when the leaf touches persisted or reusable surface state.
- The implementation must not introduce hidden mesh execution in authored modeling paths.
- The implementation must not broaden industry interchange, patch-family, or mesh compatibility scope beyond what this leaf names.
- Bounded fixture, sampling, and performance limits named in the manifest must be represented in code or tests before this leaf is marked complete.

## Verification

Test strategy:

  - assert box/polyhedron planar, prism ruled, cylinder/cone/sphere/torus
    revolution, heightmap/displacement native

Automated or review verification must prove that the implementation satisfies the manifest responsibilities and that failure modes are explicit. Where this spec touches serialization, verification must include round-trip or refusal coverage. Where this spec touches modeling output, verification must include surface-native output checks and no-hidden-mesh-fallback checks. Where this spec touches tessellation, verification must prove mesh output is created only at the explicit tessellation boundary and records lossiness or approximation metadata when applicable.

## Manifest Assessment

Score:

- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 3 x 1 = 3
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
- Total: 17.5

Split decision:

- Review for split.
- Keep together because this is an audit and guardrail, not a family producer
  implementation.

Open questions / nuance resolved for implementation:

- A future sweep primitive should be separate from this audit because it is a
  new producer, not a correction to existing primitive choices.

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
