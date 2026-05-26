# Surface Spec 171: Drafting Surface Defaults (v1.0)

## Overview

Move drafting geometry from mesh-default annotations to surfaced annotation bodies or surface consumer records.

## Backlink

- [Architecture: Mesh Execution Tessellation Boundary Architecture](../architecture/mesh-execution-tessellation-boundary-architecture.md)

## Scope

This specification promotes the manifest candidate `Drafting Surface Defaults` into a final implementation leaf. It covers the behavior, data ownership, reuse boundary, and verification expectations needed to implement this leaf without returning to the architecture document for hidden decisions.

## Responsibilities

- Functions/methods:
  - `make_line`
  - `make_plane`
  - `make_arrow`
  - `make_dimension`
- Data structures/models:
  - `SurfaceBody`
  - `SurfaceConsumerCollection`
- Dependencies/services:
  - `drafting.py`
  - `_surface_primitives.py`
  - `text.py`
- Returns/outputs/signals:
  - surfaced drafting body
  - consumer collection
  - explicit mesh compatibility
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: surface primitive constructors
  - Additions to existing reusable library/module: `drafting.py`
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes public drafting default return type
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - dimension text generation bounded by label content
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- `src/impression/modeling/drafting.py`

Routes:

- drafting API to surface primitive/text helpers to tessellation

Reuse/extraction decision:

- add to existing `drafting.py`; reuse surface primitives and text surface
    extrusion

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- drafting APIs return surface bodies or consumer collections by default

Data ownership:

- drafting authored truth lives in surface annotations

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

- unit tests for each drafting helper, transforms, color metadata, and
    explicit tessellation

Automated or review verification must prove that the implementation satisfies the manifest responsibilities and that failure modes are explicit. Where this spec touches serialization, verification must include round-trip or refusal coverage. Where this spec touches modeling output, verification must include surface-native output checks and no-hidden-mesh-fallback checks.

## Manifest Assessment

Score:

- Functions/methods: 4 x 2 = 8
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 3 x 1 = 3
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
- Total: 22.5

Split decision:

- Review for split.
- Cohesion reason: all listed helpers are one annotation subsystem and share the
  same surface default and consumer-collection decision.

Open questions / nuance resolved for implementation:

- `make_dimension` may remain a collection instead of a single `SurfaceBody`
  because it is naturally a composed consumer annotation.

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
