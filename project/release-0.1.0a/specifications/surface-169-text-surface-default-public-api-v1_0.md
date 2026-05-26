# Surface Spec 169: Text Surface Default Public API (v1.0)

## Overview

Make text authored APIs surface-default while preserving profile generation and orientation semantics.

## Backlink

- [Architecture: Mesh Execution Tessellation Boundary Architecture](../architecture/mesh-execution-tessellation-boundary-architecture.md)

## Scope

This specification promotes the manifest candidate `Text Surface Default Public API` into a final implementation leaf. It covers the behavior, data ownership, reuse boundary, and verification expectations needed to implement this leaf without returning to the architecture document for hidden decisions.

## Responsibilities

- Functions/methods:
  - `make_text`
  - `text`
  - `_surface_text_extrude`
- Data structures/models:
  - `Section`
  - `SurfaceBody`
- Dependencies/services:
  - `text.py`
  - `_surface_ops.py`
  - font loading
- Returns/outputs/signals:
  - surfaced text body
  - deterministic placeholder for empty text
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: `text_profiles`, `sections_from_paths`
  - Additions to existing reusable library/module: `text.py`
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes public text default return type
- Security/privacy-sensitive behavior:
  - local font path handling only
- Performance-sensitive behavior:
  - outline processing bounded by glyph count
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- `src/impression/modeling/text.py`

Routes:

- text API to profiles to surface extrusion to tessellation

Reuse/extraction decision:

- add to existing `text.py`; reuse profile generation

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- `make_text` and `text` return `SurfaceBody` by default

Data ownership:

- text profiles feed surface extrusion

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

- unit tests for default return type, orientation, color metadata, and empty text

Automated or review verification must prove that the implementation satisfies the manifest responsibilities and that failure modes are explicit. Where this spec touches serialization, verification must include round-trip or refusal coverage. Where this spec touches modeling output, verification must include surface-native output checks and no-hidden-mesh-fallback checks.

## Manifest Assessment

Score:

- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 2 x 0.5 = 1
- Adding code to an existing library/module: 1 x 1 = 1
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
- Total: 23

Split decision:

- Review for split. Cohesion reason: public text default, profile reuse, and surface extrusion are one behavior change.

Open questions / nuance resolved for implementation:

- Empty text should avoid serializing a visible fake body.

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
