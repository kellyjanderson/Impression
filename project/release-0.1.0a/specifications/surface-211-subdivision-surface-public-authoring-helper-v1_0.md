# Surface Spec 211: Subdivision Surface Public Authoring Helper (v1.0)

## Overview

Provide a bounded public authoring route for `SubdivisionSurfacePatch` cage and crease payloads.

## Backlink

- [Architecture: Patch Family Integration Architecture](../architecture/patch-family-integration-architecture.md)

## Scope

This specification promotes the manifest candidate `Subdivision Surface Public Authoring Helper` into a final implementation leaf. It covers the behavior, data ownership, reuse boundary, and verification expectations needed to implement this leaf without returning to the architecture document for hidden decisions.

## Responsibilities

- Functions/methods:
  - subdivision surface helper
  - cage/crease validation wrapper
- Data structures/models:
  - `SubdivisionSurfacePatch`
  - control cage
  - crease payload
- Dependencies/services:
  - `src/impression/modeling/surface.py`
  - public modeling exports
- Returns/outputs/signals:
  - subdivision `SurfaceBody`
  - validation diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: `SubdivisionSurfacePatch`
  - Additions to existing reusable library/module: public modeling helper
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - adds public authoring API
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bound default subdivision level for helper fixtures
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

  - `src/impression/modeling/surface.py` or small public helper module

Routes:

  - public helper to `SubdivisionSurfacePatch` to `make_surface_body`

Reuse/extraction decision:

  - add wrapper only; keep refinement logic in existing surface module

UI field/control inventory:

  - not applicable

## Data And Defaults

Chosen defaults / parameters:

  - default scheme is `catmull_clark`, default subdivision level is bounded

Data ownership:

  - authoring helper owns body wrapping; patch owns cage validation

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

  - valid cage returns `SurfaceBody`; invalid cage/crease refuses

Automated or review verification must prove that the implementation satisfies the manifest responsibilities and that failure modes are explicit. Where this spec touches serialization, verification must include round-trip or refusal coverage. Where this spec touches modeling output, verification must include surface-native output checks and no-hidden-mesh-fallback checks. Where this spec touches tessellation, verification must prove mesh output is created only at the explicit tessellation boundary and records lossiness or approximation metadata when applicable.

## Manifest Assessment

Score:

- Functions/methods: 2 x 2 = 4
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 2 x 1 = 2
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
- Keep together because the helper and validation wrapper are one public
  authoring route.

Open questions / nuance resolved for implementation:

- Limit-surface evaluation remains patch-family behavior; helper only creates
  authored payloads.

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
