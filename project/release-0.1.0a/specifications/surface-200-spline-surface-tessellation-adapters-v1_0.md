# Surface Spec 200: Spline Surface Tessellation Adapters (v1.0)

## Overview

Tessellate B-spline, NURBS, and sweep surface patches from family-native surface records rather than mesh-producing loft executors.

## Backlink

- [Architecture: Patch Family Integration Architecture](../architecture/patch-family-integration-architecture.md)

## Scope

This specification promotes the manifest candidate `Spline Surface Tessellation Adapters` into a final implementation leaf. It covers the behavior, data ownership, reuse boundary, and verification expectations needed to implement this leaf without returning to the architecture document for hidden decisions.

## Responsibilities

- Functions/methods:
  - B-spline tessellation adapter
  - NURBS tessellation adapter
  - sweep tessellation adapter
- Data structures/models:
  - spline tessellation fixture
  - rational weight fixture
  - sweep rail/section fixture
- Dependencies/services:
  - `src/impression/modeling/tessellation.py`
  - spline patch families
- Returns/outputs/signals:
  - `SurfaceTessellationResult`
  - spline-family tessellation diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: tessellation adapter framework
  - Additions to existing reusable library/module: spline adapters/tests
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - adds tessellation adapter behavior and tests
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded knot grids, section counts, and preview tessellation requests
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

  - `src/impression/modeling/tessellation.py`

Routes:

  - `tessellate_surface_patch`, `tessellate_surface_body`

Reuse/extraction decision:

  - reuse shared spline basis utilities from loft specs

UI field/control inventory:

  - not applicable

## Data And Defaults

Chosen defaults / parameters:

  - preview-quality tessellation with explicit parameter-domain sampling

Data ownership:

  - tessellation owns mesh output; spline patches retain authored truth

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

  - one smoke per spline-like family asserts metadata, source identity, and no
    mutation of control nets or weight grids

Automated or review verification must prove that the implementation satisfies the manifest responsibilities and that failure modes are explicit. Where this spec touches serialization, verification must include round-trip or refusal coverage. Where this spec touches modeling output, verification must include surface-native output checks and no-hidden-mesh-fallback checks. Where this spec touches tessellation, verification must prove mesh output is created only at the explicit tessellation boundary and records lossiness or approximation metadata when applicable.

## Manifest Assessment

Score:

- Functions/methods: 3 x 2 = 6
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
- Total: 19.5

Split decision:

- Review for split.
- Keep together because these adapters share parameter-domain sampling and
  spline basis evaluation.

Open questions / nuance resolved for implementation:

- Sweep tessellation should consume the stored sweep patch, not the loft
  planner that produced it.

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
