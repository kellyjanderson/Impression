# Surface Spec 199: Sampled Surface Tessellation Adapters (v1.0)

## Overview

Tessellate heightmap and displacement patches at the explicit boundary while preserving authored source payloads and lossiness metadata.

## Backlink

- [Architecture: Patch Family Integration Architecture](../architecture/patch-family-integration-architecture.md)

## Scope

This specification promotes the manifest candidate `Sampled Surface Tessellation Adapters` into a final implementation leaf. It covers the behavior, data ownership, reuse boundary, and verification expectations needed to implement this leaf without returning to the architecture document for hidden decisions.

## Responsibilities

- Functions/methods:
  - heightmap tessellation adapter
  - displacement tessellation adapter
- Data structures/models:
  - sampled grid fixture
  - displacement source patch fixture
  - lossiness metadata
- Dependencies/services:
  - `src/impression/modeling/tessellation.py`
  - sampled patch families
- Returns/outputs/signals:
  - `SurfaceTessellationResult`
  - sampled-family tessellation diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: tessellation adapter framework
  - Additions to existing reusable library/module: sampled adapters/tests
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - adds tessellation adapter behavior and tests
- Security/privacy-sensitive behavior:
  - no external image or source-file reads during tessellation
- Performance-sensitive behavior:
  - bounded sampled grids and preview-quality tessellation requests
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

  - `src/impression/modeling/tessellation.py`

Routes:

  - `tessellate_surface_patch`, `tessellate_surface_body`

Reuse/extraction decision:

  - reuse shared grid sampling helper if the codec specs create one

UI field/control inventory:

  - not applicable

## Data And Defaults

Chosen defaults / parameters:

  - preview-quality tessellation for minimal sampled fixtures

Data ownership:

  - tessellation owns mesh output; sampled patches retain authored truth

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

  - heightmap and displacement smoke tests assert mesh output metadata, source
    family identity, and no mutation of stored patch payloads

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
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 20.5

Split decision:

- Review for split.
- Keep together because heightmap and displacement share sampled-grid adapter
  mechanics.

Open questions / nuance resolved for implementation:

- Displacement tessellation must preserve source patch identity even when the
  generated mesh samples displaced positions.

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
