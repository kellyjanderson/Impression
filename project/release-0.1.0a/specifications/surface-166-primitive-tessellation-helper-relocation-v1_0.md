# Surface Spec 166: Primitive Tessellation Helper Relocation (v1.0)

## Overview

Define the narrow case where primitive mesh helpers may move behind tessellation because they consume surface/tessellation inputs.

## Backlink

- [Architecture: Mesh Execution Tessellation Boundary Architecture](../architecture/mesh-execution-tessellation-boundary-architecture.md)

## Scope

This specification promotes the manifest candidate `Primitive Tessellation Helper Relocation` into a final implementation leaf. It covers the behavior, data ownership, reuse boundary, and verification expectations needed to implement this leaf without returning to the architecture document for hidden decisions.

## Responsibilities

- Functions/methods:
  - tessellation helper adapter
  - `_orient_mesh` relocation or deletion
- Data structures/models:
  - tessellation helper contract
  - surface-to-mesh adapter record
- Dependencies/services:
  - `tessellation.py`
  - `primitives.py`
- Returns/outputs/signals:
  - tessellated mesh result
  - boundary violation diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current helper math if applicable
  - Additions to existing reusable library/module: `tessellation.py`
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - relocates helper code behind tessellation
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - relocated helpers must obey tessellation request bounds
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- `src/impression/modeling/tessellation.py`

Routes:

- surface body to tessellation helper route

Reuse/extraction decision:

- add to existing `tessellation.py` only when helper is truly boundary-owned

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- relocated helpers consume `SurfaceBody`, `SurfacePatch`, or tessellation requests, never authored primitive arguments

Data ownership:

- tessellation owns mesh output

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

- tests for boundary inputs and no direct authored primitive route

Automated or review verification must prove that the implementation satisfies the manifest responsibilities and that failure modes are explicit. Where this spec touches serialization, verification must include round-trip or refusal coverage. Where this spec touches modeling output, verification must include surface-native output checks and no-hidden-mesh-fallback checks.

## Manifest Assessment

Score:

- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
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
- Total: 16.5

Split decision:

- Review for split. Cohesion reason: this is one tessellation-boundary relocation rule.

Open questions / nuance resolved for implementation:

- If a helper still consumes primitive constructor arguments, it belongs in compatibility, not tessellation.

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
