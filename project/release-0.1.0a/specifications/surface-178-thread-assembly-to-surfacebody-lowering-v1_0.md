# Surface Spec 178: Thread Assembly To SurfaceBody Lowering (v1.0)

## Overview

Lower thread assemblies to `SurfaceBody` when their dependencies are available, including boolean-dependent assemblies.

## Backlink

- [Architecture: Mesh Execution Tessellation Boundary Architecture](../architecture/mesh-execution-tessellation-boundary-architecture.md)

## Scope

This specification promotes the manifest candidate `Thread Assembly To SurfaceBody Lowering` into a final implementation leaf. It covers the behavior, data ownership, reuse boundary, and verification expectations needed to implement this leaf without returning to the architecture document for hidden decisions.

## Responsibilities

- Functions/methods:
  - thread assembly lowering
  - thread primitive operand lowering
  - thread boolean handoff
- Data structures/models:
  - `ThreadSurfaceAssembly`
  - `SurfaceBody`
  - thread operand record
- Dependencies/services:
  - `threading.py`
  - surface booleans
  - surface primitives
- Returns/outputs/signals:
  - lowered surface body
  - explicit unsupported dependency diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: thread surface assembly records
  - Additions to existing reusable library/module: `threading.py`
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - adds surfaced lowering behavior
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - lowering and tessellation must respect thread quality budgets
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- `src/impression/modeling/threading.py`

Routes:

- thread assembly to surface primitive/boolean to surface body

Reuse/extraction decision:

- add to existing `threading.py`; depend on surface boolean contracts

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- boolean-dependent assemblies refuse explicitly until surface CSG is ready

Data ownership:

- assembly owns authored structure; surface body owns lowered geometry

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

- tests for rod, cutter, nut, relief lowering and unsupported dependency cases

Automated or review verification must prove that the implementation satisfies the manifest responsibilities and that failure modes are explicit. Where this spec touches serialization, verification must include round-trip or refusal coverage. Where this spec touches modeling output, verification must include surface-native output checks and no-hidden-mesh-fallback checks.

## Manifest Assessment

Score:

- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 3 x 1 = 3
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
- Total: 20.5

Split decision:

- Review for split. Cohesion reason: assembly lowering is one dependency-aware surface route.

Open questions / nuance resolved for implementation:

- Surface boolean readiness is an explicit dependency, not a license for mesh fallback.

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
