# Surface Spec 187: Mesh Utility Quarantine (v1.0)

## Overview

Move retained mesh utilities into explicit mesh-tool modules and prevent surfaced modeling modules from depending on mesh execution casually.

## Backlink

- [Architecture: Mesh Execution Tessellation Boundary Architecture](../architecture/mesh-execution-tessellation-boundary-architecture.md)

## Scope

This specification promotes the manifest candidate `Mesh Utility Quarantine` into a final implementation leaf. It covers the behavior, data ownership, reuse boundary, and verification expectations needed to implement this leaf without returning to the architecture document for hidden decisions.

## Responsibilities

- Functions/methods:
  - mesh utility import boundary checker
  - mesh tool module routing
- Data structures/models:
  - mesh utility classification
- Dependencies/services:
  - `_ops_mesh.py`
  - `group.py`
  - `transform.py`
  - mesh analysis/repair modules
- Returns/outputs/signals:
  - quarantined module layout
  - import-boundary diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: retained mesh utility code
  - Additions to existing reusable library/module: mesh utility modules
  - New reusable library/module to create: optional `modeling/mesh_tools`
    namespace
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - moves modules/imports
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - none beyond import/static check cost
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- `src/impression/modeling/_ops_mesh.py`
  - candidate `src/impression/modeling/mesh_tools/`

Routes:

- explicit mesh tool API route

Reuse/extraction decision:

- extract or wrap retained mesh utilities into explicit namespace

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- retained mesh tools are explicit and never imported as authored surface
    execution

Data ownership:

- mesh tools own mesh-only inputs/outputs

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

- static import tests plus unit tests for retained mesh tool behavior

Automated or review verification must prove that the implementation satisfies the manifest responsibilities and that failure modes are explicit. Where this spec touches serialization, verification must include round-trip or refusal coverage. Where this spec touches modeling output, verification must include surface-native output checks and no-hidden-mesh-fallback checks.

## Manifest Assessment

Score:

- Functions/methods: 2 x 2 = 4
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 4 x 1 = 4
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 1 x 3 = 3
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
- Cohesion reason: the import boundary and module quarantine must be specified
  together to avoid moving code without enforcing the boundary.

Open questions / nuance resolved for implementation:

- Static boundary enforcement should avoid blocking legitimate tessellation
  imports.

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
