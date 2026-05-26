# Surface Spec 146: Subdivision Surface Evaluation And Tessellation (v1.0)

## Overview

Define Catmull-Clark refinement, deterministic finite-level evaluation, and tessellation approximation metadata.

## Backlink

- [Architecture: Full Surface Patch Family Architecture](../architecture/full-surface-patch-family-architecture.md)

## Scope

This specification promotes the manifest candidate `Subdivision Surface Evaluation And Tessellation` into a final implementation leaf. It covers the behavior, data ownership, reuse boundary, and verification expectations needed to implement this leaf without returning to the architecture document for hidden decisions.

## Responsibilities

- Functions/methods:
  - finite-level evaluator
  - Catmull-Clark refinement
  - subdivision tessellation adapter
- Data structures/models:
  - refinement result
  - approximation metadata
- Dependencies/services:
  - subdivision patch payload
  - tessellation
  - evaluator utility
- Returns/outputs/signals:
  - evaluated subdivision surface
  - tessellated approximation
  - refinement diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: subdivision payload
  - Additions to existing reusable library/module: patch family module, tessellation
  - New reusable library/module to create: none beyond evaluator utility
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - writes or changes only the scoped artifact described by this spec
- Security/privacy-sensitive behavior:
  - none unless explicitly named in this candidate
- Performance-sensitive behavior:
  - bounded to the scoped artifact described by this spec
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- evaluator/tessellation implementation

Routes:

- subdivision evaluator to tessellation adapter

Reuse/extraction decision:

- add to evaluator utility and tessellation module

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- finite-level Catmull-Clark evaluation carries approximation metadata

Data ownership:

- evaluator owns refinement; tessellation owns mesh output

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

- refinement level, crease behavior, tessellation, and metadata tests

Automated or review verification must prove that the implementation satisfies the manifest responsibilities and that failure modes are explicit. Where this spec touches serialization, verification must include round-trip or refusal coverage. Where this spec touches modeling output, verification must include surface-native output checks and no-hidden-mesh-fallback checks.

## Manifest Assessment

Score:

- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 3 x 1 = 3
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 2 x 1 = 2
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
- Total: 21.5

Split decision:

- Review for split. Cohesion reason: refinement and tessellation are one approximation execution boundary.

Open questions / nuance resolved for implementation:

- Limit exactness is not required for V1 but must not be hidden.

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
