# Surface Spec 203: Loft Patch Family Selection Record (v1.0)

## Overview

Add a deterministic record that explains why loft selected ruled, B-spline, NURBS, or sweep output for a planned transition.

## Backlink

- [Architecture: Patch Family Integration Architecture](../architecture/patch-family-integration-architecture.md)

## Scope

This specification promotes the manifest candidate `Loft Patch Family Selection Record` into a final implementation leaf. It covers the behavior, data ownership, reuse boundary, and verification expectations needed to implement this leaf without returning to the architecture document for hidden decisions.

## Responsibilities

- Functions/methods:
  - loft family classifier
  - family selection diagnostic helper
- Data structures/models:
  - loft family selection record
  - intent evidence record
- Dependencies/services:
  - loft planner/executor
  - topology correspondence records
- Returns/outputs/signals:
  - selected patch family
  - refusal reason for unsupported intent
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: loft plan and correspondence structures
  - Additions to existing reusable library/module: loft family selection helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes loft plan/executor metadata
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - classifier must be linear in station/transition count
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

  - `src/impression/modeling/loft.py`

Routes:

  - loft planner to loft executor

Reuse/extraction decision:

  - add to existing loft planning records

UI field/control inventory:

  - not applicable

## Data And Defaults

Chosen defaults / parameters:

  - simple two-station compatible transitions stay ruled unless explicit smooth
    or sweep intent is present

Data ownership:

  - loft plan owns selection evidence; executor consumes it

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

  - classifier tests for ruled, B-spline, NURBS, sweep, and unsupported cases

Automated or review verification must prove that the implementation satisfies the manifest responsibilities and that failure modes are explicit. Where this spec touches serialization, verification must include round-trip or refusal coverage. Where this spec touches modeling output, verification must include surface-native output checks and no-hidden-mesh-fallback checks. Where this spec touches tessellation, verification must prove mesh output is created only at the explicit tessellation boundary and records lossiness or approximation metadata when applicable.

## Manifest Assessment

Score:

- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 2 x 0.5 = 1
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 17

Split decision:

- Review for split.
- Keep together because classifier and selection diagnostics are one planner
  contract.

Open questions / nuance resolved for implementation:

- NURBS selection requires an explicit rational/conic intent signal until
  automatic conic detection is specified.

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
