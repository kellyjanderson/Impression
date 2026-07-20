# Surface Spec 332: Higher-Order CSG Residual And Degeneracy Diagnostics (v1.0)

## Overview

Implement `Higher-Order CSG Residual And Degeneracy Diagnostics` as a final surface-body CSG completion leaf.

This specification is part of the surface-native CSG completion program. It
turns a reviewed specification-manifest candidate into executable work that must
keep CSG surface-native, explicit, diagnosable, and free of hidden mesh fallback.

## Backlink

- [Architecture: Higher-Order Parametric CSG Routes Architecture](../architecture/higher-order-parametric-csg-routes-architecture.md)

## Scope

This specification promotes the manifest candidate `Higher-Order CSG Residual And Degeneracy Diagnostics` into a final
implementation leaf.

This specification covers:

- the records, helpers, diagnostics, operation routes, and evidence named by the
  owning manifest candidate
- the exact implementation boundary required by the owning architecture
- deterministic refusal behavior for unsupported, unsafe, unavailable,
  ambiguous, under-evidenced, or non-convergent CSG states

## Manifest Candidate

Higher-Order CSG Residual And Degeneracy Diagnostics

Discovery purpose:
- Standardize residual, convergence, ambiguity, and degeneracy diagnostics for
  all higher-order parametric CSG routes.

Responsibilities:
- Functions/methods:
  - residual collector
  - degeneracy classifier
  - route diagnostic formatter
- Data structures/models:
  - residual record
  - degeneracy record
  - ambiguity diagnostic
- Dependencies/services:
  - intersection kernel
  - CSG operation planner
  - route registry
- Returns/outputs/signals:
  - structured route diagnostics
  - non-executable plan diagnostics
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: current diagnostic records
  - Additions to existing reusable library/module: higher-order route
    diagnostic helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - diagnostics must not require re-solving the pair
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/csg.py`
  - `src/impression/modeling/surface_intersections.py`
- Chosen defaults / parameters:
  - unresolved ambiguity blocks execution but planning continues to collect all
    diagnostics
- Test strategy:
  - non-convergence, ambiguous sweep, overlap, singularity, and budget refusal
    diagnostics
- Data ownership:
  - route owns local diagnostics; planner owns aggregate diagnostics
- Routes:
  - solver route to planner diagnostics to user-facing error payload
- Open questions / nuance discovered:
  - diagnostic locations need both patch-local and body-level identifiers
- Readiness blockers:
  - none

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 1 = 0
- Destructive/write behavior: 0 x 2 = 0
- Security/privacy-sensitive behavior: 0 x 2 = 0
- Performance-sensitive behavior: 1 x 1 = 1
- Cross-screen reusable behavior: 0 x 1 = 0
- Test scenarios/fixtures: 4 x 1 = 4
- Readiness blockers: 0 x 2 = 0
- Total: 23.5

Split decision:
- Review for split. Cohesion reason: this candidate owns shared diagnostics
  only and is intentionally independent of solver route implementation.

## Implementation Boundary

The implementation must stay inside the owner modules named by the manifest
candidate unless a smaller reusable helper is required by the architecture.

Required boundaries:

- preserve `SurfaceBody` as the runtime result type
- preserve authored patch-family payloads and provenance
- use tessellation only after a valid surface body exists
- refuse unsupported or ambiguous CSG plans before execution
- emit structured diagnostics for every unresolved ambiguity, unsupported route,
  non-convergence, missing mapping, invalid shell, dirty reference, or unsafe
  payload state named by this leaf

## Data And Defaults

The chosen defaults, routes, data ownership, open questions, and readiness
blockers are the project-readiness fields in the manifest candidate above.

Additional defaults:

- supported means executable, not merely available in the family matrix
- declared-tolerance routes must record tolerance, residual, convergence state,
  and affected patch ids
- mesh-backed fragments, mesh caps, mesh CSG, or triangle-fragment boolean
  execution are invalid implementation shortcuts
- planning may accumulate all diagnostics, but execution cannot proceed while
  unresolved blocking diagnostics remain

## Behavior

The implementation must:

- satisfy every responsibility listed in the manifest candidate with explicit
  records, helpers, diagnostics, route rows, or evidence gates
- update the CSG capability, executable support, or evidence matrix whenever the
  leaf changes a supported route
- preserve authored topology and patch-family intent instead of automatically
  resolving ambiguous authored inputs
- perform useful automatic solving where the architecture defines a deterministic
  route with bounded residuals and diagnostics
- keep unavailable, unsupported, unsafe, non-convergent, ambiguous, or
  non-applicable states inspectable and deterministic

## Verification

The test strategy is the strategy named by the manifest candidate.

Additional verification requirements:

- add focused unit coverage for each new record, helper, diagnostic, route row,
  solver adapter, mapping stage, or evidence gate introduced by this leaf
- add negative coverage for malformed, unsupported, unsafe, ambiguous,
  non-convergent, missing-evidence, dirty-reference, or non-applicable states
  named by this leaf
- include no-hidden-mesh-fallback assertions for every route or reconstruction
  stage touched by this leaf
- include `.impress` round-trip or tessellation-boundary checks when the leaf
  changes persisted surface-body shape or reference evidence
- update reference or diagnostic fixtures when this leaf changes visible model
  output or durable refusal behavior

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- `Higher-Order CSG Residual And Degeneracy Diagnostics` is implemented in the owner/module named by the manifest candidate
- all manifest responsibilities are represented by explicit records, helpers,
  diagnostics, operation rows, route records, or evidence gates
- unsupported, unsafe, unavailable, ambiguous, non-convergent, missing-evidence,
  dirty-reference, or non-applicable cases fail with deterministic diagnostics
  rather than hidden fallback behavior
- no implementation path performs mesh CSG or uses tessellation as a substitute
  for surface-body execution
- verification requirements are covered by implementation tests and any paired
  test specification created from this leaf
