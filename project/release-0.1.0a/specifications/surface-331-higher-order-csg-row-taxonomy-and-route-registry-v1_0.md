# Surface Spec 331: Higher-Order CSG Row Taxonomy And Route Registry (v1.0)

## Overview

Implement `Higher-Order CSG Row Taxonomy And Route Registry` as a final surface-body CSG completion leaf.

This specification is part of the surface-native CSG completion program. It
turns a reviewed specification-manifest candidate into executable work that must
keep CSG surface-native, explicit, diagnosable, and free of hidden mesh fallback.

## Backlink

- [Architecture: Higher-Order Parametric CSG Routes Architecture](../architecture/higher-order-parametric-csg-routes-architecture.md)

## Scope

This specification promotes the manifest candidate `Higher-Order CSG Row Taxonomy And Route Registry` into a final
implementation leaf.

This specification covers:

- the records, helpers, diagnostics, operation routes, and evidence named by the
  owning manifest candidate
- the exact implementation boundary required by the owning architecture
- deterministic refusal behavior for unsupported, unsafe, unavailable,
  ambiguous, under-evidenced, or non-convergent CSG states

## Manifest Candidate

Higher-Order CSG Row Taxonomy And Route Registry

Discovery purpose:
- Create the route registry that turns higher-order CSG row labels into
  executable pair-class routes.

Responsibilities:
- Functions/methods:
  - higher-order row classifier
  - route lookup
  - executable row report
- Data structures/models:
  - route registry row
  - pair-class enum
  - route support diagnostic
- Dependencies/services:
  - CSG capability matrix
  - patch-family registry
  - executable coverage gate
- Returns/outputs/signals:
  - executable route id
  - missing route diagnostic
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: current CSG family support rows
  - Additions to existing reusable library/module: route registry fields
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
  - bounded matrix scan
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/csg.py`
- Chosen defaults / parameters:
  - unregistered rows fail executable completion
- Test strategy:
  - route lookup tests for every higher-order pair class and operation
- Data ownership:
  - CSG owns route executability truth
- Routes:
  - capability matrix to route registry to executable coverage report
- Open questions / nuance discovered:
  - route exactness must be visible separately from family availability
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
- Test scenarios/fixtures: 2 x 1 = 2
- Readiness blockers: 0 x 2 = 0
- Total: 18.5

Split decision:
- Review for split. Cohesion reason: this candidate owns only classification
  and registry evidence, not solver implementation.

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

- `Higher-Order CSG Row Taxonomy And Route Registry` is implemented in the owner/module named by the manifest candidate
- all manifest responsibilities are represented by explicit records, helpers,
  diagnostics, operation rows, route records, or evidence gates
- unsupported, unsafe, unavailable, ambiguous, non-convergent, missing-evidence,
  dirty-reference, or non-applicable cases fail with deterministic diagnostics
  rather than hidden fallback behavior
- no implementation path performs mesh CSG or uses tessellation as a substitute
  for surface-body execution
- verification requirements are covered by implementation tests and any paired
  test specification created from this leaf
