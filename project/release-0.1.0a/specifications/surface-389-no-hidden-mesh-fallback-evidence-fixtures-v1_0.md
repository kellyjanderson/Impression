# Surface Spec 389: No-Hidden-Mesh-Fallback Evidence Fixtures (v1.0)

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: this is a manifest-derived final leaf with one owner/module boundary, one integration route, and one verification surface. Related helper, artifact, and diagnostic responsibilities remain inside that single route rather than forming independently deliverable changes.

## Overview

Implement `No-Hidden-Mesh-Fallback Evidence Fixtures` as a final reference CSG gap-closure leaf.

This specification promotes a reviewed manifest candidate into executable work
for the reference-test expansion program. The implementation must keep model
truth surface-native, produce dirty STL references only for successful
`SurfaceBody` routes, and use structured diagnostic evidence for intended
refusal routes.

## Backlink

- [Architecture: Patch-Family Reference CSG Completion Architecture](../architecture/patch-family-reference-csg-completion-architecture.md)

## Scope

This specification promotes the manifest candidate `No-Hidden-Mesh-Fallback Evidence Fixtures` into a final
implementation leaf.

This specification covers:

- the records, helpers, diagnostics, route decisions, artifact writers, and
  evidence named by the owning manifest candidate
- the implementation owner/module and integration route named by the manifest
- the fixture or diagnostic evidence required by the unchecked reference-test
  progression item this leaf supports

## Manifest Candidate

### Candidate Spec: No-Hidden-Mesh-Fallback Evidence Fixtures

Discovery purpose:
- Add fixture assertions that prove patch-family CSG references cross the mesh
  boundary only after a surface-native result exists.

Responsibilities:
- Functions/methods:
  - no-hidden-mesh-fallback assertion helper
  - fallback audit fixture builder
- Data structures/models:
  - fallback-boundary audit record
  - solver-route evidence payload
- Dependencies/services:
  - CSG solver diagnostics
  - reference fixture registry
- Returns/outputs/signals:
  - fallback audit signal
  - fixture diagnostic context
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: CSG diagnostics
  - Additions to existing reusable library/module: reference audit helpers
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
  - bounded diagnostic probe count
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `tests/test_reference_stl_expansion.py`
  - `src/impression/modeling/csg.py`
- Chosen defaults / parameters:
  - fixture fails if boolean truth comes from tessellated mesh output
- Test strategy:
  - no-hidden-mesh-fallback assertions for all patch-family fixture groups
- Data ownership:
  - CSG owns solver-route evidence; tests own reference audit
- Routes:
  - CSG result diagnostics to reference audit helper
- Reuse/extraction decision:
  - add to existing CSG/reference modules
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- The solver-route evidence payload must be stable enough for tests without
  coupling them to private helper names.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 1 x 2 = 2
- Total: 15.5

Readiness blockers:
- [ ] Stable solver-route evidence payload must exist.

Split decision:
- Small.

## Implementation Boundary

The implementation must stay inside the owner modules named by the manifest
candidate unless a smaller reusable helper is required by the architecture.

Required boundaries:

- preserve `SurfaceBody` as the successful runtime result type
- create dirty STL artifacts only after a successful surface-native result
  exists
- represent intended refusal cases as structured diagnostic evidence, not as
  fabricated success STLs
- keep tessellation, export, and review artifacts downstream of modeled surface
  truth
- avoid hidden mesh fallback before the explicit tessellation/export boundary

## Data And Defaults

The chosen defaults, routes, data ownership, open questions, readiness blockers,
and reuse decisions are the project-readiness fields in the manifest candidate
above.

Additional defaults:

- dirty reference artifacts are unreviewed and do not count as promoted/gold
  evidence
- unsupported routes must report deterministic diagnostics with the operation,
  operand family or fixture route, and missing capability
- reference fixture ids, source entrypoints, artifact paths, and review context
  fields must be deterministic

## Behavior

The implementation must:

- satisfy every responsibility listed in the manifest candidate with explicit
  code, records, diagnostics, route rows, tests, or fixture artifacts
- preserve the data ownership and route boundaries named by the manifest
- update the reference fixture registry when the leaf creates a reviewable
  fixture
- update generation tests when the leaf creates or changes a dirty STL artifact
- update diagnostic/refusal tests when the leaf defines expected refusal
  evidence
- preserve no-hidden-mesh-fallback evidence for every supported route touched by
  this leaf

## Verification

The required test strategy is the strategy named by the manifest candidate.

Additional verification requirements:

- focused unit coverage for each new helper, record, route, or diagnostic
- fixture registry coverage for every new reviewable fixture
- STL signal checks for every successful dirty STL artifact
- diagnostic evidence checks for every intended refusal fixture
- no-hidden-mesh-fallback assertions for every CSG or loft route that this leaf
  exposes through reference artifacts

## Refinement Status

Final manifest-derived implementation leaf. Current manifest score: 15.5. 1 readiness blocker(s) are carried as explicit implementation prerequisites. Split review result: - Small.

## Review History

- Pass 1 - Template Completeness: Added the required Work Units section and confirmed the spec retains backlink, scope, manifest candidate, implementation boundary, data/defaults, behavior, verification, refinement status, child-spec, and acceptance sections.
- Pass 2 - Responsibility Boundary: Reviewed owner/module, route, data ownership, reuse plan, write behavior, and no-hidden-mesh-fallback boundaries. No additional implementation responsibility was discovered.
- Pass 3 - Readiness And Sequencing: Reviewed readiness blockers and dependency order. Existing blockers remain explicit implementation prerequisites; no hidden blocker requires guessing before implementation.
- Pass 4 - Split Review: Rechecked cohesion against manifest split policy and IWU sizing. The spec remains a final leaf because the work has one coherent route and one verification surface; no split was required.
- Pass 5 - Final Rescore: Recomputed the manifest score from the listed categories. Score is unchanged, remains below the forced-split threshold, and the spec is ready for implementation subject to any carried blockers.
- Rescore summary: manifest score remains 15.5; IWU count remains 1; child specifications remain none.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- `No-Hidden-Mesh-Fallback Evidence Fixtures` is implemented in the owner/module named by the manifest
- the specified ownership, routing, reuse, artifact, and diagnostic boundaries
  are preserved
- all readiness blockers are resolved or explicitly carried as implementation
  blockers before coding begins
- the verification strategy passes for this leaf
