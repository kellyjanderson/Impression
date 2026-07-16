# Surface Spec 391: Loft Self-Intersection Diagnostic Reference Fixture (v1.0)

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: this is a manifest-derived final leaf with one owner/module boundary, one integration route, and one verification surface. Related helper, artifact, and diagnostic responsibilities remain inside that single route rather than forming independently deliverable changes.

## Overview

Implement `Loft Self-Intersection Diagnostic Reference Fixture` as a final reference CSG gap-closure leaf.

This specification promotes a reviewed manifest candidate into executable work
for the reference-test expansion program. The implementation must keep model
truth surface-native, produce dirty STL references only for successful
`SurfaceBody` routes, and use structured diagnostic evidence for intended
refusal routes.

## Backlink

- [Architecture: Loft Self-Intersection Reference Architecture](../architecture/loft-self-intersection-reference-architecture.md)

## Scope

This specification promotes the manifest candidate `Loft Self-Intersection Diagnostic Reference Fixture` into a final
implementation leaf.

This specification covers:

- the records, helpers, diagnostics, route decisions, artifact writers, and
  evidence named by the owning manifest candidate
- the implementation owner/module and integration route named by the manifest
- the fixture or diagnostic evidence required by the unchecked reference-test
  progression item this leaf supports

## Manifest Candidate

### Candidate Spec: Loft Self-Intersection Diagnostic Reference Fixture

Discovery purpose:
- Represent `RT-LOFT-037` as reference-test evidence when the intended result
  is refusal rather than a success STL.

Responsibilities:
- Functions/methods:
  - diagnostic reference evidence producer
  - self-intersection fixture builder
- Data structures/models:
  - invalid loft reference record
  - diagnostic artifact payload
- Dependencies/services:
  - loft validity detector
  - reference fixture registry
- Returns/outputs/signals:
  - reference evidence artifact
  - fixture registry row
- UI surfaces/components:
  - reference review app
- UI fields/elements:
  - purpose
  - methodology
  - diagnostic description
- Reusable code plan:
  - Existing code reused as-is: reference fixture context
  - Additions to existing reusable library/module: reference diagnostic fixture
    helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - may write diagnostic reference artifacts
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded to one self-intersection fixture
- Cross-screen reusable behavior:
  - diagnostic context reused by reference review details

Project readiness fields:
- Implementation owner/module:
  - `tests/test_reference_stl_expansion.py`
  - `tests/reference_review_fixtures/stl_review_sources.py`
- Chosen defaults / parameters:
  - refusal fixtures do not write success STLs
- Test strategy:
  - reference diagnostic fixture test and registry test
- Data ownership:
  - fixture registry owns review evidence
- Routes:
  - loft validity diagnostic to reference evidence producer
- Reuse/extraction decision:
  - add to existing reference fixture modules
- UI field/control inventory:
  - purpose, methodology, diagnostic description

Open questions / nuance discovered:
- The fixture schema may need a diagnostic artifact kind distinct from dirty
  STL when the correct result is refusal.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 3 x 1 = 3
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 1 x 2 = 2
- Total: 23.5

Readiness blockers:
- [ ] Reference fixture schema needs a diagnostic artifact policy for expected
  refusal cases.

Split decision:
- Review for split. Cohesion reason: this is one diagnostic reference fixture
  contract and has already been separated from geometry validity detection.

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

Final manifest-derived implementation leaf. Current manifest score: 23.5. 1 readiness blocker(s) are carried as explicit implementation prerequisites. Split review result: Review for split. Cohesion reason: this is one diagnostic reference fixture contract and has already been separated from geometry validity detection.

## Review History

- Pass 1 - Template Completeness: Added the required Work Units section and confirmed the spec retains backlink, scope, manifest candidate, implementation boundary, data/defaults, behavior, verification, refinement status, child-spec, and acceptance sections.
- Pass 2 - Responsibility Boundary: Reviewed owner/module, route, data ownership, reuse plan, write behavior, and no-hidden-mesh-fallback boundaries. No additional implementation responsibility was discovered.
- Pass 3 - Readiness And Sequencing: Reviewed readiness blockers and dependency order. Existing blockers remain explicit implementation prerequisites; no hidden blocker requires guessing before implementation.
- Pass 4 - Split Review: Rechecked cohesion against manifest split policy and IWU sizing. The spec remains a final leaf because the work has one coherent route and one verification surface; no split was required.
- Pass 5 - Final Rescore: Recomputed the manifest score from the listed categories. Score is unchanged, remains below the forced-split threshold, and the spec is ready for implementation subject to any carried blockers.
- Rescore summary: manifest score remains 23.5; IWU count remains 1; child specifications remain none.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- `Loft Self-Intersection Diagnostic Reference Fixture` is implemented in the owner/module named by the manifest
- the specified ownership, routing, reuse, artifact, and diagnostic boundaries
  are preserved
- all readiness blockers are resolved or explicitly carried as implementation
  blockers before coding begins
- the verification strategy passes for this leaf
