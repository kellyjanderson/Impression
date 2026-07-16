# Patch-Family Reference CSG Completion Architecture

## Overview

This document defines the architecture needed to complete the unchecked
patch-family CSG reference fixtures in the reference-test expansion plan.

The existing CSG architecture defines broad solver responsibilities. This
document narrows that work to the reference fixture contract: which patch-family
routes must produce success fixtures, which routes should produce explicit
refusal fixtures, and which no-hidden-mesh-fallback evidence is required before
dirty STL references can be generated.

## Related Architecture

This document extends:

- [Reference CSG Gap Closure Architecture](reference-csg-gap-closure-architecture.md)
- [Higher-Order Surface CSG Solver Architecture](higher-order-surface-csg-solver-architecture.md)
- [Higher-Order Parametric CSG Routes Architecture](higher-order-parametric-csg-routes-architecture.md)
- [Sampled and Implicit Surface CSG Support Architecture](sampled-implicit-surface-csg-support-architecture.md)
- [Sampled and Implicit CSG Unsupported Row Implementation Architecture](sampled-implicit-csg-unsupported-row-implementation-architecture.md)
- [Full Surface Patch Family Architecture](full-surface-patch-family-architecture.md)

## Reference Items Owned

This architecture owns these unchecked reference items:

- `RT-PATCH-CSG-001` planar patch CSG against box and sphere cutters
- `RT-PATCH-CSG-003` revolution patch CSG from cylinder/cone/sphere/torus-like
  bodies
- `RT-PATCH-CSG-004` B-spline patch CSG
- `RT-PATCH-CSG-005` NURBS patch CSG
- `RT-PATCH-CSG-006` sweep patch CSG
- `RT-PATCH-CSG-007` subdivision patch CSG
- `RT-PATCH-CSG-011` mixed planar/ruled/revolution matrix
- `RT-PATCH-CSG-012` sampled/implicit mixed-family promotion fixtures
- `RT-PATCH-CSG-013` unsupported-family route fixtures
- `RT-PATCH-CSG-014` no-hidden-mesh-fallback fixture evidence

## Components

### Patch-Family Reference Matrix

The reference matrix maps each planned fixture to:

- operation
- operand families
- source builder
- expected support state
- expected result family
- expected artifact type
- no-hidden-mesh-fallback evidence

This matrix is reference-facing. It is derived from the solver matrix but does
not replace the solver matrix.

### Success Fixture Producer

The producer owns successful STL fixtures for exact and declared-tolerance
routes. It must build operands using public Impression modeling APIs, execute
the surface CSG route, and export only the resulting `SurfaceBody`.

### Promotion Fixture Producer

The producer owns cases where the result family is not the same as either
input family, such as sampled-to-implicit or sampled-to-subdivision promotion.

Promotion fixtures must record:

- source families
- result family
- tolerance or lossiness metadata
- provenance from the original operands

### Refusal Fixture Producer

Unsupported-family fixtures are valid when refusal is the intended product.
They should not produce an STL pretending to be success. Instead they should
produce diagnostic evidence that the route refused before mesh fallback.

### No-Hidden-Mesh-Fallback Evidence

Every patch-family fixture must prove that triangles were not used as boolean
truth. Acceptable evidence includes:

- result family records
- operation provenance
- solver-route diagnostics
- explicit tessellation-boundary markers after `SurfaceBody` result creation

## Data Flow

```text
reference item
-> patch-family reference matrix
-> success / promotion / refusal producer
-> runtime support gate
-> dirty STL fixture or diagnostic refusal evidence
-> review fixture registry
```

## Cross-Domain Decisions

### Reference Fixtures Inherit Solver Matrix Truth

A patch-family reference fixture must not locally decide that an unsupported
solver row is acceptable. The solver matrix owns support state.

### Refusal Fixtures Are Not STL Success Fixtures

When the expected behavior is refusal, the artifact should be diagnostic
evidence, not a fabricated STL. STL fixtures remain for successful
surface-native outputs.

### Promotion Is A First-Class Reference Result

If a heightmap or displacement operation promotes to implicit, subdivision, or
another native surface family, the reference fixture should describe that
promotion directly in its context fields.

## Specification Manifest for Discovery

### Candidate Spec: Patch-Family Reference Matrix

Discovery purpose:
- Define the auditable matrix that maps unchecked patch-family reference items
  to operation, family pair, expected support state, and artifact strategy.

Responsibilities:
- Functions/methods:
  - patch-family reference matrix builder
  - solver-support lookup adapter
  - fixture expectation formatter
- Data structures/models:
  - patch-family reference row
  - expected artifact policy
  - no-hidden-mesh-fallback evidence requirement
- Dependencies/services:
  - CSG solver support matrix
  - reference-test expansion plan
  - fixture registry
- Returns/outputs/signals:
  - fixture-ready rows
  - refusal-evidence rows
  - missing solver route diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: CSG solver support records
  - Additions to existing reusable library/module: reference STL expansion
    helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - may write dirty STL artifacts only for successful rows
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded matrix scan and focused fixture generation
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `tests/test_reference_stl_expansion.py`
  - `tests/reference_review_fixtures/stl_review_sources.py`
- Chosen defaults / parameters:
  - success rows produce dirty STL; refusal rows produce diagnostics
- Test strategy:
  - matrix coverage test and focused generated-artifact checks
- Data ownership:
  - solver matrix owns support state; reference matrix owns artifact intent
- Routes:
  - solver matrix to reference matrix to fixture generator
- Reuse/extraction decision:
  - add to existing reference test helper modules
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- The review app currently centers STL artifacts; diagnostic refusal evidence
  may need separate review-app presentation later.

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 3 x 1 = 3
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
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
- [ ] Review-app diagnostic artifact display policy is not yet defined.

Split decision:
- Review for split. Cohesion reason: the matrix is a single reference-facing
  contract, even though it points to separate success, promotion, and refusal
  producers.

### Candidate Spec: Planar And Revolution CSG Success Fixtures

Discovery purpose:
- Create success fixture coverage for planar and revolution patch-family CSG
  rows that the solver matrix marks exact or declared-tolerance.

Responsibilities:
- Functions/methods:
  - planar/revolution fixture builders
  - solver-result assertion helper
- Data structures/models:
  - fixture context record
  - source provenance payload
- Dependencies/services:
  - analytic CSG solver routes
  - review fixture registry
- Returns/outputs/signals:
  - dirty STL artifact
  - fixture registry row
- UI surfaces/components:
  - reference review app
- UI fields/elements:
  - purpose
  - methodology
  - render description
- Reusable code plan:
  - Existing code reused as-is: current fixture source contract and STL helpers
  - Additions to existing reusable library/module: CSG reference builders
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - writes dirty STL artifacts
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded tessellation for review fixtures
- Cross-screen reusable behavior:
  - review context fields reused by fixture list/detail

Project readiness fields:
- Implementation owner/module:
  - `tests/reference_review_fixtures/stl_review_sources.py`
- Chosen defaults / parameters:
  - generated fixtures stay in dirty until human review
- Test strategy:
  - generation tests plus review registry tests
- Data ownership:
  - fixture JSON owns review context; source builders own deterministic model
- Routes:
  - source builder to CSG API to STL writer to review fixture registry
- Reuse/extraction decision:
  - add to existing reference fixture source module
- UI field/control inventory:
  - purpose, methodology, render description

Open questions / nuance discovered:
- The planar and revolution fixtures should expose visible trims rather than
  only containment or no-op cases.

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
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 0 x 2 = 0
- Total: 23.5

Readiness blockers:
- None.

Split decision:
- Review for split. Cohesion reason: the planar and revolution fixtures share
  the exact analytic route and fixture contract.

### Candidate Spec: B-Spline And NURBS CSG Success Fixtures

Discovery purpose:
- Create declared-tolerance success fixtures for B-spline and NURBS patch CSG
  rows with visible residual/provenance evidence.

Responsibilities:
- Functions/methods:
  - spline fixture builders
  - declared-tolerance assertion helper
- Data structures/models:
  - declared-tolerance evidence payload
  - source provenance payload
- Dependencies/services:
  - higher-order CSG solver routes
  - review fixture registry
- Returns/outputs/signals:
  - dirty STL artifact
  - tolerance/provenance diagnostics
- UI surfaces/components:
  - reference review app
- UI fields/elements:
  - purpose
  - methodology
  - render description
- Reusable code plan:
  - Existing code reused as-is: current fixture source contract and STL helpers
  - Additions to existing reusable library/module: CSG reference builders
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - writes dirty STL artifacts
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded tessellation for review fixtures
- Cross-screen reusable behavior:
  - review context fields reused by fixture list/detail

Project readiness fields:
- Implementation owner/module:
  - `tests/reference_review_fixtures/stl_review_sources.py`
- Chosen defaults / parameters:
  - generated fixtures stay in dirty until human review
- Test strategy:
  - generation tests plus declared-tolerance evidence assertions
- Data ownership:
  - fixture JSON owns review context; source builders own deterministic model
- Routes:
  - source builder to CSG API to STL writer to review fixture registry
- Reuse/extraction decision:
  - add to existing reference fixture source module
- UI field/control inventory:
  - purpose, methodology, render description

Open questions / nuance discovered:
- B-spline and NURBS fixtures should not collapse to primitive-equivalent
  planar or revolution cases.

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
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 0 x 2 = 0
- Total: 23.5

Readiness blockers:
- None.

Split decision:
- Review for split. Cohesion reason: B-spline and NURBS fixtures share the
  declared-tolerance route and evidence contract.

### Candidate Spec: Sweep And Subdivision CSG Success Fixtures

Discovery purpose:
- Create declared-tolerance success fixtures for sweep and subdivision CSG
  rows that reveal route-specific approximation behavior.

Responsibilities:
- Functions/methods:
  - sweep/subdivision fixture builders
  - solver-result assertion helper
- Data structures/models:
  - fixture context record
  - approximation evidence payload
- Dependencies/services:
  - higher-order CSG solver routes
  - review fixture registry
- Returns/outputs/signals:
  - dirty STL artifact
  - approximation/provenance diagnostics
- UI surfaces/components:
  - reference review app
- UI fields/elements:
  - purpose
  - methodology
  - render description
- Reusable code plan:
  - Existing code reused as-is: current fixture source contract and STL helpers
  - Additions to existing reusable library/module: CSG reference builders
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - writes dirty STL artifacts
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded tessellation for review fixtures
- Cross-screen reusable behavior:
  - review context fields reused by fixture list/detail

Project readiness fields:
- Implementation owner/module:
  - `tests/reference_review_fixtures/stl_review_sources.py`
- Chosen defaults / parameters:
  - generated fixtures stay in dirty until human review
- Test strategy:
  - generation tests plus approximation evidence assertions
- Data ownership:
  - fixture JSON owns review context; source builders own deterministic model
- Routes:
  - source builder to CSG API to STL writer to review fixture registry
- Reuse/extraction decision:
  - add to existing reference fixture source module
- UI field/control inventory:
  - purpose, methodology, render description

Open questions / nuance discovered:
- Sweep and subdivision cases need nontrivial source geometry without making
  review artifacts visually unreadable.

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
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 0 x 2 = 0
- Total: 23.5

Readiness blockers:
- None.

Split decision:
- Review for split. Cohesion reason: sweep and subdivision fixtures share the
  approximation-evidence fixture contract.

### Candidate Spec: Mixed Planar Ruled Revolution Matrix Fixtures

Discovery purpose:
- Create a small fixture matrix for mixed planar, ruled, and revolution CSG
  rows without bundling higher-order or sampled families.

Responsibilities:
- Functions/methods:
  - mixed-family fixture builders
  - matrix coverage assertion helper
- Data structures/models:
  - mixed-family fixture row
  - fixture context record
- Dependencies/services:
  - analytic/ruled CSG routes
  - review fixture registry
- Returns/outputs/signals:
  - dirty STL artifacts
  - matrix coverage signal
- UI surfaces/components:
  - reference review app
- UI fields/elements:
  - purpose
  - methodology
  - render description
- Reusable code plan:
  - Existing code reused as-is: current fixture source contract and STL helpers
  - Additions to existing reusable library/module: CSG reference builders
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - writes dirty STL artifacts
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded tessellation for review fixtures
- Cross-screen reusable behavior:
  - review context fields reused by fixture list/detail

Project readiness fields:
- Implementation owner/module:
  - `tests/reference_review_fixtures/stl_review_sources.py`
- Chosen defaults / parameters:
  - keep the matrix small and representative rather than exhaustive
- Test strategy:
  - generation tests plus matrix coverage assertion
- Data ownership:
  - fixture JSON owns review context; source builders own deterministic model
- Routes:
  - source builder to CSG API to STL writer to review fixture registry
- Reuse/extraction decision:
  - add to existing reference fixture source module
- UI field/control inventory:
  - purpose, methodology, render description

Open questions / nuance discovered:
- The matrix should avoid duplicating existing primitive CSG fixtures unless it
  proves a distinct patch-family route.

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
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 0 x 2 = 0
- Total: 23.5

Readiness blockers:
- None.

Split decision:
- Review for split. Cohesion reason: this is one intentionally small mixed
  analytic/ruled fixture matrix.

### Candidate Spec: Sampled Implicit Promotion Success Fixtures

Discovery purpose:
- Create dirty STL fixtures for sampled/implicit CSG rows whose intended result
  is a supported promoted native surface family.

Responsibilities:
- Functions/methods:
  - sampled promotion fixture builders
  - promotion evidence assertion helper
- Data structures/models:
  - promotion evidence payload
  - promoted result context record
- Dependencies/services:
  - sampled/implicit CSG policy matrix
  - implicit/heightmap/displacement composition routes
- Returns/outputs/signals:
  - dirty STL for successful promotion
  - promotion evidence signal
- UI surfaces/components:
  - reference review app
- UI fields/elements:
  - expected result family
  - methodology
- Reusable code plan:
  - Existing code reused as-is: sampled/implicit composition helpers
  - Additions to existing reusable library/module: reference fixture builders
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - writes dirty artifacts for success cases
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded sample budgets and no broad grid explosion
- Cross-screen reusable behavior:
  - diagnostic context reused by review details

Project readiness fields:
- Implementation owner/module:
  - `tests/reference_review_fixtures/stl_review_sources.py`
  - `src/impression/modeling/csg.py`
- Chosen defaults / parameters:
  - supported promotion records lossiness/provenance
- Test strategy:
  - success fixture tests and promotion evidence assertions
- Data ownership:
  - CSG owns promotion policy; fixture registry owns review context
- Routes:
  - sampled policy matrix to producer to artifact evidence
- Reuse/extraction decision:
  - add to existing CSG/reference modules
- UI field/control inventory:
  - expected result family, methodology

Open questions / nuance discovered:
- The promotion fixture should expose result family clearly enough for review.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 1 x 2 = 2
- Total: 24.5

Readiness blockers:
- [ ] Sampled/implicit promotion matrix must choose fixture expectations.

Split decision:
- Review for split. Cohesion reason: success promotion fixtures share one
  artifact strategy and one result-family evidence contract.

### Candidate Spec: Unsupported-Family Refusal Fixtures

Discovery purpose:
- Represent intentionally unsupported patch-family routes as structured
  diagnostic reference evidence instead of success STLs.

Responsibilities:
- Functions/methods:
  - unsupported-family diagnostic fixture builders
  - refusal assertion helper
- Data structures/models:
  - refusal evidence payload
  - diagnostic artifact policy record
- Dependencies/services:
  - sampled/implicit CSG policy matrix
  - reference fixture registry
- Returns/outputs/signals:
  - diagnostic evidence
  - refusal audit signal
- UI surfaces/components:
  - reference review app
- UI fields/elements:
  - diagnostic description
  - methodology
- Reusable code plan:
  - Existing code reused as-is: CSG diagnostics
  - Additions to existing reusable library/module: reference fixture builders
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - refuses unsafe implicit fields or incompatible displacement sources
- Performance-sensitive behavior:
  - bounded diagnostic probe count
- Cross-screen reusable behavior:
  - diagnostic context reused by review details

Project readiness fields:
- Implementation owner/module:
  - `tests/reference_review_fixtures/stl_review_sources.py`
  - `src/impression/modeling/csg.py`
- Chosen defaults / parameters:
  - unsafe or unrepresentable sampled cases refuse
- Test strategy:
  - refusal diagnostic tests and review fixture registry tests
- Data ownership:
  - CSG owns refusal policy; fixture registry owns review context
- Routes:
  - sampled policy matrix to diagnostic evidence
- Reuse/extraction decision:
  - add to existing CSG/reference modules
- UI field/control inventory:
  - diagnostic description, methodology

Open questions / nuance discovered:
- The review fixture schema may need a non-STL diagnostic artifact kind before
  refusal fixtures can be represented cleanly.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 1 x 2 = 2
- Total: 24.5

Readiness blockers:
- [ ] Review fixture schema needs a diagnostic artifact policy.

Split decision:
- Review for split. Cohesion reason: refusal fixtures share one diagnostic
  artifact policy and do not produce success STLs.

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

## Change History

- 2026-07-11: Completed five manifest review/update/rescore rounds. Context:
  oversized higher-order and sampled/refusal fixture candidates were split into
  eight candidates, with no remaining candidate at or above the split threshold.
- 2026-07-11: Added patch-family reference CSG completion architecture.
  Context: unchecked patch-family reference items need a fixture-facing matrix
  and separate success, promotion, refusal, and no-fallback evidence policies.
