# Loft Self-Intersection Reference Architecture

## Overview

This document defines the architecture for `RT-LOFT-037`: a loft
self-intersection detection reference case.

The reference item is not a CSG feature. It belongs to loft planning,
execution, validity diagnostics, and reference artifact policy. The expected
result may be either a stable refusal diagnostic or a generated diagnostic
artifact, but it should not produce an apparently valid STL when the loft body
is self-intersecting.

## Related Architecture

This document extends:

- [Loft Planner / Executor Architecture](loft-planner-executor-architecture.md)
- [Loft Tolerance and Degeneracy Architecture](loft-tolerance-and-degeneracy-architecture.md)
- [Loft Ambiguity and Diagnostics Architecture](loft-ambiguity-and-diagnostics.md)
- [Model Output Reference Verification](model-output-reference-verification.md)
- [Reference CSG Gap Closure Architecture](reference-csg-gap-closure-architecture.md)

## Components

### Self-Intersection Candidate Detector

The detector identifies loft plans likely to self-intersect before or during
execution.

It owns:

- profile/path sweep envelope checks
- interval crossing checks
- local frame flip checks
- branch-to-branch proximity checks
- high-twist and near-coincident station diagnostics

### Executed Surface Validity Check

The validity check runs after surface construction but before reference export.
It owns:

- patch/patch self-intersection checks within the generated loft body
- seam crossing checks
- cap/body intersection checks
- branch manifold intersection checks

### Diagnostic Reference Producer

The producer decides how refusal evidence is represented for review.

For self-intersection cases, an STL success artifact should exist only if the
intended product is a valid surface body. Otherwise the reference case should
carry structured diagnostic evidence.

## Data Flow

```text
loft inputs
-> planner ambiguity and degeneracy checks
-> executor eligibility
-> generated SurfaceBody candidate
-> self-intersection validity check
-> valid STL fixture or diagnostic refusal evidence
```

## Cross-Domain Decisions

### Self-Intersection Is A Validity Failure Unless Explicitly Repaired

The system should not export a self-intersecting loft as a successful reference
artifact unless a repair policy has explicitly produced a valid `SurfaceBody`
and recorded the repair.

### Diagnostic Fixtures Need Review Representation

Reference review currently centers on STL artifacts. A self-intersection
refusal case needs either diagnostic artifact support or a deliberate textual
fixture state.

### Planner And Executor Both Contribute Evidence

Some self-intersections are detectable from plan data; others only appear after
surface construction. The architecture keeps both evidence sources.

## Specification Manifest for Discovery

### Candidate Spec: Loft Self-Intersection Validity Detector

Discovery purpose:
- Detect planner-visible and post-execution loft self-intersections before any
  invalid loft body can be exported as a success STL.

Responsibilities:
- Functions/methods:
  - loft self-intersection detector
  - executed loft validity checker
- Data structures/models:
  - self-intersection diagnostic
  - valid/invalid loft signal
- Dependencies/services:
  - loft planner
  - loft executor
- Returns/outputs/signals:
  - valid/invalid loft signal
  - refusal diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: loft diagnostics
  - Additions to existing reusable library/module: loft validity checks
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes loft export eligibility for invalid bodies
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded pairwise patch checks or acceleration for large loft bodies
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/loft.py`
- Chosen defaults / parameters:
  - invalid self-intersecting lofts refuse before STL export
- Test strategy:
  - unit tests for planner-detectable and post-execution self-intersections
- Data ownership:
  - loft owns validity diagnostics
- Routes:
  - loft planner/executor to validity check
- Reuse/extraction decision:
  - add to existing loft module
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Large loft bodies may need an acceleration structure later, but the reference
  fixture can start with bounded focused cases.

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
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 16.5

Readiness blockers:
- None.

Split decision:
- Review for split. Cohesion reason: the detector is one loft-validity gate
  with no fixture-schema responsibilities.

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

## Change History

- 2026-07-11: Completed five manifest review/update/rescore rounds. Context:
  the original self-intersection bundle was split into validity detection and
  diagnostic-reference fixture candidates.
- 2026-07-11: Added loft self-intersection reference architecture. Context:
  `RT-LOFT-037` remained unchecked and needs diagnostic/reference policy before
  fixture generation.
