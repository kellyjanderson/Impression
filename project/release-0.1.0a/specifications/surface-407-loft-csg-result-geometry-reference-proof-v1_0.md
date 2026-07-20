# Surface Spec 407: Loft CSG Result Geometry Reference Proof (v1.0)

Date: 2026-07-15
Status: Proposed
Primary ancestor: `../architecture/acd-single-shell-loft-csg-operation-route.md`
Architecture ancestor: `../architecture/acd-single-shell-loft-csg-operation-route.md`
Manifest source: `Loft CSG Result Geometry Reference Proof`
Split provenance: none
Canonical status: Canonical

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one reference proof route from returned SurfaceBody to dirty artifact evidence.

## Purpose

Prove that the single-shell loft CSG route produces reusable result geometry for
reference artifacts and downstream evidence work.

## Scope

Owns:

- `LoftCSGReferenceGeometryProof`.
- Dirty STL/reference artifact signal checks from returned `SurfaceBody`.
- Deterministic proof records for downstream evidence.

Does not own:

- CSG execution; see Surface Spec 406.
- Section evidence bundles; see Surface Specs 417-419.

## Split Coverage

- Parent spec: none
- Parent coverage status: not applicable
- Parent responsibilities owned by this child: not applicable
- Parent responsibilities still missing from children: none

## Implementation Routing

- Primary modules/files:
  - `tests/reference_review_fixtures/stl_review_sources.py` - fixture proof helpers.
  - `tests/reference_images.py` - dirty artifact generation route.
- Supporting modules/files:
  - `src/impression/modeling/csg.py` - public CSG result source.
- GUI/QML files, if applicable:
  - not applicable.
- Reusable library/module files:
  - `tests/reference_review_fixtures/stl_review_sources.py` - loft CSG proof helper.
- Tests:
  - `tests/test_reference_stl_generation.py` - dirty proof artifact tests.

## Chosen Defaults / Parameters

- Dirty artifacts are proof inputs, not promoted gold completion evidence.
- Proof records must identify the source result geometry.
- No synthetic geometry is allowed.

## Data Ownership

- Source of truth: public CSG `SurfaceBody` result.
- Read ownership: reference fixture generation reads result geometry.
- Write ownership: reference workflow writes dirty artifacts and proof records.
- Derived/cache data: dirty artifacts can be regenerated from the public CSG result.
- Privacy/logging constraints: generated paths stay under dirty reference roots.

## Dependencies And Routes

- Domain/service dependencies:
  - Surface Spec 406 returned `SurfaceBody`.
  - reference artifact lifecycle.
  - STL tessellation boundary.
- Database dependencies:
  - not applicable.
- GUI route, if applicable:
  - not applicable.
- Background/concurrency route, if applicable:
  - not applicable.

## Application Integration

- App type: workflow
- User/caller surface: reference update workflow for loft CSG fixtures
- Invocation route: returned `SurfaceBody` to reference artifact generation
- Wiring owner/module: `tests/reference_review_fixtures/stl_review_sources.py`,
  `tests/reference_images.py`
- Observable result: dirty STL/reference artifacts exist for the supported loft
  CSG route and identify the source result geometry
- Integration validation: reference update test proving dirty artifact
  generation from the returned surface body
- Incomplete status risk: implemented in isolation if proof records are
  generated from synthetic data instead of the public CSG result

App-type-specific proof:

- Workflow: reference generation tests validate side effects and artifact paths.

## Reuse And Extraction Plan

- Existing code to reuse:
  - reference artifact lifecycle helpers.
- Current reuse readiness:
  - add to existing library/module.
- Extraction/wrapping needed:
  - none.
- Additions to existing library/modules:
  - `tests/reference_review_fixtures/stl_review_sources.py` - loft CSG proof helper.
- New reusable modules to expose:
  - none.
- One-off code justification, if any:
  - none.

## Required DTOs / Functions / Components

- DTOs/models:
  - `LoftCSGReferenceGeometryProof` - fixture id, result source, dirty path, and no-hidden-mesh evidence.
- Functions/methods:
  - `build_loft_csg_result_geometry_proof(...) -> LoftCSGReferenceGeometryProof`
  - `check_dirty_stl_artifact_signal(...) -> Diagnostic`
- UI fields / visible data, if applicable:
  - not applicable.
- UI elements / controls, if applicable:
  - not applicable.
- UI components, if applicable:
  - not applicable.

## Performance Contract

- Bounded by tessellation output size.
- Reference proof generation must not repeat CSG execution unnecessarily.

## Error And State Behavior

- Missing dirty artifact paths produce deterministic diagnostics.
- Proof generation refuses synthetic or mesh-first input.
- Dirty artifacts remain unreviewed until promotion.

## Test Strategy

- Unit tests:
  - proof record construction and dirty path validation.
- Service/DB tests:
  - not applicable.
- GUI/controller tests, if applicable:
  - not applicable.
- Integrated route tests:
  - public boolean API result to dirty reference artifact.
- Production-data rule:
  - Tests must not require production data.

## Acceptance Criteria

- Proof records are generated from public CSG `SurfaceBody` results.
- Dirty STL/reference artifacts exist at deterministic paths.
- No-hidden-mesh-fallback proof is preserved.

## Rescore And Split Review

- Manifest score: 19.5.
- IWU count: 1.
- Readiness blockers: none.
- Split decision: Review for split; proof construction and dirty artifact signal checking remain one reference-generation route.
- Review update: checked after promotion from ACD manifest; no child spec is required before implementation.

## Readiness Checklist

- [x] Primary ancestor and architecture ancestor are explicit.
- [x] Canonical status is explicit.
- [x] Split coverage is complete, or marked not applicable.
- [x] Implementation owner/module is named.
- [x] Existing code reuse/extraction decision is explicit.
- [x] Existing library/module additions or new reusable module boundaries are named, or marked not applicable.
- [x] UI fields/elements are listed, or marked not applicable.
- [x] Chosen defaults are explicit.
- [x] Data source of truth and write owner are explicit.
- [x] GUI/concurrency route is explicit, or marked not applicable.
- [x] App type and application integration route are explicit.
- [x] Integrated route validation is named.
- [x] GUI/console/API-service/mixed/library-only proof matches the app type.
- [x] Performance bounds are explicit, or marked not applicable.
- [x] Privacy/logging constraints are explicit, or marked not applicable.
- [x] Test strategy does not depend on production data.
- [x] Acceptance criteria are testable.

## Review History

- Pass 1 - Template: Promoted from ACD manifest.
- Pass 2 - Routing: Confirmed workflow proof is separate from CSG execution.
- Pass 3 - Rescore: Manifest score 19.5; review-for-split band retained.
- Pass 4 - Split Review: Proof construction and dirty signal checking remain one route.
- Pass 5 - Final: Ready as a final leaf specification.
