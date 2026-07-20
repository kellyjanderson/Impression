# Surface Spec 410: Branch Decomposition And Recomposition Records (v1.0)

Date: 2026-07-15
Status: Proposed
Primary ancestor: `../architecture/acd-branching-loft-csg-decomposition-and-recomposition-policy.md`
Architecture ancestor: `../architecture/acd-branching-loft-csg-decomposition-and-recomposition-policy.md`
Manifest source: `Branch Decomposition And Recomposition Records`
Split provenance: none
Canonical status: Canonical

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one branch decomposition and recomposition execution contract.

## Purpose

Define executable branch sub-body CSG plans and recomposition evidence for
decomposable branching lofts.

## Scope

Owns:

- `BranchDecompositionPlan`.
- `BranchSubBodyCSGPlan`.
- `BranchRecompositionRecord`.
- Branch decomposition adapter and recomposition validator.

Does not own:

- Branch policy classification; see Surface Spec 409.
- Final provenance/color resolution; see Surface Specs 411 and 412.

## Split Coverage

- Parent spec: none
- Parent coverage status: not applicable
- Parent responsibilities owned by this child: not applicable
- Parent responsibilities still missing from children: none

## Implementation Routing

- Primary modules/files:
  - `src/impression/modeling/loft.py` - decomposition records.
  - `src/impression/modeling/csg.py` - sub-body CSG and recomposition acceptance.
- Supporting modules/files:
  - `src/impression/modeling/surface.py` - shell validity records.
- GUI/QML files, if applicable:
  - not applicable.
- Reusable library/module files:
  - `src/impression/modeling/csg.py` - CSG-specific branch records.
- Tests:
  - `tests/test_surface_csg.py` - decomposition/recomposition public API probes.

## Chosen Defaults / Parameters

- Sub-body results carry source branch ids and recomposition seams.
- Missing recomposition evidence refuses the branch route.
- Canonical result shape at a branch joint must be explicit in the
  recomposition record.

## Data Ownership

- Source of truth: loft decomposition plan and CSG recomposition record.
- Read ownership: CSG execution reads decomposition plans and provenance reads
  recomposition records.
- Write ownership: loft writes decomposition; CSG writes operation and
  recomposition acceptance.
- Derived/cache data: sub-body plans derive from branch graph and policy records.
- Privacy/logging constraints: not applicable.

## Dependencies And Routes

- Domain/service dependencies:
  - Surface Spec 409 branch policy classification.
  - loft shell connectivity and closure evidence.
  - Surface Specs 411-412 provenance/color sequence.
- Database dependencies:
  - not applicable.
- GUI route, if applicable:
  - not applicable.
- Background/concurrency route, if applicable:
  - not applicable.

## Application Integration

- App type: library-only
- User/caller surface: public boolean API consumers using decomposable branching loft operands
- Invocation route: branch policy classification to decomposition plan to
  sub-body CSG to recomposition validation
- Wiring owner/module: `src/impression/modeling/loft.py`,
  `src/impression/modeling/csg.py`
- Observable result: decomposable branch operands produce sub-body CSG plans and
  recomposition evidence or refuse with a branch-specific diagnostic
- Integration validation: public boolean API tests with bounded decomposable
  branch fixtures and underconstrained refusal fixtures
- Incomplete status risk: implemented in isolation if records do not feed CSG
  execution and recomposition validation

App-type-specific proof:

- Library-only: public boolean API tests validate branch decomposition and
  recomposition.

## Reuse And Extraction Plan

- Existing code to reuse:
  - N-to-M decomposition concepts.
- Current reuse readiness:
  - add to existing library/module.
- Extraction/wrapping needed:
  - none.
- Additions to existing library/modules:
  - CSG-specific branch records.
- New reusable modules to expose:
  - none.
- One-off code justification, if any:
  - none.

## Required DTOs / Functions / Components

- DTOs/models:
  - `BranchDecompositionPlan` - branch sub-body decomposition.
  - `BranchSubBodyCSGPlan` - per-sub-body boolean plan.
  - `BranchRecompositionRecord` - recomposition seams, validity, and result shape.
- Functions/methods:
  - `plan_branch_subbody_csg(...) -> BranchDecompositionPlan`
  - `validate_branch_recomposition(...) -> BranchRecompositionRecord`
- UI fields / visible data, if applicable:
  - not applicable.
- UI elements / controls, if applicable:
  - not applicable.
- UI components, if applicable:
  - not applicable.

## Performance Contract

- Bounded by branch/sub-body count.
- No branch-level mesh fallback is allowed.

## Error And State Behavior

- Missing branch ids or recomposition seams refuse deterministically.
- Underconstrained branch inputs preserve refusal diagnostics.
- Recomposition failures do not return partial success as a valid body.

## Test Strategy

- Unit tests:
  - decomposition records, sub-body plans, recomposition validity.
- Service/DB tests:
  - not applicable.
- GUI/controller tests, if applicable:
  - not applicable.
- Integrated route tests:
  - public boolean API decomposable and refusal fixtures.
- Production-data rule:
  - Tests must not require production data.

## Acceptance Criteria

- Decomposable branch operands produce executable sub-body plans.
- Recomposition records prove the result shape or refuse.
- Public boolean API tests cover success and underconstrained refusal routes.

## Rescore And Split Review

- Manifest score: 18.5.
- IWU count: 1.
- Readiness blockers: none.
- Split decision: Review for split; decomposition and recomposition records must validate together.
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
- Pass 2 - Routing: Confirmed decomposition and recomposition form one execution contract.
- Pass 3 - Rescore: Manifest score 18.5; review-for-split band retained.
- Pass 4 - Split Review: Kept cohesive because records must validate together.
- Pass 5 - Final: Ready as a final leaf specification.
