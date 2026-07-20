# Surface Spec 409: Branching Loft CSG Policy Classification (v1.0)

Date: 2026-07-15
Status: Proposed
Primary ancestor: `../architecture/acd-branching-loft-csg-decomposition-and-recomposition-policy.md`
Architecture ancestor: `../architecture/acd-branching-loft-csg-decomposition-and-recomposition-policy.md`
Manifest source: `Branching Loft CSG Policy Classification`
Split provenance: none
Canonical status: Canonical

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one branching CSG policy classification route.

## Purpose

Classify branching loft operands before CSG execution as executable,
decomposition-required, or refused.

## Scope

Owns:

- `BranchingLoftCSGPolicyRecord`.
- `BranchingLoftCSGDiagnostic`.
- Branch policy classification and refusal diagnostics.

Does not own:

- Branch graph evidence; see Surface Spec 408.
- Decomposition/recomposition records; see Surface Spec 410.

## Split Coverage

- Parent spec: none
- Parent coverage status: not applicable
- Parent responsibilities owned by this child: not applicable
- Parent responsibilities still missing from children: none

## Implementation Routing

- Primary modules/files:
  - `src/impression/modeling/csg.py` - branch policy classifier.
  - `src/impression/modeling/loft.py` - branch evidence provider.
- Supporting modules/files:
  - `src/impression/modeling/surface.py` - shell validity records.
- GUI/QML files, if applicable:
  - not applicable.
- Reusable library/module files:
  - `src/impression/modeling/csg.py` - branch policy helper.
- Tests:
  - `tests/test_surface_csg.py` - public boolean API classification tests.

## Chosen Defaults / Parameters

- Missing or underconstrained branch graphs refuse.
- Policy classification occurs before CSG execution.
- Declared multi-body result shape remains a diagnostic policy decision until
  Surface Spec 410 resolves recomposition.

## Data Ownership

- Source of truth: loft branch graph evidence.
- Read ownership: CSG branch policy reads branch graph evidence.
- Write ownership: CSG writes policy records and refusal diagnostics.
- Derived/cache data: policy records derive from branch graph and shell evidence.
- Privacy/logging constraints: not applicable.

## Dependencies And Routes

- Domain/service dependencies:
  - Surface Spec 408 branch graph evidence.
  - loft shell validity evidence.
- Database dependencies:
  - not applicable.
- GUI route, if applicable:
  - not applicable.
- Background/concurrency route, if applicable:
  - not applicable.

## Application Integration

- App type: library-only
- User/caller surface: public boolean API consumers using branching loft operands
- Invocation route: public boolean API call to CSG eligibility and branch policy classification
- Wiring owner/module: `src/impression/modeling/csg.py`,
  `src/impression/modeling/loft.py`
- Observable result: branching loft operands are classified before execution
- Integration validation: public boolean API tests covering executable,
  decomposition-required, and refusal classifications
- Incomplete status risk: implemented in isolation if branch diagnostics are
  produced but not used by the CSG eligibility route

App-type-specific proof:

- Library-only: public boolean API tests validate policy classification.

## Reuse And Extraction Plan

- Existing code to reuse:
  - CSG eligibility diagnostics.
- Current reuse readiness:
  - add to existing library/module.
- Extraction/wrapping needed:
  - none.
- Additions to existing library/modules:
  - `src/impression/modeling/csg.py` - branch policy helper.
- New reusable modules to expose:
  - none.
- One-off code justification, if any:
  - none.

## Required DTOs / Functions / Components

- DTOs/models:
  - `BranchingLoftCSGPolicyRecord` - policy class and execution posture.
  - `BranchingLoftCSGDiagnostic` - branch refusal details.
- Functions/methods:
  - `classify_branching_loft_csg_policy(...) -> BranchingLoftCSGPolicyRecord`
  - `build_underconstrained_branch_diagnostic(...) -> BranchingLoftCSGDiagnostic`
- UI fields / visible data, if applicable:
  - not applicable.
- UI elements / controls, if applicable:
  - not applicable.
- UI components, if applicable:
  - not applicable.

## Performance Contract

- Bounded by branch graph size.
- Policy classification does not execute CSG.

## Error And State Behavior

- Underconstrained branch graphs refuse deterministically.
- Unsupported policy classes produce structured diagnostics.
- Executable classes pass to Surface Spec 410 only when evidence is complete.

## Test Strategy

- Unit tests:
  - executable, decomposition-required, and refusal branch forms.
- Service/DB tests:
  - not applicable.
- GUI/controller tests, if applicable:
  - not applicable.
- Integrated route tests:
  - public boolean API classification probes.
- Production-data rule:
  - Tests must not require production data.

## Acceptance Criteria

- Branching loft operands produce a deterministic policy class.
- Underconstrained forms refuse before execution.
- Surface Spec 410 can consume executable/decomposition-required policy records.

## Rescore And Split Review

- Manifest score: 16.5.
- IWU count: 1.
- Readiness blockers: none.
- Split decision: Review for split; classification and refusal diagnostics are one policy decision.
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
- Pass 2 - Routing: Confirmed CSG owns policy records.
- Pass 3 - Rescore: Manifest score 16.5; review-for-split band retained.
- Pass 4 - Split Review: Classification and refusal diagnostics remain one policy route.
- Pass 5 - Final: Ready as a final leaf specification.
