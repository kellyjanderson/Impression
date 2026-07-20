# Surface Spec 408: Loft Branch Graph Evidence (v1.0)

Date: 2026-07-15
Status: Proposed
Primary ancestor: `../architecture/acd-branching-loft-csg-decomposition-and-recomposition-policy.md`
Architecture ancestor: `../architecture/acd-branching-loft-csg-decomposition-and-recomposition-policy.md`
Manifest source: `Loft Branch Graph Evidence`
Split provenance: none
Canonical status: Canonical

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one loft branch graph evidence route.

## Purpose

Expose loft branch graph evidence needed by branching CSG policy
classification.

## Scope

Owns:

- `LoftBranchGraphEvidence` and branch joint records.
- Branch graph evidence building from loft decomposition graph data.
- Underconstrained branch diagnostics before CSG policy classification.

Does not own:

- Branching CSG policy classification; see Surface Spec 409.
- Branch decomposition/recomposition records; see Surface Spec 410.

## Split Coverage

- Parent spec: none
- Parent coverage status: not applicable
- Parent responsibilities owned by this child: not applicable
- Parent responsibilities still missing from children: none

## Implementation Routing

- Primary modules/files:
  - `src/impression/modeling/loft.py` - branch evidence builder.
- Supporting modules/files:
  - `src/impression/modeling/csg.py` - consuming branch policy probe.
- GUI/QML files, if applicable:
  - not applicable.
- Reusable library/module files:
  - `src/impression/modeling/loft.py` - branch graph evidence helper.
- Tests:
  - `tests/test_loft_surface_body.py` - branch graph evidence tests.
  - `tests/test_surface_csg.py` - branch policy probe.

## Chosen Defaults / Parameters

- Branch joints require explicit ownership and transition membership.
- Missing or underconstrained branch graphs refuse before CSG execution.
- Branch graph evidence is separate from CSG policy.

## Data Ownership

- Source of truth: loft decomposition graph.
- Read ownership: CSG branch policy reads branch graph evidence.
- Write ownership: loft writes branch graph evidence.
- Derived/cache data: branch graph evidence can be recomputed from loft
  decomposition records and shell evidence.
- Privacy/logging constraints: not applicable.

## Dependencies And Routes

- Domain/service dependencies:
  - `../architecture/acd-loft-shell-connectivity-and-closure-evidence.md`.
  - loft decomposition graph.
- Database dependencies:
  - not applicable.
- GUI route, if applicable:
  - not applicable.
- Background/concurrency route, if applicable:
  - not applicable.

## Application Integration

- App type: library-only
- User/caller surface: public boolean API consumers using branching loft operands
- Invocation route: loft executor branch/decomposition evidence to CSG branch policy
- Wiring owner/module: `src/impression/modeling/loft.py`
- Observable result: branching loft operands expose branch graph evidence or
  branch-specific diagnostics before CSG policy classification
- Integration validation: loft executor tests plus public boolean API
  branch-policy probes
- Incomplete status risk: implemented in isolation if branch evidence is not
  exposed to CSG policy

App-type-specific proof:

- Library-only: public boolean API branch-policy probes validate consumption.

## Reuse And Extraction Plan

- Existing code to reuse:
  - N-to-M decomposition concepts.
- Current reuse readiness:
  - add to existing library/module.
- Extraction/wrapping needed:
  - none.
- Additions to existing library/modules:
  - `src/impression/modeling/loft.py` - branch graph evidence helper.
- New reusable modules to expose:
  - none.
- One-off code justification, if any:
  - none.

## Required DTOs / Functions / Components

- DTOs/models:
  - `LoftBranchGraphEvidence` - branch nodes, joints, transitions, and membership.
  - branch joint record - ownership and transition membership.
- Functions/methods:
  - `build_loft_branch_graph_evidence(...) -> LoftBranchGraphEvidence`
  - `build_branch_joint_diagnostic(...) -> Diagnostic`
- UI fields / visible data, if applicable:
  - not applicable.
- UI elements / controls, if applicable:
  - not applicable.
- UI components, if applicable:
  - not applicable.

## Performance Contract

- Bounded by station, branch, and transition count.
- No tessellation is allowed for branch graph proof.

## Error And State Behavior

- Missing branch ownership yields structured diagnostics.
- Underconstrained branch topology refuses before boolean execution.
- Evidence is read-only once attached to the loft result.

## Test Strategy

- Unit tests:
  - executable branch graph forms and underconstrained forms.
- Service/DB tests:
  - not applicable.
- GUI/controller tests, if applicable:
  - not applicable.
- Integrated route tests:
  - public boolean API branch-policy probes.
- Production-data rule:
  - Tests must not require production data.

## Acceptance Criteria

- Branching lofts expose branch graph evidence.
- Underconstrained branch graphs refuse with deterministic diagnostics.
- Surface Spec 409 can consume branch evidence without re-deriving topology.

## Rescore And Split Review

- Manifest score: 13.5.
- IWU count: 1.
- Readiness blockers: none.
- Split decision: Small; branch graph evidence stays separate from CSG policy.
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
- Pass 2 - Routing: Confirmed loft owns branch graph evidence.
- Pass 3 - Rescore: Manifest score 13.5; no split required.
- Pass 4 - Blocker Review: Shell evidence sequence is represented by ACD lineage.
- Pass 5 - Final: Ready as a final leaf specification.
