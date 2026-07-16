# Surface Spec 404: Loft Closure And Cap Validity Evidence (v1.0)

Date: 2026-07-15
Status: Proposed
Primary ancestor: `../architecture/acd-loft-shell-connectivity-and-closure-evidence.md`
Architecture ancestor: `../architecture/acd-loft-shell-connectivity-and-closure-evidence.md`
Manifest source: `Loft Closure And Cap Validity Evidence`
Split provenance: none
Canonical status: Canonical

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one closure and cap validity eligibility contract.

## Purpose

Prove capped loft bodies are closed and cap-valid before they can enter
closed-body CSG.

## Scope

Owns:

- `LoftClosureEvidenceRecord` and `LoftCapValidityRecord`.
- Closure evidence building and cap validity classification.
- CSG eligibility handoff for closed-valid loft evidence.

Does not own:

- Boundary graph construction; see Surface Spec 403.
- CSG route selection or execution.

## Split Coverage

- Parent spec: none
- Parent coverage status: not applicable
- Parent responsibilities owned by this child: not applicable
- Parent responsibilities still missing from children: none

## Implementation Routing

- Primary modules/files:
  - `src/impression/modeling/loft.py` - closure and cap evidence.
  - `src/impression/modeling/csg.py` - eligibility gate consumption.
- Supporting modules/files:
  - `src/impression/modeling/surface.py` - cap patches and trim loops.
- GUI/QML files, if applicable:
  - not applicable.
- Reusable library/module files:
  - `src/impression/modeling/loft.py` - closure evidence helpers.
- Tests:
  - `tests/test_loft_surface_body.py` - closure/cap evidence tests.
  - `tests/test_surface_csg.py` - boolean eligibility tests.

## Chosen Defaults / Parameters

- Capped lofts require cap evidence.
- Uncapped lofts refuse for closed-body CSG.
- CSG consumes loft evidence; it does not guess closedness.

## Data Ownership

- Source of truth: loft executor output metadata.
- Read ownership: CSG eligibility reads closure evidence from returned bodies.
- Write ownership: loft executor writes evidence; CSG writes acceptance/refusal
  diagnostics.
- Derived/cache data: closure evidence can be recomputed from boundary graph,
  cap metadata, and trim loop orientation.
- Privacy/logging constraints: not applicable.

## Dependencies And Routes

- Domain/service dependencies:
  - Surface Spec 403 boundary graph evidence.
  - cap patch metadata.
  - trim loop orientation.
- Database dependencies:
  - not applicable.
- GUI route, if applicable:
  - not applicable.
- Background/concurrency route, if applicable:
  - not applicable.

## Application Integration

- App type: library-only
- User/caller surface: public modeling API consumers calling boolean operations
  with loft-authored `SurfaceBody` operands
- Invocation route: loft executor evidence to CSG eligibility gate during
  public boolean API calls
- Wiring owner/module: `src/impression/modeling/loft.py`,
  `src/impression/modeling/csg.py`
- Observable result: capped single-shell loft bodies pass closure eligibility;
  uncapped or invalid cap cases refuse with structured diagnostics
- Integration validation: loft closure tests plus public boolean API
  eligibility tests
- Incomplete status risk: implemented in isolation if CSG still derives or
  guesses closedness instead of consuming loft evidence

App-type-specific proof:

- Library-only: public boolean API eligibility probes validate the route.

## Reuse And Extraction Plan

- Existing code to reuse:
  - planar cap patches and trim loops.
  - Surface Spec 403 boundary graph evidence.
- Current reuse readiness:
  - add to existing library/module.
- Extraction/wrapping needed:
  - none.
- Additions to existing library/modules:
  - `src/impression/modeling/loft.py` - cap/closure evidence helpers.
- New reusable modules to expose:
  - none.
- One-off code justification, if any:
  - none.

## Required DTOs / Functions / Components

- DTOs/models:
  - `LoftClosureEvidenceRecord` - closed-valid signal and open-loop diagnostics.
  - `LoftCapValidityRecord` - cap presence, loop ownership, orientation, and adjacency.
- Functions/methods:
  - `build_loft_closure_evidence(...) -> LoftClosureEvidenceRecord`
  - `classify_loft_cap_validity(...) -> LoftCapValidityRecord`
- UI fields / visible data, if applicable:
  - not applicable.
- UI elements / controls, if applicable:
  - not applicable.
- UI components, if applicable:
  - not applicable.

## Performance Contract

- Bounded by cap loop and seam count.
- No mesh tessellation is allowed for closure proof.

## Error And State Behavior

- Invalid caps refuse with structured diagnostics.
- Incomplete cap evidence is not treated as successful closure.
- Uncapped lofts produce a closed-body CSG refusal diagnostic.

## Test Strategy

- Unit tests:
  - capped valid, uncapped refusal, invalid cap loop, orientation mismatch.
- Service/DB tests:
  - not applicable.
- GUI/controller tests, if applicable:
  - not applicable.
- Integrated route tests:
  - public boolean API eligibility tests.
- Production-data rule:
  - Tests must not require production data.

## Acceptance Criteria

- Closed-valid capped lofts expose closure and cap evidence.
- Invalid or uncapped lofts refuse closed-body CSG deterministically.
- CSG eligibility consumes loft evidence without topology guessing.

## Rescore And Split Review

- Manifest score: 18.
- IWU count: 1.
- Readiness blockers: none.
- Split decision: Review for split; closure evidence and cap validity stay together because CSG needs one closed-body eligibility decision.
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

- Pass 1 - Template: Promoted from ACD manifest with predecessor Surface Spec 403.
- Pass 2 - Routing: Confirmed loft owns evidence and CSG owns acceptance.
- Pass 3 - Rescore: Manifest score 18; review-for-split band retained.
- Pass 4 - Split Review: Closure and cap validity remain one eligibility contract.
- Pass 5 - Final: Ready as a final leaf specification.
