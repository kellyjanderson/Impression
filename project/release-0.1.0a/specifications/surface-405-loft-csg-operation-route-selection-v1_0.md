# Surface Spec 405: Loft CSG Operation Route Selection (v1.0)

Date: 2026-07-15
Status: Proposed
Primary ancestor: `../architecture/acd-single-shell-loft-csg-operation-route.md`
Architecture ancestor: `../architecture/acd-single-shell-loft-csg-operation-route.md`
Manifest source: `Loft CSG Operation Route Selection`
Split provenance: none
Canonical status: Canonical

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one CSG route selection outcome.

## Purpose

Select a surface-native CSG route for eligible single-shell loft operands.

## Scope

Owns:

- `LoftCSGOperationRouteRecord`.
- Loft-aware route selection after CSG eligibility acceptance.
- Unsupported loft pairing diagnostics.

Does not own:

- Loft shell evidence; see Surface Specs 403 and 404.
- Route execution; see Surface Spec 406.

## Split Coverage

- Parent spec: none
- Parent coverage status: not applicable
- Parent responsibilities owned by this child: not applicable
- Parent responsibilities still missing from children: none

## Implementation Routing

- Primary modules/files:
  - `src/impression/modeling/csg.py` - route selector and diagnostics.
- Supporting modules/files:
  - `src/impression/modeling/loft.py` - eligibility evidence source.
- GUI/QML files, if applicable:
  - not applicable.
- Reusable library/module files:
  - `src/impression/modeling/csg.py` - loft route selection helper.
- Tests:
  - `tests/test_surface_csg.py` - route selection tests through public boolean API.

## Chosen Defaults / Parameters

- Only eligible single-shell lofts may select an execution route.
- Unsupported pairings refuse before execution.
- The first supported pairing should be the smallest analytic primitive pairing
  that proves the route without branching behavior.

## Data Ownership

- Source of truth: CSG route selector.
- Read ownership: CSG execution reads route records.
- Write ownership: CSG owns route records and unsupported diagnostics.
- Derived/cache data: route records are derived from eligibility evidence and
  operand family classification.
- Privacy/logging constraints: not applicable.

## Dependencies And Routes

- Domain/service dependencies:
  - Surface Spec 403 boundary graph evidence.
  - Surface Spec 404 closure and cap evidence.
  - surface CSG route registry.
- Database dependencies:
  - not applicable.
- GUI route, if applicable:
  - not applicable.
- Background/concurrency route, if applicable:
  - not applicable.

## Application Integration

- App type: library-only
- User/caller surface: public boolean API consumers using single-shell loft operands
- Invocation route: public boolean API call to route selector after CSG
  eligibility acceptance
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: supported loft/primitive pairings select a surface-native
  route; unsupported pairings refuse with structured diagnostics
- Integration validation: public boolean API route-selection tests
- Incomplete status risk: implemented in isolation if route records are not
  used by CSG execution

App-type-specific proof:

- Library-only: public boolean API tests validate route selection.

## Reuse And Extraction Plan

- Existing code to reuse:
  - surface CSG route registry.
- Current reuse readiness:
  - add to existing library/module.
- Extraction/wrapping needed:
  - none.
- Additions to existing library/modules:
  - `src/impression/modeling/csg.py` - loft route selector.
- New reusable modules to expose:
  - none.
- One-off code justification, if any:
  - none.

## Required DTOs / Functions / Components

- DTOs/models:
  - `LoftCSGOperationRouteRecord` - operation, operands, eligibility evidence,
    and selected solver path.
- Functions/methods:
  - `select_loft_csg_route(...) -> LoftCSGOperationRouteRecord`
  - `build_unsupported_loft_pairing_diagnostic(...) -> Diagnostic`
- UI fields / visible data, if applicable:
  - not applicable.
- UI elements / controls, if applicable:
  - not applicable.
- UI components, if applicable:
  - not applicable.

## Performance Contract

- Bounded by operand patch and route count.
- Route selection must not tessellate operands.

## Error And State Behavior

- Missing eligibility evidence refuses before route selection.
- Unsupported operand pairings report deterministic diagnostics.
- Route selection never falls back to mesh execution.

## Test Strategy

- Unit tests:
  - supported route selection and unsupported diagnostics.
- Service/DB tests:
  - not applicable.
- GUI/controller tests, if applicable:
  - not applicable.
- Integrated route tests:
  - public boolean API route-selection tests.
- Production-data rule:
  - Tests must not require production data.

## Acceptance Criteria

- Eligible single-shell loft/primitive operands select a CSG route.
- Unsupported pairings refuse before execution with structured diagnostics.
- Route records are consumed by Surface Spec 406 execution.

## Rescore And Split Review

- Manifest score: 12.5.
- IWU count: 1.
- Readiness blockers: none.
- Split decision: Small; selector stays separate from execution and proof.
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
- Pass 2 - Routing: Confirmed selector is separate from executor.
- Pass 3 - Rescore: Manifest score 12.5; no split required.
- Pass 4 - Blocker Review: Predecessor evidence is covered by Surface Specs 403 and 404.
- Pass 5 - Final: Ready as a final leaf specification.
