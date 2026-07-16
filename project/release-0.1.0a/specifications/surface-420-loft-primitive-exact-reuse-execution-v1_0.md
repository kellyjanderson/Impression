# Surface Spec 420: Loft Primitive Exact Reuse Execution (v1.0)

Date: 2026-07-16
Status: Proposed
Primary ancestor: `../architecture/acd-loft-primitive-trim-fragment-execution-adapter.md`
Architecture ancestor: `../architecture/acd-loft-primitive-trim-fragment-execution-adapter.md`
Manifest source: `Loft Primitive Exact Reuse Execution`
Split provenance: `../specifications/surface-406-single-shell-loft-primitive-csg-execution-v1_0.md`
Canonical status: Canonical
Prerequisites:
- `../specifications/surface-405-loft-csg-operation-route-selection-v1_0.md` - route selection must identify eligible executor-authored loft/primitive operands before exact reuse execution can run.

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one exact no-cut/containment loft-primitive execution route.

## Purpose

Preserve and validate the exact no-cut/containment loft-primitive execution subset without treating it as the full cut-producing route.

## Scope

Owns:

- Exact loft/primitive no-cut, containment, and disjoint-union execution decisions.
- `LoftPrimitiveExecutionScopeRecord` and exact-result metadata.
- Public boolean API integration for exact reuse union, difference, and intersection cases.

Does not own:

- Intersecting trim-fragment cut execution; see Surface Specs 421 and 422.
- Branching loft, loft/loft, provenance/color, or section-evidence routes.

## Split Coverage

- Parent spec: `../specifications/surface-406-single-shell-loft-primitive-csg-execution-v1_0.md`
- Parent coverage status: 100% covered by Surface Specs 420, 421, and 422.
- Parent responsibilities owned by this child:
  - exact no-cut and containment loft/primitive execution
  - exact result metadata and no-hidden-mesh proof
  - public boolean API proof for exact reuse cases
- Parent responsibilities still missing from children:
  - none.

## Implementation Routing

- Primary modules/files:
  - `src/impression/modeling/csg.py` - exact reuse executor, execution scope record, result metadata.
- Supporting modules/files:
  - `src/impression/modeling/loft.py` - executor-authored loft evidence consumed by eligibility and route selection.
- GUI/QML files, if applicable:
  - not applicable.
- Reusable library/module files:
  - `src/impression/modeling/csg.py` - public route helper and DTO exports.
- Tests:
  - `tests/test_surface_csg.py` - public boolean API exact union, difference, intersection, and no-hidden-mesh assertions.

## Chosen Defaults / Parameters

- Exact reuse applies only when the result geometry is exactly an existing loft body, an existing primitive body, an empty result, or an exact disjoint union.
- Intersecting cut cases refuse or defer to Surface Specs 421 and 422.
- No tessellation, rasterization, or hidden mesh fallback is allowed.

## Data Ownership

- Source of truth: `src/impression/modeling/csg.py` route execution records.
- Read ownership: downstream CSG result proof, provenance, and reference fixture code may read `loft_primitive_csg` metadata.
- Write ownership: CSG writes execution scope and result metadata.
- Derived/cache data: exact result records can be recomputed from operands, route selection, and body relation checks.
- Privacy/logging constraints: no sensitive data; no operand geometry dumps in diagnostics.

## Dependencies And Routes

- Domain/service dependencies:
  - Surface Spec 405 route selection
  - exact body containment/no-cut relation helpers
  - surface boolean result validity gate
- Database dependencies:
  - not applicable.
- GUI route, if applicable:
  - not applicable.
- Background/concurrency route, if applicable:
  - not applicable.

## Prerequisite Handling

- Already implemented prerequisites:
  - `../specifications/surface-405-loft-csg-operation-route-selection-v1_0.md` - route selection is implemented and progression-complete.
- Missing prerequisite architecture:
  - none.
- Missing prerequisite specifications:
  - none.
- Unimplemented prerequisite specifications:
  - none.
- Progression handling:
  - current item may proceed before Surface Specs 421 and 422 because exact reuse is an independent route.

## Application Integration

- App type: library-only
- User/caller surface: public boolean API consumers using loft/primitive operands
- Invocation route: public boolean API to route selection to exact reuse execution to result validity gate
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: public boolean API returns exact existing-body or empty `SurfaceBooleanResult` values with loft primitive metadata and no-hidden-mesh proof
- Integration validation: public boolean API union, difference, and intersection tests for exact reuse cases
- Incomplete status risk: complete only for exact reuse; not a substitute for trim-fragment cut execution

App-type-specific proof:

- Library-only: public boolean API tests validate the consuming route.

## Reuse And Extraction Plan

- Existing code to reuse:
  - `surface_boolean_result(...)` and result finalizer - public route and validity gate.
  - existing exact body relation helpers - no-cut and containment classification.
- Current reuse readiness:
  - add to existing library/module.
- Extraction/wrapping needed:
  - none.
- Additions to existing library/modules:
  - `src/impression/modeling/csg.py` - exact loft primitive executor and execution scope DTO.
- New reusable modules to expose:
  - none.
- One-off code justification, if any:
  - none.

## Required DTOs / Functions / Components

- DTOs/models:
  - `LoftPrimitiveExecutionScopeRecord` - records exact reuse, trim-fragment, or refusal execution scope.
  - `LoftCSGResultGeometryRecord` - records result shell count, patch count, classification, and no-mesh proof.
- Functions/methods:
  - `execute_single_shell_loft_primitive_csg(...) -> SurfaceBooleanResult | None` - executes exact reuse cases.
- UI fields / visible data, if applicable:
  - not applicable.
- UI elements / controls, if applicable:
  - not applicable.
- UI components, if applicable:
  - not applicable.

## Performance Contract

- Bounded by operand count and body metadata.
- No tessellation or global patch scan beyond existing bounds and metadata queries.

## Error And State Behavior

- Intersecting cut cases are not silently approximated.
- Unsupported exact cases return structured unsupported results through the public boolean API.
- Invalid exact results fail through the existing CSG validity gate.

## Test Strategy

- Unit tests:
  - execution scope and result metadata payloads.
- Service/DB tests:
  - not applicable.
- GUI/controller tests, if applicable:
  - not applicable.
- Integrated route tests:
  - public boolean API exact union, difference, and intersection tests.
- Production-data rule:
  - Tests must not require production data.

## Acceptance Criteria

- Exact no-cut/containment loft-primitive union, difference, and intersection routes return correct `SurfaceBooleanResult` values.
- Result metadata records the loft primitive route and no-hidden-mesh proof.
- Intersecting cut cases are not claimed complete by this spec.

## Rescore And Split Review

- Manifest score: 13.5.
- IWU count: 1.
- Readiness blockers: none.
- Split decision: Small; remains one exact-reuse execution leaf.
- Review update: post-promotion review confirmed Surface Spec 406 split coverage is complete through Surface Specs 420, 421, and 422; no child split required.

## Readiness Checklist

- [x] Primary ancestor and architecture ancestor are explicit.
- [x] Canonical status is explicit.
- [x] Prerequisites are linked, implemented, or marked not applicable.
- [x] Missing prerequisite architecture has an ACD link, or is marked not applicable.
- [x] Missing prerequisite behavior has a final spec link, or is marked not applicable.
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

- Pass 1 - Template: promoted from ACD manifest with all readiness fields.
- Pass 2 - Prerequisites: linked Surface Spec 405 and confirmed no missing architecture.
- Pass 3 - Rescore: manifest score 13.5; no split required.
- Pass 4 - Split Review: exact reuse remains separate from trim-fragment cut execution.
- Pass 5 - Final: ready as a canonical implementation leaf.
- Post-promotion review - 2026-07-16: rechecked IWU count, split coverage, prerequisites, and readiness blockers; remains score 13.5, 1 IWU, no split.
