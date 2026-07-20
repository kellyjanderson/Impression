# Surface Spec 429: Loft Primitive Public Cut Executor Integration (v1.0)

Date: 2026-07-16
Status: Proposed
Primary ancestor: `../architecture/acd-loft-primitive-seam-shell-validity-execution.md`
Architecture ancestor: `../architecture/acd-loft-primitive-seam-shell-validity-execution.md`
Manifest source: `Loft Primitive Public Cut Executor Integration`
Split provenance: `../specifications/surface-422-loft-primitive-cut-shell-assembly-and-validity-v1_0.md`
Canonical status: Canonical
Prerequisites:
- `../specifications/surface-428-loft-primitive-runtime-validity-and-persistence-gate-v1_0.md` - public execution consumes accepted validity payloads.

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one public route integration boundary for cut-producing loft/primitive CSG.

## Manifest Field Carryover

- Discovery purpose: wire the complete kernel into `surface_boolean_result` without disrupting exact reuse or other CSG routes.
- Manifest responsibilities by category:
  - Functions/methods: `execute_loft_primitive_trim_fragment_csg(...)`, public result metadata/failure payload builder, route precedence guard.
  - Data structures/models: `LoftPrimitiveExecutionScopeRecord`, public executor diagnostic, result metadata payload.
  - Dependencies/services: predecessor cut-shell stages, `surface_boolean_result`, route selection and family gates.
  - Returns/outputs/signals: succeeded, unsupported, or invalid `SurfaceBooleanResult`.
  - UI surfaces/components and fields: not applicable.
  - Reusable code plan: reuse `surface_boolean_result`, exact reuse executor, adapter refusal path; add public loft primitive cut executor.
  - Database/async/write/security/cross-screen behavior: none.
  - Performance-sensitive behavior: bounded by prior stages; executor must not rerun CSG for metadata.
- Manifest open questions / nuance discovered:
  - Result metadata must support reference handoff but not embed generated artifact paths; this spec keeps artifact paths downstream.
- Manifest score at promotion: 19.5 on 2026-07-16.
- Manifest readiness blockers and resolution: none; resolved.
- Manifest split decision: review for split; cohesive public-route integration boundary.
- Manifest cleanup state: ready after spec promotion.

## Purpose

Expose cut-producing loft/primitive CSG through the public Boolean API once all predecessor kernel stages accept the result.

## Scope

Owns:

- `LoftPrimitiveExecutionScopeRecord`.
- Public route precedence and result envelope integration.
- Public result diagnostics and metadata for accepted, invalid, and unsupported cut cases.

Does not own:

- Individual kernel stages.
- Reference artifact generation.
- New public Boolean API surface.

## Split Coverage

- Parent spec: `../specifications/surface-422-loft-primitive-cut-shell-assembly-and-validity-v1_0.md`
- Parent coverage status: 100% covered by Surface Specs 423-431; this leaf owns public executor integration.
- Parent responsibilities owned by this child:
  - public cut executor route, result envelope, and route precedence proof.
- Parent responsibilities still missing from children:
  - none.

## Refinement History

| Iteration | Date | Scope reviewed | Files written before barrier | Updates made | IWU/score before | IWU/score after | Split decision | Split artifacts | Child re-review status | Next scope after readback |
|---:|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-07-16 | Manifest promotion and sizing | this spec and paired test spec | Promoted manifest candidate and set IWU to 1 | score 19.5 | 1 IWU | no split | none | not applicable | fixed-point ledger readback |

## Implementation Routing

- Primary modules/files:
  - `src/impression/modeling/csg.py` - public cut executor and `SurfaceBooleanResult` integration.
- Supporting modules/files:
  - `src/impression/modeling/__init__.py` - public export only if existing route requires it.
- GUI/QML files, if applicable:
  - not applicable.
- Reusable library/module files:
  - `src/impression/modeling/csg.py` - public route consumed by references and API callers.
- Tests:
  - `tests/test_surface_csg.py` - cavity difference, partial-overlap difference, union, intersection, invalid seam refusal, route precedence.

## Chosen Defaults / Parameters

- Exact reuse remains first.
- Cut executor handles true intersecting cut cases.
- Metadata supports reference handoff but does not contain artifact paths.

## Data Ownership

- Source of truth: public CSG `SurfaceBooleanResult`.
- Read ownership: reference handoff specs read accepted public result geometry.
- Write ownership: `src/impression/modeling/csg.py`.
- Derived/cache data: result metadata derived from accepted kernel payload.
- Privacy/logging constraints: public diagnostics avoid full geometry dumps.

## Dependencies And Routes

- Domain/service dependencies:
  - Surface Spec 428 validity payload.
  - `surface_boolean_result`.
  - exact reuse route and adapter refusal route.
- Database dependencies: not applicable.
- GUI route, if applicable: not applicable.
- Background/concurrency route, if applicable: not applicable.

## Prerequisite Handling

- Already implemented prerequisites: none.
- Missing prerequisite architecture: none.
- Missing prerequisite specifications: none.
- Unimplemented prerequisite specifications:
  - `../specifications/surface-428-loft-primitive-runtime-validity-and-persistence-gate-v1_0.md` - must be implemented first.
- Progression handling:
  - sequence after Surface Spec 428 and before reference handoff specs.

## Application Integration

- App type: library-only
- User/caller surface: public Boolean API `SurfaceBooleanResult`
- Invocation route: `surface_boolean_result` to exact reuse or trim-fragment cut executor
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: public API accepted/unsupported/invalid results with cut-shell metadata
- Integration validation: public API integration tests for accepted cut cases, structured refusals, and route precedence
- Incomplete status risk: all helper stages could pass while the public API still returns adapter-only refusal

App-type-specific proof:

- Library-only: public Boolean API integration tests are required; helper-only tests are insufficient.

## Reuse And Extraction Plan

- Existing code to reuse:
  - `surface_boolean_result`, exact reuse executor, adapter refusal path.
- Current reuse readiness:
  - add to existing public route module.
- Extraction/wrapping needed:
  - none.
- Additions to existing library/modules:
  - `src/impression/modeling/csg.py` - `execute_loft_primitive_trim_fragment_csg(...)` and result metadata builder.
- New reusable modules to expose:
  - none.
- One-off code justification, if any:
  - none.

## Required DTOs / Functions / Components

- DTOs/models:
  - `LoftPrimitiveExecutionScopeRecord` - exact reuse, trim-fragment cut, or structured refusal scope.
  - `LoftPrimitivePublicExecutorDiagnostic` - public route failure metadata.
- Functions/methods:
  - `execute_loft_primitive_trim_fragment_csg(...) -> SurfaceBooleanResult`.
  - `build_loft_primitive_surface_boolean_result(...) -> SurfaceBooleanResult`.
- UI fields / visible data, UI elements, and UI components:
  - not applicable.

## Performance Contract

- Executor must not rerun CSG for metadata.
- Bounded by the predecessor stage outputs.

## Error And State Behavior

- Unsupported and invalid outcomes return structured `SurfaceBooleanResult` diagnostics.
- Accepted outcomes return non-null surface-native result bodies.

## Test Strategy

- Unit tests:
  - result payload and route precedence helpers.
- Service/DB tests: not applicable.
- GUI/controller tests, if applicable: not applicable.
- Integrated route tests:
  - public API union, difference, intersection, invalid refusal, route precedence.
- Production-data rule:
  - Tests must not require production data.

## Acceptance Criteria

- True intersecting loft/primitive cut cases route through the public cut executor.
- Supported cases return accepted `SurfaceBooleanResult` objects with non-null surface-native bodies.
- Unsupported or invalid cases return structured diagnostics without hidden mesh fallback.

## Readiness Checklist

- [x] Primary ancestor and architecture ancestor are explicit.
- [x] Work Units count and basis are explicit.
- [x] Manifest fields are carried into spec sections or preserved as explicit provenance/history.
- [x] Canonical status is explicit.
- [x] Prerequisites are linked, implemented, or marked not applicable.
- [x] Split coverage is complete, or marked not applicable.
- [x] Refinement history records the latest completed review/update/rescore/split iteration and the files written before its write barrier.
- [x] Implementation owner/module is named.
- [x] Existing code reuse/extraction decision is explicit.
- [x] Existing library/module additions or new reusable module boundaries are named, or marked not applicable.
- [x] UI fields/elements are listed, or marked not applicable.
- [x] Chosen defaults are explicit.
- [x] Data source of truth and write owner are explicit.
- [x] App type and application integration route are explicit.
- [x] Integrated route validation is named.
- [x] Performance bounds are explicit, or marked not applicable.
- [x] Privacy/logging constraints are explicit, or marked not applicable.
- [x] Test strategy does not depend on production data.
- [x] Acceptance criteria are testable.
