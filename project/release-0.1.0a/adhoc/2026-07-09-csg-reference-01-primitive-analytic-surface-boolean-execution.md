# CSG Reference Spec 01: Primitive Analytic Surface Boolean Execution

Date: 2026-07-09
Status: Draft
Path: ad-hoc-path work

## Summary

Implement surfaced boolean execution for the primitive analytic cases needed by
the reference expansion plan: box/sphere, box/cylinder, sphere/box,
cylinder/box, and orthogonal cylinder/cylinder operations.

This extends the existing `SurfaceBooleanResult` route. It does not introduce a
mesh fallback.

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one capability boundary: analytic primitive pair execution for currently planned primitive CSG references.

## Scope

In scope:

- partial-overlap `union`, `difference`, and `intersection` for box/sphere
  pairs
- partial-overlap `difference` for box/cylinder and cylinder/box pairs
- `union` and `intersection` for orthogonal cylinder/cylinder pairs
- tangent box/sphere diagnostics used by `RT-CSG-008`
- surface-native result metadata and no-hidden-mesh fallback checks

Out of scope:

- multi-operand chaining
- lofted body CSG
- B-spline, NURBS, sweep, subdivision, implicit, heightmap, and displacement
  result preservation
- artifact generation beyond focused dirty STL fixtures after the surfaced
  result succeeds

## Implementation Boundary

Owner modules:

- `src/impression/modeling/csg.py`
- existing surface primitive builders in `src/impression/modeling/primitives.py`

Reuse:

- Reuse `prepare_surface_boolean_operands`.
- Reuse `prepare_surface_boolean_difference_operands`.
- Reuse `plan_prepared_surface_csg_operation`.
- Reuse existing planar/revolution and revolution/revolution intersection
  records.
- Add code only to the existing CSG implementation or a private CSG helper
  module if extraction reduces the size of `csg.py`.

Public API:

- No new public boolean API.
- Existing public helpers continue to return `SurfaceBooleanResult` for
  surfaced inputs.

## Required Behavior

- Supported primitive pairs return `status="succeeded"` with a valid
  `SurfaceBody`.
- Tangent cases either produce a valid exact no-volume relation or return a
  deterministic tangent diagnostic. They must not produce unstable sliver
  geometry.
- Result bodies preserve enough source metadata for reference review to explain
  operand provenance.
- Unsupported primitive geometry returns `status="unsupported"` with a
  deterministic reason.
- Any accidental mesh fallback in surfaced paths fails tests.

## Reference Items Unblocked

- `RT-CSG-001` cube union sphere
- `RT-CSG-002` cube difference sphere
- `RT-CSG-003` cube intersection sphere
- `RT-CSG-004` cylinder difference cube slot
- `RT-CSG-005` cube difference cylinder through-hole
- `RT-CSG-006` two orthogonal cylinders union
- `RT-CSG-007` two orthogonal cylinders intersection
- `RT-CSG-008` tangent sphere/cube diagnostic

## Verification

Automated tests must cover:

- `surface_boolean_result` success for every supported primitive pair above
- `boolean_union`, `boolean_difference`, and `boolean_intersection` public
  helpers preserving the surfaced route
- `tests/test_no_hidden_mesh_fallback.py` or equivalent guards for every new
  primitive route
- section-loop or bounds checks that make wrong orientation and missing cuts
  fail clearly

Reference artifact verification must cover:

- dirty STL generation for each unblocked `RT-CSG-*` item
- fixture records in `tests/reference_review_fixtures/dirty-stl-fixtures.json`
- purpose, methodology, and render description fields

Project readiness fields:
- Implementation owner/module: src/impression/modeling/csg.py
- Chosen defaults / parameters: No mesh fallback; tessellation only after a surfaced result succeeds at artifact, preview, or export boundaries.
- Test strategy: `surface_boolean_result` success for every supported primitive pair above; `boolean_union`, `boolean_difference`, and `boolean_intersection` public helpers preserving the surfaced route; `tests/test_no_hidden_mesh_fallback.py` or equivalent guards for every new primitive route; section-loop or bounds checks that make wrong orientation and missing cuts fail clearly
- Data ownership: `SurfaceBody` and `SurfaceBooleanResult` records remain the authored source of truth for CSG execution.
- Data ownership: Reference fixture metadata owns review evidence, dirty artifact context, and diagnostic evidence records.
- Routes: public boolean helpers to primitive analytic surfaced CSG execution route
- Reuse/extraction decision: Reuse existing helpers named in the implementation boundary; add private helpers only when extraction keeps owner modules cohesive.
- UI field/control inventory: not applicable; these CSG manifest entries have no UI controls or visible fields

Open questions / nuance discovered:
- Exact geometric tolerances and fixture thresholds should follow the existing CSG and reference-test helpers unless a child implementation finds a case requiring an explicit local value.
- If implementation discovers a route needs mesh-derived source truth, keep the route refused and record the missing surfaced capability instead of adding fallback execution.

## Five-Pass Review History

| Pass | Update | Rescore result |
| --- | --- | --- |
| 1 | Added manifest-style scoring need and confirmed primitive analytic scope. | Candidate below split threshold. |
| 2 | Kept multi-operand, lofted, and advanced families out of scope. | 1 IWU retained. |
| 3 | Confirmed tangent diagnostics remain in scope because they are primitive-pair behavior. | Cohesive leaf. |
| 4 | Confirmed reference artifacts are verification, not solver implementation. | Score stable. |
| 5 | Confirmed no-hidden-mesh fallback is explicit verification. | Final draft score: 23.5. |

## Manifest Assessment

Score:

- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 3 x 0.5 = 1.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 1 x 2 = 2
- Total: 23.5

Readiness blockers:
- [ ] Exact route thresholds or fixture tolerances may need confirmation during implementation.
- [x] Implementation owner/module is named or inherited from the implementation boundary.
- [x] Reuse/extraction decision is explicit.
- [x] UI field/control inventory is not applicable.
- [x] Test strategy is named by verification requirements.
- [x] Data ownership is explicit enough for implementation planning.
- [x] Privacy/logging rule is not applicable beyond deterministic diagnostics.

Split decision:

- No split required. Cohesion reason: this leaf owns only primitive analytic
  pair execution and diagnostics; trim reconstruction and multi-operand
  composition are separate specs.

## Acceptance

This spec is complete when every listed primitive reference item can generate a
dirty STL from a succeeded surfaced boolean result, and the focused surface CSG
tests pass without any mesh compatibility route.
