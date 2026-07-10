# CSG Reference Spec 05a: Higher-Order Patch Family Boolean Routes

Date: 2026-07-09
Status: Draft
Path: ad-hoc-path work

## Summary

Define execute-or-refuse CSG routes for B-spline, NURBS, sweep, and subdivision
patch families so reference fixtures can prove whether each higher-order route
is supported without using mesh truth.

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one route-policy boundary for higher-order parametric patch families.

## Scope

In scope:

- family support matrix rows for B-spline, NURBS, sweep, and subdivision CSG
- exact, declared-tolerance, adapter, or unsupported support states
- residual/tolerance diagnostics for declared-tolerance routes
- explicit refusal when a route is unsupported, non-convergent, or
  underconstrained

Out of scope:

- sampled and implicit families
- reference fixture generation
- broad exact solvers beyond the support state declared by the route matrix

## Implementation Boundary

Owner modules:

- `src/impression/modeling/csg.py`
- `src/impression/modeling/surface_intersections.py`

Reuse:

- Reuse higher-order CSG route records already defined by architecture specs.
- Reuse negative diagnostic fixture matrix patterns.
- Reuse `SurfaceBooleanResult` and the family support matrix.

## Required Behavior

- Each higher-order family route has a stable support state.
- Supported or declared-tolerance routes return surfaced CSG results.
- Unsupported routes name the family, operation, phase, and future capability.
- No higher-order route may tessellate operands for authored CSG truth.

## Reference Items Unblocked

- `RT-PATCH-CSG-004` B-spline patch CSG
- `RT-PATCH-CSG-005` NURBS patch CSG
- `RT-PATCH-CSG-006` sweep patch CSG
- `RT-PATCH-CSG-007` subdivision patch CSG
- higher-order portions of `RT-PATCH-CSG-011`
- `RT-PATCH-CSG-013` and `RT-PATCH-CSG-014` for unsupported/no-mesh evidence

## Verification

- support matrix tests for every listed family
- execute-or-refuse tests for representative operation classes
- residual/tolerance diagnostics for declared-tolerance routes
- no-hidden-mesh-fallback tests

Project readiness fields:
- Implementation owner/module: src/impression/modeling/csg.py
- Chosen defaults / parameters: No mesh fallback; tessellation only after a surfaced result succeeds at artifact, preview, or export boundaries.
- Test strategy: support matrix tests for every listed family; execute-or-refuse tests for representative operation classes; residual/tolerance diagnostics for declared-tolerance routes; no-hidden-mesh-fallback tests; Implementation owner/module: src/impression/modeling/csg.py; Chosen defaults / parameters: No mesh fallback; tessellation only after a surfaced result succeeds at artifact, preview, or export boundaries.; Chosen defaults / parameters: Unsupported routes return deterministic refusal records with family, operation, phase, and reason where applicable.; Test strategy: support matrix tests for every listed family; execute-or-refuse tests for representative operation classes; residual/tolerance diagnostics for declared-tolerance routes; no-hidden-mesh-fallback tests; Data ownership: `SurfaceBody` and `SurfaceBooleanResult` records remain the authored source of truth for CSG execution.; Data ownership: Reference fixture metadata owns review evidence, dirty artifact context, and diagnostic evidence records.; Routes: advanced family route through CSG support matrix, payload persistence, or refusal diagnostics; Reuse/extraction decision: Reuse existing CSG/surface helpers named in the implementation boundary; add private helpers only when extraction keeps owner modules cohesive.; UI field/control inventory: not applicable; these CSG manifest entries have no UI controls or visible fields; Exact geometric tolerances and fixture thresholds should follow the existing CSG and reference-test helpers unless a child implementation finds a case requiring an explicit local value.; If implementation discovers a route needs mesh-derived source truth, keep the route refused and record the missing surfaced capability instead of adding fallback execution.
- Data ownership: `SurfaceBody` and `SurfaceBooleanResult` records remain the authored source of truth for CSG execution.
- Data ownership: Reference fixture metadata owns review evidence, dirty artifact context, and diagnostic evidence records.
- Routes: higher-order patch family pair to route-policy and execute-or-refuse CSG route
- Reuse/extraction decision: Reuse existing helpers named in the implementation boundary; add private helpers only when extraction keeps owner modules cohesive.
- UI field/control inventory: not applicable; these CSG manifest entries have no UI controls or visible fields

Open questions / nuance discovered:
- Exact geometric tolerances and fixture thresholds should follow the existing CSG and reference-test helpers unless a child implementation finds a case requiring an explicit local value.
- If implementation discovers a route needs mesh-derived source truth, keep the route refused and record the missing surfaced capability instead of adding fallback execution.

## Five-Pass Review History

| Pass | Update | Rescore result |
| --- | --- | --- |
| 1 | Split from the bundled advanced-family parent. | Candidate became one higher-order route-policy leaf. |
| 2 | Limited scope to B-spline, NURBS, sweep, and subdivision. | 1 IWU retained. |
| 3 | Added explicit residual/tolerance diagnostics. | Manifest score remains below split threshold. |
| 4 | Clarified that fixture generation is out of scope. | Cohesive leaf. |
| 5 | Confirmed no hidden mesh fallback is a verification requirement. | Final draft score: 20.5. |

## Manifest Assessment

Score:

- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 2 x 0.5 = 1
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
- Total: 23

Readiness blockers:
- [ ] Exact route thresholds or fixture tolerances may need confirmation during implementation.
- [x] Implementation owner/module is named or inherited from the implementation boundary.
- [x] Reuse/extraction decision is explicit.
- [x] UI field/control inventory is not applicable.
- [x] Test strategy is named by verification requirements.
- [x] Data ownership is explicit enough for implementation planning.
- [x] Privacy/logging rule is not applicable beyond deterministic diagnostics.

Split decision:

- No split required after review. Cohesion reason: this leaf owns only
  higher-order route support and refusal policy; sampled families and fixture
  evidence are separate leaves.

## Acceptance

This spec is complete when every higher-order family route needed by the
reference plan either executes as surfaced CSG or refuses with deterministic
support-state evidence and no mesh fallback.
