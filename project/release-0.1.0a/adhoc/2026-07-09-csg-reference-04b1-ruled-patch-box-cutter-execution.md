# CSG Reference Spec 04b1: Ruled Patch Box-Cutter Execution

Date: 2026-07-09
Status: Draft
Path: ad-hoc-path work

## Summary

Execute ruled patch CSG for eligible loft side walls cut by box-style planar
cutters through the shared surfaced CSG path.

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one ruled patch box-cutter execution boundary.

## Scope

In scope:

- ruled/planar support matrix row updates
- line intersections through ruled side walls for box cutters
- authored provenance preservation for successful results

Out of scope:

- cylinder and sphere cutter diagnostics
- loft body eligibility
- reference fixture generation

## Implementation Boundary

Owner module:

- `src/impression/modeling/csg.py`

## Verification

- ruled patch box-cutter tests
- no-hidden-mesh-fallback tests

Project readiness fields:
- Implementation owner/module: src/impression/modeling/csg.py
- Chosen defaults / parameters: No mesh fallback; tessellation only after a surfaced result succeeds at artifact, preview, or export boundaries.
- Test strategy: ruled patch box-cutter tests; no-hidden-mesh-fallback tests; Implementation owner/module: src/impression/modeling/csg.py; Chosen defaults / parameters: No mesh fallback; tessellation only after a surfaced result succeeds at artifact, preview, or export boundaries.; Chosen defaults / parameters: Unsupported routes return deterministic refusal records with family, operation, phase, and reason where applicable.; Test strategy: ruled patch box-cutter tests; no-hidden-mesh-fallback tests; Data ownership: `SurfaceBody` and `SurfaceBooleanResult` records remain the authored source of truth for CSG execution.; Data ownership: Reference fixture metadata owns review evidence, dirty artifact context, and diagnostic evidence records.; Routes: CSG intersection output to patch-local trim/fragment reconstruction route; loft or ruled-patch eligibility route into surfaced CSG planner; Reuse/extraction decision: Add to existing CSG, surface, persistence, or reference helper modules named by this spec; do not create public API unless the spec says so.; UI field/control inventory: not applicable; these CSG manifest entries have no UI controls or visible fields; Exact geometric tolerances and fixture thresholds should follow the existing CSG and reference-test helpers unless a child implementation finds a case requiring an explicit local value.; If implementation discovers a route needs mesh-derived source truth, keep the route refused and record the missing surfaced capability instead of adding fallback execution.
- Data ownership: `SurfaceBody` and `SurfaceBooleanResult` records remain the authored source of truth for CSG execution.
- Data ownership: Reference fixture metadata owns review evidence, dirty artifact context, and diagnostic evidence records.
- Routes: eligible ruled patch and box cutter to surfaced CSG execution route
- Reuse/extraction decision: Reuse existing helpers named in the implementation boundary; add private helpers only when extraction keeps owner modules cohesive.
- UI field/control inventory: not applicable; these CSG manifest entries have no UI controls or visible fields

Open questions / nuance discovered:
- Exact geometric tolerances and fixture thresholds should follow the existing CSG and reference-test helpers unless a child implementation finds a case requiring an explicit local value.
- If implementation discovers a route needs mesh-derived source truth, keep the route refused and record the missing surfaced capability instead of adding fallback execution.

## Five-Pass Review History

| Pass | Update | Rescore result |
| --- | --- | --- |
| 1 | Split from ruled patch parent. | Candidate became one box-cutter execution leaf. |
| 2 | Limited scope to planar box cutters. | 1 IWU retained. |
| 3 | Moved unsupported cutter diagnostics out. | Manifest score below split threshold. |
| 4 | Confirmed trim reconstruction is prerequisite. | Readiness blocker retained. |
| 5 | Confirmed fixture generation is out of scope. | Final draft score: 21. |

## Manifest Assessment

Score:

- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
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
- Readiness blockers: 2 x 2 = 4
- Total: 24

Readiness blockers:
- [ ] Exact route thresholds or fixture tolerances may need confirmation during implementation.
- [x] Implementation owner/module is named or inherited from the implementation boundary.
- [x] Reuse/extraction decision is explicit.
- [x] UI field/control inventory is not applicable.
- [x] Test strategy is named by verification requirements.
- [x] Data ownership is explicit enough for implementation planning.
- [x] Privacy/logging rule is not applicable beyond deterministic diagnostics.

Split decision:

- No split required. Cohesion reason: one ruled patch box-cutter execution
  boundary.

## Acceptance

This spec is complete when eligible ruled patches execute planar box-cutter CSG
through surfaced results without mesh fallback.
