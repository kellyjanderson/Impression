# CSG Reference Spec 05b: Sampled And Implicit Patch Family Boolean Program

Date: 2026-07-09
Status: Draft
Path: ad-hoc-path work

## Summary

Coordinate execute-or-refuse CSG routes for implicit, heightmap, and
displacement patch families. These routes must preserve sampled/source-domain
truth or explicitly refuse; they must not use extracted mesh as authored CSG
state.

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 2 IWU.
Basis: parent program document with two 1-IWU child leaves: implicit field routes and sampled heightmap/displacement routes.

## Scope

Program scope:

- shared no-mesh constraints for implicit, heightmap, and displacement CSG
- child route sequencing
- shared `.impress` persistence expectation for successful or promoted payloads

Out of scope:

- implementation of child route behavior in this parent document
- B-spline, NURBS, sweep, or subdivision route policy
- reference fixture matrix ownership
- preview UI behavior

## Child Specifications

- [CSG Reference Spec 05b1: Implicit Field Boolean Routes](2026-07-09-csg-reference-05b1-implicit-field-boolean-routes.md)
- [CSG Reference Spec 05b2: Sampled Heightmap And Displacement Boolean Routes](2026-07-09-csg-reference-05b2-sampled-heightmap-and-displacement-boolean-routes.md)

## Implementation Boundary

Shared owner modules:

- `src/impression/modeling/csg.py`
- `src/impression/modeling/surface.py`
- `src/impression/io/impress.py` for persistence evidence only

Shared reuse:

- Reuse sampled/implicit CSG payload records.
- Reuse displacement source identity diagnostics.
- Reuse `.impress` sampled/implicit round-trip helpers.

## Required Behavior

- Child leaves must preserve payload identity through `.impress` when a route
  succeeds or promotes sampled truth.
- No child route may extract mesh truth as the authored CSG result.

## Reference Items Unblocked

- `RT-PATCH-CSG-008` through Spec 05b1
- `RT-PATCH-CSG-009` and `RT-PATCH-CSG-010` through Spec 05b2
- `RT-PATCH-CSG-012` through both child leaves
- sampled/implicit portions of `RT-PATCH-CSG-013` and `RT-PATCH-CSG-014`

## Verification

- child-leaf execute-or-refuse tests
- child-leaf `.impress` round-trip tests for successful/promoted routes
- child-leaf no-hidden-mesh-fallback tests

Project readiness fields:
- Implementation owner/module: src/impression/modeling/csg.py
- Chosen defaults / parameters: This parent has no execution defaults; child leaves own thresholds, route limits, and fixture defaults.
- Test strategy: child-leaf execute-or-refuse tests; child-leaf `.impress` round-trip tests for successful/promoted routes; child-leaf no-hidden-mesh-fallback tests
- Data ownership: Parent document owns sequencing and coverage mapping only; child leaves own implementation data.
- Routes: advanced family route through CSG support matrix, payload persistence, or refusal diagnostics
- Reuse/extraction decision: Parent reuses child leaves and does not add code directly.
- UI field/control inventory: not applicable; these CSG manifest entries have no UI controls or visible fields

Open questions / nuance discovered:
- Keep this parent out of progression-style implementation queues; only child leaves should be scheduled as executable work.
- Rollup IWU counts must stay synchronized with nested child leaves if children are split again.

## Five-Pass Review History

| Pass | Update | Rescore result |
| --- | --- | --- |
| 1 | Split from the bundled advanced-family parent. | Candidate became one sampled/implicit route-policy leaf. |
| 2 | Limited scope to implicit, heightmap, and displacement. | Initial child score still crossed split threshold. |
| 3 | Split into implicit field routes and sampled heightmap/displacement routes. | Parent rollup: 2 IWU; child leaves score separately. |
| 4 | Clarified mesh extraction is forbidden as authored state. | Parent is not an implementation leaf. |
| 5 | Confirmed persistence expectation is inherited by both child leaves. | Parent ready as program document. |

## Manifest Assessment

Score:

- Functions/methods: 0 x 2 = 0
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 0 x 1 = 0
- Existing reusable code reused as-is: 3 x 0.5 = 1.5
- Adding code to an existing library/module: 0 x 1 = 0
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 0 x 2 = 0
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 1 x 2 = 2
- Total: 7.5

Readiness blockers:
- [ ] Parent has no direct implementation owner beyond child-leaf sequencing.
- [x] Implementation owner/module is named or inherited from the implementation boundary.
- [x] Reuse/extraction decision is explicit.
- [x] UI field/control inventory is not applicable.
- [x] Test strategy is named by verification requirements.
- [x] Data ownership is explicit enough for implementation planning.
- [x] Privacy/logging rule is not applicable beyond deterministic diagnostics.

Split decision:

- Split completed. This parent is not an implementation leaf; implementation
  lives in Specs 05b1 and 05b2.

## Acceptance

This spec is complete when Specs 05b1 and 05b2 either execute or refuse with
persisted, surface-native evidence and no hidden mesh fallback.
