# CSG Reference Spec 05: Advanced Patch Family Boolean Program

Date: 2026-07-09
Status: Draft
Path: ad-hoc-path work

## Summary

Coordinate the advanced patch family CSG work required by the reference matrix:
higher-order parametric families, sampled/implicit families, and reference
evidence. Each family pair must either execute as surface-native CSG or refuse
with deterministic evidence. No family may fall back to mesh truth.

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 3 IWU.
Basis: this is now a parent program document with three 1-IWU child leaves: higher-order family routes, sampled/implicit family routes, and reference evidence.

## Scope

Program scope:

- child-spec sequencing for advanced family CSG support
- shared constraints across higher-order, sampled, implicit, and reference
  evidence routes
- coverage mapping back to the reference expansion plan

Out of scope:

- implementing child behavior in this parent document
- replacing explicit unsupported rows with vague TODOs
- using tessellated meshes as authored CSG truth

## Child Specifications

- [CSG Reference Spec 05a: Higher-Order Patch Family Boolean Routes](2026-07-09-csg-reference-05a-higher-order-patch-family-boolean-routes.md)
- [CSG Reference Spec 05b: Sampled And Implicit Patch Family Boolean Routes](2026-07-09-csg-reference-05b-sampled-and-implicit-patch-family-boolean-routes.md)
- [CSG Reference Spec 05c: Advanced Patch Family Reference Evidence Matrix](2026-07-09-csg-reference-05c-advanced-patch-family-reference-evidence-matrix.md)

## Implementation Boundary

Shared owner modules:

- `src/impression/modeling/csg.py`
- `src/impression/modeling/surface.py`
- `src/impression/modeling/surface_intersections.py` where intersection
  dispatch belongs
- `.impress` codec modules only when a successful or promoted result needs
  persistence evidence

Shared reuse:

- Reuse existing sampled/implicit CSG records and round-trip diagnostics.
- Reuse advanced patch family availability and promotion evidence records.
- Reuse negative diagnostic fixture matrix patterns.

## Required Behavior

- Every child leaf must preserve `SurfaceBody` authored truth.
- Every child leaf must preserve `SurfaceBooleanResult` surfaced outcomes.
- Every child leaf must prove no hidden mesh fallback.
- Unsupported routes include family, operation, phase, and required future
  capability.

## Reference Items Unblocked

- `RT-PATCH-CSG-004` through `RT-PATCH-CSG-014` through the child leaves.

## Verification

Automated tests must cover:

- child-leaf support matrix rows and execute-or-refuse behavior
- no hidden mesh fallback for successful, promoted, and refused routes
- `.impress` round-trip for successful or promoted family payloads where
  child leaves require persistence
- negative diagnostic fixture coverage for unsupported routes

Reference artifact verification must cover:

- dirty STL generation only for family routes that succeed as surfaced CSG
- refusal fixtures or diagnostics for unsupported routes
- review fixture descriptions that state exact, declared-tolerance, promoted,
  or unsupported expectations

## Acceptance

This spec is complete when the advanced patch-family CSG reference matrix has
no implicit gaps: every row either generates a surfaced dirty STL fixture or
records a deterministic refusal that proves no mesh fallback occurred.

Project readiness fields:
- Implementation owner/module: src/impression/modeling/csg.py
- Chosen defaults / parameters: This parent has no execution defaults; child leaves own thresholds, route limits, and fixture defaults.
- Test strategy: child-leaf support matrix rows and execute-or-refuse behavior; no hidden mesh fallback for successful, promoted, and refused routes; `.impress` round-trip for successful or promoted family payloads where child leaves require persistence; negative diagnostic fixture coverage for unsupported routes
- Data ownership: Parent document owns sequencing and coverage mapping only; child leaves own implementation data.
- Routes: child advanced-family leaves sequence support policy, sampled/implicit routes, and evidence routes
- Reuse/extraction decision: Parent reuses child leaves and does not add code directly.
- UI field/control inventory: not applicable; these CSG manifest entries have no UI controls or visible fields

Open questions / nuance discovered:
- Keep this parent out of progression-style implementation queues; only child leaves should be scheduled as executable work.
- Rollup IWU counts must stay synchronized with nested child leaves if children are split again.

## Five-Pass Review History

| Pass | Update | Rescore result |
| --- | --- | --- |
| 1 | Identified advanced family scope as bundled across solver policy, sampled/implicit routes, persistence, and evidence. | Parent would exceed split threshold. |
| 2 | Split into three child leaves and converted this file into a parent program document. | Parent rollup: 3 IWU; child leaves score separately. |
| 3 | Confirmed parent owns sequencing only and carries no implementation body. | Parent is not a final implementation leaf. |
| 4 | Confirmed child acceptance remains tied to `RT-PATCH-CSG-004` through `RT-PATCH-CSG-014`. | No further parent split. |
| 5 | Confirmed surfaced/no-mesh constraints are explicit and inherited by every child. | Parent ready as index/program document. |

## Manifest Assessment

Score:

- Functions/methods: 0 x 2 = 0
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 4 x 1 = 4
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
- Total: 8.5

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
  lives in Specs 05a, 05b, and 05c.
