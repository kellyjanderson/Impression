# CSG Reference Spec 04: Lofted And Ruled Body Boolean Program

Date: 2026-07-09
Status: Draft
Path: ad-hoc-path work

## Summary

Coordinate CSG work for lofted bodies whose shells are valid surfaced bodies
with ruled or related loft-generated side patches.

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 4 IWU.
Basis: parent program document with four final 1-IWU descendant leaves: eligibility/refusal, ruled box-cutter execution, ruled unsupported-cutter diagnostics, and loft CSG reference evidence.

## Scope

Program scope:

- child-spec sequencing for loft eligibility, ruled execution, and reference
  evidence
- shared no-mesh and loft-planner-boundary constraints
- coverage mapping back to `RT-LOFT-CSG-*`

Out of scope:

- implementation of child route behavior in this parent document
- remeshing lofts for CSG
- changing loft planner semantics
- CSG for every possible branching topology
- interactive UI changes

## Child Specifications

- [CSG Reference Spec 04a: Lofted Body Eligibility And Refusal Diagnostics](2026-07-09-csg-reference-04a-lofted-body-eligibility-and-refusal-diagnostics.md)
- [CSG Reference Spec 04b: Ruled Patch Boolean Execution Route](2026-07-09-csg-reference-04b-ruled-patch-boolean-execution-route.md)
- [CSG Reference Spec 04c: Loft CSG Reference Evidence](2026-07-09-csg-reference-04c-loft-csg-reference-evidence.md)

## Implementation Boundary

Shared owner modules:

- `src/impression/modeling/csg.py`
- `src/impression/modeling/loft.py` only for metadata or eligibility hooks that
  expose loft-generated patch provenance

Shared reuse:

- Reuse existing loft output as `SurfaceBody`.
- Reuse the family support matrix and refusal diagnostics.
- Reuse trim-fragment reconstruction from Spec 02.

## Required Behavior

- Child leaves must not mutate loft planner state.
- Child leaves must preserve source/provenance metadata.
- Child leaves must refuse unsupported topology explicitly.

## Reference Items Unblocked

- `RT-LOFT-CSG-001` through `RT-LOFT-CSG-014` through child leaves.
- `RT-PATCH-CSG-002` and ruled portions of `RT-PATCH-CSG-011` through Spec 04b.

## Verification

Automated tests must cover:

- child-leaf eligibility, ruled execution, reference evidence, and
  no-hidden-mesh fallback tests

Reference artifact verification must cover:

- dirty STL generation for every successful `RT-LOFT-CSG-*` case
- at least one explicit refusal fixture for underconstrained topology
- fixture context that explains expected visible cuts and refusal reasons

Project readiness fields:
- Implementation owner/module: src/impression/modeling/csg.py
- Chosen defaults / parameters: This parent has no execution defaults; child leaves own thresholds, route limits, and fixture defaults.
- Test strategy: child-leaf eligibility, ruled execution, reference evidence, and no-hidden-mesh fallback tests
- Data ownership: Parent document owns sequencing and coverage mapping only; child leaves own implementation data.
- Routes: child loft and ruled-patch leaves sequence eligibility, execution/refusal, and evidence routes
- Reuse/extraction decision: Parent reuses child leaves and does not add code directly.
- UI field/control inventory: not applicable; these CSG manifest entries have no UI controls or visible fields

Open questions / nuance discovered:
- Keep this parent out of progression-style implementation queues; only child leaves should be scheduled as executable work.
- Rollup IWU counts must stay synchronized with nested child leaves if children are split again.

## Five-Pass Review History

| Pass | Update | Rescore result |
| --- | --- | --- |
| 1 | Added manifest-style scoring need and confirmed loft/ruled scope. | Honest score crossed split threshold. |
| 2 | Kept loft planner changes out of scope. | Still bundled eligibility, execution, and evidence. |
| 3 | Split eligibility/refusal, ruled execution, and loft evidence. | Parent rollup: 3 IWU; child leaves score separately. |
| 4 | Confirmed successful cutter support depends on trim reconstruction. | Parent is not an implementation leaf. |
| 5 | Confirmed branching topology can remain explicit refusal evidence. | Parent ready as program document. |

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
  lives in final descendant leaves Specs 04a, 04b1, 04b2, and 04c, with
  Spec 04b retained only as a nested program document.

## Acceptance

This spec is complete when the planned loft CSG reference fixtures either
generate succeeded surfaced dirty STLs or carry exact refusal evidence for cases
that are intentionally outside the supported topology.
