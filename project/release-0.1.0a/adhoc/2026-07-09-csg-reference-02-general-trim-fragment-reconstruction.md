# CSG Reference Spec 02: General Trim Fragment Reconstruction Program

Date: 2026-07-09
Status: Draft
Path: ad-hoc-path work

## Summary

Coordinate the work that generalizes the current bounded box/box reconstruction
path into reusable trim fragment reconstruction for non-box cut curves.

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 4 IWU.
Basis: parent program document with four final 1-IWU descendant leaves: trim arrangement, fragment classification, shell/seam assembly, and cap/validity behavior.

## Scope

Program scope:

- child-spec sequencing for trim arrangement, classification, and shell assembly
- shared no-mesh and validity constraints
- coverage mapping back to reference items

Out of scope:

- implementation of child reconstruction behavior in this parent document
- generating analytic intersection records for new family pairs
- multi-operand sequencing
- advanced sampled/implicit family adapters

## Child Specifications

- [CSG Reference Spec 02a: Trim Arrangement Graph Construction](2026-07-09-csg-reference-02a-trim-arrangement-graph-construction.md)
- [CSG Reference Spec 02b: Fragment Classification And Operation Selection](2026-07-09-csg-reference-02b-fragment-classification-and-operation-selection.md)
- [CSG Reference Spec 02c: Shell Seam Assembly And Validity Gate](2026-07-09-csg-reference-02c-shell-seam-assembly-and-validity-gate.md)

## Implementation Boundary

Shared owner modules:

- `src/impression/modeling/csg.py`
- optionally a private CSG trim/arrangement helper module

Shared reuse:

- Reuse the existing box/box reconstruction metadata and validity gate.
- Reuse existing `SurfaceCSGFragmentGraphRecord` and cap-patch records.
- Reuse existing seam, adjacency, and trim-loop payload objects from the surface
  model.

## Required Behavior

- Child leaves must preserve disjoint and exact no-cut behavior.
- Child leaves must produce surfaced records consumed by the next stage.
- Child leaves must return invalid/unsupported diagnostics rather than fallback.

## Reference Items Unblocked

- `RT-CSG-009` and all non-box trim-loop primitive/patch/loft items through
  child leaves.

## Verification

Automated tests must cover:

- child-leaf trim graph, classification, shell assembly, invalid-result, and
  no-hidden-mesh fallback tests

Reference artifact verification must cover:

- at least one dirty STL where a curved cutter creates a visible concave cut
- at least one dirty STL where a cap patch is introduced by difference
- review fixture context describing which fragments should be visible

Project readiness fields:
- Implementation owner/module: src/impression/modeling/csg.py
- Chosen defaults / parameters: This parent has no execution defaults; child leaves own thresholds, route limits, and fixture defaults.
- Test strategy: child-leaf trim graph, classification, shell assembly, invalid-result, and no-hidden-mesh fallback tests
- Data ownership: Parent document owns sequencing and coverage mapping only; child leaves own implementation data.
- Routes: child reconstruction leaves sequence trim graph, classification, shell/seam, cap, and validity routes
- Reuse/extraction decision: Parent reuses child leaves and does not add code directly.
- UI field/control inventory: not applicable; these CSG manifest entries have no UI controls or visible fields

Open questions / nuance discovered:
- Keep this parent out of progression-style implementation queues; only child leaves should be scheduled as executable work.
- Rollup IWU counts must stay synchronized with nested child leaves if children are split again.

## Five-Pass Review History

| Pass | Update | Rescore result |
| --- | --- | --- |
| 1 | Added manifest-style scoring need and confirmed reconstruction scope. | Honest score crossed split threshold. |
| 2 | Kept analytic intersection generation out of scope. | Still bundled across three reconstruction stages. |
| 3 | Split trim arrangement, fragment classification, and shell assembly. | Parent rollup: 3 IWU; child leaves score separately. |
| 4 | Confirmed multi-operand sequencing remains out of scope. | Parent is not an implementation leaf. |
| 5 | Confirmed invalid result posture is inherited by child leaves. | Parent ready as program document. |

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
  lives in final descendant leaves Specs 02a, 02b, 02c1, and 02c2, with
  Spec 02c retained only as a nested program document.

## Acceptance

This spec is complete when supported non-box pairwise CSG can reconstruct valid
trimmed surfaced results from analytic cut curves, and invalid/tangent cases
fail with deterministic surfaced diagnostics.
