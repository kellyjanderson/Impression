# CSG Reference Completion Ad Hoc Specification Index

Date: 2026-07-09
Status: Draft
Path: ad-hoc-path work

## Summary

The reference test expansion plan is blocked on CSG coverage that the current
surface-first boolean implementation does not yet execute. This index defines
the ad hoc specification set for completing the CSG functionality required to
generate the remaining dirty STL reference files without creating a mesh lane.

These specifications are intentionally tied to the reference expansion plan,
not to a claim that every possible CSG operation is complete.

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 18 IWU.
Basis: this index owns no implementation itself; it sequences eighteen final implementation leaves after the five-pass review split the original broad CSG candidates into smaller surfaced CSG work units.

## Source Documents

- [Reference Test Expansion Plan](../planning/reference-test-expansion-plan.md)
- [SurfaceBody CSG Architecture](../architecture/surfacebody-csg-architecture.md)
- [Surface CSG Executable Completion Architecture](../architecture/surface-csg-executable-completion-architecture.md)
- [Higher-Order Parametric CSG Routes Architecture](../architecture/higher-order-parametric-csg-routes-architecture.md)
- [Surface CSG Trim Fragment Reconstruction Architecture](../architecture/surface-csg-trim-fragment-reconstruction-architecture.md)
- [Sampled and Implicit Surface CSG Support Architecture](../architecture/sampled-implicit-surface-csg-support-architecture.md)

## Non-Negotiable Constraints

- Authored modeling remains `SurfaceBody`.
- Public `boolean_union`, `boolean_difference`, and `boolean_intersection`
  must return surfaced `SurfaceBooleanResult` outcomes for surfaced inputs.
- Tessellation is allowed only at explicit artifact, preview, or export
  boundaries after a surfaced result has succeeded.
- Unsupported cases must return deterministic refusal diagnostics. They must
  not fall through to mesh execution.
- Reference plan items can be checked only after the surfaced operation succeeds
  before STL export.

## Specification Tree

Program and parent documents:

- [CSG Reference Spec 02: General Trim Fragment Reconstruction Program](2026-07-09-csg-reference-02-general-trim-fragment-reconstruction.md)
- [CSG Reference Spec 02c: Shell Assembly And Validity Program](2026-07-09-csg-reference-02c-shell-seam-assembly-and-validity-gate.md)
- [CSG Reference Spec 04: Lofted And Ruled Body Boolean Program](2026-07-09-csg-reference-04-lofted-and-ruled-body-boolean-execution.md)
- [CSG Reference Spec 04b: Ruled Patch Boolean Program](2026-07-09-csg-reference-04b-ruled-patch-boolean-execution-route.md)
- [CSG Reference Spec 05: Advanced Patch Family Boolean Program](2026-07-09-csg-reference-05-advanced-patch-family-boolean-policy-and-evidence.md)
- [CSG Reference Spec 05b: Sampled And Implicit Patch Family Boolean Program](2026-07-09-csg-reference-05b-sampled-and-implicit-patch-family-boolean-routes.md)
- [CSG Reference Spec 05b2: Sampled Surface Boolean Program](2026-07-09-csg-reference-05b2-sampled-heightmap-and-displacement-boolean-routes.md)
- [CSG Reference Spec 05b2b: Displacement Boolean Program](2026-07-09-csg-reference-05b2b-displacement-boolean-routes.md)
- [CSG Reference Spec 05b2b2: Displacement Successful Result Program](2026-07-09-csg-reference-05b2b2-displacement-successful-payload-persistence-route.md)
- [CSG Reference Spec 05c: Advanced Patch Family Reference Evidence Program](2026-07-09-csg-reference-05c-advanced-patch-family-reference-evidence-matrix.md)

Final implementation leaves:

- [CSG Reference Spec 01: Primitive Analytic Surface Boolean Execution](2026-07-09-csg-reference-01-primitive-analytic-surface-boolean-execution.md)
- [CSG Reference Spec 02a: Trim Arrangement Graph Construction](2026-07-09-csg-reference-02a-trim-arrangement-graph-construction.md)
- [CSG Reference Spec 02b: Fragment Classification And Operation Selection](2026-07-09-csg-reference-02b-fragment-classification-and-operation-selection.md)
- [CSG Reference Spec 02c1: Shell And Seam Assembly](2026-07-09-csg-reference-02c1-shell-and-seam-assembly.md)
- [CSG Reference Spec 02c2: Cap Patch And Validity Gate](2026-07-09-csg-reference-02c2-cap-patch-and-validity-gate.md)
- [CSG Reference Spec 03: Multi-Operand Boolean Composition](2026-07-09-csg-reference-03-multi-operand-boolean-composition.md)
- [CSG Reference Spec 04a: Lofted Body Eligibility And Refusal Diagnostics](2026-07-09-csg-reference-04a-lofted-body-eligibility-and-refusal-diagnostics.md)
- [CSG Reference Spec 04b1: Ruled Patch Box-Cutter Execution](2026-07-09-csg-reference-04b1-ruled-patch-box-cutter-execution.md)
- [CSG Reference Spec 04b2: Ruled Patch Unsupported Cutter Diagnostics](2026-07-09-csg-reference-04b2-ruled-patch-unsupported-cutter-diagnostics.md)
- [CSG Reference Spec 04c: Loft CSG Reference Evidence](2026-07-09-csg-reference-04c-loft-csg-reference-evidence.md)
- [CSG Reference Spec 05a: Higher-Order Patch Family Boolean Routes](2026-07-09-csg-reference-05a-higher-order-patch-family-boolean-routes.md)
- [CSG Reference Spec 05b1: Implicit Field Boolean Routes](2026-07-09-csg-reference-05b1-implicit-field-boolean-routes.md)
- [CSG Reference Spec 05b2a: Heightmap Boolean Routes](2026-07-09-csg-reference-05b2a-heightmap-boolean-routes.md)
- [CSG Reference Spec 05b2b1: Displacement Source Identity And Refusal Route](2026-07-09-csg-reference-05b2b1-displacement-source-identity-and-refusal-route.md)
- [CSG Reference Spec 05b2b2a: Displacement Result Construction Route](2026-07-09-csg-reference-05b2b2a-displacement-result-construction-route.md)
- [CSG Reference Spec 05b2b2b: Displacement Payload Persistence Evidence](2026-07-09-csg-reference-05b2b2b-displacement-payload-persistence-evidence.md)
- [CSG Reference Spec 05c1: Supported Advanced Patch Dirty STL Evidence](2026-07-09-csg-reference-05c1-supported-advanced-patch-dirty-stl-evidence.md)
- [CSG Reference Spec 05c2: Unsupported Advanced Patch Diagnostic Evidence](2026-07-09-csg-reference-05c2-unsupported-advanced-patch-diagnostic-evidence.md)

## Reference Plan Coverage Map

| Reference plan items | Owning ad hoc specs |
| --- | --- |
| `RT-CSG-001` through `RT-CSG-008` | Spec 01 and reconstruction child leaves under Spec 02 |
| `RT-CSG-009` | reconstruction child leaves under Spec 02 |
| `RT-CSG-010` through `RT-CSG-012` | Spec 03, plus primitive support from Spec 01 and reconstruction child leaves under Spec 02 |
| `RT-PATCH-CSG-001` through `RT-PATCH-CSG-003` | Spec 01, reconstruction child leaves under Spec 02, and ruled child leaves under Spec 04 |
| `RT-PATCH-CSG-004` through `RT-PATCH-CSG-014` | advanced-family child leaves under Spec 05 |
| `RT-LOFT-CSG-001` through `RT-LOFT-CSG-014` | reconstruction child leaves under Spec 02 and loft/ruled child leaves under Spec 04, with Spec 05 child leaves for higher-order loft surfaces when needed |

## Implementation Order

1. Implement primitive analytic execution for planar and revolution operand
   pairs.
2. Generalize trim-fragment reconstruction through Specs 02a, 02b, 02c1, and
   02c2 so non-box cut curves can produce closed surfaced shells.
3. Add deterministic multi-operand composition once pairwise results are
   reliable.
4. Enable lofted body eligibility and ruled execution through Specs 04a, 04b1,
   04b2, and 04c.
5. Add advanced-family policy, adapters, and evidence through Specs 05a through
   05c2 so every planned family either executes or refuses with durable
   diagnostics.

## Completion

This ad hoc spec set is complete when every leaf is implemented or explicitly
superseded by a fuller architecture-derived specification, and the unchecked
CSG-related items in the reference expansion plan can either generate dirty STL
fixtures or remain intentionally unchecked with exact refusal evidence.
