# CSG Reference Spec 04b: Ruled Patch Boolean Program

Date: 2026-07-09
Status: Draft
Path: ad-hoc-path work

## Summary

Coordinate ruled patch CSG execution and refusal diagnostics for eligible loft
side walls through the shared surfaced CSG path.

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 2 IWU.
Basis: parent program document with two 1-IWU child leaves: box-cutter execution and unsupported cutter diagnostics.

## Scope

Program scope:

- shared ruled patch support matrix expectations
- child sequencing for execution and unsupported cutter diagnostics
- shared provenance and no-mesh constraints

Out of scope:

- implementation of child behavior in this parent document
- loft body eligibility
- reference fixture generation

## Child Specifications

- [CSG Reference Spec 04b1: Ruled Patch Box-Cutter Execution](2026-07-09-csg-reference-04b1-ruled-patch-box-cutter-execution.md)
- [CSG Reference Spec 04b2: Ruled Patch Unsupported Cutter Diagnostics](2026-07-09-csg-reference-04b2-ruled-patch-unsupported-cutter-diagnostics.md)

## Implementation Boundary

Shared owner modules:

- `src/impression/modeling/csg.py`

## Verification

- child-leaf ruled execution, unsupported cutter diagnostic, and
  no-hidden-mesh-fallback tests

Project readiness fields:
- Implementation owner/module: src/impression/modeling/csg.py
- Chosen defaults / parameters: This parent has no execution defaults; child leaves own thresholds, route limits, and fixture defaults.
- Test strategy: child-leaf ruled execution, unsupported cutter diagnostic, and no-hidden-mesh-fallback tests; Implementation owner/module: src/impression/modeling/csg.py; Chosen defaults / parameters: This parent has no execution defaults; child leaves own thresholds, route limits, and fixture defaults.; Test strategy: child-leaf ruled execution, unsupported cutter diagnostic, and no-hidden-mesh-fallback tests; Data ownership: Parent document owns sequencing and coverage mapping only; child leaves own implementation data.; Routes: CSG intersection output to patch-local trim/fragment reconstruction route; loft or ruled-patch eligibility route into surfaced CSG planner; Reuse/extraction decision: Parent reuses child leaves and does not add code directly.; UI field/control inventory: not applicable; these CSG manifest entries have no UI controls or visible fields; Keep this parent out of progression-style implementation queues; only child leaves should be scheduled as executable work.; Rollup IWU counts must stay synchronized with nested child leaves if children are split again.
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
| 1 | Split from lofted CSG parent. | Candidate became one ruled execution leaf. |
| 2 | Kept eligibility out of scope. | Score reached split threshold. |
| 3 | Split box-cutter execution from unsupported cutter diagnostics. | Parent rollup: 2 IWU; child leaves score separately. |
| 4 | Confirmed trim reconstruction is a prerequisite. | Parent is not an implementation leaf. |
| 5 | Confirmed fixture generation is out of scope. | Parent ready as program document. |

## Manifest Assessment

Score:

- Functions/methods: 0 x 2 = 0
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 0 x 1 = 0
- Existing reusable code reused as-is: 2 x 0.5 = 1
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
- Total: 7

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
  lives in Specs 04b1 and 04b2.

## Acceptance

This spec is complete when eligible ruled patch CSG routes execute or refuse
deterministically through the surfaced path without mesh fallback.
