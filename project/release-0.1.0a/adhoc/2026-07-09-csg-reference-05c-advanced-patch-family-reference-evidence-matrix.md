# CSG Reference Spec 05c: Advanced Patch Family Reference Evidence Program

Date: 2026-07-09
Status: Draft
Path: ad-hoc-path work

## Summary

Coordinate the reference evidence matrix for advanced patch-family CSG after
route support states are explicit. Child leaves own supported dirty STL
fixtures and unsupported diagnostic evidence separately.

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 2 IWU.
Basis: parent program document with two 1-IWU child leaves: supported dirty STL evidence and unsupported diagnostic evidence.

## Scope

Program scope:

- advanced patch-family evidence sequencing
- shared fixture context requirements
- shared no-hidden-mesh-fallback evidence requirements

Out of scope:

- implementing child fixture or diagnostic behavior in this parent document
- implementing solver routes
- changing family support states
- promoting dirty artifacts to gold

## Child Specifications

- [CSG Reference Spec 05c1: Supported Advanced Patch Dirty STL Evidence](2026-07-09-csg-reference-05c1-supported-advanced-patch-dirty-stl-evidence.md)
- [CSG Reference Spec 05c2: Unsupported Advanced Patch Diagnostic Evidence](2026-07-09-csg-reference-05c2-unsupported-advanced-patch-diagnostic-evidence.md)

## Implementation Boundary

Shared owner modules/files:

- `tests/test_reference_stl_expansion.py`
- `tests/reference_review_fixtures/stl_review_sources.py`
- `tests/reference_review_fixtures/dirty-stl-fixtures.json`
- `project/release-0.1.0a/reference-stl/dirty/`

Shared reuse:

- Reuse reference artifact lifecycle helpers.
- Reuse existing fixture context fields.
- Reuse negative diagnostic fixture matrix patterns.

## Required Behavior

- Child leaves must include purpose, methodology, and render description for
  every fixture record.
- Child leaves must fail clearly when evidence no longer matches route support
  state.

## Reference Items Unblocked

- evidence completion for `RT-PATCH-CSG-004` through `RT-PATCH-CSG-014`
  through child leaves

## Verification

- child-leaf fixture inventory coverage
- child-leaf support/refusal matrix coverage
- child-leaf no-hidden-mesh-fallback evidence checks

Project readiness fields:
- Implementation owner/module: src/impression/modeling/csg.py
- Chosen defaults / parameters: This parent has no execution defaults; child leaves own thresholds, route limits, and fixture defaults.
- Test strategy: child-leaf fixture inventory coverage; child-leaf support/refusal matrix coverage; child-leaf no-hidden-mesh-fallback evidence checks; Implementation owner/module: src/impression/modeling/csg.py; Chosen defaults / parameters: This parent has no execution defaults; child leaves own thresholds, route limits, and fixture defaults.; Test strategy: child-leaf fixture inventory coverage; child-leaf support/refusal matrix coverage; child-leaf no-hidden-mesh-fallback evidence checks; Data ownership: Parent document owns sequencing and coverage mapping only; child leaves own implementation data.; Routes: advanced family route through CSG support matrix, payload persistence, or refusal diagnostics; Reuse/extraction decision: Parent reuses child leaves and does not add code directly.; UI field/control inventory: not applicable; these CSG manifest entries have no UI controls or visible fields; Keep this parent out of progression-style implementation queues; only child leaves should be scheduled as executable work.; Rollup IWU counts must stay synchronized with nested child leaves if children are split again.
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
| 1 | Split evidence ownership from advanced route implementation. | Candidate became one fixture/evidence leaf. |
| 2 | Limited scope to reference artifacts and diagnostics. | Initial child score still crossed split threshold. |
| 3 | Split supported dirty STL evidence from unsupported diagnostic evidence. | Parent rollup: 2 IWU; child leaves score separately. |
| 4 | Confirmed dirty-to-gold promotion is out of scope. | Parent is not an implementation leaf. |
| 5 | Confirmed fixture context fields are inherited by both child leaves. | Parent ready as program document. |

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
  lives in Specs 05c1 and 05c2.

## Acceptance

This spec is complete when advanced patch-family CSG evidence has no implicit
gaps: every route has either a surfaced dirty STL fixture or deterministic
refusal evidence with fixture context.
