# CSG Reference Spec 05b2b2: Displacement Successful Result Program

Date: 2026-07-09
Status: Draft
Path: ad-hoc-path work

## Summary

Coordinate the successful or promoted displacement CSG result path after source
identity is valid, including surface-native result construction and `.impress`
payload preservation.

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 2 IWU.
Basis: parent program document with two 1-IWU child leaves: result construction and persistence evidence.

## Scope

Program scope:

- shared successful displacement route sequencing
- shared source/provenance and no-mesh constraints
- shared payload identity expectation

Out of scope:

- implementation of child route behavior in this parent document
- source identity refusal behavior
- dirty STL fixture generation

## Child Specifications

- [CSG Reference Spec 05b2b2a: Displacement Result Construction Route](2026-07-09-csg-reference-05b2b2a-displacement-result-construction-route.md)
- [CSG Reference Spec 05b2b2b: Displacement Payload Persistence Evidence](2026-07-09-csg-reference-05b2b2b-displacement-payload-persistence-evidence.md)

## Implementation Boundary

Shared owner modules:

- `src/impression/modeling/csg.py`
- `src/impression/modeling/surface.py`
- `src/impression/io/impress.py` for persistence evidence only

## Reference Items Unblocked

- successful portions of `RT-PATCH-CSG-010`, `RT-PATCH-CSG-012`, and
  `RT-PATCH-CSG-014` through child leaves

## Verification

- child-leaf successful/promoted displacement result tests
- child-leaf displacement payload round-trip tests
- child-leaf result provenance tests
- child-leaf no-hidden-mesh-fallback tests

Project readiness fields:
- Implementation owner/module: src/impression/modeling/csg.py
- Chosen defaults / parameters: This parent has no execution defaults; child leaves own thresholds, route limits, and fixture defaults.
- Test strategy: child-leaf successful/promoted displacement result tests; child-leaf displacement payload round-trip tests; child-leaf result provenance tests; child-leaf no-hidden-mesh-fallback tests; Implementation owner/module: src/impression/modeling/csg.py; Chosen defaults / parameters: This parent has no execution defaults; child leaves own thresholds, route limits, and fixture defaults.; Test strategy: child-leaf successful/promoted displacement result tests; child-leaf displacement payload round-trip tests; child-leaf result provenance tests; child-leaf no-hidden-mesh-fallback tests; Data ownership: Parent document owns sequencing and coverage mapping only; child leaves own implementation data.; Routes: advanced family route through CSG support matrix, payload persistence, or refusal diagnostics; Reuse/extraction decision: Parent reuses child leaves and does not add code directly.; UI field/control inventory: not applicable; these CSG manifest entries have no UI controls or visible fields; Keep this parent out of progression-style implementation queues; only child leaves should be scheduled as executable work.; Rollup IWU counts must stay synchronized with nested child leaves if children are split again.
- Data ownership: Parent document owns sequencing and coverage mapping only; child leaves own implementation data.
- Routes: child sampled/implicit leaves sequence implicit, heightmap, displacement, persistence, and refusal routes
- Reuse/extraction decision: Parent reuses child leaves and does not add code directly.
- UI field/control inventory: not applicable; these CSG manifest entries have no UI controls or visible fields

Open questions / nuance discovered:
- Keep this parent out of progression-style implementation queues; only child leaves should be scheduled as executable work.
- Rollup IWU counts must stay synchronized with nested child leaves if children are split again.

## Five-Pass Review History

| Pass | Update | Rescore result |
| --- | --- | --- |
| 1 | Split from displacement parent. | Candidate became one successful-persistence leaf. |
| 2 | Limited scope to valid-source successful results. | Initial score reached split threshold. |
| 3 | Split result construction from persistence evidence. | Parent rollup: 2 IWU; child leaves score separately. |
| 4 | Confirmed dirty STL fixture generation is out of scope. | Parent is not an implementation leaf. |
| 5 | Confirmed source identity refusal is separate. | Parent ready as program document. |

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
  lives in Specs 05b2b2a and 05b2b2b.

## Acceptance

This spec is complete when Specs 05b2b2a and 05b2b2b can produce
surface-native successful/promoted results, preserve payload identity through
`.impress`, and prove no mesh fallback occurred.
