# CSG Reference Spec 05b2b2a: Displacement Result Construction Route

Date: 2026-07-09
Status: Draft
Path: ad-hoc-path work

## Summary

Construct successful or promoted displacement CSG results after source identity
and source-domain compatibility have already been validated.

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one successful displacement result construction boundary.

## Scope

In scope:

- successful or promoted displacement CSG result construction
- result metadata/provenance for displacement source patches
- no-hidden-mesh fallback checks for result construction

Out of scope:

- source identity refusal behavior
- `.impress` persistence evidence
- dirty STL fixture generation

## Implementation Boundary

Owner modules:

- `src/impression/modeling/csg.py`
- `src/impression/modeling/surface.py`

## Reference Items Unblocked

- construction portions of `RT-PATCH-CSG-010`, `RT-PATCH-CSG-012`, and
  `RT-PATCH-CSG-014`

## Verification

- successful/promoted displacement result tests
- result provenance tests
- no-hidden-mesh-fallback tests for successful construction routes

Project readiness fields:
- Implementation owner/module: src/impression/modeling/csg.py
- Chosen defaults / parameters: No mesh fallback; tessellation only after a surfaced result succeeds at artifact, preview, or export boundaries.
- Chosen defaults / parameters: Unsupported routes return deterministic refusal records with family, operation, phase, and reason where applicable.
- Test strategy: successful/promoted displacement result tests; result provenance tests; no-hidden-mesh-fallback tests for successful construction routes
- Data ownership: `SurfaceBody` and `SurfaceBooleanResult` records remain the authored source of truth for CSG execution.
- Data ownership: `.impress` payloads own persistence evidence only after surfaced/native or promoted payloads are produced.
- Data ownership: Reference fixture metadata owns review evidence, dirty artifact context, and diagnostic evidence records.
- Routes: advanced family route through CSG support matrix, payload persistence, or refusal diagnostics
- Reuse/extraction decision: Add to existing CSG, surface, persistence, or reference helper modules named by this spec; do not create public API unless the spec says so.
- UI field/control inventory: not applicable; these CSG manifest entries have no UI controls or visible fields

Open questions / nuance discovered:
- Exact geometric tolerances and fixture thresholds should follow the existing CSG and reference-test helpers unless a child implementation finds a case requiring an explicit local value.
- If implementation discovers a route needs mesh-derived source truth, keep the route refused and record the missing surfaced capability instead of adding fallback execution.

## Five-Pass Review History

| Pass | Update | Rescore result |
| --- | --- | --- |
| 1 | Split from displacement success parent. | Candidate became one result-construction leaf. |
| 2 | Limited scope to valid-source construction. | 1 IWU retained. |
| 3 | Moved persistence evidence out. | Manifest score below split threshold. |
| 4 | Confirmed source identity refusal is prerequisite. | Cohesive leaf. |
| 5 | Confirmed dirty STL fixture generation is out of scope. | Final draft score: 21. |

## Manifest Assessment

Score:

- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
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
- Total: 21.5

Readiness blockers:
- [ ] Exact route thresholds or fixture tolerances may need confirmation during implementation.
- [x] Implementation owner/module is named or inherited from the implementation boundary.
- [x] Reuse/extraction decision is explicit.
- [x] UI field/control inventory is not applicable.
- [x] Test strategy is named by verification requirements.
- [x] Data ownership is explicit enough for implementation planning.
- [x] Privacy/logging rule is not applicable beyond deterministic diagnostics.

Split decision:

- No split required. Cohesion reason: one valid-source displacement result
  construction boundary.

## Acceptance

This spec is complete when valid-source displacement CSG can construct
surface-native successful/promoted results with provenance and no mesh fallback.
