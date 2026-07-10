# CSG Reference Spec 04a: Lofted Body Eligibility And Refusal Diagnostics

Date: 2026-07-09
Status: Draft
Path: ad-hoc-path work

## Summary

Define eligibility and refusal diagnostics for loft-generated surfaced bodies
before they enter CSG execution.

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one lofted body eligibility/refusal boundary.

## Scope

In scope:

- single-shell and closed-valid eligibility rules
- underconstrained, branching, multi-shell, and self-intersection risk diagnostics
- loft provenance metadata needed by CSG routes

Out of scope:

- ruled patch execution
- dirty STL fixture generation

## Implementation Boundary

Owner modules:

- `src/impression/modeling/csg.py`
- `src/impression/modeling/loft.py` only for metadata/eligibility hooks

## Verification

- eligible loft body tests
- underconstrained/branching refusal tests
- no-hidden-mesh-fallback refusal tests

Project readiness fields:
- Implementation owner/module: src/impression/modeling/csg.py
- Chosen defaults / parameters: No mesh fallback; tessellation only after a surfaced result succeeds at artifact, preview, or export boundaries.
- Chosen defaults / parameters: Unsupported routes return deterministic refusal records with family, operation, phase, and reason where applicable.
- Test strategy: eligible loft body tests; underconstrained/branching refusal tests; no-hidden-mesh-fallback refusal tests
- Data ownership: `SurfaceBody` and `SurfaceBooleanResult` records remain the authored source of truth for CSG execution.
- Data ownership: Reference fixture metadata owns review evidence, dirty artifact context, and diagnostic evidence records.
- Routes: loft or ruled-patch eligibility route into surfaced CSG planner
- Reuse/extraction decision: Add to existing CSG, surface, persistence, or reference helper modules named by this spec; do not create public API unless the spec says so.
- UI field/control inventory: not applicable; these CSG manifest entries have no UI controls or visible fields

Open questions / nuance discovered:
- Exact geometric tolerances and fixture thresholds should follow the existing CSG and reference-test helpers unless a child implementation finds a case requiring an explicit local value.
- If implementation discovers a route needs mesh-derived source truth, keep the route refused and record the missing surfaced capability instead of adding fallback execution.

## Five-Pass Review History

| Pass | Update | Rescore result |
| --- | --- | --- |
| 1 | Split from lofted CSG parent. | Candidate became one eligibility leaf. |
| 2 | Kept ruled execution out of scope. | 1 IWU retained. |
| 3 | Added self-intersection risk diagnostic. | Manifest score below split threshold. |
| 4 | Confirmed loft planner semantics do not change. | Cohesive leaf. |
| 5 | Confirmed dirty STL fixtures are out of scope. | Final draft score: 19.5. |

## Manifest Assessment

Score:

- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
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
- Readiness blockers: 1 x 2 = 2
- Total: 21

Readiness blockers:
- [ ] Exact route thresholds or fixture tolerances may need confirmation during implementation.
- [x] Implementation owner/module is named or inherited from the implementation boundary.
- [x] Reuse/extraction decision is explicit.
- [x] UI field/control inventory is not applicable.
- [x] Test strategy is named by verification requirements.
- [x] Data ownership is explicit enough for implementation planning.
- [x] Privacy/logging rule is not applicable beyond deterministic diagnostics.

Split decision:

- No split required. Cohesion reason: one lofted body eligibility and refusal
  boundary.

## Acceptance

This spec is complete when lofted bodies are accepted or refused for CSG with
deterministic diagnostics and no mesh fallback.
