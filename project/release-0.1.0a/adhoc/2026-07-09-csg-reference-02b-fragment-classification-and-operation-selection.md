# CSG Reference Spec 02b: Fragment Classification And Operation Selection

Date: 2026-07-09
Status: Draft
Path: ad-hoc-path work

## Summary

Classify trim arrangement fragments and select the kept fragments for union,
difference, and intersection operations.

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one fragment classification and operation selection boundary.

## Scope

In scope:

- inside/outside/on-boundary classification for arrangement fragments
- operation selection rules for union, difference, and intersection
- tangent/coincident ambiguity diagnostics when classification is not stable

Out of scope:

- arrangement graph construction
- final shell/seam assembly

## Implementation Boundary

Owner modules:

- `src/impression/modeling/csg.py`

## Verification

- fragment classification tests for curved and planar patch fragments
- operation selection tests
- tangent/coincident ambiguity diagnostic tests

Project readiness fields:
- Implementation owner/module: src/impression/modeling/csg.py
- Chosen defaults / parameters: No mesh fallback; tessellation only after a surfaced result succeeds at artifact, preview, or export boundaries.
- Chosen defaults / parameters: Tangent or degenerate cases return deterministic diagnostics instead of unstable sliver geometry.
- Test strategy: fragment classification tests for curved and planar patch fragments; operation selection tests; tangent/coincident ambiguity diagnostic tests
- Data ownership: `SurfaceBody` and `SurfaceBooleanResult` records remain the authored source of truth for CSG execution.
- Data ownership: Reference fixture metadata owns review evidence, dirty artifact context, and diagnostic evidence records.
- Routes: CSG intersection output to patch-local trim/fragment reconstruction route
- Reuse/extraction decision: Add to existing CSG, surface, persistence, or reference helper modules named by this spec; do not create public API unless the spec says so.
- UI field/control inventory: not applicable; these CSG manifest entries have no UI controls or visible fields

Open questions / nuance discovered:
- Exact geometric tolerances and fixture thresholds should follow the existing CSG and reference-test helpers unless a child implementation finds a case requiring an explicit local value.
- If implementation discovers a route needs mesh-derived source truth, keep the route refused and record the missing surfaced capability instead of adding fallback execution.

## Five-Pass Review History

| Pass | Update | Rescore result |
| --- | --- | --- |
| 1 | Split from reconstruction parent. | Candidate became one classification leaf. |
| 2 | Limited scope to fragment classification and operation selection. | 1 IWU retained. |
| 3 | Moved graph construction out. | Manifest score below split threshold. |
| 4 | Moved shell assembly out. | Cohesive leaf. |
| 5 | Confirmed ambiguity diagnostics are in scope. | Final draft score: 19.5. |

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

- No split required. Cohesion reason: one fragment classification and selection
  boundary.

## Acceptance

This spec is complete when arrangement fragments are classified deterministically
and operation selection emits kept-fragment records or explicit ambiguity
diagnostics.
