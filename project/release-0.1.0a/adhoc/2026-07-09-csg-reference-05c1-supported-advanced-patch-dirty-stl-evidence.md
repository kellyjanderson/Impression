# CSG Reference Spec 05c1: Supported Advanced Patch Dirty STL Evidence

Date: 2026-07-09
Status: Draft
Path: ad-hoc-path work

## Summary

Generate dirty STL review fixtures for advanced patch-family CSG routes that
already execute as surfaced CSG. This leaf never creates fake STLs for
unsupported routes.

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one reference artifact generation boundary for supported advanced routes.

## Scope

In scope:

- dirty STL generation for supported advanced CSG routes
- fixture records with purpose, methodology, and render description
- inventory tests proving artifacts exist
- clear failure when a route stops returning a surfaced success

Out of scope:

- unsupported diagnostic fixtures
- solver implementation
- dirty-to-gold promotion

## Implementation Boundary

Owner files:

- `tests/test_reference_stl_expansion.py`
- `tests/reference_review_fixtures/stl_review_sources.py`
- `tests/reference_review_fixtures/dirty-stl-fixtures.json`
- `project/release-0.1.0a/reference-stl/dirty/`

## Verification

- fixture inventory coverage
- missing artifact failure tests
- no-hidden-mesh-fallback evidence for supported artifact generators

Project readiness fields:
- Implementation owner/module: src/impression/modeling/csg.py
- Chosen defaults / parameters: No mesh fallback; tessellation only after a surfaced result succeeds at artifact, preview, or export boundaries.
- Chosen defaults / parameters: Unsupported routes return deterministic refusal records with family, operation, phase, and reason where applicable.
- Test strategy: fixture inventory coverage; missing artifact failure tests; no-hidden-mesh-fallback evidence for supported artifact generators
- Data ownership: `SurfaceBody` and `SurfaceBooleanResult` records remain the authored source of truth for CSG execution.
- Data ownership: Reference fixture metadata owns review evidence, dirty artifact context, and diagnostic evidence records.
- Routes: CSG support matrix to surfaced operation planner route
- Reuse/extraction decision: Add to existing CSG, surface, persistence, or reference helper modules named by this spec; do not create public API unless the spec says so.
- UI field/control inventory: not applicable; these CSG manifest entries have no UI controls or visible fields

Open questions / nuance discovered:
- Exact geometric tolerances and fixture thresholds should follow the existing CSG and reference-test helpers unless a child implementation finds a case requiring an explicit local value.
- If implementation discovers a route needs mesh-derived source truth, keep the route refused and record the missing surfaced capability instead of adding fallback execution.

## Five-Pass Review History

| Pass | Update | Rescore result |
| --- | --- | --- |
| 1 | Split from evidence parent. | Candidate became one supported-artifact leaf. |
| 2 | Limited scope to dirty STL fixtures. | 1 IWU retained. |
| 3 | Clarified unsupported routes are out of scope. | Manifest score below split threshold. |
| 4 | Confirmed promotion is out of scope. | Cohesive leaf. |
| 5 | Confirmed route success is prerequisite. | Final draft score: 21.5. |

## Manifest Assessment

Score:

- Functions/methods: 2 x 2 = 4
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 3 x 1 = 3
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
- Readiness blockers: 2 x 2 = 4
- Total: 24

Readiness blockers:
- [ ] Exact route thresholds or fixture tolerances may need confirmation during implementation.
- [x] Implementation owner/module is named or inherited from the implementation boundary.
- [x] Reuse/extraction decision is explicit.
- [x] UI field/control inventory is not applicable.
- [x] Test strategy is named by verification requirements.
- [x] Data ownership is explicit enough for implementation planning.
- [x] Privacy/logging rule is not applicable beyond deterministic diagnostics.

Split decision:

- No split required. Cohesion reason: this leaf owns only dirty STL evidence
  for routes already proven supported.

## Acceptance

This spec is complete when every supported advanced patch-family CSG route has
a dirty STL review fixture and fixture inventory tests prove the artifacts are
present.
