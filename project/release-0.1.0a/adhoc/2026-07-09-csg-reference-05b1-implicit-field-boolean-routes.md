# CSG Reference Spec 05b1: Implicit Field Boolean Routes

Date: 2026-07-09
Status: Draft
Path: ad-hoc-path work

## Summary

Implement or explicitly refuse implicit field CSG routes without extracting mesh
truth. Successful implicit routes must use declarative field composition or a
documented surfaced adapter route.

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one implicit field composition/refusal boundary.

## Scope

In scope:

- implicit field union, difference, and intersection composition policy
- adapter refusal diagnostics for unsupported implicit operands
- `.impress` payload preservation for successful implicit results
- no-hidden-mesh fallback checks

Out of scope:

- heightmap and displacement routes
- reference fixture generation
- implicit contour extraction as authored CSG truth

## Implementation Boundary

Owner modules:

- `src/impression/modeling/csg.py`
- `src/impression/modeling/surface.py`
- `src/impression/io/impress.py` for persistence evidence only

## Reference Items Unblocked

- `RT-PATCH-CSG-008`
- implicit portions of `RT-PATCH-CSG-012`, `RT-PATCH-CSG-013`, and
  `RT-PATCH-CSG-014`

## Verification

- implicit field composition tests
- adapter refusal tests
- `.impress` round-trip tests for successful implicit payloads
- no-hidden-mesh-fallback tests

Project readiness fields:
- Implementation owner/module: src/impression/modeling/csg.py
- Chosen defaults / parameters: No mesh fallback; tessellation only after a surfaced result succeeds at artifact, preview, or export boundaries.
- Chosen defaults / parameters: Sampled or implicit source truth must be preserved, promoted with provenance, or refused explicitly.
- Test strategy: implicit field composition tests; adapter refusal tests; `.impress` round-trip tests for successful implicit payloads; no-hidden-mesh-fallback tests; Implementation owner/module: src/impression/modeling/csg.py; Chosen defaults / parameters: No mesh fallback; tessellation only after a surfaced result succeeds at artifact, preview, or export boundaries.; Chosen defaults / parameters: Unsupported routes return deterministic refusal records with family, operation, phase, and reason where applicable.; Test strategy: implicit field composition tests; adapter refusal tests; `.impress` round-trip tests for successful implicit payloads; no-hidden-mesh-fallback tests; Data ownership: `SurfaceBody` and `SurfaceBooleanResult` records remain the authored source of truth for CSG execution.; Data ownership: `.impress` payloads own persistence evidence only after surfaced/native or promoted payloads are produced.; Data ownership: Reference fixture metadata owns review evidence, dirty artifact context, and diagnostic evidence records.; Routes: advanced family route through CSG support matrix, payload persistence, or refusal diagnostics; Reuse/extraction decision: Add to existing CSG, surface, persistence, or reference helper modules named by this spec; do not create public API unless the spec says so.; UI field/control inventory: not applicable; these CSG manifest entries have no UI controls or visible fields; Exact geometric tolerances and fixture thresholds should follow the existing CSG and reference-test helpers unless a child implementation finds a case requiring an explicit local value.; If implementation discovers a route needs mesh-derived source truth, keep the route refused and record the missing surfaced capability instead of adding fallback execution.
- Data ownership: `SurfaceBody` and `SurfaceBooleanResult` records remain the authored source of truth for CSG execution.
- Data ownership: `.impress` payloads own persistence evidence only after surfaced/native or promoted payloads are produced.
- Data ownership: Reference fixture metadata owns review evidence, dirty artifact context, and diagnostic evidence records.
- Routes: implicit field operands to expression-preserving or refusal CSG route
- Reuse/extraction decision: Reuse existing helpers named in the implementation boundary; add private helpers only when extraction keeps owner modules cohesive.
- UI field/control inventory: not applicable; these CSG manifest entries have no UI controls or visible fields

Open questions / nuance discovered:
- Exact geometric tolerances and fixture thresholds should follow the existing CSG and reference-test helpers unless a child implementation finds a case requiring an explicit local value.
- If implementation discovers a route needs mesh-derived source truth, keep the route refused and record the missing surfaced capability instead of adding fallback execution.

## Five-Pass Review History

| Pass | Update | Rescore result |
| --- | --- | --- |
| 1 | Split from sampled/implicit parent. | Candidate became one implicit route leaf. |
| 2 | Limited scope to field composition/refusal. | 1 IWU retained. |
| 3 | Added persistence evidence only for successful payloads. | Manifest score below split threshold. |
| 4 | Excluded contour extraction as authored CSG truth. | Cohesive leaf. |
| 5 | Confirmed reference fixtures are out of scope. | Final draft score: 20.5. |

## Manifest Assessment

Score:

- Functions/methods: 3 x 2 = 6
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
- Readiness blockers: 1 x 2 = 2
- Total: 23

Readiness blockers:
- [ ] Exact route thresholds or fixture tolerances may need confirmation during implementation.
- [x] Implementation owner/module is named or inherited from the implementation boundary.
- [x] Reuse/extraction decision is explicit.
- [x] UI field/control inventory is not applicable.
- [x] Test strategy is named by verification requirements.
- [x] Data ownership is explicit enough for implementation planning.
- [x] Privacy/logging rule is not applicable beyond deterministic diagnostics.

Split decision:

- No split required. Cohesion reason: one implicit field route boundary with
  persistence evidence only for successful payloads.

## Acceptance

This spec is complete when implicit CSG routes either execute through
surface-native field composition or refuse with deterministic diagnostics and no
mesh fallback.
