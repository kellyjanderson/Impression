# CSG Reference Spec 03: Multi-Operand Boolean Composition

Date: 2026-07-09
Status: Draft
Path: ad-hoc-path work

## Summary

Implement deterministic multi-operand surfaced boolean composition for the
reference expansion plan. The implementation may compose pairwise surfaced
results, but it must make ordering, diagnostics, metadata, and partial-failure
behavior explicit.

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one capability boundary: deterministic composition of multiple surfaced boolean operands using existing pairwise execution.

## Scope

In scope:

- deterministic ordering for `union` operands
- deterministic base-plus-cutter order for `difference`
- partial-failure diagnostics that identify the failing step and operands
- metadata provenance across each composition step
- support for nested cutters after Spec 01 and Spec 02 pairwise operations
  succeed

Out of scope:

- new analytic pair solvers
- new trim reconstruction behavior
- performance-oriented boolean tree optimization beyond deterministic
  left-fold composition

## Implementation Boundary

Owner module:

- `src/impression/modeling/csg.py`

Reuse:

- Reuse `SurfaceBooleanOperands` for the original request.
- Add an internal composition record only if needed to report stepwise
  diagnostics.
- Reuse `surface_boolean_result` for each pairwise step.

## Required Behavior

- Multi-operand `union` returns the same result for equivalent operand sets in
  any input order, unless ordering is documented as semantically significant for
  a specific operation.
- Multi-operand `difference` applies cutters in declared order and reports that
  order in provenance.
- If a later step is unsupported or invalid, the public result is unsupported or
  invalid with the failed step, operation, and operand IDs.
- Intermediate surfaced results are not tessellated for later steps.
- Empty intermediate results are handled explicitly.

## Reference Items Unblocked

- `RT-CSG-010` nested cutters: box minus sphere minus cylinder
- `RT-CSG-011` multi-operand union chain with deterministic ordering
- `RT-CSG-012` multi-operand difference chain with deterministic ordering

## Verification

Automated tests must cover:

- input-order determinism for multi-operand union
- declared cutter-order behavior for multi-operand difference
- diagnostics for unsupported pairwise step in a chain
- provenance metadata across all operands
- no hidden mesh fallback across intermediate results

Reference artifact verification must cover:

- dirty STL generation for nested cutter, union chain, and difference chain
- fixture descriptions that state operand order and expected visible effects

Project readiness fields:
- Implementation owner/module: src/impression/modeling/csg.py
- Chosen defaults / parameters: No mesh fallback; tessellation only after a surfaced result succeeds at artifact, preview, or export boundaries.
- Test strategy: input-order determinism for multi-operand union; declared cutter-order behavior for multi-operand difference; diagnostics for unsupported pairwise step in a chain; provenance metadata across all operands; no hidden mesh fallback across intermediate results
- Data ownership: `SurfaceBody` and `SurfaceBooleanResult` records remain the authored source of truth for CSG execution.
- Data ownership: Reference fixture metadata owns review evidence, dirty artifact context, and diagnostic evidence records.
- Routes: multi-operand request to deterministic pairwise surfaced CSG composition route
- Reuse/extraction decision: Reuse existing helpers named in the implementation boundary; add private helpers only when extraction keeps owner modules cohesive.
- UI field/control inventory: not applicable; these CSG manifest entries have no UI controls or visible fields

Open questions / nuance discovered:
- Exact geometric tolerances and fixture thresholds should follow the existing CSG and reference-test helpers unless a child implementation finds a case requiring an explicit local value.
- If implementation discovers a route needs mesh-derived source truth, keep the route refused and record the missing surfaced capability instead of adding fallback execution.

## Five-Pass Review History

| Pass | Update | Rescore result |
| --- | --- | --- |
| 1 | Added manifest-style scoring need and confirmed composition scope. | Candidate below split threshold. |
| 2 | Kept pairwise solver and trim reconstruction out of scope. | 1 IWU retained. |
| 3 | Confirmed diagnostics must identify failed composition step. | Cohesive leaf. |
| 4 | Confirmed deterministic ordering is part of the public behavior. | Score stable. |
| 5 | Confirmed intermediate results stay surfaced. | Final draft score: 21.5. |

## Manifest Assessment

Score:

- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 3 x 1 = 3
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
- Total: 22

Readiness blockers:
- [ ] Exact route thresholds or fixture tolerances may need confirmation during implementation.
- [x] Implementation owner/module is named or inherited from the implementation boundary.
- [x] Reuse/extraction decision is explicit.
- [x] UI field/control inventory is not applicable.
- [x] Test strategy is named by verification requirements.
- [x] Data ownership is explicit enough for implementation planning.
- [x] Privacy/logging rule is not applicable beyond deterministic diagnostics.

Split decision:

- No split required. Cohesion reason: this leaf owns deterministic composition
  on top of already-supported pairwise surfaced results.

## Acceptance

This spec is complete when every multi-operand reference item can be generated
from surfaced results and failing chains identify the exact unsupported or
invalid step.
