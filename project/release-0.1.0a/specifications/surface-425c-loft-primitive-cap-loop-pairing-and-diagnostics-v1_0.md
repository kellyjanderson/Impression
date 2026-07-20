# Surface Spec 425c: Loft Primitive Cap Loop Pairing And Diagnostics (v1.0)

Date: 2026-07-16
Status: Proposed
Primary ancestor: `../architecture/acd-loft-primitive-generated-cap-and-topology-policy.md`
Architecture ancestor: `../architecture/acd-loft-primitive-generated-cap-and-topology-policy.md`
Manifest source: `Loft Primitive Generated Cap Construction`
Split provenance: `../specifications/surface-425-loft-primitive-generated-cap-construction-v1_0.md`
Canonical status: Canonical leaf
Prerequisites:
- `../specifications/surface-425b-loft-primitive-generated-cap-record-construction-v1_0.md` - pairing consumes generated cap records.

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one cap-loop pairing and diagnostic completeness gate.

## Purpose

Verify generated caps pair with the expected loft cut loops before topology selection.

## Scope

Owns:
- cap-loop pairing records
- unpaired or duplicate cap-loop diagnostics
- cap readiness handoff to topology selection

Does not own:
- cap support classification
- cap record construction
- shell assembly

## Split Coverage

- Parent spec: `../specifications/surface-425-loft-primitive-generated-cap-construction-v1_0.md`
- Parent coverage status: 100% covered by Surface Specs 425a-425c.
- Parent responsibilities owned by this child:
  - cap-loop pairing and diagnostics.
- Parent responsibilities still missing from children:
  - none.

## Refinement History

| Iteration | Date | Scope reviewed | Files written before barrier | Updates made | IWU/score before | IWU/score after | Split decision | Split artifacts | Child re-review status | Next scope after readback |
|---:|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-07-16 | Parent split | this spec and paired test spec | Created child spec from Surface Spec 425 | parent 3 IWU | 1 IWU | no split | none | not applicable | ledger readback |

## Implementation Routing

- Primary modules/files:
  - `src/impression/modeling/csg.py` - cap-loop pairing records and diagnostics.
- Tests:
  - `tests/test_surface_csg.py` - paired, unpaired, and duplicate cap loop cases.

## Chosen Defaults / Parameters

- Every generated cap loop must pair exactly once with a loft cut loop.
- Unpaired and duplicate pairings refuse before topology selection.

## Data Ownership

- Source of truth: cap-loop pairing records owned by CSG.
- Read ownership: topology selection consumes cap readiness records.
- Write ownership: `src/impression/modeling/csg.py`.
- Privacy/logging constraints: diagnostics avoid raw geometry dumps.

## Dependencies And Routes

- Domain/service dependencies: generated cap records and closed cut loops.
- Database dependencies: not applicable.
- GUI route, if applicable: not applicable.
- Background/concurrency route, if applicable: not applicable.

## Application Integration

- App type: library-only
- User/caller surface: public Boolean API consumers
- Invocation route: generated cap records to cap-loop pairing through the loft CSG route
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: paired cap readiness records or pairing diagnostics
- Integration validation: cap-loop pairing tests plus public-route diagnostics
- Incomplete status risk: topology selection could consume incomplete cap boundaries

## Required DTOs / Functions / Components

- DTOs/models:
  - `LoftPrimitiveCapLoopPairingRecord` - generated cap loop to loft cut loop pairing.
  - `LoftPrimitiveCapLoopPairingDiagnostic` - unpaired or duplicate pair refusal.
- Functions/methods:
  - `pair_loft_primitive_generated_cap_loops(...)`.
- UI fields / visible data, UI elements, and UI components:
  - not applicable.

## Performance Contract

- Bounded by cap loop and cut loop count.

## Error And State Behavior

- Missing, unpaired, or duplicate cap loops refuse before topology selection.

## Test Strategy

- Unit tests: exact pairing, missing pairing, duplicate pairing.
- Integrated route tests: public route reaches pairing diagnostics.
- Production-data rule: tests must not require production data.

## Review Score

Adversarial review date: 2026-07-16.

- Functions/methods: 1 x 2 = 2
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 1 x 1 = 1
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Performance-sensitive behavior: 1 x 2 = 2
- Test scenarios/fixtures: 3 x 1 = 3
- Readiness blockers: 0 x 2 = 0
- Total: 12.5

Adversarial checks: pairing is one readiness gate between generated caps and topology selection; it does not create caps or select topology.

Split decision: no split.

## Acceptance Criteria

- Cap readiness records only exist when every generated cap loop pairs exactly once.
- Pairing diagnostics prevent incomplete boundaries from reaching topology selection.
