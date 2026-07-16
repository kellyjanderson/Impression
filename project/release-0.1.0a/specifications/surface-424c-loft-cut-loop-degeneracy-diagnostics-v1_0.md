# Surface Spec 424c: Loft Cut Loop Degeneracy Diagnostics (v1.0)

Date: 2026-07-16
Status: Proposed
Primary ancestor: `../architecture/acd-loft-primitive-intersection-and-cut-loop-kernel.md`
Architecture ancestor: `../architecture/acd-loft-primitive-intersection-and-cut-loop-kernel.md`
Manifest source: `Loft Patch-Local Cut Loop Construction`
Split provenance: `../specifications/surface-424-loft-patch-local-cut-loop-construction-v1_0.md`
Canonical status: Canonical leaf
Prerequisites:
- `../specifications/surface-424b-loft-cut-loop-closure-and-boundary-participation-v1_0.md` - degeneracy diagnostics classify closed or failed loop records.

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one diagnostic classifier for tangent, grazing, zero-area, and open-loop outcomes.

## Purpose

Classify cut-loop degeneracy and produce deterministic refusal diagnostics before generated cap construction.

## Scope

Owns:
- tangent and grazing classification
- zero-area loop classification
- open-loop and invalid closure diagnostics

Does not own:
- curve inversion
- loop closure construction
- cap construction

## Split Coverage

- Parent spec: `../specifications/surface-424-loft-patch-local-cut-loop-construction-v1_0.md`
- Parent coverage status: 100% covered by Surface Specs 424a-424c.
- Parent responsibilities owned by this child:
  - tangent, grazing, zero-area, and open-loop diagnostics.
- Parent responsibilities still missing from children:
  - none.

## Refinement History

| Iteration | Date | Scope reviewed | Files written before barrier | Updates made | IWU/score before | IWU/score after | Split decision | Split artifacts | Child re-review status | Next scope after readback |
|---:|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-07-16 | Parent split | this spec and paired test spec | Created child spec from Surface Spec 424 | parent 3 IWU | 1 IWU | no split | none | not applicable | ledger readback |

## Implementation Routing

- Primary modules/files:
  - `src/impression/modeling/csg.py` - loop degeneracy classifier and diagnostics.
- Tests:
  - `tests/test_surface_csg.py` - tangent, grazing, zero-area, and open-loop refusal cases.

## Chosen Defaults / Parameters

- Tangent and zero-area loops refuse unless earlier classified as exact no-cut/touching.
- Diagnostics inherit CSG tolerance policy.

## Data Ownership

- Source of truth: loop diagnostic payloads owned by CSG.
- Read ownership: generated cap construction only consumes non-degenerate loops.
- Write ownership: `src/impression/modeling/csg.py`.
- Privacy/logging constraints: diagnostics avoid raw geometry dumps.

## Dependencies And Routes

- Domain/service dependencies: Surface Spec 424b cut-loop records.
- Database dependencies: not applicable.
- GUI route, if applicable: not applicable.
- Background/concurrency route, if applicable: not applicable.

## Application Integration

- App type: library-only
- User/caller surface: public Boolean API consumers
- Invocation route: loop closure to degeneracy diagnostics through the loft CSG route
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: accepted non-degenerate loops or deterministic refusal diagnostics
- Integration validation: public-route tests for tangent, grazing, zero-area, and open-loop refusal
- Incomplete status risk: generated cap construction could consume invalid loops

## Required DTOs / Functions / Components

- DTOs/models:
  - `LoftCutLoopDegeneracyDiagnostic` - degeneracy class, patch id, tolerance context.
- Functions/methods:
  - `classify_loft_cut_loop_degeneracy(...)`.
- UI fields / visible data, UI elements, and UI components:
  - not applicable.

## Performance Contract

- Bounded by loop and segment count.

## Error And State Behavior

- Degenerate loops refuse before cap construction.

## Test Strategy

- Unit tests: each degeneracy class.
- Integrated route tests: public route exposes refusal diagnostics.
- Production-data rule: tests must not require production data.

## Review Score

Adversarial review date: 2026-07-16.

- Functions/methods: 1 x 2 = 2
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 1 x 1 = 1
- Returns/outputs/signals: 1 x 1 = 1
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Performance-sensitive behavior: 1 x 2 = 2
- Test scenarios/fixtures: 4 x 1 = 4
- Readiness blockers: 0 x 2 = 0
- Total: 12.5

Adversarial checks: multiple degeneracy classes are cases of one diagnostic classifier, not separate implementation artifacts; generated cap behavior remains an out-of-scope consumer.

Split decision: no split.

## Acceptance Criteria

- Tangent, grazing, zero-area, and open-loop outcomes are classified deterministically.
- Degenerate loops do not reach generated cap construction.
