# Surface Spec 424b: Loft Cut Loop Closure And Boundary Participation (v1.0)

Date: 2026-07-16
Status: Proposed
Primary ancestor: `../architecture/acd-loft-primitive-intersection-and-cut-loop-kernel.md`
Architecture ancestor: `../architecture/acd-loft-primitive-intersection-and-cut-loop-kernel.md`
Manifest source: `Loft Patch-Local Cut Loop Construction`
Split provenance: `../specifications/surface-424-loft-patch-local-cut-loop-construction-v1_0.md`
Canonical status: Canonical leaf
Prerequisites:
- `../specifications/surface-424a-loft-patch-local-source-curve-inversion-v1_0.md` - loop closure consumes patch-local source curve records.

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one loop-closure step that combines patch-local segments with existing cap trims and station seams.

## Purpose

Close patch-local cut loops while preserving participation by existing cap trims and station seams.

## Scope

Owns:
- `LoftPatchLocalCutLoopRecord`
- segment ordering and closure
- cap-trim and station-seam participation records

Does not own:
- source curve inversion
- degeneracy refusal policy
- generated primitive caps

## Split Coverage

- Parent spec: `../specifications/surface-424-loft-patch-local-cut-loop-construction-v1_0.md`
- Parent coverage status: 100% covered by Surface Specs 424a-424c.
- Parent responsibilities owned by this child:
  - loop closure and boundary participation.
- Parent responsibilities still missing from children:
  - none.

## Refinement History

| Iteration | Date | Scope reviewed | Files written before barrier | Updates made | IWU/score before | IWU/score after | Split decision | Split artifacts | Child re-review status | Next scope after readback |
|---:|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-07-16 | Parent split | this spec and paired test spec | Created child spec from Surface Spec 424 | parent 3 IWU | 1 IWU | no split | none | not applicable | ledger readback |

## Implementation Routing

- Primary modules/files:
  - `src/impression/modeling/csg.py` - loop closure builder and records.
- Supporting modules/files:
  - `src/impression/modeling/surface.py` - trim-loop helpers.
- Tests:
  - `tests/test_surface_csg.py` - closed loops with cap-trim and station-seam participation.

## Chosen Defaults / Parameters

- Existing cap trims and station seams participate in closure.
- Closure failure emits diagnostics; it does not invent bridge segments.

## Data Ownership

- Source of truth: closed cut-loop records owned by CSG.
- Read ownership: degeneracy diagnostics and generated-cap construction consume closed loops.
- Write ownership: `src/impression/modeling/csg.py`.
- Privacy/logging constraints: diagnostics avoid raw geometry dumps.

## Dependencies And Routes

- Domain/service dependencies: Surface Spec 424a inversion records and surface trim helpers.
- Database dependencies: not applicable.
- GUI route, if applicable: not applicable.
- Background/concurrency route, if applicable: not applicable.

## Application Integration

- App type: library-only
- User/caller surface: public Boolean API consumers
- Invocation route: patch-local inversion to loop closure through the loft CSG route
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: closed cut loops or closure diagnostics
- Integration validation: loop closure tests plus public-route closure diagnostics
- Incomplete status risk: later stages could consume open or fabricated boundaries

## Required DTOs / Functions / Components

- DTOs/models:
  - `LoftPatchLocalCutLoopRecord` - closed patch-local loop.
  - `LoftCutLoopBoundaryParticipationRecord` - cap trim or station seam participation.
- Functions/methods:
  - `close_loft_patch_local_cut_loops(...)`.
- UI fields / visible data, UI elements, and UI components:
  - not applicable.

## Performance Contract

- Bounded by patch-local segment and boundary-participation counts.
- No tessellation fallback is permitted.

## Error And State Behavior

- Open loops and missing boundary participants return deterministic diagnostics.

## Test Strategy

- Unit tests: loop ordering, closure, cap-trim participation, station-seam participation.
- Integrated route tests: public route reaches closure diagnostics.
- Production-data rule: tests must not require production data.

## Review Score

Adversarial review date: 2026-07-16.

- Functions/methods: 1 x 2 = 2
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 1 x 1 = 1
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Performance-sensitive behavior: 1 x 2 = 2
- Test scenarios/fixtures: 4 x 1 = 4
- Readiness blockers: 0 x 2 = 0
- Total: 14.5

Adversarial checks: boundary participation is part of loop closure, not a second downstream assembly concern; source inversion and degeneracy refusal stay outside this leaf.

Split decision: no split.

## Acceptance Criteria

- Valid patch-local segments become closed cut loops.
- Existing cap trims and station seams are preserved in loop records.
- Closure failure refuses before generated cap construction.
