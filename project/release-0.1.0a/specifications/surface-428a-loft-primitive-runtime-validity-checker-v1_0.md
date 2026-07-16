# Surface Spec 428a: Loft Primitive Runtime Validity Checker (v1.0)

Date: 2026-07-16
Status: Proposed
Primary ancestor: `../architecture/acd-loft-primitive-seam-shell-validity-execution.md`
Architecture ancestor: `../architecture/acd-loft-primitive-seam-shell-validity-execution.md`
Split provenance: `../specifications/surface-428-loft-primitive-runtime-validity-and-persistence-gate-v1_0.md`
Canonical status: Canonical leaf
Prerequisites:
- `../specifications/surface-427c-loft-primitive-adjacency-rebuild-diagnostics-v1_0.md` - runtime validity checks consume adjacency-complete candidate shells.

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface.
Count: 1 IWU.
Basis: one validity checker for candidate loft/primitive CSG shells.

## Purpose

Run surface-body runtime validity checks on candidate loft/primitive CSG shells before persistence.

## Scope

Owns:

- Validity checker helper for candidate shells.
- Closure, manifoldness, orientation, and boundary-consistency diagnostics.
- Public-route validity evidence.

Does not own:

- Adjacency rebuild.
- Persistence/tessellation readiness.
- No-hidden-mesh acceptance proof.

## Split Coverage

- Parent spec: `../specifications/surface-428-loft-primitive-runtime-validity-and-persistence-gate-v1_0.md`
- Parent coverage status: 100% covered by Surface Specs 428a-428c.
- Parent responsibility covered: runtime validity checking.
- Parent responsibilities outside this leaf: persistence/readiness and no-hidden-mesh proof.
- Parent responsibilities still missing from children:
  - none.

## Refinement History

| Iteration | Date | Scope reviewed | Files written before barrier | Updates made | IWU/score before | IWU/score after | Split decision | Split artifacts | Child re-review status | Next scope after readback |
|---:|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-07-16 | Parent split creation | this spec and paired test spec | Created focused runtime-validity leaf | 3 IWU parent | 1 IWU | no split | none | pending ledger review | persistence sibling specs |

## Implementation Routing

- Primary modules/files:
  - `src/impression/modeling/csg.py` - validity gate integration.
  - `src/impression/modeling/surface.py` - reusable body validity helpers if needed.
- Supporting modules/files:
  - `tests/test_surface_csg.py` - validity diagnostics.

## API And Data Contract

- Validity evidence must identify candidate shell id, validity state, diagnostic code, and blocking source evidence.
- Invalid shells must not be persisted.

## Required DTOs / Functions / Components

- DTOs/models: runtime validity evidence record and validity diagnostic payload.
- Functions/methods: candidate shell runtime validity checker.
- UI components/fields: not applicable.

## Performance Contract

- Bounded by shell, patch, edge, and loop counts.
- No tessellation fallback is permitted.

## Error And State Behavior

- Invalid closure, non-manifold adjacency, inconsistent orientation, or stale evidence refuses deterministically.

## Test Strategy

- Unit tests: valid shell, open shell, non-manifold adjacency, inconsistent orientation.
- Integrated route tests: public route exposes validity evidence.
- Production-data rule: tests must not require production data.

## Review Score

Adversarial review date: 2026-07-16.

- Functions/methods: 1 x 2 = 2
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 1 x 1 = 1
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 2 x 1 = 2
- Performance-sensitive behavior: 1 x 2 = 2
- Test scenarios/fixtures: 4 x 1 = 4
- Readiness blockers: 0 x 2 = 0
- Total: 14.5

Adversarial checks: closure, manifoldness, orientation, and boundary consistency are cases in one validity checker; persistence and no-hidden-mesh proof stay outside this leaf.

Split decision: no split.

## Acceptance Criteria

- Valid candidate shells receive explicit runtime-valid evidence.
- Invalid shells refuse before persistence.
- Validity diagnostics identify the blocking condition.
