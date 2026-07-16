# Surface Spec 427c: Loft Primitive Adjacency Rebuild Diagnostics (v1.0)

Date: 2026-07-16
Status: Proposed
Primary ancestor: `../architecture/acd-loft-primitive-seam-shell-validity-execution.md`
Architecture ancestor: `../architecture/acd-loft-primitive-seam-shell-validity-execution.md`
Split provenance: `../specifications/surface-427-loft-primitive-seam-and-shell-assembly-v1_0.md`
Canonical status: Canonical leaf
Prerequisites:
- `../specifications/surface-427b-loft-primitive-candidate-shell-assembly-v1_0.md` - adjacency rebuild diagnostics consume candidate shell records.

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface.
Count: 1 IWU.
Basis: one adjacency rebuild diagnostic stage after candidate assembly.

## Purpose

Rebuild and diagnose candidate shell adjacency before runtime validity checks.

## Scope

Owns:

- Candidate shell adjacency rebuild helper.
- Adjacency-complete evidence and missing/duplicate/inconsistent adjacency diagnostics.
- Public-route evidence that shell adjacency is ready for validity checks.

Does not own:

- Candidate shell assembly.
- Runtime validity checking.
- Persistence or tessellation readiness.

## Split Coverage

- Parent spec: `../specifications/surface-427-loft-primitive-seam-and-shell-assembly-v1_0.md`
- Parent coverage status: 100% covered by Surface Specs 427a-427c.
- Parent responsibility covered: adjacency rebuild and diagnostics.
- Parent responsibilities outside this leaf: seam/use pairing and candidate shell assembly.
- Parent responsibilities still missing from children:
  - none.

## Refinement History

| Iteration | Date | Scope reviewed | Files written before barrier | Updates made | IWU/score before | IWU/score after | Split decision | Split artifacts | Child re-review status | Next scope after readback |
|---:|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-07-16 | Parent split creation | this spec and paired test spec | Created focused adjacency rebuild diagnostic leaf | 3 IWU parent | 1 IWU | no split | none | pending ledger review | runtime validity specs |

## Implementation Routing

- Primary modules/files:
  - `src/impression/modeling/csg.py` - adjacency rebuild diagnostics.
  - `src/impression/modeling/surface.py` - body adjacency helpers if needed.
- Supporting modules/files:
  - `tests/test_surface_csg.py` - adjacency diagnostics.

## API And Data Contract

- Adjacency evidence must identify shell id, paired uses, rebuilt adjacency links, and blocking diagnostics.
- Adjacency evidence must be returned before runtime validity checks.

## Required DTOs / Functions / Components

- DTOs/models: adjacency rebuild evidence record and adjacency diagnostic payload.
- Functions/methods: candidate shell adjacency rebuild helper.
- UI components/fields: not applicable.

## Performance Contract

- Bounded by candidate shell use and adjacency counts.
- No tessellation fallback is permitted.

## Error And State Behavior

- Missing, duplicate, or inconsistent adjacency refuses deterministically.

## Test Strategy

- Unit tests: complete adjacency, missing link, duplicate link, inconsistent link.
- Integrated route tests: public route exposes adjacency rebuild evidence.
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

Adversarial checks: adjacency rebuild and diagnostics are one validity-preparation step; runtime validity and persistence remain separate specs.

Split decision: no split.

## Acceptance Criteria

- Candidate shells produce explicit adjacency-complete evidence before validity checks.
- Invalid adjacency refuses before runtime validity.
- Diagnostics identify the offending shell/use ids.
