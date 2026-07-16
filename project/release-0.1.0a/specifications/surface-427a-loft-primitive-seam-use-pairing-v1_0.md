# Surface Spec 427a: Loft Primitive Seam Use Pairing (v1.0)

Date: 2026-07-16
Status: Proposed
Primary ancestor: `../architecture/acd-loft-primitive-seam-shell-validity-execution.md`
Architecture ancestor: `../architecture/acd-loft-primitive-seam-shell-validity-execution.md`
Split provenance: `../specifications/surface-427-loft-primitive-seam-and-shell-assembly-v1_0.md`
Canonical status: Canonical leaf
Prerequisites:
- `../specifications/surface-426c-loft-primitive-topology-orientation-and-refusal-diagnostics-v1_0.md` - seam/use pairing requires orientation-ready topology.

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface.
Count: 1 IWU.
Basis: one seam/use pairing stage before shell assembly.

## Purpose

Pair retained fragment boundaries, cap loop uses, and loft seam uses into explicit adjacency records.

## Scope

Owns:

- Seam/use pairing helper and records.
- One-to-one, one-to-many refusal, dangling-use refusal, and duplicate-use diagnostics.
- Public-route evidence that pairing completed before assembly.

Does not own:

- Candidate shell assembly.
- Final adjacency rebuild after assembly.
- Runtime validity checks.

## Split Coverage

- Parent spec: `../specifications/surface-427-loft-primitive-seam-and-shell-assembly-v1_0.md`
- Parent coverage status: 100% covered by Surface Specs 427a-427c.
- Parent responsibility covered: seam/use pairing.
- Parent responsibilities outside this leaf: shell assembly and adjacency rebuild diagnostics.
- Parent responsibilities still missing from children:
  - none.

## Refinement History

| Iteration | Date | Scope reviewed | Files written before barrier | Updates made | IWU/score before | IWU/score after | Split decision | Split artifacts | Child re-review status | Next scope after readback |
|---:|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-07-16 | Parent split creation | this spec and paired test spec | Created focused seam/use pairing leaf | 3 IWU parent | 1 IWU | no split | none | pending ledger review | shell assembly sibling specs |

## Implementation Routing

- Primary modules/files:
  - `src/impression/modeling/csg.py` - seam/use pairing records and helper.
- Supporting modules/files:
  - `tests/test_surface_csg.py` - pairing diagnostics.

## API And Data Contract

- Pairing records must identify each boundary use, counterpart use, source patch/cap id, and pairing reason.
- Pairing evidence must be observable before candidate shell assembly.

## Required DTOs / Functions / Components

- DTOs/models: seam/use pairing record and pairing diagnostic payload.
- Functions/methods: boundary-use pairing helper.
- UI components/fields: not applicable.

## Performance Contract

- Bounded by boundary-use count.
- No tessellation fallback is permitted.

## Error And State Behavior

- Dangling, duplicate, or ambiguous uses refuse before shell assembly.

## Test Strategy

- Unit tests: valid pairing, dangling use, duplicate use, ambiguous one-to-many use.
- Integrated route tests: public route exposes pairing diagnostics.
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

Adversarial checks: pairing all boundary-use types is one adjacency-preparation gate; candidate shell assembly and adjacency rebuild remain separate specs.

Split decision: no split.

## Acceptance Criteria

- Valid topology produces complete seam/use pairing records.
- Invalid pairing refuses before shell assembly.
- Pairing records preserve source patch and generated cap identity.
