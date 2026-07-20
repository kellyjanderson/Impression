# Surface Spec 427b: Loft Primitive Candidate Shell Assembly (v1.0)

Date: 2026-07-16
Status: Proposed
Primary ancestor: `../architecture/acd-loft-primitive-seam-shell-validity-execution.md`
Architecture ancestor: `../architecture/acd-loft-primitive-seam-shell-validity-execution.md`
Split provenance: `../specifications/surface-427-loft-primitive-seam-and-shell-assembly-v1_0.md`
Canonical status: Canonical leaf
Prerequisites:
- `../specifications/surface-427a-loft-primitive-seam-use-pairing-v1_0.md` - candidate assembly consumes complete seam/use pairing.

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface.
Count: 1 IWU.
Basis: one candidate-shell assembly step from paired uses.

## Purpose

Build candidate surface-body shells from retained fragments, generated caps, and complete seam/use pairing records.

## Scope

Owns:

- Candidate shell assembly helper.
- Candidate shell assembly record with participating patches/caps/fragments.
- Assembly refusal for missing participants or unsupported topology classes.

Does not own:

- Pairing creation.
- Adjacency rebuild diagnostics.
- Runtime validity and persistence.

## Split Coverage

- Parent spec: `../specifications/surface-427-loft-primitive-seam-and-shell-assembly-v1_0.md`
- Parent coverage status: 100% covered by Surface Specs 427a-427c.
- Parent responsibility covered: candidate shell assembly.
- Parent responsibilities outside this leaf: seam/use pairing and adjacency rebuild diagnostics.
- Parent responsibilities still missing from children:
  - none.

## Refinement History

| Iteration | Date | Scope reviewed | Files written before barrier | Updates made | IWU/score before | IWU/score after | Split decision | Split artifacts | Child re-review status | Next scope after readback |
|---:|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-07-16 | Parent split creation | this spec and paired test spec | Created focused candidate-shell assembly leaf | 3 IWU parent | 1 IWU | no split | none | pending ledger review | adjacency sibling spec |

## Implementation Routing

- Primary modules/files:
  - `src/impression/modeling/csg.py` - candidate shell assembly.
  - `src/impression/modeling/surface.py` - surface body construction helpers if needed.
- Supporting modules/files:
  - `tests/test_surface_csg.py` - candidate assembly tests.

## API And Data Contract

- Candidate shell records must include operation, topology class, participating source records, and assembly readiness.
- Candidate shell construction must stay surface-body native.

## Required DTOs / Functions / Components

- DTOs/models: candidate shell assembly record and assembly refusal diagnostic.
- Functions/methods: candidate shell assembly helper.
- UI components/fields: not applicable.

## Performance Contract

- Bounded by retained fragment, generated cap, and pairing record counts.
- No mesh conversion is permitted.

## Error And State Behavior

- Missing participant evidence refuses before candidate body creation.

## Test Strategy

- Unit tests: valid exterior edit, valid cavity, missing participant refusal.
- Integrated route tests: public route reaches candidate shell evidence.
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
- Test scenarios/fixtures: 3 x 1 = 3
- Readiness blockers: 0 x 2 = 0
- Total: 13.5

Adversarial checks: use of both CSG and surface helpers stays within one candidate-shell assembly artifact; adjacency rebuild and runtime validity are explicitly excluded.

Split decision: no split.

## Acceptance Criteria

- Supported topology and complete pairing produce a candidate shell record.
- Unsupported or incomplete assembly refuses deterministically.
- Candidate shell assembly does not run validity or persistence gates.
