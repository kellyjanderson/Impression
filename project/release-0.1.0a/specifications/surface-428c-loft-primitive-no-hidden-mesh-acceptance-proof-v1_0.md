# Surface Spec 428c: Loft Primitive No Hidden Mesh Acceptance Proof (v1.0)

Date: 2026-07-16
Status: Proposed
Primary ancestor: `../architecture/acd-loft-primitive-seam-shell-validity-execution.md`
Architecture ancestor: `../architecture/acd-loft-primitive-seam-shell-validity-execution.md`
Split provenance: `../specifications/surface-428-loft-primitive-runtime-validity-and-persistence-gate-v1_0.md`
Canonical status: Canonical leaf
Prerequisites:
- `../specifications/surface-428b-loft-primitive-persistence-and-tessellation-readiness-v1_0.md` - proof consumes accepted persistence/readiness records.

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface.
Count: 1 IWU.
Basis: one acceptance proof that hidden mesh fallback was not used.

## Purpose

Prove accepted loft/primitive CSG results came from surface-body construction, not hidden tessellation or mesh fallback.

## Scope

Owns:

- No-hidden-mesh acceptance evidence for loft/primitive CSG.
- Audit assertions that accepted routes do not instantiate mesh fallback during construction.
- Public-route proof fields for implementation and reference handoff.

Does not own:

- Runtime validity.
- Persistence readiness.
- Reference fixture generation.

## Split Coverage

- Parent spec: `../specifications/surface-428-loft-primitive-runtime-validity-and-persistence-gate-v1_0.md`
- Parent coverage status: 100% covered by Surface Specs 428a-428c.
- Parent responsibility covered: no-hidden-mesh acceptance proof.
- Parent responsibilities outside this leaf: runtime validity and persistence/readiness.
- Parent responsibilities still missing from children:
  - none.

## Refinement History

| Iteration | Date | Scope reviewed | Files written before barrier | Updates made | IWU/score before | IWU/score after | Split decision | Split artifacts | Child re-review status | Next scope after readback |
|---:|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-07-16 | Parent split creation | this spec and paired test spec | Created focused no-hidden-mesh proof leaf | 3 IWU parent | 1 IWU | no split | none | pending ledger review | public integration spec |

## Implementation Routing

- Primary modules/files:
  - `src/impression/modeling/csg.py` - accepted route proof fields.
  - `src/impression/modeling/tessellation.py` - negative assertion boundary if needed.
- Supporting modules/files:
  - `tests/test_surface_csg.py` - no-hidden-mesh proof coverage.

## API And Data Contract

- Accepted result evidence must include route kind, source-body kind, construction proof id, and mesh-fallback flag fixed false for this path.
- Tests must be able to assert that no mesh fallback was invoked.

## Required DTOs / Functions / Components

- DTOs/models: no-hidden-mesh proof record and missing-proof diagnostic payload.
- Functions/methods: no-hidden-mesh proof builder and fallback-invocation assertion hook.
- UI components/fields: not applicable.

## Performance Contract

- Proof construction is metadata-only and bounded by evidence record count.

## Error And State Behavior

- Missing construction evidence refuses acceptance instead of silently accepting a mesh-derived result.

## Test Strategy

- Unit tests: proof construction and missing-proof refusal.
- Integrated route tests: public route asserts mesh-fallback flag is false and no mesh fallback hook was invoked.
- Production-data rule: tests must not require production data.

## Review Score

Adversarial review date: 2026-07-16.

- Functions/methods: 1 x 2 = 2
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 1 x 1 = 1
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 2 x 1 = 2
- Performance-sensitive behavior: 0 x 2 = 0
- Test scenarios/fixtures: 3 x 1 = 3
- Readiness blockers: 0 x 2 = 0
- Total: 11.5

Adversarial checks: the proof adds observable evidence and negative fallback assertions only; it does not perform validity, persistence, tessellation, or fixture generation.

Split decision: no split.

## Acceptance Criteria

- Accepted loft/primitive CSG results include no-hidden-mesh proof.
- Missing proof refuses before reference handoff.
- Tests can detect accidental mesh fallback use.
