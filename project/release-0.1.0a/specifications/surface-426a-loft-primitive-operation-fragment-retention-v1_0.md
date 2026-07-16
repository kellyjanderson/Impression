# Surface Spec 426a: Loft Primitive Operation Fragment Retention (v1.0)

Date: 2026-07-16
Status: Proposed
Primary ancestor: `../architecture/acd-loft-primitive-generated-cap-and-topology-policy.md`
Architecture ancestor: `../architecture/acd-loft-primitive-generated-cap-and-topology-policy.md`
Split provenance: `../specifications/surface-426-loft-primitive-fragment-topology-and-operation-selection-v1_0.md`
Canonical status: Canonical leaf
Prerequisites:
- `../specifications/surface-425c-loft-primitive-cap-loop-pairing-and-diagnostics-v1_0.md` - fragment retention requires paired cap loops.

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface.
Count: 1 IWU.
Basis: one retained-fragment chooser for loft/primitive operation classes.

## Purpose

Choose which classified loft and primitive fragments are retained for union, difference, and intersection before topology classification.

## Scope

Owns:

- Operation-specific fragment-retention helper.
- Retained/excluded fragment records with source and operation provenance.
- Deterministic no-fragment and ambiguous-fragment diagnostics.

Does not own:

- Topology class selection.
- Orientation repair.
- Shell assembly.

## Split Coverage

- Parent spec: `../specifications/surface-426-loft-primitive-fragment-topology-and-operation-selection-v1_0.md`
- Parent coverage status: 100% covered by Surface Specs 426a-426c.
- Parent responsibility covered: operation-specific retained-fragment selection.
- Parent responsibilities outside this leaf: topology class classification and orientation/refusal diagnostics.
- Parent responsibilities still missing from children:
  - none.

## Refinement History

| Iteration | Date | Scope reviewed | Files written before barrier | Updates made | IWU/score before | IWU/score after | Split decision | Split artifacts | Child re-review status | Next scope after readback |
|---:|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-07-16 | Parent split creation | this spec and paired test spec | Created focused retained-fragment leaf | 3 IWU parent | 1 IWU | no split | none | pending ledger review | topology sibling specs |

## Implementation Routing

- Primary modules/files:
  - `src/impression/modeling/csg.py` - operation retention records and helper.
- Supporting modules/files:
  - `tests/test_surface_csg.py` - focused helper and route coverage.

## API And Data Contract

- Add or finalize retained-fragment records that identify source body, fragment id, operation role, retained/excluded state, and reason.
- Expose diagnostics through existing CSG result/evidence records.

## Required DTOs / Functions / Components

- DTOs/models: retained-fragment record and operation-retention diagnostic payload.
- Functions/methods: operation fragment retention helper.
- UI components/fields: not applicable.

## Performance Contract

- Bounded by classified fragment count.
- No tessellation or mesh fallback is permitted.

## Error And State Behavior

- Empty, conflicting, or stale fragment classifications return deterministic diagnostics before topology classification.

## Test Strategy

- Unit tests: retained-fragment decisions for union, difference, and intersection.
- Integrated route tests: public Boolean route exposes retention diagnostics before topology classification.
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

Adversarial checks: three Boolean operations are cases of one retention policy, while topology classification and orientation diagnostics remain separate specs.

Split decision: no split.

## Acceptance Criteria

- Each supported operation produces deterministic retained/excluded fragment records.
- Ambiguous or empty retention refuses before topology classification.
- Retention records preserve source identity and operation provenance.
