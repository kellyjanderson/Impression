# Surface Spec 426b: Loft Primitive Result Topology Classification (v1.0)

Date: 2026-07-16
Status: Proposed
Primary ancestor: `../architecture/acd-loft-primitive-generated-cap-and-topology-policy.md`
Architecture ancestor: `../architecture/acd-loft-primitive-generated-cap-and-topology-policy.md`
Split provenance: `../specifications/surface-426-loft-primitive-fragment-topology-and-operation-selection-v1_0.md`
Canonical status: Canonical leaf
Prerequisites:
- `../specifications/surface-426a-loft-primitive-operation-fragment-retention-v1_0.md` - topology classification consumes retained fragments.

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface.
Count: 1 IWU.
Basis: one topology classifier for retained loft/primitive fragments.

## Purpose

Classify retained fragments into expected result topology classes before shell assembly.

## Scope

Owns:

- `LoftPrimitiveFragmentTopologyRecord` or equivalent topology-class record.
- Empty, exterior-shell edit, interior-cavity, multi-shell, and refused topology classes.
- Deterministic topology-class evidence for the public route.

Does not own:

- Fragment-retention rules.
- Orientation diagnostics.
- Shell assembly.

## Split Coverage

- Parent spec: `../specifications/surface-426-loft-primitive-fragment-topology-and-operation-selection-v1_0.md`
- Parent coverage status: 100% covered by Surface Specs 426a-426c.
- Parent responsibility covered: result topology classification.
- Parent responsibilities outside this leaf: retained-fragment selection and orientation/refusal diagnostics.
- Parent responsibilities still missing from children:
  - none.

## Refinement History

| Iteration | Date | Scope reviewed | Files written before barrier | Updates made | IWU/score before | IWU/score after | Split decision | Split artifacts | Child re-review status | Next scope after readback |
|---:|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-07-16 | Parent split creation | this spec and paired test spec | Created focused topology-classification leaf | 3 IWU parent | 1 IWU | no split | none | pending ledger review | orientation sibling spec |

## Implementation Routing

- Primary modules/files:
  - `src/impression/modeling/csg.py` - topology classifier and records.
- Supporting modules/files:
  - `tests/test_surface_csg.py` - topology-class coverage.

## API And Data Contract

- Topology records must include operation, retained fragment ids, generated cap ids, topology class, and assembly readiness.
- Topology class must be observable from public Boolean result diagnostics.

## Required DTOs / Functions / Components

- DTOs/models: topology-class record and refused-topology diagnostic payload.
- Functions/methods: retained-fragment topology classifier.
- UI components/fields: not applicable.

## Performance Contract

- Bounded by retained fragment and generated cap counts.
- No mesh fallback is permitted.

## Error And State Behavior

- Unsupported or inconsistent retained-fragment sets refuse before shell assembly.

## Test Strategy

- Unit tests: empty, exterior edit, interior cavity, multi-shell, and refused classes.
- Integrated route tests: public route exposes topology class before assembly.
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
- Test scenarios/fixtures: 4 x 1 = 4
- Readiness blockers: 0 x 2 = 0
- Total: 13.5

Adversarial checks: multiple topology classes are outputs of one classifier; fragment retention and shell assembly remain outside the leaf.

Split decision: no split.

## Acceptance Criteria

- Supported retained-fragment sets produce deterministic topology classes.
- Unsupported topology refuses with explicit evidence.
- Topology records do not assemble or persist geometry.
