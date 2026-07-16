# Surface Spec 428b: Loft Primitive Persistence And Tessellation Readiness (v1.0)

Date: 2026-07-16
Status: Proposed
Primary ancestor: `../architecture/acd-loft-primitive-seam-shell-validity-execution.md`
Architecture ancestor: `../architecture/acd-loft-primitive-seam-shell-validity-execution.md`
Split provenance: `../specifications/surface-428-loft-primitive-runtime-validity-and-persistence-gate-v1_0.md`
Canonical status: Canonical leaf
Prerequisites:
- `../specifications/surface-428a-loft-primitive-runtime-validity-checker-v1_0.md` - persistence readiness consumes runtime-valid shells.

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface.
Count: 1 IWU.
Basis: one persistence/readiness gate for accepted shells.

## Purpose

Persist accepted loft/primitive CSG surface-body results and expose tessellation readiness without tessellating during acceptance.

## Scope

Owns:

- Accepted-result persistence gate for runtime-valid shells.
- Tessellation-readiness metadata for downstream STL/reference generation.
- Refusal for invalid or non-ready shells.

Does not own:

- Runtime validity checks.
- Tessellation execution.
- No-hidden-mesh proof.

## Split Coverage

- Parent spec: `../specifications/surface-428-loft-primitive-runtime-validity-and-persistence-gate-v1_0.md`
- Parent coverage status: 100% covered by Surface Specs 428a-428c.
- Parent responsibility covered: persistence and tessellation readiness.
- Parent responsibilities outside this leaf: runtime validity and no-hidden-mesh proof.
- Parent responsibilities still missing from children:
  - none.

## Refinement History

| Iteration | Date | Scope reviewed | Files written before barrier | Updates made | IWU/score before | IWU/score after | Split decision | Split artifacts | Child re-review status | Next scope after readback |
|---:|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-07-16 | Parent split creation | this spec and paired test spec | Created focused persistence/readiness leaf | 3 IWU parent | 1 IWU | no split | none | pending ledger review | no-hidden-mesh sibling spec |

## Implementation Routing

- Primary modules/files:
  - `src/impression/modeling/csg.py` - accepted-result persistence gate.
  - `src/impression/modeling/tessellation.py` - readiness contract if needed.
- Supporting modules/files:
  - `tests/test_surface_csg.py` - persistence readiness coverage.

## API And Data Contract

- Accepted result records must include persisted surface body, validity evidence id, topology evidence id, and tessellation-readiness state.
- Tessellation-readiness state must not require eager mesh creation.

## Required DTOs / Functions / Components

- DTOs/models: accepted-result persistence record and tessellation-readiness metadata record.
- Functions/methods: accepted-result persistence gate and metadata-only readiness helper.
- UI components/fields: not applicable.

## Performance Contract

- Bounded by accepted shell metadata and body size.
- No eager STL or mesh export is permitted in this gate.

## Error And State Behavior

- Invalid, stale, or non-ready shells refuse before persistence.

## Test Strategy

- Unit tests: valid persistence, stale evidence refusal, non-ready refusal.
- Integrated route tests: public route returns accepted surface body and readiness metadata.
- Production-data rule: tests must not require production data.

## Review Score

Adversarial review date: 2026-07-16.

- Functions/methods: 1 x 2 = 2
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 2 x 1 = 2
- Performance-sensitive behavior: 1 x 2 = 2
- Test scenarios/fixtures: 3 x 1 = 3
- Readiness blockers: 0 x 2 = 0
- Total: 15.5

Adversarial checks: this is the highest-risk leaf in the split because persistence and tessellation-readiness could drift into separate artifacts. It remains one IWU only because readiness is constrained to metadata on the accepted-result persistence gate; tessellation execution, STL export, and hidden-mesh proof are explicitly out of scope.

Split decision: no split, with implementation caution to reject any scope expansion into eager tessellation.

## Acceptance Criteria

- Runtime-valid shells can be persisted as accepted surface bodies.
- Invalid or stale shells refuse before persistence.
- Tessellation readiness is metadata-only at this stage.
