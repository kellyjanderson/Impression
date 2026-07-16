# Surface Spec 426c: Loft Primitive Topology Orientation And Refusal Diagnostics (v1.0)

Date: 2026-07-16
Status: Proposed
Primary ancestor: `../architecture/acd-loft-primitive-generated-cap-and-topology-policy.md`
Architecture ancestor: `../architecture/acd-loft-primitive-generated-cap-and-topology-policy.md`
Split provenance: `../specifications/surface-426-loft-primitive-fragment-topology-and-operation-selection-v1_0.md`
Canonical status: Canonical leaf
Prerequisites:
- `../specifications/surface-426b-loft-primitive-result-topology-classification-v1_0.md` - orientation diagnostics consume topology class records.

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface.
Count: 1 IWU.
Basis: one diagnostic gate for topology orientation and refusal reasons.

## Purpose

Validate that topology records carry enough orientation evidence to reach shell assembly, or refuse with specific diagnostics.

## Scope

Owns:

- Orientation-readiness diagnostics for selected topology classes.
- Refusal reasons for ambiguous inside/outside, inverted source normals, and cap-orientation conflicts.
- Public-route diagnostic payloads for refused topology.

Does not own:

- Topology class creation.
- Seam/use pairing.
- Runtime shell validity checks.

## Split Coverage

- Parent spec: `../specifications/surface-426-loft-primitive-fragment-topology-and-operation-selection-v1_0.md`
- Parent coverage status: 100% covered by Surface Specs 426a-426c.
- Parent responsibility covered: orientation and refusal diagnostics.
- Parent responsibilities outside this leaf: retention and topology classification.
- Parent responsibilities still missing from children:
  - none.

## Refinement History

| Iteration | Date | Scope reviewed | Files written before barrier | Updates made | IWU/score before | IWU/score after | Split decision | Split artifacts | Child re-review status | Next scope after readback |
|---:|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-07-16 | Parent split creation | this spec and paired test spec | Created focused orientation/refusal leaf | 3 IWU parent | 1 IWU | no split | none | pending ledger review | seam assembly specs |

## Implementation Routing

- Primary modules/files:
  - `src/impression/modeling/csg.py` - orientation diagnostic gate.
- Supporting modules/files:
  - `tests/test_surface_csg.py` - refused orientation cases.

## API And Data Contract

- Add a diagnostic record or extend topology records with orientation readiness, blocking reason, and affected fragment/cap ids.
- Diagnostics must be returned before shell assembly is attempted.

## Required DTOs / Functions / Components

- DTOs/models: orientation-readiness diagnostic record.
- Functions/methods: topology orientation readiness/refusal gate.
- UI components/fields: not applicable.

## Performance Contract

- Bounded by topology record, fragment, and cap counts.
- No tessellation fallback is permitted.

## Error And State Behavior

- Ambiguous or contradictory orientation evidence refuses deterministically.

## Test Strategy

- Unit tests: inverted source normal, cap conflict, ambiguous inside/outside, and ready topology.
- Integrated route tests: public route exposes orientation refusal.
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

Adversarial checks: orientation-ready and refused states are one diagnostic gate; topology creation, pairing, and shell assembly remain out of scope.

Split decision: no split.

## Acceptance Criteria

- Orientation-ready topologies can proceed to seam/use pairing.
- Ambiguous topologies refuse with precise diagnostics.
- No shell assembly occurs after an orientation refusal.
