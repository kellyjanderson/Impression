# Surface Spec 425b: Loft Primitive Generated Cap Record Construction (v1.0)

Date: 2026-07-16
Status: Proposed
Primary ancestor: `../architecture/acd-loft-primitive-generated-cap-and-topology-policy.md`
Architecture ancestor: `../architecture/acd-loft-primitive-generated-cap-and-topology-policy.md`
Manifest source: `Loft Primitive Generated Cap Construction`
Split provenance: `../specifications/surface-425-loft-primitive-generated-cap-construction-v1_0.md`
Canonical status: Canonical leaf
Prerequisites:
- `../specifications/surface-425a-loft-primitive-cap-support-classification-v1_0.md` - construction consumes supported cap classifications.

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one cap record construction step for already-supported cap classifications.

## Purpose

Create generated cap records for supported primitive cap regions without deciding support or pairing policy.

## Scope

Owns:
- `LoftPrimitiveGeneratedCapRecord`
- generated cap geometry and source identity payload
- cap construction diagnostics for supported regions

Does not own:
- support/refusal classification
- cap-loop pairing completeness
- topology selection

## Split Coverage

- Parent spec: `../specifications/surface-425-loft-primitive-generated-cap-construction-v1_0.md`
- Parent coverage status: 100% covered by Surface Specs 425a-425c.
- Parent responsibilities owned by this child:
  - generated cap record construction.
- Parent responsibilities still missing from children:
  - none.

## Refinement History

| Iteration | Date | Scope reviewed | Files written before barrier | Updates made | IWU/score before | IWU/score after | Split decision | Split artifacts | Child re-review status | Next scope after readback |
|---:|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-07-16 | Parent split | this spec and paired test spec | Created child spec from Surface Spec 425 | parent 3 IWU | 1 IWU | no split | none | not applicable | ledger readback |

## Implementation Routing

- Primary modules/files:
  - `src/impression/modeling/csg.py` - generated cap record builder.
- Supporting modules/files:
  - `src/impression/modeling/surface.py` - surface-native cap primitives.
- Tests:
  - `tests/test_surface_csg.py` - generated cap record payload tests.

## Chosen Defaults / Parameters

- Only supported cap classifications are accepted.
- Cap records preserve source identity and construction provenance.

## Data Ownership

- Source of truth: generated cap records owned by CSG.
- Read ownership: cap-loop pairing consumes generated cap records.
- Write ownership: `src/impression/modeling/csg.py`.
- Privacy/logging constraints: diagnostics avoid raw geometry dumps.

## Dependencies And Routes

- Domain/service dependencies: Surface Spec 425a support records and surface-native patch helpers.
- Database dependencies: not applicable.
- GUI route, if applicable: not applicable.
- Background/concurrency route, if applicable: not applicable.

## Application Integration

- App type: library-only
- User/caller surface: public Boolean API consumers
- Invocation route: cap support classifier to generated cap builder through the loft CSG route
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: generated cap records or supported-cap construction diagnostics
- Integration validation: cap record tests plus public-route cap construction proof
- Incomplete status risk: topology selection could consume cap records without provenance

## Required DTOs / Functions / Components

- DTOs/models:
  - `LoftPrimitiveGeneratedCapRecord` - cap geometry, source identity, construction provenance.
- Functions/methods:
  - `build_loft_primitive_generated_cap_records(...)`.
- UI fields / visible data, UI elements, and UI components:
  - not applicable.

## Performance Contract

- Bounded by supported cap count.

## Error And State Behavior

- Missing support classification refuses before construction.

## Test Strategy

- Unit tests: generated cap records for supported caps.
- Integrated route tests: public route reaches generated cap record evidence.
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

Adversarial checks: this leaf constructs records only after support classification; pairing completeness and topology readiness are not part of this deliverable.

Split decision: no split.

## Acceptance Criteria

- Supported cap classifications produce source-native generated cap records.
- Generated cap records preserve source identity and provenance.
