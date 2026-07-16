# Surface Spec 425a: Loft Primitive Cap Support Classification (v1.0)

Date: 2026-07-16
Status: Proposed
Primary ancestor: `../architecture/acd-loft-primitive-generated-cap-and-topology-policy.md`
Architecture ancestor: `../architecture/acd-loft-primitive-generated-cap-and-topology-policy.md`
Manifest source: `Loft Primitive Generated Cap Construction`
Split provenance: `../specifications/surface-425-loft-primitive-generated-cap-construction-v1_0.md`
Canonical status: Canonical leaf
Prerequisites:
- `../specifications/surface-424c-loft-cut-loop-degeneracy-diagnostics-v1_0.md` - only non-degenerate loops are classified for cap support.

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one support/refusal classification for primitive cap regions.

## Purpose

Classify which primitive cap regions can be generated surface-natively and which must refuse.

## Scope

Owns:
- cap support classification
- explicit sphere/cylinder support or refusal decisions
- unsupported cap diagnostics

Does not own:
- generated cap record construction
- cap-loop pairing
- shell topology

## Split Coverage

- Parent spec: `../specifications/surface-425-loft-primitive-generated-cap-construction-v1_0.md`
- Parent coverage status: 100% covered by Surface Specs 425a-425c.
- Parent responsibilities owned by this child:
  - supported/unsupported cap policy.
- Parent responsibilities still missing from children:
  - none.

## Refinement History

| Iteration | Date | Scope reviewed | Files written before barrier | Updates made | IWU/score before | IWU/score after | Split decision | Split artifacts | Child re-review status | Next scope after readback |
|---:|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-07-16 | Parent split | this spec and paired test spec | Created child spec from Surface Spec 425 | parent 3 IWU | 1 IWU | no split | none | not applicable | ledger readback |

## Implementation Routing

- Primary modules/files:
  - `src/impression/modeling/csg.py` - cap support classifier.
- Tests:
  - `tests/test_surface_csg.py` - supported and unsupported cap classification.

## Chosen Defaults / Parameters

- Box planar caps are supported.
- Sphere and cylinder cap regions are supported only when they can be represented by existing surface-native patch families; otherwise they refuse.
- Refusal is structured and never falls back to tessellation.

## Data Ownership

- Source of truth: CSG cap support classification.
- Read ownership: generated cap record construction consumes support decisions.
- Write ownership: `src/impression/modeling/csg.py`.
- Privacy/logging constraints: diagnostics avoid raw geometry dumps.

## Dependencies And Routes

- Domain/service dependencies: non-degenerate cut loops and primitive source records.
- Database dependencies: not applicable.
- GUI route, if applicable: not applicable.
- Background/concurrency route, if applicable: not applicable.

## Application Integration

- App type: library-only
- User/caller surface: public Boolean API consumers
- Invocation route: non-degenerate loops to cap support classifier through the loft CSG route
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: support classification or unsupported cap diagnostic
- Integration validation: public-route unsupported-cap diagnostics
- Incomplete status risk: cap construction could silently approximate unsupported regions

## Required DTOs / Functions / Components

- DTOs/models:
  - `LoftPrimitiveCapSupportClassification` - supported/refused cap support record.
  - `LoftPrimitiveUnsupportedCapDiagnostic` - unsupported cap payload.
- Functions/methods:
  - `classify_loft_primitive_cap_support(...)`.
- UI fields / visible data, UI elements, and UI components:
  - not applicable.

## Performance Contract

- Bounded by cap loop and primitive source-region count.

## Error And State Behavior

- Unsupported caps refuse before cap record construction.

## Test Strategy

- Unit tests: supported box cap, supported surface-native sphere/cylinder cap if available, unsupported cap refusal.
- Integrated route tests: public-route unsupported-cap diagnostics.
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

Adversarial checks: sphere/cylinder handling is support/refusal classification only; construction of supported cap geometry is explicitly delegated to Surface Spec 425b.

Split decision: no split.

## Acceptance Criteria

- Cap support is explicit before construction.
- Unsupported sphere/cylinder representation issues are not blockers hidden inside cap construction.
