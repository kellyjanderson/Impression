# Surface Spec 424a: Loft Patch-Local Source Curve Inversion (v1.0)

Date: 2026-07-16
Status: Proposed
Primary ancestor: `../architecture/acd-loft-primitive-intersection-and-cut-loop-kernel.md`
Architecture ancestor: `../architecture/acd-loft-primitive-intersection-and-cut-loop-kernel.md`
Manifest source: `Loft Patch-Local Cut Loop Construction`
Split provenance: `../specifications/surface-424-loft-patch-local-cut-loop-construction-v1_0.md`
Canonical status: Canonical leaf
Prerequisites:
- `../specifications/surface-423-loft-primitive-intersection-source-normalization-v1_0.md` - normalized source records are the inversion input.

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one inversion step from normalized primitive source curves into loft patch parameter domains.

## Purpose

Map supported primitive/loft source curves into patch-local curve segments with residual and tolerance diagnostics.

## Scope

Owns:
- patch-local curve inversion records
- source-to-patch parameter mapping diagnostics
- tolerance residual reporting

Does not own:
- loop closure
- degeneracy classification
- cap construction

## Split Coverage

- Parent spec: `../specifications/surface-424-loft-patch-local-cut-loop-construction-v1_0.md`
- Parent coverage status: 100% covered by Surface Specs 424a-424c.
- Parent responsibilities owned by this child:
  - patch-local source curve inversion.
- Parent responsibilities still missing from children:
  - none.

## Refinement History

| Iteration | Date | Scope reviewed | Files written before barrier | Updates made | IWU/score before | IWU/score after | Split decision | Split artifacts | Child re-review status | Next scope after readback |
|---:|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-07-16 | Parent split | this spec and paired test spec | Created child spec from Surface Spec 424 | parent 3 IWU | 1 IWU | no split | none | not applicable | ledger readback |

## Implementation Routing

- Primary modules/files:
  - `src/impression/modeling/csg.py` - patch-local inversion helper and records.
- Supporting modules/files:
  - `src/impression/modeling/loft.py` - loft patch domains.
- Tests:
  - `tests/test_surface_csg.py` - source curve inversion and residual diagnostics.

## Chosen Defaults / Parameters

- Inversion uses the existing CSG tolerance policy.
- Failed inversion returns diagnostics instead of approximating through tessellation.

## Data Ownership

- Source of truth: patch-local inversion records owned by CSG.
- Read ownership: loop closure consumes inversion records.
- Write ownership: `src/impression/modeling/csg.py`.
- Privacy/logging constraints: diagnostics avoid raw geometry dumps.

## Dependencies And Routes

- Domain/service dependencies: Surface Spec 423 source records and loft patch domains.
- Database dependencies: not applicable.
- GUI route, if applicable: not applicable.
- Background/concurrency route, if applicable: not applicable.

## Application Integration

- App type: library-only
- User/caller surface: public Boolean API consumers
- Invocation route: source normalizer to patch-local inversion through the loft CSG route
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: patch-local curve segments or inversion diagnostics
- Integration validation: unit inversion tests plus public-route inversion diagnostics
- Incomplete status risk: later loop closure could consume fabricated patch-local curves

## Required DTOs / Functions / Components

- DTOs/models:
  - `LoftPatchLocalSourceCurveRecord` - source curve mapped to one loft patch domain.
  - `LoftPatchLocalInversionDiagnostic` - failed inversion or residual payload.
- Functions/methods:
  - `invert_loft_primitive_source_curves_to_patch_domains(...)`.
- UI fields / visible data, UI elements, and UI components:
  - not applicable.

## Performance Contract

- Bounded by source curve and affected patch count.
- No mesh or tessellation fallback is permitted.

## Error And State Behavior

- Missing source records or failed inversion return deterministic diagnostics.

## Test Strategy

- Unit tests: successful inversion and failed inversion diagnostics.
- Integrated route tests: public route reaches inversion diagnostics.
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

Adversarial checks: loop closure, degeneracy classification, and cap construction are explicitly excluded, so the leaf remains one implementation outcome.

Split decision: no split.

## Acceptance Criteria

- Supported source curves produce patch-local records.
- Failed inversions refuse deterministically.
- No tessellated approximation is accepted as inversion output.
