# Surface Spec 425: Loft Primitive Generated Cap Construction (v1.0)

Date: 2026-07-16
Status: Superseded
Primary ancestor: `../architecture/acd-loft-primitive-generated-cap-and-topology-policy.md`
Architecture ancestor: `../architecture/acd-loft-primitive-generated-cap-and-topology-policy.md`
Manifest source: `Loft Primitive Generated Cap Construction`
Split provenance: `../specifications/surface-422-loft-primitive-cut-shell-assembly-and-validity-v1_0.md`
Canonical status: Superseded parent
Prerequisites:
- `../specifications/surface-424-loft-patch-local-cut-loop-construction-v1_0.md` - cap construction consumes closed patch-local cut loops.

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 3 IWU.
Basis: parent branch split into cap support classification, generated cap record construction, and cap pairing/refusal diagnostics.

## Manifest Field Carryover

- Discovery purpose: construct primitive cap records needed to close loft/primitive cut results.
- Manifest responsibilities by category:
  - Functions/methods: generated cap builder, unsupported cap classifier, cap-loop pairing diagnostic.
  - Data structures/models: `LoftPrimitiveGeneratedCapRecord`, generated cap loop, unsupported cap diagnostic.
  - Dependencies/services: cut loops, primitive source records, cap policy helpers.
  - Returns/outputs/signals: generated cap records or unsupported cap diagnostics.
  - UI surfaces/components and fields: not applicable.
  - Reusable code plan: reuse cap policy and add loft cap pairing records in `src/impression/modeling/csg.py`; no new module.
  - Database/async/security/cross-screen behavior: none.
  - Destructive/write behavior: none.
  - Performance-sensitive behavior: bounded by generated cap loop count.
- Manifest open questions / nuance discovered:
  - Sphere and cylinder support may need implicit or higher-order cap representation; this spec requires explicit support/refusal policy rather than silent approximation.
- Manifest score at promotion: 19 on 2026-07-16.
- Manifest readiness blockers and resolution: none; resolved.
- Manifest split decision: review for split; cohesive generated-cap construction boundary.
- Manifest cleanup state: ready after spec promotion.

## Purpose

Create generated primitive cap records that allow later topology and shell assembly stages to close supported loft/primitive cuts.

## Scope

Owns:

- `LoftPrimitiveGeneratedCapRecord`.
- Supported generated cap records for initial box/sphere/cylinder cases.
- Unsupported analytic cap diagnostics.

Does not own:

- Cut-loop construction.
- Fragment selection and operation topology.
- Seam/shell assembly or public executor wiring.

## Split Coverage

- Parent spec: `../specifications/surface-422-loft-primitive-cut-shell-assembly-and-validity-v1_0.md`
- Parent coverage status: 100% covered by Surface Specs 425a-425c.
- Parent responsibilities owned by this child:
  - none; superseded parent only.
- Parent responsibilities still missing from children:
  - none.
  - Cap support/refusal policy is covered by `surface-425a-loft-primitive-cap-support-classification-v1_0.md`.
  - Generated cap record construction is covered by `surface-425b-loft-primitive-generated-cap-record-construction-v1_0.md`.
  - Cap-loop pairing and unsupported diagnostics are covered by `surface-425c-loft-primitive-cap-loop-pairing-and-diagnostics-v1_0.md`.

## Refinement History

| Iteration | Date | Scope reviewed | Files written before barrier | Updates made | IWU/score before | IWU/score after | Split decision | Split artifacts | Child re-review status | Next scope after readback |
|---:|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-07-16 | Manifest promotion and sizing | this spec and paired test spec | Promoted manifest candidate and set IWU to 1 | score 19 | 1 IWU | no split | none | not applicable | fixed-point ledger readback |
| 2 | 2026-07-16 | Critical sizing review | this parent and child specs 425a-425c | Marked parent superseded and moved all executable work to children | 1 IWU | 3 IWU branch rollup | split | 425a, 425b, 425c | complete | child specs |

## Implementation Routing

- Primary modules/files:
  - `src/impression/modeling/csg.py` - generated cap records, cap builder, unsupported-cap diagnostics.
- Supporting modules/files:
  - `src/impression/modeling/surface.py` - patch/cap construction primitives.
- GUI/QML files, if applicable:
  - not applicable.
- Reusable library/module files:
  - `src/impression/modeling/csg.py` - cap builder consumed by topology selection.
- Tests:
  - `tests/test_surface_csg.py` - cap construction and unsupported cap refusal tests.

## Chosen Defaults / Parameters

- Unsupported analytic cap regions refuse until represented by supported patch families.
- Generated caps must retain source identity and cap-loop pairing evidence.
- No tessellation-derived cap may be treated as result geometry.

## Data Ownership

- Source of truth: generated cap records owned by CSG.
- Read ownership: topology selection and shell assembly consume cap records.
- Write ownership: `src/impression/modeling/csg.py`.
- Derived/cache data: recomputable from cut loops and source records.
- Privacy/logging constraints: diagnostics must avoid full geometry dumps.

## Dependencies And Routes

- Domain/service dependencies:
  - Surface Spec 424 cut-loop records.
  - primitive source records.
  - surface cap helpers.
- Database dependencies: not applicable.
- GUI route, if applicable: not applicable.
- Background/concurrency route, if applicable: not applicable.

## Prerequisite Handling

- Already implemented prerequisites:
  - none.
- Missing prerequisite architecture: none.
- Missing prerequisite specifications: none.
- Unimplemented prerequisite specifications:
  - `../specifications/surface-424-loft-patch-local-cut-loop-construction-v1_0.md` - must be implemented first.
- Progression handling:
  - sequence after Surface Spec 424.

## Application Integration

- App type: library-only
- User/caller surface: public Boolean API consumers
- Invocation route: cut-loop records to generated cap builder through the loft CSG route
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: generated cap records or unsupported cap diagnostics
- Integration validation: cap-builder tests plus public-route unsupported-cap diagnostics
- Incomplete status risk: shell assembly could invent caps or silently ignore unsupported primitive source regions

App-type-specific proof:

- Library-only: public Boolean API diagnostics prove the cap builder is reachable.

## Reuse And Extraction Plan

- Existing code to reuse:
  - cap policy and surface patch construction helpers.
- Current reuse readiness:
  - add to existing library/module.
- Extraction/wrapping needed:
  - none.
- Additions to existing library/modules:
  - `src/impression/modeling/csg.py` - generated cap records and builder.
- New reusable modules to expose:
  - none.
- One-off code justification, if any:
  - none.

## Required DTOs / Functions / Components

- DTOs/models:
  - `LoftPrimitiveGeneratedCapRecord` - cap geometry, source identity, cap-loop pairing, diagnostics.
  - `LoftPrimitiveUnsupportedCapDiagnostic` - unsupported cap refusal payload.
- Functions/methods:
  - `build_loft_primitive_generated_caps(...) -> Sequence[LoftPrimitiveGeneratedCapRecord] | Diagnostic`.
  - `classify_supported_loft_primitive_caps(...) -> CapSupportClassification`.
- UI fields / visible data, UI elements, and UI components:
  - not applicable.

## Performance Contract

- Bounded by generated cap loop count.
- No hidden mesh, raster, or tessellation fallback is permitted.

## Error And State Behavior

- Unsupported cap families return deterministic diagnostics.
- Missing or unpaired cap loops refuse before topology selection.

## Test Strategy

- Unit tests:
  - supported cap construction, unsupported cap refusal, cap-loop pairing.
- Service/DB tests: not applicable.
- GUI/controller tests, if applicable: not applicable.
- Integrated route tests:
  - public-route unsupported-cap diagnostics.
- Production-data rule:
  - Tests must not require production data.

## Acceptance Criteria

- Supported cut loops produce generated cap records with source identity and pairing evidence.
- Unsupported cap cases refuse with structured diagnostics.
- Generated caps are surface-native records, not tessellation products.

## Readiness Checklist

- [x] Primary ancestor and architecture ancestor are explicit.
- [x] Work Units count and basis are explicit.
- [x] Manifest fields are carried into spec sections or preserved as explicit provenance/history.
- [x] Canonical status is explicit.
- [x] Prerequisites are linked, implemented, or marked not applicable.
- [x] Split coverage is complete, or marked not applicable.
- [x] Refinement history records the latest completed review/update/rescore/split iteration and the files written before its write barrier.
- [x] Implementation owner/module is named.
- [x] Existing code reuse/extraction decision is explicit.
- [x] Existing library/module additions or new reusable module boundaries are named, or marked not applicable.
- [x] UI fields/elements are listed, or marked not applicable.
- [x] Chosen defaults are explicit.
- [x] Data source of truth and write owner are explicit.
- [x] App type and application integration route are explicit.
- [x] Integrated route validation is named.
- [x] Performance bounds are explicit, or marked not applicable.
- [x] Privacy/logging constraints are explicit, or marked not applicable.
- [x] Test strategy does not depend on production data.
- [x] Acceptance criteria are testable.
