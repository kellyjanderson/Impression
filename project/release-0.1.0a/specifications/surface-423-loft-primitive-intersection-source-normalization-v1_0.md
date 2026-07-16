# Surface Spec 423: Loft Primitive Intersection Source Normalization (v1.0)

Date: 2026-07-16
Status: Proposed
Primary ancestor: `../architecture/acd-loft-primitive-intersection-and-cut-loop-kernel.md`
Architecture ancestor: `../architecture/acd-loft-primitive-intersection-and-cut-loop-kernel.md`
Manifest source: `Loft Primitive Intersection Source Normalization`
Split provenance: `../specifications/surface-422-loft-primitive-cut-shell-assembly-and-validity-v1_0.md`
Canonical status: Canonical
Prerequisites:
- `../specifications/surface-421-loft-primitive-trim-fragment-adapter-v1_0.md` - adapter records must exist before source normalization can consume loft/primitive classification evidence.

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one source-normalization boundary that converts adapter evidence into supported primitive source-region records or deterministic diagnostics.

## Manifest Field Carryover

- Discovery purpose:
  - Normalize primitive/loft source evidence before patch-local cut-loop construction.
- Manifest responsibilities by category:
  - Functions/methods: source normalizer, primitive region classifier, unsupported-region diagnostic builder.
  - Data structures/models: `LoftPrimitiveIntersectionSourceRecord`, primitive source region, unsupported source diagnostic.
  - Dependencies/services: Surface Spec 421 adapter records, primitive family classification, CSG tolerance policy.
  - Returns/outputs/signals: normalized source records, unsupported-region diagnostics, no-hidden-mesh proof.
  - UI surfaces/components: not applicable.
  - UI fields/elements: not applicable.
  - Reusable code plan: reuse primitive family classification and CSG tolerances; add source normalization to `src/impression/modeling/csg.py`; no new reusable module.
  - Database queries/tables/migrations: none.
  - Async/concurrency behavior: none.
  - Destructive/write behavior: none.
  - Security/privacy-sensitive behavior: none.
  - Performance-sensitive behavior: bounded by primitive source-region and loft patch counts.
  - Cross-screen reusable behavior: not applicable.
- Manifest open questions / nuance discovered:
  - Sphere and cylinder may produce analytic regions rather than planar face records; this spec defaults to explicit support for box, sphere, and cylinder source classification, with unsupported analytic regions refused through diagnostics until represented by a later cap/topology spec.
- Manifest score at promotion:
  - 19 on 2026-07-16.
- Manifest readiness blockers and resolution:
  - none; resolution status resolved in the manifest.
- Manifest split decision:
  - Review for split; remains cohesive as one source-normalization stage.
- Manifest cleanup state:
  - ready after spec promotion; ACD-local manifest may remain until the ACD closes.

## Purpose

Provide the first executable kernel stage for loft/primitive cuts by converting trim-fragment adapter evidence into normalized primitive source-region records.

## Scope

Owns:

- `LoftPrimitiveIntersectionSourceRecord`.
- Supported primitive source-region classification for box, sphere, and cylinder operands.
- Unsupported-region diagnostics and no-hidden-mesh proof.

Does not own:

- Patch-local curve inversion or loop closure.
- Generated caps, fragment topology, shell assembly, validity gates, or reference artifact generation.

## Split Coverage

- Parent spec: `../specifications/surface-422-loft-primitive-cut-shell-assembly-and-validity-v1_0.md`
- Parent coverage status: 100% covered by Surface Specs 423-431; this leaf owns source normalization.
- Parent responsibilities owned by this child:
  - primitive/loft source evidence normalization.
  - supported/unsupported primitive source-region diagnostics.
- Parent responsibilities still missing from children:
  - none.

## Refinement History

| Iteration | Date | Scope reviewed | Files written before barrier | Updates made | IWU/score before | IWU/score after | Split decision | Split artifacts | Child re-review status | Next scope after readback |
|---:|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-07-16 | Manifest promotion and sizing | this spec and paired test spec | Promoted manifest candidate, preserved carryover fields, set IWU to 1 | score 19 | 1 IWU | no split | none | not applicable | fixed-point ledger readback |

## Implementation Routing

- Primary modules/files:
  - `src/impression/modeling/csg.py` - source records, source normalizer, unsupported diagnostics.
- Supporting modules/files:
  - `src/impression/modeling/surface.py` - primitive patch metadata and tolerance-facing structures.
- GUI/QML files, if applicable:
  - not applicable.
- Reusable library/module files:
  - `src/impression/modeling/csg.py` - reusable source-normalization helper consumed by later loft primitive CSG stages.
- Tests:
  - `tests/test_surface_csg.py` - source-record and public-route diagnostic coverage.

## Chosen Defaults / Parameters

- Box, sphere, and cylinder source regions are considered by this stage.
- Unsupported regions refuse with structured diagnostics rather than falling back to tessellation or mesh execution.
- CSG tolerance policy is inherited from the public boolean route.

## Data Ownership

- Source of truth: CSG source-normalization records.
- Read ownership: cut-loop construction reads normalized source records.
- Write ownership: `src/impression/modeling/csg.py` writes the records.
- Derived/cache data: records are recomputable from Surface Spec 421 adapter evidence.
- Privacy/logging constraints: diagnostics must not dump full geometry payloads.

## Dependencies And Routes

- Domain/service dependencies:
  - Surface Spec 421 trim-fragment adapter records.
  - primitive family classification.
  - CSG tolerance policy.
- Database dependencies:
  - not applicable.
- GUI route, if applicable:
  - not applicable.
- Background/concurrency route, if applicable:
  - not applicable.

## Prerequisite Handling

- Already implemented prerequisites:
  - `../specifications/surface-421-loft-primitive-trim-fragment-adapter-v1_0.md` - adapter record production.
- Missing prerequisite architecture:
  - none.
- Missing prerequisite specifications:
  - none.
- Unimplemented prerequisite specifications:
  - none.
- Progression handling:
  - first leaf in the cut-shell kernel sequence.

## Application Integration

- App type: library-only
- User/caller surface: public Boolean API consumers
- Invocation route: public Boolean route to loft adapter to source normalizer
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: source records or unsupported diagnostics
- Integration validation: source-record tests plus public-route diagnostics for supported and unsupported primitive source regions
- Incomplete status risk: helper records could exist while public loft CSG still reports generic adapter-only refusal

App-type-specific proof:

- Library-only: public Boolean API diagnostics prove the consuming route reaches the normalizer.

## Reuse And Extraction Plan

- Existing code to reuse:
  - primitive family classification and CSG tolerance policy.
- Current reuse readiness:
  - add to existing library/module.
- Extraction/wrapping needed:
  - none.
- Additions to existing library/modules:
  - `src/impression/modeling/csg.py` - `LoftPrimitiveIntersectionSourceRecord` and source normalizer.
- New reusable modules to expose:
  - none.
- One-off code justification, if any:
  - none.

## Required DTOs / Functions / Components

- DTOs/models:
  - `LoftPrimitiveIntersectionSourceRecord` - normalized primitive/loft source-region evidence.
  - `LoftPrimitiveUnsupportedSourceDiagnostic` - deterministic unsupported-region refusal.
- Functions/methods:
  - `normalize_loft_primitive_intersection_sources(...) -> LoftPrimitiveIntersectionSourceRecord | Diagnostic`.
  - `classify_primitive_source_regions(...) -> Sequence[PrimitiveSourceRegion]`.
- UI fields / visible data, if applicable:
  - not applicable.
- UI elements / controls, if applicable:
  - not applicable.
- UI components, if applicable:
  - not applicable.

## Performance Contract

- Bounded by primitive source-region and loft patch counts.
- No tessellation, rasterization, or mesh fallback is permitted.

## Error And State Behavior

- Unsupported source regions return deterministic diagnostics.
- Missing adapter evidence refuses before cut-loop construction.
- Diagnostics preserve no-hidden-mesh proof.

## Test Strategy

- Unit tests:
  - supported box, sphere, and cylinder source-region records.
  - unsupported-region diagnostics.
- Service/DB tests:
  - not applicable.
- GUI/controller tests, if applicable:
  - not applicable.
- Integrated route tests:
  - public Boolean route produces source diagnostics for unsupported cases.
- Production-data rule:
  - Tests must not require production data.

## Acceptance Criteria

- Supported primitive source regions produce normalized records through `src/impression/modeling/csg.py`.
- Unsupported primitive regions refuse with structured diagnostics and no-hidden-mesh evidence.
- Public loft/primitive CSG reaches the normalizer instead of returning generic adapter-only refusal for this stage.

## Readiness Checklist

- [x] Primary ancestor and architecture ancestor are explicit.
- [x] Work Units count and basis are explicit.
- [x] Manifest fields are carried into spec sections or preserved as explicit provenance/history.
- [x] Canonical status is explicit.
- [x] Prerequisites are linked, implemented, or marked not applicable.
- [x] Missing prerequisite architecture has an ACD link, or is marked not applicable.
- [x] Missing prerequisite behavior has a final spec link, or is marked not applicable.
- [x] Split coverage is complete, or marked not applicable.
- [x] Refinement history records the latest completed review/update/rescore/split iteration and the files written before its write barrier.
- [x] Implementation owner/module is named.
- [x] Existing code reuse/extraction decision is explicit.
- [x] Existing library/module additions or new reusable module boundaries are named, or marked not applicable.
- [x] UI fields/elements are listed, or marked not applicable.
- [x] Chosen defaults are explicit.
- [x] Data source of truth and write owner are explicit.
- [x] GUI/concurrency route is explicit, or marked not applicable.
- [x] App type and application integration route are explicit.
- [x] Integrated route validation is named.
- [x] GUI/console/API-service/mixed/library-only proof matches the app type.
- [x] Performance bounds are explicit, or marked not applicable.
- [x] Privacy/logging constraints are explicit, or marked not applicable.
- [x] Test strategy does not depend on production data.
- [x] Acceptance criteria are testable.
