# Surface Spec 430: Loft CSG Reference Geometry Handoff Proof (v1.0)

Date: 2026-07-16
Status: Proposed
Primary ancestor: `../architecture/acd-loft-csg-reference-geometry-handoff.md`
Architecture ancestor: `../architecture/acd-loft-csg-reference-geometry-handoff.md`
Manifest source: `Loft CSG Reference Geometry Handoff Proof`
Split provenance: `../specifications/surface-407-loft-csg-result-geometry-reference-proof-v1_0.md`
Canonical status: Canonical
Prerequisites:
- `../specifications/surface-429-loft-primitive-public-cut-executor-integration-v1_0.md` - reference handoff requires accepted public CSG result geometry.

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one workflow gate that allows dirty STL generation only from accepted public result geometry.

## Manifest Field Carryover

- Discovery purpose: gate dirty STL generation for loft CSG references on accepted public result geometry.
- Manifest responsibilities by category:
  - Functions/methods: loft CSG reference handoff builder, accepted-result validator, adapter/synthetic refusal diagnostic builder.
  - Data structures/models: `LoftCsgReferenceGeometryHandoffRecord`, accepted result body reference, reference handoff refusal payload.
  - Dependencies/services: public `SurfaceBooleanResult`, fixture source registry, STL export workflow.
  - Returns/outputs/signals: accepted handoff record, structured refusal, dirty STL source readiness signal.
  - UI surfaces/components and fields: not applicable.
  - Reusable code plan: reuse fixture source registry and STL export workflow; add loft CSG accepted-result gate.
  - Database/async/security/cross-screen behavior: none.
  - Destructive/write behavior: dirty reference artifact creation through existing reference workflow.
  - Performance-sensitive behavior: no duplicate CSG execution for metadata extraction.
- Manifest open questions / nuance discovered:
  - Final spec should name exact result fields; this spec requires `status == succeeded`, non-null body, and surface-native body identity.
- Manifest score at promotion: 22 on 2026-07-16.
- Manifest readiness blockers and resolution: none; resolved.
- Manifest split decision: review for split; cohesive dirty-STL handoff gate.
- Manifest cleanup state: ready after spec promotion.

## Purpose

Ensure loft CSG dirty STL references are generated only from accepted public CSG result geometry, never from adapter diagnostics, synthetic geometry, sampled preview data, or tessellation buffers.

## Scope

Owns:

- `LoftCsgReferenceGeometryHandoffRecord`.
- Accepted-result validation for dirty STL source readiness.
- Structured refusal for adapter-only, synthetic, sampled, or tessellated payloads.

Does not own:

- Public CSG execution.
- Section evidence generation.
- Review UI approval behavior.

## Split Coverage

- Parent spec: `../specifications/surface-407-loft-csg-result-geometry-reference-proof-v1_0.md`
- Parent coverage status: 100% covered for the result-geometry handoff path by this spec.
- Parent responsibilities owned by this child:
  - accepted public result geometry gate for dirty STL reference generation.
- Parent responsibilities still missing from children:
  - none.

## Refinement History

| Iteration | Date | Scope reviewed | Files written before barrier | Updates made | IWU/score before | IWU/score after | Split decision | Split artifacts | Child re-review status | Next scope after readback |
|---:|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-07-16 | Manifest promotion and sizing | this spec and paired test spec | Promoted manifest candidate and set IWU to 1 | score 22 | 1 IWU | no split | none | not applicable | fixed-point ledger readback |

## Implementation Routing

- Primary modules/files:
  - `tests/reference_review_fixtures/stl_review_sources.py` - accepted-result handoff and refusal payloads.
- Supporting modules/files:
  - `tests/reference_images.py` - existing artifact workflow integration where needed.
- GUI/QML files, if applicable:
  - not applicable.
- Reusable library/module files:
  - `tests/reference_review_fixtures/stl_review_sources.py` - handoff gate reused by loft CSG fixture sources.
- Tests:
  - `tests/test_reference_stl_expansion.py` and `tests/test_reference_review_source_registry.py`.

## Chosen Defaults / Parameters

- Require `SurfaceBooleanResult.status == succeeded`.
- Require non-null, surface-native result body.
- Reject adapter-only diagnostics, synthetic geometry, sampled preview data, tessellation buffers, and mesh execution payloads.

## Data Ownership

- Source of truth: accepted public CSG result body.
- Read ownership: fixture source registry reads accepted handoff records.
- Write ownership: existing reference workflow writes dirty STL artifacts after the gate passes.
- Derived/cache data: dirty STL artifacts are derived from accepted result geometry.
- Privacy/logging constraints: refusal diagnostics avoid full geometry dumps.

## Dependencies And Routes

- Domain/service dependencies:
  - Surface Spec 429 public accepted CSG result.
  - fixture source registry.
  - STL export workflow.
- Database dependencies: not applicable.
- GUI route, if applicable: not applicable.
- Background/concurrency route, if applicable: not applicable.

## Prerequisite Handling

- Already implemented prerequisites: none.
- Missing prerequisite architecture: none.
- Missing prerequisite specifications: none.
- Unimplemented prerequisite specifications:
  - `../specifications/surface-429-loft-primitive-public-cut-executor-integration-v1_0.md` - must be implemented first.
- Progression handling:
  - sequence after Surface Spec 429 and before reference STL generation work.

## Application Integration

- App type: workflow/tooling
- User/caller surface: dirty STL reference generation workflow and review fixtures
- Invocation route: fixture source registry to public CSG result to STL export
- Wiring owner/module: `tests/reference_review_fixtures/stl_review_sources.py`
- Observable result: dirty STL generated only from accepted public result body
- Integration validation: accepted handoff test, adapter-only refusal test, no-synthetic-geometry test, dirty artifact smoke
- Incomplete status risk: fixture records could generate dirty STLs from adapter diagnostics or synthetic geometry

App-type-specific proof:

- Mixed/workflow proof: fixture registry integration and dirty artifact smoke must pass; helper-only tests are insufficient.

## Reuse And Extraction Plan

- Existing code to reuse:
  - fixture source registry and STL export workflow.
- Current reuse readiness:
  - add to existing workflow module.
- Extraction/wrapping needed:
  - none.
- Additions to existing library/modules:
  - `tests/reference_review_fixtures/stl_review_sources.py` - accepted-result handoff gate.
- New reusable modules to expose:
  - none.
- One-off code justification, if any:
  - none.

## Required DTOs / Functions / Components

- DTOs/models:
  - `LoftCsgReferenceGeometryHandoffRecord` - fixture id, operation id, source path, accepted body identity, metadata, refusal diagnostics.
- Functions/methods:
  - `build_loft_csg_reference_geometry_handoff(...) -> LoftCsgReferenceGeometryHandoffRecord | Diagnostic`.
  - `validate_loft_csg_reference_result_body(...) -> None | Diagnostic`.
- UI fields / visible data, UI elements, and UI components:
  - not applicable.

## Performance Contract

- No duplicate CSG execution for metadata extraction.
- Artifact generation runs through existing bounded reference workflow.

## Error And State Behavior

- Non-success result, missing body, adapter-only payload, synthetic payload, sampled preview data, and tessellation buffers refuse before artifact writing.

## Test Strategy

- Unit tests:
  - accepted-result validator and refusal payloads.
- Service/DB tests: not applicable.
- GUI/controller tests, if applicable: not applicable.
- Integrated route tests:
  - fixture registry to STL export workflow smoke.
- Production-data rule:
  - Tests must not require production data.

## Acceptance Criteria

- Dirty STL generation requires accepted public loft CSG result geometry.
- Adapter-only, synthetic, sampled, tessellated, or mesh payloads refuse before artifact creation.
- Handoff records expose enough metadata for review fixtures without embedding generated artifact paths.

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
