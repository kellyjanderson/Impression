# Surface Spec 431: Loft CSG Section Evidence Readiness Handoff (v1.0)

Date: 2026-07-16
Status: Proposed
Primary ancestor: `../architecture/acd-loft-csg-reference-geometry-handoff.md`
Architecture ancestor: `../architecture/acd-loft-csg-reference-geometry-handoff.md`
Manifest source: `Loft CSG Section Evidence Readiness Handoff`
Split provenance: `../specifications/surface-418-loft-csg-section-artifact-generation-v1_0.md`
Canonical status: Canonical
Prerequisites:
- `../specifications/surface-430-loft-csg-reference-geometry-handoff-proof-v1_0.md` - section evidence readiness consumes accepted result handoff records.
- `../specifications/surface-417-section-evidence-contract-records-v1_0.md` - section evidence contract records define the bundle shape.
- `../specifications/surface-419-fixture-registry-integration-for-section-bundles-v1_0.md` - fixture registry integration defines section bundle registration.

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one section-evidence readiness handoff gate from accepted result geometry to section bundle generation.

## Manifest Field Carryover

- Discovery purpose: gate section artifact generation on accepted loft CSG result geometry and declared section evidence inputs.
- Manifest responsibilities by category:
  - Functions/methods: section evidence handoff builder, section readiness validator, section refusal diagnostic builder.
  - Data structures/models: `LoftCsgSectionEvidenceReadinessRecord`, section plane declaration, section bundle readiness payload.
  - Dependencies/services: public accepted CSG result body, section evidence contract records, reference fixture registry.
  - Returns/outputs/signals: section evidence readiness record, structured refusal, fixture registry payload.
  - UI surfaces/components and fields: not applicable.
  - Reusable code plan: reuse section evidence records and fixture registry; add loft section readiness handoff.
  - Database/async/security/cross-screen behavior: none.
  - Destructive/write behavior: section artifact creation through existing reference workflow.
  - Performance-sensitive behavior: avoid rerunning loft CSG for each section artifact.
- Manifest open questions / nuance discovered:
  - Section evidence must remain readiness/proof data, not a second source of geometry truth.
- Manifest score at promotion: 22 on 2026-07-16.
- Manifest readiness blockers and resolution: none; resolved.
- Manifest split decision: review for split; cohesive section-evidence handoff gate.
- Manifest cleanup state: ready after spec promotion.

## Purpose

Ensure section evidence artifacts for loft CSG are generated only from accepted result geometry plus declared section evidence inputs.

## Scope

Owns:

- `LoftCsgSectionEvidenceReadinessRecord`.
- Accepted-result and section-plane readiness validation.
- Structured refusal for missing plane, adapter-only payload, or detached evidence.

Does not own:

- Public CSG execution.
- Dirty STL handoff proof.
- Review UI rendering of section artifacts.

## Split Coverage

- Parent spec: `../specifications/surface-418-loft-csg-section-artifact-generation-v1_0.md`
- Parent coverage status: 100% covered for the readiness handoff boundary by this spec.
- Parent responsibilities owned by this child:
  - accepted result geometry and section input readiness gate.
- Parent responsibilities still missing from children:
  - none.

## Refinement History

| Iteration | Date | Scope reviewed | Files written before barrier | Updates made | IWU/score before | IWU/score after | Split decision | Split artifacts | Child re-review status | Next scope after readback |
|---:|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-07-16 | Manifest promotion and sizing | this spec and paired test spec | Promoted manifest candidate and set IWU to 1 | score 22 | 1 IWU | no split | none | not applicable | fixed-point ledger readback |

## Implementation Routing

- Primary modules/files:
  - `tests/reference_images.py` - section readiness handoff and artifact workflow integration.
  - `tests/reference_review_fixtures/stl_review_sources.py` - fixture source registry payload.
- Supporting modules/files:
  - existing section evidence contract and fixture registry helpers.
- GUI/QML files, if applicable:
  - not applicable.
- Reusable library/module files:
  - `tests/reference_images.py` - reusable section readiness gate.
- Tests:
  - `tests/test_reference_stl_expansion.py` and `tests/test_reference_review_source_registry.py`.

## Chosen Defaults / Parameters

- Require accepted result body plus explicit section plane declaration.
- Reject adapter-only, synthetic, or detached section evidence.
- Do not rerun loft CSG once an accepted handoff exists.

## Data Ownership

- Source of truth: accepted CSG result body and section evidence contracts.
- Read ownership: section artifact workflow reads readiness records.
- Write ownership: reference workflow writes section artifacts after readiness passes.
- Derived/cache data: section artifacts are derived from accepted geometry and section declarations.
- Privacy/logging constraints: diagnostics avoid full geometry dumps.

## Dependencies And Routes

- Domain/service dependencies:
  - Surface Spec 430 handoff record.
  - Surface Spec 417 section evidence contracts.
  - Surface Spec 419 fixture registry integration.
- Database dependencies: not applicable.
- GUI route, if applicable: not applicable.
- Background/concurrency route, if applicable: not applicable.

## Prerequisite Handling

- Already implemented prerequisites:
  - `../specifications/surface-417-section-evidence-contract-records-v1_0.md` - section evidence contracts.
  - `../specifications/surface-419-fixture-registry-integration-for-section-bundles-v1_0.md` - fixture registry section bundle integration.
- Missing prerequisite architecture: none.
- Missing prerequisite specifications: none.
- Unimplemented prerequisite specifications:
  - `../specifications/surface-430-loft-csg-reference-geometry-handoff-proof-v1_0.md` - must be implemented first.
- Progression handling:
  - sequence after Surface Spec 430 and before final section artifact generation.

## Application Integration

- App type: workflow/tooling
- User/caller surface: reference section evidence workflow and review fixture artifacts tab
- Invocation route: fixture source registry to accepted CSG result to section evidence bundle
- Wiring owner/module: `tests/reference_images.py`; `tests/reference_review_fixtures/stl_review_sources.py`
- Observable result: section readiness bundle only from accepted result geometry and declared section inputs
- Integration validation: section readiness test, missing plane refusal test, adapter-only refusal test, fixture registry integration test
- Incomplete status risk: section artifacts could be generated from detached evidence rather than accepted result geometry

App-type-specific proof:

- Mixed/workflow proof: section bundle workflow and fixture registry integration must be tested; helper-only tests are insufficient.

## Reuse And Extraction Plan

- Existing code to reuse:
  - section evidence records and fixture registry helpers.
- Current reuse readiness:
  - add to existing workflow modules.
- Extraction/wrapping needed:
  - none.
- Additions to existing library/modules:
  - `tests/reference_images.py` - section readiness handoff.
  - `tests/reference_review_fixtures/stl_review_sources.py` - fixture source registry payload.
- New reusable modules to expose:
  - none.
- One-off code justification, if any:
  - none.

## Required DTOs / Functions / Components

- DTOs/models:
  - `LoftCsgSectionEvidenceReadinessRecord` - accepted body, section plane, readiness bundle, registry payload.
- Functions/methods:
  - `build_loft_csg_section_evidence_handoff(...) -> LoftCsgSectionEvidenceReadinessRecord | Diagnostic`.
  - `validate_loft_csg_section_readiness(...) -> None | Diagnostic`.
- UI fields / visible data, UI elements, and UI components:
  - not applicable.

## Performance Contract

- Must avoid rerunning loft CSG per section artifact.
- Section generation uses existing bounded reference workflow.

## Error And State Behavior

- Missing section plane, adapter-only result, synthetic geometry, or detached evidence refuses before artifact writing.

## Test Strategy

- Unit tests:
  - readiness validator, missing plane refusal, detached evidence refusal.
- Service/DB tests: not applicable.
- GUI/controller tests, if applicable: not applicable.
- Integrated route tests:
  - section evidence bundle and fixture registry integration.
- Production-data rule:
  - Tests must not require production data.

## Acceptance Criteria

- Section readiness requires accepted public result geometry and explicit section inputs.
- Adapter-only or detached evidence refuses.
- Section bundles remain proof/readiness artifacts, not alternate geometry truth.

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
