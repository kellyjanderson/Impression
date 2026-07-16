# Surface Spec 419: Fixture Registry Integration For Section Bundles (v1.0)

Date: 2026-07-15
Status: Proposed
Primary ancestor: `../architecture/acd-reference-fixture-multi-artifact-section-evidence-policy.md`
Architecture ancestor: `../architecture/acd-reference-fixture-multi-artifact-section-evidence-policy.md`
Manifest source: `Fixture Registry Integration for Section Bundles`
Split provenance: `Section Evidence Bundle Producer`
Canonical status: Canonical

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one fixture registry integration route for generated section bundles.

## Purpose

Attach generated section evidence artifacts to fixture records so review and
promotion workflows consume them through the standard bundle schema.

## Scope

Owns:

- Section bundle fixture record builder.
- Dirty/gold evidence path resolver.
- Generated artifact to fixture registry integration.

Does not own:

- Section artifact generation; see Surface Spec 418.
- Review UI metadata display; see Surface Specs 415 and 416.

## Split Coverage

- Parent spec: `Section Evidence Bundle Producer`
- Parent coverage status: 100% covered with Surface Specs 417-419.
- Parent responsibilities owned by this child:
  - fixture registry integration for generated section bundles.
- Parent responsibilities still missing from children: none.

## Implementation Routing

- Primary modules/files:
  - `tests/reference_review_fixtures/stl_review_sources.py` - section bundle fixture builder.
  - `src/impression/devtools/reference_review/source_registry.py` - loader integration.
- Supporting modules/files:
  - Surface Spec 413 evidence bundle records.
  - Surface Spec 418 generated artifacts.
- GUI/QML files, if applicable:
  - not applicable.
- Reusable library/module files:
  - section fixture record builder in existing fixture generation helpers.
- Tests:
  - `tests/test_reference_review_sources.py` - fixture load integration.
  - `tests/test_reference_stl_generation.py` - generation/load integration.

## Chosen Defaults / Parameters

- Dirty and gold paths mirror section evidence role names.
- Generated section artifacts appear as typed evidence bundles.
- Promotion semantics remain owned by reference artifact lifecycle architecture.

## Data Ownership

- Source of truth: fixture record evidence references.
- Read ownership: source registry and review app read fixture records.
- Write ownership: fixture generation writes section bundle references.
- Derived/cache data: fixture records derive from generated artifact paths.
- Privacy/logging constraints: generated paths remain under dirty/gold roots.

## Dependencies And Routes

- Domain/service dependencies:
  - Surface Spec 417 section evidence contract records.
  - Surface Spec 418 section artifact generation.
  - fixture source registry.
- Database dependencies:
  - not applicable.
- GUI route, if applicable:
  - not applicable.
- Background/concurrency route, if applicable:
  - not applicable.

## Application Integration

- App type: workflow
- User/caller surface: reference fixture generation and review app fixture loading
- Invocation route: generated section artifacts to fixture record to source registry
- Wiring owner/module: `tests/reference_review_fixtures/stl_review_sources.py`,
  `src/impression/devtools/reference_review/source_registry.py`
- Observable result: generated section artifacts appear as typed evidence
  bundles on the corresponding fixture
- Integration validation: fixture generation test followed by fixture loader validation
- Incomplete status risk: implemented in isolation if generated files exist but
  fixture records do not reference them

App-type-specific proof:

- Workflow: fixture generation/load integration tests validate the route.

## Reuse And Extraction Plan

- Existing code to reuse:
  - Surface Spec 413 file fixture evidence bundle schema.
- Current reuse readiness:
  - add to existing library/module.
- Extraction/wrapping needed:
  - none.
- Additions to existing library/modules:
  - section fixture record builder.
- New reusable modules to expose:
  - none.
- One-off code justification, if any:
  - none.

## Required DTOs / Functions / Components

- DTOs/models:
  - fixture section evidence source record.
- Functions/methods:
  - `build_section_bundle_fixture_record(...) -> ReferenceEvidenceBundleRecord`
  - `resolve_dirty_gold_evidence_paths(...) -> EvidencePathSet`
- UI fields / visible data, if applicable:
  - not applicable.
- UI elements / controls, if applicable:
  - not applicable.
- UI components, if applicable:
  - not applicable.

## Performance Contract

- Bounded by generated artifact count.
- Fixture registry integration must not read artifact payloads.

## Error And State Behavior

- Missing generated artifacts produce fixture diagnostics.
- Dirty/gold path mismatches refuse fixture registration.
- Partial required bundles are not treated as promotion-ready.

## Test Strategy

- Unit tests:
  - path resolver and fixture record builder.
- Service/DB tests:
  - not applicable.
- GUI/controller tests, if applicable:
  - not applicable.
- Integrated route tests:
  - fixture generation followed by fixture loader validation.
- Production-data rule:
  - Tests must not require production data.

## Acceptance Criteria

- Generated section artifacts are referenced by typed fixture evidence bundles.
- Fixture loader validates generated section bundles.
- Review and promotion workflows can consume bundle records through the standard source registry.

## Rescore And Split Review

- Manifest score: 19.5.
- IWU count: 1.
- Readiness blockers: none.
- Split decision: Review for split; record construction and path resolution are one registry integration route.
- Review update: checked after promotion from ACD manifest; no child spec is required before implementation.

## Readiness Checklist

- [x] Primary ancestor and architecture ancestor are explicit.
- [x] Canonical status is explicit.
- [x] Split coverage is complete, or marked not applicable.
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

## Review History

- Pass 1 - Template: Promoted from split manifest parent.
- Pass 2 - Routing: Confirmed registry integration is separate from artifact generation.
- Pass 3 - Rescore: Manifest score 19.5; review-for-split band retained.
- Pass 4 - Split Review: Record construction and path resolution remain one integration route.
- Pass 5 - Final: Ready as a final leaf specification after Surface Specs 417 and 418.
