# Surface Spec 417: Section Evidence Contract Records (v1.0)

Date: 2026-07-15
Status: Proposed
Primary ancestor: `../architecture/acd-reference-fixture-multi-artifact-section-evidence-policy.md`
Architecture ancestor: `../architecture/acd-reference-fixture-multi-artifact-section-evidence-policy.md`
Manifest source: `Section Evidence Contract Records`
Split provenance: `Section Evidence Bundle Producer`
Canonical status: Canonical

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one section evidence role contract.

## Purpose

Define the section evidence role contract used by loft CSG reference fixtures.

## Scope

Owns:

- `SectionEvidenceContractRecord`.
- Required `expected`, `actual`, and `diff` role validation.
- Section plane metadata validation.

Does not own:

- Artifact generation; see Surface Spec 418.
- Fixture registry integration; see Surface Spec 419.

## Split Coverage

- Parent spec: `Section Evidence Bundle Producer`
- Parent coverage status: 100% covered with Surface Specs 417-419.
- Parent responsibilities owned by this child:
  - section evidence role contract.
- Parent responsibilities still missing from children: none.

## Implementation Routing

- Primary modules/files:
  - `src/impression/devtools/reference_review/source_registry.py` - role validation.
  - `tests/reference_review_fixtures/stl_review_sources.py` - section fixture declarations.
- Supporting modules/files:
  - Surface Spec 413 evidence bundle records.
- GUI/QML files, if applicable:
  - not applicable.
- Reusable library/module files:
  - section-specific records in existing fixture evidence contract.
- Tests:
  - `tests/test_reference_review_sources.py` - complete and missing role bundles.

## Chosen Defaults / Parameters

- Required roles are `expected`, `actual`, and `diff`.
- Section plane metadata is required for section evidence bundles.
- Role validation is called during fixture load.

## Data Ownership

- Source of truth: reference fixture evidence declarations.
- Read ownership: source registry reads section bundle roles.
- Write ownership: fixture generation writes declarations.
- Derived/cache data: validation result derives from bundle roles and metadata.
- Privacy/logging constraints: not applicable.

## Dependencies And Routes

- Domain/service dependencies:
  - Surface Spec 413 evidence bundle schema.
- Database dependencies:
  - not applicable.
- GUI route, if applicable:
  - not applicable.
- Background/concurrency route, if applicable:
  - not applicable.

## Application Integration

- App type: workflow
- User/caller surface: reference fixture bundle validation
- Invocation route: evidence bundle validation for section evidence fixtures
- Wiring owner/module: `src/impression/devtools/reference_review/source_registry.py`
- Observable result: section evidence bundles require expected, actual, and diff
  roles with section plane metadata
- Integration validation: fixture loader tests for complete and incomplete
  section bundles
- Incomplete status risk: implemented in isolation if role validation is not
  called by fixture load

App-type-specific proof:

- Workflow: fixture loader tests validate section evidence bundle roles.

## Reuse And Extraction Plan

- Existing code to reuse:
  - Surface Spec 413 file fixture evidence bundle schema.
- Current reuse readiness:
  - add to existing library/module.
- Extraction/wrapping needed:
  - none.
- Additions to existing library/modules:
  - section evidence contract records.
- New reusable modules to expose:
  - none.
- One-off code justification, if any:
  - none.

## Required DTOs / Functions / Components

- DTOs/models:
  - `SectionEvidenceContractRecord` - required roles and section plane metadata.
- Functions/methods:
  - `validate_section_evidence_roles(...) -> SectionEvidenceContractRecord`
- UI fields / visible data, if applicable:
  - not applicable.
- UI elements / controls, if applicable:
  - not applicable.
- UI components, if applicable:
  - not applicable.

## Performance Contract

- Bounded by bundle artifact count.
- Validation must not open artifact payloads.

## Error And State Behavior

- Missing required roles produce fixture diagnostics.
- Unknown optional roles are accepted only if policy allows them.
- Missing plane metadata refuses section bundle validation.

## Test Strategy

- Unit tests:
  - complete bundle, missing role, missing plane metadata.
- Service/DB tests:
  - not applicable.
- GUI/controller tests, if applicable:
  - not applicable.
- Integrated route tests:
  - fixture loader section validation.
- Production-data rule:
  - Tests must not require production data.

## Acceptance Criteria

- Section bundles validate required roles and plane metadata.
- Incomplete bundles produce deterministic diagnostics.
- Surface Spec 418 can generate artifacts against the contract.

## Rescore And Split Review

- Manifest score: 9.5.
- IWU count: 1.
- Readiness blockers: none.
- Split decision: Small; role validation is separate from artifact generation and registry integration.
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
- Pass 2 - Routing: Confirmed role contract is independent from artifact generation.
- Pass 3 - Rescore: Manifest score 9.5; no split required.
- Pass 4 - Blocker Review: File fixture schema sequence is explicit.
- Pass 5 - Final: Ready as a final leaf specification after Surface Spec 413.
