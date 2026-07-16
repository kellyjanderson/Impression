# Surface Spec 413: File Fixture Evidence Bundle Schema (v1.0)

Date: 2026-07-15
Status: Proposed
Primary ancestor: `../architecture/acd-reference-fixture-multi-artifact-section-evidence-policy.md`
Architecture ancestor: `../architecture/acd-reference-fixture-multi-artifact-section-evidence-policy.md`
Manifest source: `File Fixture Evidence Bundle Schema`
Split provenance: `Reference Evidence Bundle Schema`
Canonical status: Canonical

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one file fixture evidence bundle load contract.

## Purpose

Define typed evidence bundles for JSON fixture files while preserving existing
single-artifact fixture records.

## Scope

Owns:

- `ReferenceEvidenceBundleRecord`.
- `ReferenceEvidenceArtifactRecord`.
- File fixture evidence bundle parsing and artifact path validation.

Does not own:

- Database-backed fixture parity; see Surface Spec 414.
- Review UI display; see Surface Specs 415 and 416.

## Split Coverage

- Parent spec: `Reference Evidence Bundle Schema`
- Parent coverage status: 100% covered with Surface Specs 413-416.
- Parent responsibilities owned by this child:
  - file fixture schema and validation.
- Parent responsibilities still missing from children: none.

## Implementation Routing

- Primary modules/files:
  - `src/impression/devtools/reference_review/source_registry.py` - parser and records.
- Supporting modules/files:
  - `tests/reference_review_fixtures/stl_review_sources.py` - fixture examples.
- GUI/QML files, if applicable:
  - not applicable.
- Reusable library/module files:
  - `src/impression/devtools/reference_review/source_registry.py` - evidence records.
- Tests:
  - `tests/test_reference_review_sources.py` - loader validation and compatibility.

## Chosen Defaults / Parameters

- Existing `artifact_paths` remains the default for one-artifact fixtures.
- Typed `evidence_bundles` is additive.
- Artifact paths must stay inside allowed roots.

## Data Ownership

- Source of truth: fixture JSON file.
- Read ownership: source registry reads and exposes parsed records.
- Write ownership: fixture generation workflows write fixture file contents.
- Derived/cache data: parsed source records are derived from fixture files.
- Privacy/logging constraints: only sanitized artifact paths should appear in diagnostics.

## Dependencies And Routes

- Domain/service dependencies:
  - fixture source registry.
  - existing artifact path validation.
- Database dependencies:
  - not applicable.
- GUI route, if applicable:
  - not applicable.
- Background/concurrency route, if applicable:
  - not applicable.

## Application Integration

- App type: workflow
- User/caller surface: reference fixture file loader
- Invocation route: fixture JSON file load to source registry record construction
- Wiring owner/module: `src/impression/devtools/reference_review/source_registry.py`
- Observable result: file-backed fixtures expose typed evidence bundles without
  breaking existing one-artifact fixtures
- Integration validation: fixture loader tests for valid bundles, missing roles,
  bad paths, and simple `artifact_paths` compatibility
- Incomplete status risk: implemented in isolation if parsed bundles are not
  exposed on fixture records consumed by review and promotion routes

App-type-specific proof:

- Workflow: fixture loader tests validate read behavior and parsed record shape.

## Reuse And Extraction Plan

- Existing code to reuse:
  - fixture loading and artifact path validation.
- Current reuse readiness:
  - add to existing library/module.
- Extraction/wrapping needed:
  - none.
- Additions to existing library/modules:
  - file fixture evidence bundle records.
- New reusable modules to expose:
  - none.
- One-off code justification, if any:
  - none.

## Required DTOs / Functions / Components

- DTOs/models:
  - `ReferenceEvidenceBundleRecord` - bundle id, evidence kind, role policy, artifacts.
  - `ReferenceEvidenceArtifactRecord` - role, kind, path, stage, required flag.
- Functions/methods:
  - `parse_file_evidence_bundles(...) -> list[ReferenceEvidenceBundleRecord]`
  - `validate_evidence_artifact_path(...) -> Diagnostic | None`
- UI fields / visible data, if applicable:
  - not applicable.
- UI elements / controls, if applicable:
  - not applicable.
- UI components, if applicable:
  - not applicable.

## Performance Contract

- Bounded by fixture artifact count.
- Fixture load must not open artifact payloads unless validation requires path
  existence checks already supported by the loader.

## Error And State Behavior

- Invalid bundle roles or paths produce loader diagnostics.
- Simple `artifact_paths` remains valid.
- Missing optional artifacts do not fail required bundle validation.

## Test Strategy

- Unit tests:
  - valid bundles, missing roles, invalid paths, compatibility with `artifact_paths`.
- Service/DB tests:
  - not applicable.
- GUI/controller tests, if applicable:
  - not applicable.
- Integrated route tests:
  - fixture file to source record loader tests.
- Production-data rule:
  - Tests must not require production data.

## Acceptance Criteria

- File fixtures can declare typed evidence bundles.
- Existing simple fixture records still load.
- Invalid artifact paths are diagnosed deterministically.

## Rescore And Split Review

- Manifest score: 17.
- IWU count: 1.
- Readiness blockers: none.
- Split decision: Review for split; parsing and path validation are one file-load route.
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
- Pass 2 - Routing: Confirmed file schema is independent from DB and UI.
- Pass 3 - Rescore: Manifest score 17; review-for-split band retained.
- Pass 4 - Split Review: File parsing and path validation remain one load contract.
- Pass 5 - Final: Ready as a final leaf specification.
