# Surface Spec 414: Database Fixture Evidence Bundle Parity (v1.0)

Date: 2026-07-15
Status: Proposed
Primary ancestor: `../architecture/acd-reference-fixture-multi-artifact-section-evidence-policy.md`
Architecture ancestor: `../architecture/acd-reference-fixture-multi-artifact-section-evidence-policy.md`
Manifest source: `Database Fixture Evidence Bundle Parity`
Split provenance: `Reference Evidence Bundle Schema`
Canonical status: Canonical

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one database fixture evidence bundle parity contract.

## Purpose

Add database-backed fixture support for the same evidence bundle shape used by
file fixtures.

## Scope

Owns:

- Database evidence bundle loading and serialization.
- Database schema parity diagnostics.
- Migration or write-path updates for evidence bundle persistence.

Does not own:

- File fixture schema; see Surface Spec 413.
- Review UI display; see Surface Specs 415 and 416.

## Split Coverage

- Parent spec: `Reference Evidence Bundle Schema`
- Parent coverage status: 100% covered with Surface Specs 413-416.
- Parent responsibilities owned by this child:
  - database fixture evidence bundle parity.
- Parent responsibilities still missing from children: none.

## Implementation Routing

- Primary modules/files:
  - `src/impression/devtools/reference_review/source_registry.py` - database mapper.
- Supporting modules/files:
  - fixture database adapter, if active.
- GUI/QML files, if applicable:
  - not applicable.
- Reusable library/module files:
  - `src/impression/devtools/reference_review/source_registry.py` - DB parity mapper.
- Tests:
  - `tests/test_reference_review_sources.py` - DB parity tests when DB fixtures are active.

## Chosen Defaults / Parameters

- Database fields mirror file bundle semantics.
- Existing database fixture records remain valid.
- File and DB loaders must hydrate the same source record shape.

## Data Ownership

- Source of truth: fixture database rows.
- Read ownership: source registry reads database records.
- Write ownership: fixture database write path owns persistence.
- Derived/cache data: hydrated records are derived from database rows.
- Privacy/logging constraints: stored artifact paths are validated under
  allowed roots.

## Dependencies And Routes

- Domain/service dependencies:
  - Surface Spec 413 file evidence bundle records.
  - fixture database adapter.
- Database dependencies:
  - evidence bundle persistence field or table.
- GUI route, if applicable:
  - not applicable.
- Background/concurrency route, if applicable:
  - not applicable.

## Application Integration

- App type: workflow
- User/caller surface: database-backed reference fixture loader
- Invocation route: fixture database read/write to source registry record construction
- Wiring owner/module: `src/impression/devtools/reference_review/source_registry.py`
- Observable result: database-backed fixtures expose the same evidence bundles
  as file-backed fixtures
- Integration validation: database fixture load/save parity tests
- Incomplete status risk: implemented in isolation if file and database
  fixtures produce divergent source records

App-type-specific proof:

- Workflow: DB fixture load/save parity tests validate persistence behavior.

## Reuse And Extraction Plan

- Existing code to reuse:
  - Surface Spec 413 evidence bundle records.
- Current reuse readiness:
  - add to existing library/module.
- Extraction/wrapping needed:
  - none.
- Additions to existing library/modules:
  - database fixture mapper.
- New reusable modules to expose:
  - none.
- One-off code justification, if any:
  - none.

## Required DTOs / Functions / Components

- DTOs/models:
  - database evidence bundle row payload.
- Functions/methods:
  - `load_database_evidence_bundles(...) -> list[ReferenceEvidenceBundleRecord]`
  - `serialize_database_evidence_bundles(...) -> DatabasePayload`
- UI fields / visible data, if applicable:
  - not applicable.
- UI elements / controls, if applicable:
  - not applicable.
- UI components, if applicable:
  - not applicable.

## Performance Contract

- Bounded by fixture artifact count.
- Database loading must avoid per-artifact unbounded queries.

## Error And State Behavior

- Schema mismatches produce parity diagnostics.
- Missing optional bundle data does not break legacy rows.
- Invalid stored paths refuse hydration with diagnostics.

## Test Strategy

- Unit tests:
  - mapper serialization/deserialization.
- Service/DB tests:
  - migration/parity tests when database fixtures are active.
- GUI/controller tests, if applicable:
  - not applicable.
- Integrated route tests:
  - database fixture to source record.
- Production-data rule:
  - Tests must not require production data.

## Acceptance Criteria

- Database fixtures hydrate the same bundle records as file fixtures.
- Existing DB records remain compatible.
- Path and schema parity diagnostics are deterministic.

## Rescore And Split Review

- Manifest score: 20.5.
- IWU count: 1.
- Readiness blockers: none.
- Split decision: Review for split; DB load, save, and migration must prove parity together.
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
- Pass 2 - Routing: Confirmed DB parity is independent from file schema implementation.
- Pass 3 - Rescore: Manifest score 20.5; review-for-split band retained.
- Pass 4 - Split Review: Load, save, and migration remain one parity contract.
- Pass 5 - Final: Ready as a final leaf specification after Surface Spec 413.
