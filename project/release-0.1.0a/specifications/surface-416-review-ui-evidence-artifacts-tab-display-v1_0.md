# Surface Spec 416: Review UI Evidence Artifacts Tab Display (v1.0)

Date: 2026-07-15
Status: Proposed
Primary ancestor: `../architecture/acd-reference-fixture-multi-artifact-section-evidence-policy.md`
Architecture ancestor: `../architecture/acd-reference-fixture-multi-artifact-section-evidence-policy.md`
Manifest source: `Review UI Evidence Artifacts Tab Display`
Split provenance: `Review UI Evidence Display Contract`
Canonical status: Canonical

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one artifacts tab evidence metadata display route.

## Purpose

Show evidence artifact details in the selected fixture artifacts tab.

## Scope

Owns:

- Artifact evidence list mapping.
- Missing artifact status formatting.
- Role, kind, and path/status display in the artifacts tab.

Does not own:

- Context summary display; see Surface Spec 415.
- Opening non-STL artifact previews.

## Split Coverage

- Parent spec: `Review UI Evidence Display Contract`
- Parent coverage status: 100% covered with Surface Specs 415 and 416.
- Parent responsibilities owned by this child:
  - artifacts tab detail display.
- Parent responsibilities still missing from children: none.

## Implementation Routing

- Primary modules/files:
  - `src/impression/devtools/reference_review/ui/shell.py` - artifacts display.
- Supporting modules/files:
  - `src/impression/devtools/reference_review/source_registry.py` - bundle data.
- GUI/QML files, if applicable:
  - `src/impression/devtools/reference_review/ui/shell.py` - current review UI shell.
- Reusable library/module files:
  - artifact evidence list helper inside existing review UI module.
- Tests:
  - `tests/test_reference_review_ui.py` - view-model/smoke tests where available.

## Chosen Defaults / Parameters

- Artifacts tab lists metadata only until preview behavior is specified.
- Artifact paths are sanitized before display.
- Missing artifacts show a status row rather than breaking selection.

## Data Ownership

- Source of truth: source registry bundle records.
- Read ownership: UI reads selected fixture source record.
- Write ownership: UI does not write fixture bundle data.
- Derived/cache data: artifact display rows derive from selected fixture state.
- Privacy/logging constraints: display sanitized artifact paths.

## Dependencies And Routes

- Domain/service dependencies:
  - Surface Spec 413 file fixture schema.
  - fixture source registry.
- Database dependencies:
  - not applicable.
- GUI route, if applicable:
  - fixture selection change to artifacts tab render.
- Background/concurrency route, if applicable:
  - not applicable.

## Application Integration

- App type: GUI
- User/caller surface: reference review app artifacts tab
- Invocation route: fixture selection change to artifacts tab render
- Wiring owner/module: `src/impression/devtools/reference_review/ui/shell.py`
- Observable result: selected fixture artifacts tab lists evidence artifacts
  with role, kind, and path/status
- Integration validation: review-app smoke test or view-model test for a
  bundled fixture
- Incomplete status risk: implemented in isolation if helper output is not wired
  to fixture selection

App-type-specific proof:

- GUI: selected fixture route renders artifact evidence rows.

## Reuse And Extraction Plan

- Existing code to reuse:
  - selected fixture artifacts state.
- Current reuse readiness:
  - add to existing library/module.
- Extraction/wrapping needed:
  - none.
- Additions to existing library/modules:
  - artifact evidence list helper.
- New reusable modules to expose:
  - none.
- One-off code justification, if any:
  - none.

## Required DTOs / Functions / Components

- DTOs/models:
  - artifact evidence row display model.
- Functions/methods:
  - `map_artifact_evidence_rows(...) -> list[ArtifactEvidenceRow]`
  - `format_missing_artifact_status(...) -> str`
- UI fields / visible data, if applicable:
  - artifact role.
  - artifact kind.
  - artifact path/status.
- UI elements / controls, if applicable:
  - artifacts tab evidence list.
- UI components, if applicable:
  - existing artifacts tab component.

## Performance Contract

- Bounded by selected fixture artifact count.
- Artifact payloads are not opened during metadata display.

## Error And State Behavior

- Missing artifacts show status rows.
- Invalid bundle diagnostics originate from source registry.
- Empty bundle lists preserve existing artifacts tab behavior.

## Test Strategy

- Unit tests:
  - row mapping and missing artifact status formatting.
- Service/DB tests:
  - not applicable.
- GUI/controller tests, if applicable:
  - review-app smoke or offscreen view-model test.
- Integrated route tests:
  - fixture selection to artifacts tab render.
- Production-data rule:
  - Tests must not require production data.

## Acceptance Criteria

- Artifacts tab displays evidence role, kind, and path/status rows.
- Missing artifacts are visible but do not break selection.
- Display wiring follows selected fixture changes.

## Rescore And Split Review

- Manifest score: 22.5.
- IWU count: 1.
- Readiness blockers: none.
- Split decision: Review for split; artifact metadata listing is one GUI display route and preview opening is out of scope.
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
- Pass 2 - Routing: Confirmed artifacts detail is separate from context summary.
- Pass 3 - Rescore: Manifest score 22.5; review-for-split band retained.
- Pass 4 - Split Review: Metadata list remains one display route; preview opening is out of scope.
- Pass 5 - Final: Ready as a final leaf specification after Surface Spec 413.
