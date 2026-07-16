# Surface Spec 415: Review UI Evidence Context Tab Display (v1.0)

Date: 2026-07-15
Status: Proposed
Primary ancestor: `../architecture/acd-reference-fixture-multi-artifact-section-evidence-policy.md`
Architecture ancestor: `../architecture/acd-reference-fixture-multi-artifact-section-evidence-policy.md`
Manifest source: `Review UI Evidence Context Tab Display`
Split provenance: `Review UI Evidence Display Contract`
Canonical status: Canonical

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one context tab evidence summary display route.

## Purpose

Show evidence bundle summary metadata in the selected fixture context tab.

## Scope

Owns:

- Context evidence summary view-model mapping.
- Evidence bundle labels and artifact role summary display.
- Context-tab wiring on fixture selection.

Does not own:

- Artifacts tab details; see Surface Spec 416.
- File fixture schema; see Surface Spec 413.

## Split Coverage

- Parent spec: `Review UI Evidence Display Contract`
- Parent coverage status: 100% covered with Surface Specs 415 and 416.
- Parent responsibilities owned by this child:
  - context tab summary display.
- Parent responsibilities still missing from children: none.

## Implementation Routing

- Primary modules/files:
  - `src/impression/devtools/reference_review/ui/shell.py` - context display.
- Supporting modules/files:
  - `src/impression/devtools/reference_review/source_registry.py` - bundle data.
- GUI/QML files, if applicable:
  - `src/impression/devtools/reference_review/ui/shell.py` - current review UI shell.
- Reusable library/module files:
  - context evidence summary helper inside existing review UI module.
- Tests:
  - `tests/test_reference_review_ui.py` - view-model/smoke tests where available.

## Chosen Defaults / Parameters

- Context tab shows summary only, not full artifact paths.
- Specialized side-by-side previews are out of scope.
- Labels must be sanitized before display.

## Data Ownership

- Source of truth: source registry bundle records.
- Read ownership: UI reads selected fixture source record.
- Write ownership: UI does not write fixture bundle data.
- Derived/cache data: context display model derives from selected fixture state.
- Privacy/logging constraints: display sanitized labels only.

## Dependencies And Routes

- Domain/service dependencies:
  - Surface Spec 413 file fixture schema.
  - fixture source registry.
- Database dependencies:
  - not applicable.
- GUI route, if applicable:
  - fixture selection change to context tab render.
- Background/concurrency route, if applicable:
  - not applicable.

## Application Integration

- App type: GUI
- User/caller surface: reference review app context tab
- Invocation route: fixture selection change to context tab render
- Wiring owner/module: `src/impression/devtools/reference_review/ui/shell.py`
- Observable result: selected fixture context includes evidence bundle labels
  and role summary
- Integration validation: review-app smoke test or view-model test for a
  bundled fixture
- Incomplete status risk: implemented in isolation if helper output is not wired
  to fixture selection

App-type-specific proof:

- GUI: selected fixture route renders context evidence summary.

## Reuse And Extraction Plan

- Existing code to reuse:
  - selected fixture context state.
- Current reuse readiness:
  - add to existing library/module.
- Extraction/wrapping needed:
  - none.
- Additions to existing library/modules:
  - context evidence summary helper.
- New reusable modules to expose:
  - none.
- One-off code justification, if any:
  - none.

## Required DTOs / Functions / Components

- DTOs/models:
  - context evidence summary display model.
- Functions/methods:
  - `map_context_evidence_summary(...) -> ContextEvidenceSummary`
- UI fields / visible data, if applicable:
  - evidence bundle label.
  - artifact role summary.
- UI elements / controls, if applicable:
  - context tab evidence summary region.
- UI components, if applicable:
  - existing context tab component.

## Performance Contract

- Bounded by selected fixture evidence bundle count.
- Fixture selection must not synchronously open artifact payloads.

## Error And State Behavior

- No bundles: context tab shows no evidence summary section or an empty state
  consistent with existing context behavior.
- Missing optional artifacts show summary without hard failure.
- Invalid bundle diagnostics originate from source registry.

## Test Strategy

- Unit tests:
  - view-model mapping for bundled and unbundled fixtures.
- Service/DB tests:
  - not applicable.
- GUI/controller tests, if applicable:
  - review-app smoke or offscreen view-model test.
- Integrated route tests:
  - fixture selection to context tab render.
- Production-data rule:
  - Tests must not require production data.

## Acceptance Criteria

- Context tab displays bundle labels and role summary for bundled fixtures.
- Unbundled fixtures continue to render cleanly.
- Display wiring follows selected fixture changes.

## Rescore And Split Review

- Manifest score: 19.5.
- IWU count: 1.
- Readiness blockers: none.
- Split decision: Review for split; context summary remains one GUI display route.
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
- Pass 2 - Routing: Confirmed context display is separate from artifacts display.
- Pass 3 - Rescore: Manifest score 19.5; review-for-split band retained.
- Pass 4 - Split Review: Context summary remains one display route.
- Pass 5 - Final: Ready as a final leaf specification after Surface Spec 413.
