# Reference Review Spec 78: Preview Lifecycle And Non-Renderable Fixture Handling (v1.0)

Status: Split parent - superseded by child specs 78a and 78b

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own
verification surface.
Draft count: 1 IWU.
Basis: one preview lifecycle and non-renderable fixture handling leaf. This
count is a draft creation estimate and must be verified by `review specs`.

## Overview

Stabilize the Reference Review preview route so renderable fixtures render,
diagnostic fixtures show contextual non-renderable state, and stale or failed
preview work cannot destroy the live renderer or clear a newer good preview.

## Backlink

- [ACD: Reference Review Hybrid Stabilization](../architecture/acd-reference-review-hybrid-stabilization.md)
- [Reference Review Preview Payload Boundary Architecture](../architecture/reference-review-preview-payload-boundary-architecture.md)
- [Reference Review Preview Engine Sharing Architecture](../architecture/reference-review-preview-engine-sharing-architecture.md)

## Source Manifest

- Source candidate: `Preview Lifecycle And Non-Renderable Fixture Handling`
- Source artifact: `project/release-0.1.0a/architecture/acd-reference-review-hybrid-stabilization.md`

## Scope

This specification covers:

- one renderer lifetime per preview widget
- off-thread payload work
- UI/render-thread renderer mutation
- renderable artifact preview routing
- diagnostic/non-renderable fixture state routing
- last-good preview preservation
- display-control command routing to current preview surface

## Responsibilities

- Functions/methods:
  - selected fixture to preview request route
  - preview payload completion handler
  - non-renderable fixture state handler
  - display-control command handler
  - preview stale/failure guard
- Data structures/models:
  - existing preview payload result records
  - preview state fields for current, stale, failure, or non-renderable state
- Dependencies/services:
  - preview payload builder/controller
  - preview widget
  - shared Impression preview semantics
- Returns/outputs/signals:
  - render command or payload application request
  - non-renderable context state
  - preview diagnostic
- UI surfaces/components:
  - preview pane
  - display-control button row
- UI fields/elements:
  - preview visible state
  - display-control toggles
  - contextual diagnostic text
- Reusable code plan:
  - Existing code reused as-is:
    - current Reference Review preview widget and payload builder
    - Impression preview controller/surface semantics already in use
  - Additions to existing reusable library/module:
    - optional kit preview display option records if Spec 76 proves safe
  - New reusable library/module to create:
    - none expected
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - payload work is off-thread
  - renderer mutation is UI/render-thread only
  - stale preview completions are rejected before renderer mutation
- Destructive/write behavior:
  - no artifact writes or promotions in this leaf
- Security/privacy-sensitive behavior:
  - preview diagnostics are sanitized before display
- Performance-sensitive behavior:
  - renderer is not recreated for every fixture selection or stale/failure path
- Cross-screen reusable behavior:
  - preview display controls remain shared across renderable fixture previews

## Implementation Boundary

Owner/module:

- `src/impression/devtools/reference_review/ui/preview_widget.py`
- `src/impression/devtools/reference_review/ui/shell.py`
- `src/impression/devtools/reference_review/preview_payload_builder.py`

Routes:

- fixture selection to preview payload request
- preview payload completion to preview widget
- display-control command to preview widget

Reuse/extraction decision:

- keep fixture-specific payload decisions in Reference Review
- keep renderer semantics aligned with Impression CLI preview
- do not introduce Bench-specific file/document preview routing

## Data And Defaults

Chosen defaults / parameters:

- a fixture with a renderable artifact enters the render path
- a fixture without a renderable artifact enters the contextual
  non-renderable path
- stale or failed work preserves last good render when one exists

Data ownership:

- payload builder owns render input preparation
- preview widget owns renderer lifetime and scene mutation
- shell owns selected fixture and visible preview state

Open questions and resolved assumptions:

- resolved: diagnostic fixtures are valid review rows and must not crash the
  renderer
- resolved: stale failures cannot clear newer good preview state

Implementation prerequisites:

- Spec 76 import decisions
- Spec 77 UI-thread handoff and stale-result route

## Behavior

The implementation must:

- preserve one live renderer per preview widget lifetime;
- avoid renderer creation/destruction as a fixture-selection side effect;
- perform artifact/source payload preparation off the UI thread;
- apply preview payloads only through the UI/render thread;
- route non-renderable fixtures to contextual state without calling renderer
  scene replacement;
- preserve last-good render when stale or failed completions arrive;
- keep display-control commands functional for the current preview surface.

## Verification

Test strategy:

- focused preview payload tests for renderable artifact records;
- focused shell/preview routing tests for non-renderable diagnostics;
- stale success/failure rejection test where available;
- display-control routing test confirming commands reach the preview surface.

Additional verification requirements:

- manual selection smoke across renderable STL or `.impress` fixtures and
  diagnostic/non-renderable fixtures
- run `git diff --check`

## Readiness Fields

App type:

- GUI route inside mixed console-launched app

User/caller surface:

- preview pane in the Reference Review app

Invocation route:

- fixture list selection and preview display-control buttons

Wiring owner/module:

- shell, preview widget, and preview payload builder

Observable result:

- renderable fixtures preview; non-renderable fixtures show context; preview
  remains responsive across selection and control changes

Integration validation:

- preview payload tests, shell routing tests, display-control tests, manual GUI
  smoke

Readiness blockers:

- depends on Spec 77 for non-blocking handoff

## Review Score

- Fresh review score: 47.5
- IWU recount: 2 IWU
- Split decision: split required. Renderable preview lifecycle and
  non-renderable/stale failure state are independently failing preview routes.

## Refinement Status

Split parent. Do not implement directly.

## Child Specifications

- [Reference Review Spec 78a: Renderable Preview Lifecycle](reference-review-78a-renderable-preview-lifecycle-v1_0.md)
- [Reference Review Spec 78b: Non-Renderable Preview State And Last-Good Guard](reference-review-78b-non-renderable-preview-state-and-last-good-guard-v1_0.md)

## Split Coverage

| Parent responsibility | Child owner | Status |
| --- | --- | --- |
| one renderer lifetime per preview widget | Spec 78a | Covered |
| off-thread renderable payload work | Spec 78a | Covered |
| UI/render-thread renderer mutation | Spec 78a | Covered |
| renderable artifact preview routing | Spec 78a | Covered |
| diagnostic/non-renderable fixture state routing | Spec 78b | Covered |
| last-good preview preservation | Spec 78b | Covered |
| display-control command routing | Spec 78a | Covered |

## Acceptance

This specification is complete when preview renderer lifetime is stable,
renderable and non-renderable fixture paths are separated, and stale/failure
preview work cannot corrupt the current live preview.
