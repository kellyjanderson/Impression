# Reference Review Spec 75d: Preview Render Queue Regression Tests (v1.0)

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one regression-verification leaf for the preview render queue remediation.

## Overview

Add regression tests and manual smoke evidence for the preview render command
queue remediation.

## Backlink

- [Reference Review Spec 75: Preview Render Command Queue](reference-review-75-preview-render-command-queue-v1_0.md)
- [Reference Review Test Spec 75: Preview Render Command Queue](../test-specifications/reference-review-75-preview-render-command-queue-v1_0.md)

## Source Manifest

- This leaf is derived from ad hoc remediation spec 75.
- Manifest score: 21.5

## Scope

This specification covers:

- automated tests for Specs 75a-75c
- manual smoke checklist updates for live `.impress` fixture review

## Responsibilities

- Functions/methods:
  - test queue coalescing
  - test stale completion rejection
  - test display-toggle coalescing
  - test renderer lifecycle preservation
- Data structures/models:
  - fake renderer
  - fake payload completion
- Dependencies/services:
  - pytest
  - PySide6 offscreen tests
- Returns/outputs/signals:
  - test diagnostics
- UI surfaces/components:
  - preview pane
- UI fields/elements:
  - preview viewport
  - display-control row
- Reusable code plan:
  - Existing code reused as-is:
    - existing Reference Review shell test helpers
  - Additions to existing reusable library/module:
    - none
  - New reusable library/module to create:
    - none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - tests exercise stale/rapid adjacent paths
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - sanitized failure text remains asserted
- Performance-sensitive behavior:
  - tests assert bounded renderer update count
- Cross-screen reusable behavior:
  - none

## Implementation Boundary

Owner/module:

- `tests/test_reference_review_ui_shell.py`
- `tests/test_reference_review_async_core.py`
- optional new `tests/test_reference_review_preview_render_queue.py`

Routes:

- tests should use fake renderer surfaces where native VTK would be unstable
  offscreen

Reuse/extraction decision:

- prefer small fake renderers over launching native VTK in automated tests

## Data And Defaults

Chosen defaults / parameters:

- automated tests run offscreen
- manual smoke uses dirty `.impress` fixture file

Data ownership:

- tests own temporary payload files and fake futures

Open questions and resolved assumptions:

- manual live interaction evidence remains required because offscreen tests
  cannot prove native VTK responsiveness

Implementation prerequisites:

- Specs 75a-75c

## Behavior

The implementation must:

- add automated coverage for the queue and UI integration
- document manual smoke observations when implementation completes

## Verification

Test strategy:

- run the paired test spec checklist

Additional verification requirements:

- run required validation commands from the paired test spec

## Manifest Assessment

Score:

- Functions/methods: 3 x 2 = 6
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 1 x 1 = 1
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 0 x 1 = 0
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 1 x 1 = 1
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 21.5

Split decision:

- No split needed. Cohesion reason: this is the regression leaf for the
  preview render command queue remediation and contains no production
  implementation route.

## Refinement Status

Final leaf.

## Child Specifications

None.

## Acceptance

This specification is complete when:

- automated regression tests cover the queue remediation
- manual smoke evidence is recorded after implementation
