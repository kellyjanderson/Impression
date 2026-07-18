# Reference Review Spec 75c2: Preview Command Application Efficiency (v1.0)

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one renderer-application efficiency change that applies queued payload/display commands without duplicate renderer work.

## Overview

Apply queued payload and display commands efficiently: payload commands load
datasets once, display commands reuse decoded datasets, and one display state
produces one renderer application.

## Backlink

- [Reference Review Spec 75c: Preview Widget Command Drain And Renderer Mutation Boundary](reference-review-75c-preview-widget-command-drain-and-renderer-mutation-boundary-v1_0.md)

## Source Manifest

- This leaf is split from ad hoc remediation spec 75c.
- Manifest score: 23

## Scope

This specification covers:

- payload command application
- display-options command application without payload JSON reload
- failure/lifecycle command application
- camera reset command application
- avoiding duplicate render/apply passes for one display update

## Responsibilities

- Functions/methods:
  - apply payload command
  - apply display command
  - apply lifecycle/failure command
- Data structures/models:
  - none
- Dependencies/services:
  - `PreviewRendererLifecycleWidget`
  - `QtPreviewSurface`
- Returns/outputs/signals:
  - command application result
- UI surfaces/components:
  - preview pane
- UI fields/elements:
  - preview viewport
  - preview status text
- Reusable code plan:
  - Existing code reused as-is:
    - shared Qt preview surface
    - software preview fallback
  - Additions to existing reusable library/module:
    - preview widget command application helpers
  - New reusable library/module to create:
    - none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - renderer mutation happens only during widget-owned drain
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - sanitized diagnostics only
- Performance-sensitive behavior:
  - display commands reuse decoded datasets
  - one display command yields one renderer scene update
- Cross-screen reusable behavior:
  - none

## Implementation Boundary

Owner/module:

- `src/impression/devtools/reference_review/ui/preview_widget.py`
- `src/impression/preview_qt.py` only if a small non-reset display update hook
  is required

Routes:

- widget drain calls command application helpers
- renderer surface receives final coherent state

Reuse/extraction decision:

- keep one shared renderer path and do not add rendering logic to shell

## Data And Defaults

Chosen defaults / parameters:

- display command does not reset camera
- payload command aligns camera only for new current payload

Data ownership:

- widget owns decoded current datasets
- renderer owns render surface lifetime

Open questions and resolved assumptions:

- software fallback may continue to project in `paintEvent`

Implementation prerequisites:

- Spec 75c1

## Behavior

The implementation must:

- apply display options without re-reading payload files
- avoid duplicate render/apply passes for one display toggle
- preserve current ready/failure visible state semantics

## Verification

Test strategy:

- widget tests with fake renderer proving one scene update after coalesced
  toggles
- widget tests proving payload JSON is not reread for display-only commands
- lifecycle tests proving renderer is not recreated

Additional verification requirements:

- run Reference Review UI shell tests and preview controller tests

## Manifest Assessment

Score:

- Functions/methods: 3 x 2 = 6
- Data structures/models: 0 x 1 = 0
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 1 x 1 = 1
- Existing reusable code reused as-is: 2 x 0.5 = 1
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 2 x 2 = 4
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 2 x 1 = 2
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 23

Split decision:

- No split needed. Cohesion reason: this leaf only optimizes queued command
  application inside the preview widget drain.

## Refinement Status

Final leaf.

## Child Specifications

None.

## Acceptance

This specification is complete when queued payload/display commands mutate the
renderer efficiently without duplicate render work or payload JSON reload.
