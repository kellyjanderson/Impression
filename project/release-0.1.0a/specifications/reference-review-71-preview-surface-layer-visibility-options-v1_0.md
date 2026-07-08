# Reference Review Spec 71: Preview Surface Layer Visibility Options (v1.0)

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one renderer-facing layer-visibility application path for cached preview geometry.

## Overview

Apply layer visibility options during preview surface repaint.

## Backlink

- [Reference Review Spec 69: Preview Display Options State Record](reference-review-69-preview-display-options-state-record-v1_0.md)

## Scope

This specification covers show/hide behavior for object fill, object edges,
triangle wireframe, bounds grid, axis triad, gradient background, and polylines.

This specification does not cover color mode, lighting mode, command routing,
or toolbar layout.

## Implementation Boundary

Owner/module:

- `src/impression/devtools/reference_review/ui/preview_widget.py`

Reuse/extraction decision:

- Extend the existing preview surface paint/projection path.

## Behavior

The implementation must:

- show/hide each layer independently from cached prepared geometry
- allow object edges and triangle wireframe to be visible together
- allow object fill to be hidden while overlays remain visible
- avoid payload rebuilds, renderer recreation, and topology recomputation

## Verification

Test strategy:

- projection/paint tests for each layer visibility option
- topology cache tests proving layer changes do not rerun edge extraction

## Manifest Assessment

Score:

- Functions/methods: 1 x 2 = 2
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 1 x 1 = 1
- Returns/outputs/signals: 1 x 1 = 1
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 2 x 2 = 4
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 7 x 1 = 7
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 0 x 2 = 0
- Total: 24.5

Split decision:

- No split needed. Score is in the split-review band because seven layer
  switches are enumerated, but they are one renderer layer-visibility path.

## Refinement Status

Final leaf.

## Acceptance

This specification is complete when layer visibility options repaint cached preview geometry without renderer churn.
