# Reference Review Spec 72: Preview Surface Color And Lighting Options (v1.0)

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one renderer-facing color and lighting application path for cached preview geometry.

## Overview

Apply color mode and lighting mode options during preview surface repaint.

## Backlink

- [Reference Review Spec 69: Preview Display Options State Record](reference-review-69-preview-display-options-state-record-v1_0.md)

## Scope

This specification covers authored versus inspection color mode and flat,
face-normal, and camera-light lighting modes.

This specification does not cover layer visibility, command routing, or toolbar
layout.

## Implementation Boundary

Owner/module:

- `src/impression/devtools/reference_review/ui/preview_widget.py`

Reuse/extraction decision:

- Extend the existing preview surface paint/projection path.

## Behavior

The implementation must:

- use authored mesh and face colors when authored color mode is selected
- use inspection color when inspection color mode is selected
- fall back to inspection color when authored color data is absent
- support flat, face-normal, and camera-light modes
- avoid topology recomputation when color or lighting mode changes

## Verification

Test strategy:

- authored mesh color tests
- authored face color tests
- inspection-color fallback tests
- lighting-mode output tests
- topology cache tests proving color/lighting changes do not rerun edge extraction

## Manifest Assessment

Score:

- Functions/methods: 2 x 2 = 4
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
- UI fields/elements: 2 x 1 = 2
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 0 x 2 = 0
- Total: 21.5

Split decision:

- No split needed. Cohesive color/lighting application leaf.

## Refinement Status

Final leaf.

## Acceptance

This specification is complete when color and lighting modes repaint cached preview geometry without renderer churn.
