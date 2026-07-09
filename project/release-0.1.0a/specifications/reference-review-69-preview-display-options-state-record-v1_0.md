# Reference Review Spec 69: Preview Display Options State Record (v1.0)

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one preview display option data record with default values.

## Overview

Define the preview display options record and its default state.

## Backlink

- [Architecture: Reference Review Preview Display Controls Product Definition](../architecture/reference-review-preview-display-controls-product-definition.md)

## Scope

This specification covers the typed state record for color mode, lighting mode,
and independent display toggles.

It does not cover command routing, UI controls, or renderer application.

## Implementation Boundary

Owner/module:

- preview widget or preview pane state module

Reuse/extraction decision:

- Add a small reusable preview display state record consumed by UI and renderer layers.

## Behavior

Default state:

- color mode: inspection color
- lighting mode: face normals
- object fill: on
- object edges: on
- triangle wireframe: off
- bounds grid: on
- axis triad: on
- gradient background: on
- polylines: on

The record must be immutable or updateable through deterministic copy/update helpers.

## Verification

Test strategy:

- default state tests
- equality/copy update tests
- invalid enum value rejection tests where applicable

## Manifest Assessment

Score:

- Functions/methods: 1 x 2 = 2
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 0 x 1 = 0
- Returns/outputs/signals: 1 x 1 = 1
- Existing reusable code reused as-is: 0 x 0.5 = 0
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 9 x 1 = 9
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 0 x 2 = 0
- Total: 20

Split decision:

- No split needed. Cohesive state-record leaf; score reflects enumerated fields.

## Refinement Status

Final leaf.

## Acceptance

This specification is complete when preview display defaults can be created and updated deterministically.
