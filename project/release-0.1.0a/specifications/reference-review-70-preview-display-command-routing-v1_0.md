# Reference Review Spec 70: Preview Display Command Routing (v1.0)

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one command routing/reducer boundary for preview display option changes.

## Overview

Route preview display-control commands into preview display option state updates.

## Backlink

- [Reference Review Spec 69: Preview Display Options State Record](reference-review-69-preview-display-options-state-record-v1_0.md)

## Scope

This specification covers command ids, reducer behavior, exclusive group
updates, independent toggle updates, disabled-state rejection, and diagnostics.

It does not cover rendering the control bar or applying options to the renderer.

## Implementation Boundary

Owner/module:

- preview pane state/routing module

Reuse/extraction decision:

- Add routing helpers alongside existing preview toolbar command routing.

## Behavior

The implementation must:

- update color mode exclusively
- update lighting mode exclusively
- toggle independent display booleans independently
- reject unsupported commands deterministically
- reject display changes while preview state is not ready
- never rebuild payloads or recreate renderers

## Verification

Test strategy:

- color group routing tests
- lighting group routing tests
- independent toggle tests
- disabled-state and unsupported-command diagnostics

## Manifest Assessment

Score:

- Functions/methods: 2 x 2 = 4
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 2 x 1 = 2
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 0 x 2 = 0
- Total: 23.5

Split decision:

- No split needed. Cohesive command reducer leaf; score is in split-review band due to two exclusive groups plus toggles, all updating one state record.

## Refinement Status

Final leaf.

## Acceptance

This specification is complete when display-control commands update display options safely and deterministically.
